"""Paired evaluation of official RGB and RewardGraph checkpoints.

Both paths are reconstructed from scratch with the same environment/model seed.
Checkpoint loading happens before the paired evaluation seed is installed, so
the reported RNG contract covers only reset, inference, and environment steps.
"""

import argparse
import gc
import json
import os
import random
from pathlib import Path
from time import perf_counter

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('LAZY_LEGACY_OP', '0')
os.environ.setdefault('TORCHDYNAMO_INLINE_INBUILT_NN_MODULES', '1')

import numpy as np
import torch
from omegaconf import OmegaConf

from common import MODEL_SIZE
from common.parser import cfg_to_dataclass
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2


CONFIG_PATH = Path(__file__).with_name('config.yaml')
EPISODE_STEPS = 500
BOOTSTRAP_SEED = 20260823
BOOTSTRAP_SAMPLES = 10_000


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--rgb-checkpoint', type=Path, required=True)
	parser.add_argument('--reward-graph-checkpoint', type=Path, required=True)
	parser.add_argument('--support', type=Path, required=True)
	parser.add_argument(
		'--seed', type=int, default=1,
		help='Environment/model construction seed; match both checkpoints.',
	)
	parser.add_argument('--episodes', type=int, default=20)
	parser.add_argument(
		'--eval-seed', type=int, required=True,
		help='Planner RNG seed, reset identically after each checkpoint load.',
	)
	args = parser.parse_args()
	if args.episodes < 1:
		parser.error('--episodes must be positive.')
	for name in ('rgb_checkpoint', 'reward_graph_checkpoint', 'support'):
		path = getattr(args, name)
		if not path.is_file():
			parser.error(f'File does not exist for --{name.replace("_", "-")}: {path}')
	return args


def make_config(args, reward_graph):
	"""Build a fresh config; RewardGraph expands its latent metadata in place."""
	cfg = OmegaConf.load(CONFIG_PATH)
	cfg.task = 'reacher-easy'
	cfg.obs = 'rgb'
	cfg.seed = args.seed
	cfg.model_size = 5
	for key, value in MODEL_SIZE[5].items():
		cfg[key] = value
	cfg.multitask = False
	cfg.tasks = [cfg.task]
	cfg.task_dim = 0
	cfg.task_title = 'Reacher Easy'
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
	# This diagnostic intentionally has no eager option. Cross-architecture RNG
	# pairing relies on random operators falling back to the eager CUDA stream.
	cfg.compile = True
	cfg.compile_fallback_random = True
	cfg.flat_anchor = reward_graph
	cfg.flat_anchor_mode = 'reward_graph'
	cfg.flat_anchor_support_path = str(args.support)
	cfg.flat_anchor_event_enabled = False
	cfg.enable_wandb = False
	cfg.save_video = False
	cfg.save_csv = False
	cfg.save_agent = False
	return cfg


def capture_rng():
	return {
		'python': random.getstate(),
		'numpy': np.random.get_state(),
		'cpu': torch.get_rng_state().clone(),
		'cuda': [state.clone() for state in torch.cuda.get_rng_state_all()],
	}


def numpy_rng_equal(left, right):
	return (
		left[0] == right[0]
		and np.array_equal(left[1], right[1])
		and left[2:] == right[2:]
	)


def rng_equal(left, right):
	return (
		left['python'] == right['python']
		and numpy_rng_equal(left['numpy'], right['numpy'])
		and torch.equal(left['cpu'], right['cpu'])
		and len(left['cuda']) == len(right['cuda'])
		and all(torch.equal(a, b) for a, b in zip(left['cuda'], right['cuda']))
	)


def initial_rgb(obs, reward_graph):
	value = obs['rgb'] if reward_graph else obs
	return value.detach().cpu().clone()


def statistics(values):
	values = np.asarray(values, dtype=np.float64)
	return {
		'mean': float(values.mean()),
		'median': float(np.median(values)),
		'std': float(values.std()),
	}


def paired_statistics(delta):
	"""Summarize paired deltas without advancing NumPy's global RNG."""
	delta = np.asarray(delta, dtype=np.float64)
	local_rng = np.random.default_rng(BOOTSTRAP_SEED)
	indices = local_rng.integers(
		0, delta.size, size=(BOOTSTRAP_SAMPLES, delta.size)
	)
	bootstrap_means = delta[indices].mean(axis=1)
	ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
	return {
		**statistics(delta),
		'win_rate': float(np.mean(delta > 0.)),
		'tie_rate': float(np.mean(delta == 0.)),
		'loss_rate': float(np.mean(delta < 0.)),
		'mean_bootstrap_95_ci': [float(ci_low), float(ci_high)],
		'bootstrap_seed': BOOTSTRAP_SEED,
		'bootstrap_samples': BOOTSTRAP_SAMPLES,
	}


def evaluate_checkpoint(args, name, checkpoint, reward_graph):
	# Environment, config, and agent must all be fresh. Construction and strict
	# checkpoint loading deliberately happen outside the paired evaluation RNG.
	cfg = make_config(args, reward_graph)
	set_seed(args.seed)
	env = make_env(cfg)
	if cfg.episode_length != EPISODE_STEPS:
		raise RuntimeError(
			f'{name} episode length is {cfg.episode_length}, expected {EPISODE_STEPS}.'
		)
	cfg = cfg_to_dataclass(cfg)
	agent = TDMPC2(cfg)
	agent.load(checkpoint)
	agent.eval()

	set_seed(args.eval_seed)
	start_rng = capture_rng()
	start_time = perf_counter()
	rewards, initial_rgbs = [], []
	for episode in range(args.episodes):
		obs = env.reset()
		initial_rgbs.append(initial_rgb(obs, reward_graph))
		agent._prev_mean.zero_()
		episode_reward = 0.
		done = False
		for t in range(EPISODE_STEPS):
			torch.compiler.cudagraph_mark_step_begin()
			action = agent.act(obs, t0=(t == 0), eval_mode=True)
			obs, reward, done, _ = env.step(action)
			episode_reward += float(reward)
			if done and t + 1 != EPISODE_STEPS:
				raise RuntimeError(
					f'{name} episode {episode} terminated at step {t + 1}; '
					f'exactly {EPISODE_STEPS} steps are required.'
				)
		if not done:
			raise RuntimeError(
				f'{name} episode {episode} did not terminate at step {EPISODE_STEPS}.'
			)
		rewards.append(episode_reward)
	elapsed = perf_counter() - start_time
	end_rng = capture_rng()
	result = {
		'name': name,
		'checkpoint': str(checkpoint),
		'episode_rewards': rewards,
		**statistics(rewards),
		'episodes': args.episodes,
		'steps_per_episode': EPISODE_STEPS,
		'elapsed_seconds': elapsed,
		'elapsed_note': (
			'End-to-end diagnostic wall time includes first compile and DINO warmup; '
			'it is not a steady-state speed comparison.'
		),
	}
	print('REWARD_GRAPH_CHECKPOINT_GROUP', json.dumps(result, sort_keys=True))

	# The repository's terminal DMControl wrapper has no close method. Do not call
	# env.close(); normal deletion is sufficient for this bounded diagnostic.
	del agent, env
	gc.collect()
	torch.cuda.empty_cache()
	return result, initial_rgbs, start_rng, end_rng


def main():
	args = parse_args()
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required.')
	torch.backends.cudnn.benchmark = True
	torch.set_float32_matmul_precision('high')

	print('\nEvaluating official RGB checkpoint...')
	rgb, rgb_initial, rgb_start_rng, rgb_end_rng = evaluate_checkpoint(
		args, 'official_rgb', args.rgb_checkpoint, reward_graph=False
	)
	print('\nEvaluating RewardGraph checkpoint...')
	graph, graph_initial, graph_start_rng, graph_end_rng = evaluate_checkpoint(
		args, 'reward_graph', args.reward_graph_checkpoint, reward_graph=True
	)

	initial_rgb_equal = [
		torch.equal(rgb_obs, graph_obs)
		for rgb_obs, graph_obs in zip(rgb_initial, graph_initial)
	]
	if len(initial_rgb_equal) != args.episodes:
		raise RuntimeError('Paired evaluation returned the wrong number of resets.')
	start_rng_equal = rng_equal(rgb_start_rng, graph_start_rng)
	end_rng_equal = rng_equal(rgb_end_rng, graph_end_rng)
	if not all(initial_rgb_equal) or not start_rng_equal or not end_rng_equal:
		raise RuntimeError(
			'Paired-evaluation contract failed: '
			f'initial_rgb_equal={initial_rgb_equal}, '
			f'start_rng_equal={start_rng_equal}, end_rng_equal={end_rng_equal}.'
		)

	delta = np.asarray(graph['episode_rewards'], dtype=np.float64) - np.asarray(
		rgb['episode_rewards'], dtype=np.float64
	)
	paired = {
		'direction': 'reward_graph_minus_official_rgb',
		'episode_deltas': delta.tolist(),
		**paired_statistics(delta),
	}
	summary = {
		'seed': args.seed,
		'eval_seed': args.eval_seed,
		'compile': True,
		'compile_fallback_random': True,
		'initial_rgb_equal_by_episode': initial_rgb_equal,
		'start_rng_equal': start_rng_equal,
		'end_rng_equal': end_rng_equal,
		'official_rgb': rgb,
		'reward_graph': graph,
		'paired_delta': paired,
	}
	print('\nREWARD_GRAPH_CHECKPOINT_PAIR_OK', json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
	main()
