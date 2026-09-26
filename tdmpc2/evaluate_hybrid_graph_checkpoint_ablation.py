"""Paired, no-training ablation of HybridGraph decision corrections.

The script loads the same trained HybridGraph checkpoint for every mode and
zeros selected correction output layers *before* the first compiled plan.  It
therefore diagnoses whether the learned reward/Q/policy corrections help at
inference without retraining or changing the planner's graph topology.

The first diagnostic should compare only ``full`` and ``all_off``.  If
``all_off`` recovers, a later invocation can add ``no_pi`` and ``reward_only``
to identify the harmful correction more precisely.
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
MODES = ('full', 'no_pi', 'reward_only', 'all_off')


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--support', type=Path, required=True)
	parser.add_argument('--episodes', type=int, default=5)
	parser.add_argument('--seed', type=int, default=2,
			help='DMControl/model construction seed; match the trained checkpoint.')
	parser.add_argument('--eval-seed', type=int, default=20260823,
			help='Planner RNG seed, reset identically before every mode.')
	parser.add_argument('--modes', nargs='+', choices=MODES,
			default=['full', 'all_off'])
	parser.add_argument('--compile', action=argparse.BooleanOptionalAction,
			default=True)
	args = parser.parse_args()
	if args.episodes < 1:
		parser.error('--episodes must be positive.')
	if args.modes[0] != 'full' or len(set(args.modes)) != len(args.modes):
		parser.error('--modes must begin with full and cannot contain duplicates.')
	if not args.checkpoint.is_file():
		parser.error(f'Checkpoint does not exist: {args.checkpoint}')
	if not args.support.is_file():
		parser.error(f'Support annotations do not exist: {args.support}')
	return args


def make_config(args):
	"""Build a fresh config because TDMPC2 mutates HybridGraph latent metadata."""
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
	cfg.compile = args.compile
	# Cross-graph paired evaluation requires eager CUDA random operators.
	cfg.compile_fallback_random = args.compile
	cfg.flat_anchor = True
	cfg.flat_anchor_mode = 'hybrid_graph'
	cfg.flat_anchor_support_path = str(args.support)
	cfg.flat_anchor_event_enabled = False
	cfg.enable_wandb = False
	cfg.save_video = False
	cfg.save_csv = False
	cfg.save_agent = False
	return cfg


def clone_observation(obs):
	if hasattr(obs, 'items'):
		return {
			str(key): value.detach().cpu().clone()
			for key, value in obs.items()
		}
	return {'obs': obs.detach().cpu().clone()}


def observations_equal(left, right):
	return left.keys() == right.keys() and all(
		torch.equal(left[key], right[key]) for key in left
	)


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


def zero_output(module):
	module.output.weight.zero_()
	module.output.bias.zero_()


def zero_q_outputs(model):
	# Online and detach share storage by design; target is an independent clone.
	# Explicitly zero all three to keep the checkpoint lifecycle contract obvious.
	for params in (
		model._hybrid_q.params,
		model._detach_hybrid_q_params,
		model._target_hybrid_q_params,
	):
		params['output', 'weight'].zero_()
		params['output', 'bias'].zero_()


def trained_correction_maxabs(model):
	"""Record that the loaded checkpoint contains learned correction outputs."""
	return {
		'reward': float(model._hybrid_reward.output.weight.abs().max().item()),
		'q': float(model._hybrid_q.params['output', 'weight'].abs().max().item()),
		'pi': float(model._hybrid_pi.output.weight.abs().max().item()),
	}


@torch.no_grad()
def apply_mode(agent, mode):
	if mode in {'no_pi', 'reward_only', 'all_off'}:
		zero_output(agent.model._hybrid_pi)
	if mode in {'reward_only', 'all_off'}:
		zero_q_outputs(agent.model)
	if mode == 'all_off':
		zero_output(agent.model._hybrid_reward)


@torch.no_grad()
def disabled_output_contract(agent, mode):
	"""Verify that every correction disabled by ``mode`` is exactly zero."""
	model = agent.model
	batch = 2
	z = torch.linspace(
		-1., 1., batch * agent.cfg.latent_dim, device=agent.device
	).reshape(batch, agent.cfg.latent_dim)
	a = torch.linspace(
		-0.5, 0.5, batch * agent.cfg.action_dim, device=agent.device
	).reshape(batch, agent.cfg.action_dim)
	checks = {}
	if mode in {'no_pi', 'reward_only', 'all_off'}:
		checks['pi'] = float(model._hybrid_pi(z).abs().max().item())
	if mode in {'reward_only', 'all_off'}:
		q_input = torch.cat([z, a], dim=-1)
		checks.update({
			'q_online': float(model._hybrid_q(q_input).abs().max().item()),
			'q_detach': float(model._detach_hybrid_q(q_input).abs().max().item()),
			'q_target': float(model._target_hybrid_q(q_input).abs().max().item()),
		})
	if mode == 'all_off':
		checks['reward'] = float(
			model._hybrid_reward(torch.cat([z, a], dim=-1)).abs().max().item()
		)
	if any(value != 0. for value in checks.values()):
		raise RuntimeError(f'{mode} failed zero-output contract: {checks}')
	return checks


def evaluate_mode(args, mode):
	# A fresh cfg/env/agent is required for a paired DMControl reset sequence and
	# because HybridGraph construction expands cfg latent metadata in place.
	cfg = make_config(args)
	set_seed(args.seed)
	env = make_env(cfg)
	cfg = cfg_to_dataclass(cfg)
	agent = TDMPC2(cfg)
	agent.load(args.checkpoint)
	loaded_correction_maxabs = trained_correction_maxabs(agent.model)
	if mode == 'full' and not all(
		value > 0. for value in loaded_correction_maxabs.values()
	):
		raise RuntimeError(
			'The full checkpoint has an untrained correction output: '
			f'{loaded_correction_maxabs}'
		)
	apply_mode(agent, mode)
	zero_contract = disabled_output_contract(agent, mode)
	agent.eval()

	# Construction and checkpoint loading are outside the paired evaluation RNG.
	set_seed(args.eval_seed)
	start_rng = capture_rng()
	start = perf_counter()
	rewards, lengths, initial_observations = [], [], []
	for episode in range(args.episodes):
		obs = env.reset()
		initial_observations.append(clone_observation(obs))
		done, episode_reward, t = False, 0., 0
		agent._prev_mean.zero_()
		while not done:
			torch.compiler.cudagraph_mark_step_begin()
			action = agent.act(obs, t0=(t == 0), eval_mode=True)
			obs, reward, done, _ = env.step(action)
			episode_reward += float(reward)
			t += 1
		rewards.append(episode_reward)
		lengths.append(t)
	elapsed = perf_counter() - start
	end_rng = capture_rng()
	result = {
		'mode': mode,
		'episodes': args.episodes,
		'episode_rewards': rewards,
		'mean_reward': float(np.mean(rewards)),
		'std_reward': float(np.std(rewards)),
		'episode_lengths': lengths,
		'elapsed_seconds': elapsed,
		'loaded_correction_maxabs': loaded_correction_maxabs,
		'zero_contract': zero_contract,
	}
	print('HYBRID_GRAPH_ABLATION_MODE', json.dumps(result, sort_keys=True))

	# Gymnasium's wrapper chain exposes ``close`` even though the repository's
	# terminal DMControlWrapper does not implement it. Explicit traversal would
	# therefore raise after a completed evaluation; normal deletion is sufficient
	# for this short-lived diagnostic process.
	del agent, env
	gc.collect()
	torch.cuda.empty_cache()
	return result, initial_observations, start_rng, end_rng


def main():
	args = parse_args()
	torch.backends.cudnn.benchmark = True
	torch.set_float32_matmul_precision('high')
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required.')

	results = []
	reference_observations = reference_start_rng = reference_end_rng = None
	for mode in args.modes:
		print(f'\nEvaluating checkpoint mode: {mode}')
		result, observations, start_rng, end_rng = evaluate_mode(args, mode)
		if reference_observations is None:
			reference_observations = observations
			reference_start_rng = start_rng
			reference_end_rng = end_rng
			result['initial_observations_equal'] = True
			result['start_rng_equal'] = True
			result['end_rng_equal'] = True
		else:
			obs_equal = len(observations) == len(reference_observations) and all(
				observations_equal(left, right)
				for left, right in zip(reference_observations, observations)
			)
			result['initial_observations_equal'] = obs_equal
			result['start_rng_equal'] = rng_equal(reference_start_rng, start_rng)
			result['end_rng_equal'] = rng_equal(reference_end_rng, end_rng)
			if not all((
				result['initial_observations_equal'],
				result['start_rng_equal'],
				result['end_rng_equal'],
			)):
				raise RuntimeError(
					f'Paired-evaluation contract failed for {mode}: '
					f"obs={result['initial_observations_equal']}, "
					f"start_rng={result['start_rng_equal']}, "
					f"end_rng={result['end_rng_equal']}"
				)
		mode_rewards = np.asarray(result['episode_rewards'])
		paired_delta = (
			np.zeros_like(mode_rewards)
			if not results
			else mode_rewards - np.asarray(results[0]['episode_rewards'])
		)
		result['paired_delta_vs_full'] = paired_delta.tolist()
		result['mean_delta_vs_full'] = float(paired_delta.mean())
		results.append(result)

	summary = {
		'checkpoint': str(args.checkpoint),
		'seed': args.seed,
		'eval_seed': args.eval_seed,
		'compile': args.compile,
		'compile_fallback_random': args.compile,
		'results': results,
	}
	print('\nHYBRID_GRAPH_CHECKPOINT_ABLATION_OK', json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
	main()
