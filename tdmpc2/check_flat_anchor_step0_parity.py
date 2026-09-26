"""End-to-end step-0 parity audit for official RGB and FlatAnchor TD-MPC2.

No training is performed. The two paths are built independently with the same
seed and compared at the RGB observation, official model state, seed-action
sequence, encoded latent, and first MPPI evaluation action.
"""

import argparse
import copy
import gc
import os
import random
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch
from omegaconf import OmegaConf

from common import MODEL_SIZE
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2


CONFIG_PATH = Path(__file__).with_name('config.yaml')


def make_config(args, flat_anchor):
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
	cfg.flat_anchor = flat_anchor
	# This audit is specifically for the completed identity-residual ablation.
	# OC mode intentionally changes the encoded state and first planned action.
	cfg.flat_anchor_mode = 'residual'
	cfg.flat_anchor_support_path = str(args.support) if flat_anchor else None
	cfg.enable_wandb = False
	cfg.save_video = False
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


def clone_state_value(value):
	return value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)


def state_value_equal(left, right):
	if torch.is_tensor(left) or torch.is_tensor(right):
		return torch.is_tensor(left) and torch.is_tensor(right) \
			and torch.equal(left, right.detach().cpu())
	if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
		return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) \
			and np.array_equal(left, right)
	return type(left) is type(right) and left == right


def seed_torch(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def rgb_from_obs(obs, flat_anchor):
	return obs['rgb'] if flat_anchor else obs


def batched_device_obs(obs, flat_anchor):
	if flat_anchor:
		return obs.to('cuda:0').unsqueeze(0)
	return obs.to('cuda:0').unsqueeze(0)


def run_path(args, flat_anchor):
	cfg = make_config(args, flat_anchor)
	set_seed(args.seed)
	env = make_env(cfg)
	agent = TDMPC2(cfg)
	obs = env.reset()
	rgb = rgb_from_obs(obs, flat_anchor).detach().cpu().clone()
	post_reset_rng = capture_rng()

	random_actions = torch.stack([env.rand_act() for _ in range(16)]).cpu()

	seed_torch(args.encode_seed)
	latent = agent.model.encode(batched_device_obs(obs, flat_anchor), task=None)
	encode_rng = capture_rng()

	seed_torch(args.plan_seed)
	agent._prev_mean.zero_()
	action = agent.act(obs, t0=True, eval_mode=True).detach().cpu()
	plan_rng = capture_rng()

	state = {
		key: clone_state_value(value)
		for key, value in agent.model.state_dict().items()
	}
	return {
		'rgb': rgb,
		'random_actions': random_actions,
		'latent': latent.detach().cpu(),
		'action': action,
		'post_reset_rng': post_reset_rng,
		'encode_rng': encode_rng,
		'plan_rng': plan_rng,
		'state': state,
	}, env, agent


def max_error(left, right):
	return float((left.float() - right.float()).abs().max())


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--support', type=Path, required=True)
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--encode-seed', type=int, default=161803)
	parser.add_argument('--plan-seed', type=int, default=271828)
	parser.add_argument('--compile', action='store_true')
	args = parser.parse_args()
	if not args.support.is_file():
		raise FileNotFoundError(args.support)

	# This eager audit deliberately removes cuDNN autotuner variability. The
	# optional --compile run can be used only after eager parity passes.
	torch.backends.cudnn.benchmark = False
	torch.set_float32_matmul_precision('high')

	print('Building official RGB path...')
	official, official_env, official_agent = run_path(args, False)
	del official_env, official_agent
	gc.collect()
	torch.cuda.empty_cache()

	print('Building FlatAnchor path...')
	anchored, anchored_env, anchored_agent = run_path(args, True)
	del anchored_env, anchored_agent
	gc.collect()
	torch.cuda.empty_cache()

	missing = [key for key in official['state'] if key not in anchored['state']]
	mismatched = [
		key for key, value in official['state'].items()
		if key in anchored['state'] and not state_value_equal(value, anchored['state'][key])
	]
	extra = sorted(set(anchored['state']) - set(official['state']))
	unexpected_extra = [
		key for key in extra
		if not (
			key.startswith('_encoder.anchor.')
			or key.startswith('_encoder.fusion.')
		)
	]

	result = {
		'compile': args.compile,
		'rgb_equal': torch.equal(official['rgb'], anchored['rgb']),
		'rgb_max_error': max_error(official['rgb'], anchored['rgb']),
		'shared_state_missing': len(missing),
		'shared_state_mismatched': len(mismatched),
		'unexpected_extra_state': len(unexpected_extra),
		'random_actions_equal': torch.equal(
			official['random_actions'], anchored['random_actions']
		),
		'random_action_max_error': max_error(
			official['random_actions'], anchored['random_actions']
		),
		'latent_max_error': max_error(official['latent'], anchored['latent']),
		'first_action_max_error': max_error(official['action'], anchored['action']),
		'post_reset_rng_equal': rng_equal(
			official['post_reset_rng'], anchored['post_reset_rng']
		),
		'encode_rng_equal': rng_equal(official['encode_rng'], anchored['encode_rng']),
		'plan_rng_equal': rng_equal(official['plan_rng'], anchored['plan_rng']),
	}
	passed = (
		result['rgb_equal']
		and result['shared_state_missing'] == 0
		and result['shared_state_mismatched'] == 0
		and result['unexpected_extra_state'] == 0
		and result['random_actions_equal']
		and result['latent_max_error'] == 0.0
		and result['first_action_max_error'] == 0.0
		and result['post_reset_rng_equal']
		and result['encode_rng_equal']
		and result['plan_rng_equal']
	)
	print('FLAT_ANCHOR_STEP0_PARITY_' + ('OK' if passed else 'FAIL'), result)
	if not passed:
		if missing:
			print('missing shared keys:', missing[:10])
		if mismatched:
			print('mismatched shared keys:', mismatched[:10])
		if unexpected_extra:
			print('unexpected extra keys:', unexpected_extra[:10])
		raise SystemExit(1)


if __name__ == '__main__':
	main()
