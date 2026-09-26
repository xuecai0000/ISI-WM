"""End-to-end step-0 parity audit for RGB and HybridGraph TD-MPC2.

No training is performed. The audit loads the real frozen anchor teacher and
requires the HybridGraph path to preserve every official RGB output at
initialization, including the first MPPI evaluation action and RNG stream.
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
SCENE_DIM = 512
GRAPH_DIM = 128


def make_config(args, hybrid):
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
	cfg.compile_fallback_random = args.compile
	cfg.flat_anchor = hybrid
	cfg.flat_anchor_mode = 'hybrid_graph'
	cfg.flat_anchor_support_path = str(args.support) if hybrid else None
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


def seed_torch(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def clone_value(value):
	return value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)


def value_equal(left, right):
	if torch.is_tensor(left) or torch.is_tensor(right):
		return torch.is_tensor(left) and torch.is_tensor(right) \
			and torch.equal(left, right.detach().cpu())
	if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
		return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) \
			and np.array_equal(left, right)
	return type(left) is type(right) and left == right


def run_path(args, hybrid):
	cfg = make_config(args, hybrid)
	set_seed(args.seed)
	env = make_env(cfg)
	agent = TDMPC2(cfg)
	post_construction_rng = capture_rng()
	obs = env.reset()
	rgb = (obs['rgb'] if hybrid else obs).detach().cpu().clone()
	post_reset_rng = capture_rng()
	random_actions = torch.stack([env.rand_act() for _ in range(16)]).cpu()

	seed_torch(args.encode_seed)
	device_obs = obs.to(agent.device).unsqueeze(0)
	latent = agent.model.encode(device_obs, task=None)
	encode_rng = capture_rng()

	# The real update encodes H next frames before the current frame. Official
	# TD-MPC2 samples an independent RGB shift for every encoded frame; HybridGraph
	# must consume the same CUDA RNG and apply the same RGB crops while shifting
	# each frame's anchor covariantly.
	base_rgb = device_obs['rgb'] if hybrid else device_obs
	sequence_rgb = torch.stack([
		base_rgb,
		torch.roll(base_rgb, shifts=1, dims=-1),
		torch.roll(base_rgb, shifts=-1, dims=-2),
	])
	if hybrid:
		base_anchor = device_obs['anchor']
		sequence_obs = {
			'rgb': sequence_rgb,
			'anchor': torch.stack([base_anchor, base_anchor, base_anchor]),
		}
	else:
		sequence_obs = sequence_rgb
	seed_torch(args.sequence_seed)
	sequence_latent = agent.model.encode(sequence_obs, task=None)
	sequence_rng = capture_rng()

	probe_action = torch.tensor([[0.25, -0.5]], device=agent.device)
	seed_torch(args.probe_seed)
	next_latent = agent.model.next(latent, probe_action, None)
	reward = agent.model.reward(latent, probe_action, None)
	q_online = agent.model.Q(latent, probe_action, None, return_type='all')
	q_target = agent.model.Q(
		latent, probe_action, None, return_type='all', target=True
	)
	q_detach = agent.model.Q(
		latent, probe_action, None, return_type='all', detach=True
	)
	pi_action, pi_info = agent.model.pi(latent, None)
	probe_rng = capture_rng()

	seed_torch(args.plan_seed)
	agent._prev_mean.zero_()
	first_action = agent.act(obs, t0=True, eval_mode=True).detach().cpu()
	plan_rng = capture_rng()

	state = {
		key: clone_value(value)
		for key, value in agent.model.state_dict().items()
	}
	result = {
		'rgb': rgb,
		'random_actions': random_actions,
		'latent': latent.detach().cpu(),
		'sequence_latent': sequence_latent.detach().cpu(),
		'next_latent': next_latent.detach().cpu(),
		'reward': reward.detach().cpu(),
		'q_online': q_online.detach().cpu(),
		'q_target': q_target.detach().cpu(),
		'q_detach': q_detach.detach().cpu(),
		'pi_action': pi_action.detach().cpu(),
		'pi_mean': pi_info['mean'].detach().cpu(),
		'pi_log_std': pi_info['log_std'].detach().cpu(),
		'first_action': first_action,
		'post_construction_rng': post_construction_rng,
		'post_reset_rng': post_reset_rng,
		'encode_rng': encode_rng,
		'sequence_rng': sequence_rng,
		'probe_rng': probe_rng,
		'plan_rng': plan_rng,
		'state': state,
		'joint_dim': cfg.latent_dim,
	}
	return result, env, agent


def max_error(left, right):
	return float((left.float() - right.float()).abs().max())


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--support', type=Path, required=True)
	parser.add_argument('--seed', type=int, default=2)
	parser.add_argument('--encode-seed', type=int, default=161803)
	parser.add_argument('--sequence-seed', type=int, default=173205)
	parser.add_argument('--probe-seed', type=int, default=141421)
	parser.add_argument('--plan-seed', type=int, default=271828)
	parser.add_argument('--compile', action='store_true')
	args = parser.parse_args()
	if not args.support.is_file():
		raise FileNotFoundError(args.support)
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required for the HybridGraph parity audit.')
	torch.backends.cudnn.benchmark = False
	torch.set_float32_matmul_precision('high')

	print('Building official RGB path...')
	official, official_env, official_agent = run_path(args, False)
	del official_env, official_agent
	gc.collect()
	torch.cuda.empty_cache()

	print('Building HybridGraph path...')
	hybrid, hybrid_env, hybrid_agent = run_path(args, True)
	action_mask = hybrid_agent.model._graph_dynamics._action_role_mask.flatten().cpu()
	probe_anchor = torch.stack([
		torch.linspace(-0.8, 0.8, 25, device=hybrid_agent.device),
		torch.linspace(0.7, -0.7, 25, device=hybrid_agent.device),
	])
	fixed_shift = torch.tensor(
		[[[[1., 5.]]], [[[6., 0.]]]],
		device=hybrid_agent.device,
		dtype=torch.float32,
	)
	probe_rgb = hybrid['rgb'].to(hybrid_agent.device).unsqueeze(0).expand(
		2, -1, -1, -1
	).clone()
	_, shifted_anchor = hybrid_agent.model._oc_augmentation(
		probe_rgb, probe_anchor, fixed_shift
	)
	pad = hybrid_agent.model._oc_augmentation.pad
	delta = (pad - fixed_shift[:, 0, 0]) * (2. / 63.)
	expected_positions = probe_anchor[:, :8].reshape(2, 4, 2) + delta.unsqueeze(1)
	augmentation_contract = (
		torch.equal(shifted_anchor[:, :8], expected_positions.reshape(2, 8))
		and torch.equal(shifted_anchor[:, 8:], probe_anchor[:, 8:])
	)
	zero_corrections = all(
		float(parameter.detach().abs().max()) == 0.
		for parameter in (
			hybrid_agent.model._hybrid_reward.output.weight,
			hybrid_agent.model._hybrid_reward.output.bias,
			hybrid_agent.model._hybrid_pi.output.weight,
			hybrid_agent.model._hybrid_pi.output.bias,
			hybrid_agent.model._hybrid_q.params['output', 'weight'],
			hybrid_agent.model._hybrid_q.params['output', 'bias'],
		)
	)
	# Hybrid checkpoints stay strict: deleting even one branch tensor must fail.
	damaged_state = dict(hybrid['state'])
	damaged_key = next(
		key for key in damaged_state if key.startswith('_hybrid_reward.')
	)
	del damaged_state[damaged_key]
	try:
		hybrid_agent.load({'model': damaged_state})
	except RuntimeError:
		damaged_checkpoint_rejected = True
	else:
		damaged_checkpoint_rejected = False
	# A pure official checkpoint may warm-start the exact scene path only through
	# an explicit opt-in; the allowlisted graph branch keeps its fresh identity init.
	hybrid_agent.load(
		{'model': official['state']}, allow_official_warmstart=True
	)
	official_checkpoint_load = True
	del hybrid_env, hybrid_agent
	gc.collect()
	torch.cuda.empty_cache()

	missing = [key for key in official['state'] if key not in hybrid['state']]
	mismatched = [
		key for key, value in official['state'].items()
		if key in hybrid['state'] and not value_equal(value, hybrid['state'][key])
	]
	extra = sorted(set(hybrid['state']) - set(official['state']))
	allowed_prefixes = (
		'_encoder.anchor.',
		'_graph_dynamics.',
		'_anchor_decoder.',
		'_hybrid_reward.',
		'_hybrid_pi.',
		'_hybrid_q.',
		'_detach_hybrid_q_params.',
		'_target_hybrid_q_params.',
	)
	unexpected_extra = [
		key for key in extra if not key.startswith(allowed_prefixes)
	]

	comparisons = {
		'rgb': (official['rgb'], hybrid['rgb']),
		'random_actions': (official['random_actions'], hybrid['random_actions']),
		'latent': (official['latent'], hybrid['latent'][..., :SCENE_DIM]),
		'sequence_latent': (
			official['sequence_latent'],
			hybrid['sequence_latent'][..., :SCENE_DIM],
		),
		'next_latent': (
			official['next_latent'], hybrid['next_latent'][..., :SCENE_DIM]
		),
		'reward': (official['reward'], hybrid['reward']),
		'q_online': (official['q_online'], hybrid['q_online']),
		'q_target': (official['q_target'], hybrid['q_target']),
		'q_detach': (official['q_detach'], hybrid['q_detach']),
		'pi_action': (official['pi_action'], hybrid['pi_action']),
		'pi_mean': (official['pi_mean'], hybrid['pi_mean']),
		'pi_log_std': (official['pi_log_std'], hybrid['pi_log_std']),
		'first_action': (official['first_action'], hybrid['first_action']),
	}
	errors = {name: max_error(*values) for name, values in comparisons.items()}
	rng_checks = {
		name: rng_equal(official[name], hybrid[name])
		for name in (
			'post_construction_rng', 'post_reset_rng', 'encode_rng', 'sequence_rng',
			'probe_rng', 'plan_rng',
		)
	}
	result = {
		'compile': args.compile,
		'compile_fallback_random': args.compile,
		'joint_dim': hybrid['joint_dim'],
		'graph_dim': hybrid['latent'].shape[-1] - SCENE_DIM,
		'shared_state_missing': len(missing),
		'shared_state_mismatched': len(mismatched),
		'unexpected_extra_state': len(unexpected_extra),
		'action_role_mask': action_mask.tolist(),
		'augmentation_contract': augmentation_contract,
		'zero_corrections': zero_corrections,
		'official_checkpoint_load': official_checkpoint_load,
		'damaged_checkpoint_rejected': damaged_checkpoint_rejected,
		**{f'{name}_max_error': error for name, error in errors.items()},
		**rng_checks,
	}
	passed = (
		hybrid['joint_dim'] == SCENE_DIM + GRAPH_DIM
		and hybrid['latent'].shape[-1] == SCENE_DIM + GRAPH_DIM
		and len(missing) == len(mismatched) == len(unexpected_extra) == 0
		and torch.equal(action_mask, torch.tensor([0., 1., 1., 0.]))
		and augmentation_contract
		and zero_corrections
		and official_checkpoint_load
		and damaged_checkpoint_rejected
		and all(error == 0. for error in errors.values())
		and all(rng_checks.values())
	)
	print('HYBRID_GRAPH_STEP0_PARITY_' + ('OK' if passed else 'FAIL'), result)
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
