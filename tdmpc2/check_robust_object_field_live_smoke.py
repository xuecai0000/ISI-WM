"""Real Cutie + DMC + planning smoke for ROF-WM V0.

This check performs no training and writes no experiment result. It starts the
same isolated Cutie worker used by production runs, builds one real DMC visual
environment, validates the atomic RGB/descriptor/mask packet, executes a small
number of MPPI actions, and strictly reloads an in-memory checkpoint.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
import time


# Select headless rendering before importing dm_control/MuJoCo.
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch
from omegaconf import OmegaConf

from common import robust_object_field as rof
from common.parser import cfg_to_dataclass
from common.seed import set_seed
from envs import make_env
from envs.wrappers.cutie_object import TASK_ROLE_NAMES
from tdmpc2 import TDMPC2


PUBLIC_KEYS = {'rgb', 'object', 'object_mask', 'role_exists'}


def arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument('--runtime-config', type=Path, required=True)
	parser.add_argument('--support-path', type=Path)
	parser.add_argument('--condition', choices=('clean', 'hard'), default='clean')
	parser.add_argument('--steps', type=int, default=2)
	parser.add_argument('--latency-iterations', type=int, default=50)
	return parser.parse_args()


def load_config(args: argparse.Namespace):
	path = args.runtime_config.expanduser().resolve()
	if not path.is_file():
		raise FileNotFoundError(path)
	raw = json.loads(path.read_text(encoding='utf-8'))
	task = str(raw.get('task', ''))
	if task not in TASK_ROLE_NAMES:
		raise ValueError(f'Unsupported ROF V0 smoke task {task!r}.')
	roles = list(TASK_ROLE_NAMES[task])
	role_count = len(roles)
	if role_count not in {1, 2, 3}:
		raise ValueError(f'ROF V0 requires 1-3 real roles, got {roles!r}.')
	if int(raw.get('model_size', -1)) != 5:
		raise ValueError('ROF V0 live smoke requires the production 5M model.')

	support = args.support_path or Path(str(raw.get('cutie_object_support_path', '')))
	support = support.expanduser().resolve()
	if not support.is_file():
		raise FileNotFoundError(support)
	support_payload = json.loads(support.read_text(encoding='utf-8'))
	if support_payload.get('format') == 'cutie_indexed_mask_support_v1':
		collection = support_payload.get('collection', {})
		support_schema = collection.get('support_schema')
		if support_schema != 'generic_indexed_v1':
			raise ValueError(
				'Indexed support must declare collection.support_schema='
				"'generic_indexed_v1'."
			)
		if collection.get('task') != task:
			raise ValueError(
				f'Support task {collection.get("task")!r} != runtime task {task!r}.'
			)
		if list(support_payload.get('roles', ())) != roles:
			raise ValueError('Indexed support roles differ from the canonical task roles.')
		allow_simulator_support = True
	else:
		support_schema = str(raw.get('cutie_object_support_schema', ''))
		if support_schema != 'whole_arm_goal_v1':
			raise ValueError(f'Unrecognized support format in {support}.')
		allow_simulator_support = False
	for key in ('cutie_object_repo', 'cutie_object_checkpoint'):
		asset = Path(str(raw.get(key, ''))).expanduser().resolve()
		if not asset.exists():
			raise FileNotFoundError(f'{key}: {asset}')

	width = role_count * 5 * 64
	raw.update(
		obs='rgb',
		multitask=False,
		tasks=[task],
		task_dim=0,
		flat_anchor=True,
		flat_anchor_mode='cutie_object_only',
		cutie_object_observation_variant='full',
		cutie_object_frame_schema='cutie_query_mask_status_v1',
		cutie_object_frame_dim=590,
		cutie_object_stack_frames=3,
		cutie_object_input_dim=1770,
		cutie_object_num_roles=role_count,
		cutie_object_role_names=roles,
		cutie_object_task_role_contract='canonical_v1',
		cutie_object_support_schema=support_schema,
		cutie_object_allow_simulator_support=allow_simulator_support,
		cutie_object_support_path=str(support),
		cutie_object_allow_simulator_runtime=False,
		cutie_object_allow_simulator_kinematics_runtime=False,
		cutie_object_native_highres_enabled=False,
		cutie_object_true_entity_enabled=False,
		cutie_object_spatial_token_enabled=False,
		cutie_object_variable_graph_enabled=False,
		cutie_object_belief_enabled=False,
		cutie_object_belief_use_for_control=False,
		cutie_object_last_valid_memory=False,
		cutie_object_policy_burst_plan=None,
		cutie_object_regression_encoder=None,
		cutie_masked_rgb_enabled=False,
		object_state_supervision_enabled=False,
		object_state_supervision_collect_labels=False,
		object_state_bottleneck_enabled=False,
		visual_foreground_erosion_pixels=0,
		visual_pose_checkpoint=None,
		video_background_enabled=args.condition == 'hard',
		robust_object_field_enabled=True,
		robust_object_field_schema=rof.SCHEMA,
		robust_object_field_local_tokens=4,
		robust_object_field_token_dim=64,
		robust_object_field_context_radius=3,
		robust_object_field_query_mode='ordered_compressed',
		robust_object_field_random_shift_pad=3,
		cutie_object_only_latent_dim=width,
		latent_dim=width,
		compile=False,
		compile_fallback_random=False,
		save_agent=False,
		save_csv=False,
		save_video=False,
		enable_wandb=False,
		checkpoint=None,
		work_dir=str(path.parent / 'rof_wm_v0_live_smoke_no_output'),
	)
	cfg = cfg_to_dataclass(OmegaConf.create(raw))
	cfg.work_dir = path.parent / 'rof_wm_v0_live_smoke_no_output'
	return cfg, roles


def find_field_wrapper(env):
	cursor = env
	seen = set()
	while cursor is not None and id(cursor) not in seen:
		seen.add(id(cursor))
		if type(cursor).__name__ == 'RobustObjectFieldObservationWrapper':
			return cursor
		cursor = getattr(cursor, 'env', None)
	raise AssertionError('The real environment did not install the ROF wrapper.')


def validate_observation(obs, cfg) -> int:
	if set(obs.keys()) != PUBLIC_KEYS:
		raise AssertionError(f'Unexpected ROF public keys: {set(obs.keys())!r}.')
	expected = rof.observation_shapes(cfg)
	actual = {key: tuple(obs[key].shape) for key in PUBLIC_KEYS}
	if actual != expected:
		raise AssertionError(f'ROF observation shapes differ: {actual} != {expected}.')
	if obs['rgb'].dtype != torch.uint8:
		raise AssertionError('Real ROF RGB is not uint8.')
	if obs['object'].dtype != torch.float32:
		raise AssertionError('Real ROF descriptors are not float32.')
	if obs['object_mask'].dtype != torch.bool:
		raise AssertionError('Real ROF masks are not boolean.')
	if obs['role_exists'].dtype != torch.float32:
		raise AssertionError('Real ROF role_exists is not float32.')
	if not bool(torch.isfinite(obs['object']).all()):
		raise AssertionError('Real ROF descriptors contain non-finite values.')
	if not bool((obs['role_exists'] == 1).all()):
		raise AssertionError('ROF admitted a padding/missing role.')
	# Empty masks are a valid causal tracker measurement and are represented by
	# descriptor status fields. Exact-K means real semantic slots, not that every
	# object must be visible in every frame.
	visible = obs['object_mask'][:, -1].flatten(1).any(dim=1)
	return int((~visible).sum().item())


@torch.no_grad()
def measure_encoder_latency(agent, obs, iterations: int) -> dict[str, float]:
	if iterations < 10 or iterations > 500:
		raise ValueError('--latency-iterations must be in [10,500].')
	batch = obs.to(agent.device, non_blocking=True).unsqueeze(0)
	for _ in range(10):
		agent.model.encode(batch, task=None)
	torch.cuda.synchronize(agent.device)
	values = []
	for _ in range(iterations):
		start = time.perf_counter()
		agent.model.encode(batch, task=None)
		torch.cuda.synchronize(agent.device)
		values.append((time.perf_counter() - start) * 1000.0)
	array = np.asarray(values, dtype=np.float64)
	return {
		'mean_ms': float(array.mean()),
		'median_ms': float(np.median(array)),
		'p95_ms': float(np.percentile(array, 95)),
		'max_ms': float(array.max()),
		'iterations': int(iterations),
	}


def main() -> None:
	args = arguments()
	if args.steps < 1 or args.steps > 10:
		raise ValueError('--steps must be in [1,10].')
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required for the real ROF live smoke.')
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.set_float32_matmul_precision('high')

	cfg, roles = load_config(args)
	set_seed(int(cfg.seed))
	env = None
	agent = None
	reloaded = None
	try:
		rof.validate_config(cfg, require_obs_shape=False)
		env = make_env(cfg)
		rof.validate_config(cfg)
		field_wrapper = find_field_wrapper(env)
		ready = field_wrapper.robust_object_field_ready
		if ready is None:
			raise AssertionError('Cutie did not publish its ready record.')
		if ready.get('privileged_runtime_segmentation') is not False:
			raise AssertionError('Live smoke exposed privileged runtime segmentation.')
		if ready.get('privileged_runtime_kinematics') is not False:
			raise AssertionError('Live smoke exposed privileged runtime kinematics.')
		if ready.get('role_names') != roles:
			raise AssertionError('Cutie ready record changed the canonical role order.')
		if ready.get('policy_native_role_masks') is not True:
			raise AssertionError('Cutie did not expose native same-response role masks.')

		obs = env.reset()
		empty_role_measurements = validate_observation(obs, cfg)
		bindings = [field_wrapper.robust_object_field_last_binding_sha256]
		if not bindings[0] or len(bindings[0]) != 64:
			raise AssertionError('Initial RGB/descriptor/mask binding is missing.')

		rebuild_cfg = deepcopy(cfg)
		agent = TDMPC2(cfg).eval()
		if set(agent.model._encoder) != {'object'}:
			raise AssertionError('ROF installed an unintended scene-encoder bypass.')
		if int(agent.cfg.latent_dim) != rof.latent_dim(cfg):
			raise AssertionError('ROF controller latent width changed after construction.')
		encoder_latency = measure_encoder_latency(
			agent, obs, args.latency_iterations
		)

		rewards = []
		for step in range(args.steps):
			action = agent.act(obs, t0=(step == 0), eval_mode=True)
			if (
				tuple(action.shape) != tuple(env.action_space.shape)
				or not bool(torch.isfinite(action).all())
				or bool((action.abs() > 1).any())
			):
				raise AssertionError(f'Invalid planned action at step {step}.')
			obs, reward, done, _ = env.step(action)
			empty_role_measurements += validate_observation(obs, cfg)
			if done or not np.isfinite(float(reward)):
				raise AssertionError(
					f'Invalid transition at step {step}: reward={reward}, done={done}.'
				)
			rewards.append(float(reward))
			bindings.append(field_wrapper.robust_object_field_last_binding_sha256)
		if any(not value or len(value) != 64 for value in bindings):
			raise AssertionError('A stepped atomic observation binding is missing.')

		checkpoint = BytesIO()
		agent.save(checkpoint)
		checkpoint.seek(0)
		payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
		if payload['checkpoint_contract'].get('robust_object_field') != rof.contract(cfg):
			raise AssertionError('Live checkpoint lost the exact ROF contract.')
		reloaded = TDMPC2(rebuild_cfg).eval()
		reloaded.load(payload)

		print('ROBUST_OBJECT_FIELD_V0_LIVE_SMOKE_OK', json.dumps({
			'task': cfg.task,
			'condition': args.condition,
			'roles': roles,
			'observation_shapes': rof.observation_shapes(cfg),
			'latent_dim': int(cfg.latent_dim),
			'action_dim': int(cfg.action_dim),
			'planned_steps': args.steps,
			'rewards_finite': all(np.isfinite(rewards)),
			'atomic_bindings': len(bindings),
			'empty_role_measurements': empty_role_measurements,
			'checkpoint_strict_reload': True,
			'rof_encoder_batch1_latency': encoder_latency,
			'latency_semantics': 'rof_encoder_only_excludes_cutie_and_mppi',
			'runtime_simulator_state': False,
			'runtime_simulator_segmentation': False,
		}, sort_keys=True))
	finally:
		close = getattr(env, 'close', None) if env is not None else None
		if callable(close):
			close()
		del reloaded, agent
		torch.cuda.empty_cache()


if __name__ == '__main__':
	main()
