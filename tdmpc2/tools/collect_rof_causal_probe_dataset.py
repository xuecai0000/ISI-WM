"""Collect an offline causal-probe dataset from an existing ROF checkpoint.

This tool is diagnostic-only.  The policy receives the unchanged four-field
Robust Object Field observation while privileged dm-control state and MuJoCo
segmentation are read *after* that observation has been produced and are saved
only under the ``labels__`` NPZ namespace.  Labels are never passed to the
agent, environment step, reward, or action-selection code.

Each episode is stored as one compressed NPZ.  Observation and label arrays
have length ``T+1``; action/reward/done arrays have length ``T`` and obey
``obs[t], action[t], reward[t], done[t], obs[t+1]`` semantics.  The JSON
manifest assigns complete episodes (never frames) to train/validation/test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import uuid

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
for _local_path in (str(REPO_DIR), str(PROJECT_DIR)):
	while _local_path in sys.path:
		sys.path.remove(_local_path)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))


FORMAT = 'rof_causal_probe_dataset_v1'
MANIFEST_NAME = 'dataset_manifest.json'
TEMPORAL_SEMANTICS = 'obs_t_action_t_reward_t_done_t_obs_t_plus_1'
TASKS = (
	'acrobot-swingup',
	'walker-run',
	'reacher-easy',
	'finger-turn-easy',
	'cartpole-balance-sparse',
)
POLICY_INPUT_KEYS = (
	'policy_rgb',
	'policy_object',
	'policy_object_mask',
	'policy_role_exists',
)
LABEL_KEYS = (
	'labels__state',
	'labels__gt_role_mask',
	'labels__gt_visible',
)
AUXILIARY_KEYS = ('action', 'reward', 'done', 'episode_id', 'step')
EPISODE_KEYS = frozenset(POLICY_INPUT_KEYS + LABEL_KEYS + AUXILIARY_KEYS)


def canonical_json_bytes(payload: Any) -> bytes:
	"""Return the single canonical JSON encoding used by this dataset."""
	return (
		json.dumps(
			payload, ensure_ascii=False, sort_keys=True,
			separators=(',', ':'), allow_nan=False,
		) + '\n'
	).encode('utf-8')


def file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with Path(path).open('rb') as file:
		for block in iter(lambda: file.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def typed_array_sha256(name: str, array: np.ndarray) -> str:
	"""Hash an array with its logical name, dtype, shape, and C-order bytes."""
	if not isinstance(name, str) or not name:
		raise ValueError('Array hash name must be a non-empty string.')
	value = np.ascontiguousarray(np.asarray(array))
	digest = hashlib.sha256()
	digest.update(name.encode('utf-8'))
	digest.update(b'\0')
	digest.update(value.dtype.str.encode('ascii'))
	digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
	digest.update(value.tobytes(order='C'))
	return digest.hexdigest()


def _array_trace_sha256(arrays: Mapping[str, np.ndarray], keys: Sequence[str]) -> str:
	if set(arrays).issuperset(keys) is False:
		missing = sorted(set(keys) - set(arrays))
		raise ValueError(f'Array trace is missing {missing!r}.')
	digest = hashlib.sha256()
	for key in keys:
		digest.update(key.encode('ascii'))
		digest.update(typed_array_sha256(key, arrays[key]).encode('ascii'))
	return digest.hexdigest()


def policy_input_trace_sha256(arrays: Mapping[str, np.ndarray]) -> str:
	"""Hash only deployable policy inputs; label values cannot affect it."""
	return _array_trace_sha256(arrays, POLICY_INPUT_KEYS)


def label_trace_sha256(arrays: Mapping[str, np.ndarray]) -> str:
	"""Hash only privileged offline labels."""
	return _array_trace_sha256(arrays, LABEL_KEYS)


def _require_array(
	arrays: Mapping[str, np.ndarray], name: str, shape: tuple[int, ...], dtype,
) -> np.ndarray:
	value = np.asarray(arrays[name])
	if value.shape != shape or value.dtype != np.dtype(dtype):
		raise ValueError(
			f'{name} must be {np.dtype(dtype)} {shape}, got '
			f'{value.dtype} {value.shape}.'
		)
	if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
		raise ValueError(f'{name} contains a non-finite value.')
	return value


def validate_episode_arrays(
	arrays: Mapping[str, np.ndarray], *, role_count: int, state_dim: int,
	action_dim: int, steps: int, episode_index: int | None = None,
) -> dict[str, str]:
	"""Validate one episode asset and return its isolated trace hashes."""
	if not isinstance(arrays, Mapping) or set(arrays) != EPISODE_KEYS:
		actual = sorted(arrays) if isinstance(arrays, Mapping) else type(arrays)
		raise ValueError(
			f'Episode arrays must be exactly {sorted(EPISODE_KEYS)!r}, got {actual!r}.'
		)
	for value, label in (
		(role_count, 'role_count'), (state_dim, 'state_dim'),
		(action_dim, 'action_dim'), (steps, 'steps'),
	):
		if not isinstance(value, int) or isinstance(value, bool) or value < 1:
			raise ValueError(f'{label} must be a positive integer.')
	frames = steps + 1
	_require_array(arrays, 'policy_rgb', (frames, 9, 64, 64), np.uint8)
	_require_array(
		arrays, 'policy_object', (frames, role_count, 1770), np.float32,
	)
	_require_array(
		arrays, 'policy_object_mask',
		(frames, role_count, 3, 64, 64), np.bool_,
	)
	role_exists = _require_array(
		arrays, 'policy_role_exists', (frames, role_count), np.float32,
	)
	if not np.array_equal(role_exists, np.ones_like(role_exists)):
		raise ValueError('ROF exact-K role_exists must be all one.')
	_require_array(arrays, 'action', (steps, action_dim), np.float32)
	_require_array(arrays, 'reward', (steps,), np.float32)
	done = _require_array(arrays, 'done', (steps,), np.bool_)
	if bool(done[:-1].any()) or not bool(done[-1]):
		raise ValueError('done must be false before, and true at, the final action.')
	_require_array(arrays, 'labels__state', (frames, state_dim), np.float64)
	gt = _require_array(
		arrays, 'labels__gt_role_mask',
		(frames, role_count, 64, 64), np.bool_,
	)
	visible = _require_array(
		arrays, 'labels__gt_visible', (frames, role_count), np.bool_,
	)
	measured_visible = gt.reshape(frames, role_count, -1).any(axis=-1)
	if not np.array_equal(visible, measured_visible):
		raise ValueError('labels__gt_visible disagrees with labels__gt_role_mask.')
	if np.any(gt.sum(axis=1) > 1):
		raise ValueError('Privileged role masks overlap.')
	episode_ids = _require_array(arrays, 'episode_id', (frames,), np.int64)
	if not np.all(episode_ids == episode_ids[0]):
		raise ValueError('episode_id must be constant within an episode asset.')
	if episode_index is not None and int(episode_ids[0]) != int(episode_index):
		raise ValueError('episode_id does not match the manifest episode_index.')
	step = _require_array(arrays, 'step', (frames,), np.int64)
	if not np.array_equal(step, np.arange(frames, dtype=np.int64)):
		raise ValueError('step must be exactly [0, ..., T].')
	return {
		'policy_input_trace_sha256': policy_input_trace_sha256(arrays),
		'label_trace_sha256': label_trace_sha256(arrays),
	}


def _safe_asset_path(root: Path, relative: str) -> Path:
	if not isinstance(relative, str) or not relative:
		raise ValueError('Episode path must be a non-empty relative path.')
	path = (root / relative).resolve()
	try:
		path.relative_to(root.resolve())
	except ValueError as exc:
		raise ValueError(f'Episode path escapes the dataset root: {relative!r}.') from exc
	if path.is_symlink():
		raise ValueError(f'Symlinked episode assets are forbidden: {path}.')
	return path


def validate_dataset(manifest_path: Path) -> dict[str, Any]:
	"""Fully validate a published or incomplete dataset manifest and assets."""
	manifest_path = Path(manifest_path).resolve()
	if not manifest_path.is_file() or manifest_path.is_symlink():
		raise FileNotFoundError(manifest_path)
	payload = json.loads(manifest_path.read_text(encoding='utf-8'))
	if not isinstance(payload, dict) or payload.get('format') != FORMAT:
		raise ValueError('Causal-probe manifest format mismatch.')
	if payload.get('temporal_semantics') != TEMPORAL_SEMANTICS:
		raise ValueError('Causal-probe temporal semantics mismatch.')
	if tuple(payload.get('policy_observation_keys', ())) != POLICY_INPUT_KEYS:
		raise ValueError('Policy observation key contract mismatch.')
	if tuple(payload.get('label_keys', ())) != LABEL_KEYS:
		raise ValueError('Offline label key contract mismatch.')
	label_contract = payload.get('label_contract')
	if not isinstance(label_contract, dict) or any((
		label_contract.get('namespace') != 'labels__',
		label_contract.get('labels_never_policy_input') is not True,
		label_contract.get('queried_after_policy_observation') is not True,
		label_contract.get('same_physics_state_guard') != 'exact_before_after',
	)):
		raise ValueError('Offline label isolation contract mismatch.')
	roles = payload.get('role_names')
	state_names = label_contract.get('state_names')
	action_dim = payload.get('action_dim')
	steps = payload.get('steps')
	if (
		not isinstance(roles, list) or not roles or len(set(roles)) != len(roles)
		or not isinstance(state_names, list) or not state_names
		or not isinstance(action_dim, int) or action_dim < 1
		or not isinstance(steps, int) or steps < 1
	):
		raise ValueError('Dataset dimensions/role/state metadata are malformed.')
	episodes = payload.get('episodes')
	if not isinstance(episodes, list) or len(episodes) < 5:
		raise ValueError('Dataset requires at least five complete episodes.')
	by_id: dict[int, dict[str, Any]] = {}
	root = manifest_path.parent
	for record in episodes:
		if not isinstance(record, dict):
			raise ValueError('Episode manifest record must be an object.')
		episode_index = record.get('episode_index')
		if (
			not isinstance(episode_index, int) or isinstance(episode_index, bool)
			or episode_index < 0 or episode_index in by_id
			or record.get('steps') != steps
			or record.get('temporal_semantics') != TEMPORAL_SEMANTICS
		):
			raise ValueError(f'Malformed episode record: {record!r}.')
		path = _safe_asset_path(root, record.get('path'))
		if not path.is_file() or file_sha256(path) != record.get('sha256'):
			raise ValueError(f'Episode asset identity mismatch: {path}.')
		with np.load(path, allow_pickle=False) as archive:
			arrays = {name: archive[name] for name in archive.files}
		hashes = validate_episode_arrays(
			arrays, role_count=len(roles), state_dim=len(state_names),
			action_dim=action_dim, steps=steps, episode_index=episode_index,
		)
		for key, expected in hashes.items():
			if record.get(key) != expected:
				raise ValueError(f'{key} mismatch for episode {episode_index}.')
		by_id[episode_index] = record
	if set(by_id) != set(range(len(episodes))):
		raise ValueError('Episode indices must be contiguous from zero.')
	splits = payload.get('splits')
	if not isinstance(splits, dict) or set(splits) != {'train', 'validation', 'test'}:
		raise ValueError('Dataset must define train/validation/test episode splits.')
	seen: set[int] = set()
	for split in ('train', 'validation', 'test'):
		indices = splits[split]
		if not isinstance(indices, list) or not indices:
			raise ValueError(f'{split} split must contain complete episodes.')
		if any(not isinstance(index, int) or index not in by_id for index in indices):
			raise ValueError(f'{split} split contains an invalid episode index.')
		if seen.intersection(indices):
			raise ValueError('Episode splits overlap.')
		if any(by_id[index].get('split') != split for index in indices):
			raise ValueError('Episode record split disagrees with the split table.')
		seen.update(indices)
	if seen != set(by_id):
		raise ValueError('Episode splits do not cover the dataset exactly.')
	return payload


def _torch_observation_arrays(observation) -> dict[str, np.ndarray]:
	try:
		keys = set(observation.keys())
	except AttributeError as exc:
		raise RuntimeError('ROF policy observation must be a keyed TensorDict.') from exc
	expected = {'rgb', 'object', 'object_mask', 'role_exists'}
	if keys != expected:
		raise RuntimeError(f'ROF policy observation keys {keys!r} != {expected!r}.')
	result = {}
	for source, target, dtype in (
		('rgb', 'policy_rgb', np.uint8),
		('object', 'policy_object', np.float32),
		('object_mask', 'policy_object_mask', np.bool_),
		('role_exists', 'policy_role_exists', np.float32),
	):
		value = observation[source].detach().cpu().contiguous().numpy()
		result[target] = np.array(value, dtype=dtype, order='C', copy=True)
	return result


def _walk_children(env):
	current, seen = env, set()
	while current is not None and id(current) not in seen:
		seen.add(id(current))
		yield current
		namespace = vars(current)
		current = namespace.get('env', namespace.get('_env'))


def _find_dm_control_source(env):
	current, seen = env, set()
	while current is not None and id(current) not in seen:
		seen.add(id(current))
		namespace = vars(current)
		child = namespace.get('env', namespace.get('_env'))
		if child is not None:
			current = child
			continue
		task = getattr(current, 'task', None)
		physics = getattr(current, 'physics', None)
		if task is not None and physics is not None and callable(
			getattr(task, 'get_observation', None)
		):
			return current
		break
	raise RuntimeError('Cannot find official dm-control task/physics label source.')


def _find_camera_id(env) -> int:
	for current in _walk_children(env):
		if 'camera_id' in vars(current):
			value = vars(current)['camera_id']
			if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
				return value
	raise RuntimeError('Cannot find the concrete policy camera id.')


def _flatten_official_state(
	observation: Mapping[str, Any], schema: list[dict[str, Any]] | None,
) -> tuple[np.ndarray, list[dict[str, Any]], list[str]]:
	if not isinstance(observation, Mapping) or not observation:
		raise RuntimeError('Official dm-control observation must be a non-empty mapping.')
	actual_schema = []
	names = []
	chunks = []
	offset = 0
	for field, raw_value in observation.items():
		if not isinstance(field, str) or not field:
			raise RuntimeError('Official state field names must be non-empty strings.')
		value = np.asarray(raw_value, dtype=np.float64)
		if not np.isfinite(value).all():
			raise RuntimeError(f'Official state field {field!r} is non-finite.')
		flat = value.reshape(-1)
		shape = list(value.shape)
		actual_schema.append({
			'name': field, 'shape': shape, 'start': offset,
			'stop': offset + int(flat.size),
		})
		if value.ndim == 0:
			names.append(field)
		else:
			for index in np.ndindex(value.shape):
				names.append(f'{field}[{",".join(str(i) for i in index)}]')
		chunks.append(flat)
		offset += int(flat.size)
	if schema is not None and actual_schema != schema:
		raise RuntimeError('Official dm-control state schema changed within the dataset.')
	state = np.concatenate(chunks).astype(np.float64, copy=False)
	return np.ascontiguousarray(state), actual_schema, names


def _gt_role_masks(physics, *, camera_id: int, selections) -> np.ndarray:
	try:
		from dm_control.mujoco.wrapper.mjbindings import enums
	except ImportError as exc:
		raise RuntimeError('MuJoCo segmentation constants are unavailable.') from exc
	segmentation = np.asarray(physics.render(
		height=64, width=64, camera_id=int(camera_id), segmentation=True,
	))
	if segmentation.shape != (64, 64, 2):
		raise RuntimeError(
			f'MuJoCo segmentation must be [64,64,2], got {segmentation.shape}.'
		)
	types = {
		'geom': int(enums.mjtObj.mjOBJ_GEOM),
		'site': int(enums.mjtObj.mjOBJ_SITE),
	}
	masks = []
	for selected in selections:
		mask = np.zeros((64, 64), dtype=np.bool_)
		for object_type, object_id in selected:
			mask |= (
				(segmentation[..., 0] == int(object_id))
				& (segmentation[..., 1] == types[object_type])
			)
		masks.append(mask)
	result = np.ascontiguousarray(np.stack(masks, axis=0), dtype=np.bool_)
	if np.any(result.sum(axis=0) > 1):
		raise RuntimeError('Configured same-state GT role masks overlap.')
	return result


def _capture_labels(source, *, camera_id, selections, state_schema):
	physics = source.physics
	before = np.array(physics.get_state(), copy=True)
	state, schema, names = _flatten_official_state(
		source.task.get_observation(physics), state_schema,
	)
	masks = _gt_role_masks(physics, camera_id=camera_id, selections=selections)
	after = np.array(physics.get_state(), copy=True)
	if not np.array_equal(before, after):
		raise RuntimeError('Offline label queries changed the simulator physics state.')
	return state, masks, schema, names


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f'.{path.stem}.{uuid.uuid4().hex}.tmp.npz')
	try:
		with temporary.open('xb') as handle:
			np.savez_compressed(handle, **{
				key: np.ascontiguousarray(arrays[key]) for key in sorted(arrays)
			})
		os.replace(temporary, path)
	finally:
		if temporary.exists():
			temporary.unlink()
	return file_sha256(path)


def _json_safe_env_value(env, name, default=None):
	for current in _walk_children(env):
		if any(name in cls.__dict__ for cls in type(current).__mro__):
			return getattr(current, name)
	return default


def _source_paths(args) -> None:
	for path in (args.runtime_config, args.checkpoint):
		if not path.is_file() or path.is_symlink():
			raise FileNotFoundError(path)
	expected = (
		args.runtime_config.resolve().parent / 'models'
		/ ('final.pt' if args.checkpoint_step is None else f'eval_{args.checkpoint_step}.pt')
	)
	if args.checkpoint.resolve() != expected:
		raise ValueError(f'Checkpoint/runtime mismatch: {args.checkpoint} != {expected}.')


def _validate_args(args) -> None:
	_source_paths(args)
	if args.output_root.exists() or Path(str(args.output_root) + '.incomplete').exists():
		raise FileExistsError(args.output_root)
	if args.episodes < 5 or args.steps != 500:
		raise ValueError('Collection requires at least five complete 500-step episodes.')
	if args.train_episodes < 1 or args.validation_episodes < 1:
		raise ValueError('Train and validation episode counts must be positive.')
	if args.train_episodes + args.validation_episodes >= args.episodes:
		raise ValueError('At least one held-out test episode is required.')
	if len({args.env_seed, args.background_seed, args.planner_seed_base}) != 3:
		raise ValueError('Environment/background/planner seed domains must differ.')
	if args.checkpoint_step is not None and (
		args.checkpoint_step < 1
		or args.checkpoint_step > args.expected_training_steps
		or args.checkpoint_step % args.expected_training_eval_freq
	):
		raise ValueError('Periodic checkpoint step is outside the frozen training schedule.')


def collect(args) -> dict[str, Any]:
	"""Run the unchanged checkpoint policy and publish an isolated probe dataset."""
	import torch
	from common.seed import set_seed
	from envs import make_env
	from tdmpc2.tdmpc2 import TDMPC2
	from tdmpc2.tools import evaluate_cutie_multitask_checkpoint as evaluator
	from tdmpc2.tools.collect_cutie_multitask_support import (
		TASK_BY_NAME, _catalog, _selected_objects,
	)

	_validate_args(args)
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required for the official ROF checkpoint.')
	raw = evaluator._json(args.runtime_config)
	prepare_args = argparse.Namespace(**vars(args))
	prepare_args.backend = 'robust_object_field'
	prepare_args.erosion_pixels = 0
	prepare_args.output = args.output_root / MANIFEST_NAME
	cfg = evaluator._prepare(prepare_args, raw)
	roles = tuple(raw.get('cutie_object_role_names', ()))
	if args.task not in TASK_BY_NAME:
		raise RuntimeError(f'No privileged role selector exists for {args.task!r}.')
	spec = TASK_BY_NAME[args.task]
	if tuple(spec.roles) != roles:
		raise RuntimeError(
			f'GT role mapping {tuple(spec.roles)!r} != checkpoint roles {roles!r}; '
			'causal probe fails closed rather than relabeling the checkpoint.'
		)

	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.set_float32_matmul_precision('high')
	temporary_root = Path(str(args.output_root) + '.incomplete')
	temporary_root.mkdir(parents=True, exist_ok=False)
	env = None
	try:
		set_seed(args.env_seed)
		env = make_env(cfg)
		source = _find_dm_control_source(env)
		camera_id = _find_camera_id(env)
		catalog = _catalog(source.physics)
		selections = tuple(
			_selected_objects(catalog, selector) for selector in spec.selectors
		)
		identities = [identity for selected in selections for identity in selected]
		if len(identities) != len(set(identities)):
			raise RuntimeError('Configured GT role selectors share simulator objects.')
		agent = TDMPC2(cfg)
		agent.load(args.checkpoint)
		agent.eval()

		episode_records = []
		state_schema = None
		state_names = None
		for episode_index in range(args.episodes):
			observation = env.reset()
			planner_seed = args.planner_seed_base + episode_index
			set_seed(planner_seed)
			agent._prev_mean.zero_()
			policy_lists = {key: [] for key in POLICY_INPUT_KEYS}
			states, gt_masks, actions, rewards, dones = [], [], [], [], []
			background_source = _json_safe_env_value(env, 'active_source')
			background_start = _json_safe_env_value(env, 'frame_index')

			for step_index in range(args.steps + 1):
				policy = _torch_observation_arrays(observation)
				for key in POLICY_INPUT_KEYS:
					policy_lists[key].append(policy[key])
				state, masks, observed_schema, observed_names = _capture_labels(
					source, camera_id=camera_id, selections=selections,
					state_schema=state_schema,
				)
				if state_schema is None:
					state_schema = observed_schema
					state_names = observed_names
				elif observed_names != state_names:
					raise RuntimeError('Official dm-control state names changed.')
				states.append(state)
				gt_masks.append(masks)
				if step_index == args.steps:
					break
				torch.compiler.cudagraph_mark_step_begin()
				action = agent.act(
					observation, t0=step_index == 0, eval_mode=True,
				)
				action_array = np.asarray(
					action.detach().cpu().numpy(), dtype=np.float32,
				).reshape(-1)
				observation, reward, done, _ = env.step(action)
				actions.append(action_array)
				rewards.append(float(reward))
				dones.append(bool(done))
				if done and step_index + 1 != args.steps:
					raise RuntimeError(
						f'Environment terminated early after action {step_index}.'
					)

			arrays = {
				**{
					key: np.ascontiguousarray(np.stack(policy_lists[key], axis=0))
					for key in POLICY_INPUT_KEYS
				},
				'action': np.ascontiguousarray(np.stack(actions), dtype=np.float32),
				'reward': np.ascontiguousarray(rewards, dtype=np.float32),
				'done': np.ascontiguousarray(dones, dtype=np.bool_),
				'labels__state': np.ascontiguousarray(np.stack(states), dtype=np.float64),
				'labels__gt_role_mask': np.ascontiguousarray(
					np.stack(gt_masks), dtype=np.bool_,
				),
				'labels__gt_visible': np.ascontiguousarray(
					np.stack(gt_masks).reshape(args.steps + 1, len(roles), -1).any(axis=-1),
					dtype=np.bool_,
				),
				'episode_id': np.full(args.steps + 1, episode_index, dtype=np.int64),
				'step': np.arange(args.steps + 1, dtype=np.int64),
			}
			hashes = validate_episode_arrays(
				arrays, role_count=len(roles), state_dim=len(state_names),
				action_dim=int(cfg.action_dim), steps=args.steps,
				episode_index=episode_index,
			)
			relative = Path('episodes') / f'episode_{episode_index:04d}.npz'
			asset = temporary_root / relative
			asset_sha = _atomic_npz(asset, arrays)
			split = (
				'train' if episode_index < args.train_episodes else
				'validation' if episode_index < args.train_episodes + args.validation_episodes
				else 'test'
			)
			episode_records.append({
				'episode_index': episode_index,
				'episode_id': episode_index,
				'split': split,
				'steps': args.steps,
				'frames': args.steps + 1,
				'temporal_semantics': TEMPORAL_SEMANTICS,
				'planner_seed': planner_seed,
				'background_source': (
					'clean' if background_source is None else Path(background_source).name
				),
				'background_start_frame_index': (
					0 if background_start is None else int(background_start)
				),
				'path': relative.as_posix(),
				'sha256': asset_sha,
				**hashes,
			})
			print('ROF_CAUSAL_PROBE_EPISODE', json.dumps({
				'task': args.task, **episode_records[-1],
			}, allow_nan=False), flush=True)

		if state_schema is None or state_names is None:
			raise RuntimeError('No official state labels were collected.')
		splits = {
			split: [record['episode_index'] for record in episode_records if record['split'] == split]
			for split in ('train', 'validation', 'test')
		}
		manifest = {
			'format': FORMAT,
			'task': args.task,
			'condition': args.condition,
			'temporal_semantics': TEMPORAL_SEMANTICS,
			'policy_observation_keys': list(POLICY_INPUT_KEYS),
			'label_keys': list(LABEL_KEYS),
			'policy_input_contract': {
				'schema': 'robust_object_field_v0',
				'keys': list(POLICY_INPUT_KEYS),
				'privileged_labels_excluded': True,
				'hash_scope': 'exactly_policy_input_keys',
			},
			'role_names': list(roles),
			'role_count': len(roles),
			'action_dim': int(cfg.action_dim),
			'steps': args.steps,
			'frames_per_episode': args.steps + 1,
			'source': {
				'runtime_config': str(args.runtime_config.resolve()),
				'runtime_config_sha256': file_sha256(args.runtime_config),
				'checkpoint': str(args.checkpoint.resolve()),
				'checkpoint_sha256': file_sha256(args.checkpoint),
				'checkpoint_step': (
					args.expected_training_steps
					if args.checkpoint_step is None else args.checkpoint_step
				),
				'backend': 'robust_object_field',
				'policy_input_contract': 'robust_object_field_v0',
			},
			'collection': {
				'episodes': args.episodes,
				'condition': args.condition,
				'action_dim': int(cfg.action_dim),
				'max_steps': args.steps,
				'env_seed': args.env_seed,
				'background_seed': args.background_seed,
				'planner_seed_base': args.planner_seed_base,
				'eval_mode': True,
				'camera_id': camera_id,
				'gt_query_order': 'after_rof_observation_before_agent_action_same_state',
			},
			'label_contract': {
				'namespace': 'labels__',
				'keys': list(LABEL_KEYS),
				'role_names': list(roles),
				'labels_never_policy_input': True,
				'queried_after_policy_observation': True,
				'same_physics_state_guard': 'exact_before_after',
				'state_source': 'dm_control.task.get_observation(current_physics)',
				'state_schema': state_schema,
				'state_names': state_names,
				'gt_role_mask_source': 'same_state_mujoco_segmentation_policy_camera',
				'gt_role_mapping': 'task_static_collect_cutie_multitask_support_selectors',
				'visible_false_means_true_occlusion_not_missing_label': True,
			},
			'splits': splits,
			'episodes': episode_records,
			'perception_runtime': evaluator._metrics(env),
			'cutie_ready': evaluator._env_value(env, 'cutie_ready'),
		}
		manifest_path = temporary_root / MANIFEST_NAME
		manifest_path.write_bytes(canonical_json_bytes(manifest))
		validate_dataset(manifest_path)
		os.replace(temporary_root, args.output_root)
		return manifest
	finally:
		if env is not None and callable(getattr(env, 'close', None)):
			env.close()


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--task', choices=TASKS, required=True)
	parser.add_argument('--condition', choices=('clean', 'hard'), default='clean')
	parser.add_argument('--training-condition', choices=('clean', 'hard'), default='clean')
	parser.add_argument('--runtime-config', type=Path, required=True)
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--checkpoint-step', type=int)
	parser.add_argument('--training-seed', type=int, default=6)
	parser.add_argument('--expected-training-steps', type=int, default=100000)
	parser.add_argument('--expected-training-eval-freq', type=int, default=20000)
	parser.add_argument('--expected-training-eval-episodes', type=int, default=3)
	parser.add_argument('--episodes', type=int, default=20)
	parser.add_argument('--steps', type=int, default=500)
	parser.add_argument('--train-episodes', type=int, default=12)
	parser.add_argument('--validation-episodes', type=int, default=4)
	parser.add_argument('--env-seed', type=int, default=424243)
	parser.add_argument('--background-seed', type=int, default=1618034)
	parser.add_argument('--planner-seed-base', type=int, default=8675400)
	parser.add_argument('--output-root', type=Path, required=True)
	return parser.parse_args(argv)


def main(argv=None):
	args = parse_args(argv)
	manifest = collect(args)
	print('ROF_CAUSAL_PROBE_DATASET_COMPLETE', json.dumps({
		'task': manifest['task'],
		'condition': manifest['condition'],
		'episodes': len(manifest['episodes']),
		'manifest': str((args.output_root / MANIFEST_NAME).resolve()),
	}, allow_nan=False), flush=True)
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
