"""Collect one same-state native64/native128 Cutie support pack.

Every accepted simulator state is rendered at both resolutions before another
environment action/reset is allowed. RGB foregrounds and indexed segmentation
masks are rendered independently at each resolution; neither is resized from
the 64 support asset. The dynamic-video background intentionally matches the
runtime diagnostic: the already-selected native64 background frame is replayed
without advancing its clock and resized for the 128 foreground render. The
supported task roles and MuJoCo object selectors are imported from the
canonical multitask support collector rather than duplicated here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch


os.environ.setdefault('MUJOCO_GL', 'egl')
PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
for path in (str(REPO_DIR), str(PROJECT_DIR)):
	while path in sys.path:
		sys.path.remove(path)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))

import envs.dmcontrol as dmcontrol_env  # noqa: E402
from tdmpc2.common.cutie_paired_support import (  # noqa: E402
	FORMAT,
	MASK_GENERATION,
	PAIRING,
	RESOLUTIONS,
	RGB_GENERATION,
	SUPPORT_RECORDS,
	SUPPORT_SCHEMA,
	canonical_json_bytes,
	compute_paired_support_id,
	sha256_json,
)
from tdmpc2.tools.collect_cutie_multitask_support import (  # noqa: E402
	SUPPORT_VIDEOS,
	TASK_BY_NAME,
	RoleVisibilityError,
	_catalog,
	_named_selection,
	_selected_objects,
	_segmentation_constants,
	_selector_payload,
)


TASKS = (
	'reacher-visual-small',
	'cartpole-swingup',
	'acrobot-swingup',
)
SPLIT = 'support'
# This paired-native protocol is deliberately restricted to the three TASKS
# above, whose policy renderer is camera 0. Do not inherit a generic collector
# constant: the generic collector now derives each task's actual runtime view.
CAMERA_ID = 0


class _Config(SimpleNamespace):
	def get(self, name, default=None):
		return getattr(self, name, default)


def _decoded_sha256(value: np.ndarray) -> str:
	return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _typed_array_sha256(value: np.ndarray) -> str:
	array = np.ascontiguousarray(value)
	digest = hashlib.sha256()
	digest.update(str(array.dtype).encode('ascii'))
	digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
	digest.update(array.tobytes())
	return digest.hexdigest()


def _file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as handle:
		for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def _pickle_sha256(value) -> str:
	return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _write_json(path: Path, payload) -> None:
	path.write_bytes(canonical_json_bytes(payload))


def _find_wrapper(env, wrapper_type):
	current, seen = env, set()
	while current is not None and id(current) not in seen:
		seen.add(id(current))
		if isinstance(current, wrapper_type):
			return current
		current = getattr(current, 'env', None)
	raise RuntimeError(f'{wrapper_type.__name__} is missing from the environment chain.')


def _find_physics(env):
	current, seen = env, set()
	while current is not None and id(current) not in seen:
		seen.add(id(current))
		physics = getattr(current, 'physics', None)
		if physics is not None and callable(getattr(physics, 'render', None)):
			return physics
		current = getattr(current, 'env', None)
	raise RuntimeError('Could not find dm_control physics in the environment chain.')


def _latest_policy_rgb(observation) -> np.ndarray:
	try:
		value = observation[-3:].detach().cpu().permute(1, 2, 0).contiguous().numpy()
	except AttributeError:
		value = np.moveaxis(np.asarray(observation)[-3:], 0, -1)
	if value.shape != (64, 64, 3) or value.dtype != np.uint8:
		raise RuntimeError(f'Expected uint8 policy RGB [64,64,3], got {value.shape}.')
	return np.array(value, dtype=np.uint8, order='C', copy=True)


def _indexed_mask(physics, resolution: int, selections, role_names) -> np.ndarray:
	segmentation = np.asarray(physics.render(
		height=resolution,
		width=resolution,
		camera_id=CAMERA_ID,
		segmentation=True,
	))
	if segmentation.shape != (resolution, resolution, 2):
		raise RuntimeError(
			f'Native segmentation must be [{resolution},{resolution},2], got '
			f'{segmentation.shape}.'
		)
	constants = _segmentation_constants()
	output = np.zeros((resolution, resolution), dtype=np.uint8)
	for role_id, (role_name, selected) in enumerate(
		zip(role_names, selections), start=1
	):
		role_mask = np.zeros_like(output, dtype=np.bool_)
		for object_type, object_id in selected:
			role_mask |= (
				(segmentation[..., 0] == int(object_id))
				& (segmentation[..., 1] == constants[object_type])
			)
		if not bool(role_mask.any()):
			raise RoleVisibilityError(role_id, role_name)
		if bool(np.any(output[role_mask])):
			raise RuntimeError('Native paired role masks overlap.')
		output[role_mask] = role_id
	return np.ascontiguousarray(output)


def _torch_stack_sha256(values) -> str:
	digest = hashlib.sha256()
	for value in values:
		array = value.detach().cpu().contiguous().numpy()
		digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
		digest.update(array.tobytes())
	return digest.hexdigest()


def _background_snapshot(background) -> dict:
	compositor = background.compositor
	last_index = compositor.frame_index
	if compositor._frames is None or last_index is None:
		raise RuntimeError('Paired render requires one current background frame.')
	current = np.asarray(compositor._frames[int(last_index)])
	return {
		'active_source': str(Path(background.active_source).resolve()),
		'active_source_name': Path(background.active_source).name,
		'start_index': int(compositor._start),
		'frame_clock': int(compositor._frame),
		'last_frame_index': int(last_index),
		'internal_random_state_sha256': _pickle_sha256(
			compositor._random.get_state()
		),
		'current_native64_background_sha256': _decoded_sha256(current),
		'current_native64_background_shape': list(current.shape),
		'source_cache_order': [str(Path(path).resolve()) for path in compositor._source_cache],
		'policy_stack_sha256': _torch_stack_sha256(tuple(background._frames)),
		'last_composed_sha256': _decoded_sha256(np.asarray(background._last_frame)),
	}


def _environment_rng_hashes(env) -> list[str]:
	"""Hash task/environment RNGs reachable through the wrapper chain."""
	queue, seen, hashes = [env], set(), []
	while queue:
		current = queue.pop(0)
		if current is None or id(current) in seen:
			continue
		seen.add(id(current))
		for owner in (current, getattr(current, '_task', None), getattr(current, 'task', None)):
			if owner is None:
				continue
			for name in ('_random', 'random'):
				rng = getattr(owner, name, None)
				if callable(getattr(rng, 'get_state', None)):
					hashes.append(_pickle_sha256(rng.get_state()))
				elif getattr(rng, 'bit_generator', None) is not None:
					hashes.append(_pickle_sha256(
						copy.deepcopy(rng.bit_generator.state)
					))
		for name in ('env', '_env'):
			child = getattr(current, name, None)
			if child is not None:
				queue.append(child)
	return sorted(set(hashes))


def _rng_snapshot(action_rng, background, env) -> dict:
	result = {
		'python_global': _pickle_sha256(random.getstate()),
		'numpy_global': _pickle_sha256(np.random.get_state()),
		'action_generator': _pickle_sha256(
			copy.deepcopy(action_rng.bit_generator.state)
		),
		'background_random': _pickle_sha256(
			background.compositor._random.get_state()
		),
		'environment_and_task_rngs': _environment_rng_hashes(env),
		'torch_cpu': hashlib.sha256(
			torch.get_rng_state().cpu().numpy().tobytes()
		).hexdigest(),
	}
	if torch.cuda.is_initialized():
		result['torch_cuda'] = [
			hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
			for state in torch.cuda.get_rng_state_all()
		]
	else:
		result['torch_cuda'] = []
	return result


def _array_payload(value: np.ndarray) -> dict:
	array = np.ascontiguousarray(value)
	return {
		'dtype': str(array.dtype),
		'shape': list(array.shape),
		'values': array.tolist(),
		'sha256': _typed_array_sha256(array),
	}


def _same_state_pair(env, background, physics, action_rng, selections, roles):
	physics_before = np.array(physics.get_state(), copy=True)
	background_before = _background_snapshot(background)
	rng_before = _rng_snapshot(action_rng, background, env)
	assets = {}
	visibility_error = None
	try:
		for resolution in RESOLUTIONS:
			render = getattr(background.env, 'cutie_same_state_rgb', None)
			if not callable(render):
				raise RuntimeError('Background child lacks same-state native RGB render.')
			clean = np.asarray(render(height=resolution, width=resolution))
			if clean.shape != (resolution, resolution, 3) or clean.dtype != np.uint8:
				raise RuntimeError(f'Bad native clean RGB at {resolution}: {clean.shape}.')
			image = np.asarray(background.compositor.apply_current(clean))
			try:
				mask = _indexed_mask(physics, resolution, selections, roles)
			except RoleVisibilityError as exc:
				visibility_error = exc
				mask = None
			assets[resolution] = {
				'clean': np.array(clean, dtype=np.uint8, order='C', copy=True),
				'image': np.array(image, dtype=np.uint8, order='C', copy=True),
				'mask': mask,
			}
	finally:
		physics_after = np.array(physics.get_state(), copy=True)
		background_after = _background_snapshot(background)
		rng_after = _rng_snapshot(action_rng, background, env)
	if not np.array_equal(physics_before, physics_after):
		raise RuntimeError('Native support renders mutated the physics state.')
	if background_before != background_after:
		raise RuntimeError('Native support renders mutated background state/clock.')
	if rng_before != rng_after:
		raise RuntimeError('Native support renders mutated an audited RNG domain.')
	guard = {
		'pass': True,
		'physics_before_sha256': _typed_array_sha256(physics_before),
		'physics_after_sha256': _typed_array_sha256(physics_after),
		'physics_exact': True,
		'background_state': background_before,
		'background_before_sha256': sha256_json(background_before),
		'background_after_sha256': sha256_json(background_after),
		'background_exact': True,
		'rng_domains': rng_before,
		'rng_before_sha256': sha256_json(rng_before),
		'rng_after_sha256': sha256_json(rng_after),
		'rng_exact': True,
	}
	return physics_before, assets, guard, visibility_error


def _config(args):
	return _Config(
		task=args.task,
		obs='rgb',
		seed=args.seed,
		multitask=False,
		video_background_enabled=True,
		video_background_root=str(args.video_root.expanduser().resolve()),
		video_background_manifest_dir=(
			None if args.manifest_dir is None
			else str(args.manifest_dir.expanduser().resolve())
		),
		video_background_split=SPLIT,
		video_background_strength=1.0,
		video_background_total_frames=1000,
		video_background_source_cache_size=8,
		video_background_seed=args.background_seed,
		flat_anchor=False,
	)


def _collect(args, stage: Path) -> dict:
	spec = TASK_BY_NAME[args.task]
	for resolution in RESOLUTIONS:
		(stage / f'support_{resolution}' / 'frames').mkdir(parents=True)
		(stage / f'support_{resolution}' / 'indexed_masks').mkdir(parents=True)
	env = dmcontrol_env.make_env(_config(args))
	try:
		background = _find_wrapper(
			env, dmcontrol_env.ColorMultiVideoBackgroundWrapper
		)
		physics = _find_physics(env)
		if background.active_split != SPLIT:
			raise RuntimeError(f'Background split must be {SPLIT!r}.')
		if tuple(background.source_names) != SUPPORT_VIDEOS:
			raise RuntimeError('Support video manifest changed.')
		catalog = _catalog(physics)
		selections = tuple(
			_selected_objects(catalog, selector) for selector in spec.selectors
		)
		if set(selections[0]).intersection(selections[1]):
			raise ValueError(
				f'{spec.task} role selectors overlap in the model catalog.'
			)
		matched = {
			role: _named_selection(catalog, selected)
			for role, selected in zip(spec.roles, selections)
		}
		catalog_payload = {
			'task': args.task,
			'camera_id': CAMERA_ID,
			'segmentation_object_types': _segmentation_constants(),
			'objects': catalog,
			'matched_objects': matched,
			'role_selectors': {
				role: _selector_payload(selector)
				for role, selector in zip(spec.roles, spec.selectors)
			},
		}
		catalog_path = stage / 'geom_catalog.json'
		_write_json(catalog_path, catalog_payload)

		action_rng = np.random.default_rng(args.action_seed)
		records, covered = [], set()
		reset_ordinal = 0
		visibility_rejections = {
			f'{resolution}/{role}': 0
			for resolution in RESOLUTIONS for role in spec.roles
		}
		while len(records) < SUPPORT_RECORDS:
			if reset_ordinal >= args.max_reset_attempts:
				raise RuntimeError(
					f'Could not collect {SUPPORT_RECORDS} paired states after '
					f'{reset_ordinal} resets.'
				)
			observation = env.reset()
			reset_ordinal += 1
			active_video = Path(background.active_source).name
			if active_video not in SUPPORT_VIDEOS:
				raise RuntimeError(f'Out-of-split support source {active_video!r}.')
			if len(covered) < len(SUPPORT_VIDEOS) and active_video in covered:
				continue
			index = len(records)
			prefix_steps = 4 + 3 * index
			actions = []
			for _ in range(prefix_steps):
				action = action_rng.uniform(
					env.action_space.low, env.action_space.high
				).astype(env.action_space.dtype)
				actions.append(np.array(action, copy=True))
				observation = env.step(action)[0]
			policy64 = _latest_policy_rgb(observation)
			state, assets, guard, visibility = _same_state_pair(
				env, background, physics, action_rng, selections, spec.roles
			)
			if visibility is not None:
				for resolution, entry in assets.items():
					if entry['mask'] is None:
						visibility_rejections[
							f'{resolution}/{visibility.role_name}'
						] += 1
				continue
			if not np.array_equal(policy64, assets[64]['image']):
				raise RuntimeError(
					'Paired native64 render is not byte-identical to the current '
					'policy/background observation.'
				)
			resampling = getattr(Image, 'Resampling', Image)
			clean64_up = np.asarray(Image.fromarray(
				assets[64]['clean'], mode='RGB'
			).resize((128, 128), resample=resampling.BILINEAR))
			image64_up = np.asarray(Image.fromarray(
				assets[64]['image'], mode='RGB'
			).resize((128, 128), resample=resampling.BILINEAR))
			mask64_up = np.asarray(Image.fromarray(
				assets[64]['mask'], mode='L'
			).resize((128, 128), resample=resampling.NEAREST))
			evidence = {
				'clean128_distinct_from_bilinear64': not np.array_equal(
					assets[128]['clean'], clean64_up
				),
				'composed128_distinct_from_bilinear64': not np.array_equal(
					assets[128]['image'], image64_up
				),
				'mask128_distinct_from_nearest64': not np.array_equal(
					assets[128]['mask'], mask64_up
				),
			}
			if not all(evidence.values()):
				raise RuntimeError(
					'Paired assets do not empirically distinguish native128 from '
					'a resized 64 support asset.'
				)
			action_array = np.stack(actions, axis=0)
			record = {
				'index': index,
				'accepted_state_ordinal': index,
				'reset_ordinal': reset_ordinal,
				'random_prefix_steps': prefix_steps,
				'active_video': active_video,
				'background_frame_index': int(background.frame_index),
				'selected_names': matched,
				'physics_state': _array_payload(state),
				'actions': _array_payload(action_array),
				'same_state_guard': guard,
				'cross_resolution_evidence': evidence,
				'resolutions': {},
			}
			for resolution in RESOLUTIONS:
				entry = assets[resolution]
				name = f'support_{index:02d}.png'
				image_rel = f'support_{resolution}/frames/{name}'
				mask_rel = f'support_{resolution}/indexed_masks/{name}'
				Image.fromarray(entry['image'], mode='RGB').save(stage / image_rel)
				Image.fromarray(entry['mask'], mode='L').save(stage / mask_rel)
				record['resolutions'][str(resolution)] = {
					'image': image_rel,
					'image_sha256': _decoded_sha256(entry['image']),
					'indexed_mask': mask_rel,
					'indexed_mask_sha256': _decoded_sha256(entry['mask']),
					'native_clean_rgb_sha256': _decoded_sha256(entry['clean']),
					'role_pixel_counts': {
						role: int((entry['mask'] == role_id).sum())
						for role_id, role in enumerate(spec.roles, start=1)
					},
				}
			records.append(record)
			covered.add(active_video)
		if covered != set(SUPPORT_VIDEOS):
			raise RuntimeError(f'Support source coverage incomplete: {sorted(covered)}.')

		payload = {
			'format': FORMAT,
			'roles': list(spec.roles),
			'collection': {
				'support_schema': SUPPORT_SCHEMA,
				'task': args.task,
				'observation': 'rgb',
				'split': SPLIT,
				'environment_seed': args.seed,
				'background_seed': args.background_seed,
				'action_seed': args.action_seed,
				'episodes': SUPPORT_RECORDS,
				'camera_id': CAMERA_ID,
				'resolutions': list(RESOLUTIONS),
				'label_policy': 'simulator_segmentation_support_only',
				'diagnostic_support': True,
				'pairing': PAIRING,
				'rgb_generation': RGB_GENERATION,
				'mask_generation': MASK_GENERATION,
				'full_composed_rgb_native_high_resolution': False,
				'native_foreground_and_mask_not_resized_from_64': True,
				'allowed_videos': list(SUPPORT_VIDEOS),
				'covered_videos': sorted(covered),
				'reset_attempts': reset_ordinal,
				'visibility_rejections': visibility_rejections,
				'manifest_sha256': background.manifest_sha256,
				'combined_manifest_sha256': background.combined_manifest_sha256,
				'geom_catalog': catalog_path.name,
				'geom_catalog_sha256': _file_sha256(catalog_path),
				'collector': Path(__file__).name,
				'collector_sha256': _file_sha256(Path(__file__).resolve()),
			},
			'records': records,
		}
		payload['collection']['paired_support_id'] = compute_paired_support_id(payload)
		_write_json(stage / 'annotations.json', payload)
		return payload
	finally:
		close = getattr(env, 'close', None)
		if callable(close):
			close()


def _parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--task', choices=TASKS, required=True)
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--video-root', type=Path, required=True)
	parser.add_argument('--manifest-dir', type=Path)
	parser.add_argument('--seed', type=int, default=314159)
	parser.add_argument('--background-seed', type=int, default=314160)
	parser.add_argument('--action-seed', type=int, default=314161)
	parser.add_argument('--max-reset-attempts', type=int, default=1000)
	args = parser.parse_args(argv)
	if len({args.seed, args.background_seed, args.action_seed}) != 3:
		parser.error('environment/background/action seeds must be distinct')
	if args.max_reset_attempts < SUPPORT_RECORDS:
		parser.error(f'--max-reset-attempts must be at least {SUPPORT_RECORDS}')
	if not args.video_root.expanduser().is_dir():
		parser.error(f'--video-root is not a directory: {args.video_root}')
	return args


def main(argv=None) -> int:
	args = _parse_args(argv)
	output = args.output.expanduser().resolve()
	if output.exists():
		raise FileExistsError(f'Refusing to overwrite paired support: {output}')
	stage = output.with_name(f'{output.name}.incomplete.{os.getpid()}')
	if stage.exists():
		raise FileExistsError(stage)
	stage.mkdir(parents=True)
	try:
		payload = _collect(args, stage)
		stage.replace(output)
	except Exception:
		failed = output.with_name(f'{output.name}.failed.{os.getpid()}')
		if stage.exists() and not failed.exists():
			stage.replace(failed)
		raise
	print('CUTIE_PAIRED_NATIVE_SUPPORT_OK', json.dumps({
		'task': args.task,
		'paired_support_id': payload['collection']['paired_support_id'],
		'output': str(output),
	}, sort_keys=True), flush=True)
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
