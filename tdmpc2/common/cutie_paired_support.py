"""Pure schema helpers for paired native-resolution Cutie support.

The paired pack is diagnostic support data, not a training observation.  One
accepted simulator state owns both the 64 and 128 render/mask records.  The
identity below intentionally covers decoded asset hashes and the state/action
anchors rather than filesystem paths, so moving a complete pack does not
change its scientific identity.
"""

from __future__ import annotations

import hashlib
import json
import re


FORMAT = 'cutie_paired_native_support_v2'
SUPPORT_SCHEMA = 'generic_indexed_paired_native_v2'
RESOLUTIONS = (64, 128)
SUPPORT_RECORDS = 6
PAIRING = 'single_process_same_accepted_state_no_step_between_renders'
RGB_GENERATION = (
	'native_dmc_foreground_per_resolution_with_current_native64_'
	'video_background_resized_without_clock_advance'
)
MASK_GENERATION = 'native_mujoco_segmentation_per_resolution_no_resize'
_SHA256 = re.compile(r'[0-9a-f]{64}')


def canonical_json_bytes(value) -> bytes:
	return (
		json.dumps(
			value,
			ensure_ascii=False,
			indent=2,
			sort_keys=True,
			allow_nan=False,
		)
		+ '\n'
	).encode('utf-8')


def sha256_json(value) -> str:
	return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_sha256(value, label: str) -> str:
	if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
		raise ValueError(f'{label} must be one lowercase SHA-256 digest.')
	return value


def _strict_int(value, label: str) -> int:
	if type(value) is not int:
		raise ValueError(f'{label} must be an integer.')
	return value


def _record_identity(record: dict) -> dict:
	resolutions = record.get('resolutions')
	if not isinstance(resolutions, dict):
		raise ValueError('Paired support record resolutions must be an object.')
	assets = {}
	for resolution in RESOLUTIONS:
		entry = resolutions.get(str(resolution))
		if not isinstance(entry, dict):
			raise ValueError(
				f'Paired support record lacks resolution {resolution}.'
			)
		assets[str(resolution)] = {
			'image_sha256': require_sha256(
				entry.get('image_sha256'), f'resolution {resolution} image_sha256'
			),
			'indexed_mask_sha256': require_sha256(
				entry.get('indexed_mask_sha256'),
				f'resolution {resolution} indexed_mask_sha256',
			),
			'native_clean_rgb_sha256': require_sha256(
				entry.get('native_clean_rgb_sha256'),
				f'resolution {resolution} native_clean_rgb_sha256',
			),
		}
	state = record.get('physics_state')
	actions = record.get('actions')
	guard = record.get('same_state_guard')
	evidence = record.get('cross_resolution_evidence')
	if not all(isinstance(value, dict) for value in (state, actions, guard, evidence)):
		raise ValueError('Paired support state/actions/guard must be objects.')
	for name in (
		'clean128_distinct_from_bilinear64',
		'composed128_distinct_from_bilinear64',
		'mask128_distinct_from_nearest64',
	):
		if evidence.get(name) is not True:
			raise ValueError(f'Paired support lacks native-resolution evidence {name}.')
	index = _strict_int(record.get('index'), 'record.index')
	accepted = _strict_int(
		record.get('accepted_state_ordinal'), 'record.accepted_state_ordinal'
	)
	reset = _strict_int(record.get('reset_ordinal'), 'record.reset_ordinal')
	prefix = _strict_int(record.get('random_prefix_steps'), 'record.random_prefix_steps')
	if index < 0 or accepted != index or reset < 1 or prefix < 1:
		raise ValueError('Paired support record ordinals are invalid.')
	if any(guard.get(name) is not True for name in (
		'pass', 'physics_exact', 'background_exact', 'rng_exact'
	)):
		raise ValueError('Paired support same-state guard did not pass exactly.')
	physics_before = require_sha256(
		guard.get('physics_before_sha256'),
		'same_state_guard.physics_before_sha256',
	)
	physics_after = require_sha256(
		guard.get('physics_after_sha256'),
		'same_state_guard.physics_after_sha256',
	)
	if physics_before != physics_after or physics_before != state.get('sha256'):
		raise ValueError('Paired support physics guard hashes disagree.')
	background_before = require_sha256(
		guard.get('background_before_sha256'),
		'same_state_guard.background_before_sha256',
	)
	background_after = require_sha256(
		guard.get('background_after_sha256'),
		'same_state_guard.background_after_sha256',
	)
	rng_before = require_sha256(
		guard.get('rng_before_sha256'), 'same_state_guard.rng_before_sha256'
	)
	rng_after = require_sha256(
		guard.get('rng_after_sha256'), 'same_state_guard.rng_after_sha256'
	)
	if background_before != background_after or rng_before != rng_after:
		raise ValueError('Paired support background/RNG guard hashes disagree.')
	background_state = guard.get('background_state')
	rng_domains = guard.get('rng_domains')
	if not isinstance(background_state, dict) or set(background_state) != {
		'active_source', 'active_source_name', 'start_index', 'frame_clock',
		'last_frame_index', 'internal_random_state_sha256',
		'current_native64_background_sha256',
		'current_native64_background_shape', 'source_cache_order',
		'policy_stack_sha256', 'last_composed_sha256',
	}:
		raise ValueError('Paired support background-state snapshot is malformed.')
	if not isinstance(rng_domains, dict) or set(rng_domains) != {
		'python_global', 'numpy_global', 'action_generator',
		'background_random', 'environment_and_task_rngs', 'torch_cpu',
		'torch_cuda',
	}:
		raise ValueError('Paired support RNG-domain snapshot is malformed.')
	if sha256_json(background_state) != background_before:
		raise ValueError('Paired support background snapshot/hash disagree.')
	if sha256_json(rng_domains) != rng_before:
		raise ValueError('Paired support RNG snapshot/hash disagree.')
	for name in (
		'internal_random_state_sha256',
		'current_native64_background_sha256', 'policy_stack_sha256',
		'last_composed_sha256',
	):
		require_sha256(background_state.get(name), f'background_state.{name}')
	if background_state.get('current_native64_background_shape') != [64, 64, 3]:
		raise ValueError('Paired support current background shape changed.')
	for name in ('start_index', 'frame_clock', 'last_frame_index'):
		if type(background_state.get(name)) is not int or background_state[name] < 0:
			raise ValueError(f'background_state.{name} must be non-negative.')
	active_source = background_state.get('active_source')
	if not isinstance(active_source, str) or not active_source:
		raise ValueError('Paired support active background source is invalid.')
	source_cache_order = background_state.get('source_cache_order')
	if (
		not isinstance(source_cache_order, list)
		or not source_cache_order
		or any(not isinstance(value, str) or not value for value in source_cache_order)
	):
		raise ValueError('Paired support background cache order must be a list.')
	for name in (
		'python_global', 'numpy_global', 'action_generator',
		'background_random', 'torch_cpu',
	):
		require_sha256(rng_domains.get(name), f'rng_domains.{name}')
	for name in ('environment_and_task_rngs', 'torch_cuda'):
		values = rng_domains.get(name)
		if not isinstance(values, list):
			raise ValueError(f'rng_domains.{name} must be a list.')
		for value_index, value in enumerate(values):
			require_sha256(value, f'rng_domains.{name}[{value_index}]')
	active_video = record.get('active_video')
	frame_index = _strict_int(
		record.get('background_frame_index'), 'record.background_frame_index'
	)
	if (
		not isinstance(active_video, str)
		or not active_video.startswith('video')
		or not active_video.endswith('.mp4')
		or frame_index < 0
	):
		raise ValueError('Paired support background provenance is invalid.')
	if (
		background_state.get('active_source_name') != active_video
		or background_state.get('last_frame_index') != frame_index
		or active_source.replace('\\', '/').rsplit('/', 1)[-1] != active_video
	):
		raise ValueError('Paired support record/background snapshot disagree.')
	return {
		'index': index,
		'accepted_state_ordinal': accepted,
		'reset_ordinal': reset,
		'random_prefix_steps': prefix,
		'active_video': active_video,
		'background_frame_index': frame_index,
		'physics_state_sha256': physics_before,
		'action_sequence_sha256': require_sha256(
			actions.get('sha256'), 'actions.sha256'
		),
		'same_state_guard_pass': True,
		'background_state_before_sha256': background_before,
		'background_state_after_sha256': background_after,
		'rng_state_before_sha256': rng_before,
		'rng_state_after_sha256': rng_after,
		'cross_resolution_evidence': {
			name: True for name in (
				'clean128_distinct_from_bilinear64',
				'composed128_distinct_from_bilinear64',
				'mask128_distinct_from_nearest64',
			)
		},
		'resolutions': assets,
	}


def paired_support_identity_payload(payload: dict) -> dict:
	if not isinstance(payload, dict) or payload.get('format') != FORMAT:
		raise ValueError(f'Paired support format must be {FORMAT!r}.')
	collection = payload.get('collection')
	roles = payload.get('roles')
	records = payload.get('records')
	if not isinstance(collection, dict):
		raise ValueError('Paired support collection must be an object.')
	if (
		not isinstance(roles, list)
		or len(roles) != 2
		or len(set(roles)) != 2
		or any(not isinstance(role, str) or not role.strip() for role in roles)
	):
		raise ValueError('Paired support must declare exactly two ordered roles.')
	if not isinstance(records, list) or len(records) != SUPPORT_RECORDS:
		raise ValueError(
			f'Paired support must contain exactly {SUPPORT_RECORDS} records.'
		)
	if collection.get('support_schema') != SUPPORT_SCHEMA:
		raise ValueError(f'Paired support schema must be {SUPPORT_SCHEMA!r}.')
	if not isinstance(collection.get('task'), str) or not collection['task'].strip():
		raise ValueError('Paired support task must be a non-empty string.')
	if collection.get('split') != 'support':
		raise ValueError('Paired support split must be support.')
	seeds = [
		_strict_int(collection.get(name), f'collection.{name}')
		for name in ('environment_seed', 'background_seed', 'action_seed')
	]
	if len(set(seeds)) != 3:
		raise ValueError('Paired support seed domains must be distinct.')
	if _strict_int(collection.get('camera_id'), 'collection.camera_id') != 0:
		raise ValueError('Paired support camera_id must be zero.')
	if collection.get('resolutions') != list(RESOLUTIONS):
		raise ValueError(f'Paired support resolutions must be {list(RESOLUTIONS)}.')
	if collection.get('pairing') != PAIRING:
		raise ValueError('Paired support same-state pairing contract changed.')
	if collection.get('rgb_generation') != RGB_GENERATION:
		raise ValueError('Paired support RGB generation contract changed.')
	if collection.get('mask_generation') != MASK_GENERATION:
		raise ValueError('Paired support mask generation contract changed.')
	manifest_sha = require_sha256(
		collection.get('manifest_sha256'), 'collection.manifest_sha256'
	)
	combined_manifest_sha = require_sha256(
		collection.get('combined_manifest_sha256'),
		'collection.combined_manifest_sha256',
	)
	geom_catalog_sha = require_sha256(
		collection.get('geom_catalog_sha256'), 'collection.geom_catalog_sha256'
	)
	for expected_index, record in enumerate(records):
		if not isinstance(record, dict) or record.get('index') != expected_index:
			raise ValueError('Paired support records must be contiguous and ordered.')
	return {
		'format': FORMAT,
		'roles': list(roles),
		'collection': {
			'support_schema': collection.get('support_schema'),
			'task': collection.get('task'),
			'split': collection.get('split'),
			'environment_seed': seeds[0],
			'background_seed': seeds[1],
			'action_seed': seeds[2],
			'camera_id': 0,
			'resolutions': list(RESOLUTIONS),
			'manifest_sha256': manifest_sha,
			'combined_manifest_sha256': combined_manifest_sha,
			'pairing': PAIRING,
			'rgb_generation': RGB_GENERATION,
			'mask_generation': MASK_GENERATION,
			'geom_catalog_sha256': geom_catalog_sha,
		},
		'records': [_record_identity(record) for record in records],
	}


def compute_paired_support_id(payload: dict) -> str:
	return sha256_json(paired_support_identity_payload(payload))
