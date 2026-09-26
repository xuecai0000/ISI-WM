"""Dependency-light sensor schema for Robust Object Field V0.

The model-facing packet is exact-K and contains four arrays only:

``rgb``
	The unchanged causal three-frame RGB stack, ``uint8 [9,64,64]``.
``object``
	The unchanged causal three-frame Cutie descriptor stack,
	``float32 [K,1770]``. Its final four values per frame are the tracker status.
``object_mask``
	The native masks produced alongside those descriptors,
	``bool [K,3,64,64]``.
``role_exists``
	An all-one ``float32 [K]`` exact-role assertion. V0 has no padding roles.

Runtime simulator state, kinematics, and segmentation are outside this schema.
The role names are task-static metadata, never inferred or silently reordered.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np


SCHEMA = 'robust_object_field_v0'
IMAGE_SIZE = 64
STACK_FRAMES = 3
RGB_CHANNELS = 3 * STACK_FRAMES
FRAME_DIM = 590
OBJECT_DIM = FRAME_DIM * STACK_FRAMES
STATUS_START = 586
STATUS_NAMES = ('confidence', 'lost', 'tracker_valid', 'mask_score')
DIAGNOSTIC_FIELDS = frozenset({
	'valid',
	'lost',
	'mask_nonempty',
	'feature_finite',
	'mask_area_pixels',
	'mask_touches_border',
	'confidence',
	'mask_score',
	'runtime_ms',
})


class RobustObjectFieldObservationError(ValueError):
	"""The atomic ROF V0 observation contract was violated."""


def _readonly_copy(value, *, dtype) -> np.ndarray:
	result = np.array(value, dtype=dtype, order='C', copy=True)
	result.setflags(write=False)
	return result


def _roles(value: Sequence[str]) -> tuple[str, ...]:
	roles = tuple(value)
	if (
		not roles
		or len(set(roles)) != len(roles)
		or any(not isinstance(role, str) or not role.strip() for role in roles)
	):
		raise RobustObjectFieldObservationError(
			f'ROF V0 requires unique non-empty task-static roles, got {roles!r}.'
		)
	return roles


def _vector(diagnostics, name, *, count, dtype) -> np.ndarray:
	value = np.asarray(diagnostics[name])
	if value.shape != (count,):
		raise RobustObjectFieldObservationError(
			f'ROF V0 diagnostic {name!r} must have shape [{count}], got '
			f'{value.shape}.'
		)
	if dtype is np.bool_:
		if value.dtype != np.bool_:
			raise RobustObjectFieldObservationError(
				f'ROF V0 diagnostic {name!r} must be bool, got {value.dtype}.'
			)
	elif dtype is np.int64:
		if not np.issubdtype(value.dtype, np.integer):
			raise RobustObjectFieldObservationError(
				f'ROF V0 diagnostic {name!r} must be integer, got {value.dtype}.'
			)
	elif not np.issubdtype(value.dtype, np.floating):
		raise RobustObjectFieldObservationError(
			f'ROF V0 diagnostic {name!r} must be floating, got {value.dtype}.'
		)
	if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
		raise RobustObjectFieldObservationError(
			f'ROF V0 diagnostic {name!r} must be finite.'
		)
	return np.asarray(value, dtype=dtype)


def _digest(value) -> str:
	return hashlib.sha256(
		np.ascontiguousarray(value).tobytes(order='C')
	).hexdigest()


@dataclass(frozen=True)
class RobustObjectFieldPacket:
	"""One immutable exact-K, three-frame causal observation packet."""

	task: str
	role_names: tuple[str, ...]
	rgb: np.ndarray
	object: np.ndarray
	object_mask: np.ndarray
	role_exists: np.ndarray
	sequence_id: int
	source_rgb_sha256: str
	object_sha256: str
	object_mask_sha256: str
	binding_sha256: str

	@property
	def role_count(self) -> int:
		return len(self.role_names)

	def as_numpy_observation(self) -> dict[str, np.ndarray]:
		"""Return writable copies suitable for Gym and replay storage."""
		return {
			'rgb': np.array(self.rgb, dtype=np.uint8, order='C', copy=True),
			'object': np.array(
				self.object, dtype=np.float32, order='C', copy=True
			),
			'object_mask': np.array(
				self.object_mask, dtype=np.bool_, order='C', copy=True
			),
			'role_exists': np.array(
				self.role_exists, dtype=np.float32, order='C', copy=True
			),
		}


def build_packet(
	*,
	task: str,
	role_names: Sequence[str],
	rgb_stack,
	object_stack,
	object_mask_stack,
	diagnostics: Mapping[str, object],
	sequence_id: int,
) -> RobustObjectFieldPacket:
	"""Validate and seal one synchronous Cutie observation.

	Nothing is padded, repaired, clipped, carried from history, or obtained from
	the simulator. The current mask, descriptor status, and duplicated worker
	diagnostics must agree exactly enough to prove they are the same response.
	"""
	if not isinstance(task, str) or not task.strip():
		raise RobustObjectFieldObservationError('ROF V0 task must be non-empty.')
	roles = _roles(role_names)
	count = len(roles)
	if not isinstance(sequence_id, int) or isinstance(sequence_id, bool) or sequence_id < 0:
		raise RobustObjectFieldObservationError(
			'ROF V0 sequence_id must be a non-negative integer.'
		)

	rgb = np.asarray(rgb_stack)
	objects = np.asarray(object_stack)
	masks = np.asarray(object_mask_stack)
	if rgb.shape != (RGB_CHANNELS, IMAGE_SIZE, IMAGE_SIZE) or rgb.dtype != np.uint8:
		raise RobustObjectFieldObservationError(
			f'ROF V0 rgb must be uint8 [9,64,64], got {rgb.shape} {rgb.dtype}.'
		)
	if objects.shape != (count, OBJECT_DIM) or objects.dtype != np.float32:
		raise RobustObjectFieldObservationError(
			f'ROF V0 object must be float32 [{count},1770], got '
			f'{objects.shape} {objects.dtype}.'
		)
	if not np.isfinite(objects).all():
		raise RobustObjectFieldObservationError('ROF V0 object must be finite.')
	if masks.shape != (
		count, STACK_FRAMES, IMAGE_SIZE, IMAGE_SIZE
	) or masks.dtype != np.bool_:
		raise RobustObjectFieldObservationError(
			f'ROF V0 object_mask must be bool [{count},3,64,64], got '
			f'{masks.shape} {masks.dtype}.'
		)
	if not isinstance(diagnostics, Mapping) or set(diagnostics) != DIAGNOSTIC_FIELDS:
		actual = set(diagnostics) if isinstance(diagnostics, Mapping) else type(diagnostics)
		raise RobustObjectFieldObservationError(
			f'ROF V0 diagnostics must have exactly {sorted(DIAGNOSTIC_FIELDS)!r}, '
			f'got {actual!r}.'
		)

	valid = _vector(diagnostics, 'valid', count=count, dtype=np.bool_)
	lost = _vector(diagnostics, 'lost', count=count, dtype=np.bool_)
	nonempty = _vector(
		diagnostics, 'mask_nonempty', count=count, dtype=np.bool_
	)
	feature_finite = _vector(
		diagnostics, 'feature_finite', count=count, dtype=np.bool_
	)
	areas = _vector(
		diagnostics, 'mask_area_pixels', count=count, dtype=np.int64
	)
	border = _vector(
		diagnostics, 'mask_touches_border', count=count, dtype=np.bool_
	)
	confidence = _vector(
		diagnostics, 'confidence', count=count, dtype=np.float32
	)
	mask_score = _vector(
		diagnostics, 'mask_score', count=count, dtype=np.float32
	)
	runtime_ms = np.asarray(diagnostics['runtime_ms'])
	if (
		runtime_ms.ndim != 0
		or not np.issubdtype(runtime_ms.dtype, np.number)
		or not np.isfinite(runtime_ms).all()
		or float(runtime_ms) < 0.0
	):
		raise RobustObjectFieldObservationError(
			'ROF V0 runtime_ms must be a finite non-negative scalar.'
		)

	current_masks = masks[:, -1]
	measured_areas = current_masks.reshape(count, -1).sum(
		axis=-1, dtype=np.int64
	)
	if not np.array_equal(areas, measured_areas):
		raise RobustObjectFieldObservationError(
			'ROF V0 current masks disagree with same-response mask areas.'
		)
	if not np.array_equal(nonempty, measured_areas > 0):
		raise RobustObjectFieldObservationError(
			'ROF V0 mask_nonempty disagrees with current native masks.'
		)
	expected_border = np.concatenate((
		current_masks[:, 0], current_masks[:, -1],
		current_masks[:, :, 0], current_masks[:, :, -1],
	), axis=-1).any(axis=-1)
	if not np.array_equal(border, expected_border):
		raise RobustObjectFieldObservationError(
			'ROF V0 mask border status disagrees with current native masks.'
		)
	expected_valid = (~lost) & nonempty & feature_finite
	if not np.array_equal(valid, expected_valid):
		raise RobustObjectFieldObservationError(
			'ROF V0 tracker_valid must equal '
			'(~lost)&mask_nonempty&feature_finite.'
		)
	if np.any(confidence < 0.0) or np.any(confidence > 1.0):
		raise RobustObjectFieldObservationError(
			'ROF V0 tracker confidence must lie in [0,1].'
		)

	frames = objects.reshape(count, STACK_FRAMES, FRAME_DIM)
	current_status = frames[:, -1, STATUS_START:STATUS_START + len(STATUS_NAMES)]
	expected_status = np.stack((
		confidence,
		lost.astype(np.float32),
		valid.astype(np.float32),
		mask_score,
	), axis=-1)
	if not np.array_equal(current_status, expected_status):
		raise RobustObjectFieldObservationError(
			'ROF V0 current descriptor status is not the same Cutie response as '
			'the current native mask.'
		)

	role_exists = np.ones(count, dtype=np.float32)
	latest_hwc = np.ascontiguousarray(rgb[-3:].transpose(1, 2, 0))
	source_hash = _digest(latest_hwc)
	object_hash = _digest(objects)
	mask_hash = _digest(masks)
	binding = hashlib.sha256()
	binding.update(SCHEMA.encode('ascii'))
	binding.update(json.dumps(
		{'task': task, 'role_names': list(roles), 'sequence_id': sequence_id},
		sort_keys=True, separators=(',', ':'),
	).encode('utf-8'))
	binding.update(source_hash.encode('ascii'))
	binding.update(object_hash.encode('ascii'))
	binding.update(mask_hash.encode('ascii'))

	return RobustObjectFieldPacket(
		task=task,
		role_names=roles,
		rgb=_readonly_copy(rgb, dtype=np.uint8),
		object=_readonly_copy(objects, dtype=np.float32),
		object_mask=_readonly_copy(masks, dtype=np.bool_),
		role_exists=_readonly_copy(role_exists, dtype=np.float32),
		sequence_id=sequence_id,
		source_rgb_sha256=source_hash,
		object_sha256=object_hash,
		object_mask_sha256=mask_hash,
		binding_sha256=binding.hexdigest(),
	)


def validate_numpy_observation(observation, *, role_count: int) -> None:
	"""Validate the exact public array boundary used by replay/model adapters."""
	if not isinstance(role_count, int) or isinstance(role_count, bool) or role_count < 1:
		raise RobustObjectFieldObservationError(
			'ROF V0 role_count must be a positive integer.'
		)
	expected = {
		'rgb': ((RGB_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), np.uint8),
		'object': ((role_count, OBJECT_DIM), np.float32),
		'object_mask': (
			(role_count, STACK_FRAMES, IMAGE_SIZE, IMAGE_SIZE), np.bool_
		),
		'role_exists': ((role_count,), np.float32),
	}
	if not isinstance(observation, Mapping) or set(observation) != set(expected):
		actual = set(observation) if isinstance(observation, Mapping) else type(observation)
		raise RobustObjectFieldObservationError(
			f'ROF V0 observation keys must be exactly {sorted(expected)!r}, got '
			f'{actual!r}.'
		)
	for name, (shape, dtype) in expected.items():
		value = np.asarray(observation[name])
		if value.shape != shape or value.dtype != np.dtype(dtype):
			raise RobustObjectFieldObservationError(
				f'ROF V0 {name} must be {shape} {np.dtype(dtype)}, got '
				f'{value.shape} {value.dtype}.'
			)
		if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
			raise RobustObjectFieldObservationError(
				f'ROF V0 {name} must be finite.'
			)
	if not np.array_equal(
		observation['role_exists'], np.ones(role_count, dtype=np.float32)
	):
		raise RobustObjectFieldObservationError(
			'ROF V0 role_exists must be all one; padding roles are forbidden.'
		)


__all__ = [
	'DIAGNOSTIC_FIELDS',
	'FRAME_DIM',
	'IMAGE_SIZE',
	'OBJECT_DIM',
	'RGB_CHANNELS',
	'RobustObjectFieldObservationError',
	'RobustObjectFieldPacket',
	'SCHEMA',
	'STACK_FRAMES',
	'STATUS_NAMES',
	'STATUS_START',
	'build_packet',
	'validate_numpy_observation',
]
