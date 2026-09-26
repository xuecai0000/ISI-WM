"""Standalone adapter for the official OC-STORM Cutie object extractor.

The adapter deliberately does not hook into ``make_env`` or the training
agent.  It provides a small, explicit interface for perception experiments:

* rebuild Cutie's episode core and replay fixed permanent support at every episode;
* load fixed indexed role masks into permanent support memory;
* return one hard mask and one 2048-D OC-STORM feature per role;
* zero a role feature when Cutie's official ``defocus_cache`` marks it lost.

The model and its weights are external resources.  No alternate detector is
used when the official dependency is absent or incompatible.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

try:
	from tdmpc2.common.support_camera_contract import (
		QUADRUPED_CAMERA_ID,
		QUADRUPED_TASKS,
		validate_same_camera_contract,
	)
except ImportError:  # Support the historical ``PYTHONPATH=tdmpc2`` entrypoint.
	from common.support_camera_contract import (  # type: ignore[no-redef]
		QUADRUPED_CAMERA_ID,
		QUADRUPED_TASKS,
		validate_same_camera_contract,
	)


class CutiePreflightError(RuntimeError):
	"""Raised when a configured Cutie installation is incomplete."""


class CutieDependencyError(CutiePreflightError):
	"""Raised when the official Cutie Python modules cannot be imported."""


@dataclass(frozen=True)
class CutieOCConfig:
	"""External paths and the fixed role contract for a Cutie adapter.

	``repo_path`` is the root of the official OC-STORM checkout (the directory
	that contains ``feature_extractor``), not this TD-MPC2 repository.
	``tracker_size`` is ``(height, width)``.  ``None`` preserves the incoming
	frame size, which is the strict matched 64x64 protocol for this project.
	``support_input_size`` may explicitly freeze permanent support at 64x64 while
	runtime RGB is a same-state high-resolution diagnostic. ``mask_output_size``
	then maps masks back to canonical 64x64 before descriptor construction.
	``whole_arm_goal_v1`` requires roles ``('whole_arm', 'goal')``; the default
	legacy schema preserves the historical role contract and tensors unchanged.
	``generic_indexed_v1`` accepts any positive number of explicitly named roles whose
	indexed support masks are loaded by :func:`load_indexed_support_prompts`.
	``generic_entity_indexed_v1`` is reserved for a compiled object graph and
	accepts any positive number of tracker entities.  It does not change the
	historical two-role loader or schema.
	"""

	repo_path: str | Path
	checkpoint_path: str | Path
	role_names: tuple[str, ...]
	model_size: str = 'small'
	device: str = 'cuda:0'
	output_device: str = 'cpu'
	config_dir: str | Path | None = None
	expected_input_size: tuple[int, int] = (64, 64)
	support_input_size: tuple[int, int] | None = None
	mask_output_size: tuple[int, int] | None = None
	tracker_size: tuple[int, int] | None = None
	foreground_queries: int = 8
	amp: bool = True
	return_object_features: bool = True
	object_schema: str = 'legacy_three_object_v1'

	def validated(self) -> 'CutieOCConfig':
		if self.model_size not in {'small', 'base'}:
			raise ValueError("model_size must be 'small' or 'base'.")
		if not self.role_names:
			raise ValueError('role_names must contain at least one role.')
		if len(set(self.role_names)) != len(self.role_names):
			raise ValueError(f'role_names must be unique, got {self.role_names!r}.')
		if any(not str(name).strip() for name in self.role_names):
			raise ValueError('role_names cannot contain empty strings.')
		if self.object_schema not in _SUPPORTED_OBJECT_SCHEMAS:
			raise ValueError(
				'object_schema must be one of '
				f'{_SUPPORTED_OBJECT_SCHEMAS!r}, got {self.object_schema!r}.'
			)
		if self.object_schema == _WHOLE_ARM_GOAL_SCHEMA:
			expected_roles = _OBJECT_SCHEMA_ROLES[self.object_schema]
			if self.role_names != expected_roles:
				raise ValueError(
					f'object_schema={self.object_schema!r} requires role_names='
					f'{expected_roles!r}, got {self.role_names!r}.'
				)
		if self.foreground_queries != 8:
			raise ValueError(
				'Official OC-STORM uses exactly the first 8 foreground queries.'
			)
		for name, size in (
			('expected_input_size', self.expected_input_size),
			('support_input_size', self.support_input_size),
			('mask_output_size', self.mask_output_size),
			('tracker_size', self.tracker_size),
		):
			if size is not None and (len(size) != 2 or min(size) < 1):
				raise ValueError(f'{name} must be a positive (height, width) pair.')
		return self

	@property
	def repo(self) -> Path:
		return Path(self.repo_path).expanduser().resolve()

	@property
	def checkpoint(self) -> Path:
		return Path(self.checkpoint_path).expanduser().resolve()

	@property
	def hydra_config_dir(self) -> Path:
		if self.config_dir is not None:
			return Path(self.config_dir).expanduser().resolve()
		return self.repo / 'feature_extractor' / 'cutie' / 'cutie' / 'config'


@dataclass(frozen=True)
class CutieFrameResult:
	"""One causal Cutie result in the role order declared by the config.

	``masks`` is a CPU bool tensor ``[K, output_height, output_width]``.  The
	output defaults to the input resolution; a caller may explicitly request a
	canonical output size when Cutie consumes a perception-only high-res frame.
	``confidence`` reproduces OC-STORM's binary focus signal: one when the
	corresponding ``defocus_cache`` entry is active and zero when the role is
	lost.  ``mask_score`` separately reports the mean Cutie probability over
	the pixels assigned to that role and is diagnostic only.
	"""

	role_names: tuple[str, ...]
	masks: torch.Tensor
	centroid_xy: torch.Tensor
	object_features: torch.Tensor | None
	lost: torch.Tensor
	confidence: torch.Tensor
	mask_score: torch.Tensor
	runtime_ms: float
	input_size: tuple[int, int]
	mask_output_size: tuple[int, int]
	tracker_size: tuple[int, int]
	resized_from_input: bool


@dataclass(frozen=True)
class CutieSupportPrompts:
	"""Verified RGB support frames and fixed-ID disk masks."""

	frames: tuple[np.ndarray, ...]
	masks: tuple[np.ndarray, ...]
	role_names: tuple[str, ...]
	annotation_path: Path
	metadata: dict[str, Any]


def _rgb_sha256(frame: np.ndarray) -> str:
	"""Match the collector's hash contract: contiguous decoded RGB bytes."""
	return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def _typed_array_sha256(value: np.ndarray) -> str:
	array = np.ascontiguousarray(value)
	digest = hashlib.sha256()
	digest.update(str(array.dtype).encode('ascii'))
	digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
	digest.update(array.tobytes())
	return digest.hexdigest()


_REACHER_POINT_ROLES = ('base', 'elbow', 'control_tip', 'goal')
_REACHER_OBJECT_ROLES = ('proximal_link', 'distal_link', 'goal')
_WHOLE_ARM_GOAL_ROLES = ('whole_arm', 'goal')
_LEGACY_OBJECT_SCHEMA = 'legacy_three_object_v1'
_WHOLE_ARM_GOAL_SCHEMA = 'whole_arm_goal_v1'
_GENERIC_INDEXED_SCHEMA = 'generic_indexed_v1'
_GENERIC_ENTITY_INDEXED_SCHEMA = 'generic_entity_indexed_v1'
_OBJECT_SCHEMA_ROLES = {
	_LEGACY_OBJECT_SCHEMA: _REACHER_OBJECT_ROLES,
	_WHOLE_ARM_GOAL_SCHEMA: _WHOLE_ARM_GOAL_ROLES,
}
_SUPPORTED_OBJECT_SCHEMAS = (
	_LEGACY_OBJECT_SCHEMA,
	_WHOLE_ARM_GOAL_SCHEMA,
	_GENERIC_INDEXED_SCHEMA,
	_GENERIC_ENTITY_INDEXED_SCHEMA,
)

_GENERIC_SUPPORT_FORMAT = 'cutie_indexed_mask_support_v1'


def _support_mask_for_object_schema(
	indexed: np.ndarray,
	object_schema: str,
) -> np.ndarray:
	"""Map the legacy three-object raster to the requested Cutie object IDs."""
	if object_schema not in _OBJECT_SCHEMA_ROLES:
		raise ValueError(
			'object_schema must be one of '
			f'{tuple(_OBJECT_SCHEMA_ROLES)!r}, got {object_schema!r}.'
		)
	if object_schema == _LEGACY_OBJECT_SCHEMA:
		# Preserve the historical mask object and bytes exactly on the default path.
		return indexed
	whole_arm_goal = np.zeros_like(indexed)
	whole_arm_goal[(indexed == 1) | (indexed == 2)] = 1
	whole_arm_goal[indexed == 3] = 2
	return whole_arm_goal


def _segment_distance_squared(xx, yy, start, end):
	x0, y0 = start
	x1, y1 = end
	dx = x1 - x0
	dy = y1 - y0
	length_squared = dx * dx + dy * dy
	if length_squared <= 0:
		return (xx - x0) ** 2 + (yy - y0) ** 2
	position = ((xx - x0) * dx + (yy - y0) * dy) / length_squared
	position = np.clip(position, 0.0, 1.0)
	closest_x = x0 + position * dx
	closest_y = y0 + position * dy
	return (xx - closest_x) ** 2 + (yy - closest_y) ** 2


def load_point_support_prompts(
	annotation_path: str | Path,
	*,
	radius_px: float = 2.0,
	expected_records: int = 6,
	object_schema: str = _LEGACY_OBJECT_SCHEMA,
) -> CutieSupportPrompts:
	"""Convert the manual four-point pack into schema-selected object masks.

	This loader is deliberately strict: it accepts only the RGB-only support
	split and verifies every decoded-RGB hash. It first builds the historical
	``proximal_link`` (base-to-elbow thick segment), ``distal_link``
	(elbow-to-control-tip thick segment), and ``goal`` (disk) raster. Overlaps
	are resolved by nearest primitive, with exact ties assigned to the later
	object ID. ``whole_arm_goal_v1`` then unions the two existing arm labels and
	leaves goal pixels unchanged. It never reads simulator state.
	"""
	if object_schema not in _OBJECT_SCHEMA_ROLES:
		raise ValueError(
			'object_schema must be one of '
			f'{tuple(_OBJECT_SCHEMA_ROLES)!r}, got {object_schema!r}.'
		)
	object_roles = _OBJECT_SCHEMA_ROLES[object_schema]
	path = Path(annotation_path).expanduser().resolve()
	if not path.is_file():
		raise FileNotFoundError(f'Support annotations not found: {path}')
	if radius_px <= 0:
		raise ValueError('radius_px must be positive.')
	data = json.loads(path.read_text(encoding='utf-8'))
	point_roles = tuple(data.get('roles', ()))
	if point_roles != _REACHER_POINT_ROLES:
		raise ValueError(
			f'Expected point roles {_REACHER_POINT_ROLES!r}, got {point_roles!r}.'
		)
	collection = data.get('collection', {})
	if collection.get('split') != 'support':
		raise ValueError('Cutie prompts must come from the support split.')
	if collection.get('observation') != 'rgb':
		raise ValueError('Cutie prompts must be labeled from RGB observations.')
	if collection.get('label_policy') != 'manual_rgb_only':
		raise ValueError('Cutie prompts require label_policy=manual_rgb_only.')
	records = data.get('records', ())
	if len(records) != expected_records:
		raise ValueError(
			f'Expected {expected_records} support records, got {len(records)}.'
		)
	try:
		from PIL import Image
	except ImportError as exc:
		raise CutieDependencyError(
			'Pillow is required to load verified support RGB frames.'
		) from exc

	frames = []
	masks = []
	for expected_index, record in enumerate(records):
		if record.get('index') != expected_index:
			raise ValueError('Support record indices must be contiguous and ordered.')
		if record.get('video_split') != 'support':
			raise ValueError(f'Record {expected_index} is not from the support split.')
		image_path = (path.parent / record['image']).resolve()
		try:
			image_path.relative_to(path.parent)
		except ValueError as exc:
			raise ValueError(
				f'Support image escapes its pack directory: {image_path}'
			) from exc
		if not image_path.is_file():
			raise FileNotFoundError(f'Support image not found: {image_path}')
		frame = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.uint8)
		if frame.shape != (64, 64, 3):
			raise ValueError(
				f'Strict Cutie support frames must be 64x64 RGB, got {frame.shape}.'
			)
		expected_sha = record.get('image_sha256')
		actual_sha = _rgb_sha256(frame)
		if not expected_sha or actual_sha != expected_sha:
			raise ValueError(
				f'Support decoded-RGB hash mismatch for {image_path.name}: '
				f'expected={expected_sha}, actual={actual_sha}'
			)
		points = record.get('points', {})
		if tuple(points.keys()) != point_roles:
			raise ValueError(
				f'Record {expected_index} role order differs from {point_roles!r}.'
			)
		yy, xx = np.mgrid[:64, :64]
		centers = {}
		for role in point_roles:
			point = points[role]
			if not isinstance(point, list) or len(point) != 2:
				raise ValueError(
					f'Record {expected_index} role {role} lacks an [x,y] point.'
				)
			x, y = map(float, point)
			if not (0 <= x < 64 and 0 <= y < 64):
				raise ValueError(
					f'Record {expected_index} role {role} is outside 64x64: {point}.'
				)
			centers[role] = (x, y)
		distance = np.stack((
			_segment_distance_squared(
				xx, yy, centers['base'], centers['elbow']
			),
			_segment_distance_squared(
				xx, yy, centers['elbow'], centers['control_tip']
			),
			(xx - centers['goal'][0]) ** 2 + (yy - centers['goal'][1]) ** 2,
		))
		eligible = distance <= radius_px ** 2
		# A tiny descending bias makes exact ties prefer later role IDs.
		bias = -np.arange(
			len(_REACHER_OBJECT_ROLES), dtype=np.float64
		)[:, None, None] * 1e-9
		owner = np.argmin(distance + bias, axis=0)
		indexed = np.zeros((64, 64), dtype=np.uint8)
		covered = eligible.any(axis=0)
		indexed[covered] = (owner[covered] + 1).astype(np.uint8)
		for role_id, role in enumerate(_REACHER_OBJECT_ROLES, start=1):
			if not np.any(indexed == role_id):
				raise ValueError(
					f'Radius-{radius_px:g} prompt for role {role} vanished after '
					f'overlap resolution in record {expected_index}.'
				)
		indexed = _support_mask_for_object_schema(indexed, object_schema)
		for role_id, role in enumerate(object_roles, start=1):
			if not np.any(indexed == role_id):
				raise ValueError(
					f'Object schema {object_schema!r} produced no prompt pixels '
					f'for role {role} in record {expected_index}.'
				)
		frames.append(np.ascontiguousarray(frame))
		masks.append(indexed)

	return CutieSupportPrompts(
		frames=tuple(frames),
		masks=tuple(masks),
		role_names=object_roles,
		annotation_path=path,
		metadata={
			'task': collection.get('task'),
			'split': collection.get('split'),
			'manifest_sha256': collection.get('manifest_sha256'),
			'combined_manifest_sha256': collection.get(
				'combined_manifest_sha256'
			),
			'prompt_count': len(frames),
			'radius_px': float(radius_px),
			'source_resolution': [64, 64],
			'point_roles': list(_REACHER_POINT_ROLES),
			'object_roles': list(object_roles),
			'rasterization': (
				'two_thick_segments_plus_goal_disk'
				if object_schema == _LEGACY_OBJECT_SCHEMA
				else 'whole_arm_union_plus_goal_disk'
			),
			**(
				{'object_schema': object_schema}
				if object_schema != _LEGACY_OBJECT_SCHEMA else {}
			),
		},
	)


def load_indexed_support_prompts(
	annotation_path: str | Path,
	*,
	role_names: tuple[str, ...],
	expected_task: str,
	expected_records: int = 6,
	allow_simulator_support: bool = False,
) -> CutieSupportPrompts:
	"""Load K-role indexed masks generated from simulator segmentation.

	This is deliberately a separate, opt-in path.  The historical manual
	``whole_arm_goal_v1`` loader above is untouched and remains the default.
	Each support record owns one decoded 64x64 RGB frame and one decoded 64x64
	uint8 PNG whose values are exactly background ``0`` and role IDs ``1..K``.
	Hashes cover decoded contiguous arrays, not container/file bytes.
	"""
	if not allow_simulator_support:
		raise ValueError(
			'generic_indexed_v1 simulator support is disabled; set '
			'cutie_object_allow_simulator_support=true explicitly.'
		)
	roles = tuple(str(name) for name in role_names)
	if (
		len(roles) < 1
		or len(set(roles)) != len(roles)
		or any(not name.strip() for name in roles)
	):
		raise ValueError(
			'generic_indexed_v1 requires unique non-empty roles, '
			f'got {roles!r}.'
		)
	if expected_records < 1:
		raise ValueError('expected_records must be positive.')
	if not isinstance(expected_task, str) or not expected_task.strip():
		raise ValueError('expected_task must be a non-empty string.')

	path = Path(annotation_path).expanduser().resolve()
	if not path.is_file():
		raise FileNotFoundError(f'Support annotations not found: {path}')
	data = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(data, dict) or set(data) != {
		'format', 'roles', 'collection', 'records'
	}:
		raise ValueError(
			'Generic support JSON must contain exactly format, roles, collection, '
			'and records.'
		)
	if data.get('format') != _GENERIC_SUPPORT_FORMAT:
		raise ValueError(
			f'Generic support format must be {_GENERIC_SUPPORT_FORMAT!r}, got '
			f'{data.get("format")!r}.'
		)
	declared_roles = tuple(data.get('roles', ()))
	if declared_roles != roles:
		raise ValueError(
			f'Generic support roles must be ordered exactly as {roles!r}, got '
			f'{declared_roles!r}.'
		)
	collection = data.get('collection')
	if not isinstance(collection, dict):
		raise ValueError('Generic support collection must be an object.')
	if not isinstance(collection.get('task'), str) or not collection['task'].strip():
		raise ValueError('Generic indexed prompts require a non-empty collection.task.')
	if collection['task'] != expected_task:
		raise ValueError(
			f'Generic support task must be {expected_task!r}, got '
			f'{collection["task"]!r}.'
		)
	if collection.get('split') != 'support':
		raise ValueError('Generic indexed prompts must come from the support split.')
	if collection.get('observation') != 'rgb':
		raise ValueError('Generic indexed prompts require observation=rgb.')
	if collection.get('support_schema') != _GENERIC_INDEXED_SCHEMA:
		raise ValueError(
			'Generic indexed prompts require collection.support_schema='
			'generic_indexed_v1.'
		)
	if collection.get('label_policy') != 'simulator_segmentation_support_only':
		raise ValueError(
			'Generic indexed prompts require '
			'label_policy=simulator_segmentation_support_only.'
		)
	if collection.get('diagnostic_support') is not True:
		raise ValueError(
			'Generic indexed prompts require diagnostic_support=true provenance.'
		)
	camera_id = collection.get('camera_id')
	if 'camera_contract' in collection or expected_task in QUADRUPED_TASKS:
		camera_id = validate_same_camera_contract(
			collection,
			expected_camera_id=(
				QUADRUPED_CAMERA_ID if expected_task in QUADRUPED_TASKS else None
			),
		)
	records = data.get('records')
	if not isinstance(records, list) or len(records) != expected_records:
		raise ValueError(
			f'Expected {expected_records} generic support records, got '
			f'{len(records) if isinstance(records, list) else type(records).__name__}.'
		)
	try:
		from PIL import Image
	except ImportError as exc:
		raise CutieDependencyError(
			'Pillow is required to load verified support RGB/mask PNG files.'
		) from exc

	frames = []
	masks = []
	for expected_index, record in enumerate(records):
		if not isinstance(record, dict):
			raise ValueError(f'Generic support record {expected_index} must be an object.')
		if record.get('index') != expected_index:
			raise ValueError('Support record indices must be contiguous and ordered.')
		for field in ('image', 'image_sha256', 'indexed_mask', 'indexed_mask_sha256'):
			if not isinstance(record.get(field), str) or not record[field]:
				raise ValueError(
					f'Generic support record {expected_index} lacks {field!r}.'
				)

		image_path = (path.parent / record['image']).resolve()
		mask_path = (path.parent / record['indexed_mask']).resolve()
		for field, asset_path in (('image', image_path), ('indexed_mask', mask_path)):
			try:
				asset_path.relative_to(path.parent)
			except ValueError as exc:
				raise ValueError(
					f'Generic support {field} escapes its pack directory: {asset_path}'
				) from exc
			if not asset_path.is_file():
				raise FileNotFoundError(
					f'Generic support {field} not found: {asset_path}'
				)
		if mask_path.suffix.lower() != '.png':
			raise ValueError(f'Indexed support mask must be a PNG: {mask_path.name}')

		frame = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.uint8)
		if frame.shape != (64, 64, 3):
			raise ValueError(
				f'Generic support RGB must be 64x64, got {frame.shape}.'
			)
		frame = np.ascontiguousarray(frame)
		actual_image_sha = _rgb_sha256(frame)
		if actual_image_sha != record['image_sha256']:
			raise ValueError(
				f'Generic support decoded-RGB hash mismatch for {image_path.name}: '
				f'expected={record["image_sha256"]}, actual={actual_image_sha}'
			)

		with Image.open(mask_path) as mask_image:
			indexed = np.asarray(mask_image)
		if indexed.shape != (64, 64) or indexed.dtype != np.uint8:
			raise ValueError(
				'Generic indexed support mask must be uint8 [64,64], got '
				f'{indexed.shape} {indexed.dtype}.'
			)
		indexed = np.ascontiguousarray(indexed)
		actual_mask_sha = hashlib.sha256(indexed.tobytes(order='C')).hexdigest()
		if actual_mask_sha != record['indexed_mask_sha256']:
			raise ValueError(
				f'Generic support decoded-mask hash mismatch for {mask_path.name}: '
				f'expected={record["indexed_mask_sha256"]}, actual={actual_mask_sha}'
			)
		values = set(np.unique(indexed).tolist())
		role_ids = set(range(1, len(roles) + 1))
		allowed_ids = {0, *role_ids}
		if not values.issubset(allowed_ids) or not role_ids.issubset(values):
			raise ValueError(
				f'Generic indexed mask must contain role IDs 1..{len(roles)} and '
				f'no IDs outside 0..{len(roles)}; got '
				f'{sorted(values)!r} in record {expected_index}.'
			)
		frames.append(frame)
		masks.append(indexed)

	return CutieSupportPrompts(
		frames=tuple(frames),
		masks=tuple(masks),
		role_names=roles,
		annotation_path=path,
		metadata={
			'format': _GENERIC_SUPPORT_FORMAT,
			'support_schema': _GENERIC_INDEXED_SCHEMA,
			'task': collection.get('task'),
			'split': collection.get('split'),
			'label_policy': collection.get('label_policy'),
			'diagnostic_support': True,
			'camera_id': camera_id,
			'camera_contract': collection.get('camera_contract'),
			'prompt_count': len(frames),
			'source_resolution': [64, 64],
			'object_roles': list(roles),
		},
	)


def load_paired_native_support_prompts(
	annotation_path: str | Path,
	*,
	role_names: tuple[str, str],
	expected_task: str,
	support_resolution: int,
	expected_seed: int,
	allow_simulator_support: bool = False,
) -> CutieSupportPrompts:
	"""Load one resolution from a strict same-state native64/native128 pack."""
	from tdmpc2.common.cutie_paired_support import (
		FORMAT,
		MASK_GENERATION,
		PAIRING,
		RESOLUTIONS,
		RGB_GENERATION,
		SUPPORT_RECORDS,
		SUPPORT_SCHEMA,
		compute_paired_support_id,
		sha256_json,
	)

	if not allow_simulator_support:
		raise ValueError('Paired native simulator support requires explicit opt-in.')
	if type(support_resolution) is not int or support_resolution not in RESOLUTIONS:
		raise ValueError(f'support_resolution must be one of {RESOLUTIONS}.')
	roles = tuple(role_names)
	if (
		len(roles) != 2
		or len(set(roles)) != 2
		or any(not isinstance(role, str) or not role.strip() for role in roles)
	):
		raise ValueError('Paired native support requires two ordered unique roles.')
	path = Path(annotation_path).expanduser().resolve()
	if not path.is_file():
		raise FileNotFoundError(path)
	data = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(data, dict) or set(data) != {
		'format', 'roles', 'collection', 'records'
	}:
		raise ValueError('Paired native support has an invalid top-level schema.')
	if data.get('format') != FORMAT or tuple(data.get('roles', ())) != roles:
		raise ValueError('Paired native support format/ordered roles mismatch.')
	collection = data.get('collection')
	if not isinstance(collection, dict):
		raise ValueError('Paired native support collection must be an object.')
	expected_collection = {
		'support_schema': SUPPORT_SCHEMA,
		'task': expected_task,
		'observation': 'rgb',
		'split': 'support',
		'environment_seed': expected_seed,
		'episodes': SUPPORT_RECORDS,
		'camera_id': 0,
		'resolutions': list(RESOLUTIONS),
		'label_policy': 'simulator_segmentation_support_only',
		'diagnostic_support': True,
		'pairing': PAIRING,
		'rgb_generation': RGB_GENERATION,
		'mask_generation': MASK_GENERATION,
		'full_composed_rgb_native_high_resolution': False,
		'native_foreground_and_mask_not_resized_from_64': True,
	}
	for name, expected in expected_collection.items():
		if collection.get(name) != expected:
			raise ValueError(
				f'Paired native collection {name}={collection.get(name)!r}, '
				f'expected {expected!r}.'
			)
	expected_videos = [f'video{index}.mp4' for index in range(85, 90)]
	if collection.get('allowed_videos') != expected_videos:
		raise ValueError('Paired native support video allowlist changed.')
	if sorted(collection.get('covered_videos', ())) != expected_videos:
		raise ValueError('Paired native support did not cover every support video.')
	for name in ('background_seed', 'action_seed', 'reset_attempts'):
		if type(collection.get(name)) is not int:
			raise ValueError(f'Paired native collection {name} must be an integer.')
	if len({
		collection['environment_seed'], collection['background_seed'],
		collection['action_seed'],
	}) != 3:
		raise ValueError('Paired native support seed domains must be distinct.')
	paired_id = compute_paired_support_id(data)
	if collection.get('paired_support_id') != paired_id:
		raise ValueError('Paired native support identity hash mismatch.')
	geom_catalog = (path.parent / str(collection.get('geom_catalog', ''))).resolve()
	try:
		geom_catalog.relative_to(path.parent)
	except ValueError as exc:
		raise ValueError('Paired native geom catalog escapes its pack.') from exc
	if not geom_catalog.is_file() or hashlib.sha256(
		geom_catalog.read_bytes()
	).hexdigest() != collection.get('geom_catalog_sha256'):
		raise ValueError('Paired native geom catalog hash mismatch.')
	collector = Path(__file__).resolve().parents[1] / 'tools' / str(
		collection.get('collector', '')
	)
	if collector.name != 'collect_cutie_paired_native_support.py' or not collector.is_file():
		raise ValueError('Paired native support collector provenance is invalid.')
	collector_sha = hashlib.sha256(collector.read_bytes()).hexdigest()
	if collection.get('collector_sha256') != collector_sha:
		raise ValueError('Paired native support collector source hash changed.')

	def numeric_payload(record, name, expected_steps=None):
		value = record.get(name)
		if not isinstance(value, dict) or set(value) != {
			'dtype', 'shape', 'values', 'sha256'
		}:
			raise ValueError(f'Paired native {name} payload is malformed.')
		try:
			dtype = np.dtype(value['dtype'])
		except TypeError as exc:
			raise ValueError(f'Paired native {name} dtype is invalid.') from exc
		if dtype.kind not in 'fiu':
			raise ValueError(f'Paired native {name} dtype must be numeric.')
		shape = value['shape']
		if (
			not isinstance(shape, list)
			or any(type(item) is not int or item < 0 for item in shape)
		):
			raise ValueError(f'Paired native {name} shape is invalid.')
		array = np.asarray(value['values'], dtype=dtype)
		if list(array.shape) != shape or not np.isfinite(array).all():
			raise ValueError(f'Paired native {name} values/shape are invalid.')
		if _typed_array_sha256(array) != value['sha256']:
			raise ValueError(f'Paired native {name} hash mismatch.')
		if expected_steps is not None and (
			array.ndim < 1 or array.shape[0] != expected_steps
		):
			raise ValueError('Paired native action count/prefix mismatch.')
		return array

	try:
		from PIL import Image
	except ImportError as exc:
		raise CutieDependencyError('Pillow is required for paired support.') from exc
	records = data.get('records')
	if not isinstance(records, list) or len(records) != SUPPORT_RECORDS:
		raise ValueError(f'Paired native support requires {SUPPORT_RECORDS} records.')
	frames, masks = [], []
	physics_hashes, action_hashes, reset_ordinals = [], [], []
	previous_reset = 0
	asset_trace = hashlib.sha256()
	for index, record in enumerate(records):
		if not isinstance(record, dict) or record.get('index') != index:
			raise ValueError('Paired native records must be contiguous and ordered.')
		if record.get('accepted_state_ordinal') != index:
			raise ValueError('Paired native accepted-state ordinals changed.')
		reset = record.get('reset_ordinal')
		prefix = record.get('random_prefix_steps')
		if type(reset) is not int or reset <= previous_reset:
			raise ValueError('Paired native reset ordinals must strictly increase.')
		if type(prefix) is not int or prefix != 4 + 3 * index:
			raise ValueError('Paired native prefix-step schedule changed.')
		previous_reset = reset
		if record.get('active_video') not in collection.get('allowed_videos', ()):
			raise ValueError('Paired native record uses an out-of-split video.')
		if type(record.get('background_frame_index')) is not int or record[
			'background_frame_index'
		] < 0:
			raise ValueError('Paired native background frame index is invalid.')
		state = numeric_payload(record, 'physics_state')
		actions = numeric_payload(record, 'actions', expected_steps=prefix)
		if state.ndim != 1 or state.size < 1:
			raise ValueError('Paired native physics state must be one non-empty vector.')
		if actions.ndim != 2 or actions.shape[1] < 1:
			raise ValueError('Paired native actions must be [steps, action_dim].')
		physics_hashes.append(_typed_array_sha256(state))
		action_hashes.append(_typed_array_sha256(actions))
		reset_ordinals.append(reset)
		if set(record.get('selected_names', {})) != set(roles):
			raise ValueError('Paired native selected role provenance is malformed.')

		decoded = {}
		resolution_entries = record.get('resolutions')
		if not isinstance(resolution_entries, dict) or set(resolution_entries) != {
			str(value) for value in RESOLUTIONS
		}:
			raise ValueError('Paired native record resolution set changed.')
		for resolution in RESOLUTIONS:
			entry = resolution_entries[str(resolution)]
			if not isinstance(entry, dict):
				raise ValueError('Paired native resolution entry must be an object.')
			for field in (
				'image', 'image_sha256', 'indexed_mask',
				'indexed_mask_sha256', 'native_clean_rgb_sha256',
			):
				if not isinstance(entry.get(field), str) or not entry[field]:
					raise ValueError(f'Paired native resolution lacks {field}.')
			image_path = (path.parent / entry['image']).resolve()
			mask_path = (path.parent / entry['indexed_mask']).resolve()
			for asset in (image_path, mask_path):
				try:
					asset.relative_to(path.parent)
				except ValueError as exc:
					raise ValueError(f'Paired support asset escapes pack: {asset}') from exc
				if not asset.is_file() or asset.suffix.lower() != '.png':
					raise FileNotFoundError(asset)
			image = np.ascontiguousarray(np.asarray(
				Image.open(image_path).convert('RGB'), dtype=np.uint8
			))
			with Image.open(mask_path) as mask_image:
				mask = np.ascontiguousarray(np.asarray(mask_image))
			if image.shape != (resolution, resolution, 3):
				raise ValueError(f'Paired support RGB shape mismatch: {image.shape}.')
			if mask.shape != (resolution, resolution) or mask.dtype != np.uint8:
				raise ValueError(f'Paired support mask schema mismatch: {mask.shape}.')
			if _rgb_sha256(image) != entry['image_sha256']:
				raise ValueError('Paired support decoded RGB hash mismatch.')
			if hashlib.sha256(mask.tobytes()).hexdigest() != entry['indexed_mask_sha256']:
				raise ValueError('Paired support decoded mask hash mismatch.')
			values = set(np.unique(mask).tolist())
			if not values.issubset({0, 1, 2}) or not {1, 2}.issubset(values):
				raise ValueError('Paired support mask role IDs are invalid.')
			counts = entry.get('role_pixel_counts')
			expected_counts = {
				role: int((mask == role_id).sum())
				for role_id, role in enumerate(roles, start=1)
			}
			if counts != expected_counts:
				raise ValueError('Paired support role pixel counts mismatch.')
			asset_trace.update(entry['image_sha256'].encode('ascii'))
			asset_trace.update(entry['indexed_mask_sha256'].encode('ascii'))
			decoded[resolution] = (image, mask)
		resampling = getattr(Image, 'Resampling', Image)
		up_rgb = np.asarray(Image.fromarray(decoded[64][0], mode='RGB').resize(
			(128, 128), resample=resampling.BILINEAR
		))
		up_mask = np.asarray(Image.fromarray(decoded[64][1], mode='L').resize(
			(128, 128), resample=resampling.NEAREST
		))
		if np.array_equal(up_rgb, decoded[128][0]) or np.array_equal(
			up_mask, decoded[128][1]
		):
			raise ValueError('Paired native128 assets equal resized native64 assets.')
		evidence = record.get('cross_resolution_evidence')
		if not isinstance(evidence, dict) or not all(evidence.get(name) is True for name in (
			'clean128_distinct_from_bilinear64',
			'composed128_distinct_from_bilinear64',
			'mask128_distinct_from_nearest64',
		)):
			raise ValueError('Paired native cross-resolution evidence is invalid.')
		frames.append(decoded[support_resolution][0])
		masks.append(decoded[support_resolution][1])
	if previous_reset != collection['reset_attempts']:
		raise ValueError('Paired native final reset ordinal/reset attempts mismatch.')

	return CutieSupportPrompts(
		frames=tuple(frames),
		masks=tuple(masks),
		role_names=roles,
		annotation_path=path,
		metadata={
			'format': FORMAT,
			'support_schema': SUPPORT_SCHEMA,
			'task': expected_task,
			'split': 'support',
			'prompt_count': SUPPORT_RECORDS,
			'paired_support_id': paired_id,
			'source_resolution': [support_resolution, support_resolution],
			'object_roles': list(roles),
			'pairing': PAIRING,
			'rgb_generation': RGB_GENERATION,
			'mask_generation': MASK_GENERATION,
			'native_foreground_support': True,
			'native_mask_support': True,
			'full_composed_rgb_native_high_resolution': False,
			'physics_state_trace_sha256': sha256_json(physics_hashes),
			'action_sequence_trace_sha256': sha256_json(action_hashes),
			'reset_ordinal_trace_sha256': sha256_json(reset_ordinals),
			'asset_trace_sha256': asset_trace.hexdigest(),
		},
	)


@contextlib.contextmanager
def _temporary_sys_path(path: Path):
	value = str(path)
	already_present = value in sys.path
	if not already_present:
		sys.path.insert(0, value)
	try:
		yield
	finally:
		if not already_present:
			try:
				sys.path.remove(value)
			except ValueError:
				pass


def _required_paths(config: CutieOCConfig) -> dict[str, Path]:
	return {
		'oc_storm_repo': config.repo,
		'cutie_package': (
			config.repo / 'feature_extractor' / 'cutie' / 'cutie'
		),
		'hydra_config_dir': config.hydra_config_dir,
		'checkpoint': config.checkpoint,
	}


def _import_official_modules(repo: Path) -> tuple[type, type, Any]:
	"""Import the official Cutie runtime without OC-STORM's GUI wrapper."""
	try:
		# OC-STORM imports the vendored package through
		# ``feature_extractor.cutie.cutie...``, while Cutie's own modules use
		# absolute ``from cutie...`` imports. Both roots are therefore required.
		vendored_root = repo / 'feature_extractor' / 'cutie'
		with _temporary_sys_path(repo), _temporary_sys_path(vendored_root):
			cutie_module = importlib.import_module(
				'feature_extractor.cutie.cutie.model.cutie'
			)
			core_module = importlib.import_module(
				'feature_extractor.cutie.cutie.inference.inference_core'
			)
			args_module = importlib.import_module(
				'feature_extractor.cutie.cutie.inference.utils.args_utils'
			)
		return (
			cutie_module.CUTIE,
			core_module.InferenceCore,
			args_module.get_dataset_cfg,
		)
	except Exception as exc:
		raise CutieDependencyError(
			'Could not import the official OC-STORM Cutie modules from '
			f'{repo}. Install the dependencies from that checkout in the active '
			'environment and keep its feature_extractor directory intact. '
			f'Original import error: {type(exc).__name__}: {exc}'
		) from exc


def inspect_cutie_installation(
	config: CutieOCConfig,
	*,
	import_check: bool = True,
) -> dict[str, Any]:
	"""Strict, read-only dependency and weight preflight.

	The function raises instead of selecting DINO, SAM2, color matching, or any
	other fallback.  Model/checkpoint compatibility is checked by constructing
	``CutieOCAdapter`` separately, because that allocates GPU memory.
	"""
	config.validated()
	paths = _required_paths(config)
	missing = [name for name, path in paths.items() if not path.exists()]
	if missing:
		details = ', '.join(f'{name}={paths[name]}' for name in missing)
		raise CutiePreflightError(f'Missing required Cutie resources: {details}')
	if not paths['oc_storm_repo'].is_dir():
		raise CutiePreflightError(
			f'oc_storm_repo is not a directory: {paths["oc_storm_repo"]}'
		)
	if not paths['cutie_package'].is_dir():
		raise CutiePreflightError(
			f'cutie_package is not a directory: {paths["cutie_package"]}'
		)
	if not paths['hydra_config_dir'].is_dir():
		raise CutiePreflightError(
			f'hydra_config_dir is not a directory: {paths["hydra_config_dir"]}'
		)
	if not paths['checkpoint'].is_file():
		raise CutiePreflightError(
			f'checkpoint is not a file: {paths["checkpoint"]}'
		)

	module_status = {}
	for name in ('torch', 'numpy', 'hydra', 'omegaconf'):
		module_status[name] = importlib.util.find_spec(name) is not None
	missing_modules = [name for name, present in module_status.items() if not present]
	if missing_modules:
		raise CutieDependencyError(
			'Missing required Python modules: ' + ', '.join(missing_modules)
		)
	if import_check:
		_import_official_modules(config.repo)

	device = torch.device(config.device)
	if device.type == 'cuda':
		if not torch.cuda.is_available():
			raise CutiePreflightError(
				f'{config.device} was requested but torch.cuda.is_available() is false.'
			)
		index = torch.cuda.current_device() if device.index is None else device.index
		if index >= torch.cuda.device_count():
			raise CutiePreflightError(
				f'{config.device} is outside {torch.cuda.device_count()} visible CUDA device(s).'
			)
		device_name = torch.cuda.get_device_name(index)
	else:
		device_name = str(device)

	report = {
		'backend': 'official_oc_storm_cutie',
		'model_size': config.model_size,
		'roles': list(config.role_names),
		'num_roles': len(config.role_names),
		'device': str(device),
		'device_name': device_name,
		'repo_path': str(config.repo),
		'config_dir': str(config.hydra_config_dir),
		'checkpoint_path': str(config.checkpoint),
		'checkpoint_bytes': config.checkpoint.stat().st_size,
		'module_status': module_status,
		'import_check': bool(import_check),
		'expected_input_size': list(config.expected_input_size),
		'tracker_size': list(config.tracker_size) if config.tracker_size else None,
		'input_contract': 'same_latest_rgb_observation',
		'tracker_resize_contract': 'bilinear_no_new_sensor_information',
		'extra_sensor_pixels': False,
	}
	if config.object_schema != _LEGACY_OBJECT_SCHEMA:
		report['object_schema'] = config.object_schema
	return report


def _load_model(config: CutieOCConfig):
	CUTIE, InferenceCore, get_dataset_cfg = _import_official_modules(config.repo)
	try:
		from hydra import compose, initialize_config_dir
		from hydra.core.global_hydra import GlobalHydra
		from omegaconf import open_dict
	except Exception as exc:
		raise CutieDependencyError(
			f'Cutie requires Hydra/OmegaConf: {type(exc).__name__}: {exc}'
		) from exc

	if GlobalHydra.instance().is_initialized():
		raise CutieDependencyError(
			'The standalone Cutie adapter must be constructed before another Hydra '
			'application initializes GlobalHydra. It is intentionally not connected '
			'to the TD-MPC2 training path.'
		)
	try:
		with initialize_config_dir(
			version_base='1.3.2',
			config_dir=str(config.hydra_config_dir),
			job_name=f'tdmpc2_cutie_{config.model_size}',
		):
			model_cfg = compose(config_name=f'eval_config_{config.model_size}')
		with open_dict(model_cfg):
			model_cfg['weights'] = str(config.checkpoint)
		# This official helper is not just a dataset reader: it promotes runtime
		# values such as ``mem_every`` and ``use_long_term`` from the selected
		# dataset block into the top-level config.  Skipping it leaves those values
		# as ``None`` and makes InferenceCore fail during construction.
		get_dataset_cfg(model_cfg)
		model = CUTIE(model_cfg).to(torch.device(config.device)).eval()
		try:
			weights = torch.load(
				config.checkpoint,
				map_location=torch.device(config.device),
				weights_only=True,
			)
		except TypeError:
			weights = torch.load(
				config.checkpoint, map_location=torch.device(config.device)
			)
		model.load_weights(weights)
		processor = InferenceCore(model, cfg=model_cfg)
		return model, processor
	except Exception as exc:
		raise CutieDependencyError(
			'Official Cutie model construction or checkpoint loading failed. '
			f'config={config.hydra_config_dir}, checkpoint={config.checkpoint}. '
			f'Original error: {type(exc).__name__}: {exc}'
		) from exc


class CutieOCAdapter:
	"""Causal, role-ordered wrapper around official Cutie ``InferenceCore``."""

	def __init__(
		self,
		config: CutieOCConfig,
		*,
		_processor=None,
		_processor_factory=None,
	):
		self.config = config.validated()
		self.device = torch.device(self.config.device)
		self.output_device = torch.device(self.config.output_device)
		self._model = None
		self._processor_factory = _processor_factory
		if _processor is None:
			inspect_cutie_installation(self.config, import_check=True)
			self._model, self.processor = _load_model(self.config)
			processor_cls = type(self.processor)
			processor_cfg = getattr(self.processor, 'cfg', None)
			if processor_cfg is None:
				raise CutieDependencyError(
					'Official Cutie InferenceCore does not expose its cfg; a fresh '
					'episode core cannot be reconstructed safely.'
				)
			self._processor_factory = lambda: processor_cls(
				self._model, cfg=processor_cfg
			)
		else:
			# Dependency-free contract tests can inject an InferenceCore-compatible
			# processor/factory. Production callers should never use these private
			# arguments.
			self.processor = _processor
		if self._processor_factory is not None and not callable(self._processor_factory):
			raise TypeError('_processor_factory must be callable when provided.')
		self._frames = 0
		self._runtime_ms = 0.0
		self._prompt_frames = 0
		self._prompt_runtime_ms = 0.0
		self._permanent_prompts = 0
		self._support_prompts = None
		self._episode_hard_resets = 0
		self._episode_hard_reset_ms = 0.0
		self._support_replay_frames = 0
		self._support_replay_runtime_ms = 0.0

	@property
	def num_roles(self) -> int:
		return len(self.config.role_names)

	def reset_episode(self):
		"""Restore a history-independent episode state with the fixed support.

		Official Cutie's ``clear_non_permanent_memory()`` is not a complete
		episode reset: it retains trajectory-dependent ``last_mask``, sensory
		memory, and accumulated object values. Production adapters therefore
		construct a fresh lightweight ``InferenceCore`` around the already-loaded
		model and replay the immutable permanent support prompts. This avoids
		checkpoint/model reload while preventing one episode from influencing the
		first observation of the next.

		The legacy clear-only branch exists solely for dependency-free injected
		processors that do not provide a factory.
		"""
		if self._processor_factory is not None:
			if self._support_prompts is None or self._permanent_prompts < 1:
				raise CutiePreflightError(
					'Fresh episode reset requires verified permanent support prompts; '
					'call add_support_prompts() first.'
				)
			self._sync()
			started = time.perf_counter()
			old_processor = self.processor
			clear = getattr(old_processor, 'clear_non_permanent_memory', None)
			if callable(clear):
				clear()
			self._clear_shared_export_caches(old_processor)
			try:
				self.processor = self._processor_factory()
			except Exception as exc:
				# The old core has already been cleared and cannot be served safely.
				self.processor = old_processor
				raise CutieDependencyError(
					'Failed to construct a fresh Cutie InferenceCore for episode reset: '
					f'{type(exc).__name__}: {exc}'
				) from exc
			del old_processor
			self._clear_shared_export_caches(self.processor)
			try:
				self._install_support_prompts(
					self._support_prompts, account_as_initial=False
				)
			except Exception as exc:
				raise CutieDependencyError(
					'Failed to replay permanent support prompts into a fresh Cutie '
					f'episode core: {type(exc).__name__}: {exc}'
				) from exc
			# Replay advances the new core through prompt time indices 0..P-1.
			# Match the original cold-start sequence by resetting only its clock and
			# non-permanent suffix after support installation. This intentionally
			# retains support-derived last_mask/sensory/obj_v and permanent memory.
			clear = getattr(self.processor, 'clear_non_permanent_memory', None)
			if not callable(clear):
				raise CutieDependencyError(
					'Fresh Cutie InferenceCore does not expose '
					'clear_non_permanent_memory() after support replay.'
				)
			clear()
			self._clear_shared_export_caches(self.processor)
			self._sync()
			self._episode_hard_resets += 1
			self._episode_hard_reset_ms += (
				(time.perf_counter() - started) * 1000.0
			)
			return

		# Backward-compatible test-double path. It must never be mistaken for
		# the production isolation strategy exposed by runtime_summary().
		clear = getattr(self.processor, 'clear_non_permanent_memory', None)
		if not callable(clear):
			raise CutieDependencyError(
				'Cutie InferenceCore does not expose clear_non_permanent_memory().'
			)
		clear()

	def _clear_shared_export_caches(self, processor) -> None:
		"""Clear OC-STORM output caches attached to the shared model object."""
		network = getattr(processor, 'network', None)
		transformer = getattr(network, 'object_transformer', None)
		if transformer is None:
			return
		for name in (
			'obj_values_cache',
			'query_cache',
			'query_emb_cache',
			'query_post_process_cache',
			'q_weights_cache',
			'defocus_cache',
		):
			if hasattr(transformer, name):
				setattr(transformer, name, None)

	@staticmethod
	def _frame_array(frame) -> np.ndarray:
		if torch.is_tensor(frame):
			frame = frame.detach().cpu().numpy()
		frame = np.asarray(frame)
		if frame.ndim != 3:
			raise ValueError(f'RGB frame must be rank 3, got shape {frame.shape}.')
		if frame.shape[-1] == 3:
			pass
		elif frame.shape[0] == 3:
			frame = np.transpose(frame, (1, 2, 0))
		else:
			raise ValueError(
				f'RGB frame must be HWC or CHW with three channels, got {frame.shape}.'
			)
		if frame.dtype != np.uint8:
			raise ValueError(f'RGB frame must be uint8, got {frame.dtype}.')
		# Arrays produced by ``np.asarray(PIL.Image)`` may be contiguous but
		# read-only. ``torch.from_numpy`` warns because a write through the tensor
		# would then be undefined. The native observation is only 64x64, so make
		# one explicit writable copy at this boundary.
		return np.array(frame, dtype=np.uint8, order='C', copy=True)

	def _prepare_frame(self, frame: np.ndarray) -> torch.Tensor:
		value = torch.from_numpy(frame).to(self.device, non_blocking=True)
		value = value.permute(2, 0, 1).contiguous().float().div_(255.0)
		if self.config.tracker_size is not None:
			value = F.interpolate(
				value.unsqueeze(0),
				size=self.config.tracker_size,
				mode='bilinear',
				align_corners=False,
			).squeeze(0)
		return value

	def _prepare_prompt_mask(
		self,
		mask,
		*,
		input_size: tuple[int, int],
		tracker_size: tuple[int, int],
	) -> torch.Tensor:
		value = torch.as_tensor(mask)
		if value.ndim == 2:
			labels = value.long()
			if tuple(labels.shape) != input_size:
				raise ValueError(
					f'Indexed prompt mask must match frame size {input_size}, '
					f'got {tuple(labels.shape)}.'
				)
			if labels.min().item() < 0 or labels.max().item() > self.num_roles:
				raise ValueError(
					f'Indexed mask values must be in [0, {self.num_roles}].'
				)
			role_masks = F.one_hot(
				labels, num_classes=self.num_roles + 1
			).permute(2, 0, 1)[1:].bool()
		elif value.ndim == 3 and value.shape[0] == self.num_roles:
			if tuple(value.shape[1:]) != input_size:
				raise ValueError(
					f'Role masks must match frame size {input_size}, got {tuple(value.shape)}.'
				)
			role_masks = value.bool()
			if torch.any(role_masks.sum(dim=0) > 1):
				raise ValueError('Role prompt masks must not overlap.')
		else:
			raise ValueError(
				f'Prompt mask must be indexed [H,W] or bool [K,H,W], got {tuple(value.shape)}.'
			)
		missing = [
			name for name, present in zip(
				self.config.role_names, role_masks.flatten(1).any(dim=1).tolist()
			) if not present
		]
		if missing:
			raise ValueError(
				'Every configured role needs at least one prompt pixel; missing '
				+ ', '.join(missing)
			)
		role_masks = role_masks.float().unsqueeze(0)
		if tracker_size != input_size:
			role_masks = F.interpolate(role_masks, size=tracker_size, mode='nearest')
		return role_masks.squeeze(0).to(self.device, non_blocking=True)

	def _autocast_context(self):
		if self.config.amp and self.device.type == 'cuda':
			return torch.autocast(device_type='cuda', dtype=torch.float16)
		return contextlib.nullcontext()

	def _sync(self):
		if self.device.type == 'cuda':
			torch.cuda.synchronize(self.device)

	def _decode(
		self,
		prediction: torch.Tensor,
		*,
		input_size: tuple[int, int],
		runtime_ms: float,
	) -> CutieFrameResult:
		prediction = torch.as_tensor(prediction)
		if prediction.ndim != 3 or prediction.shape[0] != self.num_roles + 1:
			raise CutieDependencyError(
				'Cutie prediction must have shape [K+1,H,W]; got '
				f'{tuple(prediction.shape)} for K={self.num_roles}.'
			)
		labels = prediction.argmax(dim=0)
		masks = torch.stack(
			[labels == index for index in range(1, self.num_roles + 1)], dim=0
		)
		output_size = (
			tuple(self.config.mask_output_size)
			if self.config.mask_output_size is not None
			else input_size
		)
		if tuple(masks.shape[1:]) != output_size:
			masks = F.interpolate(
				masks.float().unsqueeze(0), size=output_size, mode='nearest'
			).squeeze(0).bool()
		centroids = []
		for mask in masks:
			coordinates = torch.nonzero(mask, as_tuple=False)
			if coordinates.numel() == 0:
				centroids.append(torch.full((2,), float('nan'), device=mask.device))
			else:
				yx = coordinates.float().mean(dim=0)
				centroids.append(torch.stack((yx[1], yx[0])))
		centroid_xy = torch.stack(centroids)

		network = getattr(self.processor, 'network', None)
		transformer = getattr(network, 'object_transformer', None)
		query_cache = getattr(transformer, 'query_post_process_cache', None)
		focus_cache = getattr(transformer, 'defocus_cache', None)
		if query_cache is None or focus_cache is None:
			raise CutieDependencyError(
				'Official OC-STORM feature extraction requires Cutie '
				'query_post_process_cache and defocus_cache.'
			)
		query_cache = torch.as_tensor(query_cache)
		if query_cache.ndim != 3 or query_cache.shape[0] != self.num_roles:
			raise CutieDependencyError(
				'query_post_process_cache must have shape [K,N,C], got '
				f'{tuple(query_cache.shape)}.'
			)
		if query_cache.shape[1] < self.config.foreground_queries:
			raise CutieDependencyError(
				f'Cutie exposes {query_cache.shape[1]} queries, fewer than requested '
				f'{self.config.foreground_queries}.'
			)
		features = query_cache[:, :self.config.foreground_queries].reshape(
			self.num_roles, -1
		).detach().clone()
		if features.shape[1] != 2048:
			raise CutieDependencyError(
				'Official OC-STORM Cutie features must be 8x256=2048 values per '
				f'object, got {features.shape[1]}.'
			)
		# Cutie populates these caches inside ``torch.inference_mode``. Clone the
		# exported value and use only out-of-place transforms so PyTorch 2.7 does
		# not reject an in-place update to an inference tensor.
		focus = torch.as_tensor(focus_cache).reshape(-1).detach().clone()
		if focus.numel() != self.num_roles:
			raise CutieDependencyError(
				f'defocus_cache must contain K={self.num_roles} values, got {focus.numel()}.'
			)
		focus = focus.to(device=features.device, dtype=features.dtype).clamp(0, 1)
		features = features * focus.unsqueeze(-1)
		lost = focus <= 0
		centroid_xy = centroid_xy.to(lost.device)
		centroid_xy = torch.where(
			lost.unsqueeze(-1),
			torch.full_like(centroid_xy, float('nan')),
			centroid_xy,
		)

		scores = []
		tracker_masks = torch.stack(
			[labels == index for index in range(1, self.num_roles + 1)], dim=0
		)
		for role in range(self.num_roles):
			region = tracker_masks[role]
			if bool(region.any()):
				scores.append(prediction[role + 1][region].float().mean())
			else:
				scores.append(prediction.new_zeros((), dtype=torch.float32))
		mask_score = torch.stack(scores)

		return CutieFrameResult(
			role_names=self.config.role_names,
			masks=masks.detach().cpu().bool(),
			centroid_xy=centroid_xy.detach().cpu().float(),
			object_features=(
				features.to(self.output_device).float()
				if self.config.return_object_features else None
			),
			lost=lost.detach().cpu().bool(),
			confidence=focus.detach().cpu().float(),
			mask_score=mask_score.detach().cpu().float(),
			runtime_ms=float(runtime_ms),
			input_size=input_size,
			mask_output_size=output_size,
			tracker_size=tuple(prediction.shape[-2:]),
			resized_from_input=tuple(prediction.shape[-2:]) != input_size,
		)

	def _step(
		self,
		frame,
		prompt_mask=None,
		*,
		force_permanent: bool = False,
		count_as_prompt: bool = False,
		count_as_support_replay: bool = False,
		decode_result: bool = True,
	):
		if count_as_prompt and count_as_support_replay:
			raise ValueError(
				'A Cutie step cannot be both an initial prompt and a support replay.'
			)
		frame_array = self._frame_array(frame)
		input_size = tuple(frame_array.shape[:2])
		expected_input_size = (
			tuple(self.config.support_input_size)
			if prompt_mask is not None and self.config.support_input_size is not None
			else tuple(self.config.expected_input_size)
		)
		if input_size != expected_input_size:
			raise ValueError(
				f'Cutie input must match the configured observation size '
				f'{expected_input_size}, got {input_size}. This prevents '
				'use of an uncontracted RGB resolution.'
			)
		self._sync()
		started = time.perf_counter()
		frame_tensor = self._prepare_frame(frame_array)
		tracker_size = tuple(frame_tensor.shape[-2:])
		mask_tensor = None
		if prompt_mask is not None:
			mask_tensor = self._prepare_prompt_mask(
				prompt_mask, input_size=input_size, tracker_size=tracker_size
			)
		with torch.inference_mode(), self._autocast_context():
			if mask_tensor is None:
				prediction = self.processor.step(frame_tensor)
			else:
				prediction = self.processor.step(
					frame_tensor,
					mask_tensor,
					idx_mask=False,
					force_permanent=bool(force_permanent),
				)
		self._sync()
		result = None
		if decode_result:
			result = self._decode(
				prediction, input_size=input_size, runtime_ms=0.0
			)
			self._sync()
		runtime_ms = (time.perf_counter() - started) * 1000.0
		if result is not None:
			result = replace(result, runtime_ms=float(runtime_ms))
		if count_as_prompt:
			self._prompt_frames += 1
			self._prompt_runtime_ms += runtime_ms
		elif count_as_support_replay:
			self._support_replay_frames += 1
			self._support_replay_runtime_ms += runtime_ms
		else:
			self._frames += 1
			self._runtime_ms += runtime_ms
		return result

	def _add_permanent_prompt(self, frame, prompt_mask) -> None:
		"""Add one verified support image/mask without decoding track features.

		Official OC-STORM only reads ``query_post_process_cache`` and
		``defocus_cache`` on later mask-free tracking frames. Cutie need not
		populate those caches while a permanent prompt is being installed.
		"""
		self._step(
			frame,
			prompt_mask,
			force_permanent=True,
			count_as_prompt=True,
			decode_result=False,
		)
		self._permanent_prompts += 1

	def _replay_permanent_prompt(self, frame, prompt_mask) -> None:
		"""Reinstall one logical prompt without changing initial-prompt metrics."""
		self._step(
			frame,
			prompt_mask,
			force_permanent=True,
			count_as_support_replay=True,
			decode_result=False,
		)

	def _install_support_prompts(
		self,
		support: CutieSupportPrompts,
		*,
		account_as_initial: bool,
	) -> None:
		installer = (
			self._add_permanent_prompt
			if account_as_initial
			else self._replay_permanent_prompt
		)
		for frame, mask in zip(support.frames, support.masks):
			installer(frame, mask)

	@staticmethod
	def _copy_support_prompts(support: CutieSupportPrompts) -> CutieSupportPrompts:
		"""Own immutable CPU copies so later caller mutation cannot change reset."""
		frames = tuple(
			np.array(
				CutieOCAdapter._frame_array(frame),
				dtype=np.uint8,
				order='C',
				copy=True,
			)
			for frame in support.frames
		)
		masks = []
		for mask in support.masks:
			if torch.is_tensor(mask):
				mask = mask.detach().cpu().numpy()
			masks.append(np.array(mask, order='C', copy=True))
		return CutieSupportPrompts(
			frames=frames,
			masks=tuple(masks),
			role_names=tuple(support.role_names),
			annotation_path=Path(support.annotation_path),
			metadata=copy.deepcopy(support.metadata),
		)

	def add_support_prompts(
		self, support: CutieSupportPrompts
	) -> None:
		"""Register all verified support prompts as permanent Cutie memory."""
		if self._permanent_prompts:
			raise CutiePreflightError(
				'Permanent prompts are already loaded; construct a fresh adapter '
				'instead of mixing support packs.'
			)
		if support.role_names != self.config.role_names:
			raise ValueError(
				f'Support roles {support.role_names!r} do not match adapter roles '
				f'{self.config.role_names!r}.'
			)
		if not support.frames or len(support.frames) != len(support.masks):
			raise ValueError('Support frames and masks must be non-empty and aligned.')
		owned_support = self._copy_support_prompts(support)
		self._install_support_prompts(owned_support, account_as_initial=True)
		self._support_prompts = owned_support

	def track(self, frame) -> CutieFrameResult:
		"""Track one later frame without a simulator query or future frame."""
		return self._step(frame)

	def track_episode(
		self,
		frames: Iterable[np.ndarray | torch.Tensor],
	) -> list[CutieFrameResult]:
		"""Track an ordered episode using only fixed permanent support prompts."""
		if self._permanent_prompts < 1:
			raise CutiePreflightError(
				'No permanent support prompt is loaded. Episode-first ground-truth '
				'masks are intentionally unsupported; call add_support_prompts() first.'
			)
		iterator = iter(frames)
		try:
			first_frame = next(iterator)
		except StopIteration as exc:
			raise ValueError('frames must contain at least one RGB frame.') from exc
		self.reset_episode()
		results = [self.track(first_frame)]
		results.extend(self.track(frame) for frame in iterator)
		return results

	def runtime_summary(self) -> dict[str, Any]:
		mean = self._runtime_ms / self._frames if self._frames else 0.0
		return {
			'input_size': tuple(self.config.expected_input_size),
			'support_input_size': tuple(
				self.config.support_input_size or self.config.expected_input_size
			),
			'mask_output_size': tuple(
				self.config.mask_output_size or self.config.expected_input_size
			),
			'tracker_size': tuple(
				self.config.tracker_size or self.config.expected_input_size
			),
			'frames': float(self._frames),
			'total_ms': float(self._runtime_ms),
			'ms_per_frame': float(mean),
			'prompt_frames': float(self._prompt_frames),
			'prompt_total_ms': float(self._prompt_runtime_ms),
			'permanent_prompts': float(self._permanent_prompts),
			'episode_reset_strategy': (
				'fresh_inference_core_support_replay_v1'
				if self._processor_factory is not None
				else 'legacy_clear_non_permanent_only_test_double'
			),
			'episode_hard_resets': float(self._episode_hard_resets),
			'episode_hard_reset_total_ms': float(self._episode_hard_reset_ms),
			'support_replay_frames': float(self._support_replay_frames),
			'support_replay_total_ms': float(self._support_replay_runtime_ms),
		}
