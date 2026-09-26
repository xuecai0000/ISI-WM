"""Privileged per-frame MuJoCo mask geometry for diagnostic control oracles.

This module is deliberately isolated from the deployable Cutie path.  It reads
the exact geom/site identities already frozen in a simulator-derived support
pack, renders segmentation at the same 64x64 camera as the policy observation,
and exports only visible two-role mask geometry.  No simulator state, object
pose, RGB pixel, or segmentation raster is returned to the controller.

The resulting observation is still privileged: the mask is requested from the
simulator on every frame.  It is therefore an oracle/diagnostic upper bound,
never a paper-comparable deployable method.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


IMAGE_SIZE = 64
FRAME_QUERY_DIM = 512
FRAME_CONTENT_DIM = 586
FRAME_FEATURE_DIM = 590
FORMAT = 'gt_mask_geometry_runtime_v1'
SUPPORT_FORMAT = 'cutie_indexed_mask_support_v1'
SUPPORT_SCHEMA = 'generic_indexed_v1'


class GTMaskOracleError(RuntimeError):
	"""The privileged mask source or its frozen provenance is invalid."""


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as file:
		for block in iter(lambda: file.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def _walk_env(env):
	seen = set()
	while env is not None and id(env) not in seen:
		seen.add(id(env))
		yield env
		env = getattr(env, 'env', None)


def _find_physics(env):
	for value in _walk_env(env):
		physics = getattr(value, 'physics', None)
		if physics is not None and callable(getattr(physics, 'render', None)):
			return physics
	raise GTMaskOracleError('Could not find dm_control physics in the wrapper chain.')


def _segmentation_constants() -> dict[str, int]:
	try:
		from dm_control.mujoco.wrapper.mjbindings import enums
	except ImportError as exc:
		raise GTMaskOracleError(
			'dm_control MuJoCo object-type constants are unavailable.'
		) from exc
	return {
		'geom': int(enums.mjtObj.mjOBJ_GEOM),
		'site': int(enums.mjtObj.mjOBJ_SITE),
	}


def _model_name(model, object_id: int, object_type: str) -> str | None:
	name = model.id2name(int(object_id), object_type)
	return None if name is None else str(name)


def _body_name(model, body_id: int) -> str | None:
	return _model_name(model, int(body_id), 'body')


def _mask_spatial_feature(mask: np.ndarray) -> np.ndarray:
	"""Reproduce the established 74-D Cutie mask geometry exactly."""
	mask_float = np.asarray(mask, dtype=np.float32)
	if mask_float.shape != (IMAGE_SIZE, IMAGE_SIZE):
		raise GTMaskOracleError(f'Oracle mask must be 64x64, got {mask_float.shape}.')
	occupancy = mask_float.reshape(8, 8, 8, 8).mean(axis=(1, 3)).reshape(-1)
	yx = np.argwhere(mask_float > 0.5)
	if not len(yx):
		return np.concatenate((
			occupancy,
			np.zeros(2 + 1 + 4 + 3, dtype=np.float32),
		)).astype(np.float32)
	y = yx[:, 0].astype(np.float64) / (IMAGE_SIZE - 1)
	x = yx[:, 1].astype(np.float64) / (IMAGE_SIZE - 1)
	cx, cy = float(x.mean()), float(y.mean())
	dx, dy = x - cx, y - cy
	summary = np.concatenate((
		np.asarray([cx, cy, mask_float.mean()], dtype=np.float32),
		np.asarray([x.min(), y.min(), x.max(), y.max()], dtype=np.float32),
		np.asarray(
			[(dx * dx).mean(), (dy * dy).mean(), (dx * dy).mean()],
			dtype=np.float32,
		),
	))
	return np.concatenate((occupancy, summary)).astype(np.float32)


class GTMaskGeometryClient:
	"""Cutie-client-shaped parent object backed by live MuJoCo segmentation."""

	def __init__(
		self,
		env,
		*,
		task: str,
		role_names: tuple[str, str],
		support_path: str | Path,
	):
		self._physics = _find_physics(env)
		self._task = str(task)
		self._roles = tuple(str(role) for role in role_names)
		if len(self._roles) != 2 or len(set(self._roles)) != 2:
			raise GTMaskOracleError('GT-mask geometry requires exactly two roles.')
		self._support_path = Path(support_path).expanduser().resolve()
		self._frames = 0
		self._runtime_ms = 0.0
		self._visibility_failures = {role: 0 for role in self._roles}
		self._last_diagnostics = None
		self._closed = False
		self._camera_id = 0
		self._catalog_path = None
		self._catalog_sha256 = None
		self._selections = self._load_selections()
		self.ready = {
			'status': 'ready',
			'backend': 'gt_mask_geometry',
			'task': self._task,
			'roles': self._roles,
			'support_task': self._task,
			'support_schema': SUPPORT_SCHEMA,
			'allow_simulator_support': True,
			'frame_feature_dim': FRAME_FEATURE_DIM,
			'stacked_feature_dim': 3 * FRAME_FEATURE_DIM,
			'episode_reset_strategy': 'stateless_same_frame_mujoco_segmentation_v1',
			'role_diagnostics_schema': 'cutie_role_runtime_diagnostics_v1',
			'observation_variant': 'gt_mask_geometry',
			'privileged_runtime_segmentation': True,
			'camera_id': self._camera_id,
			'image_size': (IMAGE_SIZE, IMAGE_SIZE),
			'geom_catalog_path': str(self._catalog_path),
			'geom_catalog_sha256': self._catalog_sha256,
			'device': None,
			'current_device': None,
			'device_name': None,
			'cuda_visible_devices': None,
		}

	@property
	def last_diagnostics(self):
		return self._last_diagnostics

	def _load_selections(self):
		if not self._support_path.is_file():
			raise FileNotFoundError(self._support_path)
		annotations = json.loads(self._support_path.read_text(encoding='utf-8'))
		collection = annotations.get('collection', {})
		if (
			annotations.get('format') != SUPPORT_FORMAT
			or tuple(annotations.get('roles', ())) != self._roles
			or collection.get('support_schema') != SUPPORT_SCHEMA
			or collection.get('task') != self._task
			or collection.get('split') != 'support'
			or collection.get('label_policy')
			!= 'simulator_segmentation_support_only'
			or collection.get('diagnostic_support') is not True
			or collection.get('camera_id') != 0
			or collection.get('image_size') != [IMAGE_SIZE, IMAGE_SIZE]
		):
			raise GTMaskOracleError('Support annotations are not the frozen oracle pack.')
		catalog_name = collection.get('geom_catalog')
		if not isinstance(catalog_name, str) or Path(catalog_name).name != catalog_name:
			raise GTMaskOracleError('Support geom_catalog must be one local filename.')
		catalog_path = (self._support_path.parent / catalog_name).resolve()
		try:
			catalog_path.relative_to(self._support_path.parent)
		except ValueError as exc:
			raise GTMaskOracleError('Support geom_catalog escapes its pack.') from exc
		if not catalog_path.is_file():
			raise FileNotFoundError(catalog_path)
		catalog_sha = _sha256(catalog_path)
		if catalog_sha != collection.get('geom_catalog_sha256'):
			raise GTMaskOracleError('Support geom_catalog SHA256 mismatch.')
		catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
		if (
			catalog.get('task') != self._task
			or catalog.get('camera_id') != 0
			or set(catalog.get('matched_objects', {})) != set(self._roles)
			or catalog.get('segmentation_object_types') != _segmentation_constants()
		):
			raise GTMaskOracleError('Support geom_catalog identity is invalid.')
		model = self._physics.model
		selections = []
		seen = set()
		for role in self._roles:
			items = catalog['matched_objects'][role]
			if not isinstance(items, list) or not items:
				raise GTMaskOracleError(f'Role {role!r} has no matched simulator object.')
			selected = []
			for item in items:
				if not isinstance(item, Mapping):
					raise GTMaskOracleError(f'Malformed matched object for {role!r}.')
				object_type = item.get('object_type')
				if object_type not in {'geom', 'site'}:
					raise GTMaskOracleError(f'Unsupported object type {object_type!r}.')
				object_id = item.get('id')
				body_id = item.get('body_id')
				if not isinstance(object_id, int) or not isinstance(body_id, int):
					raise GTMaskOracleError('Matched simulator IDs must be integers.')
				limit = int(model.ngeom if object_type == 'geom' else model.nsite)
				if object_id < 0 or object_id >= limit:
					raise GTMaskOracleError('Matched simulator object ID is out of range.')
				actual_body = int(
					model.geom_bodyid[object_id]
					if object_type == 'geom' else model.site_bodyid[object_id]
				)
				identity = (object_type, object_id)
				if (
					identity in seen
					or actual_body != body_id
					or _model_name(model, object_id, object_type) != item.get('name')
					or _body_name(model, actual_body) != item.get('body_name')
				):
					raise GTMaskOracleError(
						f'Runtime simulator identity disagrees for {role!r}: {item!r}.'
					)
				seen.add(identity)
				selected.append(identity)
			selections.append(tuple(selected))
		self._catalog_path = catalog_path
		self._catalog_sha256 = catalog_sha
		return tuple(selections)

	def _frame(self) -> np.ndarray:
		if self._closed:
			raise GTMaskOracleError('GT-mask oracle client is closed.')
		started = time.perf_counter()
		segmentation = np.asarray(self._physics.render(
			height=IMAGE_SIZE,
			width=IMAGE_SIZE,
			camera_id=self._camera_id,
			segmentation=True,
		))
		if segmentation.shape != (IMAGE_SIZE, IMAGE_SIZE, 2):
			raise GTMaskOracleError(
				f'MuJoCo segmentation must be [64,64,2], got {segmentation.shape}.'
			)
		constants = _segmentation_constants()
		masks = []
		occupied = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.bool_)
		for index, (role, selected) in enumerate(
			zip(self._roles, self._selections)
		):
			mask = np.zeros_like(occupied)
			for object_type, object_id in selected:
				mask |= (
					(segmentation[..., 0] == object_id)
					& (segmentation[..., 1] == constants[object_type])
				)
			if not mask.any():
				# Visible-surface oracle only: an actually out-of-view/fully occluded
				# role remains explicitly missing. Never reject or resample a state.
				self._visibility_failures[role] += 1
			if np.any(occupied & mask):
				raise GTMaskOracleError('GT role masks overlap in rendered pixels.')
			occupied |= mask
			masks.append(mask)
		masks = np.stack(masks, axis=0)
		spatial = np.stack(
			[_mask_spatial_feature(mask) for mask in masks], axis=0
		).astype(np.float32)
		query = np.zeros((2, FRAME_QUERY_DIM), dtype=np.float32)
		visible = masks.reshape(2, -1).any(axis=-1)
		status = np.stack((
			visible.astype(np.float32),
			(~visible).astype(np.float32),
			visible.astype(np.float32),
			visible.astype(np.float32),
		), axis=-1)
		feature = np.concatenate((query, spatial, status), axis=-1)
		if feature.shape != (2, FRAME_FEATURE_DIM) or not np.isfinite(feature).all():
			raise AssertionError(f'Invalid GT-mask feature shape/value: {feature.shape}.')
		border = np.concatenate(
			(masks[:, 0, :], masks[:, -1, :], masks[:, :, 0], masks[:, :, -1]),
			axis=-1,
		).any(axis=-1)
		runtime_ms = (time.perf_counter() - started) * 1000.0
		self._frames += 1
		self._runtime_ms += runtime_ms
		self._last_diagnostics = {
			'valid': np.asarray(visible, dtype=np.bool_),
			'lost': np.asarray(~visible, dtype=np.bool_),
			'mask_nonempty': np.asarray(visible, dtype=np.bool_),
			'feature_finite': np.ones(2, dtype=np.bool_),
			'mask_area_pixels': masks.reshape(2, -1).sum(axis=-1).astype(np.int64),
			'mask_touches_border': np.asarray(border, dtype=np.bool_),
			'confidence': visible.astype(np.float32),
			'mask_score': visible.astype(np.float32),
			'runtime_ms': float(runtime_ms),
		}
		return np.ascontiguousarray(feature, dtype=np.float32)

	def reset_track(self, frame) -> np.ndarray:
		return self._frame()

	def track(self, frame) -> np.ndarray:
		return self._frame()

	def metrics(self) -> dict[str, Any]:
		return {
			'worker_restarts': 0,
			'timeouts': 0,
			'format': FORMAT,
			'frames': int(self._frames),
			'ms_per_frame': (
				float(self._runtime_ms / self._frames) if self._frames else 0.0
			),
			'visibility_failures': dict(self._visibility_failures),
			'privileged_runtime_segmentation': True,
			'query_feature_policy': 'exact_zero_512_v1',
		}

	def close(self):
		self._closed = True


def zero_query_feature(frame_feature: np.ndarray) -> np.ndarray:
	"""Return Cutie mask geometry/status with the 512-D query block removed."""
	value = np.asarray(frame_feature, dtype=np.float32)
	if value.shape != (2, FRAME_FEATURE_DIM) or not np.isfinite(value).all():
		raise ValueError(f'Expected finite [2,590] frame feature, got {value.shape}.')
	output = np.array(value, dtype=np.float32, order='C', copy=True)
	output[:, :FRAME_QUERY_DIM] = 0.0
	if not np.array_equal(
		output[:, FRAME_QUERY_DIM:], value[:, FRAME_QUERY_DIM:]
	):
		raise AssertionError('Geometry-only projection changed mask/status values.')
	return output


__all__ = [
	'FORMAT',
	'GTMaskGeometryClient',
	'GTMaskOracleError',
	'zero_query_feature',
]
