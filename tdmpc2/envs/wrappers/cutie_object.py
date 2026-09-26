"""Live, causal Cutie object observations for two-role DMControl tasks.

The official OC-STORM Cutie runtime composes its own Hydra configuration.  A
TD-MPC2 training process is already inside a Hydra application by the time the
environment is constructed, so Cutie is hosted in a mandatory ``spawn`` child
process.  The child has a fresh ``GlobalHydra`` singleton and an independent
CUDA RNG stream; the parent never clears or mutates TD-MPC2's Hydra state.

By default only the newest native 64x64 RGB observation crosses the process
boundary.  An explicit diagnostic can instead send an extra 128/256 same-state
DMC RGB render after replaying the existing visual wrappers; policy RGB remains
64x64. Actions, rewards, simulator state, segmentation, and ``info`` are not
part of the protocol. The returned ``object`` observation contains three causal
frames of generic, role-ordered Cutie features with shape ``[2, 1770]``.
"""

from __future__ import annotations

import multiprocessing as mp
import hashlib
import json
import os
import random
import time
import traceback
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import torch


ROLE_NAMES = ('whole_arm', 'goal')
ACROBOT_ENTITY_NAMES = ('whole_acrobot',)
TASK_ROLE_NAMES = {
	'reacher-visual-small': ROLE_NAMES,
	'reacher-easy': ROLE_NAMES,
	'reacher-hard': ROLE_NAMES,
	'cup-catch': ('cup', 'ball'),
	'cartpole-balance': ('cart', 'pole'),
	'cartpole-balance-sparse': ('cart', 'pole'),
	'cartpole-swingup': ('cart', 'pole'),
	'cartpole-swingup-sparse': ('cart', 'pole'),
	'finger-spin': ('finger', 'spinner'),
	# Turn rewards depend on the spinner tip relative to an episode-specific
	# rendered target site.  Tracking only finger/spinner makes two episodes
	# with different goals observationally indistinguishable to an object-only
	# controller.  Finger Spin has no target-angle objective and deliberately
	# keeps its established two-role contract.
	'finger-turn-easy': ('finger', 'spinner', 'target'),
	'finger-turn-hard': ('finger', 'spinner', 'target'),
	'pendulum-swingup': ('base', 'pendulum'),
	'hopper-stand': ('torso', 'leg', 'foot'),
	'hopper-hop': ('torso', 'leg', 'foot'),
	'walker-stand': ('torso', 'right_leg', 'left_leg'),
	'walker-walk': ('torso', 'right_leg', 'left_leg'),
	'walker-run': ('torso', 'right_leg', 'left_leg'),
	'cheetah-run': ('torso', 'back_leg', 'front_leg'),
	'quadruped-run': ('torso', 'front_legs', 'back_legs'),
	'quadruped-walk': ('torso', 'front_legs', 'back_legs'),
	'acrobot-swingup': ('whole_acrobot',),
}
TASK_SUPPORT_SCHEMAS = {
	task: (
		'whole_arm_goal_v1' if task in {
			'reacher-visual-small', 'reacher-easy', 'reacher-hard'
		} else 'generic_indexed_v1'
	)
	for task in TASK_ROLE_NAMES
}
LEGACY_ACROBOT_LINKS = ('upper_arm', 'lower_arm')


def task_role_names(task, contract='canonical_v1'):
	"""Resolve roles without silently changing the canonical object taxonomy."""
	if contract == 'legacy_acrobot_links_v1':
		if task != 'acrobot-swingup':
			raise ValueError('legacy_acrobot_links_v1 is restricted to acrobot-swingup.')
		return LEGACY_ACROBOT_LINKS
	if contract != 'canonical_v1':
		raise ValueError(f'Unknown Cutie task role contract: {contract!r}.')
	return TASK_ROLE_NAMES[task]


IMAGE_SIZE = 64
NATIVE_HIGHRES_SIZES = (128, 256)
QUERY_SLOTS = 8
QUERY_DIM = 256
QUERY_FEATURE_DIM = QUERY_SLOTS * QUERY_DIM
QUERY_POOL_DIM = 2 * QUERY_DIM
MASK_POOL_SIZE = 8
MASK_SPATIAL_DIM = 64 + 2 + 1 + 4 + 3
STATUS_DIM = 4
FRAME_FEATURE_DIM = QUERY_POOL_DIM + MASK_SPATIAL_DIM + STATUS_DIM
FRAME_CONTENT_DIM = FRAME_FEATURE_DIM - STATUS_DIM
STACK_FRAMES = 3
STACKED_FEATURE_DIM = STACK_FRAMES * FRAME_FEATURE_DIM
POLICY_BURST_FORMAT = 'cutie_policy_burst_plan_v1'
POLICY_BURST_INVALID_ENCODING = 'empty_lost_v1'
POLICY_BURST_EPISODES = 20
POLICY_BURST_DECISION_STEPS = 500
OBSERVATION_VARIANTS = ('full', 'cutie_mask_geometry', 'gt_mask_geometry')
OBSERVATION_FRAME_SCHEMAS = {
	'full': 'cutie_query_mask_status_v1',
	'cutie_mask_geometry': 'cutie_mask_geometry_v1',
	'gt_mask_geometry': 'simulator_gt_mask_geometry_v1',
}


def _load_spatial_object_graph(path, *, task, source_roles):
	"""Load the declarative object/relationship schema for spatial-token control."""
	graph_path = Path(str(path)).expanduser().resolve()
	if not graph_path.is_file():
		raise FileNotFoundError(f'Spatial object graph does not exist: {graph_path}')
	payload = json.loads(graph_path.read_text(encoding='utf-8'))
	if not isinstance(payload, dict):
		raise ValueError('Spatial object graph must be a JSON object.')
	if payload.get('format') != 'support_conditioned_object_graph_v1':
		raise ValueError(f'Unsupported spatial object graph format: {payload.get("format")!r}.')
	if payload.get('task') != task:
		raise ValueError(
			f'Spatial object graph task {payload.get("task")!r} != {task!r}.'
		)
	if tuple(payload.get('source_roles', ())) != tuple(source_roles):
		raise ValueError(
			'Spatial object graph source roles do not match the live Cutie roles.'
		)
	entities = payload.get('tracking_entities')
	relations = payload.get('relations')
	if not isinstance(entities, list) or not entities:
		raise ValueError('Spatial object graph requires at least one tracking entity.')
	if not isinstance(relations, list):
		raise ValueError('Spatial object graph relations must be a list.')
	return payload, graph_path, hashlib.sha256(graph_path.read_bytes()).hexdigest()


def _whole_acrobot_spatial_frame(feature: np.ndarray) -> np.ndarray:
	"""Make the Acrobot observation invariant to upper/lower tracker identity.

	The live tracker still receives its established two support masks, but the
	controller sees only their symmetric union. The union is duplicated into two
	structural query slots; learned role embeddings and the declared revolute edge
	may specialize those slots without assigning pixels to links frame by frame.
	"""
	value = np.asarray(feature, dtype=np.float32)
	if value.shape != (2, FRAME_FEATURE_DIM):
		raise ValueError(f'Acrobot spatial source must be [2,590], got {value.shape}.')
	query = value[:, :QUERY_POOL_DIM].mean(axis=0)
	occupancies = value[
		:, QUERY_POOL_DIM:QUERY_POOL_DIM + MASK_POOL_SIZE * MASK_POOL_SIZE
	]
	union_occupancy = np.clip(occupancies.sum(axis=0), 0.0, 1.0)
	# Reconstruct a deterministic coarse mask solely to obtain the same generic
	# centroid/bbox/moment convention as the established observation contract.
	coarse = union_occupancy.reshape(MASK_POOL_SIZE, MASK_POOL_SIZE)
	dense = np.repeat(
		np.repeat(coarse, IMAGE_SIZE // MASK_POOL_SIZE, axis=0),
		IMAGE_SIZE // MASK_POOL_SIZE,
		axis=1,
	)
	spatial = _mask_spatial_feature(dense)
	status = value[:, FRAME_CONTENT_DIM:]
	merged_status = np.asarray([
		float(status[:, 0].min()),
		float(status[:, 1].max()),
		float(status[:, 2].min()),
		float(status[:, 3].min()),
	], dtype=np.float32)
	merged = np.concatenate([query, spatial, merged_status]).astype(np.float32)
	if merged.shape != (FRAME_FEATURE_DIM,) or not np.isfinite(merged).all():
		raise AssertionError('Whole-Acrobot spatial descriptor is malformed.')
	return np.stack([merged, merged], axis=0)


class CutieObjectError(RuntimeError):
	"""Base error for the live Cutie observation path."""


class CutieObjectWorkerError(CutieObjectError):
	"""The isolated Cutie process failed or returned an invalid response."""


class CutieObjectTimeoutError(CutieObjectWorkerError):
	"""The isolated Cutie process did not answer within its frozen timeout."""


@dataclass(frozen=True)
class CutieObjectWorkerConfig:
	"""Serializable production configuration passed to the spawn child."""

	repo_path: str
	checkpoint_path: str
	support_path: str
	config_dir: str | None = None
	device: str = 'cuda:0'
	tracker_height: int = 448
	tracker_width: int = 448
	native_highres_enabled: bool = False
	native_highres_size: int = 128
	model_size: str = 'small'
	prompt_radius: float = 2.0
	amp: bool = True
	worker_timeout_seconds: float = 300.0
	role_names: tuple[str, ...] = ROLE_NAMES
	support_schema: str = 'whole_arm_goal_v1'
	allow_simulator_support: bool = False
	task: str = 'reacher-visual-small'
	true_entity_enabled: bool = False
	task_role_contract: str = 'canonical_v1'
	return_masks: bool = False

	def validated(self) -> 'CutieObjectWorkerConfig':
		for name in ('repo_path', 'checkpoint_path', 'support_path'):
			value = str(getattr(self, name)).strip()
			if not value or value.lower() in {'none', 'null'}:
				raise ValueError(f'{name} must be a non-empty path.')
		# TD-MPC2 and its replay buffer both use logical cuda:0. Physical GPU
		# selection belongs to CUDA_VISIBLE_DEVICES in the launching shell.
		if self.device != 'cuda:0':
			raise ValueError(
				"Live Cutie must use logical device 'cuda:0'. Select a physical GPU "
				'with CUDA_VISIBLE_DEVICES; never pass a physical cuda index here.'
			)
		if self.tracker_height < 1 or self.tracker_width < 1:
			raise ValueError('Cutie tracker dimensions must be positive.')
		if not isinstance(self.native_highres_enabled, bool):
			raise ValueError('cutie_object_native_highres_enabled must be boolean.')
		if (
			not isinstance(self.native_highres_size, int)
			or isinstance(self.native_highres_size, bool)
			or self.native_highres_size not in NATIVE_HIGHRES_SIZES
		):
			raise ValueError(
				'cutie_object_native_highres_size must be 128 or 256.'
			)
		if self.model_size not in {'small', 'base'}:
			raise ValueError("Cutie model_size must be 'small' or 'base'.")
		if self.prompt_radius <= 0:
			raise ValueError('Cutie prompt_radius must be positive.')
		if self.worker_timeout_seconds <= 0:
			raise ValueError('Cutie worker_timeout_seconds must be positive.')
		roles = tuple(self.role_names)
		if (
			len(roles) < 1
			or len(set(roles)) != len(roles)
			or any(not isinstance(name, str) or not name.strip() for name in roles)
		):
			raise ValueError(
				'Live Cutie requires unique non-empty role_names, '
				f'got {self.role_names!r}.'
			)
		if self.support_schema not in {'whole_arm_goal_v1', 'generic_indexed_v1'}:
			raise ValueError(
				'cutie_object_support_schema must be whole_arm_goal_v1 or '
				f'generic_indexed_v1, got {self.support_schema!r}.'
			)
		if self.support_schema == 'whole_arm_goal_v1' and roles != ROLE_NAMES:
			raise ValueError(
				'whole_arm_goal_v1 requires role_names '
				f'{ROLE_NAMES!r}, got {roles!r}.'
			)
		if self.support_schema == 'generic_indexed_v1' and not self.allow_simulator_support:
			raise ValueError(
				'generic_indexed_v1 requires explicit '
				'cutie_object_allow_simulator_support=true.'
			)
		if self.task not in TASK_ROLE_NAMES:
			raise ValueError(
				f'Live Cutie task must be one of {tuple(TASK_ROLE_NAMES)!r}, '
				f'got {self.task!r}.'
			)
		expected_roles = task_role_names(self.task, self.task_role_contract)
		if roles != expected_roles:
			raise ValueError(
				f'{self.task} requires role_names={expected_roles!r}, '
				f'got {roles!r}.'
			)
		if self.task_role_contract == 'legacy_acrobot_links_v1' and self.true_entity_enabled:
			raise ValueError('Legacy Acrobot link tracking forbids true-entity conversion.')
		if not isinstance(self.return_masks, bool):
			raise ValueError('return_masks must be boolean.')
		if self.return_masks and self.true_entity_enabled:
			raise ValueError('Mask export requires direct role tracking.')
		allowed_schemas = (
			{'whole_arm_goal_v1', 'generic_indexed_v1'}
			if self.task in {'reacher-visual-small', 'reacher-easy', 'reacher-hard'}
			else {'generic_indexed_v1'}
		)
		if self.support_schema not in allowed_schemas:
			raise ValueError(
				f'{self.task} does not accept support_schema={self.support_schema!r}.'
			)
		if not isinstance(self.true_entity_enabled, bool):
			raise ValueError('cutie_object_true_entity_enabled must be boolean.')
		if self.true_entity_enabled and self.task != 'acrobot-swingup':
			raise ValueError('True-entity tracking currently requires acrobot-swingup.')
		return self


def _as_numpy(value, *, dtype=None) -> np.ndarray:
	if torch.is_tensor(value):
		value = value.detach().cpu().numpy()
	return np.asarray(value, dtype=dtype)


def _native_hwc_rgb(observation) -> np.ndarray:
	"""Extract exactly the newest frame from a 3xRGB native observation stack."""
	value = torch.as_tensor(observation)
	if value.ndim != 3 or tuple(value.shape) != (9, IMAGE_SIZE, IMAGE_SIZE):
		raise ValueError(
			'Live Cutie requires a three-frame CHW uint8 observation with shape '
			f'(9, 64, 64), got {tuple(value.shape)}.'
		)
	if value.dtype != torch.uint8:
		raise ValueError(f'Live Cutie RGB must be uint8, got {value.dtype}.')
	return np.array(
		value[-3:].detach().cpu().permute(1, 2, 0).contiguous().numpy(),
		dtype=np.uint8,
		order='C',
		copy=True,
	)


def _mask_spatial_feature(mask) -> np.ndarray:
	"""Generic 74-D mask feature used by the frozen perception probe."""
	mask_float = np.asarray(mask, dtype=np.float32)
	if mask_float.shape != (IMAGE_SIZE, IMAGE_SIZE):
		raise ValueError(f'Cutie mask must be 64x64, got {mask_float.shape}.')
	occupancy = mask_float.reshape(
		MASK_POOL_SIZE,
		IMAGE_SIZE // MASK_POOL_SIZE,
		MASK_POOL_SIZE,
		IMAGE_SIZE // MASK_POOL_SIZE,
	).mean(axis=(1, 3)).reshape(-1)
	yx = np.argwhere(mask_float > 0.5)
	if not len(yx):
		return np.concatenate([
			occupancy,
			np.zeros(2 + 1 + 4 + 3, dtype=np.float32),
		]).astype(np.float32)
	y = yx[:, 0].astype(np.float64) / (IMAGE_SIZE - 1)
	x = yx[:, 1].astype(np.float64) / (IMAGE_SIZE - 1)
	cx, cy = float(x.mean()), float(y.mean())
	dx, dy = x - cx, y - cy
	summary = np.concatenate([
		np.asarray([cx, cy, mask_float.mean()], dtype=np.float32),
		np.asarray([x.min(), y.min(), x.max(), y.max()], dtype=np.float32),
		np.asarray(
			[(dx * dx).mean(), (dy * dy).mean(), (dx * dy).mean()],
			dtype=np.float32,
		),
	])
	return np.concatenate([occupancy, summary]).astype(np.float32)


def generic_object_frame(result, *, role_names=ROLE_NAMES) -> np.ndarray:
	"""Convert one official result into finite generic features ``[K, 590]``.

	Query pooling accumulates in float64 and is invariant to the ordering of the
	eight foreground query slots. An invalid role has an exactly-zero query
	feature; its mask/status diagnostics remain visible and no past or future
	feature is substituted.
	"""
	role_names = tuple(role_names)
	if not role_names or len(set(role_names)) != len(role_names):
		raise ValueError(f'Live Cutie requires unique non-empty roles, got {role_names!r}.')
	if tuple(getattr(result, 'role_names', ())) != role_names:
		raise ValueError(
			f'Cutie roles must be {role_names!r}, got '
			f'{tuple(getattr(result, "role_names", ()))!r}.'
		)
	features = _as_numpy(result.object_features)
	masks = _as_numpy(result.masks)
	lost = _as_numpy(result.lost, dtype=np.bool_)
	confidence = _as_numpy(result.confidence, dtype=np.float32)
	mask_score = _as_numpy(result.mask_score, dtype=np.float32)
	if features.shape != (len(role_names), QUERY_FEATURE_DIM):
		raise ValueError(
			f'Cutie object_features must be [K,2048], got {features.shape}.'
		)
	if masks.shape != (len(role_names), IMAGE_SIZE, IMAGE_SIZE):
		raise ValueError(f'Cutie masks must be [K,64,64], got {masks.shape}.')
	for name, value in (
		('lost', lost),
		('confidence', confidence),
		('mask_score', mask_score),
	):
		if value.shape != (len(role_names),):
			raise ValueError(f'Cutie {name} must have shape [K], got {value.shape}.')
	if not np.isfinite(confidence).all() or not np.isfinite(mask_score).all():
		raise ValueError('Cutie confidence and mask_score must be finite.')

	feature_finite = np.isfinite(features).all(axis=-1)
	mask_nonempty = masks.reshape(len(role_names), -1).any(axis=-1)
	valid = (~lost) & mask_nonempty & feature_finite
	safe = np.nan_to_num(
		features, nan=0.0, posinf=0.0, neginf=0.0
	).astype(np.float64, copy=False)
	queries = safe.reshape(len(role_names), QUERY_SLOTS, QUERY_DIM)
	query_feature = np.concatenate([
		queries.mean(axis=1),
		queries.std(axis=1, ddof=0),
	], axis=-1).astype(np.float32)
	query_feature[~valid] = 0.0

	spatial = np.stack(
		[_mask_spatial_feature(mask) for mask in masks], axis=0
	).astype(np.float32)
	status = np.stack([
		confidence,
		lost.astype(np.float32),
		valid.astype(np.float32),
		mask_score,
	], axis=-1).astype(np.float32)
	output = np.concatenate([query_feature, spatial, status], axis=-1)
	if output.shape != (len(role_names), FRAME_FEATURE_DIM):
		raise AssertionError(f'Internal Cutie feature shape error: {output.shape}.')
	if not np.isfinite(output).all():
		raise ValueError('Live Cutie object observation must be finite.')
	return np.ascontiguousarray(output, dtype=np.float32)


def _result_diagnostics(result, *, role_names=ROLE_NAMES) -> dict[str, np.ndarray]:
	"""Return exact per-role tracker diagnostics before feature sanitization."""
	role_names = tuple(role_names)
	features = _as_numpy(result.object_features)
	masks = _as_numpy(result.masks, dtype=np.bool_)
	lost = _as_numpy(result.lost, dtype=np.bool_)
	confidence = _as_numpy(result.confidence, dtype=np.float32)
	mask_score = _as_numpy(result.mask_score, dtype=np.float32)
	expected_roles = len(role_names)
	if features.shape != (expected_roles, QUERY_FEATURE_DIM):
		raise ValueError(
			f'Cutie diagnostic features must be [{expected_roles},2048], got '
			f'{features.shape}.'
		)
	if masks.shape != (expected_roles, IMAGE_SIZE, IMAGE_SIZE):
		raise ValueError(
			f'Cutie diagnostic masks must be [{expected_roles},64,64], got '
			f'{masks.shape}.'
		)
	for name, value in (
		('lost', lost), ('confidence', confidence), ('mask_score', mask_score)
	):
		if value.shape != (expected_roles,):
			raise ValueError(
				f'Cutie diagnostic {name} must be [{expected_roles}], got '
				f'{value.shape}.'
			)
	feature_finite = np.isfinite(features).all(axis=-1)
	mask_nonempty = masks.reshape(expected_roles, -1).any(axis=-1)
	valid = (~lost) & mask_nonempty & feature_finite
	border = np.concatenate(
		(masks[:, 0, :], masks[:, -1, :], masks[:, :, 0], masks[:, :, -1]),
		axis=-1,
	).any(axis=-1)
	return {
		'valid': np.ascontiguousarray(valid, dtype=np.bool_),
		'lost': np.ascontiguousarray(lost, dtype=np.bool_),
		'mask_nonempty': np.ascontiguousarray(mask_nonempty, dtype=np.bool_),
		'feature_finite': np.ascontiguousarray(feature_finite, dtype=np.bool_),
		'mask_area_pixels': np.ascontiguousarray(
			masks.reshape(expected_roles, -1).sum(axis=-1), dtype=np.int64
		),
		'mask_touches_border': np.ascontiguousarray(border, dtype=np.bool_),
		'confidence': np.ascontiguousarray(confidence, dtype=np.float32),
		'mask_score': np.ascontiguousarray(mask_score, dtype=np.float32),
	}


def _whole_acrobot_support(support):
	"""Compile the already verified support labels into one tracker entity.

	Only the six frozen support masks are unioned. Runtime masks are produced by
	Cutie with one object identity; they are never reconstructed from two role
	tracks or thresholded pooled occupancy. Source files remain unchanged.
	"""
	if tuple(support.role_names) != TASK_ROLE_NAMES['acrobot-swingup']:
		raise ValueError('Whole-Acrobot support requires upper_arm/lower_arm source roles.')
	if support.metadata.get('task') != 'acrobot-swingup':
		raise ValueError('Whole-Acrobot support has the wrong task.')
	masks = []
	for source in support.masks:
		value = np.asarray(source)
		if value.shape != (IMAGE_SIZE, IMAGE_SIZE) or not np.isin(value, [0, 1, 2]).all():
			raise ValueError('Whole-Acrobot support must contain native64 labels 0/1/2.')
		mask = np.ascontiguousarray(value != 0, dtype=np.uint8)
		if not mask.any():
			raise ValueError('Whole-Acrobot support entity cannot be empty.')
		masks.append(mask)
	return replace(
		support,
		masks=tuple(masks),
		role_names=ACROBOT_ENTITY_NAMES,
		metadata={
			**support.metadata,
			'object_schema': 'generic_entity_indexed_v1',
			'source_roles': list(support.role_names),
			'tracking_entities': list(ACROBOT_ENTITY_NAMES),
			'entity_compilation': 'native_support_label_union_before_tracking_v1',
		},
	)


def _whole_acrobot_worker_output(result):
	"""Represent one real entity in the encoder's two structural query slots.

	Both slots contain the same entity measurement, including its own validity.
	The duplicate slots are not upper/lower-arm detections. Geometry is computed
	from the original native64 binary entity mask, before occupancy pooling.
	"""
	if tuple(result.role_names) != ACROBOT_ENTITY_NAMES:
		raise ValueError('True-entity tracking must return exactly one whole_acrobot.')
	entity_diagnostics = _result_diagnostics(result, role_names=ACROBOT_ENTITY_NAMES)
	# Reuse the established descriptor math on the original mask. Duplication
	# changes only the encoder interface, not the tracker object count or shape.
	fields = {
		name: np.repeat(_as_numpy(getattr(result, name)), 2, axis=0)
		for name in ('object_features', 'masks', 'lost', 'confidence', 'mask_score')
	}
	policy_result = SimpleNamespace(
		role_names=TASK_ROLE_NAMES['acrobot-swingup'], **fields
	)
	frame = generic_object_frame(
		policy_result, role_names=TASK_ROLE_NAMES['acrobot-swingup']
	)
	diagnostics = {
		name: np.ascontiguousarray(np.repeat(value, 2, axis=0))
		for name, value in entity_diagnostics.items()
	}
	return frame, diagnostics


def _adapter_imports():
	try:
		from perception.cutie_oc_adapter import (
			CutieOCAdapter,
			CutieOCConfig,
			load_indexed_support_prompts,
			load_point_support_prompts,
		)
	except ImportError:
		from tdmpc2.perception.cutie_oc_adapter import (
			CutieOCAdapter,
			CutieOCConfig,
			load_indexed_support_prompts,
			load_point_support_prompts,
		)
	return (
		CutieOCAdapter,
		CutieOCConfig,
		load_point_support_prompts,
		load_indexed_support_prompts,
	)


def _send_worker_error(connection, phase: str, request_id, exc: BaseException):
	payload = {
		'status': 'error',
		'phase': phase,
		'request_id': request_id,
		'error_type': type(exc).__name__,
		'error': str(exc),
		'traceback': traceback.format_exc(),
	}
	try:
		connection.send(payload)
	except (BrokenPipeError, EOFError, OSError):
		pass


def _cutie_worker_main(connection, raw_config: Mapping[str, Any]):
	"""Spawn-process entrypoint. Keep this function module-level and picklable."""
	try:
		config = CutieObjectWorkerConfig(**dict(raw_config)).validated()
		from hydra.core.global_hydra import GlobalHydra

		parent_hydra_inherited = GlobalHydra.instance().is_initialized()
		if parent_hydra_inherited:
			raise RuntimeError(
				'The spawn worker unexpectedly inherited an initialized GlobalHydra.'
			)
		random.seed(0)
		np.random.seed(0)
		torch.manual_seed(0)
		if not torch.cuda.is_available():
			raise RuntimeError('Live Cutie requires CUDA in the worker process.')
		torch.cuda.set_device(torch.device(config.device))
		torch.cuda.manual_seed_all(0)
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True

		(
			CutieOCAdapter,
			CutieOCConfig,
			load_point_support,
			load_indexed_support,
		) = _adapter_imports()
		if config.support_schema == 'whole_arm_goal_v1':
			support = load_point_support(
				config.support_path,
				radius_px=config.prompt_radius,
				expected_records=6,
				object_schema='whole_arm_goal_v1',
			)
		else:
			support = load_indexed_support(
				config.support_path,
				role_names=tuple(config.role_names),
				expected_task=config.task,
				expected_records=6,
				allow_simulator_support=config.allow_simulator_support,
			)
		support_task = support.metadata.get('task')
		if support_task != config.task:
			raise RuntimeError(
				f'Cutie support task mismatch: expected {config.task!r}, '
				f'got {support_task!r}.'
			)
		if config.true_entity_enabled:
			support = _whole_acrobot_support(support)
		tracking_roles = (
			ACROBOT_ENTITY_NAMES if config.true_entity_enabled else tuple(config.role_names)
		)
		perception_size = (
			config.native_highres_size
			if config.native_highres_enabled else IMAGE_SIZE
		)
		adapter_config = CutieOCConfig(
			repo_path=config.repo_path,
			checkpoint_path=config.checkpoint_path,
			role_names=tracking_roles,
			model_size=config.model_size,
			device=config.device,
			output_device='cpu',
			config_dir=config.config_dir,
			expected_input_size=(perception_size, perception_size),
			support_input_size=(IMAGE_SIZE, IMAGE_SIZE),
			mask_output_size=(IMAGE_SIZE, IMAGE_SIZE),
			tracker_size=(config.tracker_height, config.tracker_width),
			foreground_queries=QUERY_SLOTS,
			amp=config.amp,
			return_object_features=True,
			object_schema=(
				'generic_entity_indexed_v1' if config.true_entity_enabled else config.support_schema
			),
		)
		adapter = CutieOCAdapter(adapter_config)
		adapter.add_support_prompts(support)
		summary = adapter.runtime_summary()
		for name, expected in (
			('input_size', (perception_size, perception_size)),
			('support_input_size', (IMAGE_SIZE, IMAGE_SIZE)),
			('mask_output_size', (IMAGE_SIZE, IMAGE_SIZE)),
			('tracker_size', (config.tracker_height, config.tracker_width)),
		):
			if tuple(summary.get(name, ())) != expected:
				raise RuntimeError(
					f'Cutie adapter {name} contract mismatch: '
					f'{summary.get(name)!r} != {expected!r}.'
				)
		if summary.get('permanent_prompts') != 6.0:
			raise RuntimeError(
				'Live Cutie requires exactly six permanent support prompts; got '
				f'{summary.get("permanent_prompts")!r}.'
			)
		reset_strategy = summary.get('episode_reset_strategy')
		if reset_strategy != 'fresh_inference_core_support_replay_v1':
			raise RuntimeError(
				'Live Cutie requires history-independent episode reset; got '
				f'{reset_strategy!r}.'
			)
		connection.send({
			'status': 'ready',
			'pid': os.getpid(),
			'start_method': mp.get_start_method(allow_none=True),
			'global_hydra_initialized_before_cutie': parent_hydra_inherited,
			'roles': tuple(config.role_names),
			'true_entity_enabled': config.true_entity_enabled,
			'tracking_entities': tracking_roles,
			'policy_slot_semantics': (
				'duplicated_whole_entity_structural_queries_v1'
				if config.true_entity_enabled else 'independent_role_measurements_v1'
			),
			'spatial_geometry_source': 'native_binary_tracker_mask_before_pooling_v1',
			'task': config.task,
			'support_task': support_task,
			'support_schema': config.support_schema,
			'allow_simulator_support': config.allow_simulator_support,
			'frame_feature_dim': FRAME_FEATURE_DIM,
			'stacked_feature_dim': STACKED_FEATURE_DIM,
			'permanent_prompts': int(summary['permanent_prompts']),
			'episode_reset_strategy': reset_strategy,
			'role_diagnostics_schema': 'cutie_role_runtime_diagnostics_v1',
			'device': config.device,
			'tracker_size': (config.tracker_height, config.tracker_width),
			'perception_input_source': (
				'same_state_native_dmc_render_with_visual_wrapper_replay_v1'
				if config.native_highres_enabled
				else 'policy_observation_latest_rgb_v1'
			),
			'perception_highres_semantics': (
				'native_dmc_foreground_same_state;when_video_background_enabled_'
				'current_native64_background_frame_is_resized_without_clock_advance'
				if config.native_highres_enabled else None
			),
			'perception_input_size': (perception_size, perception_size),
			'policy_rgb_size': (IMAGE_SIZE, IMAGE_SIZE),
			'support_input_size': (IMAGE_SIZE, IMAGE_SIZE),
			'support_resolution_policy': 'frozen_native64_support_v1',
			'mask_output_size': (IMAGE_SIZE, IMAGE_SIZE),
			'mask_geometry_resolution': (IMAGE_SIZE, IMAGE_SIZE),
			'native_highres_enabled': config.native_highres_enabled,
			'extra_sensor_information': config.native_highres_enabled,
			'raw_sensor_information_parity': not config.native_highres_enabled,
			'fair_representation_comparison': not config.native_highres_enabled,
			'agent_observation_unchanged': True,
			'treatment': (
				'runtime_cutie_input_resolution_only'
				if config.native_highres_enabled else 'historical_policy_rgb64'
			),
			'model_size': config.model_size,
			'prompt_radius': config.prompt_radius,
			'amp': config.amp,
			'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
			'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
			'current_device': int(torch.cuda.current_device()),
			'device_name': torch.cuda.get_device_name(torch.cuda.current_device()),
			'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
		})
		while True:
			try:
				request = connection.recv()
			except EOFError:
				break
			request_id = request.get('request_id') if isinstance(request, dict) else None
			try:
				if not isinstance(request, dict):
					raise ValueError('Cutie worker request must be a dictionary.')
				op = request.get('op')
				if op == 'close':
					if set(request) != {'op', 'request_id'}:
						raise ValueError('Close request has unexpected fields.')
					connection.send({'status': 'closed', 'request_id': request_id})
					break
				if op not in {'reset_track', 'track'}:
					raise ValueError(f'Unknown Cutie worker operation: {op!r}.')
				if set(request) != {'op', 'request_id', 'frame'}:
					raise ValueError(
						'Track requests may contain only op/request_id/frame.'
					)
				frame = np.asarray(request['frame'])
				if (
					frame.shape != (perception_size, perception_size, 3)
					or frame.dtype != np.uint8
				):
					raise ValueError(
						'Worker RGB must match the configured perception input '
						f'[{perception_size},{perception_size},3], got '
						f'{frame.shape} {frame.dtype}.'
					)
				if op == 'reset_track':
					adapter.reset_episode()
				result = adapter.track(np.array(frame, copy=True, order='C'))
				if config.true_entity_enabled:
					frame_feature, diagnostics = _whole_acrobot_worker_output(result)
				else:
					diagnostics = _result_diagnostics(
						result, role_names=tuple(config.role_names)
					)
					frame_feature = generic_object_frame(
						result, role_names=tuple(config.role_names)
					)
				response = {
					'status': 'ok',
					'request_id': request_id,
					'object': frame_feature,
					'valid': diagnostics['valid'],
					'lost': diagnostics['lost'],
					'diagnostics': diagnostics,
					'runtime_ms': float(result.runtime_ms),
				}
				if config.return_masks:
					response['masks'] = np.array(
						_as_numpy(result.masks), dtype=np.bool_, order='C', copy=True
					)
				connection.send(response)
			except Exception as exc:
				_send_worker_error(connection, 'request', request_id, exc)
				# Tracker state after an exception is undefined. Fail closed instead
				# of serving a stale or partially updated feature on the next step.
				break
	except Exception as exc:
		_send_worker_error(connection, 'initialization', None, exc)
	finally:
		try:
			connection.close()
		except OSError:
			pass


class _SpawnCutieClient:
	"""Synchronous, fail-closed parent endpoint for the isolated worker."""

	def __init__(self, config: CutieObjectWorkerConfig):
		self.config = config.validated()
		self._closed = False
		self._next_request_id = 0
		self._timeouts = 0
		self.last_diagnostics = None
		context = mp.get_context('spawn')
		parent_connection, child_connection = context.Pipe(duplex=True)
		self._connection = parent_connection
		self._process = context.Process(
			target=_cutie_worker_main,
			args=(child_connection, asdict(self.config)),
			name='tdmpc2-live-cutie',
			daemon=True,
		)
		try:
			self._process.start()
		except Exception:
			self._closed = True
			parent_connection.close()
			raise
		finally:
			child_connection.close()
		try:
			ready = self._receive(self.config.worker_timeout_seconds, 'startup')
			self._validate_ready(ready)
		except Exception:
			self._abort()
			raise
		self.ready = dict(ready)

	def _receive(self, timeout: float, phase: str):
		if not self._connection.poll(timeout):
			if not self._process.is_alive():
				raise CutieObjectWorkerError(
					f'Cutie worker exited during {phase} with code '
					f'{self._process.exitcode}.'
				)
			self._timeouts += 1
			raise CutieObjectTimeoutError(
				f'Cutie worker timed out during {phase} after {timeout:g} seconds.'
			)
		try:
			return self._connection.recv()
		except (EOFError, BrokenPipeError, OSError) as exc:
			raise CutieObjectWorkerError(
				f'Cutie worker connection failed during {phase}: {exc}'
			) from exc

	def _validate_ready(self, response):
		if not isinstance(response, dict):
			raise CutieObjectWorkerError('Cutie worker returned a non-dictionary ready response.')
		if response.get('status') == 'error':
			raise CutieObjectWorkerError(
				'Cutie worker initialization failed: '
				f'{response.get("error_type")}: {response.get("error")}\n'
				f'{response.get("traceback", "")}'
			)
		perception_size = (
			self.config.native_highres_size
			if self.config.native_highres_enabled else IMAGE_SIZE
		)
		expected = {
			'status': 'ready',
			'start_method': 'spawn',
			'global_hydra_initialized_before_cutie': False,
			'roles': tuple(self.config.role_names),
			'task': self.config.task,
			'support_task': self.config.task,
			'support_schema': self.config.support_schema,
			'allow_simulator_support': self.config.allow_simulator_support,
			'frame_feature_dim': FRAME_FEATURE_DIM,
			'stacked_feature_dim': STACKED_FEATURE_DIM,
			'permanent_prompts': 6,
			'episode_reset_strategy': 'fresh_inference_core_support_replay_v1',
			'role_diagnostics_schema': 'cutie_role_runtime_diagnostics_v1',
			'device': 'cuda:0',
			'current_device': 0,
			'tracker_size': (
				self.config.tracker_height, self.config.tracker_width,
			),
			'perception_input_source': (
				'same_state_native_dmc_render_with_visual_wrapper_replay_v1'
				if self.config.native_highres_enabled
				else 'policy_observation_latest_rgb_v1'
			),
			'perception_highres_semantics': (
				'native_dmc_foreground_same_state;when_video_background_enabled_'
				'current_native64_background_frame_is_resized_without_clock_advance'
				if self.config.native_highres_enabled else None
			),
			'perception_input_size': (perception_size, perception_size),
			'policy_rgb_size': (IMAGE_SIZE, IMAGE_SIZE),
			'support_input_size': (IMAGE_SIZE, IMAGE_SIZE),
			'support_resolution_policy': 'frozen_native64_support_v1',
			'mask_output_size': (IMAGE_SIZE, IMAGE_SIZE),
			'mask_geometry_resolution': (IMAGE_SIZE, IMAGE_SIZE),
			'native_highres_enabled': self.config.native_highres_enabled,
			'extra_sensor_information': self.config.native_highres_enabled,
			'raw_sensor_information_parity': not self.config.native_highres_enabled,
			'fair_representation_comparison': not self.config.native_highres_enabled,
			'agent_observation_unchanged': True,
			'treatment': (
				'runtime_cutie_input_resolution_only'
				if self.config.native_highres_enabled else 'historical_policy_rgb64'
			),
			'model_size': self.config.model_size,
			'prompt_radius': self.config.prompt_radius,
			'amp': self.config.amp,
			'cudnn_benchmark': False,
			'cudnn_deterministic': True,
		}
		if self.config.true_entity_enabled:
			expected.update({
				'true_entity_enabled': True,
				'tracking_entities': ACROBOT_ENTITY_NAMES,
				'policy_slot_semantics': 'duplicated_whole_entity_structural_queries_v1',
				'spatial_geometry_source': 'native_binary_tracker_mask_before_pooling_v1',
			})
		bad = {
			key: (response.get(key), value)
			for key, value in expected.items()
			if response.get(key) != value
		}
		if response.get('pid') == os.getpid():
			bad['pid'] = (response.get('pid'), 'different from parent')
		if bad:
			raise CutieObjectWorkerError(f'Cutie worker ready contract failed: {bad!r}.')

	def _request(self, op: str, frame: np.ndarray) -> np.ndarray:
		if self._closed:
			raise CutieObjectWorkerError('Cutie worker is already closed.')
		request_id = self._next_request_id
		self._next_request_id += 1
		perception_size = (
			self.config.native_highres_size
			if self.config.native_highres_enabled else IMAGE_SIZE
		)
		frame = np.asarray(frame)
		if (
			frame.shape != (perception_size, perception_size, 3)
			or frame.dtype != np.uint8
		):
			raise CutieObjectWorkerError(
				'Cutie client RGB must be uint8 HWC at the configured perception '
				f'resolution, got {frame.shape} {frame.dtype}.'
			)
		request = {
			'op': op,
			'request_id': request_id,
			'frame': np.array(frame, dtype=np.uint8, order='C', copy=True),
		}
		try:
			self._connection.send(request)
			response = self._receive(self.config.worker_timeout_seconds, op)
		except Exception:
			self._abort()
			raise
		if not isinstance(response, dict) or response.get('request_id') != request_id:
			self._abort()
			raise CutieObjectWorkerError(
				f'Cutie worker returned an unmatched response for request {request_id}.'
			)
		if response.get('status') == 'error':
			self._abort()
			raise CutieObjectWorkerError(
				f'Cutie worker request failed: {response.get("error_type")}: '
				f'{response.get("error")}\n{response.get("traceback", "")}'
			)
		expected_fields = {
			'status', 'request_id', 'object', 'valid', 'lost', 'diagnostics',
			'runtime_ms'
		}
		if self.config.return_masks:
			expected_fields.add('masks')
		if set(response) != expected_fields or response['status'] != 'ok':
			self._abort()
			raise CutieObjectWorkerError(
				f'Cutie worker returned an invalid success response: {response!r}.'
			)
		value = np.asarray(response['object'], dtype=np.float32)
		if (
			value.shape != (len(self.config.role_names), FRAME_FEATURE_DIM)
			or not np.isfinite(value).all()
		):
			self._abort()
			raise CutieObjectWorkerError(
				f'Cutie worker returned invalid object features: {value.shape}.'
			)
		valid = np.asarray(response['valid'], dtype=np.bool_)
		lost = np.asarray(response['lost'], dtype=np.bool_)
		diagnostics = response['diagnostics']
		runtime_ms = float(response['runtime_ms'])
		if (
			valid.shape != (len(self.config.role_names),)
			or lost.shape != (len(self.config.role_names),)
			or not isinstance(diagnostics, Mapping)
			or not np.isfinite(runtime_ms)
			or runtime_ms < 0
		):
			self._abort()
			raise CutieObjectWorkerError('Cutie worker returned invalid diagnostics.')
		expected_diagnostics = {
			'valid': np.bool_,
			'lost': np.bool_,
			'mask_nonempty': np.bool_,
			'feature_finite': np.bool_,
			'mask_area_pixels': np.integer,
			'mask_touches_border': np.bool_,
			'confidence': np.floating,
			'mask_score': np.floating,
		}
		clean_diagnostics = {}
		if set(diagnostics) != set(expected_diagnostics):
			self._abort()
			raise CutieObjectWorkerError(
				f'Cutie worker diagnostic fields are invalid: {set(diagnostics)!r}.'
			)
		for name, kind in expected_diagnostics.items():
			array = np.asarray(diagnostics[name])
			if array.shape != (len(self.config.role_names),) or not np.issubdtype(
				array.dtype, kind
			):
				self._abort()
				raise CutieObjectWorkerError(
					f'Cutie worker diagnostic {name} has invalid shape/dtype: '
					f'{array.shape} {array.dtype}.'
				)
			if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
				self._abort()
				raise CutieObjectWorkerError(
					f'Cutie worker diagnostic {name} must be finite.'
				)
			clean_diagnostics[name] = array.copy()
		if not np.array_equal(clean_diagnostics['valid'], valid) or not np.array_equal(
			clean_diagnostics['lost'], lost
		):
			self._abort()
			raise CutieObjectWorkerError(
				'Cutie worker duplicated valid/lost diagnostics disagree.'
			)
		areas = clean_diagnostics['mask_area_pixels']
		if np.any(areas < 0) or np.any(areas > IMAGE_SIZE * IMAGE_SIZE):
			self._abort()
			raise CutieObjectWorkerError('Cutie mask areas are outside [0,4096].')
		if not np.array_equal(clean_diagnostics['mask_nonempty'], areas > 0):
			self._abort()
			raise CutieObjectWorkerError(
				'Cutie mask_nonempty disagrees with mask_area_pixels.'
			)
		expected_valid = (
			(~clean_diagnostics['lost'])
			& clean_diagnostics['mask_nonempty']
			& clean_diagnostics['feature_finite']
		)
		if not np.array_equal(clean_diagnostics['valid'], expected_valid):
			self._abort()
			raise CutieObjectWorkerError(
				'Cutie valid must equal (~lost)&mask_nonempty&feature_finite.'
			)
		confidence = clean_diagnostics['confidence']
		if np.any(confidence < 0.0) or np.any(confidence > 1.0):
			self._abort()
			raise CutieObjectWorkerError(
				'Cutie diagnostic confidence is outside [0,1].'
			)
		self.last_diagnostics = {
			**clean_diagnostics,
			'runtime_ms': runtime_ms,
		}
		if self.config.return_masks:
			masks = np.asarray(response['masks'])
			if masks.shape != (len(self.config.role_names), IMAGE_SIZE, IMAGE_SIZE) or masks.dtype != np.bool_:
				self._abort()
				raise CutieObjectWorkerError('Cutie returned malformed native role masks.')
			if not np.array_equal(masks.reshape(len(masks), -1).sum(-1), areas):
				self._abort()
				raise CutieObjectWorkerError('Cutie masks disagree with same-response diagnostics.')
			self.last_masks = np.array(masks, order='C', copy=True)
		return np.ascontiguousarray(value)

	def reset_track(self, frame: np.ndarray) -> np.ndarray:
		return self._request('reset_track', frame)

	def track(self, frame: np.ndarray) -> np.ndarray:
		return self._request('track', frame)

	def _abort(self):
		if getattr(self, '_closed', True):
			return
		self._closed = True
		try:
			self._connection.close()
		except OSError:
			pass
		if self._process.is_alive():
			self._process.terminate()
		self._process.join(timeout=5.0)

	def close(self):
		if self._closed:
			return
		request_id = self._next_request_id
		self._next_request_id += 1
		try:
			self._connection.send({'op': 'close', 'request_id': request_id})
			response = self._receive(
				min(10.0, self.config.worker_timeout_seconds), 'close'
			)
			if response != {'status': 'closed', 'request_id': request_id}:
				raise CutieObjectWorkerError(
					f'Cutie worker returned invalid close response: {response!r}.'
				)
		except Exception:
			# Cleanup must still complete; callers receive request-time failures,
			# while close remains safe and idempotent during exception unwinding.
			pass
		finally:
			self._closed = True
			try:
				self._connection.close()
			except OSError:
				pass
			self._process.join(timeout=5.0)
			if self._process.is_alive():
				self._process.terminate()
				self._process.join(timeout=5.0)

	def metrics(self) -> dict[str, int]:
		return {
			'worker_restarts': 0,
			'timeouts': int(self._timeouts),
		}


_MISSING = object()


def _config_value(cfg, key: str, default=_MISSING):
	name = f'cutie_object_{key}'
	try:
		value = cfg.get(name, _MISSING)
	except (AttributeError, KeyError):
		value = getattr(cfg, name, _MISSING)
	if value is not _MISSING:
		return value
	if default is not _MISSING:
		return default
	raise KeyError(f'Missing live Cutie config value {name!r}.')


def _task_name(cfg, default=None):
	try:
		return cfg.get('task', default)
	except (AttributeError, KeyError):
		return getattr(cfg, 'task', default)


def _role_names_value(value) -> tuple[str, ...]:
	if isinstance(value, str):
		raise ValueError(
			'cutie_object_role_names must be a sequence, not a string.'
		)
	try:
		roles = tuple(value)
	except TypeError as exc:
		raise ValueError('cutie_object_role_names must be a sequence.') from exc
	if not roles or len(set(roles)) != len(roles) or any(
		not isinstance(role, str) or not role.strip() for role in roles
	):
		raise ValueError('cutie_object_role_names must contain unique non-empty names.')
	return roles


def _strict_json_int(value, name: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int):
		raise ValueError(f'{name} must be a JSON integer, got {value!r}.')
	return value


def _load_policy_burst_plan(
	path_value, *, task: str, role_names: tuple[str, ...]
) -> dict[str, Any]:
	"""Load a frozen, fully expanded policy-input intervention schedule."""
	if not isinstance(path_value, (str, os.PathLike)) or not str(path_value):
		raise ValueError('cutie_object_policy_burst_plan must be a non-empty path.')
	path = Path(path_value).expanduser().resolve()
	if path.suffix.lower() != '.json' or not path.is_file():
		raise FileNotFoundError(f'Policy burst plan must be an existing JSON file: {path}')
	raw_bytes = path.read_bytes()
	try:
		payload = json.loads(raw_bytes.decode('utf-8'))
	except (UnicodeDecodeError, json.JSONDecodeError) as exc:
		raise ValueError(f'Invalid UTF-8 policy burst plan JSON: {path}') from exc
	if not isinstance(payload, dict):
		raise ValueError('Policy burst plan must be a JSON object.')
	canonical_bytes = (
		json.dumps(
			payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True
		) + '\n'
	).encode('utf-8')
	if raw_bytes != canonical_bytes:
		raise ValueError(
			'Policy burst plan must use canonical sorted compact JSON plus LF.'
		)
	expected_keys = {
		'format', 'task', 'roles', 'episodes', 'decision_steps', 'frame_dim',
		'stack_frames', 'invalid_encoding', 'events',
	}
	if set(payload) != expected_keys:
		raise ValueError(
			'Policy burst plan keys mismatch: '
			f'{sorted(payload)} != {sorted(expected_keys)}.'
		)
	if payload['format'] != POLICY_BURST_FORMAT:
		raise ValueError(f'Unsupported policy burst format {payload["format"]!r}.')
	if payload['task'] != task:
		raise ValueError(f'Policy burst task {payload["task"]!r} != {task!r}.')
	if payload['roles'] != list(role_names):
		raise ValueError(
			f'Policy burst roles {payload["roles"]!r} != {list(role_names)!r}.'
		)
	for key, expected in {
		'episodes': POLICY_BURST_EPISODES,
		'decision_steps': POLICY_BURST_DECISION_STEPS,
		'frame_dim': FRAME_FEATURE_DIM,
		'stack_frames': STACK_FRAMES,
	}.items():
		actual = _strict_json_int(payload[key], f'policy burst {key}')
		if actual != expected:
			raise ValueError(f'Policy burst {key}={actual}, expected {expected}.')
	if payload['invalid_encoding'] != POLICY_BURST_INVALID_ENCODING:
		raise ValueError(
			f'Policy burst invalid_encoding={payload["invalid_encoding"]!r}, '
			f'expected {POLICY_BURST_INVALID_ENCODING!r}.'
		)
	events = payload['events']
	if not isinstance(events, list) or len(events) != POLICY_BURST_EPISODES:
		raise ValueError(
			f'Policy burst plan requires one event for each of '
			f'{POLICY_BURST_EPISODES} episodes.'
		)
	schedule = {}
	normalized_events = []
	for expected_episode, event in enumerate(events):
		if not isinstance(event, dict) or set(event) != {
			'episode_index', 'role', 'start_decision_step', 'length'
		}:
			raise ValueError(f'Invalid policy burst event {event!r}.')
		episode = _strict_json_int(event['episode_index'], 'event episode_index')
		start = _strict_json_int(
			event['start_decision_step'], 'event start_decision_step'
		)
		length = _strict_json_int(event['length'], 'event length')
		role = event['role']
		if episode != expected_episode:
			raise ValueError(
				'Policy burst events must be ordered exactly by episode index; '
				f'position {expected_episode} contains {episode}.'
			)
		if not isinstance(role, str) or role not in role_names:
			raise ValueError(f'Unknown policy burst role {role!r}.')
		if start < 0 or length <= 0 or start + length > POLICY_BURST_DECISION_STEPS:
			raise ValueError(
				f'Policy burst event is outside decisions [0,'
				f'{POLICY_BURST_DECISION_STEPS - 1}]: {event!r}.'
			)
		role_index = role_names.index(role)
		for decision in range(start, start + length):
			key = (episode, decision)
			if key in schedule:
				raise ValueError(f'Overlapping policy burst event at {key}.')
			schedule[key] = role_index
		normalized_events.append({
			'episode_index': episode,
			'role': role,
			'start_decision_step': start,
			'length': length,
		})
	return {
		'path': str(path),
		'sha256': hashlib.sha256(raw_bytes).hexdigest(),
		'task': task,
		'roles': list(role_names),
		'episodes': POLICY_BURST_EPISODES,
		'decision_steps': POLICY_BURST_DECISION_STEPS,
		'invalid_encoding': POLICY_BURST_INVALID_ENCODING,
		'events': normalized_events,
		'schedule': schedule,
		'scheduled_role_frames': len(schedule),
	}


def _worker_config(cfg) -> CutieObjectWorkerConfig:
	config_dir = _config_value(cfg, 'config_dir', None)
	task = _task_name(cfg, 'reacher-visual-small')
	role_contract = _config_value(cfg, 'task_role_contract', 'canonical_v1')
	expected_roles = task_role_names(task, role_contract)
	expected_schema = TASK_SUPPORT_SCHEMAS.get(task, 'whole_arm_goal_v1')
	return CutieObjectWorkerConfig(
		repo_path=str(_config_value(cfg, 'repo')),
		checkpoint_path=str(_config_value(cfg, 'checkpoint')),
		support_path=str(_config_value(cfg, 'support_path')),
		config_dir=None if config_dir is None else str(config_dir),
		device=str(_config_value(cfg, 'device', 'cuda:0')),
		tracker_height=int(_config_value(cfg, 'tracker_height', 448)),
		tracker_width=int(_config_value(cfg, 'tracker_width', 448)),
		native_highres_enabled=_config_value(
			cfg, 'native_highres_enabled', False
		),
		native_highres_size=int(_config_value(cfg, 'native_highres_size', 128)),
		model_size=str(_config_value(cfg, 'model_size', 'small')),
		prompt_radius=float(_config_value(cfg, 'prompt_radius', 2.0)),
		amp=bool(_config_value(cfg, 'amp', True)),
		worker_timeout_seconds=float(
			_config_value(cfg, 'worker_timeout_seconds', 300.0)
		),
		role_names=_role_names_value(
			_config_value(cfg, 'role_names', expected_roles)
		),
		support_schema=str(
			_config_value(cfg, 'support_schema', expected_schema)
		),
		allow_simulator_support=bool(
			_config_value(cfg, 'allow_simulator_support', False)
		),
		task=str(task),
		true_entity_enabled=_config_value(cfg, 'true_entity_enabled', False),
		task_role_contract=role_contract,
		return_masks=(
			cfg.get('cutie_masked_rgb_enabled', False)
			or cfg.get('robust_object_field_enabled', False)
			or cfg.get('cutie_mask_guided_rgb_enabled', False)
		),
	).validated()


@contextmanager
def _preserve_parent_random_state():
	"""Guarantee that worker startup cannot perturb matched agent initialization."""
	python_state = random.getstate()
	numpy_state = np.random.get_state()
	devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
	try:
		with torch.random.fork_rng(devices=devices, enabled=True):
			yield
	finally:
		random.setstate(python_state)
		np.random.set_state(numpy_state)


class CutieObjectWrapper(gym.Wrapper):
	"""Export causal Cutie objects, optionally hiding RGB from the agent.

	The wrapped pixel environment always remains available internally because
	Cutie needs the latest native RGB frame.  ``cutie_hybrid`` returns both RGB
	and objects.  ``cutie_object_only`` returns only the object tensor, so RGB
	never reaches the controller or replay buffer.
	"""

	def __init__(self, env, cfg=None, *, _client=None):
		super().__init__(env)
		mode = cfg.get('flat_anchor_mode', 'cutie_hybrid') if cfg is not None else 'cutie_hybrid'
		if mode not in {'cutie_hybrid', 'cutie_object_only'}:
			raise ValueError(
				'CutieObjectWrapper requires flat_anchor_mode=cutie_hybrid or '
				'cutie_object_only.'
			)
		self._object_only = mode == 'cutie_object_only'
		observation_variant = (
			str(_config_value(cfg, 'observation_variant', 'full'))
			if cfg is not None else 'full'
		)
		if observation_variant not in OBSERVATION_VARIANTS:
			raise ValueError(
				'cutie_object_observation_variant must be one of '
				f'{OBSERVATION_VARIANTS!r}, got {observation_variant!r}.'
			)
		if observation_variant != 'full' and not self._object_only:
			raise ValueError(
				'Diagnostic mask-geometry variants require '
				'flat_anchor_mode=cutie_object_only.'
			)
		self._observation_variant = observation_variant
		self._spatial_token_enabled = bool(
			_config_value(cfg, 'spatial_token_enabled', False)
			if cfg is not None else False
		)
		self._true_entity_enabled = _config_value(cfg, 'true_entity_enabled', False)
		if not isinstance(self._true_entity_enabled, bool):
			raise ValueError('cutie_object_true_entity_enabled must be boolean.')
		if self._true_entity_enabled and (
			not self._spatial_token_enabled
			or _task_name(cfg, 'reacher-visual-small') != 'acrobot-swingup'
		):
			raise ValueError('True-entity control requires Acrobot spatial-token mode.')
		if self._spatial_token_enabled and (
			not self._object_only or observation_variant != 'full'
		):
			raise ValueError(
				'Spatial-token control requires full Cutie object-only observations.'
			)
		configured_frame_schema = (
			str(_config_value(
				cfg, 'frame_schema', OBSERVATION_FRAME_SCHEMAS[observation_variant]
			)) if cfg is not None else OBSERVATION_FRAME_SCHEMAS['full']
		)
		if configured_frame_schema != OBSERVATION_FRAME_SCHEMAS[observation_variant]:
			raise ValueError(
				f'{observation_variant!r} requires cutie_object_frame_schema='
				f'{OBSERVATION_FRAME_SCHEMAS[observation_variant]!r}, got '
				f'{configured_frame_schema!r}.'
			)
		allow_simulator_runtime = (
			_config_value(cfg, 'allow_simulator_runtime', False)
			if cfg is not None else False
		)
		if not isinstance(allow_simulator_runtime, bool):
			raise ValueError('cutie_object_allow_simulator_runtime must be boolean.')
		if (observation_variant == 'gt_mask_geometry') != allow_simulator_runtime:
			raise ValueError(
				'Runtime simulator segmentation must be explicitly enabled exactly for '
				'observation_variant=gt_mask_geometry.'
			)
		self._frame_schema = configured_frame_schema
		self._allow_simulator_runtime = allow_simulator_runtime
		last_valid_memory = (
			_config_value(cfg, 'last_valid_memory', False)
			if cfg is not None else False
		)
		if not isinstance(last_valid_memory, bool):
			raise ValueError('cutie_object_last_valid_memory must be a boolean.')
		if last_valid_memory and not self._object_only:
			raise ValueError(
				'cutie_object_last_valid_memory is restricted to '
				'flat_anchor_mode=cutie_object_only.'
			)
		if observation_variant != 'full' and last_valid_memory:
			raise ValueError('Mask-geometry diagnostics forbid last-valid memory.')
		self._last_valid_memory_enabled = last_valid_memory
		shape = tuple(env.observation_space.shape)
		if shape != (9, IMAGE_SIZE, IMAGE_SIZE):
			raise ValueError(
				'CutieObjectWrapper expects RGB observation shape (9,64,64), '
				f'got {shape}.'
			)
		if np.dtype(env.observation_space.dtype) != np.dtype(np.uint8):
			raise ValueError('CutieObjectWrapper expects a uint8 RGB observation space.')
		native_highres_enabled = (
			_config_value(cfg, 'native_highres_enabled', False)
			if cfg is not None else False
		)
		if not isinstance(native_highres_enabled, bool):
			raise ValueError('cutie_object_native_highres_enabled must be boolean.')
		native_highres_size = (
			_config_value(cfg, 'native_highres_size', 128)
			if cfg is not None else 128
		)
		if (
			not isinstance(native_highres_size, int)
			or isinstance(native_highres_size, bool)
			or native_highres_size not in NATIVE_HIGHRES_SIZES
		):
			raise ValueError(
				'cutie_object_native_highres_size must be 128 or 256.'
			)
		if native_highres_enabled and observation_variant == 'gt_mask_geometry':
			raise ValueError(
				'Cutie native high-resolution RGB is incompatible with the '
				'privileged GT-mask geometry diagnostic.'
			)
		self._native_highres_enabled = native_highres_enabled
		self._native_highres_size = native_highres_size
		self._metric_same_state_render_frames = 0
		self._metric_same_state_render_ms = 0.0
		task = _task_name(cfg, 'reacher-visual-small') if cfg is not None else 'reacher-visual-small'
		self._task = str(task)
		self._spatial_graph = None
		self._spatial_graph_path = None
		self._spatial_graph_sha256 = None
		self._role_names = ROLE_NAMES
		self._support_schema = 'whole_arm_goal_v1'
		if cfg is not None:
			if task not in TASK_ROLE_NAMES:
				raise ValueError(
					'Cutie object mode supports exactly '
					f'{tuple(TASK_ROLE_NAMES)!r}, got {task!r}.'
				)
			expected_roles = task_role_names(
				task, _config_value(cfg, 'task_role_contract', 'canonical_v1')
			)
			configured_roles = _role_names_value(
				_config_value(cfg, 'role_names', expected_roles)
			)
			if configured_roles != expected_roles:
				raise ValueError(
					f'{task} requires cutie_object_role_names={expected_roles!r}, '
					f'got {configured_roles!r}.'
				)
			expected_schema = TASK_SUPPORT_SCHEMAS[task]
			configured_schema = str(
				_config_value(cfg, 'support_schema', expected_schema)
			)
			allowed_schemas = (
				{'whole_arm_goal_v1', 'generic_indexed_v1'}
				if task in {'reacher-visual-small', 'reacher-easy', 'reacher-hard'}
				else {expected_schema}
			)
			if configured_schema not in allowed_schemas:
				raise ValueError(
					f'{task} requires cutie_object_support_schema in '
					f'{tuple(sorted(allowed_schemas))!r}, got {configured_schema!r}.'
				)
			if (
				configured_schema == 'generic_indexed_v1'
				and not bool(_config_value(cfg, 'allow_simulator_support', False))
			):
				raise ValueError(
					'generic_indexed_v1 requires explicit '
					'cutie_object_allow_simulator_support=true.'
				)
			self._role_names = configured_roles
			self._support_schema = configured_schema
			if self._spatial_token_enabled:
				graph_path = _config_value(cfg, 'spatial_graph_path', None)
				if graph_path is None:
					raise ValueError(
						'cutie_object_spatial_graph_path is required for spatial tokens.'
					)
				(
					self._spatial_graph,
					self._spatial_graph_path,
					self._spatial_graph_sha256,
				) = _load_spatial_object_graph(
					graph_path, task=self._task, source_roles=self._role_names
				)
				if self._task == 'acrobot-swingup':
					entities = self._spatial_graph['tracking_entities']
					if (
						len(entities) != 1
						or entities[0].get('name') != 'whole_acrobot'
						or tuple(entities[0].get('source_roles', ())) != self._role_names
					):
						raise ValueError(
							'Acrobot spatial graph must declare one whole_acrobot entity.'
						)
			if bool(cfg.get('multitask', False)):
				raise ValueError('Cutie object mode supports single-task training only.')
			if cfg.get('obs', None) != 'rgb':
				raise ValueError('Cutie object mode requires obs=rgb.')
			if cfg.get('model_size', None) != 5:
				raise ValueError('Cutie object mode is frozen to model_size=5.')
			variable_graph = bool(cfg.get(
				'cutie_object_variable_graph_enabled', False
			))
			if variable_graph and not self._spatial_token_enabled:
				raise ValueError('Variable object graphs require spatial tokens.')
			if variable_graph and len(self._role_names) > int(cfg.get(
				'cutie_object_variable_graph_max_roles', 8
			)):
				raise ValueError('Task role count exceeds variable graph capacity.')
			frozen_dims = {
				'num_roles': len(self._role_names),
				'frame_dim': FRAME_FEATURE_DIM,
				'stack_frames': STACK_FRAMES,
				'input_dim': STACKED_FEATURE_DIM,
			}
			for key, expected in frozen_dims.items():
				actual = int(_config_value(cfg, key, expected))
				if actual != expected:
					raise ValueError(
						f'cutie_object_{key} must be {expected}, got {actual}.'
					)
			if observation_variant != 'full':
				if configured_schema != 'generic_indexed_v1':
					raise ValueError(
						'Diagnostic mask geometry requires generic_indexed_v1 support.'
					)
				if not bool(_config_value(cfg, 'allow_simulator_support', False)):
					raise ValueError(
						'Diagnostic mask geometry requires explicit simulator support opt-in.'
					)
		self._policy_burst_plan = None
		policy_burst_path = (
			_config_value(cfg, 'policy_burst_plan', None)
			if cfg is not None else None
		)
		if policy_burst_path is not None:
			if not self._object_only:
				raise ValueError(
					'Policy burst intervention is restricted to cutie_object_only.'
				)
			self._policy_burst_plan = _load_policy_burst_plan(
				policy_burst_path, task=str(task), role_names=self._role_names
			)
		if observation_variant != 'full' and self._policy_burst_plan is not None:
			raise ValueError('Mask-geometry diagnostics forbid synthetic policy bursts.')
		if self._spatial_token_enabled and (
			self._policy_burst_plan is not None or self._last_valid_memory_enabled
		):
			raise ValueError(
				'Spatial-token control forbids diagnostic burst injection and carry memory.'
			)
		if cfg is not None and observation_variant != 'full' and bool(
			cfg.get('cutie_object_belief_enabled', False)
		):
			raise ValueError('Mask-geometry diagnostics forbid learned belief.')
		client = None
		if _client is None:
			if cfg is None:
				raise ValueError('cfg is required when a fake Cutie client is not injected.')
			if observation_variant == 'gt_mask_geometry':
				if last_valid_memory or self._policy_burst_plan is not None:
					raise ValueError(
						'GT-mask geometry forbids memory and synthetic policy bursts.'
				)
				from envs.wrappers.gt_mask_oracle import GTMaskGeometryClient
				# Match the live-Cutie construction boundary: importing/validating an
				# oracle source may not perturb policy/training RNG streams.
				with _preserve_parent_random_state():
					client = GTMaskGeometryClient(
						env,
						task=str(task),
						role_names=self._role_names,
						support_path=_config_value(cfg, 'support_path'),
					)
			else:
				with _preserve_parent_random_state():
					client = _SpawnCutieClient(_worker_config(cfg))
		else:
			client = _client
		try:
			for method in ('reset_track', 'track', 'close'):
				if not callable(getattr(client, method, None)):
					raise TypeError(f'Cutie client must implement {method}().')
			self._client = client
			self._object_frames = deque(maxlen=STACK_FRAMES)
			self._latest_source_rgb_sha256 = None
			self._closed = False
			self._metric_frames = 0
			self._metric_valid_frames = 0
			self._metric_lost_roles = 0
			self._metric_runtime_ms = 0.0
			self._metric_current_invalid_burst = 0
			self._metric_max_invalid_burst = 0
			role_count = len(self._role_names)
			self._metric_episode_index = -1
			self._metric_episode_step = -1
			self._metric_episode_state = None
			self._metric_episode_records = []
			self._metric_max_invalid_burst_event = None
			self._metric_current_invalid_roles = set()
			self._metric_current_invalid_reasons = set()
			self._metric_role_valid_frames = np.zeros(role_count, dtype=np.int64)
			self._metric_role_lost_frames = np.zeros(role_count, dtype=np.int64)
			self._metric_role_empty_mask_frames = np.zeros(role_count, dtype=np.int64)
			self._metric_role_nonfinite_frames = np.zeros(role_count, dtype=np.int64)
			self._metric_role_border_frames = np.zeros(role_count, dtype=np.int64)
			self._metric_role_area_pixels = np.zeros(role_count, dtype=np.float64)
			self._metric_role_confidence = np.zeros(role_count, dtype=np.float64)
			self._metric_role_mask_score = np.zeros(role_count, dtype=np.float64)
			self._metric_role_current_invalid_burst = np.zeros(
				role_count, dtype=np.int64
			)
			self._metric_role_max_invalid_burst = np.zeros(
				role_count, dtype=np.int64
			)
			self._metric_role_current_reasons = [set() for _ in self._role_names]
			self._metric_role_max_invalid_event = [None for _ in self._role_names]
			if self._policy_burst_plan is not None:
				self._latest_raw_object_frame_sha256 = None
				self._metric_policy_burst_applied = 0
				self._metric_policy_burst_raw_valid_overwritten = 0
				self._metric_policy_burst_raw_invalid_overlap = 0
				self._metric_policy_burst_exact_invalid_checks = 0
				self._metric_policy_burst_non_target_preserved_checks = 0
				self._metric_policy_burst_raw_source_unchanged_checks = 0
				self._metric_policy_burst_hard_zero_content_checks = 0
				self._metric_policy_burst_last_valid_content_checks = 0
				self._metric_policy_burst_memory_substitutions = 0
				self._metric_policy_burst_without_memory_history = 0
				self._metric_policy_stack_transition_checks = 0
				self._metric_policy_raw_latest_trace = hashlib.sha256()
				self._metric_policy_latest_trace = hashlib.sha256()
				self._metric_policy_stack_trace = hashlib.sha256()
				self._metric_policy_burst_role_applied = np.zeros(
					role_count, dtype=np.int64
				)
				self._metric_policy_burst_episodes = [
					{
						**event,
						'scheduled_role_frames': int(event['length']),
						'applied_role_frames': 0,
						'raw_valid_overwritten': 0,
						'raw_invalid_overlap': 0,
						'memory_substitutions': 0,
						'without_memory_history': 0,
					}
					for event in self._policy_burst_plan['events']
				]
			if self._last_valid_memory_enabled:
				# The carried content is strictly episode-local. Aggregate counters
				# remain lifetime metrics, matching the perception runtime counters.
				self._last_valid_content = [None for _ in self._role_names]
				self._last_valid_age = np.zeros(role_count, dtype=np.int64)
				self._metric_last_valid_substitutions = 0
				self._metric_last_valid_invalid_without_history = 0
				self._metric_role_last_valid_substitutions = np.zeros(
					role_count, dtype=np.int64
				)
				self._metric_role_last_valid_invalid_without_history = np.zeros(
					role_count, dtype=np.int64
				)
				self._metric_role_last_valid_max_age = np.zeros(
					role_count, dtype=np.int64
				)
			object_space = gym.spaces.Box(
					low=-np.inf,
					high=np.inf,
					shape=(len(self._role_names), STACKED_FEATURE_DIM),
					dtype=np.float32,
				)
			spaces = {'object': object_space}
			if not self._object_only:
				spaces = {'rgb': env.observation_space, **spaces}
			self.observation_space = gym.spaces.Dict(spaces)
		except Exception:
			if client is not None:
				client.close()
			raise

	def _validate_frame_feature(self, value) -> np.ndarray:
		array = np.asarray(value, dtype=np.float32)
		if array.shape != (len(self._role_names), FRAME_FEATURE_DIM):
			raise CutieObjectWorkerError(
				f'Cutie client feature must be [2,590], got {array.shape}.'
			)
		if not np.isfinite(array).all():
			raise CutieObjectWorkerError('Cutie client feature must be finite.')
		array = np.array(array, dtype=np.float32, order='C', copy=True)
		if self._observation_variant == 'cutie_mask_geometry':
			from envs.wrappers.gt_mask_oracle import zero_query_feature
			array = zero_query_feature(array)
		if self._observation_variant != 'full' and not np.array_equal(
			array[:, :QUERY_POOL_DIM],
			np.zeros((len(self._role_names), QUERY_POOL_DIM), dtype=np.float32),
		):
			raise AssertionError('Mask-geometry observation exposed a nonzero query.')
		return array

	def _spatial_policy_frame(self, feature: np.ndarray) -> np.ndarray:
		if not self._spatial_token_enabled:
			return feature
		if self._true_entity_enabled:
			if not np.array_equal(feature[0], feature[1]):
				raise CutieObjectWorkerError('True-entity structural slots must be identical.')
			return feature
		if self._task == 'acrobot-swingup':
			return (
				feature if feature.shape[0] == 1
				else _whole_acrobot_spatial_frame(feature)
			)
		return feature

	def _reset_last_valid_memory(self):
		"""Clear all episode-local carried object content."""
		for index in range(len(self._role_names)):
			self._last_valid_content[index] = None
		self._last_valid_age.fill(0)

	def _apply_last_valid_memory(self, feature: np.ndarray) -> np.ndarray:
		"""Carry only content for invalid roles while preserving current status.

		The 590-D frame layout is ``content[586] + status[4]``. A valid frame
		replaces that role's episode-local memory. An invalid frame may reuse only
		the stored content; confidence/lost/valid/mask_score always remain those of
		the current tracker result. No history from an earlier episode is eligible.
		"""
		if not self._last_valid_memory_enabled:
			return feature
		if feature.shape != (len(self._role_names), FRAME_FEATURE_DIM):
			raise AssertionError(f'Internal last-valid feature shape error: {feature.shape}.')
		valid = feature[:, FRAME_CONTENT_DIM + 2] > 0.5
		for index, is_valid in enumerate(valid):
			if bool(is_valid):
				self._last_valid_content[index] = feature[
					index, :FRAME_CONTENT_DIM
				].copy()
				self._last_valid_age[index] = 0
				continue
			content = self._last_valid_content[index]
			if content is None:
				self._last_valid_age[index] = 0
				self._metric_last_valid_invalid_without_history += 1
				self._metric_role_last_valid_invalid_without_history[index] += 1
				continue
			self._last_valid_age[index] += 1
			feature[index, :FRAME_CONTENT_DIM] = content
			self._metric_last_valid_substitutions += 1
			self._metric_role_last_valid_substitutions[index] += 1
			self._metric_role_last_valid_max_age[index] = max(
				self._metric_role_last_valid_max_age[index],
				self._last_valid_age[index],
			)
		return feature

	def _update_policy_trace(self, digest, feature: np.ndarray):
		digest.update(
			f'{self._metric_episode_index}:{self._metric_episode_step}:'.encode('ascii')
		)
		digest.update(np.ascontiguousarray(feature, dtype=np.float32).tobytes(order='C'))

	def _apply_policy_burst(self, feature: np.ndarray):
		"""Inject one canonical missing-role frame without mutating raw tracker data."""
		plan = self._policy_burst_plan
		if plan is None:
			return feature, None
		episode = int(self._metric_episode_index)
		decision = int(self._metric_episode_step)
		if episode < 0 or episode >= int(plan['episodes']):
			raise CutieObjectWorkerError(
				f'Policy burst plan has no episode {episode}; evaluation exceeded its plan.'
			)
		raw_snapshot = feature.copy()
		raw_bytes = np.ascontiguousarray(raw_snapshot).tobytes(order='C')
		self._latest_raw_object_frame_sha256 = hashlib.sha256(raw_bytes).hexdigest()
		self._update_policy_trace(self._metric_policy_raw_latest_trace, raw_snapshot)
		role_index = plan['schedule'].get((episode, decision))
		if role_index is None:
			return feature, None
		intervened = feature.copy()
		raw_valid = bool(intervened[role_index, FRAME_CONTENT_DIM + 2] > 0.5)
		intervened[role_index, :FRAME_CONTENT_DIM] = 0.0
		intervened[role_index, FRAME_CONTENT_DIM:] = np.asarray(
			[0.0, 1.0, 0.0, 0.0], dtype=np.float32
		)
		expected_status = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
		if (
			not np.array_equal(
				intervened[role_index, :FRAME_CONTENT_DIM],
				np.zeros(FRAME_CONTENT_DIM, dtype=np.float32),
			)
			or not np.array_equal(
				intervened[role_index, FRAME_CONTENT_DIM:], expected_status
			)
		):
			raise AssertionError('Policy burst did not produce canonical empty_lost_v1.')
		other_roles = [
			index for index in range(len(self._role_names)) if index != role_index
		]
		if not np.array_equal(intervened[other_roles], raw_snapshot[other_roles]):
			raise AssertionError('Policy burst mutated a non-target role.')
		if not np.array_equal(feature, raw_snapshot):
			raise AssertionError('Policy burst mutated the raw tracker feature.')
		self._metric_policy_burst_applied += 1
		self._metric_policy_burst_role_applied[role_index] += 1
		self._metric_policy_burst_exact_invalid_checks += 1
		self._metric_policy_burst_non_target_preserved_checks += 1
		self._metric_policy_burst_raw_source_unchanged_checks += 1
		record = self._metric_policy_burst_episodes[episode]
		record['applied_role_frames'] += 1
		if raw_valid:
			self._metric_policy_burst_raw_valid_overwritten += 1
			record['raw_valid_overwritten'] += 1
		else:
			self._metric_policy_burst_raw_invalid_overlap += 1
			record['raw_invalid_overlap'] += 1
		return intervened, role_index

	def _apply_policy_pipeline(self, feature: np.ndarray) -> np.ndarray:
		"""Apply eval-only burst first, then the same episode-local memory as training."""
		role_index = None
		if self._policy_burst_plan is not None:
			feature, role_index = self._apply_policy_burst(feature)
		cached_content = None
		before_role_substitutions = 0
		if role_index is not None and self._last_valid_memory_enabled:
			content = self._last_valid_content[role_index]
			cached_content = None if content is None else content.copy()
			before_role_substitutions = int(
				self._metric_role_last_valid_substitutions[role_index]
			)
		if self._last_valid_memory_enabled:
			feature = self._apply_last_valid_memory(feature)
		if role_index is not None:
			record = self._metric_policy_burst_episodes[self._metric_episode_index]
			if not self._last_valid_memory_enabled:
				if not np.array_equal(
					feature[role_index, :FRAME_CONTENT_DIM],
					np.zeros(FRAME_CONTENT_DIM, dtype=np.float32),
				):
					raise AssertionError('Hard-zero arm did not retain zero burst content.')
				self._metric_policy_burst_hard_zero_content_checks += 1
			elif cached_content is None:
				if not np.array_equal(
					feature[role_index, :FRAME_CONTENT_DIM],
					np.zeros(FRAME_CONTENT_DIM, dtype=np.float32),
				):
					raise AssertionError('Memory without history changed burst content.')
				self._metric_policy_burst_without_memory_history += 1
				record['without_memory_history'] += 1
			else:
				if not np.array_equal(
					feature[role_index, :FRAME_CONTENT_DIM], cached_content
				):
					raise AssertionError('Last-valid memory did not carry exact content.')
				if int(self._metric_role_last_valid_substitutions[role_index]) != (
					before_role_substitutions + 1
				):
					raise AssertionError('Last-valid substitution accounting drifted.')
				self._metric_policy_burst_last_valid_content_checks += 1
				self._metric_policy_burst_memory_substitutions += 1
				record['memory_substitutions'] += 1
		if self._policy_burst_plan is not None:
			self._update_policy_trace(self._metric_policy_latest_trace, feature)
		return feature

	def _record_policy_stack(self, *, reset: bool, previous=None):
		if self._policy_burst_plan is None:
			return
		frames = tuple(self._object_frames)
		if len(frames) != STACK_FRAMES:
			raise AssertionError('Policy object deque has the wrong length.')
		if reset:
			if not all(np.array_equal(frames[0], value) for value in frames[1:]):
				raise AssertionError('Reset policy deque was not initialized by repetition.')
		else:
			if previous is None or len(previous) != STACK_FRAMES:
				raise AssertionError('Missing previous policy deque for transition check.')
			if (
				not np.array_equal(frames[0], previous[1])
				or not np.array_equal(frames[1], previous[2])
			):
				raise AssertionError('Policy deque did not shift by exactly one frame.')
		self._metric_policy_stack_transition_checks += 1
		stacked = np.concatenate(frames, axis=-1)
		self._update_policy_trace(self._metric_policy_stack_trace, stacked)

	def _observation(self, rgb):
		if len(self._object_frames) != STACK_FRAMES:
			raise CutieObjectWorkerError('Cutie temporal stack is not initialized.')
		stacked = np.concatenate(tuple(self._object_frames), axis=-1)
		if stacked.shape != (len(self._role_names), STACKED_FEATURE_DIM):
			raise AssertionError(f'Internal object stack shape error: {stacked.shape}.')
		observation = {
			'object': torch.from_numpy(np.ascontiguousarray(stacked, dtype=np.float32)),
		}
		if not self._object_only:
			observation = {'rgb': rgb, **observation}
		return observation

	def _episode_metrics_snapshot(self):
		state = self._metric_episode_state
		if state is None:
			return None
		return {
			'episode_index': int(state['episode_index']),
			'frames': int(state['frames']),
			'invalid_frames_any_role': int(state['invalid_frames_any_role']),
			'first_invalid_step': state['first_invalid_step'],
			'max_invalid_burst_any_role': int(state['max_invalid_burst_any_role']),
			'per_role_invalid_frames': {
				role: int(state['per_role_invalid_frames'][index])
				for index, role in enumerate(self._role_names)
			},
			'per_role_max_invalid_burst': {
				role: int(state['per_role_max_invalid_burst'][index])
				for index, role in enumerate(self._role_names)
			},
		}

	def _begin_episode_metrics(self):
		previous = self._episode_metrics_snapshot()
		if previous is not None:
			self._metric_episode_records.append(previous)
		self._metric_episode_index += 1
		self._metric_episode_step = 0
		role_count = len(self._role_names)
		self._metric_episode_state = {
			'episode_index': self._metric_episode_index,
			'frames': 0,
			'invalid_frames_any_role': 0,
			'first_invalid_step': None,
			'current_invalid_burst_any_role': 0,
			'max_invalid_burst_any_role': 0,
			'per_role_invalid_frames': np.zeros(role_count, dtype=np.int64),
			'per_role_current_invalid_burst': np.zeros(role_count, dtype=np.int64),
			'per_role_max_invalid_burst': np.zeros(role_count, dtype=np.int64),
		}
		self._metric_current_invalid_burst = 0
		self._metric_current_invalid_roles.clear()
		self._metric_current_invalid_reasons.clear()
		self._metric_role_current_invalid_burst.fill(0)
		for reasons in self._metric_role_current_reasons:
			reasons.clear()

	def _record_metrics(self, feature: np.ndarray, *, episode_first: bool):
		status_start = QUERY_POOL_DIM + MASK_SPATIAL_DIM
		feature_valid = feature[:, status_start + 2] > 0.5
		feature_lost = feature[:, status_start + 1] > 0.5
		confidence = feature[:, status_start + 0].astype(np.float64)
		mask_score = feature[:, status_start + 3].astype(np.float64)
		area_index = QUERY_POOL_DIM + MASK_POOL_SIZE * MASK_POOL_SIZE + 2
		fallback_area = np.rint(
			feature[:, area_index].astype(np.float64) * IMAGE_SIZE * IMAGE_SIZE
		).astype(np.int64)
		mask_nonempty = fallback_area > 0
		feature_finite = np.ones(len(self._role_names), dtype=np.bool_)
		mask_area_pixels = fallback_area
		mask_touches_border = np.zeros(len(self._role_names), dtype=np.bool_)
		diagnostics = getattr(self._client, 'last_diagnostics', None)
		runtime_ms = 0.0
		if isinstance(diagnostics, Mapping):
			reported_valid = np.asarray(
				diagnostics.get('valid', feature_valid), dtype=np.bool_
			)
			reported_lost = np.asarray(
				diagnostics.get('lost', feature_lost), dtype=np.bool_
			)
			if (
				reported_valid.shape != (len(self._role_names),)
				or reported_lost.shape != (len(self._role_names),)
				or not np.array_equal(reported_valid, feature_valid)
				or not np.array_equal(reported_lost, feature_lost)
			):
				raise CutieObjectWorkerError(
					'Cutie diagnostics disagree with the exported status feature.'
				)
			for name, fallback, dtype in (
				('mask_nonempty', mask_nonempty, np.bool_),
				('feature_finite', feature_finite, np.bool_),
				('mask_area_pixels', mask_area_pixels, np.int64),
				('mask_touches_border', mask_touches_border, np.bool_),
				('confidence', confidence, np.float64),
				('mask_score', mask_score, np.float64),
			):
				value = np.asarray(diagnostics.get(name, fallback), dtype=dtype)
				if value.shape != (len(self._role_names),):
					raise CutieObjectWorkerError(
						f'Cutie diagnostic {name} has invalid shape {value.shape}.'
					)
				if name == 'mask_nonempty':
					mask_nonempty = value
				elif name == 'feature_finite':
					feature_finite = value
				elif name == 'mask_area_pixels':
					mask_area_pixels = value
				elif name == 'mask_touches_border':
					mask_touches_border = value
				elif name == 'confidence':
					confidence = value
				else:
					mask_score = value
			runtime_ms = float(diagnostics.get('runtime_ms', 0.0))
		if not np.isfinite(runtime_ms) or runtime_ms < 0:
			raise CutieObjectWorkerError('Cutie runtime_ms must be finite and non-negative.')
		if episode_first:
			self._begin_episode_metrics()
		else:
			self._metric_episode_step += 1
		state = self._metric_episode_state
		if state is None:
			raise CutieObjectWorkerError('Cutie metrics received a step before reset.')
		self._metric_frames += 1
		self._metric_valid_frames += int(bool(feature_valid.all()))
		self._metric_lost_roles += int(feature_lost.sum())
		self._metric_runtime_ms += runtime_ms
		self._metric_role_valid_frames += feature_valid.astype(np.int64)
		self._metric_role_lost_frames += feature_lost.astype(np.int64)
		self._metric_role_empty_mask_frames += (~mask_nonempty).astype(np.int64)
		self._metric_role_nonfinite_frames += (~feature_finite).astype(np.int64)
		self._metric_role_border_frames += mask_touches_border.astype(np.int64)
		self._metric_role_area_pixels += mask_area_pixels.astype(np.float64)
		self._metric_role_confidence += confidence.astype(np.float64)
		self._metric_role_mask_score += mask_score.astype(np.float64)
		state['frames'] += 1
		invalid = ~feature_valid
		state['per_role_invalid_frames'] += invalid.astype(np.int64)
		state['per_role_current_invalid_burst'][feature_valid] = 0
		state['per_role_current_invalid_burst'][invalid] += 1
		state['per_role_max_invalid_burst'] = np.maximum(
			state['per_role_max_invalid_burst'],
			state['per_role_current_invalid_burst'],
		)
		if bool(feature_valid.all()):
			self._metric_current_invalid_burst = 0
			self._metric_current_invalid_roles.clear()
			self._metric_current_invalid_reasons.clear()
			state['current_invalid_burst_any_role'] = 0
		else:
			self._metric_current_invalid_burst += 1
			state['invalid_frames_any_role'] += 1
			state['current_invalid_burst_any_role'] += 1
			state['max_invalid_burst_any_role'] = max(
				state['max_invalid_burst_any_role'],
				state['current_invalid_burst_any_role'],
			)
			if state['first_invalid_step'] is None:
				state['first_invalid_step'] = int(self._metric_episode_step)
			for index, role in enumerate(self._role_names):
				if not invalid[index]:
					continue
				self._metric_current_invalid_roles.add(role)
				if feature_lost[index]:
					self._metric_current_invalid_reasons.add(f'{role}:lost')
				if not mask_nonempty[index]:
					self._metric_current_invalid_reasons.add(f'{role}:empty_mask')
				if not feature_finite[index]:
					self._metric_current_invalid_reasons.add(f'{role}:nonfinite_feature')
			if self._metric_current_invalid_burst > self._metric_max_invalid_burst:
				self._metric_max_invalid_burst = self._metric_current_invalid_burst
				self._metric_max_invalid_burst_event = {
					'episode_index': int(self._metric_episode_index),
					'start_step': int(
						self._metric_episode_step
						- self._metric_current_invalid_burst + 1
					),
					'end_step': int(self._metric_episode_step),
					'length': int(self._metric_current_invalid_burst),
					'failed_roles': sorted(self._metric_current_invalid_roles),
					'reasons': sorted(self._metric_current_invalid_reasons),
				}
		self._metric_role_current_invalid_burst[feature_valid] = 0
		for index, role in enumerate(self._role_names):
			if feature_valid[index]:
				self._metric_role_current_reasons[index].clear()
				continue
			self._metric_role_current_invalid_burst[index] += 1
			reasons = self._metric_role_current_reasons[index]
			if feature_lost[index]:
				reasons.add('lost')
			if not mask_nonempty[index]:
				reasons.add('empty_mask')
			if not feature_finite[index]:
				reasons.add('nonfinite_feature')
			if (
				self._metric_role_current_invalid_burst[index]
				> self._metric_role_max_invalid_burst[index]
			):
				length = int(self._metric_role_current_invalid_burst[index])
				self._metric_role_max_invalid_burst[index] = length
				self._metric_role_max_invalid_event[index] = {
					'episode_index': int(self._metric_episode_index),
					'start_step': int(self._metric_episode_step - length + 1),
					'end_step': int(self._metric_episode_step),
					'length': length,
					'reasons': sorted(reasons),
				}

	def _perception_rgb(self, policy_observation) -> np.ndarray:
		"""Return Cutie's causal RGB input without changing policy bytes."""
		if not self._native_highres_enabled:
			return _native_hwc_rgb(policy_observation)
		render = getattr(self.env, 'cutie_same_state_rgb', None)
		if not callable(render):
			raise CutieObjectWorkerError(
				'Native high-resolution Cutie input requires the same-state RGB '
				'render protocol on the wrapped pixel environment.'
			)
		size = self._native_highres_size
		started = time.perf_counter()
		frame = np.asarray(render(height=size, width=size))
		if frame.shape != (size, size, 3) or frame.dtype != np.uint8:
			raise CutieObjectWorkerError(
				'Native high-resolution Cutie input must be uint8 HWC RGB, got '
				f'{frame.shape} {frame.dtype}.'
			)
		frame = np.array(frame, dtype=np.uint8, order='C', copy=True)
		self._metric_same_state_render_frames += 1
		self._metric_same_state_render_ms += (
			(time.perf_counter() - started) * 1000.0
		)
		return frame

	def reset(self, **kwargs):
		if self._last_valid_memory_enabled:
			self._reset_last_valid_memory()
		rgb = self.env.reset(**kwargs)
		native_rgb = self._perception_rgb(rgb)
		self._latest_source_rgb_sha256 = hashlib.sha256(
			native_rgb.tobytes(order='C')
		).hexdigest()
		feature = self._validate_frame_feature(
			self._client.reset_track(native_rgb)
		)
		self._record_metrics(feature, episode_first=True)
		feature = self._spatial_policy_frame(feature)
		if self._policy_burst_plan is not None or self._last_valid_memory_enabled:
			feature = self._apply_policy_pipeline(feature)
		self._object_frames.clear()
		for _ in range(STACK_FRAMES):
			self._object_frames.append(feature.copy())
		if self._policy_burst_plan is not None:
			self._record_policy_stack(reset=True)
		return self._observation(rgb)

	def step(self, action):
		if not self._object_frames:
			raise CutieObjectWorkerError('CutieObjectWrapper.step() called before reset().')
		rgb, reward, done, info = self.env.step(action)
		native_rgb = self._perception_rgb(rgb)
		self._latest_source_rgb_sha256 = hashlib.sha256(
			native_rgb.tobytes(order='C')
		).hexdigest()
		feature = self._validate_frame_feature(
			self._client.track(native_rgb)
		)
		self._record_metrics(feature, episode_first=False)
		feature = self._spatial_policy_frame(feature)
		previous = tuple(self._object_frames) if self._policy_burst_plan is not None else None
		if self._policy_burst_plan is not None or self._last_valid_memory_enabled:
			feature = self._apply_policy_pipeline(feature)
		self._object_frames.append(feature)
		if self._policy_burst_plan is not None:
			self._record_policy_stack(reset=False, previous=previous)
		return self._observation(rgb), reward, done, info

	@property
	def cutie_ready(self) -> dict[str, Any] | None:
		ready = getattr(self._client, 'ready', None)
		if not isinstance(ready, dict):
			return None
		return {
			**dict(ready),
			'observation_variant': self._observation_variant,
			'frame_schema': self._frame_schema,
			'privileged_runtime_segmentation': self._allow_simulator_runtime,
			'spatial_token_enabled': self._spatial_token_enabled,
			'true_entity_enabled': self._true_entity_enabled,
			'spatial_graph_path': (
				None if self._spatial_graph_path is None else str(self._spatial_graph_path)
			),
			'spatial_graph_sha256': self._spatial_graph_sha256,
			'whole_acrobot_identity_invariant_union': bool(
				self._spatial_token_enabled and self._task == 'acrobot-swingup'
			),
		}

	@property
	def latest_source_rgb_sha256(self) -> str | None:
		"""Hash Cutie's input frame without exposing or retaining its RGB pixels."""
		return self._latest_source_rgb_sha256

	@property
	def latest_raw_object_frame_sha256(self) -> str | None:
		"""Hash the raw 590-D tracker frame before any policy-side intervention."""
		if self._policy_burst_plan is None:
			return None
		return self._latest_raw_object_frame_sha256

	@property
	def policy_burst_plan_sha256(self) -> str | None:
		return (
			None if self._policy_burst_plan is None
			else str(self._policy_burst_plan['sha256'])
		)

	def metrics(self) -> dict[str, Any]:
		frames = self._metric_frames
		client_metrics = getattr(self._client, 'metrics', None)
		worker = client_metrics() if callable(client_metrics) else {}
		ready = self.cutie_ready or {}
		episodes = list(self._metric_episode_records)
		current_episode = self._episode_metrics_snapshot()
		if current_episode is not None:
			episodes.append(current_episode)
		role_metrics = {}
		for index, role in enumerate(self._role_names):
			role_metrics[role] = {
				'valid_frames': int(self._metric_role_valid_frames[index]),
				'invalid_frames': int(
					frames - self._metric_role_valid_frames[index]
				),
				'lost_frames': int(self._metric_role_lost_frames[index]),
				'empty_mask_frames': int(
					self._metric_role_empty_mask_frames[index]
				),
				'nonfinite_feature_frames': int(
					self._metric_role_nonfinite_frames[index]
				),
				'valid_frame_rate': (
					float(self._metric_role_valid_frames[index] / frames)
					if frames else 0.0
				),
				'lost_frame_rate': (
					float(self._metric_role_lost_frames[index] / frames)
					if frames else 0.0
				),
				'empty_mask_frame_rate': (
					float(self._metric_role_empty_mask_frames[index] / frames)
					if frames else 0.0
				),
				'nonfinite_feature_frame_rate': (
					float(self._metric_role_nonfinite_frames[index] / frames)
					if frames else 0.0
				),
				'mask_touches_border_rate': (
					float(self._metric_role_border_frames[index] / frames)
					if frames else 0.0
				),
				'mean_mask_area_pixels': (
					float(self._metric_role_area_pixels[index] / frames)
					if frames else 0.0
				),
				'mean_confidence': (
					float(self._metric_role_confidence[index] / frames)
					if frames else 0.0
				),
				'mean_mask_score': (
					float(self._metric_role_mask_score[index] / frames)
					if frames else 0.0
				),
				'max_invalid_burst': int(
					self._metric_role_max_invalid_burst[index]
				),
				'max_invalid_burst_event': self._metric_role_max_invalid_event[index],
			}
		if self._last_valid_memory_enabled:
			last_valid_memory = {
				'enabled': True,
				'content_dim': FRAME_CONTENT_DIM,
				'status_dim': STATUS_DIM,
				'substitutions': int(self._metric_last_valid_substitutions),
				'invalid_without_history': int(
					self._metric_last_valid_invalid_without_history
				),
				'per_role': {
					role: {
						'has_memory': self._last_valid_content[index] is not None,
						'age': (
							int(self._last_valid_age[index])
							if self._last_valid_content[index] is not None else None
						),
						'max_age': int(self._metric_role_last_valid_max_age[index]),
						'substitutions': int(
							self._metric_role_last_valid_substitutions[index]
						),
						'invalid_without_history': int(
							self._metric_role_last_valid_invalid_without_history[index]
						),
					}
					for index, role in enumerate(self._role_names)
				},
			}
		else:
			last_valid_memory = {
				'enabled': False,
				'content_dim': FRAME_CONTENT_DIM,
				'status_dim': STATUS_DIM,
				'substitutions': 0,
				'invalid_without_history': 0,
				'per_role': {
					role: {
						'has_memory': False,
						'age': None,
						'max_age': 0,
						'substitutions': 0,
						'invalid_without_history': 0,
					}
					for role in self._role_names
				},
			}
		if self._policy_burst_plan is not None:
			plan = self._policy_burst_plan
			episode_interventions = [dict(value) for value in self._metric_policy_burst_episodes]
			policy_intervention = {
				'enabled': True,
				'format': POLICY_BURST_FORMAT,
				'location': (
					'raw_tracker_frame_after_metrics_before_last_valid_memory_and_stack'
				),
				'invalid_encoding': POLICY_BURST_INVALID_ENCODING,
				'decision_unit': 'agent_decision_observation_index_0_to_499',
				'plan_path': plan['path'],
				'plan_sha256': plan['sha256'],
				'plan_task': plan['task'],
				'plan_roles': list(plan['roles']),
				'scheduled_events': len(plan['events']),
				'applied_events': sum(
					int(value['applied_role_frames'] == value['scheduled_role_frames'])
					for value in episode_interventions
				),
				'scheduled_role_frames': int(plan['scheduled_role_frames']),
				'applied_role_frames': int(self._metric_policy_burst_applied),
				'raw_valid_overwritten': int(
					self._metric_policy_burst_raw_valid_overwritten
				),
				'raw_invalid_overlap': int(
					self._metric_policy_burst_raw_invalid_overlap
				),
				'exact_invalid_checks': int(
					self._metric_policy_burst_exact_invalid_checks
				),
				'non_target_preserved_checks': int(
					self._metric_policy_burst_non_target_preserved_checks
				),
				'raw_source_unchanged_checks': int(
					self._metric_policy_burst_raw_source_unchanged_checks
				),
				'hard_zero_content_checks': int(
					self._metric_policy_burst_hard_zero_content_checks
				),
				'last_valid_content_checks': int(
					self._metric_policy_burst_last_valid_content_checks
				),
				'memory_substitutions': int(
					self._metric_policy_burst_memory_substitutions
				),
				'without_memory_history': int(
					self._metric_policy_burst_without_memory_history
				),
				'policy_stack_transition_checks': int(
					self._metric_policy_stack_transition_checks
				),
				'per_role_applied_frames': {
					role: int(self._metric_policy_burst_role_applied[index])
					for index, role in enumerate(self._role_names)
				},
				'per_episode': episode_interventions,
				'raw_latest_trace_sha256': self._metric_policy_raw_latest_trace.hexdigest(),
				'policy_latest_trace_sha256': self._metric_policy_latest_trace.hexdigest(),
				'policy_stack_trace_sha256': self._metric_policy_stack_trace.hexdigest(),
				'raw_tracker_accounting_excludes_synthetic_intervention': True,
				'live_environment_observation_mutated': False,
			}
		else:
			policy_intervention = {
				'enabled': False,
				'format': POLICY_BURST_FORMAT,
				'location': (
					'raw_tracker_frame_after_metrics_before_last_valid_memory_and_stack'
				),
				'invalid_encoding': POLICY_BURST_INVALID_ENCODING,
				'plan_path': None,
				'plan_sha256': None,
				'scheduled_events': 0,
				'applied_events': 0,
				'scheduled_role_frames': 0,
				'applied_role_frames': 0,
				'raw_tracker_accounting_excludes_synthetic_intervention': True,
				'live_environment_observation_mutated': False,
			}
		oracle_metrics = (
			{
				'format': worker.get('format'),
				'enabled': True,
				'privileged_runtime_segmentation': worker.get(
					'privileged_runtime_segmentation'
				) is True,
				'query_feature_policy': worker.get('query_feature_policy'),
				'frames': int(worker.get('frames', 0)),
				'ms_per_frame': float(worker.get('ms_per_frame', 0.0)),
				'visibility_failures': dict(worker.get('visibility_failures', {})),
				'geom_catalog_path': ready.get('geom_catalog_path'),
				'geom_catalog_sha256': ready.get('geom_catalog_sha256'),
				'camera_id': ready.get('camera_id'),
				'image_size': ready.get('image_size'),
			}
			if self._observation_variant == 'gt_mask_geometry'
			else {
				'enabled': False,
				'privileged_runtime_segmentation': False,
				'query_feature_policy': (
					'exact_zero_512_v1'
					if self._observation_variant == 'cutie_mask_geometry'
					else 'cutie_query_mean_std_v1'
				),
			}
		)
		return {
			'frames': int(frames),
			'invalid_frames': int(frames - self._metric_valid_frames),
			'valid_frame_rate': (
				float(self._metric_valid_frames / frames) if frames else 0.0
			),
			'lost_role_rate': (
				float(self._metric_lost_roles / (frames * len(self._role_names)))
				if frames else 0.0
			),
			'max_invalid_burst': int(self._metric_max_invalid_burst),
			'max_invalid_burst_event': self._metric_max_invalid_burst_event,
			'role_diagnostics_schema': 'cutie_role_runtime_diagnostics_v1',
			'perception_input_source': ready.get(
				'perception_input_source',
				(
					'same_state_native_dmc_render_with_visual_wrapper_replay_v1'
					if self._native_highres_enabled
					else 'policy_observation_latest_rgb_v1'
				),
			),
			'perception_input_size': list(ready.get(
				'perception_input_size',
				(
					(self._native_highres_size, self._native_highres_size)
					if self._native_highres_enabled
					else (IMAGE_SIZE, IMAGE_SIZE)
				),
			)),
			'policy_rgb_size': list(ready.get(
				'policy_rgb_size', (IMAGE_SIZE, IMAGE_SIZE)
			)),
			'support_input_size': list(ready.get(
				'support_input_size', (IMAGE_SIZE, IMAGE_SIZE)
			)),
			'support_resolution_policy': ready.get(
				'support_resolution_policy', 'frozen_native64_support_v1'
			),
			'mask_output_size': list(ready.get(
				'mask_output_size', (IMAGE_SIZE, IMAGE_SIZE)
			)),
			'mask_geometry_resolution': list(ready.get(
				'mask_geometry_resolution', (IMAGE_SIZE, IMAGE_SIZE)
			)),
			'perception_highres_semantics': ready.get(
				'perception_highres_semantics',
				(
					'native_dmc_foreground_same_state;when_video_background_enabled_'
					'current_native64_background_frame_is_resized_without_clock_advance'
					if self._native_highres_enabled else None
				),
			),
			'extra_sensor_information': bool(ready.get(
				'extra_sensor_information', self._native_highres_enabled
			)),
			'raw_sensor_information_parity': bool(ready.get(
				'raw_sensor_information_parity', not self._native_highres_enabled
			)),
			'fair_representation_comparison': bool(ready.get(
				'fair_representation_comparison', not self._native_highres_enabled
			)),
			'agent_observation_unchanged': bool(ready.get(
				'agent_observation_unchanged', True
			)),
			'treatment': ready.get(
				'treatment',
				(
					'runtime_cutie_input_resolution_only'
					if self._native_highres_enabled else 'historical_policy_rgb64'
				),
			),
			'same_state_render_frames': int(
				self._metric_same_state_render_frames
			),
			'same_state_render_ms_per_frame': (
				float(
					self._metric_same_state_render_ms
					/ self._metric_same_state_render_frames
				)
				if self._metric_same_state_render_frames else 0.0
			),
			'observation_variant': self._observation_variant,
			'frame_schema': self._frame_schema,
			'privileged_runtime_segmentation': self._allow_simulator_runtime,
			'gt_mask_oracle': oracle_metrics,
			'true_entity_enabled': self._true_entity_enabled,
			'role_metrics_semantics': (
				'duplicated_whole_entity_measurement_not_link_detection_v1'
				if self._true_entity_enabled else 'independent_role_measurements_v1'
			),
			'entity_metrics': (
				{'whole_acrobot': dict(role_metrics[self._role_names[0]])}
				if self._true_entity_enabled else {}
			),
			'role_metrics': role_metrics,
			'last_valid_memory': last_valid_memory,
			'policy_observation_intervention': policy_intervention,
			'episode_metrics': episodes,
			'ms_per_frame': (
				float(self._metric_runtime_ms / frames) if frames else 0.0
			),
			'runtime_unit': (
				'milliseconds_per_same_state_segmentation_frame'
				if self._observation_variant == 'gt_mask_geometry'
				else 'milliseconds_per_tracked_frame_excluding_support_prompts'
			),
			'worker_restarts': int(worker.get('worker_restarts', 0)),
			'timeouts': int(worker.get('timeouts', 0)),
			'episode_reset_strategy': ready.get('episode_reset_strategy'),
			'device_name': ready.get('device_name'),
			'cuda_visible_devices': ready.get('cuda_visible_devices'),
			'logical_cuda_device': ready.get('current_device'),
		}

	def close(self):
		if self._closed:
			return
		self._closed = True
		try:
			self._client.close()
		finally:
			super().close()


__all__ = [
	'CutieObjectError',
	'CutieObjectTimeoutError',
	'CutieObjectWorkerConfig',
	'CutieObjectWorkerError',
	'CutieObjectWrapper',
	'FRAME_CONTENT_DIM',
	'FRAME_FEATURE_DIM',
	'ROLE_NAMES',
	'STACKED_FEATURE_DIM',
	'TASK_ROLE_NAMES',
	'TASK_SUPPORT_SCHEMAS',
	'_whole_acrobot_spatial_frame',
	'generic_object_frame',
]
