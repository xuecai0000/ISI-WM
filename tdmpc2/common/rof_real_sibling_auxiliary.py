"""Training-only real-sibling auxiliary for ROF-WM.

The auxiliary consumes a separately materialized capsule containing only
deployable ROF observations and actions that were genuinely executed from one
shared simulator state.  Simulator state, rewards, segmentation ground truth,
and every ``oracle__`` array are forbidden from the capsule.

This module deliberately does not collect branches and does not change online
observations.  Its only job is to validate/sample the immutable training
capsule and to define the scale-aware latent ranking objective used by the
positive-only and relational training arms.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F


FORMAT = 'rof_real_sibling_auxiliary_training_capsule_v1'
STATUS = 'rof_real_sibling_auxiliary_training_capsule_complete'
CONDITION = 'clean'
POLICY_FIELDS = ('rgb', 'object', 'object_mask', 'role_exists')
ROOT_KEYS = tuple(f'root__policy_{field}' for field in POLICY_FIELDS)
FUTURE_KEYS = tuple(f'future__policy_{field}' for field in POLICY_FIELDS)
ACTION_KEYS = ('branch__action', 'branch__code')
SHARD_KEYS = frozenset(ROOT_KEYS + FUTURE_KEYS + ACTION_KEYS)
MAX_HORIZON = 5

_DEFAULTS = {
	'rof_real_sibling_aux_manifest': None,
	'rof_real_sibling_aux_manifest_sha256': None,
	'rof_real_sibling_aux_capsule_format': None,
	'rof_real_sibling_aux_source_dataset_sha256': None,
	'rof_real_sibling_aux_condition': CONDITION,
	'rof_real_sibling_aux_positive_coef': 1.0,
	'rof_real_sibling_aux_ranking_coef': 0.0,
	'rof_real_sibling_aux_margin_fraction': 0.1,
	'rof_real_sibling_aux_batch_size': 8,
	'rof_real_sibling_aux_horizon': 3,
	'rof_real_sibling_aux_update_frequency': 1,
	'rof_real_sibling_aux_seed_offset': 15485863,
}


def _get(cfg, key: str, default=None):
	return cfg.get(key, default) if hasattr(cfg, 'get') else getattr(cfg, key, default)


def _require(condition: bool, message: str) -> None:
	if not condition:
		raise ValueError(message)


def _strict_positive_int(value, name: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value < 1:
		raise ValueError(f'{name} must be a positive integer.')
	return int(value)


def _finite_nonnegative(value, name: str) -> float:
	if isinstance(value, bool):
		raise ValueError(f'{name} must be a finite non-negative number.')
	try:
		result = float(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(
			f'{name} must be a finite non-negative number.'
		) from exc
	if not math.isfinite(result) or result < 0.0:
		raise ValueError(f'{name} must be a finite non-negative number.')
	return result


def enabled(cfg) -> bool:
	value = _get(cfg, 'rof_real_sibling_aux_enabled', False)
	if type(value) is not bool:
		raise ValueError('rof_real_sibling_aux_enabled must be a strict boolean.')
	return value


def file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with Path(path).open('rb') as handle:
		for block in iter(lambda: handle.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def branch_codes(action_dim: int) -> np.ndarray:
	values = [0]
	for axis in range(action_dim):
		values.extend((axis + 1, -(axis + 1)))
	return np.asarray(values, dtype=np.int32)


def balanced_ordered_code_pairs(action_dim: int) -> np.ndarray:
	"""Return the preregistered axis-balanced ordered sibling relations."""
	_strict_positive_int(action_dim, 'action_dim')
	pairs = []
	for axis in range(action_dim):
		positive_code = axis + 1
		negative_code = -(axis + 1)
		for left, right in (
			(0, positive_code),
			(0, negative_code),
			(positive_code, negative_code),
		):
			pairs.extend(((left, right), (right, left)))
	return np.asarray(pairs, dtype=np.int64)


def validate_config(cfg, *, require_bound_manifest: bool = False) -> dict | None:
	"""Validate the opt-in contract without perturbing disabled runs."""
	if not enabled(cfg):
		# A path or serialized identity while disabled is almost certainly a stale
		# runner configuration.  Numeric defaults remain inert and are accepted.
		for key in (
			'rof_real_sibling_aux_manifest',
			'rof_real_sibling_aux_manifest_sha256',
			'rof_real_sibling_aux_capsule_format',
			'rof_real_sibling_aux_source_dataset_sha256',
		):
			if _get(cfg, key, _DEFAULTS[key]) is not None:
				raise ValueError(f'{key} requires rof_real_sibling_aux_enabled=true.')
		return None

	if not bool(_get(cfg, 'robust_object_field_enabled', False)):
		raise ValueError('Real-sibling auxiliary training requires ROF-WM.')
	if not (
		bool(_get(cfg, 'flat_anchor', False))
		and _get(cfg, 'flat_anchor_mode', None) == 'cutie_object_only'
		and not bool(_get(cfg, 'multitask', False))
	):
		raise ValueError(
			'Real-sibling auxiliary training requires single-task '
			'flat_anchor_mode=cutie_object_only.'
		)
	if _get(cfg, 'rof_real_sibling_aux_condition', CONDITION) != CONDITION:
		raise ValueError('The first real-sibling pilot is clean-training only.')
	if bool(_get(cfg, 'compile', False)):
		raise ValueError(
			'The preregistered A/B/C screen requires compile=false for every arm.'
		)

	positive_coef = _finite_nonnegative(
		_get(cfg, 'rof_real_sibling_aux_positive_coef', 1.0),
		'rof_real_sibling_aux_positive_coef',
	)
	if positive_coef <= 0.0:
		raise ValueError('Real-sibling positive coefficient must be greater than zero.')
	ranking_coef = _finite_nonnegative(
		_get(cfg, 'rof_real_sibling_aux_ranking_coef', 0.0),
		'rof_real_sibling_aux_ranking_coef',
	)
	margin_fraction = _finite_nonnegative(
		_get(cfg, 'rof_real_sibling_aux_margin_fraction', 0.1),
		'rof_real_sibling_aux_margin_fraction',
	)
	if margin_fraction > 1.0:
		raise ValueError('Real-sibling margin fraction must lie in [0,1].')
	batch_size = _strict_positive_int(
		_get(cfg, 'rof_real_sibling_aux_batch_size', 8),
		'rof_real_sibling_aux_batch_size',
	)
	horizon = _strict_positive_int(
		_get(cfg, 'rof_real_sibling_aux_horizon', 3),
		'rof_real_sibling_aux_horizon',
	)
	if horizon > MAX_HORIZON or horizon != int(_get(cfg, 'horizon', horizon)):
		raise ValueError(
			'Real-sibling horizon must equal the TD-MPC2 horizon and be at most '
			f'{MAX_HORIZON}.'
		)
	update_frequency = _strict_positive_int(
		_get(cfg, 'rof_real_sibling_aux_update_frequency', 1),
		'rof_real_sibling_aux_update_frequency',
	)
	seed_offset = _strict_positive_int(
		_get(cfg, 'rof_real_sibling_aux_seed_offset', 15485863),
		'rof_real_sibling_aux_seed_offset',
	)
	expected = {
		'rof_real_sibling_aux_positive_coef': (positive_coef, 1.0),
		'rof_real_sibling_aux_margin_fraction': (margin_fraction, 0.1),
		'rof_real_sibling_aux_batch_size': (batch_size, 8),
		'rof_real_sibling_aux_horizon': (horizon, 3),
		'rof_real_sibling_aux_update_frequency': (update_frequency, 1),
		'rof_real_sibling_aux_seed_offset': (seed_offset, 15485863),
	}
	bad = {
		key: (actual, required)
		for key, (actual, required) in expected.items()
		if actual != required
	}
	if ranking_coef not in (0.0, 1.0):
		bad['rof_real_sibling_aux_ranking_coef'] = (
			ranking_coef, '0.0 (B) or 1.0 (C)'
		)
	if bad:
		raise ValueError(
			f'Real-sibling screen drifted from its preregistered constants: {bad}.'
		)
	manifest_sha = _get(cfg, 'rof_real_sibling_aux_manifest_sha256', None)
	capsule_format = _get(cfg, 'rof_real_sibling_aux_capsule_format', None)
	source_dataset_sha = _get(
		cfg, 'rof_real_sibling_aux_source_dataset_sha256', None
	)
	if manifest_sha is not None:
		_require(
			isinstance(manifest_sha, str) and len(manifest_sha) == 64
			and all(char in '0123456789abcdef' for char in manifest_sha),
			'Bound real-sibling manifest SHA-256 is malformed.',
		)
	if capsule_format is not None:
		_require(capsule_format == FORMAT, 'Bound real-sibling capsule format changed.')
	if source_dataset_sha is not None:
		_require(
			isinstance(source_dataset_sha, str) and len(source_dataset_sha) == 64
			and all(char in '0123456789abcdef' for char in source_dataset_sha),
			'Bound source sibling dataset SHA-256 is malformed.',
		)
	if require_bound_manifest and (
		manifest_sha is None or capsule_format is None
		or source_dataset_sha is None
	):
		raise ValueError('Real-sibling training requires a bound capsule identity.')
	return {
		'format': 'rof_real_sibling_auxiliary_contract_v1',
		'arm': (
			'sibling_positive_control'
			if ranking_coef == 0.0 else 'sibling_relational'
		),
		'condition': CONDITION,
		'task': str(_get(cfg, 'task', '')),
		'ordered_roles': list(_get(cfg, 'cutie_object_role_names', ())),
		'action_dim': int(_get(cfg, 'action_dim', 0)),
		'positive_coef': positive_coef,
		'ranking_coef': ranking_coef,
		'margin_fraction': margin_fraction,
		'batch_size': batch_size,
		'horizon': horizon,
		'update_frequency': update_frequency,
		'seed_offset': seed_offset,
		'augmentation_seed_offset': seed_offset + 1,
		'manifest_sha256': manifest_sha,
		'capsule_format': capsule_format,
		'source_dataset_manifest_sha256': source_dataset_sha,
		'online_observation_changed': False,
		'oracle_model_inputs': False,
		'real_executed_siblings_only': True,
		'sampler': 'balanced_axis_relation_shuffled_bag_v1',
	}


def validate_checkpoint_record(record, expected_contract: dict | None) -> dict[str, int]:
	"""Validate checkpoint provenance without opening the training capsule."""
	if expected_contract is None:
		if record is not None:
			raise RuntimeError(
				'Real-sibling checkpoint cannot load with its auxiliary disabled.'
			)
		return {'attempts': 0, 'successful_updates': 0, 'sampled_root_pairs': 0}
	if (
		not isinstance(record, dict)
		or set(record) != {'contract', 'supervision'}
		or record.get('contract') != expected_contract
	):
		raise RuntimeError('Real-sibling checkpoint contract mismatch.')
	supervision = record.get('supervision')
	if not isinstance(supervision, dict):
		raise RuntimeError('Real-sibling checkpoint supervision record is malformed.')
	values = {}
	for key in ('attempts', 'successful_updates', 'sampled_root_pairs'):
		value = supervision.get(key)
		if not isinstance(value, int) or isinstance(value, bool) or value < 1:
			raise RuntimeError(
				f'Real-sibling checkpoint {key} must be a positive integer.'
			)
		values[key] = int(value)
	if values['successful_updates'] != values['attempts']:
		raise RuntimeError(
			'Frequency-one sibling screen requires one auxiliary term per attempt.'
		)
	expected_samples = (
		values['successful_updates'] * int(expected_contract['batch_size'])
	)
	if values['sampled_root_pairs'] != expected_samples:
		raise RuntimeError(
			'Real-sibling sampled pair count is inconsistent with its update and '
			'batch-size contract.'
		)
	metrics = supervision.get('sampler_metrics')
	if not isinstance(metrics, dict) or any((
		metrics.get('sample_calls') != values['successful_updates'],
		metrics.get('sampled_root_pairs') != expected_samples,
		not isinstance(metrics.get('root_count'), int),
		metrics.get('root_count', 0) < 1,
	)):
		raise RuntimeError('Real-sibling sampler checkpoint metrics are malformed.')
	count_maps = (
		'root_sample_counts', 'positive_code_counts', 'negative_code_counts',
		'ordered_code_pair_counts', 'relation_type_counts',
	)
	for key in count_maps:
		counts = metrics.get(key)
		if (
			not isinstance(counts, dict) or not counts
			or any(
				not isinstance(value, int) or isinstance(value, bool) or value < 0
				for value in counts.values()
			)
			or sum(counts.values()) != expected_samples
		):
			raise RuntimeError(f'Real-sibling sampler {key} is inconsistent.')
	expected_relations = {
		'zero_to_positive_axis', 'zero_to_negative_axis',
		'opposite_same_axis',
	}
	if set(metrics['relation_type_counts']) != expected_relations:
		raise RuntimeError('Real-sibling relation-type counts changed schema.')
	action_dim = int(expected_contract['action_dim'])
	allowed_codes = {int(code) for code in branch_codes(action_dim)}
	if any(
		int(code) not in allowed_codes
		for key in ('positive_code_counts', 'negative_code_counts')
		for code in metrics[key]
	):
		raise RuntimeError('Real-sibling sampler recorded an unknown action code.')
	allowed_pairs = {
		f'{int(left)}->{int(right)}'
		for left, right in balanced_ordered_code_pairs(action_dim)
	}
	if not set(metrics['ordered_code_pair_counts']).issubset(allowed_pairs):
		raise RuntimeError('Real-sibling sampler recorded a forbidden action pair.')
	return values


def _safe_asset_path(root: Path, relative: object) -> Path:
	if not isinstance(relative, str) or not relative:
		raise ValueError('Sibling capsule asset path is malformed.')
	path = Path(relative)
	if path.is_absolute() or '..' in path.parts:
		raise ValueError('Sibling capsule asset path must stay relative to its root.')
	resolved = (root / path).resolve()
	try:
		resolved.relative_to(root.resolve())
	except ValueError as exc:
		raise ValueError('Sibling capsule asset escaped its root.') from exc
	if resolved.is_symlink():
		raise ValueError('Symlinked sibling capsule assets are forbidden.')
	return resolved


def _require_array(
	arrays: Mapping[str, np.ndarray], name: str, shape: tuple[int, ...], dtype,
) -> np.ndarray:
	if name not in arrays:
		raise ValueError(f'Sibling capsule is missing {name!r}.')
	value = np.asarray(arrays[name])
	if value.shape != shape or value.dtype != np.dtype(dtype):
		raise ValueError(
			f'{name} must be {np.dtype(dtype)} {shape}, got '
			f'{value.dtype} {value.shape}.'
		)
	if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
		raise ValueError(f'{name} contains non-finite values.')
	return value


def validate_shard(
	arrays: Mapping[str, np.ndarray], *, roles: int, action_dim: int,
	horizon: int, branch_magnitude: float,
) -> dict[str, int]:
	"""Reject malformed, incomplete, fake, or oracle-bearing sibling shards."""
	if not isinstance(arrays, Mapping) or set(arrays) != SHARD_KEYS:
		actual = sorted(arrays) if isinstance(arrays, Mapping) else type(arrays)
		raise ValueError(
			f'Sibling capsule arrays must be exactly {sorted(SHARD_KEYS)!r}, '
			f'got {actual!r}.'
		)
	if any(str(key).startswith('oracle__') for key in arrays):
		raise ValueError('Oracle arrays are forbidden from sibling training capsules.')
	_strict_positive_int(roles, 'roles')
	_strict_positive_int(action_dim, 'action_dim')
	_strict_positive_int(horizon, 'horizon')
	if horizon > MAX_HORIZON:
		raise ValueError('Sibling capsule horizon exceeds the supported maximum.')
	if not math.isfinite(float(branch_magnitude)) or not 0.0 < float(branch_magnitude) <= 1.0:
		raise ValueError('Sibling branch magnitude must lie in (0,1].')
	branches = 1 + 2 * action_dim
	shapes = {
		'rgb': (9, 64, 64),
		'object': (roles, 1770),
		'object_mask': (roles, 3, 64, 64),
		'role_exists': (roles,),
	}
	dtypes = {
		'rgb': np.uint8,
		'object': np.float32,
		'object_mask': np.bool_,
		'role_exists': np.float32,
	}
	for field in POLICY_FIELDS:
		root = _require_array(
			arrays, f'root__policy_{field}', shapes[field], dtypes[field],
		)
		future = _require_array(
			arrays, f'future__policy_{field}',
			(branches, horizon) + shapes[field], dtypes[field],
		)
		if field == 'role_exists':
			if not np.array_equal(root, np.ones_like(root)):
				raise ValueError('ROF sibling root must contain exact-K real roles.')
			if not np.array_equal(future, np.ones_like(future)):
				raise ValueError('ROF sibling futures must contain exact-K real roles.')
	actions = _require_array(
		arrays, 'branch__action', (branches, horizon, action_dim), np.float32,
	)
	codes = _require_array(arrays, 'branch__code', (branches,), np.int32)
	expected_codes = branch_codes(action_dim)
	if not np.array_equal(codes, expected_codes):
		raise ValueError('Sibling branch codes are not zero plus/minus every basis.')
	expected_action = np.zeros((branches, action_dim), dtype=np.float32)
	row = 1
	for axis in range(action_dim):
		expected_action[row, axis] = np.float32(branch_magnitude)
		expected_action[row + 1, axis] = np.float32(-branch_magnitude)
		row += 2
	if not np.array_equal(actions[:, 0], expected_action):
		raise ValueError('Sibling interventions are not the declared real basis actions.')
	if horizon > 1 and not np.array_equal(
		actions[:, 1:],
		np.broadcast_to(actions[0:1, 1:], actions[:, 1:].shape),
	):
		raise ValueError('Sibling continuation actions must be identical.')
	if np.any(actions < -1.0) or np.any(actions > 1.0):
		raise ValueError('Sibling actions left the scaled [-1,1] action space.')
	return {'branches': branches, 'horizon': horizon}


def validate_manifest(path: Path, cfg=None) -> dict[str, Any]:
	path = Path(path).resolve()
	if not path.is_file() or path.is_symlink():
		raise FileNotFoundError(path)
	payload = json.loads(path.read_text(encoding='utf-8'))
	_require(isinstance(payload, dict), 'Sibling capsule manifest must be an object.')
	_require(payload.get('format') == FORMAT, 'Sibling capsule format mismatch.')
	_require(payload.get('status') == STATUS, 'Sibling capsule is incomplete.')
	_require(
		payload.get('controller_auxiliary_training_authorized') is True,
		'Sibling capsule lacks explicit auxiliary-training authorization.',
	)
	_require(payload.get('condition') == CONDITION, 'Sibling capsule must be clean-only.')
	_require(payload.get('source_split') == 'train', 'Sibling capsule must contain train roots only.')
	_require(payload.get('model_input_only') is True, 'Sibling capsule is not model-input-only.')
	_require(payload.get('oracle_arrays_present') is False, 'Sibling capsule declares oracle arrays.')
	_require(
		payload.get('root_grouping') == 'complete_real_action_sibling_family',
		'Sibling root grouping contract changed.',
	)
	_require(
		payload.get('fake_shuffled_action_futures') is False,
		'Fake action/future pairing is forbidden.',
	)
	source_dataset = payload.get('source_dataset')
	_require(isinstance(source_dataset, dict), 'Source sibling dataset identity is missing.')
	_require(
		source_dataset.get('format') == 'rof_same_state_action_branch_dataset_v1'
		and isinstance(source_dataset.get('manifest_sha256'), str)
		and len(source_dataset['manifest_sha256']) == 64
		and source_dataset.get('controller_training_authorized') is False
		and source_dataset.get(
			'validation_performed_before_privilege_stripping'
		) is True,
		'Source sibling dataset identity is malformed.',
	)
	roles = payload.get('num_roles')
	action_dim = payload.get('action_dim')
	horizon = payload.get('horizon')
	branch_magnitude = payload.get('branch_magnitude')
	_strict_positive_int(roles, 'num_roles')
	_strict_positive_int(action_dim, 'action_dim')
	_strict_positive_int(horizon, 'horizon')
	groups = payload.get('groups')
	_require(isinstance(groups, list) and groups, 'Sibling capsule has no root groups.')
	root = path.parent
	seen = set()
	validated = []
	for record in groups:
		_require(isinstance(record, dict), 'Sibling capsule group record is malformed.')
		root_id = record.get('root_id')
		_require(
			isinstance(root_id, int) and not isinstance(root_id, bool)
			and root_id >= 0 and root_id not in seen,
			f'Invalid or duplicate sibling root id {root_id!r}.',
		)
		seen.add(root_id)
		asset = _safe_asset_path(root, record.get('relative_path'))
		_require(asset.is_file(), f'Sibling shard is missing: {asset}')
		expected_sha = record.get('sha256')
		_require(
			isinstance(expected_sha, str) and len(expected_sha) == 64,
			f'Sibling root {root_id} has no SHA-256.',
		)
		_require(file_sha256(asset) == expected_sha, f'Sibling root {root_id} SHA-256 mismatch.')
		with np.load(asset, allow_pickle=False) as archive:
			arrays = {key: archive[key] for key in archive.files}
		validate_shard(
			arrays, roles=roles, action_dim=action_dim, horizon=horizon,
			branch_magnitude=float(branch_magnitude),
		)
		validated.append((root_id, asset))
	if cfg is not None:
		_require(payload.get('task') == _get(cfg, 'task'), 'Sibling capsule task mismatch.')
		_require(
			payload.get('role_names') == list(_get(cfg, 'cutie_object_role_names', ())),
			'Sibling capsule ordered roles mismatch.',
		)
		_require(roles == int(_get(cfg, 'cutie_object_num_roles', 0)), 'Sibling role count mismatch.')
		_require(action_dim == int(_get(cfg, 'action_dim', 0)), 'Sibling action width mismatch.')
		requested_horizon = int(_get(cfg, 'rof_real_sibling_aux_horizon', 3))
		_require(horizon >= requested_horizon, 'Sibling capsule is shorter than the requested horizon.')
		bound_sha = _get(cfg, 'rof_real_sibling_aux_manifest_sha256', None)
		actual_sha = file_sha256(path)
		if bound_sha is not None:
			_require(bound_sha == actual_sha, 'Bound sibling manifest SHA-256 mismatch.')
		bound_format = _get(cfg, 'rof_real_sibling_aux_capsule_format', None)
		if bound_format is not None:
			_require(bound_format == FORMAT, 'Bound sibling capsule format mismatch.')
		bound_source = _get(
			cfg, 'rof_real_sibling_aux_source_dataset_sha256', None
		)
		if bound_source is not None:
			_require(
				bound_source == source_dataset['manifest_sha256'],
				'Bound source sibling dataset SHA-256 mismatch.',
			)
	payload['_validated_groups'] = tuple(validated)
	payload['_manifest_sha256'] = file_sha256(path)
	return payload


@dataclass(frozen=True)
class SiblingBatch:
	root: Mapping[str, torch.Tensor]
	positive_future: Mapping[str, torch.Tensor]
	negative_future: Mapping[str, torch.Tensor]
	action: torch.Tensor
	root_id: torch.Tensor
	positive_code: torch.Tensor
	negative_code: torch.Tensor


class Replay:
	"""Isolated deterministic sampler over complete real-sibling root groups."""

	def __init__(self, cfg, *, device):
		path = _get(cfg, 'rof_real_sibling_aux_manifest', None)
		if path is None:
			raise ValueError('Real-sibling auxiliary manifest is required for training.')
		self.manifest_path = Path(path).resolve()
		payload = validate_manifest(self.manifest_path, cfg)
		self.manifest_sha256 = payload['_manifest_sha256']
		self.format = payload['format']
		self.task = payload['task']
		self.horizon = int(_get(cfg, 'rof_real_sibling_aux_horizon', 3))
		self.batch_size = int(_get(cfg, 'rof_real_sibling_aux_batch_size', 8))
		self.device = torch.device(device)
		self._rng = np.random.default_rng(
			int(_get(cfg, 'seed', 0))
			+ int(_get(cfg, 'rof_real_sibling_aux_seed_offset', 15485863))
		)
		self._groups = []
		for root_id, asset in payload['_validated_groups']:
			with np.load(asset, allow_pickle=False) as archive:
				arrays = {
					key: np.ascontiguousarray(archive[key]) for key in SHARD_KEYS
				}
			self._groups.append((int(root_id), arrays))
		self._samples = 0
		self._calls = 0
		self._root_bag = np.empty(0, dtype=np.int64)
		self._pair_bag = np.empty((0, 2), dtype=np.int64)
		self._root_sample_counts = {
			int(root_id): 0 for root_id, _ in self._groups
		}
		self._positive_code_counts: dict[int, int] = {}
		self._negative_code_counts: dict[int, int] = {}
		self._ordered_pair_counts: dict[str, int] = {}
		self._relation_type_counts = {
			'zero_to_positive_axis': 0,
			'zero_to_negative_axis': 0,
			'opposite_same_axis': 0,
		}
		# Each axis contributes the same three unordered physical relations and
		# both orientations.  Cycling a shuffled bag keeps 1-D and 6-D tasks on
		# the same relation distribution instead of letting cross-axis pairs
		# dominate high-dimensional actions.
		self._pair_population = balanced_ordered_code_pairs(
			int(_get(cfg, 'action_dim', 0))
		)
		if self._pair_population.shape != (
			6 * int(_get(cfg, 'action_dim', 0)), 2
		):
			raise AssertionError('Balanced sibling pair population is malformed.')

	@property
	def identity(self) -> dict[str, Any]:
		return {
			'format': self.format,
			'manifest_sha256': self.manifest_sha256,
			'task': self.task,
			'condition': CONDITION,
			'root_count': len(self._groups),
		}

	@property
	def metrics(self) -> dict[str, int]:
		return {
			'sample_calls': int(self._calls),
			'sampled_root_pairs': int(self._samples),
			'root_count': len(self._groups),
			'root_sample_counts': {
				str(key): int(value)
				for key, value in sorted(self._root_sample_counts.items())
			},
			'positive_code_counts': {
				str(key): int(value)
				for key, value in sorted(self._positive_code_counts.items())
			},
			'negative_code_counts': {
				str(key): int(value)
				for key, value in sorted(self._negative_code_counts.items())
			},
			'ordered_code_pair_counts': dict(sorted(
				self._ordered_pair_counts.items()
			)),
			'relation_type_counts': dict(self._relation_type_counts),
		}

	def _draw_root_indices(self) -> np.ndarray:
		values = []
		while len(values) < self.batch_size:
			if not self._root_bag.size:
				self._root_bag = self._rng.permutation(
					len(self._groups)
				).astype(np.int64, copy=False)
			take = min(self.batch_size - len(values), len(self._root_bag))
			values.extend(self._root_bag[:take].tolist())
			self._root_bag = self._root_bag[take:]
		return np.asarray(values, dtype=np.int64)

	def _draw_code_pairs(self) -> np.ndarray:
		values = []
		while len(values) < self.batch_size:
			if not len(self._pair_bag):
				self._pair_bag = self._pair_population[
					self._rng.permutation(len(self._pair_population))
				]
			take = min(self.batch_size - len(values), len(self._pair_bag))
			values.extend(self._pair_bag[:take].tolist())
			self._pair_bag = self._pair_bag[take:]
		return np.asarray(values, dtype=np.int64)

	def _tensor(self, values) -> torch.Tensor:
		array = np.ascontiguousarray(values)
		return torch.from_numpy(array).to(self.device).contiguous()

	def sample(self) -> SiblingBatch:
		group_index = self._draw_root_indices()
		code_pairs = self._draw_code_pairs()
		selected = [self._groups[int(index)] for index in group_index]
		codes = self._groups[0][1]['branch__code']
		code_to_index = {int(code): index for index, code in enumerate(codes)}
		positive = np.asarray([
			code_to_index[int(pair[0])] for pair in code_pairs
		], dtype=np.int64)
		negative = np.asarray([
			code_to_index[int(pair[1])] for pair in code_pairs
		], dtype=np.int64)
		root = {
			field: self._tensor(np.stack([
				arrays[f'root__policy_{field}'] for _, arrays in selected
			]))
			for field in POLICY_FIELDS
		}
		positive_future = {
			field: self._tensor(np.stack([
				arrays[f'future__policy_{field}'][int(branch_index), :self.horizon]
				for (_, arrays), branch_index in zip(selected, positive)
			]).swapaxes(0, 1))
			for field in POLICY_FIELDS
		}
		negative_future = {
			field: self._tensor(np.stack([
				arrays[f'future__policy_{field}'][int(branch_index), :self.horizon]
				for (_, arrays), branch_index in zip(selected, negative)
			]).swapaxes(0, 1))
			for field in POLICY_FIELDS
		}
		action = self._tensor(np.stack([
			arrays['branch__action'][int(branch_index), :self.horizon]
			for (_, arrays), branch_index in zip(selected, positive)
		]).swapaxes(0, 1))
		root_ids = self._tensor(np.asarray(
			[root_id for root_id, _ in selected], dtype=np.int64,
		))
		positive_codes = self._tensor(np.asarray([
			arrays['branch__code'][int(branch_index)]
			for (_, arrays), branch_index in zip(selected, positive)
		], dtype=np.int32))
		negative_codes = self._tensor(np.asarray([
			arrays['branch__code'][int(branch_index)]
			for (_, arrays), branch_index in zip(selected, negative)
		], dtype=np.int32))
		if bool((positive_codes == negative_codes).any().item()):
			raise RuntimeError('Sibling sampler selected the same branch as its negative.')
		for (root_id, _), positive_code, negative_code in zip(
			selected, code_pairs[:, 0], code_pairs[:, 1]
		):
			root_id = int(root_id)
			positive_code = int(positive_code)
			negative_code = int(negative_code)
			self._root_sample_counts[root_id] += 1
			self._positive_code_counts[positive_code] = (
				self._positive_code_counts.get(positive_code, 0) + 1
			)
			self._negative_code_counts[negative_code] = (
				self._negative_code_counts.get(negative_code, 0) + 1
			)
			pair_key = f'{positive_code}->{negative_code}'
			self._ordered_pair_counts[pair_key] = (
				self._ordered_pair_counts.get(pair_key, 0) + 1
			)
			unordered = {positive_code, negative_code}
			if 0 in unordered:
				nonzero = next(code for code in unordered if code != 0)
				relation = (
					'zero_to_positive_axis'
					if nonzero > 0 else 'zero_to_negative_axis'
				)
			else:
				relation = 'opposite_same_axis'
			self._relation_type_counts[relation] += 1
		self._calls += 1
		self._samples += self.batch_size
		return SiblingBatch(
			root=root,
			positive_future=positive_future,
			negative_future=negative_future,
			action=action,
			root_id=root_ids,
			positive_code=positive_codes,
			negative_code=negative_codes,
		)


def latent_objective(
	predicted: torch.Tensor,
	positive_target: torch.Tensor,
	negative_target: torch.Tensor,
	*, rho: float, positive_coef: float, ranking_coef: float,
	margin_fraction: float,
) -> dict[str, torch.Tensor]:
	"""Return matched prediction and scale-aware cross-sibling ranking losses.

	The margin is a fraction of the observed distance between the two real
	sibling futures.  It therefore vanishes when two actions are physically
	indistinguishable at a horizon instead of inventing an impossible fixed
	separation, while remaining invariant to latent width and task scale.
	"""
	if (
		predicted.ndim != 3
		or predicted.shape != positive_target.shape
		or predicted.shape != negative_target.shape
		or predicted.shape[0] < 1
	):
		raise ValueError('Sibling latent tensors must share shape [H,B,D].')
	for value, name in (
		(rho, 'rho'), (positive_coef, 'positive_coef'),
		(ranking_coef, 'ranking_coef'),
		(margin_fraction, 'margin_fraction'),
	):
		if isinstance(value, bool) or not math.isfinite(float(value)):
			raise ValueError(f'{name} must be finite.')
	if not 0.0 < float(rho) <= 1.0:
		raise ValueError('rho must lie in (0,1].')
	if positive_coef <= 0.0 or ranking_coef < 0.0:
		raise ValueError('Sibling loss coefficients are invalid.')
	if not 0.0 <= margin_fraction <= 1.0:
		raise ValueError('Sibling margin fraction must lie in [0,1].')
	if not all(tensor.is_floating_point() for tensor in (
		predicted, positive_target, negative_target,
	)):
		raise ValueError('Sibling latent tensors must be floating point.')
	if not torch.compiler.is_compiling() and not all(bool(
		torch.isfinite(tensor).all().item()
	) for tensor in (predicted, positive_target, negative_target)):
		raise ValueError('Sibling latent tensors contain non-finite values.')

	positive_error = (predicted - positive_target.detach()).square().mean(dim=-1)
	negative_error = (predicted - negative_target.detach()).square().mean(dim=-1)
	sibling_distance = (
		positive_target.detach() - negative_target.detach()
	).square().mean(dim=-1)
	margin = float(margin_fraction) * sibling_distance
	ranking_frame = F.relu(positive_error - negative_error + margin)
	weights = predicted.new_tensor([
		float(rho) ** index for index in range(predicted.shape[0])
	]).unsqueeze(-1)
	positive_loss = (positive_error * weights).sum() / (
		predicted.shape[0] * predicted.shape[1]
	)
	ranking_loss = (ranking_frame * weights).sum() / (
		predicted.shape[0] * predicted.shape[1]
	)
	weighted_loss = (
		float(positive_coef) * positive_loss
		+ float(ranking_coef) * ranking_loss
	)
	return {
		'positive_loss': positive_loss,
		'ranking_loss': ranking_loss,
		'weighted_loss': weighted_loss,
		'positive_mse': positive_error.mean(),
		'wrong_sibling_mse': negative_error.mean(),
		'sibling_target_distance': sibling_distance.mean(),
		'correct_minus_wrong_mse': (positive_error - negative_error).mean(),
		'ranking_accuracy': (positive_error < negative_error).to(
			predicted.dtype
		).mean(),
		'margin_violation_rate': (ranking_frame > 0).to(predicted.dtype).mean(),
	}
