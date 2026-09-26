"""Episode-isolated causal diagnostics for Robust Object Field checkpoints.

This is deliberately an *offline diagnostic*, not another controller training
runner.  It consumes one immutable dataset produced by
``collect_rof_causal_probe_dataset.py`` and asks four increasingly downstream
questions on exactly the same trajectories:

``A``  Did the online masks agree with scoring-only simulator masks, including
       the important "non-empty but wrong" case?
``B``  How much simulator state, reward, and behaviour action is *linearly*
       accessible from raw causal evidence versus the learned latent?  This is
       an accessibility diagnostic, not proof that a failed linear probe means
       the information is absent.
``C``  Does the frozen learned transition beat persistence, and does the real
       action beat a deterministic action shuffle, at 1/3/5 steps?
``D``  Is the frozen reward head calibrated to held-out rewards?  Q values are
       also compared with behaviour returns, but only descriptively: TD-MPC2's
       Q target is not the Monte-Carlo return of the collected MPC behaviour.

Simulator state and ground-truth masks live under the ``labels__`` namespace.
They are read only after collection and are never passed to a policy encoder.
All fitting and model selection splits are lists of whole episode ids supplied
by the dataset manifest.  A frame-level random split is intentionally
impossible in this module because adjacent video frames would leak heavily.

The result is fail-closed.  Missing checkpoints make B/C/D ``inconclusive``;
they never turn into a successful diagnosis by omission.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Mapping, Sequence

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
for local_path in (str(REPO_DIR), str(PROJECT_DIR)):
	while local_path in sys.path:
		sys.path.remove(local_path)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))


DATASET_FORMAT = 'rof_causal_probe_dataset_v1'
RESULT_FORMAT = 'rof_causal_ladder_result_v1'
STAGES = ('A_mask', 'B_sufficiency', 'C_dynamics', 'D_control')
STATUSES = frozenset({'passed', 'failed', 'inconclusive'})
POLICY_KEYS = (
	'policy_rgb', 'policy_object', 'policy_object_mask', 'policy_role_exists',
)
LABEL_KEYS = ('labels__state', 'labels__gt_role_mask', 'labels__gt_visible')
TRANSITION_KEYS = ('action', 'reward', 'done')
INDEX_KEYS = ('episode_id', 'step')
HORIZONS = (1, 3, 5)
EXPECTED_CHECKPOINT_STEPS = (10_000, 20_000, 30_000)
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)


def _require(condition: bool, message: str) -> None:
	if not condition:
		raise ValueError(message)


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as stream:
		for block in iter(lambda: stream.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def _json(path: Path) -> dict:
	value = json.loads(path.read_text(encoding='utf-8'))
	_require(isinstance(value, dict), f'Expected JSON object: {path}')
	return value


def _finite(value, name: str) -> np.ndarray:
	array = np.asarray(value)
	_require(np.isfinite(array).all(), f'{name} contains non-finite values.')
	return array


def _record_id(record: Mapping) -> int:
	value = record.get('episode_id', record.get('episode_index'))
	_require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
		'Episode record requires a non-negative integer episode_id.')
	return int(value)


@dataclass(frozen=True)
class Episode:
	episode_id: int
	condition: str
	path: Path
	arrays: Mapping[str, np.ndarray]

	@property
	def decisions(self) -> int:
		return int(self.arrays['action'].shape[0])


@dataclass(frozen=True)
class FrozenDataset:
	manifest_path: Path
	manifest: Mapping
	task: str
	role_names: tuple[str, ...]
	state_names: tuple[str, ...]
	action_dim: int
	episodes: tuple[Episode, ...]
	splits: Mapping[str, tuple[int, ...]]

	def split(self, name: str) -> tuple[Episode, ...]:
		wanted = set(self.splits[name])
		return tuple(ep for ep in self.episodes if ep.episode_id in wanted)


def _episode_path(manifest_path: Path, record: Mapping) -> Path:
	value = record.get('relative_path', record.get('path'))
	_require(isinstance(value, str) and value, 'Episode record requires relative_path.')
	path = Path(value)
	_require(not path.is_absolute(), 'Episode NPZ path must be relative to the manifest.')
	resolved = (manifest_path.parent / path).resolve()
	root = manifest_path.parent.resolve()
	try:
		resolved.relative_to(root)
	except ValueError as exc:
		raise ValueError('Episode NPZ escapes the immutable dataset directory.') from exc
	return resolved


def _validate_episode_arrays(
	arrays: Mapping[str, np.ndarray], *, episode_id: int, role_count: int,
	state_dim: int, action_dim: int, expected_steps: int | None,
) -> int:
	required = set(POLICY_KEYS + LABEL_KEYS + TRANSITION_KEYS + INDEX_KEYS)
	_require(set(arrays) == required,
		f'Episode {episode_id} keys {sorted(arrays)} != {sorted(required)}.')
	action = arrays['action']
	_require(action.ndim == 2 and action.shape[1] == action_dim,
		f'Episode {episode_id} action must be [T,{action_dim}].')
	steps = int(action.shape[0])
	_require(steps >= 1, f'Episode {episode_id} is empty.')
	if expected_steps is not None:
		_require(steps == expected_steps,
			f'Episode {episode_id} has {steps} decisions, expected {expected_steps}.')
	obs_len = steps + 1
	expected_shapes = {
		'policy_rgb': (obs_len, 9, 64, 64),
		'policy_object': (obs_len, role_count, 1770),
		'policy_object_mask': (obs_len, role_count, 3, 64, 64),
		'policy_role_exists': (obs_len, role_count),
		'labels__state': (obs_len, state_dim),
		'labels__gt_role_mask': (obs_len, role_count, 64, 64),
		'labels__gt_visible': (obs_len, role_count),
		'reward': (steps,),
		'done': (steps,),
		'episode_id': (obs_len,),
		'step': (obs_len,),
	}
	for name, shape in expected_shapes.items():
		_require(tuple(arrays[name].shape) == shape,
			f'Episode {episode_id} {name} shape {arrays[name].shape} != {shape}.')
	expected_dtypes = {
		'policy_rgb': np.uint8,
		'policy_object': np.float32,
		'policy_object_mask': np.bool_,
		'policy_role_exists': np.float32,
		'labels__gt_role_mask': np.bool_,
		'labels__gt_visible': np.bool_,
		'done': np.bool_,
	}
	for name, dtype in expected_dtypes.items():
		_require(arrays[name].dtype == dtype,
			f'Episode {episode_id} {name} dtype {arrays[name].dtype} != {dtype}.')
	for name in ('action', 'reward', 'labels__state'):
		_require(np.issubdtype(arrays[name].dtype, np.floating),
			f'Episode {episode_id} {name} must be floating point.')
		_finite(arrays[name], f'episode {episode_id} {name}')
	_require(np.array_equal(arrays['episode_id'], np.full(obs_len, episode_id)),
		f'Episode {episode_id} episode_id vector is misaligned.')
	_require(np.array_equal(arrays['step'], np.arange(obs_len)),
		f'Episode {episode_id} step vector is misaligned.')
	_require(np.all(arrays['policy_role_exists'] == 1.0),
		'ROF V0 causal probe forbids padded/nonexistent roles.')
	_require(not bool(arrays['done'][:-1].any()),
		f'Episode {episode_id} terminates before its final action.')
	return steps


def _validate_splits(raw: Mapping, episode_ids: Sequence[int]) -> dict[str, tuple[int, ...]]:
	_require(isinstance(raw, Mapping), 'Dataset manifest requires explicit episode_splits.')
	_require(set(raw) == {'train', 'validation', 'test'},
		'episode_splits must contain exactly train/validation/test.')
	result = {}
	for name in ('train', 'validation', 'test'):
		values = raw[name]
		_require(isinstance(values, list) and values,
			f'episode_splits.{name} must be a non-empty list.')
		_require(all(isinstance(v, int) and not isinstance(v, bool) for v in values),
			f'episode_splits.{name} contains a non-integer id.')
		_require(len(set(values)) == len(values),
			f'episode_splits.{name} contains duplicate ids.')
		result[name] = tuple(int(v) for v in values)
	flattened = [value for name in result for value in result[name]]
	_require(len(flattened) == len(set(flattened)), 'Episode splits overlap.')
	_require(set(flattened) == set(episode_ids),
		'Episode splits must cover the dataset exactly.')
	return result


def episode_split(
	episode_ids: Sequence[int], *, seed: int, train_fraction: float = 0.6,
	validation_fraction: float = 0.2,
) -> dict[str, list[int]]:
	"""Return a deterministic whole-episode split for collection tools/tests.

	Published datasets carry the resulting split explicitly; the evaluator never
	recomputes it while scoring.  This helper exists so a collector cannot be
	tempted to split adjacent frames independently.
	"""
	values = list(episode_ids)
	_require(len(values) >= 5, 'At least five unique episodes are required.')
	_require(all(isinstance(value, int) and not isinstance(value, bool) for value in values),
		'Episode ids must be integers.')
	_require(len(values) == len(set(values)), 'Episode ids must be unique.')
	_require(0.0 < train_fraction < 1.0 and 0.0 < validation_fraction < 1.0
		and train_fraction + validation_fraction < 1.0,
		'Invalid train/validation fractions.')
	rng = np.random.default_rng(int(seed))
	permuted = [values[index] for index in rng.permutation(len(values))]
	train_count = max(1, int(math.floor(len(values) * train_fraction)))
	validation_count = max(1, int(math.floor(len(values) * validation_fraction)))
	if train_count + validation_count >= len(values):
		validation_count = 1
		train_count = len(values) - 2
	return {
		'train': permuted[:train_count],
		'validation': permuted[train_count:train_count + validation_count],
		'test': permuted[train_count + validation_count:],
	}


def validate_manifest(path: Path | str, *, load_arrays: bool = True) -> FrozenDataset:
	"""Validate and load an immutable causal-probe dataset.

	The collector's own validator is called when available, then this evaluator
	revalidates the boundary it relies on.  Keeping the second check local makes
	the diagnostic artifact self-defending against a future collector change.
	"""
	manifest_path = Path(path).resolve()
	_require(manifest_path.is_file(), f'Dataset manifest not found: {manifest_path}')
	try:
		from tdmpc2.tools.collect_rof_causal_probe_dataset import validate_dataset
	except ImportError:
		validate_dataset = None
	if validate_dataset is not None:
		validate_dataset(manifest_path)
	raw = _json(manifest_path)
	_require(raw.get('format') == DATASET_FORMAT,
		f'Dataset format {raw.get("format")!r} != {DATASET_FORMAT!r}.')
	task = raw.get('task')
	_require(isinstance(task, str) and task, 'Dataset task must be non-empty.')
	label_contract = raw.get('label_contract')
	_require(isinstance(label_contract, Mapping), 'Dataset label_contract is missing.')
	_require(label_contract.get('namespace') in (None, 'labels__'),
		'Labels must use the labels__ namespace.')
	_require(label_contract.get('labels_never_policy_input') is True,
		'Privileged labels are not proven isolated from policy input.')
	role_names = raw.get('role_names')
	state_names = label_contract.get('state_names')
	_require(isinstance(role_names, list) and role_names and
		all(isinstance(v, str) and v for v in role_names) and
		len(role_names) == len(set(role_names)), 'Invalid role_names.')
	_require(isinstance(state_names, list) and state_names and
		all(isinstance(v, str) and v for v in state_names) and
		len(state_names) == len(set(state_names)), 'Invalid state_names.')
	declared_policy = raw.get('policy_observation_keys')
	_require(list(declared_policy or []) == list(POLICY_KEYS),
		f'policy input keys must be exactly {list(POLICY_KEYS)}.')
	declared_labels = raw.get('label_keys')
	_require(list(declared_labels or []) == list(LABEL_KEYS),
		f'label keys must be exactly {list(LABEL_KEYS)}.')
	collection = raw.get('collection')
	_require(isinstance(collection, Mapping), 'Dataset collection metadata is missing.')
	action_dim = int(raw.get('action_dim', -1))
	_require(action_dim > 0, 'action_dim must be positive.')
	expected_steps_value = raw.get('steps')
	expected_steps = int(expected_steps_value) if expected_steps_value is not None else None
	records = raw.get('episodes')
	_require(isinstance(records, list) and records, 'Dataset episodes must be non-empty.')
	ids = [_record_id(record) for record in records]
	_require(len(ids) == len(set(ids)), 'Dataset episode ids are not unique.')
	splits = _validate_splits(raw.get('episode_splits', raw.get('splits')), ids)
	episodes = []
	for record, episode_id in zip(records, ids):
		path_value = _episode_path(manifest_path, record)
		_require(path_value.is_file(), f'Episode shard not found: {path_value}')
		expected_sha = record.get('sha256')
		_require(isinstance(expected_sha, str) and len(expected_sha) == 64,
			f'Episode {episode_id} is missing a SHA-256.')
		_require(_sha256(path_value) == expected_sha,
			f'Episode {episode_id} NPZ SHA-256 mismatch.')
		condition = record.get('condition', raw.get('condition'))
		_require(condition in {'clean', 'hard'},
			f'Episode {episode_id} condition must be clean or hard.')
		arrays = {}
		if load_arrays:
			with np.load(path_value, allow_pickle=False) as archive:
				arrays = {name: archive[name] for name in archive.files}
			_validate_episode_arrays(
				arrays, episode_id=episode_id, role_count=len(role_names),
				state_dim=len(state_names), action_dim=action_dim,
				expected_steps=expected_steps,
			)
		episodes.append(Episode(episode_id, condition, path_value, arrays))
	return FrozenDataset(
		manifest_path=manifest_path, manifest=raw, task=task,
		role_names=tuple(role_names), state_names=tuple(state_names),
		action_dim=action_dim, episodes=tuple(episodes), splits=splits,
	)


def load_dataset(path: Path | str) -> FrozenDataset:
	return validate_manifest(path, load_arrays=True)


def _max_burst(bits: np.ndarray) -> int:
	best = current = 0
	for value in np.asarray(bits, dtype=np.bool_).tolist():
		current = current + 1 if value else 0
		best = max(best, current)
	return int(best)


def _mask_iou(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
	intersection = np.logical_and(pred, target).sum(axis=(-2, -1), dtype=np.int64)
	union = np.logical_or(pred, target).sum(axis=(-2, -1), dtype=np.int64)
	return np.divide(
		intersection, union, out=np.ones_like(intersection, dtype=np.float64),
		where=union > 0,
	)


def _identity_swap_bits(pred: np.ndarray, target: np.ndarray, visible: np.ndarray) -> np.ndarray:
	frames, roles = pred.shape[:2]
	if roles < 2:
		return np.zeros(frames, dtype=np.bool_)
	result = np.zeros(frames, dtype=np.bool_)
	identity = tuple(range(roles))
	for frame in range(frames):
		if not bool(visible[frame].all()):
			continue
		scores = np.empty((roles, roles), dtype=np.float64)
		for source in range(roles):
			for label in range(roles):
				scores[source, label] = _mask_iou(
					pred[frame, source][None], target[frame, label][None]
				)[0]
		best = max(itertools.permutations(range(roles)),
			key=lambda order: sum(scores[source, order[source]] for source in range(roles)))
		identity_score = sum(scores[source, source] for source in range(roles))
		best_score = sum(scores[source, best[source]] for source in range(roles))
		result[frame] = best != identity and best_score > identity_score + 1e-9
	return result


def compute_mask_audit(dataset: FrozenDataset) -> dict:
	"""Stage A: distinguish empty masks, non-empty wrong masks, and role swaps."""
	by_split = {}
	for split_name in ('train', 'validation', 'test'):
		role_rows = {role: [] for role in dataset.role_names}
		swap_bits_by_episode = []
		swap_eligible_by_episode = []
		for episode in dataset.split(split_name):
			pred = episode.arrays['policy_object_mask'][:, :, -1]
			gt = episode.arrays['labels__gt_role_mask']
			visible = episode.arrays['labels__gt_visible']
			iou = _mask_iou(pred, gt)
			for index, role in enumerate(dataset.role_names):
				pred_nonempty = pred[:, index].any(axis=(-2, -1))
				eligible = visible[:, index]
				wrong = eligible & pred_nonempty & (iou[:, index] < 0.5)
				failure = eligible & (iou[:, index] < 0.5)
				role_rows[role].append((eligible, pred_nonempty, iou[:, index], wrong, failure))
			swap_bits_by_episode.append(_identity_swap_bits(pred, gt, visible))
			# A role permutation is only defined when every configured role has a
			# scoring-visible GT mask.  Counting occluded frames in the denominator
			# would silently make the swap rate look better than it is.
			swap_eligible_by_episode.append(visible.all(axis=1))
		roles = {}
		for role, rows in role_rows.items():
			visible_count = sum(int(row[0].sum()) for row in rows)
			_require(visible_count > 0,
				f'No scoring-visible {role!r} frames in {split_name} split.')
			nonempty_count = sum(int((row[0] & row[1]).sum()) for row in rows)
			iou_values = np.concatenate([row[2][row[0]] for row in rows])
			wrong_count = sum(int(row[3].sum()) for row in rows)
			failure_count = sum(int(row[4].sum()) for row in rows)
			roles[role] = {
				'visible_frames': visible_count,
				'pred_nonempty_rate_on_visible': nonempty_count / visible_count,
				'mean_iou_on_visible': float(iou_values.mean()),
				'success_at_0_5_on_visible': float((iou_values >= 0.5).mean()),
				'nonempty_wrong_rate': wrong_count / visible_count,
				'failure_rate': failure_count / visible_count,
				'max_nonempty_wrong_burst': max(_max_burst(row[3]) for row in rows),
				'max_failure_burst': max(_max_burst(row[4]) for row in rows),
			}
		swaps = sum(int(bits.sum()) for bits in swap_bits_by_episode)
		swap_frames = sum(int(bits.sum()) for bits in swap_eligible_by_episode)
		by_split[split_name] = {
			'roles': roles,
			'identity_swap_frames': swaps,
			'identity_swap_eligible_frames': swap_frames,
			'identity_swap_rate': swaps / max(swap_frames, 1),
			'identity_swap_rate_on_all_roles_visible': swaps / max(swap_frames, 1),
			'max_identity_swap_burst': max(_max_burst(bits) for bits in swap_bits_by_episode),
		}
	test_roles = by_split['test']['roles']
	flags = {
		'nonempty_wrong_observed': any(
			row['nonempty_wrong_rate'] > 0 for row in test_roles.values()
		),
		'long_nonempty_wrong_burst_observed': any(
			row['max_nonempty_wrong_burst'] >= 5 for row in test_roles.values()
		),
		'identity_swap_observed': by_split['test']['identity_swap_frames'] > 0,
	}
	return {
		'schema': 'rof_causal_ladder_mask_audit_v1',
		'status': 'passed',
		'checks': {
			'episode_level_splits': True,
			'gt_namespaced_and_isolated': True,
			'current_online_mask_only': True,
			'nonempty_wrong_measured_separately': True,
		},
		'by_split': by_split,
		'findings': flags,
	}


def _pool8(value: np.ndarray) -> np.ndarray:
	shape = value.shape
	_require(shape[-2:] == (64, 64), 'Spatial evidence must be 64x64.')
	return value.reshape(*shape[:-2], 8, 8, 8, 8).mean(axis=(-3, -1))


def _raw_matrix(episode: Episode, *, decisions_only: bool) -> np.ndarray:
	limit = episode.decisions if decisions_only else episode.decisions + 1
	arrays = episode.arrays
	rgb = _pool8(arrays['policy_rgb'][:limit].astype(np.float32) / 255.0)
	masks = _pool8(arrays['policy_object_mask'][:limit].astype(np.float32))
	objects = arrays['policy_object'][:limit].astype(np.float32, copy=False)
	exists = arrays['policy_role_exists'][:limit].astype(np.float32, copy=False)
	return np.concatenate([
		rgb.reshape(limit, -1), masks.reshape(limit, -1),
		objects.reshape(limit, -1), exists.reshape(limit, -1),
	], axis=1)


def _project_raw(blocks: Sequence[np.ndarray], *, output_dim: int, seed: int) -> list[np.ndarray]:
	_require(blocks and output_dim > 0, 'Raw projection requires data and positive width.')
	width = blocks[0].shape[1]
	_require(all(block.ndim == 2 and block.shape[1] == width for block in blocks),
		'Raw evidence widths differ between episodes.')
	if width <= output_dim:
		return [block.astype(np.float64, copy=False) for block in blocks]
	# One fixed Johnson-Lindenstrauss map is fitted to no data and therefore
	# cannot leak labels or validation/test episodes into the representation.
	rng = np.random.default_rng(seed)
	projection = rng.choice((-1.0, 1.0), size=(width, output_dim)).astype(np.float32)
	projection /= math.sqrt(output_dim)
	return [(block @ projection).astype(np.float64) for block in blocks]


def _project_mapping(
	values: Mapping[int, np.ndarray], *, output_dim: int, seed: int,
	train_ids: Sequence[int] | None = None,
) -> dict[int, np.ndarray]:
	"""Apply a label-free sketch after train-episode feature normalization.

	Normalizing *after* a random projection is not equivalent here: the raw
	policy input contains thousands of Cutie-query coordinates alongside a much
	smaller mask/RGB geometry block.  Without pre-normalization, the large query
	block can swamp useful geometry before ridge ever sees it.  Statistics are
	therefore estimated from train episodes only and then frozen for
	validation/test.  The exact same treatment is used for checkpoint latents.
	"""
	ids = list(values)
	_require(ids, 'Projection mapping is empty.')
	width = values[ids[0]].shape[1]
	_require(all(
		value.ndim == 2 and value.shape[1] == width for value in values.values()
	), 'Projection mapping widths differ between episodes.')
	if train_ids is None:
		train_ids = ids
	train_ids = tuple(int(value) for value in train_ids)
	_require(train_ids and set(train_ids).issubset(values),
		'Projection train ids must be a non-empty subset of episode ids.')
	count = sum(int(values[episode_id].shape[0]) for episode_id in train_ids)
	_require(count > 0, 'Projection train split is empty.')
	feature_sum = np.zeros(width, dtype=np.float64)
	feature_square_sum = np.zeros(width, dtype=np.float64)
	for episode_id in train_ids:
		block = _finite(values[episode_id], 'projection train features').astype(
			np.float64, copy=False
		)
		feature_sum += block.sum(axis=0, dtype=np.float64)
		feature_square_sum += np.square(block).sum(axis=0, dtype=np.float64)
	mean = feature_sum / count
	variance = np.maximum(feature_square_sum / count - np.square(mean), 0.0)
	scale = np.sqrt(variance)
	scale[scale < 1e-8] = 1.0
	mean = mean.astype(np.float32)
	scale = scale.astype(np.float32)
	standardized = [
		((values[value].astype(np.float32, copy=False) - mean) / scale)
		for value in ids
	]
	blocks = _project_raw(standardized, output_dim=output_dim, seed=seed)
	return {episode_id: block for episode_id, block in zip(ids, blocks)}


def _concat_by_ids(
	values: Mapping[int, np.ndarray], ids: Iterable[int], *, transitions: bool,
) -> np.ndarray:
	rows = []
	for episode_id in ids:
		value = values[int(episode_id)]
		rows.append(value[:-1] if transitions and value.shape[0] > 1 else value)
	_require(rows, 'Probe split contains no arrays.')
	return np.concatenate(rows, axis=0)


def _metric(y_true: np.ndarray, y_pred: np.ndarray, *, names: Sequence[str] | None = None) -> dict:
	y_true = _finite(y_true, 'metric target').astype(np.float64, copy=False)
	y_pred = _finite(y_pred, 'metric prediction').astype(np.float64, copy=False)
	if y_true.ndim == 1:
		y_true = y_true[:, None]
	if y_pred.ndim == 1:
		y_pred = y_pred[:, None]
	_require(y_true.shape == y_pred.shape and y_true.shape[0] > 0,
		'Metric arrays must be aligned and non-empty.')
	error = y_pred - y_true
	rmse_by_dim = np.sqrt(np.mean(error ** 2, axis=0))
	variance = np.mean((y_true - y_true.mean(axis=0)) ** 2, axis=0)
	mse = np.mean(error ** 2, axis=0)
	r2 = np.where(variance > 1e-12, 1.0 - mse / variance, np.nan)
	scale = np.sqrt(variance)
	nrmse = np.where(scale > 1e-12, rmse_by_dim / scale, np.nan)
	result = {
		'samples': int(y_true.shape[0]),
		'rmse': float(np.sqrt(np.mean(error ** 2))),
		'macro_r2': float(np.nanmean(r2)) if np.isfinite(r2).any() else None,
		'macro_normalized_rmse': (
			float(np.nanmean(nrmse)) if np.isfinite(nrmse).any() else None
		),
	}
	if names is not None:
		_require(len(names) == y_true.shape[1], 'Metric names do not match target width.')
		result['by_dimension'] = {
			name: {
				'rmse': float(rmse_by_dim[index]),
				'r2': float(r2[index]) if np.isfinite(r2[index]) else None,
				'normalized_rmse': (
					float(nrmse[index]) if np.isfinite(nrmse[index]) else None
				),
			}
			for index, name in enumerate(names)
		}
	return result


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> dict:
	x = _finite(x, 'ridge features').astype(np.float64, copy=False)
	y = _finite(y, 'ridge labels').astype(np.float64, copy=False)
	if y.ndim == 1:
		y = y[:, None]
	_require(x.ndim == 2 and y.ndim == 2 and x.shape[0] == y.shape[0],
		'Ridge features/labels are misaligned.')
	x_mean = x.mean(axis=0)
	x_scale = x.std(axis=0)
	x_scale[x_scale < 1e-8] = 1.0
	y_mean = y.mean(axis=0)
	xn = (x - x_mean) / x_scale
	gram = xn.T @ xn
	gram.flat[::gram.shape[0] + 1] += float(alpha)
	weights = np.linalg.solve(gram, xn.T @ (y - y_mean))
	return {
		'x_mean': x_mean, 'x_scale': x_scale, 'y_mean': y_mean,
		'weights': weights, 'alpha': float(alpha),
	}


def _ridge_predict(model: Mapping, x: np.ndarray) -> np.ndarray:
	x = np.asarray(x, dtype=np.float64)
	return ((x - model['x_mean']) / model['x_scale']) @ model['weights'] + model['y_mean']


def ridge_probe(
	x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray,
	y_validation: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
	*, names: Sequence[str] | None = None,
) -> dict:
	"""Select ridge strength on validation episodes and score test episodes."""
	best = None
	for alpha in RIDGE_GRID:
		model = _ridge_fit(x_train, y_train, alpha)
		validation = _metric(y_validation, _ridge_predict(model, x_validation), names=names)
		score = validation['macro_normalized_rmse']
		if score is None:
			score = validation['rmse']
		if best is None or score < best[0]:
			best = (score, model, validation)
	_require(best is not None, 'Ridge grid produced no model.')
	_, model, validation = best
	return {
		'probe': 'linear_ridge_with_train_standardization_v1',
		'selection': 'alpha selected on validation episodes only',
		'alpha': model['alpha'],
		'feature_dim': int(x_train.shape[1]),
		'target_dim': int(np.asarray(y_train).reshape(len(y_train), -1).shape[1]),
		'validation': validation,
		'test': _metric(y_test, _ridge_predict(model, x_test), names=names),
	}


def _targets(dataset: FrozenDataset) -> dict[str, dict[int, np.ndarray]]:
	return {
		'state': {
			ep.episode_id: ep.arrays['labels__state'][:-1].astype(np.float64)
			for ep in dataset.episodes
		},
		'action': {
			ep.episode_id: ep.arrays['action'].astype(np.float64)
			for ep in dataset.episodes
		},
		'reward': {
			ep.episode_id: ep.arrays['reward'].astype(np.float64)[:, None]
			for ep in dataset.episodes
		},
	}


def _probe_bundle(
	dataset: FrozenDataset, features: Mapping[int, np.ndarray], *, include_action_for_reward: bool,
) -> dict:
	targets = _targets(dataset)
	split_ids = dataset.splits
	def matrix(name: str, values: Mapping[int, np.ndarray]) -> np.ndarray:
		return np.concatenate([values[episode_id] for episode_id in split_ids[name]], axis=0)
	state_result = ridge_probe(
		matrix('train', features), matrix('train', targets['state']),
		matrix('validation', features), matrix('validation', targets['state']),
		matrix('test', features), matrix('test', targets['state']),
		names=dataset.state_names,
	)
	action_names = tuple(f'action_{index}' for index in range(dataset.action_dim))
	action_result = ridge_probe(
		matrix('train', features), matrix('train', targets['action']),
		matrix('validation', features), matrix('validation', targets['action']),
		matrix('test', features), matrix('test', targets['action']),
		names=action_names,
	)
	reward_features = {}
	for episode_id, values in features.items():
		reward_features[episode_id] = (
			np.concatenate([values, targets['action'][episode_id]], axis=1)
			if include_action_for_reward else values
		)
	reward_result = ridge_probe(
		matrix('train', reward_features), matrix('train', targets['reward']),
		matrix('validation', reward_features), matrix('validation', targets['reward']),
		matrix('test', reward_features), matrix('test', targets['reward']),
		names=('reward',),
	)
	return {
		'state_from_representation': state_result,
		'behavior_action_from_representation': action_result,
		'reward_from_representation_and_action': reward_result,
	}


def _checkpoint_specs(values: Sequence[str]) -> list[tuple[int, Path]]:
	result = []
	for raw in values:
		step_text, separator, path_text = raw.partition('=')
		_require(separator and step_text.isdigit() and path_text,
			f'Checkpoint must be STEP=PATH, got {raw!r}.')
		step = int(step_text)
		path = Path(path_text).resolve()
		_require(step > 0 and path.is_file(), f'Invalid checkpoint {raw!r}.')
		result.append((step, path))
	_require(len({step for step, _ in result}) == len(result), 'Duplicate checkpoint steps.')
	return sorted(result)


def _runtime_args(raw: Mapping, dataset: FrozenDataset, runtime_config: Path, checkpoint: Path):
	from tdmpc2.tools import evaluate_cutie_multitask_checkpoint as base
	training_condition = 'hard' if raw.get('video_background_enabled') else 'clean'
	args = SimpleNamespace(
		task=dataset.task, backend='robust_object_field', erosion_pixels=0,
		training_seed=int(raw['seed']), training_condition=training_condition,
		expected_training_steps=int(raw['steps']),
		expected_training_eval_freq=int(raw['eval_freq']),
		expected_training_eval_episodes=int(raw['eval_episodes']),
		env_seed=918273, background_seed=918277, condition='clean', episodes=20,
		checkpoint=checkpoint, runtime_config=runtime_config,
		output=dataset.manifest_path.with_suffix('.probe-not-written.json'),
	)
	return base._prepare(args, dict(raw))


def _load_agent(
	dataset: FrozenDataset, runtime_config: Path, checkpoint: Path,
):
	import torch
	from tdmpc2.tdmpc2 import TDMPC2
	_require(torch.cuda.is_available(), 'ROF checkpoint probes require CUDA.')
	raw = _json(runtime_config)
	cfg = _runtime_args(raw, dataset, runtime_config, checkpoint)
	cfg.compile = False
	agent = TDMPC2(cfg)
	agent.load(checkpoint)
	agent.eval()
	return agent


def _encode_episodes(dataset: FrozenDataset, agent, *, batch_size: int) -> dict[int, np.ndarray]:
	import torch
	encoder = agent.model._encoder['object']
	result = {}
	pad = int(getattr(encoder.augmentation, 'pad', 3))
	with torch.no_grad():
		for episode in dataset.episodes:
			rows = []
			length = episode.decisions + 1
			for start in range(0, length, batch_size):
				stop = min(start + batch_size, length)
				obs = {
					'rgb': torch.as_tensor(
						episode.arrays['policy_rgb'][start:stop], device=agent.device
					),
					'object': torch.as_tensor(
						episode.arrays['policy_object'][start:stop], device=agent.device
					),
					'object_mask': torch.as_tensor(
						episode.arrays['policy_object_mask'][start:stop], device=agent.device
					),
					'role_exists': torch.as_tensor(
						episode.arrays['policy_role_exists'][start:stop], device=agent.device
					),
				}
				# In JointFieldShiftAug, index==pad is the identity crop.  This keeps
				# every checkpoint on an identical deterministic observation.
				shift = torch.full(
					(stop - start, 1, 1, 2), float(pad),
					device=agent.device, dtype=torch.float32,
				)
				z = encoder(obs, shift_index=shift)
				rows.append(z.detach().cpu().numpy().astype(np.float64))
			result[episode.episode_id] = np.concatenate(rows, axis=0)
	return result


def _model_batches(agent, z: np.ndarray, action: np.ndarray, kind: str, *, batch_size: int):
	import torch
	from common import math as td_math
	outputs = []
	with torch.no_grad():
		for start in range(0, len(z), batch_size):
			stop = min(start + batch_size, len(z))
			zt = torch.as_tensor(z[start:stop], device=agent.device, dtype=torch.float32)
			at = torch.as_tensor(action[start:stop], device=agent.device, dtype=torch.float32)
			if kind == 'reward':
				value = td_math.two_hot_inv(agent.model.reward(zt, at, None), agent.cfg)
			elif kind == 'q':
				logits = agent.model.Q(zt, at, None, return_type='all')
				value = td_math.two_hot_inv(logits, agent.cfg).mean(dim=0)
			else:
				raise ValueError(f'Unknown model batch kind {kind!r}.')
			outputs.append(value.detach().cpu().numpy().reshape(stop - start, -1))
	return np.concatenate(outputs, axis=0).astype(np.float64)


def _stage_c(dataset: FrozenDataset, latents: Mapping[int, np.ndarray], agent) -> dict:
	import torch
	rows = {}
	test_episodes = dataset.split('test')
	with torch.no_grad():
		for horizon in HORIZONS:
			model_errors = []
			persistence_errors = []
			shuffle_errors = []
			action_sensitivity = []
			action_shuffle_input_differences = []
			for episode in test_episodes:
				z = latents[episode.episode_id]
				action = episode.arrays['action'].astype(np.float32)
				count = episode.decisions - horizon + 1
				current = torch.as_tensor(z[:count], device=agent.device, dtype=torch.float32)
				pred = current
				shuffled = current.clone()
				rng = np.random.default_rng(104729 + episode.episode_id)
				permuted_action = action[rng.permutation(len(action))]
				for offset in range(horizon):
					real_a = torch.as_tensor(
						action[offset:offset + count], device=agent.device
					)
					shuffle_a = torch.as_tensor(
						permuted_action[offset:offset + count], device=agent.device
					)
					action_shuffle_input_differences.append(
						(real_a - shuffle_a).square().mean(dim=-1).cpu().numpy()
					)
					pred = agent.model.next(pred, real_a, None)
					shuffled = agent.model.next(shuffled, shuffle_a, None)
				target = torch.as_tensor(
					z[horizon:horizon + count], device=agent.device, dtype=torch.float32
				)
				model_errors.append((pred - target).square().mean(dim=-1).cpu().numpy())
				persistence_errors.append((current - target).square().mean(dim=-1).cpu().numpy())
				shuffle_errors.append((shuffled - target).square().mean(dim=-1).cpu().numpy())
				action_sensitivity.append((pred - shuffled).square().mean(dim=-1).cpu().numpy())
			model_mse = float(np.concatenate(model_errors).mean())
			persistence_mse = float(np.concatenate(persistence_errors).mean())
			shuffle_mse = float(np.concatenate(shuffle_errors).mean())
			rows[str(horizon)] = {
				'samples': int(sum(len(value) for value in model_errors)),
				'model_mse': model_mse,
				'persistence_mse': persistence_mse,
				'action_shuffle_mse': shuffle_mse,
				'model_over_persistence': model_mse / max(persistence_mse, 1e-12),
				'model_over_action_shuffle': model_mse / max(shuffle_mse, 1e-12),
				'action_sensitivity_mse': float(np.concatenate(action_sensitivity).mean()),
				'action_shuffle_input_mse': float(np.concatenate(
					action_shuffle_input_differences
				).mean()),
				'action_shuffle_interpretation': (
					'Only informative when action_shuffle_input_mse is non-trivial; '
					'near-constant behaviour actions make this comparison inconclusive.'
				),
			}
	return {
		'schema': 'rof_causal_ladder_dynamics_v1', 'status': 'passed',
		'checks': {
			'episode_boundaries_preserved': True,
			'identity_shift_for_checkpoint_encoder': True,
			'deterministic_within_episode_action_shuffle': True,
		},
		'horizons': rows,
	}


def _discounted_returns(reward: np.ndarray, discount: float) -> np.ndarray:
	result = np.empty_like(reward, dtype=np.float64)
	value = 0.0
	for index in range(len(reward) - 1, -1, -1):
		value = float(reward[index]) + discount * value
		result[index] = value
	return result[:, None]


def _checkpoint_td_targets(
	agent, next_z: np.ndarray, reward: np.ndarray, done: np.ndarray, *,
	batch_size: int, seed: int,
) -> np.ndarray:
	"""Evaluate the checkpoint's actual bootstrapped TD-target mechanism.

	``TDMPC2._td_target`` samples an actor action and two target critics.  We use
	a fixed, isolated RNG stream so the audit is reproducible without changing
	the surrounding process RNG.  Unlike a collected-policy Monte-Carlo return,
	this target is aligned with the objective used to train the online Q heads.
	"""
	import torch
	next_z = np.asarray(next_z)
	reward = np.asarray(reward).reshape(-1, 1)
	done = np.asarray(done).reshape(-1, 1)
	_require(len(next_z) == len(reward) == len(done),
		'TD-target inputs are misaligned.')
	device = torch.device(agent.device)
	device_index = (
		torch.cuda.current_device() if device.index is None else int(device.index)
	)
	outputs = []
	with torch.random.fork_rng(devices=[device_index], enabled=True):
		torch.manual_seed(int(seed))
		with torch.no_grad():
			for start in range(0, len(next_z), batch_size):
				stop = min(start + batch_size, len(next_z))
				zt = torch.as_tensor(
					next_z[start:stop], device=agent.device, dtype=torch.float32
				)
				rt = torch.as_tensor(
					reward[start:stop], device=agent.device, dtype=torch.float32
				)
				dt = torch.as_tensor(
					done[start:stop], device=agent.device, dtype=torch.float32
				)
				value = agent._td_target(zt, rt, dt, None)
				outputs.append(
					value.detach().cpu().numpy().reshape(stop - start, -1)
				)
	return np.concatenate(outputs, axis=0).astype(np.float64)


def _stage_d(dataset: FrozenDataset, latents: Mapping[int, np.ndarray], agent) -> dict:
	features = {episode_id: value[:-1] for episode_id, value in latents.items()}
	actions = {ep.episode_id: ep.arrays['action'].astype(np.float64) for ep in dataset.episodes}
	rewards = {ep.episode_id: ep.arrays['reward'].astype(np.float64)[:, None]
		for ep in dataset.episodes}
	dones = {ep.episode_id: ep.arrays['done'].astype(np.float64)[:, None]
		for ep in dataset.episodes}
	discount_value = agent.discount
	if hasattr(discount_value, 'detach'):
		discount_value = discount_value.detach().cpu().reshape(-1)[0].item()
	discount = float(discount_value)
	returns = {
		ep.episode_id: _discounted_returns(ep.arrays['reward'], discount)
		for ep in dataset.episodes
	}
	td_targets = {
		ep.episode_id: _checkpoint_td_targets(
			agent, latents[ep.episode_id][1:], rewards[ep.episode_id],
			dones[ep.episode_id], batch_size=1024,
			seed=32452843 + ep.episode_id,
		)
		for ep in dataset.episodes
	}
	combined = {
		episode_id: np.concatenate([features[episode_id], actions[episode_id]], axis=1)
		for episode_id in features
	}
	def matrix(split: str, values: Mapping[int, np.ndarray]) -> np.ndarray:
		return np.concatenate([values[eid] for eid in dataset.splits[split]], axis=0)
	reward_refit = ridge_probe(
		matrix('train', combined), matrix('train', rewards),
		matrix('validation', combined), matrix('validation', rewards),
		matrix('test', combined), matrix('test', rewards), names=('reward',),
	)
	q_refit = ridge_probe(
		matrix('train', combined), matrix('train', returns),
		matrix('validation', combined), matrix('validation', returns),
		matrix('test', combined), matrix('test', returns), names=('behavior_return',),
	)
	q_td_refit = ridge_probe(
		matrix('train', combined), matrix('train', td_targets),
		matrix('validation', combined), matrix('validation', td_targets),
		matrix('test', combined), matrix('test', td_targets),
		names=('checkpoint_td_target',),
	)
	test_z = matrix('test', features)
	test_action = matrix('test', actions)
	test_reward = matrix('test', rewards)
	test_return = matrix('test', returns)
	test_td_target = matrix('test', td_targets)
	reward_head = _model_batches(agent, test_z, test_action, 'reward', batch_size=1024)
	q_head = _model_batches(agent, test_z, test_action, 'q', batch_size=1024)
	shuffled_action = np.concatenate([
		actions[ep.episode_id][
			np.random.default_rng(130363 + ep.episode_id).permutation(ep.decisions)
		]
		for ep in dataset.split('test')
	], axis=0)
	q_shuffled = _model_batches(agent, test_z, shuffled_action, 'q', batch_size=1024)
	return {
		'schema': 'rof_causal_ladder_control_readout_v1', 'status': 'passed',
		'checks': {
			'episode_level_refit_splits': True,
			'checkpoint_heads_frozen': True,
			'checkpoint_td_target_aligned': True,
			'q_label_is_behavior_return_not_optimal_q': True,
		},
		'interpretation': {
			'reward_head': (
				'Directly comparable with observed one-step rewards and suitable '
				'for reward-readout diagnosis.'
			),
			'q_head': (
				'Use q_head_vs_checkpoint_td_target for an objective-aligned '
				'Bellman/readout diagnosis. The behaviour-return comparison remains '
				'descriptive because these trajectories use deterministic MPC.'
			),
			'q_ridge_refit': (
				'Predicts collected-policy behaviour return, not TD-MPC2 optimal or '
				'actor-policy Q; it is not a capacity-matched replacement critic.'
			),
		},
		'discount': discount,
		'reward_head': _metric(test_reward, reward_head, names=('reward',)),
		'reward_ridge_refit': reward_refit,
		'q_head_vs_behavior_return': _metric(
			test_return, q_head, names=('behavior_return',)
		),
		'q_head_vs_checkpoint_td_target': _metric(
			test_td_target, q_head, names=('checkpoint_td_target',)
		),
		'q_ridge_refit_vs_checkpoint_td_target': q_td_refit,
		'checkpoint_td_target_contract': {
			'semantics': 'reward_t_plus_discount_times_target_q_at_encoded_obs_t_plus_1',
			'implementation': 'TDMPC2._td_target',
			'rng': 'fixed_isolated_per_episode_actor_and_target_critic_sample',
		},
		'q_ridge_refit_vs_behavior_return': q_refit,
		'q_actual_minus_shuffled_action': {
			'mean': float((q_head - q_shuffled).mean()),
			'fraction_positive': float((q_head > q_shuffled).mean()),
			'note': (
				'diagnostic preference only; behaviour action comes from MPC, is '
				'not an optimal-action label, and shuffled actions have no '
				'counterfactual environment-return label'
			),
		},
	}


def _inconclusive(reason: str, *, checks: Mapping[str, bool] | None = None) -> dict:
	return {
		'status': 'inconclusive', 'reason': str(reason),
		'checks': dict(checks or {'required_input_available': False}),
	}


def aggregate_status(diagnostics: Mapping[str, Mapping]) -> dict:
	"""Validate stage envelopes and compute the fail-closed engineering status."""
	_require(isinstance(diagnostics, Mapping) and set(diagnostics) == set(STAGES),
		f'Diagnostics must contain exactly {STAGES}.')
	statuses = {}
	for name in STAGES:
		stage = diagnostics[name]
		_require(isinstance(stage, Mapping), f'{name} must be an object.')
		status = stage.get('status')
		_require(status in STATUSES, f'{name} has invalid status {status!r}.')
		checks = stage.get('checks')
		_require(isinstance(checks, Mapping) and checks,
			f'{name} requires non-empty checks.')
		_require(all(isinstance(value, bool) for value in checks.values()),
			f'{name} checks must all be boolean.')
		if status == 'passed':
			_require(all(checks.values()), f'{name} cannot pass with a failed check.')
		statuses[name] = status
	overall = (
		'passed' if all(value == 'passed' for value in statuses.values())
		else 'failed' if any(value == 'failed' for value in statuses.values())
		else 'inconclusive'
	)
	return {'status': overall, 'stage_status': statuses}


def validate_result(payload: Mapping) -> dict:
	_require(isinstance(payload, Mapping), 'Result must be an object.')
	_require(payload.get('format') == RESULT_FORMAT, 'Result format mismatch.')
	aggregate = aggregate_status(payload.get('diagnostics'))
	complete = aggregate['status'] == 'passed'
	expected_status = (
		'rof_causal_ladder_complete' if complete
		else 'rof_causal_ladder_inconclusive'
	)
	declared = payload.get('status')
	_require(declared == expected_status,
		f'Result status {declared!r} != computed {expected_status!r}.')
	_require(isinstance(payload.get('engineering_pass'), bool),
		'Result requires boolean engineering_pass.')
	_require(payload.get('scientific_complete') is complete,
		'scientific_complete disagrees with the four ladder stages.')
	_require(payload.get('controller_training_authorized') is False,
		'This diagnostic must never authorize controller training.')
	for name in ('C_dynamics', 'D_control'):
		stage = payload['diagnostics'][name]
		reason = str(stage.get('reason', '')).lower()
		checkpoint_absent = (
			'checkpoint' in reason and 'unavailable' in reason
		) or stage['checks'].get('checkpoint_supplied') is False
		if checkpoint_absent:
			_require(stage['status'] == 'inconclusive',
				f'{name} cannot pass without a checkpoint.')
	return aggregate


def evaluate(args) -> dict:
	dataset = load_dataset(args.dataset)
	checks = {
		'dataset_format': dataset.manifest.get('format') == DATASET_FORMAT,
		'episode_splits_explicit': set(dataset.splits) == {'train', 'validation', 'test'},
		'privileged_labels_never_policy_input': dataset.manifest['label_contract'][
			'labels_never_policy_input'
		] is True,
	}
	a_stage = compute_mask_audit(dataset)
	raw_unprojected = {
		episode.episode_id: _raw_matrix(episode, decisions_only=True)
		for episode in dataset.episodes
	}
	raw_features = _project_mapping(
		raw_unprojected, output_dim=args.raw_projection_dim,
		seed=args.probe_seed, train_ids=dataset.splits['train'],
	)
	b_raw = _probe_bundle(dataset, raw_features, include_action_for_reward=True)
	checkpoint_specs = _checkpoint_specs(args.checkpoint)
	checkpoint_outputs = {}
	if checkpoint_specs:
		_require(args.runtime_config is not None,
			'--runtime-config is required when checkpoints are supplied.')
		runtime_config = Path(args.runtime_config).resolve()
		_require(runtime_config.is_file(), f'Runtime config not found: {runtime_config}')
		source = dataset.manifest.get('source')
		_require(isinstance(source, Mapping), 'Dataset source provenance is missing.')
		_require(Path(source.get('runtime_config', '')).resolve() == runtime_config,
			'Runtime config path differs from the immutable collection source.')
		_require(source.get('runtime_config_sha256') == _sha256(runtime_config),
			'Runtime config SHA-256 differs from the immutable collection source.')
		source_checkpoint_text = source.get('checkpoint')
		_require(isinstance(source_checkpoint_text, str) and source_checkpoint_text,
			'Dataset source checkpoint provenance is missing.')
		source_checkpoint = Path(source_checkpoint_text).resolve()
		for step, checkpoint in checkpoint_specs:
			expected_checkpoint = source_checkpoint.parent / f'eval_{step}.pt'
			_require(checkpoint == expected_checkpoint.resolve(),
				f'Checkpoint {checkpoint} is not the {step}-step checkpoint from '
				f'the immutable collection run ({expected_checkpoint}).')
			agent = _load_agent(dataset, runtime_config, checkpoint)
			latents_all = _encode_episodes(dataset, agent, batch_size=args.batch_size)
			latent_features_full = {
				episode_id: values[:-1] for episode_id, values in latents_all.items()
			}
			latent_features = _project_mapping(
				latent_features_full, output_dim=args.raw_projection_dim,
				# Reuse one sketch seed across checkpoints so apparent temporal
				# trends cannot be caused by a different random projection.
				seed=args.probe_seed, train_ids=dataset.splits['train'],
			)
			checkpoint_outputs[str(step)] = {
				'checkpoint': str(checkpoint),
				'checkpoint_sha256': _sha256(checkpoint),
				'B_latent': _probe_bundle(
					dataset, latent_features, include_action_for_reward=True
				),
				'C_dynamics': _stage_c(dataset, latents_all, agent),
				'D_control': _stage_d(dataset, latents_all, agent),
			}
			del agent
			try:
				import torch
				torch.cuda.empty_cache()
			except Exception:
				pass
		observed_steps = tuple(step for step, _ in checkpoint_specs)
		all_expected = observed_steps == EXPECTED_CHECKPOINT_STEPS
		b_stage = {
			'schema': 'rof_causal_ladder_sufficiency_v1', 'status': 'passed',
			'checks': {
				'episode_level_splits': True,
				'raw_projection_is_label_free': True,
				'raw_and_latent_same_trajectories': True,
			},
			'raw_evidence': b_raw,
			'interpretation_guard': (
				'Raw-versus-latent gaps measure linear accessibility under an '
				'equal-width, train-standardized label-free sketch. They do not '
				'prove that information is absent from either representation.'
			),
			'checkpoints': {
				step: value['B_latent'] for step, value in checkpoint_outputs.items()
			},
		}
		if not all_expected:
			b_stage.update(_inconclusive(
				'checkpoint_unavailable: expected 10k/20k/30k checkpoint ladder',
				checks={
					'raw_probe_completed': True,
					'checkpoint_supplied': True,
					'all_expected_checkpoints_supplied': False,
				},
			))
			b_stage['raw_evidence'] = b_raw
			b_stage['checkpoints'] = {
				step: value['B_latent'] for step, value in checkpoint_outputs.items()
			}
		c_stage = {
			'status': 'passed' if all_expected else 'inconclusive', 'checks': {
				'checkpoint_supplied': True,
				'all_checkpoints_evaluated': True,
				'all_expected_checkpoints_supplied': all_expected,
			},
			'checkpoints': {
				step: value['C_dynamics'] for step, value in checkpoint_outputs.items()
			},
		}
		d_stage = {
			'status': 'passed' if all_expected else 'inconclusive', 'checks': {
				'checkpoint_supplied': True,
				'all_checkpoints_evaluated': True,
				'all_expected_checkpoints_supplied': all_expected,
			},
			'checkpoints': {
				step: value['D_control'] for step, value in checkpoint_outputs.items()
			},
		}
		if not all_expected:
			c_stage['reason'] = 'checkpoint_unavailable: expected 10k/20k/30k checkpoint ladder'
			d_stage['reason'] = 'checkpoint_unavailable: expected 10k/20k/30k checkpoint ladder'
	else:
		b_stage = _inconclusive(
			'Raw evidence probes completed, but no checkpoint latent was supplied.',
			checks={'raw_probe_completed': True, 'checkpoint_supplied': False},
		)
		b_stage['raw_evidence'] = b_raw
		c_stage = _inconclusive('No checkpoint supplied.')
		d_stage = _inconclusive('No checkpoint supplied.')
	diagnostics = {
		'A_mask': a_stage,
		'B_sufficiency': b_stage,
		'C_dynamics': c_stage,
		'D_control': d_stage,
	}
	stage_status = aggregate_status(diagnostics)['status']
	complete = stage_status == 'passed'
	payload = {
		'format': RESULT_FORMAT,
		'status': (
			'rof_causal_ladder_complete' if complete
			else 'rof_causal_ladder_inconclusive'
		),
		'engineering_pass': all(checks.values()),
		'scientific_complete': complete,
		'controller_training_authorized': False,
		'recommendation': (
			'diagnosis_complete_review_layer_attribution' if complete
			else 'do_not_select_solution_complete_missing_diagnostics'
		),
		'task': dataset.task,
		'dataset': {
			'manifest': str(dataset.manifest_path),
			'manifest_sha256': _sha256(dataset.manifest_path),
			'episodes': len(dataset.episodes),
			'role_names': list(dataset.role_names),
			'state_names': list(dataset.state_names),
			'episode_splits': {name: list(values) for name, values in dataset.splits.items()},
			'checks': checks,
		},
		'probe_contract': {
			'probe_seed': args.probe_seed,
			'raw_projection': 'fixed_label_free_rademacher_jl_v1',
			'projection_normalization': (
				'per-coordinate mean/std from train episodes only, applied before '
				'the equal-width random projection'
			),
			'raw_projection_dim': args.raw_projection_dim,
			'ridge_grid': list(RIDGE_GRID),
			'horizons': list(HORIZONS),
			'expected_checkpoint_steps': list(EXPECTED_CHECKPOINT_STEPS),
			'split_unit': 'episode',
			'frame_level_random_split': False,
			'labels_are_scoring_only': True,
			'linear_probe_scope': (
				'information accessibility, not a proof of information absence'
			),
			'checkpoint_trajectory_scope': (
				'all checkpoints are evaluated on one frozen trajectory set '
				'collected by the manifest source policy; earlier checkpoints may '
				'therefore be off-policy'
			),
			'q_scope': (
				'Q-versus-behaviour-return metrics are descriptive and excluded '
				'from standalone Q-head causal attribution'
			),
		},
		'diagnostics': diagnostics,
		'checkpoint_details': checkpoint_outputs,
	}
	validate_result(payload)
	return payload


def _atomic_json(path: Path, payload: Mapping) -> None:
	path = path.resolve()
	if path.exists():
		raise FileExistsError(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
	try:
		temporary.write_text(
			json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
			encoding='utf-8',
		)
		os.replace(temporary, path)
	finally:
		if temporary.exists():
			temporary.unlink()


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--dataset', type=Path, required=True)
	parser.add_argument('--runtime-config', type=Path)
	parser.add_argument(
		'--checkpoint', action='append', default=[], metavar='STEP=PATH',
		help='Repeat for ROF checkpoints such as 10000=/run/models/eval_10000.pt.',
	)
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--probe-seed', type=int, default=8675309)
	parser.add_argument('--raw-projection-dim', type=int, default=256)
	parser.add_argument('--batch-size', type=int, default=256)
	args = parser.parse_args()
	if args.raw_projection_dim < 16 or args.raw_projection_dim > 1024:
		parser.error('--raw-projection-dim must lie in [16,1024].')
	if args.batch_size < 1:
		parser.error('--batch-size must be positive.')
	payload = evaluate(args)
	_atomic_json(args.output, payload)
	print('ROF_CAUSAL_LADDER_COMPLETE')
	print(f'STATUS={payload["status"]}')
	print(f'OUTPUT={args.output.resolve()}')


if __name__ == '__main__':
	main()
