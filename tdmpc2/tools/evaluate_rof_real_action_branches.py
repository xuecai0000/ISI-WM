"""Evaluate ROF dynamics on real same-state action branches.

This is an interventional *representation/dynamics* preflight.  Every root in
the input dataset contains a shared three-frame history and several real
five-step simulator continuations that differ only in the action applied at
the root.  Clean and hard observations are background twins of the same
physical continuations.  Complete sibling sets and both twins are assigned to
one root-level train/validation/test split.

The fitted transition is deliberately small and explicit::

    innovation = reference(history) + action_response(history, action)

It is compared with a separately fitted, parameter-matched actionless model
and persistence.  Unlike shuffled-action diagnostics, candidate actions here
are genuine sibling interventions and their recorded futures are real.  The
program reports h=1/3/5 prediction error, sibling-action ranking AUC,
action-effect R2 relative to the zero-action sibling, background-twin/action
distance ratio, and deployable packet validity/bursts.  Bootstrap resampling
is clustered by root, never by branch or frame.

Only ``policy_*`` arrays and ``action`` arrays enter the encoder or fitted
models.  ``oracle__*`` arrays are read in a separate validator solely to prove
that the sibling/twin experiment is physically well formed.  Reward, return,
background identifiers, GT state/masks, policy metadata, and model metadata
are forbidden model inputs.  This tool never trains a policy and can never
authorize controller training, even when every diagnostic gate passes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Mapping, Sequence

import numpy as np

from tdmpc2.tools import evaluate_rof_action_identifiable as identifiable
from tdmpc2.tools import evaluate_rof_causal_ladder as ladder
from tdmpc2.tools import evaluate_rof_normalized_delta as normalized
from tdmpc2.tools import evaluate_rof_transition_refit as refit
from tdmpc2.tools import collect_rof_real_action_branches as collector


os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


DATASET_FORMAT = 'rof_same_state_action_branch_dataset_v1'
RESULT_FORMAT = 'rof_real_action_branch_result_v1'
RESULT_STATUS = 'rof_real_action_branch_complete'
CONDITIONS = ('clean', 'hard')
SPLITS = ('train', 'validation', 'test')
HISTORY_FRAMES = 3
HORIZONS = (1, 3, 5)
MAX_HORIZON = 5
POLICY_FIELDS = ('rgb', 'object', 'object_mask', 'role_exists')
ORACLE_FIELDS = collector.ORACLE_KEYS
COMMON_FIELDS = (
	'history__action', 'branch__action', 'branch__code', 'branch__is_zero',
)
FORBIDDEN_MODEL_INPUT_TOKENS = (
	'reward', 'return', 'done', 'background', 'seed', 'oracle', 'labels',
	'official_state', 'physics_state', 'gt_role_mask', 'checkpoint', 'metadata',
)
ACTION_MODES = identifiable.ACTION_MODES
CAPACITY_CONTROL_MODES = (
	'nonlinear_action_aware', 'nonlinear_actionless',
)
FIT_MODES = ACTION_MODES + CAPACITY_CONTROL_MODES

# These values are part of the formal scientific protocol, not tuning knobs.
# The evaluator repeats the runner's checks so a hand-written invocation cannot
# silently publish an underpowered or differently tuned result.
FORMAL_SPLIT_COUNTS = {'train': 50, 'validation': 10, 'test': 20}
FORMAL_HIDDEN_DIM = 128
FORMAL_FIT_SEEDS = (20260913, 20260917, 20260923)
FORMAL_BOOTSTRAP_RESAMPLES = 20_000
FORMAL_BOOTSTRAP_SEED_OFFSET = 1_000_003


GATE_THRESHOLDS = {
	'min_test_roots': 4,
	'min_aware_gain_vs_persistence': 0.05,
	'min_aware_gain_vs_actionless': 0.05,
	'min_ranking_auc': 0.55,
	'min_action_effect_r2': 0.05,
	'max_background_twin_over_action_distance': 1.0,
	'min_packet_valid_rate': 0.95,
	'max_packet_invalid_burst': 2,
	'ci_rule': 'root_clustered_95pct_interval_must_cross_gate_in_safe_direction',
}


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
	_require(isinstance(value, dict), f'Expected a JSON object: {path}')
	return value


def _finite(value, name: str) -> np.ndarray:
	array = np.asarray(value)
	_require(np.isfinite(array).all(), f'{name} contains non-finite values.')
	return array


def _expected_archive_keys() -> frozenset[str]:
	keys = set(COMMON_FIELDS) | set(ORACLE_FIELDS)
	for condition in CONDITIONS:
		for field in POLICY_FIELDS:
			keys.add(f'{condition}__history__policy_{field}')
			keys.add(f'{condition}__future__policy_{field}')
	return frozenset(keys)


ARCHIVE_KEYS = _expected_archive_keys()


@dataclass(frozen=True)
class RootGroup:
	root_id: int
	split: str
	path: Path
	arrays: Mapping[str, np.ndarray]
	role_count: int
	action_dim: int
	branches: int
	state_dim: int
	physics_state_dim: int


@dataclass(frozen=True)
class BranchDataset:
	manifest_path: Path
	manifest: Mapping
	task: str
	role_names: tuple[str, ...]
	action_dim: int
	groups: tuple[RootGroup, ...]
	root_splits: Mapping[str, tuple[int, ...]]

	def split(self, name: str) -> tuple[RootGroup, ...]:
		wanted = set(self.root_splits[name])
		return tuple(group for group in self.groups if group.root_id in wanted)


def _policy_shapes(
	*, branches: int, roles: int,
) -> dict[str, tuple[int, ...]]:
	return {
		'history__policy_rgb': (HISTORY_FRAMES, 9, 64, 64),
		'history__policy_object': (HISTORY_FRAMES, roles, 1770),
		'history__policy_object_mask': (
			HISTORY_FRAMES, roles, 3, 64, 64,
		),
		'history__policy_role_exists': (HISTORY_FRAMES, roles),
		'future__policy_rgb': (branches, MAX_HORIZON, 9, 64, 64),
		'future__policy_object': (branches, MAX_HORIZON, roles, 1770),
		'future__policy_object_mask': (
			branches, MAX_HORIZON, roles, 3, 64, 64,
		),
		'future__policy_role_exists': (branches, MAX_HORIZON, roles),
	}


def _max_burst(bits: np.ndarray) -> int:
	best = current = 0
	for value in np.asarray(bits, dtype=np.bool_).reshape(-1).tolist():
		current = current + 1 if value else 0
		best = max(best, current)
	return int(best)


def validate_root_arrays(
	arrays: Mapping[str, np.ndarray], *, role_count: int, action_dim: int,
	root_id: int,
) -> dict:
	"""Validate one complete sibling/twin root without exposing oracle inputs."""
	_require(set(arrays) == set(ARCHIVE_KEYS),
		f'Root {root_id} archive keys differ from the exact schema.')
	for name in arrays:
		lower = name.lower()
		if name not in ORACLE_FIELDS:
			_require(not any(token in lower for token in FORBIDDEN_MODEL_INPUT_TOKENS),
				f'Forbidden model-input token in archive key {name!r}.')
	history_action = _finite(arrays['history__action'], 'history__action')
	branch_action = _finite(arrays['branch__action'], 'branch__action')
	_require(history_action.shape == (HISTORY_FRAMES - 1, action_dim),
		'history__action shape is invalid.')
	_require(branch_action.ndim == 3 and branch_action.shape[1:] == (
		MAX_HORIZON, action_dim,
	), 'branch__action shape is invalid.')
	branches = int(branch_action.shape[0])
	_require(branches >= 1 + 2 * action_dim,
		'The branch design is too small for zero and balanced +/- actions.')
	_require(np.max(np.abs(branch_action)) <= 1.0 + 1e-6,
		'Branch action leaves the normalized action range.')
	code = np.asarray(arrays['branch__code'])
	is_zero = np.asarray(arrays['branch__is_zero'])
	_require(code.shape == (branches,) and code.dtype == np.int32,
		'branch__code must be int32 [B].')
	_require(is_zero.shape == (branches,) and is_zero.dtype == np.bool_,
		'branch__is_zero must be bool [B].')
	_require(len(set(code.tolist())) == branches, 'Branch codes are not unique.')
	_require(int(is_zero.sum()) == 1, 'Exactly one zero-action sibling is required.')
	zero_index = int(np.flatnonzero(is_zero)[0])
	intervention = branch_action[:, 0]
	_require(np.allclose(intervention[zero_index], 0.0, atol=1e-7, rtol=0.0),
		'The designated zero sibling does not execute action zero.')
	_require(np.allclose(
		branch_action[:, 1:], branch_action[zero_index:zero_index + 1, 1:],
		atol=0.0, rtol=0.0,
	), 'Sibling continuation actions after the intervention are not identical.')
	nonzero = np.flatnonzero(~is_zero)
	_require(np.linalg.matrix_rank(intervention[nonzero].astype(np.float64)) == action_dim,
		'Sibling interventions are not full rank.')
	for index in nonzero:
		_require(any(np.allclose(
			intervention[candidate], -intervention[index], atol=1e-7, rtol=0.0,
		) for candidate in nonzero if candidate != index),
		f'Branch {index} has no balanced negative sibling.')

	shapes = _policy_shapes(branches=branches, roles=role_count)
	dtypes = {
		'rgb': np.uint8, 'object': np.float32,
		'object_mask': np.bool_, 'role_exists': np.float32,
	}
	for condition in CONDITIONS:
		for phase in ('history', 'future'):
			for field in POLICY_FIELDS:
				name = f'{condition}__{phase}__policy_{field}'
				value = np.asarray(arrays[name])
				_require(value.shape == shapes[f'{phase}__policy_{field}'],
					f'{name} shape is invalid: {value.shape}.')
				_require(value.dtype == np.dtype(dtypes[field]),
					f'{name} dtype is invalid: {value.dtype}.')
				if field in {'object', 'role_exists'}:
					_finite(value, name)
				if field == 'role_exists':
					_require(np.all((value == 0.0) | (value == 1.0)),
						f'{name} must be binary.')

	official = _finite(arrays['oracle__official_state'], 'oracle official state')
	physics = _finite(arrays['oracle__physics_state'], 'oracle physics state')
	gt_mask = np.asarray(arrays['oracle__gt_role_mask'])
	gt_visible = np.asarray(arrays['oracle__gt_visible'])
	absolute_step = np.asarray(arrays['oracle__absolute_step'])
	hard_official = _finite(
		arrays['oracle__hard_official_state'], 'hard oracle official state',
	)
	hard_gt_mask = np.asarray(arrays['oracle__hard_gt_role_mask'])
	clean_prefix_action = _finite(
		arrays['oracle__clean_prefix_action'], 'clean prefix action evidence',
	)
	hard_prefix_action = _finite(
		arrays['oracle__hard_prefix_action'], 'hard prefix action evidence',
	)
	clean_full_physics = _finite(
		arrays['oracle__clean_full_physics_state'], 'clean complete physics',
	)
	hard_full_physics = _finite(
		arrays['oracle__hard_full_physics_state'], 'hard complete physics',
	)
	_require(official.ndim == 3 and official.shape[:2] == (
		branches, MAX_HORIZON + 1,
	), 'oracle__official_state shape is invalid.')
	_require(physics.ndim == 3 and physics.shape[:2] == (
		branches, MAX_HORIZON + 1,
	), 'oracle__physics_state shape is invalid.')
	_require(gt_mask.shape == (
		branches, MAX_HORIZON + 1, role_count, 64, 64,
	) and gt_mask.dtype == np.bool_, 'oracle__gt_role_mask is invalid.')
	_require(gt_visible.shape == (
		branches, MAX_HORIZON + 1, role_count,
	) and gt_visible.dtype == np.bool_, 'oracle__gt_visible is invalid.')
	_require(absolute_step.shape == (MAX_HORIZON + 1,)
		and np.issubdtype(absolute_step.dtype, np.integer),
		'oracle__absolute_step is invalid.')
	_require(np.array_equal(
		absolute_step, np.arange(absolute_step[0], absolute_step[0] + MAX_HORIZON + 1),
	), 'oracle__absolute_step must be contiguous.')
	anchor_step = int(absolute_step[0])
	_require(anchor_step >= HISTORY_FRAMES - 1,
		'The oracle anchor cannot supply the required history.')
	_require(hard_official.shape == official.shape,
		'oracle__hard_official_state shape is invalid.')
	_require(hard_gt_mask.shape == gt_mask.shape and hard_gt_mask.dtype == np.bool_,
		'oracle__hard_gt_role_mask is invalid.')
	prefix_action_shape = (branches, anchor_step, action_dim)
	_require(clean_prefix_action.shape == prefix_action_shape,
		'oracle__clean_prefix_action shape is invalid.')
	_require(hard_prefix_action.shape == prefix_action_shape,
		'oracle__hard_prefix_action shape is invalid.')
	full_shape = (branches, anchor_step + MAX_HORIZON + 1, physics.shape[-1])
	_require(clean_full_physics.shape == full_shape,
		'oracle__clean_full_physics_state shape is invalid.')
	_require(hard_full_physics.shape == full_shape,
		'oracle__hard_full_physics_state shape is invalid.')
	_require(np.array_equal(
		gt_visible, gt_mask.reshape(branches, MAX_HORIZON + 1, role_count, -1).any(-1),
	), 'Oracle GT visibility disagrees with the GT masks.')
	_require(not np.any(gt_mask.sum(axis=2) > 1), 'Oracle role masks overlap.')
	_require(np.array_equal(hard_official, official),
		'Clean/hard official-state evidence differs.')
	_require(np.array_equal(hard_gt_mask, gt_mask),
		'Clean/hard GT-mask evidence differs.')
	_require(np.array_equal(clean_prefix_action, hard_prefix_action),
		'Clean/hard prefix-action evidence differs.')
	_require(np.array_equal(
		clean_prefix_action,
		np.broadcast_to(clean_prefix_action[0:1], clean_prefix_action.shape),
	), 'Sibling prefix-action evidence is not exactly shared.')
	_require(np.array_equal(
		clean_prefix_action[0, -(HISTORY_FRAMES - 1):], history_action,
	), 'History actions are not the tail of the stored prefix.')
	_require(np.array_equal(clean_full_physics, hard_full_physics),
		'Clean/hard complete physics evidence differs.')
	_require(np.array_equal(
		clean_full_physics[:, anchor_step:anchor_step + MAX_HORIZON + 1], physics,
	), 'Oracle branch physics is not the complete-trace anchor slice.')
	for condition, full_physics in (
		('clean', clean_full_physics), ('hard', hard_full_physics),
	):
		prefix = full_physics[:, :anchor_step + 1]
		_require(np.array_equal(prefix, np.broadcast_to(prefix[0:1], prefix.shape)),
			f'{condition} sibling prefix physics is not exactly shared.')
	_require(np.array_equal(
		physics[:, 0], np.broadcast_to(physics[zero_index, 0], physics[:, 0].shape),
	), 'Sibling branches do not share exactly the same root physics state.')
	_require(np.array_equal(
		official[:, 0], np.broadcast_to(official[zero_index, 0], official[:, 0].shape),
	), 'Sibling branches do not share exactly the same root official state.')
	physical_effect = np.sqrt(np.mean(np.square(
		physics[nonzero, 1] - physics[zero_index, 1]
	), axis=1))
	_require(np.all(physical_effect > 1e-12),
		'At least one nonzero action has no immediate real physics divergence.')
	return {
		'root_id': int(root_id), 'branches': branches,
		'zero_branch_index': zero_index,
		'action_design_rank': int(np.linalg.matrix_rank(intervention[nonzero])),
		'min_immediate_physics_effect_rms': float(physical_effect.min()),
		'mean_immediate_physics_effect_rms': float(physical_effect.mean()),
		'official_state_dim': int(official.shape[-1]),
		'physics_state_dim': int(physics.shape[-1]),
		'oracle_used_for': 'dataset_validity_only',
	}


def _root_path(manifest_path: Path, record: Mapping) -> Path:
	value = record.get('relative_path')
	_require(isinstance(value, str) and value, 'Group relative_path is missing.')
	relative = Path(value)
	_require(not relative.is_absolute(), 'Group path must be relative.')
	path = (manifest_path.parent / relative).resolve()
	try:
		path.relative_to(manifest_path.parent.resolve())
	except ValueError as exc:
		raise ValueError('Group path escapes the dataset directory.') from exc
	return path


def load_branch_dataset(
	path: Path | str, *, require_collector_contract: bool = False,
) -> BranchDataset:
	manifest_path = Path(path).resolve()
	if require_collector_contract:
		# Re-run the producing collector's complete provenance, hash, replay-guard,
		# action-design, and twin-physics validator before loading model packets.
		collector.validate_dataset(manifest_path)
	manifest = _json(manifest_path)
	_require(manifest.get('format') == DATASET_FORMAT,
		'Unexpected real-action branch dataset format.')
	_require(manifest.get('status') == 'rof_same_state_action_branch_dataset_complete',
		'Real-action branch dataset is not atomically complete.')
	_require(manifest.get('controller_training_authorized') is False,
		'A diagnostic dataset must not authorize controller training.')
	_require(manifest.get('conditions') == list(CONDITIONS),
		'Clean/hard twin condition contract changed.')
	_require(manifest.get('history_frames') == HISTORY_FRAMES,
		'Branch dataset history length changed.')
	_require(manifest.get('horizon') == MAX_HORIZON,
		'Branch dataset horizon changed.')
	_require(manifest.get('policy_input_fields') == [
		f'policy_{field}' for field in POLICY_FIELDS
	], 'Declared model-input policy fields changed.')
	model_contract = manifest.get('model_input_contract')
	_require(isinstance(model_contract, Mapping)
		and model_contract.get('keys') == list(collector.MODEL_INPUT_KEYS)
		and model_contract.get('oracle_namespace_excluded') is True
		and model_contract.get('condition_id_excluded') is True
		and model_contract.get('only_executed_actions') is True,
		'Model input isolation contract changed.')
	oracle_contract = manifest.get('oracle_contract')
	_require(isinstance(oracle_contract, Mapping)
		and oracle_contract.get('namespace') == 'oracle__'
		and oracle_contract.get('keys') == list(ORACLE_FIELDS)
		and oracle_contract.get('never_model_input') is True
		and oracle_contract.get('same_state_query_guard') == 'exact_before_after',
		'Declared oracle-only contract changed.')
	branch_protocol = manifest.get('branch_protocol')
	_require(isinstance(branch_protocol, Mapping)
		and branch_protocol.get('root_grouping')
		== 'all_siblings_and_background_twins_one_split_unit'
		and branch_protocol.get('fake_shuffled_action_futures') is False
		and branch_protocol.get('interventions')
		== 'zero_plus_minus_each_scaled_action_basis'
		and branch_protocol.get('continuation')
		== 'identical_open_loop_actions_across_siblings',
		'Root grouped real-branch protocol is missing.')
	task = manifest.get('task')
	role_names = manifest.get('role_names')
	action_dim = manifest.get('action_dim')
	_require(isinstance(task, str) and task, 'Task is missing.')
	_require(isinstance(role_names, list) and role_names
		and all(isinstance(value, str) and value for value in role_names),
		'Role names are invalid.')
	_require(isinstance(action_dim, int) and not isinstance(action_dim, bool)
		and action_dim > 0, 'Action dimension is invalid.')
	records = manifest.get('groups')
	_require(isinstance(records, list) and records, 'Dataset has no root groups.')
	groups = []
	ids = []
	state_dim = physics_dim = branches = None
	for record in records:
		_require(isinstance(record, Mapping), 'Group record is not an object.')
		root_id = record.get('root_id')
		split = record.get('split')
		_require(isinstance(root_id, int) and not isinstance(root_id, bool)
			and root_id >= 0, 'Group root_id is invalid.')
		_require(split in SPLITS, f'Root {root_id} split is invalid.')
		path_value = _root_path(manifest_path, record)
		_require(path_value.is_file(), f'Root shard not found: {path_value}')
		expected_hash = record.get('sha256')
		_require(isinstance(expected_hash, str) and len(expected_hash) == 64,
			f'Root {root_id} SHA-256 is missing.')
		_require(_sha256(path_value) == expected_hash,
			f'Root {root_id} SHA-256 mismatch.')
		with np.load(path_value, allow_pickle=False) as archive:
			arrays = {name: archive[name] for name in archive.files}
		audit = validate_root_arrays(
			arrays, role_count=len(role_names), action_dim=action_dim,
			root_id=root_id,
		)
		branches = audit['branches'] if branches is None else branches
		state_dim = audit['official_state_dim'] if state_dim is None else state_dim
		physics_dim = audit['physics_state_dim'] if physics_dim is None else physics_dim
		_require(audit['branches'] == branches, 'Branch count changes across roots.')
		_require(audit['official_state_dim'] == state_dim,
			'Official-state width changes across roots.')
		_require(audit['physics_state_dim'] == physics_dim,
			'Physics-state width changes across roots.')
		groups.append(RootGroup(
			int(root_id), str(split), path_value, arrays, len(role_names),
			action_dim, branches, state_dim, physics_dim,
		))
		ids.append(int(root_id))
	_require(len(ids) == len(set(ids)), 'Root ids are not unique.')
	raw_splits = manifest.get('root_splits')
	_require(isinstance(raw_splits, Mapping) and set(raw_splits) == set(SPLITS),
		'root_splits must contain train/validation/test.')
	splits = {}
	for split in SPLITS:
		values = raw_splits[split]
		_require(isinstance(values, list) and values,
			f'root_splits.{split} is empty or invalid.')
		_require(all(isinstance(value, int) and not isinstance(value, bool)
			for value in values), f'root_splits.{split} contains a non-integer.')
		_require(len(values) == len(set(values)), f'root_splits.{split} repeats a root.')
		splits[split] = tuple(int(value) for value in values)
	flat = [root for split in SPLITS for root in splits[split]]
	_require(len(flat) == len(set(flat)) and set(flat) == set(ids),
		'Root splits overlap or do not cover every root exactly once.')
	for group in groups:
		_require(group.root_id in set(splits[group.split]),
			f'Root {group.root_id} record split disagrees with root_splits.')
	return BranchDataset(
		manifest_path, manifest, task, tuple(role_names), int(action_dim),
		tuple(groups), splits,
	)


def _validate_source_binding(
	dataset: BranchDataset, runtime_config: Path, checkpoint: Path,
) -> dict:
	source = dataset.manifest.get('source')
	_require(isinstance(source, Mapping), 'Immutable source binding is missing.')
	runtime_config = runtime_config.resolve()
	checkpoint = checkpoint.resolve()
	_require(runtime_config.is_file() and not runtime_config.is_symlink(),
		f'Runtime config is missing or symlinked: {runtime_config}')
	_require(checkpoint.is_file() and not checkpoint.is_symlink(),
		f'Checkpoint is missing or symlinked: {checkpoint}')
	_require(Path(str(source.get('runtime_config', ''))).resolve() == runtime_config,
		'Runtime-config path differs from the dataset binding.')
	_require(Path(str(source.get('checkpoint', ''))).resolve() == checkpoint,
		'Checkpoint path differs from the dataset binding.')
	_require(source.get('runtime_config_sha256') == _sha256(runtime_config),
		'Runtime-config hash differs from the dataset binding.')
	_require(source.get('checkpoint_sha256') == _sha256(checkpoint),
		'Checkpoint hash differs from the dataset binding.')
	_require(source.get('backend') == 'robust_object_field',
		'Dataset source backend is not robust_object_field.')
	checkpoint_step = source.get('checkpoint_step')
	_require(isinstance(checkpoint_step, int) and not isinstance(checkpoint_step, bool)
		and checkpoint_step > 0, 'Dataset checkpoint step is malformed.')
	return {
		'runtime_config': str(runtime_config),
		'runtime_config_sha256': _sha256(runtime_config),
		'checkpoint': str(checkpoint), 'checkpoint_sha256': _sha256(checkpoint),
		'checkpoint_step': checkpoint_step,
	}


def _load_agent(
	dataset: BranchDataset, runtime_config: Path, checkpoint: Path,
):
	# This adapter supplies only task/roles/path to the existing frozen ROF loader.
	adapter = SimpleNamespace(
		task=dataset.task, role_names=dataset.role_names,
		manifest_path=dataset.manifest_path,
	)
	return ladder._load_agent(adapter, runtime_config, checkpoint)


def _encode_fields(agent, arrays: Mapping[str, np.ndarray], *, batch_size: int) -> np.ndarray:
	"""Encode an already selected deployable packet; no metadata is accepted."""
	import torch

	_require(set(arrays) == set(POLICY_FIELDS), 'Encoder packet fields changed.')
	leading = arrays['rgb'].shape[:-3]
	count = int(np.prod(leading))
	flat = {
		field: np.asarray(arrays[field]).reshape((count,) + arrays[field].shape[len(leading):])
		for field in POLICY_FIELDS
	}
	encoder = agent.model._encoder['object']
	pad = int(getattr(encoder.augmentation, 'pad', 3))
	rows = []
	with torch.no_grad():
		for start in range(0, count, batch_size):
			stop = min(start + batch_size, count)
			packet = {
				'rgb': torch.as_tensor(flat['rgb'][start:stop], device=agent.device),
				'object': torch.as_tensor(flat['object'][start:stop], device=agent.device),
				'object_mask': torch.as_tensor(
					flat['object_mask'][start:stop], device=agent.device,
				),
				'role_exists': torch.as_tensor(
					flat['role_exists'][start:stop], device=agent.device,
				),
			}
			shift = torch.full(
				(stop - start, 1, 1, 2), float(pad), device=agent.device,
				dtype=torch.float32,
			)
			value = encoder(packet, shift_index=shift)
			rows.append(value.detach().cpu().numpy().astype(np.float32))
	return np.concatenate(rows, axis=0).reshape(leading + (-1,))


def _model_packet(group: RootGroup, condition: str, phase: str) -> dict[str, np.ndarray]:
	_require(condition in CONDITIONS and phase in {'history', 'future'},
		'Unknown condition/phase.')
	# The construction is intentionally an explicit allow-list firewall.
	return {
		field: group.arrays[f'{condition}__{phase}__policy_{field}']
		for field in POLICY_FIELDS
	}


def encode_dataset(
	dataset: BranchDataset, agent, *, batch_size: int,
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
	result = {condition: {} for condition in CONDITIONS}
	for group in dataset.groups:
		for condition in CONDITIONS:
			history_z = _encode_fields(
				agent, _model_packet(group, condition, 'history'), batch_size=batch_size,
			)
			future_z = _encode_fields(
				agent, _model_packet(group, condition, 'future'), batch_size=batch_size,
			)
			result[condition][group.root_id] = {
				'history_z': history_z,
				'future_z': future_z,
				'history_u': normalized.clr_transform(
					history_z, int(agent.cfg.get('simnorm_dim', 8)),
				),
				'future_u': normalized.clr_transform(
					future_z, int(agent.cfg.get('simnorm_dim', 8)),
				),
			}
	return result


@dataclass(frozen=True)
class BranchRows:
	sequence: normalized.SequenceRows
	root_ids: np.ndarray
	branch_indices: np.ndarray
	is_zero: np.ndarray

	def __len__(self) -> int:
		return int(len(self.root_ids))


def build_rows(
	dataset: BranchDataset, encoded_condition: Mapping[int, Mapping[str, np.ndarray]],
	root_ids: Sequence[int],
) -> BranchRows:
	histories, windows, futures, roots, branch_indices, zero_bits = [], [], [], [], [], []
	by_id = {group.root_id: group for group in dataset.groups}
	for root_id in root_ids:
		group = by_id[int(root_id)]
		encoded = encoded_condition[int(root_id)]
		history_action = np.asarray(group.arrays['history__action'], dtype=np.float32)
		branch_action = np.asarray(group.arrays['branch__action'], dtype=np.float32)
		for branch in range(group.branches):
			histories.append(encoded['history_u'])
			windows.append(np.concatenate([history_action, branch_action[branch]], axis=0))
			futures.append(encoded['future_u'][branch])
			roots.append(int(root_id))
			branch_indices.append(branch)
			zero_bits.append(bool(group.arrays['branch__is_zero'][branch]))
	sequence = normalized.SequenceRows(
		initial_u_history=np.asarray(histories, dtype=np.float32),
		action_window=np.asarray(windows, dtype=np.float32),
		target_future_u=np.asarray(futures, dtype=np.float32),
		episode_ids=np.asarray(roots, dtype=np.int64),
		start_indices=np.asarray(branch_indices, dtype=np.int64),
	)
	return BranchRows(
		sequence, np.asarray(roots, dtype=np.int64),
		np.asarray(branch_indices, dtype=np.int64),
		np.asarray(zero_bits, dtype=np.bool_),
	)


def concatenate_rows(parts: Sequence[BranchRows]) -> BranchRows:
	"""Pool background twins without exposing a condition identifier."""
	_require(parts, 'No branch rows to concatenate.')
	sequences = [part.sequence for part in parts]
	return BranchRows(
		normalized.SequenceRows(
			initial_u_history=np.concatenate([
				row.initial_u_history for row in sequences
			], axis=0),
			action_window=np.concatenate([
				row.action_window for row in sequences
			], axis=0),
			target_future_u=np.concatenate([
				row.target_future_u for row in sequences
			], axis=0),
			episode_ids=np.concatenate([row.episode_ids for row in sequences], axis=0),
			start_indices=np.concatenate([row.start_indices for row in sequences], axis=0),
		),
		np.concatenate([part.root_ids for part in parts], axis=0),
		np.concatenate([part.branch_indices for part in parts], axis=0),
		np.concatenate([part.is_zero for part in parts], axis=0),
	)


def fit_branch_normalization(
	rows: BranchRows, *, simnorm_dim: int,
) -> normalized.DeltaNormalization:
	sequence = rows.sequence
	u_rows = [sequence.initial_u_history.reshape(-1, sequence.initial_u_history.shape[-1])]
	u_rows.append(sequence.target_future_u.reshape(-1, sequence.target_future_u.shape[-1]))
	u_joined = np.concatenate(u_rows, axis=0).astype(np.float32)
	action_joined = sequence.action_window.reshape(-1, sequence.action_window.shape[-1])
	# Physical action zero, rather than a behaviour-policy mean, anchors the
	# reference branch.  The validated +/- intervention design makes this valid.
	action_mean = np.zeros(action_joined.shape[1], dtype=np.float32)
	_, action_scale, action_floored = normalized._mean_std(
		action_joined, normalized.SCALE_FLOOR,
	)
	u_mean = u_joined.mean(axis=0).astype(np.float32)
	u_mean = normalized.project_clr_innovation(u_mean[None], simnorm_dim)[0][0]
	u_scale, u_audit = normalized._group_scalar_scale(
		u_joined, u_mean, simnorm_dim=simnorm_dim, floor=normalized.SCALE_FLOOR,
	)
	anchor = sequence.initial_u_history[:, -1]
	steps = np.concatenate([
		anchor[:, None], sequence.target_future_u,
	], axis=1)
	delta_rows = np.diff(steps, axis=1).reshape(-1, steps.shape[-1])
	delta_mean = delta_rows.mean(axis=0).astype(np.float32)
	delta_mean = normalized.project_clr_innovation(delta_mean[None], simnorm_dim)[0][0]
	delta_scale, delta_audit = normalized._group_scalar_scale(
		delta_rows, delta_mean, simnorm_dim=simnorm_dim,
		floor=normalized.SCALE_FLOOR,
	)
	horizon_scale, horizon_audit = {}, {}
	for horizon in HORIZONS:
		displacement = sequence.target_future_u[:, horizon - 1] - anchor
		horizon_scale[horizon], horizon_audit[horizon] = normalized._group_scalar_scale(
			displacement, np.zeros(displacement.shape[-1], dtype=np.float32),
			simnorm_dim=simnorm_dim, floor=normalized.SCALE_FLOOR,
		)
	return normalized.DeltaNormalization(
		u_mean=u_mean, u_scale=u_scale, action_mean=action_mean,
		action_scale=action_scale, delta_mean=delta_mean, delta_scale=delta_scale,
		horizon_scale=horizon_scale, simnorm_dim=simnorm_dim,
		floor_audit={
			'u_scale': u_audit, 'delta_scale': delta_audit,
			'horizon_scale': {
				str(horizon): horizon_audit[horizon] for horizon in HORIZONS
			},
			'action_scale': {
				'dimensions': int(len(action_scale)),
				'floored_dimensions': int(action_floored),
				'floored_fraction': float(action_floored / max(len(action_scale), 1)),
			},
			'action_center': 'physical_zero_balanced_real_interventions',
		},
	)


def normalization_summary(value: normalized.DeltaNormalization) -> dict:
	result = normalized.normalization_summary(value)
	result['statistics_source'] = (
		'pooled_clean_hard_train_roots_real_sibling_branches_only'
	)
	result['action_center'] = 'physical_zero_balanced_real_interventions'
	return result


def _subset_rows(rows: BranchRows, index: np.ndarray) -> normalized.SequenceRows:
	sequence = rows.sequence
	return normalized.SequenceRows(
		initial_u_history=sequence.initial_u_history[index],
		action_window=sequence.action_window[index],
		target_future_u=sequence.target_future_u[index],
		episode_ids=sequence.episode_ids[index],
		start_indices=sequence.start_indices[index],
	)


def _nonlinear_model_factory(
	action_mode: str, normalization: normalized.DeltaNormalization, *,
	hidden_dim: int,
):
	"""Build the parameter-matched nonlinear capacity-control pair.

	Both arms instantiate exactly the same modules.  The aware arm receives the
	actual normalized action.  The actionless arm receives a fixed all-one slot,
	so every action-slot weight remains active/trainable while no executed-action
	information is available.  Subtracting the zero-slot response gives the aware
	arm an exact physical-action-zero anchor and lets either arm learn a nonlinear,
	history-dependent residual.  This pair is diagnostic only and never replaces
	the preregistered small decomposed model in the scientific gate.
	"""
	import torch
	import torch.nn as nn

	_require(action_mode in CAPACITY_CONTROL_MODES,
		f'Unknown nonlinear capacity-control mode {action_mode!r}.')
	latent_dim = int(len(normalization.u_mean))
	action_dim = int(len(normalization.action_mean))
	_require(hidden_dim > 0 and latent_dim > 0 and action_dim > 0,
		'Model dimensions must be positive.')

	class _NonlinearActionCapacityControl(nn.Module):
		def __init__(self):
			super().__init__()
			self.history_trunk = nn.Sequential(
				nn.Linear(3 * latent_dim, hidden_dim),
				nn.LayerNorm(hidden_dim), nn.SiLU(),
				nn.Linear(hidden_dim, hidden_dim),
				nn.LayerNorm(hidden_dim), nn.SiLU(),
			)
			self.reference_head = nn.Linear(hidden_dim, latent_dim)
			# Concatenation followed by SiLU supplies nonlinear history/action
			# interaction.  This exact module exists in both arms.
			self.control_mlp = nn.Sequential(
				nn.Linear(hidden_dim + action_dim, hidden_dim), nn.SiLU(),
				nn.Linear(hidden_dim, latent_dim, bias=False),
			)
			self.inverse_head = nn.Sequential(
				nn.Linear(hidden_dim + latent_dim, hidden_dim), nn.SiLU(),
				nn.Linear(hidden_dim, action_dim),
			)
			self.inverse_context_head = nn.Sequential(
				nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
				nn.Linear(hidden_dim, action_dim),
			)
			nn.init.zeros_(self.reference_head.weight)
			nn.init.zeros_(self.reference_head.bias)
			nn.init.normal_(self.control_mlp[-1].weight, mean=0.0, std=1e-3)
			self.register_buffer('u_mean', torch.as_tensor(normalization.u_mean))
			self.register_buffer('u_scale', torch.as_tensor(normalization.u_scale))
			self.register_buffer(
				'action_mean', torch.as_tensor(normalization.action_mean),
			)
			self.register_buffer(
				'action_scale', torch.as_tensor(normalization.action_scale),
			)
			self.register_buffer(
				'delta_mean', torch.as_tensor(normalization.delta_mean),
			)
			self.register_buffer(
				'delta_scale', torch.as_tensor(normalization.delta_scale),
			)
			self.action_mode = action_mode
			self.simnorm_dim = int(normalization.simnorm_dim)

		def encode_history(self, u_history):
			return self.history_trunk(identifiable._state_features(u_history, self))

		def decompose(self, u_history, action_history):
			hidden = self.encode_history(u_history)
			reference_standardized = self.reference_head(hidden)
			action = (action_history[:, -1] - self.action_mean) / self.action_scale
			if self.action_mode == 'nonlinear_actionless':
				# A constant nonzero slot keeps the identical action columns active
				# without disclosing which real sibling action was executed.
				action_slot = torch.ones_like(action)
			else:
				action_slot = action
			zero_slot = torch.zeros_like(action)
			control_standardized = (
				self.control_mlp(torch.cat([hidden, action_slot], dim=-1))
				- self.control_mlp(torch.cat([hidden, zero_slot], dim=-1))
			)
			reference = self.delta_mean + reference_standardized * self.delta_scale
			control = control_standardized * self.delta_scale
			return reference, control

		def step(self, u_history, action_history):
			reference, control = self.decompose(u_history, action_history)
			reference_groups = reference.reshape(len(reference), -1, self.simnorm_dim)
			control_groups = control.reshape(len(control), -1, self.simnorm_dim)
			projected_reference = (
				reference_groups - reference_groups.mean(dim=-1, keepdim=True)
			)
			projected_control = (
				control_groups - control_groups.mean(dim=-1, keepdim=True)
			)
			projected = projected_reference + projected_control
			correction = reference_groups + control_groups - projected
			next_u = u_history[:, -1] + projected.reshape_as(reference)
			next_groups = next_u.reshape(len(next_u), -1, self.simnorm_dim)
			next_groups = next_groups - next_groups.mean(dim=-1, keepdim=True)
			return next_groups.reshape_as(next_u), {
				'reference_delta': projected_reference.reshape_as(reference),
				'control_delta': projected_control.reshape_as(control),
				'projection_correction': correction,
				'projection_linearity_error': (
					projected - (projected_reference + projected_control)
				),
			}

		def inverse_action(self, u_history, observed_next_u):
			hidden = self.encode_history(u_history)
			observed_delta = (
				observed_next_u - u_history[:, -1] - self.delta_mean
			) / self.delta_scale
			return {
				'with_observed_innovation': self.inverse_head(torch.cat(
					[hidden, observed_delta], dim=-1,
				)),
				'context_only': self.inverse_context_head(hidden),
			}

	return _NonlinearActionCapacityControl()


def fit_model(
	action_mode: str, train: BranchRows, validation: BranchRows,
	normalization: normalized.DeltaNormalization, *, device, seed: int,
	hidden_dim: int, batch_size: int, max_epochs: int, min_epochs: int,
	patience: int, convergence_window: int, convergence_threshold: float,
	learning_rate: float, weight_decay: float, inverse_weight: float,
	control_l2_weight: float,
) -> identifiable.FittedActionIdentifiableTransition:
	"""Fit only real sibling transitions; root test groups remain untouched."""
	import torch

	_require(action_mode in FIT_MODES, 'Unknown action mode.')
	_require(max_epochs >= min_epochs >= convergence_window >= 2,
		'Invalid convergence schedule.')
	torch.manual_seed(int(seed))
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(int(seed))
	try:
		torch.use_deterministic_algorithms(True, warn_only=True)
	except TypeError:
		torch.use_deterministic_algorithms(True)
	model = (
		identifiable._model_factory(
			action_mode, normalization, hidden_dim=hidden_dim,
		)
		if action_mode in ACTION_MODES else
		_nonlinear_model_factory(
			action_mode, normalization, hidden_dim=hidden_dim,
		)
	).to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=learning_rate, weight_decay=weight_decay,
	)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode='min', factor=0.5, patience=max(3, patience // 3),
		min_lr=learning_rate / 32.0,
	)
	rng = np.random.default_rng(int(seed))
	best_loss, best_epoch, best_state, stale = math.inf, 0, None, 0
	curve = []
	for epoch in range(1, max_epochs + 1):
		permutation = rng.permutation(len(train))
		total, seen = 0.0, 0
		for start in range(0, len(permutation), batch_size):
			index = permutation[start:start + batch_size]
			batch = _subset_rows(train, index)
			optimizer.zero_grad(set_to_none=True)
			metrics = identifiable._objective(
				model, batch, normalization, device=device, batch_size=len(batch),
				inverse_weight=inverse_weight,
				control_l2_weight=control_l2_weight, training=True,
			)
			torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
			optimizer.step()
			total += metrics['objective'] * len(batch)
			seen += len(batch)
		validation_metrics = identifiable._objective(
			model, validation.sequence, normalization, device=device,
			batch_size=batch_size, inverse_weight=inverse_weight,
			control_l2_weight=control_l2_weight, training=False,
		)
		scheduler.step(validation_metrics['objective'])
		curve.append({
			'epoch': int(epoch), 'train_objective': float(total / seen),
			'validation_objective': float(validation_metrics['objective']),
			'validation_prediction': float(validation_metrics['prediction']),
			'validation_inverse': float(validation_metrics['inverse']),
			'learning_rate': float(optimizer.param_groups[0]['lr']),
		})
		value = float(validation_metrics['objective'])
		significant = value < best_loss * (1.0 - convergence_threshold)
		if value < best_loss:
			best_loss, best_epoch = value, epoch
			best_state = {
				name: tensor.detach().cpu().clone()
				for name, tensor in model.state_dict().items()
			}
		stale = 0 if significant else stale + 1
		trailing = (
			normalized._trailing_relative_improvement(curve, convergence_window)
			if len(curve) >= convergence_window else math.inf
		)
		if epoch >= min_epochs and stale >= patience and trailing <= convergence_threshold:
			break
	_require(best_state is not None and np.isfinite(best_loss), 'Model fit failed.')
	model.load_state_dict(best_state)
	model.eval()
	trailing = normalized._trailing_relative_improvement(curve, convergence_window)
	converged = bool(len(curve) >= min_epochs and trailing <= convergence_threshold)
	metadata = {
		'model': (
			identifiable.MODEL_NAME if action_mode in ACTION_MODES
			else 'parameter_matched_nonlinear_action_capacity_control'
		),
		'action_mode': action_mode,
		'fit_data': 'real_same_state_sibling_branches_only',
		'train_root_ids': sorted(set(int(value) for value in train.root_ids)),
		'validation_root_ids': sorted(set(int(value) for value in validation.root_ids)),
		'test_roots_seen_during_fit_or_selection': False,
		'train_rows': int(len(train)), 'validation_rows': int(len(validation)),
		'parameter_count': int(sum(value.numel() for value in model.parameters())),
		'best_epoch': int(best_epoch), 'epochs_ran': int(len(curve)),
		'best_validation_objective': float(best_loss), 'converged': converged,
		'learning_curve': curve,
		'normalization': normalization_summary(normalization),
		'action_response_zero_anchor': 'physical_action_zero',
		'diagnostic_only_capacity_control': bool(
			action_mode in CAPACITY_CONTROL_MODES
		),
		'action_slot_semantics': (
			'actual_normalized_executed_action'
			if action_mode != 'nonlinear_actionless'
			else 'fixed_all_ones_no_executed_action_information'
		),
	}
	return identifiable.FittedActionIdentifiableTransition(
		action_mode, model, normalization, device, metadata,
	)


def _root_bootstrap_mean(
	values: Sequence[float], *, seed: int, resamples: int,
) -> dict:
	array = _finite(values, 'bootstrap values').astype(np.float64).reshape(-1)
	_require(len(array) >= 2 and resamples >= 1000, 'Bootstrap is underspecified.')
	rng = np.random.default_rng(int(seed))
	draw = rng.integers(0, len(array), size=(resamples, len(array)))
	samples = array[draw].mean(axis=1)
	return {
		'estimate': float(array.mean()),
		'ci95': [float(value) for value in np.quantile(samples, [0.025, 0.975])],
		'root_count': int(len(array)), 'resamples': int(resamples),
		'seed': int(seed),
		'unit': 'whole_root_sibling_and_twin_group',
	}


def _root_bootstrap_statistic(
	items: Sequence[object], statistic: Callable[[Sequence[object]], float], *,
	seed: int, resamples: int,
) -> dict:
	_require(len(items) >= 2 and resamples >= 1000, 'Bootstrap is underspecified.')
	estimate = float(statistic(items))
	_require(np.isfinite(estimate), 'Statistic is non-finite.')
	rng = np.random.default_rng(int(seed))
	samples = np.empty(resamples, dtype=np.float64)
	for index in range(resamples):
		selection = rng.integers(0, len(items), size=len(items))
		samples[index] = statistic([items[int(value)] for value in selection])
	_require(np.isfinite(samples).all(), 'Bootstrap statistic is non-finite.')
	return {
		'estimate': estimate,
		'ci95': [float(value) for value in np.quantile(samples, [0.025, 0.975])],
		'root_count': int(len(items)), 'resamples': int(resamples),
		'seed': int(seed),
		'unit': 'whole_root_sibling_and_twin_group',
	}


def _relative_gain_rows(
	aware: np.ndarray, baseline: np.ndarray, *, seed: int, resamples: int,
) -> dict:
	_require(aware.shape == baseline.shape and aware.ndim == 1,
		'Paired root errors are malformed.')
	return refit.paired_episode_bootstrap(
		aware, baseline, seed=seed, resamples=resamples,
	) | {
		'unit': 'whole_root_sibling_and_twin_group',
		'seed': int(seed), 'resamples': int(resamples),
		'root_count': int(len(aware)),
	}


def _ranking_auc(error_matrix: np.ndarray) -> float:
	"""Pairwise AUC: correct sibling score versus every other real sibling."""
	error = np.asarray(error_matrix, dtype=np.float64)
	_require(error.ndim == 2 and error.shape[0] == error.shape[1]
		and len(error) >= 2, 'Sibling ranking matrix is malformed.')
	wins = []
	for target in range(len(error)):
		correct = error[target, target]
		for candidate in range(len(error)):
			if candidate == target:
				continue
			wrong = error[target, candidate]
			wins.append(float(correct < wrong) + 0.5 * float(correct == wrong))
	return float(np.mean(wins))


def _zero_effect_r2(items: Sequence[tuple[np.ndarray, np.ndarray]]) -> float:
	true = np.concatenate([np.asarray(item[0]).reshape(-1) for item in items])
	predicted = np.concatenate([np.asarray(item[1]).reshape(-1) for item in items])
	denominator = float(np.square(true).sum())
	if denominator <= 1e-18:
		return 0.0
	return float(1.0 - np.square(predicted - true).sum() / denominator)


def _condition_root_metrics(
	group: RootGroup, encoded: Mapping[str, np.ndarray],
	fits: Mapping[str, identifiable.FittedActionIdentifiableTransition],
	normalization: normalized.DeltaNormalization,
) -> dict:
	_require(set(fits) == set(FIT_MODES), 'Complete fit family is required.')
	root_rows = build_rows(
		BranchDataset(Path('.'), {}, '', tuple(), group.action_dim, (group,), {
			'train': (group.root_id,), 'validation': (group.root_id,),
			'test': (group.root_id,),
		}), {group.root_id: encoded}, [group.root_id],
	)
	sequence = root_rows.sequence
	rollouts = {
		mode: fit.rollout(sequence.initial_u_history, sequence.action_window)
		for mode, fit in fits.items()
	}
	anchor_u = sequence.initial_u_history[:, -1]
	anchor_z = normalized.clr_to_simnorm(anchor_u, normalization.simnorm_dim)
	zero_index = int(np.flatnonzero(root_rows.is_zero)[0])
	result = {'root_id': group.root_id, 'zero_branch_index': zero_index, 'horizons': {}}
	for horizon in HORIZONS:
		target_u = sequence.target_future_u[:, horizon - 1]
		target_z = encoded['future_z'][:, horizon - 1]
		metrics = {}
		predictions = [
			(
				'aware' if mode == 'action_aware' else mode,
				rollouts[mode]['predicted_u'][horizon],
				rollouts[mode]['predicted_z'][horizon],
			)
			for mode in FIT_MODES
		] + [('persistence', anchor_u, anchor_z)]
		for name, predicted_u, predicted_z in predictions:
			arrays = normalized._metric_arrays(
				predicted_u, predicted_z, anchor_u, anchor_z, target_u, target_z,
				normalization.horizon_scale[horizon], normalization.simnorm_dim,
			)
			metrics[name] = {
				'aitchison_mse': float(arrays['aitchison_mse'].mean()),
				'fixed_train_horizon_nmse': float(
					arrays['fixed_clean_horizon_nmse'].mean()
				),
				'raw_latent_mse': float(arrays['raw_latent_mse'].mean()),
			}
		true_effect = target_u[~root_rows.is_zero] - target_u[zero_index]
		aware_diagnostics = {}
		for label, mode in (
			('small', 'action_aware'),
			('nonlinear_capacity_control', 'nonlinear_action_aware'),
		):
			predicted_candidates = rollouts[mode]['predicted_u'][horizon]
			difference = predicted_candidates[None] - target_u[:, None]
			error_matrix = np.mean(np.square(difference), axis=-1)
			wrong = error_matrix[~np.eye(group.branches, dtype=np.bool_)]
			predicted_effect = (
				predicted_candidates[~root_rows.is_zero]
				- predicted_candidates[zero_index]
			)
			aware_diagnostics[label] = {
				'real_sibling_action_ranking_auc': _ranking_auc(error_matrix),
				'correct_action_aitchison_mse': float(np.diag(error_matrix).mean()),
				'other_real_sibling_action_aitchison_mse': float(wrong.mean()),
				'action_effect_predicted': predicted_effect,
				'action_effect_r2_vs_zero_sibling': _zero_effect_r2([(
					true_effect, predicted_effect,
				)]),
			}
		result['horizons'][str(horizon)] = {
			'prediction': metrics,
			'real_sibling_action_ranking_auc': aware_diagnostics['small'][
				'real_sibling_action_ranking_auc'
			],
			'correct_action_aitchison_mse': aware_diagnostics['small'][
				'correct_action_aitchison_mse'
			],
			'other_real_sibling_action_aitchison_mse': aware_diagnostics['small'][
				'other_real_sibling_action_aitchison_mse'
			],
			'action_effect_true': true_effect,
			'action_effect_predicted': aware_diagnostics['small'][
				'action_effect_predicted'
			],
			'action_effect_r2_vs_zero_sibling': aware_diagnostics['small'][
				'action_effect_r2_vs_zero_sibling'
			],
			'nonlinear_capacity_control': aware_diagnostics[
				'nonlinear_capacity_control'
			],
		}
	return result


def _packet_validity(group: RootGroup, condition: str) -> dict:
	mask = np.asarray(group.arrays[f'{condition}__future__policy_object_mask'])
	exists = np.asarray(group.arrays[f'{condition}__future__policy_role_exists']) > 0.5
	nonempty = mask[..., -1, :, :].any(axis=(-2, -1))
	valid = exists & nonempty
	return {
		'root_id': group.root_id,
		'valid_bits': valid,
		'valid_rate': float(valid.mean()),
		'max_invalid_burst': int(max(
			_max_burst(~valid[branch, :, role])
			for branch in range(group.branches) for role in range(group.role_count)
		)),
	}


def _aggregate_condition(
	root_rows: Sequence[Mapping], validity_rows: Sequence[Mapping], *,
	seed: int, resamples: int,
) -> dict:
	result = {'horizons': {}}
	for horizon in HORIZONS:
		rows = [root['horizons'][str(horizon)] for root in root_rows]
		aware = np.asarray([
			row['prediction']['aware']['aitchison_mse'] for row in rows
		], dtype=np.float64)
		actionless = np.asarray([
			row['prediction']['actionless']['aitchison_mse'] for row in rows
		], dtype=np.float64)
		nonlinear_aware = np.asarray([
			row['prediction']['nonlinear_action_aware']['aitchison_mse']
			for row in rows
		], dtype=np.float64)
		nonlinear_actionless = np.asarray([
			row['prediction']['nonlinear_actionless']['aitchison_mse']
			for row in rows
		], dtype=np.float64)
		persistence = np.asarray([
			row['prediction']['persistence']['aitchison_mse'] for row in rows
		], dtype=np.float64)
		ranking = [row['real_sibling_action_ranking_auc'] for row in rows]
		effects = [
			(row['action_effect_true'], row['action_effect_predicted']) for row in rows
		]
		nonlinear_ranking = [
			row['nonlinear_capacity_control']['real_sibling_action_ranking_auc']
			for row in rows
		]
		nonlinear_effects = [(
			row['action_effect_true'],
			row['nonlinear_capacity_control']['action_effect_predicted'],
		) for row in rows]
		result['horizons'][str(horizon)] = {
			'root_rows': [{
				'root_id': int(root_rows[index]['root_id']),
				'aware_aitchison_mse': float(aware[index]),
				'actionless_aitchison_mse': float(actionless[index]),
				'nonlinear_aware_aitchison_mse': float(nonlinear_aware[index]),
				'nonlinear_actionless_aitchison_mse': float(
					nonlinear_actionless[index]
				),
				'persistence_aitchison_mse': float(persistence[index]),
				'real_sibling_action_ranking_auc': float(ranking[index]),
				'action_effect_r2_vs_zero_sibling': float(
					rows[index]['action_effect_r2_vs_zero_sibling']
				),
				'nonlinear_real_sibling_action_ranking_auc': float(
					nonlinear_ranking[index]
				),
				'nonlinear_action_effect_r2_vs_zero_sibling': float(
					rows[index]['nonlinear_capacity_control'][
						'action_effect_r2_vs_zero_sibling'
					]
				),
			} for index in range(len(rows))],
			'aware_error': _root_bootstrap_mean(
				aware, seed=identifiable._stable_seed(seed, horizon, 'aware'),
				resamples=resamples,
			),
			'aware_relative_gain_vs_persistence': _relative_gain_rows(
				aware, persistence,
				seed=identifiable._stable_seed(seed, horizon, 'persistence'),
				resamples=resamples,
			),
			'aware_relative_gain_vs_actionless': _relative_gain_rows(
				aware, actionless,
				seed=identifiable._stable_seed(seed, horizon, 'actionless'),
				resamples=resamples,
			),
			'real_sibling_action_ranking_auc': _root_bootstrap_mean(
				ranking, seed=identifiable._stable_seed(seed, horizon, 'ranking'),
				resamples=resamples,
			),
			'action_effect_r2_vs_zero_sibling': _root_bootstrap_statistic(
				effects, _zero_effect_r2,
				seed=identifiable._stable_seed(seed, horizon, 'effect_r2'),
				resamples=resamples,
			),
			'nonlinear_capacity_control': {
				'aware_error': _root_bootstrap_mean(
					nonlinear_aware,
					seed=identifiable._stable_seed(
						seed, horizon, 'nonlinear_aware_error',
					),
					resamples=resamples,
				),
				'aware_relative_gain_vs_parameter_matched_actionless': (
					_relative_gain_rows(
						nonlinear_aware, nonlinear_actionless,
						seed=identifiable._stable_seed(
							seed, horizon, 'nonlinear_actionless',
						),
						resamples=resamples,
					)
				),
				'real_sibling_action_ranking_auc': _root_bootstrap_mean(
					nonlinear_ranking,
					seed=identifiable._stable_seed(
						seed, horizon, 'nonlinear_ranking',
					),
					resamples=resamples,
				),
				'action_effect_r2_vs_zero_sibling': _root_bootstrap_statistic(
					nonlinear_effects, _zero_effect_r2,
					seed=identifiable._stable_seed(
						seed, horizon, 'nonlinear_effect_r2',
					),
					resamples=resamples,
				),
				'diagnostic_only': True,
			},
		}
	valid_rate = [row['valid_rate'] for row in validity_rows]
	max_burst = max(int(row['max_invalid_burst']) for row in validity_rows)
	result['packet_validity'] = {
		'by_root': [{
			'root_id': int(row['root_id']), 'valid_rate': float(row['valid_rate']),
			'max_invalid_burst': int(row['max_invalid_burst']),
		} for row in validity_rows],
		'valid_rate': _root_bootstrap_mean(
			valid_rate, seed=identifiable._stable_seed(seed, 'validity'),
			resamples=resamples,
		),
		'max_invalid_burst': int(max_burst),
		'validity_derived_from': 'deployable_role_exists_and_object_mask_only',
	}
	return result


def _background_twin_metrics(
	test_groups: Sequence[RootGroup], encoded: Mapping[str, Mapping[int, Mapping]],
	*, simnorm_dim: int, seed: int, resamples: int,
) -> dict:
	result = {'horizons': {}}
	for horizon in HORIZONS:
		rows = []
		for group in test_groups:
			clean = encoded['clean'][group.root_id]['future_u'][:, horizon - 1]
			hard = encoded['hard'][group.root_id]['future_u'][:, horizon - 1]
			zero = int(np.flatnonzero(group.arrays['branch__is_zero'])[0])
			nonzero = np.flatnonzero(~group.arrays['branch__is_zero'])
			background_distance = float(np.mean(np.sqrt(np.mean(
				np.square(clean - hard), axis=1,
			))))
			clean_action = np.sqrt(np.mean(np.square(
				clean[nonzero] - clean[zero]
			), axis=1))
			hard_action = np.sqrt(np.mean(np.square(
				hard[nonzero] - hard[zero]
			), axis=1))
			action_distance = float(np.mean(np.concatenate([clean_action, hard_action])))
			ratio = background_distance / max(action_distance, 1e-12)
			rows.append({
				'root_id': group.root_id,
				'background_twin_distance': background_distance,
				'real_action_sibling_distance': action_distance,
				'background_over_action_distance': float(ratio),
			})
		ratios = [row['background_over_action_distance'] for row in rows]
		result['horizons'][str(horizon)] = {
			'by_root': rows,
			'background_over_real_action_distance': _root_bootstrap_mean(
				ratios, seed=identifiable._stable_seed(seed, horizon, 'twin_ratio'),
				resamples=resamples,
			),
		}
	result['distance_space'] = 'frozen_ROF_CLR_latent_RMS'
	result['interpretation'] = (
		'lower_than_one_means_same-physics_background_twins_are_closer_than_'
		'different-real-action_siblings'
	)
	return result


def _oracle_data_validity(dataset: BranchDataset) -> dict:
	rows = []
	for group in dataset.groups:
		audit = validate_root_arrays(
			group.arrays, role_count=group.role_count, action_dim=group.action_dim,
			root_id=group.root_id,
		)
		rows.append(audit | {'split': group.split})
	return {
		'passed': True, 'roots': rows,
		'root_anchor_exact_match': True,
		'balanced_full_rank_real_actions': True,
		'immediate_nonzero_physics_divergence': True,
		'clean_hard_share_exact_physics_branch': True,
		'oracle_fields_used_for': 'dataset_validity_only',
		'oracle_fields_used_for_model_fit_selection_or_scoring': False,
	}


def _score_gates(
	condition_metrics: Mapping, twin_metrics: Mapping, training: Mapping,
) -> dict:
	conditions = {}
	capacity_conditions = {}
	for condition in CONDITIONS:
		cell = condition_metrics[condition]
		horizons = {}
		capacity_horizons = {}
		for horizon in HORIZONS:
			row = cell['horizons'][str(horizon)]
			gain_persistence = row['aware_relative_gain_vs_persistence']
			gain_actionless = row['aware_relative_gain_vs_actionless']
			ranking = row['real_sibling_action_ranking_auc']
			r2 = row['action_effect_r2_vs_zero_sibling']
			horizons[str(horizon)] = {
				'aware_beats_persistence': bool(
					gain_persistence['relative_improvement']['ci95'][0]
					>= GATE_THRESHOLDS['min_aware_gain_vs_persistence']
				),
				'aware_beats_parameter_matched_actionless': bool(
					gain_actionless['relative_improvement']['ci95'][0]
					>= GATE_THRESHOLDS['min_aware_gain_vs_actionless']
				),
				'real_sibling_ranking_pass': bool(
					ranking['ci95'][0] >= GATE_THRESHOLDS['min_ranking_auc']
				),
				'action_effect_r2_pass': bool(
					r2['ci95'][0] >= GATE_THRESHOLDS['min_action_effect_r2']
				),
			}
			capacity = row['nonlinear_capacity_control']
			capacity_gain = capacity[
				'aware_relative_gain_vs_parameter_matched_actionless'
			]
			capacity_horizons[str(horizon)] = {
				'aware_beats_parameter_matched_actionless': bool(
					capacity_gain['relative_improvement']['ci95'][0]
					>= GATE_THRESHOLDS['min_aware_gain_vs_actionless']
				),
				'real_sibling_ranking_pass': bool(
					capacity['real_sibling_action_ranking_auc']['ci95'][0]
					>= GATE_THRESHOLDS['min_ranking_auc']
				),
				'action_effect_r2_pass': bool(
					capacity['action_effect_r2_vs_zero_sibling']['ci95'][0]
					>= GATE_THRESHOLDS['min_action_effect_r2']
				),
			}
		validity = cell['packet_validity']
		validity_pass = bool(
			validity['valid_rate']['ci95'][0] >= GATE_THRESHOLDS['min_packet_valid_rate']
			and validity['max_invalid_burst'] <= GATE_THRESHOLDS['max_packet_invalid_burst']
		)
		conditions[condition] = {
			'horizons': horizons, 'packet_validity_pass': validity_pass,
			'all_pass': bool(validity_pass and all(
				all(row.values()) for row in horizons.values()
			)),
		}
		capacity_conditions[condition] = {
			'horizons': capacity_horizons,
			'all_pass': bool(all(
				all(row.values()) for row in capacity_horizons.values()
			)),
		}
	twin_horizons = {}
	for horizon in HORIZONS:
		ratio = twin_metrics['horizons'][str(horizon)][
			'background_over_real_action_distance'
		]
		twin_horizons[str(horizon)] = bool(
			ratio['ci95'][1]
			<= GATE_THRESHOLDS['max_background_twin_over_action_distance']
		)
	all_fits_converged = bool(all(
		training[mode].get('converged') is True for mode in ACTION_MODES
	))
	parameter_matched = bool(
		training['action_aware'].get('parameter_count')
		== training['actionless'].get('parameter_count')
	)
	capacity_fits_converged = bool(all(
		training[mode].get('converged') is True
		for mode in CAPACITY_CONTROL_MODES
	))
	capacity_parameter_matched = bool(
		training['nonlinear_action_aware'].get('parameter_count')
		== training['nonlinear_actionless'].get('parameter_count')
	)
	capacity_all_pass = bool(
		capacity_fits_converged and capacity_parameter_matched
		and all(row['all_pass'] for row in capacity_conditions.values())
	)
	all_pass = bool(
		all_fits_converged and parameter_matched
		and all(row['all_pass'] for row in conditions.values())
		and all(twin_horizons.values())
	)
	return {
		'thresholds': dict(GATE_THRESHOLDS), 'conditions': conditions,
		'background_twin_horizons': twin_horizons,
		'all_fits_converged': all_fits_converged,
		'parameter_matched_actionless': parameter_matched,
		'nonlinear_capacity_control': {
			'diagnostic_only': True,
			'conditions': capacity_conditions,
			'all_fits_converged': capacity_fits_converged,
			'parameter_matched_actionless': capacity_parameter_matched,
			'all_pass': capacity_all_pass,
			'interpretation': (
				'If the preregistered small pair fails while this paired nonlinear '
				'control passes, the small dynamics family is an identified '
				'capacity/model-class confound. Failure of both remains evidence '
				'against this representation/probe combination, not proof that no '
				'visual dynamics model can work.'
			),
		},
		'action_identifiable_representation_candidate': all_pass,
		'controller_training_authorized': False,
		'next_step_if_pass': (
			'independent_design_review_then_separate_small_controller_pilot'
		),
		'next_step_if_fail': 'do_not_scale_controller_fix_representation_or_dynamics',
	}


def _validate_bootstrap_record(
	record: Mapping, *, expected_seed: int, expected_roots: int,
) -> None:
	_require(record.get('unit') == 'whole_root_sibling_and_twin_group',
		'A metric bootstrap is not root clustered.')
	_require(record.get('seed') == int(expected_seed),
		'A metric bootstrap seed changed.')
	_require(record.get('resamples') == FORMAL_BOOTSTRAP_RESAMPLES,
		'A metric bootstrap resample count changed.')
	_require(record.get('root_count') == expected_roots,
		'A metric bootstrap root count changed.')


def validate_result(payload: Mapping) -> None:
	_require(payload.get('format') == RESULT_FORMAT
		and payload.get('status') == RESULT_STATUS, 'Result is incomplete.')
	_require(payload.get('engineering_pass') is True, 'Engineering pass is missing.')
	_require(payload.get('policy_training_performed') is False,
		'Policy training is forbidden.')
	_require(payload.get('controller_training_authorized') is False,
		'This evaluator must never authorize controller training.')
	_require(payload.get('privileged_fields_used_as_model_input') is False,
		'Privileged fields entered the model.')
	_require(payload.get('reward_or_return_used') is False,
		'Reward/return use is forbidden.')
	_require(payload.get('background_id_used_as_model_input') is False,
		'Background IDs entered the model.')
	protocol = payload.get('protocol', {})
	_require(protocol.get('split_unit') == 'whole_root_sibling_and_twin_group',
		'Root-cluster split contract is missing.')
	_require(protocol.get('bootstrap_unit') == 'whole_root_sibling_and_twin_group',
		'Root-cluster bootstrap contract is missing.')
	_require(protocol.get('real_interventions_only') is True,
		'Real-intervention guard is missing.')
	_require(protocol.get('shuffled_logged_futures_used') is False,
		'Shuffled logged futures are forbidden.')
	_require(protocol.get('fit_domain') == (
		'pooled_clean_hard_background_twins_without_condition_identifier'
	), 'A background-condition selector entered fitting.')
	_require(protocol.get('condition_or_background_identifier_used_for_fit') is False,
		'A background-condition identifier entered fitting.')
	fit_seed = protocol.get('seed')
	_require(fit_seed in FORMAL_FIT_SEEDS, 'Formal fit seed changed.')
	_require(protocol.get('hidden_dim') == FORMAL_HIDDEN_DIM,
		'Formal hidden dimension changed.')
	_require(protocol.get('bootstrap_resamples') == FORMAL_BOOTSTRAP_RESAMPLES,
		'Formal bootstrap resample count changed.')
	_require(protocol.get('bootstrap_seed') == (
		int(fit_seed) + FORMAL_BOOTSTRAP_SEED_OFFSET
	), 'Formal bootstrap seed changed.')
	_require(protocol.get('formal_split_counts') == FORMAL_SPLIT_COUNTS,
		'Formal split-count lock is missing.')
	_require(protocol.get('fit_models') == list(FIT_MODES),
		'Formal fit-model family changed.')
	oracle = payload.get('oracle_data_validity', {})
	_require(oracle.get('passed') is True, 'Oracle data validity did not pass.')
	_require(oracle.get('oracle_fields_used_for_model_fit_selection_or_scoring') is False,
		'Oracle fields escaped the data-validity boundary.')
	training = payload.get('training', {})
	_require(set(training) == set(FIT_MODES),
		'Required fit pairs are incomplete.')
	for mode in FIT_MODES:
		row = training[mode]
		_require(row.get('test_roots_seen_during_fit_or_selection') is False,
			'Test roots leaked into fitting.')
		_require(not (set(row['train_root_ids']) & set(row['validation_root_ids'])),
			'Train/validation roots overlap.')
	_require(
		training['action_aware']['parameter_count']
		== training['actionless']['parameter_count'],
		'Actionless baseline is not parameter matched.',
	)
	_require(
		training['nonlinear_action_aware']['parameter_count']
		== training['nonlinear_actionless']['parameter_count'],
		'Nonlinear capacity-control baseline is not parameter matched.',
	)
	splits = payload.get('root_splits', {})
	_require(isinstance(splits, Mapping) and set(splits) == set(SPLITS),
		'Root split table is incomplete.')
	sets = {name: set(splits[name]) for name in SPLITS}
	_require({name: len(sets[name]) for name in SPLITS} == FORMAL_SPLIT_COUNTS,
		'Formal 50/10/20 root split changed.')
	_require(not (sets['train'] & sets['validation'])
		and not (sets['train'] & sets['test'])
		and not (sets['validation'] & sets['test']), 'Root splits overlap.')
	for mode in FIT_MODES:
		_require(set(training[mode]['train_root_ids']) == sets['train']
			and set(training[mode]['validation_root_ids']) == sets['validation'],
			'Fit provenance does not match the root split table.')
	metrics = payload.get('condition_metrics', {})
	_require(set(metrics) == set(CONDITIONS), 'Condition metrics are incomplete.')
	for condition in CONDITIONS:
		condition_seed = identifiable._stable_seed(
			protocol['bootstrap_seed'], condition,
		)
		_require(set(metrics[condition].get('horizons', {})) == {
			str(value) for value in HORIZONS
		}, 'Prediction horizons are incomplete.')
		for horizon in HORIZONS:
			row = metrics[condition]['horizons'][str(horizon)]
			_require({item['root_id'] for item in row['root_rows']} == sets['test'],
				'Metric rows do not match held-out roots.')
			for key, seed_part in (
				('aware_error', 'aware'),
				('aware_relative_gain_vs_persistence', 'persistence'),
				('aware_relative_gain_vs_actionless', 'actionless'),
				('real_sibling_action_ranking_auc', 'ranking'),
				('action_effect_r2_vs_zero_sibling', 'effect_r2'),
			):
				_validate_bootstrap_record(
					row[key], expected_seed=identifiable._stable_seed(
						condition_seed, horizon, seed_part,
					), expected_roots=FORMAL_SPLIT_COUNTS['test'],
				)
			capacity = row.get('nonlinear_capacity_control', {})
			_require(capacity.get('diagnostic_only') is True,
				'Nonlinear capacity control lost its diagnostic-only boundary.')
			for key, seed_part in (
				('aware_error', 'nonlinear_aware_error'),
				('aware_relative_gain_vs_parameter_matched_actionless',
				 'nonlinear_actionless'),
				('real_sibling_action_ranking_auc', 'nonlinear_ranking'),
				('action_effect_r2_vs_zero_sibling', 'nonlinear_effect_r2'),
			):
				_validate_bootstrap_record(
					capacity.get(key, {}),
					expected_seed=identifiable._stable_seed(
						condition_seed, horizon, seed_part,
					), expected_roots=FORMAL_SPLIT_COUNTS['test'],
				)
		validity = metrics[condition].get('packet_validity', {})
		_validate_bootstrap_record(
			validity.get('valid_rate', {}),
			expected_seed=identifiable._stable_seed(condition_seed, 'validity'),
			expected_roots=FORMAL_SPLIT_COUNTS['test'],
		)
	twins = payload.get('background_twin_metrics', {}).get('horizons', {})
	_require(set(twins) == {str(value) for value in HORIZONS},
		'Background-twin horizons are incomplete.')
	twin_seed = identifiable._stable_seed(protocol['bootstrap_seed'], 'twins')
	for horizon in HORIZONS:
		_validate_bootstrap_record(
			twins[str(horizon)]['background_over_real_action_distance'],
			expected_seed=identifiable._stable_seed(
				twin_seed, horizon, 'twin_ratio',
			), expected_roots=FORMAL_SPLIT_COUNTS['test'],
		)
	_require(payload.get('action_identifiable_representation_candidate') is
		payload.get('gates', {}).get('action_identifiable_representation_candidate'),
		'Top-level scientific decision disagrees with the gates.')
	_require(payload.get('gates', {}).get('controller_training_authorized') is False,
		'Gate output improperly authorizes controller training.')
	_require(payload.get('gates', {}).get(
		'nonlinear_capacity_control', {}
	).get('diagnostic_only') is True,
		'Nonlinear capacity control must remain diagnostic only.')
	json.dumps(_json_safe(payload), allow_nan=False)


def evaluate(args) -> dict:
	dataset = load_branch_dataset(args.dataset, require_collector_contract=True)
	_require({name: len(dataset.root_splits[name]) for name in SPLITS}
		== FORMAL_SPLIT_COUNTS, 'Formal evaluation requires exactly 50/10/20 roots.')
	_require(args.seed in FORMAL_FIT_SEEDS, 'Unexpected formal fit seed.')
	_require(args.hidden_dim == FORMAL_HIDDEN_DIM,
		'Formal evaluation requires hidden_dim=128.')
	_require(args.bootstrap_resamples == FORMAL_BOOTSTRAP_RESAMPLES,
		'Formal evaluation requires exactly 20000 bootstrap resamples.')
	_require(args.bootstrap_seed == args.seed + FORMAL_BOOTSTRAP_SEED_OFFSET,
		'Formal bootstrap seed must equal fit seed + 1000003.')
	source = _validate_source_binding(dataset, args.runtime_config, args.checkpoint)
	agent = _load_agent(dataset, args.runtime_config.resolve(), args.checkpoint.resolve())
	for parameter in agent.model.parameters():
		parameter.requires_grad_(False)
	agent.eval()
	simnorm_dim = int(agent.cfg.get('simnorm_dim', 8))
	encoded = encode_dataset(dataset, agent, batch_size=args.encoder_batch_size)
	# Clean/hard twins are pooled as ordinary observations.  Neither the model,
	# optimizer, nor model selection receives a condition/background identifier.
	train = concatenate_rows([
		build_rows(dataset, encoded[condition], dataset.root_splits['train'])
		for condition in CONDITIONS
	])
	validation = concatenate_rows([
		build_rows(dataset, encoded[condition], dataset.root_splits['validation'])
		for condition in CONDITIONS
	])
	normalization = fit_branch_normalization(train, simnorm_dim=simnorm_dim)
	fits, training = {}, {}
	for mode in FIT_MODES:
		fit = fit_model(
			mode, train, validation, normalization, device=agent.device,
			seed=identifiable._stable_seed(args.seed, 'pooled_twins', mode),
			hidden_dim=args.hidden_dim, batch_size=args.fit_batch_size,
			max_epochs=args.max_epochs, min_epochs=args.min_epochs,
			patience=args.patience, convergence_window=args.convergence_window,
			convergence_threshold=args.convergence_threshold,
			learning_rate=args.learning_rate, weight_decay=args.weight_decay,
			inverse_weight=args.inverse_weight,
			control_l2_weight=args.control_l2_weight,
		)
		fits[mode] = fit
		training[mode] = dict(fit.metadata)

	test_groups = dataset.split('test')
	condition_metrics = {}
	for condition in CONDITIONS:
		root_rows = [
			_condition_root_metrics(
				group, encoded[condition][group.root_id],
				fits, normalization,
			)
			for group in test_groups
		]
		validity = [_packet_validity(group, condition) for group in test_groups]
		condition_metrics[condition] = _aggregate_condition(
			root_rows, validity,
			seed=identifiable._stable_seed(args.bootstrap_seed, condition),
			resamples=args.bootstrap_resamples,
		)
	twin_metrics = _background_twin_metrics(
		test_groups, encoded, simnorm_dim=simnorm_dim,
		seed=identifiable._stable_seed(args.bootstrap_seed, 'twins'),
		resamples=args.bootstrap_resamples,
	)
	oracle = _oracle_data_validity(dataset)
	gates = _score_gates(condition_metrics, twin_metrics, training)
	payload = {
		'format': RESULT_FORMAT, 'status': RESULT_STATUS,
		'engineering_pass': True,
		'action_identifiable_representation_candidate': gates[
			'action_identifiable_representation_candidate'
		],
		'policy_training_performed': False,
		'controller_training_authorized': False,
		'privileged_fields_used_as_model_input': False,
		'reward_or_return_used': False,
		'background_id_used_as_model_input': False,
		'task': dataset.task,
		'recommendation': (
			'independent_review_before_any_small_controller_pilot'
			if gates['action_identifiable_representation_candidate']
			else 'do_not_scale_controller_fix_representation_or_dynamics'
		),
		'source': {
			'dataset': str(dataset.manifest_path),
			'dataset_sha256': _sha256(dataset.manifest_path), **source,
			'encoder_frozen': True, 'controller_heads_frozen': True,
		},
		'protocol': {
			'dataset_format': DATASET_FORMAT, 'conditions': list(CONDITIONS),
			'history_frames': HISTORY_FRAMES, 'horizons': list(HORIZONS),
			'split_unit': 'whole_root_sibling_and_twin_group',
			'bootstrap_unit': 'whole_root_sibling_and_twin_group',
			'bootstrap_resamples': int(args.bootstrap_resamples),
			'bootstrap_seed': int(args.bootstrap_seed),
			'formal_split_counts': dict(FORMAL_SPLIT_COUNTS),
			'hidden_dim': int(args.hidden_dim),
			'real_interventions_only': True,
			'shuffled_logged_futures_used': False,
			'fit_domain': (
				'pooled_clean_hard_background_twins_without_condition_identifier'
			),
			'condition_or_background_identifier_used_for_fit': False,
			'model_input_fields': [
				f'policy_{field}' for field in POLICY_FIELDS
			] + ['history__action', 'branch__action'],
			'oracle_fields': list(ORACLE_FIELDS),
			'oracle_boundary': 'dataset_validity_only',
			'fit_models': list(FIT_MODES), 'seed': int(args.seed),
		},
		'root_splits': {
			name: [int(value) for value in dataset.root_splits[name]]
			for name in SPLITS
		},
		'normalization': normalization_summary(normalization),
		'training': training, 'condition_metrics': condition_metrics,
		'background_twin_metrics': twin_metrics,
		'oracle_data_validity': oracle, 'gates': gates,
		'limitations': [
			'The diagnostic tests a frozen ROF representation and a small decomposed '
			'dynamics family; failure does not prove all visual dynamics impossible.',
			'The paired nonlinear capacity control is diagnostic only and does not '
			'participate in the preregistered candidate gate.',
			'Packet validity checks deployable nonempty masks and declared roles, not '
			'privileged semantic correctness of those masks.',
			'Passing establishes action-identifiable predictive structure on the sampled '
			'real branches, not end-to-end controller improvement.',
			'Controller training remains a separate, manually reviewed experiment.',
		],
	}
	validate_result(payload)
	return payload


def _json_safe(value):
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, Mapping):
		return {str(key): _json_safe(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [_json_safe(item) for item in value]
	return value


def _atomic_json(path: Path, payload: Mapping) -> None:
	path = path.resolve()
	if path.exists():
		raise FileExistsError(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
	try:
		temporary.write_text(
			json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + '\n',
			encoding='utf-8',
		)
		os.replace(temporary, path)
	finally:
		if temporary.exists():
			temporary.unlink()


def main(argv=None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--dataset', type=Path, required=True)
	parser.add_argument('--runtime-config', type=Path, required=True)
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--seed', type=int, default=20260913)
	parser.add_argument('--bootstrap-seed', type=int, default=None)
	parser.add_argument('--bootstrap-resamples', type=int, default=20_000)
	parser.add_argument('--encoder-batch-size', type=int, default=256)
	parser.add_argument('--fit-batch-size', type=int, default=512)
	parser.add_argument('--hidden-dim', type=int, default=FORMAL_HIDDEN_DIM)
	parser.add_argument('--max-epochs', type=int, default=200)
	parser.add_argument('--min-epochs', type=int, default=50)
	parser.add_argument('--patience', type=int, default=30)
	parser.add_argument('--convergence-window', type=int, default=20)
	parser.add_argument('--convergence-threshold', type=float, default=0.01)
	parser.add_argument('--learning-rate', type=float, default=3e-4)
	parser.add_argument('--weight-decay', type=float, default=1e-5)
	parser.add_argument('--inverse-weight', type=float, default=0.10)
	parser.add_argument('--control-l2-weight', type=float, default=1e-4)
	args = parser.parse_args(argv)
	if args.bootstrap_seed is None:
		args.bootstrap_seed = args.seed + FORMAL_BOOTSTRAP_SEED_OFFSET
	if args.bootstrap_resamples < 1000:
		parser.error('--bootstrap-resamples must be at least 1000.')
	if min(args.encoder_batch_size, args.fit_batch_size, args.hidden_dim) < 1:
		parser.error('Batch and hidden dimensions must be positive.')
	if not (args.max_epochs >= args.min_epochs >= args.convergence_window >= 2):
		parser.error('Invalid convergence schedule.')
	if args.patience < 1 or not (0.0 < args.convergence_threshold <= 0.05):
		parser.error('Invalid convergence controls.')
	if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
		parser.error('Invalid optimizer controls.')
	if args.inverse_weight < 0.0 or args.control_l2_weight < 0.0:
		parser.error('Auxiliary weights must be nonnegative.')
	payload = evaluate(args)
	_atomic_json(args.output, payload)
	print('ROF_REAL_ACTION_BRANCH_COMPLETE')
	print(
		'ACTION_IDENTIFIABLE_REPRESENTATION_CANDIDATE=',
		payload['action_identifiable_representation_candidate'], sep='',
	)
	print('CONTROLLER_TRAINING_AUTHORIZED=False')
	print(f'OUTPUT={args.output.resolve()}')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
