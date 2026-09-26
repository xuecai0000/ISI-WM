"""Past-only frozen-latent normalized-innovation diagnostic for ROF-WM.

This is an offline diagnostic, never policy or controller training.  It freezes
one already-trained Robust Object Field (ROF) encoder and asks whether its
exact SimNorm latent contains enough dynamics information for an independently
fitted action-conditioned transition.  Two models are compared with nearly
identical parameter counts:

* ``normalized_delta_markov`` sees only ``(z_t, a_t)``;
* ``normalized_delta_history`` additionally sees two past latent innovations
  and the two preceding actions.

The latent is converted group-wise from SimNorm probabilities to centered log
ratios (CLR).  Models predict a one-step CLR innovation standardized with
train-episode statistics.  The innovation is projected onto the zero-sum CLR
tangent space before it is accumulated, and conversion back to a SimNorm
latent is an exact group softmax.  Fitting uses a joint horizon-1/3/5 open-loop
loss with no teacher forcing after the initial past-only context.  This avoids the
raw-next-latent/persistence shortcut exposed by the preceding refit diagnostic.

Clean and hard models are fitted independently on exactly 12 whole train
episodes and selected on 4 whole validation episodes.  Both are evaluated on
the untouched 4 clean and 4 hard test episodes.  Policy observations and
behaviour actions are the only source arrays loaded; simulator labels, masks,
rewards, returns, and termination flags never enter this diagnostic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from tdmpc2.tools import evaluate_rof_causal_ladder as ladder
from tdmpc2.tools import evaluate_rof_transition_refit as raw_refit


FORMAT = 'rof_normalized_delta_diagnostic_v1'
STATUS = 'rof_normalized_delta_complete'
HORIZONS = (1, 3, 5)
MAX_HORIZON = max(HORIZONS)
HISTORY_LENGTH = 3
FIT_CONDITIONS = ('clean', 'hard')
ARCHITECTURES = ('normalized_delta_markov', 'normalized_delta_history')
ACTION_MODES = ('action_aware', 'actionless')
METHODS = ('persistence', 'checkpoint_transition') + ARCHITECTURES
SPLIT_COUNTS = {'train': 12, 'validation': 4, 'test': 4}
CLR_EPSILON = 1e-6
SCALE_FLOOR = 1e-4

PREREGISTERED_GATES = {
	'primary_metric': 'groupwise_clr_aitchison_mse',
	'required_horizons': [1, 3],
	'min_relative_gain_vs_persistence': 0.30,
	'min_relative_gain_vs_shuffled_action': 0.10,
	'min_relative_gain_vs_actionless': 0.10,
	'min_episode_wins_out_of_4': 3,
	'paired_bootstrap_ci_lower_must_exceed_zero': True,
	'max_h5_real_over_persistence': 1.0,
	'min_history_relative_gain_vs_markov': 0.20,
	'parameter_ratio_tolerance': 0.05,
	'convergence_window_epochs': 20,
	'max_trailing_validation_relative_improvement': 0.01,
	'min_action_std_for_attribution': 0.05,
	'min_action_range_for_attribution': 0.20,
	'min_action_cov_effective_rank_fraction': 0.50,
	'min_local_action_variance_ratio': 0.02,
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


def _stable_seed(base: int, *parts: object) -> int:
	text = '\x1f'.join([str(int(base)), *(str(value) for value in parts)])
	value = int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8], 'little')
	return int(value % (2 ** 32 - 1))


def _condition(dataset: ladder.FrozenDataset) -> str:
	return raw_refit._condition(dataset)


def _validate_exact_splits(dataset: ladder.FrozenDataset) -> None:
	_require(set(dataset.splits) == set(SPLIT_COUNTS), 'Dataset split names are invalid.')
	all_ids = []
	for name, expected in SPLIT_COUNTS.items():
		ids = [int(value) for value in dataset.splits[name]]
		_require(len(ids) == expected, f'{name} must contain exactly {expected} episodes.')
		_require(len(ids) == len(set(ids)), f'{name} contains duplicate episode ids.')
		all_ids.extend(ids)
	_require(len(all_ids) == len(set(all_ids)), 'Train/validation/test episodes overlap.')
	available = {int(episode.episode_id) for episode in dataset.episodes}
	_require(set(all_ids) == available, 'Dataset splits do not cover exactly the episodes.')


def load_policy_dataset(path: Path | str) -> ladder.FrozenDataset:
	"""Delegate to the audited loader that materializes policy fields/actions only."""
	dataset = raw_refit.load_policy_dataset(path)
	_validate_exact_splits(dataset)
	return dataset


def action_mapping(dataset: ladder.FrozenDataset) -> dict[int, np.ndarray]:
	return raw_refit.action_mapping(dataset)


def _reshape_groups(value: np.ndarray, simnorm_dim: int) -> np.ndarray:
	value = np.asarray(value)
	_require(value.ndim >= 1 and value.shape[-1] % simnorm_dim == 0,
		'Latent width must be divisible by the SimNorm group width.')
	return value.reshape(*value.shape[:-1], -1, simnorm_dim)


def clr_transform(
	z: np.ndarray, simnorm_dim: int, *, epsilon: float = CLR_EPSILON,
	return_audit: bool = False,
):
	"""Map grouped simplex probabilities to zero-sum centered log ratios."""
	z = np.asarray(z, dtype=np.float64)
	_require(epsilon > 0.0 and np.isfinite(z).all(), 'CLR input/epsilon is invalid.')
	groups = _reshape_groups(z, simnorm_dim)
	_require(float(groups.min()) >= -1e-6, 'SimNorm latent contains negative probabilities.')
	input_sum_error = np.abs(groups.sum(axis=-1) - 1.0)
	clipped = np.maximum(groups, epsilon)
	clipped_fraction = float(np.mean(groups < epsilon))
	probability = clipped / clipped.sum(axis=-1, keepdims=True)
	log_probability = np.log(probability)
	clr = log_probability - log_probability.mean(axis=-1, keepdims=True)
	clr = clr.reshape(z.shape).astype(np.float32)
	audit = {
		'epsilon': float(epsilon),
		'input_min_probability': float(groups.min()),
		'input_max_group_sum_error': float(input_sum_error.max()),
		'clipped_probability_fraction': clipped_fraction,
		'output_max_abs_group_mean': float(
			np.abs(_reshape_groups(clr, simnorm_dim).mean(axis=-1)).max()
		),
	}
	return (clr, audit) if return_audit else clr


def clr_to_simnorm(u: np.ndarray, simnorm_dim: int) -> np.ndarray:
	"""Invert CLR up to its additive group constant using a stable softmax."""
	u = np.asarray(u, dtype=np.float64)
	_require(np.isfinite(u).all(), 'CLR prediction is non-finite.')
	groups = _reshape_groups(u, simnorm_dim)
	groups = groups - groups.max(axis=-1, keepdims=True)
	exponential = np.exp(groups)
	probability = exponential / exponential.sum(axis=-1, keepdims=True)
	return probability.reshape(u.shape).astype(np.float32)


def project_clr_innovation(delta: np.ndarray, simnorm_dim: int) -> tuple[np.ndarray, dict]:
	"""Project arbitrary innovations onto each CLR group's zero-sum tangent space."""
	delta = np.asarray(delta, dtype=np.float64)
	_require(np.isfinite(delta).all(), 'CLR innovation is non-finite.')
	groups = _reshape_groups(delta, simnorm_dim)
	projected = groups - groups.mean(axis=-1, keepdims=True)
	correction = groups - projected
	return projected.reshape(delta.shape).astype(np.float32), {
		'pre_projection_max_abs_group_sum': float(np.abs(groups.sum(axis=-1)).max()),
		'projection_correction_rms': float(np.sqrt(np.mean(np.square(correction)))),
		'post_projection_max_abs_group_sum': float(
			np.abs(projected.sum(axis=-1)).max()
		),
	}


def simplex_audit(z: np.ndarray, simnorm_dim: int) -> dict:
	z = np.asarray(z, dtype=np.float64)
	_require(np.isfinite(z).all(), 'Predicted latent is non-finite.')
	groups = _reshape_groups(z, simnorm_dim)
	return {
		'min_probability': float(groups.min()),
		'max_probability': float(groups.max()),
		'max_group_sum_error': float(np.abs(groups.sum(axis=-1) - 1.0).max()),
	}


def encode_clr_latents(
	latents: Mapping[int, np.ndarray], simnorm_dim: int,
) -> tuple[dict[int, np.ndarray], dict]:
	result = {}
	audits = []
	for episode_id, value in latents.items():
		result[int(episode_id)], audit = clr_transform(
			value, simnorm_dim, return_audit=True,
		)
		audits.append(audit)
	return result, {
		'episodes': len(result),
		'epsilon': CLR_EPSILON,
		'max_input_group_sum_error': max(
			row['input_max_group_sum_error'] for row in audits
		),
		'max_clr_group_mean_error': max(
			row['output_max_abs_group_mean'] for row in audits
		),
		'max_clipped_probability_fraction_per_episode': max(
			row['clipped_probability_fraction'] for row in audits
		),
	}


def clr_epsilon_sensitivity(
	latents: Mapping[int, np.ndarray], simnorm_dim: int,
	*, epsilon_values: Sequence[float] = (1e-7, 1e-6, 1e-5),
) -> dict:
	"""Report how much the frozen CLR coordinates depend on clipping epsilon."""
	_require(CLR_EPSILON in epsilon_values, 'Sensitivity grid must include default epsilon.')
	default = {
		episode_id: clr_transform(value, simnorm_dim, epsilon=CLR_EPSILON)
		for episode_id, value in latents.items()
	}
	result = {}
	for epsilon in epsilon_values:
		squared_sum, count, maximum, clipped, probabilities = 0.0, 0, 0.0, 0, 0
		for episode_id, value in latents.items():
			candidate = clr_transform(value, simnorm_dim, epsilon=float(epsilon))
			difference = candidate.astype(np.float64) - default[episode_id].astype(np.float64)
			squared_sum += float(np.square(difference).sum())
			count += int(difference.size)
			maximum = max(maximum, float(np.abs(difference).max()))
			array = np.asarray(value)
			clipped += int(np.count_nonzero(array < epsilon))
			probabilities += int(array.size)
		result[f'{epsilon:.0e}'] = {
			'clr_rms_difference_vs_1e_minus_6': float(math.sqrt(squared_sum / count)),
			'clr_max_abs_difference_vs_1e_minus_6': maximum,
			'clipped_probability_fraction': float(clipped / probabilities),
		}
	return result


@dataclass(frozen=True)
class DeltaNormalization:
	u_mean: np.ndarray
	u_scale: np.ndarray
	action_mean: np.ndarray
	action_scale: np.ndarray
	delta_mean: np.ndarray
	delta_scale: np.ndarray
	horizon_scale: Mapping[int, np.ndarray]
	simnorm_dim: int
	floor_audit: Mapping[str, Mapping]


def _mean_std(value: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray, int]:
	value = np.asarray(value, dtype=np.float64)
	mean = value.mean(axis=0)
	std = value.std(axis=0)
	floored = int(np.count_nonzero(std < floor))
	std = np.maximum(std, floor)
	return mean.astype(np.float32), std.astype(np.float32), floored


def _group_scalar_scale(
	value: np.ndarray, center: np.ndarray, *, simnorm_dim: int, floor: float,
) -> tuple[np.ndarray, dict]:
	"""One isotropic scale per SimNorm group, broadcast across its coordinates."""
	value = np.asarray(value, dtype=np.float64)
	center = np.asarray(center, dtype=np.float64)
	_require(value.ndim == 2 and center.shape == (value.shape[1],),
		'Group-scale rows/center are malformed.')
	residual = _reshape_groups(value - center, simnorm_dim)
	group_scale = np.sqrt(np.mean(np.square(residual), axis=(0, 2)))
	floored = group_scale < floor
	group_scale = np.maximum(group_scale, floor)
	broadcast = np.repeat(group_scale, simnorm_dim).astype(np.float32)
	return broadcast, {
		'groups': int(len(group_scale)),
		'floored_groups': int(np.count_nonzero(floored)),
		'floored_fraction': float(np.mean(floored)),
		'group_scalar_min': float(group_scale.min()),
		'group_scalar_mean': float(group_scale.mean()),
		'group_scalar_max': float(group_scale.max()),
	}


def fit_delta_normalization(
	clr_latents: Mapping[int, np.ndarray], actions: Mapping[int, np.ndarray],
	train_episode_ids: Sequence[int], *, simnorm_dim: int,
) -> DeltaNormalization:
	"""Fit every statistic on selected train episodes only."""
	_require(len(train_episode_ids) == SPLIT_COUNTS['train'],
		'Normalization requires exactly the 12 train episodes.')
	u_rows, action_rows, delta_rows = [], [], []
	horizon_rows = {horizon: [] for horizon in HORIZONS}
	for episode_id in train_episode_ids:
		u = np.asarray(clr_latents[int(episode_id)], dtype=np.float32)
		action = np.asarray(actions[int(episode_id)], dtype=np.float32)
		_require(u.ndim == 2 and action.ndim == 2 and len(u) == len(action) + 1,
			f'Episode {episode_id} violates observation/action alignment.')
		_require(len(action) >= MAX_HORIZON + HISTORY_LENGTH - 1,
			f'Episode {episode_id} is too short for the past-only rollout.')
		u_rows.append(u)
		action_rows.append(action)
		delta_rows.append(np.diff(u, axis=0))
		for horizon in HORIZONS:
			horizon_rows[horizon].append(u[horizon:] - u[:-horizon])
	u_joined = np.concatenate(u_rows, axis=0)
	action_joined = np.concatenate(action_rows, axis=0)
	delta_joined = np.concatenate(delta_rows, axis=0)
	u_mean = u_joined.mean(axis=0).astype(np.float32)
	u_mean, _ = project_clr_innovation(u_mean[None], simnorm_dim)
	u_mean = u_mean[0]
	u_scale, u_floor = _group_scalar_scale(
		u_joined, u_mean, simnorm_dim=simnorm_dim, floor=SCALE_FLOOR,
	)
	action_mean, action_scale, action_floored = _mean_std(
		action_joined, SCALE_FLOOR,
	)
	delta_mean = delta_joined.mean(axis=0).astype(np.float32)
	delta_mean, _ = project_clr_innovation(delta_mean[None], simnorm_dim)
	delta_mean = delta_mean[0]
	delta_scale, delta_floor = _group_scalar_scale(
		delta_joined, delta_mean, simnorm_dim=simnorm_dim, floor=SCALE_FLOOR,
	)
	horizon_scale = {}
	horizon_floor = {}
	for horizon in HORIZONS:
		displacement = np.concatenate(horizon_rows[horizon], axis=0).astype(np.float64)
		horizon_scale[horizon], horizon_floor[horizon] = _group_scalar_scale(
			displacement, np.zeros(displacement.shape[1], dtype=np.float32),
			simnorm_dim=simnorm_dim, floor=SCALE_FLOOR,
		)
	return DeltaNormalization(
		u_mean=u_mean, u_scale=u_scale,
		action_mean=action_mean, action_scale=action_scale,
		delta_mean=delta_mean, delta_scale=delta_scale,
		horizon_scale=horizon_scale, simnorm_dim=int(simnorm_dim),
		floor_audit={
			'u_scale': u_floor,
			'action_scale': {
				'dimensions': int(len(action_scale)),
				'floored_dimensions': int(action_floored),
				'floored_fraction': float(action_floored / max(len(action_scale), 1)),
			},
			'delta_scale': delta_floor,
			'horizon_scale': {
				str(horizon): horizon_floor[horizon] for horizon in HORIZONS
			},
		},
	)


def normalization_summary(value: DeltaNormalization) -> dict:
	def stats(array: np.ndarray) -> dict:
		array = np.asarray(array, dtype=np.float64)
		return {
			'min': float(array.min()), 'mean': float(array.mean()),
			'max': float(array.max()),
		}
	return {
		'latent_dim': int(len(value.u_mean)),
		'action_dim': int(len(value.action_mean)),
		'simnorm_dim': int(value.simnorm_dim),
		'clr_epsilon': CLR_EPSILON,
		'scale_floor': SCALE_FLOOR,
		'u_scale': stats(value.u_scale),
		'action_scale': stats(value.action_scale),
		'one_step_delta_scale': stats(value.delta_scale),
		'horizon_delta_rms_scale': {
			str(horizon): stats(value.horizon_scale[horizon]) for horizon in HORIZONS
		},
		'group_scalar_scale_contract': True,
		'floor_audit': dict(value.floor_audit),
		'statistics_source': 'fit_domain_train_episodes_only',
	}


def action_coverage_audit(
	dataset: ladder.FrozenDataset, clr_latents: Mapping[int, np.ndarray],
	normalization: DeltaNormalization, episode_ids: Sequence[int],
) -> dict:
	"""Audit whether behaviour data can identify incremental action information."""
	action_rows, state_rows, lag_current, lag_next = [], [], [], []
	for episode_id in episode_ids:
		episode = next(
			row for row in dataset.episodes if int(row.episode_id) == int(episode_id)
		)
		action = np.asarray(episode.arrays['action'], dtype=np.float64)
		u = np.asarray(clr_latents[int(episode_id)], dtype=np.float64)
		_require(len(u) == len(action) + 1, 'Action coverage alignment is invalid.')
		action_rows.append(action)
		state_rows.append(u[:-1])
		if len(action) > 1:
			lag_current.append(action[:-1])
			lag_next.append(action[1:])
	action = np.concatenate(action_rows, axis=0)
	state = np.concatenate(state_rows, axis=0)
	mean = action.mean(axis=0)
	std = action.std(axis=0)
	minimum = action.min(axis=0)
	maximum = action.max(axis=0)
	range_ = maximum - minimum
	centered = action - mean
	covariance = centered.T @ centered / max(len(centered), 1)
	eigenvalues = np.linalg.eigvalsh(np.atleast_2d(covariance))
	eigenvalues = np.maximum(eigenvalues, 0.0)
	effective_rank = (
		float(np.square(eigenvalues.sum()) / np.square(eigenvalues).sum())
		if float(np.square(eigenvalues).sum()) > 1e-20 else 0.0
	)
	if lag_current:
		x = np.concatenate(lag_current, axis=0)
		y = np.concatenate(lag_next, axis=0)
		x_center = x - x.mean(axis=0)
		y_center = y - y.mean(axis=0)
		denominator = np.sqrt(
			np.mean(np.square(x_center), axis=0)
			* np.mean(np.square(y_center), axis=0)
		)
		lag1 = np.divide(
			np.mean(x_center * y_center, axis=0), denominator,
			out=np.zeros_like(denominator), where=denominator > 1e-12,
		)
	else:
		lag1 = np.zeros(action.shape[1], dtype=np.float64)
	# A bounded nearest-neighbour audit estimates whether similar frozen states
	# ever receive different behaviour actions. Low local variation makes the
	# incremental contribution of the action statistically unidentifiable.
	count = min(len(state), 256)
	index = np.unique(np.linspace(0, len(state) - 1, count, dtype=np.int64))
	state_sample = (
		state[index] - normalization.u_mean
	) / normalization.u_scale
	action_sample = action[index]
	distance = np.empty((len(index), len(index)), dtype=np.float64)
	for start in range(0, len(index), 32):
		difference = state_sample[start:start + 32, None] - state_sample[None]
		distance[start:start + 32] = np.mean(np.square(difference), axis=-1)
	np.fill_diagonal(distance, np.inf)
	nearest = np.argmin(distance, axis=1)
	local_squared_difference = np.mean(
		np.square(action_sample - action_sample[nearest]), axis=0,
	)
	local_variance_ratio = local_squared_difference / np.maximum(2.0 * std ** 2, 1e-12)
	reasons = []
	if np.any(std < PREREGISTERED_GATES['min_action_std_for_attribution']):
		reasons.append('action_std_below_threshold')
	if np.any(range_ < PREREGISTERED_GATES['min_action_range_for_attribution']):
		reasons.append('action_range_below_threshold')
	if effective_rank / max(action.shape[1], 1) < PREREGISTERED_GATES[
		'min_action_cov_effective_rank_fraction'
	]:
		reasons.append('action_covariance_effective_rank_below_threshold')
	if np.any(local_variance_ratio < PREREGISTERED_GATES['min_local_action_variance_ratio']):
		reasons.append('nearest_neighbor_local_action_variance_below_threshold')
	return {
		'episode_ids': [int(value) for value in episode_ids],
		'frames': int(len(action)), 'action_dim': int(action.shape[1]),
		'per_dimension': {
			'mean': mean.tolist(), 'std': std.tolist(),
			'min': minimum.tolist(), 'max': maximum.tolist(), 'range': range_.tolist(),
			'fraction_abs_ge_0_95': np.mean(np.abs(action) >= 0.95, axis=0).tolist(),
			'fraction_ge_0_95': np.mean(action >= 0.95, axis=0).tolist(),
			'fraction_le_minus_0_95': np.mean(action <= -0.95, axis=0).tolist(),
			'lag1_autocorrelation': lag1.tolist(),
			'nearest_neighbor_local_action_variance_ratio': local_variance_ratio.tolist(),
		},
		'covariance_effective_rank': effective_rank,
		'covariance_effective_rank_fraction': float(
			effective_rank / max(action.shape[1], 1)
		),
		'nearest_neighbor_sample_count': int(len(index)),
		'coverage_sufficient_for_incremental_action_attribution': not reasons,
		'insufficiency_reasons': reasons,
		'interpretation': (
			'This is behaviour-policy coverage only. Even adequate coverage supports '
			'incremental predictive information, not causal controllability.'
		),
	}


@dataclass(frozen=True)
class SequenceRows:
	initial_u_history: np.ndarray
	action_window: np.ndarray
	target_future_u: np.ndarray
	episode_ids: np.ndarray
	start_indices: np.ndarray

	def __len__(self) -> int:
		return int(len(self.episode_ids))


def sequence_rows(
	clr_latents: Mapping[int, np.ndarray], actions: Mapping[int, np.ndarray],
	episode_ids: Sequence[int],
) -> SequenceRows:
	"""Build past-only h=5 rows without crossing episode boundaries."""
	_require(episode_ids, 'Episode selection is empty.')
	histories, windows, futures, ids, starts_all = [], [], [], [], []
	for episode_id in episode_ids:
		u = np.asarray(clr_latents[int(episode_id)], dtype=np.float32)
		action = np.asarray(actions[int(episode_id)], dtype=np.float32)
		_require(u.ndim == 2 and action.ndim == 2 and len(u) == len(action) + 1,
			f'Episode {episode_id} violates observation T+1/action T alignment.')
		# t has two observed predecessors and five future actions/targets.
		starts = np.arange(HISTORY_LENGTH - 1, len(action) - MAX_HORIZON + 1)
		_require(len(starts) > 0, f'Episode {episode_id} has no valid rollout starts.')
		for start in starts:
			histories.append(u[start - 2:start + 1])
			windows.append(action[start - 2:start + MAX_HORIZON])
			futures.append(u[start + 1:start + MAX_HORIZON + 1])
			ids.append(int(episode_id))
			starts_all.append(int(start))
	return SequenceRows(
		initial_u_history=np.asarray(histories, dtype=np.float32),
		action_window=np.asarray(windows, dtype=np.float32),
		target_future_u=np.asarray(futures, dtype=np.float32),
		episode_ids=np.asarray(ids, dtype=np.int64),
		start_indices=np.asarray(starts_all, dtype=np.int64),
	)


def shuffled_future_rows(rows: SequenceRows, shuffled_action: np.ndarray) -> SequenceRows:
	"""Keep real past actions per start; replace only current/future actions."""
	shuffled_action = np.asarray(shuffled_action, dtype=np.float32)
	_require(shuffled_action.ndim == 2, 'Shuffled action trace must be rank two.')
	windows = rows.action_window.copy()
	for index, start in enumerate(rows.start_indices):
		replacement = shuffled_action[int(start):int(start) + MAX_HORIZON]
		_require(len(replacement) == MAX_HORIZON,
			'Shuffled future action window is truncated.')
		windows[index, HISTORY_LENGTH - 1:] = replacement
	_require(np.array_equal(
		windows[:, :HISTORY_LENGTH - 1],
		rows.action_window[:, :HISTORY_LENGTH - 1],
	), 'Action intervention modified observed past actions.')
	return SequenceRows(
		initial_u_history=rows.initial_u_history,
		action_window=windows,
		target_future_u=rows.target_future_u,
		episode_ids=rows.episode_ids,
		start_indices=rows.start_indices,
	)


def mlp_parameter_count(input_dim: int, hidden_dim: int, output_dim: int) -> int:
	"""Two Linear+LayerNorm blocks followed by a Linear output."""
	_require(input_dim > 0 and hidden_dim > 0 and output_dim > 0,
		'MLP widths must be positive.')
	return int(
		(input_dim * hidden_dim + hidden_dim)
		+ (2 * hidden_dim)
		+ (hidden_dim * hidden_dim + hidden_dim)
		+ (2 * hidden_dim)
		+ (hidden_dim * output_dim + output_dim)
	)


def architecture_input_dim(
	architecture: str, latent_dim: int, action_dim: int, action_mode: str,
) -> int:
	_require(architecture in ARCHITECTURES, f'Unknown architecture {architecture}.')
	_require(action_mode in ACTION_MODES, f'Unknown action mode {action_mode}.')
	if architecture == 'normalized_delta_markov':
		return int(latent_dim + (action_dim if action_mode == 'action_aware' else 0))
	# The actionless control has no action input at any lag.
	action_slots = 3 if action_mode == 'action_aware' else 0
	return int(3 * latent_dim + action_slots * action_dim)


def parameter_matched_widths(latent_dim: int, action_dim: int) -> dict[str, dict[str, int]]:
	"""Search four widths so architecture/action controls share one budget."""
	reference_input = architecture_input_dim(
		'normalized_delta_markov', latent_dim, action_dim, 'action_aware',
	)
	target = mlp_parameter_count(reference_input, latent_dim, latent_dim)
	upper = max(latent_dim * 2, 16)
	result = {}
	for architecture in ARCHITECTURES:
		result[architecture] = {}
		for action_mode in ACTION_MODES:
			input_dim = architecture_input_dim(
				architecture, latent_dim, action_dim, action_mode,
			)
			hidden = min(
				range(1, upper + 1),
				key=lambda width: abs(
					mlp_parameter_count(input_dim, width, latent_dim) - target
				),
			)
			count = mlp_parameter_count(input_dim, hidden, latent_dim)
			_require(
				abs(count / target - 1.0)
				<= PREREGISTERED_GATES['parameter_ratio_tolerance'],
				'Could not parameter-match all four model/control inputs within 5%.',
			)
			result[architecture][action_mode] = int(hidden)
	return result


def normalized_action_features(
	action_history, action_mean, action_scale, *, architecture: str,
	action_mode: str,
):
	"""Normalize model action features; supports NumPy and lazy Torch tensors."""
	_require(architecture in ARCHITECTURES, f'Unknown architecture {architecture}.')
	_require(action_mode in ACTION_MODES, f'Unknown action mode {action_mode}.')
	value = (action_history - action_mean) / action_scale
	if action_mode == 'actionless':
		return value[:, :0].reshape(len(value), 0)
	if architecture == 'normalized_delta_markov':
		return value[:, -1]
	try:
		import torch
		if isinstance(value, torch.Tensor):
			return value.flip(1).flatten(1)
	except ImportError:
		pass
	return np.asarray(value)[:, ::-1].reshape(len(value), -1)


def _model_factory(
	architecture: str, action_mode: str, normalization: DeltaNormalization,
	*, hidden_dim: int,
):
	import torch
	import torch.nn as nn
	_require(architecture in ARCHITECTURES, f'Unknown architecture {architecture}.')
	_require(action_mode in ACTION_MODES, f'Unknown action mode {action_mode}.')
	latent_dim = len(normalization.u_mean)
	action_dim = len(normalization.action_mean)
	input_dim = architecture_input_dim(
		architecture, latent_dim, action_dim, action_mode,
	)

	class _NormalizedDelta(nn.Module):
		def __init__(self):
			super().__init__()
			self.body = nn.Sequential(
				nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
				nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
				nn.Linear(hidden_dim, latent_dim),
			)
			# A neutral initial transition emits the train-domain mean innovation.
			nn.init.zeros_(self.body[-1].weight)
			nn.init.zeros_(self.body[-1].bias)
			self.register_buffer('u_mean', torch.as_tensor(normalization.u_mean))
			self.register_buffer('u_scale', torch.as_tensor(normalization.u_scale))
			self.register_buffer('action_mean', torch.as_tensor(normalization.action_mean))
			self.register_buffer('action_scale', torch.as_tensor(normalization.action_scale))
			self.register_buffer('delta_mean', torch.as_tensor(normalization.delta_mean))
			self.register_buffer('delta_scale', torch.as_tensor(normalization.delta_scale))
			self.architecture = architecture
			self.action_mode = action_mode

		def features(self, u_history, action_history):
			current = (u_history[:, -1] - self.u_mean) / self.u_scale
			if self.architecture == 'normalized_delta_markov':
				return torch.cat([
					current, normalized_action_features(
						action_history, self.action_mean, self.action_scale,
						architecture=self.architecture,
						action_mode=self.action_mode,
					),
				], dim=-1)
			recent = (
				u_history[:, -1] - u_history[:, -2] - self.delta_mean
			) / self.delta_scale
			previous = (
				u_history[:, -2] - u_history[:, -3] - self.delta_mean
			) / self.delta_scale
			action_features = normalized_action_features(
				action_history, self.action_mean, self.action_scale,
				architecture=self.architecture,
				action_mode=self.action_mode,
			)
			return torch.cat([current, recent, previous, action_features], dim=-1)

		def step(self, u_history, action_history):
			standardized = self.body(self.features(u_history, action_history))
			raw_delta = standardized * self.delta_scale + self.delta_mean
			groups = raw_delta.reshape(len(raw_delta), -1, normalization.simnorm_dim)
			projected = groups - groups.mean(dim=-1, keepdim=True)
			correction = groups - projected
			next_u = u_history[:, -1] + projected.reshape_as(raw_delta)
			# Re-centering is numerically defensive; analytically it is a no-op.
			next_groups = next_u.reshape(len(next_u), -1, normalization.simnorm_dim)
			next_groups = next_groups - next_groups.mean(dim=-1, keepdim=True)
			return next_groups.reshape_as(next_u), correction

	return _NormalizedDelta()


def _torch_rollout(model, initial_history, action_window, max_horizon: int):
	"""Open-loop rollout: predicted states, never future observed states, update history."""
	history = initial_history
	outputs, corrections = {}, []
	for offset in range(max_horizon):
		action_history = action_window[:, offset:offset + HISTORY_LENGTH]
		next_u, correction = model.step(history, action_history)
		history = np_or_torch_cat_history(history, next_u)
		corrections.append(correction)
		if offset + 1 in HORIZONS:
			outputs[offset + 1] = next_u
	return outputs, corrections


def np_or_torch_cat_history(history, next_u):
	"""Tiny indirection keeps the no-teacher-forcing helper contract-testable."""
	try:
		import torch
		if isinstance(history, torch.Tensor):
			return torch.cat([history[:, 1:], next_u[:, None]], dim=1)
	except ImportError:
		pass
	return np.concatenate([history[:, 1:], np.asarray(next_u)[:, None]], axis=1)


def _validation_objective(
	model, rows: SequenceRows, normalization: DeltaNormalization,
	*, device, batch_size: int,
) -> float:
	import torch
	model.eval()
	total, samples = 0.0, 0
	with torch.no_grad():
		for start in range(0, len(rows), batch_size):
			stop = min(start + batch_size, len(rows))
			history = torch.as_tensor(rows.initial_u_history[start:stop], device=device)
			actions = torch.as_tensor(rows.action_window[start:stop], device=device)
			target = torch.as_tensor(rows.target_future_u[start:stop], device=device)
			outputs, _ = _torch_rollout(model, history, actions, MAX_HORIZON)
			losses = []
			for horizon in HORIZONS:
				scale = torch.as_tensor(normalization.horizon_scale[horizon], device=device)
				losses.append(((outputs[horizon] - target[:, horizon - 1]) / scale).square().mean())
			loss = torch.stack(losses).mean()
			total += float(loss.item()) * (stop - start)
			samples += stop - start
	return total / samples


def _trailing_relative_improvement(history: Sequence[Mapping], window: int) -> float:
	_require(len(history) >= 2 and window >= 2, 'Convergence history/window is invalid.')
	rows = history[-min(window, len(history)):]
	start = float(rows[0]['validation_objective'])
	end = min(float(row['validation_objective']) for row in rows[-3:])
	return float(max(0.0, (start - end) / max(abs(start), 1e-12)))


@dataclass
class FittedDeltaTransition:
	architecture: str
	action_mode: str
	model: object
	normalization: DeltaNormalization
	device: object
	metadata: Mapping

	def rollout(self, initial_u_history: np.ndarray, action_window: np.ndarray) -> dict:
		import torch
		initial_u_history = np.asarray(initial_u_history, dtype=np.float32)
		action_window = np.asarray(action_window, dtype=np.float32)
		_require(initial_u_history.ndim == 3 and initial_u_history.shape[1] == HISTORY_LENGTH,
			'Initial past-only history has invalid shape.')
		_require(action_window.ndim == 3 and action_window.shape[1] == HISTORY_LENGTH - 1 + MAX_HORIZON,
			'Action rollout window has invalid shape.')
		with torch.no_grad():
			self.model.eval()
			history = torch.as_tensor(initial_u_history, device=self.device)
			actions = torch.as_tensor(action_window, device=self.device)
			outputs, corrections = _torch_rollout(
				self.model, history, actions, MAX_HORIZON,
			)
			result = {
				'predicted_u': {
					int(horizon): outputs[horizon].detach().cpu().numpy().astype(np.float32)
					for horizon in HORIZONS
				},
				'projection': {
					'correction_rms': float(torch.sqrt(torch.mean(torch.square(
						torch.cat([value.flatten() for value in corrections])
					))).item()),
					'pre_projection_max_abs_group_sum': float(max(
						value.shape[-1] * value.abs().max().item() for value in corrections
					)),
				},
			}
		for horizon, value in result['predicted_u'].items():
			z = clr_to_simnorm(value, self.normalization.simnorm_dim)
			result.setdefault('predicted_z', {})[horizon] = z
		audits = [
			simplex_audit(value, self.normalization.simnorm_dim)
			for value in result['predicted_z'].values()
		]
		result['projection'].update({
			'predicted_min_probability': min(row['min_probability'] for row in audits),
			'predicted_max_group_sum_error': max(
				row['max_group_sum_error'] for row in audits
			),
			'clr_post_projection_max_abs_group_mean': max(
				float(np.abs(_reshape_groups(
					value, self.normalization.simnorm_dim,
				).mean(axis=-1)).max())
				for value in result['predicted_u'].values()
			),
		})
		return result


def fit_transition(
	architecture: str, action_mode: str,
	clr_latents: Mapping[int, np.ndarray], actions: Mapping[int, np.ndarray],
	train_ids: Sequence[int], validation_ids: Sequence[int], *,
	normalization: DeltaNormalization, device, seed: int, batch_size: int,
	max_epochs: int, min_epochs: int, patience: int, convergence_window: int,
	convergence_threshold: float, learning_rate: float, weight_decay: float,
	hidden_dim: int,
) -> FittedDeltaTransition:
	"""Fit by open-loop h1/3/5 CLR-innovation NMSE, with validation selection."""
	import torch
	_require(max_epochs >= min_epochs >= convergence_window >= 2,
		'Invalid convergence schedule.')
	_require(patience >= 1 and batch_size >= 1 and learning_rate > 0.0,
		'Invalid optimization schedule.')
	train = sequence_rows(clr_latents, actions, train_ids)
	validation = sequence_rows(clr_latents, actions, validation_ids)
	torch.manual_seed(int(seed))
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(int(seed))
	try:
		torch.use_deterministic_algorithms(True, warn_only=True)
	except TypeError:
		torch.use_deterministic_algorithms(True)
	model = _model_factory(
		architecture, action_mode, normalization, hidden_dim=hidden_dim,
	).to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay),
	)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode='min', factor=0.5, patience=max(3, patience // 3),
		min_lr=learning_rate / 32.0,
	)
	rng = np.random.default_rng(int(seed))
	best_loss, best_epoch, best_state = math.inf, 0, None
	stale, history = 0, []
	for epoch in range(1, max_epochs + 1):
		model.train()
		permutation = rng.permutation(len(train))
		total = 0.0
		for start in range(0, len(permutation), batch_size):
			index = permutation[start:start + batch_size]
			history_t = torch.as_tensor(train.initial_u_history[index], device=device)
			action_t = torch.as_tensor(train.action_window[index], device=device)
			target_t = torch.as_tensor(train.target_future_u[index], device=device)
			optimizer.zero_grad(set_to_none=True)
			outputs, _ = _torch_rollout(model, history_t, action_t, MAX_HORIZON)
			losses = []
			for horizon in HORIZONS:
				scale = torch.as_tensor(normalization.horizon_scale[horizon], device=device)
				losses.append(((outputs[horizon] - target_t[:, horizon - 1]) / scale).square().mean())
			loss = torch.stack(losses).mean()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
			optimizer.step()
			total += float(loss.detach().item()) * len(index)
		validation_loss = _validation_objective(
			model, validation, normalization, device=device, batch_size=batch_size,
		)
		scheduler.step(validation_loss)
		row = {
			'epoch': int(epoch), 'train_objective': total / len(train),
			'validation_objective': float(validation_loss),
			'learning_rate': float(optimizer.param_groups[0]['lr']),
		}
		history.append(row)
		significant_improvement = validation_loss < best_loss * (
			1.0 - convergence_threshold
		)
		if validation_loss < best_loss:
			best_loss = validation_loss
			best_epoch = epoch
			best_state = {
				name: value.detach().cpu().clone()
				for name, value in model.state_dict().items()
			}
		if significant_improvement:
			stale = 0
		else:
			stale += 1
		trailing = (
			_trailing_relative_improvement(history, convergence_window)
			if len(history) >= convergence_window else math.inf
		)
		if epoch >= min_epochs and stale >= patience and trailing <= convergence_threshold:
			break
	_require(best_state is not None and np.isfinite(best_loss),
		'Normalized-delta fit never produced a finite validation model.')
	model.load_state_dict(best_state)
	model.eval()
	trailing = _trailing_relative_improvement(history, convergence_window)
	converged = bool(
		len(history) >= min_epochs and trailing <= convergence_threshold
	)
	parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
	metadata = {
		'architecture': architecture,
		'action_mode': action_mode,
		'fit_condition_statistics': 'same_domain_train_episodes_only',
		'fit_objective': 'equal_weight_h1_h3_h5_open_loop_clr_innovation_nmse',
		'teacher_forcing_after_initial_history': False,
		'parameter_count': parameter_count,
		'hidden_dim': int(hidden_dim),
		'latent_dim': int(len(normalization.u_mean)),
		'action_dim': int(len(normalization.action_mean)),
		'train_episode_ids': [int(value) for value in train_ids],
		'validation_episode_ids': [int(value) for value in validation_ids],
		'test_episodes_seen_during_fit_or_selection': False,
		'train_rows': len(train), 'validation_rows': len(validation),
		'best_epoch': int(best_epoch), 'epochs_ran': int(len(history)),
		'best_validation_objective': float(best_loss),
		'converged': converged,
		'convergence_criterion': {
			'window_epochs': int(convergence_window),
			'max_trailing_relative_improvement': float(convergence_threshold),
			'observed_trailing_relative_improvement': float(trailing),
			'min_epochs': int(min_epochs), 'patience': int(patience),
		},
		'learning_curve': history,
		'normalization': normalization_summary(normalization),
	}
	return FittedDeltaTransition(
		architecture, action_mode, model, normalization, device, metadata,
	)


def _metric_arrays(
	predicted_u: np.ndarray, predicted_z: np.ndarray,
	start_u: np.ndarray, start_z: np.ndarray,
	target_u: np.ndarray, target_z: np.ndarray,
	horizon_scale: np.ndarray,
	simnorm_dim: int,
) -> dict[str, np.ndarray]:
	"""Per-start primary/secondary errors and displacement magnitudes."""
	predicted_u = np.asarray(predicted_u, dtype=np.float64)
	predicted_z = np.asarray(predicted_z, dtype=np.float64)
	start_u = np.asarray(start_u, dtype=np.float64)
	start_z = np.asarray(start_z, dtype=np.float64)
	target_u = np.asarray(target_u, dtype=np.float64)
	target_z = np.asarray(target_z, dtype=np.float64)
	scale = np.asarray(horizon_scale, dtype=np.float64)
	_require(
		predicted_u.shape == start_u.shape == target_u.shape
		and predicted_z.shape == start_z.shape == target_z.shape,
		'Metric arrays are misaligned.',
	)
	normalized_error = (predicted_u - target_u) / scale
	clr_error_groups = _reshape_groups(predicted_u - target_u, simnorm_dim)
	predicted_delta = predicted_u - start_u
	true_delta = target_u - start_u
	return {
		'aitchison_mse': np.mean(
			np.mean(np.square(clr_error_groups), axis=-1), axis=-1,
		),
		'fixed_clean_horizon_nmse': np.mean(np.square(normalized_error), axis=1),
		'raw_latent_mse': np.mean(np.square(predicted_z - target_z), axis=1),
		'predicted_delta_rms': np.sqrt(np.mean(np.square(predicted_delta), axis=1)),
		'predicted_normalized_delta_rms': np.sqrt(np.mean(
			np.square(predicted_delta / scale), axis=1,
		)),
		'true_delta_rms': np.sqrt(np.mean(np.square(true_delta), axis=1)),
		'true_normalized_delta_rms': np.sqrt(np.mean(
			np.square(true_delta / scale), axis=1,
		)),
	}


def _checkpoint_rollout(
	predictor, z: np.ndarray, action: np.ndarray, shuffled_action: np.ndarray,
	starts: np.ndarray, action_mean: np.ndarray, *, simnorm_dim: int,
) -> dict[str, dict]:
	"""Roll out the frozen checkpoint under real, shuffled, and mean actions."""
	variants = {
		'real_action': np.asarray(action, dtype=np.float32),
		'shuffled_action': np.asarray(shuffled_action, dtype=np.float32),
		'actionless': np.broadcast_to(
			np.asarray(action_mean, dtype=np.float32), action.shape,
		).copy(),
	}
	result = {}
	for name, action_source in variants.items():
		current = np.asarray(z[starts], dtype=np.float32).copy()
		predicted = {}
		for offset in range(MAX_HORIZON):
			current = np.asarray(
				predictor.predict(current, action_source[starts + offset]),
				dtype=np.float32,
			)
			_require(current.shape == z[starts].shape,
				'Checkpoint transition changed latent shape.')
			if offset + 1 in HORIZONS:
				predicted[offset + 1] = current.copy()
		result[name] = {
			'predicted_z': predicted,
			'predicted_u': {
				horizon: clr_transform(value, simnorm_dim)
				for horizon, value in predicted.items()
			},
			'projection': {
				'source': 'checkpoint_simnorm_output',
				'max_group_sum_error': max(
					simplex_audit(value, simnorm_dim)['max_group_sum_error']
					for value in predicted.values()
				),
			},
		}
	return result


def _learned_rollouts(
	aware: FittedDeltaTransition, actionless: FittedDeltaTransition,
	rows: SequenceRows, shuffled_rows: SequenceRows,
) -> dict[str, dict]:
	_require(np.array_equal(rows.start_indices, shuffled_rows.start_indices),
		'Real/shuffled rollout starts differ.')
	return {
		'real_action': aware.rollout(rows.initial_u_history, rows.action_window),
		'shuffled_action': aware.rollout(
			rows.initial_u_history, shuffled_rows.action_window,
		),
		'actionless': actionless.rollout(rows.initial_u_history, rows.action_window),
	}


def _pooled_variant(rows: Sequence[Mapping], variant: str) -> dict:
	keys = (
		'aitchison_mse', 'fixed_clean_horizon_nmse', 'raw_latent_mse',
		'predicted_delta_rms', 'predicted_normalized_delta_rms',
		'true_delta_rms', 'true_normalized_delta_rms',
	)
	return {
		key: float(np.average(
			[row['variants'][variant][key] for row in rows],
			weights=[row['samples'] for row in rows],
		))
		for key in keys
	}


def _episode_metric(rows: Sequence[Mapping], method: str, variant: str) -> np.ndarray:
	if method == 'persistence':
		return np.asarray([row['aitchison_mse'] for row in rows], dtype=np.float64)
	return np.asarray([
		row['variants'][variant]['aitchison_mse'] for row in rows
	], dtype=np.float64)


def evaluate_condition(
	dataset: ladder.FrozenDataset, latents: Mapping[int, np.ndarray],
	clr_latents: Mapping[int, np.ndarray], fits: Mapping[str, Mapping[str, FittedDeltaTransition]],
	checkpoint, fit_normalization: DeltaNormalization,
	scoring_normalization: DeltaNormalization, *,
	bootstrap_seed: int, bootstrap_resamples: int,
) -> dict:
	"""Evaluate a complete held-out condition using paired whole-episode rows."""
	_require(set(fits) == set(ARCHITECTURES), 'Learned architecture set is incomplete.')
	for architecture in ARCHITECTURES:
		_require(set(fits[architecture]) == set(ACTION_MODES),
			f'{architecture} lacks an action-aware/actionless fit.')
	test_episodes = dataset.split('test')
	_require(len(test_episodes) == SPLIT_COUNTS['test'],
		'Evaluation requires exactly four held-out episodes.')
	condition = _condition(dataset)
	per_horizon = {
		horizon: {method: [] for method in METHODS} for horizon in HORIZONS
	}
	projection_rows = {
		method: {variant: [] for variant in ('real_action', 'shuffled_action', 'actionless')}
		for method in METHODS if method != 'persistence'
	}
	action_difference_rows = []
	for episode in test_episodes:
		episode_id = int(episode.episode_id)
		z = np.asarray(latents[episode_id], dtype=np.float32)
		u = np.asarray(clr_latents[episode_id], dtype=np.float32)
		action = np.asarray(episode.arrays['action'], dtype=np.float32)
		rng = np.random.default_rng(_stable_seed(
			bootstrap_seed, 'action_shuffle', condition, episode_id,
		))
		shuffled_action = action[rng.permutation(len(action))]
		rows = sequence_rows({episode_id: u}, {episode_id: action}, [episode_id])
		shuffled_rows = shuffled_future_rows(rows, shuffled_action)
		starts = rows.start_indices
		_require(np.array_equal(starts, shuffled_rows.start_indices),
			'Real/shuffled start indices differ.')
		action_difference_rows.append(float(np.mean(np.square(
			rows.action_window[:, HISTORY_LENGTH - 1:]
			- shuffled_rows.action_window[:, HISTORY_LENGTH - 1:]
		))))
		checkpoint_rollouts = _checkpoint_rollout(
			checkpoint, z, action, shuffled_action, starts,
			fit_normalization.action_mean, simnorm_dim=fit_normalization.simnorm_dim,
		)
		learned = {
			architecture: _learned_rollouts(
				fits[architecture]['action_aware'], fits[architecture]['actionless'],
				rows, shuffled_rows,
			)
			for architecture in ARCHITECTURES
		}
		all_rollouts = {'checkpoint_transition': checkpoint_rollouts, **learned}
		for method, variants in all_rollouts.items():
			for variant, rollout in variants.items():
				projection_rows[method][variant].append(dict(rollout['projection']))
		for horizon in HORIZONS:
			target_index = starts + horizon
			start_u, start_z = u[starts], z[starts]
			target_u, target_z = u[target_index], z[target_index]
			persistence = _metric_arrays(
				start_u, start_z, start_u, start_z, target_u, target_z,
				scoring_normalization.horizon_scale[horizon],
				scoring_normalization.simnorm_dim,
			)
			per_horizon[horizon]['persistence'].append({
				'episode_id': episode_id, 'samples': int(len(starts)),
				**{key: float(value.mean()) for key, value in persistence.items()},
			})
			for method, variants in all_rollouts.items():
				variant_rows = {}
				for variant, rollout in variants.items():
					metrics = _metric_arrays(
						rollout['predicted_u'][horizon],
						rollout['predicted_z'][horizon],
						start_u, start_z, target_u, target_z,
						scoring_normalization.horizon_scale[horizon],
						scoring_normalization.simnorm_dim,
					)
					variant_rows[variant] = {
						key: float(value.mean()) for key, value in metrics.items()
					}
				per_horizon[horizon][method].append({
					'episode_id': episode_id, 'samples': int(len(starts)),
					'variants': variant_rows,
				})

	horizon_payload = {}
	for horizon in HORIZONS:
		rows_by_method = per_horizon[horizon]
		persistence_episode = _episode_metric(
			rows_by_method['persistence'], 'persistence', 'real_action',
		)
		methods = {
			'persistence': {
				'pooled': {
					key: float(np.average(
						[row[key] for row in rows_by_method['persistence']],
						weights=[row['samples'] for row in rows_by_method['persistence']],
					))
					for key in (
						'aitchison_mse', 'fixed_clean_horizon_nmse', 'raw_latent_mse',
						'predicted_delta_rms', 'predicted_normalized_delta_rms',
						'true_delta_rms', 'true_normalized_delta_rms',
					)
				},
				'by_episode': rows_by_method['persistence'],
				'episode_mean_primary_bootstrap': raw_refit.bootstrap_episode_mean(
					persistence_episode,
					seed=_stable_seed(bootstrap_seed, condition, horizon, 'persistence'),
					resamples=bootstrap_resamples,
				),
			}
		}
		for method in METHODS:
			if method == 'persistence':
				continue
			rows = rows_by_method[method]
			variants = {}
			for variant in ('real_action', 'shuffled_action', 'actionless'):
				episode_values = _episode_metric(rows, method, variant)
				variants[variant] = {
					'pooled': _pooled_variant(rows, variant),
					'episode_mean_primary_bootstrap': raw_refit.bootstrap_episode_mean(
						episode_values,
						seed=_stable_seed(
							bootstrap_seed, condition, horizon, method, variant,
						),
						resamples=bootstrap_resamples,
					),
				}
			real = _episode_metric(rows, method, 'real_action')
			shuffled = _episode_metric(rows, method, 'shuffled_action')
			actionless = _episode_metric(rows, method, 'actionless')
			methods[method] = {
				'variants': variants,
				'by_episode': rows,
				'episode_wins_vs_persistence': int(np.sum(real < persistence_episode)),
				'paired_real_vs_persistence': raw_refit.paired_episode_bootstrap(
					real, persistence_episode,
					seed=_stable_seed(bootstrap_seed, condition, horizon, method, 'persistence'),
					resamples=bootstrap_resamples,
				),
				'paired_real_vs_shuffled_action': raw_refit.paired_episode_bootstrap(
					real, shuffled,
					seed=_stable_seed(bootstrap_seed, condition, horizon, method, 'shuffle'),
					resamples=bootstrap_resamples,
				),
				'paired_real_vs_actionless': raw_refit.paired_episode_bootstrap(
					real, actionless,
					seed=_stable_seed(bootstrap_seed, condition, horizon, method, 'actionless'),
					resamples=bootstrap_resamples,
				),
			}
		markov = _episode_metric(
			rows_by_method['normalized_delta_markov'],
			'normalized_delta_markov', 'real_action',
		)
		history = _episode_metric(
			rows_by_method['normalized_delta_history'],
			'normalized_delta_history', 'real_action',
		)
		horizon_payload[str(horizon)] = {
			'methods': methods,
			'paired_history_vs_markov': raw_refit.paired_episode_bootstrap(
				history, markov,
				seed=_stable_seed(bootstrap_seed, condition, horizon, 'history_vs_markov'),
				resamples=bootstrap_resamples,
			),
		}
	return {
		'condition': condition,
		'test_episode_ids': [int(episode.episode_id) for episode in test_episodes],
		'evaluation_start_rule': 't>=2_and_t+5_within_same_episode',
		'bootstrap': {
			'unit': 'paired_whole_episode', 'resamples': int(bootstrap_resamples),
			'base_seed': int(bootstrap_seed),
			'warning': 'Four held-out episodes give exploratory, coarse intervals.',
		},
		'action_shuffle_input_mse_episode_mean': float(np.mean(action_difference_rows)),
		'scoring_normalization': 'fixed_clean_train_group_scalar_horizon_scale',
		'projection_diagnostics_by_method': projection_rows,
		'horizons': horizon_payload,
	}


def _comparison_pass(comparison: Mapping, minimum_gain: float) -> bool:
	relative = comparison['relative_improvement']
	return bool(
		float(relative['estimate']) >= minimum_gain
		and float(relative['ci95'][0]) > 0.0
	)


def score_preregistered_gates(
	evaluation: Mapping, fits: Mapping[str, Mapping[str, Mapping[str, FittedDeltaTransition]]],
	action_coverage: Mapping,
) -> dict:
	"""Apply fixed gates; only same-domain cells drive scientific attribution."""
	matrix = {}
	for fit_condition in FIT_CONDITIONS:
		matrix[fit_condition] = {}
		fit_converged = {
			architecture: all(
				bool(fits[fit_condition][architecture][mode].metadata['converged'])
				for mode in ACTION_MODES
			)
			for architecture in ARCHITECTURES
		}
		for test_condition in FIT_CONDITIONS:
			cell = evaluation[fit_condition][test_condition]
			methods = {}
			for method in ('checkpoint_transition',) + ARCHITECTURES:
				required = {}
				for horizon in PREREGISTERED_GATES['required_horizons']:
					row = cell['horizons'][str(horizon)]['methods'][method]
					required[str(horizon)] = {
						'gain_vs_persistence': _comparison_pass(
							row['paired_real_vs_persistence'],
							PREREGISTERED_GATES['min_relative_gain_vs_persistence'],
						),
						'gain_vs_shuffled_action': _comparison_pass(
							row['paired_real_vs_shuffled_action'],
							PREREGISTERED_GATES['min_relative_gain_vs_shuffled_action'],
						),
						'gain_vs_actionless': _comparison_pass(
							row['paired_real_vs_actionless'],
							PREREGISTERED_GATES['min_relative_gain_vs_actionless'],
						),
						'episode_wins': bool(
							row['episode_wins_vs_persistence']
							>= PREREGISTERED_GATES['min_episode_wins_out_of_4']
						),
					}
				h5 = cell['horizons']['5']['methods'][method]
				real_h5 = h5['variants']['real_action']['pooled']['aitchison_mse']
				persistence_h5 = cell['horizons']['5']['methods']['persistence'][
					'pooled'
				]['aitchison_mse']
				converged = True if method == 'checkpoint_transition' else fit_converged[method]
				methods[method] = {
					'fit_converged': bool(converged),
					'required_horizon_checks': required,
					'h5_no_divergence': bool(
						real_h5 / max(persistence_h5, 1e-12)
						<= PREREGISTERED_GATES['max_h5_real_over_persistence']
					),
				}
				methods[method]['prediction_pass'] = bool(
					methods[method]['fit_converged']
					and methods[method]['h5_no_divergence']
					and all(
						checks['gain_vs_persistence'] and checks['episode_wins']
						for checks in required.values()
					)
				)
				methods[method]['incremental_action_information_pass'] = bool(
					all(
						checks['gain_vs_shuffled_action'] and checks['gain_vs_actionless']
						for checks in required.values()
					)
				)
				methods[method]['action_predictive_all_pass'] = bool(
					methods[method]['prediction_pass']
					and methods[method]['incremental_action_information_pass']
				)
			history_checks = {}
			for horizon in PREREGISTERED_GATES['required_horizons']:
				comparison = cell['horizons'][str(horizon)]['paired_history_vs_markov']
				history_checks[str(horizon)] = _comparison_pass(
					comparison,
					PREREGISTERED_GATES['min_history_relative_gain_vs_markov'],
				)
			matrix[fit_condition][test_condition] = {
				'methods': methods,
				'history_beats_markov_required_horizons': history_checks,
				'history_beats_markov_all_pass': bool(all(history_checks.values())),
			}

	attribution = {}
	for condition in FIT_CONDITIONS:
		cell = matrix[condition][condition]
		markov_prediction = cell['methods']['normalized_delta_markov']['prediction_pass']
		history_prediction = cell['methods']['normalized_delta_history']['prediction_pass']
		markov_action = cell['methods']['normalized_delta_markov'][
			'action_predictive_all_pass'
		]
		history_action = cell['methods']['normalized_delta_history'][
			'action_predictive_all_pass'
		]
		history_margin = cell['history_beats_markov_all_pass']
		converged = all(
			cell['methods'][architecture]['fit_converged']
			for architecture in ARCHITECTURES
		)
		coverage_sufficient = bool(
			action_coverage[condition]['train'][
				'coverage_sufficient_for_incremental_action_attribution'
			]
			and action_coverage[condition]['test'][
				'coverage_sufficient_for_incremental_action_attribution'
			]
		)
		if not converged:
			label = 'inconclusive_nonconverged'
		elif not coverage_sufficient and history_prediction and not markov_prediction and history_margin:
			label = 'short_history_prediction_gain_action_attribution_inconclusive_coverage'
		elif not coverage_sufficient and markov_prediction:
			label = 'markov_prediction_recoverable_action_attribution_inconclusive_coverage'
		elif not coverage_sufficient:
			label = 'tested_prediction_gate_failed_action_attribution_inconclusive_coverage'
		elif history_action and not markov_action and history_margin:
			label = 'short_history_adds_incremental_action_predictive_information'
		elif markov_action:
			label = 'markov_action_predictive_dynamics_recoverable'
		elif markov_prediction or history_prediction:
			label = 'prediction_recoverable_without_incremental_action_information'
		else:
			label = 'tested_models_fail_preregistered_prediction_gate'
		attribution[condition] = {
			'label': label,
			'markov_prediction_pass': markov_prediction,
			'history_prediction_pass': history_prediction,
			'markov_incremental_action_pass': markov_action,
			'history_incremental_action_pass': history_action,
			'history_margin_pass': history_margin,
			'all_required_fits_converged': converged,
			'action_coverage_sufficient': coverage_sufficient,
		}
	return {
		'preregistered_thresholds': dict(PREREGISTERED_GATES),
		'matrix': matrix,
		'in_domain_attribution': attribution,
		'interpretation_guard': (
			'This observational behaviour-data diagnostic measures incremental action '
			'predictive information, not causal controllability. A parameter-matched '
			'short-history prediction gain supports temporal aliasing in the tested '
			'current latent, but equal-parameter failure does not prove history is useless. '
			'Action coverage insufficiency makes action attribution inconclusive, and '
			'prediction-gate failure must not be called missing latent information.'
		),
	}


def load_previous_raw_refit(
	path: Path | None, *, task: str, clean_dataset_hash: str,
	hard_dataset_hash: str, checkpoint_hash: str,
) -> dict | None:
	"""Optionally bind and compact the prior raw-next-z residual diagnostic."""
	if path is None:
		return None
	path = path.resolve()
	_require(path.is_file(), f'Previous refit JSON not found: {path}')
	payload = json.loads(path.read_text(encoding='utf-8'))
	_require(payload.get('format') == raw_refit.FORMAT,
		'Previous refit JSON has the wrong format.')
	_require(payload.get('task') == task, 'Previous refit task differs.')
	source = payload.get('source', {})
	_require(source.get('clean_dataset_sha256') == clean_dataset_hash,
		'Previous refit clean dataset hash differs.')
	_require(source.get('hard_dataset_sha256') == hard_dataset_hash,
		'Previous refit hard dataset hash differs.')
	_require(source.get('checkpoint_sha256') == checkpoint_hash,
		'Previous refit checkpoint hash differs.')
	compact = {}
	for fit_condition in FIT_CONDITIONS:
		compact[fit_condition] = {}
		for test_condition in FIT_CONDITIONS:
			compact[fit_condition][test_condition] = {}
			for horizon in HORIZONS:
				methods = payload['evaluation'][fit_condition][test_condition][
					'horizons'
				][str(horizon)]['methods']
				compact[fit_condition][test_condition][str(horizon)] = {
					'persistence_raw_latent_mse': float(
						methods['persistence']['pooled_frame_mse']
					),
					'checkpoint_raw_latent_mse': float(
						methods['checkpoint_transition']['pooled_frames']['real_action_mse']
					),
					'raw_residual_raw_latent_mse': float(
						methods['residual_normalized_2hidden']['pooled_frames']['real_action_mse']
					),
					'raw_residual_shuffled_action_mse': float(
						methods['residual_normalized_2hidden']['pooled_frames'][
							'shuffled_action_mse'
						]
					),
				}
	return {
		'path': str(path), 'sha256': _sha256(path),
		'format': raw_refit.FORMAT,
		'note': 'Secondary raw-z reference only; it is not used for model selection or gates.',
		'metrics': compact,
	}


def validate_result(payload: Mapping) -> None:
	_require(payload.get('format') == FORMAT, 'Unexpected normalized-delta format.')
	_require(payload.get('status') == STATUS, 'Normalized-delta diagnostic is incomplete.')
	_require(payload.get('engineering_pass') is True, 'Engineering checks did not pass.')
	_require(payload.get('controller_training_authorized') is False,
		'Diagnostic must never authorize controller training.')
	_require(payload.get('policy_training_performed') is False,
		'Policy optimization is forbidden.')
	_require(payload.get('privileged_targets_used') is False,
		'Privileged targets are forbidden.')
	_require(payload.get('simulator_state_used') is False,
		'Simulator state is forbidden.')
	protocol = payload.get('protocol', {})
	_require(protocol.get('fit_conditions') == list(FIT_CONDITIONS),
		'Both independent fit domains are required.')
	_require(protocol.get('episode_split_counts') == SPLIT_COUNTS,
		'Protocol must bind the exact 12/4/4 whole-episode split.')
	_require(protocol.get('teacher_forcing_after_initial_history') is False,
		'Open-loop rollout cannot use future teacher forcing.')
	_require(protocol.get('primary_metric') == PREREGISTERED_GATES['primary_metric'],
		'Primary metric differs from preregistration.')
	_require(payload.get('gates', {}).get('preregistered_thresholds') == PREREGISTERED_GATES,
		'Preregistered gate thresholds changed after evaluation.')
	training = payload.get('training', {})
	_require(set(training) == set(FIT_CONDITIONS), 'Fit-domain training matrix is incomplete.')
	all_converged = True
	for fit_condition in FIT_CONDITIONS:
		_require(set(training[fit_condition]) == set(ARCHITECTURES),
			f'{fit_condition} architecture fits are incomplete.')
		counts = {}
		for architecture in ARCHITECTURES:
			_require(set(training[fit_condition][architecture]) == set(ACTION_MODES),
				f'{fit_condition}/{architecture} action controls are incomplete.')
			for mode in ACTION_MODES:
				metadata = training[fit_condition][architecture][mode]
				_require(len(metadata.get('train_episode_ids', [])) == SPLIT_COUNTS['train'],
					'Training split size is not 12 episodes.')
				_require(len(metadata.get('validation_episode_ids', [])) == SPLIT_COUNTS['validation'],
					'Validation split size is not 4 episodes.')
				_require(metadata.get('test_episodes_seen_during_fit_or_selection') is False,
					'Test episodes leaked into fitting/selection.')
				all_converged = all_converged and bool(metadata.get('converged'))
				counts[(architecture, mode)] = int(metadata['parameter_count'])
		reference = counts[('normalized_delta_markov', 'action_aware')]
		for name, count in counts.items():
			_require(abs(count / reference - 1.0)
				<= PREREGISTERED_GATES['parameter_ratio_tolerance'],
				f'Model/control {name} is not matched to the common active budget.')
	_require(payload.get('scientific_complete') == bool(all_converged),
		'Scientific-complete flag disagrees with convergence records.')
	coverage = payload.get('action_coverage_audit', {})
	_require(set(coverage) == set(FIT_CONDITIONS),
		'Action coverage condition audit is incomplete.')
	for condition in FIT_CONDITIONS:
		_require(set(coverage[condition]) == {'train', 'test'},
			f'{condition} action coverage train/test audit is incomplete.')
		for split in ('train', 'test'):
			row = coverage[condition][split]
			_require('coverage_sufficient_for_incremental_action_attribution' in row,
				'Action coverage conclusion is missing.')
			_require('not causal controllability' in row.get('interpretation', ''),
				'Action coverage interpretation overclaims causality.')
	_require(set(payload.get('clr_epsilon_sensitivity', {})) == set(FIT_CONDITIONS),
		'CLR epsilon sensitivity audit is incomplete.')
	evaluation = payload.get('evaluation', {})
	_require(set(evaluation) == set(FIT_CONDITIONS), 'Evaluation fit-domain matrix is incomplete.')
	for fit_condition in FIT_CONDITIONS:
		_require(set(evaluation[fit_condition]) == set(FIT_CONDITIONS),
			f'{fit_condition} test-domain matrix is incomplete.')
		for test_condition in FIT_CONDITIONS:
			row = evaluation[fit_condition][test_condition]
			_require(len(row.get('test_episode_ids', [])) == SPLIT_COUNTS['test'],
				'Evaluation does not contain exactly four test episodes.')
			_require(set(row.get('horizons', {})) == {str(value) for value in HORIZONS},
				'Evaluation horizons are incomplete.')
			for horizon in HORIZONS:
				methods = row['horizons'][str(horizon)].get('methods', {})
				_require(set(methods) == set(METHODS), 'Evaluation methods are incomplete.')
				for method in METHODS:
					_require(len(methods[method].get('by_episode', [])) == SPLIT_COUNTS['test'],
						'Evaluation method lacks four episode rows.')
	json.dumps(payload, allow_nan=False)


def evaluate(args) -> dict:
	clean = load_policy_dataset(args.clean_dataset)
	hard = load_policy_dataset(args.hard_dataset)
	runtime_config = Path(args.runtime_config).resolve()
	checkpoint_path = Path(args.checkpoint).resolve()
	pair_checks = raw_refit.validate_dataset_pair(
		clean, hard, runtime_config, checkpoint_path,
	)
	agent = ladder._load_agent(clean, runtime_config, checkpoint_path)
	for parameter in agent.model.parameters():
		parameter.requires_grad_(False)
	agent.eval()
	simnorm_dim = int(agent.cfg.get('simnorm_dim', 8))
	raw_latents = {
		'clean': ladder._encode_episodes(clean, agent, batch_size=args.encoder_batch_size),
		'hard': ladder._encode_episodes(hard, agent, batch_size=args.encoder_batch_size),
	}
	clr_latents, clr_audits = {}, {}
	for condition in FIT_CONDITIONS:
		clr_latents[condition], clr_audits[condition] = encode_clr_latents(
			raw_latents[condition], simnorm_dim,
		)
	datasets = {'clean': clean, 'hard': hard}
	actions = {condition: action_mapping(datasets[condition]) for condition in FIT_CONDITIONS}
	normalizations = {
		condition: fit_delta_normalization(
			clr_latents[condition], actions[condition],
			datasets[condition].splits['train'], simnorm_dim=simnorm_dim,
		)
		for condition in FIT_CONDITIONS
	}
	latent_dim = int(next(iter(raw_latents['clean'].values())).shape[1])
	widths = parameter_matched_widths(latent_dim, clean.action_dim)
	fits = {condition: {} for condition in FIT_CONDITIONS}
	for fit_condition in FIT_CONDITIONS:
		for architecture in ARCHITECTURES:
			fits[fit_condition][architecture] = {}
			for action_mode in ACTION_MODES:
				fits[fit_condition][architecture][action_mode] = fit_transition(
					architecture, action_mode,
					clr_latents[fit_condition], actions[fit_condition],
					datasets[fit_condition].splits['train'],
					datasets[fit_condition].splits['validation'],
					normalization=normalizations[fit_condition], device=agent.device,
					seed=_stable_seed(args.seed, fit_condition, architecture, action_mode),
					batch_size=args.fit_batch_size, max_epochs=args.max_epochs,
					min_epochs=args.min_epochs, patience=args.patience,
					convergence_window=args.convergence_window,
					convergence_threshold=args.convergence_threshold,
					learning_rate=args.learning_rate, weight_decay=args.weight_decay,
					hidden_dim=widths[architecture][action_mode],
				)
		reference_count = fits[fit_condition]['normalized_delta_markov'][
			'action_aware'
		].metadata['parameter_count']
		for architecture in ARCHITECTURES:
			for action_mode in ACTION_MODES:
				count = fits[fit_condition][architecture][action_mode].metadata[
					'parameter_count'
				]
				_require(abs(count / reference_count - 1.0)
					<= PREREGISTERED_GATES['parameter_ratio_tolerance'],
					'Runtime four-way parameter ratio exceeds 5%.')
	checkpoint = raw_refit.CheckpointTransition(agent)
	evaluation = {}
	for fit_condition in FIT_CONDITIONS:
		evaluation[fit_condition] = {}
		for test_condition in FIT_CONDITIONS:
			evaluation[fit_condition][test_condition] = evaluate_condition(
				datasets[test_condition], raw_latents[test_condition],
				clr_latents[test_condition], fits[fit_condition], checkpoint,
				normalizations[fit_condition], normalizations['clean'],
				bootstrap_seed=_stable_seed(
					args.bootstrap_seed, fit_condition, test_condition,
				),
				bootstrap_resamples=args.bootstrap_resamples,
			)
	action_coverage = {
		condition: {
			split: action_coverage_audit(
				datasets[condition], clr_latents[condition], normalizations[condition],
				datasets[condition].splits[split],
			)
			for split in ('train', 'test')
		}
		for condition in FIT_CONDITIONS
	}
	gates = score_preregistered_gates(evaluation, fits, action_coverage)
	all_converged = all(
		fits[condition][architecture][mode].metadata['converged']
		for condition in FIT_CONDITIONS
		for architecture in ARCHITECTURES
		for mode in ACTION_MODES
	)
	clean_hash = _sha256(clean.manifest_path)
	hard_hash = _sha256(hard.manifest_path)
	checkpoint_hash = _sha256(checkpoint_path)
	previous = load_previous_raw_refit(
		args.previous_refit_json, task=clean.task,
		clean_dataset_hash=clean_hash, hard_dataset_hash=hard_hash,
		checkpoint_hash=checkpoint_hash,
	)
	payload = {
		'format': FORMAT, 'status': STATUS,
		'engineering_pass': True, 'scientific_complete': bool(all_converged),
		'controller_training_authorized': False,
		'policy_training_performed': False,
		'privileged_targets_used': False,
		'simulator_state_used': False,
		'recommendation': 'review_normalized_delta_history_attribution_only',
		'task': clean.task,
		'source': {
			'clean_dataset': str(clean.manifest_path),
			'clean_dataset_sha256': clean_hash,
			'hard_dataset': str(hard.manifest_path),
			'hard_dataset_sha256': hard_hash,
			'runtime_config': str(runtime_config),
			'runtime_config_sha256': _sha256(runtime_config),
			'checkpoint': str(checkpoint_path),
			'checkpoint_sha256': checkpoint_hash,
			'checkpoint_step': pair_checks['checkpoint_step'],
			'exact_object_encoder_only': True,
			'pair_checks': pair_checks,
		},
		'protocol': {
			'fit_conditions': list(FIT_CONDITIONS),
			'episode_split_counts': dict(SPLIT_COUNTS),
			'fit_split': 'whole_same_domain_train_episodes',
			'selection_split': 'whole_same_domain_validation_episodes',
			'test_split': 'untouched_whole_test_episodes',
			'horizons': list(HORIZONS),
			'history_length_states': HISTORY_LENGTH,
			'teacher_forcing_after_initial_history': False,
			'architectures': list(ARCHITECTURES),
			'action_controls': [
				'action_aware_real', 'action_aware_within_episode_shuffled',
				'separately_fitted_parameter_matched_model_with_no_action_inputs',
			],
			'latent_transform': (
				'grouped SimNorm probability -> epsilon-clipped renormalization -> '
				'centered log ratio; predicted raw innovation is group-zero-sum '
				'projected; next latent is exact grouped softmax'
			),
			'fit_objective': 'equal_weight_h1_h3_h5_open_loop_clr_innovation_nmse',
			'primary_metric': PREREGISTERED_GATES['primary_metric'],
			'secondary_metrics': [
				'fixed_clean_train_group_scalar_horizon_nmse',
				'raw_simnorm_latent_mse',
			],
			'scoring_normalization': 'fixed_clean_train_for_every_2x2_cell',
			'normalization_source': 'same_domain_train_episodes_only',
			'encoder_frozen': True, 'checkpoint_transition_frozen': True,
			'controller_actor_value_reward_heads_frozen': True,
			'bootstrap_unit': 'paired_whole_test_episode',
			'seed': int(args.seed), 'bootstrap_seed': int(args.bootstrap_seed),
			'bootstrap_resamples': int(args.bootstrap_resamples),
		},
		'clr_input_audit': clr_audits,
		'clr_epsilon_sensitivity': {
			condition: clr_epsilon_sensitivity(raw_latents[condition], simnorm_dim)
			for condition in FIT_CONDITIONS
		},
		'action_coverage_audit': action_coverage,
		'normalization': {
			condition: normalization_summary(normalizations[condition])
			for condition in FIT_CONDITIONS
		},
		'parameter_matching': {
			'hidden_widths': widths,
			'tolerance': PREREGISTERED_GATES['parameter_ratio_tolerance'],
			'common_reference': 'normalized_delta_markov/action_aware_hidden_equals_latent_dim',
		},
		'training': {
			condition: {
				architecture: {
					mode: dict(fits[condition][architecture][mode].metadata)
					for mode in ACTION_MODES
				}
				for architecture in ARCHITECTURES
			}
			for condition in FIT_CONDITIONS
		},
		'evaluation': evaluation,
		'gates': gates,
		'previous_raw_refit_baseline': previous,
		'limitations': [
			'Four test episodes make bootstrap intervals exploratory and coarse.',
			'This first development diagnostic uses one deterministic fit seed; it is not '
			'a multi-seed publication estimate.',
			'Within-episode action shuffling is counterfactual and may create implausible pairs; '
			'the separately fitted actionless control is therefore mandatory.',
			'Behaviour trajectories can establish incremental action-predictive information '
			'but cannot by themselves prove causal controllability.',
			'Clean/hard cross-domain cells also include closed-loop visitation differences; '
			'only same-domain cells drive the Markov/history attribution.',
			'Failure of an equal-parameter three-state past-only model does not prove history '
			'is useless; longer or wider histories remain untested.',
			'This is frozen-latent prediction, not a controller-return experiment.',
		],
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
	parser.add_argument('--clean-dataset', type=Path, required=True)
	parser.add_argument('--hard-dataset', type=Path, required=True)
	parser.add_argument('--runtime-config', type=Path, required=True)
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--previous-refit-json', type=Path)
	parser.add_argument('--seed', type=int, default=20260913)
	parser.add_argument('--bootstrap-seed', type=int, default=271828)
	parser.add_argument('--bootstrap-resamples', type=int, default=20_000)
	parser.add_argument('--encoder-batch-size', type=int, default=256)
	parser.add_argument('--fit-batch-size', type=int, default=512)
	parser.add_argument('--max-epochs', type=int, default=240)
	parser.add_argument('--min-epochs', type=int, default=50)
	parser.add_argument('--patience', type=int, default=30)
	parser.add_argument('--convergence-window', type=int, default=20)
	parser.add_argument('--convergence-threshold', type=float, default=0.01)
	parser.add_argument('--learning-rate', type=float, default=3e-4)
	parser.add_argument('--weight-decay', type=float, default=1e-5)
	args = parser.parse_args()
	if args.bootstrap_resamples < 1000:
		parser.error('--bootstrap-resamples must be at least 1000.')
	if args.encoder_batch_size < 1 or args.fit_batch_size < 1:
		parser.error('Batch sizes must be positive.')
	if not (args.max_epochs >= args.min_epochs >= args.convergence_window >= 2):
		parser.error('Require max_epochs >= min_epochs >= convergence_window >= 2.')
	if args.patience < 1 or not (0.0 < args.convergence_threshold <= 0.05):
		parser.error('Invalid patience/convergence threshold.')
	if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
		parser.error('Invalid optimizer hyperparameters.')
	payload = evaluate(args)
	_atomic_json(args.output, payload)
	print('ROF_NORMALIZED_DELTA_COMPLETE')
	for condition, row in payload['gates']['in_domain_attribution'].items():
		print(f'ATTRIBUTION_{condition.upper()}={row["label"]}')
	print(f'OUTPUT={args.output.resolve()}')


if __name__ == '__main__':
	main()
