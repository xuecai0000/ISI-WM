"""Action-identifiable decomposed-innovation diagnostic for ROF-WM.

This is a frozen-policy, frozen-encoder development diagnostic.  It asks a
narrow question before any new controller is trained: can a transition over an
existing ROF latent separate reference drift from the part of the innovation
that is incrementally predictable from the executed action?

The transition has two explicit branches::

    delta_u = reference(history_u) + control(history_u, action)

The action-response branch is multiplicative in a normalized action embedding
and is exactly zero at the fit-domain mean action.  The matched actionless
control receives a second context-only residual path of the same width.  A
shared history trunk also feeds inverse-action probes trained from the
*observed* one-step innovation and from context alone.  These probes are an
auxiliary observational diagnostic; they do not by themselves supervise or
establish the action-response branch.  Model
selection uses only real observational transitions.  Shuffled future actions
are never used as training targets; they are an evaluation-only diagnostic and
must not be interpreted as simulator interventions or causal counterfactuals.

Clean and hard fits use the immutable 12/4/4 whole-episode splits produced by
the ROF causal-ladder collector.  Simulator state, GT masks, rewards, returns,
and controller optimizers are outside this program's data boundary.  Passing
this diagnostic can authorize a same-state interventional preflight, but never
controller training by itself.
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

from tdmpc2.tools import evaluate_rof_normalized_delta as base
from tdmpc2.tools import evaluate_rof_transition_refit as raw_refit


# Required by deterministic CuBLAS kernels.  Set before the frozen agent lazily
# imports Torch/CUDA; an existing stricter caller choice is preserved.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


FORMAT = 'rof_action_identifiable_diagnostic_v1'
STATUS = 'rof_action_identifiable_complete'
MODEL_NAME = 'decomposed_action_innovation'
ACTION_MODES = ('action_aware', 'actionless')
FIT_CONDITIONS = base.FIT_CONDITIONS
HORIZONS = base.HORIZONS
MAX_HORIZON = base.MAX_HORIZON
HISTORY_LENGTH = base.HISTORY_LENGTH
SPLIT_COUNTS = base.SPLIT_COUNTS

PREREGISTERED_GATES = {
	'interpretation': 'incremental_action_predictive_information_not_causal_control',
	'persistence_horizons': [1, 3],
	'action_information_horizons': [3, 5],
	'min_relative_gain_vs_persistence': 0.30,
	'min_relative_gain_vs_shuffled_action': 0.10,
	'min_relative_gain_vs_separately_fitted_actionless': 0.10,
	'min_inverse_action_gain_vs_context_only': 0.10,
	'min_episode_wins_out_of_4': 3,
	'max_h5_real_over_persistence': 1.0,
	'min_control_over_true_delta_rms': 0.01,
	'max_control_over_true_delta_rms': 2.0,
	'paired_bootstrap_ci_lower_must_exceed_zero': True,
	'min_action_std_for_attribution': base.PREREGISTERED_GATES[
		'min_action_std_for_attribution'
	],
	'min_action_range_for_attribution': base.PREREGISTERED_GATES[
		'min_action_range_for_attribution'
	],
	'min_action_cov_effective_rank_fraction': base.PREREGISTERED_GATES[
		'min_action_cov_effective_rank_fraction'
	],
	'min_local_action_variance_ratio': base.PREREGISTERED_GATES[
		'min_local_action_variance_ratio'
	],
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


def _stable_seed(seed: int, *parts: object) -> int:
	return base._stable_seed(seed, *parts)


def _state_features(u_history, normalization):
	"""Return current state and two past innovations, all train-normalized."""
	current = (u_history[:, -1] - normalization.u_mean) / normalization.u_scale
	recent = (
		u_history[:, -1] - u_history[:, -2] - normalization.delta_mean
	) / normalization.delta_scale
	previous = (
		u_history[:, -2] - u_history[:, -3] - normalization.delta_mean
	) / normalization.delta_scale
	try:
		import torch
		if isinstance(current, torch.Tensor):
			return torch.cat([current, recent, previous], dim=-1)
	except ImportError:
		pass
	return np.concatenate([current, recent, previous], axis=-1)


def _model_factory(
	action_mode: str, normalization: base.DeltaNormalization, *, hidden_dim: int,
):
	import torch
	import torch.nn as nn

	_require(action_mode in ACTION_MODES, f'Unknown action mode {action_mode!r}.')
	latent_dim = int(len(normalization.u_mean))
	action_dim = int(len(normalization.action_mean))
	_require(hidden_dim > 0 and latent_dim > 0 and action_dim > 0,
		'Model dimensions must be positive.')

	class _DecomposedActionInnovation(nn.Module):
		def __init__(self):
			super().__init__()
			self.history_trunk = nn.Sequential(
				nn.Linear(3 * latent_dim, hidden_dim),
				nn.LayerNorm(hidden_dim), nn.SiLU(),
				nn.Linear(hidden_dim, hidden_dim),
				nn.LayerNorm(hidden_dim), nn.SiLU(),
			)
			self.reference_head = nn.Linear(hidden_dim, latent_dim)
			self.control_context = nn.Linear(hidden_dim, hidden_dim)
			self.action_embedding = nn.Linear(action_dim, hidden_dim, bias=False)
			self.control_head = nn.Linear(hidden_dim, latent_dim, bias=False)
			self.inverse_head = nn.Sequential(
				nn.Linear(hidden_dim + latent_dim, hidden_dim), nn.SiLU(),
				nn.Linear(hidden_dim, action_dim),
			)
			self.inverse_context_head = nn.Sequential(
				nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
				nn.Linear(hidden_dim, action_dim),
			)
			# Neutral reference initialization predicts the fit-domain mean drift.
			nn.init.zeros_(self.reference_head.weight)
			nn.init.zeros_(self.reference_head.bias)
			# Keep a tiny nonzero path so the action encoder receives gradients at step 1.
			nn.init.normal_(self.control_head.weight, mean=0.0, std=1e-3)
			self.register_buffer('u_mean', torch.as_tensor(normalization.u_mean))
			self.register_buffer('u_scale', torch.as_tensor(normalization.u_scale))
			self.register_buffer(
				'action_mean', torch.as_tensor(normalization.action_mean),
			)
			self.register_buffer(
				'action_scale', torch.as_tensor(normalization.action_scale),
			)
			self.register_buffer('delta_mean', torch.as_tensor(normalization.delta_mean))
			self.register_buffer(
				'delta_scale', torch.as_tensor(normalization.delta_scale),
			)
			self.action_mode = action_mode
			self.simnorm_dim = int(normalization.simnorm_dim)

		def encode_history(self, u_history):
			return self.history_trunk(_state_features(u_history, self))

		def decompose(self, u_history, action_history):
			hidden = self.encode_history(u_history)
			reference_standardized = self.reference_head(hidden)
			action = (action_history[:, -1] - self.action_mean) / self.action_scale
			context = torch.tanh(self.control_context(hidden))
			if self.action_mode == 'actionless':
				# Active context-only residual capacity is a stronger and fairer baseline
				# than leaving the whole second branch dormant.
				control_hidden = context
			else:
				control_hidden = context * self.action_embedding(action)
			control_standardized = self.control_head(control_hidden)
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
			correction = (
				reference_groups + control_groups - projected
			)
			next_u = u_history[:, -1] + projected.reshape_as(reference)
			next_groups = next_u.reshape(len(next_u), -1, self.simnorm_dim)
			next_groups = next_groups - next_groups.mean(dim=-1, keepdim=True)
			return next_groups.reshape_as(next_u), {
				'reference_delta': projected_reference.reshape_as(reference),
				'control_delta': projected_control.reshape_as(control),
				'projection_correction': correction,
				'projection_linearity_error': (
					projected - (
						projected_reference + projected_control
					)
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

	return _DecomposedActionInnovation()


def _rollout(model, initial_history, action_window, max_horizon: int):
	"""Open-loop rollout; observed future states never update the history."""
	history = initial_history
	outputs, branches = {}, []
	for offset in range(max_horizon):
		action_history = action_window[:, offset:offset + HISTORY_LENGTH]
		next_u, branch = model.step(history, action_history)
		history = base.np_or_torch_cat_history(history, next_u)
		branches.append(branch)
		if offset + 1 in HORIZONS:
			outputs[offset + 1] = next_u
	return outputs, branches


def _objective(
	model, rows: base.SequenceRows, normalization: base.DeltaNormalization, *,
	device, batch_size: int, inverse_weight: float, control_l2_weight: float,
	training: bool,
):
	import torch

	model.train(training)
	total = {'objective': 0.0, 'prediction': 0.0, 'inverse': 0.0, 'control_l2': 0.0}
	samples = 0
	context = torch.enable_grad() if training else torch.no_grad()
	with context:
		for start in range(0, len(rows), batch_size):
			stop = min(start + batch_size, len(rows))
			history = torch.as_tensor(rows.initial_u_history[start:stop], device=device)
			actions = torch.as_tensor(rows.action_window[start:stop], device=device)
			target = torch.as_tensor(rows.target_future_u[start:stop], device=device)
			outputs, branches = _rollout(model, history, actions, MAX_HORIZON)
			prediction_terms = []
			for horizon in HORIZONS:
				scale = torch.as_tensor(
					normalization.horizon_scale[horizon], device=device,
				)
				prediction_terms.append(
					((outputs[horizon] - target[:, horizon - 1]) / scale).square().mean()
				)
			prediction = torch.stack(prediction_terms).mean()
			inverse_prediction = model.inverse_action(history, target[:, 0])
			inverse_target = (
				actions[:, HISTORY_LENGTH - 1] - model.action_mean
			) / model.action_scale
			inverse = torch.stack([
				(value - inverse_target).square().mean()
				for value in inverse_prediction.values()
			]).mean()
			control_l2 = torch.stack([
				branch['control_delta'].square().mean() for branch in branches
			]).mean()
			objective = (
				prediction + float(inverse_weight) * inverse
				+ float(control_l2_weight) * control_l2
			)
			if training:
				objective.backward()
			weight = stop - start
			for key, value in (
				('objective', objective), ('prediction', prediction),
				('inverse', inverse), ('control_l2', control_l2),
			):
				total[key] += float(value.detach().item()) * weight
			samples += weight
	return {key: value / samples for key, value in total.items()}


@dataclass
class FittedActionIdentifiableTransition:
	action_mode: str
	model: object
	normalization: base.DeltaNormalization
	device: object
	metadata: Mapping

	def rollout(self, initial_history: np.ndarray, action_window: np.ndarray) -> dict:
		import torch

		initial_history = np.asarray(initial_history, dtype=np.float32)
		action_window = np.asarray(action_window, dtype=np.float32)
		_require(initial_history.ndim == 3 and initial_history.shape[1] == HISTORY_LENGTH,
			'Initial history shape is invalid.')
		_require(action_window.ndim == 3 and action_window.shape[1] == (
			HISTORY_LENGTH - 1 + MAX_HORIZON
		), 'Action window shape is invalid.')
		with torch.no_grad():
			history = torch.as_tensor(initial_history, device=self.device)
			actions = torch.as_tensor(action_window, device=self.device)
			outputs, branches = _rollout(self.model, history, actions, MAX_HORIZON)
			predicted_u = {
				horizon: outputs[horizon].detach().cpu().numpy().astype(np.float32)
				for horizon in HORIZONS
			}
			reference = torch.cat([row['reference_delta'].flatten() for row in branches])
			control = torch.cat([row['control_delta'].flatten() for row in branches])
			correction = torch.cat([
				row['projection_correction'].flatten() for row in branches
			])
			linearity = torch.cat([
				row['projection_linearity_error'].flatten() for row in branches
			])
		return {
			'predicted_u': predicted_u,
			'predicted_z': {
				horizon: base.clr_to_simnorm(value, self.normalization.simnorm_dim)
				for horizon, value in predicted_u.items()
			},
			'branch_rms': {
				'reference': float(torch.sqrt(reference.square().mean()).item()),
				'control': float(torch.sqrt(control.square().mean()).item()),
				'projection_correction': float(
					torch.sqrt(correction.square().mean()).item()
				),
				'projection_linearity_error_max': float(linearity.abs().max().item()),
			},
		}

	def inverse_error(
		self, initial_history: np.ndarray, observed_next_u: np.ndarray,
		action: np.ndarray,
	) -> dict[str, np.ndarray]:
		import torch

		with torch.no_grad():
			history = torch.as_tensor(initial_history, device=self.device)
			next_u = torch.as_tensor(observed_next_u, device=self.device)
			prediction = self.model.inverse_action(history, next_u)
			target = (
				torch.as_tensor(action, device=self.device) - self.model.action_mean
			) / self.model.action_scale
			error = (
				prediction['with_observed_innovation'] - target
			).square().mean(dim=-1)
			context_error = (
				prediction['context_only'] - target
			).square().mean(dim=-1)
			mean_error = target.square().mean(dim=-1)
		return {
			'error': error.cpu().numpy().astype(np.float64),
			'context_only_error': context_error.cpu().numpy().astype(np.float64),
			'mean_action_error': mean_error.cpu().numpy().astype(np.float64),
		}


def fit_transition(
	action_mode: str, clr_latents: Mapping[int, np.ndarray],
	actions: Mapping[int, np.ndarray], train_ids: Sequence[int],
	validation_ids: Sequence[int], *, normalization: base.DeltaNormalization,
	device, seed: int, batch_size: int, max_epochs: int, min_epochs: int,
	patience: int, convergence_window: int, convergence_threshold: float,
	learning_rate: float, weight_decay: float, hidden_dim: int,
	inverse_weight: float, control_l2_weight: float,
) -> FittedActionIdentifiableTransition:
	"""Fit on real transitions only; no shuffled action appears in this function."""
	import torch

	_require(action_mode in ACTION_MODES, 'Invalid action mode.')
	_require(max_epochs >= min_epochs >= convergence_window >= 2,
		'Invalid convergence schedule.')
	train = base.sequence_rows(clr_latents, actions, train_ids)
	validation = base.sequence_rows(clr_latents, actions, validation_ids)
	torch.manual_seed(int(seed))
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(int(seed))
	try:
		torch.use_deterministic_algorithms(True, warn_only=True)
	except TypeError:
		torch.use_deterministic_algorithms(True)
	model = _model_factory(action_mode, normalization, hidden_dim=hidden_dim).to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay),
	)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode='min', factor=0.5, patience=max(3, patience // 3),
		min_lr=learning_rate / 32.0,
	)
	rng = np.random.default_rng(int(seed))
	best_loss, best_epoch, best_state = math.inf, 0, None
	stale, learning_curve = 0, []
	for epoch in range(1, max_epochs + 1):
		permutation = rng.permutation(len(train))
		totals = {'objective': 0.0, 'prediction': 0.0, 'inverse': 0.0, 'control_l2': 0.0}
		seen = 0
		for start in range(0, len(permutation), batch_size):
			index = permutation[start:start + batch_size]
			batch = base.SequenceRows(
				initial_u_history=train.initial_u_history[index],
				action_window=train.action_window[index],
				target_future_u=train.target_future_u[index],
				episode_ids=train.episode_ids[index],
				start_indices=train.start_indices[index],
			)
			optimizer.zero_grad(set_to_none=True)
			metrics = _objective(
				model, batch, normalization, device=device, batch_size=len(batch),
				inverse_weight=inverse_weight, control_l2_weight=control_l2_weight,
				training=True,
			)
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
			optimizer.step()
			for key in totals:
				totals[key] += metrics[key] * len(batch)
			seen += len(batch)
		validation_metrics = _objective(
			model, validation, normalization, device=device, batch_size=batch_size,
			inverse_weight=inverse_weight, control_l2_weight=control_l2_weight,
			training=False,
		)
		scheduler.step(validation_metrics['objective'])
		row = {
			'epoch': int(epoch),
			**{f'train_{key}': value / seen for key, value in totals.items()},
			**{f'validation_{key}': value for key, value in validation_metrics.items()},
			'learning_rate': float(optimizer.param_groups[0]['lr']),
		}
		learning_curve.append(row)
		value = validation_metrics['objective']
		significant = value < best_loss * (1.0 - convergence_threshold)
		if value < best_loss:
			best_loss, best_epoch = value, epoch
			best_state = {
				name: tensor.detach().cpu().clone()
				for name, tensor in model.state_dict().items()
			}
		stale = 0 if significant else stale + 1
		trailing = (
			base._trailing_relative_improvement(learning_curve, convergence_window)
			if len(learning_curve) >= convergence_window else math.inf
		)
		if epoch >= min_epochs and stale >= patience and trailing <= convergence_threshold:
			break
	_require(best_state is not None and np.isfinite(best_loss),
		'Fit never produced a finite selected model.')
	model.load_state_dict(best_state)
	model.eval()
	trailing = base._trailing_relative_improvement(learning_curve, convergence_window)
	converged = bool(
		len(learning_curve) >= min_epochs and trailing <= convergence_threshold
	)
	metadata = {
		'model': MODEL_NAME, 'action_mode': action_mode,
		'fit_objective': (
			'real_action_h1_h3_h5_open_loop_prediction_plus_inverse_action;'
			'shuffled_actions_excluded_from_fit_and_selection'
		),
		'train_episode_ids': [int(value) for value in train_ids],
		'validation_episode_ids': [int(value) for value in validation_ids],
		'test_episodes_seen_during_fit_or_selection': False,
		'train_rows': int(len(train)), 'validation_rows': int(len(validation)),
		'parameter_count': int(sum(value.numel() for value in model.parameters())),
		'hidden_dim': int(hidden_dim), 'inverse_weight': float(inverse_weight),
		'control_l2_weight': float(control_l2_weight),
		'best_epoch': int(best_epoch), 'epochs_ran': int(len(learning_curve)),
		'best_validation_objective': float(best_loss), 'converged': converged,
		'convergence': {
			'window_epochs': int(convergence_window),
			'max_trailing_relative_improvement': float(convergence_threshold),
			'observed_trailing_relative_improvement': float(trailing),
			'min_epochs': int(min_epochs), 'patience': int(patience),
		},
		'learning_curve': learning_curve,
		'normalization': base.normalization_summary(normalization),
	}
	return FittedActionIdentifiableTransition(
		action_mode, model, normalization, device, metadata,
	)


def _mean_action_rows(rows: base.SequenceRows, action_mean: np.ndarray) -> base.SequenceRows:
	windows = rows.action_window.copy()
	windows[:, HISTORY_LENGTH - 1:] = np.asarray(action_mean, dtype=np.float32)
	return base.SequenceRows(
		initial_u_history=rows.initial_u_history, action_window=windows,
		target_future_u=rows.target_future_u, episode_ids=rows.episode_ids,
		start_indices=rows.start_indices,
	)


def _episode_metrics(
	prediction: Mapping, rows: base.SequenceRows, z: np.ndarray, u: np.ndarray,
	scoring: base.DeltaNormalization, horizon: int,
) -> dict[str, float]:
	starts = rows.start_indices
	metrics = base._metric_arrays(
		prediction['predicted_u'][horizon], prediction['predicted_z'][horizon],
		u[starts], z[starts], u[starts + horizon], z[starts + horizon],
		scoring.horizon_scale[horizon], scoring.simnorm_dim,
	)
	return {key: float(value.mean()) for key, value in metrics.items()}


def evaluate_condition(
	dataset, latents: Mapping[int, np.ndarray], clr_latents: Mapping[int, np.ndarray],
	fits: Mapping[str, FittedActionIdentifiableTransition],
	fit_normalization: base.DeltaNormalization,
	scoring_normalization: base.DeltaNormalization, *, bootstrap_seed: int,
	bootstrap_resamples: int,
) -> dict:
	"""Score only untouched test episodes; shuffle is evaluation-only."""
	_require(set(fits) == set(ACTION_MODES), 'Aware/actionless fits are incomplete.')
	condition = raw_refit._condition(dataset)
	episodes = dataset.split('test')
	_require(len(episodes) == SPLIT_COUNTS['test'],
		'Evaluation requires exactly four test episodes.')
	per_horizon = {horizon: [] for horizon in HORIZONS}
	inverse_rows, branch_rows = [], []
	for episode in episodes:
		episode_id = int(episode.episode_id)
		z = np.asarray(latents[episode_id], dtype=np.float32)
		u = np.asarray(clr_latents[episode_id], dtype=np.float32)
		action = np.asarray(episode.arrays['action'], dtype=np.float32)
		rows = base.sequence_rows({episode_id: u}, {episode_id: action}, [episode_id])
		rng = np.random.default_rng(_stable_seed(
			bootstrap_seed, 'evaluation_only_shuffle', condition, episode_id,
		))
		shuffled_trace = action[rng.permutation(len(action))]
		shuffled = base.shuffled_future_rows(rows, shuffled_trace)
		mean_rows = _mean_action_rows(rows, fit_normalization.action_mean)
		rollouts = {
			'real_action': fits['action_aware'].rollout(
				rows.initial_u_history, rows.action_window,
			),
			'shuffled_action': fits['action_aware'].rollout(
				rows.initial_u_history, shuffled.action_window,
			),
			'mean_action_same_model': fits['action_aware'].rollout(
				rows.initial_u_history, mean_rows.action_window,
			),
			'separately_fitted_actionless': fits['actionless'].rollout(
				rows.initial_u_history, rows.action_window,
			),
		}
		inverse = fits['action_aware'].inverse_error(
			rows.initial_u_history, rows.target_future_u[:, 0],
			rows.action_window[:, HISTORY_LENGTH - 1],
		)
		inverse_rows.append({
			'episode_id': episode_id,
			'inverse_action_nmse': float(inverse['error'].mean()),
			'context_only_action_nmse': float(inverse['context_only_error'].mean()),
			'mean_action_nmse': float(inverse['mean_action_error'].mean()),
		})
		branch_rows.append({
			'episode_id': episode_id,
			**rollouts['real_action']['branch_rms'],
		})
		for horizon in HORIZONS:
			starts = rows.start_indices
			persistence_arrays = base._metric_arrays(
				u[starts], z[starts], u[starts], z[starts],
				u[starts + horizon], z[starts + horizon],
				scoring_normalization.horizon_scale[horizon],
				scoring_normalization.simnorm_dim,
			)
			per_horizon[horizon].append({
				'episode_id': episode_id, 'samples': int(len(starts)),
				'persistence': {
					key: float(value.mean()) for key, value in persistence_arrays.items()
				},
				'variants': {
					name: _episode_metrics(
						prediction, rows, z, u, scoring_normalization, horizon,
					)
					for name, prediction in rollouts.items()
				},
			})

	def values(rows, path):
		result = []
		for row in rows:
			value = row
			for key in path:
				value = value[key]
			result.append(float(value))
		return np.asarray(result, dtype=np.float64)

	horizon_payload = {}
	for horizon in HORIZONS:
		rows = per_horizon[horizon]
		persistence = values(rows, ('persistence', 'aitchison_mse'))
		real = values(rows, ('variants', 'real_action', 'aitchison_mse'))
		shuffled = values(rows, ('variants', 'shuffled_action', 'aitchison_mse'))
		actionless = values(
			rows, ('variants', 'separately_fitted_actionless', 'aitchison_mse'),
		)
		mean_action = values(
			rows, ('variants', 'mean_action_same_model', 'aitchison_mse'),
		)
		horizon_payload[str(horizon)] = {
			'by_episode': rows,
			'pooled': {
				'persistence_aitchison_mse': float(persistence.mean()),
				'real_action_aitchison_mse': float(real.mean()),
				'shuffled_action_aitchison_mse': float(shuffled.mean()),
				'mean_action_same_model_aitchison_mse': float(mean_action.mean()),
				'actionless_aitchison_mse': float(actionless.mean()),
			},
			'episode_wins_vs_persistence': int(np.sum(real < persistence)),
			'paired_real_vs_persistence': raw_refit.paired_episode_bootstrap(
				real, persistence,
				seed=_stable_seed(bootstrap_seed, condition, horizon, 'persistence'),
				resamples=bootstrap_resamples,
			),
			'paired_real_vs_shuffled_action': raw_refit.paired_episode_bootstrap(
				real, shuffled,
				seed=_stable_seed(bootstrap_seed, condition, horizon, 'shuffle'),
				resamples=bootstrap_resamples,
			),
			'paired_real_vs_separately_fitted_actionless': (
				raw_refit.paired_episode_bootstrap(
					real, actionless,
					seed=_stable_seed(bootstrap_seed, condition, horizon, 'actionless'),
					resamples=bootstrap_resamples,
				)
			),
			'paired_real_vs_mean_action_same_model': raw_refit.paired_episode_bootstrap(
				real, mean_action,
				seed=_stable_seed(bootstrap_seed, condition, horizon, 'mean_action'),
				resamples=bootstrap_resamples,
			),
		}
	inverse_error = values(inverse_rows, ('inverse_action_nmse',))
	inverse_context = values(inverse_rows, ('context_only_action_nmse',))
	inverse_mean = values(inverse_rows, ('mean_action_nmse',))
	return {
		'condition': condition,
		'test_episode_ids': [int(episode.episode_id) for episode in episodes],
		'evaluation_start_rule': 't>=2_and_t+5_within_same_episode',
		'action_shuffle_semantics': (
			'evaluation_only_observational_input_ablation_not_do_action_counterfactual'
		),
		'bootstrap': {
			'unit': 'paired_whole_episode', 'resamples': int(bootstrap_resamples),
		},
		'horizons': horizon_payload,
		'inverse_action': {
			'by_episode': inverse_rows,
			'paired_inverse_vs_context_only': raw_refit.paired_episode_bootstrap(
				inverse_error, inverse_context,
				seed=_stable_seed(bootstrap_seed, condition, 'inverse_context'),
				resamples=bootstrap_resamples,
			),
			'paired_inverse_vs_mean': raw_refit.paired_episode_bootstrap(
				inverse_error, inverse_mean,
				seed=_stable_seed(bootstrap_seed, condition, 'inverse'),
				resamples=bootstrap_resamples,
			),
		},
		'branch_rms_by_episode': branch_rows,
	}


def _comparison_pass(comparison: Mapping, minimum: float) -> bool:
	relative = comparison['relative_improvement']
	return bool(
		float(relative['estimate']) >= minimum and float(relative['ci95'][0]) > 0.0
	)


def score_gates(evaluation: Mapping, training: Mapping, coverage: Mapping) -> dict:
	by_condition = {}
	for condition in FIT_CONDITIONS:
		cell = evaluation[condition][condition]
		persistence_checks = {}
		for horizon in PREREGISTERED_GATES['persistence_horizons']:
			row = cell['horizons'][str(horizon)]
			persistence_checks[str(horizon)] = {
				'gain_vs_persistence': _comparison_pass(
					row['paired_real_vs_persistence'],
					PREREGISTERED_GATES['min_relative_gain_vs_persistence'],
				),
				'episode_wins': bool(
					row['episode_wins_vs_persistence']
					>= PREREGISTERED_GATES['min_episode_wins_out_of_4']
				),
			}
		action_checks = {}
		for horizon in PREREGISTERED_GATES['action_information_horizons']:
			row = cell['horizons'][str(horizon)]
			action_checks[str(horizon)] = {
				'gain_vs_evaluation_only_shuffled_action': _comparison_pass(
					row['paired_real_vs_shuffled_action'],
					PREREGISTERED_GATES['min_relative_gain_vs_shuffled_action'],
				),
				'gain_vs_actionless': _comparison_pass(
					row['paired_real_vs_separately_fitted_actionless'],
					PREREGISTERED_GATES[
						'min_relative_gain_vs_separately_fitted_actionless'
					],
				),
			}
		inverse_pass = _comparison_pass(
			cell['inverse_action']['paired_inverse_vs_context_only'],
			PREREGISTERED_GATES['min_inverse_action_gain_vs_context_only'],
		)
		h5 = cell['horizons']['5']['pooled']
		h5_ratio = h5['real_action_aitchison_mse'] / max(
			h5['persistence_aitchison_mse'], 1e-12,
		)
		branch = cell['branch_rms_by_episode']
		control_rms = float(np.mean([row['control'] for row in branch]))
		true_rms = float(np.mean([
			cell['horizons']['1']['by_episode'][index]['variants']['real_action'][
				'true_delta_rms'
			]
			for index in range(len(branch))
		]))
		control_ratio = control_rms / max(true_rms, 1e-12)
		converged = all(
			bool(training[condition][mode]['converged']) for mode in ACTION_MODES
		)
		coverage_pass = bool(
			coverage[condition]['train'][
				'coverage_sufficient_for_incremental_action_attribution'
			]
			and coverage[condition]['test'][
				'coverage_sufficient_for_incremental_action_attribution'
			]
		)
		all_required = bool(
			all(all(row.values()) for row in persistence_checks.values())
			and all(all(row.values()) for row in action_checks.values())
		)
		candidate = bool(
			converged and coverage_pass and all_required and inverse_pass
			and h5_ratio <= PREREGISTERED_GATES['max_h5_real_over_persistence']
			and PREREGISTERED_GATES['min_control_over_true_delta_rms']
			<= control_ratio
			<= PREREGISTERED_GATES['max_control_over_true_delta_rms']
		)
		by_condition[condition] = {
			'persistence_horizon_checks': persistence_checks,
			'action_information_horizon_checks': action_checks,
			'inverse_action_pass': inverse_pass, 'h5_ratio': float(h5_ratio),
			'h5_no_divergence': bool(
				h5_ratio <= PREREGISTERED_GATES['max_h5_real_over_persistence']
			),
			'control_over_true_delta_rms': float(control_ratio),
			'control_magnitude_pass': bool(
				PREREGISTERED_GATES['min_control_over_true_delta_rms']
				<= control_ratio
				<= PREREGISTERED_GATES['max_control_over_true_delta_rms']
			),
			'all_fits_converged': converged, 'action_coverage_pass': coverage_pass,
			'observational_action_identifiability_candidate': candidate,
		}
	return {
		'preregistered_thresholds': dict(PREREGISTERED_GATES),
		'by_condition': by_condition,
		'both_conditions_candidate': bool(all(
			row['observational_action_identifiability_candidate']
			for row in by_condition.values()
		)),
		'true_same_state_intervention_required_before_controller_training': True,
		'interpretation_guard': (
			'Action shuffle on logged behaviour is not a do(action) intervention.  A pass '
			'only supports incremental predictive information and must be confirmed on '
			'same-state real action branches before changing or training a controller.'
		),
	}


def validate_result(payload: Mapping) -> None:
	_require(payload.get('format') == FORMAT and payload.get('status') == STATUS,
		'Result format/status is incomplete.')
	_require(payload.get('engineering_pass') is True, 'Engineering pass is missing.')
	_require(payload.get('scientific_complete') is False,
		'An observational diagnostic cannot claim scientific completion.')
	_require(payload.get('controller_training_authorized') is False,
		'This diagnostic must never authorize controller training.')
	_require(payload.get('causal_claim_authorized') is False,
		'This observational diagnostic must never authorize a causal claim.')
	_require(payload.get('policy_training_performed') is False,
		'Policy training is forbidden in the diagnostic.')
	_require(payload.get('privileged_targets_used') is False,
		'Privileged targets are forbidden.')
	_require(payload.get('simulator_state_used') is False,
		'Simulator state is forbidden.')
	protocol = payload.get('protocol', {})
	_require(protocol.get('action_shuffle_training_use') == 'forbidden',
		'Shuffled actions must be excluded from training.')
	_require(protocol.get('action_shuffle_evaluation_semantics') == (
		'observational_ablation_not_causal_counterfactual'
	), 'Shuffle interpretation guard is missing.')
	_require(protocol.get('episode_split_counts') == SPLIT_COUNTS,
		'Episode split contract changed.')
	training = payload.get('training', {})
	_require(set(training) == set(FIT_CONDITIONS), 'Training condition set is incomplete.')
	for condition in FIT_CONDITIONS:
		_require(set(training[condition]) == set(ACTION_MODES),
			'Training action-mode set is incomplete.')
		for mode in ACTION_MODES:
			row = training[condition][mode]
			train_ids = row.get('train_episode_ids', [])
			validation_ids = row.get('validation_episode_ids', [])
			_require(len(train_ids) == SPLIT_COUNTS['train'],
				'Train episode count changed.')
			_require(len(validation_ids) == SPLIT_COUNTS['validation'],
				'Validation episode count changed.')
			_require(not (set(train_ids) & set(validation_ids)),
				'Train/validation episode ids overlap.')
			_require(row.get('test_episodes_seen_during_fit_or_selection') is False,
				'Test data leaked into fitting.')
	evaluation = payload.get('evaluation', {})
	_require(set(evaluation) == set(FIT_CONDITIONS), 'Evaluation matrix is incomplete.')
	for fit_condition in FIT_CONDITIONS:
		_require(set(evaluation[fit_condition]) == set(FIT_CONDITIONS),
			'Evaluation 2x2 matrix is incomplete.')
		for test_condition in FIT_CONDITIONS:
			cell = evaluation[fit_condition][test_condition]
			_require(cell.get(
				'action_shuffle_semantics'
			) == 'evaluation_only_observational_input_ablation_not_do_action_counterfactual',
				'Per-cell shuffle guard is missing.')
			for mode in ACTION_MODES:
				fit_ids = set(training[fit_condition][mode]['train_episode_ids'])
				fit_ids.update(training[fit_condition][mode]['validation_episode_ids'])
				_require(not (fit_ids & set(cell.get('test_episode_ids', []))),
					'Test episode ids overlap fit or validation ids.')
			for branch in cell.get('branch_rms_by_episode', []):
				_require(float(branch.get('projection_linearity_error_max', math.inf))
					<= 1e-6, 'Projected branch decomposition is not additive.')
	_require(payload.get('gates', {}).get(
		'true_same_state_intervention_required_before_controller_training'
	) is True, 'A same-state intervention must remain mandatory.')
	json.dumps(payload, allow_nan=False)


def evaluate(args) -> dict:
	clean = base.load_policy_dataset(args.clean_dataset)
	hard = base.load_policy_dataset(args.hard_dataset)
	runtime_config = Path(args.runtime_config).resolve()
	checkpoint = Path(args.checkpoint).resolve()
	pair = raw_refit.validate_dataset_pair(clean, hard, runtime_config, checkpoint)
	agent = base.ladder._load_agent(clean, runtime_config, checkpoint)
	for parameter in agent.model.parameters():
		parameter.requires_grad_(False)
	agent.eval()
	simnorm_dim = int(agent.cfg.get('simnorm_dim', 8))
	raw_latents = {
		'clean': base.ladder._encode_episodes(
			clean, agent, batch_size=args.encoder_batch_size,
		),
		'hard': base.ladder._encode_episodes(
			hard, agent, batch_size=args.encoder_batch_size,
		),
	}
	clr_latents = {
		condition: base.encode_clr_latents(values, simnorm_dim)[0]
		for condition, values in raw_latents.items()
	}
	datasets = {'clean': clean, 'hard': hard}
	actions = {
		condition: base.action_mapping(dataset)
		for condition, dataset in datasets.items()
	}
	normalizations = {
		condition: base.fit_delta_normalization(
			clr_latents[condition], actions[condition],
			datasets[condition].splits['train'], simnorm_dim=simnorm_dim,
		)
		for condition in FIT_CONDITIONS
	}
	fits = {condition: {} for condition in FIT_CONDITIONS}
	for condition in FIT_CONDITIONS:
		for mode in ACTION_MODES:
			fits[condition][mode] = fit_transition(
				mode, clr_latents[condition], actions[condition],
				datasets[condition].splits['train'],
				datasets[condition].splits['validation'],
				normalization=normalizations[condition], device=agent.device,
				seed=_stable_seed(args.seed, condition, mode),
				batch_size=args.fit_batch_size, max_epochs=args.max_epochs,
				min_epochs=args.min_epochs, patience=args.patience,
				convergence_window=args.convergence_window,
				convergence_threshold=args.convergence_threshold,
				learning_rate=args.learning_rate, weight_decay=args.weight_decay,
				hidden_dim=args.hidden_dim, inverse_weight=args.inverse_weight,
				control_l2_weight=args.control_l2_weight,
			)
	evaluation = {condition: {} for condition in FIT_CONDITIONS}
	for fit_condition in FIT_CONDITIONS:
		for test_condition in FIT_CONDITIONS:
			evaluation[fit_condition][test_condition] = evaluate_condition(
				datasets[test_condition], raw_latents[test_condition],
				clr_latents[test_condition], fits[fit_condition],
				normalizations[fit_condition], normalizations['clean'],
				bootstrap_seed=_stable_seed(
					args.bootstrap_seed, fit_condition, test_condition,
				),
				bootstrap_resamples=args.bootstrap_resamples,
			)
	coverage = {
		condition: {
			split: base.action_coverage_audit(
				datasets[condition], clr_latents[condition], normalizations[condition],
				datasets[condition].splits[split],
			)
			for split in ('train', 'test')
		}
		for condition in FIT_CONDITIONS
	}
	training = {
		condition: {
			mode: dict(fits[condition][mode].metadata) for mode in ACTION_MODES
		}
		for condition in FIT_CONDITIONS
	}
	gates = score_gates(evaluation, training, coverage)
	payload = {
		'format': FORMAT, 'status': STATUS, 'engineering_pass': True,
		'observational_fit_complete': bool(all(
			training[condition][mode]['converged']
			for condition in FIT_CONDITIONS for mode in ACTION_MODES
		)),
		'scientific_complete': False, 'causal_claim_authorized': False,
		'controller_training_authorized': False,
		'policy_training_performed': False, 'privileged_targets_used': False,
		'simulator_state_used': False,
		'task': clean.task,
		'recommendation': (
			'run_true_same_state_action_branches'
			if gates['both_conditions_candidate']
			else 'do_not_train_controller_action_identifiability_gate_failed'
		),
		'source': {
			'clean_dataset': str(clean.manifest_path),
			'clean_dataset_sha256': _sha256(clean.manifest_path),
			'hard_dataset': str(hard.manifest_path),
			'hard_dataset_sha256': _sha256(hard.manifest_path),
			'runtime_config': str(runtime_config),
			'runtime_config_sha256': _sha256(runtime_config),
			'checkpoint': str(checkpoint),
			'checkpoint_sha256': _sha256(checkpoint),
			'checkpoint_step': pair['checkpoint_step'],
			'encoder_frozen': True, 'policy_heads_frozen': True,
		},
		'protocol': {
			'fit_conditions': list(FIT_CONDITIONS),
			'episode_split_counts': dict(SPLIT_COUNTS),
			'horizons': list(HORIZONS), 'history_length_states': HISTORY_LENGTH,
			'teacher_forcing_after_initial_history': False,
			'fit_objective': (
				'real_action_open_loop_prediction_plus_observational_inverse_auxiliary'
			),
			'inverse_auxiliary_semantics': (
				'with_observed_innovation_must_beat_context_only;auxiliary_does_not_'
				'itself_prove_or_supervise_causal_action_response'
			),
			'action_shuffle_training_use': 'forbidden',
			'action_shuffle_evaluation_semantics': (
				'observational_ablation_not_causal_counterfactual'
			),
			'actionless_control': (
				'separately_fitted_same_width_active_context_residual_baseline'
			),
			'bootstrap_unit': 'paired_whole_test_episode',
			'bootstrap_resamples': int(args.bootstrap_resamples),
			'seed': int(args.seed),
		},
		'normalization': {
			condition: base.normalization_summary(normalizations[condition])
			for condition in FIT_CONDITIONS
		},
		'action_coverage_audit': coverage,
		'training': training, 'evaluation': evaluation, 'gates': gates,
		'limitations': [
			'Logged behaviour can show incremental action-predictive information but '
			'cannot establish causal controllability.',
			'Clean and hard collections are independent closed-loop trajectories and are '
			'not used as same-state representation-consistency pairs.',
			'The encoder is frozen; failure can mean its state lacks stable action-relevant '
			'information rather than that every decomposed dynamics architecture must fail.',
			'Only four whole test episodes per condition make confidence intervals coarse.',
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


def main(argv=None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--clean-dataset', type=Path, required=True)
	parser.add_argument('--hard-dataset', type=Path, required=True)
	parser.add_argument('--runtime-config', type=Path, required=True)
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--seed', type=int, default=20260913)
	parser.add_argument('--bootstrap-seed', type=int, default=314159)
	parser.add_argument('--bootstrap-resamples', type=int, default=20_000)
	parser.add_argument('--encoder-batch-size', type=int, default=256)
	parser.add_argument('--fit-batch-size', type=int, default=512)
	parser.add_argument('--hidden-dim', type=int, default=512)
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
	print('ROF_ACTION_IDENTIFIABLE_COMPLETE')
	for condition, row in payload['gates']['by_condition'].items():
		print(
			f'ACTION_IDENTIFIABLE_{condition.upper()}='
			f'{row["observational_action_identifiability_candidate"]}'
		)
	print(f'OUTPUT={args.output.resolve()}')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
