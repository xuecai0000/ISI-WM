"""Offline counterfactual-input diagnosis for a frozen ROF-WM checkpoint.

This tool does not train, fine-tune, or select a controller.  It re-encodes one
immutable causal-probe dataset under five tightly scoped interventions and
then evaluates the *same frozen* 30k checkpoint:

``O``
	Original policy observation.
``R``
	For each of the three stacked RGB frames, replace pixels outside the union
	of the corresponding online role masks by the constant uint8 value 128.
``M``
	Replace only the online mask stack by episode-local, temporally aligned
	scoring-only simulator masks.  This is an offline oracle intervention.
``Q``
	Replace only descriptor coordinates ``0:512`` in each role/frame by the
	mean from clean training episodes for that role and stack position.
``RQ``
	Apply R and Q together.

All non-targeted policy fields are asserted bit-identical.  Simulator masks
remain under ``labels__`` and are used only by M; they never become training
inputs.  Improvements are one-way causal evidence that the intervened path is
a bottleneck.  Failure to improve does not prove that the path is harmless,
because the frozen downstream model was trained on O and may reject an
out-of-distribution counterfactual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from tdmpc2.tools import evaluate_rof_causal_ladder as ladder


FORMAT = 'rof_counterfactual_input_diagnostic_v1'
INTERVENTIONS = ('O', 'R', 'M', 'Q', 'RQ')
FRAME_DIM = 590
STACK_FRAMES = 3
QUERY_DIM = 512
BACKGROUND_VALUE = 128
LOCAL_TOKENS = 4
GLOBAL_TOKENS = 1


def _require(condition: bool, message: str) -> None:
	if not condition:
		raise ValueError(message)


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as stream:
		for block in iter(lambda: stream.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
	return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _condition(dataset: ladder.FrozenDataset) -> str:
	value = dataset.manifest.get('condition')
	if value is None:
		value = dataset.manifest.get('collection', {}).get('condition')
	_require(value in {'clean', 'hard'}, 'Dataset condition must be clean or hard.')
	return str(value)


def _policy_copy(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
	"""Copy exactly the four model-facing arrays; labels cannot cross here."""
	_require(set(ladder.POLICY_KEYS).issubset(arrays), 'Policy arrays are incomplete.')
	return {
		name: np.array(arrays[name], copy=True, order='C')
		for name in ladder.POLICY_KEYS
	}


def aligned_gt_mask_stack(labels: np.ndarray) -> np.ndarray:
	"""Convert current-frame GT ``[N,K,H,W]`` to causal ``[N,K,3,H,W]``.

	The online observation stack uses boundary replication at reset.  Therefore
	observation t receives GT masks at max(0,t-2), max(0,t-1), and t.  Alignment
	is episode-local, so no frame can cross an episode boundary.
	"""
	labels = np.asarray(labels)
	_require(
		labels.ndim == 4 and labels.dtype == np.bool_ and labels.shape[-2:] == (64, 64),
		'GT role masks must be bool [N,K,64,64].',
	)
	length = labels.shape[0]
	index = np.arange(length, dtype=np.int64)
	stack_index = np.stack([
		np.maximum(index - 2, 0), np.maximum(index - 1, 0), index,
	], axis=1)
	# labels[stack_index] is [N,3,K,H,W]; policy masks are [N,K,3,H,W].
	return np.ascontiguousarray(labels[stack_index].transpose(0, 2, 1, 3, 4))


def clean_train_query_mean(dataset: ladder.FrozenDataset) -> np.ndarray:
	"""Return train-only query means with shape ``[K,3,512]``."""
	_require(_condition(dataset) == 'clean', 'Query reference dataset must be clean.')
	blocks = []
	for episode in dataset.split('train'):
		objects = episode.arrays['policy_object']
		_require(
			objects.ndim == 3 and objects.shape[-1] == FRAME_DIM * STACK_FRAMES,
			'ROF object descriptors must be [N,K,1770].',
		)
		blocks.append(objects.reshape(
			len(objects), len(dataset.role_names), STACK_FRAMES, FRAME_DIM
		)[..., :QUERY_DIM])
	_require(blocks, 'Clean training query reference is empty.')
	result = np.concatenate(blocks, axis=0).mean(axis=0, dtype=np.float64)
	result = result.astype(np.float32)
	_require(np.isfinite(result).all(), 'Clean training query mean is non-finite.')
	return result


def _replace_background(rgb: np.ndarray, masks: np.ndarray) -> np.ndarray:
	_require(rgb.ndim == 4 and rgb.shape[1:] == (9, 64, 64),
		'RGB must be [N,9,64,64].')
	_require(masks.ndim == 5 and masks.shape[2:] == (3, 64, 64),
		'Online masks must be [N,K,3,64,64].')
	frames = np.array(rgb, copy=True, order='C').reshape(-1, STACK_FRAMES, 3, 64, 64)
	union = masks.any(axis=1)
	frames[...] = np.where(
		union[:, :, None], frames, np.asarray(BACKGROUND_VALUE, dtype=np.uint8)
	)
	return np.ascontiguousarray(frames.reshape(rgb.shape))


def _replace_queries(objects: np.ndarray, query_mean: np.ndarray) -> np.ndarray:
	_require(objects.ndim == 3 and objects.shape[-1] == FRAME_DIM * STACK_FRAMES,
		'Objects must be [N,K,1770].')
	_require(query_mean.shape == (objects.shape[1], STACK_FRAMES, QUERY_DIM),
		'Query mean shape differs from the target role contract.')
	frames = np.array(objects, copy=True, order='C').reshape(
		len(objects), objects.shape[1], STACK_FRAMES, FRAME_DIM
	)
	frames[..., :QUERY_DIM] = query_mean[None]
	return np.ascontiguousarray(frames.reshape(objects.shape))


def intervene_episode(
	arrays: Mapping[str, np.ndarray], mode: str, query_mean: np.ndarray,
) -> dict[str, np.ndarray]:
	"""Build one counterfactual policy episode and assert its exact scope."""
	_require(mode in INTERVENTIONS, f'Unknown intervention {mode!r}.')
	original = _policy_copy(arrays)
	result = _policy_copy(arrays)
	if 'R' in mode:
		result['policy_rgb'] = _replace_background(
			original['policy_rgb'], original['policy_object_mask']
		)
	if mode == 'M':
		_require('labels__gt_role_mask' in arrays,
			'M requires scoring-only labels__gt_role_mask.')
		result['policy_object_mask'] = aligned_gt_mask_stack(
			arrays['labels__gt_role_mask']
		)
	if 'Q' in mode:
		result['policy_object'] = _replace_queries(
			original['policy_object'], query_mean
		)
	assert_intervention_contract(original, result, arrays, mode, query_mean)
	return result


def assert_intervention_contract(
	original: Mapping[str, np.ndarray], result: Mapping[str, np.ndarray],
	all_arrays: Mapping[str, np.ndarray], mode: str, query_mean: np.ndarray,
) -> None:
	"""Fail closed if an intervention changes any unspecified policy byte."""
	_require(set(original) == set(result) == set(ladder.POLICY_KEYS),
		'Intervention boundary must contain exactly the policy keys.')
	for name in ladder.POLICY_KEYS:
		_require(result[name].shape == original[name].shape,
			f'{mode} changed shape of {name}.')
		_require(result[name].dtype == original[name].dtype,
			f'{mode} changed dtype of {name}.')
	if mode == 'O':
		for name in ladder.POLICY_KEYS:
			_require(np.array_equal(result[name], original[name]),
				'O must be bit-identical.')
		return
	_require(np.array_equal(result['policy_role_exists'], original['policy_role_exists']),
		f'{mode} changed role_exists.')
	if mode in {'R', 'Q', 'RQ'}:
		_require(np.array_equal(result['policy_object_mask'], original['policy_object_mask']),
			f'{mode} changed online masks.')
	if mode in {'R', 'M'}:
		_require(np.array_equal(result['policy_object'], original['policy_object']),
			f'{mode} changed object descriptors.')
	if mode in {'M', 'Q'}:
		_require(np.array_equal(result['policy_rgb'], original['policy_rgb']),
			f'{mode} changed RGB unexpectedly.')
	if 'R' in mode:
		expected = _replace_background(
			original['policy_rgb'], original['policy_object_mask']
		)
		_require(np.array_equal(result['policy_rgb'], expected),
			f'{mode} background replacement is malformed.')
	if mode == 'M':
		expected = aligned_gt_mask_stack(all_arrays['labels__gt_role_mask'])
		_require(np.array_equal(result['policy_object_mask'], expected),
			'M mask stack is not temporally aligned GT.')
	if 'Q' in mode:
		before = original['policy_object'].reshape(
			len(original['policy_object']), original['policy_object'].shape[1],
			STACK_FRAMES, FRAME_DIM,
		)
		after = result['policy_object'].reshape(before.shape)
		_require(np.array_equal(after[..., :QUERY_DIM], np.broadcast_to(
			query_mean, after[..., :QUERY_DIM].shape
		)), f'{mode} query replacement is malformed.')
		_require(np.array_equal(after[..., QUERY_DIM:], before[..., QUERY_DIM:]),
			f'{mode} changed descriptor geometry/status outside query[0:512].')


def _obs_to_device(policy: Mapping[str, np.ndarray], start: int, stop: int, device):
	import torch
	return {
		'rgb': torch.as_tensor(policy['policy_rgb'][start:stop], device=device),
		'object': torch.as_tensor(policy['policy_object'][start:stop], device=device),
		'object_mask': torch.as_tensor(
			policy['policy_object_mask'][start:stop], device=device
		),
		'role_exists': torch.as_tensor(
			policy['policy_role_exists'][start:stop], device=device
		),
	}


def encode_interventions(
	dataset: ladder.FrozenDataset, agent, query_mean: np.ndarray, *, batch_size: int,
	modes: Sequence[str] = INTERVENTIONS,
) -> dict[str, dict[str, dict[int, np.ndarray]]]:
	"""Encode O/R/M/Q/RQ with the same frozen encoder and identity crop."""
	import torch
	modes = tuple(modes)
	_require(modes and len(set(modes)) == len(modes),
		'Encoding modes must be unique and non-empty.')
	_require(set(modes).issubset(INTERVENTIONS), 'Unknown encoding mode requested.')
	encoder = agent.model._encoder['object']
	pad = int(getattr(encoder.augmentation, 'pad', 3))
	result = {
		mode: {'flat': {}, 'tokens': {}} for mode in modes
	}
	with torch.no_grad():
		for episode in dataset.episodes:
			policies = {
				mode: intervene_episode(episode.arrays, mode, query_mean)
				for mode in modes
			}
			length = episode.decisions + 1
			for mode in modes:
				flat_rows, token_rows = [], []
				for start in range(0, length, batch_size):
					stop = min(start + batch_size, length)
					obs = _obs_to_device(policies[mode], start, stop, agent.device)
					shift = torch.full(
						(stop - start, 1, 1, 2), float(pad),
						device=agent.device, dtype=torch.float32,
					)
					flat, tokens = encoder(obs, shift_index=shift, return_tokens=True)
					_require(flat.ndim == 2 and tokens.ndim == 4,
						'ROF encoder returned an unexpected token shape.')
					_require(tokens.shape[-2:] == (LOCAL_TOKENS + GLOBAL_TOKENS, 64),
						'Counterfactual diagnostic requires ROF 4-local+1-global tokens.')
					flat_rows.append(flat.detach().cpu().numpy().astype(np.float64))
					token_rows.append(tokens.detach().cpu().numpy().astype(np.float64))
				result[mode]['flat'][episode.episode_id] = np.concatenate(flat_rows)
				result[mode]['tokens'][episode.episode_id] = np.concatenate(token_rows)
	return result


def _concat(values: Mapping[int, np.ndarray], ids: Sequence[int]) -> np.ndarray:
	_require(ids, 'Episode id selection is empty.')
	return np.concatenate([values[int(value)] for value in ids], axis=0)


def _summary(value: np.ndarray) -> dict:
	value = np.asarray(value, dtype=np.float64).reshape(-1)
	_require(value.size > 0 and np.isfinite(value).all(), 'Summary values are invalid.')
	return {
		'samples': int(value.size),
		'mean': float(value.mean()),
		'median': float(np.median(value)),
		'p95': float(np.quantile(value, 0.95)),
		'p99': float(np.quantile(value, 0.99)),
		'max': float(value.max()),
	}


def _views(tokens: np.ndarray) -> dict[str, np.ndarray]:
	_require(tokens.ndim == 4 and tokens.shape[-2:] == (5, 64),
		'Tokens must be [N,K,5,64].')
	return {
		'full': tokens.reshape(len(tokens), -1),
		'local': tokens[:, :, :LOCAL_TOKENS].reshape(len(tokens), -1),
		'global': tokens[:, :, LOCAL_TOKENS:].reshape(len(tokens), -1),
	}


def token_perturbation(original: np.ndarray, changed: np.ndarray) -> dict:
	_require(original.shape == changed.shape, 'Perturbation tokens are misaligned.')
	result = {}
	for name, before in _views(original).items():
		after = _views(changed)[name]
		delta = after - before
		result[name] = {
			'rms_per_frame': _summary(np.sqrt(np.mean(np.square(delta), axis=1))),
			'l2_per_frame': _summary(np.sqrt(np.sum(np.square(delta), axis=1))),
		}
	return result


def fit_train_clean_ood_reference(tokens: np.ndarray) -> dict[str, dict[str, object]]:
	"""Fit label-free centroid radii using only original clean-train tokens."""
	result = {}
	for name, values in _views(tokens).items():
		center = values.mean(axis=0, dtype=np.float64)
		distance = np.sqrt(np.mean(np.square(values - center), axis=1))
		result[name] = {
			'center': center,
			'radius_95': float(np.quantile(distance, 0.95)),
			'radius_99': float(np.quantile(distance, 0.99)),
			'train_distance': _summary(distance),
		}
	return result


def score_ood(tokens: np.ndarray, reference: Mapping[str, Mapping]) -> dict:
	result = {}
	for name, values in _views(tokens).items():
		row = reference[name]
		center = np.asarray(row['center'], dtype=np.float64)
		distance = np.sqrt(np.mean(np.square(values - center), axis=1))
		radius95, radius99 = float(row['radius_95']), float(row['radius_99'])
		result[name] = {
			'reference_radius_95': radius95,
			'reference_radius_99': radius99,
			'distance_to_train_clean_centroid': _summary(distance),
			'fraction_outside_95_radius': float(np.mean(distance > radius95)),
			'fraction_outside_99_radius': float(np.mean(distance > radius99)),
			'heuristic_support_flag': bool(np.mean(distance > radius99) <= 0.05),
			'heuristic_support_flag_rule': (
				'fraction_outside_train_clean_99_radius <= 0.05; descriptive only'
			),
		}
	return result


def latent_variability(
	latents: Mapping[int, np.ndarray], episode_ids: Sequence[int],
) -> dict:
	"""Measure within-episode variation and PCA participation ratio.

	Centering is performed separately inside every episode so a difference in
	reset states cannot make an otherwise constant temporal latent look rich.
	The participation ratio is ``trace(C)^2 / trace(C^2)`` for the full
	within-episode covariance and therefore requires no arbitrary PCA cutoff.
	"""
	_require(episode_ids, 'Latent variability requires test episodes.')
	centered_rows, changes, episode_rows = [], [], {}
	width = None
	for episode_id in episode_ids:
		value = np.asarray(latents[int(episode_id)], dtype=np.float64)
		_require(value.ndim == 2 and len(value) >= 2 and np.isfinite(value).all(),
			f'Latent episode {episode_id} is invalid.')
		width = value.shape[1] if width is None else width
		_require(value.shape[1] == width, 'Latent widths differ between episodes.')
		centered = value - value.mean(axis=0, keepdims=True)
		delta = np.sqrt(np.mean(np.square(np.diff(value, axis=0)), axis=1))
		variance = np.mean(np.square(centered), axis=0)
		centered_rows.append(centered)
		changes.append(delta)
		episode_rows[str(episode_id)] = {
			'frames': int(len(value)),
			'mean_coordinate_temporal_variance': float(variance.mean()),
			'total_temporal_variance': float(variance.sum()),
			'rms_consecutive_change': _summary(delta),
		}
	centered = np.concatenate(centered_rows, axis=0)
	denominator = max(len(centered) - len(episode_ids), 1)
	covariance = (centered.T @ centered) / denominator
	total_variance = float(np.trace(covariance))
	squared_spectrum_sum = float(np.square(covariance).sum())
	participation_ratio = (
		total_variance ** 2 / squared_spectrum_sum
		if squared_spectrum_sum > 1e-24 else 0.0
	)
	all_changes = np.concatenate(changes)
	mean_coordinate_variance = total_variance / max(int(width), 1)
	return {
		'schema': 'rof_counterfactual_latent_variability_v1',
		'centering': 'per_episode_before_full_test_covariance',
		'episodes': len(episode_ids),
		'frames': int(len(centered)),
		'latent_dim': int(width),
		'within_episode_mean_coordinate_variance': mean_coordinate_variance,
		'within_episode_total_variance': total_variance,
		'rms_consecutive_change': _summary(all_changes),
		'pca_participation_ratio': float(participation_ratio),
		'pca_participation_ratio_fraction_of_latent_dim': float(
			participation_ratio / max(int(width), 1)
		),
		'near_constant_heuristic': bool(
			mean_coordinate_variance < 1e-8
			or float(all_changes.mean()) < 1e-5
		),
		'near_constant_heuristic_rule': (
			'mean_coordinate_variance < 1e-8 or mean_rms_change < 1e-5'
		),
		'by_episode': episode_rows,
	}


def _bootstrap_summary(
	values: np.ndarray, *, seed: int, resamples: int,
) -> dict:
	values = np.asarray(values, dtype=np.float64).reshape(-1)
	_require(len(values) >= 2 and np.isfinite(values).all(),
		'Bootstrap values must contain at least two finite episodes.')
	_require(resamples >= 1_000, 'At least 1000 bootstrap resamples are required.')
	rng = np.random.default_rng(seed)
	indices = rng.integers(0, len(values), size=(resamples, len(values)))
	means = values[indices].mean(axis=1)
	return {
		'estimate_episode_mean': float(values.mean()),
		'ci95': [
			float(np.quantile(means, 0.025)),
			float(np.quantile(means, 0.975)),
		],
		'episodes': int(len(values)),
		'resamples': int(resamples),
		'seed': int(seed),
	}


def stage_c_episode_bootstrap(
	dataset: ladder.FrozenDataset, latents: Mapping[int, np.ndarray], agent, *,
	bootstrap_seed: int, bootstrap_resamples: int,
) -> dict:
	"""Frozen Stage C with episode rows and paired episode bootstrap CIs."""
	import torch
	test_episodes = dataset.split('test')
	_require(len(test_episodes) == 4,
		'Counterfactual Stage C requires exactly four held-out test episodes.')
	horizons = {}
	with torch.no_grad():
		for horizon_index, horizon in enumerate(ladder.HORIZONS):
			episode_rows = {}
			pooled = {
				'model': [], 'persistence': [], 'shuffle': [],
				'action_sensitivity': [], 'action_input_difference': [],
			}
			for episode in test_episodes:
				z = np.asarray(latents[episode.episode_id], dtype=np.float32)
				action = episode.arrays['action'].astype(np.float32)
				count = episode.decisions - horizon + 1
				_require(count > 0 and len(z) == episode.decisions + 1,
					'Stage C latent/action alignment is invalid.')
				current = torch.as_tensor(z[:count], device=agent.device)
				prediction = current
				shuffled_prediction = current.clone()
				rng = np.random.default_rng(104729 + episode.episode_id)
				shuffled_action = action[rng.permutation(len(action))]
				action_differences = []
				for offset in range(horizon):
					real_action = torch.as_tensor(
						action[offset:offset + count], device=agent.device
					)
					permuted_action = torch.as_tensor(
						shuffled_action[offset:offset + count], device=agent.device
					)
					action_differences.append(
						(real_action - permuted_action).square().mean(dim=-1).cpu().numpy()
					)
					prediction = agent.model.next(prediction, real_action, None)
					shuffled_prediction = agent.model.next(
						shuffled_prediction, permuted_action, None
					)
				target = torch.as_tensor(
					z[horizon:horizon + count], device=agent.device
				)
				model_error = (prediction - target).square().mean(dim=-1).cpu().numpy()
				persistence_error = (current - target).square().mean(dim=-1).cpu().numpy()
				shuffle_error = (
					(shuffled_prediction - target).square().mean(dim=-1).cpu().numpy()
				)
				action_sensitivity = (
					(prediction - shuffled_prediction).square().mean(dim=-1).cpu().numpy()
				)
				action_difference = np.concatenate(action_differences)
				model_mse = float(model_error.mean())
				persistence_mse = float(persistence_error.mean())
				shuffle_mse = float(shuffle_error.mean())
				episode_rows[str(episode.episode_id)] = {
					'samples': int(count),
					'model_mse': model_mse,
					'persistence_mse': persistence_mse,
					'action_shuffle_mse': shuffle_mse,
					'model_over_persistence': model_mse / max(persistence_mse, 1e-12),
					'model_over_action_shuffle': model_mse / max(shuffle_mse, 1e-12),
					'model_minus_persistence_mse': model_mse - persistence_mse,
					'model_minus_action_shuffle_mse': model_mse - shuffle_mse,
					'action_sensitivity_mse': float(action_sensitivity.mean()),
					'action_shuffle_input_mse': float(action_difference.mean()),
				}
				pooled['model'].append(model_error)
				pooled['persistence'].append(persistence_error)
				pooled['shuffle'].append(shuffle_error)
				pooled['action_sensitivity'].append(action_sensitivity)
				pooled['action_input_difference'].append(action_difference)
			model_mse = float(np.concatenate(pooled['model']).mean())
			persistence_mse = float(np.concatenate(pooled['persistence']).mean())
			shuffle_mse = float(np.concatenate(pooled['shuffle']).mean())
			rows = list(episode_rows.values())
			metrics = {
				'model_over_persistence': np.asarray([
					row['model_over_persistence'] for row in rows
				]),
				'model_over_action_shuffle': np.asarray([
					row['model_over_action_shuffle'] for row in rows
				]),
				'model_minus_persistence_mse': np.asarray([
					row['model_minus_persistence_mse'] for row in rows
				]),
				'model_minus_action_shuffle_mse': np.asarray([
					row['model_minus_action_shuffle_mse'] for row in rows
				]),
			}
			horizons[str(horizon)] = {
				'pooled_frames': {
					'samples': int(sum(len(value) for value in pooled['model'])),
					'model_mse': model_mse,
					'persistence_mse': persistence_mse,
					'action_shuffle_mse': shuffle_mse,
					'model_over_persistence': model_mse / max(persistence_mse, 1e-12),
					'model_over_action_shuffle': model_mse / max(shuffle_mse, 1e-12),
					'action_sensitivity_mse': float(np.concatenate(
						pooled['action_sensitivity']
					).mean()),
					'action_shuffle_input_mse': float(np.concatenate(
						pooled['action_input_difference']
					).mean()),
				},
				'by_episode': episode_rows,
				'paired_episode_bootstrap': {
					name: _bootstrap_summary(
						value,
						seed=bootstrap_seed + 100 * horizon_index + metric_index,
						resamples=bootstrap_resamples,
					)
					for metric_index, (name, value) in enumerate(metrics.items())
				},
			}
	return {
		'schema': 'rof_counterfactual_dynamics_episode_bootstrap_v1',
		'status': 'passed',
		'checks': {
			'episode_boundaries_preserved': True,
			'exactly_four_test_episodes': True,
			'identity_shift_for_checkpoint_encoder': True,
			'deterministic_within_episode_action_shuffle': True,
			'bootstrap_unit_is_episode_not_frame': True,
		},
		'bootstrap': {
			'unit': 'paired_test_episode', 'episodes': 4,
			'resamples': bootstrap_resamples, 'base_seed': bootstrap_seed,
		},
		'horizons': horizons,
	}


def encoded_path_separation(
	encoded: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
) -> dict:
	"""Prove non-targeted local/global encoder routes stayed unchanged."""
	comparisons = {
		'O_vs_R_global': ('O', 'R', 'global'),
		'O_vs_M_global': ('O', 'M', 'global'),
		'O_vs_Q_local': ('O', 'Q', 'local'),
		'R_vs_RQ_local': ('R', 'RQ', 'local'),
		'Q_vs_RQ_global': ('Q', 'RQ', 'global'),
	}
	result = {}
	for name, (left_mode, right_mode, view) in comparisons.items():
		left_ids = tuple(encoded[left_mode]['tokens'])
		_require(left_ids == tuple(encoded[right_mode]['tokens']),
			f'{name} episode order differs.')
		maximum = 0.0
		for episode_id in left_ids:
			left = _views(encoded[left_mode]['tokens'][episode_id])[view]
			right = _views(encoded[right_mode]['tokens'][episode_id])[view]
			_require(left.shape == right.shape, f'{name} token shapes differ.')
			maximum = max(maximum, float(np.max(np.abs(left - right))))
		passed = maximum <= 1e-7
		_require(passed, f'{name} violated separated encoder route: max_abs={maximum}.')
		result[name] = {
			'left': left_mode, 'right': right_mode, 'view': view,
			'max_abs': maximum, 'tolerance': 1e-7, 'passed': passed,
		}
	return result


def intervention_effectiveness(
	dataset: ladder.FrozenDataset,
	encoded: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
	query_mean: np.ndarray,
) -> dict:
	"""Reject nominal interventions that did not reach their intended token path."""
	ids = dataset.splits['test']
	original_tokens = _concat(encoded['O']['tokens'], ids)
	views = _views(original_tokens)
	o_max = {
		name: float(np.max(np.abs(value - value))) for name, value in views.items()
	}
	_require(all(value == 0.0 for value in o_max.values()),
		'O perturbation must be exactly zero.')

	r_candidates = 0
	r_expected_changes = 0
	r_actual_changes = 0
	q_candidates = 0
	q_expected_changes = 0
	q_actual_changes = 0
	m_changed = 0
	for episode in dataset.split('test'):
		arrays = episode.arrays
		r_policy = intervene_episode(arrays, 'R', query_mean)
		q_policy = intervene_episode(arrays, 'Q', query_mean)
		m_policy = intervene_episode(arrays, 'M', query_mean)
		frames = arrays['policy_rgb'].reshape(-1, STACK_FRAMES, 3, 64, 64)
		union = arrays['policy_object_mask'].any(axis=1)
		outside = np.broadcast_to(~union[:, :, None], frames.shape)
		r_candidates += int(np.count_nonzero(outside))
		r_expected_changes += int(np.count_nonzero(
			outside & (frames != BACKGROUND_VALUE)
		))
		r_actual_changes += int(np.count_nonzero(
			r_policy['policy_rgb'] != arrays['policy_rgb']
		))
		object_frames = arrays['policy_object'].reshape(
			len(arrays['policy_object']), len(dataset.role_names),
			STACK_FRAMES, FRAME_DIM,
		)
		queries = object_frames[..., :QUERY_DIM]
		mean = np.broadcast_to(query_mean, queries.shape)
		q_candidates += int(queries.size)
		q_expected_changes += int(np.count_nonzero(queries != mean))
		q_actual_changes += int(np.count_nonzero(
			q_policy['policy_object'] != arrays['policy_object']
		))
		m_changed += int(np.count_nonzero(
			m_policy['policy_object_mask'] != arrays['policy_object_mask']
		))
	_require(r_actual_changes == r_expected_changes,
		'R raw RGB change count differs from outside-union non-128 count.')
	_require(q_actual_changes == q_expected_changes,
		'Q raw object change count differs from query-to-mean difference count.')

	def maximum(left_mode: str, right_mode: str, view: str) -> float:
		left = _views(_concat(encoded[left_mode]['tokens'], ids))[view]
		right = _views(_concat(encoded[right_mode]['tokens'], ids))[view]
		return float(np.max(np.abs(left - right)))

	r_local = maximum('O', 'R', 'local')
	q_global = maximum('O', 'Q', 'global')
	m_local = maximum('O', 'M', 'local')
	if r_actual_changes:
		_require(r_local > 1e-7,
			'R changed outside-union RGB but did not change local tokens.')
	if q_actual_changes:
		_require(q_global > 1e-7,
			'Q changed query inputs but did not change global tokens.')
	return {
		'O': {
			'status': 'effective',
			'perturbation_max_abs': o_max,
			'assertion': 'exactly_zero',
		},
		'R': {
			'status': 'effective' if r_actual_changes else 'not_applicable',
			'outside_union_rgb_elements': r_candidates,
			'outside_union_non128_elements': r_expected_changes,
			'actual_rgb_elements_changed': r_actual_changes,
			'local_token_max_abs_vs_O': r_local,
			'assertion': (
				'raw_RGB_and_local_token_changed' if r_actual_changes
				else 'all_outside_union_RGB_was_already_128'
			),
		},
		'M': {
			'status': (
				'effective' if m_changed and m_local > 1e-7
				else 'raw_changed_encoder_path_unchanged' if m_changed
				else 'not_applicable'
			),
			'mask_elements_changed': m_changed,
			'local_token_max_abs_vs_O': m_local,
			'assertion': (
				'mask_and_local_token_changed' if m_changed and m_local > 1e-7
				else 'mask_changed_but_active_encoder_path_did_not'
				if m_changed
				else 'online_and_aligned_GT_masks_were_identical'
			),
		},
		'Q': {
			'status': 'effective' if q_actual_changes else 'not_applicable',
			'query_elements': q_candidates,
			'query_elements_different_from_mean': q_expected_changes,
			'actual_object_elements_changed': q_actual_changes,
			'global_token_max_abs_vs_O': q_global,
			'assertion': (
				'query_input_and_global_token_changed' if q_actual_changes
				else 'all_target_queries_equalled_clean_train_means'
			),
		},
		'RQ': {
			'status': (
				'effective' if (r_actual_changes or q_actual_changes)
				else 'not_applicable'
			),
			'full_token_max_abs_vs_O': maximum('O', 'RQ', 'full'),
			'assertion': 'exact_R_plus_Q_composition',
		},
	}


def _source(dataset: ladder.FrozenDataset) -> Mapping:
	value = dataset.manifest.get('source')
	_require(isinstance(value, Mapping), 'Dataset source provenance is missing.')
	return value


def validate_frozen_binding(
	target: ladder.FrozenDataset, reference: ladder.FrozenDataset,
	runtime_config: Path, checkpoint: Path,
) -> dict:
	"""Bind both manifests to the same exact task/runtime/30k checkpoint."""
	_require(target.task == reference.task, 'Target/reference tasks differ.')
	_require(target.role_names == reference.role_names, 'Target/reference role order differs.')
	_require(_condition(reference) == 'clean', 'Reference dataset must be clean.')
	_require(runtime_config.is_file(), f'Runtime config not found: {runtime_config}')
	_require(checkpoint.is_file(), f'Checkpoint not found: {checkpoint}')
	runtime_sha = _sha256(runtime_config)
	checkpoint_sha = _sha256(checkpoint)
	for name, dataset in (('target', target), ('reference', reference)):
		source = _source(dataset)
		_require(Path(source.get('runtime_config', '')).resolve() == runtime_config,
			f'{name} runtime-config path differs from the frozen source.')
		_require(source.get('runtime_config_sha256') == runtime_sha,
			f'{name} runtime-config hash differs from the frozen source.')
		_require(Path(source.get('checkpoint', '')).resolve() == checkpoint,
			f'{name} checkpoint path differs from the frozen source.')
		_require(source.get('checkpoint_sha256') == checkpoint_sha,
			f'{name} checkpoint hash differs from the frozen source.')
		_require(int(source.get('checkpoint_step', -1)) == 30_000,
			f'{name} source is not the frozen 30k checkpoint.')
	return {
		'same_task': True,
		'same_ordered_roles': True,
		'reference_is_clean': True,
		'runtime_config_exact_path_and_sha256': True,
		'checkpoint_exact_path_and_sha256': True,
		'checkpoint_step_is_30000': True,
	}


def _serializable_ood_reference(reference: Mapping[str, Mapping]) -> dict:
	return {
		name: {
			'radius_95': float(row['radius_95']),
			'radius_99': float(row['radius_99']),
			'train_distance': row['train_distance'],
		}
		for name, row in reference.items()
	}


def evaluate(args) -> dict:
	target = ladder.load_dataset(args.dataset)
	reference = ladder.load_dataset(args.reference_clean_dataset)
	runtime_config = Path(args.runtime_config).resolve()
	checkpoint = Path(args.checkpoint).resolve()
	binding_checks = validate_frozen_binding(
		target, reference, runtime_config, checkpoint
	)
	query_mean = clean_train_query_mean(reference)
	agent = ladder._load_agent(target, runtime_config, checkpoint)
	# Reference O supplies label-free clean-train radii.  Reuse target O only
	# when both resolved manifests are identical; otherwise encode the clean
	# reference independently with the same frozen checkpoint.
	target_encoded = encode_interventions(
		target, agent, query_mean, batch_size=args.batch_size
	)
	path_separation = encoded_path_separation(target_encoded)
	effectiveness = intervention_effectiveness(target, target_encoded, query_mean)
	if reference.manifest_path == target.manifest_path:
		reference_encoded = target_encoded['O']
	else:
		reference_encoded = encode_interventions(
			reference, agent, query_mean, batch_size=args.batch_size, modes=('O',)
		)['O']
	train_tokens = _concat(
		reference_encoded['tokens'], reference.splits['train']
	)
	ood_reference = fit_train_clean_ood_reference(train_tokens)
	test_ids = target.splits['test']
	original_tokens = _concat(target_encoded['O']['tokens'], test_ids)
	interventions = {}
	for mode in INTERVENTIONS:
		tokens = _concat(target_encoded[mode]['tokens'], test_ids)
		interventions[mode] = {
			'description': {
				'O': 'original frozen policy observation',
				'R': 'online-union-exterior RGB set to uint8 128 per stack frame',
				'M': 'online masks replaced by aligned same-role scoring-only GT masks',
				'Q': 'query[0:512] replaced by clean-train role/stack-slot mean',
				'RQ': 'R and Q jointly; online masks/status/geometry unchanged',
			}[mode],
			'token_perturbation_vs_O': token_perturbation(original_tokens, tokens),
			'train_clean_latent_ood': score_ood(tokens, ood_reference),
			'test_latent_variability': latent_variability(
				target_encoded[mode]['flat'], test_ids
			),
			'C_dynamics': stage_c_episode_bootstrap(
				target, target_encoded[mode]['flat'], agent,
				bootstrap_seed=args.bootstrap_seed,
				bootstrap_resamples=args.bootstrap_resamples,
			),
		}
	payload = {
		'format': FORMAT,
		'status': 'rof_counterfactual_input_diagnostic_complete',
		'engineering_pass': True,
		'scientific_complete': True,
		'scientific_complete_semantics': (
			'all requested diagnostic outputs are present; this does not mean an '
			'intervention lies inside frozen-model training support'
		),
		'controller_training_authorized': False,
		'task': target.task,
		'condition': _condition(target),
		'checks': {
			**binding_checks,
			'policy_boundary_exactly_four_keys': True,
			'non_targeted_policy_fields_bit_identical': True,
			'gt_masks_scoring_only_offline': True,
			'episode_boundaries_preserved': True,
			'identity_shift_for_all_encodings': True,
			'frozen_encoder_and_dynamics': True,
			'encoded_non_target_routes_unchanged': all(
				row['passed'] for row in path_separation.values()
			),
			'intervention_effectiveness_checked': set(effectiveness) == set(INTERVENTIONS),
			'exactly_four_target_test_episodes': len(test_ids) == 4,
		},
		'provenance': {
			'target_manifest': str(target.manifest_path),
			'target_manifest_sha256': _sha256(target.manifest_path),
			'reference_clean_manifest': str(reference.manifest_path),
			'reference_clean_manifest_sha256': _sha256(reference.manifest_path),
			'runtime_config': str(runtime_config),
			'runtime_config_sha256': _sha256(runtime_config),
			'checkpoint': str(checkpoint),
			'checkpoint_sha256': _sha256(checkpoint),
			'checkpoint_step': 30_000,
			'target_test_episode_ids': list(test_ids),
			'reference_train_episode_ids': list(reference.splits['train']),
			'query_mean_sha256': _array_sha256(query_mean),
		},
		'intervention_contract': {
			'modes': list(INTERVENTIONS),
			'rgb_background_value_uint8': BACKGROUND_VALUE,
			'query_coordinates_replaced_per_frame': [0, QUERY_DIM],
			'query_mean_scope': 'clean_train_episode_x_role_x_stack_position',
			'gt_stack_alignment': 'episode_local_[max(0,t-2),max(0,t-1),t]',
			'metrics_scope': 'target_test_episodes',
		},
		'encoded_path_separation': path_separation,
		'intervention_effectiveness': effectiveness,
		'reference_train_clean_ood': _serializable_ood_reference(ood_reference),
		'interventions': interventions,
		'interpretation_guard': {
			'improvement': (
				'An improvement after a scoped intervention is one-way evidence that '
				'the intervened input path contributes to the observed failure.'
			),
			'no_improvement': (
				'No improvement cannot exclude that path: the frozen downstream model '
				'was trained on O and may reject an out-of-distribution intervention.'
			),
			'gt_scope': (
				'M changes only the mask channel. Query/status descriptors remain the '
				'original online values and are therefore not synchronized to GT. M '
				'uses simulator masks only for privileged offline scoring and cannot '
				'authorize training or deployment with simulator labels.'
			),
			'ood_scope': (
				'Radii are RMS distance to the original clean-train latent centroid; '
				'they are descriptive support checks, not calibrated probabilities.'
			),
			'r_scope': (
				'R writes every outside-online-union RGB pixel to 128, but the current '
				'ROF encoder gates RGB by a dilated union; only changed RGB in its '
				'context ring can influence local tokens. Evidence is therefore about '
				'context-ring RGB leakage, not every possible background path.'
			),
			'q_scope': (
				'Q removes only the time-varying component of query[0:512] by replacing '
				'it with a clean-train role/stack mean. It does not show that queries '
				'are useless, sufficient, or in distribution for the frozen model.'
			),
		},
	}
	validate_result(payload)
	return payload


def validate_result(payload: Mapping) -> dict:
	_require(payload.get('format') == FORMAT, 'Counterfactual result format mismatch.')
	_require(payload.get('status') == 'rof_counterfactual_input_diagnostic_complete',
		'Counterfactual diagnosis is incomplete.')
	_require(payload.get('engineering_pass') is True, 'Engineering checks did not pass.')
	_require(payload.get('scientific_complete') is True, 'Scientific output is incomplete.')
	_require('diagnostic outputs are present' in str(
		payload.get('scientific_complete_semantics', '')
	), 'scientific_complete semantics are missing.')
	_require(payload.get('controller_training_authorized') is False,
		'Counterfactual diagnostic must never authorize controller training.')
	checks = payload.get('checks')
	_require(isinstance(checks, Mapping) and checks and all(checks.values()),
		'Counterfactual checks are incomplete.')
	contract = payload.get('intervention_contract')
	_require(isinstance(contract, Mapping), 'Intervention contract is missing.')
	_require(tuple(contract.get('modes', ())) == INTERVENTIONS,
		'Intervention order/coverage is incomplete.')
	_require(set(payload.get('interventions', {})) == set(INTERVENTIONS),
		'Counterfactual intervention results are incomplete.')
	separation = payload.get('encoded_path_separation')
	_require(isinstance(separation, Mapping) and separation
		and all(row.get('passed') is True for row in separation.values()),
		'Encoded path-separation checks are incomplete.')
	effectiveness = payload.get('intervention_effectiveness')
	_require(isinstance(effectiveness, Mapping)
		and set(effectiveness) == set(INTERVENTIONS)
		and all(row.get('status') in {
			'effective', 'not_applicable', 'raw_changed_encoder_path_unchanged'
		}
			for row in effectiveness.values()),
		'Intervention effectiveness records are incomplete.')
	_require(effectiveness['O'].get('perturbation_max_abs') == {
		'full': 0.0, 'local': 0.0, 'global': 0.0,
	}, 'O perturbation is not exactly zero.')
	if effectiveness['R']['status'] == 'effective':
		_require(effectiveness['R'].get('actual_rgb_elements_changed', 0) > 0
			and effectiveness['R'].get('local_token_max_abs_vs_O', 0.0) > 1e-7,
			'R was marked effective without raw/local change.')
	else:
		_require(effectiveness['R'].get('actual_rgb_elements_changed') == 0,
			'R was marked not-applicable despite a raw RGB change.')
	if effectiveness['Q']['status'] == 'effective':
		_require(effectiveness['Q'].get('actual_object_elements_changed', 0) > 0
			and effectiveness['Q'].get('global_token_max_abs_vs_O', 0.0) > 1e-7,
			'Q was marked effective without raw/global change.')
	else:
		_require(effectiveness['Q'].get('actual_object_elements_changed') == 0,
			'Q was marked not-applicable despite a raw query change.')
	for mode in INTERVENTIONS:
		row = payload['interventions'][mode]
		_require(set(row['token_perturbation_vs_O']) == {'full', 'local', 'global'},
			f'{mode} token perturbation views are incomplete.')
		_require(set(row['train_clean_latent_ood']) == {'full', 'local', 'global'},
			f'{mode} latent OOD views are incomplete.')
		_require(all(
			'heuristic_support_flag' in value
			for value in row['train_clean_latent_ood'].values()
		), f'{mode} heuristic support flags are incomplete.')
		variability = row.get('test_latent_variability', {})
		_require(variability.get('episodes') == 4
			and isinstance(variability.get('pca_participation_ratio'), (int, float)),
			f'{mode} latent variability diagnostics are incomplete.')
		stage = row['C_dynamics']
		_require(stage.get('status') == 'passed', f'{mode} Stage C did not pass.')
		_require(set(stage.get('horizons', {})) == {'1', '3', '5'},
			f'{mode} Stage C horizons are incomplete.')
		for horizon, horizon_row in stage['horizons'].items():
			_require(len(horizon_row.get('by_episode', {})) == 4,
				f'{mode} horizon {horizon} lacks four episode rows.')
			_require(set(horizon_row.get('paired_episode_bootstrap', {})) == {
				'model_over_persistence', 'model_over_action_shuffle',
				'model_minus_persistence_mse', 'model_minus_action_shuffle_mse',
			}, f'{mode} horizon {horizon} paired bootstrap is incomplete.')
	guard = payload.get('interpretation_guard', {})
	_require(all(name in guard for name in (
		'improvement', 'no_improvement', 'gt_scope', 'ood_scope', 'r_scope', 'q_scope'
	)),
		'Interpretation guards are incomplete.')
	return {'status': 'verified', 'format': FORMAT}


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
	parser.add_argument('--reference-clean-dataset', type=Path, required=True)
	parser.add_argument('--runtime-config', type=Path, required=True)
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--batch-size', type=int, default=256)
	parser.add_argument('--bootstrap-seed', type=int, default=20260913)
	parser.add_argument('--bootstrap-resamples', type=int, default=20_000)
	args = parser.parse_args()
	if args.batch_size < 1:
		parser.error('--batch-size must be positive.')
	if args.bootstrap_resamples < 1_000:
		parser.error('--bootstrap-resamples must be at least 1000.')
	payload = evaluate(args)
	_atomic_json(args.output, payload)
	print(json.dumps({
		'status': payload['status'], 'task': payload['task'],
		'condition': payload['condition'], 'output': str(args.output.resolve()),
	}, sort_keys=True))


if __name__ == '__main__':
	main()
