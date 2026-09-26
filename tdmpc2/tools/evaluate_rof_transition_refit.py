"""Frozen-latent transition-refit diagnostic for Robust Object Field models.

This is an offline attribution experiment, not controller training.  It loads
the policy observations from one immutable clean causal-probe dataset and one
matching hard dataset, freezes the checkpoint encoder, and fits two small
action-conditioned transition models independently in each source domain. A
clean-source refit uses only clean train/validation episodes; a hard-source
refit uses only hard train/validation episodes. Both are then evaluated on
both untouched test splits, yielding a clean/clean, clean/hard, hard/hard,
hard/clean matrix without mixing domains during fitting or selection.

The comparison answers a deliberately narrow question: can a freshly fitted
transition predict the already-frozen latent substantially better than the
checkpoint transition?  A clean improvement implicates checkpoint transition
parameterisation/optimisation.  Improvement on clean but not hard implicates a
hard-domain transfer, non-Markov latent, or state-visitation problem.  Failure
of both tested refits is not proof that no transition can work.

Simulator state, masks, rewards and returns are never fitting targets here.
Only the four policy observation fields and behaviour actions are consumed.
The output can never authorise controller training.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from tdmpc2.tools import evaluate_rof_causal_ladder as ladder


FORMAT = 'rof_transition_refit_diagnostic_v1'
HORIZONS = (1, 3, 5)
ARCHITECTURES = (
	'exact_direct_1hidden_simnorm',
	'residual_normalized_2hidden',
)
METHODS = ('persistence', 'checkpoint_transition') + ARCHITECTURES
FORBIDDEN_FIT_KEYS = (
	'labels__state', 'labels__gt_role_mask', 'labels__gt_visible',
	'reward', 'done',
)


def _require(condition: bool, message: str) -> None:
	if not condition:
		raise ValueError(message)


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as stream:
		for block in iter(lambda: stream.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def _condition(dataset: ladder.FrozenDataset) -> str:
	value = dataset.manifest.get('condition')
	if value is None:
		value = dataset.manifest.get('collection', {}).get('condition')
	_require(value in {'clean', 'hard'}, 'Dataset condition must be clean or hard.')
	return str(value)


def load_policy_dataset(path: Path | str) -> ladder.FrozenDataset:
	"""Verify immutable shards, then load only policy fields and actions.

	The NPZ file-name table is checked against the causal-dataset contract, but
	privileged labels, reward and done arrays are never materialized.
	"""
	metadata = ladder.validate_manifest(path, load_arrays=False)
	allowed = tuple(ladder.POLICY_KEYS) + ('action',)
	expected_archive_keys = set(
		ladder.POLICY_KEYS + ladder.LABEL_KEYS + ladder.TRANSITION_KEYS + ladder.INDEX_KEYS
	)
	episodes = []
	role_count = len(metadata.role_names)
	for episode in metadata.episodes:
		with np.load(episode.path, allow_pickle=False) as archive:
			_require(set(archive.files) == expected_archive_keys,
				f'Episode {episode.episode_id} archive keys violate the immutable contract.')
			arrays = {name: archive[name] for name in allowed}
		action = arrays['action']
		_require(action.ndim == 2 and action.shape[1] == metadata.action_dim,
			f'Episode {episode.episode_id} action shape is invalid.')
		steps = len(action)
		_require(steps >= max(HORIZONS),
			f'Episode {episode.episode_id} is shorter than the largest horizon.')
		_require(arrays['policy_rgb'].shape == (steps + 1, 9, 64, 64),
			f'Episode {episode.episode_id} policy_rgb shape is invalid.')
		_require(arrays['policy_object'].shape == (steps + 1, role_count, 1770),
			f'Episode {episode.episode_id} policy_object shape is invalid.')
		_require(arrays['policy_object_mask'].shape == (
			steps + 1, role_count, 3, 64, 64
		), f'Episode {episode.episode_id} policy_object_mask shape is invalid.')
		_require(arrays['policy_role_exists'].shape == (steps + 1, role_count),
			f'Episode {episode.episode_id} policy_role_exists shape is invalid.')
		_require(arrays['policy_rgb'].dtype == np.uint8,
			'Policy RGB must be uint8.')
		_require(arrays['policy_object'].dtype == np.float32,
			'Policy object descriptors must be float32.')
		_require(arrays['policy_object_mask'].dtype == np.bool_,
			'Policy object masks must be bool.')
		_require(arrays['policy_role_exists'].dtype == np.float32,
			'Policy role-exists values must be float32.')
		_require(np.issubdtype(action.dtype, np.floating), 'Actions must be floating point.')
		_require(np.isfinite(action).all() and np.isfinite(arrays['policy_object']).all(),
			f'Episode {episode.episode_id} policy/action arrays are non-finite.')
		_require(np.all(arrays['policy_role_exists'] == 1.0),
			'ROF V0 refit forbids padded roles.')
		episodes.append(ladder.Episode(
			episode.episode_id, episode.condition, episode.path, arrays,
		))
	return ladder.FrozenDataset(
		manifest_path=metadata.manifest_path, manifest=metadata.manifest,
		task=metadata.task, role_names=metadata.role_names,
		state_names=metadata.state_names, action_dim=metadata.action_dim,
		episodes=tuple(episodes), splits=metadata.splits,
	)


def _stable_seed(base: int, *parts: object) -> int:
	text = '\x1f'.join([str(int(base)), *(str(part) for part in parts)])
	value = int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8], 'little')
	return int(value % (2 ** 32 - 1))


def validate_dataset_pair(
	clean: ladder.FrozenDataset, hard: ladder.FrozenDataset,
	runtime_config: Path, checkpoint: Path,
) -> dict:
	"""Fail closed unless both datasets bind the exact supplied checkpoint."""
	_require(_condition(clean) == 'clean', 'The fitting dataset must be clean.')
	_require(_condition(hard) == 'hard', 'The transfer dataset must be hard.')
	_require(clean.task == hard.task, 'Clean/hard tasks differ.')
	_require(clean.role_names == hard.role_names, 'Clean/hard role contracts differ.')
	_require(clean.action_dim == hard.action_dim, 'Clean/hard action widths differ.')
	_require(set(clean.splits) == set(hard.splits) == {
		'train', 'validation', 'test',
	}, 'Clean/hard episode split names differ.')
	_require(len(clean.split('test')) >= 2 and len(hard.split('test')) >= 2,
		'At least two held-out episodes per condition are required for bootstrap.')
	runtime_config = runtime_config.resolve()
	checkpoint = checkpoint.resolve()
	_require(runtime_config.is_file(), f'Runtime config not found: {runtime_config}')
	_require(checkpoint.is_file(), f'Checkpoint not found: {checkpoint}')
	runtime_hash = _sha256(runtime_config)
	checkpoint_hash = _sha256(checkpoint)
	steps = []
	for name, dataset in (('clean', clean), ('hard', hard)):
		source = dataset.manifest.get('source')
		_require(isinstance(source, Mapping), f'{name} source provenance is missing.')
		_require(Path(str(source.get('runtime_config', ''))).resolve() == runtime_config,
			f'{name} runtime-config path differs from immutable source.')
		_require(source.get('runtime_config_sha256') == runtime_hash,
			f'{name} runtime-config hash differs from immutable source.')
		_require(Path(str(source.get('checkpoint', ''))).resolve() == checkpoint,
			f'{name} checkpoint path differs from immutable source.')
		_require(source.get('checkpoint_sha256') == checkpoint_hash,
			f'{name} checkpoint hash differs from immutable source.')
		step = source.get('checkpoint_step')
		_require(isinstance(step, int) and not isinstance(step, bool) and step > 0,
			f'{name} checkpoint_step is malformed.')
		steps.append(int(step))
	_require(steps[0] == steps[1], 'Clean/hard checkpoint steps differ.')
	return {
		'exact_task_match': True,
		'exact_role_order_match': True,
		'exact_action_width_match': True,
		'exact_runtime_config_path_and_hash_match': True,
		'exact_checkpoint_path_hash_and_step_match': True,
		'checkpoint_step': steps[0],
	}


def one_step_arrays(
	latents: Mapping[int, np.ndarray], actions: Mapping[int, np.ndarray],
	episode_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Concatenate one-step rows without ever crossing an episode boundary."""
	_require(episode_ids, 'One-step episode selection is empty.')
	current_rows, action_rows, next_rows = [], [], []
	latent_dim = action_dim = None
	for episode_id in episode_ids:
		_require(int(episode_id) in latents and int(episode_id) in actions,
			f'Missing episode {episode_id}.')
		z = np.asarray(latents[int(episode_id)], dtype=np.float32)
		a = np.asarray(actions[int(episode_id)], dtype=np.float32)
		_require(z.ndim == 2 and a.ndim == 2 and len(z) == len(a) + 1,
			f'Episode {episode_id} violates obs T+1 / action T alignment.')
		_require(len(a) > 0 and np.isfinite(z).all() and np.isfinite(a).all(),
			f'Episode {episode_id} is empty or non-finite.')
		latent_dim = z.shape[1] if latent_dim is None else latent_dim
		action_dim = a.shape[1] if action_dim is None else action_dim
		_require(z.shape[1] == latent_dim and a.shape[1] == action_dim,
			'Latent/action widths differ between episodes.')
		current_rows.append(z[:-1])
		action_rows.append(a)
		next_rows.append(z[1:])
	return (
		np.concatenate(current_rows, axis=0),
		np.concatenate(action_rows, axis=0),
		np.concatenate(next_rows, axis=0),
	)


def action_mapping(dataset: ladder.FrozenDataset) -> dict[int, np.ndarray]:
	"""Copy actions only; no scoring label/reward can enter the fit boundary."""
	return {
		episode.episode_id: np.asarray(
			episode.arrays['action'], dtype=np.float32
		).copy()
		for episode in dataset.episodes
	}


@dataclass(frozen=True)
class Normalization:
	z_mean: np.ndarray
	z_scale: np.ndarray
	action_mean: np.ndarray
	action_scale: np.ndarray
	delta_mean: np.ndarray
	delta_scale: np.ndarray


def _mean_scale(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	value = np.asarray(value, dtype=np.float64)
	mean = value.mean(axis=0)
	scale = value.std(axis=0)
	scale[scale < 1e-6] = 1.0
	return mean.astype(np.float32), scale.astype(np.float32)


def fit_normalization(
	latents: Mapping[int, np.ndarray], actions: Mapping[int, np.ndarray],
	train_episode_ids: Sequence[int],
) -> Normalization:
	current, action, next_z = one_step_arrays(latents, actions, train_episode_ids)
	z_mean, z_scale = _mean_scale(np.concatenate([current, next_z], axis=0))
	action_mean, action_scale = _mean_scale(action)
	delta_mean, delta_scale = _mean_scale(next_z - current)
	return Normalization(
		z_mean=z_mean, z_scale=z_scale,
		action_mean=action_mean, action_scale=action_scale,
		delta_mean=delta_mean, delta_scale=delta_scale,
	)


def normalization_summary(value: Normalization) -> dict:
	return {
		'latent_dim': int(len(value.z_mean)),
		'action_dim': int(len(value.action_mean)),
		'z_scale': {
			'mean': float(value.z_scale.mean()),
			'min': float(value.z_scale.min()),
			'max': float(value.z_scale.max()),
		},
		'action_scale': {
			'mean': float(value.action_scale.mean()),
			'min': float(value.action_scale.min()),
			'max': float(value.action_scale.max()),
		},
		'delta_scale': {
			'mean': float(value.delta_scale.mean()),
			'min': float(value.delta_scale.min()),
			'max': float(value.delta_scale.max()),
		},
		'statistics_source': 'fit_domain_train_episodes_only',
	}


def _model_factory(
	architecture: str, latent_dim: int, action_dim: int, *,
	checkpoint_hidden_dim: int, simnorm_dim: int,
	normalization: Normalization,
):
	import torch
	import torch.nn as nn
	_require(architecture in ARCHITECTURES, f'Unknown refit architecture {architecture}.')
	_require(latent_dim % simnorm_dim == 0,
		'Latent width must be divisible by the checkpoint SimNorm width.')
	input_dim = latent_dim + action_dim
	class _SimNormOutput(nn.Module):
		def __init__(self, body):
			super().__init__()
			self.body = body

		def forward(self, value):
			logits = self.body(value)
			shape = logits.shape
			return torch.softmax(
				logits.reshape(*shape[:-1], -1, simnorm_dim), dim=-1
			).reshape(shape)
	class _NormalizedResidual(nn.Module):
		def __init__(self, body):
			super().__init__()
			self.body = body
			self.register_buffer('z_mean', torch.as_tensor(normalization.z_mean))
			self.register_buffer('z_scale', torch.as_tensor(normalization.z_scale))
			self.register_buffer(
				'action_mean', torch.as_tensor(normalization.action_mean)
			)
			self.register_buffer(
				'action_scale', torch.as_tensor(normalization.action_scale)
			)
			self.register_buffer(
				'delta_mean', torch.as_tensor(normalization.delta_mean)
			)
			self.register_buffer(
				'delta_scale', torch.as_tensor(normalization.delta_scale)
			)

		def forward(self, value):
			z = value[..., :latent_dim]
			action = value[..., latent_dim:]
			x = torch.cat([
				(z - self.z_mean) / self.z_scale,
				(action - self.action_mean) / self.action_scale,
			], dim=-1)
			delta = self.body(x) * self.delta_scale + self.delta_mean
			# z is already a SimNorm probability vector, not a logit.  Using
			# log(z) makes a zero residual exactly identity after softmax.
			logits = torch.log(z.clamp_min(1e-8)) + delta
			shape = logits.shape
			return torch.softmax(
				logits.reshape(*shape[:-1], -1, simnorm_dim), dim=-1
			).reshape(shape)
	if architecture == 'exact_direct_1hidden_simnorm':
		body = nn.Sequential(
			nn.Linear(input_dim, checkpoint_hidden_dim), nn.Mish(),
			nn.Linear(checkpoint_hidden_dim, latent_dim),
		)
		return _SimNormOutput(body)
	else:
		body = nn.Sequential(
			nn.Linear(input_dim, latent_dim), nn.LayerNorm(latent_dim), nn.SiLU(),
			nn.Linear(latent_dim, latent_dim), nn.LayerNorm(latent_dim), nn.SiLU(),
			nn.Linear(latent_dim, latent_dim),
		)
		return _NormalizedResidual(body)


def _normalized_blocks(
	current: np.ndarray, action: np.ndarray, next_z: np.ndarray,
	normalization: Normalization,
) -> tuple[np.ndarray, np.ndarray]:
	x = np.concatenate([
		(current - normalization.z_mean) / normalization.z_scale,
		(action - normalization.action_mean) / normalization.action_scale,
	], axis=1)
	y = (
		(next_z - current - normalization.delta_mean)
		/ normalization.delta_scale
	)
	_require(np.isfinite(x).all() and np.isfinite(y).all(),
		'Normalized transition rows are non-finite.')
	return x.astype(np.float32), y.astype(np.float32)


def _evaluate_normalized_loss(model, x: np.ndarray, y: np.ndarray, device, batch_size: int) -> float:
	import torch
	total = 0.0
	with torch.no_grad():
		model.eval()
		for start in range(0, len(x), batch_size):
			stop = min(start + batch_size, len(x))
			xb = torch.as_tensor(x[start:stop], device=device)
			yb = torch.as_tensor(y[start:stop], device=device)
			total += float((model(xb) - yb).square().sum().item())
	return total / (len(x) * y.shape[1])


@dataclass
class FittedTransition:
	architecture: str
	model: object
	normalization: Normalization
	device: object
	simnorm_dim: int
	metadata: Mapping

	def predict(self, z: np.ndarray, action: np.ndarray) -> np.ndarray:
		import torch
		z = np.asarray(z, dtype=np.float32)
		action = np.asarray(action, dtype=np.float32)
		_require(z.ndim == 2 and action.ndim == 2 and len(z) == len(action),
			'Refit prediction arrays are misaligned.')
		x = np.concatenate([z, action], axis=1).astype(np.float32)
		with torch.no_grad():
			self.model.eval()
			result = self.model(torch.as_tensor(x, device=self.device))
			groups = result.reshape(len(result), -1, self.simnorm_dim)
			_require(float(result.min().item()) >= -1e-7,
				'Refit output left the non-negative SimNorm domain.')
			_require(float((groups.sum(dim=-1) - 1.0).abs().max().item()) <= 1e-5,
				'Refit output left the unit-sum SimNorm domain.')
			return result.detach().cpu().numpy().astype(np.float32)


def fit_transition(
	architecture: str, latents: Mapping[int, np.ndarray],
	actions: Mapping[int, np.ndarray], train_ids: Sequence[int],
	validation_ids: Sequence[int], *, device, seed: int, batch_size: int,
	max_epochs: int, patience: int, learning_rate: float, weight_decay: float,
	checkpoint_hidden_dim: int, simnorm_dim: int,
) -> FittedTransition:
	"""Fit one transition; model selection uses same-domain validation only."""
	import torch
	_require(max_epochs >= 2 and patience >= 1 and patience < max_epochs,
		'Invalid early-stopping schedule.')
	_require(batch_size >= 1 and learning_rate > 0.0 and weight_decay >= 0.0,
		'Invalid fitting hyperparameters.')
	normalization = fit_normalization(latents, actions, train_ids)
	train = one_step_arrays(latents, actions, train_ids)
	validation = one_step_arrays(latents, actions, validation_ids)
	x_train = np.concatenate([train[0], train[1]], axis=1).astype(np.float32)
	y_train = train[2].astype(np.float32)
	x_validation = np.concatenate([
		validation[0], validation[1]
	], axis=1).astype(np.float32)
	y_validation = validation[2].astype(np.float32)
	fit_objective = 'raw_next_latent_mse_after_simnorm'
	torch.manual_seed(int(seed))
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(int(seed))
	try:
		torch.use_deterministic_algorithms(True, warn_only=True)
	except TypeError:
		torch.use_deterministic_algorithms(True)
	model = _model_factory(
		architecture, train[0].shape[1], train[1].shape[1],
		checkpoint_hidden_dim=checkpoint_hidden_dim, simnorm_dim=simnorm_dim,
		normalization=normalization,
	).to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
	)
	rng = np.random.default_rng(int(seed))
	best_loss = math.inf
	best_epoch = 0
	best_state = None
	stale = 0
	history = []
	for epoch in range(1, max_epochs + 1):
		model.train()
		permutation = rng.permutation(len(x_train))
		total = 0.0
		for start in range(0, len(permutation), batch_size):
			index = permutation[start:start + batch_size]
			xb = torch.as_tensor(x_train[index], device=device)
			yb = torch.as_tensor(y_train[index], device=device)
			optimizer.zero_grad(set_to_none=True)
			loss = (model(xb) - yb).square().mean()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
			optimizer.step()
			total += float(loss.detach().item()) * len(index)
		validation_loss = _evaluate_normalized_loss(
			model, x_validation, y_validation, device, batch_size
		)
		history.append({
			'epoch': epoch,
			'train_fit_objective_mse': total / len(x_train),
			'validation_fit_objective_mse': validation_loss,
		})
		if validation_loss < best_loss - 1e-8:
			best_loss = validation_loss
			best_epoch = epoch
			best_state = {
				name: value.detach().cpu().clone()
				for name, value in model.state_dict().items()
			}
			stale = 0
		else:
			stale += 1
			if stale >= patience:
				break
	_require(best_state is not None and np.isfinite(best_loss),
		'Refit never produced a finite validation model.')
	model.load_state_dict(best_state)
	model.eval()
	parameter_count = sum(parameter.numel() for parameter in model.parameters())
	runtime = FittedTransition(
		architecture, model, normalization, device, simnorm_dim, {},
	)
	validation_prediction = runtime.predict(validation[0], validation[1])
	validation_raw_mse = float(np.mean(np.square(
		validation_prediction.astype(np.float64) - validation[2].astype(np.float64)
	)))
	metadata = {
		'architecture': architecture,
		'formulation': (
			'exact_checkpoint_direct_mlp_mish_simnorm'
			if architecture == 'exact_direct_1hidden_simnorm'
			else 'alternative_normalized_log_probability_residual_two_hidden_simnorm'
		),
		'exact_checkpoint_architecture': architecture == 'exact_direct_1hidden_simnorm',
		'checkpoint_hidden_dim': int(checkpoint_hidden_dim),
		'simnorm_dim': int(simnorm_dim),
		'fit_objective': fit_objective,
		'parameter_count': int(parameter_count),
		'latent_dim': int(train[0].shape[1]),
		'action_dim': int(train[1].shape[1]),
		'train_samples': int(len(train[0])),
		'validation_samples': int(len(validation[0])),
		'train_episode_ids': [int(value) for value in train_ids],
		'validation_episode_ids': [int(value) for value in validation_ids],
		'test_episodes_seen_during_fit_or_selection': False,
		'best_epoch': int(best_epoch),
		'epochs_ran': int(len(history)),
		'early_stopped': bool(len(history) < max_epochs),
		'best_validation_fit_objective_mse': float(best_loss),
		'validation_raw_next_latent_mse': validation_raw_mse,
		'last_train_fit_objective_mse': float(history[-1]['train_fit_objective_mse']),
		'learning_curve': history,
		'normalization': normalization_summary(normalization),
		'output_contract': 'SimNorm after every one-step prediction and rollout step',
	}
	runtime.metadata = metadata
	return runtime


class CheckpointTransition:
	def __init__(self, agent):
		self.agent = agent

	def predict(self, z: np.ndarray, action: np.ndarray) -> np.ndarray:
		import torch
		with torch.no_grad():
			zt = torch.as_tensor(z, device=self.agent.device, dtype=torch.float32)
			at = torch.as_tensor(action, device=self.agent.device, dtype=torch.float32)
			value = self.agent.model.next(zt, at, None)
		return value.detach().cpu().numpy().astype(np.float32)


def rollout_episode(
	z: np.ndarray, action: np.ndarray, predictor: Callable[[np.ndarray, np.ndarray], np.ndarray],
	*, horizon: int, shuffled_action: np.ndarray,
) -> dict[str, np.ndarray]:
	"""Pure NumPy rollout boundary used by runtime and lightweight tests."""
	z = np.asarray(z, dtype=np.float32)
	action = np.asarray(action, dtype=np.float32)
	shuffled_action = np.asarray(shuffled_action, dtype=np.float32)
	_require(horizon >= 1 and len(z) == len(action) + 1,
		'Rollout violates obs T+1 / action T alignment.')
	_require(action.shape == shuffled_action.shape and action.ndim == 2,
		'Shuffled actions are misaligned.')
	count = len(action) - horizon + 1
	_require(count > 0, 'Rollout horizon exceeds episode length.')
	current = z[:count].copy()
	real = current.copy()
	shuffled = current.copy()
	action_input_differences = []
	for offset in range(horizon):
		real_action = action[offset:offset + count]
		permuted_action = shuffled_action[offset:offset + count]
		action_input_differences.append(
			np.mean(np.square(real_action - permuted_action), axis=1)
		)
		real = np.asarray(predictor(real, real_action), dtype=np.float32)
		shuffled = np.asarray(predictor(shuffled, permuted_action), dtype=np.float32)
		_require(real.shape == current.shape and shuffled.shape == current.shape,
			'Transition predictor changed latent shape.')
	target = z[horizon:horizon + count]
	return {
		'real_error': np.mean(np.square(real - target), axis=1),
		'shuffled_error': np.mean(np.square(shuffled - target), axis=1),
		'persistence_error': np.mean(np.square(current - target), axis=1),
		'action_sensitivity': np.mean(np.square(real - shuffled), axis=1),
		'action_shuffle_input_difference': np.concatenate(action_input_differences),
	}


def bootstrap_episode_mean(values: np.ndarray, *, seed: int, resamples: int) -> dict:
	values = np.asarray(values, dtype=np.float64).reshape(-1)
	_require(len(values) >= 2 and np.isfinite(values).all(),
		'Bootstrap requires at least two finite episode values.')
	_require(resamples >= 1000, 'At least 1000 episode bootstrap resamples are required.')
	rng = np.random.default_rng(int(seed))
	index = rng.integers(0, len(values), size=(resamples, len(values)))
	means = values[index].mean(axis=1)
	return {
		'estimate': float(values.mean()),
		'ci95': [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
		'episodes': int(len(values)), 'resamples': int(resamples), 'seed': int(seed),
	}


def paired_episode_bootstrap(
	candidate: np.ndarray, reference: np.ndarray, *, seed: int, resamples: int,
) -> dict:
	"""Positive improvement means candidate has lower MSE than reference."""
	candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
	reference = np.asarray(reference, dtype=np.float64).reshape(-1)
	_require(candidate.shape == reference.shape and len(candidate) >= 2,
		'Paired bootstrap arrays are misaligned or too short.')
	_require(np.isfinite(candidate).all() and np.isfinite(reference).all(),
		'Paired bootstrap arrays are non-finite.')
	_require(resamples >= 1000, 'At least 1000 episode bootstrap resamples are required.')
	rng = np.random.default_rng(int(seed))
	index = rng.integers(0, len(candidate), size=(resamples, len(candidate)))
	candidate_mean = candidate[index].mean(axis=1)
	reference_mean = reference[index].mean(axis=1)
	absolute = reference_mean - candidate_mean
	relative = 1.0 - candidate_mean / np.maximum(reference_mean, 1e-12)
	return {
		'direction': 'positive_means_candidate_lower_mse',
		'candidate_episode_mean_mse': float(candidate.mean()),
		'reference_episode_mean_mse': float(reference.mean()),
		'absolute_improvement': {
			'estimate': float(reference.mean() - candidate.mean()),
			'ci95': [float(np.quantile(absolute, 0.025)), float(np.quantile(absolute, 0.975))],
		},
		'relative_improvement': {
			'estimate': float(1.0 - candidate.mean() / max(reference.mean(), 1e-12)),
			'ci95': [float(np.quantile(relative, 0.025)), float(np.quantile(relative, 0.975))],
		},
		'episodes': int(len(candidate)), 'resamples': int(resamples), 'seed': int(seed),
	}


def _effective_rank(values: Sequence[np.ndarray]) -> float:
	centered = [value - value.mean(axis=0, keepdims=True) for value in values]
	matrix = np.concatenate(centered, axis=0).astype(np.float64, copy=False)
	denominator = max(len(matrix) - len(centered), 1)
	covariance = matrix.T @ matrix / denominator
	trace = float(np.trace(covariance))
	squared = float(np.square(covariance).sum())
	return trace * trace / squared if squared > 1e-24 else 0.0


def latent_domain_audit(
	latents: Mapping[int, np.ndarray], episode_ids: Sequence[int],
	normalization: Normalization,
) -> dict:
	_require(episode_ids, 'Latent domain audit selection is empty.')
	values = [np.asarray(latents[int(value)], dtype=np.float64) for value in episode_ids]
	_require(all(value.ndim == 2 and len(value) >= 2 for value in values),
		'Latent domain audit episodes are malformed.')
	standardized = [
		(value - normalization.z_mean) / normalization.z_scale for value in values
	]
	delta = [
		(np.diff(value, axis=0) - normalization.delta_mean)
		/ normalization.delta_scale for value in values
	]
	joined = np.concatenate(standardized, axis=0)
	joined_delta = np.concatenate(delta, axis=0)
	mean_offset = joined.mean(axis=0)
	participation_ratio = _effective_rank(values)
	return {
		'episodes': int(len(values)),
		'frames': int(len(joined)),
		'clean_train_normalized_mean_offset_rms': float(np.sqrt(np.mean(mean_offset ** 2))),
		'clean_train_normalized_latent_rms': float(np.sqrt(np.mean(joined ** 2))),
		'clean_train_normalized_delta_rms': float(np.sqrt(np.mean(joined_delta ** 2))),
		'fraction_abs_normalized_latent_gt_5': float(np.mean(np.abs(joined) > 5.0)),
		'within_episode_participation_ratio': float(participation_ratio),
		'participation_ratio_fraction_of_latent_dim': float(
			participation_ratio / max(values[0].shape[1], 1)
		),
	}


def evaluate_condition(
	dataset: ladder.FrozenDataset, latents: Mapping[int, np.ndarray],
	predictors: Mapping[str, object], *, bootstrap_seed: int, bootstrap_resamples: int,
) -> dict:
	"""Evaluate every method on held-out episodes with paired episode CIs."""
	_require(set(predictors) == set(METHODS) - {'persistence'},
		'Predictor set is incomplete.')
	test_episodes = dataset.split('test')
	_require(len(test_episodes) >= 2, 'At least two test episodes are required.')
	condition = _condition(dataset)
	horizons = {}
	for horizon in HORIZONS:
		per_method: dict[str, list[dict]] = {name: [] for name in METHODS}
		pooled: dict[str, dict[str, list[np.ndarray]]] = {
			name: {'real': [], 'shuffle': [], 'sensitivity': [], 'action_difference': []}
			for name in METHODS if name != 'persistence'
		}
		persistence_pooled = []
		for episode in test_episodes:
			z = np.asarray(latents[episode.episode_id], dtype=np.float32)
			action = np.asarray(episode.arrays['action'], dtype=np.float32)
			rng = np.random.default_rng(_stable_seed(
				bootstrap_seed, 'shuffle', condition, episode.episode_id,
			))
			shuffled_action = action[rng.permutation(len(action))]
			persistence = np.mean(np.square(
				z[:len(action) - horizon + 1]
				- z[horizon:horizon + len(action) - horizon + 1]
			), axis=1)
			persistence_pooled.append(persistence)
			per_method['persistence'].append({
				'episode_id': int(episode.episode_id),
				'samples': int(len(persistence)), 'mse': float(persistence.mean()),
			})
			for name, transition in predictors.items():
				row = rollout_episode(
					z, action, transition.predict, horizon=horizon,
					shuffled_action=shuffled_action,
				)
				pooled[name]['real'].append(row['real_error'])
				pooled[name]['shuffle'].append(row['shuffled_error'])
				pooled[name]['sensitivity'].append(row['action_sensitivity'])
				pooled[name]['action_difference'].append(
					row['action_shuffle_input_difference']
				)
				real_mse = float(row['real_error'].mean())
				shuffle_mse = float(row['shuffled_error'].mean())
				persistence_mse = float(row['persistence_error'].mean())
				per_method[name].append({
					'episode_id': int(episode.episode_id),
					'samples': int(len(row['real_error'])),
					'real_action_mse': real_mse,
					'shuffled_action_mse': shuffle_mse,
					'persistence_mse': persistence_mse,
					'real_over_persistence': real_mse / max(persistence_mse, 1e-12),
					'real_over_shuffled_action': real_mse / max(shuffle_mse, 1e-12),
					'action_sensitivity_mse': float(row['action_sensitivity'].mean()),
					'action_shuffle_input_mse': float(
						row['action_shuffle_input_difference'].mean()
					),
				})
		methods = {
			'persistence': {
				'pooled_frame_mse': float(np.concatenate(persistence_pooled).mean()),
				'by_episode': per_method['persistence'],
				'episode_mean_mse_bootstrap': bootstrap_episode_mean(
					np.asarray([row['mse'] for row in per_method['persistence']]),
					seed=_stable_seed(bootstrap_seed, condition, horizon, 'persistence'),
					resamples=bootstrap_resamples,
				),
			}
		}
		persistence_episode = np.asarray([
			row['mse'] for row in per_method['persistence']
		])
		for name in predictors:
			rows = per_method[name]
			real_episode = np.asarray([row['real_action_mse'] for row in rows])
			shuffle_episode = np.asarray([row['shuffled_action_mse'] for row in rows])
			methods[name] = {
				'pooled_frames': {
					'samples': int(sum(len(value) for value in pooled[name]['real'])),
					'real_action_mse': float(np.concatenate(pooled[name]['real']).mean()),
					'shuffled_action_mse': float(np.concatenate(pooled[name]['shuffle']).mean()),
					'persistence_mse': float(np.concatenate(persistence_pooled).mean()),
					'action_sensitivity_mse': float(
						np.concatenate(pooled[name]['sensitivity']).mean()
					),
					'action_shuffle_input_mse': float(
						np.concatenate(pooled[name]['action_difference']).mean()
					),
				},
				'by_episode': rows,
				'episode_mean_mse_bootstrap': bootstrap_episode_mean(
					real_episode,
					seed=_stable_seed(bootstrap_seed, condition, horizon, name, 'mse'),
					resamples=bootstrap_resamples,
				),
				'paired_vs_persistence': paired_episode_bootstrap(
					real_episode, persistence_episode,
					seed=_stable_seed(bootstrap_seed, condition, horizon, name, 'persistence'),
					resamples=bootstrap_resamples,
				),
				'paired_real_vs_shuffled_action': paired_episode_bootstrap(
					real_episode, shuffle_episode,
					seed=_stable_seed(bootstrap_seed, condition, horizon, name, 'shuffle'),
					resamples=bootstrap_resamples,
				),
			}
		checkpoint_episode = np.asarray([
			row['real_action_mse'] for row in per_method['checkpoint_transition']
		])
		comparisons = {}
		for name in ARCHITECTURES:
			candidate = np.asarray([
				row['real_action_mse'] for row in per_method[name]
			])
			comparisons[f'{name}_vs_checkpoint_transition'] = paired_episode_bootstrap(
				candidate, checkpoint_episode,
				seed=_stable_seed(bootstrap_seed, condition, horizon, name, 'checkpoint'),
				resamples=bootstrap_resamples,
			)
		horizons[str(horizon)] = {
			'methods': methods,
			'paired_refit_comparisons': comparisons,
		}
	return {
		'condition': condition,
		'test_episode_ids': [episode.episode_id for episode in test_episodes],
		'bootstrap': {
			'unit': 'paired_whole_episode', 'resamples': int(bootstrap_resamples),
			'base_seed': int(bootstrap_seed),
			'warning': 'With four test episodes, intervals are exploratory and coarse.',
		},
		'horizons': horizons,
	}


def classify_evidence(evaluation: Mapping, preferred: Mapping[str, str]) -> dict:
	"""Return cautious same-domain and cross-domain transition evidence."""
	_require(set(preferred) == {'clean', 'hard'}, 'Preferred refit map is incomplete.')
	for value in preferred.values():
		_require(value in ARCHITECTURES, 'Preferred architecture is invalid.')
	matrices = {}
	for architecture in ARCHITECTURES:
		matrices[architecture] = {}
		for fit_condition in ('clean', 'hard'):
			matrices[architecture][fit_condition] = {}
			for test_condition in ('clean', 'hard'):
				comparison = evaluation[fit_condition][test_condition]['horizons']['1'][
					'paired_refit_comparisons'
				][f'{architecture}_vs_checkpoint_transition']
				interval = comparison['relative_improvement']['ci95']
				cell = evaluation[fit_condition][test_condition]
				method_horizons = {
					str(horizon): cell['horizons'][str(horizon)]['methods'][architecture]
					for horizon in HORIZONS
				}
				persistence_ratios = {}
				action_gains = {}
				episode_persistence_wins = {}
				for horizon in HORIZONS:
					method = method_horizons[str(horizon)]
					pooled = method['pooled_frames']
					persistence_ratios[str(horizon)] = float(
						pooled['real_action_mse'] / max(pooled['persistence_mse'], 1e-12)
					)
					action_gains[str(horizon)] = float(
						1.0 - pooled['real_action_mse']
						/ max(pooled['shuffled_action_mse'], 1e-12)
					)
					episode_persistence_wins[str(horizon)] = int(sum(
						row['real_action_mse'] < row['persistence_mse']
						for row in method['by_episode']
					))
				scientific_gate = bool(
					interval[0] > 0.0
					and persistence_ratios['1'] <= 0.85
					and persistence_ratios['3'] <= 0.85
					and persistence_ratios['5'] <= 0.95
					and action_gains['1'] >= 0.10
					and action_gains['3'] >= 0.10
					and episode_persistence_wins['1'] >= 3
					and episode_persistence_wins['3'] >= 3
				)
				matrices[architecture][fit_condition][test_condition] = {
					'relative_improvement_vs_checkpoint': comparison['relative_improvement'],
					'bootstrap_lower_bound_positive': bool(interval[0] > 0.0),
					'model_over_persistence': persistence_ratios,
					'real_action_gain_vs_shuffled': action_gains,
					'episode_persistence_wins': episode_persistence_wins,
					'scientific_gate_pass': scientific_gate,
				}
	exact = matrices['exact_direct_1hidden_simnorm']
	strong = matrices['residual_normalized_2hidden']
	exact_clean_same = exact['clean']['clean']['scientific_gate_pass']
	exact_hard_same = exact['hard']['hard']['scientific_gate_pass']
	exact_clean_to_hard = exact['clean']['hard']['scientific_gate_pass']
	strong_clean_same = strong['clean']['clean']['scientific_gate_pass']
	strong_hard_same = strong['hard']['hard']['scientific_gate_pass']
	if exact_clean_same and exact_hard_same and not exact_clean_to_hard:
		label = 'exact_refit_succeeds_in_domain_but_clean_to_hard_transfer_fails'
	elif exact_clean_same and exact_hard_same and exact_clean_to_hard:
		label = 'checkpoint_joint_training_or_optimization_failure_supported'
	elif exact_hard_same and not exact_clean_to_hard:
		label = 'exact_hard_refit_gain_supports_domain_shift_or_visitation'
	elif strong_clean_same or strong_hard_same:
		label = 'strong_only_gain_supports_transition_formulation_or_capacity'
	elif any(
		cell['bootstrap_lower_bound_positive']
		for architecture in matrices.values()
		for fit_cells in architecture.values()
		for cell in fit_cells.values()
	):
		label = 'refit_improves_checkpoint_but_not_action_conditioned_dynamics'
	else:
		label = 'tested_markov_refits_do_not_isolate_transition_failure'
	return {
		'preferred_refit_selected_on': 'own_domain_validation_raw_next_latent_mse',
		'preferred_refit_by_fit_condition': dict(preferred),
		'one_step_domain_matrix_by_architecture': matrices,
		'attribution': label,
		'interpretation_guard': (
			'A refit is scientifically successful only when it improves over the '
			'checkpoint, beats persistence at h1/h3/h5, wins in at least three of four '
			'episodes at h1/h3, and obtains at least ten percent benefit from the real '
			'action at h1/h3. Lower raw MSE alone may be a persistence shortcut. Neither '
			'Markov refit resolves whether history repairs a non-Markov latent.'
		),
	}


def validate_result(payload: Mapping) -> None:
	_require(payload.get('format') == FORMAT, 'Unexpected refit result format.')
	_require(payload.get('status') == 'rof_transition_refit_complete',
		'Refit diagnostic is incomplete.')
	_require(payload.get('engineering_pass') is True, 'Engineering checks did not pass.')
	_require(payload.get('scientific_complete') is True, 'Scientific diagnostic incomplete.')
	_require(payload.get('controller_training_authorized') is False,
		'Refit diagnostic must never authorize controller training.')
	_require(payload.get('policy_training_performed') is False,
		'Policy/controller training is forbidden.')
	protocol = payload.get('protocol', {})
	_require(protocol.get('fit_conditions') == ['clean', 'hard'],
		'Independent clean/hard fit domains are required.')
	_require(protocol.get('selection_split') == 'same_domain_validation_episodes',
		'Model selection must use matching-domain validation episodes.')
	_require(protocol.get('privileged_labels_used') is False,
		'Privileged labels are forbidden in the refit.')
	_require(set(payload.get('training', {})) == {'clean', 'hard'},
		'Both independent fit domains are required.')
	for fit_condition in ('clean', 'hard'):
		_require(set(payload['training'][fit_condition]) == set(ARCHITECTURES),
			f'Both {fit_condition} architectures are required.')
		_require(set(payload.get('evaluation', {}).get(fit_condition, {})) == {
			'clean', 'hard',
		}, f'{fit_condition} cross-domain evaluation is incomplete.')
		for test_condition in ('clean', 'hard'):
			row = payload['evaluation'][fit_condition][test_condition]
			_require(set(row.get('horizons', {})) == {str(value) for value in HORIZONS},
				f'{fit_condition}->{test_condition} horizon set is incomplete.')
			_require(len(row.get('test_episode_ids', [])) >= 2,
				f'{fit_condition}->{test_condition} lacks held-out episodes.')
			for horizon in HORIZONS:
				methods = row['horizons'][str(horizon)].get('methods', {})
				_require(set(methods) == set(METHODS),
					f'{fit_condition}->{test_condition} horizon {horizon} methods incomplete.')
				for method in METHODS:
					_require(len(methods[method].get('by_episode', [])) == len(row['test_episode_ids']),
						f'{fit_condition}->{test_condition} {method} lacks episode rows.')
	json.dumps(payload, allow_nan=False)


def evaluate(args) -> dict:
	clean = load_policy_dataset(args.clean_dataset)
	hard = load_policy_dataset(args.hard_dataset)
	runtime_config = Path(args.runtime_config).resolve()
	checkpoint = Path(args.checkpoint).resolve()
	pair_checks = validate_dataset_pair(clean, hard, runtime_config, checkpoint)
	agent = ladder._load_agent(clean, runtime_config, checkpoint)
	for parameter in agent.model.parameters():
		parameter.requires_grad_(False)
	checkpoint_hidden_dim = int(agent.cfg.get(
		'robust_object_field_dynamics_hidden_dim', 512
	))
	simnorm_dim = int(agent.cfg.get('simnorm_dim', 8))
	checkpoint_transition_parameters = int(sum(
		parameter.numel() for parameter in agent.model._dynamics.parameters()
	))
	clean_latents = ladder._encode_episodes(clean, agent, batch_size=args.encoder_batch_size)
	hard_latents = ladder._encode_episodes(hard, agent, batch_size=args.encoder_batch_size)
	datasets = {'clean': clean, 'hard': hard}
	latents = {'clean': clean_latents, 'hard': hard_latents}
	actions = {name: action_mapping(dataset) for name, dataset in datasets.items()}
	normalizations = {
		name: fit_normalization(latents[name], actions[name], dataset.splits['train'])
		for name, dataset in datasets.items()
	}
	fits = {'clean': {}, 'hard': {}}
	for fit_condition in ('clean', 'hard'):
		for index, architecture in enumerate(ARCHITECTURES):
			fits[fit_condition][architecture] = fit_transition(
				architecture, latents[fit_condition], actions[fit_condition],
				datasets[fit_condition].splits['train'],
				datasets[fit_condition].splits['validation'],
				device=agent.device,
				seed=_stable_seed(args.seed, fit_condition, architecture, index),
				batch_size=args.fit_batch_size, max_epochs=args.max_epochs,
				patience=args.patience, learning_rate=args.learning_rate,
				weight_decay=args.weight_decay,
				checkpoint_hidden_dim=checkpoint_hidden_dim,
				simnorm_dim=simnorm_dim,
			)
		_require(
			fits[fit_condition]['exact_direct_1hidden_simnorm'].metadata[
				'parameter_count'
			] == checkpoint_transition_parameters,
			'Exact refit parameter count differs from checkpoint transition.',
		)
	preferred = {
		fit_condition: min(
			ARCHITECTURES,
			key=lambda name: fits[fit_condition][name].metadata[
				'validation_raw_next_latent_mse'
			],
		)
		for fit_condition in ('clean', 'hard')
	}
	evaluation = {}
	for fit_condition in ('clean', 'hard'):
		predictors = {
			'checkpoint_transition': CheckpointTransition(agent),
			**{name: fits[fit_condition][name] for name in ARCHITECTURES},
		}
		evaluation[fit_condition] = {
			test_condition: evaluate_condition(
				datasets[test_condition], latents[test_condition], predictors,
				bootstrap_seed=_stable_seed(
					args.bootstrap_seed, fit_condition, test_condition,
				),
				bootstrap_resamples=args.bootstrap_resamples,
			)
			for test_condition in ('clean', 'hard')
		}
	domain = {
		fit_condition: {
			'normalization_reference': normalization_summary(
				normalizations[fit_condition]
			),
			'test_domains': {
				test_condition: latent_domain_audit(
					latents[test_condition], datasets[test_condition].splits['test'],
					normalizations[fit_condition],
				)
				for test_condition in ('clean', 'hard')
			},
		}
		for fit_condition in ('clean', 'hard')
	}
	payload = {
		'format': FORMAT,
		'status': 'rof_transition_refit_complete',
		'engineering_pass': True,
		'scientific_complete': True,
		'controller_training_authorized': False,
		'policy_training_performed': False,
		'recommendation': 'review_frozen_latent_transition_attribution_only',
		'task': clean.task,
		'source': {
			'clean_dataset': str(clean.manifest_path),
			'clean_dataset_sha256': _sha256(clean.manifest_path),
			'hard_dataset': str(hard.manifest_path),
			'hard_dataset_sha256': _sha256(hard.manifest_path),
			'runtime_config': str(runtime_config),
			'runtime_config_sha256': _sha256(runtime_config),
			'checkpoint': str(checkpoint),
			'checkpoint_sha256': _sha256(checkpoint),
			'checkpoint_transition_parameters': checkpoint_transition_parameters,
			'checkpoint_dynamics_hidden_dim': checkpoint_hidden_dim,
			'checkpoint_simnorm_dim': simnorm_dim,
			'exact_refit_parameter_count_match': True,
			'checks': pair_checks,
		},
		'protocol': {
			'fit_conditions': ['clean', 'hard'],
			'fit_split': 'whole_same_domain_train_episodes',
			'selection_split': 'same_domain_validation_episodes',
			'final_evaluation_splits': ['clean_test_episodes', 'hard_test_episodes'],
			'horizons': list(HORIZONS),
			'architectures': list(ARCHITECTURES),
			'encoder_frozen': True,
			'checkpoint_transition_frozen': True,
			'controller_actor_value_reward_heads_frozen': True,
			'privileged_labels_used': False,
			'fit_inputs': ['frozen_policy_latent_t', 'behavior_action_t'],
			'fit_target': 'frozen_policy_latent_t_plus_1',
			'forbidden_fit_keys': list(FORBIDDEN_FIT_KEYS),
			'episode_boundary_crossing': False,
			'bootstrap_unit': 'whole_test_episode',
			'seed': int(args.seed),
			'bootstrap_seed': int(args.bootstrap_seed),
			'bootstrap_resamples': int(args.bootstrap_resamples),
		},
		'training': {
			fit_condition: {
				name: {
					**dict(fits[fit_condition][name].metadata),
					'fit_condition': fit_condition,
					'parameter_ratio_vs_checkpoint_transition': float(
						fits[fit_condition][name].metadata['parameter_count']
						/ max(checkpoint_transition_parameters, 1)
					),
				}
				for name in ARCHITECTURES
			}
			for fit_condition in ('clean', 'hard')
		},
		'latent_domain_audit': domain,
		'evaluation': evaluation,
		'attribution': classify_evidence(evaluation, preferred),
		'limitations': [
			'Only two specified MLP refits are tested; failure does not prove impossibility.',
			'Clean and hard refits are independent; neither sees the other domain during fitting.',
			'Action-shuffle evidence is weak when behaviour actions are nearly constant.',
			'This measures frozen-latent prediction, not counterfactual control return.',
			'Episode bootstrap intervals are coarse when the held-out split has four episodes.',
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
	parser.add_argument('--seed', type=int, default=20260913)
	parser.add_argument('--bootstrap-seed', type=int, default=314159)
	parser.add_argument('--bootstrap-resamples', type=int, default=20_000)
	parser.add_argument('--encoder-batch-size', type=int, default=256)
	parser.add_argument('--fit-batch-size', type=int, default=1024)
	parser.add_argument('--max-epochs', type=int, default=80)
	parser.add_argument('--patience', type=int, default=10)
	parser.add_argument('--learning-rate', type=float, default=3e-4)
	parser.add_argument('--weight-decay', type=float, default=1e-5)
	args = parser.parse_args()
	if args.bootstrap_resamples < 1000:
		parser.error('--bootstrap-resamples must be at least 1000.')
	if args.encoder_batch_size < 1 or args.fit_batch_size < 1:
		parser.error('Batch sizes must be positive.')
	if args.max_epochs < 2 or args.patience < 1 or args.patience >= args.max_epochs:
		parser.error('Require max-epochs >= 2 and 1 <= patience < max-epochs.')
	if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
		parser.error('Invalid optimizer hyperparameters.')
	payload = evaluate(args)
	_atomic_json(args.output, payload)
	print('ROF_TRANSITION_REFIT_COMPLETE')
	print(f'ATTRIBUTION={payload["attribution"]["attribution"]}')
	print(f'OUTPUT={args.output.resolve()}')


if __name__ == '__main__':
	main()
