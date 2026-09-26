"""Evaluation-only causal latent-state diagnostic for CutieObjectOnly.

This module does not add trainable parameters and is not part of training.  It
tests whether an already learned action-conditioned TD-MPC2 transition can
bridge a missing target-role measurement before investing in a learned filter.
"""

from __future__ import annotations

import copy

import torch


MODES = ('measurement_only', 'dynamics_prior')
NUM_ROLES = 2
FRAME_DIM = 590
STACK_FRAMES = 3
STACKED_DIM = FRAME_DIM * STACK_FRAMES
ROLE_DIM = 64
LATENT_DIM = NUM_ROLES * ROLE_DIM
LATEST_VALID_INDEX = (STACK_FRAMES - 1) * FRAME_DIM + 588


class CutieLatentDynamicsBelief:
	"""Maintain one causal posterior for a frozen ObjectOnly agent.

	Only the configured target role can use the dynamics prior.  Every valid
	measurement immediately replaces that role's prior, while the non-target
	role always comes from the current encoder output.  This deliberately narrow
	diagnostic avoids claiming a general learned belief filter.
	"""

	def __init__(self, agent, mode: str, target_role_index: int = 0):
		if mode not in MODES:
			raise ValueError(f'Unknown latent belief mode {mode!r}.')
		if target_role_index not in range(NUM_ROLES):
			raise ValueError('target_role_index must be 0 or 1.')
		if not getattr(agent, '_cutie_object_only', False):
			raise ValueError('Latent belief requires a CutieObjectOnly agent.')
		if int(agent.cfg.latent_dim) != LATENT_DIM:
			raise ValueError(f'Expected {LATENT_DIM}-D ObjectOnly latent state.')
		if bool(agent.cfg.compile):
			raise ValueError('Latent belief diagnostic requires compile=false.')
		if getattr(agent, 'training', True) or getattr(agent.model, 'training', True):
			raise ValueError('Latent belief diagnostic requires agent.eval().')
		self.agent = agent
		self.mode = mode
		self.target_role_index = target_role_index
		self._role_slice = slice(
			target_role_index * ROLE_DIM,
			(target_role_index + 1) * ROLE_DIM,
		)
		self.reset()

	def reset(self) -> None:
		self._belief = None
		self._previous_action = None
		self._last_valid_target = None
		self._invalid_streak = 0
		self._synthetic_steps_in_streak = 0
		self._natural_steps_in_streak = 0
		self._metrics = {
			'mode': self.mode,
			'target_role_index': self.target_role_index,
			'measurement_steps': 0,
			'target_valid_steps': 0,
			'target_invalid_steps': 0,
			'target_measurement_selected_steps': 0,
			'non_target_measurement_selected_steps': 0,
			'prior_computed_steps': 0,
			'prior_used_steps': 0,
			'synthetic_prior_used_steps': 0,
			'natural_prior_used_steps': 0,
			'invalid_without_prior': 0,
			'max_prior_age': 0,
			'valid_prior_comparisons': 0,
			'valid_prior_role_mse_sum': 0.0,
			'valid_prior_role_cosine_sum': 0.0,
			'valid_persistence_comparisons': 0,
			'valid_persistence_role_mse_sum': 0.0,
			'valid_prior_better_than_persistence_count': 0,
			'reacquisitions': [],
		}

	@staticmethod
	def _object_tensor(obs):
		try:
			keys = set(obs.keys())
		except (AttributeError, TypeError) as exc:
			raise ValueError('Observation must be a keyed TensorDict-like object.') from exc
		if keys != {'object'}:
			raise ValueError(f'Latent belief expects only object, got keys={keys}.')
		objects = obs['object']
		if not isinstance(objects, torch.Tensor):
			raise TypeError('Object observation must be a torch.Tensor.')
		if tuple(objects.shape) != (NUM_ROLES, STACKED_DIM):
			raise ValueError(
				f'Object observation must be {(NUM_ROLES, STACKED_DIM)}, '
				f'got {tuple(objects.shape)}.'
			)
		if objects.dtype != torch.float32 or not torch.isfinite(objects).all():
			raise ValueError('Object observation must be finite float32.')
		return objects

	@torch.no_grad()
	def observe(self, obs, *, synthetic_target_invalid: bool = False):
		"""Fuse the current measurement with a one-step action-conditioned prior."""
		objects = self._object_tensor(obs)
		valid_values = objects[:, LATEST_VALID_INDEX]
		if not torch.all((valid_values == 0) | (valid_values == 1)):
			raise ValueError('Newest role-valid status must be exactly binary.')
		target_valid = bool(valid_values[self.target_role_index].item())
		if synthetic_target_invalid and target_valid:
			raise RuntimeError(
				'Scheduled synthetic target invalidity was not present in policy input.'
			)

		agent_obs = obs.to(self.agent.device, non_blocking=True).unsqueeze(0)
		measurement = self.agent.model.encode(agent_obs, None)
		if tuple(measurement.shape) != (1, LATENT_DIM):
			raise RuntimeError(
				f'Encoder returned {tuple(measurement.shape)}, expected (1, {LATENT_DIM}).'
			)
		if measurement.dtype != torch.float32 or not torch.isfinite(measurement).all():
			raise RuntimeError('Encoder returned a non-finite or non-float32 latent.')
		self._metrics['measurement_steps'] += 1
		self._metrics[
			'target_valid_steps' if target_valid else 'target_invalid_steps'
		] += 1
		self._metrics['non_target_measurement_selected_steps'] += 1

		prior = None
		if self._belief is not None and self.mode == 'dynamics_prior':
			if self._previous_action is None:
				raise RuntimeError('Belief exists without the executed previous action.')
			prior = self.agent.model.next(
				self._belief, self._previous_action, None
			)
			if (
				tuple(prior.shape) != (1, LATENT_DIM)
				or prior.dtype != torch.float32
				or not torch.isfinite(prior).all()
			):
				raise RuntimeError('Dynamics returned an invalid prior latent.')
			self._metrics['prior_computed_steps'] += 1

		prior_mse = prior_cosine = None
		if target_valid and prior is not None:
			actual = measurement[:, self._role_slice]
			predicted = prior[:, self._role_slice]
			prior_mse = torch.mean((predicted - actual) ** 2).item()
			prior_cosine = torch.nn.functional.cosine_similarity(
				predicted, actual, dim=-1
			).mean().item()
			self._metrics['valid_prior_comparisons'] += 1
			self._metrics['valid_prior_role_mse_sum'] += float(prior_mse)
			self._metrics['valid_prior_role_cosine_sum'] += float(prior_cosine)
			if self._last_valid_target is not None:
				persistence_mse = torch.mean(
					(self._last_valid_target - actual) ** 2
				).item()
				self._metrics['valid_persistence_comparisons'] += 1
				self._metrics['valid_persistence_role_mse_sum'] += float(
					persistence_mse
				)
				self._metrics[
					'valid_prior_better_than_persistence_count'
				] += int(prior_mse < persistence_mse)

		# Compare the open-loop prior against the first real measurement after a
		# missing streak before measurement correction is applied.
		if target_valid and self._invalid_streak:
			if prior is not None and self._last_valid_target is not None:
				actual = measurement[:, self._role_slice]
				persistence_mse = torch.mean(
					(self._last_valid_target - actual) ** 2
				).item()
				persistence_cosine = torch.nn.functional.cosine_similarity(
					self._last_valid_target, actual, dim=-1
				).mean().item()
				self._metrics['reacquisitions'].append({
					'invalid_streak': self._invalid_streak,
					'synthetic_steps': self._synthetic_steps_in_streak,
					'natural_steps': self._natural_steps_in_streak,
					'prior_role_mse': float(prior_mse),
					'prior_role_cosine': float(prior_cosine),
					'persistence_role_mse': float(persistence_mse),
					'persistence_role_cosine': float(persistence_cosine),
					'prior_better_than_persistence': prior_mse < persistence_mse,
				})
			self._invalid_streak = 0
			self._synthetic_steps_in_streak = 0
			self._natural_steps_in_streak = 0

		posterior = measurement
		used_prior = False
		if not target_valid:
			self._invalid_streak += 1
			self._metrics['max_prior_age'] = max(
				self._metrics['max_prior_age'], self._invalid_streak
			)
			if synthetic_target_invalid:
				self._synthetic_steps_in_streak += 1
			else:
				self._natural_steps_in_streak += 1
			if self.mode == 'dynamics_prior' and prior is not None:
				posterior = measurement.clone()
				posterior[:, self._role_slice] = prior[:, self._role_slice]
				used_prior = True
				self._metrics['prior_used_steps'] += 1
				key = (
					'synthetic_prior_used_steps'
					if synthetic_target_invalid else 'natural_prior_used_steps'
				)
				self._metrics[key] += 1
			elif self._belief is None:
				self._metrics['invalid_without_prior'] += 1
		else:
			self._last_valid_target = measurement[:, self._role_slice].clone()
		if not used_prior:
			self._metrics['target_measurement_selected_steps'] += 1

		self._belief = posterior.clone()
		return self._belief, {
			'target_valid': target_valid,
			'used_prior': used_prior,
			'synthetic_target_invalid': bool(synthetic_target_invalid),
		}

	def record_executed_action(self, action) -> None:
		"""Record the exact clamped action passed to ``env.step``."""
		if not isinstance(action, torch.Tensor):
			raise TypeError('Executed action must be a torch.Tensor.')
		if tuple(action.shape) != (int(self.agent.cfg.action_dim),):
			raise ValueError(
				f'Executed action has shape {tuple(action.shape)}, expected '
				f'({int(self.agent.cfg.action_dim)},).'
			)
		if action.dtype != torch.float32 or not torch.isfinite(action).all():
			raise ValueError('Executed action must be finite float32.')
		self._previous_action = action.to(
			self.agent.device, non_blocking=True
		).unsqueeze(0).clone()

	@property
	def belief(self):
		if self._belief is None:
			raise RuntimeError('No belief is available before the first observation.')
		return self._belief

	def metrics(self) -> dict:
		payload = copy.deepcopy(self._metrics)
		comparisons = payload['valid_prior_comparisons']
		payload['valid_prior_role_mse_mean'] = (
			payload['valid_prior_role_mse_sum'] / comparisons
			if comparisons else None
		)
		payload['valid_prior_role_cosine_mean'] = (
			payload['valid_prior_role_cosine_sum'] / comparisons
			if comparisons else None
		)
		persistence_comparisons = payload['valid_persistence_comparisons']
		payload['valid_persistence_role_mse_mean'] = (
			payload['valid_persistence_role_mse_sum'] / persistence_comparisons
			if persistence_comparisons else None
		)
		payload.update({
			'pending_invalid_streak': self._invalid_streak,
			'pending_synthetic_steps': self._synthetic_steps_in_streak,
			'pending_natural_steps': self._natural_steps_in_streak,
		})
		return payload
