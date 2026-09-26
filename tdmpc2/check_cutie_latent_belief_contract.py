"""Dependency-light contract for the evaluation-only latent belief core."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tdmpc2.common.cutie_latent_belief import (
	CutieLatentDynamicsBelief,
	LATEST_VALID_INDEX,
	LATENT_DIM,
	STACKED_DIM,
)


class _Obs:
	def __init__(self, objects):
		self._objects = objects

	def keys(self):
		return ('object',)

	def __getitem__(self, key):
		if key != 'object':
			raise KeyError(key)
		return self._objects

	def to(self, device, non_blocking=False):
		return _Obs(self._objects.to(device))

	def unsqueeze(self, dim):
		return _Obs(self._objects.unsqueeze(dim))


class _Model:
	training = False

	def encode(self, obs, task):
		assert task is None
		objects = obs['object']
		return torch.cat((objects[:, 0, :64], objects[:, 1, :64]), dim=-1)

	def next(self, z, action, task):
		assert task is None
		# The first executed action component is an easily checked causal sentinel.
		return z + action[:, :1]


class _Agent:
	def __init__(self):
		self.training = False
		self._cutie_object_only = True
		self.cfg = SimpleNamespace(latent_dim=LATENT_DIM, compile=False, action_dim=2)
		self.device = torch.device('cpu')
		self.model = _Model()


def _obs(role0, role1, valid0=1, valid1=1):
	objects = torch.zeros(2, STACKED_DIM, dtype=torch.float32)
	objects[0, :64] = float(role0)
	objects[1, :64] = float(role1)
	objects[0, LATEST_VALID_INDEX] = float(valid0)
	objects[1, LATEST_VALID_INDEX] = float(valid1)
	return _Obs(objects)


def _check(name, fn):
	fn()
	print('PASS', name)


def main():
	agent = _Agent()

	def measurement_identity():
		core = CutieLatentDynamicsBelief(agent, 'measurement_only')
		z, info = core.observe(_obs(2, 7))
		assert not info['used_prior']
		assert torch.equal(z[:, :64], torch.full((1, 64), 2.))
		assert torch.equal(z[:, 64:], torch.full((1, 64), 7.))
		core.record_executed_action(torch.tensor([3., -9.]))
		z, info = core.observe(_obs(4, 8, valid0=0), synthetic_target_invalid=True)
		assert not info['used_prior']
		assert torch.equal(z[:, :64], torch.full((1, 64), 4.))

	def action_conditioned_target_only():
		core = CutieLatentDynamicsBelief(agent, 'dynamics_prior')
		core.observe(_obs(2, 7))
		core.record_executed_action(torch.tensor([3., -9.]))
		z, info = core.observe(
			_obs(0, 11, valid0=0, valid1=0), synthetic_target_invalid=True
		)
		assert info['used_prior']
		# Target uses 2+3 prior; goal is the current measurement 11, not prior 10.
		assert torch.equal(z[:, :64], torch.full((1, 64), 5.))
		assert torch.equal(z[:, 64:], torch.full((1, 64), 11.))
		metrics = core.metrics()
		assert metrics['prior_used_steps'] == 1
		assert metrics['synthetic_prior_used_steps'] == 1

	def reacquisition_corrects_and_scores():
		core = CutieLatentDynamicsBelief(agent, 'dynamics_prior')
		core.observe(_obs(2, 7))
		core.record_executed_action(torch.tensor([1., 0.]))
		core.observe(_obs(0, 8, valid0=0), synthetic_target_invalid=True)
		core.record_executed_action(torch.tensor([1., 0.]))
		z, info = core.observe(_obs(4, 9, valid0=1))
		assert not info['used_prior']
		assert torch.equal(z[:, :64], torch.full((1, 64), 4.))
		records = core.metrics()['reacquisitions']
		assert len(records) == 1 and records[0]['invalid_streak'] == 1
		assert records[0]['synthetic_steps'] == 1
		# Prior predicts 4 exactly; persistence remains at 2.
		assert records[0]['prior_role_mse'] == 0.
		assert records[0]['persistence_role_mse'] == 4.
		assert records[0]['prior_better_than_persistence'] is True

	def invalid_t0_and_reset_isolation():
		core = CutieLatentDynamicsBelief(agent, 'dynamics_prior')
		z, info = core.observe(_obs(0, 7, valid0=0))
		assert not info['used_prior'] and core.metrics()['invalid_without_prior'] == 1
		assert torch.equal(z[:, :64], torch.zeros(1, 64))
		core.record_executed_action(torch.tensor([5., 0.]))
		core.reset()
		z, info = core.observe(_obs(3, 6, valid0=1))
		assert not info['used_prior']
		assert core.metrics()['prior_computed_steps'] == 0
		assert torch.equal(z[:, :64], torch.full((1, 64), 3.))

	def latest_valid_only_and_schedule_guard():
		core = CutieLatentDynamicsBelief(agent, 'dynamics_prior')
		objects = _obs(2, 7)._objects
		objects[0, 588] = 1  # Oldest stack frame says valid.
		objects[0, LATEST_VALID_INDEX] = 0  # Latest frame is the only gate.
		try:
			_, info = core.observe(_Obs(objects), synthetic_target_invalid=False)
		except Exception as exc:
			raise AssertionError('Old valid status must not make latest valid.') from exc
		assert info['target_valid'] is False
		core.reset()
		try:
			core.observe(_obs(2, 7), synthetic_target_invalid=True)
		except RuntimeError:
			pass
		else:
			raise AssertionError('Scheduled burst with valid target must fail closed.')

	def schema_and_action_fail_closed():
		core = CutieLatentDynamicsBelief(agent, 'dynamics_prior')
		bad = torch.zeros(2, STACKED_DIM, dtype=torch.float32)
		bad[0, LATEST_VALID_INDEX] = .5
		bad[1, LATEST_VALID_INDEX] = 1
		try:
			core.observe(_Obs(bad))
		except ValueError:
			pass
		else:
			raise AssertionError('Non-binary valid status must be rejected.')
		try:
			core.record_executed_action(torch.zeros(3))
		except ValueError:
			pass
		else:
			raise AssertionError('Wrong action shape must be rejected.')

	for name, fn in (
		('measurement_identity', measurement_identity),
		('action_conditioned_target_only', action_conditioned_target_only),
		('reacquisition_corrects_and_scores', reacquisition_corrects_and_scores),
		('invalid_t0_and_reset_isolation', invalid_t0_and_reset_isolation),
		('latest_valid_only_and_schedule_guard', latest_valid_only_and_schedule_guard),
		('schema_and_action_fail_closed', schema_and_action_fail_closed),
	):
		_check(name, fn)
	print('CUTIE_LATENT_BELIEF_CONTRACT_OK 6/6')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
