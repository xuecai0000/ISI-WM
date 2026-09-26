"""Dependency-light causal contracts for the Acrobot GRU observer."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from perception.acrobot_state_observer import (
	AcrobotStateObserver, AcrobotStateObserverConfig, INPUT_DIM,
	load_checkpoint, observer_inputs, save_checkpoint,
)


class AcrobotStateObserverContractTest(unittest.TestCase):
	def setUp(self):
		torch.manual_seed(7)
		self.model = AcrobotStateObserver(
			AcrobotStateObserverConfig(hidden_dim=32), [0., 1.],
		).eval()

	def test_input_contract_and_physical_lengths(self):
		points = torch.zeros(2, 9, 3, 2)
		confidence = torch.ones(2, 9, 3)
		valid = torch.ones(2, 9, 1, dtype=torch.bool)
		actions = torch.zeros(2, 9, 1)
		inputs = observer_inputs(points, confidence, valid, actions)
		self.assertEqual(tuple(inputs.shape), (2, 9, INPUT_DIM))
		output, hidden = self.model(inputs)
		self.assertEqual(tuple(output['points'].shape), (2, 9, 3, 2))
		self.assertEqual(tuple(output['angular_velocity'].shape), (2, 9, 2))
		lengths = torch.linalg.vector_norm(
			output['points'][..., 1:, :] - output['points'][..., :-1, :], dim=-1,
		)
		torch.testing.assert_close(lengths, torch.full_like(lengths, 0.5))
		self.assertEqual(tuple(hidden.shape), (1, 2, 32))

	def test_future_cannot_change_past_and_step_matches_sequence(self):
		inputs = torch.randn(1, 12, INPUT_DIM)
		changed = inputs.clone(); changed[:, 7:] += 100.
		with torch.inference_mode():
			first, _ = self.model(inputs)
			second, _ = self.model(changed)
		torch.testing.assert_close(first['points'][:, :7], second['points'][:, :7])
		hidden, steps = None, []
		with torch.inference_mode():
			for frame in range(inputs.shape[1]):
				value, hidden = self.model.step(inputs[:, frame], hidden)
				steps.append(value['points'])
		torch.testing.assert_close(first['points'], torch.stack(steps, dim=1), atol=1e-6, rtol=1e-6)

	def test_checkpoint_round_trip(self):
		with TemporaryDirectory() as temporary:
			path = Path(temporary) / 'observer.pt'
			save_checkpoint(path, self.model, training={'causal': True})
			loaded, payload = load_checkpoint(path)
			self.assertEqual(payload['format'], 'acrobot_causal_state_observer_v1')
			torch.testing.assert_close(loaded.base_xz, self.model.base_xz)


if __name__ == '__main__':
	unittest.main(verbosity=2)
