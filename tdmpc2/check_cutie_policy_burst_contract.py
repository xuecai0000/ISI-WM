"""Dependency-light golden contract for fixed policy-side Cutie bursts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tdmpc2 import check_cutie_last_valid_memory_contract as base


module = base.module


def _write_plan(
	directory: Path, *, start: int, length: int, role: str = 'whole_arm',
	canonical: bool = True,
) -> Path:
	payload = {
		'format': module.POLICY_BURST_FORMAT,
		'task': 'reacher-visual-small',
		'roles': ['whole_arm', 'goal'],
		'episodes': 20,
		'decision_steps': 500,
		'frame_dim': 590,
		'stack_frames': 3,
		'invalid_encoding': module.POLICY_BURST_INVALID_ENCODING,
		'events': [
			{
				'episode_index': episode,
				'role': role,
				'start_decision_step': start,
				'length': length,
			}
			for episode in range(20)
		],
	}
	path = directory / 'burst.json'
	if canonical:
		text = json.dumps(
			payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
			allow_nan=False,
		) + '\n'
	else:
		text = json.dumps(payload, indent=2) + '\n'
	with path.open('w', encoding='utf-8', newline='\n') as file:
		file.write(text)
	return path


def _cfg(path: Path, *, memory: bool):
	cfg = base._cfg(memory)
	cfg['cutie_object_policy_burst_plan'] = str(path)
	return cfg


def _latest(observation):
	return base._parts(observation)[-1]


class CutiePolicyBurstContract(unittest.TestCase):
	def test_00_default_is_off_and_noncanonical_plan_is_rejected(self):
		config = (base.ROOT / 'tdmpc2' / 'config.yaml').read_text(encoding='utf-8')
		self.assertIn('cutie_object_policy_burst_plan: null', config)
		with tempfile.TemporaryDirectory() as temporary:
			path = _write_plan(Path(temporary), start=2, length=2, canonical=False)
			with self.assertRaisesRegex(ValueError, 'canonical'):
				module.CutieObjectWrapper(
					base._FakeEnv(), _cfg(path, memory=False),
					_client=base._FakeClient([
						base._feature((1, 10), (True, True))
					]),
				)

	def test_01_hard_zero_changes_only_latest_target_and_deque_shifts(self):
		features = [
			base._feature((value, 100 + value), (True, True))
			for value in range(1, 8)
		]
		with tempfile.TemporaryDirectory() as temporary:
			path = _write_plan(Path(temporary), start=2, length=2)
			wrapper = module.CutieObjectWrapper(
				base._FakeEnv(), _cfg(path, memory=False),
				_client=base._FakeClient(features),
			)
			observations = [wrapper.reset()]
			for _ in range(6):
				observations.append(wrapper.step(None)[0])

		parts2 = base._parts(observations[2])
		parts3 = base._parts(observations[3])
		parts4 = base._parts(observations[4])
		invalid = np.zeros(module.FRAME_FEATURE_DIM, dtype=np.float32)
		invalid[module.FRAME_CONTENT_DIM:] = [0.0, 1.0, 0.0, 0.0]
		np.testing.assert_array_equal(parts2[0][0], features[0][0])
		np.testing.assert_array_equal(parts2[1][0], features[1][0])
		np.testing.assert_array_equal(parts2[2][0], invalid)
		np.testing.assert_array_equal(parts3[0][0], features[1][0])
		np.testing.assert_array_equal(parts3[1][0], invalid)
		np.testing.assert_array_equal(parts3[2][0], invalid)
		np.testing.assert_array_equal(parts4[0][0], invalid)
		np.testing.assert_array_equal(parts4[1][0], invalid)
		np.testing.assert_array_equal(parts4[2][0], features[4][0])
		# The other role is always the untouched raw latest frame.
		np.testing.assert_array_equal(_latest(observations[2])[1], features[2][1])
		np.testing.assert_array_equal(_latest(observations[3])[1], features[3][1])

		metrics = wrapper.metrics()['policy_observation_intervention']
		self.assertEqual(metrics['applied_role_frames'], 2)
		self.assertEqual(metrics['raw_valid_overwritten'], 2)
		self.assertEqual(metrics['raw_invalid_overlap'], 0)
		self.assertEqual(metrics['hard_zero_content_checks'], 2)
		self.assertEqual(metrics['last_valid_content_checks'], 0)
		self.assertEqual(metrics['exact_invalid_checks'], 2)
		self.assertEqual(metrics['non_target_preserved_checks'], 2)
		self.assertEqual(metrics['raw_source_unchanged_checks'], 2)
		self.assertEqual(metrics['policy_stack_transition_checks'], 7)
		self.assertEqual(metrics['per_episode'][0]['applied_role_frames'], 2)

	def test_02_last_valid_carries_content_and_keeps_invalid_status(self):
		features = [
			base._feature((value, 100 + value), (True, True))
			for value in range(1, 6)
		]
		with tempfile.TemporaryDirectory() as temporary:
			path = _write_plan(Path(temporary), start=2, length=2)
			wrapper = module.CutieObjectWrapper(
				base._FakeEnv(), _cfg(path, memory=True),
				_client=base._FakeClient(features),
			)
			wrapper.reset()
			wrapper.step(None)
			second = wrapper.step(None)[0]
			third = wrapper.step(None)[0]
		for observation in (second, third):
			latest = _latest(observation)
			np.testing.assert_array_equal(
				latest[0, :module.FRAME_CONTENT_DIM],
				features[1][0, :module.FRAME_CONTENT_DIM],
			)
			np.testing.assert_array_equal(
				latest[0, module.FRAME_CONTENT_DIM:],
				np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
			)
		metrics = wrapper.metrics()
		intervention = metrics['policy_observation_intervention']
		self.assertEqual(intervention['memory_substitutions'], 2)
		self.assertEqual(intervention['last_valid_content_checks'], 2)
		self.assertEqual(intervention['without_memory_history'], 0)
		self.assertEqual(metrics['last_valid_memory']['substitutions'], 2)
		# Raw tracker health remains perfect because accounting precedes intervention.
		self.assertEqual(metrics['valid_frame_rate'], 1.0)

	def test_03_t0_has_no_memory_and_reset_never_leaks_history(self):
		features = [
			base._feature((1, 101), (True, True)),
			base._feature((2, 102), (True, True)),
		]
		with tempfile.TemporaryDirectory() as temporary:
			path = _write_plan(Path(temporary), start=0, length=1)
			wrapper = module.CutieObjectWrapper(
				base._FakeEnv(), _cfg(path, memory=True),
				_client=base._FakeClient(features),
			)
			first = wrapper.reset()
			second = wrapper.reset()
		for observation in (first, second):
			for frame in base._parts(observation):
				np.testing.assert_array_equal(
					frame[0, :module.FRAME_CONTENT_DIM],
					np.zeros(module.FRAME_CONTENT_DIM, dtype=np.float32),
				)
		metrics = wrapper.metrics()['policy_observation_intervention']
		self.assertEqual(metrics['without_memory_history'], 2)
		self.assertEqual(metrics['memory_substitutions'], 0)

	def test_04_non_target_natural_invalid_does_not_break_target_accounting(self):
		first = base._feature((1, 101), (True, True))
		mixed = base._feature((2, 102), (True, False))
		with tempfile.TemporaryDirectory() as temporary:
			path = _write_plan(Path(temporary), start=1, length=1)
			wrapper = module.CutieObjectWrapper(
				base._FakeEnv(), _cfg(path, memory=True),
				_client=base._FakeClient([first, mixed]),
			)
			wrapper.reset()
			wrapper.step(None)
		metrics = wrapper.metrics()
		self.assertEqual(metrics['last_valid_memory']['substitutions'], 2)
		self.assertEqual(
			metrics['policy_observation_intervention']['memory_substitutions'], 1
		)

	def test_05_out_of_range_and_hybrid_plans_fail_closed(self):
		with tempfile.TemporaryDirectory() as temporary:
			directory = Path(temporary)
			path = _write_plan(directory, start=499, length=2)
			with self.assertRaisesRegex(ValueError, 'outside decisions'):
				module.CutieObjectWrapper(
					base._FakeEnv(), _cfg(path, memory=False),
					_client=base._FakeClient([base._feature((1, 2), (True, True))]),
				)
			path = _write_plan(directory, start=2, length=2)
			cfg = _cfg(path, memory=False)
			cfg['flat_anchor_mode'] = 'cutie_hybrid'
			with self.assertRaisesRegex(ValueError, 'restricted'):
				module.CutieObjectWrapper(
					base._FakeEnv(), cfg,
					_client=base._FakeClient([base._feature((1, 2), (True, True))]),
				)


if __name__ == '__main__':
	suite = unittest.defaultTestLoader.loadTestsFromTestCase(CutiePolicyBurstContract)
	result = unittest.TextTestRunner(verbosity=2).run(suite)
	if not result.wasSuccessful():
		raise SystemExit(1)
	print('CUTIE_POLICY_BURST_CONTRACT_OK', {
		'format': module.POLICY_BURST_FORMAT,
		'pipeline': 'raw_metrics_then_burst_then_memory_then_stack',
		'default_plan': None,
	})
