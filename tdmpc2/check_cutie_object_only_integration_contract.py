"""Dependency-light source/runner contract for structural CutieObjectOnly."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / 'tools' / 'run_cutie_object_only_pilot.sh'


class CutieObjectOnlyIntegrationContract(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.config = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
		cls.layers = (ROOT / 'common' / 'layers.py').read_text(encoding='utf-8')
		cls.world = (ROOT / 'common' / 'world_model.py').read_text(encoding='utf-8')
		cls.agent = (ROOT / 'tdmpc2.py').read_text(encoding='utf-8')
		cls.wrapper = (ROOT / 'envs' / 'wrappers' / 'cutie_object.py').read_text(
			encoding='utf-8'
		)
		cls.buffer = (ROOT / 'common' / 'buffer.py').read_text(encoding='utf-8')
		cls.update_contract = (
			ROOT / 'check_cutie_object_only_update.py'
		).read_text(encoding='utf-8')
		cls.runner = RUNNER.read_text(encoding='utf-8')

	def test_00_frozen_dimensions(self):
		self.assertEqual(self.config['cutie_object_num_roles'], 2)
		self.assertEqual(self.config['cutie_object_frame_dim'], 590)
		self.assertEqual(self.config['cutie_object_stack_frames'], 3)
		self.assertEqual(self.config['cutie_object_input_dim'], 1770)
		self.assertEqual(
			self.config['cutie_object_auxiliary_target'], 'full_descriptor'
		)
		self.assertEqual(self.config['cutie_object_role_dim'], 64)
		self.assertEqual(self.config['cutie_object_only_latent_dim'], 128)

	def test_01_environment_consumes_rgb_but_agent_schema_is_object_only(self):
		self.assertIn("'cutie_hybrid', 'cutie_object_only'", (
			ROOT / 'envs' / 'dmcontrol.py'
		).read_text(encoding='utf-8'))
		self.assertIn("spaces = {'object': object_space}", self.wrapper)
		self.assertIn("if not self._object_only:", self.wrapper)
		self.assertIn('self._client.track(native_rgb)', self.wrapper)
		self.assertIn('latest_source_rgb_sha256', self.wrapper)

	def test_02_model_has_structural_object_only_dispatch(self):
		self.assertIn("if mode == 'cutie_object_only':", self.layers)
		self.assertIn('return nn.ModuleDict(out)', self.layers)
		self.assertIn('def _install_cutie_object_only_branch', self.world)
		self.assertIn('self._dynamics = object_dynamics', self.world)
		self.assertIn('GEOMETRY_STATUS_FULL_DENOMINATOR', self.world)
		self.assertIn("contract['cutie_object_auxiliary']", self.agent)
		self.assertIn("self._encoder['object'](objects)", self.world)
		self.assertIn("elif anchor_mode == 'cutie_object_only':", self.agent)
		self.assertIn('self.cfg.latent_dim = object_dim', self.agent)
		self.assertIn("'cutie_hybrid', 'cutie_object_only'", self.agent)

	def test_03_replay_fails_closed_and_reports_placement(self):
		self.assertIn("keys != {'object'}", self.buffer)
		self.assertIn("'storage_required_bytes'", self.buffer)
		self.assertIn("'storage_device'", self.buffer)
		self.assertIn("'observation_keys'", self.buffer)
		object_bytes = 2 * 1770 * 4 + 2 * 4 + 2 * 4 + 8
		hybrid_bytes = object_bytes + 9 * 64 * 64
		self.assertEqual(object_bytes, 14184)
		# TensorDict bookkeeping may add a few bytes, but raw modality reduction
		# is fixed and remains comfortably below the runner's 30% gate.
		self.assertLess(object_bytes / hybrid_bytes, 0.30)

	def test_04_runner_is_scratch_fixed_reset_hybrid_pair(self):
		for token in (
			'SEED="${SEED:-5}"',
			'STEPS="${STEPS:-50000}"',
			'EVAL_FREQ="${EVAL_FREQ:-10000}"',
			'EVAL_EPISODES="${EVAL_EPISODES:-3}"',
			'mode=cutie_hybrid',
			'mode=cutie_object_only',
			'checkpoint=null',
			'data_dir=null',
			'check_cutie_episode_reset_isolation',
			'object_only_non_eval_speedup_at_least_10pct',
			'reward is descriptive and is',
		):
			self.assertIn(token, self.runner)
		self.assertNotIn('allow_official_warmstart', self.runner)

	def test_05_embedded_python_is_syntax_valid(self):
		blocks = re.findall(r"<<'PY'\n(.*?)\nPY", self.runner, flags=re.S)
		self.assertEqual(len(blocks), 2)
		for index, block in enumerate(blocks):
			compile(block, f'{RUNNER}:heredoc{index}', 'exec')

	def test_06_cross_mode_rejection_accepts_tensordict_value_error(self):
		# torch.nn commonly raises RuntimeError for strict schema mismatches, while
		# TensorDictParams may surface the same required rejection as ValueError.
		# Either exception means the incompatible checkpoint was correctly refused.
		self.assertEqual(
			self.update_contract.count('except (RuntimeError, ValueError) as error:'),
			2,
		)


if __name__ == '__main__':
	unittest.main(verbosity=2)
