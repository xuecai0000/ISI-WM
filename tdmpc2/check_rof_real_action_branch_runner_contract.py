"""Static fail-closed contract for the formal real-action branch runner."""

from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / 'portable' / 'run_rof_real_action_branch_preflight.sh'
PROTOCOL = (
	REPO_ROOT / 'research_reports' / 'diagnostics'
	/ 'rof_real_action_branch_protocol_v1.json'
)


class FormalRunnerContract(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.raw = RUNNER.read_bytes()
		cls.text = cls.raw.decode('utf-8')
		cls.protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))

	def test_00_portable_shell_file(self):
		self.assertTrue(self.text.startswith('#!/usr/bin/env bash\n'))
		self.assertNotIn(b'\r\n', self.raw)
		self.assertIn('set -Eeuo pipefail', self.text)

	def test_01_preregistered_collection_is_literal(self):
		for literal in (
			'readonly ROOTS=80', 'readonly TRAIN_ROOTS=50',
			'readonly VALIDATION_ROOTS=10',
			"readonly ANCHOR_STEPS='80,160,240,320,400'",
			"readonly BRANCH_MAGNITUDE='0.8'",
			'--roots "${ROOTS}"', '--train-roots "${TRAIN_ROOTS}"',
			'--validation-roots "${VALIDATION_ROOTS}"',
			'--anchor-steps "${ANCHOR_STEPS}"',
			'--branch-magnitude "${BRANCH_MAGNITUDE}"',
		):
			self.assertIn(literal, self.text)

	def test_02_preregistered_fit_is_literal(self):
		for literal in (
			'readonly HIDDEN_DIM=128',
			"readonly FIT_SEEDS='20260913 20260917 20260923'",
			'readonly BOOTSTRAP_RESAMPLES=20000',
			'local bootstrap_seed=$((seed + 1000003))',
			'--hidden-dim "${HIDDEN_DIM}"',
			'--bootstrap-seed "${bootstrap_seed}"',
			'--bootstrap-resamples "${BOOTSTRAP_RESAMPLES}"',
		):
			self.assertIn(literal, self.text)

	def test_03_protocol_file_matches_runner(self):
		collection = self.protocol['collection']
		fit = self.protocol['fit']
		self.assertEqual(collection['roots_per_task'], 80)
		self.assertEqual(
			collection['root_splits'],
			{'train': 50, 'validation': 10, 'test': 20},
		)
		self.assertEqual(collection['anchor_steps'], [80, 160, 240, 320, 400])
		self.assertEqual(collection['branch_action_magnitude'], 0.8)
		self.assertEqual(fit['fit_seeds'], [20260913, 20260917, 20260923])
		self.assertEqual(fit['hidden_dim'], 128)
		self.assertEqual(fit['bootstrap_resamples'], 20000)
		self.assertEqual(fit['bootstrap_seed_rule'], 'fit_seed_plus_1000003')
		self.assertFalse(
			self.protocol['decision_rule']['automatic_controller_training_authorization']
		)

	def test_04_only_diagnostic_modules_are_invoked(self):
		modules = set(re.findall(r'-m\s+(tdmpc2\.[A-Za-z0-9_.]+)', self.text))
		self.assertEqual(modules, {
			'tdmpc2.check_rof_real_action_branch_contract',
			'tdmpc2.check_rof_real_action_branch_evaluator_contract',
			'tdmpc2.check_rof_real_action_branch_aggregate_contract',
			'tdmpc2.check_rof_real_action_branch_runner_contract',
			'tdmpc2.tools.collect_rof_real_action_branches',
			'tdmpc2.tools.evaluate_rof_real_action_branches',
			'tdmpc2.tools.aggregate_rof_real_action_branches',
		})
		for forbidden in (
			'-m tdmpc2.train', 'tdmpc2/train.py',
			'run_controller', 'controller_pilot.sh',
		):
			self.assertNotIn(forbidden, self.text)

	def test_05_import_dependency_closure_is_required(self):
		for path in (
			'tdmpc2/tools/evaluate_rof_action_identifiable.py',
			'tdmpc2/tools/evaluate_rof_normalized_delta.py',
			'tdmpc2/tools/evaluate_rof_transition_refit.py',
			'tdmpc2/tools/evaluate_rof_causal_ladder.py',
			'tdmpc2/check_rof_real_action_branch_aggregate_contract.py',
		):
			self.assertIn(path, self.text)

	def test_06_dataset_and_campaign_publication_are_separate(self):
		self.assertIn('DATASET_STAGE="${DATASET_ROOT}.incomplete"', self.text)
		self.assertIn('CAMPAIGN_STAGE="${OUTPUT_ROOT}.incomplete"', self.text)
		self.assertIn("('dataset', dataset, 'output', output)", self.text)
		self.assertIn(
			'mv -T -- "${CAMPAIGN_STAGE}" "${OUTPUT_ROOT}"', self.text,
		)
		self.assertNotIn('mv -T -- "${DATASET_ROOT}"', self.text)

	def test_07_reuse_is_fail_closed(self):
		for literal in (
			'REUSE_DATASET="${REUSE_DATASET:-0}"',
			'REUSE_DATASET=1 but no regular published dataset exists',
			'DATASET_MANIFEST_SHA256="$(validate_dataset)"',
			'Published dataset violates formal protocol',
			'Dataset changed during fixed-seed evaluation.',
			'Dataset changed during aggregation.',
		):
			self.assertIn(literal, self.text)

	def test_08_source_and_bound_inputs_are_revalidated(self):
		self.assertIn('snapshot_inputs bind', self.text)
		self.assertGreaterEqual(self.text.count('snapshot_inputs verify'), 3)
		for name in ('runtime_config', 'checkpoint', 'protocol'):
			self.assertIn(f"('{name}',", self.text)
		self.assertIn('Source/runtime/checkpoint/protocol binding changed', self.text)

	def test_09_three_results_match_aggregator_naming(self):
		self.assertIn('/runs/seed_${seed}.json', self.text)
		self.assertIn('--results-root "${CAMPAIGN_STAGE}/runs"', self.text)
		self.assertIn("for seed in (20260913, 20260917, 20260923):", self.text)

	def test_10_all_embedded_python_compiles(self):
		lines = self.text.splitlines()
		blocks = []
		index = 0
		while index < len(lines):
			if "<<'PY'" not in lines[index]:
				index += 1
				continue
			start = index + 1
			index = start
			while index < len(lines) and lines[index] != 'PY':
				index += 1
			self.assertLess(index, len(lines), f'Unclosed heredoc at line {start}.')
			blocks.append((start + 1, '\n'.join(lines[start:index]) + '\n'))
			index += 1
		self.assertEqual(len(blocks), 7)
		for line, source in blocks:
			compile(source, f'{RUNNER}:heredoc@{line}', 'exec')

	def test_11_final_manifest_cannot_authorize_training(self):
		self.assertIn("'policy_training_performed': False", self.text)
		self.assertIn("'controller_training_authorized': False", self.text)
		self.assertIn("'next_step_requires_independent_human_review': True", self.text)
		self.assertIn("echo 'CONTROLLER_TRAINING_AUTHORIZED=False'", self.text)


if __name__ == '__main__':
	unittest.main()
