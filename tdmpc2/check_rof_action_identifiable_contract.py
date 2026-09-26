"""Fail-closed contracts for the ROF action-identifiability diagnostic.

The static checks intentionally run without importing Torch.  Synthetic model
checks are enabled when Torch is installed (the normal experiment runtime) and
are skipped in dependency-light source-only environments.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np


def _subject():
	return importlib.import_module('tdmpc2.tools.evaluate_rof_action_identifiable')


def _normalization(subject, *, latent_dim: int = 8, action_dim: int = 2):
	assert latent_dim % 4 == 0
	zeros = np.zeros(latent_dim, dtype=np.float32)
	ones = np.ones(latent_dim, dtype=np.float32)
	action_mean = np.linspace(-0.5, 0.5, action_dim, dtype=np.float32)
	action_scale = np.linspace(1.5, 2.0, action_dim, dtype=np.float32)
	return subject.base.DeltaNormalization(
		u_mean=zeros.copy(), u_scale=ones.copy(),
		action_mean=action_mean, action_scale=action_scale,
		delta_mean=zeros.copy(), delta_scale=ones.copy(),
		horizon_scale={horizon: ones.copy() for horizon in subject.HORIZONS},
		simnorm_dim=4, floor_audit={},
	)


def _zero_sum_history(torch, *, batch: int = 3, latent_dim: int = 8):
	values = torch.arange(
		batch * 3 * latent_dim, dtype=torch.float32,
	).reshape(batch, 3, latent_dim) / 100.0
	groups = values.reshape(batch, 3, -1, 4)
	groups = groups - groups.mean(dim=-1, keepdim=True)
	return groups.reshape(batch, 3, latent_dim)


def _force_visible_control_path(torch, model) -> None:
	"""Make the tiny random control path deterministic and visibly nonzero."""
	with torch.no_grad():
		model.control_context.weight.zero_()
		model.control_context.bias.fill_(0.75)
		model.action_embedding.weight.zero_()
		model.action_embedding.weight[:, 0].fill_(1.0)
		model.control_head.weight.zero_()
		pattern = torch.tensor(
			[1.0, -1.0, 2.0, -2.0] * (len(model.u_mean) // 4),
			dtype=model.control_head.weight.dtype,
		)
		model.control_head.weight[:, 0].copy_(pattern)


def _minimal_result(subject):
	training = {
		condition: {
			mode: {
				'train_episode_ids': list(range(12)),
				'validation_episode_ids': list(range(12, 16)),
				'test_episodes_seen_during_fit_or_selection': False,
				'converged': True,
			}
			for mode in subject.ACTION_MODES
		}
		for condition in subject.FIT_CONDITIONS
	}
	cell = {
		'action_shuffle_semantics': (
			'evaluation_only_observational_input_ablation_not_do_action_counterfactual'
		),
		'test_episode_ids': list(range(16, 20)),
		'branch_rms_by_episode': [],
	}
	return {
		'format': subject.FORMAT, 'status': subject.STATUS,
		'engineering_pass': True, 'scientific_complete': False,
		'controller_training_authorized': False,
		'causal_claim_authorized': False,
		'policy_training_performed': False,
		'privileged_targets_used': False,
		'simulator_state_used': False,
		'protocol': {
			'action_shuffle_training_use': 'forbidden',
			'action_shuffle_evaluation_semantics': (
				'observational_ablation_not_causal_counterfactual'
			),
			'episode_split_counts': dict(subject.SPLIT_COUNTS),
		},
		'training': training,
		'evaluation': {
			fit: {
				test: deepcopy(cell) for test in subject.FIT_CONDITIONS
			}
			for fit in subject.FIT_CONDITIONS
		},
		'gates': {
			'true_same_state_intervention_required_before_controller_training': True,
		},
	}


def _called_functions_by_scope(module_source: str, attribute: str) -> list[str]:
	"""Return scopes containing calls whose terminal attribute/name matches."""
	tree = ast.parse(module_source)
	result: list[str] = []

	class Visitor(ast.NodeVisitor):
		def __init__(self):
			self.scope: list[str] = []

		def visit_FunctionDef(self, node):
			self.scope.append(node.name)
			self.generic_visit(node)
			self.scope.pop()

		visit_AsyncFunctionDef = visit_FunctionDef

		def visit_Call(self, node):
			name = None
			if isinstance(node.func, ast.Attribute):
				name = node.func.attr
			elif isinstance(node.func, ast.Name):
				name = node.func.id
			if name == attribute:
				result.append(self.scope[-1] if self.scope else '<module>')
			self.generic_visit(node)

	Visitor().visit(tree)
	return result


TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


class ActionIdentifiableContract(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.subject = _subject()
		cls.source = Path(cls.subject.__file__).read_text(encoding='utf-8')

	def test_00_import_is_dependency_light_and_protocol_is_fixed(self):
		prefix = self.source.split('def _model_factory', 1)[0]
		self.assertNotIn('\nimport torch', prefix)
		self.assertEqual(self.subject.HORIZONS, (1, 3, 5))
		self.assertEqual(self.subject.HISTORY_LENGTH, 3)
		self.assertEqual(self.subject.SPLIT_COUNTS, {
			'train': 12, 'validation': 4, 'test': 4,
		})
		self.assertEqual(set(self.subject.ACTION_MODES), {
			'action_aware', 'actionless',
		})
		self.assertEqual(
			self.subject.PREREGISTERED_GATES['interpretation'],
			'incremental_action_predictive_information_not_causal_control',
		)

	@unittest.skipUnless(TORCH_AVAILABLE, 'Torch is unavailable in this runtime.')
	def test_01_action_branch_is_exactly_zero_at_fit_mean(self):
		import torch

		normalization = _normalization(self.subject)
		model = self.subject._model_factory(
			'action_aware', normalization, hidden_dim=12,
		)
		_force_visible_control_path(torch, model)
		history = _zero_sum_history(torch)
		action = torch.as_tensor(normalization.action_mean).reshape(1, 1, -1)
		action = action.repeat(len(history), 3, 1)
		_, raw_control = model.decompose(history, action)
		_, branches = model.step(history, action)
		torch.testing.assert_close(raw_control, torch.zeros_like(raw_control), rtol=0, atol=0)
		torch.testing.assert_close(
			branches['control_delta'], torch.zeros_like(branches['control_delta']),
			rtol=0, atol=0,
		)

	@unittest.skipUnless(TORCH_AVAILABLE, 'Torch is unavailable in this runtime.')
	def test_02_action_changes_projected_control_and_projection_is_linear(self):
		import torch

		normalization = _normalization(self.subject)
		model = self.subject._model_factory(
			'action_aware', normalization, hidden_dim=12,
		)
		_force_visible_control_path(torch, model)
		history = _zero_sum_history(torch)
		mean = torch.as_tensor(normalization.action_mean).reshape(1, 1, -1)
		first = mean.repeat(len(history), 3, 1)
		second = first.clone()
		second[:, -1, 0] += normalization.action_scale[0]
		next_mean, mean_branches = model.step(history, first)
		next_changed, branches = model.step(history, second)
		self.assertGreater(float(branches['control_delta'].abs().max()), 0.0)
		self.assertFalse(torch.equal(next_mean, next_changed))
		grouped = branches['control_delta'].reshape(len(history), -1, 4)
		torch.testing.assert_close(
			grouped.sum(dim=-1), torch.zeros_like(grouped.sum(dim=-1)),
			rtol=0, atol=1e-6,
		)
		torch.testing.assert_close(
			branches['projection_linearity_error'],
			torch.zeros_like(branches['projection_linearity_error']),
			rtol=0, atol=0,
		)
		predicted_delta = next_changed - history[:, -1]
		torch.testing.assert_close(
			predicted_delta,
			branches['reference_delta'] + branches['control_delta'],
			rtol=1e-6, atol=1e-6,
		)
		self.assertEqual(float(mean_branches['control_delta'].abs().max()), 0.0)

	@unittest.skipUnless(TORCH_AVAILABLE, 'Torch is unavailable in this runtime.')
	def test_03_actionless_rollout_is_action_invariant_but_residual_is_active(self):
		import torch

		normalization = _normalization(self.subject)
		model = self.subject._model_factory(
			'actionless', normalization, hidden_dim=12,
		)
		_force_visible_control_path(torch, model)
		history = _zero_sum_history(torch)
		actions_a = torch.zeros(len(history), 7, 2)
		actions_b = torch.full_like(actions_a, 1e6)
		outputs_a, branches_a = self.subject._rollout(model, history, actions_a, 5)
		outputs_b, branches_b = self.subject._rollout(model, history, actions_b, 5)
		for horizon in self.subject.HORIZONS:
			torch.testing.assert_close(outputs_a[horizon], outputs_b[horizon])
		self.assertGreater(float(branches_a[0]['control_delta'].abs().max()), 0.0)
		torch.testing.assert_close(
			branches_a[0]['control_delta'], branches_b[0]['control_delta'],
		)

	@unittest.skipUnless(TORCH_AVAILABLE, 'Torch is unavailable in this runtime.')
	def test_04_rollout_is_open_loop_and_never_teacher_forces_future_truth(self):
		import torch

		seen = []

		class Spy:
			def step(self, history, action_history):
				seen.append(history.detach().clone())
				next_u = history[:, -1] + 1.0
				zero = torch.zeros_like(next_u)
				return next_u, {
					'reference_delta': zero, 'control_delta': zero,
					'projection_correction': zero.reshape(len(zero), -1, 1),
					'projection_linearity_error': zero.reshape(len(zero), -1, 1),
				}

		history = torch.tensor([[[0.0], [10.0], [20.0]]])
		actions = torch.zeros(1, 7, 1)
		outputs, _ = self.subject._rollout(Spy(), history, actions, 5)
		self.assertEqual(len(seen), 5)
		torch.testing.assert_close(seen[1][0, :, 0], torch.tensor([10.0, 20.0, 21.0]))
		torch.testing.assert_close(outputs[3], torch.tensor([[23.0]]))
		torch.testing.assert_close(outputs[5], torch.tensor([[25.0]]))
		self.assertEqual(
			list(inspect.signature(self.subject._rollout).parameters),
			['model', 'initial_history', 'action_window', 'max_horizon'],
		)

	@unittest.skipUnless(TORCH_AVAILABLE, 'Torch is unavailable in this runtime.')
	def test_05_inverse_probe_exposes_full_and_context_only_predictions(self):
		import torch

		normalization = _normalization(self.subject, action_dim=3)
		model = self.subject._model_factory(
			'action_aware', normalization, hidden_dim=12,
		)
		history = _zero_sum_history(torch)
		observed_next = history[:, -1] + 0.01
		outputs = model.inverse_action(history, observed_next)
		self.assertEqual(set(outputs), {
			'with_observed_innovation', 'context_only',
		})
		for value in outputs.values():
			self.assertEqual(tuple(value.shape), (len(history), 3))

	def test_06_shuffled_actions_are_evaluation_only_by_ast(self):
		calls = _called_functions_by_scope(self.source, 'shuffled_future_rows')
		self.assertEqual(calls, ['evaluate_condition'])
		for function in (self.subject.fit_transition, self.subject._objective):
			function_tree = ast.parse(inspect.getsource(function))
			call_names = []
			for node in ast.walk(function_tree):
				if isinstance(node, ast.Call):
					if isinstance(node.func, ast.Attribute):
						call_names.append(node.func.attr)
					elif isinstance(node.func, ast.Name):
						call_names.append(node.func.id)
			self.assertNotIn('shuffled_future_rows', call_names)
		self.assertIn(
			"shuffle is evaluation-only", inspect.getsource(self.subject.evaluate_condition),
		)

	def test_07_no_controller_import_or_training_and_encoder_is_frozen(self):
		tree = ast.parse(self.source)
		imports = []
		for node in ast.walk(tree):
			if isinstance(node, ast.Import):
				imports.extend(alias.name for alias in node.names)
			elif isinstance(node, ast.ImportFrom):
				imports.append(node.module or '')
		for forbidden in (
			'tdmpc2.tdmpc2', 'tdmpc2.trainer', 'tdmpc2.train',
		):
			self.assertNotIn(forbidden, imports)
		evaluate_source = inspect.getsource(self.subject.evaluate)
		self.assertIn('parameter.requires_grad_(False)', evaluate_source)
		self.assertIn('agent.eval()', evaluate_source)
		self.assertNotIn('agent.update(', evaluate_source)
		self.assertNotIn('agent.train(', evaluate_source)
		self.assertNotIn('controller_optimizer', self.source)

	def test_08_dataset_policy_path_is_action_plus_frozen_latent_encoder_only(self):
		# The evaluator directly consumes only logged behaviour actions.  Observation
		# arrays stay behind the already-audited frozen ROF encoder boundary.
		array_reads = re.findall(
			r"episode\.arrays\[['\"]([^'\"]+)", self.source,
		)
		self.assertEqual(set(array_reads), {'action'})
		evaluate_source = inspect.getsource(self.subject.evaluate)
		self.assertIn('base.ladder._encode_episodes(', evaluate_source)
		self.assertIn('base.action_mapping(dataset)', evaluate_source)
		for forbidden in (
			"episode.arrays['reward']", "episode.arrays['done']",
			"episode.arrays['labels__state']", "episode.arrays['labels__gt_role_mask']",
			'labels__', 'privileged_target',
		):
			if forbidden == 'privileged_target':
				# Schema flags are allowed; data reads/call arguments are not.
				continue
			self.assertNotIn(forbidden, self.source)
		self.assertNotIn("['reward']", evaluate_source)
		self.assertNotIn("['done']", evaluate_source)

	def test_09_complete_fail_closed_result_schema_passes(self):
		payload = _minimal_result(self.subject)
		self.subject.validate_result(payload)
		json.dumps(payload, allow_nan=False)

	def test_10_result_rejects_controller_authorization_and_privileged_targets(self):
		payload = _minimal_result(self.subject)
		payload['controller_training_authorized'] = True
		with self.assertRaisesRegex(ValueError, 'never authorize'):
			self.subject.validate_result(payload)
		payload = _minimal_result(self.subject)
		payload['privileged_targets_used'] = True
		with self.assertRaisesRegex(ValueError, 'Privileged'):
			self.subject.validate_result(payload)

	def test_11_result_rejects_wrong_shuffle_semantics(self):
		payload = _minimal_result(self.subject)
		payload['protocol']['action_shuffle_training_use'] = 'training_augmentation'
		with self.assertRaisesRegex(ValueError, 'excluded from training'):
			self.subject.validate_result(payload)
		payload = _minimal_result(self.subject)
		payload['evaluation']['clean']['hard']['action_shuffle_semantics'] = (
			'causal_counterfactual'
		)
		with self.assertRaisesRegex(ValueError, 'shuffle guard'):
			self.subject.validate_result(payload)

	def test_12_result_rejects_split_overlap_and_missing_intervention_guard(self):
		payload = _minimal_result(self.subject)
		payload['training']['clean']['action_aware']['validation_episode_ids'][0] = 0
		with self.assertRaisesRegex(ValueError, 'overlap'):
			self.subject.validate_result(payload)
		payload = _minimal_result(self.subject)
		payload['gates'][
			'true_same_state_intervention_required_before_controller_training'
		] = False
		with self.assertRaisesRegex(ValueError, 'same-state intervention'):
			self.subject.validate_result(payload)

	def test_13_atomic_output_refuses_overwrite(self):
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / 'result.json'
			self.subject._atomic_json(path, {'ok': True})
			with self.assertRaises(FileExistsError):
				self.subject._atomic_json(path, {'ok': False})


if __name__ == '__main__':
	unittest.main(verbosity=2)
