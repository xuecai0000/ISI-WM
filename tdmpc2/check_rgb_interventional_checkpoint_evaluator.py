"""Focused CPU/synthetic contracts for the RGB mechanism evaluator."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import types

import numpy as np

# The focused checks below exercise only NumPy aggregation and scalar runtime
# contracts.  Permit them in the bundled CPU validation runtime, which does not
# ship PyTorch; real checkpoint evaluation still imports and requires CUDA.
try:
	import torch  # noqa: F401
except ModuleNotFoundError:
	torch_stub = types.ModuleType('torch')
	torch_nn_stub = types.ModuleType('torch.nn')
	torch_functional_stub = types.ModuleType('torch.nn.functional')
	torch_stub.nn = torch_nn_stub
	torch_stub.Tensor = object
	sys.modules['torch'] = torch_stub
	sys.modules['torch.nn'] = torch_nn_stub
	sys.modules['torch.nn.functional'] = torch_functional_stub

from tools import evaluate_rgb_interventional_checkpoint as evaluator


def _runtime():
	value = {
		'task': 'acrobot-swingup',
		'obs': 'rgb',
		'obs_shape': {'rgb': [9, 64, 64]},
		'action_dim': 1,
		'multitask': False,
		'latent_dim': 512,
		'model_size': 5,
		'enc_dim': 256,
		'mlp_dim': 512,
		'num_enc_layers': 2,
		'steps': 30000,
		'eval_freq': 5000,
		'eval_episodes': 10,
		'episode_length': 500,
		'compile': False,
		'video_background_enabled': False,
		'video_background_split': 'train',
		'visual_foreground_erosion_pixels': 0,
		'horizon': 3,
		'rgb_interventional_aux_enabled': False,
		'rgb_interventional_aux_manifest': None,
		'rgb_interventional_aux_manifest_sha256': None,
		'rgb_interventional_aux_capsule_format': None,
		'rgb_interventional_aux_source_dataset_sha256': None,
		'rgb_interventional_aux_background_coef': 0.0,
		'rgb_interventional_aux_positive_coef': 0.0,
		'rgb_interventional_aux_separation_coef': 0.0,
		'rgb_interventional_aux_ranking_coef': 0.0,
		'rgb_interventional_aux_outcome_gap_threshold': 0.05,
		'rgb_interventional_aux_margin_min': 0.05,
		'rgb_interventional_aux_margin_max': 0.2,
		'rgb_interventional_aux_batch_size': 8,
		'rgb_interventional_aux_horizon': 3,
		'rgb_interventional_aux_update_frequency': 1,
		'rgb_interventional_aux_seed_offset': 32452843,
	}
	value.update({key: False for key in evaluator.FORBIDDEN_BOOLEAN_ROUTES})
	value.update({key: None for key in evaluator.FORBIDDEN_IDENTITY_ROUTES})
	return value


def test_only_eligible_pairs_are_scored():
	# Three branches, two horizons. The deliberately huge ineligible mismatch
	# must not affect any published mean.
	target = np.asarray([
		[[0.0, 0.0], [0.0, 0.0]],
		[[1.0, 0.0], [100.0, 100.0]],
		[[-1.0, 0.0], [-100.0, -100.0]],
	], dtype=np.float32)
	predicted = target.copy()
	eligible = np.zeros((3, 3, 2), dtype=np.bool_)
	eligible[0, 1, 0] = True
	eligible[1, 0, 0] = True
	samples = evaluator._comparison_samples(predicted, target, eligible)
	summary = evaluator._comparison_summary(samples)
	assert summary['eligible_comparisons'] == 2
	assert summary['correct_action_prediction_mse'] == 0.0
	assert summary['wrong_sibling_prediction_mse'] == 0.5
	assert summary['model_action_ranking_accuracy'] == 1.0
	assert summary['prediction_error_separation_fraction'] == 1.0


def test_action_misranking_is_detected():
	target = np.asarray([
		[[0.0, 0.0]], [[1.0, 0.0]], [[-1.0, 0.0]],
	], dtype=np.float32)
	# Put the first two predictions much closer to the wrong sibling while
	# retaining non-zero wrong-sibling error for a finite separation ratio.
	predicted = target.copy()
	predicted[0, 0, 0] = 0.8
	predicted[1, 0, 0] = 0.2
	eligible = np.zeros((3, 3, 1), dtype=np.bool_)
	eligible[0, 1, 0] = eligible[1, 0, 0] = True
	summary = evaluator._comparison_summary(
		evaluator._comparison_samples(predicted, target, eligible)
	)
	assert summary['eligible_comparisons'] == 2
	assert summary['model_action_ranking_accuracy'] == 0.0
	assert summary['prediction_error_separation_fraction'] < 0.0


def test_background_ratio_uses_action_scale():
	background = np.asarray([0.1, 0.3], dtype=np.float64)
	action = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
	assert np.isclose(evaluator._ratio(background, action), 0.2)
	assert evaluator._ratio(background, np.empty(0)) is None


def test_baseline_runtime_is_strictly_rgb_only():
	raw = _runtime()
	contract = evaluator._validate_runtime_contract(
		raw, task='acrobot-swingup', arm='tdmpc2', action_dim=1,
		capsule_path=Path('/unused/capsule.json'),
		capsule={'_manifest_sha256': '0' * 64}, expected_step=30000,
	)
	assert contract is None
	for key in evaluator.FORBIDDEN_BOOLEAN_ROUTES:
		bad = deepcopy(raw)
		bad[key] = True
		try:
			evaluator._validate_runtime_contract(
				bad, task='acrobot-swingup', arm='tdmpc2', action_dim=1,
				capsule_path=Path('/unused/capsule.json'),
				capsule={'_manifest_sha256': '0' * 64}, expected_step=30000,
			)
		except ValueError:
			pass
		else:
			raise AssertionError(f'Forbidden route was accepted: {key}')


def main():
	test_only_eligible_pairs_are_scored()
	test_action_misranking_is_detected()
	test_background_ratio_uses_action_scale()
	test_baseline_runtime_is_strictly_rgb_only()
	print('RGB_INTERVENTIONAL_CHECKPOINT_EVALUATOR_CHECKS_PASSED')


if __name__ == '__main__':
	main()
