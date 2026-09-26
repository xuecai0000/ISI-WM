"""One-step CUDA contract for the joint ROF real-sibling objective."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile

import torch

from check_robust_object_field_v0_update import (
	SyntheticBuffer,
	make_config,
	sample,
)
from check_rof_real_sibling_auxiliary import _capsule
from common.seed import set_seed
from tdmpc2 import TDMPC2


class Args:
	compile = False
	roles = 2
	batch_size = 2
	seed = 314159
	auxiliary_target = 'full_descriptor'
	auxiliary_reconstruction_coef = 1.0
	auxiliary_prediction_coef = 1.0


def _config(*, manifest=None, ranking=0.0):
	cfg = make_config(Args())
	cfg.compile = False
	cfg.compile_fallback_random = False
	cfg.checkpoint = '???'
	cfg.rof_real_sibling_aux_enabled = manifest is not None
	cfg.rof_real_sibling_aux_manifest = (
		str(manifest) if manifest is not None else None
	)
	cfg.rof_real_sibling_aux_manifest_sha256 = None
	cfg.rof_real_sibling_aux_capsule_format = None
	cfg.rof_real_sibling_aux_source_dataset_sha256 = None
	cfg.rof_real_sibling_aux_condition = 'clean'
	cfg.rof_real_sibling_aux_positive_coef = 1.0
	cfg.rof_real_sibling_aux_ranking_coef = float(ranking)
	cfg.rof_real_sibling_aux_margin_fraction = 0.1
	cfg.rof_real_sibling_aux_batch_size = 8
	cfg.rof_real_sibling_aux_horizon = 3
	cfg.rof_real_sibling_aux_update_frequency = 1
	cfg.rof_real_sibling_aux_seed_offset = 15485863
	return cfg


def _run(cfg):
	set_seed(cfg.seed)
	agent = TDMPC2(cfg)
	batch = sample(cfg)
	step_count = 0
	original_step = agent.optim.step

	def counted_step(*args, **kwargs):
		nonlocal step_count
		step_count += 1
		return original_step(*args, **kwargs)

	agent.optim.step = counted_step
	metrics = agent.update(SyntheticBuffer(batch))
	if step_count != 1:
		raise AssertionError(
			f'Joint base+sibling update made {step_count} model Adam steps.'
		)
	return agent, metrics, torch.cuda.get_rng_state(0).clone()


def main():
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required for the sibling update contract.')
	with tempfile.TemporaryDirectory(
		prefix='rof_real_sibling_aux_update_'
	) as directory:
		root = Path(directory)
		capsule_root = root / 'capsule'
		manifest = _capsule(
			capsule_root, task='reacher-easy',
			role_names=('role_0', 'role_1'),
		)
		baseline, baseline_metrics, baseline_rng = _run(_config())
		relational_cfg = _config(manifest=manifest, ranking=1.0)
		relational, metrics, relational_rng = _run(relational_cfg)
		if not torch.equal(baseline_rng, relational_rng):
			raise AssertionError(
				'Auxiliary training perturbed the base CUDA RNG stream.'
		)
		for name in (
			'sibling_positive_loss', 'sibling_ranking_loss',
			'sibling_weighted_loss', 'sibling_positive_mse',
			'sibling_wrong_mse', 'sibling_ranking_accuracy',
			'sibling_aux_updates_total', 'sibling_aux_samples_total',
		):
			if name not in metrics or not torch.isfinite(metrics[name]).all():
				raise AssertionError(f'Missing/non-finite sibling metric: {name}.')
		if any(name.startswith('sibling_') for name in baseline_metrics):
			raise AssertionError('Disabled baseline emitted sibling metrics.')
		if int(metrics['sibling_aux_updates_total']) != 1:
			raise AssertionError('Sibling update counter did not advance once.')
		if int(metrics['sibling_aux_samples_total']) != 8:
			raise AssertionError('Sibling sample counter did not advance by batch eight.')

		checkpoint = root / 'relational.pt'
		relational.save(checkpoint)
		payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
		record = payload['checkpoint_contract'].get(
			'rof_real_sibling_auxiliary'
		)
		if record is None or record['contract']['arm'] != 'sibling_relational':
			raise AssertionError('Checkpoint omitted relational sibling provenance.')

		# Move the capsule away.  Evaluation construction/load must rely only on
		# the bound identity serialized in runtime config and checkpoint.
		manifest.rename(root / 'training_capsule_manifest.moved.json')
		eval_cfg = copy.deepcopy(relational.cfg)
		eval_cfg.checkpoint = str(checkpoint)
		evaluator = TDMPC2(eval_cfg)
		if evaluator._rof_real_sibling_aux_replay is not None:
			raise AssertionError('Policy evaluator loaded the training replay.')
		evaluator.load(checkpoint)

	print('ROF_REAL_SIBLING_AUXILIARY_UPDATE_PASS', {
		'device': torch.cuda.get_device_name(0),
		'model_adam_steps': 1,
		'base_cuda_rng_preserved': True,
		'evaluation_without_capsule': True,
		'sibling_positive_mse': float(metrics['sibling_positive_mse']),
		'sibling_wrong_mse': float(metrics['sibling_wrong_mse']),
		'sibling_ranking_accuracy': float(metrics['sibling_ranking_accuracy']),
	})


if __name__ == '__main__':
	main()
