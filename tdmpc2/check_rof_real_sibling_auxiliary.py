"""Focused contracts for ROF real-sibling auxiliary training.

This check is CPU-only.  It exercises the immutable capsule and sampler,
objective gradients, checkpoint provenance, optimizer-step structure, and the
policy-evaluation path that must remain independent of capsule availability.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import tempfile
import textwrap

import numpy as np
import torch

from common import rof_real_sibling_auxiliary as auxiliary
from tdmpc2 import TDMPC2


class Config(dict):
	__getattr__ = dict.__getitem__

	def __setattr__(self, key, value):
		self[key] = value


def _config(manifest: Path | None, *, checkpoint='???', ranking=1.0):
	return Config(
		rof_real_sibling_aux_enabled=True,
		rof_real_sibling_aux_manifest=(str(manifest) if manifest else None),
		rof_real_sibling_aux_manifest_sha256=None,
		rof_real_sibling_aux_capsule_format=None,
		rof_real_sibling_aux_source_dataset_sha256=None,
		rof_real_sibling_aux_condition='clean',
		rof_real_sibling_aux_positive_coef=1.0,
		rof_real_sibling_aux_ranking_coef=float(ranking),
		rof_real_sibling_aux_margin_fraction=0.1,
		rof_real_sibling_aux_batch_size=8,
		rof_real_sibling_aux_horizon=3,
		rof_real_sibling_aux_update_frequency=1,
		rof_real_sibling_aux_seed_offset=15485863,
		robust_object_field_enabled=True,
		flat_anchor=True,
		flat_anchor_mode='cutie_object_only',
		multitask=False,
		compile=False,
		horizon=3,
		seed=6,
		task='contract-task',
		cutie_object_num_roles=2,
		cutie_object_role_names=['left', 'right'],
		action_dim=2,
		checkpoint=checkpoint,
	)


def _arrays(root_id: int):
	roles, action_dim, horizon = 2, 2, 5
	branches = 1 + 2 * action_dim
	shapes = {
		'rgb': (9, 64, 64),
		'object': (roles, 1770),
		'object_mask': (roles, 3, 64, 64),
		'role_exists': (roles,),
	}
	dtypes = {
		'rgb': np.uint8,
		'object': np.float32,
		'object_mask': np.bool_,
		'role_exists': np.float32,
	}
	result = {}
	for field, shape in shapes.items():
		value = np.zeros(shape, dtype=dtypes[field])
		if field == 'rgb':
			value.fill(root_id)
		elif field == 'role_exists':
			value.fill(1.0)
		result[f'root__policy_{field}'] = value
		future = np.zeros((branches, horizon) + shape, dtype=dtypes[field])
		if field == 'role_exists':
			future.fill(1.0)
		result[f'future__policy_{field}'] = future
	actions = np.zeros((branches, horizon, action_dim), dtype=np.float32)
	actions[:, 1:] = np.asarray([0.125, -0.25], dtype=np.float32)
	actions[1, 0, 0] = 0.8
	actions[2, 0, 0] = -0.8
	actions[3, 0, 1] = 0.8
	actions[4, 0, 1] = -0.8
	result['branch__action'] = actions
	result['branch__code'] = auxiliary.branch_codes(action_dim)
	return result


def _capsule(
	root: Path, *, task='contract-task', role_names=('left', 'right'),
):
	if len(role_names) != 2:
		raise ValueError('Synthetic capsule helper is fixed to two roles.')
	groups = []
	(root / 'groups').mkdir(parents=True)
	for root_id in range(3):
		path = root / 'groups' / f'root_{root_id:04d}.npz'
		with path.open('wb') as stream:
			np.savez_compressed(stream, **_arrays(root_id))
		groups.append({
			'root_id': root_id,
			'relative_path': path.relative_to(root).as_posix(),
			'sha256': auxiliary.file_sha256(path),
		})
	payload = {
		'format': auxiliary.FORMAT,
		'status': auxiliary.STATUS,
		'controller_auxiliary_training_authorized': True,
		'task': task,
		'condition': 'clean',
		'source_split': 'train',
		'model_input_only': True,
		'oracle_arrays_present': False,
		'root_grouping': 'complete_real_action_sibling_family',
		'fake_shuffled_action_futures': False,
		'role_names': list(role_names),
		'num_roles': 2,
		'action_dim': 2,
		'horizon': 5,
		'branch_magnitude': 0.8,
		'source_dataset': {
			'format': 'rof_same_state_action_branch_dataset_v1',
			'manifest_sha256': 'a' * 64,
			'controller_training_authorized': False,
			'validation_performed_before_privilege_stripping': True,
		},
		'groups': groups,
	}
	manifest = root / 'training_capsule_manifest.json'
	manifest.write_text(
		json.dumps(payload, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	return manifest


def _optimizer_step_contract():
	def count_steps(fn):
		tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
		return sum(
			isinstance(node, ast.Call)
			and isinstance(node.func, ast.Attribute)
			and node.func.attr == 'step'
			and isinstance(node.func.value, ast.Attribute)
			and node.func.value.attr == 'optim'
			for node in ast.walk(tree)
		)
	assert count_steps(TDMPC2._update) == 1
	assert count_steps(TDMPC2._real_sibling_objective) == 0
	assert count_steps(TDMPC2.update) == 0


def main():
	_optimizer_step_contract()
	with tempfile.TemporaryDirectory(prefix='rof-real-sibling-contract-') as temp:
		root = Path(temp)
		manifest = _capsule(root)
		cfg = _config(manifest)
		payload = auxiliary.validate_manifest(manifest, cfg)
		cfg.rof_real_sibling_aux_manifest_sha256 = payload['_manifest_sha256']
		cfg.rof_real_sibling_aux_capsule_format = payload['format']
		cfg.rof_real_sibling_aux_source_dataset_sha256 = payload[
			'source_dataset'
		]['manifest_sha256']
		contract = auxiliary.validate_config(cfg, require_bound_manifest=True)
		assert contract['arm'] == 'sibling_relational'

		left = auxiliary.Replay(cfg, device='cpu')
		right = auxiliary.Replay(cfg, device='cpu')
		for _ in range(3):
			left_batch = left.sample()
			right_batch = right.sample()
			assert torch.equal(left_batch.root_id, right_batch.root_id)
			assert torch.equal(left_batch.positive_code, right_batch.positive_code)
			assert torch.equal(left_batch.negative_code, right_batch.negative_code)
		metrics = left.metrics
		assert metrics['sampled_root_pairs'] == 24
		assert set(metrics['root_sample_counts'].values()) == {8}
		assert set(metrics['ordered_code_pair_counts'].values()) == {2}
		assert set(metrics['relation_type_counts'].values()) == {8}

		record = {
			'contract': contract,
			'supervision': {
				'attempts': 3,
				'successful_updates': 3,
				'sampled_root_pairs': 24,
				'sampler_metrics': metrics,
			},
		}
		assert auxiliary.validate_checkpoint_record(record, contract)[
			'successful_updates'
		] == 3

		# Evaluation reconstructs and validates provenance from bound values only.
		# The capsule path can be gone without making policy inference depend on it.
		# Config.__getattr__ deliberately raises KeyError for missing keys, which
		# makes copy.deepcopy probe ``__deepcopy__`` through that mapping hook.
		# This check only mutates scalar top-level evaluation fields, so an
		# explicit mapping copy is both sufficient and compatible.
		eval_cfg = Config(cfg)
		eval_cfg.checkpoint = str(root / 'model.pt')
		eval_cfg.rof_real_sibling_aux_manifest = str(root / 'missing-capsule.json')
		fake_agent = object.__new__(TDMPC2)
		fake_agent.cfg = eval_cfg
		TDMPC2._configure_real_sibling_auxiliary(fake_agent)
		assert fake_agent._rof_real_sibling_aux_replay is None
		assert fake_agent._rof_real_sibling_aux_training is False
		assert auxiliary.validate_checkpoint_record(
			record, fake_agent._rof_real_sibling_aux_contract
		)['sampled_root_pairs'] == 24

		bad = _arrays(0)
		bad['oracle__state'] = np.zeros(1, dtype=np.float32)
		try:
			auxiliary.validate_shard(
				bad, roles=2, action_dim=2, horizon=5,
				branch_magnitude=0.8,
			)
		except ValueError:
			pass
		else:
			raise AssertionError('Oracle-bearing training shard was accepted.')

	# Targets are stop-gradient and the relational term changes only prediction.
	prediction = torch.full((3, 4, 8), 0.8, requires_grad=True)
	positive = torch.zeros_like(prediction, requires_grad=True)
	negative = torch.ones_like(prediction, requires_grad=True)
	objective = auxiliary.latent_objective(
		prediction, positive, negative, rho=0.5, positive_coef=1.0,
		ranking_coef=1.0, margin_fraction=0.1,
	)
	objective['weighted_loss'].backward()
	assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
	assert positive.grad is None and negative.grad is None

	# An auxiliary-owned generator must not perturb the global stream used by
	# base augmentation and action sampling.
	torch.manual_seed(123)
	state = torch.get_rng_state().clone()
	generator = torch.Generator(device='cpu')
	generator.manual_seed(456)
	torch.randint(0, 7, (32,), generator=generator)
	assert torch.equal(state, torch.get_rng_state())
	shift_source = inspect.getsource(TDMPC2._sample_real_sibling_shift)
	assert 'generator=self._rof_real_sibling_shift_generator' in shift_source

	assert auxiliary.validate_config({'rof_real_sibling_aux_enabled': False}) is None
	try:
		auxiliary.validate_config({
			'rof_real_sibling_aux_enabled': False,
			'rof_real_sibling_aux_manifest': 'stale.json',
		})
	except ValueError:
		pass
	else:
		raise AssertionError('Disabled mode accepted a stale capsule path.')
	print('ROF_REAL_SIBLING_AUXILIARY_CONTRACT_PASS')


if __name__ == '__main__':
	main()
