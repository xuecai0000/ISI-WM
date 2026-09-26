"""CPU contracts for raw-RGB background/action intervention training."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from common import layers
from common import rgb_interventional_auxiliary as auxiliary
from tdmpc2 import TDMPC2


class Config(dict):
	__getattr__ = dict.__getitem__

	def __setattr__(self, key, value):
		self[key] = value


def _config(manifest=None, *, arm='joint', checkpoint='???'):
	coefs = {
		'data_matched': (0.0, 1.0, 0.0, 0.0),
		'background': (1.0, 0.0, 0.0, 0.0),
		'fork': (0.0, 1.0, 1.0, 1.0),
		'joint': (1.0, 1.0, 1.0, 1.0),
	}[arm]
	return Config(
		rgb_interventional_aux_enabled=True,
		rgb_interventional_aux_manifest=(str(manifest) if manifest else None),
		rgb_interventional_aux_manifest_sha256=None,
		rgb_interventional_aux_capsule_format=None,
		rgb_interventional_aux_source_dataset_sha256=None,
		rgb_interventional_aux_background_coef=coefs[0],
		rgb_interventional_aux_positive_coef=coefs[1],
		rgb_interventional_aux_separation_coef=coefs[2],
		rgb_interventional_aux_ranking_coef=coefs[3],
		rgb_interventional_aux_outcome_gap_threshold=0.05,
		rgb_interventional_aux_margin_min=0.05,
		rgb_interventional_aux_margin_max=0.2,
		rgb_interventional_aux_batch_size=8,
		rgb_interventional_aux_horizon=3,
		rgb_interventional_aux_update_frequency=1,
		rgb_interventional_aux_seed_offset=32452843,
		obs='rgb', obs_shape={'rgb': (9, 64, 64)},
		flat_anchor=False, cutie_mask_guided_rgb_enabled=False,
		robust_object_field_enabled=False, multitask=False, compile=False,
		horizon=3, seed=6, task='contract-task', action_dim=2,
		checkpoint=checkpoint,
	)


def _arrays(root_id):
	action_dim, horizon = 2, 5
	branches = 1 + 2 * action_dim
	result = {}
	for condition, offset in (('clean', 0), ('hard', 50)):
		result[f'{condition}__root_rgb'] = np.full(
			(9, 64, 64), root_id + offset, np.uint8
		)
		future = np.empty((branches, horizon, 9, 64, 64), np.uint8)
		for branch in range(branches):
			future[branch].fill(root_id + offset + branch)
		result[f'{condition}__future_rgb'] = future
	actions = np.zeros((branches, horizon, action_dim), np.float32)
	actions[:, 1:] = np.asarray([0.125, -0.25], np.float32)
	actions[1, 0, 0], actions[2, 0, 0] = 0.8, -0.8
	actions[3, 0, 1], actions[4, 0, 1] = 0.8, -0.8
	result['branch__action'] = actions
	result['branch__code'] = auxiliary.branch_codes(action_dim)
	gap = np.zeros((branches, branches, horizon), np.float32)
	for left in range(branches):
		for right in range(branches):
			if left != right:
				gap[left, right] = np.linspace(0.02, 0.3, horizon, dtype=np.float32)
	result['pair__outcome_gap'] = gap
	offdiag = ~np.eye(branches, dtype=np.bool_)[:, :, None]
	result['pair__eligible'] = offdiag & (gap > np.float32(0.05))
	return result


def _capsule(root):
	(root / 'groups').mkdir(parents=True)
	groups = []
	for root_id in range(3):
		path = root / 'groups' / f'root_{root_id:04d}.npz'
		with path.open('wb') as stream:
			np.savez_compressed(stream, **_arrays(root_id))
		groups.append({
			'root_id': root_id,
			'relative_path': path.relative_to(root).as_posix(),
			'sha256': auxiliary.file_sha256(path),
			'source_group_sha256': f'{root_id + 1:064x}',
		})
	payload = {
		'format': auxiliary.FORMAT,
		'status': auxiliary.STATUS,
		'controller_auxiliary_training_authorized': True,
		'task': 'contract-task',
		'source_split': 'train',
		'test_time_input': 'rgb_only',
		'condition_pair': ['clean', 'hard'],
		'exact_physical_twins': True,
		'model_input_only': True,
		'no_cutie': True,
		'cutie_arrays_present': False,
		'masks_present': False,
		'oracle_arrays_present': False,
		'raw_privileged_arrays_present': False,
		'object_arrays_present': False,
		'whole_root_families_only': True,
		'fake_shuffled_action_futures': False,
		'policy_input_keys': list(auxiliary.POLICY_INPUT_KEYS),
		'training_target_keys': list(auxiliary.TRAINING_TARGET_KEYS),
		'action_dim': 2,
		'horizon': 5,
		'branch_magnitude': 0.8,
		'source_root_ids': list(range(3)),
		'outcome_gap': {
			'threshold': 0.05,
			'normalization': 'train_root_coordinate_population_std_then_rms',
			'exact_replay_noise_floor': 0.0,
			'ineligible_margin': 0.0,
		},
		'background_twin_contract': {
			'conditions': ['clean', 'hard'],
			'exact_physics_per_branch': True,
			'exact_official_state_per_branch': True,
			'independent_fresh_reset_replay': True,
		},
		'action_sibling_contract': {
			'common_exact_physical_root': True,
			'only_real_executed_actions': True,
			'interventions': 'zero_plus_minus_each_scaled_action_basis',
			'continuation': 'identical_open_loop_actions_across_siblings',
		},
		'outcome_target_contract': {
			'source': 'oracle_official_state_train_roots_only',
			'raw_source_stripped_after_derivation': True,
			'coordinate_scale': 'population_std_all_train_branch_futures',
			'distance': 'coordinate_standardized_rms',
			'symmetric': True,
			'diagonal_zero': True,
			'scale_epsilon': 1e-6,
			'coordinate_count': 45,
			'state_dim': 3,
			'coordinate_scale_sha256': 'b' * 64,
			'exact_replay_noise_floor': 0.0,
			'eligibility_threshold': 0.05,
			'eligibility': (
				'off_diagonal_and_gap_strictly_greater_than_threshold'
			),
		},
		'source_dataset': {
			'format': 'rof_same_state_action_branch_dataset_v1',
			'manifest_sha256': 'a' * 64,
			'controller_training_authorized': False,
			'validation_performed_before_privilege_stripping': True,
			'validation_performed_before_derivation_and_stripping': True,
		},
		'groups': groups,
	}
	manifest = root / 'training_capsule_manifest.json'
	manifest.write_text(
		json.dumps(payload, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	return manifest


def _bind(cfg, manifest):
	payload = auxiliary.validate_manifest(manifest, cfg)
	cfg.rgb_interventional_aux_manifest_sha256 = payload['_manifest_sha256']
	cfg.rgb_interventional_aux_capsule_format = payload['format']
	cfg.rgb_interventional_aux_source_dataset_sha256 = payload[
		'source_dataset'
	]['manifest_sha256']
	return auxiliary.validate_config(cfg, require_bound_manifest=True)


def _shared_crop_contract():
	# Exercise the agent helper without constructing its CUDA-only __init__.
	cfg = Config(simnorm_dim=8)
	model = torch.nn.Module()
	model._encoder = torch.nn.ModuleDict({
		'rgb': layers.conv((9, 64, 64), 32, act=layers.SimNorm(cfg)),
	})
	agent = object.__new__(TDMPC2)
	torch.nn.Module.__init__(agent)
	agent.model = model
	agent.device = torch.device('cpu')
	agent._rgb_interventional_shift_generator = torch.Generator(device='cpu')
	agent._rgb_interventional_shift_generator.manual_seed(123)
	agent._rgb_interventional_shift_calls = 0
	agent._rgb_interventional_shift_rows = 0
	agent._rgb_interventional_shift_shape_trace = hashlib.sha256(
		b'rgb_interventional_augmentation_shapes_v1'
	)
	rgb = torch.randint(0, 256, (4, 9, 64, 64), dtype=torch.uint8)
	shift = TDMPC2._sample_rgb_interventional_shift(agent, 4)
	left = TDMPC2._encode_rgb_interventional_shared(agent, rgb, shift)
	right = TDMPC2._encode_rgb_interventional_shared(agent, rgb.clone(), shift)
	assert torch.equal(left, right)
	assert 'generator=self._rgb_interventional_shift_generator' in inspect.getsource(
		TDMPC2._sample_rgb_interventional_shift
	)


def _strict_shared_forward_contract():
	class ForwardModel(torch.nn.Module):
		def __init__(self):
			super().__init__()
			cfg = Config(simnorm_dim=8)
			self._encoder = torch.nn.ModuleDict({
				'rgb': layers.conv((9, 64, 64), 32, act=layers.SimNorm(cfg)),
			})
			self.action = torch.nn.Linear(2, 512, bias=False)

		def next(self, latent, action, task):
			assert task is None
			return latent + self.action(action)

	agent = object.__new__(TDMPC2)
	torch.nn.Module.__init__(agent)
	agent.model = ForwardModel()
	agent.device = torch.device('cpu')
	agent.cfg = _config(arm='joint')
	agent.cfg.rho = 0.5
	agent._rgb_interventional_aux_enabled = True
	agent._rgb_interventional_aux_contract = auxiliary.validate_config(agent.cfg)
	agent._rgb_interventional_shift_generator = torch.Generator(device='cpu')
	agent._rgb_interventional_shift_calls = 0
	agent._rgb_interventional_shift_rows = 0
	agent._rgb_interventional_shift_shape_trace = hashlib.sha256(
		b'rgb_interventional_augmentation_shapes_v1'
	)
	horizon, batch = 3, 2
	root = [
		torch.randint(0, 256, (batch, 9, 64, 64), dtype=torch.uint8)
		for _ in range(2)
	]
	future = [
		torch.randint(
			0, 256, (horizon, batch, 9, 64, 64), dtype=torch.uint8
		)
		for _ in range(4)
	]
	action = torch.randn(horizon, batch, 2)
	gap = torch.ones(horizon, batch)
	eligible = torch.ones(horizon, batch, dtype=torch.bool)

	agent._rgb_interventional_shift_generator.manual_seed(991)
	joint = TDMPC2._rgb_interventional_objective(
		agent, *root, *future, action, gap, eligible,
	)
	joint_rng = agent._rgb_interventional_shift_generator.get_state().clone()
	joint_shape_trace = agent._rgb_interventional_shift_shape_trace.hexdigest()
	assert agent._rgb_interventional_shift_calls == horizon + 1
	assert agent._rgb_interventional_shift_rows == (horizon + 1) * batch
	agent.cfg = _config(arm='data_matched')
	agent.cfg.rho = 0.5
	agent._rgb_interventional_aux_contract = auxiliary.validate_config(agent.cfg)
	agent._rgb_interventional_shift_generator.manual_seed(991)
	agent._rgb_interventional_shift_calls = 0
	agent._rgb_interventional_shift_rows = 0
	agent._rgb_interventional_shift_shape_trace = hashlib.sha256(
		b'rgb_interventional_augmentation_shapes_v1'
	)
	matched = TDMPC2._rgb_data_matched_objective(
		agent, *root, *future, action,
	)
	matched_rng = agent._rgb_interventional_shift_generator.get_state().clone()
	assert torch.equal(joint_rng, matched_rng)
	assert joint_shape_trace == agent._rgb_interventional_shift_shape_trace.hexdigest()
	assert agent._rgb_interventional_shift_calls == horizon + 1
	assert agent._rgb_interventional_shift_rows == (horizon + 1) * batch
	assert torch.isfinite(joint['weighted_loss'])
	assert torch.isfinite(matched['weighted_loss'])
	# One B-sized root crop plus one B-sized shared future crop per horizon.
	expected_generator = torch.Generator(device='cpu')
	expected_generator.manual_seed(991)
	pad = int(list(agent.model._encoder['rgb'].children())[0].pad)
	for _ in range(horizon + 1):
		torch.randint(
			0, 2 * pad + 1, (batch, 1, 1, 2), generator=expected_generator,
			dtype=torch.int64,
		)
	assert torch.equal(joint_rng, expected_generator.get_state())


def main():
	with tempfile.TemporaryDirectory(prefix='rgb-intervention-contract-') as temp:
		root = Path(temp)
		manifest = _capsule(root)
		cfg = _config(manifest, arm='joint')
		contract = _bind(cfg, manifest)
		assert contract['arm'] == 'joint'
		left, right = auxiliary.Replay(cfg, device='cpu'), auxiliary.Replay(cfg, device='cpu')
		for _ in range(3):
			a, b = left.sample(), right.sample()
			assert torch.equal(a.root_id, b.root_id)
			assert torch.equal(a.positive_code, b.positive_code)
			assert torch.equal(a.negative_code, b.negative_code)
			assert torch.equal(a.negative_action, b.negative_action)
			assert torch.equal(a.eligible, a.outcome_gap > 0.05)
		assert left.metrics['sampled_root_pairs'] == 24
		assert (
			left.metrics['sample_sequence_sha256']
			== right.metrics['sample_sequence_sha256']
		)
		record = {
			'contract': contract,
			'supervision': {
				'attempts': 3, 'successful_updates': 3,
				'sampled_root_pairs': 24,
				'sampler_metrics': left.metrics,
				'augmentation_metrics': {
					'crop_calls': 12,
					'crop_rows': 96,
					'draw_shape_sequence_sha256': 'c' * 64,
					'generator_state_sha256': 'd' * 64,
				},
				'pairing_metrics': {
					'format': 'rgb_interventional_pairing_schedule_v1',
					'mode': 'correct',
					'seed': contract['pairing']['effective_seed'],
					'batch_size': 8, 'calls': 3, 'rows': 24,
					'background_correct_pairs': 24,
					'fork_correct_pairs': 24,
					'background_correct_rate': 1.0,
					'fork_correct_rate': 1.0,
					'candidate_sequence_sha256': 'e' * 64,
					'applied_sequence_sha256': 'f' * 64,
				},
			},
		}
		assert auxiliary.validate_checkpoint_record(record, contract)[
			'sampled_root_pairs'
		] == 24
		matched_cfg = _config(manifest, arm='data_matched')
		matched_contract = _bind(matched_cfg, manifest)
		assert matched_contract['arm'] == 'data_matched'
		assert matched_contract['pair_targets_used_for_gradient'] is False
		assert matched_contract['background_twin_identity_used_for_gradient'] is False
		assert matched_contract['data_matched_transition_streams'] == [
			'clean_positive', 'hard_positive',
		]
		assert (
			matched_contract['shared_forward_contract']
			== contract['shared_forward_contract']
			== auxiliary.SHARED_FORWARD_CONTRACT
		)
		matched_replay = auxiliary.Replay(matched_cfg, device='cpu')
		joint_replay = auxiliary.Replay(cfg, device='cpu')
		for _ in range(3):
			matched_batch = matched_replay.sample()
			joint_batch = joint_replay.sample()
			for field in (
				'clean_root', 'hard_root',
				'clean_positive_future', 'hard_positive_future',
				'clean_negative_future', 'hard_negative_future',
				'action', 'negative_action', 'root_id',
				'positive_code', 'negative_code',
			):
				assert torch.equal(
					getattr(matched_batch, field), getattr(joint_batch, field)
				), field
		assert (
			matched_replay.metrics['sample_sequence_sha256']
			== joint_replay.metrics['sample_sequence_sha256']
		)
		assert matched_batch.outcome_gap is None
		assert matched_batch.eligible is None
		assert matched_batch.negative_action is not None
		# Policy evaluation binds provenance but does not open the training capsule.
		eval_cfg = Config(cfg)
		eval_cfg.checkpoint = str(root / 'model.pt')
		eval_cfg.rgb_interventional_aux_manifest = str(root / 'missing.json')
		fake_agent = object.__new__(TDMPC2)
		fake_agent.cfg = eval_cfg
		fake_agent._rof_real_sibling_aux_enabled = False
		TDMPC2._configure_rgb_interventional_auxiliary(fake_agent)
		assert fake_agent._rgb_interventional_aux_training is False
		assert fake_agent._rgb_interventional_aux_replay is None
		assert auxiliary.validate_checkpoint_record(
			record, fake_agent._rgb_interventional_aux_contract
		)['successful_updates'] == 3

		bad = _arrays(0)
		bad['oracle__official_state'] = np.zeros(1, np.float32)
		try:
			auxiliary.validate_shard(
				bad, action_dim=2, horizon=5, branch_magnitude=0.8,
				outcome_gap_threshold=0.05,
			)
		except ValueError:
			pass
		else:
			raise AssertionError('Oracle-bearing RGB training shard was accepted.')

	# Ineligible pairs create neither separation nor ranking gradients.
	prediction = torch.full((3, 4, 8), 0.5, requires_grad=True)
	positive = torch.zeros_like(prediction, requires_grad=True)
	negative = torch.ones_like(prediction, requires_grad=True)
	ineligible = auxiliary.outcome_grounded_fork_objective(
		prediction, positive, negative,
		torch.zeros(3, 4), torch.zeros(3, 4, dtype=torch.bool),
		rho=0.5, positive_coef=0.0, separation_coef=1.0,
		ranking_coef=1.0, margin_min=0.05, margin_max=0.2,
	)
	assert ineligible['separation_loss'].item() == 0.0
	assert ineligible['ranking_loss'].item() == 0.0
	ineligible['weighted_loss'].backward()
	for tensor in (prediction, positive, negative):
		assert tensor.grad is not None and torch.count_nonzero(tensor.grad) == 0

	# The data-matched objective is ordinary stop-gradient transition
	# consistency. Its API cannot receive twin/sibling identities or derived
	# outcome/eligibility targets.
	parameters = inspect.signature(
		auxiliary.transition_consistency_objective
	).parameters
	assert set(parameters) == {'predicted', 'target', 'rho'}
	prediction = torch.randn(3, 8, 16, requires_grad=True)
	target = torch.randn_like(prediction, requires_grad=True)
	matched = auxiliary.transition_consistency_objective(
		prediction, target, rho=0.5,
	)
	assert matched['loss'].item() > 0.0
	matched['loss'].backward()
	assert prediction.grad is not None and torch.count_nonzero(prediction.grad)
	assert target.grad is None
	data_matched_parameters = inspect.signature(
		TDMPC2._rgb_data_matched_objective
	).parameters
	assert not {
		'outcome_gap', 'eligible', 'pair', 'sibling', 'twin',
	} & set(data_matched_parameters)
	data_matched_source = inspect.getsource(TDMPC2._rgb_data_matched_objective)
	joint_source = inspect.getsource(TDMPC2._rgb_interventional_objective)
	assert data_matched_source.count('self._rgb_interventional_forward(') == 1
	assert joint_source.count('self._rgb_interventional_forward(') == 1
	for source in (data_matched_source, joint_source):
		assert '_sample_rgb_interventional_shift' not in source
		assert '_encode_rgb_interventional_shared' not in source
	assert 'background_invariance_loss' not in data_matched_source
	assert 'outcome_grounded_fork_objective' not in data_matched_source
	assert 'negative_action' not in data_matched_source

	# A known different outcome pushes a near-collapsed pair apart. Exact equality
	# is the unavoidable stationary point of every symmetric distance, while real
	# distinct RGB encodings begin with non-identical finite features.
	prediction = torch.zeros(3, 4, 8, requires_grad=True)
	positive = torch.zeros_like(prediction, requires_grad=True)
	negative = torch.full_like(prediction, 1e-4, requires_grad=True)
	eligible = auxiliary.outcome_grounded_fork_objective(
		prediction, positive, negative,
		torch.ones(3, 4), torch.ones(3, 4, dtype=torch.bool),
		rho=0.5, positive_coef=0.0, separation_coef=1.0,
		ranking_coef=0.0, margin_min=0.05, margin_max=0.2,
	)
	assert eligible['separation_loss'].item() > 0.0
	eligible['weighted_loss'].backward()
	assert positive.grad is not None and negative.grad is not None
	assert torch.isfinite(positive.grad).all() and torch.isfinite(negative.grad).all()

	clean = torch.randn(3, 5, requires_grad=True)
	hard = torch.randn(3, 5, requires_grad=True)
	auxiliary.background_invariance_loss(clean, hard).backward()
	assert clean.grad is not None and hard.grad is not None

	_shared_crop_contract()
	_strict_shared_forward_contract()
	assert auxiliary.validate_config({'rgb_interventional_aux_enabled': False}) is None
	try:
		auxiliary.validate_config({
			'rgb_interventional_aux_enabled': False,
			'rgb_interventional_aux_manifest': 'stale.json',
		})
	except ValueError:
		pass
	else:
		raise AssertionError('Disabled RGB auxiliary accepted a stale capsule path.')
	print('RGB_INTERVENTIONAL_AUXILIARY_CONTRACT_PASS')


if __name__ == '__main__':
	main()
