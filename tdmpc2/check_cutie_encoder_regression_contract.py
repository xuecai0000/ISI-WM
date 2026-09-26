"""CPU checks for a matched legacy/spatial encoder by auxiliary-target study.

No simulator, Cutie process, CUDA allocation or training run is constructed.
"""

from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import torch

from check_cutie_object_only_contract import make_config
from common.world_model import WorldModel
from tdmpc2 import TDMPC2


MODES = ('legacy_mlp_v1', 'spatial_graph_v1')
TARGETS = ('full_descriptor', 'geometry_status_full_denominator')


def config(mode=None, target='full_descriptor'):
	cfg = make_config(target)
	cfg.task = 'finger-spin'
	cfg.tasks = [cfg.task]
	cfg.cutie_object_role_names = ['finger', 'spinner']
	cfg.cutie_object_regression_encoder = mode
	cfg.cutie_object_spatial_token_enabled = False
	cfg.cutie_object_variable_graph_enabled = False
	cfg.cutie_object_spatial_graph_path = str(
		Path(__file__).resolve().parent / 'object_graphs' / 'finger_spin.json'
	)
	cfg.cutie_object_spatial_token_dim = 64
	cfg.cutie_object_spatial_num_heads = 4
	cfg.cutie_object_spatial_num_layers = 2
	cfg.cutie_object_variable_graph_max_roles = 8
	cfg.cutie_object_variable_graph_pool_tokens = 2
	cfg.cutie_object_variable_graph_readout = 'direct'
	cfg.cutie_object_variable_graph_primary_skip_enabled = False
	return cfg


def build(cfg):
	torch.manual_seed(271828)
	model = WorldModel(deepcopy(cfg)).eval()
	return model, torch.get_rng_state().clone()


def equal_state(left, right, label, *, exclude_encoder=False):
	def selected(model):
		return {
			key: value for key, value in model.state_dict().items()
			if not exclude_encoder or not key.startswith('_encoder.')
		}
	a, b = selected(left), selected(right)
	assert set(a) == set(b), f'{label}: state keys differ'
	for key in a:
		if isinstance(a[key], torch.Tensor):
			assert torch.equal(a[key], b[key]), f'{label}: {key} differs'
		else:
			assert a[key] == b[key], f'{label}: {key} metadata differs'


def forward_and_backward(model, value, action):
	model.zero_grad(set_to_none=True)
	objects = value.clone().requires_grad_(True)
	z = model.encode({'object': objects}, None)
	next_z = model.next(z, action, None)
	decoded = model.decode_object(z)
	future = model.decode_object(next_z)
	loss = (
		model.object_loss(decoded, value)
		+ model.object_loss(future, value)
		+ next_z.square().mean()
		+ model._pi(z).square().mean()
		+ model._reward(torch.cat([z, action], dim=-1)).mean()
	)
	loss.backward()
	grads = {
		name: None if parameter.grad is None else parameter.grad.clone()
		for name, parameter in model.named_parameters()
	}
	return z.detach(), decoded.detach(), loss.detach(), objects.grad.clone(), grads


def checkpoint_contract(legacy, models):
	"""Exercise real save/load methods on CPU models, never agent CUDA init."""
	def agent(model):
		return SimpleNamespace(
			cfg=model.cfg, model=model, _learned_object_belief=False,
			_belief_aux_updates=0, _cutie_object_mode=True,
			_cutie_object_only=True, _cutie_hybrid=False,
			_hybrid_graph=False, _reward_graph=False,
			reset_object_belief=lambda: None,
		)

	def payload(model):
		buffer = BytesIO()
		TDMPC2.save(agent(model), buffer)
		buffer.seek(0)
		return torch.load(buffer, map_location='cpu', weights_only=False)

	legacy_payload = payload(legacy)
	key = 'cutie_object_regression_encoder'
	assert key not in legacy_payload['checkpoint_contract']
	# Archived raw state dictionaries, missing metadata, and explicit null must
	# remain readable by the original/null mode with no new compatibility gate.
	for source in (
		legacy.state_dict(), legacy_payload,
		{**legacy_payload, 'checkpoint_contract': {
			**legacy_payload['checkpoint_contract'], key: None,
		}},
	):
		for mode in ('missing', None):
			cfg = config()
			if mode == 'missing':
				del cfg.cutie_object_regression_encoder
			restored, _ = build(cfg)
			TDMPC2.load(agent(restored), source)
			equal_state(legacy, restored, 'legacy checkpoint compatibility')

	for (mode, target), model in models.items():
		source = payload(model)
		assert source['checkpoint_contract'][key] == mode
		restored, _ = build(config(mode, target))
		TDMPC2.load(agent(restored), source)
		equal_state(model, restored, 'agent checkpoint round trip')
		for source_mode in (
			None, 'legacy_mlp_v1', 'spatial_graph_v1', 'legacy_mlp_k3_v1',
		):
			if source_mode == mode:
				continue
			bad = {**source, 'checkpoint_contract': {
				**source['checkpoint_contract'], key: source_mode,
			}}
			try:
				TDMPC2.load(agent(restored), bad)
			except RuntimeError as error:
				assert 'regression encoder checkpoint contract mismatch' in str(error)
			else:
				raise AssertionError('Mismatched regression encoder metadata accepted')
		# A tagged new study checkpoint must also not load into untagged mode.
		try:
			TDMPC2.load(agent(legacy), source)
		except RuntimeError as error:
			assert 'regression encoder checkpoint contract mismatch' in str(error)
		else:
			raise AssertionError('Tagged regression checkpoint accepted by legacy mode')


def main():
	torch.set_num_threads(1)
	legacy_cfg = config()
	del legacy_cfg.cutie_object_regression_encoder
	legacy, legacy_rng = build(legacy_cfg)
	null_model, null_rng = build(config())
	equal_state(legacy, null_model, 'default null compatibility')
	assert torch.equal(legacy_rng, null_rng)
	models = {}
	for mode in MODES:
		for target in TARGETS:
			model, rng = build(config(mode, target))
			models[mode, target] = model
			assert torch.equal(legacy_rng, rng), 'Regression initialization changed global RNG'
			equal_state(legacy, model, f'{mode}/{target} shared modules', exclude_encoder=True)
			assert not model._object_decoder.variable_graph
			assert not model._dynamics.variable_graph
			assert model._object_decoder.role[0].weight.shape[1] == 66
			assert model.cfg.cutie_object_spatial_token_enabled is False
			assert model.cfg.cutie_object_variable_graph_enabled is False
			if mode == 'legacy_mlp_v1':
				equal_state(legacy, model, f'{mode}/{target} legacy initialization')
			else:
				assert model._encoder['object']._variable_graph
				assert model._encoder['object'].variable_graph_readout == 'direct'
				assert model._encoder['object'].latent_dim == 128
			reloaded, _ = build(config(mode, target))
			reloaded.load_state_dict(model.state_dict(), strict=True)
			equal_state(model, reloaded, 'strict checkpoint reconstruction')
		full, geometry = (models[mode, target] for target in TARGETS)
		equal_state(full, geometry, f'{mode} target-only parameter identity')
	checkpoint_contract(legacy, models)

	torch.manual_seed(123456)
	value = torch.randn(2, 2, 1770)
	value[..., 2 * 590 + 588] = torch.tensor([1., 0.])
	action = torch.randn(2, 2)
	before = value.clone()
	baseline = models['legacy_mlp_v1', 'full_descriptor']
	a = forward_and_backward(legacy, value, action)
	b = forward_and_backward(baseline, value, action)
	for index in range(4):
		assert torch.equal(a[index], b[index]), f'Legacy forward/loss/input gradient {index} changed'
	assert set(a[4]) == set(b[4])
	for name in a[4]:
		left, right = a[4][name], b[4][name]
		assert (left is None) == (right is None), f'{name}: gradient presence changed'
		if left is not None:
			assert torch.equal(left, right), f'{name}: legacy gradient changed'
	assert torch.equal(before, value), 'Model mutated the observation'

	spatial = models['spatial_graph_v1', 'full_descriptor']
	result = forward_and_backward(spatial, value, action)
	assert result[0].shape == (2, 128)
	assert result[1].shape == (2, 2, 1770)
	assert not torch.equal(a[0], result[0]), 'Spatial intervention did not change the encoder'
	assert torch.isfinite(result[2]) and torch.isfinite(result[3]).all()
	assert torch.count_nonzero(result[3]) > 0
	for mode in MODES:
		full = models[mode, 'full_descriptor']
		geometry = models[mode, 'geometry_status_full_denominator']
		prediction = torch.randn_like(value)
		grads = []
		for model in (full, geometry):
			p = prediction.clone().requires_grad_(True)
			grad, = torch.autograd.grad(model.object_loss(p, value), p)
			grads.append(grad.reshape(2, 2, 3, 590))
		assert torch.count_nonzero(grads[1][..., :512]) == 0
		assert torch.equal(grads[0][..., 512:], grads[1][..., 512:])

	for key, value in (
		('cutie_object_variable_graph_enabled', True),
		('cutie_object_num_roles', 3),
		('cutie_object_belief_enabled', True),
		('cutie_object_observation_variant', 'gt_mask_geometry'),
	):
		cfg = config('legacy_mlp_v1')
		setattr(cfg, key, value)
		try:
			build(cfg)
		except ValueError:
			pass
		else:
			raise AssertionError(f'Regression accepted incompatible {key}')
	print('CUTIE_ENCODER_REGRESSION_CONTRACT_OK legacy_bits_rng_gradients shared_decoder_dynamics_heads four_arms checkpoint_modes')


if __name__ == '__main__':
	main()
