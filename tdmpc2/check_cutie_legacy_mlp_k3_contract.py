"""CPU contract for the explicit three-role legacy Cutie MLP extension.

No simulator, tracker worker, CUDA allocation, or training run is constructed.
The established K=2 mode is exercised by check_cutie_encoder_regression_contract;
this file proves that only the explicitly tagged K=3 mode admits a non-spatial
three-role state and that its checkpoint cannot cross the K=2 boundary.
"""

from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace

import torch

from check_cutie_object_only_contract import make_config
from common.world_model import WorldModel
from tdmpc2 import TDMPC2


MODE = 'legacy_mlp_k3_v1'


def config():
	cfg = make_config('full_descriptor')
	cfg.task = 'synthetic-locomotion-k3'
	cfg.tasks = [cfg.task]
	cfg.obs_shape = {'object': (3, 1770)}
	cfg.cutie_object_role_names = ['body', 'front', 'rear']
	cfg.cutie_object_num_roles = 3
	cfg.cutie_object_only_latent_dim = 192
	cfg.latent_dim = 192
	cfg.cutie_object_regression_encoder = MODE
	cfg.cutie_object_spatial_token_enabled = False
	cfg.cutie_object_variable_graph_enabled = False
	cfg.cutie_object_belief_enabled = False
	cfg.cutie_object_belief_use_for_control = False
	cfg.cutie_object_last_valid_memory = False
	cfg.cutie_object_true_entity_enabled = False
	cfg.object_state_supervision_enabled = False
	return cfg


def build(cfg):
	torch.manual_seed(271828)
	model = WorldModel(deepcopy(cfg)).eval()
	return model, torch.get_rng_state().clone()


def agent(model):
	return SimpleNamespace(
		cfg=model.cfg, model=model, _learned_object_belief=False,
		_belief_aux_updates=0, _cutie_object_mode=True,
		_cutie_object_only=True, _cutie_hybrid=False,
		_hybrid_graph=False, _reward_graph=False,
		reset_object_belief=lambda: None,
	)


def assert_rejected(cfg, label):
	try:
		build(cfg)
	except ValueError:
		return
	raise AssertionError(f'Accepted incompatible K=3 legacy config: {label}')


def main():
	torch.set_num_threads(1)
	cfg = config()
	model, rng = build(cfg)
	assert model._cutie_regression_encoder == MODE
	assert set(model._encoder) == {'object'}
	encoder = model._encoder['object']
	assert not encoder._spatial_token and not encoder._variable_graph
	assert not model._dynamics.variable_graph
	assert not model._object_decoder.variable_graph
	assert encoder.latent_dim == 192
	assert model._object_latent_dim == 192
	assert model.cfg.latent_dim == 192
	assert encoder.role[0].weight.shape == (256, 1773)
	assert model._dynamics.transition[0].weight.shape == (256, 194)
	assert model._object_decoder.role[0].weight.shape == (256, 67)
	assert model._pi[0].weight.shape[1] == 192

	torch.manual_seed(123456)
	objects = torch.randn(4, 3, 1770, requires_grad=True)
	action = torch.randn(4, cfg.action_dim)
	z = model.encode({'object': objects}, None)
	next_z = model.next(z, action, None)
	decoded = model.decode_object(z)
	future = model.decode_object(next_z)
	assert z.shape == next_z.shape == (4, 192)
	assert decoded.shape == future.shape == (4, 3, 1770)
	loss = (
		model.object_loss(decoded, objects.detach())
		+ model.object_loss(future, objects.detach())
		+ model._pi(z).square().mean()
		+ model._reward(torch.cat([z, action], dim=-1)).square().mean()
	)
	loss.backward()
	assert torch.isfinite(loss)
	assert objects.grad is not None and torch.isfinite(objects.grad).all()
	assert torch.count_nonzero(objects.grad) > 0

	rebuilt, rebuilt_rng = build(cfg)
	assert torch.equal(rng, rebuilt_rng)
	rebuilt.load_state_dict(model.state_dict(), strict=True)

	buffer = BytesIO()
	TDMPC2.save(agent(model), buffer)
	buffer.seek(0)
	payload = torch.load(buffer, map_location='cpu', weights_only=False)
	assert payload['checkpoint_contract']['cutie_object_regression_encoder'] == MODE
	TDMPC2.load(agent(rebuilt), payload)
	missing_observation = deepcopy(payload)
	del missing_observation['checkpoint_contract']['cutie_object_observation']
	try:
		TDMPC2.load(agent(rebuilt), missing_observation)
	except RuntimeError as error:
		assert 'observation contract mismatch' in str(error)
	else:
		raise AssertionError('K=3 checkpoint without a role observation contract was accepted')

	k2_cfg = config()
	k2_cfg.cutie_object_regression_encoder = 'legacy_mlp_v1'
	k2_cfg.cutie_object_role_names = ['body', 'target']
	k2_cfg.cutie_object_num_roles = 2
	k2_cfg.cutie_object_only_latent_dim = 128
	k2_cfg.latent_dim = 128
	k2_cfg.obs_shape = {'object': (2, 1770)}
	k2, _ = build(k2_cfg)
	try:
		TDMPC2.load(agent(k2), payload)
	except RuntimeError as error:
		assert 'regression encoder checkpoint contract mismatch' in str(error)
	else:
		raise AssertionError('K=3 checkpoint was accepted by legacy_mlp_v1')

	wrong_mode = config()
	wrong_mode.cutie_object_regression_encoder = 'legacy_mlp_v1'
	assert_rejected(wrong_mode, 'legacy_mlp_v1 with three roles')
	wrong_count = config()
	wrong_count.cutie_object_num_roles = 2
	wrong_count.obs_shape = {'object': (2, 1770)}
	assert_rejected(wrong_count, 'legacy_mlp_k3_v1 with two roles')
	wrong_width = config()
	wrong_width.latent_dim = 128
	wrong_width.cutie_object_only_latent_dim = 128
	assert_rejected(wrong_width, 'legacy_mlp_k3_v1 with 128-D control')
	for key in (
		'cutie_object_spatial_token_enabled',
		'cutie_object_variable_graph_enabled',
		'cutie_object_belief_enabled',
		'cutie_object_true_entity_enabled',
		'object_state_supervision_enabled',
	):
		bad = config()
		setattr(bad, key, True)
		assert_rejected(bad, key)

	print(
		'CUTIE_LEGACY_MLP_K3_CONTRACT_OK roles=3 latent=192 '
		'nonspatial=true gradients=true checkpoint_isolated=true'
	)


if __name__ == '__main__':
	main()
