"""Dependency-light contract for held-out evaluation of legacy K=3 runs."""

from copy import deepcopy

from tools.evaluate_cutie_multitask_checkpoint import _legacy_mlp_k3_latent


def exact_k3():
	return {
		'cutie_object_regression_encoder': 'legacy_mlp_k3_v1',
		'cutie_object_num_roles': 3,
		'cutie_object_role_dim': 64,
		'cutie_object_frame_dim': 590,
		'cutie_object_stack_frames': 3,
		'cutie_object_input_dim': 1770,
		'cutie_object_only_latent_dim': 192,
		'cutie_object_observation_variant': 'full',
		'cutie_object_spatial_token_enabled': False,
		'cutie_object_variable_graph_enabled': False,
	}


def main():
	base = exact_k3()
	assert _legacy_mlp_k3_latent(base, 'cutie_object_only') == 192
	assert _legacy_mlp_k3_latent(base, 'rgb') is None
	legacy = deepcopy(base)
	legacy['cutie_object_regression_encoder'] = 'legacy_mlp_v1'
	assert _legacy_mlp_k3_latent(legacy, 'cutie_object_only') is None

	mutations = {
		'roles': ('cutie_object_num_roles', 2),
		'width': ('cutie_object_only_latent_dim', 128),
		'spatial': ('cutie_object_spatial_token_enabled', True),
		'variable': ('cutie_object_variable_graph_enabled', True),
	}
	for label, (key, value) in mutations.items():
		bad = deepcopy(base)
		bad[key] = value
		try:
			_legacy_mlp_k3_latent(bad, 'cutie_object_only')
		except ValueError:
			pass
		else:
			raise AssertionError(f'K3 evaluator accepted invalid {label}.')
	print('CUTIE_K3_EVALUATOR_CONTRACT_OK latent=192 k2_unchanged=true')


if __name__ == '__main__':
	main()
