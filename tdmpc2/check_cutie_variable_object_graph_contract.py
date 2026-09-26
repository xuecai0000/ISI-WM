"""CPU contract for the variable-cardinality Cutie object graph.

The live tensor contains exactly K roles: no padding role or synthetic empty
token is admitted. Pooled readout retains fixed parameter shapes, while direct
readout deliberately instantiates a K*64 controller state for each task.
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from common import layers


class Config(SimpleNamespace):
	def get(self, key, default=None):
		return getattr(self, key, default)


def _graph(path: Path, roles: tuple[str, ...]):
	relations = [
		{
			'parent': roles[index],
			'child': roles[index + 1],
			'type': 'revolute_joint_v1' if index % 2 == 0 else 'spatial_target_v1',
		}
		for index in range(len(roles) - 1)
	]
	payload = {
		'format': 'support_conditioned_object_graph_v1',
		'task': f'synthetic-k{len(roles)}',
		'source_roles': list(roles),
		'relations': relations,
	}
	path.write_text(json.dumps(payload), encoding='utf-8')


def _config(path: Path, count: int, primary_skip=False, readout='pool'):
	return Config(
		multitask=False,
		action_dim=3,
		simnorm_dim=8,
		flat_anchor_mode='cutie_object_only',
		cutie_object_num_roles=count,
		cutie_object_input_dim=1770,
		cutie_object_observation_variant='full',
		cutie_object_role_dim=64,
		cutie_object_hidden_dim=256,
		cutie_object_only_latent_dim=count * 64 if readout == 'direct' else 128,
		cutie_object_spatial_token_enabled=True,
		cutie_object_spatial_graph_path=str(path),
		cutie_object_spatial_token_dim=64,
		cutie_object_spatial_num_heads=4,
		cutie_object_spatial_num_layers=2,
		cutie_object_variable_graph_enabled=True,
		cutie_object_variable_graph_max_roles=8,
		cutie_object_variable_graph_pool_tokens=2,
		cutie_object_variable_graph_primary_skip_enabled=primary_skip,
		cutie_object_variable_graph_readout=readout,
	)


def _parameter_contract(*modules):
	return tuple(
		(name, tuple(parameter.shape))
		for prefix, module in zip(('encoder', 'dynamics', 'decoder'), modules)
		for name, parameter in module.named_parameters(prefix=prefix)
	)


def main():
	contracts = {False: [], True: []}
	with tempfile.TemporaryDirectory() as directory:
		for primary_skip in (False, True):
			for count in (1, 2, 3, 5):
				roles = tuple(f'role_{index}' for index in range(count))
				path = Path(directory) / f'graph_k{count}.json'
				_graph(path, roles)
				cfg = _config(path, count, primary_skip=primary_skip)
				encoder = layers.CutieObjectEncoder(cfg)
				dynamics = layers.CutieObjectDynamics(cfg)
				decoder = layers.CutieObjectDecoder(cfg)
				contracts[primary_skip].append(
					_parameter_contract(encoder, dynamics, decoder)
				)

				objects = torch.randn(2, count, 1770, requires_grad=True)
				latent = encoder(objects)
				if latent.shape != (2, 128):
					raise AssertionError(f'K={count} latent shape {tuple(latent.shape)}')
				next_latent = dynamics(latent, torch.randn(2, 3))
				reconstruction = decoder(next_latent)
				if reconstruction.shape != (2, count, 1770):
					raise AssertionError(
						f'K={count} reconstruction shape {tuple(reconstruction.shape)}'
					)
				(reconstruction.square().mean() + next_latent.square().mean()).backward()
				if objects.grad is None or not torch.isfinite(objects.grad).all():
					raise AssertionError(f'K={count} has no finite end-to-end gradient.')

		for count in (1, 2, 3, 5):
			roles = tuple(f'direct_role_{index}' for index in range(count))
			path = Path(directory) / f'direct_graph_k{count}.json'
			_graph(path, roles)
			cfg = _config(path, count, readout='direct')
			encoder = layers.CutieObjectEncoder(cfg)
			dynamics = layers.CutieObjectDynamics(cfg)
			decoder = layers.CutieObjectDecoder(cfg)
			objects = torch.randn(2, count, 1770, requires_grad=True)
			latent = encoder(objects)
			if latent.shape != (2, count * 64):
				raise AssertionError(
					f'direct K={count} latent shape {tuple(latent.shape)}'
				)
			next_latent = dynamics(latent, torch.randn(2, 3))
			reconstruction = decoder(next_latent)
			if reconstruction.shape != (2, count, 1770):
				raise AssertionError(
					f'direct K={count} reconstruction {tuple(reconstruction.shape)}'
				)
			(reconstruction.square().mean() + next_latent.square().mean()).backward()
			if objects.grad is None or not torch.isfinite(objects.grad).all():
				raise AssertionError(f'direct K={count} has no finite gradient.')

	for primary_skip, mode_contracts in contracts.items():
		if any(contract != mode_contracts[0] for contract in mode_contracts[1:]):
			raise AssertionError(
				f'Trainable parameter names/shapes changed with K; primary_skip={primary_skip}.'
			)
	print('CUTIE_VARIABLE_OBJECT_GRAPH_CONTRACT_OK K=1,2,3,5 pool=128 direct=Kx64 no_padding=true')


if __name__ == '__main__':
	main()
