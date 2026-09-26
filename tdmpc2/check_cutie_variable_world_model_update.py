"""One real TD-MPC2 update through a three-role variable object graph."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from check_cutie_object_only_update import make_config
from common.seed import set_seed
from tdmpc2 import TDMPC2


class Buffer:
	def __init__(self, sample):
		self.sample_value = sample

	def sample(self):
		return (*self.sample_value, None)


def _run(args, graph_path, readout):
	cfg = make_config(args)
	cfg.cutie_object_num_roles = 3
	cfg.obs_shape = {'object': (3, 1770)}
	cfg.cutie_object_spatial_token_enabled = True
	cfg.cutie_object_spatial_graph_path = str(graph_path)
	cfg.cutie_object_variable_graph_enabled = True
	cfg.cutie_object_variable_graph_max_roles = 8
	cfg.cutie_object_variable_graph_pool_tokens = 2
	cfg.cutie_object_variable_graph_readout = readout
	cfg.cutie_object_variable_graph_primary_skip_enabled = False
	cfg.cutie_object_only_latent_dim = 192 if readout == 'direct' else 128
	set_seed(cfg.seed)
	agent = TDMPC2(cfg)
	time = cfg.horizon + 1
	objects = torch.randn(time, cfg.batch_size, 3, 1770, device=agent.device)
	objects[..., 2 * 590 + 588] = 1.0
	obs = TensorDict(
		{'object': objects}, batch_size=(time, cfg.batch_size), device=agent.device,
	)
	action = torch.empty(
		cfg.horizon, cfg.batch_size, cfg.action_dim, device=agent.device,
	).uniform_(-1.0, 1.0)
	reward = torch.randn(cfg.horizon, cfg.batch_size, 1, device=agent.device)
	terminated = torch.zeros_like(reward)
	metrics = agent.update(Buffer((obs, action, reward, terminated)))
	latent = agent.model.encode(obs[0], None)
	decoded = agent.model._object_decoder(latent)
	expected = 192 if readout == 'direct' else 128
	if latent.shape != (cfg.batch_size, expected):
		raise AssertionError(tuple(latent.shape))
	if decoded.shape != (cfg.batch_size, 3, 1770):
		raise AssertionError(tuple(decoded.shape))
	if not metrics or not all(torch.isfinite(torch.as_tensor(v)) for v in metrics.values()):
		raise AssertionError(f'Non-finite update metrics: {metrics!r}')
	print(f'CUTIE_VARIABLE_WORLD_MODEL_UPDATE_OK K=3 readout={readout} latent={expected}')
	del agent, obs, objects, action, reward, terminated
	torch.cuda.empty_cache()


def main():
	args = SimpleNamespace(
		compile=False, batch_size=2, seed=271828,
		auxiliary_target='full_descriptor',
	)
	with tempfile.TemporaryDirectory() as directory:
		graph_path = Path(directory) / 'three_roles.json'
		graph_path.write_text(json.dumps({
			'format': 'support_conditioned_object_graph_v1',
			'task': 'synthetic-three-role',
			'source_roles': ['agent', 'tool', 'target'],
			'relations': [
				{'parent': 'agent', 'child': 'tool', 'type': 'revolute_joint_v1'},
				{'parent': 'tool', 'child': 'target', 'type': 'spatial_target_v1'},
			],
		}), encoding='utf-8')
		for readout in ('pool', 'direct'):
			_run(args, graph_path, readout)


if __name__ == '__main__':
	main()
