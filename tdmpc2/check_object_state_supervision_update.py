"""CUDA paired-init, isolated-label, real-update and checkpoint contracts."""

from pathlib import Path
import tempfile

import torch
from tensordict import TensorDict

from check_cutie_proprio_update import make_config as base_config
from common import object_state_supervision as supervision
from common.seed import set_seed
from tdmpc2 import TDMPC2


def make_config(coef, task='acrobot-swingup'):
	cfg = base_config('cutie_only')
	cfg.task = task
	cfg.tasks = [task]
	cfg.action_dim = 2 if task == 'reacher-visual-small' else 1
	cfg.obs_shape = {'object': (2, 1770)}
	cfg.cutie_object_observation_variant = 'full'
	cfg.cutie_object_frame_schema = 'cutie_query_mask_status_v1'
	cfg.cutie_object_frame_dim = 590
	cfg.cutie_object_stack_frames = 3
	cfg.cutie_object_input_dim = 1770
	cfg.cutie_object_auxiliary_target = 'geometry_status_full_denominator'
	cfg.cutie_object_spatial_token_enabled = True
	cfg.cutie_object_spatial_graph_path = str(Path(__file__).parent / 'object_graphs' / {
		'acrobot-swingup': 'acrobot_swingup.json',
		'cartpole-swingup': 'cartpole_swingup.json',
		'reacher-visual-small': 'reacher_visual_small.json',
	}[task])
	cfg.cutie_object_role_names = {
		'acrobot-swingup': ['upper_arm', 'lower_arm'],
		'cartpole-swingup': ['cart', 'pole'],
		'reacher-visual-small': ['whole_arm', 'goal'],
	}[task]
	cfg.cutie_object_true_entity_enabled = task == 'acrobot-swingup'
	cfg.object_state_supervision_enabled = True
	cfg.object_state_supervision_collect_labels = True
	cfg.object_state_supervision_coef = coef
	return cfg


class Batch:
	def __init__(self, cfg):
		t, b = cfg.horizon + 1, cfg.batch_size
		objects = torch.randn(t, b, 2, 1770, device='cuda:0')
		for frame in range(3):
			objects[..., frame * 590 + 588] = 1.
		self.obs = TensorDict({'object': objects}, batch_size=(t, b), device='cuda:0')
		self.action = torch.rand(t - 1, b, cfg.action_dim, device='cuda:0') * 2 - 1
		self.reward = torch.rand(t - 1, b, 1, device='cuda:0')
		self.target = torch.randn(t, b, supervision.target_dim(cfg), device='cuda:0')

	def sample(self):
		return self.obs, self.action, self.reward, torch.zeros_like(self.reward), None, self.target


def main():
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required.')
	torch.set_num_threads(2)
	for task in ('acrobot-swingup', 'cartpole-swingup', 'reacher-visual-small'):
		initial = None
		for coef in (0.0, 0.1):
			set_seed(271828)
			cfg = make_config(coef, task)
			agent = TDMPC2(cfg)
			state = {k: v.detach().cpu().clone() for k, v in agent.model.state_dict().items() if torch.is_tensor(v)}
			if initial is None:
				initial = state
			else:
				assert initial.keys() == state.keys()
				assert all(torch.equal(v, state[k]) for k, v in initial.items()), 'Paired initialization differs'
			assert set(agent.model._encoder) == {'object'}, 'RGB/state encoder leaked into model'
			batch = Batch(cfg)
			with torch.no_grad():
				encoded = agent.model.encode(batch.obs[0], None).clone()
				batch.target.add_(3.)
				assert torch.equal(encoded, agent.model.encode(batch.obs[0], None)), 'Labels affected inference'
			head_before = {k: v.clone() for k, v in agent.model._state_supervision_head.state_dict().items()}
			if coef > 0:
				z = agent.model.encode(batch.obs[0], None)
				future = agent.model.next(z, batch.action[0], None)
				aux_only = (agent.model.decode_supervised_state(z) - batch.target[0]).square().mean()
				aux_only = aux_only + (agent.model.decode_supervised_state(future) - batch.target[1]).square().mean()
				aux_only.backward()
				for module in (agent.model._encoder, agent.model._dynamics):
					assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in module.parameters()), 'Auxiliary loss was detached from the world model'
				agent.optim.zero_grad(set_to_none=True)
			metrics = agent.update(batch)
			assert all(torch.isfinite(v).all() for v in metrics.values())
			assert float(metrics['state_supervision_loss']) > 0
			head_changed = any(not torch.equal(v, agent.model._state_supervision_head.state_dict()[k]) for k, v in head_before.items())
			assert head_changed == (coef > 0), 'Auxiliary coefficient did not control head update'
			assert (float(metrics['state_supervision_weighted']) > 0) == (coef > 0)
			with tempfile.TemporaryDirectory() as directory:
				checkpoint = Path(directory) / 'test.pt'
				agent.save(checkpoint)
				cfg.object_state_supervision_collect_labels = False
				agent.load(checkpoint)
				cfg.object_state_supervision_coef = 0.2
				try:
					agent.load(checkpoint)
				except RuntimeError:
					pass
				else:
					raise AssertionError('Mismatched supervision contract was accepted')
			print('STATE_SUPERVISION_UPDATE_OK', task, coef, float(metrics['state_supervision_loss']), flush=True)
			del agent, batch
			torch.cuda.empty_cache()
	print('OBJECT_STATE_SUPERVISION_CUDA_CONTRACT_OK')


if __name__ == '__main__':
	main()
