"""Real CUDA update and checkpoint contract for safe Cutie-proprio fusion."""

from copy import deepcopy
from pathlib import Path
import tempfile

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from common import MODEL_SIZE
from common import cutie_proprio as contract
from common.parser import cfg_to_dataclass
from common.seed import set_seed
from tdmpc2 import TDMPC2


CONFIG_PATH = Path(__file__).with_name('config.yaml')


def make_config(mode):
	cfg = OmegaConf.load(CONFIG_PATH)
	cfg.task = contract.TASK
	cfg.obs = 'rgb'
	cfg.model_size = 5
	for key, value in MODEL_SIZE[5].items():
		cfg[key] = value
	cfg.multitask = False
	cfg.tasks = [cfg.task]
	cfg.task_dim = 0
	cfg.action_dim = 1
	cfg.episode_length = 500
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
	cfg.batch_size = 2
	cfg.horizon = 3
	cfg.compile = False
	cfg.flat_anchor = True
	cfg.flat_anchor_mode = 'cutie_object_only'
	cfg.obs_shape = {'object': (2, 1774)}
	cfg.cutie_object_observation_variant = contract.VARIANT
	cfg.cutie_object_frame_schema = contract.FRAME_SCHEMA
	cfg.cutie_object_role_names = list(contract.ROLE_NAMES)
	cfg.cutie_object_support_schema = 'generic_indexed_v1'
	cfg.cutie_object_allow_simulator_support = True
	cfg.cutie_object_allow_simulator_runtime = False
	cfg.cutie_object_allow_simulator_kinematics_runtime = mode != 'cutie_only'
	cfg.cutie_object_repo = '/repo'
	cfg.cutie_object_checkpoint = '/checkpoint'
	cfg.cutie_object_support_path = '/support'
	cfg.cutie_object_num_roles = 2
	cfg.cutie_object_frame_dim = 1774
	cfg.cutie_object_stack_frames = 1
	cfg.cutie_object_input_dim = 1774
	cfg.cutie_object_role_dim = 64
	cfg.cutie_object_only_latent_dim = 128
	cfg.cutie_object_auxiliary_target = 'full_descriptor'
	cfg.cutie_object_last_valid_memory = False
	cfg.cutie_object_policy_burst_plan = None
	cfg.cutie_object_belief_enabled = False
	cfg.cutie_object_belief_use_for_control = False
	cfg.cutie_proprio_mode = mode
	cfg.cutie_proprio_velocity_scale = 10.0
	cfg.enable_wandb = False
	cfg.save_video = False
	cfg.seed = 271828
	return cfg_to_dataclass(cfg)


def sample(cfg):
	time = cfg.horizon + 1
	objects = torch.randn(time, cfg.batch_size, 2, 1774, device='cuda:0')
	objects[..., 1768] = 1.0
	objects[..., 1773] = 1.0
	obs = TensorDict({'object': objects}, batch_size=(time, cfg.batch_size), device='cuda:0')
	action = torch.empty(cfg.horizon, cfg.batch_size, 1, device='cuda:0').uniform_(-1, 1)
	reward = torch.randn(cfg.horizon, cfg.batch_size, 1, device='cuda:0')
	terminated = torch.zeros_like(reward)
	return obs, action, reward, terminated


class Buffer:
	def __init__(self, value):
		self.value = value

	def sample(self):
		return (*self.value, None)


def exact_initial_fusion():
	objects = torch.randn(3, 2, 1774, device='cuda:0')
	outputs = {}
	states = {}
	for mode in ('proprio_only', 'fusion'):
		set_seed(271828)
		agent = TDMPC2(make_config(mode))
		outputs[mode] = agent.model.encode({'object': objects}, None).detach().cpu()
		states[mode] = {
			name: value.detach().cpu().clone()
			for name, value in agent.model._encoder['object'].state_dict().items()
		}
		del agent
	if states['proprio_only'].keys() != states['fusion'].keys():
		raise AssertionError('Fusion and proprio encoder keys differ.')
	for name in states['proprio_only']:
		if not torch.equal(states['proprio_only'][name], states['fusion'][name]):
			raise AssertionError(f'Fusion and proprio initialization differ at {name}.')
	if not torch.equal(outputs['proprio_only'], outputs['fusion']):
		raise AssertionError('Fusion does not initialize exactly as proprio-only.')
	return tuple(outputs['fusion'].shape)


def main():
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required for Cutie-proprio update contract.')
	initial_shape = exact_initial_fusion()
	results = {}
	for mode in contract.MODES:
		set_seed(271828)
		cfg = make_config(mode)
		agent = TDMPC2(cfg)
		if cfg.latent_dim != 128 or set(agent.model._encoder) != {'object'}:
			raise AssertionError(f'{mode}: controller is not object-only 128-D.')
		before = {
			name: value.detach().clone()
			for name, value in agent.model.named_parameters()
			if value.requires_grad
		}
		metrics = agent.update(Buffer(sample(cfg)))
		if not metrics or any(
			not torch.isfinite(value).all() for value in metrics.values()
			if torch.is_tensor(value)
		):
			raise AssertionError(f'{mode}: update emitted non-finite metrics.')
		changed = sum(
			int(not torch.equal(before[name], value.detach()))
			for name, value in agent.model.named_parameters()
			if name in before
		)
		if changed < 1:
			raise AssertionError(f'{mode}: update changed no trainable parameter.')
		with tempfile.TemporaryDirectory(prefix='cutie_proprio_update_') as directory:
			path = Path(directory) / 'agent.pt'
			agent.save(path)
			payload = torch.load(path, map_location='cpu', weights_only=False)
			expected = contract.observation_contract(cfg)
			if payload['checkpoint_contract'].get('cutie_proprio_observation') != expected:
				raise AssertionError(f'{mode}: checkpoint omitted Cutie-proprio contract.')
			reloaded = TDMPC2(make_config(mode))
			reloaded.load(path)
			foreign_mode = 'fusion' if mode != 'fusion' else 'proprio_only'
			foreign = TDMPC2(make_config(foreign_mode))
			try:
				foreign.load(deepcopy(payload))
			except RuntimeError:
				pass
			else:
				raise AssertionError(f'{foreign_mode} accepted a {mode} checkpoint.')
		results[mode] = {'changed_parameter_tensors': changed}
		del agent
		torch.cuda.empty_cache()
	print('CUTIE_PROPRIO_UPDATE_OK', {
		'device': torch.cuda.get_device_name(0),
		'initial_fusion_shape': initial_shape,
		'exact_proprio_initialization': True,
		'modes': results,
	})


if __name__ == '__main__':
	main()
