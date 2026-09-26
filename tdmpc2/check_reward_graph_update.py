"""One real TensorDict update for reward-only RewardGraph TD-MPC2.

Run both paths before training::

	python tdmpc2/check_reward_graph_update.py
	python tdmpc2/check_reward_graph_update.py --compile
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('LAZY_LEGACY_OP', '0')
os.environ.setdefault('TORCHDYNAMO_INLINE_INBUILT_NN_MODULES', '1')

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tensordict import TensorDict

from common import MODEL_SIZE
from common.parser import cfg_to_dataclass
from common.seed import set_seed
from tdmpc2 import TDMPC2


CONFIG_PATH = Path(__file__).with_name('config.yaml')
SCENE_DIM = 512
GRAPH_DIM = 128
JOINT_DIM = SCENE_DIM + GRAPH_DIM
ROLE_NAMES = ('base', 'elbow', 'tip', 'goal')
ACTION_ROLE_MASK = torch.tensor([0., 1., 1., 0.])


class SyntheticBuffer:
	def __init__(self, sample):
		self._sample = sample

	def sample(self):
		return (*self._sample, None)


def make_config(args):
	cfg = OmegaConf.load(CONFIG_PATH)
	cfg.task = 'reacher-easy'
	cfg.obs = 'rgb'
	cfg.model_size = 5
	for key, value in MODEL_SIZE[5].items():
		cfg[key] = value
	cfg.multitask = False
	cfg.tasks = [cfg.task]
	cfg.task_dim = 0
	cfg.obs_shape = {'rgb': (9, 64, 64), 'anchor': (25,)}
	cfg.action_dim = 2
	cfg.episode_length = 500
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
	cfg.batch_size = args.batch_size
	cfg.horizon = 3
	cfg.compile = args.compile
	cfg.compile_fallback_random = args.compile
	cfg.flat_anchor = True
	cfg.flat_anchor_mode = 'reward_graph'
	cfg.enable_wandb = False
	cfg.save_video = False
	cfg.seed = args.seed
	return cfg_to_dataclass(cfg)


def fake_batch(cfg, device):
	time = cfg.horizon + 1
	positions = torch.empty(
		time, cfg.batch_size, len(ROLE_NAMES), 2, device=device
	).uniform_(-0.75, 0.75)
	edges = positions[..., 1:, :] - positions[..., :-1, :]
	velocities = torch.zeros(
		time, cfg.batch_size, len(ROLE_NAMES) - 1, 2, device=device
	)
	velocities[1:] = positions[1:, :, 1:, :] - positions[:-1, :, 1:, :]
	confidence = torch.full(
		(time, cfg.batch_size, len(ROLE_NAMES)), 0.9, device=device
	)
	fallback = torch.zeros(time, cfg.batch_size, 1, device=device)
	anchor = torch.cat([
		positions.flatten(-2),
		edges.flatten(-2),
		velocities.flatten(-2),
		confidence,
		fallback,
	], dim=-1)
	rgb = torch.randint(
		0, 256, (time, cfg.batch_size, 9, 64, 64),
		dtype=torch.uint8, device=device,
	)
	obs = TensorDict(
		{'rgb': rgb, 'anchor': anchor},
		batch_size=(time, cfg.batch_size),
		device=device,
	)
	action = torch.empty(
		cfg.horizon, cfg.batch_size, cfg.action_dim, device=device
	).uniform_(-1., 1.)
	reward = torch.randn(cfg.horizon, cfg.batch_size, 1, device=device)
	terminated = torch.zeros_like(reward)
	return obs, action, reward, terminated


def optimizer_parameter_ids(optimizer):
	all_ids = [
		id(parameter)
		for group in optimizer.param_groups
		for parameter in group['params']
	]
	if len(all_ids) != len(set(all_ids)):
		raise AssertionError('An optimizer contains a duplicated parameter.')
	return set(all_ids)


def module_parameter_ids(module):
	return {
		id(parameter) for parameter in module.parameters() if parameter.requires_grad
	}


def parameter_ids(parameters):
	return {id(parameter) for parameter in parameters if parameter.requires_grad}


def check_optimizer_coverage(agent):
	model_ids = optimizer_parameter_ids(agent.optim)
	pi_ids = optimizer_parameter_ids(agent.pi_optim)
	if model_ids & pi_ids:
		raise AssertionError('Model and policy optimizers overlap.')

	model_modules = {
		'role_encoder': agent.model._encoder['anchor'],
		'graph_dynamics': agent.model._graph_dynamics,
		'anchor_decoder': agent.model._anchor_decoder,
		'graph_reward': agent.model._graph_reward,
	}
	for name, module in model_modules.items():
		missing = module_parameter_ids(module) - model_ids
		if missing:
			raise AssertionError(f'{name} is missing from the model optimizer.')

	official_pi_ids = module_parameter_ids(agent.model._pi)
	if pi_ids != official_pi_ids:
		raise AssertionError(
			'RewardGraph pi_optim must contain exactly the official policy prior.'
		)

	forbidden = tuple(
		name for name in (
			'_hybrid_pi', '_hybrid_q', '_detach_hybrid_q',
			'_target_hybrid_q', '_graph_pi', '_graph_q',
		) if hasattr(agent.model, name)
	)
	if forbidden:
		raise AssertionError(f'RewardGraph created forbidden Q/pi branches: {forbidden}')

	if not hasattr(agent, '_graph_aux_params') or not hasattr(
		agent, '_graph_reward_params'
	):
		raise AssertionError('RewardGraph is missing separate gradient partitions.')
	aux_ids = parameter_ids(agent._graph_aux_params)
	reward_ids = parameter_ids(agent._graph_reward_params)
	official_ids = parameter_ids(agent._official_world_params)
	if aux_ids & reward_ids or aux_ids & official_ids or reward_ids & official_ids:
		raise AssertionError('Official/graph-aux/graph-reward gradient partitions overlap.')
	expected_aux = set().union(*(
		module_parameter_ids(agent.model._encoder['anchor']),
		module_parameter_ids(agent.model._graph_dynamics),
		module_parameter_ids(agent.model._anchor_decoder),
	))
	if aux_ids != expected_aux:
		raise AssertionError('Graph auxiliary gradient partition has wrong coverage.')
	if reward_ids != module_parameter_ids(agent.model._graph_reward):
		raise AssertionError('Graph reward gradient partition has wrong coverage.')
	if not (aux_ids | reward_ids) <= model_ids:
		raise AssertionError('A graph gradient partition is absent from model optimizer.')
	if official_ids | aux_ids | reward_ids != model_ids:
		raise AssertionError(
			'Official/graph-aux/graph-reward partitions do not exactly cover '
			'the model optimizer.'
		)

	return {
		name: len(module_parameter_ids(module))
		for name, module in {
			**model_modules,
			'official_pi': agent.model._pi,
		}.items()
	}


def final_linear(module):
	linears = [child for child in module.modules() if isinstance(child, nn.Linear)]
	if not linears:
		raise AssertionError('RewardGraph correction contains no Linear.')
	return linears[-1]


def graph_reward_contract(agent):
	"""Check graph-only input and scene-gradient isolation non-vacuously."""
	model = agent.model
	module = model._graph_reward
	linears = [child for child in module.modules() if isinstance(child, nn.Linear)]
	if linears[0].in_features != GRAPH_DIM + agent.cfg.action_dim:
		raise AssertionError(
			'Graph reward correction must consume only graph(128)+action.'
		)
	if linears[0].out_features != agent.cfg.flat_anchor_reward_graph_hidden_dim:
		raise AssertionError('Graph reward correction ignored its explicit hidden width.')
	output = final_linear(module)
	if output.out_features != max(agent.cfg.num_bins, 1):
		raise AssertionError('Graph reward correction has wrong output width.')
	if float(output.weight.detach().abs().max()) != 0.:
		raise AssertionError('Graph reward output weight is not zero initialized.')
	if output.bias is not None and float(output.bias.detach().abs().max()) != 0.:
		raise AssertionError('Graph reward output bias is not zero initialized.')

	weight_before = output.weight.detach().clone()
	bias_before = output.bias.detach().clone() if output.bias is not None else None
	with torch.no_grad():
		output.weight.fill_(0.01)
		if output.bias is not None:
			output.bias.zero_()
	try:
		z = torch.linspace(
			-0.9, 0.9, 2 * JOINT_DIM, device=agent.device
		).reshape(2, JOINT_DIM).requires_grad_(True)
		action = torch.tensor(
			[[0.2, -0.4], [-0.3, 0.5]], device=agent.device
		)
		official = model._reward(torch.cat([z[..., :SCENE_DIM], action], dim=-1))
		correction = model.reward(z, action, None) - official
		gradient = torch.autograd.grad(correction.sum(), z)[0]
		scene_gradient = float(gradient[..., :SCENE_DIM].abs().max())
		graph_gradient = float(gradient[..., SCENE_DIM:].abs().max())
		if scene_gradient != 0. or not graph_gradient > 0.:
			raise AssertionError(
				'Graph reward leaked into scene gradient or ignored graph state: '
				f'scene={scene_gradient}, graph={graph_gradient}'
			)
	finally:
		with torch.no_grad():
			output.weight.copy_(weight_before)
			if output.bias is not None:
				output.bias.copy_(bias_before)

	zero_probe = module(torch.zeros(
		2, GRAPH_DIM + agent.cfg.action_dim, device=agent.device
	))
	if torch.count_nonzero(zero_probe) != 0:
		raise AssertionError('Graph reward correction is not exact zero after restore.')
	return {
		'input_dim': linears[0].in_features,
		'hidden_dim': linears[0].out_features,
		'scene_gradient': scene_gradient,
		'graph_gradient': graph_gradient,
	}


def snapshot(module):
	return {
		name: parameter.detach().clone()
		for name, parameter in module.named_parameters()
		if parameter.requires_grad
	}


def max_change(before, module):
	current = dict(module.named_parameters())
	return max(
		(current[name].detach() - value).abs().max().item()
		for name, value in before.items()
	)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--compile', action='store_true')
	parser.add_argument('--batch-size', type=int, default=2)
	parser.add_argument('--seed', type=int, default=2)
	args = parser.parse_args()
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required for the RewardGraph update preflight.')
	set_seed(args.seed)
	torch.set_float32_matmul_precision('high')
	cfg = make_config(args)
	if cfg.latent_dim != SCENE_DIM:
		raise AssertionError('RewardGraph must start from official 512-D config.')
	agent = TDMPC2(cfg)
	if cfg.latent_dim != JOINT_DIM:
		raise AssertionError(f'RewardGraph did not create 640-D state: {cfg.latent_dim}.')
	if agent.model._dynamics[0].in_features != SCENE_DIM + cfg.action_dim:
		raise AssertionError('Official RGB dynamics input shape changed.')
	if agent.model._dynamics[-1].out_features != SCENE_DIM:
		raise AssertionError('Official RGB dynamics output shape changed.')
	if agent.model._reward[0].in_features != SCENE_DIM + cfg.action_dim:
		raise AssertionError('Official reward head input shape changed.')
	if agent.model._pi[0].in_features != SCENE_DIM:
		raise AssertionError('Official policy input shape changed.')
	action_mask = agent.model._graph_dynamics._action_role_mask.flatten().cpu()
	if not torch.equal(action_mask, ACTION_ROLE_MASK):
		raise AssertionError(f'Bad action-role mask: {action_mask.tolist()}.')

	reward_contract = graph_reward_contract(agent)
	coverage = check_optimizer_coverage(agent)
	tracked = {
		'role_encoder': agent.model._encoder['anchor'],
		'graph_dynamics': agent.model._graph_dynamics,
		'anchor_decoder': agent.model._anchor_decoder,
		'graph_reward': agent.model._graph_reward,
	}
	before = {name: snapshot(module) for name, module in tracked.items()}

	obs, action, reward, terminated = fake_batch(cfg, agent.device)
	if set(obs.keys()) != {'rgb', 'anchor'} or obs.batch_size != torch.Size([
		cfg.horizon + 1, cfg.batch_size
	]):
		raise AssertionError('Synthetic replay does not match real TensorDict contract.')
	metrics = agent.update(SyntheticBuffer((obs, action, reward, terminated)))
	for key, value in metrics.items():
		if value.device != agent.device or not torch.isfinite(value).all():
			raise AssertionError(
				f'Bad metric {key}: device={value.device}, value={value}.'
			)
	for key in (
		'rgb_consistency_loss', 'graph_consistency_loss',
		'anchor_reconstruction_loss', 'anchor_prediction_loss',
	):
		if key not in metrics or not metrics[key] > 0:
			raise AssertionError(f'Missing or zero separated loss: {key}.')
	for key in ('official_grad_norm', 'graph_aux_grad_norm', 'graph_reward_grad_norm'):
		if key not in metrics or not torch.isfinite(metrics[key]):
			raise AssertionError(f'Missing/non-finite separated gradient metric: {key}.')

	expected_total = (
		cfg.consistency_coef * metrics['rgb_consistency_loss']
		+ cfg.flat_anchor_graph_consistency_coef * metrics['graph_consistency_loss']
		+ cfg.reward_coef * metrics['reward_loss']
		+ cfg.termination_coef * metrics['termination_loss']
		+ cfg.value_coef * metrics['value_loss']
		+ cfg.flat_anchor_reconstruction_coef * metrics['anchor_reconstruction_loss']
		+ cfg.flat_anchor_prediction_coef * metrics['anchor_prediction_loss']
	)
	torch.testing.assert_close(
		metrics['total_loss'], expected_total, rtol=1e-5, atol=1e-5
	)

	changes = {name: max_change(before[name], module) for name, module in tracked.items()}
	for name, change in changes.items():
		if not change > 0.:
			raise AssertionError(f'{name} did not change on first real update.')
		for parameter_name, parameter in tracked[name].named_parameters():
			if parameter.requires_grad and not torch.isfinite(parameter).all():
				raise AssertionError(f'Non-finite parameter {name}.{parameter_name}.')

	print('REWARD_GRAPH_UPDATE_OK', {
		'compile': args.compile,
		'compile_fallback_random': args.compile,
		'joint_latent_dim': cfg.latent_dim,
		'batch_size': cfg.batch_size,
		'observation_keys': tuple(obs.keys()),
		'action_role_mask': action_mask.tolist(),
		'optimizer_tensors': coverage,
		'reward_contract': reward_contract,
		'parameter_max_change': changes,
		'rgb_consistency_loss': metrics['rgb_consistency_loss'].item(),
		'graph_consistency_loss': metrics['graph_consistency_loss'].item(),
		'anchor_reconstruction_loss': metrics['anchor_reconstruction_loss'].item(),
		'anchor_prediction_loss': metrics['anchor_prediction_loss'].item(),
		'total_loss': metrics['total_loss'].item(),
	})


if __name__ == '__main__':
	main()
