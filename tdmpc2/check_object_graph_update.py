"""One synthetic GPU update for the RGB-free ObjectGraph anchor path.

This preflight exercises ``TDMPC2.update`` with the same anchor-only
``TensorDict`` contract used by replay. It does not import the external anchor
teacher, construct DINOv2, render an environment, or query simulator state.

Run eager and compiled checks with::

	python tdmpc2/check_object_graph_update.py
	python tdmpc2/check_object_graph_update.py --compile
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('LAZY_LEGACY_OP', '0')
os.environ.setdefault('TORCHDYNAMO_INLINE_INBUILT_NN_MODULES', '1')

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from common import MODEL_SIZE
from common.parser import cfg_to_dataclass
from common.seed import set_seed
from tdmpc2 import TDMPC2


CONFIG_PATH = Path(__file__).with_name('config.yaml')
ROLE_NAMES = ('base', 'elbow', 'tip', 'goal')
EXPECTED_ACTION_ROLE_MASK = torch.tensor([0., 1., 1., 0.])


class SyntheticBuffer:
	"""Minimal replay interface that exercises ``TDMPC2.update`` itself."""

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
	# This is the real ObjectGraph replay contract: RGB is intentionally absent.
	cfg.obs_shape = {'anchor': (25,)}
	cfg.action_dim = 2
	cfg.episode_length = 500
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
	cfg.batch_size = args.batch_size
	cfg.horizon = 3
	cfg.compile = args.compile
	cfg.flat_anchor = True
	cfg.flat_anchor_mode = 'object_graph'
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
	if anchor.shape[-1] != cfg.flat_anchor_dim:
		raise AssertionError(f'Bad synthetic anchor width: {anchor.shape[-1]}.')
	obs = TensorDict(
		{'anchor': anchor},
		batch_size=(time, cfg.batch_size),
		device=device,
	)
	action = torch.empty(
		cfg.horizon, cfg.batch_size, cfg.action_dim, device=device
	).uniform_(-1., 1.)
	reward = torch.randn(cfg.horizon, cfg.batch_size, 1, device=device)
	terminated = torch.zeros_like(reward)
	return obs, action, reward, terminated


def parameter_ids(optimizer):
	return {
		id(parameter)
		for group in optimizer.param_groups
		for parameter in group['params']
	}


def check_optimizer_coverage(agent):
	model_ids = parameter_ids(agent.optim)
	pi_ids = parameter_ids(agent.pi_optim)
	modules = {
		'role_encoder': agent.model._encoder['anchor'],
		'object_graph_dynamics': agent.model._dynamics,
		'anchor_decoder': agent.model._anchor_decoder,
	}
	for name, module in modules.items():
		missing = [
			parameter_name
			for parameter_name, parameter in module.named_parameters()
			if parameter.requires_grad and id(parameter) not in model_ids
		]
		if missing:
			raise AssertionError(f'{name} parameters missing from model optimizer: {missing}')
	pi_parameters = {
		id(parameter)
		for parameter in agent.model._pi.parameters()
		if parameter.requires_grad
	}
	if not pi_parameters or not pi_parameters.issubset(pi_ids):
		raise AssertionError('Policy parameters are missing from pi_optim.')
	if pi_parameters & model_ids:
		raise AssertionError('Policy parameters must not be updated by the model optimizer.')
	if model_ids & pi_ids:
		raise AssertionError('Model and policy optimizers must have disjoint parameters.')
	return {name: len(list(module.parameters())) for name, module in modules.items()}


def snapshot_trainable(module):
	return {
		name: parameter.detach().clone()
		for name, parameter in module.named_parameters()
		if parameter.requires_grad
	}


def max_parameter_change(before, module):
	current = dict(module.named_parameters())
	return max(
		(current[name].detach() - value).abs().max().item()
		for name, value in before.items()
	)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--compile', action='store_true')
	parser.add_argument('--batch-size', type=int, default=2)
	parser.add_argument('--seed', type=int, default=1)
	args = parser.parse_args()
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required for the TD-MPC2 update preflight.')
	set_seed(args.seed)
	torch.set_float32_matmul_precision('high')
	cfg = make_config(args)
	if cfg.latent_dim != 512:
		raise AssertionError(
			f'Pre-construction model-size config must use latent 512, got {cfg.latent_dim}.'
		)
	agent = TDMPC2(cfg)
	expected_latent_dim = len(ROLE_NAMES) * cfg.flat_anchor_object_role_dim
	if expected_latent_dim != 128 or cfg.latent_dim != expected_latent_dim:
		raise AssertionError(
			'TDMPC2 did not replace the 512-D RGB state with the 128-D ObjectGraph '
			f'state: expected {expected_latent_dim}, got {cfg.latent_dim}.'
		)
	if set(agent.model._encoder.keys()) != {'anchor'}:
		raise AssertionError(
			f'ObjectGraph constructed unexpected encoders: {tuple(agent.model._encoder.keys())}.'
		)
	if agent.model._reward[0].in_features != cfg.latent_dim + cfg.action_dim:
		raise AssertionError('Reward head was not constructed for the object-only state.')
	if agent.model._pi[0].in_features != cfg.latent_dim:
		raise AssertionError('Policy head was not constructed for the object-only state.')
	device_index = agent.device.index
	if device_index is None:
		device_index = torch.cuda.current_device()
	with torch.random.fork_rng(devices=[device_index], enabled=True):
		_, pi_contract = agent.model.pi(
			torch.zeros(1, cfg.latent_dim, device=agent.device),
			None,
		)
	if pi_contract['action_prob'].device != agent.device:
		raise AssertionError(
			'Policy TensorDict leaked action_prob to '
			f"{pi_contract['action_prob'].device}; expected {agent.device}."
		)
	action_role_mask = agent.model._dynamics._action_role_mask.flatten().cpu()
	if not torch.equal(action_role_mask, EXPECTED_ACTION_ROLE_MASK):
		raise AssertionError(f'Bad action-role routing mask: {action_role_mask.tolist()}.')

	optimizer_coverage = check_optimizer_coverage(agent)
	tracked_modules = {
		'role_encoder': agent.model._encoder['anchor'],
		'object_graph_dynamics': agent.model._dynamics,
		'anchor_decoder': agent.model._anchor_decoder,
	}
	before = {
		name: snapshot_trainable(module)
		for name, module in tracked_modules.items()
	}
	obs, action, reward, terminated = fake_batch(cfg, agent.device)
	if set(obs.keys()) != {'anchor'}:
		raise AssertionError(f'Synthetic replay leaked extra observation keys: {obs.keys()}.')
	metrics = agent.update(SyntheticBuffer((obs, action, reward, terminated)))
	for key, value in metrics.items():
		if not torch.isfinite(value).all():
			raise AssertionError(f'Non-finite metric {key}: {value}')
	parameter_change = {
		name: max_parameter_change(before[name], module)
		for name, module in tracked_modules.items()
	}
	for name, change in parameter_change.items():
		if not change > 0:
			raise AssertionError(f'{name} did not update on the first optimizer step.')
		for parameter_name, parameter in tracked_modules[name].named_parameters():
			if parameter.requires_grad and not torch.isfinite(parameter).all():
				raise AssertionError(f'Non-finite parameter {name}.{parameter_name}.')
	print('OBJECT_GRAPH_UPDATE_OK', {
		'compile': args.compile,
		'preconstruction_latent_dim': 512,
		'object_graph_latent_dim': cfg.latent_dim,
		'batch_size': cfg.batch_size,
		'observation_keys': tuple(obs.keys()),
		'action_prob_device': str(pi_contract['action_prob'].device),
		'action_role_mask': action_role_mask.tolist(),
		'optimizer_tensors': optimizer_coverage,
		'parameter_max_change': parameter_change,
		'anchor_reconstruction_loss': metrics['anchor_reconstruction_loss'].item(),
		'anchor_prediction_loss': metrics['anchor_prediction_loss'].item(),
		'total_loss': metrics['total_loss'].item(),
	})


if __name__ == '__main__':
	main()
