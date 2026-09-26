"""One real synthetic GPU update for lightweight HybridGraph TD-MPC2.

Run both paths before training::

	python tdmpc2/check_hybrid_graph_update.py
	python tdmpc2/check_hybrid_graph_update.py --compile
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
	cfg.flat_anchor_mode = 'hybrid_graph'
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
		positions.flatten(-2), edges.flatten(-2), velocities.flatten(-2),
		confidence, fallback,
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


def check_optimizer_coverage(agent):
	model_ids = optimizer_parameter_ids(agent.optim)
	pi_ids = optimizer_parameter_ids(agent.pi_optim)
	if model_ids & pi_ids:
		raise AssertionError('Model and policy optimizers overlap.')
	model_modules = {
		'role_encoder': agent.model._encoder['anchor'],
		'graph_dynamics': agent.model._graph_dynamics,
		'anchor_decoder': agent.model._anchor_decoder,
		'reward_correction': agent.model._hybrid_reward,
		'q_correction': agent.model._hybrid_q,
	}
	for name, module in model_modules.items():
		missing = module_parameter_ids(module) - model_ids
		if missing:
			raise AssertionError(f'{name} is missing from the model optimizer.')
	pi_modules = {
		'official_pi': agent.model._pi,
		'pi_correction': agent.model._hybrid_pi,
	}
	for name, module in pi_modules.items():
		missing = module_parameter_ids(module) - pi_ids
		if missing:
			raise AssertionError(f'{name} is missing from pi_optim.')
	# Detach params intentionally alias online Q storage in official TD-MPC2;
	# only the independently cloned target parameters must stay out of optimizers.
	for name, module in {
		'target_q_correction': agent.model._target_hybrid_q_params,
	}.items():
		if module_parameter_ids(module) & (model_ids | pi_ids):
			raise AssertionError(f'{name} leaked into an optimizer.')
	return {
		name: len(module_parameter_ids(module))
		for name, module in {**model_modules, **pi_modules}.items()
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


def tensor_state(module):
	return {
		name: value.detach().clone()
		for name, value in module.state_dict().items()
		if torch.is_tensor(value) and value.is_floating_point()
	}


def check_polyak(before, online_after, target_after, tau):
	if set(before) != set(online_after) or set(before) != set(target_after):
		raise AssertionError('Hybrid target-Q correction state keys do not match online.')
	max_error = 0.
	for key in before:
		expected = torch.lerp(before[key], online_after[key], tau)
		error = float((target_after[key] - expected).abs().max())
		max_error = max(max_error, error)
		torch.testing.assert_close(target_after[key], expected, rtol=0., atol=1e-7)
	return max_error


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--compile', action='store_true')
	parser.add_argument('--batch-size', type=int, default=2)
	parser.add_argument('--seed', type=int, default=2)
	args = parser.parse_args()
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required for the HybridGraph update preflight.')
	set_seed(args.seed)
	torch.set_float32_matmul_precision('high')
	cfg = make_config(args)
	if cfg.latent_dim != 512:
		raise AssertionError('HybridGraph must start from the official 512-D config.')
	agent = TDMPC2(cfg)
	if cfg.latent_dim != cfg.flat_anchor_hybrid_joint_dim or cfg.latent_dim != 640:
		raise AssertionError(f'HybridGraph did not create the 640-D joint state: {cfg.latent_dim}.')
	if agent.model._dynamics[0].in_features != 512 + cfg.action_dim:
		raise AssertionError('Official RGB dynamics input shape changed.')
	if agent.model._dynamics[-1].out_features != 512:
		raise AssertionError('Official RGB dynamics output shape changed.')
	if agent.model._reward[0].in_features != 512 + cfg.action_dim:
		raise AssertionError('Official reward head input shape changed.')
	if agent.model._pi[0].in_features != 512:
		raise AssertionError('Official policy head input shape changed.')
	if agent.model._hybrid_reward.output.weight.abs().max() != 0:
		raise AssertionError('Reward correction is not zero initialized.')
	if agent.model._hybrid_pi.output.weight.abs().max() != 0:
		raise AssertionError('Policy correction is not zero initialized.')
	if agent.model._hybrid_q.params['output', 'weight'].abs().max() != 0:
		raise AssertionError('Q correction is not zero initialized.')
	action_mask = agent.model._graph_dynamics._action_role_mask.flatten().cpu()
	if not torch.equal(action_mask, ACTION_ROLE_MASK):
		raise AssertionError(f'Bad action-role mask: {action_mask.tolist()}.')

	coverage = check_optimizer_coverage(agent)
	tracked = {
		'rgb_encoder': agent.model._encoder['rgb'],
		'rgb_dynamics': agent.model._dynamics,
		'role_encoder': agent.model._encoder['anchor'],
		'graph_dynamics': agent.model._graph_dynamics,
		'anchor_decoder': agent.model._anchor_decoder,
		'reward_correction': agent.model._hybrid_reward,
		'q_correction': agent.model._hybrid_q,
		'pi_correction': agent.model._hybrid_pi,
	}
	before = {name: snapshot(module) for name, module in tracked.items()}
	online_before = tensor_state(agent.model._hybrid_q.params)
	detach_before = tensor_state(agent.model._detach_hybrid_q_params)
	target_before = tensor_state(agent.model._target_hybrid_q_params)
	if set(online_before) != set(detach_before) or set(online_before) != set(target_before):
		raise AssertionError('Hybrid Q online/detach/target state keys differ at initialization.')
	for key in online_before:
		if not torch.equal(online_before[key], detach_before[key]):
			raise AssertionError(f'Detach Q correction differs initially at {key}.')
		if not torch.equal(online_before[key], target_before[key]):
			raise AssertionError(f'Target Q correction differs initially at {key}.')
	obs, action, reward, terminated = fake_batch(cfg, agent.device)
	if set(obs.keys()) != {'rgb', 'anchor'} or obs.batch_size != torch.Size([
		cfg.horizon + 1, cfg.batch_size
	]):
		raise AssertionError('Synthetic replay does not match the real TensorDict contract.')
	metrics = agent.update(SyntheticBuffer((obs, action, reward, terminated)))
	for key, value in metrics.items():
		if value.device != agent.device or not torch.isfinite(value).all():
			raise AssertionError(f'Bad metric {key}: device={value.device}, value={value}.')
	for key in ('rgb_consistency_loss', 'graph_consistency_loss'):
		if key not in metrics or not metrics[key] > 0:
			raise AssertionError(f'Missing or zero separated loss: {key}.')

	expected_total = (
		cfg.consistency_coef * metrics['rgb_consistency_loss']
		+ cfg.flat_anchor_graph_consistency_coef * metrics['graph_consistency_loss']
		+ cfg.reward_coef * metrics['reward_loss']
		+ cfg.termination_coef * metrics['termination_loss']
		+ cfg.value_coef * metrics['value_loss']
		+ cfg.flat_anchor_reconstruction_coef * metrics['anchor_reconstruction_loss']
		+ cfg.flat_anchor_prediction_coef * metrics['anchor_prediction_loss']
	)
	torch.testing.assert_close(metrics['total_loss'], expected_total, rtol=1e-5, atol=1e-5)

	changes = {name: max_change(before[name], module) for name, module in tracked.items()}
	for name, change in changes.items():
		if not change > 0.:
			raise AssertionError(f'{name} did not change on the first real update.')
		for parameter_name, parameter in tracked[name].named_parameters():
			if parameter.requires_grad and not torch.isfinite(parameter).all():
				raise AssertionError(f'Non-finite parameter {name}.{parameter_name}.')

	online_after = tensor_state(agent.model._hybrid_q.params)
	target_after = tensor_state(agent.model._target_hybrid_q_params)
	polyak_error = check_polyak(target_before, online_after, target_after, cfg.tau)
	detach_after = tensor_state(agent.model._detach_hybrid_q_params)
	for key in online_after:
		if not torch.equal(online_after[key], detach_after[key]):
			raise AssertionError(f'Detach Q correction is stale at {key}.')

	print('HYBRID_GRAPH_UPDATE_OK', {
		'compile': args.compile,
		'compile_fallback_random': args.compile,
		'joint_latent_dim': cfg.latent_dim,
		'batch_size': cfg.batch_size,
		'observation_keys': tuple(obs.keys()),
		'action_role_mask': action_mask.tolist(),
		'optimizer_tensors': coverage,
		'parameter_max_change': changes,
		'rgb_consistency_loss': metrics['rgb_consistency_loss'].item(),
		'graph_consistency_loss': metrics['graph_consistency_loss'].item(),
		'anchor_reconstruction_loss': metrics['anchor_reconstruction_loss'].item(),
		'anchor_prediction_loss': metrics['anchor_prediction_loss'].item(),
		'target_polyak_max_error': polyak_error,
		'total_loss': metrics['total_loss'].item(),
	})


if __name__ == '__main__':
	main()
