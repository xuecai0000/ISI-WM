"""Real CUDA update/checkpoint contract for ROF-WM V0."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile


os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('LAZY_LEGACY_OP', '0')
os.environ.setdefault('TORCHDYNAMO_INLINE_INBUILT_NN_MODULES', '1')

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from common import MODEL_SIZE, robust_object_field as rof
from common.parser import cfg_to_dataclass
from common.seed import set_seed
from tdmpc2 import TDMPC2


CONFIG_PATH = Path(__file__).with_name('config.yaml')


class SyntheticBuffer:
	def __init__(self, sample):
		self._sample = sample

	def sample(self):
		return (*self._sample, None)


def arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--compile', action='store_true')
	parser.add_argument('--roles', type=int, default=2, choices=(1, 2, 3))
	parser.add_argument('--batch-size', type=int, default=2)
	parser.add_argument('--seed', type=int, default=314159)
	parser.add_argument(
		'--auxiliary-target',
		choices=('full_descriptor', 'geometry_status_full_denominator'),
		default='full_descriptor',
	)
	parser.add_argument('--auxiliary-reconstruction-coef', type=float, default=1.0)
	parser.add_argument('--auxiliary-prediction-coef', type=float, default=1.0)
	return parser.parse_args()


def make_config(args, *, field=True, context_radius=3):
	cfg = OmegaConf.load(CONFIG_PATH)
	cfg.task = 'reacher-easy'
	cfg.obs = 'rgb'
	cfg.model_size = 5
	for key, value in MODEL_SIZE[cfg.model_size].items():
		cfg[key] = value
	cfg.multitask = False
	cfg.tasks = [cfg.task]
	cfg.task_dim = 0
	cfg.action_dim = 2
	cfg.episode_length = 500
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
	cfg.batch_size = args.batch_size
	cfg.horizon = 3
	cfg.compile = args.compile
	cfg.compile_fallback_random = bool(args.compile)
	cfg.flat_anchor = True
	cfg.flat_anchor_mode = 'cutie_object_only'
	cfg.cutie_object_num_roles = args.roles
	cfg.cutie_object_role_names = [f'role_{index}' for index in range(args.roles)]
	cfg.cutie_object_only_latent_dim = args.roles * 5 * 64 if field else args.roles * 64
	cfg.latent_dim = cfg.cutie_object_only_latent_dim
	cfg.obs_shape = (
		rof.observation_shapes(cfg)
		if field else {'object': (args.roles, 1770)}
	)
	cfg.robust_object_field_enabled = field
	cfg.robust_object_field_context_radius = context_radius
	cfg.robust_object_field_random_shift_pad = 3
	cfg.cutie_object_auxiliary_target = args.auxiliary_target
	cfg.robust_object_field_auxiliary_reconstruction_coef = (
		args.auxiliary_reconstruction_coef
	)
	cfg.robust_object_field_auxiliary_prediction_coef = (
		args.auxiliary_prediction_coef
	)
	cfg.enable_wandb = False
	cfg.save_video = False
	cfg.seed = args.seed
	return cfg_to_dataclass(cfg)


def sample(cfg):
	time = cfg.horizon + 1
	batch = cfg.batch_size
	roles = cfg.cutie_object_num_roles
	rgb = torch.randint(
		0, 256, (time, batch, 9, 64, 64), dtype=torch.uint8, device='cuda:0'
	)
	objects = torch.randn(time, batch, roles, 1770, device='cuda:0')
	objects[..., 2 * 590 + 586:2 * 590 + 590] = torch.tensor(
		[0.8, 0.0, 1.0, 0.7], device='cuda:0'
	)
	masks = torch.zeros(
		time, batch, roles, 3, 64, 64, dtype=torch.bool, device='cuda:0'
	)
	for role in range(roles):
		x0, y0 = 4 + 12 * role, 7 + 9 * role
		masks[..., role, :, y0:y0 + 12, x0:x0 + 10] = True
	obs = TensorDict({
		'rgb': rgb,
		'object': objects,
		'object_mask': masks,
		'role_exists': torch.ones(
			time, batch, roles, dtype=torch.float32, device='cuda:0'
		),
	}, batch_size=(time, batch), device='cuda:0')
	action = torch.empty(
		cfg.horizon, batch, cfg.action_dim, device='cuda:0'
	).uniform_(-1.0, 1.0)
	reward = torch.randn(cfg.horizon, batch, 1, device='cuda:0')
	terminated = torch.zeros_like(reward)
	return obs, action, reward, terminated


def trainable_ids(module):
	return {id(value) for value in module.parameters() if value.requires_grad}


def optimizer_ids(optimizer):
	values = [
		id(parameter)
		for group in optimizer.param_groups
		for parameter in group['params']
	]
	if len(values) != len(set(values)):
		raise AssertionError('An optimizer contains duplicate parameters.')
	return set(values)


def snapshot(module):
	return {
		name: value.detach().clone()
		for name, value in module.named_parameters()
		if value.requires_grad
	}


def max_change(before, module):
	after = dict(module.named_parameters())
	return max(float((after[name] - value).abs().max()) for name, value in before.items())


def main():
	args = arguments()
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required for the ROF-WM update contract.')
	set_seed(args.seed)
	cfg = make_config(args)
	agent = TDMPC2(cfg)
	if cfg.latent_dim != rof.latent_dim(cfg):
		raise AssertionError('TDMPC2 did not install the packed ROF latent width.')
	if set(agent.model._encoder) != {'object'}:
		raise AssertionError('ROF-WM constructed an unexpected observation bypass.')
	if any(name.startswith('_encoder.rgb.') for name in agent.model.state_dict()):
		raise AssertionError('ROF-WM state contains a generic RGB encoder.')

	modules = {
		'encoder': agent.model._encoder['object'],
		'dynamics': agent.model._dynamics,
		'decoder': agent.model._object_decoder,
		'reward': agent.model._reward,
		'Qs': agent.model._Qs,
	}
	expected_model = set().union(*(trainable_ids(module) for module in modules.values()))
	actual_model = optimizer_ids(agent.optim)
	if expected_model != actual_model:
		raise AssertionError(
			'ROF optimizer coverage mismatch: '
			f'missing={len(expected_model - actual_model)}, '
			f'extra={len(actual_model - expected_model)}.'
		)
	if optimizer_ids(agent.pi_optim) != trainable_ids(agent.model._pi):
		raise AssertionError('ROF policy optimizer coverage mismatch.')

	before = {name: snapshot(module) for name, module in modules.items()}
	obs, action, reward, terminated = sample(cfg)
	with torch.no_grad():
		latent = agent.model.encode(obs[0], task=None)
		minimum, group_error = rof.simnorm_max_error(latent, cfg.simnorm_dim)
	if minimum < -1e-7 or group_error > 1e-5:
		raise AssertionError(
			f'ROF observation latent left SimNorm: min={minimum}, error={group_error}.'
		)
	metrics = agent.update(SyntheticBuffer((obs, action, reward, terminated)))
	for name in ('consistency_loss', 'object_reconstruction_loss', 'object_prediction_loss'):
		if name not in metrics or not torch.isfinite(metrics[name]).all():
			raise AssertionError(f'Missing or non-finite update metric {name}.')
	changes = {name: max_change(before[name], module) for name, module in modules.items()}
	for name in ('encoder', 'dynamics', 'reward', 'Qs'):
		if changes[name] <= 0.0:
			raise AssertionError(f'ROF update left {name} unchanged: {changes}.')
	auxiliary_active = bool(
		args.auxiliary_reconstruction_coef > 0.0
		or args.auxiliary_prediction_coef > 0.0
	)
	if auxiliary_active != (changes['decoder'] > 0.0):
		raise AssertionError(
			'ROF decoder update did not match the configured auxiliary weights: '
			f'active={auxiliary_active}, changes={changes}.'
		)
	if float(metrics['object_reconstruction_coef']) != float(
		args.auxiliary_reconstruction_coef
	):
		raise AssertionError('ROF reconstruction coefficient metric mismatch.')
	if float(metrics['object_prediction_coef']) != float(
		args.auxiliary_prediction_coef
	):
		raise AssertionError('ROF prediction coefficient metric mismatch.')

	with tempfile.TemporaryDirectory(prefix='rof_wm_v0_update_') as directory:
		checkpoint = Path(directory) / 'agent.pt'
		agent.save(checkpoint)
		payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
		if payload['checkpoint_contract'].get('robust_object_field') != rof.contract(cfg):
			raise AssertionError('ROF checkpoint omitted its exact scientific contract.')
		if payload['checkpoint_contract'].get(
			'robust_object_field_auxiliary'
		) != agent._robust_object_field_auxiliary_contract:
			raise AssertionError('ROF checkpoint omitted its auxiliary ablation contract.')
		reloaded = TDMPC2(make_config(args))
		reloaded.load(checkpoint)
		changed_aux_cfg = make_config(args)
		changed_aux_cfg.robust_object_field_auxiliary_prediction_coef = (
			0.0 if args.auxiliary_prediction_coef != 0.0 else 0.5
		)
		changed_aux = TDMPC2(changed_aux_cfg)
		try:
			changed_aux.load(checkpoint)
		except RuntimeError as exc:
			auxiliary_contract_rejection = str(exc).splitlines()[0]
		else:
			raise AssertionError('Different ROF auxiliary weights loaded silently.')
		changed = TDMPC2(make_config(args, context_radius=4))
		try:
			changed.load(checkpoint)
		except RuntimeError as exc:
			contract_rejection = str(exc).splitlines()[0]
		else:
			raise AssertionError('A different ROF field contract loaded silently.')

	print('ROBUST_OBJECT_FIELD_V0_UPDATE_OK', {
		'compile': bool(args.compile),
		'roles': args.roles,
		'auxiliary_target': args.auxiliary_target,
		'auxiliary_reconstruction_coef': args.auxiliary_reconstruction_coef,
		'auxiliary_prediction_coef': args.auxiliary_prediction_coef,
		'latent_dim': int(cfg.latent_dim),
		'device': torch.cuda.get_device_name(0),
		'simnorm_min': minimum,
		'simnorm_max_group_error': group_error,
		'parameter_max_changes': changes,
		'auxiliary_contract_rejection': auxiliary_contract_rejection,
		'checkpoint_contract_rejection': contract_rejection,
	})


if __name__ == '__main__':
	main()
