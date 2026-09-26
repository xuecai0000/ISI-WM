"""GPU update contract for the native Cutie object-only TDMPC2 path."""

import argparse
import os
import tempfile
from pathlib import Path


os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('LAZY_LEGACY_OP', '0')
os.environ.setdefault('TORCHDYNAMO_INLINE_INBUILT_NN_MODULES', '1')

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tensordict import TensorDict

from common import MODEL_SIZE
from common.parser import cfg_to_dataclass
from common.seed import set_seed
from tdmpc2 import TDMPC2


CONFIG_PATH = Path(__file__).with_name('config.yaml')
OBJECT_SHAPE = (2, 1770)
RGB_SHAPE = (9, 64, 64)
OBJECT_LATENT_DIM = 2 * 64


def parse_args():
	parser = argparse.ArgumentParser(
		description='Run a real eager or compiled GPU update for cutie_object_only.',
	)
	parser.add_argument('--compile', action='store_true')
	parser.add_argument('--batch-size', type=int, default=2)
	parser.add_argument('--seed', type=int, default=271828)
	parser.add_argument(
		'--auxiliary-target',
		choices=('full_descriptor', 'geometry_status_full_denominator'),
		default='full_descriptor',
	)
	return parser.parse_args()


def make_config(args, *, mode='cutie_object_only', compile_enabled=None):
	cfg = OmegaConf.load(CONFIG_PATH)
	cfg.task = 'reacher-visual-small'
	cfg.obs = 'rgb'
	cfg.model_size = 5
	for key, value in MODEL_SIZE[cfg.model_size].items():
		cfg[key] = value
	cfg.multitask = False
	cfg.tasks = [cfg.task]
	cfg.task_dim = 0
	cfg.obs_shape = (
		{'object': OBJECT_SHAPE}
		if mode == 'cutie_object_only'
		else {'rgb': RGB_SHAPE, 'object': OBJECT_SHAPE}
	)
	cfg.action_dim = 2
	cfg.episode_length = 500
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
	cfg.batch_size = args.batch_size
	cfg.horizon = 3
	cfg.compile = args.compile if compile_enabled is None else compile_enabled
	cfg.compile_fallback_random = bool(cfg.compile)
	cfg.flat_anchor = True
	cfg.flat_anchor_mode = mode
	cfg.cutie_object_auxiliary_target = args.auxiliary_target
	cfg.enable_wandb = False
	cfg.save_video = False
	cfg.seed = args.seed
	return cfg_to_dataclass(cfg)


def fail(message):
	raise AssertionError(message)


def exact(left, right, label):
	if type(left) is not type(right):
		fail(f'{label}: type mismatch {type(left).__name__} != {type(right).__name__}')
	if torch.is_tensor(left):
		if not torch.equal(left, right):
			fail(f'{label}: tensor values differ')
	elif left != right:
		fail(f'{label}: values differ ({left!r} != {right!r})')


def exact_state_dict(left, right, label):
	if set(left) != set(right):
		missing = sorted(set(left) - set(right))
		extra = sorted(set(right) - set(left))
		fail(f'{label}: state keys differ; missing={missing}, extra={extra}')
	for name in left:
		exact(left[name], right[name], f'{label}.{name}')


def module_parameter_ids(module):
	return {id(parameter) for parameter in module.parameters() if parameter.requires_grad}


def optimizer_parameter_ids(optimizer, label):
	ids = []
	for group in optimizer.param_groups:
		ids.extend(id(parameter) for parameter in group['params'])
	if len(ids) != len(set(ids)):
		fail(f'{label}: duplicate parameter appears in more than one optimizer group')
	return set(ids)


def parameter_snapshot(module):
	return {
		name: parameter.detach().clone()
		for name, parameter in module.named_parameters()
		if parameter.requires_grad
	}


def tensor_state_snapshot(module):
	return {
		name: value.detach().clone()
		for name, value in module.state_dict().items()
		if torch.is_tensor(value) and value.is_floating_point()
	}


def maximum_parameter_change(before, module, label):
	after = dict(module.named_parameters())
	if not before:
		fail(f'{label}: module has no trainable parameters')
	if not set(before).issubset(after):
		fail(f'{label}: parameters disappeared during update')
	return max(
		(after[name].detach() - value).abs().max().item()
		for name, value in before.items()
	)


def maximum_state_change(before, module, label):
	after = module.state_dict()
	if not before:
		fail(f'{label}: state contains no tensors')
	if not set(before).issubset(after):
		fail(f'{label}: state keys disappeared during update')
	return max(
		(after[name].detach() - value).abs().max().item()
		for name, value in before.items()
	)


def check_polyak(before, online_after, target_after, tau):
	if set(before) != set(online_after) or set(before) != set(target_after):
		fail('official online/target Q state keys do not match')
	max_error = 0.0
	for name in before:
		expected = torch.lerp(before[name], online_after[name], tau)
		error = float((target_after[name] - expected).abs().max())
		max_error = max(max_error, error)
		torch.testing.assert_close(target_after[name], expected, rtol=0.0, atol=1e-7)
	return max_error


def assert_module_finite(module, label):
	for name, parameter in module.named_parameters():
		if not torch.isfinite(parameter).all():
			fail(f'{label}.{name}: non-finite parameter after update')


class SyntheticBuffer:
	def __init__(self, sample):
		self._sample = sample

	def sample(self):
		return (*self._sample, None)


def make_sample(cfg, device):
	time = cfg.horizon + 1
	objects = torch.randn(
		time,
		cfg.batch_size,
		*OBJECT_SHAPE,
		device=device,
	)
	objects[..., 2 * 590 + 588] = 1.0
	obs = TensorDict(
		{'object': objects},
		batch_size=(time, cfg.batch_size),
		device=device,
	)
	action = torch.empty(
		cfg.horizon,
		cfg.batch_size,
		cfg.action_dim,
		device=device,
	).uniform_(-1.0, 1.0)
	reward = torch.randn(cfg.horizon, cfg.batch_size, 1, device=device)
	terminated = torch.zeros_like(reward)
	return obs, action, reward, terminated


def assert_native_object_only(agent, cfg, preconstruction_latent):
	if preconstruction_latent != MODEL_SIZE[5]['latent_dim']:
		fail(f'preconstruction latent_dim must be the model-size value, got {preconstruction_latent}')
	if cfg.latent_dim != OBJECT_LATENT_DIM:
		fail(f'object-only latent_dim must be {OBJECT_LATENT_DIM}, got {cfg.latent_dim}')
	if not agent._cutie_object_only or not agent._cutie_object_mode or agent._hybrid_graph:
		fail('TDMPC2 object-only mode flags are inconsistent')
	if set(cfg.obs_shape) != {'object'} or tuple(cfg.obs_shape['object']) != OBJECT_SHAPE:
		fail(f'object-only observation schema leaked another field: {cfg.obs_shape}')
	if set(agent.model._encoder.keys()) != {'object'}:
		fail(f'object-only encoder keys must be exactly object: {list(agent.model._encoder.keys())}')
	if hasattr(agent.model, '_graph_dynamics'):
		fail('object-only WorldModel must not construct graph dynamics')
	for forbidden in (
		'_hybrid_fusion', '_hybrid_dynamics', '_hybrid_reward', '_hybrid_pi',
		'_hybrid_q', '_hybrid_termination',
	):
		if hasattr(agent.model, forbidden):
			fail(f'object-only WorldModel unexpectedly contains {forbidden}')
	for name in agent.model.state_dict():
		if name.startswith('_encoder.rgb.'):
			fail(f'RGB encoder parameter leaked into object-only state: {name}')


def assert_optimizer_contract(agent, cfg):
	modules = {
		'object_encoder': agent.model._encoder['object'],
		'object_dynamics': agent.model._dynamics,
		'object_decoder': agent.model._object_decoder,
		'reward': agent.model._reward,
		'Qs': agent.model._Qs,
	}
	expected_model_ids = set()
	for module in modules.values():
		expected_model_ids |= module_parameter_ids(module)
	model_ids = optimizer_parameter_ids(agent.optim, 'model optimizer')
	pi_ids = optimizer_parameter_ids(agent.pi_optim, 'policy optimizer')
	expected_pi_ids = module_parameter_ids(agent.model._pi)
	if model_ids != expected_model_ids:
		fail(
			'model optimizer coverage mismatch: '
			f'missing={len(expected_model_ids - model_ids)}, '
			f'extra={len(model_ids - expected_model_ids)}'
		)
	if pi_ids != expected_pi_ids:
		fail(
			'policy optimizer coverage mismatch: '
			f'missing={len(expected_pi_ids - pi_ids)}, '
			f'extra={len(pi_ids - expected_pi_ids)}'
		)
	if model_ids & pi_ids:
		fail('model and policy optimizers must be disjoint')
	target_ids = module_parameter_ids(agent.model._target_Qs_params)
	if target_ids & (model_ids | pi_ids):
		fail('target Q parameters must not belong to either optimizer')
	encoder_ids = module_parameter_ids(agent.model._encoder)
	encoder_groups = [
		group for group in agent.optim.param_groups
		if {id(parameter) for parameter in group['params']} == encoder_ids
	]
	if len(encoder_groups) != 1:
		fail('object encoder must occupy exactly one dedicated optimizer group')
	expected_encoder_lr = cfg.lr * cfg.enc_lr_scale
	if encoder_groups[0]['lr'] != expected_encoder_lr:
		fail(
			f'object encoder LR mismatch: {encoder_groups[0]["lr"]} '
			f'!= {expected_encoder_lr}'
		)
	return modules, model_ids, pi_ids, target_ids


def assert_auxiliary_target_isolation(agent, args, device):
	"""Prove target selection changes only auxiliary gradients, not the model."""
	opposite_target = (
		'geometry_status_full_denominator'
		if args.auxiliary_target == 'full_descriptor'
		else 'full_descriptor'
	)
	opposite_cfg = make_config(args, compile_enabled=False)
	opposite_cfg.cutie_object_auxiliary_target = opposite_target
	# Rewind to the exact seed used immediately before constructing ``agent``.
	# The helper/target branch consumes no RNG, so the complete states must match.
	set_seed(args.seed)
	opposite = TDMPC2(opposite_cfg)
	exact_state_dict(
		agent.model.state_dict(),
		opposite.model.state_dict(),
		'cross-target pre-update model',
	)
	agent_params = sum(parameter.numel() for parameter in agent.model.parameters())
	opposite_params = sum(parameter.numel() for parameter in opposite.model.parameters())
	if agent_params != opposite_params:
		fail('auxiliary target changed the model parameter count')
	agent_decoder_params = sum(
		parameter.numel() for parameter in agent.model._object_decoder.parameters()
	)
	opposite_decoder_params = sum(
		parameter.numel() for parameter in opposite.model._object_decoder.parameters()
	)
	if agent_decoder_params != opposite_decoder_params:
		fail('auxiliary target changed the decoder parameter count')

	full_model, geometry_model = (
		(agent.model, opposite.model)
		if args.auxiliary_target == 'full_descriptor'
		else (opposite.model, agent.model)
	)
	target = torch.randn(args.batch_size, *OBJECT_SHAPE, device=device)
	target[..., 2 * 590 + 588] = 1.0
	prediction_value = torch.randn_like(target)
	full_prediction = prediction_value.clone().requires_grad_(True)
	geometry_prediction = prediction_value.clone().requires_grad_(True)
	full_loss = full_model.object_loss(full_prediction, target)
	geometry_loss = geometry_model.object_loss(geometry_prediction, target)
	full_grad, = torch.autograd.grad(full_loss, full_prediction)
	geometry_grad, = torch.autograd.grad(geometry_loss, geometry_prediction)
	full_grad_frames = full_grad.reshape(*target.shape[:-1], 3, 590)
	geometry_grad_frames = geometry_grad.reshape(*target.shape[:-1], 3, 590)
	query_gradient_nonzero = int(
		torch.count_nonzero(geometry_grad_frames[..., :512]).item()
	)
	if query_gradient_nonzero:
		fail('geometry_status auxiliary target leaked a query gradient')
	if not torch.equal(
		geometry_grad_frames[..., 512:], full_grad_frames[..., 512:]
	):
		fail('geometry/status per-element gradient changed from full_descriptor')

	cfg_beta = float(agent.cfg.flat_anchor_loss_beta)
	manual_element_loss = F.smooth_l1_loss(
		prediction_value,
		target,
		reduction='none',
		beta=cfg_beta,
	).reshape(*target.shape[:-1], 3, 590)
	manual_mask = manual_element_loss.new_zeros(590)
	manual_mask[512:] = 1.
	valid = target[..., 2 * 590 + 588].clamp(0., 1.)
	floor = float(agent.cfg.flat_anchor_loss_weight_floor)
	role_weight = floor + (1. - floor) * valid
	manual_role_loss = (
		manual_element_loss * manual_mask
	).reshape(*target.shape[:-1], -1).mean(dim=-1)
	manual_loss = (
		manual_role_loss * role_weight
	).sum() / role_weight.sum().clamp_min(1.)
	if not torch.equal(geometry_loss, manual_loss):
		fail(
			'geometry_status auxiliary loss is not the elementwise-masked '
			'full-1770 mean'
		)

	query_changed = target.clone().reshape(*target.shape[:-1], 3, 590)
	query_changed[..., :512].add_(0.5)
	query_changed = query_changed.reshape_as(target)
	encoded = geometry_model.encode({'object': target}, task=None)
	changed_encoded = geometry_model.encode({'object': query_changed}, task=None)
	encoder_query_delta = float((encoded - changed_encoded).abs().max())
	if encoder_query_delta <= 0.0:
		fail('auxiliary query masking leaked into the Full-Cutie encoder input')
	decoded = geometry_model.decode_object(encoded)
	if tuple(decoded.shape) != tuple(target.shape):
		fail(f'auxiliary target changed decoder output shape to {tuple(decoded.shape)}')

	diagnostics = {
		'pre_update_model_state_exact': True,
		'model_parameter_count': agent_params,
		'decoder_parameter_count': agent_decoder_params,
		'decoder_output_shape': tuple(decoded.shape),
		'query_gradient_nonzero': query_gradient_nonzero,
		'geometry_gradient_max_abs_difference': float((
			geometry_grad_frames[..., 512:] - full_grad_frames[..., 512:]
		).abs().max()),
		'masked_loss_full_denominator_exact': True,
		'loss_denominator_values_per_role': 1770,
		'encoder_query_delta': encoder_query_delta,
		'smooth_l1_beta': cfg_beta,
	}
	del opposite
	torch.cuda.empty_cache()
	return diagnostics


def assert_checkpoint_contract(agent, args, checkpoint_path):
	agent.save(checkpoint_path)
	payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
	expected_auxiliary = agent.model._cutie_object_auxiliary_contract
	if payload.get('checkpoint_contract', {}).get(
		'cutie_object_auxiliary'
	) != expected_auxiliary:
		fail('saved checkpoint omitted or changed its auxiliary-target contract')
	reload_cfg = make_config(args, compile_enabled=False)
	reloaded = TDMPC2(reload_cfg)
	reloaded.load(checkpoint_path)
	exact_state_dict(
		agent.model.state_dict(),
		reloaded.model.state_dict(),
		'strict same-mode reload',
	)
	opposite_target = (
		'geometry_status_full_denominator'
		if args.auxiliary_target == 'full_descriptor'
		else 'full_descriptor'
	)
	opposite_cfg = make_config(args, compile_enabled=False)
	opposite_cfg.cutie_object_auxiliary_target = opposite_target
	opposite = TDMPC2(opposite_cfg)
	try:
		opposite.load(checkpoint_path)
	except RuntimeError as error:
		cross_target_error = str(error).splitlines()[0]
	else:
		fail('checkpoint load accepted a different auxiliary target')
	hybrid_cfg = make_config(args, mode='cutie_hybrid', compile_enabled=False)
	hybrid = TDMPC2(hybrid_cfg)
	try:
		reloaded.load({'model': hybrid.model.state_dict()})
	except (RuntimeError, ValueError) as error:
		object_receiver_error = str(error).splitlines()[0]
	else:
		fail('cutie_object_only unexpectedly accepted a cutie_hybrid checkpoint')
	try:
		hybrid.load(checkpoint_path)
	except (RuntimeError, ValueError) as error:
		hybrid_receiver_error = str(error).splitlines()[0]
	else:
		fail('cutie_hybrid unexpectedly accepted a cutie_object_only checkpoint')
	del reloaded
	del opposite
	del hybrid
	torch.cuda.empty_cache()
	return {
		'object_receiver': object_receiver_error,
		'hybrid_receiver': hybrid_receiver_error,
		'cross_target': cross_target_error,
	}


def main():
	args = parse_args()
	if args.batch_size < 1:
		fail('--batch-size must be positive')
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required for the cutie_object_only update contract')
	set_seed(args.seed)
	device = torch.device('cuda:0')
	cfg = make_config(args)
	preconstruction_latent = cfg.latent_dim
	agent = TDMPC2(cfg)
	assert_native_object_only(agent, cfg, preconstruction_latent)
	modules, model_ids, pi_ids, target_ids = assert_optimizer_contract(agent, cfg)
	auxiliary_isolation = assert_auxiliary_target_isolation(agent, args, device)

	obs, action, reward, terminated = make_sample(cfg, device)
	if set(obs.keys()) != {'object'}:
		fail(f'synthetic replay observation must contain only object, got {list(obs.keys())}')
	before = {name: parameter_snapshot(module) for name, module in modules.items()}
	before['pi'] = parameter_snapshot(agent.model._pi)
	online_before = tensor_state_snapshot(agent.model._Qs.params)
	detach_before = tensor_state_snapshot(agent.model._detach_Qs_params)
	target_before = tensor_state_snapshot(agent.model._target_Qs_params)
	exact_state_dict(online_before, detach_before, 'initial online/detach Q')
	exact_state_dict(online_before, target_before, 'initial online/target Q')

	metrics = agent.update(SyntheticBuffer((obs, action, reward, terminated)))
	for name, value in metrics.items():
		if torch.is_tensor(value):
			if value.device != device:
				fail(f'metric {name} is on {value.device}, expected {device}')
			if not torch.isfinite(value).all():
				fail(f'metric {name} is non-finite: {value}')
	for required in ('object_reconstruction_loss', 'object_prediction_loss'):
		if required not in metrics:
			fail(f'update metrics are missing {required}')

	changes = {}
	for name, module in {**modules, 'pi': agent.model._pi}.items():
		assert_module_finite(module, name)
		changes[name] = maximum_parameter_change(before[name], module, name)
		if changes[name] <= 0.0:
			fail(f'{name}: real TDMPC2 update did not change any parameter')
	target_change = maximum_state_change(
		target_before,
		agent.model._target_Qs_params,
		'target_Qs',
	)
	if target_change <= 0.0:
		fail('target Q parameters did not move after the soft update')
	assert_module_finite(agent.model._target_Qs_params, 'target_Qs')
	online_after = tensor_state_snapshot(agent.model._Qs.params)
	detach_after = tensor_state_snapshot(agent.model._detach_Qs_params)
	target_after = tensor_state_snapshot(agent.model._target_Qs_params)
	exact_state_dict(online_after, detach_after, 'updated online/detach Q')
	polyak_error = check_polyak(target_before, online_after, target_after, cfg.tau)

	with tempfile.TemporaryDirectory(prefix='cutie_object_only_update_') as directory:
		checkpoint_path = Path(directory) / 'agent.pt'
		cross_mode_errors = assert_checkpoint_contract(agent, args, checkpoint_path)

	print(
		'CUTIE_OBJECT_ONLY_UPDATE_OK',
		{
			'compile': args.compile,
			'device': torch.cuda.get_device_name(device),
			'latent_dim': cfg.latent_dim,
			'auxiliary_target': args.auxiliary_target,
			'auxiliary_contract': agent.model._cutie_object_auxiliary_contract,
			'auxiliary_isolation': auxiliary_isolation,
			'observation_keys': list(obs.keys()),
			'model_optimizer_parameters': len(model_ids),
			'policy_optimizer_parameters': len(pi_ids),
			'target_q_parameters': len(target_ids),
			'parameter_max_changes': changes,
			'target_q_max_change': target_change,
			'target_q_polyak_max_error': polyak_error,
			'checkpoint_strict_reload': True,
			'cross_mode_rejected': True,
			'cross_mode_errors': cross_mode_errors,
		},
	)


if __name__ == '__main__':
	main()
