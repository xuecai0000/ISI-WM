"""Real CUDA update contract for the privileged Acrobot pose ObjectOnly path."""

import argparse
from copy import deepcopy
import os
from pathlib import Path
import tempfile


os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('LAZY_LEGACY_OP', '0')
os.environ.setdefault('TORCHDYNAMO_INLINE_INBUILT_NN_MODULES', '1')

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tensordict import TensorDict

from common import MODEL_SIZE
from common import gt_articulated_pose as pose_contract
from common.parser import cfg_to_dataclass
from common.seed import set_seed
from tdmpc2 import TDMPC2


CONFIG_PATH = Path(__file__).with_name('config.yaml')
OBJECT_SHAPE = (pose_contract.NUM_ROLES, pose_contract.INPUT_DIM)
OBJECT_LATENT_DIM = pose_contract.NUM_ROLES * pose_contract.ROLE_DIM


def parse_args():
	parser = argparse.ArgumentParser(
		description='Run an eager or compiled CUDA update for gt_articulated_pose.',
	)
	parser.add_argument('--compile', action='store_true')
	parser.add_argument('--batch-size', type=int, default=2)
	parser.add_argument('--seed', type=int, default=271828)
	return parser.parse_args()


def fail(message):
	raise AssertionError(message)


def make_config(args, *, pose=True, compile_enabled=None):
	cfg = OmegaConf.load(CONFIG_PATH)
	cfg.task = pose_contract.TASK if pose else 'reacher-visual-small'
	cfg.obs = 'state' if pose else 'rgb'
	cfg.model_size = 5
	for key, value in MODEL_SIZE[cfg.model_size].items():
		cfg[key] = value
	cfg.multitask = False
	cfg.tasks = [cfg.task]
	cfg.task_dim = 0
	cfg.action_dim = 1
	cfg.episode_length = 500
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
	cfg.batch_size = args.batch_size
	cfg.horizon = 3
	cfg.compile = args.compile if compile_enabled is None else compile_enabled
	cfg.compile_fallback_random = bool(cfg.compile)
	cfg.flat_anchor = True
	cfg.flat_anchor_mode = 'cutie_object_only'
	cfg.cutie_object_belief_enabled = False
	cfg.cutie_object_belief_use_for_control = False
	cfg.cutie_object_last_valid_memory = False
	cfg.cutie_object_policy_burst_plan = None
	cfg.enable_wandb = False
	cfg.save_video = False
	cfg.seed = args.seed
	if pose:
		cfg.obs_shape = {'object': OBJECT_SHAPE}
		cfg.cutie_object_observation_variant = pose_contract.VARIANT
		cfg.cutie_object_frame_schema = pose_contract.FRAME_SCHEMA
		cfg.cutie_object_role_names = list(pose_contract.ROLE_NAMES)
		cfg.cutie_object_num_roles = pose_contract.NUM_ROLES
		cfg.cutie_object_frame_dim = pose_contract.FRAME_DIM
		cfg.cutie_object_stack_frames = pose_contract.STACK_FRAMES
		cfg.cutie_object_input_dim = pose_contract.INPUT_DIM
		cfg.cutie_object_role_dim = pose_contract.ROLE_DIM
		cfg.cutie_object_only_latent_dim = pose_contract.LATENT_DIM
		cfg.cutie_object_allow_simulator_kinematics_runtime = True
		cfg.cutie_object_allow_simulator_runtime = False
		cfg.cutie_object_allow_simulator_support = False
		cfg.cutie_object_native_highres_enabled = False
		cfg.cutie_object_auxiliary_target = 'full_descriptor'
		cfg.visual_foreground_erosion_pixels = 0
		cfg.video_background_enabled = False
		for key in (
			'cutie_object_repo', 'cutie_object_checkpoint',
			'cutie_object_support_path', 'cutie_object_config_dir',
			'video_background_root', 'video_background_manifest_dir',
		):
			cfg[key] = None
	else:
		cfg.obs_shape = {'object': (2, 1770)}
		cfg.cutie_object_observation_variant = 'full'
		cfg.cutie_object_frame_schema = 'cutie_query_mask_status_v1'
		cfg.cutie_object_role_names = ['whole_arm', 'goal']
		cfg.cutie_object_num_roles = 2
		cfg.cutie_object_frame_dim = 590
		cfg.cutie_object_stack_frames = 3
		cfg.cutie_object_input_dim = 1770
		cfg.cutie_object_allow_simulator_kinematics_runtime = False
	return cfg_to_dataclass(cfg)


def exact_state_dict(left, right, label):
	if set(left) != set(right):
		fail(
			f'{label}: state keys differ; '
			f'missing={sorted(set(left) - set(right))[:10]}, '
			f'extra={sorted(set(right) - set(left))[:10]}'
		)
	for name in left:
		left_value, right_value = left[name], right[name]
		if torch.is_tensor(left_value) or torch.is_tensor(right_value):
			if (
				not torch.is_tensor(left_value)
				or not torch.is_tensor(right_value)
				or not torch.equal(left_value, right_value)
			):
				fail(f'{label}.{name}: tensor values differ')
		elif type(left_value) is not type(right_value) or left_value != right_value:
			# TensorDict serializes exact batch-size/device bookkeeping beside
			# tensors. It is not a tensor, but it still must round-trip exactly.
			fail(f'{label}.{name}: non-tensor state metadata differs')


def module_parameter_ids(module):
	return {id(parameter) for parameter in module.parameters() if parameter.requires_grad}


def optimizer_parameter_ids(optimizer, label):
	ids = [
		id(parameter)
		for group in optimizer.param_groups
		for parameter in group['params']
	]
	if len(ids) != len(set(ids)):
		fail(f'{label}: duplicate parameter appears in optimizer groups')
	return set(ids)


def parameter_snapshot(module):
	return {
		name: parameter.detach().clone()
		for name, parameter in module.named_parameters()
		if parameter.requires_grad
	}


def maximum_parameter_change(before, module, label):
	after = dict(module.named_parameters())
	if not before or not set(before).issubset(after):
		fail(f'{label}: invalid parameter snapshot')
	return max(
		float((after[name].detach() - value).abs().max())
		for name, value in before.items()
	)


def assert_module_finite(module, label):
	for name, parameter in module.named_parameters():
		if not torch.isfinite(parameter).all():
			fail(f'{label}.{name}: non-finite parameter')


class SyntheticBuffer:
	def __init__(self, sample):
		self._sample = sample

	def sample(self):
		return (*self._sample, None)


def make_sample(cfg, device):
	time = cfg.horizon + 1
	objects = torch.randn(
		time, cfg.batch_size, *OBJECT_SHAPE, device=device
	)
	if not torch.isfinite(objects).all():
		fail('synthetic pose replay is non-finite')
	obs = TensorDict(
		{'object': objects},
		batch_size=(time, cfg.batch_size),
		device=device,
	)
	action = torch.empty(
		cfg.horizon, cfg.batch_size, cfg.action_dim, device=device
	).uniform_(-1., 1.)
	reward = torch.randn(cfg.horizon, cfg.batch_size, 1, device=device)
	terminated = torch.zeros_like(reward)
	return obs, action, reward, terminated


def assert_pose_only(agent, cfg, preconstruction_latent):
	if preconstruction_latent != MODEL_SIZE[5]['latent_dim']:
		fail('model-size latent width changed before ObjectOnly construction')
	if cfg.latent_dim != OBJECT_LATENT_DIM or cfg.latent_dim != 128:
		fail(f'pose ObjectOnly latent must be 128-D, got {cfg.latent_dim}')
	if not agent._cutie_object_only or not agent._cutie_object_mode:
		fail('pose ObjectOnly mode flags are inconsistent')
	if agent._cutie_hybrid or agent._hybrid_graph or agent._learned_object_belief:
		fail('pose oracle enabled a hybrid/Cutie-belief controller path')
	if agent.belief_optim is not None or hasattr(agent.model, '_belief_dynamics'):
		fail('pose oracle constructed learned-belief state')
	if set(cfg.obs_shape) != {'object'} or tuple(cfg.obs_shape['object']) != (2, 21):
		fail(f'pose observation schema is not object-only [2,21]: {cfg.obs_shape}')
	if set(agent.model._encoder) != {'object'}:
		fail(f'pose encoder keys are not object-only: {list(agent.model._encoder)}')
	if hasattr(agent.model, '_graph_dynamics'):
		fail('pose ObjectOnly unexpectedly constructed graph dynamics')
	for name in agent.model.state_dict():
		if name.startswith('_encoder.rgb.') or name.startswith('_hybrid_'):
			fail(f'RGB/hybrid state leaked into pose model: {name}')
	for key in (
		'cutie_object_repo', 'cutie_object_checkpoint',
		'cutie_object_support_path', 'cutie_object_config_dir',
	):
		if cfg.get(key) is not None:
			fail(f'pose config unexpectedly uses Cutie asset {key}')
	if agent.model._cutie_object_auxiliary_contract != (
		pose_contract.auxiliary_contract(beta=cfg.flat_anchor_loss_beta)
	):
		fail('pose auxiliary contract is not the frozen SmoothL1 contract')


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
	if pi_ids != expected_pi_ids or model_ids & pi_ids:
		fail('policy optimizer coverage/partition mismatch')
	if agent.model._termination is not None:
		fail('Acrobot pose contract unexpectedly enabled termination head')
	encoder_ids = module_parameter_ids(agent.model._encoder)
	encoder_groups = [
		group for group in agent.optim.param_groups
		if {id(parameter) for parameter in group['params']} == encoder_ids
	]
	if len(encoder_groups) != 1:
		fail('pose encoder does not have one dedicated optimizer group')
	if encoder_groups[0]['lr'] != cfg.lr * cfg.enc_lr_scale:
		fail('pose encoder optimizer LR changed')
	return modules, len(model_ids), len(pi_ids)


def assert_plain_smooth_l1(agent, cfg, device):
	target = torch.randn(cfg.batch_size, *OBJECT_SHAPE, device=device)
	prediction_value = torch.randn_like(target)
	prediction = prediction_value.clone().requires_grad_(True)
	reference_prediction = prediction_value.clone().requires_grad_(True)
	loss = agent.model.object_loss(prediction, target)
	reference = F.smooth_l1_loss(
		reference_prediction,
		target,
		beta=cfg.flat_anchor_loss_beta,
	)
	gradient, = torch.autograd.grad(loss, prediction)
	reference_gradient, = torch.autograd.grad(reference, reference_prediction)
	if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
		fail('pose SmoothL1 loss/gradient is non-finite')
	if not torch.equal(loss, reference) or not torch.equal(
		gradient, reference_gradient
	):
		fail('pose auxiliary is not ordinary unweighted mean SmoothL1')
	encoded = agent.model.encode({'object': target}, task=None)
	decoded = agent.model.decode_object(encoded)
	if tuple(encoded.shape) != (cfg.batch_size, 128):
		fail(f'pose encoder emitted {tuple(encoded.shape)}, expected [B,128]')
	if tuple(decoded.shape) != tuple(target.shape):
		fail(f'pose decoder emitted {tuple(decoded.shape)}, expected {tuple(target.shape)}')
	return {
		'loss': float(loss.detach()),
		'beta': float(cfg.flat_anchor_loss_beta),
		'gradient_max': float(gradient.abs().max()),
		'encoded_shape': tuple(encoded.shape),
		'decoded_shape': tuple(decoded.shape),
		'exact_plain_smooth_l1': True,
	}


def assert_checkpoint_contract(agent, args, checkpoint_path):
	agent.save(checkpoint_path)
	payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
	checkpoint = payload.get('checkpoint_contract')
	if not isinstance(checkpoint, dict):
		fail('pose checkpoint omitted the checkpoint contract')
	expected_pose = pose_contract.observation_contract(agent.cfg)
	if checkpoint.get('gt_articulated_pose_observation') != expected_pose:
		fail('pose checkpoint omitted or changed privileged observation metadata')
	if checkpoint.get('cutie_object_auxiliary') != (
		agent.model._cutie_object_auxiliary_contract
	):
		fail('pose checkpoint omitted or changed auxiliary metadata')

	reloaded = TDMPC2(make_config(args, compile_enabled=False))
	reloaded.load(checkpoint_path)
	exact_state_dict(
		agent.model.state_dict(), reloaded.model.state_dict(), 'strict pose reload'
	)

	mutations = {
		'format': 'wrong_format',
		'privileged_simulator_kinematics': False,
		'privileged_simulator_kinematics_runtime': False,
		'task': 'wrong-task',
		'role_names': list(reversed(pose_contract.ROLE_NAMES)),
		'frame_schema': 'wrong_schema',
		'source': 'wrong_source',
		'diagnostic_only': False,
	}
	contract_errors = {}
	for key, value in mutations.items():
		foreign = {
			'model': payload['model'],
			'checkpoint_contract': deepcopy(checkpoint),
		}
		foreign['checkpoint_contract'][
			'gt_articulated_pose_observation'
		][key] = value
		try:
			reloaded.load(foreign)
		except RuntimeError as error:
			contract_errors[key] = str(error).splitlines()[0]
		else:
			fail(f'pose checkpoint accepted changed {key!r} metadata')
	missing = {
		'model': payload['model'],
		'checkpoint_contract': deepcopy(checkpoint),
	}
	missing['checkpoint_contract'].pop('gt_articulated_pose_observation')
	try:
		reloaded.load(missing)
	except RuntimeError as error:
		contract_errors['missing'] = str(error).splitlines()[0]
	else:
		fail('pose checkpoint accepted missing privileged metadata')
	legacy_source = {
		'model': payload['model'],
		'checkpoint_contract': deepcopy(checkpoint),
	}
	legacy_source['checkpoint_contract']['cutie_object_observation'] = {
		'format': 'cutie_object_observation_contract_v1',
		'variant': 'full',
		'frame_schema': 'cutie_query_mask_status_v1',
		'privileged_runtime_segmentation': False,
		'num_roles': 2,
		'frame_dim': 590,
		'stack_frames': 3,
		'input_dim': 1770,
	}
	legacy_source['checkpoint_contract'].pop(
		'gt_articulated_pose_observation', None
	)
	try:
		reloaded.load(legacy_source)
	except RuntimeError as error:
		pose_receiver_error = str(error).splitlines()[0]
	else:
		fail('pose receiver accepted a legacy Cutie ObjectOnly checkpoint contract')

	legacy = TDMPC2(make_config(args, pose=False, compile_enabled=False))
	try:
		legacy.load(checkpoint_path)
	except RuntimeError as error:
		legacy_error = str(error).splitlines()[0]
	else:
		fail('legacy Cutie ObjectOnly accepted a pose-oracle checkpoint')
	del reloaded
	del legacy
	torch.cuda.empty_cache()
	return {
		'same_mode_strict_reload': True,
		'pose_contract_mutations_rejected': contract_errors,
		'pose_receiver_cross_load_rejected': pose_receiver_error,
		'legacy_cross_load_rejected': legacy_error,
	}


def main():
	args = parse_args()
	if args.batch_size < 1:
		fail('--batch-size must be positive')
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required for gt_articulated_pose update contract')
	set_seed(args.seed)
	device = torch.device('cuda:0')
	cfg = make_config(args)
	preconstruction_latent = cfg.latent_dim
	agent = TDMPC2(cfg)
	assert_pose_only(agent, cfg, preconstruction_latent)
	modules, model_parameter_count, policy_parameter_count = (
		assert_optimizer_contract(agent, cfg)
	)
	auxiliary = assert_plain_smooth_l1(agent, cfg, device)

	obs, action, reward, terminated = make_sample(cfg, device)
	before = {name: parameter_snapshot(module) for name, module in modules.items()}
	before['pi'] = parameter_snapshot(agent.model._pi)
	metrics = agent.update(SyntheticBuffer((obs, action, reward, terminated)))
	for name, value in metrics.items():
		if torch.is_tensor(value):
			if value.device != device or not torch.isfinite(value).all():
				fail(f'update metric {name!r} is invalid: {value}')
	for required in ('object_reconstruction_loss', 'object_prediction_loss'):
		if required not in metrics or not torch.isfinite(metrics[required]):
			fail(f'pose update metric {required!r} is missing/non-finite')

	changes = {}
	for name, module in {**modules, 'pi': agent.model._pi}.items():
		assert_module_finite(module, name)
		changes[name] = maximum_parameter_change(before[name], module, name)
		if changes[name] <= 0.:
			fail(f'{name}: real pose update changed no parameter')

	with tempfile.TemporaryDirectory(prefix='gt_articulated_pose_update_') as directory:
		checkpoint = assert_checkpoint_contract(
			agent, args, Path(directory) / 'agent.pt'
		)

	print('GT_ARTICULATED_POSE_UPDATE_OK', {
		'compile': bool(args.compile),
		'device': torch.cuda.get_device_name(device),
		'observation_shape': OBJECT_SHAPE,
		'latent_dim': int(cfg.latent_dim),
		'object_only': True,
		'cutie_assets': None,
		'belief_enabled': False,
		'auxiliary': auxiliary,
		'model_optimizer_parameters': model_parameter_count,
		'policy_optimizer_parameters': policy_parameter_count,
		'parameter_max_changes': changes,
		'checkpoint': checkpoint,
	})


if __name__ == '__main__':
	main()
