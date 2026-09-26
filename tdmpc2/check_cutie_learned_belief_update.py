"""Real CUDA contract for burst-trained Cutie object belief integration."""

import argparse
import copy
import os
import tempfile
from pathlib import Path


os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('LAZY_LEGACY_OP', '0')
os.environ.setdefault('TORCHDYNAMO_INLINE_INBUILT_NN_MODULES', '1')

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from common import MODEL_SIZE
from common.buffer import Buffer
from common.parser import cfg_to_dataclass
from common.seed import set_seed
from tdmpc2 import TDMPC2


CONFIG_PATH = Path(__file__).with_name('config.yaml')
OBJECT_SHAPE = (2, 1770)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--compile', action='store_true')
	parser.add_argument(
		'--task', choices=('reacher-visual-small', 'cartpole-swingup'),
		default='reacher-visual-small',
	)
	parser.add_argument('--batch-size', type=int, default=2)
	parser.add_argument('--seed', type=int, default=57721)
	return parser.parse_args()


def make_config(args, *, belief=True, compile_enabled=None):
	cfg = OmegaConf.load(CONFIG_PATH)
	cfg.task = args.task
	cfg.obs = 'rgb'
	cfg.model_size = 5
	for key, value in MODEL_SIZE[5].items():
		cfg[key] = value
	cfg.multitask = False
	cfg.tasks = [cfg.task]
	cfg.task_dim = 0
	cfg.obs_shape = {'object': OBJECT_SHAPE}
	cfg.action_dim = 1 if cfg.task == 'cartpole-swingup' else 2
	cfg.episode_length = 500
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
	cfg.batch_size = args.batch_size
	cfg.horizon = 3
	cfg.compile = args.compile if compile_enabled is None else compile_enabled
	cfg.compile_fallback_random = bool(cfg.compile)
	cfg.flat_anchor = True
	cfg.flat_anchor_mode = 'cutie_object_only'
	cfg.cutie_object_last_valid_memory = False
	cfg.cutie_object_policy_burst_plan = None
	cfg.cutie_object_belief_enabled = bool(belief)
	cfg.cutie_object_belief_use_for_control = False
	cfg.cutie_object_belief_batch_size = args.batch_size
	cfg.cutie_object_belief_burn_in = 3
	cfg.cutie_object_belief_min_burst = 20
	cfg.cutie_object_belief_max_burst = 20
	cfg.cutie_object_belief_recovery_frames = 1
	cfg.cutie_object_belief_update_frequency = 1
	cfg.checkpoint = None
	cfg.num_samples = 32
	cfg.num_elites = 8
	cfg.num_pi_trajs = 4
	cfg.iterations = 2
	cfg.enable_wandb = False
	cfg.save_video = False
	cfg.seed = args.seed
	cfg.steps = 160
	cfg.buffer_size = 160
	return cfg_to_dataclass(cfg)


def make_objects(time, batch, device):
	raw = torch.randn(time + 2, batch, 2, 590, device=device)
	raw[..., 586:590] = torch.tensor(
		[0.9, 0.0, 1.0, 0.8], device=device
	)
	frames = torch.stack([raw[i:i + time] for i in range(3)], dim=-2)
	return frames.flatten(start_dim=-2)


def make_regular_sample(cfg, device):
	time = cfg.horizon + 1
	objects = make_objects(time, cfg.batch_size, device)
	obs = TensorDict(
		{'object': objects}, batch_size=(time, cfg.batch_size), device=device
	)
	action = torch.empty(
		cfg.horizon, cfg.batch_size, cfg.action_dim, device=device
	).uniform_(-1.0, 1.0)
	reward = torch.randn(cfg.horizon, cfg.batch_size, 1, device=device)
	terminated = torch.zeros_like(reward)
	return obs, action, reward, terminated


def make_belief_sample(cfg, device):
	time = (
		cfg.cutie_object_belief_burn_in
		+ cfg.cutie_object_belief_max_burst
		+ cfg.cutie_object_belief_recovery_frames
	)
	objects = make_objects(time, cfg.batch_size, device)
	obs = TensorDict(
		{'object': objects}, batch_size=(time, cfg.batch_size), device=device
	)
	action = torch.empty(
		time - 1, cfg.batch_size, cfg.action_dim, device=device
	).uniform_(-1.0, 1.0)
	return obs, action


class SyntheticBuffer:
	def __init__(self, regular, belief):
		self.regular = regular
		self.belief = belief

	def sample(self):
		return (*self.regular, None)

	def sample_belief(self):
		return self.belief


def make_replay_episode(episode_index, length, action_dim):
	"""Build an episode whose object/action values expose every time offset."""
	raw = torch.zeros(length + 2, 2, 590, dtype=torch.float32)
	for raw_index in range(length + 2):
		code = float(episode_index * 1000 + raw_index)
		raw[raw_index, :, :586] = code
		raw[raw_index, :, 586:590] = torch.tensor([0.9, 0.0, 1.0, 0.8])
	objects = torch.stack([raw[i:i + length] for i in range(3)], dim=-2)
	objects = objects.flatten(start_dim=-2)
	action = torch.empty(length, action_dim, dtype=torch.float32)
	action[0] = float('nan')
	for t in range(1, length):
		latest_code = float(episode_index * 1000 + t + 2)
		action[t, 0] = latest_code
		if action_dim > 1:
			action[t, 1:] = -latest_code
	return TensorDict(
		{
			'obs': TensorDict({'object': objects}, batch_size=(length,)),
			'action': action,
			'reward': torch.zeros(length),
			'terminated': torch.zeros(length),
		},
		batch_size=(length,),
	)


def exercise_real_replay(cfg):
	"""Exercise TorchRL SliceSampler alignment, wraparound, and RNG isolation."""
	buffer = Buffer(cfg)
	# Fill physical indices [0,150) with trajectories that are all too short for
	# the 24-step belief request.  The sole eligible trajectory is then written at
	# [150,160)+[0,14), forcing every long sample to cross the circular boundary.
	for episode_index in range(1, 16):
		buffer.add(make_replay_episode(
			episode_index, length=10, action_dim=cfg.action_dim
		))
	buffer.add(make_replay_episode(
		16, length=24, action_dim=cfg.action_dim
	))
	if len(buffer._buffer) != cfg.buffer_size:
		raise AssertionError('Replay wraparound did not fill the configured capacity.')
	base_sampler = buffer._buffer.sampler
	for _ in range(8):
		global_rng_before = torch.random.get_rng_state().clone()
		cuda_rng_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
		obs, action = buffer.sample_belief()
		if not torch.equal(global_rng_before, torch.random.get_rng_state()):
			raise AssertionError('Belief replay advanced the ordinary/global RNG.')
		if any(
			not torch.equal(before, after)
			for before, after in zip(cuda_rng_before, torch.cuda.get_rng_state_all())
		):
			raise AssertionError('Belief replay advanced a global CUDA RNG.')
		if buffer._buffer.sampler is not base_sampler:
			raise AssertionError('Belief replay did not restore the base sampler.')
		if tuple(obs.batch_size) != (
			buffer._belief_sequence_length, cfg.batch_size
		):
			raise AssertionError(f'Unexpected belief obs batch: {obs.batch_size}.')
		latest_code = obs['object'][..., 0, 2 * 590]
		episode_code = torch.floor(latest_code / 1000.0)
		if not torch.equal(
			episode_code, episode_code[0:1].expand_as(episode_code)
		):
			raise AssertionError('Belief replay crossed an episode sentinel.')
		if not torch.equal(
			episode_code, torch.full_like(episode_code, 16.0)
		):
			raise AssertionError(
				'Belief replay did not select the unique cross-boundary trajectory.'
			)
		if not torch.equal(
			latest_code[1:] - latest_code[:-1],
			torch.ones_like(latest_code[1:]),
		):
			raise AssertionError('Belief replay time sentinel is not contiguous.')
		if not torch.equal(action[..., 0], latest_code[1:]):
			raise AssertionError('Belief replay action is not aligned obs[t] -> obs[t+1].')
		if cfg.action_dim > 1 and not torch.equal(
			action[..., 1], -latest_code[1:]
		):
			raise AssertionError('Belief replay second action sentinel is misaligned.')
	# Counterfactual proof: inserting a belief sample must leave the very next
	# ordinary TD batch byte-identical under restored global RNG state.
	cpu_state = torch.random.get_rng_state().clone()
	cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
	expected_regular = buffer.sample()
	torch.random.set_rng_state(cpu_state)
	torch.cuda.set_rng_state_all(cuda_states)
	belief_rng_before = buffer._belief_replay_generator.get_state().clone()
	buffer.sample_belief()
	if torch.equal(
		belief_rng_before, buffer._belief_replay_generator.get_state()
	):
		raise AssertionError('Belief replay did not advance its private RNG.')
	regular = buffer.sample()
	if tuple(regular[0].batch_size) != (cfg.horizon + 1, cfg.batch_size):
		raise AssertionError('Base replay sampling broke after belief sampler swaps.')
	for key in regular[0].keys():
		if not torch.equal(regular[0][key], expected_regular[0][key]):
			raise AssertionError(f'Belief replay changed next base obs batch: {key}.')
	for index, label in ((1, 'action'), (2, 'reward'), (3, 'terminated')):
		if not torch.equal(regular[index], expected_regular[index]):
			raise AssertionError(f'Belief replay changed next base {label} batch.')
	if regular[4] is not None or expected_regular[4] is not None:
		raise AssertionError('Single-task replay unexpectedly returned task indices.')
	return buffer.metrics


def snapshot(module):
	return {
		name: parameter.detach().clone()
		for name, parameter in module.named_parameters()
		if parameter.requires_grad
	}


def max_change(before, module):
	after = dict(module.named_parameters())
	return max(
		float((after[name].detach() - value).abs().max())
		for name, value in before.items()
	)


def optimizer_ids(optimizer):
	values = [
		id(parameter)
		for group in optimizer.param_groups
		for parameter in group['params']
	]
	if len(values) != len(set(values)):
		raise AssertionError('An optimizer contains duplicate parameters.')
	return set(values)


def state_value_equal(left, right):
	"""Compare PyTorch state entries, including non-Tensor extra state."""
	if torch.is_tensor(left):
		return torch.is_tensor(right) and torch.equal(left, right)
	if type(left) is not type(right):
		return False
	if isinstance(left, dict):
		return left.keys() == right.keys() and all(
			state_value_equal(left[key], right[key]) for key in left
		)
	if isinstance(left, (tuple, list)):
		return len(left) == len(right) and all(
			state_value_equal(a, b) for a, b in zip(left, right)
		)
	try:
		return bool(left == right)
	except (TypeError, RuntimeError, ValueError):
		return False


def main():
	args = parse_args()
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required.')
	set_seed(args.seed)
	device = torch.device('cuda:0')
	cfg = make_config(args)
	replay_metrics = exercise_real_replay(cfg)
	agent = TDMPC2(cfg)
	if not agent._learned_object_belief:
		raise AssertionError('Learned belief mode did not activate.')
	if not hasattr(agent.model, '_belief_dynamics'):
		raise AssertionError('Belief transition is missing from WorldModel.')
	with tempfile.TemporaryDirectory() as directory:
		try:
			agent.save(Path(directory) / 'untrained-belief.pt')
		except RuntimeError:
			pass
		else:
			raise AssertionError('Untrained belief checkpoint was silently saved.')
	base_ids = optimizer_ids(agent.optim)
	pi_ids = optimizer_ids(agent.pi_optim)
	belief_ids = optimizer_ids(agent.belief_optim)
	if base_ids & belief_ids or pi_ids & belief_ids or base_ids & pi_ids:
		raise AssertionError('Base, policy, and belief optimizers must be disjoint.')
	if belief_ids != {
		id(parameter) for parameter in agent.model._belief_dynamics.parameters()
	}:
		raise AssertionError('Belief optimizer does not exactly cover its transition.')
	initial_world = agent.model._dynamics.state_dict()
	initial_belief = agent.model._belief_dynamics.state_dict()
	if set(initial_world) != set(initial_belief):
		raise AssertionError('Belief/world transition state keys differ at initialization.')
	for key in initial_world:
		if not torch.equal(initial_world[key], initial_belief[key]):
			raise AssertionError(f'Belief transition did not copy world dynamics: {key}')

	regular = make_regular_sample(cfg, device)
	long_sample = make_belief_sample(cfg, device)
	buffer = SyntheticBuffer(regular, long_sample)
	before_belief = snapshot(agent.model._belief_dynamics)
	before_world = snapshot(agent.model._dynamics)
	info = agent.update(buffer)
	for key, value in info.items():
		if torch.is_tensor(value) and not torch.isfinite(value).all():
			raise AssertionError(f'Non-finite update metric: {key}')
	if float(info['belief_aux_update']) != 1.0:
		raise AssertionError('Belief auxiliary update did not run.')
	if float(info['belief_age20_teacher_roles']) <= 0:
		raise AssertionError('Age-20 clean teacher received no supervision.')
	if float(info['belief_reacquisition_teacher_roles']) <= 0:
		raise AssertionError('Synthetic reacquisition received no supervision.')
	if max_change(before_belief, agent.model._belief_dynamics) <= 0:
		raise AssertionError('Belief transition parameters did not update.')
	if max_change(before_world, agent.model._dynamics) <= 0:
		raise AssertionError('Base world dynamics did not receive its normal update.')

	# Natural tracker invalidity can remove every age-20 clean teacher in one
	# stochastic batch. That age must be skipped without discarding valid earlier
	# ages or the first subsequently valid reacquisition target.
	sparse_obs = long_sample[0].clone()
	sparse_obs['object'][22, ..., -4:] = torch.tensor(
		[0.0, 1.0, 0.0, 0.0], device=device
	)
	# The same naturally invalid raw frame must remain invalid in the overlapping
	# middle stack slot of the following observation.
	sparse_obs['object'][23, ..., 590 + 586:590 + 590] = torch.tensor(
		[0.0, 1.0, 0.0, 0.0], device=device
	)
	sparse_missing = agent._sample_object_belief_missing(sparse_obs)
	before_sparse = snapshot(agent.model._belief_dynamics)
	sparse_info = agent._update_object_belief(
		sparse_obs, long_sample[1], sparse_missing
	)
	if float(sparse_info['belief_age20_teacher_roles']) != 0.0:
		raise AssertionError('Sparse teacher fixture unexpectedly retained age-20 labels.')
	if float(sparse_info['belief_age20_supervision_available']) != 0.0:
		raise AssertionError('Unavailable age-20 supervision was marked available.')
	if float(sparse_info['belief_reacquisition_supervision_available']) != 1.0:
		raise AssertionError('Valid delayed reacquisition supervision was discarded.')
	if float(sparse_info['belief_aux_update']) != 1.0:
		raise AssertionError('Partial-teacher belief batch did not update.')
	if max_change(before_sparse, agent.model._belief_dynamics) <= 0:
		raise AssertionError('Partial-teacher belief batch changed no parameters.')

	# A genuinely label-free auxiliary batch must leave both parameters and Adam
	# state bitwise unchanged, and must not masquerade as a successful update.
	empty_obs = long_sample[0].clone()
	empty_obs['object'][..., -4:] = torch.tensor(
		[0.0, 1.0, 0.0, 0.0], device=device
	)
	empty_missing = agent._sample_object_belief_missing(empty_obs)
	before_empty = snapshot(agent.model._belief_dynamics)
	optim_before_empty = copy.deepcopy(agent.belief_optim.state_dict())
	empty_info = agent._update_object_belief(
		empty_obs, long_sample[1], empty_missing
	)
	if float(empty_info['belief_aux_update']) != 0.0:
		raise AssertionError('Label-free belief batch was counted as an update.')
	if float(empty_info['belief_aux_no_teacher_skip']) != 1.0:
		raise AssertionError('Label-free belief batch was not recorded as skipped.')
	if max_change(before_empty, agent.model._belief_dynamics) != 0.0:
		raise AssertionError('Label-free belief batch changed transition parameters.')
	if not state_value_equal(optim_before_empty, agent.belief_optim.state_dict()):
		raise AssertionError('Label-free belief batch changed Adam state.')

	# Shadow mode must use the exact measurement planner and never touch the
	# online belief transition or episode-local belief state.
	agent.eval()
	single = TensorDict(
		{'object': regular[0]['object'][0, 0].detach().cpu()}, batch_size=()
	)
	prior_calls = [0]
	def count_prior(_module, _inputs, _output):
		prior_calls[0] += 1
	hook = agent.model._belief_dynamics.register_forward_hook(count_prior)
	shadow_action = agent.act(single, t0=True, eval_mode=True)
	hook.remove()
	if tuple(shadow_action.shape) != (cfg.action_dim,) or not torch.isfinite(
		shadow_action
	).all():
		raise AssertionError('Shadow measurement action is invalid.')
	if prior_calls[0] != 0:
		raise AssertionError('Shadow control unexpectedly invoked belief dynamics.')
	if (
		agent._online_object_belief is not None
		or agent._online_object_belief_action is not None
	):
		raise AssertionError('Shadow control created online belief state.')

	# Explicit held-out deployment may enable the learned prior before reset.
	agent.set_object_belief_control_for_evaluation(True)
	action0 = agent.act(single, t0=True, eval_mode=True)
	if tuple(action0.shape) != (cfg.action_dim,) or not torch.isfinite(action0).all():
		raise AssertionError('Initial belief action is invalid.')
	missing = single.clone()
	missing['object'][0, -590:] = 0.0
	missing['object'][0, -4:] = torch.tensor([0.0, 1.0, 0.0, 0.0])
	action1 = agent.act(missing, t0=False, eval_mode=True)
	if not torch.isfinite(action1).all() or agent._online_object_belief_prior_uses < 1:
		raise AssertionError('Missing role did not invoke the online belief prior.')
	agent.reset_object_belief()
	if agent._online_object_belief is not None or agent._online_object_belief_action is not None:
		raise AssertionError('Episode reset retained online belief state.')

	with tempfile.TemporaryDirectory() as directory:
		belief_path = Path(directory) / 'belief.pt'
		hard_path = Path(directory) / 'hard.pt'
		agent.save(belief_path)
		payload = torch.load(belief_path, map_location='cpu', weights_only=False)
		contract = payload.get('checkpoint_contract')
		if not isinstance(contract, dict):
			raise AssertionError('Belief checkpoint contract is missing.')
		if contract.get('cutie_object_belief_aux_updates') != 2:
			raise AssertionError('Belief checkpoint auxiliary update count is wrong.')
		if contract.get(
			'cutie_object_belief_use_for_control_during_training'
		) is not False or contract.get(
			'cutie_object_belief_collection_mode'
		) != 'measurement_only_shadow':
			raise AssertionError('Shadow checkpoint collection metadata is wrong.')
		supervision = contract.get('cutie_object_belief_supervision')
		if not isinstance(supervision, dict) or supervision.get('attempts') != 3:
			raise AssertionError('Belief supervision attempt metadata is wrong.')
		if supervision.get('successful_updates') != 2:
			raise AssertionError('Belief supervision successful-update count is wrong.')
		if supervision.get('no_teacher_skips') != 1:
			raise AssertionError('Belief supervision skip count is wrong.')
		if min(supervision.get('age20_teacher_roles', [0, 0])) < 1:
			raise AssertionError('Belief age-20 per-role cumulative coverage is missing.')
		if min(supervision.get('reacquisition_teacher_roles', [0, 0])) < 1:
			raise AssertionError('Belief reacquisition per-role cumulative coverage is missing.')
		reloaded = TDMPC2(make_config(args, compile_enabled=False))
		reloaded.load(belief_path)
		if reloaded._use_object_belief_for_control:
			raise AssertionError('Shadow checkpoint reload enabled online control.')
		if reloaded._belief_aux_updates != 2:
			raise AssertionError('Belief reload did not restore auxiliary update count.')
		if reloaded._belief_aux_attempts != 3 or reloaded._belief_aux_no_teacher_skips != 1:
			raise AssertionError('Belief reload did not restore supervision counters.')
		reloaded_state = reloaded.model.state_dict()
		for key, value in agent.model.state_dict().items():
			other = reloaded_state[key]
			if not state_value_equal(value, other):
				raise AssertionError(
					'Belief checkpoint reload differs: '
					f'{key} ({type(value).__name__}).'
				)
		hard = TDMPC2(make_config(args, belief=False, compile_enabled=False))
		hard.save(hard_path)
		for target, source, label in (
			(hard, belief_path, 'hard<-belief'),
			(reloaded, hard_path, 'belief<-hard'),
		):
			try:
				target.load(source)
			except RuntimeError:
				pass
			else:
				raise AssertionError(f'Cross-mode checkpoint was accepted: {label}')

	print('CUTIE_LEARNED_BELIEF_UPDATE_OK', {
		'task': args.task,
		'compile': bool(args.compile),
		'batch_size': int(args.batch_size),
		'belief_loss': float(info['belief_loss']),
		'belief_age20_loss': float(info['belief_age20_loss']),
		'belief_reacquisition_loss': float(info['belief_reacquisition_loss']),
		'belief_prior_uses': int(agent._online_object_belief_prior_uses),
		'belief_replay_rng_isolated': replay_metrics['belief_replay_rng_isolated'],
		'belief_sequence_length': replay_metrics['belief_sequence_length'],
	})


if __name__ == '__main__':
	main()
