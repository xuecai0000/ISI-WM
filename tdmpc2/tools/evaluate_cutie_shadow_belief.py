"""Strict same-checkpoint evaluator for the Cutie shadow-belief pilot.

The source checkpoint was collected with the learned belief trained only as an
auxiliary objective (``cutie_object_belief_use_for_control=false``).  Each
invocation is one fresh process and selects exactly one evaluation-only control
mode before the first episode: current measurement or the learned online prior.
Normal validation and the immutable canonical burst-20 intervention are both
supported.  This is a single-training-seed development diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path
from time import perf_counter


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
for local_path in (str(REPO_DIR), str(PROJECT_DIR)):
	while local_path in sys.path:
		sys.path.remove(local_path)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))

from tdmpc2.tools import evaluate_cutie_multitask_checkpoint as base
from tdmpc2.tools import evaluate_cutie_multitask_policy_burst as burst_base


FORMAT = 'cutie_shadow_belief_evaluation_v1'
SCIENTIFIC_SCOPE = (
	'single-training-seed same-shadow-checkpoint inference comparison; paired '
	'episode intervals do not quantify uncertainty across training seeds'
)
BACKEND = 'cutie_object_only'
TASK_ROLES = {
	'reacher-visual-small': ('whole_arm', 'goal'),
	'cartpole-swingup': ('cart', 'pole'),
}
TARGET_ROLES = {
	'reacher-visual-small': 'whole_arm',
	'cartpole-swingup': 'pole',
}
ARMS = ('measurement_only', 'learned_prior')
CONDITIONS = ('normal', 'burst_20')
EPISODES = 20
DECISION_STEPS = 500
NUM_ROLES = 2
FRAME_DIM = 590
STACK_FRAMES = 3
STACKED_DIM = FRAME_DIM * STACK_FRAMES
ROLE_DIM = 64
LATENT_DIM = NUM_ROLES * ROLE_DIM
EROSION_PIXELS = 0
EXPECTED_PRIOR_FORWARD_CALLS = EPISODES * (DECISION_STEPS - 1)


def _require(condition: bool, message: str) -> None:
	if not condition:
		raise RuntimeError(message)


def _hash_valid(value) -> bool:
	return (
		isinstance(value, str) and len(value) == 64
		and all(character in '0123456789abcdef' for character in value)
	)


def _trace_tensor(digest, tensor) -> None:
	array = tensor.detach().cpu().contiguous().numpy()
	digest.update(str(array.dtype).encode('ascii'))
	digest.update(struct.pack('<I', array.ndim))
	for size in array.shape:
		digest.update(struct.pack('<Q', int(size)))
	digest.update(array.tobytes(order='C'))


def _validate_source(args, raw: dict) -> None:
	if args.training_seed != 7:
		raise ValueError('Frozen shadow pilot requires training seed 7.')
	if raw.get('cutie_object_role_names') != list(TASK_ROLES[args.task]):
		raise ValueError('Source role names do not match the frozen task contract.')
	if raw.get('flat_anchor') is not True or raw.get('flat_anchor_mode') != BACKEND:
		raise ValueError('Source is not structural CutieObjectOnly.')
	if raw.get('cutie_object_belief_enabled') is not True:
		raise ValueError('Shadow source must train the learned belief.')
	if raw.get('cutie_object_belief_use_for_control') is not False:
		raise ValueError('Shadow source must collect with belief control disabled.')
	if raw.get('cutie_object_last_valid_memory') is not False:
		raise ValueError('Shadow source must disable last-valid memory.')
	if raw.get('cutie_object_policy_burst_plan') is not None:
		raise ValueError('Training source must have policy burst plan null.')
	for key, expected in {
		'latent_dim': LATENT_DIM,
		'cutie_object_num_roles': NUM_ROLES,
		'cutie_object_frame_dim': FRAME_DIM,
		'cutie_object_stack_frames': STACK_FRAMES,
		'cutie_object_input_dim': STACKED_DIM,
		'cutie_object_role_dim': ROLE_DIM,
		'cutie_object_only_latent_dim': LATENT_DIM,
		'cutie_object_belief_batch_size': 128,
		'cutie_object_belief_burn_in': 3,
		'cutie_object_belief_min_burst': 5,
		'cutie_object_belief_max_burst': 20,
		'cutie_object_belief_recovery_frames': 1,
		'cutie_object_belief_update_frequency': 4,
	}.items():
		if int(raw.get(key, -1)) != expected:
			raise ValueError(f'Source {key}={raw.get(key)!r}, expected {expected}.')
	obs_shape = raw.get('obs_shape')
	if obs_shape is not None and obs_shape != {'object': [NUM_ROLES, STACKED_DIM]}:
		raise ValueError(f'Unexpected source obs_shape {obs_shape!r}.')


def _prepare(args, raw: dict):
	data = dict(raw)
	if args.condition == 'burst_20':
		data['cutie_object_policy_burst_plan'] = str(
			args.policy_burst_plan.resolve()
		)
	cfg = base._prepare(args, data)
	_require(bool(cfg.get('cutie_object_belief_enabled', False)), (
		'Evaluation config disabled the learned-belief checkpoint contract.'
	))
	_require(not bool(cfg.get('cutie_object_belief_use_for_control', True)), (
		'Evaluation config mutated the frozen shadow collection mode.'
	))
	_require(not bool(cfg.cutie_object_last_valid_memory), (
		'Evaluation config enabled last-valid memory.'
	))
	expected_plan = (
		args.policy_burst_plan.resolve()
		if args.condition == 'burst_20' else None
	)
	actual_plan = cfg.cutie_object_policy_burst_plan
	if expected_plan is None:
		_require(actual_plan is None, 'Normal evaluation installed a burst plan.')
	else:
		_require(Path(str(actual_plan)).resolve() == expected_plan, (
			'Evaluation did not bind the exact burst plan.'
		))
	_require(not bool(cfg.compile), 'Shadow evaluation requires compile=false.')
	return cfg


def _load_checkpoint_contract(path: Path, torch) -> dict:
	payload = torch.load(path, map_location='cpu', weights_only=False)
	if not isinstance(payload, dict):
		raise RuntimeError('Shadow checkpoint payload is not a dictionary.')
	contract = payload.get('checkpoint_contract')
	if not isinstance(contract, dict):
		raise RuntimeError('Shadow checkpoint contract is missing.')
	expected = {
		'format': 'tdmpc2_checkpoint_contract_v1',
		'flat_anchor_mode': BACKEND,
		'latent_dim': LATENT_DIM,
		'cutie_object_belief_enabled': True,
		'cutie_object_belief_schema': {
			'num_roles': NUM_ROLES,
			'frame_dim': FRAME_DIM,
			'stack_frames': STACK_FRAMES,
			'input_dim': STACKED_DIM,
			'role_dim': ROLE_DIM,
		},
	}
	bad = {
		key: (contract.get(key), expected_value)
		for key, expected_value in expected.items()
		if contract.get(key) != expected_value
	}
	if bad:
		raise RuntimeError(f'Shadow checkpoint contract mismatch: {bad}.')
	training_control = contract.get(
		'cutie_object_belief_use_for_control_during_training'
	)
	collection_mode = contract.get('cutie_object_belief_collection_mode')
	if training_control is not False:
		raise RuntimeError('Checkpoint claims belief was used for training control.')
	if collection_mode != 'measurement_only_shadow':
		raise RuntimeError('Checkpoint was not collected in measurement-only mode.')
	online_control = contract.get(
		'cutie_object_belief_online_control_before_checkpoint_save'
	)
	if online_control != {'prior_role_uses': 0, 'episode_state_resets': 0}:
		raise RuntimeError(
			'Shadow checkpoint reports non-zero online belief use during training.'
		)
	aux_updates = contract.get('cutie_object_belief_aux_updates')
	if isinstance(aux_updates, bool) or not isinstance(aux_updates, int) or aux_updates < 1:
		raise RuntimeError('Checkpoint auxiliary-update count must be positive.')
	supervision = contract.get('cutie_object_belief_supervision')
	if not isinstance(supervision, dict) or supervision.get('format') != (
		'cutie_object_belief_supervision_v1'
	):
		raise RuntimeError('Checkpoint belief supervision contract is missing.')
	integer_fields = (
		'attempts', 'successful_updates', 'no_teacher_skips',
		'age20_available_updates', 'reacquisition_available_updates',
	)
	if any(
		isinstance(supervision.get(key), bool)
		or not isinstance(supervision.get(key), int)
		or supervision.get(key) < 0
		for key in integer_fields
	):
		raise RuntimeError('Checkpoint belief supervision counts are invalid.')
	for name in ('age20_teacher_roles', 'reacquisition_teacher_roles'):
		values = supervision.get(name)
		if (
			not isinstance(values, list) or len(values) != NUM_ROLES
			or any(isinstance(value, bool) or not isinstance(value, int) or value < 1
				for value in values)
		):
			raise RuntimeError(f'Checkpoint {name} coverage is invalid.')
	if (
		supervision['successful_updates'] != aux_updates
		or supervision['successful_updates'] + supervision['no_teacher_skips']
		!= supervision['attempts']
		or supervision['age20_available_updates'] > supervision['attempts']
		or supervision['reacquisition_available_updates'] > supervision['attempts']
	):
		raise RuntimeError('Checkpoint belief supervision counts are inconsistent.')
	return json.loads(json.dumps(contract, allow_nan=False))


def _observation_validity(obs, belief_helpers, torch):
	try:
		keys = set(obs.keys())
	except (AttributeError, TypeError) as exc:
		raise RuntimeError('ObjectOnly observation is not keyed.') from exc
	_require(keys == {'object'}, f'ObjectOnly exposed keys {sorted(keys)}.')
	objects = obs['object']
	_require(isinstance(objects, torch.Tensor), 'Object observation is not a tensor.')
	_require(tuple(objects.shape) == (NUM_ROLES, STACKED_DIM), (
		f'Object observation shape is {tuple(objects.shape)}.'
	))
	_require(objects.dtype == torch.float32, 'Object observation is not float32.')
	_require(bool(torch.isfinite(objects).all().item()), (
		'Object observation contains non-finite values.'
	))
	valid = belief_helpers.latest_valid(objects)
	_require(tuple(valid.shape) == (NUM_ROLES,), 'Role validity has wrong shape.')
	return valid


def _normal_checks(args, perception, records, alignment_draws, runtime):
	intervention = perception.get('policy_observation_intervention', {})
	memory = perception.get('last_valid_memory', {})
	checks = {
		'perception_frames_10020': perception.get('frames')
			== EPISODES * (DECISION_STEPS + 1),
		'worker_restarts_zero': perception.get('worker_restarts') == 0,
		'timeouts_zero': perception.get('timeouts') == 0,
		'memory_disabled': memory.get('enabled') is False,
		'memory_substitutions_zero': memory.get('substitutions') == 0,
		'intervention_disabled': intervention.get('enabled') is False,
		'intervention_applied_zero': intervention.get('applied_role_frames') == 0,
		'episode_records_20': len(records) == EPISODES,
		'episode_lengths_500': all(row['length'] == DECISION_STEPS for row in records),
		'alignment_draws_10000': alignment_draws == EPISODES * DECISION_STEPS,
		'role_accounting_exact': all(
			runtime['role_valid_decisions'][role]
			+ runtime['role_invalid_decisions'][role]
			== EPISODES * DECISION_STEPS for role in range(NUM_ROLES)
		),
	}
	return checks


def _role_diagnostic_checks(args, perception):
	frames = perception.get('frames')
	roles = tuple(TASK_ROLES[args.task])
	role_metrics = perception.get('role_metrics')
	episodes = perception.get('episode_metrics')
	frames_valid = (
		isinstance(frames, int) and not isinstance(frames, bool)
		and frames == EPISODES * (DECISION_STEPS + 1)
	)
	role_structure = (
		isinstance(role_metrics, dict) and set(role_metrics) == set(roles)
	)
	role_accounting = frames_valid and role_structure and all(
		isinstance(role_metrics[role], dict)
		and all(
			isinstance(role_metrics[role].get(key), int)
			and not isinstance(role_metrics[role][key], bool)
			and 0 <= role_metrics[role][key] <= frames
			for key in ('valid_frames', 'invalid_frames')
		)
		and role_metrics[role]['valid_frames']
			+ role_metrics[role]['invalid_frames'] == frames
		and all(
			type(role_metrics[role].get(key)) in (int, float)
			and math.isfinite(float(role_metrics[role][key]))
			and 0.0 <= float(role_metrics[role][key]) <= 1.0
			for key in (
				'valid_frame_rate', 'lost_frame_rate',
				'empty_mask_frame_rate', 'nonfinite_feature_frame_rate',
				'mask_touches_border_rate',
			)
		)
		for role in roles
	)
	def valid_episode(row, index):
		if not isinstance(row, dict):
			return False
		invalid = row.get('per_role_invalid_frames')
		bursts = row.get('per_role_max_invalid_burst')
		return (
			row.get('episode_index') == index
			and row.get('frames') == DECISION_STEPS + 1
			and isinstance(invalid, dict) and set(invalid) == set(roles)
			and isinstance(bursts, dict) and set(bursts) == set(roles)
			and all(
				isinstance(invalid[role], int)
				and not isinstance(invalid[role], bool)
				and 0 <= invalid[role] <= DECISION_STEPS + 1
				and isinstance(bursts[role], int)
				and not isinstance(bursts[role], bool)
				and 0 <= bursts[role] <= DECISION_STEPS + 1
				for role in roles
			)
		)
	episode_structure = (
		frames_valid and isinstance(episodes, list) and len(episodes) == EPISODES
		and all(valid_episode(row, index) for index, row in enumerate(episodes))
		and sum(row['frames'] for row in episodes) == frames
	)
	return {
		'role_diagnostics_frames_exact': frames_valid,
		'role_diagnostics_schema_exact': perception.get(
			'role_diagnostics_schema'
		) == 'cutie_role_runtime_diagnostics_v1',
		'role_metrics_structure_exact': role_structure,
		'role_metrics_accounting_exact': role_accounting,
		'episode_metrics_structure_exact': episode_structure,
	}


def evaluate(args):
	import numpy as np
	import torch
	from common import cutie_object_belief
	from common.seed import set_seed
	from envs import make_env
	from tdmpc2.tdmpc2 import TDMPC2

	base._validate_args(args)
	runtime_sha_before = base._sha256(args.runtime_config)
	checkpoint_sha_before = base._sha256(args.checkpoint)
	implementation_paths = {
		'evaluator': Path(__file__).resolve(),
		'base_evaluator': Path(base.__file__).resolve(),
		'burst_base_evaluator': Path(burst_base.__file__).resolve(),
		'agent': (PROJECT_DIR / 'tdmpc2.py').resolve(),
		'world_model': (PROJECT_DIR / 'common' / 'world_model.py').resolve(),
		'belief_helpers': (PROJECT_DIR / 'common' / 'cutie_object_belief.py').resolve(),
		'object_wrapper': (
			PROJECT_DIR / 'envs' / 'wrappers' / 'cutie_object.py'
		).resolve(),
	}
	implementation_sha_before = {
		name: base._sha256(path) for name, path in implementation_paths.items()
	}
	raw = base._json(args.runtime_config)
	_validate_source(args, raw)
	plan = None
	plan_sha_before = None
	if args.condition == 'burst_20':
		plan, plan_sha_before = burst_base._validate_plan(args)
	cfg = _prepare(args, raw)
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required.')
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.set_float32_matmul_precision('high')
	checkpoint_contract = _load_checkpoint_contract(args.checkpoint, torch)

	env = None
	agent = None
	hook_handle = None
	records = []
	alignment_draws = 0
	role_valid_decisions = [0, 0]
	role_invalid_decisions = [0, 0]
	role_invalid_initial_decisions = [0, 0]
	belief_forward_calls = 0
	prior_input_trace = hashlib.sha256()
	prior_output_trace = hashlib.sha256()
	perception = ready = manifest = combined_manifest = None
	wrapper_plan_sha = None
	actual_erosion = None
	started = perf_counter()

	def belief_forward_hook(_module, inputs, output):
		nonlocal belief_forward_calls
		_require(len(inputs) == 2, 'Belief transition hook received wrong inputs.')
		belief, action = inputs
		_require(tuple(belief.shape) == (1, LATENT_DIM), (
			f'Belief input shape is {tuple(belief.shape)}.'
		))
		_require(tuple(action.shape) == (1, int(agent.cfg.action_dim)), (
			f'Belief action shape is {tuple(action.shape)}.'
		))
		_require(tuple(output.shape) == (1, LATENT_DIM), (
			f'Belief output shape is {tuple(output.shape)}.'
		))
		_require(all(value.dtype == torch.float32 for value in (belief, action, output)), (
			'Belief transition observed non-float32 tensors.'
		))
		_require(all(bool(torch.isfinite(value).all().item())
			for value in (belief, action, output)), (
			'Belief transition observed non-finite tensors.'
		))
		_trace_tensor(prior_input_trace, belief)
		_trace_tensor(prior_input_trace, action)
		_trace_tensor(prior_output_trace, output)
		belief_forward_calls += 1

	try:
		set_seed(args.env_seed)
		env = make_env(cfg)
		_require(base._env_value(env, 'active_split') == base.VALIDATION_SPLIT, (
			'Validation background split was not constructed.'
		))
		actual_erosion = base._env_value(env, 'erosion_pixels')
		_require(actual_erosion in (None, EROSION_PIXELS), (
			f'Evaluation erosion is {actual_erosion!r}.'
		))
		wrapper_plan_sha = base._env_value(env, 'policy_burst_plan_sha256')
		if args.condition == 'normal':
			_require(wrapper_plan_sha is None, 'Normal wrapper loaded a burst plan.')
		else:
			_require(wrapper_plan_sha == plan_sha_before, (
				'Wrapper loaded a different burst plan.'
			))
		agent = TDMPC2(cfg)
		agent.load(args.checkpoint)
		agent.eval()
		setter = getattr(
			agent, 'set_object_belief_control_for_evaluation', None
		)
		_require(callable(setter), 'Evaluation-only belief control API is missing.')
		setter(args.arm == 'learned_prior')
		_require(bool(agent._use_object_belief_for_control) == (
			args.arm == 'learned_prior'
		), (
			'Agent did not enter the requested evaluation-only belief mode.'
		))
		_require(bool(agent._learned_object_belief), (
			'Loaded agent did not instantiate the learned belief.'
		))
		_require(int(agent._belief_aux_updates) == checkpoint_contract[
			'cutie_object_belief_aux_updates'
		], 'Loaded auxiliary-update count differs from checkpoint contract.')
		hook_handle = agent.model._belief_dynamics.register_forward_hook(
			belief_forward_hook
		)

		for episode_index in range(EPISODES):
			obs = env.reset()
			initial_valid = _observation_validity(obs, cutie_object_belief, torch)
			initial_rgb = base._rgb_hash(obs, env, BACKEND)
			initial_object = base._object_hash(obs, BACKEND)
			initial_raw_object = base._env_value(
				env, 'latest_raw_object_frame_sha256'
			)
			source = base._env_value(env, 'active_source')
			frame_index = base._env_value(env, 'frame_index')
			_require(source is not None and frame_index is not None, (
				'Initial background provenance is unavailable.'
			))
			if args.condition == 'burst_20':
				_require(_hash_valid(initial_raw_object), (
					'Initial raw object-frame hash is invalid.'
				))

			planner_seed = args.planner_seed_base + episode_index
			set_seed(planner_seed)
			agent._prev_mean.zero_()
			reset_before = int(agent._online_object_belief_resets)
			had_state = (
				agent._online_object_belief is not None
				or agent._online_object_belief_action is not None
			)
			agent.reset_object_belief()
			reset_delta = int(agent._online_object_belief_resets) - reset_before
			expected_reset = int(args.arm == 'learned_prior' and episode_index > 0)
			_require(reset_delta == expected_reset, (
				f'Episode {episode_index} reset delta is {reset_delta}.'
			))
			_require(had_state == bool(expected_reset), (
				f'Episode {episode_index} pre-reset state is inconsistent.'
			))

			rng_start = base._rng_hash(torch)
			action_trace = hashlib.sha256()
			reward_trace = hashlib.sha256()
			validity_trace = hashlib.sha256()
			reward_sum = 0.0
			info = {}
			episode_valid = [0, 0]
			episode_invalid = [0, 0]
			prior_calls_before = belief_forward_calls
			prior_uses_before = int(agent._online_object_belief_prior_uses)

			for step_index in range(DECISION_STEPS):
				torch.compiler.cudagraph_mark_step_begin()
				base._align_object_only_rgb_shift_rng(torch, BACKEND)
				alignment_draws += 1
				valid = _observation_validity(obs, cutie_object_belief, torch)
				_trace_tensor(validity_trace, valid)
				for role_index in range(NUM_ROLES):
					if bool(valid[role_index].item()):
						episode_valid[role_index] += 1
						role_valid_decisions[role_index] += 1
					else:
						episode_invalid[role_index] += 1
						role_invalid_decisions[role_index] += 1
						if step_index == 0:
							role_invalid_initial_decisions[role_index] += 1
				calls_before_step = belief_forward_calls
				action = agent.act(obs, t0=step_index == 0, eval_mode=True)
				expected_step_calls = int(
					args.arm == 'learned_prior' and step_index > 0
				)
				_require(belief_forward_calls - calls_before_step == expected_step_calls, (
					f'Episode {episode_index} step {step_index} belief call mismatch.'
				))
				_require(tuple(action.shape) == (int(agent.cfg.action_dim),), (
					'Controller returned a wrong-shape action.'
				))
				_require(action.dtype == torch.float32, (
					'Controller returned a non-float32 action.'
				))
				_require(bool(torch.isfinite(action).all().item()), (
					'Controller returned a non-finite action.'
				))
				_trace_tensor(action_trace, action)
				obs, reward, done, info = env.step(action)
				reward_sum += float(reward)
				reward_trace.update(struct.pack('<d', float(reward)))
				if done:
					length = step_index + 1
					break
			else:
				raise RuntimeError('Environment did not terminate at 500 steps.')
			_require(length == DECISION_STEPS, f'Unexpected episode length {length}.')

			end_source = base._env_value(env, 'active_source')
			end_frame_index = base._env_value(env, 'frame_index')
			_require(end_source is not None and end_frame_index is not None, (
				'Final background provenance is unavailable.'
			))
			final_object = base._object_hash(obs, BACKEND)
			final_raw_object = base._env_value(
				env, 'latest_raw_object_frame_sha256'
			)
			prior_calls = belief_forward_calls - prior_calls_before
			prior_uses = int(agent._online_object_belief_prior_uses) - prior_uses_before
			prior_opportunities = sum(episode_invalid) - int(
				(~initial_valid).sum().item()
			)
			_require(prior_calls == (
				DECISION_STEPS - 1 if args.arm == 'learned_prior' else 0
			), f'Episode {episode_index} prior call count is {prior_calls}.')
			_require(prior_uses == (
				prior_opportunities if args.arm == 'learned_prior' else 0
			), f'Episode {episode_index} prior role-use count is {prior_uses}.')

			record = {
				'episode_index': episode_index,
				'planner_seed': planner_seed,
				'planner_rng_start_sha256': rng_start,
				'planner_rng_end_sha256': base._rng_hash(torch),
				'initial_rgb_sha256': initial_rgb,
				'initial_object_sha256': initial_object,
				'final_object_sha256': final_object,
				'background_source': Path(source).name,
				'background_start_frame_index': int(frame_index),
				'background_end_source': Path(end_source).name,
				'background_end_frame_index': int(end_frame_index),
				'action_trace_sha256': action_trace.hexdigest(),
				'reward_trace_sha256': reward_trace.hexdigest(),
				'role_validity_trace_sha256': validity_trace.hexdigest(),
				'role_valid_decisions': episode_valid,
				'role_invalid_decisions': episode_invalid,
				'initial_role_valid': [bool(value) for value in initial_valid.tolist()],
				'prior_role_opportunities': prior_opportunities,
				'belief_dynamics_forward_calls': prior_calls,
				'prior_role_uses': prior_uses,
				'belief_reset_delta': reset_delta,
				'reward': reward_sum,
				'success': float(info.get('success', 0.0)),
				'length': length,
			}
			if args.condition == 'burst_20':
				record.update({
					'initial_policy_object_sha256': initial_object,
					'initial_raw_object_frame_sha256': initial_raw_object,
					'final_policy_object_sha256': final_object,
					'final_raw_object_frame_sha256': final_raw_object,
					'policy_burst_event': dict(plan['events'][episode_index]),
					'policy_burst_plan_sha256': wrapper_plan_sha,
				})
			records.append(record)
			print('CUTIE_SHADOW_BELIEF_EPISODE', json.dumps({
				'task': args.task,
				'condition': args.condition,
				'arm': args.arm,
				'episode_index': episode_index,
				'reward': reward_sum,
				'role_invalid_decisions': episode_invalid,
				'prior_role_uses': prior_uses,
			}, allow_nan=False), flush=True)

		perception = base._metrics(env)
		ready = base._env_value(env, 'cutie_ready')
		manifest = base._env_value(env, 'manifest_sha256')
		combined_manifest = base._env_value(env, 'combined_manifest_sha256')
	finally:
		if hook_handle is not None:
			hook_handle.remove()
		if env is not None and callable(getattr(env, 'close', None)):
			env.close()

	runtime_sha_after = base._sha256(args.runtime_config)
	checkpoint_sha_after = base._sha256(args.checkpoint)
	implementation_sha_after = {
		name: base._sha256(path) for name, path in implementation_paths.items()
	}
	_require(runtime_sha_after == runtime_sha_before, (
		'runtime_config changed during evaluation.'
	))
	_require(checkpoint_sha_after == checkpoint_sha_before, (
		'checkpoint changed during evaluation.'
	))
	_require(implementation_sha_after == implementation_sha_before, (
		'Evaluator implementation changed during evaluation.'
	))
	_require(isinstance(perception, dict), 'Perception metrics are unavailable.')
	_require(isinstance(ready, dict), 'Cutie ready provenance is unavailable.')

	expected_calls = EXPECTED_PRIOR_FORWARD_CALLS if args.arm == 'learned_prior' else 0
	expected_uses = (
		sum(int(row['prior_role_opportunities']) for row in records)
		if args.arm == 'learned_prior' else 0
	)
	inference_runtime = {
		'mode': args.arm,
		'enabled': bool(agent._learned_object_belief),
		'loaded_aux_updates': int(agent._belief_aux_updates),
		'belief_dynamics_forward_calls': int(belief_forward_calls),
		'expected_belief_dynamics_forward_calls': expected_calls,
		'prior_role_uses': int(agent._online_object_belief_prior_uses),
		'expected_prior_role_uses': expected_uses,
		'belief_resets': int(agent._online_object_belief_resets),
		'role_names': list(TASK_ROLES[args.task]),
		'role_valid_decisions': role_valid_decisions,
		'role_invalid_decisions': role_invalid_decisions,
		'role_invalid_initial_decisions': role_invalid_initial_decisions,
		'online_belief_state_is_none': agent._online_object_belief is None,
		'online_belief_action_is_none': agent._online_object_belief_action is None,
		'belief_prior_input_trace_sha256': prior_input_trace.hexdigest(),
		'belief_prior_output_trace_sha256': prior_output_trace.hexdigest(),
		'evaluation_control_api': (
			'TDMPC2.set_object_belief_control_for_evaluation'
		),
	}
	checks = {
		'checkpoint_belief_enabled': inference_runtime['enabled'] is True,
		'control_mode_exact': bool(agent._use_object_belief_for_control) == (
			args.arm == 'learned_prior'
		),
		'belief_forward_calls_exact': belief_forward_calls == expected_calls,
		'prior_role_uses_exact': inference_runtime['prior_role_uses'] == expected_uses,
		'belief_resets_exact': inference_runtime['belief_resets'] == (
			EPISODES - 1 if args.arm == 'learned_prior' else 0
		),
		'measurement_state_absent': (
			agent._online_object_belief is None
			and agent._online_object_belief_action is None
			if args.arm == 'measurement_only' else True
		),
		'action_traces_valid': all(
			_hash_valid(row['action_trace_sha256']) for row in records
		),
		'pairing_hashes_valid': all(
			all(_hash_valid(row[key]) for key in (
				'initial_rgb_sha256', 'initial_object_sha256',
				'planner_rng_start_sha256', 'planner_rng_end_sha256',
			)) for row in records
		),
	}
	checks.update(_role_diagnostic_checks(args, perception))
	if args.condition == 'normal':
		checks.update(_normal_checks(
			args, perception, records, alignment_draws, inference_runtime
		))
		policy_burst = None
		pairing_fields = [
			'episode_index', 'planner_seed', 'planner_rng_start_sha256',
			'planner_rng_end_sha256', 'initial_rgb_sha256',
			'initial_object_sha256', 'background_source',
			'background_start_frame_index', 'background_end_source',
			'background_end_frame_index', 'length',
		]
	else:
		plan_sha_after = base._sha256(args.policy_burst_plan)
		# The immutable v1 intervention validator distinguishes only the wrapper
		# content modes hard-zero/last-valid/legacy-learned.  Both shadow arms keep
		# the wrapper in hard-zero mode; their difference is solely inside TDMPC2.
		intervention_args = argparse.Namespace(**vars(args))
		intervention_args.arm = 'hard_zero'
		checks.update(burst_base._validate_intervention(
			intervention_args, perception, plan, plan_sha_before, plan_sha_after,
			wrapper_plan_sha, records, alignment_draws,
		))
		intervention = perception['policy_observation_intervention']
		overwrite_rate = intervention['raw_valid_overwritten'] / (
			EPISODES * args.expected_length
		)
		policy_burst = {
			'format': burst_base.PLAN_FORMAT,
			'role': args.expected_role,
			'length': args.expected_length,
			'starts': list(burst_base.STARTS),
			'episodes': EPISODES,
			'plan_sha256_before': plan_sha_before,
			'plan_sha256_after': plan_sha_after,
			'wrapper_plan_sha256': wrapper_plan_sha,
			'minimum_raw_valid_overwrite_rate': burst_base.MIN_RAW_VALID_OVERWRITE_RATE,
			'raw_valid_overwrite_rate': overwrite_rate,
			'controlled_burst_attribution_eligible': overwrite_rate
				>= burst_base.MIN_RAW_VALID_OVERWRITE_RATE,
		}
		pairing_fields = [
			'episode_index', 'planner_seed', 'planner_rng_start_sha256',
			'planner_rng_end_sha256', 'initial_rgb_sha256',
			'initial_policy_object_sha256', 'initial_raw_object_frame_sha256',
			'background_source', 'background_start_frame_index',
			'background_end_source', 'background_end_frame_index', 'length',
			'policy_burst_event', 'policy_burst_plan_sha256',
		]
	failed = sorted(name for name, passed in checks.items() if not passed)
	if failed:
		raise RuntimeError('Shadow-belief strict checks failed: ' + ', '.join(failed))

	rewards = np.asarray([row['reward'] for row in records], dtype=np.float64)
	_require(rewards.size == EPISODES and np.isfinite(rewards).all(), (
		'Incomplete or non-finite rewards.'
	))
	checkpoint = Path(raw['cutie_object_checkpoint']).resolve()
	support = Path(raw['cutie_object_support_path']).resolve()
	result = {
		'format': FORMAT,
		'scientific_scope': SCIENTIFIC_SCOPE,
		'task': args.task,
		'backend': BACKEND,
		'arm': args.arm,
		'condition': args.condition,
		'training_seed': args.training_seed,
		'inference_runtime': inference_runtime,
		'evaluation': {
			'split': base.VALIDATION_SPLIT,
			'episodes': EPISODES,
			'decision_steps_per_episode': DECISION_STEPS,
			'env_seed': args.env_seed,
			'background_seed': args.background_seed,
			'planner_seed_base': args.planner_seed_base,
			'eval_mode': True,
			'foreground_erosion_pixels': (
				0 if actual_erosion is None else int(actual_erosion)
			),
			'rgb_shift_rng_alignment': 'object_only_equivalent_cuda_randint_v1',
			'object_only_alignment_draws': alignment_draws,
			'expected_object_only_alignment_draws': EPISODES * DECISION_STEPS,
			'cross_arm_exact_pairing_fields': pairing_fields,
		},
		'provenance': {
			'runtime_config': str(args.runtime_config.resolve()),
			'runtime_config_sha256': runtime_sha_before,
			'runtime_config_sha256_after': runtime_sha_after,
			'checkpoint': str(args.checkpoint.resolve()),
			'checkpoint_sha256': checkpoint_sha_before,
			'checkpoint_sha256_after': checkpoint_sha_after,
			'checkpoint_contract': checkpoint_contract,
			'implementation': {
				name: {
					'path': str(implementation_paths[name]),
					'sha256': implementation_sha_before[name],
					'sha256_after': implementation_sha_after[name],
				} for name in implementation_paths
			},
			'source_flags': {
				'cutie_object_last_valid_memory': False,
				'cutie_object_belief_enabled': True,
				'cutie_object_belief_use_for_control': False,
				'cutie_object_policy_burst_plan': None,
			},
			'evaluation_ablation': {
				'checkpoint_config_mutated': False,
				'fresh_process_required_by_runner': True,
				'episode_outside_control_switch': True,
				'online_prior_selected': args.arm == 'learned_prior',
			},
			'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
			'device_name': torch.cuda.get_device_name(0),
			'validation_manifest_sha256': manifest,
			'combined_manifest_sha256': combined_manifest,
			'cutie_inputs': {
				'checkpoint': str(checkpoint),
				'checkpoint_sha256': base._sha256(checkpoint),
				'support': str(support),
				'support_sha256': base._sha256(support),
				'roles': raw.get('cutie_object_role_names'),
				'support_schema': raw.get('cutie_object_support_schema'),
			},
			'cutie_ready': ready,
		},
		'episodes': records,
		'summary': {
			'reward_mean': float(rewards.mean()),
			'reward_median': float(np.median(rewards)),
			'reward_std': float(rewards.std(ddof=1)),
			'reward_min': float(rewards.min()),
			'reward_max': float(rewards.max()),
			'elapsed_seconds': float(perf_counter() - started),
		},
		'perception_runtime': perception,
		'strict_checks': checks,
	}
	if policy_burst is not None:
		result['policy_burst'] = policy_burst
	return result


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--task', choices=tuple(TASK_ROLES), required=True)
	parser.add_argument('--arm', choices=ARMS, required=True)
	parser.add_argument('--condition', choices=CONDITIONS, required=True)
	parser.add_argument('--policy-burst-plan', type=Path)
	parser.add_argument('--expected-role')
	parser.add_argument('--expected-length', type=int, choices=(20,))
	parser.add_argument('--runtime-config', type=Path, required=True)
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--training-seed', type=int, default=7)
	parser.add_argument('--expected-training-steps', type=int, default=100000)
	parser.add_argument('--expected-training-eval-freq', type=int, default=20000)
	parser.add_argument('--expected-training-eval-episodes', type=int, default=3)
	parser.add_argument('--env-seed', type=int, default=424243)
	parser.add_argument('--background-seed', type=int, default=1618034)
	parser.add_argument('--planner-seed-base', type=int, default=8675400)
	parser.add_argument('--output', type=Path, required=True)
	args = parser.parse_args(argv)
	if args.condition == 'normal':
		if any(value is not None for value in (
			args.policy_burst_plan, args.expected_role, args.expected_length
		)):
			parser.error('normal condition does not accept burst arguments')
	else:
		if args.policy_burst_plan is None or args.expected_length != 20:
			parser.error('burst_20 requires --policy-burst-plan and --expected-length 20')
		if args.expected_role != TARGET_ROLES[args.task]:
			parser.error(f'{args.task} burst role must be {TARGET_ROLES[args.task]}')
	args.backend = BACKEND
	args.episodes = EPISODES
	args.erosion_pixels = EROSION_PIXELS
	return args


def main(argv=None):
	args = parse_args(argv)
	payload = evaluate(args)
	base._write(args.output, payload)
	print('CUTIE_SHADOW_BELIEF_EVALUATION_OK', json.dumps({
		'task': args.task,
		'condition': args.condition,
		'arm': args.arm,
		'reward_mean': payload['summary']['reward_mean'],
		'belief_dynamics_forward_calls': payload[
			'inference_runtime'
		]['belief_dynamics_forward_calls'],
		'prior_role_uses': payload['inference_runtime']['prior_role_uses'],
		'output': str(args.output.resolve()),
	}, allow_nan=False))
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
