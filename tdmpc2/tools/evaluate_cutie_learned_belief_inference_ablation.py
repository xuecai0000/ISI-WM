"""Same-checkpoint inference ablation for the learned Cutie object belief.

Both arms load the exact seed-7 Cartpole learned-belief checkpoint.  The
``learned_prior`` arm uses the production :meth:`TDMPC2.act` path.  The
``measurement_only`` arm keeps the learned-belief module instantiated and
loaded, but bypasses it at inference by encoding the current measurement and
calling the existing evaluation-only latent planner.  This isolates online
prior use from checkpoint training and controller quality.

This is a fixed 20-episode validation diagnostic, not a paper result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


FORMAT = 'cutie_learned_belief_inference_ablation_v1'
SCIENTIFIC_SCOPE = (
	'single-training-seed same-checkpoint inference-time ablation of the learned '
	'Cutie object prior on Cartpole normal validation; this does not compare '
	'training algorithms and is not a paper result'
)
TASK = 'cartpole-swingup'
BACKEND = 'cutie_object_only'
ARMS = ('measurement_only', 'learned_prior')
ROLE_NAMES = ('cart', 'pole')
EPISODES = 20
DECISION_STEPS = 500
EXPECTED_PRIOR_FORWARD_CALLS = EPISODES * (DECISION_STEPS - 1)
NUM_ROLES = 2
STACKED_DIM = 1770
LATENT_DIM = 128
ROLE_DIM = 64
EROSION_PIXELS = 0


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
		raise ValueError('Frozen inference ablation requires training seed 7.')
	if raw.get('cutie_object_role_names') != list(ROLE_NAMES):
		raise ValueError(
			f'Cartpole roles must be {list(ROLE_NAMES)!r}, got '
			f'{raw.get("cutie_object_role_names")!r}.'
		)
	if raw.get('flat_anchor') is not True or raw.get(
		'flat_anchor_mode'
	) != BACKEND:
		raise ValueError('Source is not structural CutieObjectOnly.')
	if raw.get('cutie_object_belief_enabled') is not True:
		raise ValueError('Source must be a learned-belief checkpoint run.')
	if raw.get('cutie_object_last_valid_memory') is not False:
		raise ValueError('Learned-belief source must have last-valid memory disabled.')
	if raw.get('cutie_object_policy_burst_plan') is not None:
		raise ValueError('Normal source must have policy burst plan null.')
	for key, expected in {
		'latent_dim': LATENT_DIM,
		'cutie_object_num_roles': NUM_ROLES,
		'cutie_object_frame_dim': 590,
		'cutie_object_stack_frames': 3,
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
			raise ValueError(
				f'Source {key}={raw.get(key)!r}, expected {expected}.'
			)
	obs_shape = raw.get('obs_shape')
	if obs_shape is not None and obs_shape != {'object': [NUM_ROLES, STACKED_DIM]}:
		raise ValueError(f'Unexpected source obs_shape {obs_shape!r}.')


def _prepare(args, raw: dict):
	cfg = base._prepare(args, raw)
	_require(bool(cfg.get('cutie_object_belief_enabled', False)), (
		'Evaluation config disabled the learned-belief checkpoint contract.'
	))
	_require(not bool(cfg.cutie_object_last_valid_memory), (
		'Evaluation config enabled last-valid memory.'
	))
	_require(cfg.cutie_object_policy_burst_plan is None, (
		'Normal evaluation config installed a policy burst plan.'
	))
	_require(not bool(cfg.compile), 'Inference ablation requires compile=false.')
	return cfg


def _load_checkpoint_contract(path: Path, torch) -> dict:
	payload = torch.load(path, map_location='cpu', weights_only=False)
	if not isinstance(payload, dict):
		raise RuntimeError('Learned-belief checkpoint payload is not a dictionary.')
	contract = payload.get('checkpoint_contract')
	if not isinstance(contract, dict):
		raise RuntimeError('Learned-belief checkpoint contract is missing.')
	expected = {
		'format': 'tdmpc2_checkpoint_contract_v1',
		'flat_anchor_mode': BACKEND,
		'latent_dim': LATENT_DIM,
		'cutie_object_belief_enabled': True,
		'cutie_object_belief_schema': {
			'num_roles': NUM_ROLES,
			'frame_dim': 590,
			'stack_frames': 3,
			'input_dim': STACKED_DIM,
			'role_dim': ROLE_DIM,
		},
	}
	bad = {
		key: (contract.get(key), value)
		for key, value in expected.items()
		if contract.get(key) != value
	}
	if bad:
		raise RuntimeError(f'Learned-belief checkpoint contract mismatch: {bad}.')
	aux_updates = contract.get('cutie_object_belief_aux_updates')
	if (
		isinstance(aux_updates, bool) or not isinstance(aux_updates, int)
		or aux_updates < 1
	):
		raise RuntimeError('Checkpoint auxiliary-update count must be positive.')
	supervision = contract.get('cutie_object_belief_supervision')
	if not isinstance(supervision, dict) or supervision.get(
		'format'
	) != 'cutie_object_belief_supervision_v1':
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
	age20_roles = supervision.get('age20_teacher_roles')
	reacquisition_roles = supervision.get('reacquisition_teacher_roles')
	for name, values in (
		('age20_teacher_roles', age20_roles),
		('reacquisition_teacher_roles', reacquisition_roles),
	):
		if (
			not isinstance(values, list) or len(values) != NUM_ROLES
			or any(
				isinstance(value, bool) or not isinstance(value, int) or value < 1
				for value in values
			)
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
	# The contract contains JSON-native scalars/lists/dicts.  Round-tripping makes
	# the returned provenance independent of the torch payload's object lifetime.
	return json.loads(json.dumps(contract, allow_nan=False))


def _observation_validity(obs, belief_helpers, torch):
	try:
		keys = set(obs.keys())
	except (AttributeError, TypeError) as exc:
		raise RuntimeError('ObjectOnly observation is not keyed.') from exc
	_require(keys == {'object'}, f'ObjectOnly exposed keys {sorted(keys)}.')
	objects = obs['object']
	_require(isinstance(objects, torch.Tensor), 'Object observation is not a tensor.')
	_require(
		tuple(objects.shape) == (NUM_ROLES, STACKED_DIM),
		f'Object observation shape is {tuple(objects.shape)}.',
	)
	_require(
		objects.dtype == torch.float32 and bool(torch.isfinite(objects).all().item()),
		'Object observation must be finite float32.',
	)
	valid = belief_helpers.latest_valid(objects)
	_require(tuple(valid.shape) == (NUM_ROLES,), 'Role validity has the wrong shape.')
	return valid


def _normal_checks(
	args, perception: dict, records: list[dict], alignment_draws: int,
	inference_runtime: dict,
) -> dict[str, bool]:
	intervention = perception.get('policy_observation_intervention', {})
	memory = perception.get('last_valid_memory', {})
	expected_calls = EXPECTED_PRIOR_FORWARD_CALLS if args.arm == 'learned_prior' else 0
	expected_uses = (
		sum(int(record['prior_role_opportunities']) for record in records)
		if args.arm == 'learned_prior' else 0
	)
	checks = {
		'perception_frames_10020': perception.get('frames')
			== EPISODES * (DECISION_STEPS + 1),
		'worker_restarts_zero': perception.get('worker_restarts') == 0,
		'timeouts_zero': perception.get('timeouts') == 0,
		'memory_disabled': memory.get('enabled') is False,
		'memory_substitutions_zero': memory.get('substitutions') == 0,
		'intervention_disabled': intervention.get('enabled') is False,
		'intervention_scheduled_zero': intervention.get('scheduled_role_frames') == 0,
		'intervention_applied_zero': intervention.get('applied_role_frames') == 0,
		'episode_records_20': len(records) == EPISODES,
		'episode_lengths_500': all(
			record.get('length') == DECISION_STEPS for record in records
		),
		'alignment_draws_10000': alignment_draws == EPISODES * DECISION_STEPS,
		'measurement_steps_10000': inference_runtime['measurement_steps']
			== EPISODES * DECISION_STEPS,
		'role_accounting_exact': all(
			inference_runtime['role_valid_decisions'][role]
			+ inference_runtime['role_invalid_decisions'][role]
			== EPISODES * DECISION_STEPS
			for role in range(NUM_ROLES)
		),
		'natural_invalid_observed': sum(
			inference_runtime['role_invalid_decisions']
		) > 0,
		'cartpole_pole_invalid_observed': inference_runtime[
			'role_invalid_decisions'
		][1] > 0,
		'belief_forward_calls_exact': inference_runtime[
			'belief_dynamics_forward_calls'
		] == expected_calls,
		'prior_role_uses_exact': inference_runtime['prior_role_uses']
			== expected_uses,
		'action_traces_valid': all(
			_hash_valid(record.get('action_trace_sha256')) for record in records
		),
		'latent_traces_valid': all(
			_hash_valid(record.get('controller_latent_trace_sha256'))
			for record in records
		),
		'pairing_hashes_valid': all(
			all(_hash_valid(record.get(key)) for key in (
				'initial_rgb_sha256', 'initial_object_sha256',
				'final_object_sha256', 'planner_rng_start_sha256',
				'planner_rng_end_sha256',
			))
			for record in records
		),
	}
	if args.arm == 'learned_prior':
		checks.update({
			'belief_enabled': inference_runtime['enabled'] is True,
			'belief_resets_19': inference_runtime['belief_resets']
				== EPISODES - 1,
			'measurement_only_latent_steps_zero': inference_runtime[
				'measurement_only_latent_steps'
			] == 0,
			'online_belief_retained_at_end': inference_runtime[
				'online_belief_state_is_none'
			] is False,
		})
	else:
		checks.update({
			'checkpoint_belief_still_enabled': inference_runtime['enabled'] is True,
			'belief_resets_zero': inference_runtime['belief_resets'] == 0,
			'measurement_only_latent_steps_10000': inference_runtime[
				'measurement_only_latent_steps'
			] == EPISODES * DECISION_STEPS,
			'online_belief_never_created': inference_runtime[
				'online_belief_state_is_none'
			] is True,
			'online_belief_action_never_created': inference_runtime[
				'online_belief_action_is_none'
			] is True,
		})
	failed = sorted(name for name, passed in checks.items() if not passed)
	if failed:
		raise RuntimeError(
			'Learned-belief inference ablation checks failed: ' + ', '.join(failed)
		)
	return checks


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
		'agent': (PROJECT_DIR / 'tdmpc2.py').resolve(),
		'world_model': (PROJECT_DIR / 'common' / 'world_model.py').resolve(),
		'belief_helpers': (
			PROJECT_DIR / 'common' / 'cutie_object_belief.py'
		).resolve(),
		'layers': (PROJECT_DIR / 'common' / 'layers.py').resolve(),
		'object_wrapper': (
			PROJECT_DIR / 'envs' / 'wrappers' / 'cutie_object.py'
		).resolve(),
	}
	implementation_sha_before = {
		name: base._sha256(path) for name, path in implementation_paths.items()
	}
	raw = base._json(args.runtime_config)
	_validate_source(args, raw)
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
	measurement_only_latent_steps = 0
	role_valid_decisions = [0, 0]
	role_invalid_decisions = [0, 0]
	role_invalid_initial_decisions = [0, 0]
	any_role_invalid_decisions = 0
	both_roles_invalid_decisions = 0
	belief_forward_calls = 0
	prior_input_trace = hashlib.sha256()
	prior_output_trace = hashlib.sha256()
	perception = ready = manifest = combined_manifest = None
	actual_erosion = None
	started = perf_counter()

	def belief_forward_hook(_module, inputs, output):
		nonlocal belief_forward_calls
		_require(len(inputs) == 2, 'Belief transition hook received wrong inputs.')
		belief, action = inputs
		_require(
			tuple(belief.shape) == (1, LATENT_DIM),
			f'Belief transition input shape is {tuple(belief.shape)}.',
		)
		_require(
			tuple(action.shape) == (1, int(agent.cfg.action_dim)),
			f'Belief transition action shape is {tuple(action.shape)}.',
		)
		_require(
			tuple(output.shape) == (1, LATENT_DIM),
			f'Belief transition output shape is {tuple(output.shape)}.',
		)
		_require(
			belief.dtype == torch.float32 and action.dtype == torch.float32
			and output.dtype == torch.float32,
			'Belief transition hook observed a non-float32 tensor.',
		)
		_require(
			bool(torch.isfinite(belief).all().item())
			and bool(torch.isfinite(action).all().item())
			and bool(torch.isfinite(output).all().item()),
			'Belief transition hook observed a non-finite tensor.',
		)
		_trace_tensor(prior_input_trace, belief)
		_trace_tensor(prior_input_trace, action)
		_trace_tensor(prior_output_trace, output)
		belief_forward_calls += 1

	try:
		set_seed(args.env_seed)
		env = make_env(cfg)
		_require(
			base._env_value(env, 'active_split') == base.VALIDATION_SPLIT,
			'Validation background split was not constructed.',
		)
		actual_erosion = base._env_value(env, 'erosion_pixels')
		_require(
			actual_erosion in (None, EROSION_PIXELS),
			f'Normal evaluation erosion is {actual_erosion!r}.',
		)
		_require(
			base._env_value(env, 'policy_burst_plan_sha256') is None,
			'Normal wrapper unexpectedly loaded a policy burst plan.',
		)
		agent = TDMPC2(cfg)
		agent.load(args.checkpoint)
		agent.eval()
		_require(agent._learned_object_belief, 'Loaded agent did not enable belief.')
		_require(
			int(agent._belief_aux_updates)
			== checkpoint_contract['cutie_object_belief_aux_updates'],
			'Loaded auxiliary-update count differs from checkpoint contract.',
		)
		hook_handle = agent.model._belief_dynamics.register_forward_hook(
			belief_forward_hook
		)

		for episode_index in range(EPISODES):
			obs = env.reset()
			initial_valid = _observation_validity(
				obs, cutie_object_belief, torch
			)
			initial_rgb = base._rgb_hash(obs, env, BACKEND)
			initial_object = base._object_hash(obs, BACKEND)
			source = base._env_value(env, 'active_source')
			frame_index = base._env_value(env, 'frame_index')
			_require(
				source is not None and frame_index is not None,
				'Initial background provenance is unavailable.',
			)

			planner_seed = args.planner_seed_base + episode_index
			set_seed(planner_seed)
			agent._prev_mean.zero_()
			reset_count_before = int(agent._online_object_belief_resets)
			had_online_state = (
				agent._online_object_belief is not None
				or agent._online_object_belief_action is not None
			)
			agent.reset_object_belief()
			reset_delta = int(agent._online_object_belief_resets) - reset_count_before
			expected_reset_delta = int(
				args.arm == 'learned_prior' and episode_index > 0
			)
			_require(
				reset_delta == expected_reset_delta,
				f'Episode {episode_index} reset delta is {reset_delta}.',
			)
			_require(
				had_online_state == bool(expected_reset_delta),
				f'Episode {episode_index} pre-reset state is inconsistent.',
			)

			rng_start = base._rng_hash(torch)
			action_trace = hashlib.sha256()
			reward_trace = hashlib.sha256()
			latent_trace = hashlib.sha256()
			validity_trace = hashlib.sha256()
			reward_sum = 0.0
			info = {}
			episode_valid = [0, 0]
			episode_invalid = [0, 0]
			episode_any_invalid = 0
			episode_both_invalid = 0
			episode_measurement_latent_steps = 0
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
				invalid_count = int((~valid).sum().item())
				episode_any_invalid += int(invalid_count > 0)
				episode_both_invalid += int(invalid_count == NUM_ROLES)
				any_role_invalid_decisions += int(invalid_count > 0)
				both_roles_invalid_decisions += int(invalid_count == NUM_ROLES)

				if args.arm == 'measurement_only':
					agent_obs = obs.to(
						agent.device, non_blocking=True
					).unsqueeze(0)
					latent = agent.model.encode(agent_obs, None)
					_require(
						tuple(latent.shape) == (1, LATENT_DIM)
						and latent.device == agent.device
						and latent.dtype == torch.float32
						and bool(torch.isfinite(latent).all().item()),
						'Measurement encoder returned an invalid latent.',
					)
					action = agent.act_from_latent(
						latent, t0=step_index == 0, eval_mode=True
					)
					measurement_only_latent_steps += 1
					episode_measurement_latent_steps += 1
					_require(
						agent._online_object_belief is None
						and agent._online_object_belief_action is None,
						'Measurement-only path created online belief state.',
					)
				else:
					calls_before_step = belief_forward_calls
					action = agent.act(
						obs, t0=step_index == 0, eval_mode=True
					)
					expected_step_calls = int(step_index > 0)
					_require(
						belief_forward_calls - calls_before_step
						== expected_step_calls,
						f'Belief forward count mismatch at episode '
						f'{episode_index}, step {step_index}.',
					)
					latent = agent._online_object_belief
					_require(
						isinstance(latent, torch.Tensor)
						and tuple(latent.shape) == (1, LATENT_DIM)
						and latent.device == agent.device
						and latent.dtype == torch.float32
						and bool(torch.isfinite(latent).all().item()),
						'Production learned belief produced an invalid latent.',
					)
					_require(
						tuple(agent._online_object_belief_action.shape)
						== (1, int(agent.cfg.action_dim)),
						'Production learned belief stored a wrong-shape action.',
					)

				_require(
					tuple(action.shape) == (int(agent.cfg.action_dim),)
					and action.dtype == torch.float32
					and bool(torch.isfinite(action).all().item()),
					'Controller returned an invalid action.',
				)
				_trace_tensor(latent_trace, latent)
				_trace_tensor(action_trace, action)
				obs, reward, done, info = env.step(action)
				reward_sum += float(reward)
				reward_trace.update(struct.pack('<d', float(reward)))
				if done:
					length = step_index + 1
					break
			else:
				raise RuntimeError('Environment did not terminate at 500 steps.')
			_require(
				length == DECISION_STEPS,
				f'Unexpected episode length {length}.',
			)

			end_source = base._env_value(env, 'active_source')
			end_frame_index = base._env_value(env, 'frame_index')
			_require(
				end_source is not None and end_frame_index is not None,
				'Final background provenance is unavailable.',
			)
			final_object = base._object_hash(obs, BACKEND)
			prior_calls = belief_forward_calls - prior_calls_before
			prior_uses = int(agent._online_object_belief_prior_uses) - prior_uses_before
			prior_opportunities = sum(episode_invalid) - int(
				(~initial_valid).sum().item()
			)
			_require(
				prior_calls == (
					DECISION_STEPS - 1 if args.arm == 'learned_prior' else 0
				),
				f'Episode {episode_index} prior call count is {prior_calls}.',
			)
			_require(
				prior_uses == (
					prior_opportunities if args.arm == 'learned_prior' else 0
				),
				f'Episode {episode_index} prior role-use count is {prior_uses}.',
			)
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
				'controller_latent_trace_sha256': latent_trace.hexdigest(),
				'role_validity_trace_sha256': validity_trace.hexdigest(),
				'role_valid_decisions': episode_valid,
				'role_invalid_decisions': episode_invalid,
				'initial_role_valid': [
					bool(value) for value in initial_valid.tolist()
				],
				'any_role_invalid_decisions': episode_any_invalid,
				'both_roles_invalid_decisions': episode_both_invalid,
				'prior_role_opportunities': prior_opportunities,
				'belief_dynamics_forward_calls': prior_calls,
				'prior_role_uses': prior_uses,
				'measurement_only_latent_steps': (
					episode_measurement_latent_steps
				),
				'belief_reset_delta': reset_delta,
				'reward': reward_sum,
				'success': float(info.get('success', 0.0)),
				'length': length,
			}
			records.append(record)
			print('CUTIE_LEARNED_BELIEF_ABLATION_EPISODE', json.dumps({
				'task': TASK,
				'arm': args.arm,
				'episode_index': episode_index,
				'reward': reward_sum,
				'role_invalid_decisions': episode_invalid,
				'belief_dynamics_forward_calls': prior_calls,
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

	inference_runtime = {
		'mode': args.arm,
		'arm': args.arm,
		'enabled': bool(agent._learned_object_belief),
		'loaded_aux_updates': int(agent._belief_aux_updates),
		'belief_dynamics_forward_calls': int(belief_forward_calls),
		'expected_belief_dynamics_forward_calls': (
			EXPECTED_PRIOR_FORWARD_CALLS if args.arm == 'learned_prior' else 0
		),
		'prior_role_uses': int(agent._online_object_belief_prior_uses),
		'belief_resets': int(agent._online_object_belief_resets),
		'role_names': list(ROLE_NAMES),
		'role_valid_decisions': role_valid_decisions,
		'role_invalid_decisions': role_invalid_decisions,
		'role_invalid_initial_decisions': role_invalid_initial_decisions,
		'any_role_invalid_decisions': any_role_invalid_decisions,
		'both_roles_invalid_decisions': both_roles_invalid_decisions,
		'measurement_steps': EPISODES * DECISION_STEPS,
		'measurement_only_latent_steps': measurement_only_latent_steps,
		'online_belief_state_is_none': agent._online_object_belief is None,
		'online_belief_action_is_none': agent._online_object_belief_action is None,
		'belief_prior_input_trace_sha256': prior_input_trace.hexdigest(),
		'belief_prior_output_trace_sha256': prior_output_trace.hexdigest(),
		'latent_shape': [1, LATENT_DIM],
		'role_latent_slices': {
			ROLE_NAMES[index]: [index * ROLE_DIM, (index + 1) * ROLE_DIM]
			for index in range(NUM_ROLES)
		},
		'latest_valid_index': 1768,
		'executed_previous_action_is_exact_clamped_env_action': True,
		'planner_entrypoint': (
			'TDMPC2.act' if args.arm == 'learned_prior'
			else 'TDMPC2.act_from_latent'
		),
	}
	strict_checks = _normal_checks(
		args, perception, records, alignment_draws, inference_runtime
	)
	rewards = np.asarray(
		[record['reward'] for record in records], dtype=np.float64
	)
	_require(
		rewards.size == EPISODES and np.isfinite(rewards).all(),
		'Incomplete or non-finite rewards.',
	)
	checkpoint = Path(raw['cutie_object_checkpoint']).resolve()
	support = Path(raw['cutie_object_support_path']).resolve()
	return {
		'format': FORMAT,
		'scientific_scope': SCIENTIFIC_SCOPE,
		'task': TASK,
		'backend': BACKEND,
		'arm': args.arm,
		'condition': 'normal',
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
			'rgb_shift_rng_alignment': (
				'object_only_equivalent_cuda_randint_v1'
			),
			'object_only_alignment_draws': alignment_draws,
			'expected_object_only_alignment_draws': (
				EPISODES * DECISION_STEPS
			),
			'cross_arm_exact_pairing_fields': [
				'episode_index', 'planner_seed', 'planner_rng_start_sha256',
				'planner_rng_end_sha256', 'initial_rgb_sha256',
				'initial_object_sha256', 'background_source',
				'background_start_frame_index', 'background_end_source',
				'background_end_frame_index', 'length',
			],
			'cross_arm_scientific_difference_fields': [
				'action_trace_sha256', 'reward_trace_sha256',
				'controller_latent_trace_sha256', 'role_validity_trace_sha256',
				'role_invalid_decisions', 'reward',
			],
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
				}
				for name in implementation_paths
			},
			'source_flags': {
				'cutie_object_last_valid_memory': False,
				'cutie_object_belief_enabled': True,
				'cutie_object_policy_burst_plan': None,
			},
			'evaluation_ablation': {
				'checkpoint_config_mutated': False,
				'belief_module_instantiated_and_loaded': True,
				'online_prior_selected': args.arm == 'learned_prior',
				'measurement_only_uses_evaluation_only_latent_planner': (
					args.arm == 'measurement_only'
				),
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
		'strict_checks': strict_checks,
	}


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--arm', choices=ARMS, required=True)
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
	parser.set_defaults(
		task=TASK,
		backend=BACKEND,
		episodes=EPISODES,
		erosion_pixels=EROSION_PIXELS,
	)
	return parser.parse_args(argv)


def main(argv=None):
	args = parse_args(argv)
	payload = evaluate(args)
	base._write(args.output, payload)
	print('CUTIE_LEARNED_BELIEF_INFERENCE_ABLATION_OK', json.dumps({
		'task': TASK,
		'arm': args.arm,
		'reward_mean': payload['summary']['reward_mean'],
		'belief_dynamics_forward_calls': payload[
			'inference_runtime'
		]['belief_dynamics_forward_calls'],
		'prior_role_uses': payload['inference_runtime']['prior_role_uses'],
		'role_invalid_decisions': payload[
			'inference_runtime'
		]['role_invalid_decisions'],
		'output': str(args.output.resolve()),
	}, allow_nan=False))
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
