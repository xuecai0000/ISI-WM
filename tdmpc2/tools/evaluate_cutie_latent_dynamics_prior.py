"""Evaluate a frozen CutieObjectOnly dynamics prior under policy-input bursts.

Both arms use the same hard-zero checkpoint. ``measurement_only`` is the
identity control; ``dynamics_prior`` substitutes only the missing whole-arm
64-D latent role with the frozen action-conditioned transition prediction.
This is a single-seed development diagnostic, not a trained belief model.
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
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
for local_path in (str(REPO_DIR), str(PROJECT_DIR)):
	while local_path in sys.path:
		sys.path.remove(local_path)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))

from tdmpc2.tools import evaluate_cutie_multitask_checkpoint as base
from tdmpc2.tools import evaluate_cutie_multitask_policy_burst as burst


FORMAT = 'cutie_latent_dynamics_prior_evaluation_v1'
SCIENTIFIC_SCOPE = (
	'single-training-seed oracle-support development diagnostic using a frozen '
	'pre-existing TD-MPC2 transition as an inference-time role prior; no new '
	'parameters are trained and this is not a learned belief or paper claim'
)
TASK = 'reacher-visual-small'
BACKEND = 'cutie_object_only'
ARMS = ('measurement_only', 'dynamics_prior')
CONDITIONS = ('normal', 'burst_5', 'burst_20', 'burst_50')
TARGET_ROLE = 'whole_arm'
TARGET_ROLE_INDEX = 0
EPISODES = 20
DECISION_STEPS = 500
MIN_RAW_VALID_OVERWRITE_RATE = 0.95


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
	digest.update(array.tobytes())


def _condition_length(condition: str) -> int:
	if condition == 'normal':
		return 0
	return int(condition.split('_', 1)[1])


def _validate_source(args, raw: dict) -> None:
	for required in (
		'cutie_object_last_valid_memory', 'cutie_object_policy_burst_plan',
	):
		if required not in raw:
			raise ValueError(f'Source runtime is missing {required}.')
	if raw.get('cutie_object_last_valid_memory') is not False:
		raise ValueError('Both arms require the same hard-zero memory=false source.')
	if raw.get('cutie_object_policy_burst_plan') is not None:
		raise ValueError('Source training must have policy burst plan null.')
	if raw.get('cutie_object_role_names') != ['whole_arm', 'goal']:
		raise ValueError('Frozen Reacher role order must be whole_arm, goal.')
	if raw.get('flat_anchor') is not True or raw.get('flat_anchor_mode') != BACKEND:
		raise ValueError('Source is not structural CutieObjectOnly.')
	if int(raw.get('latent_dim', -1)) != 128:
		raise ValueError('Source latent_dim must be 128.')
	for key, expected in {
		'cutie_object_num_roles': 2,
		'cutie_object_frame_dim': 590,
		'cutie_object_stack_frames': 3,
		'cutie_object_input_dim': 1770,
		'cutie_object_role_dim': 64,
	}.items():
		if int(raw.get(key, -1)) != expected:
			raise ValueError(f'Source {key}={raw.get(key)!r}, expected {expected}.')
	if args.training_seed != 7:
		raise ValueError('Frozen diagnostic uses training seed 7 only.')


def _validate_condition(args):
	length = _condition_length(args.condition)
	if args.expected_length != length:
		raise ValueError(
			f'Condition {args.condition} requires expected_length={length}.'
		)
	if length == 0:
		if args.policy_burst_plan is not None:
			raise ValueError('Normal evaluation must not receive a burst plan.')
		return None, None
	if args.policy_burst_plan is None:
		raise ValueError(f'{args.condition} requires --policy-burst-plan.')
	plan_args = SimpleNamespace(
		policy_burst_plan=args.policy_burst_plan,
		task=TASK,
		expected_role=TARGET_ROLE,
		expected_length=length,
	)
	return burst._validate_plan(plan_args)


def _prepare(args, raw: dict):
	evaluation_raw = dict(raw)
	evaluation_raw['cutie_object_policy_burst_plan'] = (
		str(args.policy_burst_plan.resolve())
		if args.policy_burst_plan is not None else None
	)
	cfg = base._prepare(args, evaluation_raw)
	if bool(cfg.cutie_object_last_valid_memory):
		raise RuntimeError('Evaluator accidentally enabled last-valid memory.')
	actual_plan = cfg.cutie_object_policy_burst_plan
	if args.policy_burst_plan is None:
		if actual_plan is not None:
			raise RuntimeError('Normal evaluator unexpectedly installed a plan.')
	elif Path(str(actual_plan)).resolve() != args.policy_burst_plan.resolve():
		raise RuntimeError('Evaluation config installed a different burst plan.')
	return cfg


def _rng_state(torch):
	return (
		torch.get_rng_state().clone(),
		torch.cuda.get_rng_state(0).clone(),
	)


def _set_rng_state(torch, state) -> None:
	torch.set_rng_state(state[0])
	torch.cuda.set_rng_state(state[1], 0)


def _planner_parity_contract(agent, obs, torch) -> dict:
	"""Prove the copied latent MPPI body matches the established obs path."""
	original_rng = _rng_state(torch)
	original_mean = agent._prev_mean.clone()
	checks = {}
	try:
		for t0 in (True, False):
			torch.manual_seed(13579)
			torch.cuda.manual_seed_all(13579)
			fixed_rng = _rng_state(torch)
			fixed_mean = torch.linspace(
				-0.5, 0.5, agent._prev_mean.numel(),
				device=agent.device, dtype=agent._prev_mean.dtype,
			).reshape_as(agent._prev_mean)

			_set_rng_state(torch, fixed_rng)
			agent._prev_mean.copy_(fixed_mean)
			obs_action = agent.act(obs, t0=t0, eval_mode=True)
			obs_end_rng = _rng_state(torch)
			obs_end_mean = agent._prev_mean.clone()

			_set_rng_state(torch, fixed_rng)
			agent._prev_mean.copy_(fixed_mean)
			with torch.no_grad():
				agent_obs = obs.to(agent.device, non_blocking=True).unsqueeze(0)
				measurement = agent.model.encode(agent_obs, None)
			latent_action = agent.act_from_latent(
				measurement, t0=t0, eval_mode=True
			)
			latent_end_rng = _rng_state(torch)
			latent_end_mean = agent._prev_mean.clone()

			prefix = 't0' if t0 else 'non_t0'
			checks[f'{prefix}_action_bitwise'] = torch.equal(
				obs_action, latent_action
			)
			checks[f'{prefix}_prev_mean_bitwise'] = torch.equal(
				obs_end_mean, latent_end_mean
			)
			checks[f'{prefix}_cpu_rng_bitwise'] = torch.equal(
				obs_end_rng[0], latent_end_rng[0]
			)
			checks[f'{prefix}_cuda_rng_bitwise'] = torch.equal(
				obs_end_rng[1], latent_end_rng[1]
			)
	finally:
		_set_rng_state(torch, original_rng)
		agent._prev_mean.copy_(original_mean)
	failed = [name for name, passed in checks.items() if not passed]
	if failed:
		raise RuntimeError('Latent planner parity failed: ' + ', '.join(failed))
	return checks


def _aggregate_belief(records: list[dict], arm: str) -> dict:
	metrics = [record['latent_belief_metrics'] for record in records]
	keys = (
		'measurement_steps', 'target_valid_steps', 'target_invalid_steps',
		'target_measurement_selected_steps',
		'non_target_measurement_selected_steps',
		'prior_computed_steps', 'prior_used_steps',
		'synthetic_prior_used_steps', 'natural_prior_used_steps',
		'invalid_without_prior', 'valid_prior_comparisons',
		'valid_persistence_comparisons',
		'valid_prior_better_than_persistence_count',
	)
	payload = {key: sum(int(value[key]) for value in metrics) for key in keys}
	payload['max_prior_age'] = max(int(value['max_prior_age']) for value in metrics)
	payload['valid_prior_role_mse_sum'] = sum(
		float(value['valid_prior_role_mse_sum']) for value in metrics
	)
	payload['valid_prior_role_cosine_sum'] = sum(
		float(value['valid_prior_role_cosine_sum']) for value in metrics
	)
	payload['valid_persistence_role_mse_sum'] = sum(
		float(value['valid_persistence_role_mse_sum']) for value in metrics
	)
	count = payload['valid_prior_comparisons']
	payload['valid_prior_role_mse_mean'] = (
		payload['valid_prior_role_mse_sum'] / count if count else None
	)
	payload['valid_prior_role_cosine_mean'] = (
		payload['valid_prior_role_cosine_sum'] / count if count else None
	)
	persistence_count = payload['valid_persistence_comparisons']
	payload['valid_persistence_role_mse_mean'] = (
		payload['valid_persistence_role_mse_sum'] / persistence_count
		if persistence_count else None
	)
	reacquisitions = [
		{'episode_index': record['episode_index'], **item}
		for record in records
		for item in record['latent_belief_metrics']['reacquisitions']
	]
	payload['reacquisitions'] = reacquisitions
	payload['reacquisition_count'] = len(reacquisitions)
	for field in ('prior_role_mse', 'persistence_role_mse'):
		payload[f'{field}_mean_at_reacquisition'] = (
			sum(float(item[field]) for item in reacquisitions) / len(reacquisitions)
			if reacquisitions else None
		)
	payload['prior_better_than_persistence_count'] = sum(
		int(item['prior_better_than_persistence']) for item in reacquisitions
	)
	payload['mode'] = arm
	return payload


def _normal_checks(perception: dict, records: list[dict], alignment_draws: int):
	intervention = perception.get('policy_observation_intervention', {})
	memory = perception.get('last_valid_memory', {})
	checks = {
		'perception_frames_10020': perception.get('frames') == 10020,
		'worker_restarts_zero': perception.get('worker_restarts') == 0,
		'timeouts_zero': perception.get('timeouts') == 0,
		'memory_disabled': memory.get('enabled') is False
			and memory.get('substitutions') == 0,
		'intervention_disabled': intervention.get('enabled') is False,
		'no_intervention_frames': intervention.get('scheduled_role_frames') == 0
			and intervention.get('applied_role_frames') == 0,
		'alignment_draws_10000': alignment_draws == 10000,
		'episode_records_20': len(records) == 20,
		'raw_diagnostic_not_exposed': all(
			record.get('initial_raw_object_frame_sha256') is None
			and record.get('final_raw_object_frame_sha256') is None
			for record in records
		),
	}
	failed = [key for key, value in checks.items() if not value]
	if failed:
		raise RuntimeError('Normal strict checks failed: ' + ', '.join(failed))
	return checks


def evaluate(args):
	import numpy as np
	import torch
	from common.seed import set_seed
	from envs import make_env
	from tdmpc2.common.cutie_latent_belief import CutieLatentDynamicsBelief
	from tdmpc2.tdmpc2 import TDMPC2

	base._validate_args(args)
	runtime_sha_before = base._sha256(args.runtime_config)
	checkpoint_sha_before = base._sha256(args.checkpoint)
	implementation_paths = {
		'agent': (PROJECT_DIR / 'tdmpc2.py').resolve(),
		'latent_belief': (
			PROJECT_DIR / 'common' / 'cutie_latent_belief.py'
		).resolve(),
		'object_wrapper': (
			PROJECT_DIR / 'envs' / 'wrappers' / 'cutie_object.py'
		).resolve(),
	}
	implementation_sha_before = {
		name: base._sha256(path) for name, path in implementation_paths.items()
	}
	plan, plan_sha_before = _validate_condition(args)
	raw = base._json(args.runtime_config)
	_validate_source(args, raw)
	cfg = _prepare(args, raw)
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required.')
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.set_float32_matmul_precision('high')

	env = None
	records = []
	alignment_draws = 0
	parity_checks = None
	perception = ready = manifest = combined_manifest = None
	wrapper_plan_sha = None
	started = perf_counter()
	try:
		set_seed(args.env_seed)
		env = make_env(cfg)
		_require(
			base._env_value(env, 'active_split') == base.VALIDATION_SPLIT,
			'Validation background split was not constructed.',
		)
		_require(
			base._env_value(env, 'erosion_pixels') in (None, 0),
			'Latent belief diagnostic requires erosion zero.',
		)
		wrapper_plan_sha = base._env_value(env, 'policy_burst_plan_sha256')
		_require(
			wrapper_plan_sha == plan_sha_before,
			'Wrapper plan SHA does not match evaluator plan.',
		)
		agent = TDMPC2(cfg)
		agent.load(args.checkpoint)
		agent.eval()

		for episode_index in range(EPISODES):
			obs = env.reset()
			_require(set(obs.keys()) == {'object'}, 'Controller saw a non-object key.')
			_require(tuple(obs['object'].shape) == (2, 1770), 'Wrong object shape.')
			if parity_checks is None:
				parity_checks = _planner_parity_contract(agent, obs, torch)

			initial_rgb = base._rgb_hash(obs, env, BACKEND)
			initial_object = base._object_hash(obs, BACKEND)
			initial_raw = base._env_value(env, 'latest_raw_object_frame_sha256')
			if plan is None:
				_require(
					initial_raw is None,
					'Normal wrapper unexpectedly exposed a diagnostic raw frame.',
				)
			else:
				_require(
					_hash_valid(initial_raw),
					'Initial raw object hash is unavailable.',
				)
			source = base._env_value(env, 'active_source')
			frame_index = base._env_value(env, 'frame_index')
			_require(source is not None and frame_index is not None, 'Missing background start.')

			planner_seed = args.planner_seed_base + episode_index
			set_seed(planner_seed)
			agent._prev_mean.zero_()
			controller = CutieLatentDynamicsBelief(
				agent, args.arm, target_role_index=TARGET_ROLE_INDEX
			)
			rng_start = base._rng_hash(torch)
			action_trace = hashlib.sha256()
			reward_trace = hashlib.sha256()
			belief_trace = hashlib.sha256()
			reward_sum = 0.0
			info = {}
			event = plan['events'][episode_index] if plan is not None else None

			for step_index in range(DECISION_STEPS):
				torch.compiler.cudagraph_mark_step_begin()
				base._align_object_only_rgb_shift_rng(torch, BACKEND)
				alignment_draws += 1
				synthetic_active = bool(
					event is not None
					and event['start_decision_step'] <= step_index
					< event['start_decision_step'] + event['length']
				)
				belief, _ = controller.observe(
					obs, synthetic_target_invalid=synthetic_active
				)
				_trace_tensor(belief_trace, belief)
				action = agent.act_from_latent(
					belief, t0=step_index == 0, eval_mode=True
				)
				controller.record_executed_action(action)
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
			end_frame = base._env_value(env, 'frame_index')
			_require(
				end_source is not None and end_frame is not None,
				'Missing background end provenance.',
			)
			final_object = base._object_hash(obs, BACKEND)
			final_raw = base._env_value(env, 'latest_raw_object_frame_sha256')
			if plan is None:
				_require(
					final_raw is None,
					'Normal wrapper unexpectedly exposed a diagnostic raw frame.',
				)
			else:
				_require(_hash_valid(final_raw), 'Final raw object hash is unavailable.')
			record = {
				'episode_index': episode_index,
				'planner_seed': planner_seed,
				'planner_rng_start_sha256': rng_start,
				'planner_rng_end_sha256': base._rng_hash(torch),
				'initial_rgb_sha256': initial_rgb,
				'initial_policy_object_sha256': initial_object,
				'initial_raw_object_frame_sha256': initial_raw,
				'final_policy_object_sha256': final_object,
				'final_raw_object_frame_sha256': final_raw,
				'background_source': Path(source).name,
				'background_start_frame_index': int(frame_index),
				'background_end_source': Path(end_source).name,
				'background_end_frame_index': int(end_frame),
				'policy_burst_event': dict(event) if event is not None else None,
				'policy_burst_plan_sha256': wrapper_plan_sha,
				'action_trace_sha256': action_trace.hexdigest(),
				'reward_trace_sha256': reward_trace.hexdigest(),
				'belief_trace_sha256': belief_trace.hexdigest(),
				'latent_belief_metrics': controller.metrics(),
				'reward': reward_sum,
				'success': float(info.get('success', 0.0)),
				'length': length,
			}
			records.append(record)
			print('CUTIE_LATENT_PRIOR_EVAL_EPISODE', json.dumps({
				'task': TASK, 'arm': args.arm, 'condition': args.condition,
				'episode_index': episode_index, 'reward': reward_sum,
				'prior_used_steps': record['latent_belief_metrics']['prior_used_steps'],
			}, allow_nan=False), flush=True)

		perception = base._metrics(env)
		ready = base._env_value(env, 'cutie_ready')
		manifest = base._env_value(env, 'manifest_sha256')
		combined_manifest = base._env_value(env, 'combined_manifest_sha256')
	finally:
		if env is not None and callable(getattr(env, 'close', None)):
			env.close()

	runtime_sha_after = base._sha256(args.runtime_config)
	checkpoint_sha_after = base._sha256(args.checkpoint)
	implementation_sha_after = {
		name: base._sha256(path) for name, path in implementation_paths.items()
	}
	_require(runtime_sha_after == runtime_sha_before, 'runtime_config changed during eval.')
	_require(checkpoint_sha_after == checkpoint_sha_before, 'checkpoint changed during eval.')
	_require(
		implementation_sha_after == implementation_sha_before,
		'Latent belief implementation changed during evaluation.',
	)
	plan_sha_after = (
		base._sha256(args.policy_burst_plan)
		if args.policy_burst_plan is not None else None
	)
	_require(isinstance(perception, dict), 'Perception metrics are unavailable.')
	_require(isinstance(ready, dict), 'Cutie ready provenance is unavailable.')
	_require(isinstance(parity_checks, dict), 'Planner parity contract did not run.')

	if plan is None:
		strict_checks = _normal_checks(perception, records, alignment_draws)
	else:
		intervention_args = SimpleNamespace(
			arm='hard_zero', task=TASK, expected_role=TARGET_ROLE,
			expected_length=args.expected_length,
			policy_burst_plan=args.policy_burst_plan,
		)
		strict_checks = burst._validate_intervention(
			intervention_args, perception, plan, plan_sha_before,
			plan_sha_after, wrapper_plan_sha, records, alignment_draws,
		)
		raw_overwritten = perception['policy_observation_intervention'][
			'raw_valid_overwritten'
		]
		coverage = raw_overwritten / (EPISODES * args.expected_length)
		_require(
			coverage >= MIN_RAW_VALID_OVERWRITE_RATE,
			f'Controlled burst coverage {coverage:.4f} is below '
			f'{MIN_RAW_VALID_OVERWRITE_RATE:.2f}.',
		)

	belief = _aggregate_belief(records, args.arm)
	expected_synthetic = EPISODES * args.expected_length
	controlled_reacquisitions = [
		record for record in belief['reacquisitions']
		if record.get('synthetic_steps') == args.expected_length
		and record.get('natural_steps') == 0
	] if args.expected_length else []
	belief['controlled_reacquisition_expected_count'] = (
		EPISODES if args.expected_length else 0
	)
	belief['controlled_reacquisition_count'] = len(controlled_reacquisitions)
	belief['controlled_reacquisition_eligible'] = (
		len(controlled_reacquisitions) == EPISODES
		if args.expected_length else None
	)
	belief['controlled_reacquisition_prior_mse_mean'] = (
		sum(float(item['prior_role_mse']) for item in controlled_reacquisitions)
		/ len(controlled_reacquisitions)
		if controlled_reacquisitions else None
	)
	belief['controlled_reacquisition_persistence_mse_mean'] = (
		sum(
			float(item['persistence_role_mse'])
			for item in controlled_reacquisitions
		) / len(controlled_reacquisitions)
		if controlled_reacquisitions else None
	)
	if args.arm == 'measurement_only':
		_require(
			belief['prior_computed_steps'] == 0 and belief['prior_used_steps'] == 0,
			'Measurement control unexpectedly used a prior.',
		)
	else:
		_require(
			belief['prior_computed_steps'] == EPISODES * (DECISION_STEPS - 1),
			'Dynamics prior call count mismatch.',
		)
		_require(
			belief['synthetic_prior_used_steps'] == expected_synthetic,
			'Synthetic prior-use count mismatch.',
		)
		_require(belief['invalid_without_prior'] == 0, 'Invalid target occurred at t0.')

	rewards = np.asarray([record['reward'] for record in records], dtype=np.float64)
	_require(rewards.size == EPISODES and np.isfinite(rewards).all(), 'Invalid rewards.')
	checkpoint = Path(raw['cutie_object_checkpoint']).resolve()
	support = Path(raw['cutie_object_support_path']).resolve()
	intervention = perception['policy_observation_intervention']
	raw_valid_rate = (
		intervention.get('raw_valid_overwritten', 0) / expected_synthetic
		if expected_synthetic else None
	)
	return {
		'format': FORMAT,
		'scientific_scope': SCIENTIFIC_SCOPE,
		'task': TASK,
		'backend': BACKEND,
		'arm': args.arm,
		'condition': args.condition,
		'training_seed': args.training_seed,
		'latent_belief': {
			'algorithm': 'frozen_action_conditioned_role_prior_v1',
			'mode': args.arm,
			'target_role': TARGET_ROLE,
			'target_role_index': TARGET_ROLE_INDEX,
			'target_latent_slice': [0, 64],
			'non_target_latent_slice': [64, 128],
			'latest_valid_index': 1768,
			'valid_measurement_rule': 'current_measurement_replaces_prior_immediately',
			'invalid_rule': 'target_role_prior_only_non_target_current_measurement',
			'uses_executed_previous_action': True,
			'trainable_parameters_added': 0,
			'hidden_raw_features_available_to_policy': False,
			'metrics': belief,
		},
		'policy_burst': {
			'enabled': plan is not None,
			'length': args.expected_length,
			'plan_sha256_before': plan_sha_before,
			'plan_sha256_after': plan_sha_after,
			'wrapper_plan_sha256': wrapper_plan_sha,
			'minimum_raw_valid_overwrite_rate': MIN_RAW_VALID_OVERWRITE_RATE,
			'raw_valid_overwrite_rate': raw_valid_rate,
			'controlled_burst_attribution_eligible': (
				raw_valid_rate >= MIN_RAW_VALID_OVERWRITE_RATE
				if raw_valid_rate is not None else None
			),
		},
		'evaluation': {
			'split': base.VALIDATION_SPLIT,
			'episodes': EPISODES,
			'env_seed': args.env_seed,
			'background_seed': args.background_seed,
			'planner_seed_base': args.planner_seed_base,
			'eval_mode': True,
			'object_only_alignment_draws': alignment_draws,
			'planner_from_latent_parity': parity_checks,
		},
		'provenance': {
			'runtime_config': str(args.runtime_config.resolve()),
			'runtime_config_sha256': runtime_sha_before,
			'runtime_config_sha256_after': runtime_sha_after,
			'checkpoint': str(args.checkpoint.resolve()),
			'checkpoint_sha256': checkpoint_sha_before,
			'checkpoint_sha256_after': checkpoint_sha_after,
			'evaluator': str(Path(__file__).resolve()),
			'evaluator_sha256': base._sha256(Path(__file__).resolve()),
			'base_evaluator': str(Path(base.__file__).resolve()),
			'base_evaluator_sha256': base._sha256(Path(base.__file__).resolve()),
			'burst_evaluator': str(Path(burst.__file__).resolve()),
			'burst_evaluator_sha256': base._sha256(Path(burst.__file__).resolve()),
			'agent_implementation': str(implementation_paths['agent']),
			'agent_implementation_sha256': implementation_sha_before['agent'],
			'agent_implementation_sha256_after': implementation_sha_after['agent'],
			'latent_belief_implementation': str(
				implementation_paths['latent_belief']
			),
			'latent_belief_implementation_sha256': (
				implementation_sha_before['latent_belief']
			),
			'latent_belief_implementation_sha256_after': (
				implementation_sha_after['latent_belief']
			),
			'object_wrapper_implementation': str(
				implementation_paths['object_wrapper']
			),
			'object_wrapper_implementation_sha256': (
				implementation_sha_before['object_wrapper']
			),
			'object_wrapper_implementation_sha256_after': (
				implementation_sha_after['object_wrapper']
			),
			'policy_burst_plan': (
				str(args.policy_burst_plan.resolve())
				if args.policy_burst_plan is not None else None
			),
			'policy_burst_plan_relative_to_output': (
				os.path.relpath(
					args.policy_burst_plan.resolve(),
					start=args.output.resolve().parent,
				)
				if args.policy_burst_plan is not None else None
			),
			'policy_burst_plan_sha256': plan_sha_before,
			'source_flags': {
				'cutie_object_last_valid_memory': False,
				'cutie_object_policy_burst_plan': None,
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
	parser.add_argument('--condition', choices=CONDITIONS, required=True)
	parser.add_argument('--policy-burst-plan', type=Path)
	parser.add_argument('--expected-length', type=int, choices=(0, 5, 20, 50), required=True)
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
		task=TASK, backend=BACKEND, episodes=EPISODES, erosion_pixels=0,
	)
	return parser.parse_args(argv)


def main(argv=None):
	args = parse_args(argv)
	payload = evaluate(args)
	base._write(args.output, payload)
	print('CUTIE_LATENT_DYNAMICS_PRIOR_EVAL_OK', json.dumps({
		'arm': args.arm,
		'condition': args.condition,
		'reward_mean': payload['summary']['reward_mean'],
		'prior_used_steps': payload['latent_belief']['metrics']['prior_used_steps'],
		'output': str(args.output.resolve()),
	}, allow_nan=False))
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
