"""Strict fixed-burst evaluator for structural CutieObjectOnly checkpoints.

This is deliberately a sibling of ``evaluate_cutie_multitask_checkpoint`` so
the established evaluator and its provenance hash remain immutable.  It
reuses that evaluator's environment, hashing, RNG-alignment, and atomic-output
	helpers, while adding one canonical policy-input intervention plan.  It can
	evaluate hard-zero, fixed last-valid, or a separately trained learned-belief
	checkpoint; the synthetic intervention remains a development diagnostic and
	not a final algorithm comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


FORMAT = 'cutie_multitask_policy_burst_evaluation_v1'
SCIENTIFIC_SCOPE = (
	'single-training-seed oracle-support development diagnostic of causal '
	'policy-input burst handling; arms may include a trained action-conditioned '
	'belief, but these results are not a final algorithm comparison'
)
BACKEND = 'cutie_object_only'
EPISODES = 20
EROSION_PIXELS = 0
DECISION_STEPS = 500
FRAME_DIM = 590
STACK_FRAMES = 3
STACKED_DIM = FRAME_DIM * STACK_FRAMES
PLAN_FORMAT = 'cutie_policy_burst_plan_v1'
INVALID_ENCODING = 'empty_lost_v1'
ARMS = ('hard_zero', 'last_valid', 'learned_belief')
LENGTHS = (5, 20, 50)
STARTS = (75, 150, 225, 300, 375) * 4
MIN_RAW_VALID_OVERWRITE_RATE = 0.80
TASK_ROLES = {
	'reacher-visual-small': ('whole_arm', 'goal'),
	'cup-catch': ('cup', 'ball'),
	'cartpole-swingup': ('cart', 'pole'),
	'finger-spin': ('finger', 'spinner'),
	'acrobot-swingup': ('upper_arm', 'lower_arm'),
}
TARGET_ROLES = {
	'reacher-visual-small': 'whole_arm',
	'cup-catch': 'ball',
	'cartpole-swingup': 'pole',
	'finger-spin': 'spinner',
	'acrobot-swingup': 'lower_arm',
}


def _canonical_sha256(payload: dict) -> tuple[bytes, str]:
	encoded = (
		json.dumps(
			payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
			allow_nan=False,
		) + '\n'
	).encode('utf-8')
	return encoded, hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
	if not condition:
		raise RuntimeError(message)


def _validate_plan(args) -> tuple[dict, str]:
	path = args.policy_burst_plan.resolve()
	if not path.is_file() or path.suffix.lower() != '.json':
		raise FileNotFoundError(f'Policy burst plan must be an existing JSON: {path}')
	raw_bytes = path.read_bytes()
	try:
		payload = json.loads(raw_bytes.decode('utf-8'))
	except (UnicodeDecodeError, json.JSONDecodeError) as exc:
		raise ValueError(f'Invalid UTF-8 policy burst plan: {path}') from exc
	if not isinstance(payload, dict):
		raise ValueError('Policy burst plan must be a JSON object.')
	canonical_bytes, canonical_sha = _canonical_sha256(payload)
	if raw_bytes != canonical_bytes:
		raise ValueError(
			'Policy burst plan must be canonical sorted compact JSON plus one LF.'
		)
	expected_keys = {
		'format', 'task', 'roles', 'episodes', 'decision_steps', 'frame_dim',
		'stack_frames', 'invalid_encoding', 'events',
	}
	if set(payload) != expected_keys:
		raise ValueError(
			f'Policy burst plan keys {sorted(payload)} != {sorted(expected_keys)}.'
		)
	expected_roles = list(TASK_ROLES[args.task])
	for key, expected in {
		'format': PLAN_FORMAT,
		'task': args.task,
		'roles': expected_roles,
		'episodes': EPISODES,
		'decision_steps': DECISION_STEPS,
		'frame_dim': FRAME_DIM,
		'stack_frames': STACK_FRAMES,
		'invalid_encoding': INVALID_ENCODING,
	}.items():
		if payload.get(key) != expected:
			raise ValueError(
				f'Policy burst plan {key}={payload.get(key)!r}, expected {expected!r}.'
			)
	if args.expected_role != TARGET_ROLES[args.task]:
		raise ValueError(
			f'{args.task} diagnostic target must be {TARGET_ROLES[args.task]!r}, '
			f'got {args.expected_role!r}.'
		)
	events = payload['events']
	if not isinstance(events, list) or len(events) != EPISODES:
		raise ValueError(f'Policy burst plan requires exactly {EPISODES} events.')
	for episode_index, (event, expected_start) in enumerate(zip(events, STARTS)):
		expected = {
			'episode_index': episode_index,
			'role': args.expected_role,
			'start_decision_step': expected_start,
			'length': args.expected_length,
		}
		if event != expected:
			raise ValueError(
				f'Policy burst event {episode_index}={event!r}, expected {expected!r}.'
			)
	return payload, canonical_sha


def _validate_source(args, raw: dict) -> bool:
	expected_memory = args.arm == 'last_valid'
	expected_belief = args.arm == 'learned_belief'
	if 'cutie_object_last_valid_memory' not in raw:
		raise ValueError('Source runtime is missing cutie_object_last_valid_memory.')
	if raw['cutie_object_last_valid_memory'] is not expected_memory:
		raise ValueError(
			'Source memory arm mismatch: '
			f'{raw["cutie_object_last_valid_memory"]!r} != {expected_memory!r}.'
		)
	if 'cutie_object_policy_burst_plan' not in raw:
		raise ValueError('Source runtime is missing cutie_object_policy_burst_plan.')
	if raw['cutie_object_policy_burst_plan'] is not None:
		raise ValueError(
			'Source training must have cutie_object_policy_burst_plan=null; '
			'evaluation is the only intervention site.'
		)
	if raw.get('cutie_object_role_names') != list(TASK_ROLES[args.task]):
		raise ValueError(
			f'Source roles {raw.get("cutie_object_role_names")!r} != '
			f'{list(TASK_ROLES[args.task])!r}.'
		)
	if raw.get('flat_anchor') is not True:
		raise ValueError('Source must have flat_anchor=true.')
	if raw.get('flat_anchor_mode') != BACKEND:
		raise ValueError(f'Source must have flat_anchor_mode={BACKEND}.')
	actual_belief = bool(raw.get('cutie_object_belief_enabled', False))
	if actual_belief is not expected_belief:
		raise ValueError(
			'Source learned-belief arm mismatch: '
			f'{actual_belief!r} != {expected_belief!r}.'
		)
	return expected_memory


def _prepare(args, raw: dict):
	# The source snapshot must stay plan-free.  Only this copied evaluation
	# configuration receives the frozen intervention path.
	evaluation_raw = dict(raw)
	evaluation_raw['cutie_object_policy_burst_plan'] = str(
		args.policy_burst_plan.resolve()
	)
	cfg = base._prepare(args, evaluation_raw)
	if bool(cfg.cutie_object_last_valid_memory) != (args.arm == 'last_valid'):
		raise RuntimeError('Evaluation dataclass changed the frozen memory arm.')
	if bool(cfg.get('cutie_object_belief_enabled', False)) != (
		args.arm == 'learned_belief'
	):
		raise RuntimeError('Evaluation dataclass changed the learned-belief arm.')
	if Path(str(cfg.cutie_object_policy_burst_plan)).resolve() != (
		args.policy_burst_plan.resolve()
	):
		raise RuntimeError('Evaluation dataclass did not bind the exact burst plan.')
	return cfg


def _hash_is_valid(value) -> bool:
	return (
		isinstance(value, str) and len(value) == 64
		and all(character in '0123456789abcdef' for character in value)
	)


def _validate_intervention(
	args, perception: dict, plan: dict, plan_sha_before: str,
	plan_sha_after: str, wrapper_plan_sha: str, records: list[dict],
	alignment_draws: int,
) -> dict[str, bool]:
	intervention = perception.get('policy_observation_intervention')
	if not isinstance(intervention, dict):
		raise RuntimeError('Wrapper policy_observation_intervention is unavailable.')
	memory = perception.get('last_valid_memory')
	if not isinstance(memory, dict):
		raise RuntimeError('Wrapper last_valid_memory metrics are unavailable.')
	expected_frames = EPISODES * (DECISION_STEPS + 1)
	expected_decisions = EPISODES * DECISION_STEPS
	expected_burst_frames = EPISODES * args.expected_length
	raw_valid_overwrite_rate = (
		intervention.get('raw_valid_overwritten', -1) / expected_burst_frames
	)
	expected_per_role = {
		role: expected_burst_frames if role == args.expected_role else 0
		for role in TASK_ROLES[args.task]
	}
	expected_memory_enabled = args.arm == 'last_valid'
	checks = {
		'perception_frames_10020': perception.get('frames') == expected_frames,
		'worker_restarts_zero': perception.get('worker_restarts') == 0,
		'timeouts_zero': perception.get('timeouts') == 0,
		'intervention_enabled': intervention.get('enabled') is True,
		'intervention_format': intervention.get('format') == PLAN_FORMAT,
		'intervention_location': intervention.get('location') == (
			'raw_tracker_frame_after_metrics_before_last_valid_memory_and_stack'
		),
		'invalid_encoding': intervention.get('invalid_encoding') == INVALID_ENCODING,
		'decision_unit': intervention.get('decision_unit') == (
			'agent_decision_observation_index_0_to_499'
		),
		'plan_path': Path(str(intervention.get('plan_path', ''))).resolve()
			== args.policy_burst_plan.resolve(),
		'plan_task': intervention.get('plan_task') == args.task,
		'plan_roles': intervention.get('plan_roles') == list(TASK_ROLES[args.task]),
		'plan_sha_before_after': plan_sha_before == plan_sha_after,
		'plan_sha_wrapper': intervention.get('plan_sha256') == plan_sha_before
			and wrapper_plan_sha == plan_sha_before,
		'scheduled_events_20': intervention.get('scheduled_events') == EPISODES,
		'applied_events_20': intervention.get('applied_events') == EPISODES,
		'scheduled_role_frames': intervention.get('scheduled_role_frames')
			== expected_burst_frames,
		'applied_role_frames': intervention.get('applied_role_frames')
			== expected_burst_frames,
		'raw_frame_accounting': (
			intervention.get('raw_valid_overwritten', -1)
			+ intervention.get('raw_invalid_overlap', -1)
		) == expected_burst_frames,
		'raw_valid_overwrite_rate_in_unit_interval': (
			0.0 <= raw_valid_overwrite_rate <= 1.0
		),
		'exact_invalid_checks': intervention.get('exact_invalid_checks')
			== expected_burst_frames,
		'non_target_preserved_checks': intervention.get(
			'non_target_preserved_checks'
		) == expected_burst_frames,
		'raw_source_unchanged_checks': intervention.get(
			'raw_source_unchanged_checks'
		) == expected_burst_frames,
		'policy_stack_transition_checks_10020': intervention.get(
			'policy_stack_transition_checks'
		) == expected_frames,
		'per_role_applied_frames': intervention.get('per_role_applied_frames')
			== expected_per_role,
		'raw_trace_hash': _hash_is_valid(
			intervention.get('raw_latest_trace_sha256')
		),
		'policy_latest_trace_hash': _hash_is_valid(
			intervention.get('policy_latest_trace_sha256')
		),
		'policy_stack_trace_hash': _hash_is_valid(
			intervention.get('policy_stack_trace_sha256')
		),
		'raw_tracker_excludes_intervention': intervention.get(
			'raw_tracker_accounting_excludes_synthetic_intervention'
		) is True,
		'live_environment_observation_unchanged': intervention.get(
			'live_environment_observation_mutated'
		) is False,
		'memory_arm': memory.get('enabled') is expected_memory_enabled,
		'rgb_shift_rng_alignment_draws_10000': alignment_draws
			== expected_decisions,
		'episode_records_20': len(records) == EPISODES,
	}
	if args.arm in {'hard_zero', 'learned_belief'}:
		checks.update({
			'hard_zero_content_checks': intervention.get(
				'hard_zero_content_checks'
			) == expected_burst_frames,
			'last_valid_content_checks_zero': intervention.get(
				'last_valid_content_checks'
			) == 0,
			'intervention_memory_substitutions_zero': intervention.get(
				'memory_substitutions'
			) == 0,
			'intervention_without_history_zero': intervention.get(
				'without_memory_history'
			) == 0,
			'memory_substitutions_zero': memory.get('substitutions') == 0,
			'memory_without_history_zero': memory.get('invalid_without_history') == 0,
		})
	else:
		checks.update({
			'hard_zero_content_checks_zero': intervention.get(
				'hard_zero_content_checks'
			) == 0,
			'last_valid_content_checks': intervention.get(
				'last_valid_content_checks'
			) == expected_burst_frames,
			'intervention_memory_substitutions': intervention.get(
				'memory_substitutions'
			) == expected_burst_frames,
			'intervention_without_history_zero': intervention.get(
				'without_memory_history'
			) == 0,
			'memory_substitutions_include_bursts': memory.get('substitutions', -1)
				>= expected_burst_frames,
		})
	per_episode = intervention.get('per_episode')
	checks['intervention_episode_records_20'] = (
		isinstance(per_episode, list) and len(per_episode) == EPISODES
	)
	if checks['intervention_episode_records_20']:
		for episode_index, (actual, event) in enumerate(
			zip(per_episode, plan['events'])
		):
			expected_common = {
				**event,
				'scheduled_role_frames': args.expected_length,
				'applied_role_frames': args.expected_length,
			}
			checks[f'episode_{episode_index:02d}_event'] = all(
				actual.get(key) == value for key, value in expected_common.items()
			)
			checks[f'episode_{episode_index:02d}_raw_accounting'] = (
				actual.get('raw_valid_overwritten', -1)
				+ actual.get('raw_invalid_overlap', -1)
			) == args.expected_length
			checks[f'episode_{episode_index:02d}_memory_content'] = (
				actual.get('memory_substitutions') == (
					args.expected_length if expected_memory_enabled else 0
				)
				and actual.get('without_memory_history') == 0
			)
	for episode_index, record in enumerate(records):
		checks[f'episode_{episode_index:02d}_hashes'] = all(
			_hash_is_valid(record.get(key)) for key in (
				'initial_rgb_sha256', 'initial_policy_object_sha256',
				'initial_raw_object_frame_sha256',
				'final_policy_object_sha256', 'final_raw_object_frame_sha256',
				'planner_rng_start_sha256', 'planner_rng_end_sha256',
			)
		)
		checks[f'episode_{episode_index:02d}_event_record'] = (
			record.get('policy_burst_event') == plan['events'][episode_index]
		)
	failed = sorted(name for name, passed in checks.items() if not passed)
	if failed:
		raise RuntimeError(
			'Policy-burst strict checks failed: ' + ', '.join(failed)
		)
	return checks


def evaluate(args):
	import numpy as np
	import torch
	from common.seed import set_seed
	from envs import make_env
	from tdmpc2.tdmpc2 import TDMPC2

	base._validate_args(args)
	runtime_config_sha_before = base._sha256(args.runtime_config)
	checkpoint_sha_before = base._sha256(args.checkpoint)
	plan, plan_sha_before = _validate_plan(args)
	raw = base._json(args.runtime_config)
	expected_memory = _validate_source(args, raw)
	cfg = _prepare(args, raw)
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required.')
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.set_float32_matmul_precision('high')

	env = None
	records = []
	alignment_draws = 0
	started = perf_counter()
	perception = ready = manifest = combined_manifest = None
	wrapper_plan_sha = None
	try:
		set_seed(args.env_seed)
		env = make_env(cfg)
		_require(
			base._env_value(env, 'active_split') == base.VALIDATION_SPLIT,
			'Validation background split was not constructed.',
		)
		actual_erosion = base._env_value(env, 'erosion_pixels')
		_require(
			actual_erosion in (None, 0),
			f'Policy-burst evaluation must have erosion zero, got {actual_erosion!r}.',
		)
		wrapper_plan_sha = base._env_value(env, 'policy_burst_plan_sha256')
		_require(
			wrapper_plan_sha == plan_sha_before,
			'Wrapper loaded a different policy burst plan.',
		)
		agent = TDMPC2(cfg)
		agent.load(args.checkpoint)
		agent.eval()
		for episode_index in range(EPISODES):
			obs = env.reset()
			_require(set(obs.keys()) == {'object'}, 'ObjectOnly exposed non-object input.')
			_require(
				tuple(obs['object'].shape) == (2, STACKED_DIM),
				f'Policy object shape is {tuple(obs["object"].shape)}.',
			)
			_require(
				str(obs['object'].dtype) == 'torch.float32',
				f'Policy object dtype is {obs["object"].dtype}.',
			)
			initial_rgb = base._rgb_hash(obs, env, BACKEND)
			initial_policy_object = base._object_hash(obs, BACKEND)
			initial_raw_object = base._env_value(
				env, 'latest_raw_object_frame_sha256'
			)
			_require(
				_hash_is_valid(initial_raw_object),
				'Wrapper did not expose a valid initial raw object-frame hash.',
			)
			source = base._env_value(env, 'active_source')
			frame_index = base._env_value(env, 'frame_index')
			_require(
				source is not None and frame_index is not None,
				'Initial background provenance is unavailable.',
			)
			planner_seed = args.planner_seed_base + episode_index
			set_seed(planner_seed)
			agent._prev_mean.zero_()
			rng_start = base._rng_hash(torch)
			reward_sum = 0.0
			info = {}
			for step_index in range(DECISION_STEPS):
				torch.compiler.cudagraph_mark_step_begin()
				base._align_object_only_rgb_shift_rng(torch, BACKEND)
				alignment_draws += 1
				action = agent.act(obs, t0=step_index == 0, eval_mode=True)
				obs, reward, done, info = env.step(action)
				reward_sum += float(reward)
				if done:
					length = step_index + 1
					break
			else:
				raise RuntimeError('Environment did not terminate at 500 steps.')
			_require(length == DECISION_STEPS, f'Unexpected episode length {length}.')
			final_policy_object = base._object_hash(obs, BACKEND)
			final_raw_object = base._env_value(
				env, 'latest_raw_object_frame_sha256'
			)
			end_source = base._env_value(env, 'active_source')
			end_frame_index = base._env_value(env, 'frame_index')
			record = {
				'episode_index': episode_index,
				'planner_seed': planner_seed,
				'planner_rng_start_sha256': rng_start,
				'planner_rng_end_sha256': base._rng_hash(torch),
				'initial_rgb_sha256': initial_rgb,
				'initial_policy_object_sha256': initial_policy_object,
				'initial_raw_object_frame_sha256': initial_raw_object,
				'final_policy_object_sha256': final_policy_object,
				'final_raw_object_frame_sha256': final_raw_object,
				'background_source': Path(source).name,
				'background_start_frame_index': int(frame_index),
				'background_end_source': Path(end_source).name,
				'background_end_frame_index': int(end_frame_index),
				'policy_burst_event': dict(plan['events'][episode_index]),
				'policy_burst_plan_sha256': wrapper_plan_sha,
				'reward': reward_sum,
				'success': float(info.get('success', 0.0)),
				'length': length,
			}
			records.append(record)
			print('CUTIE_POLICY_BURST_EVAL_EPISODE', json.dumps({
				'task': args.task, 'arm': args.arm, **record,
			}, allow_nan=False), flush=True)
		perception = base._metrics(env)
		ready = base._env_value(env, 'cutie_ready')
		manifest = base._env_value(env, 'manifest_sha256')
		combined_manifest = base._env_value(env, 'combined_manifest_sha256')
	finally:
		if env is not None and callable(getattr(env, 'close', None)):
			env.close()
	belief_runtime = {
		'enabled': bool(getattr(agent, '_learned_object_belief', False)),
		'loaded_aux_updates': int(getattr(agent, '_belief_aux_updates', 0)),
		'prior_role_uses': int(getattr(
			agent, '_online_object_belief_prior_uses', 0
		)),
		'episode_state_resets': int(getattr(
			agent, '_online_object_belief_resets', 0
		)),
	}
	if args.arm == 'learned_belief':
		_require(belief_runtime['enabled'], 'Learned belief did not activate.')
		_require(
			belief_runtime['loaded_aux_updates'] > 0,
			'Learned checkpoint reports no auxiliary updates.',
		)
		_require(
			belief_runtime['prior_role_uses'] >= EPISODES * args.expected_length,
			'Learned belief did not consume every controlled missing role-frame.',
		)
		_require(
			belief_runtime['episode_state_resets'] == EPISODES - 1,
			'Learned belief episode reset count is not exact.',
		)
	else:
		_require(
			not belief_runtime['enabled']
			and belief_runtime['loaded_aux_updates'] == 0
			and belief_runtime['prior_role_uses'] == 0,
			'Non-belief arm unexpectedly activated agent belief state.',
		)

	plan_sha_after = base._sha256(args.policy_burst_plan)
	runtime_config_sha_after = base._sha256(args.runtime_config)
	checkpoint_sha_after = base._sha256(args.checkpoint)
	_require(
		runtime_config_sha_after == runtime_config_sha_before,
		'Source runtime_config changed during evaluation.',
	)
	_require(
		checkpoint_sha_after == checkpoint_sha_before,
		'Source checkpoint changed during evaluation.',
	)
	_require(isinstance(perception, dict), 'Cutie runtime metrics are unavailable.')
	_require(isinstance(ready, dict), 'Cutie ready provenance is unavailable.')
	strict_checks = _validate_intervention(
		args, perception, plan, plan_sha_before, plan_sha_after,
		wrapper_plan_sha, records, alignment_draws,
	)
	rewards = np.asarray([record['reward'] for record in records], dtype=np.float64)
	_require(
		rewards.size == EPISODES and np.isfinite(rewards).all(),
		'Incomplete or non-finite policy-burst rewards.',
	)
	checkpoint = Path(raw['cutie_object_checkpoint']).resolve()
	support = Path(raw['cutie_object_support_path']).resolve()
	return {
		'format': FORMAT,
		'scientific_scope': SCIENTIFIC_SCOPE,
		'task': args.task,
		'arm': args.arm,
		'backend': BACKEND,
		'training_seed': args.training_seed,
		'policy_burst': {
			'format': PLAN_FORMAT,
			'role': args.expected_role,
			'length': args.expected_length,
			'starts': list(STARTS),
			'episodes': EPISODES,
			'plan_sha256_before': plan_sha_before,
			'plan_sha256_after': plan_sha_after,
			'wrapper_plan_sha256': wrapper_plan_sha,
			'minimum_raw_valid_overwrite_rate': MIN_RAW_VALID_OVERWRITE_RATE,
			'raw_valid_overwrite_rate': (
				perception['policy_observation_intervention'][
					'raw_valid_overwritten'
				] / (EPISODES * args.expected_length)
			),
			'controlled_burst_attribution_eligible': (
				perception['policy_observation_intervention'][
					'raw_valid_overwritten'
				] / (EPISODES * args.expected_length)
			) >= MIN_RAW_VALID_OVERWRITE_RATE,
		},
		'learned_belief_runtime': belief_runtime,
		'evaluation': {
			'split': base.VALIDATION_SPLIT,
			'episodes': EPISODES,
			'env_seed': args.env_seed,
			'background_seed': args.background_seed,
			'planner_seed_base': args.planner_seed_base,
			'eval_mode': True,
			'foreground_erosion_pixels': EROSION_PIXELS,
			'rgb_shift_rng_alignment': 'object_only_equivalent_cuda_randint_v1',
			'object_only_alignment_draws': alignment_draws,
			'expected_object_only_alignment_draws': EPISODES * DECISION_STEPS,
		},
		'provenance': {
			'runtime_config': str(args.runtime_config.resolve()),
			'runtime_config_sha256': runtime_config_sha_before,
			'runtime_config_sha256_after': runtime_config_sha_after,
			'checkpoint': str(args.checkpoint.resolve()),
			'checkpoint_sha256': checkpoint_sha_before,
			'checkpoint_sha256_after': checkpoint_sha_after,
			'evaluator': str(Path(__file__).resolve()),
			'evaluator_sha256': base._sha256(Path(__file__).resolve()),
			'base_evaluator': str(Path(base.__file__).resolve()),
			'base_evaluator_sha256': base._sha256(Path(base.__file__).resolve()),
			'policy_burst_plan': str(args.policy_burst_plan.resolve()),
			'policy_burst_plan_relative_to_output': os.path.relpath(
				args.policy_burst_plan.resolve(), start=args.output.resolve().parent
			),
			'policy_burst_plan_sha256': plan_sha_before,
			'source_flags': {
				'cutie_object_last_valid_memory': expected_memory,
				'cutie_object_belief_enabled': args.arm == 'learned_belief',
				'cutie_object_policy_burst_plan': None,
			},
			'evaluation_flags': {
				'cutie_object_last_valid_memory': expected_memory,
				'cutie_object_belief_enabled': args.arm == 'learned_belief',
				'cutie_object_policy_burst_plan': str(
					args.policy_burst_plan.resolve()
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
	parser.add_argument('--task', choices=tuple(TASK_ROLES), required=True)
	parser.add_argument('--arm', choices=ARMS, required=True)
	parser.add_argument('--policy-burst-plan', type=Path, required=True)
	parser.add_argument('--expected-role', required=True)
	parser.add_argument('--expected-length', type=int, choices=LENGTHS, required=True)
	parser.add_argument('--runtime-config', type=Path, required=True)
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--training-seed', type=int, default=7)
	parser.add_argument('--expected-training-steps', type=int, default=100000)
	parser.add_argument('--expected-training-eval-freq', type=int, default=20000)
	parser.add_argument('--expected-training-eval-episodes', type=int, default=3)
	parser.add_argument('--env-seed', type=int, default=424244)
	parser.add_argument('--background-seed', type=int, default=1618035)
	parser.add_argument('--planner-seed-base', type=int, default=8675500)
	parser.add_argument('--output', type=Path, required=True)
	parser.set_defaults(
		backend=BACKEND, episodes=EPISODES, erosion_pixels=EROSION_PIXELS,
	)
	return parser.parse_args(argv)


def main(argv=None):
	args = parse_args(argv)
	payload = evaluate(args)
	base._write(args.output, payload)
	print('CUTIE_POLICY_BURST_EVAL_OK', json.dumps({
		'task': args.task,
		'arm': args.arm,
		'role': args.expected_role,
		'length': args.expected_length,
		'reward_mean': payload['summary']['reward_mean'],
		'output': str(args.output.resolve()),
	}, allow_nan=False))
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
