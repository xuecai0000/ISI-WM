"""Frozen-checkpoint causal ablation for legacy Cutie object observations.

This evaluator answers two narrow questions without training a new model and
without reading simulator state or ground-truth masks at evaluation time:

1. Does the controller use the 512-D Cutie query/appearance block, beyond the
   78-D mask geometry/status block in each 590-D frame descriptor?
2. How sensitive is the controller to an exact 20- or 50-decision loss of one
   task-critical role?

The first deliberately supported tasks avoid experiments that already tested
whole-object zeroing or last-valid memory on Reacher/Cartpole:

* ``finger-spin``: established K=2 legacy MLP, target role ``spinner``;
* ``walker-stand``: explicit K=3 legacy MLP, target role ``torso``.

Every arm uses the same frozen checkpoint, 20 episode/environment seeds, video
background starts, and planner random streams.  Interventions are applied to a
copy of the object tensor immediately before ``agent.act``.  The environment's
live observation and Cutie tracker output are never mutated.  Policies can
naturally choose different actions, so this is paired on exogenous schedules
and initial states, not a claim that endogenous state trajectories stay equal.
"""

from __future__ import annotations

import argparse
import copy
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


FORMAT = 'object_input_causal_ablation_v1'
VALIDATION_FORMAT = 'object_input_causal_ablation_validation_v1'
ARM_FORMAT = 'object_input_causal_ablation_arm_v1'
BACKEND = 'cutie_object_only'
MODES = ('full', 'query_appearance_zero', 'target_role_drop_20', 'target_role_drop_50')
CONDITIONS = ('clean', 'hard')
FRAME_DIM = 590
QUERY_DIM = 512
GEOMETRY_STATUS_DIM = FRAME_DIM - QUERY_DIM
STACK_FRAMES = 3
STACKED_DIM = FRAME_DIM * STACK_FRAMES
DECISION_STEPS = 500
EPISODES = 20
BURST_STARTS = (75, 150, 225, 300, 375) * 4
BOOTSTRAP_SAMPLES = 10_000
SUPPORTED = {
	'finger-spin': {
		'roles': ('finger', 'spinner'),
		'target_role': 'spinner',
		'role_count': 2,
		'regression_encoder': (None, 'legacy_mlp_v1'),
		'latent_dim': 128,
		'novelty': (
			'query-vs-geometry causal attribution on a stable K2 task; not the '
			'previous whole-object-zero or last-valid-memory diagnostic'
		),
	},
	'walker-stand': {
		'roles': ('torso', 'right_leg', 'left_leg'),
		'target_role': 'torso',
		'role_count': 3,
		'regression_encoder': ('legacy_mlp_k3_v1',),
		'latent_dim': 192,
		'novelty': (
			'first fixed-burst causal attribution for the explicit K3 legacy MLP'
		),
	},
}


def _require(condition: bool, message: str) -> None:
	if not condition:
		raise RuntimeError(message)


def _hash_tensor(value) -> str:
	array = value.detach().cpu().contiguous().numpy()
	return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def _trace_update(digest, *, episode: int, step: int, value) -> None:
	digest.update(f'{episode}:{step}:'.encode('ascii'))
	digest.update(value.detach().cpu().contiguous().numpy().tobytes(order='C'))


def _clone(value):
	"""Clone a Torch tensor; NumPy support keeps contract tests dependency-light."""
	return value.clone() if callable(getattr(value, 'clone', None)) else value.copy()


def _data_pointer(value) -> int:
	if callable(getattr(value, 'data_ptr', None)):
		return int(value.data_ptr())
	return int(value.__array_interface__['data'][0])


def _burst_length(mode: str) -> int:
	if mode == 'target_role_drop_20':
		return 20
	if mode == 'target_role_drop_50':
		return 50
	return 0


def _new_diagnostics(task: str, mode: str) -> dict:
	spec = SUPPORTED[task]
	return {
		'mode': mode,
		'location': 'copied_policy_object_immediately_before_agent_act',
		'source': 'live_causal_cutie_object_observation',
		'live_environment_observation_mutated': False,
		'query_slice_per_frame': [0, QUERY_DIM],
		'preserved_geometry_status_slice_per_frame': [QUERY_DIM, FRAME_DIM],
		'roles': list(spec['roles']),
		'target_role': spec['target_role'],
		'target_role_index': spec['roles'].index(spec['target_role']),
		'decision_steps_seen': 0,
		'copied_input_checks': 0,
		'source_unchanged_checks': 0,
		'full_copy_equal_checks': 0,
		'query_zero_checks': 0,
		'query_geometry_status_preserved_checks': 0,
		'query_effective_nonzero_frames': 0,
		'target_role_zero_checks': 0,
		'non_target_roles_preserved_checks': 0,
		'outside_burst_equal_checks': 0,
		'scheduled_role_drop_frames': EPISODES * _burst_length(mode),
		'applied_role_drop_frames': 0,
		'effective_nonzero_target_overwrites': 0,
	}


def _validate_object(obs, *, task: str, torch):
	try:
		keys = set(obs.keys())
	except Exception as exc:
		raise ValueError('Object ablation requires a mapping observation.') from exc
	if keys != {'object'}:
		raise ValueError(f'Object-only observation keys {keys} != {{"object"}}.')
	value = obs['object']
	expected = (SUPPORTED[task]['role_count'], STACKED_DIM)
	if tuple(value.shape) != expected or value.dtype != torch.float32:
		raise ValueError(
			f'Expected float32 object tensor {expected}, got '
			f'{tuple(value.shape)} {value.dtype}.'
		)
	if not bool(torch.isfinite(value).all()):
		raise ValueError('Object observation contains non-finite values.')
	return value


def _apply_intervention(
	obs, *, task: str, mode: str, episode_index: int, step_index: int,
	torch, diagnostics: dict,
):
	"""Return an ablated copy and prove that the source tensor stayed immutable."""
	if task not in SUPPORTED:
		raise ValueError(f'Unsupported causal-ablation task {task!r}.')
	if mode not in MODES:
		raise ValueError(f'Unsupported causal-ablation mode {mode!r}.')
	if not 0 <= episode_index < EPISODES or not 0 <= step_index < DECISION_STEPS:
		raise ValueError('Episode/step index lies outside the frozen evaluation grid.')
	raw = _validate_object(obs, task=task, torch=torch)
	source_snapshot = _clone(raw)
	policy = _clone(raw)
	diagnostics['decision_steps_seen'] += 1
	diagnostics['copied_input_checks'] += int(
		_data_pointer(policy) != _data_pointer(raw)
	)

	if mode == 'full':
		_require(torch.equal(policy, raw), 'Full arm changed the policy input.')
		diagnostics['full_copy_equal_checks'] += 1
	elif mode == 'query_appearance_zero':
		frames = policy.reshape(SUPPORTED[task]['role_count'], STACK_FRAMES, FRAME_DIM)
		raw_frames = raw.reshape(SUPPORTED[task]['role_count'], STACK_FRAMES, FRAME_DIM)
		preserved = _clone(raw_frames[..., QUERY_DIM:])
		diagnostics['query_effective_nonzero_frames'] += int(
			bool(torch.count_nonzero(raw_frames[..., :QUERY_DIM]))
		)
		frames[..., :QUERY_DIM] = 0
		_require(
			int(torch.count_nonzero(frames[..., :QUERY_DIM])) == 0,
			'Query/appearance ablation left a non-zero query value.',
		)
		_require(
			torch.equal(frames[..., QUERY_DIM:], preserved),
			'Query/appearance ablation changed geometry/status values.',
		)
		diagnostics['query_zero_checks'] += 1
		diagnostics['query_geometry_status_preserved_checks'] += 1
	else:
		length = _burst_length(mode)
		start = BURST_STARTS[episode_index]
		active = start <= step_index < start + length
		if active:
			target = diagnostics['target_role_index']
			non_target = [
				index for index in range(SUPPORTED[task]['role_count'])
				if index != target
			]
			non_target_snapshot = _clone(raw[non_target])
			diagnostics['effective_nonzero_target_overwrites'] += int(
				bool(torch.count_nonzero(
					raw[target].reshape(STACK_FRAMES, FRAME_DIM)[..., :586]
				))
			)
			policy[target] = 0
			_require(
				int(torch.count_nonzero(policy[target])) == 0,
				'Target-role drop left a non-zero target value.',
			)
			_require(
				torch.equal(policy[non_target], non_target_snapshot),
				'Target-role drop changed a non-target role.',
			)
			diagnostics['target_role_zero_checks'] += 1
			diagnostics['non_target_roles_preserved_checks'] += 1
			diagnostics['applied_role_drop_frames'] += 1
		else:
			_require(torch.equal(policy, raw), 'Role-drop arm changed input outside burst.')
			diagnostics['outside_burst_equal_checks'] += 1

	_require(torch.equal(raw, source_snapshot), 'Intervention mutated live environment input.')
	diagnostics['source_unchanged_checks'] += 1
	# TensorWrapper returns a TensorDict, and TDMPC2.act relies on its ``to`` and
	# ``unsqueeze`` methods. Preserve that container type instead of replacing it
	# with a plain dict. The plain-mapping fallback exists only for light tests.
	policy_obs = obs.clone() if callable(getattr(obs, 'clone', None)) else dict(obs)
	policy_obs['object'] = policy
	return policy_obs


def _validate_source(raw: dict, task: str) -> dict:
	if task not in SUPPORTED:
		raise ValueError(f'Unsupported task {task!r}; choose {sorted(SUPPORTED)}.')
	spec = SUPPORTED[task]
	expected = {
		'task': task,
		'flat_anchor': True,
		'flat_anchor_mode': BACKEND,
		'cutie_object_observation_variant': 'full',
		'cutie_object_frame_schema': 'cutie_query_mask_status_v1',
		'cutie_object_role_names': list(spec['roles']),
		'cutie_object_num_roles': spec['role_count'],
		'cutie_object_frame_dim': FRAME_DIM,
		'cutie_object_stack_frames': STACK_FRAMES,
		'cutie_object_input_dim': STACKED_DIM,
		'cutie_object_role_dim': 64,
		'cutie_object_only_latent_dim': spec['latent_dim'],
		'latent_dim': spec['latent_dim'],
		'cutie_object_auxiliary_target': 'full_descriptor',
		'cutie_object_spatial_token_enabled': False,
		'cutie_object_variable_graph_enabled': False,
		'cutie_object_true_entity_enabled': False,
		'cutie_object_last_valid_memory': False,
		'cutie_object_policy_burst_plan': None,
		'cutie_object_belief_enabled': False,
		'cutie_object_belief_use_for_control': False,
		'object_state_supervision_enabled': False,
		'object_state_supervision_collect_labels': False,
		'object_state_bottleneck_enabled': False,
		# Both accepted frozen checkpoints use a six-frame, simulator-labelled
		# *offline support* pack to initialise Cutie.  This is not an online
		# privileged input, but it must be explicit in the scientific provenance.
		'cutie_object_support_schema': 'generic_indexed_v1',
		'cutie_object_allow_simulator_support': True,
		'cutie_object_allow_simulator_runtime': False,
		'cutie_object_allow_simulator_kinematics_runtime': False,
	}
	bad = {
		key: {'actual': raw.get(key), 'expected': value}
		for key, value in expected.items() if raw.get(key) != value
	}
	regression = raw.get('cutie_object_regression_encoder')
	if regression not in spec['regression_encoder']:
		bad['cutie_object_regression_encoder'] = {
			'actual': regression, 'expected_one_of': list(spec['regression_encoder']),
		}
	obs_shape = raw.get('obs_shape')
	if obs_shape != {'object': [spec['role_count'], STACKED_DIM]}:
		bad['obs_shape'] = {
			'actual': obs_shape,
			'expected': {'object': [spec['role_count'], STACKED_DIM]},
		}
	if bad:
		raise ValueError(f'Unsupported/non-causal source runtime contract: {bad}.')
	return {
		'task': task,
		'roles': list(spec['roles']),
		'target_role': spec['target_role'],
		'role_count': spec['role_count'],
		'frame_layout': {
			'frame_dim': FRAME_DIM, 'stack_frames': STACK_FRAMES,
			'query_appearance_dim': QUERY_DIM,
			'mask_geometry_status_dim': GEOMETRY_STATUS_DIM,
		},
		'regression_encoder': regression,
		'latent_dim': spec['latent_dim'],
		'novelty': spec['novelty'],
	}


def _support_provenance(raw: dict, path: Path, task: str) -> dict:
	"""Validate and report the frozen support supervision without hiding it."""
	data = base._json(path)
	collection = data.get('collection')
	checks = {
		'format': data.get('format') == 'cutie_indexed_mask_support_v1',
		'roles': data.get('roles') == list(SUPPORTED[task]['roles']),
		'collection_object': isinstance(collection, dict),
		'task': isinstance(collection, dict) and collection.get('task') == task,
		'split': isinstance(collection, dict) and collection.get('split') == 'support',
		'observation': isinstance(collection, dict)
			and collection.get('observation') == 'rgb',
		'support_schema': isinstance(collection, dict)
			and collection.get('support_schema') == 'generic_indexed_v1'
			and raw.get('cutie_object_support_schema') == 'generic_indexed_v1',
		'label_policy': isinstance(collection, dict)
			and collection.get('label_policy') == 'simulator_segmentation_support_only',
		'diagnostic_support': isinstance(collection, dict)
			and collection.get('diagnostic_support') is True,
		'explicit_opt_in': raw.get('cutie_object_allow_simulator_support') is True,
		'no_runtime_segmentation': raw.get('cutie_object_allow_simulator_runtime') is False,
		'no_runtime_kinematics':
			raw.get('cutie_object_allow_simulator_kinematics_runtime') is False,
	}
	failed = sorted(name for name, passed in checks.items() if not passed)
	if failed:
		raise ValueError('Frozen Cutie support provenance mismatch: ' + ', '.join(failed))
	return {
		'format': data['format'],
		'support_schema': collection['support_schema'],
		'label_policy': collection['label_policy'],
		'split': collection['split'],
		'task': collection['task'],
		'roles': list(data['roles']),
		'allow_simulator_support': True,
		'evaluation_runtime_simulator_segmentation': False,
		'evaluation_runtime_simulator_kinematics': False,
		'checks': checks,
	}


def _load_checkpoint_contract(path: Path, raw: dict, task: str, torch) -> dict:
	payload = torch.load(path, map_location='cpu', weights_only=False)
	if not isinstance(payload, dict) or not isinstance(payload.get('model'), dict):
		raise ValueError('Checkpoint is not a TD-MPC2 model payload.')
	contract = payload.get('checkpoint_contract')
	if not isinstance(contract, dict):
		raise ValueError('Checkpoint is missing checkpoint_contract.')
	spec = SUPPORTED[task]
	observation = contract.get('cutie_object_observation')
	auxiliary = contract.get('cutie_object_auxiliary')
	checks = {
		'format': contract.get('format') == 'tdmpc2_checkpoint_contract_v1',
		'mode': contract.get('flat_anchor_mode') == BACKEND,
		'latent_dim': contract.get('latent_dim') == spec['latent_dim'],
		'regression_encoder': contract.get('cutie_object_regression_encoder')
			== raw.get('cutie_object_regression_encoder'),
		'belief_disabled': contract.get('cutie_object_belief_enabled') is False,
		'no_state_supervision_contract': contract.get('object_state_supervision') is None,
		'observation_contract': isinstance(observation, dict),
		'observation_variant': isinstance(observation, dict)
			and observation.get('variant') == 'full',
		'no_privileged_runtime_segmentation': isinstance(observation, dict)
			and observation.get('privileged_runtime_segmentation') is False,
		'role_count': isinstance(observation, dict)
			and observation.get('num_roles') == spec['role_count'],
		'frame_dim': isinstance(observation, dict)
			and observation.get('frame_dim') == FRAME_DIM,
		'stack_frames': isinstance(observation, dict)
			and observation.get('stack_frames') == STACK_FRAMES,
		'input_dim': isinstance(observation, dict)
			and observation.get('input_dim') == STACKED_DIM,
		'non_spatial': isinstance(observation, dict)
			and observation.get('spatial_token_enabled') is False,
		'full_descriptor_auxiliary': isinstance(auxiliary, dict)
			and auxiliary.get('target') == 'full_descriptor',
	}
	failed = sorted(name for name, passed in checks.items() if not passed)
	if failed:
		raise ValueError('Checkpoint/runtime causal-ablation mismatch: ' + ', '.join(failed))
	return {
		'checks': checks,
		'format': contract['format'],
		'flat_anchor_mode': contract['flat_anchor_mode'],
		'latent_dim': contract['latent_dim'],
		'regression_encoder': contract.get('cutie_object_regression_encoder'),
		'observation': observation,
		'auxiliary': auxiliary,
	}


def _arm_args(args, condition: str):
	arm = copy.copy(args)
	arm.backend = BACKEND
	arm.condition = condition
	arm.erosion_pixels = 0
	# The base validator only uses this path for collision prevention.  This
	# evaluator publishes one aggregate output atomically after all arms finish.
	arm.output = args.output.with_name(
		f'.{args.output.name}.{condition}.causal-arm-not-written'
	)
	return arm


def _validate_diagnostics(mode: str, diagnostics: dict) -> tuple[dict[str, bool], dict[str, bool]]:
	total = EPISODES * DECISION_STEPS
	length = _burst_length(mode)
	expected_drop = EPISODES * length
	engineering = {
		'decision_steps': diagnostics['decision_steps_seen'] == total,
		'copied_each_step': diagnostics['copied_input_checks'] == total,
		'source_unchanged_each_step': diagnostics['source_unchanged_checks'] == total,
		'live_environment_observation_unchanged':
			diagnostics['live_environment_observation_mutated'] is False,
	}
	attribution = {}
	if mode == 'full':
		engineering['full_copy_equal'] = diagnostics['full_copy_equal_checks'] == total
	elif mode == 'query_appearance_zero':
		engineering.update({
			'query_zero_every_step': diagnostics['query_zero_checks'] == total,
			'geometry_status_preserved_every_step':
				diagnostics['query_geometry_status_preserved_checks'] == total,
		})
		attribution['query_ablation_effective_at_least_95pct'] = (
			diagnostics['query_effective_nonzero_frames'] >= int(0.95 * total)
		)
	else:
		engineering.update({
			'scheduled_drop_exact':
				diagnostics['scheduled_role_drop_frames'] == expected_drop,
			'applied_drop_exact': diagnostics['applied_role_drop_frames'] == expected_drop,
			'target_zero_exact': diagnostics['target_role_zero_checks'] == expected_drop,
			'non_target_preserved_exact':
				diagnostics['non_target_roles_preserved_checks'] == expected_drop,
			'outside_burst_equal': diagnostics['outside_burst_equal_checks']
				== total - expected_drop,
		})
		attribution['target_overwrite_effective_at_least_80pct'] = (
			diagnostics['effective_nonzero_target_overwrites']
			>= int(0.80 * expected_drop)
		)
	return engineering, attribution


def _evaluate_arm(args, raw: dict, *, condition: str, mode: str):
	import numpy as np
	import torch
	from common.seed import set_seed
	from envs import make_env
	from tdmpc2.tdmpc2 import TDMPC2

	arm_args = _arm_args(args, condition)
	base._validate_args(arm_args)
	cfg = base._prepare(arm_args, raw)
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required for the frozen Cutie checkpoint evaluator.')
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.set_float32_matmul_precision('high')

	diagnostics = _new_diagnostics(args.task, mode)
	env = None
	records = []
	alignment_draws = 0
	perception = ready = manifest = combined_manifest = None
	state_label_reads = None
	started = perf_counter()
	try:
		set_seed(args.env_seed)
		env = make_env(cfg)
		active_split = base._env_value(env, 'active_split')
		if condition == 'hard':
			_require(
				active_split == base.VALIDATION_SPLIT,
				'Hard arm did not construct the validation background split.',
			)
		else:
			_require(active_split is None, 'Clean arm unexpectedly constructed a background.')
		agent = TDMPC2(cfg)
		agent.load(args.checkpoint)
		agent.eval()
		for episode_index in range(EPISODES):
			obs = env.reset()
			_validate_object(obs, task=args.task, torch=torch)
			initial_rgb = base._rgb_hash(obs, env, BACKEND)
			initial_raw_object = base._object_hash(obs, BACKEND)
			source = base._env_value(env, 'active_source')
			frame_index = base._env_value(env, 'frame_index')
			if condition == 'hard':
				_require(source is not None and frame_index is not None, 'Missing background provenance.')
			else:
				source, frame_index = 'clean', 0
			planner_seed = args.planner_seed_base + episode_index
			set_seed(planner_seed)
			agent._prev_mean.zero_()
			rng_start = base._rng_hash(torch)
			reward_sum = 0.0
			info = {}
			raw_trace = hashlib.sha256()
			policy_trace = hashlib.sha256()
			prefix_raw_trace = hashlib.sha256()
			prefix_policy_trace = hashlib.sha256()
			prefix_action_trace = hashlib.sha256()
			comparison_prefix_decisions = BURST_STARTS[episode_index]
			initial_policy_object = None
			applied_before = diagnostics['applied_role_drop_frames']
			for step_index in range(DECISION_STEPS):
				torch.compiler.cudagraph_mark_step_begin()
				raw_object = obs['object']
				policy_obs = _apply_intervention(
					obs, task=args.task, mode=mode,
					episode_index=episode_index, step_index=step_index,
					torch=torch, diagnostics=diagnostics,
				)
				_trace_update(
					raw_trace, episode=episode_index, step=step_index,
					value=raw_object,
				)
				_trace_update(
					policy_trace, episode=episode_index, step=step_index,
					value=policy_obs['object'],
				)
				if step_index < comparison_prefix_decisions:
					_trace_update(
						prefix_raw_trace, episode=episode_index, step=step_index,
						value=raw_object,
					)
					_trace_update(
						prefix_policy_trace, episode=episode_index, step=step_index,
						value=policy_obs['object'],
					)
				if initial_policy_object is None:
					initial_policy_object = _hash_tensor(policy_obs['object'])
				base._align_object_only_rgb_shift_rng(torch, BACKEND)
				alignment_draws += 1
				action = agent.act(policy_obs, t0=step_index == 0, eval_mode=True)
				if step_index < comparison_prefix_decisions:
					_trace_update(
						prefix_action_trace, episode=episode_index, step=step_index,
						value=action,
					)
				obs, reward, done, info = env.step(action)
				reward_sum += float(reward)
				if done:
					length = step_index + 1
					break
			else:
				raise RuntimeError('Environment did not terminate at 500 decisions.')
			_require(length == DECISION_STEPS, f'Unexpected episode length {length}.')
			record = {
				'episode_index': episode_index,
				'planner_seed': planner_seed,
				'planner_rng_start_sha256': rng_start,
				'planner_rng_end_sha256': base._rng_hash(torch),
				'initial_rgb_sha256': initial_rgb,
				'initial_raw_object_sha256': initial_raw_object,
				'initial_policy_object_sha256': initial_policy_object,
				'raw_policy_input_trace_sha256': raw_trace.hexdigest(),
				'intervened_policy_input_trace_sha256': policy_trace.hexdigest(),
				'comparison_prefix_decisions': comparison_prefix_decisions,
				'pre_intervention_raw_object_trace_sha256': prefix_raw_trace.hexdigest(),
				'pre_intervention_policy_input_trace_sha256':
					prefix_policy_trace.hexdigest(),
				'pre_intervention_action_trace_sha256': prefix_action_trace.hexdigest(),
				'background_source': Path(source).name,
				'background_start_frame_index': int(frame_index),
				'intervention_start_decision': (
					BURST_STARTS[episode_index] if _burst_length(mode) else None
				),
				'intervention_length': _burst_length(mode),
				'intervention_applied_decisions':
					diagnostics['applied_role_drop_frames'] - applied_before,
				'reward': reward_sum,
				'success': float(info.get('success', 0.0)),
				'length': length,
			}
			records.append(record)
			print('OBJECT_INPUT_CAUSAL_ABLATION_EPISODE', json.dumps({
				'task': args.task, 'condition': condition, 'mode': mode, **record,
			}, allow_nan=False), flush=True)
		perception = base._metrics(env)
		ready = base._env_value(env, 'cutie_ready')
		manifest = base._env_value(env, 'manifest_sha256')
		combined_manifest = base._env_value(env, 'combined_manifest_sha256')
		state_label_reads = base._env_value(env, 'state_supervision_label_reads', 0)
		_require(state_label_reads in (None, 0), 'Evaluation read privileged state labels.')
	finally:
		if env is not None and callable(getattr(env, 'close', None)):
			env.close()

	rewards = np.asarray([row['reward'] for row in records], dtype=np.float64)
	_require(rewards.shape == (EPISODES,) and np.isfinite(rewards).all(), 'Bad rewards.')
	_require(isinstance(perception, dict), 'Cutie runtime provenance is unavailable.')
	_require(perception.get('frames') == EPISODES * (DECISION_STEPS + 1), 'Frame count drifted.')
	_require(isinstance(ready, dict), 'Cutie ready provenance is unavailable.')
	checks, attribution_checks = _validate_diagnostics(mode, diagnostics)
	engineering_pass = all(checks.values())
	attribution_eligible = engineering_pass and all(attribution_checks.values())
	return {
		'format': ARM_FORMAT,
		'task': args.task,
		'condition': condition,
		'mode': mode,
		'roles': list(SUPPORTED[args.task]['roles']),
		'target_role': SUPPORTED[args.task]['target_role'],
		'evaluation': {
			'episodes': EPISODES,
			'decision_steps': DECISION_STEPS,
			'env_seed': args.env_seed,
			'background_seed': args.background_seed,
			'planner_seed_base': args.planner_seed_base,
			'paired_exogenous_schedule_only': True,
			'identical_endogenous_state_trajectory_claimed': False,
			'privileged_state_or_gt_read': False,
			'rgb_shift_rng_alignment_draws': alignment_draws,
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
		'intervention': {
			**diagnostics,
			'effective_query_rate': (
				diagnostics['query_effective_nonzero_frames'] / (EPISODES * DECISION_STEPS)
				if mode == 'query_appearance_zero' else None
			),
			'effective_target_overwrite_rate': (
				diagnostics['effective_nonzero_target_overwrites']
				/ diagnostics['scheduled_role_drop_frames']
				if diagnostics['scheduled_role_drop_frames'] else None
			),
			'checks': checks,
			'engineering_pass': engineering_pass,
			'attribution_checks': attribution_checks,
			'causal_attribution_eligible': attribution_eligible,
		},
		'perception_runtime': perception,
		'provenance': {
			'validation_manifest_sha256': manifest,
			'combined_manifest_sha256': combined_manifest,
			'cutie_ready': ready,
			'state_supervision_label_reads': state_label_reads,
		},
	}


def _paired_statistics(full_rows: list[dict], ablated_rows: list[dict], *, seed: int):
	import numpy as np

	full = np.asarray([row['reward'] for row in full_rows], dtype=np.float64)
	ablated = np.asarray([row['reward'] for row in ablated_rows], dtype=np.float64)
	if full.shape != ablated.shape or full.shape != (EPISODES,):
		raise ValueError('Paired statistics require two complete 20-episode arms.')
	delta = full - ablated
	rng = np.random.default_rng(seed)
	indices = rng.integers(0, EPISODES, size=(BOOTSTRAP_SAMPLES, EPISODES))
	bootstrap = delta[indices].mean(axis=1)
	return {
		'direction': 'full_reward_minus_ablated_reward; positive_means_input_helped',
		'paired_episode_count': EPISODES,
		'mean': float(delta.mean()),
		'median': float(np.median(delta)),
		'std': float(delta.std(ddof=1)),
		'bootstrap_samples': BOOTSTRAP_SAMPLES,
		'bootstrap_95_ci': [
			float(np.quantile(bootstrap, 0.025)),
			float(np.quantile(bootstrap, 0.975)),
		],
		'fraction_full_better': float((delta > 0).mean()),
		'full_mean': float(full.mean()),
		'ablated_mean': float(ablated.mean()),
		'relative_mean_drop': (
			float(delta.mean() / abs(full.mean())) if full.mean() != 0 else None
		),
	}


def _pairing_checks(
	full_rows: list[dict], other_rows: list[dict], *, mode: str,
) -> dict[str, bool]:
	fields = (
		'episode_index', 'planner_seed', 'planner_rng_start_sha256',
		'initial_rgb_sha256', 'initial_raw_object_sha256',
		'background_source', 'background_start_frame_index', 'length',
	)
	checks = {
		'complete_20_episode_pair': len(full_rows) == len(other_rows) == EPISODES,
		**{
			f'identical_{field}': [row.get(field) for row in full_rows]
				== [row.get(field) for row in other_rows]
			for field in fields
		},
	}
	if _burst_length(mode):
		# A valid policy-input intervention must not have diverged before the
		# scheduled burst. This catches nondeterministic tracker/CUDA output as
		# well as an accidental early overwrite.
		for field in (
			'comparison_prefix_decisions',
			'pre_intervention_raw_object_trace_sha256',
			'pre_intervention_policy_input_trace_sha256',
			'pre_intervention_action_trace_sha256',
		):
			checks[f'identical_{field}'] = [row.get(field) for row in full_rows] \
				== [row.get(field) for row in other_rows]
	return checks


def _preflight(args, torch) -> dict:
	"""Validate the frozen run without constructing an environment or CUDA state."""
	if args.output.exists():
		raise FileExistsError(args.output)
	if tuple(args.conditions) != CONDITIONS:
		raise ValueError('Frozen causal ablation requires conditions in order: clean hard.')
	if args.episodes != EPISODES:
		raise ValueError(f'Frozen causal ablation requires exactly {EPISODES} episodes.')
	for path in (args.runtime_config, args.checkpoint):
		if not path.is_file():
			raise FileNotFoundError(path)
	raw = base._json(args.runtime_config)
	source_contract = _validate_source(raw, args.task)
	# Exercise the base evaluator's exact 100k/20k/3, single-task RGB, model-size,
	# background, and checkpoint-location contracts for both requested conditions.
	# _prepare is CPU/configuration-only; it does not create an environment.
	for condition in CONDITIONS:
		arm_args = _arm_args(args, condition)
		base._validate_args(arm_args)
		base._prepare(arm_args, raw)
	checkpoint_sha_before = base._sha256(args.checkpoint)
	runtime_sha_before = base._sha256(args.runtime_config)
	checkpoint_contract = _load_checkpoint_contract(
		args.checkpoint, raw, args.task, torch
	)
	cutie_checkpoint = Path(raw['cutie_object_checkpoint']).resolve()
	cutie_support = Path(raw['cutie_object_support_path']).resolve()
	for path in (cutie_checkpoint, cutie_support):
		if not path.is_file():
			raise FileNotFoundError(path)
	support_provenance = _support_provenance(raw, cutie_support, args.task)
	cutie_checkpoint_sha_before = base._sha256(cutie_checkpoint)
	cutie_support_sha_before = base._sha256(cutie_support)
	return {
		'raw': raw,
		'source_contract': source_contract,
		'checkpoint_contract': checkpoint_contract,
		'runtime_config_sha256': runtime_sha_before,
		'checkpoint_sha256': checkpoint_sha_before,
		'cutie_checkpoint': cutie_checkpoint,
		'cutie_checkpoint_sha256': cutie_checkpoint_sha_before,
		'cutie_support': cutie_support,
		'cutie_support_sha256': cutie_support_sha_before,
		'support_training_protocol': support_provenance,
	}


def validate_only(args):
	"""Publish a CPU-only readiness record; never create an env or run a policy."""
	import torch

	preflight = _preflight(args, torch)
	immutable_checks = {
		'runtime_config_unchanged': base._sha256(args.runtime_config)
			== preflight['runtime_config_sha256'],
		'checkpoint_unchanged': base._sha256(args.checkpoint)
			== preflight['checkpoint_sha256'],
		'cutie_checkpoint_unchanged': base._sha256(preflight['cutie_checkpoint'])
			== preflight['cutie_checkpoint_sha256'],
		'cutie_support_unchanged': base._sha256(preflight['cutie_support'])
			== preflight['cutie_support_sha256'],
	}
	if not all(immutable_checks.values()):
		raise RuntimeError(f'An input changed during validation: {immutable_checks}.')
	return {
		'format': VALIDATION_FORMAT,
		'status': 'object_input_causal_ablation_ready',
		'engineering_pass': True,
		'execution_launched': False,
		'task': args.task,
		'conditions': list(CONDITIONS),
		'modes': list(MODES),
		'episodes_per_condition_per_mode': EPISODES,
		'source_contract': preflight['source_contract'],
		'checkpoint_contract': preflight['checkpoint_contract'],
		'support_training_protocol': preflight['support_training_protocol'],
		'forbidden_runtime_inputs': [
			'evaluation_trajectory_ground_truth_masks',
			'evaluation_simulator_state',
			'evaluation_simulator_kinematics',
		],
		'provenance': {
			'runtime_config': str(args.runtime_config.resolve()),
			'runtime_config_sha256': preflight['runtime_config_sha256'],
			'checkpoint': str(args.checkpoint.resolve()),
			'checkpoint_sha256': preflight['checkpoint_sha256'],
			'checkpoint_kind': 'final' if args.checkpoint_step is None else 'periodic_eval',
			'checkpoint_step': (
				args.expected_training_steps
				if args.checkpoint_step is None else args.checkpoint_step
			),
			'cutie_checkpoint': str(preflight['cutie_checkpoint']),
			'cutie_checkpoint_sha256': preflight['cutie_checkpoint_sha256'],
			'cutie_support': str(preflight['cutie_support']),
			'cutie_support_sha256': preflight['cutie_support_sha256'],
			'support_schema': preflight['support_training_protocol']['support_schema'],
			'support_label_policy': preflight['support_training_protocol']['label_policy'],
			'allow_simulator_support': True,
			'evaluation_runtime_privileged_inputs': False,
			'evaluator': str(Path(__file__).resolve()),
			'evaluator_sha256': base._sha256(Path(__file__).resolve()),
			'base_evaluator': str(Path(base.__file__).resolve()),
			'base_evaluator_sha256': base._sha256(Path(base.__file__).resolve()),
		},
		'immutable_checks': immutable_checks,
	}


def evaluate(args):
	import torch

	preflight = _preflight(args, torch)
	raw = preflight['raw']
	source_contract = preflight['source_contract']
	checkpoint_contract = preflight['checkpoint_contract']
	checkpoint_sha_before = preflight['checkpoint_sha256']
	runtime_sha_before = preflight['runtime_config_sha256']
	cutie_checkpoint = preflight['cutie_checkpoint']
	cutie_support = preflight['cutie_support']
	cutie_checkpoint_sha_before = preflight['cutie_checkpoint_sha256']
	cutie_support_sha_before = preflight['cutie_support_sha256']

	conditions = {}
	all_checks = {}
	causal_checks = {}
	for condition in CONDITIONS:
		arms = {}
		for mode in MODES:
			arms[mode] = _evaluate_arm(
				args, raw, condition=condition, mode=mode
			)
		full_rows = arms['full']['episodes']
		pairing = {}
		contrasts = {}
		for mode_index, mode in enumerate(MODES[1:], start=1):
			checks = _pairing_checks(full_rows, arms[mode]['episodes'], mode=mode)
			pairing[mode] = checks
			causal_checks[f'{condition}/{mode}/paired_exogenous_schedule'] = all(
				checks.values()
			)
			contrasts[f'full_minus_{mode}'] = _paired_statistics(
				full_rows, arms[mode]['episodes'],
				seed=271_828 + 100 * CONDITIONS.index(condition) + mode_index,
			)
		for mode in MODES:
			all_checks[f'{condition}/{mode}/engineering'] = arms[mode][
				'intervention'
			]['engineering_pass']
			causal_checks[f'{condition}/{mode}/effective_intervention'] = arms[mode][
				'intervention'
			]['causal_attribution_eligible']
		conditions[condition] = {
			'arms': arms,
			'paired_exogenous_checks': pairing,
			'paired_return_contrasts': contrasts,
		}

	immutable_checks = {
		'runtime_config_unchanged': base._sha256(args.runtime_config) == runtime_sha_before,
		'checkpoint_unchanged': base._sha256(args.checkpoint) == checkpoint_sha_before,
		'cutie_checkpoint_unchanged': base._sha256(cutie_checkpoint)
			== cutie_checkpoint_sha_before,
		'cutie_support_unchanged': base._sha256(cutie_support) == cutie_support_sha_before,
	}
	all_checks.update({f'immutable/{key}': value for key, value in immutable_checks.items()})
	engineering_pass = all(all_checks.values())
	causal_attribution_eligible = engineering_pass and all(causal_checks.values())
	return {
		'format': FORMAT,
		'status': (
			'object_input_causal_ablation_complete'
			if engineering_pass else 'object_input_causal_ablation_engineering_fail'
		),
		'engineering_pass': engineering_pass,
		'causal_attribution_eligible': causal_attribution_eligible,
		'scientific_scope': (
			'single frozen training seed; evaluation-only policy-input intervention; '
			'no retraining, no evaluation-trajectory simulator state or GT; the frozen '
			'Cutie support pack remains part of the existing frontend; paired initial states and '
			'exogenous clean/hard schedules; endogenous trajectories may diverge after '
			'actions and results are diagnostic rather than a final paper claim'
		),
		'not_a_repeat_of': [
			'CutieHybrid whole RGB/object field zeroing',
			(
				'Reacher/Cartpole raw-tracker-frame-before-stack hard-zero versus '
				'last-valid-memory burst probes; this intervention is the complete '
				'stacked role slot immediately before agent.act'
			),
		],
		'source_contract': source_contract,
		'checkpoint_contract': checkpoint_contract,
		'support_training_protocol': preflight['support_training_protocol'],
		'conditions': conditions,
		'checks': all_checks,
		'causal_attribution_checks': causal_checks,
		'provenance': {
			'runtime_config': str(args.runtime_config.resolve()),
			'runtime_config_sha256': runtime_sha_before,
			'checkpoint': str(args.checkpoint.resolve()),
			'checkpoint_sha256': checkpoint_sha_before,
			'checkpoint_kind': 'final' if args.checkpoint_step is None else 'periodic_eval',
			'checkpoint_step': (
				args.expected_training_steps
				if args.checkpoint_step is None else args.checkpoint_step
			),
			'cutie_checkpoint': str(cutie_checkpoint),
			'cutie_checkpoint_sha256': cutie_checkpoint_sha_before,
			'cutie_support': str(cutie_support),
			'cutie_support_sha256': cutie_support_sha_before,
			'support_schema': preflight['support_training_protocol']['support_schema'],
			'support_label_policy': preflight['support_training_protocol']['label_policy'],
			'allow_simulator_support': True,
			'evaluation_runtime_privileged_inputs': False,
			'evaluator': str(Path(__file__).resolve()),
			'evaluator_sha256': base._sha256(Path(__file__).resolve()),
			'base_evaluator': str(Path(base.__file__).resolve()),
			'base_evaluator_sha256': base._sha256(Path(base.__file__).resolve()),
			'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
		},
		'immutable_checks': immutable_checks,
	}


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--task', choices=tuple(SUPPORTED), required=True)
	parser.add_argument('--runtime-config', type=Path, required=True)
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--checkpoint-step', type=int)
	parser.add_argument('--training-seed', type=int, default=6)
	parser.add_argument('--expected-training-steps', type=int, default=100000)
	parser.add_argument('--expected-training-eval-freq', type=int, default=20000)
	parser.add_argument('--expected-training-eval-episodes', type=int, default=3)
	parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
	parser.add_argument('--episodes', type=int, default=EPISODES)
	parser.add_argument('--env-seed', type=int, default=424243)
	parser.add_argument('--background-seed', type=int, default=1618034)
	parser.add_argument('--planner-seed-base', type=int, default=8675400)
	parser.add_argument(
		'--validate-only', action='store_true',
		help='CPU-only runtime/checkpoint/support validation; never creates an environment.',
	)
	parser.add_argument('--output', type=Path, required=True)
	return parser.parse_args(argv)


def main(argv=None):
	args = parse_args(argv)
	payload = validate_only(args) if args.validate_only else evaluate(args)
	base._write(args.output, payload)
	marker = (
		'OBJECT_INPUT_CAUSAL_ABLATION_READY'
		if args.validate_only else 'OBJECT_INPUT_CAUSAL_ABLATION_COMPLETE'
	)
	print(marker, json.dumps({
		'task': args.task,
		'status': payload['status'],
		'engineering_pass': payload['engineering_pass'],
		'output': str(args.output.resolve()),
	}, allow_nan=False), flush=True)
	return 0 if payload['engineering_pass'] else 1


if __name__ == '__main__':
	raise SystemExit(main())
