"""Aggregate the paired 500k Acrobot factorized/proprio-only experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path


MODES = ('factorized', 'factorized_proprio_only')
BACKEND = {
	'factorized': 'cutie_proprio_factorized',
	'factorized_proprio_only': 'cutie_proprio_factorized_proprio_only',
}
STEPS = 500_000
EVAL_FREQ = 25_000
EVAL_EPISODES = 10
HELD_OUT_EPISODES = 20
EXPECTED_STEPS = tuple(range(0, STEPS + 1, EVAL_FREQ))


def _load(path):
	return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha256(path):
	digest = hashlib.sha256()
	with Path(path).open('rb') as stream:
		for block in iter(lambda: stream.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def _curve(root):
	path = root / 'eval.csv'
	with path.open(newline='', encoding='utf-8') as stream:
		rows = [
			{
				'step': int(float(row['step'])),
				'reward': float(row['episode_reward']),
			}
			for row in csv.DictReader(stream)
		]
	if tuple(row['step'] for row in rows) != EXPECTED_STEPS:
		raise ValueError(f'{path}: unexpected evaluation steps.')
	if not all(math.isfinite(row['reward']) for row in rows):
		raise ValueError(f'{path}: non-finite reward.')
	return rows


def _trace(root, curve):
	path = root / 'eval_episodes.jsonl'
	rows = [
		json.loads(line)
		for line in path.read_text(encoding='utf-8').splitlines()
		if line.strip()
	]
	if len(rows) != len(EXPECTED_STEPS):
		raise ValueError(f'{path}: incomplete evaluation trace.')
	for index, (step, row, point) in enumerate(zip(EXPECTED_STEPS, rows, curve)):
		rewards = row.get('episode_rewards')
		if (
			row.get('evaluation_index') != index
			or row.get('step') != step
			or row.get('episodes') != EVAL_EPISODES
			or not isinstance(rewards, list)
			or len(rewards) != EVAL_EPISODES
		):
			raise ValueError(f'{path}: malformed row at step {step}.')
		if not all(math.isfinite(float(value)) for value in rewards):
			raise ValueError(f'{path}: non-finite reward at step {step}.')
		if not math.isclose(
			float(row['reward_mean']), point['reward'], rel_tol=1e-5, abs_tol=1e-5
		):
			raise ValueError(f'{path}: trace/CSV mismatch at step {step}.')
	return rows


def _auc(curve):
	return sum(
		(right['step'] - left['step'])
		* (left['reward'] + right['reward']) / 2.0
		for left, right in zip(curve, curve[1:])
	) / STEPS


def _validate_training(root, mode):
	root = Path(root).expanduser().resolve()
	config_path = root / 'runtime_config.json'
	config = _load(config_path)
	expected = {
		'task': 'acrobot-swingup',
		'obs': 'rgb',
		'seed': 10,
		'steps': STEPS,
		'eval_freq': EVAL_FREQ,
		'eval_episodes': EVAL_EPISODES,
		'save_eval_episode_trace': True,
		'save_eval_checkpoints': True,
		'video_background_enabled': True,
		'video_background_split': 'train',
		'flat_anchor': True,
		'flat_anchor_mode': 'cutie_object_only',
		'cutie_object_observation_variant': 'cutie_proprio',
		'cutie_object_frame_schema': 'cutie_query_mask_status_plus_proprio_v1',
		'cutie_object_role_names': ['upper_arm', 'lower_arm'],
		'cutie_object_support_schema': 'generic_indexed_v1',
		'cutie_object_num_roles': 2,
		'cutie_object_frame_dim': 1774,
		'cutie_object_stack_frames': 1,
		'cutie_object_input_dim': 1774,
		'cutie_object_only_latent_dim': 128,
		'cutie_object_allow_simulator_runtime': False,
		'cutie_object_allow_simulator_kinematics_runtime': True,
		'cutie_proprio_mode': mode,
		'visual_pose_checkpoint': None,
	}
	bad = {
		key: {'actual': config.get(key), 'expected': value}
		for key, value in expected.items()
		if config.get(key) != value
	}
	if config.get('obs_shape') != {'object': [2, 1774]}:
		bad['obs_shape'] = {
			'actual': config.get('obs_shape'),
			'expected': {'object': [2, 1774]},
		}
	if int(config.get('latent_dim', -1)) != 128:
		bad['latent_dim'] = {'actual': config.get('latent_dim'), 'expected': 128}
	if bad:
		raise ValueError(f'{mode}: runtime mismatch {bad}.')

	curve = _curve(root)
	trace = _trace(root, curve)
	# Step zero is intentionally excluded from model selection.
	best = max(curve[1:], key=lambda row: (row['reward'], -row['step']))
	periodic = {}
	for step in EXPECTED_STEPS[1:]:
		path = root / 'models' / f'eval_{step}.pt'
		if not path.is_file():
			raise FileNotFoundError(path)
		periodic[str(step)] = {
			'path': str(path), 'bytes': path.stat().st_size, 'sha256': _sha256(path),
		}
	final_path = root / 'models' / 'final.pt'
	if not final_path.is_file():
		raise FileNotFoundError(final_path)
	runtime_path = root / 'trainer_runtime.json'
	return {
		'root': str(root),
		'runtime_config': str(config_path),
		'curve': curve,
		'evaluation_trace': trace,
		'normalized_auc': _auc(curve),
		'peak_reward': float(best['reward']),
		'best_step': int(best['step']),
		'final_reward': float(curve[-1]['reward']),
		'final_reward_std': float(trace[-1]['reward_std']),
		'final_minus_peak': float(curve[-1]['reward'] - best['reward']),
		'periodic_checkpoints': periodic,
		'final_checkpoint': {
			'path': str(final_path),
			'bytes': final_path.stat().st_size,
			'sha256': _sha256(final_path),
		},
		'trainer_runtime': _load(runtime_path) if runtime_path.is_file() else None,
	}


def _validate_evaluation(path, mode, condition, selection, expected_step, root):
	path = Path(path).expanduser().resolve()
	payload = _load(path)
	if payload.get('task') != 'acrobot-swingup':
		raise ValueError(f'{path}: task mismatch.')
	if payload.get('backend') != BACKEND[mode]:
		raise ValueError(f'{path}: backend mismatch.')
	if payload.get('condition') != condition:
		raise ValueError(f'{path}: condition mismatch.')
	episodes = payload.get('episodes')
	if not isinstance(episodes, list) or len(episodes) != HELD_OUT_EPISODES:
		raise ValueError(f'{path}: incomplete held-out episodes.')
	if any(row.get('length') != 500 for row in episodes):
		raise ValueError(f'{path}: episode length mismatch.')
	provenance = payload.get('provenance', {})
	expected_kind = 'final' if selection == 'final' else 'periodic_eval'
	if (
		provenance.get('checkpoint_kind') != expected_kind
		or provenance.get('checkpoint_step') != expected_step
	):
		raise ValueError(f'{path}: checkpoint selection mismatch.')
	if Path(provenance.get('runtime_config', '')).resolve() != (
		Path(root).resolve() / 'runtime_config.json'
	):
		raise ValueError(f'{path}: runtime source mismatch.')
	return {
		'path': str(path),
		'checkpoint_kind': expected_kind,
		'checkpoint_step': int(expected_step),
		**{
			key: float(value)
			for key, value in payload['summary'].items()
			if key.startswith('reward_')
		},
		'perception_runtime': payload.get('perception_runtime'),
	}


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	for mode in MODES:
		key = mode.replace('_', '-')
		parser.add_argument(f'--{key}-root', required=True)
		for selection in ('best', 'final'):
			for condition in ('clean', 'hard'):
				parser.add_argument(
					f'--{key}-{selection}-{condition}', required=True
				)
	parser.add_argument('--output', required=True)
	args = parser.parse_args()

	runs = {}
	for mode in MODES:
		attr = mode
		root = getattr(args, f'{attr}_root')
		training = _validate_training(root, mode)
		held_out = {}
		for selection in ('best', 'final'):
			expected_step = (
				training['best_step'] if selection == 'best' else STEPS
			)
			held_out[selection] = {
				condition: _validate_evaluation(
					getattr(args, f'{attr}_{selection}_{condition}'),
					mode, condition, selection, expected_step, root,
				)
				for condition in ('clean', 'hard')
			}
		runs[mode] = {'training': training, 'held_out': held_out}

	factorized = runs['factorized']
	proprio = runs['factorized_proprio_only']
	payload = {
		'format': 'acrobot_factorized_proprio_500k_v1',
		'status': 'acrobot_factorized_proprio_500k_complete',
		'engineering_pass': True,
		'paper_claim_authorized': False,
		'protocol': {
			'task': 'acrobot-swingup',
			'training_seed': 10,
			'steps': STEPS,
			'eval_freq': EVAL_FREQ,
			'eval_episodes': EVAL_EPISODES,
			'held_out_episodes': HELD_OUT_EPISODES,
			'training_background': 'hard/train',
			'held_out_backgrounds': ['clean', 'hard/validation'],
			'model_selection': (
				'highest ten-episode training evaluation mean; step zero excluded; '
				'earliest step wins ties'
			),
			'held_out_not_used_for_selection': True,
			'world_models_per_run': 1,
			'latent': {'body': 64, 'object': 64, 'combined': 128},
		},
		'runs': runs,
		'deltas': {
			'factorized_minus_proprio_auc': (
				factorized['training']['normalized_auc']
				- proprio['training']['normalized_auc']
			),
			'factorized_minus_proprio_final': (
				factorized['training']['final_reward']
				- proprio['training']['final_reward']
			),
			'best_held_out_reward_mean': {
				condition: (
					factorized['held_out']['best'][condition]['reward_mean']
					- proprio['held_out']['best'][condition]['reward_mean']
				)
				for condition in ('clean', 'hard')
			},
		},
		'decision_rule': (
			'This paired single-seed screen tests whether vision adds value at a '
			'longer budget. Any paper claim still requires at least three seeds and '
			'a matched pixel baseline.'
		),
	}
	output = Path(args.output).expanduser().resolve()
	output.parent.mkdir(parents=True, exist_ok=True)
	temporary = output.with_name(output.name + '.incomplete')
	temporary.write_text(
		json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
		encoding='utf-8', newline='\n',
	)
	os.replace(temporary, output)
	print('ACROBOT_FACTORIZED_PROPRIO_500K_AGGREGATE_OK')
	print(f'SUMMARY={output}')


if __name__ == '__main__':
	main()
