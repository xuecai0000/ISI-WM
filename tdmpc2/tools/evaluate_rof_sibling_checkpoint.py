"""Direct held-out evaluation of a trained ROF action-sibling checkpoint.

This evaluator does not fit a probe.  It encodes a root observation, rolls the
checkpoint's own ``model.next`` forward under every genuinely executed action
branch, and compares each prediction with the matching and all non-matching
real sibling futures.  Oracle arrays in the diagnostic source are used only by
the producing collector's dataset-integrity validator; the explicit packet
allow-list below prevents them from reaching the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np

from tdmpc2.tools import collect_rof_real_action_branches as collector
from tdmpc2.tools import evaluate_rof_real_action_branches as branches
from tdmpc2.tools.collect_rof_six_task_real_action_branches import TASKS


FORMAT = 'rof_sibling_checkpoint_evaluation_v1'
STATUS = 'rof_sibling_checkpoint_evaluation_complete'
MODEL_PATH = 'encoder_then_action_conditioned_model_next'
POLICY_FIELDS = ('rgb', 'object', 'object_mask', 'role_exists')


def _require(condition, message):
	if not condition:
		raise ValueError(message)


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as stream:
		for block in iter(lambda: stream.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def _read_json(path: Path) -> dict:
	payload = json.loads(path.read_text(encoding='utf-8'))
	_require(isinstance(payload, dict), f'Expected JSON object: {path}')
	return payload


def _model_packet(group, condition: str, phase: str):
	_require(condition in ('clean', 'hard'), 'Unknown sibling condition.')
	_require(phase in ('history', 'future'), 'Unknown sibling phase.')
	return {
		field: group.arrays[f'{condition}__{phase}__policy_{field}']
		for field in POLICY_FIELDS
	}


def _encode(agent, arrays):
	"""Encode only public policy tensors using the exact identity crop."""
	import torch

	_require(set(arrays) == set(POLICY_FIELDS), 'Model packet allow-list changed.')
	leading = np.asarray(arrays['rgb']).shape[:-3]
	count = int(np.prod(leading)) if leading else 1
	flat = {
		field: np.ascontiguousarray(value).reshape(
			(count,) + np.asarray(value).shape[len(leading):]
		)
		for field, value in arrays.items()
	}
	packet = {
		field: torch.as_tensor(value, device=agent.device)
		for field, value in flat.items()
	}
	encoder = agent.model._encoder['object']
	pad = int(getattr(encoder.augmentation, 'pad', 3))
	# JointFieldShiftAug's identity crop is (pad,pad), not (0,0).
	shift = torch.full(
		(count, 1, 1, 2), float(pad), device=agent.device,
		dtype=torch.float32,
	)
	with torch.no_grad():
		value = encoder(packet, shift_index=shift)
	return value.reshape(leading + (value.shape[-1],))


def _evaluate_root(agent, group, *, condition: str, horizon: int) -> dict:
	import torch

	history = _model_packet(group, condition, 'history')
	root_packet = {
		field: np.ascontiguousarray(value[-1:])
		for field, value in history.items()
	}
	future_packet = {
		field: np.ascontiguousarray(value[:, :horizon])
		for field, value in _model_packet(group, condition, 'future').items()
	}
	root_z = _encode(agent, root_packet)
	target = _encode(agent, future_packet)
	branches_count = int(target.shape[0])
	_require(branches_count >= 3, 'A sibling root must contain >=3 actions.')
	actions = torch.as_tensor(
		np.ascontiguousarray(group.arrays['branch__action'][:, :horizon]),
		device=agent.device, dtype=torch.float32,
	)
	_require(
		tuple(actions.shape) == (branches_count, horizon, group.action_dim),
		'Sibling action tensor has an invalid shape.',
	)
	z = root_z.expand(branches_count, -1).contiguous()
	predicted = []
	with torch.no_grad():
		for index in range(horizon):
			z = agent.model.next(z, actions[:, index], None)
			predicted.append(z)
	predicted = torch.stack(predicted, dim=1)
	_require(predicted.shape == target.shape, 'Predicted/target latent shapes differ.')
	# error[predicted action branch, real target branch, horizon]
	error = (
		predicted[:, None] - target[None]
	).square().mean(dim=-1)
	identity = torch.eye(branches_count, dtype=torch.bool, device=agent.device)
	correct_by_frame = error.diagonal(dim1=0, dim2=1).transpose(0, 1)
	wrong = error[~identity].reshape(
		branches_count, branches_count - 1, horizon
	)
	correct = correct_by_frame.mean()
	wrong_mean = wrong.mean()
	ranking = (correct_by_frame[:, None] < wrong).to(torch.float32).mean()
	correct_value = float(correct.detach().cpu())
	wrong_value = float(wrong_mean.detach().cpu())
	separation = 1.0 - correct_value / wrong_value if wrong_value > 0 else float('-inf')
	values = (correct_value, wrong_value, float(ranking.detach().cpu()), separation)
	_require(all(math.isfinite(value) for value in values), 'Nonfinite sibling metric.')
	return {
		'root_id': int(group.root_id),
		'branch_count': branches_count,
		'correct_action_mse': correct_value,
		'wrong_sibling_mse': wrong_value,
		'action_separation': separation,
		'ranking_accuracy': float(ranking.detach().cpu()),
	}


def evaluate(args) -> dict:
	import torch

	for path in (args.runtime_config, args.checkpoint, args.dataset_manifest):
		path = path.resolve()
		_require(path.is_file() and not path.is_symlink(), f'Missing/symlink input: {path}')
	_require(args.output.resolve() == args.output.absolute(), 'Output path must resolve cleanly.')
	_require(not args.output.exists(), f'Refusing to overwrite {args.output}.')
	_require(args.task in TASKS, 'Task is outside the frozen six-task screen.')
	_require(args.split == 'test' and args.condition == 'clean',
		'This screen evaluates clean held-out test roots only.')
	_require(args.disable_random_shift is True,
		'--disable-random-shift is mandatory for paired checkpoint scoring.')
	_require(args.horizon == 3, 'The registered direct horizon is three.')
	raw = _read_json(args.runtime_config.resolve())
	_require(raw.get('task') == args.task, 'Runtime task mismatch.')
	_require(raw.get('steps') == args.expected_training_step == 30000,
		'Checkpoint training-step contract mismatch.')
	_require(raw.get('compile') is False, 'The six-task screen is eager-only.')
	expected_checkpoint = args.runtime_config.resolve().parent / 'models' / 'final.pt'
	_require(args.checkpoint.resolve() == expected_checkpoint.resolve(),
		'Only the runtime-paired final checkpoint is accepted.')

	collector.TASKS = TASKS
	dataset = branches.load_branch_dataset(
		args.dataset_manifest.resolve(), require_collector_contract=True,
	)
	_require(dataset.task == args.task, 'Sibling dataset task mismatch.')
	_require(tuple(raw.get('cutie_object_role_names', ())) == dataset.role_names,
		'Runtime and sibling dataset roles differ.')
	groups = dataset.split(args.split)
	_require(len(groups) == 10, 'Exactly ten complete held-out roots are required.')
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.set_float32_matmul_precision('high')
	agent = branches._load_agent(
		dataset, args.runtime_config.resolve(), args.checkpoint.resolve(),
	)
	rows = [
		_evaluate_root(agent, group, condition=args.condition, horizon=args.horizon)
		for group in groups
	]
	correct = sum(row['correct_action_mse'] for row in rows) / len(rows)
	wrong = sum(row['wrong_sibling_mse'] for row in rows) / len(rows)
	ranking = sum(row['ranking_accuracy'] for row in rows) / len(rows)
	separation = 1.0 - correct / wrong if wrong > 0 else float('-inf')
	_require(all(math.isfinite(value) for value in (correct, wrong, ranking, separation)),
		'Aggregate sibling metrics are nonfinite.')
	manifest = _read_json(args.dataset_manifest.resolve())
	return {
		'format': FORMAT,
		'status': STATUS,
		'task': args.task,
		'split': args.split,
		'condition': args.condition,
		'root_count': len(rows),
		'horizon': args.horizon,
		'model_path': MODEL_PATH,
		'random_shift': 'disabled',
		'identity_shift_index': int(agent.model._encoder['object'].augmentation.pad),
		'oracle_model_input': False,
		'metrics': {
			'correct_action_mse': correct,
			'wrong_sibling_mse': wrong,
			'action_separation': separation,
			'ranking_accuracy': ranking,
		},
		'per_root': rows,
		'provenance': {
			'runtime_config': str(args.runtime_config.resolve()),
			'runtime_config_sha256': _sha256(args.runtime_config.resolve()),
			'checkpoint': str(args.checkpoint.resolve()),
			'checkpoint_sha256': _sha256(args.checkpoint.resolve()),
			'checkpoint_step': args.expected_training_step,
			'dataset_manifest': str(args.dataset_manifest.resolve()),
			'dataset_manifest_sha256': _sha256(args.dataset_manifest.resolve()),
			'dataset_behavior_policy_checkpoint_sha256': manifest['source'][
				'checkpoint_sha256'
			],
			'evaluator_sha256': _sha256(Path(__file__).resolve()),
		},
	}


def _atomic_write(path: Path, payload: dict) -> None:
	path = path.resolve()
	path.parent.mkdir(parents=True, exist_ok=True)
	_require(not path.exists(), f'Refusing to overwrite {path}.')
	temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
	try:
		temporary.write_text(
			json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
			encoding='utf-8',
		)
		os.replace(temporary, path)
	finally:
		if temporary.exists():
			temporary.unlink()


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--task', choices=TASKS, required=True)
	parser.add_argument('--runtime-config', type=Path, required=True)
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--dataset-manifest', type=Path, required=True)
	parser.add_argument('--split', choices=('test',), default='test')
	parser.add_argument('--condition', choices=('clean',), default='clean')
	parser.add_argument('--expected-training-step', type=int, default=30000)
	parser.add_argument('--horizon', type=int, default=3)
	parser.add_argument('--disable-random-shift', action='store_true')
	parser.add_argument('--output', type=Path, required=True)
	return parser.parse_args(argv)


def main(argv=None):
	args = parse_args(argv)
	payload = evaluate(args)
	_atomic_write(args.output, payload)
	print('ROF_SIBLING_CHECKPOINT_EVALUATION_COMPLETE', json.dumps({
		'task': payload['task'], 'root_count': payload['root_count'],
		'output': str(args.output.resolve()),
	}, allow_nan=False), flush=True)
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
