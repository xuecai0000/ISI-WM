"""Held-out causal long-sequence gate for the action-conditioned observer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))

from tdmpc2.perception.acrobot_state_observer import (  # noqa: E402
	load_checkpoint, observer_inputs, pose_angles,
)
from tdmpc2.perception.acrobot_state_observer_data import AcrobotObserverDataset  # noqa: E402


def _sha256(path):
	digest = hashlib.sha256()
	with Path(path).open('rb') as stream:
		for chunk in iter(lambda: stream.read(1024 * 1024), b''):
			digest.update(chunk)
	return digest.hexdigest()


def _longest_burst(values):
	longest = current = 0
	for value in values:
		current = current + 1 if value else 0
		longest = max(longest, current)
	return longest


@torch.inference_mode()
def evaluate(args):
	device = torch.device(args.device)
	model, checkpoint = load_checkpoint(args.checkpoint, device=device)
	dataset = AcrobotObserverDataset(args.test_manifest)
	training_files = set(checkpoint.get('training', {}).get('train_episode_sha256', ()))
	training_files.update(checkpoint.get('training', {}).get('validation_episode_sha256', ()))
	test_files = {record['sha256'] for record in dataset.manifest['episodes']}
	if training_files & test_files:
		raise ValueError('Observer held-out traces overlap training/validation.')
	loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)
	records = {condition: [] for condition in ('clean', 'hard')}
	for batch in loader:
		measured = batch['measured_points'].to(device)
		confidence = batch['confidence'].to(device)
		valid = batch['mask_valid'].to(device)
		actions = batch['previous_action'].to(device)
		target_points = batch['target_points'].to(device)
		target_omega = batch['target_omega'].to(device)
		inputs = observer_inputs(
			measured, confidence, valid, actions, dt=model.config.control_dt,
		)
		output, _ = model(inputs)
		point_error = torch.linalg.vector_norm(
			output['points'] - target_points, dim=-1,
		) * 2.
		predicted_angle = pose_angles(output['points'])
		target_angle = pose_angles(target_points)
		angle_error = torch.atan2(
			torch.sin(predicted_angle - target_angle),
			torch.cos(predicted_angle - target_angle),
		).abs() * 180. / np.pi
		omega_error = (output['angular_velocity'] - target_omega).abs()
		condition = 'clean' if int(batch['condition_index'][0]) == 0 else 'hard'
		records[condition].append({
			'episode': int(batch['episode_index'][0]),
			'point_error': point_error[0].cpu().numpy(),
			'angle_error': angle_error[0].cpu().numpy(),
			'omega_error': omega_error[0].cpu().numpy(),
		})

	conditions, all_pass = {}, True
	for condition in ('clean', 'hard'):
		point = np.concatenate([record['point_error'] for record in records[condition]])
		angle = np.concatenate([record['angle_error'] for record in records[condition]])
		omega = np.concatenate([record['omega_error'] for record in records[condition]])
		bursts = []
		for record in records[condition]:
			bursts.append(_longest_burst(
				(record['point_error'] > args.failure_threshold_world).any(axis=-1)
			))
		metrics = {
			'mean_point_error_world': float(point.mean()),
			'p90_point_error_world': float(np.percentile(point, 90)),
			'pck_world_at_threshold': float((point <= args.failure_threshold_world).mean()),
			'mean_link_angle_error_deg': float(angle.mean()),
			'p90_link_angle_error_deg': float(np.percentile(angle, 90)),
			'mean_angular_velocity_error': float(omega.mean()),
			'p90_angular_velocity_error': float(np.percentile(omega, 90)),
			'max_consecutive_failure_frames': int(max(bursts)),
			'episode_p95_failure_burst': float(np.percentile(bursts, 95)),
		}
		checks = {
			'pck_world': metrics['pck_world_at_threshold'] >= args.minimum_pck,
			'p90_world_error': metrics['p90_point_error_world'] <= args.maximum_p90_world,
			'p90_angular_velocity_error': (
				metrics['p90_angular_velocity_error'] <= args.maximum_p90_omega
			),
			'max_failure_burst': metrics['max_consecutive_failure_frames'] <= args.maximum_burst,
			'episode_p95_failure_burst': (
				metrics['episode_p95_failure_burst'] <= args.maximum_p95_burst
			),
		}
		conditions[condition] = {'metrics': metrics, 'checks': checks, 'pass': all(checks.values())}
		all_pass = all_pass and conditions[condition]['pass']

	sample = dataset[0]
	measured = sample['measured_points'][None].to(device)
	confidence = sample['confidence'][None].to(device)
	valid = sample['mask_valid'][None].to(device)
	actions = sample['previous_action'][None].to(device)
	inputs = observer_inputs(measured, confidence, valid, actions, dt=model.config.control_dt)
	latencies, hidden = [], None
	for index in range(args.latency_frames + 20):
		frame_input = inputs[:, index % inputs.shape[1]]
		if device.type == 'cuda':
			torch.cuda.synchronize(device)
		started = time.perf_counter()
		_, hidden = model.step(frame_input, hidden)
		if device.type == 'cuda':
			torch.cuda.synchronize(device)
		if index >= 20:
			latencies.append((time.perf_counter() - started) * 1000.)
	observer_latency = {
		'mean_ms': float(statistics.mean(latencies)),
		'p95_ms': float(np.percentile(latencies, 95)),
		'frames': len(latencies),
	}
	end_to_end_p95 = args.frontend_p95_ms + observer_latency['p95_ms']
	latency_pass = end_to_end_p95 <= args.maximum_end_to_end_p95_ms
	passed = all_pass and latency_pass
	result = {
		'format': 'acrobot_causal_state_observer_evaluation_v1',
		'status': 'pass' if passed else 'no_go',
		'engineering_pass': True,
		'controller_pilot_authorized': bool(passed),
		'checkpoint': str(args.checkpoint.resolve()),
		'checkpoint_sha256': _sha256(args.checkpoint),
		'test_manifest': str(args.test_manifest.resolve()),
		'conditions': conditions,
		'observer_latency': observer_latency,
		'frontend_p95_ms': args.frontend_p95_ms,
		'estimated_end_to_end_p95_ms': float(end_to_end_p95),
		'latency_pass': latency_pass,
		'gates': {
			'failure_threshold_world': args.failure_threshold_world,
			'minimum_pck': args.minimum_pck,
			'maximum_p90_world': args.maximum_p90_world,
			'maximum_p90_omega': args.maximum_p90_omega,
			'maximum_burst': args.maximum_burst,
			'maximum_p95_burst': args.maximum_p95_burst,
			'maximum_end_to_end_p95_ms': args.maximum_end_to_end_p95_ms,
		},
		'recommendation': (
			'run_20k_controller_pilot' if passed
			else 'do_not_train_controller_state_observer_failed_long_sequence_gate'
		),
	}
	args.output.parent.mkdir(parents=True, exist_ok=True)
	if args.output.exists():
		raise FileExistsError(f'Refusing to overwrite observer evaluation: {args.output}')
	args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
	print(json.dumps(result, indent=2), flush=True)
	return result


def parse_args(argv=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--test-manifest', type=Path, required=True)
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--device', default='cuda:0')
	parser.add_argument('--workers', type=int, default=2)
	parser.add_argument('--latency-frames', type=int, default=1000)
	parser.add_argument('--frontend-p95-ms', type=float, default=7.1)
	parser.add_argument('--failure-threshold-world', type=float, default=0.05)
	parser.add_argument('--minimum-pck', type=float, default=0.98)
	parser.add_argument('--maximum-p90-world', type=float, default=0.05)
	parser.add_argument('--maximum-p90-omega', type=float, default=1.0)
	parser.add_argument('--maximum-burst', type=int, default=5)
	parser.add_argument('--maximum-p95-burst', type=float, default=3.)
	parser.add_argument('--maximum-end-to-end-p95-ms', type=float, default=15.)
	return parser.parse_args(argv)


if __name__ == '__main__':
	evaluate(parse_args())
