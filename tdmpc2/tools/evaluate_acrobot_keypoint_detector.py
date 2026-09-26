"""Held-out long-sequence gate for the Acrobot keypoint detector."""

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

from tdmpc2.perception.acrobot_keypoint_data import (  # noqa: E402
	AcrobotKeypointSequenceDataset, angular_error, load_manifest,
)
from tdmpc2.perception.acrobot_keypoint_detector import load_checkpoint  # noqa: E402


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
	dataset = AcrobotKeypointSequenceDataset(
		args.test_manifest, history=model.config.history, conditions=('clean', 'hard'),
	)
	training_metadata = checkpoint.get('training', {})
	training_files = set(training_metadata.get('train_episode_sha256', ()))
	training_files.update(training_metadata.get('validation_episode_sha256', ()))
	training_trajectories = {
		tuple(value) for name in ('train_trajectories', 'validation_trajectories')
		for value in training_metadata.get(name, ())
	}
	for key in ('train_manifest', 'validation_manifest'):
		path = training_metadata.get(key)
		if path and Path(path).is_file():
			_, manifest = load_manifest(path)
			training_files.update(record['sha256'] for record in manifest['episodes'])
			training_trajectories.update(
				(record.get('env_seed'), record.get('action_seed'), record.get('pose_seed'))
				for record in manifest['episodes']
			)
	test_files = {record['sha256'] for record in dataset.manifest['episodes']}
	if training_files & test_files:
		raise ValueError('Held-out test episodes overlap training/validation archives.')
	test_trajectories = {
		(record.get('env_seed'), record.get('action_seed'), record.get('pose_seed'))
		for record in dataset.manifest['episodes']
	}
	if training_trajectories & test_trajectories:
		raise ValueError('Held-out test simulator trajectories overlap training/validation.')
	loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
	records = []
	for batch in loader:
		rgb = batch['rgb'].to(device)
		actions = batch['actions'].to(device)
		target = batch['world_xz'].to(device)
		output = model(rgb, actions)
		world_error = torch.linalg.vector_norm(output['world_xz'] - target, dim=-1) * 2.
		angle = angular_error(output['world_xz'], target) * 180. / np.pi
		omega_error = (
			output['angular_velocity'] - batch['global_omega'].to(device)
		).abs()
		image_xy = (output['image_xy'] + 1.) * (model.config.image_size - 1.) / 2.
		pixel_error = torch.linalg.vector_norm(
			image_xy - batch['pixel_xy'].to(device), dim=-1,
		)
		for row in range(rgb.shape[0]):
			records.append({
				'episode': int(batch['episode_index'][row]),
				'frame': int(batch['frame_index'][row]),
				'condition': 'clean' if int(batch['condition_index'][row]) == 0 else 'hard',
				'world_error': world_error[row].cpu().numpy(),
				'pixel_error': pixel_error[row].cpu().numpy(),
				'angle_error': angle[row].cpu().numpy(),
				'omega_error': omega_error[row].cpu().numpy(),
				'confidence': output['confidence'][row].cpu().numpy(),
			})

	latencies = []
	sample = dataset[0]
	rgb = sample['rgb'][None].to(device)
	actions = sample['actions'][None].to(device)
	for index in range(args.latency_frames + 20):
		if device.type == 'cuda':
			torch.cuda.synchronize(device)
		started = time.perf_counter()
		model(rgb, actions)
		if device.type == 'cuda':
			torch.cuda.synchronize(device)
		if index >= 20:
			latencies.append((time.perf_counter() - started) * 1000.)

	conditions = {}
	all_pass = True
	for condition in ('clean', 'hard'):
		selected = [record for record in records if record['condition'] == condition]
		world = np.stack([record['world_error'] for record in selected])
		pixels = np.stack([record['pixel_error'] for record in selected])
		angles = np.stack([record['angle_error'] for record in selected])
		omega = np.stack([record['omega_error'] for record in selected])
		confidence = np.stack([record['confidence'] for record in selected])
		episode_bursts = []
		for episode in sorted({record['episode'] for record in selected}):
			sequence = sorted(
				(record for record in selected if record['episode'] == episode),
				key=lambda record: record['frame'],
			)
			failed = [bool((record['world_error'] > args.failure_threshold_world).any()) for record in sequence]
			episode_bursts.append(_longest_burst(failed))
		metrics = {
			'mean_point_error_world': float(world.mean()),
			'p90_point_error_world': float(np.percentile(world, 90)),
			'pck_world_at_threshold': float((world <= args.failure_threshold_world).mean()),
			'pck_pixel_at_3': float((pixels <= 3.).mean()),
			'mean_link_angle_error_deg': float(angles.mean()),
			'p90_link_angle_error_deg': float(np.percentile(angles, 90)),
			'mean_angular_velocity_error': float(omega.mean()),
			'p90_angular_velocity_error': float(np.percentile(omega, 90)),
			'mean_confidence': float(confidence.mean()),
			'max_consecutive_failure_frames': int(max(episode_bursts)),
			'episode_p95_failure_burst': float(np.percentile(episode_bursts, 95)),
		}
		checks = {
			'pck_world': metrics['pck_world_at_threshold'] >= args.minimum_pck,
			'p90_world_error': metrics['p90_point_error_world'] <= args.maximum_p90_world,
			'p90_angular_velocity_error': (
				metrics['p90_angular_velocity_error'] <= args.maximum_p90_omega
			),
			'max_failure_burst': metrics['max_consecutive_failure_frames'] <= args.maximum_burst,
			'episode_p95_failure_burst': metrics['episode_p95_failure_burst'] <= args.maximum_p95_burst,
		}
		conditions[condition] = {'metrics': metrics, 'checks': checks, 'pass': all(checks.values())}
		all_pass = all_pass and conditions[condition]['pass']
	latency = {
		'mean_ms': float(statistics.mean(latencies)),
		'p95_ms': float(np.percentile(latencies, 95)),
		'frames': len(latencies),
	}
	latency_pass = latency['p95_ms'] <= args.maximum_latency_p95_ms
	artifact_root = args.output.resolve().parent
	def artifact_path(path):
		path = Path(path).resolve()
		try:
			return path.relative_to(artifact_root).as_posix()
		except ValueError:
			return str(path)
	result = {
		'format': 'acrobot_keypoint_long_sequence_evaluation_v1',
		'status': 'pass' if all_pass and latency_pass else 'no_go',
		'engineering_pass': True,
		'controller_pilot_authorized': bool(all_pass and latency_pass),
		'checkpoint': artifact_path(args.checkpoint),
		'checkpoint_sha256': _sha256(args.checkpoint),
		'test_manifest': artifact_path(args.test_manifest),
		'test_episode_sha256_overlap_with_training': False,
		'conditions': conditions,
		'latency': latency,
		'latency_pass': latency_pass,
		'gates': {
			'failure_threshold_world': args.failure_threshold_world,
			'minimum_pck': args.minimum_pck,
			'maximum_p90_world': args.maximum_p90_world,
			'maximum_p90_omega': args.maximum_p90_omega,
			'maximum_burst': args.maximum_burst,
			'maximum_p95_burst': args.maximum_p95_burst,
			'maximum_latency_p95_ms': args.maximum_latency_p95_ms,
		},
		'recommendation': (
			'run_20k_controller_pilot' if all_pass and latency_pass
			else 'do_not_train_controller_improve_keypoint_detector'
		),
	}
	args.output.parent.mkdir(parents=True, exist_ok=True)
	if args.output.exists():
		raise FileExistsError(f'Refusing to overwrite evaluation: {args.output}')
	args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
	print(json.dumps(result, indent=2), flush=True)
	return result


def parse_args(argv=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--test-manifest', type=Path, required=True)
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--device', default='cuda:0')
	parser.add_argument('--batch-size', type=int, default=256)
	parser.add_argument('--workers', type=int, default=4)
	parser.add_argument('--latency-frames', type=int, default=500)
	parser.add_argument('--failure-threshold-world', type=float, default=0.05)
	parser.add_argument('--minimum-pck', type=float, default=0.98)
	parser.add_argument('--maximum-p90-world', type=float, default=0.05)
	parser.add_argument('--maximum-p90-omega', type=float, default=1.0)
	parser.add_argument('--maximum-burst', type=int, default=5)
	parser.add_argument('--maximum-p95-burst', type=float, default=3.)
	parser.add_argument('--maximum-latency-p95-ms', type=float, default=10.)
	return parser.parse_args(argv)


if __name__ == '__main__':
	evaluate(parse_args())
