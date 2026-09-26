"""Held-out long-sequence gate for Cutie-masked Acrobot keypoints."""

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

from tdmpc2.perception.acrobot_keypoint_data import angular_error  # noqa: E402
from tdmpc2.perception.acrobot_masked_keypoint_data import (  # noqa: E402
	AcrobotPairedMaskedKeypointDataset,
)
from tdmpc2.perception.acrobot_masked_keypoint_detector import load_checkpoint  # noqa: E402
from tdmpc2.perception.acrobot_pose_filter import CausalAcrobotPoseFilter  # noqa: E402


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


def _angle_error(predicted, target):
	predicted = torch.as_tensor(predicted[None], dtype=torch.float32)
	target = torch.as_tensor(target[None], dtype=torch.float32)
	return angular_error(predicted, target)[0].numpy() * 180. / np.pi


@torch.inference_mode()
def evaluate(args):
	device = torch.device(args.device)
	model, checkpoint = load_checkpoint(args.checkpoint, device=device)
	dataset = AcrobotPairedMaskedKeypointDataset(
		args.test_manifest, args.test_masks, history=model.config.history,
	)
	training_files = set(checkpoint.get('training', {}).get('train_episode_sha256', ()))
	training_files.update(checkpoint.get('training', {}).get('validation_episode_sha256', ()))
	test_files = {record['sha256'] for record in dataset.raw_manifest['episodes']}
	if training_files & test_files:
		raise ValueError('Held-out test archives overlap training/validation.')
	loader = DataLoader(
		dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
	)
	raw_records = {condition: [] for condition in ('clean', 'hard')}
	for batch in loader:
		actions = batch['actions'].to(device)
		for condition in ('clean', 'hard'):
			output = model(
				batch[f'rgb_{condition}'].to(device), actions,
				batch[f'mask_{condition}'].to(device),
			)
			image_xy = (output['image_xy'] + 1.) * (model.config.image_size - 1.) / 2.
			for row in range(actions.shape[0]):
				raw_records[condition].append({
					'episode': int(batch['episode_index'][row]),
					'frame': int(batch['frame_index'][row]),
					'points': output['world_xz'][row].cpu().numpy(),
					'image_xy': image_xy[row].cpu().numpy(),
					'confidence': output['confidence'][row].cpu().numpy(),
					'target': batch['world_xz'][row].numpy(),
					'target_pixel': batch['pixel_xy'][row].numpy(),
					'target_omega': batch['global_omega'][row].numpy(),
					'mask_valid': bool(batch[f'mask_valid_{condition}'][row]),
					'cutie_runtime_ms': float(batch[f'cutie_runtime_ms_{condition}'][row]),
				})

	conditions = {}
	all_pass = True
	filter_runtime = []
	for condition in ('clean', 'hard'):
		filtered_records = []
		for episode in sorted({record['episode'] for record in raw_records[condition]}):
			pose_filter = CausalAcrobotPoseFilter(
				dt=args.control_dt, alpha=args.filter_alpha,
				window=args.filter_window,
				minimum_confidence=args.minimum_confidence,
			)
			for record in sorted(
				(value for value in raw_records[condition] if value['episode'] == episode),
				key=lambda value: value['frame'],
			):
				started = time.perf_counter()
				points, omega = pose_filter.update(record['points'], record['confidence'])
				filter_runtime.append((time.perf_counter() - started) * 1000.)
				world_error = np.linalg.norm(points - record['target'], axis=-1) * 2.
				filtered_records.append({
					**record,
					'filtered_points': points,
					'world_error': world_error,
					'pixel_error': np.linalg.norm(
						record['image_xy'] - record['target_pixel'], axis=-1,
					),
					'angle_error': _angle_error(points, record['target']),
					'omega_error': np.abs(omega - record['target_omega']),
				})
		world = np.stack([record['world_error'] for record in filtered_records])
		pixels = np.stack([record['pixel_error'] for record in filtered_records])
		angles = np.stack([record['angle_error'] for record in filtered_records])
		omega = np.stack([record['omega_error'] for record in filtered_records])
		confidence = np.stack([record['confidence'] for record in filtered_records])
		cutie_runtime = np.asarray([
			record['cutie_runtime_ms'] for record in filtered_records
		], dtype=np.float64)
		episode_bursts = []
		for episode in sorted({record['episode'] for record in filtered_records}):
			sequence = [
				record for record in filtered_records if record['episode'] == episode
			]
			episode_bursts.append(_longest_burst([
				bool((record['world_error'] > args.failure_threshold_world).any())
				for record in sequence
			]))
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
			'mask_valid_rate': float(np.mean([r['mask_valid'] for r in filtered_records])),
			'cutie_mean_ms': float(cutie_runtime.mean()),
			'cutie_p95_ms': float(np.percentile(cutie_runtime, 95)),
			'max_consecutive_failure_frames': int(max(episode_bursts)),
			'episode_p95_failure_burst': float(np.percentile(episode_bursts, 95)),
		}
		checks = {
			'fallback_rate': (1. - metrics['mask_valid_rate']) <= args.maximum_fallback_rate,
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

	latencies = []
	sample = dataset[0]
	rgb = sample['rgb_hard'][None].to(device)
	actions = sample['actions'][None].to(device)
	mask = sample['mask_hard'][None].to(device)
	for index in range(args.latency_frames + 20):
		if device.type == 'cuda':
			torch.cuda.synchronize(device)
		started = time.perf_counter()
		model(rgb, actions, mask)
		if device.type == 'cuda':
			torch.cuda.synchronize(device)
		if index >= 20:
			latencies.append((time.perf_counter() - started) * 1000.)
	keypoint_latency = {
		'mean_ms': float(statistics.mean(latencies)),
		'p95_ms': float(np.percentile(latencies, 95)),
		'filter_mean_ms': float(statistics.mean(filter_runtime)),
		'filter_p95_ms': float(np.percentile(filter_runtime, 95)),
	}
	end_to_end_p95 = max(
		conditions[name]['metrics']['cutie_p95_ms'] for name in conditions
	) + keypoint_latency['p95_ms'] + keypoint_latency['filter_p95_ms']
	latency_pass = end_to_end_p95 <= args.maximum_end_to_end_p95_ms
	status_pass = all_pass and latency_pass
	result = {
		'format': 'acrobot_masked_keypoint_long_sequence_evaluation_v2',
		'status': 'pass' if status_pass else 'no_go',
		'engineering_pass': True,
		'controller_pilot_authorized': bool(status_pass),
		'checkpoint': str(args.checkpoint.resolve()),
		'checkpoint_sha256': _sha256(args.checkpoint),
		'test_manifest': str(args.test_manifest.resolve()),
		'test_masks': str(args.test_masks.resolve()),
		'conditions': conditions,
		'keypoint_latency': keypoint_latency,
		'estimated_end_to_end_p95_ms': float(end_to_end_p95),
		'latency_pass': latency_pass,
		'gates': vars(args) | {},
		'recommendation': (
			'run_20k_controller_pilot' if status_pass
			else 'do_not_train_controller_masked_keypoint_failed_long_sequence_gate'
		),
	}
	# argparse Paths/devices are not JSON values; expose only scalar gate settings.
	result['gates'] = {
		'failure_threshold_world': args.failure_threshold_world,
		'maximum_fallback_rate': args.maximum_fallback_rate,
		'minimum_pck': args.minimum_pck,
		'maximum_p90_world': args.maximum_p90_world,
		'maximum_p90_omega': args.maximum_p90_omega,
		'maximum_burst': args.maximum_burst,
		'maximum_p95_burst': args.maximum_p95_burst,
		'maximum_end_to_end_p95_ms': args.maximum_end_to_end_p95_ms,
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
	parser.add_argument('--test-masks', type=Path, required=True)
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--device', default='cuda:0')
	parser.add_argument('--batch-size', type=int, default=256)
	parser.add_argument('--workers', type=int, default=4)
	parser.add_argument('--latency-frames', type=int, default=500)
	parser.add_argument('--control-dt', type=float, default=0.04)
	parser.add_argument('--filter-alpha', type=float, default=1.0)
	parser.add_argument('--filter-window', type=int, default=5)
	parser.add_argument('--minimum-confidence', type=float, default=0.15)
	parser.add_argument('--failure-threshold-world', type=float, default=0.05)
	parser.add_argument('--maximum-fallback-rate', type=float, default=0.20)
	parser.add_argument('--minimum-pck', type=float, default=0.98)
	parser.add_argument('--maximum-p90-world', type=float, default=0.05)
	parser.add_argument('--maximum-p90-omega', type=float, default=1.0)
	parser.add_argument('--maximum-burst', type=int, default=5)
	parser.add_argument('--maximum-p95-burst', type=float, default=3.)
	parser.add_argument('--maximum-end-to-end-p95-ms', type=float, default=15.)
	return parser.parse_args(argv)


if __name__ == '__main__':
	evaluate(parse_args())
