"""Train a causal action-conditioned observer on frozen keypoint traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))

from tdmpc2.perception.acrobot_state_observer import (  # noqa: E402
	AcrobotStateObserver, AcrobotStateObserverConfig, observer_inputs,
	pose_angles, save_checkpoint,
)
from tdmpc2.perception.acrobot_state_observer_data import AcrobotObserverDataset  # noqa: E402


def _sha256(path):
	digest = hashlib.sha256()
	with Path(path).open('rb') as stream:
		for chunk in iter(lambda: stream.read(1024 * 1024), b''):
			digest.update(chunk)
	return digest.hexdigest()


def _augment_bursts(points, confidence, valid, *, maximum, probability, generator):
	points = points.clone()
	confidence = confidence.clone()
	valid = valid.clone()
	batch, frames = points.shape[:2]
	for row in range(batch):
		if float(torch.rand((), generator=generator)) > probability:
			continue
		bursts = int(torch.randint(1, 4, (), generator=generator))
		for _ in range(bursts):
			length = int(torch.randint(4, maximum + 1, (), generator=generator))
			start = int(torch.randint(1, max(2, frames - length + 1), (), generator=generator))
			end = min(frames, start + length)
			points[row, start:end] = points[row, start - 1]
			confidence[row, start:end] = 0.
			valid[row, start:end] = False
	return points, confidence, valid


def _loss(model, batch, device, *, augment=False, args=None, generator=None):
	measured = batch['measured_points'].to(device)
	confidence = batch['confidence'].to(device)
	valid = batch['mask_valid'].to(device)
	actions = batch['previous_action'].to(device)
	target_points = batch['target_points'].to(device)
	target_omega = batch['target_omega'].to(device)
	if augment:
		measured, confidence, valid = _augment_bursts(
			measured, confidence, valid,
			maximum=args.maximum_dropout_burst,
			probability=args.dropout_probability,
			generator=generator,
		)
	inputs = observer_inputs(
		measured, confidence, valid, actions, dt=model.config.control_dt,
	)
	output, _ = model(inputs)
	target_angles = pose_angles(target_points)
	target_vectors = torch.stack((torch.sin(target_angles), torch.cos(target_angles)), dim=-1)
	point = F.smooth_l1_loss(output['points'], target_points, beta=0.02)
	angle = F.smooth_l1_loss(output['angle_vectors'], target_vectors, beta=0.02)
	omega = F.smooth_l1_loss(
		output['angular_velocity'] / model.config.omega_scale,
		target_omega / model.config.omega_scale,
		beta=0.05,
	)
	# Encourage coherent causal evolution without accessing future inputs.
	predicted_delta = output['points'][:, 1:] - output['points'][:, :-1]
	target_delta = target_points[:, 1:] - target_points[:, :-1]
	dynamics = F.smooth_l1_loss(predicted_delta, target_delta, beta=0.01)
	total = point + 0.5 * angle + 0.5 * omega + 0.2 * dynamics
	point_error = torch.linalg.vector_norm(output['points'] - target_points, dim=-1) * 2.
	omega_error = (output['angular_velocity'] - target_omega).abs()
	return total, {
		'loss': total,
		'point_loss': point,
		'angle_vector_loss': angle,
		'omega_loss': omega,
		'dynamics_loss': dynamics,
		'mean_point_error_world': point_error.mean(),
		'p90_point_error_world': torch.quantile(point_error, 0.9),
		'mean_omega_error': omega_error.mean(),
		'p90_omega_error': torch.quantile(omega_error, 0.9),
	}


@torch.inference_mode()
def _validate(model, loader, device):
	model.eval()
	totals, count = {}, 0
	for batch in loader:
		_, metrics = _loss(model, batch, device)
		batch_size = batch['measured_points'].shape[0]
		count += batch_size
		for name, value in metrics.items():
			totals[name] = totals.get(name, 0.) + float(value) * batch_size
	return {name: value / count for name, value in totals.items()}


def train(args):
	random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
	device = torch.device(args.device)
	train_data = AcrobotObserverDataset(args.train_manifest)
	validation_data = AcrobotObserverDataset(args.validation_manifest)
	train_files = {record['sha256'] for record in train_data.manifest['episodes']}
	validation_files = {record['sha256'] for record in validation_data.manifest['episodes']}
	if train_files & validation_files:
		raise ValueError('Observer train/validation traces overlap.')
	base_points = np.concatenate([
		value['target_world_xz'][:, 0] for value in train_data._episodes
	])
	base_xz = np.median(base_points, axis=0).astype(np.float32)
	if np.max(np.linalg.norm(base_points - base_xz, axis=-1)) > 1e-5:
		raise ValueError('Acrobot base is not fixed under the declared camera/world contract.')
	config = AcrobotStateObserverConfig(
		hidden_dim=args.hidden_dim, layers=args.layers,
		control_dt=args.control_dt,
	)
	model = AcrobotStateObserver(config, base_xz).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
	train_loader = DataLoader(
		train_data, batch_size=args.batch_size, shuffle=True,
		num_workers=args.workers, pin_memory=device.type == 'cuda',
	)
	validation_loader = DataLoader(
		validation_data, batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=device.type == 'cuda',
	)
	generator = torch.Generator(device='cpu').manual_seed(args.seed + 17)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	best, best_metrics, updates = float('inf'), None, 0
	for epoch in range(args.epochs):
		model.train()
		for batch in train_loader:
			optimizer.zero_grad(set_to_none=True)
			loss, _ = _loss(
				model, batch, device, augment=True, args=args, generator=generator,
			)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
			optimizer.step(); updates += 1
		metrics = _validate(model, validation_loader, device)
		print('ACROBOT_STATE_OBSERVER_EPOCH', json.dumps({
			'epoch': epoch + 1, 'updates': updates, **metrics,
		}), flush=True)
		score = metrics['mean_point_error_world'] + 0.02 * metrics['mean_omega_error']
		if score < best:
			best, best_metrics = score, metrics
			save_checkpoint(args.output, model, training={
				'train_manifest': str(Path(args.train_manifest).resolve()),
				'validation_manifest': str(Path(args.validation_manifest).resolve()),
				'train_manifest_sha256': _sha256(args.train_manifest),
				'validation_manifest_sha256': _sha256(args.validation_manifest),
				'train_episode_sha256': sorted(train_files),
				'validation_episode_sha256': sorted(validation_files),
				'maximum_dropout_burst': args.maximum_dropout_burst,
				'dropout_probability': args.dropout_probability,
				'epochs': epoch + 1, 'updates': updates, 'seed': args.seed,
			}, metrics=metrics)
	print(f'CHECKPOINT={args.output.resolve()}')
	print('BEST_VALIDATION', json.dumps(best_metrics), flush=True)


def parse_args(argv=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--train-manifest', type=Path, required=True)
	parser.add_argument('--validation-manifest', type=Path, required=True)
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--device', default='cuda:0')
	parser.add_argument('--epochs', type=int, default=80)
	parser.add_argument('--batch-size', type=int, default=8)
	parser.add_argument('--workers', type=int, default=2)
	parser.add_argument('--hidden-dim', type=int, default=96)
	parser.add_argument('--layers', type=int, default=1)
	parser.add_argument('--control-dt', type=float, default=0.04)
	parser.add_argument('--learning-rate', type=float, default=5e-4)
	parser.add_argument('--maximum-dropout-burst', type=int, default=96)
	parser.add_argument('--dropout-probability', type=float, default=0.8)
	parser.add_argument('--seed', type=int, default=271828)
	return parser.parse_args(argv)


if __name__ == '__main__':
	train(parse_args())
