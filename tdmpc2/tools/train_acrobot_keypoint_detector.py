"""Train the compact Acrobot visual keypoint detector offline."""

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

from tdmpc2.perception.acrobot_keypoint_data import (  # noqa: E402
	AcrobotKeypointSequenceDataset, angular_error, gaussian_heatmaps,
)
from tdmpc2.perception.acrobot_keypoint_detector import (  # noqa: E402
	AcrobotKeypointConfig, AcrobotKeypointNet, save_checkpoint,
)


def _sha256(path):
	digest = hashlib.sha256()
	with Path(path).open('rb') as stream:
		for chunk in iter(lambda: stream.read(1024 * 1024), b''):
			digest.update(chunk)
	return digest.hexdigest()


def _loss(model, batch, device):
	rgb = batch['rgb'].to(device)
	actions = batch['actions'].to(device)
	target_world = batch['world_xz'].to(device)
	target_omega = batch['global_omega'].to(device)
	pixel_xy = batch['pixel_xy'].to(device)
	visible = batch['pixel_visible'].to(device)
	output = model(rgb, actions)
	target_heatmaps = gaussian_heatmaps(
		pixel_xy, visible, input_size=model.config.image_size,
		output_size=output['heatmaps'].shape[-1],
	)
	target_distribution = target_heatmaps.flatten(2)
	target_distribution = target_distribution / target_distribution.sum(
		dim=-1, keepdim=True
	).clamp_min(1e-8)
	per_point_heatmap = -(
		target_distribution * F.log_softmax(output['heatmaps'].flatten(2), dim=-1)
	).sum(dim=-1)
	heatmap = (
		per_point_heatmap * visible.float()
	).sum() / visible.float().sum().clamp_min(1.)
	world = F.smooth_l1_loss(output['world_xz'], target_world, beta=0.02)
	lengths = torch.linalg.vector_norm(
		output['world_xz'][:, 1:] - output['world_xz'][:, :-1], dim=-1,
	)
	topology = F.smooth_l1_loss(lengths, torch.full_like(lengths, 0.5), beta=0.02)
	point_error = torch.linalg.vector_norm(
		output['world_xz'] - target_world, dim=-1,
	)
	confidence_target = torch.exp(-point_error.detach() / 0.04).clamp(0., 1.)
	confidence = F.binary_cross_entropy(output['confidence'], confidence_target)
	omega = F.smooth_l1_loss(output['angular_velocity'], target_omega, beta=0.5)
	total = (
		world + 0.25 * heatmap + 0.20 * topology
		+ 0.05 * confidence + 0.05 * omega
	)
	metrics = {
		'loss': total, 'world_loss': world, 'heatmap_loss': heatmap,
		'topology_loss': topology,
		'angular_velocity_loss': omega,
		'point_error_world': point_error.mean() * 2.,
		'angle_error_rad': angular_error(output['world_xz'], target_world).mean(),
	}
	return total, metrics


@torch.inference_mode()
def _validate(model, loader, device):
	model.eval()
	totals, count = {}, 0
	for batch in loader:
		_, metrics = _loss(model, batch, device)
		batch_size = batch['rgb'].shape[0]
		count += batch_size
		for name, value in metrics.items():
			totals[name] = totals.get(name, 0.) + float(value) * batch_size
	return {name: value / count for name, value in totals.items()}


def train(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	device = torch.device(args.device)
	train_data = AcrobotKeypointSequenceDataset(
		args.train_manifest, history=args.history, conditions=tuple(args.conditions),
	)
	validation_data = AcrobotKeypointSequenceDataset(
		args.validation_manifest, history=args.history, conditions=tuple(args.conditions),
	)
	train_files = {entry['sha256'] for entry in train_data.manifest['episodes']}
	validation_files = {entry['sha256'] for entry in validation_data.manifest['episodes']}
	if train_files & validation_files:
		raise ValueError('Train/validation episode archives overlap.')
	train_trajectories = {
		(entry.get('env_seed'), entry.get('action_seed'), entry.get('pose_seed'))
		for entry in train_data.manifest['episodes']
	}
	validation_trajectories = {
		(entry.get('env_seed'), entry.get('action_seed'), entry.get('pose_seed'))
		for entry in validation_data.manifest['episodes']
	}
	all_seed_values = {
		value for pair in train_trajectories | validation_trajectories for value in pair
	}
	if None in all_seed_values:
		raise ValueError('Dataset manifest lacks immutable environment/action seeds.')
	if train_trajectories & validation_trajectories:
		raise ValueError('Train/validation simulator trajectories overlap.')
	config = AcrobotKeypointConfig(
		history=args.history, image_size=train_data.manifest['resolution'],
		action_dim=train_data._episodes[0]['actions'].shape[-1],
		use_foreground_mask=False, base_channels=args.base_channels,
	)
	model = AcrobotKeypointNet(config).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
	train_loader = DataLoader(
		train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
		pin_memory=device.type == 'cuda', drop_last=True,
	)
	validation_loader = DataLoader(
		validation_data, batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=device.type == 'cuda',
	)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	best, best_metrics = float('inf'), None
	updates = 0
	for epoch in range(args.epochs):
		model.train()
		for batch in train_loader:
			optimizer.zero_grad(set_to_none=True)
			loss, _ = _loss(model, batch, device)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
			optimizer.step()
			updates += 1
		metrics = _validate(model, validation_loader, device)
		print('ACROBOT_KEYPOINT_EPOCH', json.dumps({
			'epoch': epoch + 1, 'updates': updates, **metrics,
		}), flush=True)
		if metrics['point_error_world'] < best:
			best = metrics['point_error_world']
			best_metrics = metrics
			save_checkpoint(args.output, model, training={
				'train_manifest': str(Path(args.train_manifest).resolve()),
				'validation_manifest': str(Path(args.validation_manifest).resolve()),
				'train_manifest_sha256': _sha256(args.train_manifest),
				'validation_manifest_sha256': _sha256(args.validation_manifest),
				'train_episode_sha256': sorted(train_files),
				'validation_episode_sha256': sorted(validation_files),
				'train_trajectories': [list(value) for value in sorted(train_trajectories)],
				'validation_trajectories': [
					list(value) for value in sorted(validation_trajectories)
				],
				'conditions': list(args.conditions), 'seed': args.seed,
				'epochs': epoch + 1, 'updates': updates,
			}, metrics=metrics)
	print(f'CHECKPOINT={args.output.resolve()}')
	print('BEST_VALIDATION', json.dumps(best_metrics), flush=True)


def parse_args(argv=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--train-manifest', type=Path, required=True)
	parser.add_argument('--validation-manifest', type=Path, required=True)
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--conditions', nargs='+', choices=('clean', 'hard'), default=('clean', 'hard'))
	parser.add_argument('--history', type=int, default=4)
	parser.add_argument('--base-channels', type=int, default=24)
	parser.add_argument('--epochs', type=int, default=30)
	parser.add_argument('--batch-size', type=int, default=128)
	parser.add_argument('--workers', type=int, default=4)
	parser.add_argument('--learning-rate', type=float, default=3e-4)
	parser.add_argument('--device', default='cuda:0')
	parser.add_argument('--seed', type=int, default=271828)
	return parser.parse_args(argv)


if __name__ == '__main__':
	train(parse_args())
