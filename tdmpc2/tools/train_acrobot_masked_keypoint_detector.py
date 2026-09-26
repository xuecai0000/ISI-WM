"""Train the paired whole-mask Acrobot keypoint detector."""

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

from tdmpc2.perception.acrobot_keypoint_data import angular_error, gaussian_heatmaps  # noqa: E402
from tdmpc2.perception.acrobot_masked_keypoint_data import (  # noqa: E402
	AcrobotPairedMaskedKeypointDataset, fit_pixel_to_world_homography,
)
from tdmpc2.perception.acrobot_masked_keypoint_detector import (  # noqa: E402
	AcrobotMaskedKeypointConfig, AcrobotMaskedKeypointNet, save_checkpoint,
)


def _sha256(path):
	digest = hashlib.sha256()
	with Path(path).open('rb') as stream:
		for chunk in iter(lambda: stream.read(1024 * 1024), b''):
			digest.update(chunk)
	return digest.hexdigest()


def _condition_loss(model, batch, condition, device):
	rgb = batch[f'rgb_{condition}'].to(device)
	mask = batch[f'mask_{condition}'].to(device)
	actions = batch['actions'].to(device)
	target_world = batch['world_xz'].to(device)
	pixel_xy = batch['pixel_xy'].to(device)
	visible = batch['pixel_visible'].to(device)
	output = model(rgb, actions, mask)
	target_heatmaps = gaussian_heatmaps(
		pixel_xy, visible, input_size=model.config.image_size,
		output_size=output['heatmaps'].shape[-1],
	)
	target_distribution = target_heatmaps.flatten(2)
	target_distribution /= target_distribution.sum(dim=-1, keepdim=True).clamp_min(1e-8)
	point_weight = visible.float()
	def weighted_sample_mean(value):
		return value.reshape(value.shape[0], -1).mean()
	heatmap_per_point = -(
		target_distribution * F.log_softmax(output['heatmaps'].flatten(2), dim=-1)
	).sum(dim=-1)
	heatmap = (heatmap_per_point * point_weight).sum() / point_weight.sum().clamp_min(1.)
	predicted_pixel = (output['image_xy'] + 1.) * (model.config.image_size - 1.) / 2.
	pixel_error = torch.linalg.vector_norm(predicted_pixel - pixel_xy, dim=-1)
	pixel = weighted_sample_mean(F.smooth_l1_loss(
		predicted_pixel / 63., pixel_xy / 63., beta=0.01, reduction='none',
	))
	world = weighted_sample_mean(F.smooth_l1_loss(
		output['world_xz'], target_world, beta=0.02, reduction='none',
	))
	lengths = torch.linalg.vector_norm(
		output['world_xz'][:, 1:] - output['world_xz'][:, :-1], dim=-1,
	)
	topology = weighted_sample_mean(F.smooth_l1_loss(
		lengths, torch.full_like(lengths, 0.5), beta=0.02, reduction='none',
	))
	confidence_target = torch.exp(-pixel_error.detach() / 3.)
	confidence = F.binary_cross_entropy(output['confidence'], confidence_target)
	return output, {
		'heatmap_loss': heatmap, 'pixel_loss': pixel, 'world_loss': world,
		'topology_loss': topology, 'confidence_loss': confidence,
		'point_error_world': weighted_sample_mean(torch.linalg.vector_norm(
			output['world_xz'] - target_world, dim=-1,
		)) * 2.,
		'pixel_error': weighted_sample_mean(pixel_error),
		'angle_error_rad': weighted_sample_mean(
			angular_error(output['world_xz'], target_world)
		),
	}


def _loss(model, batch, device):
	outputs, metrics = {}, {}
	for condition in ('clean', 'hard'):
		outputs[condition], values = _condition_loss(model, batch, condition, device)
		for name, value in values.items():
			metrics[f'{condition}_{name}'] = value
	consistency = F.smooth_l1_loss(
		outputs['clean']['image_xy'], outputs['hard']['image_xy'], beta=0.02,
	)
	supervised = sum(
		metrics[f'{condition}_world_loss']
		+ 0.25 * metrics[f'{condition}_heatmap_loss']
		+ 0.25 * metrics[f'{condition}_pixel_loss']
		+ 0.15 * metrics[f'{condition}_topology_loss']
		+ 0.05 * metrics[f'{condition}_confidence_loss']
		for condition in ('clean', 'hard')
	) / 2.
	total = supervised + 0.20 * consistency
	metrics['paired_consistency_loss'] = consistency
	metrics['loss'] = total
	return total, metrics


@torch.inference_mode()
def _validate(model, loader, device):
	model.eval()
	totals, count = {}, 0
	for batch in loader:
		_, metrics = _loss(model, batch, device)
		batch_size = batch['actions'].shape[0]
		count += batch_size
		for name, value in metrics.items():
			totals[name] = totals.get(name, 0.) + float(value) * batch_size
	return {name: value / count for name, value in totals.items()}


def train(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	device = torch.device(args.device)
	train_data = AcrobotPairedMaskedKeypointDataset(
		args.train_manifest, args.train_masks, history=args.history,
	)
	validation_data = AcrobotPairedMaskedKeypointDataset(
		args.validation_manifest, args.validation_masks, history=args.history,
	)
	train_files = {entry['sha256'] for entry in train_data.raw_manifest['episodes']}
	validation_files = {entry['sha256'] for entry in validation_data.raw_manifest['episodes']}
	if train_files & validation_files:
		raise ValueError('Train/validation episode archives overlap.')
	homography, calibration = fit_pixel_to_world_homography(args.train_manifest)
	if calibration['max_error_world'] > 1e-3:
		raise ValueError(f'Fixed-camera calibration is not exact enough: {calibration}.')
	config = AcrobotMaskedKeypointConfig(
		history=args.history,
		image_size=train_data.raw_manifest['resolution'],
		action_dim=train_data._episodes[0][0]['actions'].shape[-1],
		base_channels=args.base_channels,
		mask_dilation_pixels=args.mask_dilation_pixels,
		background_keep=args.background_keep,
	)
	model = AcrobotMaskedKeypointNet(config, homography).to(device)
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
		print('ACROBOT_MASKED_KEYPOINT_EPOCH', json.dumps({
			'epoch': epoch + 1, 'updates': updates, **metrics,
		}), flush=True)
		score = metrics['hard_point_error_world']
		if score < best:
			best, best_metrics = score, metrics
			save_checkpoint(
				args.output, model,
				calibration={
					'pixel_to_normalized_world_xz': homography.tolist(),
					**calibration,
					'source_manifest': str(Path(args.train_manifest).resolve()),
					'source_manifest_sha256': _sha256(args.train_manifest),
				},
				training={
					'train_manifest': str(Path(args.train_manifest).resolve()),
					'validation_manifest': str(Path(args.validation_manifest).resolve()),
					'train_masks': str(Path(args.train_masks).resolve()),
					'validation_masks': str(Path(args.validation_masks).resolve()),
					'train_episode_sha256': sorted(train_files),
					'validation_episode_sha256': sorted(validation_files),
					'paired_clean_hard': True,
					'seed': args.seed, 'epochs': epoch + 1, 'updates': updates,
				},
				metrics=metrics,
			)
	print(f'CHECKPOINT={args.output.resolve()}')
	print('CAMERA_CALIBRATION', json.dumps(calibration), flush=True)
	print('BEST_VALIDATION', json.dumps(best_metrics), flush=True)


def parse_args(argv=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--train-manifest', type=Path, required=True)
	parser.add_argument('--train-masks', type=Path, required=True)
	parser.add_argument('--validation-manifest', type=Path, required=True)
	parser.add_argument('--validation-masks', type=Path, required=True)
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--history', type=int, default=4)
	parser.add_argument('--base-channels', type=int, default=24)
	parser.add_argument('--mask-dilation-pixels', type=int, default=3)
	parser.add_argument('--background-keep', type=float, default=0.0)
	parser.add_argument('--epochs', type=int, default=30)
	parser.add_argument('--batch-size', type=int, default=128)
	parser.add_argument('--workers', type=int, default=4)
	parser.add_argument('--learning-rate', type=float, default=3e-4)
	parser.add_argument('--device', default='cuda:0')
	parser.add_argument('--seed', type=int, default=271828)
	return parser.parse_args(argv)


if __name__ == '__main__':
	train(parse_args())
