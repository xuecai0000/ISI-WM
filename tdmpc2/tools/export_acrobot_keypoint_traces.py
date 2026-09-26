"""Freeze causal V2 keypoint outputs for recurrent observer experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))

from tdmpc2.perception.acrobot_masked_keypoint_data import (  # noqa: E402
	AcrobotPairedMaskedKeypointDataset,
)
from tdmpc2.perception.acrobot_masked_keypoint_detector import load_checkpoint  # noqa: E402


FORMAT = 'acrobot_causal_keypoint_trace_dataset_v1'


def _sha256(path):
	digest = hashlib.sha256()
	with Path(path).open('rb') as stream:
		for chunk in iter(lambda: stream.read(1024 * 1024), b''):
			digest.update(chunk)
	return digest.hexdigest()


@torch.inference_mode()
def export(args):
	root = args.output.resolve()
	if root.exists():
		raise FileExistsError(f'Refusing to overwrite trace dataset: {root}')
	root.mkdir(parents=True)
	device = torch.device(args.device)
	model, checkpoint = load_checkpoint(args.checkpoint, device=device)
	dataset = AcrobotPairedMaskedKeypointDataset(
		args.input_manifest, args.mask_manifest, history=model.config.history,
	)
	loader = DataLoader(
		dataset, batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=device.type == 'cuda',
	)
	episodes = []
	for raw, masks in dataset._episodes:
		frames = raw['world_xz'].shape[0]
		value = {
			'actions': np.array(raw['actions'], np.float32, copy=True),
			'target_world_xz': np.array(raw['world_xz'], np.float32, copy=True),
			'target_omega': np.array(raw['global_omega'], np.float32, copy=True),
		}
		for condition in ('clean', 'hard'):
			value[f'predicted_world_xz_{condition}'] = np.zeros((frames, 3, 2), np.float32)
			value[f'confidence_{condition}'] = np.zeros((frames, 3), np.float32)
			value[f'mask_valid_{condition}'] = np.array(
				masks[f'valid_{condition}'], np.bool_, copy=True,
			)
		episodes.append(value)

	for batch in loader:
		actions = batch['actions'].to(device)
		for condition in ('clean', 'hard'):
			output = model(
				batch[f'rgb_{condition}'].to(device), actions,
				batch[f'mask_{condition}'].to(device),
			)
			points = output['world_xz'].cpu().numpy()
			confidence = output['confidence'].cpu().numpy()
			for row in range(actions.shape[0]):
				episode = int(batch['episode_index'][row])
				frame = int(batch['frame_index'][row])
				episodes[episode][f'predicted_world_xz_{condition}'][frame] = points[row]
				episodes[episode][f'confidence_{condition}'][frame] = confidence[row]

	records = []
	for episode_index, (source, arrays) in enumerate(zip(
		dataset.raw_manifest['episodes'], episodes, strict=True,
	)):
		path = root / f'episode_{episode_index:04d}.npz'
		np.savez_compressed(path, **arrays)
		record = {
			'episode_index': episode_index, 'file': path.name,
			'sha256': _sha256(path),
			'source_episode_sha256': source['sha256'],
			'frames': int(arrays['target_world_xz'].shape[0]),
		}
		records.append(record)
		print('ACROBOT_KEYPOINT_TRACE_EPISODE', json.dumps(record), flush=True)
	payload = {
		'format': FORMAT, 'task': 'acrobot-swingup',
		'conditions': ['clean', 'hard'],
		'causal': True,
		'checkpoint': str(args.checkpoint.resolve()),
		'checkpoint_sha256': _sha256(args.checkpoint),
		'source_manifest': str(args.input_manifest.resolve()),
		'source_manifest_sha256': _sha256(args.input_manifest),
		'mask_manifest': str(args.mask_manifest.resolve()),
		'mask_manifest_sha256': _sha256(args.mask_manifest),
		'checkpoint_format': checkpoint['format'],
		'episodes': records,
	}
	manifest = root / 'manifest.json'
	manifest.write_text(json.dumps(payload, indent=2), encoding='utf-8')
	print(f'MANIFEST={manifest}', flush=True)
	return payload


def parse_args(argv=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--input-manifest', type=Path, required=True)
	parser.add_argument('--mask-manifest', type=Path, required=True)
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--device', default='cuda:0')
	parser.add_argument('--batch-size', type=int, default=256)
	parser.add_argument('--workers', type=int, default=4)
	return parser.parse_args(argv)


if __name__ == '__main__':
	export(parse_args())
