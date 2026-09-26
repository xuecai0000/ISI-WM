"""Dataset utilities shared by Acrobot keypoint training and evaluation."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


DATASET_FORMAT = 'acrobot_keypoint_sequence_dataset_v1'


def load_manifest(path):
	path = Path(path).resolve()
	payload = json.loads(path.read_text(encoding='utf-8'))
	if payload.get('format') != DATASET_FORMAT:
		raise ValueError(f'Unsupported keypoint dataset manifest: {path}.')
	if payload.get('task') != 'acrobot-swingup':
		raise ValueError('Keypoint dataset must contain acrobot-swingup only.')
	if tuple(payload.get('point_names', ())) != ('base', 'elbow', 'tip'):
		raise ValueError('Keypoint dataset point order is incompatible.')
	return path, payload


class AcrobotKeypointSequenceDataset(Dataset):
	"""Causal windows indexed from immutable per-episode archives."""

	def __init__(self, manifest, *, history=4, conditions=('clean', 'hard')):
		self.manifest_path, self.manifest = load_manifest(manifest)
		self.history = int(history)
		self.conditions = tuple(conditions)
		if self.history < 3 or not set(self.conditions).issubset({'clean', 'hard'}):
			raise ValueError('Invalid history or condition selection.')
		self._episodes = []
		self._index = []
		for episode_index, record in enumerate(self.manifest['episodes']):
			path = self.manifest_path.parent / record['file']
			archive = np.load(path, allow_pickle=False)
			arrays = {name: archive[name] for name in archive.files}
			archive.close()
			frames = arrays['world_xz'].shape[0]
			if arrays['actions'].shape[0] != frames - 1:
				raise ValueError(f'Action/frame mismatch in {path}.')
			self._episodes.append(arrays)
			for condition in self.conditions:
				for frame_index in range(frames):
					self._index.append((episode_index, condition, frame_index))

	def __len__(self):
		return len(self._index)

	def __getitem__(self, index):
		episode_index, condition, frame_index = self._index[index]
		episode = self._episodes[episode_index]
		frame_indices = np.arange(frame_index - self.history + 1, frame_index + 1)
		frame_indices = np.clip(frame_indices, 0, None)
		rgb = episode[f'rgb_{condition}'][frame_indices]
		action_dim = episode['actions'].shape[-1]
		actions = np.zeros((self.history - 1, action_dim), dtype=np.float32)
		for slot, action_index in enumerate(range(
			frame_index - self.history + 1, frame_index,
		)):
			if action_index >= 0:
				actions[slot] = episode['actions'][action_index]
		return {
			'rgb': torch.from_numpy(np.array(rgb, copy=True)).permute(0, 3, 1, 2).float() / 255.,
			'actions': torch.from_numpy(actions),
			'world_xz': torch.from_numpy(np.array(episode['world_xz'][frame_index], copy=True)),
			'global_omega': torch.from_numpy(
				np.array(episode['global_omega'][frame_index], copy=True)
			),
			'pixel_xy': torch.from_numpy(np.array(episode['pixel_xy'][frame_index], copy=True)),
			'pixel_visible': torch.from_numpy(
				np.array(episode['pixel_visible'][frame_index], copy=True)
			),
			'episode_index': episode_index,
			'frame_index': frame_index,
			'condition_index': 0 if condition == 'clean' else 1,
		}


def gaussian_heatmaps(pixel_xy, visible, *, input_size, output_size, sigma=1.25):
	"""Create low-resolution Gaussian heatmap supervision."""
	device, dtype = pixel_xy.device, pixel_xy.dtype
	scale = (output_size - 1.) / (input_size - 1.)
	centers = pixel_xy * scale
	y = torch.arange(output_size, device=device, dtype=dtype)
	x = torch.arange(output_size, device=device, dtype=dtype)
	grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
	distance = (
		(grid_x[None, None] - centers[..., 0, None, None]).square()
		+ (grid_y[None, None] - centers[..., 1, None, None]).square()
	)
	heatmaps = torch.exp(-distance / (2. * sigma * sigma))
	return heatmaps * visible[..., None, None].to(dtype)


def angular_error(predicted, target):
	"""Absolute global-link angle error [B,2] in radians."""
	predicted_delta = predicted[:, 1:] - predicted[:, :-1]
	target_delta = target[:, 1:] - target[:, :-1]
	predicted_theta = torch.atan2(predicted_delta[..., 0], predicted_delta[..., 1])
	target_theta = torch.atan2(target_delta[..., 0], target_delta[..., 1])
	delta = torch.atan2(
		torch.sin(predicted_theta - target_theta),
		torch.cos(predicted_theta - target_theta),
	)
	return delta.abs()
