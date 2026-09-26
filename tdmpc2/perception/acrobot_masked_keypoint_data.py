"""Paired clean/hard data and fixed-camera calibration for masked keypoints."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
	from tdmpc2.perception.acrobot_keypoint_data import gaussian_heatmaps
except (ImportError, ModuleNotFoundError):
	from perception.acrobot_keypoint_data import gaussian_heatmaps


MASK_DATASET_FORMAT = 'acrobot_whole_cutie_mask_dataset_v1'


def load_mask_manifest(path):
	path = Path(path).resolve()
	payload = json.loads(path.read_text(encoding='utf-8'))
	if payload.get('format') != MASK_DATASET_FORMAT:
		raise ValueError(f'Unsupported whole-Acrobot mask manifest: {path}.')
	if payload.get('task') != 'acrobot-swingup':
		raise ValueError('Whole-object masks must contain acrobot-swingup only.')
	return path, payload


class AcrobotPairedMaskedKeypointDataset(Dataset):
	"""Return same-state clean/hard causal windows with their Cutie masks."""

	def __init__(self, raw_manifest, mask_manifest, *, history=4):
		try:
			from tdmpc2.perception.acrobot_keypoint_data import load_manifest
		except (ImportError, ModuleNotFoundError):
			from perception.acrobot_keypoint_data import load_manifest
		self.raw_manifest_path, self.raw_manifest = load_manifest(raw_manifest)
		self.mask_manifest_path, self.mask_manifest = load_mask_manifest(mask_manifest)
		self.history = int(history)
		if self.history < 3:
			raise ValueError('history must be at least three.')
		if self.mask_manifest.get('source_manifest_sha256') != self._sha256(
			self.raw_manifest_path
		):
			raise ValueError('Mask dataset is not bound to this raw manifest.')
		if len(self.raw_manifest['episodes']) != len(self.mask_manifest['episodes']):
			raise ValueError('Raw/mask episode counts differ.')
		self._episodes = []
		self._index = []
		for episode_index, (raw_record, mask_record) in enumerate(zip(
			self.raw_manifest['episodes'], self.mask_manifest['episodes'], strict=True,
		)):
			if raw_record['sha256'] != mask_record.get('source_episode_sha256'):
				raise ValueError(f'Mask source mismatch for episode {episode_index}.')
			raw_path = self.raw_manifest_path.parent / raw_record['file']
			mask_path = self.mask_manifest_path.parent / mask_record['file']
			with np.load(raw_path, allow_pickle=False) as archive:
				raw = {name: archive[name] for name in archive.files}
			with np.load(mask_path, allow_pickle=False) as archive:
				masks = {name: archive[name] for name in archive.files}
			frames = raw['world_xz'].shape[0]
			for condition in ('clean', 'hard'):
				if masks[f'mask_{condition}'].shape != (frames, 64, 64):
					raise ValueError(f'Invalid {condition} mask shape in {mask_path}.')
				if masks[f'valid_{condition}'].shape != (frames,):
					raise ValueError(f'Invalid {condition} validity shape in {mask_path}.')
			self._episodes.append((raw, masks))
			self._index.extend((episode_index, frame_index) for frame_index in range(frames))

	@staticmethod
	def _sha256(path):
		import hashlib
		digest = hashlib.sha256()
		with Path(path).open('rb') as stream:
			for chunk in iter(lambda: stream.read(1024 * 1024), b''):
				digest.update(chunk)
		return digest.hexdigest()

	def __len__(self):
		return len(self._index)

	def __getitem__(self, index):
		episode_index, frame_index = self._index[index]
		raw, masks = self._episodes[episode_index]
		indices = np.clip(
			np.arange(frame_index - self.history + 1, frame_index + 1), 0, None,
		)
		action_dim = raw['actions'].shape[-1]
		actions = np.zeros((self.history - 1, action_dim), dtype=np.float32)
		for slot, action_index in enumerate(range(
			frame_index - self.history + 1, frame_index,
		)):
			if action_index >= 0:
				actions[slot] = raw['actions'][action_index]
		result = {
			'actions': torch.from_numpy(actions),
			'world_xz': torch.from_numpy(np.array(raw['world_xz'][frame_index], copy=True)),
			'global_omega': torch.from_numpy(
				np.array(raw['global_omega'][frame_index], copy=True)
			),
			'pixel_xy': torch.from_numpy(np.array(raw['pixel_xy'][frame_index], copy=True)),
			'pixel_visible': torch.from_numpy(
				np.array(raw['pixel_visible'][frame_index], copy=True)
			),
			'episode_index': episode_index,
			'frame_index': frame_index,
		}
		for condition in ('clean', 'hard'):
			rgb = np.array(raw[f'rgb_{condition}'][indices], copy=True)
			mask = np.array(masks[f'mask_{condition}'][indices], copy=True)
			valid_history = np.array(
				masks[f'valid_{condition}'][indices], dtype=np.bool_, copy=True,
			)
			# A lost track must not make the keypoint input a black frame. An
			# all-one mask is a causal fallback to raw RGB, with no simulator state.
			mask[~valid_history] = True
			result[f'rgb_{condition}'] = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.
			result[f'mask_{condition}'] = torch.from_numpy(mask[:, None]).float()
			result[f'mask_valid_{condition}'] = bool(masks[f'valid_{condition}'][frame_index])
			result[f'cutie_runtime_ms_{condition}'] = float(
				masks[f'runtime_ms_{condition}'][frame_index]
			)
		return result


def fit_pixel_to_world_homography(raw_manifest):
	"""Fit the single fixed planar camera transform from training labels."""
	try:
		from tdmpc2.perception.acrobot_keypoint_data import load_manifest
	except (ImportError, ModuleNotFoundError):
		from perception.acrobot_keypoint_data import load_manifest
	manifest_path, manifest = load_manifest(raw_manifest)
	pixels, world = [], []
	for record in manifest['episodes']:
		with np.load(manifest_path.parent / record['file'], allow_pickle=False) as archive:
			visible = archive['pixel_visible'].reshape(-1)
			pixels.append(archive['pixel_xy'].reshape(-1, 2)[visible])
			world.append(archive['world_xz'].reshape(-1, 2)[visible])
	pixels = np.concatenate(pixels).astype(np.float64)
	world = np.concatenate(world).astype(np.float64)
	u, v = pixels[:, 0], pixels[:, 1]
	x, z = world[:, 0], world[:, 1]
	a = np.zeros((2 * len(pixels), 8), dtype=np.float64)
	b = np.zeros(2 * len(pixels), dtype=np.float64)
	a[0::2, :3] = np.stack((u, v, np.ones_like(u)), axis=-1)
	a[0::2, 6:] = np.stack((-x * u, -x * v), axis=-1)
	b[0::2] = x
	a[1::2, 3:6] = np.stack((u, v, np.ones_like(u)), axis=-1)
	a[1::2, 6:] = np.stack((-z * u, -z * v), axis=-1)
	b[1::2] = z
	h, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
	if rank != 8:
		raise ValueError('Fixed-camera homography calibration is rank deficient.')
	homography = np.asarray([
		[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.],
	])
	mapped = np.concatenate((pixels, np.ones((len(pixels), 1))), axis=-1) @ homography.T
	mapped = mapped[:, :2] / mapped[:, 2:]
	error = np.linalg.norm(mapped - world, axis=-1) * 2.
	metrics = {
		'samples': int(len(error)),
		'mean_error_world': float(error.mean()),
		'p99_error_world': float(np.percentile(error, 99)),
		'max_error_world': float(error.max()),
	}
	return homography.astype(np.float32), metrics


__all__ = [
	'AcrobotPairedMaskedKeypointDataset', 'MASK_DATASET_FORMAT',
	'fit_pixel_to_world_homography', 'gaussian_heatmaps', 'load_mask_manifest',
]
