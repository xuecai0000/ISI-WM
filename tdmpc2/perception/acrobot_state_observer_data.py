"""Immutable full-episode dataset for the causal Acrobot state observer."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


FORMAT = 'acrobot_causal_keypoint_trace_dataset_v1'


class AcrobotObserverDataset(Dataset):
	def __init__(self, manifest):
		self.manifest_path = Path(manifest).resolve()
		self.manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
		if self.manifest.get('format') != FORMAT or self.manifest.get('task') != 'acrobot-swingup':
			raise ValueError(f'Unsupported observer trace manifest: {manifest}.')
		self._episodes = []
		self._index = []
		for episode_index, record in enumerate(self.manifest['episodes']):
			with np.load(self.manifest_path.parent / record['file'], allow_pickle=False) as archive:
				value = {name: archive[name] for name in archive.files}
			frames = value['target_world_xz'].shape[0]
			if value['actions'].shape != (frames - 1, 1):
				raise ValueError('Observer action/frame shape mismatch.')
			self._episodes.append(value)
			self._index.extend((episode_index, condition) for condition in ('clean', 'hard'))

	def __len__(self):
		return len(self._index)

	def __getitem__(self, index):
		episode_index, condition = self._index[index]
		episode = self._episodes[episode_index]
		frames = episode['target_world_xz'].shape[0]
		previous_action = np.zeros((frames, 1), dtype=np.float32)
		previous_action[1:] = episode['actions']
		return {
			'measured_points': torch.from_numpy(np.array(
				episode[f'predicted_world_xz_{condition}'], copy=True,
			)),
			'confidence': torch.from_numpy(np.array(
				episode[f'confidence_{condition}'], copy=True,
			)),
			'mask_valid': torch.from_numpy(np.array(
				episode[f'mask_valid_{condition}'][:, None], copy=True,
			)),
			'previous_action': torch.from_numpy(previous_action),
			'target_points': torch.from_numpy(np.array(episode['target_world_xz'], copy=True)),
			'target_omega': torch.from_numpy(np.array(episode['target_omega'], copy=True)),
			'episode_index': episode_index,
			'condition_index': 0 if condition == 'clean' else 1,
		}


__all__ = ['AcrobotObserverDataset', 'FORMAT']
