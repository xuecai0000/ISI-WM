"""Dependency-light contracts for the masked Acrobot keypoint V2 route."""

from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import unittest

import numpy as np
import torch

from perception.acrobot_masked_keypoint_detector import (
	AcrobotMaskedKeypointConfig, AcrobotMaskedKeypointNet,
	load_checkpoint, save_checkpoint,
)
from perception.acrobot_pose_filter import CausalAcrobotPoseFilter
from perception.acrobot_masked_keypoint_data import AcrobotPairedMaskedKeypointDataset


class AcrobotMaskedKeypointContractTest(unittest.TestCase):
	def test_lost_mask_falls_back_to_raw_rgb(self):
		with TemporaryDirectory() as temporary:
			root = Path(temporary)
			raw_root, mask_root = root / 'raw', root / 'masks'
			raw_root.mkdir(); mask_root.mkdir()
			raw_file = raw_root / 'episode_0000.npz'
			frames = 4
			np.savez_compressed(
				raw_file,
				rgb_clean=np.full((frames, 64, 64, 3), 9, np.uint8),
				rgb_hard=np.full((frames, 64, 64, 3), 11, np.uint8),
				actions=np.zeros((frames - 1, 1), np.float32),
				world_xz=np.zeros((frames, 3, 2), np.float32),
				global_omega=np.zeros((frames, 2), np.float32),
				pixel_xy=np.zeros((frames, 3, 2), np.float32),
				pixel_visible=np.ones((frames, 3), np.bool_),
			)
			raw_sha = hashlib.sha256(raw_file.read_bytes()).hexdigest()
			raw_manifest = raw_root / 'manifest.json'
			raw_manifest.write_text(json.dumps({
				'format': 'acrobot_keypoint_sequence_dataset_v1',
				'task': 'acrobot-swingup', 'point_names': ['base', 'elbow', 'tip'],
				'resolution': 64,
				'episodes': [{'file': raw_file.name, 'sha256': raw_sha}],
			}), encoding='utf-8')
			mask_file = mask_root / 'episode_0000.npz'
			valid = np.asarray([True, False, True, True])
			values = np.zeros((frames, 64, 64), np.bool_)
			values[:, 20:40, 20:40] = True
			common = {
				'mask_clean': values, 'mask_hard': values,
				'valid_clean': valid, 'valid_hard': valid,
				'runtime_ms_clean': np.ones(frames),
				'runtime_ms_hard': np.ones(frames),
			}
			np.savez_compressed(mask_file, **common)
			mask_manifest = mask_root / 'manifest.json'
			mask_manifest.write_text(json.dumps({
				'format': 'acrobot_whole_cutie_mask_dataset_v1',
				'task': 'acrobot-swingup',
				'source_manifest_sha256': hashlib.sha256(raw_manifest.read_bytes()).hexdigest(),
				'episodes': [{
					'file': mask_file.name, 'source_episode_sha256': raw_sha,
				}],
			}), encoding='utf-8')
			dataset = AcrobotPairedMaskedKeypointDataset(
				raw_manifest, mask_manifest, history=3,
			)
			sample = dataset[1]
			self.assertFalse(sample['mask_valid_clean'])
			self.assertTrue(bool(sample['mask_clean'][-1].all()))
			self.assertGreater(float(sample['rgb_clean'][-1].sum()), 0.)

	def test_shapes_fixed_mapping_and_checkpoint(self):
		h = np.asarray([
			[0.02, 0., -0.63],
			[0., -0.02, 0.63],
			[0., 0., 1.],
		], dtype=np.float32)
		config = AcrobotMaskedKeypointConfig(history=4, image_size=64, action_dim=1)
		model = AcrobotMaskedKeypointNet(config, h)
		output = model(
			torch.zeros(2, 4, 3, 64, 64), torch.zeros(2, 3, 1),
			torch.ones(2, 4, 1, 64, 64),
		)
		self.assertEqual(tuple(output['heatmaps'].shape), (2, 3, 16, 16))
		self.assertEqual(tuple(output['world_xz'].shape), (2, 3, 2))
		self.assertNotIn('angular_velocity', output)
		with TemporaryDirectory() as temporary:
			path = Path(temporary) / 'v2.pt'
			save_checkpoint(path, model, calibration={'fixed': True})
			loaded, metadata = load_checkpoint(path)
			np.testing.assert_allclose(
				loaded.pixel_to_world_homography.numpy(), h, atol=1e-7,
			)
			self.assertEqual(metadata['format'], 'acrobot_masked_keypoint_detector_v2')

	def test_mask_suppresses_background_and_preserves_context_margin(self):
		config = AcrobotMaskedKeypointConfig(
			history=4, mask_dilation_pixels=0, background_keep=0.,
		)
		model = AcrobotMaskedKeypointNet(config, torch.eye(3))
		rgb = torch.ones(1, 4, 3, 64, 64)
		mask = torch.zeros(1, 4, 1, 64, 64)
		mask[..., 20:40, 20:40] = 1.
		prepared = model._prepare_inputs(rgb, mask).reshape(1, 4, 4, 64, 64)
		self.assertEqual(float(prepared[:, :, :3, :10, :10].sum()), 0.)
		self.assertGreater(float(prepared[:, :, :3, 25:30, 25:30].sum()), 0.)

	def test_filter_is_causal_and_enforces_two_half_length_links(self):
		value = CausalAcrobotPoseFilter(dt=0.1, alpha=1., window=4)
		omegas = []
		for index in range(6):
			theta = 0.2 * index
			base = np.asarray([0., 1.])
			elbow = base + 0.5 * np.asarray([np.sin(theta), np.cos(theta)])
			tip = elbow + 0.5 * np.asarray([np.sin(-theta), np.cos(-theta)])
			points, omega = value.update(np.stack((base, elbow, tip)), np.ones(3))
			length = np.linalg.norm(points[1:] - points[:-1], axis=-1)
			np.testing.assert_allclose(length, 0.5, atol=1e-6)
			omegas.append(omega)
		np.testing.assert_allclose(omegas[-1], [2., -2.], atol=1e-5)


if __name__ == '__main__':
	unittest.main(verbosity=2)
