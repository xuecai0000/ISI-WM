"""CPU-only contracts for the clean-train masked-RGB representation ablation."""

import importlib.util
import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from check_cutie_object_wrapper_contract import module as sensor, _FakeEnv, _FakeClient
from tools.evaluate_cutie_multitask_checkpoint import (
	_validate_training_condition, _object_hash, parse_args,
)


# Load only these wrappers, not optional simulator suites or the real tracker.
package = types.ModuleType('masked_rgb_contract_package')
package.__path__ = []
sys.modules[package.__name__] = package
sys.modules[package.__name__ + '.cutie_object'] = sensor
spec = importlib.util.spec_from_file_location(
	package.__name__ + '.cutie_masked_rgb',
	Path(__file__).parent / 'envs/wrappers/cutie_masked_rgb.py',
)
masked = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = masked
spec.loader.exec_module(masked)


def config():
	return dict(
		task='reacher-visual-small', multitask=False, obs='rgb', model_size=5,
		flat_anchor=False, cutie_masked_rgb_enabled=True,
		cutie_masked_rgb_schema=masked.SCHEMA,
		cutie_object_observation_variant='full',
		cutie_object_frame_schema='cutie_query_mask_status_v1',
		cutie_object_num_roles=2,
	)


class MaskClient(_FakeClient):
	def _next(self, op, frame):
		feature = super()._next(op, frame)
		self.last_masks = np.zeros((2, 64, 64), dtype=np.bool_)
		index = len(self.frames)
		self.last_masks[0, index, 3] = True
		self.last_masks[1, 20, 30] = True
		return feature


class MaskedRGBContracts(unittest.TestCase):
	def test_colors_location_union_and_empty_mask(self):
		rgb = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
		masks = np.zeros((2, 64, 64), dtype=np.bool_)
		masks[0, 5:8, 6:11] = True
		masks[1, 6:10, 8:13] = True
		out = masked.mask_native_rgb(rgb, masks, 2).transpose(1, 2, 0)
		np.testing.assert_array_equal(out[masks.any(0)], rgb[masks.any(0)])
		self.assertTrue((out[~masks.any(0)] == 0).all())
		self.assertFalse(masked.mask_native_rgb(rgb, masks & False, 2).any())
		with self.assertRaises(sensor.CutieObjectWorkerError):
			masked.mask_native_rgb(rgb, masks.astype(np.float32), 2)
		with self.assertRaises(sensor.CutieObjectWorkerError):
			masked.mask_native_rgb(rgb, masks[:, ::2, ::2], 2)

	def test_causal_stack_reset_source_hash_and_no_objects(self):
		env, client = _FakeEnv(), MaskClient()
		cfg = config()
		wrapper = masked.CutieMaskedRGBWrapper(env, cfg, _client=client)
		self.assertIs(wrapper.observation_space, env.observation_space)
		with self.assertRaises(sensor.CutieObjectWorkerError):
			wrapper.step(torch.zeros(2))
		self.assertFalse(client.ops)
		first = wrapper.reset()
		self.assertEqual(tuple(first.shape), (9, 64, 64))
		self.assertEqual(first.dtype, torch.uint8)
		self.assertIsNone(_object_hash(first, 'cutie_masked_rgb'))
		torch.testing.assert_close(first[:3], first[3:6])
		torch.testing.assert_close(first[:3], first[6:])
		second, reward, done, _ = wrapper.step(torch.zeros(2))
		torch.testing.assert_close(second[:6], first[3:])
		self.assertEqual(second[-1, 1, 3].item(), 0)
		self.assertEqual(second[-1, 2, 3].item(), 23)
		self.assertEqual(first[-1, 1, 3].item(), 22)
		self.assertEqual(reward, 7.25)
		self.assertFalse(done)
		third = wrapper.reset()
		torch.testing.assert_close(third[:3], third[6:])
		self.assertEqual(third[-1, 1, 3].item(), 0)
		self.assertEqual(third[-1, 3, 3].item(), 33)
		self.assertEqual(client.ops, ['reset_track', 'track', 'reset_track'])
		self.assertEqual(wrapper.metrics()['masked_rgb_frames'], 3)
		self.assertFalse(cfg['flat_anchor'])
		self.assertNotIn('flat_anchor_mode', cfg)
		wrapper.close()
		self.assertEqual(client.close_calls, 1)

	def test_privilege_and_variant_fail_closed(self):
		for key, value in {
			'flat_anchor': True,
			'cutie_object_allow_simulator_runtime': True,
			'cutie_object_allow_simulator_kinematics_runtime': True,
			'object_state_supervision_enabled': True,
			'object_state_bottleneck_enabled': True,
			'cutie_object_native_highres_enabled': True,
			'cutie_object_spatial_token_enabled': True,
			'cutie_object_last_valid_memory': True,
			'cutie_object_observation_variant': 'gt_mask_geometry',
		}.items():
			with self.subTest(key=key), self.assertRaises(ValueError):
				masked.validate_masked_rgb_config({**config(), key: value})

	def test_explicit_archival_acrobot_taxonomy(self):
		self.assertEqual(sensor.task_role_names('acrobot-swingup'), ('whole_acrobot',))
		self.assertEqual(sensor.task_role_names('acrobot-swingup', 'legacy_acrobot_links_v1'), ('upper_arm', 'lower_arm'))
		base = sensor.CutieObjectWorkerConfig(
			repo_path='/repo', checkpoint_path='/weights', support_path='/support',
			task='acrobot-swingup', role_names=('upper_arm', 'lower_arm'),
			support_schema='generic_indexed_v1', allow_simulator_support=True,
			task_role_contract='legacy_acrobot_links_v1', return_masks=True,
		)
		self.assertTrue(base.validated().return_masks)
		for bad in (
			replace(base, task_role_contract='canonical_v1'),
			replace(base, task='cartpole-swingup'),
			replace(base, role_names=('lower_arm', 'upper_arm')),
			replace(base, true_entity_enabled=True),
		):
			with self.assertRaises(ValueError):
				bad.validated()

	def test_clean_train_requires_explicit_evaluator_contract(self):
		raw = {'video_background_enabled': False, 'video_background_split': 'train'}
		_validate_training_condition(raw, 'clean')
		with self.assertRaises(ValueError):
			_validate_training_condition(raw, 'hard')
		_validate_training_condition({**raw, 'video_background_enabled': True}, 'hard')
		args = parse_args([
			'--task', 'reacher-easy', '--backend', 'cutie_masked_rgb',
			'--training-condition', 'clean', '--runtime-config', '/cfg',
			'--checkpoint', '/checkpoint', '--output', '/output',
			'--measure-online-latency',
		])
		self.assertTrue(args.measure_online_latency)
		self.assertEqual(args.training_condition, 'clean')


if __name__ == '__main__':
	unittest.main(verbosity=2)
