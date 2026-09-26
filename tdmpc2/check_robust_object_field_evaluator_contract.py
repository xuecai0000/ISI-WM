"""Focused held-out ROF evaluator contracts; no live environment or weights.

The NumPy/hash/provenance tests run without ML dependencies. Reconstruction
tests additionally use the real parser, model and wrapper validators when the
normal TD-MPC2 dependencies are installed. Skips are explicit, not a GPU pass.
"""

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from common import robust_object_field_observation as schema
from tools import evaluate_cutie_multitask_checkpoint as evaluator


ML_AVAILABLE = all(
	importlib.util.find_spec(name) is not None
	for name in ('torch', 'omegaconf', 'gymnasium', 'hydra')
)


class ArrayTensor:
	"""Only the CPU tensor methods used by the evaluator's hash boundary."""
	def __init__(self, array):
		self.array = np.asarray(array)

	def detach(self):
		return self

	def cpu(self):
		return self

	def contiguous(self):
		return ArrayTensor(np.ascontiguousarray(self.array))

	def numpy(self):
		return self.array

	def __getitem__(self, key):
		return ArrayTensor(self.array[key])

	def permute(self, *axes):
		return ArrayTensor(self.array.transpose(axes))


def observation(roles=2):
	return {
		'rgb': ArrayTensor(np.zeros((9, 64, 64), np.uint8)),
		'object': ArrayTensor(np.zeros((roles, 1770), np.float32)),
		'object_mask': ArrayTensor(np.zeros((roles, 3, 64, 64), np.bool_)),
		'role_exists': ArrayTensor(np.ones(roles, np.float32)),
	}


def runtime():
	roles = ['whole_arm', 'goal']
	ready = {
		'treatment': schema.SCHEMA, 'policy_observation': schema.SCHEMA,
		'policy_rgb': True, 'policy_cutie_descriptors': True,
		'policy_native_role_masks': True, 'role_names': roles, 'role_count': 2,
		'role_axis': 'task_static_exact_k_no_padding', 'role_exists': 'all_one',
		'rgb_descriptor_mask_binding': (
			'same_synchronous_cutie_response_native64_three_frame_causal_v1'
		),
		'privileged_runtime_segmentation': False,
		'privileged_runtime_kinematics': False,
	}
	metrics = {
		'robust_object_field_schema': schema.SCHEMA,
		'robust_object_field_frames': 10020,
		'robust_object_field_role_count': 2,
		'robust_object_field_last_sequence_id': 10019,
		**{f'robust_object_field_last_{key}_sha256': 'a' * 64
		   for key in ('source_rgb', 'object', 'mask', 'binding')},
	}
	return {'cutie_object_role_names': roles}, metrics, ready


class EvaluatorContracts(unittest.TestCase):
	def test_backend_is_explicit_and_legacy_cannot_relabel_rof(self):
		self.assertIn('robust_object_field', evaluator.BACKENDS)
		self.assertIsNone(evaluator._robust_object_field_latent({}, 'rgb'))
		with self.assertRaisesRegex(ValueError, 'explicit evaluator backend'):
			evaluator._robust_object_field_latent(
				{'robust_object_field_enabled': True}, 'cutie_object_only'
			)

	def test_exact_one_two_three_role_observation_hashes(self):
		for roles in (1, 2, 3):
			obs = observation(roles)
			self.assertEqual(
				evaluator._object_hash(obs, 'robust_object_field'),
				hashlib.sha256(obs['object'].array.tobytes()).hexdigest(),
			)
			obs['role_exists'].array[-1] = 0
			with self.assertRaisesRegex(ValueError, 'padding roles'):
				evaluator._object_hash(obs, 'robust_object_field')
		obs = observation()
		obs.pop('object_mask')
		with self.assertRaisesRegex(RuntimeError, 'observation keys'):
			evaluator._object_hash(obs, 'robust_object_field')
		obs = observation()
		obs['object_mask'] = ArrayTensor(obs['object_mask'].array.astype(np.float32))
		with self.assertRaisesRegex(ValueError, 'object_mask'):
			evaluator._object_hash(obs, 'robust_object_field')

	def test_rgb_is_bound_to_tracker_source(self):
		obs = observation()
		class Env:
			@property
			def latest_source_rgb_sha256(self):
				return hashlib.sha256(bytes(64 * 64 * 3)).hexdigest()
		self.assertEqual(
			evaluator._rgb_hash(obs, Env(), 'robust_object_field'),
			Env().latest_source_rgb_sha256,
		)
		obs['rgb'].array[-1, 0, 0] = 1
		with self.assertRaisesRegex(RuntimeError, 'differs from the Cutie source'):
			evaluator._rgb_hash(obs, Env(), 'robust_object_field')

	def test_rof_does_not_consume_an_extra_noop_shift_draw(self):
		torch_stub = SimpleNamespace(randint=Mock(), float32=object())
		evaluator._align_object_only_rgb_shift_rng(torch_stub, 'robust_object_field')
		torch_stub.randint.assert_not_called()
		evaluator._align_object_only_rgb_shift_rng(torch_stub, 'cutie_object_only')
		torch_stub.randint.assert_called_once()

	def test_runtime_rejects_legacy_wrapper_incomplete_frames_and_privilege(self):
		raw, metrics, ready = runtime()
		evaluator._robust_object_field_runtime(raw, metrics, ready)
		for key, value in (
			('policy_native_role_masks', False),
			('privileged_runtime_kinematics', True),
			('role_names', ['goal', 'whole_arm']),
		):
			with self.assertRaises(RuntimeError):
				evaluator._robust_object_field_runtime(raw, metrics, {**ready, key: value})
		for key, value in (
			('robust_object_field_frames', 501),
			('robust_object_field_last_sequence_id', 500),
			('robust_object_field_last_mask_sha256', 'x' * 64),
		):
			with self.assertRaises(RuntimeError):
				evaluator._robust_object_field_runtime(raw, {**metrics, key: value}, ready)

	@unittest.skipUnless(ML_AVAILABLE, 'requires the normal TD-MPC2 ML dependencies')
	def test_real_source_validation_and_reconstruction_for_exact_k(self):
		from check_robust_object_field_parser_contract import make_raw, CONFIG_PATH
		from common import robust_object_field as rof
		from common.parser import parse_cfg
		for task, roles in (
			('acrobot-swingup', ['whole_acrobot']),
			('reacher-easy', ['whole_arm', 'goal']),
			('hopper-hop', ['torso', 'leg', 'foot']),
		):
			source = make_raw(task, roles)
			for key, value in {
				'seed': 6, 'steps': 100000, 'eval_freq': 20000,
				'eval_episodes': 3, 'episode_length': 500,
				'video_background_enabled': True, 'video_background_split': 'train',
				'visual_foreground_erosion_pixels': 0,
			}.items():
				source[key] = value
			with patch('hydra.utils.get_original_cwd', return_value=str(CONFIG_PATH.parent)):
				raw = vars(parse_cfg(source))
			raw['obs_shape'] = rof.observation_shapes(raw)
			args = SimpleNamespace(
				task=task, backend='robust_object_field', training_seed=6,
				expected_training_steps=100000, expected_training_eval_freq=20000,
				expected_training_eval_episodes=3, training_condition='hard',
				condition='clean', erosion_pixels=0, env_seed=424243,
				background_seed=1618034, episodes=20,
				checkpoint=Path('models/final.pt'), output=Path('evaluation.json'),
			)
			before = deepcopy(raw)
			cfg = evaluator._prepare(args, raw)
			self.assertEqual(raw, before)
			self.assertEqual(cfg.latent_dim, len(roles) * 320)
			self.assertEqual(cfg.cutie_object_only_latent_dim, len(roles) * 320)
			self.assertFalse(cfg.video_background_enabled)
			self.assertEqual(cfg.video_background_split, 'validation')
			rof.validate_config(cfg)
			for key, value in (
				('latent_dim', 512), ('robust_object_field_enabled', False),
				('object_state_supervision_collect_labels', True),
				('cutie_object_allow_simulator_kinematics_runtime', True),
				('robust_object_field_random_shift_pad', 0),
			):
				with self.assertRaises(ValueError):
					evaluator._prepare(args, {**raw, key: value})
			with self.assertRaisesRegex(ValueError, 'foreground erosion'):
				evaluator._robust_object_field_latent(raw, args.backend, 1)


if __name__ == '__main__':
	unittest.main(verbosity=2)
