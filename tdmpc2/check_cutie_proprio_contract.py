"""Contracts for unchanged Cutie tokens plus safe proprioceptive fusion."""

import argparse
import json
from types import SimpleNamespace
import time
import unittest

import gymnasium as gym
import numpy as np
import torch

from common import cutie_proprio as contract
from common import init, layers
from envs.wrappers.cutie_proprio import CutieProprioWrapper


class Config(SimpleNamespace):
	def get(self, key, default=None):
		return getattr(self, key, default)


def make_config(mode, **overrides):
	values = {
		'task': contract.TASK, 'obs': 'rgb', 'seed': 8,
		'multitask': False, 'model_size': 5,
		'flat_anchor': True, 'flat_anchor_mode': 'cutie_object_only',
		'cutie_object_observation_variant': contract.VARIANT,
		'cutie_object_frame_schema': contract.FRAME_SCHEMA,
		'cutie_object_role_names': list(contract.ROLE_NAMES),
		'cutie_object_support_schema': 'generic_indexed_v1',
		'cutie_object_allow_simulator_support': True,
		'cutie_object_allow_simulator_runtime': False,
		'cutie_object_allow_simulator_kinematics_runtime': mode != 'cutie_only',
		'cutie_object_repo': '/repo', 'cutie_object_checkpoint': '/checkpoint',
		'cutie_object_support_path': '/support', 'cutie_object_config_dir': None,
		'cutie_object_num_roles': 2, 'cutie_object_frame_dim': 1774,
		'cutie_object_stack_frames': 1, 'cutie_object_input_dim': 1774,
		'cutie_object_role_dim': 64, 'cutie_object_hidden_dim': 256,
		'cutie_object_only_latent_dim': 128,
		'cutie_object_auxiliary_target': 'full_descriptor',
		'cutie_object_native_highres_enabled': False,
		'cutie_object_native_highres_size': 128,
		'cutie_object_last_valid_memory': False,
		'cutie_object_policy_burst_plan': None,
		'cutie_object_belief_enabled': False,
		'cutie_object_belief_use_for_control': False,
		'cutie_proprio_mode': mode, 'cutie_proprio_velocity_scale': 10.0,
		'simnorm_dim': 8,
		'video_background_enabled': False,
		'visual_foreground_erosion_pixels': 0,
	}
	values.update(overrides)
	return Config(**values)


class _Physics:
	def __init__(self):
		self.model = SimpleNamespace(nq=2, nv=2)
		self.data = SimpleNamespace(
			qpos=np.asarray([0.2, -0.4]), qvel=np.asarray([1.0, 2.0]),
		)


class _Pixels(gym.Env):
	def __init__(self, with_physics=True):
		self.observation_space = gym.spaces.Box(
			0, 255, shape=(9, 64, 64), dtype=np.uint8,
		)
		self.action_space = gym.spaces.Box(-1., 1., shape=(1,), dtype=np.float32)
		if with_physics:
			self.physics = _Physics()

	def reset(self, **kwargs):
		return torch.zeros(9, 64, 64, dtype=torch.uint8)

	def step(self, action):
		return self.reset(), 1.0, False, {}


class _Client:
	ready = {'backend': 'fake-cutie'}

	def __init__(self):
		self.calls = 0

	def _feature(self):
		self.calls += 1
		value = np.full((2, 590), self.calls, dtype=np.float32)
		value[:, 588] = 1.0
		return value

	def reset_track(self, rgb):
		return self._feature()

	def track(self, rgb):
		return self._feature()

	def close(self):
		pass

	def metrics(self):
		return {'calls': self.calls}


class CutieProprioContractTest(unittest.TestCase):
	def test_equal_shape_and_exact_field_ablation(self):
		for mode in contract.MODES:
			with self.subTest(mode=mode):
				env = CutieProprioWrapper(
					_Pixels(with_physics=mode != 'cutie_only'),
					make_config(mode), _client=_Client(),
				)
				value = env.reset()['object'].numpy()
				self.assertEqual(value.shape, (2, 1774))
				if mode in {'proprio_only', 'factorized_proprio_only'}:
					self.assertTrue((value[:, :1770] == 0).all())
				else:
					self.assertFalse((value[:, :1770] == 0).all())
				if mode == 'cutie_only':
					self.assertTrue((value[:, 1770:] == 0).all())
				else:
					self.assertTrue((value[:, 1773] == 1).all())
				env.close()

	def test_fusion_initializes_to_exact_proprio_latent(self):
		objects = torch.randn(3, 2, 1774)
		outputs = {}
		for mode in ('proprio_only', 'fusion'):
			torch.manual_seed(271828)
			encoder = layers.CutieObjectEncoder(make_config(mode)).apply(init.weight_init)
			encoder.reset_safe_fusion_output()
			outputs[mode] = encoder(objects)
		self.assertTrue(torch.equal(outputs['proprio_only'], outputs['fusion']))

	def test_factorized_proprio_control_is_architecture_matched(self):
		objects = torch.randn(3, 2, 1774)
		objects[..., :1770] = 0.0
		outputs = {}
		states = {}
		for mode in ('factorized', 'factorized_proprio_only'):
			torch.manual_seed(271828)
			encoder = layers.CutieObjectEncoder(make_config(mode)).apply(init.weight_init)
			outputs[mode] = encoder(objects)
			states[mode] = encoder.state_dict()
		self.assertEqual(states['factorized'].keys(), states['factorized_proprio_only'].keys())
		for name in states['factorized']:
			self.assertTrue(torch.equal(
				states['factorized'][name], states['factorized_proprio_only'][name]
			), name)
		self.assertTrue(torch.equal(
			outputs['factorized'], outputs['factorized_proprio_only']
		))

	def test_contract_rejects_hidden_privilege_and_wrong_shape(self):
		with self.assertRaisesRegex(ValueError, 'config mismatch'):
			contract.validate_config(make_config(
				'cutie_only', cutie_object_allow_simulator_kinematics_runtime=True,
			))
		with self.assertRaisesRegex(ValueError, 'config mismatch'):
			contract.validate_config(make_config('fusion', cutie_object_input_dim=1770))


def real_env_smoke(args):
	from envs.dmcontrol import make_env

	cfg = make_config(
		args.mode,
		cutie_object_repo=args.cutie_repo,
		cutie_object_checkpoint=args.cutie_checkpoint,
		cutie_object_support_path=args.support,
		cutie_object_device='cuda:0',
		cutie_object_worker_timeout_seconds=180,
	)
	env = make_env(cfg)
	try:
		observation = env.reset()
		value = np.asarray(observation['object'])
		assert value.shape == (2, 1774) and np.isfinite(value).all()
		action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
		latencies = []
		for _ in range(args.smoke_steps):
			started = time.perf_counter()
			next_observation, _, _, _ = env.step(action)
			latencies.append((time.perf_counter() - started) * 1000.0)
			next_value = np.asarray(next_observation['object'])
			assert next_value.shape == (2, 1774) and np.isfinite(next_value).all()
		current = env
		wrapper_metrics = None
		while current is not None:
			metrics = getattr(current, 'metrics', None)
			if callable(metrics) and hasattr(current, 'cutie_proprio_ready'):
				wrapper_metrics = metrics()
				break
			current = getattr(current, 'env', None)
		print('CUTIE_PROPRIO_LATENCY', json.dumps({
			'mode': args.mode,
			'steps': args.smoke_steps,
			'mean_step_ms': float(np.mean(latencies)),
			'p95_step_ms': float(np.percentile(latencies, 95)),
			'packing_mean_ms': (
				wrapper_metrics.get('mean_packing_runtime_ms')
				if wrapper_metrics else None
			),
		}, sort_keys=True))
	finally:
		env.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--real-env-smoke', action='store_true')
	parser.add_argument('--mode', choices=contract.MODES, default='fusion')
	parser.add_argument('--cutie-repo')
	parser.add_argument('--cutie-checkpoint')
	parser.add_argument('--support')
	parser.add_argument('--smoke-steps', type=int, default=2)
	args, remaining = parser.parse_known_args()
	program = unittest.main(argv=[__file__, *remaining], exit=False, verbosity=2)
	if not program.result.wasSuccessful():
		raise SystemExit(1)
	if args.real_env_smoke:
		if not all((args.cutie_repo, args.cutie_checkpoint, args.support)):
			raise SystemExit('real env smoke requires Cutie repo/checkpoint/support')
		if args.smoke_steps < 1 or args.smoke_steps > 500:
			raise SystemExit('--smoke-steps must be in [1,500]')
		real_env_smoke(args)
		print(f'CUTIE_PROPRIO_REAL_ENV_SMOKE_OK mode={args.mode}')
