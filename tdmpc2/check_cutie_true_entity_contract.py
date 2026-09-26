"""Dependency-light checks for the opt-in single-entity Acrobot input path.

Execute production function ASTs with a minimal tensor/CUDA stub so the geometry
and worker protocol can be checked on a machine without PyTorch. This is not a
GPU model smoke: the paired runner must still exercise real Cutie before training.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, replace
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


SOURCE = Path(__file__).resolve().parent / 'envs' / 'wrappers' / 'cutie_object.py'
TREE = ast.parse(SOURCE.read_text(encoding='utf-8'), filename=str(SOURCE))
FUNCTIONS = {
	'_as_numpy', '_mask_spatial_feature', 'generic_object_frame',
	'_result_diagnostics', '_whole_acrobot_support', '_whole_acrobot_worker_output',
	'_whole_acrobot_spatial_frame', '_cutie_worker_main',
}
CONSTANTS = {
	'ROLE_NAMES', 'ACROBOT_ENTITY_NAMES', 'TASK_ROLE_NAMES', 'IMAGE_SIZE',
	'NATIVE_HIGHRES_SIZES', 'QUERY_SLOTS', 'QUERY_DIM', 'QUERY_FEATURE_DIM',
	'QUERY_POOL_DIM', 'MASK_POOL_SIZE', 'MASK_SPATIAL_DIM', 'STATUS_DIM',
	'FRAME_FEATURE_DIM', 'FRAME_CONTENT_DIM', 'STACK_FRAMES', 'STACKED_FEATURE_DIM',
}
SELECTED = []
for node in TREE.body:
	if isinstance(node, ast.Assign) and any(
		isinstance(target, ast.Name) and target.id in CONSTANTS for target in node.targets
	):
		SELECTED.append(node)
	elif isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
		SELECTED.append(node)
	elif isinstance(node, ast.ClassDef) and node.name == 'CutieObjectWorkerConfig':
		SELECTED.append(node)
	elif isinstance(node, ast.ClassDef) and node.name == 'CutieObjectWrapper':
		for method in node.body:
			if isinstance(method, ast.FunctionDef) and method.name == '_spatial_policy_frame':
				SELECTED.append(method)


cuda = SimpleNamespace(
	is_available=lambda: True,
	set_device=lambda _: None,
	manual_seed_all=lambda _: None,
	current_device=lambda: 0,
	get_device_name=lambda _: 'CONTRACT_STUB_NOT_REAL_CUDA',
)
torch_stub = SimpleNamespace(
	is_tensor=lambda _: False,
	manual_seed=lambda _: None,
	cuda=cuda,
	device=lambda value: value,
	backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=False, deterministic=True)),
)
namespace = {
	'__name__': __name__, 'np': np, 'torch': torch_stub,
	'asdict': asdict, 'dataclass': dataclass, 'replace': replace,
	'SimpleNamespace': SimpleNamespace, 'os': os, 'mp': mp, 'random': random,
	'CutieObjectWorkerError': RuntimeError,
}
exec(compile(ast.Module(body=SELECTED, type_ignores=[]), str(SOURCE), 'exec'), namespace)


@dataclass(frozen=True)
class Support:
	frames: tuple
	masks: tuple
	role_names: tuple
	annotation_path: Path
	metadata: dict


def support_fixture():
	mask = np.zeros((64, 64), dtype=np.uint8)
	mask[8:32, 30:32] = 1
	mask[32:56, 30:32] = 2
	return Support(
		frames=tuple(np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(6)),
		masks=tuple(mask.copy() for _ in range(6)),
		role_names=('upper_arm', 'lower_arm'),
		annotation_path=Path('contract_support.json'),
		metadata={'task': 'acrobot-swingup', 'source': 'frozen-support'},
	)


def result_fixture(*, lost=False):
	mask = np.zeros((1, 64, 64), dtype=bool)
	mask[:, 8:56, 30:32] = True
	return SimpleNamespace(
		role_names=('whole_acrobot',), masks=mask,
		object_features=np.ones((1, 2048), dtype=np.float32),
		lost=np.asarray([lost]), confidence=np.asarray([0.8], dtype=np.float32),
		mask_score=np.asarray([0.9], dtype=np.float32), runtime_ms=3.0,
	)


class TrueEntityContract(unittest.TestCase):
	def test_support_compilation_precedes_tracking_and_never_mutates_source(self):
		source = support_fixture()
		before = hashlib.sha256(b''.join(x.tobytes() for x in source.masks)).hexdigest()
		compiled = namespace['_whole_acrobot_support'](source)
		self.assertEqual(compiled.role_names, ('whole_acrobot',))
		self.assertIs(compiled.frames, source.frames)
		self.assertEqual(compiled.annotation_path, source.annotation_path)
		for old, new in zip(source.masks, compiled.masks):
			np.testing.assert_array_equal(new, old != 0)
			self.assertEqual(new.dtype, np.uint8)
		self.assertEqual(before, hashlib.sha256(b''.join(x.tobytes() for x in source.masks)).hexdigest())
		self.assertNotIn('tracking_entities', source.metadata)
		with self.assertRaises(ValueError):
			namespace['_whole_acrobot_support'](replace(source, role_names=('cart', 'pole')))

	def test_thin_shape_geometry_survives_even_below_half_cell_occupancy(self):
		result = result_fixture()
		frame, diagnostics = namespace['_whole_acrobot_worker_output'](result)
		self.assertEqual(frame.shape, (2, 590))
		np.testing.assert_array_equal(frame[0], frame[1])
		self.assertAlmostEqual(float(frame[0, 512:576].max()), 0.25)
		self.assertAlmostEqual(float(frame[0, 578]), 96 / 4096)
		self.assertGreater(float(frame[0, 576]), 0.0)
		self.assertGreater(float(frame[0, 577]), 0.0)
		np.testing.assert_array_equal(diagnostics['mask_area_pixels'], [96, 96])
		np.testing.assert_array_equal(diagnostics['valid'], [True, True])
		wrapper = SimpleNamespace(
			_spatial_token_enabled=True, _true_entity_enabled=True, _task='acrobot-swingup'
		)
		# The policy must not pool/threshold the already-correct raw geometry again.
		self.assertIs(namespace['_spatial_policy_frame'](wrapper, frame), frame)
		changed = frame.copy()
		changed[1, 0] += 1.0
		with self.assertRaises(RuntimeError):
			namespace['_spatial_policy_frame'](wrapper, changed)

	def test_validity_is_entity_measurement_and_not_an_invented_link_status(self):
		result = result_fixture(lost=True)
		frame, diagnostics = namespace['_whole_acrobot_worker_output'](result)
		np.testing.assert_array_equal(frame[:, :512], np.zeros((2, 512)))
		np.testing.assert_array_equal(diagnostics['valid'], [False, False])
		np.testing.assert_array_equal(diagnostics['lost'], [True, True])
		self.assertAlmostEqual(float(frame[0, 578]), 96 / 4096)
		result = result_fixture()
		result.object_features[0, 0] = np.nan
		frame, diagnostics = namespace['_whole_acrobot_worker_output'](result)
		self.assertTrue(np.isfinite(frame).all())
		np.testing.assert_array_equal(diagnostics['valid'], [False, False])
		result.role_names = ('upper_arm', 'lower_arm')
		with self.assertRaises(ValueError):
			namespace['_whole_acrobot_worker_output'](result)

	def test_legacy_path_and_other_tasks_remain_opt_in_only(self):
		Config = namespace['CutieObjectWorkerConfig']
		default = Config(repo_path='/repo', checkpoint_path='/weight', support_path='/support')
		self.assertFalse(default.validated().true_entity_enabled)
		with self.assertRaises(ValueError):
			replace(default, true_entity_enabled=True).validated()
		with self.assertRaises(ValueError):
			replace(default, true_entity_enabled='false').validated()
		feature = np.zeros((2, 590), dtype=np.float32)
		wrapper = SimpleNamespace(
			_spatial_token_enabled=True, _true_entity_enabled=False, _task='reacher-visual-small'
		)
		self.assertIs(namespace['_spatial_policy_frame'](wrapper, feature), feature)
		wrapper._task = 'acrobot-swingup'
		np.testing.assert_array_equal(
			namespace['_spatial_policy_frame'](wrapper, feature),
			namespace['_whole_acrobot_spatial_frame'](feature),
		)

	def test_worker_uses_one_adapter_identity_and_only_rgb_requests(self):
		received = {}
		source = support_fixture()
		class Adapter:
			def __init__(self, cfg):
				received['config'] = cfg
			def add_support_prompts(self, support):
				received['support'] = support
			def runtime_summary(self):
				return {
					'input_size': (64, 64), 'support_input_size': (64, 64),
					'mask_output_size': (64, 64), 'tracker_size': (448, 448),
					'permanent_prompts': 6.0,
					'episode_reset_strategy': 'fresh_inference_core_support_replay_v1',
				}
			def reset_episode(self):
				received['reset'] = True
			def track(self, frame):
				self_assert = received.get('reset', False)
				assert self_assert, 'Worker tracked before resetting episode history.'
				received['frame'] = frame.copy()
				return result_fixture()
		class Connection:
			def __init__(self):
				self.sent = []
				self.requests = [
					{'op': 'reset_track', 'request_id': 1, 'frame': source.frames[0]},
					{'op': 'close', 'request_id': 2},
				]
			def recv(self):
				return self.requests.pop(0)
			def send(self, value):
				self.sent.append(value)
			def close(self):
				pass
		config = namespace['CutieObjectWorkerConfig'](
			repo_path='/repo', checkpoint_path='/weight', support_path='/support',
			task='acrobot-swingup', role_names=('upper_arm', 'lower_arm'),
			support_schema='generic_indexed_v1', allow_simulator_support=True,
			true_entity_enabled=True,
		)
		namespace['_adapter_imports'] = lambda: (
			Adapter, SimpleNamespace, None, lambda *args, **kwargs: source
		)
		def fail_on_worker_error(connection, phase, request_id, exc):
			raise AssertionError(f'Worker {phase} failed: {exc}') from exc
		namespace['_send_worker_error'] = fail_on_worker_error
		global_hydra = ModuleType('hydra.core.global_hydra')
		global_hydra.GlobalHydra = SimpleNamespace(
			instance=lambda: SimpleNamespace(is_initialized=lambda: False)
		)
		connection = Connection()
		with patch.dict(sys.modules, {'hydra.core.global_hydra': global_hydra}):
			namespace['_cutie_worker_main'](connection, asdict(config))
		self.assertEqual(received['config'].role_names, ('whole_acrobot',))
		self.assertEqual(received['config'].object_schema, 'generic_entity_indexed_v1')
		self.assertEqual(received['support'].role_names, ('whole_acrobot',))
		self.assertEqual(connection.sent[0]['tracking_entities'], ('whole_acrobot',))
		self.assertEqual(connection.sent[0]['roles'], ('upper_arm', 'lower_arm'))
		self.assertEqual(connection.sent[1]['object'].shape, (2, 590))
		self.assertEqual(connection.sent[-1]['status'], 'closed')


if __name__ == '__main__':
	unittest.main(verbosity=2)
