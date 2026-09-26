"""Dependency-light contracts for privileged GT-mask geometry observations."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np



_ORACLE_PATH = Path(__file__).resolve().parent / 'envs' / 'wrappers' / 'gt_mask_oracle.py'
_SPEC = importlib.util.spec_from_file_location('gt_mask_oracle_contract_target', _ORACLE_PATH)
if _SPEC is None or _SPEC.loader is None:
	raise ImportError(_ORACLE_PATH)
oracle = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(oracle)


class _Model:
	ngeom = 2
	nsite = 0
	geom_bodyid = np.asarray([1, 2], dtype=np.int32)
	site_bodyid = np.asarray([], dtype=np.int32)

	_NAMES = {
		('geom', 0): 'cart',
		('geom', 1): 'pole_1',
		('body', 1): 'cart',
		('body', 2): 'pole_1',
	}

	def id2name(self, object_id, object_type):
		return self._NAMES.get((str(object_type), int(object_id)))


class _Physics:
	def __init__(self, segmentation):
		self.model = _Model()
		self.segmentation = np.asarray(segmentation)
		self.calls = []

	def render(self, **kwargs):
		self.calls.append(dict(kwargs))
		return self.segmentation.copy()


def _write_pack(root: Path) -> Path:
	catalog = {
		'task': 'cartpole-swingup',
		'camera_id': 0,
		'segmentation_object_types': {'geom': 5, 'site': 6},
		'matched_objects': {
			'cart': [{
				'object_type': 'geom', 'id': 0, 'name': 'cart',
				'body_id': 1, 'body_name': 'cart',
			}],
			'pole': [{
				'object_type': 'geom', 'id': 1, 'name': 'pole_1',
				'body_id': 2, 'body_name': 'pole_1',
			}],
		},
	}
	catalog_path = root / 'geom_catalog.json'
	catalog_path.write_text(json.dumps(catalog), encoding='utf-8')
	digest = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
	annotations = {
		'format': 'cutie_indexed_mask_support_v1',
		'roles': ['cart', 'pole'],
		'collection': {
			'support_schema': 'generic_indexed_v1',
			'task': 'cartpole-swingup',
			'split': 'support',
			'label_policy': 'simulator_segmentation_support_only',
			'diagnostic_support': True,
			'camera_id': 0,
			'image_size': [64, 64],
			'geom_catalog': 'geom_catalog.json',
			'geom_catalog_sha256': digest,
		},
	}
	path = root / 'annotations.json'
	path.write_text(json.dumps(annotations), encoding='utf-8')
	return path


def main():
	original_constants = oracle._segmentation_constants
	oracle._segmentation_constants = lambda: {'geom': 5, 'site': 6}
	checks = {}
	try:
		with tempfile.TemporaryDirectory() as temporary:
			root = Path(temporary)
			support = _write_pack(root)
			segmentation = np.full((64, 64, 2), -1, dtype=np.int32)
			segmentation[20:30, 5:20, 0] = 0
			segmentation[20:30, 5:20, 1] = 5
			segmentation[10:45, 31:34, 0] = 1
			segmentation[10:45, 31:34, 1] = 5
			physics = _Physics(segmentation)
			env = SimpleNamespace(physics=physics)
			client = oracle.GTMaskGeometryClient(
				env,
				task='cartpole-swingup',
				role_names=('cart', 'pole'),
				support_path=support,
			)
			first = client.reset_track(np.zeros((64, 64, 3), dtype=np.uint8))
			second = client.track(np.full((64, 64, 3), 255, dtype=np.uint8))
			checks['shape_dtype_finite'] = (
				first.shape == (2, 590)
				and first.dtype == np.float32
				and np.isfinite(first).all()
			)
			checks['query_exact_zero'] = np.array_equal(
				first[:, :512], np.zeros((2, 512), dtype=np.float32)
			)
			checks['rgb_invariant'] = np.array_equal(first, second)
			checks['status_valid'] = np.array_equal(
				first[:, 586:],
				np.tile(np.asarray([1, 0, 1, 1], dtype=np.float32), (2, 1)),
			)
			checks['same_camera_render'] = all(
				call == {
					'height': 64, 'width': 64, 'camera_id': 0,
					'segmentation': True,
				}
				for call in physics.calls
			)
			before_tail = first[:, 512:].copy()
			projected = oracle.zero_query_feature(
				np.concatenate((
					np.ones((2, 512), dtype=np.float32), before_tail,
				), axis=-1)
			)
			checks['cutie_geometry_projection'] = (
				np.array_equal(projected[:, :512], np.zeros((2, 512), np.float32))
				and np.array_equal(projected[:, 512:], before_tail)
			)

			physics.segmentation[10:45, 31:34] = -1
			empty = client.track(np.zeros((64, 64, 3), dtype=np.uint8))
			checks['empty_visible_surface_is_missing'] = np.array_equal(
				empty[1, 586:], np.asarray([0, 1, 0, 0], dtype=np.float32)
			)
			metrics = client.metrics()
			checks['privileged_provenance'] = (
				metrics['privileged_runtime_segmentation'] is True
				and metrics['frames'] == 3
				and metrics['visibility_failures'] == {'cart': 0, 'pole': 1}
			)
			client.close()
	finally:
		oracle._segmentation_constants = original_constants
	checks = {name: bool(passed) for name, passed in checks.items()}
	failed = [name for name, passed in checks.items() if not passed]
	if failed:
		raise AssertionError({'failed': failed, 'checks': checks})
	print('GT_MASK_GEOMETRY_CONTRACT_OK', json.dumps(checks, sort_keys=True))


if __name__ == '__main__':
	main()
