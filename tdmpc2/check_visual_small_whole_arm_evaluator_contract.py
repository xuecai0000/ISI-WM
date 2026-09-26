"""Synthetic integration contract for the whole-arm Cutie evaluator path.

The tracker is a local deterministic fake, while support calibration, decoder,
evaluator normalization, mask persistence, cache serialization, and cache
reload are real.  No checkpoint, audit/blind data, simulator, or GPU is used.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
	sys.path.insert(0, str(ROOT.parent))


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	if spec is None or spec.loader is None:
		raise RuntimeError(f'Could not import {path}')
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


ab_contract = _load_module(
	'whole_arm_evaluator_ab_fixtures',
	ROOT / 'check_visual_small_perception_ab_contract.py',
)
whole_contract = _load_module(
	'whole_arm_evaluator_decoder_fixtures',
	ROOT / 'check_visual_small_whole_arm_decoder_contract.py',
)
EVALUATOR = ab_contract.EVALUATOR
POINT_ROLES = ('base', 'elbow', 'control_tip', 'goal')
OBJECT_ROLES = ('whole_arm', 'goal')


def _dump_json(path: Path, value) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', encoding='utf-8', newline='\n') as stream:
		json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
		stream.write('\n')


class _FakeCutieAdapter:
	last_config = None
	last_support = None

	def __init__(self, config):
		type(self).last_config = config
		self.config = config

	def add_support_prompts(self, support) -> None:
		type(self).last_support = support
		if tuple(support.role_names) != OBJECT_ROLES:
			raise AssertionError(support.role_names)

	def track_episode(self, frames):
		masks, _, _, _ = whole_contract._normal_masks(0.0, 1.0)
		stacked = np.stack([masks[role] for role in OBJECT_ROLES])
		results = []
		for index, _ in enumerate(frames):
			if index == 0:
				confidence = [0.8, 0.4]
				lost = [True, False]
			elif index == 1:
				confidence = [0.25, 0.9]
				lost = [False, True]
			else:
				confidence = [0.2, 0.4]
				lost = [False, False]
			results.append(SimpleNamespace(
				role_names=OBJECT_ROLES,
				masks=stacked,
				object_features=np.asarray(
					[[1.0, 2.0], [3.0, 4.0]], dtype=np.float32
				),
				confidence=np.asarray(confidence, dtype=np.float64),
				lost=np.asarray(lost, dtype=bool),
				runtime_ms=1.25,
				input_size=(64, 64),
				tracker_size=(448, 448),
			))
		return results

	def runtime_summary(self):
		return {'backend': 'synthetic_contract_tracker'}


class VisualSmallWholeArmEvaluatorContractTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls._temporary = tempfile.TemporaryDirectory(
			prefix='visual_small_whole_arm_evaluator_contract_'
		)
		cls.root = Path(cls._temporary.name)
		cls.support = whole_contract._make_support(cls.root / 'support')
		cls.audit_path, cls.audit_payload = ab_contract._make_audit(
			cls.root / 'audit'
		)
		cls.audit = EVALUATOR.load_audit(cls.audit_path)
		cls.dino_cache = ab_contract._make_cache(
			cls.root / 'audit',
			audit_path=cls.audit_path,
			audit_payload=cls.audit_payload,
			backend='dino',
			offset_px=1.0,
			name='whole_arm_contract_dino',
		)
		cls.legacy_cutie_cache = ab_contract._make_cache(
			cls.root / 'audit',
			audit_path=cls.audit_path,
			audit_payload=cls.audit_payload,
			backend='cutie',
			offset_px=0.0,
			name='whole_arm_contract_legacy_cutie',
		)

	@classmethod
	def tearDownClass(cls):
		cls._temporary.cleanup()

	def make_decoder(self):
		return EVALUATOR._build_cutie_point_decoder(
			EVALUATOR.CUTIE_DECODER_WHOLE_ARM,
			support_path=self.support,
		)

	def test_cli_and_factory_enforce_exact_decoder_schema_pairing(self):
		decoder = self.make_decoder()
		self.assertEqual(
			EVALUATOR.CUTIE_DECODER_WHOLE_ARM,
			'whole_arm_kinematic_v2',
		)
		self.assertEqual(
			tuple(decoder.metadata()['input_roles']), OBJECT_ROLES
		)
		for key, expected in (
			EVALUATOR.CUTIE_DECODER_WHOLE_ARM_V2_IDENTITY.items()
		):
			self.assertEqual(decoder.metadata()[key], expected)
		args = EVALUATOR.parse_args([
			'--annotations', str(self.audit_path),
			'--output-dir', str(self.root / 'cli'),
			'--dino-cache', str(self.dino_cache),
			'--cutie-cache', str(self.root / 'whole_cache.json'),
			'--cutie-support', str(self.support),
			'--cutie-point-decoder', EVALUATOR.CUTIE_DECODER_WHOLE_ARM,
			'--cutie-object-schema', EVALUATOR.CUTIE_OBJECT_SCHEMA_WHOLE_ARM,
		])
		self.assertEqual(
			args.cutie_point_decoder, EVALUATOR.CUTIE_DECODER_WHOLE_ARM
		)
		self.assertEqual(
			args.cutie_object_schema, EVALUATOR.CUTIE_OBJECT_SCHEMA_WHOLE_ARM
		)
		invalid_pairs = (
			(
				EVALUATOR.CUTIE_DECODER_WHOLE_ARM,
				EVALUATOR.CUTIE_OBJECT_SCHEMA_LEGACY,
			),
			(
				EVALUATOR.CUTIE_DECODER_KINEMATIC,
				EVALUATOR.CUTIE_OBJECT_SCHEMA_WHOLE_ARM,
			),
			(
				EVALUATOR.CUTIE_DECODER_LEGACY,
				EVALUATOR.CUTIE_OBJECT_SCHEMA_WHOLE_ARM,
			),
		)
		for decoder_mode, object_schema in invalid_pairs:
			with self.subTest(pair=(decoder_mode, object_schema)):
				with mock.patch('sys.stderr', new=io.StringIO()):
					with self.assertRaises(SystemExit):
						EVALUATOR.parse_args([
							'--annotations', str(self.audit_path),
							'--output-dir', str(self.root / 'invalid_cli'),
							'--dino-cache', str(self.dino_cache),
							'--cutie-cache', str(self.root / 'whole_cache.json'),
							'--cutie-support', str(self.support),
							'--cutie-point-decoder', decoder_mode,
							'--cutie-object-schema', object_schema,
						])
		# The retired v1 name is not accepted as a CLI alias for v2 semantics.
		with mock.patch('sys.stderr', new=io.StringIO()):
			with self.assertRaises(SystemExit):
				EVALUATOR.parse_args([
					'--annotations', str(self.audit_path),
					'--output-dir', str(self.root / 'retired_v1_cli'),
					'--dino-cache', str(self.dino_cache),
					'--cutie-cache', str(self.root / 'whole_cache.json'),
					'--cutie-support', str(self.support),
					'--cutie-point-decoder',
					EVALUATOR.CUTIE_DECODER_WHOLE_ARM_V1,
					'--cutie-object-schema',
					EVALUATOR.CUTIE_OBJECT_SCHEMA_WHOLE_ARM,
				])

	def test_live_persistence_cache_roundtrip_and_report_schema(self):
		import tdmpc2.perception.cutie_oc_adapter as adapter_module

		asset_root = self.root / 'live_output'
		decoder = self.make_decoder()
		with mock.patch.object(
			adapter_module,
			'inspect_cutie_installation',
			return_value={'contract': 'synthetic'},
		), mock.patch.object(
			adapter_module, 'CutieOCAdapter', _FakeCutieAdapter
		):
			live = EVALUATOR.run_cutie_adapter(
				audit=self.audit,
				repo_path=self.root / 'unused_repo',
				checkpoint_path=self.root / 'unused_checkpoint.pth',
				support_path=self.support,
				device='cpu',
				tracker_size=(448, 448),
				asset_dir=asset_root / 'prediction_assets',
				cutie_decoder=decoder,
				cutie_point_decoder=EVALUATOR.CUTIE_DECODER_WHOLE_ARM,
				cutie_object_schema=EVALUATOR.CUTIE_OBJECT_SCHEMA_WHOLE_ARM,
			)

		self.assertEqual(_FakeCutieAdapter.last_config.role_names, OBJECT_ROLES)
		self.assertEqual(
			_FakeCutieAdapter.last_config.object_schema,
			EVALUATOR.CUTIE_OBJECT_SCHEMA_WHOLE_ARM,
		)
		self.assertEqual(_FakeCutieAdapter.last_support.role_names, OBJECT_ROLES)
		self.assertEqual(
			live['metadata']['object_schema'],
			EVALUATOR.CUTIE_OBJECT_SCHEMA_WHOLE_ARM,
		)
		self.assertEqual(live['metadata']['object_roles'], list(OBJECT_ROLES))
		self.assertEqual(
			live['metadata']['point_decoder_mode'],
			EVALUATOR.CUTIE_DECODER_WHOLE_ARM_V2,
		)
		for sequence in live['sequences'].values():
			for frame in sequence['frames']:
				self.assertEqual(tuple(frame['object_mask_paths']), OBJECT_ROLES)
				self.assertEqual(tuple(frame['object_feature_paths']), OBJECT_ROLES)
				self.assertEqual(
					frame['point_decoder'], EVALUATOR.CUTIE_DECODER_WHOLE_ARM
				)
				self.assertEqual(
					set(frame['decoder_diagnostics']['backend_object_lost']),
					set(OBJECT_ROLES),
				)
				self.assertFalse(frame['lost']['base'])
			self.assertEqual(
				sequence['frames'][0]['backend_object_lost'],
				{'whole_arm': True, 'goal': False},
			)
			self.assertIsNone(sequence['frames'][0]['points']['elbow'])
			self.assertIsNone(sequence['frames'][0]['points']['control_tip'])
			self.assertTrue(sequence['frames'][0]['lost']['elbow'])
			self.assertTrue(sequence['frames'][0]['lost']['control_tip'])
			self.assertEqual(sequence['frames'][0]['confidence']['elbow'], 0.0)
			self.assertEqual(
				sequence['frames'][0]['confidence']['control_tip'], 0.0
			)
			self.assertFalse(sequence['frames'][0]['lost']['goal'])
			self.assertEqual(sequence['frames'][0]['confidence']['goal'], 0.4)
			self.assertIsNone(sequence['frames'][1]['points']['goal'])
			self.assertTrue(sequence['frames'][1]['lost']['goal'])
			self.assertEqual(sequence['frames'][1]['confidence']['goal'], 0.0)
			self.assertLessEqual(sequence['frames'][2]['confidence']['elbow'], 0.2)
			self.assertLessEqual(
				sequence['frames'][2]['confidence']['control_tip'], 0.2
			)
			self.assertEqual(sequence['frames'][2]['confidence']['goal'], 0.4)

		payload = EVALUATOR._cache_json_value(live, self.audit, asset_root)
		self.assertEqual(
			payload['metadata']['point_decoder_mode'],
			EVALUATOR.CUTIE_DECODER_WHOLE_ARM_V2,
		)
		cache_path = asset_root / 'whole_cache.json'
		_dump_json(cache_path, payload)
		roundtrip = EVALUATOR.load_prediction_cache(
			cache_path,
			backend='cutie',
			audit=self.audit,
			cutie_decoder=self.make_decoder(),
			cutie_point_decoder=EVALUATOR.CUTIE_DECODER_WHOLE_ARM,
			cutie_object_schema=EVALUATOR.CUTIE_OBJECT_SCHEMA_WHOLE_ARM,
		)
		self.assertFalse(roundtrip['metadata']['legacy_object_schema_inferred'])
		self.assertEqual(roundtrip['metadata']['point_decoder'], decoder.metadata())
		self.assertEqual(
			roundtrip['metadata']['point_decoder_mode'],
			EVALUATOR.CUTIE_DECODER_WHOLE_ARM_V2,
		)
		for sequence_id in live['sequences']:
			for left, right in zip(
				live['sequences'][sequence_id]['frames'],
				roundtrip['sequences'][sequence_id]['frames'],
			):
				self.assertEqual(left['points'], right['points'])
				self.assertEqual(left['confidence'], right['confidence'])
				self.assertEqual(left['lost'], right['lost'])
				self.assertEqual(
					left['backend_object_confidence'],
					right['backend_object_confidence'],
				)
				self.assertEqual(
					left['backend_object_lost'], right['backend_object_lost']
				)
				self.assertEqual(
					left['decoder_diagnostics']['backend_object_confidence'],
					right['decoder_diagnostics']['backend_object_confidence'],
				)
				self.assertEqual(
					left['decoder_diagnostics']['backend_object_lost'],
					right['decoder_diagnostics']['backend_object_lost'],
				)
				self.assertIn('cached_point_confidence', right['decoder_diagnostics'])

		tampered_payload = copy.deepcopy(payload)
		for sequence in tampered_payload['sequences']:
			for frame in sequence['frames']:
				frame['points'] = {role: [63.0, 0.0] for role in POINT_ROLES}
				frame['confidence'] = {role: 0.0 for role in POINT_ROLES}
				frame['lost'] = {role: True for role in POINT_ROLES}
		tampered_path = asset_root / 'whole_cache_tampered_status.json'
		_dump_json(tampered_path, tampered_payload)
		tampered = EVALUATOR.load_prediction_cache(
			tampered_path,
			backend='cutie',
			audit=self.audit,
			cutie_decoder=self.make_decoder(),
			cutie_point_decoder=EVALUATOR.CUTIE_DECODER_WHOLE_ARM,
			cutie_object_schema=EVALUATOR.CUTIE_OBJECT_SCHEMA_WHOLE_ARM,
		)
		for sequence_id in roundtrip['sequences']:
			for authoritative, ignored in zip(
				roundtrip['sequences'][sequence_id]['frames'],
				tampered['sequences'][sequence_id]['frames'],
			):
				self.assertEqual(authoritative['points'], ignored['points'])
				self.assertEqual(authoritative['confidence'], ignored['confidence'])
				self.assertEqual(authoritative['lost'], ignored['lost'])
				self.assertEqual(
					authoritative['backend_object_confidence'],
					ignored['backend_object_confidence'],
				)
				self.assertEqual(
					authoritative['backend_object_lost'],
					ignored['backend_object_lost'],
				)
				self.assertTrue(all(
					ignored['decoder_diagnostics']['cached_point_lost'].values()
				))

		report = EVALUATOR.evaluate(
			annotations=self.audit_path,
			output_dir=self.root / 'report_output',
			dino_cache=self.dino_cache,
			dino_live=None,
			cutie_cache=cache_path,
			cutie_point_decoder=EVALUATOR.CUTIE_DECODER_WHOLE_ARM,
			cutie_object_schema=EVALUATOR.CUTIE_OBJECT_SCHEMA_WHOLE_ARM,
			cutie_support_path=self.support,
		)
		self.assertEqual(report['definitions']['cutie_object_schema'], {
			'name': EVALUATOR.CUTIE_OBJECT_SCHEMA_WHOLE_ARM,
			'roles': list(OBJECT_ROLES),
			'point_decoder_mode': EVALUATOR.CUTIE_DECODER_WHOLE_ARM_V2,
			'point_decoder_metadata': decoder.metadata(),
			'mask_iou_roles': ['controlled_arm', 'goal'],
			'backend_status_policy': EVALUATOR._whole_arm_backend_status_policy(),
		})

	def test_whole_cache_metadata_and_old_masks_fail_closed(self):
		decoder = self.make_decoder()
		with self.assertRaises(EVALUATOR.ContractError):
			EVALUATOR.load_prediction_cache(
				self.legacy_cutie_cache,
				backend='cutie',
				audit=self.audit,
				cutie_decoder=decoder,
				cutie_point_decoder=EVALUATOR.CUTIE_DECODER_WHOLE_ARM,
				cutie_object_schema=EVALUATOR.CUTIE_OBJECT_SCHEMA_WHOLE_ARM,
			)

		legacy_payload = json.loads(
			self.legacy_cutie_cache.read_text(encoding='utf-8')
		)
		legacy_payload['metadata'].update({
			'object_schema': EVALUATOR.CUTIE_OBJECT_SCHEMA_WHOLE_ARM,
			'object_roles': list(OBJECT_ROLES),
			'point_decoder': decoder.metadata(),
			'point_decoder_mode': EVALUATOR.CUTIE_DECODER_WHOLE_ARM_V2,
			'backend_status_policy': EVALUATOR._whole_arm_backend_status_policy(),
		})
		for sequence in legacy_payload['sequences']:
			for frame in sequence['frames']:
				frame['backend_object_confidence'] = {
					'whole_arm': 0.75, 'goal': 0.8,
				}
				frame['backend_object_lost'] = {
					'whole_arm': False, 'goal': False,
				}
				frame['confidence'] = {role: 0.5 for role in POINT_ROLES}
				frame['lost'] = {role: False for role in POINT_ROLES}
				frame['point_decoder'] = EVALUATOR.CUTIE_DECODER_WHOLE_ARM_V2
		valid_whole_payload = copy.deepcopy(legacy_payload)
		for sequence in valid_whole_payload['sequences']:
			for frame in sequence['frames']:
				# Role-correct masks are sufficient here: status validation happens
				# independently of whether this proximal-only arm decodes as lost.
				frame['masks'] = {
					'whole_arm': frame['masks']['proximal_link'],
					'goal': frame['masks']['goal'],
				}
				frame['features'] = None
		for name, mutation in (
			('old_three_masks', lambda value: None),
			('missing_decoder', lambda value: value['metadata'].pop(
				'point_decoder'
			)),
			('missing_decoder_mode', lambda value: value['metadata'].pop(
				'point_decoder_mode'
			)),
			('retired_v1_decoder_mode', lambda value: value['metadata'].__setitem__(
				'point_decoder_mode', EVALUATOR.CUTIE_DECODER_WHOLE_ARM_V1
			)),
			('missing_status_policy', lambda value: value['metadata'].pop(
				'backend_status_policy'
			)),
			('wrong_schema', lambda value: value['metadata'].__setitem__(
				'object_schema', EVALUATOR.CUTIE_OBJECT_SCHEMA_LEGACY
			)),
			('wrong_roles', lambda value: value['metadata'].__setitem__(
				'object_roles', ['goal', 'whole_arm']
			)),
			('wrong_decoder', lambda value: value['metadata']['point_decoder'].__setitem__(
				'support_annotations_sha256', '0' * 64
			)),
			('retired_v1_decoder_metadata', lambda value: value['metadata'][
				'point_decoder'
			].update({
				'format': 'visual_small_whole_arm_point_decoder_v1',
				'version': 1,
				'algorithm': 'base_connected_zhang_suen_geodesic_arc_length_v1',
			})),
			('retired_v1_frame_mode', lambda value: value['sequences'][0][
				'frames'
			][0].__setitem__(
				'point_decoder', EVALUATOR.CUTIE_DECODER_WHOLE_ARM_V1
			)),
			('missing_backend_confidence', lambda value: value['sequences'][0][
				'frames'
			][0].pop('backend_object_confidence')),
			('missing_backend_lost', lambda value: value['sequences'][0][
				'frames'
			][0].pop('backend_object_lost')),
			('bad_backend_confidence', lambda value: value['sequences'][0][
				'frames'
			][0]['backend_object_confidence'].__setitem__('whole_arm', 1.01)),
			('null_backend_confidence', lambda value: value['sequences'][0][
				'frames'
			][0]['backend_object_confidence'].__setitem__('whole_arm', None)),
			('bad_backend_lost', lambda value: value['sequences'][0][
				'frames'
			][0]['backend_object_lost'].__setitem__('goal', 'false')),
		):
			with self.subTest(mutation=name):
				payload = copy.deepcopy(
					legacy_payload if name == 'old_three_masks'
					else valid_whole_payload
				)
				mutation(payload)
				path = self.root / f'fail_closed_{name}.json'
				_dump_json(path, payload)
				with self.assertRaises(EVALUATOR.ContractError):
					EVALUATOR.load_prediction_cache(
						path,
						backend='cutie',
						audit=self.audit,
						cutie_decoder=self.make_decoder(),
						cutie_point_decoder=EVALUATOR.CUTIE_DECODER_WHOLE_ARM,
						cutie_object_schema=EVALUATOR.CUTIE_OBJECT_SCHEMA_WHOLE_ARM,
					)

	def test_whole_mask_iou_reports_only_arm_union_and_goal(self):
		root = self.root / 'mask_iou'
		root.mkdir(parents=True, exist_ok=True)
		ground_truth = np.zeros((64, 64), dtype=np.uint8)
		ground_truth[28:36, 30:42] = 1
		ground_truth[28:36, 42:54] = 2
		ground_truth[10:16, 10:16] = 3
		ground_truth_path = root / 'ground_truth.png'
		Image.fromarray(ground_truth, mode='L').save(ground_truth_path)
		whole_arm_path = root / 'whole_arm.png'
		goal_path = root / 'goal.png'
		Image.fromarray(np.isin(ground_truth, (1, 2)).astype(np.uint8) * 255).save(
			whole_arm_path
		)
		Image.fromarray((ground_truth == 3).astype(np.uint8) * 255).save(goal_path)
		ious = EVALUATOR._mask_iou(ground_truth_path, {
			'object_mask_paths': {
				'whole_arm': whole_arm_path,
				'goal': goal_path,
			},
		})
		self.assertEqual(set(ious), {'controlled_arm', 'goal'})
		self.assertEqual(ious, {'controlled_arm': 1.0, 'goal': 1.0})


if __name__ == '__main__':
	unittest.main(verbosity=2)
