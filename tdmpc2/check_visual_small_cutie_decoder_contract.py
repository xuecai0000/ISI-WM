"""Dependency-free contract for the support-calibrated Cutie point decoder.

The fixtures are synthetic 64x64 masks and a six-frame RGB-only support pack.
No Cutie checkpoint, environment, simulator, audit labels, or GPU is needed.
The contract deliberately covers the observed same-colour link-role collapse as
well as cache/live serialization parity and a 96-frame temporal stress sequence.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
	sys.path.insert(0, str(ROOT.parent))


DECODER_PATH = ROOT / 'perception' / 'visual_small_cutie_decoder.py'
_SPEC = importlib.util.spec_from_file_location(
	'visual_small_cutie_decoder_contract_target', DECODER_PATH
)
if _SPEC is None or _SPEC.loader is None:
	raise RuntimeError(f'Could not import decoder from {DECODER_PATH}')
decoder_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = decoder_module
_SPEC.loader.exec_module(decoder_module)
OBJECT_ROLES = decoder_module.OBJECT_ROLES
POINT_ROLES = decoder_module.POINT_ROLES
VisualSmallCutieDecoderContractError = (
	decoder_module.VisualSmallCutieDecoderContractError
)
VisualSmallCutiePointDecoder = decoder_module.VisualSmallCutiePointDecoder

AB_CONTRACT_PATH = ROOT / 'check_visual_small_perception_ab_contract.py'
_AB_SPEC = importlib.util.spec_from_file_location(
	'visual_small_perception_ab_fixture_provider', AB_CONTRACT_PATH
)
if _AB_SPEC is None or _AB_SPEC.loader is None:
	raise RuntimeError(f'Could not import A/B fixtures from {AB_CONTRACT_PATH}')
ab_contract = importlib.util.module_from_spec(_AB_SPEC)
sys.modules[_AB_SPEC.name] = ab_contract
_AB_SPEC.loader.exec_module(ab_contract)
EVALUATOR = ab_contract.EVALUATOR


BASE = np.asarray([31.5, 31.5], dtype=np.float64)
L1 = 12.0
L2 = 12.0
MANIFEST_SHA256 = '487f0a19166a61375140bce2024ddffbc4e0c84730534e3e392d7115029fde9c'
COMBINED_MANIFEST_SHA256 = (
	'3afb66d6db1c85771c62b7e75bcfccb0539613c4876071f1d5bb4f68626e9d99'
)
FORBIDDEN_INFERENCE_NAMES = {
	'physics', 'qpos', 'qvel', 'geom_xpos', 'site_xpos', 'body_xpos',
	'simulator_state', 'privileged_state', 'ground_truth', 'reward', 'audit',
}


def _raw_rgb_sha256(image: np.ndarray) -> str:
	return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _dump_json(path: Path, payload, *, compact: bool = False) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', encoding='utf-8', newline='\n') as stream:
		if compact:
			json.dump(payload, stream, separators=(',', ':'), ensure_ascii=False)
		else:
			json.dump(payload, stream, indent=2, ensure_ascii=False)
		stream.write('\n')


def _make_support(root: Path, *, compact: bool = False) -> Path:
	frames = root / 'support_frames'
	frames.mkdir(parents=True, exist_ok=True)
	records = []
	angles = (-2.4, -1.6, -0.7, 0.2, 1.1, 2.2)
	for index, angle in enumerate(angles):
		image = np.empty((64, 64, 3), dtype=np.uint8)
		image[..., 0] = 20 + 11 * index
		image[..., 1] = 80 + 7 * index
		image[..., 2] = 130 - 9 * index
		image[0, 0] = (index, index + 1, index + 2)
		name = f'support_{index:02d}.png'
		Image.fromarray(image, mode='RGB').save(frames / name)
		direction_1 = np.asarray([math.cos(angle), math.sin(angle)])
		direction_2 = np.asarray([
			math.cos(angle + 0.65), math.sin(angle + 0.65)
		])
		elbow = BASE + L1 * direction_1
		tip = elbow + L2 * direction_2
		goal = np.asarray([10.0 + 5.0 * index, 12.0 + 2.0 * (index % 3)])
		records.append({
			'index': index,
			'episode': index,
			'image': f'support_frames/{name}',
			'source': f'video{85 + (index % 5)}.mp4',
			'active_video': f'video{85 + (index % 5)}.mp4',
			'image_sha256': _raw_rgb_sha256(image),
			'video_split': 'support',
			'manifest_sha256': MANIFEST_SHA256,
			'points': {
				'base': BASE.tolist(),
				'elbow': elbow.tolist(),
				'control_tip': tip.tolist(),
				'goal': goal.tolist(),
			},
		})
	payload = {
		'format': 'few_shot_task_anchor_annotations_v1',
		'coordinate_convention': '[x, y] in the original 64x64 RGB frame',
		'roles': list(POINT_ROLES),
		'collection': {
			'task': 'reacher-visual-small',
			'observation': 'rgb',
			'split': 'support',
			'label_policy': 'manual_rgb_only',
			'episodes': 6,
			'manifest_sha256': MANIFEST_SHA256,
			'combined_manifest_sha256': COMBINED_MANIFEST_SHA256,
		},
		'records': records,
	}
	path = root / ('annotations.compact.json' if compact else 'annotations.json')
	_dump_json(path, payload, compact=compact)
	return path


def _segment_mask(start_xy, end_xy, *, radius: float = 2.0) -> np.ndarray:
	start = np.asarray(start_xy, dtype=np.float64)
	end = np.asarray(end_xy, dtype=np.float64)
	yy, xx = np.mgrid[:64, :64]
	delta = end - start
	length_squared = float(np.dot(delta, delta))
	if length_squared <= 0.0:
		return (xx - start[0]) ** 2 + (yy - start[1]) ** 2 <= radius ** 2
	position = (
		(xx - start[0]) * delta[0] + (yy - start[1]) * delta[1]
	) / length_squared
	position = np.clip(position, 0.0, 1.0)
	closest_x = start[0] + position * delta[0]
	closest_y = start[1] + position * delta[1]
	return (xx - closest_x) ** 2 + (yy - closest_y) ** 2 <= radius ** 2


def _disk_mask(center_xy, *, radius: float = 2.5) -> np.ndarray:
	center = np.asarray(center_xy, dtype=np.float64)
	yy, xx = np.mgrid[:64, :64]
	return (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius ** 2


def _arm_geometry(first_angle: float = 0.0, bend: float = math.pi / 2):
	elbow = BASE + L1 * np.asarray([
		math.cos(first_angle), math.sin(first_angle)
	])
	tip = elbow + L2 * np.asarray([
		math.cos(first_angle + bend), math.sin(first_angle + bend)
	])
	return elbow, tip


def _normal_masks(first_angle: float = 0.0, bend: float = math.pi / 2):
	elbow, tip = _arm_geometry(first_angle, bend)
	goal = np.asarray([18.0, 15.0], dtype=np.float64)
	return {
		'proximal_link': _segment_mask(BASE, elbow),
		'distal_link': _segment_mask(elbow, tip),
		'goal': _disk_mask(goal),
	}, elbow, tip, goal


def _point_error(point, target) -> float:
	if point is None:
		return float('inf')
	return float(np.linalg.norm(
		np.asarray(point, dtype=np.float64) - np.asarray(target, dtype=np.float64)
	))


def _all_finite_probabilities(values) -> bool:
	return all(
		isinstance(value, (int, float, np.floating))
		and math.isfinite(float(value))
		and 0.0 <= float(value) <= 1.0
		for value in values
	)


class VisualSmallCutieDecoderContractTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls._temporary = tempfile.TemporaryDirectory(
			prefix='visual_small_cutie_decoder_contract_'
		)
		cls.root = Path(cls._temporary.name)
		cls.support_path = _make_support(cls.root / 'support')
		cls.integration_root = cls.root / 'evaluator_integration'
		cls.audit_path, cls.audit_payload = ab_contract._make_audit(
			cls.integration_root
		)
		cls.dino_cache = ab_contract._make_cache(
			cls.integration_root,
			audit_path=cls.audit_path,
			audit_payload=cls.audit_payload,
			backend='dino',
			offset_px=1.0,
			name='decoder_contract_dino',
		)
		cls.legacy_cutie_cache = ab_contract._make_cache(
			cls.integration_root,
			audit_path=cls.audit_path,
			audit_payload=cls.audit_payload,
			backend='cutie',
			offset_px=0.1,
			name='decoder_contract_cutie',
		)

	@classmethod
	def tearDownClass(cls):
		cls._temporary.cleanup()

	def make_decoder(self, **kwargs) -> VisualSmallCutiePointDecoder:
		return VisualSmallCutiePointDecoder.from_support(
			self.support_path, **kwargs
		)

	def test_support_only_calibration_sha_lengths_and_bands_are_frozen(self):
		decoder = self.make_decoder()
		metadata = decoder.metadata()
		self.assertEqual(
			metadata['support_annotations_sha256'],
			hashlib.sha256(self.support_path.read_bytes()).hexdigest(),
		)
		self.assertAlmostEqual(metadata['base'][0], BASE[0], places=12)
		self.assertAlmostEqual(metadata['base'][1], BASE[1], places=12)
		self.assertAlmostEqual(metadata['L1'], L1, places=10)
		self.assertAlmostEqual(metadata['L2'], L2, places=10)
		self.assertEqual(metadata['support_records'], 6)
		self.assertEqual(metadata['source_resolution'], [64, 64])
		self.assertEqual(metadata['inner_lower'], 0.25)
		self.assertEqual(metadata['inner_upper'], 0.60)
		self.assertGreaterEqual(
			metadata['proximal_coverage_threshold'], metadata['inner_upper']
		)
		json.dumps(metadata, allow_nan=False)

		compact_path = _make_support(self.root / 'compact_support', compact=True)
		compact = VisualSmallCutiePointDecoder.from_support(compact_path)
		self.assertAlmostEqual(compact.metadata()['L1'], metadata['L1'], places=12)
		self.assertNotEqual(
			compact.metadata()['support_annotations_sha256'],
			metadata['support_annotations_sha256'],
		)

	def test_support_pack_rejects_privileged_fields_and_tampered_rgb_evidence(self):
		payload = json.loads(self.support_path.read_text(encoding='utf-8'))
		for index, forbidden in enumerate((
			{'physics': {'qpos': [0.0]}},
			{'state': [0.0, 1.0]},
			{'action': [0.0, 0.0]},
			{'reward': 1.0},
			{'ground_truth': {'elbow': [1.0, 2.0]}},
		)):
			with self.subTest(forbidden=next(iter(forbidden))):
				value = copy.deepcopy(payload)
				value.update(forbidden)
				# Keep the mutated JSON beside the verified support_frames so a
				# missing privileged-field rejection cannot be masked by path failure.
				path = self.support_path.parent / f'forbidden_support_{index}.json'
				_dump_json(path, value)
				with self.assertRaises(VisualSmallCutieDecoderContractError):
					VisualSmallCutiePointDecoder.from_support(path)

		tampered_path = _make_support(self.root / 'tampered_support')
		tampered_payload = json.loads(tampered_path.read_text(encoding='utf-8'))
		first_image = tampered_path.parent / tampered_payload['records'][0]['image']
		Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8)).save(first_image)
		with self.assertRaises(VisualSmallCutieDecoderContractError):
			VisualSmallCutiePointDecoder.from_support(tampered_path)

	def test_inference_interface_has_no_rgb_gt_audit_or_physics_channel(self):
		signature = inspect.signature(VisualSmallCutiePointDecoder.decode)
		self.assertEqual(tuple(signature.parameters), ('self', 'object_masks'))
		tree = ast.parse(inspect.getsource(decoder_module))
		used_names = set()
		for node in ast.walk(tree):
			if isinstance(node, ast.Name):
				used_names.add(node.id.lower())
			elif isinstance(node, ast.Attribute):
				used_names.add(node.attr.lower())
		self.assertFalse(used_names.intersection(FORBIDDEN_INFERENCE_NAMES))
		imports = {
			alias.name
			for node in ast.walk(tree)
			if isinstance(node, ast.Import)
			for alias in node.names
		}
		imports.update(
			node.module or ''
			for node in ast.walk(tree)
			if isinstance(node, ast.ImportFrom)
		)
		self.assertFalse(any(
			name.startswith(('dm_control', 'mujoco', 'gym', 'tdmpc2.envs'))
			for name in imports
		))

		decoder = self.make_decoder()
		masks, _, _, _ = _normal_masks()
		with (
			mock.patch.object(Path, 'read_bytes', side_effect=AssertionError(
				'decode must not read files'
			)),
			mock.patch.object(Image, 'open', side_effect=AssertionError(
				'decode must not read source pixels'
			)),
		):
			result = decoder.decode(masks)
		self.assertFalse(any(result.lost.values()))

	def test_normal_two_masks_preserve_legacy_point_accuracy(self):
		decoder = self.make_decoder()
		masks, elbow, tip, goal = _normal_masks()
		original = {role: mask.copy() for role, mask in masks.items()}
		result = decoder.decode(dict(reversed(tuple(masks.items()))))
		self.assertLessEqual(_point_error(result.points['base'], BASE), 1e-12)
		self.assertLessEqual(_point_error(result.points['elbow'], elbow), 1.0)
		self.assertLessEqual(_point_error(result.points['control_tip'], tip), 1.5)
		self.assertLessEqual(_point_error(result.points['goal'], goal), 0.75)
		self.assertFalse(any(result.lost.values()))
		self.assertTrue(_all_finite_probabilities(result.point_confidence.values()))
		self.assertTrue(0.0 <= result.geometry_confidence <= 1.0)
		for role in OBJECT_ROLES:
			np.testing.assert_array_equal(masks[role], original[role])

	def test_role_collapse_proximal_swallowing_arm_recovers_elbow_and_tip(self):
		decoder = self.make_decoder()
		masks, elbow, tip, _ = _normal_masks()
		masks['proximal_link'] = np.logical_or(
			masks['proximal_link'], masks['distal_link']
		)
		fragment_start = elbow + 0.55 * (tip - elbow)
		masks['distal_link'] = _segment_mask(fragment_start, tip)
		result = decoder.decode(masks)
		self.assertLessEqual(_point_error(result.points['elbow'], elbow), 1.25)
		self.assertLessEqual(_point_error(result.points['control_tip'], tip), 1.5)
		self.assertFalse(result.lost['elbow'])
		self.assertFalse(result.lost['control_tip'])
		self.assertTrue(result.diagnostics['role_collapse'])

	def test_short_proximal_uses_base_attached_union(self):
		decoder = self.make_decoder()
		masks, elbow, tip, _ = _normal_masks()
		first = BASE + 0.35 * (elbow - BASE)
		overlap = BASE + 0.25 * (elbow - BASE)
		masks['proximal_link'] = _segment_mask(BASE, first)
		masks['distal_link'] = np.logical_or(
			_segment_mask(overlap, elbow), _segment_mask(elbow, tip)
		)
		result = decoder.decode(masks)
		self.assertEqual(
			result.diagnostics['source_used'], 'proximal_or_distal_union'
		)
		self.assertLessEqual(_point_error(result.points['elbow'], elbow), 1.25)
		self.assertFalse(result.lost['elbow'])

	def test_base_attached_component_beats_nearer_speck_and_larger_blob(self):
		decoder = self.make_decoder()
		masks, elbow, _, _ = _normal_masks()
		proximal = masks['proximal_link'].copy()
		# Detach the real arm by one blank column while keeping it within the
		# calibrated attachment radius. A one-pixel base speck is closer, and a
		# far background component is larger; neither may win over the attached arm.
		proximal[:, :33] = False
		proximal[31, 31] = True
		proximal[2:15, 2:15] = True
		masks['proximal_link'] = proximal
		result = decoder.decode(masks)
		self.assertLessEqual(_point_error(result.points['elbow'], elbow), 1.5)
		selected = result.diagnostics['proximal_component']
		self.assertTrue(selected['base_attached'])
		self.assertGreater(selected['component_pixels'], 10)

	def test_no_base_attached_arm_fails_closed_instead_of_confident_hallucination(self):
		decoder = self.make_decoder()
		masks, _, _, _ = _normal_masks()
		masks['proximal_link'] = _segment_mask([38.0, 31.5], [45.0, 31.5])
		masks['distal_link'] = np.zeros((64, 64), dtype=bool)
		result = decoder.decode(masks)
		self.assertIsNone(result.points['elbow'])
		self.assertIsNone(result.points['control_tip'])
		self.assertTrue(result.lost['elbow'])
		self.assertTrue(result.lost['control_tip'])
		self.assertEqual(result.point_confidence['elbow'], 0.0)
		self.assertEqual(result.point_confidence['control_tip'], 0.0)

	def test_bad_link_length_and_folded_ambiguity_cannot_be_high_confidence(self):
		decoder = self.make_decoder()
		masks, elbow, _, _ = _normal_masks()
		long_tip = elbow + np.asarray([0.0, 22.0])
		masks['distal_link'] = _segment_mask(elbow, long_tip)
		length_bad = decoder.decode(masks)
		self.assertGreater(
			length_bad.diagnostics['distal_endpoint']['tip_length_residual_px'],
			5.0,
		)
		self.assertLess(length_bad.point_confidence['control_tip'], 0.5)

		# The second link folds back through the first-link inner annulus while
		# Cutie's proximal role contains both. If geometry cannot resolve this
		# visual ambiguity it must fail closed, never emit a confident wrong point.
		# A nearly folded-back second link ends close to the base, so its pixels
		# cross the same radial band as the true proximal direction.
		folded_tip = BASE + np.asarray([0.0, -4.0])
		folded = {
			'proximal_link': np.logical_or(
				_segment_mask(BASE, elbow), _segment_mask(elbow, folded_tip)
			),
			'distal_link': _segment_mask(elbow, folded_tip),
			'goal': masks['goal'],
		}
		ambiguous = decoder.decode(folded)
		elbow_error = _point_error(ambiguous.points['elbow'], elbow)
		if elbow_error > 4.0:
			self.assertTrue(
				ambiguous.lost['elbow']
				or ambiguous.point_confidence['elbow'] < 0.5
			)

	def test_invalid_masks_and_parameters_fail_before_decoding(self):
		decoder = self.make_decoder()
		masks, _, _, _ = _normal_masks()
		invalid_masks = []
		missing = dict(masks)
		missing.pop('goal')
		invalid_masks.append(missing)
		wrong_shape = dict(masks)
		wrong_shape['goal'] = np.zeros((448, 448), dtype=bool)
		invalid_masks.append(wrong_shape)
		non_finite = dict(masks)
		non_finite['goal'] = np.full((64, 64), np.nan)
		invalid_masks.append(non_finite)
		non_binary = dict(masks)
		non_binary['goal'] = np.full((64, 64), -1, dtype=np.int8)
		invalid_masks.append(non_binary)
		for index, value in enumerate(invalid_masks):
			with self.subTest(mask_case=index):
				with self.assertRaises(VisualSmallCutieDecoderContractError):
					decoder.decode(value)

		invalid_parameters = (
			{'inner_lower': 0.0},
			{'inner_lower': 0.7, 'inner_upper': 0.6},
			{'inner_upper': 1.01},
			{'inner_upper': 0.60, 'proximal_coverage_threshold': 0.59},
			{'base_attachment_radius_fraction': 0.0},
			{'base_attachment_radius_min_px': 0.0},
			{'distal_endpoint_percentile': 0.0},
		)
		for kwargs in invalid_parameters:
			with self.subTest(parameters=kwargs):
				with self.assertRaises(ValueError):
					VisualSmallCutiePointDecoder(
						decoder.calibration, **kwargs
					)

	def test_declared_safe_inner_bands_all_recover_role_collapse(self):
		masks, elbow, tip, _ = _normal_masks()
		masks['proximal_link'] = np.logical_or(
			masks['proximal_link'], masks['distal_link']
		)
		masks['distal_link'] = _segment_mask(elbow + 0.5 * (tip - elbow), tip)
		for lower, upper in ((0.20, 0.50), (0.25, 0.60), (0.30, 0.65), (0.40, 0.75)):
			with self.subTest(inner=(lower, upper)):
				decoder = self.make_decoder(
					inner_lower=lower,
					inner_upper=upper,
					proximal_coverage_threshold=upper,
				)
				result = decoder.decode(masks)
				self.assertLessEqual(
					_point_error(result.points['elbow'], elbow), 1.75
				)

	def test_png_cache_and_live_masks_decode_identically(self):
		live_masks, _, _, _ = _normal_masks(first_angle=0.35, bend=0.8)
		cache_root = self.root / 'cache_masks'
		cache_root.mkdir(exist_ok=True)
		cached_masks = {}
		for role, mask in live_masks.items():
			path = cache_root / f'{role}.png'
			Image.fromarray(mask.astype(np.uint8) * 255, mode='L').save(path)
			cached_masks[role] = np.asarray(Image.open(path).convert('L')) > 0
		live_decoder = self.make_decoder()
		cache_decoder = self.make_decoder()
		live_decoder.reset()
		cache_decoder.reset()
		live = live_decoder.decode(live_masks).as_dict()
		cached = cache_decoder.decode(cached_masks).as_dict()
		self.assertEqual(live, cached)
		self.assertEqual(live_decoder.metadata(), cache_decoder.metadata())

	def test_96_frame_continuity_determinism_reset_and_no_imputation(self):
		decoder = self.make_decoder()
		serialized_first_run = []
		maximum_elbow_jump = 0.0
		maximum_tip_jump = 0.0
		for episode in range(6):
			decoder.reset()
			previous_elbow = None
			previous_tip = None
			for t in range(16):
				angle = -2.5 + episode * 0.8 + t * 0.018
				masks, elbow, tip, _ = _normal_masks(angle, bend=0.72)
				result = decoder.decode(masks)
				self.assertFalse(result.lost['elbow'])
				self.assertFalse(result.lost['control_tip'])
				self.assertLessEqual(_point_error(result.points['elbow'], elbow), 1.5)
				self.assertLessEqual(_point_error(result.points['control_tip'], tip), 1.75)
				current_elbow = np.asarray(result.points['elbow'])
				current_tip = np.asarray(result.points['control_tip'])
				if previous_elbow is not None:
					maximum_elbow_jump = max(
						maximum_elbow_jump,
						float(np.linalg.norm(current_elbow - previous_elbow)),
					)
					maximum_tip_jump = max(
						maximum_tip_jump,
						float(np.linalg.norm(current_tip - previous_tip)),
					)
				previous_elbow = current_elbow
				previous_tip = current_tip
				serialized_first_run.append(result.as_dict())
		self.assertEqual(len(serialized_first_run), 96)
		self.assertLessEqual(maximum_elbow_jump, 2.0)
		self.assertLessEqual(maximum_tip_jump, 2.5)

		second_decoder = self.make_decoder()
		serialized_second_run = []
		for episode in range(6):
			second_decoder.reset()
			for t in range(16):
				angle = -2.5 + episode * 0.8 + t * 0.018
				masks, _, _, _ = _normal_masks(angle, bend=0.72)
				serialized_second_run.append(second_decoder.decode(masks).as_dict())
		self.assertEqual(serialized_first_run, serialized_second_run)

		good_masks, _, _, _ = _normal_masks()
		bad_masks = {
			'proximal_link': _segment_mask([38.0, 31.5], [45.0, 31.5]),
			'distal_link': np.zeros((64, 64), dtype=bool),
			'goal': good_masks['goal'],
		}
		second_decoder.reset()
		first = second_decoder.decode(good_masks).as_dict()
		bad = second_decoder.decode(bad_masks)
		third = second_decoder.decode(good_masks).as_dict()
		self.assertTrue(bad.lost['elbow'])
		self.assertIsNone(bad.points['elbow'])
		self.assertEqual(first, third)

	def test_evaluator_cli_accepts_cached_masks_with_explicit_support_decoder(self):
		args = EVALUATOR.parse_args([
			'--annotations', str(self.audit_path),
			'--output-dir', str(self.root / 'cli_output'),
			'--dino-cache', str(self.dino_cache),
			'--cutie-cache', str(self.legacy_cutie_cache),
			'--cutie-support', str(self.support_path),
			'--cutie-point-decoder', EVALUATOR.CUTIE_DECODER_KINEMATIC,
		])
		self.assertEqual(args.cutie_cache, self.legacy_cutie_cache)
		self.assertEqual(args.cutie_support, self.support_path)
		self.assertEqual(
			args.cutie_point_decoder, EVALUATOR.CUTIE_DECODER_KINEMATIC
		)

	def test_legacy_cache_is_explicitly_redecoded_and_ignores_cached_points(self):
		audit = EVALUATOR.load_audit(self.audit_path)
		decoder = self.make_decoder()
		baseline = EVALUATOR.load_prediction_cache(
			self.legacy_cutie_cache,
			backend='cutie',
			audit=audit,
			cutie_decoder=decoder,
		)
		self.assertTrue(baseline['metadata']['legacy_cache_redecoded'])
		self.assertEqual(
			baseline['metadata']['point_decoder'], decoder.metadata()
		)
		self.assertEqual(
			baseline['metadata']['point_decoder']['support_annotations_sha256'],
			hashlib.sha256(self.support_path.read_bytes()).hexdigest(),
		)

		payload = json.loads(self.legacy_cutie_cache.read_text(encoding='utf-8'))
		for sequence in payload['sequences']:
			for frame in sequence['frames']:
				frame['points'] = {
					role: [63.0, 0.0] for role in POINT_ROLES
				}
				frame['confidence'] = {
					role: 0.0 for role in OBJECT_ROLES
				}
				frame['lost'] = {
					role: True for role in OBJECT_ROLES
				}
		tampered_path = self.integration_root / 'tampered_cached_fields_cutie.json'
		_dump_json(tampered_path, payload)
		tampered = EVALUATOR.load_prediction_cache(
			tampered_path,
			backend='cutie',
			audit=audit,
			cutie_decoder=self.make_decoder(),
		)
		for sequence_id in baseline['sequences']:
			left_frames = baseline['sequences'][sequence_id]['frames']
			right_frames = tampered['sequences'][sequence_id]['frames']
			for left, right in zip(left_frames, right_frames):
				self.assertEqual(left['points'], right['points'])
				self.assertEqual(left['confidence'], right['confidence'])
				self.assertEqual(left['lost'], right['lost'])
				self.assertEqual(
					left['point_decoder'], EVALUATOR.CUTIE_DECODER_KINEMATIC
				)

	def test_real_legacy_point_status_cache_and_new_cache_roundtrip(self):
		audit = EVALUATOR.load_audit(self.audit_path)
		legacy_predictions = EVALUATOR.load_prediction_cache(
			self.legacy_cutie_cache,
			backend='cutie',
			audit=audit,
		)
		# Historical live caches passed normalized predictions through the cache
		# writer. Consequently confidence/lost contain the four task point roles,
		# not the three raw Cutie object roles used by the adapter fixture above.
		legacy_payload = EVALUATOR._cache_json_value(
			legacy_predictions, audit, self.integration_root
		)
		legacy_payload['metadata'].pop('point_decoder', None)
		legacy_payload['metadata'].pop('legacy_cache_redecoded', None)
		cached_confidence = {role: 0.0 for role in POINT_ROLES}
		cached_lost = {role: True for role in POINT_ROLES}
		for sequence in legacy_payload['sequences']:
			for frame in sequence['frames']:
				self.assertEqual(tuple(frame['confidence']), POINT_ROLES)
				self.assertEqual(tuple(frame['lost']), POINT_ROLES)
				# Valid but deliberately contradictory cached status must be checked
				# and retained for diagnosis, never overwrite mask-only decoding.
				frame['confidence'] = dict(cached_confidence)
				frame['lost'] = dict(cached_lost)
		legacy_path = self.integration_root / 'real_legacy_writer_cutie.json'
		_dump_json(legacy_path, legacy_payload)

		baseline = EVALUATOR.load_prediction_cache(
			self.legacy_cutie_cache,
			backend='cutie',
			audit=audit,
			cutie_decoder=self.make_decoder(),
		)
		redecoded = EVALUATOR.load_prediction_cache(
			legacy_path,
			backend='cutie',
			audit=audit,
			cutie_decoder=self.make_decoder(),
		)
		self.assertTrue(redecoded['metadata']['legacy_cache_redecoded'])
		for sequence_id in baseline['sequences']:
			left_frames = baseline['sequences'][sequence_id]['frames']
			right_frames = redecoded['sequences'][sequence_id]['frames']
			for left, right in zip(left_frames, right_frames):
				self.assertEqual(left['points'], right['points'])
				self.assertEqual(left['confidence'], right['confidence'])
				self.assertEqual(left['lost'], right['lost'])
				self.assertEqual(
					right['decoder_diagnostics'][
						'cached_point_confidence'
					],
					cached_confidence,
				)
				self.assertEqual(
					right['decoder_diagnostics']['cached_point_lost'],
					cached_lost,
				)

		for field, role, invalid in (
			('confidence', 'elbow', 1.01),
			('lost', 'control_tip', 'true'),
		):
			with self.subTest(invalid_cached_status=(field, role)):
				invalid_payload = copy.deepcopy(legacy_payload)
				invalid_payload['sequences'][0]['frames'][0][field][role] = invalid
				invalid_path = self.integration_root / f'invalid_{field}_cutie.json'
				_dump_json(invalid_path, invalid_payload)
				with self.assertRaises(EVALUATOR.ContractError):
					EVALUATOR.load_prediction_cache(
						invalid_path,
						backend='cutie',
						audit=audit,
						cutie_decoder=self.make_decoder(),
					)

		new_payload = EVALUATOR._cache_json_value(
			redecoded, audit, self.integration_root
		)
		for sequence in new_payload['sequences']:
			for frame in sequence['frames']:
				self.assertEqual(tuple(frame['confidence']), POINT_ROLES)
				self.assertEqual(tuple(frame['lost']), POINT_ROLES)
		new_path = self.integration_root / 'kinematic_roundtrip_cutie.json'
		_dump_json(new_path, new_payload)
		roundtripped = EVALUATOR.load_prediction_cache(
			new_path,
			backend='cutie',
			audit=audit,
			cutie_decoder=self.make_decoder(),
		)
		self.assertFalse(roundtripped['metadata']['legacy_cache_redecoded'])
		self.assertEqual(
			roundtripped['metadata']['point_decoder'],
			redecoded['metadata']['point_decoder'],
		)
		for sequence_id in redecoded['sequences']:
			left_frames = redecoded['sequences'][sequence_id]['frames']
			right_frames = roundtripped['sequences'][sequence_id]['frames']
			for left, right in zip(left_frames, right_frames):
				self.assertEqual(left['points'], right['points'])
				self.assertEqual(left['confidence'], right['confidence'])
				self.assertEqual(left['lost'], right['lost'])

	def test_new_cache_decoder_metadata_mismatch_fails_closed(self):
		audit = EVALUATOR.load_audit(self.audit_path)
		decoder = self.make_decoder()
		payload = json.loads(self.legacy_cutie_cache.read_text(encoding='utf-8'))
		payload['metadata']['point_decoder'] = decoder.metadata()
		matching_path = self.integration_root / 'matching_decoder_metadata_cutie.json'
		_dump_json(matching_path, payload)
		matching = EVALUATOR.load_prediction_cache(
			matching_path,
			backend='cutie',
			audit=audit,
			cutie_decoder=self.make_decoder(),
		)
		self.assertFalse(matching['metadata']['legacy_cache_redecoded'])

		mismatched = copy.deepcopy(payload)
		mismatched['metadata']['point_decoder'][
			'support_annotations_sha256'
		] = '0' * 64
		mismatch_path = self.integration_root / 'mismatched_decoder_metadata_cutie.json'
		_dump_json(mismatch_path, mismatched)
		with self.assertRaises(EVALUATOR.ContractError):
			EVALUATOR.load_prediction_cache(
				mismatch_path,
				backend='cutie',
				audit=audit,
				cutie_decoder=self.make_decoder(),
			)

	def test_default_legacy_cache_decoder_remains_bitwise_unchanged(self):
		audit = EVALUATOR.load_audit(self.audit_path)
		loaded = EVALUATOR.load_prediction_cache(
			self.legacy_cutie_cache,
			backend='cutie',
			audit=audit,
		)
		self.assertFalse(loaded['metadata']['legacy_cache_redecoded'])
		self.assertEqual(
			loaded['metadata']['point_decoder'],
			EVALUATOR._legacy_cutie_decoder_metadata(),
		)
		for sequence_id, sequence in loaded['sequences'].items():
			for frame in sequence['frames']:
				masks = EVALUATOR._read_object_masks(
					frame['object_mask_paths'], (64, 64, 3)
				)
				expected = EVALUATOR.points_from_cutie_masks(masks)
				self.assertEqual(frame['points'], expected)
				self.assertEqual(
					frame['point_decoder'], EVALUATOR.CUTIE_DECODER_LEGACY
				)


if __name__ == '__main__':
	unittest.main(verbosity=2)
