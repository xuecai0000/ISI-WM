"""Independent contract for the support-calibrated whole-arm point decoder.

All inference fixtures are synthetic 64x64 binary masks.  The support fixture
contains only generated RGB frames and manual-style point labels.  No Cutie
checkpoint, adapter, evaluator, audit set, simulator, or GPU is imported.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
DECODER_PATH = ROOT / 'perception' / 'visual_small_whole_arm_decoder.py'
_SPEC = importlib.util.spec_from_file_location(
	'visual_small_whole_arm_decoder_contract_target',
	DECODER_PATH,
)
if _SPEC is None or _SPEC.loader is None:
	raise RuntimeError(f'Could not import decoder from {DECODER_PATH}')
decoder_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = decoder_module
_SPEC.loader.exec_module(decoder_module)

OBJECT_ROLES = decoder_module.OBJECT_ROLES
POINT_ROLES = decoder_module.POINT_ROLES
VisualSmallWholeArmDecoderContractError = (
	decoder_module.VisualSmallWholeArmDecoderContractError
)
VisualSmallWholeArmPointDecoder = decoder_module.VisualSmallWholeArmPointDecoder


BASE = np.asarray([31.5, 31.5], dtype=np.float64)
L1 = 12.0
L2 = 12.0
MANIFEST_SHA256 = '4' * 64
COMBINED_MANIFEST_SHA256 = 'a' * 64


def _rgb_sha256(image: np.ndarray) -> str:
	return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _make_support(
	root: Path,
	*,
	compact: bool = False,
	proximal_length_px: float = L1,
	distal_length_px: float = L2,
	distal_bend: float = 0.8,
) -> Path:
	frames = root / 'support_frames'
	frames.mkdir(parents=True, exist_ok=True)
	records = []
	for index, angle in enumerate((-2.2, -1.3, -0.4, 0.5, 1.4, 2.3)):
		image = np.empty((64, 64, 3), dtype=np.uint8)
		image[..., 0] = 30 + 13 * index
		image[..., 1] = 100 - 7 * index
		image[..., 2] = 50 + 9 * index
		image[0, 0] = (index, index + 1, index + 2)
		name = f'support_{index:02d}.png'
		Image.fromarray(image, mode='RGB').save(frames / name)
		direction_1 = np.asarray([math.cos(angle), math.sin(angle)])
		direction_2 = np.asarray([
			math.cos(angle + distal_bend),
			math.sin(angle + distal_bend),
		])
		elbow = BASE + proximal_length_px * direction_1
		tip = elbow + distal_length_px * direction_2
		records.append({
			'index': index,
			'episode': index,
			'image': f'support_frames/{name}',
			'source': f'video{85 + index % 5}.mp4',
			'active_video': f'video{85 + index % 5}.mp4',
			'image_sha256': _rgb_sha256(image),
			'video_split': 'support',
			'manifest_sha256': MANIFEST_SHA256,
			'points': {
				'base': BASE.tolist(),
				'elbow': elbow.tolist(),
				'control_tip': tip.tolist(),
				'goal': [10.0 + 5.0 * index, 12.0 + index],
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
	with path.open('w', encoding='utf-8', newline='\n') as stream:
		if compact:
			json.dump(payload, stream, separators=(',', ':'), ensure_ascii=False)
		else:
			json.dump(payload, stream, indent=2, ensure_ascii=False)
		stream.write('\n')
	return path


def _segment_mask(start_xy, end_xy, *, radius: float = 2.2) -> np.ndarray:
	start = np.asarray(start_xy, dtype=np.float64)
	end = np.asarray(end_xy, dtype=np.float64)
	yy, xx = np.mgrid[:64, :64]
	delta = end - start
	length_squared = float(np.dot(delta, delta))
	position = (
		(xx - start[0]) * delta[0] + (yy - start[1]) * delta[1]
	) / max(length_squared, 1e-12)
	position = np.clip(position, 0.0, 1.0)
	closest_x = start[0] + position * delta[0]
	closest_y = start[1] + position * delta[1]
	return (
		(xx - closest_x) ** 2 + (yy - closest_y) ** 2 <= radius ** 2
	)


def _disk_mask(center_xy, *, radius: float = 2.3) -> np.ndarray:
	center = np.asarray(center_xy, dtype=np.float64)
	yy, xx = np.mgrid[:64, :64]
	return (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius ** 2


def _geometry(first_angle: float, bend: float):
	elbow = BASE + L1 * np.asarray([
		math.cos(first_angle),
		math.sin(first_angle),
	])
	tip = elbow + L2 * np.asarray([
		math.cos(first_angle + bend),
		math.sin(first_angle + bend),
	])
	return elbow, tip


def _normal_masks(first_angle: float, bend: float):
	elbow, tip = _geometry(first_angle, bend)
	goal = np.asarray([18.5, 14.5], dtype=np.float64)
	whole_arm = np.logical_or(
		_segment_mask(BASE, elbow),
		_segment_mask(elbow, tip),
	)
	return {
		'whole_arm': whole_arm,
		'goal': _disk_mask(goal),
	}, elbow, tip, goal


def _error(point, expected) -> float:
	if point is None:
		return float('inf')
	return float(np.linalg.norm(
		np.asarray(point, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
	))


class VisualSmallWholeArmDecoderContractTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls._temporary = tempfile.TemporaryDirectory(
			prefix='visual_small_whole_arm_decoder_contract_'
		)
		cls.root = Path(cls._temporary.name)
		cls.support = _make_support(cls.root / 'support')

	@classmethod
	def tearDownClass(cls):
		cls._temporary.cleanup()

	def make_decoder(self, **kwargs):
		return VisualSmallWholeArmPointDecoder.from_support(
			self.support,
			**kwargs,
		)

	def make_decoder_with_centerline_shortfall(
		self,
		masks,
		shortfall_px,
		*,
		maximum_endpoint_shortfall_px=4.0,
	):
		"""Build a synthetic calibration with an exact requested shortfall."""
		probe = self.make_decoder(
			maximum_endpoint_shortfall_px=maximum_endpoint_shortfall_px,
		)
		centerline_length_px = probe.decode(masks).diagnostics[
			'centerline_length_px'
		]
		expected_total = centerline_length_px + float(shortfall_px)
		proximal_length_px = probe.calibration.proximal_length_px
		token = hashlib.sha256(
			f'{expected_total!r}:{maximum_endpoint_shortfall_px!r}'.encode('ascii')
		).hexdigest()[:12]
		support = _make_support(
			self.root / f'support_shortfall_{token}',
			proximal_length_px=proximal_length_px,
			distal_length_px=expected_total - proximal_length_px,
			# Fold the synthetic distal link back toward the base so even a
			# deliberately long branch fixture keeps every support point in-frame.
			distal_bend=math.pi,
		)
		return VisualSmallWholeArmPointDecoder.from_support(
			support,
			maximum_endpoint_shortfall_px=maximum_endpoint_shortfall_px,
		)

	def test_support_only_calibration_and_metadata_are_frozen(self):
		decoder = self.make_decoder()
		metadata = decoder.metadata()
		self.assertEqual(metadata['format'], 'visual_small_whole_arm_point_decoder_v2')
		self.assertEqual(metadata['version'], 2)
		self.assertEqual(
			metadata['algorithm'],
			'base_connected_zhang_suen_geodesic_arc_length_v2',
		)
		self.assertEqual(
			metadata['short_centerline_policy'],
			'bounded_shortfall_endpoint_v2',
		)
		self.assertEqual(metadata['input_roles'], ['whole_arm', 'goal'])
		self.assertEqual(metadata['base_xy'], BASE.tolist())
		self.assertAlmostEqual(metadata['L1'], L1, places=12)
		self.assertAlmostEqual(metadata['L2'], L2, places=12)
		self.assertEqual(metadata['support_records'], 6)
		self.assertEqual(
			metadata['support_annotations_sha256'],
			hashlib.sha256(self.support.read_bytes()).hexdigest(),
		)
		json.dumps(metadata, allow_nan=False)

	def test_support_raw_sha_prevents_silent_cache_calibration_change(self):
		compact = _make_support(self.root / 'support_compact', compact=True)
		first = self.make_decoder().metadata()
		second = VisualSmallWholeArmPointDecoder.from_support(compact).metadata()
		self.assertNotEqual(
			first['support_annotations_sha256'],
			second['support_annotations_sha256'],
		)
		self.assertEqual(first['base_xy'], second['base_xy'])
		self.assertEqual(first['L1'], second['L1'])
		self.assertEqual(first['L2'], second['L2'])

	def test_endpoint_shortfall_limit_must_be_strictly_less_than_L2(self):
		calibrated_L2 = self.make_decoder().calibration.distal_length_px
		just_below_L2 = math.nextafter(calibrated_L2, 0.0)
		decoder = self.make_decoder(
			maximum_endpoint_shortfall_px=just_below_L2,
		)
		self.assertEqual(
			decoder.metadata()['maximum_endpoint_shortfall_px'],
			just_below_L2,
		)
		with self.assertRaisesRegex(ValueError, 'distal link length'):
			self.make_decoder(maximum_endpoint_shortfall_px=calibrated_L2)

	def test_normal_whole_arm_recovers_points_by_arc_length(self):
		decoder = self.make_decoder()
		cases = (
			(0.0, math.pi / 2),
			(-0.7, 1.0),
			(1.35, -1.1),
			(2.7, 0.9),
		)
		for first_angle, bend in cases:
			with self.subTest(first_angle=first_angle, bend=bend):
				masks, elbow, tip, goal = _normal_masks(first_angle, bend)
				result = decoder.decode(masks)
				self.assertFalse(result.lost['elbow'], result.diagnostics)
				self.assertFalse(result.lost['control_tip'], result.diagnostics)
				self.assertFalse(result.lost['goal'], result.diagnostics)
				self.assertLessEqual(_error(result.points['elbow'], elbow), 2.0)
				self.assertLessEqual(_error(result.points['control_tip'], tip), 2.0)
				self.assertLessEqual(_error(result.points['goal'], goal), 0.75)
				self.assertEqual(result.diagnostics['failure_reasons'], [])

	def test_moderate_pose_grid_has_bounded_synthetic_error(self):
		decoder = self.make_decoder()
		for first_angle in np.linspace(-math.pi, math.pi, 9)[:-1]:
			for bend in np.linspace(-1.5, 1.5, 7):
				with self.subTest(first_angle=first_angle, bend=bend):
					masks, elbow, tip, _ = _normal_masks(
						float(first_angle), float(bend)
					)
					result = decoder.decode(masks)
					self.assertFalse(result.lost['control_tip'], result.diagnostics)
					self.assertLessEqual(
						_error(result.points['elbow'], elbow), 2.1
					)
					self.assertLessEqual(
						_error(result.points['control_tip'], tip), 1.3
					)

	def test_exactly_two_object_roles_are_required(self):
		decoder = self.make_decoder()
		masks, _, _, _ = _normal_masks(0.0, 1.0)
		for bad in (
			{'whole_arm': masks['whole_arm']},
			{**masks, 'proximal_link': masks['whole_arm']},
			{'proximal_link': masks['whole_arm'], 'goal': masks['goal']},
		):
			with self.assertRaises(VisualSmallWholeArmDecoderContractError):
				decoder.decode(bad)

	def test_mask_shape_and_binary_values_are_strict(self):
		decoder = self.make_decoder()
		masks, _, _, _ = _normal_masks(0.0, 1.0)
		bad_shape = dict(masks)
		bad_shape['whole_arm'] = np.zeros((448, 448), dtype=bool)
		with self.assertRaises(VisualSmallWholeArmDecoderContractError):
			decoder.decode(bad_shape)
		bad_value = dict(masks)
		bad_value['goal'] = masks['goal'].astype(np.float32)
		bad_value['goal'][0, 0] = 0.5
		with self.assertRaises(VisualSmallWholeArmDecoderContractError):
			decoder.decode(bad_value)

	def test_disconnected_whole_arm_fails_closed(self):
		decoder = self.make_decoder()
		masks, _, _, _ = _normal_masks(0.0, 1.0)
		masks['whole_arm'] = masks['whole_arm'].copy()
		masks['whole_arm'][4:7, 55:58] = True
		result = decoder.decode(masks)
		self.assertTrue(result.lost['elbow'])
		self.assertTrue(result.lost['control_tip'])
		self.assertIn('whole_arm_disconnected', result.diagnostics['failure_reasons'])

	def test_substantial_branch_fails_closed(self):
		decoder = self.make_decoder()
		masks, elbow, _, _ = _normal_masks(0.0, 0.9)
		branch_end = elbow + np.asarray([2.0, -10.0])
		masks['whole_arm'] = np.logical_or(
			masks['whole_arm'],
			_segment_mask(elbow, branch_end, radius=1.8),
		)
		result = decoder.decode(masks)
		self.assertTrue(result.lost['elbow'], result.diagnostics)
		self.assertTrue(result.lost['control_tip'], result.diagnostics)
		self.assertIn(
			'whole_arm_abnormal_branch',
			result.diagnostics['failure_reasons'],
		)

	def test_short_partial_arm_fails_closed(self):
		decoder = self.make_decoder()
		elbow, _ = _geometry(0.3, 1.0)
		masks = {
			'whole_arm': _segment_mask(BASE, elbow),
			'goal': _disk_mask([18.5, 14.5]),
		}
		result = decoder.decode(masks)
		self.assertTrue(result.lost['elbow'])
		self.assertTrue(result.lost['control_tip'])
		self.assertIn(
			'whole_arm_arc_length_insufficient',
			result.diagnostics['failure_reasons'],
		)

	def test_tiny_endpoint_shortfall_uses_observed_tip_without_loss(self):
		masks, _, _, _ = _normal_masks(-0.7, 1.0)
		probe = self.make_decoder().decode(masks)
		decoder = self.make_decoder_with_centerline_shortfall(masks, 0.002)
		result = decoder.decode(masks)
		self.assertAlmostEqual(
			result.diagnostics['support_total_length_px']
			- result.diagnostics['centerline_length_px'],
			0.002,
			places=9,
		)
		self.assertFalse(result.lost['elbow'], result.diagnostics)
		self.assertFalse(result.lost['control_tip'], result.diagnostics)
		self.assertLessEqual(
			_error(result.points['elbow'], probe.points['elbow']),
			1e-12,
		)
		self.assertNotIn(
			'arc_length_sampling_failed',
			result.diagnostics['failure_reasons'],
		)
		self.assertEqual(
			result.diagnostics['observed_elbow_arc_length_px'],
			result.diagnostics['support_L1_px'],
		)
		self.assertEqual(
			result.diagnostics['tip_sampling'],
			'bounded_shortfall_endpoint_v2',
		)
		self.assertTrue(
			result.diagnostics['short_centerline_policy_applied']
		)
		self.assertAlmostEqual(
			result.diagnostics['observed_tip_arc_length_px'],
			result.diagnostics['centerline_length_px'],
			places=12,
		)
		self.assertAlmostEqual(
			result.diagnostics['endpoint_shortfall_px'],
			0.002,
			places=9,
		)
		self.assertGreater(
			result.diagnostics['raw_expected_to_observed_length_ratio'],
			1.0,
		)

	def test_exact_limit_endpoint_shortfall_does_not_fail_sampling(self):
		maximum_shortfall = 4.0
		masks, _, _, _ = _normal_masks(1.35, -1.1)
		decoder = self.make_decoder_with_centerline_shortfall(
			masks,
			maximum_shortfall,
			maximum_endpoint_shortfall_px=maximum_shortfall,
		)
		result = decoder.decode(masks)
		self.assertFalse(result.lost['elbow'], result.diagnostics)
		self.assertFalse(result.lost['control_tip'], result.diagnostics)
		self.assertEqual(result.diagnostics['failure_reasons'], [])
		self.assertEqual(
			result.diagnostics['tip_sampling'],
			'bounded_shortfall_endpoint_v2',
		)
		self.assertAlmostEqual(
			result.diagnostics['endpoint_shortfall_px'],
			maximum_shortfall,
			places=9,
		)

	def test_shortfall_beyond_limit_still_fails_closed(self):
		maximum_shortfall = 4.0
		masks, _, _, _ = _normal_masks(0.0, math.pi / 2)
		decoder = self.make_decoder_with_centerline_shortfall(
			masks,
			maximum_shortfall + 1e-6,
			maximum_endpoint_shortfall_px=maximum_shortfall,
		)
		result = decoder.decode(masks)
		self.assertTrue(result.lost['elbow'])
		self.assertTrue(result.lost['control_tip'])
		self.assertIn(
			'whole_arm_arc_length_insufficient',
			result.diagnostics['failure_reasons'],
		)

	def test_short_centerline_with_abnormal_branch_still_fails_closed(self):
		masks, elbow, _, _ = _normal_masks(0.0, 0.9)
		branch_end = elbow + np.asarray([2.0, -10.0])
		masks['whole_arm'] = np.logical_or(
			masks['whole_arm'],
			_segment_mask(elbow, branch_end, radius=1.8),
		)
		decoder = self.make_decoder_with_centerline_shortfall(masks, 0.002)
		result = decoder.decode(masks)
		self.assertTrue(result.lost['elbow'], result.diagnostics)
		self.assertTrue(result.lost['control_tip'], result.diagnostics)
		self.assertIn(
			'whole_arm_abnormal_branch',
			result.diagnostics['failure_reasons'],
		)
		self.assertFalse(
			result.diagnostics['short_centerline_policy_applied']
		)

	def test_overlong_tail_cannot_move_tip_beyond_support_length(self):
		decoder = self.make_decoder()
		expected_elbow = BASE + np.asarray([L1, 0.0])
		expected_tip = BASE + np.asarray([L1 + L2, 0.0])
		bounded_tail = decoder.decode({
			'whole_arm': _segment_mask(
				BASE, BASE + np.asarray([28.0, 0.0])
			),
			'goal': _disk_mask([18.5, 14.5]),
		})
		self.assertFalse(bounded_tail.lost['control_tip'], bounded_tail.diagnostics)
		self.assertGreater(
			bounded_tail.diagnostics['centerline_length_px'],
			L1 + L2 + 4.0,
		)
		self.assertLessEqual(
			_error(bounded_tail.points['elbow'], expected_elbow), 0.75
		)
		self.assertLessEqual(
			_error(bounded_tail.points['control_tip'], expected_tip), 0.75
		)
		self.assertEqual(
			bounded_tail.diagnostics['tip_sampling'],
			'support_total_arc_length_on_validated_centerline',
		)

		excessive_tail = decoder.decode({
			'whole_arm': _segment_mask(
				BASE, BASE + np.asarray([30.0, 0.0])
			),
			'goal': _disk_mask([18.5, 14.5]),
		})
		self.assertTrue(excessive_tail.lost['control_tip'])
		self.assertIn(
			'whole_arm_arc_length_excessive',
			excessive_tail.diagnostics['failure_reasons'],
		)

	def test_missing_base_attachment_fails_closed(self):
		decoder = self.make_decoder()
		masks = {
			'whole_arm': _segment_mask([3.0, 3.0], [25.0, 3.0]),
			'goal': _disk_mask([18.5, 14.5]),
		}
		result = decoder.decode(masks)
		self.assertTrue(result.lost['elbow'])
		self.assertTrue(result.lost['control_tip'])
		self.assertIn(
			'whole_arm_not_base_attached',
			result.diagnostics['failure_reasons'],
		)

	def test_inference_api_has_no_privileged_or_rgb_inputs(self):
		signature = inspect.signature(VisualSmallWholeArmPointDecoder.decode)
		self.assertEqual(tuple(signature.parameters), ('self', 'object_masks'))
		source = inspect.getsource(VisualSmallWholeArmPointDecoder.decode).lower()
		for forbidden in (
			'physics', 'qpos', 'qvel', 'simulator', 'ground_truth',
			'audit', 'reward', 'image', 'rgb',
		):
			self.assertNotIn(forbidden, source)

	def test_decode_is_deterministic_stateless_and_json_safe(self):
		decoder = self.make_decoder()
		masks, _, _, _ = _normal_masks(-0.8, 1.1)
		first = decoder.decode(masks).as_dict()
		decoder.reset()
		second = decoder.decode({
			role: np.array(mask, copy=True) for role, mask in masks.items()
		}).as_dict()
		self.assertEqual(first, second)
		json.dumps(first, allow_nan=False)
		for role in POINT_ROLES:
			self.assertGreaterEqual(first['point_confidence'][role], 0.0)
			self.assertLessEqual(first['point_confidence'][role], 1.0)


if __name__ == '__main__':
	unittest.main(verbosity=2)
