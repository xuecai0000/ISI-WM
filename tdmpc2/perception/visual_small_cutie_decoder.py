"""Support-calibrated Cutie mask decoder for ``reacher-visual-small``.

The decoder is intentionally a small, deterministic perception component. It
receives only the three object masks emitted by the Cutie adapter and never
reads RGB pixels, simulator state, rewards, or audit labels at inference time.
Its fixed base and link-length calibration are derived from a strictly
validated, manually RGB-labeled support pack.

Cutie can occasionally merge the two same-colour, physically connected links.
For that reason the elbow is not the far endpoint of the ``proximal_link``
mask.  Instead, the decoder estimates the first-link direction from an inner
annulus around the support-derived base and projects it to the support-derived
first-link length.  A short proximal prediction falls back to the union of the
two link masks.  The control tip retains the established 95%-trimmed distal
axis endpoint rule.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np


POINT_ROLES = ('base', 'elbow', 'control_tip', 'goal')
OBJECT_ROLES = ('proximal_link', 'distal_link', 'goal')
SUPPORT_FORMAT = 'few_shot_task_anchor_annotations_v1'
SUPPORT_TASK = 'reacher-visual-small'
DECODER_FORMAT = 'visual_small_cutie_point_decoder_v1'
DECODER_NAME = 'visual_small_cutie_point_decoder'
DECODER_VERSION = 1
SOURCE_RESOLUTION = (64, 64)
FORBIDDEN_FIELDS = {
	'physics',
	'qpos',
	'qvel',
	'geom_xpos',
	'site_xpos',
	'body_xpos',
	'simulator_state',
	'privileged_state',
	'state',
	'action',
	'reward',
	'ground_truth',
	'get_state',
}


class VisualSmallCutieDecoderContractError(ValueError):
	"""Raised when support data or inference masks violate the decoder contract."""


def _sha256_bytes(value: bytes) -> str:
	return hashlib.sha256(value).hexdigest()


def _decoded_rgb_sha256(frame: np.ndarray) -> str:
	return _sha256_bytes(np.ascontiguousarray(frame).tobytes())


def _is_sha256(value: Any) -> bool:
	return (
		isinstance(value, str)
		and len(value) == 64
		and all(character in '0123456789abcdef' for character in value)
	)


def _field_token(value: Any) -> str:
	return re.sub(r'[^a-z0-9]+', '_', str(value).lower()).strip('_')


def _reject_privileged_fields(value: Any, location: str = 'support') -> None:
	if isinstance(value, Mapping):
		for key, child in value.items():
			token = _field_token(key)
			if token in FORBIDDEN_FIELDS or token.startswith('physics_'):
				raise VisualSmallCutieDecoderContractError(
					f'Forbidden privileged field {key!r} at {location}; '
					'the decoder calibration must be RGB/manual-label only.'
				)
			_reject_privileged_fields(child, f'{location}.{key}')
	elif isinstance(value, list):
		for index, child in enumerate(value):
			_reject_privileged_fields(child, f'{location}[{index}]')


def _finite_point(value: Any, *, location: str) -> np.ndarray:
	if not isinstance(value, list) or len(value) != 2:
		raise VisualSmallCutieDecoderContractError(
			f'{location} must be an explicit [x, y] RGB point.'
		)
	try:
		point = np.asarray(value, dtype=np.float64)
	except (TypeError, ValueError) as exc:
		raise VisualSmallCutieDecoderContractError(
			f'{location} must contain two numeric coordinates.'
		) from exc
	if not np.isfinite(point).all():
		raise VisualSmallCutieDecoderContractError(
			f'{location} must contain finite coordinates.'
		)
	if not (
		0.0 <= point[0] < SOURCE_RESOLUTION[1]
		and 0.0 <= point[1] < SOURCE_RESOLUTION[0]
	):
		raise VisualSmallCutieDecoderContractError(
			f'{location} lies outside the native 64x64 RGB frame: {value!r}.'
		)
	return point


@dataclass(frozen=True)
class VisualSmallCutieCalibration:
	"""Immutable calibration and provenance derived from an RGB-only support pack."""

	annotation_path: Path
	support_annotations_sha256: str
	manifest_sha256: str
	combined_manifest_sha256: str
	base_xy: tuple[float, float]
	proximal_length_px: float
	distal_length_px: float
	base_max_deviation_px: float
	proximal_length_std_px: float
	distal_length_std_px: float
	support_records: int
	source_resolution: tuple[int, int] = SOURCE_RESOLUTION

	def metadata(self) -> dict[str, Any]:
		"""Return JSON-serializable calibration metadata."""
		return {
			'support_annotations_sha256': self.support_annotations_sha256,
			'manifest_sha256': self.manifest_sha256,
			'combined_manifest_sha256': self.combined_manifest_sha256,
			'base': [float(value) for value in self.base_xy],
			'L1': float(self.proximal_length_px),
			'L2': float(self.distal_length_px),
			'base_max_deviation_px': float(self.base_max_deviation_px),
			'L1_std_px': float(self.proximal_length_std_px),
			'L2_std_px': float(self.distal_length_std_px),
			'support_records': int(self.support_records),
			'source_resolution': list(self.source_resolution),
		}


def load_visual_small_cutie_calibration(
	annotation_path: str | Path,
	*,
	expected_records: int = 6,
) -> VisualSmallCutieCalibration:
	"""Load and strictly validate the permanent RGB-only support calibration.

	Every support image is decoded and checked against its recorded RGB SHA
	before calibration values are computed. Pillow is needed only during this
	one-time construction; ``decode`` itself depends solely on NumPy masks. The raw
	annotations SHA freezes both the point labels and the declared RGB evidence.
	The arithmetic mean is used for the fixed base and both fixed link lengths.
	"""
	path = Path(annotation_path).expanduser().resolve()
	if not path.is_file():
		raise FileNotFoundError(f'Support annotations not found: {path}')
	if not isinstance(expected_records, int) or expected_records <= 0:
		raise ValueError('expected_records must be a positive integer.')
	raw_annotations = path.read_bytes()
	try:
		data = json.loads(raw_annotations.decode('utf-8'))
	except (UnicodeDecodeError, json.JSONDecodeError) as exc:
		raise VisualSmallCutieDecoderContractError(
			f'Support annotations must be valid UTF-8 JSON: {path}'
		) from exc
	if not isinstance(data, Mapping):
		raise VisualSmallCutieDecoderContractError(
			'Support annotations must contain a JSON object.'
		)
	_reject_privileged_fields(data)
	if data.get('format') != SUPPORT_FORMAT:
		raise VisualSmallCutieDecoderContractError(
			f'Support format must be {SUPPORT_FORMAT!r}.'
		)
	if data.get('coordinate_convention') != '[x, y] in the original 64x64 RGB frame':
		raise VisualSmallCutieDecoderContractError(
			'Support coordinate_convention must describe [x, y] in the original '
			'64x64 RGB frame.'
		)
	if tuple(data.get('roles', ())) != POINT_ROLES:
		raise VisualSmallCutieDecoderContractError(
			f'Support roles must be ordered exactly as {POINT_ROLES!r}.'
		)

	collection = data.get('collection')
	if not isinstance(collection, Mapping):
		raise VisualSmallCutieDecoderContractError(
			'Support collection metadata is required.'
		)
	if collection.get('task') != SUPPORT_TASK:
		raise VisualSmallCutieDecoderContractError(
			f'Support task must be {SUPPORT_TASK!r}.'
		)
	if collection.get('observation') != 'rgb':
		raise VisualSmallCutieDecoderContractError(
			'Support observations must be RGB-only.'
		)
	if collection.get('split') != 'support':
		raise VisualSmallCutieDecoderContractError(
			'Support calibration must come from split=support.'
		)
	if collection.get('label_policy') != 'manual_rgb_only':
		raise VisualSmallCutieDecoderContractError(
			'Support label_policy must be manual_rgb_only.'
		)
	if collection.get('episodes') != expected_records:
		raise VisualSmallCutieDecoderContractError(
			f'collection.episodes must be {expected_records}.'
		)
	manifest_sha256 = collection.get('manifest_sha256')
	combined_manifest_sha256 = collection.get('combined_manifest_sha256')
	if not _is_sha256(manifest_sha256):
		raise VisualSmallCutieDecoderContractError(
			'collection.manifest_sha256 must be a lowercase SHA-256.'
		)
	if not _is_sha256(combined_manifest_sha256):
		raise VisualSmallCutieDecoderContractError(
			'collection.combined_manifest_sha256 must be a lowercase SHA-256.'
		)

	records = data.get('records')
	if not isinstance(records, list) or len(records) != expected_records:
		raise VisualSmallCutieDecoderContractError(
			f'Support records must contain exactly {expected_records} entries.'
		)
	try:
		from PIL import Image
	except ImportError as exc:
		raise VisualSmallCutieDecoderContractError(
			'Pillow is required once to verify the support RGB evidence.'
		) from exc
	bases: list[np.ndarray] = []
	elbows: list[np.ndarray] = []
	tips: list[np.ndarray] = []
	for expected_index, record in enumerate(records):
		location = f'records[{expected_index}]'
		if not isinstance(record, Mapping):
			raise VisualSmallCutieDecoderContractError(
				f'{location} must be an object.'
			)
		if record.get('index') != expected_index:
			raise VisualSmallCutieDecoderContractError(
				'Support record indices must be contiguous and ordered.'
			)
		if record.get('video_split') != 'support':
			raise VisualSmallCutieDecoderContractError(
				f'{location}.video_split must be support.'
			)
		if record.get('manifest_sha256') != manifest_sha256:
			raise VisualSmallCutieDecoderContractError(
				f'{location}.manifest_sha256 does not match the support manifest.'
			)
		image_value = record.get('image')
		if not isinstance(image_value, str) or not image_value:
			raise VisualSmallCutieDecoderContractError(
				f'{location}.image must be a relative support RGB path.'
			)
		image_relative = Path(image_value)
		if image_relative.is_absolute() or '..' in image_relative.parts:
			raise VisualSmallCutieDecoderContractError(
				f'{location}.image must stay inside the support pack.'
			)
		image_path = (path.parent / image_relative).resolve()
		try:
			image_path.relative_to(path.parent)
		except ValueError as exc:
			raise VisualSmallCutieDecoderContractError(
				f'{location}.image escapes the support pack: {image_path}'
			) from exc
		if not image_path.is_file():
			raise FileNotFoundError(f'Support RGB frame not found: {image_path}')
		expected_image_sha256 = record.get('image_sha256')
		if not _is_sha256(expected_image_sha256):
			raise VisualSmallCutieDecoderContractError(
				f'{location}.image_sha256 must be a lowercase SHA-256.'
			)
		with Image.open(image_path) as image:
			frame = np.array(image.convert('RGB'), dtype=np.uint8, copy=True)
		if frame.shape != (64, 64, 3):
			raise VisualSmallCutieDecoderContractError(
				f'{location}.image must decode to 64x64 RGB, got {frame.shape}.'
			)
		actual_image_sha256 = _decoded_rgb_sha256(frame)
		if actual_image_sha256 != expected_image_sha256:
			raise VisualSmallCutieDecoderContractError(
				f'{location} decoded-RGB SHA mismatch: expected '
				f'{expected_image_sha256}, actual {actual_image_sha256}.'
			)
		points = record.get('points')
		if not isinstance(points, Mapping) or tuple(points.keys()) != POINT_ROLES:
			raise VisualSmallCutieDecoderContractError(
				f'{location}.points must be ordered exactly as {POINT_ROLES!r}.'
			)
		base = _finite_point(points['base'], location=f'{location}.points.base')
		elbow = _finite_point(points['elbow'], location=f'{location}.points.elbow')
		tip = _finite_point(
			points['control_tip'], location=f'{location}.points.control_tip'
		)
		# Goal is required and validated even though it is not part of the link
		# calibration. This prevents a partially annotated pack from qualifying.
		_finite_point(points['goal'], location=f'{location}.points.goal')
		if float(np.linalg.norm(elbow - base)) <= 0.0:
			raise VisualSmallCutieDecoderContractError(
				f'{location} has a zero-length proximal link.'
			)
		if float(np.linalg.norm(tip - elbow)) <= 0.0:
			raise VisualSmallCutieDecoderContractError(
				f'{location} has a zero-length distal link.'
			)
		bases.append(base)
		elbows.append(elbow)
		tips.append(tip)

	base_values = np.stack(bases)
	elbow_values = np.stack(elbows)
	tip_values = np.stack(tips)
	base_xy = base_values.mean(axis=0)
	proximal_lengths = np.linalg.norm(elbow_values - base_values, axis=1)
	distal_lengths = np.linalg.norm(tip_values - elbow_values, axis=1)
	base_deviation = np.linalg.norm(base_values - base_xy, axis=1)
	return VisualSmallCutieCalibration(
		annotation_path=path,
		support_annotations_sha256=_sha256_bytes(raw_annotations),
		manifest_sha256=str(manifest_sha256),
		combined_manifest_sha256=str(combined_manifest_sha256),
		base_xy=(float(base_xy[0]), float(base_xy[1])),
		proximal_length_px=float(proximal_lengths.mean()),
		distal_length_px=float(distal_lengths.mean()),
		base_max_deviation_px=float(base_deviation.max()),
		proximal_length_std_px=float(proximal_lengths.std()),
		distal_length_std_px=float(distal_lengths.std()),
		support_records=len(records),
	)


@dataclass(frozen=True)
class _Component:
	mask: np.ndarray
	size: int
	min_base_distance: float
	first_yx: tuple[int, int]


def _components(mask: np.ndarray, base_xy: np.ndarray) -> list[_Component]:
	"""Return deterministic 8-connected components with base distances."""
	mask = np.asarray(mask, dtype=bool)
	visited = np.zeros_like(mask, dtype=bool)
	height, width = mask.shape
	result: list[_Component] = []
	for start_y, start_x in np.argwhere(mask):
		start_y = int(start_y)
		start_x = int(start_x)
		if visited[start_y, start_x]:
			continue
		stack = [(start_y, start_x)]
		visited[start_y, start_x] = True
		pixels: list[tuple[int, int]] = []
		while stack:
			y, x = stack.pop()
			pixels.append((y, x))
			for dy in (-1, 0, 1):
				for dx in (-1, 0, 1):
					if dx == 0 and dy == 0:
						continue
					ny, nx = y + dy, x + dx
					if (
						0 <= ny < height
						and 0 <= nx < width
						and mask[ny, nx]
						and not visited[ny, nx]
					):
						visited[ny, nx] = True
						stack.append((ny, nx))
		component_mask = np.zeros_like(mask, dtype=bool)
		yx = np.asarray(pixels, dtype=np.int64)
		component_mask[yx[:, 0], yx[:, 1]] = True
		xy = yx[:, ::-1].astype(np.float64)
		minimum_distance = float(np.linalg.norm(xy - base_xy, axis=1).min())
		result.append(_Component(
			mask=component_mask,
			size=len(pixels),
			min_base_distance=minimum_distance,
			first_yx=min(pixels),
		))
	return result


def _base_component(
	mask: np.ndarray,
	base_xy: np.ndarray,
	*,
	attachment_radius_px: float,
) -> tuple[np.ndarray, dict[str, Any]]:
	"""Prefer substantial components that actually touch the calibrated base.

	Among components whose nearest pixel is within the attachment radius, the
	largest wins; distance and lexicographic location break ties. This prevents a
	one-pixel base-near speck from stealing the arm. With no attached component,
	the nearest component is returned only for diagnostics: ``decode`` fails
	closed instead of using it to emit an elbow.
	"""
	components = _components(mask, base_xy)
	if not components:
		return np.zeros_like(mask, dtype=bool), {
			'components': 0,
			'attached_components': 0,
			'component_pixels': 0,
			'min_base_distance_px': None,
			'base_attached': False,
			'base_component_fallback': True,
		}
	attached = [
		item for item in components
		if item.min_base_distance <= attachment_radius_px
	]
	if attached:
		selected = min(
			attached,
			key=lambda item: (
				-item.size,
				item.min_base_distance,
				item.first_yx,
			),
		)
	else:
		selected = min(
			components,
			key=lambda item: (
				item.min_base_distance,
				-item.size,
				item.first_yx,
			),
		)
	base_attached = bool(attached)
	return selected.mask, {
		'components': len(components),
		'attached_components': len(attached),
		'component_pixels': selected.size,
		'min_base_distance_px': selected.min_base_distance,
		'base_attached': bool(base_attached),
		'base_component_fallback': not base_attached,
	}


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
	"""Keep the largest 8-connected component of a small binary mask."""
	mask = np.asarray(mask, dtype=bool)
	components = _components(mask, np.zeros(2, dtype=np.float64))
	if not components:
		return np.zeros_like(mask, dtype=bool), {
			'components': 0,
			'component_pixels': 0,
			'total_pixels': 0,
			'component_fraction': 0.0,
		}
	selected = min(
		components,
		key=lambda item: (-item.size, item.first_yx),
	)
	total_pixels = int(mask.sum())
	return selected.mask, {
		'components': len(components),
		'component_pixels': selected.size,
		'total_pixels': total_pixels,
		'component_fraction': float(selected.size / max(total_pixels, 1)),
	}


def _trimmed_axis_endpoint(
	mask: np.ndarray,
	origin_xy: np.ndarray,
	*,
	percentile: float = 95.0,
) -> tuple[list[float] | None, dict[str, Any]]:
	"""Equivalent 95%-trimmed distal endpoint with JSON-safe diagnostics."""
	component, component_diagnostics = _largest_component(mask)
	yx = np.argwhere(component)
	if len(yx) < 2:
		return None, {
			**component_diagnostics,
			'percentile': float(percentile),
			'axis_anisotropy': 0.0,
			'endpoint_available': False,
		}
	coordinates = yx[:, ::-1].astype(np.float64)
	origin = np.asarray(origin_xy, dtype=np.float64)
	center = coordinates.mean(axis=0)
	centered = coordinates - center
	covariance = centered.T @ centered / max(len(coordinates) - 1, 1)
	eigenvalues, eigenvectors = np.linalg.eigh(covariance)
	axis = eigenvectors[:, int(np.argmax(eigenvalues))]
	if float(np.dot(axis, center - origin)) < 0.0:
		axis = -axis
	if abs(float(np.dot(axis, center - origin))) < 1e-8:
		farthest = coordinates[
			int(np.argmax(np.linalg.norm(coordinates - origin, axis=1)))
		]
		direction = farthest - origin
		direction_norm = float(np.linalg.norm(direction))
		if direction_norm > 1e-8:
			axis = direction / direction_norm
	perpendicular = np.asarray([-axis[1], axis[0]], dtype=np.float64)
	offsets = coordinates - origin
	longitudinal = offsets @ axis
	transverse = offsets @ perpendicular
	target = float(np.percentile(longitudinal, percentile))
	spread = max(float(np.ptp(longitudinal)), 1.0)
	band = np.abs(longitudinal - target) <= max(0.75, 0.03 * spread)
	if not band.any():
		band[int(np.argmin(np.abs(longitudinal - target)))] = True
	cross_center = float(np.median(transverse[band]))
	endpoint = origin + axis * target + perpendicular * cross_center
	height, width = component.shape
	endpoint[0] = np.clip(endpoint[0], 0.0, width - 0.5)
	endpoint[1] = np.clip(endpoint[1], 0.0, height - 0.5)
	eigenvalue_sum = float(np.maximum(eigenvalues, 0.0).sum())
	anisotropy = (
		0.0 if eigenvalue_sum <= 1e-12
		else float((eigenvalues[-1] - eigenvalues[0]) / eigenvalue_sum)
	)
	return endpoint.astype(np.float64).tolist(), {
		**component_diagnostics,
		'percentile': float(percentile),
		'axis_anisotropy': float(np.clip(anisotropy, 0.0, 1.0)),
		'longitudinal_span_px': float(np.ptp(longitudinal)),
		'endpoint_available': True,
	}


@dataclass(frozen=True)
class VisualSmallCutieDecodeResult:
	"""Four decoded control points plus GT-free confidence and diagnostics."""

	points: dict[str, list[float] | None]
	point_confidence: dict[str, float]
	geometry_confidence: float
	lost: dict[str, bool]
	diagnostics: dict[str, Any]

	def as_dict(self) -> dict[str, Any]:
		"""Return a JSON-serializable representation."""
		return {
			'points': self.points,
			'point_confidence': self.point_confidence,
			'geometry_confidence': float(self.geometry_confidence),
			'lost': self.lost,
			'diagnostics': self.diagnostics,
		}


class VisualSmallCutiePointDecoder:
	"""Decode stable task points from Cutie's three fixed object-role masks."""

	def __init__(
		self,
		calibration: VisualSmallCutieCalibration,
		*,
		inner_lower: float = 0.25,
		inner_upper: float = 0.60,
		proximal_coverage_threshold: float = 0.60,
		base_attachment_radius_fraction: float = 0.25,
		base_attachment_radius_min_px: float = 1.5,
		direction_coherence_floor: float = 0.90,
		distal_endpoint_percentile: float = 95.0,
	):
		if not isinstance(calibration, VisualSmallCutieCalibration):
			raise TypeError('calibration must be a VisualSmallCutieCalibration.')
		for name, value in (
			('inner_lower', inner_lower),
			('inner_upper', inner_upper),
			('proximal_coverage_threshold', proximal_coverage_threshold),
			('base_attachment_radius_fraction', base_attachment_radius_fraction),
		):
			if not math.isfinite(float(value)) or not 0.0 < float(value) <= 1.0:
				raise ValueError(f'{name} must lie in (0, 1].')
		if not inner_lower < inner_upper:
			raise ValueError('inner_lower must be smaller than inner_upper.')
		if proximal_coverage_threshold < inner_upper:
			raise ValueError(
				'proximal_coverage_threshold must be at least inner_upper.'
			)
		if (
			not math.isfinite(float(base_attachment_radius_min_px))
			or base_attachment_radius_min_px <= 0.0
		):
			raise ValueError('base_attachment_radius_min_px must be positive.')
		if (
			not math.isfinite(float(direction_coherence_floor))
			or not 0.0 <= direction_coherence_floor < 1.0
		):
			raise ValueError('direction_coherence_floor must lie in [0, 1).')
		if (
			not math.isfinite(float(distal_endpoint_percentile))
			or not 0.0 < distal_endpoint_percentile <= 100.0
		):
			raise ValueError('distal_endpoint_percentile must lie in (0, 100].')
		self.calibration = calibration
		self.inner_lower = float(inner_lower)
		self.inner_upper = float(inner_upper)
		self.proximal_coverage_threshold = float(proximal_coverage_threshold)
		self.base_attachment_radius_fraction = float(
			base_attachment_radius_fraction
		)
		self.base_attachment_radius_min_px = float(
			base_attachment_radius_min_px
		)
		self.direction_coherence_floor = float(direction_coherence_floor)
		self.distal_endpoint_percentile = float(distal_endpoint_percentile)

	@classmethod
	def from_support(
		cls,
		annotation_path: str | Path,
		*,
		expected_records: int = 6,
		**decoder_kwargs: Any,
	) -> 'VisualSmallCutiePointDecoder':
		"""Construct a decoder only after strict support-pack verification."""
		calibration = load_visual_small_cutie_calibration(
			annotation_path, expected_records=expected_records
		)
		return cls(calibration, **decoder_kwargs)

	def reset(self) -> None:
		"""Reset episode state (the current decoder is deliberately stateless)."""
		return None

	def metadata(self) -> dict[str, Any]:
		"""Return the complete JSON-safe frozen decoder contract."""
		return {
			'format': DECODER_FORMAT,
			'name': DECODER_NAME,
			'version': DECODER_VERSION,
			**self.calibration.metadata(),
			'inner_lower': self.inner_lower,
			'inner_upper': self.inner_upper,
			'proximal_coverage_threshold': self.proximal_coverage_threshold,
			'base_attachment_radius_fraction': self.base_attachment_radius_fraction,
			'base_attachment_radius_min_px': self.base_attachment_radius_min_px,
			'direction_coherence_floor': self.direction_coherence_floor,
			'distal_endpoint_percentile': self.distal_endpoint_percentile,
			'component_policy': (
				'largest_base_attached_8_connected_then_union_if_short_fail_closed'
			),
		}

	def _validated_masks(
		self, object_masks: Mapping[str, np.ndarray]
	) -> dict[str, np.ndarray]:
		if not isinstance(object_masks, Mapping):
			raise VisualSmallCutieDecoderContractError(
				'object_masks must be a role-to-mask mapping.'
			)
		if set(object_masks) != set(OBJECT_ROLES):
			raise VisualSmallCutieDecoderContractError(
				f'object_masks must contain exactly {OBJECT_ROLES!r}.'
			)
		result: dict[str, np.ndarray] = {}
		for role in OBJECT_ROLES:
			mask = np.asarray(object_masks[role])
			if mask.ndim != 2 or mask.shape != self.calibration.source_resolution:
				raise VisualSmallCutieDecoderContractError(
					f'{role} mask must have shape '
					f'{self.calibration.source_resolution}, got {mask.shape}.'
				)
			if np.issubdtype(mask.dtype, np.bool_):
				result[role] = np.ascontiguousarray(mask, dtype=bool)
				continue
			if (
				not np.issubdtype(mask.dtype, np.number)
				or np.issubdtype(mask.dtype, np.complexfloating)
			):
				raise VisualSmallCutieDecoderContractError(
					f'{role} mask must be boolean or numeric binary 0/1.'
				)
			if not np.isfinite(mask).all() or not np.logical_or(
				mask == 0, mask == 1
			).all():
				raise VisualSmallCutieDecoderContractError(
					f'{role} mask contains non-finite or non-binary values.'
				)
			result[role] = np.ascontiguousarray(mask, dtype=bool)
		return result

	def decode(
		self, object_masks: Mapping[str, np.ndarray]
	) -> VisualSmallCutieDecodeResult:
		"""Decode one frame using masks alone.

		The method is deterministic and stateless. In particular, call order and
		cache/live evaluation order cannot affect its output.
		"""
		masks = self._validated_masks(object_masks)
		base = np.asarray(self.calibration.base_xy, dtype=np.float64)
		L1 = self.calibration.proximal_length_px
		attachment_radius = max(
			self.base_attachment_radius_min_px,
			self.base_attachment_radius_fraction * L1,
		)

		proximal_component, proximal_diagnostics = _base_component(
			masks['proximal_link'],
			base,
			attachment_radius_px=attachment_radius,
		)
		proximal_xy = np.argwhere(proximal_component)[:, ::-1].astype(np.float64)
		proximal_radius = (
			np.linalg.norm(proximal_xy - base, axis=1)
			if len(proximal_xy)
			else np.asarray([], dtype=np.float64)
		)
		proximal_max_radius = (
			float(proximal_radius.max()) if len(proximal_radius) else 0.0
		)
		coverage_radius = self.proximal_coverage_threshold * L1
		use_union = (
			not bool(proximal_diagnostics['base_attached'])
			or proximal_max_radius < coverage_radius
		)
		if use_union:
			selected_component, selected_diagnostics = _base_component(
				np.logical_or(masks['proximal_link'], masks['distal_link']),
				base,
				attachment_radius_px=attachment_radius,
			)
			source_used = 'proximal_or_distal_union'
		else:
			selected_component = proximal_component
			selected_diagnostics = proximal_diagnostics
			source_used = 'proximal_link'

		selected_is_attached = bool(selected_diagnostics['base_attached'])
		# A remote blob is useful diagnostic evidence but is never allowed to
		# produce a point. This is the no-silent-fallback contract.
		geometry_component = (
			selected_component
			if selected_is_attached
			else np.zeros_like(selected_component, dtype=bool)
		)
		selected_yx = np.argwhere(geometry_component)
		selected_xy = selected_yx[:, ::-1].astype(np.float64)
		selected_offsets = selected_xy - base
		selected_radius = (
			np.linalg.norm(selected_offsets, axis=1)
			if len(selected_offsets)
			else np.asarray([], dtype=np.float64)
		)
		lower_radius = self.inner_lower * L1
		upper_radius = self.inner_upper * L1
		inner = (
			(selected_radius >= lower_radius) & (selected_radius <= upper_radius)
			if len(selected_radius)
			else np.asarray([], dtype=bool)
		)
		inner_offsets = selected_offsets[inner]
		if len(inner_offsets):
			inner_norms = np.linalg.norm(inner_offsets, axis=1)
			valid = inner_norms > 1e-12
			unit_directions = inner_offsets[valid] / inner_norms[valid, None]
		else:
			unit_directions = np.empty((0, 2), dtype=np.float64)
		# Mean raw offsets reproduce the center-line direction of a thick inner
		# segment. Mean unit directions are kept separately as the ambiguity /
		# directional-coherence diagnostic used by confidence.
		direction_vector = (
			inner_offsets[valid].mean(axis=0)
			if len(unit_directions)
			else np.zeros(2, dtype=np.float64)
		)
		direction_norm = float(np.linalg.norm(direction_vector))
		direction_coherence = (
			float(np.linalg.norm(unit_directions.mean(axis=0)))
			if len(unit_directions)
			else 0.0
		)
		elbow: list[float] | None
		if direction_norm <= 1e-12:
			elbow = None
		else:
			direction = direction_vector / direction_norm
			elbow_array = base + direction * L1
			height, width = selected_component.shape
			elbow_array[0] = np.clip(elbow_array[0], 0.0, width - 0.5)
			elbow_array[1] = np.clip(elbow_array[1], 0.0, height - 0.5)
			elbow = elbow_array.astype(np.float64).tolist()

		if elbow is None:
			tip = None
			tip_diagnostics = {
				'components': 0,
				'component_pixels': 0,
				'total_pixels': int(masks['distal_link'].sum()),
				'component_fraction': 0.0,
				'percentile': self.distal_endpoint_percentile,
				'axis_anisotropy': 0.0,
				'endpoint_available': False,
				'skipped_without_elbow': True,
			}
		else:
			tip, tip_diagnostics = _trimmed_axis_endpoint(
				masks['distal_link'],
				np.asarray(elbow, dtype=np.float64),
				percentile=self.distal_endpoint_percentile,
			)
			tip_diagnostics['skipped_without_elbow'] = False
		if elbow is not None and tip is not None:
			tip_length_px = float(np.linalg.norm(
				np.asarray(tip, dtype=np.float64)
				- np.asarray(elbow, dtype=np.float64)
			))
			tip_length_residual_px = abs(
				tip_length_px - self.calibration.distal_length_px
			)
			# The trimmed endpoint is an image estimator rather than a hard
			# kinematic projection. Keep its coordinate unchanged and use the
			# support-derived L2 only as a smooth confidence residual.
			length_scale = max(
				2.0,
				4.0 * self.calibration.distal_length_std_px,
			)
			tip_length_score = float(math.exp(
				-tip_length_residual_px / length_scale
			))
		else:
			tip_length_px = None
			tip_length_residual_px = None
			tip_length_score = 0.0
		tip_diagnostics.update({
			'support_L2_px': float(self.calibration.distal_length_px),
			'tip_length_px': tip_length_px,
			'tip_length_residual_px': tip_length_residual_px,
			'tip_length_score': tip_length_score,
		})

		goal_component, goal_diagnostics = _largest_component(masks['goal'])
		goal_yx = np.argwhere(goal_component)
		goal = (
			None
			if not len(goal_yx)
			else goal_yx[:, ::-1].astype(np.float64).mean(axis=0).tolist()
		)

		base_attached = selected_is_attached
		attachment_confidence = 1.0 if base_attached else 0.0
		pixel_confidence = min(1.0, len(unit_directions) / 4.0)
		coherence_confidence = float(np.clip(
			(
				direction_coherence - self.direction_coherence_floor
			) / (1.0 - self.direction_coherence_floor),
			0.0,
			1.0,
		))
		elbow_confidence = float(np.clip(
			coherence_confidence * pixel_confidence * attachment_confidence,
			0.0,
			1.0,
		))
		if elbow is None:
			elbow_confidence = 0.0
		tip_confidence = float(np.clip(
			float(tip_diagnostics.get('component_fraction', 0.0))
			* float(tip_diagnostics.get('axis_anisotropy', 0.0))
			* min(1.0, float(tip_diagnostics.get('component_pixels', 0)) / 4.0)
			* tip_length_score,
			0.0,
			1.0,
		))
		if tip is None:
			tip_confidence = 0.0
		goal_confidence = float(np.clip(
			float(goal_diagnostics['component_fraction'])
			* min(1.0, float(goal_diagnostics['component_pixels']) / 3.0),
			0.0,
			1.0,
		))
		if goal is None:
			goal_confidence = 0.0
		point_confidence = {
			'base': 1.0,
			'elbow': elbow_confidence,
			'control_tip': tip_confidence,
			'goal': goal_confidence,
		}
		geometry_confidence = float(min(
			point_confidence['elbow'],
			point_confidence['control_tip'],
			point_confidence['goal'],
		))
		points = {
			'base': [float(base[0]), float(base[1])],
			'elbow': elbow,
			'control_tip': tip,
			'goal': goal,
		}
		lost = {role: points[role] is None for role in POINT_ROLES}
		proximal_role_overreach = proximal_max_radius > 1.25 * L1
		distal_pixels = int(tip_diagnostics.get('total_pixels', 0))
		distal_fragmented = (
			distal_pixels > 0
			and float(tip_diagnostics.get('component_fraction', 0.0)) < 0.75
		)
		role_collapse_reasons = []
		if proximal_role_overreach:
			role_collapse_reasons.append('proximal_radial_overreach')
		if distal_pixels < 2:
			role_collapse_reasons.append('distal_missing_or_too_small')
		if distal_fragmented:
			role_collapse_reasons.append('distal_fragmented')
		diagnostics = {
			'source_used': source_used,
			'proximal_max_radius': proximal_max_radius,
			'proximal_coverage_radius': float(coverage_radius),
			'proximal_coverage_sufficient': not use_union,
			'selected_pixels': int(geometry_component.sum()),
			'candidate_component_pixels': int(selected_component.sum()),
			'inner_selected_pixels': int(len(unit_directions)),
			'inner_lower_radius': float(lower_radius),
			'inner_upper_radius': float(upper_radius),
			'direction_coherence': direction_coherence,
			'direction_coherence_confidence': coherence_confidence,
			'base_attachment_radius': float(attachment_radius),
			'base_component_fallback': bool(
				selected_diagnostics['base_component_fallback']
			),
			'role_collapse': bool(role_collapse_reasons),
			'role_collapse_reasons': role_collapse_reasons,
			'proximal_role_overreach': proximal_role_overreach,
			'distal_fragmented': distal_fragmented,
			'proximal_component': proximal_diagnostics,
			'selected_component': selected_diagnostics,
			'distal_endpoint': tip_diagnostics,
			'goal_component': goal_diagnostics,
			'geometry_confidence': geometry_confidence,
		}
		return VisualSmallCutieDecodeResult(
			points=points,
			point_confidence=point_confidence,
			geometry_confidence=geometry_confidence,
			lost=lost,
			diagnostics=diagnostics,
		)


__all__ = (
	'VisualSmallCutieCalibration',
	'VisualSmallCutieDecodeResult',
	'VisualSmallCutieDecoderContractError',
	'VisualSmallCutiePointDecoder',
	'load_visual_small_cutie_calibration',
)
