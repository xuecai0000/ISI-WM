"""Leakage-safe A/B evaluation for Visual-Small perception backends.

The evaluator consumes only a manually labelled RGB audit and backend outputs.
It never imports an environment, simulator, reward, or physics API.  Every
16-frame sequence is sent to a backend in temporal order, while point metrics
are computed only at the manually labelled times ``0, 5, 10, 15``.

Prediction cache contract (``visual_small_perception_predictions_v1``)::

    {
      "format": "visual_small_perception_predictions_v1",
      "backend": "dino",  # or "cutie"
      "audit_sha256": "...",
      "roles": ["base", "elbow", "control_tip", "goal"],
      "sequences": [{
        "sequence_id": "...", "split": "train", "episode": 0,
        "source": "video0.mp4", "start_frame": 10,
        "frames": [{
          "index": 0, "t": 0,
          "points": {"base": [x, y], ...},
          "confidence": {"base": 0.9, ...},
          "lost": {"base": false, ...},
          "runtime_ms": 7.8,
          "mask": "optional/indexed_mask.png"
        }, ...]
      }]
    }

Live Cutie evaluation uses the in-tree ``CutieOCAdapter`` and requires the
external official OC-STORM repository/checkpoint plus the permanent six-frame
support annotations.  Support is loaded once, then ``track_episode`` processes
each ordered audit sequence.  Audit masks are evaluation targets only and are
never tracker prompts.

A Cutie cache declares an explicit object schema. ``legacy_three_object_v1``
provides ``proximal_link``, ``distal_link``, and ``goal`` masks;
``whole_arm_goal_v1`` provides ``whole_arm`` and ``goal`` masks and is valid
only with ``whole_arm_kinematic_v2``. Cached point fields are always ignored.
Kinematic decoders derive the fixed base and link lengths exclusively from the
explicit RGB-only support pack. Object schema, roles, decoder mode, and support
SHA are recorded in every new cache and report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw


AUDIT_FORMAT = 'visual_small_perception_audit_v1'
CACHE_FORMAT = 'visual_small_perception_predictions_v1'
REPORT_FORMAT = 'visual_small_perception_ab_report_v1'
ROLES = ('base', 'elbow', 'control_tip', 'goal')
SPLITS = ('train', 'validation')
ANNOTATION_TIMES = (0, 5, 10, 15)
SEQUENCE_LENGTH = 16
EPISODES_PER_SPLIT = 3
PCK_THRESHOLD_PX = 4.0
CONFIDENT_WRONG_ERROR_PX = 6.0
FORBIDDEN_FIELDS = {
	'physics',
	'qpos',
	'qvel',
	'geom_xpos',
	'site_xpos',
	'body_xpos',
	'simulator_state',
	'privileged_state',
	'reward',
	'ground_truth',
	'get_state',
}
CUTIE_OBJECTS = ('proximal_link', 'distal_link', 'goal')
CUTIE_WHOLE_ARM_OBJECTS = ('whole_arm', 'goal')
CUTIE_OBJECT_SCHEMA_LEGACY = 'legacy_three_object_v1'
CUTIE_OBJECT_SCHEMA_WHOLE_ARM = 'whole_arm_goal_v1'
CUTIE_OBJECT_SCHEMA_ROLES = {
	CUTIE_OBJECT_SCHEMA_LEGACY: CUTIE_OBJECTS,
	CUTIE_OBJECT_SCHEMA_WHOLE_ARM: CUTIE_WHOLE_ARM_OBJECTS,
}
CUTIE_OBJECT_SCHEMA_CHOICES = tuple(CUTIE_OBJECT_SCHEMA_ROLES)
CUTIE_DECODER_LEGACY = 'legacy_pca_v1'
CUTIE_DECODER_KINEMATIC = 'support_kinematic_v1'
CUTIE_DECODER_WHOLE_ARM_V1 = 'whole_arm_kinematic_v1'
CUTIE_DECODER_WHOLE_ARM_V2 = 'whole_arm_kinematic_v2'
# Public current-version alias retained for in-tree library callers. The v1
# string is deliberately not a CLI choice because its implementation is no
# longer shipped; accepting it would risk silently running v2 semantics.
CUTIE_DECODER_WHOLE_ARM = CUTIE_DECODER_WHOLE_ARM_V2
CUTIE_DECODER_CHOICES = (
	CUTIE_DECODER_LEGACY,
	CUTIE_DECODER_KINEMATIC,
	CUTIE_DECODER_WHOLE_ARM,
)
CUTIE_DECODER_WHOLE_ARM_V2_IDENTITY = {
	'format': 'visual_small_whole_arm_point_decoder_v2',
	'name': 'visual_small_whole_arm_point_decoder',
	'version': 2,
	'algorithm': 'base_connected_zhang_suen_geodesic_arc_length_v2',
	'short_centerline_policy': 'bounded_shortfall_endpoint_v2',
}
AUDIT_MASK_OBJECTS = {'1': 'proximal_link', '2': 'distal_link', '3': 'goal'}
MASK_METRICS = ('proximal_link', 'distal_link', 'controlled_arm', 'goal')
ROLE_COLORS = {
	'base': (0, 180, 255),
	'elbow': (255, 180, 0),
	'control_tip': (255, 40, 80),
	'goal': (80, 230, 80),
}


class ContractError(ValueError):
	"""Raised before inference when an audit or cache breaks the contract."""


def _legacy_cutie_decoder_metadata() -> dict[str, Any]:
	return {
		'format': 'visual_small_cutie_point_decoder_v1',
		'name': CUTIE_DECODER_LEGACY,
		'version': 1,
		'base': [31.5, 31.5],
		'link_endpoint': 'largest_component_pca_95_percent_cross_section',
		'goal': 'largest_component_centroid',
	}


def _whole_arm_backend_status_policy() -> dict[str, Any]:
	"""Frozen fusion contract for official Cutie status and mask geometry."""
	return {
		'format': 'visual_small_whole_arm_backend_status_v1',
		'confidence_fusion': 'minimum_v1',
		'whole_arm_lost_points': ['elbow', 'control_tip'],
		'goal_lost_points': ['goal'],
		'base_policy': 'decoder_only_unchanged',
		'cached_point_status': 'diagnostic_only',
	}


def _cutie_object_roles(object_schema: str) -> tuple[str, ...]:
	try:
		return CUTIE_OBJECT_SCHEMA_ROLES[object_schema]
	except KeyError as exc:
		raise ContractError(
			f'Unknown Cutie object schema {object_schema!r}; expected '
			f'{CUTIE_OBJECT_SCHEMA_CHOICES}.'
		) from exc


def _validate_cutie_decoder_schema_pair(
	decoder_mode: str,
	object_schema: str,
) -> tuple[str, ...]:
	if decoder_mode not in CUTIE_DECODER_CHOICES:
		raise ContractError(
			f'Unknown Cutie point decoder {decoder_mode!r}; expected '
			f'{CUTIE_DECODER_CHOICES}.'
		)
	roles = _cutie_object_roles(object_schema)
	expected_schema = (
		CUTIE_OBJECT_SCHEMA_WHOLE_ARM
		if decoder_mode == CUTIE_DECODER_WHOLE_ARM
		else CUTIE_OBJECT_SCHEMA_LEGACY
	)
	if object_schema != expected_schema:
		raise ContractError(
			f'Cutie point decoder {decoder_mode!r} requires object schema '
			f'{expected_schema!r}, got {object_schema!r}.'
		)
	return roles


def _infer_cutie_decoder_mode(cutie_decoder: Any | None) -> str:
	"""Infer the public mode for direct library callers using a decoder object."""
	if cutie_decoder is None:
		return CUTIE_DECODER_LEGACY
	try:
		metadata = cutie_decoder.metadata()
	except Exception as exc:
		raise ContractError(
			f'Cutie decoder metadata() failed: {type(exc).__name__}: {exc}'
		) from exc
	if not isinstance(metadata, Mapping):
		raise ContractError('Cutie decoder metadata() must return an object.')
	if tuple(metadata.get('input_roles', ())) == CUTIE_WHOLE_ARM_OBJECTS:
		mismatched = {
			key: metadata.get(key)
			for key, expected in CUTIE_DECODER_WHOLE_ARM_V2_IDENTITY.items()
			if metadata.get(key) != expected
		}
		if mismatched:
			raise ContractError(
				'Whole-arm decoder metadata does not identify the supported '
				f'{CUTIE_DECODER_WHOLE_ARM_V2!r} implementation; expected '
				f'{CUTIE_DECODER_WHOLE_ARM_V2_IDENTITY}, got mismatched fields '
				f'{mismatched}. The v1 algorithm cannot be used as v2.'
			)
		return CUTIE_DECODER_WHOLE_ARM_V2
	return CUTIE_DECODER_KINEMATIC


def _resolve_cutie_decoder_mode(
	cutie_decoder: Any | None,
	requested_mode: str | None,
) -> str:
	inferred_mode = _infer_cutie_decoder_mode(cutie_decoder)
	mode = inferred_mode if requested_mode is None else requested_mode
	if mode == CUTIE_DECODER_LEGACY and cutie_decoder is not None:
		raise ContractError('legacy_pca_v1 cannot be paired with a decoder object.')
	if mode != CUTIE_DECODER_LEGACY and cutie_decoder is None:
		raise ContractError(f'{mode} requires a decoder object.')
	if cutie_decoder is not None and mode != inferred_mode:
		raise ContractError(
			f'Cutie decoder object identifies as {inferred_mode!r}, not the '
			f'explicitly requested {mode!r}.'
		)
	return mode


def _build_cutie_point_decoder(
	mode: str,
	*,
	support_path: Path | None,
):
	"""Build the optional support-only decoder without importing it by default."""
	if mode not in CUTIE_DECODER_CHOICES:
		raise ContractError(
			f'Unknown Cutie point decoder {mode!r}; expected {CUTIE_DECODER_CHOICES}.'
		)
	if mode == CUTIE_DECODER_LEGACY:
		return None
	if support_path is None:
		raise ContractError(
			f'{mode} requires an explicit RGB-only Cutie '
			'support annotations file.'
		)
	try:
		if mode == CUTIE_DECODER_WHOLE_ARM:
			from tdmpc2.perception.visual_small_whole_arm_decoder import (
				VisualSmallWholeArmPointDecoder,
			)
			decoder_class = VisualSmallWholeArmPointDecoder
		else:
			from tdmpc2.perception.visual_small_cutie_decoder import (
				VisualSmallCutiePointDecoder,
			)
			decoder_class = VisualSmallCutiePointDecoder
		decoder = decoder_class.from_support(support_path)
		# Validate the constructed implementation identity immediately. In
		# particular, a v1 whole-arm class must never satisfy a v2 CLI request.
		_resolve_cutie_decoder_mode(decoder, mode)
		return decoder
	except Exception as exc:
		raise ContractError(
			f'Unable to construct {mode} from {support_path}: '
			f'{type(exc).__name__}: {exc}'
		) from exc


class _Config(dict):
	def __getattr__(self, key):
		try:
			return self[key]
		except KeyError as exc:
			raise AttributeError(key) from exc


def _json_load(path: Path) -> dict[str, Any]:
	try:
		with path.open(encoding='utf-8') as stream:
			value = json.load(stream)
	except (OSError, json.JSONDecodeError) as exc:
		raise ContractError(f'Unable to read JSON {path}: {exc}') from exc
	if not isinstance(value, dict):
		raise ContractError(f'Expected a JSON object in {path}.')
	return value


def _json_dump(path: Path, value: Any):
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', encoding='utf-8', newline='\n') as stream:
		json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
		stream.write('\n')


def _file_sha256(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def _image_sha256(image: np.ndarray) -> str:
	return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _field_token(value: Any) -> str:
	return re.sub(r'[^a-z0-9]+', '_', str(value).lower()).strip('_')


def _reject_privileged_fields(value: Any, location: str = 'audit'):
	if isinstance(value, Mapping):
		for key, child in value.items():
			token = _field_token(key)
			if token in FORBIDDEN_FIELDS or token.startswith('physics_'):
				raise ContractError(
					f'Forbidden privileged field {key!r} at {location}; '
					'the perception audit must be RGB/manual-label only.'
				)
			_reject_privileged_fields(child, f'{location}.{key}')
	elif isinstance(value, list):
		for index, child in enumerate(value):
			_reject_privileged_fields(child, f'{location}[{index}]')


def _point(value: Any, *, location: str, allow_none: bool) -> list[float] | None:
	if value is None and allow_none:
		return None
	if not isinstance(value, (list, tuple)) or len(value) != 2:
		raise ContractError(f'{location} must be [x, y]{" or null" if allow_none else ""}.')
	try:
		point = [float(value[0]), float(value[1])]
	except (TypeError, ValueError) as exc:
		raise ContractError(f'{location} contains a non-numeric coordinate.') from exc
	if not all(math.isfinite(item) for item in point):
		raise ContractError(f'{location} contains a non-finite coordinate.')
	return point


def _role_mapping(
	value: Any,
	*,
	location: str,
	converter,
	allow_scalar: bool = False,
) -> dict[str, Any]:
	if allow_scalar and not isinstance(value, Mapping):
		return {role: converter(value, f'{location}.{role}') for role in ROLES}
	if not isinstance(value, Mapping) or set(value) != set(ROLES):
		raise ContractError(f'{location} must contain exactly the ordered roles {ROLES}.')
	return {role: converter(value[role], f'{location}.{role}') for role in ROLES}


def _resolve_asset(
	root: Path,
	value: Any,
	*,
	location: str,
	require_relative: bool = True,
) -> Path | None:
	if value is None:
		return None
	if not isinstance(value, str) or not value:
		raise ContractError(f'{location} must be a non-empty path string or null.')
	root = root.expanduser().resolve()
	path = Path(value).expanduser()
	if path.is_absolute() and require_relative:
		raise ContractError(f'{location} must be relative to its JSON asset root.')
	if not path.is_absolute():
		path = root / path
	path = path.resolve()
	try:
		path.relative_to(root)
	except ValueError as exc:
		raise ContractError(f'{location} escapes its JSON asset root: {path}') from exc
	if not path.is_file():
		raise ContractError(f'Missing asset for {location}: {path}')
	return path


def _collection_splits(collection: Mapping[str, Any]) -> set[str]:
	value = collection.get('splits', ())
	if isinstance(value, Mapping):
		return {str(key) for key in value}
	if isinstance(value, (list, tuple)):
		return {str(item) for item in value}
	return set()


def _split_manifest(split: str) -> tuple[set[str], str]:
	path = (
		Path(__file__).resolve().parents[1] / 'envs' / 'background_manifests'
		/ f'color_multi_{split}.json'
	)
	data = _json_load(path)
	if data.get('name') != split or not isinstance(data.get('sources'), list):
		raise ContractError(f'Invalid packaged {split} manifest: {path}')
	return set(map(str, data['sources'])), _file_sha256(path)


def _sequence_id(sequence: Mapping[str, Any]) -> str:
	explicit = sequence.get('sequence_id', sequence.get('id'))
	if explicit is not None:
		if not isinstance(explicit, str) or not explicit.strip():
			raise ContractError('sequence_id must be a non-empty string.')
		return explicit
	return '{split}:episode-{episode}:{source}@{start}'.format(
		split=sequence.get('split'),
		episode=sequence.get('episode'),
		source=sequence.get('source'),
		start=sequence.get('start_frame'),
	)


def load_audit(path: Path) -> dict[str, Any]:
	"""Validate and materialize a strict RGB-only audit."""
	path = path.expanduser().resolve()
	data = _json_load(path)
	_reject_privileged_fields(data)
	if data.get('format') != AUDIT_FORMAT:
		raise ContractError(
			f'Expected format={AUDIT_FORMAT!r}, got {data.get("format")!r}.'
		)
	if tuple(data.get('roles', ())) != ROLES:
		raise ContractError(f'Audit roles must be ordered exactly as {ROLES}.')
	collection = data.get('collection')
	if not isinstance(collection, Mapping):
		raise ContractError('Audit collection metadata is required.')
	if collection.get('no_test') is not True:
		raise ContractError('collection.no_test must be true.')
	if collection.get('task') != 'reacher-visual-small':
		raise ContractError('collection.task must be reacher-visual-small.')
	if collection.get('label_policy') != 'manual_rgb_only':
		raise ContractError('collection.label_policy must be manual_rgb_only.')
	declared_splits = _collection_splits(collection)
	if declared_splits and declared_splits != set(SPLITS):
		raise ContractError(
			f'collection.splits must contain train and validation only, got '
			f'{sorted(declared_splits)}.'
		)
	sequences = data.get('sequences')
	if not isinstance(sequences, list) or not sequences:
		raise ContractError('Audit must contain at least one sequence.')

	seen_ids = set()
	seen_splits = set()
	manifests = {split: _split_manifest(split) for split in SPLITS}
	normalized_sequences = []
	for sequence_index, sequence in enumerate(sequences):
		location = f'sequences[{sequence_index}]'
		if not isinstance(sequence, Mapping):
			raise ContractError(f'{location} must be an object.')
		split = str(sequence.get('split', ''))
		if split not in SPLITS:
			raise ContractError(
				f'{location}.split={split!r} is forbidden; only train/validation are allowed.'
			)
		seen_splits.add(split)
		allowed_sources, manifest_sha256 = manifests[split]
		if sequence.get('source') not in allowed_sources:
			raise ContractError(
				f'{location}.source={sequence.get("source")!r} is outside the packaged '
				f'{split} manifest (test/support sources are forbidden).'
			)
		if sequence.get('manifest_sha256') != manifest_sha256:
			raise ContractError(f'{location}.manifest_sha256 does not match {split}.')
		sequence_id = _sequence_id(sequence)
		if sequence_id in seen_ids:
			raise ContractError(f'Duplicate sequence_id {sequence_id!r}.')
		seen_ids.add(sequence_id)
		frames = sequence.get('frames')
		if not isinstance(frames, list) or len(frames) != SEQUENCE_LENGTH:
			raise ContractError(
				f'{location}.frames must contain exactly {SEQUENCE_LENGTH} frames.'
			)
		normalized_frames = []
		for expected_t, frame in enumerate(frames):
			frame_location = f'{location}.frames[{expected_t}]'
			if not isinstance(frame, Mapping):
				raise ContractError(f'{frame_location} must be an object.')
			t = int(frame.get('t', -1))
			if t != expected_t:
				raise ContractError(
					f'{frame_location}.t must be {expected_t}; sequences must be contiguous.'
				)
			annotation_required = bool(frame.get(
				'annotation_required', t in ANNOTATION_TIMES
			))
			if annotation_required != (t in ANNOTATION_TIMES):
				raise ContractError(
					f'{frame_location}.annotation_required must be '
					f'{t in ANNOTATION_TIMES} for t={t}.'
				)
			points_value = frame.get('points')
			points = _role_mapping(
				points_value,
				location=f'{frame_location}.points',
				converter=lambda item, item_location: _point(
					item, location=item_location, allow_none=not annotation_required
				),
			)
			if annotation_required and any(points[role] is None for role in ROLES):
				raise ContractError(f'{frame_location} is missing a required point label.')
			if not annotation_required and any(points[role] is not None for role in ROLES):
				raise ContractError(
					f'{frame_location} contains point ground truth outside {ANNOTATION_TIMES}.'
				)
			image_path = _resolve_asset(
				path.parent, frame.get('image'), location=f'{frame_location}.image'
			)
			image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.uint8)
			if image.shape != (64, 64, 3) or image.dtype != np.uint8:
				raise ContractError(
					f'{frame_location}.image must be the native 64x64 uint8 RGB observation.'
				)
			fingerprint = frame.get('image_sha256')
			if not isinstance(fingerprint, str) or fingerprint != _image_sha256(image):
				raise ContractError(f'Image fingerprint mismatch at {frame_location}.')
			for role, point in points.items():
				if point is not None and not (
					0 <= point[0] < image.shape[1] and 0 <= point[1] < image.shape[0]
				):
					raise ContractError(
						f'{frame_location}.points.{role} lies outside the RGB frame.'
					)
			mask_record = frame.get('cutie_mask')
			if isinstance(mask_record, Mapping):
				if mask_record.get('encoding') != 'indexed_png_uint8':
					raise ContractError(
						f'{frame_location}.cutie_mask has an unsupported encoding.'
					)
				objects = mask_record.get('objects')
				if objects != AUDIT_MASK_OBJECTS:
					raise ContractError(
						f'{frame_location}.cutie_mask objects must map string labels 1/2/3 '
						'to proximal_link/distal_link/goal.'
					)
				mask_value = mask_record.get('image')
				status = mask_record.get('status')
				if mask_value is None and status != 'not_required':
					raise ContractError(
						f'{frame_location}.cutie_mask without an image must be not_required.'
					)
				if mask_value is not None and status != 'optional_manual':
					raise ContractError(
						f'{frame_location}.cutie_mask with an image must be optional_manual.'
					)
			else:
				# Compatibility with early hand-written audits that stored only a path.
				mask_value = mask_record
				mask_record = None
			cutie_mask = _resolve_asset(
				path.parent, mask_value, location=f'{frame_location}.cutie_mask.image'
			)
			if cutie_mask is not None:
				mask = np.asarray(Image.open(cutie_mask))
				if mask.shape[:2] != image.shape[:2] or mask.ndim != 2:
					raise ContractError(
						f'{frame_location}.cutie_mask must be a same-size indexed PNG.'
					)
				if not set(np.unique(mask).tolist()).issubset({0, 1, 2, 3}):
					raise ContractError(
						f'{frame_location}.cutie_mask may contain labels 0, 1, 2, 3 only.'
					)
			normalized_frames.append({
				'index': int(frame.get('index', expected_t)),
				't': t,
				'image': str(frame['image']),
				'image_path': image_path,
				'image_sha256': fingerprint,
				'image_shape': tuple(image.shape),
				'points': points,
				'cutie_mask': (
					dict(mask_record) if isinstance(mask_record, Mapping)
					else (str(mask_value) if mask_value else None)
				),
				'cutie_mask_path': cutie_mask,
				'source_frame_index': frame.get('source_frame_index'),
				'environment_step': frame.get('environment_step'),
				'annotation_required': annotation_required,
			})
		normalized_sequences.append({
			'sequence_id': sequence_id,
			'split': split,
			'episode': sequence.get('episode'),
			'source': sequence.get('source'),
			'start_frame': sequence.get('start_frame'),
			'warmup_steps': sequence.get('warmup_steps'),
			'manifest_sha256': sequence.get('manifest_sha256'),
			'frames': normalized_frames,
		})
	if seen_splits != set(SPLITS):
		raise ContractError(
			f'Audit must contain both train and validation sequences, got {sorted(seen_splits)}.'
		)
	for split in SPLITS:
		split_sequences = [item for item in normalized_sequences if item['split'] == split]
		if len(split_sequences) != EPISODES_PER_SPLIT:
			raise ContractError(
				f'Audit must contain exactly {EPISODES_PER_SPLIT} {split} sequences.'
			)
		if [item['episode'] for item in split_sequences] != list(range(EPISODES_PER_SPLIT)):
			raise ContractError(f'{split} episodes must be ordered 0, 1, 2.')
		if len({item['source'] for item in split_sequences}) != EPISODES_PER_SPLIT:
			raise ContractError(f'{split} audit sequences must use distinct sources.')
	return {
		'path': path,
		'sha256': _file_sha256(path),
		'collection': dict(collection),
		'sequences': normalized_sequences,
	}


def _confidence(value: Any, location: str) -> float | None:
	if value is None:
		return None
	try:
		result = float(value)
	except (TypeError, ValueError) as exc:
		raise ContractError(f'{location} must be numeric or null.') from exc
	if not math.isfinite(result) or not 0.0 <= result <= 1.0:
		raise ContractError(f'{location} must lie in [0, 1].')
	return result


def _boolean(value: Any, location: str) -> bool:
	if not isinstance(value, (bool, np.bool_)):
		raise ContractError(f'{location} must be boolean.')
	return bool(value)


def _runtime(value: Any, location: str) -> float:
	try:
		result = float(value)
	except (TypeError, ValueError) as exc:
		raise ContractError(f'{location} must be a non-negative number.') from exc
	if not math.isfinite(result) or result < 0:
		raise ContractError(f'{location} must be a non-negative finite number.')
	return result


def _largest_component(mask: np.ndarray) -> np.ndarray:
	"""Keep the largest 8-connected component of a small binary mask."""
	mask = np.asarray(mask, dtype=bool)
	if mask.ndim != 2 or not mask.any():
		return np.zeros_like(mask, dtype=bool)
	visited = np.zeros_like(mask, dtype=bool)
	best = []
	height, width = mask.shape
	for start_y, start_x in np.argwhere(mask):
		if visited[start_y, start_x]:
			continue
		stack = [(int(start_y), int(start_x))]
		visited[start_y, start_x] = True
		component = []
		while stack:
			y, x = stack.pop()
			component.append((y, x))
			for dy in (-1, 0, 1):
				for dx in (-1, 0, 1):
					if dx == 0 and dy == 0:
						continue
					ny, nx = y + dy, x + dx
					if (
						0 <= ny < height and 0 <= nx < width
						and mask[ny, nx] and not visited[ny, nx]
					):
						visited[ny, nx] = True
						stack.append((ny, nx))
		if len(component) > len(best):
			best = component
	result = np.zeros_like(mask, dtype=bool)
	if best:
		y, x = np.asarray(best, dtype=np.int64).T
		result[y, x] = True
	return result


def _trimmed_axis_endpoint(
	mask: np.ndarray,
	origin_xy: list[float] | np.ndarray,
	*,
	percentile: float = 95.0,
) -> list[float] | None:
	"""Estimate a thick link's far end without selecting its outermost pixel."""
	component = _largest_component(mask)
	yx = np.argwhere(component)
	if len(yx) < 2:
		return None
	coordinates = yx[:, ::-1].astype(np.float64)
	origin = np.asarray(origin_xy, dtype=np.float64)
	center = coordinates.mean(axis=0)
	centered = coordinates - center
	covariance = centered.T @ centered / max(len(coordinates) - 1, 1)
	eigenvalues, eigenvectors = np.linalg.eigh(covariance)
	axis = eigenvectors[:, int(np.argmax(eigenvalues))]
	if float(np.dot(axis, center - origin)) < 0:
		axis = -axis
	if abs(float(np.dot(axis, center - origin))) < 1e-8:
		farthest = coordinates[int(np.argmax(np.linalg.norm(coordinates - origin, axis=1)))]
		direction = farthest - origin
		if float(np.linalg.norm(direction)) > 1e-8:
			axis = direction / np.linalg.norm(direction)
	perpendicular = np.asarray([-axis[1], axis[0]], dtype=np.float64)
	offsets = coordinates - origin
	longitudinal = offsets @ axis
	transverse = offsets @ perpendicular
	target = float(np.percentile(longitudinal, percentile))
	spread = max(float(np.ptp(longitudinal)), 1.0)
	band = np.abs(longitudinal - target) <= max(0.75, 0.03 * spread)
	if not band.any():
		band[int(np.argmin(np.abs(longitudinal - target)))] = True
	# The transverse median is the center of the selected 95%-projection
	# cross-section, avoiding the systematic link-thickness overshoot of a
	# farthest-pixel endpoint.
	cross_center = float(np.median(transverse[band]))
	endpoint = origin + axis * target + perpendicular * cross_center
	height, width = component.shape
	endpoint[0] = np.clip(endpoint[0], 0.0, width - 0.5)
	endpoint[1] = np.clip(endpoint[1], 0.0, height - 0.5)
	return endpoint.astype(np.float64).tolist()


def points_from_cutie_masks(
	object_masks: Mapping[str, np.ndarray],
	*,
	base_xy: tuple[float, float] = (31.5, 31.5),
) -> dict[str, list[float] | None]:
	"""Map the fixed three-object Cutie contract to four control points."""
	if set(object_masks) != set(CUTIE_OBJECTS):
		raise ContractError(f'Cutie masks must contain exactly {CUTIE_OBJECTS}.')
	base = [float(base_xy[0]), float(base_xy[1])]
	elbow = _trimmed_axis_endpoint(object_masks['proximal_link'], base)
	tip = (
		None if elbow is None
		else _trimmed_axis_endpoint(object_masks['distal_link'], elbow)
	)
	goal_component = _largest_component(object_masks['goal'])
	goal_yx = np.argwhere(goal_component)
	goal = None
	if len(goal_yx):
		goal = goal_yx[:, ::-1].astype(np.float64).mean(axis=0).tolist()
	return {'base': base, 'elbow': elbow, 'control_tip': tip, 'goal': goal}


def _object_path_mapping(
	value: Any,
	*,
	root: Path,
	location: str,
	object_roles: tuple[str, ...] = CUTIE_OBJECTS,
	require_relative: bool = True,
) -> dict[str, Path]:
	if value is None:
		return {}
	if not isinstance(value, Mapping) or set(value) != set(object_roles):
		raise ContractError(f'{location} must contain exactly {object_roles}.')
	return {
		role: _resolve_asset(
			root, value[role], location=f'{location}.{role}',
			require_relative=require_relative,
		)
		for role in object_roles
	}


def _read_object_masks(paths: Mapping[str, Path], image_shape) -> dict[str, np.ndarray]:
	result = {}
	for role, path in paths.items():
		mask = np.asarray(Image.open(path))
		if mask.ndim != 2 or tuple(mask.shape) != tuple(image_shape[:2]):
			raise ContractError(
				f'Cutie object mask {path} must have shape {tuple(image_shape[:2])}.'
			)
		result[role] = mask.astype(bool)
	return result


def _normalize_prediction_frame(
	frame: Mapping[str, Any],
	*,
	location: str,
	asset_root: Path,
	expected_t: int,
	backend: str,
	image_shape: tuple[int, ...],
	trusted_assets: bool = False,
	cutie_decoder: Any | None = None,
	cutie_point_decoder: str | None = None,
	cutie_object_schema: str = CUTIE_OBJECT_SCHEMA_LEGACY,
	require_backend_object_status: bool = False,
	require_point_decoder_mode: bool = False,
) -> dict[str, Any]:
	if int(frame.get('t', -1)) != expected_t:
		raise ContractError(f'{location}.t must be {expected_t}.')
	if backend == 'cutie':
		decoder_mode = _resolve_cutie_decoder_mode(
			cutie_decoder, cutie_point_decoder
		)
		object_roles = _validate_cutie_decoder_schema_pair(
			decoder_mode, cutie_object_schema
		)
	else:
		decoder_mode = None
		object_roles = CUTIE_OBJECTS
	if backend == 'cutie':
		declared_decoder_mode = frame.get('point_decoder')
		if declared_decoder_mode is None:
			if require_point_decoder_mode:
				raise ContractError(
					f'{location} must declare point_decoder={decoder_mode!r}.'
				)
		elif (
			require_point_decoder_mode
			and declared_decoder_mode != decoder_mode
		):
			raise ContractError(
				f'{location}.point_decoder={declared_decoder_mode!r} does not '
				f'match requested {decoder_mode!r}; decoder-version migration is '
				'not allowed.'
			)
	object_mask_paths = _object_path_mapping(
		frame.get('masks'), root=asset_root, location=f'{location}.masks',
		object_roles=object_roles,
		require_relative=not trusted_assets,
	)
	object_feature_paths = _object_path_mapping(
		frame.get('features'), root=asset_root, location=f'{location}.features',
		object_roles=object_roles,
		require_relative=not trusted_assets,
	)
	mask_value = frame.get('mask', frame.get('cutie_mask'))
	mask_path = _resolve_asset(
		asset_root, mask_value, location=f'{location}.mask',
		require_relative=not trusted_assets,
	)
	# Cutie comparison is defined by the explicitly selected object-mask schema.
	# Never trust cached keypoints as a substitute: even when present, recompute
	# all four task points from the declared masks.
	derived_from_masks = backend == 'cutie'
	decoder_result = None
	decoder_runtime_ms = 0.0
	decoder_diagnostics = None
	geometry_confidence = None
	backend_object_confidence = None
	backend_object_lost = None
	if derived_from_masks:
		if not object_mask_paths:
			raise ContractError(
				f'{location} requires Cutie object masks {object_roles}.'
			)
		object_masks = _read_object_masks(object_mask_paths, image_shape)
		if cutie_decoder is None:
			points = points_from_cutie_masks(object_masks)
		else:
			started = perf_counter()
			try:
				decoder_result = cutie_decoder.decode(object_masks)
			except Exception as exc:
				raise ContractError(
					f'{location} Cutie point decoding failed: '
					f'{type(exc).__name__}: {exc}'
				) from exc
			decoder_runtime_ms = (perf_counter() - started) * 1000.0
			if not isinstance(decoder_result.points, Mapping) or set(
				decoder_result.points
			) != set(ROLES):
				raise ContractError(f'{location} decoder points must contain {ROLES}.')
			points = {
				role: _point(
					decoder_result.points[role],
					location=f'{location}.decoded_points.{role}',
					allow_none=True,
				)
				for role in ROLES
			}
	else:
		points = _role_mapping(
			frame.get('points'),
			location=f'{location}.points',
			converter=lambda value, item_location: _point(
				value, location=item_location, allow_none=True
			),
		)
	confidence_value = frame.get('confidence')
	lost_value = frame.get('lost', False)
	if decoder_result is not None:
		if not isinstance(decoder_result.point_confidence, Mapping) or set(
			decoder_result.point_confidence
		) != set(ROLES):
			raise ContractError(
				f'{location} decoder point_confidence must contain {ROLES}.'
			)
		if not isinstance(decoder_result.lost, Mapping) or set(
			decoder_result.lost
		) != set(ROLES):
			raise ContractError(f'{location} decoder lost must contain {ROLES}.')
		confidence = {
			role: _confidence(
				decoder_result.point_confidence[role],
				f'{location}.decoder_point_confidence.{role}',
			)
			for role in ROLES
		}
		lost = {
			role: _boolean(
				decoder_result.lost[role], f'{location}.decoder_lost.{role}'
			)
			for role in ROLES
		}
		geometry_confidence = _confidence(
			decoder_result.geometry_confidence,
			f'{location}.decoder_geometry_confidence',
		)
		decoder_diagnostics = decoder_result.diagnostics
		try:
			json.dumps(decoder_diagnostics, allow_nan=False)
		except (TypeError, ValueError) as exc:
			raise ContractError(
				f'{location} decoder diagnostics must be finite JSON data.'
			) from exc
		# Validate and retain Cutie's object-level status for diagnosis, but do
		# not let the old proximal/distal role mapping overwrite the task-specific
		# geometry confidence produced from the masks.
		decoder_diagnostics = dict(decoder_diagnostics)
		if decoder_mode == CUTIE_DECODER_WHOLE_ARM:
			has_persisted_confidence = 'backend_object_confidence' in frame
			has_persisted_lost = 'backend_object_lost' in frame
			if has_persisted_confidence != has_persisted_lost:
				raise ContractError(
					f'{location} must provide backend_object_confidence and '
					'backend_object_lost together.'
				)
			if require_backend_object_status and not has_persisted_confidence:
				raise ContractError(
					f'{location} whole-arm cache requires persisted '
					'backend_object_confidence and backend_object_lost.'
				)
			raw_confidence = (
				frame['backend_object_confidence']
				if has_persisted_confidence else confidence_value
			)
			raw_lost = (
				frame['backend_object_lost']
				if has_persisted_lost else lost_value
			)
			if not isinstance(raw_confidence, Mapping) or set(
				raw_confidence
			) != set(object_roles):
				raise ContractError(
					f'{location} whole-arm backend object confidence must contain '
					f'exactly {object_roles}.'
				)
			if not isinstance(raw_lost, Mapping) or set(raw_lost) != set(
				object_roles
			):
				raise ContractError(
					f'{location} whole-arm backend object lost must contain exactly '
					f'{object_roles}.'
				)
			backend_object_confidence = {
				role: _confidence(
					raw_confidence[role],
					f'{location}.backend_object_confidence.{role}',
				)
				for role in object_roles
			}
			if any(value is None for value in backend_object_confidence.values()):
				raise ContractError(
					f'{location} whole-arm backend object confidence values '
					'must be numeric and cannot be null.'
				)
			backend_object_lost = {
				role: _boolean(
					raw_lost[role], f'{location}.backend_object_lost.{role}'
				)
				for role in object_roles
			}
			decoder_diagnostics['backend_object_confidence'] = dict(
				backend_object_confidence
			)
			decoder_diagnostics['backend_object_lost'] = dict(
				backend_object_lost
			)
			decoder_diagnostics['backend_status_policy'] = (
				_whole_arm_backend_status_policy()
			)
		if isinstance(confidence_value, Mapping):
			confidence_roles = tuple(confidence_value)
			if set(confidence_roles) == set(object_roles) and not (
				decoder_mode == CUTIE_DECODER_WHOLE_ARM
				and 'backend_object_confidence' in frame
			):
				diagnostic_key = 'backend_object_confidence'
				roles_to_validate = object_roles
			elif set(confidence_roles) == set(ROLES):
				# Historical caches were written after normalization and therefore
				# contain four point-role confidences rather than the adapter's three
				# object-role confidences. Validate them for cache integrity, but never
				# use them to override the mask-only decoder.
				diagnostic_key = 'cached_point_confidence'
				roles_to_validate = ROLES
			else:
				raise ContractError(
					f'{location}.confidence must contain exactly {object_roles} '
					f'or historical point roles {ROLES}.'
				)
			decoder_diagnostics[diagnostic_key] = {
				role: _confidence(
					confidence_value[role], f'{location}.confidence.{role}'
				)
				for role in roles_to_validate
			}
		elif confidence_value is not None:
			decoder_diagnostics['legacy_cached_scalar_confidence'] = _confidence(
				confidence_value, f'{location}.confidence'
			)
		if isinstance(lost_value, Mapping):
			lost_roles = tuple(lost_value)
			if set(lost_roles) == set(object_roles) and not (
				decoder_mode == CUTIE_DECODER_WHOLE_ARM
				and 'backend_object_lost' in frame
			):
				diagnostic_key = 'backend_object_lost'
				roles_to_validate = object_roles
			elif set(lost_roles) == set(ROLES):
				diagnostic_key = 'cached_point_lost'
				roles_to_validate = ROLES
			else:
				raise ContractError(
					f'{location}.lost must contain exactly {object_roles} '
					f'or historical point roles {ROLES}.'
				)
			decoder_diagnostics[diagnostic_key] = {
				role: _boolean(lost_value[role], f'{location}.lost.{role}')
				for role in roles_to_validate
			}
		else:
			decoder_diagnostics['legacy_cached_scalar_lost'] = _boolean(
				lost_value, f'{location}.lost'
			)
		if decoder_mode == CUTIE_DECODER_WHOLE_ARM:
			# Official Cutie defocus is independent evidence. Fuse confidence by
			# the frozen minimum rule and fail closed on a lost tracked object.
			for object_role, point_roles in (
				('whole_arm', ('elbow', 'control_tip')),
				('goal', ('goal',)),
			):
				for role in point_roles:
					confidence[role] = min(
						confidence[role], backend_object_confidence[object_role]
					)
					if backend_object_lost[object_role]:
						points[role] = None
						confidence[role] = 0.0
						lost[role] = True
			geometry_confidence = min(
				confidence['elbow'],
				confidence['control_tip'],
				confidence['goal'],
			)
	elif isinstance(confidence_value, Mapping) and set(confidence_value) == set(CUTIE_OBJECTS):
		object_confidence = {
			role: _confidence(confidence_value[role], f'{location}.confidence.{role}')
			for role in CUTIE_OBJECTS
		}
		confidence = {
			'base': 1.0,
			'elbow': object_confidence['proximal_link'],
			'control_tip': object_confidence['distal_link'],
			'goal': object_confidence['goal'],
		}
	elif confidence_value is None and derived_from_masks:
		confidence = {
			role: (0.0 if points[role] is None else 1.0) for role in ROLES
		}
	else:
		confidence = _role_mapping(
			confidence_value,
			location=f'{location}.confidence',
			converter=_confidence,
			allow_scalar=True,
		)
	if decoder_result is not None:
		pass
	elif isinstance(lost_value, Mapping) and set(lost_value) == set(CUTIE_OBJECTS):
		object_lost = {
			role: _boolean(lost_value[role], f'{location}.lost.{role}')
			for role in CUTIE_OBJECTS
		}
		lost = {
			'base': False,
			'elbow': object_lost['proximal_link'],
			'control_tip': object_lost['proximal_link'] or object_lost['distal_link'],
			'goal': object_lost['goal'],
		}
	else:
		lost = _role_mapping(
			lost_value,
			location=f'{location}.lost',
			converter=_boolean,
			allow_scalar=True,
		)
	for role in ROLES:
		lost[role] = bool(lost[role] or points[role] is None)
	backend_runtime_ms = _runtime(
		frame.get('backend_runtime_ms', frame.get('runtime_ms')),
		f'{location}.backend_runtime_ms',
	)
	feature_path = _resolve_asset(
		asset_root, frame.get('feature'), location=f'{location}.feature',
		require_relative=not trusted_assets,
	)
	return {
		'index': int(frame.get('index', expected_t)),
		't': expected_t,
		'points': points,
		'confidence': confidence,
		'lost': lost,
		'runtime_ms': backend_runtime_ms + decoder_runtime_ms,
		'backend_runtime_ms': backend_runtime_ms,
		'decoder_runtime_ms': decoder_runtime_ms,
		'mask': str(mask_value) if mask_value is not None else None,
		'mask_path': mask_path,
		'object_masks': {role: str(path) for role, path in object_mask_paths.items()},
		'object_mask_paths': object_mask_paths,
		'feature': str(frame['feature']) if frame.get('feature') else None,
		'feature_path': feature_path,
		'object_features': {role: str(path) for role, path in object_feature_paths.items()},
		'object_feature_paths': object_feature_paths,
		'points_derived_from_masks': derived_from_masks,
		'point_decoder': (
			decoder_mode if derived_from_masks else None
		),
		'object_schema': cutie_object_schema if derived_from_masks else None,
		'backend_object_confidence': backend_object_confidence,
		'backend_object_lost': backend_object_lost,
		'geometry_confidence': geometry_confidence,
		'decoder_diagnostics': decoder_diagnostics,
		'fallback': bool(frame.get('fallback', False)),
	}


def load_prediction_cache(
	path: Path,
	*,
	backend: str,
	audit: Mapping[str, Any],
	cutie_decoder: Any | None = None,
	cutie_point_decoder: str | None = None,
	cutie_object_schema: str = CUTIE_OBJECT_SCHEMA_LEGACY,
) -> dict[str, Any]:
	path = path.expanduser().resolve()
	cutie_decoder_mode = None
	cutie_object_roles = CUTIE_OBJECTS
	if backend == 'cutie':
		cutie_decoder_mode = _resolve_cutie_decoder_mode(
			cutie_decoder, cutie_point_decoder
		)
		cutie_object_roles = _validate_cutie_decoder_schema_pair(
			cutie_decoder_mode, cutie_object_schema
		)
	data = _json_load(path)
	_reject_privileged_fields(data, location=f'{backend}_cache')
	if data.get('format') != CACHE_FORMAT:
		raise ContractError(f'{path} is not a {CACHE_FORMAT} cache.')
	if data.get('backend') != backend:
		raise ContractError(
			f'{path} backend={data.get("backend")!r}, expected {backend!r}.'
		)
	if tuple(data.get('roles', ())) != ROLES:
		raise ContractError(f'{path} roles must be ordered exactly as {ROLES}.')
	if data.get('audit_sha256') != audit['sha256']:
		raise ContractError(
			f'{path} was not produced for audit SHA-256 {audit["sha256"]}.'
		)
	sequences = data.get('sequences')
	if not isinstance(sequences, list):
		raise ContractError(f'{path} is missing sequences.')
	metadata = data.get('metadata', {})
	if not isinstance(metadata, Mapping):
		raise ContractError(f'{path} metadata must be an object.')
	resolution = metadata.get('resolution')
	if not isinstance(resolution, Mapping):
		raise ContractError(f'{path} metadata.resolution is required.')
	if (
		resolution.get('source') != 'native64'
		or resolution.get('native_size') != [64, 64]
		or resolution.get('true_high_resolution') is not False
	):
		raise ContractError(
			f'{path} must declare native64 input and true_high_resolution=false.'
		)
	processed_size = resolution.get('processed_size')
	if not isinstance(processed_size, list) or len(processed_size) != 2 or min(processed_size) < 1:
		raise ContractError(f'{path} metadata.resolution.processed_size is invalid.')
	if backend == 'dino' and processed_size != [448, 448]:
		raise ContractError(
			f'{path} must use the matched current DINO 448x448 preprocessing contract.'
		)
	if backend == 'cutie':
		for field in ('requested_tracker_size', 'actual_tracker_size', 'processed_size'):
			value = resolution.get(field)
			if not isinstance(value, list) or len(value) != 2 or min(value) < 1:
				raise ContractError(f'{path} metadata.resolution.{field} is invalid.')
		if not (
			resolution['requested_tracker_size']
			== resolution['actual_tracker_size']
			== resolution['processed_size']
			== [448, 448]
		):
			raise ContractError(
				f'{path} Cutie cache must declare matched '
				'requested=actual=processed=[448, 448].'
			)
	metadata = dict(metadata)
	legacy_cache_redecoded = False
	if backend == 'cutie':
		cached_object_schema = metadata.get('object_schema')
		cached_object_roles = metadata.get('object_roles')
		legacy_object_schema_inferred = False
		if cached_object_schema is None:
			if cutie_object_schema != CUTIE_OBJECT_SCHEMA_LEGACY:
				raise ContractError(
					f'{path} lacks metadata.object_schema; historical caches are '
					'three-object only and cannot be used as whole-arm caches.'
				)
			legacy_object_schema_inferred = True
		elif cached_object_schema != cutie_object_schema:
			raise ContractError(
				f'{path} object_schema={cached_object_schema!r}, explicitly '
				f'requested {cutie_object_schema!r}; no schema migration is allowed.'
			)
		if cached_object_roles is None:
			if cutie_object_schema != CUTIE_OBJECT_SCHEMA_LEGACY:
				raise ContractError(
					f'{path} lacks metadata.object_roles required by '
					f'{cutie_object_schema!r}.'
				)
			legacy_object_schema_inferred = True
		elif tuple(cached_object_roles) != cutie_object_roles:
			raise ContractError(
				f'{path} metadata.object_roles must be ordered exactly as '
				f'{cutie_object_roles}.'
			)
		if (
			cutie_object_schema == CUTIE_OBJECT_SCHEMA_WHOLE_ARM
			and metadata.get('backend_status_policy')
			!= _whole_arm_backend_status_policy()
		):
			raise ContractError(
				f'{path} metadata.backend_status_policy does not match the '
				'whole-arm confidence/lost fusion contract.'
			)
		cached_decoder_mode = metadata.get('point_decoder_mode')
		if cutie_decoder_mode == CUTIE_DECODER_WHOLE_ARM_V2:
			if cached_decoder_mode is None:
				raise ContractError(
					f'{path} lacks metadata.point_decoder_mode required by the '
					f'{CUTIE_DECODER_WHOLE_ARM_V2!r} cache contract. Old v1 '
					'whole-arm caches cannot be reused or silently re-decoded.'
				)
			if cached_decoder_mode != cutie_decoder_mode:
				raise ContractError(
					f'{path} point_decoder_mode={cached_decoder_mode!r}, explicitly '
					f'requested {cutie_decoder_mode!r}; decoder-version migration is '
					'not allowed.'
				)
		cached_decoder = metadata.get('point_decoder')
		requested_decoder = (
			_legacy_cutie_decoder_metadata()
			if cutie_decoder is None else cutie_decoder.metadata()
		)
		if cached_decoder is None:
			if cutie_object_schema == CUTIE_OBJECT_SCHEMA_WHOLE_ARM:
				raise ContractError(
					f'{path} lacks metadata.point_decoder required by the '
					'whole-arm cache contract.'
				)
			if cutie_decoder is not None:
				legacy_cache_redecoded = True
		elif cached_decoder != requested_decoder:
			raise ContractError(
				f'{path} point_decoder metadata does not match the explicitly '
				'requested decoder/support calibration.'
			)
		metadata['point_decoder'] = requested_decoder
		metadata['point_decoder_mode'] = cutie_decoder_mode
		metadata['legacy_cache_redecoded'] = legacy_cache_redecoded
		metadata['object_schema'] = cutie_object_schema
		metadata['object_roles'] = list(cutie_object_roles)
		metadata['legacy_object_schema_inferred'] = legacy_object_schema_inferred
	audit_by_id = {item['sequence_id']: item for item in audit['sequences']}
	normalized = {}
	for sequence_index, sequence in enumerate(sequences):
		location = f'{backend}_cache.sequences[{sequence_index}]'
		if not isinstance(sequence, Mapping):
			raise ContractError(f'{location} must be an object.')
		sequence_id = _sequence_id(sequence)
		if sequence_id not in audit_by_id:
			raise ContractError(f'{location} has unknown sequence_id {sequence_id!r}.')
		if sequence_id in normalized:
			raise ContractError(f'{location} duplicates sequence_id {sequence_id!r}.')
		audit_sequence = audit_by_id[sequence_id]
		if sequence.get('split') != audit_sequence['split']:
			raise ContractError(f'{location}.split does not match the audit.')
		frames = sequence.get('frames')
		if not isinstance(frames, list) or len(frames) != SEQUENCE_LENGTH:
			raise ContractError(f'{location} must contain all {SEQUENCE_LENGTH} frames.')
		if cutie_decoder is not None:
			cutie_decoder.reset()
		normalized_frames = []
		for t, frame in enumerate(frames):
			normalized_frames.append(_normalize_prediction_frame(
				frame,
				location=f'{location}.frames[{t}]',
				asset_root=path.parent,
				expected_t=t,
				backend=backend,
				image_shape=audit_sequence['frames'][t]['image_shape'],
				cutie_decoder=cutie_decoder,
				cutie_point_decoder=cutie_decoder_mode,
				cutie_object_schema=cutie_object_schema,
				require_backend_object_status=(
					cutie_object_schema == CUTIE_OBJECT_SCHEMA_WHOLE_ARM
				),
				require_point_decoder_mode=(
					cutie_decoder_mode == CUTIE_DECODER_WHOLE_ARM_V2
				),
			))
		normalized[sequence_id] = {
			'sequence_id': sequence_id,
			'split': audit_sequence['split'],
			'frames': normalized_frames,
		}
	if set(normalized) != set(audit_by_id):
		missing = sorted(set(audit_by_id) - set(normalized))
		raise ContractError(f'{path} is missing audit sequences: {missing}.')
	return {
		'backend': backend,
		'source': 'cache',
		'cache_path': path,
		'metadata': metadata,
		'sequences': normalized,
	}


def _persist_binary_mask(mask: Any, path: Path, image_shape: tuple[int, ...]) -> Path:
	if hasattr(mask, 'detach'):
		mask = mask.detach().cpu().numpy()
	array = np.asarray(mask)
	if array.ndim != 2 or tuple(array.shape) != tuple(image_shape[:2]):
		raise ContractError(
			f'Adapter mask must have shape {tuple(image_shape[:2])}, got {array.shape}.'
		)
	path.parent.mkdir(parents=True, exist_ok=True)
	Image.fromarray(array.astype(bool).astype(np.uint8) * 255, mode='L').save(path)
	return path.resolve()


def _persist_features(features: Any, path: Path) -> Path:
	if hasattr(features, 'detach'):
		features = features.detach().cpu().numpy()
	array = np.asarray(features, dtype=np.float32)
	if array.ndim != 1 or not np.isfinite(array).all():
		raise ContractError(f'Cutie object feature must be a finite vector, got {array.shape}.')
	path.parent.mkdir(parents=True, exist_ok=True)
	np.save(path, array, allow_pickle=False)
	return path.resolve()


def run_cutie_adapter(
	*,
	audit: Mapping[str, Any],
	repo_path: Path,
	checkpoint_path: Path,
	support_path: Path,
	device: str,
	tracker_size: tuple[int, int],
	asset_dir: Path,
	model_size: str = 'small',
	config_dir: Path | None = None,
	prompt_radius: float = 2.0,
	amp: bool = True,
	cutie_decoder: Any | None = None,
	cutie_point_decoder: str | None = None,
	cutie_object_schema: str = CUTIE_OBJECT_SCHEMA_LEGACY,
) -> dict[str, Any]:
	"""Run the in-tree official OC-STORM Cutie adapter with permanent support.

	Audit masks and point labels are never passed to the tracker.  The verified
	manual support pack is loaded exactly once into permanent memory, then every
	audit sequence is processed by ``track_episode`` in temporal order.
	"""
	if tuple(tracker_size) != (448, 448):
		raise ContractError(
			'Matched A/B evaluation requires Cutie tracker_size=(448, 448).'
		)
	cutie_decoder_mode = _resolve_cutie_decoder_mode(
		cutie_decoder, cutie_point_decoder
	)
	object_roles = _validate_cutie_decoder_schema_pair(
		cutie_decoder_mode, cutie_object_schema
	)
	project_root = Path(__file__).resolve().parents[2]
	if str(project_root) not in sys.path:
		sys.path.insert(0, str(project_root))
	try:
		from tdmpc2.perception.cutie_oc_adapter import (
			CutieOCAdapter,
			CutieOCConfig,
			inspect_cutie_installation,
			load_point_support_prompts,
		)
	except Exception as exc:
		raise ContractError(
			f'Unable to import the in-tree CutieOCAdapter: {type(exc).__name__}: {exc}'
		) from exc
	config = CutieOCConfig(
		repo_path=repo_path,
		checkpoint_path=checkpoint_path,
		role_names=object_roles,
		model_size=model_size,
		device=device,
		output_device='cpu',
		config_dir=config_dir,
		expected_input_size=(64, 64),
		tracker_size=tuple(tracker_size),
		amp=bool(amp),
		return_object_features=True,
		object_schema=cutie_object_schema,
	)
	try:
		preflight_result = inspect_cutie_installation(config, import_check=True)
		support = load_point_support_prompts(
			support_path,
			radius_px=prompt_radius,
			object_schema=cutie_object_schema,
		)
		adapter = CutieOCAdapter(config)
		adapter.add_support_prompts(support)
	except Exception as exc:
		raise ContractError(
			f'Cutie preflight/model/support initialization failed: '
			f'{type(exc).__name__}: {exc}'
		) from exc
	sequences = {}
	actual_tracker_sizes = set()
	for sequence in audit['sequences']:
		if cutie_decoder is not None:
			cutie_decoder.reset()
		images = [
			np.asarray(Image.open(frame['image_path']).convert('RGB'), dtype=np.uint8)
			for frame in sequence['frames']
		]
		try:
			results = adapter.track_episode(images)
		except Exception as exc:
			raise ContractError(
				f'Cutie failed on sequence {sequence["sequence_id"]!r}: '
				f'{type(exc).__name__}: {exc}'
			) from exc
		if len(results) != SEQUENCE_LENGTH:
			raise ContractError(
				f'Cutie returned {len(results)} frames for {sequence["sequence_id"]!r}.'
			)
		prediction_frames = []
		for t, (audit_frame, result) in enumerate(zip(sequence['frames'], results)):
			if tuple(result.role_names) != object_roles:
				raise ContractError(f'Cutie result roles must be {object_roles}.')
			if tuple(result.input_size) != (64, 64):
				raise ContractError(f'Cutie consumed non-native input size {result.input_size}.')
			actual_tracker_sizes.add(tuple(int(value) for value in result.tracker_size))
			if tuple(result.tracker_size) != tuple(tracker_size):
				raise ContractError(
					f'Cutie actual tracker size {result.tracker_size} differs from '
					f'requested {tuple(tracker_size)}.'
				)
			sequence_asset_dir = asset_dir / _slug(sequence['sequence_id'])
			mask_paths = {}
			feature_paths = {}
			for role_index, role in enumerate(object_roles):
				mask_paths[role] = str(_persist_binary_mask(
					result.masks[role_index],
					sequence_asset_dir / 'masks' / f't_{t:02d}_{role}.png',
					audit_frame['image_shape'],
				))
				if result.object_features is not None:
					feature_paths[role] = str(_persist_features(
						result.object_features[role_index],
						sequence_asset_dir / 'features' / f't_{t:02d}_{role}.npy',
					))
			frame_value = {
				'index': audit_frame['index'],
				't': t,
				'masks': mask_paths,
				'features': feature_paths or None,
				'confidence': dict(zip(object_roles, result.confidence.tolist())),
				'lost': dict(zip(object_roles, result.lost.tolist())),
				'runtime_ms': float(result.runtime_ms),
			}
			prediction_frames.append(_normalize_prediction_frame(
				frame_value,
				location=f'cutie_adapter.{sequence["sequence_id"]}.frames[{t}]',
				asset_root=asset_dir.resolve(),
				expected_t=t,
				backend='cutie',
				image_shape=audit_frame['image_shape'],
				trusted_assets=True,
				cutie_decoder=cutie_decoder,
				cutie_point_decoder=cutie_decoder_mode,
				cutie_object_schema=cutie_object_schema,
			))
		sequences[sequence['sequence_id']] = {
			'sequence_id': sequence['sequence_id'],
			'split': sequence['split'],
			'frames': prediction_frames,
		}
	return {
		'backend': 'cutie',
		'source': 'official_oc_storm_cutie',
		'cache_path': None,
		'metadata': {
			'preflight': dict(preflight_result),
			'support': support.metadata,
			'runtime': adapter.runtime_summary(),
			'object_schema': cutie_object_schema,
			'object_roles': list(object_roles),
			**(
				{'backend_status_policy': _whole_arm_backend_status_policy()}
				if cutie_object_schema == CUTIE_OBJECT_SCHEMA_WHOLE_ARM else {}
			),
			'point_decoder': (
				_legacy_cutie_decoder_metadata()
				if cutie_decoder is None else cutie_decoder.metadata()
			),
			'point_decoder_mode': cutie_decoder_mode,
			'legacy_cache_redecoded': False,
			'resolution': {
				'source': 'native64',
				'native_size': [64, 64],
				'processed_size': list(next(iter(actual_tracker_sizes))),
				'requested_tracker_size': list(tracker_size),
				'actual_tracker_size': list(next(iter(actual_tracker_sizes))),
				'true_high_resolution': False,
				'resize_note': 'Interpolation adds no sensor information.',
			},
		},
		'sequences': sequences,
	}


def _load_anchor_config(path: Path) -> dict[str, Any]:
	try:
		import yaml
	except ImportError as exc:
		raise ContractError('PyYAML is required for live DINO evaluation.') from exc
	with path.open(encoding='utf-8') as stream:
		value = yaml.safe_load(stream)
	if not isinstance(value, Mapping):
		raise ContractError(f'Anchor config is not a mapping: {path}')
	return dict(value)


def _load_teacher_class(path: Path):
	path = path.expanduser().resolve()
	if not path.is_file():
		raise ContractError(f'DINO anchor implementation not found: {path}')
	module_name = f'_visual_small_dino_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}'
	spec = importlib.util.spec_from_file_location(module_name, path)
	if spec is None or spec.loader is None:
		raise ContractError(f'Unable to import DINO anchor implementation: {path}')
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	spec.loader.exec_module(module)
	try:
		return module.CausalAnchorStateTeacher
	except AttributeError as exc:
		raise ContractError(f'{path} has no CausalAnchorStateTeacher.') from exc


def _validate_manual_support_pack(path: Path):
	"""Close the support provenance gap before the legacy teacher is constructed."""
	path = path.expanduser().resolve()
	data = _json_load(path)
	_reject_privileged_fields(data, location='dino_support')
	if tuple(data.get('roles', ())) != ROLES:
		raise ContractError(f'DINO support roles must be ordered exactly as {ROLES}.')
	collection = data.get('collection')
	if not isinstance(collection, Mapping) or (
		collection.get('split') != 'support'
		or collection.get('observation') != 'rgb'
		or collection.get('label_policy') != 'manual_rgb_only'
	):
		raise ContractError(
			'DINO support must declare split=support, observation=rgb, and '
			'label_policy=manual_rgb_only.'
		)
	records = data.get('records')
	if not isinstance(records, list) or len(records) != 6:
		raise ContractError('DINO support must contain exactly six records.')
	for expected_index, record in enumerate(records):
		location = f'dino_support.records[{expected_index}]'
		if not isinstance(record, Mapping) or record.get('index') != expected_index:
			raise ContractError(f'{location} must have a contiguous ordered index.')
		if record.get('video_split') != 'support':
			raise ContractError(f'{location}.video_split must be support.')
		image_path = _resolve_asset(path.parent, record.get('image'), location=f'{location}.image')
		image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.uint8)
		if image.shape != (64, 64, 3) or record.get('image_sha256') != _image_sha256(image):
			raise ContractError(f'{location} image/hash breaks the native64 RGB contract.')
		_role_mapping(
			record.get('points'),
			location=f'{location}.points',
			converter=lambda value, item_location: _point(
				value, location=item_location, allow_none=False
			),
		)


def _device_sync(torch, device):
	if device.type == 'cuda':
		torch.cuda.synchronize(device)


def run_dino_anchor(
	*,
	audit: Mapping[str, Any],
	config_path: Path,
	impl_path: Path,
	support_path: Path,
	device_name: str | None,
	repo: str | None,
	checkpoint: Path | None,
) -> dict[str, Any]:
	"""Run the repository's frozen CausalAnchorStateTeacher on audit RGB only."""
	try:
		import torch
	except ImportError as exc:
		raise ContractError('PyTorch is required for live DINO evaluation.') from exc
	config_values = _load_anchor_config(config_path)
	required = (
		'support_path', 'dino_model', 'dino_repo', 'dino_input_size',
		'color_window', 'color_score_weight', 'goal_dino_score_weight',
		'goal_color_score_weight', 'role_contrast_weight', 'structured_candidates',
		'structured_nms_radius', 'structure_weight', 'structure_length_tolerance',
		'temporal_beam_size', 'temporal_unary_scale_floor',
		'temporal_position_scale', 'temporal_position_weight',
		'temporal_velocity_scale', 'temporal_velocity_weight',
		'temporal_elbow_weight',
	)
	teacher_values = {}
	for key in required:
		config_key = f'flat_anchor_{key}'
		if config_key not in config_values:
			raise ContractError(f'Missing {config_key} in {config_path}.')
		teacher_values[key] = config_values[config_key]
	support_path = support_path.expanduser().resolve()
	if support_path.suffix.lower() != '.json':
		raise ContractError('Live DINO evaluation requires a manual JSON support pack.')
	_validate_manual_support_pack(support_path)
	teacher_values['support_path'] = str(support_path)
	teacher_values['allow_diagnostic_support'] = False
	teacher_values['dino_repo'] = repo or teacher_values['dino_repo']
	teacher_values['dino_checkpoint'] = (
		str(checkpoint.expanduser().resolve()) if checkpoint is not None
		else config_values.get('flat_anchor_dino_checkpoint')
	)
	teacher_values['confidence_scale'] = config_values.get(
		'flat_anchor_confidence_scale', 2.0
	)
	if int(teacher_values['dino_input_size']) != 448:
		raise ContractError(
			'Current matched DINO evaluation requires dino_input_size=448; '
			'it is still an interpolation of the native 64x64 observation.'
		)
	device = torch.device(device_name or config_values.get('flat_anchor_device', 'cuda:0'))
	teacher_class = _load_teacher_class(impl_path)
	teacher = teacher_class(_Config(teacher_values), device=device)
	sequences = {}
	for sequence in audit['sequences']:
		state = teacher.initial_state(batch=1, beam_size=teacher.beam_size, device=device)
		prediction_frames = []
		for t, audit_frame in enumerate(sequence['frames']):
			# PIL-backed arrays can be read-only even when contiguous. Make the
			# native 64x64 RGB boundary explicitly writable before torch conversion.
			image = np.array(
				Image.open(audit_frame['image_path']).convert('RGB'),
				dtype=np.uint8,
				copy=True,
			)
			image_tensor = torch.as_tensor(image, device='cpu').unsqueeze(0)
			is_first = torch.tensor([t == 0], dtype=torch.bool, device=device)
			_device_sync(torch, device)
			started = perf_counter()
			observation, state = teacher.extract(
				image_tensor, state, is_first, output_device=device
			)
			_device_sync(torch, device)
			runtime_ms = 1000.0 * (perf_counter() - started)
			values = observation.detach().cpu().float().reshape(25).numpy()
			height, width = image.shape[:2]
			scale = np.asarray([width - 1, height - 1], dtype=np.float32)
			points = ((values[:8].reshape(4, 2) + 1.0) * 0.5 * scale).tolist()
			confidences = values[20:24].tolist()
			fallback = bool(values[24] >= 0.5)
			prediction_frames.append({
				'index': audit_frame['index'],
				't': t,
				'points': dict(zip(ROLES, points)),
				'confidence': dict(zip(ROLES, confidences)),
				'lost': {
					role: bool(fallback or confidences[index] <= 0.0)
					for index, role in enumerate(ROLES)
				},
				'runtime_ms': runtime_ms,
				'mask': None,
				'mask_path': None,
				'feature': None,
				'feature_path': None,
				'fallback': fallback,
			})
		sequences[sequence['sequence_id']] = {
			'sequence_id': sequence['sequence_id'],
			'split': sequence['split'],
			'frames': prediction_frames,
		}
	return {
		'backend': 'dino',
		'source': 'live_anchor',
		'cache_path': None,
		'metadata': {
			'impl_path': str(impl_path.expanduser().resolve()),
			'support_path': str(support_path),
			'device': str(device),
			'teacher_metrics': teacher.metrics(),
			'resolution': {
				'source': 'native64',
				'native_size': [64, 64],
				'processed_size': [
					int(teacher_values['dino_input_size']),
					int(teacher_values['dino_input_size']),
				],
				'true_high_resolution': False,
				'resize_note': 'Interpolation adds no sensor information.',
			},
		},
		'sequences': sequences,
	}


def _slug(value: Any) -> str:
	result = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('._')
	return result or 'sequence'


def _relative_asset(path: Path | None, root: Path) -> str | None:
	if path is None:
		return None
	try:
		return path.resolve().relative_to(root.resolve()).as_posix()
	except ValueError as exc:
		raise ContractError(f'Prediction asset escapes cache root {root}: {path}') from exc


def _cache_json_value(
	predictions: Mapping[str, Any], audit: Mapping[str, Any], cache_root: Path
) -> dict[str, Any]:
	sequences = []
	for audit_sequence in audit['sequences']:
		prediction_sequence = predictions['sequences'][audit_sequence['sequence_id']]
		frames = []
		for frame in prediction_sequence['frames']:
			cache_frame = {
				'index': frame['index'],
				't': frame['t'],
				'points': frame['points'],
				'confidence': frame['confidence'],
				'lost': frame['lost'],
				'runtime_ms': frame['runtime_ms'],
				'backend_runtime_ms': frame.get(
					'backend_runtime_ms', frame['runtime_ms']
				),
				'decoder_runtime_ms': frame.get('decoder_runtime_ms', 0.0),
				'fallback': frame.get('fallback', False),
				'mask': _relative_asset(frame.get('mask_path'), cache_root),
				'masks': (
					{
						role: _relative_asset(path, cache_root)
						for role, path in frame.get('object_mask_paths', {}).items()
					} or None
				),
				'feature': _relative_asset(frame.get('feature_path'), cache_root),
				'features': (
					{
						role: _relative_asset(path, cache_root)
						for role, path in frame.get('object_feature_paths', {}).items()
					} or None
				),
				'points_derived_from_masks': frame.get('points_derived_from_masks', False),
				'point_decoder': frame.get('point_decoder'),
				'geometry_confidence': frame.get('geometry_confidence'),
				'decoder_diagnostics': frame.get('decoder_diagnostics'),
			}
			if frame.get('backend_object_confidence') is not None:
				cache_frame['backend_object_confidence'] = dict(
					frame['backend_object_confidence']
				)
			if frame.get('backend_object_lost') is not None:
				cache_frame['backend_object_lost'] = dict(
					frame['backend_object_lost']
				)
			frames.append(cache_frame)
		sequences.append({
			'sequence_id': audit_sequence['sequence_id'],
			'split': audit_sequence['split'],
			'episode': audit_sequence['episode'],
			'source': audit_sequence['source'],
			'start_frame': audit_sequence['start_frame'],
			'frames': frames,
		})
	return {
		'format': CACHE_FORMAT,
		'backend': predictions['backend'],
		'audit_sha256': audit['sha256'],
		'roles': list(ROLES),
		'metadata': predictions.get('metadata', {}),
		'sequences': sequences,
	}


def _percentile(values: list[float], percentile: float) -> float | None:
	if not values:
		return None
	return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _summary(values: list[float]) -> dict[str, Any]:
	return {
		'n': len(values),
		'mean': float(np.mean(values)) if values else None,
		'median': float(np.median(values)) if values else None,
		'p95': _percentile(values, 95),
	}


def _rankdata(values: np.ndarray) -> np.ndarray:
	order = np.argsort(values, kind='mergesort')
	ranks = np.empty(len(values), dtype=np.float64)
	start = 0
	while start < len(values):
		end = start + 1
		while end < len(values) and values[order[end]] == values[order[start]]:
			end += 1
		ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
		start = end
	return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
	if len(left) < 2 or len(left) != len(right):
		return None
	left_rank = _rankdata(np.asarray(left, dtype=np.float64))
	right_rank = _rankdata(np.asarray(right, dtype=np.float64))
	if float(left_rank.std()) == 0.0 or float(right_rank.std()) == 0.0:
		return None
	return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _identity_assignment(
	predicted: Mapping[str, list[float] | None],
	ground_truth: Mapping[str, list[float]],
	lost: Mapping[str, bool],
) -> dict[str, bool]:
	valid_roles = [
		role for role in ROLES if predicted[role] is not None and not lost[role]
	]
	result = {role: False for role in ROLES}
	if not valid_roles:
		return result
	best_targets = None
	best_cost = math.inf
	for targets in itertools.permutations(ROLES, len(valid_roles)):
		cost = sum(
			float(np.linalg.norm(
				np.asarray(predicted[role]) - np.asarray(ground_truth[target])
			))
			for role, target in zip(valid_roles, targets)
		)
		if cost < best_cost:
			best_cost = cost
			best_targets = targets
	for role, target in zip(valid_roles, best_targets or ()):
		result[role] = role != target
	return result


def _mask_iou(
	ground_truth_path: Path, prediction_frame: Mapping[str, Any]
) -> dict[str, float]:
	ground_truth = np.asarray(Image.open(ground_truth_path))
	object_paths = prediction_frame.get('object_mask_paths', {})
	if object_paths:
		objects = _read_object_masks(object_paths, (*ground_truth.shape, 3))
		if set(objects) == set(CUTIE_WHOLE_ARM_OBJECTS):
			prediction_masks = {
				'controlled_arm': objects['whole_arm'],
				'goal': objects['goal'],
			}
		elif set(objects) == set(CUTIE_OBJECTS):
			prediction_masks = {
				'proximal_link': objects['proximal_link'],
				'distal_link': objects['distal_link'],
				'controlled_arm': np.logical_or(
					objects['proximal_link'], objects['distal_link']
				),
				'goal': objects['goal'],
			}
		else:
			raise ContractError(
				'Object-mask IoU requires either the legacy three-object roles '
				f'or whole-arm roles {CUTIE_WHOLE_ARM_OBJECTS}.'
			)
	else:
		prediction_path = prediction_frame.get('mask_path')
		if prediction_path is None:
			raise ContractError('Mask IoU requested without a prediction mask.')
		prediction = np.asarray(Image.open(prediction_path))
		if prediction.ndim != 2 or prediction.shape != ground_truth.shape:
			raise ContractError(
				f'Prediction mask {prediction_path} does not match GT mask '
				f'{ground_truth_path}.'
			)
		prediction_masks = {
			'proximal_link': prediction == 1,
			'distal_link': prediction == 2,
			'controlled_arm': np.isin(prediction, (1, 2)),
			'goal': prediction == 3,
		}
	ground_truth_masks = {
		'proximal_link': ground_truth == 1,
		'distal_link': ground_truth == 2,
		'controlled_arm': np.isin(ground_truth, (1, 2)),
		'goal': ground_truth == 3,
	}
	result = {}
	for name in prediction_masks:
		left = ground_truth_masks[name]
		right = prediction_masks[name]
		union = int(np.logical_or(left, right).sum())
		result[name] = 1.0 if union == 0 else float(np.logical_and(left, right).sum() / union)
	return result


def compute_split_metrics(
	*,
	audit: Mapping[str, Any],
	predictions: Mapping[str, Any],
	split: str,
	confidence_threshold: float,
) -> dict[str, Any]:
	role_values = {
		role: {'epe': [], 'pck_hits': 0, 'total': 0, 'lost': 0,
			'id_switch': 0, 'confident_wrong': 0}
		for role in ROLES
	}
	tip_goal_vector_errors = []
	tip_goal_distance_errors = []
	tip_goal_signed_distance_errors = []
	tip_goal_gt_distances = []
	tip_goal_pred_distances = []
	runtimes = []
	mask_values = {name: [] for name in MASK_METRICS}
	sequence_count = 0
	labeled_frames = 0
	for sequence in audit['sequences']:
		if sequence['split'] != split:
			continue
		sequence_count += 1
		prediction_sequence = predictions['sequences'][sequence['sequence_id']]
		for audit_frame, prediction_frame in zip(
			sequence['frames'], prediction_sequence['frames']
		):
			runtimes.append(float(prediction_frame['runtime_ms']))
			if not audit_frame['annotation_required']:
				continue
			labeled_frames += 1
			ground_truth = audit_frame['points']
			predicted = prediction_frame['points']
			lost = prediction_frame['lost']
			identity_switch = _identity_assignment(predicted, ground_truth, lost)
			for role in ROLES:
				values = role_values[role]
				values['total'] += 1
				if lost[role] or predicted[role] is None:
					values['lost'] += 1
					continue
				error = float(np.linalg.norm(
					np.asarray(predicted[role], dtype=np.float64)
					- np.asarray(ground_truth[role], dtype=np.float64)
				))
				values['epe'].append(error)
				values['pck_hits'] += int(error <= PCK_THRESHOLD_PX)
				values['id_switch'] += int(identity_switch[role])
				confidence = prediction_frame['confidence'][role]
				values['confident_wrong'] += int(
					confidence is not None
					and confidence >= confidence_threshold
					and error > CONFIDENT_WRONG_ERROR_PX
				)
			roles_available = all(
				prediction_frame['points'][role] is not None
				and not prediction_frame['lost'][role]
				for role in ('control_tip', 'goal')
			)
			if roles_available:
				gt_vector = (
					np.asarray(audit_frame['points']['goal'], dtype=np.float64)
					- np.asarray(audit_frame['points']['control_tip'], dtype=np.float64)
				)
				pred_vector = (
					np.asarray(prediction_frame['points']['goal'], dtype=np.float64)
					- np.asarray(prediction_frame['points']['control_tip'], dtype=np.float64)
				)
				gt_distance = float(np.linalg.norm(gt_vector))
				pred_distance = float(np.linalg.norm(pred_vector))
				tip_goal_vector_errors.append(float(np.linalg.norm(pred_vector - gt_vector)))
				tip_goal_distance_errors.append(abs(pred_distance - gt_distance))
				tip_goal_signed_distance_errors.append(pred_distance - gt_distance)
				tip_goal_gt_distances.append(gt_distance)
				tip_goal_pred_distances.append(pred_distance)
			if (
				audit_frame['cutie_mask_path'] is not None
				and (
					prediction_frame.get('mask_path') is not None
					or prediction_frame.get('object_mask_paths')
				)
			):
				ious = _mask_iou(audit_frame['cutie_mask_path'], prediction_frame)
				for name, value in ious.items():
					mask_values[name].append(value)

	roles = {}
	for role, values in role_values.items():
		total = values['total']
		roles[role] = {
			'epe': _summary(values['epe']),
			'pck_at_4': float(values['pck_hits'] / total) if total else None,
			'pck_hits': values['pck_hits'],
			'total': total,
			'lost_count': values['lost'],
			'lost_rate': float(values['lost'] / total) if total else None,
			'id_switch_count': values['id_switch'],
			'id_switch_rate': float(values['id_switch'] / total) if total else None,
			'confident_wrong_count': values['confident_wrong'],
			'confident_wrong_rate': (
				float(values['confident_wrong'] / total) if total else None
			),
		}
	total_instances = sum(item['total'] for item in role_values.values())
	incident_counts = {
		'lost': sum(item['lost'] for item in role_values.values()),
		'id_switch': sum(item['id_switch'] for item in role_values.values()),
		'confident_wrong': sum(
			item['confident_wrong'] for item in role_values.values()
		),
	}
	mask_report = {
		name: _summary(values) for name, values in mask_values.items() if values
	}
	if mask_report:
		macro_values = [
			value
			for name in MASK_METRICS
			for value in mask_values[name]
		]
		mask_report['macro'] = _summary(macro_values)
	return {
		'sequences': sequence_count,
		'frames': len(runtimes),
		'labeled_frames': labeled_frames,
		'roles': roles,
		'tip_goal': {
			'vector_error': _summary(tip_goal_vector_errors),
			'distance_abs_error': _summary(tip_goal_distance_errors),
			'distance_signed_bias': (
				float(np.mean(tip_goal_signed_distance_errors))
				if tip_goal_signed_distance_errors else None
			),
			'distance_spearman': _spearman(
				tip_goal_gt_distances, tip_goal_pred_distances
			),
			'paired_distances': len(tip_goal_gt_distances),
		},
		'incidents': {
			'total_role_frames': total_instances,
			**{
				f'{name}_count': count
				for name, count in incident_counts.items()
			},
			**{
				f'{name}_rate': float(count / total_instances) if total_instances else None
				for name, count in incident_counts.items()
			},
		},
		'mask_iou': mask_report or None,
		'runtime_ms': _summary(runtimes),
	}


def _gate(name: str, value: float | None, operator: str, threshold: float) -> dict[str, Any]:
	if operator == '<=':
		passed = value is not None and value <= threshold
	elif operator == '>=':
		passed = value is not None and value >= threshold
	else:
		raise ValueError(operator)
	return {
		'name': name,
		'value': value,
		'operator': operator,
		'threshold': threshold,
		'pass': bool(passed),
	}


def absolute_gates(metrics: Mapping[str, Any], max_ms_per_frame: float | None) -> dict[str, Any]:
	checks = []
	role_limits = {
		'base': (3.0, 6.0),
		'elbow': (4.0, 8.0),
		'control_tip': (3.0, 6.0),
		'goal': (3.0, 6.0),
	}
	for role, (mean_limit, p95_limit) in role_limits.items():
		role_metrics = metrics['roles'][role]
		checks.extend([
			_gate(f'{role}.epe.mean', role_metrics['epe']['mean'], '<=', mean_limit),
			_gate(f'{role}.epe.p95', role_metrics['epe']['p95'], '<=', p95_limit),
			_gate(f'{role}.pck_at_4', role_metrics['pck_at_4'], '>=', 0.80),
		])
	checks.extend([
		_gate(
			'tip_goal.distance_spearman',
			metrics['tip_goal']['distance_spearman'], '>=', 0.80,
		),
		_gate('incidents.lost_rate', metrics['incidents']['lost_rate'], '<=', 0.10),
		_gate(
			'incidents.id_switch_rate', metrics['incidents']['id_switch_rate'], '<=', 0.05
		),
		_gate(
			'incidents.confident_wrong_rate',
			metrics['incidents']['confident_wrong_rate'], '<=', 0.05,
		),
	])
	if max_ms_per_frame is not None:
		checks.append(_gate(
			'runtime_ms.mean', metrics['runtime_ms']['mean'], '<=', max_ms_per_frame
		))
	return {'pass': all(check['pass'] for check in checks), 'checks': checks}


def evaluate_backend(
	*,
	audit: Mapping[str, Any],
	predictions: Mapping[str, Any],
	confidence_threshold: float,
	max_ms_per_frame: float | None,
) -> dict[str, Any]:
	splits = {}
	for split in SPLITS:
		metrics = compute_split_metrics(
			audit=audit,
			predictions=predictions,
			split=split,
			confidence_threshold=confidence_threshold,
		)
		metrics['gates'] = absolute_gates(metrics, max_ms_per_frame)
		splits[split] = metrics
	return {
		'source': predictions['source'],
		'cache_path': (
			str(predictions['cache_path']) if predictions.get('cache_path') else None
		),
		'metadata': predictions.get('metadata', {}),
		'splits': splits,
		'absolute_gate_pass': all(item['gates']['pass'] for item in splits.values()),
	}


def selection_decision(
	method_reports: Mapping[str, Any],
	*,
	minimum_improvement_px: float,
	minimum_relative_improvement: float,
) -> dict[str, Any]:
	if 'cutie' not in method_reports:
		dino_ready = bool(method_reports['dino']['absolute_gate_pass'])
		return {
			'status': 'dino_only_ready' if dino_ready else 'perception_not_ready',
			'recommended_backend': 'dino' if dino_ready else None,
			'selected_backend': 'dino' if dino_ready else None,
			'cutie_rl_eligible': False,
			'pass': dino_ready,
			'reasons': [
				'A Cutie result is required before changing the RL perception backend.'
				if dino_ready else
				'DINO fails an absolute gate and Cutie has not been evaluated.'
			],
		}
	dino = method_reports['dino']['splits']['validation']
	cutie = method_reports['cutie']['splits']['validation']
	dino_values = [dino['roles'][role]['epe']['mean'] for role in ('control_tip', 'goal')]
	cutie_values = [cutie['roles'][role]['epe']['mean'] for role in ('control_tip', 'goal')]
	reasons = []
	comparable = all(value is not None for value in dino_values + cutie_values)
	if not comparable:
		improvement_px = relative = None
		non_regression = False
	else:
		dino_average = float(np.mean(dino_values))
		cutie_average = float(np.mean(cutie_values))
		improvement_px = dino_average - cutie_average
		relative = improvement_px / max(dino_average, 1e-12)
		non_regression = all(
			cutie_value <= dino_value
			for cutie_value, dino_value in zip(cutie_values, dino_values)
		)
	dino_ready = bool(method_reports['dino']['absolute_gate_pass'])
	cutie_ready = bool(method_reports['cutie']['absolute_gate_pass'])
	comparison_pass = bool(
		comparable and non_regression
		and improvement_px >= minimum_improvement_px
		and relative >= minimum_relative_improvement
	)
	comparison_applied = dino_ready and cutie_ready
	if cutie_ready and not dino_ready:
		status, selected = 'cutie_selected', 'cutie'
		reasons.append('Cutie is the only backend that passes every absolute gate.')
	elif comparison_applied and comparison_pass:
		status, selected = 'cutie_selected', 'cutie'
		reasons.append('Both pass absolute gates and Cutie passes the validation improvement rule.')
	elif dino_ready:
		status, selected = 'keep_dino', 'dino'
		if not cutie_ready:
			reasons.append('Cutie fails at least one absolute train/validation gate.')
		elif not comparable:
			reasons.append('Validation tip/goal EPE is incomplete; keep the passing DINO backend.')
		else:
			if improvement_px < minimum_improvement_px:
				reasons.append(
					f'Validation tip/goal mean EPE improves by {improvement_px:.3f}px, '
					f'below {minimum_improvement_px:.3f}px.'
				)
			if relative < minimum_relative_improvement:
				reasons.append(
					f'Validation tip/goal relative EPE improvement is {relative:.3%}, '
					f'below {minimum_relative_improvement:.3%}.'
				)
			if not non_regression:
				reasons.append(
					'At least one of validation control_tip/goal mean EPE regresses.'
				)
	else:
		status, selected = 'perception_not_ready', None
		reasons.append(
			'Both Cutie and DINO fail at least one absolute gate; '
			'inspect the failed gates and collect independent evidence before '
			'RL integration.'
		)
	eligible = selected == 'cutie'
	return {
		'status': status,
		'recommended_backend': selected,
		'selected_backend': selected,
		'cutie_rl_eligible': eligible,
		'pass': selected is not None,
		'validation_tip_goal_mean_epe_improvement_px': improvement_px,
		'validation_tip_goal_mean_epe_relative_improvement': relative,
		'minimum_improvement_px': minimum_improvement_px,
		'minimum_relative_improvement': minimum_relative_improvement,
		'per_role_non_regression': non_regression,
		'comparison_applied': comparison_applied,
		'comparison_pass': comparison_pass if comparison_applied else None,
		'dino_absolute_gate_pass': dino_ready,
		'cutie_absolute_gate_pass': cutie_ready,
		'reasons': reasons,
	}


def _draw_cross(draw: ImageDraw.ImageDraw, point, color, radius=3, width=1):
	x, y = (float(point[0]), float(point[1]))
	draw.line((x - radius, y, x + radius, y), fill=color, width=width)
	draw.line((x, y - radius, x, y + radius), fill=color, width=width)


def write_overlays(
	*,
	audit: Mapping[str, Any],
	predictions: Mapping[str, Mapping[str, Any]],
	output_dir: Path,
) -> list[str]:
	paths = []
	backend_styles = {'dino': 'cross', 'cutie': 'box'}
	for sequence in audit['sequences']:
		for audit_frame in sequence['frames']:
			if not audit_frame['annotation_required']:
				continue
			image = Image.open(audit_frame['image_path']).convert('RGB')
			draw = ImageDraw.Draw(image)
			for role in ROLES:
				point = audit_frame['points'][role]
				color = ROLE_COLORS[role]
				x, y = point
				draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color, outline=(0, 0, 0))
			for backend, backend_predictions in predictions.items():
				frame = backend_predictions['sequences'][sequence['sequence_id']]['frames'][audit_frame['t']]
				for role in ROLES:
					point = frame['points'][role]
					if point is None or frame['lost'][role]:
						continue
					color = ROLE_COLORS[role]
					if backend_styles.get(backend) == 'box':
						x, y = point
						draw.rectangle((x - 3, y - 3, x + 3, y + 3), outline=color, width=1)
					else:
						_draw_cross(draw, point, color, radius=4, width=1)
			legend = 'GT dot | DINO + | Cutie box'
			draw.rectangle((0, 0, min(image.width - 1, 153), 9), fill=(0, 0, 0))
			draw.text((1, 0), legend, fill=(255, 255, 255))
			path = (
				output_dir / sequence['split'] / _slug(sequence['sequence_id'])
				/ f't_{audit_frame["t"]:02d}.png'
			)
			path.parent.mkdir(parents=True, exist_ok=True)
			image.save(path)
			paths.append(str(path.resolve()))
	return paths


def write_csv(path: Path, report: Mapping[str, Any]):
	rows = []
	for backend, backend_report in report['methods'].items():
		for split, metrics in backend_report['splits'].items():
			for role, role_metrics in metrics['roles'].items():
				for name in ('mean', 'median', 'p95'):
					rows.append((backend, split, 'role', role, f'epe_{name}', role_metrics['epe'][name]))
				for name in ('pck_at_4', 'lost_rate', 'id_switch_rate', 'confident_wrong_rate'):
					rows.append((backend, split, 'role', role, name, role_metrics[name]))
			for group, summary_key in (
				('tip_goal_vector_error', 'vector_error'),
				('tip_goal_distance_abs_error', 'distance_abs_error'),
			):
				for name in ('mean', 'median', 'p95'):
					rows.append((backend, split, 'summary', '', f'{group}_{name}', metrics['tip_goal'][summary_key][name]))
			rows.extend([
				(backend, split, 'summary', '', 'tip_goal_distance_spearman', metrics['tip_goal']['distance_spearman']),
				(backend, split, 'summary', '', 'runtime_ms_mean', metrics['runtime_ms']['mean']),
				(backend, split, 'summary', '', 'runtime_ms_median', metrics['runtime_ms']['median']),
				(backend, split, 'summary', '', 'runtime_ms_p95', metrics['runtime_ms']['p95']),
			])
			if metrics['mask_iou']:
				for label, summary in metrics['mask_iou'].items():
					for name in ('mean', 'median', 'p95'):
						rows.append((backend, split, 'mask', label, f'iou_{name}', summary[name]))
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', encoding='utf-8', newline='') as stream:
		writer = csv.writer(stream)
		writer.writerow(('backend', 'split', 'scope', 'role_or_label', 'metric', 'value'))
		writer.writerows(rows)


def evaluate(
	*,
	annotations: Path,
	output_dir: Path,
	dino_cache: Path | None,
	dino_live: Mapping[str, Any] | None,
	cutie_cache: Path | None = None,
	cutie_live: Mapping[str, Any] | None = None,
	confidence_threshold: float = 0.5,
	max_ms_per_frame: float | None = None,
	minimum_improvement_px: float = 0.5,
	minimum_relative_improvement: float = 0.05,
	cutie_tracker_size: tuple[int, int] = (448, 448),
	cutie_point_decoder: str = CUTIE_DECODER_LEGACY,
	cutie_object_schema: str = CUTIE_OBJECT_SCHEMA_LEGACY,
	cutie_support_path: Path | None = None,
) -> dict[str, Any]:
	"""Library entry point; writes report JSON/CSV, caches, and overlays."""
	if (dino_cache is None) == (dino_live is None):
		raise ContractError('Provide exactly one of dino_cache or dino_live.')
	if cutie_cache is not None and cutie_live is not None:
		raise ContractError('Provide at most one of cutie_cache or cutie_live.')
	cutie_object_roles = _validate_cutie_decoder_schema_pair(
		cutie_point_decoder, cutie_object_schema
	)
	audit = load_audit(annotations)
	output_dir = output_dir.expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)
	if dino_cache is not None:
		dino_predictions = load_prediction_cache(dino_cache, backend='dino', audit=audit)
	else:
		dino_predictions = run_dino_anchor(audit=audit, **dict(dino_live))
	predictions = {'dino': dino_predictions}
	cutie_requested = cutie_cache is not None or cutie_live is not None
	resolved_cutie_support = cutie_support_path
	if cutie_live is not None:
		live_support = dict(cutie_live).get('support_path')
		if resolved_cutie_support is None:
			resolved_cutie_support = live_support
		elif live_support is not None and Path(resolved_cutie_support).expanduser().resolve() != Path(
			live_support
		).expanduser().resolve():
			raise ContractError(
				'Cutie permanent prompts and point decoder must use the same '
				'RGB-only support annotations file.'
			)
	cutie_decoder = (
		_build_cutie_point_decoder(
			cutie_point_decoder, support_path=resolved_cutie_support
		)
		if cutie_requested else None
	)
	if cutie_cache is not None:
		predictions['cutie'] = load_prediction_cache(
			cutie_cache,
			backend='cutie',
			audit=audit,
			cutie_decoder=cutie_decoder,
			cutie_point_decoder=cutie_point_decoder,
			cutie_object_schema=cutie_object_schema,
		)
	elif cutie_live is not None:
		predictions['cutie'] = run_cutie_adapter(
			audit=audit,
			asset_dir=output_dir / 'prediction_assets' / 'cutie_masks',
			cutie_decoder=cutie_decoder,
			cutie_point_decoder=cutie_point_decoder,
			cutie_object_schema=cutie_object_schema,
			**dict(cutie_live),
		)
	if 'cutie' in predictions:
		resolution = predictions['cutie']['metadata']['resolution']
		if tuple(resolution['requested_tracker_size']) != tuple(cutie_tracker_size):
			raise ContractError(
				f'Cutie cache/live requested tracker size '
				f'{resolution["requested_tracker_size"]} does not match evaluator '
				f'{tuple(cutie_tracker_size)}.'
			)

	for backend, backend_predictions in predictions.items():
		if backend_predictions['source'] != 'cache':
			cache_path = output_dir / f'{backend}_predictions.json'
			_json_dump(
				cache_path,
				_cache_json_value(backend_predictions, audit, output_dir),
			)
			backend_predictions['cache_path'] = cache_path
	methods = {
		backend: evaluate_backend(
			audit=audit,
			predictions=backend_predictions,
			confidence_threshold=confidence_threshold,
			max_ms_per_frame=max_ms_per_frame,
		)
		for backend, backend_predictions in predictions.items()
	}
	selection = selection_decision(
		methods,
		minimum_improvement_px=minimum_improvement_px,
		minimum_relative_improvement=minimum_relative_improvement,
	)
	overlays = write_overlays(
		audit=audit, predictions=predictions, output_dir=output_dir / 'overlays'
	)
	cutie_schema_report = (
		{
			'name': cutie_object_schema,
			'roles': list(cutie_object_roles),
			'point_decoder_mode': predictions['cutie']['metadata'][
				'point_decoder_mode'
			],
			'point_decoder_metadata': predictions['cutie']['metadata'][
				'point_decoder'
			],
			'mask_iou_roles': (
				['controlled_arm', 'goal']
				if cutie_object_schema == CUTIE_OBJECT_SCHEMA_WHOLE_ARM
				else list(MASK_METRICS)
			),
			'backend_status_policy': (
				_whole_arm_backend_status_policy()
				if cutie_object_schema == CUTIE_OBJECT_SCHEMA_WHOLE_ARM
				else None
			),
		}
		if 'cutie' in predictions else None
	)
	report = {
		'format': REPORT_FORMAT,
		'audit': {
			'path': str(audit['path']),
			'sha256': audit['sha256'],
			'format': AUDIT_FORMAT,
			'splits': list(SPLITS),
			'sequences': len(audit['sequences']),
			'no_test': True,
			'privileged_fields_rejected': sorted(FORBIDDEN_FIELDS),
		},
		'definitions': {
			'point_metric_frames': list(ANNOTATION_TIMES),
			'processed_frames_per_sequence': SEQUENCE_LENGTH,
			'pck_threshold_px': PCK_THRESHOLD_PX,
			'confident_wrong_confidence_threshold': confidence_threshold,
			'confident_wrong_error_threshold_px': CONFIDENT_WRONG_ERROR_PX,
			'lost': (
				'DINO and legacy Cutie: explicit lost flag or missing role point. '
				'Support-kinematic three-object Cutie: missing mask-derived geometry; '
				'cached object focus/lost remains diagnostic-only. Whole-arm Cutie: '
				'mask geometry OR official object defocus fails the affected points '
				'closed, and point confidence is the minimum of decoder and official '
				'object confidence. whole_arm_kinematic_v2 keeps the elbow at '
				'the calibrated L1 target for bounded endpoint shortfall and uses '
				'the observed centerline endpoint as the control tip.'
			),
			'id_switch': 'minimum-cost one-to-one role assignment maps a prediction to another GT role',
			'mask_labels': AUDIT_MASK_OBJECTS,
			'cutie_object_schema': cutie_schema_report,
			'mask_note': (
				'Cutie cached points are ignored and all points are deterministically '
				'recomputed from the object masks declared by cutie_object_schema. '
				'For whole_arm_goal_v1, controlled_arm IoU compares whole_arm against '
				'the union of audit proximal/distal labels and no split-link IoU is '
				'reported. Exact schema, roles, decoder, and calibration are recorded '
				'in methods.cutie.metadata. Mask IoU is optional and is not a hard gate.'
			),
		},
		'methods': methods,
		'selection': selection,
		'hard_gate_pass': bool(selection['pass']),
		'outputs': {
			'json': str((output_dir / 'report.json').resolve()),
			'csv': str((output_dir / 'metrics.csv').resolve()),
			'overlays': overlays,
		},
	}
	_json_dump(output_dir / 'report.json', report)
	write_csv(output_dir / 'metrics.csv', report)
	return report


def parse_args(argv=None):
	root = Path(__file__).resolve().parents[2]
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--annotations', type=Path, required=True)
	parser.add_argument('--output-dir', type=Path, required=True)
	parser.add_argument('--dino-cache', type=Path)
	parser.add_argument('--anchor-config', type=Path, default=root / 'tdmpc2' / 'config.yaml')
	parser.add_argument('--dino-impl', type=Path, default=root / '_anchor_source' / 'anchor_state.py')
	parser.add_argument('--dino-support', type=Path)
	parser.add_argument('--dino-device')
	parser.add_argument('--dino-repo')
	parser.add_argument('--dino-checkpoint', type=Path)
	parser.add_argument('--cutie-cache', type=Path)
	parser.add_argument(
		'--cutie-point-decoder',
		choices=CUTIE_DECODER_CHOICES,
		default=CUTIE_DECODER_LEGACY,
		help=(
			'Mask-to-point decoder. Kinematic modes must be selected explicitly '
			'and bind outputs to --cutie-support SHA/calibration.'
		),
	)
	parser.add_argument(
		'--cutie-object-schema',
		choices=CUTIE_OBJECT_SCHEMA_CHOICES,
		default=CUTIE_OBJECT_SCHEMA_LEGACY,
		help=(
			'Explicit tracker object protocol. whole_arm_kinematic_v2 requires '
			'whole_arm_goal_v1; all other decoders require legacy_three_object_v1.'
		),
	)
	parser.add_argument('--cutie-oc-repo', type=Path)
	parser.add_argument('--cutie-checkpoint', type=Path)
	parser.add_argument(
		'--cutie-support', type=Path,
		help='Permanent RGB-only support annotations; defaults to --dino-support.',
	)
	parser.add_argument('--cutie-device', default='cuda:0')
	parser.add_argument('--cutie-tracker-size', nargs=2, type=int, default=(448, 448),
		metavar=('HEIGHT', 'WIDTH'))
	parser.add_argument('--cutie-model-size', choices=('small', 'base'), default='small')
	parser.add_argument('--cutie-config-dir', type=Path)
	parser.add_argument('--cutie-prompt-radius', type=float, default=2.0)
	parser.add_argument('--cutie-amp', action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument('--confidence-threshold', type=float, default=0.5)
	parser.add_argument('--max-ms-per-frame', type=float)
	parser.add_argument('--minimum-improvement-px', type=float, default=0.5)
	parser.add_argument('--minimum-relative-improvement', type=float, default=0.05)
	parser.add_argument(
		'--fail-on-gate', action=argparse.BooleanOptionalAction, default=True,
		help='Exit 2 after writing outputs when the applicable hard gate fails.',
	)
	args = parser.parse_args(argv)
	if not 0 <= args.confidence_threshold <= 1:
		parser.error('--confidence-threshold must lie in [0, 1].')
	if args.max_ms_per_frame is not None and args.max_ms_per_frame <= 0:
		parser.error('--max-ms-per-frame must be positive.')
	if args.minimum_improvement_px < 0 or args.minimum_relative_improvement < 0:
		parser.error('Improvement thresholds must be non-negative.')
	if args.dino_cache is None and args.dino_support is None:
		parser.error('--dino-support is required when --dino-cache is absent.')
	cutie_live_requested = any((
		args.cutie_oc_repo is not None,
		args.cutie_checkpoint is not None,
		args.cutie_config_dir is not None,
	))
	if args.cutie_cache is not None and cutie_live_requested:
		parser.error('Use --cutie-cache or live Cutie options, not both.')
	if cutie_live_requested:
		if args.cutie_oc_repo is None or args.cutie_checkpoint is None:
			parser.error('Live Cutie requires --cutie-oc-repo and --cutie-checkpoint.')
		if args.cutie_support is None and args.dino_support is None:
			parser.error('Live Cutie requires --cutie-support or --dino-support.')
	cutie_requested = args.cutie_cache is not None or cutie_live_requested
	try:
		_validate_cutie_decoder_schema_pair(
			args.cutie_point_decoder, args.cutie_object_schema
		)
	except ContractError as exc:
		parser.error(str(exc))
	if args.cutie_point_decoder != CUTIE_DECODER_LEGACY:
		if not cutie_requested:
			parser.error(
				f'--cutie-point-decoder {args.cutie_point_decoder} requires '
				'--cutie-cache or live Cutie.'
			)
		if args.cutie_support is None and args.dino_support is None:
			parser.error(
				f'{args.cutie_point_decoder} requires --cutie-support or '
				'--dino-support.'
			)
	if min(args.cutie_tracker_size) < 1:
		parser.error('--cutie-tracker-size values must be positive.')
	if (args.cutie_cache is not None or cutie_live_requested) and tuple(
		args.cutie_tracker_size
	) != (448, 448):
		parser.error(
			'Matched DINO/Cutie A/B requires --cutie-tracker-size 448 448.'
		)
	if args.cutie_prompt_radius <= 0:
		parser.error('--cutie-prompt-radius must be positive.')
	return args


def main(argv=None) -> int:
	args = parse_args(argv)
	dino_live = None
	if args.dino_cache is None:
		dino_live = {
			'config_path': args.anchor_config,
			'impl_path': args.dino_impl,
			'support_path': args.dino_support,
			'device_name': args.dino_device,
			'repo': args.dino_repo,
			'checkpoint': args.dino_checkpoint,
		}
	cutie_live = None
	if args.cutie_oc_repo is not None:
		cutie_live = {
			'repo_path': args.cutie_oc_repo,
			'checkpoint_path': args.cutie_checkpoint,
			'support_path': args.cutie_support or args.dino_support,
			'device': args.cutie_device,
			'tracker_size': tuple(args.cutie_tracker_size),
			'model_size': args.cutie_model_size,
			'config_dir': args.cutie_config_dir,
			'prompt_radius': args.cutie_prompt_radius,
			'amp': args.cutie_amp,
		}
	try:
		report = evaluate(
			annotations=args.annotations,
			output_dir=args.output_dir,
			dino_cache=args.dino_cache,
			dino_live=dino_live,
			cutie_cache=args.cutie_cache,
			cutie_live=cutie_live,
			confidence_threshold=args.confidence_threshold,
			max_ms_per_frame=args.max_ms_per_frame,
			minimum_improvement_px=args.minimum_improvement_px,
			minimum_relative_improvement=args.minimum_relative_improvement,
			cutie_tracker_size=tuple(args.cutie_tracker_size),
			cutie_point_decoder=args.cutie_point_decoder,
			cutie_object_schema=args.cutie_object_schema,
			cutie_support_path=(args.cutie_support or args.dino_support),
		)
	except ContractError as exc:
		print(f'PERCEPTION_AB_CONTRACT_ERROR {exc}', file=sys.stderr)
		return 1
	print(json.dumps({
		'event': 'VISUAL_SMALL_PERCEPTION_AB',
		'hard_gate_pass': report['hard_gate_pass'],
		'selection': report['selection'],
		'report': report['outputs']['json'],
		'csv': report['outputs']['csv'],
	}, sort_keys=True, allow_nan=False))
	return 0 if report['hard_gate_pass'] or not args.fail_on_gate else 2


if __name__ == '__main__':
	raise SystemExit(main())
