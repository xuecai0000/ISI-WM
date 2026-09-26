"""Extract causal whole-arm/goal Cutie features from RGB rollout packs.

This is an argparse program, not a Hydra application.  It validates the full
rollout pack before importing the Cutie runtime, installs the verified support
pack once, resets non-permanent memory at every episode boundary, and tracks
native 64x64 RGB frames strictly in temporal order.  Transition arrays are
validated by the collector's public loader but are never passed to Cutie.

The public ``load_rollout_manifest`` and ``validate_feature_arrays`` helpers
intentionally do not import torch or the external OC-STORM checkout.  Contract
tests and ``--help`` therefore work on machines without a GPU/Cutie install.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import traceback
from typing import Any, Mapping

import numpy as np
from PIL import Image, UnidentifiedImageError


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROLLOUT_FORMAT = 'visual_small_object_rollout_v1'
FEATURE_FORMAT = 'visual_small_cutie_object_features_v1'
TASK = 'reacher-visual-small'
ALLOWED_SPLITS = ('train', 'validation')
OBJECT_SCHEMA = 'whole_arm_goal_v1'
ROLES = ('whole_arm', 'goal')
INPUT_SIZE = (64, 64)
TRACKER_SIZE = (448, 448)
FEATURE_DIM = 2048
FOREGROUND_QUERIES = 8

_SHA256_RE = re.compile(r'[0-9a-f]{64}')
_FEATURE_ARRAY_KEYS = {
	'features',
	'masks',
	'centroid_xy',
	'confidence',
	'lost',
	'valid',
	'mask_score',
	'runtime_ms',
}


class ContractError(RuntimeError):
	"""Raised before publishing output when an input/output contract is broken."""


@dataclass(frozen=True)
class FrameRecord:
	t: int
	image_path: Path
	image_relative_path: str
	image_sha256: str
	source_frame_index: int


@dataclass(frozen=True)
class SequenceRecord:
	sequence_id: str
	split: str
	episode: int
	source: str
	env_seed: int
	background_seed: int
	action_seed: int
	num_transitions: int
	frames: tuple[FrameRecord, ...]
	transitions_path: Path
	transitions_relative_path: str
	transitions_sha256: str
	source_selection_reset_attempt: int
	ended_by_environment: bool


@dataclass(frozen=True)
class RolloutManifest:
	path: Path
	root: Path
	file_sha256: str
	collection: Mapping[str, Any]
	sequences: tuple[SequenceRecord, ...]


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], location: str):
	actual = set(value)
	if actual != expected:
		missing = sorted(expected - actual)
		extra = sorted(actual - expected)
		raise ContractError(
			f'{location} keys differ from the frozen contract; '
			f'missing={missing}, extra={extra}.'
		)


def _require_sha256(value: Any, location: str) -> str:
	if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
		raise ContractError(f'{location} must be a lowercase SHA-256.')
	return value


def _sha256_file(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as handle:
		for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
			digest.update(chunk)
	return digest.hexdigest()


def _decoded_rgb_sha256(frame: np.ndarray) -> str:
	return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def _load_verified_rgb(path: Path, expected_sha256: str) -> np.ndarray:
	try:
		with Image.open(path) as image:
			if image.format != 'PNG':
				raise ContractError(f'Frame is not a PNG file: {path}.')
			frame = np.asarray(image.convert('RGB'), dtype=np.uint8)
	except (OSError, UnidentifiedImageError) as exc:
		raise ContractError(f'Could not decode rollout frame {path}: {exc}.') from exc
	if frame.shape != (INPUT_SIZE[0], INPUT_SIZE[1], 3):
		raise ContractError(
			f'Cutie accepts only decoded native64 RGB, got {frame.shape} at {path}.'
		)
	if frame.dtype != np.uint8:
		raise ContractError(f'Decoded RGB must be uint8 at {path}, got {frame.dtype}.')
	frame = np.ascontiguousarray(frame)
	actual_sha256 = _decoded_rgb_sha256(frame)
	if actual_sha256 != expected_sha256:
		raise ContractError(
			f'Decoded RGB SHA mismatch at {path}: expected={expected_sha256}, '
			f'actual={actual_sha256}.'
		)
	return frame


def _collector_asset_path(root: Path, raw_relative_path: str, *, suffix: str) -> Path:
	"""Resolve an asset already proved safe by the collector's public validator."""

	resolved = root.joinpath(*PurePosixPath(raw_relative_path).parts).resolve(strict=True)
	try:
		resolved.relative_to(root)
	except ValueError as exc:
		raise ContractError(f'Validated rollout asset now escapes its root: {resolved}.') from exc
	if not resolved.is_file() or resolved.suffix.lower() != suffix:
		raise ContractError(f'Validated rollout asset must be a {suffix} file: {resolved}.')
	return resolved


def _adapt_validated_frame(root: Path, frame: Mapping[str, Any]) -> FrameRecord:
	image_path = _collector_asset_path(root, frame['image'], suffix='.png')
	# Enforce extractor-specific native64/PNG semantics now, before any Cutie
	# import. The same bytes are verified again at the model-consumption boundary.
	_load_verified_rgb(image_path, frame['image_sha256'])
	return FrameRecord(
		t=int(frame['t']),
		image_path=image_path,
		image_relative_path=frame['image'],
		image_sha256=frame['image_sha256'],
		source_frame_index=int(frame['source_frame_index']),
	)


def load_rollout_manifest(path: str | Path) -> RolloutManifest:
	"""Validate a rollout with the collector's canonical loader and adapt it.

	The collector owns the exact JSON schema, split/source allowlists, seed
	contracts, path containment, decoded-RGB hashes, N+1 alignment, and transition
	NPZ validation. Keeping a second parser here would allow the two contracts to
	drift. Frames are decoded and hashed again immediately before Cutie sees them.
	"""

	if str(REPO_ROOT) not in sys.path:
		sys.path.insert(0, str(REPO_ROOT))
	try:
		from tdmpc2.tools.collect_visual_small_object_rollouts import (
			ALLOWED_SPLITS as COLLECTOR_ALLOWED_SPLITS,
			FORMAT as COLLECTOR_FORMAT,
			TASK as COLLECTOR_TASK,
			load_and_validate_rollout_manifest,
		)
	except Exception as exc:
		raise ContractError(
			'Could not import the rollout collector contract: '
			f'{type(exc).__name__}: {exc}'
		) from exc
	if (
		COLLECTOR_FORMAT != ROLLOUT_FORMAT
		or COLLECTOR_TASK != TASK
		or tuple(COLLECTOR_ALLOWED_SPLITS) != ALLOWED_SPLITS
	):
		raise ContractError('Extractor and collector contract constants disagree.')
	try:
		validated = load_and_validate_rollout_manifest(path)
	except Exception as exc:
		raise ContractError(
			f'Rollout collector validation failed: {type(exc).__name__}: {exc}'
		) from exc
	payload = validated.payload
	root = Path(validated.root).resolve(strict=True)
	sequences = []
	for sequence in payload['sequences']:
		frames = tuple(
			_adapt_validated_frame(root, frame)
			for frame in sequence['frames']
		)
		asset = sequence['transitions_asset']
		sequences.append(SequenceRecord(
			sequence_id=sequence['sequence_id'],
			split=sequence['split'],
			episode=int(sequence['episode']),
			source=sequence['source'],
			env_seed=int(sequence['env_seed']),
			background_seed=int(sequence['background_seed']),
			action_seed=int(sequence['action_seed']),
			num_transitions=int(sequence['num_transitions']),
			frames=frames,
			transitions_path=_collector_asset_path(root, asset['path'], suffix='.npz'),
			transitions_relative_path=asset['path'],
			transitions_sha256=asset['file_sha256'],
			source_selection_reset_attempt=int(
				sequence['source_selection_reset_attempt']
			),
			ended_by_environment=bool(sequence['ended_by_environment']),
		))
	return RolloutManifest(
		path=Path(validated.path).resolve(strict=True),
		root=root,
		file_sha256=validated.file_sha256,
		collection=dict(payload['collection']),
		sequences=tuple(sequences),
	)


def validate_feature_arrays(
	arrays: Mapping[str, np.ndarray],
	*,
	num_frames: int | None = None,
) -> dict[str, Any]:
	"""Validate one per-episode feature NPZ payload and return JSON-safe metadata."""

	if not isinstance(arrays, Mapping):
		raise ContractError('Feature arrays must be supplied as a mapping.')
	_require_exact_keys(arrays, _FEATURE_ARRAY_KEYS, 'feature arrays')
	features = np.asarray(arrays['features'])
	if features.ndim != 3:
		raise ContractError(f'features must be rank 3, got {features.shape}.')
	frames = int(features.shape[0])
	if frames < 1:
		raise ContractError('Feature arrays must contain at least one frame.')
	if num_frames is not None and frames != num_frames:
		raise ContractError(f'features has {frames} frames, expected {num_frames}.')
	expected_shapes = {
		'features': (frames, len(ROLES), FEATURE_DIM),
		'masks': (frames, len(ROLES), INPUT_SIZE[0], INPUT_SIZE[1]),
		'centroid_xy': (frames, len(ROLES), 2),
		'confidence': (frames, len(ROLES)),
		'lost': (frames, len(ROLES)),
		'valid': (frames, len(ROLES)),
		'mask_score': (frames, len(ROLES)),
		'runtime_ms': (frames,),
	}
	expected_dtypes = {
		'features': np.dtype(np.float32),
		'masks': np.dtype(np.uint8),
		'centroid_xy': np.dtype(np.float32),
		'confidence': np.dtype(np.float32),
		'lost': np.dtype(np.bool_),
		'valid': np.dtype(np.bool_),
		'mask_score': np.dtype(np.float32),
		'runtime_ms': np.dtype(np.float32),
	}
	metadata = {}
	for name in sorted(_FEATURE_ARRAY_KEYS):
		array = np.asarray(arrays[name])
		if tuple(array.shape) != expected_shapes[name]:
			raise ContractError(
				f'{name} shape must be {expected_shapes[name]}, got {array.shape}.'
			)
		if array.dtype != expected_dtypes[name]:
			raise ContractError(
				f'{name} dtype must be {expected_dtypes[name]}, got {array.dtype}.'
			)
		metadata[name] = {'shape': list(array.shape), 'dtype': array.dtype.name}
	if not np.isfinite(features).all():
		raise ContractError('features contains NaN or infinity.')
	masks = np.asarray(arrays['masks'])
	if np.any((masks != 0) & (masks != 1)):
		raise ContractError('masks must use uint8 values 0/1 only.')
	confidence = np.asarray(arrays['confidence'])
	mask_score = np.asarray(arrays['mask_score'])
	runtime_ms = np.asarray(arrays['runtime_ms'])
	for name, array in (
		('confidence', confidence),
		('mask_score', mask_score),
		('runtime_ms', runtime_ms),
	):
		if not np.isfinite(array).all():
			raise ContractError(f'{name} contains NaN or infinity.')
	if np.any((confidence < 0) | (confidence > 1)):
		raise ContractError('confidence must be in [0,1].')
	if np.any((mask_score < 0) | (mask_score > 1)):
		raise ContractError('mask_score must be in [0,1].')
	if np.any(runtime_ms < 0):
		raise ContractError('runtime_ms must be non-negative.')
	lost = np.asarray(arrays['lost'])
	valid = np.asarray(arrays['valid'])
	if np.any(features[lost] != np.float32(0.0)):
		raise ContractError(
			'Every official-lost role feature must be elementwise exactly zero.'
		)
	mask_nonempty = masks.reshape(frames, len(ROLES), -1).any(axis=-1)
	feature_finite = np.isfinite(features).all(axis=-1)
	expected_valid = (~lost) & mask_nonempty & feature_finite
	if not np.array_equal(valid, expected_valid):
		raise ContractError(
			'valid must equal (~official_lost) & mask_nonempty & feature_finite.'
		)
	centroid_xy = np.asarray(arrays['centroid_xy'])
	if np.isinf(centroid_xy).any():
		raise ContractError('centroid_xy may contain NaN for invalid roles, never infinity.')
	if not np.isfinite(centroid_xy[valid]).all():
		raise ContractError('Every valid role must have a finite centroid_xy.')
	metadata['coverage'] = {
		'valid_count': [int(valid[:, index].sum()) for index in range(len(ROLES))],
		'valid_rate': [float(valid[:, index].mean()) for index in range(len(ROLES))],
		'official_lost_count': [
			int(lost[:, index].sum()) for index in range(len(ROLES))
		],
		'mask_empty_count': [
			int((~mask_nonempty[:, index]).sum()) for index in range(len(ROLES))
		],
	}
	return metadata


def _to_numpy(value: Any, *, dtype: np.dtype) -> np.ndarray:
	if hasattr(value, 'detach'):
		value = value.detach().cpu().numpy()
	return np.asarray(value, dtype=dtype)


def _extract_sequence(adapter: Any, sequence: SequenceRecord) -> dict[str, np.ndarray]:
	frames = len(sequence.frames)
	arrays = {
		'features': np.empty((frames, len(ROLES), FEATURE_DIM), dtype=np.float32),
		'masks': np.empty(
			(frames, len(ROLES), INPUT_SIZE[0], INPUT_SIZE[1]), dtype=np.uint8
		),
		'centroid_xy': np.empty((frames, len(ROLES), 2), dtype=np.float32),
		'confidence': np.empty((frames, len(ROLES)), dtype=np.float32),
		'lost': np.empty((frames, len(ROLES)), dtype=np.bool_),
		'valid': np.empty((frames, len(ROLES)), dtype=np.bool_),
		'mask_score': np.empty((frames, len(ROLES)), dtype=np.float32),
		'runtime_ms': np.empty((frames,), dtype=np.float32),
	}
	adapter.reset_episode()
	for t, frame_record in enumerate(sequence.frames):
		# Re-decode and re-hash immediately before tracking to close the gap between
		# manifest validation and model consumption.
		frame = _load_verified_rgb(frame_record.image_path, frame_record.image_sha256)
		try:
			result = adapter.track(frame)
		except Exception as exc:
			raise ContractError(
				f'Cutie tracking failed for {sequence.sequence_id!r} at t={t}: '
				f'{type(exc).__name__}: {exc}'
			) from exc
		if tuple(result.role_names) != ROLES:
			raise ContractError(f'Cutie result role order must be exactly {ROLES!r}.')
		if tuple(result.input_size) != INPUT_SIZE:
			raise ContractError(f'Cutie consumed non-native input {result.input_size}.')
		if tuple(result.tracker_size) != TRACKER_SIZE:
			raise ContractError(
				f'Cutie tracker size {result.tracker_size} differs from {TRACKER_SIZE}.'
			)
		if result.object_features is None:
			raise ContractError('Cutie did not return object features.')
		features = _to_numpy(result.object_features, dtype=np.float32)
		masks = _to_numpy(result.masks, dtype=np.bool_)
		centroid_xy = _to_numpy(result.centroid_xy, dtype=np.float32)
		confidence = _to_numpy(result.confidence, dtype=np.float32)
		lost = _to_numpy(result.lost, dtype=np.bool_)
		mask_score = _to_numpy(result.mask_score, dtype=np.float32)
		if features.shape != (len(ROLES), FEATURE_DIM):
			raise ContractError(f'Cutie feature shape is {features.shape}, expected (2,2048).')
		if masks.shape != (len(ROLES), INPUT_SIZE[0], INPUT_SIZE[1]):
			raise ContractError(f'Cutie mask shape is invalid: {masks.shape}.')
		if centroid_xy.shape != (len(ROLES), 2):
			raise ContractError(f'Cutie centroid shape is invalid: {centroid_xy.shape}.')
		for name, value in (
			('confidence', confidence),
			('lost', lost),
			('mask_score', mask_score),
		):
			if value.shape != (len(ROLES),):
				raise ContractError(f'Cutie {name} shape is invalid: {value.shape}.')
		if not np.isfinite(features).all():
			raise ContractError(
				f'Cutie returned a non-finite feature for {sequence.sequence_id!r} t={t}.'
			)
		mask_nonempty = masks.reshape(len(ROLES), -1).any(axis=-1)
		feature_finite = np.isfinite(features).all(axis=-1)
		valid = (~lost) & mask_nonempty & feature_finite
		arrays['features'][t] = features
		arrays['masks'][t] = masks.astype(np.uint8)
		arrays['centroid_xy'][t] = centroid_xy
		arrays['confidence'][t] = confidence
		arrays['lost'][t] = lost
		arrays['valid'][t] = valid
		arrays['mask_score'][t] = mask_score
		runtime_ms = float(result.runtime_ms)
		if not np.isfinite(runtime_ms) or runtime_ms < 0:
			raise ContractError(f'Cutie returned invalid runtime_ms={runtime_ms!r}.')
		arrays['runtime_ms'][t] = runtime_ms
	validate_feature_arrays(arrays, num_frames=frames)
	return arrays


def _hash_tree(root: Path, suffixes: tuple[str, ...]) -> dict[str, Any]:
	root = root.expanduser().resolve(strict=True)
	if not root.is_dir():
		raise ContractError(f'Hash-tree root is not a directory: {root}.')
	entries = []
	for candidate in sorted(root.rglob('*'), key=lambda value: value.as_posix()):
		if not candidate.is_file() or candidate.suffix.lower() not in suffixes:
			continue
		resolved = candidate.resolve(strict=True)
		try:
			resolved.relative_to(root)
		except ValueError as exc:
			raise ContractError(f'Hash-tree file escapes {root}: {candidate}.') from exc
		relative = candidate.relative_to(root).as_posix()
		entries.append((relative, _sha256_file(resolved)))
	if not entries:
		raise ContractError(f'No {suffixes!r} files found below {root}.')
	digest = hashlib.sha256()
	for relative, file_sha256 in entries:
		digest.update(relative.encode('utf-8'))
		digest.update(b'\0')
		digest.update(file_sha256.encode('ascii'))
		digest.update(b'\n')
	return {
		'path': str(root),
		'tree_sha256': digest.hexdigest(),
		'file_count': len(entries),
		'files': [
			{'path': relative, 'file_sha256': file_sha256}
			for relative, file_sha256 in entries
		],
	}


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
	path.parent.mkdir(parents=True, exist_ok=True)
	if path.exists():
		raise FileExistsError(f'Refusing to overwrite feature asset: {path}.')
	file_descriptor, temporary_name = tempfile.mkstemp(
		prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent
	)
	temporary_path = Path(temporary_name)
	try:
		with os.fdopen(file_descriptor, 'wb') as handle:
			np.savez_compressed(handle, **arrays)
			handle.flush()
			os.fsync(handle.fileno())
		with np.load(temporary_path, allow_pickle=False) as persisted:
			persisted_arrays = {name: persisted[name] for name in persisted.files}
			validate_feature_arrays(
				persisted_arrays, num_frames=int(arrays['features'].shape[0])
			)
		os.replace(temporary_path, path)
	except Exception:
		try:
			temporary_path.unlink(missing_ok=True)
		except OSError:
			pass
		raise
	return _sha256_file(path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> str:
	path.parent.mkdir(parents=True, exist_ok=True)
	if path.exists():
		raise FileExistsError(f'Refusing to overwrite manifest: {path}.')
	encoded = (json.dumps(
		payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False
	) + '\n').encode('utf-8')
	file_descriptor, temporary_name = tempfile.mkstemp(
		prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent
	)
	temporary_path = Path(temporary_name)
	try:
		with os.fdopen(file_descriptor, 'wb') as handle:
			handle.write(encoded)
			handle.flush()
			os.fsync(handle.fileno())
		os.replace(temporary_path, path)
	except Exception:
		try:
			temporary_path.unlink(missing_ok=True)
		except OSError:
			pass
		raise
	return hashlib.sha256(encoded).hexdigest()


def _safe_support_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
	allowed = (
		'task',
		'split',
		'manifest_sha256',
		'combined_manifest_sha256',
		'prompt_count',
		'radius_px',
		'source_resolution',
		'object_roles',
		'rasterization',
		'object_schema',
	)
	return {name: metadata.get(name) for name in allowed}


def extract_features(args: argparse.Namespace) -> tuple[Path, str]:
	"""Run live extraction.  Torch/Cutie imports occur only inside this function."""

	rollout = load_rollout_manifest(args.rollout_manifest)
	if not np.isfinite(args.prompt_radius) or args.prompt_radius <= 0:
		raise ContractError('--prompt-radius must be finite and positive.')
	output = Path(args.output).expanduser().resolve(strict=False)
	output.parent.mkdir(parents=True, exist_ok=True)
	if output.exists():
		raise FileExistsError(f'Refusing to overwrite existing output: {output}.')
	support_path = Path(args.support_annotations).expanduser().resolve(strict=True)
	checkpoint_path = Path(args.checkpoint).expanduser().resolve(strict=True)
	oc_storm_repo = Path(args.oc_storm_repo).expanduser().resolve(strict=True)
	if not support_path.is_file():
		raise ContractError(f'Support annotations are not a file: {support_path}.')
	if not checkpoint_path.is_file():
		raise ContractError(f'Cutie checkpoint is not a file: {checkpoint_path}.')
	if not oc_storm_repo.is_dir():
		raise ContractError(f'OC-STORM repository is not a directory: {oc_storm_repo}.')
	support_annotations_sha256 = _sha256_file(support_path)

	if str(REPO_ROOT) not in sys.path:
		sys.path.insert(0, str(REPO_ROOT))
	try:
		from tdmpc2.perception.cutie_oc_adapter import (
			CutieOCAdapter,
			CutieOCConfig,
			inspect_cutie_installation,
			load_point_support_prompts,
		)
	except Exception as exc:
		raise ContractError(
			f'Unable to import the standalone Cutie adapter: '
			f'{type(exc).__name__}: {exc}'
		) from exc

	config = CutieOCConfig(
		repo_path=oc_storm_repo,
		checkpoint_path=checkpoint_path,
		role_names=ROLES,
		model_size=args.model_size,
		device=args.device,
		output_device='cpu',
		config_dir=args.config_dir,
		expected_input_size=INPUT_SIZE,
		tracker_size=TRACKER_SIZE,
		foreground_queries=FOREGROUND_QUERIES,
		amp=not args.no_amp,
		return_object_features=True,
		object_schema=OBJECT_SCHEMA,
	)
	config = config.validated()
	config_dir = config.hydra_config_dir
	checkpoint_sha256 = _sha256_file(checkpoint_path)
	config_provenance = _hash_tree(config_dir, ('.yaml', '.yml'))
	cutie_code_root = oc_storm_repo / 'feature_extractor' / 'cutie'
	cutie_code_provenance = _hash_tree(cutie_code_root, ('.py', '.yaml', '.yml'))
	adapter_path = PROJECT_ROOT / 'perception' / 'cutie_oc_adapter.py'
	extractor_path = Path(__file__).resolve()
	adapter_sha256 = _sha256_file(adapter_path)
	extractor_sha256 = _sha256_file(extractor_path)
	try:
		preflight = inspect_cutie_installation(config, import_check=True)
		support = load_point_support_prompts(
			support_path,
			radius_px=args.prompt_radius,
			object_schema=OBJECT_SCHEMA,
		)
	except Exception as exc:
		raise ContractError(
			f'Cutie preflight/support validation failed: {type(exc).__name__}: {exc}'
		) from exc
	if _sha256_file(support_path) != support_annotations_sha256:
		raise ContractError('Support annotations changed while they were being loaded.')
	if tuple(support.role_names) != ROLES:
		raise ContractError(f'Support object roles must be exactly {ROLES!r}.')
	support_metadata = _safe_support_metadata(support.metadata)
	if support_metadata['split'] != 'support':
		raise ContractError('Permanent prompts must come only from the support split.')
	if support_metadata['task'] != TASK:
		raise ContractError(f'Support task must be exactly {TASK!r}.')
	if tuple(support_metadata['object_roles'] or ()) != ROLES:
		raise ContractError(f'Support metadata roles must be exactly {ROLES!r}.')
	if support_metadata['object_schema'] != OBJECT_SCHEMA:
		raise ContractError(f'Support object schema must be {OBJECT_SCHEMA!r}.')
	if support_metadata['source_resolution'] != list(INPUT_SIZE):
		raise ContractError('Support prompts must be defined on native 64x64 RGB.')
	_require_sha256(
		support_metadata['manifest_sha256'], 'support.metadata.manifest_sha256'
	)
	_require_sha256(
		support_metadata['combined_manifest_sha256'],
		'support.metadata.combined_manifest_sha256',
	)
	support_manifest_path = (
		PROJECT_ROOT / 'envs' / 'background_manifests' / 'color_multi_support.json'
	).resolve(strict=True)
	if _sha256_file(support_manifest_path) != support_metadata['manifest_sha256']:
		raise ContractError('Support annotations bind the wrong support-manifest SHA.')
	if (
		support_metadata['combined_manifest_sha256']
		!= rollout.collection['combined_manifest_sha256']
	):
		raise ContractError('Support annotations bind the wrong combined-manifest SHA.')
	try:
		adapter = CutieOCAdapter(config)
		adapter.add_support_prompts(support)
	except Exception as exc:
		raise ContractError(
			f'Cutie model/support initialization failed: {type(exc).__name__}: {exc}'
		) from exc
	if _sha256_file(checkpoint_path) != checkpoint_sha256:
		raise ContractError('Cutie checkpoint changed while the model was initialized.')

	lock_path = output.parent / f'.{output.name}.lock'
	try:
		lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
	except FileExistsError as exc:
		raise FileExistsError(f'Another extraction owns or left lock {lock_path}.') from exc
	staging: Path | None = None
	try:
		with os.fdopen(lock_descriptor, 'w', encoding='ascii') as lock_handle:
			lock_handle.write(f'pid={os.getpid()}\n')
			lock_handle.flush()
			os.fsync(lock_handle.fileno())
		staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}.tmp-', dir=output.parent))
		sequence_payloads = []
		for sequence in rollout.sequences:
			arrays = _extract_sequence(adapter, sequence)
			array_metadata = validate_feature_arrays(
				arrays, num_frames=sequence.num_transitions + 1
			)
			asset_relative_path = (
				Path('episodes') / f'{sequence.episode:06d}_{sequence.sequence_id}.npz'
			)
			asset_path = staging / asset_relative_path
			asset_sha256 = _atomic_write_npz(asset_path, arrays)
			runtime = arrays['runtime_ms'].astype(np.float64)
			sequence_payloads.append({
				'sequence_id': sequence.sequence_id,
				'split': sequence.split,
				'episode': sequence.episode,
				'source': sequence.source,
				'env_seed': sequence.env_seed,
				'background_seed': sequence.background_seed,
				'action_seed': sequence.action_seed,
				'source_selection_reset_attempt': (
					sequence.source_selection_reset_attempt
				),
				'ended_by_environment': sequence.ended_by_environment,
				'num_transitions': sequence.num_transitions,
				'frames': [
					{
						't': frame.t,
						'image_sha256': frame.image_sha256,
						'source_frame_index': frame.source_frame_index,
					}
					for frame in sequence.frames
				],
				'rollout_transitions_asset': {
					'path': sequence.transitions_relative_path,
					'file_sha256': sequence.transitions_sha256,
				},
				'npz_asset': {
					'path': asset_relative_path.as_posix(),
					'file_sha256': asset_sha256,
					'arrays': {
						name: value for name, value in array_metadata.items()
						if name != 'coverage'
					},
				},
				'coverage': array_metadata['coverage'],
				'runtime_ms': {
					'total': float(runtime.sum()),
					'mean': float(runtime.mean()),
					'p95': float(np.percentile(runtime, 95)),
				},
			})

		stable_perception_provenance = {
			'algorithm': 'visual_small_cutie_object_features_v1',
			'object_schema': OBJECT_SCHEMA,
			'roles': list(ROLES),
			'native_input_size': list(INPUT_SIZE),
			'tracker_size': list(TRACKER_SIZE),
			'foreground_queries': FOREGROUND_QUERIES,
			'feature_dim': FEATURE_DIM,
			'query_order': 'official_query_post_process_cache_first_8_flattened',
			'model_size': args.model_size,
			'amp': not args.no_amp,
			'prompt_radius': float(args.prompt_radius),
			'checkpoint_file_sha256': checkpoint_sha256,
			'support_annotations_file_sha256': support_annotations_sha256,
			'support_manifest_sha256': support_metadata['manifest_sha256'],
			'combined_manifest_sha256': support_metadata['combined_manifest_sha256'],
			'config_tree_sha256': config_provenance['tree_sha256'],
			'cutie_code_tree_sha256': cutie_code_provenance['tree_sha256'],
			'adapter_file_sha256': adapter_sha256,
			'extractor_file_sha256': extractor_sha256,
		}
		manifest_payload = {
			'format': FEATURE_FORMAT,
			'rollout_manifest_sha256': rollout.file_sha256,
			'perception_provenance': stable_perception_provenance,
			'roles': list(ROLES),
			'object_schema': OBJECT_SCHEMA,
			'collection': {
				'task': TASK,
				'split': rollout.collection['split'],
				'num_sequences': len(sequence_payloads),
				'num_frames': sum(
					sequence.num_transitions + 1 for sequence in rollout.sequences
				),
				'native_input_size': list(INPUT_SIZE),
				'tracker_size': list(TRACKER_SIZE),
				'no_test': True,
				'no_support_trajectories': True,
				'causal': True,
				'episode_memory_reset': True,
				'permanent_support_loaded_once': True,
				'transitions_passed_to_cutie': False,
				'rollout_labels_passed_to_cutie': False,
			},
			'feature_contract': {
				'feature_dim': FEATURE_DIM,
				'foreground_queries': FOREGROUND_QUERIES,
				'query_dim': FEATURE_DIM // FOREGROUND_QUERIES,
				'query_order': 'official_query_post_process_cache_first_8_flattened',
				'valid_definition': (
					'(~official_lost) & mask_nonempty & feature_finite'
				),
				'missing_policy': (
					'No future fill and no success-only filtering; consumers must mask valid=false.'
				),
				'mask_values': [0, 1],
				'centroid_convention': '[x,y] in native 64x64 RGB; NaN is allowed when invalid',
			},
			'provenance': {
				'rollout': {
					'manifest_path': str(rollout.path),
					'manifest_file_sha256': rollout.file_sha256,
					'format': ROLLOUT_FORMAT,
					'collection': dict(rollout.collection),
				},
				'background_manifests': {
					'rollout_split': rollout.collection['split'],
					'support_manifest_path': str(support_manifest_path),
					'rollout_split_manifest_sha256': (
						rollout.collection['manifest_sha256']
					),
					'support_manifest_sha256': support_metadata['manifest_sha256'],
					'combined_manifest_sha256': (
						rollout.collection['combined_manifest_sha256']
					),
				},
				'support': {
					'annotations_path': str(support_path),
					'annotations_file_sha256': support_annotations_sha256,
					'metadata': support_metadata,
				},
				'cutie': {
					'backend': 'official_oc_storm_cutie',
					'oc_storm_repo': str(oc_storm_repo),
					'model_size': args.model_size,
					'device': args.device,
					'amp': not args.no_amp,
					'checkpoint_path': str(checkpoint_path),
					'checkpoint_file_sha256': checkpoint_sha256,
					'config': config_provenance,
					'code': cutie_code_provenance,
					'preflight': dict(preflight),
				},
				'in_tree': {
					'adapter_path': str(adapter_path),
					'adapter_file_sha256': adapter_sha256,
					'extractor_path': str(extractor_path),
					'extractor_file_sha256': extractor_sha256,
				},
			},
			'runtime': adapter.runtime_summary(),
			'sequences': sequence_payloads,
		}
		manifest_sha256 = _atomic_write_json(
			staging / 'manifest.json', manifest_payload
		)
		if output.exists():
			raise FileExistsError(f'Refusing to replace output created concurrently: {output}.')
		staging.rename(output)
		staging = None
		return output, manifest_sha256
	finally:
		if staging is not None:
			shutil.rmtree(staging, ignore_errors=True)
		try:
			lock_path.unlink(missing_ok=True)
		except OSError:
			pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			'Extract causal whole_arm/goal Cutie object features from one frozen '
			'Visual-Small train or validation rollout manifest.'
		)
	)
	parser.add_argument('--rollout-manifest', type=Path, required=True)
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--oc-storm-repo', type=Path, required=True)
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--support-annotations', type=Path, required=True)
	parser.add_argument('--config-dir', type=Path)
	parser.add_argument('--model-size', choices=('small', 'base'), default='small')
	parser.add_argument('--device', default='cuda:0')
	parser.add_argument('--prompt-radius', type=float, default=2.0)
	parser.add_argument('--no-amp', action='store_true')
	parser.add_argument(
		'--traceback', action='store_true', help='Print the chained traceback on failure.'
	)
	return parser.parse_args(argv)


def main(argv: list[str] | None = None):
	args = parse_args(argv)
	try:
		output, manifest_sha256 = extract_features(args)
	except (ContractError, FileExistsError, FileNotFoundError, ValueError) as exc:
		if args.traceback:
			traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
		raise SystemExit(f'CUTIE_OBJECT_FEATURE_EXTRACTION_FAILED: {exc}') from exc
	print(
		'CUTIE_OBJECT_FEATURE_EXTRACTION_OK '
		+ json.dumps({
			'output': str(output),
			'manifest': str(output / 'manifest.json'),
			'manifest_file_sha256': manifest_sha256,
		}, sort_keys=True)
	)


if __name__ == '__main__':
	main()
