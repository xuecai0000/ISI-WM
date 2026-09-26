"""Strict fixed-support SAM 2.1 video-segmentation backend.

This module is intentionally independent from the TD-MPC2 environment and
training process.  It implements the perception-only protocol used to compare
video object segmentation backends:

* a fixed, task-level support pack supplies RGB plus indexed masks;
* every support frame is replayed in order as a SAM 2.1 mask prompt;
* episode frames never receive a mask, point, box, GT value, or correction;
* episode propagation is forward-only and guarded against future-frame reads;
* outputs contain masks and diagnostics, never model-specific embeddings.

Official SAM 2.1 does not expose Cutie-style permanent support memory.  The
only faithful use of several fixed support images is therefore a documented
``fixed_support_prefix_v1`` protocol: support frames occupy indices ``0..S-1``
in a fresh predictor state and episode frames occupy ``S..S+T-1``.  All
support masks are installed before forward propagation starts at index ``S``.

The external SAM 2 checkout and checkpoint are immutable caller-provided
resources.  Nothing is downloaded and there is no fallback backend.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as np

from tdmpc2.common.unified_vos import (
	BACKEND_FORMAT,
	file_sha256,
	validate_backend_inputs,
	write_json,
)


class SAM21BackendError(RuntimeError):
	"""Base error for the strict SAM 2.1 backend."""


class SAM21PreflightError(SAM21BackendError):
	"""Raised when configured resources or runtime contracts are invalid."""


class SAM21DependencyError(SAM21PreflightError):
	"""Raised when the official SAM 2.1 implementation cannot be imported."""


class SAM21ProtocolError(SAM21BackendError):
	"""Raised when support, episode, or predictor output violates the protocol."""


class SAM21CausalityError(SAM21ProtocolError):
	"""Raised when the predictor attempts to read a future episode frame."""


_MODEL_CONFIGS = {
	'tiny': 'configs/sam2.1/sam2.1_hiera_t.yaml',
	'small': 'configs/sam2.1/sam2.1_hiera_s.yaml',
	'base_plus': 'configs/sam2.1/sam2.1_hiera_b+.yaml',
	'large': 'configs/sam2.1/sam2.1_hiera_l.yaml',
}
_OUTPUT_FORMAT = 'sam21_fixed_support_video_segmentation_v1'
_PROMPT_PROTOCOL = 'fixed_support_prefix_v1'


@dataclass(frozen=True)
class SAM21VideoConfig:
	"""Official SAM 2.1 resources and task-independent role contract."""

	repo_path: str | Path
	checkpoint_path: str | Path
	role_names: tuple[str, ...]
	model_size: str = 'large'
	model_config: str | None = None
	device: str = 'cuda:0'
	expected_input_size: tuple[int, int] = (64, 64)
	expected_support_frames: int = 6
	mask_threshold: float = 0.0
	min_mask_pixels: int = 1
	amp_dtype: str = 'bfloat16'
	offload_video_to_cpu: bool = True
	offload_state_to_cpu: bool = False
	vos_optimized: bool = False
	exclusive_masks: bool = True

	def validated(self) -> 'SAM21VideoConfig':
		if self.model_size not in _MODEL_CONFIGS:
			raise ValueError(
				f'model_size must be one of {tuple(_MODEL_CONFIGS)!r}, '
				f'got {self.model_size!r}.'
			)
		expected_config = _MODEL_CONFIGS[self.model_size]
		if self.resolved_model_config != expected_config:
			raise ValueError(
				'Strict SAM 2.1 model/config pairing requires '
				f'{expected_config!r} for model_size={self.model_size!r}; got '
				f'{self.resolved_model_config!r}.'
			)
		roles = tuple(str(role) for role in self.role_names)
		if not roles or len(set(roles)) != len(roles):
			raise ValueError('role_names must be non-empty and unique.')
		if any(not role.strip() for role in roles):
			raise ValueError('role_names cannot contain empty strings.')
		if len(self.expected_input_size) != 2 or min(self.expected_input_size) < 1:
			raise ValueError('expected_input_size must be a positive (height, width).')
		if self.expected_support_frames < 1:
			raise ValueError('expected_support_frames must be positive.')
		if not np.isfinite(self.mask_threshold):
			raise ValueError('mask_threshold must be finite.')
		if self.min_mask_pixels < 1:
			raise ValueError('min_mask_pixels must be positive.')
		if self.amp_dtype not in {'none', 'float16', 'bfloat16'}:
			raise ValueError(
				"amp_dtype must be one of 'none', 'float16', or 'bfloat16'."
			)
		return self

	@property
	def repo(self) -> Path:
		return Path(self.repo_path).expanduser().resolve()

	@property
	def checkpoint(self) -> Path:
		return Path(self.checkpoint_path).expanduser().resolve()

	@property
	def resolved_model_config(self) -> str:
		return (self.model_config or _MODEL_CONFIGS[self.model_size]).replace('\\', '/')

	@property
	def config_file(self) -> Path:
		# Hydra package configs live below <checkout>/sam2/configs in the official repo.
		return self.repo / 'sam2' / Path(self.resolved_model_config)


@dataclass(frozen=True)
class SAM21SupportPack:
	"""Immutable fixed support RGB and indexed role masks."""

	rgb: np.ndarray
	indexed_masks: np.ndarray
	source_path: Path | None = None


@dataclass(frozen=True)
class SAM21EpisodeResult:
	"""Role-ordered SAM 2.1 predictions for episode frames only."""

	role_names: tuple[str, ...]
	predicted_indexed: np.ndarray
	confidence: np.ndarray
	mask_score: np.ndarray
	lost: np.ndarray
	valid: np.ndarray
	area_pixels: np.ndarray
	centroid_xy: np.ndarray
	frame_runtime_ms: np.ndarray
	raw_overlap_pixels: np.ndarray
	diagnostics: dict[str, Any]

	def save(self, output_npz: str | Path, output_json: str | Path) -> None:
		"""Atomically save the unified arrays and human-readable diagnostics."""
		npz_path = Path(output_npz).expanduser().resolve()
		json_path = Path(output_json).expanduser().resolve()
		for path in (npz_path, json_path):
			if path.exists():
				raise FileExistsError(f'Refusing to overwrite existing output: {path}')
			if not path.parent.is_dir():
				raise FileNotFoundError(f'Output parent does not exist: {path.parent}')
		npz_tmp = npz_path.with_name(npz_path.name + '.incomplete')
		json_tmp = json_path.with_name(json_path.name + '.incomplete')
		try:
			predicted_masks = np.stack([
				self.predicted_indexed == role_id
				for role_id in range(1, len(self.role_names) + 1)
			], axis=1)
			with npz_tmp.open('wb') as handle:
				np.savez_compressed(
					handle,
					predicted_masks=predicted_masks.astype(np.bool_, copy=False),
					reported_confidence=self.confidence.astype(
						np.float32, copy=False
					),
					reported_lost=self.lost.astype(np.bool_, copy=False),
					runtime_ms=self.frame_runtime_ms.astype(np.float64, copy=False),
				)
			json_tmp.write_text(
				json.dumps(self.diagnostics, indent=2, sort_keys=True) + '\n',
				encoding='utf-8',
			)
			os.replace(npz_tmp, npz_path)
			os.replace(json_tmp, json_path)
		finally:
			for path in (npz_tmp, json_tmp):
				if path.exists():
					path.unlink()


def _typed_array_sha256(value: np.ndarray) -> str:
	array = np.ascontiguousarray(value)
	digest = hashlib.sha256()
	digest.update(str(array.dtype).encode('ascii'))
	digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
	digest.update(array.tobytes())
	return digest.hexdigest()


def _file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as handle:
		for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
			digest.update(chunk)
	return digest.hexdigest()


def _tree_snapshot(root: Path) -> dict[str, Any]:
	"""Hash the official Python/config implementation without reading weights."""
	root = root.expanduser().resolve()
	if not root.is_dir():
		raise FileNotFoundError(f'SAM 2.1 implementation tree not found: {root}')
	paths = sorted(
		path for path in root.rglob('*')
		if path.is_file() and path.suffix.lower() in {'.py', '.yaml', '.yml'}
	)
	if not paths:
		raise SAM21PreflightError(
			f'SAM 2.1 implementation/config tree is empty: {root}'
		)
	files = [{
		'path': path.relative_to(root).as_posix(),
		'bytes': path.stat().st_size,
		'sha256': _file_sha256(path),
	} for path in paths]
	digest = hashlib.sha256(
		(json.dumps(files, sort_keys=True, separators=(',', ':')) + '\n').encode(
			'utf-8'
		)
	).hexdigest()
	return {'root': str(root), 'files': files, 'tree_sha256': digest}


def _validated_rgb(
	value: np.ndarray,
	*,
	name: str,
	expected_size: tuple[int, int],
) -> np.ndarray:
	array = np.asarray(value)
	if array.ndim != 4 or array.shape[-1] != 3:
		raise SAM21ProtocolError(
			f'{name} must be uint8 [T,H,W,3], got shape {array.shape}.'
		)
	if array.dtype != np.uint8:
		raise SAM21ProtocolError(f'{name} must be uint8, got {array.dtype}.')
	if tuple(array.shape[1:3]) != tuple(expected_size):
		raise SAM21ProtocolError(
			f'{name} must use resolution {expected_size}, got {array.shape[1:3]}.'
		)
	if array.shape[0] < 1:
		raise SAM21ProtocolError(f'{name} must contain at least one frame.')
	return np.array(array, dtype=np.uint8, order='C', copy=True)


def _validated_support(
	support: SAM21SupportPack,
	config: SAM21VideoConfig,
) -> SAM21SupportPack:
	rgb = _validated_rgb(
		support.rgb,
		name='support.rgb',
		expected_size=config.expected_input_size,
	)
	masks = np.asarray(support.indexed_masks)
	expected_shape = (rgb.shape[0], *config.expected_input_size)
	if masks.dtype != np.uint8 or masks.shape != expected_shape:
		raise SAM21ProtocolError(
			'support.indexed_masks must be uint8 '
			f'{expected_shape}, got {masks.shape} {masks.dtype}.'
		)
	if rgb.shape[0] != config.expected_support_frames:
		raise SAM21ProtocolError(
			f'Expected exactly {config.expected_support_frames} fixed support frames, '
			f'got {rgb.shape[0]}.'
		)
	num_roles = len(config.role_names)
	unique = set(np.unique(masks).tolist())
	if not unique.issubset(set(range(num_roles + 1))):
		raise SAM21ProtocolError(
			f'Support mask values must be role IDs 0..{num_roles}, got {sorted(unique)}.'
		)
	for frame_index, indexed in enumerate(masks):
		missing = [
			role
			for role_id, role in enumerate(config.role_names, start=1)
			if not np.any(indexed == role_id)
		]
		if missing:
			raise SAM21ProtocolError(
				f'Support frame {frame_index} has no prompt pixels for roles {missing!r}.'
			)
	return SAM21SupportPack(
		rgb=rgb,
		indexed_masks=np.array(masks, dtype=np.uint8, order='C', copy=True),
		source_path=support.source_path,
	)


def load_sam21_support_npz(
	path: str | Path,
	config: SAM21VideoConfig,
) -> SAM21SupportPack:
	"""Read only ``rgb`` and ``indexed_masks`` from a fixed support archive."""
	source = Path(path).expanduser().resolve()
	if not source.is_file():
		raise FileNotFoundError(f'SAM 2.1 support archive not found: {source}')
	try:
		with np.load(source, allow_pickle=False) as archive:
			if 'rgb' not in archive.files or 'indexed_masks' not in archive.files:
				raise SAM21ProtocolError(
					'Support NPZ must contain rgb and indexed_masks arrays.'
				)
			# Deliberately access only the two declared support inputs.
			support = SAM21SupportPack(
				rgb=np.asarray(archive['rgb']),
				indexed_masks=np.asarray(archive['indexed_masks']),
				source_path=source,
			)
	except (OSError, ValueError) as exc:
		raise SAM21ProtocolError(
			f'Could not read fixed support archive {source}: {exc}'
		) from exc
	return _validated_support(support, config)


def load_sam21_episode_rgb(
	path: str | Path,
	config: SAM21VideoConfig,
) -> np.ndarray:
	"""Read only the RGB array; GT arrays in the archive remain untouched."""
	source = Path(path).expanduser().resolve()
	if not source.is_file():
		raise FileNotFoundError(f'SAM 2.1 episode archive not found: {source}')
	try:
		with np.load(source, allow_pickle=False) as archive:
			if 'rgb' not in archive.files:
				raise SAM21ProtocolError('Episode NPZ must contain an rgb array.')
			# Never access gt_indexed (or any other evaluation-only array) here.
			rgb = np.asarray(archive['rgb'])
	except (OSError, ValueError) as exc:
		raise SAM21ProtocolError(
			f'Could not read episode RGB from {source}: {exc}'
		) from exc
	return _validated_rgb(
		rgb,
		name='episode.rgb',
		expected_size=config.expected_input_size,
	)


def _path_is_within(path: Path, parent: Path) -> bool:
	try:
		path.resolve().relative_to(parent.resolve())
		return True
	except ValueError:
		return False


def _import_official_builder(repo: Path):
	package = repo / 'sam2'
	if str(repo) not in sys.path:
		sys.path.insert(0, str(repo))
	importlib.invalidate_caches()
	already = sys.modules.get('sam2')
	if already is not None:
		origin = Path(getattr(already, '__file__', '') or '.').resolve()
		if not _path_is_within(origin, package):
			raise SAM21DependencyError(
				'Another sam2 package is already imported from '
				f'{origin}; expected the configured checkout {package}. Run this '
				'backend in its dedicated fresh process.'
			)
	try:
		sam2_module = importlib.import_module('sam2')
		build_module = importlib.import_module('sam2.build_sam')
	except Exception as exc:
		raise SAM21DependencyError(
			'Could not import the official facebookresearch/sam2 checkout from '
			f'{repo}. Install that checkout in this worker environment without '
			f'downloading weights at runtime. Original error: '
			f'{type(exc).__name__}: {exc}'
		) from exc
	origin = Path(getattr(sam2_module, '__file__', '') or '.').resolve()
	if not _path_is_within(origin, package):
		raise SAM21DependencyError(
			f'Imported sam2 from {origin}, outside configured checkout {package}.'
		)
	builder = getattr(build_module, 'build_sam2_video_predictor', None)
	if not callable(builder):
		raise SAM21DependencyError(
			'Official sam2.build_sam does not expose build_sam2_video_predictor.'
		)
	return builder


def _require_torch():
	try:
		return importlib.import_module('torch')
	except Exception as exc:
		raise SAM21DependencyError(
			f'SAM 2.1 requires PyTorch in the worker environment: {exc}'
		) from exc


def inspect_sam21_installation(
	config: SAM21VideoConfig,
	*,
	import_check: bool = True,
) -> dict[str, Any]:
	"""Read-only, no-download resource and dependency preflight."""
	config.validated()
	paths = {
		'repo': config.repo,
		'package': config.repo / 'sam2',
		'build_module': config.repo / 'sam2' / 'build_sam.py',
		'model_config': config.config_file,
		'checkpoint': config.checkpoint,
	}
	missing = [name for name, path in paths.items() if not path.exists()]
	if missing:
		details = ', '.join(f'{name}={paths[name]}' for name in missing)
		raise SAM21PreflightError(f'Missing required SAM 2.1 resources: {details}')
	if not paths['repo'].is_dir() or not paths['package'].is_dir():
		raise SAM21PreflightError(
			f'Configured SAM 2.1 checkout is not a directory: {config.repo}'
		)
	if not paths['checkpoint'].is_file() or paths['checkpoint'].stat().st_size < 1:
		raise SAM21PreflightError(
			f'SAM 2.1 checkpoint is missing or empty: {paths["checkpoint"]}'
		)
	module_status = {
		name: importlib.util.find_spec(name) is not None
		for name in (
			'numpy', 'torch', 'torchvision', 'PIL', 'hydra', 'omegaconf', 'tqdm'
		)
	}
	missing_modules = [name for name, present in module_status.items() if not present]
	if missing_modules:
		raise SAM21DependencyError(
			'Missing SAM 2.1 worker modules: ' + ', '.join(missing_modules)
		)
	if import_check:
		_import_official_builder(config.repo)
	torch = _require_torch()
	device = torch.device(config.device)
	if device.type == 'cuda':
		if not torch.cuda.is_available():
			raise SAM21PreflightError(
				f'{config.device} requested but CUDA is unavailable.'
			)
		index = torch.cuda.current_device() if device.index is None else device.index
		if index >= torch.cuda.device_count():
			raise SAM21PreflightError(
				f'{config.device} is outside {torch.cuda.device_count()} visible device(s).'
			)
		device_name = torch.cuda.get_device_name(index)
	else:
		device_name = str(device)
	return {
		'backend': 'official_sam_2_1_video_predictor',
		'model_size': config.model_size,
		'model_config': config.resolved_model_config,
		'repo_path': str(config.repo),
		'checkpoint_path': str(config.checkpoint),
		'checkpoint_bytes': config.checkpoint.stat().st_size,
		'roles': list(config.role_names),
		'device': str(device),
		'device_name': device_name,
		'module_status': module_status,
		'prompt_protocol': _PROMPT_PROTOCOL,
		'expected_support_frames': config.expected_support_frames,
		'episode_prompt_policy': 'forbidden',
		'propagation_direction': 'forward_only',
		'object_embedding_output': False,
		'downloads': False,
	}


def _build_predictor(config: SAM21VideoConfig):
	builder = _import_official_builder(config.repo)
	parameters = inspect.signature(builder).parameters
	kwargs: dict[str, Any] = {}
	for name, value in (
		('device', config.device),
		('mode', 'eval'),
		('apply_postprocessing', True),
	):
		if name in parameters:
			kwargs[name] = value
	if config.vos_optimized:
		if 'vos_optimized' not in parameters:
			raise SAM21DependencyError(
				'vos_optimized=true requires the current official SAM 2 predictor API.'
			)
		kwargs['vos_optimized'] = True
	try:
		predictor = builder(
			config.resolved_model_config,
			str(config.checkpoint),
			**kwargs,
		)
	except Exception as exc:
		raise SAM21DependencyError(
			'Official SAM 2.1 predictor construction failed for '
			f'config={config.resolved_model_config}, checkpoint={config.checkpoint}. '
			f'Original error: {type(exc).__name__}: {exc}'
		) from exc
	for method in (
		'init_state',
		'add_new_mask',
		'propagate_in_video_preflight',
		'propagate_in_video',
	):
		if not callable(getattr(predictor, method, None)):
			raise SAM21DependencyError(
				f'Official SAM 2.1 predictor lacks required method {method}().'
			)
	return predictor


class _CausalRGBFrameStore:
	"""Lazy per-frame preprocessing with a hard future-access boundary."""

	def __init__(
		self,
		rgb: np.ndarray,
		*,
		image_size: int,
		offload_video_to_cpu: bool,
		img_mean: Iterable[float],
		img_std: Iterable[float],
		compute_device,
	):
		self._rgb = rgb
		self._image_size = int(image_size)
		self._offload = bool(offload_video_to_cpu)
		self._mean_values = tuple(float(v) for v in img_mean)
		self._std_values = tuple(float(v) for v in img_std)
		self._compute_device = compute_device
		self._allowed_max = 0
		self.max_accessed_index = -1
		self.access_count = 0
		self.future_access_attempts = 0

	@property
	def shape(self):
		return (len(self), 3, self._image_size, self._image_size)

	@property
	def allowed_max(self) -> int:
		return self._allowed_max

	def __len__(self) -> int:
		return int(self._rgb.shape[0])

	def allow_through(self, frame_index: int) -> None:
		index = int(frame_index)
		if index < self._allowed_max:
			raise SAM21CausalityError(
				f'Causal frame boundary cannot move backward: '
				f'{self._allowed_max} -> {index}.'
			)
		if index >= len(self):
			raise SAM21CausalityError(
				f'Causal frame boundary {index} is outside {len(self)} frames.'
			)
		self._allowed_max = index

	def __getitem__(self, frame_index: int):
		if not isinstance(frame_index, (int, np.integer)):
			raise SAM21CausalityError(
				'SAM 2.1 attempted a non-scalar frame access; strict causal '
				f'backend only permits one explicit frame at a time: {frame_index!r}.'
			)
		index = int(frame_index)
		if index < 0 or index >= len(self):
			raise IndexError(index)
		if index > self._allowed_max:
			self.future_access_attempts += 1
			raise SAM21CausalityError(
				f'SAM 2.1 attempted to read future frame {index} while the causal '
				f'boundary is {self._allowed_max}.'
			)
		self.access_count += 1
		self.max_accessed_index = max(self.max_accessed_index, index)
		try:
			from PIL import Image
		except Exception as exc:
			raise SAM21DependencyError(
				f'Pillow is required for exact SAM 2 frame resizing: {exc}'
			) from exc
		torch = _require_torch()
		image = Image.fromarray(self._rgb[index], mode='RGB')
		# Match the official loader: PIL resize, RGB uint8 -> float [0,1].
		resized = np.array(
			image.resize((self._image_size, self._image_size)),
			dtype=np.uint8,
			copy=True,
		)
		# Official loader divides in NumPy (float64), then assigns into its
		# preallocated float32 video tensor. Preserve that rounding order.
		normalized = resized / 255.0
		tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()
		device = torch.device('cpu') if self._offload else self._compute_device
		tensor = tensor.to(device)
		mean = torch.tensor(
			self._mean_values, dtype=torch.float32, device=device
		)[:, None, None]
		std = torch.tensor(
			self._std_values, dtype=torch.float32, device=device
		)[:, None, None]
		return tensor.sub_(mean).div_(std)


def _predictor_module(predictor) -> ModuleType:
	module = sys.modules.get(type(predictor).__module__)
	if module is None:
		try:
			module = importlib.import_module(type(predictor).__module__)
		except Exception as exc:
			raise SAM21DependencyError(
				f'Cannot locate predictor module {type(predictor).__module__}: {exc}'
			) from exc
	if not hasattr(module, 'load_video_frames'):
		raise SAM21DependencyError(
			'Predictor module does not expose the official load_video_frames global; '
			'exact in-memory causal input cannot be enforced with this SAM 2 version.'
		)
	return module


def _max_invalid_burst(values: np.ndarray) -> int:
	maximum = current = 0
	for value in values.tolist():
		if bool(value):
			current += 1
			maximum = max(maximum, current)
		else:
			current = 0
	return maximum


class SAM21VideoAdapter:
	"""Episode-isolated adapter around official ``SAM2VideoPredictor``."""

	def __init__(self, config: SAM21VideoConfig, *, _predictor=None):
		self.config = config.validated()
		if _predictor is None:
			inspect_sam21_installation(self.config, import_check=True)
			self.predictor = _build_predictor(self.config)
		else:
			# Used only by dependency-free protocol tests.
			self.predictor = _predictor
		for method in (
			'init_state',
			'add_new_mask',
			'propagate_in_video_preflight',
			'propagate_in_video',
		):
			if not callable(getattr(self.predictor, method, None)):
				raise TypeError(f'Injected predictor lacks required method {method}().')

	def _sync(self) -> None:
		torch = _require_torch()
		device = torch.device(self.config.device)
		if device.type == 'cuda':
			torch.cuda.synchronize(device)

	def _contexts(self):
		torch = _require_torch()
		inference = torch.inference_mode()
		device = torch.device(self.config.device)
		if device.type != 'cuda' or self.config.amp_dtype == 'none':
			autocast = contextlib.nullcontext()
		else:
			dtype = (
				torch.bfloat16
				if self.config.amp_dtype == 'bfloat16'
				else torch.float16
			)
			autocast = torch.autocast('cuda', dtype=dtype)
		return inference, autocast

	def _init_causal_state(self, all_rgb: np.ndarray):
		"""Call official init_state with a guarded exact-array frame loader."""
		module = _predictor_module(self.predictor)
		original_loader = module.load_video_frames
		created: dict[str, _CausalRGBFrameStore] = {}

		def guarded_loader(
			video_path,
			image_size,
			offload_video_to_cpu,
			img_mean=(0.485, 0.456, 0.406),
			img_std=(0.229, 0.224, 0.225),
			async_loading_frames=False,
			compute_device=None,
			**kwargs,
		):
			if async_loading_frames:
				raise SAM21CausalityError(
					'async_loading_frames is forbidden by the strict causal protocol.'
				)
			if created:
				raise SAM21ProtocolError('SAM 2 initialized the video loader more than once.')
			store = _CausalRGBFrameStore(
				all_rgb,
				image_size=int(image_size),
				offload_video_to_cpu=bool(offload_video_to_cpu),
				img_mean=img_mean,
				img_std=img_std,
				compute_device=compute_device,
			)
			created['store'] = store
			height, width = map(int, all_rgb.shape[1:3])
			return store, height, width

		module.load_video_frames = guarded_loader
		try:
			parameters = inspect.signature(self.predictor.init_state).parameters
			kwargs: dict[str, Any] = {}
			if 'offload_video_to_cpu' in parameters:
				kwargs['offload_video_to_cpu'] = self.config.offload_video_to_cpu
			if 'offload_state_to_cpu' in parameters:
				kwargs['offload_state_to_cpu'] = self.config.offload_state_to_cpu
			if 'async_loading_frames' in parameters:
				kwargs['async_loading_frames'] = False
			state = self.predictor.init_state(
				video_path='tdmpc2_strict_causal_npz_rgb_v1',
				**kwargs,
			)
		finally:
			module.load_video_frames = original_loader
		store = created.get('store')
		if store is None:
			raise SAM21DependencyError(
				'SAM 2 init_state did not call its load_video_frames global; exact '
				'causal NPZ input is unsupported by this implementation.'
			)
		if state.get('images') is not store:
			raise SAM21DependencyError(
				'SAM 2 init_state replaced the guarded frame store; future-frame '
				'access can no longer be enforced.'
			)
		if int(state.get('num_frames', -1)) != len(store):
			raise SAM21ProtocolError(
				'SAM 2 inference state reports an unexpected frame count: '
				f'{state.get("num_frames")} vs {len(store)}.'
			)
		return state, store

	@staticmethod
	def _normalize_logits(logits, *, expected_roles: int, expected_size):
		if hasattr(logits, 'detach'):
			logits = logits.detach().float().cpu().numpy()
		array = np.asarray(logits, dtype=np.float32)
		if array.ndim == 4 and array.shape[1] == 1:
			array = array[:, 0]
		if array.ndim != 3 or array.shape[0] != expected_roles:
			raise SAM21ProtocolError(
				'SAM 2 output logits must have shape [K,1,H,W] or [K,H,W], '
				f'got {array.shape} for K={expected_roles}.'
			)
		if tuple(array.shape[1:]) != tuple(expected_size):
			raise SAM21ProtocolError(
				f'SAM 2 output resolution must be {expected_size}, got {array.shape[1:]}.'
			)
		if not np.isfinite(array).all():
			raise SAM21ProtocolError('SAM 2 output logits contain non-finite values.')
		return np.ascontiguousarray(array)

	def _ordered_logits(self, object_ids, logits) -> np.ndarray:
		ids = [int(value.item() if hasattr(value, 'item') else value) for value in object_ids]
		expected_ids = list(range(1, len(self.config.role_names) + 1))
		if sorted(ids) != expected_ids or len(set(ids)) != len(ids):
			raise SAM21ProtocolError(
				f'SAM 2 object IDs must be exactly {expected_ids}, got {ids}.'
			)
		array = self._normalize_logits(
			logits,
			expected_roles=len(ids),
			expected_size=self.config.expected_input_size,
		)
		order = [ids.index(role_id) for role_id in expected_ids]
		return array[order]

	def track_episode(
		self,
		*,
		support: SAM21SupportPack,
		episode_rgb: np.ndarray,
		support_source: str | Path | None = None,
		episode_source: str | Path | None = None,
	) -> SAM21EpisodeResult:
		"""Run one fresh support-conditioned, episode-unprompted sequence."""
		support = _validated_support(support, self.config)
		episode = _validated_rgb(
			episode_rgb,
			name='episode.rgb',
			expected_size=self.config.expected_input_size,
		)
		all_rgb = np.concatenate((support.rgb, episode), axis=0)
		support_count = support.rgb.shape[0]
		episode_count, height, width, _ = episode.shape
		num_roles = len(self.config.role_names)
		prompt_calls: list[dict[str, Any]] = []

		inference_context, autocast_context = self._contexts()
		with inference_context, autocast_context:
			self._sync()
			started = time.perf_counter()
			state, frame_store = self._init_causal_state(all_rgb)
			self._sync()
			initialization_ms = (time.perf_counter() - started) * 1000.0

			self._sync()
			started = time.perf_counter()
			for support_index in range(support_count):
				frame_store.allow_through(support_index)
				indexed = support.indexed_masks[support_index]
				for role_id, role_name in enumerate(self.config.role_names, start=1):
					mask = np.ascontiguousarray(indexed == role_id)
					try:
						self.predictor.add_new_mask(
							inference_state=state,
							frame_idx=support_index,
							obj_id=role_id,
							mask=mask,
						)
					except Exception as exc:
						raise SAM21ProtocolError(
							f'Failed to install fixed support frame {support_index}, '
							f'role {role_name!r}: {type(exc).__name__}: {exc}'
						) from exc
					prompt_calls.append({
						'frame_index': support_index,
						'role_id': role_id,
						'role_name': role_name,
						'pixels': int(mask.sum()),
					})
			self._sync()
			prompt_runtime_ms = (time.perf_counter() - started) * 1000.0

			# Official SAM 2 defers mask-memory encoding until propagation preflight.
			# Account this fixed-support work outside episode per-frame latency.
			self._sync()
			started = time.perf_counter()
			try:
				self.predictor.propagate_in_video_preflight(state)
			except Exception as exc:
				raise SAM21ProtocolError(
					'Failed to finalize fixed support memory before episode '
					f'propagation: {type(exc).__name__}: {exc}'
				) from exc
			self._sync()
			support_memory_finalize_ms = (
				(time.perf_counter() - started) * 1000.0
			)

			try:
				generator = iter(self.predictor.propagate_in_video(
					inference_state=state,
					start_frame_idx=support_count,
					max_frame_num_to_track=episode_count,
					reverse=False,
				))
			except Exception as exc:
				raise SAM21ProtocolError(
					f'Could not start forward-only SAM 2 propagation: {exc}'
				) from exc

			predicted = np.zeros((episode_count, height, width), dtype=np.uint8)
			confidence = np.zeros((episode_count, num_roles), dtype=np.float32)
			mask_score = np.zeros_like(confidence)
			lost = np.ones((episode_count, num_roles), dtype=np.bool_)
			valid = np.zeros_like(lost)
			area = np.zeros((episode_count, num_roles), dtype=np.int32)
			centroid = np.full(
				(episode_count, num_roles, 2), np.nan, dtype=np.float32
			)
			frame_runtime = np.zeros((episode_count,), dtype=np.float64)
			overlap_pixels = np.zeros((episode_count,), dtype=np.int32)

			for episode_index in range(episode_count):
				expected_frame = support_count + episode_index
				frame_store.allow_through(expected_frame)
				self._sync()
				frame_started = time.perf_counter()
				try:
					out_frame, object_ids, raw_logits = next(generator)
				except StopIteration as exc:
					raise SAM21ProtocolError(
						f'SAM 2 stopped before episode frame {episode_index}.'
					) from exc
				except Exception as exc:
					if isinstance(exc, SAM21BackendError):
						raise
					raise SAM21ProtocolError(
						f'SAM 2 failed on episode frame {episode_index}: '
						f'{type(exc).__name__}: {exc}'
					) from exc
				self._sync()
				actual_frame = int(
					out_frame.item() if hasattr(out_frame, 'item') else out_frame
				)
				if actual_frame != expected_frame:
					raise SAM21ProtocolError(
						'SAM 2 propagation must be consecutive and forward-only: '
						f'expected frame {expected_frame}, got {actual_frame}.'
					)
				logits = self._ordered_logits(object_ids, raw_logits)
				raw_positive = logits > self.config.mask_threshold
				overlap_pixels[episode_index] = int(
					(raw_positive.sum(axis=0) > 1).sum()
				)
				if self.config.exclusive_masks:
					winner = logits.argmax(axis=0)
					any_positive = raw_positive.any(axis=0)
					indexed = np.where(any_positive, winner + 1, 0).astype(np.uint8)
					role_masks = np.stack([
						indexed == role_id for role_id in range(1, num_roles + 1)
					])
				else:
					if np.any(raw_positive.sum(axis=0) > 1):
						raise SAM21ProtocolError(
							'Non-exclusive SAM masks cannot be represented by predicted_indexed; '
							'enable exclusive_masks.'
						)
					role_masks = raw_positive
					indexed = np.zeros((height, width), dtype=np.uint8)
					for role_id, role_mask in enumerate(role_masks, start=1):
						indexed[role_mask] = role_id
				predicted[episode_index] = indexed
				probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
				for role_index, role_mask in enumerate(role_masks):
					pixels = int(role_mask.sum())
					area[episode_index, role_index] = pixels
					is_valid = pixels >= self.config.min_mask_pixels
					valid[episode_index, role_index] = is_valid
					lost[episode_index, role_index] = not is_valid
					if pixels:
						score = float(probability[role_index][role_mask].mean())
						confidence[episode_index, role_index] = score
						mask_score[episode_index, role_index] = score
						yx = np.argwhere(role_mask).mean(axis=0)
						centroid[episode_index, role_index] = (yx[1], yx[0])
				frame_runtime[episode_index] = (
					(time.perf_counter() - frame_started) * 1000.0
				)

		if len(prompt_calls) != support_count * num_roles:
			raise SAM21ProtocolError('Internal fixed-support prompt accounting mismatch.')
		if any(call['frame_index'] >= support_count for call in prompt_calls):
			raise SAM21ProtocolError('An episode frame received a forbidden prompt.')
		if frame_store.future_access_attempts:
			raise SAM21CausalityError(
				f'Observed {frame_store.future_access_attempts} future-frame access attempt(s).'
			)
		expected_last_frame = support_count + episode_count - 1
		if frame_store.max_accessed_index != expected_last_frame:
			raise SAM21ProtocolError(
				'SAM 2 did not causally read every requested episode frame: '
				f'last accessed={frame_store.max_accessed_index}, '
				f'expected={expected_last_frame}.'
			)

		role_diagnostics = {}
		for index, role in enumerate(self.config.role_names):
			role_diagnostics[role] = {
				'valid_frame_rate': float(valid[:, index].mean()),
				'invalid_frames': int(lost[:, index].sum()),
				'max_invalid_burst': _max_invalid_burst(lost[:, index]),
				'mean_area_pixels': float(area[:, index].mean()),
				'mean_mask_score': float(mask_score[:, index].mean()),
			}
		diagnostics = {
			'format': _OUTPUT_FORMAT,
			'backend': 'official_sam_2_1_video_predictor',
			'model_size': self.config.model_size,
			'model_config': self.config.resolved_model_config,
			'roles': list(self.config.role_names),
			'input_size': list(self.config.expected_input_size),
			'support_frames': support_count,
			'episode_frames': episode_count,
			'prompt_protocol': _PROMPT_PROTOCOL,
			'prompt_frame_indices': list(range(support_count)),
			'prompt_calls': prompt_calls,
			'episode_prompt_used': False,
			'episode_gt_read_by_backend': False,
			'propagation_direction': 'forward_only',
			'strict_causal_frame_gate': True,
			'future_access_attempts': frame_store.future_access_attempts,
			'max_accessed_frame_index': frame_store.max_accessed_index,
			'frame_store_access_count': frame_store.access_count,
			'object_embedding_output': False,
			'exclusive_masks': self.config.exclusive_masks,
			'mask_threshold': self.config.mask_threshold,
			'min_mask_pixels': self.config.min_mask_pixels,
			'confidence_semantics': (
				'mean_sigmoid_mask_logit_over_final_positive_pixels_diagnostic_only'
			),
			'initialization_ms': float(initialization_ms),
			'prompt_runtime_ms': float(prompt_runtime_ms),
			'support_memory_finalize_ms': float(support_memory_finalize_ms),
			'episode_total_ms': float(frame_runtime.sum()),
			'ms_per_episode_frame': float(frame_runtime.mean()),
			'raw_overlap_pixels_total': int(overlap_pixels.sum()),
			'role_diagnostics': role_diagnostics,
			'support_rgb_sha256': _typed_array_sha256(support.rgb),
			'support_indexed_sha256': _typed_array_sha256(support.indexed_masks),
			'episode_rgb_sha256': _typed_array_sha256(episode),
			'support_source': str(support_source) if support_source else None,
			'episode_source': str(episode_source) if episode_source else None,
		}
		return SAM21EpisodeResult(
			role_names=self.config.role_names,
			predicted_indexed=predicted,
			confidence=confidence,
			mask_score=mask_score,
			lost=lost,
			valid=valid,
			area_pixels=area,
			centroid_xy=centroid,
			frame_runtime_ms=frame_runtime,
			raw_overlap_pixels=overlap_pixels,
			diagnostics=diagnostics,
		)


def _parse_args(argv: list[str] | None = None):
	parser = argparse.ArgumentParser(
		description=(
			'Run official SAM 2.1 with fixed multi-frame support masks and no '
			'episode GT prompts. The command never downloads resources.'
		)
	)
	parser.add_argument('--sam2-repo', type=Path, required=True)
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--model-size', choices=tuple(_MODEL_CONFIGS), default='large')
	parser.add_argument('--model-config')
	parser.add_argument('--device', default='cuda:0')
	parser.add_argument('--seed', type=int, default=2718281)
	parser.add_argument('--mask-threshold', type=float, default=0.0)
	parser.add_argument('--min-mask-pixels', type=int, default=1)
	parser.add_argument(
		'--amp-dtype', choices=('none', 'float16', 'bfloat16'), default='bfloat16'
	)
	parser.add_argument('--offload-video-to-cpu', action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument('--offload-state-to-cpu', action='store_true')
	parser.add_argument('--vos-optimized', action='store_true')
	parser.add_argument(
		'--inputs',
		type=Path,
		help='GT-free unified_vos_backend_inputs_v1 manifest.',
	)
	parser.add_argument('--output-root', type=Path)
	parser.add_argument(
		'--allow-nonstandard-counts',
		action='store_true',
		help='Contract-test only; scientific runs must not set this flag.',
	)
	parser.add_argument('--preflight-only', action='store_true')
	parser.add_argument('--traceback', action='store_true')
	return parser.parse_args(argv)


def _prediction_masks(result: SAM21EpisodeResult) -> np.ndarray:
	return np.stack([
		result.predicted_indexed == role_id
		for role_id in range(1, len(result.role_names) + 1)
	], axis=1).astype(np.bool_, copy=False)


def _validate_prediction_archive(
	path: Path,
	*,
	frames: int,
	roles: int,
	resolution: int,
) -> None:
	expected = {
		'predicted_masks', 'reported_confidence', 'reported_lost', 'runtime_ms'
	}
	with np.load(path, allow_pickle=False) as archive:
		if set(archive.files) != expected:
			raise SAM21ProtocolError(
				f'Unified prediction NPZ keys changed: {archive.files!r}.'
			)
		shapes = {
			'predicted_masks': (frames, roles, resolution, resolution),
			'reported_confidence': (frames, roles),
			'reported_lost': (frames, roles),
			'runtime_ms': (frames,),
		}
		dtypes = {
			'predicted_masks': np.dtype(np.bool_),
			'reported_confidence': np.dtype(np.float32),
			'reported_lost': np.dtype(np.bool_),
			'runtime_ms': np.dtype(np.float64),
		}
		for name in sorted(expected):
			value = archive[name]
			if value.shape != shapes[name] or value.dtype != dtypes[name]:
				raise SAM21ProtocolError(
					f'{name} must be {dtypes[name]} {shapes[name]}, got '
					f'{value.dtype} {value.shape}.'
				)
			if name in {'reported_confidence', 'runtime_ms'} and not np.isfinite(value).all():
				raise SAM21ProtocolError(f'{name} contains non-finite values.')


def _run_unified_manifest(args) -> tuple[Path, Path]:
	total_started = time.perf_counter()
	if args.inputs is None or args.output_root is None:
		raise SAM21ProtocolError(
			'Benchmark execution requires --inputs and --output-root.'
		)
	input_path = args.inputs.expanduser().resolve()
	input_manifest_sha = file_sha256(input_path)
	payload, support_paths, episode_paths = validate_backend_inputs(
		input_path,
		strict_counts=not args.allow_nonstandard_counts,
	)
	resolution = int(payload['resolution'])
	counts = payload['counts']
	output_root = args.output_root.expanduser().resolve()
	incomplete_root = output_root.with_name(output_root.name + '.incomplete')
	if output_root.exists():
		raise FileExistsError(f'Refusing to overwrite existing output: {output_root}')
	if incomplete_root.exists():
		raise FileExistsError(
			f'Refusing to overwrite existing incomplete output: {incomplete_root}'
		)
	if not output_root.parent.is_dir():
		raise FileNotFoundError(
			f'Output-root parent does not exist: {output_root.parent}'
		)

	first_task = payload['tasks'][0]
	base_config = SAM21VideoConfig(
		repo_path=args.sam2_repo,
		checkpoint_path=args.checkpoint,
		role_names=tuple(payload['roles'][first_task]),
		model_size=args.model_size,
		model_config=args.model_config,
		device=args.device,
		expected_input_size=(resolution, resolution),
		expected_support_frames=6,
		mask_threshold=args.mask_threshold,
		min_mask_pixels=args.min_mask_pixels,
		amp_dtype=args.amp_dtype,
		offload_video_to_cpu=args.offload_video_to_cpu,
		offload_state_to_cpu=args.offload_state_to_cpu,
		vos_optimized=args.vos_optimized,
	)
	torch = _require_torch()
	device = torch.device(base_config.device)
	if device.type != 'cuda' or (device.index not in (None, 0)):
		raise SAM21PreflightError(
			'Unified SAM 2.1 workers require logical device cuda:0; select the '
			'physical GPU with CUDA_VISIBLE_DEVICES.'
		)
	if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
		raise SAM21PreflightError(
			'Unified SAM 2.1 workers require exactly one visible logical CUDA device.'
		)
	if args.seed < 0:
		raise ValueError('--seed must be non-negative.')
	cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
	if not isinstance(cuda_visible_devices, str) or not cuda_visible_devices.strip():
		raise SAM21PreflightError(
			'CUDA_VISIBLE_DEVICES must be one explicit non-empty worker selector.'
		)
	if os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
		raise SAM21PreflightError('CUDA_DEVICE_ORDER must be PCI_BUS_ID.')
	if not str(os.environ.get('BENCHMARK_GPU_UUID', '')).startswith('GPU-'):
		raise SAM21PreflightError('BENCHMARK_GPU_UUID must bind the physical GPU.')
	adapter_path = Path(__file__).resolve()
	adapter_sha = _file_sha256(adapter_path)
	implementation_before = _tree_snapshot(base_config.repo / 'sam2')
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	preflight = inspect_sam21_installation(base_config, import_check=True)
	checkpoint_sha = _file_sha256(base_config.checkpoint)
	preflight['checkpoint_sha256'] = checkpoint_sha
	predictor = _build_predictor(base_config)

	incomplete_root.mkdir()
	results: dict[str, Any] = {}
	diagnostic_index: dict[str, Any] = {}
	for task in payload['tasks']:
		roles = tuple(payload['roles'][task])
		config = replace(base_config, role_names=roles)
		adapter = SAM21VideoAdapter(config, _predictor=predictor)
		support_path = support_paths[task]
		support = load_sam21_support_npz(support_path, config)
		if file_sha256(support_path) != payload['support'][task]['arrays_sha256']:
			raise SAM21ProtocolError(f'Fixed support changed while loading {task}.')
		results[task] = {}
		diagnostic_index[task] = {}
		for condition in payload['conditions']:
			results[task][condition] = []
			diagnostic_index[task][condition] = []
			for episode_index in range(int(counts['episodes'])):
				episode_path = episode_paths[(task, condition, episode_index)]
				episode_rgb = load_sam21_episode_rgb(episode_path, config)
				expected_rgb_sha = payload['episodes'][task][condition][
					episode_index
				]['rgb_arrays_sha256']
				if file_sha256(episode_path) != expected_rgb_sha:
					raise SAM21ProtocolError(
						f'Episode RGB changed while loading '
						f'{task}/{condition}/{episode_index}.'
					)
				if episode_rgb.shape[0] != int(counts['frames_per_episode']):
					raise SAM21ProtocolError(
						f'Episode frame count changed for {task}/{condition}/'
						f'{episode_index}: {episode_rgb.shape[0]}.'
					)
				print(
					'SAM21_EPISODE_START '
					f'task={task} condition={condition} episode={episode_index}',
					flush=True,
				)
				episode_started = time.perf_counter()
				result = adapter.track_episode(
					support=support,
					episode_rgb=episode_rgb,
					support_source=payload['support'][task]['arrays'],
					episode_source=payload['episodes'][task][condition][
						episode_index
					]['rgb_arrays'],
				)
				episode_wallclock_ms = (
					time.perf_counter() - episode_started
				) * 1000.0
				raw_runtime = result.frame_runtime_ms.astype(np.float64, copy=True)
				overhead_ms = max(0.0, episode_wallclock_ms - float(raw_runtime.sum()))
				result = replace(
					result,
					frame_runtime_ms=raw_runtime + overhead_ms / raw_runtime.shape[0],
				)
				relative_npz = Path('predictions') / task / condition / (
					f'episode_{episode_index:03d}.npz'
				)
				relative_json = Path('diagnostics') / task / condition / (
					f'episode_{episode_index:03d}.json'
				)
				npz_path = incomplete_root / relative_npz
				json_path = incomplete_root / relative_json
				npz_path.parent.mkdir(parents=True, exist_ok=True)
				json_path.parent.mkdir(parents=True, exist_ok=True)
				result.save(npz_path, json_path)
				_validate_prediction_archive(
					npz_path,
					frames=int(counts['frames_per_episode']),
					roles=len(roles),
					resolution=resolution,
				)
				predicted_masks = _prediction_masks(result)
				runtime_ms = result.frame_runtime_ms.astype(np.float64, copy=False)
				results[task][condition].append({
					'episode_index': episode_index,
					'frames': int(counts['frames_per_episode']),
					'prediction_arrays': relative_npz.as_posix(),
					'prediction_arrays_sha256': file_sha256(npz_path),
					'predicted_mask_trace_sha256': _typed_array_sha256(
						predicted_masks
					),
					'runtime_trace_sha256': _typed_array_sha256(runtime_ms),
				})
				diagnostic_index[task][condition].append({
					'episode_index': episode_index,
					'path': relative_json.as_posix(),
					'sha256': file_sha256(json_path),
				})
				print(
					'SAM21_EPISODE_END '
					f'task={task} condition={condition} episode={episode_index} '
					f'ms_per_frame={result.diagnostics["ms_per_episode_frame"]:.3f}',
					flush=True,
				)

	implementation_after = _tree_snapshot(base_config.repo / 'sam2')
	if implementation_after != implementation_before:
		raise SAM21PreflightError(
			'Official SAM 2.1 implementation/config tree changed during inference.'
		)
	if _file_sha256(base_config.checkpoint) != checkpoint_sha:
		raise SAM21PreflightError('SAM 2.1 checkpoint changed during inference.')
	if _file_sha256(adapter_path) != adapter_sha:
		raise SAM21PreflightError('SAM 2.1 adapter implementation changed during inference.')
	if file_sha256(input_path) != input_manifest_sha:
		raise SAM21PreflightError('GT-free backend input manifest changed during inference.')
	manifest = {
		'format': BACKEND_FORMAT,
		'status': 'complete',
		'backend': 'sam21',
		'dataset_id': payload['dataset_id'],
		'input_manifest_sha256': input_manifest_sha,
		'protocol': {
			'support_only_prompts': True,
			'support_frames_replayed_each_episode': 6,
			'episode_first_frame_gt_prompt': False,
			'mid_episode_reprompt': False,
			'episode_ground_truth_read': False,
			'causal_frame_order': True,
			'prompt_adapter': 'sam21_fixed_six_frame_indexed_mask_prefix_v1',
			'model_internal_features_exported': False,
			'reported_confidence_cross_backend_comparable': False,
			'strict_source_pixel_access_gate': True,
			'online_deployment_eligible': True,
			'runtime_ms_semantics': (
				'episode_end_to_end_amortized_per_frame_excluding_npz_io_'
				'and_model_construction_v1'
			),
		},
		'backend_provenance': {
			'model_family': 'sam2.1',
			'model_size': base_config.model_size,
			'implementation': implementation_before,
			'adapter_implementation': 'tdmpc2.perception.sam21_video_backend',
			'adapter_path': str(adapter_path),
			'adapter_sha256': adapter_sha,
			'implementation_format': _OUTPUT_FORMAT,
			'official_preflight': preflight,
			'checkpoint': str(base_config.checkpoint),
			'checkpoint_sha256': checkpoint_sha,
			'python': sys.version,
			'platform': platform.platform(),
			'torch': torch.__version__,
			'cuda_runtime': torch.version.cuda,
			'cuda_visible_devices': cuda_visible_devices,
			'gpu_uuid': os.environ.get('BENCHMARK_GPU_UUID'),
			'cuda_device_order': os.environ.get('CUDA_DEVICE_ORDER'),
			'logical_cuda_device': 0,
			'device_name': torch.cuda.get_device_name(0),
			'seed': args.seed,
			'amp_dtype': base_config.amp_dtype,
			'wallclock_seconds': time.perf_counter() - total_started,
			'prompt_protocol': _PROMPT_PROTOCOL,
			'fixed_support_frames': 6,
			'episode_prompt_used': False,
			'episode_gt_read_by_backend': False,
			'propagation_direction': 'forward_only',
			'strict_causal_frame_gate': True,
			'object_embedding_output': False,
			'exclusive_masks': True,
			'confidence_ranked': False,
			'confidence_semantics': (
				'mean_sigmoid_mask_logit_over_final_positive_pixels_diagnostic_only'
			),
			'diagnostics': diagnostic_index,
		},
		'results': results,
	}
	manifest_path = incomplete_root / 'backend_predictions.json'
	write_json(manifest_path, manifest)
	os.replace(incomplete_root, output_root)
	return output_root, output_root / manifest_path.name


def main(argv: list[str] | None = None) -> int:
	args = _parse_args(argv)
	try:
		if args.preflight_only:
			config = SAM21VideoConfig(
				repo_path=args.sam2_repo,
				checkpoint_path=args.checkpoint,
				role_names=('role_1', 'role_2'),
				model_size=args.model_size,
				model_config=args.model_config,
				device=args.device,
				amp_dtype=args.amp_dtype,
				vos_optimized=args.vos_optimized,
			)
			preflight = inspect_sam21_installation(config, import_check=True)
			preflight['checkpoint_sha256'] = _file_sha256(config.checkpoint)
			predictor = _build_predictor(config)
			del predictor
			torch = _require_torch()
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
			preflight['model_construction'] = 'passed'
			print('SAM21_VIDEO_BACKEND_PREFLIGHT_OK', json.dumps(preflight, sort_keys=True))
			return 0
		output_root, manifest_path = _run_unified_manifest(args)
	except Exception as exc:
		if args.traceback:
			traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
		print(
			f'SAM21_VIDEO_BACKEND_FAILED: {type(exc).__name__}: {exc}',
			file=sys.stderr,
		)
		return 2
	print('SAM21_VIDEO_BACKEND_COMPLETE')
	print(f'OUTPUT_ROOT={output_root}')
	print(f'PREDICTIONS={manifest_path}')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
