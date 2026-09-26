"""Dependency-light contracts for the strict SAM 2.1 backend.

The real model is intentionally not imported.  A predictor test double checks
support prompting, role ordering, output schema, GT-free episode loading, and
the hard future-frame access gate.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = REPO_ROOT / 'tdmpc2' / 'perception' / 'sam21_video_backend.py'
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))


class _FakeDevice:
	def __init__(self, value):
		self.value = str(value)
		self.type = self.value.split(':', 1)[0]
		self.index = None

	def __str__(self):
		return self.value


class _FakeTensor:
	def __init__(self, value=None):
		self.value = value

	def __getitem__(self, key):
		return self

	def permute(self, *dims):
		return self

	def float(self):
		return self

	def div_(self, value):
		return self

	def sub_(self, value):
		return self

	def to(self, *args, **kwargs):
		return self


def _install_fake_torch() -> None:
	if 'torch' in sys.modules:
		return
	fake = SimpleNamespace(
		device=_FakeDevice,
		inference_mode=contextlib.nullcontext,
		from_numpy=lambda value: _FakeTensor(value),
		tensor=lambda value, **kwargs: _FakeTensor(value),
		float32='float32',
		cuda=SimpleNamespace(
			is_available=lambda: False,
			synchronize=lambda device=None: None,
		),
	)
	sys.modules['torch'] = fake


def _load_backend():
	_install_fake_torch()
	spec = importlib.util.spec_from_file_location(
		'tdmpc2_sam21_backend_contract_module', BACKEND_PATH
	)
	if spec is None or spec.loader is None:
		raise RuntimeError(f'Cannot load backend source: {BACKEND_PATH}')
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return module


# The adapter temporarily replaces this exact module-global name.  It mirrors
# the official SAM2VideoPredictor module contract.
def load_video_frames(*args, **kwargs):  # pragma: no cover - always patched
	raise AssertionError('adapter did not install its guarded loader')


class _FakePredictor:
	image_size = 16

	def __init__(self, *, prefetch_future=False):
		self.prompt_calls = []
		self.prefetch_future = prefetch_future

	def init_state(
		self,
		video_path,
		offload_video_to_cpu=True,
		offload_state_to_cpu=False,
		async_loading_frames=False,
	):
		images, height, width = load_video_frames(
			video_path=video_path,
			image_size=self.image_size,
			offload_video_to_cpu=offload_video_to_cpu,
			async_loading_frames=async_loading_frames,
			compute_device=_FakeDevice('cpu'),
		)
		images[0]
		return {
			'images': images,
			'num_frames': len(images),
			'video_height': height,
			'video_width': width,
		}

	def add_new_mask(self, *, inference_state, frame_idx, obj_id, mask):
		inference_state['images'][frame_idx]
		self.prompt_calls.append((int(frame_idx), int(obj_id), int(mask.sum())))
		return frame_idx, [obj_id], np.zeros((1, 1, *mask.shape), np.float32)

	def propagate_in_video_preflight(self, inference_state):
		return None

	def propagate_in_video(
		self,
		*,
		inference_state,
		start_frame_idx,
		max_frame_num_to_track,
		reverse,
	):
		assert reverse is False
		height = inference_state['video_height']
		width = inference_state['video_width']
		for frame_idx in range(
			start_frame_idx, start_frame_idx + max_frame_num_to_track
		):
			if self.prefetch_future and frame_idx + 1 < inference_state['num_frames']:
				# This must be rejected before any image/tensor preprocessing happens.
				inference_state['images'][frame_idx + 1]
			inference_state['images'][frame_idx]
			logits = np.full((2, 1, height, width), -4.0, dtype=np.float32)
			# Deliberately return object IDs in reverse order. Axis 0 is role 2.
			logits[0, 0, :, width // 2:] = 3.0
			logits[1, 0, :, :width // 2] = 2.0
			yield frame_idx, [2, 1], logits


def _support(frames=6, height=8, width=8):
	rgb = np.zeros((frames, height, width, 3), dtype=np.uint8)
	indexed = np.zeros((frames, height, width), dtype=np.uint8)
	indexed[:, :, :width // 2] = 1
	indexed[:, :, width // 2:] = 2
	return rgb, indexed


def main() -> None:
	backend = _load_backend()
	config = backend.SAM21VideoConfig(
		repo_path='unused-test-repo',
		checkpoint_path='unused-test-checkpoint',
		role_names=('left_role', 'right_role'),
		model_size='large',
		device='cpu',
		expected_input_size=(8, 8),
		expected_support_frames=6,
		amp_dtype='none',
	)
	support_rgb, support_masks = _support()
	support = backend.SAM21SupportPack(
		rgb=support_rgb,
		indexed_masks=support_masks,
	)
	episode = np.zeros((3, 8, 8, 3), dtype=np.uint8)
	predictor = _FakePredictor()
	adapter = backend.SAM21VideoAdapter(config, _predictor=predictor)
	result = adapter.track_episode(support=support, episode_rgb=episode)

	expected_prompt_order = [
		(frame, role_id)
		for frame in range(6)
		for role_id in (1, 2)
	]
	assert [(frame, role) for frame, role, _ in predictor.prompt_calls] == expected_prompt_order
	assert all(frame < 6 for frame, _, _ in predictor.prompt_calls)
	assert result.predicted_indexed.shape == (3, 8, 8)
	assert np.all(result.predicted_indexed[:, :, :4] == 1)
	assert np.all(result.predicted_indexed[:, :, 4:] == 2)
	assert result.confidence.dtype == np.float32
	assert result.lost.dtype == np.bool_
	assert not result.lost.any()
	assert result.frame_runtime_ms.dtype == np.float64
	assert result.diagnostics['episode_prompt_used'] is False
	assert result.diagnostics['episode_gt_read_by_backend'] is False
	assert result.diagnostics['future_access_attempts'] == 0

	with tempfile.TemporaryDirectory(prefix='sam21_contract_') as directory:
		root = Path(directory)
		npz_path = root / 'prediction.npz'
		json_path = root / 'diagnostics.json'
		result.save(npz_path, json_path)
		with np.load(npz_path, allow_pickle=False) as archive:
			assert set(archive.files) == {
				'predicted_masks',
				'reported_confidence',
				'reported_lost',
				'runtime_ms',
			}
			assert archive['predicted_masks'].shape == (3, 2, 8, 8)
			assert archive['predicted_masks'].dtype == np.bool_
			assert archive['reported_confidence'].dtype == np.float32
			assert archive['reported_lost'].dtype == np.bool_
			assert archive['runtime_ms'].dtype == np.float64

		# An object-dtype GT array would fail under allow_pickle=False if read.
		# Successful loading therefore proves the backend touched only RGB.
		episode_path = root / 'rgb_only_view.npz'
		with episode_path.open('wb') as handle:
			np.savez_compressed(
				handle,
				rgb=episode,
				gt_indexed=np.asarray([{'forbidden': True}], dtype=object),
			)
		loaded = backend.load_sam21_episode_rgb(episode_path, config)
		assert np.array_equal(loaded, episode)

	bad_predictor = _FakePredictor(prefetch_future=True)
	bad_adapter = backend.SAM21VideoAdapter(config, _predictor=bad_predictor)
	try:
		bad_adapter.track_episode(support=support, episode_rgb=episode)
	except backend.SAM21CausalityError as exc:
		assert 'future frame' in str(exc)
	else:
		raise AssertionError('Future-frame prefetch was not rejected.')

	missing_role = support_masks.copy()
	missing_role[2][missing_role[2] == 2] = 0
	try:
		backend.SAM21VideoAdapter(config, _predictor=_FakePredictor()).track_episode(
			support=backend.SAM21SupportPack(support_rgb, missing_role),
			episode_rgb=episode,
		)
	except backend.SAM21ProtocolError as exc:
		assert 'no prompt pixels' in str(exc)
	else:
		raise AssertionError('Missing fixed-support role was not rejected.')

	print('SAM21_VIDEO_BACKEND_CONTRACT_OK')


if __name__ == '__main__':
	main()
