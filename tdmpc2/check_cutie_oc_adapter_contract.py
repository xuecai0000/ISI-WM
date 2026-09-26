"""Dependency-free contract test for the standalone Cutie OC adapter."""

from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from tdmpc2.perception import cutie_oc_adapter as cutie_module  # noqa: E402
from tdmpc2.perception.cutie_oc_adapter import (  # noqa: E402
	CutieOCAdapter,
	CutieOCConfig,
	CutieSupportPrompts,
)
from tdmpc2.tools import check_cutie_oc_preflight as preflight_module  # noqa: E402


class _FakeObjectTransformer:
	def __init__(self, roles):
		self.roles = roles
		self.obj_values_cache = None
		self.query_cache = None
		self.query_emb_cache = None
		self.query_post_process_cache = None
		self.q_weights_cache = None
		self.defocus_cache = None

	def populate_tracking_caches(self):
		self.obj_values_cache = torch.tensor([1.0])
		self.query_cache = torch.tensor([2.0])
		self.query_emb_cache = torch.tensor([3.0])
		self.query_post_process_cache = torch.arange(
			self.roles * 16 * 256, dtype=torch.float32
		).reshape(self.roles, 16, 256)
		self.defocus_cache = (
			torch.tensor([1, 0, 1], dtype=torch.float32)
			if self.roles == 3 else torch.ones(self.roles, dtype=torch.float32)
		)
		self.q_weights_cache = torch.tensor([4.0])


class _FakeNetwork:
	def __init__(self, roles):
		self.object_transformer = _FakeObjectTransformer(roles)


class _FakeProcessor:
	def __init__(self, roles, *, network=None):
		self.roles = roles
		self.network = network if network is not None else _FakeNetwork(roles)
		self.reset_calls = 0
		self.prompt_calls = 0

	def clear_non_permanent_memory(self):
		self.reset_calls += 1

	def step(self, frame, mask=None, idx_mask=False, force_permanent=False):
		assert frame.shape == (3, 8, 8)
		if mask is not None:
			assert mask.shape == (self.roles, 8, 8)
			assert idx_mask is False
			self.prompt_calls += 1
			# Match official Cutie: permanent prompt installation need not fill
			# the feature caches consumed by OC-STORM.
			self.network.object_transformer.query_post_process_cache = None
			self.network.object_transformer.defocus_cache = None
		else:
			self.network.object_transformer.populate_tracking_caches()
		prediction = torch.full((self.roles + 1, 8, 8), 0.01)
		prediction[0] = 0.1
		if self.roles == 3:
			prediction[1, :4, :4] = 0.9
			prediction[2, :4, 4:] = 0.9
			prediction[3, 4:, :] = 0.9
		else:
			prediction[1, :4, :] = 0.9
			prediction[2, 4:, :] = 0.9
		return prediction


def _check_dataset_cfg_is_applied_before_model_construction(roles):
	"""Regression for OC-STORM's mutating ``get_dataset_cfg`` contract."""
	events = []
	model_cfg = {'weights': None, 'mem_every': None, 'use_long_term': None}

	def compose(*, config_name):
		assert config_name == 'eval_config_small'
		events.append('compose')
		return model_cfg

	def get_dataset_cfg(cfg):
		assert cfg is model_cfg
		assert str(cfg['weights']).endswith('synthetic-cutie.pth')
		cfg['mem_every'] = 5
		cfg['use_long_term'] = True
		events.append('dataset_cfg')
		return {'mem_every': 5, 'use_long_term': True}

	class FakeCUTIE:
		def __init__(self, cfg):
			assert cfg['mem_every'] == 5
			assert cfg['use_long_term'] is True
			events.append('model')

		def to(self, device):
			assert device.type == 'cpu'
			return self

		def eval(self):
			return self

		def load_weights(self, weights):
			assert weights == {'synthetic': True}
			events.append('weights')

	class FakeInferenceCore:
		def __init__(self, model, *, cfg):
			assert isinstance(model, FakeCUTIE)
			assert cfg['mem_every'] == 5
			events.append('processor')

	class FakeGlobalHydraInstance:
		def is_initialized(self):
			return False

	class FakeGlobalHydra:
		@staticmethod
		def instance():
			return FakeGlobalHydraInstance()

	hydra = types.ModuleType('hydra')
	hydra.compose = compose
	hydra.initialize_config_dir = lambda **_kwargs: contextlib.nullcontext()
	hydra_core = types.ModuleType('hydra.core')
	hydra_global = types.ModuleType('hydra.core.global_hydra')
	hydra_global.GlobalHydra = FakeGlobalHydra
	omegaconf = types.ModuleType('omegaconf')
	omegaconf.open_dict = lambda _cfg: contextlib.nullcontext()
	config = CutieOCConfig(
		repo_path='synthetic-oc-storm',
		checkpoint_path='synthetic-cutie.pth',
		role_names=roles,
		device='cpu',
	)
	modules = {
		'hydra': hydra,
		'hydra.core': hydra_core,
		'hydra.core.global_hydra': hydra_global,
		'omegaconf': omegaconf,
	}
	with (
		mock.patch.dict(sys.modules, modules),
		mock.patch.object(
			cutie_module,
			'_import_official_modules',
			return_value=(FakeCUTIE, FakeInferenceCore, get_dataset_cfg),
		),
		mock.patch.object(
			cutie_module.torch, 'load', return_value={'synthetic': True}
		),
	):
		model, processor = cutie_module._load_model(config)
	assert isinstance(model, FakeCUTIE)
	assert isinstance(processor, FakeInferenceCore)
	assert events == ['compose', 'dataset_cfg', 'model', 'weights', 'processor']


def _check_dependency_diagnostic_reports_every_missing_module():
	missing = {'hydra', 'einops', 'PIL'}

	def fake_find_spec(module):
		return None if module in missing else object()

	with mock.patch.object(
		preflight_module.importlib.util, 'find_spec', side_effect=fake_find_spec
	):
		without_support = preflight_module._runtime_dependency_status(
			include_support=False
		)
		assert 'PIL' not in without_support
		try:
			preflight_module._check_runtime_dependencies(include_support=True)
		except RuntimeError as exc:
			message = str(exc)
			assert 'hydra (pip package hydra-core)' in message
			assert 'einops (pip package einops)' in message
			assert 'PIL (pip package Pillow)' in message
			assert 'python -m pip install hydra-core einops Pillow' in message
		else:
			raise AssertionError('All missing Cutie modules must fail in one report.')


def _check_fresh_core_support_replay_is_history_independent(config, support, frame):
	"""A production-style reset must replace the core and replay fixed support."""
	shared_network = _FakeNetwork(len(config.role_names))
	initial = _FakeProcessor(len(config.role_names), network=shared_network)
	created = []

	def factory():
		processor = _FakeProcessor(len(config.role_names), network=shared_network)
		created.append(processor)
		return processor

	adapter = CutieOCAdapter(
		config,
		_processor=initial,
		_processor_factory=factory,
	)
	adapter.add_support_prompts(support)
	assert initial.prompt_calls == 1
	# Populate the shared OC-STORM export caches with episode-dependent output.
	adapter.track(frame)
	cache_names = (
		'obj_values_cache', 'query_cache', 'query_emb_cache',
		'query_post_process_cache', 'q_weights_cache', 'defocus_cache',
	)
	assert all(
		getattr(shared_network.object_transformer, name) is not None
		for name in cache_names
	)

	adapter.reset_episode()
	assert initial.reset_calls == 1
	assert len(created) == 1
	assert adapter.processor is created[0]
	assert created[0].prompt_calls == 1
	assert created[0].reset_calls == 1
	assert all(
		getattr(shared_network.object_transformer, name) is None
		for name in cache_names
	)

	adapter.track(frame)
	first_result = adapter.runtime_summary()
	adapter.reset_episode()
	assert len(created) == 2
	assert created[0].reset_calls == 2
	assert adapter.processor is created[1]
	assert created[1].prompt_calls == 1
	assert created[1].reset_calls == 1
	second_result = adapter.runtime_summary()

	assert first_result['prompt_frames'] == 1.0
	assert first_result['permanent_prompts'] == 1.0
	assert first_result['episode_hard_resets'] == 1.0
	assert first_result['support_replay_frames'] == 1.0
	assert second_result['prompt_frames'] == 1.0
	assert second_result['permanent_prompts'] == 1.0
	assert second_result['episode_hard_resets'] == 2.0
	assert second_result['support_replay_frames'] == 2.0
	assert second_result['episode_reset_strategy'] == (
		'fresh_inference_core_support_replay_v1'
	)
	# Replay uses owned copies; caller mutation cannot change later episodes.
	assert adapter._support_prompts.frames[0] is not support.frames[0]
	assert adapter._support_prompts.masks[0] is not support.masks[0]


def _check_cross_resolution_support_and_canonical_mask_output(roles):
	"""High-res runtime keeps frozen support and descriptor geometry at 64-scale."""
	processor = _FakeProcessor(len(roles))
	config = CutieOCConfig(
		repo_path='does-not-exist-and-must-not-be-touched',
		checkpoint_path='does-not-exist-and-must-not-be-touched.pth',
		role_names=roles,
		device='cpu',
		output_device='cpu',
		expected_input_size=(16, 16),
		support_input_size=(8, 8),
		mask_output_size=(8, 8),
		tracker_size=(8, 8),
		amp=False,
	)
	adapter = CutieOCAdapter(config, _processor=processor)
	support_frame = np.zeros((8, 8, 3), dtype=np.uint8)
	support_mask = np.zeros((8, 8), dtype=np.uint8)
	support_mask[:4, :4] = 1
	support_mask[:4, 4:] = 2
	support_mask[4:, :] = 3
	adapter.add_support_prompts(CutieSupportPrompts(
		frames=(support_frame,),
		masks=(support_mask,),
		role_names=roles,
		annotation_path=Path('synthetic-cross-resolution-support.json'),
		metadata={'split': 'support'},
	))
	result = adapter.track(np.zeros((16, 16, 3), dtype=np.uint8))
	assert result.input_size == (16, 16)
	assert result.mask_output_size == (8, 8)
	assert result.tracker_size == (8, 8)
	assert result.masks.shape == (3, 8, 8)
	assert result.centroid_xy.shape == (3, 2)
	summary = adapter.runtime_summary()
	assert summary['input_size'] == (16, 16)
	assert summary['support_input_size'] == (8, 8)
	assert summary['mask_output_size'] == (8, 8)
	assert summary['tracker_size'] == (8, 8)
	try:
		adapter.track(np.zeros((8, 8, 3), dtype=np.uint8))
	except ValueError as exc:
		assert '(16, 16)' in str(exc)
	else:
		raise AssertionError('Runtime frames must not silently fall back to support size.')


def main():
	roles = ('proximal_link', 'distal_link', 'goal')
	_check_cross_resolution_support_and_canonical_mask_output(roles)
	_check_dataset_cfg_is_applied_before_model_construction(roles)
	_check_dependency_diagnostic_reports_every_missing_module()
	processor = _FakeProcessor(len(roles))
	config = CutieOCConfig(
		repo_path='does-not-exist-and-must-not-be-touched',
		checkpoint_path='does-not-exist-and-must-not-be-touched.pth',
		role_names=roles,
		device='cpu',
		output_device='cpu',
		expected_input_size=(8, 8),
		amp=False,
	)
	adapter = CutieOCAdapter(config, _processor=processor)
	frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
	first_mask = np.zeros((8, 8), dtype=np.uint8)
	first_mask[:4, :4] = 1
	first_mask[:4, 4:] = 2
	first_mask[4:, :] = 3
	support = CutieSupportPrompts(
		frames=(frames[0],),
		masks=(first_mask,),
		role_names=roles,
		annotation_path=Path('synthetic-contract-only.json'),
		metadata={'split': 'support'},
	)
	adapter.add_support_prompts(support)
	results = adapter.track_episode(frames)

	assert processor.reset_calls == 1
	assert processor.prompt_calls == 1
	assert len(results) == len(frames)
	for result in results:
		assert result.role_names == roles
		assert result.masks.shape == (3, 8, 8)
		assert result.masks.dtype == torch.bool
		assert result.centroid_xy.shape == (3, 2)
		assert result.object_features.shape == (3, 2048)
		assert torch.count_nonzero(result.object_features[1]) == 0
		assert result.lost.tolist() == [False, True, False]
		assert result.confidence.tolist() == [1.0, 0.0, 1.0]
		assert torch.all(result.mask_score > 0.8)
		assert result.runtime_ms >= 0.0
		assert result.input_size == (8, 8)
		assert result.tracker_size == (8, 8)
		assert result.resized_from_input is False

	summary = adapter.runtime_summary()
	assert summary['frames'] == float(len(frames))
	assert summary['prompt_frames'] == 1.0
	assert summary['permanent_prompts'] == 1.0
	assert summary['episode_reset_strategy'] == (
		'legacy_clear_non_permanent_only_test_double'
	)
	assert summary['ms_per_frame'] >= 0.0

	_check_fresh_core_support_replay_is_history_independent(
		config, support, frames[0]
	)

	missing_role = first_mask.copy()
	missing_role[missing_role == 3] = 0
	missing_support = CutieSupportPrompts(
		frames=(frames[0],),
		masks=(missing_role,),
		role_names=roles,
		annotation_path=Path('synthetic-missing-role.json'),
		metadata={'split': 'support'},
	)
	missing_adapter = CutieOCAdapter(
		config, _processor=_FakeProcessor(len(roles))
	)
	try:
		missing_adapter.add_support_prompts(missing_support)
	except ValueError as exc:
		assert 'goal' in str(exc)
	else:
		raise AssertionError('A missing role prompt must fail closed.')

	# The legacy default must return the exact historical indexed prompt bytes.
	legacy_mask = cutie_module._support_mask_for_object_schema(
		first_mask, 'legacy_three_object_v1'
	)
	assert legacy_mask is first_mask
	assert legacy_mask.tobytes() == first_mask.tobytes()

	# The explicit whole-arm schema unions the existing proximal/distal raster,
	# preserves goal pixels, and drives Cutie with exactly two object IDs.
	whole_roles = ('whole_arm', 'goal')
	whole_mask = cutie_module._support_mask_for_object_schema(
		first_mask, 'whole_arm_goal_v1'
	)
	assert set(np.unique(whole_mask).tolist()) == {1, 2}
	assert np.array_equal(whole_mask == 1, (first_mask == 1) | (first_mask == 2))
	assert np.array_equal(whole_mask == 2, first_mask == 3)
	whole_config = CutieOCConfig(
		repo_path='does-not-exist-and-must-not-be-touched',
		checkpoint_path='does-not-exist-and-must-not-be-touched.pth',
		role_names=whole_roles,
		device='cpu',
		output_device='cpu',
		expected_input_size=(8, 8),
		amp=False,
		object_schema='whole_arm_goal_v1',
	)
	whole_adapter = CutieOCAdapter(
		whole_config, _processor=_FakeProcessor(len(whole_roles))
	)
	whole_support = CutieSupportPrompts(
		frames=(frames[0],),
		masks=(whole_mask,),
		role_names=whole_roles,
		annotation_path=Path('synthetic-whole-arm-contract-only.json'),
		metadata={'split': 'support', 'object_schema': 'whole_arm_goal_v1'},
	)
	whole_adapter.add_support_prompts(whole_support)
	whole_results = whole_adapter.track_episode(frames)
	for result in whole_results:
		assert result.role_names == whole_roles
		assert result.masks.shape == (2, 8, 8)
		assert result.object_features.shape == (2, 2048)
		assert result.lost.tolist() == [False, False]
		assert result.confidence.tolist() == [1.0, 1.0]

	try:
		CutieOCConfig(
			repo_path='unused',
			checkpoint_path='unused',
			role_names=roles,
			object_schema='whole_arm_goal_v1',
		).validated()
	except ValueError as exc:
		assert 'whole_arm' in str(exc)
	else:
		raise AssertionError('Whole-arm schema must reject legacy three-role names.')

	print('CUTIE_OC_ADAPTER_CONTRACT_OK', {
		'roles': roles,
		'feature_shape': tuple(results[0].object_features.shape),
		'lost': results[0].lost.tolist(),
		'frames': len(results),
		'dataset_cfg_applied': True,
		'dependency_batch_diagnostic': True,
		'whole_arm_roles': whole_roles,
		'whole_arm_feature_shape': tuple(whole_results[0].object_features.shape),
		'legacy_prompt_bytes_unchanged': True,
	})


if __name__ == '__main__':
	main()
