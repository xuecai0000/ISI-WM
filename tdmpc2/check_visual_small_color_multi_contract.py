"""Strict environment contract for Visual-Small Color-multi experiments.

This audit is intentionally independent of training. It pins the task geometry,
the TD-MPC2 temporal/image wrappers, immutable video splits, support provenance,
and the observation/RNG parity between official RGB and RewardGraph.

Run on the experiment server with the real external videos and manually labelled
support pack::

    python tdmpc2/check_visual_small_color_multi_contract.py \
        --video-root /path/to/video_hard \
        --support /path/to/annotations.json

The production-path checks read RGB observations and explicit background debug
properties only. A separate negative-control audit compares DMControl's public
easy/Visual-Small observations and rewards under fixed actions; no test reads
physics coordinates, and no such values enter the learner or support pack.
"""

import argparse
import ast
from collections import OrderedDict
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import textwrap
from types import SimpleNamespace

os.environ.setdefault('MUJOCO_GL', 'egl')

import gymnasium as gym
import numpy as np
from PIL import Image
import torch
import yaml

from dm_control import suite

from envs import make_env
from envs.dmcontrol import DMControlWrapper, Pixels, make_env as make_dmcontrol_env
from envs.tasks import reacher as custom_reacher
from envs.wrappers.flat_anchor import FlatAnchorWrapper
from envs.wrappers.timeout import Timeout
from envs.wrappers.video_background import (
	ColorMultiSplitSelector,
	ColorMultiVideoBackgroundWrapper,
)


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / 'config.yaml'
MANIFEST_DIR = ROOT / 'envs' / 'background_manifests'
COLLECTOR_PATH = ROOT / 'tools' / 'collect_visual_small_color_support.py'
TRAIN_PATH = ROOT / 'train.py'
ROLES = ('base', 'elbow', 'control_tip', 'goal')
SPLIT_INDICES = {
	'train': range(0, 70),
	'validation': range(70, 80),
	'test': range(80, 85),
	'support': range(85, 90),
}
MANIFEST_SHA256 = {
	'train': 'f988e6b02682c7a600559f45588d505084e70e7ba857a6e82ba7a5e120a03a22',
	'validation': 'b0786a2c8e51e2fadffebff48e3445c96d8d98632e29f22c207a7827a7694b9e',
	'test': 'a7ba627b17a00164a02363f400e7b4b5cf38127a792c4f015ee1e436cc780d5b',
	'support': '487f0a19166a61375140bce2024ddffbc4e0c84730534e3e392d7115029fde9c',
}
COMBINED_MANIFEST_SHA256 = (
	'3afb66d6db1c85771c62b7e75bcfccb0539613c4876071f1d5bb4f68626e9d99'
)
FORBIDDEN_PRIVILEGED_NAMES = {
	'physics',
	'qpos',
	'qvel',
	'geom_xpos',
	'site_xpos',
	'body_xpos',
	'ground_truth',
	'simulator_state',
	'privileged_state',
}
FORBIDDEN_SUPPORT_KEYS = FORBIDDEN_PRIVILEGED_NAMES | {
	'action',
	'reward',
	'state',
}


class Config(dict):
	"""Attribute-accessible copy of the flat YAML configuration."""

	def __getattr__(self, key):
		try:
			return self[key]
		except KeyError as exc:
			raise AttributeError(key) from exc

	def __setattr__(self, key, value):
		self[key] = value


class _ArraySpec:
	def __init__(self, shape, minimum=-1., maximum=1., dtype=np.float32):
		self.shape = tuple(shape)
		self.minimum = np.full(self.shape, minimum, dtype=dtype)
		self.maximum = np.full(self.shape, maximum, dtype=dtype)
		self.dtype = np.dtype(dtype)


class _CountingDMEnv:
	"""Minimal dm_env-like object used to count low-level action repeats."""

	def __init__(self):
		self.step_calls = 0
		self.last_action = None

	def observation_spec(self):
		return OrderedDict(observation=_ArraySpec((1,)))

	def action_spec(self):
		return _ArraySpec((2,))

	def reset(self):
		return SimpleNamespace(
			observation=OrderedDict(observation=np.zeros(1, dtype=np.float32))
		)

	def step(self, action):
		self.step_calls += 1
		self.last_action = action.copy()
		return SimpleNamespace(
			observation=OrderedDict(
				observation=np.array([self.step_calls], dtype=np.float32)
			),
			reward=1.25,
		)


class _RenderEnv(gym.Env):
	"""Small deterministic renderer used to verify Pixels stacking."""

	def __init__(self):
		super().__init__()
		self.observation_space = gym.spaces.Box(-1., 1., shape=(1,), dtype=np.float32)
		self.action_space = gym.spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
		self.render_calls = 0

	def reset(self, *, seed=None, options=None):
		return np.zeros(1, dtype=np.float32)

	def step(self, action):
		return np.zeros(1, dtype=np.float32), 0., False, {}

	def render(self, width=64, height=64, camera_id=None):
		self.render_calls += 1
		frame = np.empty((height, width, 3), dtype=np.uint8)
		frame[..., 0] = self.render_calls
		frame[..., 1] = 2 * self.render_calls
		frame[..., 2] = 3 * self.render_calls
		return frame


class _NeverDoneEnv(gym.Env):
	def __init__(self):
		super().__init__()
		self.observation_space = gym.spaces.Box(-1., 1., shape=(1,), dtype=np.float32)
		self.action_space = gym.spaces.Box(-1., 1., shape=(1,), dtype=np.float32)

	def reset(self, *, seed=None, options=None):
		return np.zeros(1, dtype=np.float32)

	def step(self, action):
		return np.zeros(1, dtype=np.float32), 0., False, {}


def _load_config():
	with CONFIG_PATH.open(encoding='utf-8') as stream:
		return Config(yaml.safe_load(stream))


def _sha256(path):
	return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _raw_image_sha256(path):
	image = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
	return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _numpy_rng_equal(left, right):
	return (
		left[0] == right[0]
		and np.array_equal(left[1], right[1])
		and left[2:] == right[2:]
	)


def _capture_rng():
	return {
		'python': random.getstate(),
		'numpy': np.random.get_state(),
		'torch': torch.get_rng_state().clone(),
		'cuda': [state.clone() for state in torch.cuda.get_rng_state_all()],
	}


def _rng_equal(left, right):
	return (
		left['python'] == right['python']
		and _numpy_rng_equal(left['numpy'], right['numpy'])
		and torch.equal(left['torch'], right['torch'])
		and len(left['cuda']) == len(right['cuda'])
		and all(torch.equal(a, b) for a, b in zip(left['cuda'], right['cuda']))
	)


def _assert_rng_equal(left, right, label):
	if not _rng_equal(left, right):
		raise AssertionError(f'{label} advanced a process-global RNG stream.')


def _seed_all(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def _wrapper_chain(env):
	chain = []
	seen = set()
	current = env
	while current is not None and id(current) not in seen:
		seen.add(id(current))
		chain.append(type(current).__name__)
		current = getattr(current, 'env', None)
	return tuple(chain)


def _find_background(env):
	seen = set()
	current = env
	while current is not None and id(current) not in seen:
		seen.add(id(current))
		if isinstance(current, ColorMultiVideoBackgroundWrapper):
			return current
		current = getattr(current, 'env', None)
	raise AssertionError('ColorMultiVideoBackgroundWrapper is missing from the chain.')


def _rgb(observation, reward_graph):
	value = observation['rgb'] if reward_graph else observation
	value = torch.as_tensor(value).detach().cpu()
	assert value.shape == (9, 64, 64), value.shape
	assert value.dtype == torch.uint8, value.dtype
	return value.clone()


def _validated_anchor(observation):
	anchor = torch.as_tensor(observation['anchor']).detach().cpu().float().reshape(25)
	assert torch.isfinite(anchor).all(), 'Anchor contains non-finite values.'
	assert bool(((anchor[:8] >= -1.) & (anchor[:8] <= 1.)).all()), (
		'Anchor role positions must be normalized to [-1, 1].'
	)
	assert bool(((anchor[20:24] >= 0.) & (anchor[20:24] <= 1.)).all()), (
		'Anchor confidences must lie in [0, 1].'
	)
	assert float(anchor[24]) in (0., 1.), 'Anchor fallback flag must be binary.'
	return anchor.clone()


def check_fixed_task_and_wrappers():
	defaults = _load_config()
	assert defaults.video_background_enabled is False
	assert defaults.video_background_root is None
	assert defaults.video_background_split == 'train'
	assert defaults.video_background_manifest_dir is None
	assert defaults.video_background_strength == 1.0
	assert defaults.video_background_total_frames == 1000
	assert defaults.video_background_source_cache_size == 8
	assert defaults.flat_anchor is False

	assert custom_reacher._VISUAL_SMALL_TARGET == .03
	assert custom_reacher._VISUAL_SMALL_REWARD_TARGET == .05
	assert ('reacher', 'visual_small') in suite.ALL_TASKS
	raw_env = suite.load(
		'reacher',
		'visual_small',
		task_kwargs={'random': 314159},
		visualize_reward=False,
	)
	task = raw_env._task
	assert type(task) is custom_reacher.VisualSmallReacher
	assert task._target_size == .03
	assert task._reward_target_radius == .05

	# Apart from the rendered target radius, Visual Small must be the official
	# easy task: paired public observations, rewards, discounts and step types are
	# bitwise/equal under the same seed and low-level actions.
	easy = suite.load(
		'reacher',
		'easy',
		task_kwargs={'random': 271828},
		visualize_reward=False,
	)
	visual = suite.load(
		'reacher',
		'visual_small',
		task_kwargs={'random': 271828},
		visualize_reward=False,
	)
	easy_reset = easy.reset()
	visual_reset = visual.reset()
	assert easy_reset.step_type == visual_reset.step_type
	assert tuple(easy_reset.observation) == tuple(visual_reset.observation)
	assert all(
		np.array_equal(easy_reset.observation[key], visual_reset.observation[key])
		for key in easy_reset.observation
	)
	for index in range(32):
		action = np.array([
			math.sin(.37 * (index + 1)),
			math.cos(.53 * (index + 1)),
		], dtype=easy.action_spec().dtype)
		easy_step = easy.step(action)
		visual_step = visual.step(action)
		assert easy_step.step_type == visual_step.step_type
		assert easy_step.reward == visual_step.reward
		assert easy_step.discount == visual_step.discount
		assert tuple(easy_step.observation) == tuple(visual_step.observation)
		assert all(
			np.array_equal(easy_step.observation[key], visual_step.observation[key])
			for key in easy_step.observation
		)

	counting = _CountingDMEnv()
	wrapped = DMControlWrapper(counting, domain='reacher')
	_, reward, done, _ = wrapped.step(np.array([.25, -.5], dtype=np.float64))
	assert counting.step_calls == 2
	assert reward == 2.5
	assert done is False
	assert counting.last_action.dtype == np.float32

	pixels = Pixels(_RenderEnv(), Config())
	reset_rgb = pixels.reset()
	assert reset_rgb.shape == (9, 64, 64)
	assert reset_rgb.dtype == torch.uint8
	assert torch.equal(reset_rgb[:3], reset_rgb[3:6])
	assert torch.equal(reset_rgb[3:6], reset_rgb[6:9])
	step_rgb, _, _, _ = pixels.step(np.zeros(2, dtype=np.float32))
	assert torch.equal(step_rgb[:6], reset_rgb[3:])

	timeout = Timeout(_NeverDoneEnv(), max_episode_steps=500)
	timeout.reset()
	for _ in range(499):
		_, _, done, _ = timeout.step(np.zeros(1, dtype=np.float32))
		assert done is False
	_, _, done, _ = timeout.step(np.zeros(1, dtype=np.float32))
	assert done is True

	# Pin the construction order statically as well as in the runtime chain below.
	source = inspect.getsource(make_dmcontrol_env)
	positions = [
		source.index('env = Pixels('),
		source.index('env = ColorMultiVideoBackgroundWrapper('),
		source.index('env = FlatAnchorWrapper('),
		source.index('env = Timeout('),
	]
	assert positions == sorted(positions), 'Wrapper order is not Pixels->Video->Anchor->Timeout.'
	return {
		'visual_target_radius': task._target_size,
		'reward_target_radius': task._reward_target_radius,
		'action_repeat': counting.step_calls,
		'rgb_shape': tuple(reset_rgb.shape),
		'episode_length': timeout.max_episode_steps,
		'easy_dynamics_parity_steps': 32,
	}


def check_manifests(video_root):
	selector = ColorMultiSplitSelector(video_root)
	assert tuple(selector.splits) == tuple(SPLIT_INDICES)
	assert selector.combined_manifest_sha256 == COMBINED_MANIFEST_SHA256
	all_names = set()
	counts = {}
	for split, indices in SPLIT_INDICES.items():
		path = MANIFEST_DIR / f'color_multi_{split}.json'
		assert _sha256(path) == MANIFEST_SHA256[split]
		payload = json.loads(path.read_text(encoding='utf-8'))
		expected = tuple(f'video{index}.mp4' for index in indices)
		assert set(payload) == {'schema_version', 'name', 'sources'}
		assert payload['schema_version'] == 1
		assert payload['name'] == split
		assert tuple(payload['sources']) == expected
		assert selector.source_names(split) == expected
		assert selector.manifest_sha256(split) == MANIFEST_SHA256[split]
		assert not all_names.intersection(expected)
		all_names.update(expected)
		resolved = selector.resolve(split)
		assert tuple(path.name for path in resolved) == expected
		counts[split] = len(expected)
	assert all(f'video{index}.mp4' not in all_names for index in range(90, 100))
	return counts


def _walk_json_keys(value):
	if isinstance(value, dict):
		for key, child in value.items():
			yield str(key)
			yield from _walk_json_keys(child)
	elif isinstance(value, list):
		for child in value:
			yield from _walk_json_keys(child)


def check_support_pack(support_path):
	support_path = Path(support_path).expanduser().resolve()
	assert support_path.is_file(), support_path
	payload = json.loads(support_path.read_text(encoding='utf-8'))
	assert payload['format'] == 'few_shot_task_anchor_annotations_v1'
	assert tuple(payload['roles']) == ROLES
	collection = payload.get('collection')
	assert isinstance(collection, dict), 'Support pack is missing audited collection metadata.'
	allowed = tuple(f'video{index}.mp4' for index in SPLIT_INDICES['support'])
	assert collection['task'] == 'reacher-visual-small'
	assert collection['observation'] == 'rgb'
	assert collection['split'] == 'support'
	assert tuple(collection['allowed_videos']) == allowed
	assert collection['manifest_sha256'] == MANIFEST_SHA256['support']
	assert collection['combined_manifest_sha256'] == COMBINED_MANIFEST_SHA256
	assert collection['label_policy'] == 'manual_rgb_only'
	assert collection['episodes'] == len(payload['records'])
	assert len(payload['records']) == 6
	assert set(collection['covered_videos']) == set(allowed)

	for key in _walk_json_keys(payload):
		assert key.lower() not in FORBIDDEN_SUPPORT_KEYS, (
			f'Privileged support metadata key is forbidden: {key!r}'
		)

	root = support_path.parent
	seen_images = set()
	seen_hashes = set()
	for expected_index, record in enumerate(payload['records']):
		assert record['index'] == expected_index
		assert record['episode'] == expected_index
		assert record['video_split'] == 'support'
		assert record['active_video'] in allowed
		assert record['source'] == record['active_video']
		assert record['manifest_sha256'] == MANIFEST_SHA256['support']
		assert isinstance(record['frame_index'], int)
		assert record['frame_index'] >= 0
		image_rel = Path(record['image'])
		assert not image_rel.is_absolute() and '..' not in image_rel.parts
		image_path = (root / image_rel).resolve()
		assert root == image_path.parent.parent or root in image_path.parents
		assert image_path.is_file()
		assert str(image_path) not in seen_images
		seen_images.add(str(image_path))
		actual_sha = _raw_image_sha256(image_path)
		assert record['image_sha256'] == actual_sha
		assert actual_sha not in seen_hashes
		seen_hashes.add(actual_sha)
		assert set(record['points']) == set(ROLES)
		for role in ROLES:
			point = record['points'][role]
			assert isinstance(point, list) and len(point) == 2, (
				f'{record["image"]}: {role} must be manually labelled [x, y].'
			)
			assert all(isinstance(value, (int, float)) for value in point)
			assert all(math.isfinite(float(value)) and 0. <= float(value) <= 63. for value in point)

	# The explicit support split is disjoint from the held-out test videos.
	assert set(allowed).isdisjoint(
		f'video{index}.mp4' for index in SPLIT_INDICES['test']
	)
	assert {record['active_video'] for record in payload['records']} == set(allowed)
	return {
		'annotations_sha256': _sha256(support_path),
		'frames': len(payload['records']),
		'videos': sorted({record['active_video'] for record in payload['records']}),
	}


def _function_tree(function):
	return ast.parse(textwrap.dedent(inspect.getsource(function)))


def _assert_no_privileged_names(tree, label):
	used = set()
	for node in ast.walk(tree):
		if isinstance(node, ast.Name):
			used.add(node.id.lower())
		elif isinstance(node, ast.Attribute):
			used.add(node.attr.lower())
	forbidden = sorted(used.intersection(FORBIDDEN_PRIVILEGED_NAMES))
	assert not forbidden, f'{label} uses privileged names: {forbidden}'


def check_no_privileged_leakage():
	augment_tree = _function_tree(FlatAnchorWrapper._augment)
	_assert_no_privileged_names(augment_tree, 'FlatAnchorWrapper._augment')
	extract_calls = [
		node for node in ast.walk(augment_tree)
		if isinstance(node, ast.Call)
		and isinstance(node.func, ast.Attribute)
		and node.func.attr == 'extract'
	]
	assert len(extract_calls) == 1
	call = extract_calls[0]
	assert len(call.args) == 3
	assert isinstance(call.args[0], ast.Name) and call.args[0].id == 'image'
	assert isinstance(call.args[1], ast.Attribute) and call.args[1].attr == '_state'
	assert isinstance(call.args[2], ast.Name) and call.args[2].id == 'first'

	background_tree = ast.parse(inspect.getsource(
		__import__('envs.wrappers.video_background', fromlist=['*'])
	))
	_assert_no_privileged_names(background_tree, 'video_background.py')

	assert COLLECTOR_PATH.is_file(), COLLECTOR_PATH
	collector_source = COLLECTOR_PATH.read_text(encoding='utf-8')
	collector_tree = ast.parse(collector_source)
	_assert_no_privileged_names(collector_tree, COLLECTOR_PATH.name)
	assert 'make_env' in collector_source
	assert '[-3:]' in collector_source

	# Training must never place validation/test/support video frames in replay.
	train_source = TRAIN_PATH.read_text(encoding='utf-8')
	assert "cfg.get('obs', None) != 'rgb'" in train_source
	assert "cfg.get('video_background_split', None) != 'train'" in train_source
	assert 'Use evaluate.py for validation/test.' in train_source
	return {
		'anchor_extract_args': ('image', 'state', 'first'),
		'collector': COLLECTOR_PATH.name,
		'training_split_gate': 'train-only',
	}


def _make_runtime_config(args, reward_graph):
	cfg = _load_config()
	cfg.task = 'reacher-visual-small'
	cfg.obs = 'rgb'
	cfg.seed = args.seed
	cfg.multitask = False
	cfg.video_background_enabled = True
	cfg.video_background_root = str(args.video_root)
	cfg.video_background_split = 'train'
	cfg.video_background_manifest_dir = None
	cfg.video_background_strength = 1.0
	cfg.video_background_total_frames = 1000
	cfg.flat_anchor = reward_graph
	cfg.flat_anchor_mode = 'reward_graph'
	cfg.flat_anchor_support_path = str(args.support) if reward_graph else None
	cfg.flat_anchor_allow_diagnostic_support = False
	if args.anchor_impl is not None:
		cfg.flat_anchor_impl_path = str(args.anchor_impl)
	if args.dino_repo is not None:
		cfg.flat_anchor_dino_repo = str(args.dino_repo)
	if args.dino_checkpoint is not None:
		cfg.flat_anchor_dino_checkpoint = str(args.dino_checkpoint)
	return cfg


def _fixed_actions(count):
	actions = []
	for index in range(count):
		actions.append(torch.tensor([
			math.sin(.71 * (index + 1)),
			math.cos(1.13 * (index + 1)),
		], dtype=torch.float32))
	return actions


def _run_runtime_path(args, reward_graph, actions):
	cfg = _make_runtime_config(args, reward_graph)
	_seed_all(args.seed)
	initial_rng = _capture_rng()
	env = make_env(cfg)
	_assert_rng_equal(initial_rng, _capture_rng(), 'environment construction')
	assert cfg.episode_length == 500
	assert cfg.obs_shape == (
		{'rgb': (9, 64, 64), 'anchor': (25,)}
		if reward_graph else {'rgb': (9, 64, 64)}
	)

	chain = _wrapper_chain(env)
	assert 'TensorWrapper' in chain
	assert 'Timeout' in chain
	assert 'ColorMultiVideoBackgroundWrapper' in chain
	assert 'Pixels' in chain
	if reward_graph:
		assert 'FlatAnchorWrapper' in chain
		assert chain.index('FlatAnchorWrapper') < chain.index(
			'ColorMultiVideoBackgroundWrapper'
		)
	else:
		assert 'FlatAnchorWrapper' not in chain

	observation = env.reset()
	_assert_rng_equal(initial_rng, _capture_rng(), 'environment reset')
	if reward_graph:
		assert set(observation.keys()) == {'rgb', 'anchor'}
		assert tuple(observation['anchor'].shape) == (25,)
		anchors = [_validated_anchor(observation)]
	else:
		anchors = []
	background = _find_background(env)
	assert background.active_split == 'train'
	assert tuple(background.source_names) == tuple(
		f'video{index}.mp4' for index in SPLIT_INDICES['train']
	)
	assert background.active_source.name in background.source_names
	assert background.manifest_sha256 == MANIFEST_SHA256['train']
	assert background.combined_manifest_sha256 == COMBINED_MANIFEST_SHA256

	first_rgb = _rgb(observation, reward_graph)
	assert torch.equal(first_rgb[:3], first_rgb[3:6])
	assert torch.equal(first_rgb[3:6], first_rgb[6:])
	initial_source = background.active_source.name
	initial_frame_index = background.frame_index
	assert isinstance(initial_frame_index, int)
	frame_count = len(background.compositor._frames)
	assert 1 < frame_count <= 1000

	frames = [first_rgb]
	rewards = []
	dones = []
	frame_indices = [initial_frame_index]
	for step_index, action in enumerate(actions):
		previous_rgb = frames[-1]
		previous_index = background.frame_index
		for _ in range(2):
			background.render(width=64, height=64)
			assert background.frame_index == previous_index
		observation, reward, done, _ = env.step(action)
		_assert_rng_equal(initial_rng, _capture_rng(), f'environment step {step_index}')
		current_rgb = _rgb(observation, reward_graph)
		if reward_graph:
			anchors.append(_validated_anchor(observation))
		assert torch.equal(current_rgb[:6], previous_rgb[3:]), (
			f'Background wrapper rewrote causal history at step {step_index}.'
		)
		expected_index = (previous_index + 1) % frame_count
		assert background.frame_index == expected_index
		assert background.active_source.name == initial_source
		frames.append(current_rgb)
		rewards.append(float(torch.as_tensor(reward)))
		dones.append(bool(done))
		frame_indices.append(background.frame_index)

	seed_actions = torch.stack([env.rand_act() for _ in range(16)])
	_assert_rng_equal(initial_rng, _capture_rng(), 'Gym action-space sampling')
	try:
		env.close()
	except (AttributeError, NotImplementedError):
		pass
	return {
		'chain': chain,
		'frames': frames,
		'rewards': rewards,
		'dones': dones,
		'frame_indices': frame_indices,
		'active_source': initial_source,
		'seed_actions': seed_actions,
		'anchors': anchors,
		'final_rng': _capture_rng(),
	}


def check_runtime_parity(args):
	actions = _fixed_actions(args.runtime_steps)
	official = _run_runtime_path(args, False, actions)
	graph = _run_runtime_path(args, True, actions)
	assert official['chain'] != graph['chain']
	assert official['active_source'] == graph['active_source']
	assert official['frame_indices'] == graph['frame_indices']
	assert official['rewards'] == graph['rewards']
	assert official['dones'] == graph['dones']
	assert len(official['frames']) == len(graph['frames'])
	assert all(
		torch.equal(left, right)
		for left, right in zip(official['frames'], graph['frames'])
	), 'Official RGB and RewardGraph received different RGB observations.'
	assert torch.equal(official['seed_actions'], graph['seed_actions'])
	assert _rng_equal(official['final_rng'], graph['final_rng'])
	fallback_fraction = float(torch.stack(graph['anchors'])[:, 24].mean())
	assert fallback_fraction <= args.max_fallback_fraction, (
		f'Anchor fallback_fraction={fallback_fraction:.6f} exceeds '
		f'{args.max_fallback_fraction:.6f}.'
	)
	return {
		'steps': args.runtime_steps,
		'active_source': official['active_source'],
		'frame_indices': official['frame_indices'],
		'rgb_bitwise_equal': True,
		'reward_equal': True,
		'action_space_rng_equal': True,
		'global_rng_equal': True,
		'anchors_checked': len(graph['anchors']),
		'fallback_fraction': fallback_fraction,
		'max_fallback_fraction': args.max_fallback_fraction,
	}


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--video-root', type=Path, required=True)
	parser.add_argument('--support', type=Path, required=True)
	parser.add_argument('--anchor-impl', type=Path)
	parser.add_argument('--dino-repo', type=Path)
	parser.add_argument('--dino-checkpoint', type=Path)
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--runtime-steps', type=int, default=64)
	parser.add_argument('--max-fallback-fraction', type=float, default=.05)
	args = parser.parse_args()
	if args.runtime_steps < 1:
		parser.error('--runtime-steps must be positive')
	if not 0. <= args.max_fallback_fraction <= 1.:
		parser.error('--max-fallback-fraction must lie in [0, 1]')
	return args


def main():
	args = parse_args()
	if torch.cuda.is_available():
		# Initialize CUDA before the paired RNG snapshots. CUDA_VISIBLE_DEVICES=1
		# still appears as logical cuda:0, which is the intended server contract.
		torch.cuda.init()
	torch.backends.cudnn.benchmark = False
	result = {
		'fixed_environment': check_fixed_task_and_wrappers(),
		'manifest_counts': check_manifests(args.video_root),
		'manifest_sha256': dict(MANIFEST_SHA256),
		'combined_manifest_sha256': COMBINED_MANIFEST_SHA256,
		'support': check_support_pack(args.support),
		'no_privileged_leakage': check_no_privileged_leakage(),
		'runtime_parity': check_runtime_parity(args),
	}
	print('VISUAL_SMALL_COLOR_MULTI_CONTRACT_OK', json.dumps(result, sort_keys=True))


if __name__ == '__main__':
	main()
