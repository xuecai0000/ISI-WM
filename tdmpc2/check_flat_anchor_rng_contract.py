"""Verify that FlatAnchor preserves official TD-MPC2 initialization and RNG.

This test does not load DINO or create an environment. It checks that, for a
matched seed, every official WorldModel tensor and the post-construction RNG
state are identical with FlatAnchor enabled, while the zero-initialized fusion
produces the exact official RGB latent.
"""

import random
from types import SimpleNamespace

import numpy as np
import torch

from common.world_model import WorldModel
from envs.wrappers.flat_anchor import _preserve_random_state


class Config(SimpleNamespace):
	def get(self, key, default=None):
		return getattr(self, key, default)


def make_config(flat_anchor):
	return Config(
		multitask=False,
		obs='rgb',
		obs_shape={
			'rgb': (9, 64, 64),
			**({'anchor': (25,)} if flat_anchor else {}),
		},
		action_dim=2,
		task_dim=0,
		num_channels=32,
		latent_dim=512,
		mlp_dim=512,
		num_bins=101,
		num_q=5,
		dropout=0.01,
		episodic=False,
		log_std_min=-10,
		log_std_max=2,
		simnorm_dim=8,
		flat_anchor=flat_anchor,
		flat_anchor_dim=25,
		flat_anchor_hidden_dim=128,
		flat_anchor_embed_dim=64,
		flat_anchor_fusion_dim=512,
	)


def numpy_state_equal(left, right):
	return (
		left[0] == right[0]
		and np.array_equal(left[1], right[1])
		and left[2:] == right[2:]
	)


def seed_all(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)


def capture_rng_state():
	return {
		'python': random.getstate(),
		'numpy': np.random.get_state(),
		'torch': torch.get_rng_state().clone(),
		'cuda': [state.clone() for state in torch.cuda.get_rng_state_all()],
	}


def assert_rng_state_equal(left, right, label):
	assert left['python'] == right['python'], f'{label}: Python RNG differs.'
	assert numpy_state_equal(left['numpy'], right['numpy']), f'{label}: NumPy RNG differs.'
	assert torch.equal(left['torch'], right['torch']), f'{label}: Torch CPU RNG differs.'
	assert len(left['cuda']) == len(right['cuda'])
	assert all(torch.equal(a, b) for a, b in zip(left['cuda'], right['cuda'])), \
		f'{label}: Torch CUDA RNG differs.'


def state_value_equal(left, right):
	"""Compare tensor state entries and TensorDict metadata safely."""
	if torch.is_tensor(left) or torch.is_tensor(right):
		return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
	if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
		return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) \
			and np.array_equal(left, right)
	return type(left) is type(right) and left == right


def check_rng_context():
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	random.seed(271828)
	np.random.seed(271828)
	torch.manual_seed(271828)
	python_before = random.getstate()
	numpy_before = np.random.get_state()
	torch_before = torch.get_rng_state().clone()
	cuda_before = [state.clone() for state in torch.cuda.get_rng_state_all()]

	with _preserve_random_state(device):
		random.random()
		np.random.random()
		torch.rand(8)
		if device.type == 'cuda':
			for index in range(torch.cuda.device_count()):
				torch.rand(8, device=f'cuda:{index}')

	assert random.getstate() == python_before, 'Python RNG was not restored.'
	assert numpy_state_equal(np.random.get_state(), numpy_before), 'NumPy RNG was not restored.'
	assert torch.equal(torch.get_rng_state(), torch_before), 'Torch CPU RNG was not restored.'
	cuda_after = torch.cuda.get_rng_state_all()
	assert len(cuda_after) == len(cuda_before)
	assert all(torch.equal(a, b) for a, b in zip(cuda_after, cuda_before)), \
		'Torch CUDA RNG was not restored.'
	return device.type == 'cuda'


def check_model_contract():
	seed = 314159
	seed_all(seed)
	official = WorldModel(make_config(False))
	official_rng = capture_rng_state()
	official_state = official.state_dict()

	seed_all(seed)
	anchored = WorldModel(make_config(True))
	anchored_rng = capture_rng_state()
	anchored_state = anchored.state_dict()

	assert_rng_state_equal(
		official_rng,
		anchored_rng,
		'FlatAnchor vs official model construction',
	)
	for key, value in official_state.items():
		assert key in anchored_state, f'Missing official parameter/buffer: {key}'
		assert state_value_equal(value, anchored_state[key]), \
			f'Official initialization changed for: {key}'
	extra = set(anchored_state) - set(official_state)
	assert extra
	assert all(
		key.startswith('_encoder.anchor.') or key.startswith('_encoder.fusion.')
		for key in extra
	), f'Unexpected FlatAnchor state entries: {sorted(extra)}'

	official.eval()
	anchored.eval()
	obs = torch.randint(0, 256, (1, 9, 64, 64), dtype=torch.uint8)
	anchor = torch.randn(1, 25)
	torch.manual_seed(161803)
	official_latent = official.encode(obs, task=None)
	official_encode_rng = torch.get_rng_state().clone()
	torch.manual_seed(161803)
	anchored_latent = anchored.encode({'rgb': obs, 'anchor': anchor}, task=None)
	anchored_encode_rng = torch.get_rng_state().clone()
	assert torch.equal(official_encode_rng, anchored_encode_rng), \
		'FlatAnchor encoding advanced RNG beyond the official RGB encoder.'
	assert torch.equal(official_latent, anchored_latent), \
		'Zero-initialized FlatAnchor is not exactly the official RGB latent.'

	return {
		'shared_tensors': len(official_state),
		'extra_tensors': len(extra),
		'identity_error': float((official_latent - anchored_latent).abs().max()),
	}


if __name__ == '__main__':
	cuda_checked = check_rng_context()
	result = check_model_contract()
	result['cuda_checked'] = cuda_checked
	print('FLAT_ANCHOR_RNG_CONTRACT_OK', result)
