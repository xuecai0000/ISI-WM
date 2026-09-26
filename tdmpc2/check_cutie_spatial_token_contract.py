"""CPU contract for identity-invariant spatial Cutie object control."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from common import layers
from envs.wrappers.cutie_object import (
	FRAME_CONTENT_DIM,
	FRAME_FEATURE_DIM,
	MASK_POOL_SIZE,
	QUERY_POOL_DIM,
	_whole_acrobot_spatial_frame,
)


class Config(SimpleNamespace):
	def get(self, key, default=None):
		return getattr(self, key, default)


def _config():
	return Config(
		multitask=False,
		action_dim=1,
		simnorm_dim=8,
		flat_anchor_mode='cutie_object_only',
		cutie_object_num_roles=2,
		cutie_object_input_dim=1770,
		cutie_object_observation_variant='full',
		cutie_object_role_dim=64,
		cutie_object_hidden_dim=256,
		cutie_object_spatial_token_enabled=True,
		cutie_object_spatial_graph_path=str(
			Path(__file__).resolve().parent / 'object_graphs' / 'acrobot_swingup.json'
		),
		cutie_object_spatial_token_dim=64,
		cutie_object_spatial_num_heads=4,
		cutie_object_spatial_num_layers=2,
	)


def _source_frame():
	rng = np.random.default_rng(271828)
	value = rng.normal(size=(2, FRAME_FEATURE_DIM)).astype(np.float32)
	value[:, QUERY_POOL_DIM:QUERY_POOL_DIM + 64] = rng.uniform(
		0.0, 0.6, size=(2, 64)
	)
	value[:, FRAME_CONTENT_DIM:] = np.asarray([
		[0.8, 0.0, 1.0, 0.7],
		[0.9, 0.0, 1.0, 0.6],
	], dtype=np.float32)
	return value


def main():
	source = _source_frame()
	whole = _whole_acrobot_spatial_frame(source)
	swapped = _whole_acrobot_spatial_frame(source[::-1].copy())
	if whole.shape != (2, FRAME_FEATURE_DIM):
		raise AssertionError(whole.shape)
	if not np.array_equal(whole[0], whole[1]):
		raise AssertionError('Both latent structural slots must see one whole object.')
	if not np.array_equal(whole, swapped):
		raise AssertionError('Whole-object observation changed after a role swap.')
	expected_union = np.clip(
		source[:, QUERY_POOL_DIM:QUERY_POOL_DIM + MASK_POOL_SIZE ** 2].sum(axis=0),
		0.0, 1.0,
	)
	if not np.allclose(
		whole[0, QUERY_POOL_DIM:QUERY_POOL_DIM + MASK_POOL_SIZE ** 2],
		expected_union,
	):
		raise AssertionError('Whole-object occupancy is not the symmetric union.')

	cfg = _config()
	encoder = layers.CutieObjectEncoder(cfg)
	dynamics = layers.CutieObjectDynamics(cfg)
	objects = torch.randn(4, 2, 1770, requires_grad=True)
	z = encoder(objects)
	if z.shape != (4, 128) or not torch.isfinite(z).all():
		raise AssertionError(f'Invalid spatial latent {tuple(z.shape)}.')
	next_z = dynamics(z, torch.randn(4, 1))
	if next_z.shape != (4, 128) or not torch.isfinite(next_z).all():
		raise AssertionError(f'Invalid next latent {tuple(next_z.shape)}.')
	next_z.square().mean().backward()
	if objects.grad is None or not torch.isfinite(objects.grad).all():
		raise AssertionError('Spatial encoder did not propagate a finite gradient.')
	print('CUTIE_SPATIAL_TOKEN_CONTRACT_OK')


if __name__ == '__main__':
	main()
