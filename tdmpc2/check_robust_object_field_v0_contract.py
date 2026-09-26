"""Executable contracts for the minimal ROF-WM V0 model.

This test is intentionally independent of dm_control and Cutie weights.  It
proves the tensor/domain/gradient contract before an expensive perception or
control run is allowed to start.  Run it in the normal TD-MPC2 environment.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch


TD_ROOT = Path(__file__).resolve().parent
if str(TD_ROOT) not in sys.path:
	sys.path.insert(0, str(TD_ROOT))

from common import robust_object_field as rof  # noqa: E402


class Config(dict):
	"""Small dict/attribute config matching OmegaConf's used interface."""

	def __getattr__(self, name):
		try:
			return self[name]
		except KeyError as exc:
			raise AttributeError(name) from exc

	def __setattr__(self, name, value):
		self[name] = value


def config(roles=2, **updates):
	role_names = tuple(f'role_{index}' for index in range(roles))
	value = Config({
		'robust_object_field_enabled': True,
		'robust_object_field_schema': rof.SCHEMA,
		'robust_object_field_local_tokens': 4,
		'robust_object_field_token_dim': 64,
		'robust_object_field_context_radius': 3,
		'robust_object_field_query_mode': 'ordered_compressed',
		'robust_object_field_random_shift_pad': 0,
		'robust_object_field_dynamics_hidden_dim': 128,
		'robust_object_field_decoder_hidden_dim': 128,
		'flat_anchor': True,
		'flat_anchor_mode': 'cutie_object_only',
		'obs': 'rgb',
		'multitask': False,
		'task': 'contract-task',
		'cutie_object_observation_variant': 'full',
		'cutie_object_frame_schema': 'cutie_query_mask_status_v1',
		'cutie_object_frame_dim': 590,
		'cutie_object_stack_frames': 3,
		'cutie_object_input_dim': 1770,
		'cutie_object_num_roles': roles,
		'cutie_object_role_names': role_names,
		'cutie_object_role_dim': 64,
		'cutie_object_only_latent_dim': roles * 5 * 64,
		'latent_dim': roles * 5 * 64,
		'action_dim': 3,
		'simnorm_dim': 8,
		'cutie_masked_rgb_enabled': False,
		'cutie_object_allow_simulator_runtime': False,
		'cutie_object_allow_simulator_kinematics_runtime': False,
		'cutie_object_spatial_token_enabled': False,
		'cutie_object_variable_graph_enabled': False,
		'cutie_object_belief_enabled': False,
		'cutie_object_last_valid_memory': False,
		'cutie_object_policy_burst_plan': None,
		'cutie_object_regression_encoder': None,
		'object_state_supervision_enabled': False,
		'object_state_supervision_collect_labels': False,
		'object_state_bottleneck_enabled': False,
	})
	value['obs_shape'] = rof.observation_shapes(value)
	value.update(updates)
	return value


def observation(cfg, batch=2):
	roles = cfg.cutie_object_num_roles
	rgb = torch.zeros(batch, 9, 64, 64, dtype=torch.float32)
	masks = torch.zeros(batch, roles, 3, 64, 64, dtype=torch.bool)
	for role in range(roles):
		x0, y0 = 5 + 11 * role, 8 + 7 * role
		masks[:, role, :, y0:y0 + 12, x0:x0 + 10] = True
		rgb[:, :, y0:y0 + 12, x0:x0 + 10] = 31.0 + 40.0 * role
	rgb = rgb.to(torch.uint8)
	objects = torch.randn(batch, roles, 1770)
	# The latest 4-D status is finite and bounded like the real wrapper.
	objects[..., 2 * 590 + 586:2 * 590 + 590] = torch.tensor(
		[0.8, 0.0, 1.0, 0.7]
	)
	return {
		'rgb': rgb,
		'object': objects,
		'object_mask': masks,
		'role_exists': torch.ones(batch, roles, dtype=torch.float32),
	}


class RobustObjectFieldContracts(unittest.TestCase):
	def assert_simnorm(self, value, cfg):
		minimum, max_error = rof.simnorm_max_error(value, cfg.simnorm_dim)
		self.assertGreaterEqual(minimum, -1e-7)
		self.assertLessEqual(max_error, 1e-5)

	def test_exact_k_shapes_domain_and_decoder_for_one_two_three_roles(self):
		for roles in (1, 2, 3):
			with self.subTest(roles=roles):
				cfg = config(roles)
				rof.validate_config(cfg)
				encoder = rof.RobustObjectFieldEncoder(cfg)
				latent, tokens = encoder(
					observation(cfg), return_tokens=True
				)
				self.assertEqual(tuple(tokens.shape), (2, roles, 5, 64))
				self.assertEqual(tuple(latent.shape), (2, roles * 5 * 64))
				self.assert_simnorm(tokens, cfg)
				dynamics = rof.RobustObjectFieldDynamics(cfg)
				next_latent = dynamics(latent, torch.zeros(2, cfg.action_dim))
				self.assertEqual(next_latent.shape, latent.shape)
				self.assert_simnorm(next_latent, cfg)
				decoded = rof.RobustObjectFieldDecoder(cfg)(next_latent)
				self.assertEqual(tuple(decoded.shape), (2, roles, 1770))

	def test_encoder_and_dynamics_receive_finite_nonzero_gradients(self):
		cfg = config(2)
		obs = observation(cfg)
		encoder = rof.RobustObjectFieldEncoder(cfg)
		dynamics = rof.RobustObjectFieldDynamics(cfg)
		decoder = rof.RobustObjectFieldDecoder(cfg)
		latent = encoder(obs)
		next_latent = dynamics(latent, torch.randn(2, cfg.action_dim))
		loss = decoder(next_latent).square().mean()
		loss.backward()
		for name, module in (
			('encoder', encoder), ('dynamics', dynamics), ('decoder', decoder),
		):
			gradients = [
				parameter.grad for parameter in module.parameters()
				if parameter.requires_grad
			]
			self.assertTrue(gradients, name)
			self.assertTrue(all(value is not None for value in gradients), name)
			self.assertTrue(all(torch.isfinite(value).all() for value in gradients), name)
			self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0.0)

	def test_spatial_destruction_changes_local_field_tokens(self):
		cfg = config(2)
		torch.manual_seed(7)
		encoder = rof.RobustObjectFieldEncoder(cfg).eval()
		obs = observation(cfg)
		_, original = encoder(obs, return_tokens=True)
		permuted = {key: value.clone() for key, value in obs.items()}
		permuted['object_mask'] = torch.flip(
			permuted['object_mask'], dims=(-2, -1)
		)
		_, destroyed = encoder(permuted, return_tokens=True)
		self.assertFalse(torch.allclose(original[..., :4, :], destroyed[..., :4, :]))

	def test_opt_in_temporal_role_geometry_preserves_ordered_motion(self):
		masks = torch.zeros(1, 1, 3, 64, 64, dtype=torch.float32)
		masks[:, :, 0, 16:24, 8:16] = 1.0
		masks[:, :, 1, 16:24, 16:24] = 1.0
		masks[:, :, 2, 16:24, 24:32] = 1.0
		mask8, centroids, spreads, areas, signal = rof._temporal_role_geometry(masks)
		self.assertEqual(tuple(mask8.shape), (1, 1, 3, 8, 8))
		self.assertEqual(tuple(centroids.shape), (1, 1, 3, 2))
		self.assertEqual(tuple(spreads.shape), (1, 1, 3, 2))
		self.assertEqual(tuple(areas.shape), (1, 1, 3))
		self.assertEqual(tuple(signal.shape), (1, 1, 6))
		self.assertGreater(float(signal[0, 0, 0]), 0.0)
		self.assertEqual(float(signal[0, 0, 1]), 0.0)
		self.assertGreater(float(signal[0, 0, 2]), 0.0)
		self.assertEqual(float(signal[0, 0, 3]), 0.0)
		torch.testing.assert_close(signal[..., 4:], torch.zeros_like(signal[..., 4:]))
		self.assertGreaterEqual(float(signal.min()), -1.0)
		self.assertLessEqual(float(signal.max()), 1.0)

		# Reversing history while keeping the current frame fixed changes the
		# ordered displacement signal; this is the information V0 did not expose
		# explicitly to each local role token.
		reversed_history = masks.clone()
		reversed_history[:, :, 0] = masks[:, :, 1]
		reversed_history[:, :, 1] = masks[:, :, 0]
		_, _, _, _, reversed_signal = rof._temporal_role_geometry(reversed_history)
		self.assertFalse(torch.allclose(signal, reversed_signal))

	def test_temporal_role_geometry_encoder_is_opt_in_and_shape_stable(self):
		v0_cfg = config(2)
		v1_cfg = config(2, robust_object_field_temporal_role_geometry=True)
		v0_encoder = rof.RobustObjectFieldEncoder(v0_cfg)
		v1_encoder = rof.RobustObjectFieldEncoder(v1_cfg)
		self.assertEqual(v0_encoder.local_projection[0].in_features, 75)
		self.assertEqual(v1_encoder.local_projection[0].in_features, 211)
		self.assertNotIn('temporal_role_geometry', rof.contract(v0_cfg))
		self.assertTrue(rof.contract(v1_cfg)['temporal_role_geometry'])
		latent, tokens = v1_encoder(observation(v1_cfg), return_tokens=True)
		self.assertEqual(tuple(tokens.shape), (2, 2, 5, 64))
		self.assertEqual(tuple(latent.shape), (2, 2 * 5 * 64))
		self.assert_simnorm(tokens, v1_cfg)

	def test_missing_temporal_flag_is_identical_to_explicit_v0_false(self):
		implicit_cfg = config(2)
		explicit_cfg = config(2, robust_object_field_temporal_role_geometry=False)
		torch.manual_seed(20260913)
		implicit = rof.RobustObjectFieldEncoder(implicit_cfg).eval()
		torch.manual_seed(20260913)
		explicit = rof.RobustObjectFieldEncoder(explicit_cfg).eval()
		for name, value in implicit.state_dict().items():
			torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
		obs = observation(implicit_cfg)
		torch.testing.assert_close(implicit(obs), explicit(obs), rtol=0, atol=0)

	def test_temporal_role_geometry_fails_closed_on_shape_domain_and_flag(self):
		with self.assertRaisesRegex(ValueError, r'\[B,K,3,64,64\]'):
			rof._temporal_role_geometry(torch.zeros(1, 1, 2, 64, 64))
		with self.assertRaisesRegex(ValueError, 'floating masks'):
			rof._temporal_role_geometry(
				torch.zeros(1, 1, 3, 64, 64, dtype=torch.bool)
			)
		bad_domain = torch.zeros(1, 1, 3, 64, 64)
		bad_domain[..., 0, 0] = 1.1
		with self.assertRaisesRegex(ValueError, r'\[0,1\]'):
			rof._temporal_role_geometry(bad_domain)
		bad_flag = config(2, robust_object_field_temporal_role_geometry='true')
		with self.assertRaisesRegex(ValueError, 'strict boolean'):
			rof.validate_config(bad_flag)

	def test_packed_shape_round_trip(self):
		shape = rof.PackedFieldShape(3, 5, 64)
		tokens = torch.randn(2, 3, 5, 64)
		torch.testing.assert_close(shape.unpack(shape.pack(tokens)), tokens)

	def test_joint_shift_keeps_rgb_and_masks_aligned(self):
		augment = rof.JointFieldShiftAug(pad=3)
		rgb = torch.zeros(1, 9, 64, 64)
		masks = torch.zeros(1, 2, 3, 64, 64)
		rgb[:, :, 20:30, 12:22] = 1.0
		masks[:, 0, :, 20:30, 12:22] = 1.0
		shift = torch.tensor([[[[5, 1]]]])
		shifted_rgb, shifted_masks = augment(rgb, masks, shift_index=shift)
		self.assertEqual(tuple(shifted_rgb.shape), tuple(rgb.shape))
		self.assertEqual(tuple(shifted_masks.shape), tuple(masks.shape))
		rgb_support = shifted_rgb[:, 0] > 0.5
		mask_support = shifted_masks[:, 0, 0] > 0.5
		torch.testing.assert_close(rgb_support, mask_support)

	def test_fail_closed_for_privilege_padding_and_wrong_width(self):
		for update in (
			{'cutie_object_allow_simulator_runtime': True},
			{'cutie_object_allow_simulator_kinematics_runtime': True},
			{'object_state_supervision_enabled': True},
			{'multitask': True},
			{'cutie_object_belief_enabled': True},
			{'cutie_object_only_latent_dim': 128},
		):
			bad = config(2, **update)
			with self.subTest(update=update), self.assertRaises(ValueError):
				rof.validate_config(bad)
		cfg = config(2)
		encoder = rof.RobustObjectFieldEncoder(cfg)
		obs = observation(cfg)
		obs['role_exists'][0, 1] = 0
		with self.assertRaisesRegex(ValueError, 'padding is forbidden'):
			encoder(obs)
		obs = observation(cfg)
		obs['object_mask'] = obs['object_mask'].to(torch.uint8)
		with self.assertRaisesRegex(ValueError, 'must be boolean'):
			encoder(obs)


if __name__ == '__main__':
	unittest.main(verbosity=2)
