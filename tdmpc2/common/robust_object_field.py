"""ROF-WM V0: causal, part-free spatial fields inside tracked objects.

V0 is deliberately small and falsifiable. It consumes the same causal RGB,
Cutie descriptors, and native masks as the existing baselines, preserves four
coordinate-anchored local measurements plus one ordered-query identity token
per real role, and emits the exact SimNorm domain used by its dynamics.

There is no recurrent observer, padding role, task-specific part parser,
Transformer, or privileged simulator input in this module. Those exclusions
are part of the versioned scientific contract, not temporary implementation
details.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEMA = 'robust_object_field_v0'
IMAGE_SIZE = 64
STACK_FRAMES = 3
RGB_CHANNELS = 3 * STACK_FRAMES
FRAME_DIM = 590
OBJECT_DIM = FRAME_DIM * STACK_FRAMES
QUERY_DIM = 512
STATUS_START = 586
STATUS_DIM = 4
DEFAULT_LOCAL_TOKENS = 4
DEFAULT_TOKEN_DIM = 64
TEMPORAL_ROLE_GEOMETRY_FLAG = 'robust_object_field_temporal_role_geometry'
TEMPORAL_ROLE_SIGNAL_DIM = 6


def _get(cfg, key, default=None):
	return cfg.get(key, default) if hasattr(cfg, 'get') else getattr(cfg, key, default)


def enabled(cfg) -> bool:
	return bool(_get(cfg, 'robust_object_field_enabled', False))


def num_tokens(cfg) -> int:
	return int(_get(cfg, 'robust_object_field_local_tokens', DEFAULT_LOCAL_TOKENS)) + 1


def latent_dim(cfg) -> int:
	return (
		int(_get(cfg, 'cutie_object_num_roles', 2))
		* num_tokens(cfg)
		* int(_get(cfg, 'robust_object_field_token_dim', DEFAULT_TOKEN_DIM))
	)


def observation_shapes(cfg) -> dict[str, tuple[int, ...]]:
	roles = int(_get(cfg, 'cutie_object_num_roles', 2))
	return {
		'rgb': (RGB_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
		'object': (roles, OBJECT_DIM),
		'object_mask': (roles, STACK_FRAMES, IMAGE_SIZE, IMAGE_SIZE),
		'role_exists': (roles,),
	}


def contract(cfg) -> dict:
	payload = {
		'format': 'robust_object_field_contract_v0',
		'schema': str(_get(cfg, 'robust_object_field_schema', SCHEMA)),
		'task': str(_get(cfg, 'task', '')),
		'ordered_roles': list(_get(cfg, 'cutie_object_role_names', ())),
		'num_roles': int(_get(cfg, 'cutie_object_num_roles', 2)),
		'single_task_exact_k': True,
		'padding_roles': False,
		'simulator_runtime_inputs': False,
		'stack_frames': STACK_FRAMES,
		'image_size': IMAGE_SIZE,
		'local_tokens': int(_get(
			cfg, 'robust_object_field_local_tokens', DEFAULT_LOCAL_TOKENS
		)),
		'global_tokens': 1,
		'token_dim': int(_get(
			cfg, 'robust_object_field_token_dim', DEFAULT_TOKEN_DIM
		)),
		'latent_dim': latent_dim(cfg),
		'simnorm_dim': int(_get(cfg, 'simnorm_dim', 8)),
		'context_radius': int(_get(cfg, 'robust_object_field_context_radius', 3)),
		'query_mode': str(_get(
			cfg, 'robust_object_field_query_mode', 'ordered_compressed'
		)),
		'random_shift_pad': int(_get(
			cfg, 'robust_object_field_random_shift_pad', 3
		)),
	}
	# Keep the serialized V0 contract byte-for-byte compatible when the
	# opt-in trajectory path is disabled.  Enabled checkpoints must record the
	# representation change so they cannot silently load as V0.
	if bool(_get(cfg, TEMPORAL_ROLE_GEOMETRY_FLAG, False)):
		payload['temporal_role_geometry'] = True
	return payload


def validate_config(cfg, *, require_obs_shape=True) -> None:
	"""Fail closed on every V0 scientific and tensor-shape assumption."""
	if not enabled(cfg):
		raise ValueError('ROF-WM validation requires robust_object_field_enabled=true.')
	expected = {
		'flat_anchor': True,
		'flat_anchor_mode': 'cutie_object_only',
		'obs': 'rgb',
		'multitask': False,
		'cutie_object_observation_variant': 'full',
		'cutie_object_frame_schema': 'cutie_query_mask_status_v1',
		'cutie_object_frame_dim': FRAME_DIM,
		'cutie_object_stack_frames': STACK_FRAMES,
		'cutie_object_input_dim': OBJECT_DIM,
		'cutie_masked_rgb_enabled': False,
		'cutie_object_allow_simulator_runtime': False,
		'cutie_object_allow_simulator_kinematics_runtime': False,
		'cutie_object_spatial_token_enabled': False,
		'cutie_object_variable_graph_enabled': False,
		'cutie_object_belief_enabled': False,
		'cutie_object_last_valid_memory': False,
		'object_state_supervision_enabled': False,
		'object_state_supervision_collect_labels': False,
		'object_state_bottleneck_enabled': False,
	}
	bad = {
		key: (_get(cfg, key, None), value)
		for key, value in expected.items()
		if _get(cfg, key, None) != value
	}
	if str(_get(cfg, 'robust_object_field_schema', SCHEMA)) != SCHEMA:
		bad['robust_object_field_schema'] = (
			_get(cfg, 'robust_object_field_schema', None), SCHEMA
		)
	roles = int(_get(cfg, 'cutie_object_num_roles', 0))
	if roles not in {1, 2, 3}:
		bad['cutie_object_num_roles'] = (roles, 'one of {1,2,3}')
	role_names = tuple(_get(cfg, 'cutie_object_role_names', ()))
	if len(role_names) != roles or len(set(role_names)) != roles:
		bad['cutie_object_role_names'] = (role_names, f'{roles} unique ordered roles')
	local_tokens = int(_get(
		cfg, 'robust_object_field_local_tokens', DEFAULT_LOCAL_TOKENS
	))
	if local_tokens != DEFAULT_LOCAL_TOKENS:
		bad['robust_object_field_local_tokens'] = (
			local_tokens, DEFAULT_LOCAL_TOKENS
		)
	token_dim = int(_get(cfg, 'robust_object_field_token_dim', DEFAULT_TOKEN_DIM))
	simnorm_dim = int(_get(cfg, 'simnorm_dim', 8))
	if token_dim != DEFAULT_TOKEN_DIM or simnorm_dim <= 0 or token_dim % simnorm_dim:
		bad['token_domain'] = (
			(token_dim, simnorm_dim),
			f'token_dim={DEFAULT_TOKEN_DIM} divisible by positive simnorm_dim',
		)
	context_radius = int(_get(cfg, 'robust_object_field_context_radius', 3))
	if not 1 <= context_radius <= 8:
		bad['robust_object_field_context_radius'] = (context_radius, '[1,8]')
	shift_pad = int(_get(cfg, 'robust_object_field_random_shift_pad', 3))
	if not 0 <= shift_pad <= 8:
		bad['robust_object_field_random_shift_pad'] = (shift_pad, '[0,8]')
	if str(_get(
		cfg, 'robust_object_field_query_mode', 'ordered_compressed'
	)) != 'ordered_compressed':
		bad['robust_object_field_query_mode'] = (
			_get(cfg, 'robust_object_field_query_mode', None),
			'ordered_compressed',
		)
	temporal_role_geometry = _get(cfg, TEMPORAL_ROLE_GEOMETRY_FLAG, False)
	if type(temporal_role_geometry) is not bool:
		bad[TEMPORAL_ROLE_GEOMETRY_FLAG] = (
			temporal_role_geometry, 'a strict boolean',
		)
	if int(_get(cfg, 'cutie_object_only_latent_dim', -1)) != latent_dim(cfg):
		bad['cutie_object_only_latent_dim'] = (
			_get(cfg, 'cutie_object_only_latent_dim', None), latent_dim(cfg)
		)
	if int(_get(cfg, 'latent_dim', -1)) != latent_dim(cfg):
		bad['latent_dim'] = (_get(cfg, 'latent_dim', None), latent_dim(cfg))
	if _get(cfg, 'cutie_object_policy_burst_plan', None) is not None:
		bad['cutie_object_policy_burst_plan'] = (
			_get(cfg, 'cutie_object_policy_burst_plan'), None
		)
	if _get(cfg, 'cutie_object_regression_encoder', None) is not None:
		bad['cutie_object_regression_encoder'] = (
			_get(cfg, 'cutie_object_regression_encoder'), None
		)
	if require_obs_shape:
		actual_shapes = {
			key: tuple(value) for key, value in dict(_get(cfg, 'obs_shape', {})).items()
		}
		if actual_shapes != observation_shapes(cfg):
			bad['obs_shape'] = (actual_shapes, observation_shapes(cfg))
	if bad:
		raise ValueError(f'ROF-WM V0 contract mismatch: {bad}.')


def _simnorm(x: torch.Tensor, group_dim: int) -> torch.Tensor:
	if x.shape[-1] % group_dim:
		raise ValueError('ROF token width must be divisible by SimNorm group width.')
	shape = x.shape
	return F.softmax(x.reshape(*shape[:-1], -1, group_dim), dim=-1).reshape(shape)


def simnorm_max_error(x: torch.Tensor, group_dim: int) -> tuple[float, float]:
	"""Return minimum value and maximum group-sum error for contract tests."""
	groups = x.reshape(*x.shape[:-1], -1, group_dim)
	return float(x.min().detach()), float((groups.sum(-1) - 1.0).abs().max().detach())


def _temporal_role_geometry(masks: torch.Tensor, field_size=8):
	"""Return ordered per-frame role geometry and a bounded motion signal.

	The six signal values are two consecutive normalized centroid
	displacements ``(dx, dy) / 2`` followed by the corresponding normalized
	area changes.  Coordinates and areas therefore remain in ``[-1, 1]``.
	Empty masks are represented by a zero centroid/area; the existing Cutie
	status fields retain responsibility for distinguishing absence from a
	physical jump.
	"""
	if masks.ndim != 5 or tuple(masks.shape[2:]) != (
		STACK_FRAMES, IMAGE_SIZE, IMAGE_SIZE
	):
		raise ValueError(
			'ROF temporal role geometry expects masks [B,K,3,64,64].'
		)
	if not masks.is_floating_point():
		raise ValueError('ROF temporal role geometry expects floating masks.')
	if not isinstance(field_size, int) or not 2 <= field_size <= IMAGE_SIZE:
		raise ValueError('ROF temporal role field size must be an integer in [2,64].')
	compiling = torch.compiler.is_compiling()
	if not compiling:
		if not bool(torch.isfinite(masks).all().item()):
			raise ValueError('ROF temporal role masks must be finite.')
		if not bool(((masks >= 0.0) & (masks <= 1.0)).all().item()):
			raise ValueError('ROF temporal role masks must lie in [0,1].')

	batch, roles = masks.shape[:2]
	mask_fields = F.adaptive_avg_pool2d(
		masks.reshape(batch * roles * STACK_FRAMES, 1, IMAGE_SIZE, IMAGE_SIZE),
		(field_size, field_size),
	).reshape(batch, roles, STACK_FRAMES, field_size, field_size)
	axis = torch.linspace(
		-1.0, 1.0, field_size, device=masks.device, dtype=masks.dtype
	)
	yy, xx = torch.meshgrid(axis, axis, indexing='ij')
	coords = torch.stack([xx, yy], dim=-1)
	mass = mask_fields.sum(dim=(-2, -1))
	centroids = (
		mask_fields.unsqueeze(-1)
		* coords.view(1, 1, 1, field_size, field_size, 2)
	).sum(dim=(-3, -2)) / mass.clamp_min(1e-6).unsqueeze(-1)
	centroids = torch.where(
		(mass > 1e-6).unsqueeze(-1), centroids, torch.zeros_like(centroids)
	)
	centered = (
		coords.view(1, 1, 1, field_size, field_size, 2)
		- centroids.unsqueeze(-2).unsqueeze(-2)
	)
	variance = (
		mask_fields.unsqueeze(-1) * centered.square()
	).sum(dim=(-3, -2)) / mass.clamp_min(1e-6).unsqueeze(-1)
	spreads = variance.clamp_min(0.02 ** 2).sqrt()
	areas = mask_fields.mean(dim=(-2, -1))
	centroid_delta = (centroids[:, :, 1:] - centroids[:, :, :-1]) / 2.0
	area_delta = areas[:, :, 1:] - areas[:, :, :-1]
	signal = torch.cat([
		centroid_delta.reshape(batch, roles, -1), area_delta,
	], dim=-1)
	expected = {
		'masks': (batch, roles, STACK_FRAMES, field_size, field_size),
		'centroids': (batch, roles, STACK_FRAMES, 2),
		'spreads': (batch, roles, STACK_FRAMES, 2),
		'areas': (batch, roles, STACK_FRAMES),
		'signal': (batch, roles, TEMPORAL_ROLE_SIGNAL_DIM),
	}
	actual = {
		'masks': tuple(mask_fields.shape), 'centroids': tuple(centroids.shape),
		'spreads': tuple(spreads.shape), 'areas': tuple(areas.shape),
		'signal': tuple(signal.shape),
	}
	if actual != expected:
		raise AssertionError(f'ROF temporal geometry shape mismatch: {actual} != {expected}.')
	if not compiling:
		if not bool(torch.isfinite(signal).all().item()):
			raise FloatingPointError('ROF temporal role signal is non-finite.')
		if not bool(((signal >= -1.0) & (signal <= 1.0)).all().item()):
			raise FloatingPointError('ROF temporal role signal left [-1,1].')
	return mask_fields, centroids, spreads, areas, signal


class JointFieldShiftAug(nn.Module):
	"""Apply one TD-MPC2-style integer shift to RGB and every role mask."""

	def __init__(self, pad=3):
		super().__init__()
		self.pad = int(pad)

	def forward(self, rgb, masks, shift_index=None):
		if rgb.ndim != 4 or masks.ndim != 5:
			raise ValueError('Joint field shift expects RGB [B,9,H,W], masks [B,K,3,H,W].')
		batch, _, height, width = rgb.shape
		if height != width or masks.shape[0] != batch or tuple(masks.shape[-2:]) != (
			height, width
		):
			raise ValueError('RGB and role masks must have matching square image dimensions.')
		if self.pad == 0:
			return rgb.float(), masks.float()
		if shift_index is None:
			shift_index = torch.randint(
				0, 2 * self.pad + 1, (batch, 1, 1, 2),
				device=rgb.device, dtype=torch.float32,
			)
		if tuple(shift_index.shape) != (batch, 1, 1, 2):
			raise ValueError('ROF shift index must have shape [B,1,1,2].')
		rgb_pad = F.pad(rgb.float(), (self.pad,) * 4, mode='replicate')
		flat_masks = masks.float().reshape(batch, -1, height, width)
		mask_pad = F.pad(flat_masks, (self.pad,) * 4, mode='constant', value=0.0)
		padded = height + 2 * self.pad
		eps = 1.0 / padded
		axis = torch.linspace(
			-1.0 + eps, 1.0 - eps, padded,
			device=rgb.device, dtype=rgb_pad.dtype,
		)[:height]
		axis = axis.unsqueeze(0).repeat(height, 1).unsqueeze(2)
		grid = torch.cat([axis, axis.transpose(1, 0)], dim=2)
		grid = grid.unsqueeze(0).repeat(batch, 1, 1, 1)
		grid = grid + shift_index.to(grid.dtype) * (2.0 / padded)
		shifted_rgb = F.grid_sample(
			rgb_pad, grid, mode='bilinear', padding_mode='zeros', align_corners=False
		)
		shifted_masks = F.grid_sample(
			mask_pad, grid, mode='nearest', padding_mode='zeros', align_corners=False
		).reshape_as(masks)
		return shifted_rgb, shifted_masks


class RobustObjectFieldEncoder(nn.Module):
	"""Encode exact-K Cutie observations into five 64-D SimNorm tokens per role."""

	def __init__(self, cfg):
		super().__init__()
		validate_config(cfg)
		self.num_roles = int(_get(cfg, 'cutie_object_num_roles', 2))
		self.local_tokens = int(_get(
			cfg, 'robust_object_field_local_tokens', DEFAULT_LOCAL_TOKENS
		))
		self.token_dim = int(_get(
			cfg, 'robust_object_field_token_dim', DEFAULT_TOKEN_DIM
		))
		self.simnorm_dim = int(_get(cfg, 'simnorm_dim', 8))
		self.context_radius = int(_get(cfg, 'robust_object_field_context_radius', 3))
		self.temporal_role_geometry = bool(_get(
			cfg, TEMPORAL_ROLE_GEOMETRY_FLAG, False
		))
		self.latent_dim = latent_dim(cfg)
		self.augmentation = JointFieldShiftAug(
			_get(cfg, 'robust_object_field_random_shift_pad', 3)
		)
		# RGB[9], union mask[3], context ring[3], fixed XY[2].
		self.backbone = nn.Sequential(
			nn.Conv2d(17, 32, 5, stride=2, padding=2), nn.ReLU(inplace=False),
			nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=False),
			nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=False),
		)
		# V0: pooled feature[64] + anchor/centroid/spread[6]
		#     + occupancy[1] + current status[4] = 75.
		# Temporal opt-in: ordered pooled features[3*64] + geometry[6]
		#     + ordered occupancy[3] + current status[4] + motion[6] = 211.
		local_input_dim = (
			STACK_FRAMES * 64 + 6 + STACK_FRAMES + STATUS_DIM
			+ TEMPORAL_ROLE_SIGNAL_DIM
			if self.temporal_role_geometry else 75
		)
		self.local_projection = nn.Sequential(
			nn.Linear(local_input_dim, 128), nn.Mish(),
			nn.Linear(128, self.token_dim),
		)
		# Preserve the order of all three 512-D query summaries and status fields.
		self.global_projection = nn.Sequential(
			nn.Linear(STACK_FRAMES * (QUERY_DIM + STATUS_DIM), 256),
			nn.Mish(), nn.Linear(256, self.token_dim),
		)
		self.register_buffer(
			'_anchor_offsets',
			torch.tensor([
				[-0.75, -0.75], [0.75, -0.75],
				[-0.75, 0.75], [0.75, 0.75],
			], dtype=torch.float32),
			persistent=True,
		)

	def _validate_observation(self, obs):
		try:
			rgb = obs['rgb']
			objects = obs['object']
			masks = obs['object_mask']
			role_exists = obs['role_exists']
		except (KeyError, TypeError) as exc:
			raise ValueError(
				"ROF observation requires rgb/object/object_mask/role_exists."
			) from exc
		batch = rgb.shape[0] if rgb.ndim == 4 else None
		expected = {
			'rgb': (batch, RGB_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
			'object': (batch, self.num_roles, OBJECT_DIM),
			'object_mask': (
				batch, self.num_roles, STACK_FRAMES, IMAGE_SIZE, IMAGE_SIZE
			),
			'role_exists': (batch, self.num_roles),
		}
		actual = {
			'rgb': tuple(rgb.shape), 'object': tuple(objects.shape),
			'object_mask': tuple(masks.shape),
			'role_exists': tuple(role_exists.shape),
		}
		if rgb.ndim != 4 or actual != expected:
			raise ValueError(f'ROF observation shape mismatch: {actual} != {expected}.')
		if rgb.dtype != torch.uint8:
			raise ValueError('ROF causal RGB must be uint8.')
		if objects.dtype != torch.float32:
			raise ValueError('ROF Cutie descriptors must be float32.')
		if masks.dtype != torch.bool:
			raise ValueError('ROF object masks must be boolean.')
		if role_exists.dtype != torch.float32:
			raise ValueError('ROF role_exists must be float32.')
		# Data-dependent Python assertions would break the compiled update graph.
		# The wrapper and replay buffer enforce them before compilation; retain the
		# same fail-closed checks for every eager/direct call.
		compiling = torch.compiler.is_compiling()
		if not compiling and not bool(torch.isfinite(objects).all().item()):
			raise ValueError('ROF Cutie descriptors must be finite.')
		if not compiling and not bool((role_exists == 1).all().item()):
			raise ValueError('ROF V0 admits exact-K real roles only; padding is forbidden.')
		return rgb, objects, masks, role_exists

	def forward(self, obs, *, shift_index=None, return_tokens=False):
		rgb, objects, masks, role_exists = self._validate_observation(obs)
		rgb, masks = self.augmentation(rgb, masks, shift_index=shift_index)
		batch = rgb.shape[0]
		frames = rgb.reshape(batch, STACK_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)
		union = masks.amax(dim=1)
		kernel = 2 * self.context_radius + 1
		dilated_union = F.max_pool2d(
			union.reshape(batch * STACK_FRAMES, 1, IMAGE_SIZE, IMAGE_SIZE),
			kernel, stride=1, padding=self.context_radius,
		).reshape(batch, STACK_FRAMES, IMAGE_SIZE, IMAGE_SIZE)
		dilated_union = dilated_union.clamp(0.0, 1.0)
		ring = (dilated_union - union).clamp(0.0, 1.0)
		gated_rgb = (frames.div(255.0).sub(0.5)) * dilated_union.unsqueeze(2)
		axis = torch.linspace(-1.0, 1.0, IMAGE_SIZE, device=rgb.device, dtype=rgb.dtype)
		yy, xx = torch.meshgrid(axis, axis, indexing='ij')
		xy64 = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
		backbone_input = torch.cat([
			gated_rgb.reshape(batch, RGB_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
			union, ring, xy64,
		], dim=1)
		feature = self.backbone(backbone_input)
		if tuple(feature.shape[-2:]) != (8, 8):
			raise AssertionError(f'ROF backbone must produce 8x8 fields, got {feature.shape}.')

		axis8 = torch.linspace(-1.0, 1.0, 8, device=rgb.device, dtype=feature.dtype)
		yy8, xx8 = torch.meshgrid(axis8, axis8, indexing='ij')
		coords = torch.stack([xx8, yy8], dim=-1)
		offsets = self._anchor_offsets.to(device=rgb.device, dtype=feature.dtype)
		object_frames = objects.reshape(
			batch, self.num_roles, STACK_FRAMES, FRAME_DIM
		)
		status = object_frames[:, :, -1, STATUS_START:STATUS_START + STATUS_DIM]
		if self.temporal_role_geometry:
			mask8, centroids, spreads, _, motion = _temporal_role_geometry(
				masks, field_size=8
			)
			role_support = F.max_pool2d(
				masks.reshape(
					batch * self.num_roles * STACK_FRAMES,
					1, IMAGE_SIZE, IMAGE_SIZE,
				),
				kernel, stride=1, padding=self.context_radius,
			)
			role_support = F.adaptive_max_pool2d(
				role_support, (8, 8)
			).reshape(
				batch, self.num_roles, STACK_FRAMES, 8, 8
			).clamp(0.0, 1.0)
			anchors_by_frame = (
				centroids.unsqueeze(-2)
				+ offsets.view(1, 1, 1, self.local_tokens, 2)
				* spreads.unsqueeze(-2)
			)
			scale_by_frame = spreads.unsqueeze(-2).clamp_min(0.20)
			delta = (
				coords.view(1, 1, 1, 1, 8, 8, 2)
				- anchors_by_frame.unsqueeze(-2).unsqueeze(-2)
			) / scale_by_frame.unsqueeze(-2).unsqueeze(-2)
			weights = (
				torch.exp(-0.5 * delta.square().sum(-1))
				* role_support.unsqueeze(3)
			)
			weight_sum = weights.sum(dim=(-2, -1)).clamp_min(1e-6)
			pooled_by_frame = torch.einsum(
				'bktqhw,bchw->bktqc', weights, feature
			) / weight_sum.unsqueeze(-1)
			pooled = pooled_by_frame.permute(0, 1, 3, 2, 4).reshape(
				batch, self.num_roles, self.local_tokens, STACK_FRAMES * 64
			)
			occupancy = (
				torch.einsum('bktqhw,bkthw->bktq', weights, mask8)
				/ weight_sum
			).permute(0, 1, 3, 2)
			anchors = anchors_by_frame[:, :, -1]
			centroid = centroids[:, :, -1]
			spread = spreads[:, :, -1]
			local_context = torch.cat([
				anchors,
				centroid.unsqueeze(-2).expand(-1, -1, self.local_tokens, -1),
				spread.unsqueeze(-2).expand(-1, -1, self.local_tokens, -1),
				occupancy,
				status.unsqueeze(-2).expand(-1, -1, self.local_tokens, -1),
				motion.unsqueeze(-2).expand(-1, -1, self.local_tokens, -1),
			], dim=-1)
		else:
			# Preserve the original V0 current-mask pooling path exactly.
			current_masks = masks[:, :, -1]
			role_support = F.max_pool2d(
				current_masks.reshape(
					batch * self.num_roles, 1, IMAGE_SIZE, IMAGE_SIZE
				),
				kernel, stride=1, padding=self.context_radius,
			).reshape(batch * self.num_roles, 1, IMAGE_SIZE, IMAGE_SIZE)
			role_support = F.adaptive_max_pool2d(role_support, (8, 8)).reshape(
				batch, self.num_roles, 8, 8
			).clamp(0.0, 1.0)
			mask8 = F.adaptive_avg_pool2d(
				current_masks.reshape(
					batch * self.num_roles, 1, IMAGE_SIZE, IMAGE_SIZE
				),
				(8, 8),
			).reshape(batch, self.num_roles, 8, 8)
			mass = mask8.sum(dim=(-2, -1)).clamp_min(1e-6)
			centroid = (
				mask8.unsqueeze(-1) * coords.view(1, 1, 8, 8, 2)
			).sum(dim=(-3, -2)) / mass.unsqueeze(-1)
			centered = (
				coords.view(1, 1, 8, 8, 2)
				- centroid.unsqueeze(-2).unsqueeze(-2)
			)
			variance = (
				mask8.unsqueeze(-1) * centered.square()
			).sum(dim=(-3, -2)) / mass.unsqueeze(-1)
			spread = variance.clamp_min(0.02 ** 2).sqrt()
			anchors = (
				centroid.unsqueeze(-2)
				+ offsets.view(1, 1, self.local_tokens, 2)
				* spread.unsqueeze(-2)
			)
			scale = spread.unsqueeze(-2).clamp_min(0.20)
			delta = (
				coords.view(1, 1, 1, 8, 8, 2)
				- anchors.unsqueeze(-2).unsqueeze(-2)
			) / scale.unsqueeze(-2).unsqueeze(-2)
			weights = (
				torch.exp(-0.5 * delta.square().sum(-1))
				* role_support.unsqueeze(2)
			)
			weight_sum = weights.sum(dim=(-2, -1)).clamp_min(1e-6)
			pooled = torch.einsum('bkqhw,bchw->bkqc', weights, feature)
			pooled = pooled / weight_sum.unsqueeze(-1)
			occupancy = torch.einsum('bkqhw,bkhw->bkq', weights, mask8)
			occupancy = occupancy / weight_sum
			local_context = torch.cat([
				anchors,
				centroid.unsqueeze(-2).expand(-1, -1, self.local_tokens, -1),
				spread.unsqueeze(-2).expand(-1, -1, self.local_tokens, -1),
				occupancy.unsqueeze(-1),
				status.unsqueeze(-2).expand(-1, -1, self.local_tokens, -1),
			], dim=-1)
		local_logits = self.local_projection(torch.cat([pooled, local_context], dim=-1))
		local_tokens = _simnorm(local_logits, self.simnorm_dim)

		ordered_query_status = torch.cat([
			object_frames[..., :QUERY_DIM],
			object_frames[..., STATUS_START:STATUS_START + STATUS_DIM],
		], dim=-1).reshape(batch, self.num_roles, -1)
		global_token = _simnorm(
			self.global_projection(ordered_query_status), self.simnorm_dim
		).unsqueeze(-2)
		tokens = torch.cat([local_tokens, global_token], dim=-2)
		tokens = tokens * role_exists.to(tokens.dtype).unsqueeze(-1).unsqueeze(-1)
		flat = tokens.reshape(batch, self.latent_dim)
		if (
			not torch.compiler.is_compiling()
			and not bool(torch.isfinite(flat).all().item())
		):
			raise FloatingPointError('ROF encoder produced a non-finite latent.')
		return (flat, tokens) if return_tokens else flat


class RobustObjectFieldDynamics(nn.Module):
	"""Action-conditioned V0 transition with the same exact SimNorm domain."""

	def __init__(self, cfg):
		super().__init__()
		validate_config(cfg)
		self.latent_dim = latent_dim(cfg)
		self.action_dim = int(_get(cfg, 'action_dim'))
		self.simnorm_dim = int(_get(cfg, 'simnorm_dim', 8))
		hidden = int(_get(cfg, 'robust_object_field_dynamics_hidden_dim', 512))
		self.transition = nn.Sequential(
			nn.Linear(self.latent_dim + self.action_dim, hidden),
			nn.Mish(), nn.Linear(hidden, self.latent_dim),
		)

	def forward(self, latent, action):
		if latent.shape[-1] != self.latent_dim:
			raise ValueError(f'ROF latent width must be {self.latent_dim}.')
		if action.shape[-1] != self.action_dim or action.shape[:-1] != latent.shape[:-1]:
			raise ValueError('ROF latent and action leading shapes must match.')
		result = _simnorm(
			self.transition(torch.cat([latent, action], dim=-1)), self.simnorm_dim
		)
		if (
			not torch.compiler.is_compiling()
			and not bool(torch.isfinite(result).all().item())
		):
			raise FloatingPointError('ROF dynamics produced a non-finite latent.')
		return result


class RobustObjectFieldDecoder(nn.Module):
	"""Shared per-role decoder for the unchanged Cutie auxiliary target."""

	def __init__(self, cfg):
		super().__init__()
		validate_config(cfg)
		self.num_roles = int(_get(cfg, 'cutie_object_num_roles', 2))
		self.role_latent_dim = num_tokens(cfg) * int(_get(
			cfg, 'robust_object_field_token_dim', DEFAULT_TOKEN_DIM
		))
		self.latent_dim = latent_dim(cfg)
		hidden = int(_get(cfg, 'robust_object_field_decoder_hidden_dim', 256))
		self.decoder = nn.Sequential(
			nn.Linear(self.role_latent_dim, hidden), nn.Mish(),
			nn.Linear(hidden, OBJECT_DIM),
		)

	def forward(self, latent):
		if latent.shape[-1] != self.latent_dim:
			raise ValueError(f'ROF decoder latent width must be {self.latent_dim}.')
		roles = latent.reshape(*latent.shape[:-1], self.num_roles, self.role_latent_dim)
		return self.decoder(roles)


@dataclass(frozen=True)
class PackedFieldShape:
	num_roles: int
	num_tokens: int
	token_dim: int

	@property
	def latent_dim(self):
		return self.num_roles * self.num_tokens * self.token_dim

	def pack(self, tokens):
		if tuple(tokens.shape[-3:]) != (
			self.num_roles, self.num_tokens, self.token_dim
		):
			raise ValueError('ROF token tensor has the wrong trailing shape.')
		return tokens.reshape(*tokens.shape[:-3], self.latent_dim)

	def unpack(self, latent):
		if latent.shape[-1] != self.latent_dim:
			raise ValueError('ROF packed latent has the wrong width.')
		return latent.reshape(
			*latent.shape[:-1], self.num_roles, self.num_tokens, self.token_dim
		)
