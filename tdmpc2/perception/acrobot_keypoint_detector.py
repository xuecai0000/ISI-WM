"""Small causal Acrobot keypoint detector used by visual pose observations.

The network predicts image-space heatmaps for base/elbow/tip and the normalized
world XZ locations consumed by the established articulated-pose controller.
Simulator kinematics are training labels only and are never read by inference.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CHECKPOINT_FORMAT = 'acrobot_keypoint_detector_v1'
POINT_NAMES = ('base', 'elbow', 'tip')


@dataclass(frozen=True)
class AcrobotKeypointConfig:
	history: int = 4
	image_size: int = 64
	action_dim: int = 1
	use_foreground_mask: bool = False
	base_channels: int = 24

	def validated(self):
		if self.history < 3:
			raise ValueError('Keypoint history must contain at least three frames.')
		if self.image_size < 32 or self.image_size % 8:
			raise ValueError('Keypoint image_size must be >=32 and divisible by 8.')
		if self.action_dim < 1 or self.base_channels < 8 or self.base_channels % 8:
			raise ValueError('Invalid action_dim/base_channels for keypoint model.')
		return self


class SpatialSoftArgmax2d(nn.Module):
	def forward(self, heatmaps):
		batch, points, height, width = heatmaps.shape
		probability = F.softmax(heatmaps.reshape(batch, points, -1), dim=-1)
		y = torch.linspace(-1., 1., height, device=heatmaps.device, dtype=heatmaps.dtype)
		x = torch.linspace(-1., 1., width, device=heatmaps.device, dtype=heatmaps.dtype)
		grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
		grid = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
		return probability @ grid


class AcrobotKeypointNet(nn.Module):
	"""Compact spatial bottleneck with causal RGB/action context."""

	def __init__(self, config: AcrobotKeypointConfig):
		super().__init__()
		self.config = config.validated()
		per_frame = 3 + int(config.use_foreground_mask)
		channels = config.base_channels
		self.encoder = nn.Sequential(
			nn.Conv2d(config.history * per_frame, channels, 5, stride=2, padding=2),
			nn.GroupNorm(4, channels), nn.SiLU(),
			nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
			nn.GroupNorm(8, channels * 2), nn.SiLU(),
			nn.Conv2d(channels * 2, channels * 4, 3, stride=2, padding=1),
			nn.GroupNorm(8, channels * 4), nn.SiLU(),
		)
		action_features = channels * 2
		self.action_encoder = nn.Sequential(
			nn.Linear((config.history - 1) * config.action_dim, action_features),
			nn.SiLU(), nn.Linear(action_features, channels * 4),
		)
		self.heatmap_head = nn.Sequential(
			nn.ConvTranspose2d(channels * 4, channels * 2, 4, stride=2, padding=1),
			nn.GroupNorm(8, channels * 2), nn.SiLU(),
			nn.Conv2d(channels * 2, len(POINT_NAMES), 1),
		)
		self.soft_argmax = SpatialSoftArgmax2d()
		self.world_head = nn.Sequential(
			nn.Linear(2 + channels * 4 + len(POINT_NAMES), channels * 2), nn.SiLU(),
			nn.Linear(channels * 2, 2),
		)
		self.confidence_head = nn.Linear(channels * 4, len(POINT_NAMES))
		self.angular_velocity_head = nn.Linear(channels * 4, 2)

	def forward(self, rgb, actions, foreground_mask=None):
		"""Accept RGB [B,T,3,H,W], actions [B,T-1,A], optional masks."""
		if rgb.ndim != 5 or rgb.shape[1:3] != (self.config.history, 3):
			raise ValueError(f'Invalid keypoint RGB shape {tuple(rgb.shape)}.')
		if actions.shape[1:] != (self.config.history - 1, self.config.action_dim):
			raise ValueError(f'Invalid keypoint action shape {tuple(actions.shape)}.')
		if rgb.shape[-2:] != (self.config.image_size, self.config.image_size):
			rgb = F.interpolate(
				rgb.flatten(0, 1), size=(self.config.image_size, self.config.image_size),
				mode='bilinear', align_corners=False,
			).unflatten(0, rgb.shape[:2])
		inputs = [rgb]
		if self.config.use_foreground_mask:
			if foreground_mask is None:
				raise ValueError('This checkpoint requires a foreground mask.')
			if foreground_mask.ndim != 5 or foreground_mask.shape[1:3] != (
				self.config.history, 1,
			):
				raise ValueError(
					f'Invalid keypoint foreground mask shape {tuple(foreground_mask.shape)}.'
				)
			if foreground_mask.shape[-2:] != rgb.shape[-2:]:
				foreground_mask = F.interpolate(
					foreground_mask.flatten(0, 1), size=rgb.shape[-2:], mode='nearest',
				).unflatten(0, foreground_mask.shape[:2])
			inputs.append(foreground_mask)
		stacked = torch.cat(inputs, dim=2).flatten(1, 2)
		features = self.encoder(stacked)
		action_context = self.action_encoder(actions.flatten(1))
		features = features + action_context[:, :, None, None]
		heatmaps = self.heatmap_head(features)
		image_xy = self.soft_argmax(heatmaps)
		global_features = features.mean(dim=(-2, -1))
		point_ids = torch.eye(
			len(POINT_NAMES), device=rgb.device, dtype=rgb.dtype,
		)[None].expand(rgb.shape[0], -1, -1)
		world_input = torch.cat((
			image_xy,
			global_features[:, None].expand(-1, len(POINT_NAMES), -1),
			point_ids,
		), dim=-1)
		world_xz = self.world_head(world_input)
		confidence = torch.sigmoid(self.confidence_head(global_features))
		angular_velocity = self.angular_velocity_head(global_features)
		return {
			'heatmaps': heatmaps,
			'image_xy': image_xy,
			'world_xz': world_xz,
			'confidence': confidence,
			'angular_velocity': angular_velocity,
		}


def save_checkpoint(path, model, *, training=None, metrics=None):
	payload = {
		'format': CHECKPOINT_FORMAT,
		'point_names': list(POINT_NAMES),
		'model_config': asdict(model.config),
		'model_state_dict': model.state_dict(),
		'training': dict(training or {}),
		'metrics': dict(metrics or {}),
	}
	torch.save(payload, str(path))


def load_checkpoint(path, *, device='cpu'):
	payload = torch.load(str(path), map_location=device, weights_only=False)
	if not isinstance(payload, dict) or payload.get('format') != CHECKPOINT_FORMAT:
		raise ValueError(f'Unsupported Acrobot keypoint checkpoint: {path}.')
	if tuple(payload.get('point_names', ())) != POINT_NAMES:
		raise ValueError('Acrobot keypoint checkpoint point ordering is incompatible.')
	config = AcrobotKeypointConfig(**payload['model_config']).validated()
	model = AcrobotKeypointNet(config).to(device)
	model.load_state_dict(payload['model_state_dict'], strict=True)
	model.eval()
	return model, payload


class AcrobotKeypointPredictor:
	"""NumPy inference adapter used by the environment wrapper."""

	def __init__(self, checkpoint, *, device='cpu'):
		self.device = torch.device(device)
		self.model, self.metadata = load_checkpoint(checkpoint, device=self.device)
		self.config = self.model.config

	@torch.inference_mode()
	def predict(self, rgb, actions, foreground_mask: Optional[np.ndarray] = None):
		rgb = np.asarray(rgb)
		if rgb.shape != (
			self.config.history, self.config.image_size, self.config.image_size, 3,
		):
			raise ValueError(f'Invalid inference RGB history shape {rgb.shape}.')
		rgb_tensor = torch.as_tensor(
			rgb, device=self.device, dtype=torch.float32,
		).permute(0, 3, 1, 2)[None] / 255.
		action_tensor = torch.as_tensor(
			actions, device=self.device, dtype=torch.float32,
		)[None]
		mask_tensor = None
		if foreground_mask is not None:
			mask_tensor = torch.as_tensor(
				foreground_mask, device=self.device, dtype=torch.float32,
			)[:, None][None]
		output = self.model(rgb_tensor, action_tensor, mask_tensor)
		return {
			name: value[0].detach().cpu().numpy()
			for name, value in output.items()
		}
