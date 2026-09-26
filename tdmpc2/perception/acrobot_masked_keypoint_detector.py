"""Mask-conditioned Acrobot keypoints with fixed camera calibration.

Cutie supplies one whole-Acrobot mask.  This network only localizes the
base/elbow/tip inside that foreground.  Image points are mapped to normalized
world XZ coordinates by one immutable camera homography stored in the
checkpoint; there is no learned world-coordinate or velocity head.
"""

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
	from tdmpc2.perception.acrobot_keypoint_detector import SpatialSoftArgmax2d
except (ImportError, ModuleNotFoundError):
	# Contract scripts execute from inside tdmpc2/, where tdmpc2.py shadows the
	# namespace package; production module execution uses the absolute import.
	from perception.acrobot_keypoint_detector import SpatialSoftArgmax2d


CHECKPOINT_FORMAT = 'acrobot_masked_keypoint_detector_v2'
POINT_NAMES = ('base', 'elbow', 'tip')


@dataclass(frozen=True)
class AcrobotMaskedKeypointConfig:
	history: int = 4
	image_size: int = 64
	action_dim: int = 1
	base_channels: int = 24
	mask_dilation_pixels: int = 3
	background_keep: float = 0.0

	def validated(self):
		if self.history < 3:
			raise ValueError('Masked keypoint history must contain at least three frames.')
		if self.image_size < 32 or self.image_size % 8:
			raise ValueError('image_size must be >=32 and divisible by 8.')
		if self.action_dim < 1 or self.base_channels < 8 or self.base_channels % 8:
			raise ValueError('Invalid action_dim/base_channels.')
		if self.mask_dilation_pixels < 0 or self.mask_dilation_pixels > 16:
			raise ValueError('mask_dilation_pixels must be in [0,16].')
		if not 0.0 <= self.background_keep <= 1.0:
			raise ValueError('background_keep must be in [0,1].')
		return self


def _validate_homography(value):
	value = torch.as_tensor(value, dtype=torch.float32)
	if value.shape != (3, 3) or not torch.isfinite(value).all():
		raise ValueError('pixel_to_world_homography must be finite [3,3].')
	if abs(float(value[2, 2])) < 1e-8:
		raise ValueError('Homography normalization element cannot be zero.')
	return value / value[2, 2]


class AcrobotMaskedKeypointNet(nn.Module):
	"""Small causal heatmap network fed masked RGB plus the whole-object mask."""

	def __init__(self, config, pixel_to_world_homography):
		super().__init__()
		self.config = config.validated()
		self.register_buffer(
			'pixel_to_world_homography',
			_validate_homography(pixel_to_world_homography),
			persistent=True,
		)
		channels = config.base_channels
		self.encoder = nn.Sequential(
			nn.Conv2d(config.history * 4, channels, 5, stride=2, padding=2),
			nn.GroupNorm(4, channels), nn.SiLU(),
			nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
			nn.GroupNorm(8, channels * 2), nn.SiLU(),
			nn.Conv2d(channels * 2, channels * 4, 3, stride=2, padding=1),
			nn.GroupNorm(8, channels * 4), nn.SiLU(),
		)
		self.action_encoder = nn.Sequential(
			nn.Linear((config.history - 1) * config.action_dim, channels * 2),
			nn.SiLU(), nn.Linear(channels * 2, channels * 4),
		)
		self.heatmap_head = nn.Sequential(
			nn.ConvTranspose2d(channels * 4, channels * 2, 4, stride=2, padding=1),
			nn.GroupNorm(8, channels * 2), nn.SiLU(),
			nn.Conv2d(channels * 2, len(POINT_NAMES), 1),
		)
		self.soft_argmax = SpatialSoftArgmax2d()
		self.confidence_head = nn.Linear(channels * 4, len(POINT_NAMES))

	def _prepare_inputs(self, rgb, foreground_mask):
		if rgb.ndim != 5 or rgb.shape[1:3] != (self.config.history, 3):
			raise ValueError(f'Invalid masked keypoint RGB shape {tuple(rgb.shape)}.')
		if foreground_mask.ndim != 5 or foreground_mask.shape[1:3] != (
			self.config.history, 1,
		):
			raise ValueError(f'Invalid whole-object mask shape {tuple(foreground_mask.shape)}.')
		if rgb.shape[-2:] != (self.config.image_size, self.config.image_size):
			rgb = F.interpolate(
				rgb.flatten(0, 1), size=(self.config.image_size, self.config.image_size),
				mode='bilinear', align_corners=False,
			).unflatten(0, rgb.shape[:2])
		if foreground_mask.shape[-2:] != rgb.shape[-2:]:
			foreground_mask = F.interpolate(
				foreground_mask.flatten(0, 1), size=rgb.shape[-2:], mode='nearest',
			).unflatten(0, foreground_mask.shape[:2])
		mask = foreground_mask.clamp(0., 1.)
		if self.config.mask_dilation_pixels:
			radius = self.config.mask_dilation_pixels
			mask = F.max_pool2d(
				mask.flatten(0, 1), kernel_size=2 * radius + 1,
				stride=1, padding=radius,
			).unflatten(0, foreground_mask.shape[:2])
		keep = self.config.background_keep + (1. - self.config.background_keep) * mask
		return torch.cat((rgb * keep, mask), dim=2).flatten(1, 2)

	def image_to_world(self, image_xy):
		"""Map normalized image coordinates to normalized world XZ."""
		pixel = (image_xy + 1.) * (self.config.image_size - 1.) / 2.
		ones = torch.ones_like(pixel[..., :1])
		homogeneous = torch.cat((pixel, ones), dim=-1)
		mapped = homogeneous @ self.pixel_to_world_homography.T
		denominator = mapped[..., 2:]
		denominator = torch.where(
			denominator.abs() < 1e-6,
			torch.where(denominator < 0., -torch.ones_like(denominator), torch.ones_like(denominator)) * 1e-6,
			denominator,
		)
		return mapped[..., :2] / denominator

	def forward(self, rgb, actions, foreground_mask):
		if actions.shape[1:] != (self.config.history - 1, self.config.action_dim):
			raise ValueError(f'Invalid masked keypoint action shape {tuple(actions.shape)}.')
		features = self.encoder(self._prepare_inputs(rgb, foreground_mask))
		features = features + self.action_encoder(actions.flatten(1))[:, :, None, None]
		heatmaps = self.heatmap_head(features)
		image_xy = self.soft_argmax(heatmaps)
		global_features = features.mean(dim=(-2, -1))
		return {
			'heatmaps': heatmaps,
			'image_xy': image_xy,
			'world_xz': self.image_to_world(image_xy),
			'confidence': torch.sigmoid(self.confidence_head(global_features)),
		}


def save_checkpoint(path, model, *, calibration=None, training=None, metrics=None):
	payload = {
		'format': CHECKPOINT_FORMAT,
		'point_names': list(POINT_NAMES),
		'model_config': asdict(model.config),
		'model_state_dict': model.state_dict(),
		'calibration': dict(calibration or {}),
		'training': dict(training or {}),
		'metrics': dict(metrics or {}),
	}
	torch.save(payload, str(path))


def load_checkpoint(path, *, device='cpu'):
	payload = torch.load(str(path), map_location=device, weights_only=False)
	if not isinstance(payload, dict) or payload.get('format') != CHECKPOINT_FORMAT:
		raise ValueError(f'Unsupported masked Acrobot keypoint checkpoint: {path}.')
	if tuple(payload.get('point_names', ())) != POINT_NAMES:
		raise ValueError('Masked keypoint checkpoint point ordering is incompatible.')
	config = AcrobotMaskedKeypointConfig(**payload['model_config']).validated()
	homography = payload['model_state_dict']['pixel_to_world_homography']
	model = AcrobotMaskedKeypointNet(config, homography).to(device)
	model.load_state_dict(payload['model_state_dict'], strict=True)
	model.eval()
	return model, payload


class AcrobotMaskedKeypointPredictor:
	def __init__(self, checkpoint, *, device='cpu'):
		self.device = torch.device(device)
		self.model, self.metadata = load_checkpoint(checkpoint, device=self.device)
		self.config = self.model.config

	@torch.inference_mode()
	def predict(self, rgb, actions, foreground_mask: Optional[np.ndarray] = None):
		if foreground_mask is None:
			raise ValueError('The V2 keypoint predictor requires whole-object Cutie masks.')
		rgb = np.asarray(rgb)
		mask = np.asarray(foreground_mask)
		expected_rgb = (
			self.config.history, self.config.image_size, self.config.image_size, 3,
		)
		expected_mask = expected_rgb[:3]
		if rgb.shape != expected_rgb or mask.shape != expected_mask:
			raise ValueError(f'Invalid V2 inference inputs: rgb={rgb.shape}, mask={mask.shape}.')
		output = self.model(
			torch.as_tensor(rgb, device=self.device, dtype=torch.float32)
				.permute(0, 3, 1, 2)[None] / 255.,
			torch.as_tensor(actions, device=self.device, dtype=torch.float32)[None],
			torch.as_tensor(mask, device=self.device, dtype=torch.float32)[:, None][None],
		)
		return {name: value[0].detach().cpu().numpy() for name, value in output.items()}
