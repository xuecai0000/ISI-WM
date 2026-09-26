"""Small causal recurrent observer for articulated Acrobot pose."""

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CHECKPOINT_FORMAT = 'acrobot_causal_state_observer_v1'
INPUT_DIM = 21


def pose_angles(points):
	delta = points[..., 1:, :] - points[..., :-1, :]
	return torch.atan2(delta[..., 0], delta[..., 1])


def observer_inputs(points, confidence, mask_valid, actions, *, dt=0.04):
	"""Build causal per-frame measurements; actions[t] means action before frame t."""
	if points.ndim != 4 or points.shape[-2:] != (3, 2):
		raise ValueError(f'Observer points must be [B,T,3,2], got {points.shape}.')
	if confidence.shape != points.shape[:2] + (3,):
		raise ValueError('Observer confidence shape mismatch.')
	if mask_valid.shape != points.shape[:2] + (1,):
		raise ValueError('Observer mask validity shape mismatch.')
	if actions.shape != points.shape[:2] + (1,):
		raise ValueError('Observer previous-action shape mismatch.')
	delta = torch.zeros_like(points)
	delta[:, 1:] = (points[:, 1:] - points[:, :-1]) / dt
	angles = pose_angles(points)
	angle_features = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(2)
	return torch.cat((
		points.flatten(2), delta.flatten(2).clamp(-20., 20.), angle_features,
		confidence, mask_valid.to(points.dtype), actions,
	), dim=-1)


@dataclass(frozen=True)
class AcrobotStateObserverConfig:
	input_dim: int = INPUT_DIM
	hidden_dim: int = 96
	layers: int = 1
	link_length: float = 0.5
	omega_scale: float = 10.0
	control_dt: float = 0.04

	def validated(self):
		if self.input_dim != INPUT_DIM or self.hidden_dim < 32 or self.layers < 1:
			raise ValueError('Invalid state-observer dimensions.')
		if self.link_length <= 0 or self.omega_scale <= 0 or self.control_dt <= 0:
			raise ValueError('Invalid observer physical constants.')
		return self


class AcrobotStateObserver(nn.Module):
	def __init__(self, config, base_xz):
		super().__init__()
		self.config = config.validated()
		base = torch.as_tensor(base_xz, dtype=torch.float32)
		if base.shape != (2,) or not torch.isfinite(base).all():
			raise ValueError('Observer base_xz must be finite [2].')
		self.register_buffer('base_xz', base, persistent=True)
		self.input_encoder = nn.Sequential(
			nn.Linear(config.input_dim, config.hidden_dim), nn.LayerNorm(config.hidden_dim),
			nn.SiLU(),
		)
		self.gru = nn.GRU(
			config.hidden_dim, config.hidden_dim, num_layers=config.layers,
			batch_first=True,
		)
		self.head = nn.Sequential(
			nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU(),
			nn.Linear(config.hidden_dim, 6),
		)

	def _decode(self, raw):
		vectors = raw[..., :4].reshape(*raw.shape[:-1], 2, 2)
		vectors = F.normalize(vectors, dim=-1, eps=1e-6)
		omega = torch.tanh(raw[..., 4:]) * self.config.omega_scale
		base = self.base_xz.expand(*raw.shape[:-1], 2)
		elbow = base + self.config.link_length * vectors[..., 0, :]
		tip = elbow + self.config.link_length * vectors[..., 1, :]
		points = torch.stack((base, elbow, tip), dim=-2)
		return {
			'points': points, 'angle_vectors': vectors,
			'angular_velocity': omega,
		}

	def forward(self, inputs, hidden=None):
		if inputs.ndim != 3 or inputs.shape[-1] != self.config.input_dim:
			raise ValueError(f'Invalid observer input shape {tuple(inputs.shape)}.')
		features = self.input_encoder(inputs)
		output, hidden = self.gru(features, hidden)
		return self._decode(self.head(output)), hidden

	def step(self, inputs, hidden=None):
		if inputs.ndim != 2:
			raise ValueError('Observer step input must be [B,D].')
		output, hidden = self.forward(inputs[:, None], hidden)
		return {name: value[:, 0] for name, value in output.items()}, hidden


def save_checkpoint(path, model, *, training=None, metrics=None):
	torch.save({
		'format': CHECKPOINT_FORMAT,
		'model_config': asdict(model.config),
		'model_state_dict': model.state_dict(),
		'training': dict(training or {}),
		'metrics': dict(metrics or {}),
	}, str(path))


def load_checkpoint(path, *, device='cpu'):
	payload = torch.load(str(path), map_location=device, weights_only=False)
	if not isinstance(payload, dict) or payload.get('format') != CHECKPOINT_FORMAT:
		raise ValueError(f'Unsupported observer checkpoint: {path}.')
	config = AcrobotStateObserverConfig(**payload['model_config']).validated()
	base = payload['model_state_dict']['base_xz']
	model = AcrobotStateObserver(config, base).to(device)
	model.load_state_dict(payload['model_state_dict'], strict=True)
	model.eval()
	return model, payload


__all__ = [
	'AcrobotStateObserver', 'AcrobotStateObserverConfig', 'CHECKPOINT_FORMAT',
	'INPUT_DIM', 'load_checkpoint', 'observer_inputs', 'pose_angles',
	'save_checkpoint',
]
