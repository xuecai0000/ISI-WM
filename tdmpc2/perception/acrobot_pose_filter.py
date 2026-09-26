"""Dependency-light causal smoothing and velocity recovery for Acrobot pose."""

from collections import deque
import math

import numpy as np


def _angles(points):
	delta = points[1:] - points[:-1]
	return np.arctan2(delta[:, 0], delta[:, 1])


class CausalAcrobotPoseFilter:
	"""EMA points, enforce link topology, and fit velocity over recent angles."""

	def __init__(self, *, dt=0.04, alpha=1.0, window=5, minimum_confidence=0.15):
		if dt <= 0 or not 0.0 < alpha <= 1.0 or window < 3:
			raise ValueError('Invalid causal pose-filter configuration.')
		self.dt = float(dt)
		self.alpha = float(alpha)
		self.window = int(window)
		self.minimum_confidence = float(minimum_confidence)
		self._angle_history = deque(maxlen=self.window)
		self._points = None
		self._frame = 0

	def reset(self):
		self._angle_history.clear()
		self._points = None
		self._frame = 0

	@staticmethod
	def _enforce_topology(points):
		value = np.asarray(points, dtype=np.float64).copy()
		for index in (1, 2):
			delta = value[index] - value[index - 1]
			length = float(np.linalg.norm(delta))
			if length > 1e-8:
				value[index] = value[index - 1] + delta * (0.5 / length)
		return value

	def update(self, points, confidence=None):
		points = np.asarray(points, dtype=np.float64)
		if points.shape != (3, 2) or not np.isfinite(points).all():
			raise ValueError('Causal pose filter requires finite points [3,2].')
		if confidence is None:
			confidence = np.ones(3, dtype=np.float64)
		confidence = np.asarray(confidence, dtype=np.float64)
		if confidence.shape != (3,) or not np.isfinite(confidence).all():
			raise ValueError('Causal pose filter requires finite confidence [3].')
		if self._points is None:
			filtered = points
		else:
			# Do not turn confidence into a permanent position lag. A trusted
			# point is accepted at the configured rate; only a genuinely
			# low-confidence point is held for one causal step.
			alpha = np.where(
				confidence >= self.minimum_confidence, self.alpha, 0.,
			)
			filtered = self._points + alpha[:, None] * (points - self._points)
		filtered = self._enforce_topology(filtered)
		angles = _angles(filtered)
		if self._angle_history:
			previous = self._angle_history[-1]
			angles = previous + (angles - previous + math.pi) % (2 * math.pi) - math.pi
		self._angle_history.append(angles)
		if len(self._angle_history) < 2:
			omega = np.zeros(2, dtype=np.float64)
		else:
			y = np.stack(self._angle_history)
			x = np.arange(len(y), dtype=np.float64) * self.dt
			x = x - x.mean()
			omega = (x[:, None] * (y - y.mean(axis=0))).sum(axis=0) / np.square(x).sum()
		self._points = filtered
		self._frame += 1
		return filtered.astype(np.float32), omega.astype(np.float32)
