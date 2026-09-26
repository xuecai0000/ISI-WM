"""Support-calibrated whole-arm mask decoder for ``reacher-visual-small``.

The decoder consumes exactly two binary masks: ``whole_arm`` and ``goal``.
It never receives RGB pixels, simulator state, rewards, audit labels, or task
ground truth at inference time.  The fixed base and link lengths are derived
once from the existing, manually RGB-labelled support pack.

The arm is decoded as a single articulated object.  A base-connected component
is thinned to a one-pixel centreline, the unambiguous path away from the base is
measured by geodesic arc length.  The elbow is sampled at the support-derived
``L1`` distance.  A centreline at least as long as ``L1 + L2`` is sampled at
that exact support distance; a bounded endpoint shortfall instead uses the
observed endpoint as the control tip.  Disconnection, a substantial branch, a
missing base attachment, or insufficient arc-length evidence fails closed
instead of emitting a plausible-looking point.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

if __package__ == 'tdmpc2.perception':
	from .visual_small_cutie_decoder import (
		VisualSmallCutieCalibration,
		VisualSmallCutieDecoderContractError,
		load_visual_small_cutie_calibration,
	)
else:  # Keep the standalone contract independent of adapter/torch imports.
	_CALIBRATION_PATH = Path(__file__).with_name('visual_small_cutie_decoder.py')
	_CALIBRATION_SPEC = importlib.util.spec_from_file_location(
		'_visual_small_whole_arm_support_calibration',
		_CALIBRATION_PATH,
	)
	if _CALIBRATION_SPEC is None or _CALIBRATION_SPEC.loader is None:
		raise RuntimeError(
			f'Could not load support calibration module: {_CALIBRATION_PATH}'
		)
	_calibration_module = importlib.util.module_from_spec(_CALIBRATION_SPEC)
	sys.modules[_CALIBRATION_SPEC.name] = _calibration_module
	_CALIBRATION_SPEC.loader.exec_module(_calibration_module)
	VisualSmallCutieCalibration = _calibration_module.VisualSmallCutieCalibration
	VisualSmallCutieDecoderContractError = (
		_calibration_module.VisualSmallCutieDecoderContractError
	)
	load_visual_small_cutie_calibration = (
		_calibration_module.load_visual_small_cutie_calibration
	)


POINT_ROLES = ('base', 'elbow', 'control_tip', 'goal')
OBJECT_ROLES = ('whole_arm', 'goal')
DECODER_FORMAT = 'visual_small_whole_arm_point_decoder_v2'
DECODER_NAME = 'visual_small_whole_arm_point_decoder'
DECODER_VERSION = 2


class VisualSmallWholeArmDecoderContractError(ValueError):
	"""Raised when support data or masks violate the whole-arm contract."""


# The calibration is deliberately the same immutable support-only object used
# by the earlier role-mask decoder.  Renaming the public alias avoids inventing
# a second support format while keeping this module's API task-specific.
VisualSmallWholeArmCalibration = VisualSmallCutieCalibration


def load_visual_small_whole_arm_calibration(
	annotation_path: str | Path,
	*,
	expected_records: int = 6,
) -> VisualSmallWholeArmCalibration:
	"""Load fixed ``base``, ``L1``, and ``L2`` from verified RGB-only support."""
	try:
		return load_visual_small_cutie_calibration(
			annotation_path,
			expected_records=expected_records,
		)
	except VisualSmallCutieDecoderContractError as exc:
		raise VisualSmallWholeArmDecoderContractError(str(exc)) from exc


@dataclass(frozen=True)
class _Component:
	mask: np.ndarray
	size: int
	min_base_distance_px: float
	first_yx: tuple[int, int]


def _components(mask: np.ndarray, base_xy: np.ndarray) -> list[_Component]:
	"""Return deterministic 8-connected components and base distances."""
	mask = np.asarray(mask, dtype=bool)
	visited = np.zeros_like(mask, dtype=bool)
	height, width = mask.shape
	result: list[_Component] = []
	for start_y_value, start_x_value in np.argwhere(mask):
		start_y = int(start_y_value)
		start_x = int(start_x_value)
		if visited[start_y, start_x]:
			continue
		stack = [(start_y, start_x)]
		visited[start_y, start_x] = True
		pixels: list[tuple[int, int]] = []
		while stack:
			y, x = stack.pop()
			pixels.append((y, x))
			for dy in (-1, 0, 1):
				for dx in (-1, 0, 1):
					if dx == 0 and dy == 0:
						continue
					ny, nx = y + dy, x + dx
					if (
						0 <= ny < height
						and 0 <= nx < width
						and mask[ny, nx]
						and not visited[ny, nx]
					):
						visited[ny, nx] = True
						stack.append((ny, nx))
		component_mask = np.zeros_like(mask, dtype=bool)
		yx = np.asarray(pixels, dtype=np.int64)
		component_mask[yx[:, 0], yx[:, 1]] = True
		xy = yx[:, ::-1].astype(np.float64)
		minimum_distance = float(np.linalg.norm(xy - base_xy, axis=1).min())
		result.append(_Component(
			mask=component_mask,
			size=len(pixels),
			min_base_distance_px=minimum_distance,
			first_yx=min(pixels),
		))
	return result


def _zhang_suen_skeleton(mask: np.ndarray) -> np.ndarray:
	"""Deterministic Zhang-Suen thinning for a small binary mask."""
	image = np.asarray(mask, dtype=bool).copy()
	if not image.any():
		return image
	changed = True
	while changed:
		changed = False
		for phase in (0, 1):
			padded = np.pad(image, 1, mode='constant', constant_values=False)
			remove: list[tuple[int, int]] = []
			for y_value, x_value in np.argwhere(image):
				y = int(y_value) + 1
				x = int(x_value) + 1
				p2 = bool(padded[y - 1, x])
				p3 = bool(padded[y - 1, x + 1])
				p4 = bool(padded[y, x + 1])
				p5 = bool(padded[y + 1, x + 1])
				p6 = bool(padded[y + 1, x])
				p7 = bool(padded[y + 1, x - 1])
				p8 = bool(padded[y, x - 1])
				p9 = bool(padded[y - 1, x - 1])
				neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
				count = sum(neighbors)
				if count < 2 or count > 6:
					continue
				cycle = neighbors + (neighbors[0],)
				transitions = sum(
					(not cycle[index]) and cycle[index + 1]
					for index in range(8)
				)
				if transitions != 1:
					continue
				if phase == 0:
					if p2 and p4 and p6:
						continue
					if p4 and p6 and p8:
						continue
				else:
					if p2 and p4 and p8:
						continue
					if p2 and p6 and p8:
						continue
				remove.append((y - 1, x - 1))
			if remove:
				changed = True
				for y, x in remove:
					image[y, x] = False
	return image


Node = tuple[int, int]
Graph = dict[Node, tuple[tuple[Node, float], ...]]


def _skeleton_graph(skeleton: np.ndarray) -> Graph:
	"""Build a weighted 8-neighbour graph of the one-pixel centreline."""
	nodes = {tuple(int(value) for value in yx) for yx in np.argwhere(skeleton)}
	graph: Graph = {}
	for y, x in sorted(nodes):
		neighbors: list[tuple[Node, float]] = []
		for dy in (-1, 0, 1):
			for dx in (-1, 0, 1):
				if dx == 0 and dy == 0:
					continue
				candidate = (y + dy, x + dx)
				if candidate not in nodes:
					continue
				neighbors.append((candidate, math.hypot(dx, dy)))
		graph[(y, x)] = tuple(sorted(neighbors))
	return graph


def _dijkstra(
	graph: Graph,
	starts: Mapping[Node, float],
) -> tuple[dict[Node, float], dict[Node, Node]]:
	"""Shortest paths with deterministic tie handling."""
	distance = {node: float('inf') for node in graph}
	previous: dict[Node, Node] = {}
	queue: list[tuple[float, Node]] = []
	for node, initial_distance in sorted(starts.items()):
		value = float(initial_distance)
		if node not in graph or value >= distance[node]:
			continue
		distance[node] = value
		heapq.heappush(queue, (value, node))
	while queue:
		current_distance, node = heapq.heappop(queue)
		if current_distance > distance[node] + 1e-12:
			continue
		for neighbor, weight in graph[node]:
			candidate = current_distance + weight
			if candidate < distance[neighbor] - 1e-12:
				distance[neighbor] = candidate
				previous[neighbor] = node
				heapq.heappush(queue, (candidate, neighbor))
			elif abs(candidate - distance[neighbor]) <= 1e-12:
				old_previous = previous.get(neighbor)
				if old_previous is None or node < old_previous:
					previous[neighbor] = node
	return distance, previous


def _path_to(previous: Mapping[Node, Node], start: Node, end: Node) -> list[Node]:
	path = [end]
	seen = {end}
	while path[-1] != start:
		parent = previous.get(path[-1])
		if parent is None or parent in seen:
			return []
		seen.add(parent)
		path.append(parent)
	path.reverse()
	return path


def _sample_polyline(
	coordinates: np.ndarray,
	target_distance: float,
) -> list[float] | None:
	"""Linearly interpolate a polyline at a fixed arc length."""
	if coordinates.ndim != 2 or coordinates.shape[1:] != (2,) or not len(coordinates):
		return None
	if len(coordinates) == 1:
		return coordinates[0].astype(np.float64).tolist()
	segment_lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
	cumulative = np.concatenate((np.zeros(1), np.cumsum(segment_lengths)))
	if target_distance < -1e-12 or target_distance > cumulative[-1] + 1e-12:
		return None
	target = float(np.clip(target_distance, 0.0, cumulative[-1]))
	index = int(np.searchsorted(cumulative, target, side='right') - 1)
	index = min(index, len(segment_lengths) - 1)
	length = float(segment_lengths[index])
	if length <= 1e-12:
		point = coordinates[index]
	else:
		fraction = (target - cumulative[index]) / length
		point = coordinates[index] + fraction * (
			coordinates[index + 1] - coordinates[index]
		)
	return point.astype(np.float64).tolist()


def _geodesic_mask_endpoint(
	component_mask: np.ndarray,
	base_xy: np.ndarray,
	*,
	link_length_px: float,
) -> tuple[list[float] | None, dict[str, Any]]:
	"""Return the unique far mask cap, compensating for skeleton retraction."""
	graph = _skeleton_graph(component_mask)
	if not graph:
		return None, {
			'available': False,
			'cap_pixels': 0,
			'cap_components': 0,
			'maximum_geodesic_distance_px': None,
		}
	start = min(
		graph,
		key=lambda node: (
			float(np.linalg.norm(
				np.asarray([node[1], node[0]], dtype=np.float64) - base_xy
			)),
			node,
		),
	)
	distance, _ = _dijkstra(graph, {start: 0.0})
	finite_distance = {
		node: value for node, value in distance.items() if math.isfinite(value)
	}
	if not finite_distance:
		return None, {
			'available': False,
			'cap_pixels': 0,
			'cap_components': 0,
			'maximum_geodesic_distance_px': None,
		}
	maximum_distance = max(finite_distance.values())
	cap_band_px = max(2.0, 0.15 * link_length_px)
	cap_nodes = {
		node for node, value in finite_distance.items()
		if value >= maximum_distance - cap_band_px
	}
	cap_mask = np.zeros_like(component_mask, dtype=bool)
	for y, x in cap_nodes:
		cap_mask[y, x] = True
	cap_components = _components(cap_mask, base_xy)
	if len(cap_components) != 1:
		return None, {
			'available': False,
			'cap_pixels': len(cap_nodes),
			'cap_components': len(cap_components),
			'cap_band_px': cap_band_px,
			'maximum_geodesic_distance_px': maximum_distance,
		}
	# Use the centre of the maximally distant pixels as a centreline anchor.
	# Averaging the wider near-tip cap would retract the evidence slightly and
	# can make a valid 24px arm appear a fraction of a pixel too short.
	farthest_nodes = {
		node for node, value in finite_distance.items()
		if value >= maximum_distance - 1e-9
	}
	point = np.asarray(
		[[node[1], node[0]] for node in sorted(farthest_nodes)],
		dtype=np.float64,
	).mean(axis=0)
	return point.tolist(), {
		'available': True,
		'cap_pixels': len(cap_nodes),
		'cap_components': 1,
		'farthest_pixels': len(farthest_nodes),
		'cap_band_px': cap_band_px,
		'maximum_geodesic_distance_px': maximum_distance,
	}


def _largest_component_centroid(
	mask: np.ndarray,
) -> tuple[list[float] | None, dict[str, Any]]:
	components = _components(mask, np.zeros(2, dtype=np.float64))
	total_pixels = int(mask.sum())
	if len(components) != 1:
		return None, {
			'components': len(components),
			'total_pixels': total_pixels,
			'component_pixels': 0 if not components else max(item.size for item in components),
			'unambiguous': False,
		}
	component = components[0]
	yx = np.argwhere(component.mask)
	if len(yx) < 3:
		return None, {
			'components': 1,
			'total_pixels': total_pixels,
			'component_pixels': len(yx),
			'unambiguous': False,
		}
	centroid = yx[:, ::-1].astype(np.float64).mean(axis=0)
	return centroid.tolist(), {
		'components': 1,
		'total_pixels': total_pixels,
		'component_pixels': len(yx),
		'unambiguous': True,
	}


@dataclass(frozen=True)
class VisualSmallWholeArmDecodeResult:
	"""Decoded task points, confidence, lost flags, and GT-free diagnostics."""

	points: dict[str, list[float] | None]
	point_confidence: dict[str, float]
	geometry_confidence: float
	lost: dict[str, bool]
	diagnostics: dict[str, Any]

	def as_dict(self) -> dict[str, Any]:
		return {
			'points': self.points,
			'point_confidence': self.point_confidence,
			'geometry_confidence': float(self.geometry_confidence),
			'lost': self.lost,
			'diagnostics': self.diagnostics,
		}


class VisualSmallWholeArmPointDecoder:
	"""Recover reacher task points from one whole-arm mask and one goal mask."""

	def __init__(
		self,
		calibration: VisualSmallWholeArmCalibration,
		*,
		base_attachment_radius_fraction: float = 0.25,
		base_attachment_radius_min_px: float = 1.5,
		maximum_branch_extent_px: float = 2.5,
		maximum_endpoint_shortfall_px: float = 4.0,
		maximum_total_length_fraction: float = 1.35,
	):
		if not isinstance(calibration, VisualSmallCutieCalibration):
			raise TypeError('calibration must be a support-derived calibration.')
		for name, value in (
			('base_attachment_radius_fraction', base_attachment_radius_fraction),
			('maximum_total_length_fraction', maximum_total_length_fraction),
		):
			if not math.isfinite(float(value)) or float(value) <= 0.0:
				raise ValueError(f'{name} must be positive and finite.')
		for name, value in (
			('base_attachment_radius_min_px', base_attachment_radius_min_px),
			('maximum_branch_extent_px', maximum_branch_extent_px),
			('maximum_endpoint_shortfall_px', maximum_endpoint_shortfall_px),
		):
			if not math.isfinite(float(value)) or float(value) < 0.0:
				raise ValueError(f'{name} must be finite and non-negative.')
		if maximum_total_length_fraction <= 1.0:
			raise ValueError('maximum_total_length_fraction must exceed 1.')
		if (
			float(maximum_endpoint_shortfall_px)
			>= float(calibration.distal_length_px)
		):
			raise ValueError(
				'maximum_endpoint_shortfall_px must be strictly smaller than the '
				'support-derived distal link length.'
			)
		self.calibration = calibration
		self.base_attachment_radius_fraction = float(
			base_attachment_radius_fraction
		)
		self.base_attachment_radius_min_px = float(base_attachment_radius_min_px)
		self.maximum_branch_extent_px = float(maximum_branch_extent_px)
		self.maximum_endpoint_shortfall_px = float(
			maximum_endpoint_shortfall_px
		)
		self.maximum_total_length_fraction = float(
			maximum_total_length_fraction
		)

	@classmethod
	def from_support(
		cls,
		annotation_path: str | Path,
		*,
		expected_records: int = 6,
		**decoder_kwargs: Any,
	) -> 'VisualSmallWholeArmPointDecoder':
		calibration = load_visual_small_whole_arm_calibration(
			annotation_path,
			expected_records=expected_records,
		)
		return cls(calibration, **decoder_kwargs)

	def reset(self) -> None:
		"""Reset episode state (the decoder is deliberately stateless)."""
		return None

	def metadata(self) -> dict[str, Any]:
		calibration_metadata = self.calibration.metadata()
		return {
			'format': DECODER_FORMAT,
			'name': DECODER_NAME,
			'version': DECODER_VERSION,
			**calibration_metadata,
			'base_xy': list(calibration_metadata['base']),
			'input_roles': list(OBJECT_ROLES),
			'algorithm': 'base_connected_zhang_suen_geodesic_arc_length_v2',
			'base_attachment_radius_fraction': self.base_attachment_radius_fraction,
			'base_attachment_radius_min_px': self.base_attachment_radius_min_px,
			'maximum_branch_extent_px': self.maximum_branch_extent_px,
			'maximum_endpoint_shortfall_px': self.maximum_endpoint_shortfall_px,
			'maximum_total_length_fraction': self.maximum_total_length_fraction,
			'component_policy': (
				'exactly_one_base_attached_8_connected_whole_arm_fail_closed'
			),
			'short_centerline_policy': 'bounded_shortfall_endpoint_v2',
		}

	def _validated_masks(
		self,
		object_masks: Mapping[str, np.ndarray],
	) -> dict[str, np.ndarray]:
		if not isinstance(object_masks, Mapping):
			raise VisualSmallWholeArmDecoderContractError(
				'object_masks must be a role-to-mask mapping.'
			)
		if set(object_masks) != set(OBJECT_ROLES):
			raise VisualSmallWholeArmDecoderContractError(
				f'object_masks must contain exactly {OBJECT_ROLES!r}.'
			)
		result: dict[str, np.ndarray] = {}
		for role in OBJECT_ROLES:
			mask = np.asarray(object_masks[role])
			if mask.ndim != 2 or mask.shape != self.calibration.source_resolution:
				raise VisualSmallWholeArmDecoderContractError(
					f'{role} mask must have shape '
					f'{self.calibration.source_resolution}, got {mask.shape}.'
				)
			if np.issubdtype(mask.dtype, np.bool_):
				result[role] = np.ascontiguousarray(mask, dtype=bool)
				continue
			if (
				not np.issubdtype(mask.dtype, np.number)
				or np.issubdtype(mask.dtype, np.complexfloating)
				or not np.isfinite(mask).all()
				or not np.logical_or(mask == 0, mask == 1).all()
			):
				raise VisualSmallWholeArmDecoderContractError(
					f'{role} mask must be boolean or numeric binary 0/1.'
				)
			result[role] = np.ascontiguousarray(mask, dtype=bool)
		return result

	def decode(
		self,
		object_masks: Mapping[str, np.ndarray],
	) -> VisualSmallWholeArmDecodeResult:
		"""Decode one frame from whole-arm and goal masks only."""
		masks = self._validated_masks(object_masks)
		base = np.asarray(self.calibration.base_xy, dtype=np.float64)
		L1 = float(self.calibration.proximal_length_px)
		L2 = float(self.calibration.distal_length_px)
		expected_total = L1 + L2
		attachment_radius = max(
			self.base_attachment_radius_min_px,
			self.base_attachment_radius_fraction * L1,
		)

		components = _components(masks['whole_arm'], base)
		attached = [
			item for item in components
			if item.min_base_distance_px <= attachment_radius
		]
		selected = (
			min(
				attached,
				key=lambda item: (
					-item.size,
					item.min_base_distance_px,
					item.first_yx,
				),
			)
			if attached
			else None
		)
		component_mask = (
			selected.mask
			if selected is not None
			else np.zeros_like(masks['whole_arm'], dtype=bool)
		)
		failure_reasons: list[str] = []
		if len(components) == 0:
			failure_reasons.append('whole_arm_empty')
		elif len(components) != 1:
			failure_reasons.append('whole_arm_disconnected')
		if selected is None:
			failure_reasons.append('whole_arm_not_base_attached')

		skeleton = _zhang_suen_skeleton(component_mask)
		graph = _skeleton_graph(skeleton)
		path_nodes: list[Node] = []
		start_distance_px: float | None = None
		centerline_length_px = 0.0
		max_branch_extent_px = 0.0
		off_path_pixels = 0
		reachable_pixels = 0
		if graph:
			start = min(
				graph,
				key=lambda node: (
					float(np.linalg.norm(
						np.asarray([node[1], node[0]], dtype=np.float64) - base
					)),
					node,
				),
			)
			start_xy = np.asarray([start[1], start[0]], dtype=np.float64)
			start_distance_px = float(np.linalg.norm(start_xy - base))
			distance, previous = _dijkstra(graph, {start: 0.0})
			reachable = {
				node for node, value in distance.items() if math.isfinite(value)
			}
			reachable_pixels = len(reachable)
			if len(reachable) != len(graph):
				failure_reasons.append('skeleton_disconnected')
			if reachable:
				end = max(
					reachable,
					key=lambda node: (
						distance[node],
						float(np.linalg.norm(
							np.asarray([node[1], node[0]], dtype=np.float64) - base
						)),
						node,
					),
				)
				path_nodes = _path_to(previous, start, end)
				path_set = set(path_nodes)
				off_path = set(graph) - path_set
				off_path_pixels = len(off_path)
				if path_nodes:
					branch_distance, _ = _dijkstra(
						graph,
						{node: 0.0 for node in path_nodes},
					)
					finite_branch_distance = [
						branch_distance[node]
						for node in off_path
						if math.isfinite(branch_distance[node])
					]
					max_branch_extent_px = (
						max(finite_branch_distance)
						if finite_branch_distance
						else 0.0
					)
		else:
			failure_reasons.append('skeleton_empty')

		# Zhang-Suen thinning intentionally retracts rounded end caps.  The
		# skeleton is nevertheless derived from the already verified, unique
		# base-attached component, so prepend the fixed base below rather than
		# rejecting a healthy arm merely because its medial line starts inward.
		if path_nodes:
			path_xy = np.asarray(
				[[node[1], node[0]] for node in path_nodes],
				dtype=np.float64,
			)
			if float(np.linalg.norm(path_xy[0] - base)) > 1e-12:
				path_xy = np.vstack((base, path_xy))
			elif len(path_xy):
				path_xy[0] = base
			skeleton_path_length_px = float(
				np.linalg.norm(np.diff(path_xy, axis=0), axis=1).sum()
			) if len(path_xy) > 1 else 0.0
			tip_endpoint, tip_endpoint_diagnostics = (
				_geodesic_mask_endpoint(
					component_mask,
					base,
					link_length_px=L2,
				)
			)
			if tip_endpoint is None:
				centerline_xy = path_xy
				tip_endpoint_diagnostics['skeleton_extension_px'] = None
				failure_reasons.append('whole_arm_tip_endpoint_unavailable')
			else:
				tip_endpoint_array = np.asarray(tip_endpoint, dtype=np.float64)
				tip_endpoint_diagnostics['skeleton_extension_px'] = float(
					np.linalg.norm(tip_endpoint_array - path_xy[-1])
				)
				if float(np.linalg.norm(tip_endpoint_array - path_xy[-1])) > 1e-12:
					centerline_xy = np.vstack((path_xy, tip_endpoint_array))
				else:
					centerline_xy = path_xy
			centerline_length_px = float(
				np.linalg.norm(np.diff(centerline_xy, axis=0), axis=1).sum()
			) if len(centerline_xy) > 1 else 0.0
		else:
			path_xy = np.empty((0, 2), dtype=np.float64)
			centerline_xy = path_xy
			skeleton_path_length_px = 0.0
			tip_endpoint = None
			tip_endpoint_diagnostics = {
				'available': False,
				'cap_pixels': 0,
				'cap_components': 0,
				'maximum_geodesic_distance_px': None,
				'skeleton_extension_px': None,
			}
		if max_branch_extent_px > self.maximum_branch_extent_px:
			failure_reasons.append('whole_arm_abnormal_branch')
		shortfall_px = expected_total - centerline_length_px
		if shortfall_px > self.maximum_endpoint_shortfall_px + 1e-12:
			failure_reasons.append('whole_arm_arc_length_insufficient')
		if centerline_length_px > self.maximum_total_length_fraction * expected_total:
			failure_reasons.append('whole_arm_arc_length_excessive')

		whole_arm_valid = not failure_reasons
		if whole_arm_valid:
			observed_elbow_arc_length_px = L1
			elbow = _sample_polyline(centerline_xy, L1)
			if shortfall_px > 0.0:
				# A tolerated shortfall represents a retracted distal cap.  Preserve
				# the support-calibrated elbow distance and use the verified observed
				# endpoint for the tip instead of requesting an unavailable arc length.
				tip = (
					centerline_xy[-1].astype(np.float64).tolist()
					if len(centerline_xy)
					else None
				)
				tip_sampling = 'bounded_shortfall_endpoint_v2'
				observed_tip_arc_length_px = centerline_length_px
			else:
				# The far mask cap only repairs skeleton end-cap retraction and supplies
				# length evidence. It is not the task tip: an overlong false-positive
				# tail must never move the control point beyond support kinematics.
				tip = _sample_polyline(centerline_xy, expected_total)
				tip_sampling = 'support_total_arc_length_on_validated_centerline'
				observed_tip_arc_length_px = expected_total
			if elbow is None or tip is None:
				whole_arm_valid = False
				failure_reasons.append('arc_length_sampling_failed')
				elbow = None
				tip = None
				observed_tip_arc_length_px = None
		else:
			elbow = None
			tip = None
			tip_sampling = 'fail_closed'
			observed_elbow_arc_length_px = None
			observed_tip_arc_length_px = None

		goal, goal_diagnostics = _largest_component_centroid(masks['goal'])
		length_residual_px = abs(centerline_length_px - expected_total)
		length_confidence = float(math.exp(
			-length_residual_px / max(2.0, self.maximum_endpoint_shortfall_px)
		))
		branch_confidence = float(math.exp(
			-max_branch_extent_px / max(self.maximum_branch_extent_px, 1e-6)
		))
		arm_confidence = (
			float(np.clip(length_confidence * branch_confidence, 0.0, 1.0))
			if whole_arm_valid
			else 0.0
		)
		goal_confidence = 1.0 if goal is not None else 0.0
		points = {
			'base': [float(base[0]), float(base[1])],
			'elbow': elbow,
			'control_tip': tip,
			'goal': goal,
		}
		point_confidence = {
			'base': 1.0,
			'elbow': arm_confidence,
			'control_tip': arm_confidence,
			'goal': goal_confidence,
		}
		lost = {role: points[role] is None for role in POINT_ROLES}
		geometry_confidence = float(min(
			point_confidence['elbow'],
			point_confidence['control_tip'],
			point_confidence['goal'],
		))
		diagnostics = {
			'failure_reasons': failure_reasons,
			'whole_arm_valid': whole_arm_valid,
			'whole_arm_component': {
				'components': len(components),
				'attached_components': len(attached),
				'total_pixels': int(masks['whole_arm'].sum()),
				'selected_pixels': 0 if selected is None else int(selected.size),
				'min_base_distance_px': (
					None if selected is None else float(selected.min_base_distance_px)
				),
				'base_attached': selected is not None,
			},
			'base_attachment_radius_px': float(attachment_radius),
			'skeleton_pixels': len(graph),
			'reachable_skeleton_pixels': reachable_pixels,
			'path_pixels': len(path_nodes),
			'off_path_pixels': off_path_pixels,
			'junction_pixels': sum(
				len(neighbors) >= 3 for neighbors in graph.values()
			),
			'max_branch_extent_px': float(max_branch_extent_px),
			'maximum_branch_extent_px': self.maximum_branch_extent_px,
			'skeleton_start_distance_px': start_distance_px,
			'skeleton_path_length_px': skeleton_path_length_px,
			'centerline_length_px': centerline_length_px,
			'support_L1_px': L1,
			'support_L2_px': L2,
			'support_total_length_px': expected_total,
			'total_length_residual_px': float(length_residual_px),
			'endpoint_shortfall_px': float(shortfall_px),
			'observed_elbow_arc_length_px': observed_elbow_arc_length_px,
			'observed_tip_arc_length_px': observed_tip_arc_length_px,
			'raw_expected_to_observed_length_ratio': (
				None
				if centerline_length_px <= 1e-12
				else float(expected_total / centerline_length_px)
			),
			'maximum_endpoint_shortfall_px': self.maximum_endpoint_shortfall_px,
			'tip_sampling': tip_sampling,
			'short_centerline_policy_applied': bool(
				whole_arm_valid
				and tip_sampling == 'bounded_shortfall_endpoint_v2'
			),
			'tip_endpoint': tip_endpoint_diagnostics,
			'goal_component': goal_diagnostics,
			'geometry_confidence': geometry_confidence,
		}
		return VisualSmallWholeArmDecodeResult(
			points=points,
			point_confidence=point_confidence,
			geometry_confidence=geometry_confidence,
			lost=lost,
			diagnostics=diagnostics,
		)


__all__ = (
	'VisualSmallWholeArmCalibration',
	'VisualSmallWholeArmDecodeResult',
	'VisualSmallWholeArmDecoderContractError',
	'VisualSmallWholeArmPointDecoder',
	'load_visual_small_whole_arm_calibration',
)
