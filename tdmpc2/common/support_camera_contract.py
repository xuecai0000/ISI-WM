"""Shared metadata contract for RGB/indexed-mask support camera alignment.

The masks covered by this module are privileged *offline support labels*.
Runtime policy observations and tracker frames must never request simulator
segmentation.  Keeping this contract dependency-free lets collectors, loaders,
and audit tools all enforce exactly the same fail-closed metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral


SAME_CAMERA_CONTRACT_SCHEMA = "rgb_mask_same_state_same_camera_v1"
SUPPORT_SEGMENTATION_SCOPE = "offline_support_only"
QUADRUPED_TASKS = frozenset(("quadruped-run", "quadruped-walk"))
QUADRUPED_CAMERA_ID = 2

_CONTRACT_KEYS = frozenset((
	"schema",
	"rgb_camera_id",
	"mask_camera_id",
	"same_simulator_state",
	"segmentation_scope",
	"runtime_segmentation_allowed",
))


def _camera_id(value, field: str) -> int:
	if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
		raise ValueError(f"{field} must be a non-negative integer, got {value!r}.")
	return int(value)


def build_same_camera_contract(camera_id: int) -> dict:
	"""Return canonical support-only camera metadata for one renderer view."""

	camera_id = _camera_id(camera_id, "camera_id")
	return {
		"schema": SAME_CAMERA_CONTRACT_SCHEMA,
		"rgb_camera_id": camera_id,
		"mask_camera_id": camera_id,
		"same_simulator_state": True,
		"segmentation_scope": SUPPORT_SEGMENTATION_SCOPE,
		"runtime_segmentation_allowed": False,
	}


def validate_same_camera_contract(
	metadata: Mapping,
	*,
	expected_camera_id: int | None = None,
) -> int:
	"""Validate an explicit same-state/same-camera support contract.

	``metadata`` is either a support collection or its geometry catalog.  Both
	must carry the legacy scalar ``camera_id`` and the explicit structured
	contract so a stale camera-0 Quadruped pack cannot pass by omission.
	"""

	if not isinstance(metadata, Mapping):
		raise ValueError("Camera-contract metadata must be an object.")
	camera_id = _camera_id(metadata.get("camera_id"), "camera_id")
	contract = metadata.get("camera_contract")
	if not isinstance(contract, Mapping):
		raise ValueError("Explicit camera_contract metadata is required.")
	if frozenset(contract) != _CONTRACT_KEYS:
		raise ValueError(
			"camera_contract must contain exactly "
			f"{sorted(_CONTRACT_KEYS)!r}, got {sorted(contract)!r}."
		)
	expected = build_same_camera_contract(camera_id)
	if dict(contract) != expected:
		raise ValueError(
			"RGB and support mask must use the same simulator state and camera, "
			"and segmentation must remain support-only; "
			f"expected {expected!r}, got {dict(contract)!r}."
		)
	if expected_camera_id is not None:
		required = _camera_id(expected_camera_id, "expected_camera_id")
		if camera_id != required:
			raise ValueError(
				f"Support camera {camera_id} does not match required camera {required}."
			)
	return camera_id
