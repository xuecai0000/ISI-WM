"""Validate RGB/indexed-mask camera alignment metadata for Cutie support.

This dependency-light audit never opens a simulator and never obtains runtime
ground truth.  It validates the collector's explicit same-state/same-camera
contract, decoded asset hashes, role IDs/counts, and geometry-catalog hash.  An
optional contact sheet supports the separate human semantic-overlay review.

New packs are strict by default.  ``--allow-legacy-single-camera-field`` exists
only to inspect already-generated packs that predate the explicit contract; a
legacy result is reported as non-explicit and must not be used to certify a new
Quadruped run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from tdmpc2.common.support_camera_contract import (  # noqa: E402
	QUADRUPED_CAMERA_ID,
	QUADRUPED_TASKS,
	validate_same_camera_contract,
)


FORMAT = "cutie_indexed_mask_support_v1"
SUPPORT_SCHEMA = "generic_indexed_v1"
_TOP_LEVEL_FIELDS = {"format", "roles", "collection", "records"}
_PALETTE = np.asarray((
	(0, 0, 0),
	(255, 165, 0),
	(0, 210, 255),
	(255, 0, 210),
	(80, 220, 80),
	(235, 70, 70),
	(130, 90, 255),
	(255, 225, 50),
), dtype=np.uint8)


class SupportCameraValidationError(ValueError):
	"""Raised when a support pack cannot prove the camera contract."""


def _decoded_sha256(array: np.ndarray) -> str:
	return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def _annotation_path(path: str | Path) -> Path:
	path = Path(path).expanduser().resolve()
	if path.is_dir():
		path = path / "annotations.json"
	if not path.is_file():
		raise FileNotFoundError(f"Support annotations not found: {path}")
	return path


def _asset_path(pack_root: Path, relative, field: str) -> Path:
	if not isinstance(relative, str) or not relative:
		raise SupportCameraValidationError(f"{field} must be a non-empty path.")
	path = (pack_root / relative).resolve()
	try:
		path.relative_to(pack_root)
	except ValueError as exc:
		raise SupportCameraValidationError(
			f"{field} escapes the support directory: {path}"
		) from exc
	if not path.is_file():
		raise FileNotFoundError(f"{field} not found: {path}")
	return path


def _legacy_camera_id(collection: dict, expected_camera_id: int | None) -> int:
	value = collection.get("camera_id")
	if isinstance(value, bool) or not isinstance(value, int) or value < 0:
		raise SupportCameraValidationError(
			f"Legacy collection.camera_id must be a non-negative integer, got {value!r}."
		)
	if expected_camera_id is not None and value != expected_camera_id:
		raise SupportCameraValidationError(
			f"Support camera {value} does not match required camera {expected_camera_id}."
		)
	return value


def validate_support_pack(
	annotation_path: str | Path,
	*,
	expected_task: str | None = None,
	expected_camera_id: int | None = None,
	allow_legacy_single_camera_field: bool = False,
) -> dict:
	"""Return a deterministic summary or fail closed on any contract mismatch."""

	path = _annotation_path(annotation_path)
	try:
		payload = json.loads(path.read_text(encoding="utf-8"))
	except json.JSONDecodeError as exc:
		raise SupportCameraValidationError(f"Invalid JSON in {path}: {exc}") from exc
	if not isinstance(payload, dict) or set(payload) != _TOP_LEVEL_FIELDS:
		raise SupportCameraValidationError(
			f"Support JSON must contain exactly {sorted(_TOP_LEVEL_FIELDS)!r}."
		)
	if payload.get("format") != FORMAT:
		raise SupportCameraValidationError(
			f"Support format must be {FORMAT!r}, got {payload.get('format')!r}."
		)
	roles = payload.get("roles")
	if (
		not isinstance(roles, list)
		or not roles
		or any(not isinstance(role, str) or not role.strip() for role in roles)
		or len(set(roles)) != len(roles)
	):
		raise SupportCameraValidationError("roles must be a non-empty unique string list.")
	if len(roles) >= len(_PALETTE):
		raise SupportCameraValidationError(
			f"Validator palette supports at most {len(_PALETTE) - 1} roles."
		)

	collection = payload.get("collection")
	if not isinstance(collection, dict):
		raise SupportCameraValidationError("collection must be an object.")
	task = collection.get("task")
	if not isinstance(task, str) or not task:
		raise SupportCameraValidationError("collection.task must be a non-empty string.")
	if expected_task is not None and task != expected_task:
		raise SupportCameraValidationError(
			f"Support task {task!r} does not match required task {expected_task!r}."
		)
	if (
		collection.get("support_schema") != SUPPORT_SCHEMA
		or collection.get("observation") != "rgb"
		or collection.get("split") != "support"
		or collection.get("label_policy") != "simulator_segmentation_support_only"
		or collection.get("diagnostic_support") is not True
	):
		raise SupportCameraValidationError(
			"Support provenance must be generic_indexed_v1 RGB support with "
			"simulator_segmentation_support_only and diagnostic_support=true."
		)

	if task in QUADRUPED_TASKS:
		if expected_camera_id is not None and expected_camera_id != QUADRUPED_CAMERA_ID:
			raise SupportCameraValidationError(
				f"Quadruped runtime camera is fixed at {QUADRUPED_CAMERA_ID}, not "
				f"{expected_camera_id}."
			)
		expected_camera_id = QUADRUPED_CAMERA_ID
	explicit_contract = "camera_contract" in collection
	if explicit_contract:
		try:
			camera_id = validate_same_camera_contract(
				collection, expected_camera_id=expected_camera_id
			)
		except ValueError as exc:
			raise SupportCameraValidationError(str(exc)) from exc
	elif allow_legacy_single_camera_field:
		camera_id = _legacy_camera_id(collection, expected_camera_id)
	else:
		raise SupportCameraValidationError(
			"Explicit camera_contract metadata is required; regenerate this pack."
		)

	image_size = collection.get("image_size")
	if (
		not isinstance(image_size, list)
		or len(image_size) != 2
		or any(isinstance(value, bool) or not isinstance(value, int) or value < 1
			for value in image_size)
	):
		raise SupportCameraValidationError(
			f"collection.image_size must be [height,width], got {image_size!r}."
		)
	height, width = image_size

	pack_root = path.parent
	catalog_path = _asset_path(
		pack_root, collection.get("geom_catalog"), "collection.geom_catalog"
	)
	if _file_sha256(catalog_path) != collection.get("geom_catalog_sha256"):
		raise SupportCameraValidationError("Geometry-catalog file hash mismatch.")
	try:
		catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
	except json.JSONDecodeError as exc:
		raise SupportCameraValidationError(
			f"Invalid geometry-catalog JSON: {exc}"
		) from exc
	if not isinstance(catalog, dict) or catalog.get("task") != task:
		raise SupportCameraValidationError("Geometry-catalog task mismatch.")
	if explicit_contract:
		try:
			catalog_camera = validate_same_camera_contract(
				catalog, expected_camera_id=camera_id
			)
		except ValueError as exc:
			raise SupportCameraValidationError(
				f"Geometry-catalog camera contract: {exc}"
			) from exc
	else:
		catalog_camera = _legacy_camera_id(catalog, camera_id)
	if catalog_camera != camera_id:
		raise SupportCameraValidationError("Collection/catalog camera mismatch.")

	records = payload.get("records")
	episodes = collection.get("episodes")
	if (
		isinstance(episodes, bool)
		or not isinstance(episodes, int)
		or episodes < 1
		or not isinstance(records, list)
		or len(records) != episodes
	):
		raise SupportCameraValidationError(
			f"records must match positive collection.episodes, got {episodes!r}/"
			f"{len(records) if isinstance(records, list) else type(records).__name__}."
		)

	seen_assets = set()
	area_by_role = {role: [] for role in roles}
	for index, record in enumerate(records):
		if not isinstance(record, dict):
			raise SupportCameraValidationError(f"Record {index} must be an object.")
		if record.get("index") != index or record.get("episode") != index:
			raise SupportCameraValidationError(
				f"Record {index} index/episode must both equal its ordered position."
			)
		image_path = _asset_path(pack_root, record.get("image"), f"record {index} image")
		mask_path = _asset_path(
			pack_root, record.get("indexed_mask"), f"record {index} indexed_mask"
		)
		for asset in (image_path, mask_path):
			if asset in seen_assets:
				raise SupportCameraValidationError(f"Support asset reused: {asset}")
			seen_assets.add(asset)

		with Image.open(image_path) as image_file:
			image = np.asarray(image_file.convert("RGB"), dtype=np.uint8)
		with Image.open(mask_path) as mask_file:
			mask = np.asarray(mask_file)
		if image.shape != (height, width, 3):
			raise SupportCameraValidationError(
				f"Record {index} RGB shape {image.shape} != {(height, width, 3)}."
			)
		if mask.shape != (height, width) or mask.dtype != np.uint8:
			raise SupportCameraValidationError(
				f"Record {index} mask must be uint8 {(height, width)}, got "
				f"{mask.shape} {mask.dtype}."
			)
		if _decoded_sha256(image) != record.get("image_sha256"):
			raise SupportCameraValidationError(f"Record {index} decoded RGB hash mismatch.")
		if _decoded_sha256(mask) != record.get("indexed_mask_sha256"):
			raise SupportCameraValidationError(f"Record {index} decoded mask hash mismatch.")
		values = set(map(int, np.unique(mask)))
		expected_values = set(range(len(roles) + 1))
		if values != expected_values:
			raise SupportCameraValidationError(
				f"Record {index} indexed IDs {sorted(values)!r} != "
				f"{sorted(expected_values)!r}."
			)
		counts = {
			role: int((mask == role_id).sum())
			for role_id, role in enumerate(roles, start=1)
		}
		if record.get("role_pixel_counts") != counts:
			raise SupportCameraValidationError(
				f"Record {index} role_pixel_counts mismatch: expected {counts!r}."
			)
		for role, count in counts.items():
			area_by_role[role].append(count)

	return {
		"status": "ok",
		"task": task,
		"camera_id": camera_id,
		"explicit_same_camera_contract": explicit_contract,
		"same_simulator_state": True if explicit_contract else None,
		"runtime_segmentation_allowed": False if explicit_contract else None,
		"record_count": len(records),
		"roles": roles,
		"role_area_min_max": {
			role: [min(values), max(values)] for role, values in area_by_role.items()
		},
		"annotations_sha256": _file_sha256(path),
		"geom_catalog_sha256": _file_sha256(catalog_path),
		"semantic_alignment": "requires_contact_sheet_review",
	}


def write_contact_sheet(annotation_path: str | Path, output_path: str | Path) -> Path:
	"""Write deterministic RGB/labels/overlay rows after validation succeeds."""

	path = _annotation_path(annotation_path)
	payload = json.loads(path.read_text(encoding="utf-8"))
	roles = payload["roles"]
	height, width = payload["collection"]["image_size"]
	scale = max(1, 256 // max(height, width))
	panel_width, panel_height = width * scale, height * scale
	margin, header, row_label = 12, 34, 22
	board = Image.new(
		"RGB",
		(3 * panel_width + 4 * margin,
		 header + len(payload["records"]) * (panel_height + row_label + margin)),
		"white",
	)
	draw = ImageDraw.Draw(board)
	draw.text(
		(margin, 10),
		f"{payload['collection']['task']} camera {payload['collection']['camera_id']}: "
		"RGB | indexed labels | overlay",
		fill="black",
	)
	for index, record in enumerate(payload["records"]):
		with Image.open(path.parent / record["image"]) as image_file:
			image = np.asarray(image_file.convert("RGB"), dtype=np.uint8)
		with Image.open(path.parent / record["indexed_mask"]) as mask_file:
			mask = np.asarray(mask_file, dtype=np.uint8)
		labels = _PALETTE[mask]
		overlay = image.copy()
		foreground = mask != 0
		overlay[foreground] = (
			(image[foreground].astype(np.uint16) + labels[foreground].astype(np.uint16))
			// 2
		).astype(np.uint8)
		y = header + index * (panel_height + row_label + margin)
		draw.text((margin, y), f"record {index}; roles={','.join(roles)}", fill="black")
		for column, pixels in enumerate((image, labels, overlay)):
			x = margin + column * (panel_width + margin)
			panel = Image.fromarray(pixels).resize(
				(panel_width, panel_height), Image.Resampling.NEAREST
			)
			board.paste(panel, (x, y + row_label))
	output = Path(output_path).expanduser().resolve()
	output.parent.mkdir(parents=True, exist_ok=True)
	board.save(output)
	return output


def _parse_args(argv=None):
	parser = argparse.ArgumentParser(
		description="Validate a Cutie support pack's same-camera contract."
	)
	parser.add_argument("annotations", type=Path)
	parser.add_argument("--task")
	parser.add_argument("--expected-camera", type=int)
	parser.add_argument(
		"--allow-legacy-single-camera-field",
		action="store_true",
		help="audit old metadata without certifying an explicit camera contract",
	)
	parser.add_argument("--report", type=Path)
	parser.add_argument("--contact-sheet", type=Path)
	return parser.parse_args(argv), parser


def main(argv=None) -> None:
	args, parser = _parse_args(argv)
	try:
		summary = validate_support_pack(
			args.annotations,
			expected_task=args.task,
			expected_camera_id=args.expected_camera,
			allow_legacy_single_camera_field=args.allow_legacy_single_camera_field,
		)
		if args.contact_sheet is not None:
			summary["contact_sheet"] = str(
				write_contact_sheet(args.annotations, args.contact_sheet)
			)
	except (OSError, SupportCameraValidationError, ValueError) as exc:
		parser.exit(2, f"CUTIE_SUPPORT_CAMERA_FAILED: {exc}\n")
	text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
	if args.report is not None:
		report = args.report.expanduser().resolve()
		report.parent.mkdir(parents=True, exist_ok=True)
		report.write_text(text, encoding="utf-8")
	print("CUTIE_SUPPORT_CAMERA_OK")
	print(text, end="")


if __name__ == "__main__":
	main()
