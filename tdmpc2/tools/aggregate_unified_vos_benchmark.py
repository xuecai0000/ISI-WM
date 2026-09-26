"""Strict model-neutral aggregation for the unified VOS benchmark."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import posixpath
import re
import stat
from typing import Any

import numpy as np

from tdmpc2.common.unified_vos import (
	BACKEND_FORMAT,
	BACKENDS,
	CONDITIONS,
	SUMMARY_FORMAT,
	TASK_ROLES,
	file_sha256,
	load_json,
	require_sha256,
	resolve_member,
	validate_dataset_files,
	write_json,
)
from tdmpc2.common.unified_vos_snapshot import (
	ALL_FILES_SELECTION,
	CODE_SELECTION,
	FORMAT as EXTERNAL_INPUTS_FORMAT,
)
from tdmpc2.common.unified_vos_environment_snapshot import (
	ENVIRONMENT_LABELS,
	FORMAT as ENVIRONMENT_INPUTS_FORMAT,
)


IOU_SUCCESS_THRESHOLD = 0.50
TOLERANCE_RADIUS_PIXELS = 2
MIN_VISIBLE_RECALL = 0.95
MIN_MEAN_IOU = 0.50
MIN_TOLERANT_F1 = 0.80
MIN_IDENTITY_ACCURACY = 0.95
MAX_FAILURE_BURST = 10
MAX_FALSE_POSITIVE_RATE = 0.05
MAX_ROLE_OVERLAP_RATE = 0.005
MAX_MEAN_RUNTIME_MS = 100.0
MAX_P95_RUNTIME_MS = 150.0
MAX_REGRESSION_VS_CUTIE = 0.02
ONLINE_BACKENDS = ("cutie", "sam21")
OPTIONAL_OFFLINE_BACKEND = "sam31"
EXPECTED_DEPLOYMENT_PROTOCOL = {
	"cutie": (True, True),
	"sam21": (True, True),
	"sam31": (False, False),
}
SCORING_ISOLATION_FORMAT = "unified_vos_scoring_isolation_gate_v1"
_MODE = re.compile(r"[0-7]{3}")
_UTC_SECOND = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def _relative_artifact(path: Path, root: Path, label: str) -> str:
	path = path.expanduser().resolve()
	root = root.expanduser().resolve()
	try:
		return path.relative_to(root).as_posix()
	except ValueError as exc:
		raise ValueError(f"{label} must be contained by the summary root: {path}") from exc


def _resolve_directory_member(root: Path, relative: Any, label: str) -> Path:
	if not isinstance(relative, str) or not relative or "\\" in relative:
		raise ValueError(f"{label} must be a non-empty POSIX relative path.")
	if Path(relative).is_absolute():
		raise ValueError(f"{label} must be relative to the summary root.")
	path = (root / relative).resolve()
	try:
		path.relative_to(root.resolve())
	except ValueError as exc:
		raise ValueError(f"{label} escapes the summary root: {relative!r}.") from exc
	if not path.is_dir():
		raise FileNotFoundError(path)
	return path


def _validate_external_snapshot(payload: dict[str, Any]) -> None:
	if set(payload) != {"format", "trees", "files"}:
		raise ValueError("External-input snapshot schema changed.")
	if payload.get("format") != EXTERNAL_INPUTS_FORMAT:
		raise ValueError("External-input snapshot format changed.")
	if set(payload.get("trees", {})) != {
		"local_python", "video_hard", "background_manifests",
		"cutie_source", "sam21_source", "sam31_source",
	}:
		raise ValueError("External-input tree snapshot is incomplete.")
	if set(payload.get("files", {})) != {
		"local_config", "runner", "cutie_checkpoint", "sam21_checkpoint",
		"sam31_checkpoint", "sam31_bpe",
	}:
		raise ValueError("External-input file snapshot is incomplete.")
	expected_selections = {
		"local_python": CODE_SELECTION,
		"video_hard": ALL_FILES_SELECTION,
		"background_manifests": ALL_FILES_SELECTION,
		"cutie_source": CODE_SELECTION,
		"sam21_source": CODE_SELECTION,
		"sam31_source": CODE_SELECTION,
	}
	for name, record in payload["trees"].items():
		if not isinstance(record, dict) or set(record) != {"root", "selection", "files"}:
			raise ValueError(f"External-input tree {name} is malformed.")
		root = record["root"]
		if not isinstance(root, str) or not root or not Path(root).expanduser().is_absolute():
			raise ValueError(f"External-input tree root {name} is malformed.")
		if record["selection"] != expected_selections[name]:
			raise ValueError(f"External-input tree selection {name} changed.")
		files = record["files"]
		if not isinstance(files, list) or not files:
			raise ValueError(f"External-input tree {name} is empty.")

		regular: dict[str, dict[str, Any]] = {}
		symlinks: list[dict[str, Any]] = []
		paths: list[str] = []
		for item in files:
			if not isinstance(item, dict):
				raise ValueError(f"External-input tree row {name} is malformed.")
			path = item.get("path")
			if (
				not isinstance(path, str)
				or not path
				or "\\" in path
				or posixpath.isabs(path)
				or posixpath.normpath(path) != path
				or path == ".."
				or path.startswith("../")
			):
				raise ValueError(f"External-input tree path {name} is malformed.")
			if type(item.get("bytes")) is not int or item["bytes"] < 0:
				raise ValueError(f"External-input tree size {name}/{path} is malformed.")
			require_sha256(item.get("sha256"), f"external tree {name}/{path} SHA")
			kind = item.get("kind")
			if kind == "regular_file":
				if set(item) != {"path", "kind", "bytes", "sha256"}:
					raise ValueError(f"External regular-file row {name}/{path} is malformed.")
				regular[path] = item
			elif kind == "symlink_file":
				if set(item) != {
					"path", "kind", "link_target", "target_path", "bytes", "sha256"
				}:
					raise ValueError(f"External symlink row {name}/{path} is malformed.")
				symlinks.append(item)
			else:
				raise ValueError(f"External-input tree row kind {name}/{path} changed.")
			paths.append(path)
		if paths != sorted(paths) or len(paths) != len(set(paths)):
			raise ValueError(f"External-input tree paths {name} are not sorted/unique.")

		for item in symlinks:
			path = item["path"]
			link_target = item["link_target"]
			target_path = item["target_path"]
			if (
				not isinstance(link_target, str)
				or not link_target
				or "\\" in link_target
				or posixpath.isabs(link_target)
			):
				raise ValueError(f"External symlink target {name}/{path} is malformed.")
			resolved = posixpath.normpath(
				posixpath.join(posixpath.dirname(path), link_target)
			)
			if (
				not isinstance(target_path, str)
				or resolved != target_path
				or resolved == ".."
				or resolved.startswith("../")
			):
				raise ValueError(f"External symlink resolution {name}/{path} is malformed.")
			target = regular.get(target_path)
			if target is None:
				raise ValueError(f"External symlink target {name}/{path} is not regular.")
			if item["bytes"] != target["bytes"] or item["sha256"] != target["sha256"]:
				raise ValueError(f"External symlink target digest {name}/{path} disagrees.")

	for name, record in payload["files"].items():
		if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
			raise ValueError(f"External-input file {name} is malformed.")
		path = record["path"]
		if not isinstance(path, str) or not path or not Path(path).expanduser().is_absolute():
			raise ValueError(f"External-input file path {name} is malformed.")
		if type(record.get("bytes")) is not int or record["bytes"] < 1:
			raise ValueError(f"External-input file size {name} is malformed.")
		require_sha256(record.get("sha256"), f"external file {name} SHA")


def _validate_environment_snapshot(payload: dict[str, Any]) -> None:
	if set(payload) != {"format", "environments"}:
		raise ValueError("Python-environment snapshot schema changed.")
	if payload.get("format") != ENVIRONMENT_INPUTS_FORMAT:
		raise ValueError("Python-environment snapshot format changed.")
	environments = payload.get("environments")
	if not isinstance(environments, dict) or tuple(environments) != ENVIRONMENT_LABELS:
		raise ValueError("Python-environment labels/order changed.")
	for label in ENVIRONMENT_LABELS:
		record = environments[label]
		if not isinstance(record, dict) or set(record) != {
			"executable", "target", "distributions"
		}:
			raise ValueError(f"Environment snapshot {label} is malformed.")
		executable = record["executable"]
		if not isinstance(executable, dict) or set(executable) != {
			"path", "bytes", "sha256"
		}:
			raise ValueError(f"Environment executable {label} is malformed.")
		raw_path = executable.get("path")
		if not isinstance(raw_path, str) or not raw_path:
			raise ValueError(f"Environment executable path {label} is missing.")
		path = Path(raw_path).expanduser()
		if not path.is_absolute():
			raise ValueError(f"Environment executable path {label} is not absolute.")
		path = path.resolve(strict=True)
		if str(path) != raw_path or not path.is_file():
			raise ValueError(f"Environment executable path {label} is not canonical.")
		if type(executable.get("bytes")) is not int or executable["bytes"] < 1:
			raise ValueError(f"Environment executable size {label} is invalid.")
		if path.stat().st_size != executable["bytes"]:
			raise ValueError(f"Environment executable size changed for {label}.")
		if file_sha256(path) != require_sha256(
			executable.get("sha256"), f"environment executable {label} SHA"
		):
			raise ValueError(f"Environment executable changed for {label}.")
		target = record["target"]
		if not isinstance(target, dict) or set(target) != {
			"sys_version", "sys_prefix", "platform"
		}:
			raise ValueError(f"Environment target {label} is malformed.")
		if any(not isinstance(target[key], str) or not target[key] for key in target):
			raise ValueError(f"Environment target {label} has an empty field.")
		distributions = record["distributions"]
		if not isinstance(distributions, list):
			raise ValueError(f"Environment distribution inventory {label} is malformed.")
		validated = []
		for index, row in enumerate(distributions):
			if not isinstance(row, dict) or set(row) != {"name", "version"}:
				raise ValueError(
					f"Environment distribution {label}[{index}] is malformed."
				)
			if any(not isinstance(row[key], str) or not row[key] for key in row):
				raise ValueError(
					f"Environment distribution {label}[{index}] has an empty field."
				)
			validated.append({"name": row["name"], "version": row["version"]})
		expected = sorted(
			validated,
			key=lambda row: (row["name"].casefold(), row["name"], row["version"]),
		)
		if validated != expected:
			raise ValueError(f"Environment distributions are not sorted for {label}.")


def _mode(path: Path) -> str:
	return f"{stat.S_IMODE(path.stat().st_mode):03o}"


def _parse_utc_second(value: Any, label: str) -> datetime:
	if not isinstance(value, str) or _UTC_SECOND.fullmatch(value) is None:
		raise ValueError(f"{label} must be a UTC timestamp with whole-second precision.")
	return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def _validate_scoring_isolation_gate(
	path: Path,
	*,
	summary_root: Path,
	dataset_root: Path,
	dataset_manifest: Path,
	dataset: dict[str, Any],
) -> dict[str, Any]:
	path = path.expanduser().resolve()
	if _relative_artifact(path, summary_root, "scoring-isolation gate") != (
		"provenance/scoring_isolation_gate.json"
	):
		raise ValueError("Scoring-isolation gate is not in its fixed provenance location.")
	payload = load_json(path)
	if set(payload) != {
		"format", "status", "roots", "worker_view", "scoring_lock",
		"timing_utc", "dataset_manifest_sha256",
	}:
		raise ValueError("Scoring-isolation gate schema changed.")
	if (
		payload.get("format") != SCORING_ISOLATION_FORMAT
		or payload.get("status") != "complete"
	):
		raise ValueError("Scoring-isolation gate is incomplete.")
	if payload.get("dataset_manifest_sha256") != file_sha256(dataset_manifest):
		raise ValueError("Scoring-isolation gate names a different dataset manifest.")

	roots = payload.get("roots")
	if not isinstance(roots, dict) or set(roots) != {
		"dataset_resolved_at_execution", "worker_resolved_at_execution",
		"scoring_resolved_at_execution", "dataset_worker_disjoint",
		"dataset_contains_worker", "worker_contains_dataset",
		"dataset_relative_to_summary_root", "worker_relative_to_summary_root",
		"scoring_relative_to_summary_root",
	}:
		raise ValueError("Scoring-isolation root evidence is malformed.")
	execution_paths = []
	for key in (
		"dataset_resolved_at_execution", "worker_resolved_at_execution",
		"scoring_resolved_at_execution",
	):
		value = roots.get(key)
		if not isinstance(value, str) or not value or not Path(value).is_absolute():
			raise ValueError(f"Scoring-isolation historical root {key} is invalid.")
		execution_paths.append(Path(value))
	execution_dataset, execution_worker, execution_scoring = execution_paths
	execution_dataset_contains_worker = execution_dataset in execution_worker.parents
	execution_worker_contains_dataset = execution_worker in execution_dataset.parents
	execution_disjoint = (
		execution_dataset != execution_worker
		and not execution_dataset_contains_worker
		and not execution_worker_contains_dataset
	)
	if execution_scoring != execution_dataset / "scoring":
		raise ValueError("Historical scoring root was not below the historical dataset root.")
	dataset_recorded = _resolve_directory_member(
		summary_root, roots["dataset_relative_to_summary_root"], "dataset root"
	)
	worker_root = _resolve_directory_member(
		summary_root, roots["worker_relative_to_summary_root"], "worker root"
	)
	scoring_root = _resolve_directory_member(
		summary_root, roots["scoring_relative_to_summary_root"], "scoring root"
	)
	if dataset_recorded != dataset_root.resolve() or scoring_root != (dataset_root / "scoring").resolve():
		raise ValueError("Scoring-isolation roots do not match the aggregated dataset.")
	dataset_contains_worker = dataset_recorded in worker_root.parents
	worker_contains_dataset = worker_root in dataset_recorded.parents
	disjoint = (
		dataset_recorded != worker_root
		and not dataset_contains_worker
		and not worker_contains_dataset
	)
	if (
		roots["dataset_worker_disjoint"] is not disjoint
		or roots["dataset_contains_worker"] is not dataset_contains_worker
		or roots["worker_contains_dataset"] is not worker_contains_dataset
		or roots["dataset_worker_disjoint"] is not execution_disjoint
		or roots["dataset_contains_worker"] is not execution_dataset_contains_worker
		or roots["worker_contains_dataset"] is not execution_worker_contains_dataset
		or not disjoint
	):
		raise ValueError("Dataset and backend-worker roots are not disjoint.")

	worker = payload.get("worker_view")
	if not isinstance(worker, dict) or set(worker) != {
		"backend_inputs_relative", "backend_inputs_sha256",
		"root_mode_after_readonly", "backend_inputs_mode_after_readonly",
		"scoring_entry_absent",
	}:
		raise ValueError("Scoring-isolation worker-view evidence is malformed.")
	worker_manifest = resolve_member(
		worker_root, worker["backend_inputs_relative"], "worker backend inputs"
	)
	worker_sha = file_sha256(worker_manifest)
	if worker_sha != require_sha256(
		worker.get("backend_inputs_sha256"), "worker backend-input SHA"
	) or worker_sha != dataset["backend_inputs"]["sha256"]:
		raise ValueError("Worker backend-input manifest differs from the dataset binding.")
	for key in ("root_mode_after_readonly", "backend_inputs_mode_after_readonly"):
		if not isinstance(worker.get(key), str) or _MODE.fullmatch(worker[key]) is None:
			raise ValueError(f"Invalid worker-view mode: {key}.")
	if (
		_mode(worker_root) != worker["root_mode_after_readonly"]
		or _mode(worker_manifest) != worker["backend_inputs_mode_after_readonly"]
		or int(worker["root_mode_after_readonly"], 8) & 0o222
		or int(worker["backend_inputs_mode_after_readonly"], 8) & 0o222
	):
		raise ValueError("Worker input view is no longer read-only.")
	actual_scoring_entries = [
		member for member in worker_root.rglob("*")
		if "scoring" in member.relative_to(worker_root).parts
	]
	if worker.get("scoring_entry_absent") is not True or actual_scoring_entries:
		raise ValueError("Worker input view contains a scoring-named entry.")

	lock = payload.get("scoring_lock")
	if not isinstance(lock, dict) or set(lock) != {
		"tree_relative_to_dataset_root", "root_mode_before_lock",
		"root_mode_while_locked", "root_mode_after_restore",
		"probe_relative_to_dataset_root", "probe_mode_before_lock",
		"probe_mode_after_restore", "probe_sha256_before_lock",
		"probe_sha256_after_restore", "probe_readable_before_lock",
		"probe_readable_while_locked", "locked_read_probe_exit_code",
		"locked_read_probe_outcome",
		"locked_read_probe_log_relative_to_summary_root",
		"locked_read_probe_log_sha256",
	}:
		raise ValueError("Scoring-lock evidence is malformed.")
	if lock.get("tree_relative_to_dataset_root") != "scoring":
		raise ValueError("Scoring-lock gate names an unexpected tree.")
	for key in (
		"root_mode_before_lock", "root_mode_while_locked",
		"root_mode_after_restore", "probe_mode_before_lock",
		"probe_mode_after_restore",
	):
		if not isinstance(lock.get(key), str) or _MODE.fullmatch(lock[key]) is None:
			raise ValueError(f"Invalid scoring-lock mode: {key}.")
	if lock["root_mode_while_locked"] != "000":
		raise ValueError("Scoring root was not mode 000 during backend inference.")
	if (
		int(lock["root_mode_before_lock"], 8) & 0o500 != 0o500
		or int(lock["probe_mode_before_lock"], 8) & 0o400 != 0o400
		or int(lock["root_mode_after_restore"], 8) & 0o700 != 0o700
		or int(lock["probe_mode_after_restore"], 8) & 0o600 != 0o600
	):
		raise ValueError("Scoring tree was not readable before or restored after locking.")
	probe = resolve_member(
		dataset_root, lock["probe_relative_to_dataset_root"], "scoring probe"
	)
	try:
		probe.relative_to(scoring_root)
	except ValueError as exc:
		raise ValueError("Scoring probe is outside the scoring-only tree.") from exc
	before_sha = require_sha256(
		lock.get("probe_sha256_before_lock"), "scoring probe pre-lock SHA"
	)
	after_sha = require_sha256(
		lock.get("probe_sha256_after_restore"), "scoring probe restored SHA"
	)
	if before_sha != after_sha or file_sha256(probe) != after_sha:
		raise ValueError("Scoring probe bytes changed across the permission lock.")
	if (
		_mode(scoring_root) != lock["root_mode_after_restore"]
		or _mode(probe) != lock["probe_mode_after_restore"]
	):
		raise ValueError("Scoring tree current modes differ from restored modes.")
	if (
		lock.get("probe_readable_before_lock") is not True
		or lock.get("probe_readable_while_locked") is not False
		or type(lock.get("locked_read_probe_exit_code")) is not int
		or lock["locked_read_probe_exit_code"] != 1
		or lock.get("locked_read_probe_outcome") != "permission_error_same_uid"
	):
		raise ValueError("Same-user locked scoring read probe did not fail.")
	probe_log = resolve_member(
		summary_root,
		lock["locked_read_probe_log_relative_to_summary_root"],
		"locked scoring read-probe log",
	)
	if file_sha256(probe_log) != require_sha256(
		lock.get("locked_read_probe_log_sha256"), "locked read-probe log SHA"
	):
		raise ValueError("Locked scoring read-probe log changed.")
	if b"PermissionError" not in probe_log.read_bytes():
		raise ValueError("Locked scoring read-probe log is not a permission failure.")

	timing = payload.get("timing_utc")
	if not isinstance(timing, dict) or set(timing) != {
		"lock_started", "read_probe_completed", "restored", "gate_written"
	}:
		raise ValueError("Scoring-isolation timing evidence is malformed.")
	timestamps = [_parse_utc_second(timing[key], key) for key in (
		"lock_started", "read_probe_completed", "restored", "gate_written"
	)]
	if timestamps != sorted(timestamps):
		raise ValueError("Scoring-isolation timestamps are out of order.")
	return payload


@dataclass
class RoleTotals:
	frames: int = 0
	gt_visible: int = 0
	pred_nonempty: int = 0
	visible_detected: int = 0
	false_positive: int = 0
	iou_sum_visible: float = 0.0
	tolerant_f1_sum_visible: float = 0.0
	identity_correct_visible: int = 0
	success_visible: int = 0
	centroid_error_sum_visible_detected: float = 0.0
	centroid_error_count: int = 0
	current_failure_burst: int = 0
	max_failure_burst: int = 0


def _array_trace(value: np.ndarray) -> str:
	array = np.ascontiguousarray(value)
	digest = hashlib.sha256()
	digest.update(str(array.dtype).encode("ascii"))
	digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
	digest.update(array.tobytes())
	return digest.hexdigest()


def _labeled_sequence_trace(label: bytes, value: np.ndarray) -> str:
	digest = hashlib.sha256()
	for item in value:
		array = np.ascontiguousarray(item)
		digest.update(label)
		digest.update(str(array.dtype).encode("ascii"))
		digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
		digest.update(array.tobytes())
	return digest.hexdigest()


def _load_npz(path: Path, expected_keys: set[str]) -> dict[str, np.ndarray]:
	with np.load(path, allow_pickle=False) as archive:
		if set(archive.files) != expected_keys:
			raise ValueError(f"Unexpected arrays in {path}: {archive.files}.")
		return {key: np.ascontiguousarray(archive[key]) for key in archive.files}


def _validate_decoded_dataset_artifacts(
	dataset: dict[str, Any], dataset_root: Path
) -> None:
	"""Decode scoring/RGB artifacts and independently recompute their traces."""
	resolution = int(dataset["resolution"])
	frames = int(dataset["counts"]["frames_per_episode"])
	actions_count = int(dataset["counts"]["actions_per_episode"])
	for task, roles in TASK_ROLES.items():
		clean_records = dataset["episodes"][task]["clean"]
		hard_records = dataset["episodes"][task]["hard"]
		for clean_record, hard_record in zip(clean_records, hard_records):
			decoded = {}
			for condition, record in (("clean", clean_record), ("hard", hard_record)):
				scoring_path = resolve_member(
					dataset_root, record["arrays"], "scoring arrays"
				)
				rgb_path = resolve_member(
					dataset_root, record["rgb_arrays"], "RGB arrays"
				)
				scoring = _load_npz(
					scoring_path, {"gt_indexed", "actions", "physics_states"}
				)
				rgb = _load_npz(rgb_path, {"rgb"})["rgb"]
				gt = scoring["gt_indexed"]
				actions = scoring["actions"]
				physics_states = scoring["physics_states"]
				if rgb.shape != (frames, resolution, resolution, 3) or rgb.dtype != np.uint8:
					raise ValueError(f"{task}/{condition}: frozen RGB schema changed.")
				if gt.shape != (frames, resolution, resolution) or gt.dtype != np.uint8:
					raise ValueError(f"{task}/{condition}: frozen GT schema changed.")
				if set(np.unique(gt).tolist()) - set(range(len(roles) + 1)):
					raise ValueError(f"{task}/{condition}: GT contains an unknown role ID.")
				if (
					actions.ndim != 2
					or actions.shape[0] != actions_count
					or not np.issubdtype(actions.dtype, np.floating)
					or not np.isfinite(actions).all()
				):
					raise ValueError(f"{task}/{condition}: frozen action schema changed.")
				if (
					physics_states.ndim != 2
					or physics_states.shape[0] != frames
					or not np.issubdtype(physics_states.dtype, np.floating)
					or not np.isfinite(physics_states).all()
				):
					raise ValueError(f"{task}/{condition}: frozen physics schema changed.")
				for label, value, expected in (
					(b"rgb", rgb, record["rgb_trace_sha256"]),
					(b"gt", gt, record["gt_trace_sha256"]),
					(b"action", actions, record["action_trace_sha256"]),
					(b"state", physics_states, record["physics_trace_sha256"]),
				):
					if _labeled_sequence_trace(label, value) != expected:
						raise ValueError(f"{task}/{condition}: decoded {label!r} trace changed.")
				visible = {
					role: int((gt == role_id).reshape(frames, -1).any(axis=1).sum())
					for role_id, role in enumerate(roles, start=1)
				}
				if visible != record["role_gt_visible_frames"]:
					raise ValueError(f"{task}/{condition}: GT visibility accounting changed.")
				decoded[condition] = (gt, actions, physics_states, rgb)
			clean_gt, clean_actions, clean_physics, clean_rgb = decoded["clean"]
			hard_gt, hard_actions, hard_physics, hard_rgb = decoded["hard"]
			if not (
				np.array_equal(clean_gt, hard_gt)
				and np.array_equal(clean_actions, hard_actions)
				and np.array_equal(clean_physics, hard_physics)
			):
				raise ValueError(f"{task}: decoded clean/hard trajectory pairing failed.")
			if np.array_equal(clean_rgb, hard_rgb):
				raise ValueError(f"{task}: hard background did not change decoded RGB.")


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
	if radius < 0 or mask.ndim != 2:
		raise ValueError("Dilation requires one 2-D mask and non-negative radius.")
	output = np.zeros_like(mask, dtype=np.bool_)
	height, width = mask.shape
	for dy in range(-radius, radius + 1):
		for dx in range(-radius, radius + 1):
			if dx * dx + dy * dy > radius * radius:
				continue
			source_y0 = max(0, -dy)
			source_y1 = min(height, height - dy)
			source_x0 = max(0, -dx)
			source_x1 = min(width, width - dx)
			target_y0 = source_y0 + dy
			target_y1 = source_y1 + dy
			target_x0 = source_x0 + dx
			target_x1 = source_x1 + dx
			output[target_y0:target_y1, target_x0:target_x1] |= mask[
				source_y0:source_y1, source_x0:source_x1
			]
	return output


def _tolerant_f1(predicted: np.ndarray, gt: np.ndarray) -> float:
	predicted_count = int(predicted.sum())
	gt_count = int(gt.sum())
	if not predicted_count or not gt_count:
		return 0.0
	gt_dilated = _dilate(gt, TOLERANCE_RADIUS_PIXELS)
	predicted_dilated = _dilate(predicted, TOLERANCE_RADIUS_PIXELS)
	precision = float(np.logical_and(predicted, gt_dilated).sum()) / predicted_count
	recall = float(np.logical_and(gt, predicted_dilated).sum()) / gt_count
	return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
	yy, xx = np.nonzero(mask)
	if yy.size == 0:
		return None
	return float(xx.mean()), float(yy.mean())


class MaskScorer:
	def __init__(self, roles: tuple[str, ...], resolution: int):
		self.roles = roles
		self.resolution = resolution
		self.totals = [RoleTotals() for _ in roles]
		self.frames = 0
		self.role_overlap_frames = 0
		self.role_swap_frames = 0
		self.runtime_ms: list[float] = []

	def begin_episode(self) -> None:
		for totals in self.totals:
			totals.current_failure_burst = 0

	def record(
		self,
		*,
		predicted: np.ndarray,
		gt_indexed: np.ndarray,
		runtime_ms: float,
	) -> None:
		role_count = len(self.roles)
		if predicted.shape != (role_count, self.resolution, self.resolution):
			raise ValueError(f"Predicted mask shape changed: {predicted.shape}.")
		if gt_indexed.shape != (self.resolution, self.resolution):
			raise ValueError(f"GT mask shape changed: {gt_indexed.shape}.")
		if not math.isfinite(float(runtime_ms)) or float(runtime_ms) <= 0:
			raise ValueError(f"Backend runtime must be finite and positive: {runtime_ms}.")
		predicted = np.asarray(predicted, dtype=np.bool_)
		if np.any(predicted.sum(axis=0) > 1):
			self.role_overlap_frames += 1
		gt_masks = np.stack(
			[gt_indexed == role_id for role_id in range(1, role_count + 1)]
		)
		intersections = np.zeros((role_count, role_count), dtype=np.int64)
		for pred_index in range(role_count):
			for gt_index in range(role_count):
				intersections[pred_index, gt_index] = int(
					np.logical_and(predicted[pred_index], gt_masks[gt_index]).sum()
				)
		if role_count == 2 and bool(gt_masks.reshape(role_count, -1).any(axis=1).all()):
			diagonal = int(intersections[0, 0] + intersections[1, 1])
			cross = int(intersections[0, 1] + intersections[1, 0])
			self.role_swap_frames += int(cross > diagonal and cross > 0)

		self.frames += 1
		self.runtime_ms.append(float(runtime_ms))
		diagonal_pixels = math.sqrt(2.0) * self.resolution
		for index, totals in enumerate(self.totals):
			pred = predicted[index]
			gt = gt_masks[index]
			pred_nonempty = bool(pred.any())
			gt_visible = bool(gt.any())
			totals.frames += 1
			totals.pred_nonempty += int(pred_nonempty)
			if not gt_visible:
				totals.false_positive += int(pred_nonempty)
				totals.current_failure_burst = 0
				continue
			totals.gt_visible += 1
			totals.visible_detected += int(pred_nonempty)
			intersection = int(intersections[index, index])
			union = int(np.logical_or(pred, gt).sum())
			iou = intersection / union if union else 0.0
			f1 = _tolerant_f1(pred, gt)
			other = np.delete(intersections[index], index)
			identity_correct = bool(
				intersection > 0 and (other.size == 0 or intersection > int(other.max()))
			)
			success = pred_nonempty and identity_correct and iou >= IOU_SUCCESS_THRESHOLD
			totals.iou_sum_visible += iou
			totals.tolerant_f1_sum_visible += f1
			totals.identity_correct_visible += int(identity_correct)
			totals.success_visible += int(success)
			pred_centroid = _centroid(pred)
			gt_centroid = _centroid(gt)
			if pred_centroid is not None and gt_centroid is not None:
				distance = math.hypot(
					pred_centroid[0] - gt_centroid[0],
					pred_centroid[1] - gt_centroid[1],
				) / diagonal_pixels
				totals.centroid_error_sum_visible_detected += distance
				totals.centroid_error_count += 1
			if success:
				totals.current_failure_burst = 0
			else:
				totals.current_failure_burst += 1
				totals.max_failure_burst = max(
					totals.max_failure_burst, totals.current_failure_burst
				)

	@staticmethod
	def _latency(values: list[float]) -> dict[str, float]:
		array = np.asarray(values, dtype=np.float64)
		if array.size == 0 or not np.isfinite(array).all():
			raise RuntimeError("Backend latency series is empty or non-finite.")
		return {
			"mean_ms": float(array.mean()),
			"median_ms": float(np.median(array)),
			"p95_ms": float(np.percentile(array, 95)),
			"max_ms": float(array.max()),
		}

	def summary(self) -> dict[str, Any]:
		if self.frames <= 0:
			raise RuntimeError("No VOS frames were scored.")
		per_role = {}
		for role, totals in zip(self.roles, self.totals):
			invisible = totals.frames - totals.gt_visible
			per_role[role] = {
				"frames": totals.frames,
				"gt_visible_frames": totals.gt_visible,
				"predicted_nonempty_rate": totals.pred_nonempty / totals.frames,
				"visible_recall": (
					totals.visible_detected / totals.gt_visible
					if totals.gt_visible else None
				),
				"false_positive_rate_when_gt_invisible": (
					totals.false_positive / invisible if invisible else None
				),
				"mean_iou_on_gt_visible_frames": (
					totals.iou_sum_visible / totals.gt_visible
					if totals.gt_visible else None
				),
				"tolerant_f1_radius_2_on_gt_visible_frames": (
					totals.tolerant_f1_sum_visible / totals.gt_visible
					if totals.gt_visible else None
				),
				"identity_accuracy_on_gt_visible_frames": (
					totals.identity_correct_visible / totals.gt_visible
					if totals.gt_visible else None
				),
				"success_at_iou_0_5_rate_on_gt_visible_frames": (
					totals.success_visible / totals.gt_visible
					if totals.gt_visible else None
				),
				"mean_centroid_error_fraction_of_diagonal": (
					totals.centroid_error_sum_visible_detected / totals.centroid_error_count
					if totals.centroid_error_count else None
				),
				"max_failure_burst_at_iou_0_5_or_identity": totals.max_failure_burst,
			}
		return {
			"frames": self.frames,
			"role_overlap_frames": self.role_overlap_frames,
			"role_overlap_frame_rate": self.role_overlap_frames / self.frames,
			"role_swap_frames": self.role_swap_frames,
			"role_swap_frame_rate": self.role_swap_frames / self.frames,
			"per_role": per_role,
			"latency": self._latency(self.runtime_ms),
		}


def _validate_backend_manifest(
	path: Path,
	*,
	backend: str,
	dataset: dict[str, Any],
) -> tuple[dict[str, Any], dict[tuple[str, str, int], Path]]:
	path = path.expanduser().resolve()
	payload = load_json(path)
	if set(payload) != {
		"format", "status", "backend", "dataset_id", "input_manifest_sha256",
		"protocol", "backend_provenance", "results",
	}:
		raise ValueError(f"{backend}: prediction manifest schema changed.")
	if (
		payload.get("format") != BACKEND_FORMAT
		or payload.get("status") != "complete"
		or payload.get("backend") != backend
		or payload.get("dataset_id") != dataset["dataset_id"]
	):
		raise ValueError(f"{backend}: prediction identity/status mismatch.")
	if payload.get("input_manifest_sha256") != dataset["backend_inputs"]["sha256"]:
		raise ValueError(f"{backend}: did not consume the frozen GT-free input manifest.")
	protocol = payload.get("protocol")
	if not isinstance(protocol, dict):
		raise ValueError(f"{backend}: protocol is missing.")
	for key, expected in {
		"support_only_prompts": True,
		"support_frames_replayed_each_episode": 6,
		"episode_first_frame_gt_prompt": False,
		"mid_episode_reprompt": False,
		"episode_ground_truth_read": False,
		"causal_frame_order": True,
		"model_internal_features_exported": False,
		"reported_confidence_cross_backend_comparable": False,
		"runtime_ms_semantics": (
			"episode_end_to_end_amortized_per_frame_excluding_npz_io_"
			"and_model_construction_v1"
		),
	}.items():
		if protocol.get(key) != expected:
			raise ValueError(f"{backend}: protocol {key} changed.")
	prompt_adapter = protocol.get("prompt_adapter")
	if not isinstance(prompt_adapter, str) or not prompt_adapter:
		raise ValueError(f"{backend}: prompt adapter provenance is missing.")
	if type(protocol.get("strict_source_pixel_access_gate")) is not bool:
		raise ValueError(f"{backend}: source-pixel causality gate is missing.")
	if type(protocol.get("online_deployment_eligible")) is not bool:
		raise ValueError(f"{backend}: deployment-eligibility flag is missing.")
	expected_online, expected_source_gate = EXPECTED_DEPLOYMENT_PROTOCOL[backend]
	if (
		protocol["online_deployment_eligible"] is not expected_online
		or protocol["strict_source_pixel_access_gate"] is not expected_source_gate
	):
		raise ValueError(
			f"{backend}: deployment/source-access classification changed."
		)
	if protocol["online_deployment_eligible"] and not protocol[
		"strict_source_pixel_access_gate"
	]:
		raise ValueError(f"{backend}: an offline full-video backend cannot be deployable.")
	provenance = payload.get("backend_provenance")
	if not isinstance(provenance, dict):
		raise ValueError(f"{backend}: provenance is missing.")
	for key in (
		"checkpoint_sha256", "device_name", "cuda_visible_devices", "gpu_uuid"
	):
		if not isinstance(provenance.get(key), str) or not provenance[key]:
			raise ValueError(f"{backend}: provenance {key} is missing.")
	if provenance.get("cuda_device_order") != "PCI_BUS_ID":
		raise ValueError(f"{backend}: CUDA_DEVICE_ORDER must be PCI_BUS_ID.")
	require_sha256(provenance["checkpoint_sha256"], f"{backend}.checkpoint_sha256")
	if provenance.get("logical_cuda_device") != 0:
		raise ValueError(f"{backend}: logical CUDA device must be zero.")

	results = payload.get("results")
	if not isinstance(results, dict) or set(results) != set(TASK_ROLES):
		raise ValueError(f"{backend}: result task set is incomplete.")
	root = path.parent
	paths = {}
	expected_episodes = dataset["counts"]["episodes"]
	expected_frames = dataset["counts"]["frames_per_episode"]
	for task in TASK_ROLES:
		condition_map = results[task]
		if not isinstance(condition_map, dict) or set(condition_map) != set(CONDITIONS):
			raise ValueError(f"{backend}/{task}: condition set is incomplete.")
		for condition in CONDITIONS:
			records = condition_map[condition]
			if not isinstance(records, list) or len(records) != expected_episodes:
				raise ValueError(f"{backend}/{task}/{condition}: episode count changed.")
			for episode_index, record in enumerate(records):
				if not isinstance(record, dict) or set(record) != {
					"episode_index", "frames", "prediction_arrays",
					"prediction_arrays_sha256", "predicted_mask_trace_sha256",
					"runtime_trace_sha256",
				}:
					raise ValueError(f"{backend}: prediction record schema changed.")
				if record["episode_index"] != episode_index or record["frames"] != expected_frames:
					raise ValueError(f"{backend}: prediction ordering/count changed.")
				prediction_path = resolve_member(
					root, record["prediction_arrays"], f"{backend} prediction arrays"
				)
				if file_sha256(prediction_path) != require_sha256(
					record["prediction_arrays_sha256"], f"{backend} prediction SHA"
				):
					raise ValueError(f"{backend}: prediction artifact changed.")
				require_sha256(record["predicted_mask_trace_sha256"], "mask trace")
				require_sha256(record["runtime_trace_sha256"], "runtime trace")
				paths[(task, condition, episode_index)] = prediction_path
	return payload, paths


def _validate_optional_failure(path: Path, summary_root: Path) -> dict[str, Any]:
	path = path.expanduser().resolve()
	_relative_artifact(path, summary_root, "SAM 3.1 optional-failure record")
	payload = load_json(path)
	if set(payload) != {
		"format", "status", "backend", "exit_code", "log",
		"log_sha256", "online_deployment_eligible", "recommendation",
	}:
		raise ValueError("SAM 3.1 optional-failure record schema changed.")
	if (
		payload.get("format") != "unified_vos_optional_backend_failure_v1"
		or payload.get("status") != "diagnostic_failed"
		or payload.get("backend") != OPTIONAL_OFFLINE_BACKEND
		or payload.get("online_deployment_eligible") is not False
		or type(payload.get("exit_code")) is not int
		or payload["exit_code"] <= 0
		or payload.get("recommendation") != "online_selection_continues_without_offline_diagnostic"
	):
		raise ValueError("SAM 3.1 optional-failure identity/status changed.")
	log_path = resolve_member(summary_root, payload.get("log"), "SAM 3.1 failure log")
	if file_sha256(log_path) != require_sha256(
		payload.get("log_sha256"), "SAM 3.1 failure log SHA"
	):
		raise ValueError("SAM 3.1 optional-failure log changed.")
	return payload


def _score_backend(
	*,
	backend: str,
	backend_payload: dict[str, Any],
	prediction_paths: dict[tuple[str, str, int], Path],
	dataset: dict[str, Any],
	dataset_root: Path,
) -> dict[str, Any]:
	resolution = int(dataset["resolution"])
	frames_per_episode = int(dataset["counts"]["frames_per_episode"])
	episodes = int(dataset["counts"]["episodes"])
	task_summaries = {}
	for task, roles in TASK_ROLES.items():
		task_summaries[task] = {}
		for condition in CONDITIONS:
			scorer = MaskScorer(roles, resolution)
			for episode_index in range(episodes):
				dataset_record = dataset["episodes"][task][condition][episode_index]
				gt_path = resolve_member(
					dataset_root, dataset_record["arrays"], "scoring arrays"
				)
				gt = _load_npz(
					gt_path, {"gt_indexed", "actions", "physics_states"}
				)["gt_indexed"]
				prediction = _load_npz(
					prediction_paths[(task, condition, episode_index)],
					{"predicted_masks", "reported_confidence", "reported_lost", "runtime_ms"},
				)
				masks = prediction["predicted_masks"]
				confidence = prediction["reported_confidence"]
				lost = prediction["reported_lost"]
				runtime = prediction["runtime_ms"]
				expected_mask_shape = (
					frames_per_episode, len(roles), resolution, resolution
				)
				if masks.shape != expected_mask_shape or masks.dtype != np.bool_:
					raise ValueError(f"{backend}: predicted mask schema changed.")
				if gt.shape != (frames_per_episode, resolution, resolution) or gt.dtype != np.uint8:
					raise ValueError("Frozen GT array schema changed.")
				if confidence.shape != (frames_per_episode, len(roles)):
					raise ValueError(f"{backend}: confidence shape changed.")
				if lost.shape != confidence.shape or lost.dtype != np.bool_:
					raise ValueError(f"{backend}: reported-lost schema changed.")
				if runtime.shape != (frames_per_episode,):
					raise ValueError(f"{backend}: runtime shape changed.")
				if not np.isfinite(confidence).all() or not np.isfinite(runtime).all():
					raise ValueError(f"{backend}: non-finite diagnostics.")
				record = backend_payload["results"][task][condition][episode_index]
				if _array_trace(masks) != record["predicted_mask_trace_sha256"]:
					raise ValueError(f"{backend}: decoded mask trace mismatch.")
				if _array_trace(runtime) != record["runtime_trace_sha256"]:
					raise ValueError(f"{backend}: decoded runtime trace mismatch.")
				scorer.begin_episode()
				for frame_index in range(frames_per_episode):
					scorer.record(
						predicted=masks[frame_index],
						gt_indexed=gt[frame_index],
						runtime_ms=float(runtime[frame_index]),
					)
			task_summaries[task][condition] = scorer.summary()
	return task_summaries


def _backend_quality(
	backend: str,
	metrics: dict[str, Any],
	cutie_metrics: dict[str, Any],
	backend_protocol: dict[str, Any],
) -> dict[str, Any]:
	cells = []
	checks = {}
	regression_checks = {}
	for task, roles in TASK_ROLES.items():
		for condition in CONDITIONS:
			summary = metrics[task][condition]
			cutie = cutie_metrics[task][condition]
			checks[f"{task}/{condition}/role_overlap"] = (
				summary["role_overlap_frame_rate"] <= MAX_ROLE_OVERLAP_RATE
			)
			checks[f"{task}/{condition}/role_swap"] = summary["role_swap_frames"] == 0
			checks[f"{task}/{condition}/latency"] = (
				summary["latency"]["mean_ms"] <= MAX_MEAN_RUNTIME_MS
			)
			checks[f"{task}/{condition}/latency_p95"] = (
				summary["latency"]["p95_ms"] <= MAX_P95_RUNTIME_MS
			)
			for role in roles:
				value = summary["per_role"][role]
				baseline = cutie["per_role"][role]
				false_positive = value["false_positive_rate_when_gt_invisible"]
				cell_checks = {
					"visible_recall": value["visible_recall"] is not None
					and value["visible_recall"] >= MIN_VISIBLE_RECALL,
					"mean_iou": value["mean_iou_on_gt_visible_frames"] is not None
					and value["mean_iou_on_gt_visible_frames"] >= MIN_MEAN_IOU,
					"tolerant_f1": value["tolerant_f1_radius_2_on_gt_visible_frames"] is not None
					and value["tolerant_f1_radius_2_on_gt_visible_frames"] >= MIN_TOLERANT_F1,
					"identity": value["identity_accuracy_on_gt_visible_frames"] is not None
					and value["identity_accuracy_on_gt_visible_frames"] >= MIN_IDENTITY_ACCURACY,
					"failure_burst": value["max_failure_burst_at_iou_0_5_or_identity"]
					<= MAX_FAILURE_BURST,
					"false_positive": false_positive is None
					or false_positive <= MAX_FALSE_POSITIVE_RATE,
				}
				for name, passed in cell_checks.items():
					checks[f"{task}/{condition}/{role}/{name}"] = bool(passed)
				quality = (
					0.30 * float(value["mean_iou_on_gt_visible_frames"] or 0.0)
					+ 0.25 * float(value["tolerant_f1_radius_2_on_gt_visible_frames"] or 0.0)
					+ 0.20 * float(value["visible_recall"] or 0.0)
					+ 0.15 * float(value["identity_accuracy_on_gt_visible_frames"] or 0.0)
					+ 0.10 * float(value["success_at_iou_0_5_rate_on_gt_visible_frames"] or 0.0)
				)
				cells.append({
					"task": task,
					"condition": condition,
					"role": role,
					"quality": quality,
				})
				if backend != "cutie":
					for name in (
						"visible_recall",
						"mean_iou_on_gt_visible_frames",
						"tolerant_f1_radius_2_on_gt_visible_frames",
						"identity_accuracy_on_gt_visible_frames",
					):
						baseline_name = name
						candidate_name = name
						regression_checks[f"{task}/{condition}/{role}/{name}"] = (
							float(value[candidate_name] or 0.0)
							>= float(baseline[baseline_name] or 0.0) - MAX_REGRESSION_VS_CUTIE
						)
	quality_values = [cell["quality"] for cell in cells]
	absolute_pass = all(checks.values())
	no_regression = all(regression_checks.values()) if regression_checks else True
	online_deployment_eligible = backend_protocol["online_deployment_eligible"]
	return {
		"absolute_checks": checks,
		"absolute_quality_pass": absolute_pass,
		"no_regression_vs_cutie_checks": regression_checks,
		"no_regression_vs_cutie": no_regression,
		"online_deployment_eligible": online_deployment_eligible,
		"controller_pilot_eligible": absolute_pass and online_deployment_eligible,
		"universal_eligible": absolute_pass and online_deployment_eligible,
		"worst_role_condition_quality": min(quality_values),
		"macro_role_condition_quality": sum(quality_values) / len(quality_values),
		"cells": cells,
	}


def aggregate(args) -> dict[str, Any]:
	summary_root = args.output.resolve().parent
	dataset = validate_dataset_files(args.dataset_manifest, strict_counts=True)
	dataset_root = args.dataset_manifest.resolve().parent
	_validate_decoded_dataset_artifacts(dataset, dataset_root)
	_relative_artifact(args.dataset_manifest, summary_root, "dataset manifest")
	external_inputs = load_json(args.external_inputs)
	_validate_external_snapshot(external_inputs)
	_relative_artifact(args.external_inputs, summary_root, "external-input snapshot")
	environment_inputs = load_json(args.environment_inputs)
	_validate_environment_snapshot(environment_inputs)
	if _relative_artifact(
		args.environment_inputs, summary_root, "Python-environment snapshot"
	) != "provenance/environment_inputs.json":
		raise ValueError(
			"Python-environment snapshot is not in its fixed provenance location."
		)
	if dataset.get("backend_inputs", {}).get("gt_paths_disclosed") is not False:
		raise ValueError("Dataset did not publish a GT-free backend input view.")
	backend_inputs_path = resolve_member(
		dataset_root, dataset["backend_inputs"]["path"], "backend_inputs"
	)
	if file_sha256(backend_inputs_path) != dataset["backend_inputs"]["sha256"]:
		raise ValueError("GT-free backend input manifest changed.")
	isolation_gate = _validate_scoring_isolation_gate(
		args.scoring_isolation,
		summary_root=summary_root,
		dataset_root=dataset_root,
		dataset_manifest=args.dataset_manifest,
		dataset=dataset,
	)

	manifest_paths: dict[str, Path] = {
		"cutie": args.cutie,
		"sam21": args.sam21,
	}
	if (args.sam31 is None) == (args.sam31_failure is None):
		raise ValueError("Provide exactly one of --sam31 or --sam31-failure.")
	optional_failure = None
	if args.sam31 is not None:
		manifest_paths[OPTIONAL_OFFLINE_BACKEND] = args.sam31
	else:
		optional_failure = _validate_optional_failure(args.sam31_failure, summary_root)
	available_backends = tuple(manifest_paths)
	backend_payloads = {}
	prediction_paths = {}
	for backend in available_backends:
		_relative_artifact(manifest_paths[backend], summary_root, f"{backend} manifest")
		backend_payloads[backend], prediction_paths[backend] = _validate_backend_manifest(
			manifest_paths[backend], backend=backend, dataset=dataset
		)
	checkpoint_snapshot_names = {
		"cutie": "cutie_checkpoint",
		"sam21": "sam21_checkpoint",
		"sam31": "sam31_checkpoint",
	}
	for backend, payload in backend_payloads.items():
		expected_sha = external_inputs["files"][checkpoint_snapshot_names[backend]]["sha256"]
		if payload["backend_provenance"]["checkpoint_sha256"] != expected_sha:
			raise ValueError(f"{backend}: checkpoint differs from the immutable snapshot.")
	device_names = {
		backend: payload["backend_provenance"]["device_name"]
		for backend, payload in backend_payloads.items()
	}
	if len(set(device_names.values())) != 1:
		raise ValueError(f"All backends must run on the same GPU model: {device_names}.")
	gpu_uuids = {
		backend: payload["backend_provenance"]["gpu_uuid"]
		for backend, payload in backend_payloads.items()
	}
	if len(set(gpu_uuids.values())) != 1:
		raise ValueError(f"All backends must run on the same physical GPU: {gpu_uuids}.")

	metrics = {}
	for backend in available_backends:
		metrics[backend] = _score_backend(
			backend=backend,
			backend_payload=backend_payloads[backend],
			prediction_paths=prediction_paths[backend],
			dataset=dataset,
			dataset_root=dataset_root,
		)
	quality = {
		backend: _backend_quality(
			backend,
			metrics[backend],
			metrics["cutie"],
			backend_payloads[backend]["protocol"],
		)
		for backend in available_backends
	}
	eligible = [
		backend for backend in ONLINE_BACKENDS
		if quality[backend]["universal_eligible"]
	]
	def ranking_key(backend: str):
		return (
			quality[backend]["worst_role_condition_quality"],
			quality[backend]["macro_role_condition_quality"],
			-metrics[backend]["acrobot-swingup"]["hard"]["latency"]["mean_ms"],
		)

	online_ranking = sorted(
		ONLINE_BACKENDS,
		key=ranking_key,
		reverse=True,
	)
	all_treatment_ranking = sorted(
		available_backends,
		key=ranking_key,
		reverse=True,
	)
	winner = next((backend for backend in online_ranking if backend in eligible), None)
	status = (
		"unified_vos_controller_pilot_go"
		if winner is not None
		else "unified_vos_controller_pilot_no_go"
	)
	sam31_protocol_completed = optional_failure is None
	payload = {
		"format": SUMMARY_FORMAT,
		"status": status,
		"engineering_pass": True,
		"controller_pilot_go": winner is not None,
		"scientific_selection_pass": False,
		"paper_claim_ready": False,
		"recommended_backend": winner,
		"recommendation": (
			f"integrate_{winner}_mask_frontend_then_run_one_controller_pilot"
			if winner is not None
			else "do_not_train_controller_improve_or_change_visual_frontend"
		),
		"scope": {
			"evidence_level": "single_seed_random_policy_preflight",
			"allowed_claim": "frontend_candidate_for_one_controller_pilot",
			"disallowed_claim": "scientific_or_paper_level_backend_winner",
			"single_backend_for_all_tasks": True,
			"task_specific_backend_selection_allowed": False,
			"tasks": list(TASK_ROLES),
			"conditions": list(CONDITIONS),
			"controller_training_steps": 0,
			"model_internal_features_compared": False,
			"selection_unit": "complete_deployable_backend_with_same_fixed_support_supervision",
			"fixed_support_source_assets_are_identical": True,
			"prompt_encodings_are_backend_specific": True,
			"sam31_receives_deterministic_points_derived_from_support_masks": (
				sam31_protocol_completed
			),
			"sam31_support_prompt_protocol_completed": sam31_protocol_completed,
			"raw_prompt_information_parity": False,
			"extra_sensor_information_vs_policy64": True,
			"fair_controller_or_representation_comparison": False,
			"offline_backends_are_reported_but_cannot_be_recommended": True,
			"online_selection_requires_backends": list(ONLINE_BACKENDS),
			"sam31_is_optional_offline_diagnostic": True,
			"sam31_resources_and_environment_required_by_this_runner": True,
			"sam31_execution_failure_does_not_block_online_selection": True,
			"scoring_isolation_gate_validated": isolation_gate["status"] == "complete",
			"worker_input_root_disjoint_from_dataset_root": isolation_gate["roots"][
				"dataset_worker_disjoint"
			],
			"worker_input_view_has_no_scoring_entry": isolation_gate["worker_view"][
				"scoring_entry_absent"
			],
			"scoring_tree_permissions_removed_during_backend_inference": (
				isolation_gate["scoring_lock"]["root_mode_while_locked"] == "000"
				and isolation_gate["scoring_lock"]["probe_readable_while_locked"] is False
			),
			"gt_access_claim": (
				"not_disclosed_and_same_user_locked_read_probe_denied_"
				"as_validated_by_scoring_isolation_gate"
			),
			"task_specific_role_ontology_required": True,
			"support_labels_per_task": 6,
			"simulator_support_labels_used": True,
			"two_roles_per_task": True,
			"fixed_camera": True,
			"trajectory_policy": "single_seed_uniform_random_actions",
			"high_motion_controller_trajectory_not_tested": True,
			"integration_changes_sensor_and_perception_budget": True,
		},
		"optional_diagnostics": {
			"sam31_status": "complete" if optional_failure is None else "diagnostic_failed",
			"sam31_support_prompt_protocol": (
				"completed" if sam31_protocol_completed
				else "not_completed_due_to_diagnostic_failure"
			),
			"sam31_failure": optional_failure,
		},
		"thresholds": {
			"iou_success": IOU_SUCCESS_THRESHOLD,
			"tolerant_f1_radius_pixels": TOLERANCE_RADIUS_PIXELS,
			"min_visible_recall": MIN_VISIBLE_RECALL,
			"min_mean_iou": MIN_MEAN_IOU,
			"min_tolerant_f1": MIN_TOLERANT_F1,
			"min_identity_accuracy": MIN_IDENTITY_ACCURACY,
			"max_failure_burst": MAX_FAILURE_BURST,
			"max_false_positive_rate": MAX_FALSE_POSITIVE_RATE,
			"max_role_overlap_rate": MAX_ROLE_OVERLAP_RATE,
			"max_mean_runtime_ms": MAX_MEAN_RUNTIME_MS,
			"max_p95_runtime_ms": MAX_P95_RUNTIME_MS,
			"max_regression_vs_cutie": MAX_REGRESSION_VS_CUTIE,
			"no_regression_is_pareto_diagnostic_not_go_gate": True,
		},
		"online_quality_ranking": online_ranking,
		"eligible_online_backends": [
			backend for backend in online_ranking if backend in eligible
		],
		"quality_ranking_all_treatments": all_treatment_ranking,
		"quality": quality,
		"metrics": metrics,
		"provenance": {
			"dataset_manifest_relative_to_summary_root": _relative_artifact(
				args.dataset_manifest, summary_root, "dataset manifest"
			),
			"dataset_manifest_sha256": file_sha256(args.dataset_manifest),
			"dataset_id": dataset["dataset_id"],
			"backend_inputs_sha256": dataset["backend_inputs"]["sha256"],
			"external_inputs_relative_to_summary_root": _relative_artifact(
				args.external_inputs, summary_root, "external-input snapshot"
			),
			"external_inputs_sha256": file_sha256(args.external_inputs),
			"environment_inputs_relative_to_summary_root": _relative_artifact(
				args.environment_inputs, summary_root, "Python-environment snapshot"
			),
			"environment_inputs_sha256": file_sha256(args.environment_inputs),
			"scoring_isolation_relative_to_summary_root": _relative_artifact(
				args.scoring_isolation, summary_root, "scoring-isolation gate"
			),
			"scoring_isolation_sha256": file_sha256(args.scoring_isolation),
			"backend_manifests": {
				backend: {
					"path_relative_to_summary_root": _relative_artifact(
						manifest_paths[backend], summary_root, f"{backend} manifest"
					),
					"sha256": file_sha256(manifest_paths[backend]),
					"checkpoint_sha256": backend_payloads[backend]["backend_provenance"]["checkpoint_sha256"],
					"prompt_adapter": backend_payloads[backend]["protocol"]["prompt_adapter"],
				}
				for backend in available_backends
			},
			"device_names": device_names,
			"gpu_uuids": gpu_uuids,
		},
	}
	write_json(args.output, payload)
	return payload


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--dataset-manifest", type=Path, required=True)
	parser.add_argument("--cutie", type=Path, required=True)
	parser.add_argument("--sam21", type=Path, required=True)
	parser.add_argument("--sam31", type=Path)
	parser.add_argument("--sam31-failure", type=Path)
	parser.add_argument("--external-inputs", type=Path, required=True)
	parser.add_argument("--environment-inputs", type=Path, required=True)
	parser.add_argument("--scoring-isolation", type=Path, required=True)
	parser.add_argument("--output", type=Path, required=True)
	return parser


def main() -> None:
	args = build_parser().parse_args()
	for name in (
		"dataset_manifest", "cutie", "sam21", "sam31", "sam31_failure",
		"external_inputs", "environment_inputs", "scoring_isolation", "output"
	):
		value = getattr(args, name)
		if value is not None:
			setattr(args, name, value.expanduser().resolve())
	if args.output.exists():
		raise FileExistsError(args.output)
	payload = aggregate(args)
	print(json.dumps({
		"status": payload["status"],
		"recommended_backend": payload["recommended_backend"],
		"online_quality_ranking": payload["online_quality_ranking"],
		"eligible_online_backends": payload["eligible_online_backends"],
		"quality_ranking_all_treatments": payload["quality_ranking_all_treatments"],
		"summary": str(args.output),
	}, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
	main()
