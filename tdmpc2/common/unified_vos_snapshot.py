"""Immutable external-input snapshot for the unified VOS benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
from typing import Any

from tdmpc2.common.unified_vos import file_sha256, load_json, write_json


FORMAT = "unified_vos_external_inputs_v2"
CODE_SELECTION = "code_config_source_safe_file_symlinks_v2"
ALL_FILES_SELECTION = "all_regular_files_safe_file_symlinks_v2"
CODE_SUFFIXES = {
	".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp",
	".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml",
}
IGNORED_DIRS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache"}


def _regular_file_record(path: Path, root: Path) -> dict[str, Any]:
	before = path.stat()
	digest = file_sha256(path)
	after = path.stat()
	if (
		before.st_dev != after.st_dev
		or before.st_ino != after.st_ino
		or before.st_size != after.st_size
		or before.st_mtime_ns != after.st_mtime_ns
	):
		raise RuntimeError(f"Immutable input file changed while hashing: {path}")
	return {
		"path": path.relative_to(root).as_posix(),
		"kind": "regular_file",
		"bytes": after.st_size,
		"sha256": digest,
	}


def _symlink_file_record(path: Path, root: Path) -> dict[str, Any]:
	"""Bind one relative, in-tree symlink to one regular file.

	Directory symlinks and symlink chains are intentionally unsupported.  Keeping
	the accepted case this narrow makes both traversal and later provenance
	validation unambiguous.
	"""
	try:
		link_target = os.readlink(path)
	except OSError as exc:
		raise ValueError(f"Could not read input symlink: {path}") from exc
	if (
		not isinstance(link_target, str)
		or not link_target
		or Path(link_target).is_absolute()
		or "\\" in link_target
	):
		raise ValueError(f"Input file symlink target must be relative POSIX: {path}")

	# abspath normalizes '.' and '..' without dereferencing any path component.
	target = Path(os.path.abspath(os.fspath(path.parent / link_target)))
	try:
		target_relative = target.relative_to(root)
	except ValueError as exc:
		raise ValueError(f"Input file symlink escapes its immutable tree: {path}") from exc

	# Check every component lexically.  This rejects a target reached through a
	# directory symlink as well as a direct symlink-to-symlink chain.
	current = root
	for index, part in enumerate(target_relative.parts):
		current = current / part
		try:
			mode = current.lstat().st_mode
		except FileNotFoundError as exc:
			raise ValueError(f"Dangling input file symlink is not allowed: {path}") from exc
		if stat.S_ISLNK(mode):
			kind = "directory" if index < len(target_relative.parts) - 1 else "file chain"
			raise ValueError(f"Symlinked input {kind} is not allowed: {current}")
		if index < len(target_relative.parts) - 1 and not stat.S_ISDIR(mode):
			raise ValueError(f"Invalid input symlink target path: {path}")

	mode = target.lstat().st_mode
	if stat.S_ISDIR(mode):
		raise ValueError(f"Symlinked input directory is not allowed: {path}")
	if not stat.S_ISREG(mode):
		raise ValueError(f"Input file symlink target is not a regular file: {path}")
	before = target.stat()
	digest = file_sha256(target)
	after = target.stat()
	if (
		before.st_dev != after.st_dev
		or before.st_ino != after.st_ino
		or before.st_size != after.st_size
		or before.st_mtime_ns != after.st_mtime_ns
		or os.readlink(path) != link_target
	):
		raise RuntimeError(f"Immutable input symlink changed while hashing: {path}")
	return {
		"path": path.relative_to(root).as_posix(),
		"kind": "symlink_file",
		"link_target": link_target,
		"target_path": target_relative.as_posix(),
		"bytes": after.st_size,
		"sha256": digest,
	}


def _files(root: Path, *, code_only: bool) -> list[dict[str, Any]]:
	root = root.expanduser().resolve()
	if not root.is_dir():
		raise FileNotFoundError(root)
	rows: list[dict[str, Any]] = []
	symlink_rows: list[dict[str, Any]] = []
	for directory, names, filenames in os.walk(root, followlinks=False):
		base = Path(directory)
		for name in tuple(names):
			path = base / name
			if path.is_symlink():
				raise ValueError(f"Symlinked input directory is not allowed: {path}")
			if name in IGNORED_DIRS:
				names.remove(name)
		for name in filenames:
			path = base / name
			if path.is_symlink():
				record = _symlink_file_record(path, root)
				if not code_only or path.suffix.lower() in CODE_SUFFIXES:
					symlink_rows.append(record)
				continue
			if code_only and path.suffix.lower() not in CODE_SUFFIXES:
				continue
			if not path.is_file():
				continue
			rows.append(_regular_file_record(path, root))

	regular = {row["path"]: row for row in rows}
	for row in symlink_rows:
		target = regular.get(row["target_path"])
		if target is None:
			raise ValueError(
				f"Input symlink target is outside the selected file inventory: {row['path']}"
			)
		if row["bytes"] != target["bytes"] or row["sha256"] != target["sha256"]:
			raise RuntimeError(f"Input symlink target changed during inventory: {row['path']}")
	rows.extend(symlink_rows)
	rows.sort(key=lambda row: row["path"])
	if not rows:
		raise ValueError(f"Immutable input tree is empty: {root}")
	return rows


def _tree(root: Path, *, code_only: bool) -> dict[str, Any]:
	return {
		"root": str(root.expanduser().resolve()),
		"selection": CODE_SELECTION if code_only else ALL_FILES_SELECTION,
		"files": _files(root, code_only=code_only),
	}


def _file(path: Path) -> dict[str, Any]:
	path = path.expanduser().resolve()
	if not path.is_file() or path.is_symlink() or path.stat().st_size < 1:
		raise FileNotFoundError(f"Required immutable file is missing/empty: {path}")
	return {
		"path": str(path),
		"bytes": path.stat().st_size,
		"sha256": file_sha256(path),
	}


def build(args) -> dict[str, Any]:
	repo = args.repo_root.expanduser().resolve()
	local_python = repo / "tdmpc2"
	config = local_python / "config.yaml"
	runner = local_python / "tools" / "run_unified_vos_benchmark.sh"
	return {
		"format": FORMAT,
		"trees": {
			"local_python": _tree(local_python, code_only=True),
			"video_hard": _tree(args.video_root, code_only=False),
			"background_manifests": _tree(args.manifest_dir, code_only=False),
			"cutie_source": _tree(
				args.oc_repo / "feature_extractor" / "cutie" / "cutie",
				code_only=True,
			),
			"sam21_source": _tree(args.sam21_repo, code_only=True),
			"sam31_source": _tree(args.sam31_repo, code_only=True),
		},
		"files": {
			"local_config": _file(config),
			"runner": _file(runner),
			"cutie_checkpoint": _file(args.cutie_checkpoint),
			"sam21_checkpoint": _file(args.sam21_checkpoint),
			"sam31_checkpoint": _file(args.sam31_checkpoint),
			"sam31_bpe": _file(args.sam31_bpe),
		},
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--repo-root", type=Path, required=True)
	parser.add_argument("--video-root", type=Path, required=True)
	parser.add_argument("--manifest-dir", type=Path, required=True)
	parser.add_argument("--oc-repo", type=Path, required=True)
	parser.add_argument("--cutie-checkpoint", type=Path, required=True)
	parser.add_argument("--sam21-repo", type=Path, required=True)
	parser.add_argument("--sam21-checkpoint", type=Path, required=True)
	parser.add_argument("--sam31-repo", type=Path, required=True)
	parser.add_argument("--sam31-checkpoint", type=Path, required=True)
	parser.add_argument("--sam31-bpe", type=Path, required=True)
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("--output", type=Path)
	group.add_argument("--verify", type=Path)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	payload = build(args)
	if args.output is not None:
		output = args.output.expanduser().resolve()
		if output.exists() or not output.parent.is_dir():
			raise FileExistsError(output)
		write_json(output, payload)
		print(f"UNIFIED_VOS_EXTERNAL_INPUTS_BOUND {output}", flush=True)
		return
	expected = load_json(args.verify.expanduser().resolve())
	if expected != payload:
		raise RuntimeError("Unified VOS immutable external inputs changed during the run.")
	print("UNIFIED_VOS_EXTERNAL_INPUTS_UNCHANGED", flush=True)


if __name__ == "__main__":
	main()
