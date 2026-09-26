"""Bind the Python environments used by the unified VOS benchmark.

The benchmark intentionally runs its four stages in independent Python
environments.  This module asks each target interpreter to describe itself,
then records the interpreter binary and its complete installed-distribution
inventory.  It never installs, imports, or downloads a benchmark dependency.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from tdmpc2.common.unified_vos import file_sha256, load_json, write_json


FORMAT = "unified_vos_environment_snapshot_v1"
ENVIRONMENT_LABELS = ("core", "cutie", "sam21", "sam31")
PROBE_TIMEOUT_SECONDS = 120

_PROBE = r"""
import importlib.metadata
import json
import platform
import sys

rows = []
for distribution in importlib.metadata.distributions():
	name = distribution.metadata.get("Name")
	version = distribution.version
	if not isinstance(name, str) or not name.strip():
		raise RuntimeError("Installed distribution has no non-empty Name metadata.")
	if not isinstance(version, str) or not version.strip():
		raise RuntimeError(f"Installed distribution {name!r} has no version.")
	rows.append({"name": name, "version": version})
rows.sort(key=lambda row: (row["name"].casefold(), row["name"], row["version"]))
payload = {
	"executable": sys.executable,
	"sys_version": sys.version,
	"sys_prefix": sys.prefix,
	"platform": platform.platform(),
	"distributions": rows,
}
print(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
"""


def _resolve_executable(value: str) -> Path:
	if not isinstance(value, str) or not value.strip():
		raise ValueError("Python executable must be a non-empty string.")
	raw = os.path.expanduser(value.strip())
	if Path(raw).is_absolute() or Path(raw).parent != Path("."):
		candidate = Path(raw)
	else:
		located = shutil.which(raw)
		if located is None:
			raise FileNotFoundError(f"Python executable is not on PATH: {raw}")
		candidate = Path(located)
	try:
		resolved = candidate.resolve(strict=True)
	except (FileNotFoundError, OSError) as error:
		raise FileNotFoundError(f"Python executable does not exist: {candidate}") from error
	if not resolved.is_file():
		raise FileNotFoundError(f"Python executable is not a regular file: {resolved}")
	return resolved


def _executable_file(path: Path) -> dict[str, Any]:
	stat = path.stat()
	if stat.st_size < 1:
		raise ValueError(f"Python executable is empty: {path}")
	return {
		"path": str(path),
		"bytes": stat.st_size,
		"sha256": file_sha256(path),
	}


def _validate_probe(value: Any, *, executable: Path) -> dict[str, Any]:
	if not isinstance(value, dict) or set(value) != {
		"executable",
		"sys_version",
		"sys_prefix",
		"platform",
		"distributions",
	}:
		raise RuntimeError("Target Python returned an invalid environment-probe schema.")
	reported = value["executable"]
	if not isinstance(reported, str) or not reported:
		raise RuntimeError("Target Python returned an invalid sys.executable.")
	try:
		reported_path = Path(reported).expanduser().resolve(strict=True)
	except (FileNotFoundError, OSError) as error:
		raise RuntimeError(
			f"Target Python reported a missing sys.executable: {reported}"
		) from error
	if reported_path != executable:
		raise RuntimeError(
			"Python launcher resolved to a different target executable: "
			f"requested={executable}, reported={reported_path}"
		)
	for key in ("sys_version", "sys_prefix", "platform"):
		if not isinstance(value[key], str) or not value[key]:
			raise RuntimeError(f"Target Python returned an invalid {key}.")
	distributions = value["distributions"]
	if not isinstance(distributions, list):
		raise RuntimeError("Target Python returned an invalid distribution inventory.")
	validated: list[dict[str, str]] = []
	for index, row in enumerate(distributions):
		if not isinstance(row, dict) or set(row) != {"name", "version"}:
			raise RuntimeError(f"Invalid installed distribution at index {index}.")
		if any(not isinstance(row[key], str) or not row[key] for key in row):
			raise RuntimeError(f"Invalid installed distribution at index {index}.")
		validated.append({"name": row["name"], "version": row["version"]})
	expected_order = sorted(
		validated,
		key=lambda row: (row["name"].casefold(), row["name"], row["version"]),
	)
	if validated != expected_order:
		raise RuntimeError("Target Python distribution inventory is not exactly sorted.")
	return {
		"sys_version": value["sys_version"],
		"sys_prefix": value["sys_prefix"],
		"platform": value["platform"],
		"distributions": validated,
	}


def _snapshot_environment(executable_value: str) -> dict[str, Any]:
	executable = _resolve_executable(executable_value)
	file_before = _executable_file(executable)
	try:
		completed = subprocess.run(
			[str(executable), "-I", "-c", _PROBE],
			check=False,
			capture_output=True,
			text=True,
			timeout=PROBE_TIMEOUT_SECONDS,
		)
	except (OSError, subprocess.SubprocessError) as error:
		raise RuntimeError(f"Could not probe target Python: {executable}") from error
	if completed.returncode != 0:
		raise RuntimeError(
			f"Target Python probe failed for {executable} with rc={completed.returncode}: "
			f"{completed.stderr.strip()}"
		)
	if completed.stderr:
		raise RuntimeError(
			f"Target Python probe emitted stderr for {executable}: "
			f"{completed.stderr.strip()}"
		)
	try:
		probe = json.loads(completed.stdout)
	except (json.JSONDecodeError, TypeError) as error:
		raise RuntimeError(f"Target Python returned invalid JSON: {executable}") from error
	identity = _validate_probe(probe, executable=executable)
	file_after = _executable_file(executable)
	if file_after != file_before:
		raise RuntimeError(f"Python executable changed while it was being probed: {executable}")
	return {
		"executable": file_after,
		"target": {
			"sys_version": identity["sys_version"],
			"sys_prefix": identity["sys_prefix"],
			"platform": identity["platform"],
		},
		"distributions": identity["distributions"],
	}


def _parse_environments(values: list[str]) -> dict[str, str]:
	parsed: dict[str, str] = {}
	for value in values:
		if "=" not in value:
			raise ValueError(
				f"Invalid --environment {value!r}; expected LABEL=PYTHON_EXECUTABLE."
			)
		label, executable = value.split("=", 1)
		label = label.strip()
		if label not in ENVIRONMENT_LABELS:
			raise ValueError(
				f"Unknown environment label {label!r}; expected {ENVIRONMENT_LABELS}."
			)
		if label in parsed:
			raise ValueError(f"Duplicate environment label: {label}")
		if not executable.strip():
			raise ValueError(f"Empty Python executable for environment: {label}")
		parsed[label] = executable.strip()
	missing = set(ENVIRONMENT_LABELS) - set(parsed)
	if missing:
		raise ValueError(f"Missing environment labels: {sorted(missing)}")
	return parsed


def build(environment_values: list[str]) -> dict[str, Any]:
	environments = _parse_environments(environment_values)
	return {
		"format": FORMAT,
		"environments": {
			label: _snapshot_environment(environments[label])
			for label in ENVIRONMENT_LABELS
		},
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--environment",
		action="append",
		required=True,
		metavar="LABEL=PYTHON_EXECUTABLE",
		help="Repeat exactly once for core, cutie, sam21, and sam31.",
	)
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("--output", type=Path)
	group.add_argument("--verify", type=Path)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	payload = build(args.environment)
	if args.output is not None:
		output = args.output.expanduser().resolve()
		if output.exists() or not output.parent.is_dir():
			raise FileExistsError(output)
		write_json(output, payload)
		print(f"UNIFIED_VOS_ENVIRONMENTS_BOUND {output}", flush=True)
		return
	expected = load_json(args.verify.expanduser().resolve())
	if expected != payload:
		raise RuntimeError("Unified VOS Python environments changed during the run.")
	print("UNIFIED_VOS_ENVIRONMENTS_UNCHANGED", flush=True)


if __name__ == "__main__":
	main()
