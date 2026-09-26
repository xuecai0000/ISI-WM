"""Immutable input snapshot for the object-graph tokenizer preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from tdmpc2.common.unified_vos import (
    validate_backend_inputs,
    validate_dataset_files,
)
from tdmpc2.tools.aggregate_unified_vos_benchmark import (
    _validate_backend_manifest,
    _validate_decoded_dataset_artifacts,
)


FORMAT = "object_graph_tokenizer_preflight_inputs_v1"
LOCAL_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".sh", ".xml"}
CUTIE_SUFFIXES = {".py", ".yaml", ".yml"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _reject_symlink_components(
    path: Path, label: str, *, containment_root: Path | None = None
) -> Path:
    """Return an absolute lexical path after rejecting symlink components."""
    original = path.expanduser().absolute()
    if containment_root is None:
        components = [*reversed(original.parents), original]
    else:
        root = containment_root.expanduser().absolute()
        try:
            relative = original.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{label} escapes its immutable root: {original}") from exc
        components = [root]
        cursor = root
        for part in relative.parts:
            cursor = cursor / part
            components.append(cursor)
    for component in components:
        if component.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {component}")
    return original


def _file(
    path: Path,
    label: str,
    *,
    allow_symlink: bool = False,
    containment_root: Path | None = None,
) -> dict[str, Any]:
    original = path.expanduser().absolute()
    if not allow_symlink:
        _reject_symlink_components(
            original, label, containment_root=containment_root
        )
    if not original.exists():
        raise FileNotFoundError(f"{label}: {original}")
    if original.is_symlink() and not allow_symlink:
        raise ValueError(f"{label} cannot be a symlink: {original}")
    resolved = original.resolve(strict=True)
    if containment_root is not None:
        root = containment_root.expanduser().resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{label} resolves outside its immutable root.") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} must resolve to a regular file: {resolved}")
    before = (resolved.stat().st_size, _sha256(resolved))
    after = (resolved.stat().st_size, _sha256(resolved))
    if before != after:
        raise RuntimeError(f"{label} changed while being snapshotted.")
    return {"path": str(resolved), "bytes": before[0], "sha256": before[1]}


def _tree(root: Path, suffixes: set[str], label: str) -> dict[str, Any]:
    original = _reject_symlink_components(root, label)
    root = original.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"{label} must be a directory: {root}")
    entries = sorted(root.rglob("*"))
    for path in entries:
        if path.is_symlink():
            raise ValueError(f"{label} tree symlink is not allowed: {path}")
    candidates = sorted(
        path
        for path in entries
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix.lower() in suffixes
    )
    if not candidates:
        raise ValueError(f"{label} tree is empty: {root}")
    rows = []
    for path in candidates:
        before = (path.stat().st_size, _sha256(path))
        after = (path.stat().st_size, _sha256(path))
        if before != after:
            raise RuntimeError(f"{label} source changed while being snapshotted: {path}")
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": before[0],
            "sha256": before[1],
        })
    if [row["path"] for row in rows] != sorted(row["path"] for row in rows):
        raise AssertionError("Internal tree ordering failure.")
    return {
        "root": str(root),
        "suffixes": sorted(suffixes),
        "files": rows,
        "tree_sha256": hashlib.sha256(_canonical(rows)).hexdigest(),
    }


_ENVIRONMENT_SCRIPT = r'''
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys

distributions = []
for dist in metadata.distributions():
    name = dist.metadata.get("Name") or ""
    version = dist.version or ""
    if name:
        distributions.append([name.lower(), version])
distributions.sort()
try:
    import torch
    torch_record = {
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
except Exception as exc:
    torch_record = {"error_type": type(exc).__name__, "error": str(exc)}
print(json.dumps({
    "sys_executable": str(Path(sys.executable).resolve()),
    "sys_version": sys.version,
    "sys_prefix": str(Path(sys.prefix).resolve()),
    "platform": platform.platform(),
    "distributions": distributions,
    "torch": torch_record,
}, sort_keys=True, allow_nan=False))
'''


def _environment(python: Path) -> dict[str, Any]:
    executable = _file(python, "Cutie Python executable", allow_symlink=True)
    completed = subprocess.run(
        [str(python), "-I", "-c", _ENVIRONMENT_SCRIPT],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("Cutie environment probe returned unexpected output.")
    payload = json.loads(lines[0])
    if not isinstance(payload, dict) or payload.get("sys_executable") != executable["path"]:
        raise RuntimeError("Cutie environment executable identity mismatch.")
    return {"executable": executable, "environment": payload}


def build(
    *, source_root: Path, python: Path, oc_repo: Path, checkpoint: Path
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    package_root = project_root / "tdmpc2"
    source_root = _reject_symlink_components(
        source_root, "source benchmark root"
    ).resolve(strict=True)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    source_files = {
        "summary": source_root / "unified_vos_summary.json",
        "dataset_manifest": source_root / "dataset" / "dataset_manifest.json",
        "backend_inputs": source_root / "dataset" / "backend_inputs.json",
        "worker_backend_inputs": source_root / "worker_inputs" / "backend_inputs.json",
        "cutie_manifest": source_root / "backends" / "cutie" / "backend_predictions.json",
    }
    oc_repo = _reject_symlink_components(
        oc_repo, "OC-STORM repository"
    ).resolve(strict=True)
    cutie_root = oc_repo / "feature_extractor" / "cutie" / "cutie"
    source_records = {
        name: _file(
            path,
            f"source {name}",
            containment_root=source_root,
        )
        for name, path in source_files.items()
    }
    dataset = validate_dataset_files(
        source_files["dataset_manifest"], strict_counts=True
    )
    _validate_decoded_dataset_artifacts(
        dataset, source_files["dataset_manifest"].parent
    )
    worker_inputs, _, _ = validate_backend_inputs(
        source_files["worker_backend_inputs"], strict_counts=True
    )
    baseline_manifest, baseline_prediction_paths = _validate_backend_manifest(
        source_files["cutie_manifest"], backend="cutie", dataset=dataset
    )
    if (
        worker_inputs.get("dataset_id") != dataset.get("dataset_id")
        or baseline_manifest.get("dataset_id") != dataset.get("dataset_id")
    ):
        raise RuntimeError("Frozen source artifact identities differ.")
    if source_records["backend_inputs"]["sha256"] != source_records[
        "worker_backend_inputs"
    ]["sha256"]:
        raise RuntimeError("Published and worker-view backend input manifests differ.")
    local_source = _tree(package_root, LOCAL_SUFFIXES, "local source")
    cutie_source = _tree(cutie_root, CUTIE_SUFFIXES, "Cutie source")
    checkpoint_record = _file(checkpoint, "Cutie checkpoint")
    environment_record = _environment(python)

    try:
        baseline = json.loads(
            source_files["cutie_manifest"].read_text(encoding="utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Frozen Cutie baseline manifest is invalid.") from exc
    baseline_provenance = baseline.get("backend_provenance")
    if (
        baseline.get("status") != "complete"
        or baseline.get("backend") != "cutie"
        or not isinstance(baseline_provenance, dict)
    ):
        raise RuntimeError("Frozen Cutie baseline is incomplete.")
    expected_baseline = {
        "model_size": "small",
        "tracker_size": [448, 448],
        "checkpoint_sha256": checkpoint_record["sha256"],
        "seed": 2718281,
        "amp": True,
    }
    for name, expected in expected_baseline.items():
        if baseline_provenance.get(name) != expected:
            raise RuntimeError(
                f"Frozen Cutie baseline {name} differs from this preflight."
            )
    baseline_implementation = baseline_provenance.get("implementation")
    if (
        not isinstance(baseline_implementation, dict)
        or baseline_implementation.get("files") != cutie_source["files"]
    ):
        raise RuntimeError("Frozen Cutie implementation differs from the live tree.")
    probed_environment = environment_record["environment"]
    probed_torch = probed_environment.get("torch")
    if (
        probed_environment.get("sys_version") != baseline_provenance.get("python")
        or not isinstance(probed_torch, dict)
        or probed_torch.get("version") != baseline_provenance.get("torch")
        or probed_torch.get("cuda_runtime") != baseline_provenance.get("cuda_runtime")
    ):
        raise RuntimeError("Frozen Cutie Python/Torch/CUDA environment changed.")
    return {
        "format": FORMAT,
        "source_benchmark_root": str(source_root),
        "source_files": source_records,
        "source_artifact_contract": {
            "dataset_id": dataset["dataset_id"],
            "resolution": dataset["resolution"],
            "counts": dataset["counts"],
            "baseline_prediction_files": len(baseline_prediction_paths),
            "strict_counts": True,
            "decoded_dataset_arrays_validated": True,
            "worker_inputs_validated": True,
            "baseline_predictions_rehashed": True,
        },
        "baseline_cutie_contract": {
            **expected_baseline,
            "python": baseline_provenance["python"],
            "torch": baseline_provenance["torch"],
            "cuda_runtime": baseline_provenance["cuda_runtime"],
            "implementation_files": cutie_source["files"],
        },
        "local_source": local_source,
        "cutie_source": cutie_source,
        "cutie_checkpoint": checkpoint_record,
        "cutie_environment": environment_record,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--oc-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()
    payload = build(
        source_root=args.source_root,
        python=args.python,
        oc_repo=args.oc_repo,
        checkpoint=args.checkpoint,
    )
    if args.output is not None:
        path = args.output.expanduser().resolve()
        if path.exists() or not path.parent.is_dir():
            raise FileExistsError(path)
        path.write_bytes(_canonical(payload))
        print(json.dumps({"status": "bound", "path": str(path)}, allow_nan=False))
        return
    path = args.verify.expanduser().resolve(strict=True)
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing != payload or path.read_bytes() != _canonical(existing):
        raise RuntimeError("Object-graph immutable input snapshot changed.")
    print(json.dumps({"status": "verified", "path": str(path)}, allow_nan=False))


if __name__ == "__main__":
    main()
