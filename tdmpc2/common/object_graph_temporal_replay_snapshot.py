"""Immutable inputs for the CPU-only object-graph temporal replay preflight.

The replay consumes two previously published results: the frozen unified-VOS
dataset and the v1 object-graph backend.  This snapshot validates every
referenced source artifact and records the exact local implementation and
Python environment.  Verification rebuilds the complete record instead of
trusting timestamps or manifests alone.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
from typing import Any, Iterable

from tdmpc2.common.unified_vos import (
    CONDITIONS,
    TASK_ROLES,
    file_sha256,
    load_json,
    require_sha256,
    validate_backend_inputs,
)
from tdmpc2.tools.aggregate_object_graph_tokenizer_preflight import (
    BACKEND_FORMAT as V1_BACKEND_FORMAT,
    BACKEND_NAME as V1_BACKEND_NAME,
    SUMMARY_FORMAT as V1_SUMMARY_FORMAT,
    _source_artifacts,
)
from tdmpc2.tools.aggregate_unified_vos_benchmark import (
    _validate_backend_manifest,
)
from tdmpc2.perception.support_conditioned_object_graph import load_object_graph


FORMAT = "object_graph_temporal_replay_inputs_v1"
LOCAL_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".sh", ".xml"}
V1_BACKEND_ARRAY_KEYS = {
    "role_masks",
    "entity_masks",
    "descriptors",
    "keypoints_xy",
    "role_valid",
    "role_confidence",
    "role_lost",
    "role_mask_score",
    "entity_valid",
    "entity_confidence",
    "entity_lost",
    "entity_mask_score",
    "cutie_runtime_ms",
    "parser_runtime_ms",
    "end_to_end_runtime_ms",
}
V1_TRACE_KEYS = {
    "role_mask_trace_sha256",
    "entity_mask_trace_sha256",
    "descriptor_trace_sha256",
    "keypoint_trace_sha256",
    "role_status_trace_sha256",
    "entity_status_trace_sha256",
    "cutie_runtime_trace_sha256",
    "parser_runtime_trace_sha256",
    "end_to_end_runtime_trace_sha256",
}


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_symlink_components(path: Path, label: str) -> Path:
    path = path.expanduser().absolute()
    for component in (*reversed(path.parents), path):
        if component.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {component}")
    return path


def _root(path: Path, label: str) -> Path:
    path = _reject_symlink_components(path, label).resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"{label} must be a directory: {path}")
    return path


def _contained_file(path: Path, root: Path, label: str) -> Path:
    lexical = _reject_symlink_components(path, label)
    root = root.resolve(strict=True)
    try:
        lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes {root}: {lexical}") from exc
    resolved = lexical.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} resolves outside {root}: {resolved}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file: {resolved}")
    return resolved


def _member(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"{label} must be a non-empty POSIX relative path.")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{label} is not a canonical contained path: {relative!r}")
    return _contained_file(root.joinpath(*pure.parts), root, label)


def _record(path: Path, root: Path, label: str, expected_sha: str | None = None) -> dict[str, Any]:
    path = _contained_file(path, root, label)
    before = path.stat()
    digest = _sha256(path)
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"{label} changed while being inventoried: {path}")
    if expected_sha is not None and digest != require_sha256(expected_sha, label):
        raise RuntimeError(f"{label} differs from its manifest digest: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": before.st_size,
        "sha256": digest,
    }


def _inventory(
    root: Path,
    entries: Iterable[tuple[Path, str, str | None]],
    label: str,
) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for path, item_label, expected_sha in entries:
        record = _record(path, root, item_label, expected_sha)
        previous = records.get(record["path"])
        if previous is not None and previous != record:
            raise RuntimeError(f"{label} contains conflicting duplicate paths.")
        records[record["path"]] = record
    rows = [records[name] for name in sorted(records)]
    if not rows:
        raise ValueError(f"{label} is empty.")
    return {
        "root": str(root),
        "files": rows,
        "inventory_sha256": hashlib.sha256(_canonical(rows)).hexdigest(),
    }


def _local_tree(root: Path) -> dict[str, Any]:
    root = _root(root, "local tdmpc2 source")
    paths = sorted(root.rglob("*"))
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Local source symlink is not allowed: {path}")
    entries = [
        (path, f"local source {path.relative_to(root).as_posix()}", None)
        for path in paths
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix.lower() in LOCAL_SUFFIXES
    ]
    result = _inventory(root, entries, "local source inventory")
    result["suffixes"] = sorted(LOCAL_SUFFIXES)
    return result


def _environment() -> dict[str, Any]:
    executable = Path(sys.executable).resolve(strict=True)
    distributions = sorted(
        [
            (dist.metadata.get("Name") or "").lower(),
            dist.version or "",
        ]
        for dist in metadata.distributions()
        if dist.metadata.get("Name")
    )
    try:
        import numpy as np

        numpy_version: str | None = np.__version__
    except Exception as exc:  # pragma: no cover - fail closed on the server
        raise RuntimeError("NumPy is required for temporal replay.") from exc
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {None, ""}:
        raise RuntimeError("Temporal replay snapshot must run with CUDA disabled.")
    return {
        "sys_executable": str(executable),
        "sys_version": sys.version,
        "sys_prefix": str(Path(sys.prefix).resolve()),
        "platform": platform.platform(),
        "numpy": numpy_version,
        "distributions": [list(item) for item in distributions],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _source_inventory(source_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_summary, dataset, dataset_path, baseline_path = _source_artifacts(source_root)
    baseline, baseline_paths = _validate_backend_manifest(
        baseline_path, backend="cutie", dataset=dataset
    )
    worker_path = source_root / "worker_inputs" / "backend_inputs.json"
    worker, support_paths, episode_paths = validate_backend_inputs(
        worker_path, strict_counts=True
    )
    if worker.get("dataset_id") != dataset.get("dataset_id"):
        raise RuntimeError("Source worker/dataset identity mismatch.")

    entries: list[tuple[Path, str, str | None]] = [
        (source_root / "unified_vos_summary.json", "source summary", None),
        (dataset_path, "source dataset manifest", None),
        (dataset_path.parent / "backend_inputs.json", "published backend inputs", None),
        (worker_path, "worker backend inputs", None),
        (baseline_path, "source Cutie manifest", None),
    ]
    for task, entry in dataset["support"].items():
        entries.append(
            (
                _member(dataset_path.parent, entry["arrays"], f"dataset support {task}"),
                f"dataset support {task}",
                entry["arrays_sha256"],
            )
        )
    for task, condition_map in dataset["episodes"].items():
        for condition, records in condition_map.items():
            for record in records:
                tag = f"{task}/{condition}/{record['episode_index']}"
                entries.extend(
                    [
                        (
                            _member(dataset_path.parent, record["arrays"], f"scoring {tag}"),
                            f"scoring {tag}",
                            record["arrays_sha256"],
                        ),
                        (
                            _member(dataset_path.parent, record["rgb_arrays"], f"dataset RGB {tag}"),
                            f"dataset RGB {tag}",
                            record["rgb_arrays_sha256"],
                        ),
                    ]
                )
    for task, path in support_paths.items():
        entries.append(
            (path, f"worker support {task}", worker["support"][task]["arrays_sha256"])
        )
    for (task, condition, episode), path in episode_paths.items():
        expected = worker["episodes"][task][condition][episode]["rgb_arrays_sha256"]
        entries.append((path, f"worker RGB {task}/{condition}/{episode}", expected))
    for (task, condition, episode), path in baseline_paths.items():
        expected = baseline["results"][task][condition][episode]["prediction_arrays_sha256"]
        entries.append((path, f"Cutie prediction {task}/{condition}/{episode}", expected))
    return source_summary, dataset, _inventory(source_root, entries, "source inventory")


def _validate_v1_summary(
    *, v1_root: Path, source_root: Path, source_summary: dict[str, Any], dataset: dict[str, Any]
) -> tuple[dict[str, Any], Path, Path]:
    summary_path = _contained_file(
        v1_root / "object_graph_tokenizer_summary.json", v1_root, "v1 summary"
    )
    summary = load_json(summary_path)
    if (
        summary.get("format") != V1_SUMMARY_FORMAT
        or summary.get("engineering_pass") is not True
        or summary.get("controller_training_authorized") is not False
        or summary.get("scientific_go") is not False
        or summary.get("scope", {}).get("controller_training_steps") != 0
    ):
        raise RuntimeError("The v1 object-graph preflight is not an eligible replay source.")
    provenance = summary.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("The v1 summary provenance is missing.")
    if (
        provenance.get("source_benchmark_root") != str(source_root)
        or provenance.get("source_dataset_id") != dataset.get("dataset_id")
        or provenance.get("source_summary_sha256")
        != file_sha256(source_root / "unified_vos_summary.json")
        or provenance.get("source_dataset_manifest_sha256")
        != file_sha256(source_root / "dataset" / "dataset_manifest.json")
        or provenance.get("source_cutie_manifest_sha256")
        != file_sha256(source_root / "backends" / "cutie" / "backend_predictions.json")
    ):
        raise RuntimeError("The v1 summary is bound to a different frozen source.")
    backend_path = _member(
        v1_root,
        provenance.get("object_graph_backend_manifest_relative_to_summary_root"),
        "v1 backend manifest",
    )
    if file_sha256(backend_path) != require_sha256(
        provenance.get("object_graph_backend_manifest_sha256"), "v1 backend manifest SHA"
    ):
        raise RuntimeError("The v1 backend manifest changed after publication.")
    isolation_path = _member(
        v1_root,
        provenance.get("scoring_isolation_relative_to_summary_root"),
        "v1 scoring isolation gate",
    )
    if file_sha256(isolation_path) != require_sha256(
        provenance.get("scoring_isolation_sha256"), "v1 isolation SHA"
    ):
        raise RuntimeError("The v1 scoring-isolation record changed after publication.")
    if source_summary.get("engineering_pass") is not True:
        raise RuntimeError("The frozen unified benchmark is not engineering-valid.")
    return summary, backend_path, isolation_path


def _v1_inventory(
    *, v1_root: Path, source_root: Path, source_summary: dict[str, Any], dataset: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary, backend_path, isolation_path = _validate_v1_summary(
        v1_root=v1_root,
        source_root=source_root,
        source_summary=source_summary,
        dataset=dataset,
    )
    backend = load_json(backend_path)
    if (
        backend.get("format") != V1_BACKEND_FORMAT
        or backend.get("status") != "complete"
        or backend.get("backend") != V1_BACKEND_NAME
        or backend.get("dataset_id") != dataset.get("dataset_id")
        or backend.get("roles") != {task: list(roles) for task, roles in TASK_ROLES.items()}
        or backend.get("protocol", {}).get("episode_ground_truth_read") is not False
        or backend.get("protocol", {}).get("strict_source_pixel_access_gate") is not True
    ):
        raise RuntimeError("The v1 backend identity or GT-free protocol changed.")
    graphs = backend.get("graphs")
    results = backend.get("results")
    if not isinstance(graphs, dict) or set(graphs) != set(TASK_ROLES):
        raise RuntimeError("The v1 graph metadata is incomplete.")
    if not isinstance(results, dict) or set(results) != set(TASK_ROLES):
        raise RuntimeError("The v1 result map is incomplete.")
    entries: list[tuple[Path, str, str | None]] = [
        (v1_root / "object_graph_tokenizer_summary.json", "v1 summary", None),
        (backend_path, "v1 backend manifest", summary["provenance"]["object_graph_backend_manifest_sha256"]),
        (isolation_path, "v1 isolation gate", summary["provenance"]["scoring_isolation_sha256"]),
        (v1_root / "provenance" / "immutable_inputs.json", "v1 immutable inputs", None),
    ]
    episodes = int(dataset["counts"]["episodes"])
    frames = int(dataset["counts"]["frames_per_episode"])
    for task, roles in TASK_ROLES.items():
        graph_meta = graphs[task].get("graph") if isinstance(graphs[task], dict) else None
        entity_names = graph_meta.get("entity_names") if isinstance(graph_meta, dict) else None
        if not isinstance(entity_names, list) or not entity_names or len(set(entity_names)) != len(entity_names):
            raise RuntimeError(f"The v1 entity order is malformed for {task}.")
        condition_map = results[task]
        if not isinstance(condition_map, dict) or set(condition_map) != set(CONDITIONS):
            raise RuntimeError(f"The v1 condition set changed for {task}.")
        for condition in CONDITIONS:
            records = condition_map[condition]
            if not isinstance(records, list) or len(records) != episodes:
                raise RuntimeError(f"The v1 episode count changed for {task}/{condition}.")
            for episode, record in enumerate(records):
                if not isinstance(record, dict) or set(record) != {
                    "episode_index",
                    "frames",
                    "entity_count",
                    "role_count",
                    "prediction_arrays",
                    "prediction_arrays_sha256",
                    "array_shapes",
                    "traces",
                }:
                    raise RuntimeError("The v1 prediction record schema changed.")
                if (
                    record["episode_index"] != episode
                    or record["frames"] != frames
                    or record["entity_count"] != len(entity_names)
                    or record["role_count"] != len(roles)
                    or set(record.get("traces", {})) != V1_TRACE_KEYS
                ):
                    raise RuntimeError("The v1 prediction counts or traces changed.")
                for digest in record["traces"].values():
                    require_sha256(digest, "v1 trace SHA")
                prediction_path = _member(
                    backend_path.parent,
                    record["prediction_arrays"],
                    f"v1 prediction {task}/{condition}/{episode}",
                )
                entries.append(
                    (
                        prediction_path,
                        f"v1 prediction {task}/{condition}/{episode}",
                        record["prediction_arrays_sha256"],
                    )
                )
    return backend, _inventory(v1_root, entries, "v1 preflight inventory")


def build(*, source_root: Path, v1_root: Path) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    package_root = project_root / "tdmpc2"
    v1_graph_path = package_root / "object_graphs" / "acrobot_swingup.json"
    v2_graph_path = (
        package_root
        / "object_graphs_temporal_v2"
        / "acrobot_swingup_temporal_v2.json"
    )
    source_root = _root(source_root, "unified benchmark root")
    v1_root = _root(v1_root, "v1 object-graph preflight root")
    if source_root == v1_root or source_root in v1_root.parents or v1_root in source_root.parents:
        raise ValueError("Unified and v1 preflight roots must be strictly disjoint.")
    source_summary, dataset, source_inventory = _source_inventory(source_root)
    v1_backend, v1_inventory = _v1_inventory(
        v1_root=v1_root,
        source_root=source_root,
        source_summary=source_summary,
        dataset=dataset,
    )
    v1_graph = load_object_graph(v1_graph_path)
    v2_graph = load_object_graph(v2_graph_path)
    published_graph = v1_backend["graphs"].get("acrobot-swingup")
    if not isinstance(published_graph, dict) or (
        published_graph.get("graph_file_sha256") != file_sha256(v1_graph_path)
        or published_graph.get("graph") != v1_graph.metadata()
        or published_graph.get("tokenizer", {}).get("graph_sha256")
        != v1_graph.graph_sha256
    ):
        raise RuntimeError("The live v1 Acrobot graph differs from its published backend.")
    if (
        v1_graph.task != "acrobot-swingup"
        or v2_graph.task != v1_graph.task
        or v2_graph.source_roles != v1_graph.source_roles
        or v2_graph.semantic_roles != v1_graph.semantic_roles
        or v2_graph.entity_names != v1_graph.entity_names
    ):
        raise RuntimeError("The temporal graph changed the frozen Acrobot ontology.")
    projector_types = [
        entity["projector"]["type"]
        for entity in v2_graph.canonical_payload["tracking_entities"]
    ]
    if projector_types != ["ordered_chain_temporal_v2"]:
        raise RuntimeError("The temporal replay graph does not select temporal v2.")
    graph_inventory = _inventory(
        package_root,
        [
            (v1_graph_path, "live v1 Acrobot graph", published_graph["graph_file_sha256"]),
            (v2_graph_path, "live temporal-v2 Acrobot graph", None),
        ],
        "graph inventory",
    )
    return {
        "format": FORMAT,
        "source_benchmark_root": str(source_root),
        "v1_preflight_root": str(v1_root),
        "dataset_id": dataset["dataset_id"],
        "v1_backend_dataset_id": v1_backend["dataset_id"],
        "source_inventory": source_inventory,
        "v1_inventory": v1_inventory,
        "graph_contract": {
            "v1_graph_sha256": v1_graph.graph_sha256,
            "v2_graph_sha256": v2_graph.graph_sha256,
            "projector_types": projector_types,
            "inventory": graph_inventory,
        },
        "local_source": _local_tree(package_root),
        "environment": _environment(),
        "cpu_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-benchmark-root", type=Path, required=True)
    parser.add_argument("--v1-preflight-root", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()
    payload = build(
        source_root=args.source_benchmark_root,
        v1_root=args.v1_preflight_root,
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
        raise RuntimeError("Temporal replay immutable inputs changed.")
    print(json.dumps({"status": "verified", "path": str(path)}, allow_nan=False))


if __name__ == "__main__":
    main()
