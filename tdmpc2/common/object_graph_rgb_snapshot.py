"""Immutable inputs for the current-RGB ordered-chain candidate preflight.

The diagnostic consumes three already-published artifacts (the frozen unified
VOS benchmark, the v1 object-graph result, and the mask-only top-K result), a
local frozen DINOv2 checkout/checkpoint, and the current ``tdmpc2`` sources.
Every referenced tree is rebuilt and hashed during verification.  Ground truth
is inventoried here, but the runner removes access before launching the RGB
backend; only the privileged scorer runs after permissions are restored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tdmpc2.common.object_graph_temporal_replay_snapshot import (
    _canonical,
    _contained_file,
    _environment,
    _inventory,
    _local_tree,
    _member,
    _reject_symlink_components,
    _root,
    _source_inventory,
    _v1_inventory,
)
from tdmpc2.common.unified_vos import TASK_ROLES, file_sha256, load_json, require_sha256
from tdmpc2.perception.ordered_chain_rgb import (
    FORMAT as GENERATOR_FORMAT,
    MAX_CANDIDATES,
    PROTOCOL as GENERATOR_PROTOCOL,
    SOURCE_CODES,
)
from tdmpc2.perception.support_conditioned_object_graph import load_object_graph
from tdmpc2.tools.replay_object_graph_rgb_candidates import (
    ARMS,
    BACKEND,
    FORMAT as BACKEND_FORMAT,
    OUTPUT_ARRAY_KEYS,
    PROTOCOL as BACKEND_PROTOCOL,
)


FORMAT = "object_graph_rgb_candidate_inputs_v1"
SUMMARY_FORMAT = "object_graph_rgb_candidate_coverage_summary_v1"
ISOLATION_FORMAT = "object_graph_rgb_candidate_scoring_isolation_v1"
MASK_TOPK_SUMMARY_FORMAT = "object_graph_topk_candidate_coverage_summary_v1"


def _strictly_disjoint(roots: dict[str, Path]) -> None:
    items = list(roots.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(
                    f"RGB preflight roots must be strictly disjoint "
                    f"({left_name}/{right_name})."
                )


def _whole_tree(root: Path, label: str) -> dict[str, Any]:
    """Inventory every non-cache regular member of a published/external tree."""
    root = _root(root, label)
    entries: list[tuple[Path, str, str | None]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"{label} contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"{label} contains a special member: {path}")
        relative = path.relative_to(root)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        entries.append((path, f"{label} {relative.as_posix()}", None))
    result = _inventory(root, entries, f"{label} inventory")
    result["policy"] = "all_regular_files_excluding_git_and_python_caches_v1"
    return result


def _external_file(path: Path, label: str) -> dict[str, Any]:
    lexical = _reject_symlink_components(path, label)
    resolved = lexical.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file: {resolved}")
    before = resolved.stat()
    digest = file_sha256(resolved)
    after = resolved.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise RuntimeError(f"{label} changed while being inventoried: {resolved}")
    return {"path": str(resolved), "bytes": before.st_size, "sha256": digest}


def _mask_topk_inventory(
    *,
    root: Path,
    source_root: Path,
    v1_root: Path,
    dataset_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _root(root, "published mask-only top-K preflight root")
    summary_path = _contained_file(
        root / "object_graph_topk_candidate_coverage_summary.json",
        root,
        "published mask-only top-K summary",
    )
    summary = load_json(summary_path)
    if summary_path.read_bytes() != _canonical(summary):
        raise ValueError("Published mask-only top-K summary is not canonical JSON.")
    scope = summary.get("scope")
    provenance = summary.get("provenance")
    if (
        summary.get("format") != MASK_TOPK_SUMMARY_FORMAT
        or summary.get("engineering_pass") is not True
        or summary.get("controller_training_authorized") is not False
        or summary.get("scientific_go") is not False
        or not isinstance(scope, dict)
        or scope.get("controller_training_steps") != 0
        or not isinstance(provenance, dict)
        or provenance.get("source_benchmark_root") != str(source_root)
        or provenance.get("source_dataset_id") != dataset_id
        or provenance.get("source_summary_sha256")
        != file_sha256(source_root / "unified_vos_summary.json")
        or provenance.get("v1_preflight_root") != str(v1_root)
        or provenance.get("v1_preflight_summary_sha256")
        != file_sha256(v1_root / "object_graph_tokenizer_summary.json")
    ):
        raise RuntimeError("Published mask-only top-K result is not an eligible source.")

    bindings = (
        (
            "topk_backend_manifest_relative_to_summary_root",
            "topk_backend_manifest_sha256",
            "mask-only top-K backend manifest",
        ),
        (
            "scoring_isolation_relative_to_summary_root",
            "scoring_isolation_sha256",
            "mask-only top-K scoring isolation",
        ),
        (
            "immutable_inputs_relative_to_summary_root",
            "immutable_inputs_sha256",
            "mask-only top-K immutable inputs",
        ),
    )
    bound: dict[str, dict[str, Any]] = {}
    for path_field, sha_field, label in bindings:
        member = _member(root, provenance.get(path_field), label)
        expected = require_sha256(provenance.get(sha_field), f"{label} SHA")
        actual = file_sha256(member)
        if actual != expected:
            raise RuntimeError(f"{label} changed after publication.")
        bound[path_field] = {
            "path": member.relative_to(root).as_posix(),
            "sha256": actual,
        }
    tree = _whole_tree(root, "published mask-only top-K preflight root")
    return summary, {
        "summary": {
            "path": summary_path.relative_to(root).as_posix(),
            "sha256": file_sha256(summary_path),
        },
        "bound_artifacts": bound,
        "tree": tree,
    }


def _graph_contract(*, package_root: Path, v1_backend: dict[str, Any]) -> dict[str, Any]:
    graph_path = package_root / "object_graphs" / "acrobot_swingup.json"
    graph = load_object_graph(graph_path)
    published = v1_backend.get("graphs", {}).get("acrobot-swingup")
    if not isinstance(published, dict):
        raise RuntimeError("The published v1 Acrobot graph metadata is missing.")
    if (
        published.get("graph_file_sha256") != file_sha256(graph_path)
        or published.get("graph") != graph.metadata()
        or published.get("tokenizer", {}).get("graph_sha256")
        != graph.graph_sha256
    ):
        raise RuntimeError("The live Acrobot graph differs from published v1.")
    if (
        graph.task != "acrobot-swingup"
        or list(graph.source_roles) != list(TASK_ROLES["acrobot-swingup"])
        or list(graph.semantic_roles) != list(TASK_ROLES["acrobot-swingup"])
        or len(graph.entity_names) != 1
    ):
        raise RuntimeError("The Acrobot graph ontology changed.")
    projector_types = [
        entry["projector"]["type"]
        for entry in graph.canonical_payload["tracking_entities"]
    ]
    if projector_types != ["ordered_chain_segments_v1"]:
        raise RuntimeError("RGB preflight must preserve the published v1 projector.")
    inventory = _inventory(
        package_root,
        [(graph_path, "live published-v1 Acrobot graph", published["graph_file_sha256"])],
        "RGB graph inventory",
    )
    return {
        "task": graph.task,
        "source_roles": list(graph.source_roles),
        "semantic_roles": list(graph.semantic_roles),
        "entity_names": list(graph.entity_names),
        "projector_types": projector_types,
        "graph_sha256": graph.graph_sha256,
        "inventory": inventory,
    }


def build(
    *,
    source_root: Path,
    v1_root: Path,
    mask_topk_root: Path,
    dino_repo: Path,
    dino_checkpoint: Path,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    package_root = project_root / "tdmpc2"
    source_root = _root(source_root, "unified benchmark root")
    v1_root = _root(v1_root, "v1 object-graph preflight root")
    mask_topk_root = _root(mask_topk_root, "mask-only top-K preflight root")
    dino_repo = _root(dino_repo, "DINOv2 repository")
    _strictly_disjoint(
        {
            "source": source_root,
            "v1": v1_root,
            "mask_topk": mask_topk_root,
            "dino_repo": dino_repo,
        }
    )

    source_summary, dataset, source_inventory = _source_inventory(source_root)
    v1_backend, v1_inventory = _v1_inventory(
        v1_root=v1_root,
        source_root=source_root,
        source_summary=source_summary,
        dataset=dataset,
    )
    _, mask_topk_inventory = _mask_topk_inventory(
        root=mask_topk_root,
        source_root=source_root,
        v1_root=v1_root,
        dataset_id=dataset["dataset_id"],
    )
    graph_contract = _graph_contract(package_root=package_root, v1_backend=v1_backend)
    dino_checkpoint_record = _external_file(dino_checkpoint, "DINOv2 checkpoint")
    dino_tree = _whole_tree(dino_repo, "DINOv2 repository")
    return {
        "format": FORMAT,
        "source_benchmark_root": str(source_root),
        "v1_preflight_root": str(v1_root),
        "mask_topk_preflight_root": str(mask_topk_root),
        "dataset_id": dataset["dataset_id"],
        "v1_backend_dataset_id": v1_backend["dataset_id"],
        "source_inventory": source_inventory,
        "v1_inventory": v1_inventory,
        "mask_topk_inventory": mask_topk_inventory,
        "graph_contract": graph_contract,
        "dino_contract": {
            "repository": dino_tree,
            "checkpoint": dino_checkpoint_record,
        },
        "implementation_contract": {
            "generator_format": GENERATOR_FORMAT,
            "generator_protocol": GENERATOR_PROTOCOL,
            "generator_source_codes": dict(SOURCE_CODES),
            "max_candidates": MAX_CANDIDATES,
            "backend": BACKEND,
            "backend_format": BACKEND_FORMAT,
            "backend_protocol": BACKEND_PROTOCOL,
            "backend_arms": list(ARMS),
            "backend_output_array_keys": sorted(OUTPUT_ARRAY_KEYS),
            "summary_format": SUMMARY_FORMAT,
            "scoring_isolation_format": ISOLATION_FORMAT,
        },
        "local_source": _local_tree(package_root),
        "environment": _environment(),
        "scope": {
            "episode_ground_truth_available_to_backend": False,
            "current_rgb_backend_input": True,
            "fixed_support_rgb_backend_input": True,
            "fixed_labelled_support_masks_backend_input": True,
            "backend_temporal_state": False,
            "controller_training_steps": 0,
            "controller_training_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-benchmark-root", type=Path, required=True)
    parser.add_argument("--v1-preflight-root", type=Path, required=True)
    parser.add_argument("--mask-topk-preflight-root", type=Path, required=True)
    parser.add_argument("--dino-repo", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()
    path = (args.output if args.output is not None else args.verify).expanduser().absolute()
    if path.name != "immutable_inputs.json" or path.parent.name != "provenance":
        raise ValueError(
            "RGB immutable inputs must use <summary-root>/provenance/immutable_inputs.json."
        )
    summary_root = _root(path.parent.parent, "RGB preflight summary root")
    _strictly_disjoint(
        {
            "source": _root(args.source_benchmark_root, "unified benchmark root"),
            "v1": _root(args.v1_preflight_root, "v1 object-graph preflight root"),
            "mask_topk": _root(
                args.mask_topk_preflight_root, "mask-only top-K preflight root"
            ),
            "dino_repo": _root(args.dino_repo, "DINOv2 repository"),
            "output": summary_root,
        }
    )
    payload = build(
        source_root=args.source_benchmark_root,
        v1_root=args.v1_preflight_root,
        mask_topk_root=args.mask_topk_preflight_root,
        dino_repo=args.dino_repo,
        dino_checkpoint=args.dino_checkpoint,
    )
    path = path.resolve()
    if args.output is not None:
        if path.exists() or not path.parent.is_dir():
            raise FileExistsError(path)
        path.write_bytes(_canonical(payload))
        print(json.dumps({"status": "bound", "path": str(path)}, allow_nan=False))
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing != payload or path.read_bytes() != _canonical(existing):
        raise RuntimeError("RGB candidate immutable inputs changed.")
    print(json.dumps({"status": "verified", "path": str(path)}, allow_nan=False))


if __name__ == "__main__":
    main()
