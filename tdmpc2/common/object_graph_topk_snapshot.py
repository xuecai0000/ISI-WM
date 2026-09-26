"""Immutable inputs for the CPU-only ordered-chain top-K preflight.

The preflight consumes two already-published artifacts: the frozen unified-VOS
benchmark and the v1 object-graph tokenizer result.  This module reuses their
strict inventory validators, binds the live Acrobot ontology to the published
v1 backend, and inventories the complete local ``tdmpc2`` implementation and
Python environment.  Verification rebuilds the record byte-for-byte; no
timestamp-only trust is used.

Ground-truth scoring files are inventoried here, but the runner removes access
to them before launching the candidate backend.  The backend itself remains a
stateless, current-entity-mask-only CPU replay.  Controller training is outside
the scope of this snapshot and of the runner that consumes it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tdmpc2.common.object_graph_temporal_replay_snapshot import (
    _canonical,
    _environment,
    _inventory,
    _local_tree,
    _root,
    _source_inventory,
    _v1_inventory,
)
from tdmpc2.common.unified_vos import TASK_ROLES, file_sha256
from tdmpc2.perception.ordered_chain_topk import (
    FORMAT as GENERATOR_FORMAT,
    MAX_CANDIDATES,
    PROTOCOL as GENERATOR_PROTOCOL,
    SOURCE_CODES,
)
from tdmpc2.perception.support_conditioned_object_graph import load_object_graph
from tdmpc2.tools.replay_object_graph_topk_candidates import (
    BACKEND,
    EXPECTED_MAX_CANDIDATES,
    FORMAT as BACKEND_FORMAT,
    PROTOCOL as BACKEND_PROTOCOL,
)


FORMAT = "object_graph_topk_candidate_inputs_v1"
SUMMARY_FORMAT = "object_graph_topk_candidate_coverage_summary_v1"
ISOLATION_FORMAT = "object_graph_topk_candidate_scoring_isolation_v1"


def _graph_contract(
    *, package_root: Path, v1_backend: dict[str, Any]
) -> dict[str, Any]:
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
        raise RuntimeError("The live v1 Acrobot graph differs from its published backend.")
    if (
        graph.task != "acrobot-swingup"
        or list(graph.source_roles) != list(TASK_ROLES["acrobot-swingup"])
        or list(graph.semantic_roles) != list(TASK_ROLES["acrobot-swingup"])
        or len(graph.entity_names) != 1
    ):
        raise RuntimeError("The Acrobot graph ontology changed.")
    projectors = [
        entity["projector"]["type"]
        for entity in graph.canonical_payload["tracking_entities"]
    ]
    if projectors != ["ordered_chain_segments_v1"]:
        raise RuntimeError("The top-K preflight must preserve the published v1 projector.")
    if MAX_CANDIDATES != EXPECTED_MAX_CANDIDATES or MAX_CANDIDATES != 4:
        raise RuntimeError("The fixed top-K prefix contract changed.")
    inventory = _inventory(
        package_root,
        [
            (
                graph_path,
                "live published-v1 Acrobot graph",
                published["graph_file_sha256"],
            )
        ],
        "top-K graph inventory",
    )
    return {
        "task": graph.task,
        "source_roles": list(graph.source_roles),
        "semantic_roles": list(graph.semantic_roles),
        "entity_names": list(graph.entity_names),
        "projector_types": projectors,
        "graph_sha256": graph.graph_sha256,
        "inventory": inventory,
    }


def build(
    *,
    source_benchmark_root: Path | None = None,
    v1_preflight_root: Path | None = None,
    source_root: Path | None = None,
    v1_root: Path | None = None,
) -> dict[str, Any]:
    """Build the canonical snapshot.

    ``source_root``/``v1_root`` are narrow compatibility aliases for the
    privileged scorer, whose internal validator follows the older temporal
    snapshot call convention.  Supplying both spellings fails closed.
    """
    if source_benchmark_root is not None and source_root is not None:
        raise TypeError("Specify only one source benchmark root spelling.")
    if v1_preflight_root is not None and v1_root is not None:
        raise TypeError("Specify only one v1 preflight root spelling.")
    source_argument = (
        source_benchmark_root if source_benchmark_root is not None else source_root
    )
    v1_argument = v1_preflight_root if v1_preflight_root is not None else v1_root
    if source_argument is None or v1_argument is None:
        raise TypeError("Both source benchmark and v1 preflight roots are required.")
    project_root = Path(__file__).resolve().parents[2]
    package_root = project_root / "tdmpc2"
    source_root = _root(source_argument, "unified benchmark root")
    v1_root = _root(v1_argument, "v1 object-graph preflight root")
    if source_root == v1_root or source_root in v1_root.parents or v1_root in source_root.parents:
        raise ValueError("Unified and v1 preflight roots must be strictly disjoint.")

    source_summary, dataset, source_inventory = _source_inventory(source_root)
    v1_backend, v1_inventory = _v1_inventory(
        v1_root=v1_root,
        source_root=source_root,
        source_summary=source_summary,
        dataset=dataset,
    )
    graph_contract = _graph_contract(package_root=package_root, v1_backend=v1_backend)
    return {
        "format": FORMAT,
        "source_benchmark_root": str(source_root),
        "v1_preflight_root": str(v1_root),
        "dataset_id": dataset["dataset_id"],
        "v1_backend_dataset_id": v1_backend["dataset_id"],
        "source_inventory": source_inventory,
        "v1_inventory": v1_inventory,
        "graph_contract": graph_contract,
        "implementation_contract": {
            "generator_format": GENERATOR_FORMAT,
            "generator_protocol": GENERATOR_PROTOCOL,
            "generator_source_codes": dict(SOURCE_CODES),
            "max_candidates": MAX_CANDIDATES,
            "backend": BACKEND,
            "backend_format": BACKEND_FORMAT,
            "backend_protocol": BACKEND_PROTOCOL,
            "summary_format": SUMMARY_FORMAT,
            "scoring_isolation_format": ISOLATION_FORMAT,
        },
        "local_source": _local_tree(package_root),
        "environment": _environment(),
        "scope": {
            "cpu_only": True,
            "episode_ground_truth_available_to_backend": False,
            "backend_temporal_state": False,
            "controller_training_steps": 0,
            "controller_training_authorized": False,
        },
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
        source_benchmark_root=args.source_benchmark_root,
        v1_preflight_root=args.v1_preflight_root,
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
        raise RuntimeError("Top-K preflight immutable inputs changed.")
    print(json.dumps({"status": "verified", "path": str(path)}, allow_nan=False))


if __name__ == "__main__":
    main()
