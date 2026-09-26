"""Real-GPU contract for history-independent live Cutie episode reset.

The probe uses only the frozen support RGB frames. Two independent spawn
workers observe the same probe frame, follow different 500-frame histories,
then reset and observe the identical frame again. The exported generic object
features must be bitwise identical within and across workers.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
for local_path in (str(REPO_DIR), str(PROJECT_DIR)):
    while local_path in sys.path:
        sys.path.remove(local_path)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))


FORMAT = "cutie_episode_reset_isolation_v1"
RESET_STRATEGY = "fresh_inference_core_support_replay_v1"
ROLE_NAMES = ("whole_arm", "goal")
TASK_ROLE_NAMES = {
    "reacher-visual-small": ROLE_NAMES,
    "cup-catch": ("cup", "ball"),
    "cartpole-swingup": ("cart", "pole"),
    "finger-spin": ("finger", "spinner"),
    "acrobot-swingup": ("upper_arm", "lower_arm"),
}
PARTITIONS = {
    "query": (0, 512),
    "occupancy": (512, 576),
    "centroid_area": (576, 579),
    "bbox": (579, 583),
    "moments": (583, 586),
    "status": (586, 590),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _canonical_feature(value) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (2, 590):
        raise ValueError(f"Expected Cutie feature [2,590], got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("Cutie reset probe feature contains a non-finite value.")
    return np.ascontiguousarray(array, dtype=np.dtype("<f4"))


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(_canonical_feature(value).tobytes(order="C")).hexdigest()


def _canonical_sequence(value) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[1:] != (2, 590):
        raise ValueError(f"Expected Cutie feature sequence [T,2,590], got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("Cutie reset probe sequence contains a non-finite value.")
    return np.ascontiguousarray(array, dtype=np.dtype("<f4"))


def _sequence_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(_canonical_sequence(value).tobytes(order="C")).hexdigest()


def _comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left = _canonical_feature(left)
    right = _canonical_feature(right)
    delta = left.astype(np.float64) - right.astype(np.float64)
    absolute = np.abs(delta)

    def metrics(start: int, end: int) -> dict[str, Any]:
        local = absolute[:, start:end]
        return {
            "array_equal": bool(np.array_equal(left[:, start:end], right[:, start:end])),
            "nonzero_diff_count": int(np.count_nonzero(local)),
            "max_abs": float(local.max(initial=0.0)),
            "mean_abs": float(local.mean()) if local.size else 0.0,
            "rms": float(math.sqrt(float(np.mean(local * local)))) if local.size else 0.0,
        }

    return {
        "left_sha256": _array_sha256(left),
        "right_sha256": _array_sha256(right),
        "byte_equal": bool(left.tobytes(order="C") == right.tobytes(order="C")),
        "array_equal": bool(np.array_equal(left, right)),
        "nonzero_diff_count": int(np.count_nonzero(absolute)),
        "max_abs": float(absolute.max(initial=0.0)),
        "mean_abs": float(absolute.mean()),
        "rms": float(math.sqrt(float(np.mean(absolute * absolute)))),
        "partitions": {
            name: metrics(start, end)
            for name, (start, end) in PARTITIONS.items()
        },
    }


def _sequence_comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left = _canonical_sequence(left)
    right = _canonical_sequence(right)
    if left.shape != right.shape:
        raise ValueError(f"Sequence shape mismatch: {left.shape} vs {right.shape}.")
    frames = [_comparison(a, b) for a, b in zip(left, right)]
    first_mismatch = next(
        (index for index, item in enumerate(frames) if not item["byte_equal"]),
        None,
    )
    return {
        "left_sha256": _sequence_sha256(left),
        "right_sha256": _sequence_sha256(right),
        "byte_equal": bool(left.tobytes(order="C") == right.tobytes(order="C")),
        "array_equal": bool(np.array_equal(left, right)),
        "frames": len(frames),
        "first_mismatch_frame": first_mismatch,
        "per_frame": frames,
    }


def _runtime_config_contract(payload: dict[str, Any]) -> None:
    task = payload.get("task")
    if task not in TASK_ROLE_NAMES:
        raise ValueError(
            f"Reset isolation task must be one of {tuple(TASK_ROLE_NAMES)!r}, "
            f"got {task!r}."
        )
    expected_roles = TASK_ROLE_NAMES[task]
    roles = tuple(payload.get("cutie_object_role_names", expected_roles))
    if roles != expected_roles:
        raise ValueError(
            f"{task} requires cutie_object_role_names={expected_roles!r}, got {roles!r}."
        )
    schema = payload.get(
        "cutie_object_support_schema",
        "whole_arm_goal_v1" if task == "reacher-visual-small" else None,
    )
    allowed_schemas = (
        {"whole_arm_goal_v1", "generic_indexed_v1"}
        if task == "reacher-visual-small"
        else {"generic_indexed_v1"}
    )
    if schema not in allowed_schemas:
        raise ValueError(
            f"{task} reset isolation does not accept support schema {schema!r}."
        )
    if schema == "generic_indexed_v1" and payload.get(
        "cutie_object_allow_simulator_support"
    ) is not True:
        raise ValueError(
            "generic_indexed_v1 requires cutie_object_allow_simulator_support=true."
        )
    expected = {
        "obs": "rgb",
        "model_size": 5,
        "flat_anchor": True,
        "cutie_object_device": "cuda:0",
        "cutie_object_tracker_height": 448,
        "cutie_object_tracker_width": 448,
        "cutie_object_model_size": "small",
        "cutie_object_prompt_radius": 2.0,
        "cutie_object_amp": True,
        "cutie_object_worker_timeout_seconds": 180.0,
    }
    bad = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if bad:
        raise ValueError(f"Runtime config is not the frozen Cutie object contract: {bad!r}.")
    if payload.get("flat_anchor_mode") not in {
        "cutie_hybrid", "cutie_object_only"
    }:
        raise ValueError(
            "Reset isolation applies only to cutie_hybrid or cutie_object_only."
        )


def _external_source_paths(repo_path: Path) -> dict[str, Path]:
    cutie_root = repo_path / "feature_extractor" / "cutie" / "cutie"
    base = cutie_root / "inference"
    paths = {
        "inference_core": base / "inference_core.py",
        "memory_manager": base / "memory_manager.py",
        "object_manager": base / "object_manager.py",
        "image_feature_store": base / "image_feature_store.py",
        "kv_memory_store": base / "kv_memory_store.py",
        "object_transformer": (
            cutie_root / "model" / "transformer" / "object_transformer.py"
        ),
        "cutie_model": cutie_root / "model" / "cutie.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot bind the reset probe to the vendored Cutie sources: " + ", ".join(missing)
        )
    return paths


def _config_tree_sha256(config_dir: Path) -> tuple[str, list[str]]:
    files = sorted(
        path for path in config_dir.rglob("*.yaml") if path.is_file()
    ) + sorted(path for path in config_dir.rglob("*.yml") if path.is_file())
    if not files:
        raise FileNotFoundError(f"Cutie config directory contains no YAML: {config_dir}")
    digest = hashlib.sha256()
    relative_paths = []
    for path in files:
        relative = path.relative_to(config_dir).as_posix()
        relative_paths.append(relative)
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest(), relative_paths


def _run_worker(config, probe_frame, continuation_frames, pollution_frames):
    from tdmpc2.envs.wrappers.cutie_object import _SpawnCutieClient

    client = _SpawnCutieClient(config)
    try:
        ready = dict(client.ready)
        if ready.get("episode_reset_strategy") != RESET_STRATEGY:
            raise RuntimeError(f"Unexpected episode reset strategy: {ready!r}")
        fresh = _canonical_feature(client.reset_track(probe_frame))
        fresh_continuation = [
            _canonical_feature(client.track(frame)) for frame in continuation_frames
        ]
        history_digest = hashlib.sha256()
        history_first = None
        history_last = None
        for frame in pollution_frames:
            tracked = _canonical_feature(client.track(frame))
            history_digest.update(tracked.tobytes(order="C"))
            if history_first is None:
                history_first = _array_sha256(tracked)
            history_last = _array_sha256(tracked)
        post_history = _canonical_feature(client.reset_track(probe_frame))
        post_continuation = [
            _canonical_feature(client.track(frame)) for frame in continuation_frames
        ]
        return {
            "ready": ready,
            "fresh": fresh,
            "post_history": post_history,
            "fresh_sequence": np.stack([fresh, *fresh_continuation], axis=0),
            "post_sequence": np.stack([post_history, *post_continuation], axis=0),
            "tracked_frames": 2 + 2 * len(continuation_frames) + len(pollution_frames),
            "pollution_feature_sha256": history_digest.hexdigest(),
            "pollution_first_feature_sha256": history_first,
            "pollution_last_feature_sha256": history_last,
        }
    finally:
        client.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pollution-length", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    runtime_config = args.runtime_config.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".json":
        raise ValueError("--output must end in .json so its .npz asset cannot alias it.")
    if not runtime_config.is_file():
        raise FileNotFoundError(runtime_config)
    if output.exists() or output.with_suffix(".npz").exists():
        raise FileExistsError(f"Refusing to overwrite reset probe output: {output}")
    if args.pollution_length < 1:
        raise ValueError("--pollution-length must be positive.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise ValueError(
            "Expose exactly one physical GPU with CUDA_VISIBLE_DEVICES; the worker uses cuda:0."
        )

    payload = _json_load(runtime_config)
    _runtime_config_contract(payload)
    repo_path = Path(payload["cutie_object_repo"]).expanduser().resolve()
    checkpoint_path = Path(payload["cutie_object_checkpoint"]).expanduser().resolve()
    support_path = Path(payload["cutie_object_support_path"]).expanduser().resolve()
    raw_config_dir = payload.get("cutie_object_config_dir")
    config_dir = (
        None
        if raw_config_dir is None
        else Path(raw_config_dir).expanduser().resolve()
    )
    effective_config_dir = config_dir or (
        repo_path / "feature_extractor" / "cutie" / "cutie" / "config"
    )
    for path in (repo_path, checkpoint_path, support_path, effective_config_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    from tdmpc2.envs.wrappers.cutie_object import CutieObjectWorkerConfig
    from tdmpc2.perception.cutie_oc_adapter import (
        load_indexed_support_prompts,
        load_point_support_prompts,
    )

    task = str(payload["task"])
    roles = TASK_ROLE_NAMES[task]
    support_schema = str(payload.get("cutie_object_support_schema", "whole_arm_goal_v1"))
    allow_simulator_support = bool(
        payload.get("cutie_object_allow_simulator_support", False)
    )
    if support_schema == "whole_arm_goal_v1":
        support = load_point_support_prompts(
            support_path,
            radius_px=2.0,
            expected_records=6,
            object_schema="whole_arm_goal_v1",
        )
    else:
        support = load_indexed_support_prompts(
            support_path,
            role_names=roles,
            expected_task=task,
            expected_records=6,
            allow_simulator_support=allow_simulator_support,
        )
    if (
        len(support.frames) != 6
        or tuple(support.role_names) != roles
        or support.metadata.get("task") != task
    ):
        raise ValueError(
            "The reset probe support pack does not match its frozen task/roles."
        )
    frames = tuple(
        np.array(frame, dtype=np.uint8, order="C", copy=True)
        for frame in support.frames
    )
    if any(frame.shape != (64, 64, 3) for frame in frames):
        raise ValueError("Support RGB frames must all be native uint8 [64,64,3].")
    probe_frame = frames[-1]
    forward = tuple(frames[index % len(frames)] for index in range(args.pollution_length))
    reverse_frames = tuple(reversed(frames))
    reverse = tuple(
        reverse_frames[index % len(reverse_frames)]
        for index in range(args.pollution_length)
    )
    continuation = frames
    config = CutieObjectWorkerConfig(
        repo_path=str(repo_path),
        checkpoint_path=str(checkpoint_path),
        support_path=str(support_path),
        config_dir=None if config_dir is None else str(config_dir),
        device="cuda:0",
        tracker_height=448,
        tracker_width=448,
        model_size="small",
        prompt_radius=2.0,
        amp=True,
        worker_timeout_seconds=float(payload["cutie_object_worker_timeout_seconds"]),
        role_names=roles,
        support_schema=support_schema,
        allow_simulator_support=allow_simulator_support,
        task=task,
    ).validated()

    run_a = _run_worker(config, probe_frame, continuation, forward)
    gc.collect()
    run_b = _run_worker(config, probe_frame, continuation, reverse)
    gc.collect()

    arrays = {
        "fresh_a": run_a["fresh"],
        "post_forward": run_a["post_history"],
        "fresh_b": run_b["fresh"],
        "post_reverse": run_b["post_history"],
        "fresh_sequence_a": run_a["fresh_sequence"],
        "post_sequence_forward": run_a["post_sequence"],
        "fresh_sequence_b": run_b["fresh_sequence"],
        "post_sequence_reverse": run_b["post_sequence"],
    }
    comparisons = {
        "fresh_a_vs_fresh_b": _comparison(arrays["fresh_a"], arrays["fresh_b"]),
        "fresh_a_vs_post_forward": _comparison(
            arrays["fresh_a"], arrays["post_forward"]
        ),
        "fresh_b_vs_post_reverse": _comparison(
            arrays["fresh_b"], arrays["post_reverse"]
        ),
        "post_forward_vs_post_reverse": _comparison(
            arrays["post_forward"], arrays["post_reverse"]
        ),
        "fresh_sequence_a_vs_fresh_sequence_b": _sequence_comparison(
            arrays["fresh_sequence_a"], arrays["fresh_sequence_b"]
        ),
        "fresh_sequence_a_vs_post_sequence_forward": _sequence_comparison(
            arrays["fresh_sequence_a"], arrays["post_sequence_forward"]
        ),
        "fresh_sequence_b_vs_post_sequence_reverse": _sequence_comparison(
            arrays["fresh_sequence_b"], arrays["post_sequence_reverse"]
        ),
        "post_sequence_forward_vs_post_sequence_reverse": _sequence_comparison(
            arrays["post_sequence_forward"], arrays["post_sequence_reverse"]
        ),
    }
    valid_index = 588
    endpoint_names = ("fresh_a", "post_forward", "fresh_b", "post_reverse")
    all_valid = all(
        bool((arrays[name][:, valid_index] > 0.5).all()) for name in endpoint_names
    )
    histories_diverged = bool(
        run_a["pollution_feature_sha256"] != run_b["pollution_feature_sha256"]
    )
    passed = bool(
        all_valid
        and histories_diverged
        and all(item["byte_equal"] for item in comparisons.values())
    )

    source_paths = _external_source_paths(repo_path)
    config_tree_sha256, config_files = _config_tree_sha256(effective_config_dir)
    import torch
    array_path = output.with_suffix(".npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_array = array_path.with_name(
        array_path.stem + f".tmp.{os.getpid()}" + array_path.suffix
    )
    np.savez(temporary_array, **arrays)
    os.replace(temporary_array, array_path)
    report = {
        "format": FORMAT,
        "status": "episode_reset_isolation_pass" if passed else "episode_reset_isolation_fail",
        "pass": passed,
        "reset_strategy": RESET_STRATEGY,
        "task": task,
        "role_names": list(roles),
        "support_schema": support_schema,
        "runtime_config": str(runtime_config),
        "runtime_config_sha256": _sha256(runtime_config),
        "pollution_length": args.pollution_length,
        "probe_frame_sha256": hashlib.sha256(probe_frame.tobytes(order="C")).hexdigest(),
        "support_frame_sha256": [
            hashlib.sha256(frame.tobytes(order="C")).hexdigest() for frame in frames
        ],
        "pollution_sequences": {
            "forward_indices": [index % 6 for index in range(args.pollution_length)],
            "reverse_indices": [5 - (index % 6) for index in range(args.pollution_length)],
        },
        "all_features_finite_and_roles_valid": all_valid,
        "pollution_histories_diverged": histories_diverged,
        "features": {
            name: {
                "shape": list(array.shape),
                "sha256": (
                    _array_sha256(array)
                    if array.ndim == 2
                    else _sequence_sha256(array)
                ),
            }
            for name, array in arrays.items()
        },
        "comparisons": comparisons,
        "workers": {
            "forward": {
                "tracked_frames": run_a["tracked_frames"],
                "pollution_feature_sha256": run_a["pollution_feature_sha256"],
                "pollution_first_feature_sha256": run_a[
                    "pollution_first_feature_sha256"
                ],
                "pollution_last_feature_sha256": run_a[
                    "pollution_last_feature_sha256"
                ],
                "ready": run_a["ready"],
            },
            "reverse": {
                "tracked_frames": run_b["tracked_frames"],
                "pollution_feature_sha256": run_b["pollution_feature_sha256"],
                "pollution_first_feature_sha256": run_b[
                    "pollution_first_feature_sha256"
                ],
                "pollution_last_feature_sha256": run_b[
                    "pollution_last_feature_sha256"
                ],
                "ready": run_b["ready"],
            },
        },
        "provenance": {
            "probe_file_sha256": _sha256(Path(__file__).resolve()),
            "adapter_file_sha256": _sha256(
                PROJECT_DIR / "perception" / "cutie_oc_adapter.py"
            ),
            "wrapper_file_sha256": _sha256(
                PROJECT_DIR / "envs" / "wrappers" / "cutie_object.py"
            ),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "support_path": str(support_path),
            "support_sha256": _sha256(support_path),
            "cutie_sources": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in source_paths.items()
            },
            "effective_config_dir": str(effective_config_dir),
            "effective_config_tree_sha256": config_tree_sha256,
            "effective_config_files": config_files,
            "cuda_visible_devices": visible,
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "arrays_asset": {
            "path": array_path.name,
            "sha256": _sha256(array_path),
        },
    }
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(report, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    os.replace(temporary, output)
    print(
        "CUTIE_EPISODE_RESET_ISOLATION_" + ("OK" if passed else "FAIL"),
        json.dumps(
            {
                "output": str(output),
                "pass": passed,
                "pollution_histories_diverged": histories_diverged,
                "fresh_sha256": report["features"]["fresh_a"]["sha256"],
                "comparisons": {
                    name: item["byte_equal"] for name, item in comparisons.items()
                },
            },
            sort_keys=True,
        ),
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
