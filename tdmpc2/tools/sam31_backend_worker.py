"""Run official SAM 3.1 on the GT-free unified VOS benchmark inputs.

This module is deliberately an isolated process boundary. It accepts only
``unified_vos_backend_inputs_v1`` (fixed support RGB+masks and episode RGB),
never opens the dataset/scoring manifest, and writes role-ordered masks in the
common backend prediction schema.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
from time import perf_counter
import traceback
from typing import Any, Mapping

import numpy as np

from tdmpc2.common.unified_vos import (
    BACKEND_FORMAT,
    file_sha256,
    validate_backend_inputs,
    write_json,
)
from tdmpc2.perception.sam31_backend import (
    CAUSAL_PROTOCOL,
    EXPECTED_SUPPORT_FRAMES,
    PROMPT_PROTOCOL,
    PROMPT_SAMPLER,
    OfficialSam31Config,
    OfficialSam31Runtime,
    Sam31BackendError,
    Sam31ContractError,
    build_support_point_prompts,
    inspect_official_installation,
    load_episode_rgb,
    load_frozen_support,
    run_official_sam31_episode,
    typed_array_sha256,
)


PREDICTION_KEYS = {
    "predicted_masks",
    "reported_confidence",
    "reported_lost",
    "runtime_ms",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        help="GT-free unified_vos_backend_inputs_v1 manifest.",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--sam3-repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--bpe", required=True, type=Path)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Logical CUDA device; unified runs require cuda:0 in a one-GPU process.",
    )
    parser.add_argument("--positive-points", type=int, default=4)
    parser.add_argument("--negative-points", type=int, default=4)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--use-fa3", action="store_true")
    parser.add_argument("--use-rope-real", action="store_true")
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--protocol-smoke-only", action="store_true")
    parser.add_argument(
        "--allow-nonstandard-counts",
        action="store_true",
        help="Contract-test only; scientific runs must not set this flag.",
    )
    parser.add_argument("--traceback", action="store_true")
    return parser.parse_args(argv)


def _require_isolated_cuda(device: str, seed: int) -> dict[str, Any]:
    if device != "cuda:0":
        raise Sam31ContractError(
            "unified SAM 3.1 inference requires --device cuda:0; select the "
            "physical GPU with CUDA_VISIBLE_DEVICES"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        raise Sam31ContractError(
            "CUDA_VISIBLE_DEVICES must explicitly select exactly one physical GPU"
        )
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise Sam31ContractError("CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    if not str(os.environ.get("BENCHMARK_GPU_UUID", "")).startswith("GPU-"):
        raise Sam31ContractError("BENCHMARK_GPU_UUID must bind the physical GPU")
    try:
        import torch
    except Exception as exc:
        raise Sam31ContractError(f"PyTorch import failed: {exc}") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Sam31ContractError(
            "SAM 3.1 worker requires exactly one logical CUDA device; set "
            "CUDA_VISIBLE_DEVICES to one physical GPU"
        )
    if type(seed) is not int or seed < 0:
        raise Sam31ContractError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    return {
        "cuda_visible_devices": visible,
        "logical_cuda_device": 0,
        "device_name": torch.cuda.get_device_name(0),
        "seed": seed,
    }


def _validate_prediction_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    frames: int,
    roles: int,
    resolution: int,
) -> dict[str, np.ndarray]:
    if set(arrays) != PREDICTION_KEYS:
        raise Sam31ContractError(
            f"unified prediction keys changed: {sorted(arrays)!r}"
        )
    expected_shapes = {
        "predicted_masks": (frames, roles, resolution, resolution),
        "reported_confidence": (frames, roles),
        "reported_lost": (frames, roles),
        "runtime_ms": (frames,),
    }
    expected_dtypes = {
        "predicted_masks": np.dtype(np.bool_),
        "reported_confidence": np.dtype(np.float32),
        "reported_lost": np.dtype(np.bool_),
        "runtime_ms": np.dtype(np.float64),
    }
    normalized: dict[str, np.ndarray] = {}
    for name in sorted(PREDICTION_KEYS):
        value = np.ascontiguousarray(arrays[name])
        if value.shape != expected_shapes[name] or value.dtype != expected_dtypes[name]:
            raise Sam31ContractError(
                f"{name} must be {expected_dtypes[name]} {expected_shapes[name]}, "
                f"got {value.dtype} {value.shape}"
            )
        normalized[name] = value
    confidence = normalized["reported_confidence"]
    runtime_ms = normalized["runtime_ms"]
    if not np.isfinite(confidence).all() or np.any(
        (confidence < 0.0) | (confidence > 1.0)
    ):
        raise Sam31ContractError("reported_confidence must be finite and in [0,1]")
    if not np.isfinite(runtime_ms).all() or np.any(runtime_ms <= 0.0):
        raise Sam31ContractError("runtime_ms must be finite and positive")
    masks = normalized["predicted_masks"]
    if np.any(masks.sum(axis=1) > 1):
        raise Sam31ContractError("SAM 3.1 emitted overlapping role masks")
    inferred_lost = ~masks.reshape(frames, roles, -1).any(axis=2)
    if not np.array_equal(normalized["reported_lost"], inferred_lost):
        raise Sam31ContractError("reported_lost disagrees with empty predicted masks")
    return normalized


def _save_prediction(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite prediction: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("xb") as handle:
        np.savez_compressed(
            handle,
            predicted_masks=arrays["predicted_masks"],
            reported_confidence=arrays["reported_confidence"],
            reported_lost=arrays["reported_lost"],
            runtime_ms=arrays["runtime_ms"],
        )
    os.replace(temporary, path)


def _validate_saved_prediction(
    path: Path,
    *,
    frames: int,
    roles: int,
    resolution: int,
) -> None:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != PREDICTION_KEYS:
            raise Sam31ContractError(
                f"saved prediction NPZ keys changed: {archive.files!r}"
            )
        _validate_prediction_arrays(
            {name: np.array(archive[name], copy=True) for name in archive.files},
            frames=frames,
            roles=roles,
            resolution=resolution,
        )


def _prompt_provenance(
    support,
    *,
    support_entry: Mapping[str, Any],
    positive_points: int,
    negative_points: int,
) -> tuple[dict[str, Any], str]:
    prompts, trace_sha256 = build_support_point_prompts(
        support,
        positive_points=positive_points,
        negative_points=negative_points,
    )
    records = []
    for prompt in prompts:
        record = prompt.as_json()
        record["positive_points"] = int((prompt.labels == 1).sum())
        record["negative_points"] = int((prompt.labels == 0).sum())
        records.append(record)
    return (
        {
            "support_arrays_sha256": support_entry["arrays_sha256"],
            "support_asset_trace_sha256": support_entry["asset_trace_sha256"],
            "support_frames": EXPECTED_SUPPORT_FRAMES,
            "role_order": list(support.role_names),
            "adapter": PROMPT_PROTOCOL,
            "point_sampling_algorithm": PROMPT_SAMPLER,
            "sampling_order": (
                "positive mask centroid-nearest then greedy farthest pixels; "
                "negative complement centroid-nearest then greedy farthest pixels"
            ),
            "coordinate_convention": "normalized_xy_pixel_centers_open_unit_interval",
            "positive_points_per_role_frame_requested": positive_points,
            "negative_points_per_role_frame_requested": negative_points,
            "prompt_record_count": len(records),
            "prompt_trace_sha256": trace_sha256,
            "records": records,
        },
        trace_sha256,
    )


def _implementation_provenance() -> dict[str, Any]:
    worker = Path(__file__).resolve()
    implementation = Path(
        sys.modules["tdmpc2.perception.sam31_backend"].__file__
    ).resolve()
    return {
        "worker": str(worker),
        "worker_sha256": file_sha256(worker),
        "adapter": str(implementation),
        "adapter_sha256": file_sha256(implementation),
    }


def _protocol_smoke(args: argparse.Namespace) -> dict[str, Any]:
    if args.inputs is None:
        raise Sam31ContractError("--protocol-smoke-only requires --inputs")
    input_path = args.inputs.expanduser().resolve()
    inputs, support_paths, episode_paths = validate_backend_inputs(
        input_path,
        strict_counts=not args.allow_nonstandard_counts,
    )
    input_sha = file_sha256(input_path)
    cuda = _require_isolated_cuda(args.device, args.seed)
    task = inputs["tasks"][0]
    roles = tuple(inputs["roles"][task])
    support = load_frozen_support(
        support_paths[task],
        role_names=roles,
        expected_frames=EXPECTED_SUPPORT_FRAMES,
    )
    episode_path = episode_paths[(task, inputs["conditions"][0], 0)]
    episode_rgb, _, episode_trace = load_episode_rgb(episode_path)
    smoke_frames = min(8, int(episode_rgb.shape[0]))
    config = OfficialSam31Config(
        repo_path=args.sam3_repo,
        checkpoint_path=args.checkpoint,
        bpe_path=args.bpe,
        role_names=roles,
        device=args.device,
        use_fa3=bool(args.use_fa3),
        use_rope_real=bool(args.use_rope_real),
        compile_model=bool(args.compile_model),
    )
    runtime = OfficialSam31Runtime(config)
    before = dict(runtime.installation)
    with tempfile.TemporaryDirectory(prefix="sam31_protocol_smoke_") as temporary:
        arrays, metadata = run_official_sam31_episode(
            config=config,
            support=support,
            episode_rgb=np.ascontiguousarray(episode_rgb[:smoke_frames]),
            frame_directory=Path(temporary) / "frames",
            positive_points=args.positive_points,
            negative_points=args.negative_points,
            runtime=runtime,
        )
    normalized = _validate_prediction_arrays(
        arrays,
        frames=smoke_frames,
        roles=len(roles),
        resolution=int(inputs["resolution"]),
    )
    nonempty = normalized["predicted_masks"].reshape(smoke_frames, len(roles), -1).any(axis=2)
    per_role_nonempty = nonempty.sum(axis=0).astype(np.int64).tolist()
    if any(value <= 0 for value in per_role_nonempty):
        raise Sam31ContractError(
            "official SAM 3.1 point-only protocol smoke produced an always-empty role: "
            f"{dict(zip(roles, per_role_nonempty))}"
        )
    inputs_after, _, _ = validate_backend_inputs(
        input_path,
        strict_counts=not args.allow_nonstandard_counts,
    )
    if inputs_after != inputs or file_sha256(input_path) != input_sha:
        raise Sam31ContractError("GT-free inputs changed during protocol smoke")
    if runtime.recheck_installation() != before:
        raise Sam31ContractError("SAM 3.1 installation changed during protocol smoke")
    return {
        "format": "sam31_fixed_support_point_protocol_smoke_v1",
        "task": task,
        "roles": list(roles),
        "frames": smoke_frames,
        "episode_rgb_trace_sha256": episode_trace,
        "per_role_nonempty_frames": dict(zip(roles, per_role_nonempty)),
        "prompt_trace_sha256": metadata["prompt_trace_sha256"],
        "checkpoint_contract": before["checkpoint_contract"],
        "cuda": cuda,
        "pass": True,
    }


def _run(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.inputs is None or args.output_root is None:
        raise Sam31ContractError(
            "benchmark execution requires --inputs and --output-root"
        )
    input_path = args.inputs.expanduser().resolve()
    inputs, support_paths, episode_paths = validate_backend_inputs(
        input_path,
        strict_counts=not args.allow_nonstandard_counts,
    )
    input_manifest_sha256 = file_sha256(input_path)
    output_root = args.output_root.expanduser().resolve()
    incomplete_root = output_root.with_name(output_root.name + ".incomplete")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    if incomplete_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing incomplete output: {incomplete_root}"
        )
    if not output_root.parent.is_dir():
        raise FileNotFoundError(f"Output parent does not exist: {output_root.parent}")

    cuda = _require_isolated_cuda(args.device, args.seed)
    resolution = int(inputs["resolution"])
    frames_per_episode = int(inputs["counts"]["frames_per_episode"])
    episode_count = int(inputs["counts"]["episodes"])
    supports = {}
    prompt_provenance = {}
    prompt_trace_by_task = {}
    for task in inputs["tasks"]:
        roles = tuple(inputs["roles"][task])
        support = load_frozen_support(
            support_paths[task],
            role_names=roles,
            expected_frames=EXPECTED_SUPPORT_FRAMES,
        )
        if support.rgb.shape != (
            EXPECTED_SUPPORT_FRAMES,
            resolution,
            resolution,
            3,
        ):
            raise Sam31ContractError(
                f"support resolution changed for {task}: {support.rgb.shape}"
            )
        supports[task] = support
        provenance, trace_sha256 = _prompt_provenance(
            support,
            support_entry=inputs["support"][task],
            positive_points=args.positive_points,
            negative_points=args.negative_points,
        )
        prompt_provenance[task] = provenance
        prompt_trace_by_task[task] = trace_sha256

    first_task = inputs["tasks"][0]
    base_config = OfficialSam31Config(
        repo_path=args.sam3_repo,
        checkpoint_path=args.checkpoint,
        bpe_path=args.bpe,
        role_names=tuple(inputs["roles"][first_task]),
        device=args.device,
        use_fa3=bool(args.use_fa3),
        use_rope_real=bool(args.use_rope_real),
        compile_model=bool(args.compile_model),
    )
    implementation = _implementation_provenance()
    total_started = perf_counter()
    runtime = OfficialSam31Runtime(base_config)
    official_before = dict(runtime.installation)

    incomplete_root.mkdir()
    results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    diagnostics: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for task in inputs["tasks"]:
        roles = tuple(inputs["roles"][task])
        config = OfficialSam31Config(
            repo_path=args.sam3_repo,
            checkpoint_path=args.checkpoint,
            bpe_path=args.bpe,
            role_names=roles,
            device=args.device,
            use_fa3=bool(args.use_fa3),
            use_rope_real=bool(args.use_rope_real),
            compile_model=bool(args.compile_model),
        )
        results[task] = {}
        diagnostics[task] = {}
        for condition in inputs["conditions"]:
            records = []
            diagnostic_records = []
            for episode_index in range(episode_count):
                episode_path = episode_paths[(task, condition, episode_index)]
                episode_rgb, _, episode_rgb_trace = load_episode_rgb(episode_path)
                if episode_rgb.shape != (
                    frames_per_episode,
                    resolution,
                    resolution,
                    3,
                ):
                    raise Sam31ContractError(
                        f"episode RGB shape changed for {task}/{condition}/"
                        f"{episode_index}: {episode_rgb.shape}"
                    )
                print(
                    "SAM31_EPISODE_START "
                    f"task={task} condition={condition} episode={episode_index}",
                    flush=True,
                )
                episode_started = perf_counter()
                with tempfile.TemporaryDirectory(prefix="sam31_fixed_support_") as temporary:
                    arrays, metadata = run_official_sam31_episode(
                        config=config,
                        support=supports[task],
                        episode_rgb=episode_rgb,
                        frame_directory=Path(temporary) / "frames",
                        positive_points=args.positive_points,
                        negative_points=args.negative_points,
                        runtime=runtime,
                    )
                # Include lossless frame staging, full-video init/preprocessing,
                # support replay, propagation, and final-mask decoding. Dataset NPZ
                # I/O and one-time model construction are excluded for every backend.
                episode_wallclock_ms = (perf_counter() - episode_started) * 1000.0
                raw_runtime = np.ascontiguousarray(arrays["runtime_ms"], dtype=np.float64)
                overhead_ms = max(0.0, episode_wallclock_ms - float(raw_runtime.sum()))
                arrays["runtime_ms"] = raw_runtime + overhead_ms / raw_runtime.shape[0]
                if metadata["prompt_trace_sha256"] != prompt_trace_by_task[task]:
                    raise Sam31ContractError("support prompt trace changed during inference")
                normalized = _validate_prediction_arrays(
                    arrays,
                    frames=frames_per_episode,
                    roles=len(roles),
                    resolution=resolution,
                )
                relative = (
                    Path("predictions")
                    / task
                    / condition
                    / f"episode_{episode_index:03d}.npz"
                )
                destination = incomplete_root / relative
                _save_prediction(destination, normalized)
                _validate_saved_prediction(
                    destination,
                    frames=frames_per_episode,
                    roles=len(roles),
                    resolution=resolution,
                )
                mask_trace = typed_array_sha256(normalized["predicted_masks"])
                runtime_trace = typed_array_sha256(normalized["runtime_ms"])
                record = {
                    "episode_index": episode_index,
                    "frames": frames_per_episode,
                    "prediction_arrays": relative.as_posix(),
                    "prediction_arrays_sha256": file_sha256(destination),
                    "predicted_mask_trace_sha256": mask_trace,
                    "runtime_trace_sha256": runtime_trace,
                }
                records.append(record)
                diagnostic_records.append(
                    {
                        "episode_index": episode_index,
                        "episode_rgb_typed_sha256": episode_rgb_trace,
                        "support_replay_runtime_ms": metadata["support_runtime_ms"],
                        "mean_episode_frame_runtime_ms": float(
                            normalized["runtime_ms"].mean()
                        ),
                        "lost_frames_by_role": normalized["reported_lost"]
                        .sum(axis=0)
                        .astype(np.int64)
                        .tolist(),
                        "overlap_pixels_before_resolution_total": metadata[
                            "overlap_pixels_before_resolution_total"
                        ],
                        "start_session_dispatch": metadata["start_session_dispatch"],
                    }
                )
                print(
                    "SAM31_EPISODE_END "
                    f"task={task} condition={condition} episode={episode_index} "
                    f"ms_per_frame={normalized['runtime_ms'].mean():.3f} "
                    f"mask_trace_sha256={mask_trace}",
                    flush=True,
                )
            results[task][condition] = records
            diagnostics[task][condition] = diagnostic_records

    # Revalidate every GT-free input and every official source/weight hash after
    # inference. This never discovers or opens the sibling scoring tree.
    inputs_after, _, _ = validate_backend_inputs(
        input_path,
        strict_counts=not args.allow_nonstandard_counts,
    )
    if inputs_after != inputs or file_sha256(input_path) != input_manifest_sha256:
        raise Sam31ContractError("GT-free backend inputs changed during inference")
    official_after = runtime.recheck_installation()
    if official_after != official_before:
        raise Sam31ContractError("official SAM 3.1 sources or weights changed during inference")
    implementation_after = _implementation_provenance()
    if implementation_after != implementation:
        raise Sam31ContractError("SAM 3.1 adapter implementation changed during inference")

    protocol = {
        "support_only_prompts": True,
        "support_frames_replayed_each_episode": EXPECTED_SUPPORT_FRAMES,
        "episode_first_frame_gt_prompt": False,
        "mid_episode_reprompt": False,
        "episode_ground_truth_read": False,
        "causal_frame_order": True,
        "model_internal_features_exported": False,
        "reported_confidence_cross_backend_comparable": False,
        "prompt_adapter": PROMPT_PROTOCOL,
        "point_sampling_algorithm": PROMPT_SAMPLER,
        "official_api_mask_prompt": False,
        "prompt_api_not_equivalent": True,
        "comparison_class": "offline_fixed_video_backend_diagnostic",
        "propagation_protocol": CAUSAL_PROTOCOL,
        "propagation_direction": "forward_only",
        "official_max_frame_num_to_track_argument": 0,
        "effective_frames_per_propagation_call": 1,
        "strict_causal_model_inference": False,
        "strict_source_pixel_access_gate": False,
        "online_deployment_eligible": False,
        "runtime_ms_semantics": (
            "episode_end_to_end_amortized_per_frame_excluding_npz_io_"
            "and_model_construction_v1"
        ),
        "source_pixel_access_limitation": (
            "official init_state preprocesses the complete staged RGB sequence; "
            "future frames are not passed through model propagation before their turn"
        ),
        "confidence_semantics": "official_out_probs_diagnostic_only",
    }
    manifest = {
        "format": BACKEND_FORMAT,
        "status": "complete",
        "backend": "sam31",
        "dataset_id": inputs["dataset_id"],
        "input_manifest_sha256": input_manifest_sha256,
        "protocol": protocol,
        "backend_provenance": {
            "model_family": "sam3.1_multiplex",
            "checkpoint_sha256": official_before["checkpoint_sha256"],
            "device_name": cuda["device_name"],
            "cuda_visible_devices": cuda["cuda_visible_devices"],
            "gpu_uuid": os.environ.get("BENCHMARK_GPU_UUID"),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
            "logical_cuda_device": cuda["logical_cuda_device"],
            "seed": cuda["seed"],
            "python": sys.version,
            "platform": platform.platform(),
            "implementation": implementation,
            "official_installation": official_before,
            "model_options": {
                "use_fa3": bool(args.use_fa3),
                "use_rope_real": bool(args.use_rope_real),
                "compile": bool(args.compile_model),
                "output_probability_threshold": float(
                    base_config.output_probability_threshold
                ),
            },
            "support_point_replay": prompt_provenance,
            "episode_diagnostics": diagnostics,
            "wallclock_seconds": perf_counter() - total_started,
        },
        "results": results,
    }
    manifest_path = incomplete_root / "backend_predictions.json"
    write_json(manifest_path, manifest)
    os.replace(incomplete_root, output_root)
    return output_root, output_root / manifest_path.name


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.preflight_only and args.protocol_smoke_only:
            raise Sam31ContractError(
                "--preflight-only and --protocol-smoke-only are mutually exclusive"
            )
        if args.preflight_only:
            cuda = _require_isolated_cuda(args.device, args.seed)
            config = OfficialSam31Config(
                repo_path=args.sam3_repo,
                checkpoint_path=args.checkpoint,
                bpe_path=args.bpe,
                role_names=("role_1", "role_2"),
                device=args.device,
                use_fa3=bool(args.use_fa3),
                use_rope_real=bool(args.use_rope_real),
                compile_model=bool(args.compile_model),
            )
            runtime = OfficialSam31Runtime(config)
            report = {"cuda": cuda, "installation": runtime.installation}
            print("SAM31_UNIFIED_BACKEND_PREFLIGHT_OK", flush=True)
            print(report, flush=True)
            return 0
        if args.protocol_smoke_only:
            report = _protocol_smoke(args)
            print("SAM31_UNIFIED_BACKEND_PROTOCOL_SMOKE_OK", flush=True)
            print(report, flush=True)
            return 0
        output_root, manifest_path = _run(args)
    except (Sam31BackendError, ValueError, OSError, RuntimeError) as exc:
        if args.traceback:
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        print(
            f"SAM31_UNIFIED_BACKEND_FAILED {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 4
    print("SAM31_UNIFIED_BACKEND_COMPLETE")
    print(f"OUTPUT_ROOT={output_root}")
    print(f"PREDICTIONS={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
