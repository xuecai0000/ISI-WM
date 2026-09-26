"""Run the Cutie baseline on GT-free frozen unified-VOS inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
for local_path in (str(REPO_DIR), str(PROJECT_DIR)):
	while local_path in sys.path:
		sys.path.remove(local_path)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))

from tdmpc2.common.unified_vos import (  # noqa: E402
	BACKEND_FORMAT,
	file_sha256,
	validate_backend_inputs,
	write_json,
)


def _array_trace(value: np.ndarray) -> str:
	array = np.ascontiguousarray(value)
	digest = hashlib.sha256()
	digest.update(str(array.dtype).encode("ascii"))
	digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
	digest.update(array.tobytes())
	return digest.hexdigest()


def _tree_snapshot(root: Path) -> dict[str, Any]:
	root = root.expanduser().resolve()
	if not root.is_dir():
		raise FileNotFoundError(root)
	paths = sorted(
		path for path in root.rglob("*")
		if path.is_file() and path.suffix.lower() in {".py", ".yaml", ".yml"}
	)
	if not paths:
		raise ValueError(f"Cutie implementation tree is empty: {root}")
	files = [{
		"path": path.relative_to(root).as_posix(),
		"bytes": path.stat().st_size,
		"sha256": file_sha256(path),
	} for path in paths]
	digest = hashlib.sha256(
		(json.dumps(files, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
	).hexdigest()
	return {"root": str(root), "files": files, "tree_sha256": digest}


def _load_npz(path: Path, expected_keys: set[str]) -> dict[str, np.ndarray]:
	with np.load(path, allow_pickle=False) as archive:
		if set(archive.files) != expected_keys:
			raise ValueError(f"Unexpected arrays in {path}: {archive.files}.")
		return {key: np.ascontiguousarray(archive[key]) for key in archive.files}


def _save_prediction(
	path: Path,
	*,
	predicted_masks: np.ndarray,
	reported_confidence: np.ndarray,
	reported_lost: np.ndarray,
	runtime_ms: np.ndarray,
) -> str:
	path.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(
		path,
		predicted_masks=np.ascontiguousarray(predicted_masks, dtype=np.bool_),
		reported_confidence=np.ascontiguousarray(reported_confidence, dtype=np.float32),
		reported_lost=np.ascontiguousarray(reported_lost, dtype=np.bool_),
		runtime_ms=np.ascontiguousarray(runtime_ms, dtype=np.float64),
	)
	return file_sha256(path)


def run(args) -> dict[str, Any]:
	import torch
	from tdmpc2.perception.cutie_oc_adapter import (
		CutieOCAdapter,
		CutieOCConfig,
		CutieSupportPrompts,
	)

	input_path = args.inputs.expanduser().resolve()
	input_manifest_sha = file_sha256(input_path)
	inputs, support_paths, episode_paths = validate_backend_inputs(
		input_path, strict_counts=args.strict_counts
	)
	output_root = args.output_root.expanduser().resolve()
	incomplete_root = output_root.with_name(output_root.name + ".incomplete")
	if output_root.exists():
		raise FileExistsError(args.output_root)
	if incomplete_root.exists():
		raise FileExistsError(incomplete_root)
	if not output_root.parent.is_dir():
		raise FileNotFoundError(output_root.parent)
	if not args.oc_storm_repo.is_dir():
		raise FileNotFoundError(args.oc_storm_repo)
	if not args.checkpoint.is_file():
		raise FileNotFoundError(args.checkpoint)
	if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
		raise RuntimeError(
			"Cutie backend requires exactly one logical CUDA device; set "
			"CUDA_VISIBLE_DEVICES to one physical GPU."
		)
	if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
		raise RuntimeError("CUDA_DEVICE_ORDER must be PCI_BUS_ID.")
	if not str(os.environ.get("BENCHMARK_GPU_UUID", "")).startswith("GPU-"):
		raise RuntimeError("BENCHMARK_GPU_UUID must bind the physical GPU.")
	if len({args.seed}) != 1 or args.seed < 0:
		raise ValueError("Cutie seed must be non-negative.")
	incomplete_root.mkdir(exist_ok=False)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	cutie_root = (
		args.oc_storm_repo / "feature_extractor" / "cutie" / "cutie"
	).resolve()
	implementation_before = _tree_snapshot(cutie_root)
	checkpoint_sha = file_sha256(args.checkpoint)
	resolution = int(inputs["resolution"])
	results: dict[str, dict[str, list[dict[str, Any]]]] = {}
	runtime_by_task = {}
	total_started = perf_counter()
	for task in inputs["tasks"]:
		roles = tuple(inputs["roles"][task])
		support_arrays = _load_npz(
			support_paths[task], {"rgb", "indexed_masks"}
		)
		frames = support_arrays["rgb"]
		masks = support_arrays["indexed_masks"]
		if frames.shape != (6, resolution, resolution, 3):
			raise RuntimeError(f"{task}: invalid support RGB shape {frames.shape}.")
		if masks.shape != (6, resolution, resolution):
			raise RuntimeError(f"{task}: invalid support mask shape {masks.shape}.")
		support = CutieSupportPrompts(
			frames=tuple(np.array(value, copy=True, order="C") for value in frames),
			masks=tuple(np.array(value, copy=True, order="C") for value in masks),
			role_names=roles,
			annotation_path=support_paths[task],
			metadata={
				"format": "unified_vos_support_arrays_v1",
				"task": task,
				"prompt_count": 6,
				"source_resolution": [resolution, resolution],
			},
		)
		config = CutieOCConfig(
			repo_path=args.oc_storm_repo,
			checkpoint_path=args.checkpoint,
			role_names=roles,
			model_size=args.model_size,
			device="cuda:0",
			output_device="cpu",
			expected_input_size=(resolution, resolution),
			support_input_size=(resolution, resolution),
			mask_output_size=(resolution, resolution),
			tracker_size=(args.tracker_size, args.tracker_size),
			foreground_queries=8,
			amp=not args.disable_amp,
			return_object_features=False,
			object_schema="generic_indexed_v1",
		)
		adapter = CutieOCAdapter(config)
		adapter.add_support_prompts(support)
		results[task] = {}
		for condition in inputs["conditions"]:
			records = []
			for episode_index in range(int(inputs["counts"]["episodes"])):
				episode = _load_npz(
					episode_paths[(task, condition, episode_index)], {"rgb"}
				)["rgb"]
				expected_shape = (
					int(inputs["counts"]["frames_per_episode"]),
					resolution,
					resolution,
					3,
				)
				if episode.shape != expected_shape or episode.dtype != np.uint8:
					raise RuntimeError(
						f"{task}/{condition}/{episode_index}: RGB shape changed: "
						f"{episode.shape} {episode.dtype}."
					)
				episode_started = perf_counter()
				adapter.reset_episode()
				prediction = np.zeros(
					(expected_shape[0], len(roles), resolution, resolution), dtype=np.bool_
				)
				confidence = np.empty((expected_shape[0], len(roles)), dtype=np.float32)
				lost = np.empty((expected_shape[0], len(roles)), dtype=np.bool_)
				runtime_ms = np.empty(expected_shape[0], dtype=np.float64)
				for frame_index, frame in enumerate(episode):
					result = adapter.track(frame)
					prediction[frame_index] = result.masks.numpy()
					confidence[frame_index] = result.confidence.numpy()
					lost[frame_index] = result.lost.numpy()
					runtime_ms[frame_index] = float(result.runtime_ms)
				episode_wallclock_ms = (perf_counter() - episode_started) * 1000.0
				overhead_ms = max(0.0, episode_wallclock_ms - float(runtime_ms.sum()))
				runtime_ms += overhead_ms / expected_shape[0]
				if not np.isfinite(confidence).all() or not np.isfinite(runtime_ms).all():
					raise RuntimeError("Cutie emitted non-finite diagnostics.")
				relative = (
					Path("predictions") / task / condition / f"episode_{episode_index:03d}.npz"
				)
				prediction_sha = _save_prediction(
					incomplete_root / relative,
					predicted_masks=prediction,
					reported_confidence=confidence,
					reported_lost=lost,
					runtime_ms=runtime_ms,
				)
				record = {
					"episode_index": episode_index,
					"frames": expected_shape[0],
					"prediction_arrays": relative.as_posix(),
					"prediction_arrays_sha256": prediction_sha,
					"predicted_mask_trace_sha256": _array_trace(prediction),
					"runtime_trace_sha256": _array_trace(runtime_ms),
				}
				records.append(record)
				print("UNIFIED_VOS_BACKEND_EPISODE", json.dumps({
					"backend": "cutie",
					"task": task,
					"condition": condition,
					"episode_index": episode_index,
					"mask_trace_sha256": record["predicted_mask_trace_sha256"],
				}, allow_nan=False), flush=True)
			results[task][condition] = records
		runtime_by_task[task] = adapter.runtime_summary()
		del adapter
		torch.cuda.empty_cache()

	implementation_after = _tree_snapshot(cutie_root)
	if implementation_after != implementation_before:
		raise RuntimeError("Cutie implementation changed during inference.")
	if file_sha256(args.checkpoint) != checkpoint_sha:
		raise RuntimeError("Cutie checkpoint changed during inference.")
	inputs_after, _, _ = validate_backend_inputs(
		input_path, strict_counts=args.strict_counts
	)
	if inputs_after != inputs or file_sha256(input_path) != input_manifest_sha:
		raise RuntimeError("GT-free backend inputs changed during Cutie inference.")
	payload = {
		"format": BACKEND_FORMAT,
		"status": "complete",
		"backend": "cutie",
		"dataset_id": inputs["dataset_id"],
		"input_manifest_sha256": input_manifest_sha,
		"protocol": {
			"support_only_prompts": True,
			"support_frames_replayed_each_episode": 6,
			"episode_first_frame_gt_prompt": False,
			"mid_episode_reprompt": False,
			"episode_ground_truth_read": False,
			"causal_frame_order": True,
			"prompt_adapter": "exact_indexed_mask_permanent_support_replay_v1",
			"model_internal_features_exported": False,
			"reported_confidence_cross_backend_comparable": False,
			"strict_source_pixel_access_gate": True,
			"online_deployment_eligible": True,
			"runtime_ms_semantics": (
				"episode_end_to_end_amortized_per_frame_excluding_npz_io_"
				"and_model_construction_v1"
			),
		},
		"backend_provenance": {
			"model_family": "cutie",
			"model_size": args.model_size,
			"tracker_size": [args.tracker_size, args.tracker_size],
			"oc_storm_repo": str(args.oc_storm_repo),
			"implementation": implementation_before,
			"checkpoint": str(args.checkpoint),
			"checkpoint_sha256": checkpoint_sha,
			"seed": args.seed,
			"python": sys.version,
			"platform": platform.platform(),
			"torch": torch.__version__,
			"cuda_runtime": torch.version.cuda,
			"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
			"gpu_uuid": os.environ.get("BENCHMARK_GPU_UUID"),
			"cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
			"logical_cuda_device": 0,
			"device_name": torch.cuda.get_device_name(0),
			"amp": not args.disable_amp,
			"wallclock_seconds": perf_counter() - total_started,
			"runtime_by_task": runtime_by_task,
		},
		"results": results,
	}
	write_json(incomplete_root / "backend_predictions.json", payload)
	os.replace(incomplete_root, output_root)
	return payload


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--inputs", type=Path)
	parser.add_argument("--output-root", type=Path)
	parser.add_argument("--oc-storm-repo", type=Path, required=True)
	parser.add_argument("--checkpoint", type=Path, required=True)
	parser.add_argument("--model-size", choices=("small", "base"), default="small")
	parser.add_argument("--tracker-size", type=int, default=448)
	parser.add_argument("--seed", type=int, default=2718281)
	parser.add_argument("--disable-amp", action="store_true")
	parser.add_argument("--strict-counts", action="store_true")
	parser.add_argument("--preflight-only", action="store_true")
	return parser


def main() -> None:
	args = build_parser().parse_args()
	for name in ("inputs", "output_root", "oc_storm_repo", "checkpoint"):
		value = getattr(args, name)
		if value is not None:
			setattr(args, name, value.expanduser().resolve())
	if args.preflight_only:
		import torch
		from tdmpc2.perception.cutie_oc_adapter import (
			CutieOCAdapter,
			CutieOCConfig,
			inspect_cutie_installation,
		)

		if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
			raise RuntimeError("CUDA_DEVICE_ORDER must be PCI_BUS_ID.")
		if not str(os.environ.get("BENCHMARK_GPU_UUID", "")).startswith("GPU-"):
			raise RuntimeError("BENCHMARK_GPU_UUID must bind the physical GPU.")
		config = CutieOCConfig(
			repo_path=args.oc_storm_repo,
			checkpoint_path=args.checkpoint,
			role_names=("role_1", "role_2"),
			model_size=args.model_size,
			device="cuda:0",
			output_device="cpu",
			expected_input_size=(128, 128),
			support_input_size=(128, 128),
			mask_output_size=(128, 128),
			tracker_size=(args.tracker_size, args.tracker_size),
			return_object_features=False,
			object_schema="generic_indexed_v1",
		)
		report = inspect_cutie_installation(config, import_check=True)
		adapter = CutieOCAdapter(config)
		del adapter
		torch.cuda.empty_cache()
		print("UNIFIED_VOS_CUTIE_PREFLIGHT_OK", json.dumps(report, allow_nan=False))
		return
	if args.inputs is None or args.output_root is None:
		raise ValueError("benchmark execution requires --inputs and --output-root")
	payload = run(args)
	print("UNIFIED_VOS_CUTIE_COMPLETE", json.dumps({
		"dataset_id": payload["dataset_id"],
		"manifest": str(args.output_root / "backend_predictions.json"),
	}, allow_nan=False), flush=True)


if __name__ == "__main__":
	main()
