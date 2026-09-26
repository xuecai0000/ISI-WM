"""Run support-conditioned object-graph Cutie on frozen GT-free VOS inputs.

This is deliberately a new backend rather than a modification of the
historical two-role Cutie baseline.  Fixed support role masks are projected to
declared tracking entities, Cutie tracks those entities causally, and the
support-conditioned graph tokenizer projects the current entity result back to
ordered semantic role masks and 590-D role descriptors.  Episode ground truth
is not accepted by this process.
"""

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


OBJECT_GRAPH_BACKEND_FORMAT = "object_graph_cutie_backend_predictions_v1"
BACKEND_NAME = "object_graph_cutie"
PREDICTION_KEYS = {
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


def _graph_tree_snapshot(root: Path) -> dict[str, Any]:
	root = root.expanduser().resolve()
	if not root.is_dir() or root.is_symlink():
		raise ValueError(f"Graph directory must be a regular directory: {root}")
	paths = sorted(root.glob("*.json"))
	if not paths:
		raise ValueError(f"Graph directory contains no JSON files: {root}")
	files = []
	for path in paths:
		if not path.is_file() or path.is_symlink():
			raise ValueError(f"Graph inputs must be regular non-symlink files: {path}")
		files.append({
			"path": path.relative_to(root).as_posix(),
			"bytes": path.stat().st_size,
			"sha256": file_sha256(path),
		})
	digest = hashlib.sha256(
		(json.dumps(files, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
	).hexdigest()
	return {"root": str(root), "files": files, "tree_sha256": digest}


def _implementation_snapshot() -> dict[str, Any]:
	paths = (
		Path(__file__).resolve(),
		PROJECT_DIR / "perception" / "support_conditioned_object_graph.py",
		PROJECT_DIR / "perception" / "cutie_oc_adapter.py",
		PROJECT_DIR / "common" / "unified_vos.py",
	)
	result = {}
	for path in paths:
		if not path.is_file() or path.is_symlink():
			raise ValueError(f"Backend implementation must be a regular file: {path}")
		result[path.relative_to(REPO_DIR).as_posix()] = {
			"bytes": path.stat().st_size,
			"sha256": file_sha256(path),
		}
	return result


def _load_npz(path: Path, expected_keys: set[str]) -> dict[str, np.ndarray]:
	with np.load(path, allow_pickle=False) as archive:
		if set(archive.files) != expected_keys:
			raise ValueError(f"Unexpected arrays in {path}: {archive.files}.")
		return {key: np.ascontiguousarray(archive[key]) for key in archive.files}


def _validate_cuda_binding(torch) -> None:
	if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
		raise RuntimeError(
			"Object-graph Cutie requires exactly one logical CUDA device; set "
			"CUDA_VISIBLE_DEVICES to one physical GPU."
		)
	if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
		raise RuntimeError("CUDA_DEVICE_ORDER must be PCI_BUS_ID.")
	if not str(os.environ.get("BENCHMARK_GPU_UUID", "")).startswith("GPU-"):
		raise RuntimeError("BENCHMARK_GPU_UUID must bind the physical GPU.")


def _load_graphs(graph_dir: Path, tasks: tuple[str, ...], roles: dict[str, Any]):
	from tdmpc2.perception.support_conditioned_object_graph import load_object_graph

	graphs = {}
	paths = {}
	for path in sorted(graph_dir.glob("*.json")):
		graph = load_object_graph(path)
		if graph.task not in tasks:
			continue
		if graph.task in graphs:
			raise ValueError(f"Duplicate graph for task {graph.task!r} in {graph_dir}.")
		expected_roles = tuple(roles[graph.task])
		if graph.source_roles != expected_roles or graph.semantic_roles != expected_roles:
			raise ValueError(
				f"{graph.task}: graph roles {graph.source_roles!r}/{graph.semantic_roles!r} "
				f"do not equal frozen input roles {expected_roles!r}."
			)
		if len(graph.semantic_roles) != 2:
			raise ValueError(f"{graph.task}: V1 benchmark requires exactly two role tokens.")
		graphs[graph.task] = graph
		paths[graph.task] = path.resolve()
	missing = set(tasks) - set(graphs)
	if missing:
		raise ValueError(f"Missing object graphs for frozen tasks: {sorted(missing)!r}.")
	return graphs, paths


def _validate_prediction_arrays(
	arrays: dict[str, np.ndarray],
	*,
	frames: int,
	roles: int,
	entities: int,
	resolution: int,
) -> None:
	if set(arrays) != PREDICTION_KEYS:
		raise RuntimeError("Object-graph prediction array schema changed.")
	expected = {
		"role_masks": ((frames, roles, resolution, resolution), np.bool_),
		"entity_masks": ((frames, entities, resolution, resolution), np.bool_),
		"descriptors": ((frames, roles, 590), np.float32),
		"keypoints_xy": ((frames, roles, 2, 2), np.float32),
		"role_valid": ((frames, roles), np.bool_),
		"role_confidence": ((frames, roles), np.float32),
		"role_lost": ((frames, roles), np.bool_),
		"role_mask_score": ((frames, roles), np.float32),
		"entity_valid": ((frames, entities), np.bool_),
		"entity_confidence": ((frames, entities), np.float32),
		"entity_lost": ((frames, entities), np.bool_),
		"entity_mask_score": ((frames, entities), np.float32),
		"cutie_runtime_ms": ((frames,), np.float64),
		"parser_runtime_ms": ((frames,), np.float64),
		"end_to_end_runtime_ms": ((frames,), np.float64),
	}
	for name, (shape, dtype) in expected.items():
		value = arrays[name]
		if value.shape != shape or value.dtype != dtype:
			raise RuntimeError(
				f"{name} contract changed: {value.shape}/{value.dtype}, expected {shape}/{dtype}."
			)
		if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
			raise RuntimeError(f"{name} contains non-finite values.")
	if np.any(arrays["role_masks"].astype(np.uint8).sum(axis=1) > 1):
		raise RuntimeError("Semantic role masks overlap.")
	if np.any(arrays["entity_masks"].astype(np.uint8).sum(axis=1) > 1):
		raise RuntimeError("Tracking entity masks overlap.")
	if np.any(arrays["role_valid"] & arrays["role_lost"]):
		raise RuntimeError("A lost semantic role cannot be valid.")
	if np.any(arrays["entity_valid"] & arrays["entity_lost"]):
		raise RuntimeError("A lost tracking entity cannot be valid.")
	for name in ("role_confidence", "role_mask_score", "entity_confidence", "entity_mask_score"):
		if np.any((arrays[name] < 0.0) | (arrays[name] > 1.0)):
			raise RuntimeError(f"{name} must remain in [0,1].")
	for name in ("cutie_runtime_ms", "parser_runtime_ms", "end_to_end_runtime_ms"):
		if np.any(arrays[name] < 0.0):
			raise RuntimeError(f"{name} cannot be negative.")


def _save_prediction(path: Path, arrays: dict[str, np.ndarray]) -> str:
	path.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(path, **{key: np.ascontiguousarray(arrays[key]) for key in sorted(arrays)})
	return file_sha256(path)


def run(args) -> dict[str, Any]:
	import torch
	from tdmpc2.perception.cutie_oc_adapter import (
		CutieOCAdapter,
		CutieOCConfig,
		CutieSupportPrompts,
	)
	from tdmpc2.perception.support_conditioned_object_graph import (
		FRAME_DIM,
		SupportConditionedObjectGraphTokenizer,
		project_support_to_entities,
	)

	if FRAME_DIM != 590:
		raise RuntimeError("Object-graph descriptor frame dimension changed.")
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
	if not args.checkpoint.is_file() or args.checkpoint.is_symlink():
		raise FileNotFoundError(args.checkpoint)
	if not args.graph_dir.is_dir() or args.graph_dir.is_symlink():
		raise FileNotFoundError(args.graph_dir)
	_validate_cuda_binding(torch)
	if args.seed < 0:
		raise ValueError("Cutie seed must be non-negative.")

	tasks = tuple(inputs["tasks"])
	graph_tree_before = _graph_tree_snapshot(args.graph_dir)
	graphs, graph_paths = _load_graphs(args.graph_dir, tasks, inputs["roles"])
	implementation_before = _implementation_snapshot()
	cutie_root = (
		args.oc_storm_repo / "feature_extractor" / "cutie" / "cutie"
	).resolve()
	cutie_before = _tree_snapshot(cutie_root)
	checkpoint_sha = file_sha256(args.checkpoint)
	incomplete_root.mkdir(exist_ok=False)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	resolution = int(inputs["resolution"])
	frame_count = int(inputs["counts"]["frames_per_episode"])
	results: dict[str, dict[str, list[dict[str, Any]]]] = {}
	graph_manifest: dict[str, Any] = {}
	runtime_by_task = {}
	total_started = perf_counter()
	for task in tasks:
		graph = graphs[task]
		roles = tuple(inputs["roles"][task])
		support_arrays = _load_npz(support_paths[task], {"rgb", "indexed_masks"})
		frames = support_arrays["rgb"]
		source_masks = support_arrays["indexed_masks"]
		if frames.shape != (6, resolution, resolution, 3) or frames.dtype != np.uint8:
			raise RuntimeError(f"{task}: invalid support RGB {frames.shape}/{frames.dtype}.")
		if source_masks.shape != (6, resolution, resolution):
			raise RuntimeError(f"{task}: invalid support mask shape {source_masks.shape}.")
		entity_support = project_support_to_entities(frames, source_masks, graph)
		support = CutieSupportPrompts(
			frames=tuple(np.array(value, copy=True, order="C") for value in entity_support.rgb),
			masks=tuple(np.array(value, copy=True, order="C") for value in entity_support.indexed_masks),
			role_names=graph.entity_names,
			annotation_path=support_paths[task],
			metadata={
				**entity_support.metadata,
				"task": task,
				"source_support_arrays_sha256": file_sha256(support_paths[task]),
			},
		)
		config = CutieOCConfig(
			repo_path=args.oc_storm_repo,
			checkpoint_path=args.checkpoint,
			role_names=graph.entity_names,
			model_size=args.model_size,
			device="cuda:0",
			output_device="cpu",
			expected_input_size=(resolution, resolution),
			support_input_size=(resolution, resolution),
			mask_output_size=(resolution, resolution),
			tracker_size=(args.tracker_size, args.tracker_size),
			foreground_queries=8,
			amp=not args.disable_amp,
			return_object_features=True,
			object_schema="generic_entity_indexed_v1",
		)
		adapter = CutieOCAdapter(config)
		adapter.add_support_prompts(support)
		tokenizer = SupportConditionedObjectGraphTokenizer(graph, source_masks)
		graph_manifest[task] = {
			"graph_path": str(graph_paths[task]),
			"graph_file_sha256": file_sha256(graph_paths[task]),
			"graph": graph.metadata(),
			"entity_support": entity_support.metadata,
			"tokenizer": tokenizer.metadata(),
		}
		results[task] = {}
		for condition in inputs["conditions"]:
			records = []
			for episode_index in range(int(inputs["counts"]["episodes"])):
				episode = _load_npz(
					episode_paths[(task, condition, episode_index)], {"rgb"}
				)["rgb"]
				expected_shape = (frame_count, resolution, resolution, 3)
				if episode.shape != expected_shape or episode.dtype != np.uint8:
					raise RuntimeError(
						f"{task}/{condition}/{episode_index}: RGB changed: "
						f"{episode.shape}/{episode.dtype}."
					)
				entity_count = len(graph.entities)
				role_count = len(roles)
				arrays = {
					"role_masks": np.zeros((frame_count, role_count, resolution, resolution), np.bool_),
					"entity_masks": np.zeros((frame_count, entity_count, resolution, resolution), np.bool_),
					"descriptors": np.zeros((frame_count, role_count, FRAME_DIM), np.float32),
					"keypoints_xy": np.zeros((frame_count, role_count, 2, 2), np.float32),
					"role_valid": np.zeros((frame_count, role_count), np.bool_),
					"role_confidence": np.zeros((frame_count, role_count), np.float32),
					"role_lost": np.ones((frame_count, role_count), np.bool_),
					"role_mask_score": np.zeros((frame_count, role_count), np.float32),
					"entity_valid": np.zeros((frame_count, entity_count), np.bool_),
					"entity_confidence": np.zeros((frame_count, entity_count), np.float32),
					"entity_lost": np.ones((frame_count, entity_count), np.bool_),
					"entity_mask_score": np.zeros((frame_count, entity_count), np.float32),
					"cutie_runtime_ms": np.zeros(frame_count, np.float64),
					"parser_runtime_ms": np.zeros(frame_count, np.float64),
					"end_to_end_runtime_ms": np.zeros(frame_count, np.float64),
				}
				episode_started = perf_counter()
				adapter.reset_episode()
				tokenizer.reset_episode()
				for frame_index, frame in enumerate(episode):
					frame_started = perf_counter()
					result = adapter.track(frame)
					if result.object_features is None:
						raise RuntimeError("Cutie did not export required entity query features.")
					entity_masks = result.masks.numpy()
					entity_features = result.object_features.detach().cpu().numpy()
					entity_lost = result.lost.numpy()
					entity_confidence = result.confidence.numpy()
					entity_mask_score = result.mask_score.numpy()
					token = tokenizer.project(
						entity_masks=entity_masks,
						entity_features=entity_features,
						entity_lost=entity_lost,
						entity_confidence=entity_confidence,
						entity_mask_score=entity_mask_score,
					)
					if token.role_names != roles:
						raise RuntimeError("Tokenizer role order changed during inference.")
					arrays["entity_masks"][frame_index] = entity_masks
					arrays["entity_lost"][frame_index] = entity_lost
					arrays["entity_confidence"][frame_index] = entity_confidence
					arrays["entity_mask_score"][frame_index] = entity_mask_score
					arrays["entity_valid"][frame_index] = (
						(~entity_lost)
						& entity_masks.reshape(entity_count, -1).any(axis=1)
						& np.isfinite(entity_features).all(axis=1)
					)
					arrays["role_masks"][frame_index] = token.masks
					arrays["descriptors"][frame_index] = token.descriptors
					arrays["keypoints_xy"][frame_index] = token.keypoints_xy
					arrays["role_valid"][frame_index] = token.valid
					arrays["role_confidence"][frame_index] = token.confidence
					arrays["role_lost"][frame_index] = token.lost
					arrays["role_mask_score"][frame_index] = token.mask_score
					arrays["cutie_runtime_ms"][frame_index] = float(result.runtime_ms)
					arrays["parser_runtime_ms"][frame_index] = float(token.runtime_ms)
					arrays["end_to_end_runtime_ms"][frame_index] = (
						perf_counter() - frame_started
					) * 1000.0
				episode_wallclock_ms = (perf_counter() - episode_started) * 1000.0
				overhead_ms = max(
					0.0,
					episode_wallclock_ms - float(arrays["end_to_end_runtime_ms"].sum()),
				)
				arrays["end_to_end_runtime_ms"] += overhead_ms / frame_count
				_validate_prediction_arrays(
					arrays,
					frames=frame_count,
					roles=role_count,
					entities=entity_count,
					resolution=resolution,
				)
				relative = (
					Path("predictions") / task / condition / f"episode_{episode_index:03d}.npz"
				)
				prediction_sha = _save_prediction(incomplete_root / relative, arrays)
				traces = {
					"role_mask_trace_sha256": _array_trace(arrays["role_masks"]),
					"entity_mask_trace_sha256": _array_trace(arrays["entity_masks"]),
					"descriptor_trace_sha256": _array_trace(arrays["descriptors"]),
					"keypoint_trace_sha256": _array_trace(arrays["keypoints_xy"]),
					"role_status_trace_sha256": _array_trace(np.concatenate((
						arrays["role_valid"].astype(np.float32)[..., None],
						arrays["role_confidence"][..., None],
						arrays["role_lost"].astype(np.float32)[..., None],
						arrays["role_mask_score"][..., None],
					), axis=-1)),
					"entity_status_trace_sha256": _array_trace(np.concatenate((
						arrays["entity_valid"].astype(np.float32)[..., None],
						arrays["entity_confidence"][..., None],
						arrays["entity_lost"].astype(np.float32)[..., None],
						arrays["entity_mask_score"][..., None],
					), axis=-1)),
					"cutie_runtime_trace_sha256": _array_trace(arrays["cutie_runtime_ms"]),
					"parser_runtime_trace_sha256": _array_trace(arrays["parser_runtime_ms"]),
					"end_to_end_runtime_trace_sha256": _array_trace(arrays["end_to_end_runtime_ms"]),
				}
				record = {
					"episode_index": episode_index,
					"frames": frame_count,
					"entity_count": entity_count,
					"role_count": role_count,
					"prediction_arrays": relative.as_posix(),
					"prediction_arrays_sha256": prediction_sha,
					"array_shapes": {
						name: list(value.shape) for name, value in sorted(arrays.items())
					},
					"traces": traces,
				}
				records.append(record)
				print("OBJECT_GRAPH_CUTIE_EPISODE", json.dumps({
					"task": task,
					"condition": condition,
					"episode_index": episode_index,
					"role_mask_trace_sha256": traces["role_mask_trace_sha256"],
					"entity_mask_trace_sha256": traces["entity_mask_trace_sha256"],
					"descriptor_trace_sha256": traces["descriptor_trace_sha256"],
				}, allow_nan=False), flush=True)
			results[task][condition] = records
		runtime_by_task[task] = adapter.runtime_summary()
		del tokenizer
		del adapter
		torch.cuda.empty_cache()

	if _tree_snapshot(cutie_root) != cutie_before:
		raise RuntimeError("Cutie implementation changed during inference.")
	if _implementation_snapshot() != implementation_before:
		raise RuntimeError("Object-graph backend implementation changed during inference.")
	if _graph_tree_snapshot(args.graph_dir) != graph_tree_before:
		raise RuntimeError("Object graph inputs changed during inference.")
	if file_sha256(args.checkpoint) != checkpoint_sha:
		raise RuntimeError("Cutie checkpoint changed during inference.")
	inputs_after, _, _ = validate_backend_inputs(input_path, strict_counts=args.strict_counts)
	if inputs_after != inputs or file_sha256(input_path) != input_manifest_sha:
		raise RuntimeError("GT-free backend inputs changed during inference.")
	for task in tasks:
		if file_sha256(graph_paths[task]) != graph_manifest[task]["graph_file_sha256"]:
			raise RuntimeError(f"{task}: selected graph changed during inference.")

	payload = {
		"format": OBJECT_GRAPH_BACKEND_FORMAT,
		"status": "complete",
		"backend": BACKEND_NAME,
		"dataset_id": inputs["dataset_id"],
		"input_manifest_sha256": input_manifest_sha,
		"roles": {task: list(inputs["roles"][task]) for task in tasks},
		"graphs": graph_manifest,
		"protocol": {
			"base_backend_format": BACKEND_FORMAT,
			"support_only_prompts": True,
			"support_frames_replayed_each_episode": 6,
			"episode_first_frame_gt_prompt": False,
			"mid_episode_reprompt": False,
			"episode_ground_truth_read": False,
			"causal_frame_order": True,
			"prompt_adapter": "declared_union_to_entity_indexed_support_v1",
			"role_projector": "support_conditioned_object_graph_v1",
			"model_internal_features_exported": True,
			"reported_confidence_cross_backend_comparable": False,
			"strict_source_pixel_access_gate": True,
			"online_deployment_eligible": True,
			"runtime_ms_semantics": {
				"cutie_runtime_ms": "adapter_track_internal_per_current_frame_v1",
				"parser_runtime_ms": "tokenizer_project_internal_per_current_frame_v1",
				"end_to_end_runtime_ms": (
					"episode_end_to_end_amortized_per_frame_including_reset_support_replay_"
					"excluding_npz_io_and_model_construction_v1"
				),
			},
		},
		"backend_provenance": {
			"model_family": "cutie",
			"treatment": "support_conditioned_object_graph_v1",
			"model_size": args.model_size,
			"tracker_size": [args.tracker_size, args.tracker_size],
			"oc_storm_repo": str(args.oc_storm_repo),
			"cutie_implementation": cutie_before,
			"object_graph_implementation": implementation_before,
			"graph_tree": graph_tree_before,
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


def _run_preflight(args) -> dict[str, Any]:
	import torch
	from tdmpc2.perception.cutie_oc_adapter import (
		CutieOCAdapter,
		CutieOCConfig,
		CutieSupportPrompts,
		inspect_cutie_installation,
	)
	from tdmpc2.perception.support_conditioned_object_graph import (
		QUERY_FEATURE_DIM,
		SupportConditionedObjectGraphTokenizer,
		project_support_to_entities,
	)

	_validate_cuda_binding(torch)
	if not args.oc_storm_repo.is_dir():
		raise FileNotFoundError(args.oc_storm_repo)
	if not args.checkpoint.is_file() or args.checkpoint.is_symlink():
		raise FileNotFoundError(args.checkpoint)
	if not args.graph_dir.is_dir() or args.graph_dir.is_symlink():
		raise FileNotFoundError(args.graph_dir)
	input_path = args.inputs.expanduser().resolve()
	input_manifest_sha = file_sha256(input_path)
	inputs, support_paths, episode_paths = validate_backend_inputs(
		input_path, strict_counts=args.strict_counts
	)
	tasks = tuple(inputs["tasks"])
	graph_tree_before = _graph_tree_snapshot(args.graph_dir)
	graphs, graph_paths = _load_graphs(args.graph_dir, tasks, inputs["roles"])
	single_entity_tasks = [task for task in tasks if len(graphs[task].entities) == 1]
	if not single_entity_tasks:
		raise RuntimeError("Preflight requires at least one declared single-entity graph.")
	task = single_entity_tasks[0]
	graph = graphs[task]
	graph_file_sha = file_sha256(graph_paths[task])
	roles = tuple(inputs["roles"][task])
	if (
		graph.entities[0].source_roles != graph.source_roles
		or graph.entities[0].projector.roles != graph.semantic_roles
	):
		raise RuntimeError(
			"Single-entity preflight must union all source roles and project all semantic roles."
		)
	resolution = int(inputs["resolution"])
	support_arrays = _load_npz(support_paths[task], {"rgb", "indexed_masks"})
	support_rgb = support_arrays["rgb"]
	source_masks = support_arrays["indexed_masks"]
	if (
		support_rgb.shape != (6, resolution, resolution, 3)
		or support_rgb.dtype != np.uint8
		or source_masks.shape != (6, resolution, resolution)
	):
		raise RuntimeError("Single-entity preflight support arrays changed shape.")
	entity_support = project_support_to_entities(support_rgb, source_masks, graph)
	if graph.entity_names != (graph.entities[0].name,):
		raise RuntimeError("Single-entity graph name contract changed.")
	support = CutieSupportPrompts(
		frames=tuple(np.array(value, dtype=np.uint8, order="C", copy=True) for value in entity_support.rgb),
		masks=tuple(np.array(value, order="C", copy=True) for value in entity_support.indexed_masks),
		role_names=graph.entity_names,
		annotation_path=support_paths[task],
		metadata={
			**entity_support.metadata,
			"task": task,
			"source_support_arrays_sha256": file_sha256(support_paths[task]),
		},
	)
	condition = inputs["conditions"][0]
	episode_path = episode_paths[(task, condition, 0)]
	episode = _load_npz(episode_path, {"rgb"})["rgb"]
	if (
		episode.shape != (
			int(inputs["counts"]["frames_per_episode"]),
			resolution,
			resolution,
			3,
		)
		or episode.dtype != np.uint8
	):
		raise RuntimeError("Single-entity preflight RGB episode changed shape.")
	cutie_root = (
		args.oc_storm_repo / "feature_extractor" / "cutie" / "cutie"
	).resolve()
	cutie_before = _tree_snapshot(cutie_root)
	checkpoint_sha = file_sha256(args.checkpoint)
	implementation_before = _implementation_snapshot()
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	config = CutieOCConfig(
		repo_path=args.oc_storm_repo,
		checkpoint_path=args.checkpoint,
		role_names=graph.entity_names,
		model_size=args.model_size,
		device="cuda:0",
		output_device="cpu",
		expected_input_size=(resolution, resolution),
		support_input_size=(resolution, resolution),
		mask_output_size=(resolution, resolution),
		tracker_size=(args.tracker_size, args.tracker_size),
		return_object_features=True,
		object_schema="generic_entity_indexed_v1",
		amp=not args.disable_amp,
	)
	report = inspect_cutie_installation(config, import_check=True)
	adapter = CutieOCAdapter(config)
	adapter.add_support_prompts(support)
	tokenizer = SupportConditionedObjectGraphTokenizer(graph, source_masks)
	adapter.reset_episode()
	tokenizer.reset_episode()
	result = adapter.track(episode[0])
	if result.object_features is None:
		raise RuntimeError("Single-entity Cutie preflight did not export query features.")
	entity_masks = result.masks.numpy()
	entity_features = result.object_features.detach().cpu().numpy()
	entity_lost = result.lost.numpy()
	entity_confidence = result.confidence.numpy()
	entity_mask_score = result.mask_score.numpy()
	if (
		result.role_names != graph.entity_names
		or entity_masks.shape != (1, resolution, resolution)
		or entity_masks.dtype != np.bool_
		or entity_features.shape != (1, QUERY_FEATURE_DIM)
		or entity_features.dtype != np.float32
		or entity_lost.shape != (1,)
		or entity_lost.dtype != np.bool_
		or entity_confidence.shape != (1,)
		or entity_confidence.dtype != np.float32
		or entity_mask_score.shape != (1,)
		or entity_mask_score.dtype != np.float32
		or not np.isfinite(entity_features).all()
		or not np.isfinite(entity_confidence).all()
		or not np.isfinite(entity_mask_score).all()
		or np.any((entity_confidence < 0.0) | (entity_confidence > 1.0))
		or np.any((entity_mask_score < 0.0) | (entity_mask_score > 1.0))
		or not np.isfinite(result.runtime_ms)
		or result.runtime_ms < 0.0
	):
		raise RuntimeError("Official Cutie K=1 output was squeezed or malformed.")
	token = tokenizer.project(
		entity_masks=entity_masks,
		entity_features=entity_features,
		entity_lost=entity_lost,
		entity_confidence=entity_confidence,
		entity_mask_score=entity_mask_score,
	)
	if (
		token.role_names != roles
		or token.masks.shape != (2, resolution, resolution)
		or token.masks.dtype != np.bool_
		or token.descriptors.shape != (2, 590)
		or token.descriptors.dtype != np.float32
		or token.keypoints_xy.shape != (2, 2, 2)
		or token.keypoints_xy.dtype != np.float32
		or token.valid.shape != (2,)
		or token.valid.dtype != np.bool_
		or token.confidence.shape != (2,)
		or token.confidence.dtype != np.float32
		or token.lost.shape != (2,)
		or token.lost.dtype != np.bool_
		or token.mask_score.shape != (2,)
		or token.mask_score.dtype != np.float32
		or not np.isfinite(token.descriptors).all()
		or not np.isfinite(token.keypoints_xy).all()
		or not np.isfinite(token.confidence).all()
		or not np.isfinite(token.mask_score).all()
		or np.any((token.confidence < 0.0) | (token.confidence > 1.0))
		or np.any((token.mask_score < 0.0) | (token.mask_score > 1.0))
		or np.any(token.valid & token.lost)
		or np.any(token.masks.astype(np.uint8).sum(axis=0) > 1)
		or not np.isfinite(token.runtime_ms)
		or token.runtime_ms < 0.0
	):
		raise RuntimeError("Single-entity two-role tokenizer output contract failed.")
	del tokenizer
	del adapter
	torch.cuda.empty_cache()
	if _tree_snapshot(cutie_root) != cutie_before:
		raise RuntimeError("Cutie implementation changed during preflight.")
	if _implementation_snapshot() != implementation_before:
		raise RuntimeError("Object-graph implementation changed during preflight.")
	if _graph_tree_snapshot(args.graph_dir) != graph_tree_before:
		raise RuntimeError("Object graph inputs changed during preflight.")
	if file_sha256(args.checkpoint) != checkpoint_sha:
		raise RuntimeError("Cutie checkpoint changed during preflight.")
	inputs_after, _, _ = validate_backend_inputs(input_path, strict_counts=args.strict_counts)
	if inputs_after != inputs or file_sha256(input_path) != input_manifest_sha:
		raise RuntimeError("GT-free backend inputs changed during preflight.")
	if file_sha256(graph_paths[task]) != graph_file_sha:
		raise RuntimeError("Selected single-entity graph changed during preflight.")
	return {
		"status": "complete",
		"dataset_id": inputs["dataset_id"],
		"input_manifest_sha256": input_manifest_sha,
		"task": task,
		"condition": condition,
		"episode_index": 0,
		"frame_index": 0,
		"entity_count": 1,
		"role_count": 2,
		"entity_names": list(graph.entity_names),
		"role_names": list(token.role_names),
		"entity_mask_shape": list(entity_masks.shape),
		"entity_feature_shape": list(entity_features.shape),
		"role_mask_shape": list(token.masks.shape),
		"descriptor_shape": list(token.descriptors.shape),
		"graph_sha256": graph.graph_sha256,
		"graph_file_sha256": graph_file_sha,
		"object_schema": "generic_entity_indexed_v1",
		"return_object_features": True,
		"checkpoint_sha256": checkpoint_sha,
		"cutie_tree_sha256": cutie_before["tree_sha256"],
		"inspection": report,
	}


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--inputs", type=Path)
	parser.add_argument("--output-root", type=Path)
	parser.add_argument("--graph-dir", type=Path)
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
	for name in ("inputs", "output_root", "graph_dir", "oc_storm_repo", "checkpoint"):
		value = getattr(args, name)
		if value is not None:
			setattr(args, name, value.expanduser().resolve())
	if args.tracker_size < 1:
		raise ValueError("tracker-size must be positive.")
	if args.seed < 0:
		raise ValueError("seed must be non-negative.")
	if args.preflight_only:
		if args.inputs is None or args.graph_dir is None:
			raise ValueError("preflight requires --inputs and --graph-dir")
		payload = _run_preflight(args)
		print("OBJECT_GRAPH_CUTIE_PREFLIGHT_OK", json.dumps(payload, allow_nan=False))
		return
	if args.inputs is None or args.output_root is None or args.graph_dir is None:
		raise ValueError(
			"benchmark execution requires --inputs, --output-root, and --graph-dir"
		)
	payload = run(args)
	print("OBJECT_GRAPH_CUTIE_COMPLETE", json.dumps({
		"dataset_id": payload["dataset_id"],
		"manifest": str(args.output_root / "backend_predictions.json"),
	}, allow_nan=False), flush=True)


if __name__ == "__main__":
	main()
