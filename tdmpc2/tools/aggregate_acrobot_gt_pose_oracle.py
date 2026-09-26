"""Bind and aggregate the fixed seed-6 Acrobot GT-pose oracle experiment.

This module is deliberately dependency-light.  ``bind`` validates both frozen
source runs and records hashes before any oracle work is allowed to start.
``verify`` rechecks that binding without rewriting it.  ``aggregate`` performs
the same immutable-input check, validates the completed oracle run, and writes
the three-curve descriptive summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORMAT = "acrobot_gt_pose_oracle_clean100k_v1"
INPUT_FORMAT = "acrobot_gt_pose_oracle_inputs_v1"
TASK = "acrobot-swingup"
SEED = 6
STEPS = 100_000
EVAL_FREQ = 20_000
EVAL_EPISODES = 10
EXPECTED_STEPS = [0, 20_000, 40_000, 60_000, 80_000, 100_000]
OBJECT_EXP = "cutie_object_only_acrobot_clean100k_seed6_20260831_154217"
STATE_EXP = "tdmpc2_state_acrobot_clean100k_seed6_20260831_154217"
ORACLE_EXP = "acrobot_gt_pose_oracle_clean100k_v1_seed6"
ROLE_NAMES = ["upper_arm", "lower_arm"]
EXPECTED_POSE_OBSERVATION_CONTRACT = {
	"format": "gt_articulated_pose_observation_contract_v1",
	"variant": "gt_articulated_pose",
	"privileged_simulator_kinematics": True,
	"privileged_simulator_kinematics_runtime": True,
	"task": TASK,
	"role_names": ROLE_NAMES,
	"frame_schema": "acrobot_gt_articulated_pose_v1",
	"frame_fields": [
		"start_x", "start_z", "end_x", "end_z",
		"sin_theta", "cos_theta", "absolute_frame_omega",
	],
	"source": (
		"dm_control.physics.named.data.xpos[upper_arm,lower_arm]+"
		"named.data.site_xpos[tip]"
	),
	"diagnostic_only": True,
	"position_normalization_total_link_length": 2.0,
	"angular_velocity": "signed_wrapped_absolute_frame_delta_over_sim_time",
	"num_roles": 2,
	"frame_dim": 7,
	"stack_frames": 3,
	"input_dim": 21,
}
EXPECTED_POSE_AUXILIARY_CONTRACT = {
	"format": "gt_articulated_pose_auxiliary_contract_v1",
	"target": "articulated_pose",
	"loss": "smooth_l1",
	"reduction": "mean",
	"beta": 0.1,
	"finite_values_required": True,
	"applies_to": ["current_reconstruction", "future_prediction"],
	"decoder_output_dim": 21,
}
EXPECTED_OBJECT_OBSERVATION_CONTRACT = {
	"format": "cutie_object_observation_contract_v1",
	"variant": "gt_articulated_pose",
	"frame_schema": "acrobot_gt_articulated_pose_v1",
	"privileged_runtime_segmentation": False,
	"num_roles": 2,
	"frame_dim": 7,
	"stack_frames": 3,
	"input_dim": 21,
}
EXPECTED_CUTIE_FULL_OBSERVATION_CONTRACT = {
	"format": "cutie_object_observation_contract_v1",
	"variant": "full",
	"frame_schema": "cutie_query_mask_status_v1",
	"privileged_runtime_segmentation": False,
	"num_roles": 2,
	"frame_dim": 590,
	"stack_frames": 3,
	"input_dim": 1770,
}
EXPECTED_CUTIE_FULL_AUXILIARY_CONTRACT = {
	"format": "cutie_object_auxiliary_contract_v1",
	"target": "full_descriptor",
	"effective_target": "full_descriptor",
	"normalization": "full_descriptor",
	"applies_to": ["current_reconstruction", "future_prediction"],
	"decoder_output_dim": 1770,
	"query_values_per_frame": 512,
	"geometry_status_values_per_frame": 78,
	"supervised_values_per_frame": 590,
	"loss_denominator_values_per_role": 1770,
}

SOURCE_ARTIFACTS = {
	"runtime_config": Path("runtime_config.json"),
	"eval_csv": Path("eval.csv"),
	"checkpoint": Path("models/final.pt"),
	"trainer_runtime": Path("trainer_runtime.json"),
}
IMPLEMENTATION_RULES = (
	"tdmpc2/**/*.py",
	"tdmpc2/config.yaml",
	"tdmpc2/tools/run_acrobot_gt_pose_oracle_clean100k_seed6.sh",
)


class ContractError(RuntimeError):
	"""A frozen experiment contract was not satisfied."""


def _require(condition: bool, message: str) -> None:
	if not condition:
		raise ContractError(message)


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as file:
		for block in iter(lambda: file.read(1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def _artifact(path: Path, *, allow_empty: bool = False) -> dict[str, Any]:
	_require(path.is_file(), f"Missing artifact: {path}")
	size = path.stat().st_size
	_require(allow_empty or size > 0, f"Empty artifact: {path}")
	return {
		"path": str(path.resolve()),
		"bytes": size,
		"sha256": _sha256(path),
	}


def _is_standard_tensordict_metadata(name: Any, value: Any, torch: Any) -> bool:
	"""Recognize the non-tensor bookkeeping emitted by TensorDict.state_dict.

	TensorDict serializes its batch size and device alongside actual parameter
	tensors. They are not model weights, but rejecting them would reject every
	otherwise valid TensorDict-backed checkpoint. Keep this allowlist narrow so
	that arbitrary non-tensor checkpoint contents remain a contract failure.
	"""
	if not isinstance(name, str) or "." not in name:
		return False
	prefix, field = name.rsplit(".", 1)
	if prefix not in {
		"_Qs.params",
		"_detach_Qs_params",
		"_target_Qs_params",
	}:
		return False
	if field == "__batch_size":
		if isinstance(value, torch.Size):
			return True
		return (
			isinstance(value, (tuple, list))
			and all(type(dimension) is int and dimension >= 0 for dimension in value)
		)
	if field == "__device":
		if value is None or isinstance(value, torch.device):
			return True
		if isinstance(value, str):
			try:
				torch.device(value)
				return True
			except (TypeError, RuntimeError):
				return False
	return False


def _read_json(path: Path) -> dict[str, Any]:
	try:
		payload = json.loads(path.read_text(encoding="utf-8"))
	except Exception as exc:
		raise ContractError(f"Cannot read JSON {path}: {exc}") from exc
	_require(isinstance(payload, dict), f"JSON object required: {path}")
	return payload


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(path.name + ".incomplete")
	if path.exists() or temporary.exists():
		raise FileExistsError(f"Refusing to overwrite: {path}")
	try:
		with temporary.open("x", encoding="utf-8", newline="\n") as file:
			json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
			file.write("\n")
		os.replace(temporary, path)
	finally:
		if temporary.exists():
			temporary.unlink()


def _exact_int(payload: dict[str, Any], key: str, expected: int, label: str) -> None:
	value = payload.get(key)
	_require(
		type(value) is int and value == expected,
		f"{label}.{key} must be integer {expected}, got {value!r}",
	)


def _exact_bool(
	payload: dict[str, Any], key: str, expected: bool, label: str
) -> None:
	value = payload.get(key)
	_require(
		value is expected,
		f"{label}.{key} must be {expected!r}, got {value!r}",
	)


def _finite_number(value: Any, label: str) -> float:
	try:
		result = float(value)
	except (TypeError, ValueError) as exc:
		raise ContractError(f"{label} is not numeric: {value!r}") from exc
	_require(math.isfinite(result), f"{label} is non-finite: {value!r}")
	return result


def _curve(path: Path) -> list[dict[str, float | int]]:
	try:
		with path.open("r", encoding="utf-8", newline="") as file:
			reader = csv.DictReader(file)
			fieldnames = reader.fieldnames
			rows = list(reader)
	except Exception as exc:
		raise ContractError(f"Cannot read evaluation curve {path}: {exc}") from exc
	_require(fieldnames == ["step", "episode_reward"], (
		f"{path} header must be exactly step,episode_reward, got {fieldnames!r}"
	))
	_require(len(rows) == len(EXPECTED_STEPS), (
		f"{path} must contain exactly {len(EXPECTED_STEPS)} rows, got {len(rows)}"
	))
	curve = []
	for index, (row, expected_step) in enumerate(zip(rows, EXPECTED_STEPS)):
		_require("step" in row and "episode_reward" in row, (
			f"{path} row {index} lacks step/episode_reward"
		))
		step = _finite_number(row["step"], f"{path}:row{index}.step")
		_require(step == float(expected_step), (
			f"{path} row {index} step must be {expected_step}, got {step}"
		))
		reward = _finite_number(
			row["episode_reward"], f"{path}:row{index}.episode_reward"
		)
		curve.append({"step": expected_step, "episode_reward": reward})
	return curve


def _validate_common_config(
	cfg: dict[str, Any], *, exp_name: str, label: str
) -> None:
	_require(cfg.get("task") == TASK, f"{label}.task mismatch")
	_require(cfg.get("exp_name") == exp_name, f"{label}.exp_name mismatch")
	_exact_int(cfg, "seed", SEED, label)
	_exact_int(cfg, "steps", STEPS, label)
	_exact_int(cfg, "eval_freq", EVAL_FREQ, label)
	_exact_int(cfg, "eval_episodes", EVAL_EPISODES, label)
	_exact_bool(cfg, "video_background_enabled", False, label)
	_exact_bool(cfg, "compile", True, label)
	_exact_bool(cfg, "compile_fallback_random", True, label)
	_exact_bool(cfg, "save_csv", True, label)
	_exact_bool(cfg, "save_agent", True, label)


def _validate_trainer_runtime(payload: dict[str, Any], label: str) -> None:
	_exact_int(payload, "seed", SEED, label)
	_exact_int(payload, "steps", STEPS, label)
	for key in (
		"elapsed_seconds",
		"eval_seconds",
		"non_eval_seconds",
		"training_non_eval_steps_per_second",
	):
		value = _finite_number(payload.get(key), f"{label}.{key}")
		_require(value >= 0.0, f"{label}.{key} must be non-negative")


def _load_finite_checkpoint(path: Path, label: str):
	try:
		import torch
		payload = torch.load(path, map_location="cpu", weights_only=False)
	except Exception as exc:
		raise ContractError(f"Cannot load {label} checkpoint {path}: {exc}") from exc
	_require(isinstance(payload, dict), f"{label} checkpoint payload must be a mapping")
	model = payload.get("model")
	_require(isinstance(model, dict) and model, f"{label} checkpoint model is missing")
	non_tensors = [
		name for name, value in model.items()
		if not torch.is_tensor(value)
		and not _is_standard_tensordict_metadata(name, value, torch)
	]
	_require(not non_tensors, (
		f"{label} checkpoint state contains non-standard non-tensors: "
		f"{non_tensors[:10]!r}"
	))
	bad_tensors = [
		name for name, value in model.items()
		if torch.is_tensor(value) and not torch.isfinite(value).all()
	]
	_require(not bad_tensors, (
		f"{label} checkpoint contains non-finite tensors: {bad_tensors[:10]!r}"
	))
	contract = payload.get("checkpoint_contract")
	_require(isinstance(contract, dict), f"{label} checkpoint contract is missing")
	_require(contract.get("format") == "tdmpc2_checkpoint_contract_v1", (
		f"{label} checkpoint contract format mismatch"
	))
	return model, contract


def _validate_source_checkpoint(path: Path, *, kind: str) -> dict[str, Any]:
	model, contract = _load_finite_checkpoint(path, kind)
	keys = tuple(model)
	if kind == "object_only":
		_require(contract.get("flat_anchor_mode") == "cutie_object_only", (
			"ObjectOnly source checkpoint mode mismatch"
		))
		_require(contract.get("latent_dim") == 128, (
			"ObjectOnly source checkpoint latent width mismatch"
		))
		_require(contract.get("cutie_object_belief_enabled") is False, (
			"ObjectOnly source checkpoint unexpectedly enabled belief"
		))
		_require(contract.get("cutie_object_belief_aux_updates") == 0, (
			"ObjectOnly source checkpoint unexpectedly contains belief updates"
		))
		_require(contract.get("cutie_object_observation") == (
			EXPECTED_CUTIE_FULL_OBSERVATION_CONTRACT
		), "ObjectOnly source checkpoint observation contract mismatch")
		_require(contract.get("cutie_object_auxiliary") == (
			EXPECTED_CUTIE_FULL_AUXILIARY_CONTRACT
		), "ObjectOnly source checkpoint auxiliary contract mismatch")
		_require("gt_articulated_pose_observation" not in contract, (
			"ObjectOnly source checkpoint leaked GT-pose metadata"
		))
		_require(any(name.startswith("_encoder.object.") for name in keys), (
			"ObjectOnly source checkpoint lacks the object encoder"
		))
		_require(any(name.startswith("_object_decoder.") for name in keys), (
			"ObjectOnly source checkpoint lacks the object decoder"
		))
		forbidden = [
			name for name in keys
			if name.startswith("_encoder.rgb.") or name.startswith("_hybrid_")
		]
		_require(not forbidden, (
			f"ObjectOnly source checkpoint leaked RGB/hybrid state: {forbidden[:10]!r}"
		))
	else:
		_require(contract.get("flat_anchor_mode") != "cutie_object_only", (
			"State source checkpoint claims ObjectOnly mode"
		))
		for key in (
			"cutie_object_observation",
			"cutie_object_auxiliary",
			"gt_articulated_pose_observation",
		):
			_require(key not in contract, (
				f"State source checkpoint contains forbidden {key} contract"
			))
		forbidden = [
			name for name in keys
			if name.startswith("_encoder.object.")
			or name.startswith("_object_decoder.")
			or name.startswith("_hybrid_")
		]
		_require(not forbidden, (
			f"State source checkpoint contains ObjectOnly/hybrid state: {forbidden[:10]!r}"
		))
	return json.loads(json.dumps(contract, allow_nan=False))


def _validate_source(root: Path, *, kind: str) -> dict[str, Any]:
	exp_name = OBJECT_EXP if kind == "object_only" else STATE_EXP
	_require(root.is_dir(), f"Missing {kind} source root: {root}")
	_require(root.name == exp_name, (
		f"{kind} source root basename must be {exp_name!r}, got {root.name!r}"
	))
	paths = {name: root / relative for name, relative in SOURCE_ARTIFACTS.items()}
	artifacts = {name: _artifact(path) for name, path in paths.items()}
	cfg = _read_json(paths["runtime_config"])
	_validate_common_config(cfg, exp_name=exp_name, label=f"{kind}.config")
	trainer = _read_json(paths["trainer_runtime"])
	_validate_trainer_runtime(trainer, f"{kind}.trainer_runtime")
	curve = _curve(paths["eval_csv"])
	checkpoint_contract = _validate_source_checkpoint(paths["checkpoint"], kind=kind)
	_exact_int(cfg, "model_size", 5, f"{kind}.config")

	if kind == "state":
		_require(cfg.get("obs") == "state", "state source must use obs=state")
		_require(cfg.get("flat_anchor") is False, (
			"state source must have flat_anchor=false"
		))
		_exact_int(cfg, "latent_dim", 512, "state.config")
	else:
		_require(cfg.get("flat_anchor") is True, (
			"ObjectOnly source must have flat_anchor=true"
		))
		_require(cfg.get("flat_anchor_mode") == "cutie_object_only", (
			"ObjectOnly source mode mismatch"
		))
		_require(cfg.get("obs") == "rgb", "clean ObjectOnly source must use obs=rgb")
		_require(cfg.get("cutie_object_observation_variant") == "full", (
			"clean ObjectOnly source must use the full observation variant"
		))
		_require(cfg.get("cutie_object_frame_schema") == (
			"cutie_query_mask_status_v1"
		), "clean ObjectOnly frame schema mismatch")
		_require(cfg.get("cutie_object_role_names") == ROLE_NAMES, (
			"clean ObjectOnly role order mismatch"
		))
		for key, expected in (
			("cutie_object_num_roles", 2),
			("cutie_object_frame_dim", 590),
			("cutie_object_stack_frames", 3),
			("cutie_object_input_dim", 1770),
			("cutie_object_role_dim", 64),
			("cutie_object_hidden_dim", 256),
			("cutie_object_only_latent_dim", 128),
			("latent_dim", 128),
		):
			_exact_int(cfg, key, expected, "object_only.config")
		_require(cfg.get("cutie_object_auxiliary_target") == "full_descriptor", (
			"clean ObjectOnly source auxiliary target must be full_descriptor"
		))
		_require(_finite_number(
			cfg.get("flat_anchor_loss_beta"),
			"object_only.config.flat_anchor_loss_beta",
		) == 0.1, "clean ObjectOnly flat_anchor_loss_beta must be 0.1")
		for key in (
			"cutie_object_allow_simulator_runtime",
			"cutie_object_native_highres_enabled",
			"cutie_object_last_valid_memory",
			"cutie_object_belief_enabled",
			"cutie_object_belief_use_for_control",
		):
			_exact_bool(cfg, key, False, "object_only.config")
		_require(cfg.get("cutie_object_policy_burst_plan") is None, (
			"clean ObjectOnly source must not use a burst plan"
		))
		_require(cfg.get("visual_foreground_erosion_pixels") == 0, (
			"clean ObjectOnly source must not erode the foreground"
		))
		# The key is new relative to the frozen source. Missing and false both
		# mean that privileged simulator kinematics were not enabled.
		_require(cfg.get(
			"cutie_object_allow_simulator_kinematics_runtime", False
		) is False, "clean ObjectOnly source leaked simulator kinematics")

	return {
		"kind": kind,
		"exp_name": exp_name,
		"root": str(root.resolve()),
		"artifacts": artifacts,
		"curve": curve,
		"checkpoint_contract": checkpoint_contract,
		"config_contract": {
			"seed": SEED,
			"steps": STEPS,
			"eval_freq": EVAL_FREQ,
			"eval_episodes": EVAL_EPISODES,
			"background_enabled": False,
			"compile": True,
			"compile_fallback_random": True,
		},
	}


def _implementation_paths(repo: Path) -> tuple[str, ...]:
	repo = repo.resolve()
	package = repo / "tdmpc2"
	_require(package.is_dir(), f"TD-MPC2 package root is missing: {package}")
	paths = {
		path.relative_to(repo).as_posix()
		for path in package.rglob("*.py")
		if path.is_file()
	}
	paths.update({
		"tdmpc2/config.yaml",
		"tdmpc2/tools/run_acrobot_gt_pose_oracle_clean100k_seed6.sh",
	})
	result = tuple(sorted(paths))
	_require(result, "Implementation inventory is empty")
	return result


def _bind_implementation(repo: Path) -> dict[str, dict[str, Any]]:
	_require(repo.is_dir(), f"Repository root is missing: {repo}")
	implementation = {}
	for relative in _implementation_paths(repo):
		path = repo / relative
		# A Python package marker may intentionally be zero bytes.  It still
		# belongs in the immutable source inventory and is protected by its
		# exact size/hash record.
		implementation[relative] = _artifact(
			path, allow_empty=Path(relative).name == "__init__.py"
		)
	return implementation


def _implementation_tree_sha256(
	implementation: dict[str, dict[str, Any]],
) -> str:
	rows = [
		{
			"relative_path": relative,
			"bytes": record["bytes"],
			"sha256": record["sha256"],
		}
		for relative, record in implementation.items()
	]
	encoded = json.dumps(
		rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
	).encode("utf-8")
	return hashlib.sha256(encoded).hexdigest()


def _bind_dm_control_acrobot() -> dict[str, dict[str, Any]]:
	"""Resolve and hash the server-installed Acrobot Python/XML definition."""
	try:
		from dm_control.suite import acrobot
	except Exception as exc:
		raise ContractError(f"Cannot import server dm_control.suite.acrobot: {exc}") from exc
	source_value = inspect.getsourcefile(acrobot)
	_require(isinstance(source_value, str) and source_value, (
		"inspect.getsourcefile(dm_control.suite.acrobot) returned no path"
	))
	source = Path(source_value).resolve()
	xml_candidates = (
		source.with_suffix(".xml"),
		source.parent / "acrobot.xml",
	)
	xml_files = []
	for candidate in xml_candidates:
		resolved = candidate.resolve()
		if resolved.is_file() and resolved not in xml_files:
			xml_files.append(resolved)
	_require(len(xml_files) == 1, (
		"Expected exactly one Acrobot XML adjacent to the installed module, got "
		f"{[str(path) for path in xml_files]!r}"
	))
	return {
		"dm_control_suite_acrobot_python": {
			**_artifact(source),
			"module": str(acrobot.__name__),
			"located_by": "inspect.getsourcefile",
		},
		"dm_control_suite_acrobot_xml": {
			**_artifact(xml_files[0]),
			"located_relative_to_python_source": True,
		},
	}


def bind(args: argparse.Namespace) -> dict[str, Any]:
	repo = args.repo.resolve()
	implementation = _bind_implementation(repo)
	payload = {
		"format": INPUT_FORMAT,
		"bound_at_utc": datetime.now(timezone.utc).isoformat(),
		"contract": {
			"task": TASK,
			"seed": SEED,
			"steps": STEPS,
			"eval_freq": EVAL_FREQ,
			"eval_episodes": EVAL_EPISODES,
			"expected_curve_steps": EXPECTED_STEPS,
			"oracle_exp_name": ORACLE_EXP,
		},
		"source_process_wait": {
			"argv_token": f"exp_name={OBJECT_EXP}",
			"procfs_exact_argv": True,
			"poll_seconds": 10,
			"timeout_seconds": 7200,
			"started_epoch_seconds": args.wait_started_epoch,
			"ended_epoch_seconds": args.wait_ended_epoch,
			"polls_with_match": args.wait_polls_with_match,
		},
		"sources": {
			"object_only": _validate_source(
				args.source_object_root.resolve(), kind="object_only"
			),
			"state": _validate_source(args.source_state_root.resolve(), kind="state"),
		},
		"implementation": implementation,
		"implementation_inventory": {
			"root": str(repo),
			"rules": list(IMPLEMENTATION_RULES),
			"files": len(implementation),
			"tree_sha256": _implementation_tree_sha256(implementation),
		},
		"server_dm_control_acrobot": _bind_dm_control_acrobot(),
	}
	_require(args.wait_ended_epoch >= args.wait_started_epoch, (
		"Invalid source-process wait timestamps"
	))
	_atomic_write(args.output, payload)
	return payload


def _load_bound_inputs(path: Path, expected_sha256: str) -> dict[str, Any]:
	_require(path.is_file(), f"Missing bound inputs: {path}")
	actual_inputs_sha = _sha256(path)
	_require(actual_inputs_sha == expected_sha256, (
		f"Bound inputs changed: expected {expected_sha256}, got {actual_inputs_sha}"
	))
	payload = _read_json(path)
	_require(payload.get("format") == INPUT_FORMAT, "Bound-input format mismatch")
	return payload


def _verify_artifact_record(record: dict[str, Any], label: str) -> None:
	_require(isinstance(record, dict), f"Invalid artifact record: {label}")
	path_value = record.get("path")
	_require(isinstance(path_value, str) and path_value, f"Missing path: {label}")
	path = Path(path_value)
	_require(path.is_file(), f"Immutable artifact disappeared: {path}")
	_require(path.stat().st_size == record.get("bytes"), (
		f"Immutable artifact size changed: {path}"
	))
	_require(_sha256(path) == record.get("sha256"), (
		f"Immutable artifact hash changed: {path}"
	))


def verify_bound(payload: dict[str, Any]) -> None:
	contract = payload.get("contract")
	_require(contract == {
		"task": TASK,
		"seed": SEED,
		"steps": STEPS,
		"eval_freq": EVAL_FREQ,
		"eval_episodes": EVAL_EPISODES,
		"expected_curve_steps": EXPECTED_STEPS,
		"oracle_exp_name": ORACLE_EXP,
	}, "Bound experiment contract changed")
	wait = payload.get("source_process_wait")
	_require(isinstance(wait, dict), "Source-process wait provenance is missing")
	_require(wait.get("argv_token") == f"exp_name={OBJECT_EXP}", (
		"Source-process exact argv token changed"
	))
	_require(wait.get("procfs_exact_argv") is True, (
		"Source-process wait was not recorded as exact /proc argv"
	))
	_require(wait.get("poll_seconds") == 10, "Source-process poll interval changed")
	_require(wait.get("timeout_seconds") == 7200, "Source-process timeout changed")
	for key in ("started_epoch_seconds", "ended_epoch_seconds", "polls_with_match"):
		_require(type(wait.get(key)) is int and wait[key] >= 0, (
			f"Invalid source-process wait field {key}: {wait.get(key)!r}"
		))
	_require(wait["ended_epoch_seconds"] >= wait["started_epoch_seconds"], (
		"Source-process wait timestamps are reversed"
	))
	sources = payload.get("sources")
	_require(isinstance(sources, dict) and set(sources) == {"object_only", "state"}, (
		"Bound source set mismatch"
	))
	for source_name, expected_exp in (
		("object_only", OBJECT_EXP), ("state", STATE_EXP)
	):
		source = sources[source_name]
		_require(source.get("exp_name") == expected_exp, (
			f"Bound {source_name} experiment changed"
		))
		artifacts = source.get("artifacts")
		_require(isinstance(artifacts, dict) and set(artifacts) == set(
			SOURCE_ARTIFACTS
		), f"Bound {source_name} artifact set mismatch")
		for artifact_name, record in artifacts.items():
			_verify_artifact_record(record, f"{source_name}.{artifact_name}")
		curve_record = source.get("curve")
		curve_path = Path(artifacts["eval_csv"]["path"])
		_require(_curve(curve_path) == curve_record, (
			f"Bound {source_name} curve changed"
		))
	implementation = payload.get("implementation")
	_require(isinstance(implementation, dict), "Bound implementation is missing")
	inventory = payload.get("implementation_inventory")
	_require(isinstance(inventory, dict), "Implementation inventory metadata is missing")
	root_value = inventory.get("root")
	_require(isinstance(root_value, str) and root_value, (
		"Implementation inventory root is missing"
	))
	root = Path(root_value).resolve()
	_require(inventory.get("rules") == list(IMPLEMENTATION_RULES), (
		"Implementation inventory rules changed"
	))
	_require(inventory.get("files") == len(implementation), (
		"Implementation inventory file count changed"
	))
	_require(inventory.get("tree_sha256") == (
		_implementation_tree_sha256(implementation)
	), "Implementation inventory combined hash changed")
	current_paths = _implementation_paths(root)
	_require(tuple(implementation) == current_paths, (
		"Bound implementation path set/order mismatch"
	))
	for relative, record in implementation.items():
		expected_path = (root / relative).resolve()
		_require(Path(record.get("path", "")).resolve() == expected_path, (
			f"Bound implementation path target changed: {relative}"
		))
		_verify_artifact_record(record, f"implementation.{relative}")
	external = payload.get("server_dm_control_acrobot")
	_require(isinstance(external, dict) and set(external) == {
		"dm_control_suite_acrobot_python", "dm_control_suite_acrobot_xml",
	}, "Server dm_control Acrobot provenance set mismatch")
	for name, record in external.items():
		_verify_artifact_record(record, f"server_dm_control_acrobot.{name}")


def verify(args: argparse.Namespace) -> dict[str, Any]:
	payload = _load_bound_inputs(args.inputs, args.expected_inputs_sha256)
	verify_bound(payload)
	return payload


def _validate_oracle_config(cfg: dict[str, Any]) -> None:
	_validate_common_config(cfg, exp_name=ORACLE_EXP, label="oracle.config")
	_require(cfg.get("obs") == "state", "oracle must use obs=state")
	_exact_bool(cfg, "flat_anchor", True, "oracle.config")
	_require(cfg.get("flat_anchor_mode") == "cutie_object_only", (
		"oracle must use flat_anchor_mode=cutie_object_only"
	))
	_require(cfg.get("cutie_object_observation_variant") == "gt_articulated_pose", (
		"oracle observation variant mismatch"
	))
	_require(cfg.get("cutie_object_frame_schema") == (
		"acrobot_gt_articulated_pose_v1"
	), "oracle frame schema mismatch")
	_require(cfg.get("cutie_object_role_names") == ROLE_NAMES, (
		"oracle role order mismatch"
	))
	for key, expected in (
		("cutie_object_num_roles", 2),
		("cutie_object_frame_dim", 7),
		("cutie_object_stack_frames", 3),
		("cutie_object_input_dim", 21),
		("cutie_object_role_dim", 64),
		("cutie_object_hidden_dim", 256),
		("cutie_object_only_latent_dim", 128),
	):
		_exact_int(cfg, key, expected, "oracle.config")
	_exact_bool(
		cfg, "cutie_object_allow_simulator_kinematics_runtime", True,
		"oracle.config",
	)
	_require(cfg.get("cutie_object_auxiliary_target") == "full_descriptor", (
		"oracle auxiliary target must be full_descriptor"
	))
	_require(_finite_number(
		cfg.get("flat_anchor_loss_beta"), "oracle.config.flat_anchor_loss_beta"
	) == 0.1, "oracle flat_anchor_loss_beta must be 0.1")
	for key in (
		"cutie_object_allow_simulator_runtime",
		"cutie_object_allow_simulator_support",
		"cutie_object_native_highres_enabled",
		"cutie_object_last_valid_memory",
		"cutie_object_belief_enabled",
		"cutie_object_belief_use_for_control",
	):
		_exact_bool(cfg, key, False, "oracle.config")
	for key in (
		"cutie_object_repo",
		"cutie_object_checkpoint",
		"cutie_object_support_path",
		"cutie_object_config_dir",
		"cutie_object_policy_burst_plan",
		"video_background_root",
		"video_background_manifest_dir",
	):
		_require(cfg.get(key) is None, f"oracle.config.{key} must be null")
	_require(cfg.get("visual_foreground_erosion_pixels") == 0, (
		"oracle must not use foreground erosion"
	))
	_require(cfg.get("obs_shape") == {"object": [2, 21]}, (
		f"oracle obs_shape mismatch: {cfg.get('obs_shape')!r}"
	))
	_exact_int(cfg, "latent_dim", 128, "oracle.config")


def _validate_oracle_checkpoint(path: Path) -> dict[str, Any]:
	try:
		import torch
		payload = torch.load(path, map_location="cpu", weights_only=False)
	except Exception as exc:
		raise ContractError(f"Cannot load oracle checkpoint {path}: {exc}") from exc
	_require(isinstance(payload, dict), "Oracle checkpoint payload must be a mapping")
	model = payload.get("model")
	_require(isinstance(model, dict) and model, "Oracle checkpoint model is missing")
	bad_tensors = [
		name for name, value in model.items()
		if torch.is_tensor(value) and not torch.isfinite(value).all()
	]
	_require(not bad_tensors, (
		f"Oracle checkpoint contains non-finite tensors: {bad_tensors[:10]!r}"
	))
	contract = payload.get("checkpoint_contract")
	_require(isinstance(contract, dict), "Oracle checkpoint contract is missing")
	_require(contract.get("format") == "tdmpc2_checkpoint_contract_v1", (
		"Oracle checkpoint contract format mismatch"
	))
	_require(contract.get("flat_anchor_mode") == "cutie_object_only", (
		"Oracle checkpoint mode mismatch"
	))
	_require(contract.get("latent_dim") == 128, "Oracle checkpoint latent width mismatch")
	_require(contract.get("cutie_object_belief_enabled") is False, (
		"Oracle checkpoint unexpectedly enabled object belief"
	))
	_require(contract.get("cutie_object_belief_aux_updates") == 0, (
		"Oracle checkpoint unexpectedly contains belief updates"
	))
	_require(contract.get("cutie_object_observation") == (
		EXPECTED_OBJECT_OBSERVATION_CONTRACT
	), "Oracle checkpoint generic object observation contract mismatch")
	_require(contract.get("gt_articulated_pose_observation") == (
		EXPECTED_POSE_OBSERVATION_CONTRACT
	), "Oracle checkpoint GT-pose observation contract mismatch")
	_require(contract.get("cutie_object_auxiliary") == (
		EXPECTED_POSE_AUXILIARY_CONTRACT
	), "Oracle checkpoint pose auxiliary contract mismatch")
	return json.loads(json.dumps(contract, allow_nan=False))


def _validate_oracle(root: Path) -> dict[str, Any]:
	_require(root.is_dir(), f"Missing oracle run root: {root}")
	_require(root.name == ORACLE_EXP, (
		f"Oracle root basename must be {ORACLE_EXP!r}, got {root.name!r}"
	))
	paths = {name: root / relative for name, relative in SOURCE_ARTIFACTS.items()}
	artifacts = {name: _artifact(path) for name, path in paths.items()}
	cfg = _read_json(paths["runtime_config"])
	_validate_oracle_config(cfg)
	trainer = _read_json(paths["trainer_runtime"])
	_validate_trainer_runtime(trainer, "oracle.trainer_runtime")
	curve = _curve(paths["eval_csv"])
	checkpoint_contract = _validate_oracle_checkpoint(paths["checkpoint"])
	return {
		"kind": "gt_articulated_pose_oracle",
		"exp_name": ORACLE_EXP,
		"root": str(root.resolve()),
		"artifacts": artifacts,
		"curve": curve,
		"final_episode_reward": curve[-1]["episode_reward"],
		"peak_episode_reward": max(row["episode_reward"] for row in curve),
		"checkpoint_contract": checkpoint_contract,
	}


def _curve_summary(source: dict[str, Any]) -> dict[str, Any]:
	curve = source["curve"]
	return {
		"exp_name": source["exp_name"],
		"root": source["root"],
		"curve": curve,
		"final_episode_reward": curve[-1]["episode_reward"],
		"peak_episode_reward": max(row["episode_reward"] for row in curve),
		"artifacts": source["artifacts"],
		"checkpoint_contract": source["checkpoint_contract"],
	}


def _differences(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
	_require(len(left) == len(right) == len(EXPECTED_STEPS), "Curve length mismatch")
	result = []
	for expected_step, left_row, right_row in zip(EXPECTED_STEPS, left, right):
		_require(left_row["step"] == right_row["step"] == expected_step, (
			"Curve schedule-alignment step mismatch"
		))
		result.append({
			"step": expected_step,
			"episode_reward_delta": (
				left_row["episode_reward"] - right_row["episode_reward"]
			),
		})
	return result


def aggregate_payload(args: argparse.Namespace) -> dict[str, Any]:
	expected_inputs = (args.output.parent / "provenance" / "inputs.json").resolve()
	_require(args.inputs.resolve() == expected_inputs, (
		"Aggregate inputs must be provenance/inputs.json relative to the summary root: "
		f"{args.inputs.resolve()} != {expected_inputs}"
	))
	inputs = verify(args)
	oracle = _validate_oracle(args.oracle_root.resolve())
	object_only = _curve_summary(inputs["sources"]["object_only"])
	state = _curve_summary(inputs["sources"]["state"])
	oracle_curve = oracle["curve"]
	object_curve = object_only["curve"]
	state_curve = state["curve"]
	state_final = state["final_episode_reward"]
	_require(state_final != 0.0, (
		"State final reward is zero; oracle/state final ratio is undefined"
	))
	ratio = oracle["final_episode_reward"] / state_final
	_require(math.isfinite(ratio), "Oracle/state final ratio is non-finite")
	inputs_sha = _sha256(args.inputs)
	curves = {
		"object_only": object_only,
		"state": state,
		"gt_articulated_pose_oracle": oracle,
	}
	comparisons = {
		"curve_deltas": {
			"oracle_minus_object_only": _differences(oracle_curve, object_curve),
			"oracle_minus_state": _differences(oracle_curve, state_curve),
			"object_only_minus_state": _differences(object_curve, state_curve),
		},
		"final_episode_reward_deltas": {
			"oracle_minus_object_only": (
				oracle["final_episode_reward"] - object_only["final_episode_reward"]
			),
			"oracle_minus_state": (
				oracle["final_episode_reward"] - state["final_episode_reward"]
			),
			"object_only_minus_state": (
				object_only["final_episode_reward"] - state["final_episode_reward"]
			),
		},
		"peak_episode_reward_deltas": {
			"oracle_minus_object_only": (
				oracle["peak_episode_reward"] - object_only["peak_episode_reward"]
			),
			"oracle_minus_state": (
				oracle["peak_episode_reward"] - state["peak_episode_reward"]
			),
			"object_only_minus_state": (
				object_only["peak_episode_reward"] - state["peak_episode_reward"]
			),
		},
		"oracle_over_state_final_episode_reward_ratio": ratio,
	}
	return {
		"format": FORMAT,
		"status": "acrobot_gt_pose_oracle_engineering_pass",
		"engineering": {
			"status": "pass",
			"gates": {
				"source_process_wait_contract": True,
				"source_runs_complete_and_exact": True,
				"source_artifacts_immutable": True,
				"implementation_immutable": True,
				"server_dm_control_acrobot_immutable": True,
				"oracle_run_complete_and_exact": True,
				"three_curves_schedule_aligned": True,
			},
			"failures": [],
		},
		"scientific": {
			"status": "single_seed_descriptive_only",
			"engineering_gate": False,
			"scope": (
				"Three independently trained policies at fixed Acrobot seed 6 with "
				"the same step/evaluation schedule. Reward deltas and ratios are "
				"schedule-aligned descriptive values; they are not episode-paired, "
				"a cross-seed algorithm claim, or an engineering pass/fail gate."
			),
			"oracle_final_above_object_only": (
				oracle["final_episode_reward"] > object_only["final_episode_reward"]
			),
			"oracle_final_above_state": (
				oracle["final_episode_reward"] > state["final_episode_reward"]
			),
		},
		"contract": inputs["contract"],
		"curves": curves,
		"comparisons": comparisons,
		"provenance": {
			"inputs_relative_to_summary_root": "provenance/inputs.json",
			"inputs_execution_time_staging_path": str(args.inputs.resolve()),
			"inputs_sha256": inputs_sha,
			"implementation": inputs["implementation"],
			"implementation_inventory": inputs["implementation_inventory"],
			"server_dm_control_acrobot": inputs["server_dm_control_acrobot"],
			"source_process_wait": inputs["source_process_wait"],
			"aggregated_at_utc": datetime.now(timezone.utc).isoformat(),
		},
	}


def _add_verify_arguments(parser: argparse.ArgumentParser) -> None:
	parser.add_argument("--inputs", type=Path, required=True)
	parser.add_argument("--expected-inputs-sha256", required=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	subparsers = parser.add_subparsers(dest="command", required=True)

	bind_parser = subparsers.add_parser("bind", help="Validate and hash sources")
	bind_parser.add_argument("--repo", type=Path, required=True)
	bind_parser.add_argument("--source-object-root", type=Path, required=True)
	bind_parser.add_argument("--source-state-root", type=Path, required=True)
	bind_parser.add_argument("--wait-started-epoch", type=int, required=True)
	bind_parser.add_argument("--wait-ended-epoch", type=int, required=True)
	bind_parser.add_argument("--wait-polls-with-match", type=int, required=True)
	bind_parser.add_argument("--output", type=Path, required=True)

	verify_parser = subparsers.add_parser("verify", help="Recheck bound hashes")
	_add_verify_arguments(verify_parser)

	aggregate_parser = subparsers.add_parser(
		"aggregate", help="Validate oracle and write the three-curve summary"
	)
	_add_verify_arguments(aggregate_parser)
	aggregate_parser.add_argument("--oracle-root", type=Path, required=True)
	aggregate_parser.add_argument("--output", type=Path, required=True)

	args = parser.parse_args(argv)
	if args.command in {"verify", "aggregate"}:
		_require(len(args.expected_inputs_sha256) == 64 and all(
			character in "0123456789abcdef"
			for character in args.expected_inputs_sha256
		), "--expected-inputs-sha256 must be a lowercase SHA-256 digest")
	return args


def main(argv: list[str] | None = None) -> int:
	args = parse_args(argv)
	try:
		if args.command == "bind":
			payload = bind(args)
			print(json.dumps({
				"status": "sources_bound",
				"sources": sorted(payload["sources"]),
			}, allow_nan=False), flush=True)
		elif args.command == "verify":
			verify(args)
			print("ACROBOT_GT_POSE_IMMUTABLE_INPUTS_OK", flush=True)
		else:
			payload = aggregate_payload(args)
			_atomic_write(args.output, payload)
			print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
		return 0
	except Exception as exc:
		if args.command == "aggregate" and not args.output.exists():
			failure = {
				"format": FORMAT,
				"status": "acrobot_gt_pose_oracle_engineering_fail",
				"engineering": {
					"status": "fail",
					"gates": {},
					"failures": [f"{type(exc).__name__}: {exc}"],
				},
				"scientific": {
					"status": "not_evaluated_engineering_failure",
					"engineering_gate": False,
				},
			}
			try:
				_atomic_write(args.output, failure)
			except Exception as write_exc:
				print(f"Could not write failure summary: {write_exc}", file=sys.stderr)
		print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
		return 4


if __name__ == "__main__":
	raise SystemExit(main())
