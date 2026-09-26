"""Dependency-light contracts for the frozen unified VOS benchmark."""

from __future__ import annotations

import copy
import hashlib
import tempfile
from pathlib import Path

import numpy as np

from tdmpc2.common.unified_vos import (
	BACKEND_INPUT_FORMAT,
	CONDITIONS,
	DATASET_FORMAT,
	SUPPORT_FORMAT,
	TASK_ROLES,
	compute_dataset_id,
	file_sha256,
	validate_backend_inputs,
	validate_dataset_files,
	write_json,
)
from tdmpc2.common.unified_vos_snapshot import (
	ALL_FILES_SELECTION,
	CODE_SELECTION,
	FORMAT as EXTERNAL_INPUTS_FORMAT,
	_tree as snapshot_tree,
)
from tdmpc2.tools.aggregate_unified_vos_benchmark import (
	MaskScorer,
	_validate_decoded_dataset_artifacts,
	_validate_external_snapshot,
)
from tdmpc2.tools import collect_unified_vos_dataset as dataset_collector


def _trace(value: np.ndarray) -> str:
	array = np.ascontiguousarray(value)
	digest = hashlib.sha256()
	digest.update(str(array.dtype).encode("ascii"))
	digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
	digest.update(array.tobytes())
	return digest.hexdigest()


def _sequence_trace(label: bytes, value: np.ndarray) -> str:
	digest = hashlib.sha256()
	for item in value:
		array = np.ascontiguousarray(item)
		digest.update(label)
		digest.update(str(array.dtype).encode("ascii"))
		digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
		digest.update(array.tobytes())
	return digest.hexdigest()


def _npz(path: Path, **arrays: np.ndarray) -> str:
	path.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(path, **arrays)
	return file_sha256(path)


def _fixture(root: Path) -> tuple[Path, Path]:
	resolution, episodes, actions, frames = 8, 1, 1, 2
	support = {}
	episode_map = {}
	for task, roles in TASK_ROLES.items():
		rgb_support = np.zeros((6, resolution, resolution, 3), dtype=np.uint8)
		mask_support = np.zeros((6, resolution, resolution), dtype=np.uint8)
		mask_support[:, 1:3, 1:3] = 1
		mask_support[:, 5:7, 5:7] = 2
		relative_support = Path("support") / task / "support.npz"
		support_sha = _npz(
			root / relative_support, rgb=rgb_support, indexed_masks=mask_support
		)
		support[task] = {
			"format": SUPPORT_FORMAT,
			"roles": list(roles),
			"records": 6,
			"resolution": resolution,
			"arrays": relative_support.as_posix(),
			"arrays_sha256": support_sha,
			"asset_trace_sha256": _trace(rgb_support)[:32] + _trace(mask_support)[:32],
			"source": f"/immutable/{task}/annotations.json",
			"source_format": "test_paired_support_v1",
			"source_sha256": hashlib.sha256(task.encode()).hexdigest(),
			"source_metadata": {},
		}
		episode_map[task] = {}
		for condition_index, condition in enumerate(CONDITIONS):
			rgb = np.full(
				(frames, resolution, resolution, 3), condition_index, dtype=np.uint8
			)
			gt = np.zeros((frames, resolution, resolution), dtype=np.uint8)
			gt[:, 1:3, 1:3] = 1
			gt[:, 5:7, 5:7] = 2
			action_values = np.zeros((actions, 1), dtype=np.float32)
			physics_states = np.zeros((frames, 2), dtype=np.float64)
			rgb_relative = Path("inputs") / task / condition / "episode_000.npz"
			gt_relative = Path("scoring") / task / condition / "episode_000.npz"
			rgb_sha = _npz(root / rgb_relative, rgb=rgb)
			gt_sha = _npz(
				root / gt_relative,
				gt_indexed=gt,
				actions=action_values,
				physics_states=physics_states,
			)
			episode_map[task][condition] = [{
				"episode_index": 0,
				"frames": frames,
				"actions": actions,
				"arrays": gt_relative.as_posix(),
				"arrays_sha256": gt_sha,
				"rgb_arrays": rgb_relative.as_posix(),
				"rgb_arrays_sha256": rgb_sha,
				"rgb_trace_sha256": _sequence_trace(b"rgb", rgb),
				"gt_trace_sha256": _sequence_trace(b"gt", gt),
				"action_trace_sha256": _sequence_trace(b"action", action_values),
				"physics_trace_sha256": _sequence_trace(b"state", physics_states),
				"background_trace_sha256": (
					hashlib.sha256(f"{task}/background".encode()).hexdigest()
					if condition == "hard" else None
				),
				"random_policy_reward": 0.0,
				"role_gt_visible_frames": {role: frames for role in roles},
			}]

	protocol = {
		"tracker_only": True,
		"controller_constructed": False,
		"controller_training_steps": 0,
		"episode_gt_disclosed_in_backend_inputs": False,
		"episode_gt_use": "offline_scoring_only",
		"prompt_source": "fixed_six_frame_support_pack_only",
		"episode_first_frame_gt_prompt": False,
		"mid_episode_reprompt": False,
		"causal_frame_order": True,
		"same_frames_all_backends": True,
		"allowed_claim": "video_segmentation_frontend_selection",
		"disallowed_claim": "controller_or_representation_advantage",
	}
	dataset = {
		"format": DATASET_FORMAT,
		"status": "complete",
		"tasks": list(TASK_ROLES),
		"conditions": list(CONDITIONS),
		"roles": {task: list(roles) for task, roles in TASK_ROLES.items()},
		"resolution": resolution,
		"counts": {
			"episodes": episodes,
			"actions_per_episode": actions,
			"frames_per_episode": frames,
		},
		"protocol": protocol,
		"seeds": {"environment": 1, "background": 2, "action": 3, "support": 4},
		"render": {},
		"support": support,
		"episodes": episode_map,
	}
	dataset["dataset_id"] = compute_dataset_id(dataset)
	backend_inputs = {
		"format": BACKEND_INPUT_FORMAT,
		"status": "complete",
		"dataset_id": dataset["dataset_id"],
		"tasks": list(TASK_ROLES),
		"conditions": list(CONDITIONS),
		"roles": dataset["roles"],
		"resolution": resolution,
		"counts": dataset["counts"],
		"protocol": {
			"support_only_prompts": True,
			"episode_ground_truth_present": False,
			"episode_first_frame_gt_prompt": False,
			"mid_episode_reprompt": False,
			"causal_frame_order": True,
		},
		"support": {
			task: {
				"roles": entry["roles"],
				"records": 6,
				"resolution": resolution,
				"arrays": entry["arrays"],
				"arrays_sha256": entry["arrays_sha256"],
				"asset_trace_sha256": entry["asset_trace_sha256"],
			}
			for task, entry in support.items()
		},
		"episodes": {
			task: {
				condition: [{
					"episode_index": 0,
					"frames": frames,
					"rgb_arrays": records[0]["rgb_arrays"],
					"rgb_arrays_sha256": records[0]["rgb_arrays_sha256"],
					"rgb_trace_sha256": records[0]["rgb_trace_sha256"],
				}]
				for condition, records in conditions.items()
			}
			for task, conditions in episode_map.items()
		},
	}
	inputs_path = root / "backend_inputs.json"
	write_json(inputs_path, backend_inputs)
	dataset["backend_inputs"] = {
		"path": inputs_path.name,
		"sha256": file_sha256(inputs_path),
		"gt_paths_disclosed": False,
	}
	dataset_path = root / "dataset_manifest.json"
	write_json(dataset_path, dataset)
	return dataset_path, inputs_path


def _expect_failure(function) -> None:
	try:
		function()
	except (ValueError, FileNotFoundError):
		return
	raise AssertionError("Expected a fail-closed validation error.")


def _numpy_action_transport_contract() -> None:
	"""Reproduce the collector wrapper chain's NumPy-only step boundary."""
	source = Path(dataset_collector.__file__).read_text(encoding="utf-8")
	assert "torch.from_numpy" not in source
	assert source.count("_step_environment(env, action)") == 2

	class ActionSpace:
		shape = (2,)
		dtype = np.dtype(np.float32)

	class NumpyOnlyDMControl:
		def __init__(self):
			self.received: list[np.ndarray] = []

		def step(self, action):
			# This is the operation that failed when the collector sent a Tensor.
			converted = action.astype(np.float32)
			assert isinstance(action, np.ndarray)
			assert action.flags.c_contiguous
			self.received.append(converted.copy())
			# A wrapper is allowed to reuse/mutate its private transport copy.
			action.fill(np.float32(99.0))
			return None, float(converted.sum()), False, {}

	class PassthroughWrapper:
		action_space = ActionSpace()

		def __init__(self, env):
			self.env = env

		def step(self, action):
			return self.env.step(action)

	actions = np.asarray(((-0.5, 0.25), (0.75, -0.125)), dtype=np.float32)
	expected = actions.copy()
	bottom = NumpyOnlyDMControl()
	env = PassthroughWrapper(PassthroughWrapper(bottom))
	digest = hashlib.sha256()
	for action in actions:
		dataset_collector._digest_array(digest, b"action", action)
		_, _, done, _ = dataset_collector._step_environment(env, action)
		assert not done

	# Transport copies cannot mutate the arrays saved in the scoring archive,
	# and their action trace must remain exactly the pre-transport trace used by
	# both clean and hard records of a paired trajectory.
	assert np.array_equal(actions, expected)
	received = np.stack(bottom.received)
	assert np.array_equal(received, expected)
	assert digest.hexdigest() == _sequence_trace(b"action", expected)
	assert _sequence_trace(b"action", received) == _sequence_trace(b"action", expected)


def _scoring_permission_gate_contract() -> None:
	"""Keep the GT lock atomic and exactly reversible in the shell runner."""
	runner = Path(__file__).resolve().parent / "tools" / "run_unified_vos_benchmark.sh"
	source = runner.read_text(encoding="utf-8")
	assert "chmod -R a-rwx" not in source
	assert "chmod -R u+rwX" not in source
	lock = 'chmod 000 -- "$DATASET_ROOT/scoring"'
	restore = 'chmod "$SCORING_ROOT_MODE_BEFORE" -- "$DATASET_ROOT/scoring"'
	assert source.count(lock) == 1
	assert source.count(restore) == 2  # normal completion and EXIT cleanup
	assert source.index('SCORING_ROOT_MODE_BEFORE="$(file_mode') < source.index(lock)
	assert source.index("SCORING_LOCKED=1") < source.index(lock)
	assert source.index(lock) < source.index("PROTOCOL_SMOKE_START")
	assert source.index("BACKEND_START backend=cutie") < source.rindex(restore)
	assert source.index("BACKEND_START backend=sam21") < source.rindex(restore)
	assert source.index("BACKEND_START backend=sam31") < source.rindex(restore)
	assert '[[ "$SCORING_ROOT_MODE_LOCKED" == 000 && ! -r "$scoring_probe" ]]' in source
	assert "LOCKED_READ_PROBE_RC == 1" in source
	assert "grep -Fq 'PermissionError'" in source


def _external_snapshot_fixture(root: Path, tree: dict) -> dict:
	all_files_tree = copy.deepcopy(tree)
	all_files_tree["selection"] = ALL_FILES_SELECTION
	code_tree = copy.deepcopy(tree)
	code_tree["selection"] = CODE_SELECTION
	required = root / "required.bin"
	required.write_bytes(b"required immutable input")
	required_record = {
		"path": str(required.resolve()),
		"bytes": required.stat().st_size,
		"sha256": file_sha256(required),
	}
	return {
		"format": EXTERNAL_INPUTS_FORMAT,
		"trees": {
			"local_python": copy.deepcopy(code_tree),
			"video_hard": copy.deepcopy(all_files_tree),
			"background_manifests": copy.deepcopy(all_files_tree),
			"cutie_source": copy.deepcopy(code_tree),
			"sam21_source": copy.deepcopy(code_tree),
			"sam31_source": copy.deepcopy(code_tree),
		},
		"files": {
			name: copy.deepcopy(required_record)
			for name in (
				"local_config", "runner", "cutie_checkpoint", "sam21_checkpoint",
				"sam31_checkpoint", "sam31_bpe",
			)
		},
	}


def _snapshot_symlink_contract(root: Path) -> None:
	fixture = root / "snapshot_symlink"
	config = fixture / "sam2" / "configs" / "sam2"
	config.mkdir(parents=True)
	target = config / "sam2_hiera_l.yaml"
	target.write_bytes(b"model:\n  image_size: 1024\n")
	alias = fixture / "sam2" / "sam2_hiera_l.yaml"
	link_target = "configs/sam2/sam2_hiera_l.yaml"
	try:
		alias.symlink_to(link_target)
	except (NotImplementedError, OSError) as exc:
		# Windows without Developer Mode cannot create test symlinks.  The same
		# contract runs unskipped in the Linux benchmark environment.
		print(f"UNIFIED_VOS_SYMLINK_CONTRACT_SKIPPED {type(exc).__name__}", flush=True)
		return

	tree = snapshot_tree(fixture, code_only=True)
	rows = {row["path"]: row for row in tree["files"]}
	alias_row = rows["sam2/sam2_hiera_l.yaml"]
	target_row = rows["sam2/configs/sam2/sam2_hiera_l.yaml"]
	assert tree["selection"] == CODE_SELECTION
	assert alias_row == {
		"path": "sam2/sam2_hiera_l.yaml",
		"kind": "symlink_file",
		"link_target": link_target,
		"target_path": "sam2/configs/sam2/sam2_hiera_l.yaml",
		"bytes": target_row["bytes"],
		"sha256": target_row["sha256"],
	}
	assert target_row["kind"] == "regular_file"

	payload = _external_snapshot_fixture(root, tree)
	_validate_external_snapshot(payload)
	tampered = copy.deepcopy(payload)
	symlink_row = next(
		row for row in tampered["trees"]["sam21_source"]["files"]
		if row["kind"] == "symlink_file"
	)
	symlink_row["sha256"] = "0" * 64
	_expect_failure(lambda: _validate_external_snapshot(tampered))

	# The raw link text is part of the identity even if it resolves to the same
	# regular file.
	alias.unlink()
	alias.symlink_to("configs/sam2/./sam2_hiera_l.yaml")
	rewritten_link = snapshot_tree(fixture, code_only=True)
	assert rewritten_link != tree
	assert next(
		row for row in rewritten_link["files"] if row["kind"] == "symlink_file"
	)["link_target"] == "configs/sam2/./sam2_hiera_l.yaml"

	# Dereferenced target bytes are independently bound by the symlink row.
	target.write_bytes(b"model:\n  image_size: 2048\n")
	changed_target = snapshot_tree(fixture, code_only=True)
	assert changed_target != rewritten_link

	def rejection_case(name: str, target_text: str, *, second_link: str | None = None):
		case = root / name
		case.mkdir()
		(case / "selected.yaml").write_text("selected: true\n", encoding="utf-8")
		link = case / "alias.yaml"
		link.symlink_to(target_text)
		if second_link is not None:
			(case / target_text).symlink_to(second_link)
		_expect_failure(lambda: snapshot_tree(case, code_only=True))

	outside = root / "outside.yaml"
	outside.write_text("outside: true\n", encoding="utf-8")
	rejection_case("snapshot_escape", "../outside.yaml")
	rejection_case("snapshot_absolute", str(outside.resolve()))
	rejection_case("snapshot_dangling", "missing.yaml")
	rejection_case("snapshot_chain", "middle.yaml", second_link="selected.yaml")

	cycle = root / "snapshot_cycle"
	cycle.mkdir()
	(cycle / "selected.yaml").write_text("selected: true\n", encoding="utf-8")
	(cycle / "first.yaml").symlink_to("second.yaml")
	(cycle / "second.yaml").symlink_to("first.yaml")
	_expect_failure(lambda: snapshot_tree(cycle, code_only=True))

	directory = root / "snapshot_directory_link"
	directory.mkdir()
	(directory / "selected.yaml").write_text("selected: true\n", encoding="utf-8")
	(directory / "real_directory").mkdir()
	(directory / "linked_directory").symlink_to("real_directory", target_is_directory=True)
	_expect_failure(lambda: snapshot_tree(directory, code_only=True))


def main() -> None:
	_numpy_action_transport_contract()
	_scoring_permission_gate_contract()
	with tempfile.TemporaryDirectory(prefix="unified_vos_contract_") as temporary:
		root = Path(temporary)
		_snapshot_symlink_contract(root)
		dataset_path, inputs_path = _fixture(root)
		dataset = validate_dataset_files(dataset_path, strict_counts=False)
		_validate_decoded_dataset_artifacts(dataset, root)
		inputs, support_paths, episode_paths = validate_backend_inputs(
			inputs_path, strict_counts=False
		)
		assert dataset["dataset_id"] == inputs["dataset_id"]
		assert len(support_paths) == 3 and len(episode_paths) == 6

		malicious = copy.deepcopy(inputs)
		record = malicious["episodes"]["acrobot-swingup"]["clean"][0]
		record["rgb_arrays"] = "scoring/acrobot-swingup/clean/episode_000.npz"
		record["rgb_arrays_sha256"] = file_sha256(root / record["rgb_arrays"])
		malicious_path = root / "malicious_scoring_alias.json"
		write_json(malicious_path, malicious)
		_expect_failure(lambda: validate_backend_inputs(malicious_path, strict_counts=False))

		unexpected = copy.deepcopy(inputs)
		unexpected["episodes"]["acrobot-swingup"]["clean"][0]["gt_path"] = "forbidden"
		unexpected_path = root / "unexpected_gt_field.json"
		write_json(unexpected_path, unexpected)
		_expect_failure(lambda: validate_backend_inputs(unexpected_path, strict_counts=False))

		gt = np.zeros((8, 8), dtype=np.uint8)
		gt[1:3, 1:3] = 1
		gt[5:7, 5:7] = 2
		perfect = np.stack((gt == 1, gt == 2))
		scorer = MaskScorer(("one", "two"), 8)
		scorer.begin_episode()
		scorer.record(predicted=perfect, gt_indexed=gt, runtime_ms=1.0)
		summary = scorer.summary()
		for role in ("one", "two"):
			assert summary["per_role"][role]["mean_iou_on_gt_visible_frames"] == 1.0
			assert summary["per_role"][role]["identity_accuracy_on_gt_visible_frames"] == 1.0

		swapped = perfect[::-1]
		scorer = MaskScorer(("one", "two"), 8)
		scorer.begin_episode()
		scorer.record(predicted=swapped, gt_indexed=gt, runtime_ms=1.0)
		summary = scorer.summary()
		assert summary["role_swap_frames"] == 1
		assert all(
			value["identity_accuracy_on_gt_visible_frames"] == 0.0
			for value in summary["per_role"].values()
		)

	print("UNIFIED_VOS_CONTRACT_OK", flush=True)


if __name__ == "__main__":
	main()
