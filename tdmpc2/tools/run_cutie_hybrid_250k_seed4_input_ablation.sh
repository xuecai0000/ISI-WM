#!/usr/bin/env bash
# Frozen input-information ablation for the trained seed-4 250k CutieHybrid checkpoint.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${VIDEO_ROOT:?Set VIDEO_ROOT to the existing video_hard directory}"
: "${SUPPORT:?Set SUPPORT to the verified six-frame support annotations.json}"
: "${OC_REPO:?Set OC_REPO to the OC-STORM checkout}"
: "${CUTIE_CKPT:?Set CUTIE_CKPT to cutie-small-mega.pth}"

PY="${PY:-python}"
MANIFEST_DIR="${MANIFEST_DIR:-$REPO_ROOT/tdmpc2/envs/background_manifests}"
GPU_ABLATION="${GPU_ABLATION:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/logs/_heldout/cutie_hybrid_250k_seed4_input_ablation_v1}"
STAGING="${OUTPUT_ROOT}.incomplete"

EPISODES=20
ENV_SEED=424242
BACKGROUND_SEED=1618033
PLANNER_SEED_BASE=8675309
TRAINING_SEED=4
TRAINING_STEPS=250000
TRAINING_EVAL_FREQ=25000
TRAINING_EVAL_EPISODES=5
CUTIE_EXP="cutie_hybrid250k_cutie_hybrid_250k_pair_v1_seed4"
CUTIE_RUN="$REPO_ROOT/logs/reacher-visual-small/4/$CUTIE_EXP"
TRAINING_SUMMARY="$REPO_ROOT/logs/_launch/cutie_hybrid_250k_pair_v1_seed4/paired_summary.json"
MODES=(none rgb_zero object_zero)

if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
	echo "This runner requires Bash >=5.1." >&2
	exit 2
fi
if [[ ! "$GPU_ABLATION" =~ ^(0|[1-9][0-9]*)$ ]]; then
	echo "GPU_ABLATION must be one canonical physical GPU index, got: $GPU_ABLATION" >&2
	exit 2
fi
if [[ "$PY" == */* ]]; then
	[[ -x "$PY" ]] || { echo "Python is not executable: $PY" >&2; exit 2; }
else
	command -v "$PY" >/dev/null || { echo "Python is not on PATH: $PY" >&2; exit 2; }
fi
for path in "$VIDEO_ROOT" "$MANIFEST_DIR" "$SUPPORT" "$OC_REPO" "$CUTIE_CKPT"; do
	[[ -e "$path" ]] || { echo "Required input does not exist: $path" >&2; exit 2; }
done
for path in \
	"$CUTIE_RUN/runtime_config.json" \
	"$CUTIE_RUN/models/final.pt" \
	"$TRAINING_SUMMARY"; do
	[[ -s "$path" ]] || { echo "Missing completed source artifact: $path" >&2; exit 2; }
done
for path in "$OUTPUT_ROOT" "$STAGING"; do
	[[ ! -e "$path" ]] || { echo "Refusing to overwrite: $path" >&2; exit 3; }
done

"$PY" - \
	"$VIDEO_ROOT" "$MANIFEST_DIR" "$SUPPORT" "$OC_REPO" "$CUTIE_CKPT" \
	"$CUTIE_RUN/runtime_config.json" "$CUTIE_RUN" "$CUTIE_EXP" \
	"$TRAINING_SUMMARY" <<'PY'
import json
import math
import sys
from pathlib import Path

video_root, manifest_dir, support, oc_repo, cutie_checkpoint = (
    Path(value).expanduser().resolve() for value in sys.argv[1:6]
)
runtime_path = Path(sys.argv[6]).resolve()
run_root = Path(sys.argv[7]).resolve()
expected_exp = sys.argv[8]
training_summary_path = Path(sys.argv[9]).resolve()

with runtime_path.open(encoding="utf-8") as file:
    runtime = json.load(file)
if not isinstance(runtime, dict):
    raise SystemExit("Cutie runtime config is not a JSON object.")
expected = {
    "task": "reacher-visual-small",
    "obs": "rgb",
    "multitask": False,
    "model_size": 5,
    "steps": 250000,
    "seed": 4,
    "eval_freq": 25000,
    "eval_episodes": 5,
    "episode_length": 500,
    "video_background_enabled": True,
    "video_background_split": "train",
    "flat_anchor": True,
    "flat_anchor_mode": "cutie_hybrid",
    "latent_dim": 640,
    "flat_anchor_scene_dim": 512,
    "exp_name": expected_exp,
}
bad = {key: (runtime.get(key), value) for key, value in expected.items()
       if runtime.get(key) != value}
if bad:
    raise SystemExit(f"Frozen seed-4 Cutie source mismatch: {bad}")
expected_paths = {
    "video_background_root": video_root,
    "video_background_manifest_dir": manifest_dir,
    "cutie_object_support_path": support,
    "cutie_object_repo": oc_repo,
    "cutie_object_checkpoint": cutie_checkpoint,
}
for key, expected_path in expected_paths.items():
    actual = Path(runtime[key]).expanduser().resolve()
    if actual != expected_path:
        raise SystemExit(f"External input mismatch: {key}={actual}, expected {expected_path}")
if runtime_path != run_root / "runtime_config.json":
    raise SystemExit("Runtime config/run-root binding mismatch.")
if not (run_root / "models" / "final.pt").is_file():
    raise SystemExit("Bound final checkpoint is missing.")

with training_summary_path.open(encoding="utf-8") as file:
    source = json.load(file)
if not isinstance(source, dict):
    raise SystemExit("Source paired summary is not a JSON object.")
source_expected = {
    "total_steps": 250000,
    "eval_freq": 25000,
    "eval_episodes": 5,
    "expected_cutie_frames": 278055,
}
bad = {key: (source.get(key), value) for key, value in source_expected.items()
       if source.get(key) != value}
if bad:
    raise SystemExit(f"Source paired summary mismatch: {bad}")
cutie_run = source.get("runs", {}).get("cutie", {})
if Path(cutie_run.get("root", "")).resolve() != run_root:
    raise SystemExit("Source paired summary does not bind the Cutie run root.")
rewards = cutie_run.get("eval_rewards")
if not isinstance(rewards, list) or len(rewards) != 11 or not all(
    math.isfinite(float(value)) for value in rewards
):
    raise SystemExit("Source Cutie training evaluation rewards are invalid.")
print("CUTIE_HYBRID_SEED4_INPUT_ABLATION_SOURCE_OK")
PY

GPU_INFO="$(CUDA_VISIBLE_DEVICES="$GPU_ABLATION" "$PY" -c \
	'import json,torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(json.dumps({"name":torch.cuda.get_device_name(0),"capability":torch.cuda.get_device_capability(0)},sort_keys=True))')"
echo "Ablation physical GPU $GPU_ABLATION: $GPU_INFO"

ACTIVE_PID=""
OWN_STAGING=0
cleanup() {
	local rc=$? failed
	trap - EXIT INT TERM
	if [[ -n "$ACTIVE_PID" ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
		kill "$ACTIVE_PID" 2>/dev/null || true
		wait "$ACTIVE_PID" 2>/dev/null || true
	fi
	if (( rc != 0 && OWN_STAGING == 1 )) && [[ -d "$STAGING" ]]; then
		failed="${OUTPUT_ROOT}.failed.$(date +%Y%m%d_%H%M%S).$$"
		if mv -T -- "$STAGING" "$failed"; then
			echo "Preserved failed ablation at: $failed" >&2
		fi
	fi
	exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p -- "$(dirname -- "$STAGING")"
mkdir -- "$STAGING"
OWN_STAGING=1
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Output appeared while acquiring staging." >&2; exit 3; }

echo "[1/5] Running evaluator, spawn, wrapper, model, and ablation contracts"
"$PY" -c 'import tdmpc2.tools.evaluate_cutie_hybrid_heldout; from tdmpc2.tdmpc2 import TDMPC2; print("HELDOUT_IMPORT_OK", TDMPC2.__name__)'
"$PY" -m tdmpc2.check_cutie_hybrid_heldout_spawn_contract
"$PY" -m tdmpc2.check_cutie_hybrid_heldout_training_launch_contract
"$PY" -m tdmpc2.check_cutie_hybrid_input_ablation_contract
"$PY" tdmpc2/check_cutie_object_wrapper_contract.py
CUDA_VISIBLE_DEVICES="$GPU_ABLATION" "$PY" tdmpc2/check_cutie_hybrid_contract.py
"$PY" tdmpc2/check_cutie_oc_adapter_contract.py

echo "[2/5] Proving history-independent Cutie episode reset on physical GPU $GPU_ABLATION"
CUDA_VISIBLE_DEVICES="$GPU_ABLATION" "$PY" \
	-m tdmpc2.tools.check_cutie_episode_reset_isolation \
	--runtime-config "$CUTIE_RUN/runtime_config.json" \
	--output "$STAGING/reset_isolation.json" \
	--pollution-length 500

echo "[3/5] Running official whole-arm/goal Cutie preflight on physical GPU $GPU_ABLATION"
CUDA_VISIBLE_DEVICES="$GPU_ABLATION" "$PY" -m tdmpc2.tools.check_cutie_oc_preflight \
	--oc-storm-repo "$OC_REPO" \
	--checkpoint "$CUTIE_CKPT" \
	--support-annotations "$SUPPORT" \
	--object-schema whole_arm_goal_v1 \
	--model-size small \
	--prompt-radius 2.0 \
	--tracker-size 448 448 \
	--device cuda:0 \
	--sha256

run_mode() {
	local mode=$1 output="$STAGING/$1.json" log="$STAGING/$1.log" rc
	echo "Starting mode=$mode; log=$log"
	env CUDA_VISIBLE_DEVICES="$GPU_ABLATION" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" -m tdmpc2.tools.evaluate_cutie_hybrid_heldout \
		--runtime-config "$CUTIE_RUN/runtime_config.json" \
		--checkpoint "$CUTIE_RUN/models/final.pt" \
		--backend cutie_hybrid \
		--observation-ablation "$mode" \
		--training-seed "$TRAINING_SEED" \
		--output "$output" \
		--episodes "$EPISODES" \
		--env-seed "$ENV_SEED" \
		--background-seed "$BACKGROUND_SEED" \
		--planner-seed-base "$PLANNER_SEED_BASE" \
		--expected-training-steps "$TRAINING_STEPS" \
		--expected-training-eval-freq "$TRAINING_EVAL_FREQ" \
		--expected-training-eval-episodes "$TRAINING_EVAL_EPISODES" \
		>"$log" 2>&1 &
	ACTIVE_PID=$!
	set +e
	wait "$ACTIVE_PID"
	rc=$?
	set -e
	ACTIVE_PID=""
	return "$rc"
}

echo "[4/5] Running full, RGB-input-zero, and object-input-zero sequentially"
for mode in "${MODES[@]}"; do
	run_mode "$mode"
done

echo "[5/5] Verifying exact pairing and writing ablation_summary.json"
"$PY" - \
	"$STAGING" "$EPISODES" "$ENV_SEED" "$BACKGROUND_SEED" \
	"$PLANNER_SEED_BASE" "$GPU_ABLATION" "$TRAINING_SUMMARY" \
	"$CUTIE_RUN" <<'PY'
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
episodes_expected = int(sys.argv[2])
env_seed = int(sys.argv[3])
background_seed = int(sys.argv[4])
planner_seed_base = int(sys.argv[5])
gpu_selector = sys.argv[6]
training_summary_path = Path(sys.argv[7]).resolve()
expected_run = Path(sys.argv[8]).resolve()
modes = ("none", "rgb_zero", "object_zero")
runs = {}


def load(path):
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise AssertionError((path, "payload is not an object"))
    return value


def require_sha256(value, location):
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise AssertionError((location, value))
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


reset_report_path = root / "reset_isolation.json"
reset_report = load(reset_report_path)
expected_reset = {
    "format": "cutie_episode_reset_isolation_v1",
    "status": "episode_reset_isolation_pass",
    "pass": True,
    "reset_strategy": "fresh_inference_core_support_replay_v1",
    "runtime_config": str((expected_run / "runtime_config.json").resolve()),
    "pollution_length": 500,
    "all_features_finite_and_roles_valid": True,
    "pollution_histories_diverged": True,
}
bad_reset = {
    key: (reset_report.get(key), expected)
    for key, expected in expected_reset.items()
    if reset_report.get(key) != expected
}
if bad_reset:
    raise AssertionError((reset_report_path, "reset isolation", bad_reset))
reset_comparisons = reset_report.get("comparisons")
if not isinstance(reset_comparisons, dict) or len(reset_comparisons) != 8 or not all(
    isinstance(value, dict) and value.get("byte_equal") is True
    for value in reset_comparisons.values()
):
    raise AssertionError((reset_report_path, "reset comparisons", reset_comparisons))
reset_workers = reset_report.get("workers")
if not isinstance(reset_workers, dict) or set(reset_workers) != {"forward", "reverse"}:
    raise AssertionError((reset_report_path, "reset workers", reset_workers))
for name, worker in reset_workers.items():
    if (
        not isinstance(worker, dict)
        or worker.get("tracked_frames") != 514
        or worker.get("ready", {}).get("episode_reset_strategy")
        != "fresh_inference_core_support_replay_v1"
    ):
        raise AssertionError((reset_report_path, name, worker))
reset_asset = reset_report.get("arrays_asset")
if not isinstance(reset_asset, dict) or reset_asset.get("path") != "reset_isolation.npz":
    raise AssertionError((reset_report_path, "reset asset", reset_asset))
reset_asset_path = root / reset_asset["path"]
if not reset_asset_path.is_file() or file_sha256(reset_asset_path) != reset_asset.get("sha256"):
    raise AssertionError((reset_asset_path, "reset asset SHA mismatch"))
reset_report_sha256 = file_sha256(reset_report_path)


def reward_stats(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def paired_stats(left, right, label):
    deltas = [a - b for a, b in zip(left, right)]
    mean = statistics.mean(deltas)
    sample_std = statistics.stdev(deltas)
    standard_error = sample_std / math.sqrt(len(deltas))
    half_width = 2.093024054408263 * standard_error
    wins = sum(value > 0 for value in deltas)
    ties = sum(value == 0 for value in deltas)
    losses = sum(value < 0 for value in deltas)
    return {
        "definition": label,
        "mean": mean,
        "median": statistics.median(deltas),
        "sample_std": sample_std,
        "standard_error": standard_error,
        "min": min(deltas),
        "max": max(deltas),
        "conditional_episode_mean_95pct_t_interval_df19": [
            mean - half_width,
            mean + half_width,
        ],
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / len(deltas),
        "paired_deltas": deltas,
    }


expected_evaluation = {
    "split": "validation",
    "episodes": episodes_expected,
    "environment_seed": env_seed,
    "background_seed": background_seed,
    "planner_seed_base": planner_seed_base,
    "eval_mode": True,
    "compile": False,
    "reset_planner_rng_each_episode": True,
    "reset_previous_plan_each_episode": True,
}
expected_source_identity = {
    "steps": 250000,
    "eval_freq": 25000,
    "eval_episodes": 5,
    "episode_length": 500,
}
expected_frames = episodes_expected * 501

for mode in modes:
    path = root / f"{mode}.json"
    payload = load(path)
    if payload.get("format") != "cutie_hybrid_heldout_evaluation_v1":
        raise AssertionError((path, payload.get("format")))
    if payload.get("task") != "reacher-visual-small":
        raise AssertionError((path, payload.get("task")))
    if payload.get("backend") != "cutie_hybrid" or payload.get("training_seed") != 4:
        raise AssertionError((path, payload.get("backend"), payload.get("training_seed")))
    if payload.get("observation_ablation") != mode:
        raise AssertionError((path, "observation ablation", payload.get("observation_ablation")))
    if payload.get("evaluation") != expected_evaluation:
        raise AssertionError((path, "evaluation protocol", payload.get("evaluation")))
    if payload.get("expected_source_training_launch") != expected_source_identity:
        raise AssertionError((path, "source launch"))
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise AssertionError((path, "missing provenance"))
    if Path(provenance.get("runtime_config", "")).resolve() != expected_run / "runtime_config.json":
        raise AssertionError((path, "runtime config binding"))
    if Path(provenance.get("checkpoint", "")).resolve() != expected_run / "models" / "final.pt":
        raise AssertionError((path, "checkpoint binding"))
    if provenance.get("cuda_visible_devices") != gpu_selector:
        raise AssertionError((path, "physical GPU selector"))
    if provenance.get("logical_cuda_device") != 0:
        raise AssertionError((path, "logical CUDA device"))
    for key in (
        "runtime_config_sha256", "checkpoint_sha256", "evaluator_sha256",
        "validation_manifest_sha256", "combined_manifest_sha256",
    ):
        require_sha256(provenance.get(key), f"{path}:{key}")
    rows = payload.get("episodes")
    if not isinstance(rows, list) or len(rows) != episodes_expected:
        raise AssertionError((path, "episode count"))
    for index, row in enumerate(rows):
        if row.get("episode_index") != index or row.get("planner_seed") != planner_seed_base + index:
            raise AssertionError((path, index, "episode/planner identity"))
        if row.get("length") != 500:
            raise AssertionError((path, index, "episode length"))
        if not math.isfinite(float(row.get("reward", math.nan))):
            raise AssertionError((path, index, "reward"))
        if not math.isfinite(float(row.get("success", math.nan))):
            raise AssertionError((path, index, "success"))
        for key in (
            "planner_rng_start_sha256", "planner_rng_end_sha256",
            "initial_rgb_sha256", "initial_object_sha256",
            "initial_observation_sha256",
        ):
            require_sha256(row.get(key), f"{path}:episode{index}:{key}")
        if not isinstance(row.get("background_source"), str) or not row["background_source"]:
            raise AssertionError((path, index, "background source"))
        if not isinstance(row.get("background_start_frame_index"), int):
            raise AssertionError((path, index, "background frame"))
    runtime = payload.get("perception_runtime")
    if not isinstance(runtime, dict):
        raise AssertionError((path, "missing perception runtime"))
    runtime_gates = {
        "frames": int(runtime.get("frames", -1)) == expected_frames,
        "valid_frame_rate": float(runtime.get("valid_frame_rate", -1.0)) >= 0.95,
        "max_invalid_burst": int(runtime.get("max_invalid_burst", 10**9)) <= 5,
        "worker_restarts": int(runtime.get("worker_restarts", -1)) == 0,
        "timeouts": int(runtime.get("timeouts", -1)) == 0,
        "ms_per_frame": float(runtime.get("ms_per_frame", math.inf)) <= 800.0,
        "runtime_unit": runtime.get("runtime_unit")
        == "milliseconds_per_tracked_frame_excluding_support_prompts",
    }
    diagnostics = payload.get("observation_ablation_diagnostics")
    expected_zero_checks = 0 if mode == "none" else episodes_expected * 500
    expected_diagnostics = {
        "mode": mode,
        "zeroed_field": {"none": None, "rgb_zero": "rgb", "object_zero": "object"}[mode],
        "preserved_field": {"none": None, "rgb_zero": "object", "object_zero": "rgb"}[mode],
        "decision_steps": episodes_expected * 500,
        "zero_checks": expected_zero_checks,
        "preserved_field_checks": expected_zero_checks,
        "zeroed_nonzero_count_max": 0,
        "zeroed_max_abs": 0.0,
        "location": "copied_agent_input_before_encoding_every_decision_step",
        "live_environment_observation_mutated": False,
    }
    if diagnostics != expected_diagnostics:
        raise AssertionError((path, "ablation diagnostics", diagnostics, expected_diagnostics))
    ready = provenance.get("cutie_ready")
    expected_ready = {
        "status": "ready",
        "start_method": "spawn",
        "global_hydra_initialized_before_cutie": False,
        "roles": ["whole_arm", "goal"],
        "frame_feature_dim": 590,
        "stacked_feature_dim": 1770,
        "permanent_prompts": 6,
        "episode_reset_strategy": "fresh_inference_core_support_replay_v1",
        "device": "cuda:0",
        "tracker_size": [448, 448],
        "model_size": "small",
        "prompt_radius": 2.0,
        "amp": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "current_device": 0,
    }
    if not isinstance(ready, dict):
        raise AssertionError((path, "missing Cutie ready report"))
    bad_ready = {
        key: (ready.get(key), expected)
        for key, expected in expected_ready.items()
        if ready.get(key) != expected
    }
    if bad_ready:
        raise AssertionError((path, "Cutie ready mismatch", bad_ready))
    runs[mode] = {
        "payload": payload,
        "runtime_gates": runtime_gates,
        "ablation_diagnostics": diagnostics,
        "rewards": [float(row["reward"]) for row in rows],
        "successes": [float(row["success"]) for row in rows],
    }

reference = runs["none"]["payload"]
reference_provenance = reference["provenance"]
current_artifact_hashes = {
    "runtime_config_sha256": file_sha256(expected_run / "runtime_config.json"),
    "checkpoint_sha256": file_sha256(expected_run / "models" / "final.pt"),
}
for key, current_sha256 in current_artifact_hashes.items():
    if reference_provenance.get(key) != current_sha256:
        raise AssertionError(
            (key, "source artifact changed during ablation", reference_provenance.get(key), current_sha256)
        )
for mode in modes[1:]:
    payload = runs[mode]["payload"]
    provenance = payload["provenance"]
    for key in (
        "source_training_protocol", "cutie_training_protocol",
        "expected_source_training_launch",
    ):
        if payload.get(key) != reference.get(key):
            raise AssertionError((mode, key, "protocol mismatch"))
    for key in (
        "runtime_config_sha256", "checkpoint_sha256", "evaluator_sha256",
        "validation_manifest_sha256", "combined_manifest_sha256", "cutie_inputs",
    ):
        if provenance.get(key) != reference_provenance.get(key):
            raise AssertionError((mode, key, "provenance mismatch"))
    if (
        provenance.get("device_name"), tuple(provenance.get("device_capability", ()))
    ) != (
        reference_provenance.get("device_name"),
        tuple(reference_provenance.get("device_capability", ())),
    ):
        raise AssertionError((mode, "GPU identity mismatch"))
    for index, (reference_row, row) in enumerate(
        zip(reference["episodes"], payload["episodes"])
    ):
        condition_keys = (
            "episode_index", "planner_seed", "planner_rng_start_sha256",
            "planner_rng_end_sha256", "initial_rgb_sha256",
            "initial_object_sha256", "background_source",
            "initial_observation_sha256",
            "background_start_frame_index", "length",
        )
        if tuple(reference_row[key] for key in condition_keys) != tuple(
            row[key] for key in condition_keys
        ):
            raise AssertionError((mode, index, "exact pairing mismatch"))

source_training_bytes = training_summary_path.read_bytes()
source_summary_sha256 = hashlib.sha256(source_training_bytes).hexdigest()
source_training = json.loads(source_training_bytes)
if not isinstance(source_training, dict):
    raise AssertionError("Source training summary is not a JSON object.")
expected_source_identity = {
    "total_steps": 250000,
    "eval_freq": 25000,
    "eval_episodes": 5,
    "expected_cutie_frames": 278055,
}
bad_source_identity = {
    key: (source_training.get(key), expected)
    for key, expected in expected_source_identity.items()
    if source_training.get(key) != expected
}
if bad_source_identity:
    raise AssertionError(("source training identity", bad_source_identity))
source_runs = source_training.get("runs")
source_cutie_run = source_runs.get("cutie") if isinstance(source_runs, dict) else None
if (
    not isinstance(source_cutie_run, dict)
    or Path(source_cutie_run.get("root", "")).resolve() != expected_run
):
    raise AssertionError(("source training Cutie run root", source_cutie_run, expected_run))
source_rewards = source_cutie_run.get("eval_rewards")
if (
    not isinstance(source_rewards, list)
    or len(source_rewards) != 11
    or not all(math.isfinite(float(value)) for value in source_rewards)
):
    raise AssertionError(("source training Cutie rewards", source_rewards))
source_status = source_training.get("status")
source_gates = source_training.get("gates")
expected_source_gate_keys = {
    "frames", "valid_frame_rate", "max_invalid_burst", "worker_restarts",
    "timeouts", "ms_per_frame", "runtime_unit", "wallclock_seconds",
}
if (
    not isinstance(source_status, str)
    or not isinstance(source_gates, dict)
    or set(source_gates) != expected_source_gate_keys
    or not all(isinstance(value, bool) for value in source_gates.values())
):
    raise AssertionError("Source training gates are invalid.")
source_elapsed = float(source_training.get("elapsed_seconds", math.nan))
source_runtime = source_training.get("cutie_runtime")
if not math.isfinite(source_elapsed) or source_elapsed < 0 or not isinstance(source_runtime, dict):
    raise AssertionError("Source training elapsed/runtime is invalid.")
recomputed_source_gates = {
    "frames": int(source_runtime.get("frames", -1)) == 278055,
    "valid_frame_rate": float(source_runtime.get("valid_frame_rate", -1.0)) >= 0.95,
    "max_invalid_burst": int(source_runtime.get("max_invalid_burst", 10**9)) <= 5,
    "worker_restarts": int(source_runtime.get("worker_restarts", -1)) == 0,
    "timeouts": int(source_runtime.get("timeouts", -1)) == 0,
    "ms_per_frame": float(source_runtime.get("ms_per_frame", math.inf)) <= 800.0,
    "runtime_unit": source_runtime.get("runtime_unit")
    == "milliseconds_per_tracked_frame_excluding_support_prompts",
    "wallclock_seconds": source_elapsed <= 7800.0,
}
if source_gates != recomputed_source_gates:
    raise AssertionError(("source runtime gates do not recompute", source_gates, recomputed_source_gates))
source_runtime_pass = all(source_gates.values())
if source_status.endswith("_pass") != source_runtime_pass:
    raise AssertionError("Source training status/gates are inconsistent.")

conditions = {}
for mode in modes:
    payload = runs[mode]["payload"]
    rewards = runs[mode]["rewards"]
    conditions[mode] = {
        "definition": {
            "none": "unaltered RGB and object inputs",
            "rgb_zero": "RGB input zeroed after live Cutie tracking and before agent encoding",
            "object_zero": "object input zeroed after live Cutie tracking and before agent encoding",
        }[mode],
        "reward": reward_stats(rewards),
        "rewards": rewards,
        "success_rate": statistics.mean(runs[mode]["successes"]),
        "elapsed_seconds": float(payload["summary"]["elapsed_seconds"]),
        "perception_runtime": payload["perception_runtime"],
        "observation_ablation_diagnostics": runs[mode]["ablation_diagnostics"],
        "runtime_gates": runs[mode]["runtime_gates"],
        "runtime_pass": all(runs[mode]["runtime_gates"].values()),
    }

full = runs["none"]["rewards"]
rgb_zero = runs["rgb_zero"]["rewards"]
object_zero = runs["object_zero"]["rewards"]
pairwise = {
    "full_minus_rgb_input_zero": paired_stats(
        full, rgb_zero,
        "positive values measure information lost when RGB is hidden while objects remain",
    ),
    "full_minus_object_input_zero": paired_stats(
        full, object_zero,
        "positive values measure information lost when objects are hidden while RGB remains",
    ),
    "rgb_input_zero_minus_object_input_zero": paired_stats(
        rgb_zero, object_zero,
        "positive values favor object information over RGB information in this frozen checkpoint",
    ),
}
all_ablation_runtime_pass = all(
    conditions[mode]["runtime_pass"] for mode in modes
)
all_runtime_pass = source_runtime_pass and all_ablation_runtime_pass
if all_runtime_pass:
    overall_status = "input_ablation_all_runtime_pass"
elif not source_runtime_pass and not all_ablation_runtime_pass:
    overall_status = "source_training_and_input_ablation_runtime_fail"
elif not source_runtime_pass:
    overall_status = "input_ablation_complete_source_training_runtime_fail"
else:
    overall_status = "input_ablation_runtime_fail"
summary = {
    "format": "cutie_hybrid_250k_seed4_input_ablation_summary_v1",
    "status": overall_status,
    "scientific_scope": (
        "inference-time input-information ablation conditional on the trained seed-4 "
        "250k CutieHybrid checkpoint and twenty paired validation episodes; zero inputs "
        "do not remove encoder/dynamics modules, do not measure a structural fast path, "
        "do not predict object-only retraining performance, and the test split is untouched"
    ),
    "source_training": {
        "paired_summary": str(training_summary_path),
        "paired_summary_sha256": source_summary_sha256,
        "status": source_status,
        "gates": source_gates,
        "elapsed_seconds": source_elapsed,
        "cutie_runtime": source_runtime,
        "runtime_pass": source_runtime_pass,
    },
    "episode_reset_isolation": {
        "report": str(reset_report_path),
        "report_sha256": reset_report_sha256,
        "arrays_asset": str(reset_asset_path),
        "arrays_asset_sha256": reset_asset["sha256"],
        "strategy": reset_report["reset_strategy"],
        "pollution_histories_diverged": True,
        "all_bitwise_comparisons_pass": True,
    },
    "checkpoint_provenance": {
        "runtime_config": reference_provenance["runtime_config"],
        "runtime_config_sha256": reference_provenance["runtime_config_sha256"],
        "checkpoint": reference_provenance["checkpoint"],
        "checkpoint_sha256": reference_provenance["checkpoint_sha256"],
    },
    "heldout_protocol": {
        "split": "validation",
        "test_split_accessed": False,
        "training_seed": 4,
        "source_training_steps": 250000,
        "episodes": episodes_expected,
        "environment_seed": env_seed,
        "background_seed": background_seed,
        "planner_seed_base": planner_seed_base,
        "same_checkpoint": True,
        "same_physical_gpu_sequential": True,
        "physical_gpu_selector": gpu_selector,
        "exact_initial_rgb_and_object_observations": True,
        "planner_rng_boundary_hashes_match": True,
        "live_cutie_runs_in_all_conditions": True,
        "ablation_applied_to_agent_input_every_decision_step": True,
        "episode_reset_isolation_probe_pass": True,
        "evaluator_sha256": reference_provenance["evaluator_sha256"],
        "validation_manifest_sha256": reference_provenance["validation_manifest_sha256"],
        "combined_manifest_sha256": reference_provenance["combined_manifest_sha256"],
        "device_name": reference_provenance["device_name"],
        "device_capability": reference_provenance["device_capability"],
    },
    "conditions": conditions,
    "pairwise_paired_episode_statistics": pairwise,
    "pairing_integrity_pass": True,
    "source_training_runtime_pass": source_runtime_pass,
    "input_ablation_runtime_pass": all_ablation_runtime_pass,
    "all_ablation_runtime_gates_pass": all_ablation_runtime_pass,
    "all_runtime_gates_pass": all_runtime_pass,
    "eligible_for_structural_object_only_speed_claim": False,
    "eligible_for_cross_training_seed_algorithm_claim": False,
}
with (root / "ablation_summary.json").open(
    "x", encoding="utf-8", newline="\n"
) as file:
    json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    file.write("\n")
print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
PY

ALL_RUNTIME_PASS="$("$PY" -c \
	'import json,sys; print(str(bool(json.load(open(sys.argv[1], encoding="utf-8"))["all_runtime_gates_pass"])).lower())' \
	"$STAGING/ablation_summary.json")"
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Output appeared before publication." >&2; exit 3; }
mv -T -- "$STAGING" "$OUTPUT_ROOT"
OWN_STAGING=0
trap - EXIT INT TERM
if [[ "$ALL_RUNTIME_PASS" != "true" ]]; then
	echo "CUTIE_HYBRID_INPUT_ABLATION_OR_SOURCE_RUNTIME_GATE_FAILED" >&2
	"$PY" -c \
		'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); print("STATUS="+p["status"]); print("SOURCE_TRAINING_RUNTIME_PASS="+str(p["source_training_runtime_pass"])); print("INPUT_ABLATION_RUNTIME_PASS="+str(p["input_ablation_runtime_pass"]))' \
		"$OUTPUT_ROOT/ablation_summary.json" >&2
	echo "OUTPUT_ROOT=$OUTPUT_ROOT" >&2
	exit 4
fi
echo "CUTIE_HYBRID_250K_SEED4_INPUT_ABLATION_COMPLETE"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "SUMMARY=$OUTPUT_ROOT/ablation_summary.json"
