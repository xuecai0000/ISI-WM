#!/usr/bin/env bash
# Strict single-training-seed held-out validation for the 250k seed-4 pair.

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
GPU_HELDOUT="${GPU_HELDOUT:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/logs/_heldout/cutie_hybrid_250k_seed4_validation_v1}"
STAGING="${OUTPUT_ROOT}.incomplete"

EPISODES=20
ENV_SEED=424242
BACKGROUND_SEED=1618033
PLANNER_SEED_BASE=8675309
TRAINING_SEED=4
TRAINING_STEPS=250000
TRAINING_EVAL_FREQ=25000
TRAINING_EVAL_EPISODES=5
TRAINING_MAX_WALLCLOCK_SECONDS=7800
RGB_EXP="rgb250k_cutie_hybrid_250k_pair_v1_seed4"
CUTIE_EXP="cutie_hybrid250k_cutie_hybrid_250k_pair_v1_seed4"
RGB_RUN="$REPO_ROOT/logs/reacher-visual-small/4/$RGB_EXP"
CUTIE_RUN="$REPO_ROOT/logs/reacher-visual-small/4/$CUTIE_EXP"
TRAINING_SUMMARY="$REPO_ROOT/logs/_launch/cutie_hybrid_250k_pair_v1_seed4/paired_summary.json"

if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
	echo "This runner requires Bash >=5.1." >&2
	exit 2
fi
if [[ ! "$GPU_HELDOUT" =~ ^(0|[1-9][0-9]*)$ ]]; then
	echo "GPU_HELDOUT must be one canonical physical GPU index, got: $GPU_HELDOUT" >&2
	exit 2
fi
for value_name in EPISODES ENV_SEED BACKGROUND_SEED PLANNER_SEED_BASE; do
	value="${!value_name}"
	if [[ ! "$value" =~ ^[0-9]+$ ]]; then
		echo "$value_name must be a non-negative integer, got: $value" >&2
		exit 2
	fi
done
if (( EPISODES != 20 )); then
	echo "This frozen single-seed held-out protocol requires EPISODES=20." >&2
	exit 2
fi
if (( ENV_SEED == BACKGROUND_SEED || ENV_SEED == PLANNER_SEED_BASE || BACKGROUND_SEED == PLANNER_SEED_BASE )); then
	echo "Environment, background, and planner seed domains must differ." >&2
	exit 2
fi
if (( PLANNER_SEED_BASE + EPISODES - 1 >= 4294967296 )); then
	echo "Planner episode seeds must remain within uint32." >&2
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
for path in "$OUTPUT_ROOT" "$STAGING"; do
	[[ ! -e "$path" ]] || { echo "Refusing to overwrite: $path" >&2; exit 3; }
done
for path in \
	"$RGB_RUN/runtime_config.json" "$RGB_RUN/models/final.pt" \
	"$CUTIE_RUN/runtime_config.json" "$CUTIE_RUN/models/final.pt" \
	"$TRAINING_SUMMARY"; do
	[[ -s "$path" ]] || { echo "Missing completed training artifact: $path" >&2; exit 2; }
done

"$PY" - \
	"$VIDEO_ROOT" "$MANIFEST_DIR" "$SUPPORT" "$OC_REPO" "$CUTIE_CKPT" \
	"$RGB_RUN/runtime_config.json" "$CUTIE_RUN/runtime_config.json" \
	"$RGB_EXP" "$CUTIE_EXP" "$TRAINING_SUMMARY" "$RGB_RUN" "$CUTIE_RUN" <<'PY'
import json
import math
import sys
from pathlib import Path

video_root, manifest_dir, support, oc_repo, cutie_checkpoint = (
    Path(value).expanduser().resolve() for value in sys.argv[1:6]
)
rgb_path, cutie_path = (Path(value).resolve() for value in sys.argv[6:8])
rgb_exp, cutie_exp = sys.argv[8:10]
training_summary_path = Path(sys.argv[10]).resolve()
rgb_run, cutie_run = (Path(value).resolve() for value in sys.argv[11:13])

payloads = {}
for name, path in (("rgb", rgb_path), ("cutie", cutie_path)):
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise AssertionError((path, "runtime config is not an object"))
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
    }
    bad = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    expected_exp = rgb_exp if name == "rgb" else cutie_exp
    if payload.get("exp_name") != expected_exp:
        bad["exp_name"] = (payload.get("exp_name"), expected_exp)
    if bad:
        raise SystemExit(f"Frozen source protocol mismatch in {path}: {bad}")
    actual_video_root = Path(payload["video_background_root"]).expanduser().resolve()
    actual_manifest_dir = Path(payload["video_background_manifest_dir"]).expanduser().resolve()
    if actual_video_root != video_root or actual_manifest_dir != manifest_dir:
        raise SystemExit(
            f"External background path mismatch in {path}: "
            f"{actual_video_root}, {actual_manifest_dir}"
        )
    payloads[name] = payload

rgb = payloads["rgb"]
if rgb.get("flat_anchor") is not False or int(rgb.get("latent_dim", -1)) != 512:
    raise SystemExit("RGB source is not the frozen official 512-D baseline.")

cutie = payloads["cutie"]
if (
    cutie.get("flat_anchor") is not True
    or cutie.get("flat_anchor_mode") != "cutie_hybrid"
    or int(cutie.get("latent_dim", -1)) != 640
    or int(cutie.get("flat_anchor_scene_dim", -1)) != 512
):
    raise SystemExit("Cutie source is not the frozen 512+128-D hybrid.")
expected_cutie_paths = {
    "cutie_object_support_path": support,
    "cutie_object_repo": oc_repo,
    "cutie_object_checkpoint": cutie_checkpoint,
}
for key, expected in expected_cutie_paths.items():
    actual = Path(cutie[key]).expanduser().resolve()
    if actual != expected:
        raise SystemExit(
            f"External Cutie path mismatch in {cutie_path}: "
            f"{key}={actual}, expected {expected}"
        )

with training_summary_path.open(encoding="utf-8") as file:
    training_summary = json.load(file)
if not isinstance(training_summary, dict):
    raise SystemExit("Source paired training summary is not a JSON object.")
expected_summary = {
    "total_steps": 250000,
    "eval_freq": 25000,
    "eval_episodes": 5,
    "expected_cutie_frames": 278055,
}
bad_summary = {
    key: (training_summary.get(key), expected)
    for key, expected in expected_summary.items()
    if training_summary.get(key) != expected
}
if bad_summary:
    raise SystemExit(f"Source paired training summary protocol mismatch: {bad_summary}")
summary_runs = training_summary.get("runs")
if not isinstance(summary_runs, dict):
    raise SystemExit("Source paired training summary is missing runs.")
for name, expected_root in (("rgb", rgb_run), ("cutie", cutie_run)):
    run = summary_runs.get(name)
    if not isinstance(run, dict) or Path(run.get("root", "")).resolve() != expected_root:
        raise SystemExit(
            f"Source paired training summary {name} root mismatch: "
            f"{None if not isinstance(run, dict) else run.get('root')}, expected {expected_root}"
        )
    rewards = run.get("eval_rewards")
    if (
        not isinstance(rewards, list)
        or len(rewards) != 11
        or not all(math.isfinite(float(value)) for value in rewards)
    ):
        raise SystemExit(f"Source paired training summary {name} rewards are invalid.")
training_gates = training_summary.get("gates")
training_status = training_summary.get("status")
if (
    not isinstance(training_gates, dict)
    or not training_gates
    or not all(isinstance(value, bool) for value in training_gates.values())
    or not isinstance(training_status, str)
    or (training_status.endswith("_pass") != all(training_gates.values()))
):
    raise SystemExit("Source paired training runtime status/gates are inconsistent.")
print("SEED4_250K_HELDOUT_EXTERNAL_PROVENANCE_OK")
PY

GPU_INFO="$(CUDA_VISIBLE_DEVICES="$GPU_HELDOUT" "$PY" -c \
	'import json,torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(json.dumps({"name":torch.cuda.get_device_name(0),"capability":torch.cuda.get_device_capability(0)},sort_keys=True))')"
echo "Held-out physical GPU $GPU_HELDOUT: $GPU_INFO"

ACTIVE_PIDS=()
OWN_STAGING=0
cleanup() {
	local rc=$? pid failed
	trap - EXIT INT TERM
	for pid in "${ACTIVE_PIDS[@]}"; do
		if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; fi
	done
	for pid in "${ACTIVE_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
	if (( rc != 0 && OWN_STAGING == 1 )) && [[ -d "$STAGING" ]]; then
		failed="${OUTPUT_ROOT}.failed.$(date +%Y%m%d_%H%M%S).$$"
		if mv -T -- "$STAGING" "$failed"; then
			echo "Preserved failed run at: $failed" >&2
		else
			echo "Could not preserve failed staging directory: $STAGING" >&2
		fi
	fi
	exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p -- "$(dirname -- "$STAGING")"
if ! mkdir -- "$STAGING"; then
	echo "Could not acquire exclusive staging directory: $STAGING" >&2
	exit 3
fi
OWN_STAGING=1
if [[ -e "$OUTPUT_ROOT" ]]; then
	echo "Output appeared while acquiring staging; refusing to overwrite: $OUTPUT_ROOT" >&2
	exit 3
fi
mkdir -p -- "$STAGING/seed4"

echo "[1/4] Running frozen integration and multiprocessing-spawn contracts"
"$PY" -c 'import tdmpc2.tools.evaluate_cutie_hybrid_heldout; from tdmpc2.tdmpc2 import TDMPC2; print("HELDOUT_IMPORT_OK", TDMPC2.__name__)'
"$PY" -m tdmpc2.check_cutie_hybrid_heldout_spawn_contract
"$PY" -m tdmpc2.check_cutie_hybrid_heldout_training_launch_contract
"$PY" tdmpc2/check_cutie_object_wrapper_contract.py
CUDA_VISIBLE_DEVICES="$GPU_HELDOUT" "$PY" tdmpc2/check_cutie_hybrid_contract.py
"$PY" tdmpc2/check_cutie_oc_adapter_contract.py

echo "[2/4] Running official whole-arm/goal Cutie preflight on physical GPU $GPU_HELDOUT"
CUDA_VISIBLE_DEVICES="$GPU_HELDOUT" "$PY" -m tdmpc2.tools.check_cutie_oc_preflight \
	--oc-storm-repo "$OC_REPO" \
	--checkpoint "$CUTIE_CKPT" \
	--support-annotations "$SUPPORT" \
	--object-schema whole_arm_goal_v1 \
	--model-size small \
	--prompt-radius 2.0 \
	--tracker-size 448 448 \
	--device cuda:0 \
	--sha256

run_one() {
	local backend=$1 run_root output log
	if [[ "$backend" == "rgb" ]]; then
		run_root="$RGB_RUN"
		output="$STAGING/seed4/rgb.json"
		log="$STAGING/seed4/rgb.log"
	else
		run_root="$CUTIE_RUN"
		output="$STAGING/seed4/cutie.json"
		log="$STAGING/seed4/cutie.log"
	fi
	exec env CUDA_VISIBLE_DEVICES="$GPU_HELDOUT" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" -m tdmpc2.tools.evaluate_cutie_hybrid_heldout \
		--runtime-config "$run_root/runtime_config.json" \
		--checkpoint "$run_root/models/final.pt" \
		--backend "$backend" \
		--training-seed "$TRAINING_SEED" \
		--output "$output" \
		--episodes "$EPISODES" \
		--env-seed "$ENV_SEED" \
		--background-seed "$BACKGROUND_SEED" \
		--planner-seed-base "$PLANNER_SEED_BASE" \
		--expected-training-steps "$TRAINING_STEPS" \
		--expected-training-eval-freq "$TRAINING_EVAL_FREQ" \
		--expected-training-eval-episodes "$TRAINING_EVAL_EPISODES" \
		>"$log" 2>&1
}

run_solo() {
	local backend=$1 rc pid
	run_one "$backend" &
	pid=$!
	ACTIVE_PIDS=("$pid")
	set +e
	wait "$pid"
	rc=$?
	set -e
	ACTIVE_PIDS=()
	return "$rc"
}

echo "[3/4] Evaluating seed-4 RGB and Cutie checkpoints sequentially on physical GPU $GPU_HELDOUT"
echo "Validation logs: $STAGING/seed4/{rgb,cutie}.log"
run_solo rgb
run_solo cutie_hybrid

echo "[4/4] Checking exact episode pairing and writing the single-seed summary"
"$PY" - \
	"$STAGING" "$EPISODES" "$ENV_SEED" "$BACKGROUND_SEED" \
	"$PLANNER_SEED_BASE" "$GPU_HELDOUT" "$TRAINING_SUMMARY" \
	"$RGB_RUN" "$CUTIE_RUN" "$TRAINING_MAX_WALLCLOCK_SECONDS" <<'PY'
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
expected_rgb_run = Path(sys.argv[8]).resolve()
expected_cutie_run = Path(sys.argv[9]).resolve()
training_max_wallclock = int(sys.argv[10])
training_seed = 4
runs = {}


def load(path):
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise AssertionError((path, "payload is not an object"))
    return value


def require_sha256(value, location):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AssertionError((location, value))
    return value


training_summary_bytes = training_summary_path.read_bytes()
training_summary_sha256 = hashlib.sha256(training_summary_bytes).hexdigest()
training_summary = json.loads(training_summary_bytes)
if not isinstance(training_summary, dict):
    raise AssertionError("Source paired training summary is not an object.")
expected_training_summary = {
    "total_steps": 250000,
    "eval_freq": 25000,
    "eval_episodes": 5,
    "expected_cutie_frames": 278055,
}
bad_training_summary = {
    key: (training_summary.get(key), expected)
    for key, expected in expected_training_summary.items()
    if training_summary.get(key) != expected
}
if bad_training_summary:
    raise AssertionError(("source paired training protocol", bad_training_summary))
source_training_status = training_summary.get("status")
source_training_gates = training_summary.get("gates")
expected_training_gate_keys = {
    "frames",
    "valid_frame_rate",
    "max_invalid_burst",
    "worker_restarts",
    "timeouts",
    "ms_per_frame",
    "runtime_unit",
    "wallclock_seconds",
}
if (
    not isinstance(source_training_status, str)
    or not isinstance(source_training_gates, dict)
    or set(source_training_gates) != expected_training_gate_keys
    or not all(isinstance(value, bool) for value in source_training_gates.values())
):
    raise AssertionError("Source paired training runtime status/gates are invalid.")
all_source_training_runtime_pass = all(source_training_gates.values())
if source_training_status.endswith("_pass") != all_source_training_runtime_pass:
    raise AssertionError("Source paired training runtime status/gates are inconsistent.")
source_training_elapsed = float(training_summary.get("elapsed_seconds", math.nan))
if not math.isfinite(source_training_elapsed) or source_training_elapsed < 0:
    raise AssertionError(("source training elapsed_seconds", source_training_elapsed))
source_training_runtime = training_summary.get("cutie_runtime")
if not isinstance(source_training_runtime, dict):
    raise AssertionError("Source paired training Cutie runtime is missing.")
recomputed_source_training_gates = {
    "frames": int(source_training_runtime.get("frames", -1)) == 278055,
    "valid_frame_rate": float(source_training_runtime.get("valid_frame_rate", -1.0)) >= 0.95,
    "max_invalid_burst": int(source_training_runtime.get("max_invalid_burst", 10**9)) <= 5,
    "worker_restarts": int(source_training_runtime.get("worker_restarts", -1)) == 0,
    "timeouts": int(source_training_runtime.get("timeouts", -1)) == 0,
    "ms_per_frame": float(source_training_runtime.get("ms_per_frame", math.inf)) <= 800.0,
    "runtime_unit": source_training_runtime.get("runtime_unit")
    == "milliseconds_per_tracked_frame_excluding_support_prompts",
    "wallclock_seconds": source_training_elapsed <= training_max_wallclock,
}
if source_training_gates != recomputed_source_training_gates:
    raise AssertionError(
        (
            "source paired training runtime gates do not recompute",
            source_training_gates,
            recomputed_source_training_gates,
        )
    )
source_training_runs = training_summary.get("runs")
if not isinstance(source_training_runs, dict):
    raise AssertionError("Source paired training run roots are missing.")
for name, expected_root in (
    ("rgb", expected_rgb_run),
    ("cutie", expected_cutie_run),
):
    run = source_training_runs.get(name)
    if not isinstance(run, dict) or Path(run.get("root", "")).resolve() != expected_root:
        raise AssertionError(("source paired training run root", name, run, expected_root))
    rewards = run.get("eval_rewards")
    if (
        not isinstance(rewards, list)
        or len(rewards) != 11
        or not all(math.isfinite(float(value)) for value in rewards)
    ):
        raise AssertionError(("source paired training eval rewards", name, rewards))
source_training_deltas = training_summary.get("paired_eval_reward_delta_cutie_minus_rgb")
if not isinstance(source_training_deltas, list) or len(source_training_deltas) != 11:
    raise AssertionError("Source paired training deltas are missing or incomplete.")
for index, (rgb_reward, cutie_reward, delta) in enumerate(
    zip(
        source_training_runs["rgb"]["eval_rewards"],
        source_training_runs["cutie"]["eval_rewards"],
        source_training_deltas,
    )
):
    expected_delta = float(cutie_reward) - float(rgb_reward)
    if not math.isfinite(float(delta)) or not math.isclose(
        float(delta), expected_delta, rel_tol=0.0, abs_tol=1e-9
    ):
        raise AssertionError(("source paired training delta", index, delta, expected_delta))


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
expected_source_launch = {
    "task": "reacher-visual-small",
    "obs": "rgb",
    "model_size": 5,
    "steps": 250000,
    "eval_freq": 25000,
    "eval_episodes": 5,
    "episode_length": 500,
    "video_background_enabled": True,
    "video_background_split": "train",
    "compile": True,
    "compile_fallback_random": True,
}
expected_source_training_identity = {
    "steps": 250000,
    "eval_freq": 25000,
    "eval_episodes": 5,
    "episode_length": 500,
}

for backend, filename in (("rgb", "rgb.json"), ("cutie_hybrid", "cutie.json")):
    path = root / "seed4" / filename
    payload = load(path)
    if payload.get("format") != "cutie_hybrid_heldout_evaluation_v1":
        raise AssertionError((path, payload.get("format")))
    if payload.get("task") != "reacher-visual-small":
        raise AssertionError((path, payload.get("task")))
    if payload.get("backend") != backend or payload.get("training_seed") != training_seed:
        raise AssertionError((path, payload.get("backend"), payload.get("training_seed")))
    if payload.get("evaluation") != expected_evaluation:
        raise AssertionError((path, payload.get("evaluation")))
    if payload.get("expected_source_training_launch") != expected_source_training_identity:
        raise AssertionError((path, payload.get("expected_source_training_launch")))
    protocol = payload.get("source_training_protocol")
    if not isinstance(protocol, dict):
        raise AssertionError((path, "missing source training protocol"))
    bad_source = {
        key: (protocol.get(key), expected)
        for key, expected in expected_source_launch.items()
        if protocol.get(key) != expected
    }
    if bad_source:
        raise AssertionError((path, "source launch mismatch", bad_source))
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise AssertionError((path, "missing provenance"))
    for key in (
        "runtime_config_sha256",
        "checkpoint_sha256",
        "evaluator_sha256",
        "validation_manifest_sha256",
        "combined_manifest_sha256",
    ):
        require_sha256(provenance.get(key), f"{path}:{key}")
    if provenance.get("cuda_visible_devices") != gpu_selector:
        raise AssertionError((path, "wrong physical GPU selector", provenance.get("cuda_visible_devices")))
    if provenance.get("logical_cuda_device") != 0:
        raise AssertionError((path, "wrong logical CUDA device"))
    rows = payload.get("episodes")
    if not isinstance(rows, list) or len(rows) != episodes_expected:
        raise AssertionError((path, "episode count"))
    for index, row in enumerate(rows):
        if row.get("episode_index") != index:
            raise AssertionError((path, index, row))
        if row.get("planner_seed") != planner_seed_base + index:
            raise AssertionError((path, index, "planner seed", row.get("planner_seed")))
        if row.get("length") != 500:
            raise AssertionError((path, index, "episode length", row.get("length")))
        if not math.isfinite(float(row.get("reward", math.nan))):
            raise AssertionError((path, index, "reward", row.get("reward")))
        if not math.isfinite(float(row.get("success", math.nan))):
            raise AssertionError((path, index, "success", row.get("success")))
        for key in ("planner_rng_start_sha256", "planner_rng_end_sha256", "initial_rgb_sha256"):
            require_sha256(row.get(key), f"{path}:episode{index}:{key}")
        if not isinstance(row.get("background_source"), str) or not row["background_source"]:
            raise AssertionError((path, index, "background source"))
        if not isinstance(row.get("background_start_frame_index"), int):
            raise AssertionError((path, index, "background frame index"))
    runs[backend] = payload

if runs["rgb"]["source_training_protocol"] != runs["cutie_hybrid"]["source_training_protocol"]:
    raise AssertionError("RGB/Cutie source training protocols differ.")

rgb_provenance = runs["rgb"]["provenance"]
cutie_provenance = runs["cutie_hybrid"]["provenance"]
for name, provenance, expected_root in (
    ("rgb", rgb_provenance, expected_rgb_run),
    ("cutie", cutie_provenance, expected_cutie_run),
):
    if Path(provenance.get("runtime_config", "")).resolve() != expected_root / "runtime_config.json":
        raise AssertionError((name, "runtime config provenance", provenance.get("runtime_config")))
    if Path(provenance.get("checkpoint", "")).resolve() != expected_root / "models" / "final.pt":
        raise AssertionError((name, "checkpoint provenance", provenance.get("checkpoint")))
for key in ("evaluator_sha256", "validation_manifest_sha256", "combined_manifest_sha256"):
    if rgb_provenance[key] != cutie_provenance[key]:
        raise AssertionError((key, rgb_provenance[key], cutie_provenance[key]))
device_identity = (
    rgb_provenance.get("device_name"),
    tuple(rgb_provenance.get("device_capability", ())),
)
if device_identity != (
    cutie_provenance.get("device_name"),
    tuple(cutie_provenance.get("device_capability", ())),
):
    raise AssertionError("RGB/Cutie evaluations used different GPU identities.")
if not isinstance(device_identity[0], str) or len(device_identity[1]) != 2:
    raise AssertionError(("invalid GPU identity", device_identity))

if (
    runs["rgb"].get("cutie_training_protocol") is not None
    or rgb_provenance.get("cutie_inputs") is not None
    or rgb_provenance.get("cutie_ready") is not None
    or runs["rgb"].get("perception_runtime") is not None
):
    raise AssertionError("RGB held-out evaluation contains Cutie-only state.")
cutie_protocol = runs["cutie_hybrid"].get("cutie_training_protocol")
cutie_inputs = cutie_provenance.get("cutie_inputs")
cutie_ready = cutie_provenance.get("cutie_ready")
if not isinstance(cutie_protocol, dict) or not isinstance(cutie_inputs, dict):
    raise AssertionError("Cutie protocol/input provenance is missing.")
for key in ("checkpoint_sha256", "support_annotations_sha256"):
    require_sha256(cutie_inputs.get(key), f"cutie_inputs:{key}")
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
if not isinstance(cutie_ready, dict):
    raise AssertionError("Cutie ready report is missing.")
bad_ready = {
    key: (cutie_ready.get(key), expected)
    for key, expected in expected_ready.items()
    if cutie_ready.get(key) != expected
}
if bad_ready:
    raise AssertionError(("Cutie ready mismatch", bad_ready))

rgb_rows = runs["rgb"]["episodes"]
cutie_rows = runs["cutie_hybrid"]["episodes"]
for index, (rgb_row, cutie_row) in enumerate(zip(rgb_rows, cutie_rows)):
    rgb_condition = (
        rgb_row["episode_index"],
        rgb_row["planner_seed"],
        rgb_row["planner_rng_start_sha256"],
        rgb_row["planner_rng_end_sha256"],
        rgb_row["initial_rgb_sha256"],
        rgb_row["background_source"],
        rgb_row["background_start_frame_index"],
        rgb_row["length"],
    )
    cutie_condition = (
        cutie_row["episode_index"],
        cutie_row["planner_seed"],
        cutie_row["planner_rng_start_sha256"],
        cutie_row["planner_rng_end_sha256"],
        cutie_row["initial_rgb_sha256"],
        cutie_row["background_source"],
        cutie_row["background_start_frame_index"],
        cutie_row["length"],
    )
    if rgb_condition != cutie_condition:
        raise AssertionError(("held-out pairing mismatch", index, rgb_condition, cutie_condition))

runtime = runs["cutie_hybrid"].get("perception_runtime")
if not isinstance(runtime, dict):
    raise AssertionError("Cutie perception runtime is missing.")
expected_frames = episodes_expected * 501
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
all_heldout_runtime_pass = all(runtime_gates.values())
all_runtime_pass = all_source_training_runtime_pass and all_heldout_runtime_pass

rgb_rewards = [float(row["reward"]) for row in rgb_rows]
cutie_rewards = [float(row["reward"]) for row in cutie_rows]
rgb_successes = [float(row["success"]) for row in rgb_rows]
cutie_successes = [float(row["success"]) for row in cutie_rows]
deltas = [cutie - rgb for rgb, cutie in zip(rgb_rewards, cutie_rewards)]
delta_mean = statistics.mean(deltas)
delta_std = statistics.stdev(deltas)
delta_se = delta_std / math.sqrt(len(deltas))
t_critical_df19 = 2.093024054408263
half_width = t_critical_df19 * delta_se
win_count = sum(delta > 0 for delta in deltas)
tie_count = sum(delta == 0 for delta in deltas)
loss_count = sum(delta < 0 for delta in deltas)
reward_direction_gates = {
    "mean_delta_positive": delta_mean > 0,
    "median_delta_positive": statistics.median(deltas) > 0,
    "paired_win_rate_gt_half": win_count / len(deltas) > 0.5,
}
reward_direction_pass = all(reward_direction_gates.values())
if all_runtime_pass:
    overall_status = "single_seed_validation_evidence_runtime_pass"
elif not all_source_training_runtime_pass and not all_heldout_runtime_pass:
    overall_status = "source_training_and_heldout_runtime_fail"
elif not all_source_training_runtime_pass:
    overall_status = "source_training_runtime_fail"
else:
    overall_status = "heldout_runtime_fail"

summary = {
    "format": "cutie_hybrid_250k_seed4_heldout_summary_v1",
    "status": overall_status,
    "scientific_scope": (
        "held-out validation evidence conditional on one trained seed (seed 4); "
        "episode-level uncertainty does not estimate training-seed variance, "
        "the test split remains untouched, and no algorithm-level claim is made"
    ),
    "source_training_launch": {
        "training_seed": training_seed,
        "steps": 250000,
        "eval_freq": 25000,
        "eval_episodes": 5,
        "max_wallclock_seconds": training_max_wallclock,
        "rgb_run": str(Path(rgb_provenance["runtime_config"]).resolve().parent),
        "cutie_run": str(Path(cutie_provenance["runtime_config"]).resolve().parent),
    },
    "source_training_runtime": {
        "paired_summary": str(training_summary_path),
        "paired_summary_sha256": training_summary_sha256,
        "status": source_training_status,
        "gates": source_training_gates,
        "all_runtime_gates_pass": all_source_training_runtime_pass,
        "elapsed_seconds": source_training_elapsed,
        "expected_cutie_frames": training_summary["expected_cutie_frames"],
        "cutie_runtime": source_training_runtime,
    },
    "heldout_protocol": {
        "split": "validation",
        "test_split_accessed": False,
        "episodes": episodes_expected,
        "environment_seed": env_seed,
        "background_seed": background_seed,
        "planner_seed_base": planner_seed_base,
        "same_physical_gpu_sequential": True,
        "physical_gpu_selector": gpu_selector,
        "device_name": device_identity[0],
        "device_capability": list(device_identity[1]),
        "exact_initial_conditions": True,
        "planner_rng_boundary_hashes_match": True,
        "evaluator_sha256": rgb_provenance["evaluator_sha256"],
        "validation_manifest_sha256": rgb_provenance["validation_manifest_sha256"],
        "combined_manifest_sha256": rgb_provenance["combined_manifest_sha256"],
    },
    "source_training_protocol": runs["rgb"]["source_training_protocol"],
    "cutie_training_protocol": cutie_protocol,
    "cutie_input_provenance": cutie_inputs,
    "checkpoint_provenance": {
        "rgb": {
            "runtime_config": rgb_provenance["runtime_config"],
            "runtime_config_sha256": rgb_provenance["runtime_config_sha256"],
            "checkpoint": rgb_provenance["checkpoint"],
            "checkpoint_sha256": rgb_provenance["checkpoint_sha256"],
        },
        "cutie": {
            "runtime_config": cutie_provenance["runtime_config"],
            "runtime_config_sha256": cutie_provenance["runtime_config_sha256"],
            "checkpoint": cutie_provenance["checkpoint"],
            "checkpoint_sha256": cutie_provenance["checkpoint_sha256"],
        },
    },
    "paired_episode_statistics": {
        "episodes": len(deltas),
        "rgb_reward": {
            "mean": statistics.mean(rgb_rewards),
            "median": statistics.median(rgb_rewards),
            "sample_std": statistics.stdev(rgb_rewards),
            "min": min(rgb_rewards),
            "max": max(rgb_rewards),
        },
        "cutie_reward": {
            "mean": statistics.mean(cutie_rewards),
            "median": statistics.median(cutie_rewards),
            "sample_std": statistics.stdev(cutie_rewards),
            "min": min(cutie_rewards),
            "max": max(cutie_rewards),
        },
        "mean_delta_cutie_minus_rgb": delta_mean,
        "median_delta_cutie_minus_rgb": statistics.median(deltas),
        "delta_sample_std": delta_std,
        "delta_standard_error": delta_se,
        "delta_min": min(deltas),
        "delta_max": max(deltas),
        "conditional_episode_mean_delta_95pct_t_interval_df19": [
            delta_mean - half_width,
            delta_mean + half_width,
        ],
        "paired_win_tie_loss": {
            "wins": win_count,
            "ties": tie_count,
            "losses": loss_count,
            "win_rate": win_count / len(deltas),
            "tie_rate": tie_count / len(deltas),
            "loss_rate": loss_count / len(deltas),
        },
        "rgb_rewards": rgb_rewards,
        "cutie_rewards": cutie_rewards,
        "paired_deltas": deltas,
        "rgb_success_rate": statistics.mean(rgb_successes),
        "cutie_success_rate": statistics.mean(cutie_successes),
        "paired_success_rate_delta": (
            statistics.mean(cutie_successes) - statistics.mean(rgb_successes)
        ),
        "reward_direction_gates": reward_direction_gates,
        "reward_direction_pass": reward_direction_pass,
    },
    "heldout_elapsed_seconds": {
        "rgb": float(runs["rgb"]["summary"]["elapsed_seconds"]),
        "cutie": float(runs["cutie_hybrid"]["summary"]["elapsed_seconds"]),
    },
    "cutie_runtime": runtime,
    "heldout_runtime_gates": runtime_gates,
    "pairing_integrity_pass": True,
    "source_training_runtime_pass": all_source_training_runtime_pass,
    "heldout_runtime_pass": all_heldout_runtime_pass,
    "eligible_for_cross_seed_algorithm_decision": False,
    "all_heldout_runtime_gates_pass": all_heldout_runtime_pass,
    "all_runtime_gates_pass": all_runtime_pass,
}
summary_path = root / "heldout_summary.json"
with summary_path.open("x", encoding="utf-8", newline="\n") as file:
    json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    file.write("\n")
print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
PY

ALL_RUNTIME_PASS="$("$PY" -c \
	'import json,sys; print(str(bool(json.load(open(sys.argv[1], encoding="utf-8"))["all_runtime_gates_pass"])).lower())' \
	"$STAGING/heldout_summary.json")"
if [[ -e "$OUTPUT_ROOT" ]]; then
	echo "Output appeared before publication; refusing to overwrite: $OUTPUT_ROOT" >&2
	exit 3
fi
mv -T -- "$STAGING" "$OUTPUT_ROOT"
OWN_STAGING=0
trap - EXIT INT TERM
if [[ "$ALL_RUNTIME_PASS" != "true" ]]; then
	echo "CUTIE_HYBRID_250K_SEED4_HELDOUT_RUNTIME_GATE_FAILED" >&2
	"$PY" -c \
		'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); print("OVERALL_STATUS="+p["status"]); print("SOURCE_TRAINING_STATUS="+p["source_training_runtime"]["status"]); print("SOURCE_TRAINING_GATES="+json.dumps(p["source_training_runtime"]["gates"],sort_keys=True)); print("HELDOUT_GATES="+json.dumps(p["heldout_runtime_gates"],sort_keys=True))' \
		"$OUTPUT_ROOT/heldout_summary.json" >&2
	echo "OUTPUT_ROOT=$OUTPUT_ROOT" >&2
	echo "SUMMARY=$OUTPUT_ROOT/heldout_summary.json" >&2
	exit 4
fi
echo "CUTIE_HYBRID_250K_SEED4_HELDOUT_EVALUATION_COMPLETE"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "SUMMARY=$OUTPUT_ROOT/heldout_summary.json"
