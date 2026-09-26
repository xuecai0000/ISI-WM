#!/usr/bin/env bash
# Strict paired held-out validation of the three trained RGB/CutieHybrid seeds.

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
GPU_RGB="${GPU_RGB:-0}"
GPU_CUTIE="${GPU_CUTIE:-1}"
EPISODES="${EPISODES:-20}"
ENV_SEED="${ENV_SEED:-424242}"
BACKGROUND_SEED="${BACKGROUND_SEED:-1618033}"
PLANNER_SEED_BASE="${PLANNER_SEED_BASE:-8675309}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/logs/_heldout/cutie_hybrid_20k_validation_v1}"
STAGING="${OUTPUT_ROOT}.incomplete"
SEEDS=(1 2 3)

if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
	echo "This runner requires Bash >=5.1." >&2
	exit 2
fi
for gpu_name in GPU_RGB GPU_CUTIE; do
	gpu_value="${!gpu_name}"
	if [[ ! "$gpu_value" =~ ^(0|[1-9][0-9]*)$ ]]; then
		echo "$gpu_name must be one canonical physical GPU index, got: $gpu_value" >&2
		exit 2
	fi
done
if [[ "$GPU_RGB" == "$GPU_CUTIE" ]]; then
	echo "GPU_RGB and GPU_CUTIE must be different physical GPU indices." >&2
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
	echo "This frozen held-out protocol requires EPISODES=20." >&2
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

RUNTIME_CONFIGS=()
for seed in "${SEEDS[@]}"; do
	RGB_RUN="$REPO_ROOT/logs/reacher-visual-small/$seed/rgb20k_cutie_hybrid_20k_pair_v1_seed$seed"
	CUTIE_RUN="$REPO_ROOT/logs/reacher-visual-small/$seed/cutie_hybrid20k_cutie_hybrid_20k_pair_v1_seed$seed"
	RUNTIME_CONFIGS+=("$RGB_RUN/runtime_config.json" "$CUTIE_RUN/runtime_config.json")
	for path in \
		"$RGB_RUN/runtime_config.json" "$RGB_RUN/models/final.pt" \
		"$CUTIE_RUN/runtime_config.json" "$CUTIE_RUN/models/final.pt"; do
		[[ -s "$path" ]] || { echo "Missing trained artifact: $path" >&2; exit 2; }
	done
done

"$PY" - "$VIDEO_ROOT" "$MANIFEST_DIR" "$SUPPORT" "$OC_REPO" "$CUTIE_CKPT" \
	"${RUNTIME_CONFIGS[@]}" <<'PY'
import json
import sys
from pathlib import Path

video_root, manifest_dir, support, oc_repo, cutie_checkpoint = (
    Path(value).expanduser().resolve() for value in sys.argv[1:6]
)
configs = [Path(value).resolve() for value in sys.argv[6:]]
if len(configs) != 6:
    raise AssertionError(configs)
for index, path in enumerate(configs):
    with path.open(encoding="utf-8") as file:
        cfg = json.load(file)
    actual_video_root = Path(cfg["video_background_root"]).expanduser().resolve()
    actual_manifest_dir = Path(cfg["video_background_manifest_dir"]).expanduser().resolve()
    if actual_video_root != video_root or actual_manifest_dir != manifest_dir:
        raise SystemExit(
            f"External background path mismatch in {path}: "
            f"{actual_video_root}, {actual_manifest_dir}"
        )
    if index % 2 == 1:
        expected = {
            "cutie_object_support_path": support,
            "cutie_object_repo": oc_repo,
            "cutie_object_checkpoint": cutie_checkpoint,
        }
        for key, value in expected.items():
            actual = Path(cfg[key]).expanduser().resolve()
            if actual != value:
                raise SystemExit(
                    f"External Cutie path mismatch in {path}: {key}={actual}, expected {value}"
                )
print("HELDOUT_EXTERNAL_PROVENANCE_OK")
PY

RGB_GPU_INFO="$(CUDA_VISIBLE_DEVICES="$GPU_RGB" "$PY" -c \
	'import json,torch; print(json.dumps({"name":torch.cuda.get_device_name(0),"capability":torch.cuda.get_device_capability(0)},sort_keys=True))')"
CUTIE_GPU_INFO="$(CUDA_VISIBLE_DEVICES="$GPU_CUTIE" "$PY" -c \
	'import json,torch; print(json.dumps({"name":torch.cuda.get_device_name(0),"capability":torch.cuda.get_device_capability(0)},sort_keys=True))')"
if [[ "$RGB_GPU_INFO" != "$CUTIE_GPU_INFO" ]]; then
	echo "Strict pairing requires matching GPU models/capabilities." >&2
	echo "RGB GPU:   $RGB_GPU_INFO" >&2
	echo "Cutie GPU: $CUTIE_GPU_INFO" >&2
	exit 2
fi

echo "[1/4] Running frozen integration contracts"
"$PY" -c 'import tdmpc2.tools.evaluate_cutie_hybrid_heldout; from tdmpc2.tdmpc2 import TDMPC2; print("HELDOUT_IMPORT_OK", TDMPC2.__name__)'
"$PY" -m tdmpc2.check_cutie_hybrid_heldout_spawn_contract
"$PY" tdmpc2/check_cutie_object_wrapper_contract.py
CUDA_VISIBLE_DEVICES="$GPU_CUTIE" "$PY" tdmpc2/check_cutie_hybrid_contract.py
"$PY" tdmpc2/check_cutie_oc_adapter_contract.py

echo "[2/4] Running official whole-arm/goal Cutie preflight"
CUDA_VISIBLE_DEVICES="$GPU_CUTIE" "$PY" -m tdmpc2.tools.check_cutie_oc_preflight \
	--oc-storm-repo "$OC_REPO" \
	--checkpoint "$CUTIE_CKPT" \
	--support-annotations "$SUPPORT" \
	--object-schema whole_arm_goal_v1 \
	--model-size small \
	--prompt-radius 2.0 \
	--tracker-size 448 448 \
	--device cuda:0 \
	--sha256

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
		if mv -- "$STAGING" "$failed"; then
			echo "Preserved failed run at: $failed" >&2
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

run_one() {
	local backend=$1 seed=$2 gpu=$3 run_root output log
	if [[ "$backend" == "rgb" ]]; then
		run_root="$REPO_ROOT/logs/reacher-visual-small/$seed/rgb20k_cutie_hybrid_20k_pair_v1_seed$seed"
		output="$STAGING/seed$seed/rgb.json"
		log="$STAGING/seed$seed/rgb.log"
	else
		run_root="$REPO_ROOT/logs/reacher-visual-small/$seed/cutie_hybrid20k_cutie_hybrid_20k_pair_v1_seed$seed"
		output="$STAGING/seed$seed/cutie.json"
		log="$STAGING/seed$seed/cutie.log"
	fi
	exec env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" -m tdmpc2.tools.evaluate_cutie_hybrid_heldout \
		--runtime-config "$run_root/runtime_config.json" \
		--checkpoint "$run_root/models/final.pt" \
		--backend "$backend" \
		--training-seed "$seed" \
		--output "$output" \
		--episodes "$EPISODES" \
		--env-seed "$ENV_SEED" \
		--background-seed "$BACKGROUND_SEED" \
		--planner-seed-base "$PLANNER_SEED_BASE" \
		>"$log" 2>&1
}

run_parallel_two() {
	local backend_a=$1 seed_a=$2 gpu_a=$3 backend_b=$4 seed_b=$5 gpu_b=$6
	run_one "$backend_a" "$seed_a" "$gpu_a" &
	PID_A=$!
	ACTIVE_PIDS=("$PID_A")
	run_one "$backend_b" "$seed_b" "$gpu_b" &
	PID_B=$!
	ACTIVE_PIDS+=("$PID_B")
	set +e
	wait -n -p FINISHED_PID "$PID_A" "$PID_B"
	FIRST_RC=$?
	set -e
	if [[ "$FINISHED_PID" == "$PID_A" ]]; then OTHER_PID=$PID_B; else OTHER_PID=$PID_A; fi
	if (( FIRST_RC != 0 )); then
		echo "Worker $FINISHED_PID failed with rc=$FIRST_RC; stopping sibling $OTHER_PID." >&2
		kill "$OTHER_PID" 2>/dev/null || true
	fi
	set +e
	wait "$OTHER_PID"
	OTHER_RC=$?
	set -e
	ACTIVE_PIDS=()
	if (( FIRST_RC != 0 )); then return "$FIRST_RC"; fi
	if (( OTHER_RC != 0 )); then return "$OTHER_RC"; fi
}

run_solo() {
	local backend=$1 seed=$2 gpu=$3 rc
	run_one "$backend" "$seed" "$gpu" &
	PID_A=$!
	ACTIVE_PIDS=("$PID_A")
	set +e
	wait "$PID_A"
	rc=$?
	set -e
	ACTIVE_PIDS=()
	return "$rc"
}

echo "[3/4] Evaluating six checkpoints on the frozen validation sequence"
echo "Each RGB/Cutie seed pair uses the same physical GPU; two GPUs run queues in parallel."
for seed in "${SEEDS[@]}"; do
	SEED_DIR="$STAGING/seed$seed"
	mkdir -p "$SEED_DIR"
	echo "  seed $seed: $SEED_DIR/{rgb,cutie}.log"
done
run_parallel_two rgb 1 "$GPU_RGB" rgb 2 "$GPU_CUTIE"
run_parallel_two cutie_hybrid 1 "$GPU_RGB" cutie_hybrid 2 "$GPU_CUTIE"
run_solo rgb 3 "$GPU_RGB"
run_solo cutie_hybrid 3 "$GPU_RGB"

echo "[4/4] Checking exact pairing and aggregating by training seed"
"$PY" - "$STAGING" "$EPISODES" "$ENV_SEED" "$BACKGROUND_SEED" "$PLANNER_SEED_BASE" <<'PY'
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
seeds = (1, 2, 3)
backends = ("rgb", "cutie")
runs = {}
reference_conditions = None
device_identity = None
source_training_protocol = None
cutie_training_protocol = None
cutie_input_provenance = None
evaluator_sha256 = None
validation_manifest_sha256 = None
combined_manifest_sha256 = None
runtime_gates = {}
seed_summaries = {}

def load(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)

def mean(values):
    return sum(values) / len(values)

for seed in seeds:
    for backend in backends:
        path = root / f"seed{seed}" / f"{backend}.json"
        payload = load(path)
        expected_backend = "rgb" if backend == "rgb" else "cutie_hybrid"
        if payload.get("format") != "cutie_hybrid_heldout_evaluation_v1":
            raise AssertionError((path, payload.get("format")))
        if payload.get("backend") != expected_backend or payload.get("training_seed") != seed:
            raise AssertionError((path, payload.get("backend"), payload.get("training_seed")))
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
        if payload.get("evaluation") != expected_evaluation:
            raise AssertionError((path, payload.get("evaluation")))
        protocol = payload.get("source_training_protocol")
        if not isinstance(protocol, dict):
            raise AssertionError((path, "missing source training protocol"))
        if source_training_protocol is None:
            source_training_protocol = protocol
        elif protocol != source_training_protocol:
            raise AssertionError((path, "source training protocol mismatch"))
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            raise AssertionError((path, "missing provenance"))
        for key, current in (
            ("evaluator_sha256", provenance.get("evaluator_sha256")),
            ("validation_manifest_sha256", provenance.get("validation_manifest_sha256")),
            ("combined_manifest_sha256", provenance.get("combined_manifest_sha256")),
        ):
            if not isinstance(current, str) or len(current) != 64:
                raise AssertionError((path, key, current))
            if key == "evaluator_sha256":
                if evaluator_sha256 is None: evaluator_sha256 = current
                elif evaluator_sha256 != current: raise AssertionError((path, key))
            elif key == "validation_manifest_sha256":
                if validation_manifest_sha256 is None: validation_manifest_sha256 = current
                elif validation_manifest_sha256 != current: raise AssertionError((path, key))
            else:
                if combined_manifest_sha256 is None: combined_manifest_sha256 = current
                elif combined_manifest_sha256 != current: raise AssertionError((path, key))
        if backend == "cutie":
            current_cutie_protocol = payload.get("cutie_training_protocol")
            current_cutie_inputs = provenance.get("cutie_inputs")
            ready = provenance.get("cutie_ready")
            if not isinstance(current_cutie_protocol, dict) or not isinstance(current_cutie_inputs, dict):
                raise AssertionError((path, "missing Cutie protocol/provenance"))
            if cutie_training_protocol is None:
                cutie_training_protocol = current_cutie_protocol
                cutie_input_provenance = current_cutie_inputs
            elif (
                current_cutie_protocol != cutie_training_protocol
                or current_cutie_inputs != cutie_input_provenance
            ):
                raise AssertionError((path, "Cutie protocol/input mismatch"))
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
        elif (
            payload.get("cutie_training_protocol") is not None
            or provenance.get("cutie_inputs") is not None
            or provenance.get("cutie_ready") is not None
        ):
            raise AssertionError((path, "RGB run contains Cutie provenance"))
        episode_rows = payload.get("episodes")
        if not isinstance(episode_rows, list) or len(episode_rows) != episodes_expected:
            raise AssertionError((path, "episode count"))
        for index, row in enumerate(episode_rows):
            if row["episode_index"] != index or row["planner_seed"] != planner_seed_base + index:
                raise AssertionError((path, index, row))
            if row["length"] != 500 or not math.isfinite(float(row["reward"])):
                raise AssertionError((path, index, row["length"], row["reward"]))
        conditions = [
            (
                row["episode_index"], row["planner_seed"],
                row["planner_rng_start_sha256"], row["planner_rng_end_sha256"],
                row["initial_rgb_sha256"], row["background_source"],
                row["background_start_frame_index"], row["length"],
            )
            for row in episode_rows
        ]
        if reference_conditions is None:
            reference_conditions = conditions
        elif conditions != reference_conditions:
            raise AssertionError(f"Held-out pairing mismatch in {path}")
        identity = (
            payload["provenance"]["device_name"],
            tuple(payload["provenance"]["device_capability"]),
        )
        if device_identity is None:
            device_identity = identity
        elif identity != device_identity:
            raise AssertionError(("heterogeneous GPU pairing", device_identity, identity))
        runs[(seed, backend)] = payload

    if (
        runs[(seed, "rgb")]["provenance"]["cuda_visible_devices"]
        != runs[(seed, "cutie")]["provenance"]["cuda_visible_devices"]
    ):
        raise AssertionError((seed, "RGB/Cutie did not use the same physical GPU selector"))
    rgb_rewards = [float(row["reward"]) for row in runs[(seed, "rgb")]["episodes"]]
    cutie_rewards = [float(row["reward"]) for row in runs[(seed, "cutie")]["episodes"]]
    deltas = [cutie - rgb for rgb, cutie in zip(rgb_rewards, cutie_rewards)]
    runtime = runs[(seed, "cutie")]["perception_runtime"]
    expected_frames = episodes_expected * 501
    gates = {
        "frames": int(runtime.get("frames", -1)) == expected_frames,
        "valid_frame_rate": float(runtime.get("valid_frame_rate", -1.0)) >= 0.95,
        "max_invalid_burst": int(runtime.get("max_invalid_burst", 10**9)) <= 5,
        "worker_restarts": int(runtime.get("worker_restarts", -1)) == 0,
        "timeouts": int(runtime.get("timeouts", -1)) == 0,
        "ms_per_frame": float(runtime.get("ms_per_frame", math.inf)) <= 800.0,
        "runtime_unit": runtime.get("runtime_unit") == "milliseconds_per_tracked_frame_excluding_support_prompts",
    }
    runtime_gates[str(seed)] = gates
    seed_summaries[str(seed)] = {
        "rgb_reward_mean": mean(rgb_rewards),
        "cutie_reward_mean": mean(cutie_rewards),
        "mean_delta_cutie_minus_rgb": mean(deltas),
        "paired_win_rate": sum(delta > 0 for delta in deltas) / len(deltas),
        "rgb_rewards": rgb_rewards,
        "cutie_rewards": cutie_rewards,
        "paired_deltas": deltas,
        "cutie_runtime": runtime,
        "runtime_gates": gates,
    }

rgb_seed_means = [seed_summaries[str(seed)]["rgb_reward_mean"] for seed in seeds]
cutie_seed_means = [seed_summaries[str(seed)]["cutie_reward_mean"] for seed in seeds]
delta_seed_means = [seed_summaries[str(seed)]["mean_delta_cutie_minus_rgb"] for seed in seeds]
macro_delta = mean(delta_seed_means)
seed_delta_sd = statistics.stdev(delta_seed_means)
t_critical_df2 = 4.3026527297
half_width = t_critical_df2 * seed_delta_sd / math.sqrt(len(seeds))
all_runtime_pass = all(all(gates.values()) for gates in runtime_gates.values())
summary = {
    "format": "cutie_hybrid_heldout_summary_v1",
    "status": (
        "heldout_validation_runtime_pass" if all_runtime_pass
        else "heldout_validation_runtime_fail"
    ),
    "scientific_scope": (
        "held-out validation development evidence with three training seeds; "
        "the test split remains untouched and no final algorithm claim is made"
    ),
    "pairing": {
        "exact_initial_conditions": True,
        "exact_planner_rng_start_and_end": True,
        "episodes_per_checkpoint": episodes_expected,
        "training_seeds": list(seeds),
        "environment_seed": env_seed,
        "background_seed": background_seed,
        "planner_seed_base": planner_seed_base,
        "device_name": device_identity[0],
        "device_capability": list(device_identity[1]),
        "evaluator_sha256": evaluator_sha256,
        "validation_manifest_sha256": validation_manifest_sha256,
        "combined_manifest_sha256": combined_manifest_sha256,
    },
    "source_training_protocol": source_training_protocol,
    "cutie_training_protocol": cutie_training_protocol,
    "cutie_input_provenance": cutie_input_provenance,
    "per_seed": seed_summaries,
    "macro_by_training_seed": {
        "rgb_reward_mean": mean(rgb_seed_means),
        "cutie_reward_mean": mean(cutie_seed_means),
        "mean_delta_cutie_minus_rgb": macro_delta,
        "seed_mean_deltas": delta_seed_means,
        "seed_delta_sample_std": seed_delta_sd,
        "mean_delta_95pct_t_interval_n3": [macro_delta - half_width, macro_delta + half_width],
        "all_three_seed_means_positive": all(delta > 0 for delta in delta_seed_means),
    },
    "runtime_gates": runtime_gates,
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
mv -- "$STAGING" "$OUTPUT_ROOT"
trap - EXIT INT TERM
if [[ "$ALL_RUNTIME_PASS" != "true" ]]; then
	echo "CUTIE_HYBRID_HELDOUT_RUNTIME_GATE_FAILED" >&2
	echo "OUTPUT_ROOT=$OUTPUT_ROOT" >&2
	echo "SUMMARY=$OUTPUT_ROOT/heldout_summary.json" >&2
	exit 4
fi
echo "CUTIE_HYBRID_HELDOUT_EVALUATION_COMPLETE"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "SUMMARY=$OUTPUT_ROOT/heldout_summary.json"
