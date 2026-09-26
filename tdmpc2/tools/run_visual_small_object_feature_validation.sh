#!/usr/bin/env bash
# Run the leakage-safe Visual-Small object-feature validation on two GPUs.
#
# Required environment variables:
#   VIDEO_ROOT, SUPPORT, OC_REPO, CUTIE_CKPT
# Optional:
#   PY, OUTPUT_ROOT, MODE=smoke|formal, GPU_TRAIN=0, GPU_VALIDATION=1,
#   MANIFEST_DIR

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${VIDEO_ROOT:?Set VIDEO_ROOT to the existing video_hard directory}"
: "${SUPPORT:?Set SUPPORT to the verified six-frame support annotations.json}"
: "${OC_REPO:?Set OC_REPO to the OC-STORM checkout}"
: "${CUTIE_CKPT:?Set CUTIE_CKPT to cutie-small-mega.pth}"

PY="${PY:-python}"
MODE="${MODE:-smoke}"
GPU_TRAIN="${GPU_TRAIN:-0}"
GPU_VALIDATION="${GPU_VALIDATION:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/logs/perception/object_feature_validation_${MODE}_v1}"
MANIFEST_DIR="${MANIFEST_DIR:-$REPO_ROOT/tdmpc2/envs/background_manifests}"

if [[ "$GPU_TRAIN" == "$GPU_VALIDATION" ]]; then
	echo "GPU_TRAIN and GPU_VALIDATION must name two different physical GPUs." >&2
	exit 2
fi
case "$MODE" in
	smoke)
		TRAIN_EPISODES_PER_SOURCE=1
		VALIDATION_EPISODES_PER_SOURCE=1
		STEPS_PER_EPISODE=32
		;;
	formal)
		TRAIN_EPISODES_PER_SOURCE=2
		VALIDATION_EPISODES_PER_SOURCE=4
		STEPS_PER_EPISODE=250
		;;
	*)
		echo "MODE must be smoke or formal, got: $MODE" >&2
		exit 2
		;;
esac
if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
	echo "This runner requires Bash >=5.1 for fail-fast parallel waits." >&2
	exit 2
fi

if [[ -e "$OUTPUT_ROOT" ]]; then
	echo "Refusing to overwrite existing OUTPUT_ROOT: $OUTPUT_ROOT" >&2
	exit 3
fi
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/tmp/train" "$OUTPUT_ROOT/tmp/validation"

# Preserve failed logs while freeing the requested output name for a clean retry.
# Also stop any surviving sibling if the script is interrupted.
ACTIVE_PIDS=()
RUN_COMPLETE=0
cleanup() {
	local rc=$?
	local pid failed_root stamp
	trap - EXIT INT TERM
	if (( RUN_COMPLETE == 0 && rc == 0 )); then
		rc=1
	fi
	for pid in "${ACTIVE_PIDS[@]}"; do
		if kill -0 "$pid" 2>/dev/null; then
			kill "$pid" 2>/dev/null || true
		fi
	done
	for pid in "${ACTIVE_PIDS[@]}"; do
		wait "$pid" 2>/dev/null || true
	done
	if (( rc != 0 || RUN_COMPLETE == 0 )) && [[ -d "$OUTPUT_ROOT" ]]; then
		stamp="$(date -u +%Y%m%dT%H%M%SZ)"
		failed_root="${OUTPUT_ROOT}.failed.${stamp}.$$"
		if mv -- "$OUTPUT_ROOT" "$failed_root"; then
			echo "FAILED_OUTPUT_PRESERVED=$failed_root" >&2
			echo "The original OUTPUT_ROOT is free for a clean retry." >&2
		else
			echo "Could not archive failed OUTPUT_ROOT: $OUTPUT_ROOT" >&2
		fi
	fi
	exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

TRAIN_ROLLOUT="$OUTPUT_ROOT/rollouts/train"
VALIDATION_ROLLOUT="$OUTPUT_ROOT/rollouts/validation"
TRAIN_FEATURES="$OUTPUT_ROOT/features/train"
VALIDATION_FEATURES="$OUTPUT_ROOT/features/validation"
REPORT="$OUTPUT_ROOT/object_feature_probe_report.json"

echo "[1/5] Running dependency-free contracts"
"$PY" tdmpc2/check_visual_small_object_feature_validation_contract.py \
	2>&1 | tee "$OUTPUT_ROOT/logs/contracts.log"

echo "[2/5] Running official whole-arm Cutie preflight on physical GPU $GPU_TRAIN"
CUDA_VISIBLE_DEVICES="$GPU_TRAIN" "$PY" -m tdmpc2.tools.check_cutie_oc_preflight \
	--oc-storm-repo "$OC_REPO" \
	--checkpoint "$CUTIE_CKPT" \
	--object-schema whole_arm_goal_v1 \
	--support-annotations "$SUPPORT" \
	--prompt-radius 2.0 \
	--tracker-size 448 448 \
	--device cuda:0 \
	--sha256 \
	2>&1 | tee "$OUTPUT_ROOT/logs/cutie_preflight.log"

run_parallel_pair() {
	local train_pid validation_pid train_rc=0 validation_rc=0
	local finished_pid first_rc other_pid pair_rc=0
	"$@" train &
	train_pid=$!
	ACTIVE_PIDS=("$train_pid")
	"$@" validation &
	validation_pid=$!
	ACTIVE_PIDS+=("$validation_pid")
	set +e
	wait -n -p finished_pid "$train_pid" "$validation_pid"
	first_rc=$?
	set -e
	if [[ "$finished_pid" == "$train_pid" ]]; then
		train_rc=$first_rc
		other_pid=$validation_pid
	else
		validation_rc=$first_rc
		other_pid=$train_pid
	fi
	if (( first_rc != 0 )); then
		echo "One parallel worker failed (pid=$finished_pid rc=$first_rc); stopping its sibling." >&2
		kill "$other_pid" 2>/dev/null || true
	fi
	set +e
	if [[ "$other_pid" == "$train_pid" ]]; then
		wait "$train_pid"; train_rc=$?
	else
		wait "$validation_pid"; validation_rc=$?
	fi
	set -e
	ACTIVE_PIDS=()
	if (( first_rc != 0 )); then
		pair_rc=$first_rc
	elif (( train_rc != 0 )); then
		pair_rc=$train_rc
	elif (( validation_rc != 0 )); then
		pair_rc=$validation_rc
	fi
	if (( pair_rc != 0 )); then
		echo "Parallel pair failed: train_rc=$train_rc validation_rc=$validation_rc" >&2
		return "$pair_rc"
	fi
}

collect_one() {
	local split=$1 gpu episodes output log
	if [[ "$split" == train ]]; then
		gpu=$GPU_TRAIN; episodes=$TRAIN_EPISODES_PER_SOURCE
		output=$TRAIN_ROLLOUT; log="$OUTPUT_ROOT/logs/collect_train.log"
	else
		gpu=$GPU_VALIDATION; episodes=$VALIDATION_EPISODES_PER_SOURCE
		output=$VALIDATION_ROLLOUT; log="$OUTPUT_ROOT/logs/collect_validation.log"
	fi
	exec env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl \
	TMPDIR="$OUTPUT_ROOT/tmp/$split" \
	"$PY" -m tdmpc2.tools.collect_visual_small_object_rollouts \
		--output "$output" \
		--video-root "$VIDEO_ROOT" \
		--manifest-dir "$MANIFEST_DIR" \
		--split "$split" \
		--episodes-per-source "$episodes" \
		--steps-per-episode "$STEPS_PER_EPISODE" \
		--seed 271828 \
		>"$log" 2>&1
}

echo "[3/5] Collecting balanced train and validation rollouts in parallel"
run_parallel_pair collect_one

extract_one() {
	local split=$1 gpu rollout output log
	if [[ "$split" == train ]]; then
		gpu=$GPU_TRAIN; rollout=$TRAIN_ROLLOUT; output=$TRAIN_FEATURES
		log="$OUTPUT_ROOT/logs/extract_train.log"
	else
		gpu=$GPU_VALIDATION; rollout=$VALIDATION_ROLLOUT; output=$VALIDATION_FEATURES
		log="$OUTPUT_ROOT/logs/extract_validation.log"
	fi
	exec env CUDA_VISIBLE_DEVICES="$gpu" TMPDIR="$OUTPUT_ROOT/tmp/$split" \
	"$PY" -m tdmpc2.tools.extract_visual_small_cutie_object_features \
		--rollout-manifest "$rollout/manifest.json" \
		--output "$output" \
		--oc-storm-repo "$OC_REPO" \
		--checkpoint "$CUTIE_CKPT" \
		--support-annotations "$SUPPORT" \
		--config-dir "$OC_REPO/feature_extractor/cutie/cutie/config" \
		--model-size small \
		--device cuda:0 \
		--prompt-radius 2.0 \
		>"$log" 2>&1
}

echo "[4/5] Extracting causal Cutie features in two isolated processes"
run_parallel_pair extract_one

echo "[5/5] Running frozen train-only probes (validation is transformed once)"
"$PY" -m tdmpc2.tools.evaluate_visual_small_object_feature_probes \
	--train-rollout "$TRAIN_ROLLOUT/manifest.json" \
	--train-features "$TRAIN_FEATURES/manifest.json" \
	--validation-rollout "$VALIDATION_ROLLOUT/manifest.json" \
	--validation-features "$VALIDATION_FEATURES/manifest.json" \
	--output "$REPORT" \
	2>&1 | tee "$OUTPUT_ROOT/logs/probes.log"

"$PY" - "$REPORT" "$MODE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
mode = sys.argv[2]
report = json.loads(path.read_text(encoding="utf-8"))
status = report["status"]
gate_names = (
    "data_quality",
    "object_information",
    "action_increment",
    "RGB_noninferiority",
    "background_leakage_control",
)
gate_status = {name: status[name]["status"] for name in gate_names}
if mode == "smoke":
    decision = "SMOKE_COMPLETE_FORMAL_REQUIRED"
elif all(value == "pass" for value in gate_status.values()):
    decision = "ELIGIBLE_FOR_SMALL_MATCHED_RL_PILOT"
elif any(value == "fail" for value in gate_status.values()):
    decision = "NOT_ELIGIBLE_FOR_RL_PILOT"
else:
    decision = "INCONCLUSIVE"
print("OBJECT_FEATURE_VALIDATION_RESULT")
print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
print(f"PIPELINE_STATUS=COMPLETE")
print(f"SCIENTIFIC_DECISION={decision}")
print("SCIENTIFIC_SCOPE=eligibility only; no causal background invariance established")
print(f"REPORT={path.resolve()}")
PY

RUN_COMPLETE=1
