#!/usr/bin/env bash
# Two-GPU tracker-only preflight: native64 baseline versus native128 candidate.
# Native256 is exploratory; Cartpole runs by default as the health control.
# This runner never invokes train.py and never launches controller training.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${VIDEO_ROOT:?Set VIDEO_ROOT to the frozen video_hard directory}"
: "${OC_REPO:?Set OC_REPO to the OC-STORM checkout}"
: "${CUTIE_CKPT:?Set CUTIE_CKPT to cutie-small-mega.pth}"

PY="${PY:-python}"
MANIFEST_DIR="${MANIFEST_DIR:-$REPO_ROOT/tdmpc2/envs/background_manifests}"
SUPPORT_ACROBOT="${SUPPORT_ACROBOT:-$REPO_ROOT/datasets/cutie_multitask_support_v2_seed314159/acrobot-swingup/annotations.json}"
SUPPORT_CARTPOLE="${SUPPORT_CARTPOLE:-$REPO_ROOT/datasets/cutie_multitask_support_v1_seed314159/cartpole-swingup/annotations.json}"
GPU_64="${GPU_64:-0}"
GPU_128="${GPU_128:-1}"
RUN_256="${RUN_256:-0}"
RUN_CARTPOLE="${RUN_CARTPOLE:-1}"
EPISODES="${EPISODES:-20}"
STEPS="${STEPS:-500}"
ENV_SEED="${ENV_SEED:-424243}"
BACKGROUND_SEED="${BACKGROUND_SEED:-1618034}"
ACTION_SEED="${ACTION_SEED:-8675400}"
CUTIE_SEED="${CUTIE_SEED:-2718281}"
SUPPORT_SEED="${SUPPORT_SEED:-314159}"
RUN_TAG="${RUN_TAG:-cutie_native_resolution_tracker_preflight_v1}"
BASE="${OUTPUT_ROOT:-$REPO_ROOT/logs/_diagnostic/$RUN_TAG}"
STAGE="${BASE}.incomplete"
SUMMARY="$STAGE/preflight_summary.json"

for name in GPU_64 GPU_128; do
	value="${!name}"
	[[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || {
		echo "$name must be a non-negative physical GPU index: $value" >&2
		exit 2
	}
done
[[ "$GPU_64" != "$GPU_128" ]] || {
	echo "GPU_64 and GPU_128 must differ for the paired two-GPU preflight." >&2
	exit 2
}
for name in RUN_256 RUN_CARTPOLE; do
	value="${!name}"
	[[ "$value" == 0 || "$value" == 1 ]] || {
		echo "$name must be 0 or 1, got $value" >&2
		exit 2
	}
done
[[ "$EPISODES" =~ ^[1-9][0-9]*$ && "$STEPS" =~ ^[1-9][0-9]*$ ]] || {
	echo "EPISODES and STEPS must be positive integers." >&2
	exit 2
}
if [[ "$PY" == */* ]]; then
	[[ -x "$PY" ]] || { echo "Python is not executable: $PY" >&2; exit 2; }
else
	command -v "$PY" >/dev/null || { echo "Python is not on PATH: $PY" >&2; exit 2; }
fi
for path in "$VIDEO_ROOT" "$OC_REPO" "$CUTIE_CKPT" "$MANIFEST_DIR" "$SUPPORT_ACROBOT"; do
	[[ -e "$path" ]] || { echo "Missing immutable input: $path" >&2; exit 2; }
done
if (( RUN_CARTPOLE )); then
	[[ -f "$SUPPORT_CARTPOLE" ]] || {
		echo "Missing optional Cartpole support: $SUPPORT_CARTPOLE" >&2
		exit 2
	}
fi
[[ "$(basename -- "${VIDEO_ROOT%/}")" == video_hard ]] || {
	echo "VIDEO_ROOT must be the frozen video_hard directory." >&2
	exit 2
}
for path in "$BASE" "$STAGE"; do
	[[ ! -e "$path" ]] || {
		echo "Refusing to overwrite existing output: $path" >&2
		exit 3
	}
done

mkdir -p "$STAGE/contracts" "$STAGE/acrobot-swingup"
ACTIVE_PIDS=()
PROMOTED=0

terminate_tree() {
	local parent=$1 child
	while IFS= read -r child; do
		[[ -n "$child" ]] && terminate_tree "$child"
	done < <(pgrep -P "$parent" 2>/dev/null || true)
	kill -TERM "$parent" 2>/dev/null || true
}

archive_on_exit() {
	local rc=$? pid failed
	trap - EXIT INT TERM
	for pid in "${ACTIVE_PIDS[@]:-}"; do
		[[ -n "$pid" ]] && terminate_tree "$pid"
	done
	for pid in "${ACTIVE_PIDS[@]:-}"; do
		[[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
	done
	if (( PROMOTED == 0 )) && [[ -d "$STAGE" ]]; then
		if [[ ! -f "$SUMMARY" ]]; then
			"$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps({"format":"cutie_native_resolution_tracker_preflight_summary_v1","status":"runner_engineering_fail","runner_exit_code":int(sys.argv[2]),"failure":"runner exited before strict aggregation"},indent=2)+"\n",encoding="utf-8")' "$SUMMARY" "$rc" 2>/dev/null || true
		fi
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
		[[ ! -e "$failed" ]] || failed="${failed}.${RANDOM}"
		mv -- "$STAGE" "$failed"
		echo "CUTIE_NATIVE_RESOLUTION_PREFLIGHT_FAILED_ARCHIVE=$failed" >&2
	fi
	exit "$rc"
}
trap archive_on_exit EXIT INT TERM

echo "[1/4] Dependency-light core and native-resolution contracts"
"$PY" -m tdmpc2.check_cutie_native_highres_contract \
	>"$STAGE/contracts/native_highres_core.log" 2>&1
"$PY" -m tdmpc2.check_cutie_native_resolution_tracker_contract \
	>"$STAGE/contracts/native_resolution_tracker.log" 2>&1
"$PY" -m tdmpc2.check_cutie_native_resolution_wrapper_contract \
	>"$STAGE/contracts/native_resolution_wrapper.log" 2>&1

support_for_task() {
	case "$1" in
		acrobot-swingup) printf '%s' "$SUPPORT_ACROBOT" ;;
		cartpole-swingup) printf '%s' "$SUPPORT_CARTPOLE" ;;
		*) return 2 ;;
	esac
}

run_one() {
	local task=$1 resolution=$2 gpu=$3 task_dir support output log rc
	task_dir="$STAGE/$task"
	support="$(support_for_task "$task")"
	output="$task_dir/resolution_${resolution}.json"
	log="$task_dir/resolution_${resolution}.log"
	mkdir -p "$task_dir"
	echo "TRACKER_START task=$task resolution=$resolution gpu=$gpu" | tee "$log"
	set +e
	CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m \
		tdmpc2.tools.evaluate_cutie_native_resolution_tracker \
		--task "$task" \
		--resolution "$resolution" \
		--oc-storm-repo "$OC_REPO" \
		--cutie-checkpoint "$CUTIE_CKPT" \
		--support-annotations "$support" \
		--video-root "$VIDEO_ROOT" \
		--manifest-dir "$MANIFEST_DIR" \
		--episodes "$EPISODES" \
		--steps "$STEPS" \
		--env-seed "$ENV_SEED" \
		--background-seed "$BACKGROUND_SEED" \
		--action-seed "$ACTION_SEED" \
		--cutie-seed "$CUTIE_SEED" \
		--expected-support-seed "$SUPPORT_SEED" \
		--output "$output" >>"$log" 2>&1
	rc=$?
	set -e
	printf '%s\n' "$rc" >"$task_dir/resolution_${resolution}.rc"
	echo "TRACKER_END task=$task resolution=$resolution gpu=$gpu rc=$rc" | tee -a "$log"
	return "$rc"
}

run_pair() {
	local task=$1 pid64 pid128 rc64 rc128
	run_one "$task" 64 "$GPU_64" & pid64=$!; ACTIVE_PIDS+=("$pid64")
	run_one "$task" 128 "$GPU_128" & pid128=$!; ACTIVE_PIDS+=("$pid128")
	set +e
	wait "$pid64"; rc64=$?
	wait "$pid128"; rc128=$?
	set -e
	ACTIVE_PIDS=()
	if (( rc64 != 0 || rc128 != 0 )); then
		echo "$task paired tracker jobs failed: rc64=$rc64 rc128=$rc128" >&2
		return 4
	fi
	if (( RUN_256 )); then
		echo "[exploratory] native256 is excluded from all GO/NO-GO gates"
		run_one "$task" 256 "$GPU_64"
	fi
}

echo "[2/4] Acrobot native64/native128 on the same fixed trajectory"
run_pair acrobot-swingup

TASK_ARGS=(acrobot-swingup)
if (( RUN_CARTPOLE )); then
	echo "[3/4] Cartpole health-control pair"
	mkdir -p "$STAGE/cartpole-swingup"
	run_pair cartpole-swingup
	TASK_ARGS+=(cartpole-swingup)
else
	echo "[3/4] Cartpole health control skipped (RUN_CARTPOLE=0); aggregate cannot emit overall science GO"
fi

echo "[4/4] Strict pairing, quality, improvement, and latency aggregation"
set +e
"$PY" -m tdmpc2.tools.aggregate_cutie_native_resolution_tracker \
	--input-root "$STAGE" \
	--tasks "${TASK_ARGS[@]}" \
	--expected-gpu-64 "$GPU_64" \
	--expected-gpu-128 "$GPU_128" \
	--expected-gpu-256 "$GPU_64" \
	--output "$SUMMARY" >"$STAGE/aggregate.log" 2>&1
AGGREGATE_RC=$?
set -e
if (( AGGREGATE_RC != 0 )); then
	cat "$STAGE/aggregate.log" >&2
	exit "$AGGREGATE_RC"
fi

mv -- "$STAGE" "$BASE"
PROMOTED=1
trap - EXIT INT TERM
echo "CUTIE_NATIVE_RESOLUTION_TRACKER_PREFLIGHT_COMPLETE"
echo "SUMMARY=$BASE/preflight_summary.json"
echo "No controller training was launched. Read recommendation before any training."
