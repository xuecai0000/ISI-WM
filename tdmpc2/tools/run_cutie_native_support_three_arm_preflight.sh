#!/usr/bin/env bash
# Three-arm tracker-only native runtime/support resolution preflight.
#
# A = runtime64/support64
# B = runtime128/support64
# C = runtime128/support128
#
# B and C run concurrently on matched GPU models. A and C use the same
# physical GPU. This script never invokes train.py or launches controller
# training; a scientific GO only permits a separately preregistered pilot.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${VIDEO_ROOT:?Set VIDEO_ROOT to the frozen video_hard directory}"
: "${OC_REPO:?Set OC_REPO to the OC-STORM checkout}"
: "${CUTIE_CKPT:?Set CUTIE_CKPT to cutie-small-mega.pth}"

PY="${PY:-python}"
MANIFEST_DIR="${MANIFEST_DIR:-$REPO_ROOT/tdmpc2/envs/background_manifests}"
GPU_B="${GPU_B:-0}"
GPU_A_C="${GPU_A_C:-1}"
RUN_CARTPOLE="${RUN_CARTPOLE:-1}"
EPISODES="${EPISODES:-20}"
STEPS="${STEPS:-500}"
ENV_SEED="${ENV_SEED:-424243}"
BACKGROUND_SEED="${BACKGROUND_SEED:-1618034}"
ACTION_SEED="${ACTION_SEED:-8675400}"
CUTIE_SEED="${CUTIE_SEED:-2718281}"
SUPPORT_SEED="${SUPPORT_SEED:-314159}"
SUPPORT_BACKGROUND_SEED="${SUPPORT_BACKGROUND_SEED:-314160}"
SUPPORT_ACTION_SEED="${SUPPORT_ACTION_SEED:-314161}"
MAX_SUPPORT_RESET_ATTEMPTS="${MAX_SUPPORT_RESET_ATTEMPTS:-1000}"
RUN_TAG="${RUN_TAG:-cutie_native_support_three_arm_preflight_v2}"
BASE="${OUTPUT_ROOT:-$REPO_ROOT/logs/_diagnostic/$RUN_TAG}"
STAGE="${BASE}.incomplete"
SUMMARY="$STAGE/three_arm_summary.json"

for name in GPU_B GPU_A_C; do
	value="${!name}"
	[[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || {
		echo "$name must be a non-negative physical GPU index: $value" >&2
		exit 2
	}
done
[[ "$GPU_B" != "$GPU_A_C" ]] || {
	echo "GPU_B and GPU_A_C must differ for concurrent B/C evaluation." >&2
	exit 2
}
[[ "$RUN_CARTPOLE" == 0 || "$RUN_CARTPOLE" == 1 ]] || {
	echo "RUN_CARTPOLE must be 0 or 1, got $RUN_CARTPOLE" >&2
	exit 2
}
for name in EPISODES STEPS MAX_SUPPORT_RESET_ATTEMPTS; do
	value="${!name}"
	[[ "$value" =~ ^[1-9][0-9]*$ ]] || {
		echo "$name must be a positive integer: $value" >&2
		exit 2
	}
done
[[ "$EPISODES" == 20 && "$STEPS" == 500 ]] || {
	echo "This scientific preflight is fixed to EPISODES=20 and STEPS=500." >&2
	exit 2
}
for name in ENV_SEED BACKGROUND_SEED ACTION_SEED CUTIE_SEED SUPPORT_SEED SUPPORT_BACKGROUND_SEED SUPPORT_ACTION_SEED; do
	value="${!name}"
	[[ "$value" =~ ^-?(0|[1-9][0-9]*)$ ]] || {
		echo "$name must be an integer: $value" >&2
		exit 2
	}
done
[[ "$SUPPORT_SEED" != "$SUPPORT_BACKGROUND_SEED" \
	&& "$SUPPORT_SEED" != "$SUPPORT_ACTION_SEED" \
	&& "$SUPPORT_BACKGROUND_SEED" != "$SUPPORT_ACTION_SEED" ]] || {
	echo "The three support collection seeds must be distinct." >&2
	exit 2
}
[[ "$ENV_SEED" != "$BACKGROUND_SEED" \
	&& "$ENV_SEED" != "$ACTION_SEED" \
	&& "$ENV_SEED" != "$CUTIE_SEED" \
	&& "$BACKGROUND_SEED" != "$ACTION_SEED" \
	&& "$BACKGROUND_SEED" != "$CUTIE_SEED" \
	&& "$ACTION_SEED" != "$CUTIE_SEED" ]] || {
	echo "The environment/background/action/Cutie evaluation seeds must be distinct." >&2
	exit 2
}
ALL_SEEDS=(
	"$ENV_SEED" "$BACKGROUND_SEED" "$ACTION_SEED" "$CUTIE_SEED"
	"$SUPPORT_SEED" "$SUPPORT_BACKGROUND_SEED" "$SUPPORT_ACTION_SEED"
)
if [[ "$(printf '%s\n' "${ALL_SEEDS[@]}" | sort -u | wc -l)" -ne "${#ALL_SEEDS[@]}" ]]; then
	echo "Support-collection and evaluation seed domains must all be distinct." >&2
	exit 2
fi
if [[ "$PY" == */* ]]; then
	[[ -x "$PY" ]] || { echo "Python is not executable: $PY" >&2; exit 2; }
else
	command -v "$PY" >/dev/null || { echo "Python is not on PATH: $PY" >&2; exit 2; }
fi
for path in "$VIDEO_ROOT" "$OC_REPO" "$CUTIE_CKPT" "$MANIFEST_DIR"; do
	[[ -e "$path" ]] || { echo "Missing immutable input: $path" >&2; exit 2; }
done
[[ -d "$VIDEO_ROOT" && -d "$OC_REPO" && -f "$CUTIE_CKPT" && -d "$MANIFEST_DIR" ]] || {
	echo "VIDEO_ROOT/OC_REPO/MANIFEST_DIR must be directories and CUTIE_CKPT a file." >&2
	exit 2
}
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

mkdir -p "$STAGE/contracts" "$STAGE/provenance" "$STAGE/support" \
	"$STAGE/acrobot-swingup"
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
			"$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps({"format":"cutie_native_support_three_arm_preflight_summary_v2","status":"runner_engineering_fail","engineering_pass":False,"runner_exit_code":int(sys.argv[2]),"failure":"runner exited before strict aggregation","automatic_controller_training_launched":False},indent=2)+"\n",encoding="utf-8")' "$SUMMARY" "$rc" 2>/dev/null || true
		fi
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
		[[ ! -e "$failed" ]] || failed="${failed}.${RANDOM}"
		mv -- "$STAGE" "$failed"
		echo "CUTIE_NATIVE_SUPPORT_THREE_ARM_FAILED_ARCHIVE=$failed" >&2
	fi
	exit "$rc"
}
trap archive_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "[1/5] Dependency-light native-resolution contracts and source binding"
bash -n "$0"
"$PY" -m tdmpc2.check_cutie_native_highres_contract \
	>"$STAGE/contracts/native_highres_core.log" 2>&1
"$PY" -m tdmpc2.check_cutie_paired_native_support_contract \
	>"$STAGE/contracts/paired_native_support.log" 2>&1
"$PY" -m tdmpc2.check_cutie_native_resolution_tracker_contract \
	>"$STAGE/contracts/native_resolution_tracker.log" 2>&1
"$PY" -m tdmpc2.check_cutie_native_resolution_wrapper_contract \
	>"$STAGE/contracts/native_resolution_wrapper.log" 2>&1
CUDA_VISIBLE_DEVICES="$GPU_B" "$PY" -c 'import json,torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(json.dumps({"logical_cuda_device":0,"device_name":torch.cuda.get_device_name(0)}))' \
	>"$STAGE/contracts/gpu_b.json" 2>"$STAGE/contracts/gpu_b.log"
CUDA_VISIBLE_DEVICES="$GPU_A_C" "$PY" -c 'import json,torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(json.dumps({"logical_cuda_device":0,"device_name":torch.cuda.get_device_name(0)}))' \
	>"$STAGE/contracts/gpu_a_c.json" 2>"$STAGE/contracts/gpu_a_c.log"
"$PY" -c 'import json,sys; from pathlib import Path; left=json.loads(Path(sys.argv[1]).read_text()); right=json.loads(Path(sys.argv[2]).read_text()); assert left["logical_cuda_device"]==right["logical_cuda_device"]==0; assert left["device_name"]==right["device_name"], (left,right)' \
	"$STAGE/contracts/gpu_b.json" "$STAGE/contracts/gpu_a_c.json"
"$PY" -m tdmpc2.common.cutie_external_snapshot \
	--repo-root "$REPO_ROOT" \
	--video-root "$VIDEO_ROOT" \
	--manifest-dir "$MANIFEST_DIR" \
	--oc-repo "$OC_REPO" \
	--cutie-checkpoint "$CUTIE_CKPT" \
	--output "$STAGE/provenance/external_inputs.json" \
	>"$STAGE/contracts/external_inputs.log" 2>&1

IMPLEMENTATION_FILES=(
	"tdmpc2/config.yaml"
	"tdmpc2/common/cutie_external_snapshot.py"
	"tdmpc2/common/cutie_paired_support.py"
	"tdmpc2/check_cutie_object_wrapper_contract.py"
	"tdmpc2/check_cutie_native_highres_contract.py"
	"tdmpc2/check_cutie_native_resolution_tracker_contract.py"
	"tdmpc2/check_cutie_native_resolution_wrapper_contract.py"
	"tdmpc2/check_cutie_paired_native_support_contract.py"
	"tdmpc2/perception/cutie_oc_adapter.py"
	"tdmpc2/tools/collect_cutie_multitask_support.py"
	"tdmpc2/tools/collect_cutie_paired_native_support.py"
	"tdmpc2/tools/evaluate_cutie_native_resolution_tracker.py"
	"tdmpc2/tools/aggregate_cutie_native_resolution_tracker.py"
	"tdmpc2/tools/run_cutie_native_resolution_tracker_preflight.sh"
	"tdmpc2/tools/aggregate_cutie_native_support_three_arm.py"
	"tdmpc2/tools/run_cutie_native_support_three_arm_preflight.sh"
	"tdmpc2/envs/dmcontrol.py"
	"tdmpc2/envs/wrappers/cutie_object.py"
	"tdmpc2/envs/wrappers/foreground_stress.py"
	"tdmpc2/envs/wrappers/tensor.py"
	"tdmpc2/envs/wrappers/video_background.py"
)
for path in "${IMPLEMENTATION_FILES[@]}"; do
	[[ -f "$path" ]] || { echo "Missing implementation input: $path" >&2; exit 2; }
done
"$PY" -c 'import hashlib,json,sys; from pathlib import Path; root=Path(sys.argv[1]).resolve(); out=Path(sys.argv[2]); rows=[]
for raw in sys.argv[3:]:
 p=(root/raw).resolve(); p.relative_to(root); data=p.read_bytes(); rows.append({"path":raw,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()})
out.write_text(json.dumps({"format":"cutie_native_support_three_arm_implementation_v2","files":rows},sort_keys=True,indent=2)+"\n",encoding="utf-8")' \
	"$REPO_ROOT" "$STAGE/provenance/implementation_files.json" "${IMPLEMENTATION_FILES[@]}"

TASK_ARGS=(acrobot-swingup)
if (( RUN_CARTPOLE )); then
	mkdir -p "$STAGE/cartpole-swingup"
	TASK_ARGS+=(cartpole-swingup)
fi

collect_support() {
	local task=$1 output log
	output="$STAGE/support/$task"
	log="$STAGE/support/${task}.collect.log"
	echo "SUPPORT_START task=$task output=$output" | tee "$log"
	CUDA_VISIBLE_DEVICES="$GPU_A_C" "$PY" -m \
		tdmpc2.tools.collect_cutie_paired_native_support \
		--task "$task" \
		--output "$output" \
		--video-root "$VIDEO_ROOT" \
		--manifest-dir "$MANIFEST_DIR" \
		--seed "$SUPPORT_SEED" \
		--background-seed "$SUPPORT_BACKGROUND_SEED" \
		--action-seed "$SUPPORT_ACTION_SEED" \
		--max-reset-attempts "$MAX_SUPPORT_RESET_ATTEMPTS" >>"$log" 2>&1
	echo "SUPPORT_END task=$task rc=0" | tee -a "$log"
}

run_tracked() {
	local pid rc
	"$@" & pid=$!
	ACTIVE_PIDS=("$pid")
	set +e
	wait "$pid"; rc=$?
	set -e
	ACTIVE_PIDS=()
	return "$rc"
}

echo "[2/5] Same-state paired native64/native128 support collection"
for task in "${TASK_ARGS[@]}"; do
	run_tracked collect_support "$task"
done

arm_pair() {
	case "$1" in
		A) printf '64 64' ;;
		B) printf '128 64' ;;
		C) printf '128 128' ;;
		*) return 2 ;;
	esac
}

arm_slug() {
	case "$1" in
		A) printf 'arm_a_runtime64_support64' ;;
		B) printf 'arm_b_runtime128_support64' ;;
		C) printf 'arm_c_runtime128_support128' ;;
		*) return 2 ;;
	esac
}

run_one() {
	local task=$1 arm=$2 gpu=$3 runtime support slug task_dir annotations output log rc
	# ``read`` reports EOF as failure when the producer has no newline, which
	# would trip ``set -e`` even after assigning both values. Keep arm_pair's
	# compact static contract and append the record delimiter here.
	read -r runtime support < <(arm_pair "$arm"; printf '\n')
	slug="$(arm_slug "$arm")"
	task_dir="$STAGE/$task"
	annotations="$STAGE/support/$task/annotations.json"
	output="$task_dir/${slug}.json"
	log="$task_dir/${slug}.log"
	mkdir -p "$task_dir"
	echo "TRACKER_START task=$task arm=$arm runtime=$runtime support=$support gpu=$gpu" | tee "$log"
	set +e
	CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m \
		tdmpc2.tools.evaluate_cutie_native_resolution_tracker \
		--task "$task" \
		--resolution "$runtime" \
		--support-resolution "$support" \
		--oc-storm-repo "$OC_REPO" \
		--cutie-checkpoint "$CUTIE_CKPT" \
		--support-annotations "$annotations" \
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
	printf '%s\n' "$rc" >"$task_dir/${slug}.rc"
	echo "TRACKER_END task=$task arm=$arm runtime=$runtime support=$support gpu=$gpu rc=$rc" | tee -a "$log"
	return "$rc"
}

echo "[3/5] Arm A runtime64/support64 controls on the A/C GPU"
for task in "${TASK_ARGS[@]}"; do
	run_tracked run_one "$task" A "$GPU_A_C"
done

run_bc_pair() {
	local task=$1 pid_b pid_c rc_b rc_c
	run_one "$task" B "$GPU_B" & pid_b=$!; ACTIVE_PIDS+=("$pid_b")
	run_one "$task" C "$GPU_A_C" & pid_c=$!; ACTIVE_PIDS+=("$pid_c")
	set +e
	wait "$pid_b"; rc_b=$?
	wait "$pid_c"; rc_c=$?
	set -e
	ACTIVE_PIDS=()
	if (( rc_b != 0 || rc_c != 0 )); then
		echo "$task B/C paired jobs failed: B=$rc_b C=$rc_c" >&2
		return 4
	fi
}

echo "[4/5] Arms B/C paired at fixed runtime128 on two matched GPUs"
for task in "${TASK_ARGS[@]}"; do
	run_bc_pair "$task"
done

echo "[5/5] Strict support, trajectory, quality, health, and latency aggregation"
set +e
"$PY" -m tdmpc2.tools.aggregate_cutie_native_support_three_arm \
	--input-root "$STAGE" \
	--tasks "${TASK_ARGS[@]}" \
	--expected-gpu-b "$GPU_B" \
	--expected-gpu-a-c "$GPU_A_C" \
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
echo "CUTIE_NATIVE_SUPPORT_THREE_ARM_PREFLIGHT_COMPLETE"
echo "SUMMARY=$BASE/three_arm_summary.json"
echo "No controller training was launched. Read recommendation before any training."
