#!/usr/bin/env bash
# Fixed seed-6, 100k Acrobot privileged articulated-pose oracle.
#
# This runner waits for (and never signals) the named clean ObjectOnly source,
# validates and hashes both completed comparator runs, executes contracts and a
# real DMControl smoke, waits briefly for an idle selected GPU, and only then
# starts the single oracle training child.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python}"
GPU_ORACLE="${GPU_ORACLE:-0}"
SOURCE_OBJECT_ROOT="${SOURCE_OBJECT_ROOT:-/home/<USER>/world/tdmpc2_2026/logs/acrobot-swingup/6/cutie_object_only_acrobot_clean100k_seed6_20260831_154217}"
SOURCE_STATE_ROOT="${SOURCE_STATE_ROOT:-/home/<USER>/world/tdmpc2_2026/logs/acrobot-swingup/6/tdmpc2_state_acrobot_clean100k_seed6_20260831_154217}"

readonly TASK=acrobot-swingup
readonly SEED=6
readonly STEPS=100000
readonly EVAL_FREQ=20000
readonly EVAL_EPISODES=10
readonly SOURCE_OBJECT_EXP=cutie_object_only_acrobot_clean100k_seed6_20260831_154217
readonly SOURCE_STATE_EXP=tdmpc2_state_acrobot_clean100k_seed6_20260831_154217
readonly RUN_TAG=acrobot_gt_pose_oracle_clean100k_v1_seed6
readonly SOURCE_POLL_SECONDS=10
readonly SOURCE_WAIT_TIMEOUT_SECONDS=7200
readonly GPU_POLL_SECONDS=5
readonly GPU_WAIT_TIMEOUT_SECONDS=120
readonly BASE="$REPO_ROOT/logs/_diagnostic/$RUN_TAG"
readonly STAGE="${BASE}.incomplete"
readonly ORACLE_ROOT="$REPO_ROOT/logs/$TASK/$SEED/$RUN_TAG"
readonly SUMMARY="$STAGE/acrobot_gt_pose_oracle_summary.json"
readonly INPUTS="$STAGE/provenance/inputs.json"

ACTIVE_PIDS=()
PROMOTED=0
STAGE_OWNED=0
ORACLE_ROOT_OWNED=0
TEMP_ROOT=""

terminate_tree() {
	local parent=$1 child attempt
	while IFS= read -r child; do
		[[ -n "$child" ]] && terminate_tree "$child"
	done < <(pgrep -P "$parent" 2>/dev/null || true)
	kill -TERM "$parent" 2>/dev/null || true
	for attempt in {1..20}; do
		kill -0 "$parent" 2>/dev/null || return 0
		sleep 0.25
	done
	kill -KILL "$parent" 2>/dev/null || true
}

archive_on_exit() {
	local rc=$? pid failed relocation_state=absent destination=training_root/run
	trap - EXIT INT TERM
	for pid in "${ACTIVE_PIDS[@]:-}"; do
		[[ -n "$pid" ]] && terminate_tree "$pid"
	done
	for pid in "${ACTIVE_PIDS[@]:-}"; do
		[[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
	done
	ACTIVE_PIDS=()
	if [[ -n "$TEMP_ROOT" && -d "$TEMP_ROOT" ]]; then
		rm -f -- "$TEMP_ROOT/inputs.json" "$TEMP_ROOT/inputs.json.incomplete"
		rmdir -- "$TEMP_ROOT" 2>/dev/null || true
	fi
	if (( PROMOTED == 0 && STAGE_OWNED == 1 )) && [[ -d "$STAGE" ]]; then
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
		while [[ -e "$failed" ]]; do
			failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$.${RANDOM}"
		done
		if (( ORACLE_ROOT_OWNED == 1 )) && [[ -e "$ORACLE_ROOT" ]]; then
			mkdir -p "$STAGE/training_root"
			if [[ -e "$STAGE/$destination" ]]; then
				relocation_state=move_refused_destination_exists
			elif mv -- "$ORACLE_ROOT" "$STAGE/$destination"; then
				relocation_state=moved
			else
				relocation_state=move_failed
			fi
		elif [[ -e "$ORACLE_ROOT" ]]; then
			relocation_state=present_not_owned_not_touched
		fi
		"$PY" - "$SUMMARY" "$STAGE/provenance/failure_archive.json" \
			"$ORACLE_ROOT" "$destination" "$relocation_state" "$failed" "$rc" <<'PY' || true
import json
import os
import sys
from pathlib import Path

summary_path, relocation_path = map(Path, sys.argv[1:3])
original_root, relative, state, failed_root = sys.argv[3:7]
runner_rc = int(sys.argv[7])
relocation = {
    'format': 'acrobot_gt_pose_oracle_failure_archive_v1',
    'failed_archive_root': str(Path(failed_root).resolve()),
    'oracle_training_root': {
        'execution_time_path': original_root,
        'archived_relative_to_summary_root': relative if state == 'moved' else None,
        'state': state,
    },
    'runner_exit_code': runner_rc,
}
relocation_path.parent.mkdir(parents=True, exist_ok=True)
temporary = relocation_path.with_name(relocation_path.name + '.incomplete')
temporary.write_text(
    json.dumps(relocation, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
    encoding='utf-8', newline='\n',
)
os.replace(temporary, relocation_path)
if summary_path.is_file():
    try:
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
    except Exception:
        summary = {}
else:
    summary = {}
summary.setdefault('format', 'acrobot_gt_pose_oracle_clean100k_v1')
old_engineering = summary.get('engineering')
if (
    summary.get('status') == 'acrobot_gt_pose_oracle_engineering_pass'
    or (isinstance(old_engineering, dict) and old_engineering.get('status') == 'pass')
):
    summary['pre_failure_aggregate'] = {
        'status': summary.get('status'),
        'engineering': summary.get('engineering'),
        'scientific': summary.get('scientific'),
    }
summary['status'] = 'acrobot_gt_pose_oracle_engineering_fail'
failures = old_engineering.get('failures') if isinstance(old_engineering, dict) else []
if not isinstance(failures, list):
    failures = []
summary['engineering'] = {'status': 'fail', 'gates': {}, 'failures': failures}
failures.append(f'runner exited with code {runner_rc}')
summary['scientific'] = {
    'status': 'not_evaluated_engineering_failure', 'engineering_gate': False,
}
summary['runner_exit_code'] = runner_rc
summary['failure_archive'] = {
    'root': str(Path(failed_root).resolve()),
    'manifest_relative_to_summary_root': 'provenance/failure_archive.json',
}
replacement = summary_path.with_name(summary_path.name + '.failure-update')
replacement.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
    encoding='utf-8', newline='\n',
)
os.replace(replacement, summary_path)
PY
		mv -- "$STAGE" "$failed"
		echo "ACROBOT_GT_POSE_ORACLE_FAILED_ARCHIVE=$failed" >&2
	fi
	exit "$rc"
}

trap archive_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_owned() {
	local pid rc
	"$@" &
	pid=$!
	ACTIVE_PIDS=("$pid")
	if wait "$pid"; then rc=0; else rc=$?; fi
	ACTIVE_PIDS=()
	return "$rc"
}

if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
	echo "Bash >=5.1 is required." >&2
	exit 2
fi
if [[ "$PY" == */* ]]; then
	[[ -x "$PY" ]] || { echo "Python is not executable: $PY" >&2; exit 2; }
else
	command -v "$PY" >/dev/null || { echo "Python is not on PATH: $PY" >&2; exit 2; }
fi
[[ -d /proc && -r /proc/self/cmdline ]] || {
	echo "Readable Linux /proc argv is required for the source wait." >&2
	exit 2
}

echo "[1/8] Waiting for the exact clean ObjectOnly training argv (never signaling it)"
WAIT_STARTED_EPOCH="$(date +%s)"
WAIT_DEADLINE=$(( WAIT_STARTED_EPOCH + SOURCE_WAIT_TIMEOUT_SECONDS ))
WAIT_POLLS_WITH_MATCH=0
while :; do
	SOURCE_MATCHES="$("$PY" - "$SOURCE_OBJECT_EXP" <<'PY'
import os
import sys
from pathlib import Path

target = f'exp_name={sys.argv[1]}'
matches = []
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():
        continue
    try:
        raw = (entry / 'cmdline').read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    argv = [os.fsdecode(value) for value in raw.split(b'\0') if value]
    is_train = any(
        value == 'tdmpc2/train.py' or value.endswith('/tdmpc2/train.py')
        for value in argv
    )
    if is_train and target in argv:
        matches.append(int(entry.name))
for pid in sorted(matches):
    print(pid)
PY
)"
	NOW="$(date +%s)"
	if [[ -z "$SOURCE_MATCHES" ]]; then
		WAIT_ENDED_EPOCH="$NOW"
		break
	fi
	WAIT_POLLS_WITH_MATCH=$(( WAIT_POLLS_WITH_MATCH + 1 ))
	echo "SOURCE_STILL_RUNNING exact_exp_argv=$SOURCE_OBJECT_EXP pids=${SOURCE_MATCHES//$'\n'/,}"
	if (( NOW >= WAIT_DEADLINE )); then
		echo "Timed out after ${SOURCE_WAIT_TIMEOUT_SECONDS}s; source was not signaled." >&2
		exit 5
	fi
	WAIT_REMAINING=$(( WAIT_DEADLINE - NOW ))
	WAIT_SLEEP=$SOURCE_POLL_SECONDS
	(( WAIT_REMAINING < WAIT_SLEEP )) && WAIT_SLEEP=$WAIT_REMAINING
	run_owned sleep "$WAIT_SLEEP"
done
echo "SOURCE_EXIT_CONFIRMED elapsed_seconds=$(( WAIT_ENDED_EPOCH - WAIT_STARTED_EPOCH ))"

# Source validation happens before any diagnostic or training output directory
# is created.  An incomplete comparator therefore cannot start the oracle.
TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/acrobot_gt_pose_oracle.XXXXXX")"
BOUND_INPUTS="$TEMP_ROOT/inputs.json"
echo "[2/8] Validating complete clean ObjectOnly/state sources and binding hashes"
run_owned "$PY" -B -m tdmpc2.tools.aggregate_acrobot_gt_pose_oracle bind \
	--repo "$REPO_ROOT" \
	--source-object-root "$SOURCE_OBJECT_ROOT" \
	--source-state-root "$SOURCE_STATE_ROOT" \
	--wait-started-epoch "$WAIT_STARTED_EPOCH" \
	--wait-ended-epoch "$WAIT_ENDED_EPOCH" \
	--wait-polls-with-match "$WAIT_POLLS_WITH_MATCH" \
	--output "$BOUND_INPUTS"

[[ "$GPU_ORACLE" =~ ^(0|[1-9][0-9]*)$ ]] || {
	echo "GPU_ORACLE must be a non-negative physical GPU index: $GPU_ORACLE" >&2
	exit 2
}
for command_name in bash pgrep nvidia-smi; do
	command -v "$command_name" >/dev/null || {
		echo "Required command is unavailable: $command_name" >&2
		exit 2
	}
done
for path in "$BASE" "$STAGE" "$ORACLE_ROOT"; do
	[[ ! -e "$path" ]] || {
		echo "Refusing to overwrite existing output: $path" >&2
		exit 3
	}
done
nvidia-smi --id="$GPU_ORACLE" --query-gpu=index \
	--format=csv,noheader,nounits >/dev/null

mkdir -p "$(dirname -- "$STAGE")"
trap '' INT TERM
if ! mkdir -- "$STAGE"; then
	trap 'exit 130' INT
	trap 'exit 143' TERM
	echo "Atomic staging-root reservation failed; refusing to overwrite: $STAGE" >&2
	exit 3
fi
STAGE_OWNED=1
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -- "$STAGE/contracts" "$STAGE/provenance" "$STAGE/smoke" "$STAGE/training"
mv -- "$BOUND_INPUTS" "$INPUTS"
rmdir -- "$TEMP_ROOT"
TEMP_ROOT=""
INPUTS_SHA256="$("$PY" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$INPUTS")"
[[ "$INPUTS_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
	echo "Failed to hash bound inputs." >&2
	exit 4
}

run_contract() {
	local label=$1
	shift
	echo "CONTRACT_START label=$label"
	if run_owned "$@" >>"$STAGE/contracts/${label}.log" 2>&1; then
		printf '0\n' >"$STAGE/contracts/${label}.rc"
		echo "CONTRACT_END label=$label rc=0"
	else
		local rc=$?
		printf '%s\n' "$rc" >"$STAGE/contracts/${label}.rc"
		echo "CONTRACT_END label=$label rc=$rc" >&2
		return "$rc"
	fi
}

echo "[3/8] Static and dependency-light contracts"
run_contract shell_syntax bash -n "$0"
run_contract python_compile "$PY" -B -m py_compile \
	tdmpc2/common/gt_articulated_pose.py \
	tdmpc2/envs/wrappers/gt_articulated_pose.py \
	tdmpc2/check_gt_articulated_pose_contract.py \
	tdmpc2/check_gt_articulated_pose_update.py \
	tdmpc2/tools/aggregate_acrobot_gt_pose_oracle.py
run_contract gt_articulated_pose "$PY" -B tdmpc2/check_gt_articulated_pose_contract.py
run_contract object_only "$PY" -B tdmpc2/check_cutie_object_only_contract.py
run_contract object_only_integration "$PY" -B tdmpc2/check_cutie_object_only_integration_contract.py

echo "[4/8] Actual DMControl environment smoke"
run_contract gt_articulated_pose_real_env env MUJOCO_GL=egl \
	"$PY" -B tdmpc2/check_gt_articulated_pose_contract.py --real-env-smoke

echo "[5/8] Rechecking immutable source and implementation hashes"
run_owned "$PY" -B -m tdmpc2.tools.aggregate_acrobot_gt_pose_oracle verify \
	--inputs "$INPUTS" --expected-inputs-sha256 "$INPUTS_SHA256" \
	>"$STAGE/provenance/immutable_pretraining.log" 2>&1

query_gpu_compute_pids() {
	local output line trimmed compact
	if ! output="$(nvidia-smi --id="$GPU_ORACLE" --query-compute-apps=pid \
		--format=csv,noheader,nounits 2>&1)"; then
		echo "nvidia-smi compute query failed: $output" >&2
		return 2
	fi
	output="${output//$'\r'/}"
	while IFS= read -r line; do
		trimmed="${line#"${line%%[![:space:]]*}"}"
		trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
		[[ -z "$trimmed" ]] && continue
		case "$trimmed" in
			"No running processes found"|"No running compute processes found") continue ;;
		esac
		compact="${trimmed//[[:space:]]/}"
		[[ "$compact" =~ ^[0-9]+$ ]] || {
			echo "Unexpected nvidia-smi compute row: $line" >&2
			return 2
		}
		printf '%s\n' "$compact"
	done <<<"$output"
}

wait_for_gpu_idle() {
	local label=$1 log="$STAGE/provenance/gpu_idle_wait_${1}.log"
	local started deadline now remaining sleep_seconds compute_pids
	started="$(date +%s)"
	deadline=$(( started + GPU_WAIT_TIMEOUT_SECONDS ))
	while :; do
		if ! compute_pids="$(query_gpu_compute_pids)"; then
			return 2
		fi
		now="$(date +%s)"
		if [[ -z "$compute_pids" ]]; then
			printf 'label=%s gpu=%s idle_epoch_seconds=%s waited_seconds=%s\n' \
				"$label" "$GPU_ORACLE" "$now" "$(( now - started ))" >>"$log"
			return 0
		fi
		printf 'label=%s gpu=%s busy_epoch_seconds=%s compute_pids=%s\n' \
			"$label" "$GPU_ORACLE" "$now" "${compute_pids//$'\n'/,}" >>"$log"
		if (( now >= deadline )); then
			echo "GPU_ORACLE remained busy for ${GPU_WAIT_TIMEOUT_SECONDS}s during $label; no process was signaled." >&2
			return 5
		fi
		remaining=$(( deadline - now ))
		sleep_seconds=$GPU_POLL_SECONDS
		(( remaining < sleep_seconds )) && sleep_seconds=$remaining
		run_owned sleep "$sleep_seconds"
	done
}

echo "[6/8] CUDA update contracts and final idle-GPU acquisition"
echo "Waiting at most ${GPU_WAIT_TIMEOUT_SECONDS}s for GPU_ORACLE=$GPU_ORACLE before contracts"
wait_for_gpu_idle precontract
run_contract gt_articulated_pose_update_eager env \
	CUDA_VISIBLE_DEVICES="$GPU_ORACLE" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
	"$PY" -B tdmpc2/check_gt_articulated_pose_update.py
run_contract gt_articulated_pose_update_compile env \
	CUDA_VISIBLE_DEVICES="$GPU_ORACLE" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
	"$PY" -B tdmpc2/check_gt_articulated_pose_update.py --compile
run_owned "$PY" -B -m tdmpc2.tools.aggregate_acrobot_gt_pose_oracle verify \
	--inputs "$INPUTS" --expected-inputs-sha256 "$INPUTS_SHA256" \
	>"$STAGE/provenance/immutable_after_cuda_contracts.log" 2>&1
echo "Waiting at most ${GPU_WAIT_TIMEOUT_SECONDS}s for the final pretraining idle check"
wait_for_gpu_idle pretraining

TRAIN_ARGS=(
	"task=$TASK"
	obs=state
	model_size=5
	"steps=$STEPS"
	"seed=$SEED"
	"eval_freq=$EVAL_FREQ"
	"eval_episodes=$EVAL_EPISODES"
	video_background_enabled=false
	video_background_root=null
	video_background_manifest_dir=null
	visual_foreground_erosion_pixels=0
	compile=true
	compile_fallback_random=true
	enable_wandb=false
	wandb_project=none
	wandb_entity=none
	save_csv=true
	save_video=false
	save_agent=true
	checkpoint=null
	data_dir=null
	obs_shapes=null
	action_dims=null
	episode_lengths=null
	flat_anchor=true
	flat_anchor_mode=cutie_object_only
	flat_anchor_loss_beta=0.1
	cutie_object_observation_variant=gt_articulated_pose
	cutie_object_frame_schema=acrobot_gt_articulated_pose_v1
	cutie_object_role_names=[upper_arm,lower_arm]
	cutie_object_allow_simulator_kinematics_runtime=true
	cutie_object_allow_simulator_runtime=false
	cutie_object_allow_simulator_support=false
	cutie_object_repo=null
	cutie_object_checkpoint=null
	cutie_object_support_path=null
	cutie_object_config_dir=null
	cutie_object_num_roles=2
	cutie_object_frame_dim=7
	cutie_object_stack_frames=3
	cutie_object_input_dim=21
	cutie_object_auxiliary_target=full_descriptor
	cutie_object_role_dim=64
	cutie_object_hidden_dim=256
	cutie_object_only_latent_dim=128
	cutie_object_native_highres_enabled=false
	cutie_object_last_valid_memory=false
	cutie_object_policy_burst_plan=null
	cutie_object_belief_enabled=false
	cutie_object_belief_use_for_control=false
	"exp_name=$RUN_TAG"
	"hydra.run.dir=$STAGE/hydra"
	hydra.job.chdir=false
)

echo "[7/8] Training fixed clean Acrobot GT articulated-pose oracle"
mkdir -p "$(dirname -- "$ORACLE_ROOT")"
trap '' INT TERM
if ! mkdir -- "$ORACLE_ROOT"; then
	trap 'exit 130' INT
	trap 'exit 143' TERM
	echo "Atomic oracle-root reservation failed; refusing to overwrite: $ORACLE_ROOT" >&2
	exit 3
fi
ORACLE_ROOT_OWNED=1
trap 'exit 130' INT
trap 'exit 143' TERM
echo "TRAIN_START gpu=$GPU_ORACLE root=$ORACLE_ROOT"
if run_owned env CUDA_VISIBLE_DEVICES="$GPU_ORACLE" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
	"$PY" tdmpc2/train.py "${TRAIN_ARGS[@]}" \
	>"$STAGE/training/oracle.log" 2>&1; then
	TRAIN_RC=0
else
	TRAIN_RC=$?
fi
printf '%s\n' "$TRAIN_RC" >"$STAGE/training/oracle.rc"
echo "TRAIN_END gpu=$GPU_ORACLE rc=$TRAIN_RC"
(( TRAIN_RC == 0 )) || exit "$TRAIN_RC"

echo "[8/8] Immutable recheck, strict three-curve aggregation, and promotion"
run_owned "$PY" -B -m tdmpc2.tools.aggregate_acrobot_gt_pose_oracle verify \
	--inputs "$INPUTS" --expected-inputs-sha256 "$INPUTS_SHA256" \
	>"$STAGE/provenance/immutable_posttraining.log" 2>&1
run_owned "$PY" -B -m tdmpc2.tools.aggregate_acrobot_gt_pose_oracle aggregate \
	--inputs "$INPUTS" --expected-inputs-sha256 "$INPUTS_SHA256" \
	--oracle-root "$ORACLE_ROOT" --output "$SUMMARY" \
	>"$STAGE/aggregate.log" 2>&1

[[ ! -e "$BASE" ]] || { echo "Promotion target appeared during run: $BASE" >&2; exit 3; }
mv -- "$STAGE" "$BASE"
PROMOTED=1
trap - EXIT INT TERM
echo "ACROBOT_GT_POSE_ORACLE_COMPLETE"
echo "ORACLE_ROOT=$ORACLE_ROOT"
echo "SUMMARY=$BASE/acrobot_gt_pose_oracle_summary.json"
