#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
CURRENT_SNAPSHOT="${CURRENT_SNAPSHOT:-<DATA_PATH>/tdmpc2_runs/object_state_supervision_v1_20260909_233824}"
CURRENT_PATTERN="$CURRENT_SNAPSHOT/tdmpc2/tools/run_object_state_supervision_paired.sh"
WAIT_SECONDS="${WAIT_SECONDS:-18000}"
POLL_SECONDS="${POLL_SECONDS:-30}"
RUN_TAG="${RUN_TAG:-object_state_supervision_acrobot_500k_v1_20260910_codex1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/logs/_diagnostic/$RUN_TAG}"

[[ "$WAIT_SECONDS" =~ ^[0-9]+$ && "$POLL_SECONDS" =~ ^[0-9]+$ ]] || {
    echo 'Wait durations must be nonnegative integers.' >&2
    exit 2
}
deadline=$(( $(date +%s) + WAIT_SECONDS ))
echo "WAITING_FOR_CURRENT_RUN pattern=$CURRENT_PATTERN deadline_epoch=$deadline"
gpu_locks_available() (
    exec 201>"/tmp/tdmpc2_object_state_supervision_gpu_0.lock"
    exec 202>"/tmp/tdmpc2_object_state_supervision_gpu_1.lock"
    flock -n 201 && flock -n 202
)
while pgrep -f -- "$CURRENT_PATTERN" >/dev/null || ! gpu_locks_available; do
    if (( $(date +%s) >= deadline )); then
        echo 'CURRENT_RUN_WAIT_TIMEOUT' >&2
        exit 124
    fi
    sleep "$POLL_SECONDS"
done
echo 'CURRENT_RUN_RELEASED_GPUS'

cd "$REPO_ROOT"
echo "ACROBOT_500K_DISPATCH output=$OUTPUT_ROOT"
export OUTPUT_ROOT RUN_TAG
export TASKS_CSV=acrobot-swingup
export STEPS=500000 EVAL_FREQ=25000 EVAL_EPISODES=10
export SEED=12 GPU0=0 GPU1=1 COMPILE=true
export SAVE_EVAL_CHECKPOINTS=true PRIVILEGED_STATE_SCORE=true
exec bash tdmpc2/tools/run_object_state_supervision_paired.sh
