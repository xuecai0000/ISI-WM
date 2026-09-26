#!/usr/bin/env bash
# Start only the explicitly named, independently copied experiment workspace.
set -Eeuo pipefail
ROOT=${1:?Absolute independent workspace required}
TAG=${2:?Unique run tag required}
[[ "$ROOT" == <DATA_PATH>/tdmpc2_runs/* && -d "$ROOT/tdmpc2" ]]
[[ "$TAG" =~ ^object_state_supervision_paired_v1_[a-zA-Z0-9_]+$ ]]
cd "$ROOT"
mkdir -p logs/_diagnostic
BASE="$ROOT/logs/_diagnostic/$TAG"
LOG="$BASE.runner.log"
[[ ! -e "$LOG" && ! -e "$BASE" && ! -e "$BASE.incomplete" ]]
nohup env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
  RUN_TAG="$TAG" OUTPUT_ROOT="$BASE" \
  bash -c 'bash tdmpc2/tools/run_object_state_supervision_paired.sh; rc=$?; printf "RUNNER_RC=%s\n" "$rc"; exit "$rc"' \
  > "$LOG" 2>&1 < /dev/null &
RUN_PID=$!
printf '%s\n' "$RUN_PID" > "$BASE.pid"
printf 'PID=%s\nOUTPUT_ROOT=%s\nRUNLOG=%s\n' "$RUN_PID" "$BASE" "$LOG"
