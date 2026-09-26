#!/usr/bin/env bash
# Fast matched development run: official RGB versus live CutieHybrid.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export STEPS=20000
export EVAL_FREQ=5000
export EVAL_EPISODES=3
export MAX_WALLCLOCK_SECONDS="${PAIR_MAX_WALLCLOCK_SECONDS:-5400}"
export RUN_TAG="${PAIR_RUN_TAG:-cutie_hybrid_20k_pair_v1}"
export SUMMARY_NAME=paired_summary.json
export RUN_SCOPE=paired_20k_development

exec bash "$SCRIPT_DIR/run_cutie_hybrid_smoke.sh"
