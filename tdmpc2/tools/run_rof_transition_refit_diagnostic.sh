#!/usr/bin/env bash
set -euo pipefail

# Offline diagnostic only: this script never launches environment interaction or
# controller training.  It refits transition models on immutable, frozen policy
# latents collected by the causal-ladder diagnostic.

SOURCE_ROOT="${SOURCE_ROOT:-/root/tdmpc2_runs/rof_causal_ladder_source_v1_20260913_155806}"
DATA_ROOT="${DATA_ROOT:-/root/tdmpc2_runs/rof_causal_ladder_v1_20260913_155806}"
MODEL_ROOT="${MODEL_ROOT:-/root/tdmpc2_runs/rof_wm_v1_diagnostics_30k_v1_20260913_1315/workspaces}"
PYTHON_BIN="${PYTHON_BIN:-/root/conda_envs/tdmpc2_2026/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new destination path.}"
INCOMPLETE="${OUTPUT_ROOT}.incomplete"

if [[ -e "$OUTPUT_ROOT" || -e "$INCOMPLETE" ]]; then
	echo "Refusing to overwrite an existing diagnostic root: $OUTPUT_ROOT" >&2
	exit 2
fi

mkdir -p "$INCOMPLETE/logs" "$INCOMPLETE/results"

tasks=(
	"walker-run"
	"reacher-easy"
	"finger-turn-easy"
	"cartpole-balance-sparse"
)
gpus=(3 4 5 6)
experiments=(
	"rof_v1_temporal_walker-run_seed6"
	"rof_v1_temporal_reacher-easy_seed6"
	"rof_v1_target_finger-turn-easy_seed6"
	"rof_v1_geometry_aux_cartpole-balance-sparse_seed6"
)

cd "$SOURCE_ROOT"
"$PYTHON_BIN" -m py_compile \
	tdmpc2/tools/evaluate_rof_transition_refit.py \
	tdmpc2/check_rof_transition_refit_contract.py
"$PYTHON_BIN" -m tdmpc2.check_rof_transition_refit_contract \
	>"$INCOMPLETE/logs/contracts.log" 2>&1

declare -a pids=()
for index in "${!tasks[@]}"; do
	task="${tasks[$index]}"
	gpu="${gpus[$index]}"
	experiment="${experiments[$index]}"
	run_root="$MODEL_ROOT/$experiment/logs/$task/6/$experiment"
	clean="$DATA_ROOT/full_clean/datasets/$task/dataset_manifest.json"
	hard="$DATA_ROOT/full_hard/datasets/$task/dataset_manifest.json"
	runtime="$run_root/runtime_config.json"
	checkpoint="$run_root/models/final.pt"
	for required in "$clean" "$hard" "$runtime" "$checkpoint"; do
		[[ -f "$required" ]] || { echo "Missing immutable input: $required" >&2; exit 3; }
	done
	echo "REFIT_START task=$task gpu=$gpu"
	(
		export CUDA_VISIBLE_DEVICES="$gpu"
		export CUBLAS_WORKSPACE_CONFIG=:4096:8
		timeout --foreground --signal=TERM --kill-after=30s 7200 \
			"$PYTHON_BIN" -B -m tdmpc2.tools.evaluate_rof_transition_refit \
			--clean-dataset "$clean" \
			--hard-dataset "$hard" \
			--runtime-config "$runtime" \
			--checkpoint "$checkpoint" \
			--output "$INCOMPLETE/results/$task.json" \
			--seed 20260913 \
			--bootstrap-seed 314159 \
			--bootstrap-resamples 20000 \
			--encoder-batch-size 256 \
			--fit-batch-size 1024 \
			--max-epochs 80 \
			--patience 10 \
			--learning-rate 3e-4 \
			--weight-decay 1e-5
	) >"$INCOMPLETE/logs/$task.log" 2>&1 &
	pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
	task="${tasks[$index]}"
	gpu="${gpus[$index]}"
	if wait "${pids[$index]}"; then
		echo "REFIT_END task=$task gpu=$gpu rc=0"
	else
		rc=$?
		echo "REFIT_END task=$task gpu=$gpu rc=$rc" >&2
		failed=1
	fi
done

if [[ "$failed" -ne 0 ]]; then
	echo "ROF_TRANSITION_REFIT_FAILED_ROOT=$INCOMPLETE" >&2
	exit 1
fi

"$PYTHON_BIN" - "$INCOMPLETE" "${tasks[@]}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

from tdmpc2.tools.evaluate_rof_transition_refit import validate_result

root = Path(sys.argv[1]).resolve()
tasks = sys.argv[2:]
entries = {}
for task in tasks:
    path = root / 'results' / f'{task}.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    validate_result(payload)
    if payload['task'] != task:
        raise ValueError(f'Task mismatch in {path}')
    entries[task] = {
        'path': str(path.relative_to(root)),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'attribution': payload['attribution'],
    }
summary = {
    'format': 'rof_transition_refit_campaign_v1',
    'status': 'rof_transition_refit_campaign_complete',
    'engineering_pass': True,
    'scientific_complete': True,
    'controller_training_authorized': False,
    'policy_training_performed': False,
    'tasks': tasks,
    'entries': entries,
}
target = root / 'rof_transition_refit_summary.json'
temporary = root / f'.{target.name}.{os.getpid()}.tmp'
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
os.replace(temporary, target)
PY

mv "$INCOMPLETE" "$OUTPUT_ROOT"
echo "ROF_TRANSITION_REFIT_CAMPAIGN_COMPLETE"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "SUMMARY=$OUTPUT_ROOT/rof_transition_refit_summary.json"
