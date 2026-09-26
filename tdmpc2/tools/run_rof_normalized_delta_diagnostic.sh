#!/usr/bin/env bash
set -euo pipefail

# Frozen-latent development diagnostic only. This script never launches an
# environment, policy optimization, controller training, or privileged labels.

SOURCE_ROOT="${SOURCE_ROOT:-/root/tdmpc2_runs/rof_causal_ladder_source_v1_20260913_155806}"
DATA_ROOT="${DATA_ROOT:-/root/tdmpc2_runs/rof_causal_ladder_v1_20260913_155806}"
MODEL_ROOT="${MODEL_ROOT:-/root/tdmpc2_runs/rof_wm_v1_diagnostics_30k_v1_20260913_1315/workspaces}"
PREVIOUS_REFIT_ROOT="${PREVIOUS_REFIT_ROOT:-/root/tdmpc2_runs/rof_transition_refit_v1_20260913_181134/results}"
PYTHON_BIN="${PYTHON_BIN:-/root/conda_envs/tdmpc2_2026/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new destination path.}"
INCOMPLETE="${OUTPUT_ROOT}.incomplete"

if [[ -e "$OUTPUT_ROOT" || -e "$INCOMPLETE" ]]; then
	echo "Refusing to overwrite an existing diagnostic root: $OUTPUT_ROOT" >&2
	exit 2
fi

mkdir -p "$INCOMPLETE/logs" "$INCOMPLETE/results" "$INCOMPLETE/provenance"

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
fit_seeds=(20260913 20260917 20260923)

cd "$SOURCE_ROOT"
"$PYTHON_BIN" -m py_compile \
	tdmpc2/tools/evaluate_rof_transition_refit.py \
	tdmpc2/tools/evaluate_rof_normalized_delta.py \
	tdmpc2/check_rof_transition_refit_contract.py \
	tdmpc2/check_rof_normalized_delta_contract.py
"$PYTHON_BIN" -B -m tdmpc2.check_rof_transition_refit_contract \
	>"$INCOMPLETE/logs/transition_contracts.log" 2>&1
"$PYTHON_BIN" -B -m tdmpc2.check_rof_normalized_delta_contract \
	>"$INCOMPLETE/logs/normalized_delta_contracts.log" 2>&1
sha256sum \
	tdmpc2/tools/evaluate_rof_transition_refit.py \
	tdmpc2/tools/evaluate_rof_normalized_delta.py \
	tdmpc2/check_rof_transition_refit_contract.py \
	tdmpc2/check_rof_normalized_delta_contract.py \
	>"$INCOMPLETE/provenance/source_sha256.txt"

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
	previous="$PREVIOUS_REFIT_ROOT/$task.json"
	for required in "$clean" "$hard" "$runtime" "$checkpoint" "$previous"; do
		[[ -f "$required" ]] || { echo "Missing immutable input: $required" >&2; exit 3; }
	done
	mkdir -p "$INCOMPLETE/results/$task"
	(
		export CUDA_VISIBLE_DEVICES="$gpu"
		export CUBLAS_WORKSPACE_CONFIG=:4096:8
		export PYTHONPATH="$SOURCE_ROOT"
		for fit_seed in "${fit_seeds[@]}"; do
			echo "NORMALIZED_DELTA_START task=$task gpu=$gpu fit_seed=$fit_seed"
			timeout --foreground --signal=TERM --kill-after=30s 7200 \
				"$PYTHON_BIN" -B -m tdmpc2.tools.evaluate_rof_normalized_delta \
				--clean-dataset "$clean" \
				--hard-dataset "$hard" \
				--runtime-config "$runtime" \
				--checkpoint "$checkpoint" \
				--previous-refit-json "$previous" \
				--output "$INCOMPLETE/results/$task/seed_$fit_seed.json" \
				--seed "$fit_seed" \
				--bootstrap-seed 314159 \
				--bootstrap-resamples 20000 \
				--encoder-batch-size 256 \
				--fit-batch-size 512 \
				--max-epochs 120 \
				--min-epochs 50 \
				--patience 25 \
				--convergence-window 20 \
				--convergence-threshold 0.01 \
				--learning-rate 3e-4 \
				--weight-decay 1e-5
			echo "NORMALIZED_DELTA_END task=$task gpu=$gpu fit_seed=$fit_seed rc=0"
		done
	) >"$INCOMPLETE/logs/$task.log" 2>&1 &
	pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
	task="${tasks[$index]}"
	gpu="${gpus[$index]}"
	if wait "${pids[$index]}"; then
		echo "TASK_END task=$task gpu=$gpu rc=0"
	else
		rc=$?
		echo "TASK_END task=$task gpu=$gpu rc=$rc" >&2
		failed=1
	fi
done

if [[ "$failed" -ne 0 ]]; then
	echo "ROF_NORMALIZED_DELTA_CAMPAIGN_FAILED_ROOT=$INCOMPLETE" >&2
	exit 1
fi

"$PYTHON_BIN" - "$INCOMPLETE" "${tasks[@]}" -- "${fit_seeds[@]}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

from tdmpc2.tools.evaluate_rof_normalized_delta import validate_result

root = Path(sys.argv[1]).resolve()
separator = sys.argv.index('--')
tasks = sys.argv[2:separator]
seeds = [int(value) for value in sys.argv[separator + 1:]]
entries = {}
all_scientific_complete = True
for task in tasks:
    entries[task] = {}
    for seed in seeds:
        path = root / 'results' / task / f'seed_{seed}.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        validate_result(payload)
        if payload['task'] != task or payload['protocol']['seed'] != seed:
            raise ValueError(f'Task/seed mismatch in {path}')
        all_scientific_complete &= bool(payload['scientific_complete'])
        entries[task][str(seed)] = {
            'path': str(path.relative_to(root)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'scientific_complete': bool(payload['scientific_complete']),
            'in_domain_attribution': payload['gates']['in_domain_attribution'],
        }
summary = {
    'format': 'rof_normalized_delta_campaign_v1',
    'status': 'rof_normalized_delta_campaign_complete',
    'engineering_pass': True,
    'scientific_complete': bool(all_scientific_complete),
    'controller_training_authorized': False,
    'policy_training_performed': False,
    'privileged_targets_used': False,
    'tasks': tasks,
    'fit_seeds': seeds,
    'protocol': {
        'frozen_encoder_checkpoint': True,
        'whole_episode_split': '12_train_4_validation_4_test',
        'fit_domains': ['clean', 'hard'],
        'test_domains': ['clean', 'hard'],
        'horizons': [1, 3, 5],
        'development_diagnostic_not_controller_training': True,
    },
    'entries': entries,
}
target = root / 'rof_normalized_delta_summary.json'
temporary = root / f'.{target.name}.{os.getpid()}.tmp'
temporary.write_text(
    json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n',
    encoding='utf-8',
)
os.replace(temporary, target)
PY

mv "$INCOMPLETE" "$OUTPUT_ROOT"
echo "ROF_NORMALIZED_DELTA_CAMPAIGN_COMPLETE"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "SUMMARY=$OUTPUT_ROOT/rof_normalized_delta_summary.json"
