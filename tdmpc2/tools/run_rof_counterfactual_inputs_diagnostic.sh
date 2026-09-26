#!/usr/bin/env bash
set -uo pipefail

SOURCE_ROOT=/root/tdmpc2_runs/rof_causal_ladder_source_v1_20260913_155806
DATA_ROOT=/root/tdmpc2_runs/rof_causal_ladder_v1_20260913_155806
RUN_ROOT=/root/tdmpc2_runs/rof_wm_v1_diagnostics_30k_v1_20260913_1315/workspaces
PYTHON_BIN=/root/conda_envs/tdmpc2_2026/bin/python

if [[ $# -ne 1 ]]; then
	echo "usage: $0 OUTPUT_ROOT" >&2
	exit 2
fi

OUTPUT_ROOT=$1
if [[ -e "$OUTPUT_ROOT" ]]; then
	echo "output root already exists: $OUTPUT_ROOT" >&2
	exit 2
fi
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/results/clean" "$OUTPUT_ROOT/results/hard"

run_task() {
	local task=$1
	local gpu=$2
	local experiment=$3
	local checkpoint_root="$RUN_ROOT/$experiment/logs/$task/6/$experiment"
	local log="$OUTPUT_ROOT/logs/$task.log"
	local condition

	exec >"$log" 2>&1
	echo "COUNTERFACTUAL_TASK_START task=$task gpu=$gpu"
	for condition in clean hard; do
		echo "COUNTERFACTUAL_CONDITION_START task=$task condition=$condition gpu=$gpu"
		env \
			CUDA_VISIBLE_DEVICES="$gpu" \
			MUJOCO_GL=egl \
			PYOPENGL_PLATFORM=egl \
			PYTHONPATH="$SOURCE_ROOT" \
			"$PYTHON_BIN" -B -m tdmpc2.tools.evaluate_rof_counterfactual_inputs \
			--dataset "$DATA_ROOT/full_${condition}/datasets/$task/dataset_manifest.json" \
			--reference-clean-dataset "$DATA_ROOT/full_clean/datasets/$task/dataset_manifest.json" \
			--runtime-config "$checkpoint_root/runtime_config.json" \
			--checkpoint "$checkpoint_root/models/final.pt" \
			--output "$OUTPUT_ROOT/results/$condition/$task.json" \
			--batch-size 128 \
			--bootstrap-seed 20260913 \
			--bootstrap-resamples 20000
		local rc=$?
		echo "COUNTERFACTUAL_CONDITION_END task=$task condition=$condition gpu=$gpu rc=$rc"
		if [[ $rc -ne 0 ]]; then
			return "$rc"
		fi
	done
	echo "COUNTERFACTUAL_TASK_END task=$task gpu=$gpu rc=0"
}

run_task walker-run 3 rof_v1_temporal_walker-run_seed6 &
pid_walker=$!
run_task reacher-easy 4 rof_v1_temporal_reacher-easy_seed6 &
pid_reacher=$!
run_task finger-turn-easy 5 rof_v1_target_finger-turn-easy_seed6 &
pid_finger=$!
run_task cartpole-balance-sparse 6 rof_v1_geometry_aux_cartpole-balance-sparse_seed6 &
pid_cartpole=$!

rc=0
for pid in "$pid_walker" "$pid_reacher" "$pid_finger" "$pid_cartpole"; do
	if ! wait "$pid"; then
		rc=1
	fi
done

if [[ $rc -ne 0 ]]; then
	echo "ROF_COUNTERFACTUAL_INPUTS_FAILED output_root=$OUTPUT_ROOT"
	exit "$rc"
fi

echo "ROF_COUNTERFACTUAL_INPUTS_COMPLETE output_root=$OUTPUT_ROOT"
