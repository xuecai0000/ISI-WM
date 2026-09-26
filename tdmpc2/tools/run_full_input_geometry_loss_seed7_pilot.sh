#!/usr/bin/env bash
# Seed-7 two-task development kill-test for Full-Cutie input with a
# geometry/status-only object auxiliary target and the full 1770 denominator.
# The frozen historical Full-Cutie checkpoint is reevaluated, never retrained.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${VIDEO_ROOT:?Set VIDEO_ROOT to the frozen video_hard directory}"
: "${OC_REPO:?Set OC_REPO to the OC-STORM checkout}"
: "${CUTIE_CKPT:?Set CUTIE_CKPT to cutie-small-mega.pth}"

PY="${PY:-python}"
MANIFEST_DIR="${MANIFEST_DIR:-$REPO_ROOT/tdmpc2/envs/background_manifests}"
SUPPORT_BASE="${SUPPORT_BASE:-$REPO_ROOT/datasets/cutie_multitask_support_v1_seed314159}"
SOURCE_MEMORY_ROOT="${SOURCE_MEMORY_ROOT:-$REPO_ROOT/logs/_diagnostic/cutie_object_memory_probe_100k_v1_seed7}"
GPU_REACHER="${GPU_REACHER:-0}"
GPU_CARTPOLE="${GPU_CARTPOLE:-1}"

readonly SEED=7 STEPS=100000 EVAL_FREQ=20000 EVAL_EPISODES=3
readonly HELDOUT_EPISODES=20 ENV_SEED=424243 BACKGROUND_SEED=1618034
readonly PLANNER_SEED_BASE=8675400
readonly TARGET=geometry_status_full_denominator
readonly ARM=full_input_geometry_loss
readonly RUN_TAG=full_input_geometry_loss_100k_v1_seed7
readonly FORMAT=full_input_geometry_loss_seed7_pilot_v1
readonly BASE="$REPO_ROOT/logs/_diagnostic/$RUN_TAG"
readonly STAGE="${BASE}.incomplete"
readonly SUMMARY="$STAGE/full_input_geometry_loss_summary.json"
readonly -a TASKS=(reacher-visual-small cartpole-swingup)

for name in GPU_REACHER GPU_CARTPOLE; do
	value="${!name}"
	[[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || {
		echo "$name is invalid: $value" >&2; exit 2;
	}
done
[[ "$GPU_REACHER" != "$GPU_CARTPOLE" ]] || {
	echo "GPU_REACHER and GPU_CARTPOLE must differ." >&2; exit 2;
}
(( BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 1) )) || {
	echo "Bash >=5.1 is required." >&2; exit 2;
}
if [[ "$PY" == */* ]]; then
	[[ -x "$PY" ]] || { echo "Python is not executable: $PY" >&2; exit 2; }
else
	command -v "$PY" >/dev/null || { echo "Python is not on PATH: $PY" >&2; exit 2; }
fi
for path in "$VIDEO_ROOT" "$OC_REPO" "$CUTIE_CKPT" "$MANIFEST_DIR" \
	"$SUPPORT_BASE" "$SOURCE_MEMORY_ROOT"; do
	[[ -e "$path" ]] || { echo "Missing immutable input: $path" >&2; exit 2; }
done
[[ "$(basename -- "${VIDEO_ROOT%/}")" == video_hard ]] || {
	echo "VIDEO_ROOT must be the frozen video_hard directory." >&2; exit 2;
}

task_key() { printf '%s' "${1//-/_}"; }

experiment_name() {
	local task=$1
	printf 'cutie_object_%s100k_%s_%s' "$ARM" "$RUN_TAG" "$(task_key "$task")"
}

run_root() {
	local task=$1
	printf '%s/logs/%s/%s/%s' "$REPO_ROOT" "$task" "$SEED" \
		"$(experiment_name "$task")"
}

anchor_root() {
	local task=$1
	printf '%s/logs/%s/%s/cutie_object_hard_zero100k_cutie_object_memory_probe_100k_v1_seed7_%s' \
		"$REPO_ROOT" "$task" "$SEED" "$(task_key "$task")"
}

for path in "$BASE" "$STAGE"; do
	[[ ! -e "$path" ]] || {
		echo "Refusing existing runner output: $path" >&2; exit 3;
	}
done
for task in "${TASKS[@]}"; do
	for path in \
		"$SUPPORT_BASE/$task/annotations.json" \
		"$SOURCE_MEMORY_ROOT/tasks/$task/evaluations/normal/hard_zero.json" \
		"$(anchor_root "$task")/runtime_config.json" \
		"$(anchor_root "$task")/models/final.pt"; do
		[[ -f "$path" ]] || { echo "Missing task input: $path" >&2; exit 2; }
	done
	[[ ! -e "$(run_root "$task")" ]] || {
		echo "Refusing existing training root: $(run_root "$task")" >&2; exit 3;
	}
done

mkdir -p "$STAGE/contracts" "$STAGE/provenance" "$STAGE/tasks"
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
	local rc=$? pid task root destination failed relocations
	trap - EXIT INT TERM
	for pid in "${ACTIVE_PIDS[@]:-}"; do
		[[ -n "$pid" ]] && terminate_tree "$pid"
	done
	for pid in "${ACTIVE_PIDS[@]:-}"; do
		[[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
	done
	if (( PROMOTED == 0 )) && [[ -d "$STAGE" ]]; then
		if [[ ! -f "$SUMMARY" ]]; then
			"$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps({"format":sys.argv[3],"status":"full_input_geometry_loss_seed7_engineering_fail","runner_exit_code":int(sys.argv[2]),"failure":"runner exited before strict aggregation"},indent=2)+"\n",encoding="utf-8")' "$SUMMARY" "$rc" "$FORMAT" 2>/dev/null || true
		fi
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
		[[ ! -e "$failed" ]] || failed="${failed}.${RANDOM}"
		relocations="$STAGE/provenance/failed_training_root_relocations.tsv"
		: >"$relocations"
		for task in "${TASKS[@]}"; do
			root="$(run_root "$task")"
			destination="training_roots/$task/$ARM"
			if [[ -e "$root" ]]; then
				mkdir -p "$STAGE/training_roots/$task"
				if mv -- "$root" "$STAGE/$destination"; then
					printf '%s\t%s\t%s\t%s\ttrue\n' "$task" "$ARM" "$root" "$destination" >>"$relocations"
				else
					printf '%s\t%s\t%s\t%s\tfalse\n' "$task" "$ARM" "$root" "$destination" >>"$relocations"
				fi
			else
				printf '%s\t%s\t%s\t%s\tabsent\n' "$task" "$ARM" "$root" "$destination" >>"$relocations"
			fi
		done
		"$PY" - "$relocations" \
			"$STAGE/provenance/failed_training_root_relocations.json" "$failed" <<'PY'
import json
import os
import sys
from pathlib import Path

source_path, output_path, failed_root = map(Path, sys.argv[1:4])
records = []
for line in source_path.read_text(encoding='utf-8').splitlines():
    task, arm, original, relative, state = line.split('\t')
    records.append({
        'task': task,
        'arm': arm,
        'original_training_root': original,
        'archived_relative_to_summary_root': relative,
        'source_state': state,
        'moved': state == 'true',
    })
payload = {
    'format': 'full_input_geometry_loss_failed_relocations_v1',
    'failed_archive_root': str(failed_root.resolve()),
    'records': records,
}
temporary = output_path.with_name(output_path.name + '.incomplete')
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
    encoding='utf-8', newline='\n',
)
os.replace(temporary, output_path)
PY
		mv -- "$STAGE" "$failed"
		echo "FULL_INPUT_GEOMETRY_LOSS_FAILED_ARCHIVE=$failed" >&2
	fi
	exit "$rc"
}
trap archive_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

write_rc() { printf '%s\n' "$2" >"$1"; }

echo "[1/6] Static and dependency-light auxiliary-target contracts"
bash -n "$0"
set +e
(
	set -e
	"$PY" -B -m py_compile \
		tdmpc2/common/cutie_object_auxiliary.py \
		tdmpc2/common/world_model.py tdmpc2/tdmpc2.py \
		tdmpc2/check_cutie_object_auxiliary_target_contract.py \
		tdmpc2/check_cutie_object_only_update.py \
		tdmpc2/tools/check_full_input_geometry_loss_env.py \
		tdmpc2/tools/evaluate_full_input_geometry_loss.py \
		tdmpc2/tools/full_input_geometry_loss_protocol.py
	"$PY" -B tdmpc2/check_cutie_object_auxiliary_target_contract.py
	"$PY" -B tdmpc2/check_cutie_object_wrapper_contract.py
	"$PY" -B tdmpc2/check_cutie_object_only_integration_contract.py
	"$PY" -B tdmpc2/check_cutie_multitask_support_contract.py
) >"$STAGE/contracts/dependency_light.log" 2>&1
dependency_rc=$?
set -e
write_rc "$STAGE/contracts/dependency_light.rc" "$dependency_rc"
if (( dependency_rc != 0 )); then
	echo "Dependency-light contracts failed: $dependency_rc" >&2; exit 4
fi

echo "[2/6] Bind immutable source, code, support, manifest, and video trees"
"$PY" -B -m tdmpc2.tools.full_input_geometry_loss_protocol bind \
	--repo "$REPO_ROOT" --source-memory-root "$SOURCE_MEMORY_ROOT" \
	--support-base "$SUPPORT_BASE" --video-root "$VIDEO_ROOT" \
	--manifest-dir "$MANIFEST_DIR" --oc-repo "$OC_REPO" \
	--cutie-checkpoint "$CUTIE_CKPT" --gpu-reacher "$GPU_REACHER" \
	--gpu-cartpole "$GPU_CARTPOLE" --output "$STAGE/provenance/inputs.json"

run_gpu_contract() {
	local label=$1 gpu=$2 mode=$3 log="$STAGE/contracts/${1}.log" rc
	local -a mode_args=()
	[[ "$mode" == compile ]] && mode_args+=(--compile)
	echo "GPU_CONTRACT_START label=$label gpu=$gpu mode=$mode"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
		"$PY" tdmpc2/check_cutie_object_only_update.py \
		--auxiliary-target "$TARGET" "${mode_args[@]}" >"$log" 2>&1
	rc=$?
	set -e
	write_rc "$STAGE/contracts/${label}.rc" "$rc"
	echo "GPU_CONTRACT_END label=$label gpu=$gpu mode=$mode rc=$rc"
	return "$rc"
}

echo "[3/6] Eager and compiled GPU isolation/gradient contracts"
set +e
run_gpu_contract eager_gpu_reacher "$GPU_REACHER" eager & p0=$!; ACTIVE_PIDS+=("$p0")
run_gpu_contract compile_gpu_cartpole "$GPU_CARTPOLE" compile & p1=$!; ACTIVE_PIDS+=("$p1")
wait "$p0"; gpu_r0=$?
wait "$p1"; gpu_r1=$?
set -e
ACTIVE_PIDS=()
if (( gpu_r0 != 0 || gpu_r1 != 0 )); then
	echo "Auxiliary-target GPU contracts failed: $gpu_r0 $gpu_r1" >&2; exit 4
fi

run_preflight() {
	local gpu=$1 task=$2 dir=$3 out="$3/preflight.json" log="$3/preflight.log" rc
	echo "PREFLIGHT_START task=$task arm=$ARM gpu=$gpu" | tee "$log"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" -B -m tdmpc2.tools.check_full_input_geometry_loss_env \
		--task "$task" --runtime-config "$(anchor_root "$task")/runtime_config.json" \
		--steps 16 --output "$out" >>"$log" 2>&1
	rc=$?
	set -e
	write_rc "$dir/preflight.rc" "$rc"
	echo "PREFLIGHT_END task=$task arm=$ARM gpu=$gpu rc=$rc" | tee -a "$log"
}

run_training() {
	local gpu=$1 task=$2 dir=$3 support exp root log hydra rc role0 role1
	support="$SUPPORT_BASE/$task/annotations.json"
	exp="$(experiment_name "$task")"
	root="$(run_root "$task")"
	log="$dir/train.log"
	hydra="$dir/hydra"
	case "$task" in
		reacher-visual-small) role0=whole_arm; role1=goal ;;
		cartpole-swingup) role0=cart; role1=pole ;;
		*) return 2 ;;
	esac
	local -a train_args=(
		"task=$task" obs=rgb model_size=5 "steps=$STEPS" "seed=$SEED"
		"eval_freq=$EVAL_FREQ" "eval_episodes=$EVAL_EPISODES"
		video_background_enabled=true "video_background_root=$VIDEO_ROOT"
		"video_background_manifest_dir=$MANIFEST_DIR" video_background_split=train
		"+video_background_seed=$SEED" video_background_strength=1.0
		video_background_total_frames=1000 video_background_source_cache_size=8
		visual_foreground_erosion_pixels=0 compile=true compile_fallback_random=true
		enable_wandb=false wandb_project=none wandb_entity=none save_csv=true
		save_video=false save_agent=true checkpoint=null data_dir=null obs_shapes=null
		action_dims=null episode_lengths=null flat_anchor=true
		flat_anchor_mode=cutie_object_only
		"cutie_object_repo=$OC_REPO" "cutie_object_checkpoint=$CUTIE_CKPT"
		"cutie_object_support_path=$support" cutie_object_support_schema=generic_indexed_v1
		"cutie_object_role_names=[$role0,$role1]"
		cutie_object_allow_simulator_support=true
		cutie_object_observation_variant=full
		cutie_object_frame_schema=cutie_query_mask_status_v1
		cutie_object_allow_simulator_runtime=false
		"cutie_object_auxiliary_target=$TARGET"
		cutie_object_last_valid_memory=false cutie_object_policy_burst_plan=null
		cutie_object_belief_enabled=false cutie_object_belief_use_for_control=false
		cutie_object_config_dir=null cutie_object_device=cuda:0
		cutie_object_tracker_height=448 cutie_object_tracker_width=448
		cutie_object_model_size=small cutie_object_prompt_radius=2.0
		cutie_object_amp=true cutie_object_worker_timeout_seconds=180
		cutie_object_num_roles=2 cutie_object_frame_dim=590
		cutie_object_stack_frames=3 cutie_object_input_dim=1770
		cutie_object_role_dim=64 cutie_object_hidden_dim=256
		cutie_object_joint_dim=640 cutie_object_only_latent_dim=128
		"exp_name=$exp" "hydra.run.dir=$hydra" hydra.job.chdir=false
	)
	echo "TRAIN_START task=$task arm=$ARM gpu=$gpu root=$root" | tee "$log"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" tdmpc2/train.py "${train_args[@]}" >>"$log" 2>&1
	rc=$?
	set -e
	write_rc "$dir/train.rc" "$rc"
	echo "TRAIN_END task=$task arm=$ARM gpu=$gpu rc=$rc" | tee -a "$log"
}

run_new_evaluation() {
	local gpu=$1 task=$2 dir=$3 root out log rc
	root="$(run_root "$task")"
	out="$dir/evaluations/$ARM.json"
	log="$dir/evaluations/$ARM.log"
	if [[ ! -f "$root/runtime_config.json" || ! -f "$root/models/final.pt" ]]; then
		write_rc "$dir/evaluations/$ARM.rc" 66; return 0
	fi
	echo "EVAL_START task=$task arm=$ARM gpu=$gpu" | tee "$log"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" -B -m tdmpc2.tools.evaluate_full_input_geometry_loss \
		--task "$task" --runtime-config "$root/runtime_config.json" \
		--checkpoint "$root/models/final.pt" --training-seed "$SEED" \
		--expected-training-steps "$STEPS" --expected-training-eval-freq "$EVAL_FREQ" \
		--expected-training-eval-episodes "$EVAL_EPISODES" \
		--episodes "$HELDOUT_EPISODES" --env-seed "$ENV_SEED" \
		--background-seed "$BACKGROUND_SEED" \
		--planner-seed-base "$PLANNER_SEED_BASE" --output "$out" >>"$log" 2>&1
	rc=$?
	set -e
	write_rc "$dir/evaluations/$ARM.rc" "$rc"
	echo "EVAL_END task=$task arm=$ARM gpu=$gpu rc=$rc" | tee -a "$log"
}

run_anchor_evaluation() {
	local gpu=$1 task=$2 dir=$3 root out log rc
	root="$(anchor_root "$task")"
	out="$dir/evaluations/full_cutie.json"
	log="$dir/evaluations/full_cutie.log"
	echo "EVAL_START task=$task arm=historical_full_cutie gpu=$gpu" | tee "$log"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" -B -m tdmpc2.tools.evaluate_cutie_multitask_checkpoint \
		--task "$task" --backend cutie_object_only \
		--runtime-config "$root/runtime_config.json" --checkpoint "$root/models/final.pt" \
		--training-seed "$SEED" --expected-training-steps "$STEPS" \
		--expected-training-eval-freq "$EVAL_FREQ" \
		--expected-training-eval-episodes "$EVAL_EPISODES" \
		--episodes "$HELDOUT_EPISODES" --env-seed "$ENV_SEED" \
		--background-seed "$BACKGROUND_SEED" \
		--planner-seed-base "$PLANNER_SEED_BASE" --erosion-pixels 0 \
		--output "$out" >>"$log" 2>&1
	rc=$?
	set -e
	write_rc "$dir/evaluations/full_cutie.rc" "$rc"
	echo "EVAL_END task=$task arm=historical_full_cutie gpu=$gpu rc=$rc" | tee -a "$log"
}

run_task() {
	local gpu=$1 task=$2 dir="$STAGE/tasks/$2" preflight_rc train_rc
	mkdir -p "$dir/evaluations"
	printf '%s\n' "$gpu" >"$dir/gpu"
	run_preflight "$gpu" "$task" "$dir"
	preflight_rc="$(<"$dir/preflight.rc")"
	if (( preflight_rc == 0 )); then
		run_training "$gpu" "$task" "$dir"
		train_rc="$(<"$dir/train.rc")"
		if (( train_rc == 0 )); then
			run_new_evaluation "$gpu" "$task" "$dir"
		else
			write_rc "$dir/evaluations/$ARM.rc" 125
		fi
	else
		write_rc "$dir/train.rc" 125
		write_rc "$dir/evaluations/$ARM.rc" 125
	fi
	run_anchor_evaluation "$gpu" "$task" "$dir"
}

echo "[4/6] Two-task 100k training plus paired 20-episode held-out evaluation"
run_task "$GPU_REACHER" reacher-visual-small & p0=$!; ACTIVE_PIDS+=("$p0")
run_task "$GPU_CARTPOLE" cartpole-swingup & p1=$!; ACTIVE_PIDS+=("$p1")
set +e
wait "$p0"; task_r0=$?
wait "$p1"; task_r1=$?
set -e
ACTIVE_PIDS=()
if (( task_r0 != 0 || task_r1 != 0 )); then
	echo "Task worker shell failed: $task_r0 $task_r1" >&2; exit 4
fi

echo "[5/6] Strict aggregation and exact historical-anchor regression"
set +e
"$PY" -B -m tdmpc2.tools.full_input_geometry_loss_protocol aggregate \
	--repo "$REPO_ROOT" --stage "$STAGE" \
	--inputs "$STAGE/provenance/inputs.json" --run-tag "$RUN_TAG" \
	--output "$SUMMARY"
aggregate_rc=$?
set -e
if (( aggregate_rc != 0 )); then
	exit "$aggregate_rc"
fi

echo "[6/6] Promote immutable diagnostic"
mv -- "$STAGE" "$BASE"
PROMOTED=1
trap - EXIT INT TERM
echo "FULL_INPUT_GEOMETRY_LOSS_SEED7_PILOT_COMPLETE"
echo "SUMMARY=$BASE/full_input_geometry_loss_summary.json"
