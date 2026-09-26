#!/usr/bin/env bash
# Two-task, single-seed privileged GT-mask geometry diagnostic.
#
# New scratch arms:
#   cutie_mask_geometry = tracked mask geometry/status, query block exact zero
#   gt_mask_geometry    = same-state MuJoCo mask geometry, query exact zero
# A frozen Full-Cutie checkpoint is freshly reevaluated as a regression anchor.

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
readonly RUN_TAG=gt_mask_geometry_100k_v1
readonly FORMAT=gt_mask_geometry_seed7_pilot_v1
readonly BASE="$REPO_ROOT/logs/_diagnostic/${RUN_TAG}_seed${SEED}"
readonly STAGE="${BASE}.incomplete"
readonly SUMMARY="$STAGE/gt_mask_geometry_summary.json"
readonly -a TASKS=(reacher-visual-small cartpole-swingup)
readonly -a ARMS=(gt_mask_geometry cutie_mask_geometry)

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

arm_schema() {
	case "$1" in
		cutie_mask_geometry) printf '%s' cutie_mask_geometry_v1 ;;
		gt_mask_geometry) printf '%s' simulator_gt_mask_geometry_v1 ;;
		*) return 2 ;;
	esac
}

arm_privilege() {
	case "$1" in
		cutie_mask_geometry) printf '%s' false ;;
		gt_mask_geometry) printf '%s' true ;;
		*) return 2 ;;
	esac
}

experiment_name() {
	local task=$1 arm=$2
	printf 'cutie_object_%s100k_%s_seed%s_%s' \
		"$arm" "$RUN_TAG" "$SEED" "$(task_key "$task")"
}

run_root() {
	local task=$1 arm=$2
	printf '%s/logs/%s/%s/%s' "$REPO_ROOT" "$task" "$SEED" \
		"$(experiment_name "$task" "$arm")"
}

anchor_root() {
	local task=$1
	printf '%s/logs/%s/%s/cutie_object_hard_zero100k_cutie_object_memory_probe_100k_v1_seed7_%s' \
		"$REPO_ROOT" "$task" "$SEED" "$(task_key "$task")"
}

for path in "$BASE" "$STAGE"; do
	[[ ! -e "$path" ]] || { echo "Refusing existing runner output: $path" >&2; exit 3; }
done
for task in "${TASKS[@]}"; do
	for path in \
		"$SUPPORT_BASE/$task/annotations.json" \
		"$SUPPORT_BASE/$task/geom_catalog.json" \
		"$SOURCE_MEMORY_ROOT/tasks/$task/evaluations/normal/hard_zero.json" \
		"$(anchor_root "$task")/runtime_config.json" \
		"$(anchor_root "$task")/models/final.pt"; do
		[[ -f "$path" ]] || { echo "Missing task input: $path" >&2; exit 2; }
	done
	for arm in "${ARMS[@]}"; do
		[[ ! -e "$(run_root "$task" "$arm")" ]] || {
			echo "Refusing existing training root: $(run_root "$task" "$arm")" >&2; exit 3;
		}
	done
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
	local rc=$? pid task arm root destination failed relocations
	trap - EXIT INT TERM
	for pid in "${ACTIVE_PIDS[@]:-}"; do
		[[ -n "$pid" ]] && terminate_tree "$pid"
	done
	for pid in "${ACTIVE_PIDS[@]:-}"; do
		[[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
	done
	if (( PROMOTED == 0 )) && [[ -d "$STAGE" ]]; then
		if [[ ! -f "$SUMMARY" ]]; then
			"$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps({"format":sys.argv[3],"status":"gt_mask_geometry_engineering_fail","runner_exit_code":int(sys.argv[2]),"failure":"runner exited before aggregation"},indent=2)+"\n",encoding="utf-8")' "$SUMMARY" "$rc" "$FORMAT" 2>/dev/null || true
		fi
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
		[[ ! -e "$failed" ]] || failed="${failed}.${RANDOM}"
		relocations="$STAGE/provenance/failed_training_root_relocations.tsv"
		: >"$relocations"
		for task in "${TASKS[@]}"; do
			for arm in "${ARMS[@]}"; do
				root="$(run_root "$task" "$arm")"
				destination="training_roots/$task/$arm"
				if [[ -e "$root" ]]; then
					mkdir -p "$STAGE/training_roots/$task"
					if mv -- "$root" "$STAGE/$destination"; then
						printf '%s\t%s\t%s\t%s\ttrue\n' "$task" "$arm" "$root" "$destination" >>"$relocations"
					else
						printf '%s\t%s\t%s\t%s\tfalse\n' "$task" "$arm" "$root" "$destination" >>"$relocations"
					fi
				else
					printf '%s\t%s\t%s\t%s\tabsent\n' "$task" "$arm" "$root" "$destination" >>"$relocations"
				fi
			done
		done
		"$PY" - "$SUMMARY" "$relocations" \
			"$STAGE/provenance/failed_training_root_relocations.json" "$failed" <<'PY'
import json
import os
import sys
from pathlib import Path

summary_path, source_path, output_path, failed_root = map(Path, sys.argv[1:5])
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
    'format': 'gt_mask_geometry_failed_training_root_relocations_v1',
    'failed_archive_root': str(failed_root.resolve()),
    'records': records,
}
temporary = output_path.with_name(output_path.name + '.incomplete')
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
    encoding='utf-8', newline='\n',
)
os.replace(temporary, output_path)
if summary_path.is_file():
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    for record in records:
        if not record['moved']:
            continue
        arm_report = summary.get('tasks', {}).get(record['task'], {}).get(
            'arms', {}
        ).get(record['arm'], {})
        arm_report['execution_time_training_root'] = record[
            'original_training_root'
        ]
        arm_report['archived_training_root_relative_to_summary_root'] = record[
            'archived_relative_to_summary_root'
        ]
        original = Path(record['original_training_root'])
        for artifact in arm_report.get('artifacts', {}).values():
            try:
                suffix = Path(artifact['path']).resolve().relative_to(
                    original.resolve()
                )
            except (KeyError, TypeError, ValueError):
                continue
            artifact['archived_relative_to_summary_root'] = (
                Path(record['archived_relative_to_summary_root']) / suffix
            ).as_posix()
    summary['failure_archive'] = {
        'root': str(failed_root.resolve()),
        'training_root_relocations_relative_to_summary_root': (
            'provenance/failed_training_root_relocations.json'
        ),
    }
    replacement = summary_path.with_name(summary_path.name + '.incomplete')
    replacement.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8', newline='\n',
    )
    os.replace(replacement, summary_path)
PY
		rm -f -- "$relocations"
		mv -- "$STAGE" "$failed"
		echo "GT_MASK_GEOMETRY_FAILED_ARCHIVE=$failed" >&2
	fi
	exit "$rc"
}
trap archive_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

write_rc() { printf '%s\n' "$2" >"$1"; }
run_contract() {
	local label=$1; shift
	echo "[contract] $label"
	{ echo "===== $label ====="; "$@"; } >>"$STAGE/contracts/contracts.log" 2>&1
}

echo "[1/6] Static and dependency-light contracts"
bash -n "$0"
run_contract py_compile "$PY" -B -m py_compile \
	tdmpc2/envs/wrappers/gt_mask_oracle.py \
	tdmpc2/envs/wrappers/cutie_object.py \
	tdmpc2/common/world_model.py tdmpc2/tdmpc2.py \
	tdmpc2/check_gt_mask_geometry_contract.py \
	tdmpc2/tools/check_gt_mask_geometry_env.py \
	tdmpc2/tools/evaluate_gt_mask_geometry_oracle.py \
	tdmpc2/tools/gt_mask_geometry_protocol.py
run_contract gt_mask_geometry "$PY" -B tdmpc2/check_gt_mask_geometry_contract.py
run_contract object_wrapper "$PY" -B tdmpc2/check_cutie_object_wrapper_contract.py
run_contract object_only_integration "$PY" -B tdmpc2/check_cutie_object_only_integration_contract.py
run_contract multitask_support "$PY" -B tdmpc2/check_cutie_multitask_support_contract.py

echo "[2/6] Bind immutable sources and implementation"
"$PY" -B -m tdmpc2.tools.gt_mask_geometry_protocol bind \
	--repo "$REPO_ROOT" --source-memory-root "$SOURCE_MEMORY_ROOT" \
	--support-base "$SUPPORT_BASE" --video-root "$VIDEO_ROOT" \
	--manifest-dir "$MANIFEST_DIR" --oc-repo "$OC_REPO" \
	--cutie-checkpoint "$CUTIE_CKPT" --gpu-reacher "$GPU_REACHER" \
	--gpu-cartpole "$GPU_CARTPOLE" --output "$STAGE/provenance/inputs.json"

run_gpu_contract() {
	local label=$1 gpu=$2 log="$STAGE/contracts/${1}.log" rc
	echo "GPU_CONTRACT_START label=$label gpu=$gpu"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
		"$PY" tdmpc2/check_cutie_object_only_update.py --compile >"$log" 2>&1
	rc=$?
	set -e
	write_rc "$STAGE/contracts/${label}.rc" "$rc"
	echo "GPU_CONTRACT_END label=$label gpu=$gpu rc=$rc"
	return "$rc"
}

echo "[3/6] Matched ObjectOnly compile contracts"
set +e
run_gpu_contract gpu_reacher "$GPU_REACHER" & p0=$!; ACTIVE_PIDS+=("$p0")
run_gpu_contract gpu_cartpole "$GPU_CARTPOLE" & p1=$!; ACTIVE_PIDS+=("$p1")
wait "$p0"; r0=$?
wait "$p1"; r1=$?
set -e
ACTIVE_PIDS=()
if (( r0 != 0 || r1 != 0 )); then
	echo "GPU contracts failed: $r0 $r1" >&2; exit 4
fi

run_preflight() {
	local gpu=$1 task=$2 arm=$3 dir=$4 rc
	local out="$dir/preflight/${arm}.json" log="$dir/preflight/${arm}.log"
	echo "PREFLIGHT_START task=$task arm=$arm gpu=$gpu" | tee "$log"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" -B -m tdmpc2.tools.check_gt_mask_geometry_env \
		--task "$task" --variant "$arm" \
		--runtime-config "$(anchor_root "$task")/runtime_config.json" \
		--steps 16 --output "$out" >>"$log" 2>&1
	rc=$?
	set -e
	write_rc "$dir/preflight/${arm}.rc" "$rc"
	echo "PREFLIGHT_END task=$task arm=$arm gpu=$gpu rc=$rc" | tee -a "$log"
}

run_training() {
	local gpu=$1 task=$2 arm=$3 dir=$4 support schema privilege exp root log hydra rc
	support="$SUPPORT_BASE/$task/annotations.json"
	schema="$(arm_schema "$arm")"
	privilege="$(arm_privilege "$arm")"
	exp="$(experiment_name "$task" "$arm")"
	root="$(run_root "$task" "$arm")"
	log="$dir/${arm}.train.log"
	hydra="$dir/hydra_${arm}"
	local role0 role1
	case "$task" in
		reacher-visual-small) role0=whole_arm; role1=goal ;;
		cartpole-swingup) role0=cart; role1=pole ;;
		*) return 2 ;;
	esac
	local -a args=(
		"task=$task" obs=rgb model_size=5 "steps=$STEPS" "seed=$SEED"
		"eval_freq=$EVAL_FREQ" "eval_episodes=$EVAL_EPISODES"
		video_background_enabled=true "video_background_root=$VIDEO_ROOT"
		"video_background_manifest_dir=$MANIFEST_DIR" video_background_split=train
		"+video_background_seed=$SEED" video_background_strength=1.0
		video_background_total_frames=1000 video_background_source_cache_size=8
		visual_foreground_erosion_pixels=0 compile=true compile_fallback_random=true
		enable_wandb=false wandb_project=none wandb_entity=none save_csv=true
		save_video=false save_agent=true checkpoint=null data_dir=null obs_shapes=null
		action_dims=null episode_lengths=null flat_anchor=true flat_anchor_mode=cutie_object_only
		"cutie_object_repo=$OC_REPO" "cutie_object_checkpoint=$CUTIE_CKPT"
		"cutie_object_support_path=$support" cutie_object_support_schema=generic_indexed_v1
		"cutie_object_role_names=[$role0,$role1]" cutie_object_allow_simulator_support=true
		"cutie_object_observation_variant=$arm" "cutie_object_frame_schema=$schema"
		"cutie_object_allow_simulator_runtime=$privilege"
		cutie_object_last_valid_memory=false cutie_object_policy_burst_plan=null
		cutie_object_belief_enabled=false cutie_object_belief_use_for_control=false
		cutie_object_config_dir=null cutie_object_device=cuda:0
		cutie_object_tracker_height=448 cutie_object_tracker_width=448
		cutie_object_model_size=small cutie_object_prompt_radius=2.0 cutie_object_amp=true
		cutie_object_worker_timeout_seconds=180 cutie_object_num_roles=2
		cutie_object_frame_dim=590 cutie_object_stack_frames=3 cutie_object_input_dim=1770
		cutie_object_role_dim=64 cutie_object_hidden_dim=256 cutie_object_joint_dim=640
		cutie_object_only_latent_dim=128 "exp_name=$exp" "hydra.run.dir=$hydra"
		hydra.job.chdir=false
	)
	echo "TRAIN_START task=$task arm=$arm gpu=$gpu root=$root" | tee "$log"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" tdmpc2/train.py "${args[@]}" >>"$log" 2>&1
	rc=$?
	set -e
	write_rc "$dir/${arm}.train.rc" "$rc"
	echo "TRAIN_END task=$task arm=$arm gpu=$gpu rc=$rc" | tee -a "$log"
}

run_new_evaluation() {
	local gpu=$1 task=$2 arm=$3 dir=$4 root out log rc
	root="$(run_root "$task" "$arm")"
	out="$dir/evaluations/${arm}.json"
	log="$dir/evaluations/${arm}.log"
	if [[ ! -f "$root/runtime_config.json" || ! -f "$root/models/final.pt" ]]; then
		write_rc "$dir/evaluations/${arm}.rc" 66; return 0
	fi
	echo "EVAL_START task=$task arm=$arm gpu=$gpu" | tee "$log"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" -B -m tdmpc2.tools.evaluate_gt_mask_geometry_oracle \
		--task "$task" --arm "$arm" --runtime-config "$root/runtime_config.json" \
		--checkpoint "$root/models/final.pt" --training-seed "$SEED" \
		--expected-training-steps "$STEPS" --expected-training-eval-freq "$EVAL_FREQ" \
		--expected-training-eval-episodes "$EVAL_EPISODES" --episodes "$HELDOUT_EPISODES" \
		--env-seed "$ENV_SEED" --background-seed "$BACKGROUND_SEED" \
		--planner-seed-base "$PLANNER_SEED_BASE" --output "$out" >>"$log" 2>&1
	rc=$?
	set -e
	write_rc "$dir/evaluations/${arm}.rc" "$rc"
	echo "EVAL_END task=$task arm=$arm gpu=$gpu rc=$rc" | tee -a "$log"
}

run_full_anchor_evaluation() {
	local gpu=$1 task=$2 dir=$3 root out log rc
	root="$(anchor_root "$task")"
	out="$dir/evaluations/full_cutie.json"
	log="$dir/evaluations/full_cutie.log"
	echo "EVAL_START task=$task arm=full_cutie_anchor gpu=$gpu" | tee "$log"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" -B -m tdmpc2.tools.evaluate_cutie_multitask_checkpoint \
		--task "$task" --backend cutie_object_only \
		--runtime-config "$root/runtime_config.json" --checkpoint "$root/models/final.pt" \
		--training-seed "$SEED" --expected-training-steps "$STEPS" \
		--expected-training-eval-freq "$EVAL_FREQ" \
		--expected-training-eval-episodes "$EVAL_EPISODES" --episodes "$HELDOUT_EPISODES" \
		--env-seed "$ENV_SEED" --background-seed "$BACKGROUND_SEED" \
		--planner-seed-base "$PLANNER_SEED_BASE" --erosion-pixels 0 \
		--output "$out" >>"$log" 2>&1
	rc=$?
	set -e
	write_rc "$dir/evaluations/full_cutie.rc" "$rc"
	echo "EVAL_END task=$task arm=full_cutie_anchor gpu=$gpu rc=$rc" | tee -a "$log"
}

run_task() {
	local gpu=$1 task=$2 dir="$STAGE/tasks/$2" arm preflight_rc train_rc
	mkdir -p "$dir/preflight" "$dir/evaluations"
	printf '%s\n' "$gpu" >"$dir/gpu"
	for arm in "${ARMS[@]}"; do
		run_preflight "$gpu" "$task" "$arm" "$dir"
		preflight_rc="$(cat "$dir/preflight/${arm}.rc")"
		if (( preflight_rc == 0 )); then
			run_training "$gpu" "$task" "$arm" "$dir"
			train_rc="$(cat "$dir/${arm}.train.rc")"
			if (( train_rc == 0 )); then
				run_new_evaluation "$gpu" "$task" "$arm" "$dir"
			else
				write_rc "$dir/evaluations/${arm}.rc" 125
			fi
		else
			write_rc "$dir/${arm}.train.rc" 125
			write_rc "$dir/evaluations/${arm}.rc" 125
		fi
	done
	run_full_anchor_evaluation "$gpu" "$task" "$dir"
}

echo "[4/6] Two-task GT/Cutie geometry scratch training and held-out evaluation"
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

echo "[5/6] Strict aggregation and frozen-anchor regression"
set +e
"$PY" -B -m tdmpc2.tools.gt_mask_geometry_protocol aggregate \
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
echo "GT_MASK_GEOMETRY_PILOT_COMPLETE"
echo "SUMMARY=$BASE/gt_mask_geometry_summary.json"
