#!/usr/bin/env bash
# Matched architecture: auxiliary state loss coefficient is the only arm change.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-<DATA_PATH>/conda_envs/tdmpc2_2026/bin/python}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
VIDEO_ROOT="${VIDEO_ROOT:-<BASELINE_PATH>/HRSSM-main/env/data/video_hard}"
CUTIE_REPO="${CUTIE_REPO:-<DATA_PATH>/r2_hrssm_third_party/OC-STORM}"
CUTIE_CHECKPOINT="${CUTIE_CHECKPOINT:-$CUTIE_REPO/feature_extractor/cutie/weights/cutie-small-mega.pth}"
SUPPORT_ROOT_V1="${SUPPORT_ROOT_V1:-$REPO_ROOT/datasets/cutie_multitask_support_v1_seed314159}"
SUPPORT_ROOT_V2="${SUPPORT_ROOT_V2:-$REPO_ROOT/datasets/cutie_multitask_support_v2_seed314159}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-object_state_supervision_paired_v1_$STAMP}"
BASE="${OUTPUT_ROOT:-$REPO_ROOT/logs/_diagnostic/$RUN_TAG}"
STAGE="${BASE}.incomplete"
SEED="${SEED:-12}"; STEPS="${STEPS:-100000}"
EVAL_FREQ="${EVAL_FREQ:-10000}"; EVAL_EPISODES="${EVAL_EPISODES:-10}"
COMPILE="${COMPILE:-true}"
TASKS_CSV="${TASKS_CSV:-acrobot-swingup,cartpole-swingup,reacher-visual-small}"
SAVE_EVAL_CHECKPOINTS="${SAVE_EVAL_CHECKPOINTS:-false}"
PRIVILEGED_STATE_SCORE="${PRIVILEGED_STATE_SCORE:-false}"
export PY GPU0 GPU1 VIDEO_ROOT CUTIE_REPO CUTIE_CHECKPOINT SUPPORT_ROOT_V1 SUPPORT_ROOT_V2
export STAMP RUN_TAG SEED STEPS EVAL_FREQ EVAL_EPISODES COMPILE
export TASKS_CSV SAVE_EVAL_CHECKPOINTS
export PRIVILEGED_STATE_SCORE
export OUTPUT_ROOT="$BASE"

task_fields() {
    case "$1" in
        acrobot-swingup) printf '%s\n' upper_arm lower_arm acrobot_swingup.json true ;;
        cartpole-swingup) printf '%s\n' cart pole cartpole_swingup.json false ;;
        reacher-visual-small) printf '%s\n' whole_arm goal reacher_visual_small.json false ;;
        *) return 2 ;;
    esac
}

run_logged() {
    local identity=$1 logfile=$2 rc
    shift 2
    printf '%q ' "$@" > "$STAGE/commands/$identity.sh.txt"
    printf '\n' >> "$STAGE/commands/$identity.sh.txt"
    if "$@" > "$logfile" 2>&1; then rc=0; else rc=$?; fi
    printf '%s\n' "$rc" > "$STAGE/status/$identity.rc"
    return "$rc"
}

best_step() {
    "$PY" -c 'import csv,sys
rows=[(int(float(r["step"])),float(r["episode_reward"])) for r in csv.DictReader(open(sys.argv[1],newline="",encoding="utf-8"))]
rows=[row for row in rows if row[0]>0]
print(max(rows,key=lambda row:(row[1],-row[0]))[0])' "$1"
}

run_state_score() {
    local task=$1 arm=$2 gpu=$3 condition=$4 run_root=$5 selection=$6 step=$7
    local checkpoint output rc
    [[ "$PRIVILEGED_STATE_SCORE" == true ]] || return 0
    if [[ "$selection" == best ]]; then
        checkpoint="$run_root/models/eval_${step}.pt"
    else
        checkpoint="$run_root/models/final.pt"
    fi
    output="$STAGE/state_scores/${task}_${arm}_${selection}_${condition}.json"
    echo "STATE_SCORE_START task=$task mode=$arm selection=$selection step=$step condition=$condition gpu=$gpu"
    local -a command=(env "CUDA_VISIBLE_DEVICES=$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1
        "$PY" -B -m tdmpc2.tools.score_object_state_decoder
        --task "$task" --backend cutie_object_only --condition "$condition"
        --runtime-config "$run_root/runtime_config.json" --checkpoint "$checkpoint"
        --training-seed "$SEED" --expected-training-steps "$STEPS"
        --expected-training-eval-freq "$EVAL_FREQ"
        --expected-training-eval-episodes "$EVAL_EPISODES"
        --episodes 20 --env-seed 424243 --background-seed 1618034
        --planner-seed-base 8675400 --erosion-pixels 0 --output "$output")
    if [[ "$selection" == best ]]; then command+=(--checkpoint-step "$step"); fi
    if run_logged "${task}_${arm}_${selection}_${condition}_state_score" \
        "$STAGE/state_scores/${task}_${arm}_${selection}_${condition}.log" "${command[@]}"; then rc=0; else rc=$?; fi
    echo "STATE_SCORE_END task=$task mode=$arm selection=$selection step=$step condition=$condition gpu=$gpu rc=$rc"
    return "$rc"
}

run_task() {
    local task=$1 arm=$2 gpu=$3 coefficient=$4
    local role0 role1 graph support true_entity run_name run_root rc condition selected
    local -a fields command
    mapfile -t fields < <(task_fields "$task")
    role0="${fields[0]}"; role1="${fields[1]}"
    graph="$REPO_ROOT/tdmpc2/object_graphs/${fields[2]}"; true_entity="${fields[3]}"
    if [[ "$task" == acrobot-swingup ]]; then
        support="$SUPPORT_ROOT_V2/$task/annotations.json"
    else
        support="$SUPPORT_ROOT_V1/$task/annotations.json"
    fi
    [[ -f "$graph" && -f "$support" ]] || { echo "Missing support/graph for $task" >&2; return 2; }
    run_name="${RUN_TAG}_${task}_${arm}"
    run_root="$REPO_ROOT/logs/$task/$SEED/$run_name"
    [[ ! -e "$run_root" ]] || { echo "Refusing overwrite: $run_root" >&2; return 3; }
    echo "TRAIN_START task=$task mode=$arm gpu=$gpu steps=$STEPS root=$run_root"
    command=(env "CUDA_VISIBLE_DEVICES=$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1
        "$PY" tdmpc2/train.py
        "task=$task" obs=rgb model_size=5 "steps=$STEPS" "seed=$SEED"
        "eval_freq=$EVAL_FREQ" "eval_episodes=$EVAL_EPISODES" save_eval_episode_trace=true
        video_background_enabled=true "video_background_root=$VIDEO_ROOT"
        video_background_split=train video_background_manifest_dir=null
        visual_foreground_erosion_pixels=0
        "compile=$COMPILE" compile_fallback_random=true
        enable_wandb=false wandb_project=none wandb_entity=none
        save_csv=true save_video=false save_agent=true "save_eval_checkpoints=$SAVE_EVAL_CHECKPOINTS"
        checkpoint=null data_dir=null obs_shapes=null action_dims=null episode_lengths=null
        flat_anchor=true flat_anchor_mode=cutie_object_only flat_anchor_loss_beta=0.1
        cutie_object_observation_variant=full cutie_object_frame_schema=cutie_query_mask_status_v1
        "cutie_object_role_names=[$role0,$role1]" cutie_object_support_schema=generic_indexed_v1
        cutie_object_allow_simulator_support=true cutie_object_allow_simulator_runtime=false
        cutie_object_allow_simulator_kinematics_runtime=false
        "cutie_object_repo=$CUTIE_REPO" "cutie_object_checkpoint=$CUTIE_CHECKPOINT"
        "cutie_object_support_path=$support" cutie_object_config_dir=null
        cutie_object_device=cuda:0 cutie_object_tracker_height=448 cutie_object_tracker_width=448
        cutie_object_model_size=small cutie_object_prompt_radius=2.0 cutie_object_amp=true
        cutie_object_worker_timeout_seconds=180 cutie_object_num_roles=2
        cutie_object_frame_dim=590 cutie_object_stack_frames=3 cutie_object_input_dim=1770
        cutie_object_auxiliary_target=geometry_status_full_denominator
        cutie_object_role_dim=64 cutie_object_hidden_dim=256 cutie_object_only_latent_dim=128
        cutie_object_spatial_token_enabled=true "cutie_object_spatial_graph_path=$graph"
        cutie_object_spatial_token_dim=64 cutie_object_spatial_num_heads=4 cutie_object_spatial_num_layers=2
        cutie_object_native_highres_enabled=false cutie_object_native_highres_size=128
        "cutie_object_true_entity_enabled=$true_entity"
        object_state_supervision_enabled=true object_state_supervision_collect_labels=true
        "object_state_supervision_coef=$coefficient"
        cutie_object_last_valid_memory=false cutie_object_policy_burst_plan=null
        cutie_object_belief_enabled=false cutie_object_belief_use_for_control=false
        visual_pose_checkpoint=null hydra.job.chdir=false
        "exp_name=$run_name" "hydra.run.dir=$STAGE/hydra/${task}_${arm}")
    if run_logged "${task}_${arm}_train" "$STAGE/training/${task}_${arm}.log" "${command[@]}"; then rc=0; else rc=$?; fi
    echo "TRAIN_END task=$task mode=$arm gpu=$gpu rc=$rc"
    (( rc == 0 )) || return "$rc"
    for condition in clean hard; do
        echo "EVAL_START task=$task mode=$arm condition=$condition gpu=$gpu episodes=20"
        command=(env "CUDA_VISIBLE_DEVICES=$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1
            "$PY" -B -m tdmpc2.tools.evaluate_cutie_multitask_checkpoint
            --task "$task" --backend cutie_object_only --condition "$condition"
            --runtime-config "$run_root/runtime_config.json"
            --checkpoint "$run_root/models/final.pt" --training-seed "$SEED"
            --expected-training-steps "$STEPS" --expected-training-eval-freq "$EVAL_FREQ"
            --expected-training-eval-episodes "$EVAL_EPISODES"
            --episodes 20 --env-seed 424243 --background-seed 1618034
            --planner-seed-base 8675400 --erosion-pixels 0
            --output "$STAGE/evaluations/${task}_${arm}_${condition}.json")
        if run_logged "${task}_${arm}_${condition}" "$STAGE/evaluations/${task}_${arm}_${condition}.log" "${command[@]}"; then rc=0; else rc=$?; fi
        echo "EVAL_END task=$task mode=$arm condition=$condition gpu=$gpu rc=$rc"
        (( rc == 0 )) || return "$rc"
        run_state_score "$task" "$arm" "$gpu" "$condition" "$run_root" final "$STEPS"
    done
    if [[ "$SAVE_EVAL_CHECKPOINTS" == true ]]; then
        selected="$(best_step "$run_root/eval.csv")"
        printf '%s\n' "$selected" > "$STAGE/training/${task}_${arm}.best_step"
        echo "BEST_CHECKPOINT task=$task mode=$arm step=$selected path=$run_root/models/eval_${selected}.pt"
        for condition in clean hard; do
            command=(env "CUDA_VISIBLE_DEVICES=$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1
                "$PY" -B -m tdmpc2.tools.evaluate_cutie_multitask_checkpoint
                --task "$task" --backend cutie_object_only --condition "$condition"
                --runtime-config "$run_root/runtime_config.json"
                --checkpoint "$run_root/models/eval_${selected}.pt" --checkpoint-step "$selected"
                --training-seed "$SEED" --expected-training-steps "$STEPS"
                --expected-training-eval-freq "$EVAL_FREQ"
                --expected-training-eval-episodes "$EVAL_EPISODES"
                --episodes 20 --env-seed 424243 --background-seed 1618034
                --planner-seed-base 8675400 --erosion-pixels 0
                --output "$STAGE/evaluations/${task}_${arm}_best_${condition}.json")
            if run_logged "${task}_${arm}_best_${condition}" \
                "$STAGE/evaluations/${task}_${arm}_best_${condition}.log" "${command[@]}"; then rc=0; else rc=$?; fi
            (( rc == 0 )) || return "$rc"
            run_state_score "$task" "$arm" "$gpu" "$condition" "$run_root" best "$selected"
        done
    fi
}

# Each queue is a separate session so interruption only terminates this run's
# descendants, including tracker workers. No pgrep/pkill of unrelated jobs.
if [[ "${1:-}" == --queue ]]; then
    ARM=$2; GPU=$3; COEFFICIENT=$4
    trap 'rc=$?; printf "%s\n" "$rc" > "$STAGE/queues/$ARM.rc"; exit "$rc"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    IFS=',' read -r -a selected_tasks <<< "$TASKS_CSV"
    for task in "${selected_tasks[@]}"; do
        run_task "$task" "$ARM" "$GPU" "$COEFFICIENT"
    done
    exit 0
fi

PROMOTED=0; STAGE_OWNED=0; LOCK_OWNED=0
QUEUE_PIDS=()
LOCK="${BASE}.lock"
cleanup() {
    local rc=$? pid failed
    trap - EXIT INT TERM
    for pid in "${QUEUE_PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    for pid in "${QUEUE_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
    if (( rc != 0 && PROMOTED == 0 && STAGE_OWNED == 1 )) && [[ -d "$STAGE" ]]; then
        failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
        mv -- "$STAGE" "$failed"
        echo "OBJECT_STATE_SUPERVISION_PAIRED_FAILED_ARCHIVE=$failed" >&2
    fi
    if (( LOCK_OWNED == 1 )); then rmdir -- "$LOCK" 2>/dev/null || true; fi
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo 'Unsafe run tag.' >&2; exit 2; }
[[ "$BASE" == /* && "$BASE" != / && "$BASE" != "$REPO_ROOT" ]] || { echo 'Output must be an absolute run directory.' >&2; exit 2; }
for gpu in "$GPU0" "$GPU1"; do
    [[ "$gpu" =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo "Invalid single-GPU identifier: $gpu" >&2; exit 2; }
done
[[ "$GPU0" != "$GPU1" ]] || { echo 'GPU0 and GPU1 must differ.' >&2; exit 2; }
for value in "$STEPS" "$EVAL_FREQ" "$EVAL_EPISODES" "$SEED"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo 'Budget and seed must be nonnegative integers.' >&2; exit 2; }
done
[[ "$COMPILE" == true || "$COMPILE" == false ]] || { echo 'COMPILE must be true or false.' >&2; exit 2; }
[[ "$SAVE_EVAL_CHECKPOINTS" == true || "$SAVE_EVAL_CHECKPOINTS" == false ]] || { echo 'SAVE_EVAL_CHECKPOINTS must be true or false.' >&2; exit 2; }
[[ "$PRIVILEGED_STATE_SCORE" == true || "$PRIVILEGED_STATE_SCORE" == false ]] || { echo 'PRIVILEGED_STATE_SCORE must be true or false.' >&2; exit 2; }
IFS=',' read -r -a selected_tasks <<< "$TASKS_CSV"
(( ${#selected_tasks[@]} > 0 )) || { echo 'TASKS_CSV must not be empty.' >&2; exit 2; }
for task in "${selected_tasks[@]}"; do task_fields "$task" >/dev/null || { echo "Unsupported task: $task" >&2; exit 2; }; done
for path in "$PY" "$CUTIE_CHECKPOINT"; do [[ -f "$path" ]] || { echo "Missing $path" >&2; exit 2; }; done
for path in "$VIDEO_ROOT" "$CUTIE_REPO" "$SUPPORT_ROOT_V1" "$SUPPORT_ROOT_V2"; do
    [[ -d "$path" ]] || { echo "Missing $path" >&2; exit 2; }
done
for command in setsid flock; do command -v "$command" >/dev/null || { echo "Missing $command" >&2; exit 2; }; done
[[ ! -e "$BASE" && ! -e "$STAGE" ]] || { echo "Refusing overwrite: $BASE" >&2; exit 3; }
mkdir -p -- "$(dirname -- "$BASE")"
mkdir -- "$LOCK" || { echo "Run already owned: $LOCK" >&2; exit 3; }
LOCK_OWNED=1
# Advisory GPU locks coordinate other invocations of this paired runner, not
# arbitrary third-party jobs; the operator must inspect nvidia-smi before launch.
exec 201>"/tmp/tdmpc2_object_state_supervision_gpu_${GPU0}.lock"
exec 202>"/tmp/tdmpc2_object_state_supervision_gpu_${GPU1}.lock"
flock -n 201 || { echo "GPU queue lock busy: $GPU0" >&2; exit 3; }
flock -n 202 || { echo "GPU queue lock busy: $GPU1" >&2; exit 3; }
mkdir -- "$STAGE"
STAGE_OWNED=1
mkdir -- "$STAGE/contracts" "$STAGE/training" "$STAGE/evaluations" "$STAGE/state_scores" "$STAGE/hydra" "$STAGE/queues" "$STAGE/status" "$STAGE/commands"

echo '[1/5] Source binding and static contracts (single-seed screening, no scientific go/no-go gate)'
"$PY" -m py_compile tdmpc2/common/world_model.py tdmpc2/tdmpc2.py \
    tdmpc2/envs/wrappers/cutie_object.py tdmpc2/trainer/online_trainer.py \
    tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py \
    tdmpc2/tools/score_object_state_decoder.py \
    tdmpc2/tools/aggregate_object_state_supervision_paired.py \
    >"$STAGE/contracts/py_compile.log" 2>&1
"$PY" -B -m tdmpc2.tools.aggregate_object_state_supervision_paired record \
    --stage "$STAGE" --repo "$REPO_ROOT" --run-tag "$RUN_TAG" --seed "$SEED" \
    --tasks "$TASKS_CSV" \
    --steps "$STEPS" --eval-freq "$EVAL_FREQ" --eval-episodes "$EVAL_EPISODES" \
    --gpu0 "$GPU0" --gpu1 "$GPU1" --video-root "$VIDEO_ROOT" \
    --cutie-repo "$CUTIE_REPO" --cutie-checkpoint "$CUTIE_CHECKPOINT" \
    --support-root-v1 "$SUPPORT_ROOT_V1" --support-root-v2 "$SUPPORT_ROOT_V2" \
    >"$STAGE/contracts/protocol.log" 2>&1

echo "[2/5] GPU $GPU0: auxiliary coefficient 0; Acrobot -> Cartpole -> Reacher"
setsid bash "$SCRIPT_DIR/run_object_state_supervision_paired.sh" --queue state_aux_off "$GPU0" 0 \
    >"$STAGE/queues/state_aux_off.log" 2>&1 &
QUEUE_PIDS+=("$!")
echo "QUEUE_START arm=state_aux_off gpu=$GPU0 pid=${QUEUE_PIDS[0]}"
echo "[3/5] GPU $GPU1: auxiliary coefficient 0.1; Acrobot -> Cartpole -> Reacher"
setsid bash "$SCRIPT_DIR/run_object_state_supervision_paired.sh" --queue state_aux_on "$GPU1" 0.1 \
    >"$STAGE/queues/state_aux_on.log" 2>&1 &
QUEUE_PIDS+=("$!")
echo "QUEUE_START arm=state_aux_on gpu=$GPU1 pid=${QUEUE_PIDS[1]}"
printf '%s\n' "${QUEUE_PIDS[@]}" > "$STAGE/queues/pids.txt"
echo '[4/5] Waiting for both independent queues, including clean/hard held-out evaluation'
if wait "${QUEUE_PIDS[0]}"; then RC0=0; else RC0=$?; fi
if wait "${QUEUE_PIDS[1]}"; then RC1=0; else RC1=$?; fi
QUEUE_PIDS=()
echo "QUEUE_END arm=state_aux_off gpu=$GPU0 rc=$RC0"
echo "QUEUE_END arm=state_aux_on gpu=$GPU1 rc=$RC1"
(( RC0 == 0 && RC1 == 0 )) || exit 1

echo '[5/5] Strict paired aggregation and atomic publication'
"$PY" -B -m tdmpc2.tools.aggregate_object_state_supervision_paired aggregate --stage "$STAGE" \
    >"$STAGE/aggregate.log" 2>&1
mv -- "$STAGE" "$BASE"
PROMOTED=1
rmdir -- "$LOCK"
LOCK_OWNED=0
trap - EXIT INT TERM
echo 'OBJECT_STATE_SUPERVISION_PAIRED_COMPLETE'
echo "OUTPUT_ROOT=$BASE"
echo "SUMMARY=$BASE/object_state_supervision_paired_summary.json"
