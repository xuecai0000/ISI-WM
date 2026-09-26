#!/usr/bin/env bash
# Evaluate and aggregate two already-trained Acrobot state-supervision arms.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-<DATA_PATH>/conda_envs/tdmpc2_2026/bin/python}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
VIDEO_ROOT="${VIDEO_ROOT:-<BASELINE_PATH>/HRSSM-main/env/data/video_hard}"
CUTIE_REPO="${CUTIE_REPO:-<DATA_PATH>/r2_hrssm_third_party/OC-STORM}"
CUTIE_CHECKPOINT="${CUTIE_CHECKPOINT:-$CUTIE_REPO/feature_extractor/cutie/weights/cutie-small-mega.pth}"
SUPPORT_ROOT_V1="${SUPPORT_ROOT_V1:-$REPO_ROOT/datasets/cutie_multitask_support_v1_seed314159}"
SUPPORT_ROOT_V2="${SUPPORT_ROOT_V2:-$REPO_ROOT/datasets/cutie_multitask_support_v2_seed314159}"
RUN_TAG="${RUN_TAG:?RUN_TAG must name the completed paired training run}"
BASE="${OUTPUT_ROOT:?OUTPUT_ROOT must name a new recovery result directory}"
STAGE="${BASE}.incomplete"
SEED="${SEED:-12}"
STEPS="${STEPS:-500000}"
EVAL_FREQ="${EVAL_FREQ:-25000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
TASK=acrobot-swingup

for path in "$PY" "$CUTIE_CHECKPOINT"; do [[ -f "$path" ]] || { echo "Missing $path" >&2; exit 2; }; done
for path in "$VIDEO_ROOT" "$CUTIE_REPO" "$SUPPORT_ROOT_V1" "$SUPPORT_ROOT_V2"; do
    [[ -d "$path" ]] || { echo "Missing $path" >&2; exit 2; }
done
[[ "$BASE" == /* && "$BASE" != / && "$BASE" != "$REPO_ROOT" ]] || { echo 'Unsafe output root.' >&2; exit 2; }
[[ ! -e "$BASE" && ! -e "$STAGE" ]] || { echo "Refusing overwrite: $BASE" >&2; exit 3; }

mkdir -p -- "$(dirname -- "$BASE")"
mkdir -- "$STAGE"
mkdir -- "$STAGE/contracts" "$STAGE/training" "$STAGE/evaluations" \
    "$STAGE/state_scores" "$STAGE/queues" "$STAGE/status" "$STAGE/commands"

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

evaluate_arm() {
    local arm=$1 gpu=$2 run_root condition selection step checkpoint suffix rc
    run_root="$REPO_ROOT/logs/$TASK/$SEED/${RUN_TAG}_${TASK}_${arm}"
    [[ -f "$run_root/models/final.pt" && -f "$run_root/eval.csv" && -f "$run_root/runtime_config.json" ]] || {
        echo "Incomplete training output: $run_root" >&2
        return 2
    }
    printf '0\n' > "$STAGE/status/${TASK}_${arm}_train.rc"
    step="$(best_step "$run_root/eval.csv")"
    [[ -f "$run_root/models/eval_${step}.pt" ]] || { echo "Missing best checkpoint for $arm: $step" >&2; return 2; }
    printf '%s\n' "$step" > "$STAGE/training/${TASK}_${arm}.best_step"

    for selection in final best; do
        if [[ "$selection" == final ]]; then
            checkpoint="$run_root/models/final.pt"
            step=$STEPS
            suffix=''
        else
            step="$(cat "$STAGE/training/${TASK}_${arm}.best_step")"
            checkpoint="$run_root/models/eval_${step}.pt"
            suffix='_best'
        fi
        for condition in clean hard; do
            echo "EVAL_START task=$TASK mode=$arm selection=$selection condition=$condition gpu=$gpu"
            local -a eval_command=(env "CUDA_VISIBLE_DEVICES=$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1
                "$PY" -B -m tdmpc2.tools.evaluate_cutie_multitask_checkpoint
                --task "$TASK" --backend cutie_object_only --condition "$condition"
                --runtime-config "$run_root/runtime_config.json" --checkpoint "$checkpoint"
                --training-seed "$SEED" --expected-training-steps "$STEPS"
                --expected-training-eval-freq "$EVAL_FREQ" --expected-training-eval-episodes "$EVAL_EPISODES"
                --episodes 20 --env-seed 424243 --background-seed 1618034
                --planner-seed-base 8675400 --erosion-pixels 0
                --output "$STAGE/evaluations/${TASK}_${arm}${suffix}_${condition}.json")
            [[ "$selection" == final ]] || eval_command+=(--checkpoint-step "$step")
            if run_logged "${TASK}_${arm}${suffix}_${condition}" \
                "$STAGE/evaluations/${TASK}_${arm}${suffix}_${condition}.log" "${eval_command[@]}"; then rc=0; else rc=$?; fi
            echo "EVAL_END task=$TASK mode=$arm selection=$selection condition=$condition gpu=$gpu rc=$rc"
            (( rc == 0 )) || return "$rc"

            echo "STATE_SCORE_START task=$TASK mode=$arm selection=$selection condition=$condition gpu=$gpu"
            local -a score_command=(env "CUDA_VISIBLE_DEVICES=$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1
                "$PY" -B -m tdmpc2.tools.score_object_state_decoder
                --task "$TASK" --backend cutie_object_only --condition "$condition"
                --runtime-config "$run_root/runtime_config.json" --checkpoint "$checkpoint"
                --training-seed "$SEED" --expected-training-steps "$STEPS"
                --expected-training-eval-freq "$EVAL_FREQ" --expected-training-eval-episodes "$EVAL_EPISODES"
                --episodes 20 --env-seed 424243 --background-seed 1618034
                --planner-seed-base 8675400 --erosion-pixels 0
                --output "$STAGE/state_scores/${TASK}_${arm}_${selection}_${condition}.json")
            [[ "$selection" == final ]] || score_command+=(--checkpoint-step "$step")
            if run_logged "${TASK}_${arm}_${selection}_${condition}_state_score" \
                "$STAGE/state_scores/${TASK}_${arm}_${selection}_${condition}.log" "${score_command[@]}"; then rc=0; else rc=$?; fi
            echo "STATE_SCORE_END task=$TASK mode=$arm selection=$selection condition=$condition gpu=$gpu rc=$rc"
            (( rc == 0 )) || return "$rc"
        done
    done
}

echo '[1/4] Recording Acrobot-only recovery protocol'
"$PY" -B -m tdmpc2.tools.aggregate_object_state_supervision_paired record \
    --stage "$STAGE" --repo "$REPO_ROOT" --run-tag "$RUN_TAG" --seed "$SEED" \
    --tasks "$TASK" --steps "$STEPS" --eval-freq "$EVAL_FREQ" --eval-episodes "$EVAL_EPISODES" \
    --gpu0 "$GPU0" --gpu1 "$GPU1" --video-root "$VIDEO_ROOT" \
    --cutie-repo "$CUTIE_REPO" --cutie-checkpoint "$CUTIE_CHECKPOINT" \
    --support-root-v1 "$SUPPORT_ROOT_V1" --support-root-v2 "$SUPPORT_ROOT_V2" \
    > "$STAGE/contracts/protocol.log" 2>&1

echo '[2/4] Evaluating existing final and best checkpoints on two GPUs'
evaluate_arm state_aux_off "$GPU0" > "$STAGE/queues/state_aux_off.log" 2>&1 &
pid0=$!
evaluate_arm state_aux_on "$GPU1" > "$STAGE/queues/state_aux_on.log" 2>&1 &
pid1=$!
if wait "$pid0"; then rc0=0; else rc0=$?; fi
if wait "$pid1"; then rc1=0; else rc1=$?; fi
printf '%s\n' "$rc0" > "$STAGE/queues/state_aux_off.rc"
printf '%s\n' "$rc1" > "$STAGE/queues/state_aux_on.rc"
(( rc0 == 0 && rc1 == 0 )) || { echo "Recovery evaluation failed: off=$rc0 on=$rc1" >&2; exit 1; }

echo '[3/4] Aggregating paired Acrobot result'
"$PY" -B -m tdmpc2.tools.aggregate_object_state_supervision_paired aggregate --stage "$STAGE" \
    > "$STAGE/aggregate.log" 2>&1

echo '[4/4] Publishing recovery result'
mv -- "$STAGE" "$BASE"
echo 'OBJECT_STATE_SUPERVISION_ACROBOT_RECOVERY_COMPLETE'
echo "OUTPUT_ROOT=$BASE"
echo "SUMMARY=$BASE/object_state_supervision_paired_summary.json"
