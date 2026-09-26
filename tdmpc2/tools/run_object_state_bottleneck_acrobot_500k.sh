#!/usr/bin/env bash
# Two-seed final test of a mandatory pure-visual predicted-state bottleneck.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-<DATA_PATH>/conda_envs/tdmpc2_2026/bin/python}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
SEED0="${SEED0:-12}"; SEED1="${SEED1:-13}"
STEPS="${STEPS:-500000}"; EVAL_FREQ="${EVAL_FREQ:-25000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
VIDEO_ROOT="${VIDEO_ROOT:-<BASELINE_PATH>/HRSSM-main/env/data/video_hard}"
CUTIE_REPO="${CUTIE_REPO:-<DATA_PATH>/r2_hrssm_third_party/OC-STORM}"
CUTIE_CHECKPOINT="${CUTIE_CHECKPOINT:-$CUTIE_REPO/feature_extractor/cutie/weights/cutie-small-mega.pth}"
SUPPORT="${SUPPORT:-$REPO_ROOT/datasets/cutie_multitask_support_v2_seed314159/acrobot-swingup/annotations.json}"
GRAPH="${GRAPH:-$REPO_ROOT/tdmpc2/object_graphs/acrobot_swingup.json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-object_state_bottleneck_acrobot_500k_v1_$STAMP}"
BASE="${OUTPUT_ROOT:-$REPO_ROOT/logs/_diagnostic/$RUN_TAG}"
STAGE="${BASE}.incomplete"
export PY GPU0 GPU1 SEED0 SEED1 STEPS EVAL_FREQ EVAL_EPISODES
export VIDEO_ROOT CUTIE_REPO CUTIE_CHECKPOINT SUPPORT GRAPH RUN_TAG BASE STAGE

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
print(max((r for r in rows if r[0]>0),key=lambda r:(r[1],-r[0]))[0])' "$1"
}

run_evaluation() {
	local seed=$1 gpu=$2 run_root=$3 selection=$4 condition=$5 step checkpoint suffix rc
	if [[ "$selection" == best ]]; then
		step="$(best_step "$run_root/eval.csv")"
		checkpoint="$run_root/models/eval_${step}.pt"
		suffix="_best"
	else
		step="$STEPS"; checkpoint="$run_root/models/final.pt"; suffix=""
	fi
	echo "EVAL_START seed=$seed selection=$selection step=$step condition=$condition gpu=$gpu"
	local -a command=(env "CUDA_VISIBLE_DEVICES=$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1
		"$PY" -B -m tdmpc2.tools.evaluate_cutie_multitask_checkpoint
		--task acrobot-swingup --backend cutie_object_only --condition "$condition"
		--runtime-config "$run_root/runtime_config.json" --checkpoint "$checkpoint"
		--training-seed "$seed" --expected-training-steps "$STEPS"
		--expected-training-eval-freq "$EVAL_FREQ"
		--expected-training-eval-episodes "$EVAL_EPISODES"
		--episodes 20 --env-seed 424243 --background-seed 1618034
		--planner-seed-base 8675400 --erosion-pixels 0
		--output "$STAGE/evaluations/seed${seed}${suffix}_${condition}.json")
	if [[ "$selection" == best ]]; then command+=(--checkpoint-step "$step"); fi
	if run_logged "seed${seed}_${selection}_${condition}_eval" \
		"$STAGE/evaluations/seed${seed}${suffix}_${condition}.log" "${command[@]}"; then rc=0; else rc=$?; fi
	echo "EVAL_END seed=$seed selection=$selection condition=$condition gpu=$gpu rc=$rc"
	(( rc == 0 )) || return "$rc"

	echo "STATE_SCORE_START seed=$seed selection=$selection condition=$condition gpu=$gpu"
	command=(env "CUDA_VISIBLE_DEVICES=$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1
		"$PY" -B -m tdmpc2.tools.score_object_state_decoder
		--task acrobot-swingup --backend cutie_object_only --condition "$condition"
		--runtime-config "$run_root/runtime_config.json" --checkpoint "$checkpoint"
		--training-seed "$seed" --expected-training-steps "$STEPS"
		--expected-training-eval-freq "$EVAL_FREQ"
		--expected-training-eval-episodes "$EVAL_EPISODES"
		--episodes 20 --env-seed 424243 --background-seed 1618034
		--planner-seed-base 8675400 --erosion-pixels 0
		--output "$STAGE/state_scores/seed${seed}_${selection}_${condition}.json")
	if [[ "$selection" == best ]]; then command+=(--checkpoint-step "$step"); fi
	if run_logged "seed${seed}_${selection}_${condition}_state" \
		"$STAGE/state_scores/seed${seed}_${selection}_${condition}.log" "${command[@]}"; then rc=0; else rc=$?; fi
	echo "STATE_SCORE_END seed=$seed selection=$selection condition=$condition gpu=$gpu rc=$rc"
	return "$rc"
}

run_seed() {
	local seed=$1 gpu=$2 run_name run_root rc selection condition
	run_name="${RUN_TAG}_seed${seed}"
	run_root="$REPO_ROOT/logs/acrobot-swingup/$seed/$run_name"
	[[ ! -e "$run_root" ]] || { echo "Refusing overwrite: $run_root" >&2; return 3; }
	printf '%s\n' "$run_root" > "$STAGE/training/seed${seed}.root"
	echo "TRAIN_START seed=$seed gpu=$gpu steps=$STEPS root=$run_root"
	local -a command=(env "CUDA_VISIBLE_DEVICES=$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1
		"$PY" tdmpc2/train.py task=acrobot-swingup obs=rgb model_size=5
		"steps=$STEPS" "seed=$seed" "eval_freq=$EVAL_FREQ" "eval_episodes=$EVAL_EPISODES"
		save_eval_episode_trace=true video_background_enabled=true
		"video_background_root=$VIDEO_ROOT" video_background_split=train
		video_background_manifest_dir=null visual_foreground_erosion_pixels=0
		compile=true compile_fallback_random=true enable_wandb=false
		wandb_project=none wandb_entity=none save_csv=true save_video=false
		save_agent=true save_eval_checkpoints=true checkpoint=null data_dir=null
		obs_shapes=null action_dims=null episode_lengths=null
		flat_anchor=true flat_anchor_mode=cutie_object_only flat_anchor_loss_beta=0.1
		flat_anchor_reconstruction_coef=0.0 flat_anchor_prediction_coef=0.0
		cutie_object_observation_variant=full
		cutie_object_frame_schema=cutie_query_mask_status_v1
		cutie_object_role_names=[upper_arm,lower_arm]
		cutie_object_support_schema=generic_indexed_v1
		cutie_object_allow_simulator_support=true
		cutie_object_allow_simulator_runtime=false
		cutie_object_allow_simulator_kinematics_runtime=false
		"cutie_object_repo=$CUTIE_REPO" "cutie_object_checkpoint=$CUTIE_CHECKPOINT"
		"cutie_object_support_path=$SUPPORT" cutie_object_config_dir=null
		cutie_object_device=cuda:0 cutie_object_tracker_height=448
		cutie_object_tracker_width=448 cutie_object_model_size=small
		cutie_object_prompt_radius=2.0 cutie_object_amp=true
		cutie_object_worker_timeout_seconds=180 cutie_object_num_roles=2
		cutie_object_frame_dim=590 cutie_object_stack_frames=3
		cutie_object_input_dim=1770
		cutie_object_auxiliary_target=geometry_status_full_denominator
		cutie_object_role_dim=64 cutie_object_hidden_dim=256
		cutie_object_only_latent_dim=128 cutie_object_spatial_token_enabled=true
		"cutie_object_spatial_graph_path=$GRAPH" cutie_object_spatial_token_dim=64
		cutie_object_spatial_num_heads=4 cutie_object_spatial_num_layers=2
		cutie_object_native_highres_enabled=false cutie_object_native_highres_size=128
		cutie_object_true_entity_enabled=true
		object_state_supervision_enabled=true object_state_supervision_collect_labels=true
		object_state_supervision_coef=1.0 object_state_bottleneck_enabled=true
		object_state_bottleneck_hidden_dim=128 cutie_object_last_valid_memory=false
		cutie_object_policy_burst_plan=null cutie_object_belief_enabled=false
		cutie_object_belief_use_for_control=false visual_pose_checkpoint=null
		hydra.job.chdir=false "exp_name=$run_name"
		"hydra.run.dir=$STAGE/hydra/seed${seed}")
	if run_logged "seed${seed}_train" "$STAGE/training/seed${seed}.log" "${command[@]}"; then rc=0; else rc=$?; fi
	echo "TRAIN_END seed=$seed gpu=$gpu rc=$rc"
	(( rc == 0 )) || return "$rc"
	for selection in final best; do
		for condition in clean hard; do
			run_evaluation "$seed" "$gpu" "$run_root" "$selection" "$condition"
		done
	done
}

if [[ "${1:-}" == --seed ]]; then
	run_seed "$2" "$3"
	exit $?
fi

PROMOTED=0; STAGE_OWNED=0; LOCK_OWNED=0
PIDS=()
LOCK="${BASE}.lock"
cleanup() {
	local rc=$? failed pid
	trap - EXIT INT TERM
	for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
	for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
	if (( rc != 0 && PROMOTED == 0 && STAGE_OWNED == 1 )) && [[ -d "$STAGE" ]]; then
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
		mv -- "$STAGE" "$failed"
		echo "OBJECT_STATE_BOTTLENECK_FAILED_ARCHIVE=$failed" >&2
	fi
	if (( LOCK_OWNED == 1 )); then rmdir -- "$LOCK" 2>/dev/null || true; fi
	exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_.-]+$ ]] || exit 2
[[ "$BASE" == /* && "$BASE" != / && "$BASE" != "$REPO_ROOT" ]] || exit 2
[[ "$GPU0" != "$GPU1" && "$SEED0" != "$SEED1" ]] || exit 2
for path in "$PY" "$CUTIE_CHECKPOINT" "$SUPPORT" "$GRAPH"; do [[ -f "$path" ]] || { echo "Missing $path" >&2; exit 2; }; done
for path in "$VIDEO_ROOT" "$CUTIE_REPO"; do [[ -d "$path" ]] || { echo "Missing $path" >&2; exit 2; }; done
[[ ! -e "$BASE" && ! -e "$STAGE" ]] || { echo "Refusing overwrite: $BASE" >&2; exit 3; }
mkdir -p -- "$(dirname -- "$BASE")"
mkdir -- "$LOCK"; LOCK_OWNED=1
exec 201>"/tmp/tdmpc2_object_state_bottleneck_gpu_${GPU0}.lock"
exec 202>"/tmp/tdmpc2_object_state_bottleneck_gpu_${GPU1}.lock"
flock -n 201; flock -n 202
mkdir -- "$STAGE"; STAGE_OWNED=1
mkdir -- "$STAGE/contracts" "$STAGE/training" "$STAGE/evaluations" \
	"$STAGE/state_scores" "$STAGE/hydra" "$STAGE/queues" \
	"$STAGE/status" "$STAGE/commands"

echo '[1/5] Static contracts and mandatory no-bypass CUDA test'
"$PY" -m py_compile tdmpc2/common/object_state_supervision.py \
	tdmpc2/common/layers.py tdmpc2/common/world_model.py tdmpc2/tdmpc2.py \
	tdmpc2/check_object_state_bottleneck.py \
	tdmpc2/tools/aggregate_object_state_bottleneck_500k.py \
	>"$STAGE/contracts/py_compile.log" 2>&1
env CUDA_VISIBLE_DEVICES="$GPU0" "$PY" -B tdmpc2/check_object_state_bottleneck.py \
	>"$STAGE/contracts/bottleneck_cuda.log" 2>&1

echo "[2/5] GPU $GPU0: seed $SEED0, 500k mandatory state bottleneck"
setsid bash "$0" --seed "$SEED0" "$GPU0" >"$STAGE/queues/seed${SEED0}.log" 2>&1 &
PIDS+=("$!")
echo "[3/5] GPU $GPU1: seed $SEED1, 500k mandatory state bottleneck"
setsid bash "$0" --seed "$SEED1" "$GPU1" >"$STAGE/queues/seed${SEED1}.log" 2>&1 &
PIDS+=("$!")
printf '%s\n' "${PIDS[@]}" > "$STAGE/queues/pids.txt"
echo '[4/5] Waiting for both seeds and clean/hard final plus best evaluations'
if wait "${PIDS[0]}"; then RC0=0; else RC0=$?; fi
if wait "${PIDS[1]}"; then RC1=0; else RC1=$?; fi
PIDS=()
echo "QUEUE_END seed=$SEED0 gpu=$GPU0 rc=$RC0"
echo "QUEUE_END seed=$SEED1 gpu=$GPU1 rc=$RC1"
(( RC0 == 0 && RC1 == 0 )) || exit 1

echo '[5/5] Strict aggregation and atomic publication'
"$PY" -B -m tdmpc2.tools.aggregate_object_state_bottleneck_500k \
	--stage "$STAGE" --steps "$STEPS" --eval-freq "$EVAL_FREQ" \
	--eval-episodes "$EVAL_EPISODES" --seeds "$SEED0,$SEED1" \
	>"$STAGE/aggregate.log" 2>&1
mv -- "$STAGE" "$BASE"; PROMOTED=1
rmdir -- "$LOCK"; LOCK_OWNED=0
trap - EXIT INT TERM
echo 'OBJECT_STATE_BOTTLENECK_500K_COMPLETE'
echo "OUTPUT_ROOT=$BASE"
echo "SUMMARY=$BASE/object_state_bottleneck_500k_summary.json"
