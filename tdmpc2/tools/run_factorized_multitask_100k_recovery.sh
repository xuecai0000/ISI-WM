#!/usr/bin/env bash
# Resume evaluation and the queued Cartpole run after an evaluator-only failure.

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-<DATA_PATH>/conda_envs/tdmpc2_2026/bin/python}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
VIDEO_ROOT="${VIDEO_ROOT:-<BASELINE_PATH>/HRSSM-main/env/data/video_hard}"
CUTIE_REPO="${CUTIE_REPO:-<DATA_PATH>/r2_hrssm_third_party/OC-STORM}"
CUTIE_CHECKPOINT="${CUTIE_CHECKPOINT:-$CUTIE_REPO/feature_extractor/cutie/weights/cutie-small-mega.pth}"
SOURCE_TAG="factorized_multitask_100k_v1_20260909_013207"
BASE="${OUTPUT_ROOT:-$REPO_ROOT/logs/_diagnostic/${SOURCE_TAG}_recovery_v2}"
STAGE="${BASE}.incomplete"
LOCK="$REPO_ROOT/logs/_diagnostic/.factorized_multitask_100k_recovery.lock"
SEED=10; STEPS=100000; EVAL_FREQ=10000; EVAL_EPISODES=10
ACROBOT_ROOT="$REPO_ROOT/logs/acrobot-swingup/$SEED/${SOURCE_TAG}_acrobot-swingup"
REACHER_ROOT="$REPO_ROOT/logs/reacher-visual-small/$SEED/${SOURCE_TAG}_reacher-visual-small"
CARTPOLE_ROOT="$REPO_ROOT/logs/cartpole-swingup/$SEED/${SOURCE_TAG}_cartpole-swingup"

PIDS=(); PROMOTED=0; LOCK_OWNED=0
cleanup() {
	local rc=$? failed pid
	trap - EXIT INT TERM
	for pid in "${PIDS[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
	for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
	if (( LOCK_OWNED )); then rmdir -- "$LOCK" 2>/dev/null || true; fi
	if (( rc != 0 && PROMOTED == 0 )) && [[ -d "$STAGE" ]]; then
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
		mv -- "$STAGE" "$failed"
		echo "FACTORIZED_MULTITASK_RECOVERY_FAILED_ARCHIVE=$failed" >&2
	fi
	exit "$rc"
}
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM

for root in "$ACROBOT_ROOT" "$REACHER_ROOT"; do
	[[ -f "$root/runtime_config.json" && -f "$root/models/final.pt" ]] || {
		echo "Completed source run missing: $root" >&2; exit 2;
	}
done
[[ ! -e "$CARTPOLE_ROOT" ]] || { echo "Cartpole target exists: $CARTPOLE_ROOT" >&2; exit 3; }
[[ ! -e "$BASE" && ! -e "$STAGE" ]] || { echo "Recovery target exists: $BASE" >&2; exit 3; }
mkdir -- "$LOCK"; LOCK_OWNED=1
mkdir -p -- "$STAGE/contracts" "$STAGE/training" "$STAGE/evaluations" "$STAGE/hydra"

echo "[1/4] Evaluator compilation and immutable source checkpoint checks"
"$PY" -m py_compile tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py \
	tdmpc2/tools/aggregate_factorized_multitask_100k.py \
	>"$STAGE/contracts/py_compile.log" 2>&1

evaluate_task() {
	local task=$1 root=$2 gpu=$3 condition rc
	for condition in clean hard; do
		echo "EVAL_START task=$task condition=$condition gpu=$gpu episodes=20"
		env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
			"$PY" -B -m tdmpc2.tools.evaluate_cutie_multitask_checkpoint \
			--task "$task" --backend cutie_proprio_factorized --condition "$condition" \
			--runtime-config "$root/runtime_config.json" --checkpoint "$root/models/final.pt" \
			--training-seed "$SEED" --expected-training-steps "$STEPS" \
			--expected-training-eval-freq "$EVAL_FREQ" \
			--expected-training-eval-episodes "$EVAL_EPISODES" \
			--episodes 20 --env-seed 424243 --background-seed 1618034 \
			--planner-seed-base 8675400 --erosion-pixels 0 \
			--output "$STAGE/evaluations/${task}_${condition}.json" \
			>"$STAGE/evaluations/${task}_${condition}.log" 2>&1
		rc=$?; echo "EVAL_END task=$task condition=$condition gpu=$gpu rc=$rc"
		(( rc == 0 )) || return "$rc"
	done
}

train_cartpole() {
	local support="$REPO_ROOT/datasets/cutie_multitask_support_v1_seed314159/cartpole-swingup/annotations.json" rc
	echo "TRAIN_START task=cartpole-swingup mode=factorized gpu=$GPU0 root=$CARTPOLE_ROOT"
	env CUDA_VISIBLE_DEVICES="$GPU0" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" tdmpc2/train.py task=cartpole-swingup obs=rgb model_size=5 \
		steps="$STEPS" seed="$SEED" eval_freq="$EVAL_FREQ" eval_episodes="$EVAL_EPISODES" \
		save_eval_episode_trace=true video_background_enabled=true \
		video_background_root="$VIDEO_ROOT" video_background_split=train \
		video_background_manifest_dir=null visual_foreground_erosion_pixels=0 \
		compile=true compile_fallback_random=true enable_wandb=false \
		wandb_project=none wandb_entity=none save_csv=true save_video=false \
		save_agent=true checkpoint=null data_dir=null obs_shapes=null action_dims=null \
		episode_lengths=null flat_anchor=true flat_anchor_mode=cutie_object_only \
		flat_anchor_loss_beta=0.1 cutie_object_observation_variant=cutie_proprio \
		cutie_object_frame_schema=cutie_query_mask_status_plus_proprio_v1 \
		cutie_object_role_names=[cart,pole] cutie_object_support_schema=generic_indexed_v1 \
		cutie_object_allow_simulator_support=true cutie_object_allow_simulator_runtime=false \
		cutie_object_allow_simulator_kinematics_runtime=true \
		cutie_object_repo="$CUTIE_REPO" cutie_object_checkpoint="$CUTIE_CHECKPOINT" \
		cutie_object_support_path="$support" cutie_object_config_dir=null \
		cutie_object_device=cuda:0 cutie_object_tracker_height=448 \
		cutie_object_tracker_width=448 cutie_object_model_size=small \
		cutie_object_prompt_radius=2.0 cutie_object_amp=true \
		cutie_object_worker_timeout_seconds=180 cutie_object_num_roles=2 \
		cutie_object_frame_dim=1774 cutie_object_stack_frames=1 \
		cutie_object_input_dim=1774 cutie_object_auxiliary_target=full_descriptor \
		cutie_object_role_dim=64 cutie_object_hidden_dim=256 \
		cutie_object_only_latent_dim=128 cutie_object_native_highres_enabled=false \
		cutie_object_native_highres_size=128 cutie_object_last_valid_memory=false \
		cutie_object_policy_burst_plan=null cutie_object_belief_enabled=false \
		cutie_object_belief_use_for_control=false cutie_proprio_mode=factorized \
		cutie_proprio_velocity_scale=10.0 visual_pose_checkpoint=null hydra.job.chdir=false \
		exp_name="${SOURCE_TAG}_cartpole-swingup" \
		hydra.run.dir="$STAGE/hydra/cartpole-swingup" \
		>"$STAGE/training/cartpole-swingup.log" 2>&1
	rc=$?; echo "TRAIN_END task=cartpole-swingup mode=factorized gpu=$GPU0 rc=$rc"
	(( rc == 0 )) || return "$rc"
	evaluate_task cartpole-swingup "$CARTPOLE_ROOT" "$GPU0"
}

echo "[2/4] GPU $GPU0: Acrobot evaluation, then Cartpole train and evaluation"
(
	evaluate_task acrobot-swingup "$ACROBOT_ROOT" "$GPU0"
	train_cartpole
) > >(tee "$STAGE/training/gpu${GPU0}_recovery.log") 2>&1 & PIDS+=("$!")

echo "[3/4] GPU $GPU1: Reacher clean and held-out-hard evaluation"
(
	evaluate_task reacher-visual-small "$REACHER_ROOT" "$GPU1"
) > >(tee "$STAGE/training/gpu${GPU1}_recovery.log") 2>&1 & PIDS+=("$!")

set +e; wait "${PIDS[0]}"; RC0=$?; wait "${PIDS[1]}"; RC1=$?; set -e
PIDS=(); echo "GPU_QUEUE_END gpu=$GPU0 rc=$RC0"; echo "GPU_QUEUE_END gpu=$GPU1 rc=$RC1"
(( RC0 == 0 && RC1 == 0 )) || exit 1

echo "[4/4] Strict aggregation and atomic publication"
"$PY" -B -m tdmpc2.tools.aggregate_factorized_multitask_100k \
	--acrobot_swingup-root "$ACROBOT_ROOT" \
	--acrobot_swingup-clean "$STAGE/evaluations/acrobot-swingup_clean.json" \
	--acrobot_swingup-hard "$STAGE/evaluations/acrobot-swingup_hard.json" \
	--cartpole_swingup-root "$CARTPOLE_ROOT" \
	--cartpole_swingup-clean "$STAGE/evaluations/cartpole-swingup_clean.json" \
	--cartpole_swingup-hard "$STAGE/evaluations/cartpole-swingup_hard.json" \
	--reacher_visual_small-root "$REACHER_ROOT" \
	--reacher_visual_small-clean "$STAGE/evaluations/reacher-visual-small_clean.json" \
	--reacher_visual_small-hard "$STAGE/evaluations/reacher-visual-small_hard.json" \
	--output "$STAGE/factorized_multitask_100k_summary.json" \
	>"$STAGE/aggregate.log" 2>&1
mv -- "$STAGE" "$BASE"; PROMOTED=1; rmdir -- "$LOCK"; LOCK_OWNED=0
trap - EXIT INT TERM
echo "FACTORIZED_MULTITASK_RECOVERY_COMPLETE"
echo "OUTPUT_ROOT=$BASE"
echo "SUMMARY=$BASE/factorized_multitask_100k_summary.json"
