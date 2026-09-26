#!/usr/bin/env bash
# Paired long-budget Acrobot screen on two GPUs: factorized versus proprio-only.

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
SUPPORT="${SUPPORT:-$REPO_ROOT/datasets/cutie_multitask_support_v2_seed314159/acrobot-swingup/annotations.json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="acrobot_factorized_proprio_500k_v1_${STAMP}"
BASE="${OUTPUT_ROOT:-$REPO_ROOT/logs/_diagnostic/$RUN_TAG}"
STAGE="${BASE}.incomplete"
LOCK="$REPO_ROOT/logs/_diagnostic/.acrobot_factorized_proprio_500k.lock"

readonly TASK=acrobot-swingup
readonly SEED=10
readonly STEPS=500000
readonly EVAL_FREQ=25000
readonly EVAL_EPISODES=10
readonly HELD_OUT_EPISODES=20

PROMOTED=0
LOCK_OWNED=0
STAGE_OWNED=0
ARM_PIDS=()
cleanup() {
	local rc=$? failed pid
	trap - EXIT INT TERM
	for pid in "${ARM_PIDS[@]:-}"; do
		kill -TERM "$pid" 2>/dev/null || true
	done
	for pid in "${ARM_PIDS[@]:-}"; do
		wait "$pid" 2>/dev/null || true
	done
	if (( LOCK_OWNED == 1 )); then rmdir -- "$LOCK" 2>/dev/null || true; fi
	if (( rc != 0 && PROMOTED == 0 && STAGE_OWNED == 1 )) && [[ -d "$STAGE" ]]; then
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
		mv -- "$STAGE" "$failed"
		echo "ACROBOT_FACTORIZED_PROPRIO_500K_FAILED_ARCHIVE=$failed" >&2
	fi
	exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ "$GPU0" != "$GPU1" ]] || { echo "GPU0 and GPU1 must differ." >&2; exit 2; }
for path in "$PY" "$CUTIE_CHECKPOINT" "$SUPPORT"; do
	[[ -f "$path" || -x "$path" ]] || { echo "Required file missing: $path" >&2; exit 2; }
done
for path in "$CUTIE_REPO" "$VIDEO_ROOT"; do
	[[ -d "$path" ]] || { echo "Required directory missing: $path" >&2; exit 2; }
done
[[ ! -e "$BASE" && ! -e "$STAGE" ]] || { echo "Refusing overwrite: $BASE" >&2; exit 3; }
mkdir -p -- "$REPO_ROOT/logs/_diagnostic"
mkdir -- "$LOCK" || { echo "Another paired 500k run owns $LOCK" >&2; exit 3; }
LOCK_OWNED=1
mkdir -p -- "$STAGE/contracts" "$STAGE/training" "$STAGE/evaluations" "$STAGE/hydra"
STAGE_OWNED=1

echo "[1/5] Static contracts and periodic-checkpoint support"
"$PY" -m py_compile \
	tdmpc2/trainer/online_trainer.py \
	tdmpc2/common/cutie_proprio.py \
	tdmpc2/envs/wrappers/cutie_proprio.py \
	tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py \
	tdmpc2/tools/aggregate_acrobot_factorized_proprio_500k.py \
	tdmpc2/check_acrobot_factorized_proprio_500k_contract.py \
	tdmpc2/check_factorized_cutie_proprio_update.py \
	>"$STAGE/contracts/py_compile.log" 2>&1
"$PY" -B tdmpc2/check_acrobot_factorized_proprio_500k_contract.py \
	>"$STAGE/contracts/long_budget_cpu.log" 2>&1
"$PY" -B tdmpc2/check_cutie_proprio_contract.py \
	>"$STAGE/contracts/cutie_proprio.log" 2>&1
env CUDA_VISIBLE_DEVICES="$GPU0" MUJOCO_GL=egl \
	"$PY" -B tdmpc2/check_factorized_cutie_proprio_update.py \
	>"$STAGE/contracts/factorized_update.log" 2>&1

COMMON_ARGS=(
	"task=$TASK" obs=rgb model_size=5 "steps=$STEPS" "seed=$SEED"
	"eval_freq=$EVAL_FREQ" "eval_episodes=$EVAL_EPISODES"
	save_eval_episode_trace=true save_eval_checkpoints=true
	video_background_enabled=true "video_background_root=$VIDEO_ROOT"
	video_background_split=train video_background_manifest_dir=null
	visual_foreground_erosion_pixels=0
	compile=true compile_fallback_random=true
	enable_wandb=false wandb_project=none wandb_entity=none
	save_csv=true save_video=false save_agent=true checkpoint=null data_dir=null
	obs_shapes=null action_dims=null episode_lengths=null
	flat_anchor=true flat_anchor_mode=cutie_object_only flat_anchor_loss_beta=0.1
	cutie_object_observation_variant=cutie_proprio
	cutie_object_frame_schema=cutie_query_mask_status_plus_proprio_v1
	cutie_object_role_names=[upper_arm,lower_arm]
	cutie_object_support_schema=generic_indexed_v1
	cutie_object_allow_simulator_support=true
	cutie_object_allow_simulator_runtime=false
	cutie_object_allow_simulator_kinematics_runtime=true
	"cutie_object_repo=$CUTIE_REPO" "cutie_object_checkpoint=$CUTIE_CHECKPOINT"
	"cutie_object_support_path=$SUPPORT" cutie_object_config_dir=null
	cutie_object_device=cuda:0 cutie_object_tracker_height=448
	cutie_object_tracker_width=448 cutie_object_model_size=small
	cutie_object_prompt_radius=2.0 cutie_object_amp=true
	cutie_object_worker_timeout_seconds=180
	cutie_object_num_roles=2 cutie_object_frame_dim=1774
	cutie_object_stack_frames=1 cutie_object_input_dim=1774
	cutie_object_auxiliary_target=full_descriptor
	cutie_object_role_dim=64 cutie_object_hidden_dim=256
	cutie_object_only_latent_dim=128
	cutie_object_native_highres_enabled=false cutie_object_native_highres_size=128
	cutie_object_last_valid_memory=false cutie_object_policy_burst_plan=null
	cutie_object_belief_enabled=false cutie_object_belief_use_for_control=false
	cutie_proprio_velocity_scale=10.0 visual_pose_checkpoint=null
	hydra.job.chdir=false
)

backend_for_mode() {
	case "$1" in
		factorized) printf '%s\n' cutie_proprio_factorized ;;
		factorized_proprio_only) printf '%s\n' cutie_proprio_factorized_proprio_only ;;
		*) return 2 ;;
	esac
}

best_step() {
	"$PY" -c 'import csv,sys
rows=[(int(float(r["step"])),float(r["episode_reward"])) for r in csv.DictReader(open(sys.argv[1],newline="",encoding="utf-8"))]
rows=[r for r in rows if r[0]>0]
print(max(rows,key=lambda r:(r[1],-r[0]))[0])' "$1"
}

evaluate_selection() {
	local mode=$1 gpu=$2 run_root=$3 selection=$4 step=$5 backend checkpoint
	backend="$(backend_for_mode "$mode")"
	if [[ "$selection" == best ]]; then
		checkpoint="$run_root/models/eval_${step}.pt"
	else
		checkpoint="$run_root/models/final.pt"
	fi
	for condition in clean hard; do
		local output="$STAGE/evaluations/${mode}_${selection}_${condition}.json"
		local log="$STAGE/evaluations/${mode}_${selection}_${condition}.log"
		echo "EVAL_START mode=$mode selection=$selection step=$step condition=$condition gpu=$gpu"
		args=(
			--task "$TASK" --backend "$backend" --condition "$condition"
			--runtime-config "$run_root/runtime_config.json"
			--checkpoint "$checkpoint" --training-seed "$SEED"
			--expected-training-steps "$STEPS"
			--expected-training-eval-freq "$EVAL_FREQ"
			--expected-training-eval-episodes "$EVAL_EPISODES"
			--episodes "$HELD_OUT_EPISODES"
			--env-seed 424243 --background-seed 1618034
			--planner-seed-base 8675400 --erosion-pixels 0 --output "$output"
		)
		if [[ "$selection" == best ]]; then args+=(--checkpoint-step "$step"); fi
		env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
			"$PY" -B -m tdmpc2.tools.evaluate_cutie_multitask_checkpoint \
			"${args[@]}" >"$log" 2>&1
		echo "EVAL_END mode=$mode selection=$selection step=$step condition=$condition gpu=$gpu rc=0"
	done
}

run_arm() (
	set -Eeuo pipefail
	local mode=$1 gpu=$2 run_name run_root selected rc
	local active_pid=""
	arm_cleanup() {
		local arm_rc=$?
		trap - EXIT INT TERM
		if [[ -n "$active_pid" ]]; then
			kill -TERM "$active_pid" 2>/dev/null || true
			wait "$active_pid" 2>/dev/null || true
		fi
		exit "$arm_rc"
	}
	trap arm_cleanup EXIT
	trap 'exit 130' INT
	trap 'exit 143' TERM
	run_name="${RUN_TAG}_${mode}"
	run_root="$REPO_ROOT/logs/$TASK/$SEED/$run_name"
	[[ ! -e "$run_root" ]] || { echo "Refusing overwrite: $run_root" >&2; exit 3; }
	echo "TRAIN_START mode=$mode gpu=$gpu root=$run_root"
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" tdmpc2/train.py "${COMMON_ARGS[@]}" \
		"cutie_proprio_mode=$mode" \
		"exp_name=$run_name" "hydra.run.dir=$STAGE/hydra/$mode" \
		>"$STAGE/training/${mode}.log" 2>&1 &
	active_pid=$!
	set +e
	wait "$active_pid"; rc=$?
	set -e
	active_pid=""
	printf '%s\n' "$rc" >"$STAGE/training/${mode}.rc"
	echo "TRAIN_END mode=$mode gpu=$gpu rc=$rc"
	(( rc == 0 )) || exit "$rc"
	selected="$(best_step "$run_root/eval.csv")"
	printf '%s\n' "$selected" >"$STAGE/training/${mode}.best_step"
	echo "BEST_CHECKPOINT mode=$mode step=$selected path=$run_root/models/eval_${selected}.pt"
	evaluate_selection "$mode" "$gpu" "$run_root" best "$selected"
	evaluate_selection "$mode" "$gpu" "$run_root" final "$STEPS"
	trap - EXIT INT TERM
)

echo "[2/5] GPU $GPU0: factorized Cutie plus proprio for 500k"
run_arm factorized "$GPU0" > >(tee "$STAGE/training/gpu${GPU0}_factorized.log") 2>&1 &
ARM_PIDS+=("$!")

echo "[3/5] GPU $GPU1: architecture-matched factorized proprio-only for 500k"
run_arm factorized_proprio_only "$GPU1" > >(tee "$STAGE/training/gpu${GPU1}_factorized_proprio_only.log") 2>&1 &
ARM_PIDS+=("$!")

echo "[4/5] Waiting for both paired arms and held-out evaluations"
set +e
wait "${ARM_PIDS[0]}"; RC0=$?
wait "${ARM_PIDS[1]}"; RC1=$?
set -e
ARM_PIDS=()
echo "GPU_ARM_END mode=factorized gpu=$GPU0 rc=$RC0"
echo "GPU_ARM_END mode=factorized_proprio_only gpu=$GPU1 rc=$RC1"
(( RC0 == 0 && RC1 == 0 )) || exit 1

echo "[5/5] Strict aggregation and atomic publication"
FACTORIZED_ROOT="$REPO_ROOT/logs/$TASK/$SEED/${RUN_TAG}_factorized"
PROPRIO_ROOT="$REPO_ROOT/logs/$TASK/$SEED/${RUN_TAG}_factorized_proprio_only"
"$PY" -B -m tdmpc2.tools.aggregate_acrobot_factorized_proprio_500k \
	--factorized-root "$FACTORIZED_ROOT" \
	--factorized-best-clean "$STAGE/evaluations/factorized_best_clean.json" \
	--factorized-best-hard "$STAGE/evaluations/factorized_best_hard.json" \
	--factorized-final-clean "$STAGE/evaluations/factorized_final_clean.json" \
	--factorized-final-hard "$STAGE/evaluations/factorized_final_hard.json" \
	--factorized-proprio-only-root "$PROPRIO_ROOT" \
	--factorized-proprio-only-best-clean "$STAGE/evaluations/factorized_proprio_only_best_clean.json" \
	--factorized-proprio-only-best-hard "$STAGE/evaluations/factorized_proprio_only_best_hard.json" \
	--factorized-proprio-only-final-clean "$STAGE/evaluations/factorized_proprio_only_final_clean.json" \
	--factorized-proprio-only-final-hard "$STAGE/evaluations/factorized_proprio_only_final_hard.json" \
	--output "$STAGE/acrobot_factorized_proprio_500k_summary.json" \
	>"$STAGE/aggregate.log" 2>&1
mv -- "$STAGE" "$BASE"
PROMOTED=1
rmdir -- "$LOCK"
LOCK_OWNED=0
trap - EXIT INT TERM
echo "ACROBOT_FACTORIZED_PROPRIO_500K_COMPLETE"
echo "OUTPUT_ROOT=$BASE"
echo "SUMMARY=$BASE/acrobot_factorized_proprio_500k_summary.json"
