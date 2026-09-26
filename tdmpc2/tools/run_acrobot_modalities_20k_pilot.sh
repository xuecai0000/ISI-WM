#!/usr/bin/env bash
# Fixed-budget Acrobot visual/proprio/fusion screening pilot.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-<DATA_PATH>/conda_envs/tdmpc2_2026/bin/python}"
GPU="${GPU:-1}"
VIDEO_ROOT="${VIDEO_ROOT:-<BASELINE_PATH>/HRSSM-main/env/data/video_hard}"
VISUAL_POSE_CHECKPOINT="${VISUAL_POSE_CHECKPOINT:-/home/<USER>/world/tdmpc2_2026/logs/_diagnostic/acrobot_keypoint_preflight_v1_fix1_20260908_135810/checkpoints/acrobot_keypoint_detector_v1.pt}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="acrobot_modalities_20k_v1_${STAMP}"
BASE="${OUTPUT_ROOT:-$REPO_ROOT/logs/_diagnostic/$RUN_TAG}"
STAGE="${BASE}.incomplete"
SUMMARY="$STAGE/acrobot_modalities_20k_summary.json"
LOCK="$REPO_ROOT/logs/_diagnostic/.acrobot_modalities_20k.lock"

readonly TASK=acrobot-swingup
readonly SEED=8
readonly STEPS=20000
readonly EVAL_FREQ=5000
readonly EVAL_EPISODES=3

ACTIVE_PID=""
PROMOTED=0
LOCK_OWNED=0
STAGE_OWNED=0

cleanup() {
	local rc=$? failed
	trap - EXIT INT TERM
	if [[ -n "$ACTIVE_PID" ]]; then
		kill -TERM "$ACTIVE_PID" 2>/dev/null || true
		wait "$ACTIVE_PID" 2>/dev/null || true
	fi
	if (( LOCK_OWNED == 1 )); then
		rmdir -- "$LOCK" 2>/dev/null || true
	fi
	if (( rc != 0 && PROMOTED == 0 && STAGE_OWNED == 1 )) && [[ -d "$STAGE" ]]; then
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
		mv -- "$STAGE" "$failed"
		echo "ACROBOT_MODALITIES_PILOT_FAILED_ARCHIVE=$failed" >&2
	fi
	exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ -x "$PY" ]] || { echo "Python is not executable: $PY" >&2; exit 2; }
[[ -d "$VIDEO_ROOT" ]] || { echo "Video root does not exist: $VIDEO_ROOT" >&2; exit 2; }
[[ -f "$VISUAL_POSE_CHECKPOINT" ]] || {
	echo "Visual pose checkpoint does not exist: $VISUAL_POSE_CHECKPOINT" >&2
	exit 2
}
[[ ! -e "$BASE" && ! -e "$STAGE" ]] || {
	echo "Refusing to overwrite existing output: $BASE" >&2
	exit 3
}
mkdir -p -- "$REPO_ROOT/logs/_diagnostic"
if ! mkdir -- "$LOCK"; then
	echo "Another Acrobot modalities pilot owns $LOCK" >&2
	exit 3
fi
LOCK_OWNED=1
mkdir -p -- "$STAGE/contracts" "$STAGE/training" "$STAGE/hydra"
STAGE_OWNED=1

echo "[1/5] Static contracts and equal-shape zero-ablation checks"
"$PY" -B tdmpc2/check_multimodal_articulated_pose_contract.py \
	>"$STAGE/contracts/multimodal.log" 2>&1
"$PY" -m py_compile \
	tdmpc2/common/multimodal_articulated_pose.py \
	tdmpc2/envs/wrappers/multimodal_articulated_pose.py \
	tdmpc2/common/layers.py tdmpc2/common/world_model.py tdmpc2/tdmpc2.py \
	>"$STAGE/contracts/py_compile.log" 2>&1

echo "[2/5] Real Acrobot environment smoke for visual, proprio, and fusion"
for mode in visual_only proprio_only fusion; do
	env CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl \
		"$PY" -B tdmpc2/check_multimodal_articulated_pose_contract.py \
		--real-env-smoke --checkpoint "$VISUAL_POSE_CHECKPOINT" --mode "$mode" \
		>"$STAGE/contracts/real_env_${mode}.log" 2>&1
done

COMMON_ARGS=(
	"task=$TASK" obs=rgb model_size=5 "steps=$STEPS" "seed=$SEED"
	"eval_freq=$EVAL_FREQ" "eval_episodes=$EVAL_EPISODES"
	video_background_enabled=true "video_background_root=$VIDEO_ROOT"
	video_background_split=train video_background_manifest_dir=null
	visual_foreground_erosion_pixels=0
	compile=true compile_fallback_random=true
	enable_wandb=false wandb_project=none wandb_entity=none
	save_csv=true save_video=false save_agent=true checkpoint=null data_dir=null
	obs_shapes=null action_dims=null episode_lengths=null
	flat_anchor=true flat_anchor_mode=cutie_object_only flat_anchor_loss_beta=0.1
	cutie_object_observation_variant=multimodal_articulated_pose
	cutie_object_frame_schema=acrobot_multimodal_articulated_pose_v1
	cutie_object_role_names=[upper_arm,lower_arm]
	cutie_object_allow_simulator_runtime=false
	cutie_object_allow_simulator_support=false
	cutie_object_repo=null cutie_object_checkpoint=null
	cutie_object_support_path=null cutie_object_config_dir=null
	cutie_object_num_roles=2 cutie_object_frame_dim=25
	cutie_object_stack_frames=1 cutie_object_input_dim=25
	cutie_object_auxiliary_target=full_descriptor
	cutie_object_role_dim=64 cutie_object_hidden_dim=256
	cutie_object_only_latent_dim=128
	cutie_object_native_highres_enabled=false
	cutie_object_last_valid_memory=false cutie_object_policy_burst_plan=null
	cutie_object_belief_enabled=false cutie_object_belief_use_for_control=false
	"visual_pose_checkpoint=$VISUAL_POSE_CHECKPOINT"
	visual_pose_device=cuda:0 visual_pose_history=4 visual_pose_image_size=64
	visual_pose_control_dt=0.04 visual_pose_confidence_threshold=0.0
	visual_pose_use_cutie_mask=false multimodal_proprio_velocity_scale=10.0
	hydra.job.chdir=false
)

run_mode() {
	local mode=$1 privilege=$2 run_name run_root rc
	run_name="${RUN_TAG}_${mode}"
	run_root="$REPO_ROOT/logs/$TASK/$SEED/$run_name"
	[[ ! -e "$run_root" ]] || {
		echo "Refusing to overwrite training root: $run_root" >&2
		return 3
	}
	echo "TRAIN_START mode=$mode gpu=$GPU root=$run_root"
	env CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" tdmpc2/train.py "${COMMON_ARGS[@]}" \
		"articulated_pose_modalities=$mode" \
		"cutie_object_allow_simulator_kinematics_runtime=$privilege" \
		"exp_name=$run_name" "hydra.run.dir=$STAGE/hydra/$mode" \
		>"$STAGE/training/${mode}.log" 2>&1 &
	ACTIVE_PID=$!
	if wait "$ACTIVE_PID"; then rc=0; else rc=$?; fi
	ACTIVE_PID=""
	printf '%s\n' "$rc" >"$STAGE/training/${mode}.rc"
	echo "TRAIN_END mode=$mode gpu=$GPU rc=$rc"
	(( rc == 0 )) || return "$rc"
}

echo "[3/5] Training visual-only and proprio-only controls"
VISUAL_ROOT="$REPO_ROOT/logs/$TASK/$SEED/${RUN_TAG}_visual_only"
PROPRIO_ROOT="$REPO_ROOT/logs/$TASK/$SEED/${RUN_TAG}_proprio_only"
run_mode visual_only false
run_mode proprio_only true

echo "[4/5] Training visual plus proprio fusion"
FUSION_ROOT="$REPO_ROOT/logs/$TASK/$SEED/${RUN_TAG}_fusion"
run_mode fusion true

echo "[5/5] Strict aggregation and atomic publication"
"$PY" -B -m tdmpc2.tools.aggregate_acrobot_modalities_pilot \
	--visual-only-root "$VISUAL_ROOT" \
	--proprio-only-root "$PROPRIO_ROOT" \
	--fusion-root "$FUSION_ROOT" --output "$SUMMARY" \
	>"$STAGE/aggregate.log" 2>&1

[[ ! -e "$BASE" ]] || { echo "Publication target appeared: $BASE" >&2; exit 3; }
mv -- "$STAGE" "$BASE"
PROMOTED=1
rmdir -- "$LOCK"
LOCK_OWNED=0
trap - EXIT INT TERM
echo "ACROBOT_MODALITIES_20K_PILOT_COMPLETE"
echo "OUTPUT_ROOT=$BASE"
echo "SUMMARY=$BASE/acrobot_modalities_20k_summary.json"
