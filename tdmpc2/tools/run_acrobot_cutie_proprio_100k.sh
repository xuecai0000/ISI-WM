#!/usr/bin/env bash
# Paired 100k Acrobot comparison: proprio-only versus safe Cutie+proprio fusion.

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-<DATA_PATH>/conda_envs/tdmpc2_2026/bin/python}"
GPU="${GPU:-1}"
VIDEO_ROOT="${VIDEO_ROOT:-<BASELINE_PATH>/HRSSM-main/env/data/video_hard}"
CUTIE_REPO="${CUTIE_REPO:-<DATA_PATH>/r2_hrssm_third_party/OC-STORM}"
CUTIE_CHECKPOINT="${CUTIE_CHECKPOINT:-$CUTIE_REPO/feature_extractor/cutie/weights/cutie-small-mega.pth}"
SUPPORT="${SUPPORT:-$REPO_ROOT/datasets/cutie_multitask_support_v2_seed314159/acrobot-swingup/annotations.json}"
SOURCE_20K_SUMMARY="${SOURCE_20K_SUMMARY:-$REPO_ROOT/logs/_diagnostic/acrobot_cutie_proprio_20k_v1_20260908_212536/acrobot_cutie_proprio_20k_summary.json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="acrobot_cutie_proprio_100k_v1_${STAMP}"
BASE="${OUTPUT_ROOT:-$REPO_ROOT/logs/_diagnostic/$RUN_TAG}"
STAGE="${BASE}.incomplete"
SUMMARY="$STAGE/acrobot_cutie_proprio_100k_summary.json"
LOCK="$REPO_ROOT/logs/_diagnostic/.acrobot_cutie_proprio_100k.lock"

readonly TASK=acrobot-swingup
readonly SEED=9
readonly STEPS=100000
readonly EVAL_FREQ=10000
readonly EVAL_EPISODES=10

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
	if (( LOCK_OWNED == 1 )); then rmdir -- "$LOCK" 2>/dev/null || true; fi
	if (( rc != 0 && PROMOTED == 0 && STAGE_OWNED == 1 )) && [[ -d "$STAGE" ]]; then
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
		mv -- "$STAGE" "$failed"
		echo "ACROBOT_CUTIE_PROPRIO_100K_FAILED_ARCHIVE=$failed" >&2
	fi
	exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for path in "$PY" "$CUTIE_CHECKPOINT" "$SUPPORT" "$SOURCE_20K_SUMMARY"; do
	[[ -f "$path" || -x "$path" ]] || { echo "Required file missing: $path" >&2; exit 2; }
done
[[ -d "$CUTIE_REPO" ]] || { echo "Cutie repo missing: $CUTIE_REPO" >&2; exit 2; }
[[ -d "$VIDEO_ROOT" ]] || { echo "Video root missing: $VIDEO_ROOT" >&2; exit 2; }
[[ ! -e "$BASE" && ! -e "$STAGE" ]] || { echo "Refusing overwrite: $BASE" >&2; exit 3; }
mkdir -p -- "$REPO_ROOT/logs/_diagnostic"
mkdir -- "$LOCK" || { echo "Another Cutie-proprio 100k run owns $LOCK" >&2; exit 3; }
LOCK_OWNED=1
mkdir -p -- "$STAGE/contracts" "$STAGE/training" "$STAGE/hydra"
STAGE_OWNED=1

echo "[1/4] Static contracts and exact safe-fusion initialization"
"$PY" -m py_compile tdmpc2/trainer/online_trainer.py \
	tdmpc2/tools/aggregate_acrobot_cutie_proprio_100k.py \
	tdmpc2/common/cutie_proprio.py tdmpc2/envs/wrappers/cutie_proprio.py \
	tdmpc2/check_cutie_proprio_contract.py tdmpc2/check_cutie_proprio_update.py \
	>"$STAGE/contracts/py_compile.log" 2>&1
"$PY" -B tdmpc2/check_cutie_proprio_contract.py \
	>"$STAGE/contracts/cutie_proprio.log" 2>&1
env CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl \
	"$PY" -B tdmpc2/check_cutie_proprio_update.py \
	>"$STAGE/contracts/cutie_proprio_update.log" 2>&1

COMMON_ARGS=(
	"task=$TASK" obs=rgb model_size=5 "steps=$STEPS" "seed=$SEED"
	"eval_freq=$EVAL_FREQ" "eval_episodes=$EVAL_EPISODES"
	save_eval_episode_trace=true
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

run_mode() {
	local mode=$1 run_name run_root rc
	run_name="${RUN_TAG}_${mode}"
	run_root="$REPO_ROOT/logs/$TASK/$SEED/$run_name"
	[[ ! -e "$run_root" ]] || { echo "Refusing overwrite: $run_root" >&2; return 3; }
	echo "TRAIN_START mode=$mode gpu=$GPU root=$run_root"
	env CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" tdmpc2/train.py "${COMMON_ARGS[@]}" \
		"cutie_proprio_mode=$mode" \
		"exp_name=$run_name" "hydra.run.dir=$STAGE/hydra/$mode" \
		>"$STAGE/training/${mode}.log" 2>&1 &
	ACTIVE_PID=$!
	if wait "$ACTIVE_PID"; then rc=0; else rc=$?; fi
	ACTIVE_PID=""
	printf '%s\n' "$rc" >"$STAGE/training/${mode}.rc"
	echo "TRAIN_END mode=$mode gpu=$GPU rc=$rc"
	(( rc == 0 )) || return "$rc"
}

echo "[2/4] Training proprio-only for 100k steps with 10-episode evaluations"
PROPRIO_ROOT="$REPO_ROOT/logs/$TASK/$SEED/${RUN_TAG}_proprio_only"
run_mode proprio_only
echo "[3/4] Training safe Cutie plus proprio fusion for 100k steps"
FUSION_ROOT="$REPO_ROOT/logs/$TASK/$SEED/${RUN_TAG}_fusion"
run_mode fusion

echo "[4/4] Strict aggregation and atomic publication"
"$PY" -B -m tdmpc2.tools.aggregate_acrobot_cutie_proprio_100k \
	--proprio-only-root "$PROPRIO_ROOT" --fusion-root "$FUSION_ROOT" \
	--source-20k-summary "$SOURCE_20K_SUMMARY" --output "$SUMMARY" \
	>"$STAGE/aggregate.log" 2>&1
[[ ! -e "$BASE" ]] || { echo "Publication target appeared: $BASE" >&2; exit 3; }
mv -- "$STAGE" "$BASE"
PROMOTED=1
rmdir -- "$LOCK"
LOCK_OWNED=0
trap - EXIT INT TERM
echo "ACROBOT_CUTIE_PROPRIO_100K_COMPLETE"
echo "OUTPUT_ROOT=$BASE"
echo "SUMMARY=$BASE/acrobot_cutie_proprio_100k_summary.json"
