#!/usr/bin/env bash
# Broad RGB-only DMC screen for the variable-cardinality object graph.

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
SUPPORT_ROOT="${SUPPORT_ROOT:-$REPO_ROOT/support/cutie_extended_support_v1_seed314159}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG_OVERRIDE:-variable_object_graph_broad_screen_30k_v1_${STAMP}}"
BASE="${OUTPUT_ROOT:-$REPO_ROOT/logs/_diagnostic/$RUN_TAG}"
STAGE="${BASE}.incomplete"
LOCK="$REPO_ROOT/logs/_diagnostic/.variable_object_graph_broad_screen.lock"
SEED="${SEED:-25}"
STEPS="${STEPS:-30000}"
EVAL_FREQ=5000
EVAL_EPISODES=5
HELDOUT_EPISODES=20
REUSE_RUN_TAG="${REUSE_RUN_TAG:-}"
REUSE_STAGE="${REUSE_STAGE:-}"

TASKS=(
	acrobot-swingup cup-catch cartpole-swingup cheetah-run finger-spin
	cartpole-balance cartpole-balance-sparse cartpole-swingup-sparse
	reacher-easy reacher-hard finger-turn-easy finger-turn-hard
	pendulum-swingup hopper-stand hopper-hop
	quadruped-run quadruped-walk walker-stand walker-walk walker-run
)
QUEUE0=(acrobot-swingup cartpole-balance cartpole-swingup cheetah-run hopper-hop pendulum-swingup quadruped-walk walker-stand walker-walk reacher-hard)
QUEUE1=(cup-catch cartpole-balance-sparse cartpole-swingup-sparse finger-spin finger-turn-easy finger-turn-hard hopper-stand quadruped-run reacher-easy walker-run)

PROMOTED=0; LOCK_OWNED=0; STAGE_OWNED=0; PIDS=()
cleanup() {
	local rc=$? failed pid
	trap - EXIT INT TERM
	for pid in "${PIDS[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
	for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
	if (( LOCK_OWNED == 1 )); then rmdir -- "$LOCK" 2>/dev/null || true; fi
	if (( rc != 0 && PROMOTED == 0 && STAGE_OWNED == 1 )) && [[ -d "$STAGE" ]]; then
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
		mv -- "$STAGE" "$failed"
		echo "VARIABLE_OBJECT_GRAPH_BROAD_SCREEN_FAILED_ARCHIVE=$failed" >&2
	fi
	exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for path in "$PY" "$CUTIE_CHECKPOINT"; do
	[[ -f "$path" || -x "$path" ]] || { echo "Required file missing: $path" >&2; exit 2; }
done
for path in "$CUTIE_REPO" "$VIDEO_ROOT" "$SUPPORT_ROOT"; do
	[[ -d "$path" ]] || { echo "Required directory missing: $path" >&2; exit 2; }
done
if [[ -n "$REUSE_RUN_TAG" || -n "$REUSE_STAGE" ]]; then
	[[ -n "$REUSE_RUN_TAG" && -d "$REUSE_STAGE" ]] || {
		echo 'REUSE_RUN_TAG and an existing REUSE_STAGE must be provided together.' >&2
		exit 2
	}
fi
[[ "$GPU0" != "$GPU1" ]] || { echo 'GPU0 and GPU1 must differ.' >&2; exit 2; }
[[ ! -e "$BASE" && ! -e "$STAGE" ]] || { echo "Refusing overwrite: $BASE" >&2; exit 3; }
mkdir -p -- "$REPO_ROOT/logs/_diagnostic"
mkdir -- "$LOCK" || { echo "Another run owns $LOCK" >&2; exit 3; }
LOCK_OWNED=1
mkdir -p -- "$STAGE/contracts" "$STAGE/training" "$STAGE/evaluations" "$STAGE/hydra"
STAGE_OWNED=1

echo '[1/5] Variable-cardinality contracts, generated task graphs, and complete support packs'
"$PY" -m py_compile \
	tdmpc2/common/buffer.py tdmpc2/common/layers.py tdmpc2/common/world_model.py tdmpc2/tdmpc2.py \
	tdmpc2/envs/wrappers/cutie_object.py tdmpc2/perception/cutie_oc_adapter.py \
	tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py \
	tdmpc2/tools/aggregate_variable_object_graph_regression.py \
	tdmpc2/tools/build_standard_variable_object_graphs.py \
	tdmpc2/tools/check_full20_object_graph_inputs.py \
	tdmpc2/check_cutie_variable_object_graph_contract.py \
	>"$STAGE/contracts/py_compile.log" 2>&1
"$PY" -B tdmpc2/check_cutie_variable_object_graph_contract.py \
	>"$STAGE/contracts/variable_k.log" 2>&1
"$PY" -B -m tdmpc2.tools.build_standard_variable_object_graphs \
	--output tdmpc2/object_graphs --check >"$STAGE/contracts/graphs.log" 2>&1
"$PY" -B -m tdmpc2.tools.check_full20_object_graph_inputs \
	--support-root "$SUPPORT_ROOT" --graph-root tdmpc2/object_graphs \
	>"$STAGE/contracts/full20_inputs.log" 2>&1

task_spec() {
	case "$1" in
		acrobot-swingup)
			printf '%s\n' 'whole_acrobot' 'acrobot_swingup.json' ;;
		cup-catch)
			printf '%s\n' 'cup,ball' 'cup_catch.json' ;;
		cartpole-balance|cartpole-balance-sparse|cartpole-swingup|cartpole-swingup-sparse)
			printf '%s\n' 'cart,pole' "${1//-/_}.json" ;;
		reacher-easy|reacher-hard)
			printf '%s\n' 'whole_arm,goal' "${1//-/_}.json" ;;
		finger-spin|finger-turn-easy|finger-turn-hard)
			printf '%s\n' 'finger,spinner' "${1//-/_}.json" ;;
		pendulum-swingup)
			printf '%s\n' 'base,pendulum' 'pendulum_swingup.json' ;;
		hopper-stand|hopper-hop)
			printf '%s\n' 'torso,leg,foot' "${1//-/_}.json" ;;
		walker-stand|walker-walk|walker-run)
			printf '%s\n' 'torso,right_leg,left_leg' "${1//-/_}.json" ;;
		cheetah-run)
			printf '%s\n' 'torso,back_leg,front_leg' 'cheetah_run.json' ;;
		quadruped-run|quadruped-walk)
			printf '%s\n' 'torso,front_legs,back_legs' "${1//-/_}.json" ;;
		*) return 2 ;;
	esac
}

run_task() {
	local task=$1 gpu=$2 roles_csv graph_name graph support num_roles latent_dim run_name run_root
	local reuse_root source_eval rc condition
	local -a fields
	mapfile -t fields < <(task_spec "$task")
	roles_csv="${fields[0]}"; graph_name="${fields[1]}"
	num_roles=$(awk -F, '{print NF}' <<<"$roles_csv")
	latent_dim=$((num_roles * 64))
	graph="$REPO_ROOT/tdmpc2/object_graphs/$graph_name"
	support="$SUPPORT_ROOT/$task/annotations.json"
	for path in "$support" "$graph"; do
		[[ -f "$path" ]] || { echo "Task input missing: $path" >&2; return 2; }
	done
	run_name="${RUN_TAG}_${task}"
	run_root="$REPO_ROOT/logs/$task/$SEED/$run_name"
	reuse_root=""
	if [[ -n "$REUSE_RUN_TAG" ]]; then
		reuse_root="$REPO_ROOT/logs/$task/$SEED/${REUSE_RUN_TAG}_${task}"
	fi
	if [[ -n "$reuse_root" && -f "$reuse_root/runtime_config.json" && -f "$reuse_root/models/final.pt" ]]; then
		run_root="$reuse_root"
		echo "TRAIN_REUSED task=$task roles=$num_roles latent_dim=$latent_dim readout=direct root=$run_root"
	else
		[[ ! -e "$run_root" ]] || { echo "Refusing overwrite: $run_root" >&2; return 3; }
		echo "TRAIN_START task=$task roles=$num_roles latent_dim=$latent_dim readout=direct gpu=$gpu root=$run_root"
		env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" tdmpc2/train.py \
		"task=$task" obs=rgb model_size=5 "steps=$STEPS" "seed=$SEED" \
		"eval_freq=$EVAL_FREQ" "eval_episodes=$EVAL_EPISODES" save_eval_episode_trace=true \
		video_background_enabled=true "video_background_root=$VIDEO_ROOT" \
		video_background_split=train video_background_manifest_dir=null \
		visual_foreground_erosion_pixels=0 compile=false compile_fallback_random=true \
		enable_wandb=false wandb_project=none wandb_entity=none \
		save_csv=true save_video=false save_agent=true checkpoint=null data_dir=null \
		obs_shapes=null action_dims=null episode_lengths=null \
		flat_anchor=true flat_anchor_mode=cutie_object_only flat_anchor_loss_beta=0.1 \
		cutie_object_observation_variant=full cutie_object_frame_schema=cutie_query_mask_status_v1 \
		"cutie_object_role_names=[$roles_csv]" cutie_object_support_schema=generic_indexed_v1 \
		cutie_object_allow_simulator_support=true cutie_object_allow_simulator_runtime=false \
		cutie_object_allow_simulator_kinematics_runtime=false \
		"cutie_object_repo=$CUTIE_REPO" "cutie_object_checkpoint=$CUTIE_CHECKPOINT" \
		"cutie_object_support_path=$support" cutie_object_config_dir=null \
		cutie_object_device=cuda:0 cutie_object_tracker_height=448 cutie_object_tracker_width=448 \
		cutie_object_model_size=small cutie_object_prompt_radius=2.0 cutie_object_amp=true \
		cutie_object_worker_timeout_seconds=180 "cutie_object_num_roles=$num_roles" \
		cutie_object_frame_dim=590 cutie_object_stack_frames=3 cutie_object_input_dim=1770 \
		cutie_object_auxiliary_target=geometry_status_full_denominator \
		cutie_object_role_dim=64 cutie_object_hidden_dim=256 "cutie_object_only_latent_dim=$latent_dim" \
		cutie_object_spatial_token_enabled=true cutie_object_variable_graph_enabled=true \
		cutie_object_variable_graph_max_roles=8 cutie_object_variable_graph_pool_tokens=2 \
		cutie_object_variable_graph_primary_skip_enabled=false \
		cutie_object_variable_graph_readout=direct \
		"cutie_object_spatial_graph_path=$graph" cutie_object_spatial_token_dim=64 \
		cutie_object_spatial_num_heads=4 cutie_object_spatial_num_layers=2 \
		cutie_object_native_highres_enabled=false cutie_object_native_highres_size=128 \
		cutie_object_last_valid_memory=false cutie_object_policy_burst_plan=null \
		cutie_object_belief_enabled=false cutie_object_belief_use_for_control=false \
		visual_pose_checkpoint=null hydra.job.chdir=false \
			"exp_name=$run_name" "hydra.run.dir=$STAGE/hydra/$task" \
			>"$STAGE/training/${task}.log" 2>&1
		rc=$?; printf '%s\n' "$rc" >"$STAGE/training/${task}.rc"
		echo "TRAIN_END task=$task roles=$num_roles latent_dim=$latent_dim readout=direct gpu=$gpu rc=$rc"
		(( rc == 0 )) || return "$rc"
	fi
	printf '%s\n' "$run_root" >"$STAGE/training/${task}.root"
	for condition in clean hard; do
		source_eval="$REUSE_STAGE/evaluations/${task}_${condition}.json"
		if [[ -n "$REUSE_STAGE" && "$run_root" == "$reuse_root" && -f "$source_eval" ]]; then
			cp -- "$source_eval" "$STAGE/evaluations/${task}_${condition}.json"
			echo "EVAL_REUSED task=$task condition=$condition source=$source_eval"
			continue
		fi
		echo "EVAL_START task=$task condition=$condition gpu=$gpu episodes=$HELDOUT_EPISODES"
		env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
			"$PY" -B -m tdmpc2.tools.evaluate_cutie_multitask_checkpoint \
			--task "$task" --backend cutie_object_only --condition "$condition" \
			--runtime-config "$run_root/runtime_config.json" --checkpoint "$run_root/models/final.pt" \
			--training-seed "$SEED" --expected-training-steps "$STEPS" \
			--expected-training-eval-freq "$EVAL_FREQ" --expected-training-eval-episodes "$EVAL_EPISODES" \
			--episodes "$HELDOUT_EPISODES" --env-seed 424250 --background-seed 1618040 \
			--planner-seed-base 8675600 --erosion-pixels 0 \
			--output "$STAGE/evaluations/${task}_${condition}.json" \
			>"$STAGE/evaluations/${task}_${condition}.log" 2>&1
		rc=$?; echo "EVAL_END task=$task condition=$condition gpu=$gpu rc=$rc"
		(( rc == 0 )) || return "$rc"
	done
}

run_queue() { local gpu=$1; shift; local task; for task in "$@"; do run_task "$task" "$gpu"; done; }

echo "[2/5] GPU $GPU0 queue: ${QUEUE0[*]}"
run_queue "$GPU0" "${QUEUE0[@]}" > >(tee "$STAGE/training/gpu${GPU0}_queue.log") 2>&1 & PIDS+=("$!")
echo "[3/5] GPU $GPU1 queue: ${QUEUE1[*]}"
run_queue "$GPU1" "${QUEUE1[@]}" > >(tee "$STAGE/training/gpu${GPU1}_queue.log") 2>&1 & PIDS+=("$!")

echo '[4/5] Waiting for both GPU queues and held-out evaluations'
set +e
wait "${PIDS[0]}"; RC0=$?
wait "${PIDS[1]}"; RC1=$?
set -e
PIDS=()
echo "GPU_QUEUE_END gpu=$GPU0 rc=$RC0"
echo "GPU_QUEUE_END gpu=$GPU1 rc=$RC1"
(( RC0 == 0 && RC1 == 0 )) || exit 1

echo '[5/5] Strict aggregation and atomic publication'
ARGS=()
for task in "${TASKS[@]}"; do
	run_root="$(<"$STAGE/training/${task}.root")"
	ARGS+=(--entry "$task" "$run_root" "$STAGE/evaluations/${task}_clean.json" "$STAGE/evaluations/${task}_hard.json")
done
"$PY" -B -m tdmpc2.tools.aggregate_variable_object_graph_regression \
	"${ARGS[@]}" --output "$STAGE/variable_object_graph_broad_screen_summary.json" \
	>"$STAGE/aggregate.log" 2>&1
mv -- "$STAGE" "$BASE"
PROMOTED=1
rmdir -- "$LOCK"; LOCK_OWNED=0
trap - EXIT INT TERM
echo 'VARIABLE_OBJECT_GRAPH_BROAD_SCREEN_COMPLETE'
echo "OUTPUT_ROOT=$BASE"
echo "SUMMARY=$BASE/variable_object_graph_broad_screen_summary.json"
