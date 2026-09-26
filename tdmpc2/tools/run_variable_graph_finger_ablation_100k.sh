#!/usr/bin/env bash
# Same-seed 100k Finger diagnosis: fixed pooling vs direct K-role state.

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"; cd "$REPO_ROOT"
PY="${PY:-<DATA_PATH>/conda_envs/tdmpc2_2026/bin/python}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
VIDEO_ROOT="${VIDEO_ROOT:-<BASELINE_PATH>/HRSSM-main/env/data/video_hard}"
CUTIE_REPO="${CUTIE_REPO:-<DATA_PATH>/r2_hrssm_third_party/OC-STORM}"
CUTIE_CHECKPOINT="${CUTIE_CHECKPOINT:-$CUTIE_REPO/feature_extractor/cutie/weights/cutie-small-mega.pth}"
SUPPORT="${SUPPORT:-$REPO_ROOT/datasets/cutie_multitask_support_v1_seed314159/finger-spin/annotations.json}"
GRAPH="$REPO_ROOT/tdmpc2/object_graphs/finger_spin.json"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
TAG="variable_graph_finger_ablation_100k_v1_${STAMP}"
BASE="${OUTPUT_ROOT:-$REPO_ROOT/logs/_diagnostic/$TAG}"; STAGE="${BASE}.incomplete"
LOCK="$REPO_ROOT/logs/_diagnostic/.variable_graph_finger_ablation.lock"
SEED=24; STEPS=100000; EVAL_FREQ=10000; EVAL_EPISODES=10
PROMOTED=0; LOCK_OWNED=0; STAGE_OWNED=0; PIDS=()
cleanup() {
	local rc=$? failed pid; trap - EXIT INT TERM
	for pid in "${PIDS[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
	for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
	if (( LOCK_OWNED )); then rmdir -- "$LOCK" 2>/dev/null || true; fi
	if (( rc != 0 && PROMOTED == 0 && STAGE_OWNED == 1 )) && [[ -d "$STAGE" ]]; then
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"; mv -- "$STAGE" "$failed"
		echo "VARIABLE_GRAPH_FINGER_ABLATION_FAILED_ARCHIVE=$failed" >&2
	fi
	exit "$rc"
}
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
for path in "$PY" "$CUTIE_CHECKPOINT" "$SUPPORT" "$GRAPH"; do
	[[ -f "$path" || -x "$path" ]] || { echo "Missing: $path" >&2; exit 2; }
done
[[ "$GPU0" != "$GPU1" && ! -e "$BASE" && ! -e "$STAGE" ]] || exit 3
mkdir -p "$REPO_ROOT/logs/_diagnostic"; mkdir "$LOCK"; LOCK_OWNED=1
mkdir -p "$STAGE/contracts" "$STAGE/training" "$STAGE/evaluations" "$STAGE/hydra"; STAGE_OWNED=1
echo '[1/4] Variable-K and safe-skip contracts'
"$PY" -m py_compile tdmpc2/common/layers.py tdmpc2/common/world_model.py tdmpc2/tdmpc2.py \
	tdmpc2/check_cutie_variable_object_graph_contract.py \
	tdmpc2/tools/aggregate_variable_graph_finger_ablation.py \
	>"$STAGE/contracts/py_compile.log" 2>&1
"$PY" -B tdmpc2/check_cutie_variable_object_graph_contract.py >"$STAGE/contracts/variable_k.log" 2>&1

run_arm() {
	local name=$1 gpu=$2 readout=$3 latent_dim=$4
	local run_name run_root rc condition
	run_name="${TAG}_${name}"
	run_root="$REPO_ROOT/logs/finger-spin/$SEED/$run_name"
	[[ ! -e "$run_root" ]] || return 3
	echo "TRAIN_START arm=$name gpu=$gpu readout=$readout latent_dim=$latent_dim root=$run_root"
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 "$PY" tdmpc2/train.py \
		task=finger-spin obs=rgb model_size=5 steps=$STEPS seed=$SEED \
		eval_freq=$EVAL_FREQ eval_episodes=$EVAL_EPISODES save_eval_episode_trace=true \
		video_background_enabled=true "video_background_root=$VIDEO_ROOT" \
		video_background_split=train video_background_manifest_dir=null \
		visual_foreground_erosion_pixels=0 compile=false compile_fallback_random=false \
		enable_wandb=false wandb_project=none wandb_entity=none save_csv=true save_video=false \
		save_agent=true checkpoint=null data_dir=null obs_shapes=null action_dims=null episode_lengths=null \
		flat_anchor=true flat_anchor_mode=cutie_object_only flat_anchor_loss_beta=0.1 \
		cutie_object_observation_variant=full cutie_object_frame_schema=cutie_query_mask_status_v1 \
		cutie_object_role_names='[finger,spinner]' cutie_object_support_schema=generic_indexed_v1 \
		cutie_object_allow_simulator_support=true cutie_object_allow_simulator_runtime=false \
		cutie_object_allow_simulator_kinematics_runtime=false "cutie_object_repo=$CUTIE_REPO" \
		"cutie_object_checkpoint=$CUTIE_CHECKPOINT" "cutie_object_support_path=$SUPPORT" \
		cutie_object_config_dir=null cutie_object_device=cuda:0 cutie_object_tracker_height=448 \
		cutie_object_tracker_width=448 cutie_object_model_size=small cutie_object_prompt_radius=2.0 \
		cutie_object_amp=true cutie_object_worker_timeout_seconds=180 cutie_object_num_roles=2 \
		cutie_object_frame_dim=590 cutie_object_stack_frames=3 cutie_object_input_dim=1770 \
		cutie_object_auxiliary_target=geometry_status_full_denominator cutie_object_role_dim=64 \
		cutie_object_hidden_dim=256 "cutie_object_only_latent_dim=$latent_dim" \
		cutie_object_spatial_token_enabled=true cutie_object_variable_graph_enabled=true \
		cutie_object_variable_graph_max_roles=8 cutie_object_variable_graph_pool_tokens=2 \
		cutie_object_variable_graph_primary_skip_enabled=false \
		"cutie_object_variable_graph_readout=$readout" \
		"cutie_object_spatial_graph_path=$GRAPH" cutie_object_spatial_token_dim=64 \
		cutie_object_spatial_num_heads=4 cutie_object_spatial_num_layers=2 \
		cutie_object_native_highres_enabled=false cutie_object_native_highres_size=128 \
		cutie_object_last_valid_memory=false cutie_object_policy_burst_plan=null \
		cutie_object_belief_enabled=false cutie_object_belief_use_for_control=false \
		visual_pose_checkpoint=null hydra.job.chdir=false "exp_name=$run_name" \
		"hydra.run.dir=$STAGE/hydra/$name" >"$STAGE/training/${name}.log" 2>&1
	rc=$?; echo "TRAIN_END arm=$name gpu=$gpu rc=$rc"; (( rc == 0 )) || return "$rc"
	for condition in clean hard; do
		echo "EVAL_START arm=$name condition=$condition gpu=$gpu episodes=20"
		env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 "$PY" -B \
			-m tdmpc2.tools.evaluate_cutie_multitask_checkpoint --task finger-spin \
			--backend cutie_object_only --condition "$condition" \
			--runtime-config "$run_root/runtime_config.json" --checkpoint "$run_root/models/final.pt" \
			--training-seed $SEED --expected-training-steps $STEPS \
			--expected-training-eval-freq $EVAL_FREQ --expected-training-eval-episodes $EVAL_EPISODES \
			--episodes 20 --env-seed 424251 --background-seed 1618041 \
			--planner-seed-base 8675700 --erosion-pixels 0 \
			--output "$STAGE/evaluations/${name}_${condition}.json" \
			>"$STAGE/evaluations/${name}_${condition}.log" 2>&1
		rc=$?; echo "EVAL_END arm=$name condition=$condition gpu=$gpu rc=$rc"; (( rc == 0 )) || return "$rc"
	done
}

echo '[2/4] Paired 100k Finger training: pooled 128-D versus direct Kx64'
run_arm attention_pool "$GPU0" pool 128 > >(tee "$STAGE/training/gpu${GPU0}.queue.log") 2>&1 & PIDS+=("$!")
run_arm direct_roles "$GPU1" direct 128 > >(tee "$STAGE/training/gpu${GPU1}.queue.log") 2>&1 & PIDS+=("$!")
echo '[3/4] Waiting for both training and held-out evaluation arms'
set +e; wait "${PIDS[0]}"; RC0=$?; wait "${PIDS[1]}"; RC1=$?; set -e; PIDS=()
echo "GPU_ARM_END attention_pool=$RC0 direct_roles=$RC1"; (( RC0 == 0 && RC1 == 0 )) || exit 1
echo '[4/4] Strict aggregation and atomic publication'
ARGS=()
for spec in 'attention_pool:pool' 'direct_roles:direct'; do
	name="${spec%%:*}"; readout="${spec##*:}"; run_root="$REPO_ROOT/logs/finger-spin/$SEED/${TAG}_${name}"
	ARGS+=(--arm "$name" "$run_root" "$STAGE/evaluations/${name}_clean.json" \
		"$STAGE/evaluations/${name}_hard.json" "$readout")
done
"$PY" -B -m tdmpc2.tools.aggregate_variable_graph_finger_ablation "${ARGS[@]}" \
	--output "$STAGE/variable_graph_finger_ablation_summary.json" >"$STAGE/aggregate.log" 2>&1
mv -- "$STAGE" "$BASE"; PROMOTED=1; rmdir "$LOCK"; LOCK_OWNED=0; trap - EXIT INT TERM
echo 'VARIABLE_GRAPH_FINGER_ABLATION_COMPLETE'
echo "OUTPUT_ROOT=$BASE"
echo "SUMMARY=$BASE/variable_graph_finger_ablation_summary.json"
