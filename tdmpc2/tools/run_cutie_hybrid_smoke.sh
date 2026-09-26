#!/usr/bin/env bash
# Two-GPU engineering smoke for official RGB versus live CutieHybrid.
#
# Required: VIDEO_ROOT, SUPPORT, OC_REPO, CUTIE_CKPT
# Optional: PY, MANIFEST_DIR, GPU_RGB=0, GPU_CUTIE=1, SEED=1,
#           RUN_TAG=cutie_hybrid_3k_smoke_v1, STEPS=3000,
#           EVAL_FREQ=3000, EVAL_EPISODES=1, MAX_WALLCLOCK_SECONDS=4500,
#           SUMMARY_NAME=smoke_summary.json, RUN_SCOPE=engineering_smoke

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${VIDEO_ROOT:?Set VIDEO_ROOT to the existing video_hard directory}"
: "${SUPPORT:?Set SUPPORT to the verified six-frame support annotations.json}"
: "${OC_REPO:?Set OC_REPO to the OC-STORM checkout}"
: "${CUTIE_CKPT:?Set CUTIE_CKPT to cutie-small-mega.pth}"

PY="${PY:-python}"
MANIFEST_DIR="${MANIFEST_DIR:-$REPO_ROOT/tdmpc2/envs/background_manifests}"
GPU_RGB="${GPU_RGB:-0}"
GPU_CUTIE="${GPU_CUTIE:-1}"
SEED="${SEED:-1}"
RUN_TAG="${RUN_TAG:-cutie_hybrid_3k_smoke_v1}"
STEPS="${STEPS:-3000}"
EVAL_FREQ="${EVAL_FREQ:-3000}"
EVAL_EPISODES="${EVAL_EPISODES:-1}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-4500}"
SUMMARY_NAME="${SUMMARY_NAME:-smoke_summary.json}"
RUN_SCOPE="${RUN_SCOPE:-engineering_smoke}"
if (( STEPS % 1000 == 0 )); then STEP_LABEL="$((STEPS / 1000))k"; else STEP_LABEL="$STEPS"; fi
RGB_EXP="rgb${STEP_LABEL}_${RUN_TAG}_seed${SEED}"
CUTIE_EXP="cutie_hybrid${STEP_LABEL}_${RUN_TAG}_seed${SEED}"
LAUNCH="$REPO_ROOT/logs/_launch/${RUN_TAG}_seed${SEED}"
RGB_OUT="$REPO_ROOT/logs/reacher-visual-small/$SEED/$RGB_EXP"
CUTIE_OUT="$REPO_ROOT/logs/reacher-visual-small/$SEED/$CUTIE_EXP"

if [[ "$GPU_RGB" == "$GPU_CUTIE" ]]; then
	echo "GPU_RGB and GPU_CUTIE must be different physical GPU indices." >&2
	exit 2
fi
for value_name in STEPS EVAL_FREQ EVAL_EPISODES MAX_WALLCLOCK_SECONDS; do
	value="${!value_name}"
	if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
		echo "$value_name must be a positive integer, got: $value" >&2
		exit 2
	fi
done
if (( STEPS % 500 != 0 || EVAL_FREQ % 500 != 0 || STEPS % EVAL_FREQ != 0 )); then
	echo "STEPS and EVAL_FREQ must be multiples of 500, and STEPS must be divisible by EVAL_FREQ." >&2
	exit 2
fi
if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
	echo "This runner requires Bash >=5.1." >&2
	exit 2
fi
if [[ "$PY" == */* ]]; then
	[[ -x "$PY" ]] || { echo "Python is not executable: $PY" >&2; exit 2; }
else
	command -v "$PY" >/dev/null || { echo "Python is not on PATH: $PY" >&2; exit 2; }
fi
for path in "$VIDEO_ROOT" "$MANIFEST_DIR" "$SUPPORT" "$OC_REPO" "$CUTIE_CKPT"; do
	[[ -e "$path" ]] || { echo "Required input does not exist: $path" >&2; exit 2; }
done
for path in "$RGB_OUT" "$CUTIE_OUT" "$LAUNCH"; do
	[[ ! -e "$path" ]] || { echo "Refusing to overwrite: $path" >&2; exit 3; }
done

echo "[1/4] Running local integration contracts"
"$PY" tdmpc2/check_cutie_object_wrapper_contract.py
"$PY" tdmpc2/check_cutie_hybrid_contract.py
"$PY" tdmpc2/check_cutie_oc_adapter_contract.py

echo "[2/4] Running official whole-arm/goal Cutie preflight on physical GPU $GPU_CUTIE"
CUDA_VISIBLE_DEVICES="$GPU_CUTIE" "$PY" -m tdmpc2.tools.check_cutie_oc_preflight \
	--oc-storm-repo "$OC_REPO" \
	--checkpoint "$CUTIE_CKPT" \
	--support-annotations "$SUPPORT" \
	--object-schema whole_arm_goal_v1 \
	--model-size small \
	--prompt-radius 2.0 \
	--tracker-size 448 448 \
	--device cuda:0 \
	--sha256

mkdir -p "$LAUNCH"

COMMON=(
	task=reacher-visual-small
	obs=rgb
	model_size=5
	"steps=$STEPS"
	"seed=$SEED"
	"eval_freq=$EVAL_FREQ"
	"eval_episodes=$EVAL_EPISODES"
	video_background_enabled=true
	"video_background_root=$VIDEO_ROOT"
	"video_background_manifest_dir=$MANIFEST_DIR"
	video_background_split=train
	video_background_strength=1.0
	video_background_total_frames=1000
	video_background_source_cache_size=8
	compile=true
	compile_fallback_random=true
	enable_wandb=false
	wandb_project=none
	wandb_entity=none
	save_csv=true
	save_video=false
	save_agent=true
	checkpoint=null
	data_dir=null
	obs_shapes=null
	action_dims=null
	episode_lengths=null
)

CUTIE=(
	flat_anchor=true
	flat_anchor_mode=cutie_hybrid
	"cutie_object_repo=$OC_REPO"
	"cutie_object_checkpoint=$CUTIE_CKPT"
	"cutie_object_support_path=$SUPPORT"
	cutie_object_config_dir=null
	cutie_object_device=cuda:0
	cutie_object_tracker_height=448
	cutie_object_tracker_width=448
	cutie_object_model_size=small
	cutie_object_prompt_radius=2.0
	cutie_object_amp=true
	cutie_object_worker_timeout_seconds=180
	cutie_object_num_roles=2
	cutie_object_frame_dim=590
	cutie_object_stack_frames=3
	cutie_object_input_dim=1770
	cutie_object_role_dim=64
	cutie_object_hidden_dim=256
	cutie_object_joint_dim=640
)

ACTIVE_PIDS=()
cleanup() {
	local rc=$? pid
	trap - EXIT INT TERM
	for pid in "${ACTIVE_PIDS[@]}"; do
		if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; fi
	done
	for pid in "${ACTIVE_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
	exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_one() {
	local arm=$1 gpu exp log hydra_dir
	if [[ "$arm" == rgb ]]; then
		gpu=$GPU_RGB; exp=$RGB_EXP; log="$LAUNCH/rgb.log"; hydra_dir="$LAUNCH/hydra_rgb"
		exec env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
			"$PY" tdmpc2/train.py "${COMMON[@]}" \
			flat_anchor=false "exp_name=$exp" \
			"hydra.run.dir=$hydra_dir" hydra.job.chdir=false \
			>"$log" 2>&1
	else
		gpu=$GPU_CUTIE; exp=$CUTIE_EXP; log="$LAUNCH/cutie.log"; hydra_dir="$LAUNCH/hydra_cutie"
		exec env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
			"$PY" tdmpc2/train.py "${COMMON[@]}" "${CUTIE[@]}" \
			"exp_name=$exp" "hydra.run.dir=$hydra_dir" hydra.job.chdir=false \
			>"$log" 2>&1
	fi
}

echo "[3/4] Launching matched $STEPS-step runs (RGB GPU $GPU_RGB, Cutie GPU $GPU_CUTIE)"
echo "Progress logs: $LAUNCH/rgb.log and $LAUNCH/cutie.log"
echo "In another terminal: tail -f '$LAUNCH/cutie.log'"
START_SECONDS="$(date +%s)"
run_one rgb & RGB_PID=$!
ACTIVE_PIDS=("$RGB_PID")
run_one cutie & CUTIE_PID=$!
ACTIVE_PIDS+=("$CUTIE_PID")

set +e
wait -n -p FINISHED_PID "$RGB_PID" "$CUTIE_PID"
FIRST_RC=$?
set -e
if [[ "$FINISHED_PID" == "$RGB_PID" ]]; then OTHER_PID=$CUTIE_PID; else OTHER_PID=$RGB_PID; fi
if (( FIRST_RC != 0 )); then
	echo "Worker $FINISHED_PID failed with rc=$FIRST_RC; stopping sibling $OTHER_PID." >&2
	kill "$OTHER_PID" 2>/dev/null || true
fi
set +e
wait "$OTHER_PID"; OTHER_RC=$?
set -e
ACTIVE_PIDS=()
if (( FIRST_RC != 0 )); then exit "$FIRST_RC"; fi
if (( OTHER_RC != 0 )); then exit "$OTHER_RC"; fi
ELAPSED_SECONDS=$(( $(date +%s) - START_SECONDS ))

echo "[4/4] Verifying artifacts and engineering gates"
"$PY" - "$RGB_OUT" "$CUTIE_OUT" "$ELAPSED_SECONDS" "$LAUNCH/$SUMMARY_NAME" \
	"$STEPS" "$EVAL_FREQ" "$EVAL_EPISODES" "$MAX_WALLCLOCK_SECONDS" "$RUN_SCOPE" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

import torch

rgb_root, cutie_root = map(Path, sys.argv[1:3])
elapsed = int(sys.argv[3])
summary_path = Path(sys.argv[4])
total_steps = int(sys.argv[5])
eval_freq = int(sys.argv[6])
eval_episodes = int(sys.argv[7])
max_wallclock = int(sys.argv[8])
run_scope = sys.argv[9]
expected_eval_steps = [float(step) for step in range(0, total_steps + 1, eval_freq)]
results = {}
for name, root, latent_dim in (
    ('rgb', rgb_root, 512), ('cutie', cutie_root, 640),
):
    cfg_path = root / 'runtime_config.json'
    csv_path = root / 'eval.csv'
    checkpoint = root / 'models' / 'final.pt'
    for path in (cfg_path, csv_path, checkpoint):
        if not path.is_file():
            raise AssertionError(f'missing {path}')
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    if cfg['latent_dim'] != latent_dim:
        raise AssertionError((name, cfg['latent_dim'], latent_dim))
    rows = list(csv.DictReader(csv_path.open(encoding='utf-8')))
    steps = [float(row['step']) for row in rows]
    rewards = [float(row['episode_reward']) for row in rows]
    if steps != expected_eval_steps or not all(map(math.isfinite, rewards)):
        raise AssertionError((name, steps, rewards))
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = payload['model'] if 'model' in payload else payload
    bad = [
        key for key, value in state.items()
        if torch.is_tensor(value) and not torch.isfinite(value).all()
    ]
    if bad:
        raise AssertionError((name, 'nonfinite checkpoint tensors', bad[:10]))
    results[name] = {'root': str(root), 'eval_rewards': rewards}

runtime_path = cutie_root / 'perception_runtime.json'
if not runtime_path.is_file():
    raise AssertionError(f'missing {runtime_path}')
runtime = json.loads(runtime_path.read_text(encoding='utf-8'))
training_resets = total_steps // 500
eval_calls = total_steps // eval_freq + 1
expected_frames = (
    total_steps + training_resets + eval_calls * eval_episodes * 501
)
gates = {
    'frames': runtime['frames'] == expected_frames,
    'valid_frame_rate': runtime['valid_frame_rate'] >= 0.95,
    'max_invalid_burst': runtime['max_invalid_burst'] <= 5,
    'worker_restarts': runtime['worker_restarts'] == 0,
    'timeouts': runtime['timeouts'] == 0,
    'ms_per_frame': runtime['ms_per_frame'] <= 800.0,
    'runtime_unit': runtime['runtime_unit'] == 'milliseconds_per_tracked_frame_excluding_support_prompts',
    'wallclock_seconds': elapsed <= max_wallclock,
}
rgb_rewards = results['rgb']['eval_rewards']
cutie_rewards = results['cutie']['eval_rewards']
summary = {
    'status': f'{run_scope}_pass' if all(gates.values()) else f'{run_scope}_fail',
    'scientific_scope': (
        'paired development evidence only; multiple seeds and held-out evaluation '
        'are required for an algorithm decision'
    ),
    'total_steps': total_steps,
    'eval_freq': eval_freq,
    'eval_episodes': eval_episodes,
    'expected_cutie_frames': expected_frames,
    'elapsed_seconds': elapsed,
    'gates': gates,
    'cutie_runtime': runtime,
    'runs': results,
    'paired_eval_reward_delta_cutie_minus_rgb': [
        cutie - rgb for rgb, cutie in zip(rgb_rewards, cutie_rewards)
    ],
}
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
    encoding='utf-8',
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
if not all(gates.values()):
    raise SystemExit(4)
PY

echo "CUTIE_HYBRID_RUN_OK"
echo "RGB_OUT=$RGB_OUT"
echo "CUTIE_OUT=$CUTIE_OUT"
echo "SUMMARY=$LAUNCH/$SUMMARY_NAME"
