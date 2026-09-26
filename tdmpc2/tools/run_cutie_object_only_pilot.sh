#!/usr/bin/env bash
# Two-GPU scratch pilot: fixed-reset CutieHybrid versus structural ObjectOnly.
#
# RGB remains the camera/perception input in both arms. The ObjectOnly arm
# exposes only object[2,1770] to TD-MPC2 and replay; it has no RGB encoder,
# scene latent, scene dynamics, or hybrid correction heads.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${VIDEO_ROOT:?Set VIDEO_ROOT to video_hard}"
: "${SUPPORT:?Set SUPPORT to verified six-frame annotations.json}"
: "${OC_REPO:?Set OC_REPO to the OC-STORM checkout}"
: "${CUTIE_CKPT:?Set CUTIE_CKPT to cutie-small-mega.pth}"

PY="${PY:-python}"
MANIFEST_DIR="${MANIFEST_DIR:-$REPO_ROOT/tdmpc2/envs/background_manifests}"
GPU_HYBRID="${GPU_HYBRID:-0}"
GPU_OBJECT="${GPU_OBJECT:-1}"
SEED="${SEED:-5}"
STEPS="${STEPS:-50000}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-3}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-7200}"
RUN_TAG="${RUN_TAG:-cutie_object_only_50k_pair_v1}"
if (( STEPS % 1000 == 0 )); then STEP_LABEL="$((STEPS / 1000))k"; else STEP_LABEL="$STEPS"; fi
HYBRID_EXP="cutie_hybrid${STEP_LABEL}_${RUN_TAG}_seed${SEED}"
OBJECT_EXP="cutie_object_only${STEP_LABEL}_${RUN_TAG}_seed${SEED}"
LAUNCH="$REPO_ROOT/logs/_launch/${RUN_TAG}_seed${SEED}"
HYBRID_OUT="$REPO_ROOT/logs/reacher-visual-small/$SEED/$HYBRID_EXP"
OBJECT_OUT="$REPO_ROOT/logs/reacher-visual-small/$SEED/$OBJECT_EXP"
SUMMARY="$LAUNCH/object_only_pilot_summary.json"

for name in GPU_HYBRID GPU_OBJECT SEED STEPS EVAL_FREQ EVAL_EPISODES MAX_WALLCLOCK_SECONDS; do
	value="${!name}"
	[[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || {
		echo "$name must be a non-negative integer: $value" >&2
		exit 2
	}
done
if [[ "$GPU_HYBRID" == "$GPU_OBJECT" ]]; then
	echo "GPU_HYBRID and GPU_OBJECT must be different physical GPU indices." >&2
	exit 2
fi
if (( STEPS < 500 || STEPS % 500 != 0 || EVAL_FREQ % 500 != 0 || STEPS % EVAL_FREQ != 0 || EVAL_EPISODES < 1 || MAX_WALLCLOCK_SECONDS < 1 )); then
	echo "Invalid frozen pilot step/eval/wallclock values." >&2
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
for path in "$LAUNCH" "$HYBRID_OUT" "$OBJECT_OUT"; do
	[[ ! -e "$path" ]] || { echo "Refusing to overwrite: $path" >&2; exit 3; }
done

echo "[1/5] Running model, wrapper, and reset contracts"
"$PY" tdmpc2/check_cutie_object_wrapper_contract.py
"$PY" tdmpc2/check_cutie_oc_adapter_contract.py
"$PY" tdmpc2/check_cutie_hybrid_contract.py
"$PY" tdmpc2/check_cutie_object_only_contract.py
"$PY" tdmpc2/check_flat_anchor_contract.py
"$PY" tdmpc2/check_cutie_object_only_integration_contract.py
CUDA_VISIBLE_DEVICES="$GPU_OBJECT" "$PY" tdmpc2/check_cutie_object_only_update.py
CUDA_VISIBLE_DEVICES="$GPU_OBJECT" "$PY" tdmpc2/check_cutie_object_only_update.py --compile

echo "[2/5] Running Cutie preflight on both physical GPUs"
for gpu in "$GPU_HYBRID" "$GPU_OBJECT"; do
	CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m tdmpc2.tools.check_cutie_oc_preflight \
		--oc-storm-repo "$OC_REPO" \
		--checkpoint "$CUTIE_CKPT" \
		--support-annotations "$SUPPORT" \
		--object-schema whole_arm_goal_v1 \
		--model-size small \
		--prompt-radius 2.0 \
		--tracker-size 448 448 \
		--device cuda:0 \
		--sha256
done

mkdir -p "$LAUNCH"
RESET_CONFIG="$LAUNCH/reset_probe_runtime_config.json"
RESET_REPORT="$LAUNCH/reset_isolation.json"
"$PY" - "$RESET_CONFIG" "$OC_REPO" "$CUTIE_CKPT" "$SUPPORT" <<'PY'
import json
import sys
from pathlib import Path

output, repo, checkpoint, support = map(Path, sys.argv[1:])
payload = {
    'task': 'reacher-visual-small',
    'obs': 'rgb',
    'model_size': 5,
    'flat_anchor': True,
    'flat_anchor_mode': 'cutie_object_only',
    'cutie_object_repo': str(repo.resolve()),
    'cutie_object_checkpoint': str(checkpoint.resolve()),
    'cutie_object_support_path': str(support.resolve()),
    'cutie_object_config_dir': None,
    'cutie_object_device': 'cuda:0',
    'cutie_object_tracker_height': 448,
    'cutie_object_tracker_width': 448,
    'cutie_object_model_size': 'small',
    'cutie_object_prompt_radius': 2.0,
    'cutie_object_amp': True,
    'cutie_object_worker_timeout_seconds': 180.0,
}
output.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
    encoding='utf-8',
)
PY
CUDA_VISIBLE_DEVICES="$GPU_OBJECT" "$PY" -m tdmpc2.tools.check_cutie_episode_reset_isolation \
	--runtime-config "$RESET_CONFIG" \
	--output "$RESET_REPORT" \
	--pollution-length 500

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
	flat_anchor=true
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
	cutie_object_only_latent_dim=128
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
	local arm=$1 gpu exp mode log hydra_dir
	if [[ "$arm" == hybrid ]]; then
		gpu=$GPU_HYBRID
		exp=$HYBRID_EXP
		mode=cutie_hybrid
		log="$LAUNCH/hybrid.log"
		hydra_dir="$LAUNCH/hydra_hybrid"
	else
		gpu=$GPU_OBJECT
		exp=$OBJECT_EXP
		mode=cutie_object_only
		log="$LAUNCH/object_only.log"
		hydra_dir="$LAUNCH/hydra_object_only"
	fi
	exec env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" tdmpc2/train.py "${COMMON[@]}" \
		"flat_anchor_mode=$mode" "exp_name=$exp" \
		"hydra.run.dir=$hydra_dir" hydra.job.chdir=false \
		>"$log" 2>&1
}

echo "[3/5] Launching $STEPS-step scratch pair"
echo "Hybrid log: $LAUNCH/hybrid.log"
echo "ObjectOnly log: $LAUNCH/object_only.log"
START_SECONDS="$(date +%s)"
run_one hybrid & HYBRID_PID=$!
ACTIVE_PIDS=("$HYBRID_PID")
run_one object_only & OBJECT_PID=$!
ACTIVE_PIDS+=("$OBJECT_PID")

set +e
wait -n -p FINISHED_PID "$HYBRID_PID" "$OBJECT_PID"
FIRST_RC=$?
set -e
if [[ "$FINISHED_PID" == "$HYBRID_PID" ]]; then OTHER_PID=$OBJECT_PID; else OTHER_PID=$HYBRID_PID; fi
if (( FIRST_RC != 0 )); then
	echo "Worker $FINISHED_PID failed rc=$FIRST_RC; stopping sibling $OTHER_PID." >&2
	kill "$OTHER_PID" 2>/dev/null || true
fi
set +e
wait "$OTHER_PID"
OTHER_RC=$?
set -e
ACTIVE_PIDS=()
if (( FIRST_RC != 0 )); then exit "$FIRST_RC"; fi
if (( OTHER_RC != 0 )); then exit "$OTHER_RC"; fi
ELAPSED_SECONDS=$(( $(date +%s) - START_SECONDS ))

echo "[4/5] Verifying structural isolation, replay placement, and speed"
"$PY" - "$HYBRID_OUT" "$OBJECT_OUT" "$RESET_REPORT" "$SUMMARY" \
	"$STEPS" "$EVAL_FREQ" "$EVAL_EPISODES" "$ELAPSED_SECONDS" \
	"$MAX_WALLCLOCK_SECONDS" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

import torch

hybrid_root, object_root, reset_report, summary_path = map(Path, sys.argv[1:5])
steps, eval_freq, eval_episodes, elapsed, max_wallclock = map(int, sys.argv[5:10])
expected_eval_steps = [float(step) for step in range(0, steps + 1, eval_freq)]
expected_frames = steps + steps // 500 + (steps // eval_freq + 1) * eval_episodes * 501
reset = json.loads(reset_report.read_text(encoding='utf-8'))
if reset.get('status') != 'episode_reset_isolation_pass':
    raise AssertionError(('reset isolation', reset.get('status')))

results = {}
for name, root, mode, latent_dim, obs_keys in (
    ('hybrid', hybrid_root, 'cutie_hybrid', 640, ['object', 'rgb']),
    ('object_only', object_root, 'cutie_object_only', 128, ['object']),
):
    paths = {
        key: root / relative
        for key, relative in {
            'config': 'runtime_config.json',
            'eval': 'eval.csv',
            'checkpoint': 'models/final.pt',
            'perception': 'perception_runtime.json',
            'replay': 'replay_runtime.json',
            'trainer': 'trainer_runtime.json',
        }.items()
    }
    for path in paths.values():
        if not path.is_file():
            raise AssertionError(f'missing {path}')
    cfg = json.loads(paths['config'].read_text(encoding='utf-8'))
    if cfg.get('flat_anchor_mode') != mode or cfg.get('latent_dim') != latent_dim:
        raise AssertionError((name, cfg.get('flat_anchor_mode'), cfg.get('latent_dim')))
    if sorted(cfg.get('obs_shape', {})) != obs_keys:
        raise AssertionError((name, 'obs_shape', cfg.get('obs_shape')))
    if name == 'object_only' and cfg['obs_shape'] != {'object': [2, 1770]}:
        raise AssertionError(('object-only observation schema', cfg['obs_shape']))
    rows = list(csv.DictReader(paths['eval'].open(encoding='utf-8')))
    eval_steps = [float(row['step']) for row in rows]
    rewards = [float(row['episode_reward']) for row in rows]
    if eval_steps != expected_eval_steps or not all(map(math.isfinite, rewards)):
        raise AssertionError((name, eval_steps, rewards))
    payload = torch.load(paths['checkpoint'], map_location='cpu', weights_only=False)
    state = payload['model'] if 'model' in payload else payload
    bad = [
        key for key, value in state.items()
        if torch.is_tensor(value) and not torch.isfinite(value).all()
    ]
    if bad:
        raise AssertionError((name, 'nonfinite checkpoint', bad[:10]))
    if name == 'object_only':
        forbidden = [
            key for key in state
            if key.startswith('_encoder.rgb.') or key.startswith('_hybrid_')
        ]
        if forbidden:
            raise AssertionError(('object-only checkpoint leaked RGB/hybrid state', forbidden[:10]))
    perception = json.loads(paths['perception'].read_text(encoding='utf-8'))
    replay = json.loads(paths['replay'].read_text(encoding='utf-8'))
    trainer = json.loads(paths['trainer'].read_text(encoding='utf-8'))
    if replay.get('observation_keys') != obs_keys:
        raise AssertionError((name, 'replay keys', replay.get('observation_keys')))
    if replay.get('capacity') != steps or replay.get('storage_device') != 'cuda:0':
        raise AssertionError((name, 'replay placement', replay))
    if not math.isfinite(float(trainer.get(
        'training_non_eval_steps_per_second', math.nan
    ))):
        raise AssertionError((name, 'trainer speed', trainer))
    runtime_gates = {
        'frames': perception.get('frames') == expected_frames,
        'valid_frame_rate': perception.get('valid_frame_rate', -1) >= 0.95,
        'max_invalid_burst': perception.get('max_invalid_burst', 999999) <= 5,
        'worker_restarts': perception.get('worker_restarts') == 0,
        'timeouts': perception.get('timeouts') == 0,
        'ms_per_frame': perception.get('ms_per_frame', math.inf) <= 800.0,
        'runtime_unit': perception.get('runtime_unit') == (
            'milliseconds_per_tracked_frame_excluding_support_prompts'
        ),
        'episode_reset_strategy': perception.get('episode_reset_strategy') == (
            'fresh_inference_core_support_replay_v1'
        ),
    }
    results[name] = {
        'root': str(root),
        'eval_rewards': rewards,
        'runtime_gates': runtime_gates,
        'perception_runtime': perception,
        'replay_runtime': replay,
        'trainer_runtime': trainer,
    }

hybrid = results['hybrid']
objects = results['object_only']
hybrid_bytes = int(hybrid['replay_runtime']['storage_required_bytes'])
object_bytes = int(objects['replay_runtime']['storage_required_bytes'])
hybrid_sps = float(hybrid['trainer_runtime']['training_non_eval_steps_per_second'])
object_sps = float(objects['trainer_runtime']['training_non_eval_steps_per_second'])
speedup = object_sps / hybrid_sps
storage_ratio = object_bytes / hybrid_bytes
gates = {
    'reset_isolation': True,
    'hybrid_runtime': all(hybrid['runtime_gates'].values()),
    'object_only_runtime': all(objects['runtime_gates'].values()),
    'object_only_replay_is_cuda': objects['replay_runtime']['storage_device'] == 'cuda:0',
    'hybrid_replay_is_cuda': hybrid['replay_runtime']['storage_device'] == 'cuda:0',
    'object_only_storage_at_most_30pct_of_hybrid': storage_ratio <= 0.30,
    'matched_gpu_model': (
        hybrid['perception_runtime'].get('device_name')
        == objects['perception_runtime'].get('device_name')
        and hybrid['perception_runtime'].get('device_name') is not None
    ),
    'wallclock_seconds': elapsed <= max_wallclock,
}
outcomes = {
    # This is descriptive concurrent two-GPU throughput, not a hardware-neutral
    # structural timing theorem. It therefore cannot invalidate an otherwise
    # healthy model/run when transient GPU load crosses the target threshold.
    'object_only_non_eval_speedup_at_least_10pct': speedup >= 1.10,
}
summary = {
    'format': 'cutie_object_only_scratch_pilot_v1',
    'status': 'object_only_pilot_engineering_pass' if all(gates.values()) else 'object_only_pilot_engineering_fail',
    'scientific_scope': (
        'single-seed scratch development evidence; reward is descriptive and is '
        'not an engineering gate or a cross-seed algorithm conclusion'
    ),
    'seed': int(results['object_only']['trainer_runtime'].get('seed', 0) or 0),
    'steps': steps,
    'eval_freq': eval_freq,
    'eval_episodes': eval_episodes,
    'expected_cutie_frames_per_arm': expected_frames,
    'elapsed_seconds': elapsed,
    'gates': gates,
    'outcomes': outcomes,
    'speed_measurement_scope': (
        'concurrent matched-GPU-model engineering throughput; excludes final '
        'checkpoint serialization but is not a same-device AB/BA benchmark'
    ),
    'object_only_over_hybrid_non_eval_speed_ratio': speedup,
    'object_only_over_hybrid_replay_storage_ratio': storage_ratio,
    'eval_reward_delta_object_only_minus_hybrid': [
        obj - full
        for full, obj in zip(hybrid['eval_rewards'], objects['eval_rewards'])
    ],
    'reset_isolation': reset,
    'runs': results,
}
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
    encoding='utf-8',
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
if not all(gates.values()):
    raise SystemExit(4)
PY

echo "[5/5] CUTIE_OBJECT_ONLY_PILOT_OK"
echo "HYBRID_OUT=$HYBRID_OUT"
echo "OBJECT_OUT=$OBJECT_OUT"
echo "SUMMARY=$SUMMARY"
ACTIVE_PIDS=()
trap - EXIT INT TERM
