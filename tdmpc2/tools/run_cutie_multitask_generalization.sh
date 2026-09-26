#!/usr/bin/env bash
# Frozen two-GPU, five-task Cutie generalization pilot.
# Reward is an outcome; completeness, structure, runtime health, and exact
# held-out pairing are engineering gates.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${VIDEO_ROOT:?Set VIDEO_ROOT to video_hard}"
: "${OC_REPO:?Set OC_REPO to the OC-STORM checkout}"
: "${CUTIE_CKPT:?Set CUTIE_CKPT to cutie-small-mega.pth}"

PY="${PY:-python}"
MANIFEST_DIR="${MANIFEST_DIR:-$REPO_ROOT/tdmpc2/envs/background_manifests}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

readonly SEED=6 SUPPORT_SEED=314159 STEPS=100000 EVAL_FREQ=20000
readonly EVAL_EPISODES=3 HELDOUT_EPISODES=20
readonly ENV_SEED=424243 BACKGROUND_SEED=1618034 PLANNER_SEED_BASE=8675400
readonly RUN_TAG=cutie_object_multitask_100k_v1
readonly FORMAT=cutie_object_multitask_generalization_v1
readonly BASE="$REPO_ROOT/logs/_generalization/${RUN_TAG}_seed${SEED}"
readonly STAGE="${BASE}.incomplete"
readonly SUMMARY="$STAGE/generalization_summary.json"
readonly SUPPORT_BASE="$REPO_ROOT/datasets/cutie_multitask_support_v1_seed${SUPPORT_SEED}"

readonly -a TASKS=(reacher-visual-small cup-catch cartpole-swingup finger-spin acrobot-swingup)
readonly -a ARMS=(rgb cutie_hybrid cutie_object_only)
# Static queues balance the three foreground-stress tasks across the two GPUs.
readonly -a GPU0_TASKS=(reacher-visual-small cup-catch)
readonly -a GPU1_TASKS=(cartpole-swingup finger-spin acrobot-swingup)

for name in GPU0 GPU1; do
	value="${!name}"
	[[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "$name is invalid: $value" >&2; exit 2; }
done
[[ "$GPU0" != "$GPU1" ]] || { echo "GPU0 and GPU1 must differ." >&2; exit 2; }
(( BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 1) )) || {
	echo "Bash >=5.1 is required." >&2; exit 2;
}
if [[ "$PY" == */* ]]; then
	[[ -x "$PY" ]] || { echo "Python is not executable: $PY" >&2; exit 2; }
else
	command -v "$PY" >/dev/null || { echo "Python is not on PATH: $PY" >&2; exit 2; }
fi
for path in "$VIDEO_ROOT" "$MANIFEST_DIR" "$OC_REPO" "$CUTIE_CKPT"; do
	[[ -e "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done
for path in tdmpc2/tools/collect_cutie_multitask_support.py tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py; do
	[[ -f "$path" ]] || { echo "Missing required tool: $path" >&2; exit 2; }
done
for path in "$BASE" "$STAGE" "$SUPPORT_BASE"; do
	[[ ! -e "$path" ]] || { echo "Refusing to overwrite: $path" >&2; exit 3; }
done

experiment_name() {
	local task=$1 arm=$2
	printf '%s100k_%s_seed%s_%s' "$arm" "$RUN_TAG" "$SEED" "${task//-/_}"
}

run_root() {
	local task=$1 arm=$2
	printf '%s/logs/%s/%s/%s' "$REPO_ROOT" "$task" "$SEED" "$(experiment_name "$task" "$arm")"
}

# Refuse the complete launch before writing anything if any arm already exists.
for task in "${TASKS[@]}"; do
	for arm in "${ARMS[@]}"; do
		path="$(run_root "$task" "$arm")"
		[[ ! -e "$path" ]] || { echo "Existing run would be overwritten: $path" >&2; exit 3; }
	done
done

mkdir -p "$STAGE/contracts" "$STAGE/support_logs" "$STAGE/tasks"
START_SECONDS="$(date +%s)"
ACTIVE_PIDS=()
PROMOTED=0

archive_on_exit() {
	local rc=$? pid failed
	trap - EXIT INT TERM
	for pid in "${ACTIVE_PIDS[@]:-}"; do
		[[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
	done
	for pid in "${ACTIVE_PIDS[@]:-}"; do
		[[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
	done
	if (( PROMOTED == 0 )) && [[ -d "$STAGE" ]]; then
		if [[ ! -f "$SUMMARY" ]]; then
			"$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps({"format":sys.argv[3],"status":"generalization_engineering_fail","runner_exit_code":int(sys.argv[2]),"failure":"runner exited before complete aggregation"},indent=2)+"\n",encoding="utf-8")' "$SUMMARY" "$rc" "$FORMAT" 2>/dev/null || true
		fi
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
		[[ ! -e "$failed" ]] || failed="${failed}.${RANDOM}"
		mv -- "$STAGE" "$failed"
		echo "GENERALIZATION_FAILED_ARCHIVE=$failed" >&2
	fi
	exit "$rc"
}
trap archive_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_contract() {
	local label=$1; shift
	echo "[contract] $label"
	{ echo "===== $label ====="; "$@"; } >>"$STAGE/contracts/contracts.log" 2>&1
}

echo "[1/5] Global contracts"
run_contract cutie_oc_adapter "$PY" tdmpc2/check_cutie_oc_adapter_contract.py
run_contract cutie_object_wrapper "$PY" tdmpc2/check_cutie_object_wrapper_contract.py
run_contract cutie_hybrid "$PY" tdmpc2/check_cutie_hybrid_contract.py
run_contract cutie_object_only "$PY" tdmpc2/check_cutie_object_only_contract.py
run_contract flat_anchor "$PY" tdmpc2/check_flat_anchor_contract.py
run_contract object_only_integration "$PY" tdmpc2/check_cutie_object_only_integration_contract.py
run_contract multitask_support "$PY" tdmpc2/check_cutie_multitask_support_contract.py
# Execute the self-contained module file directly. ``python -m tdmpc2.envs...``
# would import tdmpc2/envs/__init__.py before this file can normalize the
# repository's legacy top-level import path.
run_contract foreground_stress "$PY" tdmpc2/envs/wrappers/foreground_stress.py

echo "[2/5] Global GPU contracts"
run_contract "object_eager_gpu${GPU0}" env CUDA_VISIBLE_DEVICES="$GPU0" "$PY" tdmpc2/check_cutie_object_only_update.py
run_contract "object_compile_gpu${GPU0}" env CUDA_VISIBLE_DEVICES="$GPU0" "$PY" tdmpc2/check_cutie_object_only_update.py --compile
run_contract "object_eager_gpu${GPU1}" env CUDA_VISIBLE_DEVICES="$GPU1" "$PY" tdmpc2/check_cutie_object_only_update.py

write_rc() { printf '%s\n' "$2" >"$1"; }

read_support_roles() {
	"$PY" - "$1" "$2" <<'PY'
import json, re, sys
from pathlib import Path
p, task = Path(sys.argv[1]), sys.argv[2]
d = json.loads(p.read_text(encoding='utf-8'))
c, roles = d.get('collection', {}), d.get('roles')
if d.get('format') != 'cutie_indexed_mask_support_v1': raise ValueError(d.get('format'))
if c.get('task') != task or c.get('support_schema') != 'generic_indexed_v1': raise ValueError(c)
if not isinstance(roles, list) or len(roles) != 2 or len(set(roles)) != 2: raise ValueError(roles)
if any(not isinstance(x, str) or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', x) is None for x in roles): raise ValueError(roles)
print(*roles)
PY
}

write_preflight_config() {
	"$PY" - "$1" "$2" "$3" "$4" "$5" "$OC_REPO" "$CUTIE_CKPT" <<'PY'
import json, sys
from pathlib import Path
out, support, repo, ckpt = map(Path, (sys.argv[1], sys.argv[3], sys.argv[6], sys.argv[7]))
d = {
 'task':sys.argv[2], 'obs':'rgb', 'model_size':5, 'flat_anchor':True,
 'flat_anchor_mode':'cutie_object_only', 'cutie_object_repo':str(repo.resolve()),
 'cutie_object_checkpoint':str(ckpt.resolve()), 'cutie_object_support_path':str(support.resolve()),
 'cutie_object_support_schema':'generic_indexed_v1',
 'cutie_object_role_names':[sys.argv[4],sys.argv[5]],
 'cutie_object_allow_simulator_support':True, 'cutie_object_config_dir':None,
 'cutie_object_device':'cuda:0', 'cutie_object_tracker_height':448,
 'cutie_object_tracker_width':448, 'cutie_object_model_size':'small',
 'cutie_object_prompt_radius':2.0, 'cutie_object_amp':True,
 'cutie_object_worker_timeout_seconds':180.0,
}
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x',encoding='utf-8',newline='\n') as f: json.dump(d,f,indent=2,allow_nan=False); f.write('\n')
PY
}

run_training() {
	local gpu=$1 task=$2 arm=$3 support=$4 role0=$5 role1=$6 dir=$7
	local flat mode exp root log hydra rc
	case "$arm" in
		rgb) flat=false; mode=object_graph ;;
		cutie_hybrid) flat=true; mode=cutie_hybrid ;;
		cutie_object_only) flat=true; mode=cutie_object_only ;;
		*) return 2 ;;
	esac
	exp="$(experiment_name "$task" "$arm")"; root="$(run_root "$task" "$arm")"
	log="$dir/${arm}.train.log"; hydra="$dir/hydra_${arm}"
	local -a args=(
		"task=$task" obs=rgb model_size=5 "steps=$STEPS" "seed=$SEED"
		"eval_freq=$EVAL_FREQ" "eval_episodes=$EVAL_EPISODES"
		video_background_enabled=true "video_background_root=$VIDEO_ROOT"
		"video_background_manifest_dir=$MANIFEST_DIR" video_background_split=train
		"+video_background_seed=$SEED" video_background_strength=1.0
		video_background_total_frames=1000 video_background_source_cache_size=8
		visual_foreground_erosion_pixels=0 compile=true compile_fallback_random=true
		enable_wandb=false wandb_project=none wandb_entity=none save_csv=true
		save_video=false save_agent=true checkpoint=null data_dir=null obs_shapes=null
		action_dims=null episode_lengths=null "flat_anchor=$flat" "flat_anchor_mode=$mode"
		"cutie_object_repo=$OC_REPO" "cutie_object_checkpoint=$CUTIE_CKPT"
		"cutie_object_support_path=$support" cutie_object_support_schema=generic_indexed_v1
		"cutie_object_role_names=[$role0,$role1]" cutie_object_allow_simulator_support=true
		cutie_object_config_dir=null cutie_object_device=cuda:0
		cutie_object_tracker_height=448 cutie_object_tracker_width=448
		cutie_object_model_size=small cutie_object_prompt_radius=2.0 cutie_object_amp=true
		cutie_object_worker_timeout_seconds=180 cutie_object_num_roles=2
		cutie_object_frame_dim=590 cutie_object_stack_frames=3 cutie_object_input_dim=1770
		cutie_object_role_dim=64 cutie_object_hidden_dim=256 cutie_object_joint_dim=640
		cutie_object_only_latent_dim=128 "exp_name=$exp" "hydra.run.dir=$hydra"
		hydra.job.chdir=false
	)
	echo "TRAIN_START task=$task arm=$arm gpu=$gpu root=$root" | tee "$log"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" tdmpc2/train.py "${args[@]}" >>"$log" 2>&1
	rc=$?
	set -e
	write_rc "$dir/${arm}.train.rc" "$rc"
	echo "TRAIN_END task=$task arm=$arm gpu=$gpu rc=$rc" | tee -a "$log"
}

run_evaluation() {
	local gpu=$1 task=$2 arm=$3 erosion=$4 dir=$5 root runtime checkpoint out log rc ed
	ed="$dir/evaluations/erosion${erosion}"; mkdir -p "$ed"
	out="$ed/${arm}.json"; log="$ed/${arm}.log"; root="$(run_root "$task" "$arm")"
	runtime="$root/runtime_config.json"; checkpoint="$root/models/final.pt"
	if [[ "$(<"$dir/${arm}.train.rc")" != 0 ]]; then
		echo "Skipped because training failed." >"$log"; write_rc "$ed/${arm}.rc" 125; return 0
	fi
	if [[ ! -f "$runtime" || ! -f "$checkpoint" ]]; then
		echo "Missing successful-training artifacts." >"$log"; write_rc "$ed/${arm}.rc" 66; return 0
	fi
	echo "EVAL_START task=$task arm=$arm erosion=$erosion gpu=$gpu" | tee "$log"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" -m tdmpc2.tools.evaluate_cutie_multitask_checkpoint \
		--task "$task" --backend "$arm" --runtime-config "$runtime" --checkpoint "$checkpoint" \
		--training-seed "$SEED" --expected-training-steps "$STEPS" \
		--expected-training-eval-freq "$EVAL_FREQ" \
		--expected-training-eval-episodes "$EVAL_EPISODES" --episodes "$HELDOUT_EPISODES" \
		--env-seed "$ENV_SEED" --background-seed "$BACKGROUND_SEED" \
		--planner-seed-base "$PLANNER_SEED_BASE" --erosion-pixels "$erosion" \
		--output "$out" >>"$log" 2>&1
	rc=$?
	set -e
	write_rc "$ed/${arm}.rc" "$rc"
	echo "EVAL_END task=$task arm=$arm erosion=$erosion gpu=$gpu rc=$rc" | tee -a "$log"
}

skip_cutie_training() {
	local dir=$1 reason=$2 arm
	for arm in cutie_hybrid cutie_object_only; do
		echo "$reason" >"$dir/${arm}.train.log"; write_rc "$dir/${arm}.train.rc" 125
	done
}

run_task() {
	local gpu=$1 task=$2 dir="$STAGE/tasks/$2" support_dir="$SUPPORT_BASE/$2"
	local support="$SUPPORT_BASE/$2/annotations.json" rc roles role0 role1 extra preflight
	mkdir -p "$dir"; printf '%s\n' "$gpu" >"$dir/gpu"
	echo "SUPPORT_START task=$task gpu=$gpu"
	set +e
	env MUJOCO_GL=egl PYTHONUNBUFFERED=1 "$PY" -m tdmpc2.tools.collect_cutie_multitask_support \
		--task "$task" --video-root "$VIDEO_ROOT" --manifest-dir "$MANIFEST_DIR" \
		--output "$support_dir" --seed "$SUPPORT_SEED" >"$STAGE/support_logs/$task.log" 2>&1
	rc=$?
	set -e
	(( rc != 0 )) || [[ -f "$support" ]] || rc=66
	write_rc "$dir/support.rc" "$rc"
	if (( rc != 0 )); then
		write_rc "$dir/preflight.rc" 125
		run_training "$gpu" "$task" rgb /dev/null unused unused "$dir"
		skip_cutie_training "$dir" "Skipped: support collection failed rc=$rc."
	else
		set +e; roles="$(read_support_roles "$support" "$task" 2>>"$STAGE/support_logs/$task.log")"; rc=$?; set -e
		if (( rc != 0 )); then
			write_rc "$dir/preflight.rc" "$rc"
			run_training "$gpu" "$task" rgb /dev/null unused unused "$dir"
			skip_cutie_training "$dir" "Skipped: support role validation failed rc=$rc."
		else
			read -r role0 role1 extra <<<"$roles"
			printf '%s\n%s\n' "$role0" "$role1" >"$dir/roles"
			write_preflight_config "$support_dir/preflight_runtime_config.json" "$task" "$support" "$role0" "$role1"
			set +e
			env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
				"$PY" -m tdmpc2.tools.check_cutie_episode_reset_isolation \
				--runtime-config "$support_dir/preflight_runtime_config.json" \
				--output "$dir/reset_isolation.json" --pollution-length 32 >"$dir/preflight.log" 2>&1
			preflight=$?
			set -e; write_rc "$dir/preflight.rc" "$preflight"
			run_training "$gpu" "$task" rgb "$support" "$role0" "$role1" "$dir"
			if (( preflight == 0 )); then
				run_training "$gpu" "$task" cutie_hybrid "$support" "$role0" "$role1" "$dir"
				run_training "$gpu" "$task" cutie_object_only "$support" "$role0" "$role1" "$dir"
			else
				skip_cutie_training "$dir" "Skipped: Cutie worker/reset preflight failed rc=$preflight."
			fi
		fi
	fi
	local -a erosions=(0)
	case "$task" in reacher-visual-small|cup-catch|cartpole-swingup) erosions+=(1 2);; esac
	for erosion in "${erosions[@]}"; do
		for arm in "${ARMS[@]}"; do run_evaluation "$gpu" "$task" "$arm" "$erosion" "$dir"; done
	done
	echo "TASK_DONE task=$task gpu=$gpu"
}

gpu_worker() {
	local gpu=$1 task rc; shift
	for task in "$@"; do
		run_task "$gpu" "$task" || { rc=$?; mkdir -p "$STAGE/tasks/$task"; write_rc "$STAGE/tasks/$task/worker.rc" "$rc"; }
	done
}

echo "[3/5] Two-GPU task queues"
echo "GPU $GPU0: ${GPU0_TASKS[*]}"
echo "GPU $GPU1: ${GPU1_TASKS[*]}"
gpu_worker "$GPU0" "${GPU0_TASKS[@]}" & PID0=$!; ACTIVE_PIDS+=("$PID0")
gpu_worker "$GPU1" "${GPU1_TASKS[@]}" & PID1=$!; ACTIVE_PIDS+=("$PID1")
set +e
wait "$PID0"; WORKER0_RC=$?
wait "$PID1"; WORKER1_RC=$?
set -e
ACTIVE_PIDS=()
write_rc "$STAGE/gpu0_worker.rc" "$WORKER0_RC"
write_rc "$STAGE/gpu1_worker.rc" "$WORKER1_RC"

echo "[4/5] Strict aggregation"
set +e
"$PY" - "$STAGE" "$SUMMARY" "$REPO_ROOT" "$START_SECONDS" "$GPU0" "$GPU1" \
	"$SUPPORT_BASE" "$OC_REPO" "$CUTIE_CKPT" <<'PY'
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

stage, summary_path, repo = map(Path, sys.argv[1:4])
started, gpu0, gpu1 = int(sys.argv[4]), sys.argv[5], sys.argv[6]
support_base = Path(sys.argv[7])
expected_oc_repo, expected_cutie_checkpoint = map(Path, sys.argv[8:10])
tasks = ('reacher-visual-small', 'cup-catch', 'cartpole-swingup', 'finger-spin', 'acrobot-swingup')
arms = ('rgb', 'cutie_hybrid', 'cutie_object_only')
stress_tasks = {'reacher-visual-small', 'cup-catch', 'cartpole-swingup'}
steps, eval_freq, eval_episodes = 100000, 20000, 3
expected_steps = list(range(0, steps + 1, eval_freq))
expected_train_frames = steps + steps // 500 + (steps // eval_freq + 1) * eval_episodes * 501
alignment_contract = 'object_only_equivalent_cuda_randint_v1'
expected_shift_draws = 20 * 500
expected_evaluator = repo / 'tdmpc2' / 'tools' / 'evaluate_cutie_multitask_checkpoint.py'
pair_fields = (
    'initial_rgb_sha256', 'background_source', 'background_start_frame_index',
    'planner_seed', 'planner_rng_start_sha256', 'planner_rng_end_sha256', 'length',
)


def load_json(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def read_rc(path):
    try:
        return int(path.read_text(encoding='utf-8').strip())
    except Exception:
        return None


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def same_path(value, expected):
    try:
        return Path(value).resolve() == Path(expected).resolve()
    except (TypeError, ValueError, OSError):
        return False


def positive(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0


def runtime_gates(value, frames):
    if not isinstance(value, dict):
        return {'present': False}
    return {
        'present': True,
        'frames': value.get('frames') == frames,
        'valid_frame_rate': value.get('valid_frame_rate', -1) >= 0.95,
        'max_invalid_burst': value.get('max_invalid_burst', 10**9) <= 5,
        'worker_restarts': value.get('worker_restarts') == 0,
        'timeouts': value.get('timeouts') == 0,
        'ms_per_frame': positive(value.get('ms_per_frame')) and float(value['ms_per_frame']) <= 800,
        'runtime_unit': value.get('runtime_unit') == 'milliseconds_per_tracked_frame_excluding_support_prompts',
        'episode_reset_strategy': value.get('episode_reset_strategy') == 'fresh_inference_core_support_replay_v1',
    }


def gates_pass(value):
    return isinstance(value, dict) and bool(value) and all(value.values())


def exp_name(task, arm):
    return f'{arm}100k_cutie_object_multitask_100k_v1_seed6_{task.replace("-", "_")}'


jobs, structure, health, pairing_errors = [], [], [], []
evaluator_hashes, validation_manifest_hashes, combined_manifest_hashes = set(), set(), set()
reports = {}
for task in tasks:
    directory = stage / 'tasks' / task
    assigned_gpu = gpu0 if task in {'reacher-visual-small', 'cup-catch'} else gpu1
    support_path = support_base / task / 'annotations.json'
    support_rc, preflight_rc = read_rc(directory / 'support.rc'), read_rc(directory / 'preflight.rc')
    task_report = {
        'assigned_gpu': assigned_gpu,
        'support': {'rc': support_rc, 'path': str(support_path)},
        'preflight': {
            'rc': preflight_rc,
            'report_relative_to_summary_root': f'tasks/{task}/reset_isolation.json',
        },
        'training': {'arms': {}},
        'validation': {},
    }
    roles = None
    if support_rc == 0 and support_path.is_file():
        try:
            support = load_json(support_path)
            roles = support.get('roles')
            collection = support.get('collection', {})
            valid = (
                support.get('format') == 'cutie_indexed_mask_support_v1'
                and isinstance(roles, list) and len(roles) == 2 and len(set(roles)) == 2
                and collection.get('task') == task
                and collection.get('support_schema') == 'generic_indexed_v1'
            )
            task_report['support'].update({
                'sha256': digest(support_path), 'roles': roles,
                'format': support.get('format'), 'collection': collection,
            })
            if not valid:
                structure.append(f'{task}: invalid support contract')
        except Exception as exc:
            structure.append(f'{task}: support parse failed: {exc}')
    else:
        jobs.append(f'{task}: support rc={support_rc}')
    if preflight_rc != 0:
        jobs.append(f'{task}: preflight rc={preflight_rc}')
    else:
        reset_path = directory / 'reset_isolation.json'
        try:
            reset = load_json(reset_path)
            comparisons = reset.get('comparisons', {})
            workers = reset.get('workers', {})
            ready_values = [
                workers.get(name, {}).get('ready', {}) for name in ('forward', 'reverse')
            ]
            reset_checks = {
                'format': reset.get('format') == 'cutie_episode_reset_isolation_v1',
                'status': reset.get('status') == 'episode_reset_isolation_pass',
                'pass': reset.get('pass') is True,
                'task': reset.get('task') == task,
                'roles': reset.get('role_names') == roles,
                'support_schema': reset.get('support_schema') == 'generic_indexed_v1',
                'strategy': reset.get('reset_strategy') == 'fresh_inference_core_support_replay_v1',
                'comparisons': bool(comparisons) and all(
                    item.get('byte_equal') is True for item in comparisons.values()
                ),
                'worker_ready_roles': all(value.get('roles') == roles for value in ready_values),
                'worker_ready_strategy': all(
                    value.get('episode_reset_strategy') == 'fresh_inference_core_support_replay_v1'
                    for value in ready_values
                ),
            }
            task_report['preflight'].update({
                'sha256': digest(reset_path), 'checks': reset_checks,
                'workers': workers,
            })
            if not all(reset_checks.values()):
                structure.append(
                    f'{task}: preflight checks {[k for k,v in reset_checks.items() if not v]}'
                )
        except Exception as exc:
            structure.append(f'{task}: preflight report parse failed: {exc}')

    for arm in arms:
        root = repo / 'logs' / task / '6' / exp_name(task, arm)
        rc = read_rc(directory / f'{arm}.train.rc')
        arm_report = {'rc': rc, 'root': str(root)}
        task_report['training']['arms'][arm] = arm_report
        if rc != 0:
            jobs.append(f'{task}/{arm}: training rc={rc}')
            continue
        paths = {
            'config': root / 'runtime_config.json', 'eval': root / 'eval.csv',
            'checkpoint': root / 'models' / 'final.pt',
            'trainer': root / 'trainer_runtime.json', 'replay': root / 'replay_runtime.json',
        }
        if arm != 'rgb':
            paths['perception'] = root / 'perception_runtime.json'
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            arm_report['missing'] = missing
            structure.append(f'{task}/{arm}: missing artifacts {missing}')
            continue
        try:
            cfg, trainer, replay = (load_json(paths[key]) for key in ('config', 'trainer', 'replay'))
            with paths['eval'].open(encoding='utf-8', newline='') as file:
                rows = list(csv.DictReader(file))
            curve_steps = [int(float(row['step'])) for row in rows]
            rewards = [float(row['episode_reward']) for row in rows]
            if curve_steps != expected_steps or not all(math.isfinite(x) for x in rewards):
                raise ValueError(f'bad training eval curve: {curve_steps}, {rewards}')
            expected = {
                'rgb': (False, None, 512, {'rgb'}, None),
                'cutie_hybrid': (True, 'cutie_hybrid', 640, {'rgb', 'object'}, ['object', 'rgb']),
                'cutie_object_only': (True, 'cutie_object_only', 128, {'object'}, ['object']),
            }[arm]
            flat, mode, latent, obs_keys, replay_keys = expected
            checks = {
                'task_seed_protocol': cfg.get('task') == task and cfg.get('seed') == 6,
                'schedule': cfg.get('steps') == steps and cfg.get('eval_freq') == eval_freq and cfg.get('eval_episodes') == eval_episodes,
                'model_size': cfg.get('model_size') == 5,
                'mode': bool(cfg.get('flat_anchor')) is flat and (not flat or cfg.get('flat_anchor_mode') == mode),
                'latent_dim': cfg.get('latent_dim') == latent,
                'observation_keys': set(cfg.get('obs_shape', {})) == obs_keys,
                'replay_keys': replay.get('observation_keys') == replay_keys,
                'throughput': positive(trainer.get('training_non_eval_steps_per_second')),
                'replay_bytes': positive(replay.get('storage_required_bytes')),
                'generic_roles': arm == 'rgb' or (
                    cfg.get('cutie_object_support_schema') == 'generic_indexed_v1'
                    and cfg.get('cutie_object_role_names') == roles
                    and cfg.get('cutie_object_allow_simulator_support') is True
                ),
                'stable_support_path': arm == 'rgb' or same_path(
                    cfg.get('cutie_object_support_path'), support_path
                ),
                'cutie_repo_path': arm == 'rgb' or same_path(
                    cfg.get('cutie_object_repo'), expected_oc_repo
                ),
                'cutie_checkpoint_path': arm == 'rgb' or same_path(
                    cfg.get('cutie_object_checkpoint'), expected_cutie_checkpoint
                ),
            }
            if arm == 'cutie_object_only':
                checks['object_only_obs_shape'] = cfg.get('obs_shape') == {'object': [2, 1770]}
            payload = torch.load(paths['checkpoint'], map_location='cpu', weights_only=False)
            state = payload.get('model', payload) if isinstance(payload, dict) else payload
            if not isinstance(state, dict):
                raise ValueError('checkpoint state is not a mapping')
            keys = set(state)
            checks['checkpoint_finite'] = all(
                not torch.is_tensor(value) or bool(torch.isfinite(value).all()) for value in state.values()
            )
            rgb_keys = any(key.startswith('_encoder.rgb.') for key in keys)
            object_keys = any(key.startswith('_encoder.object.') for key in keys)
            hybrid_keys = any(key.startswith('_hybrid_') for key in keys)
            checks['checkpoint_structure'] = {
                'rgb': rgb_keys and not object_keys and not hybrid_keys,
                'cutie_hybrid': rgb_keys and object_keys and hybrid_keys,
                'cutie_object_only': object_keys and not rgb_keys and not hybrid_keys,
            }[arm]
            perception, perception_checks = None, None
            if arm != 'rgb':
                perception = load_json(paths['perception'])
                perception_checks = runtime_gates(perception, expected_train_frames)
                if not gates_pass(perception_checks):
                    health.append(f'{task}/{arm}: training runtime {perception_checks}')
            if not all(checks.values()):
                structure.append(f'{task}/{arm}: failed checks {[k for k,v in checks.items() if not v]}')
            arm_report.update({
                'structural_checks': checks,
                'training_eval_steps': curve_steps, 'training_eval_rewards': rewards,
                'throughput_non_eval_steps_per_second': trainer.get('training_non_eval_steps_per_second'),
                'trainer_runtime': trainer, 'replay_runtime': replay,
                'perception_runtime': perception, 'perception_gates': perception_checks,
                'checkpoint': str(paths['checkpoint']), 'checkpoint_sha256': digest(paths['checkpoint']),
            })
        except Exception as exc:
            structure.append(f'{task}/{arm}: artifact parse failed: {exc}')

    for erosion in ((0, 1, 2) if task in stress_tasks else (0,)):
        name, payloads = f'erosion{erosion}', {}
        condition = {'arms': {}, 'pairing': None}
        task_report['validation'][name] = condition
        for arm in arms:
            eval_dir = directory / 'evaluations' / name
            rc, output = read_rc(eval_dir / f'{arm}.rc'), eval_dir / f'{arm}.json'
            arm_report = {
                'rc': rc,
                'output_relative_to_summary_root': (
                    f'tasks/{task}/evaluations/{name}/{arm}.json'
                ),
            }
            condition['arms'][arm] = arm_report
            if rc != 0:
                jobs.append(f'{task}/{name}/{arm}: evaluation rc={rc}')
                continue
            if not output.is_file():
                structure.append(f'{task}/{name}/{arm}: missing evaluation JSON')
                continue
            try:
                payload = load_json(output)
                episodes = payload.get('episodes')
                evaluation = payload.get('evaluation', {})
                provenance = payload.get('provenance', {})
                cutie_inputs = provenance.get('cutie_inputs')
                cutie_ready = provenance.get('cutie_ready')
                expected_run = repo / 'logs' / task / '6' / exp_name(task, arm)
                expected_runtime = expected_run / 'runtime_config.json'
                expected_final = expected_run / 'models' / 'final.pt'
                base_provenance_valid = (
                    same_path(provenance.get('runtime_config'), expected_runtime)
                    and provenance.get('runtime_config_sha256') == digest(expected_runtime)
                    and same_path(provenance.get('checkpoint'), expected_final)
                    and provenance.get('checkpoint_sha256') == digest(expected_final)
                    and provenance.get('evaluator_sha256') == digest(expected_evaluator)
                    and provenance.get('cuda_visible_devices') == assigned_gpu
                    and isinstance(provenance.get('device_name'), str)
                    and bool(provenance.get('device_name'))
                )
                if arm == 'rgb':
                    cutie_inputs_valid = cutie_inputs is None and cutie_ready is None
                else:
                    cutie_inputs_valid = (
                        isinstance(cutie_inputs, dict)
                        and same_path(cutie_inputs.get('checkpoint'), expected_cutie_checkpoint)
                        and cutie_inputs.get('checkpoint_sha256') == digest(expected_cutie_checkpoint)
                        and same_path(cutie_inputs.get('support'), support_path)
                        and cutie_inputs.get('support_sha256') == digest(support_path)
                        and cutie_inputs.get('roles') == roles
                        and cutie_inputs.get('support_schema') == 'generic_indexed_v1'
                        and isinstance(cutie_ready, dict)
                        and cutie_ready.get('task') == task
                        and cutie_ready.get('support_task') == task
                        and cutie_ready.get('roles') == roles
                        and cutie_ready.get('support_schema') == 'generic_indexed_v1'
                        and cutie_ready.get('allow_simulator_support') is True
                        and cutie_ready.get('cuda_visible_devices') == assigned_gpu
                    )
                expected_alignment_draws = expected_shift_draws if arm == 'cutie_object_only' else 0
                valid = (
                    payload.get('format') == 'cutie_multitask_checkpoint_evaluation_v1'
                    and payload.get('task') == task and payload.get('backend') == arm
                    and payload.get('erosion_pixels') == erosion
                    and evaluation.get('rgb_shift_rng_alignment') == alignment_contract
                    and evaluation.get('expected_rgb_shift_draws_per_backend') == expected_shift_draws
                    and evaluation.get('object_only_alignment_draws') == expected_alignment_draws
                    and evaluation.get('actual_foreground_erosion_pixels') == erosion
                    and base_provenance_valid
                    and cutie_inputs_valid
                    and isinstance(episodes, list) and len(episodes) == 20
                    and [row.get('episode_index') for row in episodes] == list(range(20))
                )
                if not valid:
                    raise ValueError('evaluation envelope/RNG-alignment contract mismatch')
                values = [float(row['reward']) for row in episodes]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError('non-finite rewards')
                runtime, checks = payload.get('perception_runtime'), None
                if arm == 'rgb':
                    if runtime is not None:
                        raise ValueError('RGB unexpectedly exposes Cutie runtime')
                else:
                    checks = runtime_gates(runtime, 20 * 501)
                    if not gates_pass(checks):
                        health.append(f'{task}/{name}/{arm}: evaluation runtime {checks}')
                arm_report.update({
                    'reward_mean': statistics.fmean(values), 'reward_median': statistics.median(values),
                    'reward_sample_std': statistics.stdev(values), 'rewards': values,
                    'elapsed_seconds': payload.get('summary', {}).get('elapsed_seconds'),
                    'perception_runtime': runtime, 'perception_gates': checks,
                    'rgb_shift_rng_alignment': evaluation.get('rgb_shift_rng_alignment'),
                    'expected_rgb_shift_draws_per_backend': evaluation.get(
                        'expected_rgb_shift_draws_per_backend'
                    ),
                    'object_only_alignment_draws': evaluation.get('object_only_alignment_draws'),
                    'actual_foreground_erosion_pixels': evaluation.get(
                        'actual_foreground_erosion_pixels'
                    ),
                    'provenance': provenance,
                    'sha256': digest(output),
                })
                evaluator_hashes.add(provenance.get('evaluator_sha256'))
                validation_manifest_hashes.add(provenance.get('validation_manifest_sha256'))
                combined_manifest_hashes.add(provenance.get('combined_manifest_sha256'))
                payloads[arm] = payload
            except Exception as exc:
                structure.append(f'{task}/{name}/{arm}: evaluation parse failed: {exc}')

        if set(payloads) != set(arms):
            pairing_errors.append(f'{task}/{name}: incomplete arms {sorted(payloads)}')
            continue
        mismatches = {}
        episode_sets = [payloads[arm]['episodes'] for arm in arms]
        for field in pair_fields:
            mismatches[field] = [
                index for index, rows in enumerate(zip(*episode_sets))
                if len({row.get(field) for row in rows}) != 1
            ]
        mismatches['initial_object_sha256_hybrid_vs_object_only'] = [
            index for index, (left, right) in enumerate(zip(
                payloads['cutie_hybrid']['episodes'], payloads['cutie_object_only']['episodes']
            )) if not left.get('initial_object_sha256')
            or left.get('initial_object_sha256') != right.get('initial_object_sha256')
        ]
        condition['pairing'] = {'exact': not any(mismatches.values()), 'mismatch_episode_indices': mismatches}
        if any(mismatches.values()):
            pairing_errors.append(f'{task}/{name}: {mismatches}')
        device_names = {
            payloads[arm].get('provenance', {}).get('device_name') for arm in arms
        }
        if len(device_names) != 1 or None in device_names:
            pairing_errors.append(f'{task}/{name}: device names differ {sorted(map(str, device_names))}')
    reports[task] = task_report

for label, values in (
    ('evaluator_sha256', evaluator_hashes),
    ('validation_manifest_sha256', validation_manifest_hashes),
    ('combined_manifest_sha256', combined_manifest_hashes),
):
    if len(values) != 1 or None in values:
        structure.append(f'cross-evaluation {label} mismatch: {sorted(map(str, values))}')

worker_rcs = {'gpu0': read_rc(stage / 'gpu0_worker.rc'), 'gpu1': read_rc(stage / 'gpu1_worker.rc')}
if any(value != 0 for value in worker_rcs.values()):
    jobs.append(f'worker return codes {worker_rcs}')

per_task, ratio_count, rgb_count = {}, 0, 0
for task in tasks:
    arm_reports = reports[task]['validation']['erosion0']['arms']
    means = {arm: arm_reports[arm].get('reward_mean') for arm in arms}
    complete = all(isinstance(value, (int, float)) and math.isfinite(value) for value in means.values())
    ratio_pass = bool(complete and means['cutie_object_only'] >= 0.8 * means['cutie_hybrid'])
    rgb_pass = bool(complete and means['cutie_object_only'] > means['rgb'])
    ratio_count += int(ratio_pass); rgb_count += int(rgb_pass)
    per_task[task] = {
        'erosion0_reward_means': means,
        'object_only_minus_hybrid': means['cutie_object_only'] - means['cutie_hybrid'] if complete else None,
        'object_only_minus_rgb': means['cutie_object_only'] - means['rgb'] if complete else None,
        'object_only_at_least_80pct_hybrid': ratio_pass,
        'object_only_better_than_rgb': rgb_pass,
    }

reward = {
    'scope': 'single-seed 20-episode validation outcome; not an engineering gate or final algorithm claim',
    'per_task': per_task,
    'object_only_at_least_80pct_hybrid': {'count': ratio_count, 'required': 4, 'pass': ratio_count >= 4},
    'object_only_better_than_rgb': {'count': rgb_count, 'required': 3, 'pass': rgb_count >= 3},
}
reward['go_criteria_pass'] = (
    reward['object_only_at_least_80pct_hybrid']['pass']
    and reward['object_only_better_than_rgb']['pass']
)


def finite_metric(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def ratio(numerator, denominator):
    numerator, denominator = finite_metric(numerator), finite_metric(denominator)
    return numerator / denominator if numerator is not None and denominator is not None and denominator > 0 else None


efficiency_per_task = {}
speed_oh, storage_oh, speed_or, storage_or = [], [], [], []
for task in tasks:
    training = reports[task]['training']['arms']
    throughput = {
        arm: finite_metric(training[arm].get('throughput_non_eval_steps_per_second'))
        for arm in arms
    }
    replay_bytes = {
        arm: finite_metric(training[arm].get('replay_runtime', {}).get('storage_required_bytes'))
        for arm in arms
    }
    ratios = {
        'object_only_over_hybrid_speed': ratio(
            throughput['cutie_object_only'], throughput['cutie_hybrid']
        ),
        'object_only_over_hybrid_storage': ratio(
            replay_bytes['cutie_object_only'], replay_bytes['cutie_hybrid']
        ),
        'object_only_over_rgb_speed': ratio(throughput['cutie_object_only'], throughput['rgb']),
        'object_only_over_rgb_storage': ratio(replay_bytes['cutie_object_only'], replay_bytes['rgb']),
    }
    efficiency_per_task[task] = {
        'non_eval_steps_per_second': throughput,
        'replay_storage_required_bytes': replay_bytes,
        'ratios': ratios,
    }
    for value, destination in (
        (ratios['object_only_over_hybrid_speed'], speed_oh),
        (ratios['object_only_over_hybrid_storage'], storage_oh),
        (ratios['object_only_over_rgb_speed'], speed_or),
        (ratios['object_only_over_rgb_storage'], storage_or),
    ):
        if value is not None:
            destination.append(value)


def aggregate_ratio(values):
    return {'values': values, 'count': len(values), 'median': statistics.median(values) if values else None}


efficiency = {
    'scope': (
        'within-task same-GPU sequential engineering throughput; descriptive only, '
        'not an algorithm or engineering pass gate'
    ),
    'ratio_interpretation': {
        'speed': 'greater than 1 means ObjectOnly has higher non-eval steps/s',
        'storage': 'less than 1 means ObjectOnly replay uses fewer bytes',
    },
    'per_task': efficiency_per_task,
    'aggregate': {
        'object_only_over_hybrid_speed': aggregate_ratio(speed_oh),
        'object_only_over_hybrid_storage': aggregate_ratio(storage_oh),
        'object_only_over_rgb_speed': aggregate_ratio(speed_or),
        'object_only_over_rgb_storage': aggregate_ratio(storage_or),
    },
}

stress_per_task = {}
for task in tasks:
    if task not in stress_tasks:
        continue
    condition_means = {}
    for erosion in (0, 1, 2):
        arm_reports = reports[task]['validation'][f'erosion{erosion}']['arms']
        means = {arm: finite_metric(arm_reports[arm].get('reward_mean')) for arm in arms}
        condition_means[f'erosion{erosion}'] = {
            'reward_means': means,
            'object_only_minus_hybrid': (
                means['cutie_object_only'] - means['cutie_hybrid']
                if means['cutie_object_only'] is not None and means['cutie_hybrid'] is not None else None
            ),
            'object_only_minus_rgb': (
                means['cutie_object_only'] - means['rgb']
                if means['cutie_object_only'] is not None and means['rgb'] is not None else None
            ),
        }
    retention = {}
    for arm in arms:
        base = condition_means['erosion0']['reward_means'][arm]
        retention[arm] = {
            'erosion1_over_erosion0': ratio(
                condition_means['erosion1']['reward_means'][arm], base
            ),
            'erosion2_over_erosion0': ratio(
                condition_means['erosion2']['reward_means'][arm], base
            ),
        }
    stress_per_task[task] = {'conditions': condition_means, 'stress_retention': retention}

stress = {
    'scope': '20-episode validation foreground-erosion outcome; descriptive reward evidence only',
    'retention_definition': 'mean reward at erosion N divided by erosion-0 mean when erosion-0 mean > 0',
    'per_task': stress_per_task,
}
engineering = {
    'all_jobs_completed': not jobs,
    'training_checkpoint_and_support_structure': not structure,
    'cutie_runtime_health': not health,
    'strict_initial_background_planner_pairing': not pairing_errors,
}
engineering_pass = all(engineering.values())
summary = {
    'format': 'cutie_object_multitask_generalization_v1',
    'status': 'generalization_engineering_pass' if engineering_pass else 'generalization_engineering_fail',
    'scientific_scope': 'five-task seed-6 development feasibility with simulator-derived support-only masks',
    'protocol': {
        'tasks': list(tasks), 'arms': list(arms), 'training_seed': 6, 'support_seed': 314159,
        'steps': steps, 'eval_freq': eval_freq, 'training_eval_episodes': eval_episodes,
        'heldout_episodes': 20,
        'validation_erosions': {task: ([0,1,2] if task in stress_tasks else [0]) for task in tasks},
        'gpu_queues': {
            'gpu0': {'physical_index': gpu0, 'tasks': ['reacher-visual-small','cup-catch']},
            'gpu1': {'physical_index': gpu1, 'tasks': ['cartpole-swingup','finger-spin','acrobot-swingup']},
        },
        'within_task_serial_order': list(arms),
        'stable_support_root': str(support_base),
        'rgb_shift_rng_alignment': alignment_contract,
        'expected_rgb_shift_draws_per_backend': expected_shift_draws,
        'strict_pairing_fields': list(pair_fields) + ['initial_object_sha256_hybrid_vs_object_only'],
    },
    'engineering_gates': engineering,
    'failures': {'jobs': jobs, 'structure': structure, 'runtime_health': health, 'pairing': pairing_errors},
    'reward_outcome': reward,
    'efficiency_outcome': efficiency,
    'stress_outcome': stress,
    'recommendation': (
        'continue_object_only_training' if engineering_pass and reward['go_criteria_pass']
        else ('stop_and_fix_engineering_before_reward_interpretation' if not engineering_pass
              else 'do_not_scale_yet_reward_go_criteria_not_met')
    ),
    'worker_return_codes': worker_rcs,
    'elapsed_seconds': int(time.time()) - started,
    'tasks': reports,
}
temporary = summary_path.with_name(summary_path.name + '.tmp')
with temporary.open('x', encoding='utf-8', newline='\n') as file:
    json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    file.write('\n')
temporary.replace(summary_path)
print(json.dumps({
    'status': summary['status'], 'engineering_gates': engineering,
    'reward_outcome': reward, 'recommendation': summary['recommendation'],
    'summary': str(summary_path),
}, ensure_ascii=False, indent=2, allow_nan=False))
raise SystemExit(0 if engineering_pass else 4)
PY
SUMMARY_RC=$?
set -e
if (( SUMMARY_RC != 0 )); then exit "$SUMMARY_RC"; fi

echo "[5/5] Promoting canonical result"
mv -- "$STAGE" "$BASE"
PROMOTED=1
trap - EXIT INT TERM
echo "CUTIE_MULTITASK_GENERALIZATION_COMPLETE"
echo "SUMMARY=$BASE/generalization_summary.json"
