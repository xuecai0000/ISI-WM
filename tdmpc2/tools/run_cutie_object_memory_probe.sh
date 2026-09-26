#!/usr/bin/env bash
# Frozen two-GPU diagnostic for the Cutie ObjectOnly last-valid-memory switch.
#
# This is deliberately not a paper protocol. Both tasks use the existing
# simulator-segmentation-derived support packs. It evaluates the normal
# validation video split plus canonical policy-input missing-role bursts at the
# wrapper's raw590 -> burst -> last-valid -> 3-frame-stack boundary. These are
# controlled synthetic interventions, not visual occlusions or learned belief.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${VIDEO_ROOT:?Set VIDEO_ROOT to the existing video_hard directory}"
: "${OC_REPO:?Set OC_REPO to the OC-STORM checkout}"
: "${CUTIE_CKPT:?Set CUTIE_CKPT to cutie-small-mega.pth}"

PY="${PY:-python}"
MANIFEST_DIR="${MANIFEST_DIR:-$REPO_ROOT/tdmpc2/envs/background_manifests}"
SUPPORT_BASE="${SUPPORT_BASE:-$REPO_ROOT/datasets/cutie_multitask_support_v1_seed314159}"
GPU_REACHER="${GPU_REACHER:-0}"
GPU_CARTPOLE="${GPU_CARTPOLE:-1}"

readonly SEED=7 SUPPORT_SEED=314159 STEPS=100000 EVAL_FREQ=20000
readonly EVAL_EPISODES=3 HELDOUT_EPISODES=20
readonly ENV_SEED=424243 BACKGROUND_SEED=1618034 PLANNER_SEED_BASE=8675400
readonly RUN_TAG=cutie_object_memory_probe_100k_v1
readonly FORMAT=cutie_object_memory_probe_v1
readonly BASE="$REPO_ROOT/logs/_diagnostic/${RUN_TAG}_seed${SEED}"
readonly STAGE="${BASE}.incomplete"
readonly SUMMARY="$STAGE/memory_probe_summary.json"
readonly -a TASKS=(reacher-visual-small cartpole-swingup)
readonly -a ARMS=(hard_zero last_valid)

for name in GPU_REACHER GPU_CARTPOLE; do
	value="${!name}"
	[[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || {
		echo "$name is invalid: $value" >&2
		exit 2
	}
done
[[ "$GPU_REACHER" != "$GPU_CARTPOLE" ]] || {
	echo "GPU_REACHER and GPU_CARTPOLE must differ." >&2
	exit 2
}
(( BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 1) )) || {
	echo "Bash >=5.1 is required." >&2
	exit 2
}
if [[ "$PY" == */* ]]; then
	[[ -x "$PY" ]] || { echo "Python is not executable: $PY" >&2; exit 2; }
else
	command -v "$PY" >/dev/null || { echo "Python is not on PATH: $PY" >&2; exit 2; }
fi
for path in "$VIDEO_ROOT" "$MANIFEST_DIR" "$SUPPORT_BASE" "$OC_REPO" "$CUTIE_CKPT"; do
	[[ -e "$path" ]] || { echo "Missing required immutable input: $path" >&2; exit 2; }
done
[[ "$(basename -- "${VIDEO_ROOT%/}")" == video_hard ]] || {
	echo "VIDEO_ROOT must be the frozen video_hard directory, got: $VIDEO_ROOT" >&2
	exit 2
}
for path in \
	tdmpc2/config.yaml \
	tdmpc2/train.py \
	tdmpc2/check_cutie_last_valid_memory_contract.py \
	tdmpc2/check_cutie_policy_burst_contract.py \
	tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py \
	tdmpc2/tools/evaluate_cutie_multitask_policy_burst.py \
	tdmpc2/tools/check_cutie_episode_reset_isolation.py; do
	[[ -f "$path" ]] || { echo "Missing required repository file: $path" >&2; exit 2; }
done
for task in "${TASKS[@]}"; do
	[[ -f "$SUPPORT_BASE/$task/annotations.json" ]] || {
		echo "Missing frozen diagnostic support: $SUPPORT_BASE/$task/annotations.json" >&2
		exit 2
	}
done

arm_memory() {
	case "$1" in
		hard_zero) printf '%s' false ;;
		last_valid) printf '%s' true ;;
		*) return 2 ;;
	esac
}

experiment_name() {
	local task=$1 arm=$2
	printf 'cutie_object_%s100k_%s_seed%s_%s' \
		"$arm" "$RUN_TAG" "$SEED" "${task//-/_}"
}

run_root() {
	local task=$1 arm=$2
	printf '%s/logs/%s/%s/%s' "$REPO_ROOT" "$task" "$SEED" \
		"$(experiment_name "$task" "$arm")"
}

# Refuse every owned output before creating the staging directory. Training
# outputs live outside STAGE, so this also prevents a failed run from being
# accidentally resumed or mixed with a later invocation.
for path in "$BASE" "$STAGE"; do
	[[ ! -e "$path" ]] || { echo "Refusing existing runner output: $path" >&2; exit 3; }
done
for task in "${TASKS[@]}"; do
	for arm in "${ARMS[@]}"; do
		path="$(run_root "$task" "$arm")"
		[[ ! -e "$path" ]] || { echo "Refusing existing training root: $path" >&2; exit 3; }
	done
done

mkdir -p "$STAGE/contracts" "$STAGE/provenance" "$STAGE/tasks"
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
			"$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps({"format":sys.argv[3],"status":"object_memory_probe_engineering_fail","runner_exit_code":int(sys.argv[2]),"failure":"runner exited before complete aggregation","scientific_scope":"diagnostic only; simulator-derived oracle support; normal held-out plus synthetic policy-input bursts"},indent=2)+"\n",encoding="utf-8")' "$SUMMARY" "$rc" "$FORMAT" 2>/dev/null || true
		fi
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
		[[ ! -e "$failed" ]] || failed="${failed}.${RANDOM}"
		mv -- "$STAGE" "$failed"
		echo "CUTIE_OBJECT_MEMORY_PROBE_FAILED_ARCHIVE=$failed" >&2
	fi
	exit "$rc"
}
trap archive_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

write_rc() { printf '%s\n' "$2" >"$1"; }

run_contract() {
	local label=$1
	shift
	echo "[contract] $label"
	{ echo "===== $label ====="; "$@"; } >>"$STAGE/contracts/contracts.log" 2>&1
}

echo "[1/6] Dependency-light contracts"
run_contract cutie_object_wrapper "$PY" tdmpc2/check_cutie_object_wrapper_contract.py
run_contract cutie_object_only "$PY" tdmpc2/check_cutie_object_only_contract.py
run_contract last_valid_memory "$PY" tdmpc2/check_cutie_last_valid_memory_contract.py
run_contract policy_burst "$PY" -m tdmpc2.check_cutie_policy_burst_contract
run_contract object_only_integration "$PY" tdmpc2/check_cutie_object_only_integration_contract.py
run_contract multitask_support "$PY" tdmpc2/check_cutie_multitask_support_contract.py

# Bind the new switch and the two pre-existing privileged support packs before
# any CUDA process starts. This report is immutable provenance for aggregation.
"$PY" - "$REPO_ROOT" "$SUPPORT_BASE" "$VIDEO_ROOT" "$MANIFEST_DIR" \
	"$OC_REPO" "$CUTIE_CKPT" "$STAGE/provenance/inputs.json" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

repo, support_base, video_root, manifest_dir, oc_repo = map(
    Path, sys.argv[1:6]
)
checkpoint, output = map(Path, sys.argv[6:8])

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

config = repo / 'tdmpc2' / 'config.yaml'
text = config.read_text(encoding='utf-8')
matches = re.findall(
    r'^cutie_object_last_valid_memory\s*:\s*(true|false)\s*(?:#.*)?$',
    text,
    flags=re.MULTILINE | re.IGNORECASE,
)
if matches != ['false']:
    raise ValueError(
        'config.yaml must declare exactly one fail-safe default '
        f'cutie_object_last_valid_memory: false; got {matches!r}'
    )
plan_defaults = re.findall(
    r'^cutie_object_policy_burst_plan\s*:\s*(null)\s*(?:#.*)?$',
    text,
    flags=re.MULTILINE | re.IGNORECASE,
)
if plan_defaults != ['null']:
    raise ValueError(
        'config.yaml must declare exactly one fail-safe default '
        f'cutie_object_policy_burst_plan: null; got {plan_defaults!r}'
    )

expected_roles = {
    'reacher-visual-small': ['whole_arm', 'goal'],
    'cartpole-swingup': ['cart', 'pole'],
}
supports = {}
for task, roles in expected_roles.items():
    path = support_base / task / 'annotations.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    collection = data.get('collection', {})
    checks = {
        'format': data.get('format') == 'cutie_indexed_mask_support_v1',
        'roles': data.get('roles') == roles,
        'task': collection.get('task') == task,
        'schema': collection.get('support_schema') == 'generic_indexed_v1',
        'support_split': collection.get('split') == 'support',
        'oracle_label_policy': (
            collection.get('label_policy') == 'simulator_segmentation_support_only'
        ),
        'diagnostic_support': collection.get('diagnostic_support') is True,
    }
    if not all(checks.values()):
        raise ValueError(f'{task} support is not the frozen diagnostic oracle pack: {checks}')
    supports[task] = {
        'path': str(path.resolve()),
        'sha256': digest(path),
        'roles': roles,
        'checks': checks,
        'label_source': 'MuJoCo simulator segmentation on support split only',
    }

implementation = {}
for relative in (
    'tdmpc2/config.yaml',
    'tdmpc2/train.py',
    'tdmpc2/tdmpc2.py',
    'tdmpc2/common/layers.py',
    'tdmpc2/common/buffer.py',
    'tdmpc2/envs/wrappers/cutie_object.py',
    'tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py',
    'tdmpc2/tools/evaluate_cutie_multitask_policy_burst.py',
    'tdmpc2/check_cutie_last_valid_memory_contract.py',
    'tdmpc2/check_cutie_policy_burst_contract.py',
    'tdmpc2/tools/run_cutie_object_memory_probe.sh',
):
    path = repo / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    implementation[relative] = {'path': str(path.resolve()), 'sha256': digest(path)}

plans_root = output.parent.parent / 'plans'
targets = {
    'reacher-visual-small': 'whole_arm',
    'cartpole-swingup': 'pole',
}
starts = [75, 150, 225, 300, 375]
plans = {}
for task, roles in expected_roles.items():
    task_dir = plans_root / task
    task_dir.mkdir(parents=True, exist_ok=False)
    plans[task] = {}
    for length in (5, 20, 50):
        payload = {
            'format': 'cutie_policy_burst_plan_v1',
            'task': task,
            'roles': roles,
            'episodes': 20,
            'decision_steps': 500,
            'frame_dim': 590,
            'stack_frames': 3,
            'invalid_encoding': 'empty_lost_v1',
            'events': [
                {
                    'episode_index': episode,
                    'role': targets[task],
                    'start_decision_step': starts[episode % len(starts)],
                    'length': length,
                }
                for episode in range(20)
            ],
        }
        raw = (
            json.dumps(
                payload, sort_keys=True, separators=(',', ':'),
                ensure_ascii=True, allow_nan=False,
            ) + '\n'
        ).encode('utf-8')
        path = task_dir / f'burst_{length}.json'
        with path.open('xb') as file:
            file.write(raw)
        plans[task][str(length)] = {
            'path': str(path.resolve()),
            'relative_to_summary_root': f'plans/{task}/burst_{length}.json',
            'sha256': hashlib.sha256(raw).hexdigest(),
            'target_role': targets[task],
            'length': length,
            'starts': [starts[episode % len(starts)] for episode in range(20)],
            'events': payload['events'],
        }

payload = {
    'format': 'cutie_object_memory_probe_inputs_v1',
    'scientific_scope': (
        'diagnostic/oracle-support feasibility only; simulator-derived support masks '
        'are privileged; controlled policy-input bursts are synthetic rather than '
        'visual occlusions; this is not a paper-comparable RGB-only protocol'
    ),
    'config': {
        'path': str(config.resolve()),
        'sha256': digest(config),
        'cutie_object_last_valid_memory_default': False,
        'cutie_object_policy_burst_plan_default': None,
    },
    'video_root': str(video_root.resolve()),
    'manifest_dir': str(manifest_dir.resolve()),
    'oc_repo': str(oc_repo.resolve()),
    'cutie_checkpoint': {
        'path': str(checkpoint.resolve()),
        'sha256': digest(checkpoint),
    },
    'supports': supports,
    'implementation': implementation,
    'plans': plans,
}
with output.open('x', encoding='utf-8', newline='\n') as file:
    json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
    file.write('\n')
print('CUTIE_OBJECT_MEMORY_INPUTS_OK', json.dumps({
    'output': str(output), 'tasks': list(expected_roles),
    'diagnostic_oracle_support': True,
}))
PY

run_gpu_contract() {
	local label=$1 gpu=$2 log="$STAGE/contracts/${1}.log" rc
	echo "GPU_CONTRACT_START label=$label gpu=$gpu"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
		"$PY" tdmpc2/check_cutie_object_only_update.py --compile >"$log" 2>&1
	rc=$?
	set -e
	write_rc "$STAGE/contracts/${label}.rc" "$rc"
	echo "GPU_CONTRACT_END label=$label gpu=$gpu rc=$rc"
	return "$rc"
}

echo "[2/6] Matched object-only compile preflight on both GPUs"
set +e
run_gpu_contract gpu_reacher "$GPU_REACHER" & PID_GPU_REACHER=$!
ACTIVE_PIDS+=("$PID_GPU_REACHER")
run_gpu_contract gpu_cartpole "$GPU_CARTPOLE" & PID_GPU_CARTPOLE=$!
ACTIVE_PIDS+=("$PID_GPU_CARTPOLE")
wait "$PID_GPU_REACHER"; GPU_REACHER_PREFLIGHT_RC=$?
wait "$PID_GPU_CARTPOLE"; GPU_CARTPOLE_PREFLIGHT_RC=$?
set -e
ACTIVE_PIDS=()
if (( GPU_REACHER_PREFLIGHT_RC != 0 || GPU_CARTPOLE_PREFLIGHT_RC != 0 )); then
	echo "GPU compile preflight failed: reacher=$GPU_REACHER_PREFLIGHT_RC cartpole=$GPU_CARTPOLE_PREFLIGHT_RC" >&2
	exit 4
fi

write_preflight_config() {
	local output=$1 task=$2 support=$3 role0=$4 role1=$5 memory=$6
	"$PY" - "$output" "$task" "$support" "$role0" "$role1" "$memory" \
		"$OC_REPO" "$CUTIE_CKPT" <<'PY'
import json
import sys
from pathlib import Path

output, support, repo, checkpoint = map(Path, (sys.argv[1], sys.argv[3], sys.argv[7], sys.argv[8]))
memory = {'true': True, 'false': False}[sys.argv[6]]
payload = {
    'task': sys.argv[2],
    'obs': 'rgb',
    'model_size': 5,
    'flat_anchor': True,
    'flat_anchor_mode': 'cutie_object_only',
    'cutie_object_repo': str(repo.resolve()),
    'cutie_object_checkpoint': str(checkpoint.resolve()),
    'cutie_object_support_path': str(support.resolve()),
    'cutie_object_support_schema': 'generic_indexed_v1',
    'cutie_object_role_names': [sys.argv[4], sys.argv[5]],
    'cutie_object_allow_simulator_support': True,
    'cutie_object_last_valid_memory': memory,
    'cutie_object_policy_burst_plan': None,
    'cutie_object_config_dir': None,
    'cutie_object_device': 'cuda:0',
    'cutie_object_tracker_height': 448,
    'cutie_object_tracker_width': 448,
    'cutie_object_model_size': 'small',
    'cutie_object_prompt_radius': 2.0,
    'cutie_object_amp': True,
    'cutie_object_worker_timeout_seconds': 180.0,
}
output.parent.mkdir(parents=True, exist_ok=True)
with output.open('x', encoding='utf-8', newline='\n') as file:
    json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
    file.write('\n')
PY
}

run_training() {
	local gpu=$1 task=$2 arm=$3 support=$4 role0=$5 role1=$6 dir=$7
	local exp root log hydra rc memory
	memory="$(arm_memory "$arm")"
	exp="$(experiment_name "$task" "$arm")"
	root="$(run_root "$task" "$arm")"
	log="$dir/${arm}.train.log"
	hydra="$dir/hydra_${arm}"
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
		action_dims=null episode_lengths=null flat_anchor=true flat_anchor_mode=cutie_object_only
		"cutie_object_repo=$OC_REPO" "cutie_object_checkpoint=$CUTIE_CKPT"
		"cutie_object_support_path=$support" cutie_object_support_schema=generic_indexed_v1
		"cutie_object_role_names=[$role0,$role1]" cutie_object_allow_simulator_support=true
		"cutie_object_last_valid_memory=$memory"
		cutie_object_policy_burst_plan=null
		cutie_object_config_dir=null cutie_object_device=cuda:0
		cutie_object_tracker_height=448 cutie_object_tracker_width=448
		cutie_object_model_size=small cutie_object_prompt_radius=2.0 cutie_object_amp=true
		cutie_object_worker_timeout_seconds=180 cutie_object_num_roles=2
		cutie_object_frame_dim=590 cutie_object_stack_frames=3 cutie_object_input_dim=1770
		cutie_object_role_dim=64 cutie_object_hidden_dim=256 cutie_object_joint_dim=640
		cutie_object_only_latent_dim=128 "exp_name=$exp" "hydra.run.dir=$hydra"
		hydra.job.chdir=false
	)
	echo "TRAIN_START task=$task arm=$arm memory=$memory gpu=$gpu root=$root" | tee "$log"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" tdmpc2/train.py "${args[@]}" >>"$log" 2>&1
	rc=$?
	set -e
	write_rc "$dir/${arm}.train.rc" "$rc"
	echo "TRAIN_END task=$task arm=$arm memory=$memory gpu=$gpu rc=$rc" | tee -a "$log"
}

run_evaluation() {
	local gpu=$1 task=$2 arm=$3 dir=$4 root outdir out log rc
	root="$(run_root "$task" "$arm")"
	outdir="$dir/evaluations/normal"
	mkdir -p "$outdir"
	out="$outdir/${arm}.json"
	log="$outdir/${arm}.log"
	if [[ -e "$out" || -e "${out}.incomplete" ]]; then
		echo "Refusing existing evaluation output: $out" >&2
		return 3
	fi
	if [[ ! -f "$root/runtime_config.json" || ! -f "$root/models/final.pt" ]]; then
		echo "Missing successful-training artifacts: $root" >"$log"
		write_rc "$outdir/${arm}.rc" 66
		return 0
	fi
	echo "EVAL_START task=$task arm=$arm condition=normal gpu=$gpu root=$root" | tee "$log"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" -m tdmpc2.tools.evaluate_cutie_multitask_checkpoint \
		--task "$task" --backend cutie_object_only \
		--runtime-config "$root/runtime_config.json" --checkpoint "$root/models/final.pt" \
		--training-seed "$SEED" --expected-training-steps "$STEPS" \
		--expected-training-eval-freq "$EVAL_FREQ" \
		--expected-training-eval-episodes "$EVAL_EPISODES" --episodes "$HELDOUT_EPISODES" \
		--env-seed "$ENV_SEED" --background-seed "$BACKGROUND_SEED" \
		--planner-seed-base "$PLANNER_SEED_BASE" --erosion-pixels 0 --output "$out" \
		>>"$log" 2>&1
	rc=$?
	set -e
	write_rc "$outdir/${arm}.rc" "$rc"
	echo "EVAL_END task=$task arm=$arm condition=normal gpu=$gpu rc=$rc" | tee -a "$log"
}

run_burst_evaluation() {
	local gpu=$1 task=$2 arm=$3 length=$4 dir=$5 root plan role outdir out log rc
	root="$(run_root "$task" "$arm")"
	plan="$STAGE/plans/$task/burst_${length}.json"
	case "$task" in
		reacher-visual-small) role=whole_arm ;;
		cartpole-swingup) role=pole ;;
		*) return 2 ;;
	esac
	outdir="$dir/evaluations/burst_${length}"
	mkdir -p "$outdir"
	out="$outdir/${arm}.json"
	log="$outdir/${arm}.log"
	if [[ -e "$out" || -e "${out}.incomplete" ]]; then
		echo "Refusing existing burst evaluation output: $out" >&2
		return 3
	fi
	if [[ ! -f "$plan" ]]; then
		echo "Missing canonical burst plan: $plan" >"$log"
		write_rc "$outdir/${arm}.rc" 66
		return 0
	fi
	if [[ ! -f "$root/runtime_config.json" || ! -f "$root/models/final.pt" ]]; then
		echo "Missing successful-training artifacts: $root" >"$log"
		write_rc "$outdir/${arm}.rc" 66
		return 0
	fi
	echo "EVAL_START task=$task arm=$arm condition=burst_${length} role=$role gpu=$gpu root=$root" | tee "$log"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" -m tdmpc2.tools.evaluate_cutie_multitask_policy_burst \
		--task "$task" --arm "$arm" --policy-burst-plan "$plan" \
		--expected-role "$role" --expected-length "$length" \
		--runtime-config "$root/runtime_config.json" --checkpoint "$root/models/final.pt" \
		--training-seed "$SEED" --expected-training-steps "$STEPS" \
		--expected-training-eval-freq "$EVAL_FREQ" \
		--expected-training-eval-episodes "$EVAL_EPISODES" \
		--env-seed "$ENV_SEED" --background-seed "$BACKGROUND_SEED" \
		--planner-seed-base "$PLANNER_SEED_BASE" --output "$out" >>"$log" 2>&1
	rc=$?
	set -e
	write_rc "$outdir/${arm}.rc" "$rc"
	echo "EVAL_END task=$task arm=$arm condition=burst_${length} role=$role gpu=$gpu rc=$rc" | tee -a "$log"
}

run_task() {
	local gpu=$1 task=$2 dir="$STAGE/tasks/$2" support="$SUPPORT_BASE/$2/annotations.json"
	local role0 role1 arm memory preflight rc length
	case "$task" in
		reacher-visual-small) role0=whole_arm; role1=goal ;;
		cartpole-swingup) role0=cart; role1=pole ;;
		*) return 2 ;;
	esac
	mkdir -p "$dir/preflight" "$dir/evaluations/normal"
	for length in 5 20 50; do
		mkdir -p "$dir/evaluations/burst_${length}"
	done
	printf '%s\n' "$gpu" >"$dir/gpu"
	for arm in "${ARMS[@]}"; do
		memory="$(arm_memory "$arm")"
		preflight="$dir/preflight/${arm}.runtime_config.json"
		write_preflight_config "$preflight" "$task" "$support" "$role0" "$role1" "$memory"
		echo "PREFLIGHT_START task=$task arm=$arm memory=$memory gpu=$gpu"
		set +e
		env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
			"$PY" -m tdmpc2.tools.check_cutie_episode_reset_isolation \
			--runtime-config "$preflight" --output "$dir/preflight/${arm}.reset.json" \
			--pollution-length 32 >"$dir/preflight/${arm}.log" 2>&1
		rc=$?
		set -e
		write_rc "$dir/preflight/${arm}.rc" "$rc"
		echo "PREFLIGHT_END task=$task arm=$arm memory=$memory gpu=$gpu rc=$rc"
		if (( rc != 0 )); then
			echo "Skipped: reset-isolation preflight failed rc=$rc" >"$dir/${arm}.train.log"
			write_rc "$dir/${arm}.train.rc" 125
			write_rc "$dir/evaluations/normal/${arm}.rc" 125
			for length in 5 20 50; do
				write_rc "$dir/evaluations/burst_${length}/${arm}.rc" 125
			done
			continue
		fi
		run_training "$gpu" "$task" "$arm" "$support" "$role0" "$role1" "$dir"
		if [[ "$(<"$dir/${arm}.train.rc")" == 0 ]]; then
			run_evaluation "$gpu" "$task" "$arm" "$dir"
			for length in 5 20 50; do
				run_burst_evaluation "$gpu" "$task" "$arm" "$length" "$dir"
			done
		else
			echo "Skipped: training failed." >"$dir/evaluations/normal/${arm}.log"
			write_rc "$dir/evaluations/normal/${arm}.rc" 125
			for length in 5 20 50; do
				echo "Skipped: training failed." >"$dir/evaluations/burst_${length}/${arm}.log"
				write_rc "$dir/evaluations/burst_${length}/${arm}.rc" 125
			done
		fi
	done
	echo "TASK_DONE task=$task gpu=$gpu"
}

echo "[3/6] Fixed two-GPU task queues"
echo "GPU $GPU_REACHER: reacher-visual-small, arms hard_zero then last_valid"
echo "GPU $GPU_CARTPOLE: cartpole-swingup, arms hard_zero then last_valid"
run_task "$GPU_REACHER" reacher-visual-small & PID_REACHER=$!
ACTIVE_PIDS+=("$PID_REACHER")
run_task "$GPU_CARTPOLE" cartpole-swingup & PID_CARTPOLE=$!
ACTIVE_PIDS+=("$PID_CARTPOLE")
set +e
wait "$PID_REACHER"; WORKER_REACHER_RC=$?
wait "$PID_CARTPOLE"; WORKER_CARTPOLE_RC=$?
set -e
ACTIVE_PIDS=()
write_rc "$STAGE/gpu_reacher_worker.rc" "$WORKER_REACHER_RC"
write_rc "$STAGE/gpu_cartpole_worker.rc" "$WORKER_CARTPOLE_RC"

echo "[4/6] Strict aggregation"
set +e
"$PY" - "$STAGE" "$SUMMARY" "$REPO_ROOT" "$START_SECONDS" \
	"$GPU_REACHER" "$GPU_CARTPOLE" "$SUPPORT_BASE" "$OC_REPO" "$CUTIE_CKPT" <<'PY'
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
started = int(sys.argv[4])
gpu_reacher, gpu_cartpole = sys.argv[5:7]
support_base, expected_oc_repo, expected_checkpoint = map(Path, sys.argv[7:10])
inputs_path = stage / 'provenance' / 'inputs.json'
normal_evaluator_path = repo / 'tdmpc2' / 'tools' / 'evaluate_cutie_multitask_checkpoint.py'
burst_evaluator_path = repo / 'tdmpc2' / 'tools' / 'evaluate_cutie_multitask_policy_burst.py'
tasks = ('reacher-visual-small', 'cartpole-swingup')
arms = ('hard_zero', 'last_valid')
roles_by_task = {
    'reacher-visual-small': ['whole_arm', 'goal'],
    'cartpole-swingup': ['cart', 'pole'],
}
gpu_by_task = {
    'reacher-visual-small': gpu_reacher,
    'cartpole-swingup': gpu_cartpole,
}
memory_by_arm = {'hard_zero': False, 'last_valid': True}
seed, steps, eval_freq, eval_episodes = 7, 100000, 20000, 3
heldout_episodes = 20
expected_eval_steps = list(range(0, steps + 1, eval_freq))
expected_train_frames = steps + steps // 500 + len(expected_eval_steps) * eval_episodes * 501
expected_eval_frames = heldout_episodes * 501
expected_alignment_draws = heldout_episodes * 500
pair_fields = (
    'initial_rgb_sha256', 'initial_object_sha256',
    'background_source', 'background_start_frame_index',
    'planner_seed', 'planner_rng_start_sha256', 'planner_rng_end_sha256', 'length',
)
burst_pair_fields = (
    'initial_rgb_sha256', 'initial_policy_object_sha256',
    'initial_raw_object_frame_sha256', 'background_source',
    'background_start_frame_index', 'background_end_source',
    'background_end_frame_index', 'planner_seed',
    'planner_rng_start_sha256', 'planner_rng_end_sha256', 'length',
    'policy_burst_event', 'policy_burst_plan_sha256',
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
    value = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

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

def experiment_name(task, arm):
    return f'cutie_object_{arm}100k_cutie_object_memory_probe_100k_v1_seed7_{task.replace("-", "_")}'

def run_root(task, arm):
    return repo / 'logs' / task / str(seed) / experiment_name(task, arm)

def runtime_checks(value, expected_frames):
    if not isinstance(value, dict):
        return {'present': False}
    return {
        'present': True,
        'frames': value.get('frames') == expected_frames,
        # Valid rate and invalid-burst length are scientific observations in
        # this probe, not health gates: Cartpole pole losses motivate the test.
        'valid_frame_rate_finite': math.isfinite(float(value.get('valid_frame_rate', float('nan')))),
        'worker_restarts': value.get('worker_restarts') == 0,
        'timeouts': value.get('timeouts') == 0,
        'ms_per_frame': positive(value.get('ms_per_frame')) and float(value['ms_per_frame']) <= 800,
        'runtime_unit': value.get('runtime_unit') == 'milliseconds_per_tracked_frame_excluding_support_prompts',
        'reset_strategy': value.get('episode_reset_strategy') == 'fresh_inference_core_support_replay_v1',
    }

def memory_runtime_checks(value, expected_enabled, roles):
    payload = value.get('last_valid_memory') if isinstance(value, dict) else None
    if not isinstance(payload, dict):
        return {'present': False}
    per_role = payload.get('per_role')
    checks = {
        'present': True,
        'enabled': payload.get('enabled') is expected_enabled,
        'content_dim': payload.get('content_dim') == 586,
        'status_dim': payload.get('status_dim') == 4,
        'substitutions_nonnegative': (
            isinstance(payload.get('substitutions'), int)
            and payload.get('substitutions') >= 0
        ),
        'invalid_without_history_nonnegative': (
            isinstance(payload.get('invalid_without_history'), int)
            and payload.get('invalid_without_history') >= 0
        ),
        'roles': isinstance(per_role, dict) and set(per_role) == set(roles),
    }
    if not expected_enabled:
        checks['disabled_is_noop'] = (
            payload.get('substitutions') == 0
            and payload.get('invalid_without_history') == 0
            and isinstance(per_role, dict)
            and all(
                item.get('has_memory') is False
                and item.get('age') is None
                and item.get('max_age') == 0
                and item.get('substitutions') == 0
                and item.get('invalid_without_history') == 0
                for item in per_role.values()
            )
        )
    return checks

def paired_statistics(hard_zero, last_valid):
    if len(hard_zero) != heldout_episodes or len(last_valid) != heldout_episodes:
        raise ValueError('Paired reward vectors must each contain 20 episodes.')
    deltas = [right - left for left, right in zip(hard_zero, last_valid)]
    if not all(math.isfinite(value) for value in deltas):
        raise ValueError('Paired reward deltas must be finite.')
    mean = statistics.fmean(deltas)
    sample_std = statistics.stdev(deltas)
    standard_error = sample_std / math.sqrt(len(deltas))
    t95_df19 = 2.093024054408263
    half_width = t95_df19 * standard_error
    wins = sum(value > 0 for value in deltas)
    ties = sum(value == 0 for value in deltas)
    return {
        'hard_zero_reward_mean': statistics.fmean(hard_zero),
        'last_valid_reward_mean': statistics.fmean(last_valid),
        'last_valid_minus_hard_zero_mean': mean,
        'last_valid_minus_hard_zero_median': statistics.median(deltas),
        'last_valid_minus_hard_zero_sample_std': sample_std,
        'last_valid_minus_hard_zero_standard_error': standard_error,
        'last_valid_minus_hard_zero_t95_interval_df19': [
            mean - half_width, mean + half_width,
        ],
        'paired_deltas': deltas,
        'win_tie_loss': [wins, ties, len(deltas) - wins - ties],
        'episodes': len(deltas),
        'interval_scope': (
            'conditional paired-episode t interval for one training seed; '
            'not uncertainty across training seeds'
        ),
    }

inputs = load_json(inputs_path)
jobs, structure, runtime_failures, pairing_failures = [], [], [], []
reports, eval_payloads = {}, {}
normal_evaluator_hashes, burst_evaluator_hashes = set(), set()
manifest_hashes, combined_manifest_hashes, device_names = set(), set(), set()

try:
    immutable_checks = {
        'config': digest(repo / 'tdmpc2' / 'config.yaml')
        == inputs.get('config', {}).get('sha256'),
        'cutie_checkpoint': digest(expected_checkpoint)
        == inputs.get('cutie_checkpoint', {}).get('sha256'),
        'support_paths': set(inputs.get('supports', {})) == set(tasks),
    }
    for task in tasks:
        support_path = support_base / task / 'annotations.json'
        recorded = inputs.get('supports', {}).get(task, {})
        immutable_checks[f'{task}_support_path'] = same_path(
            recorded.get('path'), support_path
        )
        immutable_checks[f'{task}_support_sha256'] = (
            digest(support_path) == recorded.get('sha256')
        )
    for relative, recorded in inputs.get('implementation', {}).items():
        path = repo / relative
        immutable_checks[f'implementation:{relative}'] = (
            same_path(recorded.get('path'), path)
            and digest(path) == recorded.get('sha256')
        )
    for task in tasks:
        for length in (5, 20, 50):
            recorded = inputs.get('plans', {}).get(task, {}).get(str(length), {})
            path = stage / 'plans' / task / f'burst_{length}.json'
            raw = path.read_bytes()
            payload = json.loads(raw.decode('utf-8'))
            canonical = (
                json.dumps(
                    payload, sort_keys=True, separators=(',', ':'),
                    ensure_ascii=True, allow_nan=False,
                ) + '\n'
            ).encode('utf-8')
            immutable_checks[f'{task}_burst_{length}_path'] = same_path(
                recorded.get('path'), path
            )
            immutable_checks[f'{task}_burst_{length}_sha256'] = (
                raw == canonical and digest(path) == recorded.get('sha256')
            )
    if not all(immutable_checks.values()):
        structure.append(
            'immutable input changed during run: '
            f'{[key for key, value in immutable_checks.items() if not value]}'
        )
except Exception as exc:
    immutable_checks = {'parse': False}
    structure.append(f'immutable input recheck failed: {exc}')

for task in tasks:
    directory = stage / 'tasks' / task
    support_path = support_base / task / 'annotations.json'
    roles = roles_by_task[task]
    assigned_gpu = gpu_by_task[task]
    task_report = {
        'assigned_physical_gpu': assigned_gpu,
        'roles': roles,
        'support': inputs.get('supports', {}).get(task),
        'arms': {},
        'pairing': {},
        'burst_conditions': {},
    }
    payloads = {}
    for arm in arms:
        expected_memory = memory_by_arm[arm]
        root = run_root(task, arm)
        preflight_rc = read_rc(directory / 'preflight' / f'{arm}.rc')
        train_rc = read_rc(directory / f'{arm}.train.rc')
        eval_dir = directory / 'evaluations' / 'normal'
        eval_rc = read_rc(eval_dir / f'{arm}.rc')
        arm_report = {
            'memory_enabled': expected_memory,
            'root': str(root),
            'preflight_rc': preflight_rc,
            'training_rc': train_rc,
            'evaluation_rc': eval_rc,
        }
        task_report['arms'][arm] = arm_report
        if preflight_rc != 0 or train_rc != 0 or eval_rc != 0:
            jobs.append(
                f'{task}/{arm}: preflight={preflight_rc}, train={train_rc}, eval={eval_rc}'
            )
            continue

        reset_path = directory / 'preflight' / f'{arm}.reset.json'
        required = {
            'config': root / 'runtime_config.json',
            'curve': root / 'eval.csv',
            'checkpoint': root / 'models' / 'final.pt',
            'trainer': root / 'trainer_runtime.json',
            'replay': root / 'replay_runtime.json',
            'perception': root / 'perception_runtime.json',
            'reset': reset_path,
            'evaluation': eval_dir / f'{arm}.json',
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            structure.append(f'{task}/{arm}: missing artifacts {missing}')
            arm_report['missing'] = missing
            continue

        try:
            cfg = load_json(required['config'])
            trainer = load_json(required['trainer'])
            replay = load_json(required['replay'])
            perception = load_json(required['perception'])
            reset = load_json(required['reset'])
            with required['curve'].open(encoding='utf-8', newline='') as file:
                rows = list(csv.DictReader(file))
            curve_steps = [int(float(row['step'])) for row in rows]
            curve_rewards = [float(row['episode_reward']) for row in rows]
            config_checks = {
                'task_seed': cfg.get('task') == task and cfg.get('seed') == seed,
                'schedule': (
                    cfg.get('steps') == steps and cfg.get('eval_freq') == eval_freq
                    and cfg.get('eval_episodes') == eval_episodes
                ),
                'dynamic_video_train': (
                    cfg.get('video_background_enabled') is True
                    and cfg.get('video_background_split') == 'train'
                    and cfg.get('video_background_seed') == seed
                    and float(cfg.get('video_background_strength', -1)) == 1.0
                ),
                'normal_geometry': cfg.get('visual_foreground_erosion_pixels') == 0,
                'model': cfg.get('model_size') == 5,
                'object_only_mode': (
                    cfg.get('flat_anchor') is True
                    and cfg.get('flat_anchor_mode') == 'cutie_object_only'
                    and cfg.get('latent_dim') == 128
                    and cfg.get('obs_shape') == {'object': [2, 1770]}
                ),
                'memory_switch': cfg.get('cutie_object_last_valid_memory') is expected_memory,
                'training_burst_disabled': (
                    'cutie_object_policy_burst_plan' in cfg
                    and cfg['cutie_object_policy_burst_plan'] is None
                ),
                'diagnostic_oracle_support': (
                    cfg.get('cutie_object_support_schema') == 'generic_indexed_v1'
                    and cfg.get('cutie_object_role_names') == roles
                    and cfg.get('cutie_object_allow_simulator_support') is True
                    and same_path(cfg.get('cutie_object_support_path'), support_path)
                ),
                'cutie_inputs': (
                    same_path(cfg.get('cutie_object_repo'), expected_oc_repo)
                    and same_path(cfg.get('cutie_object_checkpoint'), expected_checkpoint)
                ),
                'replay_object_only': replay.get('observation_keys') == ['object'],
                'replay_cuda': replay.get('storage_device') == 'cuda:0',
                'throughput': positive(trainer.get('training_non_eval_steps_per_second')),
                'curve': (
                    curve_steps == expected_eval_steps
                    and all(math.isfinite(value) for value in curve_rewards)
                ),
            }
            reset_comparisons = reset.get('comparisons', {})
            reset_checks = {
                'pass': reset.get('pass') is True,
                'task': reset.get('task') == task,
                'roles': reset.get('role_names') == roles,
                'comparisons': bool(reset_comparisons) and all(
                    item.get('byte_equal') is True for item in reset_comparisons.values()
                ),
            }
            train_runtime_checks = runtime_checks(perception, expected_train_frames)
            train_memory_checks = memory_runtime_checks(
                perception, expected_memory, roles
            )
            if not all(config_checks.values()):
                structure.append(
                    f'{task}/{arm}: config/artifact checks '
                    f'{[key for key, value in config_checks.items() if not value]}'
                )
            if not all(reset_checks.values()):
                structure.append(
                    f'{task}/{arm}: reset checks '
                    f'{[key for key, value in reset_checks.items() if not value]}'
                )
            if not all(train_runtime_checks.values()):
                runtime_failures.append(f'{task}/{arm}: training {train_runtime_checks}')
            if not all(train_memory_checks.values()):
                runtime_failures.append(
                    f'{task}/{arm}: training memory {train_memory_checks}'
                )

            checkpoint_payload = torch.load(
                required['checkpoint'], map_location='cpu', weights_only=False
            )
            state = (
                checkpoint_payload.get('model', checkpoint_payload)
                if isinstance(checkpoint_payload, dict) else checkpoint_payload
            )
            if not isinstance(state, dict):
                raise ValueError('checkpoint state is not a mapping')
            keys = set(state)
            checkpoint_checks = {
                'finite': all(
                    not torch.is_tensor(value) or bool(torch.isfinite(value).all())
                    for value in state.values()
                ),
                'object_encoder': any(key.startswith('_encoder.object.') for key in keys),
                'no_rgb_encoder': not any(key.startswith('_encoder.rgb.') for key in keys),
                'no_hybrid_branch': not any(key.startswith('_hybrid_') for key in keys),
            }
            if not all(checkpoint_checks.values()):
                structure.append(f'{task}/{arm}: checkpoint structure {checkpoint_checks}')

            evaluation = load_json(required['evaluation'])
            episodes = evaluation.get('episodes')
            protocol = evaluation.get('evaluation', {})
            provenance = evaluation.get('provenance', {})
            cutie_inputs = provenance.get('cutie_inputs', {})
            cutie_ready = provenance.get('cutie_ready', {})
            eval_checks = {
                'format': evaluation.get('format') == 'cutie_multitask_checkpoint_evaluation_v1',
                'task_backend': (
                    evaluation.get('task') == task
                    and evaluation.get('backend') == 'cutie_object_only'
                    and evaluation.get('training_seed') == seed
                ),
                'normal_condition': (
                    evaluation.get('erosion_pixels') == 0
                    and protocol.get('split') == 'validation'
                    and protocol.get('episodes') == heldout_episodes
                    and protocol.get('env_seed') == 424243
                    and protocol.get('background_seed') == 1618034
                    and protocol.get('planner_seed_base') == 8675400
                    and protocol.get('actual_foreground_erosion_pixels') == 0
                ),
                'rng_alignment': (
                    protocol.get('rgb_shift_rng_alignment') == 'object_only_equivalent_cuda_randint_v1'
                    and protocol.get('object_only_alignment_draws') == expected_alignment_draws
                    and protocol.get('expected_rgb_shift_draws_per_backend') == expected_alignment_draws
                ),
                'episodes': (
                    isinstance(episodes, list) and len(episodes) == heldout_episodes
                    and [row.get('episode_index') for row in episodes]
                    == list(range(heldout_episodes))
                ),
                'source_provenance': (
                    same_path(provenance.get('runtime_config'), required['config'])
                    and provenance.get('runtime_config_sha256') == digest(required['config'])
                    and same_path(provenance.get('checkpoint'), required['checkpoint'])
                    and provenance.get('checkpoint_sha256') == digest(required['checkpoint'])
                    and provenance.get('evaluator_sha256') == digest(normal_evaluator_path)
                    and provenance.get('cuda_visible_devices') == assigned_gpu
                ),
                'cutie_provenance': (
                    same_path(cutie_inputs.get('checkpoint'), expected_checkpoint)
                    and cutie_inputs.get('checkpoint_sha256') == digest(expected_checkpoint)
                    and same_path(cutie_inputs.get('support'), support_path)
                    and cutie_inputs.get('support_sha256') == digest(support_path)
                    and cutie_inputs.get('roles') == roles
                    and cutie_inputs.get('support_schema') == 'generic_indexed_v1'
                    and cutie_ready.get('allow_simulator_support') is True
                ),
            }
            if not all(eval_checks.values()):
                structure.append(
                    f'{task}/{arm}: evaluation checks '
                    f'{[key for key, value in eval_checks.items() if not value]}'
                )
            eval_runtime = evaluation.get('perception_runtime')
            normal_intervention = (
                eval_runtime.get('policy_observation_intervention', {})
                if isinstance(eval_runtime, dict) else {}
            )
            normal_intervention_checks = {
                'disabled': normal_intervention.get('enabled') is False,
                'plan_path': normal_intervention.get('plan_path') is None,
                'plan_sha256': normal_intervention.get('plan_sha256') is None,
                'scheduled_events': normal_intervention.get('scheduled_events') == 0,
                'applied_events': normal_intervention.get('applied_events') == 0,
                'scheduled_role_frames': normal_intervention.get('scheduled_role_frames') == 0,
                'applied_role_frames': normal_intervention.get('applied_role_frames') == 0,
                'raw_health_excludes_synthetic': normal_intervention.get(
                    'raw_tracker_accounting_excludes_synthetic_intervention'
                ) is True,
                'live_observation_unmutated': normal_intervention.get(
                    'live_environment_observation_mutated'
                ) is False,
            }
            if not all(normal_intervention_checks.values()):
                runtime_failures.append(
                    f'{task}/{arm}: normal intervention {normal_intervention_checks}'
                )
            eval_runtime_checks = runtime_checks(eval_runtime, expected_eval_frames)
            eval_memory_checks = memory_runtime_checks(
                eval_runtime, expected_memory, roles
            )
            if not all(eval_runtime_checks.values()):
                runtime_failures.append(f'{task}/{arm}: evaluation {eval_runtime_checks}')
            if not all(eval_memory_checks.values()):
                runtime_failures.append(
                    f'{task}/{arm}: evaluation memory {eval_memory_checks}'
                )
            values = [float(row['reward']) for row in episodes] if isinstance(episodes, list) else []
            if len(values) != heldout_episodes or not all(math.isfinite(value) for value in values):
                raise ValueError('evaluation rewards are incomplete or non-finite')

            arm_report.update({
                'config_checks': config_checks,
                'reset_checks': reset_checks,
                'checkpoint_checks': checkpoint_checks,
                'training_eval_steps': curve_steps,
                'training_eval_rewards': curve_rewards,
                'trainer_runtime': trainer,
                'replay_runtime': replay,
                'training_perception_runtime': perception,
                'training_runtime_checks': train_runtime_checks,
                'training_memory_checks': train_memory_checks,
                'checkpoint': str(required['checkpoint']),
                'checkpoint_sha256': digest(required['checkpoint']),
                'evaluation_relative_to_summary_root': (
                    f'tasks/{task}/evaluations/normal/{arm}.json'
                ),
                'evaluation_checks': eval_checks,
                'evaluation_perception_runtime': eval_runtime,
                'evaluation_runtime_checks': eval_runtime_checks,
                'evaluation_memory_checks': eval_memory_checks,
                'normal_intervention_checks': normal_intervention_checks,
                'reward_mean': statistics.fmean(values),
                'reward_median': statistics.median(values),
                'reward_sample_std': statistics.stdev(values),
                'rewards': values,
                'evaluation_elapsed_seconds': evaluation.get('summary', {}).get('elapsed_seconds'),
            })
            payloads[arm] = evaluation
            normal_evaluator_hashes.add(provenance.get('evaluator_sha256'))
            manifest_hashes.add(provenance.get('validation_manifest_sha256'))
            combined_manifest_hashes.add(provenance.get('combined_manifest_sha256'))
            device_names.add(provenance.get('device_name'))
        except Exception as exc:
            structure.append(f'{task}/{arm}: artifact parse failed: {exc}')

    if set(payloads) != set(arms):
        pairing_failures.append(f'{task}: incomplete evaluation arms {sorted(payloads)}')
        task_report['pairing']['normal'] = {
            'exact': False, 'failure': f'incomplete arms {sorted(payloads)}',
        }
    else:
        mismatches = {
            field: [
                index
                for index, (left, right) in enumerate(zip(
                    payloads['hard_zero']['episodes'], payloads['last_valid']['episodes']
                ))
                if left.get(field) != right.get(field)
            ]
            for field in pair_fields
        }
        task_report['pairing']['normal'] = {
            'exact': not any(mismatches.values()),
            'fields': list(pair_fields),
            'mismatch_episode_indices': mismatches,
        }
        if any(mismatches.values()):
            pairing_failures.append(f'{task}: {mismatches}')

    condition_payloads = {'normal': payloads}
    for burst_length in (5, 20, 50):
        condition_name = f'burst_{burst_length}'
        eval_dir = directory / 'evaluations' / condition_name
        plan_path = stage / 'plans' / task / f'burst_{burst_length}.json'
        plan_meta = inputs.get('plans', {}).get(task, {}).get(str(burst_length), {})
        expected_plan_sha = plan_meta.get('sha256')
        expected_events = plan_meta.get('events')
        expected_target = plan_meta.get('target_role')
        expected_burst_frames = heldout_episodes * burst_length
        expected_role_frames = {
            role: expected_burst_frames if role == expected_target else 0
            for role in roles
        }
        condition_report = {
            'plan_relative_to_summary_root': f'plans/{task}/burst_{burst_length}.json',
            'plan_sha256': expected_plan_sha,
            'target_role': expected_target,
            'length': burst_length,
            'arms': {},
        }
        task_report['burst_conditions'][condition_name] = condition_report
        burst_payloads = {}
        for arm in arms:
            expected_memory = memory_by_arm[arm]
            root = run_root(task, arm)
            runtime_config = root / 'runtime_config.json'
            checkpoint = root / 'models' / 'final.pt'
            rc = read_rc(eval_dir / f'{arm}.rc')
            output = eval_dir / f'{arm}.json'
            burst_report = {
                'rc': rc,
                'memory_enabled': expected_memory,
                'output_relative_to_summary_root': (
                    f'tasks/{task}/evaluations/{condition_name}/{arm}.json'
                ),
            }
            condition_report['arms'][arm] = burst_report
            if rc != 0:
                jobs.append(f'{task}/{condition_name}/{arm}: evaluation rc={rc}')
                continue
            if not output.is_file():
                structure.append(f'{task}/{condition_name}/{arm}: missing evaluation JSON')
                continue
            try:
                evaluation = load_json(output)
                episodes = evaluation.get('episodes')
                policy_burst = evaluation.get('policy_burst', {})
                protocol = evaluation.get('evaluation', {})
                provenance = evaluation.get('provenance', {})
                source_flags = provenance.get('source_flags')
                evaluation_flags = provenance.get('evaluation_flags')
                cutie_inputs = provenance.get('cutie_inputs', {})
                cutie_ready = provenance.get('cutie_ready', {})
                perception = evaluation.get('perception_runtime')
                intervention = (
                    perception.get('policy_observation_intervention', {})
                    if isinstance(perception, dict) else {}
                )
                memory = (
                    perception.get('last_valid_memory', {})
                    if isinstance(perception, dict) else {}
                )
                strict_checks = evaluation.get('strict_checks')
                burst_checks = {
                    'format': evaluation.get('format')
                    == 'cutie_multitask_policy_burst_evaluation_v1',
                    'scope_is_diagnostic_not_learned_belief': (
                        isinstance(evaluation.get('scientific_scope'), str)
                        and 'not a learned belief' in evaluation['scientific_scope']
                    ),
                    'task_arm_backend_seed': (
                        evaluation.get('task') == task
                        and evaluation.get('arm') == arm
                        and evaluation.get('backend') == 'cutie_object_only'
                        and evaluation.get('training_seed') == seed
                    ),
                    'policy_burst_envelope': (
                        policy_burst.get('format') == 'cutie_policy_burst_plan_v1'
                        and policy_burst.get('role') == expected_target
                        and policy_burst.get('length') == burst_length
                        and policy_burst.get('starts') == plan_meta.get('starts')
                        and policy_burst.get('episodes') == heldout_episodes
                        and policy_burst.get('plan_sha256_before') == expected_plan_sha
                        and policy_burst.get('plan_sha256_after') == expected_plan_sha
                        and policy_burst.get('wrapper_plan_sha256') == expected_plan_sha
                        and policy_burst.get('minimum_raw_valid_overwrite_rate') == 0.8
                        and isinstance(policy_burst.get('raw_valid_overwrite_rate'), float)
                        and 0.0 <= policy_burst.get('raw_valid_overwrite_rate') <= 1.0
                        and policy_burst.get('controlled_burst_attribution_eligible')
                        is (
                            policy_burst.get('raw_valid_overwrite_rate') >= 0.8
                        )
                    ),
                    'evaluation_protocol': (
                        protocol.get('split') == 'validation'
                        and protocol.get('episodes') == heldout_episodes
                        and protocol.get('env_seed') == 424243
                        and protocol.get('background_seed') == 1618034
                        and protocol.get('planner_seed_base') == 8675400
                        and protocol.get('eval_mode') is True
                        and protocol.get('foreground_erosion_pixels') == 0
                    ),
                    'rng_alignment_10000': (
                        protocol.get('rgb_shift_rng_alignment')
                        == 'object_only_equivalent_cuda_randint_v1'
                        and protocol.get('object_only_alignment_draws') == 10000
                        and protocol.get('expected_object_only_alignment_draws') == 10000
                    ),
                    'source_paths_and_hashes': (
                        same_path(provenance.get('runtime_config'), runtime_config)
                        and provenance.get('runtime_config_sha256') == digest(runtime_config)
                        and same_path(provenance.get('checkpoint'), checkpoint)
                        and provenance.get('checkpoint_sha256') == digest(checkpoint)
                        and provenance.get('runtime_config_sha256_after')
                        == digest(runtime_config)
                        and provenance.get('checkpoint_sha256_after')
                        == digest(checkpoint)
                        and same_path(provenance.get('evaluator'), burst_evaluator_path)
                        and provenance.get('evaluator_sha256') == digest(burst_evaluator_path)
                        and same_path(
                            provenance.get('base_evaluator'), normal_evaluator_path
                        )
                        and provenance.get('base_evaluator_sha256')
                        == digest(normal_evaluator_path)
                    ),
                    'source_flags_plan_null': source_flags == {
                        'cutie_object_last_valid_memory': expected_memory,
                        'cutie_object_policy_burst_plan': None,
                    },
                    'evaluation_flags_exact_plan': (
                        isinstance(evaluation_flags, dict)
                        and evaluation_flags.get('cutie_object_last_valid_memory')
                        is expected_memory
                        and same_path(
                            evaluation_flags.get('cutie_object_policy_burst_plan'),
                            plan_path,
                        )
                    ),
                    'plan_provenance': (
                        same_path(provenance.get('policy_burst_plan'), plan_path)
                        and provenance.get('policy_burst_plan_sha256') == expected_plan_sha
                        and (
                            output.parent
                            / provenance.get('policy_burst_plan_relative_to_output', '')
                        ).resolve() == plan_path.resolve()
                    ),
                    'gpu_provenance': (
                        provenance.get('cuda_visible_devices') == assigned_gpu
                        and isinstance(provenance.get('device_name'), str)
                        and bool(provenance.get('device_name'))
                    ),
                    'cutie_provenance': (
                        same_path(cutie_inputs.get('checkpoint'), expected_checkpoint)
                        and cutie_inputs.get('checkpoint_sha256') == digest(expected_checkpoint)
                        and same_path(cutie_inputs.get('support'), support_path)
                        and cutie_inputs.get('support_sha256') == digest(support_path)
                        and cutie_inputs.get('roles') == roles
                        and cutie_inputs.get('support_schema') == 'generic_indexed_v1'
                        and cutie_ready.get('allow_simulator_support') is True
                    ),
                    'episodes_20': (
                        isinstance(episodes, list)
                        and len(episodes) == heldout_episodes
                        and [row.get('episode_index') for row in episodes]
                        == list(range(heldout_episodes))
                    ),
                    'strict_checks_all_true': (
                        isinstance(strict_checks, dict) and bool(strict_checks)
                        and all(value is True for value in strict_checks.values())
                    ),
                    'strict_stack_checks_10020': (
                        isinstance(strict_checks, dict)
                        and strict_checks.get('policy_stack_transition_checks_10020')
                        is True
                    ),
                    'strict_rng_draws_10000': (
                        isinstance(strict_checks, dict)
                        and strict_checks.get('rgb_shift_rng_alignment_draws_10000')
                        is True
                    ),
                    'intervention_enabled': intervention.get('enabled') is True,
                    'intervention_location': intervention.get('location') == (
                        'raw_tracker_frame_after_metrics_before_last_valid_memory_and_stack'
                    ),
                    'intervention_plan': (
                        intervention.get('format') == 'cutie_policy_burst_plan_v1'
                        and intervention.get('invalid_encoding') == 'empty_lost_v1'
                        and intervention.get('decision_unit')
                        == 'agent_decision_observation_index_0_to_499'
                        and same_path(intervention.get('plan_path'), plan_path)
                        and intervention.get('plan_sha256') == expected_plan_sha
                        and intervention.get('plan_task') == task
                        and intervention.get('plan_roles') == roles
                    ),
                    'intervention_events_20': (
                        intervention.get('scheduled_events') == heldout_episodes
                        and intervention.get('applied_events') == heldout_episodes
                    ),
                    'intervention_role_frames': (
                        intervention.get('scheduled_role_frames') == expected_burst_frames
                        and intervention.get('applied_role_frames') == expected_burst_frames
                        and intervention.get('per_role_applied_frames')
                        == expected_role_frames
                    ),
                    'intervention_raw_accounting': (
                        intervention.get('raw_valid_overwritten', -1)
                        + intervention.get('raw_invalid_overlap', -1)
                        == expected_burst_frames
                    ),
                    'intervention_exact_checks': (
                        intervention.get('exact_invalid_checks') == expected_burst_frames
                        and intervention.get('non_target_preserved_checks')
                        == expected_burst_frames
                        and intervention.get('raw_source_unchanged_checks')
                        == expected_burst_frames
                        and intervention.get('policy_stack_transition_checks') == 10020
                    ),
                    'synthetic_excluded_from_raw_tracker_health': intervention.get(
                        'raw_tracker_accounting_excludes_synthetic_intervention'
                    ) is True,
                    'live_environment_observation_unmutated': intervention.get(
                        'live_environment_observation_mutated'
                    ) is False,
                    'memory_arm': memory.get('enabled') is expected_memory,
                }
                if arm == 'hard_zero':
                    burst_checks['hard_zero_content_contract'] = (
                        intervention.get('hard_zero_content_checks')
                        == expected_burst_frames
                        and intervention.get('last_valid_content_checks') == 0
                        and intervention.get('memory_substitutions') == 0
                        and intervention.get('without_memory_history') == 0
                    )
                else:
                    burst_checks['last_valid_content_contract'] = (
                        intervention.get('hard_zero_content_checks') == 0
                        and intervention.get('last_valid_content_checks')
                        == expected_burst_frames
                        and intervention.get('memory_substitutions')
                        == expected_burst_frames
                        and intervention.get('without_memory_history') == 0
                    )
                if isinstance(episodes, list):
                    burst_checks['episode_events_and_plan_sha'] = all(
                        row.get('policy_burst_event') == expected_events[index]
                        and row.get('policy_burst_plan_sha256') == expected_plan_sha
                        for index, row in enumerate(episodes)
                    )
                if not all(burst_checks.values()):
                    structure.append(
                        f'{task}/{condition_name}/{arm}: burst checks '
                        f'{[key for key, value in burst_checks.items() if not value]}'
                    )
                burst_runtime_checks = runtime_checks(perception, expected_eval_frames)
                burst_memory_checks = memory_runtime_checks(
                    perception, expected_memory, roles
                )
                if not all(burst_runtime_checks.values()):
                    runtime_failures.append(
                        f'{task}/{condition_name}/{arm}: raw runtime '
                        f'{burst_runtime_checks}'
                    )
                if not all(burst_memory_checks.values()):
                    runtime_failures.append(
                        f'{task}/{condition_name}/{arm}: memory runtime '
                        f'{burst_memory_checks}'
                    )
                values = (
                    [float(row['reward']) for row in episodes]
                    if isinstance(episodes, list) else []
                )
                if len(values) != heldout_episodes or not all(
                    math.isfinite(value) for value in values
                ):
                    raise ValueError('burst rewards are incomplete or non-finite')
                burst_report.update({
                    'checks': burst_checks,
                    'strict_checks': strict_checks,
                    'raw_tracker_runtime': perception,
                    'raw_runtime_checks': burst_runtime_checks,
                    'memory_runtime_checks': burst_memory_checks,
                    'policy_observation_intervention': intervention,
                    'raw_valid_overwrite_rate': policy_burst.get(
                        'raw_valid_overwrite_rate'
                    ),
                    'controlled_burst_attribution_eligible': policy_burst.get(
                        'controlled_burst_attribution_eligible'
                    ),
                    'reward_mean': statistics.fmean(values),
                    'reward_median': statistics.median(values),
                    'reward_sample_std': statistics.stdev(values),
                    'rewards': values,
                    'elapsed_seconds': evaluation.get('summary', {}).get(
                        'elapsed_seconds'
                    ),
                    'sha256': digest(output),
                })
                burst_payloads[arm] = evaluation
                burst_evaluator_hashes.add(provenance.get('evaluator_sha256'))
                manifest_hashes.add(provenance.get('validation_manifest_sha256'))
                combined_manifest_hashes.add(
                    provenance.get('combined_manifest_sha256')
                )
                device_names.add(provenance.get('device_name'))
            except Exception as exc:
                structure.append(
                    f'{task}/{condition_name}/{arm}: burst artifact parse failed: {exc}'
                )
        if set(burst_payloads) != set(arms):
            pairing_failures.append(
                f'{task}/{condition_name}: incomplete arms {sorted(burst_payloads)}'
            )
            task_report['pairing'][condition_name] = {
                'exact': False,
                'failure': f'incomplete arms {sorted(burst_payloads)}',
            }
        else:
            mismatches = {
                field: [
                    index
                    for index, (left, right) in enumerate(zip(
                        burst_payloads['hard_zero']['episodes'],
                        burst_payloads['last_valid']['episodes'],
                    ))
                    if left.get(field) != right.get(field)
                ]
                for field in burst_pair_fields
            }
            task_report['pairing'][condition_name] = {
                'exact': not any(mismatches.values()),
                'fields': list(burst_pair_fields),
                'mismatch_episode_indices': mismatches,
            }
            if any(mismatches.values()):
                pairing_failures.append(
                    f'{task}/{condition_name}: {mismatches}'
                )
        condition_payloads[condition_name] = burst_payloads
    reports[task] = task_report
    eval_payloads[task] = condition_payloads

for label, values in (
    ('normal_evaluator_sha256', normal_evaluator_hashes),
    ('burst_evaluator_sha256', burst_evaluator_hashes),
    ('validation_manifest_sha256', manifest_hashes),
    ('combined_manifest_sha256', combined_manifest_hashes),
):
    if len(values) != 1 or None in values:
        structure.append(f'cross-evaluation {label} mismatch: {sorted(map(str, values))}')
if (
    len(device_names) != 1 or None in device_names
    or not all('RTX 4090' in str(value) for value in device_names)
):
    structure.append(f'GPU model names are not matched: {sorted(map(str, device_names))}')

worker_rcs = {
    'reacher': read_rc(stage / 'gpu_reacher_worker.rc'),
    'cartpole': read_rc(stage / 'gpu_cartpole_worker.rc'),
}
if any(value != 0 for value in worker_rcs.values()):
    jobs.append(f'worker return codes {worker_rcs}')
gpu_contract_rcs = {
    'reacher': read_rc(stage / 'contracts' / 'gpu_reacher.rc'),
    'cartpole': read_rc(stage / 'contracts' / 'gpu_cartpole.rc'),
}
if any(value != 0 for value in gpu_contract_rcs.values()):
    jobs.append(f'GPU compile preflight return codes {gpu_contract_rcs}')

normal_outcomes = {}
burst_outcomes = {}
normal_no_regression = []
long_burst_improvement = []
for task in tasks:
    hard = reports[task]['arms']['hard_zero'].get('rewards')
    memory = reports[task]['arms']['last_valid'].get('rewards')
    if not isinstance(hard, list) or not isinstance(memory, list):
        normal_outcomes[task] = {'complete': False}
        burst_outcomes[task] = {'complete': False, 'conditions': {}}
        continue
    normal_stats = paired_statistics(hard, memory)
    hard_mean = normal_stats['hard_zero_reward_mean']
    memory_mean = normal_stats['last_valid_reward_mean']
    no_regression = memory_mean >= 0.95 * hard_mean if hard_mean > 0 else memory_mean >= hard_mean
    normal_no_regression.append(no_regression)
    normal_outcomes[task] = {
        'complete': True,
        **normal_stats,
        'last_valid_over_hard_zero_mean_retention': (
            memory_mean / hard_mean if hard_mean > 0 else None
        ),
        'last_valid_at_least_95pct_hard_zero': no_regression,
        'memory_activity': {
            arm: {
                'training': reports[task]['arms'][arm]
                .get('training_perception_runtime', {})
                .get('last_valid_memory'),
                'normal_heldout': reports[task]['arms'][arm]
                .get('evaluation_perception_runtime', {})
                .get('last_valid_memory'),
            }
            for arm in arms
        },
    }
    condition_outcomes = {}
    complete = True
    for burst_length in (5, 20, 50):
        condition = reports[task]['burst_conditions'][f'burst_{burst_length}']
        hard_burst = condition['arms']['hard_zero'].get('rewards')
        memory_burst = condition['arms']['last_valid'].get('rewards')
        if not isinstance(hard_burst, list) or not isinstance(memory_burst, list):
            condition_outcomes[f'burst_{burst_length}'] = {'complete': False}
            complete = False
            continue
        stats = paired_statistics(hard_burst, memory_burst)
        attribution_eligible = all(
            condition['arms'][arm].get('controlled_burst_attribution_eligible') is True
            for arm in arms
        )
        condition_outcomes[f'burst_{burst_length}'] = {
            'complete': True,
            'target_role': reports[task]['burst_conditions'][
                f'burst_{burst_length}'
            ]['target_role'],
            'length': burst_length,
            **stats,
            'controlled_burst_attribution_eligible': attribution_eligible,
            'minimum_raw_valid_overwrite_rate': 0.8,
            'raw_valid_overwrite_rate_by_arm': {
                arm: condition['arms'][arm].get('raw_valid_overwrite_rate')
                for arm in arms
            },
            'last_valid_improves_mean_reward': (
                stats['last_valid_minus_hard_zero_mean'] > 0
            ),
        }
    task_long_go = bool(
        complete
        and condition_outcomes['burst_20']['controlled_burst_attribution_eligible']
        and condition_outcomes['burst_50']['controlled_burst_attribution_eligible']
        and condition_outcomes['burst_20']['last_valid_improves_mean_reward']
        and condition_outcomes['burst_50']['last_valid_improves_mean_reward']
    )
    long_burst_improvement.append(task_long_go)
    burst_outcomes[task] = {
        'complete': complete,
        'conditions': condition_outcomes,
        'development_long_burst_improvement_20_and_50': task_long_go,
    }

engineering = {
    'both_gpu_compile_preflights': all(value == 0 for value in gpu_contract_rcs.values()),
    'all_jobs_completed': not jobs,
    'artifact_and_protocol_structure': not structure,
    'runtime_health': not runtime_failures,
    'strict_normal_and_burst_pairing': not pairing_failures,
    'diagnostic_oracle_support_provenance': all(
        all(item.get('checks', {}).values())
        for item in inputs.get('supports', {}).values()
    ),
}
engineering_pass = all(engineering.values())
normal_gate = len(normal_no_regression) == len(tasks) and all(normal_no_regression)
burst_gate = len(long_burst_improvement) == len(tasks) and all(long_burst_improvement)
development_go = bool(engineering_pass and normal_gate and burst_gate)
summary = {
    'format': 'cutie_object_memory_probe_v1',
    'status': (
        'object_memory_probe_engineering_pass'
        if engineering_pass else 'object_memory_probe_engineering_fail'
    ),
    'scientific_scope': (
        'single-training-seed diagnostic with privileged simulator-derived support; '
        'normal held-out plus controlled policy-input missing-role bursts; synthetic '
        'interventions are excluded from raw tracker health; fixed last-valid memory '
        'is not a learned belief and this is not sufficient for an algorithm, visual '
        'occlusion robustness, or paper claim'
    ),
    'protocol': {
        'tasks': list(tasks),
        'arms': {
            'hard_zero': {'cutie_object_last_valid_memory': False},
            'last_valid': {'cutie_object_last_valid_memory': True},
        },
        'training_seed': seed,
        'support_seed': 314159,
        'steps': steps,
        'eval_freq': eval_freq,
        'training_eval_episodes': eval_episodes,
        'heldout_episodes': heldout_episodes,
        'training_background': {'split': 'train', 'strength': 1.0, 'seed': seed},
        'heldout': {
            'split': 'validation', 'erosion_pixels': 0,
            'env_seed': 424243, 'background_seed': 1618034,
            'planner_seed_base': 8675400,
        },
        'gpu_queues': {
            'reacher': {'physical_index': gpu_reacher, 'task': 'reacher-visual-small'},
            'cartpole': {'physical_index': gpu_cartpole, 'task': 'cartpole-swingup'},
        },
        'within_task_serial_order': list(arms),
        'arm_semantics': {
            'hard_zero': (
                'no last-valid memory; synthetic scheduled frames use exact '
                'empty_lost_v1 content/status (the normal condition is simply no-memory)'
            ),
            'last_valid': (
                'episode-local causal last-valid 586-D content with the current '
                '4-D confidence/lost/valid/mask-score status preserved'
            ),
        },
        'strict_pairing_fields': {
            'normal': list(pair_fields),
            'policy_burst': list(burst_pair_fields),
        },
        'policy_input_burst_evaluation': {
            'enabled': True,
            'lengths': [5, 20, 50],
            'starts': [75, 150, 225, 300, 375] * 4,
            'minimum_raw_valid_overwrite_rate': 0.8,
            'conditions_are_synthetic_tracker_output_interventions': True,
            'conditions_are_visual_occlusions': False,
            'pipeline': 'raw590_then_burst_then_last_valid_then_3_frame_stack',
        },
        'support_class': 'diagnostic_oracle_support_from_simulator_segmentation',
    },
    'input_provenance': inputs,
    'immutable_input_recheck': immutable_checks,
    'engineering_gates': engineering,
    'failures': {
        'jobs': jobs,
        'structure': structure,
        'runtime_health': runtime_failures,
        'pairing': pairing_failures,
    },
    'scientific_outcomes': {
        'scope': 'descriptive reward evidence only; never an engineering gate',
        'uncertainty_scope': (
            'paired episode statistics and t intervals are conditional on one '
            'training seed; training-seed uncertainty is not estimated'
        ),
        'normal_retention': {
            'per_task': normal_outcomes,
            'all_tasks_last_valid_at_least_95pct_hard_zero': normal_gate,
        },
        'burst_improvement': {
            'per_task': burst_outcomes,
            'all_tasks_positive_at_burst_20_and_50': burst_gate,
        },
        'development_go': development_go,
    },
    'recommendation': (
        'development_go_continue_fixed_last_valid_memory_research'
        if development_go
        else (
            'development_no_go_do_not_scale_fixed_last_valid_memory_yet'
            if engineering_pass else 'fix_engineering_before_reward_interpretation'
        )
    ),
    'worker_return_codes': worker_rcs,
    'gpu_compile_preflight_return_codes': gpu_contract_rcs,
    'elapsed_seconds': int(time.time()) - started,
    'tasks': reports,
}
temporary = summary_path.with_name(summary_path.name + '.tmp')
with temporary.open('x', encoding='utf-8', newline='\n') as file:
    json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    file.write('\n')
temporary.replace(summary_path)
print(json.dumps({
    'status': summary['status'],
    'scientific_scope': summary['scientific_scope'],
    'engineering_gates': engineering,
    'scientific_outcomes': summary['scientific_outcomes'],
    'recommendation': summary['recommendation'],
    'summary': str(summary_path),
}, ensure_ascii=False, indent=2, allow_nan=False))
raise SystemExit(0 if engineering_pass else 4)
PY
SUMMARY_RC=$?
set -e
if (( SUMMARY_RC != 0 )); then
	exit "$SUMMARY_RC"
fi

echo "[5/6] Promoting immutable diagnostic result"
mv -- "$STAGE" "$BASE"
PROMOTED=1
trap - EXIT INT TERM

echo "[6/6] Complete"
echo "CUTIE_OBJECT_MEMORY_PROBE_COMPLETE"
echo "SUMMARY=$BASE/memory_probe_summary.json"
