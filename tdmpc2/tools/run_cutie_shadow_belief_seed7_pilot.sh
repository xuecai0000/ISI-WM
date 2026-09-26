#!/usr/bin/env bash
# Two-GPU seed-7 shadow-belief pilot.
#
# The learned belief is optimized from replay but is forbidden from controlling
# training collection.  Each final checkpoint is evaluated in four fresh
# processes: measurement/prior x normal/canonical burst-20.  Frozen hard-zero
# results from the v1 memory probe are read-only anchors.

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
SOURCE_MEMORY_ROOT="${SOURCE_MEMORY_ROOT:-$REPO_ROOT/logs/_diagnostic/cutie_object_memory_probe_100k_v1_seed7}"
GPU_REACHER="${GPU_REACHER:-0}"
GPU_CARTPOLE="${GPU_CARTPOLE:-1}"

readonly SEED=7 STEPS=100000 EVAL_FREQ=20000 EVAL_EPISODES=3
readonly HELDOUT_EPISODES=20 ENV_SEED=424243 BACKGROUND_SEED=1618034
readonly PLANNER_SEED_BASE=8675400
readonly RUN_TAG=cutie_learned_belief_seed7_pilot_v5_shadow
readonly FORMAT=cutie_learned_belief_seed7_pilot_v5_shadow
readonly BASE="$REPO_ROOT/logs/_diagnostic/$RUN_TAG"
readonly STAGE="${BASE}.incomplete"
readonly SUMMARY="$STAGE/shadow_belief_seed7_summary.json"
readonly -a TASKS=(reacher-visual-small cartpole-swingup)
readonly -a CONDITIONS=(normal burst_20)
readonly -a ARMS=(measurement_only learned_prior)

for name in GPU_REACHER GPU_CARTPOLE; do
	value="${!name}"
	[[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || {
		echo "$name is invalid: $value" >&2; exit 2;
	}
done
[[ "$GPU_REACHER" != "$GPU_CARTPOLE" ]] || {
	echo "GPU_REACHER and GPU_CARTPOLE must differ." >&2; exit 2;
}
(( BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 1) )) || {
	echo "Bash >=5.1 is required." >&2; exit 2;
}
if [[ "$PY" == */* ]]; then
	[[ -x "$PY" ]] || { echo "Python is not executable: $PY" >&2; exit 2; }
else
	command -v "$PY" >/dev/null || { echo "Python is not on PATH: $PY" >&2; exit 2; }
fi
command -v pgrep >/dev/null || { echo "pgrep is required." >&2; exit 2; }
for path in "$VIDEO_ROOT" "$MANIFEST_DIR" "$SUPPORT_BASE" "$SOURCE_MEMORY_ROOT" \
	"$OC_REPO" "$CUTIE_CKPT"; do
	[[ -e "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done
[[ "$(basename -- "${VIDEO_ROOT%/}")" == video_hard ]] || {
	echo "VIDEO_ROOT must be the frozen video_hard directory." >&2; exit 2;
}
for path in \
	tdmpc2/config.yaml \
	tdmpc2/train.py \
	tdmpc2/common/buffer.py \
	tdmpc2/common/cutie_object_belief.py \
	tdmpc2/common/layers.py \
	tdmpc2/common/world_model.py \
	tdmpc2/tdmpc2.py \
	tdmpc2/envs/dmcontrol.py \
	tdmpc2/envs/wrappers/cutie_object.py \
	tdmpc2/envs/wrappers/foreground_stress.py \
	tdmpc2/envs/wrappers/video_background.py \
	tdmpc2/trainer/online_trainer.py \
	tdmpc2/check_cutie_learned_belief_contract.py \
	tdmpc2/check_cutie_learned_belief_update.py \
	tdmpc2/tools/evaluate_cutie_shadow_belief.py \
	tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py \
	tdmpc2/tools/evaluate_cutie_multitask_policy_burst.py; do
	[[ -f "$path" ]] || { echo "Missing repository file: $path" >&2; exit 2; }
done

source_root() {
	local task=$1
	printf '%s/logs/%s/7/cutie_object_hard_zero100k_cutie_object_memory_probe_100k_v1_seed7_%s' \
		"$REPO_ROOT" "$task" "${task//-/_}"
}

experiment_name() {
	local task=$1
	printf 'cutie_object_shadow_belief100k_%s_%s' "$RUN_TAG" "${task//-/_}"
}

run_root() {
	local task=$1
	printf '%s/logs/%s/7/%s' "$REPO_ROOT" "$task" "$(experiment_name "$task")"
}

for path in "$BASE" "$STAGE"; do
	[[ ! -e "$path" ]] || { echo "Refusing existing output: $path" >&2; exit 3; }
done
for task in "${TASKS[@]}"; do
	root="$(source_root "$task")"
	[[ -f "$root/runtime_config.json" && -f "$root/models/final.pt" ]] || {
		echo "Missing hard-zero source checkpoint: $root" >&2; exit 2;
	}
	for path in \
		"$SOURCE_MEMORY_ROOT/tasks/$task/evaluations/normal/hard_zero.json" \
		"$SOURCE_MEMORY_ROOT/tasks/$task/evaluations/burst_20/hard_zero.json" \
		"$SOURCE_MEMORY_ROOT/plans/$task/burst_20.json" \
		"$SUPPORT_BASE/$task/annotations.json"; do
		[[ -f "$path" ]] || { echo "Missing frozen task input: $path" >&2; exit 2; }
	done
	[[ ! -e "$(run_root "$task")" ]] || {
		echo "Refusing existing shadow training root: $(run_root "$task")" >&2; exit 3;
	}
done

mkdir -p "$STAGE/contracts" "$STAGE/provenance" "$STAGE/tasks"
START_SECONDS="$(date +%s)"
ACTIVE_PIDS=()
PROMOTED=0

terminate_tree() {
	local parent=$1 child
	while IFS= read -r child; do
		[[ -n "$child" ]] && terminate_tree "$child"
	done < <(pgrep -P "$parent" 2>/dev/null || true)
	kill -TERM "$parent" 2>/dev/null || true
}

archive_on_exit() {
	local rc=$? pid task root archive_dir failed relocations_tsv relocations_json
	trap - EXIT INT TERM
	for pid in "${ACTIVE_PIDS[@]:-}"; do
		[[ -n "$pid" ]] && terminate_tree "$pid"
	done
	for pid in "${ACTIVE_PIDS[@]:-}"; do
		[[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
	done
	if (( PROMOTED == 0 )) && [[ -d "$STAGE" ]]; then
		if [[ ! -f "$SUMMARY" ]]; then
			"$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps({"format":sys.argv[3],"status":"shadow_engineering_fail","runner_exit_code":int(sys.argv[2]),"failure":"runner exited before aggregation"},indent=2)+"\n",encoding="utf-8")' "$SUMMARY" "$rc" "$FORMAT" 2>/dev/null || true
		fi
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
		[[ ! -e "$failed" ]] || failed="${failed}.${RANDOM}"
		relocations_tsv="$STAGE/provenance/failed_training_root_relocations.tsv"
		relocations_json="$STAGE/provenance/failed_training_root_relocations.json"
		mkdir -p "$STAGE/provenance"
		: >"$relocations_tsv"
		# Hydra training roots are outside STAGE. Archive only these exact
		# attempt-scoped roots so evidence survives and reruns remain unambiguous.
		for task in "${TASKS[@]}"; do
			root="$(run_root "$task")"
			if [[ -e "$root" ]]; then
				archive_dir="$STAGE/training_roots/$task"
				mkdir -p "$archive_dir"
				if mv -- "$root" "$archive_dir/run"; then
					printf '%s\t%s\t%s\ttrue\n' "$task" "$root" \
						"training_roots/$task/run" >>"$relocations_tsv"
				else
					printf '%s\t%s\t%s\tfalse\n' "$task" "$root" \
						"training_roots/$task/run" >>"$relocations_tsv"
					echo "WARNING: failed to archive training root: $root" >&2
				fi
			else
				printf '%s\t%s\t%s\tabsent\n' "$task" "$root" \
					"training_roots/$task/run" >>"$relocations_tsv"
			fi
		done
		"$PY" - "$SUMMARY" "$relocations_tsv" "$relocations_json" "$failed" <<'PY'
import json, os, sys
from pathlib import Path

summary, source, output, failed_root = map(Path, sys.argv[1:5])
rows = []
for line in source.read_text(encoding='utf-8').splitlines():
    task, original, relative, state = line.split('\t')
    rows.append({
        'task': task,
        'original_training_root': original,
        'archived_relative_to_summary_root': relative,
        'source_state': state,
        'moved': state == 'true',
    })
payload = {
    'format': 'cutie_shadow_failed_training_root_relocations_v1',
    'failed_archive_root': str(failed_root.resolve()),
    'records': rows,
}
temporary = output.with_name(output.name + '.incomplete')
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
    encoding='utf-8', newline='\n',
)
os.replace(temporary, output)
if summary.is_file():
    report = json.loads(summary.read_text(encoding='utf-8'))
    report['failure_archive'] = {
        'root': str(failed_root.resolve()),
        'training_root_relocations_relative_to_summary_root': (
            'provenance/failed_training_root_relocations.json'
        ),
        'artifact_resolution': (
            'For moved training roots, replace the original absolute root with '
            'the matching archived_relative_to_summary_root entry.'
        ),
    }
    replacement = summary.with_name(summary.name + '.incomplete')
    replacement.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8', newline='\n',
    )
    os.replace(replacement, summary)
PY
		rm -f -- "$relocations_tsv"
		mv -- "$STAGE" "$failed"
		echo "CUTIE_SHADOW_BELIEF_FAILED_ARCHIVE=$failed" >&2
	fi
	exit "$rc"
}
trap archive_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

write_rc() { printf '%s\n' "$2" >"$1"; }
run_contract() {
	local label=$1; shift
	echo "[contract] $label"
	{ echo "===== $label ====="; "$@"; } >>"$STAGE/contracts/contracts.log" 2>&1
}

echo "[1/6] Static contracts"
bash -n "$0"
run_contract learned_belief "$PY" -m tdmpc2.check_cutie_learned_belief_contract
run_contract evaluator_compile "$PY" -m py_compile \
	tdmpc2/tools/evaluate_cutie_shadow_belief.py

echo "[2/6] Bind immutable sources and implementation"
"$PY" - "$REPO_ROOT" "$SOURCE_MEMORY_ROOT" "$SUPPORT_BASE" \
	"$VIDEO_ROOT" "$MANIFEST_DIR" "$OC_REPO" "$CUTIE_CKPT" \
	"$STAGE/provenance/inputs.json" <<'PY'
import hashlib, json, sys
from pathlib import Path

repo, source, support, video, manifests, oc_repo, cutie, output = map(
    Path, sys.argv[1:9]
)
tasks = ('reacher-visual-small', 'cartpole-swingup')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
def file_map(root, suffixes=None):
    result = {}
    for path in sorted(item for item in root.rglob('*') if item.is_file()):
        if suffixes is not None and path.suffix.lower() not in suffixes:
            continue
        result[path.relative_to(root).as_posix()] = {
            'path': str(path.resolve()), 'sha256': digest(path),
        }
    if not result:
        raise RuntimeError(f'No immutable files under {root}')
    return result
def metadata_inventory(root):
    rows = []
    for path in sorted(item for item in root.rglob('*') if item.is_file()):
        stat = path.stat()
        rows.append(
            f'{path.relative_to(root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n'
        )
    if not rows:
        raise RuntimeError(f'Empty video inventory: {root}')
    encoded = ''.join(rows).encode('utf-8')
    return {'files': len(rows), 'sha256': hashlib.sha256(encoded).hexdigest()}

summary = source / 'memory_probe_summary.json'
payload = json.loads(summary.read_text(encoding='utf-8'))
if payload.get('status') != 'object_memory_probe_engineering_pass':
    raise RuntimeError('Frozen memory-probe anchor is not engineering-pass.')
implementation = {}
for relative in (
    'tdmpc2/config.yaml', 'tdmpc2/train.py', 'tdmpc2/common/buffer.py',
    'tdmpc2/common/cutie_object_belief.py', 'tdmpc2/common/layers.py',
    'tdmpc2/common/world_model.py',
    'tdmpc2/tdmpc2.py', 'tdmpc2/envs/dmcontrol.py',
    'tdmpc2/envs/wrappers/cutie_object.py',
    'tdmpc2/envs/wrappers/foreground_stress.py',
    'tdmpc2/envs/wrappers/video_background.py',
    'tdmpc2/trainer/online_trainer.py',
    'tdmpc2/check_cutie_learned_belief_contract.py',
    'tdmpc2/check_cutie_learned_belief_update.py',
    'tdmpc2/tools/evaluate_cutie_shadow_belief.py',
    'tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py',
    'tdmpc2/tools/evaluate_cutie_multitask_policy_burst.py',
    'tdmpc2/tools/run_cutie_shadow_belief_seed7_pilot.sh',
):
    path = repo / relative
    implementation[relative] = {'path': str(path.resolve()), 'sha256': digest(path)}
anchors = {}
support_files = {}
for task in tasks:
    task_key = task.replace('-', '_')
    root = repo / 'logs' / task / '7' / (
        f'cutie_object_hard_zero100k_cutie_object_memory_probe_100k_v1_seed7_{task_key}'
    )
    files = {
        'runtime_config': root / 'runtime_config.json',
        'checkpoint': root / 'models' / 'final.pt',
        'normal': source / 'tasks' / task / 'evaluations' / 'normal' / 'hard_zero.json',
        'burst_20': source / 'tasks' / task / 'evaluations' / 'burst_20' / 'hard_zero.json',
        'plan_burst_20': source / 'plans' / task / 'burst_20.json',
        'support': support / task / 'annotations.json',
    }
    anchors[task] = {}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        anchors[task][name] = {'path': str(path.resolve()), 'sha256': digest(path)}
    support_files[task] = file_map(support / task)
cutie_root = oc_repo / 'feature_extractor' / 'cutie' / 'cutie'
oc_sources = {}
for name, relative in {
    'inference_core': 'inference/inference_core.py',
    'object_manager': 'inference/object_manager.py',
    'image_feature_store': 'inference/image_feature_store.py',
    'kv_memory_store': 'inference/kv_memory_store.py',
    'memory_manager': 'inference/memory_manager.py',
    'object_transformer': 'model/transformer/object_transformer.py',
    'cutie_model': 'model/cutie.py',
}.items():
    path = cutie_root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    oc_sources[name] = {'path': str(path.resolve()), 'sha256': digest(path)}
video_inventory = metadata_inventory(video)
result = {
    'format': 'cutie_shadow_belief_inputs_v1',
    'source_memory_summary': {'path': str(summary.resolve()), 'sha256': digest(summary)},
    'hard_zero_anchors': anchors,
    'support_task_files': support_files,
    'cutie_checkpoint': {'path': str(cutie.resolve()), 'sha256': digest(cutie)},
    'implementation': implementation,
    'external': {
        'video_root': {
            'path': str(video.resolve()),
            'inventory_files': video_inventory['files'],
            'inventory_sha256': video_inventory['sha256'],
        },
        'manifest_dir': {
            'path': str(manifests.resolve()),
            'files': file_map(manifests),
        },
        'oc_repo': {
            'path': str(oc_repo.resolve()),
            'cutie_sources': oc_sources,
            'config_files': file_map(cutie_root / 'config', {'.yaml', '.yml'}),
        },
    },
}
with output.open('x', encoding='utf-8', newline='\n') as f:
    json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write('\n')
print('CUTIE_SHADOW_INPUTS_OK', output)
PY

run_gpu_contract() {
	local label=$1 gpu=$2 mode=$3 task=$4 log="$STAGE/contracts/${1}.log" rc
	local -a extra=()
	[[ "$mode" == compile ]] && extra+=(--compile)
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" tdmpc2/check_cutie_learned_belief_update.py \
		--task "$task" "${extra[@]}" >"$log" 2>&1
	rc=$?
	set -e
	write_rc "$STAGE/contracts/${label}.rc" "$rc"
	return "$rc"
}

echo "[3/6] Real replay and compile GPU contracts"
set +e
run_gpu_contract eager_gpu_reacher "$GPU_REACHER" eager reacher-visual-small & P0=$!; ACTIVE_PIDS+=("$P0")
run_gpu_contract compile_gpu_cartpole "$GPU_CARTPOLE" compile cartpole-swingup & P1=$!; ACTIVE_PIDS+=("$P1")
wait "$P0"; RC0=$?
wait "$P1"; RC1=$?
set -e
ACTIVE_PIDS=()
(( RC0 == 0 && RC1 == 0 )) || { echo "GPU contracts failed: $RC0 $RC1" >&2; exit 4; }

run_training() {
	local gpu=$1 task=$2 dir=$3 support role0 role1 exp root log hydra rc
	support="$SUPPORT_BASE/$task/annotations.json"
	case "$task" in
		reacher-visual-small) role0=whole_arm; role1=goal ;;
		cartpole-swingup) role0=cart; role1=pole ;;
		*) return 2 ;;
	esac
	exp="$(experiment_name "$task")"
	root="$(run_root "$task")"
	log="$dir/shadow.train.log"
	hydra="$dir/hydra_shadow"
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
		cutie_object_last_valid_memory=false cutie_object_policy_burst_plan=null
		cutie_object_config_dir=null cutie_object_device=cuda:0
		cutie_object_tracker_height=448 cutie_object_tracker_width=448
		cutie_object_model_size=small cutie_object_prompt_radius=2.0 cutie_object_amp=true
		cutie_object_worker_timeout_seconds=180 cutie_object_num_roles=2
		cutie_object_frame_dim=590 cutie_object_stack_frames=3 cutie_object_input_dim=1770
		cutie_object_role_dim=64 cutie_object_hidden_dim=256 cutie_object_joint_dim=640
		cutie_object_only_latent_dim=128 cutie_object_belief_enabled=true
		cutie_object_belief_use_for_control=false
		cutie_object_belief_batch_size=128 cutie_object_belief_burn_in=3
		cutie_object_belief_min_burst=5 cutie_object_belief_max_burst=20
		cutie_object_belief_recovery_frames=1 cutie_object_belief_update_frequency=4
		cutie_object_belief_lr=0.0003 cutie_object_belief_loss_coef=1.0
		cutie_object_belief_reacquisition_coef=1.0
		cutie_object_belief_mask_seed_offset=104729
		cutie_object_belief_replay_seed_offset=130363
		"exp_name=$exp" "hydra.run.dir=$hydra" hydra.job.chdir=false
	)
	echo "TRAIN_START task=$task mode=measurement_only_shadow gpu=$gpu root=$root" | tee "$log"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" tdmpc2/train.py "${args[@]}" >>"$log" 2>&1
	rc=$?
	set -e
	write_rc "$dir/shadow.train.rc" "$rc"
	echo "TRAIN_END task=$task gpu=$gpu rc=$rc" | tee -a "$log"
}

run_eval() {
	local gpu=$1 task=$2 dir=$3 condition=$4 arm=$5 root outdir out log role rc
	root="$(run_root "$task")"
	outdir="$dir/evaluations/$condition"
	mkdir -p "$outdir"
	out="$outdir/$arm.json"
	log="$outdir/$arm.log"
	local -a burst_args=()
	if [[ "$condition" == burst_20 ]]; then
		case "$task" in
			reacher-visual-small) role=whole_arm ;;
			cartpole-swingup) role=pole ;;
		esac
		burst_args=(
			--policy-burst-plan "$SOURCE_MEMORY_ROOT/plans/$task/burst_20.json"
			--expected-role "$role" --expected-length 20
		)
	fi
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" -m tdmpc2.tools.evaluate_cutie_shadow_belief \
		--task "$task" --condition "$condition" --arm "$arm" \
		--runtime-config "$root/runtime_config.json" --checkpoint "$root/models/final.pt" \
		--training-seed "$SEED" --expected-training-steps "$STEPS" \
		--expected-training-eval-freq "$EVAL_FREQ" \
		--expected-training-eval-episodes "$EVAL_EPISODES" \
		--env-seed "$ENV_SEED" --background-seed "$BACKGROUND_SEED" \
		--planner-seed-base "$PLANNER_SEED_BASE" "${burst_args[@]}" \
		--output "$out" >"$log" 2>&1
	rc=$?
	set -e
	write_rc "$outdir/$arm.rc" "$rc"
	echo "EVAL_END task=$task condition=$condition arm=$arm rc=$rc" | tee -a "$log"
}

run_task() {
	local gpu=$1 task=$2 dir="$STAGE/tasks/$2" condition arm
	mkdir -p "$dir/evaluations/normal" "$dir/evaluations/burst_20"
	printf '%s\n' "$gpu" >"$dir/gpu"
	run_training "$gpu" "$task" "$dir"
	if [[ "$(<"$dir/shadow.train.rc")" == 0 ]]; then
		# Every call below is a new Python process and a new agent/environment.
		for condition in "${CONDITIONS[@]}"; do
			for arm in "${ARMS[@]}"; do
				run_eval "$gpu" "$task" "$dir" "$condition" "$arm"
			done
		done
	else
		for condition in "${CONDITIONS[@]}"; do
			for arm in "${ARMS[@]}"; do
				write_rc "$dir/evaluations/$condition/$arm.rc" 125
			done
		done
	fi
}

echo "[4/6] Two-task shadow training and four fresh-process evaluations"
run_task "$GPU_REACHER" reacher-visual-small & PR=$!; ACTIVE_PIDS+=("$PR")
run_task "$GPU_CARTPOLE" cartpole-swingup & PC=$!; ACTIVE_PIDS+=("$PC")
set +e
wait "$PR"; RCR=$?
wait "$PC"; RCC=$?
set -e
ACTIVE_PIDS=()
write_rc "$STAGE/reacher_worker.rc" "$RCR"
write_rc "$STAGE/cartpole_worker.rc" "$RCC"

echo "[5/6] Strict aggregation and shadow gate"
set +e
"$PY" - "$STAGE" "$SUMMARY" "$REPO_ROOT" "$START_SECONDS" \
	"$GPU_REACHER" "$GPU_CARTPOLE" <<'PY'
import csv, hashlib, json, math, statistics, sys, time
from pathlib import Path
import torch

stage, summary_path, repo = map(Path, sys.argv[1:4])
started = int(sys.argv[4])
gpu_map = {'reacher-visual-small': sys.argv[5], 'cartpole-swingup': sys.argv[6]}
tasks = ('reacher-visual-small', 'cartpole-swingup')
conditions = ('normal', 'burst_20')
arms = ('measurement_only', 'learned_prior')
task_roles = {
    'reacher-visual-small': ('whole_arm', 'goal'),
    'cartpole-swingup': ('cart', 'pole'),
}

def load(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'JSON root is not object: {path}')
    return value
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
def rc(path):
    try: return int(path.read_text(encoding='utf-8').strip())
    except Exception: return None
def rewards(payload):
    values = [float(row['reward']) for row in payload.get('episodes', [])]
    if len(values) != 20 or not all(math.isfinite(value) for value in values):
        raise ValueError('Expected 20 finite episode rewards.')
    return values
def paired(left, right):
    delta = [b - a for a, b in zip(left, right)]
    mean = statistics.fmean(delta)
    sd = statistics.stdev(delta)
    half = 2.093024054 * sd / math.sqrt(20)
    return {
        'left_mean': statistics.fmean(left),
        'right_mean': statistics.fmean(right),
        'right_minus_left_mean': mean,
        'median_delta': statistics.median(delta),
        'win_tie_loss': [sum(x > 0 for x in delta), sum(x == 0 for x in delta), sum(x < 0 for x in delta)],
        'paired_t_95pct_interval': [mean - half, mean + half],
        'interval_scope': 'conditional paired-episode t interval for one frozen training seed; not uncertainty across training seeds',
        'paired_deltas': delta,
    }
def pair_exact(left, right, fields):
    mismatches = {
        field: [index for index, (a, b) in enumerate(zip(left, right)) if a.get(field) != b.get(field)]
        for field in fields
    }
    return {key: value for key, value in mismatches.items() if value}
def artifact_entry(path):
    item = {'sha256': digest(path)}
    try:
        relative = path.relative_to(stage).as_posix()
    except ValueError:
        item['path'] = str(path.resolve())
    else:
        item['relative_to_summary_root'] = relative
        item['execution_time_staging_path'] = str(path.resolve())
    return item
def verify_file_map(items, label):
    failures = []
    if not isinstance(items, dict) or not items:
        return [f'{label} file map is empty']
    for relative, item in items.items():
        try:
            path = Path(item['path'])
            if not path.is_file() or digest(path) != item['sha256']:
                failures.append(f'{label} changed: {relative}')
        except Exception as exc:
            failures.append(f'{label} recheck failed for {relative}: {exc}')
    return failures
def metadata_inventory(root):
    rows = []
    for path in sorted(item for item in root.rglob('*') if item.is_file()):
        stat = path.stat()
        rows.append(f'{path.relative_to(root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n')
    encoded = ''.join(rows).encode('utf-8')
    return len(rows), hashlib.sha256(encoded).hexdigest()
def nonnegative_int(value, upper=None):
    return (
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        and (upper is None or value <= upper)
    )
def role_diagnostics_exact(value, roles, frames, expected_episodes):
    if not isinstance(value, dict) or value.get('frames') != frames:
        return False
    if value.get('role_diagnostics_schema') != 'cutie_role_runtime_diagnostics_v1':
        return False
    metrics = value.get('role_metrics')
    episodes = value.get('episode_metrics')
    if not isinstance(metrics, dict) or set(metrics) != set(roles):
        return False
    for role in roles:
        row = metrics[role]
        if not isinstance(row, dict):
            return False
        if not all(nonnegative_int(row.get(key), frames) for key in (
            'valid_frames', 'invalid_frames', 'lost_frames',
            'empty_mask_frames', 'nonfinite_feature_frames',
        )):
            return False
        if row['valid_frames'] + row['invalid_frames'] != frames:
            return False
        for key in (
            'valid_frame_rate', 'lost_frame_rate', 'empty_mask_frame_rate',
            'nonfinite_feature_frame_rate', 'mask_touches_border_rate',
        ):
            metric = row.get(key)
            if type(metric) not in (int, float) or not math.isfinite(float(metric)):
                return False
            if not 0.0 <= float(metric) <= 1.0:
                return False
        for key in ('mean_mask_area_pixels', 'mean_confidence', 'mean_mask_score'):
            metric = row.get(key)
            if type(metric) not in (int, float) or not math.isfinite(float(metric)):
                return False
        if not nonnegative_int(row.get('max_invalid_burst'), frames):
            return False
    if not isinstance(episodes, list) or len(episodes) != expected_episodes:
        return False
    for index, row in enumerate(episodes):
        if not isinstance(row, dict) or row.get('episode_index') != index:
            return False
        if row.get('frames') != 501:
            return False
        invalid = row.get('per_role_invalid_frames')
        bursts = row.get('per_role_max_invalid_burst')
        if not isinstance(invalid, dict) or set(invalid) != set(roles):
            return False
        if not isinstance(bursts, dict) or set(bursts) != set(roles):
            return False
        if not all(
            nonnegative_int(invalid[role], 501)
            and nonnegative_int(bursts[role], 501)
            for role in roles
        ):
            return False
    return sum(row['frames'] for row in episodes) == frames
def evaluator_implementation_exact(payload, inputs):
    mapping = {
        'evaluator': 'tdmpc2/tools/evaluate_cutie_shadow_belief.py',
        'base_evaluator': 'tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py',
        'burst_base_evaluator': 'tdmpc2/tools/evaluate_cutie_multitask_policy_burst.py',
        'agent': 'tdmpc2/tdmpc2.py',
        'world_model': 'tdmpc2/common/world_model.py',
        'belief_helpers': 'tdmpc2/common/cutie_object_belief.py',
        'object_wrapper': 'tdmpc2/envs/wrappers/cutie_object.py',
    }
    actual = payload.get('provenance', {}).get('implementation')
    if not isinstance(actual, dict) or set(actual) != set(mapping):
        return False
    for name, relative in mapping.items():
        expected = inputs['implementation'][relative]
        row = actual[name]
        if not isinstance(row, dict):
            return False
        try:
            path_equal = Path(row.get('path', '')).resolve() == Path(expected['path']).resolve()
        except Exception:
            return False
        if not path_equal or row.get('sha256') != expected['sha256']:
            return False
        if row.get('sha256_after') != expected['sha256']:
            return False
    return True
def anchor_identity_exact(
    payload, condition, task, anchor_items, inputs, source_memory_summary
):
    provenance = payload.get('provenance', {})
    evaluation = payload.get('evaluation', {})
    episodes = payload.get('episodes')
    cutie = provenance.get('cutie_inputs')
    common = (
        payload.get('task') == task
        and payload.get('training_seed') == 7
        and payload.get('backend') == 'cutie_object_only'
        and evaluation.get('split') == 'validation'
        and evaluation.get('episodes') == 20
        and evaluation.get('env_seed') == 424243
        and evaluation.get('background_seed') == 1618034
        and evaluation.get('planner_seed_base') == 8675400
        and isinstance(episodes, list) and len(episodes) == 20
        and [row.get('episode_index') for row in episodes] == list(range(20))
        and provenance.get('runtime_config_sha256') == anchor_items['runtime_config']['sha256']
        and provenance.get('checkpoint_sha256') == anchor_items['checkpoint']['sha256']
        and isinstance(cutie, dict)
        and cutie.get('checkpoint_sha256') == inputs['cutie_checkpoint']['sha256']
        and cutie.get('support_sha256') == anchor_items['support']['sha256']
        and cutie.get('roles') == list(task_roles[task])
        and cutie.get('support_schema') == 'generic_indexed_v1'
    )
    if condition == 'normal':
        historical_evaluator = source_memory_summary['input_provenance'][
            'implementation'
        ]['tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py']['sha256']
        return common and (
            payload.get('format') == 'cutie_multitask_checkpoint_evaluation_v1'
            and payload.get('erosion_pixels') == 0
            and evaluation.get('actual_foreground_erosion_pixels') == 0
            and provenance.get('evaluator_sha256') == historical_evaluator
            and role_diagnostics_exact(
                payload.get('perception_runtime'), task_roles[task], 10020, 20
            )
        )
    strict = payload.get('strict_checks')
    policy = payload.get('policy_burst', {})
    historical_evaluator = source_memory_summary['input_provenance'][
        'implementation'
    ]['tdmpc2/tools/evaluate_cutie_multitask_policy_burst.py']['sha256']
    return common and (
        payload.get('format') == 'cutie_multitask_policy_burst_evaluation_v1'
        and payload.get('arm') == 'hard_zero'
        and isinstance(strict, dict) and bool(strict) and all(strict.values())
        and policy.get('format') == 'cutie_policy_burst_plan_v1'
        and policy.get('length') == 20 and policy.get('episodes') == 20
        and provenance.get('policy_burst_plan_sha256')
            == anchor_items['plan_burst_20']['sha256']
        and provenance.get('runtime_config_sha256_after')
            == anchor_items['runtime_config']['sha256']
        and provenance.get('checkpoint_sha256_after')
            == anchor_items['checkpoint']['sha256']
        and provenance.get('evaluator_sha256') == historical_evaluator
        and role_diagnostics_exact(
            payload.get('perception_runtime'), task_roles[task], 10020, 20
        )
    )

inputs_path = stage / 'provenance' / 'inputs.json'
inputs = load(inputs_path)
source_memory_summary = load(Path(inputs['source_memory_summary']['path']))
engineering = []
reports = {}
try:
    if digest(Path(inputs['source_memory_summary']['path'])) != inputs['source_memory_summary']['sha256']:
        engineering.append('source memory summary changed')
    if digest(Path(inputs['cutie_checkpoint']['path'])) != inputs['cutie_checkpoint']['sha256']:
        engineering.append('Cutie checkpoint changed')
    for relative, item in inputs['implementation'].items():
        path = repo / relative
        if path.resolve() != Path(item['path']).resolve() or digest(path) != item['sha256']:
            engineering.append(f'implementation changed: {relative}')
    external = inputs['external']
    engineering.extend(verify_file_map(
        external['manifest_dir']['files'], 'background manifest'
    ))
    engineering.extend(verify_file_map(
        external['oc_repo']['cutie_sources'], 'Cutie source'
    ))
    engineering.extend(verify_file_map(
        external['oc_repo']['config_files'], 'Cutie config'
    ))
    for task, items in inputs['support_task_files'].items():
        engineering.extend(verify_file_map(items, f'{task} support'))
    video_count, video_sha = metadata_inventory(Path(external['video_root']['path']))
    if (
        video_count != external['video_root']['inventory_files']
        or video_sha != external['video_root']['inventory_sha256']
    ):
        engineering.append('video_hard inventory changed during run')
except Exception as exc:
    engineering.append(f'immutable implementation recheck failed: {exc}')

for task in tasks:
    task_dir = stage / 'tasks' / task
    task_key = task.replace('-', '_')
    root = repo / 'logs' / task / '7' / f'cutie_object_shadow_belief100k_cutie_learned_belief_seed7_pilot_v5_shadow_{task_key}'
    paths = {
        'runtime': root / 'runtime_config.json',
        'checkpoint': root / 'models' / 'final.pt',
        'curve': root / 'eval.csv',
        'trainer': root / 'trainer_runtime.json',
        'replay': root / 'replay_runtime.json',
        'perception': root / 'perception_runtime.json',
    }
    codes = {'train': rc(task_dir / 'shadow.train.rc')}
    for condition in conditions:
        for arm in arms:
            key = f'{condition}_{arm}'
            codes[key] = rc(task_dir / 'evaluations' / condition / f'{arm}.rc')
            paths[key] = task_dir / 'evaluations' / condition / f'{arm}.json'
    missing = [str(path) for path in paths.values() if not path.is_file()]
    report = {'gpu': gpu_map[task], 'new_root': str(root), 'job_return_codes': codes, 'missing': missing}
    reports[task] = report
    if any(value != 0 for value in codes.values()):
        engineering.append(f'{task} job rc: {codes}')
    if missing:
        engineering.append(f'{task} missing artifacts: {missing}')
        continue
    try:
        runtime = load(paths['runtime'])
        trainer = load(paths['trainer'])
        replay = load(paths['replay'])
        perception = load(paths['perception'])
        checkpoint = torch.load(paths['checkpoint'], map_location='cpu', weights_only=False)
        contract = checkpoint.get('checkpoint_contract') if isinstance(checkpoint, dict) else None
        supervision = contract.get('cutie_object_belief_supervision') if isinstance(contract, dict) else None
        payloads = {
            condition: {arm: load(paths[f'{condition}_{arm}']) for arm in arms}
            for condition in conditions
        }
        with paths['curve'].open(encoding='utf-8', newline='') as f:
            curve = list(csv.DictReader(f))
        counts_ok = (
            isinstance(supervision, dict)
            and supervision.get('format') == 'cutie_object_belief_supervision_v1'
            and all(isinstance(supervision.get(key), int) and not isinstance(supervision.get(key), bool) and supervision.get(key) >= 0 for key in (
                'attempts', 'successful_updates', 'no_teacher_skips',
                'age20_available_updates', 'reacquisition_available_updates'))
            and all(isinstance(supervision.get(key), list) and len(supervision[key]) == 2 and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in supervision[key]) for key in (
                'age20_teacher_roles', 'reacquisition_teacher_roles'))
        )
        if counts_ok:
            attempts = supervision['attempts']
            updates = supervision['successful_updates']
            skips = supervision['no_teacher_skips']
            counts_ok = updates + skips == attempts and updates == contract.get('cutie_object_belief_aux_updates')
        else:
            attempts = updates = skips = 0
        checks = {
            'runtime_shadow_control_false': runtime.get('cutie_object_belief_use_for_control') is False,
            'runtime_belief_enabled': runtime.get('cutie_object_belief_enabled') is True,
            'runtime_memory_off': runtime.get('cutie_object_last_valid_memory') is False,
            'runtime_burst_off': runtime.get('cutie_object_policy_burst_plan') is None,
            'runtime_protocol': runtime.get('steps') == 100000 and runtime.get('seed') == 7 and runtime.get('eval_freq') == 20000 and runtime.get('eval_episodes') == 3,
            'runtime_object_only': runtime.get('flat_anchor_mode') == 'cutie_object_only' and runtime.get('obs_shape') == {'object': [2, 1770]} and runtime.get('latent_dim') == 128,
            'checkpoint_shadow_control_false': isinstance(contract, dict) and contract.get('cutie_object_belief_use_for_control_during_training') is False,
            'checkpoint_shadow_collection_mode': isinstance(contract, dict) and contract.get('cutie_object_belief_collection_mode') == 'measurement_only_shadow',
            'checkpoint_training_online_prior_zero': isinstance(contract, dict) and contract.get('cutie_object_belief_online_control_before_checkpoint_save') == {'prior_role_uses': 0, 'episode_state_resets': 0},
            'checkpoint_aux_updates': counts_ok and updates > 1000,
            'checkpoint_age20_coverage': counts_ok and min(supervision['age20_teacher_roles']) > 0 and sum(supervision['age20_teacher_roles']) / max(attempts, 1) >= 1.0,
            'checkpoint_reacquisition_coverage': counts_ok and min(supervision['reacquisition_teacher_roles']) > 0,
            'checkpoint_skip_rate': counts_ok and skips / max(attempts, 1) <= 0.01,
            'replay_isolated': replay.get('belief_sequence_length') == 24 and replay.get('belief_batch_size') == 128 and replay.get('belief_replay_rng_isolated') is True,
            'trainer_complete': trainer.get('steps') == 100000 and trainer.get('training_non_eval_steps_per_second', 0) > 0,
            'curve_exact': [int(float(row['step'])) for row in curve] == [0, 20000, 40000, 60000, 80000, 100000],
            'perception_process_healthy': perception.get('worker_restarts') == 0 and perception.get('timeouts') == 0,
            'perception_role_diagnostics_exact': role_diagnostics_exact(
                perception, task_roles[task], 109218, 218
            ),
        }
        runtime_sha = digest(paths['runtime'])
        checkpoint_sha = digest(paths['checkpoint'])
        for condition in conditions:
            for arm in arms:
                value = payloads[condition][arm]
                prefix = f'{condition}_{arm}'
                inference = value.get('inference_runtime', {})
                checks[f'{prefix}_identity'] = value.get('format') == 'cutie_shadow_belief_evaluation_v1' and value.get('task') == task and value.get('condition') == condition and value.get('arm') == arm
                checks[f'{prefix}_same_checkpoint'] = value.get('provenance', {}).get('runtime_config_sha256') == runtime_sha and value.get('provenance', {}).get('checkpoint_sha256') == checkpoint_sha
                checks[f'{prefix}_strict'] = bool(value.get('strict_checks')) and all(value['strict_checks'].values())
                checks[f'{prefix}_gpu'] = value.get('provenance', {}).get('cuda_visible_devices') == gpu_map[task]
                checks[f'{prefix}_implementation'] = evaluator_implementation_exact(
                    value, inputs
                )
                if arm == 'measurement_only':
                    checks[f'{prefix}_online_prior_uses_zero'] = inference.get('prior_role_uses') == 0 and inference.get('belief_dynamics_forward_calls') == 0 and inference.get('belief_resets') == 0
                else:
                    checks[f'{prefix}_online_prior_active'] = inference.get('belief_dynamics_forward_calls') == 9980 and inference.get('prior_role_uses') == inference.get('expected_prior_role_uses')
        pairing = {}
        for condition in conditions:
            measurement = payloads[condition]['measurement_only']
            learned = payloads[condition]['learned_prior']
            fields = measurement['evaluation']['cross_arm_exact_pairing_fields']
            mismatch = pair_exact(measurement['episodes'], learned['episodes'], fields)
            pairing[f'{condition}_same_checkpoint_arms'] = {'exact': not mismatch, 'mismatches': mismatch}
            checks[f'{condition}_same_checkpoint_pairing'] = not mismatch
        anchor_items = inputs['hard_zero_anchors'][task]
        anchors = {}
        for name, item in anchor_items.items():
            path = Path(item['path'])
            checks[f'anchor_{name}_immutable'] = digest(path) == item['sha256']
        anchors['normal'] = load(Path(anchor_items['normal']['path']))
        anchors['burst_20'] = load(Path(anchor_items['burst_20']['path']))
        checks['anchor_normal_identity_protocol_provenance'] = anchor_identity_exact(
            anchors['normal'], 'normal', task, anchor_items, inputs,
            source_memory_summary,
        )
        checks['anchor_burst_20_identity_protocol_provenance'] = anchor_identity_exact(
            anchors['burst_20'], 'burst_20', task, anchor_items, inputs,
            source_memory_summary,
        )
        checks['anchor_burst_20_bound_to_source_summary'] = (
            anchor_items['burst_20']['sha256']
            == source_memory_summary['tasks'][task]['burst_conditions'][
                'burst_20'
            ]['arms']['hard_zero']['sha256']
        )
        anchor_fields = {
            'normal': (
                'episode_index', 'planner_seed', 'planner_rng_start_sha256',
                'planner_rng_end_sha256', 'initial_rgb_sha256',
                'initial_object_sha256', 'background_source',
                'background_start_frame_index', 'length'),
            'burst_20': (
                'episode_index', 'planner_seed', 'planner_rng_start_sha256',
                'planner_rng_end_sha256', 'initial_rgb_sha256',
                'initial_policy_object_sha256', 'initial_raw_object_frame_sha256',
                'background_source', 'background_start_frame_index',
                'background_end_source', 'background_end_frame_index', 'length',
                'policy_burst_event', 'policy_burst_plan_sha256'),
        }
        for condition in conditions:
            mismatch = pair_exact(
                anchors[condition]['episodes'],
                payloads[condition]['measurement_only']['episodes'],
                anchor_fields[condition],
            )
            pairing[f'{condition}_hard_zero_anchor'] = {'exact': not mismatch, 'mismatches': mismatch}
            checks[f'{condition}_hard_zero_pairing'] = not mismatch
        hard_normal = rewards(anchors['normal'])
        hard_burst = rewards(anchors['burst_20'])
        measurement_normal = rewards(payloads['normal']['measurement_only'])
        learned_normal = rewards(payloads['normal']['learned_prior'])
        measurement_burst = rewards(payloads['burst_20']['measurement_only'])
        learned_burst = rewards(payloads['burst_20']['learned_prior'])
        hard_mean = statistics.fmean(hard_normal)
        measurement_mean = statistics.fmean(measurement_normal)
        normal_retention = measurement_mean / hard_mean if hard_mean > 0 else None
        burst_eligible = {
            arm: payloads['burst_20'][arm].get('policy_burst', {}).get('controlled_burst_attribution_eligible') is True
            for arm in arms
        }
        checks['reacher_burst_attribution'] = (
            all(burst_eligible.values()) if task == 'reacher-visual-small' else True
        )
        failed = sorted(key for key, passed in checks.items() if not passed)
        report.update(
            checks=checks,
            failed_checks=failed,
            checkpoint_contract=contract,
            pairing=pairing,
            rewards={
                'normal': {
                    'hard_zero_anchor_mean': hard_mean,
                    'measurement_only_mean': measurement_mean,
                    'learned_prior_mean': statistics.fmean(learned_normal),
                    'measurement_retention_of_hard_zero': normal_retention,
                    'learned_prior_vs_measurement_only': paired(measurement_normal, learned_normal),
                    'measurement_only_vs_hard_zero': paired(hard_normal, measurement_normal),
                },
                'burst_20': {
                    'hard_zero_anchor_mean': statistics.fmean(hard_burst),
                    'measurement_only_mean': statistics.fmean(measurement_burst),
                    'learned_prior_mean': statistics.fmean(learned_burst),
                    'learned_prior_vs_measurement_only': paired(measurement_burst, learned_burst),
                    'measurement_only_vs_hard_zero': paired(hard_burst, measurement_burst),
                    'controlled_attribution_eligible': burst_eligible,
                    'scope': 'controlled attribution for Reacher; robustness stress only for Cartpole',
                },
            },
            shadow_gate={
                'measurement_normal_retention_at_least_95pct': normal_retention is not None and normal_retention >= 0.95,
                'reacher_controlled_burst_attribution': all(burst_eligible.values()) if task == 'reacher-visual-small' else True,
            },
            burst_scope=(
                'controlled_attribution' if task == 'reacher-visual-small'
                else 'robustness_stress_only_due_to_natural_tracker_overlap'
            ),
            artifacts={name: artifact_entry(path) for name, path in paths.items()},
        )
        if failed:
            engineering.append(f'{task} checks failed: {failed}')
    except Exception as exc:
        engineering.append(f'{task} aggregation exception: {type(exc).__name__}: {exc}')
        report['exception'] = repr(exc)

engineering_pass = not engineering
shadow_gate_pass = engineering_pass and all(
    report.get('shadow_gate') and all(report['shadow_gate'].values())
    for report in reports.values()
)
result = {
    'format': 'cutie_learned_belief_seed7_pilot_v5_shadow',
    'status': 'shadow_engineering_pass' if engineering_pass else 'shadow_engineering_fail',
    'scientific_scope': 'single-training-seed shadow-belief development gate; not a paper result',
    'elapsed_hours': (time.time() - started) / 3600.0,
    'engineering_pass': engineering_pass,
    'engineering_failures': engineering,
    'shadow_gate_pass': shadow_gate_pass,
    'recommendation': (
        'shadow_design_valid_compare_belief_effect' if shadow_gate_pass
        else ('stop_and_fix_engineering' if not engineering_pass else 'shadow_controller_not_preserved')
    ),
    'tasks': reports,
    'inputs': {
        'relative_to_summary_root': 'provenance/inputs.json',
        'sha256': digest(inputs_path),
    },
    'summary_relative_paths_authoritative': True,
}
tmp = summary_path.with_suffix('.json.incomplete')
with tmp.open('x', encoding='utf-8', newline='\n') as f:
    json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write('\n')
tmp.replace(summary_path)
print('CUTIE_SHADOW_BELIEF_SUMMARY', json.dumps({
    'status': result['status'], 'shadow_gate': shadow_gate_pass,
    'recommendation': result['recommendation'], 'summary': str(summary_path),
}, ensure_ascii=False))
raise SystemExit(0 if engineering_pass else 4)
PY
AGG_RC=$?
set -e
(( AGG_RC == 0 )) || exit "$AGG_RC"

echo "[6/6] Promote immutable diagnostic"
mv -- "$STAGE" "$BASE"
PROMOTED=1
trap - EXIT INT TERM
echo "CUTIE_SHADOW_BELIEF_SEED7_PILOT_COMPLETE"
echo "SUMMARY=$BASE/shadow_belief_seed7_summary.json"
