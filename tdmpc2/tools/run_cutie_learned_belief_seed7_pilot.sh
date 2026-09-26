#!/usr/bin/env bash
# Two-GPU seed-7 disaster gate for the burst-trained Cutie object belief.
#
# This is a development diagnostic, not a paper protocol. It reuses the
# immutable seed-7 hard-zero/last-valid checkpoints and simulator-derived
# support packs from the completed memory probe, trains only the new learned
# belief arm from scratch, and compares normal plus canonical burst-20 held-out
# validation. Reacher is the controlled-burst attribution task; Cartpole is a
# natural-missing deployment stress because its frozen source already has less
# than 80% raw-valid overlap in the scheduled windows. A negative result is still promoted; only broken jobs,
# provenance, checkpoint identity, or pairing are engineering failures.

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
readonly RUN_TAG=cutie_learned_belief_seed7_pilot_v4
readonly FORMAT=cutie_learned_belief_seed7_pilot_v4
readonly BASE="$REPO_ROOT/logs/_diagnostic/$RUN_TAG"
readonly STAGE="${BASE}.incomplete"
readonly SUMMARY="$STAGE/learned_belief_seed7_summary.json"
readonly -a TASKS=(reacher-visual-small cartpole-swingup)

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
command -v pgrep >/dev/null || { echo "pgrep is required for recursive worker cleanup." >&2; exit 2; }
for path in "$VIDEO_ROOT" "$MANIFEST_DIR" "$SUPPORT_BASE" "$SOURCE_MEMORY_ROOT" "$OC_REPO" "$CUTIE_CKPT"; do
	[[ -e "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done
[[ "$(basename -- "${VIDEO_ROOT%/}")" == video_hard ]] || {
	echo "VIDEO_ROOT must be the frozen video_hard directory." >&2; exit 2;
}
for path in \
	tdmpc2/config.yaml \
	tdmpc2/common/buffer.py \
	tdmpc2/common/cutie_object_belief.py \
	tdmpc2/common/world_model.py \
	tdmpc2/tdmpc2.py \
	tdmpc2/check_cutie_learned_belief_contract.py \
	tdmpc2/check_cutie_learned_belief_update.py \
	tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py \
	tdmpc2/tools/evaluate_cutie_multitask_policy_burst.py \
	tdmpc2/tools/check_cutie_episode_reset_isolation.py; do
	[[ -f "$path" ]] || { echo "Missing repository file: $path" >&2; exit 2; }
done

source_root() {
	local task=$1 arm=$2
	printf '%s/logs/%s/7/cutie_object_%s100k_cutie_object_memory_probe_100k_v1_seed7_%s' \
		"$REPO_ROOT" "$task" "$arm" "${task//-/_}"
}

experiment_name() {
	local task=$1
	printf 'cutie_object_learned_belief100k_%s_%s' "$RUN_TAG" "${task//-/_}"
}

run_root() {
	local task=$1
	printf '%s/logs/%s/7/%s' "$REPO_ROOT" "$task" "$(experiment_name "$task")"
}

for path in "$BASE" "$STAGE"; do
	[[ ! -e "$path" ]] || { echo "Refusing existing runner output: $path" >&2; exit 3; }
done
for task in "${TASKS[@]}"; do
	[[ -f "$SUPPORT_BASE/$task/annotations.json" ]] || {
		echo "Missing support pack for $task" >&2; exit 2;
	}
	for arm in hard_zero last_valid; do
		root="$(source_root "$task" "$arm")"
		[[ -f "$root/runtime_config.json" && -f "$root/models/final.pt" ]] || {
			echo "Missing immutable source arm: $root" >&2; exit 2;
		}
	done
	[[ ! -e "$(run_root "$task")" ]] || {
		echo "Refusing existing learned training root: $(run_root "$task")" >&2; exit 3;
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
	local rc=$? pid failed task root archive_dir
	trap - EXIT INT TERM
	for pid in "${ACTIVE_PIDS[@]:-}"; do
		[[ -n "$pid" ]] && terminate_tree "$pid"
	done
	for pid in "${ACTIVE_PIDS[@]:-}"; do
		[[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
	done
	if (( PROMOTED == 0 )) && [[ -d "$STAGE" ]]; then
		if [[ ! -f "$SUMMARY" ]]; then
			"$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps({"format":sys.argv[3],"status":"learned_belief_seed7_engineering_fail","runner_exit_code":int(sys.argv[2]),"failure":"runner exited before complete aggregation"},indent=2)+"\n",encoding="utf-8")' "$SUMMARY" "$rc" "$FORMAT" 2>/dev/null || true
		fi
		# Training roots live outside STAGE so Hydra can preserve the established
		# log layout.  On failure, move only these two exact attempt-scoped roots
		# into the archive; this both preserves the evidence and makes v4 rerunnable.
		for task in "${TASKS[@]}"; do
			root="$(run_root "$task")"
			if [[ -e "$root" ]]; then
				archive_dir="$STAGE/training_roots/$task"
				mkdir -p "$archive_dir"
				if mv -- "$root" "$archive_dir/run"; then
					echo "ARCHIVED_FAILED_TRAINING_ROOT task=$task relative=training_roots/$task/run" >&2
				else
					echo "WARNING_FAILED_TO_ARCHIVE_TRAINING_ROOT task=$task path=$root" >&2
				fi
			fi
		done
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
		[[ ! -e "$failed" ]] || failed="${failed}.${RANDOM}"
		mv -- "$STAGE" "$failed"
		echo "CUTIE_LEARNED_BELIEF_SEED7_FAILED_ARCHIVE=$failed" >&2
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

echo "[1/6] Static and dependency-light contracts"
bash -n "$0"
run_contract learned_belief "$PY" -m tdmpc2.check_cutie_learned_belief_contract
run_contract object_wrapper "$PY" tdmpc2/check_cutie_object_wrapper_contract.py
run_contract object_integration "$PY" tdmpc2/check_cutie_object_only_integration_contract.py
run_contract policy_burst "$PY" -m tdmpc2.check_cutie_policy_burst_contract

echo "[2/6] Bind immutable seed-7 sources and implementation"
"$PY" - "$REPO_ROOT" "$SOURCE_MEMORY_ROOT" "$SUPPORT_BASE" "$VIDEO_ROOT" \
	"$MANIFEST_DIR" "$OC_REPO" "$CUTIE_CKPT" "$STAGE/provenance/inputs.json" <<'PY'
import hashlib, json, sys
from pathlib import Path

repo, source, support_base, video_root, manifests, oc_repo, cutie, output = map(Path, sys.argv[1:9])
tasks = ('reacher-visual-small', 'cartpole-swingup')
roles = {
    'reacher-visual-small': ['whole_arm', 'goal'],
    'cartpole-swingup': ['cart', 'pole'],
}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'JSON root is not an object: {path}')
    return value

def file_map(root, suffixes=None):
    result = {}
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        if suffixes is not None and path.suffix.lower() not in suffixes:
            continue
        relative = path.relative_to(root).as_posix()
        result[relative] = {'path': str(path.resolve()), 'sha256': digest(path)}
    if not result:
        raise RuntimeError(f'No immutable files found under {root}')
    return result

def metadata_inventory(root):
    rows = []
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        stat = path.stat()
        rows.append(f'{path.relative_to(root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n')
    if not rows:
        raise RuntimeError(f'Video inventory is empty: {root}')
    encoded = ''.join(rows).encode('utf-8')
    return {'files': len(rows), 'sha256': hashlib.sha256(encoded).hexdigest()}

def config_tree_digest(root):
    files = sorted(p for p in root.rglob('*.yaml') if p.is_file())
    files += sorted(p for p in root.rglob('*.yml') if p.is_file())
    h = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode('utf-8')
        data = path.read_bytes()
        h.update(len(relative).to_bytes(8, 'little')); h.update(relative)
        h.update(len(data).to_bytes(8, 'little')); h.update(data)
    return h.hexdigest()

summary_path = source / 'memory_probe_summary.json'
summary = load(summary_path)
if summary.get('status') != 'object_memory_probe_engineering_pass':
    raise RuntimeError(f'Source memory probe is not engineering pass: {summary.get("status")}')
gates = summary.get('engineering_gates')
if not isinstance(gates, dict) or not gates or not all(gates.values()):
    raise RuntimeError(f'Source engineering gates are not all true: {gates}')

implementation_rel = (
    'tdmpc2/config.yaml',
    'tdmpc2/common/buffer.py',
    'tdmpc2/common/cutie_object_belief.py',
    'tdmpc2/common/world_model.py',
    'tdmpc2/tdmpc2.py',
    'tdmpc2/trainer/online_trainer.py',
    'tdmpc2/check_cutie_learned_belief_contract.py',
    'tdmpc2/check_cutie_learned_belief_update.py',
    'tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py',
    'tdmpc2/tools/evaluate_cutie_multitask_policy_burst.py',
    'tdmpc2/tools/check_cutie_episode_reset_isolation.py',
    'tdmpc2/tools/run_cutie_learned_belief_seed7_pilot.sh',
)
implementation = {}
for relative in implementation_rel:
    path = repo / relative
    implementation[relative] = {'path': str(path.resolve()), 'sha256': digest(path)}

sources = {}
for task in tasks:
    sources[task] = {'arms': {}}
    for arm in ('hard_zero', 'last_valid'):
        root = repo / 'logs' / task / '7' / (
            f'cutie_object_{arm}100k_cutie_object_memory_probe_100k_v1_seed7_'
            + task.replace('-', '_')
        )
        runtime = root / 'runtime_config.json'
        checkpoint = root / 'models' / 'final.pt'
        normal = source / 'tasks' / task / 'evaluations' / 'normal' / f'{arm}.json'
        burst20 = source / 'tasks' / task / 'evaluations' / 'burst_20' / f'{arm}.json'
        for path in (runtime, checkpoint, normal, burst20):
            if not path.is_file():
                raise FileNotFoundError(path)
        sources[task]['arms'][arm] = {
            'root': str(root.resolve()),
            'runtime_config': {'path': str(runtime.resolve()), 'sha256': digest(runtime)},
            'checkpoint': {'path': str(checkpoint.resolve()), 'sha256': digest(checkpoint)},
            'normal': {'path': str(normal.resolve()), 'sha256': digest(normal)},
            'burst_20': {'path': str(burst20.resolve()), 'sha256': digest(burst20)},
        }
    plan = source / 'plans' / task / 'burst_20.json'
    support = support_base / task / 'annotations.json'
    for path in (plan, support):
        if not path.is_file():
            raise FileNotFoundError(path)
    plan_payload = load(plan)
    if plan_payload.get('task') != task or plan_payload.get('roles') != roles[task]:
        raise RuntimeError(f'Source burst plan task/roles mismatch: {task}')
    if [event.get('length') for event in plan_payload.get('events', [])] != [20] * 20:
        raise RuntimeError(f'Source burst plan is not exact 20x20: {task}')
    sources[task]['plan_burst_20'] = {'path': str(plan.resolve()), 'sha256': digest(plan)}
    sources[task]['support'] = {'path': str(support.resolve()), 'sha256': digest(support)}
    sources[task]['support_files'] = file_map(support.parent)

cutie_root = oc_repo / 'feature_extractor' / 'cutie' / 'cutie'
cutie_source_paths = {
    'inference_core': cutie_root / 'inference' / 'inference_core.py',
    'object_manager': cutie_root / 'inference' / 'object_manager.py',
    'image_feature_store': cutie_root / 'inference' / 'image_feature_store.py',
    'kv_memory_store': cutie_root / 'inference' / 'kv_memory_store.py',
    'memory_manager': cutie_root / 'inference' / 'memory_manager.py',
    'object_transformer': cutie_root / 'model' / 'transformer' / 'object_transformer.py',
    'cutie_model': cutie_root / 'model' / 'cutie.py',
}
for path in cutie_source_paths.values():
    if not path.is_file():
        raise FileNotFoundError(path)
cutie_sources = {
    name: {'path': str(path.resolve()), 'sha256': digest(path)}
    for name, path in cutie_source_paths.items()
}
cutie_config_root = cutie_root / 'config'
cutie_config_files = file_map(cutie_config_root, {'.yaml', '.yml'})
cutie_config_tree_sha256 = config_tree_digest(cutie_config_root)
manifest_files = file_map(manifests)
video_inventory = metadata_inventory(video_root)

payload = {
    'format': 'cutie_learned_belief_seed7_inputs_v4',
    'scientific_scope': (
        'single-training-seed development disaster gate; simulator-derived oracle '
        'support; synthetic policy-input burst; not a paper result'
    ),
    'source_memory_summary': {'path': str(summary_path.resolve()), 'sha256': digest(summary_path)},
    'sources': sources,
    'implementation': implementation,
    'external': {
        'video_root': {
            'path': str(video_root.resolve()),
            'inventory_files': video_inventory['files'],
            'inventory_sha256': video_inventory['sha256'],
        },
        'manifest_dir': {
            'path': str(manifests.resolve()),
            'files': manifest_files,
        },
        'oc_repo': {
            'path': str(oc_repo.resolve()),
            'cutie_sources': cutie_sources,
            'config_files': cutie_config_files,
            'config_tree_sha256': cutie_config_tree_sha256,
        },
        'cutie_checkpoint': {'path': str(cutie.resolve()), 'sha256': digest(cutie)},
    },
}
with output.open('x', encoding='utf-8', newline='\n') as f:
    json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write('\n')
print('CUTIE_LEARNED_BELIEF_INPUTS_OK', json.dumps({'output': str(output), 'tasks': tasks}))
PY

run_gpu_contract() {
	local label=$1 gpu=$2 mode=$3 task=$4 log="$STAGE/contracts/${1}.log" rc
	local -a extra=()
	[[ "$mode" == compile ]] && extra+=(--compile)
	echo "GPU_CONTRACT_START label=$label gpu=$gpu mode=$mode task=$task"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" tdmpc2/check_cutie_learned_belief_update.py \
		--task "$task" "${extra[@]}" >"$log" 2>&1
	rc=$?
	set -e
	write_rc "$STAGE/contracts/${label}.rc" "$rc"
	echo "GPU_CONTRACT_END label=$label gpu=$gpu task=$task rc=$rc"
	return "$rc"
}

echo "[3/6] Real replay, eager-update, compile-planner GPU contracts"
set +e
run_gpu_contract eager_gpu_reacher "$GPU_REACHER" eager reacher-visual-small & P0=$!; ACTIVE_PIDS+=("$P0")
run_gpu_contract compile_gpu_cartpole "$GPU_CARTPOLE" compile cartpole-swingup & P1=$!; ACTIVE_PIDS+=("$P1")
wait "$P0"; RC0=$?
wait "$P1"; RC1=$?
set -e
ACTIVE_PIDS=()
(( RC0 == 0 && RC1 == 0 )) || {
	echo "Learned-belief GPU contracts failed: $RC0 $RC1" >&2; exit 4;
}

write_preflight_config() {
	local task=$1 output=$2 source_runtime
	source_runtime="$(source_root "$task" hard_zero)/runtime_config.json"
	"$PY" - "$source_runtime" "$output" <<'PY'
import json, sys
from pathlib import Path
source, output = map(Path, sys.argv[1:3])
payload = json.loads(source.read_text(encoding='utf-8'))
payload['cutie_object_last_valid_memory'] = False
payload['cutie_object_policy_burst_plan'] = None
payload['cutie_object_belief_enabled'] = True
with output.open('x', encoding='utf-8', newline='\n') as f:
    json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write('\n')
PY
}

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
	log="$dir/learned_belief.train.log"
	hydra="$dir/hydra_learned_belief"
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
		cutie_object_belief_batch_size=128 cutie_object_belief_burn_in=3
		cutie_object_belief_min_burst=5 cutie_object_belief_max_burst=20
		cutie_object_belief_recovery_frames=1 cutie_object_belief_update_frequency=4
		cutie_object_belief_lr=0.0003 cutie_object_belief_loss_coef=1.0
		cutie_object_belief_reacquisition_coef=1.0
		cutie_object_belief_mask_seed_offset=104729
		cutie_object_belief_replay_seed_offset=130363
		"exp_name=$exp" "hydra.run.dir=$hydra" hydra.job.chdir=false
	)
	echo "TRAIN_START task=$task arm=learned_belief gpu=$gpu root=$root" | tee "$log"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" tdmpc2/train.py "${args[@]}" >>"$log" 2>&1
	rc=$?
	set -e
	write_rc "$dir/learned_belief.train.rc" "$rc"
	echo "TRAIN_END task=$task arm=learned_belief gpu=$gpu rc=$rc" | tee -a "$log"
}

run_normal_eval() {
	local gpu=$1 task=$2 dir=$3 root outdir out log rc
	root="$(run_root "$task")"; outdir="$dir/evaluations/normal"; mkdir -p "$outdir"
	out="$outdir/learned_belief.json"; log="$outdir/learned_belief.log"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" -m tdmpc2.tools.evaluate_cutie_multitask_checkpoint \
		--task "$task" --backend cutie_object_only \
		--runtime-config "$root/runtime_config.json" --checkpoint "$root/models/final.pt" \
		--training-seed "$SEED" --expected-training-steps "$STEPS" \
		--expected-training-eval-freq "$EVAL_FREQ" \
		--expected-training-eval-episodes "$EVAL_EPISODES" --episodes "$HELDOUT_EPISODES" \
		--env-seed "$ENV_SEED" --background-seed "$BACKGROUND_SEED" \
		--planner-seed-base "$PLANNER_SEED_BASE" --erosion-pixels 0 --output "$out" >"$log" 2>&1
	rc=$?
	set -e
	write_rc "$outdir/learned_belief.rc" "$rc"
	echo "EVAL_END task=$task condition=normal rc=$rc" | tee -a "$log"
}

run_burst20_eval() {
	local gpu=$1 task=$2 dir=$3 root plan role outdir out log rc
	root="$(run_root "$task")"; plan="$SOURCE_MEMORY_ROOT/plans/$task/burst_20.json"
	case "$task" in reacher-visual-small) role=whole_arm;; cartpole-swingup) role=pole;; esac
	outdir="$dir/evaluations/burst_20"; mkdir -p "$outdir"
	out="$outdir/learned_belief.json"; log="$outdir/learned_belief.log"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" -m tdmpc2.tools.evaluate_cutie_multitask_policy_burst \
		--task "$task" --arm learned_belief --policy-burst-plan "$plan" \
		--expected-role "$role" --expected-length 20 \
		--runtime-config "$root/runtime_config.json" --checkpoint "$root/models/final.pt" \
		--training-seed "$SEED" --expected-training-steps "$STEPS" \
		--expected-training-eval-freq "$EVAL_FREQ" \
		--expected-training-eval-episodes "$EVAL_EPISODES" \
		--env-seed "$ENV_SEED" --background-seed "$BACKGROUND_SEED" \
		--planner-seed-base "$PLANNER_SEED_BASE" --output "$out" >"$log" 2>&1
	rc=$?
	set -e
	write_rc "$outdir/learned_belief.rc" "$rc"
	echo "EVAL_END task=$task condition=burst_20 rc=$rc" | tee -a "$log"
}

run_task() {
	local gpu=$1 task=$2 dir="$STAGE/tasks/$2" preflight rc
	mkdir -p "$dir/preflight" "$dir/evaluations/normal" "$dir/evaluations/burst_20"
	printf '%s\n' "$gpu" >"$dir/gpu"
	preflight="$dir/preflight/runtime_config.json"
	write_preflight_config "$task" "$preflight"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" -m tdmpc2.tools.check_cutie_episode_reset_isolation \
		--runtime-config "$preflight" --output "$dir/preflight/reset.json" \
		--pollution-length 32 >"$dir/preflight/reset.log" 2>&1
	rc=$?
	set -e
	write_rc "$dir/preflight/reset.rc" "$rc"
	if (( rc != 0 )); then
		write_rc "$dir/learned_belief.train.rc" 125
		write_rc "$dir/evaluations/normal/learned_belief.rc" 125
		write_rc "$dir/evaluations/burst_20/learned_belief.rc" 125
		return 0
	fi
	run_training "$gpu" "$task" "$dir"
	if [[ "$(<"$dir/learned_belief.train.rc")" == 0 ]]; then
		run_normal_eval "$gpu" "$task" "$dir"
		run_burst20_eval "$gpu" "$task" "$dir"
	else
		write_rc "$dir/evaluations/normal/learned_belief.rc" 125
		write_rc "$dir/evaluations/burst_20/learned_belief.rc" 125
	fi
}

echo "[4/6] Two-task seed-7 learned-belief training and evaluation"
run_task "$GPU_REACHER" reacher-visual-small & PR=$!; ACTIVE_PIDS+=("$PR")
run_task "$GPU_CARTPOLE" cartpole-swingup & PC=$!; ACTIVE_PIDS+=("$PC")
set +e
wait "$PR"; RCR=$?
wait "$PC"; RCC=$?
set -e
ACTIVE_PIDS=()
write_rc "$STAGE/reacher_worker.rc" "$RCR"
write_rc "$STAGE/cartpole_worker.rc" "$RCC"

echo "[5/6] Strict aggregation and seed-7 disaster gate"
set +e
"$PY" - "$STAGE" "$SUMMARY" "$REPO_ROOT" "$SOURCE_MEMORY_ROOT" \
	"$START_SECONDS" "$GPU_REACHER" "$GPU_CARTPOLE" "$CUTIE_CKPT" <<'PY'
import csv, hashlib, json, math, statistics, sys, time
from pathlib import Path
import torch

stage, summary_path, repo, source = map(Path, sys.argv[1:5])
started = int(sys.argv[5]); gpu_map = {'reacher-visual-small': sys.argv[6], 'cartpole-swingup': sys.argv[7]}
cutie_checkpoint = Path(sys.argv[8])
tasks = ('reacher-visual-small', 'cartpole-swingup')
roles = {'reacher-visual-small': ['whole_arm', 'goal'], 'cartpole-swingup': ['cart', 'pole']}

def load(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict): raise ValueError(f'JSON root is not object: {path}')
    return value
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def rc(path):
    try: return int(path.read_text().strip())
    except Exception: return None
def same_path(a,b):
    try: return Path(a).resolve()==Path(b).resolve()
    except Exception: return False
def verify_file_map(items, label):
    failures=[]
    if not isinstance(items,dict) or not items:
        return [f'{label} file map is empty']
    for relative,item in items.items():
        try:
            path=Path(item['path'])
            if not path.is_file() or digest(path)!=item['sha256']:
                failures.append(f'{label} changed: {relative}')
        except Exception as exc:
            failures.append(f'{label} recheck failed for {relative}: {exc}')
    return failures
def metadata_inventory(root):
    rows=[]
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        stat=path.stat()
        rows.append(f'{path.relative_to(root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n')
    encoded=''.join(rows).encode('utf-8')
    return len(rows),hashlib.sha256(encoded).hexdigest()
def rewards(payload):
    values=[float(row['reward']) for row in payload.get('episodes',[])]
    if len(values)!=20 or not all(math.isfinite(v) for v in values):
        raise ValueError('Expected 20 finite episode rewards')
    return values
def paired(left,right):
    delta=[b-a for a,b in zip(left,right)]
    return {'left_mean':statistics.fmean(left),'right_mean':statistics.fmean(right),
            'right_minus_left_mean':statistics.fmean(delta),'median_delta':statistics.median(delta),
            'win_tie_loss':[sum(x>0 for x in delta),sum(x==0 for x in delta),sum(x<0 for x in delta)],
            'paired_deltas':delta}

inputs_path=stage/'provenance'/'inputs.json'; inputs=load(inputs_path)
engineering=[]; reports={}
try:
    if digest(cutie_checkpoint)!=inputs['external']['cutie_checkpoint']['sha256']:
        engineering.append('external Cutie checkpoint changed')
    if digest(source/'memory_probe_summary.json')!=inputs['source_memory_summary']['sha256']:
        engineering.append('source memory summary changed')
    for relative,item in inputs['implementation'].items():
        path=repo/relative
        if not same_path(item['path'],path) or digest(path)!=item['sha256']:
            engineering.append(f'implementation changed: {relative}')
    external=inputs['external']
    engineering.extend(verify_file_map(
        external['manifest_dir']['files'], 'background manifest'
    ))
    engineering.extend(verify_file_map(
        external['oc_repo']['cutie_sources'], 'Cutie source'
    ))
    engineering.extend(verify_file_map(
        external['oc_repo']['config_files'], 'Cutie config'
    ))
    video_root=Path(external['video_root']['path'])
    video_count,video_sha=metadata_inventory(video_root)
    if (
        video_count!=external['video_root']['inventory_files']
        or video_sha!=external['video_root']['inventory_sha256']
    ):
        engineering.append('video_hard inventory changed during the run')
except Exception as exc:
    engineering.append(f'immutable recheck failed: {exc}')

pair_normal=('initial_rgb_sha256','initial_object_sha256','background_source',
             'background_start_frame_index','planner_seed','planner_rng_start_sha256',
             'planner_rng_end_sha256','length')
pair_burst=('initial_rgb_sha256','initial_policy_object_sha256','initial_raw_object_frame_sha256',
            'background_source','background_start_frame_index','background_end_source',
            'background_end_frame_index','planner_seed','planner_rng_start_sha256',
            'planner_rng_end_sha256','length','policy_burst_event','policy_burst_plan_sha256')

for task in tasks:
    directory=stage/'tasks'/task
    new_root=repo/'logs'/task/'7'/f'cutie_object_learned_belief100k_cutie_learned_belief_seed7_pilot_v4_{task.replace("-","_")}'
    codes={
        'reset':rc(directory/'preflight'/'reset.rc'),
        'train':rc(directory/'learned_belief.train.rc'),
        'normal':rc(directory/'evaluations'/'normal'/'learned_belief.rc'),
        'burst_20':rc(directory/'evaluations'/'burst_20'/'learned_belief.rc'),
    }
    if any(v!=0 for v in codes.values()): engineering.append(f'{task} job rc: {codes}')
    required={
        'runtime':new_root/'runtime_config.json','checkpoint':new_root/'models'/'final.pt',
        'curve':new_root/'eval.csv','trainer':new_root/'trainer_runtime.json',
        'replay':new_root/'replay_runtime.json','perception':new_root/'perception_runtime.json',
        'reset':directory/'preflight'/'reset.json',
        'normal':directory/'evaluations'/'normal'/'learned_belief.json',
        'burst':directory/'evaluations'/'burst_20'/'learned_belief.json',
    }
    missing=[str(p) for p in required.values() if not p.is_file()]
    report={'gpu':gpu_map[task],'job_return_codes':codes,'new_root':str(new_root),'missing':missing}
    reports[task]=report
    if missing:
        engineering.append(f'{task} missing artifacts: {missing}'); continue
    try:
        runtime=load(required['runtime']); trainer=load(required['trainer']); replay=load(required['replay'])
        perception=load(required['perception']); reset=load(required['reset'])
        normal=load(required['normal']); burst=load(required['burst'])
        payload=torch.load(required['checkpoint'],map_location='cpu',weights_only=False)
        contract=payload.get('checkpoint_contract') if isinstance(payload,dict) else None
        supervision=(
            contract.get('cutie_object_belief_supervision')
            if isinstance(contract,dict) else None
        )
        def count_list(value):
            return (
                isinstance(value,list) and len(value)==2
                and all(isinstance(item,int) and not isinstance(item,bool) and item>=0
                        for item in value)
            )
        integer_supervision_fields=(
            'attempts','successful_updates','no_teacher_skips',
            'age20_available_updates','reacquisition_available_updates',
        )
        supervision_counts_ok=(
            isinstance(supervision,dict)
            and supervision.get('format')=='cutie_object_belief_supervision_v1'
            and all(isinstance(supervision.get(key),int)
                    and not isinstance(supervision.get(key),bool)
                    and supervision.get(key)>=0
                    for key in integer_supervision_fields)
            and count_list(supervision.get('age20_teacher_roles'))
            and count_list(supervision.get('reacquisition_teacher_roles'))
        )
        if supervision_counts_ok:
            attempts=supervision['attempts']
            successes=supervision['successful_updates']
            skips=supervision['no_teacher_skips']
            age20_roles=supervision['age20_teacher_roles']
            reacquisition_roles=supervision['reacquisition_teacher_roles']
            supervision_counts_ok=(
                successes+skips==attempts
                and successes==contract.get('cutie_object_belief_aux_updates')
                and supervision['age20_available_updates']<=attempts
                and supervision['reacquisition_available_updates']<=attempts
            )
        else:
            attempts=successes=skips=0
            age20_roles=reacquisition_roles=[0,0]
        with required['curve'].open(encoding='utf-8',newline='') as f: curve=list(csv.DictReader(f))
        checks={
            'runtime_belief': runtime.get('cutie_object_belief_enabled') is True,
            'runtime_memory_off': runtime.get('cutie_object_last_valid_memory') is False,
            'runtime_burst_off': runtime.get('cutie_object_policy_burst_plan') is None,
            'runtime_protocol': runtime.get('steps')==100000 and runtime.get('seed')==7 and runtime.get('eval_freq')==20000 and runtime.get('eval_episodes')==3,
            'runtime_object_only': runtime.get('flat_anchor_mode')=='cutie_object_only' and runtime.get('obs_shape')=={'object':[2,1770]} and runtime.get('latent_dim')==128,
            'belief_hparams': runtime.get('cutie_object_belief_batch_size')==128 and runtime.get('cutie_object_belief_burn_in')==3 and runtime.get('cutie_object_belief_min_burst')==5 and runtime.get('cutie_object_belief_max_burst')==20 and runtime.get('cutie_object_belief_recovery_frames')==1 and runtime.get('cutie_object_belief_update_frequency')==4,
            'checkpoint_contract': isinstance(contract,dict) and contract.get('cutie_object_belief_enabled') is True and isinstance(contract.get('cutie_object_belief_aux_updates'),int) and contract.get('cutie_object_belief_aux_updates',0)>1000,
            'checkpoint_schema': isinstance(contract,dict) and contract.get('cutie_object_belief_schema')=={'num_roles':2,'frame_dim':590,'stack_frames':3,'input_dim':1770,'role_dim':64},
            'checkpoint_supervision_counts': supervision_counts_ok and attempts>1000 and successes>1000,
            'checkpoint_age20_coverage': supervision_counts_ok and min(age20_roles)>0 and sum(age20_roles)/max(attempts,1)>=1.0,
            'checkpoint_reacquisition_coverage': supervision_counts_ok and min(reacquisition_roles)>0,
            'checkpoint_no_teacher_skip_rate': supervision_counts_ok and skips/max(attempts,1)<=0.01,
            'replay': replay.get('observation_keys')==['object'] and replay.get('storage_device')=='cuda:0' and replay.get('belief_sequence_length')==24 and replay.get('belief_batch_size')==128 and replay.get('belief_replay_rng_isolated') is True,
            'trainer': trainer.get('steps')==100000 and isinstance(trainer.get('training_non_eval_steps_per_second'),(int,float)) and trainer.get('training_non_eval_steps_per_second')>0,
            'curve': [int(float(row['step'])) for row in curve]==[0,20000,40000,60000,80000,100000],
            'perception_process': perception.get('worker_restarts')==0 and perception.get('timeouts')==0 and perception.get('frames')==109218,
            'reset': reset.get('pass') is True,
            'normal_format': normal.get('format')=='cutie_multitask_checkpoint_evaluation_v1',
            'burst_format': burst.get('format')=='cutie_multitask_policy_burst_evaluation_v1' and burst.get('arm')=='learned_belief',
            'normal_identity': normal.get('task')==task and normal.get('backend')=='cutie_object_only' and normal.get('training_seed')==7 and normal.get('evaluation',{}).get('episodes')==20,
            'burst_identity': burst.get('task')==task and burst.get('backend')=='cutie_object_only' and burst.get('training_seed')==7 and burst.get('evaluation',{}).get('episodes')==20,
            'belief_runtime': burst.get('learned_belief_runtime',{}).get('enabled') is True and burst.get('learned_belief_runtime',{}).get('loaded_aux_updates')==contract.get('cutie_object_belief_aux_updates') and burst.get('learned_belief_runtime',{}).get('prior_role_uses',0)>=400,
            'gpu': normal.get('provenance',{}).get('cuda_visible_devices')==gpu_map[task] and burst.get('provenance',{}).get('cuda_visible_devices')==gpu_map[task],
            'runtime_inputs': same_path(runtime.get('cutie_object_checkpoint'),inputs['external']['cutie_checkpoint']['path']) and digest(Path(runtime['cutie_object_checkpoint']))==inputs['external']['cutie_checkpoint']['sha256'] and same_path(runtime.get('cutie_object_support_path'),inputs['sources'][task]['support']['path']) and digest(Path(runtime['cutie_object_support_path']))==inputs['sources'][task]['support']['sha256'] and runtime.get('cutie_object_role_names')==roles[task],
            'normal_provenance': normal.get('provenance',{}).get('runtime_config_sha256')==digest(required['runtime']) and normal.get('provenance',{}).get('checkpoint_sha256')==digest(required['checkpoint']) and normal.get('provenance',{}).get('evaluator_sha256')==inputs['implementation']['tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py']['sha256'],
            'burst_provenance': burst.get('provenance',{}).get('runtime_config_sha256')==digest(required['runtime']) and burst.get('provenance',{}).get('checkpoint_sha256')==digest(required['checkpoint']) and burst.get('provenance',{}).get('evaluator_sha256')==inputs['implementation']['tdmpc2/tools/evaluate_cutie_multitask_policy_burst.py']['sha256'] and burst.get('provenance',{}).get('base_evaluator_sha256')==inputs['implementation']['tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py']['sha256'],
            'eval_cutie_inputs': all(payload.get('provenance',{}).get('cutie_inputs',{}).get('checkpoint_sha256')==inputs['external']['cutie_checkpoint']['sha256'] and payload.get('provenance',{}).get('cutie_inputs',{}).get('support_sha256')==inputs['sources'][task]['support']['sha256'] and payload.get('provenance',{}).get('cutie_inputs',{}).get('roles')==roles[task] for payload in (normal,burst)),
        }
        failed=[k for k,v in checks.items() if not v]
        artifact_records={}
        for name,path in required.items():
            if name in {'runtime','checkpoint','curve','trainer','replay','perception'}:
                artifact_records[name]={'path':str(path.resolve()),'sha256':digest(path)}
            else:
                artifact_records[name]={
                    'relative_to_summary_root':path.relative_to(stage).as_posix(),
                    'sha256':digest(path),
                }
        report.update(checks=checks,failed_checks=failed,checkpoint_contract=contract,
                      trainer=trainer,replay=replay,perception_runtime=perception,
                      artifacts=artifact_records)
        if failed: engineering.append(f'{task} checks failed: {failed}')
        source_payload={}
        for arm in ('hard_zero','last_valid'):
            src=inputs['sources'][task]['arms'][arm]
            for key in ('runtime_config','checkpoint','normal','burst_20'):
                p=Path(src[key]['path'])
                if digest(p)!=src[key]['sha256']: engineering.append(f'{task}/{arm}/{key} source changed')
            source_payload[arm]={
                'normal':load(Path(src['normal']['path'])),
                'burst':load(Path(src['burst_20']['path'])),
            }
        for key in ('plan_burst_20','support'):
            item=inputs['sources'][task][key]
            if digest(Path(item['path']))!=item['sha256']:
                engineering.append(f'{task}/{key} source changed')
        engineering.extend(verify_file_map(
            inputs['sources'][task]['support_files'], f'{task} support asset'
        ))
        reset_provenance=reset.get('provenance',{})
        reset_sources=reset_provenance.get('cutie_sources',{})
        expected_sources=inputs['external']['oc_repo']['cutie_sources']
        reset_source_ok=(set(reset_sources)==set(expected_sources) and all(
            reset_sources[name].get('sha256')==expected_sources[name]['sha256']
            and same_path(reset_sources[name].get('path'),expected_sources[name]['path'])
            for name in expected_sources
        ))
        reset_ok=(
            reset_provenance.get('checkpoint_sha256')==inputs['external']['cutie_checkpoint']['sha256']
            and reset_provenance.get('support_sha256')==inputs['sources'][task]['support']['sha256']
            and reset_provenance.get('probe_file_sha256')==inputs['implementation']['tdmpc2/tools/check_cutie_episode_reset_isolation.py']['sha256']
            and reset_source_ok
            and reset_provenance.get('effective_config_tree_sha256')==inputs['external']['oc_repo']['config_tree_sha256']
        )
        report['reset_provenance_exact']=reset_ok
        if not reset_ok: engineering.append(f'{task} reset Cutie/support provenance mismatch')
        for condition,new_payload,fields in (('normal',normal,pair_normal),('burst_20',burst,pair_burst)):
            new_eps=new_payload['episodes']
            for arm in ('hard_zero','last_valid'):
                old=source_payload[arm]['normal' if condition=='normal' else 'burst']['episodes']
                mismatches={field:[i for i,(a,b) in enumerate(zip(old,new_eps)) if a.get(field)!=b.get(field)] for field in fields}
                bad={k:v for k,v in mismatches.items() if v}
                report.setdefault('pairing',{}).setdefault(condition,{})[arm]={'exact':not bad,'mismatches':bad}
                if bad: engineering.append(f'{task}/{condition}/{arm} pairing mismatch: {bad}')
        hn=rewards(source_payload['hard_zero']['normal']); ln=rewards(source_payload['last_valid']['normal']); nn=rewards(normal)
        hb=rewards(source_payload['hard_zero']['burst']); lb=rewards(source_payload['last_valid']['burst']); nb=rewards(burst)
        hard_normal=statistics.fmean(hn); learned_normal=statistics.fmean(nn)
        loss=max(0.0,hard_normal-statistics.fmean(hb))
        recovery=(statistics.fmean(nb)-statistics.fmean(hb))/loss if loss>0 else None
        report['rewards']={
            'normal':{'hard_zero':hard_normal,'last_valid':statistics.fmean(ln),'learned_belief':learned_normal,
                      'learned_vs_hard':paired(hn,nn),'learned_vs_last':paired(ln,nn)},
            'burst_20':{'hard_zero':statistics.fmean(hb),'last_valid':statistics.fmean(lb),'learned_belief':statistics.fmean(nb),
                        'learned_vs_hard':paired(hb,nb),'learned_vs_last':paired(lb,nb),
                        'hard_zero_drop_from_normal':loss,'learned_recovery_fraction':recovery,
                        'attribution_eligible':burst.get('policy_burst',{}).get('controlled_burst_attribution_eligible') is True},
        }
        report['disaster_gate']={
            'normal_retention_at_least_80pct': learned_normal>=0.8*hard_normal,
            'burst_reward_at_least_80pct_hard_normal': statistics.fmean(nb)>=0.8*hard_normal,
            'controlled_attribution_gate': (
                burst.get('policy_burst',{}).get('controlled_burst_attribution_eligible') is True
                if task=='reacher-visual-small' else True
            ),
            'learned_checkpoint_trained': all(checks[key] for key in (
                'checkpoint_contract','checkpoint_supervision_counts',
                'checkpoint_age20_coverage','checkpoint_reacquisition_coverage',
                'checkpoint_no_teacher_skip_rate',
            )),
        }
        report['burst_attribution_scope']={
            'required_for_disaster_gate': task=='reacher-visual-small',
            'eligible': burst.get('policy_burst',{}).get('controlled_burst_attribution_eligible') is True,
            'cartpole_reason_if_not_required': (
                'frozen source has below-80pct raw-valid overlap; use normal return '
                'for natural-missing behavior and burst return only as robustness stress'
                if task=='cartpole-swingup' else None
            ),
        }
    except Exception as exc:
        engineering.append(f'{task} aggregation exception: {type(exc).__name__}: {exc}')
        report['exception']=repr(exc)

engineering_pass=not engineering
disaster_pass=engineering_pass and all(all(v for v in report.get('disaster_gate',{}).values()) for report in reports.values())
payload={
    'format':'cutie_learned_belief_seed7_pilot_v4',
    'status':'learned_belief_seed7_engineering_pass' if engineering_pass else 'learned_belief_seed7_engineering_fail',
    'scientific_scope':'single-seed oracle-support synthetic-burst development gate; not a paper result',
    'elapsed_hours':(time.time()-started)/3600.0,
    'engineering_pass':engineering_pass,'engineering_failures':engineering,
    'seed7_disaster_gate_pass':disaster_pass,
    'recommendation':'go_build_seed8_seed9_core' if disaster_pass else ('stop_and_fix_engineering' if not engineering_pass else 'no_go_inspect_learned_belief_before_scaling'),
    'tasks':reports,
    'inputs':{
        'relative_to_summary_root':'provenance/inputs.json',
        'execution_path':str(inputs_path.resolve()),
        'execution_path_is_staging_only':True,
        'sha256':digest(inputs_path),
    },
    'summary_relative_paths_authoritative':True,
}
tmp=summary_path.with_suffix('.json.incomplete')
with tmp.open('x',encoding='utf-8',newline='\n') as f:
    json.dump(payload,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
tmp.replace(summary_path)
print('CUTIE_LEARNED_BELIEF_SEED7_SUMMARY',json.dumps({'status':payload['status'],'disaster_gate':disaster_pass,'recommendation':payload['recommendation'],'summary':str(summary_path)},ensure_ascii=False))
raise SystemExit(0 if engineering_pass else 4)
PY
AGG_RC=$?
set -e
(( AGG_RC == 0 )) || exit "$AGG_RC"

echo "[6/6] Promote immutable diagnostic"
mv -- "$STAGE" "$BASE"
PROMOTED=1
trap - EXIT INT TERM
echo "CUTIE_LEARNED_BELIEF_SEED7_PILOT_COMPLETE"
echo "SUMMARY=$BASE/learned_belief_seed7_summary.json"
