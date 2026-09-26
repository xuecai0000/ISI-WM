#!/usr/bin/env bash
# Partial, no-overwrite recovery for the seed-6 multitask pilot.
#
# This runner deliberately does not resume or rewrite the v1 experiment. It
# imports one explicitly named v1 failed archive as immutable evidence, fills
# only the two missing Acrobot Cutie arms with the v2 upper/lower-arm support
# representation, and re-evaluates the existing Cup/Cartpole checkpoints at
# normal (zero-erosion) validation. Tracker health is an outcome: an unhealthy
# tracker still produces a promoted diagnostic result. Only incomplete jobs,
# malformed artifacts, source mutation, or non-exact pairing fail the runner.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${SOURCE_V1:?Set SOURCE_V1 to the exact v1 failed archive directory}"
: "${VIDEO_ROOT:?Set VIDEO_ROOT to video_hard}"
: "${OC_REPO:?Set OC_REPO to the OC-STORM checkout}"
: "${CUTIE_CKPT:?Set CUTIE_CKPT to cutie-small-mega.pth}"

PY="${PY:-python}"
MANIFEST_DIR="${MANIFEST_DIR:-$REPO_ROOT/tdmpc2/envs/background_manifests}"
GPU_DIAGNOSTIC="${GPU_DIAGNOSTIC:-0}"
GPU_ACROBOT="${GPU_ACROBOT:-1}"

readonly SEED=6 SUPPORT_SEED=314159 STEPS=100000 EVAL_FREQ=20000
readonly EVAL_EPISODES=3 HELDOUT_EPISODES=20
readonly ENV_SEED=424243 BACKGROUND_SEED=1618034 PLANNER_SEED_BASE=8675400
readonly RUN_TAG=cutie_object_multitask_100k_recovery_v2
readonly FORMAT=cutie_object_multitask_recovery_v2
readonly BASE="$REPO_ROOT/logs/_generalization/${RUN_TAG}_seed${SEED}"
readonly STAGE="${BASE}.incomplete"
readonly SUMMARY="$STAGE/recovery_summary.json"
readonly SUPPORT_DIR="$REPO_ROOT/datasets/cutie_multitask_support_v2_seed${SUPPORT_SEED}/acrobot-swingup"
readonly -a DIAGNOSTIC_TASKS=(cup-catch cartpole-swingup)
readonly -a ARMS=(rgb cutie_hybrid cutie_object_only)
readonly -a CUTIE_ARMS=(cutie_hybrid cutie_object_only)

for name in GPU_DIAGNOSTIC GPU_ACROBOT; do
	value="${!name}"
	[[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || {
		echo "$name is invalid: $value" >&2; exit 2;
	}
done
[[ "$GPU_DIAGNOSTIC" != "$GPU_ACROBOT" ]] || {
	echo "GPU_DIAGNOSTIC and GPU_ACROBOT must differ." >&2; exit 2;
}
(( BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 1) )) || {
	echo "Bash >=5.1 is required." >&2; exit 2;
}
if [[ "$PY" == */* ]]; then
	[[ -x "$PY" ]] || { echo "Python is not executable: $PY" >&2; exit 2; }
else
	command -v "$PY" >/dev/null || { echo "Python is not on PATH: $PY" >&2; exit 2; }
fi
for path in "$SOURCE_V1" "$VIDEO_ROOT" "$MANIFEST_DIR" "$OC_REPO" "$CUTIE_CKPT"; do
	[[ -e "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done
SOURCE_V1="$(cd -- "$SOURCE_V1" && pwd)"
readonly SOURCE_V1
readonly SOURCE_SUMMARY="$SOURCE_V1/generalization_summary.json"
[[ -f "$SOURCE_SUMMARY" ]] || { echo "Missing v1 summary: $SOURCE_SUMMARY" >&2; exit 2; }
for path in \
	tdmpc2/tools/collect_cutie_multitask_support.py \
	tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py; do
	[[ -f "$path" ]] || { echo "Missing required tool: $path" >&2; exit 2; }
done

new_experiment_name() {
	local arm=$1
	printf '%s100k_%s_seed%s_acrobot_swingup' "$arm" "$RUN_TAG" "$SEED"
}

new_run_root() {
	local arm=$1
	printf '%s/logs/acrobot-swingup/%s/%s' "$REPO_ROOT" "$SEED" "$(new_experiment_name "$arm")"
}

source_experiment_name() {
	local task=$1 arm=$2
	printf '%s100k_cutie_object_multitask_100k_v1_seed%s_%s' \
		"$arm" "$SEED" "${task//-/_}"
}

source_run_root() {
	local task=$1 arm=$2
	printf '%s/logs/%s/%s/%s' "$REPO_ROOT" "$task" "$SEED" \
		"$(source_experiment_name "$task" "$arm")"
}

# Every output owned by v2 is new. Existing v1 inputs are intentionally not in
# this refusal list because they are imported read-only and hash-checked twice.
for path in "$BASE" "$STAGE" "$SUPPORT_DIR"; do
	[[ ! -e "$path" ]] || { echo "Refusing to overwrite v2 output: $path" >&2; exit 3; }
done
for arm in "${CUTIE_ARMS[@]}"; do
	path="$(new_run_root "$arm")"
	[[ ! -e "$path" ]] || { echo "Existing v2 run would be overwritten: $path" >&2; exit 3; }
done

mkdir -p "$STAGE/contracts" "$STAGE/imports" "$STAGE/tasks/acrobot-swingup/evaluations/erosion0"
for task in "${DIAGNOSTIC_TASKS[@]}"; do
	mkdir -p "$STAGE/tasks/$task/normal_recheck"
done
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
			"$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps({"format":sys.argv[3],"status":"recovery_engineering_fail","runner_exit_code":int(sys.argv[2]),"failure":"runner exited before complete aggregation"},indent=2)+"\n",encoding="utf-8")' "$SUMMARY" "$rc" "$FORMAT" 2>/dev/null || true
		fi
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
		[[ ! -e "$failed" ]] || failed="${failed}.${RANDOM}"
		mv -- "$STAGE" "$failed"
		echo "CUTIE_MULTITASK_RECOVERY_FAILED_ARCHIVE=$failed" >&2
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

echo "[1/6] Dependency-light contracts"
run_contract cutie_object_wrapper "$PY" tdmpc2/check_cutie_object_wrapper_contract.py
run_contract cutie_hybrid "$PY" tdmpc2/check_cutie_hybrid_contract.py
run_contract cutie_object_only "$PY" tdmpc2/check_cutie_object_only_contract.py
run_contract multitask_support "$PY" tdmpc2/check_cutie_multitask_support_contract.py
run_contract object_only_integration "$PY" tdmpc2/check_cutie_object_only_integration_contract.py
run_contract "object_compile_gpu${GPU_ACROBOT}" env CUDA_VISIBLE_DEVICES="$GPU_ACROBOT" \
	"$PY" tdmpc2/check_cutie_object_only_update.py --compile

echo "[2/6] Binding immutable v1 inputs"
"$PY" - "$REPO_ROOT" "$SOURCE_V1" "$STAGE/imports/source_v1.json" <<'PY'
import csv, hashlib, json, math, sys
from pathlib import Path

repo, source, output = map(Path, sys.argv[1:4])
summary_path = source / 'generalization_summary.json'

def load(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict): raise ValueError(path)
    return value

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def exp(task, arm):
    return f'{arm}100k_cutie_object_multitask_100k_v1_seed6_{task.replace("-", "_")}'

summary = load(summary_path)
protocol = summary.get('protocol', {})
if not source.name.startswith('cutie_object_multitask_100k_v1_seed6.failed.'):
    raise ValueError(f'Unexpected SOURCE_V1 directory name: {source.name!r}')
if summary.get('format') != 'cutie_object_multitask_generalization_v1':
    raise ValueError(f'Unexpected source format: {summary.get("format")!r}')
if summary.get('status') != 'generalization_engineering_fail':
    raise ValueError(f'SOURCE_V1 must be the frozen failed archive, got {summary.get("status")!r}')
if (protocol.get('training_seed'), protocol.get('steps'), protocol.get('eval_freq'),
        protocol.get('training_eval_episodes'), protocol.get('heldout_episodes')) != (6,100000,20000,3,20):
    raise ValueError(f'Unexpected source protocol: {protocol}')

files = {str(summary_path.resolve())}
roots = {}
for task in ('cup-catch', 'cartpole-swingup'):
    roots[task] = {}
    for arm in ('rgb','cutie_hybrid','cutie_object_only'):
        root = repo / 'logs' / task / '6' / exp(task, arm)
        roots[task][arm] = str(root.resolve())
        for path in (root/'runtime_config.json', root/'models'/'final.pt'):
            if not path.is_file(): raise FileNotFoundError(path)
            files.add(str(path.resolve()))
        evaluation = source/'tasks'/task/'evaluations'/'erosion0'/f'{arm}.json'
        if not evaluation.is_file(): raise FileNotFoundError(evaluation)
        files.add(str(evaluation.resolve()))
        raw = load(root/'runtime_config.json')
        if raw.get('task') != task or raw.get('seed') != 6 or raw.get('steps') != 100000:
            raise ValueError(f'Bad source runtime config: {root}')

task, arm = 'acrobot-swingup', 'rgb'
root = repo / 'logs' / task / '6' / exp(task, arm)
roots[task] = {arm: str(root.resolve())}
for path in (root/'runtime_config.json', root/'models'/'final.pt', root/'eval.csv'):
    if not path.is_file(): raise FileNotFoundError(path)
    files.add(str(path.resolve()))
evaluation = source/'tasks'/task/'evaluations'/'erosion0'/'rgb.json'
if not evaluation.is_file(): raise FileNotFoundError(evaluation)
files.add(str(evaluation.resolve()))
source_rgb_eval = load(evaluation)
current_evaluator = repo/'tdmpc2'/'tools'/'evaluate_cutie_multitask_checkpoint.py'
if source_rgb_eval.get('provenance',{}).get('evaluator_sha256') != sha(current_evaluator):
    raise ValueError('Source Acrobot RGB evaluator SHA does not match the current frozen evaluator')
files.add(str(current_evaluator.resolve()))
if (source_rgb_eval.get('task'), source_rgb_eval.get('backend'),
        source_rgb_eval.get('erosion_pixels'), len(source_rgb_eval.get('episodes',[]))) != (
        'acrobot-swingup','rgb',0,20):
    raise ValueError('Bad source Acrobot RGB evaluation envelope')
raw = load(root/'runtime_config.json')
if raw.get('task') != task or raw.get('seed') != 6 or raw.get('steps') != 100000:
    raise ValueError('Bad source Acrobot RGB runtime config')
with (root/'eval.csv').open(encoding='utf-8', newline='') as f:
    rows = list(csv.DictReader(f))
if [int(float(row['step'])) for row in rows] != [0,20000,40000,60000,80000,100000]:
    raise ValueError('Bad source Acrobot RGB training curve')
if not all(math.isfinite(float(row['episode_reward'])) for row in rows):
    raise ValueError('Non-finite source Acrobot RGB reward')

for task in ('cup-catch','cartpole-swingup'):
    for arm in ('cutie_hybrid','cutie_object_only'):
        raw = load(Path(roots[task][arm])/'runtime_config.json')
        support = Path(raw['cutie_object_support_path']).resolve()
        if not support.is_file(): raise FileNotFoundError(support)
        files.add(str(support))
        annotations=load(support)
        for record in annotations.get('records',[]):
            mask=(support.parent/record['indexed_mask']).resolve()
            if not mask.is_file(): raise FileNotFoundError(mask)
            files.add(str(mask))

payload = {
    'format':'cutie_object_multitask_recovery_source_v1',
    'source_root':str(source.resolve()), 'source_status':summary['status'],
    'source_summary_sha256':sha(summary_path), 'roots':roots,
    'files':[{'path':path,'sha256':sha(Path(path))} for path in sorted(files)],
}
with output.open('x',encoding='utf-8',newline='\n') as f:
    json.dump(payload,f,indent=2,allow_nan=False); f.write('\n')
print('SOURCE_V1_IMPORT_OK', json.dumps({'files':len(files),'output':str(output)}))
PY

read_support_roles() {
	"$PY" - "$1" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8'))
if d.get('format') != 'cutie_indexed_mask_support_v1': raise ValueError(d.get('format'))
if d.get('roles') != ['upper_arm','lower_arm']: raise ValueError(d.get('roles'))
c=d.get('collection',{})
if c.get('task') != 'acrobot-swingup' or c.get('support_schema') != 'generic_indexed_v1': raise ValueError(c)
if len(d.get('records',[])) != 6: raise ValueError('Expected six support records')
for row in d['records']:
    counts=row.get('role_pixel_counts',{})
    if counts.get('upper_arm',0) <= 0 or counts.get('lower_arm',0) <= 0: raise ValueError(counts)
    selected=row.get('selected_names',{})
    if any(item.get('object_type') == 'site' for values in selected.values() for item in values):
        raise ValueError('Acrobot v2 support may not contain sites')
print('upper_arm lower_arm')
PY
}

write_preflight_config() {
	"$PY" - "$1" "$2" "$OC_REPO" "$CUTIE_CKPT" <<'PY'
import json, sys
from pathlib import Path
out,support,repo,ckpt=map(Path,sys.argv[1:5])
d={'task':'acrobot-swingup','obs':'rgb','model_size':5,'flat_anchor':True,
 'flat_anchor_mode':'cutie_object_only','cutie_object_repo':str(repo.resolve()),
 'cutie_object_checkpoint':str(ckpt.resolve()),'cutie_object_support_path':str(support.resolve()),
 'cutie_object_support_schema':'generic_indexed_v1',
 'cutie_object_role_names':['upper_arm','lower_arm'],'cutie_object_allow_simulator_support':True,
 'cutie_object_config_dir':None,'cutie_object_device':'cuda:0',
 'cutie_object_tracker_height':448,'cutie_object_tracker_width':448,
 'cutie_object_model_size':'small','cutie_object_prompt_radius':2.0,
 'cutie_object_amp':True,'cutie_object_worker_timeout_seconds':180.0}
with out.open('x',encoding='utf-8',newline='\n') as f:
    json.dump(d,f,indent=2,allow_nan=False); f.write('\n')
PY
}

echo "[3/6] Acrobot v2 support and reset isolation"
ACROBOT_DIR="$STAGE/tasks/acrobot-swingup"
SUPPORT_RC=0
set +e
env CUDA_VISIBLE_DEVICES="$GPU_ACROBOT" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
	"$PY" -m tdmpc2.tools.collect_cutie_multitask_support \
	--task acrobot-swingup --video-root "$VIDEO_ROOT" --manifest-dir "$MANIFEST_DIR" \
	--output "$SUPPORT_DIR" --seed "$SUPPORT_SEED" >"$ACROBOT_DIR/support.log" 2>&1
SUPPORT_RC=$?
set -e
write_rc "$ACROBOT_DIR/support.rc" "$SUPPORT_RC"
PREFLIGHT_RC=125
if (( SUPPORT_RC == 0 )); then
	set +e
	roles="$(read_support_roles "$SUPPORT_DIR/annotations.json" 2>>"$ACROBOT_DIR/support.log")"
	ROLE_RC=$?
	set -e
	if (( ROLE_RC == 0 )) && [[ "$roles" == "upper_arm lower_arm" ]]; then
		printf 'upper_arm\nlower_arm\n' >"$ACROBOT_DIR/roles"
		write_preflight_config "$SUPPORT_DIR/preflight_runtime_config.json" "$SUPPORT_DIR/annotations.json"
		set +e
		env CUDA_VISIBLE_DEVICES="$GPU_ACROBOT" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
			"$PY" -m tdmpc2.tools.check_cutie_episode_reset_isolation \
			--runtime-config "$SUPPORT_DIR/preflight_runtime_config.json" \
			--output "$ACROBOT_DIR/reset_isolation.json" --pollution-length 32 \
			>"$ACROBOT_DIR/preflight.log" 2>&1
		PREFLIGHT_RC=$?
		set -e
	else
		if (( ROLE_RC != 0 )); then PREFLIGHT_RC="$ROLE_RC"; else PREFLIGHT_RC=65; fi
	fi
fi
write_rc "$ACROBOT_DIR/preflight.rc" "$PREFLIGHT_RC"

run_training() {
	local arm=$1 dir="$ACROBOT_DIR" flat=true mode exp root log hydra rc
	case "$arm" in
		cutie_hybrid) mode=cutie_hybrid ;;
		cutie_object_only) mode=cutie_object_only ;;
		*) return 2 ;;
	esac
	exp="$(new_experiment_name "$arm")"; root="$(new_run_root "$arm")"
	log="$dir/${arm}.train.log"; hydra="$dir/hydra_${arm}"
	local -a args=(
		task=acrobot-swingup obs=rgb model_size=5 "steps=$STEPS" "seed=$SEED"
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
		"cutie_object_support_path=$SUPPORT_DIR/annotations.json"
		cutie_object_support_schema=generic_indexed_v1
		cutie_object_role_names=[upper_arm,lower_arm] cutie_object_allow_simulator_support=true
		cutie_object_config_dir=null cutie_object_device=cuda:0
		cutie_object_tracker_height=448 cutie_object_tracker_width=448
		cutie_object_model_size=small cutie_object_prompt_radius=2.0 cutie_object_amp=true
		cutie_object_worker_timeout_seconds=180 cutie_object_num_roles=2
		cutie_object_frame_dim=590 cutie_object_stack_frames=3 cutie_object_input_dim=1770
		cutie_object_role_dim=64 cutie_object_hidden_dim=256 cutie_object_joint_dim=640
		cutie_object_only_latent_dim=128 "exp_name=$exp" "hydra.run.dir=$hydra"
		hydra.job.chdir=false
	)
	echo "TRAIN_START task=acrobot-swingup arm=$arm gpu=$GPU_ACROBOT root=$root" | tee "$log"
	set +e
	env CUDA_VISIBLE_DEVICES="$GPU_ACROBOT" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" tdmpc2/train.py "${args[@]}" >>"$log" 2>&1
	rc=$?
	set -e
	write_rc "$dir/${arm}.train.rc" "$rc"
	echo "TRAIN_END task=acrobot-swingup arm=$arm gpu=$GPU_ACROBOT rc=$rc" | tee -a "$log"
}

run_evaluation() {
	local gpu=$1 task=$2 arm=$3 root=$4 outdir=$5 out log rc
	mkdir -p "$outdir"
	out="$outdir/${arm}.json"; log="$outdir/${arm}.log"
	if [[ -e "$out" || -e "${out}.incomplete" ]]; then
		echo "Refusing existing evaluation output: $out" >&2; return 3
	fi
	if [[ ! -f "$root/runtime_config.json" || ! -f "$root/models/final.pt" ]]; then
		echo "Missing source artifacts: $root" >"$log"; write_rc "$outdir/${arm}.rc" 66; return 0
	fi
	echo "EVAL_START task=$task arm=$arm erosion=0 gpu=$gpu root=$root" | tee "$log"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" -m tdmpc2.tools.evaluate_cutie_multitask_checkpoint \
		--task "$task" --backend "$arm" --runtime-config "$root/runtime_config.json" \
		--checkpoint "$root/models/final.pt" --training-seed "$SEED" \
		--expected-training-steps "$STEPS" --expected-training-eval-freq "$EVAL_FREQ" \
		--expected-training-eval-episodes "$EVAL_EPISODES" --episodes "$HELDOUT_EPISODES" \
		--env-seed "$ENV_SEED" --background-seed "$BACKGROUND_SEED" \
		--planner-seed-base "$PLANNER_SEED_BASE" --erosion-pixels 0 --output "$out" \
		>>"$log" 2>&1
	rc=$?
	set -e
	write_rc "$outdir/${arm}.rc" "$rc"
	echo "EVAL_END task=$task arm=$arm erosion=0 gpu=$gpu rc=$rc" | tee -a "$log"
}

diagnostic_worker() {
	local task arm root outdir
	for task in "${DIAGNOSTIC_TASKS[@]}"; do
		outdir="$STAGE/tasks/$task/normal_recheck"
		for arm in "${ARMS[@]}"; do
			root="$(source_run_root "$task" "$arm")"
			run_evaluation "$GPU_DIAGNOSTIC" "$task" "$arm" "$root" "$outdir"
		done
	done
}

acrobot_worker() {
	local arm root outdir="$ACROBOT_DIR/evaluations/erosion0"
	if (( PREFLIGHT_RC != 0 )); then
		for arm in "${CUTIE_ARMS[@]}"; do
			echo "Skipped: Acrobot support/reset preflight rc=$PREFLIGHT_RC" >"$ACROBOT_DIR/${arm}.train.log"
			write_rc "$ACROBOT_DIR/${arm}.train.rc" 125
			write_rc "$outdir/${arm}.rc" 125
		done
		return 0
	fi
	for arm in "${CUTIE_ARMS[@]}"; do
		run_training "$arm"
		if [[ "$(<"$ACROBOT_DIR/${arm}.train.rc")" == 0 ]]; then
			root="$(new_run_root "$arm")"
			run_evaluation "$GPU_ACROBOT" acrobot-swingup "$arm" "$root" "$outdir"
		else
			write_rc "$outdir/${arm}.rc" 125
		fi
	done
}

echo "[4/6] Parallel partial recovery"
echo "GPU $GPU_ACROBOT: Acrobot Hybrid/ObjectOnly training and validation"
echo "GPU $GPU_DIAGNOSTIC: Cup/Cartpole old-checkpoint normal validation recheck"
diagnostic_worker & PID_DIAGNOSTIC=$!; ACTIVE_PIDS+=("$PID_DIAGNOSTIC")
acrobot_worker & PID_ACROBOT=$!; ACTIVE_PIDS+=("$PID_ACROBOT")
set +e
wait "$PID_DIAGNOSTIC"; DIAGNOSTIC_WORKER_RC=$?
wait "$PID_ACROBOT"; ACROBOT_WORKER_RC=$?
set -e
ACTIVE_PIDS=()
write_rc "$STAGE/diagnostic_worker.rc" "$DIAGNOSTIC_WORKER_RC"
write_rc "$STAGE/acrobot_worker.rc" "$ACROBOT_WORKER_RC"

echo "[5/6] Recovery aggregation"
set +e
"$PY" - "$STAGE" "$SUMMARY" "$REPO_ROOT" "$SOURCE_V1" "$SUPPORT_DIR" \
	"$GPU_DIAGNOSTIC" "$GPU_ACROBOT" "$START_SECONDS" "$OC_REPO" "$CUTIE_CKPT" <<'PY'
import csv, functools, hashlib, json, math, statistics, sys, time
from pathlib import Path

from PIL import Image

stage, summary_path, repo, source, support_dir = map(Path, sys.argv[1:6])
gpu_diag, gpu_acro, started = sys.argv[6], sys.argv[7], int(sys.argv[8])
expected_oc, expected_ckpt = map(Path, sys.argv[9:11])
arms = ('rgb','cutie_hybrid','cutie_object_only')
cutie_arms = ('cutie_hybrid','cutie_object_only')
pair_fields = ('initial_rgb_sha256','background_source','background_start_frame_index',
               'planner_seed','planner_rng_start_sha256','planner_rng_end_sha256','length')
expected_evaluator = repo/'tdmpc2'/'tools'/'evaluate_cutie_multitask_checkpoint.py'

def load(path):
    value=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value,dict): raise ValueError(path)
    return value
@functools.lru_cache(maxsize=None)
def sha(path):
    path=Path(path)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def rc(path):
    try:return int(path.read_text(encoding='utf-8').strip())
    except Exception:return None
def same_path(value, expected):
    try:return Path(value).resolve()==Path(expected).resolve()
    except Exception:return False
def positive(value):
    try:value=float(value)
    except Exception:return False
    return math.isfinite(value) and value>0
def role_diagnostics_gates(value, roles, expected_episodes):
    if not isinstance(value,dict): return False
    metrics=value.get('role_metrics'); episodes=value.get('episode_metrics')
    if value.get('role_diagnostics_schema')!='cutie_role_runtime_diagnostics_v1': return False
    if not isinstance(metrics,dict) or set(metrics)!=set(roles): return False
    required={'valid_frame_rate','lost_frame_rate','empty_mask_frame_rate',
      'nonfinite_feature_frame_rate','mask_touches_border_rate','mean_mask_area_pixels',
      'mean_confidence','mean_mask_score','max_invalid_burst','max_invalid_burst_event'}
    for role in roles:
        row=metrics[role]
        if not isinstance(row,dict) or not required.issubset(row): return False
        for key in required-{'max_invalid_burst_event'}:
            item=row.get(key)
            if not isinstance(item,(int,float)) or not math.isfinite(float(item)): return False
    if not isinstance(episodes,list) or len(episodes)!=expected_episodes: return False
    if [row.get('episode_index') for row in episodes]!=list(range(expected_episodes)): return False
    if sum(int(row.get('frames',-1)) for row in episodes)!=int(value.get('frames',-2)): return False
    return all(isinstance(row.get('per_role_invalid_frames'),dict)
               and set(row['per_role_invalid_frames'])==set(roles)
               and isinstance(row.get('per_role_max_invalid_burst'),dict)
               and set(row['per_role_max_invalid_burst'])==set(roles) for row in episodes)
def runtime_gates(value, frames, roles=None, expected_episodes=None):
    if not isinstance(value,dict): return {'present':False}
    result={'present':True,'frames':value.get('frames')==frames,
      'valid_frame_rate':value.get('valid_frame_rate',-1)>=.95,
      'max_invalid_burst':value.get('max_invalid_burst',10**9)<=5,
      'worker_restarts':value.get('worker_restarts')==0,'timeouts':value.get('timeouts')==0,
      'ms_per_frame':positive(value.get('ms_per_frame')) and float(value['ms_per_frame'])<=800,
      'runtime_unit':value.get('runtime_unit')=='milliseconds_per_tracked_frame_excluding_support_prompts',
      'episode_reset_strategy':value.get('episode_reset_strategy')=='fresh_inference_core_support_replay_v1'}
    if roles is not None:
        result['role_diagnostics']=role_diagnostics_gates(value,roles,expected_episodes)
    return result
def gates_pass(value): return isinstance(value,dict) and bool(value) and all(value.values())
def old_exp(task,arm): return f'{arm}100k_cutie_object_multitask_100k_v1_seed6_{task.replace("-","_")}'
def old_root(task,arm): return repo/'logs'/task/'6'/old_exp(task,arm)
def new_exp(arm): return f'{arm}100k_cutie_object_multitask_100k_recovery_v2_seed6_acrobot_swingup'
def new_root(arm): return repo/'logs'/'acrobot-swingup'/'6'/new_exp(arm)

jobs=[]; structure=[]; pairing_errors=[]; health=[]
source_import=load(stage/'imports'/'source_v1.json')
source_mutations=[]
for item in source_import.get('files',[]):
    path=Path(item.get('path',''))
    if not path.is_file() or sha(path)!=item.get('sha256'): source_mutations.append(str(path))
if source_mutations: structure.append(f'source v1 mutated/missing: {source_mutations}')

support_path=support_dir/'annotations.json'
support_rc, preflight_rc = rc(stage/'tasks'/'acrobot-swingup'/'support.rc'), rc(stage/'tasks'/'acrobot-swingup'/'preflight.rc')
if support_rc!=0: jobs.append(f'acrobot support rc={support_rc}')
if preflight_rc!=0: jobs.append(f'acrobot preflight rc={preflight_rc}')
support=None
if support_path.is_file():
    try:
        support=load(support_path)
        if support.get('roles')!=['upper_arm','lower_arm'] or support.get('collection',{}).get('task')!='acrobot-swingup':
            raise ValueError('role/task mismatch')
        if any(item.get('object_type')=='site' for row in support.get('records',[])
               for values in row.get('selected_names',{}).values() for item in values):
            raise ValueError('support unexpectedly contains a site')
    except Exception as exc: structure.append(f'acrobot support invalid: {exc}')
else: structure.append(f'missing Acrobot support: {support_path}')
reset_report=None
if preflight_rc==0:
    reset_path=stage/'tasks'/'acrobot-swingup'/'reset_isolation.json'
    try:
        reset=load(reset_path); comparisons=reset.get('comparisons',{}); provenance=reset.get('provenance',{})
        reset_checks={'format':reset.get('format')=='cutie_episode_reset_isolation_v1',
          'status':reset.get('status')=='episode_reset_isolation_pass','pass':reset.get('pass') is True,
          'task':reset.get('task')=='acrobot-swingup','roles':reset.get('role_names')==['upper_arm','lower_arm'],
          'schema':reset.get('support_schema')=='generic_indexed_v1',
          'strategy':reset.get('reset_strategy')=='fresh_inference_core_support_replay_v1',
          'comparisons':bool(comparisons) and all(item.get('byte_equal') is True for item in comparisons.values()),
          'support':same_path(provenance.get('support_path'),support_path)
                    and provenance.get('support_sha256')==sha(support_path),
          'gpu':provenance.get('cuda_visible_devices')==gpu_acro}
        if not all(reset_checks.values()):
            structure.append(f'acrobot reset isolation checks {[k for k,v in reset_checks.items() if not v]}')
        reset_report={'path':str(reset_path),'sha256':sha(reset_path),'checks':reset_checks}
    except Exception as exc:
        structure.append(f'acrobot reset isolation parse: {exc}')

training={}
expected_train_frames=100000+100000//500+(100000//20000+1)*3*501
for arm in cutie_arms:
    directory=stage/'tasks'/'acrobot-swingup'; root=new_root(arm)
    train_rc=rc(directory/f'{arm}.train.rc')
    report={'rc':train_rc,'root':str(root)}; training[arm]=report
    if train_rc!=0:
        jobs.append(f'acrobot/{arm} training rc={train_rc}'); continue
    paths={'config':root/'runtime_config.json','checkpoint':root/'models'/'final.pt',
           'eval':root/'eval.csv','trainer':root/'trainer_runtime.json',
           'replay':root/'replay_runtime.json','perception':root/'perception_runtime.json'}
    missing=[str(p) for p in paths.values() if not p.is_file()]
    if missing: structure.append(f'acrobot/{arm} missing {missing}'); report['missing']=missing; continue
    try:
        cfg, perception = load(paths['config']), load(paths['perception'])
        expected_mode='cutie_hybrid' if arm=='cutie_hybrid' else 'cutie_object_only'
        expected_latent=640 if arm=='cutie_hybrid' else 128
        checks={'task_seed':cfg.get('task')=='acrobot-swingup' and cfg.get('seed')==6,
          'schedule':cfg.get('steps')==100000 and cfg.get('eval_freq')==20000 and cfg.get('eval_episodes')==3,
          'mode':cfg.get('flat_anchor') is True and cfg.get('flat_anchor_mode')==expected_mode,
          'latent':cfg.get('latent_dim')==expected_latent,
          'roles':cfg.get('cutie_object_role_names')==['upper_arm','lower_arm'],
          'support':same_path(cfg.get('cutie_object_support_path'),support_path),
          'oc_repo':same_path(cfg.get('cutie_object_repo'),expected_oc),
          'cutie_checkpoint':same_path(cfg.get('cutie_object_checkpoint'),expected_ckpt)}
        with paths['eval'].open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
        checks['curve']=[int(float(row['step'])) for row in rows]==[0,20000,40000,60000,80000,100000]
        checks['finite_curve']=all(math.isfinite(float(row['episode_reward'])) for row in rows)
        if not all(checks.values()): structure.append(f'acrobot/{arm} checks {[k for k,v in checks.items() if not v]}')
        pg=runtime_gates(perception,expected_train_frames,
                         ('upper_arm','lower_arm'),218)
        if not pg.get('role_diagnostics'):
            structure.append(f'acrobot/{arm} missing or malformed role diagnostics')
        if not gates_pass(pg): health.append(f'acrobot/{arm} training: {pg}')
        report.update({'checks':checks,'checkpoint_sha256':sha(paths['checkpoint']),
                       'training_rewards':[float(row['episode_reward']) for row in rows],
                       'perception_runtime':perception,'perception_gates':pg,
                       'trainer_runtime':load(paths['trainer']),'replay_runtime':load(paths['replay'])})
    except Exception as exc: structure.append(f'acrobot/{arm} parse: {exc}')

def parse_eval(path, task, arm, expected_root, expected_gpu):
    payload=load(path); episodes=payload.get('episodes'); provenance=payload.get('provenance',{})
    evaluation=payload.get('evaluation',{})
    if not (payload.get('format')=='cutie_multitask_checkpoint_evaluation_v1'
            and payload.get('task')==task and payload.get('backend')==arm
            and payload.get('erosion_pixels')==0 and isinstance(episodes,list) and len(episodes)==20
            and [row.get('episode_index') for row in episodes]==list(range(20))
            and evaluation.get('rgb_shift_rng_alignment')=='object_only_equivalent_cuda_randint_v1'
            and evaluation.get('expected_rgb_shift_draws_per_backend')==10000
            and evaluation.get('object_only_alignment_draws')==(10000 if arm=='cutie_object_only' else 0)
            and evaluation.get('actual_foreground_erosion_pixels')==0):
        raise ValueError('evaluation envelope mismatch')
    if not (same_path(provenance.get('runtime_config'),expected_root/'runtime_config.json')
            and same_path(provenance.get('checkpoint'),expected_root/'models'/'final.pt')
            and provenance.get('runtime_config_sha256')==sha(expected_root/'runtime_config.json')
            and provenance.get('checkpoint_sha256')==sha(expected_root/'models'/'final.pt')
            and provenance.get('evaluator_sha256')==sha(expected_evaluator)
            and provenance.get('cuda_visible_devices')==str(expected_gpu)):
        raise ValueError('evaluation provenance mismatch')
    if arm!='rgb':
        raw=load(expected_root/'runtime_config.json')
        support=Path(raw['cutie_object_support_path']).resolve()
        checkpoint=Path(raw['cutie_object_checkpoint']).resolve()
        inputs=provenance.get('cutie_inputs'); ready=provenance.get('cutie_ready')
        if not (isinstance(inputs,dict) and isinstance(ready,dict)
                and same_path(inputs.get('support'),support)
                and inputs.get('support_sha256')==sha(support)
                and same_path(inputs.get('checkpoint'),checkpoint)
                and inputs.get('checkpoint_sha256')==sha(checkpoint)
                and inputs.get('roles')==raw.get('cutie_object_role_names')
                and inputs.get('support_schema')=='generic_indexed_v1'
                and ready.get('task')==task and ready.get('support_task')==task
                and ready.get('roles')==raw.get('cutie_object_role_names')
                and ready.get('support_schema')=='generic_indexed_v1'
                and ready.get('cuda_visible_devices')==str(expected_gpu)):
            raise ValueError('Cutie input/worker provenance mismatch')
    values=[float(row['reward']) for row in episodes]
    if not all(math.isfinite(v) for v in values): raise ValueError('non-finite reward')
    gates=None
    if arm!='rgb':
        role_names={'cup-catch':('cup','ball'),'cartpole-swingup':('cart','pole'),
                    'acrobot-swingup':('upper_arm','lower_arm')}[task]
        gates=runtime_gates(payload.get('perception_runtime'),20*501,role_names,20)
        if not gates.get('role_diagnostics'):
            raise ValueError('missing or malformed role runtime diagnostics')
        if not gates_pass(gates): health.append(f'{task}/{arm} fresh erosion0: {gates}')
    return payload, {'path':str(path),'sha256':sha(path),'reward_mean':statistics.fmean(values),
                     'reward_median':statistics.median(values),'rewards':values,
                     'perception_runtime':payload.get('perception_runtime'),'perception_gates':gates}

evaluations={}
for task in ('cup-catch','cartpole-swingup','acrobot-swingup'):
    reports={}; payloads={}
    for arm in arms:
        if task=='acrobot-swingup' and arm=='rgb':
            path=source/'tasks'/task/'evaluations'/'erosion0'/'rgb.json'; root=old_root(task,arm)
            expected_gpu=source_import.get('source_root') and load(path).get('provenance',{}).get('cuda_visible_devices')
        elif task=='acrobot-swingup':
            path=stage/'tasks'/task/'evaluations'/'erosion0'/f'{arm}.json'; root=new_root(arm); expected_gpu=gpu_acro
            eval_rc=rc(stage/'tasks'/task/'evaluations'/'erosion0'/f'{arm}.rc')
            if eval_rc!=0: jobs.append(f'{task}/{arm} eval rc={eval_rc}'); continue
        else:
            path=stage/'tasks'/task/'normal_recheck'/f'{arm}.json'; root=old_root(task,arm); expected_gpu=gpu_diag
            eval_rc=rc(stage/'tasks'/task/'normal_recheck'/f'{arm}.rc')
            if eval_rc!=0: jobs.append(f'{task}/{arm} normal recheck rc={eval_rc}'); continue
        if not path.is_file(): structure.append(f'missing evaluation {path}'); continue
        try:
            payload, report=parse_eval(path,task,arm,root,expected_gpu)
            payloads[arm]=payload; reports[arm]=report
        except Exception as exc: structure.append(f'{task}/{arm} evaluation parse: {exc}')
    pairing=None
    if set(payloads)!=set(arms): pairing_errors.append(f'{task}: incomplete arms {sorted(payloads)}')
    else:
        mismatch={field:[i for i,rows in enumerate(zip(*(payloads[a]['episodes'] for a in arms)))
                         if len({row.get(field) for row in rows})!=1] for field in pair_fields}
        mismatch['initial_object_hybrid_vs_object_only']=[i for i,(x,y) in enumerate(zip(
            payloads['cutie_hybrid']['episodes'],payloads['cutie_object_only']['episodes']))
            if not x.get('initial_object_sha256') or x.get('initial_object_sha256')!=y.get('initial_object_sha256')]
        device_names={payloads[a].get('provenance',{}).get('device_name') for a in arms}
        mismatch['device_name'] = [] if len(device_names)==1 and None not in device_names else list(range(20))
        pairing={'exact':not any(mismatch.values()),'mismatch_episode_indices':mismatch}
        if not pairing['exact']: pairing_errors.append(f'{task}: {mismatch}')
    evaluations[task]={'arms':reports,'pairing':pairing,
      'source_semantics':('v1 RGB plus recovery-v2 Cutie arms' if task=='acrobot-swingup'
                          else 'fresh recovery-v2 evaluation of all v1 checkpoints')}

worker_rcs={'diagnostic':rc(stage/'diagnostic_worker.rc'),'acrobot':rc(stage/'acrobot_worker.rc')}
if any(value!=0 for value in worker_rcs.values()): jobs.append(f'worker return codes {worker_rcs}')
source_summary=load(source/'generalization_summary.json')
historical={}
for task in ('cup-catch','cartpole-swingup'):
    historical[task]={arm:source_summary.get('tasks',{}).get(task,{}).get('training',{}).get('arms',{}).get(arm,{}).get('perception_gates')
                      for arm in cutie_arms}

# Describe the original six support masks without inventing a post-hoc pass
# threshold. These statistics help distinguish tiny/thin support from a worker
# or checkpoint failure in the fresh normal-condition tracker diagnostics.
support_diagnostics={}
for task in ('cup-catch','cartpole-swingup'):
    cfg=load(old_root(task,'cutie_hybrid')/'runtime_config.json')
    annotations_path=Path(cfg['cutie_object_support_path']).resolve()
    annotations=load(annotations_path); roles=annotations.get('roles')
    if not isinstance(roles,list) or len(roles)!=2:
        structure.append(f'{task}: invalid source support roles {roles!r}'); continue
    per_role={role:{'areas':[],'touch_border_frames':0} for role in roles}
    try:
        for record in annotations.get('records',[]):
            mask=Image.open(annotations_path.parent/record['indexed_mask']).convert('L')
            width,height=mask.size; pixels=list(mask.getdata())
            if (width,height)!=(64,64): raise ValueError((width,height))
            for role_id,role in enumerate(roles,1):
                area=sum(value==role_id for value in pixels)
                border=(any(pixels[x]==role_id or pixels[(height-1)*width+x]==role_id for x in range(width))
                        or any(pixels[y*width]==role_id or pixels[y*width+width-1]==role_id for y in range(height)))
                per_role[role]['areas'].append(area)
                per_role[role]['touch_border_frames']+=int(border)
        if len(annotations.get('records',[]))!=6 or any(len(row['areas'])!=6 for row in per_role.values()):
            raise ValueError('expected exactly six support masks')
        support_diagnostics[task]={'support':str(annotations_path),'support_sha256':sha(annotations_path),
          'records':6,'roles':{
            role:{'area_pixels_min':min(row['areas']),'area_pixels_mean':statistics.fmean(row['areas']),
                  'area_pixels_max':max(row['areas']),'touch_border_frames':row['touch_border_frames']}
            for role,row in per_role.items()}}
    except Exception as exc:
        structure.append(f'{task}: source support diagnostics failed: {exc}')
engineering={'source_v1_immutable':not source_mutations,'all_required_jobs_completed':not jobs,
             'artifact_structure':not structure,'strict_pairing':not pairing_errors}
engineering_pass=all(engineering.values())
tracker_healthy=not health
status=('recovery_engineering_fail' if not engineering_pass else
        ('recovery_complete_tracker_healthy' if tracker_healthy else 'recovery_complete_tracker_unhealthy'))
code_hashes={str(path.relative_to(repo)):sha(path) for path in (
    repo/'tdmpc2'/'tools'/'run_cutie_multitask_recovery_v2.sh',
    repo/'tdmpc2'/'tools'/'collect_cutie_multitask_support.py',
    repo/'tdmpc2'/'tools'/'check_cutie_episode_reset_isolation.py',
    repo/'tdmpc2'/'tools'/'evaluate_cutie_multitask_checkpoint.py',
    repo/'tdmpc2'/'envs'/'wrappers'/'cutie_object.py',
    repo/'tdmpc2'/'check_cutie_multitask_support_contract.py')}
summary={'format':'cutie_object_multitask_recovery_v2','status':status,
 'scientific_scope':'partial seed-6 engineering recovery; it does not rewrite the v1 run or constitute a multi-seed algorithm claim',
 'protocol':{'source_v1':str(source.resolve()),'source_v1_status':source_summary.get('status'),
   'training_seed':6,'steps':100000,'eval_freq':20000,'training_eval_episodes':3,'heldout_episodes':20,
   'normal_validation_erosion_pixels':0,'acrobot_roles':['upper_arm','lower_arm'],
   'acrobot_source_semantics':'fixed target omitted; two articulated moving links are represented',
   'pressure_erosion_results':'not rerun and not a recovery gate',
   'gpu_diagnostic':gpu_diag,'gpu_acrobot':gpu_acro,'code_sha256':code_hashes},
 'engineering_gates':engineering,'tracker_health_outcome':{'pass':tracker_healthy,'failures':health,
   'note':'tracker health is reported but is not an infrastructure/promotion gate'},
 'historical_v1_training_runtime_health':historical,
 'source_support_diagnostics':support_diagnostics,
 'failures':{'jobs':jobs,'structure':structure,'pairing':pairing_errors},
 'source_import':source_import,'acrobot':{'support':{'path':str(support_path),'sha256':sha(support_path) if support_path.is_file() else None,
   'support_rc':support_rc,'preflight_rc':preflight_rc,'reset_isolation':reset_report},'training':training},
 'normal_validation':evaluations,'worker_return_codes':worker_rcs,'elapsed_seconds':int(time.time())-started,
 'recommendation':('fix_recovery_structure_before_interpretation' if not engineering_pass else
   ('eligible_for_500k_multiseed_scaleup' if tracker_healthy else 'diagnose_normal_tracker_failures_before_scaleup'))}
tmp=summary_path.with_name(summary_path.name+'.tmp')
with tmp.open('x',encoding='utf-8',newline='\n') as f:
    json.dump(summary,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
tmp.replace(summary_path)
print(json.dumps({'status':status,'engineering_gates':engineering,
                  'tracker_health_pass':tracker_healthy,'summary':str(summary_path)},indent=2))
raise SystemExit(0 if engineering_pass else 4)
PY
SUMMARY_RC=$?
set -e
if (( SUMMARY_RC != 0 )); then exit "$SUMMARY_RC"; fi

echo "[6/6] Promoting immutable recovery result"
mv -- "$STAGE" "$BASE"
PROMOTED=1
trap - EXIT INT TERM
echo "CUTIE_MULTITASK_RECOVERY_COMPLETE"
echo "SUMMARY=$BASE/recovery_summary.json"
