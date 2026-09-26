#!/usr/bin/env bash
# Evaluation-only, same-checkpoint Cartpole inference ablation.
#
# Both arms load the immutable completed-v4 learned-belief checkpoint.  The
# learned_prior arm uses the production online belief, while measurement_only
# plans directly from the current encoded measurement.  They are deliberately
# separate processes but run serially on the same physical GPU as the original
# v4 Cartpole validation, so the learned_prior arm can be required to reproduce
# that frozen validation exactly.  A scientifically negative result is still a
# successful diagnostic; only provenance, execution, structure, or pairing
# failures return 4.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python}"
SOURCE_V4="${SOURCE_V4:-$REPO_ROOT/logs/_diagnostic/cutie_learned_belief_seed7_pilot_v4}"
GPU_CARTPOLE="${GPU_CARTPOLE:-1}"

readonly RUN_TAG=cutie_cartpole_belief_inference_ablation_v1_seed7
readonly FORMAT=cutie_cartpole_belief_inference_ablation_v1
readonly BASE="$REPO_ROOT/logs/_diagnostic/$RUN_TAG"
readonly STAGE="${BASE}.incomplete"
readonly SUMMARY="$STAGE/cartpole_belief_inference_ablation_summary.json"
readonly SOURCE_SUMMARY="$SOURCE_V4/learned_belief_seed7_summary.json"
readonly TASK=cartpole-swingup
readonly SEED=7 STEPS=100000 EVAL_FREQ=20000 EVAL_EPISODES=3
readonly HELDOUT_EPISODES=20 ENV_SEED=424243 BACKGROUND_SEED=1618034
readonly PLANNER_SEED_BASE=8675400

[[ "$GPU_CARTPOLE" =~ ^(0|[1-9][0-9]*)$ ]] || {
	echo "GPU_CARTPOLE is invalid: $GPU_CARTPOLE" >&2; exit 2;
}
(( BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 1) )) || {
	echo "Bash >=5.1 is required." >&2; exit 2;
}
if [[ "$PY" == */* ]]; then
	[[ -x "$PY" ]] || { echo "Python is not executable: $PY" >&2; exit 2; }
else
	command -v "$PY" >/dev/null || { echo "Python is not on PATH: $PY" >&2; exit 2; }
fi
command -v pgrep >/dev/null || {
	echo "pgrep is required for recursive worker cleanup." >&2; exit 2;
}
for path in \
	"$SOURCE_V4" \
	"$SOURCE_SUMMARY" \
	tdmpc2/check_cutie_latent_belief_contract.py \
	tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py \
	tdmpc2/tools/evaluate_cutie_learned_belief_inference_ablation.py; do
	[[ -e "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done
for path in "$BASE" "$STAGE"; do
	[[ ! -e "$path" ]] || {
		echo "Refusing existing diagnostic output: $path" >&2; exit 3;
	}
done

mkdir -p "$STAGE/provenance" "$STAGE/contracts" "$STAGE/evaluations"
START_SECONDS="$(date +%s)"
ACTIVE_PID=""
PROMOTED=0

terminate_tree() {
	local parent=$1 child
	while IFS= read -r child; do
		[[ -n "$child" ]] && terminate_tree "$child"
	done < <(pgrep -P "$parent" 2>/dev/null || true)
	kill -TERM "$parent" 2>/dev/null || true
}

archive_on_exit() {
	local rc=$? failed
	trap - EXIT INT TERM
	if [[ -n "$ACTIVE_PID" ]]; then
		terminate_tree "$ACTIVE_PID"
		wait "$ACTIVE_PID" 2>/dev/null || true
	fi
	if (( PROMOTED == 0 )) && [[ -d "$STAGE" ]]; then
		if [[ ! -f "$SUMMARY" ]]; then
			"$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps({"format":sys.argv[3],"status":"cartpole_belief_inference_ablation_engineering_fail","runner_exit_code":int(sys.argv[2]),"failure":"runner exited before complete aggregation"},indent=2)+"\n",encoding="utf-8")' "$SUMMARY" "$rc" "$FORMAT" 2>/dev/null || true
		fi
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
		[[ ! -e "$failed" ]] || failed="${failed}.${RANDOM}"
		mv -- "$STAGE" "$failed"
		echo "CUTIE_CARTPOLE_BELIEF_INFERENCE_ABLATION_FAILED_ARCHIVE=$failed" >&2
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
	{ echo "===== $label ====="; "$@"; } \
		>>"$STAGE/contracts/contracts.log" 2>&1
}

echo "[1/5] Dependency-light contracts"
bash -n "$0" || exit 4
run_contract learned_belief "$PY" -m \
	tdmpc2.check_cutie_learned_belief_contract || exit 4
run_contract latent_belief "$PY" -m \
	tdmpc2.check_cutie_latent_belief_contract || exit 4
run_contract evaluator_help "$PY" -m \
	tdmpc2.tools.evaluate_cutie_learned_belief_inference_ablation --help || exit 4

echo "[2/5] Bind immutable completed-v4 source and implementation"
set +e
"$PY" - "$REPO_ROOT" "$SOURCE_V4" "$SOURCE_SUMMARY" \
	"$STAGE/provenance/inputs.json" "$GPU_CARTPOLE" <<'PY'
import hashlib, json, sys
from pathlib import Path

repo, source, source_summary_path, output = map(Path, sys.argv[1:5])
gpu_cartpole = sys.argv[5]
task = 'cartpole-swingup'

def load(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'JSON root is not an object: {path}')
    return value

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def same_path(left, right):
    return Path(left).resolve() == Path(right).resolve()

def safe_relative(root, relative, label):
    if not isinstance(relative, str):
        raise TypeError(f'{label} relative path is not a string')
    item = Path(relative)
    if item.is_absolute() or '..' in item.parts:
        raise ValueError(f'{label} path is unsafe: {relative!r}')
    resolved = (root / item).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f'{label} escapes source root: {relative!r}') from exc
    return resolved

def require_record(record, label, expected_path=None):
    if not isinstance(record, dict):
        raise TypeError(f'{label} record is not an object')
    path = Path(record['path']).resolve()
    if expected_path is not None and path != Path(expected_path).resolve():
        raise RuntimeError(f'{label} path mismatch: {path} != {expected_path}')
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = digest(path)
    if actual != record.get('sha256'):
        raise RuntimeError(f'{label} hash mismatch')
    return {'path': str(path), 'sha256': actual}

def verify_file_map(items, label):
    if not isinstance(items, dict) or not items:
        raise RuntimeError(f'{label} file map is empty')
    result = {}
    for name, record in items.items():
        result[name] = require_record(record, f'{label}/{name}')
    return result

def metadata_inventory(root):
    rows = []
    for path in sorted(item for item in root.rglob('*') if item.is_file()):
        stat = path.stat()
        rows.append(
            f'{path.relative_to(root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n'
        )
    if not rows:
        raise RuntimeError(f'Video inventory is empty: {root}')
    encoded = ''.join(rows).encode('utf-8')
    return {'files': len(rows), 'sha256': hashlib.sha256(encoded).hexdigest()}

def config_tree_digest(root):
    files = sorted(path for path in root.rglob('*.yaml') if path.is_file())
    files += sorted(path for path in root.rglob('*.yml') if path.is_file())
    h = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode('utf-8')
        data = path.read_bytes()
        h.update(len(relative).to_bytes(8, 'little')); h.update(relative)
        h.update(len(data).to_bytes(8, 'little')); h.update(data)
    if not files:
        raise RuntimeError(f'Cutie config tree is empty: {root}')
    return h.hexdigest()

source = source.resolve()
summary = load(source_summary_path)
if summary.get('format') != 'cutie_learned_belief_seed7_pilot_v4':
    raise RuntimeError(f'Unexpected v4 summary format: {summary.get("format")!r}')
if (
    summary.get('status') != 'learned_belief_seed7_engineering_pass'
    or summary.get('engineering_pass') is not True
):
    raise RuntimeError('Completed v4 source is not an engineering pass.')
summary_inputs = summary.get('inputs')
if not isinstance(summary_inputs, dict):
    raise RuntimeError('v4 summary inputs record is missing.')
inputs_path = safe_relative(
    source, summary_inputs.get('relative_to_summary_root'), 'v4 inputs'
)
if not inputs_path.is_file() or digest(inputs_path) != summary_inputs.get('sha256'):
    raise RuntimeError('v4 inputs relative path/hash does not resolve exactly.')
source_inputs = load(inputs_path)
if source_inputs.get('format') != 'cutie_learned_belief_seed7_inputs_v4':
    raise RuntimeError(f'Unexpected v4 inputs format: {source_inputs.get("format")!r}')

task_report = summary.get('tasks', {}).get(task)
if not isinstance(task_report, dict):
    raise RuntimeError('v4 Cartpole task report is missing.')
expected_root = (
    repo / 'logs' / task / '7' /
    'cutie_object_learned_belief100k_'
    'cutie_learned_belief_seed7_pilot_v4_cartpole_swingup'
).resolve()
if not same_path(task_report.get('new_root'), expected_root):
    raise RuntimeError('v4 Cartpole training root is not the canonical expected root.')
runtime = expected_root / 'runtime_config.json'
checkpoint = expected_root / 'models' / 'final.pt'
for path in (runtime, checkpoint):
    if not path.is_file():
        raise FileNotFoundError(path)
artifacts = task_report.get('artifacts')
if not isinstance(artifacts, dict):
    raise RuntimeError('v4 Cartpole artifact records are missing.')
for name, path in (('runtime', runtime), ('checkpoint', checkpoint)):
    record = artifacts.get(name)
    if (
        not isinstance(record, dict)
        or not same_path(record.get('path'), path)
        or record.get('sha256') != digest(path)
    ):
        raise RuntimeError(f'v4 {name} artifact does not bind the canonical source.')

old_normal_record = artifacts.get('normal')
if not isinstance(old_normal_record, dict):
    raise RuntimeError('v4 learned normal artifact record is missing.')
old_normal = safe_relative(
    source, old_normal_record.get('relative_to_summary_root'),
    'v4 learned normal evaluation',
)
if not old_normal.is_file() or digest(old_normal) != old_normal_record.get('sha256'):
    raise RuntimeError('v4 learned normal relative artifact/hash mismatch.')
old_normal_payload = load(old_normal)
if old_normal_payload.get('provenance', {}).get(
    'cuda_visible_devices'
) != gpu_cartpole:
    raise RuntimeError(
        'GPU_CARTPOLE does not match the physical GPU used by frozen v4 '
        f'Cartpole normal: requested={gpu_cartpole!r}, '
        f'source={old_normal_payload.get("provenance", {}).get("cuda_visible_devices")!r}'
    )

cartpole_sources = source_inputs.get('sources', {}).get(task)
if not isinstance(cartpole_sources, dict):
    raise RuntimeError('v4 inputs lack Cartpole source provenance.')
hard_normal_record = cartpole_sources.get('arms', {}).get('hard_zero', {}).get('normal')
hard_normal = require_record(hard_normal_record, 'v4 hard-zero normal')['path']

source_implementation = {}
for relative, record in source_inputs.get('implementation', {}).items():
    source_implementation[relative] = require_record(
        record, f'v4 implementation/{relative}', repo / relative
    )
if not source_implementation:
    raise RuntimeError('v4 implementation map is empty.')

external = source_inputs.get('external')
if not isinstance(external, dict):
    raise RuntimeError('v4 external provenance is missing.')
cutie_checkpoint = require_record(
    external.get('cutie_checkpoint'), 'Cutie checkpoint'
)
manifest_files = verify_file_map(
    external.get('manifest_dir', {}).get('files'), 'background manifest'
)
cutie_sources = verify_file_map(
    external.get('oc_repo', {}).get('cutie_sources'), 'Cutie source'
)
cutie_config_files = verify_file_map(
    external.get('oc_repo', {}).get('config_files'), 'Cutie config'
)
cutie_config_root = (
    Path(external.get('oc_repo', {}).get('path', '')).resolve()
    / 'feature_extractor' / 'cutie' / 'cutie' / 'config'
)
if not cutie_config_root.is_dir():
    raise FileNotFoundError(cutie_config_root)
cutie_config_tree_sha256 = config_tree_digest(cutie_config_root)
if cutie_config_tree_sha256 != external.get('oc_repo', {}).get(
    'config_tree_sha256'
):
    raise RuntimeError('Cutie config tree differs from frozen v4 provenance.')
video_root = Path(external.get('video_root', {}).get('path', '')).resolve()
if not video_root.is_dir():
    raise FileNotFoundError(video_root)
video_inventory = metadata_inventory(video_root)
if (
    video_inventory['files'] != external['video_root'].get('inventory_files')
    or video_inventory['sha256'] != external['video_root'].get('inventory_sha256')
):
    raise RuntimeError('video_hard inventory differs from frozen v4 provenance.')

support = require_record(cartpole_sources.get('support'), 'Cartpole support')
support_files = verify_file_map(
    cartpole_sources.get('support_files'), 'Cartpole support asset'
)

implementation_rel = (
    'tdmpc2/config.yaml',
    'tdmpc2/common/buffer.py',
    'tdmpc2/common/cutie_object_belief.py',
    'tdmpc2/common/layers.py',
    'tdmpc2/common/world_model.py',
    'tdmpc2/envs/wrappers/cutie_object.py',
    'tdmpc2/tdmpc2.py',
    'tdmpc2/check_cutie_latent_belief_contract.py',
    'tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py',
    'tdmpc2/tools/evaluate_cutie_learned_belief_inference_ablation.py',
    'tdmpc2/tools/run_cutie_cartpole_belief_inference_ablation.sh',
)
implementation = {}
for relative in implementation_rel:
    path = (repo / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    implementation[relative] = {'path': str(path), 'sha256': digest(path)}

payload = {
    'format': 'cutie_cartpole_belief_inference_ablation_inputs_v1',
    'scientific_scope': (
        'single-seed same-checkpoint evaluation-only development diagnostic; '
        'normal validation only; not a paper result'
    ),
    'source_v4': {
        'root': str(source),
        'summary': {
            'path': str(source_summary_path.resolve()),
            'sha256': digest(source_summary_path),
        },
        'inputs': {'path': str(inputs_path), 'sha256': digest(inputs_path)},
        'cartpole_training_root': str(expected_root),
        'runtime_config': {'path': str(runtime), 'sha256': digest(runtime)},
        'checkpoint': {'path': str(checkpoint), 'sha256': digest(checkpoint)},
        'old_learned_normal': {
            'path': str(old_normal), 'sha256': digest(old_normal),
        },
        'hard_zero_normal': {
            'path': str(Path(hard_normal)), 'sha256': digest(Path(hard_normal)),
        },
        'implementation': source_implementation,
        'external': {
            'video_root': {
                'path': str(video_root),
                'inventory_files': video_inventory['files'],
                'inventory_sha256': video_inventory['sha256'],
            },
            'manifest_files': manifest_files,
            'cutie_checkpoint': cutie_checkpoint,
            'cutie_sources': cutie_sources,
            'cutie_config_files': cutie_config_files,
            'cutie_config_root': str(cutie_config_root),
            'cutie_config_tree_sha256': cutie_config_tree_sha256,
        },
        'cartpole_support': support,
        'cartpole_support_files': support_files,
    },
    'implementation': implementation,
    'protocol': {
        'task': task,
        'training_seed': 7,
        'episodes': 20,
        'condition': 'normal',
        'env_seed': 424243,
        'background_seed': 1618034,
        'planner_seed_base': 8675400,
        'training_performed': False,
        'same_physical_gpu_required': True,
        'physical_gpu_for_both_arms': gpu_cartpole,
    },
}
with output.open('x', encoding='utf-8', newline='\n') as file:
    json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
    file.write('\n')
print('CUTIE_CARTPOLE_BELIEF_INFERENCE_INPUTS_OK', json.dumps({
    'source': str(source), 'checkpoint': str(checkpoint), 'output': str(output),
}, allow_nan=False))
PY
BIND_RC=$?
set -e
(( BIND_RC == 0 )) || exit 4

set +e
RUNTIME_CONFIG="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["source_v4"]["runtime_config"]["path"])' "$STAGE/provenance/inputs.json")"; RUNTIME_RC=$?
CHECKPOINT="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["source_v4"]["checkpoint"]["path"])' "$STAGE/provenance/inputs.json")"; CHECKPOINT_RC=$?
set -e
(( RUNTIME_RC == 0 && CHECKPOINT_RC == 0 )) || exit 4
[[ -f "$RUNTIME_CONFIG" && -f "$CHECKPOINT" ]] || {
	echo "Bound v4 runtime/checkpoint disappeared." >&2; exit 4;
}

run_evaluation() {
	local arm=$1 out="$STAGE/evaluations/${1}.json"
	local log="$STAGE/evaluations/${1}.log" rc
	echo "EVAL_START task=$TASK arm=$arm gpu=$GPU_CARTPOLE"
	set +e
	env CUDA_VISIBLE_DEVICES="$GPU_CARTPOLE" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" -m tdmpc2.tools.evaluate_cutie_learned_belief_inference_ablation \
		--arm "$arm" --runtime-config "$RUNTIME_CONFIG" \
		--checkpoint "$CHECKPOINT" --training-seed "$SEED" \
		--expected-training-steps "$STEPS" \
		--expected-training-eval-freq "$EVAL_FREQ" \
		--expected-training-eval-episodes "$EVAL_EPISODES" \
		--env-seed "$ENV_SEED" --background-seed "$BACKGROUND_SEED" \
		--planner-seed-base "$PLANNER_SEED_BASE" --output "$out" \
		>"$log" 2>&1 &
	ACTIVE_PID=$!
	wait "$ACTIVE_PID"
	rc=$?
	ACTIVE_PID=""
	set -e
	write_rc "$STAGE/evaluations/${arm}.rc" "$rc"
	echo "EVAL_END task=$TASK arm=$arm gpu=$GPU_CARTPOLE rc=$rc" | tee -a "$log"
}

echo "[3/5] Same-GPU independent-process normal validation"
# Keep this order: the production arm first makes an exact regression against
# the original v4 GPU-1 result before running the causal measurement bypass.
run_evaluation learned_prior
run_evaluation measurement_only

echo "[4/5] Strict aggregation and same-checkpoint decomposition"
set +e
"$PY" - "$STAGE" "$SUMMARY" "$REPO_ROOT" "$START_SECONDS" \
	"$GPU_CARTPOLE" <<'PY'
import hashlib, json, math, statistics, sys, time
from pathlib import Path

stage, summary_path, repo = map(Path, sys.argv[1:4])
started = int(sys.argv[4])
gpu = sys.argv[5]
inputs_path = stage / 'provenance' / 'inputs.json'
arms = ('learned_prior', 'measurement_only')

def load(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'JSON root is not an object: {path}')
    return value

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_rc(path):
    try:
        return int(path.read_text(encoding='utf-8').strip())
    except Exception:
        return None

def same_path(left, right):
    try:
        return Path(left).resolve() == Path(right).resolve()
    except Exception:
        return False

def valid_hash(value):
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in '0123456789abcdef' for character in value)
    )

def metadata_inventory(root):
    rows = []
    for path in sorted(item for item in root.rglob('*') if item.is_file()):
        stat = path.stat()
        rows.append(
            f'{path.relative_to(root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n'
        )
    encoded = ''.join(rows).encode('utf-8')
    return len(rows), hashlib.sha256(encoded).hexdigest()

def config_tree_digest(root):
    files = sorted(path for path in root.rglob('*.yaml') if path.is_file())
    files += sorted(path for path in root.rglob('*.yml') if path.is_file())
    h = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode('utf-8')
        data = path.read_bytes()
        h.update(len(relative).to_bytes(8, 'little')); h.update(relative)
        h.update(len(data).to_bytes(8, 'little')); h.update(data)
    if not files:
        raise RuntimeError(f'Cutie config tree is empty: {root}')
    return h.hexdigest()

def recheck_record(record, label, failures, expected_path=None):
    try:
        path = Path(record['path']).resolve()
        if expected_path is not None and path != Path(expected_path).resolve():
            failures.append(f'{label}: path changed')
            return
        if not path.is_file() or digest(path) != record['sha256']:
            failures.append(f'{label}: missing or hash changed')
    except Exception as exc:
        failures.append(f'{label}: recheck error {type(exc).__name__}: {exc}')

def finite_rewards(payload):
    values = [float(row['reward']) for row in payload.get('episodes', [])]
    if len(values) != 20 or not all(math.isfinite(value) for value in values):
        raise ValueError('Expected exactly 20 finite rewards.')
    return values

def paired_stats(left, right, left_label, right_label):
    if len(left) != 20 or len(right) != 20:
        raise ValueError('Paired statistics require 20 episodes per arm.')
    deltas = [float(b) - float(a) for a, b in zip(left, right)]
    mean = statistics.fmean(deltas)
    sample_std = statistics.stdev(deltas)
    half = 2.093024054408263 * sample_std / math.sqrt(20)
    return {
        f'{left_label}_reward_mean': statistics.fmean(left),
        f'{right_label}_reward_mean': statistics.fmean(right),
        f'mean_delta_{right_label}_minus_{left_label}': mean,
        'median_delta': statistics.median(deltas),
        'paired_delta_sample_std': sample_std,
        'paired_delta_95pct_t_interval_df19': [mean - half, mean + half],
        'win_tie_loss_for_right': [
            sum(value > 0 for value in deltas),
            sum(value == 0 for value in deltas),
            sum(value < 0 for value in deltas),
        ],
        'paired_deltas': deltas,
        'interval_scope': (
            'conditional paired-episode t interval for one frozen training seed; '
            'not uncertainty across training seeds'
        ),
    }

inputs = load(inputs_path)
source = inputs['source_v4']
jobs = []
structure = []
pairing_failures = []
regression_failures = []
immutable_failures = []
payloads = {}
reports = {}

# After-run recheck of every source or implementation item snapshotted before
# evaluation.  The evaluator also checks its runtime/checkpoint/implementation
# before and after each individual process.
for name in (
    'summary', 'inputs', 'runtime_config', 'checkpoint',
    'old_learned_normal', 'hard_zero_normal', 'cartpole_support',
):
    recheck_record(source[name], f'source_v4/{name}', immutable_failures)
for relative, record in source.get('implementation', {}).items():
    recheck_record(
        record, f'v4 implementation/{relative}', immutable_failures,
        repo / relative,
    )
for relative, record in inputs.get('implementation', {}).items():
    recheck_record(
        record, f'ablation implementation/{relative}', immutable_failures,
        repo / relative,
    )
external = source.get('external', {})
recheck_record(
    external.get('cutie_checkpoint', {}), 'Cutie checkpoint', immutable_failures
)
for label, key in (
    ('background manifest', 'manifest_files'),
    ('Cutie source', 'cutie_sources'),
    ('Cutie config', 'cutie_config_files'),
):
    items = external.get(key)
    if not isinstance(items, dict) or not items:
        immutable_failures.append(f'{label}: frozen file map missing')
    else:
        for relative, record in items.items():
            recheck_record(record, f'{label}/{relative}', immutable_failures)
support_files = source.get('cartpole_support_files')
if not isinstance(support_files, dict) or not support_files:
    immutable_failures.append('Cartpole support asset map missing')
else:
    for relative, record in support_files.items():
        recheck_record(
            record, f'Cartpole support asset/{relative}', immutable_failures
        )
try:
    video = external['video_root']
    count, video_hash = metadata_inventory(Path(video['path']))
    if count != video['inventory_files'] or video_hash != video['inventory_sha256']:
        immutable_failures.append('video_hard inventory changed')
except Exception as exc:
    immutable_failures.append(f'video_hard recheck error: {exc}')
try:
    config_root = Path(external['cutie_config_root'])
    if (
        not config_root.is_dir()
        or config_tree_digest(config_root)
        != external['cutie_config_tree_sha256']
    ):
        immutable_failures.append('Cutie config tree changed')
except Exception as exc:
    immutable_failures.append(f'Cutie config tree recheck error: {exc}')

worker_rcs = {
    arm: read_rc(stage / 'evaluations' / f'{arm}.rc') for arm in arms
}
if any(value != 0 for value in worker_rcs.values()):
    jobs.append(f'evaluation return codes: {worker_rcs}')

expected_evaluator_hash = inputs['implementation'][
    'tdmpc2/tools/evaluate_cutie_learned_belief_inference_ablation.py'
]['sha256']
expected_base_hash = inputs['implementation'][
    'tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py'
]['sha256']
expected_runtime = source['runtime_config']
expected_checkpoint = source['checkpoint']
expected_cutie = source['external']['cutie_checkpoint']
expected_support = source['cartpole_support']
evaluator_implementation_rel = {
    'evaluator':
        'tdmpc2/tools/evaluate_cutie_learned_belief_inference_ablation.py',
    'base_evaluator': 'tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py',
    'agent': 'tdmpc2/tdmpc2.py',
    'world_model': 'tdmpc2/common/world_model.py',
    'belief_helpers': 'tdmpc2/common/cutie_object_belief.py',
    'layers': 'tdmpc2/common/layers.py',
    'object_wrapper': 'tdmpc2/envs/wrappers/cutie_object.py',
}

for arm in arms:
    path = stage / 'evaluations' / f'{arm}.json'
    report = {
        'relative_to_summary_root': f'evaluations/{arm}.json',
        'return_code': worker_rcs[arm],
    }
    reports[arm] = report
    if not path.is_file():
        structure.append(f'{arm}: evaluation JSON missing')
        continue
    try:
        value = load(path)
        payloads[arm] = value
        episodes = value.get('episodes')
        inference = value.get('inference_runtime')
        provenance = value.get('provenance')
        perception = value.get('perception_runtime')
        strict = value.get('strict_checks')
        if not isinstance(episodes, list):
            episodes = []
        if not isinstance(inference, dict):
            inference = {}
        if not isinstance(provenance, dict):
            provenance = {}
        if not isinstance(perception, dict):
            perception = {}
        if not isinstance(strict, dict):
            strict = {}
        implementation = provenance.get('implementation', {})
        cutie_inputs = provenance.get('cutie_inputs', {})
        checks = {
            'format': value.get('format')
                == 'cutie_learned_belief_inference_ablation_v1',
            'identity': value.get('task') == 'cartpole-swingup'
                and value.get('backend') == 'cutie_object_only'
                and value.get('arm') == arm
                and value.get('condition') == 'normal'
                and value.get('training_seed') == 7,
            'evaluation_protocol': value.get('evaluation', {}).get('episodes') == 20
                and value.get('evaluation', {}).get('env_seed') == 424243
                and value.get('evaluation', {}).get('background_seed') == 1618034
                and value.get('evaluation', {}).get('planner_seed_base') == 8675400
                and value.get('evaluation', {}).get('object_only_alignment_draws')
                    == 10000,
            'episodes': len(episodes) == 20
                and all(row.get('episode_index') == index
                        and row.get('length') == 500
                        for index, row in enumerate(episodes)),
            'strict_evaluator_checks': bool(strict) and all(strict.values()),
            'runtime_checkpoint_identity':
                same_path(provenance.get('runtime_config'), expected_runtime['path'])
                and provenance.get('runtime_config_sha256')
                    == expected_runtime['sha256']
                and provenance.get('runtime_config_sha256_after')
                    == expected_runtime['sha256']
                and same_path(provenance.get('checkpoint'), expected_checkpoint['path'])
                and provenance.get('checkpoint_sha256')
                    == expected_checkpoint['sha256']
                and provenance.get('checkpoint_sha256_after')
                    == expected_checkpoint['sha256'],
            'evaluator_identity': implementation.get('evaluator', {}).get('sha256')
                == expected_evaluator_hash
                and implementation.get('evaluator', {}).get('sha256_after')
                    == expected_evaluator_hash
                and implementation.get('base_evaluator', {}).get('sha256')
                    == expected_base_hash
                and implementation.get('base_evaluator', {}).get('sha256_after')
                    == expected_base_hash,
            'all_evaluator_implementation_immutable': bool(implementation)
                and set(implementation) == set(evaluator_implementation_rel)
                and all(
                    isinstance(implementation.get(name), dict)
                    and same_path(
                        implementation[name].get('path'),
                        inputs['implementation'][relative]['path'],
                    )
                    and implementation[name].get('sha256')
                        == inputs['implementation'][relative]['sha256']
                    and implementation[name].get('sha256_after')
                        == inputs['implementation'][relative]['sha256']
                    and valid_hash(implementation[name].get('sha256'))
                    for name, relative in evaluator_implementation_rel.items()
                ),
            'cutie_inputs': cutie_inputs.get('checkpoint_sha256')
                == expected_cutie['sha256']
                and same_path(cutie_inputs.get('checkpoint'), expected_cutie['path'])
                and cutie_inputs.get('support_sha256') == expected_support['sha256']
                and same_path(cutie_inputs.get('support'), expected_support['path'])
                and cutie_inputs.get('roles') == ['cart', 'pole'],
            'normal_perception_runtime': perception.get('frames') == 10020
                and perception.get('worker_restarts') == 0
                and perception.get('timeouts') == 0
                and perception.get('last_valid_memory', {}).get('enabled') is False
                and perception.get('policy_observation_intervention', {}).get('enabled')
                    is False,
            'same_requested_physical_gpu': provenance.get('cuda_visible_devices') == gpu,
            'inference_mode': inference.get('mode') == arm
                and inference.get('arm') == arm
                and inference.get('enabled') is True
                and inference.get('measurement_steps') == 10000,
            'role_accounting': isinstance(inference.get('role_invalid_decisions'), list)
                and len(inference['role_invalid_decisions']) == 2
                and sum(inference['role_invalid_decisions']) > 0
                and all(
                    inference.get('role_valid_decisions', [None, None])[index]
                    + inference['role_invalid_decisions'][index] == 10000
                    for index in range(2)
                ),
            'episode_hashes': all(
                all(valid_hash(row.get(key)) for key in (
                    'planner_rng_start_sha256', 'planner_rng_end_sha256',
                    'initial_rgb_sha256', 'initial_object_sha256',
                    'final_object_sha256', 'action_trace_sha256',
                    'reward_trace_sha256', 'controller_latent_trace_sha256',
                    'role_validity_trace_sha256',
                )) for row in episodes
            ),
        }
        if arm == 'learned_prior':
            checks.update({
                'production_entrypoint': inference.get('planner_entrypoint')
                    == 'TDMPC2.act',
                'belief_forward_calls': inference.get(
                    'belief_dynamics_forward_calls'
                ) == 9980,
                'prior_role_uses': isinstance(inference.get('prior_role_uses'), int)
                    and inference['prior_role_uses'] > 0
                    and inference['prior_role_uses']
                    == sum(row.get('prior_role_opportunities', -1)
                           for row in episodes),
                'belief_resets': inference.get('belief_resets') == 19,
                'measurement_bypass_unused': inference.get(
                    'measurement_only_latent_steps'
                ) == 0,
                'online_state_retained': inference.get(
                    'online_belief_state_is_none'
                ) is False,
            })
        else:
            checks.update({
                'measurement_entrypoint': inference.get('planner_entrypoint')
                    == 'TDMPC2.act_from_latent',
                'belief_forward_calls_zero': inference.get(
                    'belief_dynamics_forward_calls'
                ) == 0,
                'prior_role_uses_zero': inference.get('prior_role_uses') == 0,
                'belief_resets_zero': inference.get('belief_resets') == 0,
                'measurement_bypass_steps': inference.get(
                    'measurement_only_latent_steps'
                ) == 10000,
                'online_state_absent': inference.get(
                    'online_belief_state_is_none'
                ) is True and inference.get('online_belief_action_is_none') is True,
            })
        failed = sorted(key for key, passed in checks.items() if not passed)
        if failed:
            structure.append(f'{arm}: checks failed {failed}')
        reward_values = finite_rewards(value)
        report.update({
            'sha256': digest(path),
            'checks': checks,
            'failed_checks': failed,
            'reward_mean': statistics.fmean(reward_values),
            'reward_median': statistics.median(reward_values),
            'reward_sample_std': statistics.stdev(reward_values),
            'inference_runtime': inference,
            'perception_runtime': perception,
            'device_name': provenance.get('device_name'),
        })
    except Exception as exc:
        structure.append(f'{arm}: parse failed {type(exc).__name__}: {exc}')

pair_fields = (
    'episode_index', 'planner_seed', 'planner_rng_start_sha256',
    'planner_rng_end_sha256', 'initial_rgb_sha256', 'initial_object_sha256',
    'background_source', 'background_start_frame_index',
    'background_end_source', 'background_end_frame_index', 'length',
)
pairing = {'exact': False, 'fields': list(pair_fields)}
if set(payloads) == set(arms):
    left = payloads['learned_prior']['episodes']
    right = payloads['measurement_only']['episodes']
    mismatches = {
        field: [index for index, (a, b) in enumerate(zip(left, right))
                if a.get(field) != b.get(field)]
        for field in pair_fields
    }
    bad = {key: value for key, value in mismatches.items() if value}
    pairing = {
        'exact': not bad, 'fields': list(pair_fields),
        'mismatch_episode_indices': bad,
    }
    if bad:
        pairing_failures.append(f'new arms pairing mismatch: {bad}')
else:
    pairing_failures.append(f'new arms incomplete: {sorted(payloads)}')

old_learned = hard_zero = None
try:
    old_learned = load(Path(source['old_learned_normal']['path']))
    hard_zero = load(Path(source['hard_zero_normal']['path']))
    for label, value in (('old learned', old_learned), ('hard-zero', hard_zero)):
        if (
            value.get('format') != 'cutie_multitask_checkpoint_evaluation_v1'
            or value.get('task') != 'cartpole-swingup'
            or value.get('backend') != 'cutie_object_only'
            or value.get('training_seed') != 7
            or value.get('evaluation', {}).get('episodes') != 20
        ):
            raise RuntimeError(f'{label} frozen evaluation identity is invalid')
except Exception as exc:
    regression_failures.append(f'frozen source parse failed: {exc}')

old_reproduction = {'exact': False}
hard_anchor_pairing = {'exact': False}
old_pair_fields = (
    'episode_index', 'initial_rgb_sha256', 'initial_object_sha256',
    'background_source', 'background_start_frame_index', 'planner_seed',
    'planner_rng_start_sha256', 'planner_rng_end_sha256', 'length', 'reward',
)
if old_learned is not None and 'learned_prior' in payloads:
    old_rows = old_learned.get('episodes', [])
    new_rows = payloads['learned_prior'].get('episodes', [])
    mismatches = {
        field: [index for index, (old, new) in enumerate(zip(old_rows, new_rows))
                if old.get(field) != new.get(field)]
        for field in old_pair_fields
    }
    bad = {key: value for key, value in mismatches.items() if value}
    exact = len(old_rows) == len(new_rows) == 20 and not bad
    old_reproduction = {
        'exact': exact, 'fields': list(old_pair_fields),
        'source_sha256': source['old_learned_normal']['sha256'],
        'mismatch_episode_indices': bad,
    }
    if not exact:
        regression_failures.append(
            f'learned_prior did not exactly reproduce v4 normal: {bad}'
        )
else:
    regression_failures.append('learned_prior/source reproduction unavailable')

anchor_fields = tuple(field for field in old_pair_fields if field != 'reward')
if hard_zero is not None and 'measurement_only' in payloads:
    old_rows = hard_zero.get('episodes', [])
    new_rows = payloads['measurement_only'].get('episodes', [])
    mismatches = {
        field: [index for index, (old, new) in enumerate(zip(old_rows, new_rows))
                if old.get(field) != new.get(field)]
        for field in anchor_fields
    }
    bad = {key: value for key, value in mismatches.items() if value}
    exact = len(old_rows) == len(new_rows) == 20 and not bad
    hard_anchor_pairing = {
        'exact': exact, 'fields': list(anchor_fields),
        'source_sha256': source['hard_zero_normal']['sha256'],
        'mismatch_episode_indices': bad,
    }
    if not exact:
        pairing_failures.append(f'hard-zero anchor pairing mismatch: {bad}')
else:
    pairing_failures.append('hard-zero/measurement-only anchor pairing unavailable')

device_names = {
    payload.get('provenance', {}).get('device_name') for payload in payloads.values()
}
old_learned_provenance = (
    old_learned.get('provenance', {}) if isinstance(old_learned, dict) else {}
)
same_gpu = (
    len(payloads) == 2
    and all(payload.get('provenance', {}).get('cuda_visible_devices') == gpu
            for payload in payloads.values())
    and len(device_names) == 1 and None not in device_names
    and old_learned_provenance.get('cuda_visible_devices') == gpu
    and old_learned_provenance.get('device_name') in device_names
)
if not same_gpu:
    structure.append(
        f'same physical GPU/model check failed: requested={gpu}, devices={device_names}'
    )

manifest_payloads = list(payloads.values())
if isinstance(old_learned, dict):
    manifest_payloads.append(old_learned)
if isinstance(hard_zero, dict):
    manifest_payloads.append(hard_zero)
validation_manifest_hashes = {
    value.get('provenance', {}).get('validation_manifest_sha256')
    for value in manifest_payloads
}
combined_manifest_hashes = {
    value.get('provenance', {}).get('combined_manifest_sha256')
    for value in manifest_payloads
}
manifest_identity = (
    len(manifest_payloads) == 4
    and len(validation_manifest_hashes) == 1
    and None not in validation_manifest_hashes
    and len(combined_manifest_hashes) == 1
    and None not in combined_manifest_hashes
)
if not manifest_identity:
    structure.append(
        'validation/combined manifest identity failed: '
        f'validation={validation_manifest_hashes}, combined={combined_manifest_hashes}'
    )

runtime_hashes = {
    value.get('provenance', {}).get('runtime_config_sha256')
    for value in payloads.values()
}
checkpoint_hashes = {
    value.get('provenance', {}).get('checkpoint_sha256')
    for value in payloads.values()
}
same_runtime_and_checkpoint = (
    len(payloads) == 2
    and runtime_hashes == {source['runtime_config']['sha256']}
    and checkpoint_hashes == {source['checkpoint']['sha256']}
    and None not in runtime_hashes
    and None not in checkpoint_hashes
)

engineering_gates = {
    'dependency_light_contracts': True,
    'all_evaluations_completed': not jobs,
    'immutable_source_and_implementation': not immutable_failures,
    'artifact_and_protocol_structure': not structure,
    'strict_new_arm_pairing': not pairing_failures,
    'learned_prior_exactly_reproduces_v4_normal': not regression_failures,
    'same_physical_gpu_and_model': same_gpu,
    'same_runtime_and_checkpoint': same_runtime_and_checkpoint,
    'validation_and_combined_manifests_exact': manifest_identity,
}
engineering_pass = all(engineering_gates.values())

scientific = {
    'available': False,
    'scope': (
        'conditional paired-episode development evidence for one frozen '
        'training seed; not across-training-seed uncertainty'
    ),
}
classification = 'engineering_failure_do_not_interpret_rewards'
recommendation = 'fix_engineering_before_reward_interpretation'
if engineering_pass:
    hard_rewards = finite_rewards(hard_zero)
    on_rewards = finite_rewards(payloads['learned_prior'])
    off_rewards = finite_rewards(payloads['measurement_only'])
    off_minus_on = paired_stats(
        on_rewards, off_rewards, 'learned_prior', 'measurement_only'
    )
    hard_minus_off = paired_stats(
        off_rewards, hard_rewards, 'measurement_only', 'hard_zero'
    )
    hard_minus_on = paired_stats(
        on_rewards, hard_rewards, 'learned_prior', 'hard_zero'
    )
    hard_mean = statistics.fmean(hard_rewards)
    on_mean = statistics.fmean(on_rewards)
    off_mean = statistics.fmean(off_rewards)
    total_gap = hard_mean - on_mean
    inference_component = off_mean - on_mean
    checkpoint_remainder = hard_mean - off_mean
    residual = total_gap - (inference_component + checkpoint_remainder)
    ci = off_minus_on['paired_delta_95pct_t_interval_df19']
    wins = off_minus_on['win_tie_loss_for_right'][0]
    prior_harm = (
        hard_mean > 0
        and inference_component >= 0.05 * hard_mean
        and ci[0] > 0
        and wins >= 12
    )
    recovery_fraction = (
        inference_component / total_gap if total_gap > 0 else None
    )
    off_retention = off_mean / hard_mean if hard_mean > 0 else None
    if (
        prior_harm and recovery_fraction is not None
        and recovery_fraction >= 0.50 and off_retention is not None
        and off_retention >= 0.90
    ):
        classification = 'inference_time_belief_is_primary_gap_source'
        recommendation = 'redesign_inference_time_belief_before_scaling'
    elif prior_harm and off_retention is not None and off_retention < 0.80:
        classification = 'mixed_inference_time_belief_and_checkpoint_remainder_gap'
        recommendation = 'redesign_belief_and_inspect_checkpoint_remainder'
    elif prior_harm:
        classification = 'inference_time_belief_materially_contributes_to_gap'
        recommendation = 'redesign_inference_time_belief_then_recheck_checkpoint_remainder'
    elif off_retention is not None and off_retention < 0.80:
        classification = 'checkpoint_remainder_dominant_or_diagnostic_inconclusive'
        recommendation = 'inspect_remainder_of_learned_controller_checkpoint'
    else:
        classification = 'no_strong_evidence_of_material_inference_time_belief_harm'
        recommendation = 'retain_checkpoint_and_validate_with_additional_training_seeds'
    scientific = {
        'available': True,
        'scope': (
            'conditional paired-episode development evidence for one frozen '
            'training seed; not across-training-seed uncertainty'
        ),
        'notation': {
            'H': 'frozen hard-zero checkpoint normal reward',
            'On': 'same learned checkpoint with production learned prior',
            'Off': 'same learned checkpoint using current-measurement latent only',
        },
        'reward_means': {
            'H_hard_zero': hard_mean,
            'On_learned_prior': on_mean,
            'Off_measurement_only': off_mean,
        },
        'paired_statistics': {
            'Off_minus_On': off_minus_on,
            'H_minus_Off': hard_minus_off,
            'H_minus_On': hard_minus_on,
        },
        'same_checkpoint_decomposition': {
            'identity': 'H-On = (Off-On) + (H-Off)',
            'H_minus_On_total_gap': total_gap,
            'Off_minus_On_inference_time_belief_component': inference_component,
            'H_minus_Off_checkpoint_remainder_component': checkpoint_remainder,
            'numeric_residual': residual,
            'inference_component_fraction_of_positive_total_gap': recovery_fraction,
            'Off_over_H_reward_retention': off_retention,
        },
        'classification': classification,
        'classification_language': (
            'This distinguishes inference-time belief use from the remainder of '
            'the learned controller checkpoint; it does not identify a policy-only '
            'effect and does not compare training algorithms.'
        ),
        'prior_harm_gate': {
            'pass': prior_harm,
            'requirements': {
                'Off_minus_On_at_least_5pct_of_H': inference_component
                    >= 0.05 * hard_mean,
                'paired_t_interval_lower_above_zero': ci[0] > 0,
                'paired_wins_at_least_12_of_20': wins >= 12,
            },
        },
    }

summary = {
    'format': 'cutie_cartpole_belief_inference_ablation_v1',
    'status': (
        'cartpole_belief_inference_ablation_engineering_pass'
        if engineering_pass
        else 'cartpole_belief_inference_ablation_engineering_fail'
    ),
    'scientific_scope': (
        'single-training-seed same-checkpoint inference-time ablation on '
        'Cartpole normal validation; no training; not a paper result'
    ),
    'protocol': {
        'task': 'cartpole-swingup',
        'condition': 'normal',
        'training_seed': 7,
        'episodes': 20,
        'arms': {
            'learned_prior': 'production TDMPC2.act with learned online belief',
            'measurement_only': (
                'same checkpoint; current object measurement encoded every step; '
                'evaluation-only TDMPC2.act_from_latent; no belief dynamics call'
            ),
        },
        'training_performed': False,
        'independent_processes': True,
        'execution_order': ['learned_prior', 'measurement_only'],
        'physical_gpu_for_both_arms': gpu,
        'strict_pairing_fields': list(pair_fields),
    },
    'engineering_pass': engineering_pass,
    'engineering_gates': engineering_gates,
    'failures': {
        'jobs': jobs,
        'immutable': immutable_failures,
        'structure': structure,
        'pairing': pairing_failures,
        'source_regression': regression_failures,
    },
    'pairing': pairing,
    'source_regressions': {
        'learned_prior_vs_v4_learned_normal': old_reproduction,
        'measurement_only_vs_hard_zero_anchor_exogenous_pairing': hard_anchor_pairing,
    },
    'scientific_outcome': scientific,
    'recommendation': recommendation,
    'inputs': {
        'relative_to_summary_root': 'provenance/inputs.json',
        'sha256': digest(inputs_path),
    },
    'evaluation_reports': reports,
    'worker_return_codes': worker_rcs,
    'elapsed_seconds': int(time.time()) - started,
    'summary_relative_paths_authoritative': True,
}
temporary = summary_path.with_name(summary_path.name + '.incomplete')
with temporary.open('x', encoding='utf-8', newline='\n') as file:
    json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    file.write('\n')
temporary.replace(summary_path)
print(json.dumps({
    'status': summary['status'],
    'engineering_gates': engineering_gates,
    'scientific_classification': classification,
    'recommendation': recommendation,
    'summary': str(summary_path),
}, ensure_ascii=False, indent=2, allow_nan=False))
raise SystemExit(0 if engineering_pass else 4)
PY
AGG_RC=$?
set -e
(( AGG_RC == 0 )) || exit 4

echo "[5/5] Promote immutable diagnostic"
mv -- "$STAGE" "$BASE"
PROMOTED=1
trap - EXIT INT TERM
echo "CUTIE_CARTPOLE_BELIEF_INFERENCE_ABLATION_COMPLETE"
echo "SUMMARY=$BASE/cartpole_belief_inference_ablation_summary.json"
