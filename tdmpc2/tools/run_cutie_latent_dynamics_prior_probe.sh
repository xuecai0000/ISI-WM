#!/usr/bin/env bash
# Frozen, evaluation-only two-GPU diagnostic for an action-conditioned latent
# prior. This runner never trains or mutates the seed-7 ObjectOnly source run.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${VIDEO_ROOT:?Set VIDEO_ROOT to the existing video_hard directory}"
: "${OC_REPO:?Set OC_REPO to the OC-STORM checkout}"
: "${CUTIE_CKPT:?Set CUTIE_CKPT to cutie-small-mega.pth}"

PY="${PY:-python}"
GPU_MEASUREMENT="${GPU_MEASUREMENT:-0}"
GPU_PRIOR="${GPU_PRIOR:-1}"
readonly SEED=7 STEPS=100000 EVAL_FREQ=20000 TRAIN_EVAL_EPISODES=3
readonly EPISODES=20 ENV_SEED=424243 BACKGROUND_SEED=1618034
readonly PLANNER_SEED_BASE=8675400
readonly RUN_TAG=cutie_latent_dynamics_prior_probe_v1_seed7
readonly FORMAT=cutie_latent_dynamics_prior_probe_v1
readonly BASE="$REPO_ROOT/logs/_diagnostic/$RUN_TAG"
readonly STAGE="${BASE}.incomplete"
readonly SUMMARY="$STAGE/latent_dynamics_prior_summary.json"
readonly SOURCE_BASE="$REPO_ROOT/logs/_diagnostic/cutie_object_memory_probe_100k_v1_seed7"
readonly SOURCE_SUMMARY="${SOURCE_SUMMARY:-$SOURCE_BASE/memory_probe_summary.json}"
readonly SOURCE_RUN="$REPO_ROOT/logs/reacher-visual-small/7/cutie_object_hard_zero100k_cutie_object_memory_probe_100k_v1_seed7_reacher_visual_small"
readonly RUNTIME_CONFIG="$SOURCE_RUN/runtime_config.json"
readonly CHECKPOINT="$SOURCE_RUN/models/final.pt"
readonly -a ARMS=(measurement_only dynamics_prior)
readonly -a CONDITIONS=(normal burst_5 burst_20 burst_50)
readonly -a PACKAGE_FILES=(
	tdmpc2/tdmpc2.py
	tdmpc2/common/cutie_latent_belief.py
	tdmpc2/check_cutie_latent_belief_contract.py
	tdmpc2/tools/evaluate_cutie_latent_dynamics_prior.py
	tdmpc2/tools/run_cutie_latent_dynamics_prior_probe.sh
)

for name in GPU_MEASUREMENT GPU_PRIOR; do
	value="${!name}"
	[[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || {
		echo "$name is invalid: $value" >&2
		exit 2
	}
done
[[ "$GPU_MEASUREMENT" != "$GPU_PRIOR" ]] || {
	echo "GPU_MEASUREMENT and GPU_PRIOR must differ." >&2
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
for path in "$VIDEO_ROOT" "$OC_REPO" "$CUTIE_CKPT" "$SOURCE_SUMMARY" \
	"$RUNTIME_CONFIG" "$CHECKPOINT"; do
	[[ -e "$path" ]] || { echo "Missing immutable input: $path" >&2; exit 2; }
done
[[ "$(basename -- "${VIDEO_ROOT%/}")" == video_hard ]] || {
	echo "VIDEO_ROOT must be the frozen video_hard directory: $VIDEO_ROOT" >&2
	exit 2
}
for path in "${PACKAGE_FILES[@]}" \
	tdmpc2/envs/wrappers/cutie_object.py \
	tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py \
	tdmpc2/tools/evaluate_cutie_multitask_policy_burst.py \
	tdmpc2/check_cutie_object_wrapper_contract.py \
	tdmpc2/check_cutie_object_only_contract.py \
	tdmpc2/check_cutie_policy_burst_contract.py \
	tdmpc2/check_cutie_object_only_integration_contract.py; do
	[[ -f "$path" ]] || { echo "Missing repository file: $path" >&2; exit 2; }
done

for path in "$BASE" "$STAGE"; do
	[[ ! -e "$path" ]] || { echo "Refusing existing output: $path" >&2; exit 3; }
done
mkdir -p "$STAGE/contracts" "$STAGE/provenance" "$STAGE/plans/reacher-visual-small"
for condition in "${CONDITIONS[@]}"; do
	mkdir -p "$STAGE/evaluations/$condition"
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
			"$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps({"format":sys.argv[3],"status":"latent_dynamics_prior_probe_engineering_fail","runner_exit_code":int(sys.argv[2]),"failure":"runner exited before complete aggregation","scientific_scope":"single-seed evaluation-only diagnostic; no training"},indent=2)+"\n",encoding="utf-8")' "$SUMMARY" "$rc" "$FORMAT" 2>/dev/null || true
		fi
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
		[[ ! -e "$failed" ]] || failed="${failed}.${RANDOM}"
		mv -- "$STAGE" "$failed"
		echo "CUTIE_LATENT_DYNAMICS_PRIOR_PROBE_FAILED_ARCHIVE=$failed" >&2
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

echo "[1/5] Dependency-light contracts and immutable input binding"
run_contract runner_bash_syntax bash -n "$0"
run_contract evaluator_ast "$PY" -c 'import ast,pathlib; ast.parse(pathlib.Path("tdmpc2/tools/evaluate_cutie_latent_dynamics_prior.py").read_text(encoding="utf-8"))'
run_contract object_wrapper "$PY" tdmpc2/check_cutie_object_wrapper_contract.py
run_contract object_only "$PY" tdmpc2/check_cutie_object_only_contract.py
run_contract policy_burst "$PY" -m tdmpc2.check_cutie_policy_burst_contract
run_contract latent_belief "$PY" -m tdmpc2.check_cutie_latent_belief_contract
run_contract object_only_integration "$PY" tdmpc2/check_cutie_object_only_integration_contract.py

"$PY" - "$REPO_ROOT" "$SOURCE_SUMMARY" "$RUNTIME_CONFIG" "$CHECKPOINT" \
	"$VIDEO_ROOT" "$OC_REPO" "$CUTIE_CKPT" "$STAGE/provenance/inputs.json" \
	"${PACKAGE_FILES[@]}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

repo, source_summary, runtime_config, checkpoint, video_root, oc_repo, cutie_ckpt, output = map(
    Path, sys.argv[1:9]
)
package_files = sys.argv[9:]

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def load(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected object: {path}')
    return value

def same_path(value, expected):
    try:
        return Path(value).resolve() == Path(expected).resolve()
    except (TypeError, ValueError, OSError):
        return False

source = load(source_summary)
if source.get('format') != 'cutie_object_memory_probe_v1':
    raise ValueError('SOURCE_SUMMARY is not the frozen memory probe v1 result.')
if source.get('status') != 'object_memory_probe_engineering_pass':
    raise ValueError('The frozen source memory probe did not pass engineering.')
gates = source.get('engineering_gates')
if not isinstance(gates, dict) or not gates or not all(value is True for value in gates.values()):
    raise ValueError(f'Source engineering gates are not all true: {gates!r}')

runtime = load(runtime_config)
support = Path(runtime.get('cutie_object_support_path', '')).resolve()
if not support.is_file():
    raise FileNotFoundError(f'Frozen runtime support is unavailable: {support}')
source_arm = source.get('tasks', {}).get('reacher-visual-small', {}).get('arms', {}).get('hard_zero', {})
checks = {
    'task_seed': runtime.get('task') == 'reacher-visual-small' and runtime.get('seed') == 7,
    'schedule': runtime.get('steps') == 100000 and runtime.get('eval_freq') == 20000 and runtime.get('eval_episodes') == 3,
    'planning_horizon': runtime.get('horizon') == 3,
    'object_only': runtime.get('flat_anchor') is True and runtime.get('flat_anchor_mode') == 'cutie_object_only' and runtime.get('latent_dim') == 128 and runtime.get('obs_shape') == {'object': [2, 1770]},
    'roles': runtime.get('cutie_object_role_names') == ['whole_arm', 'goal'],
    'memory_off': runtime.get('cutie_object_last_valid_memory') is False,
    'burst_off': 'cutie_object_policy_burst_plan' in runtime and runtime.get('cutie_object_policy_burst_plan') is None,
    'video_hard': runtime.get('video_background_enabled') is True and runtime.get('video_background_split') == 'train' and same_path(runtime.get('video_background_root'), video_root),
    'oc_repo': same_path(runtime.get('cutie_object_repo'), oc_repo),
    'cutie_checkpoint': same_path(runtime.get('cutie_object_checkpoint'), cutie_ckpt),
    'source_checkpoint_path': same_path(source_arm.get('checkpoint'), checkpoint),
    'source_checkpoint_sha': source_arm.get('checkpoint_sha256') == digest(checkpoint),
}
if not all(checks.values()):
    raise ValueError(f'Frozen source mismatch: {[key for key, value in checks.items() if not value]}')

source_eval_root = source_summary.parent / 'tasks' / 'reacher-visual-small' / 'evaluations'
source_evaluations = {}
for condition in ('normal', 'burst_5', 'burst_20', 'burst_50'):
    path = source_eval_root / condition / 'hard_zero.json'
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = load(path)
    episodes = payload.get('episodes')
    if not isinstance(episodes, list) or len(episodes) != 20:
        raise ValueError(f'Frozen source {condition} does not have 20 episodes.')
    rewards = [float(row['reward']) for row in episodes]
    if not all(math.isfinite(value) for value in rewards):
        raise ValueError(f'Frozen source {condition} rewards are non-finite.')
    source_evaluations[condition] = {
        'path': str(path.resolve()), 'sha256': digest(path), 'rewards': rewards,
        'reward_mean': sum(rewards) / len(rewards),
        'policy_burst_plan_sha256': episodes[0].get('policy_burst_plan_sha256'),
    }

implementation = {}
for relative in package_files + [
    'tdmpc2/envs/wrappers/cutie_object.py',
    'tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py',
    'tdmpc2/tools/evaluate_cutie_multitask_policy_burst.py',
]:
    path = repo / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    implementation[relative] = {'path': str(path.resolve()), 'sha256': digest(path)}

starts = [75, 150, 225, 300, 375]
plans = {}
for length in (5, 20, 50):
    payload = {
        'format': 'cutie_policy_burst_plan_v1',
        'task': 'reacher-visual-small',
        'roles': ['whole_arm', 'goal'],
        'episodes': 20,
        'decision_steps': 500,
        'frame_dim': 590,
        'stack_frames': 3,
        'invalid_encoding': 'empty_lost_v1',
        'events': [
            {
                'episode_index': episode,
                'role': 'whole_arm',
                'start_decision_step': starts[episode % len(starts)],
                'length': length,
            }
            for episode in range(20)
        ],
    }
    raw = (json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False) + '\n').encode('utf-8')
    path = output.parent.parent / 'plans' / 'reacher-visual-small' / f'burst_{length}.json'
    with path.open('xb') as file:
        file.write(raw)
    plan_sha = hashlib.sha256(raw).hexdigest()
    if plan_sha != source_evaluations[f'burst_{length}']['policy_burst_plan_sha256']:
        raise ValueError(f'New burst_{length} plan is not canonical-identical to the frozen source.')
    plans[f'burst_{length}'] = {
        'relative_to_summary_root': f'plans/reacher-visual-small/burst_{length}.json',
        'staging_execution_path': str(path.resolve()),
        'staging_execution_path_is_stable_after_promotion': False,
        'sha256': plan_sha,
        'events': payload['events'], 'length': length,
    }

payload = {
    'format': 'cutie_latent_dynamics_prior_probe_inputs_v1',
    'scientific_scope': 'single-seed evaluation-only diagnostic; frozen transition and frozen checkpoint; no training',
    'source_memory_probe': {
        'path': str(source_summary.resolve()), 'sha256': digest(source_summary),
        'format': source.get('format'), 'status': source.get('status'),
        'engineering_gates': gates, 'hard_zero_arm_checks': checks,
        'evaluations': source_evaluations,
    },
    'source_artifacts': {
        'runtime_config': str(runtime_config.resolve()),
        'runtime_config_sha256': digest(runtime_config),
        'checkpoint': str(checkpoint.resolve()),
        'checkpoint_sha256': digest(checkpoint),
    },
    'external_inputs': {
        'video_root': str(video_root.resolve()),
        'oc_repo': str(oc_repo.resolve()),
        'cutie_checkpoint': str(cutie_ckpt.resolve()),
        'cutie_checkpoint_sha256': digest(cutie_ckpt),
        'support': str(support),
        'support_sha256': digest(support),
    },
    'implementation': implementation,
    'package_files_no_tar': package_files,
    'plans': plans,
}
with output.open('x', encoding='utf-8', newline='\n') as file:
    json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
    file.write('\n')
(output.parent / 'package_files.txt').write_text('\n'.join(package_files) + '\n', encoding='utf-8', newline='\n')
print('CUTIE_LATENT_PRIOR_INPUTS_OK', json.dumps({
    'source_summary_sha256': payload['source_memory_probe']['sha256'],
    'checkpoint_sha256': payload['source_artifacts']['checkpoint_sha256'],
    'source_means': {key: value['reward_mean'] for key, value in source_evaluations.items()},
    'plans': {key: value['sha256'] for key, value in plans.items()},
}, allow_nan=False))
PY

run_evaluation() {
	local gpu=$1 arm=$2 condition=$3 length out log rc
	local -a plan=()
	length=0
	if [[ "$condition" != normal ]]; then
		length="${condition#burst_}"
		plan=(--policy-burst-plan "$STAGE/plans/reacher-visual-small/${condition}.json")
	fi
	out="$STAGE/evaluations/$condition/$arm.json"
	log="$STAGE/evaluations/$condition/$arm.log"
	[[ ! -e "$out" && ! -e "${out}.incomplete" ]] || return 3
	echo "EVAL_START arm=$arm condition=$condition gpu=$gpu" | tee "$log"
	set +e
	env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
		"$PY" -m tdmpc2.tools.evaluate_cutie_latent_dynamics_prior \
		--arm "$arm" --condition "$condition" "${plan[@]}" \
		--expected-length "$length" --runtime-config "$RUNTIME_CONFIG" \
		--checkpoint "$CHECKPOINT" --training-seed "$SEED" \
		--expected-training-steps "$STEPS" --expected-training-eval-freq "$EVAL_FREQ" \
		--expected-training-eval-episodes "$TRAIN_EVAL_EPISODES" \
		--env-seed "$ENV_SEED" --background-seed "$BACKGROUND_SEED" \
		--planner-seed-base "$PLANNER_SEED_BASE" --output "$out" >>"$log" 2>&1
	rc=$?
	set -e
	write_rc "$STAGE/evaluations/$condition/$arm.rc" "$rc"
	echo "EVAL_END arm=$arm condition=$condition gpu=$gpu rc=$rc" | tee -a "$log"
	return "$rc"
}

run_arm() {
	local gpu=$1 arm=$2 condition rc=0
	printf '%s\n' "$gpu" >"$STAGE/${arm}.gpu"
	for condition in "${CONDITIONS[@]}"; do
		if run_evaluation "$gpu" "$arm" "$condition"; then
			continue
		else
			rc=$?
			for remaining in "${CONDITIONS[@]}"; do
				if [[ ! -f "$STAGE/evaluations/$remaining/$arm.rc" ]]; then
					echo "Skipped after prior evaluation failure." >"$STAGE/evaluations/$remaining/$arm.log"
					write_rc "$STAGE/evaluations/$remaining/$arm.rc" 125
				fi
			done
			return "$rc"
		fi
	done
}

echo "[2/5] Parallel frozen-checkpoint evaluation (no training)"
echo "GPU $GPU_MEASUREMENT: measurement_only normal + burst 5/20/50"
echo "GPU $GPU_PRIOR: dynamics_prior normal + burst 5/20/50"
run_arm "$GPU_MEASUREMENT" measurement_only & PID_MEASUREMENT=$!
ACTIVE_PIDS+=("$PID_MEASUREMENT")
run_arm "$GPU_PRIOR" dynamics_prior & PID_PRIOR=$!
ACTIVE_PIDS+=("$PID_PRIOR")
set +e
wait "$PID_MEASUREMENT"; MEASUREMENT_RC=$?
wait "$PID_PRIOR"; PRIOR_RC=$?
set -e
ACTIVE_PIDS=()
write_rc "$STAGE/measurement_only.worker.rc" "$MEASUREMENT_RC"
write_rc "$STAGE/dynamics_prior.worker.rc" "$PRIOR_RC"

echo "[3/5] Strict aggregation, source regression, pairing, and development GO"
set +e
"$PY" - "$STAGE" "$SUMMARY" "$REPO_ROOT" "$START_SECONDS" \
	"$GPU_MEASUREMENT" "$GPU_PRIOR" "$SOURCE_SUMMARY" "$RUNTIME_CONFIG" \
	"$CHECKPOINT" <<'PY'
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

stage, summary_path, repo = map(Path, sys.argv[1:4])
started = int(sys.argv[4])
gpu_measurement, gpu_prior = sys.argv[5:7]
source_summary, runtime_config, checkpoint = map(Path, sys.argv[7:10])
inputs_path = stage / 'provenance' / 'inputs.json'
arms = ('measurement_only', 'dynamics_prior')
conditions = ('normal', 'burst_5', 'burst_20', 'burst_50')
gpu_by_arm = {'measurement_only': gpu_measurement, 'dynamics_prior': gpu_prior}
length_by_condition = {'normal': 0, 'burst_5': 5, 'burst_20': 20, 'burst_50': 50}
expected_evaluator = repo / 'tdmpc2' / 'tools' / 'evaluate_cutie_latent_dynamics_prior.py'
pair_fields = (
    'initial_rgb_sha256', 'initial_policy_object_sha256',
    'initial_raw_object_frame_sha256', 'background_source',
    'background_start_frame_index', 'background_end_source',
    'background_end_frame_index', 'planner_seed',
    'planner_rng_start_sha256', 'planner_rng_end_sha256', 'length',
    'policy_burst_event', 'policy_burst_plan_sha256',
)

def load(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected object: {path}')
    return value

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

def read_rc(path):
    try:
        return int(path.read_text(encoding='utf-8').strip())
    except Exception:
        return None

def all_true(mapping):
    return isinstance(mapping, dict) and bool(mapping) and all(value is True for value in mapping.values())

def paired_stats(left, right):
    if len(left) != 20 or len(right) != 20:
        raise ValueError('Paired reward vectors must contain 20 episodes each.')
    delta = [b - a for a, b in zip(left, right)]
    mean = statistics.fmean(delta)
    sd = statistics.stdev(delta)
    half = 2.093024054408263 * sd / math.sqrt(20)
    return {
        'measurement_only_reward_mean': statistics.fmean(left),
        'dynamics_prior_reward_mean': statistics.fmean(right),
        'dynamics_prior_minus_measurement_only_mean': mean,
        'dynamics_prior_minus_measurement_only_median': statistics.median(delta),
        'paired_delta_sample_std': sd,
        'paired_delta_95pct_t_interval_df19': [mean - half, mean + half],
        'win_tie_loss': [sum(x > 0 for x in delta), sum(x == 0 for x in delta), sum(x < 0 for x in delta)],
        'paired_deltas': delta,
    }

jobs, structure, pairing_failures, regression_failures = [], [], [], []
reports, payloads = {}, {}
try:
    inputs = load(inputs_path)
except Exception as exc:
    inputs = {}
    structure.append(f'input provenance unavailable: {exc}')

# Recheck every immutable source and implementation input after GPU work.
immutable = {}
try:
    source_record = inputs['source_memory_probe']
    artifacts = inputs['source_artifacts']
    immutable['source_summary_path'] = same_path(source_record['path'], source_summary)
    immutable['source_summary_sha256'] = digest(source_summary) == source_record['sha256']
    immutable['runtime_config_path'] = same_path(artifacts['runtime_config'], runtime_config)
    immutable['runtime_config_sha256'] = digest(runtime_config) == artifacts['runtime_config_sha256']
    immutable['checkpoint_path'] = same_path(artifacts['checkpoint'], checkpoint)
    immutable['checkpoint_sha256'] = digest(checkpoint) == artifacts['checkpoint_sha256']
    external = inputs['external_inputs']
    external_cutie = Path(external['cutie_checkpoint'])
    external_support = Path(external['support'])
    immutable['external_video_root'] = Path(external['video_root']).is_dir() and Path(external['video_root']).name == 'video_hard'
    immutable['external_oc_repo'] = Path(external['oc_repo']).is_dir()
    immutable['external_cutie_checkpoint'] = external_cutie.is_file() and digest(external_cutie) == external['cutie_checkpoint_sha256']
    immutable['external_support'] = external_support.is_file() and digest(external_support) == external['support_sha256']
    for relative, record in inputs['implementation'].items():
        path = repo / relative
        immutable[f'implementation:{relative}'] = same_path(record['path'], path) and digest(path) == record['sha256']
    for condition, record in inputs['plans'].items():
        relative = f'plans/reacher-visual-small/{condition}.json'
        path = stage / relative
        immutable[f'plan:{condition}:relative'] = record.get('relative_to_summary_root') == relative
        immutable[f'plan:{condition}:staging_execution_path'] = (
            record.get('staging_execution_path_is_stable_after_promotion') is False
            and same_path(record.get('staging_execution_path'), path)
        )
        immutable[f'plan:{condition}:sha256'] = digest(path) == record['sha256']
    for condition, record in source_record['evaluations'].items():
        path = Path(record['path'])
        immutable[f'source_eval:{condition}'] = path.is_file() and digest(path) == record['sha256']
    if not all(immutable.values()):
        structure.append('immutable inputs changed: ' + ', '.join(key for key, value in immutable.items() if not value))
except Exception as exc:
    immutable['recheck_parse'] = False
    structure.append(f'immutable input recheck failed: {exc}')

evaluator_hashes, runtime_hashes, checkpoint_hashes, manifest_hashes, combined_hashes, device_names = (set() for _ in range(6))
for condition in conditions:
    length = length_by_condition[condition]
    reports[condition] = {'length': length, 'arms': {}, 'pairing': {}}
    payloads[condition] = {}
    for arm in arms:
        path = stage / 'evaluations' / condition / f'{arm}.json'
        rc = read_rc(stage / 'evaluations' / condition / f'{arm}.rc')
        report = {
            'rc': rc,
            'evaluation_relative_to_summary_root': f'evaluations/{condition}/{arm}.json',
            'staging_execution_output': str(path),
            'staging_execution_output_is_stable_after_promotion': False,
        }
        reports[condition]['arms'][arm] = report
        if rc != 0:
            jobs.append(f'{condition}/{arm}: rc={rc}')
            continue
        if not path.is_file():
            structure.append(f'{condition}/{arm}: output missing')
            continue
        try:
            value = load(path)
            episodes = value.get('episodes')
            protocol = value.get('evaluation', {})
            provenance = value.get('provenance', {})
            belief = value.get('latent_belief', {})
            metrics = belief.get('metrics', {})
            perception = value.get('perception_runtime', {})
            intervention = perception.get('policy_observation_intervention', {})
            memory = perception.get('last_valid_memory', {})
            cutie_inputs = provenance.get('cutie_inputs', {})
            cutie_ready = provenance.get('cutie_ready', {})
            plan_record = inputs.get('plans', {}).get(condition)
            expected_plan_sha = plan_record.get('sha256') if isinstance(plan_record, dict) else None
            expected_plan = stage / 'plans' / 'reacher-visual-small' / f'{condition}.json' if length else None
            parity = protocol.get('planner_from_latent_parity')
            implementation = inputs.get('implementation', {})
            agent_record = implementation.get('tdmpc2/tdmpc2.py', {})
            belief_record = implementation.get('tdmpc2/common/cutie_latent_belief.py', {})
            wrapper_record = implementation.get('tdmpc2/envs/wrappers/cutie_object.py', {})
            base_record = implementation.get('tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py', {})
            burst_record = implementation.get('tdmpc2/tools/evaluate_cutie_multitask_policy_burst.py', {})
            external = inputs.get('external_inputs', {})
            eval_checks = {
                'format': value.get('format') == 'cutie_latent_dynamics_prior_evaluation_v1',
                'identity': value.get('task') == 'reacher-visual-small' and value.get('backend') == 'cutie_object_only' and value.get('arm') == arm and value.get('condition') == condition and value.get('training_seed') == 7,
                'episodes': isinstance(episodes, list) and len(episodes) == 20 and [row.get('episode_index') for row in episodes] == list(range(20)),
                'protocol': protocol.get('split') == 'validation' and protocol.get('episodes') == 20 and protocol.get('env_seed') == 424243 and protocol.get('background_seed') == 1618034 and protocol.get('planner_seed_base') == 8675400 and protocol.get('object_only_alignment_draws') == 10000,
                'planner_parity': all_true(parity),
                'source_runtime': same_path(provenance.get('runtime_config'), runtime_config) and provenance.get('runtime_config_sha256') == digest(runtime_config) and provenance.get('runtime_config_sha256_after') == digest(runtime_config),
                'source_checkpoint': same_path(provenance.get('checkpoint'), checkpoint) and provenance.get('checkpoint_sha256') == digest(checkpoint) and provenance.get('checkpoint_sha256_after') == digest(checkpoint),
                'evaluator': same_path(provenance.get('evaluator'), expected_evaluator) and provenance.get('evaluator_sha256') == digest(expected_evaluator),
                'base_evaluator': same_path(provenance.get('base_evaluator'), base_record.get('path')) and provenance.get('base_evaluator_sha256') == base_record.get('sha256'),
                'burst_evaluator': same_path(provenance.get('burst_evaluator'), burst_record.get('path')) and provenance.get('burst_evaluator_sha256') == burst_record.get('sha256'),
                'agent_implementation': same_path(provenance.get('agent_implementation'), agent_record.get('path')) and provenance.get('agent_implementation_sha256') == agent_record.get('sha256') and provenance.get('agent_implementation_sha256_after') == agent_record.get('sha256'),
                'belief_implementation': same_path(provenance.get('latent_belief_implementation'), belief_record.get('path')) and provenance.get('latent_belief_implementation_sha256') == belief_record.get('sha256') and provenance.get('latent_belief_implementation_sha256_after') == belief_record.get('sha256'),
                'wrapper_implementation': same_path(provenance.get('object_wrapper_implementation'), wrapper_record.get('path')) and provenance.get('object_wrapper_implementation_sha256') == wrapper_record.get('sha256') and provenance.get('object_wrapper_implementation_sha256_after') == wrapper_record.get('sha256'),
                'source_flags': provenance.get('source_flags') == {'cutie_object_last_valid_memory': False, 'cutie_object_policy_burst_plan': None},
                'assigned_gpu': provenance.get('cuda_visible_devices') == gpu_by_arm[arm],
                'cutie_inputs': same_path(cutie_inputs.get('checkpoint'), external.get('cutie_checkpoint')) and cutie_inputs.get('checkpoint_sha256') == external.get('cutie_checkpoint_sha256') and same_path(cutie_inputs.get('support'), external.get('support')) and cutie_inputs.get('support_sha256') == external.get('support_sha256') and cutie_inputs.get('roles') == ['whole_arm', 'goal'] and cutie_inputs.get('support_schema') == 'generic_indexed_v1' and cutie_ready.get('allow_simulator_support') is True,
                'belief_definition': belief.get('mode') == arm and belief.get('target_role') == 'whole_arm' and belief.get('target_latent_slice') == [0, 64] and belief.get('non_target_latent_slice') == [64, 128] and belief.get('uses_executed_previous_action') is True and belief.get('trainable_parameters_added') == 0 and belief.get('hidden_raw_features_available_to_policy') is False,
                'runtime': perception.get('frames') == 10020 and perception.get('worker_restarts') == 0 and perception.get('timeouts') == 0 and math.isfinite(float(perception.get('valid_frame_rate', float('nan')))) and math.isfinite(float(perception.get('ms_per_frame', float('nan')))) and 0 < float(perception.get('ms_per_frame', 0)) <= 800 and perception.get('runtime_unit') == 'milliseconds_per_tracked_frame_excluding_support_prompts' and perception.get('episode_reset_strategy') == 'fresh_inference_core_support_replay_v1',
                'memory_disabled': memory.get('enabled') is False and memory.get('substitutions') == 0 and memory.get('invalid_without_history') == 0,
                'raw_accounting_and_live_observation': intervention.get('raw_tracker_accounting_excludes_synthetic_intervention') is True and intervention.get('live_environment_observation_mutated') is False,
                'strict_checks': all_true(value.get('strict_checks')),
            }
            if length == 0:
                eval_checks['plan_disabled'] = provenance.get('policy_burst_plan') is None and provenance.get('policy_burst_plan_sha256') is None and intervention.get('enabled') is False and intervention.get('applied_role_frames') == 0
            else:
                relative_from_output = provenance.get('policy_burst_plan_relative_to_output')
                resolved_relative = (
                    (path.parent / relative_from_output).resolve()
                    if isinstance(relative_from_output, str) else None
                )
                eval_checks['plan_exact'] = same_path(provenance.get('policy_burst_plan'), expected_plan) and resolved_relative == expected_plan.resolve() and provenance.get('policy_burst_plan_sha256') == expected_plan_sha and value.get('policy_burst', {}).get('plan_sha256_before') == expected_plan_sha and value.get('policy_burst', {}).get('plan_sha256_after') == expected_plan_sha and value.get('policy_burst', {}).get('wrapper_plan_sha256') == expected_plan_sha
                eval_checks['intervention_exact'] = intervention.get('enabled') is True and intervention.get('scheduled_events') == 20 and intervention.get('applied_events') == 20 and intervention.get('scheduled_role_frames') == 20 * length and intervention.get('applied_role_frames') == 20 * length and intervention.get('exact_invalid_checks') == 20 * length and intervention.get('non_target_preserved_checks') == 20 * length
                eval_checks['coverage'] = value.get('policy_burst', {}).get('controlled_burst_attribution_eligible') is True and float(value.get('policy_burst', {}).get('raw_valid_overwrite_rate', -1)) >= 0.95
            if arm == 'measurement_only':
                eval_checks['measurement_control'] = metrics.get('prior_computed_steps') == 0 and metrics.get('prior_used_steps') == 0 and metrics.get('synthetic_prior_used_steps') == 0
            else:
                eval_checks['prior_calls'] = metrics.get('prior_computed_steps') == 9980 and metrics.get('synthetic_prior_used_steps') == 20 * length and metrics.get('invalid_without_prior') == 0
            if not all(eval_checks.values()):
                structure.append(f'{condition}/{arm}: checks failed ' + str([key for key, item in eval_checks.items() if not item]))
            rewards = [float(row['reward']) for row in episodes] if isinstance(episodes, list) else []
            if len(rewards) != 20 or not all(math.isfinite(item) for item in rewards):
                raise ValueError('reward vector is incomplete or non-finite')
            report.update({
                'sha256': digest(path), 'checks': eval_checks,
                'reward_mean': statistics.fmean(rewards),
                'reward_median': statistics.median(rewards),
                'reward_sample_std': statistics.stdev(rewards),
                'rewards': rewards, 'belief_metrics': metrics,
                'policy_burst': value.get('policy_burst'),
                'perception_runtime': perception,
                'elapsed_seconds': value.get('summary', {}).get('elapsed_seconds'),
            })
            payloads[condition][arm] = value
            evaluator_hashes.add(provenance.get('evaluator_sha256'))
            runtime_hashes.add(provenance.get('runtime_config_sha256'))
            checkpoint_hashes.add(provenance.get('checkpoint_sha256'))
            manifest_hashes.add(provenance.get('validation_manifest_sha256'))
            combined_hashes.add(provenance.get('combined_manifest_sha256'))
            device_names.add(provenance.get('device_name'))
        except Exception as exc:
            structure.append(f'{condition}/{arm}: parse failed: {exc}')

    if set(payloads[condition]) == set(arms):
        left = payloads[condition]['measurement_only']['episodes']
        right = payloads[condition]['dynamics_prior']['episodes']
        mismatches = {
            field: [index for index, (a, b) in enumerate(zip(left, right)) if a.get(field) != b.get(field)]
            for field in pair_fields
        }
        exact = not any(mismatches.values())
        reports[condition]['pairing'] = {'exact': exact, 'fields': list(pair_fields), 'mismatch_episode_indices': mismatches}
        if not exact:
            pairing_failures.append(f'{condition}: {mismatches}')
    else:
        reports[condition]['pairing'] = {'exact': False, 'failure': f'incomplete arms {sorted(payloads[condition])}'}
        pairing_failures.append(f'{condition}: incomplete arms {sorted(payloads[condition])}')

# When Reacher has no natural whole-arm invalidity, the prior arm never fuses a
# prior and is required to be an exact identity control, not merely similar.
if set(payloads.get('normal', {})) == set(arms):
    left_value = payloads['normal']['measurement_only']
    right_value = payloads['normal']['dynamics_prior']
    left_invalid = int(left_value['latent_belief']['metrics']['target_invalid_steps'])
    right_invalid = int(right_value['latent_belief']['metrics']['target_invalid_steps'])
    identity = {
        'measurement_only_target_invalid_steps': left_invalid,
        'dynamics_prior_target_invalid_steps': right_invalid,
        'exact_identity_required': left_invalid == 0 and right_invalid == 0,
    }
    if identity['exact_identity_required']:
        left_rows, right_rows = left_value['episodes'], right_value['episodes']
        identity_mismatches = {
            field: [
                index for index, (left, right) in enumerate(zip(left_rows, right_rows))
                if left.get(field) != right.get(field)
            ]
            for field in ('reward', 'action_trace_sha256', 'belief_trace_sha256')
        }
        identity['mismatch_episode_indices'] = identity_mismatches
        identity['exact'] = not any(identity_mismatches.values())
        if not identity['exact']:
            pairing_failures.append(f'normal no-intervention identity failed: {identity_mismatches}')
    else:
        identity.update({
            'exact': None,
            'reason': 'natural target invalidity activated the prior; reward identity is not expected',
            'dynamics_prior_natural_prior_used_steps': int(
                right_value['latent_belief']['metrics']['natural_prior_used_steps']
            ),
        })
    reports['normal']['no_intervention_identity'] = identity
else:
    reports['normal']['no_intervention_identity'] = {
        'exact_identity_required': None, 'exact': False,
        'reason': 'normal arms are incomplete',
    }
    pairing_failures.append('normal no-intervention identity could not be checked')

# The identity arm must exactly reproduce the already-completed hard-zero run.
for condition in conditions:
    try:
        source_record = inputs['source_memory_probe']['evaluations'][condition]
        source_value = load(Path(source_record['path']))
        old_rows = source_value['episodes']
        new_rows = payloads[condition]['measurement_only']['episodes']
        old_rewards = [float(row['reward']) for row in old_rows]
        new_rewards = [float(row['reward']) for row in new_rows]
        reward_exact = old_rewards == new_rewards
        common_fields = [field for field in pair_fields if field in old_rows[0]]
        field_mismatches = {
            field: [index for index, (old, new) in enumerate(zip(old_rows, new_rows)) if old.get(field) != new.get(field)]
            for field in common_fields
        }
        exact = reward_exact and not any(field_mismatches.values())
        reports[condition]['source_hard_zero_regression'] = {
            'exact': exact, 'source_sha256': source_record['sha256'],
            'reward_vector_bitwise_numeric_equal': reward_exact,
            'common_pair_fields': common_fields,
            'mismatch_episode_indices': field_mismatches,
        }
        if not exact:
            regression_failures.append(f'{condition}: identity arm did not reproduce source')
    except Exception as exc:
        regression_failures.append(f'{condition}: regression parse failed: {exc}')

worker_rcs = {
    'measurement_only': read_rc(stage / 'measurement_only.worker.rc'),
    'dynamics_prior': read_rc(stage / 'dynamics_prior.worker.rc'),
}
if any(value != 0 for value in worker_rcs.values()):
    jobs.append(f'worker return codes {worker_rcs}')
cross_checks = {
    'one_evaluator_hash': evaluator_hashes == {digest(expected_evaluator)},
    'one_runtime_hash': runtime_hashes == {digest(runtime_config)},
    'one_checkpoint_hash': checkpoint_hashes == {digest(checkpoint)},
    'one_validation_manifest_hash': len(manifest_hashes) == 1 and None not in manifest_hashes,
    'one_combined_manifest_hash': len(combined_hashes) == 1 and None not in combined_hashes,
    'matched_rtx4090_model': len(device_names) == 1 and None not in device_names and all('RTX 4090' in str(value) for value in device_names),
}
if not all(cross_checks.values()):
    structure.append('cross-evaluation checks failed: ' + str([key for key, value in cross_checks.items() if not value]))

engineering = {
    'dependency_light_contracts': True,
    'all_evaluations_completed': not jobs,
    'immutable_source_and_implementation': bool(immutable) and all(immutable.values()),
    'artifact_and_protocol_structure': not structure,
    'strict_arm_pairing': not pairing_failures,
    'measurement_only_reproduces_frozen_hard_zero': not regression_failures,
    'matched_gpu_model': cross_checks.get('matched_rtx4090_model') is True,
}
engineering_pass = all(engineering.values())

outcomes = {}
for condition in conditions:
    left = reports[condition]['arms']['measurement_only'].get('rewards')
    right = reports[condition]['arms']['dynamics_prior'].get('rewards')
    outcomes[condition] = paired_stats(left, right) if isinstance(left, list) and isinstance(right, list) else {'complete': False}
normal = outcomes.get('normal', {})
normal_left = normal.get('measurement_only_reward_mean')
normal_right = normal.get('dynamics_prior_reward_mean')
normal_retention = normal_right / normal_left if isinstance(normal_left, (int, float)) and normal_left > 0 else None
normal_gate = normal_retention is not None and normal_retention >= 0.95
normal_metrics = reports['normal']['arms']['dynamics_prior'].get('belief_metrics', {})
normal_prior_mse = normal_metrics.get('valid_prior_role_mse_mean')
normal_persistence_mse = normal_metrics.get('valid_persistence_role_mse_mean')
normal_prediction_improvement = (
    1.0 - normal_prior_mse / normal_persistence_mse
    if isinstance(normal_prior_mse, (int, float))
    and isinstance(normal_persistence_mse, (int, float))
    and math.isfinite(normal_prior_mse) and math.isfinite(normal_persistence_mse)
    and normal_persistence_mse > 0 else None
)
normal_prediction_required_comparisons = math.ceil(20 * 499 * 0.95)
normal_prediction_comparisons = normal_metrics.get('valid_persistence_comparisons')
normal_prediction_coverage_gate = (
    isinstance(normal_prediction_comparisons, int)
    and normal_prediction_comparisons >= normal_prediction_required_comparisons
    and normal_metrics.get('valid_prior_comparisons')
    == normal_prediction_comparisons
)
normal_prediction_gate = (
    normal_prediction_coverage_gate
    and
    normal_prediction_improvement is not None
    and normal_prediction_improvement >= 0.20
)

burst_diagnostics = {}
for condition in ('burst_5', 'burst_20', 'burst_50'):
    stats = outcomes.get(condition, {})
    baseline = stats.get('measurement_only_reward_mean')
    prior = stats.get('dynamics_prior_reward_mean')
    lost = max(0.0, normal_left - baseline) if isinstance(normal_left, (int, float)) and isinstance(baseline, (int, float)) else None
    metrics = reports[condition]['arms']['dynamics_prior'].get('belief_metrics', {})
    controlled_reacquisitions = [
        row for row in metrics.get('reacquisitions', [])
        if row.get('invalid_streak') == length_by_condition[condition]
        and row.get('synthetic_steps') == length_by_condition[condition]
        and row.get('natural_steps') == 0
    ]
    prior_mse = (
        statistics.fmean(float(row['prior_role_mse']) for row in controlled_reacquisitions)
        if controlled_reacquisitions else None
    )
    persistence_mse = (
        statistics.fmean(float(row['persistence_role_mse']) for row in controlled_reacquisitions)
        if controlled_reacquisitions else None
    )
    reacquisition_exact = len(controlled_reacquisitions) == 20
    prediction_gate = isinstance(prior_mse, (int, float)) and isinstance(persistence_mse, (int, float)) and math.isfinite(prior_mse) and math.isfinite(persistence_mse) and prior_mse < persistence_mse
    coverage = reports[condition]['arms']['dynamics_prior'].get('policy_burst', {}).get('controlled_burst_attribution_eligible') is True and reports[condition]['arms']['measurement_only'].get('policy_burst', {}).get('controlled_burst_attribution_eligible') is True
    recovery_fraction = (
        (prior - baseline) / lost
        if isinstance(prior, (int, float)) and isinstance(baseline, (int, float))
        and isinstance(lost, (int, float)) and lost > 0 else None
    )
    wins = stats.get('win_tie_loss', [0, 0, 0])[0]
    burst_diagnostics[condition] = {
        'normal_measurement_only_mean': normal_left,
        'measurement_only_burst_mean': baseline,
        'dynamics_prior_burst_mean': prior,
        'measurement_only_drop_from_normal': lost,
        'dynamics_prior_minus_measurement_only_mean': (
            prior - baseline
            if isinstance(prior, (int, float)) and isinstance(baseline, (int, float))
            else None
        ),
        'recovery_fraction_of_measurement_only_drop': recovery_fraction,
        'paired_wins': wins,
        'controlled_burst_attribution_eligible': coverage,
        'controlled_reacquisition_count': len(controlled_reacquisitions),
        'controlled_reacquisitions_exactly_20': reacquisition_exact,
        'reacquisition_prior_mse': prior_mse,
        'reacquisition_persistence_mse': persistence_mse,
        'prior_better_than_persistence_at_reacquisition': prediction_gate,
        'prediction_diagnostic_pass': coverage and reacquisition_exact and prediction_gate,
    }

burst20 = burst_diagnostics['burst_20']
burst20_gate = (
    burst20['prediction_diagnostic_pass']
    and isinstance(burst20['dynamics_prior_minus_measurement_only_mean'], (int, float))
    and burst20['dynamics_prior_minus_measurement_only_mean'] > 0
    and isinstance(burst20['recovery_fraction_of_measurement_only_drop'], (int, float))
    and burst20['recovery_fraction_of_measurement_only_drop'] >= 0.30
    and burst20['paired_wins'] >= 12
)
burst20.update({
    'investment_required_minimum_recovery_fraction': 0.30,
    'investment_required_minimum_paired_wins': 12,
    'investment_gate_pass': burst20_gate,
})

# Fifty missing frames are far beyond the source model's training horizon of
# three. This is deliberately a stronger secondary result, never a veto on
# whether to invest in a real learned filter.
burst50 = burst_diagnostics['burst_50']
strong_long_horizon_gate = (
    burst50['prediction_diagnostic_pass']
    and isinstance(burst50['dynamics_prior_minus_measurement_only_mean'], (int, float))
    and burst50['dynamics_prior_minus_measurement_only_mean'] > 0
    and isinstance(burst50['recovery_fraction_of_measurement_only_drop'], (int, float))
    and burst50['recovery_fraction_of_measurement_only_drop'] >= 0.50
    and burst50['paired_wins'] >= 12
)
burst50.update({
    'strong_required_minimum_recovery_fraction': 0.50,
    'strong_required_minimum_paired_wins': 12,
    'strong_long_horizon_gate_pass': strong_long_horizon_gate,
    'included_in_investment_go': False,
})
burst_diagnostics['burst_5']['reward_improvement_required'] = False
burst_diagnostics['burst_5']['included_in_investment_go'] = False

investment_go = bool(
    engineering_pass and normal_gate and normal_prediction_gate and burst20_gate
)

summary = {
    'format': 'cutie_latent_dynamics_prior_probe_v1',
    'status': 'latent_dynamics_prior_probe_engineering_pass' if engineering_pass else 'latent_dynamics_prior_probe_engineering_fail',
    'scientific_scope': 'single-training-seed, evaluation-only diagnostic using one frozen seed-7 ObjectOnly checkpoint and its frozen TD-MPC2 transition; simulator-derived support and synthetic tracker-output bursts; no new parameters, no training, not a learned belief, not a paper claim',
    'protocol': {
        'task': 'reacher-visual-small', 'training_seed': 7,
        'source_planning_horizon': 3,
        'arms': {
            'measurement_only': 'current encoder measurement at every decision; identity control',
            'dynamics_prior': 'current measurement when valid; frozen action-conditioned role prior only when whole_arm is invalid',
        },
        'conditions': list(conditions), 'episodes': 20,
        'heldout': {'split': 'validation', 'env_seed': 424243, 'background_seed': 1618034, 'planner_seed_base': 8675400},
        'physical_gpu_by_arm': gpu_by_arm,
        'strict_pairing_fields': list(pair_fields),
        'training_performed': False,
    },
    'input_provenance': inputs,
    'immutable_input_recheck': immutable,
    'cross_evaluation_checks': cross_checks,
    'engineering_gates': engineering,
    'failures': {
        'jobs': jobs, 'structure': structure, 'pairing': pairing_failures,
        'source_regression': regression_failures,
    },
    'scientific_outcomes': {
        'scope': 'conditional paired-episode development evidence for one frozen training seed',
        'conditions': outcomes,
        'normal_retention': {
            'dynamics_prior_over_measurement_only': normal_retention,
            'required_minimum': 0.95, 'pass': normal_gate,
        },
        'normal_one_step_prediction': {
            'valid_prior_comparisons': normal_metrics.get('valid_prior_comparisons'),
            'valid_persistence_comparisons': normal_metrics.get('valid_persistence_comparisons'),
            'required_minimum_comparisons': normal_prediction_required_comparisons,
            'comparison_coverage_pass': normal_prediction_coverage_gate,
            'prior_role_mse_mean': normal_prior_mse,
            'persistence_role_mse_mean': normal_persistence_mse,
            'relative_mse_improvement': normal_prediction_improvement,
            'required_minimum_relative_improvement': 0.20,
            'pass': normal_prediction_gate,
        },
        'burst_diagnostics': burst_diagnostics,
        'investment_go': investment_go,
        'strong_long_horizon_gate': strong_long_horizon_gate,
        'decision_rule': (
            'investment_go requires engineering pass, >=95% normal reward retention, '
            '>=20% normal one-step MSE improvement over persistence, and burst20 '
            'positive reward with >=30% loss recovery, >=12 paired wins, exact 20 '
            'controlled reacquisitions, and prior MSE below persistence; burst50 is '
            'reported separately because 50 exceeds the training horizon 3'
        ),
    },
    'recommendation': (
        'go_build_and_train_a_real_learned_belief_filter'
        if investment_go else (
            'no_go_do_not_scale_this_frozen_dynamics_prior_yet'
            if engineering_pass else 'fix_engineering_before_reward_interpretation'
        )
    ),
    'worker_return_codes': worker_rcs,
    'elapsed_seconds': int(time.time()) - started,
    'evaluation_reports': reports,
}
temporary = summary_path.with_name(summary_path.name + '.tmp')
with temporary.open('x', encoding='utf-8', newline='\n') as file:
    json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    file.write('\n')
temporary.replace(summary_path)
print(json.dumps({
    'status': summary['status'], 'engineering_gates': engineering,
    'scientific_outcomes': summary['scientific_outcomes'],
    'recommendation': summary['recommendation'], 'summary': str(summary_path),
}, ensure_ascii=False, indent=2, allow_nan=False))
raise SystemExit(0 if engineering_pass else 4)
PY
SUMMARY_RC=$?
set -e
if (( SUMMARY_RC != 0 )); then
	exit "$SUMMARY_RC"
fi

echo "[4/5] Promoting immutable diagnostic result"
mv -- "$STAGE" "$BASE"
PROMOTED=1
trap - EXIT INT TERM

echo "[5/5] Complete"
echo "CUTIE_LATENT_DYNAMICS_PRIOR_PROBE_COMPLETE"
echo "SUMMARY=$BASE/latent_dynamics_prior_summary.json"
echo "PACKAGE_FILES=$BASE/provenance/package_files.txt"
