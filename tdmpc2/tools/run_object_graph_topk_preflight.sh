#!/usr/bin/env bash
# CPU-only, controller-free top-K candidate-coverage preflight.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${SOURCE_BENCHMARK_ROOT:?Set SOURCE_BENCHMARK_ROOT to the published unified-VOS root}"
: "${V1_PREFLIGHT_ROOT:?Set V1_PREFLIGHT_ROOT to the published v1 object-graph preflight root}"

PY_REPLAY="${PY_TOPK:-${PY_CORE:-python}}"
TRACKED_TIMEOUT_SECONDS="${TRACKED_TIMEOUT_SECONDS:-7200}"
RUN_TAG="${RUN_TAG:-object_graph_topk_candidate_preflight_v1}"
BASE="${OUTPUT_ROOT:-$REPO_ROOT/logs/_diagnostic/$RUN_TAG}"

[[ "$TRACKED_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
	echo "TRACKED_TIMEOUT_SECONDS must be a positive integer." >&2
	exit 2
}

if [[ "$PY_REPLAY" == */* ]]; then
	[[ -x "$PY_REPLAY" ]] || { echo "Python is not executable: $PY_REPLAY" >&2; exit 2; }
else
	PY_REPLAY="$(command -v -- "$PY_REPLAY")" || {
		echo "Python is not on PATH." >&2
		exit 2
	}
fi
PY_REPLAY="$("$PY_REPLAY" -I -c 'import sys; from pathlib import Path; print(Path(sys.executable).resolve(strict=True))')"
[[ -x "$PY_REPLAY" ]] || { echo "Resolved Python is not executable: $PY_REPLAY" >&2; exit 2; }

for path in "$SOURCE_BENCHMARK_ROOT" "$V1_PREFLIGHT_ROOT"; do
	[[ -d "$path" ]] || { echo "Missing directory: $path" >&2; exit 2; }
done
SOURCE_ROOT_CANONICAL="$("$PY_REPLAY" -I -c 'import sys; from pathlib import Path
p=Path(sys.argv[1]).expanduser().absolute()
for component in (*reversed(p.parents),p):
 if component.is_symlink(): raise ValueError(f"Source root contains a symlink component: {component}")
print(p.resolve(strict=True))' "$SOURCE_BENCHMARK_ROOT")"
V1_ROOT_CANONICAL="$("$PY_REPLAY" -I -c 'import sys; from pathlib import Path
p=Path(sys.argv[1]).expanduser().absolute()
for component in (*reversed(p.parents),p):
 if component.is_symlink(): raise ValueError(f"v1 root contains a symlink component: {component}")
print(p.resolve(strict=True))' "$V1_PREFLIGHT_ROOT")"
SOURCE_BENCHMARK_ROOT="$SOURCE_ROOT_CANONICAL"
V1_PREFLIGHT_ROOT="$V1_ROOT_CANONICAL"

STAGE="${BASE}.incomplete"
V1_GRAPH="$REPO_ROOT/tdmpc2/object_graphs/acrobot_swingup.json"
BACKEND_INPUTS="$SOURCE_BENCHMARK_ROOT/worker_inputs/backend_inputs.json"
DATASET_MANIFEST="$SOURCE_BENCHMARK_ROOT/dataset/dataset_manifest.json"
SCORING_ROOT="$SOURCE_BENCHMARK_ROOT/dataset/scoring"
V1_SUMMARY="$V1_PREFLIGHT_ROOT/object_graph_tokenizer_summary.json"
V1_BACKEND_MANIFEST="$V1_PREFLIGHT_ROOT/backends/object_graph_cutie/backend_predictions.json"
TOPK_BACKEND_ROOT="$STAGE/backends/object_graph_topk_candidates"
TOPK_BACKEND_MANIFEST="$TOPK_BACKEND_ROOT/backend_predictions.json"
TOPK_BACKEND_RELATIVE="backends/object_graph_topk_candidates/backend_predictions.json"
SUMMARY="$STAGE/object_graph_topk_candidate_coverage_summary.json"
INPUTS="$STAGE/provenance/immutable_inputs.json"
ISOLATION_GATE="$STAGE/provenance/scoring_isolation.json"

for path in "$SCORING_ROOT"; do
	[[ -d "$path" ]] || { echo "Missing directory: $path" >&2; exit 2; }
done
for path in "$V1_GRAPH" "$BACKEND_INPUTS" "$DATASET_MANIFEST" "$V1_SUMMARY" "$V1_BACKEND_MANIFEST"; do
	[[ -f "$path" ]] || { echo "Missing immutable input: $path" >&2; exit 2; }
done

"$PY_REPLAY" -I -c 'import sys; from pathlib import Path
source,v1,output=(Path(value).resolve() for value in sys.argv[1:4])
for left,right,label in ((source,v1,"source/v1"),(source,output,"source/output"),(v1,output,"v1/output")):
 if left == right or left in right.parents or right in left.parents:
  raise ValueError(f"Top-K preflight roots must be strictly disjoint ({label}).")' \
	"$SOURCE_BENCHMARK_ROOT" "$V1_PREFLIGHT_ROOT" "$BASE"

SOURCE_LOCK_KEY="$("$PY_REPLAY" -I -c 'import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest())' "$SOURCE_BENCHMARK_ROOT")"
# Use the exact lock namespace shared by the v1 and temporal object-graph
# workflows.  No workflow may restore scoring permissions while another owns
# this canonical frozen source.
LOCK_PARENT="$(dirname -- "$SOURCE_BENCHMARK_ROOT")/.object_graph_tokenizer_locks"
SOURCE_LOCK="$LOCK_PARENT/${SOURCE_LOCK_KEY}.lock"
mkdir -p -- "$(dirname -- "$BASE")" "$LOCK_PARENT"
[[ -d "$LOCK_PARENT" && ! -L "$LOCK_PARENT" ]] || {
	echo "Source-lock parent must be a regular directory: $LOCK_PARENT" >&2
	exit 2
}
[[ ! -e "$BASE" && ! -L "$BASE" ]] || {
	echo "Refusing to overwrite output: $BASE" >&2
	exit 3
}
[[ ! -e "$STAGE" && ! -L "$STAGE" ]] || {
	echo "Refusing to share or overwrite stage: $STAGE" >&2
	exit 3
}

STAGE_OWNED=0
PROMOTED=0
SCORING_LOCKED=0
SCORING_ROOT_MODE_BEFORE=""
SOURCE_LOCK_OWNED=0
ACTIVE_PIDS=()

# Deliberately replace, rather than extend, any caller-provided module path.
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/tdmpc2"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=""

archive_on_exit() {
	local rc=$? pid attempt alive failed
	trap - EXIT INT TERM
	for pid in "${ACTIVE_PIDS[@]:-}"; do
		[[ -n "$pid" ]] && kill -TERM -- "-$pid" 2>/dev/null || true
	done
	for attempt in {1..20}; do
		alive=0
		for pid in "${ACTIVE_PIDS[@]:-}"; do
			if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then alive=1; fi
		done
		(( alive == 0 )) && break
		sleep 0.5
	done
	for pid in "${ACTIVE_PIDS[@]:-}"; do
		[[ -n "$pid" ]] && kill -KILL -- "-$pid" 2>/dev/null || true
	done
	for pid in "${ACTIVE_PIDS[@]:-}"; do
		[[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
	done
	if (( SCORING_LOCKED == 1 )) && [[ -d "$SCORING_ROOT" ]]; then
		if chmod "$SCORING_ROOT_MODE_BEFORE" -- "$SCORING_ROOT"; then
			SCORING_LOCKED=0
		else
			echo "WARNING: failed to restore source scoring permissions." >&2
		fi
	fi
	if (( SOURCE_LOCK_OWNED == 1 )) && [[ -d "$SOURCE_LOCK" ]]; then
		if rmdir -- "$SOURCE_LOCK"; then
			SOURCE_LOCK_OWNED=0
		else
			echo "WARNING: failed to release source benchmark lock: $SOURCE_LOCK" >&2
		fi
	fi
	if (( STAGE_OWNED == 1 && PROMOTED == 0 )) && [[ -d "$STAGE" ]]; then
		"$PY_REPLAY" -c 'import json,sys; from pathlib import Path
p=Path(sys.argv[1]); rc=int(sys.argv[2]); payload={}
try:
 payload=json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
except Exception:
 payload={}
payload.update({"format":"object_graph_topk_candidate_coverage_summary_v1","status":"runner_engineering_fail","engineering_pass":False,"development_candidate":False,"selected_k":None,"controller_training_authorized":False,"scientific_go":False,"runner_exit_code":rc,"recommendation":"fix_engineering_failure_do_not_train_controller","scope":{"controller_training_steps":0}})
p.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")' \
			"$SUMMARY" "$rc" 2>/dev/null || true
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
		[[ ! -e "$failed" && ! -L "$failed" ]] || failed="${failed}.${RANDOM}"
		mv -T -- "$STAGE" "$failed"
		echo "OBJECT_GRAPH_TOPK_PREFLIGHT_FAILED_ARCHIVE=$failed" >&2
	fi
	exit "$rc"
}
# Install cleanup before taking ownership of either the stage or source lock.
trap archive_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! mkdir -- "$STAGE"; then
	echo "Refusing to share or overwrite stage: $STAGE" >&2
	exit 3
fi
STAGE_OWNED=1
mkdir -- "$STAGE/backends" "$STAGE/contracts" "$STAGE/logs" "$STAGE/provenance"
if ! mkdir -- "$SOURCE_LOCK"; then
	echo "Another object-graph workflow owns this source benchmark: $SOURCE_LOCK" >&2
	exit 4
fi
SOURCE_LOCK_OWNED=1
echo "SOURCE_BENCHMARK_LOCK_ACQUIRED=$SOURCE_LOCK"

run_tracked() {
	local pid rc
	setsid timeout --foreground --signal=TERM --kill-after=30s \
		"$TRACKED_TIMEOUT_SECONDS" "$@" & pid=$!
	ACTIVE_PIDS=("$pid")
	set +e
	wait "$pid"; rc=$?
	set -e
	ACTIVE_PIDS=()
	return "$rc"
}

file_mode() {
	local raw
	raw="$(stat -c '%a' -- "$1")"
	printf '%03o' "$((8#$raw))"
}

file_sha256() {
	"$PY_REPLAY" -I -c 'import hashlib,sys; from pathlib import Path; print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())' "$1"
}

echo "[1/5] Static CPU contracts and complete immutable source binding"
bash -n "$0"
"$PY_REPLAY" -B -m py_compile \
	tdmpc2/perception/support_conditioned_object_graph.py \
	tdmpc2/perception/ordered_chain_topk.py \
	tdmpc2/tools/replay_object_graph_topk_candidates.py \
	tdmpc2/tools/aggregate_object_graph_topk_candidates.py \
	tdmpc2/common/object_graph_temporal_replay_snapshot.py \
	tdmpc2/common/object_graph_topk_snapshot.py \
	tdmpc2/check_ordered_chain_topk_contract.py \
	tdmpc2/check_object_graph_topk_preflight_contract.py \
	tdmpc2/check_support_conditioned_object_graph_contract.py \
	>"$STAGE/contracts/python_compile.log" 2>&1
"$PY_REPLAY" -B -m tdmpc2.check_ordered_chain_topk_contract \
	>"$STAGE/contracts/ordered_chain_topk.log" 2>&1
"$PY_REPLAY" -B -m tdmpc2.check_object_graph_topk_preflight_contract \
	>"$STAGE/contracts/topk_preflight.log" 2>&1
"$PY_REPLAY" -B -m tdmpc2.check_support_conditioned_object_graph_contract \
	>"$STAGE/contracts/object_graph.log" 2>&1
"$PY_REPLAY" -B -m tdmpc2.tools.replay_object_graph_topk_candidates --help \
	>"$STAGE/contracts/backend_cli.log" 2>&1
"$PY_REPLAY" -B -m tdmpc2.tools.aggregate_object_graph_topk_candidates --help \
	>"$STAGE/contracts/aggregate_cli.log" 2>&1
"$PY_REPLAY" -B -m tdmpc2.common.object_graph_topk_snapshot \
	--source-benchmark-root "$SOURCE_BENCHMARK_ROOT" \
	--v1-preflight-root "$V1_PREFLIGHT_ROOT" \
	--output "$INPUTS" >"$STAGE/contracts/inputs_bind.log" 2>&1

echo "[2/5] Removing scoring-only GT access and replaying stateless top-K candidates on CPU"
scoring_probe="$(find "$SCORING_ROOT" -type f -print -quit)"
[[ -n "$scoring_probe" && -f "$scoring_probe" && -r "$scoring_probe" ]] || {
	echo "Could not bind a readable scoring probe before locking." >&2
	exit 4
}
SCORING_ROOT_MODE_BEFORE="$(file_mode "$SCORING_ROOT")"
SCORING_PROBE_SHA_BEFORE="$(file_sha256 "$scoring_probe")"
SCORING_PROBE_RELATIVE="$("$PY_REPLAY" -I -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).resolve(strict=True).relative_to(Path(sys.argv[2]).resolve(strict=True)).as_posix())' "$scoring_probe" "$SOURCE_BENCHMARK_ROOT")"
SCORING_LOCK_STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SCORING_LOCKED=1
chmod 000 -- "$SCORING_ROOT"
[[ "$(file_mode "$SCORING_ROOT")" == 000 && ! -r "$scoring_probe" ]] || {
	echo "Source scoring tree remained readable after lock." >&2
	exit 4
}
set +e
"$PY_REPLAY" -I -c 'import sys; from pathlib import Path; Path(sys.argv[1]).read_bytes()' \
	"$scoring_probe" >"$STAGE/contracts/scoring_read_probe.log" 2>&1
probe_rc=$?
set -e
(( probe_rc == 1 )) && grep -Fq PermissionError "$STAGE/contracts/scoring_read_probe.log" || {
	echo "Same-user scoring probe did not fail with PermissionError." >&2
	exit 4
}

echo "REPLAY_START backend=object_graph_topk_candidates device=cpu"
run_tracked env CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 \
	"$PY_REPLAY" -B -m tdmpc2.tools.replay_object_graph_topk_candidates \
	--inputs "$BACKEND_INPUTS" \
	--v1-backend-manifest "$V1_BACKEND_MANIFEST" \
	--v1-graph "$V1_GRAPH" \
	--output-root "$TOPK_BACKEND_ROOT" \
	--strict-counts >"$STAGE/logs/topk_replay.log" 2>&1
echo "REPLAY_END backend=object_graph_topk_candidates device=cpu rc=0"
BACKEND_COMPLETED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "[3/5] Stopping replay, restoring GT, and sealing the isolation record"
(( ${#ACTIVE_PIDS[@]} == 0 )) || { echo "A replay worker is still active." >&2; exit 4; }
chmod "$SCORING_ROOT_MODE_BEFORE" -- "$SCORING_ROOT"
SCORING_LOCKED=0
[[ -r "$scoring_probe" ]] || { echo "Scoring permissions were not restored." >&2; exit 4; }
SCORING_RESTORED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SCORING_ROOT_MODE_RESTORED="$(file_mode "$SCORING_ROOT")"
SCORING_PROBE_SHA_RESTORED="$(file_sha256 "$scoring_probe")"
[[ "$SCORING_ROOT_MODE_RESTORED" == "$SCORING_ROOT_MODE_BEFORE" \
	&& "$SCORING_PROBE_SHA_RESTORED" == "$SCORING_PROBE_SHA_BEFORE" ]] || {
	echo "Scoring root mode or probe bytes changed across isolation." >&2
	exit 4
}
[[ -f "$TOPK_BACKEND_MANIFEST" ]] || { echo "Top-K backend manifest is missing." >&2; exit 4; }

"$PY_REPLAY" -c 'import hashlib,json,sys; from pathlib import Path
output,source,v1,worker,scoring,probe,log,backend=map(Path,sys.argv[1:9])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "format":"object_graph_topk_candidate_scoring_isolation_v1","status":"complete",
 "source_benchmark_root":str(source.resolve()),"v1_preflight_root":str(v1.resolve()),
 "worker_input_root":str(worker.resolve()),"scoring_root":str(scoring.resolve()),
 "scoring_probe_relative_to_source":sys.argv[9],
 "root_mode_before":sys.argv[10],"root_mode_locked":"000","root_mode_restored":sys.argv[11],
 "probe_sha256_before":sys.argv[12],"probe_sha256_restored":sys.argv[13],
 "locked_utc":sys.argv[14],"backend_completed_utc":sys.argv[15],"restored_utc":sys.argv[16],
 "backend_completed_before_restore":True,"cpu_only":True,"cuda_visible_devices":"",
 "topk_backend_manifest_relative_to_summary_root":sys.argv[17],
 "topk_backend_manifest_sha256":sha(backend),
 "same_uid_read_probe":{"exit_code":1,"error_type":"PermissionError","log_relative_to_summary_root":"contracts/scoring_read_probe.log","log_sha256":sha(log)}
}
output.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")' \
	"$ISOLATION_GATE" "$SOURCE_BENCHMARK_ROOT" "$V1_PREFLIGHT_ROOT" \
	"$SOURCE_BENCHMARK_ROOT/worker_inputs" "$SCORING_ROOT" "$scoring_probe" \
	"$STAGE/contracts/scoring_read_probe.log" "$TOPK_BACKEND_MANIFEST" \
	"$SCORING_PROBE_RELATIVE" "$SCORING_ROOT_MODE_BEFORE" "$SCORING_ROOT_MODE_RESTORED" \
	"$SCORING_PROBE_SHA_BEFORE" "$SCORING_PROBE_SHA_RESTORED" \
	"$SCORING_LOCK_STARTED_UTC" "$BACKEND_COMPLETED_UTC" "$SCORING_RESTORED_UTC" \
	"$TOPK_BACKEND_RELATIVE"

echo "[4/5] Privileged K=1/2/4 scoring and full immutable revalidation"
"$PY_REPLAY" -B -m tdmpc2.common.object_graph_topk_snapshot \
	--source-benchmark-root "$SOURCE_BENCHMARK_ROOT" \
	--v1-preflight-root "$V1_PREFLIGHT_ROOT" \
	--verify "$INPUTS" >"$STAGE/contracts/inputs_verify_before_aggregate.log" 2>&1

run_tracked env CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 \
	"$PY_REPLAY" -B -m tdmpc2.tools.aggregate_object_graph_topk_candidates \
	--source-benchmark-root "$SOURCE_BENCHMARK_ROOT" \
	--v1-preflight-root "$V1_PREFLIGHT_ROOT" \
	--v1-backend-manifest "$V1_BACKEND_MANIFEST" \
	--topk-backend "$TOPK_BACKEND_MANIFEST" \
	--v1-graph "$V1_GRAPH" \
	--isolation-gate "$ISOLATION_GATE" \
	--immutable-inputs "$INPUTS" \
	--output "$SUMMARY" >"$STAGE/logs/aggregate.log" 2>&1

"$PY_REPLAY" -B -m tdmpc2.common.object_graph_topk_snapshot \
	--source-benchmark-root "$SOURCE_BENCHMARK_ROOT" \
	--v1-preflight-root "$V1_PREFLIGHT_ROOT" \
	--verify "$INPUTS" >"$STAGE/contracts/inputs_verify_after_aggregate.log" 2>&1
"$PY_REPLAY" -I -c 'import json,sys; from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if p.get("controller_training_authorized") is not False or p.get("scientific_go") is not False or p.get("scope",{}).get("controller_training_steps") != 0:
 raise RuntimeError("Top-K diagnostic attempted to authorize controller training or scientific claims.")' "$SUMMARY"

echo "[5/5] Atomic publication (never launches controller training)"
[[ ! -e "$BASE" && ! -L "$BASE" ]] || { echo "Output appeared before promotion: $BASE" >&2; exit 3; }
mv -T -- "$STAGE" "$BASE"
PROMOTED=1
if rmdir -- "$SOURCE_LOCK"; then
	SOURCE_LOCK_OWNED=0
	echo "SOURCE_BENCHMARK_LOCK_RELEASED=$SOURCE_LOCK"
else
	echo "WARNING: published successfully but could not release source lock: $SOURCE_LOCK" >&2
fi
echo "OBJECT_GRAPH_TOPK_PREFLIGHT_COMPLETE"
echo "OUTPUT_ROOT=$BASE"
echo "SUMMARY=$BASE/object_graph_topk_candidate_coverage_summary.json"
