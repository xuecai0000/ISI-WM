#!/usr/bin/env bash
# Controller-free Cutie entity->semantic-role graph development preflight.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${SOURCE_BENCHMARK_ROOT:?Set SOURCE_BENCHMARK_ROOT to the completed unified-VOS fix4 root}"
: "${OC_REPO:?Set OC_REPO to the OC-STORM checkout}"
: "${CUTIE_CKPT:?Set CUTIE_CKPT to cutie-small-mega.pth}"

PY_CORE="${PY_CORE:-python}"
PY_CUTIE="${PY_CUTIE:-$PY_CORE}"
GPU="${GPU:-1}"
RUN_TAG="${RUN_TAG:-object_graph_tokenizer_preflight_v1}"
BASE="${OUTPUT_ROOT:-$REPO_ROOT/logs/_diagnostic/$RUN_TAG}"
STAGE="${BASE}.incomplete"
GRAPH_DIR="$REPO_ROOT/tdmpc2/object_graphs"
BACKEND_INPUTS="$SOURCE_BENCHMARK_ROOT/worker_inputs/backend_inputs.json"
DATASET_MANIFEST="$SOURCE_BENCHMARK_ROOT/dataset/dataset_manifest.json"
SCORING_ROOT="$SOURCE_BENCHMARK_ROOT/dataset/scoring"
BASELINE_MANIFEST="$SOURCE_BENCHMARK_ROOT/backends/cutie/backend_predictions.json"
GRAPH_BACKEND_ROOT="$STAGE/backends/object_graph_cutie"
GRAPH_BACKEND_MANIFEST="$GRAPH_BACKEND_ROOT/backend_predictions.json"
SUMMARY="$STAGE/object_graph_tokenizer_summary.json"
INPUTS="$STAGE/provenance/immutable_inputs.json"
ISOLATION_GATE="$STAGE/provenance/scoring_isolation.json"
TRACKED_TIMEOUT_SECONDS="${TRACKED_TIMEOUT_SECONDS:-43200}"
BACKEND_SEED=2718281

[[ "$GPU" =~ ^(0|[1-9][0-9]*)$ ]] || {
	echo "GPU must be a non-negative physical index." >&2
	exit 2
}
[[ "$TRACKED_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
	echo "TRACKED_TIMEOUT_SECONDS must be a positive integer." >&2
	exit 2
}

check_python() {
	local executable=$1
	if [[ "$executable" == */* ]]; then
		[[ -x "$executable" ]] || { echo "Python is not executable: $executable" >&2; exit 2; }
	else
		command -v "$executable" >/dev/null || { echo "Python is not on PATH: $executable" >&2; exit 2; }
	fi
}
check_python "$PY_CORE"
check_python "$PY_CUTIE"

resolve_python() {
	local executable=$1
	if [[ "$executable" != */* ]]; then
		executable="$(command -v -- "$executable")"
	fi
	printf '%s\n' "$executable"
}
PY_CORE="$(resolve_python "$PY_CORE")"
PY_CUTIE="$(resolve_python "$PY_CUTIE")"
PY_CORE_REAL="$("$PY_CORE" -I -c 'import sys; from pathlib import Path; print(Path(sys.executable).resolve(strict=True))')"
PY_CUTIE_REAL="$("$PY_CUTIE" -I -c 'import sys; from pathlib import Path; print(Path(sys.executable).resolve(strict=True))')"
[[ "$PY_CORE_REAL" == "$PY_CUTIE_REAL" ]] || {
	echo "PY_CORE and PY_CUTIE must resolve to the same frozen environment." >&2
	exit 2
}
PY_CORE="$PY_CORE_REAL"
PY_CUTIE="$PY_CUTIE_REAL"

for path in "$SOURCE_BENCHMARK_ROOT" "$OC_REPO" "$GRAPH_DIR" "$SCORING_ROOT"; do
	[[ -d "$path" ]] || { echo "Missing directory: $path" >&2; exit 2; }
done
for path in \
	"$CUTIE_CKPT" "$BACKEND_INPUTS" "$DATASET_MANIFEST" "$BASELINE_MANIFEST" \
	"$SOURCE_BENCHMARK_ROOT/unified_vos_summary.json"; do
	[[ -f "$path" ]] || { echo "Missing immutable file: $path" >&2; exit 2; }
done

SOURCE_ROOT_CANONICAL="$("$PY_CORE" -I -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).resolve(strict=True))' "$SOURCE_BENCHMARK_ROOT")"
SOURCE_LOCK_KEY="$("$PY_CORE" -I -c 'import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest())' "$SOURCE_ROOT_CANONICAL")"
# Keep the lock beside the canonical frozen source rather than inside this
# checkout, so every checkout/worktree that uses the same source competes for
# the same fail-closed lock.
LOCK_PARENT="$(dirname -- "$SOURCE_ROOT_CANONICAL")/.object_graph_tokenizer_locks"
SOURCE_LOCK="$LOCK_PARENT/${SOURCE_LOCK_KEY}.lock"
mkdir -p -- "$(dirname -- "$BASE")" "$LOCK_PARENT"
[[ -d "$LOCK_PARENT" && ! -L "$LOCK_PARENT" ]] || {
	echo "Source-lock parent must be a regular directory: $LOCK_PARENT" >&2
	exit 2
}
[[ ! -e "$BASE" ]] || { echo "Refusing to overwrite output: $BASE" >&2; exit 3; }
if ! mkdir -- "$STAGE"; then
	echo "Refusing to share or overwrite stage: $STAGE" >&2
	exit 3
fi
STAGE_OWNED=1
PROMOTED=0
SCORING_LOCKED=0
SCORING_ROOT_MODE_BEFORE=""
SOURCE_LOCK_OWNED=0
ACTIVE_PIDS=()

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/tdmpc2"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

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
	# Restore source GT only after every worker process group is gone.
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
		"$PY_CORE" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); rc=int(sys.argv[2]); payload={}
try:
 payload=json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
except Exception:
 payload={}
payload.update({"format":"object_graph_tokenizer_preflight_summary_v1","status":"runner_engineering_fail","engineering_pass":False,"development_candidate":False,"controller_training_authorized":False,"scientific_go":False,"runner_exit_code":rc,"recommendation":"fix_engineering_failure_do_not_train_controller"})
p.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")' \
			"$SUMMARY" "$rc" 2>/dev/null || true
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
		[[ ! -e "$failed" ]] || failed="${failed}.${RANDOM}"
		mv -T -- "$STAGE" "$failed"
		echo "OBJECT_GRAPH_TOKENIZER_FAILED_ARCHIVE=$failed" >&2
	fi
	exit "$rc"
}
trap archive_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -- "$STAGE/backends" "$STAGE/contracts" "$STAGE/logs" "$STAGE/provenance"
if ! mkdir -- "$SOURCE_LOCK"; then
	echo "Another object-graph preflight owns this source benchmark: $SOURCE_LOCK" >&2
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

wait_gpu_idle() {
	local elapsed=0 active
	while (( elapsed <= 120 )); do
		if ! active="$(nvidia-smi --id="$BENCHMARK_GPU_UUID" --query-compute-apps=pid \
			--format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"; then
			echo "Failed to query compute processes for GPU=$GPU." >&2
			return 4
		fi
		[[ -z "$active" ]] && return 0
		(( elapsed == 0 )) && echo "Waiting at most 120s for GPU=$GPU to become idle"
		sleep 5
		elapsed=$((elapsed + 5))
	done
	echo "GPU=$GPU remained busy; no process was signaled: $active" >&2
	return 5
}

file_mode() {
	local raw
	raw="$(stat -c '%a' -- "$1")"
	printf '%03o' "$((8#$raw))"
}

file_sha256() {
	"$PY_CORE" -I -c 'import hashlib,sys; from pathlib import Path; print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())' "$1"
}

echo "[1/5] Static contracts, source identity, GPU pairing, and immutable inputs"
bash -n "$0"
"$PY_CUTIE" -B -m py_compile \
	tdmpc2/perception/support_conditioned_object_graph.py \
	tdmpc2/perception/cutie_oc_adapter.py \
	tdmpc2/tools/run_unified_vos_object_graph_cutie_backend.py \
	tdmpc2/tools/aggregate_object_graph_tokenizer_preflight.py \
	tdmpc2/common/object_graph_preflight_snapshot.py \
	tdmpc2/check_support_conditioned_object_graph_contract.py \
	>"$STAGE/contracts/python_compile.log" 2>&1
"$PY_CUTIE" -B -m tdmpc2.check_support_conditioned_object_graph_contract \
	>"$STAGE/contracts/object_graph.log" 2>&1

command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required." >&2; exit 2; }
BENCHMARK_GPU_UUID="$(nvidia-smi --id="$GPU" --query-gpu=uuid --format=csv,noheader,nounits)"
[[ "$BENCHMARK_GPU_UUID" == GPU-* && "$BENCHMARK_GPU_UUID" != *$'\n'* ]] || {
	echo "Could not bind one physical GPU UUID for index $GPU." >&2
	exit 4
}
export BENCHMARK_GPU_UUID
SOURCE_GPU_UUID="$("$PY_CORE" -c 'import json,sys; from pathlib import Path; p=json.loads((Path(sys.argv[1])/"backends/cutie/backend_predictions.json").read_text()); print(p["backend_provenance"]["gpu_uuid"])' "$SOURCE_BENCHMARK_ROOT")"
[[ "$SOURCE_GPU_UUID" == "$BENCHMARK_GPU_UUID" ]] || {
	echo "GPU UUID differs from the frozen Cutie baseline: source=$SOURCE_GPU_UUID selected=$BENCHMARK_GPU_UUID" >&2
	exit 4
}

"$PY_CORE" -B -m tdmpc2.common.object_graph_preflight_snapshot \
	--source-root "$SOURCE_BENCHMARK_ROOT" \
	--python "$PY_CUTIE" \
	--oc-repo "$OC_REPO" \
	--checkpoint "$CUTIE_CKPT" \
	--output "$INPUTS" >"$STAGE/contracts/inputs_bind.log" 2>&1

echo "[2/5] Locking scoring-only GT and running a real single-entity Cutie smoke"
scoring_probe="$(find "$SCORING_ROOT" -type f -print -quit)"
[[ -n "$scoring_probe" && -f "$scoring_probe" && -r "$scoring_probe" ]] || {
	echo "Could not bind a readable scoring probe before locking." >&2
	exit 4
}
SCORING_ROOT_MODE_BEFORE="$(file_mode "$SCORING_ROOT")"
SCORING_PROBE_SHA_BEFORE="$(file_sha256 "$scoring_probe")"
SCORING_PROBE_RELATIVE="$("$PY_CORE" -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).resolve(strict=True).relative_to(Path(sys.argv[2]).resolve(strict=True)).as_posix())' "$scoring_probe" "$SOURCE_BENCHMARK_ROOT")"
SCORING_LOCK_STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SCORING_LOCKED=1
chmod 000 -- "$SCORING_ROOT"
[[ "$(file_mode "$SCORING_ROOT")" == 000 && ! -r "$scoring_probe" ]] || {
	echo "Source scoring tree remained readable after lock." >&2
	exit 4
}
set +e
"$PY_CUTIE" -I -c 'import sys; from pathlib import Path; Path(sys.argv[1]).read_bytes()' \
	"$scoring_probe" >"$STAGE/contracts/scoring_read_probe.log" 2>&1
probe_rc=$?
set -e
(( probe_rc == 1 )) && grep -Fq PermissionError "$STAGE/contracts/scoring_read_probe.log" || {
	echo "Same-user scoring probe did not fail with PermissionError." >&2
	exit 4
}

wait_gpu_idle
echo "MODEL_PREFLIGHT_START backend=object_graph_cutie gpu=$GPU entities=1"
run_tracked env CUDA_VISIBLE_DEVICES="$BENCHMARK_GPU_UUID" \
	"$PY_CUTIE" -B -m tdmpc2.tools.run_unified_vos_object_graph_cutie_backend \
	--inputs "$BACKEND_INPUTS" \
	--graph-dir "$GRAPH_DIR" \
	--oc-storm-repo "$OC_REPO" \
	--checkpoint "$CUTIE_CKPT" \
	--model-size small --tracker-size 448 --seed "$BACKEND_SEED" \
	--strict-counts --preflight-only \
	>"$STAGE/logs/object_graph_preflight.log" 2>&1
echo "MODEL_PREFLIGHT_END backend=object_graph_cutie gpu=$GPU rc=0"

echo "[3/5] Grouped-entity Cutie and semantic role tokenizer on frozen GT-free RGB"
wait_gpu_idle
echo "BACKEND_START backend=object_graph_cutie gpu=$GPU"
run_tracked env CUDA_VISIBLE_DEVICES="$BENCHMARK_GPU_UUID" \
	"$PY_CUTIE" -B -m tdmpc2.tools.run_unified_vos_object_graph_cutie_backend \
	--inputs "$BACKEND_INPUTS" \
	--output-root "$GRAPH_BACKEND_ROOT" \
	--graph-dir "$GRAPH_DIR" \
	--oc-storm-repo "$OC_REPO" \
	--checkpoint "$CUTIE_CKPT" \
	--model-size small --tracker-size 448 --seed "$BACKEND_SEED" \
	--strict-counts >"$STAGE/logs/object_graph_cutie.log" 2>&1
echo "BACKEND_END backend=object_graph_cutie gpu=$GPU rc=0"
BACKEND_COMPLETED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "[4/5] Restoring GT access, immutable recheck, and strict three-level scoring"
(( ${#ACTIVE_PIDS[@]} == 0 )) || { echo "A backend is still active." >&2; exit 4; }
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
"$PY_CORE" -c 'import hashlib,json,sys; from pathlib import Path
output,source,worker,scoring,probe,log,backend=map(Path,sys.argv[1:8])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "format":"object_graph_scoring_isolation_gate_v1","status":"complete",
 "source_benchmark_root":str(source.resolve()),"worker_input_root":str(worker.resolve()),
 "scoring_root":str(scoring.resolve()),"scoring_probe_relative_to_source":sys.argv[8],
 "root_mode_before":sys.argv[9],"root_mode_locked":"000","root_mode_restored":sys.argv[10],
 "probe_sha256_before":sys.argv[11],"probe_sha256_restored":sys.argv[12],
 "locked_utc":sys.argv[13],"backend_completed_utc":sys.argv[14],"restored_utc":sys.argv[15],
 "backend_completed_before_restore":True,"backend_manifest_sha256":sha(backend),
 "same_uid_read_probe":{"exit_code":1,"error_type":"PermissionError","log_relative_to_summary_root":"contracts/scoring_read_probe.log","log_sha256":sha(log)}
}
output.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")' \
	"$ISOLATION_GATE" "$SOURCE_BENCHMARK_ROOT" "$SOURCE_BENCHMARK_ROOT/worker_inputs" \
	"$SCORING_ROOT" "$scoring_probe" "$STAGE/contracts/scoring_read_probe.log" \
	"$GRAPH_BACKEND_MANIFEST" "$SCORING_PROBE_RELATIVE" "$SCORING_ROOT_MODE_BEFORE" \
	"$SCORING_ROOT_MODE_RESTORED" "$SCORING_PROBE_SHA_BEFORE" \
	"$SCORING_PROBE_SHA_RESTORED" "$SCORING_LOCK_STARTED_UTC" \
	"$BACKEND_COMPLETED_UTC" "$SCORING_RESTORED_UTC"

"$PY_CORE" -B -m tdmpc2.common.object_graph_preflight_snapshot \
	--source-root "$SOURCE_BENCHMARK_ROOT" \
	--python "$PY_CUTIE" \
	--oc-repo "$OC_REPO" \
	--checkpoint "$CUTIE_CKPT" \
	--verify "$INPUTS" >"$STAGE/contracts/inputs_verify_before_aggregate.log" 2>&1

"$PY_CORE" -B -m tdmpc2.tools.aggregate_object_graph_tokenizer_preflight \
	--source-benchmark-root "$SOURCE_BENCHMARK_ROOT" \
	--object-graph-backend "$GRAPH_BACKEND_MANIFEST" \
	--graph-dir "$GRAPH_DIR" \
	--isolation-gate "$ISOLATION_GATE" \
	--output "$SUMMARY" >"$STAGE/logs/aggregate.log" 2>&1

"$PY_CORE" -B -m tdmpc2.common.object_graph_preflight_snapshot \
	--source-root "$SOURCE_BENCHMARK_ROOT" \
	--python "$PY_CUTIE" \
	--oc-repo "$OC_REPO" \
	--checkpoint "$CUTIE_CKPT" \
	--verify "$INPUTS" >"$STAGE/contracts/inputs_verify_after_aggregate.log" 2>&1

echo "[5/5] Atomic publication (never launches controller training)"
[[ ! -e "$BASE" ]] || { echo "Output appeared before promotion: $BASE" >&2; exit 3; }
mv -T -- "$STAGE" "$BASE"
PROMOTED=1
if rmdir -- "$SOURCE_LOCK"; then
	SOURCE_LOCK_OWNED=0
	echo "SOURCE_BENCHMARK_LOCK_RELEASED=$SOURCE_LOCK"
else
	# A stale owned lock is fail-closed for future runs and does not invalidate
	# the already atomically published, fully verified result.
	echo "WARNING: published successfully but could not release source lock: $SOURCE_LOCK" >&2
fi
echo "OBJECT_GRAPH_TOKENIZER_PREFLIGHT_COMPLETE"
echo "OUTPUT_ROOT=$BASE"
echo "SUMMARY=$BASE/object_graph_tokenizer_summary.json"
