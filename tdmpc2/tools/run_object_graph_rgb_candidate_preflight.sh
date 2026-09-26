#!/usr/bin/env bash
# Current-RGB, controller-free ordered-chain candidate-coverage preflight.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${SOURCE_BENCHMARK_ROOT:?Set SOURCE_BENCHMARK_ROOT to the published unified-VOS root}"
: "${V1_PREFLIGHT_ROOT:?Set V1_PREFLIGHT_ROOT to the published v1 object-graph root}"
: "${MASK_TOPK_PREFLIGHT_ROOT:?Set MASK_TOPK_PREFLIGHT_ROOT to the published mask-only Top-K root}"
: "${DINO_REPO:?Set DINO_REPO to the local DINOv2 checkout}"
: "${DINO_CHECKPOINT:?Set DINO_CHECKPOINT to the frozen DINOv2 checkpoint}"

PY_RGB="${PY_RGB:-${PY_CORE:-python}}"
GPU="${GPU:-1}"
DINO_MODEL="${DINO_MODEL:-dinov2_vits14_reg}"
DINO_INPUT_SIZE="${DINO_INPUT_SIZE:-224}"
TRACKED_TIMEOUT_SECONDS="${TRACKED_TIMEOUT_SECONDS:-14400}"
RUN_TAG="${RUN_TAG:-object_graph_rgb_candidate_preflight_v1}"
BASE="${OUTPUT_ROOT:-$REPO_ROOT/logs/_diagnostic/$RUN_TAG}"

[[ "$GPU" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "GPU must be a non-negative physical index." >&2; exit 2; }
[[ "$DINO_INPUT_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "DINO_INPUT_SIZE must be a positive integer." >&2; exit 2; }
[[ "$DINO_MODEL" == "dinov2_vits14_reg" && "$DINO_INPUT_SIZE" == "224" ]] || {
	echo "This frozen preflight requires DINO_MODEL=dinov2_vits14_reg and DINO_INPUT_SIZE=224." >&2
	exit 2
}
[[ "$TRACKED_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "TRACKED_TIMEOUT_SECONDS must be a positive integer." >&2; exit 2; }

if [[ "$PY_RGB" == */* ]]; then
	[[ -x "$PY_RGB" ]] || { echo "Python is not executable: $PY_RGB" >&2; exit 2; }
else
	PY_RGB="$(command -v -- "$PY_RGB")" || { echo "Python is not on PATH." >&2; exit 2; }
fi
PY_RGB="$("$PY_RGB" -I -c 'import sys; from pathlib import Path; print(Path(sys.executable).resolve(strict=True))')"
[[ -x "$PY_RGB" ]] || { echo "Resolved Python is not executable: $PY_RGB" >&2; exit 2; }

for path in "$SOURCE_BENCHMARK_ROOT" "$V1_PREFLIGHT_ROOT" "$MASK_TOPK_PREFLIGHT_ROOT" "$DINO_REPO"; do
	[[ -d "$path" ]] || { echo "Missing directory: $path" >&2; exit 2; }
done
[[ -f "$DINO_CHECKPOINT" ]] || { echo "Missing immutable file: $DINO_CHECKPOINT" >&2; exit 2; }

canonical_root() {
	"$PY_RGB" -I -c 'import sys
from pathlib import Path
p=Path(sys.argv[1]).expanduser().absolute()
for component in (*reversed(p.parents),p):
    if component.is_symlink():
        raise ValueError(f"Root contains a symlink component: {component}")
print(p.resolve(strict=True))' "$1"
}
SOURCE_BENCHMARK_ROOT="$(canonical_root "$SOURCE_BENCHMARK_ROOT")"
V1_PREFLIGHT_ROOT="$(canonical_root "$V1_PREFLIGHT_ROOT")"
MASK_TOPK_PREFLIGHT_ROOT="$(canonical_root "$MASK_TOPK_PREFLIGHT_ROOT")"
DINO_REPO="$(canonical_root "$DINO_REPO")"
DINO_CHECKPOINT="$("$PY_RGB" -I -c 'import sys
from pathlib import Path
p=Path(sys.argv[1]).expanduser().absolute()
for component in (*reversed(p.parents),p):
    if component.is_symlink():
        raise ValueError(f"Checkpoint contains a symlink component: {component}")
p=p.resolve(strict=True)
if not p.is_file():
    raise ValueError(f"Checkpoint is not a regular file: {p}")
print(p)' "$DINO_CHECKPOINT")"

STAGE="${BASE}.incomplete"
V1_GRAPH="$REPO_ROOT/tdmpc2/object_graphs/acrobot_swingup.json"
BACKEND_INPUTS="$SOURCE_BENCHMARK_ROOT/worker_inputs/backend_inputs.json"
SCORING_ROOT="$SOURCE_BENCHMARK_ROOT/dataset/scoring"
V1_BACKEND_MANIFEST="$V1_PREFLIGHT_ROOT/backends/object_graph_cutie/backend_predictions.json"
MASK_TOPK_SUMMARY="$MASK_TOPK_PREFLIGHT_ROOT/object_graph_topk_candidate_coverage_summary.json"
RGB_BACKEND_ROOT="$STAGE/backends/object_graph_rgb_candidates"
RGB_BACKEND_MANIFEST="$RGB_BACKEND_ROOT/backend_predictions.json"
RGB_BACKEND_RELATIVE="backends/object_graph_rgb_candidates/backend_predictions.json"
SUMMARY="$STAGE/object_graph_rgb_candidate_coverage_summary.json"
INPUTS="$STAGE/provenance/immutable_inputs.json"
ISOLATION_GATE="$STAGE/provenance/scoring_isolation.json"
BACKEND_TREE_SEAL="$STAGE/provenance/rgb_backend_tree.json"

[[ -d "$SCORING_ROOT" ]] || { echo "Missing directory: $SCORING_ROOT" >&2; exit 2; }
for path in "$V1_GRAPH" "$BACKEND_INPUTS" "$V1_BACKEND_MANIFEST" "$MASK_TOPK_SUMMARY" "$SOURCE_BENCHMARK_ROOT/unified_vos_summary.json"; do
	[[ -f "$path" ]] || { echo "Missing immutable input: $path" >&2; exit 2; }
done

"$PY_RGB" -I -c 'import sys
from pathlib import Path
roots=[Path(value).resolve() for value in sys.argv[1:6]]
names=("source","v1","mask_topk","dino_repo","output")
for i,left in enumerate(roots):
    for j,right in enumerate(roots[i+1:],i+1):
        if left == right or left in right.parents or right in left.parents:
            raise ValueError(f"RGB preflight roots must be strictly disjoint ({names[i]}/{names[j]}).")' \
	"$SOURCE_BENCHMARK_ROOT" "$V1_PREFLIGHT_ROOT" "$MASK_TOPK_PREFLIGHT_ROOT" \
	"$DINO_REPO" "$BASE"

SOURCE_LOCK_KEY="$("$PY_RGB" -I -c 'import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest())' "$SOURCE_BENCHMARK_ROOT")"
LOCK_PARENT="$(dirname -- "$SOURCE_BENCHMARK_ROOT")/.object_graph_tokenizer_locks"
SOURCE_LOCK="$LOCK_PARENT/${SOURCE_LOCK_KEY}.lock"
mkdir -p -- "$(dirname -- "$BASE")" "$LOCK_PARENT"
[[ -d "$LOCK_PARENT" && ! -L "$LOCK_PARENT" ]] || { echo "Source-lock parent must be a regular directory: $LOCK_PARENT" >&2; exit 2; }
[[ ! -e "$BASE" && ! -L "$BASE" ]] || { echo "Refusing to overwrite output: $BASE" >&2; exit 3; }
[[ ! -e "$STAGE" && ! -L "$STAGE" ]] || { echo "Refusing to share or overwrite stage: $STAGE" >&2; exit 3; }

STAGE_OWNED=0
PROMOTED=0
SCORING_LOCKED=0
SCORING_ROOT_MODE_BEFORE=""
SOURCE_LOCK_OWNED=0
ACTIVE_PIDS=()

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/tdmpc2"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

archive_on_exit() {
	local rc=$? pid attempt alive failed
	trap - EXIT INT TERM
	for pid in "${ACTIVE_PIDS[@]:-}"; do [[ -n "$pid" ]] && kill -TERM -- "-$pid" 2>/dev/null || true; done
	for attempt in {1..20}; do
		alive=0
		for pid in "${ACTIVE_PIDS[@]:-}"; do
			if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then alive=1; fi
		done
		(( alive == 0 )) && break
		sleep 0.5
	done
	for pid in "${ACTIVE_PIDS[@]:-}"; do [[ -n "$pid" ]] && kill -KILL -- "-$pid" 2>/dev/null || true; done
	for pid in "${ACTIVE_PIDS[@]:-}"; do [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true; done
	if (( SCORING_LOCKED == 1 )) && [[ -d "$SCORING_ROOT" ]]; then
		if chmod "$SCORING_ROOT_MODE_BEFORE" -- "$SCORING_ROOT"; then SCORING_LOCKED=0; else echo "WARNING: failed to restore source scoring permissions." >&2; fi
	fi
	if (( SOURCE_LOCK_OWNED == 1 )) && [[ -d "$SOURCE_LOCK" ]]; then
		if rmdir -- "$SOURCE_LOCK"; then SOURCE_LOCK_OWNED=0; else echo "WARNING: failed to release source benchmark lock: $SOURCE_LOCK" >&2; fi
	fi
	if (( STAGE_OWNED == 1 && PROMOTED == 0 )) && [[ -d "$STAGE" ]]; then
		"$PY_RGB" -c 'import json,sys
from pathlib import Path
p=Path(sys.argv[1]); rc=int(sys.argv[2]); payload={}
try:
    payload=json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
except Exception:
    payload={}
payload.update({"format":"object_graph_rgb_candidate_coverage_summary_v1","status":"runner_engineering_fail","engineering_pass":False,"development_candidate":False,"selected_k":None,"controller_training_authorized":False,"scientific_go":False,"runner_exit_code":rc,"recommendation":"fix_engineering_failure_do_not_train_controller","scope":{"controller_training_steps":0}})
p.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")' "$SUMMARY" "$rc" 2>/dev/null || true
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
		[[ ! -e "$failed" && ! -L "$failed" ]] || failed="${failed}.${RANDOM}"
		mv -T -- "$STAGE" "$failed"
		echo "OBJECT_GRAPH_RGB_PREFLIGHT_FAILED_ARCHIVE=$failed" >&2
	fi
	exit "$rc"
}
trap archive_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! mkdir -- "$STAGE"; then echo "Refusing to share or overwrite stage: $STAGE" >&2; exit 3; fi
STAGE_OWNED=1
mkdir -- "$STAGE/backends" "$STAGE/contracts" "$STAGE/logs" "$STAGE/provenance"
if ! mkdir -- "$SOURCE_LOCK"; then echo "Another object-graph workflow owns this source benchmark: $SOURCE_LOCK" >&2; exit 4; fi
SOURCE_LOCK_OWNED=1
echo "SOURCE_BENCHMARK_LOCK_ACQUIRED=$SOURCE_LOCK"

run_tracked() {
	local pid rc
	setsid timeout --foreground --signal=TERM --kill-after=30s "$TRACKED_TIMEOUT_SECONDS" "$@" & pid=$!
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
		if ! active="$(nvidia-smi --id="$BENCHMARK_GPU_UUID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"; then
			echo "Failed to query compute processes for GPU=$GPU." >&2; return 4
		fi
		[[ -z "$active" ]] && return 0
		(( elapsed == 0 )) && echo "Waiting at most 120s for GPU=$GPU to become idle"
		sleep 5; elapsed=$((elapsed + 5))
	done
	echo "GPU=$GPU remained busy; no process was signaled: $active" >&2; return 5
}

file_mode() {
	local raw
	raw="$(stat -c '%a' -- "$1")"
	printf '%03o' "$((8#$raw))"
}

file_sha256() {
	"$PY_RGB" -I -c 'import hashlib,sys; from pathlib import Path; print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())' "$1"
}

seal_backend_tree() {
	local mode=$1
	"$PY_RGB" -I - "$RGB_BACKEND_ROOT" "$BACKEND_TREE_SEAL" "$mode" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
seal = Path(sys.argv[2])
mode = sys.argv[3]
rows = []
for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
    if path.is_symlink():
        raise ValueError(f"RGB backend contains a symlink: {path}")
    if path.is_dir():
        continue
    if not path.is_file():
        raise ValueError(f"RGB backend contains a special member: {path}")
    data = path.read_bytes()
    rows.append(
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    )
if not rows:
    raise ValueError("RGB backend tree is empty.")
payload = {
    "format": "object_graph_rgb_backend_tree_seal_v1",
    "root_relative_to_summary": "backends/object_graph_rgb_candidates",
    "files": rows,
}
encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
if mode == "bind":
    if seal.exists():
        raise FileExistsError(seal)
    seal.write_bytes(encoded)
elif mode == "verify":
    if seal.read_bytes() != encoded:
        raise RuntimeError("RGB backend tree changed after it was sealed.")
else:
    raise ValueError(mode)
PY
}

echo "[1/5] Static contracts, source identity, GPU pairing, and immutable inputs"
bash -n "$0"
"$PY_RGB" -B -m py_compile \
	tdmpc2/perception/support_conditioned_object_graph.py \
	tdmpc2/perception/ordered_chain_topk.py \
	tdmpc2/perception/ordered_chain_rgb.py \
	tdmpc2/tools/replay_object_graph_rgb_candidates.py \
	tdmpc2/tools/aggregate_object_graph_rgb_candidates.py \
	tdmpc2/common/object_graph_temporal_replay_snapshot.py \
	tdmpc2/common/object_graph_rgb_snapshot.py \
	tdmpc2/check_support_conditioned_object_graph_contract.py \
	tdmpc2/check_ordered_chain_topk_contract.py \
	tdmpc2/check_ordered_chain_rgb_contract.py \
	tdmpc2/check_object_graph_rgb_preflight_contract.py \
	>"$STAGE/contracts/python_compile.log" 2>&1
"$PY_RGB" -B -m tdmpc2.check_support_conditioned_object_graph_contract >"$STAGE/contracts/object_graph.log" 2>&1
"$PY_RGB" -B -m tdmpc2.check_ordered_chain_topk_contract >"$STAGE/contracts/ordered_chain_topk.log" 2>&1
"$PY_RGB" -B -m tdmpc2.check_ordered_chain_rgb_contract >"$STAGE/contracts/ordered_chain_rgb.log" 2>&1
"$PY_RGB" -B -m tdmpc2.check_object_graph_rgb_preflight_contract >"$STAGE/contracts/rgb_preflight.log" 2>&1
"$PY_RGB" -B -m tdmpc2.tools.replay_object_graph_rgb_candidates --help >"$STAGE/contracts/backend_cli.log" 2>&1
"$PY_RGB" -B -m tdmpc2.tools.aggregate_object_graph_rgb_candidates --help >"$STAGE/contracts/aggregate_cli.log" 2>&1

command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required." >&2; exit 2; }
BENCHMARK_GPU_UUID="$(nvidia-smi --id="$GPU" --query-gpu=uuid --format=csv,noheader,nounits)"
[[ "$BENCHMARK_GPU_UUID" == GPU-* && "$BENCHMARK_GPU_UUID" != *$'\n'* ]] || {
	echo "Could not bind one physical GPU UUID for index $GPU." >&2; exit 4
}
export BENCHMARK_GPU_UUID
SOURCE_GPU_UUID="$("$PY_RGB" -I -c 'import json,sys; from pathlib import Path; p=json.loads((Path(sys.argv[1])/"backends/cutie/backend_predictions.json").read_text()); print(p["backend_provenance"]["gpu_uuid"])' "$SOURCE_BENCHMARK_ROOT")"
[[ "$SOURCE_GPU_UUID" == "$BENCHMARK_GPU_UUID" ]] || {
	echo "GPU UUID differs from frozen Cutie: source=$SOURCE_GPU_UUID selected=$BENCHMARK_GPU_UUID" >&2; exit 4
}

env CUDA_VISIBLE_DEVICES="" "$PY_RGB" -B -m tdmpc2.common.object_graph_rgb_snapshot \
	--source-benchmark-root "$SOURCE_BENCHMARK_ROOT" \
	--v1-preflight-root "$V1_PREFLIGHT_ROOT" \
	--mask-topk-preflight-root "$MASK_TOPK_PREFLIGHT_ROOT" \
	--dino-repo "$DINO_REPO" --dino-checkpoint "$DINO_CHECKPOINT" \
	--output "$INPUTS" >"$STAGE/contracts/inputs_bind.log" 2>&1

echo "[2/5] Removing scoring-only GT and running a real current-RGB DINO smoke"
scoring_probe="$(find "$SCORING_ROOT" -type f -print -quit)"
[[ -n "$scoring_probe" && -f "$scoring_probe" && -r "$scoring_probe" ]] || {
	echo "Could not bind a readable scoring probe before locking." >&2; exit 4
}
SCORING_ROOT_MODE_BEFORE="$(file_mode "$SCORING_ROOT")"
SCORING_PROBE_SHA_BEFORE="$(file_sha256 "$scoring_probe")"
SCORING_PROBE_RELATIVE="$("$PY_RGB" -I -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).resolve(strict=True).relative_to(Path(sys.argv[2]).resolve(strict=True)).as_posix())' "$scoring_probe" "$SOURCE_BENCHMARK_ROOT")"
SCORING_LOCK_STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SCORING_LOCKED=1
chmod 000 -- "$SCORING_ROOT"
[[ "$(file_mode "$SCORING_ROOT")" == 000 && ! -r "$scoring_probe" ]] || {
	echo "Source scoring tree remained readable after lock." >&2; exit 4
}
set +e
"$PY_RGB" -I -c 'import sys; from pathlib import Path; Path(sys.argv[1]).read_bytes()' "$scoring_probe" >"$STAGE/contracts/scoring_read_probe.log" 2>&1
probe_rc=$?
set -e
(( probe_rc == 1 )) && grep -Fq PermissionError "$STAGE/contracts/scoring_read_probe.log" || {
	echo "Same-user scoring probe did not fail with PermissionError." >&2; exit 4
}

wait_gpu_idle
echo "MODEL_PREFLIGHT_START backend=object_graph_rgb_candidates gpu=$GPU"
run_tracked env CUDA_VISIBLE_DEVICES="$BENCHMARK_GPU_UUID" PYTHONHASHSEED=0 \
	"$PY_RGB" -B -m tdmpc2.tools.replay_object_graph_rgb_candidates \
	--inputs "$BACKEND_INPUTS" --v1-backend-manifest "$V1_BACKEND_MANIFEST" \
	--v1-graph "$V1_GRAPH" --dino-repo "$DINO_REPO" --dino-checkpoint "$DINO_CHECKPOINT" \
	--dino-model "$DINO_MODEL" --dino-input-size "$DINO_INPUT_SIZE" \
	--strict-counts --preflight-only >"$STAGE/logs/rgb_model_preflight.log" 2>&1
echo "MODEL_PREFLIGHT_END backend=object_graph_rgb_candidates gpu=$GPU rc=0"

echo "[3/5] Current-RGB and spatial-shuffle candidate replay on frozen GT-free inputs"
wait_gpu_idle
echo "REPLAY_START backend=object_graph_rgb_candidates gpu=$GPU"
run_tracked env CUDA_VISIBLE_DEVICES="$BENCHMARK_GPU_UUID" PYTHONHASHSEED=0 \
	"$PY_RGB" -B -m tdmpc2.tools.replay_object_graph_rgb_candidates \
	--inputs "$BACKEND_INPUTS" --v1-backend-manifest "$V1_BACKEND_MANIFEST" \
	--v1-graph "$V1_GRAPH" --dino-repo "$DINO_REPO" --dino-checkpoint "$DINO_CHECKPOINT" \
	--dino-model "$DINO_MODEL" --dino-input-size "$DINO_INPUT_SIZE" \
	--output-root "$RGB_BACKEND_ROOT" --strict-counts >"$STAGE/logs/rgb_backend.log" 2>&1
echo "REPLAY_END backend=object_graph_rgb_candidates gpu=$GPU rc=0"
BACKEND_COMPLETED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
[[ -f "$RGB_BACKEND_MANIFEST" ]] || { echo "RGB backend manifest is missing." >&2; exit 4; }
seal_backend_tree bind

echo "[4/5] Restoring GT, privileged scoring, and full immutable revalidation"
(( ${#ACTIVE_PIDS[@]} == 0 )) || { echo "An RGB backend worker is still active." >&2; exit 4; }
chmod "$SCORING_ROOT_MODE_BEFORE" -- "$SCORING_ROOT"
SCORING_LOCKED=0
[[ -r "$scoring_probe" ]] || { echo "Scoring permissions were not restored." >&2; exit 4; }
SCORING_RESTORED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SCORING_ROOT_MODE_RESTORED="$(file_mode "$SCORING_ROOT")"
SCORING_PROBE_SHA_RESTORED="$(file_sha256 "$scoring_probe")"
[[ "$SCORING_ROOT_MODE_RESTORED" == "$SCORING_ROOT_MODE_BEFORE" \
	&& "$SCORING_PROBE_SHA_RESTORED" == "$SCORING_PROBE_SHA_BEFORE" ]] || {
	echo "Scoring root mode or probe bytes changed across isolation." >&2; exit 4
}

"$PY_RGB" -c 'import hashlib,json,sys
from pathlib import Path
output,source,v1,topk,worker,scoring,probe,log,backend=map(Path,sys.argv[1:10])
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
payload={
    "format":"object_graph_rgb_candidate_scoring_isolation_v1",
    "status":"complete",
    "source_benchmark_root":str(source.resolve()),
    "v1_preflight_root":str(v1.resolve()),
    "mask_topk_preflight_root":str(topk.resolve()),
    "worker_input_root":str(worker.resolve()),
    "scoring_root":str(scoring.resolve()),
    "scoring_probe_relative_to_source":sys.argv[10],
    "root_mode_before":sys.argv[11],
    "root_mode_locked":"000",
    "root_mode_restored":sys.argv[12],
    "probe_sha256_before":sys.argv[13],
    "probe_sha256_restored":sys.argv[14],
    "locked_utc":sys.argv[15],
    "backend_completed_utc":sys.argv[16],
    "restored_utc":sys.argv[17],
    "backend_completed_before_restore":True,
    "backend_device":"cuda:0",
    "cuda_visible_devices":sys.argv[18],
    "BENCHMARK_GPU_UUID":sys.argv[18],
    "rgb_backend_manifest_relative_to_summary_root":sys.argv[19],
    "rgb_backend_manifest_sha256":sha(backend),
    "same_uid_read_probe":{
        "exit_code":1,
        "error_type":"PermissionError",
        "log_relative_to_summary_root":"contracts/scoring_read_probe.log",
        "log_sha256":sha(log),
    },
}
output.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")' \
	"$ISOLATION_GATE" "$SOURCE_BENCHMARK_ROOT" "$V1_PREFLIGHT_ROOT" \
	"$MASK_TOPK_PREFLIGHT_ROOT" "$SOURCE_BENCHMARK_ROOT/worker_inputs" \
	"$SCORING_ROOT" "$scoring_probe" "$STAGE/contracts/scoring_read_probe.log" \
	"$RGB_BACKEND_MANIFEST" "$SCORING_PROBE_RELATIVE" "$SCORING_ROOT_MODE_BEFORE" \
	"$SCORING_ROOT_MODE_RESTORED" "$SCORING_PROBE_SHA_BEFORE" \
	"$SCORING_PROBE_SHA_RESTORED" "$SCORING_LOCK_STARTED_UTC" \
	"$BACKEND_COMPLETED_UTC" "$SCORING_RESTORED_UTC" "$BENCHMARK_GPU_UUID" \
	"$RGB_BACKEND_RELATIVE"

env CUDA_VISIBLE_DEVICES="" "$PY_RGB" -B -m tdmpc2.common.object_graph_rgb_snapshot \
	--source-benchmark-root "$SOURCE_BENCHMARK_ROOT" \
	--v1-preflight-root "$V1_PREFLIGHT_ROOT" \
	--mask-topk-preflight-root "$MASK_TOPK_PREFLIGHT_ROOT" \
	--dino-repo "$DINO_REPO" --dino-checkpoint "$DINO_CHECKPOINT" \
	--verify "$INPUTS" >"$STAGE/contracts/inputs_verify_before_aggregate.log" 2>&1

run_tracked env CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 \
	"$PY_RGB" -B -m tdmpc2.tools.aggregate_object_graph_rgb_candidates \
	--source-benchmark-root "$SOURCE_BENCHMARK_ROOT" \
	--v1-preflight-root "$V1_PREFLIGHT_ROOT" \
	--mask-topk-preflight-root "$MASK_TOPK_PREFLIGHT_ROOT" \
	--rgb-backend-manifest "$RGB_BACKEND_MANIFEST" --v1-graph "$V1_GRAPH" \
	--dino-repo "$DINO_REPO" --dino-checkpoint "$DINO_CHECKPOINT" \
	--isolation-gate "$ISOLATION_GATE" --immutable-inputs "$INPUTS" \
	--output "$SUMMARY" >"$STAGE/logs/aggregate.log" 2>&1

env CUDA_VISIBLE_DEVICES="" "$PY_RGB" -B -m tdmpc2.common.object_graph_rgb_snapshot \
	--source-benchmark-root "$SOURCE_BENCHMARK_ROOT" \
	--v1-preflight-root "$V1_PREFLIGHT_ROOT" \
	--mask-topk-preflight-root "$MASK_TOPK_PREFLIGHT_ROOT" \
	--dino-repo "$DINO_REPO" --dino-checkpoint "$DINO_CHECKPOINT" \
	--verify "$INPUTS" >"$STAGE/contracts/inputs_verify_after_aggregate.log" 2>&1
seal_backend_tree verify
"$PY_RGB" -I -c 'import json,sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if p.get("engineering_pass") is not True:
    raise RuntimeError("RGB aggregate did not complete engineering validation.")
if p.get("controller_training_authorized") is not False or p.get("scientific_go") is not False or p.get("scope",{}).get("controller_training_steps") != 0:
    raise RuntimeError("RGB diagnostic attempted to authorize controller training or scientific claims.")' "$SUMMARY"

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
echo "OBJECT_GRAPH_RGB_PREFLIGHT_COMPLETE"
echo "OUTPUT_ROOT=$BASE"
echo "SUMMARY=$BASE/object_graph_rgb_candidate_coverage_summary.json"
