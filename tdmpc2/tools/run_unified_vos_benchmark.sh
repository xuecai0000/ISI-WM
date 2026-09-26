#!/usr/bin/env bash
# Frozen, controller-free VOS frontend selection across three tasks and two backgrounds.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${VIDEO_ROOT:?Set VIDEO_ROOT to the frozen video_hard directory}"
: "${OC_REPO:?Set OC_REPO to the OC-STORM checkout}"
: "${CUTIE_CKPT:?Set CUTIE_CKPT to cutie-small-mega.pth}"
: "${PY_SAM21:?Set PY_SAM21 to the dedicated official SAM 2.1 Python}"
: "${SAM21_REPO:?Set SAM21_REPO to the official facebookresearch/sam2 checkout}"
: "${SAM21_CKPT:?Set SAM21_CKPT to sam2.1_hiera_large.pt}"
: "${PY_SAM31:?Set PY_SAM31 to the dedicated official SAM 3.1 Python >=3.12}"
: "${SAM31_REPO:?Set SAM31_REPO to the official facebookresearch/sam3 checkout}"
: "${SAM31_CKPT:?Set SAM31_CKPT to the SAM 3.1 checkpoint}"
: "${SAM31_BPE:?Set SAM31_BPE to the tokenizer BPE file used by SAM 3.1}"

PY_CORE="${PY_CORE:-python}"
PY_CUTIE="${PY_CUTIE:-$PY_CORE}"
MANIFEST_DIR="${MANIFEST_DIR:-$REPO_ROOT/tdmpc2/envs/background_manifests}"
GPU="${GPU:-0}"
RUN_TAG="${RUN_TAG:-unified_vos_benchmark_128_v1}"
BASE="${OUTPUT_ROOT:-$REPO_ROOT/logs/_diagnostic/$RUN_TAG}"
STAGE="${BASE}.incomplete"
SUMMARY="$STAGE/unified_vos_summary.json"
EXTERNAL_INPUTS="$STAGE/provenance/external_inputs.json"
ENVIRONMENT_INPUTS="$STAGE/provenance/environment_inputs.json"
SCORING_ISOLATION="$STAGE/provenance/scoring_isolation_gate.json"
DATASET_ROOT="$STAGE/dataset"
DATASET_MANIFEST="$DATASET_ROOT/dataset_manifest.json"
WORKER_INPUT_ROOT="$STAGE/worker_inputs"
BACKEND_INPUTS="$WORKER_INPUT_ROOT/backend_inputs.json"
SAM31_FAILURE="$STAGE/backends/sam31_failure.json"

EPISODES=20
STEPS=500
RESOLUTION=128
ENV_SEED=424243
BACKGROUND_SEED=1618034
ACTION_SEED=8675400
SUPPORT_SEED=314159
SUPPORT_BACKGROUND_SEED=314160
SUPPORT_ACTION_SEED=314161
BACKEND_SEED=2718281
TRACKED_TIMEOUT_SECONDS="${TRACKED_TIMEOUT_SECONDS:-43200}"
TASKS=(reacher-visual-small cartpole-swingup acrobot-swingup)

[[ "$GPU" =~ ^(0|[1-9][0-9]*)$ ]] || {
	echo "GPU must be a non-negative physical index: $GPU" >&2
	exit 2
}
[[ "$TRACKED_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
	echo "TRACKED_TIMEOUT_SECONDS must be a positive integer." >&2
	exit 2
}
[[ "$(basename -- "${VIDEO_ROOT%/}")" == video_hard ]] || {
	echo "VIDEO_ROOT must be the frozen video_hard directory." >&2
	exit 2
}

check_python() {
	local executable=$1
	if [[ "$executable" == */* ]]; then
		[[ -x "$executable" ]] || {
			echo "Python is not executable: $executable" >&2
			exit 2
		}
	else
		command -v "$executable" >/dev/null || {
			echo "Python is not on PATH: $executable" >&2
			exit 2
		}
	fi
}

for executable in "$PY_CORE" "$PY_CUTIE" "$PY_SAM21" "$PY_SAM31"; do
	check_python "$executable"
done
for path in \
	"$VIDEO_ROOT" "$MANIFEST_DIR" "$OC_REPO" "$CUTIE_CKPT" \
	"$SAM21_REPO" "$SAM21_CKPT" "$SAM31_REPO" "$SAM31_CKPT" "$SAM31_BPE"; do
	[[ -e "$path" ]] || { echo "Missing immutable input: $path" >&2; exit 2; }
done
[[ -d "$VIDEO_ROOT" && -d "$MANIFEST_DIR" && -d "$OC_REPO" \
	&& -d "$SAM21_REPO" && -d "$SAM31_REPO" ]] || {
	echo "All repository/video/manifest inputs must be directories." >&2
	exit 2
}
[[ -f "$CUTIE_CKPT" && -f "$SAM21_CKPT" && -f "$SAM31_CKPT" \
	&& -f "$SAM31_BPE" ]] || {
	echo "All checkpoint/BPE inputs must be regular files." >&2
	exit 2
}

mkdir -p -- "$(dirname -- "$BASE")"
[[ ! -e "$BASE" ]] || {
	echo "Refusing to overwrite existing output: $BASE" >&2
	exit 3
}
if ! mkdir -- "$STAGE"; then
	echo "Refusing to share or overwrite active stage: $STAGE" >&2
	exit 3
fi
STAGE_OWNED=1
PROMOTED=0
SCORING_LOCKED=0
ACTIVE_PIDS=()

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/tdmpc2${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

terminate_tree() {
	local parent=$1 child
	while IFS= read -r child; do
		[[ -n "$child" ]] && terminate_tree "$child"
	done < <(pgrep -P "$parent" 2>/dev/null || true)
	kill -TERM "$parent" 2>/dev/null || true
}

archive_on_exit() {
	local rc=$? pid failed attempt alive
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
	# Never restore GT visibility while a backend process group may still exist.
	if (( SCORING_LOCKED == 1 )) && [[ -d "$DATASET_ROOT/scoring" ]]; then
		if chmod "$SCORING_ROOT_MODE_BEFORE" -- "$DATASET_ROOT/scoring"; then
			SCORING_LOCKED=0
		else
			echo "Failed to restore scoring permissions during failure cleanup." >&2
		fi
	fi
	if (( STAGE_OWNED == 1 && PROMOTED == 0 )) && [[ -d "$STAGE" ]]; then
		# A failure after aggregation (for example an atomic-promotion race) must
		# never leave an engineering-pass summary inside a failed archive.
		"$PY_CORE" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); rc=int(sys.argv[2]); payload={};
try:
 payload=json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
except Exception:
 payload={}
original=payload.get("status")
payload.update({"format":"unified_vos_benchmark_summary_v1","status":"runner_engineering_fail","engineering_pass":False,"controller_pilot_go":False,"scientific_selection_pass":False,"paper_claim_ready":False,"recommended_backend":None,"eligible_online_backends":[],"runner_exit_code":rc,"recommendation":"fix_engineering_failure_do_not_train_controller"})
if original is not None and original != "runner_engineering_fail": payload["pre_failure_aggregate_status"]=original
p.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")' "$SUMMARY" "$rc" 2>/dev/null || true
		failed="${BASE}.failed.$(date +%Y%m%d_%H%M%S).$$"
		[[ ! -e "$failed" ]] || failed="${failed}.${RANDOM}"
		mv -T -- "$STAGE" "$failed"
		echo "UNIFIED_VOS_BENCHMARK_FAILED_ARCHIVE=$failed" >&2
	fi
	exit "$rc"
}
trap archive_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -- "$STAGE/contracts" "$STAGE/logs" "$STAGE/provenance" \
	"$STAGE/support" "$STAGE/backends"

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
	command -v nvidia-smi >/dev/null || {
		echo "nvidia-smi is required for the same-GPU no-concurrency gate." >&2
		return 2
	}
	while (( elapsed <= 120 )); do
		active="$(nvidia-smi --id="$BENCHMARK_GPU_UUID" --query-compute-apps=pid \
			--format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
		[[ -z "$active" ]] && return 0
		(( elapsed == 0 )) && echo "Waiting at most 120s for physical GPU=$GPU to become idle"
		sleep 5
		elapsed=$((elapsed + 5))
	done
	echo "GPU=$GPU remained busy; no external process was signaled: $active" >&2
	return 5
}

record_optional_sam31_failure() {
	local rc=$1 log=$2
	[[ ! -e "$SAM31_FAILURE" && -f "$log" ]] || return 2
	"$PY_CORE" -c 'import hashlib,json,sys; from pathlib import Path; output=Path(sys.argv[1]); log=Path(sys.argv[2]); root=Path(sys.argv[3]); rc=int(sys.argv[4]); relative=log.resolve().relative_to(root.resolve()).as_posix(); digest=hashlib.sha256(log.read_bytes()).hexdigest(); payload={"format":"unified_vos_optional_backend_failure_v1","status":"diagnostic_failed","backend":"sam31","exit_code":rc,"log":relative,"log_sha256":digest,"online_deployment_eligible":False,"recommendation":"online_selection_continues_without_offline_diagnostic"}; output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")' \
		"$SAM31_FAILURE" "$log" "$STAGE" "$rc"
}

file_mode() {
	local raw
	raw="$(stat -c '%a' -- "$1")"
	printf '%03o' "$((8#$raw))"
}

file_sha256() {
	"$PY_CORE" -I -c 'import hashlib,sys; from pathlib import Path; print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())' "$1"
}

resolved_path() {
	"$PY_CORE" -I -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).resolve(strict=True))' "$1"
}

snapshot_args=(
	--repo-root "$REPO_ROOT"
	--video-root "$VIDEO_ROOT"
	--manifest-dir "$MANIFEST_DIR"
	--oc-repo "$OC_REPO"
	--cutie-checkpoint "$CUTIE_CKPT"
	--sam21-repo "$SAM21_REPO"
	--sam21-checkpoint "$SAM21_CKPT"
	--sam31-repo "$SAM31_REPO"
	--sam31-checkpoint "$SAM31_CKPT"
	--sam31-bpe "$SAM31_BPE"
)

environment_args=(
	--environment "core=$PY_CORE"
	--environment "cutie=$PY_CUTIE"
	--environment "sam21=$PY_SAM21"
	--environment "sam31=$PY_SAM31"
)

echo "[1/7] Static contracts, isolated GPU environments, and immutable inputs"
bash -n "$0"
command -v nvidia-smi >/dev/null || {
	echo "nvidia-smi is required." >&2
	exit 2
}
command -v timeout >/dev/null && command -v setsid >/dev/null || {
	echo "GNU timeout and setsid are required for bounded process cleanup." >&2
	exit 2
}
BENCHMARK_GPU_UUID="$(nvidia-smi --id="$GPU" --query-gpu=uuid \
	--format=csv,noheader,nounits | tr -d '[:space:]')"
[[ "$BENCHMARK_GPU_UUID" =~ ^GPU-[0-9A-Fa-f-]+$ ]] || {
	echo "Could not bind one physical GPU UUID for index $GPU: $BENCHMARK_GPU_UUID" >&2
	exit 2
}
export BENCHMARK_GPU_UUID
"$PY_CORE" -m py_compile \
	tdmpc2/common/cutie_paired_support.py \
	tdmpc2/common/unified_vos.py \
	tdmpc2/common/unified_vos_snapshot.py \
	tdmpc2/common/unified_vos_environment_snapshot.py \
	tdmpc2/check_unified_vos_contract.py \
	tdmpc2/perception/sam21_video_backend.py \
	tdmpc2/perception/sam31_backend.py \
	tdmpc2/tools/check_sam21_video_backend.py \
	tdmpc2/check_sam31_backend_contract.py \
	tdmpc2/check_cutie_paired_native_support_contract.py \
	tdmpc2/tools/collect_cutie_multitask_support.py \
	tdmpc2/tools/collect_cutie_paired_native_support.py \
	tdmpc2/tools/collect_unified_vos_dataset.py \
	tdmpc2/tools/run_unified_vos_cutie_backend.py \
	tdmpc2/tools/sam31_backend_worker.py \
	tdmpc2/tools/aggregate_unified_vos_benchmark.py
"$PY_CORE" -m tdmpc2.check_unified_vos_contract \
	>"$STAGE/contracts/unified_vos.log" 2>&1
"$PY_CORE" -m tdmpc2.tools.check_sam21_video_backend \
	>"$STAGE/contracts/sam21_dependency_light.log" 2>&1
"$PY_CORE" -m tdmpc2.check_sam31_backend_contract \
	>"$STAGE/contracts/sam31_dependency_light.log" 2>&1
"$PY_CORE" -m tdmpc2.check_cutie_paired_native_support_contract \
	>"$STAGE/contracts/paired_support.log" 2>&1
"$PY_CORE" -m tdmpc2.check_cutie_native_highres_contract \
	>"$STAGE/contracts/native_highres.log" 2>&1
"$PY_CORE" -m tdmpc2.check_cutie_native_resolution_wrapper_contract \
	>"$STAGE/contracts/native_wrapper.log" 2>&1

for item in "core:$PY_CORE" "cutie:$PY_CUTIE" "sam21:$PY_SAM21" "sam31:$PY_SAM31"; do
	label="${item%%:*}"
	executable="${item#*:}"
	CUDA_VISIBLE_DEVICES="$BENCHMARK_GPU_UUID" "$executable" -c \
		'import json,os,torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(json.dumps({"cuda_visible_devices":os.environ["CUDA_VISIBLE_DEVICES"],"gpu_uuid":os.environ["BENCHMARK_GPU_UUID"],"cuda_device_order":os.environ["CUDA_DEVICE_ORDER"],"logical_cuda_device":0,"device_name":torch.cuda.get_device_name(0),"torch":torch.__version__,"cuda":torch.version.cuda}))' \
		>"$STAGE/contracts/gpu_${label}.json" 2>"$STAGE/contracts/gpu_${label}.log"
done
"$PY_CORE" -c 'import json,sys; from pathlib import Path; rows=[json.loads(Path(p).read_text()) for p in sys.argv[1:]]; assert all(r["logical_cuda_device"]==0 and r["cuda_device_order"]=="PCI_BUS_ID" for r in rows); names={r["device_name"] for r in rows}; uuids={r["gpu_uuid"] for r in rows}; assert len(names)==len(uuids)==1, rows' \
	"$STAGE/contracts/gpu_core.json" "$STAGE/contracts/gpu_cutie.json" \
	"$STAGE/contracts/gpu_sam21.json" "$STAGE/contracts/gpu_sam31.json"
"$PY_CORE" -m tdmpc2.common.unified_vos_snapshot \
	"${snapshot_args[@]}" --output "$EXTERNAL_INPUTS" \
	>"$STAGE/contracts/external_inputs_bind.log" 2>&1
"$PY_CORE" -m tdmpc2.common.unified_vos_environment_snapshot \
	"${environment_args[@]}" --output "$ENVIRONMENT_INPUTS" \
	>"$STAGE/contracts/environment_inputs_bind.log" 2>&1

wait_gpu_idle
echo "MODEL_PREFLIGHT_START backend=cutie gpu=$GPU"
run_tracked env CUDA_VISIBLE_DEVICES="$BENCHMARK_GPU_UUID" \
	"$PY_CUTIE" -B -m tdmpc2.tools.run_unified_vos_cutie_backend \
	--oc-storm-repo "$OC_REPO" --checkpoint "$CUTIE_CKPT" \
	--model-size small --tracker-size 448 --preflight-only \
	>"$STAGE/contracts/cutie_real_model_preflight.log" 2>&1
echo "MODEL_PREFLIGHT_END backend=cutie gpu=$GPU rc=0"

wait_gpu_idle
echo "MODEL_PREFLIGHT_START backend=sam21 gpu=$GPU"
run_tracked env CUDA_VISIBLE_DEVICES="$BENCHMARK_GPU_UUID" \
	"$PY_SAM21" -B -m tdmpc2.perception.sam21_video_backend \
	--sam2-repo "$SAM21_REPO" --checkpoint "$SAM21_CKPT" \
	--model-size large --device cuda:0 --seed "$BACKEND_SEED" \
	--amp-dtype bfloat16 --preflight-only \
	>"$STAGE/contracts/sam21_real_model_preflight.log" 2>&1
echo "MODEL_PREFLIGHT_END backend=sam21 gpu=$GPU rc=0"

SAM31_READY=1
wait_gpu_idle
echo "MODEL_PREFLIGHT_START backend=sam31 gpu=$GPU online_eligible=false"
if run_tracked env CUDA_VISIBLE_DEVICES="$BENCHMARK_GPU_UUID" \
	"$PY_SAM31" -B -m tdmpc2.tools.sam31_backend_worker \
	--sam3-repo "$SAM31_REPO" --checkpoint "$SAM31_CKPT" --bpe "$SAM31_BPE" \
	--device cuda:0 --seed "$BACKEND_SEED" --preflight-only --traceback \
	>"$STAGE/logs/sam31_preflight.log" 2>&1; then
	echo "MODEL_PREFLIGHT_END backend=sam31 gpu=$GPU rc=0"
else
	sam31_rc=$?
	SAM31_READY=0
	record_optional_sam31_failure "$sam31_rc" "$STAGE/logs/sam31_preflight.log"
	echo "MODEL_PREFLIGHT_END backend=sam31 gpu=$GPU rc=$sam31_rc optional_diagnostic_skipped=true"
fi

echo "[2/7] Same-state native128 six-frame support for all three tasks"
for task in "${TASKS[@]}"; do
	log="$STAGE/logs/support_${task}.log"
	echo "SUPPORT_START task=$task"
	run_tracked env CUDA_VISIBLE_DEVICES="$BENCHMARK_GPU_UUID" \
		"$PY_CORE" -B -m tdmpc2.tools.collect_cutie_paired_native_support \
		--task "$task" \
		--output "$STAGE/support/$task" \
		--video-root "$VIDEO_ROOT" \
		--manifest-dir "$MANIFEST_DIR" \
		--seed "$SUPPORT_SEED" \
		--background-seed "$SUPPORT_BACKGROUND_SEED" \
		--action-seed "$SUPPORT_ACTION_SEED" \
		--max-reset-attempts 1000 >"$log" 2>&1
	echo "SUPPORT_END task=$task rc=0"
done

echo "[3/7] Freeze identical RGB trajectories and scoring-only GT masks"
support_args=()
for task in "${TASKS[@]}"; do
	support_args+=(--support "$task=$STAGE/support/$task/annotations.json")
done
echo "DATASET_START resolution=$RESOLUTION episodes=$EPISODES steps=$STEPS"
run_tracked env CUDA_VISIBLE_DEVICES="$BENCHMARK_GPU_UUID" \
	"$PY_CORE" -B -m tdmpc2.tools.collect_unified_vos_dataset \
	--output "$DATASET_ROOT" \
	--worker-output "$WORKER_INPUT_ROOT" \
	"${support_args[@]}" \
	--video-root "$VIDEO_ROOT" \
	--manifest-dir "$MANIFEST_DIR" \
	--resolution "$RESOLUTION" \
	--episodes "$EPISODES" \
	--steps "$STEPS" \
	--env-seed "$ENV_SEED" \
	--background-seed "$BACKGROUND_SEED" \
	--action-seed "$ACTION_SEED" \
	--support-seed "$SUPPORT_SEED" \
	>"$STAGE/logs/dataset.log" 2>&1
echo "DATASET_END rc=0"

chmod -R a-w -- "$WORKER_INPUT_ROOT"
DATASET_RESOLVED="$(resolved_path "$DATASET_ROOT")"
WORKER_RESOLVED="$(resolved_path "$WORKER_INPUT_ROOT")"
SCORING_RESOLVED="$(resolved_path "$DATASET_ROOT/scoring")"
"$PY_CORE" -I -c 'import sys; from pathlib import Path; dataset=Path(sys.argv[1]); worker=Path(sys.argv[2]); assert dataset != worker and dataset not in worker.parents and worker not in dataset.parents' \
	"$DATASET_RESOLVED" "$WORKER_RESOLVED"
worker_scoring_entry="$(find "$WORKER_INPUT_ROOT" -mindepth 1 -name scoring -print -quit)"
[[ -z "$worker_scoring_entry" ]] || {
	echo "GT-free worker root unexpectedly contains a scoring-named entry: $worker_scoring_entry" >&2
	exit 4
}
WORKER_ROOT_MODE_READONLY="$(file_mode "$WORKER_INPUT_ROOT")"
WORKER_MANIFEST_MODE_READONLY="$(file_mode "$BACKEND_INPUTS")"
(( (8#$WORKER_ROOT_MODE_READONLY & 8#222) == 0 )) || {
	echo "GT-free worker root is still writable." >&2
	exit 4
}
(( (8#$WORKER_MANIFEST_MODE_READONLY & 8#222) == 0 )) || {
	echo "GT-free worker manifest is still writable." >&2
	exit 4
}

# Resolve the scoring probe before removing directory traversal permission.
# Calling find after chmod 000 would fail for the non-root benchmark user.
scoring_probe="$(find "$DATASET_ROOT/scoring" -type f -print -quit)"
[[ -n "$scoring_probe" && -f "$scoring_probe" && -r "$scoring_probe" ]] || {
	echo "Could not bind one readable scoring-only probe before the lock." >&2
	exit 4
}
SCORING_ROOT_MODE_BEFORE="$(file_mode "$DATASET_ROOT/scoring")"
SCORING_PROBE_MODE_BEFORE="$(file_mode "$scoring_probe")"
SCORING_PROBE_SHA_BEFORE="$(file_sha256 "$scoring_probe")"
SCORING_LOCK_STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SCORING_LOCKED=1
# Backends are launched only after this point and inherit no descriptor for
# scoring.  Removing traversal permission from the root atomically makes the
# entire subtree unreadable without recursively changing child permissions.
chmod 000 -- "$DATASET_ROOT/scoring"
SCORING_ROOT_MODE_LOCKED="$(file_mode "$DATASET_ROOT/scoring")"
[[ "$SCORING_ROOT_MODE_LOCKED" == 000 && ! -r "$scoring_probe" ]] || {
	echo "Scoring-only GT tree is still readable during backend inference." >&2
	exit 4
}
LOCKED_READ_PROBE_LOG="$STAGE/contracts/scoring_locked_read_probe.log"
set +e
"$PY_CORE" -I -c 'import sys; from pathlib import Path; Path(sys.argv[1]).read_bytes()' \
	"$scoring_probe" >"$LOCKED_READ_PROBE_LOG" 2>&1
LOCKED_READ_PROBE_RC=$?
set -e
(( LOCKED_READ_PROBE_RC == 1 )) && grep -Fq 'PermissionError' "$LOCKED_READ_PROBE_LOG" || {
	echo "Same-user scoring read probe did not fail specifically with PermissionError." >&2
	exit 4
}
[[ -s "$LOCKED_READ_PROBE_LOG" ]] || {
	echo "Locked scoring read probe failed without auditable diagnostics." >&2
	exit 4
}
SCORING_READ_PROBE_COMPLETED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if (( SAM31_READY == 1 )); then
	wait_gpu_idle
	echo "PROTOCOL_SMOKE_START backend=sam31 gpu=$GPU online_eligible=false"
	if run_tracked env CUDA_VISIBLE_DEVICES="$BENCHMARK_GPU_UUID" \
		"$PY_SAM31" -B -m tdmpc2.tools.sam31_backend_worker \
		--inputs "$BACKEND_INPUTS" \
		--sam3-repo "$SAM31_REPO" --checkpoint "$SAM31_CKPT" --bpe "$SAM31_BPE" \
		--device cuda:0 --seed "$BACKEND_SEED" --protocol-smoke-only --traceback \
		>"$STAGE/logs/sam31_protocol_smoke.log" 2>&1; then
		echo "PROTOCOL_SMOKE_END backend=sam31 gpu=$GPU rc=0"
	else
		sam31_rc=$?
		SAM31_READY=0
		record_optional_sam31_failure "$sam31_rc" "$STAGE/logs/sam31_protocol_smoke.log"
		echo "PROTOCOL_SMOKE_END backend=sam31 gpu=$GPU rc=$sam31_rc optional_diagnostic_skipped=true"
	fi
fi

echo "[4/7] Cutie baseline on the frozen GT-free inputs"
wait_gpu_idle
echo "BACKEND_START backend=cutie gpu=$GPU"
run_tracked env CUDA_VISIBLE_DEVICES="$BENCHMARK_GPU_UUID" \
	"$PY_CUTIE" -B -m tdmpc2.tools.run_unified_vos_cutie_backend \
	--inputs "$BACKEND_INPUTS" \
	--output-root "$STAGE/backends/cutie" \
	--oc-storm-repo "$OC_REPO" \
	--checkpoint "$CUTIE_CKPT" \
	--model-size small \
	--tracker-size 448 \
	--seed "$BACKEND_SEED" \
	--strict-counts >"$STAGE/logs/cutie.log" 2>&1
echo "BACKEND_END backend=cutie gpu=$GPU rc=0"

echo "[5/7] SAM 2.1-L on the exact same frozen inputs"
wait_gpu_idle
echo "BACKEND_START backend=sam21 gpu=$GPU"
run_tracked env CUDA_VISIBLE_DEVICES="$BENCHMARK_GPU_UUID" \
	"$PY_SAM21" -B -m tdmpc2.perception.sam21_video_backend \
	--inputs "$BACKEND_INPUTS" \
	--output-root "$STAGE/backends/sam21" \
	--sam2-repo "$SAM21_REPO" \
	--checkpoint "$SAM21_CKPT" \
	--model-size large \
	--device cuda:0 \
	--seed "$BACKEND_SEED" \
	--amp-dtype bfloat16 \
	--traceback >"$STAGE/logs/sam21.log" 2>&1
echo "BACKEND_END backend=sam21 gpu=$GPU rc=0"

echo "[6/7] SAM 3.1 offline full-video diagnostic on the same inputs"
if (( SAM31_READY == 1 )); then
	wait_gpu_idle
	echo "BACKEND_START backend=sam31 gpu=$GPU online_eligible=false"
	if run_tracked env CUDA_VISIBLE_DEVICES="$BENCHMARK_GPU_UUID" \
		"$PY_SAM31" -B -m tdmpc2.tools.sam31_backend_worker \
		--inputs "$BACKEND_INPUTS" \
		--output-root "$STAGE/backends/sam31" \
		--sam3-repo "$SAM31_REPO" \
		--checkpoint "$SAM31_CKPT" \
		--bpe "$SAM31_BPE" \
		--device cuda:0 \
		--seed "$BACKEND_SEED" \
		--traceback >"$STAGE/logs/sam31.log" 2>&1; then
		echo "BACKEND_END backend=sam31 gpu=$GPU rc=0"
	else
		sam31_rc=$?
		SAM31_READY=0
		record_optional_sam31_failure "$sam31_rc" "$STAGE/logs/sam31.log"
		echo "BACKEND_END backend=sam31 gpu=$GPU rc=$sam31_rc optional_diagnostic_failed=true"
	fi
else
	echo "BACKEND_SKIP backend=sam31 reason=optional_preflight_or_protocol_smoke_failed"
fi

echo "[7/7] Immutable recheck, mask-only scoring, and controller-pilot decision"
(( ${#ACTIVE_PIDS[@]} == 0 )) || {
	echo "Refusing to restore scoring permissions while a tracked backend is active." >&2
	exit 4
}
chmod "$SCORING_ROOT_MODE_BEFORE" -- "$DATASET_ROOT/scoring"
SCORING_RESTORED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SCORING_ROOT_MODE_RESTORED="$(file_mode "$DATASET_ROOT/scoring")"
SCORING_PROBE_MODE_RESTORED="$(file_mode "$scoring_probe")"
SCORING_PROBE_SHA_RESTORED="$(file_sha256 "$scoring_probe")"
[[ -r "$scoring_probe" && "$SCORING_PROBE_SHA_RESTORED" == "$SCORING_PROBE_SHA_BEFORE" ]] || {
	echo "Scoring probe was not restored byte-for-byte after backend inference." >&2
	exit 4
}
SCORING_LOCKED=0
"$PY_CORE" -m tdmpc2.common.unified_vos_snapshot \
	"${snapshot_args[@]}" --verify "$EXTERNAL_INPUTS" \
	>"$STAGE/contracts/external_inputs_verify.log" 2>&1
"$PY_CORE" -m tdmpc2.common.unified_vos_environment_snapshot \
	"${environment_args[@]}" --verify "$ENVIRONMENT_INPUTS" \
	>"$STAGE/contracts/environment_inputs_verify.log" 2>&1
SCORING_GATE_WRITTEN_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY_CORE" -I -c '
import hashlib, json, sys
from pathlib import Path

(
    output, summary_root, dataset_root, worker_root, scoring_root,
    dataset_manifest, worker_manifest, probe, probe_log,
    root_before, root_locked, root_restored, probe_before,
    probe_restored, worker_root_mode, worker_manifest_mode,
    read_rc, lock_started, read_completed, restored, gate_written,
    probe_sha_before,
) = sys.argv[1:]
output = Path(output)
summary_root = Path(summary_root).resolve(strict=True)
dataset_root = Path(dataset_root).resolve(strict=True)
worker_root = Path(worker_root).resolve(strict=True)
scoring_root = Path(scoring_root).resolve(strict=True)
dataset_manifest = Path(dataset_manifest).resolve(strict=True)
worker_manifest = Path(worker_manifest).resolve(strict=True)
probe = Path(probe).resolve(strict=True)
probe_log = Path(probe_log).resolve(strict=True)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
dataset_contains_worker = dataset_root in worker_root.parents
worker_contains_dataset = worker_root in dataset_root.parents
disjoint = dataset_root != worker_root and not dataset_contains_worker and not worker_contains_dataset
if not disjoint:
    raise RuntimeError("Dataset and worker roots are not disjoint.")
probe_relative = probe.relative_to(dataset_root).as_posix()
if not probe_relative.startswith("scoring/"):
    raise RuntimeError("Scoring probe is outside the scoring tree.")
payload = {
    "format": "unified_vos_scoring_isolation_gate_v1",
    "status": "complete",
    "roots": {
        "dataset_resolved_at_execution": str(dataset_root),
        "worker_resolved_at_execution": str(worker_root),
        "scoring_resolved_at_execution": str(scoring_root),
        "dataset_relative_to_summary_root": dataset_root.relative_to(summary_root).as_posix(),
        "worker_relative_to_summary_root": worker_root.relative_to(summary_root).as_posix(),
        "scoring_relative_to_summary_root": scoring_root.relative_to(summary_root).as_posix(),
        "dataset_worker_disjoint": disjoint,
        "dataset_contains_worker": dataset_contains_worker,
        "worker_contains_dataset": worker_contains_dataset,
    },
    "worker_view": {
        "backend_inputs_relative": worker_manifest.relative_to(worker_root).as_posix(),
        "backend_inputs_sha256": sha(worker_manifest),
        "root_mode_after_readonly": worker_root_mode,
        "backend_inputs_mode_after_readonly": worker_manifest_mode,
        "scoring_entry_absent": True,
    },
    "scoring_lock": {
        "tree_relative_to_dataset_root": scoring_root.relative_to(dataset_root).as_posix(),
        "root_mode_before_lock": root_before,
        "root_mode_while_locked": root_locked,
        "root_mode_after_restore": root_restored,
        "probe_relative_to_dataset_root": probe_relative,
        "probe_mode_before_lock": probe_before,
        "probe_mode_after_restore": probe_restored,
        "probe_sha256_before_lock": probe_sha_before,
        "probe_sha256_after_restore": sha(probe),
        "probe_readable_before_lock": True,
        "probe_readable_while_locked": False,
        "locked_read_probe_exit_code": int(read_rc),
        "locked_read_probe_outcome": "permission_error_same_uid",
        "locked_read_probe_log_relative_to_summary_root": probe_log.relative_to(summary_root).as_posix(),
        "locked_read_probe_log_sha256": sha(probe_log),
    },
    "timing_utc": {
        "lock_started": lock_started,
        "read_probe_completed": read_completed,
        "restored": restored,
        "gate_written": gate_written,
    },
    "dataset_manifest_sha256": sha(dataset_manifest),
}
if output.exists() or output.parent.resolve(strict=True) != (summary_root / "provenance").resolve(strict=True):
    raise RuntimeError("Unsafe or existing scoring-isolation output.")
output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
' \
	"$SCORING_ISOLATION" "$STAGE" "$DATASET_RESOLVED" "$WORKER_RESOLVED" \
	"$SCORING_RESOLVED" "$DATASET_MANIFEST" "$BACKEND_INPUTS" "$scoring_probe" \
	"$LOCKED_READ_PROBE_LOG" "$SCORING_ROOT_MODE_BEFORE" \
	"$SCORING_ROOT_MODE_LOCKED" "$SCORING_ROOT_MODE_RESTORED" \
	"$SCORING_PROBE_MODE_BEFORE" "$SCORING_PROBE_MODE_RESTORED" \
	"$WORKER_ROOT_MODE_READONLY" "$WORKER_MANIFEST_MODE_READONLY" \
	"$LOCKED_READ_PROBE_RC" "$SCORING_LOCK_STARTED_UTC" \
	"$SCORING_READ_PROBE_COMPLETED_UTC" "$SCORING_RESTORED_UTC" \
	"$SCORING_GATE_WRITTEN_UTC" "$SCORING_PROBE_SHA_BEFORE"
if (( SAM31_READY == 1 )); then
	sam31_aggregate_args=(--sam31 "$STAGE/backends/sam31/backend_predictions.json")
else
	sam31_aggregate_args=(--sam31-failure "$SAM31_FAILURE")
fi
"$PY_CORE" -B -m tdmpc2.tools.aggregate_unified_vos_benchmark \
	--dataset-manifest "$DATASET_MANIFEST" \
	--cutie "$STAGE/backends/cutie/backend_predictions.json" \
	--sam21 "$STAGE/backends/sam21/backend_predictions.json" \
	"${sam31_aggregate_args[@]}" \
	--external-inputs "$EXTERNAL_INPUTS" \
	--environment-inputs "$ENVIRONMENT_INPUTS" \
	--scoring-isolation "$SCORING_ISOLATION" \
	--output "$SUMMARY" >"$STAGE/logs/aggregate.log" 2>&1

mv -T -- "$STAGE" "$BASE"
PROMOTED=1
SUMMARY="$BASE/unified_vos_summary.json"
echo "UNIFIED_VOS_BENCHMARK_COMPLETE"
echo "SUMMARY=$SUMMARY"
echo "No controller training was launched. This is a single-seed random-policy preflight."
