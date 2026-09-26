#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PY_CORE="${PY_CORE:-<DATA_PATH>/conda_envs/tdmpc2_2026/bin/python}"
SOURCE_KEYPOINT_ROOT="${SOURCE_KEYPOINT_ROOT:?Set SOURCE_KEYPOINT_ROOT}"
SOURCE_MASK_ROOT="${SOURCE_MASK_ROOT:?Set SOURCE_MASK_ROOT}"
SOURCE_FRONTEND_ROOT="${SOURCE_FRONTEND_ROOT:?Set SOURCE_FRONTEND_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set a new OUTPUT_ROOT path}"
WORK_ROOT="${OUTPUT_ROOT}.incomplete"
FRONTEND_CHECKPOINT="$SOURCE_FRONTEND_ROOT/checkpoints/acrobot_masked_keypoint_fallback_v2.pt"

for path in "$PY_CORE" "$FRONTEND_CHECKPOINT"; do
	if [[ ! -e "$path" ]]; then echo "Missing required input: $path" >&2; exit 2; fi
done
for split in train validation test; do
	for path in \
		"$SOURCE_KEYPOINT_ROOT/datasets/$split/manifest.json" \
		"$SOURCE_MASK_ROOT/masks/$split/manifest.json"; do
		if [[ ! -f "$path" ]]; then echo "Missing immutable source: $path" >&2; exit 2; fi
	done
done
if [[ -e "$OUTPUT_ROOT" || -e "$WORK_ROOT" ]]; then
	echo "Refusing to overwrite existing output: $OUTPUT_ROOT" >&2; exit 3
fi
mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/traces" "$WORK_ROOT/checkpoints"

failed_archive() {
	rc=$?
	if [[ $rc -ne 0 && -d "$WORK_ROOT" ]]; then
		failed="${OUTPUT_ROOT}.failed.$(date +%Y%m%d_%H%M%S).$$"
		mv "$WORK_ROOT" "$failed"
		echo "ACROBOT_STATE_OBSERVER_PREFLIGHT_FAILED_ARCHIVE=$failed"
	fi
	exit "$rc"
}
trap failed_archive EXIT

cd "$PROJECT_ROOT"
echo '[1/5] Static causality, streaming parity, topology, and checkpoint contracts'
"$PY_CORE" -B tdmpc2/check_acrobot_state_observer_contract.py \
	>"$WORK_ROOT/logs/contracts.log" 2>&1

echo '[2/5] Freezing causal frontend traces for disjoint train/validation/test episodes'
for split in train validation test; do
	echo "TRACE_EXPORT_START split=$split"
	"$PY_CORE" -B -m tdmpc2.tools.export_acrobot_keypoint_traces \
		--checkpoint "$FRONTEND_CHECKPOINT" \
		--input-manifest "$SOURCE_KEYPOINT_ROOT/datasets/$split/manifest.json" \
		--mask-manifest "$SOURCE_MASK_ROOT/masks/$split/manifest.json" \
		--output "$WORK_ROOT/traces/$split" --device cuda:0 \
		>"$WORK_ROOT/logs/traces_$split.log" 2>&1
	echo "TRACE_EXPORT_END split=$split rc=0"
done

echo '[3/5] Training action-conditioned GRU with random causal dropout bursts'
"$PY_CORE" -B -m tdmpc2.tools.train_acrobot_state_observer \
	--train-manifest "$WORK_ROOT/traces/train/manifest.json" \
	--validation-manifest "$WORK_ROOT/traces/validation/manifest.json" \
	--output "$WORK_ROOT/checkpoints/acrobot_state_observer_v1.pt" \
	--device cuda:0 --epochs 80 --batch-size 8 --maximum-dropout-burst 96 \
	>"$WORK_ROOT/logs/train.log" 2>&1

echo '[4/5] Held-out real-error long sequences and batch-one streaming latency gate'
"$PY_CORE" -B -m tdmpc2.tools.evaluate_acrobot_state_observer \
	--checkpoint "$WORK_ROOT/checkpoints/acrobot_state_observer_v1.pt" \
	--test-manifest "$WORK_ROOT/traces/test/manifest.json" \
	--output "$WORK_ROOT/acrobot_state_observer_summary.json" \
	--device cuda:0 --frontend-p95-ms 7.59 \
	>"$WORK_ROOT/logs/evaluate.log" 2>&1

echo '[5/5] Atomic publication; controller training is never launched here'
mv "$WORK_ROOT" "$OUTPUT_ROOT"
trap - EXIT
echo 'ACROBOT_STATE_OBSERVER_PREFLIGHT_COMPLETE'
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "SUMMARY=$OUTPUT_ROOT/acrobot_state_observer_summary.json"
echo 'RUNNER_RC=0'
