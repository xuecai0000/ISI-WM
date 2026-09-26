#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PY_CORE="${PY_CORE:-<DATA_PATH>/conda_envs/tdmpc2_2026/bin/python}"
SOURCE_KEYPOINT_ROOT="${SOURCE_KEYPOINT_ROOT:?Set SOURCE_KEYPOINT_ROOT}"
SOURCE_MASK_ROOT="${SOURCE_MASK_ROOT:?Set SOURCE_MASK_ROOT to the published V2 preflight}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set a new OUTPUT_ROOT path}"
WORK_ROOT="${OUTPUT_ROOT}.incomplete"

for split in train validation test; do
	for path in \
		"$SOURCE_KEYPOINT_ROOT/datasets/$split/manifest.json" \
		"$SOURCE_MASK_ROOT/masks/$split/manifest.json"; do
		if [[ ! -f "$path" ]]; then
			echo "Missing immutable source: $path" >&2
			exit 2
		fi
	done
done
if [[ ! -x "$PY_CORE" ]]; then
	echo "Missing Python interpreter: $PY_CORE" >&2
	exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -e "$WORK_ROOT" ]]; then
	echo "Refusing to overwrite existing output: $OUTPUT_ROOT" >&2
	exit 3
fi
mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/checkpoints"

failed_archive() {
	rc=$?
	if [[ $rc -ne 0 && -d "$WORK_ROOT" ]]; then
		failed="${OUTPUT_ROOT}.failed.$(date +%Y%m%d_%H%M%S).$$"
		mv "$WORK_ROOT" "$failed"
		echo "ACROBOT_KEYPOINT_FALLBACK_PREFLIGHT_FAILED_ARCHIVE=$failed"
	fi
	exit "$rc"
}
trap failed_archive EXIT

cd "$PROJECT_ROOT"
echo '[1/4] Static causal raw-RGB fallback and no-lag filter contracts'
"$PY_CORE" -B tdmpc2/check_acrobot_masked_keypoint_contract.py \
	>"$WORK_ROOT/logs/contracts.log" 2>&1

echo '[2/4] Paired training with raw RGB substituted only on Cutie-lost frames'
"$PY_CORE" -B -m tdmpc2.tools.train_acrobot_masked_keypoint_detector \
	--train-manifest "$SOURCE_KEYPOINT_ROOT/datasets/train/manifest.json" \
	--train-masks "$SOURCE_MASK_ROOT/masks/train/manifest.json" \
	--validation-manifest "$SOURCE_KEYPOINT_ROOT/datasets/validation/manifest.json" \
	--validation-masks "$SOURCE_MASK_ROOT/masks/validation/manifest.json" \
	--output "$WORK_ROOT/checkpoints/acrobot_masked_keypoint_fallback_v2.pt" \
	--device cuda:0 --epochs 30 --batch-size 128 \
	>"$WORK_ROOT/logs/train.log" 2>&1

echo '[3/4] Held-out fallback, no-lag causal filtering, burst, and latency gate'
"$PY_CORE" -B -m tdmpc2.tools.evaluate_acrobot_masked_keypoint_detector \
	--checkpoint "$WORK_ROOT/checkpoints/acrobot_masked_keypoint_fallback_v2.pt" \
	--test-manifest "$SOURCE_KEYPOINT_ROOT/datasets/test/manifest.json" \
	--test-masks "$SOURCE_MASK_ROOT/masks/test/manifest.json" \
	--output "$WORK_ROOT/acrobot_masked_keypoint_fallback_summary.json" \
	--device cuda:0 --filter-alpha 1.0 --maximum-fallback-rate 0.20 \
	>"$WORK_ROOT/logs/evaluate.log" 2>&1

echo '[4/4] Atomic publication; controller training is never launched here'
mv "$WORK_ROOT" "$OUTPUT_ROOT"
trap - EXIT
echo 'ACROBOT_KEYPOINT_FALLBACK_PREFLIGHT_COMPLETE'
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "SUMMARY=$OUTPUT_ROOT/acrobot_masked_keypoint_fallback_summary.json"
echo 'RUNNER_RC=0'
