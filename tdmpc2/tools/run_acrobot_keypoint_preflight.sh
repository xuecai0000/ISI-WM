#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PY_CORE="${PY_CORE:-<DATA_PATH>/conda_envs/tdmpc2_2026/bin/python}"
VIDEO_ROOT="${VIDEO_ROOT:?Set VIDEO_ROOT to the video_hard directory}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set a new OUTPUT_ROOT path}"
WORK_ROOT="${OUTPUT_ROOT}.incomplete"

if [[ ! -x "$PY_CORE" ]]; then
	echo "Missing Python interpreter: $PY_CORE" >&2
	exit 2
fi
if [[ ! -d "$VIDEO_ROOT" ]]; then
	echo "Missing video background root: $VIDEO_ROOT" >&2
	exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -e "$WORK_ROOT" ]]; then
	echo "Refusing to overwrite existing output: $OUTPUT_ROOT" >&2
	exit 3
fi
mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/datasets" "$WORK_ROOT/checkpoints"

failed_archive() {
	rc=$?
	if [[ $rc -ne 0 && -d "$WORK_ROOT" ]]; then
		failed="${OUTPUT_ROOT}.failed.$(date +%Y%m%d_%H%M%S).$$"
		mv "$WORK_ROOT" "$failed"
		echo "ACROBOT_KEYPOINT_PREFLIGHT_FAILED_ARCHIVE=$failed"
	fi
	exit "$rc"
}
trap failed_archive EXIT

cd "$PROJECT_ROOT"
echo '[1/5] Static contract and visual-pose interface tests'
"$PY_CORE" -B tdmpc2/check_visual_articulated_pose_contract.py \
	>"$WORK_ROOT/logs/contracts.log" 2>&1

manifest_args=()
if [[ -n "${VIDEO_MANIFEST_DIR:-}" ]]; then
	manifest_args=(--manifest-dir "$VIDEO_MANIFEST_DIR")
fi

echo '[2/5] Collecting disjoint train/validation/test simulator labels'
"$PY_CORE" -B -m tdmpc2.tools.collect_acrobot_keypoint_dataset \
	--output "$WORK_ROOT/datasets/train" --video-root "$VIDEO_ROOT" \
	"${manifest_args[@]}" --background-split train --episodes 40 \
	--seed 271828 --action-seed 314159 --background-seed 161803 \
	--pose-seed 141421 --initial-state-mode uniform \
	>"$WORK_ROOT/logs/collect_train.log" 2>&1
"$PY_CORE" -B -m tdmpc2.tools.collect_acrobot_keypoint_dataset \
	--output "$WORK_ROOT/datasets/validation" --video-root "$VIDEO_ROOT" \
	"${manifest_args[@]}" --background-split validation --episodes 10 \
	--seed 1271828 --action-seed 1314159 --background-seed 1161803 \
	--pose-seed 1141421 --initial-state-mode uniform \
	>"$WORK_ROOT/logs/collect_validation.log" 2>&1
"$PY_CORE" -B -m tdmpc2.tools.collect_acrobot_keypoint_dataset \
	--output "$WORK_ROOT/datasets/test" --video-root "$VIDEO_ROOT" \
	"${manifest_args[@]}" --background-split test --episodes 20 \
	--seed 2271828 --action-seed 2314159 --background-seed 2161803 \
	--pose-seed 2141421 --initial-state-mode uniform \
	>"$WORK_ROOT/logs/collect_test.log" 2>&1

echo '[3/5] Training the causal base/elbow/tip detector'
"$PY_CORE" -B -m tdmpc2.tools.train_acrobot_keypoint_detector \
	--train-manifest "$WORK_ROOT/datasets/train/manifest.json" \
	--validation-manifest "$WORK_ROOT/datasets/validation/manifest.json" \
	--output "$WORK_ROOT/checkpoints/acrobot_keypoint_detector_v1.pt" \
	--device cuda:0 --epochs 30 --batch-size 128 \
	>"$WORK_ROOT/logs/train.log" 2>&1

echo '[4/5] Held-out clean/hard long-sequence and batch-one latency gate'
"$PY_CORE" -B -m tdmpc2.tools.evaluate_acrobot_keypoint_detector \
	--checkpoint "$WORK_ROOT/checkpoints/acrobot_keypoint_detector_v1.pt" \
	--test-manifest "$WORK_ROOT/datasets/test/manifest.json" \
	--output "$WORK_ROOT/acrobot_keypoint_preflight_summary.json" \
	--device cuda:0 \
	>"$WORK_ROOT/logs/evaluate.log" 2>&1

echo '[5/5] Atomic publication; controller training is never launched here'
mv "$WORK_ROOT" "$OUTPUT_ROOT"
trap - EXIT
echo 'ACROBOT_KEYPOINT_PREFLIGHT_COMPLETE'
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "SUMMARY=$OUTPUT_ROOT/acrobot_keypoint_preflight_summary.json"
echo 'RUNNER_RC=0'
