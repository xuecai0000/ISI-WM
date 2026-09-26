#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PY_CORE="${PY_CORE:-<DATA_PATH>/conda_envs/tdmpc2_2026/bin/python}"
SOURCE_KEYPOINT_ROOT="${SOURCE_KEYPOINT_ROOT:?Set SOURCE_KEYPOINT_ROOT to the published direct-RGB keypoint preflight}"
SOURCE_BENCHMARK_ROOT="${SOURCE_BENCHMARK_ROOT:?Set SOURCE_BENCHMARK_ROOT to the published unified VOS benchmark}"
OC_STORM_REPO="${OC_STORM_REPO:-<DATA_PATH>/r2_hrssm_third_party/OC-STORM}"
CUTIE_CHECKPOINT="${CUTIE_CHECKPOINT:-$OC_STORM_REPO/feature_extractor/cutie/weights/cutie-small-mega.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set a new OUTPUT_ROOT path}"
WORK_ROOT="${OUTPUT_ROOT}.incomplete"
SUPPORT_NPZ="$SOURCE_BENCHMARK_ROOT/worker_inputs/support/acrobot-swingup/support.npz"
GRAPH="$PROJECT_ROOT/tdmpc2/object_graphs/acrobot_swingup.json"

for path in "$PY_CORE" "$CUTIE_CHECKPOINT" "$SUPPORT_NPZ" "$GRAPH"; do
	if [[ ! -e "$path" ]]; then
		echo "Missing required input: $path" >&2
		exit 2
	fi
done
for split in train validation test; do
	if [[ ! -f "$SOURCE_KEYPOINT_ROOT/datasets/$split/manifest.json" ]]; then
		echo "Missing source keypoint split: $split" >&2
		exit 2
	fi
done
if [[ -e "$OUTPUT_ROOT" || -e "$WORK_ROOT" ]]; then
	echo "Refusing to overwrite existing output: $OUTPUT_ROOT" >&2
	exit 3
fi
mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/masks" "$WORK_ROOT/checkpoints"

failed_archive() {
	rc=$?
	if [[ $rc -ne 0 && -d "$WORK_ROOT" ]]; then
		failed="${OUTPUT_ROOT}.failed.$(date +%Y%m%d_%H%M%S).$$"
		mv "$WORK_ROOT" "$failed"
		echo "ACROBOT_MASKED_KEYPOINT_PREFLIGHT_FAILED_ARCHIVE=$failed"
	fi
	exit "$rc"
}
trap failed_archive EXIT

cd "$PROJECT_ROOT"
echo '[1/6] Static V2 mask, calibration, and causal-filter contracts'
"$PY_CORE" -B tdmpc2/check_acrobot_masked_keypoint_contract.py \
	>"$WORK_ROOT/logs/contracts.log" 2>&1

echo '[2/6] Whole-Acrobot Cutie masks on immutable paired RGB splits'
for split in train validation test; do
	echo "MASK_REPLAY_START split=$split"
	"$PY_CORE" -B -m tdmpc2.tools.generate_acrobot_whole_cutie_masks \
		--input-manifest "$SOURCE_KEYPOINT_ROOT/datasets/$split/manifest.json" \
		--support-npz "$SUPPORT_NPZ" --graph "$GRAPH" \
		--oc-storm-repo "$OC_STORM_REPO" --checkpoint "$CUTIE_CHECKPOINT" \
		--output "$WORK_ROOT/masks/$split" --device cuda:0 \
		>"$WORK_ROOT/logs/masks_$split.log" 2>&1
	echo "MASK_REPLAY_END split=$split rc=0"
done

echo '[3/6] Fixed-camera calibration and paired clean/hard keypoint training'
"$PY_CORE" -B -m tdmpc2.tools.train_acrobot_masked_keypoint_detector \
	--train-manifest "$SOURCE_KEYPOINT_ROOT/datasets/train/manifest.json" \
	--train-masks "$WORK_ROOT/masks/train/manifest.json" \
	--validation-manifest "$SOURCE_KEYPOINT_ROOT/datasets/validation/manifest.json" \
	--validation-masks "$WORK_ROOT/masks/validation/manifest.json" \
	--output "$WORK_ROOT/checkpoints/acrobot_masked_keypoint_detector_v2.pt" \
	--device cuda:0 --epochs 30 --batch-size 128 \
	>"$WORK_ROOT/logs/train.log" 2>&1

echo '[4/6] Held-out causal long-sequence, burst, velocity, and total-latency gate'
"$PY_CORE" -B -m tdmpc2.tools.evaluate_acrobot_masked_keypoint_detector \
	--checkpoint "$WORK_ROOT/checkpoints/acrobot_masked_keypoint_detector_v2.pt" \
	--test-manifest "$SOURCE_KEYPOINT_ROOT/datasets/test/manifest.json" \
	--test-masks "$WORK_ROOT/masks/test/manifest.json" \
	--output "$WORK_ROOT/acrobot_masked_keypoint_preflight_summary.json" \
	--device cuda:0 \
	>"$WORK_ROOT/logs/evaluate.log" 2>&1

echo '[5/6] Record decision; controller training remains unauthorized unless every gate passes'
"$PY_CORE" - "$WORK_ROOT/acrobot_masked_keypoint_preflight_summary.json" <<'PY' \
	>"$WORK_ROOT/logs/decision.log" 2>&1
import json, sys
p = json.load(open(sys.argv[1], encoding='utf-8'))
print('status:', p['status'])
print('engineering_pass:', p['engineering_pass'])
print('controller_pilot_authorized:', p['controller_pilot_authorized'])
print('recommendation:', p['recommendation'])
PY

echo '[6/6] Atomic publication; this preflight never launches controller training'
mv "$WORK_ROOT" "$OUTPUT_ROOT"
trap - EXIT
echo 'ACROBOT_MASKED_KEYPOINT_PREFLIGHT_COMPLETE'
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "SUMMARY=$OUTPUT_ROOT/acrobot_masked_keypoint_preflight_summary.json"
echo 'RUNNER_RC=0'
