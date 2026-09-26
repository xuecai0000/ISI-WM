#!/usr/bin/env bash
# Relay an already-running four-task regression into the Finger ablation and broad screen.

set -Eeuo pipefail
PY=<DATA_PATH>/conda_envs/tdmpc2_2026/bin/python
PHASE1_ROOT=<DATA_PATH>/tdmpc2_runs/universal_variable_object_graph_v1_20260910_codex1
PHASE1_BASE="$PHASE1_ROOT/logs/_diagnostic/variable_object_graph_regression_30k_v1_20260910_codex1_recovery_v1"
PHASE1_LOG="${PHASE1_BASE}.runner.log"
PHASE1_SUMMARY="$PHASE1_BASE/variable_object_graph_regression_summary.json"

PHASE2_ROOT=<DATA_PATH>/tdmpc2_runs/variable_graph_finger_ablation_v1_20260910_codex1
PHASE2_BASE="$PHASE2_ROOT/logs/_diagnostic/variable_graph_finger_ablation_100k_v1_codex1_fix1"
PHASE2_LOG="${PHASE2_BASE}.runner.log"
PHASE2_SUMMARY="$PHASE2_BASE/variable_graph_finger_ablation_summary.json"

PHASE3_ROOT=<DATA_PATH>/tdmpc2_runs/variable_object_graph_broad_screen_v1_20260911_codex1
PHASE3_BASE="$PHASE3_ROOT/logs/_diagnostic/variable_object_graph_broad_screen_30k_v1_codex1"
PHASE3_LOG="${PHASE3_BASE}.runner.log"
PHASE3_SUMMARY="$PHASE3_BASE/variable_object_graph_broad_screen_summary.json"
SUPPORT_LOG="$PHASE3_ROOT/logs/_diagnostic/extended_support_collect_v4.runner.log"

wait_for_phase1() {
	while [[ ! -f "$PHASE1_SUMMARY" ]]; do
		if grep -q '^RUNNER_RC=[1-9]' "$PHASE1_LOG" 2>/dev/null; then
			echo 'CAMPAIGN_STOP phase=1 reason=upstream_failed'
			return 1
		fi
		echo "CAMPAIGN_WAIT phase=1 time=$(date --iso-8601=seconds)"
		sleep 30
	done
	echo 'CAMPAIGN_PHASE_COMPLETE phase=1'
}

run_phase2() {
	if [[ -f "$PHASE2_SUMMARY" ]]; then
		echo 'CAMPAIGN_REUSE phase=2'
		return 0
	fi
	if [[ -e "$PHASE2_BASE" || -e "${PHASE2_BASE}.incomplete" || -e "$PHASE2_LOG" ]]; then
		echo 'CAMPAIGN_STOP phase=2 reason=ambiguous_existing_output'
		return 3
	fi
	cd "$PHASE2_ROOT"
	echo 'CAMPAIGN_START phase=2 experiment=finger_ablation_100k'
	env OUTPUT_ROOT="$PHASE2_BASE" \
		bash tdmpc2/tools/run_variable_graph_finger_ablation_100k.sh \
		>"$PHASE2_LOG" 2>&1
	echo 'CAMPAIGN_PHASE_COMPLETE phase=2'
}

run_phase3() {
	if [[ -f "$PHASE3_SUMMARY" ]]; then
		echo 'CAMPAIGN_REUSE phase=3'
		return 0
	fi
	grep -q '^SUPPORT_ALL_COMPLETE count=20$' "$SUPPORT_LOG" || {
		echo 'CAMPAIGN_STOP phase=3 reason=support_incomplete'
		return 2
	}
	if [[ -e "$PHASE3_BASE" || -e "${PHASE3_BASE}.incomplete" || -e "$PHASE3_LOG" ]]; then
		echo 'CAMPAIGN_STOP phase=3 reason=ambiguous_existing_output'
		return 3
	fi
	cd "$PHASE3_ROOT"
	echo 'CAMPAIGN_START phase=3 experiment=broad_screen readout=direct'
	env RUN_TAG_OVERRIDE=variable_object_graph_broad_screen_30k_v1_codex1 \
		OUTPUT_ROOT="$PHASE3_BASE" \
		bash tdmpc2/tools/run_variable_object_graph_broad_screen.sh \
		>"$PHASE3_LOG" 2>&1
	echo 'CAMPAIGN_PHASE_COMPLETE phase=3'
}

echo "CAMPAIGN_STARTED time=$(date --iso-8601=seconds)"
wait_for_phase1
run_phase2
echo 'CAMPAIGN_SELECTION readout=direct reason=no_role_compression'
run_phase3
echo "CAMPAIGN_COMPLETE summary=$PHASE3_SUMMARY"
