"""Print a compact report for an object-state-supervision paired run."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ARMS = ("state_aux_off", "state_aux_on")
CONDITIONS = ("clean", "hard")


def load(path: Path):
	with path.open(encoding="utf-8") as stream:
		return json.load(stream)


def compact_summary(path: Path):
	payload = load(path)
	task_runs = payload["runs"]["acrobot-swingup"]
	result = {"path": str(path), "deltas": payload["deltas"]["acrobot-swingup"], "arms": {}}
	for arm in ARMS:
		training = task_runs[arm]["training"]
		curve = training["curve"]
		peak = max(curve, key=lambda row: (row["reward"], -row["step"]))
		result["arms"][arm] = {
			"auc": training["normalized_auc"],
			"final_reward": training["final_reward"],
			"peak_reward": training["peak_reward"],
			"peak_step": peak["step"],
			"final_eval": training["final_eval"],
			"held_out_final": {
				condition: {
					key: task_runs[arm]["held_out"][condition][key]
					for key in ("reward_mean", "reward_std", "reward_median", "reward_min", "reward_max")
				}
				for condition in CONDITIONS
			},
		}
	return result


def detailed_recovery(root: Path):
	result = {"root": str(root), "held_out": {}, "state_scores": {}}
	for arm in ARMS:
		result["held_out"][arm] = {}
		result["state_scores"][arm] = {}
		for selection in ("final", "best"):
			result["held_out"][arm][selection] = {}
			result["state_scores"][arm][selection] = {}
			for condition in CONDITIONS:
				suffix = "" if selection == "final" else "_best"
				evaluation = load(root / "evaluations" / f"acrobot-swingup_{arm}{suffix}_{condition}.json")
				rewards = [float(row["reward"]) for row in evaluation["episodes"]]
				result["held_out"][arm][selection][condition] = reward_stats(rewards)
				score_payload = load(
					root / "state_scores" / f"acrobot-swingup_{arm}_{selection}_{condition}.json"
				)
				score = score_payload["score"]
				result["state_scores"][arm][selection][condition] = {
					"reward_mean": score_payload["reward_mean"],
					"reward_std": score_payload["reward_std"],
					"rmse_mean": score["frame_rmse_mean"],
					"rmse_median": score["frame_rmse_median"],
					"rmse_p90": score["frame_rmse_p90"],
					"rmse_p95": score["frame_rmse_p95"],
					"rmse_max": score["frame_rmse_max"],
					"failure_bursts": score["failure_bursts"],
				}
	return result


def reward_stats(values):
	return {
		"mean": statistics.mean(values),
		"std": statistics.stdev(values) if len(values) > 1 else 0.0,
		"median": statistics.median(values),
		"min": min(values),
		"max": max(values),
	}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--recovery-root", type=Path, required=True)
	parser.add_argument("--summary-100k", type=Path)
	args = parser.parse_args()
	result = {
		"result_500k": compact_summary(
			args.recovery_root / "object_state_supervision_paired_summary.json"
		),
		"details_500k": detailed_recovery(args.recovery_root),
	}
	if args.summary_100k is not None:
		result["result_100k"] = compact_summary(args.summary_100k)
	print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
	main()
