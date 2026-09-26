"""Dependency-light contract for CutieHybrid held-out input ablations."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
for local_path in (str(REPO_DIR), str(PROJECT_DIR)):
    while local_path in sys.path:
        sys.path.remove(local_path)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))

from tdmpc2.tools import evaluate_cutie_hybrid_heldout as evaluator


class FakeObservation(dict):
    def clone(self):
        return FakeObservation({key: value.clone() for key, value in self.items()})


def diagnostics(mode):
    return {
        "mode": mode,
        "zeroed_field": None,
        "preserved_field": None,
        "decision_steps": 0,
        "zero_checks": 0,
        "preserved_field_checks": 0,
        "zeroed_nonzero_count_max": 0,
        "zeroed_max_abs": 0.0,
        "location": "copied_agent_input_before_encoding_every_decision_step",
        "live_environment_observation_mutated": False,
    }


class CutieHybridInputAblationContract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.runner_path = (
            PROJECT_DIR / "tools" / "run_cutie_hybrid_250k_seed4_input_ablation.sh"
        )
        cls.runner = cls.runner_path.read_text(encoding="utf-8")
        cls.obs = FakeObservation({
            "rgb": torch.arange(18, dtype=torch.uint8).reshape(2, 3, 3),
            "object": torch.linspace(-1.0, 1.0, 12).reshape(2, 6),
        })

    def test_default_mode_preserves_legacy_evaluation(self):
        args = evaluator.parse_args([
            "--runtime-config", "runtime_config.json",
            "--checkpoint", "final.pt",
            "--backend", "cutie_hybrid",
            "--training-seed", "4",
            "--output", "output.json",
        ])
        self.assertEqual(args.observation_ablation, "none")
        diag = diagnostics("none")
        result = evaluator._apply_observation_ablation(self.obs, "none", torch, diag)
        self.assertIs(result, self.obs)
        self.assertEqual(diag["decision_steps"], 1)
        self.assertEqual(diag["zero_checks"], 0)

    def test_rgb_zero_uses_copy_and_preserves_object(self):
        original_rgb = self.obs["rgb"].clone()
        original_object = self.obs["object"].clone()
        diag = diagnostics("rgb_zero")
        result = evaluator._apply_observation_ablation(
            self.obs, "rgb_zero", torch, diag
        )
        self.assertIsNot(result, self.obs)
        self.assertEqual(int(torch.count_nonzero(result["rgb"])), 0)
        self.assertTrue(torch.equal(result["object"], original_object))
        self.assertTrue(torch.equal(self.obs["rgb"], original_rgb))
        self.assertTrue(torch.equal(self.obs["object"], original_object))
        self.assertEqual(diag["decision_steps"], 1)
        self.assertEqual(diag["zero_checks"], 1)
        self.assertEqual(diag["preserved_field_checks"], 1)

    def test_object_zero_uses_copy_and_preserves_rgb(self):
        original_rgb = self.obs["rgb"].clone()
        original_object = self.obs["object"].clone()
        diag = diagnostics("object_zero")
        result = evaluator._apply_observation_ablation(
            self.obs, "object_zero", torch, diag
        )
        self.assertEqual(int(torch.count_nonzero(result["object"])), 0)
        self.assertTrue(torch.equal(result["rgb"], original_rgb))
        self.assertTrue(torch.equal(self.obs["object"], original_object))
        self.assertEqual(diag["zeroed_nonzero_count_max"], 0)
        self.assertEqual(diag["zeroed_max_abs"], 0.0)

    def test_schema_and_unknown_modes_fail_closed(self):
        bad = FakeObservation({"rgb": self.obs["rgb"].clone()})
        with self.assertRaisesRegex(ValueError, "exact keys"):
            evaluator._apply_observation_ablation(bad, "rgb_zero", torch)
        with self.assertRaisesRegex(ValueError, "Unsupported observation ablation"):
            evaluator._apply_observation_ablation(self.obs, "latent_zero", torch)

    def test_runner_freezes_three_exact_sequential_conditions(self):
        for literal in (
            "EPISODES=20",
            "ENV_SEED=424242",
            "BACKGROUND_SEED=1618033",
            "PLANNER_SEED_BASE=8675309",
            "TRAINING_SEED=4",
            "TRAINING_STEPS=250000",
            "TRAINING_EVAL_FREQ=25000",
            "TRAINING_EVAL_EPISODES=5",
            "MODES=(none rgb_zero object_zero)",
            '--observation-ablation "$mode"',
            'for mode in "${MODES[@]}"',
            '"format": "cutie_hybrid_250k_seed4_input_ablation_summary_v1"',
            '"test_split_accessed": False',
            '"live_cutie_runs_in_all_conditions": True',
            '"episode_reset_isolation_probe_pass": True',
            '"eligible_for_structural_object_only_speed_claim": False',
        ):
            self.assertIn(literal, self.runner)
        self.assertIn(
            '"$PY" -m tdmpc2.check_cutie_hybrid_input_ablation_contract',
            self.runner,
        )
        self.assertIn(
            "tdmpc2.tools.check_cutie_episode_reset_isolation",
            self.runner,
        )
        self.assertLess(
            self.runner.index("check_cutie_episode_reset_isolation"),
            self.runner.index("run_mode \"$mode\""),
        )
        self.assertLess(
            self.runner.index("run_mode \"$mode\""),
            self.runner.index("ablation_summary.json"),
        )

    def test_runner_is_lf_only_and_python_heredocs_compile(self):
        raw = self.runner_path.read_bytes()
        self.assertNotIn(b"\r", raw)
        lines = self.runner.splitlines()
        blocks = []
        index = 0
        while index < len(lines):
            if lines[index].rstrip().endswith("<<'PY'"):
                end = index + 1
                while end < len(lines) and lines[end] != "PY":
                    end += 1
                self.assertLess(end, len(lines), f"unterminated heredoc at {index + 1}")
                source = "\n".join(lines[index + 1:end]) + "\n"
                compile(source, f"runner-heredoc-{index + 2}", "exec")
                blocks.append((index + 2, end))
                index = end
            index += 1
        self.assertEqual(len(blocks), 2)

    def test_synthetic_three_condition_aggregation(self):
        lines = self.runner.splitlines()
        blocks = []
        index = 0
        while index < len(lines):
            if lines[index].rstrip().endswith("<<'PY'"):
                end = index + 1
                while lines[end] != "PY":
                    end += 1
                blocks.append("\n".join(lines[index + 1:end]) + "\n")
                index = end
            index += 1
        aggregate_source = blocks[1]
        sha = "a" * 64
        ready = {
            "status": "ready",
            "start_method": "spawn",
            "global_hydra_initialized_before_cutie": False,
            "roles": ["whole_arm", "goal"],
            "frame_feature_dim": 590,
            "stacked_feature_dim": 1770,
            "permanent_prompts": 6,
            "episode_reset_strategy": "fresh_inference_core_support_replay_v1",
            "device": "cuda:0",
            "tracker_size": [448, 448],
            "model_size": "small",
            "prompt_radius": 2.0,
            "amp": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "current_device": 0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run_root = root / "source_run"
            (run_root / "models").mkdir(parents=True)
            runtime_config = run_root / "runtime_config.json"
            checkpoint = run_root / "models" / "final.pt"
            runtime_config.write_bytes(b"synthetic-runtime-config")
            checkpoint.write_bytes(b"synthetic-checkpoint")
            runtime_sha = hashlib.sha256(runtime_config.read_bytes()).hexdigest()
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            training_summary = root / "source_summary.json"
            source_gates = {
                "frames": True,
                "valid_frame_rate": True,
                "max_invalid_burst": True,
                "worker_restarts": True,
                "timeouts": True,
                "ms_per_frame": True,
                "runtime_unit": True,
                "wallclock_seconds": True,
            }
            training_summary.write_text(json.dumps({
                "status": "source_runtime_pass",
                "total_steps": 250000,
                "eval_freq": 25000,
                "eval_episodes": 5,
                "expected_cutie_frames": 278055,
                "gates": source_gates,
                "elapsed_seconds": 100.0,
                "cutie_runtime": {
                    "frames": 278055,
                    "valid_frame_rate": 1.0,
                    "max_invalid_burst": 0,
                    "worker_restarts": 0,
                    "timeouts": 0,
                    "ms_per_frame": 7.0,
                    "runtime_unit": "milliseconds_per_tracked_frame_excluding_support_prompts",
                },
                "runs": {
                    "cutie": {
                        "root": str(run_root),
                        "eval_rewards": [float(index) for index in range(11)],
                    },
                },
            }), encoding="utf-8")
            reset_asset = root / "reset_isolation.npz"
            reset_asset.write_bytes(b"synthetic-reset-arrays")
            reset_comparison_names = (
                "fresh_a_vs_fresh_b",
                "fresh_a_vs_post_forward",
                "fresh_b_vs_post_reverse",
                "post_forward_vs_post_reverse",
                "fresh_sequence_a_vs_fresh_sequence_b",
                "fresh_sequence_a_vs_post_sequence_forward",
                "fresh_sequence_b_vs_post_sequence_reverse",
                "post_sequence_forward_vs_post_sequence_reverse",
            )
            (root / "reset_isolation.json").write_text(json.dumps({
                "format": "cutie_episode_reset_isolation_v1",
                "status": "episode_reset_isolation_pass",
                "pass": True,
                "reset_strategy": "fresh_inference_core_support_replay_v1",
                "runtime_config": str(runtime_config.resolve()),
                "runtime_config_sha256": runtime_sha,
                "pollution_length": 500,
                "all_features_finite_and_roles_valid": True,
                "pollution_histories_diverged": True,
                "comparisons": {
                    name: {"byte_equal": True}
                    for name in reset_comparison_names
                },
                "workers": {
                    name: {"tracked_frames": 514, "ready": ready}
                    for name in ("forward", "reverse")
                },
                "arrays_asset": {
                    "path": reset_asset.name,
                    "sha256": hashlib.sha256(reset_asset.read_bytes()).hexdigest(),
                },
            }), encoding="utf-8")
            common_rows = [{
                "episode_index": index,
                "planner_seed": 8675309 + index,
                "planner_rng_start_sha256": sha,
                "planner_rng_end_sha256": sha,
                "initial_rgb_sha256": sha,
                "initial_object_sha256": sha,
                "initial_observation_sha256": sha,
                "background_source": f"video{index % 10 + 70}.mp4",
                "background_start_frame_index": index,
                "success": 0.0,
                "length": 500,
            } for index in range(20)]
            for mode, reward in (("none", 30.0), ("rgb_zero", 20.0), ("object_zero", 10.0)):
                rows = [dict(row, reward=reward + row["episode_index"]) for row in common_rows]
                zero_checks = 0 if mode == "none" else 10_000
                payload = {
                    "format": "cutie_hybrid_heldout_evaluation_v1",
                    "task": "reacher-visual-small",
                    "backend": "cutie_hybrid",
                    "observation_ablation": mode,
                    "training_seed": 4,
                    "expected_source_training_launch": {
                        "steps": 250000, "eval_freq": 25000,
                        "eval_episodes": 5, "episode_length": 500,
                    },
                    "source_training_protocol": {"same": True},
                    "cutie_training_protocol": {"same": True},
                    "evaluation": {
                        "split": "validation", "episodes": 20,
                        "environment_seed": 424242,
                        "background_seed": 1618033,
                        "planner_seed_base": 8675309,
                        "eval_mode": True, "compile": False,
                        "reset_planner_rng_each_episode": True,
                        "reset_previous_plan_each_episode": True,
                    },
                    "provenance": {
                        "runtime_config": str(run_root / "runtime_config.json"),
                        "runtime_config_sha256": runtime_sha,
                        "checkpoint": str(run_root / "models" / "final.pt"),
                        "checkpoint_sha256": checkpoint_sha,
                        "evaluator_sha256": sha,
                        "cuda_visible_devices": "0",
                        "logical_cuda_device": 0,
                        "device_name": "Synthetic GPU",
                        "device_capability": [8, 9],
                        "validation_manifest_sha256": sha,
                        "combined_manifest_sha256": sha,
                        "cutie_inputs": {"checkpoint_sha256": sha},
                        "cutie_ready": ready,
                    },
                    "episodes": rows,
                    "summary": {"elapsed_seconds": 1.0},
                    "perception_runtime": {
                        "frames": 10020, "valid_frame_rate": 1.0,
                        "max_invalid_burst": 0, "worker_restarts": 0,
                        "timeouts": 0, "ms_per_frame": 7.0,
                        "runtime_unit": "milliseconds_per_tracked_frame_excluding_support_prompts",
                    },
                    "observation_ablation_diagnostics": {
                        "mode": mode,
                        "zeroed_field": {"none": None, "rgb_zero": "rgb", "object_zero": "object"}[mode],
                        "preserved_field": {"none": None, "rgb_zero": "object", "object_zero": "rgb"}[mode],
                        "decision_steps": 10000,
                        "zero_checks": zero_checks,
                        "preserved_field_checks": zero_checks,
                        "zeroed_nonzero_count_max": 0,
                        "zeroed_max_abs": 0.0,
                        "location": "copied_agent_input_before_encoding_every_decision_step",
                        "live_environment_observation_mutated": False,
                    },
                }
                (root / f"{mode}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            original_argv = sys.argv
            sys.argv = [
                "aggregate", str(root), "20", "424242", "1618033",
                "8675309", "0", str(training_summary), str(run_root),
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(aggregate_source, "synthetic-aggregate", "exec"), {})
            finally:
                sys.argv = original_argv
            summary = json.loads((root / "ablation_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "input_ablation_all_runtime_pass")
            self.assertTrue(summary["all_runtime_gates_pass"])
            contrasts = summary["pairwise_paired_episode_statistics"]
            self.assertEqual(contrasts["full_minus_rgb_input_zero"]["mean"], 10.0)
            self.assertEqual(contrasts["full_minus_object_input_zero"]["mean"], 20.0)
            self.assertEqual(contrasts["rgb_input_zero_minus_object_input_zero"]["mean"], 10.0)

            failed_source = json.loads(training_summary.read_text(encoding="utf-8"))
            failed_source["status"] = "source_runtime_fail"
            failed_source["gates"]["max_invalid_burst"] = False
            failed_source["cutie_runtime"]["max_invalid_burst"] = 32
            training_summary.write_text(json.dumps(failed_source), encoding="utf-8")
            (root / "ablation_summary.json").unlink()
            original_argv = sys.argv
            sys.argv = [
                "aggregate", str(root), "20", "424242", "1618033",
                "8675309", "0", str(training_summary), str(run_root),
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(aggregate_source, "synthetic-source-fail", "exec"), {})
            finally:
                sys.argv = original_argv
            failed_summary = json.loads(
                (root / "ablation_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                failed_summary["status"],
                "input_ablation_complete_source_training_runtime_fail",
            )
            self.assertFalse(failed_summary["source_training_runtime_pass"])
            self.assertTrue(failed_summary["input_ablation_runtime_pass"])
            self.assertFalse(failed_summary["all_runtime_gates_pass"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
