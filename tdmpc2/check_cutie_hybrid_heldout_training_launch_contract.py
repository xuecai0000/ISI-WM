"""Dependency-light contract for held-out source-training launch binding."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
for local_path in (str(REPO_DIR), str(PROJECT_DIR)):
    while local_path in sys.path:
        sys.path.remove(local_path)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))

from tdmpc2.tools import evaluate_cutie_hybrid_heldout as evaluator


class HeldoutTrainingLaunchContract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.runner = (
            PROJECT_DIR / "tools" / "run_cutie_hybrid_250k_seed4_heldout_eval.sh"
        ).read_text(encoding="utf-8")

    def test_legacy_defaults_remain_frozen(self):
        args = evaluator.parse_args([
            "--runtime-config", "runtime_config.json",
            "--checkpoint", "final.pt",
            "--backend", "rgb",
            "--training-seed", "1",
            "--output", "output.json",
        ])
        self.assertEqual(args.expected_training_steps, 20_000)
        self.assertEqual(args.expected_training_eval_freq, 5_000)
        self.assertEqual(args.expected_training_eval_episodes, 3)

    def test_250k_seed4_overrides_are_explicit(self):
        args = evaluator.parse_args([
            "--runtime-config", "runtime_config.json",
            "--checkpoint", "final.pt",
            "--backend", "cutie_hybrid",
            "--training-seed", "4",
            "--output", "output.json",
            "--expected-training-steps", "250000",
            "--expected-training-eval-freq", "25000",
            "--expected-training-eval-episodes", "5",
        ])
        self.assertEqual(args.expected_training_steps, 250_000)
        self.assertEqual(args.expected_training_eval_freq, 25_000)
        self.assertEqual(args.expected_training_eval_episodes, 5)

    def test_exact_250k_launch_accepts_and_mismatch_fails_closed(self):
        payload = {
            "steps": 250_000,
            "eval_freq": 25_000,
            "eval_episodes": 5,
            "episode_length": 500,
        }
        frozen = evaluator._validate_frozen_training_launch(
            payload,
            expected_training_steps=250_000,
            expected_training_eval_freq=25_000,
            expected_training_eval_episodes=5,
        )
        self.assertEqual(frozen, payload)
        for key, wrong in (
            ("steps", 20_000),
            ("eval_freq", 5_000),
            ("eval_episodes", 3),
            ("episode_length", 499),
        ):
            bad = dict(payload)
            bad[key] = wrong
            with self.assertRaisesRegex(ValueError, "explicitly frozen"):
                evaluator._validate_frozen_training_launch(
                    bad,
                    expected_training_steps=250_000,
                    expected_training_eval_freq=25_000,
                    expected_training_eval_episodes=5,
                )

    def test_seed4_runner_freezes_identity_and_passes_explicit_source_protocol(self):
        for literal in (
            "EPISODES=20",
            "ENV_SEED=424242",
            "BACKGROUND_SEED=1618033",
            "PLANNER_SEED_BASE=8675309",
            "TRAINING_SEED=4",
            "TRAINING_STEPS=250000",
            "TRAINING_EVAL_FREQ=25000",
            "TRAINING_EVAL_EPISODES=5",
            '--expected-training-steps "$TRAINING_STEPS"',
            '--expected-training-eval-freq "$TRAINING_EVAL_FREQ"',
            '--expected-training-eval-episodes "$TRAINING_EVAL_EPISODES"',
            "run_solo rgb",
            "run_solo cutie_hybrid",
            '"split": "validation"',
            '"test_split_accessed": False',
            '"format": "cutie_hybrid_250k_seed4_heldout_summary_v1"',
        ):
            self.assertIn(literal, self.runner)
        self.assertLess(
            self.runner.index("run_solo rgb"),
            self.runner.index("run_solo cutie_hybrid"),
        )

    def test_runner_executes_this_contract_before_evaluation(self):
        invocation = (
            '"$PY" -m '
            "tdmpc2.check_cutie_hybrid_heldout_training_launch_contract"
        )
        self.assertIn(invocation, self.runner)
        self.assertLess(self.runner.index(invocation), self.runner.index("run_solo rgb"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
