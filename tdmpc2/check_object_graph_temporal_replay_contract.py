"""Dependency-light static contracts for the temporal-v2 offline replay.

These checks deliberately inspect source and JSON without importing NumPy,
Torch, Cutie, dm-control, or either scoring implementation.  They guard the
runner lifecycle and the separation between the GT-free replay backend and the
privileged, post-replay aggregator.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
TOOLS = ROOT / "tools"
COMMON = ROOT / "common"
V1_GRAPH_ROOT = ROOT / "object_graphs"
V2_GRAPH_ROOT = ROOT / "object_graphs_temporal_v2"
RUNNER = TOOLS / "run_object_graph_temporal_replay_preflight.sh"
BACKEND = TOOLS / "replay_object_graph_temporal_v2.py"
AGGREGATOR = TOOLS / "aggregate_object_graph_temporal_replay.py"
SNAPSHOT = COMMON / "object_graph_temporal_replay_snapshot.py"
V1_ACROBOT_SHA256 = (
    "a4a01168fa0ab9ad831b6c4b5741747a407cc78ec7ba60fd49b767be98597857"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cli_options(path: Path) -> set[str]:
    tree = ast.parse(_text(path), filename=str(path))
    options: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        value = node.args[0]
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            options.add(value.value)
    return options


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(_text(path), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


class TemporalReplayStaticContracts(unittest.TestCase):
    def test_required_files_and_graph_namespace_are_additive(self) -> None:
        for path in (RUNNER, BACKEND, AGGREGATOR, SNAPSHOT):
            self.assertTrue(path.is_file(), path)
            self.assertFalse(path.is_symlink(), path)
        self.assertEqual(
            {path.name for path in V1_GRAPH_ROOT.glob("*.json")},
            {
                "acrobot_swingup.json",
                "cartpole_swingup.json",
                "reacher_visual_small.json",
            },
        )
        self.assertEqual(
            {path.name for path in V2_GRAPH_ROOT.glob("*.json")},
            {"acrobot_swingup_temporal_v2.json"},
        )
        v1_path = V1_GRAPH_ROOT / "acrobot_swingup.json"
        v2_path = V2_GRAPH_ROOT / "acrobot_swingup_temporal_v2.json"
        self.assertEqual(_sha256(v1_path), V1_ACROBOT_SHA256)
        v1 = json.loads(_text(v1_path))
        v2 = json.loads(_text(v2_path))
        self.assertEqual(v1["task"], "acrobot-swingup")
        self.assertEqual(v2["task"], v1["task"])
        self.assertEqual(v2["source_roles"], v1["source_roles"])
        self.assertEqual(v2["semantic_roles"], v1["semantic_roles"])
        self.assertEqual(
            [entity["name"] for entity in v2["tracking_entities"]],
            [entity["name"] for entity in v1["tracking_entities"]],
        )
        self.assertEqual(
            [entity["projector"]["type"] for entity in v2["tracking_entities"]],
            ["ordered_chain_temporal_v2"],
        )

    def test_cli_schemas_are_frozen_and_explicit(self) -> None:
        self.assertEqual(
            _cli_options(BACKEND),
            {
                "--inputs",
                "--v1-backend-manifest",
                "--v1-graph",
                "--graph",
                "--output-root",
                "--strict-counts",
            },
        )
        self.assertEqual(
            _cli_options(AGGREGATOR),
            {
                "--source-benchmark-root",
                "--v1-preflight-root",
                "--v2-replay-backend",
                "--v1-graph",
                "--graph",
                "--isolation-gate",
                "--immutable-inputs",
                "--output",
            },
        )
        self.assertEqual(
            _cli_options(SNAPSHOT),
            {
                "--source-benchmark-root",
                "--v1-preflight-root",
                "--output",
                "--verify",
            },
        )

    def test_backend_is_cpu_mask_replay_not_training_or_episode_gt(self) -> None:
        source = _text(BACKEND)
        lowered = source.lower()
        self.assertTrue({"numpy"}.issubset(_import_roots(BACKEND)))
        self.assertTrue(
            {"torch", "dm_control", "gym", "gymnasium", "subprocess"}.isdisjoint(
                _import_roots(BACKEND)
            )
        )
        for forbidden in (
            "tdmpc2/train.py",
            "nvidia-smi",
            "dataset/scoring",
            "env.physics",
            "simulator.step",
        ):
            self.assertNotIn(forbidden, lowered)
        for required in (
            '"episode_rgb_decoded": False',
            '"episode_ground_truth_read": False',
            '"simulator_state_read": False',
            '"actions_read": False',
            '"rewards_read": False',
            '"episode_entity_arrays_loaded_before_replay": True',
            '"future_entity_frames_parser_input": False',
            '"appearance_features_replayed": False',
            '"descriptors_emitted": False',
            '"controller_training_eligible": False',
            '"cpu_only": True',
            '"support_rgb_schema_validated_not_used": True',
            "project_temporal_geometry(",
            'os.environ.get("CUDA_VISIBLE_DEVICES") != ""',
            'write_json(incomplete_root / "backend_predictions.json"',
            "os.replace(incomplete_root, output_root)",
        ):
            self.assertIn(required, source)

    def test_snapshot_rebuilds_full_inputs_implementation_and_environment(self) -> None:
        source = _text(SNAPSHOT)
        self.assertTrue(
            {"torch", "dm_control", "gym", "gymnasium", "subprocess"}.isdisjoint(
                _import_roots(SNAPSHOT)
            )
        )
        for required in (
            'FORMAT = "object_graph_temporal_replay_inputs_v1"',
            "_reject_symlink_components",
            "_source_artifacts(source_root)",
            "_validate_backend_manifest(",
            "validate_backend_inputs(",
            'dataset["episodes"]',
            "baseline_paths.items()",
            "episode_paths.items()",
            'f"v1 prediction {task}/{condition}/{episode}"',
            '"local_source": _local_tree(package_root)',
            '"environment": _environment()',
            '"cpu_only": True',
            'v1_graph_path = package_root / "object_graphs" / "acrobot_swingup.json"',
            '"object_graphs_temporal_v2"',
            "existing != payload",
            "path.read_bytes() != _canonical(existing)",
        ):
            self.assertIn(required, source)

    def test_runner_installs_traps_before_claiming_stage(self) -> None:
        runner = _text(RUNNER)
        owned_zero = runner.index("STAGE_OWNED=0")
        trap_exit = runner.index("trap archive_on_exit EXIT")
        stage_claim = runner.index('if ! mkdir -- "$STAGE"')
        owned_one = runner.index("STAGE_OWNED=1", stage_claim)
        children = runner.index('mkdir -- "$STAGE/backends"', owned_one)
        self.assertLess(owned_zero, trap_exit)
        self.assertLess(trap_exit, stage_claim)
        self.assertLess(stage_claim, owned_one)
        self.assertLess(owned_one, children)
        self.assertIn("trap 'exit 130' INT", runner[trap_exit:stage_claim])
        self.assertIn("trap 'exit 143' TERM", runner[trap_exit:stage_claim])

    def test_failure_cleanup_kills_workers_before_gt_restore_and_downgrades(self) -> None:
        runner = _text(RUNNER)
        cleanup = runner[runner.index("archive_on_exit() {"):runner.index("trap archive_on_exit EXIT")]
        term = cleanup.index('kill -TERM -- "-$pid"')
        wait = cleanup.index('wait "$pid"')
        restore = cleanup.index('chmod "$SCORING_ROOT_MODE_BEFORE"')
        release = cleanup.index('rmdir -- "$SOURCE_LOCK"')
        archive = cleanup.index('mv -T -- "$STAGE" "$failed"')
        self.assertLess(term, wait)
        self.assertLess(wait, restore)
        self.assertLess(restore, release)
        self.assertLess(release, archive)
        for required in (
            '"status":"runner_engineering_fail"',
            '"engineering_pass":False',
            '"development_candidate":False',
            '"controller_training_authorized":False',
            '"scientific_go":False',
            '"runner_exit_code":rc',
            '"recommendation":"fix_engineering_failure_do_not_train_controller"',
        ):
            self.assertIn(required, cleanup)

    def test_runner_locks_only_scoring_root_and_holds_source_lock_through_promotion(self) -> None:
        runner = _text(RUNNER)
        self.assertEqual(runner.count('chmod 000 -- "$SCORING_ROOT"'), 1)
        self.assertNotIn("chmod -R", runner)
        lock_acquire = runner.index('if ! mkdir -- "$SOURCE_LOCK"')
        chmod = runner.index('chmod 000 -- "$SCORING_ROOT"')
        replay = runner.index("REPLAY_START backend=object_graph_temporal_v2")
        restore = runner.index('chmod "$SCORING_ROOT_MODE_BEFORE"', replay)
        aggregate = runner.index(
            "tdmpc2.tools.aggregate_object_graph_temporal_replay", restore
        )
        final_stage = runner.index('echo "[5/5]')
        promote = runner.index('mv -T -- "$STAGE" "$BASE"', final_stage)
        release = runner.index('rmdir -- "$SOURCE_LOCK"', promote)
        self.assertLess(lock_acquire, chmod)
        self.assertLess(chmod, replay)
        self.assertLess(replay, restore)
        self.assertLess(restore, aggregate)
        self.assertLess(aggregate, promote)
        self.assertLess(promote, release)
        self.assertIn("SOURCE_LOCK_OWNED=0", runner[release:])

    def test_runner_is_cpu_only_controller_free_and_tracks_replay_and_aggregate(self) -> None:
        runner = _text(RUNNER)
        lowered = runner.lower()
        self.assertNotIn("tdmpc2/train.py", lowered)
        self.assertNotIn("nvidia-smi", lowered)
        self.assertNotIn("cuda:0", lowered)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', runner)
        self.assertIn('export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/tdmpc2"', runner)
        self.assertNotIn('${PYTHONPATH:+:$PYTHONPATH}', runner)
        self.assertEqual(
            runner.count("run_tracked env CUDA_VISIBLE_DEVICES=\"\""), 2
        )
        self.assertIn("setsid timeout --foreground", runner)
        self.assertIn("tdmpc2.tools.replay_object_graph_temporal_v2", runner)
        self.assertIn("tdmpc2.tools.aggregate_object_graph_temporal_replay", runner)
        self.assertIn("--isolation-gate \"$ISOLATION_GATE\"", runner)
        self.assertIn("--immutable-inputs \"$INPUTS\"", runner)
        self.assertIn("--v1-graph \"$V1_GRAPH\"", runner)
        self.assertIn("--graph \"$GRAPH\"", runner)

    def test_isolation_gate_is_exactly_bound_by_aggregator(self) -> None:
        runner = _text(RUNNER)
        aggregate = _text(AGGREGATOR)
        for required in (
            '"format":"object_graph_temporal_replay_scoring_isolation_v1"',
            '"root_mode_locked":"000"',
            '"backend_completed_before_restore":True',
            '"cpu_only":True',
            '"cuda_visible_devices":""',
            '"v2_backend_manifest_relative_to_summary_root"',
            '"same_uid_read_probe":{"exit_code":1,"error_type":"PermissionError"',
        ):
            self.assertIn(required, runner)
        for required in (
            'ISOLATION_FORMAT = "object_graph_temporal_replay_scoring_isolation_v1"',
            "_validate_isolation_gate(",
            "args.isolation_gate",
            'payload["root_mode_locked"] != "000"',
            'payload.get("backend_completed_before_restore") is not True',
            'payload.get("cpu_only") is not True',
            'payload.get("cuda_visible_devices") != ""',
            'payload.get("v2_backend_manifest_sha256") != file_sha256(v2_manifest_path)',
            'read_probe.get("error_type") != "PermissionError"',
            '"scoring_isolation_relative_to_summary_root"',
            '"scoring_isolation_sha256"',
            "_validate_immutable_inputs(",
            "build_immutable_inputs(",
            'relative != "provenance/immutable_inputs.json"',
            "payload != rebuilt",
            "path.read_bytes() != canonical_json_bytes(payload)",
            '"immutable_inputs_relative_to_summary_root"',
            '"immutable_inputs_sha256"',
        ):
            self.assertIn(required, aggregate)


if __name__ == "__main__":
    program = unittest.main(verbosity=2, exit=False)
    if not program.result.wasSuccessful():
        raise SystemExit(1)
    print("OBJECT_GRAPH_TEMPORAL_REPLAY_CONTRACT_OK", flush=True)
