"""Lightweight FlatAnchor configuration/shape contract checks.

Run with::

    python tdmpc2/check_flat_anchor_contract.py

This check never imports or downloads DINOv2. It only verifies that the
disabled-by-default configuration and the 25-D observation layout agree with
the frozen anchor teacher contract.
"""

from pathlib import Path
import unittest

import yaml


CONFIG_PATH = Path(__file__).with_name("config.yaml")
ROLE_COUNT = 4
RELATION_COUNT = 3
MOVING_ROLE_COUNT = 3
EXPECTED_INPUT_DIM = (
    ROLE_COUNT * 2  # role positions
    + RELATION_COUNT * 2  # base->elbow, elbow->tip, tip->goal
    + MOVING_ROLE_COUNT * 2  # elbow/tip/goal velocities
    + ROLE_COUNT  # per-role confidence
    + 1  # structured-decoder fallback flag
)


def load_config():
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def validate_flat_anchor_config(config, *, enabled=None, support_path=None):
    """Validate static values without touching the filesystem or DINOv2."""
    active = bool(config["flat_anchor"] if enabled is None else enabled)
    mode = config["flat_anchor_mode"]
    default_support = (
        config["cutie_object_support_path"]
        if mode in {"cutie_hybrid", "cutie_object_only"}
        else config["flat_anchor_support_path"]
    )
    support = default_support if support_path is None else support_path

    if config["flat_anchor_dim"] != EXPECTED_INPUT_DIM:
        raise ValueError(
            "flat_anchor_dim must match the 25-D teacher contract: "
            f"expected {EXPECTED_INPUT_DIM}, got {config['flat_anchor_dim']}"
        )
    if config["flat_anchor_mode"] not in {
        "residual", "oc", "object_graph", "hybrid_graph", "reward_graph",
        "cutie_hybrid", "cutie_object_only",
    }:
        raise ValueError(
            "flat_anchor_mode must be residual, oc, object_graph, hybrid_graph, "
            "reward_graph, cutie_hybrid, or cutie_object_only"
        )
    if config["flat_anchor_mode"] == "oc":
        joint_dim = (
            config["flat_anchor_scene_dim"]
            + config["flat_anchor_num_roles"] * config["flat_anchor_role_dim"]
        )
        if joint_dim != config["flat_anchor_joint_dim"]:
            raise ValueError("OC scene and role dimensions must sum to joint_dim")
        if config["flat_anchor_token_dim"] % config["flat_anchor_num_heads"]:
            raise ValueError("OC token dimension must be divisible by num_heads")
    if config["flat_anchor_mode"] in {
        "object_graph", "hybrid_graph", "reward_graph"
    }:
        if config["flat_anchor_num_roles"] != ROLE_COUNT:
            raise ValueError("Object graph requires exactly four fixed roles")
        object_dim = (
            config["flat_anchor_num_roles"]
            * config["flat_anchor_object_role_dim"]
        )
        if object_dim != 128:
            raise ValueError("Object graph control state must be 128-D")
        if config["flat_anchor_object_hidden_dim"] < config["flat_anchor_object_role_dim"]:
            raise ValueError("Object graph hidden_dim must be at least role_dim")
        if config["flat_anchor_event_max_interval"] < 1:
            raise ValueError("Event refresh max_interval must be positive")
        if config["flat_anchor_event_min_refresh_gap"] < 4:
            raise ValueError("Event min_refresh_gap must cap repeated DINO at 20%")
        for key in (
            "flat_anchor_event_confidence_threshold",
            "flat_anchor_event_color_confidence_threshold",
        ):
            if not 0 <= config[key] <= 1:
                raise ValueError(f"{key} must be in [0, 1]")
        if config["flat_anchor_event_color_search_radius"] < 1:
            raise ValueError("Event color search radius must be positive")
    if config["flat_anchor_mode"] == "hybrid_graph":
        graph_dim = (
            config["flat_anchor_num_roles"]
            * config["flat_anchor_object_role_dim"]
        )
        if (
            config["flat_anchor_scene_dim"] + graph_dim
            != config["flat_anchor_hybrid_joint_dim"]
        ):
            raise ValueError("HybridGraph joint_dim must equal scene_dim+graph_dim")
        if config["flat_anchor_hybrid_hidden_dim"] < graph_dim:
            raise ValueError("HybridGraph correction hidden_dim must cover graph_dim")
        if config["flat_anchor_graph_consistency_coef"] <= 0:
            raise ValueError("HybridGraph graph consistency coefficient must be positive")
    if config["flat_anchor_mode"] == "reward_graph":
        graph_dim = (
            config["flat_anchor_num_roles"]
            * config["flat_anchor_object_role_dim"]
        )
        if (
            config["flat_anchor_scene_dim"] + graph_dim
            != config["flat_anchor_hybrid_joint_dim"]
        ):
            raise ValueError("RewardGraph joint_dim must equal scene_dim+graph_dim")
        if config["flat_anchor_reward_graph_hidden_dim"] < 1:
            raise ValueError("RewardGraph reward hidden_dim must be positive")
        if config["flat_anchor_graph_consistency_coef"] <= 0:
            raise ValueError("RewardGraph graph consistency coefficient must be positive")
    if config["flat_anchor_mode"] in {"cutie_hybrid", "cutie_object_only"}:
        if config["cutie_object_num_roles"] != 2:
            raise ValueError("Cutie object modes require whole_arm and goal roles")
        if (
            config["cutie_object_frame_dim"] != 590
            or config["cutie_object_stack_frames"] != 3
            or config["cutie_object_input_dim"] != 1770
        ):
            raise ValueError("Cutie object modes require three stacked 590-D frames")
        object_dim = (
            config["cutie_object_num_roles"] * config["cutie_object_role_dim"]
        )
        if object_dim != 128 or config["cutie_object_only_latent_dim"] != object_dim:
            raise ValueError("Cutie object-only state must be 128-D")
        if config["flat_anchor_mode"] == "cutie_hybrid" and (
            config["flat_anchor_scene_dim"] + object_dim
            != config["cutie_object_joint_dim"]
        ):
            raise ValueError("CutieHybrid joint state must equal scene+objects")
    if active and not support:
        raise ValueError(
            "A verified support path must be supplied explicitly when flat_anchor=true"
        )
    if config["flat_anchor_allow_diagnostic_support"]:
        raise ValueError("Diagnostic/simulator-bootstrap support must remain disabled")
    if config["flat_anchor_dino_input_size"] % 14:
        raise ValueError("flat_anchor_dino_input_size must be divisible by 14")
    color_window = config["flat_anchor_color_window"]
    if color_window < 1 or color_window % 2 == 0:
        raise ValueError("flat_anchor_color_window must be a positive odd integer")
    if config["flat_anchor_structured_candidates"] < 2:
        raise ValueError("flat_anchor_structured_candidates must be at least 2")
    if config["flat_anchor_temporal_beam_size"] < 1:
        raise ValueError("flat_anchor_temporal_beam_size must be positive")


class FlatAnchorContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def test_official_path_is_unchanged_by_default(self):
        self.assertIs(self.config["flat_anchor"], False)
        self.assertEqual(self.config["flat_anchor_mode"], "object_graph")
        self.assertIs(self.config["compile_fallback_random"], False)
        self.assertIsNone(self.config["flat_anchor_support_path"])
        validate_flat_anchor_config(self.config)

    def test_anchor_observation_is_25_dimensional(self):
        self.assertEqual(EXPECTED_INPUT_DIM, 25)
        self.assertEqual(self.config["flat_anchor_dim"], EXPECTED_INPUT_DIM)

    def test_encoder_dimensions_are_explicit(self):
        self.assertEqual(self.config["flat_anchor_hidden_dim"], 128)
        self.assertEqual(self.config["flat_anchor_embed_dim"], 64)
        self.assertEqual(
            self.config["flat_anchor_fusion_dim"],
            self.config["latent_dim"],
        )
        self.assertEqual(self.config["flat_anchor_num_roles"], ROLE_COUNT)
        self.assertEqual(self.config["flat_anchor_role_input_dim"], 10)
        self.assertEqual(
            self.config["flat_anchor_scene_dim"]
            + ROLE_COUNT * self.config["flat_anchor_role_dim"],
            self.config["flat_anchor_joint_dim"],
        )
        self.assertEqual(
            ROLE_COUNT * self.config["flat_anchor_object_role_dim"],
            128,
        )
        self.assertEqual(self.config["flat_anchor_hybrid_joint_dim"], 640)
        self.assertEqual(self.config["flat_anchor_hybrid_hidden_dim"], 128)
        self.assertEqual(self.config["flat_anchor_reward_graph_hidden_dim"], 64)
        self.assertEqual(
            self.config["flat_anchor_graph_consistency_coef"],
            self.config["consistency_coef"],
        )

    def test_teacher_interface_is_explicit(self):
        self.assertTrue(self.config["flat_anchor_impl_path"].endswith("anchor_state.py"))
        self.assertEqual(self.config["flat_anchor_device"], "cuda:0")

    def test_enabling_requires_an_explicit_support_pack(self):
        with self.assertRaisesRegex(ValueError, "must be supplied explicitly"):
            validate_flat_anchor_config(self.config, enabled=True)
        validate_flat_anchor_config(
            self.config,
            enabled=True,
            support_path="manual_support.json",
        )

    def test_completed_ablation_modes_remain_explicitly_selectable(self):
        for mode in (
            "oc", "residual", "hybrid_graph", "reward_graph",
            "cutie_hybrid", "cutie_object_only",
        ):
            with self.subTest(mode=mode):
                validate_flat_anchor_config({**self.config, "flat_anchor_mode": mode})


if __name__ == "__main__":
    unittest.main(verbosity=2)
