"""Dependency-light contracts for the isolated SAM 3.1 backend."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

BACKEND_PATH = Path(__file__).resolve().parent / "perception" / "sam31_backend.py"
SPEC = importlib.util.spec_from_file_location("tdmpc2_sam31_contract_backend", BACKEND_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load SAM 3.1 backend: {BACKEND_PATH}")
BACKEND = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BACKEND
SPEC.loader.exec_module(BACKEND)

EXPECTED_SUPPORT_FRAMES = BACKEND.EXPECTED_SUPPORT_FRAMES
FrozenSupport = BACKEND.FrozenSupport
Sam31ContractError = BACKEND.Sam31ContractError
build_support_point_prompts = BACKEND.build_support_point_prompts
decode_official_frame_output = BACKEND.decode_official_frame_output
load_episode_rgb = BACKEND.load_episode_rgb
load_frozen_support = BACKEND.load_frozen_support
mask_to_relative_point_prompt = BACKEND.mask_to_relative_point_prompt


ROLES = ("role_a", "role_b")


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    rgb = np.zeros((EXPECTED_SUPPORT_FRAMES, 16, 16, 3), dtype=np.uint8)
    masks = np.zeros((EXPECTED_SUPPORT_FRAMES, 16, 16), dtype=np.uint8)
    for frame in range(EXPECTED_SUPPORT_FRAMES):
        rgb[frame, :, :, 0] = frame
        masks[frame, 2:5, 2 + frame % 2 : 5 + frame % 2] = 1
        masks[frame, 9:14, 10:12] = 2
    return rgb, masks


def _expect_error(function, needle: str) -> None:
    try:
        function()
    except Sam31ContractError as exc:
        if needle not in str(exc):
            raise AssertionError(f"expected {needle!r} in {exc!r}") from exc
    else:
        raise AssertionError("expected Sam31ContractError")


def main() -> None:
    source = BACKEND_PATH.read_text(encoding="utf-8")
    for marker in (
        "def _checkpoint_contract(self)",
        '"exact_coverage": True',
        "missing or unexpected or non_tensors or shape_mismatches",
        "official close_session retained inference state",
    ):
        assert marker in source, marker
    rgb, indexed = _fixture()
    with tempfile.TemporaryDirectory(prefix="sam31_contract_") as temporary:
        root = Path(temporary)
        support_path = root / "support.npz"
        episode_path = root / "episode.npz"
        np.savez_compressed(support_path, rgb=rgb, indexed_masks=indexed)
        # Include a sentinel GT member to prove the RGB-only loader neither needs
        # nor returns it.  Its deliberately incompatible dtype would fail any
        # accidental mask validation in the backend.
        np.savez_compressed(
            episode_path,
            rgb=rgb[:2],
            gt_indexed=np.asarray(["DO_NOT_READ"], dtype=np.str_),
        )
        support = load_frozen_support(support_path, role_names=ROLES)
        episode, _, _ = load_episode_rgb(episode_path)
        assert episode.shape == (2, 16, 16, 3)
        assert support.rgb.shape[0] == EXPECTED_SUPPORT_FRAMES

        prompts_a, digest_a = build_support_point_prompts(support)
        prompts_b, digest_b = build_support_point_prompts(support)
        assert digest_a == digest_b and len(prompts_a) == 12
        for left, right in zip(prompts_a, prompts_b):
            np.testing.assert_array_equal(left.points_xy, right.points_xy)
            np.testing.assert_array_equal(left.labels, right.labels)
            assert left.object_id in (1, 2)
            assert np.all((left.points_xy > 0) & (left.points_xy < 1))

        role_mask = indexed[0] == 1
        points, labels = mask_to_relative_point_prompt(role_mask)
        assert points.shape == (8, 2)
        np.testing.assert_array_equal(labels, [1, 1, 1, 1, 0, 0, 0, 0])

        official = {
            "out_obj_ids": np.asarray([2, 1]),
            "out_probs": np.asarray([0.7, 0.9], dtype=np.float32),
            "out_binary_masks": np.stack((indexed[0] == 2, indexed[0] == 1)),
        }
        masks, confidence, valid, overlap = decode_official_frame_output(
            official, role_count=2, height=16, width=16
        )
        np.testing.assert_array_equal(masks[0], indexed[0] == 1)
        np.testing.assert_array_equal(masks[1], indexed[0] == 2)
        np.testing.assert_allclose(confidence, [0.9, 0.7])
        np.testing.assert_array_equal(valid, [True, True])
        assert overlap == 0

        missing_role = indexed.copy()
        missing_role[0][missing_role[0] == 2] = 0
        bad_support = root / "bad_support.npz"
        np.savez_compressed(bad_support, rgb=rgb, indexed_masks=missing_role)
        _expect_error(
            lambda: load_frozen_support(bad_support, role_names=ROLES),
            "has no pixels",
        )

        # Constructing the dataclass itself remains dependency-light.
        frozen = FrozenSupport(
            rgb=support.rgb,
            indexed_masks=support.indexed_masks,
            role_names=support.role_names,
            source_path=support.source_path,
            rgb_sha256=support.rgb_sha256,
            masks_sha256=support.masks_sha256,
        )
        assert frozen.role_names == ROLES

    print("SAM31_BACKEND_CONTRACT_OK")


if __name__ == "__main__":
    main()
