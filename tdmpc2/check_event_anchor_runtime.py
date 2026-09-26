"""Short dense-DINO versus event-anchor runtime/accuracy preflight.

The comparison uses only RGB frames from one fixed random reacher-easy
trajectory.  Simulator positions, rewards, and other privileged state are never
read.  The support set should use a different seed from ``--seed``.
"""

import argparse
import json
import pathlib
import time

import numpy as np
import torch
import yaml
from dm_control import suite

from envs.wrappers.event_anchor_state import EventTriggeredAnchorStateTeacher
from envs.wrappers.flat_anchor import (
    _ConfigView,
    _event_config,
    _load_teacher_class,
    _teacher_config,
)


ROLES = ("base", "elbow", "control_tip", "goal")
CONFIG_PATH = pathlib.Path(__file__).with_name("config.yaml")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-path", required=True)
    parser.add_argument(
        "--impl-path",
        default="<BASELINE_PATH>/r2dreamer-main/anchor_state.py",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--max-refresh-fraction", type=float, default=0.25)
    parser.add_argument("--min-speedup", type=float, default=2.5)
    parser.add_argument("--max-elbow-mean-px", type=float, default=3.0)
    parser.add_argument("--max-elbow-p95-px", type=float, default=6.0)
    parser.add_argument("--max-tip-mean-px", type=float, default=2.5)
    parser.add_argument("--max-tip-p95-px", type=float, default=5.0)
    parser.add_argument("--max-goal-mean-px", type=float, default=2.5)
    parser.add_argument("--max-goal-p95-px", type=float, default=5.0)
    parser.add_argument(
        "--open-loop-diagnostic",
        action="store_true",
        help=(
            "Accept finite cheap proposals between scheduled DINO refreshes. "
            "This diagnoses tracker accuracy separately from confidence calibration; "
            "it is never used by the training wrapper."
        ),
    )
    return parser.parse_args()


def make_config(args):
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    values.update(
        flat_anchor_support_path=args.support_path,
        flat_anchor_impl_path=args.impl_path,
        flat_anchor_device=args.device,
    )
    if args.open_loop_diagnostic:
        # The production wrapper remains fail-closed. This diagnostic disables
        # only its three calibrated rejection thresholds so we can determine
        # whether proposals are accurate but under-confident, or actually wrong.
        values.update(
            flat_anchor_event_confidence_threshold=0.0,
            flat_anchor_event_color_confidence_threshold=-1.0,
            flat_anchor_event_structure_tolerance=1e9,
        )
    return _ConfigView(values)


def collect_rgb_trajectory(steps, seed, height, width):
    if steps < 64:
        raise ValueError("--steps must be at least 64 for a meaningful event-runtime gate")
    env = suite.load(
        "reacher", "easy", task_kwargs={"random": seed}, visualize_reward=False
    )
    rng = np.random.RandomState(seed)
    spec = env.action_spec()
    env.reset()
    frames = [env.physics.render(height=height, width=width, camera_id=0)]
    for _ in range(steps - 1):
        action = rng.uniform(spec.minimum, spec.maximum).astype(spec.dtype)
        # Match TD-MPC2's DMControlWrapper action repeat.
        for _ in range(2):
            env.step(action)
        frames.append(env.physics.render(height=height, width=width, camera_id=0))
    return np.ascontiguousarray(frames, dtype=np.uint8)


def run_locator(locator, frames):
    state = locator.initial_state(
        batch=1, beam_size=locator.beam_size, device=torch.device("cpu")
    )
    outputs = []
    started = time.perf_counter()
    for index, frame in enumerate(frames):
        image = torch.from_numpy(frame).unsqueeze(0)
        first = torch.tensor([index == 0], dtype=torch.bool)
        anchor, state = locator.extract(
            image, state, first, output_device=torch.device("cpu")
        )
        outputs.append(anchor.reshape(25).numpy().copy())
    elapsed = time.perf_counter() - started
    return np.stack(outputs), elapsed


def role_position_error_px(dense, event, height, width):
    dense_points = dense[:, :8].reshape(len(dense), len(ROLES), 2)
    event_points = event[:, :8].reshape(len(event), len(ROLES), 2)
    pixel_scale = np.asarray([(width - 1) / 2.0, (height - 1) / 2.0])
    error = np.linalg.norm((dense_points - event_points) * pixel_scale, axis=-1)
    return {
        role: {
            "mean_px": float(error[:, index].mean()),
            "p95_px": float(np.percentile(error[:, index], 95)),
            "max_px": float(error[:, index].max()),
        }
        for index, role in enumerate(ROLES)
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required for the configured DINO teacher")
    cfg = make_config(args)
    frames = collect_rgb_trajectory(
        args.steps, args.seed, args.height, args.width
    )
    teacher_cls = _load_teacher_class(cfg.flat_anchor_impl_path)
    dense_locator = teacher_cls(_teacher_config(cfg), device=torch.device(args.device))

    # Exclude one-time CUDA kernel/library warm-up from both paths. This call
    # does not touch the teacher's temporal state or its frame counters.
    dense_locator._component_scores(frames[:1])
    dense, dense_elapsed = run_locator(dense_locator, frames)
    event_locator = EventTriggeredAnchorStateTeacher(
        dense_locator, _event_config(cfg)
    )
    event, event_elapsed = run_locator(event_locator, frames)
    event_metrics = event_locator.metrics()
    report = {
        "frames": int(len(frames)),
        "trajectory_seed": int(args.seed),
        "open_loop_diagnostic": bool(args.open_loop_diagnostic),
        "shape": {
            "dense": list(dense.shape),
            "event": list(event.shape),
        },
        "finite": {
            "dense": bool(np.isfinite(dense).all()),
            "event": bool(np.isfinite(event).all()),
        },
        "role_position_error": role_position_error_px(
            dense, event, args.height, args.width
        ),
        "dense_ms_per_frame": float(1000.0 * dense_elapsed / len(frames)),
        "event_ms_per_frame": float(1000.0 * event_elapsed / len(frames)),
        "speedup": float(dense_elapsed / max(event_elapsed, 1e-12)),
        "refresh_fraction": float(event_metrics["refresh_fraction"]),
        "dino_frames": int(event_metrics["dino_frames"]),
        "fast_frames": int(event_metrics["fast_frames"]),
        "guarded_frames": int(event_metrics["guarded_frames"]),
        "fast_attempt_frames": int(event_metrics["fast_attempt_frames"]),
        "zncc_scores": {
            key.removeprefix("zncc_"): float(value)
            for key, value in event_metrics.items()
            if key.startswith("zncc_")
        },
        "trigger_counts": {
            key.removeprefix("trigger_"): int(value)
            for key, value in event_metrics.items()
            if key.startswith("trigger_")
        },
        "acceptance_thresholds": {
            "max_refresh_fraction": float(args.max_refresh_fraction),
            "min_speedup": float(args.min_speedup),
            "max_elbow_mean_px": float(args.max_elbow_mean_px),
            "max_elbow_p95_px": float(args.max_elbow_p95_px),
            "max_tip_mean_px": float(args.max_tip_mean_px),
            "max_tip_p95_px": float(args.max_tip_p95_px),
            "max_goal_mean_px": float(args.max_goal_mean_px),
            "max_goal_p95_px": float(args.max_goal_p95_px),
        },
    }
    if dense.shape != (args.steps, 25) or event.shape != dense.shape:
        raise RuntimeError(f"Unexpected anchor shapes: {dense.shape}, {event.shape}")
    if not report["finite"]["dense"] or not report["finite"]["event"]:
        raise RuntimeError("Non-finite anchor output detected")
    failures = []
    accounted_frames = (
        report["dino_frames"] + report["fast_frames"] + report["guarded_frames"]
    )
    if accounted_frames != report["frames"]:
        failures.append(
            f"frame accounting {accounted_frames} != {report['frames']}"
        )
    if report["refresh_fraction"] > args.max_refresh_fraction:
        failures.append(
            f"refresh_fraction={report['refresh_fraction']:.3f} > "
            f"{args.max_refresh_fraction:.3f}"
        )
    if report["speedup"] < args.min_speedup:
        failures.append(
            f"speedup={report['speedup']:.3f} < {args.min_speedup:.3f}"
        )
    for role, mean_limit, p95_limit in (
        ("elbow", args.max_elbow_mean_px, args.max_elbow_p95_px),
        ("control_tip", args.max_tip_mean_px, args.max_tip_p95_px),
        ("goal", args.max_goal_mean_px, args.max_goal_p95_px),
    ):
        error = report["role_position_error"][role]
        if error["mean_px"] > mean_limit:
            failures.append(
                f"{role}.mean_px={error['mean_px']:.3f} > {mean_limit:.3f}"
            )
        if error["p95_px"] > p95_limit:
            failures.append(
                f"{role}.p95_px={error['p95_px']:.3f} > {p95_limit:.3f}"
            )
    if failures:
        print("EVENT_ANCHOR_RUNTIME_REJECTED", json.dumps(report, indent=2, sort_keys=True))
        raise RuntimeError("; ".join(failures))
    print("EVENT_ANCHOR_RUNTIME_OK", json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
