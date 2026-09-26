"""Strict held-out evaluator for matched RGB and CutieHybrid checkpoints.

This entry point deliberately avoids Hydra so the live Cutie worker can start in
a fresh spawned interpreter. It reconstructs the training architecture from the
post-environment runtime config, evaluates on the validation background split,
and writes every episode rather than only an aggregate mean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
# This repository still uses both package imports (``tdmpc2.tools``) and
# legacy top-level imports (``common``, ``envs``, ``perception``). Canonicalize
# their order before multiprocessing ``spawn`` inherits sys.path: the package
# root must precede PROJECT_DIR so tdmpc2.py cannot shadow the tdmpc2 package,
# while both local roots must precede unrelated site-packages.
for local_path in (str(REPO_DIR), str(PROJECT_DIR)):
    while local_path in sys.path:
        sys.path.remove(local_path)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))


FORMAT = "cutie_hybrid_heldout_evaluation_v1"
TASK = "reacher-visual-small"
BACKENDS = ("rgb", "cutie_hybrid")
OBSERVATION_ABLATIONS = ("none", "rgb_zero", "object_zero")
VALIDATION_SPLIT = "validation"
SOURCE_PROTOCOL_KEYS = (
    "task",
    "obs",
    "episodic",
    "model_size",
    "steps",
    "eval_freq",
    "eval_episodes",
    "batch_size",
    "reward_coef",
    "value_coef",
    "termination_coef",
    "consistency_coef",
    "rho",
    "lr",
    "enc_lr_scale",
    "grad_clip_norm",
    "tau",
    "discount_denom",
    "discount_min",
    "discount_max",
    "buffer_size",
    "mpc",
    "iterations",
    "num_samples",
    "num_elites",
    "num_pi_trajs",
    "horizon",
    "min_std",
    "max_std",
    "temperature",
    "log_std_min",
    "log_std_max",
    "entropy_coef",
    "num_bins",
    "vmin",
    "vmax",
    "num_channels",
    "num_q",
    "dropout",
    "simnorm_dim",
    "compile",
    "compile_fallback_random",
    "video_background_enabled",
    "video_background_root",
    "video_background_manifest_dir",
    "video_background_split",
    "video_background_strength",
    "video_background_total_frames",
    "video_background_source_cache_size",
    "action_dim",
    "episode_length",
    "seed_steps",
)
CUTIE_PROTOCOL_KEYS = (
    "flat_anchor",
    "flat_anchor_mode",
    "flat_anchor_scene_dim",
    "flat_anchor_loss_weight_floor",
    "flat_anchor_loss_beta",
    "flat_anchor_reconstruction_coef",
    "flat_anchor_prediction_coef",
    "flat_anchor_graph_consistency_coef",
    "flat_anchor_hybrid_joint_dim",
    "flat_anchor_hybrid_hidden_dim",
    "cutie_object_repo",
    "cutie_object_checkpoint",
    "cutie_object_support_path",
    "cutie_object_config_dir",
    "cutie_object_device",
    "cutie_object_tracker_height",
    "cutie_object_tracker_width",
    "cutie_object_model_size",
    "cutie_object_prompt_radius",
    "cutie_object_amp",
    "cutie_object_worker_timeout_seconds",
    "cutie_object_num_roles",
    "cutie_object_frame_dim",
    "cutie_object_stack_frames",
    "cutie_object_input_dim",
    "cutie_object_role_dim",
    "cutie_object_hidden_dim",
    "cutie_object_joint_dim",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _protocol_subset(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"Runtime config is missing frozen protocol fields: {missing}")
    return {key: payload[key] for key in keys}


def _walk_env(env):
    visited = set()
    while env is not None and id(env) not in visited:
        visited.add(id(env))
        yield env
        env = getattr(env, "env", None)


def _declares(instance, name: str) -> bool:
    return any(name in cls.__dict__ for cls in type(instance).__mro__)


def _env_value(env, name: str, default=None):
    for candidate in _walk_env(env):
        if _declares(candidate, name):
            return getattr(candidate, name)
    return default


def _perception_metrics(env) -> dict[str, Any] | None:
    for candidate in _walk_env(env):
        if _declares(candidate, "metrics"):
            metrics = getattr(candidate, "metrics")
            if callable(metrics):
                value = metrics()
                return dict(value) if isinstance(value, dict) else None
    return None


def _rgb_tensor(obs):
    try:
        keys = set(obs.keys())
    except Exception:
        keys = set()
    rgb = obs["rgb"] if "rgb" in keys else obs
    return rgb.detach().cpu().contiguous()


def _rgb_sha256(obs) -> str:
    array = _rgb_tensor(obs).numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _object_sha256(obs) -> str | None:
    try:
        keys = set(obs.keys())
    except Exception:
        return None
    if "object" not in keys:
        return None
    array = obs["object"].detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _observation_sha256(obs) -> str:
    """Hash the complete pristine policy observation with key/shape/dtype framing."""
    digest = hashlib.sha256()
    try:
        keys = sorted(obs.keys())
    except Exception:
        keys = []
    items = [(str(key), obs[key]) for key in keys] if keys else [("<tensor>", obs)]
    for key, value in items:
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _apply_observation_ablation(obs, mode: str, torch, diagnostics=None):
    """Remove one information channel without changing the live environment.

    The held-out environment and Cutie worker still run normally in every
    CutieHybrid condition.  Only the copied observation passed to ``agent.act``
    is changed, so rewards, environment provenance, tracker runtime, and the
    checkpoint all remain directly comparable.
    """
    if diagnostics is not None:
        diagnostics["decision_steps"] += 1
    if mode == "none":
        return obs
    if mode not in OBSERVATION_ABLATIONS:
        raise ValueError(f"Unsupported observation ablation: {mode!r}.")
    try:
        keys = set(obs.keys())
    except Exception as exc:
        raise ValueError("Observation ablation requires a keyed CutieHybrid observation.") from exc
    if keys != {"rgb", "object"}:
        raise ValueError(
            "CutieHybrid observation ablation requires exact keys "
            f"['object', 'rgb'], got {sorted(keys)}."
        )
    field = "rgb" if mode == "rgb_zero" else "object"
    preserved_field = "object" if field == "rgb" else "rgb"
    ablated = obs.clone()
    original = ablated[field]
    ablated[field] = torch.zeros_like(original)
    nonzero_count = int(torch.count_nonzero(ablated[field]).item())
    max_abs = float(ablated[field].detach().abs().max().item())
    if nonzero_count != 0 or max_abs != 0.0:
        raise RuntimeError(f"Failed to zero observation field {field!r}.")
    if not torch.equal(ablated[preserved_field], obs[preserved_field]):
        raise RuntimeError(f"Observation ablation changed preserved field {preserved_field!r}.")
    if diagnostics is not None:
        diagnostics["zero_checks"] += 1
        diagnostics["preserved_field_checks"] += 1
        diagnostics["zeroed_nonzero_count_max"] = max(
            diagnostics["zeroed_nonzero_count_max"], nonzero_count
        )
        diagnostics["zeroed_max_abs"] = max(
            diagnostics["zeroed_max_abs"], max_abs
        )
    return ablated


def _torch_rng_sha256(torch) -> str:
    digest = hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes())
    if torch.cuda.is_available():
        digest.update(torch.cuda.get_rng_state(0).cpu().numpy().tobytes())
    return digest.hexdigest()


def _validate_frozen_training_launch(
    payload: dict[str, Any],
    *,
    expected_training_steps: int,
    expected_training_eval_freq: int,
    expected_training_eval_episodes: int,
) -> dict[str, int]:
    frozen_launch = {
        "steps": int(expected_training_steps),
        "eval_freq": int(expected_training_eval_freq),
        "eval_episodes": int(expected_training_eval_episodes),
        "episode_length": 500,
    }
    bad_launch = {
        key: (payload.get(key), expected)
        for key, expected in frozen_launch.items()
        if payload.get(key) != expected
    }
    if bad_launch:
        raise ValueError(
            "Runtime config does not match the explicitly frozen training launch: "
            f"{bad_launch}"
        )
    return frozen_launch


def _prepare_config(
    payload: dict[str, Any],
    *,
    backend: str,
    training_seed: int,
    checkpoint: Path,
    output: Path,
    episodes: int,
    env_seed: int,
    background_seed: int,
    expected_training_steps: int,
    expected_training_eval_freq: int,
    expected_training_eval_episodes: int,
):
    from common import MODEL_SIZE
    from common.parser import cfg_to_dataclass
    from omegaconf import OmegaConf

    data = dict(payload)
    if data.get("task") != TASK or data.get("obs") != "rgb":
        raise ValueError("Held-out evaluator only accepts reacher-visual-small RGB runs.")
    if data.get("multitask") is not False or data.get("model_size") != 5:
        raise ValueError("Held-out evaluator requires the frozen single-task model-size-5 run.")
    if data.get("video_background_enabled") is not True:
        raise ValueError("Training runtime config did not use dynamic video backgrounds.")
    if data.get("video_background_split") != "train":
        raise ValueError("Source runtime config was not produced by the train split.")
    if int(data.get("seed", -1)) != int(training_seed):
        raise ValueError(
            f"Runtime training seed {data.get('seed')!r} does not match "
            f"--training-seed={training_seed}."
        )
    _validate_frozen_training_launch(
        data,
        expected_training_steps=expected_training_steps,
        expected_training_eval_freq=expected_training_eval_freq,
        expected_training_eval_episodes=expected_training_eval_episodes,
    )

    if backend == "rgb":
        if data.get("flat_anchor") is not False or int(data.get("latent_dim", -1)) != 512:
            raise ValueError("RGB runtime config is not the frozen 512-D baseline.")
    else:
        if (
            data.get("flat_anchor") is not True
            or data.get("flat_anchor_mode") != "cutie_hybrid"
            or int(data.get("latent_dim", -1)) != 640
            or int(data.get("flat_anchor_scene_dim", -1)) != 512
        ):
            raise ValueError("Cutie runtime config is not the frozen 512+128-D hybrid.")
    frozen_model = MODEL_SIZE[5]
    for key, expected in frozen_model.items():
        recorded = data.get(key)
        if key == "latent_dim" and backend == "cutie_hybrid":
            recorded = data.get("flat_anchor_scene_dim")
        if recorded != expected:
            raise ValueError(
                f"Runtime model-size field {key}={recorded!r}, expected {expected!r}."
            )
    # Logger records Cutie's post-construction joint width. Restore every frozen
    # model-size field before reconstructing TDMPC2; it expands 512 -> 640 once.
    data.update(frozen_model)

    data.update(
        checkpoint=str(checkpoint),
        seed=int(env_seed),
        video_background_seed=int(background_seed),
        video_background_split=VALIDATION_SPLIT,
        eval_episodes=int(episodes),
        save_video=False,
        save_csv=False,
        save_agent=False,
        enable_wandb=False,
        compile=False,
        compile_fallback_random=False,
        exp_name=f"heldout_{backend}",
        work_dir=str(output.parent),
    )
    cfg = cfg_to_dataclass(OmegaConf.create(data))
    cfg.work_dir = output.parent
    return cfg


def _validate_args(args) -> None:
    if args.backend not in BACKENDS:
        raise ValueError(f"Unsupported backend: {args.backend}")
    if args.observation_ablation not in OBSERVATION_ABLATIONS:
        raise ValueError(
            f"Unsupported observation ablation: {args.observation_ablation!r}."
        )
    if args.backend != "cutie_hybrid" and args.observation_ablation != "none":
        raise ValueError("Observation ablation is supported only for CutieHybrid checkpoints.")
    if args.episodes < 1:
        raise ValueError("--episodes must be positive.")
    for value, name in (
        (args.expected_training_steps, "expected-training-steps"),
        (args.expected_training_eval_freq, "expected-training-eval-freq"),
        (args.expected_training_eval_episodes, "expected-training-eval-episodes"),
    ):
        if value < 1:
            raise ValueError(f"--{name} must be positive.")
    for value, name in (
        (args.env_seed, "env-seed"),
        (args.background_seed, "background-seed"),
        (args.planner_seed_base, "planner-seed-base"),
    ):
        if value < 0 or value >= 2**32:
            raise ValueError(f"--{name} must be a uint32 value.")
    if len({args.env_seed, args.background_seed, args.planner_seed_base}) != 3:
        raise ValueError("Environment, background, and planner seed domains must differ.")
    if args.planner_seed_base + args.episodes - 1 >= 2**32:
        raise ValueError("Planner episode seeds must remain within uint32.")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    for path in (args.runtime_config, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_checkpoint = args.runtime_config.resolve().parent / "models" / "final.pt"
    if args.runtime_config.name != "runtime_config.json":
        raise ValueError("--runtime-config must name runtime_config.json.")
    if args.checkpoint.resolve() != expected_checkpoint.resolve():
        raise ValueError(
            "Checkpoint/runtime binding mismatch: expected "
            f"{expected_checkpoint}, got {args.checkpoint.resolve()}."
        )


def evaluate(args) -> dict[str, Any]:
    import numpy as np
    import torch

    from common.seed import set_seed
    from envs import make_env
    from tdmpc2.tdmpc2 import TDMPC2

    _validate_args(args)
    runtime_payload = _json_load(args.runtime_config)
    source_training_protocol = _protocol_subset(runtime_payload, SOURCE_PROTOCOL_KEYS)
    cutie_training_protocol = None
    cutie_input_provenance = None
    if args.backend == "cutie_hybrid":
        cutie_training_protocol = _protocol_subset(runtime_payload, CUTIE_PROTOCOL_KEYS)
        cutie_checkpoint = Path(runtime_payload["cutie_object_checkpoint"]).resolve()
        cutie_support = Path(runtime_payload["cutie_object_support_path"]).resolve()
        for path, label in (
            (cutie_checkpoint, "Cutie checkpoint"),
            (cutie_support, "Cutie support annotations"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")
        cutie_input_provenance = {
            "checkpoint": str(cutie_checkpoint),
            "checkpoint_sha256": _sha256(cutie_checkpoint),
            "support_annotations": str(cutie_support),
            "support_annotations_sha256": _sha256(cutie_support),
        }
    cfg = _prepare_config(
        runtime_payload,
        backend=args.backend,
        training_seed=args.training_seed,
        checkpoint=args.checkpoint,
        output=args.output,
        episodes=args.episodes,
        env_seed=args.env_seed,
        background_seed=args.background_seed,
        expected_training_steps=args.expected_training_steps,
        expected_training_eval_freq=args.expected_training_eval_freq,
        expected_training_eval_episodes=args.expected_training_eval_episodes,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for held-out TD-MPC2 evaluation.")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    env = None
    episodes = []
    perception = None
    manifest_sha256 = None
    combined_manifest_sha256 = None
    cutie_ready = None
    ablation_diagnostics = {
        "mode": args.observation_ablation,
        "zeroed_field": {
            "none": None,
            "rgb_zero": "rgb",
            "object_zero": "object",
        }[args.observation_ablation],
        "preserved_field": {
            "none": None,
            "rgb_zero": "object",
            "object_zero": "rgb",
        }[args.observation_ablation],
        "decision_steps": 0,
        "zero_checks": 0,
        "preserved_field_checks": 0,
        "zeroed_nonzero_count_max": 0,
        "zeroed_max_abs": 0.0,
        "location": "copied_agent_input_before_encoding_every_decision_step",
        "live_environment_observation_mutated": False,
    }
    started = perf_counter()
    try:
        set_seed(args.env_seed)
        env = make_env(cfg)
        if _env_value(env, "active_split") != VALIDATION_SPLIT:
            raise RuntimeError("Environment did not construct the validation background split.")
        manifest_sha256 = _env_value(env, "manifest_sha256")
        combined_manifest_sha256 = _env_value(env, "combined_manifest_sha256")
        if not manifest_sha256 or not combined_manifest_sha256:
            raise RuntimeError("Background manifest provenance is unavailable.")
        agent = TDMPC2(cfg)
        agent.load(args.checkpoint)
        agent.eval()

        for episode_index in range(args.episodes):
            obs = env.reset()
            if _env_value(env, "active_split") != VALIDATION_SPLIT:
                raise RuntimeError("Background split changed during evaluation.")
            initial_rgb_sha256 = _rgb_sha256(obs)
            initial_object_sha256 = _object_sha256(obs)
            initial_observation_sha256 = _observation_sha256(obs)
            if args.backend == "cutie_hybrid" and initial_object_sha256 is None:
                raise RuntimeError("CutieHybrid observation did not contain object features.")
            if args.backend == "rgb" and initial_object_sha256 is not None:
                raise RuntimeError("RGB baseline unexpectedly contained object features.")
            source = _env_value(env, "active_source")
            source_frame_index = _env_value(env, "frame_index")
            if source is None or source_frame_index is None:
                raise RuntimeError("Background source/frame provenance is unavailable.")
            planner_seed = args.planner_seed_base + episode_index
            if planner_seed >= 2**32:
                raise ValueError("Planner episode seed exceeds uint32.")
            set_seed(planner_seed)
            agent._prev_mean.zero_()
            planner_rng_start_sha256 = _torch_rng_sha256(torch)

            reward_sum = 0.0
            info = {}
            episode_length = int(cfg.episode_length)
            for step_index in range(episode_length):
                torch.compiler.cudagraph_mark_step_begin()
                agent_obs = _apply_observation_ablation(
                    obs, args.observation_ablation, torch, ablation_diagnostics
                )
                action = agent.act(agent_obs, t0=step_index == 0, eval_mode=True)
                obs, reward, done, info = env.step(action)
                reward_sum += float(reward)
                if done:
                    step = step_index + 1
                    break
            else:
                raise RuntimeError(
                    f"Environment did not terminate after {episode_length} steps."
                )
            if step != episode_length:
                raise RuntimeError(f"Unexpected early episode termination at step {step}.")
            episode_record = {
                "episode_index": episode_index,
                "planner_seed": planner_seed,
                "planner_rng_start_sha256": planner_rng_start_sha256,
                "planner_rng_end_sha256": _torch_rng_sha256(torch),
                "initial_rgb_sha256": initial_rgb_sha256,
                "initial_object_sha256": initial_object_sha256,
                "initial_observation_sha256": initial_observation_sha256,
                "background_source": Path(source).name,
                "background_start_frame_index": int(source_frame_index),
                "reward": reward_sum,
                "success": float(info.get("success", 0.0)),
                "length": step,
            }
            episodes.append(episode_record)
            print(
                "CUTIE_HYBRID_HELDOUT_EPISODE",
                json.dumps(
                    {
                        "backend": args.backend,
                        "training_seed": args.training_seed,
                        "observation_ablation": args.observation_ablation,
                        **episode_record,
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                flush=True,
            )
        perception = _perception_metrics(env)
        cutie_ready = _env_value(env, "cutie_ready")
    finally:
        close = getattr(env, "close", None) if env is not None else None
        if callable(close):
            close()

    elapsed = perf_counter() - started
    rewards = np.asarray([episode["reward"] for episode in episodes], dtype=np.float64)
    if rewards.size != args.episodes or not np.isfinite(rewards).all():
        raise RuntimeError("Held-out rewards are incomplete or non-finite.")
    if args.backend == "cutie_hybrid":
        if not isinstance(perception, dict):
            raise RuntimeError("Cutie evaluation did not expose perception metrics.")
        expected_frames = args.episodes * (int(cfg.episode_length) + 1)
        if int(perception.get("frames", -1)) != expected_frames:
            raise RuntimeError(
                f"Cutie tracked {perception.get('frames')} frames, expected {expected_frames}."
            )
        if not isinstance(cutie_ready, dict):
            raise RuntimeError("Cutie worker ready/provenance report is unavailable.")
    elif perception is not None:
        raise RuntimeError("RGB baseline unexpectedly exposed perception metrics.")
    expected_decision_steps = args.episodes * int(cfg.episode_length)
    if ablation_diagnostics["decision_steps"] != expected_decision_steps:
        raise RuntimeError(
            "Observation ablation decision-step count mismatch: "
            f"{ablation_diagnostics['decision_steps']} != {expected_decision_steps}."
        )
    expected_checks = (
        0 if args.observation_ablation == "none" else expected_decision_steps
    )
    for key in ("zero_checks", "preserved_field_checks"):
        if ablation_diagnostics[key] != expected_checks:
            raise RuntimeError(
                f"Observation ablation {key} mismatch: "
                f"{ablation_diagnostics[key]} != {expected_checks}."
            )

    payload = {
        "format": FORMAT,
        "task": TASK,
        "backend": args.backend,
        "observation_ablation": args.observation_ablation,
        "training_seed": int(args.training_seed),
        "expected_source_training_launch": {
            "steps": int(args.expected_training_steps),
            "eval_freq": int(args.expected_training_eval_freq),
            "eval_episodes": int(args.expected_training_eval_episodes),
            "episode_length": 500,
        },
        "source_training_protocol": source_training_protocol,
        "cutie_training_protocol": cutie_training_protocol,
        "evaluation": {
            "split": VALIDATION_SPLIT,
            "episodes": int(args.episodes),
            "environment_seed": int(args.env_seed),
            "background_seed": int(args.background_seed),
            "planner_seed_base": int(args.planner_seed_base),
            "eval_mode": True,
            "compile": False,
            "reset_planner_rng_each_episode": True,
            "reset_previous_plan_each_episode": True,
        },
        "provenance": {
            "runtime_config": str(args.runtime_config.resolve()),
            "runtime_config_sha256": _sha256(args.runtime_config),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "evaluator_sha256": _sha256(Path(__file__).resolve()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_cuda_device": 0,
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "validation_manifest_sha256": str(manifest_sha256),
            "combined_manifest_sha256": str(combined_manifest_sha256),
            "cutie_inputs": cutie_input_provenance,
            "cutie_ready": cutie_ready,
        },
        "episodes": episodes,
        "summary": {
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std(ddof=1)) if rewards.size > 1 else 0.0,
            "reward_min": float(rewards.min()),
            "reward_max": float(rewards.max()),
            "elapsed_seconds": float(elapsed),
        },
        "perception_runtime": perception,
        "observation_ablation_diagnostics": ablation_diagnostics,
    }
    return payload


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError(f"Refusing stale temporary output: {temporary}")
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as file:
            file.write(text)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--env-seed", type=int, default=424242)
    parser.add_argument("--background-seed", type=int, default=1618033)
    parser.add_argument("--planner-seed-base", type=int, default=8675309)
    parser.add_argument("--expected-training-steps", type=int, default=20_000)
    parser.add_argument("--expected-training-eval-freq", type=int, default=5_000)
    parser.add_argument("--expected-training-eval-episodes", type=int, default=3)
    parser.add_argument(
        "--observation-ablation",
        choices=OBSERVATION_ABLATIONS,
        default="none",
        help=(
            "Inference-only information ablation for a CutieHybrid checkpoint. "
            "The live environment and Cutie worker remain enabled."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    payload = evaluate(args)
    _write_json_exclusive(args.output, payload)
    print(
        "CUTIE_HYBRID_HELDOUT_EVAL_OK",
        json.dumps(
            {
                "backend": args.backend,
                "training_seed": args.training_seed,
                "observation_ablation": args.observation_ablation,
                "reward_mean": payload["summary"]["reward_mean"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
