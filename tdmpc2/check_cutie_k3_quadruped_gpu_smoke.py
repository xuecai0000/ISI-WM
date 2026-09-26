"""Real GPU smoke test for the three-role legacy Cutie controller.

This check deliberately performs no training.  It reconstructs one generated
K=3 Quadruped runtime, starts the real visual environment and Cutie worker,
executes one planned 12-D action, and proves that a freshly saved checkpoint
strictly reloads into the same architecture.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path

# Select headless MuJoCo rendering before importing the environment package.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from omegaconf import OmegaConf

from common.parser import cfg_to_dataclass
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2


ENCODER = "legacy_mlp_k3_v1"
QUADRUPED_TASKS = {"quadruped-run", "quadruped-walk"}
ROLES = ["torso", "front_legs", "back_legs"]


def _args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("--runtime-config", type=Path, required=True)
	return parser.parse_args()


def _load_config(path: Path):
	path = path.resolve()
	if not path.is_file():
		raise FileNotFoundError(path)
	raw = json.loads(path.read_text(encoding="utf-8"))
	if raw.get("task") not in QUADRUPED_TASKS:
		raise ValueError(f"Expected a Quadruped task, got {raw.get('task')!r}.")
	expected = {
		"obs": "rgb",
		"flat_anchor": True,
		"flat_anchor_mode": "cutie_object_only",
		"cutie_object_regression_encoder": ENCODER,
		"cutie_object_num_roles": 3,
		"cutie_object_role_names": ROLES,
		"cutie_object_only_latent_dim": 192,
		"latent_dim": 192,
		"cutie_object_spatial_token_enabled": False,
		"cutie_object_variable_graph_enabled": False,
		"cutie_object_allow_simulator_runtime": False,
		"cutie_object_allow_simulator_kinematics_runtime": False,
		"object_state_supervision_enabled": False,
	}
	bad = {key: (raw.get(key), value) for key, value in expected.items()
		   if raw.get(key) != value}
	if bad:
		raise ValueError(f"Unsafe K=3 runtime config: {bad}")
	if int(raw.get("action_dim", -1)) != 12:
		raise ValueError("Quadruped template must declare a 12-D action.")

	# Preserve the production visual path while disabling compilation and output.
	raw.update(
		compile=False,
		compile_fallback_random=False,
		# Training enters through parse_cfg(), which makes single-task task_dim
		# zero.  The serialized template can still carry MODEL_SIZE's multitask
		# default (96), so reproduce the post-parse single-task runtime here.
		multitask=False,
		task_dim=0,
		tasks=[raw["task"]],
		save_agent=False,
		save_csv=False,
		save_video=False,
		enable_wandb=False,
		checkpoint=None,
		work_dir=str(path.parent / "gpu_smoke_no_output"),
	)
	cfg = cfg_to_dataclass(OmegaConf.create(raw))
	cfg.work_dir = path.parent / "gpu_smoke_no_output"
	return cfg


def main() -> None:
	args = _args()
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required for this smoke test.")
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.set_float32_matmul_precision("high")

	cfg = _load_config(args.runtime_config)
	env = None
	agent = None
	reloaded = None
	try:
		set_seed(int(cfg.seed))
		env = make_env(cfg)
		if int(cfg.action_dim) != 12 or tuple(env.action_space.shape) != (12,):
			raise AssertionError(
				f"Real environment action contract is {cfg.action_dim}, "
				f"{env.action_space.shape}; expected 12."
			)
		if tuple(cfg.obs_shape.get("object", ())) != (3, 1770):
			raise AssertionError(f"Unexpected object observation shape: {cfg.obs_shape}")

		obs = env.reset()
		# TensorWrapper returns a TensorDict rather than a builtin dict.
		if set(obs.keys()) != {"object"}:
			raise AssertionError("K=3 visual environment did not return object observations.")
		if tuple(obs["object"].shape) != (3, 1770):
			raise AssertionError(
				f"Real K=3 observation has shape {tuple(obs['object'].shape)}."
			)

		rebuild_cfg = deepcopy(cfg)
		agent = TDMPC2(cfg).eval()
		if int(agent.cfg.latent_dim) != 192:
			raise AssertionError(f"Agent latent width is {agent.cfg.latent_dim}, not 192.")
		action = agent.act(obs, t0=True, eval_mode=True)
		if (
			tuple(action.shape) != (12,)
			or not torch.isfinite(action).all()
			or bool((action.abs() > 1).any())
		):
			raise AssertionError(f"Invalid planned action: shape={tuple(action.shape)}")
		_, reward, done, _ = env.step(action)
		if not np.isfinite(float(reward)) or bool(done):
			raise AssertionError(f"Invalid first transition: reward={reward}, done={done}")

		checkpoint = BytesIO()
		agent.save(checkpoint)
		checkpoint.seek(0)
		payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
		contract = payload.get("checkpoint_contract", {})
		if contract.get("cutie_object_regression_encoder") != ENCODER:
			raise AssertionError("Saved checkpoint lost the K=3 encoder contract.")
		if int(contract.get("latent_dim", -1)) != 192:
			raise AssertionError("Saved checkpoint lost the 192-D latent contract.")
		reloaded = TDMPC2(rebuild_cfg).eval()
		reloaded.load(payload)

		print(
			"CUTIE_K3_QUADRUPED_GPU_SMOKE_OK "
			+ json.dumps(
				{
					"task": cfg.task,
					"roles": ROLES,
					"observation_shape": list(obs["object"].shape),
					"latent_dim": int(agent.cfg.latent_dim),
					"action_dim": int(action.numel()),
					"reward_finite": True,
					"checkpoint_strict_reload": True,
					"runtime_simulator_state": False,
				},
				sort_keys=True,
			)
		)
	finally:
		close = getattr(env, "close", None) if env is not None else None
		if callable(close):
			close()
		del reloaded, agent
		torch.cuda.empty_cache()


if __name__ == "__main__":
	main()
