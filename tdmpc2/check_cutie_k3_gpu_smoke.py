"""Real-GPU smoke test for a three-role legacy Cutie controller.

This opens the task's actual RGB-only environment and Cutie tracker, checks the
``[3, 1770]`` observation contract, executes one planned action, and verifies
that the resulting K=3/192 checkpoint reloads strictly.  It never trains and
does not enable simulator state, runtime masks, or runtime kinematics.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from omegaconf import OmegaConf

from common.parser import cfg_to_dataclass
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2


TASKS = {
	"cheetah-run": (("torso", "back_leg", "front_leg"), 6),
	"hopper-hop": (("torso", "leg", "foot"), 4),
	"hopper-stand": (("torso", "leg", "foot"), 4),
	"walker-run": (("torso", "right_leg", "left_leg"), 6),
	"walker-stand": (("torso", "right_leg", "left_leg"), 6),
	"walker-walk": (("torso", "right_leg", "left_leg"), 6),
}
ENCODER = "legacy_mlp_k3_v1"


def _load_config(path: Path):
	raw = json.loads(path.resolve().read_text(encoding="utf-8"))
	task = raw.get("task")
	if task not in TASKS:
		raise ValueError(f"Unsupported K=3 smoke task: {task!r}")
	roles, action_dim = TASKS[task]
	expected = {
		"obs": "rgb",
		"flat_anchor": True,
		"flat_anchor_mode": "cutie_object_only",
		"cutie_object_regression_encoder": ENCODER,
		"cutie_object_num_roles": 3,
		"cutie_object_role_names": list(roles),
		"cutie_object_only_latent_dim": 192,
		"latent_dim": 192,
		"cutie_object_spatial_token_enabled": False,
		"cutie_object_variable_graph_enabled": False,
		"cutie_object_allow_simulator_runtime": False,
		"cutie_object_allow_simulator_kinematics_runtime": False,
		"object_state_supervision_enabled": False,
		"action_dim": action_dim,
	}
	bad = {key: (raw.get(key), value) for key, value in expected.items()
		   if raw.get(key) != value}
	if bad:
		raise ValueError(f"Unsafe K=3 runtime config: {bad}")
	raw.update(
		compile=False,
		compile_fallback_random=False,
		multitask=False,
		task_dim=0,
		tasks=[task],
		save_agent=False,
		save_csv=False,
		save_video=False,
		enable_wandb=False,
		checkpoint=None,
		work_dir=str(path.resolve().parent / "gpu_smoke_no_output"),
	)
	cfg = cfg_to_dataclass(OmegaConf.create(raw))
	cfg.work_dir = path.resolve().parent / "gpu_smoke_no_output"
	return cfg, roles, action_dim


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--runtime-config", type=Path, required=True)
	args = parser.parse_args()
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required for this smoke test.")
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.set_float32_matmul_precision("high")

	cfg, roles, action_dim = _load_config(args.runtime_config)
	env = agent = reloaded = None
	try:
		set_seed(int(cfg.seed))
		env = make_env(cfg)
		if int(cfg.action_dim) != action_dim or tuple(env.action_space.shape) != (action_dim,):
			raise AssertionError(
				f"Action contract is {cfg.action_dim}, {env.action_space.shape}; "
				f"expected {action_dim}."
			)
		if tuple(cfg.obs_shape.get("object", ())) != (3, 1770):
			raise AssertionError(f"Unexpected object shape: {cfg.obs_shape}")

		obs = env.reset()
		if set(obs.keys()) != {"object"} or tuple(obs["object"].shape) != (3, 1770):
			raise AssertionError(f"Unexpected real observation: {obs.keys()}")
		rebuild_cfg = deepcopy(cfg)
		agent = TDMPC2(cfg).eval()
		if int(agent.cfg.latent_dim) != 192:
			raise AssertionError(f"Latent width is {agent.cfg.latent_dim}, not 192.")
		action = agent.act(obs, t0=True, eval_mode=True)
		if tuple(action.shape) != (action_dim,) or not torch.isfinite(action).all():
			raise AssertionError(f"Invalid planned action: {tuple(action.shape)}")
		_, reward, done, _ = env.step(action)
		if not np.isfinite(float(reward)) or bool(done):
			raise AssertionError(f"Invalid first transition: reward={reward}, done={done}")

		checkpoint = BytesIO()
		agent.save(checkpoint)
		checkpoint.seek(0)
		payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
		contract = payload.get("checkpoint_contract", {})
		if contract.get("cutie_object_regression_encoder") != ENCODER:
			raise AssertionError("Checkpoint lost the K=3 encoder contract.")
		if int(contract.get("latent_dim", -1)) != 192:
			raise AssertionError("Checkpoint lost the 192-D latent contract.")
		reloaded = TDMPC2(rebuild_cfg).eval()
		reloaded.load(payload)

		print("CUTIE_K3_GPU_SMOKE_OK " + json.dumps({
			"task": cfg.task,
			"roles": list(roles),
			"observation_shape": list(obs["object"].shape),
			"latent_dim": 192,
			"action_dim": int(action.numel()),
			"checkpoint_strict_reload": True,
			"runtime_simulator_state": False,
		}, sort_keys=True))
	finally:
		close = getattr(env, "close", None) if env is not None else None
		if callable(close):
			close()
		del reloaded, agent
		torch.cuda.empty_cache()


if __name__ == "__main__":
	main()
