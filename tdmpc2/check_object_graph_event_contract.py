"""CPU-only contract for fast ObjectGraph/Event-Anchor TD-MPC2.

This check does not import the external anchor teacher, construct DINOv2, or
create an environment. Run on the server with::

    python tdmpc2/check_object_graph_event_contract.py
"""

from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from common.world_model import WorldModel
from envs.wrappers.event_anchor_state import scheduled_refresh_reasons


CONFIG_PATH = Path(__file__).with_name("config.yaml")
ROLE_NAMES = ("base", "elbow", "tip", "goal")
ACTION_ROLE_MASK = torch.tensor([0.0, 1.0, 1.0, 0.0])


class Config(SimpleNamespace):
	def get(self, key, default=None):
		return getattr(self, key, default)


def load_defaults():
	with CONFIG_PATH.open(encoding="utf-8") as stream:
		return yaml.safe_load(stream)


def make_model_config(defaults):
	role_dim = defaults["flat_anchor_object_role_dim"]
	return Config(
		multitask=False,
		tasks=["reacher-easy"],
		obs="rgb",
		obs_shape={"anchor": (25,)},
		action_dim=2,
		task_dim=0,
		num_channels=32,
		latent_dim=len(ROLE_NAMES) * role_dim,
		mlp_dim=64,
		num_bins=101,
		num_q=2,
		dropout=0.01,
		episodic=False,
		log_std_min=-10,
		log_std_max=2,
		simnorm_dim=8,
		flat_anchor=True,
		flat_anchor_mode="object_graph",
		flat_anchor_dim=25,
		flat_anchor_hidden_dim=128,
		flat_anchor_num_roles=len(ROLE_NAMES),
		flat_anchor_object_role_dim=role_dim,
		flat_anchor_object_hidden_dim=defaults["flat_anchor_object_hidden_dim"],
		flat_anchor_loss_weight_floor=0.1,
		flat_anchor_loss_beta=0.1,
	)


def check_defaults(defaults):
	assert defaults["flat_anchor"] is False, \
		"The official TD-MPC2 path must remain selected by default."
	assert defaults["flat_anchor_mode"] == "object_graph"
	assert defaults["flat_anchor_event_enabled"] is False
	assert defaults["flat_anchor_num_roles"] == len(ROLE_NAMES)
	assert defaults["flat_anchor_object_role_dim"] == 32
	assert defaults["flat_anchor_object_hidden_dim"] == 64
	assert defaults["flat_anchor_event_max_interval"] == 12
	assert defaults["flat_anchor_event_min_refresh_gap"] == 4
	for key in (
		"flat_anchor_event_confidence_threshold",
		"flat_anchor_event_color_confidence_threshold",
	):
		assert 0.0 <= defaults[key] <= 1.0
	assert defaults["flat_anchor_event_structure_tolerance"] > 0.0
	assert defaults["flat_anchor_event_color_search_radius"] > 0


def check_event_schedule(defaults):
	max_interval = defaults["flat_anchor_event_max_interval"]
	assert scheduled_refresh_reasons([True], [0], max_interval) == {"reset"}
	assert scheduled_refresh_reasons(
		[False], [max_interval - 1], max_interval
	) == set()
	assert scheduled_refresh_reasons(
		[False], [max_interval], max_interval
	) == {"max_interval"}
	# Reset dominates timeout so one frame produces one scheduled refresh reason.
	assert scheduled_refresh_reasons(
		[True], [max_interval], max_interval
	) == {"reset"}


def check_object_graph(defaults):
	cfg = make_model_config(defaults)
	model = WorldModel(cfg)
	assert set(model._encoder.keys()) == {"anchor"}, \
		"ObjectGraph must not construct or plan through an RGB encoder."

	batch = 3
	anchor = torch.randn(batch, 25)
	anchor[:, 20:24] = torch.sigmoid(anchor[:, 20:24])
	anchor[:, 24] = 0.0
	z, used_anchor = model.encode({"anchor": anchor}, None, return_anchor=True)
	expected_dim = len(ROLE_NAMES) * defaults["flat_anchor_object_role_dim"]
	assert expected_dim == 128
	assert z.shape == (batch, expected_dim)
	assert torch.equal(used_anchor, anchor)

	mask = model._dynamics._action_role_mask.flatten().cpu()
	assert torch.equal(mask, ACTION_ROLE_MASK), \
		"Actions must be exposed only to the controllable elbow and tip roles."

	# Inspect the actual local-transition input, rather than checking only the
	# stored mask. Action channels follow prev/current/next role features.
	captured = {}
	def capture_transition_input(_module, args):
		captured["input"] = args[0].detach()

	hook = model._dynamics.transition.register_forward_pre_hook(
		capture_transition_input
	)
	action = torch.tensor([[0.25, -0.75]]).expand(batch, -1)
	next_z = model._dynamics(z, action)
	hook.remove()
	role_dim = defaults["flat_anchor_object_role_dim"]
	routed_action = captured["input"][..., 3 * role_dim:3 * role_dim + 2]
	expected_action = torch.zeros(batch, len(ROLE_NAMES), 2)
	expected_action[:, 1:3] = action.unsqueeze(1)
	assert torch.equal(routed_action, expected_action)
	assert next_z.shape == z.shape
	assert model.decode_anchor(next_z).shape == (batch, 25)
	return tuple(z.shape), tuple(next_z.shape)


def main():
	torch.manual_seed(23)
	defaults = load_defaults()
	check_defaults(defaults)
	check_event_schedule(defaults)
	latent_shape, next_shape = check_object_graph(defaults)
	print("OBJECT_GRAPH_EVENT_CONTRACT_OK", {
		"default_enabled": defaults["flat_anchor"],
		"mode": defaults["flat_anchor_mode"],
		"roles": ROLE_NAMES,
		"latent_shape": latent_shape,
		"next_shape": next_shape,
		"action_role_mask": ACTION_ROLE_MASK.tolist(),
		"event_max_interval": defaults["flat_anchor_event_max_interval"],
		"event_min_refresh_gap": defaults["flat_anchor_event_min_refresh_gap"],
	})


if __name__ == "__main__":
	main()
