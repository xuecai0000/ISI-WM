"""Formal Hydra/parse_cfg width contract for ROF-WM V0."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from common import robust_object_field as rof
from common.parser import parse_cfg


CONFIG_PATH = Path(__file__).with_name('config.yaml')


TASKS = (
	('acrobot-swingup', ['whole_acrobot'], 320),
	('reacher-easy', ['whole_arm', 'goal'], 640),
	('hopper-hop', ['torso', 'leg', 'foot'], 960),
)


def make_raw(task, roles):
	cfg = OmegaConf.load(CONFIG_PATH)
	cfg.task = task
	cfg.obs = 'rgb'
	cfg.model_size = 5
	cfg.flat_anchor = True
	cfg.flat_anchor_mode = 'cutie_object_only'
	cfg.robust_object_field_enabled = True
	cfg.cutie_object_num_roles = len(roles)
	cfg.cutie_object_role_names = list(roles)
	# These deliberately wrong pre-parse values model config.yaml and prove that
	# MODEL_SIZE plus the formal ROF derivation resolves the final width.
	cfg.cutie_object_only_latent_dim = 128
	cfg.latent_dim = 512
	return cfg


def main():
	results = {}
	with patch('hydra.utils.get_original_cwd', return_value=str(CONFIG_PATH.parent)):
		for task, roles, expected in TASKS:
			cfg = parse_cfg(make_raw(task, roles))
			rof.validate_config(cfg, require_obs_shape=False)
			actual = (int(cfg.cutie_object_only_latent_dim), int(cfg.latent_dim))
			if actual != (expected, expected):
				raise AssertionError(
					f'{task}: formal parser emitted {actual}, expected {(expected, expected)}.'
				)
			results[task] = expected

	# A forbidden scientific option must fail in the same formal entry point.
	bad = make_raw('reacher-easy', ['whole_arm', 'goal'])
	bad.cutie_object_last_valid_memory = True
	with patch('hydra.utils.get_original_cwd', return_value=str(CONFIG_PATH.parent)):
		try:
			parse_cfg(bad)
		except ValueError:
			pass
		else:
			raise AssertionError('Formal parser accepted forbidden last-valid memory.')

	print('ROBUST_OBJECT_FIELD_PARSER_CONTRACT_OK', results)


if __name__ == '__main__':
	main()
