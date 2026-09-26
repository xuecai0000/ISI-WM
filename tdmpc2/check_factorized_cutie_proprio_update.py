"""One synthetic CUDA update for the factorized Cutie-proprio world model."""

import torch

from check_cutie_proprio_update import Buffer, make_config, sample
from common.seed import set_seed
from tdmpc2 import TDMPC2


def main():
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required.')
	results = {}
	for mode in ('factorized', 'factorized_proprio_only'):
		set_seed(271828)
		cfg = make_config(mode)
		agent = TDMPC2(cfg)
		if cfg.latent_dim != 128:
			raise AssertionError(f'Expected 128-D latent, got {cfg.latent_dim}.')
		dynamics = agent.model._dynamics
		if not getattr(dynamics, 'factorized', False):
			raise AssertionError(f'{mode}: factorized dynamics were not constructed.')
		before = {
			name: value.detach().clone()
			for name, value in agent.model.named_parameters()
			if value.requires_grad
		}
		batch = sample(cfg)
		if mode == 'factorized_proprio_only':
			batch[0]['object'][..., :1770] = 0.0
		metrics = agent.update(Buffer(batch))
		if not metrics or any(
			not torch.isfinite(value).all()
			for value in metrics.values() if torch.is_tensor(value)
		):
			raise AssertionError(f'{mode}: update emitted non-finite metrics.')
		changed = sum(
			int(not torch.equal(before[name], value.detach()))
			for name, value in agent.model.named_parameters() if name in before
		)
		if changed < 1:
			raise AssertionError(f'{mode}: update changed no trainable parameter.')
		results[mode] = {'changed_parameter_tensors': changed}
		del agent
		torch.cuda.empty_cache()
	print('FACTORIZED_CUTIE_PROPRIO_UPDATE_OK', {
		'device': torch.cuda.get_device_name(0),
		'latent_dim': 128,
		'body_dim': dynamics.factor_dim,
		'object_dim': dynamics.factor_dim,
		'modes': results,
	})


if __name__ == '__main__':
	main()
