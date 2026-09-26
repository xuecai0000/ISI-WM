"""CUDA contracts for the mandatory predicted-state control bottleneck."""

from pathlib import Path
import tempfile

import torch
from tensordict import TensorDict

from check_object_state_supervision_update import Batch, make_config
from common import object_state_supervision as supervision
from common.seed import set_seed
from tdmpc2 import TDMPC2


def main():
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required.')
	torch.set_num_threads(2)
	cfg = make_config(1.0, 'acrobot-swingup')
	cfg.object_state_bottleneck_enabled = True
	cfg.object_state_bottleneck_hidden_dim = 128
	set_seed(271828)
	agent = TDMPC2(cfg)
	state_dim = supervision.target_dim(cfg)
	assert state_dim == 6
	assert set(agent.model._encoder) == {'object'}
	assert agent.model._dynamics.state_dim == state_dim

	batch = Batch(cfg)
	with torch.no_grad():
		z = agent.model.encode(batch.obs[0], None)
		assert z.shape == (cfg.batch_size, cfg.latent_dim)
		assert torch.count_nonzero(z[..., state_dim:]) == 0
		next_z = agent.model.next(z, batch.action[0], None)
		assert torch.count_nonzero(next_z[..., state_dim:]) == 0
		torch.testing.assert_close(agent.model.decode_supervised_state(z), z[..., :state_dim])
		# Labels are not part of encode or any deployed model call.
		before = z.clone()
		batch.target.add_(1000.)
		torch.testing.assert_close(before, agent.model.encode(batch.obs[0], None))

	metrics = agent.update(batch)
	assert all(torch.isfinite(value).all() for value in metrics.values())
	assert float(metrics['state_supervision_weighted']) > 0
	assert any(
		parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
		for parameter in agent.model._state_supervision_head.parameters()
	) is False, 'Optimizer should clear gradients after update.'

	with tempfile.TemporaryDirectory() as directory:
		checkpoint = Path(directory) / 'bottleneck.pt'
		agent.save(checkpoint)
		payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
		contract = payload['checkpoint_contract']['object_state_supervision']
		assert contract['controller_input_contains_predicted_state'] is True
		assert contract['controller_visual_bypass'] is False
		assert contract['bottleneck_state_dim'] == state_dim
		cfg.object_state_supervision_collect_labels = False
		agent.load(checkpoint)

	print('OBJECT_STATE_BOTTLENECK_CUDA_CONTRACT_OK', flush=True)


if __name__ == '__main__':
	main()
