"""CPU-only structural checks for OC-style anchor TD-MPC2.

This test does not import the frozen DINO teacher or create an environment.
Run on the server with::

    python tdmpc2/check_oc_anchor_contract.py
"""

from types import SimpleNamespace

import torch

from common.world_model import WorldModel


class Config(SimpleNamespace):
	def get(self, key, default=None):
		return getattr(self, key, default)


def make_config():
	return Config(
		multitask=False,
		obs='rgb',
		obs_shape={'rgb': (9, 64, 64), 'anchor': (25,)},
		action_dim=2,
		task_dim=0,
		num_channels=32,
		latent_dim=768,
		mlp_dim=512,
		num_bins=101,
		num_q=5,
		dropout=0.01,
		episodic=False,
		log_std_min=-10,
		log_std_max=2,
		simnorm_dim=8,
		flat_anchor=True,
		flat_anchor_mode='oc',
		flat_anchor_dim=25,
		flat_anchor_hidden_dim=128,
		flat_anchor_num_roles=4,
		flat_anchor_role_input_dim=10,
		flat_anchor_scene_dim=512,
		flat_anchor_role_dim=64,
		flat_anchor_joint_dim=768,
		flat_anchor_token_dim=256,
		flat_anchor_num_heads=4,
		flat_anchor_num_layers=2,
		flat_anchor_loss_weight_floor=0.1,
		flat_anchor_loss_beta=0.1,
	)


def check_augmentation(model):
	# Use an asymmetric, explicit crop and a one-pixel marker so this test checks
	# both the coordinate sign and RGB/anchor covariance (not merely that all
	# four points received some identical delta).
	rgb = torch.zeros(1, 9, 64, 64)
	x, y = 30, 31
	rgb[0, :, y, x] = 255.
	anchor = torch.zeros(1, 25)
	anchor[:, :8] = torch.tensor([
		x * 2. / 63. - 1., y * 2. / 63. - 1.,
		0., 0., 0., 0., 0., 0.,
	])
	anchor[:, 8:] = torch.arange(17).float()
	shift_index = torch.tensor([[[[0, 6]]]])
	shifted_rgb, shifted = model._oc_augmentation(
		rgb, anchor, shift_index=shift_index
	)
	expected_pixel_delta = torch.tensor([3, -3])
	expected_x = x + int(expected_pixel_delta[0])
	expected_y = y + int(expected_pixel_delta[1])
	marker = shifted_rgb[0, 0]
	marker_index = int(marker.argmax())
	marker_y, marker_x = divmod(marker_index, marker.shape[-1])
	assert (marker_x, marker_y) == (expected_x, expected_y), \
		f'RGB marker moved to {(marker_x, marker_y)}, expected {(expected_x, expected_y)}.'
	expected_xy = torch.tensor([
		expected_x * 2. / 63. - 1.,
		expected_y * 2. / 63. - 1.,
	])
	assert torch.allclose(shifted[0, :2], expected_xy, atol=1e-6), \
		f'Anchor moved to {shifted[0, :2]}, expected {expected_xy}.'
	before = anchor[:, :8].reshape(1, 4, 2)
	after = shifted[:, :8].reshape(1, 4, 2)
	delta = after - before
	assert torch.allclose(delta, delta[:, :1].expand_as(delta)), \
		'All role positions must receive the same image translation.'
	assert torch.equal(shifted[:, 8:], anchor[:, 8:]), \
		'Relations, velocities, confidence, and fallback must be translation invariant.'
	return float(delta.abs().max())


def check_confidence_weighting(model):
	high_target = torch.zeros(1, 25)
	high_target[:, 20:24] = 1.
	low_target = high_target.clone()
	low_target[:, 20] = 0.
	high_prediction = high_target.clone()
	low_prediction = low_target.clone()
	high_prediction[:, 0] = 1.
	low_prediction[:, 0] = 1.
	high = model.anchor_loss(high_prediction, high_target)
	low = model.anchor_loss(low_prediction, low_target)
	assert high > low, 'Low-confidence localization errors must receive less weight.'
	return float(high), float(low)


def check_role_schema(model):
	anchor = torch.tensor([[
		1., 2., 3., 4., 5., 6., 7., 8.,
		9., 10., 11., 12., 13., 14.,
		15., 16., 17., 18., 19., 20.,
		.1, .2, .3, .4, 1.,
	]])
	features = model._encoder['oc'].role_inputs(anchor)[..., :10]
	expected = torch.tensor([[
		[1., 2., 0., 0., 0., 0., 9., 10., .1, 1.],
		[3., 4., 15., 16., 9., 10., 11., 12., .2, 1.],
		[5., 6., 17., 18., 11., 12., 13., 14., .3, 1.],
		[7., 8., 19., 20., 13., 14., 0., 0., .4, 1.],
	]])
	assert torch.equal(features, expected), '25-D fields were routed to the wrong role.'


def main():
	torch.manual_seed(7)
	model = WorldModel(make_config())
	model.train()
	check_role_schema(model)
	rgb = torch.randint(0, 256, (3, 9, 64, 64), dtype=torch.uint8)
	anchor = torch.randn(3, 25)
	anchor[:, 20:24] = torch.sigmoid(anchor[:, 20:24])
	anchor[:, 24] = 0.

	max_shift = check_augmentation(model)
	torch.manual_seed(13)
	z, used_anchor = model.encode({'rgb': rgb, 'anchor': anchor}, None, return_anchor=True)
	assert z.shape == (3, 768)
	assert used_anchor.shape == (3, 25)

	action = torch.randn(3, 2).clamp(-1, 1)
	next_z = model.next(z, action, None)
	assert next_z.shape == (3, 768)
	decoded = model.decode_anchor(next_z)
	assert decoded.shape == (3, 25)
	assert torch.isfinite(next_z).all() and torch.isfinite(decoded).all()

	# The auxiliary object decoder must not read or reconstruct from scene state.
	changed_scene = next_z.clone()
	changed_scene[:, :model.cfg.flat_anchor_scene_dim] = torch.randn_like(
		changed_scene[:, :model.cfg.flat_anchor_scene_dim]
	)
	assert torch.equal(decoded, model.decode_anchor(changed_scene))

	loss = model.anchor_loss(decoded, used_anchor)
	loss.backward()
	role_grad = model._encoder['oc'].role[0].weight.grad
	attention_grad = model._dynamics.blocks[0].attn.qkv.weight.grad
	decoder_grad = model._anchor_decoder.decoder[0].weight.grad
	for name, gradient in {
		'role_encoder': role_grad,
		'token_attention': attention_grad,
		'anchor_decoder': decoder_grad,
	}.items():
		assert gradient is not None and torch.isfinite(gradient).all(), f'{name} gradient is invalid.'
		assert torch.count_nonzero(gradient), f'{name} must receive a nonzero first-step gradient.'

	# Sequence input follows replay layout [T, B, ...].
	seq_rgb = rgb[:2].unsqueeze(1)
	seq_anchor = anchor[:2].unsqueeze(1)
	torch.manual_seed(17)
	seq_z, seq_used_anchor = model.encode(
		{'rgb': seq_rgb, 'anchor': seq_anchor},
		None,
		return_anchor=True,
	)
	assert seq_z.shape == (2, 1, 768)
	assert seq_used_anchor.shape == (2, 1, 25)
	seq_delta = (
		seq_used_anchor[..., :8].reshape(2, 1, 4, 2)
		- seq_anchor[..., :8].reshape(2, 1, 4, 2)
	)
	assert torch.allclose(seq_delta, seq_delta[:1].expand_as(seq_delta)), \
		'Every timestep in a sampled trajectory must share one spatial shift.'

	high_loss, low_loss = check_confidence_weighting(model)
	print('OC_ANCHOR_CONTRACT_OK', {
		'latent': tuple(z.shape),
		'sequence_latent': tuple(seq_z.shape),
		'max_joint_shift': max_shift,
		'high_conf_loss': high_loss,
		'low_conf_loss': low_loss,
	})


if __name__ == '__main__':
	main()
