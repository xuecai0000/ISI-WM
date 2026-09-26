"""Dependency-light model contract for the live CutieHybrid pilot.

This test never constructs an environment, Cutie, Hydra application, support
pack, checkpoint, or simulator.  It exercises the real WorldModel on CPU and
proves that the new branch is step-zero identical to the official RGB model,
while reward, Q, and policy can all become object-conditioned after training.
"""

from types import SimpleNamespace

import torch

from common.world_model import WorldModel


class Config(SimpleNamespace):
	def get(self, key, default=None):
		return getattr(self, key, default)


def make_config(cutie: bool) -> Config:
	return Config(
		multitask=False,
		tasks=['reacher-visual-small'],
		task_dim=0,
		action_dim=2,
		obs='rgb',
		obs_shape={
			'rgb': (9, 64, 64),
			**({'object': (2, 1770)} if cutie else {}),
		},
		num_channels=32,
		latent_dim=512,
		mlp_dim=512,
		num_bins=101,
		episodic=False,
		num_q=5,
		dropout=0.01,
		log_std_min=-10,
		log_std_max=2,
		tau=0.01,
		simnorm_dim=8,
		flat_anchor=cutie,
		flat_anchor_mode='cutie_hybrid',
		flat_anchor_scene_dim=512,
		flat_anchor_hybrid_hidden_dim=128,
		flat_anchor_loss_weight_floor=0.1,
		flat_anchor_loss_beta=0.1,
		cutie_object_num_roles=2,
		cutie_object_frame_dim=590,
		cutie_object_stack_frames=3,
		cutie_object_input_dim=1770,
		cutie_object_role_dim=64,
		cutie_object_hidden_dim=256,
		cutie_object_joint_dim=640,
	)


def exact(left, right, label):
	left_is_tensor = torch.is_tensor(left)
	right_is_tensor = torch.is_tensor(right)
	if left_is_tensor != right_is_tensor:
		raise AssertionError(
			f'{label} changed state type: {type(left).__name__} versus '
			f'{type(right).__name__}.'
		)
	if not left_is_tensor:
		if type(left) is not type(right) or left != right:
			raise AssertionError(
				f'{label} lost step-zero parity: {left!r} versus {right!r}.'
			)
		return
	if not torch.equal(left, right):
		error = float((left.float() - right.float()).abs().max())
		raise AssertionError(f'{label} lost step-zero parity (max error {error}).')


def main():
	torch.set_num_threads(1)
	seed = 271828
	torch.manual_seed(seed)
	official_cfg = make_config(False)
	official = WorldModel(official_cfg).eval()
	torch.manual_seed(seed)
	cutie_cfg = make_config(True)
	cutie = WorldModel(cutie_cfg).eval()

	if official_cfg.latent_dim != 512 or cutie_cfg.latent_dim != 640:
		raise AssertionError('Official/Cutie latent widths must be 512/640.')
	cutie_state = cutie.state_dict()
	for name, value in official.state_dict().items():
		if name not in cutie_state:
			raise AssertionError(f'CutieHybrid removed official state tensor {name}.')
		exact(value, cutie_state[name], f'official state {name}')

	batch = 3
	rgb = torch.randint(0, 256, (batch, 9, 64, 64), dtype=torch.uint8)
	objects_a = torch.randn(batch, 2, 1770)
	objects_b = objects_a.clone()
	objects_b[:, 0].add_(0.75)

	torch.manual_seed(141421)
	z_rgb = official.encode(rgb, task=None)
	torch.manual_seed(141421)
	z_a = cutie.encode({'rgb': rgb, 'object': objects_a}, task=None)
	exact(z_rgb, z_a[..., :512], 'RGB encoding')
	if z_a.shape != (batch, 640):
		raise AssertionError(f'Unexpected joint latent shape {tuple(z_a.shape)}.')

	action = torch.tensor([[0.2, -0.4]]).expand(batch, -1)
	next_rgb = official.next(z_rgb, action, None)
	next_a = cutie.next(z_a, action, None)
	exact(next_rgb, next_a[..., :512], 'scene dynamics')
	if next_a[..., 512:].shape != (batch, 128):
		raise AssertionError('Object imagination must roll exactly 128 latent values.')

	exact(
		official.reward(z_rgb, action, None),
		cutie.reward(z_a, action, None),
		'reward logits',
	)
	exact(
		official.Q(z_rgb, action, None, return_type='all'),
		cutie.Q(z_a, action, None, return_type='all'),
		'Q logits',
	)
	exact(
		official.Q(z_rgb, action, None, return_type='all', target=True),
		cutie.Q(z_a, action, None, return_type='all', target=True),
		'target Q logits',
	)
	exact(
		official.Q(z_rgb, action, None, return_type='all', detach=True),
		cutie.Q(z_a, action, None, return_type='all', detach=True),
		'detached Q logits',
	)
	torch.manual_seed(173205)
	official_action, official_pi = official.pi(z_rgb, None)
	torch.manual_seed(173205)
	cutie_action, cutie_pi = cutie.pi(z_a, None)
	exact(official_action, cutie_action, 'sampled policy action')
	exact(official_pi['mean'], cutie_pi['mean'], 'policy mean')
	exact(official_pi['log_std'], cutie_pi['log_std'], 'policy log-std')

	zero_tensors = [
		cutie._hybrid_reward.output.weight,
		cutie._hybrid_reward.output.bias,
		cutie._hybrid_pi.output.weight,
		cutie._hybrid_pi.output.bias,
		cutie._hybrid_q.params['output', 'weight'],
		cutie._hybrid_q.params['output', 'bias'],
	]
	if any(torch.count_nonzero(value).item() for value in zero_tensors):
		raise AssertionError('Every Cutie decision correction must start at exact zero.')

	decoded = cutie.decode_object(z_a)
	if decoded.shape != objects_a.shape or not torch.isfinite(decoded).all():
		raise AssertionError('Object decoder returned an invalid tensor.')
	object_loss = cutie.object_loss(decoded, objects_a)
	if object_loss.ndim or not torch.isfinite(object_loss):
		raise AssertionError('Object reconstruction loss must be a finite scalar.')
	other_action = torch.tensor([[-0.7, 0.6]]).expand(batch, -1)
	other_next = cutie.next(z_a, other_action, None)
	if float((next_a[..., 512:] - other_next[..., 512:]).abs().max()) <= 1e-7:
		raise AssertionError('Object imagination dynamics ignored the action.')

	# Emulate learned correction outputs and prove all three control heads can use
	# the compact object state.  Keep the scene fixed to isolate object influence.
	torch.manual_seed(314159)
	z_b_encoded = cutie.encode({'rgb': rgb, 'object': objects_b}, task=None)
	z_b = torch.cat([z_a[..., :512], z_b_encoded[..., 512:]], dim=-1)
	if torch.equal(z_a[..., 512:], z_b[..., 512:]):
		raise AssertionError('Object encoder ignored a changed whole-arm observation.')
	with torch.no_grad():
		cutie._hybrid_reward.output.weight.normal_(std=0.05)
		cutie._hybrid_pi.output.weight.normal_(std=0.05)
		cutie._hybrid_q.params['output', 'weight'].normal_(std=0.05)

	reward_delta = float((
		cutie.reward(z_a, action, None) - cutie.reward(z_b, action, None)
	).abs().max())
	q_delta = float((
		cutie.Q(z_a, action, None, return_type='all')
		- cutie.Q(z_b, action, None, return_type='all')
	).abs().max())
	torch.manual_seed(161803)
	_, pi_a = cutie.pi(z_a, None)
	torch.manual_seed(161803)
	_, pi_b = cutie.pi(z_b, None)
	pi_delta = float((pi_a['mean'] - pi_b['mean']).abs().max())
	if min(reward_delta, q_delta, pi_delta) <= 1e-7:
		raise AssertionError(
			'Learned reward/Q/policy corrections must all respond to object state: '
			f'{reward_delta=}, {q_delta=}, {pi_delta=}.'
		)

	# One real backward pass must reach every new trainable subsystem and remain
	# finite. This catches optimizer-looking integrations whose graph was detached.
	cutie.zero_grad(set_to_none=True)
	torch.manual_seed(223607)
	z_train = cutie.encode({'rgb': rgb, 'object': objects_a}, task=None)
	next_train = cutie.next(z_train, action, None)
	train_action, _ = cutie.pi(z_train, None)
	train_loss = (
		cutie.reward(z_train, action, None).mean()
		+ cutie.Q(z_train, action, None, return_type='all').mean()
		+ train_action.mean()
		+ cutie.object_loss(cutie.decode_object(next_train), objects_a)
	)
	train_loss.backward()
	modules = {
		'object_encoder': cutie._encoder['object'],
		'object_dynamics': cutie._graph_dynamics,
		'object_decoder': cutie._object_decoder,
		'reward_correction': cutie._hybrid_reward,
		'q_correction': cutie._hybrid_q,
		'pi_correction': cutie._hybrid_pi,
	}
	for name, module in modules.items():
		gradients = [
			parameter.grad for parameter in module.parameters()
			if parameter.requires_grad and parameter.grad is not None
		]
		if not gradients:
			raise AssertionError(f'{name} received no gradient.')
		if not all(torch.isfinite(gradient).all() for gradient in gradients):
			raise AssertionError(f'{name} received a non-finite gradient.')
		if not any(torch.count_nonzero(gradient).item() for gradient in gradients):
			raise AssertionError(f'{name} gradients are all zero.')

	print('CUTIE_HYBRID_CONTRACT_OK', {
		'joint_shape': tuple(z_a.shape),
		'object_prediction_shape': tuple(next_a[..., 512:].shape),
		'decoder_shape': tuple(decoded.shape),
		'reward_object_delta': reward_delta,
		'q_object_delta': q_delta,
		'pi_object_delta': pi_delta,
		'backward_loss': float(train_loss.detach()),
	})


if __name__ == '__main__':
	main()
