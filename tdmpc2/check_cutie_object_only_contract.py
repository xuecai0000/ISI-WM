"""CPU contract for the structural Cutie object-only world model.

The contract never constructs Cutie, Hydra, a simulator, or a GPU process.  It
proves that RGB is absent from the agent/model schema, the complete latent is
the two-role 128-D object state, every control head consumes that state, and
the object reconstruction/prediction path remains trainable.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from common import layers
from common.world_model import WorldModel


class Config(SimpleNamespace):
	def get(self, key, default=None):
		return getattr(self, key, default)


def make_config(auxiliary_target='full_descriptor') -> Config:
	cfg = Config(
		multitask=False,
		tasks=['reacher-visual-small'],
		task_dim=0,
		action_dim=2,
		obs='rgb',  # Camera input remains required by the environment-side Cutie.
		obs_shape={'object': (2, 1770)},
		num_channels=32,
		latent_dim=128,
		mlp_dim=512,
		num_bins=101,
		episodic=False,
		num_q=5,
		dropout=0.01,
		log_std_min=-10,
		log_std_max=2,
		tau=0.01,
		simnorm_dim=8,
		flat_anchor=True,
		flat_anchor_mode='cutie_object_only',
		flat_anchor_loss_weight_floor=0.1,
		flat_anchor_loss_beta=0.1,
		cutie_object_num_roles=2,
		cutie_object_frame_dim=590,
		cutie_object_stack_frames=3,
		cutie_object_input_dim=1770,
		cutie_object_observation_variant='full',
		cutie_object_role_dim=64,
		cutie_object_hidden_dim=256,
		cutie_object_only_latent_dim=128,
	)
	if auxiliary_target is not None:
		cfg.cutie_object_auxiliary_target = auxiliary_target
	return cfg


def _assert_finite_nonzero_grad(module, label):
	gradients = [
		parameter.grad
		for parameter in module.parameters()
		if parameter.requires_grad and parameter.grad is not None
	]
	if not gradients:
		raise AssertionError(f'{label} received no gradient.')
	if not all(torch.isfinite(gradient).all() for gradient in gradients):
		raise AssertionError(f'{label} received a non-finite gradient.')
	if not any(torch.count_nonzero(gradient).item() for gradient in gradients):
		raise AssertionError(f'{label} gradients are all zero.')


def _weighted_object_mean(element_loss, target, floor=0.1):
	valid = target[..., 2 * 590 + 588].clamp(0., 1.)
	role_weight = floor + (1. - floor) * valid
	role_loss = element_loss.reshape(*target.shape[:-1], -1).mean(dim=-1)
	return (role_loss * role_weight).sum() / role_weight.sum().clamp_min(1.)


def _assert_auxiliary_target_contract(objects):
	"""Freeze old loss bits and the new masked/full-denominator gradient scale."""
	torch.manual_seed(223607)
	target = objects.detach().clone()
	prediction_value = torch.randn_like(target)

	# Missing-key legacy config and the explicit default must execute the same
	# operations and produce bitwise-identical losses and gradients.
	legacy = WorldModel(make_config(auxiliary_target=None)).eval()
	explicit = WorldModel(make_config('full_descriptor')).eval()
	legacy_prediction = prediction_value.clone().requires_grad_(True)
	explicit_prediction = prediction_value.clone().requires_grad_(True)
	legacy_loss = legacy.object_loss(legacy_prediction, target)
	explicit_loss = explicit.object_loss(explicit_prediction, target)
	legacy_grad, = torch.autograd.grad(legacy_loss, legacy_prediction)
	explicit_grad, = torch.autograd.grad(explicit_loss, explicit_prediction)
	reference_prediction = prediction_value.clone().requires_grad_(True)
	reference_loss = _weighted_object_mean(
		F.smooth_l1_loss(
			reference_prediction,
			target,
			reduction='none',
			beta=0.1,
		),
		target,
	)
	reference_grad, = torch.autograd.grad(reference_loss, reference_prediction)
	if not torch.equal(legacy_loss, explicit_loss):
		raise AssertionError('Explicit full_descriptor changed the legacy loss bits.')
	if not torch.equal(legacy_grad, explicit_grad):
		raise AssertionError('Explicit full_descriptor changed the legacy gradient bits.')
	if not torch.equal(legacy_loss, reference_loss):
		raise AssertionError('Default loss changed from the frozen Full-Cutie formula.')
	if not torch.equal(legacy_grad, reference_grad):
		raise AssertionError('Default gradient changed from the frozen Full-Cutie formula.')

	geometry = WorldModel(
		make_config('geometry_status_full_denominator')
	).eval()
	geometry.load_state_dict(explicit.state_dict(), strict=True)
	geometry_prediction = prediction_value.clone().requires_grad_(True)
	geometry_loss = geometry.object_loss(geometry_prediction, target)
	geometry_grad, = torch.autograd.grad(geometry_loss, geometry_prediction)

	manual_prediction = prediction_value.clone().requires_grad_(True)
	manual_element_loss = F.smooth_l1_loss(
		manual_prediction,
		target,
		reduction='none',
		beta=0.1,
	).reshape(*target.shape[:-1], 3, 590)
	manual_mask = manual_element_loss.new_zeros(590)
	manual_mask[512:] = 1.
	manual_loss = _weighted_object_mean(
		manual_element_loss * manual_mask,
		target,
	)
	manual_grad, = torch.autograd.grad(manual_loss, manual_prediction)
	if not torch.equal(geometry_loss, manual_loss):
		raise AssertionError(
			'geometry_status target did not use a masked full-1770 mean.'
		)
	if not torch.equal(geometry_grad, manual_grad):
		raise AssertionError(
			'geometry_status gradients differ from the masked full-1770 reference.'
		)

	geometry_frames = geometry_grad.reshape(*target.shape[:-1], 3, 590)
	full_frames = explicit_grad.reshape(*target.shape[:-1], 3, 590)
	if torch.count_nonzero(geometry_frames[..., :512]).item() != 0:
		raise AssertionError('geometry_status target leaked a query gradient.')
	if not torch.equal(geometry_frames[..., 512:], full_frames[..., 512:]):
		raise AssertionError(
			'geometry/status gradient scale changed relative to full_descriptor.'
		)

	# Auxiliary masking must not mask the live Full-Cutie encoder input.
	changed_query = target.clone()
	changed_query[..., :512].add_(0.75)
	encoded = geometry.encode({'object': target}, task=None)
	changed_encoded = geometry.encode({'object': changed_query}, task=None)
	if torch.equal(encoded, changed_encoded):
		raise AssertionError('Auxiliary target masking leaked into the encoder input.')
	decoded = geometry.decode_object(encoded)
	if decoded.shape != target.shape:
		raise AssertionError('Auxiliary target changed the 1770-D decoder output.')


def main():
	torch.set_num_threads(1)
	torch.manual_seed(271828)
	cfg = make_config()
	model = WorldModel(cfg).eval()

	if cfg.latent_dim != 128:
		raise AssertionError(f'Object-only latent changed to {cfg.latent_dim}.')
	if set(model._encoder) != {'object'}:
		raise AssertionError(
			f'Object-only encoder leaked another modality: {set(model._encoder)}.'
		)
	if not isinstance(model._encoder['object'], layers.CutieObjectEncoder):
		raise AssertionError('Object-only did not install CutieObjectEncoder.')
	if not isinstance(model._dynamics, layers.CutieObjectDynamics):
		raise AssertionError('Object-only did not install role-aware object dynamics.')
	for forbidden in ('_graph_dynamics', '_hybrid_reward', '_hybrid_pi', '_hybrid_q'):
		if hasattr(model, forbidden):
			raise AssertionError(f'Object-only unexpectedly installed {forbidden}.')
	state_keys = tuple(model.state_dict())
	if any(key.startswith('_encoder.rgb.') for key in state_keys):
		raise AssertionError('Object-only state dict contains an RGB encoder.')
	if any(key.startswith('_hybrid_') for key in state_keys):
		raise AssertionError('Object-only state dict contains hybrid correction heads.')

	batch = 4
	objects = torch.randn(batch, 2, 1770)
	objects[..., 2 * 590 + 588] = 1.0  # Latest per-role valid flag.
	_assert_auxiliary_target_contract(objects)
	z = model.encode({'object': objects}, task=None)
	if z.shape != (batch, 128) or not torch.isfinite(z).all():
		raise AssertionError(f'Invalid object-only latent: {tuple(z.shape)}.')
	sequence = objects.unsqueeze(0).repeat(3, 1, 1, 1)
	sequence_z, used = model.encode(
		{'object': sequence}, task=None, return_object=True
	)
	if sequence_z.shape != (3, batch, 128) or not torch.equal(used, sequence):
		raise AssertionError('Object-only sequence encoding changed the observation.')
	try:
		model.encode({}, task=None)
	except ValueError:
		pass
	else:
		raise AssertionError('Object-only accepted a missing object key.')
	# Schema enforcement lives at the wrapper and replay boundaries to keep this
	# compiled hot path free of Python key-set graph breaks. Even if a caller
	# supplies an extra RGB tensor, the structural model has no RGB module and its
	# output is bitwise unaffected.
	leaked_rgb_z = model.encode({
		'rgb': torch.randint(0, 256, (batch, 9, 64, 64), dtype=torch.uint8),
		'object': objects,
	}, task=None)
	if not torch.equal(z, leaked_rgb_z):
		raise AssertionError('An ignored RGB key changed the object-only latent.')

	changed = objects.clone()
	changed[:, 0, :512].add_(0.5)
	changed_z = model.encode({'object': changed}, task=None)
	if torch.equal(z, changed_z):
		raise AssertionError('Object encoder ignored changed object information.')

	action_a = torch.tensor([[0.25, -0.4]]).expand(batch, -1)
	action_b = torch.tensor([[-0.7, 0.6]]).expand(batch, -1)
	next_a = model.next(z, action_a, None)
	next_b = model.next(z, action_b, None)
	if next_a.shape != (batch, 128) or not torch.isfinite(next_a).all():
		raise AssertionError('Object dynamics returned an invalid latent.')
	if float((next_a - next_b).abs().max()) <= 1e-7:
		raise AssertionError('Object dynamics ignored the action.')
	decoded = model.decode_object(next_a)
	if decoded.shape != objects.shape or not torch.isfinite(decoded).all():
		raise AssertionError('Object decoder returned an invalid prediction.')

	# Reward/Q output layers start at zero in TD-MPC2. Emulate a learned model and
	# verify that the standard heads—not hybrid correction heads—use object state.
	with torch.no_grad():
		model._reward[-1].weight.normal_(std=0.03)
		model._Qs.params['2', 'weight'].normal_(std=0.03)
	reward_delta = float((
		model.reward(z, action_a, None)
		- model.reward(changed_z, action_a, None)
	).abs().max())
	q_delta = float((
		model.Q(z, action_a, None, return_type='all')
		- model.Q(changed_z, action_a, None, return_type='all')
	).abs().max())
	torch.manual_seed(141421)
	_, pi_a = model.pi(z, None)
	torch.manual_seed(141421)
	_, pi_b = model.pi(changed_z, None)
	pi_delta = float((pi_a['mean'] - pi_b['mean']).abs().max())
	if min(reward_delta, q_delta, pi_delta) <= 1e-7:
		raise AssertionError(
			'Object-only reward/Q/policy did not all use object state: '
			f'{reward_delta=}, {q_delta=}, {pi_delta=}.'
		)

	model.zero_grad(set_to_none=True)
	torch.manual_seed(173205)
	train_z = model.encode({'object': objects}, task=None)
	train_next = model.next(train_z, action_a, None)
	train_action, _ = model.pi(train_z, None)
	train_loss = (
		model.reward(train_z, action_a, None).mean()
		+ model.Q(train_z, action_a, None, return_type='all').mean()
		+ train_action.mean()
		+ model.object_loss(model.decode_object(train_next), objects)
	)
	train_loss.backward()
	for label, module in {
		'object_encoder': model._encoder['object'],
		'object_dynamics': model._dynamics,
		'object_decoder': model._object_decoder,
		'reward': model._reward,
		'Q': model._Qs,
		'policy': model._pi,
	}.items():
		_assert_finite_nonzero_grad(module, label)

	# The mode has its own strict checkpoint schema.
	reloaded = WorldModel(make_config()).eval()
	reloaded.load_state_dict(model.state_dict(), strict=True)
	if set(reloaded._encoder) != {'object'}:
		raise AssertionError('Strict reload reconstructed an invalid encoder schema.')

	print('CUTIE_OBJECT_ONLY_CONTRACT_OK', {
		'observation_keys': ['object'],
		'latent_shape': tuple(z.shape),
		'next_shape': tuple(next_a.shape),
		'decoder_shape': tuple(decoded.shape),
		'reward_object_delta': reward_delta,
		'q_object_delta': q_delta,
		'pi_object_delta': pi_delta,
		'backward_loss': float(train_loss.detach()),
		'parameter_count': model.total_params,
	})


if __name__ == '__main__':
	main()
