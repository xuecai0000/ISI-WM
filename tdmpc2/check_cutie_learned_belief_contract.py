"""Dependency-light contracts for burst-trained Cutie object belief semantics."""

import torch

from tdmpc2.common import cutie_object_belief as belief


def fail(message):
	raise AssertionError(message)


def check_schema():
	cfg = {
		'cutie_object_num_roles': 2,
		'cutie_object_frame_dim': 590,
		'cutie_object_stack_frames': 3,
		'cutie_object_input_dim': 1770,
		'cutie_object_role_dim': 64,
		'cutie_object_only_latent_dim': 128,
	}
	belief.validate_schema(cfg)
	for key, bad in (
		('cutie_object_num_roles', 3),
		('cutie_object_input_dim', 1769),
		('cutie_object_role_dim', 63),
	):
		broken = dict(cfg)
		broken[key] = bad
		try:
			belief.validate_schema(broken)
		except ValueError:
			pass
		else:
			fail(f'schema accepted invalid {key}={bad}')


def check_role_correction():
	prior = torch.arange(2 * 128, dtype=torch.float32).reshape(2, 128)
	measurement = prior + 1000.0
	valid = torch.tensor([[True, False], [False, True]])
	posterior = belief.role_corrected_posterior(prior, measurement, valid)
	roles = posterior.reshape(2, 2, 64)
	torch.testing.assert_close(roles[0, 0], measurement.reshape(2, 2, 64)[0, 0])
	torch.testing.assert_close(roles[0, 1], prior.reshape(2, 2, 64)[0, 1])
	torch.testing.assert_close(roles[1, 0], prior.reshape(2, 2, 64)[1, 0])
	torch.testing.assert_close(roles[1, 1], measurement.reshape(2, 2, 64)[1, 1])


def coherent_objects(time=7, batch=1):
	raw = torch.zeros(time + 2, batch, 2, 590)
	for t in range(time + 2):
		for role in range(2):
			raw[t, :, role, :586] = 1000 * role + t
			raw[t, :, role, 586:590] = torch.tensor([0.9, 0.0, 1.0, 0.8])
	frames = torch.stack([raw[i:i + time] for i in range(3)], dim=-2)
	return raw, frames.flatten(start_dim=-2)


def check_stack_corruption():
	raw, objects = coherent_objects()
	clean = objects.clone()
	mask = torch.zeros(7, 1, 2, dtype=torch.bool)
	mask[2, 0, 0] = True
	corrupted, affected = belief.rebuild_stacks_with_missing_frames(objects, mask)
	if not torch.equal(objects, clean):
		fail('clean teacher was mutated by stack corruption')
	frames = corrupted.reshape(7, 1, 2, 3, 590)
	missing = belief.canonical_missing_frame_like(raw[0, 0, 0])
	for obs_index, slot in ((2, 2), (3, 1), (4, 0)):
		if not torch.equal(frames[obs_index, 0, 0, slot], missing):
			fail(f'missing raw frame did not propagate to obs={obs_index} slot={slot}')
	if not torch.equal(frames[:, :, 1], clean.reshape(7, 1, 2, 3, 590)[:, :, 1]):
		fail('non-target role changed during corruption')
	expected_affected = torch.zeros_like(affected)
	expected_affected[2:5, :, 0] = True
	if not torch.equal(affected, expected_affected):
		fail('affected-stack mask does not match exact three-frame propagation')
	zero_mask = torch.zeros_like(mask)
	rebuilt, rebuilt_affected = belief.rebuild_stacks_with_missing_frames(
		objects, zero_mask
	)
	if not torch.equal(rebuilt, objects) or rebuilt_affected.any():
		fail('zero corruption schedule must reconstruct the clean stack bitwise')


class TinyTransition(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.delta = torch.nn.Parameter(torch.tensor(0.01))

	def forward(self, state, action):
		return state + self.delta * action[:, :1].expand_as(state)


def check_age20_gradient_and_reacquisition():
	transition = TinyTransition()
	batch, time = 3, 25
	measurement = torch.zeros(time, batch, 128)
	action = torch.ones(time - 1, batch, 2)
	valid = torch.ones(time, batch, 2, dtype=torch.bool)
	valid[3:23, :, 0] = False
	belief_state = measurement[0]
	age20_loss = None
	for t in range(1, time):
		prior = transition(belief_state, action[t - 1])
		belief_state = belief.role_corrected_posterior(
			prior, measurement[t], valid[t]
		)
		if t == 22:
			age20_loss = belief_state.reshape(batch, 2, 64)[:, 0].square().mean()
	if age20_loss is None or not torch.isfinite(age20_loss):
		fail('age-20 loss is unavailable or non-finite')
	age20_loss.backward()
	if transition.delta.grad is None or float(transition.delta.grad.abs()) <= 0:
		fail('age-20 loss did not backpropagate through the transition')
	# The first valid observation after the burst must erase accumulated role-0
	# drift exactly, while the always-valid second role stayed measured throughout.
	if not torch.equal(belief_state, measurement[-1]):
		fail('reacquisition did not correct the belief exactly')


def check_initial_status_and_missing_encoding():
	frame = torch.randn(4, 590)
	missing = belief.canonical_missing_frame_like(frame)
	if not torch.equal(missing[:, :586], torch.zeros_like(missing[:, :586])):
		fail('missing content is not zero')
	expected = torch.tensor([0.0, 1.0, 0.0, 0.0]).expand(4, 4)
	if not torch.equal(missing[:, 586:590], expected):
		fail('missing status is not [0,1,0,0]')
	_, objects = coherent_objects(time=2, batch=2)
	status = belief.latest_status(objects)
	if status.shape != (2, 2, 2, 4) or not belief.latest_valid(objects).all():
		fail('latest status/valid extraction used the wrong stack slot')
	broken = objects.clone()
	broken[0, 0, 0, -4 + 2] = 0.6
	try:
		belief.latest_valid(broken)
	except ValueError:
		pass
	else:
		fail('non-binary valid status was silently accepted')


def check_available_teacher_group_mean():
	losses = torch.tensor([2.0, 100.0, 4.0], requires_grad=True)
	counts = torch.tensor([3.0, 0.0, 1.0])
	mean, available = belief.mean_over_available_teacher_groups(losses, counts)
	if not torch.equal(available, torch.tensor([True, False, True])):
		fail('teacher availability mask is wrong')
	torch.testing.assert_close(mean, torch.tensor(3.0))
	mean.backward()
	torch.testing.assert_close(
		losses.grad, torch.tensor([0.5, 0.0, 0.5])
	)
	zero_mean, zero_available = belief.mean_over_available_teacher_groups(
		torch.tensor([7.0, 9.0]), torch.zeros(2)
	)
	if zero_available.any() or float(zero_mean) != 0.0:
		fail('all-unavailable teacher groups must return a zero masked mean')


def check_delayed_reacquisition_latch():
	pending = torch.zeros(2, 2, dtype=torch.bool)
	missing = torch.zeros_like(pending)
	missing[:, 0] = True
	# Synthetic burst ends, but the real tracker is still invalid.
	clean_valid = torch.tensor([[False, True], [False, True]])
	pending, reacquired = belief.advance_reacquisition_pending(
		pending, missing, torch.zeros_like(missing), clean_valid
	)
	if reacquired.any() or not pending[:, 0].all():
		fail('reacquisition latch did not retain a naturally invalid burst ending')
	# A second naturally invalid frame must not clear or duplicate the event.
	pending, reacquired = belief.advance_reacquisition_pending(
		pending, torch.zeros_like(missing), torch.zeros_like(missing), clean_valid
	)
	if reacquired.any() or not pending[:, 0].all():
		fail('pending reacquisition was lost before a clean-valid teacher')
	# The first clean-valid frame fires exactly once and clears the latch.
	clean_valid[:, 0] = True
	pending, reacquired = belief.advance_reacquisition_pending(
		pending, torch.zeros_like(missing), torch.zeros_like(missing), clean_valid
	)
	if not reacquired[:, 0].all() or pending.any():
		fail('delayed clean-valid frame did not fire exactly one reacquisition')
	pending, reacquired = belief.advance_reacquisition_pending(
		pending, torch.zeros_like(missing), torch.zeros_like(missing), clean_valid
	)
	if reacquired.any() or pending.any():
		fail('reacquisition latch fired more than once')


def main():
	tests = (
		('schema', check_schema),
		('role_correction', check_role_correction),
		('stack_corruption', check_stack_corruption),
		('age20_gradient_and_reacquisition', check_age20_gradient_and_reacquisition),
		('status_and_missing_encoding', check_initial_status_and_missing_encoding),
		('available_teacher_group_mean', check_available_teacher_group_mean),
		('delayed_reacquisition_latch', check_delayed_reacquisition_latch),
	)
	for name, test in tests:
		test()
		print('PASS', name)
	print(f'CUTIE_LEARNED_BELIEF_CONTRACT_OK {len(tests)}/{len(tests)}')


if __name__ == '__main__':
	main()
