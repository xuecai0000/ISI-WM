"""Pure tensor helpers for the causal Cutie object belief.

The helpers in this module intentionally know nothing about environments or
policies.  They define the observation schema, exact synthetic missing-frame
encoding, three-frame stack propagation, and role-wise posterior correction
shared by training contracts and the online agent.
"""

from __future__ import annotations

import torch


NUM_ROLES = 2
FRAME_DIM = 590
STACK_FRAMES = 3
OBJECT_DIM = FRAME_DIM * STACK_FRAMES
ROLE_DIM = 64
LATENT_DIM = NUM_ROLES * ROLE_DIM
STATUS_START = 586
STATUS_DIM = 4
LATEST_VALID_INDEX = (STACK_FRAMES - 1) * FRAME_DIM + STATUS_START + 2


def validate_schema(cfg) -> None:
	"""Fail closed unless the frozen two-role Cutie schema is configured."""
	checks = {
		'num_roles': int(cfg.get('cutie_object_num_roles', NUM_ROLES)) == NUM_ROLES,
		'frame_dim': int(cfg.get('cutie_object_frame_dim', FRAME_DIM)) == FRAME_DIM,
		'stack_frames': int(cfg.get('cutie_object_stack_frames', STACK_FRAMES)) == STACK_FRAMES,
		'object_dim': int(cfg.get('cutie_object_input_dim', OBJECT_DIM)) == OBJECT_DIM,
		'role_dim': int(cfg.get('cutie_object_role_dim', ROLE_DIM)) == ROLE_DIM,
		'latent_dim': int(cfg.get('cutie_object_only_latent_dim', LATENT_DIM)) == LATENT_DIM,
	}
	if not all(checks.values()):
		raise ValueError(
			'Cutie learned belief requires the exact two-role 3x590 -> 2x64 '
			f'schema; failed={sorted(key for key, value in checks.items() if not value)}.'
		)


def latest_status(objects: torch.Tensor) -> torch.Tensor:
	"""Return ``[..., role, confidence/lost/valid/mask_score]``."""
	if tuple(objects.shape[-2:]) != (NUM_ROLES, OBJECT_DIM):
		raise ValueError(
			'Cutie objects must end in (2,1770), got '
			f'{tuple(objects.shape)}.'
		)
	frames = objects.reshape(*objects.shape[:-1], STACK_FRAMES, FRAME_DIM)
	status = frames[..., -1, STATUS_START:STATUS_START + STATUS_DIM]
	if not torch.isfinite(status).all():
		raise ValueError('Cutie object status contains non-finite values.')
	return status


def latest_valid(objects: torch.Tensor) -> torch.Tensor:
	"""Return a boolean validity tensor with one value per role."""
	valid = latest_status(objects)[..., 2]
	is_binary = (valid == 0.0) | (valid == 1.0)
	if not bool(is_binary.all().item()):
		raise ValueError('Cutie object valid status must be exactly binary {0,1}.')
	return valid.to(torch.bool)


def role_corrected_posterior(
	prior: torch.Tensor,
	measurement: torch.Tensor,
	valid: torch.Tensor,
) -> torch.Tensor:
	"""Use measurement for visible roles and the action prior when missing."""
	if prior.shape != measurement.shape or prior.shape[-1] != LATENT_DIM:
		raise ValueError(
			'Belief prior/measurement must match with final width 128, got '
			f'{tuple(prior.shape)} and {tuple(measurement.shape)}.'
		)
	if valid.shape != prior.shape[:-1] + (NUM_ROLES,):
		raise ValueError(
			'Belief valid mask must match the latent leading dimensions and two '
			f'roles, got {tuple(valid.shape)}.'
		)
	prior_roles = prior.reshape(*prior.shape[:-1], NUM_ROLES, ROLE_DIM)
	measurement_roles = measurement.reshape(
		*measurement.shape[:-1], NUM_ROLES, ROLE_DIM
	)
	posterior = torch.where(valid.unsqueeze(-1), measurement_roles, prior_roles)
	return posterior.flatten(start_dim=-2)


def mean_over_available_teacher_groups(
	losses: torch.Tensor,
	counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Average group losses without letting missing labels dilute the result.

	Each entry in ``losses`` is already normalized within one age/group. Groups
	with zero clean teacher count are unavailable, not zero-error examples, and
	must therefore be excluded from the across-group mean.
	"""
	if losses.ndim != 1 or counts.shape != losses.shape:
		raise ValueError(
			'Teacher-group losses/counts must be matching one-dimensional tensors.'
		)
	if not torch.isfinite(losses).all() or not torch.isfinite(counts).all():
		raise ValueError('Teacher-group losses/counts must be finite.')
	if bool((counts < 0).any().item()):
		raise ValueError('Teacher-group counts cannot be negative.')
	available = counts > 0
	weights = available.to(losses.dtype)
	mean = (losses * weights).sum() / weights.sum().clamp_min(1.0)
	return mean, available


def advance_reacquisition_pending(
	pending: torch.Tensor,
	previous_missing: torch.Tensor,
	current_missing: torch.Tensor,
	current_clean_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Latch a burst ending until that role first has a clean valid teacher.

	A real tracker may remain naturally invalid on the exact frame where the
	synthetic burst ends.  In that case reacquisition supervision is deferred,
	not silently discarded.  The latch clears exactly once at the first later
	clean-valid frame.
	"""
	for name, value in (
		('previous_missing', previous_missing),
		('current_missing', current_missing),
		('current_clean_valid', current_clean_valid),
	):
		if value.dtype != torch.bool or value.shape != pending.shape:
			raise ValueError(
				f'{name} must be bool with shape {tuple(pending.shape)}, got '
				f'{value.dtype} {tuple(value.shape)}.'
			)
	if pending.dtype != torch.bool:
		raise ValueError('Reacquisition pending state must be boolean.')
	pending = pending | (previous_missing & ~current_missing)
	reacquired = pending & ~current_missing & current_clean_valid
	return pending & ~reacquired, reacquired


def canonical_missing_frame_like(frame: torch.Tensor) -> torch.Tensor:
	"""Create the exact policy-burst missing encoding for a 590-D frame."""
	if frame.shape[-1] != FRAME_DIM:
		raise ValueError(f'Expected a 590-D frame, got {tuple(frame.shape)}.')
	content = torch.zeros_like(frame[..., :STATUS_START])
	status = torch.zeros_like(frame[..., STATUS_START:])
	status[..., 1] = 1.0
	return torch.cat([content, status], dim=-1)


def rebuild_stacks_with_missing_frames(
	objects: torch.Tensor,
	missing_latest: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Inject raw-frame missingness and rebuild every overlapping 3-frame stack.

	Args:
		objects: Clean replay observations with shape ``[T,B,2,1770]``.
		missing_latest: Boolean schedule ``[T,B,2]`` over the raw frame represented
			by each observation's latest stack slot. Index zero must stay unmasked so
			the sampled sequence always has a causal initial measurement.

	Returns:
		A corrupted tensor with the same shape and a boolean ``[T,B,2]`` mask
		indicating which resulting stacks contain at least one synthetic frame.
	"""
	if objects.ndim != 4 or tuple(objects.shape[-2:]) != (NUM_ROLES, OBJECT_DIM):
		raise ValueError(
			'Belief corruption expects [T,B,2,1770], got '
			f'{tuple(objects.shape)}.'
		)
	if missing_latest.dtype != torch.bool or missing_latest.shape != objects.shape[:3]:
		raise ValueError(
			'Belief missing schedule must be bool [T,B,2], got '
			f'{missing_latest.dtype} {tuple(missing_latest.shape)}.'
		)
	if bool(missing_latest[0].any().item()):
		raise ValueError('Belief corruption cannot mask the first sampled frame.')

	time, batch = objects.shape[:2]
	clean_frames = objects.reshape(
		time, batch, NUM_ROLES, STACK_FRAMES, FRAME_DIM
	)
	# obs[0] supplies the two causal prefix frames plus raw frame zero. Every
	# later raw frame is recovered from the latest slot of its stored stack.
	raw = torch.cat([
		clean_frames[0].permute(2, 0, 1, 3),
		clean_frames[1:, ..., -1, :],
	], dim=0)
	prefix_mask = torch.zeros(
		(2, batch, NUM_ROLES), dtype=torch.bool, device=objects.device
	)
	raw_mask = torch.cat([prefix_mask, missing_latest], dim=0)
	missing_frame = canonical_missing_frame_like(raw)
	corrupted_raw = torch.where(raw_mask.unsqueeze(-1), missing_frame, raw)
	corrupted_frames = torch.stack([
		corrupted_raw[offset:offset + time]
		for offset in range(STACK_FRAMES)
	], dim=-2)
	stack_affected = torch.stack([
		raw_mask[offset:offset + time]
		for offset in range(STACK_FRAMES)
	], dim=-1).any(dim=-1)
	return corrupted_frames.flatten(start_dim=-2), stack_affected
