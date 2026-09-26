"""Pure contract helpers for Cutie object auxiliary targets.

This module deliberately has no Torch, Hydra, Cutie, or simulator dependency so
checkpoint readers and source-level contracts can validate the scientific arm
without constructing a model.
"""


FORMAT = 'cutie_object_auxiliary_contract_v1'
FULL_DESCRIPTOR = 'full_descriptor'
GEOMETRY_STATUS_FULL_DENOMINATOR = 'geometry_status_full_denominator'
LEGACY_GEOMETRY_STATUS_ACTIVE_DENOMINATOR = (
	'geometry_status_active_denominator_legacy'
)
TARGETS = frozenset({FULL_DESCRIPTOR, GEOMETRY_STATUS_FULL_DENOMINATOR})
OBSERVATION_VARIANTS = frozenset({
	'full', 'cutie_mask_geometry', 'gt_mask_geometry',
})


def contract(observation_variant='full', target=FULL_DESCRIPTOR):
	"""Return the exact auxiliary-loss contract for a Cutie observation.

	``full_descriptor`` preserves the established behavior bit for bit. The two
	geometry-only observation variants historically excluded their reserved zero
	query block and averaged over the 234 active values per role; that legacy
	behavior remains the default for those diagnostic inputs.

	``geometry_status_full_denominator`` is the new Full-Cutie ablation. It masks
	the 512 query values in each frame only after elementwise loss evaluation and
	still averages over all 1770 descriptor positions. Consequently every
	geometry/status element has exactly the same gradient scale as the established
	Full-Cutie objective, while query gradients are exactly zero.
	"""
	observation_variant = str(observation_variant)
	target = str(target)
	if observation_variant not in OBSERVATION_VARIANTS:
		raise ValueError(
			f'Unknown cutie_object_observation_variant {observation_variant!r}.'
		)
	if target not in TARGETS:
		raise ValueError(
			f'Unknown cutie_object_auxiliary_target {target!r}; '
			f'expected one of {sorted(TARGETS)}.'
		)
	if (
		target == GEOMETRY_STATUS_FULL_DENOMINATOR
		and observation_variant != 'full'
	):
		raise ValueError(
			'geometry_status_full_denominator requires the unchanged Full-Cutie '
			'observation_variant="full" input.'
		)

	if target == GEOMETRY_STATUS_FULL_DENOMINATOR:
		effective_target = target
		normalization = 'full_descriptor'
		supervised_values_per_frame = 78
		loss_denominator_values_per_role = 1770
	elif observation_variant == 'full':
		effective_target = FULL_DESCRIPTOR
		normalization = 'full_descriptor'
		supervised_values_per_frame = 590
		loss_denominator_values_per_role = 1770
	else:
		# Frozen compatibility for the pre-existing geometry-input diagnostics.
		effective_target = LEGACY_GEOMETRY_STATUS_ACTIVE_DENOMINATOR
		normalization = 'geometry_status_descriptor'
		supervised_values_per_frame = 78
		loss_denominator_values_per_role = 3 * 78

	return {
		'format': FORMAT,
		'target': target,
		'effective_target': effective_target,
		'normalization': normalization,
		'applies_to': ['current_reconstruction', 'future_prediction'],
		'decoder_output_dim': 1770,
		'query_values_per_frame': 512,
		'geometry_status_values_per_frame': 78,
		'supervised_values_per_frame': supervised_values_per_frame,
		'loss_denominator_values_per_role': loss_denominator_values_per_role,
	}


def legacy_contract(observation_contract=None):
	"""Interpret a checkpoint that predates explicit auxiliary metadata."""
	variant = (
		observation_contract.get('variant', 'full')
		if isinstance(observation_contract, dict) else 'full'
	)
	return contract(variant, FULL_DESCRIPTOR)
