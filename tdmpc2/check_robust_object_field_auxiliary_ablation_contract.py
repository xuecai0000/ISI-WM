"""Focused CPU contracts for the ROF-WM auxiliary-loss ablation."""

from pathlib import Path
import unittest

from tdmpc2 import (
	ROBUST_OBJECT_FIELD_AUXILIARY_FORMAT,
	robust_object_field_auxiliary_contract,
)


ROOT = Path(__file__).resolve().parent


def config(**updates):
	value = {
		'robust_object_field_enabled': True,
		'flat_anchor': True,
		'flat_anchor_mode': 'cutie_object_only',
		'cutie_object_auxiliary_target': 'full_descriptor',
		'flat_anchor_reconstruction_coef': 1.0,
		'flat_anchor_prediction_coef': 1.0,
		'robust_object_field_auxiliary_reconstruction_coef': 1.0,
		'robust_object_field_auxiliary_prediction_coef': 1.0,
	}
	value.update(updates)
	return value


class RobustObjectFieldAuxiliaryAblationContract(unittest.TestCase):
	def test_default_preserves_established_full_descriptor_weights(self):
		self.assertEqual(robust_object_field_auxiliary_contract(config()), {
			'format': ROBUST_OBJECT_FIELD_AUXILIARY_FORMAT,
			'target': 'full_descriptor',
			'query_supervision': 'enabled',
			'reconstruction_multiplier': 1.0,
			'prediction_multiplier': 1.0,
			'effective_reconstruction_coef': 1.0,
			'effective_prediction_coef': 1.0,
		})

	def test_geometry_target_disables_query_without_changing_weights(self):
		contract = robust_object_field_auxiliary_contract(config(
			cutie_object_auxiliary_target='geometry_status_full_denominator',
		))
		self.assertEqual(contract['query_supervision'], 'disabled')
		self.assertEqual(contract['effective_reconstruction_coef'], 1.0)
		self.assertEqual(contract['effective_prediction_coef'], 1.0)

	def test_independent_zero_and_fractional_multipliers(self):
		contract = robust_object_field_auxiliary_contract(config(
			flat_anchor_reconstruction_coef=0.5,
			flat_anchor_prediction_coef=0.25,
			robust_object_field_auxiliary_reconstruction_coef=0.0,
			robust_object_field_auxiliary_prediction_coef=0.4,
		))
		self.assertEqual(contract['effective_reconstruction_coef'], 0.0)
		self.assertAlmostEqual(contract['effective_prediction_coef'], 0.1)

	def test_invalid_or_silent_noop_configs_fail_closed(self):
		for name in (
			'robust_object_field_auxiliary_reconstruction_coef',
			'robust_object_field_auxiliary_prediction_coef',
			'flat_anchor_reconstruction_coef',
			'flat_anchor_prediction_coef',
		):
			for value in (-0.1, float('nan'), float('inf'), True, 'not-a-number'):
				with self.subTest(name=name, value=value), self.assertRaises(ValueError):
					robust_object_field_auxiliary_contract(config(**{name: value}))
		with self.assertRaisesRegex(ValueError, 'cutie_object_auxiliary_target'):
			robust_object_field_auxiliary_contract(config(
				cutie_object_auxiliary_target='geometry_only',
			))
		with self.assertRaisesRegex(ValueError, 'robust_object_field_enabled'):
			robust_object_field_auxiliary_contract(config(
				robust_object_field_enabled=False,
				robust_object_field_auxiliary_prediction_coef=0.0,
			))
		with self.assertRaisesRegex(ValueError, 'cutie_object_only'):
			robust_object_field_auxiliary_contract(config(
				flat_anchor_mode='residual',
			))

	def test_non_rof_default_is_ignored_for_legacy_compatibility(self):
		self.assertIsNone(robust_object_field_auxiliary_contract(config(
			robust_object_field_enabled=False,
		)))

	def test_config_loss_metrics_and_checkpoint_wiring_are_explicit(self):
		configuration = (ROOT / 'config.yaml').read_text(encoding='utf-8')
		agent = (ROOT / 'tdmpc2.py').read_text(encoding='utf-8')
		self.assertIn(
			'robust_object_field_auxiliary_reconstruction_coef: 1.0',
			configuration,
		)
		self.assertIn(
			'robust_object_field_auxiliary_prediction_coef: 1.0',
			configuration,
		)
		self.assertIn("contract['robust_object_field_auxiliary']", agent)
		self.assertIn('object_reconstruction_weighted_loss', agent)
		self.assertIn('ROF auxiliary checkpoint contract mismatch', agent)


if __name__ == '__main__':
	unittest.main(verbosity=2)
