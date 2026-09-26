"""Dependency-free contract for the Full-Cutie auxiliary-target ablation."""

from pathlib import Path
import unittest

from common import cutie_object_auxiliary


ROOT = Path(__file__).resolve().parent


class CutieObjectAuxiliaryTargetContract(unittest.TestCase):
	def test_default_full_descriptor_is_frozen(self):
		contract = cutie_object_auxiliary.contract('full')
		self.assertEqual(contract, {
			'format': 'cutie_object_auxiliary_contract_v1',
			'target': 'full_descriptor',
			'effective_target': 'full_descriptor',
			'normalization': 'full_descriptor',
			'applies_to': ['current_reconstruction', 'future_prediction'],
			'decoder_output_dim': 1770,
			'query_values_per_frame': 512,
			'geometry_status_values_per_frame': 78,
			'supervised_values_per_frame': 590,
			'loss_denominator_values_per_role': 1770,
		})
		self.assertEqual(
			cutie_object_auxiliary.legacy_contract(),
			contract,
		)

	def test_geometry_status_keeps_full_denominator(self):
		contract = cutie_object_auxiliary.contract(
			'full', 'geometry_status_full_denominator'
		)
		self.assertEqual(contract['effective_target'], contract['target'])
		self.assertEqual(contract['normalization'], 'full_descriptor')
		self.assertEqual(contract['supervised_values_per_frame'], 78)
		self.assertEqual(contract['loss_denominator_values_per_role'], 1770)
		self.assertEqual(contract['decoder_output_dim'], 1770)

	def test_old_geometry_observation_default_is_preserved(self):
		for variant in ('cutie_mask_geometry', 'gt_mask_geometry'):
			contract = cutie_object_auxiliary.legacy_contract({'variant': variant})
			self.assertEqual(contract['target'], 'full_descriptor')
			self.assertEqual(
				contract['effective_target'],
				'geometry_status_active_denominator_legacy',
			)
			self.assertEqual(contract['loss_denominator_values_per_role'], 234)

	def test_unknown_target_and_observation_fail_closed(self):
		with self.assertRaisesRegex(ValueError, 'auxiliary_target'):
			cutie_object_auxiliary.contract('full', 'geometry_status')
		with self.assertRaisesRegex(ValueError, 'observation_variant'):
			cutie_object_auxiliary.contract('unknown', 'full_descriptor')
		with self.assertRaisesRegex(ValueError, 'Full-Cutie'):
			cutie_object_auxiliary.contract(
				'gt_mask_geometry', 'geometry_status_full_denominator'
			)

	def test_config_and_checkpoint_wiring_are_explicit(self):
		config = (ROOT / 'config.yaml').read_text(encoding='utf-8')
		world = (ROOT / 'common' / 'world_model.py').read_text(encoding='utf-8')
		agent = (ROOT / 'tdmpc2.py').read_text(encoding='utf-8')
		self.assertIn(
			'cutie_object_auxiliary_target: full_descriptor', config
		)
		self.assertIn('feature_mask[512:] = 1.', world)
		self.assertIn("contract['cutie_object_auxiliary']", agent)
		self.assertIn('Cutie object auxiliary contract mismatch', agent)


if __name__ == '__main__':
	unittest.main(verbosity=2)
