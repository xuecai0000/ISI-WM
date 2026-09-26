"""CPU-only contracts for the RGB shuffled-pair negative control."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import rgb_interventional_pairing as pairing  # noqa: E402


def _identities():
	return {
		'root_id': np.asarray([10, 11, 12, 13, 14, 15, 16, 17], np.int64),
		'positive_code': np.asarray([0, 1, -1, 2, -2, 0, 1, -1], np.int32),
		'negative_code': np.asarray([1, -1, 0, -2, 2, -2, -1, 1], np.int32),
	}


def _schedule(mode):
	return pairing.Schedule(
		seed=6 + pairing.SEED_OFFSET, batch_size=8, mode=mode,
	)


def _assert_sha(value):
	assert isinstance(value, str) and len(value) == 64
	assert all(character in '0123456789abcdef' for character in value)


def main() -> int:
	# Configuration is fail-closed and legacy missing keys resolve to correct.
	legacy = pairing.config({}, base_arm='joint', enabled=True)
	assert legacy['mode'] == 'correct' and legacy['arm'] == 'joint'
	assert legacy['effective_seed'] == pairing.SEED_OFFSET
	assert legacy['batch_size'] == 8
	for mode, arm in pairing.MODE_TO_ARM.items():
		contract = pairing.config({
			'rgb_interventional_aux_pairing_mode': mode,
			'rgb_interventional_aux_pairing_seed_offset': pairing.SEED_OFFSET,
		}, base_arm='joint', enabled=True)
		assert contract['arm'] == arm
	try:
		pairing.config({
			'rgb_interventional_aux_pairing_mode': 'shuffle_both',
		}, base_arm='real_fork_only', enabled=True)
	except ValueError:
		pass
	else:
		raise AssertionError('Non-joint shuffled pairing was accepted.')
	try:
		pairing.config({
			'rgb_interventional_aux_pairing_mode': 'shuffle_both',
		}, base_arm='joint', enabled=False)
	except ValueError:
		pass
	else:
		raise AssertionError('Disabled shuffled pairing was accepted.')

	identities = _identities()
	correct = _schedule('correct')
	background = _schedule('shuffle_background')
	fork = _schedule('shuffle_fork')
	both = _schedule('shuffle_both')
	schedules = (correct, background, fork, both)
	first = [schedule.sample(**identities) for schedule in schedules]
	identity = np.arange(8)
	assert np.array_equal(first[0].background, identity)
	assert np.array_equal(first[0].fork, identity)
	assert not np.array_equal(first[1].background, identity)
	assert np.array_equal(first[1].fork, identity)
	assert np.array_equal(first[2].background, identity)
	assert not np.array_equal(first[2].fork, identity)
	assert not np.array_equal(first[3].background, identity)
	assert not np.array_equal(first[3].fork, identity)
	assert first[1].background_correct_pairs == 0
	assert first[2].fork_correct_pairs == 0
	assert first[3].background_correct_pairs == 0
	assert first[3].fork_correct_pairs == 0

	# All modes consume the same candidate RNG stream. Only applied indices differ.
	candidate_hashes = {schedule.metrics['candidate_sequence_sha256'] for schedule in schedules}
	assert len(candidate_hashes) == 1
	assert correct.metrics['applied_sequence_sha256'] != both.metrics['applied_sequence_sha256']
	for schedule in schedules:
		_assert_sha(schedule.metrics['candidate_sequence_sha256'])
		_assert_sha(schedule.metrics['applied_sequence_sha256'])

	# Repeated construction and repeated calls are byte-deterministic.
	left, right = _schedule('shuffle_both'), _schedule('shuffle_both')
	for _ in range(10):
		left_indices = left.sample(**identities)
		right_indices = right.sample(**identities)
		assert np.array_equal(left_indices.background, right_indices.background)
		assert np.array_equal(left_indices.fork, right_indices.fork)
	assert left.metrics == right.metrics
	contract = pairing.config({
		'seed': 6,
		'rgb_interventional_aux_batch_size': 8,
		'rgb_interventional_aux_pairing_mode': 'shuffle_both',
	}, base_arm='joint', enabled=True)
	pairing.validate_metrics(left.metrics, contract=contract, updates=10)
	corrupt = dict(left.metrics)
	corrupt['fork_correct_pairs'] = 1
	try:
		pairing.validate_metrics(corrupt, contract=contract, updates=10)
	except ValueError:
		pass
	else:
		raise AssertionError('Corrupt pairing metrics were accepted.')

	# Duplicate identities cannot accidentally recreate a correct semantic pair.
	duplicate = {
		'root_id': np.asarray([1, 1, 2, 2, 3, 3, 4, 4], np.int64),
		'positive_code': identities['positive_code'],
		'negative_code': np.asarray([1, -1, 1, -1, 1, -1, 1, -1], np.int32),
	}
	indices = _schedule('shuffle_both').sample(**duplicate)
	assert not np.any(duplicate['root_id'] == duplicate['root_id'][indices.background])
	fork_key = np.stack([duplicate['root_id'], duplicate['negative_code']], axis=1)
	assert not np.any(np.all(fork_key == fork_key[indices.fork], axis=1))

	# An impossible batch fails instead of silently retaining a correct pair.
	impossible = dict(identities)
	impossible['root_id'] = np.ones(8, np.int64)
	try:
		_schedule('shuffle_both').sample(**impossible)
	except ValueError as error:
		assert 'no correct-pair-free permutation' in str(error)
	else:
		raise AssertionError('Impossible background derangement did not fail closed.')

	print('RGB_INTERVENTIONAL_PAIRING_CHECK_OK', {
		'format': pairing.FORMAT,
		'modes': list(pairing.MODES),
		'candidate_sequence_sha256': both.metrics['candidate_sequence_sha256'],
	})
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
