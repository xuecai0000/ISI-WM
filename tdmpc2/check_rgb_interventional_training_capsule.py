"""CPU-only contracts for the strict RGB interventional materializer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from tools import materialize_rgb_interventional_training_capsule as materializer


def _source_arrays(root_id: int, *, split: str):
	action_dim, horizon, state_dim = 2, 2, 3
	branches = 1 + 2 * action_dim
	actions = np.zeros((branches, horizon, action_dim), dtype=np.float32)
	actions[:, 1] = np.asarray([0.125, -0.25], dtype=np.float32)
	actions[1, 0, 0] = 0.8
	actions[2, 0, 0] = -0.8
	actions[3, 0, 1] = 0.8
	actions[4, 0, 1] = -0.8
	result = {
		'branch__action': actions,
		'branch__code': materializer.branch_codes(action_dim),
		# These source-only arrays prove that the explicit output allow-list is
		# actually stripping object and raw privileged namespaces.
		'clean__history__policy_object': np.full((3, 2), root_id, np.float32),
		'oracle__physics_state': np.full((1,), root_id, np.float64),
	}
	for condition_index, condition in enumerate(('clean', 'hard')):
		history = np.empty((3,) + materializer.RGB_SHAPE, dtype=np.uint8)
		for index in range(3):
			history[index].fill(root_id * 20 + condition_index * 7 + index)
		future = np.empty(
			(branches, horizon) + materializer.RGB_SHAPE, dtype=np.uint8,
		)
		for branch in range(branches):
			for step in range(horizon):
				future[branch, step].fill(
					root_id * 20 + condition_index * 7 + branch + step
				)
		result[f'{condition}__history__policy_rgb'] = history
		result[f'{condition}__future__policy_rgb'] = future

	# Train roots are deliberately moderate.  Non-train roots are extreme so a
	# scale fitted from validation/test data would visibly change the target.
	official = np.zeros((branches, horizon + 1, state_dim), dtype=np.float64)
	official[:, 0] = np.asarray([root_id, -root_id, 1.0])
	multiplier = 1.0 + root_id if split == 'train' else 10000.0
	for branch in range(branches):
		for step in range(1, horizon + 1):
			official[branch, step] = np.asarray((
				multiplier * branch * step,
				multiplier * (branch - 2) * (step + 1),
				1.0,
			))
	result['oracle__official_state'] = official
	return result


def _source_dataset(root: Path) -> tuple[Path, dict[int, dict[str, np.ndarray]]]:
	(root / 'groups').mkdir(parents=True)
	groups = []
	arrays_by_root = {}
	splits = ('train', 'train', 'validation', 'test')
	for root_id, split in enumerate(splits):
		arrays = _source_arrays(root_id, split=split)
		arrays_by_root[root_id] = arrays
		path = root / 'groups' / f'root_{root_id:04d}.npz'
		with path.open('wb') as handle:
			np.savez_compressed(handle, **arrays)
		groups.append({
			'root_id': root_id,
			'split': split,
			'relative_path': path.relative_to(root).as_posix(),
			'sha256': materializer.file_sha256(path),
		})
	payload = {
		'format': materializer.SOURCE_FORMAT,
		'controller_training_authorized': False,
		'task': 'reacher-easy',
		'action_dim': 2,
		'state_dim': 3,
		'horizon': 2,
		'branch_magnitude': 0.8,
		'root_splits': {
			'train': [0, 1], 'validation': [2], 'test': [3],
		},
		'groups': groups,
	}
	manifest = root / 'dataset_manifest.json'
	manifest.write_text(
		json.dumps(payload, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	return manifest, arrays_by_root


def _assert_raises(exception, fn, *args, **kwargs):
	try:
		fn(*args, **kwargs)
	except exception:
		return
	raise AssertionError(f'{fn.__name__} did not raise {exception.__name__}.')


def main():
	with tempfile.TemporaryDirectory(prefix='rgb-interventional-capsule-') as temp:
		root = Path(temp)
		source_manifest, source_arrays = _source_dataset(root / 'source')
		output = root / 'capsule'
		validation_calls = []
		original_validator = materializer.collector.validate_dataset

		def strict_validator(path):
			validation_calls.append(Path(path).resolve())
			return json.loads(Path(path).read_text(encoding='utf-8'))

		materializer.collector.validate_dataset = strict_validator
		try:
			args = argparse.Namespace(
				source_manifest=source_manifest,
				output_root=output,
				source_split='train',
				authorize_auxiliary_training=(
					'I_UNDERSTAND_RGB_ONLY_WITH_DERIVED_TARGETS'
				),
			)
			payload = materializer.materialize(args)
		finally:
			materializer.collector.validate_dataset = original_validator

		assert validation_calls == [source_manifest.resolve()]
		assert output.is_dir()
		assert not output.with_name(output.name + '.incomplete').exists()
		assert payload['source_root_ids'] == [0, 1]
		assert [record['root_id'] for record in payload['groups']] == [0, 1]
		manifest = output / materializer.MANIFEST_NAME
		validated = materializer.validate_capsule(manifest)
		assert validated['test_time_input'] == 'rgb_only'
		assert validated['no_cutie'] is True
		assert validated['raw_privileged_arrays_present'] is False

		# Independently recompute the train-only scale and target.  The extreme
		# validation/test outcomes must not influence either value.
		expected_moments = materializer._OnlineCoordinateMoments()
		for root_id in (0, 1):
			expected_moments.update(
				source_arrays[root_id]['oracle__official_state'][:, 1:, :]
				.reshape(-1, 3)
			)
		expected_scale = expected_moments.scale(
			epsilon=materializer.SCALE_EPSILON
		)
		expected_gap, expected_eligible = materializer.derive_pair_targets(
			source_arrays[0]['oracle__official_state'], expected_scale,
		)
		first_path = output / payload['groups'][0]['relative_path']
		with np.load(first_path, allow_pickle=False) as archive:
			assert set(archive.files) == materializer.SHARD_KEYS
			assert not any(
				fragment in key.lower()
				for key in archive.files
				for fragment in materializer.FORBIDDEN_KEY_FRAGMENTS
			)
			assert np.array_equal(
				archive['clean__root_rgb'],
				source_arrays[0]['clean__history__policy_rgb'][-1],
			)
			assert np.array_equal(
				archive['hard__root_rgb'],
				source_arrays[0]['hard__history__policy_rgb'][-1],
			)
			assert np.array_equal(archive['pair__outcome_gap'], expected_gap)
			assert np.array_equal(archive['pair__eligible'], expected_eligible)
			assert not archive['pair__eligible'][0, 0].any()
			assert np.array_equal(
				archive['pair__eligible'],
				np.swapaxes(archive['pair__eligible'], 0, 1),
			)
		expected_scale_sha = materializer.typed_array_sha256(
			'official_state_coordinate_scale', expected_scale,
		)
		assert (
			payload['outcome_target_contract']['coordinate_scale_sha256']
			== expected_scale_sha
		)
		assert payload['outcome_target_contract']['coordinate_count'] == 20

		# Publication is immutable and cannot silently replace a completed run.
		materializer.collector.validate_dataset = strict_validator
		try:
			_assert_raises(FileExistsError, materializer.materialize, args)
		finally:
			materializer.collector.validate_dataset = original_validator

		# The output validator fails closed if a raw privileged array is injected,
		# even when an attacker also updates the file and manifest hashes.
		manifest_payload = json.loads(manifest.read_text(encoding='utf-8'))
		with np.load(first_path, allow_pickle=False) as archive:
			tampered = {key: archive[key] for key in archive.files}
		tampered['oracle__official_state'] = np.zeros((1,), dtype=np.float64)
		with first_path.open('wb') as handle:
			np.savez_compressed(handle, **tampered)
		manifest_payload['groups'][0]['sha256'] = materializer.file_sha256(first_path)
		manifest.write_text(
			json.dumps(manifest_payload, sort_keys=True, allow_nan=False) + '\n',
			encoding='utf-8',
		)
		_assert_raises(ValueError, materializer.validate_capsule, manifest)

	# Eligibility is not an action-identity label: identical realized outcomes
	# stay ineligible even when their executed actions differ.
	official = np.zeros((3, 3, 2), dtype=np.float64)
	gap, eligible = materializer.derive_pair_targets(
		official, np.ones(2, dtype=np.float64),
	)
	assert not gap.any() and not eligible.any()
	print('RGB_INTERVENTIONAL_TRAINING_CAPSULE_CONTRACT_PASS')


if __name__ == '__main__':
	main()
