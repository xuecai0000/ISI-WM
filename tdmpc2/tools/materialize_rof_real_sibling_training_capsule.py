"""Strip a validated ROF real-branch dataset into a training-only capsule.

The source dataset contains privileged arrays solely to prove that every
sibling starts from the same physical state.  Those arrays must never coexist
with controller training inputs.  This one-way materializer first revalidates
the complete source, selects whole roots from its train split, copies only
causal ROF observations and genuinely executed actions, and publishes a new
immutable manifest with no oracle namespace.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
for _path in (str(REPO_DIR), str(PROJECT_DIR)):
	while _path in sys.path:
		sys.path.remove(_path)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))

from common import rof_real_sibling_auxiliary as auxiliary
from tools import collect_rof_real_action_branches as collector


SCREEN_TASKS = (
	'finger-spin', 'cartpole-swingup', 'reacher-easy',
	'cup-catch', 'walker-walk', 'acrobot-swingup',
)


def _canonical_json_bytes(payload) -> bytes:
	return (
		json.dumps(
			payload, ensure_ascii=False, sort_keys=True,
			separators=(',', ':'), allow_nan=False,
		) + '\n'
	).encode('utf-8')


def _atomic_npz(path: Path, arrays) -> str:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f'.{path.stem}.{uuid.uuid4().hex}.tmp.npz')
	with temporary.open('wb') as handle:
		np.savez_compressed(handle, **{
			key: np.ascontiguousarray(value) for key, value in arrays.items()
		})
	os.replace(temporary, path)
	return auxiliary.file_sha256(path)


def _source_asset(manifest_path: Path, record: dict) -> Path:
	relative = record.get('relative_path')
	if not isinstance(relative, str) or not relative:
		raise ValueError('Source branch record has no relative_path.')
	path = Path(relative)
	if path.is_absolute() or '..' in path.parts:
		raise ValueError('Source branch asset path escapes its manifest root.')
	resolved = (manifest_path.parent / path).resolve()
	try:
		resolved.relative_to(manifest_path.parent.resolve())
	except ValueError as exc:
		raise ValueError('Source branch asset path escaped its root.') from exc
	if not resolved.is_file() or resolved.is_symlink():
		raise FileNotFoundError(resolved)
	return resolved


def materialize(args) -> dict:
	source_manifest = Path(args.source_manifest).resolve()
	output_root = Path(args.output_root).resolve()
	stage = output_root.with_name(output_root.name + '.incomplete')
	if output_root.exists() or stage.exists():
		raise FileExistsError(
			f'Refusing to overwrite sibling capsule output or stage: {output_root}'
		)
	if args.condition != auxiliary.CONDITION:
		raise ValueError('The first real-sibling training capsule is clean-only.')
	if args.source_split != 'train':
		raise ValueError('Only complete source train roots may enter training capsules.')
	if args.authorize_auxiliary_training != 'I_UNDERSTAND_MODEL_INPUT_ONLY':
		raise ValueError(
			'Explicit --authorize-auxiliary-training '
			'I_UNDERSTAND_MODEL_INPUT_ONLY is required.'
		)

	# This validates every source shard, its real-action design, clean/hard
	# physics twins, same-root proof, hashes, and split grouping before any
	# privileged-bearing file is opened below.
	# The legacy validator intentionally owns a two-task allow-list.  This
	# dedicated materializer expands it only inside this process for the six-task
	# preregistered screen; all tensor/provenance guards remain unchanged.
	collector.TASKS = SCREEN_TASKS
	collector.validate_dataset(source_manifest)
	source = json.loads(source_manifest.read_text(encoding='utf-8'))
	selected = [
		record for record in source['groups']
		if record.get('split') == args.source_split
	]
	if not selected:
		raise ValueError('Source branch dataset has no train roots.')

	stage.mkdir(parents=True)
	groups = []
	try:
		for record in selected:
			asset = _source_asset(source_manifest, record)
			if auxiliary.file_sha256(asset) != record.get('sha256'):
				raise ValueError(
					f'Source branch root {record.get("root_id")} changed after validation.'
				)
			with np.load(asset, allow_pickle=False) as archive:
				# The explicit names below are the privilege boundary.  Do not copy
				# arrays by prefix or archive iteration.
				arrays = {
					f'root__policy_{field}': np.ascontiguousarray(
						archive[f'{args.condition}__history__policy_{field}'][-1]
					)
					for field in auxiliary.POLICY_FIELDS
				}
				arrays.update({
					f'future__policy_{field}': np.ascontiguousarray(
						archive[f'{args.condition}__future__policy_{field}']
					)
					for field in auxiliary.POLICY_FIELDS
				})
				arrays['branch__action'] = np.ascontiguousarray(
					archive['branch__action']
				)
				arrays['branch__code'] = np.ascontiguousarray(
					archive['branch__code']
				)
			if any(key.startswith('oracle__') for key in arrays):
				raise AssertionError('Oracle array crossed the materializer boundary.')
			auxiliary.validate_shard(
				arrays,
				roles=int(source['role_count']),
				action_dim=int(source['action_dim']),
				horizon=int(source['horizon']),
				branch_magnitude=float(source['branch_magnitude']),
			)
			relative = Path('groups') / f'root_{int(record["root_id"]):04d}.npz'
			groups.append({
				'root_id': int(record['root_id']),
				'relative_path': relative.as_posix(),
				'sha256': _atomic_npz(stage / relative, arrays),
				'source_group_sha256': str(record['sha256']),
			})

		manifest = {
			'format': auxiliary.FORMAT,
			'status': auxiliary.STATUS,
			'controller_auxiliary_training_authorized': True,
			'task': source['task'],
			'condition': args.condition,
			'source_split': args.source_split,
			'model_input_only': True,
			'oracle_arrays_present': False,
			'policy_observation_unchanged': True,
			'root_grouping': 'complete_real_action_sibling_family',
			'fake_shuffled_action_futures': False,
			'role_names': list(source['role_names']),
			'num_roles': int(source['role_count']),
			'action_dim': int(source['action_dim']),
			'horizon': int(source['horizon']),
			'branch_magnitude': float(source['branch_magnitude']),
			'source_dataset': {
				'format': source['format'],
				'manifest_sha256': auxiliary.file_sha256(source_manifest),
				'controller_training_authorized': source.get(
					'controller_training_authorized'
				),
				'validation_performed_before_privilege_stripping': True,
			},
			'groups': groups,
		}
		manifest_path = stage / 'training_capsule_manifest.json'
		manifest_path.write_bytes(_canonical_json_bytes(manifest))
		auxiliary.validate_manifest(manifest_path)
		os.replace(stage, output_root)
		return manifest
	except Exception:
		# Preserve the incomplete tree for forensic inspection; never publish it.
		raise


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--source-manifest', type=Path, required=True)
	parser.add_argument('--output-root', type=Path, required=True)
	parser.add_argument('--source-split', default='train', choices=('train',))
	parser.add_argument('--condition', default='clean', choices=('clean',))
	parser.add_argument('--authorize-auxiliary-training', required=True)
	return parser.parse_args(argv)


def main(argv=None):
	args = parse_args(argv)
	manifest = materialize(args)
	path = (Path(args.output_root).resolve() / 'training_capsule_manifest.json')
	print('ROF_REAL_SIBLING_TRAINING_CAPSULE_COMPLETE', json.dumps({
		'task': manifest['task'],
		'roots': len(manifest['groups']),
		'manifest': str(path),
		'manifest_sha256': auxiliary.file_sha256(path),
	}, allow_nan=False), flush=True)
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
