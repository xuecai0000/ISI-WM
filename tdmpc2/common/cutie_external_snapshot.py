"""Immutable external-input snapshot for Cutie perception diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re


FORMAT = 'cutie_native_support_external_inputs_v2'
COMPONENTS = (
	'local_python', 'video_tree', 'manifest_tree', 'external_cutie_python',
	'external_cutie_config',
)
_SHA256 = re.compile(r'[0-9a-f]{64}')


def _canonical_bytes(value) -> bytes:
	return (
		json.dumps(
			value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
		) + '\n'
	).encode('utf-8')


def _sha256_bytes(value: bytes) -> str:
	return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as handle:
		for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def _require_sha(value, label: str) -> str:
	if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
		raise ValueError(f'{label} must be one lowercase SHA-256 digest.')
	return value


def _tree_files(root: Path, *, python_only: bool = False) -> list[dict]:
	root = root.expanduser().resolve()
	if not root.is_dir():
		raise FileNotFoundError(root)
	paths = sorted(
		path for path in root.rglob('*')
		if path.is_file() and (not python_only or path.suffix == '.py')
	)
	if not paths:
		raise ValueError(f'External input tree is empty: {root}')
	return [
		{
			'path': path.relative_to(root).as_posix(),
			'bytes': path.stat().st_size,
			'sha256': _file_sha256(path),
		}
		for path in paths
	]


def _component(root: Path, *, python_only: bool = False) -> dict:
	root = root.expanduser().resolve()
	files = _tree_files(root, python_only=python_only)
	return {
		'root': str(root),
		'python_only': bool(python_only),
		'files': files,
		'tree_sha256': _sha256_bytes(_canonical_bytes(files)),
	}


def build_snapshot(
	*, repo_root: Path, video_root: Path, manifest_dir: Path, oc_repo: Path,
	cutie_checkpoint: Path,
) -> dict:
	repo_root = repo_root.expanduser().resolve()
	video_root = video_root.expanduser().resolve()
	manifest_dir = manifest_dir.expanduser().resolve()
	oc_repo = oc_repo.expanduser().resolve()
	cutie_checkpoint = cutie_checkpoint.expanduser().resolve()
	if not (repo_root / 'tdmpc2').is_dir():
		raise FileNotFoundError(repo_root / 'tdmpc2')
	if video_root.name != 'video_hard' or not video_root.is_dir():
		raise ValueError('video_root must be the frozen video_hard directory.')
	if not manifest_dir.is_dir() or not oc_repo.is_dir():
		raise FileNotFoundError('Manifest or OC-STORM root is missing.')
	if not cutie_checkpoint.is_file():
		raise FileNotFoundError(cutie_checkpoint)
	cutie_root = oc_repo / 'feature_extractor' / 'cutie' / 'cutie'
	config_root = cutie_root / 'config'
	components = {
		'local_python': _component(repo_root / 'tdmpc2', python_only=True),
		'video_tree': _component(video_root),
		'manifest_tree': _component(manifest_dir),
		'external_cutie_python': _component(cutie_root, python_only=True),
		'external_cutie_config': _component(config_root),
	}
	payload = {
		'format': FORMAT,
		'paths': {
			'repo_root': str(repo_root),
			'video_root': str(video_root),
			'manifest_dir': str(manifest_dir),
			'oc_repo': str(oc_repo),
			'cutie_checkpoint': str(cutie_checkpoint),
		},
		'components': components,
		'cutie_checkpoint': {
			'bytes': cutie_checkpoint.stat().st_size,
			'sha256': _file_sha256(cutie_checkpoint),
		},
	}
	payload['snapshot_id'] = _sha256_bytes(_canonical_bytes(payload))
	return payload


def validate_snapshot(payload: dict) -> dict:
	if not isinstance(payload, dict) or set(payload) != {
		'format', 'paths', 'components', 'cutie_checkpoint', 'snapshot_id'
	}:
		raise ValueError('External input snapshot schema is invalid.')
	if payload.get('format') != FORMAT:
		raise ValueError('External input snapshot format changed.')
	paths = payload.get('paths')
	components = payload.get('components')
	checkpoint = payload.get('cutie_checkpoint')
	if not isinstance(paths, dict) or set(paths) != {
		'repo_root', 'video_root', 'manifest_dir', 'oc_repo',
		'cutie_checkpoint'
	}:
		raise ValueError('External input path schema is invalid.')
	if not isinstance(components, dict) or tuple(components) != COMPONENTS:
		raise ValueError('External input component order/set changed.')
	if not isinstance(checkpoint, dict) or set(checkpoint) != {'bytes', 'sha256'}:
		raise ValueError('Cutie checkpoint snapshot is malformed.')
	stored_id = _require_sha(payload.get('snapshot_id'), 'snapshot_id')
	identity = {key: value for key, value in payload.items() if key != 'snapshot_id'}
	if _sha256_bytes(_canonical_bytes(identity)) != stored_id:
		raise ValueError('External input snapshot identity mismatch.')
	repo_root = Path(paths['repo_root']).expanduser().resolve()
	video_root = Path(paths['video_root']).expanduser().resolve()
	manifest_dir = Path(paths['manifest_dir']).expanduser().resolve()
	oc_repo = Path(paths['oc_repo']).expanduser().resolve()
	cutie_path = Path(paths['cutie_checkpoint']).expanduser().resolve()
	expected_roots = {
		'local_python': repo_root / 'tdmpc2',
		'video_tree': video_root,
		'manifest_tree': manifest_dir,
		'external_cutie_python': (
			oc_repo / 'feature_extractor' / 'cutie' / 'cutie'
		),
		'external_cutie_config': (
			oc_repo / 'feature_extractor' / 'cutie' / 'cutie' / 'config'
		),
	}
	if video_root.name != 'video_hard':
		raise ValueError('Snapshot video root is not video_hard.')
	for name in COMPONENTS:
		entry = components[name]
		if not isinstance(entry, dict) or set(entry) != {
			'root', 'python_only', 'files', 'tree_sha256'
		}:
			raise ValueError(f'External component {name} is malformed.')
		if Path(entry['root']).expanduser().resolve() != expected_roots[name].resolve():
			raise ValueError(f'External component {name} root changed.')
		python_only = name in {'local_python', 'external_cutie_python'}
		if entry['python_only'] is not python_only:
			raise ValueError(f'External component {name} filter changed.')
		files = _tree_files(expected_roots[name], python_only=python_only)
		if files != entry['files']:
			raise ValueError(f'External component {name} changed during the run.')
		tree_sha = _sha256_bytes(_canonical_bytes(files))
		if tree_sha != _require_sha(entry['tree_sha256'], f'{name}.tree_sha256'):
			raise ValueError(f'External component {name} tree hash mismatch.')
	if not cutie_path.is_file():
		raise FileNotFoundError(cutie_path)
	if (
		type(checkpoint.get('bytes')) is not int
		or checkpoint['bytes'] != cutie_path.stat().st_size
		or _require_sha(checkpoint.get('sha256'), 'cutie_checkpoint.sha256')
		!= _file_sha256(cutie_path)
	):
		raise ValueError('Cutie checkpoint changed during the run.')
	return {
		'format': FORMAT,
		'snapshot_id': stored_id,
		'paths': {key: str(Path(value).expanduser().resolve()) for key, value in paths.items()},
		'components': {
			name: {
				'files': len(components[name]['files']),
				'tree_sha256': components[name]['tree_sha256'],
			}
			for name in COMPONENTS
		},
		'cutie_checkpoint': dict(checkpoint),
	}


def _atomic_write(path: Path, payload: dict) -> None:
	path = path.expanduser().resolve()
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(path.name + f'.incomplete.{os.getpid()}')
	with temporary.open('x', encoding='utf-8', newline='\n') as handle:
		json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
		handle.write('\n')
	os.replace(temporary, path)


def main(argv=None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--repo-root', type=Path, required=True)
	parser.add_argument('--video-root', type=Path, required=True)
	parser.add_argument('--manifest-dir', type=Path, required=True)
	parser.add_argument('--oc-repo', type=Path, required=True)
	parser.add_argument('--cutie-checkpoint', type=Path, required=True)
	parser.add_argument('--output', type=Path, required=True)
	args = parser.parse_args(argv)
	if args.output.exists():
		raise FileExistsError(args.output)
	payload = build_snapshot(
		repo_root=args.repo_root,
		video_root=args.video_root,
		manifest_dir=args.manifest_dir,
		oc_repo=args.oc_repo,
		cutie_checkpoint=args.cutie_checkpoint,
	)
	validate_snapshot(payload)
	_atomic_write(args.output, payload)
	print('CUTIE_EXTERNAL_INPUT_SNAPSHOT_OK', json.dumps({
		'snapshot_id': payload['snapshot_id'],
		'output': str(args.output.expanduser().resolve()),
	}, sort_keys=True), flush=True)
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
