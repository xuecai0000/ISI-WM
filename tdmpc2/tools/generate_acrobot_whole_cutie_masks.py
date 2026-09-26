"""Generate causal whole-Acrobot Cutie masks for an immutable RGB dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))

from tdmpc2.perception.acrobot_keypoint_data import load_manifest  # noqa: E402
from tdmpc2.perception.acrobot_masked_keypoint_data import MASK_DATASET_FORMAT  # noqa: E402
from tdmpc2.perception.cutie_oc_adapter import (  # noqa: E402
	CutieOCAdapter, CutieOCConfig, CutieSupportPrompts,
)
from tdmpc2.perception.support_conditioned_object_graph import (  # noqa: E402
	load_object_graph, project_support_to_entities,
)


def _sha256(path):
	digest = hashlib.sha256()
	with Path(path).open('rb') as stream:
		for chunk in iter(lambda: stream.read(1024 * 1024), b''):
			digest.update(chunk)
	return digest.hexdigest()


def _support(args):
	graph = load_object_graph(args.graph)
	if graph.task != 'acrobot-swingup' or graph.entity_names != ('whole_acrobot',):
		raise ValueError('Graph must compile Acrobot to exactly one whole_acrobot entity.')
	with np.load(args.support_npz, allow_pickle=False) as archive:
		if set(archive.files) != {'rgb', 'indexed_masks'}:
			raise ValueError('Support NPZ must contain exactly rgb and indexed_masks.')
		entity = project_support_to_entities(
			archive['rgb'], archive['indexed_masks'], graph,
		)
	prompts = CutieSupportPrompts(
		frames=tuple(np.array(value, copy=True) for value in entity.rgb),
		masks=tuple(np.array(value, copy=True) for value in entity.indexed_masks),
		role_names=graph.entity_names,
		annotation_path=args.support_npz.resolve(),
		metadata={
			**entity.metadata,
			'task': graph.task,
			'source_support_arrays_sha256': _sha256(args.support_npz),
		},
	)
	return graph, prompts


def generate(args):
	manifest_path, manifest = load_manifest(args.input_manifest)
	root = args.output.resolve()
	if root.exists():
		raise FileExistsError(f'Refusing to overwrite mask dataset: {root}')
	root.mkdir(parents=True)
	graph, support = _support(args)
	support_size = tuple(int(value) for value in support.frames[0].shape[:2])
	config = CutieOCConfig(
		repo_path=args.oc_storm_repo,
		checkpoint_path=args.checkpoint,
		role_names=('whole_acrobot',),
		model_size=args.model_size,
		device=args.device,
		output_device='cpu',
		expected_input_size=(64, 64),
		support_input_size=support_size,
		mask_output_size=(64, 64),
		tracker_size=(args.tracker_size, args.tracker_size),
		foreground_queries=8,
		amp=not args.disable_amp,
		return_object_features=False,
		object_schema='generic_entity_indexed_v1',
	)
	adapter = CutieOCAdapter(config)
	adapter.add_support_prompts(support)
	records = []
	for episode_index, source_record in enumerate(manifest['episodes']):
		source_path = manifest_path.parent / source_record['file']
		with np.load(source_path, allow_pickle=False) as archive:
			rgb = {
				condition: np.array(archive[f'rgb_{condition}'], copy=True)
				for condition in ('clean', 'hard')
			}
		arrays = {}
		for condition in ('clean', 'hard'):
			frames = rgb[condition]
			masks = np.zeros(frames.shape[:3], dtype=np.bool_)
			valid = np.zeros(frames.shape[0], dtype=np.bool_)
			confidence = np.zeros(frames.shape[0], dtype=np.float32)
			mask_score = np.zeros(frames.shape[0], dtype=np.float32)
			runtime_ms = np.zeros(frames.shape[0], dtype=np.float64)
			adapter.reset_episode()
			for frame_index, frame in enumerate(frames):
				result = adapter.track(frame)
				mask = result.masks[0].numpy()
				lost = bool(result.lost[0])
				masks[frame_index] = mask
				valid[frame_index] = bool(mask.any()) and not lost
				confidence[frame_index] = float(result.confidence[0])
				mask_score[frame_index] = float(result.mask_score[0])
				runtime_ms[frame_index] = result.runtime_ms
			arrays[f'mask_{condition}'] = masks
			arrays[f'valid_{condition}'] = valid
			arrays[f'confidence_{condition}'] = confidence
			arrays[f'mask_score_{condition}'] = mask_score
			arrays[f'runtime_ms_{condition}'] = runtime_ms
		path = root / f'episode_{episode_index:04d}.npz'
		np.savez_compressed(path, **arrays)
		record = {
			'episode_index': episode_index,
			'file': path.name,
			'sha256': _sha256(path),
			'source_episode_sha256': source_record['sha256'],
			'frames': int(rgb['clean'].shape[0]),
			'valid_rate_clean': float(arrays['valid_clean'].mean()),
			'valid_rate_hard': float(arrays['valid_hard'].mean()),
			'mean_runtime_ms': float(np.concatenate((
				arrays['runtime_ms_clean'], arrays['runtime_ms_hard'],
			)).mean()),
		}
		records.append(record)
		print('ACROBOT_WHOLE_CUTIE_EPISODE', json.dumps(record), flush=True)
	payload = {
		'format': MASK_DATASET_FORMAT,
		'task': 'acrobot-swingup',
		'entity_names': ['whole_acrobot'],
		'source_manifest': str(manifest_path),
		'source_manifest_sha256': _sha256(manifest_path),
		'support_npz': str(args.support_npz.resolve()),
		'support_npz_sha256': _sha256(args.support_npz),
		'graph': str(args.graph.resolve()),
		'graph_sha256': graph.graph_sha256,
		'checkpoint_sha256': _sha256(args.checkpoint),
		'tracker_size': args.tracker_size,
		'causal': True,
		'label_access': 'rgb_keys_only_no_pose_or_segmentation_labels',
		'episodes': records,
	}
	manifest_out = root / 'manifest.json'
	manifest_out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
	print(f'MANIFEST={manifest_out}', flush=True)
	return payload


def parse_args(argv=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--input-manifest', type=Path, required=True)
	parser.add_argument('--support-npz', type=Path, required=True)
	parser.add_argument('--graph', type=Path, required=True)
	parser.add_argument('--oc-storm-repo', type=Path, required=True)
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--device', default='cuda:0')
	parser.add_argument('--model-size', choices=('small', 'base'), default='small')
	parser.add_argument('--tracker-size', type=int, default=448)
	parser.add_argument('--disable-amp', action='store_true')
	args = parser.parse_args(argv)
	for name in ('input_manifest', 'support_npz', 'graph', 'checkpoint'):
		if not getattr(args, name).is_file():
			parser.error(f'--{name.replace("_", "-")} is not a file.')
	if not args.oc_storm_repo.is_dir() or args.tracker_size < 64:
		parser.error('A valid OC-STORM repo and tracker-size>=64 are required.')
	return args


if __name__ == '__main__':
	generate(parse_args())
