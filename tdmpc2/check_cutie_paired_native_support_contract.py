"""Stdlib-only contract for paired native64/native128 Cutie support v2."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from tdmpc2.common.cutie_paired_support import (
	FORMAT,
	MASK_GENERATION,
	PAIRING,
	RGB_GENERATION,
	SUPPORT_SCHEMA,
	compute_paired_support_id,
	sha256_json,
)
from tdmpc2.common.cutie_external_snapshot import (
	build_snapshot,
	validate_snapshot,
)


ROOT = PROJECT_DIR
COLLECTOR = ROOT / 'tools' / 'collect_cutie_paired_native_support.py'
MULTITASK_COLLECTOR = ROOT / 'tools' / 'collect_cutie_multitask_support.py'
ADAPTER = ROOT / 'perception' / 'cutie_oc_adapter.py'
EVALUATOR = ROOT / 'tools' / 'evaluate_cutie_native_resolution_tracker.py'
SHA = '1' * 64


def _payload():
	records = []
	for index in range(6):
		active_video = f'video{85 + index % 5}.mp4'
		background_state = {
			'active_source': f'/frozen/video_hard/{active_video}',
			'active_source_name': active_video,
			'start_index': 0,
			'frame_clock': index,
			'last_frame_index': index,
			'internal_random_state_sha256': SHA,
			'current_native64_background_sha256': SHA,
			'current_native64_background_shape': [64, 64, 3],
			'source_cache_order': [f'/frozen/video_hard/{active_video}'],
			'policy_stack_sha256': SHA,
			'last_composed_sha256': SHA,
		}
		rng_domains = {
			'python_global': SHA,
			'numpy_global': SHA,
			'action_generator': SHA,
			'background_random': SHA,
			'environment_and_task_rngs': [SHA],
			'torch_cpu': SHA,
			'torch_cuda': [],
		}
		records.append({
			'index': index,
			'accepted_state_ordinal': index,
			'reset_ordinal': index + 1,
			'random_prefix_steps': 4 + 3 * index,
			'active_video': active_video,
			'background_frame_index': index,
			'physics_state': {'sha256': SHA},
			'actions': {'sha256': SHA},
			'same_state_guard': {
				'pass': True,
				'physics_exact': True,
				'background_exact': True,
				'rng_exact': True,
				'physics_before_sha256': SHA,
				'physics_after_sha256': SHA,
				'background_state': background_state,
				'background_before_sha256': sha256_json(background_state),
				'background_after_sha256': sha256_json(background_state),
				'rng_domains': rng_domains,
				'rng_before_sha256': sha256_json(rng_domains),
				'rng_after_sha256': sha256_json(rng_domains),
			},
			'cross_resolution_evidence': {
				'clean128_distinct_from_bilinear64': True,
				'composed128_distinct_from_bilinear64': True,
				'mask128_distinct_from_nearest64': True,
			},
			'resolutions': {
				str(resolution): {
					'image': f'support_{resolution}/frames/support_{index:02d}.png',
					'image_sha256': SHA,
					'indexed_mask_sha256': SHA,
					'native_clean_rgb_sha256': SHA,
				} for resolution in (64, 128)
			},
		})
	return {
		'format': FORMAT,
		'roles': ['upper_arm', 'lower_arm'],
		'collection': {
			'support_schema': SUPPORT_SCHEMA,
			'task': 'acrobot-swingup',
			'split': 'support',
			'environment_seed': 1,
			'background_seed': 2,
			'action_seed': 3,
			'camera_id': 0,
			'resolutions': [64, 128],
			'manifest_sha256': SHA,
			'combined_manifest_sha256': SHA,
			'geom_catalog_sha256': SHA,
			'pairing': PAIRING,
			'rgb_generation': RGB_GENERATION,
			'mask_generation': MASK_GENERATION,
		},
		'records': records,
	}


class PairedNativeSupportContract(unittest.TestCase):
	def test_collector_accepts_all_benchmark_tasks_via_canonical_specs(self):
		source = COLLECTOR.read_text(encoding='utf-8')
		tree = ast.parse(source, filename=str(COLLECTOR))
		tasks = None
		for node in tree.body:
			if not isinstance(node, ast.Assign):
				continue
			if any(isinstance(target, ast.Name) and target.id == 'TASKS' for target in node.targets):
				tasks = ast.literal_eval(node.value)
				break
		self.assertEqual(tasks, (
			'reacher-visual-small',
			'cartpole-swingup',
			'acrobot-swingup',
		))
		canonical_imports = set()
		for node in tree.body:
			if (
				isinstance(node, ast.ImportFrom)
				and node.module == 'tdmpc2.tools.collect_cutie_multitask_support'
			):
				canonical_imports.update(alias.name for alias in node.names)
		self.assertTrue({
			'TASK_BY_NAME', '_catalog', '_selected_objects', '_named_selection',
			'_selector_payload', '_segmentation_constants', 'RoleVisibilityError',
		}.issubset(canonical_imports))
		for required in (
			'spec = TASK_BY_NAME[args.task]',
			'_selected_objects(catalog, selector) for selector in spec.selectors',
			"f'{spec.task} role selectors overlap in the model catalog.'",
			"'segmentation_object_types': _segmentation_constants()",
			"parser.add_argument('--task', choices=TASKS, required=True)",
		):
			self.assertIn(required, source)
		canonical = MULTITASK_COLLECTOR.read_text(encoding='utf-8')
		for required in (
			'task="reacher-visual-small"',
			'roles=("whole_arm", "goal")',
			'geom=(r"arm", r"hand", r"finger")',
			'RoleSelector(geom=(r"target",))',
		):
			self.assertIn(required, canonical)

	def test_identity_is_stable_but_covers_assets_and_guards(self):
		payload = _payload()
		identity = compute_paired_support_id(payload)
		self.assertEqual(len(identity), 64)
		from tdmpc2.common.cutie_paired_support import paired_support_identity_payload
		self.assertEqual(
			[record['index'] for record in paired_support_identity_payload(payload)['records']],
			list(range(6)),
		)
		relocated = copy.deepcopy(payload)
		relocated['records'][0]['resolutions']['64']['image'] = 'moved/support.png'
		self.assertEqual(compute_paired_support_id(relocated), identity)
		changed = copy.deepcopy(payload)
		changed['records'][0]['resolutions']['128']['image_sha256'] = '2' * 64
		self.assertNotEqual(compute_paired_support_id(changed), identity)
		for field in ('pass', 'physics_exact', 'background_exact', 'rng_exact'):
			bad = copy.deepcopy(payload)
			bad['records'][0]['same_state_guard'][field] = False
			with self.assertRaises(ValueError):
				compute_paired_support_id(bad)
		for field in ('background_state', 'rng_domains'):
			bad = copy.deepcopy(payload)
			del bad['records'][0]['same_state_guard'][field]
			with self.assertRaises(ValueError):
				compute_paired_support_id(bad)
		bad = copy.deepcopy(payload)
		bad['records'][0]['same_state_guard']['background_state'][
			'last_frame_index'
		] += 1
		with self.assertRaises(ValueError):
			compute_paired_support_id(bad)

	def test_schema_roles_ordinals_and_generation_fail_closed(self):
		mutations = []
		duplicate_roles = _payload()
		duplicate_roles['roles'] = ['arm', 'arm']
		mutations.append(duplicate_roles)
		bad_index = _payload()
		bad_index['records'][2]['index'] = 3
		mutations.append(bad_index)
		bad_reset = _payload()
		bad_reset['records'][0]['reset_ordinal'] = 0
		mutations.append(bad_reset)
		bad_generation = _payload()
		bad_generation['collection']['mask_generation'] = 'resize64'
		mutations.append(bad_generation)
		for payload in mutations:
			with self.assertRaises(ValueError):
				compute_paired_support_id(payload)

	def test_collector_has_native_pair_and_exact_no_mutation_guards(self):
		source = COLLECTOR.read_text(encoding='utf-8')
		for required in (
			'for resolution in RESOLUTIONS:',
			'render(height=resolution, width=resolution)',
			'_indexed_mask(physics, resolution, selections, roles)',
			"'physics_exact': True",
			"'background_exact': True",
			"'rng_exact': True",
			"'reset_ordinal': reset_ordinal",
			"'physics_state': _array_payload(state)",
			"'actions': _array_payload(action_array)",
			'Paired assets do not empirically distinguish native128',
		):
			self.assertIn(required, source)

	def test_loader_and_evaluator_bind_three_arm_interface(self):
		adapter = ADAPTER.read_text(encoding='utf-8')
		evaluator = EVALUATOR.read_text(encoding='utf-8')
		for required in (
			'def load_paired_native_support_prompts(',
			"asset.relative_to(path.parent)",
			"Paired support decoded RGB hash mismatch",
			"Paired support role pixel counts mismatch",
			"Paired native final reset ordinal/reset attempts mismatch",
		):
			self.assertIn(required, adapter)
		for required in (
			'PAIRED_FORMAT = "cutie_native_support_tracker_evaluation_v2"',
			'"--support-resolution"',
			'args.support_resolution, args.support_resolution',
			'arm = f"runtime{args.resolution}_support{args.support_resolution}"',
			'load_paired_native_support_prompts(',
		):
			self.assertIn(required, evaluator)

	def test_external_input_snapshot_rehashes_every_bound_tree(self):
		with tempfile.TemporaryDirectory() as temporary:
			root = Path(temporary)
			local_python = root / 'tdmpc2'
			video = root / 'video_hard'
			manifest = root / 'manifests'
			cutie = root / 'oc' / 'feature_extractor' / 'cutie' / 'cutie'
			config = cutie / 'config'
			for directory in (local_python, video, manifest, config):
				directory.mkdir(parents=True, exist_ok=True)
			(local_python / 'module.py').write_text('LOCAL = 1\n', encoding='utf-8')
			(video / 'video0.mp4').write_bytes(b'video-bytes')
			(manifest / 'validation.json').write_text('{}\n', encoding='utf-8')
			(cutie / 'model.py').write_text('VALUE = 1\n', encoding='utf-8')
			(config / 'default.yaml').write_text('model: small\n', encoding='utf-8')
			checkpoint = root / 'cutie-small-mega.pth'
			checkpoint.write_bytes(b'checkpoint')
			snapshot = build_snapshot(
				repo_root=root,
				video_root=video,
				manifest_dir=manifest,
				oc_repo=root / 'oc',
				cutie_checkpoint=checkpoint,
			)
			validated = validate_snapshot(snapshot)
			self.assertEqual(validated['snapshot_id'], snapshot['snapshot_id'])
			for target in (
				local_python / 'module.py',
				video / 'video0.mp4',
				manifest / 'validation.json',
				cutie / 'model.py',
				config / 'default.yaml',
				checkpoint,
			):
				original = target.read_bytes()
				target.write_bytes(original + b'changed')
				with self.assertRaises(ValueError):
					validate_snapshot(snapshot)
				target.write_bytes(original)


if __name__ == '__main__':
	unittest.main(verbosity=2)
