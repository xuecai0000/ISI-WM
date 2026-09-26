"""Read-only preflight for the standalone official OC-STORM Cutie adapter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from tdmpc2.perception.cutie_oc_adapter import (  # noqa: E402
	CutieOCAdapter,
	CutieOCConfig,
	CutiePreflightError,
	inspect_cutie_installation,
	load_point_support_prompts,
)


_CUTIE_RUNTIME_PACKAGES = {
	'torch': 'torch',
	'numpy': 'numpy',
	'hydra': 'hydra-core',
	'omegaconf': 'omegaconf',
	'einops': 'einops',
}
_CUTIE_SUPPORT_PACKAGES = {'PIL': 'Pillow'}


def _runtime_dependency_status(*, include_support: bool) -> dict[str, bool]:
	"""Report all adapter-path imports without stopping at the first failure."""
	packages = dict(_CUTIE_RUNTIME_PACKAGES)
	if include_support:
		packages.update(_CUTIE_SUPPORT_PACKAGES)
	return {
		module: importlib.util.find_spec(module) is not None
		for module in packages
	}


def _check_runtime_dependencies(*, include_support: bool) -> dict[str, bool]:
	status = _runtime_dependency_status(include_support=include_support)
	packages = dict(_CUTIE_RUNTIME_PACKAGES)
	if include_support:
		packages.update(_CUTIE_SUPPORT_PACKAGES)
	missing_modules = [module for module, present in status.items() if not present]
	if missing_modules:
		missing_packages = list(dict.fromkeys(
			packages[module] for module in missing_modules
		))
		details = ', '.join(
			f'{module} (pip package {packages[module]})'
			for module in missing_modules
		)
		raise CutiePreflightError(
			'Missing Cutie runtime modules in the active interpreter: '
			f'{details}. Install them together with: python -m pip install '
			+ ' '.join(missing_packages)
		)
	return status


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as handle:
		for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
			digest.update(chunk)
	return digest.hexdigest()


def parse_args():
	parser = argparse.ArgumentParser(
		description=(
			'Check official OC-STORM Cutie code, dependencies, weights, and GPU. '
			'The check never downloads resources and never selects a fallback.'
		)
	)
	parser.add_argument('--oc-storm-repo', type=Path, required=True)
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument(
		'--object-schema',
		choices=('legacy_three_object_v1', 'whole_arm_goal_v1'),
		default='legacy_three_object_v1',
		help='Cutie object IDs to install and return.',
	)
	parser.add_argument(
		'--roles',
		nargs='+',
		help=(
			'Optional explicit role order. Defaults to proximal_link distal_link goal '
			'for the legacy schema or whole_arm goal for whole_arm_goal_v1.'
		),
	)
	parser.add_argument('--model-size', choices=('small', 'base'), default='small')
	parser.add_argument('--device', default='cuda:0')
	parser.add_argument('--config-dir', type=Path)
	parser.add_argument(
		'--support-annotations',
		type=Path,
		help=(
			'Optional existing six-frame RGB-only support annotations. When model '
			'loading is enabled, all prompts are also installed as permanent memory.'
		),
	)
	parser.add_argument('--prompt-radius', type=float, default=2.0)
	parser.add_argument(
		'--tracker-size',
		type=int,
		nargs=2,
		metavar=('HEIGHT', 'WIDTH'),
		help='Omit for the strict input-resolution-matched protocol.',
	)
	parser.add_argument(
		'--skip-model-load',
		action='store_true',
		help='Only check files/imports; by default the checkpoint is loaded on GPU.',
	)
	parser.add_argument(
		'--sha256', action='store_true', help='Also hash the external checkpoint.'
	)
	parser.add_argument(
		'--traceback',
		action='store_true',
		help='On failure, print the complete chained exception traceback.',
	)
	return parser.parse_args()


def main():
	args = parse_args()
	default_roles = (
		('whole_arm', 'goal')
		if args.object_schema == 'whole_arm_goal_v1'
		else ('proximal_link', 'distal_link', 'goal')
	)
	config = CutieOCConfig(
		repo_path=args.oc_storm_repo,
		checkpoint_path=args.checkpoint,
		role_names=tuple(args.roles) if args.roles else default_roles,
		model_size=args.model_size,
		device=args.device,
		config_dir=args.config_dir,
		tracker_size=tuple(args.tracker_size) if args.tracker_size else None,
		object_schema=args.object_schema,
	)
	try:
		dependency_status = _check_runtime_dependencies(
			include_support=args.support_annotations is not None
		)
		report = inspect_cutie_installation(config, import_check=True)
		report['runtime_dependency_status'] = dependency_status
		support = None
		if args.support_annotations is not None:
			support = load_point_support_prompts(
				args.support_annotations,
				radius_px=args.prompt_radius,
				object_schema=args.object_schema,
			)
			if support.role_names != config.role_names:
				raise ValueError(
					f'Support roles {support.role_names!r} do not match --roles '
					f'{config.role_names!r}.'
				)
			report['support'] = support.metadata
		if args.sha256:
			report['checkpoint_sha256'] = _sha256(config.checkpoint)
		if not args.skip_model_load:
			adapter = CutieOCAdapter(config)
			if support is not None:
				adapter.add_support_prompts(support)
				report['permanent_prompt_runtime'] = adapter.runtime_summary()
				# A permanent prompt step intentionally does not expose OC-STORM's
				# query/defocus caches. Probe one subsequent mask-free frame so this
				# preflight verifies the schema-selected Kx2048 feature path as well as memory
				# installation.
				adapter.reset_episode()
				probe = adapter.track(support.frames[-1])
				features = probe.object_features
				expected_shape = (len(config.role_names), 2048)
				if features is None or tuple(features.shape) != expected_shape:
					raise CutiePreflightError(
						'Cutie tracking probe did not produce the required object '
						f'features {expected_shape}; got '
						f'{None if features is None else tuple(features.shape)}.'
					)
				if not bool(features.isfinite().all().item()):
					raise CutiePreflightError(
						'Cutie tracking probe produced non-finite object features.'
					)
				report['tracking_probe'] = {
					'feature_shape': list(features.shape),
					'lost': probe.lost.tolist(),
					'runtime_ms': probe.runtime_ms,
				}
				report['adapter_runtime'] = adapter.runtime_summary()
			report['model_load'] = True
			report['feature_dim'] = config.foreground_queries * 256
			del adapter
		else:
			report['model_load'] = False
	except (CutiePreflightError, ValueError) as exc:
		if args.traceback:
			traceback.print_exception(
				type(exc), exc, exc.__traceback__, file=sys.stderr
			)
		raise SystemExit(f'CUTIE_OC_PREFLIGHT_FAILED: {exc}') from exc
	print('CUTIE_OC_PREFLIGHT_OK', json.dumps(report, sort_keys=True))


if __name__ == '__main__':
	main()
