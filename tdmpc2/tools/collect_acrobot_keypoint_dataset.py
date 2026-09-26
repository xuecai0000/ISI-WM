"""Collect causal Acrobot RGB sequences with simulator-only pose labels.

The produced labels are privileged training/evaluation data. Controller-time
code never receives this archive and never calls the helpers in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


os.environ.setdefault('MUJOCO_GL', 'egl')
PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))

from tdmpc2.common.visual_articulated_pose import TOTAL_LINK_LENGTH  # noqa: E402


FORMAT = 'acrobot_keypoint_sequence_dataset_v1'
POINT_NAMES = ('base', 'elbow', 'tip')


class Config(SimpleNamespace):
	def get(self, name, default=None):
		return getattr(self, name, default)


def _find(env, predicate, description):
	current, seen = env, set()
	while current is not None and id(current) not in seen:
		seen.add(id(current))
		if predicate(current):
			return current
		current = getattr(current, 'env', getattr(current, '_env', None))
	raise RuntimeError(f'Could not find {description} in environment chain.')


def _physics(env):
	wrapper = _find(
		env, lambda value: value.__class__.__name__ == 'DMControlWrapper',
		'DMControlWrapper',
	)
	physics = getattr(wrapper.env, 'physics', None)
	if physics is None:
		raise RuntimeError('DMControlWrapper has no dm_control physics instance.')
	return physics


def _points_world(physics):
	points = np.asarray([
		physics.named.data.xpos['upper_arm'],
		physics.named.data.xpos['lower_arm'],
		physics.named.data.site_xpos['tip'],
	], dtype=np.float64)
	if points.shape != (3, 3) or not np.isfinite(points).all():
		raise RuntimeError(f'Invalid Acrobot world points {points!r}.')
	return points


def _reset_pose_coverage(physics, rng):
	"""Sample the full two-link configuration space for perception coverage."""
	if int(physics.model.nq) != 2 or int(physics.model.nv) != 2:
		raise RuntimeError('Acrobot keypoint collection requires nq=nv=2.')
	qpos = rng.uniform(-np.pi, np.pi, size=2)
	qvel = rng.uniform(-4., 4., size=2)
	with physics.reset_context():
		physics.data.qpos[:] = qpos
		physics.data.qvel[:] = qvel
	return qpos.astype(np.float32), qvel.astype(np.float32)


def _project_points(physics, points, *, width, height, camera_id=0):
	"""Project MuJoCo world points with the active fixed camera calibration."""
	position = np.asarray(physics.data.cam_xpos[camera_id], dtype=np.float64)
	rotation = np.asarray(physics.data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
	camera = (np.asarray(points, dtype=np.float64) - position) @ rotation
	depth = -camera[:, 2]
	fovy = float(physics.model.cam_fovy[camera_id]) * np.pi / 180.
	tan_half_y = np.tan(fovy / 2.)
	aspect = float(width) / float(height)
	x_ndc = camera[:, 0] / np.maximum(depth * tan_half_y * aspect, 1e-12)
	y_ndc = camera[:, 1] / np.maximum(depth * tan_half_y, 1e-12)
	pixel = np.stack((
		(x_ndc + 1.) * (width - 1.) / 2.,
		(1. - y_ndc) * (height - 1.) / 2.,
	), axis=-1)
	visible = (
		(depth > 0.) & (pixel[:, 0] >= 0.) & (pixel[:, 0] < width)
		& (pixel[:, 1] >= 0.) & (pixel[:, 1] < height)
	)
	if not np.isfinite(pixel).all():
		raise RuntimeError('Camera projection produced non-finite keypoints.')
	return pixel.astype(np.float32), visible.astype(np.bool_)


def _config(args, seed):
	return Config(
		task='acrobot-swingup', obs='rgb', seed=seed, multitask=False,
		flat_anchor=False, cutie_object_observation_variant='full',
		visual_foreground_erosion_pixels=0,
		video_background_enabled=True,
		video_background_root=str(args.video_root),
		video_background_manifest_dir=(
			str(args.manifest_dir) if args.manifest_dir is not None else None
		),
		video_background_split=args.background_split,
		video_background_strength=1.0,
		video_background_total_frames=args.background_total_frames,
		video_background_source_cache_size=args.background_cache_size,
		video_background_seed=args.background_seed + seed,
	)


def _sha256(path):
	digest = hashlib.sha256()
	with Path(path).open('rb') as stream:
		for chunk in iter(lambda: stream.read(1024 * 1024), b''):
			digest.update(chunk)
	return digest.hexdigest()


def collect(args):
	from tdmpc2.envs import dmcontrol

	root = args.output.resolve()
	if root.exists():
		raise FileExistsError(f'Refusing to overwrite dataset directory: {root}')
	root.mkdir(parents=True)
	records = []
	for episode_index in range(args.episodes):
		seed = args.seed + episode_index
		env = dmcontrol.make_env(_config(args, seed))
		try:
			physics = _physics(env)
			background = _find(
				env, lambda value: value.__class__.__name__
				== 'ColorMultiVideoBackgroundWrapper',
				'video-background wrapper',
			)
			rng = np.random.default_rng(args.action_seed + episode_index)
			pose_rng = np.random.default_rng(args.pose_seed + episode_index)
			env.reset()
			if args.initial_state_mode == 'uniform':
				initial_qpos, initial_qvel = _reset_pose_coverage(physics, pose_rng)
			else:
				initial_qpos = np.asarray(physics.data.qpos, dtype=np.float32).copy()
				initial_qvel = np.asarray(physics.data.qvel, dtype=np.float32).copy()
			clean = np.empty((args.steps + 1, args.resolution, args.resolution, 3), np.uint8)
			hard = np.empty_like(clean)
			actions = np.empty((args.steps,) + env.action_space.shape, np.float32)
			world_xz = np.empty((args.steps + 1, 3, 2), np.float32)
			pixel_xy = np.empty((args.steps + 1, 3, 2), np.float32)
			pixel_visible = np.empty((args.steps + 1, 3), np.bool_)
			physics_time = np.empty(args.steps + 1, np.float64)
			for frame_index in range(args.steps + 1):
				clean[frame_index] = physics.render(
					height=args.resolution, width=args.resolution, camera_id=0,
				)
				hard[frame_index] = background.cutie_same_state_rgb(
					height=args.resolution, width=args.resolution,
				)
				points = _points_world(physics)
				world_xz[frame_index] = points[:, [0, 2]] / TOTAL_LINK_LENGTH
				pixel_xy[frame_index], pixel_visible[frame_index] = _project_points(
					physics, points, width=args.resolution, height=args.resolution,
				)
				physics_time[frame_index] = float(physics.data.time)
				if frame_index == args.steps:
					break
				action = rng.uniform(env.action_space.low, env.action_space.high).astype(
					env.action_space.dtype
				)
				actions[frame_index] = action
				_, _, done, _ = env.step(action)
				if done and frame_index + 1 != args.steps:
					raise RuntimeError(f'Acrobot episode {episode_index} ended early.')
			delta = world_xz[:, 1:] - world_xz[:, :-1]
			theta = np.arctan2(delta[..., 0], delta[..., 1])
			global_omega = np.zeros((args.steps + 1, 2), dtype=np.float32)
			angle_delta = (np.diff(theta, axis=0) + np.pi) % (2. * np.pi) - np.pi
			dt = np.diff(physics_time)
			if np.any(~np.isfinite(dt)) or np.any(dt <= 0.):
				raise RuntimeError(f'Invalid Acrobot control time deltas: {dt}.')
			global_omega[1:] = angle_delta / dt[:, None]
			path = root / f'episode_{episode_index:04d}.npz'
			np.savez_compressed(
				path, rgb_clean=clean, rgb_hard=hard, actions=actions,
				world_xz=world_xz, global_omega=global_omega,
				pixel_xy=pixel_xy, pixel_visible=pixel_visible,
			)
			record = {
				'episode_index': episode_index,
				'env_seed': seed,
				'action_seed': args.action_seed + episode_index,
				'pose_seed': args.pose_seed + episode_index,
				'initial_qpos': initial_qpos.tolist(),
				'initial_qvel': initial_qvel.tolist(),
				'file': path.name,
				'sha256': _sha256(path),
				'background_source': Path(background.active_source).name,
				'frames': args.steps + 1,
			}
			records.append(record)
			print('ACROBOT_KEYPOINT_DATASET_EPISODE', json.dumps(record), flush=True)
		finally:
			env.close()
	manifest = {
		'format': FORMAT,
		'task': 'acrobot-swingup',
		'point_names': list(POINT_NAMES),
		'resolution': args.resolution,
		'position_normalization_total_link_length': TOTAL_LINK_LENGTH,
		'camera_id': 0,
		'angular_velocity': 'signed_wrapped_absolute_link_angle_delta_over_sim_time',
		'background_split': args.background_split,
		'env_seed_base': args.seed,
		'action_seed_base': args.action_seed,
		'background_seed_base': args.background_seed,
		'pose_seed_base': args.pose_seed,
		'initial_state_mode': args.initial_state_mode,
		'episodes': records,
	}
	manifest_path = root / 'manifest.json'
	manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
	print(f'MANIFEST={manifest_path}', flush=True)
	return manifest


def parse_args(argv=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--video-root', type=Path, required=True)
	parser.add_argument('--manifest-dir', type=Path)
	parser.add_argument('--background-split', choices=('train', 'validation', 'test'), required=True)
	parser.add_argument('--episodes', type=int, default=40)
	parser.add_argument('--steps', type=int, default=500)
	parser.add_argument('--resolution', type=int, default=64)
	parser.add_argument('--seed', type=int, default=271828)
	parser.add_argument('--action-seed', type=int, default=314159)
	parser.add_argument('--background-seed', type=int, default=161803)
	parser.add_argument('--pose-seed', type=int, default=141421)
	parser.add_argument('--initial-state-mode', choices=('native', 'uniform'), default='uniform')
	parser.add_argument('--background-total-frames', type=int, default=1000)
	parser.add_argument('--background-cache-size', type=int, default=8)
	args = parser.parse_args(argv)
	if args.episodes < 1 or args.steps < 3 or args.resolution != 64:
		parser.error('episodes>=1, steps>=3, and resolution=64 are required.')
	return args


if __name__ == '__main__':
	collect(parse_args())
