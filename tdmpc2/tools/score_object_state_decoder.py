"""Privileged offline scoring for a train-only object-state decoder.

The controller still receives only its normal visual observation. Simulator
state is read by this separate scorer and is never passed to the policy,
planner, encoder, dynamics, reward model, or value model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from tdmpc2.tools import evaluate_cutie_multitask_checkpoint as checkpoint_eval


FORMAT = 'object_state_decoder_privileged_score_v1'
THRESHOLDS = (0.05, 0.1, 0.2)


def _longest_burst(values, threshold):
	best = current = 0
	for value in values:
		if value > threshold:
			current += 1
			best = max(best, current)
		else:
			current = 0
	return int(best)


def _summarize(errors, fields):
	errors = np.asarray(errors, dtype=np.float64)
	if errors.ndim != 2 or errors.shape[1] != len(fields):
		raise RuntimeError(f'Unexpected decoder error shape: {errors.shape}.')
	if not np.isfinite(errors).all():
		raise RuntimeError('Decoder produced non-finite errors.')
	abs_error = np.abs(errors)
	frame_rmse = np.sqrt(np.mean(np.square(errors), axis=1))
	return {
		'frames': int(errors.shape[0]),
		'target_dim': int(errors.shape[1]),
		'frame_rmse_mean': float(frame_rmse.mean()),
		'frame_rmse_median': float(np.median(frame_rmse)),
		'frame_rmse_p90': float(np.quantile(frame_rmse, 0.9)),
		'frame_rmse_p95': float(np.quantile(frame_rmse, 0.95)),
		'frame_rmse_max': float(frame_rmse.max()),
		'failure_bursts': {
			str(threshold): {
				'failure_rate': float(np.mean(frame_rmse > threshold)),
				'max_consecutive_frames': _longest_burst(frame_rmse, threshold),
			}
			for threshold in THRESHOLDS
		},
		'per_field': {
			field: {
				'mae': float(abs_error[:, index].mean()),
				'rmse': float(np.sqrt(np.mean(np.square(errors[:, index])))),
				'p90_absolute_error': float(np.quantile(abs_error[:, index], 0.9)),
				'p95_absolute_error': float(np.quantile(abs_error[:, index], 0.95)),
			}
			for index, field in enumerate(fields)
		},
	}


def score(args):
	import torch
	from common.seed import set_seed
	from envs import make_env
	from envs.wrappers.object_state_supervision import find_state_supervision_wrapper
	from tdmpc2.tdmpc2 import TDMPC2

	checkpoint_eval._validate_args(args)
	raw = checkpoint_eval._json(args.runtime_config)
	if not raw.get('object_state_supervision_enabled', False):
		raise ValueError('Checkpoint was not built with an object-state decoder.')
	cfg = checkpoint_eval._prepare(args, raw)
	cfg.object_state_supervision_collect_labels = True
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required.')
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.set_float32_matmul_precision('high')

	env = None
	errors = []
	rewards = []
	try:
		set_seed(args.env_seed)
		env = make_env(cfg)
		wrapper = find_state_supervision_wrapper(env)
		if wrapper is None:
			raise RuntimeError('State-supervision wrapper is unavailable.')
		agent = TDMPC2(cfg)
		agent.load(args.checkpoint)
		agent.eval()
		for episode_index in range(args.episodes):
			obs = env.reset()
			set_seed(args.planner_seed_base + episode_index)
			agent._prev_mean.zero_()
			reward_sum = 0.0
			for step_index in range(int(cfg.episode_length)):
				torch.compiler.cudagraph_mark_step_begin()
				checkpoint_eval._align_object_only_rgb_shift_rng(torch, args.backend)
				action = agent.act(obs, t0=step_index == 0, eval_mode=True)

				# Preserve all planner RNG streams: this diagnostic must not alter
				# the visual-only action sequence it is observing.
				cpu_rng = torch.random.get_rng_state()
				cuda_rng = torch.cuda.get_rng_state()
				with torch.no_grad():
					model_obs = obs.to(agent.device, non_blocking=True).unsqueeze(0)
					latent = agent.model.encode(model_obs, task=None)
					prediction = agent.model.decode_supervised_state(latent)[0]
				prediction = prediction.detach().cpu().numpy().astype(np.float64)
				torch.random.set_rng_state(cpu_rng)
				torch.cuda.set_rng_state(cuda_rng)
				target = wrapper.get_object_state_target().astype(np.float64)
				if prediction.shape != target.shape:
					raise RuntimeError(
						f'Prediction/target shape mismatch: {prediction.shape} != {target.shape}.'
					)
				errors.append(prediction - target)
				obs, reward, done, _ = env.step(action)
				reward_sum += float(reward)
				if done:
					if step_index + 1 != int(cfg.episode_length):
						raise RuntimeError('Episode terminated at an unexpected length.')
					break
			else:
				raise RuntimeError('Environment did not terminate.')
			rewards.append(reward_sum)
			print('OBJECT_STATE_DECODER_SCORE_EPISODE', json.dumps({
				'condition': args.condition, 'episode_index': episode_index,
				'reward': reward_sum,
			}, allow_nan=False), flush=True)
		metrics = wrapper.state_supervision_metrics()
		if metrics['label_reads'] != args.episodes * (int(cfg.episode_length) + 1):
			raise RuntimeError(f'Unexpected privileged label-read count: {metrics}.')
	finally:
		if env is not None and callable(getattr(env, 'close', None)):
			env.close()

	fields = raw.get('object_state_supervision_target_fields')
	if not isinstance(fields, list):
		from common import object_state_supervision
		fields = list(object_state_supervision.contract(cfg)['target_fields'])
	result = {
		'format': FORMAT,
		'task': args.task,
		'condition': args.condition,
		'checkpoint': str(args.checkpoint.resolve()),
		'checkpoint_sha256': checkpoint_eval._sha256(args.checkpoint),
		'runtime_config': str(args.runtime_config.resolve()),
		'episodes': args.episodes,
		'policy_input': 'visual_only_unchanged',
		'privileged_target_usage': 'offline_scoring_only_after_action_selection',
		'controller_input_contains_state': False,
		'planner_rng_preserved_around_decoder_probe': True,
		'score': _summarize(errors, fields),
		'reward_mean': float(np.mean(rewards)),
		'reward_std': float(np.std(rewards, ddof=1)),
		'label_audit': metrics,
		'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
		'device_name': torch.cuda.get_device_name(0),
	}
	return result


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--task', choices=checkpoint_eval.TASKS, required=True)
	parser.add_argument('--backend', choices=('cutie_object_only',), default='cutie_object_only')
	parser.add_argument('--condition', choices=checkpoint_eval.CONDITIONS, required=True)
	parser.add_argument('--runtime-config', type=Path, required=True)
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--checkpoint-step', type=int)
	parser.add_argument('--training-seed', type=int, required=True)
	parser.add_argument('--expected-training-steps', type=int, required=True)
	parser.add_argument('--expected-training-eval-freq', type=int, required=True)
	parser.add_argument('--expected-training-eval-episodes', type=int, required=True)
	parser.add_argument('--episodes', type=int, default=20)
	parser.add_argument('--env-seed', type=int, default=424243)
	parser.add_argument('--background-seed', type=int, default=1618034)
	parser.add_argument('--planner-seed-base', type=int, default=8675400)
	parser.add_argument('--erosion-pixels', type=int, default=0)
	parser.add_argument('--output', type=Path, required=True)
	return parser.parse_args(argv)


def main(argv=None):
	args = parse_args(argv)
	if args.output.exists():
		raise FileExistsError(args.output)
	payload = score(args)
	checkpoint_eval._write(args.output, payload)
	print('OBJECT_STATE_DECODER_SCORE_COMPLETE', json.dumps({
		'condition': args.condition,
		'frame_rmse_mean': payload['score']['frame_rmse_mean'],
		'frame_rmse_p90': payload['score']['frame_rmse_p90'],
		'output': str(args.output.resolve()),
	}, allow_nan=False), flush=True)


if __name__ == '__main__':
	main()
