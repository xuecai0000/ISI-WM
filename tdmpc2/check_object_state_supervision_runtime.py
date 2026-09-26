"""Real Cutie/simulator smoke for pure-visual state-supervised replay.

This performs only short diagnostic rollouts. It does not train a controller.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch

from check_object_state_supervision_update import make_config
from common import object_state_supervision as supervision
from common.buffer import Buffer
from common.seed import set_seed
from envs import make_env
from envs.wrappers.object_state_supervision import without_state_labels
from trainer.online_trainer import OnlineTrainer


def prepare_config(args, task):
    cfg = make_config(0.1, task)
    cfg.seed = 271828
    cfg.steps = 32
    cfg.buffer_size = 32
    cfg.batch_size = 2
    cfg.horizon = 3
    cfg.cutie_object_repo = str(args.cutie_repo)
    cfg.cutie_object_checkpoint = str(args.checkpoint)
    support_version = 'v2' if task == 'acrobot-swingup' else 'v1'
    cfg.cutie_object_support_path = str(
        args.original_repo / 'datasets' / f'cutie_multitask_support_{support_version}_seed314159'
        / task / 'annotations.json'
    )
    cfg.cutie_object_device = 'cuda:0'
    cfg.cutie_object_tracker_height = 448
    cfg.cutie_object_tracker_width = 448
    cfg.cutie_object_amp = True
    cfg.cutie_object_native_highres_enabled = False
    cfg.video_background_enabled = True
    cfg.video_background_root = str(args.video_root)
    cfg.video_background_split = 'train'
    cfg.video_background_manifest_dir = None
    cfg.video_background_seed = 314159
    return cfg


def check_sample_alignment(buffer, episode):
    direct = buffer._prepare_batch(episode.unsqueeze(1))
    torch.testing.assert_close(direct[5][:, 0].cpu(), episode[supervision.REPLAY_KEY])
    torch.testing.assert_close(direct[1][:, 0].cpu(), episode['action'][1:])
    buffer.add(episode)
    result = buffer.sample()
    obs, actions, _, _, _, targets = result
    assert targets.shape == (buffer.cfg.horizon + 1, buffer.cfg.batch_size, supervision.target_dim(buffer.cfg))
    assert set(obs.keys()) == {'object'}
    # Check each real sampled slice against its exact source subsequence,
    # including observation/target row zero and shifted executed actions.
    source_targets = episode[supervision.REPLAY_KEY]
    for batch in range(buffer.cfg.batch_size):
        matches = []
        for start in range(len(episode) - buffer.cfg.horizon):
            stop = start + buffer.cfg.horizon + 1
            if torch.equal(targets[:, batch].cpu(), source_targets[start:stop]):
                matches.append(start)
        assert matches, 'Sampled targets were not a contiguous original episode slice'
        valid = any(
            torch.equal(obs['object'][:, batch].cpu(), episode['obs']['object'][start:start + buffer.cfg.horizon + 1])
            and torch.equal(actions[:, batch].cpu(), episode['action'][start + 1:start + buffer.cfg.horizon + 1])
            for start in matches
        )
        assert valid, 'Replay misaligned state labels, images and executed actions'


def run_task(args, task):
    cfg = prepare_config(args, task)
    set_seed(cfg.seed)
    env = None
    try:
        env = make_env(cfg)
        assert env.state_supervision_label_reads == 0
        trainer = OnlineTrainer.__new__(OnlineTrainer)
        trainer.cfg, trainer.env = cfg, env
        observation = env.reset()
        assert set(observation.keys()) == {'object'}
        assert tuple(observation['object'].shape) == (2, 1770)
        rows = [trainer.to_td(observation)]
        for _ in range(8):
            action = env.rand_act()
            observation, reward, done, info = env.step(action)
            assert not done
            assert set(observation.keys()) == {'object'}
            assert supervision.REPLAY_KEY not in info
            rows.append(trainer.to_td(observation, action, reward, info['terminated']))
        assert env.state_supervision_label_reads == 9
        episode = torch.cat(rows)
        assert torch.isfinite(episode[supervision.REPLAY_KEY]).all()
        buffer = Buffer(cfg)
        check_sample_alignment(buffer, episode)
        reads = env.state_supervision_label_reads
        with without_state_labels(env):
            env.reset()
            for _ in range(2):
                env.step(env.rand_act())
            assert env.state_supervision_label_reads == reads
            try:
                env.get_object_state_target()
            except RuntimeError:
                pass
            else:
                raise AssertionError('Evaluation allowed state-target access')
        env.reset()
        assert env.state_supervision_label_reads == reads + 1
        metrics = env.state_supervision_metrics()
        assert env.metrics() is not None, 'State wrapper hid Cutie metrics'
        del buffer, trainer, rows, episode
    finally:
        if env is not None:
            env.close()
        torch.cuda.empty_cache()

    evaluation_cfg = deepcopy(cfg)
    evaluation_cfg.object_state_supervision_collect_labels = False
    evaluation_cfg.video_background_enabled = False
    evaluation = None
    try:
        evaluation = make_env(evaluation_cfg)
        evaluation.reset()
        evaluation.step(evaluation.rand_act())
        assert evaluation.state_supervision_label_reads == 0
        assert evaluation._source is None
    finally:
        if evaluation is not None:
            evaluation.close()
        torch.cuda.empty_cache()
    print('OBJECT_STATE_SUPERVISION_RUNTIME_TASK_OK', json.dumps({
        'task': task, 'target_dim': supervision.target_dim(cfg),
        'training_label_reads': metrics['label_reads'],
        'periodic_evaluation_extra_label_reads': 0,
        'heldout_evaluation_label_reads': 0,
        'real_replay_alignment': 'pass',
        'true_entity_tracker': bool(cfg.cutie_object_true_entity_enabled),
    }), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-repo', type=Path, required=True)
    parser.add_argument('--video-root', type=Path, required=True)
    parser.add_argument('--cutie-repo', type=Path, default=Path('<DATA_PATH>/r2_hrssm_third_party/OC-STORM'))
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--task', choices=list(supervision.SCHEMAS), action='append')
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = args.cutie_repo / 'feature_extractor/cutie/weights/cutie-small-mega.pth'
    torch.set_num_threads(2)
    for task in args.task or list(supervision.SCHEMAS):
        run_task(args, task)
    print('OBJECT_STATE_SUPERVISION_RUNTIME_OK', flush=True)


if __name__ == '__main__':
    main()
