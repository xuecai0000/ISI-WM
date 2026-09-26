"""Frozen legacy-checkpoint review and matched Finger encoder/target regression.

Never resumes or overwrites a previous run. The two GPU queues use identical
historical training settings; only the explicitly listed experimental factors
differ. Old validation seeds are regression data, not a new unseen test set.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import time


REPO = Path(__file__).resolve().parents[2]
ORIGINAL = Path('/home/<USER>/world/tdmpc2_2026')
ARMS = {
    'legacy_full': ('legacy_mlp_v1', 'full_descriptor'),
    'legacy_geometry': ('legacy_mlp_v1', 'geometry_status_full_denominator'),
    'spatial_full': ('spatial_graph_v1', 'full_descriptor'),
    'spatial_geometry': ('spatial_graph_v1', 'geometry_status_full_denominator'),
}
TASKS = ('finger-spin', 'cup-catch', 'cartpole-swingup')
PRINT_LOCK = threading.Lock()
PROCESS_LOCK = threading.RLock()
PROCESSES = set()
STOP = threading.Event()


def emit(event, **fields):
    with PRINT_LOCK:
        print(event, json.dumps(fields, sort_keys=True), flush=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def old_root(task):
    name = 'cutie_object_only100k_cutie_object_multitask_100k_v1_seed6_' + task.replace('-', '_')
    return ORIGINAL / 'logs' / task / '6' / name


def verify_sources(manifest):
    for path, expected in manifest['files'].items():
        if digest(path) != expected:
            raise RuntimeError(f'Frozen source/input changed: {path}')
    for path, expected in manifest.get('video_stats', {}).items():
        stat = Path(path).stat()
        if {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns} != expected:
            raise RuntimeError(f'Background video metadata changed: {path}')


def stop_children(*_):
    STOP.set()
    with PROCESS_LOCK:
        current = list(PROCESSES)
    for process in current:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def run_command(command, log, gpu=None):
    if STOP.is_set():
        raise RuntimeError('Run interrupted; no additional jobs will start.')
    env = dict(os.environ, MUJOCO_GL='egl', PYTHONUNBUFFERED='1')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('x', encoding='utf-8') as handle:
        with PROCESS_LOCK:
            if STOP.is_set():
                raise RuntimeError('Interrupted before process creation.')
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            PROCESSES.add(process)
        try:
            stopped_at = None
            while True:
                try:
                    rc = process.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    if STOP.is_set():
                        if stopped_at is None:
                            stopped_at = time.monotonic()
                        sig = signal.SIGKILL if time.monotonic() - stopped_at >= 30 else signal.SIGTERM
                        try:
                            os.killpg(process.pid, sig)
                        except ProcessLookupError:
                            pass
        finally:
            # A crashed leader can leave its own spawned perception workers.
            # Signal this exact owned group even if its leader has already exited.
            if process.returncode != 0 or STOP.is_set():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(.1)
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            with PROCESS_LOCK:
                PROCESSES.discard(process)
    if rc:
        raise RuntimeError(f'Command failed rc={rc}; see {log}')


def parallel(jobs):
    """Observe either queue's failure immediately, then reap only owned groups."""
    results = {}
    def guarded(function, arguments):
        try:
            return function(*arguments)
        except BaseException:
            stop_children()
            raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(guarded, function, arguments) for function, arguments in jobs]
        try:
            for future in concurrent.futures.as_completed(futures):
                results.update(future.result())
        except BaseException:
            stop_children()
            for future in futures:
                future.cancel()
            raise
    return results


def evaluate(root, task, condition, seed, output, gpu):
    command = [sys.executable, '-B', '-m', 'tdmpc2.tools.evaluate_cutie_multitask_checkpoint',
               '--task', task, '--backend', 'cutie_object_only', '--condition', condition,
               '--runtime-config', str(root / 'runtime_config.json'),
               '--checkpoint', str(root / 'models/final.pt'),
               '--training-seed', str(seed), '--expected-training-steps', '100000',
               '--expected-training-eval-freq', '20000', '--expected-training-eval-episodes', '3',
               '--episodes', '20', '--env-seed', '424243', '--background-seed', '1618034',
               '--planner-seed-base', '8675400', '--erosion-pixels', '0', '--output', str(output)]
    emit('EVAL_START', task=task, condition=condition, seed=seed, gpu=gpu, root=str(root))
    run_command(command, output.with_suffix('.log'), gpu)
    data = read_json(output)
    if (data['task'] != task or data['training_seed'] != seed or data['condition'] != condition
            or len(data['episodes']) != 20
            or data['provenance']['checkpoint_sha256'] != digest(root / 'models/final.pt')):
        raise RuntimeError(f'Evaluation provenance mismatch: {output}')
    emit('EVAL_END', task=task, condition=condition, seed=seed, gpu=gpu,
         mean=data['summary']['reward_mean'], std=data['summary']['reward_std'], rc=0)
    return data['summary']


def prepare_config(task, seed, arm, tag, stage):
    from omegaconf import OmegaConf

    old = read_json(old_root(task) / 'runtime_config.json')
    expected = dict(steps=100000, eval_freq=20000, eval_episodes=3, seed=6,
                    obs='rgb', flat_anchor=True, flat_anchor_mode='cutie_object_only',
                    cutie_object_num_roles=2, cutie_object_frame_dim=590,
                    cutie_object_stack_frames=3, cutie_object_input_dim=1770,
                    cutie_object_role_dim=64, cutie_object_only_latent_dim=128,
                    video_background_enabled=True, video_background_split='train')
    for key, value in expected.items():
        if old.get(key) != value:
            raise ValueError(f'Historical identity mismatch {task}: {key}={old.get(key)!r}')
    encoder, target = ARMS[arm]
    name = f'{tag}_{task}_{arm}'
    root = REPO / 'logs' / task / str(seed) / name
    if root.exists():
        raise FileExistsError(root)
    default = OmegaConf.load(REPO / 'tdmpc2/config.yaml')
    cfg = OmegaConf.merge(default, OmegaConf.create(old))
    updates = dict(seed=seed, exp_name=name, checkpoint=None, data_dir=None,
                   obs_shapes=None, action_dims=None, episode_lengths=None,
                   enable_wandb=False, save_video=False, save_csv=True, save_agent=True,
                   save_eval_episode_trace=True, cutie_object_regression_encoder=encoder,
                   cutie_object_auxiliary_target=target,
                   cutie_object_spatial_token_enabled=False,
                   cutie_object_variable_graph_enabled=False,
                   cutie_object_spatial_graph_path=str(REPO / 'tdmpc2/object_graphs/finger_spin.json')
                       if encoder == 'spatial_graph_v1' else None,
                   cutie_object_belief_enabled=False, cutie_object_belief_use_for_control=False,
                   object_state_supervision_enabled=False, object_state_supervision_collect_labels=False,
                   cutie_object_allow_simulator_runtime=False,
                   cutie_object_allow_simulator_kinematics_runtime=False)
    cfg = OmegaConf.merge(cfg, OmegaConf.create(updates))
    # Hydra struct mode requires these fields to exist when parse_cfg replaces
    # them. Clear their values, do not remove the required placeholders.
    for key in ('work_dir', 'task_title', 'bin_size'):
        cfg[key] = None
    cfg.hydra = dict(job=dict(chdir=False), run=dict(dir=str(stage / 'hydra' / f'{task}_{seed}_{arm}')))
    path = stage / 'configs' / f'{task}_{seed}_{arm}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(OmegaConf.to_yaml(cfg))
    resolved = OmegaConf.to_container(cfg, resolve=True)
    # Preserve every inherited protocol field and record explicit differences.
    write_new(path.with_suffix('.changes.json'), {
        'old_runtime': str(old_root(task) / 'runtime_config.json'),
        'overrides': updates, 'derived_keys_reset': ['work_dir', 'task_title', 'bin_size'],
        'all_new_default_keys': {k: v for k, v in resolved.items() if k not in old},
        'all_changed_existing_keys': {k: {'old': old[k], 'new': v}
                                      for k, v in resolved.items() if k in old and old[k] != v},
        'scientific_factors': {'encoder_package': encoder, 'auxiliary_target': target},
    })
    return path, root


def training_summary(root, expected_seed, expected_arm):
    cfg = read_json(root / 'runtime_config.json')
    enc, target = ARMS[expected_arm]
    checks = dict(seed=expected_seed, steps=100000, eval_freq=20000, eval_episodes=3,
                  cutie_object_regression_encoder=enc, cutie_object_auxiliary_target=target,
                  cutie_object_spatial_token_enabled=False, cutie_object_variable_graph_enabled=False,
                  cutie_object_allow_simulator_runtime=False,
                  cutie_object_allow_simulator_kinematics_runtime=False)
    for k, v in checks.items():
        if cfg.get(k) != v:
            raise ValueError(f'Runtime mismatch {k}: {root}')
    with (root / 'eval.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    steps = [float(row['step']) for row in rows]
    rewards = [float(row['episode_reward']) for row in rows]
    if steps != list(range(0, 100001, 20000)) or not all(map(math.isfinite, rewards)):
        raise ValueError(f'Incomplete/nonfinite learning curve: {root}')
    area = sum((rewards[i] + rewards[i-1]) * (steps[i]-steps[i-1]) / 2
               for i in range(1, len(steps))) / 100000
    return dict(root=str(root), auc=area, final=rewards[-1], peak=max(rewards),
                steps=steps, rewards=rewards, checkpoint_sha256=digest(root / 'models/final.pt'))


def reuse_review(source, stage, current_manifest):
    """Reuse successful review only with identical evaluator/model/input bytes."""
    source = source.resolve()
    prior = read_json(source / 'provenance/manifest.json')
    root_candidates = [Path(path).parents[2] for path in prior['files']
                       if path.endswith('/tdmpc2/tools/run_legacy_object_regression.py')]
    if len(root_candidates) != 1:
        raise ValueError('Cannot identify prior review source snapshot.')
    prior_repo = root_candidates[0]
    excluded = Path('tdmpc2/tools/run_legacy_object_regression.py')
    for raw, expected in prior['files'].items():
        path = Path(raw)
        if path.is_relative_to(prior_repo):
            relative = path.relative_to(prior_repo)
            if relative == excluded:
                continue  # Only orchestration/config generation may change.
            current = REPO / relative
        else:
            current = path
        if current_manifest['files'].get(str(current)) != expected:
            raise RuntimeError(f'Cannot reuse review after source/input change: {current}')
    if prior.get('video_stats') != current_manifest.get('video_stats'):
        raise RuntimeError('Cannot reuse review after background metadata change.')
    results, records = {}, {}
    for task in TASKS:
        old_path = source / 'review' / f'{task}_hard.json'
        data = read_json(old_path)
        if (data.get('task') != task or data.get('training_seed') != 6
                or data.get('condition') != 'hard' or len(data.get('episodes', [])) != 20
                or data['provenance']['checkpoint_sha256'] != digest(old_root(task) / 'models/final.pt')
                or data['provenance']['evaluator_sha256'] != digest(REPO / 'tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py')):
            raise ValueError(f'Cannot reuse invalid review: {old_path}')
        destination = stage / 'review' / old_path.name
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copy2(old_path, destination)
        records[task] = {'source': str(old_path), 'sha256': digest(old_path)}
        results[task] = data['summary']
        emit('EVAL_REUSED', task=task, source=str(old_path), mean=data['summary']['reward_mean'])
    write_new(stage / 'provenance/reused_review.json', records)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[6, 26])
    parser.add_argument('--gpus', nargs=2, type=int, default=[0, 1])
    parser.add_argument('--review-only', action='store_true')
    parser.add_argument('--reuse-review', type=Path)
    args = parser.parse_args()
    if len(set(args.gpus)) != 2 or args.seeds != [6, 26]:
        raise ValueError('Frozen plan requires distinct GPUs and ordered seeds 6,26.')
    base = args.output_root.resolve()
    stage = base.with_name(base.name + '.incomplete')
    if base.exists() or stage.exists():
        raise FileExistsError(base)
    stage.mkdir(parents=True)
    for name in ('contracts', 'configs', 'training', 'evaluations', 'review', 'provenance'):
        (stage / name).mkdir()
    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)
    emit('LEGACY_REGRESSION_START', output_root=str(base), stage=str(stage))
    try:
        files = {}
        for path in sorted((REPO / 'tdmpc2').rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.yaml', '.json'):
                files[str(path)] = digest(path)
        for task in TASKS:
            old = old_root(task)
            raw = read_json(old / 'runtime_config.json')
            inputs = [old / 'runtime_config.json', old / 'models/final.pt',
                      Path(raw['cutie_object_checkpoint']), Path(raw['cutie_object_support_path'])]
            manifests = Path(raw.get('video_background_manifest_dir') or REPO / 'tdmpc2/envs/background_manifests')
            inputs.extend(manifests.glob('*.json'))
            # Bind actual support RGB and mask files, not just annotations metadata.
            inputs.extend(p for p in Path(raw['cutie_object_support_path']).parent.rglob('*') if p.is_file())
            for path in inputs:
                files[str(path)] = digest(path)
            for path in Path(raw['cutie_object_repo']).rglob('*.py'):
                if '__pycache__' not in path.parts:
                    files[str(path)] = digest(path)
        # Video bytes are large shared read-only assets. Record their actual
        # path, size and mtime; do not mislabel these stat checks as byte hashes.
        video_stats = {}
        for video in sorted(Path(raw['video_background_root']).glob('*.mp4')):
            stat = video.stat()
            video_stats[str(video.resolve())] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        manifest = dict(files=files, seeds=args.seeds, gpus=args.gpus, arms=ARMS,
                        video_stats=video_stats, video_integrity_scope='size/mtime only, not byte hashes',
                        old_training_protocol='100k; eval every20k/3episodes; original train background',
                        evaluation_scope='reused historical validation for development, not final unseen test')
        write_new(stage / 'provenance/manifest.json', manifest)
        run_command([sys.executable, '-m', 'pip', 'freeze'], stage / 'provenance/pip_freeze.log')
        run_command(['nvidia-smi'], stage / 'provenance/gpu.log')
        emit('PHASE', name='static_contracts')
        for test in ('check_cutie_encoder_regression_contract.py',
                     'check_legacy_runner_config_contract.py',
                     'check_cutie_support_camera_contract.py',
                     'check_cutie_object_only_contract.py', 'check_cutie_object_auxiliary_target_contract.py'):
            run_command([sys.executable, '-B', str(REPO / 'tdmpc2' / test)], stage / 'contracts' / (test + '.log'))
        verify_sources(manifest)
        emit('PHASE', name='old_checkpoint_review')
        def review_queue(tasks, gpu):
            result = {}
            for task in tasks:
                output = stage / 'review' / f'{task}_hard.json'
                result[task] = evaluate(old_root(task), task, 'hard', 6, output, gpu)
            return result
        review = reuse_review(args.reuse_review, stage, manifest) if args.reuse_review else parallel([
            (review_queue, (('finger-spin', 'cartpole-swingup'), args.gpus[0])),
            (review_queue, (('cup-catch',), args.gpus[1])),
        ])
        write_new(stage / 'legacy_checkpoint_review.json', review)
        emit('OLD_CHECKPOINT_REVIEW_COMPLETE', results=review)
        verify_sources(manifest)
        if args.review_only:
            emit('REVIEW_ONLY_COMPLETE', stage=str(stage))
            return
        # Do not automatically spend ten training jobs if an old checkpoint
        # unexpectedly collapses. The coordinating agent reviews these exact
        # artifacts and writes a release record bound to their byte hashes.
        review_hashes = {task: digest(stage / 'review' / f'{task}_hard.json') for task in TASKS}
        emit('AWAITING_LEGACY_REVIEW', expected_hashes=review_hashes,
             release_path=str(stage / 'review_release.json'))
        deadline = time.monotonic() + 3600
        release_path = stage / 'review_release.json'
        while not release_path.exists():
            if STOP.is_set() or time.monotonic() >= deadline:
                raise RuntimeError('Checkpoint review was not released; training has not started.')
            time.sleep(1)
        release = read_json(release_path)
        if (release.get('decision') != 'continue_matched_finger_2x2'
                or release.get('review_sha256') != review_hashes):
            raise ValueError('Review release does not match the evaluated artifacts.')
        manifest['files'][str(release_path)] = digest(release_path)
        jobs = {}
        for seed in args.seeds:
            for arm in ARMS:
                jobs[('finger-spin', seed, arm)] = prepare_config('finger-spin', seed, arm, base.name, stage)
        for task in ('cup-catch', 'cartpole-swingup'):
            jobs[(task, 6, 'legacy_full')] = prepare_config(task, 6, 'legacy_full', base.name, stage)
        for cfg_path, _ in jobs.values():
            manifest['files'][str(cfg_path)] = digest(cfg_path)
            manifest['files'][str(cfg_path.with_suffix('.changes.json'))] = digest(cfg_path.with_suffix('.changes.json'))
        write_new(stage / 'provenance/training_manifest.json', manifest)
        write_new(stage / 'plan.json', {'runs': [dict(task=k[0], seed=k[1], arm=k[2],
                    config=str(v[0]), root=str(v[1])) for k,v in jobs.items()], 'review': review})
        def train_one(task, seed, arm, gpu):
            verify_sources(manifest)
            cfg_path, root = jobs[(task, seed, arm)]
            emit('TRAIN_START', task=task, seed=seed, arm=arm, gpu=gpu, root=str(root))
            log = stage / 'training' / f'{task}_seed{seed}_{arm}.log'
            run_command([sys.executable, '-u', str(REPO / 'tdmpc2/train.py'),
                         '--config-path', str(cfg_path.parent), '--config-name', cfg_path.stem], log, gpu)
            result = {'training': training_summary(root, seed, arm), 'held_out': {}}
            emit('TRAIN_END', task=task, seed=seed, arm=arm, gpu=gpu, rc=0,
                 auc=result['training']['auc'], final=result['training']['final'])
            for condition in ('clean', 'hard'):
                output = stage / 'evaluations' / f'{task}_seed{seed}_{arm}_{condition}.json'
                result['held_out'][condition] = evaluate(root, task, condition, seed, output, gpu)
            verify_sources(manifest)
            write_new(stage / 'evaluations' / f'{task}_seed{seed}_{arm}_summary.json', result)
            return result
        def queue(gpu, first_arms, second_arms, extra_task):
            results = {}
            for seed, arms in zip(args.seeds, (first_arms, second_arms)):
                for arm in arms:
                    key = f'finger-spin/seed{seed}/{arm}'
                    results[key] = train_one('finger-spin', seed, arm, gpu)
            results[f'{extra_task}/seed6/legacy_full'] = train_one(extra_task, 6, 'legacy_full', gpu)
            return results
        emit('PHASE', name='matched_100k_training', runs=len(jobs))
        results = parallel([
            (queue, (args.gpus[0], ('legacy_full', 'legacy_geometry'),
                     ('spatial_full', 'spatial_geometry'), 'cartpole-swingup')),
            (queue, (args.gpus[1], ('spatial_full', 'spatial_geometry'),
                     ('legacy_full', 'legacy_geometry'), 'cup-catch')),
        ])
        verify_sources(manifest)
        write_new(stage / 'legacy_object_regression_summary.json', dict(
            status='complete', scientific_scope='two-seed development comparison, no automatic scientific go',
            old_checkpoint_review=review, results=results, protocol=manifest))
        stage.rename(base)
        emit('LEGACY_OBJECT_REGRESSION_COMPLETE', summary=str(base / 'legacy_object_regression_summary.json'))
        print('RUNNER_RC=0', flush=True)
    except BaseException as exc:
        stop_children()
        emit('LEGACY_OBJECT_REGRESSION_FAILED', error=repr(exc), preserved_stage=str(stage))
        print('RUNNER_RC=1', flush=True)
        raise


if __name__ == '__main__':
    main()
