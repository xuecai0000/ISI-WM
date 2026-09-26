"""Record and strictly aggregate a paired, training-only state-supervision screen.

Uses only the Python standard library. Scientific success is deliberately not a
publication gate: a complete negative result is a successfully completed run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


TASKS = ('acrobot-swingup', 'cartpole-swingup', 'reacher-visual-small')
ARMS = {'state_aux_off': 0.0, 'state_aux_on': 0.1}
LOCATION_FIELDS = {'exp_name', 'work_dir'}
SOURCE_EXTENSIONS = {'.py', '.yaml', '.yml', '.sh', '.json'}


def load(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.incomplete')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def source_hashes(repo):
    source = Path(repo) / 'tdmpc2'
    return {
        str(path.relative_to(repo)): sha256(path)
        for path in sorted(source.rglob('*'))
        if path.is_file() and path.suffix in SOURCE_EXTENSIONS
        and '__pycache__' not in path.parts
    }


def record(args):
    if args.steps <= 0 or args.eval_freq <= 0 or args.steps % args.eval_freq:
        raise ValueError('Steps must be a positive multiple of eval frequency.')
    if args.eval_freq % 500 or args.eval_episodes < 1:
        raise ValueError('Evaluation frequency must align with 500-step episodes.')
    selected_tasks = tuple(item.strip() for item in args.tasks.split(',') if item.strip())
    if not selected_tasks or len(selected_tasks) != len(set(selected_tasks)):
        raise ValueError('Tasks must be a nonempty, duplicate-free comma-separated list.')
    unknown = sorted(set(selected_tasks) - set(TASKS))
    if unknown:
        raise ValueError(f'Unsupported tasks: {unknown}')
    repo = args.repo.resolve()
    inputs = {}
    for task in selected_tasks:
        support_root = args.support_root_v2 if task == 'acrobot-swingup' else args.support_root_v1
        annotation = (support_root / task / 'annotations.json').resolve()
        inputs[str(annotation)] = sha256(annotation)
    checkpoint = args.cutie_checkpoint.resolve()
    inputs[str(checkpoint)] = sha256(checkpoint)
    source = source_hashes(repo)
    if not source:
        raise ValueError('Empty source inventory.')
    run_tag = args.run_tag
    payload = {
        'format': 'object_state_supervision_paired_protocol_v1',
        'repo_root': str(repo), 'run_tag': run_tag,
        'tasks': list(selected_tasks), 'arms': ARMS,
        'seed': args.seed, 'steps': args.steps, 'eval_freq': args.eval_freq,
        'eval_episodes': args.eval_episodes, 'held_out_episodes': 20,
        'held_out_env_seed': 424243, 'held_out_background_seed': 1618034,
        'held_out_planner_seed_base': 8675400,
        'training_background': 'hard/train',
        'periodic_eval_semantics': 'existing trainer reset sequence; not claimed fixed seeds',
        'held_out_semantics': '20 fixed-seed episodes; clean and hard/validation',
        'policy_inputs': 'frozen Cutie spatial object tokens only; three-frame history',
        'training_labels': 'simulator labels used only by auxiliary loss; collected in both arms',
        'evaluation_labels': 'disabled; zero simulator-supervision label reads',
        'architecture_identical': True,
        'only_scientific_difference': 'object_state_supervision_coef: 0.0 versus 0.1',
        'gpu_by_arm': {'state_aux_off': args.gpu0, 'state_aux_on': args.gpu1},
        'source_sha256': source, 'external_input_sha256': inputs,
        'video_root': str(args.video_root.resolve()),
        'cutie_repo': str(args.cutie_repo.resolve()),
        'source_binding_scope': 'all tdmpc2 source/config files, support annotations, Cutie checkpoint; background manifest hashes also checked at evaluation',
        'runs': {
            task: {
                arm: str(repo / 'logs' / task / str(args.seed) / f'{run_tag}_{task}_{arm}')
                for arm in ARMS
            } for task in selected_tasks
        },
        'paper_claim_authorized': False,
    }
    write_new(args.stage / 'protocol.json', payload)
    print('OBJECT_STATE_SUPERVISION_PROTOCOL_RECORDED', flush=True)


def verify_sources(protocol):
    current = source_hashes(Path(protocol['repo_root']))
    if current != protocol['source_sha256']:
        changed = sorted(key for key in set(current) | set(protocol['source_sha256'])
                         if current.get(key) != protocol['source_sha256'].get(key))
        raise ValueError(f'Source changed during paired run: {changed}')
    for path, expected in protocol['external_input_sha256'].items():
        if sha256(path) != expected:
            raise ValueError(f'Bound external input changed: {path}')


def check_equal(actual, expected, context):
    bad = {key: (actual.get(key), value) for key, value in expected.items()
           if actual.get(key) != value}
    if bad:
        raise ValueError(f'{context}: contract mismatch {bad}')


def finite(values, context):
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f'{context}: missing/nonfinite values')
    return values


def reward_stats(values):
    values = finite(values, 'rewards')
    variance = statistics.variance(values) if len(values) > 1 else 0.0
    return {'reward_mean': statistics.mean(values), 'reward_std': math.sqrt(variance),
            'reward_variance': variance, 'reward_median': statistics.median(values),
            'reward_min': min(values), 'reward_max': max(values), 'std_ddof': 1}


def validate_training(task, arm, protocol):
    root = Path(protocol['runs'][task][arm])
    config = load(root / 'runtime_config.json')
    check_equal(config, {
        'task': task, 'obs': 'rgb', 'seed': protocol['seed'], 'model_size': 5,
        'steps': protocol['steps'], 'eval_freq': protocol['eval_freq'],
        'eval_episodes': protocol['eval_episodes'], 'save_eval_episode_trace': True,
        'video_background_enabled': True, 'video_background_split': 'train',
        'flat_anchor': True, 'flat_anchor_mode': 'cutie_object_only',
        'cutie_object_observation_variant': 'full',
        'cutie_object_spatial_token_enabled': True,
        'cutie_object_true_entity_enabled': task == 'acrobot-swingup',
        'object_state_supervision_enabled': True,
        'object_state_supervision_collect_labels': True,
        'object_state_supervision_coef': ARMS[arm],
        'cutie_object_allow_simulator_runtime': False,
        'cutie_object_allow_simulator_kinematics_runtime': False,
        'cutie_object_frame_dim': 590, 'cutie_object_stack_frames': 3,
        'cutie_object_input_dim': 1770, 'cutie_object_only_latent_dim': 128,
        'cutie_object_belief_enabled': False, 'cutie_object_belief_use_for_control': False,
    }, f'{task}/{arm}')
    with (root / 'eval.csv').open(newline='', encoding='utf-8') as stream:
        curve = [{'step': int(float(row['step'])), 'reward': float(row['episode_reward'])}
                 for row in csv.DictReader(stream)]
    expected_steps = list(range(0, protocol['steps'] + 1, protocol['eval_freq']))
    if [row['step'] for row in curve] != expected_steps:
        raise ValueError(f'{root}: evaluation curve incomplete or duplicated')
    finite([row['reward'] for row in curve], root)
    trace = [json.loads(line) for line in (root / 'eval_episodes.jsonl').read_text(
        encoding='utf-8').splitlines() if line.strip()]
    if [int(row['step']) for row in trace] != expected_steps:
        raise ValueError(f'{root}: evaluation episode trace steps mismatch')
    for row, point in zip(trace, curve):
        rewards = finite(row['episode_rewards'], root)
        if len(rewards) != protocol['eval_episodes'] or row.get('episodes') != len(rewards):
            raise ValueError(f'{root}: evaluation episode count mismatch')
        if any(length != 500 for length in row.get('episode_lengths', [])):
            raise ValueError(f'{root}: evaluation episode length mismatch')
        if not math.isclose(statistics.mean(rewards), point['reward'], rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(f'{root}: trace and curve rewards disagree')
    checkpoint = root / 'models' / 'final.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    runtime = load(root / 'trainer_runtime.json')
    if runtime.get('steps') != protocol['steps']:
        raise ValueError(f'{root}: trainer did not finish expected steps')
    auc = sum((b['step'] - a['step']) * (a['reward'] + b['reward']) / 2
              for a, b in zip(curve, curve[1:])) / protocol['steps']
    return config, {
        'root': str(root), 'curve': curve, 'normalized_auc': auc,
        'final_reward': curve[-1]['reward'], 'peak_reward': max(r['reward'] for r in curve),
        'final_eval': reward_stats(trace[-1]['episode_rewards']),
        'trainer_runtime': runtime, 'perception_runtime': load(root / 'perception_runtime.json'),
        'runtime_config_sha256': sha256(root / 'runtime_config.json'),
        'checkpoint_sha256': sha256(checkpoint), 'checkpoint': str(checkpoint),
    }


def validate_evaluation(task, arm, condition, protocol, stage, training):
    path = stage / 'evaluations' / f'{task}_{arm}_{condition}.json'
    payload = load(path)
    check_equal(payload, {'task': task, 'backend': 'cutie_object_only',
                         'condition': condition, 'training_seed': protocol['seed']}, path)
    check_equal(payload['evaluation'], {
        'episodes': 20, 'env_seed': protocol['held_out_env_seed'],
        'background_seed': protocol['held_out_background_seed'],
        'planner_seed_base': protocol['held_out_planner_seed_base'],
        'split': 'clean' if condition == 'clean' else 'validation',
    }, path)
    provenance = payload['provenance']
    check_equal(provenance, {'runtime_config_sha256': training['runtime_config_sha256'],
                            'checkpoint_sha256': training['checkpoint_sha256']}, path)
    evaluation_protocol = payload.get('protocol', {})
    supervision = evaluation_protocol.get('object_state_supervision', {})
    if not isinstance(supervision, dict) or evaluation_protocol.get('state_supervision_collect_labels') is not False:
        raise ValueError(f'{path}: evaluation labels were not explicitly disabled')
    if evaluation_protocol.get('state_supervision_label_reads') != 0:
        raise ValueError(f'{path}: evaluation label-read audit missing or nonzero')
    check_equal(supervision, {'enabled': True, 'coefficient': ARMS[arm],
                             'controller_input_contains_state': False}, path)
    ready = provenance.get('cutie_ready', {})
    check_equal(ready, {
        'true_entity_enabled': task == 'acrobot-swingup',
        'spatial_geometry_source': 'native_binary_tracker_mask_before_pooling_v1',
    }, path)
    episodes = payload['episodes']
    if len(episodes) != 20 or [row['episode_index'] for row in episodes] != list(range(20)):
        raise ValueError(f'{path}: held-out episode count/order mismatch')
    for index, row in enumerate(episodes):
        if row['length'] != 500 or row['planner_seed'] != protocol['held_out_planner_seed_base'] + index:
            raise ValueError(f'{path}: episode length/seed mismatch')
    stats = reward_stats([row['reward'] for row in episodes])
    for key in ('reward_mean', 'reward_std'):
        if not math.isclose(stats[key], payload['summary'][key], rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f'{path}: reward summary inconsistent')
    return payload, {'path': str(path), **stats, 'episodes': episodes,
                     'perception_runtime': payload.get('perception_runtime'),
                     'state_supervision': evaluation_protocol}


def aggregate(args):
    stage = args.stage.resolve()
    protocol = load(stage / 'protocol.json')
    verify_sources(protocol)
    runs, deltas = {}, {}
    for arm in ARMS:
        if (stage / 'queues' / f'{arm}.rc').read_text().strip() != '0':
            raise ValueError(f'{arm}: queue did not complete successfully')
    selected_tasks = tuple(protocol.get('tasks', ()))
    if not selected_tasks or len(selected_tasks) != len(set(selected_tasks)):
        raise ValueError('Protocol task selection is empty or duplicated.')
    if set(selected_tasks) - set(TASKS):
        raise ValueError(f'Protocol contains unsupported tasks: {selected_tasks}')
    for task in selected_tasks:
        runs[task], configurations, evaluation_payloads = {}, {}, {}
        for arm in ARMS:
            for phase in ('train', 'clean', 'hard'):
                rc = stage / 'status' / f'{task}_{arm}_{phase}.rc'
                if rc.read_text().strip() != '0':
                    raise ValueError(f'{rc}: phase did not complete successfully')
            config, training = validate_training(task, arm, protocol)
            configurations[arm] = {key: value for key, value in config.items()
                                   if key not in LOCATION_FIELDS | {'object_state_supervision_coef'}}
            held_out, evaluation_payloads[arm] = {}, {}
            for condition in ('clean', 'hard'):
                raw, result = validate_evaluation(task, arm, condition, protocol, stage, training)
                held_out[condition] = result
                evaluation_payloads[arm][condition] = raw
            runs[task][arm] = {'training': training, 'held_out': held_out}
        if configurations['state_aux_off'] != configurations['state_aux_on']:
            left, right = configurations['state_aux_off'], configurations['state_aux_on']
            changed = {key: (left.get(key), right.get(key)) for key in set(left) | set(right)
                       if left.get(key) != right.get(key)}
            raise ValueError(f'{task}: non-auxiliary configuration differences {changed}')
        for condition in ('clean', 'hard'):
            left, right = (evaluation_payloads[arm][condition] for arm in ARMS)
            for field in ('validation_manifest_sha256', 'combined_manifest_sha256', 'cutie_inputs'):
                if left['provenance'].get(field) != right['provenance'].get(field):
                    raise ValueError(f'{task}/{condition}: paired {field} differs')
            for lrow, rrow in zip(left['episodes'], right['episodes']):
                for field in ('initial_rgb_sha256', 'background_source', 'background_start_frame_index'):
                    if lrow.get(field) != rrow.get(field):
                        raise ValueError(f'{task}/{condition}: paired initial conditions differ: {field}')
        off, on = (runs[task][arm] for arm in ARMS)
        deltas[task] = {
            'aux_on_minus_off_auc': on['training']['normalized_auc'] - off['training']['normalized_auc'],
            'aux_on_minus_off_final': on['training']['final_reward'] - off['training']['final_reward'],
            'held_out_mean': {condition: on['held_out'][condition]['reward_mean'] - off['held_out'][condition]['reward_mean']
                              for condition in ('clean', 'hard')},
        }
    verify_sources(protocol)
    result = {'format': 'object_state_supervision_paired_summary_v1',
              'status': 'object_state_supervision_paired_complete', 'engineering_pass': True,
              'paper_claim_authorized': False, 'protocol': protocol, 'runs': runs, 'deltas': deltas,
              'note': 'One paired training seed is a directional screen, not a scientific superiority claim. Negative returns do not prevent publication of results.'}
    write_new(stage / 'object_state_supervision_paired_summary.json', result)
    print('OBJECT_STATE_SUPERVISION_PAIRED_AGGREGATE_OK', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('record', 'aggregate', 'verify'))
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--run-tag')
    parser.add_argument('--tasks', default=','.join(TASKS))
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--eval-freq', type=int, default=10000)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--gpu0', default='0')
    parser.add_argument('--gpu1', default='1')
    for name in ('video-root', 'cutie-repo', 'cutie-checkpoint', 'support-root-v1', 'support-root-v2'):
        parser.add_argument(f'--{name}', type=Path)
    args = parser.parse_args(argv)
    if args.mode == 'record':
        record(args)
    elif args.mode == 'verify':
        verify_sources(load(args.stage / 'protocol.json'))
        print('OBJECT_STATE_SUPERVISION_SOURCES_VERIFIED', flush=True)
    else:
        aggregate(args)


if __name__ == '__main__':
    main()
