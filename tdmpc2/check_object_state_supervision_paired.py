"""CPU-only standard-library contracts for the paired runner aggregator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


SPEC = importlib.util.spec_from_file_location(
    'paired_aggregate', Path(__file__).parent / 'tools' / 'aggregate_object_state_supervision_paired.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class PairedAggregationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.stage = self.repo / 'screen.incomplete'
        self.stage.mkdir()
        (self.repo / 'tdmpc2').mkdir()
        (self.repo / 'tdmpc2' / 'source.py').write_text('VALUE = 1\n', encoding='utf-8')
        self.protocol = {
            'repo_root': str(self.repo), 'seed': 12, 'steps': 500,
            'eval_freq': 500, 'eval_episodes': 2,
            'held_out_env_seed': 424243, 'held_out_background_seed': 1618034,
            'held_out_planner_seed_base': 8675400,
            'source_sha256': MODULE.source_hashes(self.repo), 'external_input_sha256': {},
            'runs': {task: {arm: str(self.repo / 'logs' / task / arm) for arm in MODULE.ARMS}
                     for task in MODULE.TASKS},
        }
        write(self.stage / 'protocol.json', self.protocol)
        for arm in MODULE.ARMS:
            path = self.stage / 'queues' / f'{arm}.rc'
            path.parent.mkdir(exist_ok=True)
            path.write_text('0\n', encoding='utf-8')
        for task in MODULE.TASKS:
            for arm, coefficient in MODULE.ARMS.items():
                root = Path(self.protocol['runs'][task][arm])
                config = {
                    'task': task, 'obs': 'rgb', 'seed': 12, 'model_size': 5,
                    'steps': 500, 'eval_freq': 500, 'eval_episodes': 2,
                    'save_eval_episode_trace': True, 'video_background_enabled': True,
                    'video_background_split': 'train', 'flat_anchor': True,
                    'flat_anchor_mode': 'cutie_object_only',
                    'cutie_object_observation_variant': 'full',
                    'cutie_object_spatial_token_enabled': True,
                    'cutie_object_true_entity_enabled': task == 'acrobot-swingup',
                    'object_state_supervision_enabled': True,
                    'object_state_supervision_collect_labels': True,
                    'object_state_supervision_coef': coefficient,
                    'cutie_object_allow_simulator_runtime': False,
                    'cutie_object_allow_simulator_kinematics_runtime': False,
                    'cutie_object_frame_dim': 590, 'cutie_object_stack_frames': 3,
                    'cutie_object_input_dim': 1770, 'cutie_object_only_latent_dim': 128,
                    'cutie_object_belief_enabled': False, 'cutie_object_belief_use_for_control': False,
                    'exp_name': arm, 'work_dir': str(root), 'extra_architecture_field': 7,
                }
                write(root / 'runtime_config.json', config)
                score = 5 if arm == 'state_aux_off' else 1
                (root / 'eval.csv').write_text(
                    f'step,episode_reward\n0,0\n500,{score}\n', encoding='utf-8')
                trace = [{'step': step, 'episodes': 2, 'episode_rewards': [reward, reward],
                          'episode_lengths': [500, 500]} for step, reward in ((0, 0), (500, score))]
                (root / 'eval_episodes.jsonl').write_text(
                    '\n'.join(json.dumps(row) for row in trace), encoding='utf-8')
                (root / 'models').mkdir()
                (root / 'models' / 'final.pt').write_bytes(b'fixture checkpoint')
                write(root / 'trainer_runtime.json', {'steps': 500, 'elapsed_seconds': 10})
                write(root / 'perception_runtime.json', {'frames': 501})
                for phase in ('train', 'clean', 'hard'):
                    path = self.stage / 'status' / f'{task}_{arm}_{phase}.rc'
                    path.parent.mkdir(exist_ok=True)
                    path.write_text('0\n', encoding='utf-8')
                for condition in ('clean', 'hard'):
                    episodes = [{'episode_index': index, 'length': 500,
                                 'planner_seed': 8675400 + index, 'reward': score + index,
                                 'initial_rgb_sha256': f'rgb{index}', 'background_source': condition,
                                 'background_start_frame_index': index} for index in range(20)]
                    payload = {
                        'task': task, 'backend': 'cutie_object_only', 'condition': condition,
                        'training_seed': 12,
                        'evaluation': {'episodes': 20, 'env_seed': 424243,
                                       'background_seed': 1618034, 'planner_seed_base': 8675400,
                                       'split': 'clean' if condition == 'clean' else 'validation'},
                        'provenance': {
                            'runtime_config_sha256': MODULE.sha256(root / 'runtime_config.json'),
                            'checkpoint_sha256': MODULE.sha256(root / 'models' / 'final.pt'),
                            'cutie_ready': {'true_entity_enabled': task == 'acrobot-swingup',
                                            'spatial_geometry_source': 'native_binary_tracker_mask_before_pooling_v1'},
                        },
                        'protocol': {'object_state_supervision': {'enabled': True,
                                      'coefficient': coefficient, 'controller_input_contains_state': False},
                                     'state_supervision_collect_labels': False,
                                     'state_supervision_label_reads': 0},
                        'episodes': episodes, 'summary': MODULE.reward_stats([row['reward'] for row in episodes]),
                        'perception_runtime': {'frames': 10020},
                    }
                    write(self.stage / 'evaluations' / f'{task}_{arm}_{condition}.json', payload)

    def aggregate(self):
        MODULE.aggregate(SimpleNamespace(stage=self.stage))

    def test_complete_negative_result_is_published(self):
        self.aggregate()
        result = MODULE.load(self.stage / 'object_state_supervision_paired_summary.json')
        self.assertTrue(result['engineering_pass'])
        self.assertFalse(result['paper_claim_authorized'])
        self.assertEqual(result['deltas']['acrobot-swingup']['aux_on_minus_off_auc'], -2)
        stats = result['runs']['acrobot-swingup']['state_aux_on']['held_out']['clean']
        self.assertAlmostEqual(stats['reward_variance'], stats['reward_std'] ** 2)

    def test_nonzero_evaluation_label_reads_rejected(self):
        path = self.stage / 'evaluations' / 'acrobot-swingup_state_aux_on_clean.json'
        value = MODULE.load(path)
        value['protocol']['state_supervision_label_reads'] = 1
        write(path, value)
        with self.assertRaisesRegex(ValueError, 'label-read'):
            self.aggregate()

    def test_architecture_difference_rejected(self):
        root = Path(self.protocol['runs']['acrobot-swingup']['state_aux_on'])
        config = MODULE.load(root / 'runtime_config.json')
        config['extra_architecture_field'] = 8
        write(root / 'runtime_config.json', config)
        for condition in ('clean', 'hard'):
            path = self.stage / 'evaluations' / f'acrobot-swingup_state_aux_on_{condition}.json'
            value = MODULE.load(path)
            value['provenance']['runtime_config_sha256'] = MODULE.sha256(root / 'runtime_config.json')
            write(path, value)
        with self.assertRaisesRegex(ValueError, 'non-auxiliary'):
            self.aggregate()

    def test_changed_source_rejected(self):
        (self.repo / 'tdmpc2' / 'source.py').write_text('VALUE = 2\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            self.aggregate()

    def test_incomplete_queue_rejected(self):
        (self.stage / 'queues' / 'state_aux_on.rc').write_text('1\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'queue did not complete'):
            self.aggregate()

    def test_unpaired_initial_image_rejected(self):
        path = self.stage / 'evaluations' / 'acrobot-swingup_state_aux_on_clean.json'
        value = MODULE.load(path)
        value['episodes'][1]['initial_rgb_sha256'] = 'changed'
        write(path, value)
        with self.assertRaisesRegex(ValueError, 'paired initial conditions'):
            self.aggregate()


if __name__ == '__main__':
    unittest.main()
