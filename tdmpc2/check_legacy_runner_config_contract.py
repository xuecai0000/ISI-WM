"""Exercise generated run YAML through actual Hydra struct-mode parsing on CPU."""
import json
from pathlib import Path
import tempfile

from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from common.parser import parse_cfg
from tools.run_legacy_object_regression import ARMS, REPO, old_root, prepare_config


def main():
    keys = ('seed_steps', 'batch_size', 'lr', 'horizon', 'rho', 'num_samples',
            'iterations', 'compile', 'compile_fallback_random',
            'flat_anchor_reconstruction_coef', 'flat_anchor_prediction_coef',
            'flat_anchor_loss_beta', 'cutie_object_support_path',
            'video_background_root', 'video_background_manifest_dir',
            'eval_freq', 'eval_episodes')
    with tempfile.TemporaryDirectory(prefix='runner_config_contract_', dir=REPO) as tmp:
        for task in ('finger-spin', 'cup-catch', 'cartpole-swingup'):
            old = json.loads((old_root(task) / 'runtime_config.json').read_text())
            for arm in (ARMS if task == 'finger-spin' else ('legacy_full',)):
                path, root = prepare_config(task, 6, arm, 'config_contract_only', Path(tmp))
                with initialize_config_dir(config_dir=str(path.parent), version_base='1.1'):
                    composed = compose(config_name=path.stem, return_hydra_config=True)
                    HydraConfig.instance().set_config(composed)
                    # Hydra passes the task config to train(), without its own
                    # internal hydra node, and enables struct-mode there.
                    data = OmegaConf.to_container(composed, resolve=False)
                    data.pop('hydra')
                    cfg = OmegaConf.create(data)
                    OmegaConf.set_struct(cfg, True)
                    parsed = parse_cfg(cfg)
                for key in keys:
                    assert getattr(parsed, key) == old[key], (task, arm, key)
                assert parsed.work_dir == root, (parsed.work_dir, root)
                assert parsed.bin_size > 0 and parsed.task_title
                assert parsed.cutie_object_spatial_token_enabled is False
                assert parsed.cutie_object_variable_graph_enabled is False
                print('HYDRA_RUNNER_CONFIG_OK', task, arm, flush=True)


if __name__ == '__main__':
    main()
