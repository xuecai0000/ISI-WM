# ISI-WM

Code for ISI-WM.

## Setup

Python 3.11, PyTorch 2.x (CUDA), dm-control 1.0.16, mujoco 3.1.2,
imageio, imageio-ffmpeg.

```bash
pip install torch dm-control==1.0.16 mujoco==3.1.2 imageio imageio-ffmpeg
```

## Layout

```
configs/          configuration files
tools/            evaluation script
envs/wrappers/    video-background compositor
data_protocol/    background data split and pairing notes
```

## Evaluation

```bash
python -m tools.evaluate_cutie_multitask_checkpoint \
    --task cup-catch --backend rgb \
    --condition hard --background-split test \
    --training-condition hard \
    --runtime-config <run_dir>/runtime_config.json \
    --checkpoint <run_dir>/models/final.pt \
    --training-seed 6 --env-seed 424243 --background-seed 1618034 \
    --episodes 20 --output out.json
```

Conditions: `--condition clean`, `--condition hard --background-split train` (seen),
`--condition hard --background-split test` (unseen).

## Data

See `data_protocol/` for the video pool split and compositor specification.
