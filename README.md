# ISI-WM

Training and evaluation code for ISI-WM, a pixel-based world-model agent
trained with two training-time interventional objectives (background
invariance and outcome-grounded action sensitivity) on top of TD-MPC2.

Reference pseudocode for both objectives is provided in the paper (Algorithm 1).
The complete training implementation will be released upon acceptance.

## Setup

Python 3.11, PyTorch 2.x (CUDA), dm-control 1.0.16, mujoco 3.1.2,
imageio, imageio-ffmpeg.

```bash
pip install torch dm-control==1.0.16 mujoco==3.1.2 imageio imageio-ffmpeg
```

## Layout

```
configs/          training configuration (all tasks share one setting;
                  fully specifies the interventional objective constants)
tools/            evaluation script (reproduces all reported numbers)
envs/wrappers/    video-background compositor
data_protocol/    background data split, compositor spec, pairing contract
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

Conditions: `--condition clean` (no video), `--condition hard
--background-split train` (seen), `--condition hard --background-split test`
(unseen).

## Data

See `data_protocol/` for the 100-video pool split (train 0-79 /
val 80-84 / test 90-99), blue-key compositor specification, and the
intervention pairing contract.
