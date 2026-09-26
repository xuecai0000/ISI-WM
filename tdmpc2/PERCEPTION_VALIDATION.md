# Visual-Small Color-multi 感知验证

这套验证只比较感知，不训练强化学习。DINO 与 Cutie 都读取同一批原生
`64x64` RGB；二者内部的 `448x448` 只算插值，不得称为高分辨率输入。
审计只使用 train/validation 视频，test split 完全封存。

## 1. 服务器变量与静态检查

```bash
cd /home/<USER>/world/tdmpc2_2026

PY=<DATA_PATH>/conda_envs/tdmpc2_2026/bin/python
VIDEO_ROOT=<BASELINE_PATH>/HRSSM-main/env/data/video_hard
MANIFEST_DIR=/home/<USER>/world/tdmpc2_2026/tdmpc2/envs/background_manifests
SUPPORT=<DATA_PATH>/tdmpc2_2026/datasets/flat_anchor/reacher_visual_small_color_support_seed314159/annotations.json
AUDIT=<DATA_PATH>/tdmpc2_2026/datasets/perception/reacher_visual_small_color_audit_seed271828_v1
OUT=<DATA_PATH>/tdmpc2_2026/results/perception/reacher_visual_small_native64_dino_vs_cutie_v1

"$PY" tdmpc2/check_visual_small_perception_audit.py
"$PY" tdmpc2/check_visual_small_perception_ab_contract.py
"$PY" tdmpc2/check_cutie_oc_adapter_contract.py
```

## 2. 采集无泄漏审计帧

输出目录应是一个新的目录；不要用旧审计结果覆盖新实验。

```bash
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl "$PY" \
  tdmpc2/tools/collect_visual_small_perception_audit.py \
  --output "$AUDIT" \
  --video-root "$VIDEO_ROOT" \
  --manifest-dir "$MANIFEST_DIR"

"$PY" tdmpc2/check_visual_small_perception_audit.py \
  --audit "$AUDIT/annotations.json"

"$PY" tdmpc2/tools/annotate_visual_small_perception_audit.py \
  --audit "$AUDIT/annotations.json" \
  --output "$AUDIT/annotate_points.html"
```

把 `annotate_points.html` 复制到本机浏览器打开。它内嵌 24 张图；每张只需
依次点击 elbow、control tip、goal，base 自动固定为 `[31.5,31.5]`。
点击导出后，把 `annotations.completed.json` 放回 `$AUDIT`，再执行：

```bash
"$PY" tdmpc2/check_visual_small_perception_audit.py \
  --audit "$AUDIT/annotations.completed.json" \
  --require-manual-labels
```

## 3. 获取官方 OC-STORM Cutie 代码与唯一需要的权重

本实验不需要 OC-STORM 的预生成分割数据，也不需要 RITM 权重。现有四点
support pack 会被确定性栅格化为 proximal link、distal link、goal 三个对象。

```bash
OC_REPO=<DATA_PATH>/r2_hrssm_third_party/OC-STORM
CUTIE_CKPT="$OC_REPO/feature_extractor/cutie/weights/cutie-small-mega.pth"

if [ ! -d "$OC_REPO/.git" ]; then
  git clone --depth 1 https://github.com/weipu-zhang/OC-STORM.git "$OC_REPO"
fi
mkdir -p "$(dirname "$CUTIE_CKPT")"
if [ ! -s "$CUTIE_CKPT" ]; then
  wget -c -O "$CUTIE_CKPT.part" \
    https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-small-mega.pth \
    && mv "$CUTIE_CKPT.part" "$CUTIE_CKPT"
fi

git -C "$OC_REPO" rev-parse HEAD
sha256sum "$CUTIE_CKPT"
```

先使用现有 TD-MPC2 环境做严格预检；不要直接安装 OC-STORM 的完整训练依赖
并升级当前 PyTorch：

```bash
CUDA_VISIBLE_DEVICES=1 "$PY" -m tdmpc2.tools.check_cutie_oc_preflight \
  --oc-storm-repo "$OC_REPO" \
  --checkpoint "$CUTIE_CKPT" \
  --support-annotations "$SUPPORT" \
  --tracker-size 448 448 \
  --device cuda:0 \
  --sha256
```

如果这里报告缺依赖，保存完整报错，只补它明确指出的依赖；不要运行整个
OC-STORM `requirements.txt` 覆盖当前环境。

## 4. 同图 DINO/Cutie A/B

```bash
CUDA_VISIBLE_DEVICES=1 "$PY" \
  tdmpc2/tools/evaluate_visual_small_perception_ab.py \
  --annotations "$AUDIT/annotations.completed.json" \
  --output-dir "$OUT" \
  --anchor-config /home/<USER>/world/tdmpc2_2026/tdmpc2/config.yaml \
  --dino-impl <BASELINE_PATH>/r2dreamer-main/anchor_state.py \
  --dino-support "$SUPPORT" \
  --dino-device cuda:0 \
  --dino-repo <DATA_PATH>/r2_hrssm_third_party/dinov2 \
  --dino-checkpoint <DATA_PATH>/r2_hrssm_third_party/checkpoints/dinov2_vits14_reg4_pretrain.pth \
  --cutie-oc-repo "$OC_REPO" \
  --cutie-checkpoint "$CUTIE_CKPT" \
  --cutie-support "$SUPPORT" \
  --cutie-device cuda:0 \
  --cutie-tracker-size 448 448
```

退出码 `0` 表示至少一个后端通过绝对质量门，退出码 `2` 表示科学门禁失败，
但两种情况下都会先写出：

- `$OUT/report.json`
- `$OUT/metrics.csv`
- `$OUT/overlays/`（24 张人工真值/DINO/Cutie叠加图）
- `$OUT/dino_predictions.json`
- `$OUT/cutie_predictions.json`
- `$OUT/prediction_assets/`（Cutie mask 与 2048D OC 对象特征）

最终三态结论是 `cutie_selected`、`keep_dino` 或
`perception_not_ready`。最后一种状态禁止继续训练 RL；下一步应先验证真正的
高分辨率观测，而不是继续修改融合头。
