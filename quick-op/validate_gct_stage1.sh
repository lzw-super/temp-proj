#!/bin/bash
# 验证 GCTStream Head-Only Stage1 训练效果
# 对比原始模型 vs 训练后模型

set -e

DATA_ROOT="/home/shared_files/datasets/dovsg/Replica/room0"
ORIGINAL_MODEL="/home/shared_files/model_weights/linbo_map/lingbot-map.pt"
TRAINED_CKPT="try_train/checkpoints/gct_stage1_5k/checkpoint_final.pt"
NUM_SAMPLES=${1:-20}

echo "============================================================"
echo "GCTStream Stage1 Validation"
echo "============================================================"
echo "  Original:  ${ORIGINAL_MODEL}"
echo "  Trained:   ${TRAINED_CKPT}"
echo "  Samples:   ${NUM_SAMPLES}"
echo "============================================================"

conda run -n lingbot-map --no-capture-output python try_train/validate_gct_stage1.py \
  --data_root "${DATA_ROOT}" \
  --original_model "${ORIGINAL_MODEL}" \
  --trained_checkpoint "${TRAINED_CKPT}" \
  --num_samples "${NUM_SAMPLES}"
