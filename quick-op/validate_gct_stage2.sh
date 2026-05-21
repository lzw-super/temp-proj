#!/bin/bash
# 验证 GCTStream Stage2 训练效果
# 对比原始 lingbot-map 模型 vs Stage2 训练后模型
#
# 复用 validate_gct_stage1.py（因 Stage2 checkpoint 与 Stage1 格式完全一致，
# 都包含 full_model_state_dict）
#
# 用法:
#   bash quick-op/validate_gct_stage2.sh           # 默认20个样本
#   bash quick-op/validate_gct_stage2.sh 50        # 自定义样本数

set -e

DATA_ROOT="/home/shared_files/datasets/dovsg/Replica/room0"
ORIGINAL_MODEL="/home/shared_files/model_weights/linbo_map/lingbot-map.pt"
TRAINED_CKPT="try_train/checkpoints/gct_stage2_5k/checkpoint_final.pt"
NUM_SAMPLES=${1:-20}

echo "============================================================"
echo "GCTStream Stage2 Validation"
echo "============================================================"
echo "  Original:  ${ORIGINAL_MODEL}"
echo "  Stage2:    ${TRAINED_CKPT}"
echo "  Samples:   ${NUM_SAMPLES}"
echo "============================================================"

conda run -n lingbot-map --no-capture-output python try_train/validate_gct_stage1.py \
  --data_root "${DATA_ROOT}" \
  --original_model "${ORIGINAL_MODEL}" \
  --trained_checkpoint "${TRAINED_CKPT}" \
  --num_samples "${NUM_SAMPLES}"
