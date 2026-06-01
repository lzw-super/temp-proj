#!/bin/bash
# GCTStream Head-Only Stage1 训练脚本
# 基于完整GCTStream模型，冻结aggregator，只训练depth_head + camera_head
#
# 用法:
#   bash quick-op/train_gct_stage1.sh           # 默认5000 iterations
#   bash quick-op/train_gct_stage1.sh 1000      # 自定义iterations

set -e

# ============== 配置 ==============
DATA_ROOT="/home/shared_files/datasets/dovsg/Replica/room0"
CHECKPOINT="/home/shared_files/model_weights/linbo_map/lingbot-map.pt"
OUTPUT_DIR="try_train/checkpoints/gct_stage1_5k"
TOTAL_ITERS=${1:-5000}
LOG_EVERY=50
SAVE_EVERY=1000
# ================================

echo "============================================================"
echo "GCTStream Head-Only Stage1 Training"
echo "============================================================"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Iterations: ${TOTAL_ITERS}"
echo "  Log every:  ${LOG_EVERY}"
echo "  Save every: ${SAVE_EVERY}"
echo "============================================================"

conda run -n lingbot-map --no-capture-output python try_train/train_replica_gct.py \
  --data_root "${DATA_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --total_iterations "${TOTAL_ITERS}" \
  --log_every "${LOG_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --random_init_heads \
  --use_sdpa 
