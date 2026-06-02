#!/bin/bash
# GCTStream Head-Only Stage1 训练脚本
# 基于完整GCTStream模型，冻结aggregator，只训练depth_head + camera_head
#
# 用法:
#   bash quick-op/train_gct_stage1.sh                           # 默认配置
#   bash quick-op/train_gct_stage1.sh 1000                      # 自定义iterations
#   bash quick-op/train_gct_stage1.sh 1000 2 24 spatial_nearby  # 全部自定义
#
# 输出目录命名: gct_stage1_{min_views}-{max_views}_{sampler_type}_{iters}iters

set -e

# ============== 配置 ==============
DATA_ROOT="/home/shared_files/datasets/dovsg/Replica/room0"
CHECKPOINT="/home/shared_files/model_weights/linbo_map/lingbot-map.pt"
TOTAL_ITERS=${1:-5000}
MIN_VIEWS=${2:-2}
MAX_VIEWS=${3:-20}
SAMPLER_TYPE=${4:-"spatial_nearby"}
LOG_EVERY=50
SAVE_EVERY=1000

# 动态生成输出目录名
OUTPUT_DIR="try_train/checkpoints/gct_stage1_${MIN_VIEWS}-${MAX_VIEWS}_${SAMPLER_TYPE}_${TOTAL_ITERS}iters"
# ================================

echo "============================================================"
echo "GCTStream Head-Only Stage1 Training"
echo "============================================================"
echo "  Checkpoint:   ${CHECKPOINT}"
echo "  Output:       ${OUTPUT_DIR}"
echo "  Iterations:   ${TOTAL_ITERS}"
echo "  Views:        ${MIN_VIEWS}-${MAX_VIEWS}"
echo "  Sampler:      ${SAMPLER_TYPE}"
echo "  Log every:    ${LOG_EVERY}"
echo "  Save every:   ${SAVE_EVERY}"
echo "============================================================"

conda run -n lingbot-map --no-capture-output python try_train/train_replica_gct.py \
  --data_root "${DATA_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --total_iterations "${TOTAL_ITERS}" \
  --log_every "${LOG_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --min_views "${MIN_VIEWS}" \
  --max_views "${MAX_VIEWS}" \
  --sampler_type "${SAMPLER_TYPE}" \
  --random_init_heads \
  --use_sdpa 
