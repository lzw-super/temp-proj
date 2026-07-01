#!/bin/bash
# LZW-Map Stage1 training with a frozen original DINOv2 ViT-B backbone.
#
# Usage:
#   bash quick-op/train_lzw_map_stage1.sh
#   bash quick-op/train_lzw_map_stage1.sh 200
#   bash quick-op/train_lzw_map_stage1.sh 5000 2 4 spatial_nearby 2e-4 0.1 0.05 500
#
# Environment variables:
#   DINOV2_REPO  可改：local original DINOv2 repository path
#   RESUME       可改：optional LZW-Map checkpoint path
#
# 48GB 显存调参提示：
#   - 当前默认 2-20 views 接近 24GB 级别配置；48GB 可优先试 2-24 或 2-32。
#   - Stage1 使用 global attention，一次处理全部 views；显存会随 MAX_VIEWS 明显上涨。
#   - 当前 quick 脚本没有暴露 --batch_size。Python 入口支持 batch_size，但由于
#     min_views/max_views 会随机产生不同 view 数，batch_size > 1 建议只在
#     MIN_VIEWS == MAX_VIEWS 时使用，或先实现 padding/collate。
#   - 优先调 MAX_VIEWS，再考虑 batch_size；不要一开始同时大幅增加。

set -e

DATA_ROOT="/home/shared_files/datasets/dovsg/Replica/room0"  # 可改：训练数据路径
DINOV2_REPO="${DINOV2_REPO:-/home/lizhengwu/desktop/temp_proj/dinov2}"  # 可改：也可用环境变量覆盖
TOTAL_ITERS=${1:-5000}                 # 可改：总训练步数；正式训练可 20000+
MIN_VIEWS=${2:-2}                      # 可改：每个 sample 最少 views；通常保持 2
MAX_VIEWS=${3:-20}                     # 可改：最吃显存；48GB 建议先试 24/32
SAMPLER_TYPE=${4:-"spatial_nearby"}    # 可改：spatial_nearby 或 temporal_nearby
LR=${5:-"2e-4"}                        # 可改：学习率；增大 views 后建议先保持 2e-4
POSE_WEIGHT=${6:-"0.1"}                # 可改：absolute pose loss 权重
REL_POSE_WEIGHT=${7:-"0.05"}           # 可改：relative pose loss 权重
REL_POSE_START=${8:-500}               # 可改：relative pose loss 启用步数；长 views 可试 1000/2000
LOG_EVERY=100                          # 可改：日志间隔，不影响训练结果
SAVE_EVERY=1000                        # 可改：checkpoint 间隔；越小占磁盘越多

OUTPUT_DIR="try_train/checkpoints/lzw_map_stage1_frozen_dinov2_${MIN_VIEWS}-${MAX_VIEWS}_${SAMPLER_TYPE}_${TOTAL_ITERS}iters"  # 可改：输出目录命名

RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
  if [[ ! -f "${RESUME}" ]]; then
    echo "ERROR: RESUME=${RESUME} does not exist" >&2
    exit 1
  fi
  RESUME_ARGS+=(--resume "${RESUME}")
fi

echo "========================================================================"
echo "LZW-Map Stage1 Training"
echo "========================================================================"
echo "  Dataset:          ${DATA_ROOT}"
echo "  DINOv2 repo:      ${DINOV2_REPO}"
echo "  Backbone:         dinov2_vitb14_reg, pretrained, frozen"
echo "  Trainable:        LZW aggregator blocks + camera head + depth head"
echo "  Views:            ${MIN_VIEWS}-${MAX_VIEWS}"
echo "  Sampler:          ${SAMPLER_TYPE}"
echo "  Learning rate:    ${LR}"
echo "  Fused loss:       depth + ${POSE_WEIGHT}*abs_pose + ${REL_POSE_WEIGHT}*rel_pose"
echo "  Rel pose starts:  iteration ${REL_POSE_START}"
echo "  Iterations:       ${TOTAL_ITERS}"
echo "  Output:           ${OUTPUT_DIR}"
if [[ -n "${RESUME:-}" ]]; then
  echo "  Resume from:      ${RESUME}"
fi
echo "========================================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n lingbot-map --no-capture-output python try_train/train_lzw_map_stage1.py \
  --data_root "${DATA_ROOT}" \
  --dinov2_repo "${DINOV2_REPO}" \
  --output_dir "${OUTPUT_DIR}" \
  --total_iterations "${TOTAL_ITERS}" \
  --min_views "${MIN_VIEWS}" \
  --max_views "${MAX_VIEWS}" \
  --sampler_type "${SAMPLER_TYPE}" \
  --lr "${LR}" \
  --pose_weight "${POSE_WEIGHT}" \
  --rel_pose_weight "${REL_POSE_WEIGHT}" \
  --rel_pose_start_iter "${REL_POSE_START}" \
  --log_every "${LOG_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --use_sdpa \
  "${RESUME_ARGS[@]}"
