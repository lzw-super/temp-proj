#!/bin/bash
# Validate LZW-Map Stage1 outputs against init, original LingBot-Map, and GT.
#
# Usage:
#   bash quick-op/validate_lzw_map_stage1.sh
#   bash quick-op/validate_lzw_map_stage1.sh 20 path/to/checkpoint_final.pt
#   bash quick-op/validate_lzw_map_stage1.sh 20 path/to/checkpoint_final.pt 4
#
# Environment variables:
#   DINOV2_REPO  可改：local original DINOv2 repository path
#
# 可调参数说明：
#   - NUM_SAMPLES 控制验证样本数；越大统计越稳，耗时越长。
#   - NUM_VIEWS 控制每个验证 sample 的 view 数；建议与训练/目标推理 views 对齐。
#   - NUM_VIS 只控制输出图片里可视化多少个 sample，不影响指标统计。
#   - ORIGINAL_MODEL 用于和原始 LingBot-Map 做对比；只看 LZW 自身时也仍需传入脚本。
#   - Stage1 validation 固定使用 temporal_nearby、无数据增强、max_dim=518。
#   - 位姿指标会输出 Official XYZW AUC@3/5/15/30、Racc/Tacc、Sim(3) ATE 和 RPE；
#     AUC 越高越好，ATE/RPE 越低越好。
#   - scale_aligned_abs/rel_pose_loss 使用官方 XYZW loss；legacy_wxyz_* 仅作为旧 loss 诊断。
#   - NUM_VIEWS < 3 时 Sim(3) ATE 和 pairwise AUC 统计都较弱，位姿更建议看 Official XYZW RPE 指标。

set -e

DATA_ROOT="/home/shared_files/datasets/dovsg/Replica/room0"  # 可改：Replica 验证数据路径
ORIGINAL_MODEL="/home/shared_files/model_weights/linbo_map/lingbot-map.pt"  # 可改：原始 LingBot-Map 权重
DINOV2_REPO="${DINOV2_REPO:-/home/lizhengwu/desktop/temp_proj/dinov2}"  # 可改：也可用环境变量覆盖
NUM_SAMPLES=${1:-20}  # 可改：验证样本数；快速检查可 5/10，正式对比建议 20+
TRAINED_CKPT=${2:-"try_train/checkpoints/lzw_map_stage1_frozen_dinov2_2-20_spatial_nearby_xyzw_5000iters/checkpoint_final.pt"}  # 可改：Stage1 checkpoint
NUM_VIEWS=${3:-20}  # 可改：每个样本 views；建议与训练上限或目标推理 views 对齐
NUM_VIS=${4:-5}  # 可改：保存可视化样本数；不影响指标

OUTPUT_DIR="$(dirname "${TRAINED_CKPT}")/vis_validation"  # 可改：验证结果输出目录

echo "========================================================================"
echo "LZW-Map Stage1 Validation"
echo "========================================================================"
echo "  Dataset:       ${DATA_ROOT}"
echo "  LingBot orig:  ${ORIGINAL_MODEL}"
echo "  LZW trained:   ${TRAINED_CKPT}"
echo "  DINOv2 repo:   ${DINOV2_REPO}"
echo "  Samples:       ${NUM_SAMPLES}"
echo "  Views:         ${NUM_VIEWS}"
echo "  Visual samples:${NUM_VIS}"
echo "  Output:        ${OUTPUT_DIR}"
echo "  Pose metrics:  AUC@3/5/15/30, ATE, RPE (Official XYZW)"
echo "========================================================================"

conda run -n lingbot-map --no-capture-output python try_train/validate_lzw_map_stage1.py \
  --data_root "${DATA_ROOT}" \
  --original_model "${ORIGINAL_MODEL}" \
  --trained_checkpoint "${TRAINED_CKPT}" \
  --dinov2_repo "${DINOV2_REPO}" \
  --num_samples "${NUM_SAMPLES}" \
  --num_views "${NUM_VIEWS}" \
  --num_vis "${NUM_VIS}" \
  --output_dir "${OUTPUT_DIR}" \
  --use_sdpa
