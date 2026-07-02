#!/bin/bash
# Validate LZW-Map Stage2 with strict GCA streaming forward.
#
# Usage:
#   bash quick-op/validate_lzw_map_stage2.sh
#   bash quick-op/validate_lzw_map_stage2.sh 20 path/to/checkpoint_final.pt
#   bash quick-op/validate_lzw_map_stage2.sh 20 path/to/checkpoint_final.pt 16 4 8 5
#
# Positional args:
#   1 NUM_SAMPLES       validation samples, default 20
#   2 STAGE2_CKPT       Stage2 checkpoint path
#   3 NUM_VIEWS         views per sample, default 16
#   4 STAGE2_K          GCA local window k, default 4
#   5 ANCHOR_FRAMES     Stage2 anchor frames, default 8
#   6 NUM_VIS           visualized samples, default 5
#
# Environment variables:
#   DINOV2_REPO         可改：local original DINOv2 repository path
#
# 可调参数说明：
#   - NUM_SAMPLES 控制验证样本数；越大统计越稳，耗时越长。
#   - NUM_VIEWS 控制每个验证 sample 的长序列 views；建议与 Stage2 训练 views_end
#     或目标部署序列长度对齐。
#   - STAGE2_K 是验证时 GCA streaming 的 local window；应与训练 k_max 或目标推理
#     window 对齐。k 不宜超过 NUM_VIEWS - 1。
#   - ANCHOR_FRAMES 是前多少帧作为 scale/anchor context；通常保持 8。
#   - NUM_VIS 只控制输出图片里可视化多少个 sample，不影响指标统计。
#   - Python 入口还支持 --max_frame_num，当前 quick 脚本未暴露，默认 400；
#     如果 NUM_VIEWS > 400，需要同步扩展脚本传入更大的 --max_frame_num。
#   - Stage2 validation 固定使用 temporal_nearby、无数据增强、max_dim=518。
#   - 位姿指标会输出 Official XYZW AUC@3/5/15/30、Racc/Tacc、Sim(3) ATE 和 RPE；
#     AUC 越高越好，ATE/RPE 越低越好。
#   - scale_aligned_abs/rel_pose_loss 使用官方 XYZW loss；legacy_wxyz_* 仅作为旧 loss 诊断。
#   - Stage2 的 pose anchor-scale norm 是训练期选项；验证时不用额外开关，
#     只需传入对应 poseNorm1/poseNorm0 checkpoint 路径。

set -e

DATA_ROOT="/home/shared_files/datasets/dovsg/Replica/room0"  # 可改：Replica 验证数据路径
ORIGINAL_MODEL="/home/shared_files/model_weights/linbo_map/lingbot-map.pt"  # 可改：原始 LingBot-Map 权重
DINOV2_REPO="${DINOV2_REPO:-/home/lizhengwu/desktop/temp_proj/dinov2}"  # 可改：也可用环境变量覆盖
DEFAULT_STAGE2_CKPT="try_train/checkpoints/lzw_map_stage2_full_15-20_k4-8_xyzw_poseNorm1_5000iters/checkpoint_final.pt"  # 可改：默认 Stage2 checkpoint

NUM_SAMPLES=${1:-20}  # 可改：验证样本数；快速检查可 5/10，正式对比建议 20+
STAGE2_CKPT=${2:-"${DEFAULT_STAGE2_CKPT}"}  # 可改：Stage2 checkpoint
NUM_VIEWS=${3:-20}  # 可改：验证长序列 views；48GB 训练后可试 64/96/128
STAGE2_K=${4:-8}  # 可改：验证 GCA local window；建议对齐训练 k_max，如 32/64
ANCHOR_FRAMES=${5:-8}  # 可改：anchor/scale frames；通常保持 8
NUM_VIS=${6:-5}  # 可改：保存可视化样本数；不影响指标

OUTPUT_DIR="$(dirname "${STAGE2_CKPT}")/vis_validation_stage2"  # 可改：验证结果输出目录

if [[ ! -f "${STAGE2_CKPT}" ]]; then
  echo "ERROR: STAGE2_CKPT=${STAGE2_CKPT} does not exist" >&2
  exit 1
fi

echo "========================================================================"
echo "LZW-Map Stage2 Validation (GCA streaming)"
echo "========================================================================"
echo "  Dataset:        ${DATA_ROOT}"
echo "  LingBot orig:   ${ORIGINAL_MODEL}"
echo "  LZW Stage2:     ${STAGE2_CKPT}"
echo "  DINOv2 repo:    ${DINOV2_REPO}"
echo "  Samples:        ${NUM_SAMPLES}"
echo "  Views:          ${NUM_VIEWS}"
echo "  GCA window k:   ${STAGE2_K}"
echo "  Anchor frames:  ${ANCHOR_FRAMES}"
echo "  Visual samples: ${NUM_VIS}"
echo "  Output:         ${OUTPUT_DIR}"
echo "  Pose metrics:   AUC@3/5/15/30, ATE, RPE (Official XYZW)"
echo "========================================================================"

conda run -n lingbot-map --no-capture-output python try_train/validate_lzw_map_stage2.py \
  --data_root "${DATA_ROOT}" \
  --original_model "${ORIGINAL_MODEL}" \
  --stage2_checkpoint "${STAGE2_CKPT}" \
  --dinov2_repo "${DINOV2_REPO}" \
  --num_samples "${NUM_SAMPLES}" \
  --num_views "${NUM_VIEWS}" \
  --num_vis "${NUM_VIS}" \
  --stage2_sliding_window "${STAGE2_K}" \
  --stage2_num_frame_for_scale "${ANCHOR_FRAMES}" \
  --output_dir "${OUTPUT_DIR}" \
  --use_sdpa
