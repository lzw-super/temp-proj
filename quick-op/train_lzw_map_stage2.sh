#!/bin/bash
# LZW-Map Stage2 training with GCA streaming.
#
# Usage:
#   bash quick-op/train_lzw_map_stage2.sh
#   bash quick-op/train_lzw_map_stage2.sh 200
#   bash quick-op/train_lzw_map_stage2.sh 5000 path/to/stage1_checkpoint.pt
#   bash quick-op/train_lzw_map_stage2.sh 5000 path/to/stage1_checkpoint.pt 15 20 4 8 1e-4 0.1 0.05 500 xyzw 1
#   RESUME=auto bash quick-op/train_lzw_map_stage2.sh 5000
#
# Resume examples:
#   # 自动从当前 OUTPUT_DIR 中选择最新 checkpoint_iter_*.pt 续训；参数要和原训练保持一致。
#   RESUME=auto bash quick-op/train_lzw_map_stage2.sh 5000 \
#     try_train/checkpoints/lzw_map_stage1_frozen_dinov2_2-20_spatial_nearby_xyzw_5000iters/checkpoint_final.pt \
#     15 20 4 8 1e-4 0.1 0.05 500 xyzw 1
#
#   # 或者显式指定断点；TOTAL_ITERS 是最终总步数，不是追加步数。
#   RESUME=try_train/checkpoints/lzw_map_stage2_full_15-20_k4-8_xyzw_poseNorm1_5000iters/checkpoint_iter_3000.pt \
#     bash quick-op/train_lzw_map_stage2.sh 5000 \
#     try_train/checkpoints/lzw_map_stage1_frozen_dinov2_2-20_spatial_nearby_xyzw_5000iters/checkpoint_final.pt \
#     15 20 4 8 1e-4 0.1 0.05 500 xyzw 1
#
# Positional args:
#   11 POSE_QUAT_CONVENTION  xyzw 官方 loss；wxyz 复现旧本地 loss
#   12 POSE_ANCHOR_NORM      1/0，是否把 Stage2 GT pose translation 除以 anchor scale
#
# Environment variables:
#   DINOV2_REPO  可改：local original DINOv2 repository path
#   STAGE1_CKPT  可改：optional Stage1 LZW-Map checkpoint path
#   RESUME       可改：optional Stage2 checkpoint path, or "auto"
#   HEAD_ONLY    可改：set to 1 to freeze the whole aggregator and train heads only
#   LOG_FILE     可改：保存完整训练终端日志的路径；默认写到 OUTPUT_DIR/logs/train_时间戳.log
#
# 48GB 显存调参提示：
#   - 已有 15->20 views, k=4->8, full 模式记录约 22.3 GiB allocated，
#     接近 24GB；48GB 可优先试 24->64, k=8->32。
#   - 如果 24->64 稳定，再试 24->96 或 24->128, k=16->64。
#   - VIEWS_END 必须 <= Python 入口的 --max_frame_num；当前默认 max_frame_num=400，
#     所以 VIEWS_END <= 400 不需要额外修改。
#   - 当前 quick 脚本没有暴露 --batch_size。Stage2 的 batch_size > 1 比 Stage1 更可行，
#     但仍建议先扩大 views/k，稳定后再通过 Python 命令或扩展本脚本尝试 --batch_size 2。
#   - POSE_ANCHOR_NORM 只控制 pose translation 的 anchor-scale 归一化；
#     depth anchor-scale normalization 仍保持开启，用于匹配当前 Stage2 默认训练路径。

set -e

DATA_ROOT="/home/shared_files/datasets/dovsg/Replica/room0"  # 可改：训练数据路径
DINOV2_REPO="${DINOV2_REPO:-/home/lizhengwu/desktop/temp_proj/dinov2}"  # 可改：也可用环境变量覆盖
DEFAULT_STAGE1_CKPT="try_train/checkpoints/lzw_map_stage1_frozen_dinov2_2-20_spatial_nearby_xyzw_5000iters/checkpoint_final.pt"  # 可改：默认 Stage1 初始化权重
TOTAL_ITERS=${1:-5000}                 # 可改：总训练步数；长序列正式训练可 50000+
STAGE1_CKPT=${2:-${STAGE1_CKPT:-${DEFAULT_STAGE1_CKPT}}}  # 可改：Stage1 checkpoint
VIEWS_START=${3:-15}                   # 可改：curriculum 起始 views；48GB 可试 24
VIEWS_END=${4:-20}                     # 可改：curriculum 目标 views；48GB 先试 64，再试 96/128
K_MIN=${5:-4}                          # 可改：GCA/local relative pose 最小窗口；48GB 可试 8/16
K_MAX=${6:-8}                          # 可改：GCA/local relative pose 最大窗口；48GB 可试 32/64
LR=${7:-"1e-4"}                        # 可改：学习率；full 模式可先试 1e-4/2e-4，head-only 可试 5e-4
POSE_WEIGHT=${8:-"0.1"}                # 可改：absolute pose loss 权重
REL_POSE_WEIGHT=${9:-"0.05"}           # 可改：local relative pose loss 权重
REL_POSE_START=${10:-500}              # 可改：relative pose loss 启用步数；长 views 可试 1000/2000
POSE_QUAT_CONVENTION=${11:-"xyzw"}     # 可改：xyzw 官方实现；wxyz 复现旧 loss
POSE_ANCHOR_NORM=${12:-1}              # 可改：1 开启 Stage2 pose anchor-scale norm；0 关闭
LOG_EVERY=100                          # 可改：日志间隔，不影响训练结果
SAVE_EVERY=1000                        # 可改：checkpoint 间隔；越小占磁盘越多
WARMUP_ITERS=1000                      # 可改：views curriculum warmup；长训练建议约 total iters 的 5%
NUM_FRAME_FOR_SCALE=8                  # 可改：anchor/scale frames；通常保持 8

MODE_SUFFIX="full"
HEAD_ONLY_ARGS=()
if [[ "${HEAD_ONLY:-0}" == "1" ]]; then
  MODE_SUFFIX="head_only"
  HEAD_ONLY_ARGS=(--head_only)
fi
POSE_NORM_SUFFIX="poseNorm${POSE_ANCHOR_NORM}"
POSE_NORM_ARGS=(--use_pose_anchor_scale_norm)
if [[ "${POSE_ANCHOR_NORM}" == "0" ]]; then
  POSE_NORM_ARGS=(--no_pose_anchor_scale_norm)
elif [[ "${POSE_ANCHOR_NORM}" != "1" ]]; then
  echo "ERROR: POSE_ANCHOR_NORM must be 1 or 0, got ${POSE_ANCHOR_NORM}" >&2
  exit 1
fi

OUTPUT_DIR="try_train/checkpoints/lzw_map_stage2_${MODE_SUFFIX}_${VIEWS_START}-${VIEWS_END}_k${K_MIN}-${K_MAX}_${POSE_QUAT_CONVENTION}_${POSE_NORM_SUFFIX}_${TOTAL_ITERS}iters"  # 可改：输出目录命名
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "ERROR: STAGE1_CKPT=${STAGE1_CKPT} does not exist" >&2
  exit 1
fi

RESUME_CKPT="${RESUME:-}"
RESUME_ARGS=()
if [[ "${RESUME_CKPT}" == "auto" ]]; then
  RESUME_CKPT=$(find "${OUTPUT_DIR}" -maxdepth 1 -name 'checkpoint_iter_*.pt' -printf '%f\n' 2>/dev/null \
    | sort -V \
    | tail -1)
  if [[ -z "${RESUME_CKPT}" ]]; then
    echo "ERROR: RESUME=auto but no checkpoint_iter_*.pt found in ${OUTPUT_DIR}" >&2
    exit 1
  fi
  RESUME_CKPT="${OUTPUT_DIR}/${RESUME_CKPT}"
fi

if [[ -n "${RESUME_CKPT}" ]]; then
  if [[ ! -f "${RESUME_CKPT}" ]]; then
    echo "ERROR: RESUME=${RESUME_CKPT} does not exist" >&2
    exit 1
  fi
  RESUME_ARGS=(--resume "${RESUME_CKPT}")
fi

echo "========================================================================"
echo "LZW-Map Stage2 Training (GCA streaming)"
echo "========================================================================"
echo "  Dataset:          ${DATA_ROOT}"
echo "  DINOv2 repo:      ${DINOV2_REPO}"
echo "  Stage1 ckpt:      ${STAGE1_CKPT}"
echo "  Backbone:         dinov2_vitb14_reg, pretrained, frozen patch_embed"
if [[ "${HEAD_ONLY:-0}" == "1" ]]; then
  echo "  Trainable:        camera head + depth head (aggregator frozen)"
else
  echo "  Trainable:        LZW aggregator blocks + camera head + depth head"
fi
echo "  Views curriculum: ${VIEWS_START} -> ${VIEWS_END}"
echo "  Local window k:   [${K_MIN}, ${K_MAX}]"
echo "  Anchor frames:    ${NUM_FRAME_FOR_SCALE}"
echo "  Learning rate:    ${LR}"
echo "  Fused loss:       depth + ${POSE_WEIGHT}*abs_pose + ${REL_POSE_WEIGHT}*local_rel_pose"
echo "  Rel pose starts:  iteration ${REL_POSE_START}"
echo "  Pose quat loss:   ${POSE_QUAT_CONVENTION}"
echo "  Pose anchor norm: ${POSE_ANCHOR_NORM}"
echo "  Iterations:       ${TOTAL_ITERS}"
echo "  Output:           ${OUTPUT_DIR}"
echo "  Log file:         ${LOG_FILE}"
if [[ -n "${RESUME_CKPT}" ]]; then
  echo "  Resume from:      ${RESUME_CKPT}"
fi
echo "========================================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONUNBUFFERED=1 \
conda run -n lingbot-map --no-capture-output python -u try_train/train_lzw_map_stage2.py \
  --data_root "${DATA_ROOT}" \
  --dinov2_repo "${DINOV2_REPO}" \
  --stage1_checkpoint "${STAGE1_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --total_iterations "${TOTAL_ITERS}" \
  --views_start "${VIEWS_START}" \
  --views_end "${VIEWS_END}" \
  --warmup_iterations "${WARMUP_ITERS}" \
  --k_min "${K_MIN}" \
  --k_max "${K_MAX}" \
  --num_frame_for_scale "${NUM_FRAME_FOR_SCALE}" \
  --lr "${LR}" \
  --pose_weight "${POSE_WEIGHT}" \
  --rel_pose_weight "${REL_POSE_WEIGHT}" \
  --rel_pose_start_iter "${REL_POSE_START}" \
  --pose_quat_convention "${POSE_QUAT_CONVENTION}" \
  --log_every "${LOG_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --use_sdpa \
  "${POSE_NORM_ARGS[@]}" \
  "${HEAD_ONLY_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
