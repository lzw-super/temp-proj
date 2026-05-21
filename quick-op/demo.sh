#!/bin/bash

# ==============================================================================
# LingBot-Map 3D重建运行脚本
# 功能: 运行LingBot-Map模型进行3D场景重建
# 作者: AI Assistant
# 日期: 2026-04-29
# ==============================================================================

# --- 配置参数 (可根据需要修改) --- 
PYTHON="/home/lizhengwu/miniconda3/envs/lingbot-map/bin/python"
NAME="office2"
# 模型权重文件路径
MODEL_PATH="/home/shared_files/model_weights/linbo_map/lingbot-map.pt"
# 输入图像文件夹路径
IMAGE_FOLDER="example/${NAME}/rgb"
# 输出目录
OUTPUT_DIR="${NAME}_output"
# 深度和位姿导出路径
EXPORT_DEPTH_POSE_PATH="${OUTPUT_DIR}/all"
# 视频数据导出路径
EXPORT_VIDEO_PATH="./video_data_${NAME}"

# --- 脚本逻辑 ---

# 1. 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 错误: 找不到模型文件 '$MODEL_PATH'"
    echo "   请检查路径是否正确。"
    exit 1
fi

# 2. 创建必要的输出目录 
mkdir -p "$OUTPUT_DIR"
mkdir -p "$EXPORT_DEPTH_POSE_PATH"
mkdir -p "$(dirname "$EXPORT_VIDEO_PATH")"

# 3. 打印运行信息
echo "🚀 开始运行 LingBot-Map 3D重建..."
echo "   模型路径: $MODEL_PATH"
echo "   输入图像: $IMAGE_FOLDER"
echo "   输出目录: $OUTPUT_DIR"
echo "--------------------------------------------------"

# 4. 执行核心命令
# 使用 "${BASH_SOURCE[0]%/*}" 确保脚本在任何目录下都能正确找到 demo.py
$PYTHON /home/lizhengwu/desktop/temp_proj/lingbot-map/demo.py \
    --model_path "$MODEL_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --mask_sky \
    --use_sdpa \
    --no_viewer \
    --export_depth_pose "$EXPORT_DEPTH_POSE_PATH" \
    --image_size 518 \
    --export_video_data "$EXPORT_VIDEO_PATH" \
    --save_video \
    --export_ply 





# 5. 检查命令执行结果
if [ $? -eq 0 ]; then
    echo "--------------------------------------------------"
    echo "✅ 任务执行成功！"
    echo "   结果已保存在: $OUTPUT_DIR"
else
    echo "--------------------------------------------------"
    echo "❌ 任务执行失败，请检查上方的错误日志。"
    exit 1
fi