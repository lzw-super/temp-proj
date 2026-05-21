#!/bin/bash

# ==============================================================================
# 视频离线测试脚本
# 功能: 对生成的视频数据进行离线测试/处理
# 作者: AI Assistant
# 日期: 2026-04-29
# ==============================================================================

# --- 配置参数 ---
PYTHON="/home/lizhengwu/miniconda3/envs/lingbot-map/bin/python"
# 输入数据目录 (对应上一个脚本的输出目录)
DATA_DIR="./video_data"
# 输出视频路径
OUTPUT_PATH="test_result.mp4"
# 帧率
FPS=10
# 分辨率
RESOLUTION="1920x1080"
# 置信度阈值
CONF_THRESHOLD=1.5
# 下采样因子
DOWNSAMPLE_FACTOR=10

# --- 脚本逻辑 ---

# 1. 检查输入数据目录是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ 错误: 找不到输入数据目录 '$DATA_DIR'"
    echo "   请确保先运行了生成数据的脚本 (run_lingbot_map.sh)。"
    exit 1
fi

# 2. 打印运行信息
echo "🎬 开始运行视频离线测试..."
echo "   输入目录: $DATA_DIR"
echo "   输出文件: $OUTPUT_PATH"
echo "   参数设置: FPS=$FPS, RES=$RESOLUTION, CONF=$CONF_THRESHOLD"
echo "--------------------------------------------------"

# 3. 执行核心命令
# 同样使用相对路径查找脚本，确保灵活性
$PYTHON /home/lizhengwu/desktop/temp_proj/lingbot-map/test_video_offline.py \
    --data_dir "$DATA_DIR" \
    --output_path "$OUTPUT_PATH" \
    --fps "$FPS" \
    --resolution "$RESOLUTION" \
    --conf_threshold "$CONF_THRESHOLD" \
    --downsample_factor "$DOWNSAMPLE_FACTOR"

# 4. 检查执行结果
if [ $? -eq 0 ]; then
    echo "--------------------------------------------------"
    echo "✅ 视频测试处理完成！"
    echo "   结果保存在: $OUTPUT_PATH"
else
    echo "--------------------------------------------------"
    echo "❌ 处理失败，请检查上方的错误日志。"
    exit 1
fi