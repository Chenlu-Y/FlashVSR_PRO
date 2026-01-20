#!/bin/bash
# 从临时文件恢复视频的脚本

# 使用方法：
# ./recover_video.sh <输入视频路径> <输出视频路径> [FPS]

INPUT_VIDEO="${1:-/app/input/core_1080.mp4}"
OUTPUT_VIDEO="${2:-/app/output/core_8K_recovered.mp4}"
FPS="${3:-24}"

# 获取视频目录名
VIDEO_BASENAME=$(basename "$INPUT_VIDEO")
VIDEO_NAME="${VIDEO_BASENAME%.*}"
VIDEO_NAME=$(echo "$VIDEO_NAME" | sed 's/[^a-zA-Z0-9_-]/_/g')
SCALE=4
DIR_NAME="${VIDEO_NAME}_${SCALE}x"

echo "=========================================="
echo "查找临时文件并恢复视频"
echo "=========================================="
echo "输入视频: $INPUT_VIDEO"
echo "输出视频: $OUTPUT_VIDEO"
echo "FPS: $FPS"
echo "目录名: $DIR_NAME"
echo ""

# 查找临时文件目录
WORKER_DIR="/tmp/flashvsr_multigpu/${DIR_NAME}"
CHECKPOINT_DIR="/tmp/flashvsr_checkpoints/${DIR_NAME}"

echo "检查临时文件位置..."
echo "  Worker目录: $WORKER_DIR"
echo "  Checkpoint目录: $CHECKPOINT_DIR"
echo ""

# 检查worker目录
if [ -d "$WORKER_DIR" ]; then
    WORKER_FILES=$(find "$WORKER_DIR" -name "worker_*.pt" -type f 2>/dev/null)
    WORKER_COUNT=$(echo "$WORKER_FILES" | wc -l)
    if [ "$WORKER_COUNT" -gt 0 ]; then
        echo "✓ 找到 $WORKER_COUNT 个worker文件:"
        for f in $WORKER_FILES; do
            SIZE=$(du -h "$f" | cut -f1)
            echo "    - $(basename $f) ($SIZE)"
        done
        echo ""
        echo "开始恢复视频..."
        docker exec -w /app/FlashVSR_Ultra_Fast flashvsr_ultra_fast python /app/FlashVSR_Ultra_Fast/recover_from_workers.py \
            "$WORKER_DIR" \
            "$OUTPUT_VIDEO" \
            --fps "$FPS"
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ 视频恢复成功: $OUTPUT_VIDEO"
        else
            echo ""
            echo "✗ 视频恢复失败"
            exit 1
        fi
    else
        echo "✗ Worker目录存在但没有找到worker文件"
    fi
else
    echo "✗ Worker目录不存在: $WORKER_DIR"
    echo ""
    echo "可能的原因："
    echo "  1. Worker进程在保存文件之前就OOM了"
    echo "  2. 临时文件已经被清理"
    echo "  3. 文件保存在其他位置"
    echo ""
    echo "建议："
    echo "  1. 检查是否有其他临时目录:"
    echo "     docker exec flashvsr_ultra_fast find /tmp -name '*core*' -type d"
    echo "  2. 重新运行推理（已修复OOM问题）"
    exit 1
fi
