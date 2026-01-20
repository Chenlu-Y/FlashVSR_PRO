#!/bin/bash
# 安全清理 /tmp 目录的脚本
# 用于在 docker commit 前清理临时文件，减小镜像大小

CONTAINER_NAME="flashvsr_ultra_fast"

echo "=========================================="
echo "安全清理容器 /tmp 目录"
echo "=========================================="

# 检查容器状态
CONTAINER_STATE=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "unknown")
if [ "$CONTAINER_STATE" = "paused" ]; then
    echo "警告: 容器处于暂停状态，正在恢复..."
    docker unpause "$CONTAINER_NAME"
    sleep 2
fi

echo ""
echo "1. 检查当前 /tmp 目录大小..."
docker exec "$CONTAINER_NAME" bash -c "du -sh /tmp 2>/dev/null || echo '无法计算'"

echo ""
echo "2. 分析可安全删除的文件..."
echo "----------------------------------------"

# 分类统计
echo "   a) block-sparse-attention 编译目录："
docker exec "$CONTAINER_NAME" bash -c "
    if [ -d '/tmp/block_sparse_attention_build' ]; then
        echo '     - /tmp/block_sparse_attention_build'
        du -sh /tmp/block_sparse_attention_build 2>/dev/null || echo '       无法计算大小'
    fi
    if [ -d '/tmp/Block-Sparse-Attention' ]; then
        echo '     - /tmp/Block-Sparse-Attention'
        du -sh /tmp/Block-Sparse-Attention 2>/dev/null || echo '       无法计算大小'
    fi
"

echo ""
echo "   b) FlashVSR 临时文件（如果处理已完成，可以删除）："
docker exec "$CONTAINER_NAME" bash -c "
    for dir in flashvsr_checkpoints flashvsr_multigpu flashvsr_segments; do
        if [ -d \"/tmp/\$dir\" ]; then
            echo \"     - /tmp/\$dir\"
            du -sh \"/tmp/\$dir\" 2>/dev/null || echo '       无法计算大小'
        fi
    done
"

echo ""
echo "   c) 大型 .npy 临时文件（NumPy 数组，如果处理已完成可删除）："
docker exec "$CONTAINER_NAME" bash -c "
    find /tmp -name '*.npy' -type f -size +100M 2>/dev/null | while read file; do
        size=\$(du -h \"\$file\" 2>/dev/null | cut -f1)
        echo \"     - \$size - \$file\"
    done
"

echo ""
echo "   d) Python 临时目录（通常可安全删除）："
docker exec "$CONTAINER_NAME" bash -c "
    find /tmp -maxdepth 1 -type d -name 'tmp*' -o -name 'hsperfdata_*' 2>/dev/null | while read dir; do
        if [ -d \"\$dir\" ]; then
            size=\$(du -sh \"\$dir\" 2>/dev/null | cut -f1)
            echo \"     - \$size - \$dir\"
        fi
    done | head -10
"

echo ""
echo "3. 开始清理..."
echo "----------------------------------------"

# 询问用户确认（可以通过参数跳过）
SKIP_CONFIRM=${1:-""}
if [ -z "$SKIP_CONFIRM" ]; then
    echo ""
    read -p "是否继续清理？这将删除上述临时文件 (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消清理"
        exit 0
    fi
fi

echo ""
echo "正在清理..."

# 1. 清理 block-sparse-attention 编译目录（如果编译已成功）
echo "   [1/5] 清理 block-sparse-attention 编译目录..."
docker exec "$CONTAINER_NAME" bash -c "
    if [ -d '/tmp/block_sparse_attention_build' ]; then
        rm -rf /tmp/block_sparse_attention_build && echo '     ✓ 已删除 /tmp/block_sparse_attention_build' || echo '     ✗ 删除失败'
    else
        echo '     - 目录不存在，跳过'
    fi
    if [ -d '/tmp/Block-Sparse-Attention' ]; then
        rm -rf /tmp/Block-Sparse-Attention && echo '     ✓ 已删除 /tmp/Block-Sparse-Attention' || echo '     ✗ 删除失败'
    else
        echo '     - 目录不存在，跳过'
    fi
"

# 2. 清理大型 .npy 文件（NumPy 临时数组）
echo ""
echo "   [2/5] 清理大型 .npy 临时文件..."
docker exec "$CONTAINER_NAME" bash -c "
    find /tmp -name '*.npy' -type f -size +100M 2>/dev/null | while read file; do
        size=\$(du -h \"\$file\" 2>/dev/null | cut -f1)
        rm -f \"\$file\" && echo \"     ✓ 已删除 \$size - \$file\" || echo \"     ✗ 删除失败: \$file\"
    done
    echo '     - 清理完成'
"

# 3. 清理 FlashVSR 临时文件（如果处理已完成）
echo ""
echo "   [3/5] 清理 FlashVSR 临时文件..."
echo "     提示: 这些文件用于断点续传，如果不需要恢复可以删除"
docker exec "$CONTAINER_NAME" bash -c "
    for dir in flashvsr_checkpoints flashvsr_multigpu flashvsr_segments; do
        if [ -d \"/tmp/\$dir\" ]; then
            rm -rf \"/tmp/\$dir\" && echo \"     ✓ 已删除 /tmp/\$dir\" || echo \"     ✗ 删除失败: /tmp/\$dir\"
        fi
    done
"

# 4. 清理 Python 临时目录
echo ""
echo "   [4/5] 清理 Python 临时目录..."
docker exec "$CONTAINER_NAME" bash -c "
    find /tmp -maxdepth 1 -type d \\( -name 'tmp*' -o -name 'hsperfdata_*' \\) 2>/dev/null | while read dir; do
        if [ -d \"\$dir\" ] && [ \"\$dir\" != \"/tmp\" ]; then
            rm -rf \"\$dir\" && echo \"     ✓ 已删除 \$dir\" || echo \"     ✗ 删除失败: \$dir\"
        fi
    done
"

# 5. 清理其他临时文件
echo ""
echo "   [5/5] 清理其他临时文件..."
docker exec "$CONTAINER_NAME" bash -c "
    # 清理 .pyc 文件
    find /tmp -name '*.pyc' -delete 2>/dev/null && echo '     ✓ 已清理 .pyc 文件' || true
    # 清理 .pyo 文件
    find /tmp -name '*.pyo' -delete 2>/dev/null && echo '     ✓ 已清理 .pyo 文件' || true
    # 清理空目录
    find /tmp -type d -empty -delete 2>/dev/null || true
    echo '     - 清理完成'
"

echo ""
echo "4. 清理后 /tmp 目录大小..."
docker exec "$CONTAINER_NAME" bash -c "du -sh /tmp 2>/dev/null || echo '无法计算'"

echo ""
echo "=========================================="
echo "清理完成！"
echo "=========================================="
echo ""
echo "提示:"
echo "  - 如果还需要恢复处理，请保留 flashvsr_* 目录"
echo "  - 清理后可以执行 docker commit 来减小镜像大小"
echo ""
