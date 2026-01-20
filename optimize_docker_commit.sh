#!/bin/bash
# 优化 docker commit 过程的脚本
# 用于在 commit 前清理不必要的文件，减少镜像大小

set -e

CONTAINER_NAME="flashvsr_ultra_fast"
NEW_TAG="flashvsr_ultra_fast:v8"

echo "=========================================="
echo "优化 Docker Commit 过程"
echo "=========================================="

echo ""
echo "1. 检查容器状态..."
if ! docker ps -a | grep -q "$CONTAINER_NAME"; then
    echo "错误: 容器 $CONTAINER_NAME 不存在"
    exit 1
fi

CONTAINER_STATE=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME")
echo "   容器状态: $CONTAINER_STATE"

if [ "$CONTAINER_STATE" = "paused" ]; then
    echo "   警告: 容器处于暂停状态，正在恢复..."
    docker unpause "$CONTAINER_NAME"
    sleep 2
fi

echo ""
echo "2. 检查容器内临时文件和缓存..."
# 在容器内清理常见的临时文件和缓存
docker exec "$CONTAINER_NAME" bash -c "
    echo '  清理 pip 缓存...'
    pip cache purge 2>/dev/null || true
    
    echo '  清理 apt 缓存...'
    apt-get clean 2>/dev/null || true
    rm -rf /var/lib/apt/lists/* 2>/dev/null || true
    
    echo '  清理临时编译文件...'
    find /tmp -type f -name '*.o' -o -name '*.so' -o -name '*.a' 2>/dev/null | head -20
    find /tmp -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    find /tmp -type f -name '*.pyc' -delete 2>/dev/null || true
    
    echo '  清理 Python 缓存...'
    find /usr/local/lib/python3.*/dist-packages -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    find /usr/local/lib/python3.*/dist-packages -type f -name '*.pyc' -delete 2>/dev/null || true
    
    echo '  检查 block-sparse-attention 构建目录大小...'
    if [ -d '/tmp/block_sparse_attention_build' ]; then
        du -sh /tmp/block_sparse_attention_build 2>/dev/null || true
        echo '  提示: 如果编译成功，可以删除构建目录以减小镜像大小'
    fi
    
    echo '  清理完成！'
"

echo ""
echo "3. 检查当前镜像大小..."
docker images "$CONTAINER_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" | head -5

echo ""
echo "4. 估算容器更改大小..."
# 使用 docker diff 查看更改（可能很慢，所以只显示前20个）
echo "   检查文件更改（前20个）..."
docker diff "$CONTAINER_NAME" | head -20

TOTAL_CHANGES=$(docker diff "$CONTAINER_NAME" | wc -l)
echo "   总更改项数: $TOTAL_CHANGES"

echo ""
echo "5. 执行优化的 commit..."
echo "   开始 commit（这可能需要几分钟，请耐心等待）..."
echo "   提示: 如果很慢，可以在另一个终端运行 'docker stats $CONTAINER_NAME' 查看资源使用"

# 使用 --pause=false 可以加速（如果容器已暂停）
START_TIME=$(date +%s)
docker commit \
    --pause=false \
    -a "Chenlu" \
    -m "compile block-sparse-attention" \
    "$CONTAINER_NAME" \
    "$NEW_TAG"
END_TIME=$(date +%s)

ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "Commit 完成！"
echo "=========================================="
echo "   耗时: ${ELAPSED} 秒"
echo "   新镜像: $NEW_TAG"

echo ""
echo "6. 验证新镜像..."
docker images "$NEW_TAG" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

echo ""
echo "7. 镜像大小对比..."
echo "   旧镜像 (v7):"
docker images "flashvsr_ultra_fast:v7" --format "   {{.Size}}" 2>/dev/null || echo "   未找到"
echo "   新镜像 (v8):"
docker images "$NEW_TAG" --format "   {{.Size}}" 2>/dev/null || echo "   未找到"

echo ""
echo "完成！"
