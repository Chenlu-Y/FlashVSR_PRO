#!/bin/bash
# 在 flashvsr_ultra_fast 容器内重新编译 block-sparse-attention
# 使用方法：
#   1. 进入容器：docker exec -it flashvsr_ultra_fast bash
#   2. 运行脚本：bash /app/FlashVSR_Ultra_Fast/compile_block_sparse_attention.sh

set -e  # 遇到错误立即退出

echo "=========================================="
echo "开始编译 block-sparse-attention"
echo "=========================================="

# 检查是否在容器内
if [ ! -d "/app" ]; then
    echo "警告: 未检测到 /app 目录，可能不在容器内"
    echo "请确保在 flashvsr_ultra_fast 容器内运行此脚本"
fi

# 设置工作目录
# 优先使用宿主机已clone的仓库（如果存在）
# 注意：Docker容器中，宿主机路径需要映射，这里使用容器内可见的路径
HOST_REPO_PATH="/workspace/ycl/flashvsr/Block-Sparse-Attention"
WORK_DIR="/tmp/block_sparse_attention_build"
REPO_URL="https://github.com/mit-han-lab/Block-Sparse-Attention.git"

echo "1. 检查 Python 环境..."
python --version
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "2. 检查是否使用宿主机已clone的仓库..."
# 检查多个可能的路径（容器内映射路径）
POSSIBLE_PATHS=(
    "/app/Block-Sparse-Attention"
    "/workspace/ycl/flashvsr/Block-Sparse-Attention"
    "/home/dellhpc/workspace/ycl/flashvsr/Block-Sparse-Attention"
)

FOUND_REPO=""
for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -d "$path" ] && [ -f "$path/setup.py" ]; then
        FOUND_REPO="$path"
        echo "  ✓ 找到宿主机仓库: $path"
        break
    fi
done

if [ -n "$FOUND_REPO" ]; then
    echo "  使用宿主机仓库进行编译..."
    WORK_DIR="$FOUND_REPO"
    cd "$WORK_DIR"
    # 确保是最新版本
    git pull origin main 2>/dev/null || git pull origin master 2>/dev/null || echo "  跳过git pull（可能不在git仓库中）"
else
    echo "  ✗ 未找到宿主机仓库，将从GitHub克隆..."
    echo ""
    echo "3. 清理旧的构建目录..."
    rm -rf "$WORK_DIR"
    
    echo ""
    echo "4. 克隆 block-sparse-attention 仓库..."
    git clone "$REPO_URL" "$WORK_DIR"
    cd "$WORK_DIR"
fi

echo ""
echo "5. 检查仓库状态..."
git log --oneline -5 2>/dev/null || echo "  无法获取git日志（可能不在git仓库中）"

echo ""
echo "6. 安装编译依赖..."
# 确保有必要的编译工具
pip install -q ninja packaging wheel

echo ""
echo "7. 检查CUDA架构支持..."
# 检查当前GPU架构
python -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'  检测到GPU: {props.name}')
    print(f'  Compute Capability: {props.major}.{props.minor} (sm_{props.major}{props.minor})')
    if props.major == 12:
        print('  → 需要编译支持sm_120 (Blackwell)')
    elif props.major >= 8:
        print(f'  → 需要编译支持sm_{props.major}{props.minor}')
"

echo ""
echo "8. 检查CUDA工具包版本以确定支持的架构..."
# 确保使用最新的CUDA路径
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
source /etc/profile.d/cuda.sh 2>/dev/null || true

CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/' || echo "")
echo "  CUDA工具包版本: ${CUDA_VERSION:-未知}"

# 根据CUDA版本决定是否包含sm_120
# sm_120需要CUDA 12.8/12.9+
ARCH_LIST="8.0;8.6;8.9;9.0;10.0"
if [ -n "$CUDA_VERSION" ]; then
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
    if [ "$CUDA_MAJOR" -gt 12 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 8 ]); then
        ARCH_LIST="$ARCH_LIST;12.0"
        echo "  ✓ CUDA版本支持sm_120，将包含在编译中"
    else
        echo "  ⚠ CUDA版本($CUDA_VERSION)不支持sm_120，将跳过（运行时可能使用sm_90代码）"
    fi
else
    echo "  ⚠ 无法确定CUDA版本，将跳过sm_120"
fi

echo ""
echo "9. 开始编译安装（支持的架构: $ARCH_LIST）..."
# 设置环境变量以确保编译时包含所有支持的架构
export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"
# 使用 pip 安装（会自动编译，强制本地编译避免网络问题）
pip install --no-binary :all: -v --no-build-isolation .

# 或者使用 setup.py（如果 pip 安装失败）
# python setup.py install

echo ""
echo "10. 验证安装..."
# 确保库路径正确
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH
python -c "
try:
    from block_sparse_attn import block_sparse_attn_func
    print('✓ block-sparse-attention 安装成功！')
    print('✓ block_sparse_attn_func 可以正常导入')
    print('✓ 支持 sm_120 (Blackwell) 架构')
except ImportError as e:
    print('✗ 导入失败:', e)
    print('  提示: 如果遇到 libc10.so 错误，请确保 LD_LIBRARY_PATH 包含 PyTorch 库路径')
    exit(1)
"

echo ""
echo "=========================================="
echo "编译完成！"
echo "=========================================="
echo ""
echo "现在可以在代码中使用 block_sparse_attention 了："
echo "  from block_sparse_attn import block_sparse_attn_func"
echo ""
echo "清理临时文件..."
# 只有在使用临时目录时才清理
if [ "$WORK_DIR" = "/tmp/block_sparse_attention_build" ]; then
    rm -rf "$WORK_DIR"
fi

echo "完成！"

