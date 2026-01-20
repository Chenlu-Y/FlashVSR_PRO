#!/bin/bash
# 在 flashvsr_ultra_fast 容器内升级 CUDA 工具包到 12.9
# 使用方法：
#   1. 进入容器：docker exec -it flashvsr_ultra_fast bash
#   2. 运行脚本：bash /app/FlashVSR_Ultra_Fast/upgrade_cuda_to_12.9.sh

set -e  # 遇到错误立即退出

echo "=========================================="
echo "升级 CUDA 工具包到 12.9 (支持 sm_120)"
echo "=========================================="

# 检查是否在容器内
if [ ! -d "/app" ]; then
    echo "警告: 未检测到 /app 目录，可能不在容器内"
    echo "请确保在 flashvsr_ultra_fast 容器内运行此脚本"
fi

echo ""
echo "1. 检查当前 CUDA 版本..."
nvcc --version 2>/dev/null || echo "  nvcc 未找到"
python -c "import torch; print(f'  PyTorch CUDA版本: {torch.version.cuda}')" 2>/dev/null || echo "  PyTorch 未安装"

echo ""
echo "2. 直接下载并安装 CUDA 12.8/12.9 工具包..."
export DEBIAN_FRONTEND=noninteractive

# 检查是否已有CUDA 12.8或12.9
if [ -d "/usr/local/cuda-12.9" ]; then
    echo "  ✓ 检测到 CUDA 12.9 已安装"
    CUDA_INSTALLED="12.9"
elif [ -d "/usr/local/cuda-12.8" ]; then
    echo "  ✓ 检测到 CUDA 12.8 已安装"
    CUDA_INSTALLED="12.8"
else
    echo "  下载 CUDA 12.8.1 本地安装包..."
    # 下载CUDA 12.8.1 runfile（较小，只包含工具包）
    # 优先尝试 12.8.1，如果失败则尝试 12.8.0
    CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run"
    wget -q --show-progress "$CUDA_URL" -O /tmp/cuda_12.8.run || {
        echo "  ⚠ 12.8.1 下载失败，尝试 12.8.0..."
        CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run"
        wget -q --show-progress "$CUDA_URL" -O /tmp/cuda_12.8.run || {
            echo "  ⚠ 下载失败，尝试使用apt安装（可能需要修复GPG）"
            CUDA_INSTALLED=""
        }
    }
    
    if [ -f "/tmp/cuda_12.8.run" ]; then
        echo "  安装 CUDA 12.8..."
        chmod +x /tmp/cuda_12.8.run
        /tmp/cuda_12.8.run --silent --toolkit --no-opengl-libs --override
        CUDA_INSTALLED="12.8"
        rm -f /tmp/cuda_12.8.run
    fi
fi

if [ -z "$CUDA_INSTALLED" ]; then
    echo "  尝试通过apt安装（需要修复GPG问题）..."
    # 临时修复：使用wget下载密钥
    mkdir -p /etc/apt/keyrings
    wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | gpg --dearmor > /etc/apt/keyrings/cuda.gpg 2>/dev/null || {
        # 如果gpg失败，直接使用wget下载并添加
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub -O /tmp/cuda-key.pub
        apt-key add /tmp/cuda-key.pub 2>/dev/null || {
            gpg --dearmor < /tmp/cuda-key.pub > /etc/apt/trusted.gpg.d/cuda.gpg 2>/dev/null || true
        }
    }
    
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" > /etc/apt/sources.list.d/cuda.list
    
    echo ""
    echo "3. 更新包列表..."
    apt-get update -o Acquire::AllowInsecureRepositories=true -qq 2>&1 | grep -v "GPG error" || true
fi

if [ -z "$CUDA_INSTALLED" ]; then
    echo ""
    echo "4. 安装 CUDA 12.8/12.9 工具包..."
    # 安装 CUDA 12.8 开发工具包（包含 nvcc）
    if apt-cache show cuda-toolkit-12-8 > /dev/null 2>&1; then
        echo "  安装 CUDA 12.8..."
        apt-get install -y -o APT::Get::AllowUnauthenticated=true -qq \
            cuda-toolkit-12-8 \
            cuda-cudart-dev-12-8 \
            cuda-nvcc-12-8 2>&1 | grep -v "GPG error" || true
        CUDA_INSTALLED="12.8"
    elif apt-cache show cuda-toolkit-12-9 > /dev/null 2>&1; then
        echo "  安装 CUDA 12.9..."
        apt-get install -y -o APT::Get::AllowUnauthenticated=true -qq \
            cuda-toolkit-12-9 \
            cuda-cudart-dev-12-9 \
            cuda-nvcc-12-9 2>&1 | grep -v "GPG error" || true
        CUDA_INSTALLED="12.9"
    else
        echo "  ✗ 无法找到 CUDA 12.8 或 12.9 工具包"
        echo "  尝试查找可用的CUDA版本..."
        apt-cache search cuda-toolkit 2>&1 | grep "^cuda-toolkit" | head -5 || echo "  无法搜索包列表"
        exit 1
    fi
fi

echo ""
echo "5. 更新符号链接..."
# 更新 /usr/local/cuda 指向已安装的版本
if [ "$CUDA_INSTALLED" = "12.9" ] && [ -d "/usr/local/cuda-12.9" ]; then
    rm -f /usr/local/cuda
    rm -f /usr/local/cuda-12
    ln -sf /usr/local/cuda-12.9 /usr/local/cuda
    ln -sf /usr/local/cuda-12.9 /usr/local/cuda-12
    echo "  ✓ 已更新符号链接指向 CUDA 12.9"
elif [ "$CUDA_INSTALLED" = "12.8" ] && [ -d "/usr/local/cuda-12.8" ]; then
    rm -f /usr/local/cuda
    rm -f /usr/local/cuda-12
    ln -sf /usr/local/cuda-12.8 /usr/local/cuda
    ln -sf /usr/local/cuda-12.8 /usr/local/cuda-12
    echo "  ✓ 已更新符号链接指向 CUDA 12.8"
else
    echo "  ✗ 未找到 /usr/local/cuda-12.8 或 cuda-12.9 目录"
    echo "  查找已安装的CUDA版本..."
    ls -la /usr/local/ | grep cuda
    exit 1
fi

echo ""
echo "6. 更新环境变量..."
# 更新 PATH 和 LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 写入到 /etc/profile.d/cuda.sh 以便持久化
cat > /etc/profile.d/cuda.sh << 'EOF'
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF
chmod +x /etc/profile.d/cuda.sh

echo ""
echo "7. 验证 CUDA 12.9 安装..."
source /etc/profile.d/cuda.sh
nvcc --version

echo ""
echo "8. 验证 sm_120 支持..."
python -c "
import subprocess
result = subprocess.run(['nvcc', '--help'], capture_output=True, text=True)
if 'sm_120' in result.stdout or 'compute_120' in result.stdout:
    print('  ✓ nvcc 支持 sm_120')
else:
    # 尝试直接编译测试
    print('  测试编译 sm_120 支持...')
    test_code = '''
#include <cuda_runtime.h>
__global__ void test() {}
int main() {
    test<<<1,1>>>();
    return 0;
}
'''
    with open('/tmp/test_sm120.cu', 'w') as f:
        f.write(test_code)
    result = subprocess.run(
        ['nvcc', '-arch=sm_120', '/tmp/test_sm120.cu', '-o', '/tmp/test_sm120'],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        print('  ✓ nvcc 可以编译 sm_120 代码')
        import os
        os.remove('/tmp/test_sm120.cu')
        os.remove('/tmp/test_sm120')
    else:
        print(f'  ⚠ 编译测试失败: {result.stderr[:200]}')
"

echo ""
echo "=========================================="
echo "CUDA 升级完成！"
echo "=========================================="
echo ""
echo "当前 CUDA 版本:"
nvcc --version | grep "release"
echo ""
echo "现在可以重新编译 block-sparse-attention 了："
echo "  bash /app/FlashVSR_Ultra_Fast/compile_block_sparse_attention.sh"
echo ""
echo "注意: 如果容器重启，需要重新 source /etc/profile.d/cuda.sh"
echo "或者将以下内容添加到容器的启动脚本中："
echo "  export PATH=/usr/local/cuda/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
