# ComfyUI-FlashVSR_Ultra_Fast
在低显存环境下运行 FlashVSR，同时保持无伪影高质量输出。  
**[[📃English](./README.md)]**

## 更新日志
#### 2025-10-31
- **新增:** 独立的 `infer_video.py` 脚本，无需 ComfyUI 即可处理视频
- **新增:** 多GPU并行处理 (`--multi_gpu`) - 自动将视频按帧分割到多个GPU
- **新增:** 自适应tile批处理 (`--adaptive_batch_size`) - 根据GPU显存动态调整tile并发数
- **新增:** GPU显存监控和优化，充分利用24-32GB显卡
- **新增:** 总耗时统计功能，便于性能监控
- **改进:** 视频读取增加OpenCV兜底，提升编解码器兼容性
- **修复:** 空帧处理和负维度错误

#### 2025-10-24
- 新增长视频管道, 可显著降低长视频放大的显存用量  

#### 2025-10-22
- 使用`Sparse_SageAttention`替换了`Block-Sparse-Attention`, 无需编译安装任何自定义内核, 开箱即用.  
- 支持在 RTX50 系列显卡上运行.

#### 2025-10-21
- 项目首次发布, 引入了`tile_dit`等功能, 大幅度降低显存需求  

## 预览
![](./img/preview.jpg)

## 使用说明

### 独立推理脚本（新增！）

使用 `infer_video.py` 可在不使用 ComfyUI 的情况下直接处理视频：

#### Docker容器内运行

```bash
# 方法一：进入容器后运行
docker exec -it flashvsr_ultra_fast bash
cd /app/FlashVSR_Ultra_Fast
python infer_video.py \
  --input /app/input/video.mp4 \
  --output /app/output/output.mp4 \
  --mode tiny \
  --scale 4 \
  --tiled_dit True \
  --tile_size 256 \
  --tile_overlap 24 \
  --model_dir /app/models

# 方法二：从宿主机直接运行
docker exec flashvsr_ultra_fast python /app/FlashVSR_Ultra_Fast/infer_video.py \
  --input /app/input/video.mp4 \
  --output /app/output/output.mp4 \
  --mode tiny \
  --scale 4 \
  --tiled_dit True \
  --model_dir /app/models
```

#### 完整参数说明

**必需参数：**
- `--input`: 输入视频路径
- `--output`: 输出视频路径

**超分相关参数：**
- `--scale`: 超分倍数，可选值：`2`, `3`, `4`（推荐使用 `4`）
  - `2`: 2倍放大，显存占用最低
  - `3`: 3倍放大，平衡质量和显存占用（**完全支持，不限于2的倍数**）
  - `4`: 4倍放大，效果最好（推荐）
- `--mode`: 运行模式
  - `tiny`: 快速模式（默认，显存占用较低，推荐）
  - `tiny-long`: 长视频模式（显存占用最低，适合超长视频）
  - `full`: 高质量模式（显存占用较高，质量最好）

**显存优化参数：**
- `--tiled_dit`: 启用DiT分块计算（`True`/`False`，默认`False`），显存不足时强烈推荐启用
- `--tile_size`: 分块大小（默认`256`），显存不足时可减小到`128`
- `--tile_overlap`: 分块重叠大小（默认`24`），建议为`tile_size`的10-15%
- `--tiled_vae`: 启用VAE分块解码（`True`/`False`，默认`True`）
- `--unload_dit`: 解码前卸载DiT模型（`True`/`False`，默认`False`），显存非常紧张时使用

**多GPU并行处理参数 ⚡：**
- `--multi_gpu`: 启用多GPU并行处理（无需参数值，直接添加即可）
  - 自动将视频按帧分割到多个GPU并行处理
  - 需要2个或以上GPU
  - 接近线性的加速比（2个GPU约2倍速度）
  - 适用于长视频（>500帧）
- `--adaptive_batch_size`: 启用自适应批处理大小（无需参数值，直接添加即可）
  - 根据GPU显存动态调整同时处理的tile数量
  - 需要启用 `--tiled_dit`
  - 大显存GPU（24GB+）效果最明显

**设备与精度参数：**
- `--device`: 指定使用的GPU设备（默认`cuda:0`），启用`--multi_gpu`时会被忽略
- `--precision`: 计算精度（默认`bf16`），可选`fp16`、`bf16`、`fp32`

**高级参数（质量调优）：**
- `--color_fix`: 颜色修正（`True`/`False`，默认`True`）
- `--sparse_ratio`: 稀疏比率（默认`2.0`，范围`1.5-2.0`），`2.0`更稳定
- `--kv_ratio`: KV缓存比率（默认`3.0`，范围`1.0-3.0`），`3.0`质量更高
- `--local_range`: 局部范围（默认`11`，可选`9`或`11`），`11`更稳定
- `--attention_mode`: 注意力模式（默认`sparse_sage_attention`）

**其他参数：**
- `--model_dir`: 模型目录路径（默认：`/app/models`）
- `--seed`: 随机种子（默认`0`）

#### 常用命令示例

**基础4倍超分（显存充足）：**
```bash
python infer_video.py \
  --input /app/input/video.mp4 \
  --output /app/output/output_4x.mp4 \
  --mode tiny \
  --scale 4 \
  --tiled_dit True
```

**3倍放大（平衡质量和显存）：**
```bash
python infer_video.py \
  --input /app/input/video.mp4 \
  --output /app/output/output_3x.mp4 \
  --mode tiny \
  --scale 3 \
  --tiled_dit True \
  --tile_size 256
```

**低显存模式：**
```bash
python infer_video.py \
  --input /app/input/video.mp4 \
  --output /app/output/output_4x.mp4 \
  --mode tiny-long \
  --scale 4 \
  --tiled_dit True \
  --tile_size 128 \
  --tile_overlap 16 \
  --unload_dit True
```

**多GPU加速（2个以上GPU）：**
```bash
python infer_video.py \
  --input /app/input/video.mp4 \
  --output /app/output/output_4x.mp4 \
  --mode tiny \
  --scale 4 \
  --tiled_dit True \
  --multi_gpu \
  --adaptive_batch_size
```

**根据显存大小选择配置：**
- **显存 < 12GB**: `--mode tiny-long --scale 2 --tiled_dit True --tile_size 128 --unload_dit True`
- **显存 12-16GB**: `--mode tiny --scale 4 --tiled_dit True --tile_size 256`
- **显存 16-24GB**: `--mode tiny --scale 4 --tiled_dit True --tile_size 256 --adaptive_batch_size`
- **显存 > 24GB**: `--mode full --scale 4 --tiled_dit True --tile_size 512 --adaptive_batch_size`

详细优化指南请参阅 [OPTIMIZATION_GUIDE.md](./OPTIMIZATION_GUIDE.md)。

### ComfyUI 节点

- **mode（模式）：**  
  `tiny` → 更快（默认）；`tiny-long` → 长视频低显存；`full` → 更高质量  
- **scale（放大倍数）：**  
  支持 `2`, `3`, `4` 倍放大（不限于2的倍数），通常使用 `4` 效果更好；如果显存不足，可使用 `2` 或 `3`  
- **color_fix（颜色修正）：**  
  使用小波变换方法修正输出视频的颜色偏差。  
- **tiled_vae（VAE分块解码）：**  
  启用后可显著降低显存占用，但会降低解码速度。  
- **tiled_dit（DiT分块计算）：**  
  大幅减少显存占用，但会降低推理速度。  
- **tile_size / tile_overlap（分块大小与重叠）：**  
  控制输入视频在推理时的分块方式。  
- **unload_dit（卸载DiT模型）：**  
  解码前卸载 DiT 模型以降低显存峰值，但会略微降低速度。  

## 安装步骤

#### 安装节点:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast.git
python -m pip install -r ComfyUI-FlashVSR_Ultra_Fast/requirements.txt
```
> 📢: 要在RTX20系或更早的GPU上运行, 请安装`triton<3.3.0`:  

> ```bash
# Windows
python -m pip install -U triton-windows<3.3.0
# Linux
python -m pip install -U triton<3.3.0
```

#### 模型下载:
- 从[这里](https://huggingface.co/JunhaoZhuang/FlashVSR)下载整个`FlashVSR`文件夹和它里面的所有文件, 并将其放到`ComfyUI/models`目录中。  

```
├── ComfyUI/models/FlashVSR
|     ├── LQ_proj_in.ckpt
|     ├── TCDecoder.ckpt
|     ├── diffusion_pytorch_model_streaming_dmd.safetensors
|     ├── Wan2.1_VAE.pth
```

## 性能优化

### 多GPU并行处理
对于拥有2+个GPU的系统，启用 `--multi_gpu` 可获得接近线性的加速：
- 自动将视频帧分割到可用GPU
- 每个GPU独立处理视频片段
- 结果无缝合并，处理重叠区域

### 自适应批处理大小
启用 `--adaptive_batch_size` 最大化GPU利用率：
- 根据可用显存动态调整tile批处理大小
- 对于32GB GPU，可同时处理6-16个tiles
- 如果显存变化，处理过程中自动重新平衡

**预期性能：**
- **双GPU + 自适应批处理**: 相比单GPU提升3-5倍
- **显存使用**: 32GB GPU峰值使用20-25GB（未优化时约13GB）

## 故障排除

### GPU访问问题

**问题：`RuntimeError: No CUDA GPUs are available`**

**解决方案：**
1. 检查宿主机GPU：`nvidia-smi`
2. 检查容器GPU访问：`docker exec flashvsr_ultra_fast nvidia-smi`
3. 如果容器内无法访问GPU，重启容器：
   ```bash
   docker-compose down
   docker-compose up -d
   docker exec flashvsr_ultra_fast nvidia-smi
   ```
4. 检查Docker GPU支持：`docker info | grep -i runtime`（应该看到`nvidia`）
5. 如果缺少nvidia runtime，安装nvidia-docker2或nvidia-container-toolkit

### 显存不足（OOM）

**解决方案：**
- 使用 `--mode tiny-long`
- 减小 `--tile_size`（如128或64）
- 启用 `--unload_dit True`
- 使用 `--scale 2` 或 `3` 而不是 `4`
- 减小 `--kv_ratio`（如1.0）

### 多GPU使用注意事项

- 需要至少2个GPU才能启用 `--multi_gpu`
- 每个GPU都会加载完整模型，显存需求不变
- 适用于长视频，短视频可能不会显著加速
- 确保所有GPU都有足够显存

## 致谢
- [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) @OpenImagingLab  
- [Sparse_SageAttention](https://github.com/jt-zhang/Sparse_SageAttention_API) @jt-zhang
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
