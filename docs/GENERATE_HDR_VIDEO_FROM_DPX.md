# 从 DPX 文件生成 HDR 视频

## 快速开始

使用以下命令将 DPX 文件转换为 HDR 视频：

```bash
python utils/io/hdr_video_encode.py \
    --input /app/output/test_hdr_8K/ \
    --output /app/output/test_hdr_8K_hdr10.mp4 \
    --fps 30.0 \
    --hdr_format hdr10 \
    --crf 18 \
    --preset slow \
    --simple
```

**关键参数**：
- `--input`: DPX 文件目录（包含 `frame_*.dpx` 文件）
- `--output`: 输出 HDR 视频路径
- `--fps`: 帧率（与原始视频一致）
- `--simple`: 使用简化模式（推荐）

> **注意**：默认假设 DPX 是 sRGB 编码（推荐）。如果 DPX 是线性 RGB 格式，添加 `--dpx_is_linear` 参数（不推荐，FFmpeg 转换可能有问题）。

## 完整工作流

### 步骤 1：从 Checkpoint 重新生成线性 RGB DPX

```bash
python tools/recover_distributed_inference.py \
    --checkpoint_dir /app/tmp/checkpoints/test_hdr_8K/ \
    --merge_partial \
    --output /app/output/test_hdr_8K_linear/ \
    --output_mode pictures \
    --output_format dpx10 \
    --fps 30.0 \
    --world_size 8
```

**注意**：恢复工具默认保存为线性 RGB（HDR格式），无需额外参数。

### 步骤 2：生成 HDR 视频

#### 方法 1：使用工具脚本（推荐）

```bash
python utils/io/hdr_video_encode.py \
    --input /app/output/test_hdr_8K_linear/ \
    --output /app/output/test_hdr_8K_hdr10.mp4 \
    --fps 30.0 \
    --hdr_format hdr10 \
    --crf 18 \
    --preset slow \
    --simple
```

#### 方法 2：直接使用 FFmpeg

```bash
ffmpeg -y \
    -framerate 30.0 \
    -pattern_type glob \
    -i "/app/output/test_hdr_8K_linear/frame_*.dpx" \
    -vf "scale=7680:4320:flags=lanczos,format=gbrpf32le,colorspace=bt2020:all=bt2020:trc=smpte2084:format=yuv420p10le" \
    -color_primaries bt2020 \
    -color_trc smpte2084 \
    -colorspace bt2020nc \
    -c:v libx265 \
    -preset slow \
    -crf 18 \
    -pix_fmt yuv420p10le \
    -x265-params "hdr10-opt=1:hdr10=1:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)" \
    /app/output/test_hdr_8K_hdr10.mp4
```

## 参数说明

### HDR 格式

- `hdr10`: 使用 PQ (Perceptual Quantizer) 曲线，需要 HDR10 兼容的显示器
- `hlg`: 使用 HLG (Hybrid Log-Gamma) 曲线，兼容性更好

### 质量参数

- `--crf 18`: 高质量（文件较大）
- `--crf 20`: 平衡（推荐）
- `--crf 22`: 标准质量
- `--crf 24-28`: 压缩率更高（文件更小）

### 编码预设

- `ultrafast`: 最快，质量稍低
- `fast`: 快速
- `medium`: 平衡（默认）
- `slow`: 较慢，质量更好（推荐）
- `veryslow`: 最慢，质量最好

### 亮度设置

- `--max_hdr_nits 1000`: 最大亮度 1000 nits（默认，避免过亮）
- `--max_hdr_nits 4000`: 最大亮度 4000 nits（适合高端 HDR 显示器）

## 验证生成的 HDR 视频

```bash
# 检查视频信息
ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,r_frame_rate,color_space,color_primaries,color_trc \
    -of default=noprint_wrappers=1 \
    /app/output/test_hdr_8K_hdr10.mp4
```

应该看到：
- `color_primaries=bt2020`
- `color_trc=smpte2084` (HDR10) 或 `arib-std-b67` (HLG)
- `color_space=bt2020nc`

## 常见问题

### 1. 视频全黑

**原因**：DPX 文件可能是 sRGB 编码的（SDR格式），而不是线性 RGB（HDR格式）

**解决方案**：
- 确保从 checkpoint 重新生成 DPX 时使用恢复工具（默认保存为线性 RGB）
- 或使用 `--dpx_is_srgb` 参数告诉工具输入是 sRGB 格式

### 2. 视频过亮或过暗

**原因**：亮度范围设置不当

**解决方案**：
- 调整 `--max_hdr_nits` 参数
- 对于标准 HDR 显示器，使用 1000 nits
- 对于高端 HDR 显示器，可以使用 4000 nits

### 3. 编码速度慢

**解决方案**：
- 使用更快的 preset（如 `fast` 或 `medium`）
- 降低分辨率（如果不需要原始分辨率）
- 使用硬件加速（如果支持）

### 4. 文件太大

**解决方案**：
- 增加 CRF 值（如 20 或 22）
- 使用更快的 preset（文件会稍大但编码更快）

## 完整示例

```bash
# 1. 从 checkpoint 重新生成线性 RGB DPX
python tools/recover_distributed_inference.py \
    --checkpoint_dir /app/tmp/checkpoints/test_hdr_8K/ \
    --merge_partial \
    --output /app/output/test_hdr_8K_linear/ \
    --output_mode pictures \
    --output_format dpx10 \
    --fps 30.0 \
    --world_size 8

# 2. 生成 HDR10 视频
python utils/io/hdr_video_encode.py \
    --input /app/output/test_hdr_8K_linear/ \
    --output /app/output/test_hdr_8K_hdr10.mp4 \
    --fps 30.0 \
    --hdr_format hdr10 \
    --crf 18 \
    --preset slow \
    --max_hdr_nits 1000 \
    --simple

# 3. 验证视频
ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,r_frame_rate,color_space,color_primaries,color_trc \
    -of default=noprint_wrappers=1 \
    /app/output/test_hdr_8K_hdr10.mp4
```

## 总结

✅ **推荐工作流：使用 sRGB 编码的 DPX（默认行为）**

✅ **使用 `--simple` 模式生成 HDR 视频（推荐）**

✅ **不推荐使用 `--dpx_is_linear`**（FFmpeg 的线性 RGB 转换可能有问题）

✅ **生成 DPX 时不要使用 `--dpx_linear_rgb`**（使用默认的 sRGB 编码）

现在您可以轻松地从 DPX 文件生成 HDR 视频了！
