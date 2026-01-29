# DPX 文件格式说明：sRGB vs 线性 RGB

## 问题：为什么 DPX 文件是 SDR 格式的？

**您说得对！** DPX 文件**应该**是 HDR 格式的，但之前的实现中应用了 sRGB 伽马校正，导致保存为 SDR 格式。

## 两种 DPX 格式

### 1. sRGB 编码（SDR格式）- 默认

**特点**：
- 应用了 sRGB 伽马校正
- 值范围：0-1（已归一化）
- **优点**：在标准显示器上正确显示
- **缺点**：不是真正的 HDR，转换为 HDR 视频可能有问题

**保存流程**：
```
HDR 值 (0-57)
  → 归一化到 [0, 1]
  → 应用 sRGB 伽马校正
  → 保存为 DPX（sRGB 编码，SDR格式）
```

### 2. 线性 RGB（HDR格式）- 新增选项

**特点**：
- 保持线性 RGB，不应用伽马校正
- 值范围：0-1（归一化后）
- **优点**：真正的 HDR 格式，可用于生成 HDR 视频
- **缺点**：在标准显示器上查看时会显示为灰色（需要 HDR 显示器或正确的颜色空间转换）

**保存流程**：
```
HDR 值 (0-57)
  → 归一化到 [0, 1]
  → 保持线性 RGB（不应用伽马）
  → 保存为 DPX（线性 RGB，HDR格式）
```

## 使用方法

### 保存为 sRGB DPX（SDR格式，默认）

用于在标准显示器上查看：

```bash
python scripts/infer_video_distributed.py \
    --input /app/input/test_hdr_4K.mov \
    --output /app/output/test_hdr_8K/ \
    --output_mode pictures \
    --output_format dpx10 \
    --hdr_mode \
    --tone_mapping_method logarithmic \
    ...
```

**默认行为**：应用 sRGB 伽马校正，保存为 SDR 格式

### 保存为线性 RGB DPX（HDR格式）

用于生成 HDR 视频：

```bash
python scripts/infer_video_distributed.py \
    --input /app/input/test_hdr_4K.mov \
    --output /app/output/test_hdr_8K_linear/ \
    --output_mode pictures \
    --output_format dpx10 \
    --hdr_mode \
    --tone_mapping_method logarithmic \
    --dpx_linear_rgb \
    ...
```

**关键参数**：`--dpx_linear_rgb` - 保存为线性 RGB（HDR格式）

## 从 Checkpoint 重新生成

恢复工具默认保存为**线性 RGB（HDR格式）**，因为通常用于生成 HDR 视频：

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

## 生成 HDR 视频

从线性 RGB DPX 生成 HDR 视频：

```bash
ffmpeg -y \
    -framerate 30.0 \
    -pattern_type glob \
    -i "/app/output/test_hdr_8K_linear/frame_*.dpx" \
    -vf "scale=7680:4320:flags=lanczos" \
    -color_primaries bt2020 \
    -color_trc smpte2084 \
    -colorspace bt2020nc \
    -c:v libx265 \
    -preset slow \
    -crf 18 \
    -pix_fmt yuv420p10le \
    -x265-params "hdr10-opt=1:hdr10=1" \
    /app/output/test_hdr_8K_hdr10.mp4
```

## 生成 SDR 视频

从 sRGB DPX 生成 SDR 视频（推荐）：

```bash
python utils/io/sdr_video_encode.py \
    --input /app/output/test_hdr_8K/ \
    --output /app/output/test_hdr_8K_sdr.mp4 \
    --fps 30.0
```

## 对比

| 格式 | 编码方式 | 用途 | 显示器显示 |
|------|---------|------|-----------|
| **sRGB DPX** | sRGB 伽马校正 | 标准显示器查看、生成 SDR 视频 | ✅ 正确显示 |
| **线性 RGB DPX** | 线性 RGB | 生成 HDR 视频 | ⚠️ 显示为灰色（需要 HDR 显示器） |

## 推荐工作流

### 方案 1：生成两种格式（推荐）

```bash
# 1. 生成 sRGB DPX（用于查看和 SDR 视频）
python scripts/infer_video_distributed.py \
    --input /app/input/test_hdr_4K.mov \
    --output /app/output/test_hdr_8K_srgb/ \
    --output_mode pictures \
    --output_format dpx10 \
    --hdr_mode \
    ...

# 2. 从 checkpoint 重新生成线性 RGB DPX（用于 HDR 视频）
python tools/recover_distributed_inference.py \
    --checkpoint_dir /app/tmp/checkpoints/test_hdr_8K_srgb/ \
    --merge_partial \
    --output /app/output/test_hdr_8K_linear/ \
    --output_mode pictures \
    --output_format dpx10 \
    --fps 30.0 \
    --world_size 8

# 3. 生成 SDR 视频（从 sRGB DPX）
python utils/io/sdr_video_encode.py \
    --input /app/output/test_hdr_8K_srgb/ \
    --output /app/output/test_hdr_8K_sdr.mp4 \
    --fps 30.0

# 4. 生成 HDR 视频（从线性 RGB DPX）
ffmpeg -y \
    -framerate 30.0 \
    -pattern_type glob \
    -i "/app/output/test_hdr_8K_linear/frame_*.dpx" \
    -vf "scale=7680:4320:flags=lanczos" \
    -color_primaries bt2020 \
    -color_trc smpte2084 \
    -colorspace bt2020nc \
    -c:v libx265 \
    -preset slow \
    -crf 18 \
    -pix_fmt yuv420p10le \
    -x265-params "hdr10-opt=1:hdr10=1" \
    /app/output/test_hdr_8K_hdr10.mp4
```

### 方案 2：直接生成线性 RGB DPX

如果只需要 HDR 视频：

```bash
python scripts/infer_video_distributed.py \
    --input /app/input/test_hdr_4K.mov \
    --output /app/output/test_hdr_8K_linear/ \
    --output_mode pictures \
    --output_format dpx10 \
    --hdr_mode \
    --dpx_linear_rgb \
    ...
```

## 总结

✅ **推荐使用 sRGB 编码的 DPX**（默认行为）：FFmpeg 转换更稳定，结果更可靠

✅ **`--dpx_linear_rgb` 参数仍然可用**，但不推荐使用（FFmpeg 的线性 RGB 转换可能有问题）

✅ **默认行为**：保存为 sRGB 编码，使用标准的 HDR 视频转换流程

✅ **如果遇到 HDR 视频亮度问题**：确保不使用 `--dpx_linear_rgb` 和 `--dpx_is_linear` 参数

推荐的 HDR 工作流：
1. 生成 DPX 时使用默认设置（sRGB 编码）
2. 转换为 HDR 视频时使用默认设置（不需要 `--dpx_is_linear`）
