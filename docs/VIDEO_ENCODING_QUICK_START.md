# DPX 转视频快速指南（修复灰色和过亮问题）

## 问题说明

- **HDR 视频**：亮度太高，设备看不了
- **SDR 视频**：整体灰色，饱和度和亮度不对

## 快速解决方案

### 1. 生成 SDR 视频（推荐，所有设备都能看）

```bash
python utils/io/sdr_video_encode.py \
    --input /app/output/test_hdr_8K/ \
    --output /app/output/test_hdr_8K_sdr.mp4 \
    --fps 30.0 \
    --crf 18 \
    --preset slow
```

**关键**：这个工具会正确指定输入为 sRGB，输出也为 sRGB，避免灰色问题。

### 2. 生成 HDR 视频（如果设备支持）

```bash
python utils/io/hdr_video_encode.py \
    --input /app/output/test_hdr_8K/ \
    --output /app/output/test_hdr_8K_hdr10.mp4 \
    --fps 30.0 \
    --hdr_format hdr10 \
    --max_hdr_nits 1000 \
    --crf 18 \
    --preset slow \
    --simple
```

**关键修复**：
- `--max_hdr_nits 1000`：设置合理的亮度范围（避免过亮）
- `--simple`：使用简化模式，避免复杂的颜色空间转换问题

## 直接使用 FFmpeg（如果工具不可用）

### SDR 视频（修复灰色问题）

```bash
ffmpeg -y \
    -framerate 30.0 \
    -pattern_type glob \
    -i "/app/output/test_hdr_8K/frame_*.dpx" \
    -vf "scale=7680:4320:flags=lanczos" \
    -color_primaries bt709 \
    -color_trc bt709 \
    -colorspace bt709 \
    -c:v libx264 \
    -preset slow \
    -crf 18 \
    -pix_fmt yuv420p \
    -movflags +faststart \
    /app/output/test_hdr_8K_sdr.mp4
```

**关键参数**：
- `-color_primaries bt709`：指定输入为 sRGB
- `-color_trc bt709`：指定输入伽马为 sRGB
- `-colorspace bt709`：指定输入颜色空间为 sRGB

### HDR 视频（修复过亮问题）

```bash
ffmpeg -y \
    -framerate 30.0 \
    -pattern_type glob \
    -i "/app/output/test_hdr_8K/frame_*.dpx" \
    -color_primaries bt709 \
    -color_trc bt709 \
    -colorspace bt709 \
    -vf "scale=7680:4320:flags=lanczos" \
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

**关键修复**：
- 输入指定为 `bt709`（sRGB）
- 亮度范围设置为 `L(10000000,50)` = 1000 nits（避免过亮）

## 为什么会出现这些问题？

### 灰色视频的原因

1. **DPX 文件是 sRGB 编码的**（应用了伽马校正）
2. **FFmpeg 默认假设输入是线性 RGB**
3. **错误的颜色空间转换**导致灰色显示

**解决方案**：显式指定输入为 sRGB (bt709)

### HDR 视频过亮的原因

1. **亮度范围设置过高**（可能使用了原始 HDR 最大值）
2. **颜色空间转换不正确**

**解决方案**：
- 设置合理的亮度范围（1000 nits）
- 显式指定输入颜色空间

## 验证生成的视频

### 检查 SDR 视频

```bash
ffprobe -v error -select_streams v:0 \
    -show_entries stream=color_primaries,color_trc,colorspace \
    /app/output/test_hdr_8K_sdr.mp4
```

应该看到：`bt709`（所有三个都是）

### 检查 HDR 视频

```bash
ffprobe -v error -select_streams v:0 \
    -show_entries stream=color_primaries,color_trc,colorspace \
    /app/output/test_hdr_8K_hdr10.mp4
```

应该看到：
- `color_primaries=bt2020`
- `color_trc=smpte2084` (HDR10)
- `color_space=bt2020nc`

## 推荐工作流

1. **生成 SDR 视频**（用于日常查看和分享）
   ```bash
   python utils/io/sdr_video_encode.py \
       --input /app/output/test_hdr_8K/ \
       --output /app/output/test_hdr_8K_sdr.mp4 \
       --fps 30.0
   ```

2. **生成 HDR 视频**（用于 HDR 设备，可选）
   ```bash
   python utils/io/hdr_video_encode.py \
       --input /app/output/test_hdr_8K/ \
       --output /app/output/test_hdr_8K_hdr10.mp4 \
       --fps 30.0 \
       --max_hdr_nits 1000 \
       --simple
   ```

## 如果仍然有问题

1. **检查 DPX 文件**：确认文件是否正确生成
2. **检查颜色空间**：使用 `ffprobe` 检查视频元数据
3. **尝试不同的播放器**：VLC、mpv 等
4. **调整亮度范围**：如果 HDR 仍然过亮，降低 `--max_hdr_nits`（如 500 或 800）
