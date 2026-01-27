# 将 DPX 序列转换为 HDR 视频

## 概述

生成的 DPX 文件可以转换为 HDR 视频（H.265/HEVC with HDR10/HLG）。由于当前 DPX 文件保存时应用了 sRGB 伽马校正，需要正确的颜色空间转换。

## 方法 1：使用提供的工具脚本（推荐）

```bash
python utils/io/hdr_video_encode.py \
    --input /app/output/test_hdr_8K/ \
    --output /app/output/test_hdr_8K_hdr10.mp4 \
    --fps 30.0 \
    --hdr_format hdr10 \
    --crf 18 \
    --preset slow
```

参数说明：
- `--input`: DPX 文件目录（包含 `frame_*.dpx` 文件）
- `--output`: 输出视频路径
- `--fps`: 帧率（与原始视频一致）
- `--hdr_format`: `hdr10` 或 `hlg`
- `--crf`: 质量参数（18-28，越小质量越高，文件越大）
- `--preset`: 编码速度（ultrafast, fast, medium, slow, veryslow）
- `--simple`: 使用简化模式（如果遇到颜色空间转换问题）

## 方法 2：直接使用 FFmpeg 命令

### HDR10 格式（推荐，兼容性最好）

```bash
ffmpeg -y \
    -framerate 30.0 \
    -pattern_type glob \
    -i "/app/output/test_hdr_8K/frame_*.dpx" \
    -vf "scale=7680:4320:flags=lanczos,format=p010le" \
    -color_primaries bt2020 \
    -color_trc smpte2084 \
    -colorspace bt2020nc \
    -c:v libx265 \
    -preset slow \
    -crf 18 \
    -pix_fmt yuv420p10le \
    -x265-params "hdr10-opt=1:hdr10=1:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1000)" \
    /app/output/test_hdr_8K_hdr10.mp4
```

### HLG 格式

```bash
ffmpeg -y \
    -framerate 30.0 \
    -pattern_type glob \
    -i "/app/output/test_hdr_8K/frame_*.dpx" \
    -vf "scale=7680:4320:flags=lanczos,format=p010le" \
    -color_primaries bt2020 \
    -color_trc arib-std-b67 \
    -colorspace bt2020nc \
    -c:v libx265 \
    -preset slow \
    -crf 18 \
    -pix_fmt yuv420p10le \
    -x265-params "hdr10-opt=1" \
    /app/output/test_hdr_8K_hlg.mp4
```

## 重要说明

### 颜色空间转换

当前 DPX 文件是 **sRGB 编码**的（应用了伽马校正），而 HDR 视频需要：

1. **HDR10**: 使用 PQ (Perceptual Quantizer) 曲线，需要线性 RGB → PQ
2. **HLG**: 使用 Hybrid Log-Gamma 曲线，需要线性 RGB → HLG

FFmpeg 会自动处理颜色空间转换，但可能需要指定正确的输入颜色空间：

```bash
# 如果遇到颜色问题，可以显式指定输入为 sRGB
ffmpeg -y \
    -framerate 30.0 \
    -pattern_type glob \
    -i "/app/output/test_hdr_8K/frame_*.dpx" \
    -vf "scale=7680:4320:flags=lanczos,format=p010le,colorspace=bt709:iall=bt709:fast=1" \
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

### 参数调整

- **CRF 值**：
  - 18: 高质量（文件较大）
  - 20: 平衡（推荐）
  - 22: 标准质量
  - 24-28: 压缩率更高（文件更小）

- **Preset**：
  - `ultrafast`: 最快，质量稍低
  - `fast`: 快速
  - `medium`: 平衡（默认）
  - `slow`: 较慢，质量更好（推荐）
  - `veryslow`: 最慢，质量最好

### 验证 HDR 视频

```bash
# 检查视频信息
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,color_space,color_primaries,color_trc -of default=noprint_wrappers=1 /app/output/test_hdr_8K_hdr10.mp4
```

应该看到：
- `color_primaries=bt2020`
- `color_trc=smpte2084` (HDR10) 或 `arib-std-b67` (HLG)
- `color_space=bt2020nc`

## 常见问题

### 1. 颜色不正确

如果生成的 HDR 视频颜色不正确，尝试：
- 使用 `--simple` 模式
- 或手动指定输入颜色空间（如上所示）

### 2. 编码速度慢

- 使用更快的 preset（如 `fast` 或 `medium`）
- 降低分辨率（如果不需要原始分辨率）
- 使用硬件加速（如果支持）

### 3. 文件太大

- 增加 CRF 值（如 20 或 22）
- 使用更快的 preset（文件会稍大但编码更快）

## 示例：完整工作流

```bash
# 1. 超分生成 DPX 序列
python scripts/infer_video_distributed.py \
    --input /app/input/test_hdr_4K.mov \
    --output /app/output/test_hdr_8K/ \
    --output_mode pictures \
    --output_format dpx10 \
    --hdr_mode \
    --tone_mapping_method logarithmic \
    --mode tiny \
    --scale 2 \
    --fps 30.0

# 2. 转换为 HDR 视频
python utils/io/hdr_video_encode.py \
    --input /app/output/test_hdr_8K/ \
    --output /app/output/test_hdr_8K_hdr10.mp4 \
    --fps 30.0 \
    --hdr_format hdr10 \
    --crf 18 \
    --preset slow
```

## 注意事项

1. **HDR10 vs HLG**：
   - HDR10: 更广泛支持，需要显示设备支持 HDR10
   - HLG: 向后兼容 SDR 显示器，但需要支持 HLG 的设备

2. **播放器支持**：
   - VLC, mpv 等支持 HDR 视频播放
   - 需要 HDR 显示器才能看到真正的 HDR 效果

3. **文件大小**：
   - HDR 视频文件通常比 SDR 大 2-3 倍
   - 8K HDR 视频可能达到几十 GB
