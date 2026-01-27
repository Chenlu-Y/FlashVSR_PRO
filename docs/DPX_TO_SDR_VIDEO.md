# 将 DPX 序列转换为 SDR 视频（修复灰色问题）

## 问题说明

生成的 DPX 文件是 **sRGB 编码**的（应用了伽马校正），但直接用 ffmpeg 转换为 SDR 视频时，如果没有正确指定颜色空间，会导致：
- 视频整体偏灰
- 饱和度和亮度不正确
- 看不到原视频内容

## 解决方案

使用专门的 SDR 视频编码工具，正确指定输入和输出颜色空间。

## 方法 1：使用工具脚本（推荐）

```bash
python utils/io/sdr_video_encode.py \
    --input /app/output/test_hdr_8K/ \
    --output /app/output/test_hdr_8K_sdr.mp4 \
    --fps 30.0 \
    --crf 18 \
    --preset slow
```

## 方法 2：直接使用 FFmpeg 命令

**关键**：必须显式指定输入为 sRGB (bt709)，输出也为 sRGB (bt709)

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

## 为什么需要显式指定颜色空间？

1. **DPX 文件是 sRGB 编码的**：
   - 保存时应用了 sRGB 伽马校正
   - 值范围：0-1（已归一化）

2. **FFmpeg 默认行为**：
   - 如果不指定输入颜色空间，FFmpeg 可能假设输入是线性 RGB
   - 这会导致错误的颜色空间转换，产生灰色视频

3. **正确的转换**：
   - 输入：sRGB (bt709) → 输出：sRGB (bt709)
   - 不需要颜色空间转换，只需要编码

## 参数说明

- `--crf 18`：高质量（可调整为 20-22 以减小文件）
- `--preset slow`：编码速度与质量平衡
- `-color_primaries bt709`：sRGB 使用 bt709 原色
- `-color_trc bt709`：sRGB 使用 bt709 传输特性（伽马 2.2）
- `-colorspace bt709`：sRGB 使用 bt709 颜色空间

## 验证生成的视频

```bash
# 检查视频信息
ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,r_frame_rate,color_space,color_primaries,color_trc \
    -of default=noprint_wrappers=1 \
    /app/output/test_hdr_8K_sdr.mp4
```

应该看到：
- `color_primaries=bt709`
- `color_trc=bt709`
- `color_space=bt709`

## 常见问题

### 1. 视频仍然是灰色的

**原因**：可能没有正确指定输入颜色空间

**解决**：确保命令中包含：
```bash
-color_primaries bt709 \
-color_trc bt709 \
-colorspace bt709
```

### 2. 颜色过饱和或过淡

**原因**：可能进行了不必要的颜色空间转换

**解决**：输入和输出都使用 bt709，不要转换

### 3. 文件太大

- 增加 CRF 值（如 20 或 22）
- 使用更快的 preset（文件会稍大但编码更快）

## 完整工作流示例

```bash
# 1. 超分生成 DPX 序列
python scripts/infer_video_distributed.py \
    --input /app/input/test_hdr_4K.mov \
    --output /app/output/test_hdr_8K/ \
    --output_mode pictures \
    --output_format dpx10 \
    --hdr_mode \
    --tone_mapping_method logarithmic \
    ...

# 2. 转换为 SDR 视频（正确显示）
python utils/io/sdr_video_encode.py \
    --input /app/output/test_hdr_8K/ \
    --output /app/output/test_hdr_8K_sdr.mp4 \
    --fps 30.0 \
    --crf 18 \
    --preset slow
```

## 对比：错误 vs 正确

### ❌ 错误的命令（会导致灰色视频）

```bash
ffmpeg -y \
    -framerate 30.0 \
    -pattern_type glob \
    -i "/app/output/test_hdr_8K/frame_*.dpx" \
    -vf "scale=7680:4320" \
    -c:v libx264 \
    /app/output/test_hdr_8K_sdr.mp4
```

**问题**：没有指定颜色空间，FFmpeg 可能假设输入是线性 RGB

### ✅ 正确的命令

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
    /app/output/test_hdr_8K_sdr.mp4
```

**关键**：显式指定输入和输出都是 bt709（sRGB）
