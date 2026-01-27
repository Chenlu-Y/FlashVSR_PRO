# HDR 视频全黑问题解决方案

## 问题原因

**根本原因**：DPX 文件是 **sRGB 编码**的（已经是 SDR 格式），不能直接转换为 HDR。

### 技术细节

1. **DPX 文件保存流程**：
   ```
   HDR 值 (0-57) 
   → 归一化到 [0, 1]
   → 应用 sRGB 伽马校正
   → 保存为 DPX（sRGB 编码）
   ```

2. **转换为 HDR 的问题**：
   - DPX 文件已经是 sRGB（SDR）格式
   - sRGB → HDR 转换需要：sRGB → 线性 RGB → HDR 曲线
   - FFmpeg 可能无法自动完成这个转换，导致全黑

## 解决方案

### 方案 1：使用 SDR 视频（推荐）

**DPX 文件已经是 SDR 格式，直接生成 SDR 视频即可**：

```bash
python utils/io/sdr_video_encode.py \
    --input /app/output/test_hdr_8K/ \
    --output /app/output/test_hdr_8K_sdr.mp4 \
    --fps 30.0 \
    --crf 18 \
    --preset slow
```

**优点**：
- ✅ 颜色和亮度正确
- ✅ 所有设备都能播放
- ✅ 文件更小
- ✅ 不会出现全黑问题

### 方案 2：从 Checkpoint 重新生成线性 RGB 的 DPX

如果需要真正的 HDR 视频，需要从 checkpoint 重新生成 DPX，**不应用 sRGB 伽马校正**：

1. **修改 DPX 保存函数**（临时方案）：
   - 注释掉 sRGB 伽马校正
   - 保存为线性 RGB

2. **使用恢复工具重新生成**：
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

3. **从线性 RGB DPX 生成 HDR 视频**：
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

### 方案 3：使用修复后的 HDR 编码工具（尝试）

已修复的 HDR 编码工具使用 colorspace 滤镜进行转换：

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

**注意**：如果仍然全黑，说明 colorspace 滤镜无法正确处理转换。

## 推荐工作流

### 日常使用（推荐）

**直接生成 SDR 视频**，因为：
1. DPX 文件已经是 SDR 格式
2. 颜色和亮度正确
3. 所有设备都能播放

```bash
python utils/io/sdr_video_encode.py \
    --input /app/output/test_hdr_8K/ \
    --output /app/output/test_hdr_8K_sdr.mp4 \
    --fps 30.0
```

### 如果需要真正的 HDR 视频

1. **修改代码**：在 `save_frame_as_dpx10` 函数中，添加选项控制是否应用 sRGB 伽马
2. **重新生成 DPX**：不应用 sRGB 伽马，保存为线性 RGB
3. **生成 HDR 视频**：从线性 RGB DPX 生成 HDR 视频

## 为什么会出现全黑？

1. **DPX 是 sRGB 编码**：值范围 0-1，已应用伽马校正
2. **FFmpeg 转换失败**：sRGB → HDR 转换可能失败
3. **颜色空间不匹配**：输入是 SDR，输出是 HDR，转换不正确

## 快速检查

```bash
# 检查 DPX 文件的颜色空间（如果可能）
ffprobe -v error -select_streams v:0 \
    -show_entries stream=color_primaries,color_trc,colorspace \
    /app/output/test_hdr_8K/frame_000000.dpx
```

## 总结

**最佳实践**：
- ✅ 使用 SDR 视频编码工具（推荐）
- ❌ 避免从 sRGB DPX 直接生成 HDR 视频
- ✅ 如果需要 HDR，从 checkpoint 重新生成线性 RGB DPX

**当前 DPX 文件已经是 SDR 格式，直接生成 SDR 视频是最佳选择！**
