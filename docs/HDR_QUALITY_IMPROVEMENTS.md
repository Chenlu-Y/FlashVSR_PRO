# HDR 质量改进指南

## 默认行为（已优化）

**现在默认使用全局 Tone Mapping 参数**，无需额外指定。所有帧使用相同的 `l_max`，避免帧间频闪。

## 推荐命令

```bash
docker exec -w /app/FlashVSR_Ultra_Fast flashvsr_ultra_fast python scripts/infer_video_distributed.py \
    --input /app/input/test_hdr_4K.mov \
    --output /app/output/test_hdr_8K/ \
    --output_mode pictures \
    --output_format dpx10 \
    --hdr_mode \
    --tone_mapping_method logarithmic \
    --mode tiny \
    --scale 2 \
    --fps 30.0 \
    --tiled_dit True \
    --unload_dit False \
    --tiled_vae True \
    --attention_mode block_sparse_attention \
    --use_shared_memory true \
    --devices all \
    --tile_size 512 \
    --tile_overlap 384 \
    --color_fix True \
    --adaptive_tile_batch False \
    --tile_batch_size 32
```

**关键参数**：
- `--hdr_mode`：启用 HDR 处理流程
- `--tile_overlap 384`：增加重叠（原来是 256），减少边界伪影

## 可选参数

| 参数 | 说明 |
|------|------|
| `--per_frame_tone_mapping` | 使用每帧独立参数（不推荐，可能导致频闪） |
| `--global_l_max 57.0` | 手动指定全局 l_max（如果已知原始视频的最大亮度值） |

## 技术说明

### 为什么会频闪？

旧版本每帧独立计算 `l_max`：

```
帧 1: l_max = 55.0 → 还原结果 A
帧 2: l_max = 57.0 → 还原结果 B（亮度跳变！）
帧 3: l_max = 54.0 → 还原结果 C（又跳变！）
```

**现在的默认行为**（全局参数）：

```
所有帧: l_max = 57.0（segment 的最大值）
帧 1, 2, 3... → 还原结果一致，无频闪
```

## 生成 HDR 视频

从线性 RGB DPX 生成 HDR 视频：

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

## 减少边界伪影

如果高亮区域（如海面）有明显光斑，可以：

1. **增加 tile 重叠**：`--tile_overlap 384` 或更高
2. **增大 tile 尺寸**：`--tile_size 768 --tile_overlap 384`
