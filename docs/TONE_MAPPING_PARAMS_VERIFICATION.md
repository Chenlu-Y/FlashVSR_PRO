# Tone Mapping 参数保存和还原验证

## 确认：参数确实被保存

### 1. 参数保存时机

在 **HDR → SDR 转换时**（第1882-1897行），代码会：

```python
# 步骤1: 应用 Tone Mapping，同时获取参数
segment_frames, tone_mapping_params = apply_tone_mapping_to_frames(
    segment_frames,
    method=args.tone_mapping_method,
    exposure=args.tone_mapping_exposure,
    per_frame=True  # 每帧独立参数
)

# 步骤2: 保存参数到 JSON 文件
params_file = os.path.join(checkpoint_dir, f"rank_{rank}_tone_mapping_params.json")
serialized_params = serialize_tone_mapping_params(tone_mapping_params)
with open(params_file, 'w') as f:
    json.dump(serialized_params, f, indent=2)
```

### 2. 保存的参数内容

对于 **logarithmic** 方法，每帧保存的参数包括：

```json
{
  "method": "logarithmic",
  "exposure": 1.0,
  "l_max": 57.0,        // 压缩时的最大值 (hdr * exposure).max()
  "max_hdr": 57.0       // 原始 HDR 最大值 hdr.max()
}
```

**关键参数说明**：
- `l_max`: 这是压缩时使用的最大值，用于逆映射公式
- `max_hdr`: 原始 HDR 最大值，用于限制还原后的范围
- `exposure`: 曝光调整值，用于逆映射时还原

### 3. 参数使用时机

在 **超分后还原 HDR** 时（第1923-1930行），代码会：

```python
# 使用保存的参数进行逆映射
if tone_mapping_params is not None:
    segment_output = apply_inverse_tone_mapping_to_frames(
        segment_output,  # 超分后的 SDR 帧
        tone_mapping_params  # 之前保存的参数
    )
```

### 4. 逆映射公式（logarithmic）

使用保存的参数进行还原：

```python
# 从参数中获取
l_max = params['l_max']  # 例如 57.0
exposure = params['exposure']  # 例如 1.0

# 逆映射公式
ldr = exp(sdr * log(1 + l_max)) - 1
hdr = ldr / exposure

# 限制到原始范围
hdr = clamp(hdr, 0.0, max_hdr * 1.1)
```

## 验证参数文件

### 方法 1：使用验证工具

```bash
python tools/verify_tone_mapping_params.py \
    --checkpoint_dir /app/tmp/checkpoints/test_hdr_8K/ \
    --rank 0
```

### 方法 2：手动检查

```bash
# 查看参数文件
cat /app/tmp/checkpoints/test_hdr_8K/rank_0_tone_mapping_params.json | head -20
```

应该看到类似这样的内容：

```json
[
  {
    "method": "logarithmic",
    "exposure": 1.0,
    "l_max": 57.0753,
    "max_hdr": 57.0753
  },
  {
    "method": "logarithmic",
    "exposure": 1.0,
    "l_max": 56.5582,
    "max_hdr": 56.5582
  },
  ...
]
```

## 参数文件位置

参数文件保存在 checkpoint 目录中：

```
/app/tmp/checkpoints/{输出名}/
  ├── rank_0_tone_mapping_params.json
  ├── rank_1_tone_mapping_params.json
  ├── rank_2_tone_mapping_params.json
  └── ...
```

## 参数完整性检查清单

✅ **参数保存**：
- [x] 在 HDR → SDR 转换时保存
- [x] 每帧独立参数（per_frame=True）
- [x] 保存到 JSON 文件
- [x] 包含所有必需字段

✅ **参数使用**：
- [x] 在超分后使用参数还原
- [x] 从内存中读取（如果可用）
- [x] 可以从文件重新加载（恢复工具）

✅ **参数内容**：
- [x] method: 方法名称
- [x] exposure: 曝光值
- [x] l_max: 压缩时的最大值（关键！）
- [x] max_hdr: 原始 HDR 最大值（关键！）

## 验证流程

1. **运行推理**：
   ```bash
   python scripts/infer_video_distributed.py \
       --input /app/input/test_hdr_4K.mov \
       --output /app/output/test_hdr_8K/ \
       --hdr_mode \
       --tone_mapping_method logarithmic \
       ...
   ```

2. **检查日志**：
   应该看到：
   ```
   [Rank X] [HDR] Tone Mapping 参数已保存: /app/tmp/checkpoints/.../rank_X_tone_mapping_params.json
   ```

3. **验证参数文件**：
   ```bash
   python tools/verify_tone_mapping_params.py \
       --checkpoint_dir /app/tmp/checkpoints/test_hdr_8K/
   ```

4. **检查还原结果**：
   日志中应该看到：
   ```
   [Rank X] [HDR] HDR 还原完成，范围: [0.0000, 57.0753]
   ```

## 常见问题

### Q: 参数文件不存在？

**A**: 检查：
1. 是否启用了 `--hdr_mode`
2. 是否检测到 HDR 值（值 > 1.0）
3. checkpoint 目录路径是否正确

### Q: 参数文件为空？

**A**: 可能是：
1. 没有检测到 HDR 值（所有值 <= 1.0）
2. 处理过程中出错

### Q: 还原后的 HDR 值不正确？

**A**: 检查：
1. 参数文件中的 `l_max` 和 `max_hdr` 是否合理
2. 是否使用了正确的方法（logarithmic/reinhard/aces）
3. 参数文件是否与帧数匹配

## 总结

✅ **确认**：参数确实在 HDR → SDR 转换时被保存
✅ **确认**：参数包含所有必需信息（l_max, max_hdr, exposure, method）
✅ **确认**：参数在超分后用于还原 HDR
✅ **确认**：参数文件可以用于从 checkpoint 重新生成 DPX

可以放心重新运行推理进行验证！
