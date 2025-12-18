# FlashVSR 放大倍数机制详解

## 放大倍数的确定位置和原理

### 1. 参数定义位置

**文件**: `infer_video.py` 第1335行

```python
parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4], help="Upscale factor")
```

**限制说明**: 
- 代码中硬编码限制为 `[2, 3, 4]`
- 这是**唯一**限制放大倍数的地方
- 理论上可以修改为其他值，但需要确保模型支持

---

### 2. 放大倍数的应用流程

#### 阶段1: 输入预处理（双三次插值放大）

**文件**: `infer_video.py` 第132-177行

**关键函数**: `prepare_input_tensor()`

```python
def prepare_input_tensor(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16):
    """Prepare video tensor by upscaling and padding."""
    N0, h0, w0, _ = image_tensor.shape
    multiple = 128
    # 计算放大后的尺寸
    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=multiple)
    # ...
    # 对每一帧进行双三次插值放大
    tensor_chw = tensor_upscale_then_center_crop(frame_slice, scale, tW, tH)
```

**原理**:
1. **计算目标尺寸**: `sW = w0 * scale`, `sH = h0 * scale`
2. **双三次插值**: 使用 `F.interpolate(..., mode='bicubic')` 将输入从 `(h0, w0)` 放大到 `(h0*scale, w0*scale)`
3. **对齐到128的倍数**: 确保尺寸是128的倍数（模型要求）

**代码位置**: `infer_video.py` 第141-150行
```python
def tensor_upscale_then_center_crop(frame_tensor: torch.Tensor, scale: int, tW: int, tH: int):
    h0, w0, c = frame_tensor.shape
    tensor_bchw = frame_tensor.permute(2, 0, 1).unsqueeze(0)
    sW, sH = w0 * scale, h0 * scale  # ← 这里应用scale
    upscaled = F.interpolate(tensor_bchw, size=(sH, sW), mode='bicubic', align_corners=False)
    # 然后进行中心裁剪对齐到128的倍数
```

---

#### 阶段2: 模型推理（DiT + TCDecoder）

**文件**: `src/pipelines/flashvsr_tiny.py` 等

**流程**:
1. **DiT模型**: 处理放大后的低质量视频，生成潜在表示（latents）
2. **TCDecoder**: 将latents解码为最终的高质量视频

**关键**: 模型**不直接处理scale参数**，而是处理已经放大到目标尺寸的输入视频。

---

#### 阶段3: TCDecoder的空间上采样

**文件**: `src/models/TCDecoder.py` 第175-208行

**关键代码**:
```python
decoder_space_upscale=(True, True, True),  # 3层空间上采样
# ...
nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),  # 第1层：2倍
nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),  # 第2层：2倍
nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),  # 第3层：2倍
```

**原理**:
- TCDecoder有**3层固定的2倍上采样层**
- 理论上可以实现 `2^3 = 8倍` 放大
- 但实际只支持到4倍，因为：
  1. 输入已经通过双三次插值放大到目标尺寸
  2. TCDecoder主要用于**质量增强**而非尺寸放大
  3. 模型训练时只针对2/3/4倍进行了优化

---

### 3. 为什么限制为2/3/4倍？

#### 技术原因

1. **模型架构限制**:
   - TCDecoder的3层2倍上采样是固定的
   - 虽然理论上可以支持更大倍数，但未经过训练验证

2. **训练数据限制**:
   - FlashVSR模型只在2/3/4倍数据上训练
   - 其他倍数可能导致质量下降

3. **显存和性能考虑**:
   - 更大的放大倍数需要更多显存
   - 处理时间会显著增加

#### 代码中的限制

**位置1**: `infer_video.py` 第1335行
```python
choices=[2, 3, 4]  # ← 硬编码限制
```

**位置2**: `nodes.py` 第419-423行（ComfyUI节点）
```python
"scale": ("INT", {
    "default": 2,
    "min": 2,
    "max": 4,  # ← 最大4倍
}),
```

---

### 4. 放大倍数的实际工作流程

```
原始视频 (1920x1080)
    ↓
[阶段1: 双三次插值] ← scale=4
    ↓
放大后输入 (7680x4320)  ← 1920*4 x 1080*4
    ↓
[阶段2: DiT模型处理]
    ↓
潜在表示 (latents)
    ↓
[阶段3: TCDecoder解码]
    ↓ (3层2倍上采样，但输入已经是目标尺寸)
最终输出 (7680x4320)  ← 4倍放大完成
```

**关键理解**:
- **scale参数主要作用于阶段1**（双三次插值）
- **TCDecoder的上采样层主要用于质量增强**，而非尺寸放大
- 最终的放大倍数 = scale参数值

---

### 5. 是否可以支持其他倍数？

#### 理论上可以

如果修改代码支持5倍或更大倍数：

1. **修改参数限制**:
```python
# infer_video.py 第1335行
parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4, 5, 6], ...)
```

2. **确保尺寸对齐**:
   - 放大后的尺寸必须是128的倍数
   - `compute_scaled_and_target_dims()` 函数会自动处理

3. **潜在问题**:
   - 模型未在5倍+数据上训练，质量可能下降
   - 显存需求会大幅增加
   - 处理时间会显著增加

#### 实际建议

- **2倍**: 显存占用最低，速度快
- **3倍**: 平衡质量和显存（**完全支持，不限于2的倍数**）
- **4倍**: 最佳质量（推荐）
- **>4倍**: 不推荐，质量可能下降

---

### 6. 关键代码位置总结

| 功能 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 参数定义 | `infer_video.py` | 1335 | `choices=[2, 3, 4]` 限制 |
| 尺寸计算 | `infer_video.py` | 132-139 | `compute_scaled_and_target_dims()` |
| 双三次插值 | `infer_video.py` | 141-150 | `tensor_upscale_then_center_crop()` |
| 输入准备 | `infer_video.py` | 152-177 | `prepare_input_tensor()` |
| TCDecoder上采样 | `src/models/TCDecoder.py` | 193, 198, 203 | 3层2倍上采样 |
| 输出尺寸计算 | `infer_video.py` | 1139, 1167 | `H * args.scale, W * args.scale` |

---

### 7. 总结

1. **放大倍数的确定**: 由 `--scale` 参数决定，限制在 `[2, 3, 4]`
2. **放大原理**: 
   - 主要通过**双三次插值**将输入放大到目标尺寸
   - TCDecoder的2倍上采样层主要用于**质量增强**
3. **为什么支持3倍**: 
   - 3倍不是2的倍数，但完全支持
   - 因为放大是通过双三次插值实现的，不依赖2的幂次
4. **限制原因**: 
   - 模型训练数据限制
   - 质量保证
   - 显存和性能考虑

