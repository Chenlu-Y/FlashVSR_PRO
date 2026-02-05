# FlashVSR_Ultra_Fast
在低显存环境下运行 FlashVSR，同时保持无伪影高质量输出。  
**[[📃English](./README.md)]**

## 更新日志
#### 2025-10-31
- **新增:** 独立的 `infer_video.py` 脚本，无需 ComfyUI 即可处理视频
- **新增:** 多GPU并行处理 (`--multi_gpu`) - 自动将视频按帧分割到多个GPU
- **新增:** 自适应tile批处理 (`--adaptive_batch_size`) - 根据GPU显存动态调整tile并发数
- **新增:** 流式处理模式 (`--streaming`) - 处理长视频时以块为单位处理，降低显存占用
- **新增:** 分段处理模式 (`--segmented`) - 单GPU场景下将视频分成多个子段处理
- **新增:** 断点续传功能 (`--resume`) - 自动检测并合并之前运行中已完成的帧
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


## 安装步骤

📢: 要在RTX20系或更早的GPU上运行, 请安装`triton<3.3.0`:  

```bash
# Windows
python -m pip install -U triton-windows<3.3.0
# Linux
python -m pip install -U triton<3.3.0
```

### 模型下载

- 从[这里](https://huggingface.co/JunhaoZhuang/FlashVSR)下载整个`FlashVSR`文件夹和它里面的所有文件, 并将其放到模型目录中（默认：`/app/models/v1.1/`）

```
├── FlashVSR
|     ├── LQ_proj_in.ckpt
|     ├── TCDecoder.ckpt
|     ├── diffusion_pytorch_model_streaming_dmd.safetensors
|     ├── Wan2.1_VAE.pth
```

## 使用说明

### 脚本概览

提供两个推理脚本：

1. **`scripts/infer_video.py`**: 单进程推理，支持通过帧分割实现多GPU支持
2. **`scripts/infer_video_distributed.py`**: 真正的分布式推理，支持模型并行（推荐用于多GPU环境）

### 参数参考

#### 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | str | **必需** | 输入视频路径或图像序列目录 |
| `--output` | str | None | 输出视频路径（未指定时自动生成） |
| `--model_ver` | str | `1.1` | 模型版本：`1.0` 或 `1.1` |
| `--mode` | str | `tiny` | 模型模式：`tiny`（更快）、`full`（更高质量）、`tiny-long`（适合长视频） |
| `--device` | str | `cuda:0` | 使用的设备（使用 `--multi_gpu` 时会被忽略） |
| `--scale` | int | `2` (infer_video.py)<br>`4` (infer_video_distributed.py) | 放大倍数：`2`、`3` 或 `4` |

#### 质量与处理参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--color_fix` | bool | `True` | 使用小波变换修正输出视频颜色 |
| `--tiled_vae` | bool | `True` | 使用分块VAE以降低显存占用（速度较慢） |
| `--tiled_dit` | bool | `False` | 使用分块DiT以显著降低显存占用（速度较慢） |
| `--tile_size` | int | `256` | 分块处理时的tile大小 |
| `--tile_overlap` | int | `24` | tile重叠像素数 |
| `--tile_shift` | bool | `True` | 启用 tile 错位双遍推理，两遍结果平均以减轻网格接缝（仅当 `--tiled_dit` 开启时生效） |
| `--unload_dit` | bool | `False` | 解码前卸载DiT以降低显存峰值（速度较慢） |
| `--precision` | str | `bf16` | 精度：`fp32`、`fp16` 或 `bf16` |
| `--attention_mode` | str | `sparse_sage_attention` | 注意力模式：`sparse_sage_attention` 或 `block_sparse_attention` |
| `--sparse_ratio` | float | `2.0` | 稀疏注意力比率 |
| `--kv_ratio` | float | `3.0` | KV缓存比率 |
| `--local_range` | int | `11` | 局部注意力范围 |
| `--seed` | int | `0` | 随机种子，用于可重现性 |

#### 性能优化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--multi_gpu` | 标志 | False | 启用多GPU并行处理（按帧将视频分割到多个GPU） |
| `--adaptive_batch_size` | 标志 | False | 启用自适应tile批处理大小（根据GPU显存动态调整） |
| `--streaming` | 标志 | False | 启用流式处理模式（以块为单位处理长视频，降低显存占用） |
| `--segmented` | 标志 | False | 启用分段处理模式（将视频分成多个子段处理，类似 `--multi_gpu` 但用于单worker） |
| `--segment_overlap` | int | `2` | 段/块之间的重叠帧数（范围：1-10，推荐：1-5） |
| `--max-segment-frames` | int | None | 分段模式中每段的最大帧数（默认：根据显存自动计算） |

#### 恢复与检查点参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--resume` | 标志 | False | 从检查点恢复：自动检测并合并之前运行中已完成的帧（与 `--multi_gpu` 和 `--segmented` 配合使用） |
| `--clean-checkpoint` | 标志 | False | 开始前清理检查点目录（禁用恢复） |

#### 分布式推理参数（仅 infer_video_distributed.py）

**输入 / 输出相关（I/O & Format）**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--output_mode` | str | `video` | 输出模式：`video`（单个视频文件）或 `pictures`（图像序列） |
| `--output_format` | str | `mp4` / `png` | 当 `output_mode=video` 时：`mp4` / `mov` / `mkv`；当 `output_mode=pictures` 时：`png` / `dpx` |
| `--output_bit_depth` | int | `8` | 输出位深：`8` 或 `10`（视频模式映射到 `yuv420p` / `yuv420p10le`，图片模式映射到 8bit PNG 或 10bit DPX） |
| `--input_fps` | float | `30.0` | 输入为图像序列时的帧率（视频文件会自动探测 FPS，不使用该参数） |
| `--output_fps` | float | `None` | 输出视频的帧率，仅当 `output_mode=video` 时生效。未指定时自动继承输入 FPS |

**动态范围 / HDR 通道（Dynamic Range & HDR）**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dynamic_range` | str | `sdr` | 动态范围：`sdr`（标准动态范围）/ `hdr`（高动态范围）/ `auto`（自动检测） |
| `--hdr_preprocess` | str | `hlg` | HDR 预处理方式（仅当 `--dynamic_range=hdr` 时生效）：`hlg`（PQ→HLG 工作流，推荐）或 `tone_mapping`（可逆 Tone Mapping 工作流） |
| `--tone_mapping_method` | str | `logarithmic` | Tone Mapping 方法（仅当 `--hdr_preprocess=tone_mapping` 时生效）：`reinhard` / `logarithmic` / `aces` |
| `--tone_mapping_exposure` | float | `1.0` | Tone Mapping 曝光调整（仅在 `tone_mapping` 工作流下使用） |
| `--global_l_max` | float | None | 全局 `l_max`（可选，仅 `tone_mapping` 工作流使用，用于控制全局压缩曲线，避免频闪） |
| `--hdr_transfer` | str | `hdr10` | HDR 传输方式（仅当 `output_mode=video` 且 `dynamic_range=hdr` 时生效）：`hdr10`（PQ，10-bit + 元数据）或 `hlg`（HLG 曲线）。SDR 或输出图片序列时忽略。 |

**分布式 / 调度相关（Distributed & Scheduling）**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--devices` | str | None | 使用的GPU设备：`all`（使用所有GPU）或逗号分隔的索引如 `0,1,2` 或 `0-2`（范围） |
| `--master_addr` | str | `localhost` | 分布式训练的主地址 |
| `--master_port` | int | `29500` | 分布式训练的主端口 |
| `--use_shared_memory` | bool | `True` | 使用共享内存（`/dev/shm`）加载模型（降低显存占用） |
| `--cleanup_mmap` | bool | `False` | 保存结果后清理内存映射画布文件 |
| `--tile_batch_size` | int | `0` | 同时处理的tile数量（0 = 根据GPU显存自动检测） |
| `--adaptive_tile_batch` | bool | `True` | 根据可用GPU显存启用自适应tile批处理大小 |
| `--max_frames` | int | None | 处理的最大帧数（用于测试） |

### 全参数指令模板

#### infer_video.py - 完整模板

```bash
python scripts/infer_video.py \
  --input ./inputs/video.mp4 \
  --output ./results/output.mp4 \
  --model_ver 1.1 \
  --mode tiny \
  --device cuda:0 \
  --scale 4 \
  --color_fix True \
  --tiled_vae True \
  --tiled_dit False \
  --tile_size 256 \
  --tile_overlap 24 \
  --unload_dit False \
  --sparse_ratio 2.0 \
  --kv_ratio 3.0 \
  --local_range 11 \
  --seed 0 \
  --precision bf16 \
  --attention_mode sparse_sage_attention \
  --multi_gpu \
  --adaptive_batch_size \
  --streaming \
  --segmented \
  --resume \
  --segment_overlap 2 \
  --max-segment-frames 100
```

#### infer_video_distributed.py - 完整模板

```bash
python scripts/infer_video_distributed.py \
  --input ./inputs/video.mp4 \
  --output ./results/output.mp4 \
  --output_mode video \
  --output_format mp4 \
  --output_bit_depth 8 \
  --input_fps 30.0 \
  --output_fps 30.0 \
  --dynamic_range sdr \
  --model_ver 1.1 \
  --mode tiny \
  --scale 4 \
  --precision bf16 \
  --attention_mode sparse_sage_attention \
  --segment_overlap 2 \
  --color_fix True \
  --tiled_vae True \
  --tiled_dit False \
  --tile_size 256 \
  --tile_overlap 24 \
  --unload_dit False \
  --sparse_ratio 2.0 \
  --kv_ratio 3.0 \
  --local_range 11 \
  --seed 0 \
  --master_addr localhost \
  --master_port 29500 \
  --use_shared_memory True \
  --cleanup_mmap False \
  --tile_batch_size 0 \
  --adaptive_tile_batch True \
  --max_frames None \
  --devices all
```

### 场景化指令模板

#### 场景1：单GPU快速处理（默认设置）

适用：快速测试、小视频、高显存GPU（24GB+）

```bash
python scripts/infer_video.py \
  --input ./inputs/video.mp4 \
  --output ./results/output.mp4 \
  --mode tiny \
  --scale 4
```

#### 场景2：单GPU低显存（8-16GB）

适用：显存有限，需要降低显存占用

```bash
python scripts/infer_video.py \
  --input ./inputs/video.mp4 \
  --output ./results/output.mp4 \
  --mode tiny \
  --scale 4 \
  --tiled_dit True \
  --tiled_vae True \
  --tile_size 256 \
  --tile_overlap 24 \
  --unload_dit True
```

#### 场景3：多GPU设置（2+ GPU）

适用：拥有多个GPU，追求最大速度

```bash
python scripts/infer_video.py \
  --input ./inputs/video.mp4 \
  --output ./results/output.mp4 \
  --mode tiny \
  --scale 4 \
  --multi_gpu \
  --adaptive_batch_size
```

或使用分布式版本（推荐）：

```bash
python scripts/infer_video_distributed.py \
  --input ./inputs/video.mp4 \
  --output ./results/output.mp4 \
  --mode tiny \
  --scale 4 \
  --devices all
```

#### 场景4：长视频处理

适用：超长视频，需要避免OOM错误

```bash
python scripts/infer_video.py \
  --input ./inputs/long_video.mp4 \
  --output ./results/output.mp4 \
  --mode tiny-long \
  --scale 4 \
  --streaming \
  --segmented \
  --segment_overlap 2
```

#### 场景5：高质量输出

适用：追求最高质量，速度次要

```bash
python scripts/infer_video.py \
  --input ./inputs/video.mp4 \
  --output ./results/output.mp4 \
  --mode full \
  --scale 4 \
  --precision fp32 \
  --color_fix True
```

#### 场景6：从中断恢复处理

适用：从崩溃或中断中恢复

```bash
python scripts/infer_video.py \
  --input ./inputs/video.mp4 \
  --output ./results/output.mp4 \
  --mode tiny \
  --scale 4 \
  --multi_gpu \
  --resume
```

#### 场景7：HDR视频处理（仅分布式版本）

适用：HDR输入视频，保留HDR信息

**方式一：直接输出 HDR 视频（一条命令）**  
内部会先写临时 DPX，再编码为 HDR 视频，最后删除临时文件。

```bash
python scripts/infer_video_distributed.py \
  --input ./inputs/hdr_video.mp4 \
  --output ./results/hdr_output.mp4 \
  --output_mode video \
  --dynamic_range hdr \
  --hdr_preprocess hlg \
  --hdr_transfer hdr10 \
  --mode tiny \
  --scale 2 \
  --tiled_dit True \
  --tiled_vae True \
  --tile_size 512 \
  --tile_overlap 384 \
  --color_fix True \
  --devices all
```

- `--hdr_transfer hdr10`：输出 HDR10（PQ）；若需 HLG 编码视频则用 `hlg`。

**方式二：先输出 DPX 序列，再手动编码**

步骤1：运行推理（输出 HLG 编码 DPX）

```bash
python scripts/infer_video_distributed.py \
  --input ./inputs/hdr_video.mp4 \
  --output_mode pictures \
  --output ./results/hdr_frames \
  --output_format dpx \
  --output_bit_depth 10 \
  --dynamic_range hdr \
  --hdr_preprocess hlg \
  --mode tiny \
  --scale 2 \
  --tiled_dit True \
  --tiled_vae True \
  --tile_size 512 \
  --tile_overlap 384 \
  --color_fix True \
  --devices all
```

步骤2：将 DPX 编码为 HDR 视频

```bash
python utils/io/hdr_video_encode.py \
  --input ./results/hdr_frames/ \
  --output ./results/hdr_output.mp4 \
  --fps 30.0 \
  --hdr_format hdr10 \
  --hlg_workflow \
  --crf 18 \
  --preset slow \
  --simple
```

**关键参数：**
- `--dynamic_range hdr`：使用 HDR 通道
- `--hdr_preprocess hlg`：HLG 工作流（推荐）；或 `tone_mapping` 使用可逆 Tone Mapping
- `--hdr_transfer`：在 `output_mode=video` 且 HDR 时：`hdr10`（PQ）或 `hlg`
- `--output_format dpx --output_bit_depth 10`：输出 DPX 序列时使用
- `--tile_overlap 384`：增加重叠区域，减少 HDR 边界伪影

#### 场景8：图像序列输入/输出

适用：处理图像序列，逐帧控制

```bash
python scripts/infer_video_distributed.py \
  --input ./inputs/frames/ \
  --output_mode pictures \
  --output ./results/output_frames \
  --output_format png \
  --output_bit_depth 8 \
  --input_fps 30.0 \
  --mode tiny \
  --scale 4 \
  --devices all
```

#### 场景9：限制帧数测试

适用：快速测试、调试

```bash
python scripts/infer_video_distributed.py \
  --input ./inputs/video.mp4 \
  --output ./results/test_output.mp4 \
  --mode tiny \
  --scale 4 \
  --max_frames 10 \
  --devices all
```

#### 场景10：最大性能（多GPU + 所有优化）

适用：生产环境，最大吞吐量

```bash
python scripts/infer_video_distributed.py \
  --input ./inputs/video.mp4 \
  --output ./results/output.mp4 \
  --mode tiny \
  --scale 4 \
  --devices all \
  --adaptive_tile_batch True \
  --use_shared_memory True \
  --tile_batch_size 0 \
  --precision bf16
```

#### 根据显存大小选择配置

- **显存 < 12GB**: `--mode tiny-long --scale 2 --tiled_dit True --tile_size 128 --unload_dit True`
- **显存 12-16GB**: `--mode tiny --scale 4 --tiled_dit True --tile_size 256`
- **显存 16-24GB**: `--mode tiny --scale 4 --tiled_dit True --tile_size 256 --adaptive_batch_size`
- **显存 > 24GB**: `--mode full --scale 4 --tiled_dit True --tile_size 512 --adaptive_batch_size`

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

### 流式处理模式

为长视频启用 `--streaming`：
- 以块为单位处理视频，降低显存占用
- 当画布显存超过阈值时自动启用
- 推荐用于超过1000帧的视频

### 分段处理模式

为单GPU场景启用 `--segmented`：
- 类似 `--multi_gpu`，但在单个worker内工作
- 将视频分成多个子段独立处理
- 可与 `--multi_gpu` 组合使用，实现两层分割

**预期性能：**
- **双GPU + 自适应批处理**: 相比单GPU提升3-5倍
- **显存使用**: 32GB GPU峰值使用20-25GB（未优化时约13GB）
- **流式处理模式**: 可处理任意长度的视频，显存占用恒定

## Multi-GPU + Segmented 模式详解

### 概述
当同时使用 `--multi_gpu` 和 `--segmented` 时，视频会经过两层分割：
1. **第一层（multi_gpu）**：按GPU数量分割成多个worker segments
2. **第二层（segmented）**：每个worker内部再分割成多个sub-segments

### 详细流程

#### 1. 第一层分割：multi_gpu模式

**分割逻辑：**
- 函数：`split_video_by_frames(frames, num_gpus, overlap=segment_overlap)`
- 计算方式：
  ```python
  segment_size = N // num_gpus  # N是总帧数
  for i in range(num_gpus):
      start_idx = max(0, i * segment_size - overlap if i > 0 else 0)
      end_idx = min(N, (i + 1) * segment_size + overlap if i < num_gpus - 1 else N)
  ```

**示例（612帧，2个GPU，overlap=2）：**
- **Segment 0 (Worker 0)**: frames 0-308 (共308帧)
- **Segment 1 (Worker 1)**: frames 304-612 (共308帧)
  - 注意：有4帧overlap (308-304=4)

#### 2. 目录结构和文件命名

**主目录名：**
- 函数：`get_video_based_dir_name(input_path, scale)`
- 格式：`{视频名}_{scale}x`
- 示例：`3D_cat_1080_30fps_4x`

**multi_gpu checkpoint：**
- **路径**：`/tmp/flashvsr_checkpoints/{video_dir_name}/`
- **文件**：`checkpoint.json`
- **内容**：记录每个worker的绝对帧范围和输出文件路径

**multi_gpu worker输出：**
- **路径**：`/tmp/flashvsr_multigpu/{video_dir_name}/`
- **文件命名**：`worker_{worker_id}_{uuid}.pt`
  - `worker_id`: 0, 1, 2, ... (对应segment索引)
  - `uuid`: 随机UUID，避免文件名冲突

#### 3. 第二层分割：segmented模式（在worker内部）

**如果worker启用了segmented模式：**
每个worker进程会：
1. 接收分配给它的frames（例如Worker 0接收frames 0-308）
2. 在worker内部，再次分割成多个sub-segments
3. 每个sub-segment独立处理并保存

**segmented目录结构：**
- **路径**：`/tmp/flashvsr_segments/{video_dir_name}/`
- **video_dir_name的确定**：
  - 如果在worker模式下：`worker_{worker_start_idx}_{worker_end_idx}_{scale}x`
    - 示例：`worker_0_308_4x` (Worker 0处理frames 0-308)
  - 如果不在worker模式：使用`get_video_based_dir_name(input_path, scale)`

**segmented文件命名：**
- **.pt文件**：`segment_{seg_idx:04d}.pt`
  - `seg_idx`: 0, 1, 2, ... (sub-segment索引，从0开始)
- **.json元数据文件**：`segment_{seg_idx:04d}.json`
  - 记录绝对帧范围（相对于原始视频）

#### 4. 完整示例流程

假设：视频612帧，2个GPU，启用segmented，每个sub-segment最大100帧

**步骤1：multi_gpu分割**
```
原始视频: 612帧
├── Worker 0: frames 0-308 (308帧)
└── Worker 1: frames 304-612 (308帧)
```

**步骤2：Worker 0内部segmented分割**
```
Worker 0接收: 308帧
├── Sub-segment 0: frames 0-100 (相对于worker: 0-100, 绝对: 0-100)
├── Sub-segment 1: frames 98-200 (相对于worker: 98-200, 绝对: 98-200)
├── Sub-segment 2: frames 198-300 (相对于worker: 198-300, 绝对: 198-300)
└── Sub-segment 3: frames 298-308 (相对于worker: 298-308, 绝对: 298-308)

保存位置: /tmp/flashvsr_segments/worker_0_308_4x/
├── segment_0000.pt + segment_0000.json
├── segment_0001.pt + segment_0001.json
├── segment_0002.pt + segment_0002.json
└── segment_0003.pt + segment_0003.json
```

**步骤3：Worker 0合并sub-segments**
```
Worker 0处理完所有sub-segments后：
1. 按seg_idx顺序加载所有sub-segments
2. 处理overlap（跳过重复帧）
3. 合并成最终输出
4. 保存到: /tmp/flashvsr_multigpu/3D_cat_1080_30fps_4x/worker_0_{uuid}.pt
```

**步骤4：主进程合并所有workers**
```
主进程：
1. 从checkpoint.json读取所有worker信息
2. 按start_idx排序
3. 加载每个worker的输出文件
4. 处理overlap（Worker 1跳过前4帧）
5. 合并成最终视频
```

#### 5. 关键点总结

1. **目录命名规则**：
   - multi_gpu: `/tmp/flashvsr_multigpu/{video_dir_name}/`
   - segmented (worker模式): `/tmp/flashvsr_segments/worker_{start}_{end}_{scale}x/`
   - segmented (非worker模式): `/tmp/flashvsr_segments/{video_dir_name}/`
   - checkpoint: `/tmp/flashvsr_checkpoints/{video_dir_name}/`

2. **文件命名规则**：
   - worker输出: `worker_{worker_id}_{uuid}.pt`
   - sub-segment: `segment_{seg_idx:04d}.pt` + `segment_{seg_idx:04d}.json`

3. **帧范围记录**：
   - checkpoint.json: 记录worker的绝对帧范围
   - segment_*.json: 记录sub-segment的绝对帧范围（相对于原始视频）

4. **Overlap处理**：
   - multi_gpu层：worker之间有overlap（例如4帧）
   - segmented层：sub-segment之间有overlap（例如2帧）
   - 合并时都会跳过overlap部分

5. **断点续传**：
   - multi_gpu: 检查`/tmp/flashvsr_checkpoints/{video_dir_name}/checkpoint.json`
   - segmented: 检查`/tmp/flashvsr_segments/{video_dir_name}/segment_*.pt`文件
   - 使用`--resume`参数启用断点续传，否则默认覆盖重新开始

### 正常跑与从 Checkpoint 恢复（参数与用法）

当前入口脚本为 **`scripts/infer_video.py`**（支持单卡与多卡分布式）。推理结果与合并逻辑与 **恢复工具** `tools/recover_distributed_inference.py` 共用同一套流式合并实现（`utils/io/streaming_merge_output`），因此正常跑完成后的输出与“先中断再恢复合并”的输出一致。

---

#### 一、正常跑（scripts/infer_video.py）

**必需参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `--input` | str | **必需**。输入路径：视频文件（如 video.mp4）或图像序列目录（如 /path/to/frames/） |

**常用可选参数（含默认值）：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--output` | str | 自动生成 | 输出路径：`output_mode=video` 时为视频文件；`output_mode=pictures` 时为目录 |
| `--output_mode` | str | `video` | 输出形式：`video`（单视频文件）或 `pictures`（图像序列） |
| `--output_format` | str | 见下 | 视频：`mp4`/`mov`/`mkv`（未设则尽量继承输入）；序列帧：`png`/`dpx` |
| `--output_bit_depth` | int | `8` | 位深：8 或 10（视频→yuv420p/yuv420p10le；序列帧→8bit PNG / 10bit DPX） |
| `--input_fps` | float | `30.0` | 输入为图像序列时的帧率；视频文件时自动探测，可忽略 |
| `--output_fps` | float | None | 输出视频帧率；未设则继承输入 FPS |
| `--dynamic_range` | str | `sdr` | 动态范围：`sdr` / `hdr` / `auto` |
| `--hdr_preprocess` | str | `hlg` | HDR 时预处理：`hlg`（推荐）或 `tone_mapping` |
| `--tone_mapping_method` | str | `logarithmic` | 仅 `hdr_preprocess=tone_mapping` 时：`reinhard` / `logarithmic` / `aces` |
| `--tone_mapping_exposure` | float | `1.0` | Tone Mapping 曝光调整 |
| `--global_l_max` | float | None | 可选；Tone Mapping 工作流下全局 l_max，减少闪烁 |
| `--hdr_transfer` | str | `hdr10` | HDR 视频输出：`hdr10` 或 `hlg` |
| `--model_ver` | str | `1.1` | 模型版本：`1.0` / `1.1` |
| `--mode` | str | `tiny` | 模型模式：`tiny` / `full` / `tiny-long` |
| `--scale` | int | `4` | 放大倍数：2 / 3 / 4 |
| `--precision` | str | `bf16` | 精度：`fp32` / `fp16` / `bf16` |
| `--tile_size` | int | `256` | Tile 边长（像素） |
| `--tile_overlap` | int | `24` | Tile 重叠像素 |
| `--segment_overlap` | int | `2` | 多卡时分段重叠帧数 |
| `--devices` | str | None | GPU：`all` 或 `0,1,2`、`0-2` 等 |
| `--resume` | flag | 默认 True | 从 checkpoint 续跑（存在则接着跑） |
| `--no-resume` / `--force` | flag | - | 清空 checkpoint 从头跑 |
| `--use_shared_memory` | bool | True | 使用 /dev/shm 加载模型 |
| `--cleanup_mmap` | bool | False | 保存结果后是否删除 mmap 画布文件 |
| `--tile_batch_size` | int | `0` | 每批 tile 数；0 表示自动 |
| `--adaptive_tile_batch` | bool | True | 是否根据显存自适应 tile 批大小 |
| `--max_frames` | int | None | 仅处理前 N 帧（测试用） |

**Checkpoint 目录：** 由 `--output` 与 `--output_mode` 自动推断，启动时打印，例如：`[Main] Checkpoint directory: /app/tmp/checkpoints/输出名`。其中“输出名”：视频模式为输出文件主名，序列帧模式为输出目录名。

**正常跑示例：**

```bash
# 多卡、输出视频
python scripts/infer_video.py \
  --input video.mp4 \
  --output output_4x.mp4 \
  --mode tiny \
  --scale 4 \
  --devices all

# 输出序列帧（PNG）
python scripts/infer_video.py \
  --input video.mp4 \
  --output_mode pictures \
  --output /path/to/out_frames \
  --output_format png \
  --devices all

# 输出 10-bit DPX 序列（HDR）
python scripts/infer_video.py \
  --input hdr_input/ \
  --output_mode pictures \
  --output /path/to/dpx_out \
  --output_format dpx \
  --output_bit_depth 10 \
  --dynamic_range hdr \
  --hdr_preprocess hlg \
  --devices all
```

---

#### 二、从 Checkpoint 恢复（tools/recover_distributed_inference.py）

用于：推理中断或部分 rank 失败后，从已有 checkpoint 目录**仅做流式合并与输出**（与正常跑使用同一套合并逻辑）。

**必需参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `--checkpoint_dir` | str | **必需**。checkpoint 目录（与正常跑日志中的 `[Main] Checkpoint directory: ...` 一致，如 `/app/tmp/checkpoints/输出名`） |

**子命令与合并相关参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--status` | flag | - | 检查各 rank 状态（完成/失败/进度等） |
| `--merge_partial` | flag | - | 流式合并已有 rank 结果并写出（部分 rank 失败也可合并） |
| `--output` | str | - | **使用 `--merge_partial` 时必需**。输出路径：视频文件或序列帧目录 |
| `--fps` | float | `30.0` | 输出视频帧率（或序列帧名义 FPS） |
| `--world_size` | int | None | rank 总数；不设则从 checkpoint 目录自动推断 |
| `--total_frames` | int | None | 期望总帧数（用于填充不足或校验；不设则按实际合并帧数） |
| `--output_mode` | str | `video` | 与正常跑一致：`video` 或 `pictures` |
| `--output_format` | str | `png` | 仅 `output_mode=pictures` 时：`png` 或 `dpx10` |
| `--output_frame_prefix` | str | None | 序列帧文件名前缀（如 H001_11261139_C001） |
| `--output_frame_digits` | int | `6` | 序列帧索引位数 |
| `--workers` | int | `0` | 写序列帧时的线程数；0 表示自动 |
| `--recover_rank` | int | - | 仅打印恢复某一 rank 的说明，不执行合并 |

**恢复示例：**

```bash
# 查看 checkpoint 下各 rank 状态
python tools/recover_distributed_inference.py \
  --checkpoint_dir /app/tmp/checkpoints/输出名 \
  --status

# 合并为视频（与正常跑同逻辑，流式合并）
python tools/recover_distributed_inference.py \
  --checkpoint_dir /app/tmp/checkpoints/输出名 \
  --merge_partial \
  --output output_partial.mp4 \
  --fps 30.0

# 合并为序列帧（PNG）
python tools/recover_distributed_inference.py \
  --checkpoint_dir /app/tmp/checkpoints/输出名 \
  --merge_partial \
  --output /path/to/out_frames \
  --output_mode pictures \
  --output_format png \
  --fps 30.0

# 合并为 10-bit DPX 序列
python tools/recover_distributed_inference.py \
  --checkpoint_dir /app/tmp/checkpoints/输出名 \
  --merge_partial \
  --output /path/to/dpx_out \
  --output_mode pictures \
  --output_format dpx10 \
  --fps 30.0

# 合并为序列帧并指定文件名前缀、索引位数、写帧线程数（如 H001_11261139_C001.00000000.png）
python tools/recover_distributed_inference.py \
  --checkpoint_dir /app/tmp/checkpoints/输出名 \
  --merge_partial \
  --output /path/to/out_frames \
  --output_mode pictures \
  --output_format png \
  --output_frame_prefix "H001_11261139_C001" \
  --output_frame_digits 8 \
  --workers 32 \
  --fps 30.0
```

**说明：** 正常跑完成后由 Rank 0 调用的合并与恢复工具的 `--merge_partial` 均走 `utils/io/streaming_merge_output.streaming_merge_from_checkpoint`，因此输出格式与画质一致；恢复时只需保证 `--output_mode`、`--output_format`、`--fps` 等与正常跑意图一致即可。

---

### 恢复工具（旧版说明，保留参考）

如果处理过程中断，可以使用恢复工具手动合并已完成的文件：

**从 checkpoint 合并为视频：**
```bash
python tools/recover_distributed_inference.py \
  --checkpoint_dir /app/tmp/checkpoints/{输出名} \
  --merge_partial \
  --output /path/to/recovered.mp4 \
  --fps 30
```

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
- 使用 `--scale 2` 或 `3` 或`4`
- 减小 `--kv_ratio`（如1.0）
- 启用 `--streaming` 或 `--segmented` 模式

### 视频读取问题

**问题：`torchvision read_video failed: PyAV is not installed`**

**解决方案：**
1. 安装 PyAV（推荐，以获得更好的性能和兼容性）：
   ```bash
   pip install av
   ```
   或者重新安装所有依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 如果未安装 PyAV，代码会自动回退到 OpenCV 或 FFmpeg，功能不受影响，但性能可能略低
3. 确保已安装 FFmpeg（用于视频编解码）：
   ```bash
   # Ubuntu/Debian
   sudo apt-get install -y ffmpeg libavcodec-dev
   
   # 或在 Docker 容器中
   apt-get update && apt-get install -y ffmpeg libavcodec-dev
   ```

### 多GPU使用注意事项

- 需要至少2个GPU才能启用 `--multi_gpu`
- 每个GPU都会加载完整模型，显存需求不变
- 适用于长视频，短视频可能不会显著加速
- 确保所有GPU都有足够显存

## 致谢
- [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) @OpenImagingLab  
- [Sparse_SageAttention](https://github.com/jt-zhang/Sparse_SageAttention_API) @jt-zhang
