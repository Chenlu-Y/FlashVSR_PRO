# FlashVSR_Ultra_Fast
Running FlashVSR on lower VRAM without any artifacts.   
**[[📃中文版本](./README_zh.md)]**

## Changelog
#### 2025-10-31
- **New:** Standalone `infer_video.py` script for video processing
- **New:** Multi-GPU parallel processing (`--multi_gpu`) - automatically splits video by frames across GPUs
- **New:** Adaptive tile batch size (`--adaptive_batch_size`) - dynamically adjusts tile concurrency based on GPU memory
- **New:** Streaming mode (`--streaming`) - processes long videos in chunks to reduce memory usage
- **New:** Segmented mode (`--segmented`) - processes video in sub-segments for single GPU scenarios
- **New:** Resume from checkpoint (`--resume`) - automatically detects and merges completed frames from previous runs
- **New:** GPU memory monitoring and optimization for 24-32GB GPUs
- **New:** Total elapsed time tracking for performance monitoring
- **Improved:** Video reading with OpenCV fallback for better codec compatibility
- **Fixed:** Empty frame handling and negative dimension errors

#### 2025-10-24
- Added long video pipeline that significantly reduces VRAM usage when upscaling long videos.

#### 2025-10-22
- Replaced `Block-Sparse-Attention` with `Sparse_Sage`, removing the need to compile any custom kernels.  
- Added support for running on RTX 50 series GPUs.

#### 2025-10-21
- Initial this project, introducing features such as `tile_dit` to significantly reducing VRAM usage.


## Installation

📢: For Turing or older GPU, please install `triton<3.3.0`:  

```bash
# Windows
python -m pip install -U triton-windows<3.3.0
# Linux
python -m pip install -U triton<3.3.0
```

### Models

- Download the entire `FlashVSR` folder with all the files inside it from [here](https://huggingface.co/JunhaoZhuang/FlashVSR) and put it in your model directory (default: `/app/models/v1.1/`)

```
├── FlashVSR
|     ├── LQ_proj_in.ckpt
|     ├── TCDecoder.ckpt
|     ├── diffusion_pytorch_model_streaming_dmd.safetensors
|     ├── Wan2.1_VAE.pth
```

## Usage

### Scripts Overview

Two inference scripts are available:

1. **`scripts/infer_video.py`**: Single-process inference with multi-GPU support via frame splitting
2. **`scripts/infer_video_distributed.py`**: True distributed inference with model parallelism (recommended for multi-GPU setups)

### Parameter Reference

#### Basic Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--input` | str | **required** | Input video path or image sequence directory |
| `--output` | str | None | Output video path (auto-generated if not specified) |
| `--model_ver` | str | `1.1` | Model version: `1.0` or `1.1` |
| `--mode` | str | `tiny` | Model mode: `tiny` (faster), `full` (higher quality), `tiny-long` (for long videos) |
| `--device` | str | `cuda:0` | Device to use (ignored if `--multi_gpu` is used) |
| `--scale` | int | `2` (infer_video.py)<br>`4` (infer_video_distributed.py) | Upscale factor: `2`, `3`, or `4` |

#### Quality & Processing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--color_fix` | bool | `True` | Use wavelet transform to correct output video color |
| `--tiled_vae` | bool | `True` | Use tiled VAE for lower VRAM consumption (slower) |
| `--tiled_dit` | bool | `False` | Use tiled DiT to significantly reduce VRAM usage (slower) |
| `--tile_size` | int | `256` | Tile size for tiled processing |
| `--tile_overlap` | int | `24` | Tile overlap in pixels |
| `--tile_shift` | bool | `True` | Use tile shift (two-pass with offset grid); average two passes to reduce grid seams (only when `--tiled_dit` is on) |
| `--unload_dit` | bool | `False` | Unload DiT before decoding to reduce VRAM peak (slower) |
| `--precision` | str | `bf16` | Precision: `fp32`, `fp16`, or `bf16` |
| `--attention_mode` | str | `sparse_sage_attention` | Attention mode: `sparse_sage_attention` or `block_sparse_attention` |
| `--sparse_ratio` | float | `2.0` | Sparse attention ratio |
| `--kv_ratio` | float | `3.0` | KV cache ratio |
| `--local_range` | int | `11` | Local attention range |
| `--seed` | int | `0` | Random seed for reproducibility |

#### Performance Optimization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--multi_gpu` | flag | False | Enable multi-GPU parallel processing (splits video by frames across GPUs) |
| `--adaptive_batch_size` | flag | False | Enable adaptive tile batch size (dynamically adjusts based on GPU memory) |
| `--streaming` | flag | False | Enable streaming mode for long videos (processes in chunks to reduce memory) |
| `--segmented` | flag | False | Enable segmented processing mode (processes video in sub-segments, similar to `--multi_gpu` but for single worker) |
| `--segment_overlap` | int | `2` | Number of overlap frames between segments/chunks (range: 1-10, recommended: 1-5) |
| `--max-segment-frames` | int | None | Maximum frames per segment in segmented mode (default: auto-calculate based on memory) |

#### Recovery & Checkpoint Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--resume` | flag | False | Resume from checkpoint: automatically detects and merges completed frames from previous runs (works with `--multi_gpu` and `--segmented`) |
| `--clean-checkpoint` | flag | False | Clean checkpoint directory before starting (disable resume) |

#### Distributed Inference Parameters (infer_video_distributed.py only)

**I/O & Format**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--output_mode` | str | `video` | Output mode: `video` (single video file) or `pictures` (image sequence) |
| `--output_format` | str | `auto` | When `output_mode=video`: `mp4` / `mov` / `mkv`. If not set (auto), tries to inherit the input video container when the input is a video file, otherwise falls back to `mp4`. When `output_mode=pictures`: `png` / `dpx`, default `png`. |
| `--output_bit_depth` | int | `8` | Output bit depth: `8` or `10`. For video this maps to `yuv420p` / `yuv420p10le`; for pictures to 8-bit PNG or 10-bit DPX |
| `--input_fps` | float | `30.0` | Frame rate of input image sequences. Ignored when input is a video file (FPS is probed automatically) |
| `--output_fps` | float | `None` | Output video FPS, only used when `output_mode=video`. If not set, it automatically inherits the probed input FPS |

**Dynamic Range & HDR Pipeline**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dynamic_range` | str | `sdr` | Dynamic range: `sdr` (Standard Dynamic Range), `hdr` (High Dynamic Range), or `auto` (auto-detect) |
| `--hdr_preprocess` | str | `hlg` | HDR preprocessing (only used when `--dynamic_range=hdr`): `hlg` (PQ→HLG workflow, recommended) or `tone_mapping` (reversible tone-mapping workflow) |
| `--tone_mapping_method` | str | `logarithmic` | Tone mapping method (only when `--hdr_preprocess=tone_mapping`): `reinhard`, `logarithmic`, or `aces` |
| `--tone_mapping_exposure` | float | `1.0` | Tone mapping exposure adjustment (used in `tone_mapping` workflow) |
| `--global_l_max` | float | None | Global `l_max` value (optional, `tone_mapping` workflow only) to control the compression curve and avoid flicker |
| `--hdr_transfer` | str | `hdr10` | HDR transfer when `output_mode=video` and `dynamic_range=hdr`: `hdr10` (PQ, 10-bit + metadata) or `hlg` (HLG curve). Ignored for SDR or `output_mode=pictures`. |

**Distributed & Scheduling**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--devices` | str | None | GPU devices to use: `all` (use all GPUs) or comma-separated indices like `0,1,2` or `0-2` (range) |
| `--master_addr` | str | `localhost` | Master address for distributed training |
| `--master_port` | int | `29500` | Master port for distributed training |
| `--use_shared_memory` | bool | `True` | Use shared memory (`/dev/shm`) for model loading (reduces memory usage) |
| `--cleanup_mmap` | bool | `False` | Clean up memory-mapped canvas files after saving results |
| `--tile_batch_size` | int | `0` | Number of tiles to process simultaneously (0 = auto-detect based on GPU memory) |
| `--adaptive_tile_batch` | bool | `True` | Enable adaptive tile batch size based on available GPU memory |
| `--max_frames` | int | None | Maximum number of frames to process (for testing) |

### Full Parameter Command Templates

#### infer_video.py - Full Template

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

#### infer_video_distributed.py - Full Template

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

### Scenario-Based Command Templates

#### Scenario 1: Single GPU, Fast Processing (Default Settings)

Best for: Quick tests, small videos, high VRAM GPUs (24GB+)

```bash
python scripts/infer_video.py \
  --input ./inputs/video.mp4 \
  --output ./results/output.mp4 \
  --mode tiny \
  --scale 4
```

#### Scenario 2: Single GPU, Low VRAM (8-16GB)

Best for: Limited VRAM, need to reduce memory usage

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

#### Scenario 3: Multi-GPU Setup (2+ GPUs)

Best for: Multiple GPUs available, want maximum speed

```bash
python scripts/infer_video.py \
  --input ./inputs/video.mp4 \
  --output ./results/output.mp4 \
  --mode tiny \
  --scale 4 \
  --multi_gpu \
  --adaptive_batch_size
```

Or use the distributed version (recommended):

```bash
python scripts/infer_video_distributed.py \
  --input ./inputs/video.mp4 \
  --output ./results/output.mp4 \
  --mode tiny \
  --scale 4 \
  --devices all
```

#### Scenario 4: Long Video Processing

Best for: Very long videos, need to avoid OOM errors

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

#### Scenario 5: High Quality Output

Best for: Maximum quality, speed is secondary

```bash
python scripts/infer_video.py \
  --input ./inputs/video.mp4 \
  --output ./results/output.mp4 \
  --mode full \
  --scale 4 \
  --precision fp32 \
  --color_fix True
```

#### Scenario 6: Resume from Interrupted Processing

Best for: Recovering from crashes or interruptions

```bash
python scripts/infer_video.py \
  --input ./inputs/video.mp4 \
  --output ./results/output.mp4 \
  --mode tiny \
  --scale 4 \
  --multi_gpu \
  --resume
```

#### Scenario 7: HDR Video Processing (Distributed Only)

Best for: HDR input videos, preserving HDR information

**Option A: Direct HDR video output (one command)**  
Internally writes temporary DPX, encodes to HDR video, then removes temp files.

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

- `--hdr_transfer hdr10`: Output HDR10 (PQ); use `hlg` for HLG-encoded video.

**Option B: Output DPX sequence, then encode manually**

Step 1: Run inference (output HLG-encoded DPX)

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

Step 2: Encode DPX to HDR video

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

**Key parameters:**
- `--dynamic_range hdr`: Use HDR pipeline
- `--hdr_preprocess hlg`: HLG workflow (recommended); or `tone_mapping` for reversible tone mapping
- `--hdr_transfer`: When `output_mode=video` and HDR: `hdr10` (PQ) or `hlg`
- `--output_format dpx --output_bit_depth 10`: For DPX sequence output
- `--tile_overlap 384`: Increased overlap to reduce boundary artifacts in HDR

#### Scenario 8: Image Sequence Input/Output

Best for: Working with image sequences, frame-by-frame control

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

#### Scenario 9: Testing with Limited Frames

Best for: Quick testing, debugging

```bash
python scripts/infer_video_distributed.py \
  --input ./inputs/video.mp4 \
  --output ./results/test_output.mp4 \
  --mode tiny \
  --scale 4 \
  --max_frames 10 \
  --devices all
```

#### Scenario 10: Maximum Performance (Multi-GPU + All Optimizations)

Best for: Production environments, maximum throughput

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

## Performance Optimization

### Multi-GPU Processing

For systems with 2+ GPUs, enable `--multi_gpu` to achieve near-linear speedup:
- Automatically splits video frames across available GPUs
- Each GPU processes a video segment independently
- Results are merged seamlessly with overlap handling

### Adaptive Batch Size

Enable `--adaptive_batch_size` to maximize GPU utilization:
- Dynamically adjusts tile batch size based on available VRAM
- For 32GB GPUs, can process 6-16 tiles concurrently
- Automatically rebalances during processing if memory changes

### Streaming Mode

Enable `--streaming` for long videos:
- Processes video in chunks to reduce memory usage
- Automatically enabled when canvas memory exceeds threshold
- Recommended for videos longer than 1000 frames

### Segmented Mode

Enable `--segmented` for single GPU scenarios:
- Similar to `--multi_gpu` but works within a single worker
- Processes video in sub-segments independently
- Can be combined with `--multi_gpu` for two-layer segmentation

**Expected Performance:**
- **2 GPUs + Adaptive Batch**: 3-5x speedup compared to single GPU
- **VRAM Usage**: 20-25GB peak on 32GB GPUs (vs 13GB without optimizations)
- **Streaming Mode**: Can handle videos of any length with constant memory usage

## Multi-GPU + Segmented Mode Details

### Overview
When using both `--multi_gpu` and `--segmented`, the video goes through two layers of segmentation:
1. **First layer (multi_gpu)**: Split into multiple worker segments by GPU count
2. **Second layer (segmented)**: Each worker internally splits into multiple sub-segments

### Detailed Process

#### 1. First Layer: multi_gpu Mode

**Segmentation Logic:**
- Function: `split_video_by_frames(frames, num_gpus, overlap=segment_overlap)`
- Calculation:
  ```python
  segment_size = N // num_gpus  # N is total frame count
  for i in range(num_gpus):
      start_idx = max(0, i * segment_size - overlap if i > 0 else 0)
      end_idx = min(N, (i + 1) * segment_size + overlap if i < num_gpus - 1 else N)
  ```

**Example (612 frames, 2 GPUs, overlap=2):**
- **Segment 0 (Worker 0)**: frames 0-308 (308 frames)
- **Segment 1 (Worker 1)**: frames 304-612 (308 frames)
  - Note: 4 frames overlap (308-304=4)

#### 2. Directory Structure and File Naming

**Main Directory Name:**
- Function: `get_video_based_dir_name(input_path, scale)`
- Format: `{video_name}_{scale}x`
- Example: `3D_cat_1080_30fps_4x`

**multi_gpu checkpoint:**
- **Path**: `/tmp/flashvsr_checkpoints/{video_dir_name}/`
- **File**: `checkpoint.json`
- **Content**: Records absolute frame ranges and output file paths for each worker

**multi_gpu worker output:**
- **Path**: `/tmp/flashvsr_multigpu/{video_dir_name}/`
- **File naming**: `worker_{worker_id}_{uuid}.pt`
  - `worker_id`: 0, 1, 2, ... (corresponds to segment index)
  - `uuid`: Random UUID to avoid filename conflicts

#### 3. Second Layer: segmented Mode (within worker)

**If worker enables segmented mode:**
Each worker process will:
1. Receive frames assigned to it (e.g., Worker 0 receives frames 0-308)
2. Split internally into multiple sub-segments
3. Process and save each sub-segment independently

**segmented directory structure:**
- **Path**: `/tmp/flashvsr_segments/{video_dir_name}/`
- **video_dir_name determination**:
  - In worker mode: `worker_{worker_start_idx}_{worker_end_idx}_{scale}x`
    - Example: `worker_0_308_4x` (Worker 0 processes frames 0-308)
  - Non-worker mode: Use `get_video_based_dir_name(input_path, scale)`

**segmented file naming:**
- **.pt file**: `segment_{seg_idx:04d}.pt`
  - `seg_idx`: 0, 1, 2, ... (sub-segment index, starting from 0)
- **.json metadata file**: `segment_{seg_idx:04d}.json`
  - Records absolute frame ranges (relative to original video)

#### 4. Complete Example Flow

Assume: 612 frames video, 2 GPUs, segmented enabled, max 100 frames per sub-segment

**Step 1: multi_gpu segmentation**
```
Original video: 612 frames
├── Worker 0: frames 0-308 (308 frames)
└── Worker 1: frames 304-612 (308 frames)
```

**Step 2: Worker 0 internal segmented segmentation**
```
Worker 0 receives: 308 frames
├── Sub-segment 0: frames 0-100 (relative to worker: 0-100, absolute: 0-100)
├── Sub-segment 1: frames 98-200 (relative to worker: 98-200, absolute: 98-200)
├── Sub-segment 2: frames 198-300 (relative to worker: 198-300, absolute: 198-300)
└── Sub-segment 3: frames 298-308 (relative to worker: 298-308, absolute: 298-308)

Save location: /tmp/flashvsr_segments/worker_0_308_4x/
├── segment_0000.pt + segment_0000.json
├── segment_0001.pt + segment_0001.json
├── segment_0002.pt + segment_0002.json
└── segment_0003.pt + segment_0003.json
```

**Step 3: Worker 0 merge sub-segments**
```
After Worker 0 processes all sub-segments:
1. Load all sub-segments in seg_idx order
2. Handle overlap (skip duplicate frames)
3. Merge into final output
4. Save to: /tmp/flashvsr_multigpu/3D_cat_1080_30fps_4x/worker_0_{uuid}.pt
```

**Step 4: Main process merge all workers**
```
Main process:
1. Read all worker info from checkpoint.json
2. Sort by start_idx
3. Load each worker's output file
4. Handle overlap (Worker 1 skips first 4 frames)
5. Merge into final video
```

#### 5. Key Points Summary

1. **Directory naming rules**:
   - multi_gpu: `/tmp/flashvsr_multigpu/{video_dir_name}/`
   - segmented (worker mode): `/tmp/flashvsr_segments/worker_{start}_{end}_{scale}x/`
   - segmented (non-worker mode): `/tmp/flashvsr_segments/{video_dir_name}/`
   - checkpoint: `/tmp/flashvsr_checkpoints/{video_dir_name}/`

2. **File naming rules**:
   - worker output: `worker_{worker_id}_{uuid}.pt`
   - sub-segment: `segment_{seg_idx:04d}.pt` + `segment_{seg_idx:04d}.json`

3. **Frame range recording**:
   - checkpoint.json: Records absolute frame ranges for workers
   - segment_*.json: Records absolute frame ranges for sub-segments (relative to original video)

4. **Overlap handling**:
   - multi_gpu layer: Overlap between workers (e.g., 4 frames)
   - segmented layer: Overlap between sub-segments (e.g., 2 frames)
   - Overlap is skipped during merging

5. **Resume from checkpoint**:
   - multi_gpu: Check `/tmp/flashvsr_checkpoints/{video_dir_name}/checkpoint.json`
   - segmented: Check `/tmp/flashvsr_segments/{video_dir_name}/segment_*.pt` files
   - Use `--resume` parameter to enable resume, otherwise defaults to overwrite and restart

### Recovery Tools

If processing is interrupted, you can use recovery tools to manually merge completed files:

**Recover from worker files:**
```bash
python tools/recover_distributed_inference.py /tmp/flashvsr_multigpu/{video_dir_name} /app/output/recovered.mp4 --fps 30
```

**Find unmerged files:**
```bash
python tools/find_unmerged.py
```

## Acknowledgments
- [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) @OpenImagingLab  
- [Sparse_SageAttention](https://github.com/jt-zhang/Sparse_SageAttention_API) @jt-zhang
