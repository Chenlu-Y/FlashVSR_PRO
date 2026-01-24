#!/usr/bin/env python3
"""
FlashVSR 分布式推理脚本
使用 torch.distributed 实现真正的模型并行和数据并行

关键优化：
1. 使用共享内存（/dev/shm）存储模型权重，避免每个进程重复加载
2. 使用 torch.distributed 进行进程间通信和同步
3. 支持8卡及以上大规模并行推理
4. 错开模型加载时间，减少内存峰值
5. 保留与原版 infer_video.py 相同的接口和功能

针对长视频和大分辨率视频的优化（v2）：
6. 内存映射 Canvas：使用 numpy.memmap 存储 canvas，避免内存不足（支持16K-32K视频）
7. Tile 级流式输出：处理完一个 tile 就写入磁盘，立即释放内存
8. 断点续跑：支持从中断点恢复，已处理的 tiles 会自动跳过
9. 修复工具：提供 recover_distributed_inference.py 用于检查和恢复任务

使用方法：
    # 基本用法（自动使用所有可见GPU）
    python infer_video_distributed.py --input video.mp4 --output output.mp4
    
    # 指定使用的GPU（使用 --devices 参数）
    python infer_video_distributed.py --input video.mp4 --output output.mp4 --devices 0,1,2,3
    
    # 16K-32K 大分辨率视频（推荐使用更大的 tile_size）
    python infer_video_distributed.py \
        --input 16k_video.mp4 \
        --output 16k_2x.mp4 \
        --mode tiny \
        --scale 2 \
        --tile_size 512 \
        --tile_overlap 48 \
        --devices all
    
    # 完整参数示例（包含新功能）
    python infer_video_distributed.py \
        --input video.mp4 \
        --output output_4x.mp4 \
        --model_ver 1.1 \
        --mode tiny \
        --scale 4 \
        --precision bf16 \
        --tile_size 256 \
        --tile_overlap 24 \
        --segment_overlap 2 \
        --use_shared_memory true \
        --devices all \
        --cleanup_mmap false  # 保留内存映射文件用于恢复（默认）
    
    # 断点续跑：如果任务中断，直接重新运行相同命令即可自动恢复
    # 已处理的 tiles 会自动跳过，从断点继续
    
    # 检查状态和恢复（使用恢复工具）
    python recover_distributed_inference.py \
        --checkpoint_dir /app/output/flashvsr_distributed/{video_dir_name} \
        --status
    
    # 合并部分结果（即使有些 rank 失败）
    python recover_distributed_inference.py \
        --checkpoint_dir /app/output/flashvsr_distributed/{video_dir_name} \
        --merge_partial \
        --output output_partial.mp4 \
        --fps 30.0

与原版 infer_video.py 的区别：
1. 使用 torch.distributed 而非 multiprocessing.Process
2. 模型加载使用共享内存优化（如果可用）
3. 更好的进程间同步和错误处理
4. 专为8卡及以上大规模并行设计

注意事项：
- 需要至少2个GPU才能运行
- 确保 /dev/shm 有足够空间（建议至少50GB）用于模型共享
- 如果共享内存不可用，会自动回退到错开加载策略

内存映射和断点续跑：
- Canvas 使用内存映射文件存储，避免内存不足（支持16K-32K视频）
- 每个 tile 处理完立即写入磁盘，即使中断也不会丢失已处理的数据
- 支持断点续跑：重新运行相同命令会自动从断点恢复
- 内存映射文件保存在 checkpoint_dir 中，可用于恢复和调试
- 使用 --cleanup_mmap true 可以在保存结果后删除内存映射文件以节省空间

恢复工具：
- 使用 recover_distributed_inference.py 检查任务状态
- 可以合并部分结果（即使有些 rank 失败）
- 支持手动恢复失败的 rank
"""

import os
import sys
import math
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# 设置 LD_LIBRARY_PATH 以支持 block_sparse_attention
_torch_lib_path = "/usr/local/lib/python3.10/dist-packages/torch/lib"
if os.path.exists(_torch_lib_path):
    _current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if _torch_lib_path not in _current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{_torch_lib_path}:{_current_ld_path}"

import torch.nn.functional as F
import torchvision
import cv2
from tqdm import tqdm
from einops import rearrange
from typing import List, Tuple, Optional
import uuid
import time
import hashlib
import json
import glob
import gc
import shutil
import tempfile
import numpy as np

# 将项目根目录添加到 sys.path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ====== FlashVSR modules ======
from src.models.model_manager import ModelManager
from src.models.TCDecoder import build_tcdecoder
from src.models.utils import Buffer_LQ4x_Proj, clean_vram
from src.models import wan_video_dit
from src.pipelines.flashvsr_full import FlashVSRFullPipeline
from src.pipelines.flashvsr_tiny import FlashVSRTinyPipeline
from src.pipelines.flashvsr_tiny_long import FlashVSRTinyLongPipeline

# ==============================================================
#                      Utility Functions (从原版复制)
# ==============================================================

def log(message: str, message_type: str = 'normal', rank: int = 0):
    """Colored logging for console output (with flush for real-time output)."""
    if dist.is_initialized() and rank != 0:
        return  # 只在 rank 0 打印日志
    
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    elif message_type == 'info':
        message = '\033[1;33m' + message + '\033[m'
    print(message, flush=True)  # 确保实时输出

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def tensor2video(frames: torch.Tensor):
    """Convert tensor (B,C,F,H,W) to normalized video tensor (F,H,W,C)"""
    video_squeezed = frames.squeeze(0)
    video_permuted = rearrange(video_squeezed, "C F H W -> F H W C")
    video_final = (video_permuted.float() + 1.0) / 2.0
    return video_final

def largest_8n1_leq(n):
    """Return largest (8n+1) less than or equal to n."""
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def compute_scaled_and_target_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128):
    """Compute scaled and target dimensions aligned to multiple."""
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid input size")
    sW, sH = w0 * scale, h0 * scale
    tW = max(multiple, (sW // multiple) * multiple)
    tH = max(multiple, (sH // multiple) * multiple)
    return sW, sH, tW, tH

def tensor_upscale_then_center_crop(frame_tensor: torch.Tensor, scale: int, tW: int, tH: int):
    """Upscale and center-crop a tensor frame."""
    h0, w0, c = frame_tensor.shape
    tensor_bchw = frame_tensor.permute(2, 0, 1).unsqueeze(0)
    sW, sH = w0 * scale, h0 * scale
    upscaled = F.interpolate(tensor_bchw, size=(sH, sW), mode='bicubic', align_corners=False)
    l = max(0, (sW - tW) // 2)
    t = max(0, (sH - tH) // 2)
    cropped = upscaled[:, :, t:t + tH, l:l + tW]
    return cropped.squeeze(0)

def prepare_input_tensor(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16):
    """Prepare video tensor by upscaling and padding."""
    N0, h0, w0, _ = image_tensor.shape
    multiple = 128
    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=multiple)
    num_frames_with_padding = N0 + 4
    F_ = largest_8n1_leq(num_frames_with_padding)
    if F_ == 0:
        raise RuntimeError(f"Not enough frames after padding: {num_frames_with_padding}")

    frames = []
    for i in range(F_):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_chw = tensor_upscale_then_center_crop(frame_slice, scale, tW, tH)
        tensor_out = tensor_chw * 2.0 - 1.0
        tensor_out = tensor_out.to('cpu').to(dtype)
        frames.append(tensor_out)

    vid_stacked = torch.stack(frames, 0)
    vid_final = vid_stacked.permute(1, 0, 2, 3).unsqueeze(0)
    
    del vid_stacked
    clean_vram()
    
    return vid_final, tH, tW, F_

def get_gpu_memory_info(device: str) -> Tuple[float, float]:
    """Get GPU memory info (used, total) in GB."""
    # 处理 torch.device 对象
    if isinstance(device, torch.device):
        device = str(device)
    if not device.startswith("cuda:"):
        return 0.0, 0.0
    try:
        idx = int(device.split(":")[1])
        torch.cuda.set_device(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(idx) / (1024**3)
        used = reserved  # 使用reserved memory作为使用量
        return used, total
    except Exception as e:
        log(f"Error getting GPU memory info: {e}", "warning")
        return 0.0, 0.0

def get_available_memory_gb(device: str) -> float:
    """Get available GPU memory in GB."""
    used, total = get_gpu_memory_info(device)
    return total - used

def estimate_tile_memory(tile_size: int, num_frames: int, scale: int, dtype_size: int = 2) -> float:
    """Estimate memory needed for processing one tile in GB.
    
    Args:
        tile_size: Tile size in pixels
        num_frames: Number of frames
        scale: Upscale factor
        dtype_size: Size of dtype in bytes (2 for fp16/bf16, 4 for fp32)
    """
    # 更准确的显存估算
    # 输入：tile_size^2 * num_frames * 3 * dtype_size
    input_size = tile_size * tile_size * num_frames * 3 * dtype_size / (1024**3)
    
    # 输出：tile_size^2 * scale^2 * num_frames * 3 * dtype_size
    output_size = (tile_size * scale) * (tile_size * scale) * num_frames * 3 * dtype_size / (1024**3)
    
    # 中间激活：由于使用tiled处理，实际激活显存较小，约5-8x输入
    # 同时处理多个tile时，某些激活可以共享
    intermediate_size = input_size * 6  # 从12倍降低到6倍，更符合实际
    
    # 添加一些额外开销（梯度缓冲区等，虽然inference不需要，但框架可能有保留）
    overhead = 0.5  # 0.5GB额外开销
    
    return input_size + intermediate_size + output_size + overhead

def determine_optimal_batch_size(device: str, tile_coords: List[Tuple[int, int, int, int]], 
                                  frames: torch.Tensor, args, rank: int = 0) -> int:
    """Determine optimal batch size based on available GPU memory.
    
    Args:
        device: GPU device string (e.g., 'cuda:0')
        tile_coords: List of tile coordinates
        frames: Input frames tensor
        args: Arguments object
        rank: Rank ID for logging
    
    Returns:
        Optimal batch size (number of tiles to process simultaneously)
    """
    # 如果用户指定了 batch_size，直接使用
    if hasattr(args, 'tile_batch_size') and args.tile_batch_size > 0:
        batch_size = args.tile_batch_size
        log(f"[Rank {rank}] Using user-specified tile_batch_size: {batch_size}", "info", rank)
        return min(batch_size, len(tile_coords))
    
    # 如果禁用了自适应batch size，返回1
    if hasattr(args, 'adaptive_tile_batch') and not args.adaptive_tile_batch:
        return 1
    
    # 处理 torch.device 对象
    if isinstance(device, torch.device):
        device = str(device)
    if not device.startswith("cuda:"):
        return 1
    
    try:
        # 获取模型加载后的实际可用显存
        available_gb = get_available_memory_gb(device)
        used_gb, total_gb = get_gpu_memory_info(device)
        N = frames.shape[0]
        
        # 估算单个tile所需内存
        tile_size = args.tile_size
        dtype_size = 2 if args.precision in ["fp16", "bf16"] else 4
        tile_memory = estimate_tile_memory(tile_size, N, args.scale, dtype_size)
        
        # 对于大显存GPU（>=24GB），使用更激进的安全边界（只保留1GB）
        # 对于小显存GPU，保留2GB安全边界
        if total_gb >= 24:
            safe_memory = max(1.0, available_gb - 1.0)
            max_batch_limit = 16  # 大显存GPU可以支持更多并发
        else:
            safe_memory = max(2.0, available_gb - 2.0)
            max_batch_limit = 8
        
        # 计算可以同时处理的tile数量
        max_batch = max(1, int(safe_memory / tile_memory))
        optimal_batch = min(max_batch, max_batch_limit, len(tile_coords))
        
        if optimal_batch > 1:
            log(f"[Rank {rank}] [Batch Optimization] GPU: {device}, Total: {total_gb:.1f}GB, Used: {used_gb:.1f}GB, "
                f"Available: {available_gb:.2f}GB", "info", rank)
            log(f"[Rank {rank}] [Batch Optimization] Estimated per-tile: {tile_memory:.2f}GB, "
                f"Safe memory: {safe_memory:.2f}GB, Using batch_size={optimal_batch}", "info", rank)
        
        return optimal_batch
    except Exception as e:
        log(f"[Rank {rank}] Error determining optimal batch size: {e}, falling back to batch_size=1", "warning", rank)
        return 1

def calculate_tile_coords(height, width, tile_size, overlap):
    """Calculate tile coordinates for patch-based inference."""
    coords = []
    stride = tile_size - overlap
    num_rows = math.ceil((height - overlap) / stride)
    num_cols = math.ceil((width - overlap) / stride)
    for r in range(num_rows):
        for c in range(num_cols):
            y1, x1 = r * stride, c * stride
            y2, x2 = min(y1 + tile_size, height), min(x1 + tile_size, width)
            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)
            coords.append((x1, y1, x2, y2))
    return coords

def create_feather_mask(size, overlap):
    """Create blending mask for overlapping tiles with Gaussian blur."""
    import torch.nn.functional as F
    
    H, W = size
    mask = torch.ones(1, 1, H, W, dtype=torch.float32)
    
    ramp = torch.linspace(0, 1, overlap, dtype=torch.float32)
    sigma = max(1.0, overlap / 3.0)
    kernel_size = int(2 * sigma * 2) + 1
    if kernel_size > 1 and kernel_size <= overlap:
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        ramp_padded = F.pad(ramp.unsqueeze(0).unsqueeze(0), (kernel_size//2, kernel_size//2), mode='reflect')
        ramp_blurred = F.conv1d(ramp_padded, gaussian_1d.unsqueeze(0).unsqueeze(0), padding=0)
        ramp = ramp_blurred.squeeze()
        if ramp.max() > ramp.min():
            ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min())
    
    ramp_h = ramp.view(1, 1, -1, 1)
    ramp_w = ramp.view(1, 1, 1, -1)
    
    mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp_w)
    mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp_w.flip(-1))
    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp_h)
    mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp_h.flip(-2))
    
    return mask

def get_total_frame_count(input_path: str) -> int:
    """获取输入的总帧数（不加载所有帧到内存）。"""
    if os.path.isfile(input_path):
        # 视频文件：使用 OpenCV 获取帧数
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count
    elif os.path.isdir(input_path):
        # 图片序列：统计图片文件数量
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_files = []
        for filename in os.listdir(input_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(filename)
        image_files.sort(key=lambda x: natural_sort_key(x))
        return len(image_files)
    else:
        raise RuntimeError(f"Input path does not exist: {input_path}")

def read_input_frames_range(input_path: str, start_idx: int, end_idx: int, fps: float = 30.0) -> Tuple[torch.Tensor, float]:
    """读取指定范围的帧（流式读取，避免一次性加载所有帧）。
    
    Args:
        input_path: 输入路径（视频文件或图片序列目录）
        start_idx: 起始帧索引（包含）
        end_idx: 结束帧索引（不包含）
        fps: 帧率（用于图片序列）
    
    Returns:
        (frames_tensor, fps): 帧 tensor (N, H, W, C) 和 fps
    """
    if os.path.isfile(input_path):
        # 视频文件：读取指定范围的帧
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 1000:
            fps = 30.0
        
        # 跳转到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        
        frames = []
        for i in range(end_idx - start_idx):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
            frames.append(frame_tensor)
        
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames read from video range {start_idx}-{end_idx}")
        
        video_tensor = torch.stack(frames, dim=0)
        return video_tensor, fps
        
    elif os.path.isdir(input_path):
        # 图片序列：读取指定范围的图片
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_files = []
        for filename in os.listdir(input_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(filename)
        
        image_files.sort(key=lambda x: natural_sort_key(x))
        
        if end_idx > len(image_files):
            end_idx = len(image_files)
        if start_idx >= len(image_files):
            raise RuntimeError(f"Start index {start_idx} exceeds total frames {len(image_files)}")
        
        frames = []
        for idx in range(start_idx, end_idx):
            img_path = os.path.join(input_path, image_files[idx])
            try:
                img = cv2.imread(img_path)
                if img is None:
                    log(f"[read_input_frames_range] Warning: Failed to read {img_path}, skipping", "warning", 0)
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(img_rgb).float() / 255.0
                frames.append(frame_tensor)
            except Exception as e:
                log(f"[read_input_frames_range] Warning: Error reading {img_path}: {e}, skipping", "warning", 0)
                continue
        
        if not frames:
            raise RuntimeError(f"No valid frames read from range {start_idx}-{end_idx}")
        
        video_tensor = torch.stack(frames, dim=0)
        return video_tensor, fps
    else:
        raise RuntimeError(f"Input path does not exist: {input_path}")

def split_video_by_frames(total_frames: int, num_workers: int, overlap: int = 2, force_num_workers: bool = False):
    """根据总帧数分割视频段（不加载所有帧到内存）。
    
    优化版本：确保分割后的segments在去掉overlap后能正确合并回原始总帧数。
    
    Args:
        total_frames: 总帧数
        num_workers: 期望的worker数量
        overlap: 重叠帧数
        force_num_workers: 如果为True，强制使用指定的num_workers，即使帧数不足
    
    确保每个 segment 至少有 21 帧（FlashVSR 的最小要求）。
    注意：prepare_input_tensor 会减少帧数（例如 19帧→17帧，21帧→25帧），所以需要至少 21 输入帧才能确保处理后≥17帧。
    如果 force_num_workers=True，会强制使用指定的 num_workers，但每个 segment 仍然会通过扩展来满足最小帧数要求。
    
    分割策略：
    1. 计算每个segment的基础大小（不考虑overlap）
    2. 为每个segment添加overlap（除了第一个和最后一个）
    3. 确保每个segment至少有最小帧数要求
    4. 验证：去掉overlap后，所有segments的帧数总和应该等于原始总帧数
    """
    N = total_frames
    min_frames_per_segment = 21  # FlashVSR 的最小要求
    
    # 如果强制指定worker数量，不减少worker数量
    if not force_num_workers:
        # 如果视频太短，减少 worker 数量
        if N < min_frames_per_segment * num_workers:
            log(f"[Split] Video has only {N} frames, reducing workers from {num_workers} to {min(1, N // min_frames_per_segment)}", "warning", 0)
            num_workers = max(1, N // min_frames_per_segment)
    else:
        # 强制使用指定的worker数量，即使帧数不足
        if N < min_frames_per_segment * num_workers:
            log(f"[Split] WARNING: Video has only {N} frames, but forcing {num_workers} workers. Each worker will get ~{N // num_workers} frames (minimum required: {min_frames_per_segment})", "warning", 0)
            if N < num_workers:
                log(f"[Split] WARNING: Total frames ({N}) < num_workers ({num_workers}). Some workers may get 0 frames.", "warning", 0)
    
    # 计算每个segment的基础大小（不考虑overlap）
    base_segment_size = N // num_workers if num_workers > 0 else N
    remainder = N % num_workers  # 余数，需要分配给前面的segments
    
    segments = []
    current_start = 0
    
    for i in range(num_workers):
        # 计算这个segment的基础大小（考虑余数分配）
        segment_base_size = base_segment_size + (1 if i < remainder else 0)
        
        # 计算segment的边界（添加overlap）
        if i == 0:
            # 第一个segment：从0开始，添加后面的overlap
            start_idx = 0
            end_idx = min(N, segment_base_size + overlap)
        elif i == num_workers - 1:
            # 最后一个segment：添加前面的overlap，到N结束
            start_idx = max(0, current_start - overlap)
            end_idx = N
        else:
            # 中间segments：前后都添加overlap
            start_idx = max(0, current_start - overlap)
            end_idx = min(N, current_start + segment_base_size + overlap)
        
        # 确保每个 segment 至少有最小帧数（FlashVSR 要求）
        actual_frames = end_idx - start_idx
        if actual_frames < min_frames_per_segment:
            if i < num_workers - 1:
                # 如果不是最后一个 segment，向后扩展
                end_idx = min(N, start_idx + min_frames_per_segment)
            else:
                # 如果是最后一个 segment，向前扩展
                start_idx = max(0, end_idx - min_frames_per_segment)
        
        # 确保至少分配1帧（如果总帧数足够）
        if end_idx <= start_idx and N > 0:
            if i < N:
                start_idx = i
                end_idx = min(i + 1, N)
            else:
                start_idx = N
                end_idx = N
        
        segments.append((start_idx, end_idx))
        current_start = end_idx - overlap  # 下一个segment的起始位置（考虑overlap会被去掉）
    
    # 验证分割结果：计算去掉overlap后的总帧数
    total_after_overlap = 0
    for i, (start, end) in enumerate(segments):
        frames_read = end - start
        if i > 0:
            frames_read -= overlap  # 去掉前面的overlap
        if i < len(segments) - 1:
            frames_read -= overlap  # 去掉后面的overlap
        total_after_overlap += max(0, frames_read)
    
    if total_after_overlap != N:
        log(f"[Split] WARNING: After removing overlaps, total frames ({total_after_overlap}) != input frames ({N}). Difference: {N - total_after_overlap} frames.", "warning", 0)
        log(f"[Split] This may be due to segment size adjustments. The merge logic will handle this.", "info", 0)
    
    return segments

def natural_sort_key(filename: str):
    """自然排序键函数，正确处理文件名中的数字部分。
    
    例如：
    - frame_1.jpg < frame_2.jpg < frame_10.jpg（而不是字典序的 frame_1.jpg < frame_10.jpg < frame_2.jpg）
    - img_001.jpg < img_002.jpg < img_100.jpg
    
    Args:
        filename: 文件名
    
    Returns:
        用于排序的键（元组）
    """
    import re
    # 将文件名分割为文本和数字部分
    # 例如 "frame_123.jpg" -> ["frame_", "123", ".jpg"]
    parts = re.split(r'(\d+)', filename)
    # 将数字部分转换为整数，文本部分保持原样
    # 例如 ["frame_", "123", ".jpg"] -> ["frame_", 123, ".jpg"]
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)

def read_image_sequence(image_dir: str, fps: float = 30.0) -> Tuple[torch.Tensor, float]:
    """Read image sequence from directory and convert to tensor.
    
    支持自然排序，正确处理数字开头的文件名（如 frame_1.jpg, frame_2.jpg, ..., frame_10.jpg）。
    无论数字从几开始（0, 1, 100等），都能正确排序。
    
    Args:
        image_dir: Directory containing image files
        fps: Frames per second (default: 30.0)
    
    Returns:
        (frames_tensor, fps): Frames tensor (N, H, W, C) and fps
    """
    if not os.path.isdir(image_dir):
        raise RuntimeError(f"Image directory does not exist: {image_dir}")
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # 获取所有图片文件并使用自然排序（正确处理数字部分）
    image_files = []
    for filename in os.listdir(image_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_dir, filename))
    
    # 使用自然排序，正确处理文件名中的数字
    image_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    if not image_files:
        raise RuntimeError(f"No image files found in directory: {image_dir}")
    
    log(f"[read_image_sequence] Found {len(image_files)} images in {image_dir}", "info", 0)
    log(f"[read_image_sequence] Starting to load images (this may take a while for large sequences)...", "info", 0)
    
    frames = []
    for idx, img_path in enumerate(image_files):
        try:
            img = cv2.imread(img_path)
            if img is None:
                log(f"[read_image_sequence] Warning: Failed to read {img_path}, skipping", "warning", 0)
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(img_rgb).float() / 255.0
            frames.append(frame_tensor)
            
            # 每10%或每10张图片打印一次进度（取较小值，确保有足够提示）
            progress_interval = max(1, min(len(image_files) // 10, 10))
            if (idx + 1) % progress_interval == 0 or (idx + 1) == len(image_files):
                percentage = 100.0 * (idx + 1) / len(image_files)
                log(f"[read_image_sequence] Loaded {idx + 1}/{len(image_files)} images ({percentage:.1f}%)...", "info", 0)
        except Exception as e:
            log(f"[read_image_sequence] Warning: Error reading {img_path}: {e}, skipping", "warning", 0)
            continue
    
    if not frames:
        raise RuntimeError(f"No valid images read from directory: {image_dir}")
    
    log(f"[read_image_sequence] Stacking {len(frames)} frames into tensor (this may take a moment)...", "info", 0)
    video_tensor = torch.stack(frames, dim=0)
    log(f"[read_image_sequence] Successfully loaded {len(frames)} frames, tensor shape: {video_tensor.shape}", "info", 0)
    return video_tensor, fps

def read_video_to_tensor(video_path: str) -> Tuple[torch.Tensor, float]:
    """Read video file and convert to tensor."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 1000:  # Sanity check
        fps = 30.0
    
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
        frames.append(frame_tensor)
    
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from video: {video_path}")
    
    video_tensor = torch.stack(frames, dim=0)
    return video_tensor, fps

def read_input_to_tensor(input_path: str, fps: float = 30.0) -> Tuple[torch.Tensor, float]:
    """Read input (video file or image sequence directory) and convert to tensor.
    
    Automatically detects whether input_path is:
    - A video file (if it's a file with video extension)
    - An image sequence directory (if it's a directory)
    
    Args:
        input_path: Path to video file or image sequence directory
        fps: Frames per second (used for image sequence, default: 30.0)
    
    Returns:
        (frames_tensor, fps): Frames tensor (N, H, W, C) and fps
    """
    if os.path.isfile(input_path):
        # 检查是否是视频文件
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg'}
        if any(input_path.lower().endswith(ext) for ext in video_extensions):
            log(f"[read_input] Detected video file: {input_path}", "info", 0)
            return read_video_to_tensor(input_path)
        else:
            raise RuntimeError(f"Unsupported file format: {input_path}. Expected video file or image directory.")
    elif os.path.isdir(input_path):
        # 图片序列目录
        log(f"[read_input] Detected image sequence directory: {input_path}", "info", 0)
        return read_image_sequence(input_path, fps)
    else:
        raise RuntimeError(f"Input path does not exist: {input_path}")

def get_default_output_path(input_path: str, scale: int = 4) -> str:
    """Generate default output path based on input path.
    
    Args:
        input_path: Input video file or image directory path
        scale: Upscale factor (for naming)
    
    Returns:
        Default output video path
    """
    if os.path.isfile(input_path):
        # 视频文件：替换扩展名
        base_name = os.path.splitext(input_path)[0]
        return f"{base_name}_{scale}x.mp4"
    elif os.path.isdir(input_path):
        # 图片目录：在目录同级创建视频文件
        dir_name = os.path.basename(input_path.rstrip('/'))
        parent_dir = os.path.dirname(input_path) if os.path.dirname(input_path) else "."
        return os.path.join(parent_dir, f"{dir_name}_{scale}x.mp4")
    else:
        # Fallback
        return "output.mp4"

def save_video(frames: torch.Tensor, output_path: str, fps: float = 30.0):
    """Save video tensor to file using FFmpeg (with robust error handling).
    
    如果 FFmpeg pipe 方法失败，会自动回退到临时文件方法。
    """
    import subprocess
    import tempfile
    import os
    
    F, H, W, C = frames.shape
    frames_np = (frames.clamp(0, 1) * 255).byte().cpu().numpy().astype('uint8')
    
    # 方法1：尝试使用 pipe（快速，但可能遇到 BrokenPipeError）
    try:
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{W}x{H}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', 'pipe:0',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-crf', '18',
            output_path
        ]
        
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 分批写入，避免一次性写入太多数据导致 BrokenPipeError
        batch_size = 10  # 每次写入10帧
        for i in range(0, F, batch_size):
            end_idx = min(i + batch_size, F)
            batch = frames_np[i:end_idx]
            try:
                process.stdin.write(batch.tobytes())
            except BrokenPipeError:
                # 如果 pipe 断开，回退到临时文件方法
                process.stdin.close()
                process.wait()
                log(f"[save_video] Pipe method failed, falling back to temp file method...", "warning")
                raise BrokenPipeError("Pipe broken, will use temp file method")
        
        process.stdin.close()
        process.wait()
        
        if process.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            log(f"[save_video] Successfully saved video using FFmpeg pipe: {output_path}", "info")
            return
        
        # 如果返回码非0，回退到临时文件方法
        stderr = process.stderr.read().decode('utf-8', errors='ignore')
        log(f"[save_video] FFmpeg pipe failed (return code {process.returncode}), falling back to temp file method...", "warning")
        if stderr:
            log(f"[save_video] FFmpeg error: {stderr[:200]}", "warning")
    except (BrokenPipeError, OSError) as e:
        log(f"[save_video] Pipe method failed: {e}, using temp file method...", "warning")
    
    # 方法2：使用临时文件（更健壮，避免 BrokenPipeError）
    tmp_yuv_path = None
    try:
        # 创建临时文件
        tmp_yuv = tempfile.NamedTemporaryFile(suffix='.yuv', delete=False)
        tmp_yuv_path = tmp_yuv.name
        
        # 分批写入临时文件
        batch_size = 50  # 临时文件可以写入更多帧
        for i in range(0, F, batch_size):
            end_idx = min(i + batch_size, F)
            batch = frames_np[i:end_idx]
            tmp_yuv.write(batch.tobytes())
            if (i + batch_size) % 100 == 0 or end_idx == F:
                log(f"[save_video] Writing frames to temp file: {end_idx}/{F} ({100*end_idx//F}%)", "info")
        
        tmp_yuv.close()
        
        # 使用 FFmpeg 从临时文件编码
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{W}x{H}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', tmp_yuv_path,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-crf', '18',
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            log(f"[save_video] Successfully saved video using FFmpeg temp file: {output_path}", "info")
            return
        else:
            stderr = result.stderr.decode('utf-8', errors='ignore')
            raise RuntimeError(f"FFmpeg failed with return code {result.returncode}: {stderr[:500]}")
    finally:
        # 清理临时文件
        if tmp_yuv_path and os.path.exists(tmp_yuv_path):
            try:
                os.unlink(tmp_yuv_path)
            except:
                pass

# ==============================================================
#              分布式模型加载和初始化
# ==============================================================

def init_pipeline_distributed(rank: int, world_size: int, mode: str, dtype: torch.dtype, model_dir: str, use_shared_memory: bool = True, device_id: int = None):
    """在分布式环境中初始化pipeline，使用共享内存优化模型加载。
    
    关键优化：
    1. 使用 /dev/shm (共享内存文件系统) 存储模型权重，所有进程共享同一份模型文件
    2. 错开加载时间，避免同时加载导致内存峰值
    3. 如果共享内存不可用，回退到错开加载策略
    
    Args:
        device_id: 实际使用的 GPU 设备索引（如果为 None，则使用 rank）
    """
    if device_id is None:
        device_id = rank
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)
    
    model_path = model_dir
    ckpt_path = os.path.join(model_path, "diffusion_pytorch_model_streaming_dmd.safetensors")
    vae_path = os.path.join(model_path, "Wan2.1_VAE.pth")
    lq_path = os.path.join(model_path, "LQ_proj_in.ckpt")
    tcd_path = os.path.join(model_path, "TCDecoder.ckpt")
    prompt_path = os.path.join(_project_root, "posi_prompt.pth")
    
    # 验证文件存在
    required_files = [ckpt_path]
    if mode == "full":
        required_files.append(vae_path)
    else:
        required_files.extend([lq_path, tcd_path])
    
    for p in required_files:
        if not os.path.exists(p):
            raise RuntimeError(f"[Rank {rank}] Missing model file: {p}")
    
    log(f"[Rank {rank}] Loading model weights...", "info", rank)
    
    # 尝试使用共享内存优化（/dev/shm）
    shm_base = "/dev/shm/flashvsr_models"
    use_shm = use_shared_memory and os.path.exists("/dev/shm")
    
    if use_shm and rank == 0:
        # Rank 0: 加载模型并保存到共享内存
        os.makedirs(shm_base, exist_ok=True)
        log(f"[Rank 0] Using shared memory for model loading: {shm_base}", "info", rank)
        
        # 检查共享内存中是否已有模型
        shm_ckpt = os.path.join(shm_base, "model_ckpt.pt")
        shm_vae = os.path.join(shm_base, "model_vae.pt") if mode == "full" else None
        shm_tcd = os.path.join(shm_base, "model_tcd.pt") if mode != "full" else None
        shm_lq = os.path.join(shm_base, "model_lq.pt")
        
        if not os.path.exists(shm_ckpt):
            log(f"[Rank 0] Loading model to shared memory (one-time operation)...", "info", rank)
            mm = ModelManager(torch_dtype=dtype, device="cpu")
            if mode == "full":
                mm.load_models([ckpt_path, vae_path])
                pipe = FlashVSRFullPipeline.from_model_manager(mm, device="cpu")
                pipe.vae.model.encoder = None
                pipe.vae.model.conv1 = None
            else:
                mm.load_models([ckpt_path])
                pipe = FlashVSRTinyPipeline.from_model_manager(mm, device="cpu") if mode == "tiny" else FlashVSRTinyLongPipeline.from_model_manager(mm, device="cpu")
                multi_scale_channels = [512, 256, 128, 128]
                pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, device="cpu", dtype=dtype, new_latent_channels=16+768)
                pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location="cpu"), strict=False)
                pipe.TCDecoder.clean_mem()
            
            pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to("cpu", dtype=dtype)
            pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu"), strict=True)
            
            # 保存到共享内存
            torch.save(pipe.dit.state_dict() if hasattr(pipe, 'dit') else None, shm_ckpt)
            if mode == "full" and hasattr(pipe, 'vae'):
                torch.save(pipe.vae.state_dict(), shm_vae)
            if mode != "full" and hasattr(pipe, 'TCDecoder'):
                torch.save(pipe.TCDecoder.state_dict(), shm_tcd)
            torch.save(pipe.denoising_model().LQ_proj_in.state_dict() if hasattr(pipe, 'denoising_model') else None, shm_lq)
            
            del mm, pipe
            gc.collect()
            log(f"[Rank 0] Model saved to shared memory", "info", rank)
        
        # Rank 0 也从共享内存加载（保持一致性）
        # 注意：不使用 barrier，让其他进程可以立即开始加载（通过文件存在性检查）
        # 注意：这里一定要让 pipeline 的 device 属性就是目标 GPU，
        # 否则后面 init_cross_kv 时 ctx 会在 CPU 而权重在 CUDA，导致 device mismatch。
        mm = ModelManager(torch_dtype=dtype, device="cpu")
        if mode == "full":
            mm.load_models([ckpt_path, vae_path])  # 仍然需要原始文件来初始化结构
            pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
            pipe.vae.model.encoder = None
            pipe.vae.model.conv1 = None
            if os.path.exists(shm_vae):
                pipe.vae.load_state_dict(torch.load(shm_vae, map_location="cpu"))
        else:
            mm.load_models([ckpt_path])
            pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device) if mode == "tiny" else FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
            multi_scale_channels = [512, 256, 128, 128]
            pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, device="cpu", dtype=dtype, new_latent_channels=16+768)
            if os.path.exists(shm_tcd):
                pipe.TCDecoder.load_state_dict(torch.load(shm_tcd, map_location="cpu"), strict=False)
            pipe.TCDecoder.clean_mem()
        
        pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to("cpu", dtype=dtype)
        if os.path.exists(shm_lq):
            pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(shm_lq, map_location="cpu"), strict=True)
        
        del mm
        gc.collect()
    else:
        # 不使用共享内存或非 rank 0: 错开加载
        if rank != 0:
            # 如果使用共享内存，等待 rank 0 保存完成（通过文件存在性检查）
            if use_shm:
                shm_ckpt = os.path.join(shm_base, "model_ckpt.pt")
                max_wait = 300  # 最多等待5分钟
                wait_start = time.time()
                while not os.path.exists(shm_ckpt) and (time.time() - wait_start) < max_wait:
                    time.sleep(1)
                if not os.path.exists(shm_ckpt):
                    log(f"[Rank {rank}] WARNING: Shared memory model file not found after waiting", "warning", rank)
            
            delay = rank * 2.0  # 每个进程延迟2秒
            log(f"[Rank {rank}] Waiting {delay:.1f}s before loading model (staggered loading)...", "info", rank)
            time.sleep(delay)
        
        mm = ModelManager(torch_dtype=dtype, device="cpu")
        if mode == "full":
            mm.load_models([ckpt_path, vae_path])
            pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
            pipe.vae.model.encoder = None
            pipe.vae.model.conv1 = None
        else:
            mm.load_models([ckpt_path])
            pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device) if mode == "tiny" else FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
            multi_scale_channels = [512, 256, 128, 128]
            pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, device="cpu", dtype=dtype, new_latent_channels=16+768)
            pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location="cpu"), strict=False)
            pipe.TCDecoder.clean_mem()
        
        pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to("cpu", dtype=dtype)
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu"), strict=True)
        
        del mm
        gc.collect()
        
        # 不使用 barrier：每个 rank 独立加载，完成后继续
    
    # 所有进程将模型移到对应GPU
    log(f"[Rank {rank}] Moving model to GPU {rank}...", "info", rank)
    pipe.to(device, dtype=dtype)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path=prompt_path)
    pipe.load_models_to_device(["dit", "vae"])
    
    gc.collect()
    torch.cuda.empty_cache()
    
    log(f"[Rank {rank}] Pipeline initialized successfully", "finish", rank)
    return pipe, device

# ==============================================================
#              单 GPU 推理函数
# ==============================================================

def run_single_gpu_inference(args, total_frames: int, input_fps: float, device_id: int):
    """在单 GPU 上运行推理（不使用分布式，简化流程）。"""
    import time as time_module
    
    log(f"[Single-GPU] Starting single GPU inference on GPU {device_id}", "info")
    
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)
    
    # 处理 attention_mode
    if args.attention_mode == "sparse_sage_attention":
        wan_video_dit.USE_BLOCK_ATTN = False
    else:
        wan_video_dit.USE_BLOCK_ATTN = True
        if not wan_video_dit.BLOCK_ATTN_AVAILABLE:
            log(f"[Single-GPU] Warning: block_sparse_attention not available. Auto-switching to sparse_sage_attention", "warning")
            wan_video_dit.USE_BLOCK_ATTN = False
    
    # 初始化pipeline（单 GPU 模式，不需要共享内存优化）
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.precision, torch.bfloat16)
    
    model_dir = f"/app/models/v{args.model_ver}"
    pipe, device_obj = init_pipeline_distributed(0, 1, args.mode, dtype, model_dir, use_shared_memory=False, device_id=device_id)
    device_str = str(device_obj) if isinstance(device_obj, torch.device) else device_obj
    
    # 读取所有帧（单 GPU 模式可以一次性加载）
    log(f"[Single-GPU] Reading all {total_frames} frames...", "info")
    segment_frames = read_input_frames_range(args.input, 0, total_frames, fps=input_fps)[0]
    log(f"[Single-GPU] Loaded {segment_frames.shape[0]} frames, shape: {segment_frames.shape}", "info")
    
    # 验证帧数
    if segment_frames.shape[0] < 21:
        log(f"[Single-GPU] ERROR: Video has only {segment_frames.shape[0]} frames, minimum is 21. Cannot process.", "error")
        raise ValueError(f"Video too short: {segment_frames.shape[0]} frames (minimum: 21)")
    
    # 运行推理
    log(f"[Single-GPU] Starting inference on {segment_frames.shape[0]} frames...", "info")
    output = run_inference_distributed_segment(pipe, segment_frames, device_str, dtype, args, rank=0, checkpoint_dir=None)
    log(f"[Single-GPU] ✓ Inference completed, output shape: {output.shape}", "info")
    
    # 保存结果到临时目录（与分布式模式保持一致）
    video_dir_name = f"{hashlib.md5(args.input.encode()).hexdigest()}_{args.scale}x"
    temp_base = getattr(args, 'temp_base', '/app/output')
    checkpoint_dir = os.path.join(temp_base, 'flashvsr_distributed', video_dir_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    output_file = os.path.join(checkpoint_dir, "rank_0_result.pt")
    log(f"[Single-GPU] Saving result to {output_file}...", "info")
    torch.save(output, output_file)
    log(f"[Single-GPU] ✓ Result saved: {output.shape[0]} frames, file size: {os.path.getsize(output_file) / (1024**2):.2f} MB", "finish")
    
    # 创建完成标志
    done_file = os.path.join(checkpoint_dir, "rank_0_done.flag")
    with open(done_file, 'w') as f:
        f.write("rank_0_completed\n")
    
    log(f"[Single-GPU] ✓ Inference completed!", "finish")
    
    # 直接保存最终输出（单 GPU 模式不需要合并）
    log(f"[Single-GPU] Saving final output...", "info")
    output_mode = getattr(args, 'output_mode', 'video')
    output_path = args.output if args.output else get_default_output_path(args.input, args.scale)
    
    if output_mode == "pictures":
        log(f"[Single-GPU] Saving frames to {output_path} (image sequence mode)...", "info")
        os.makedirs(output_path, exist_ok=True)
        output_np = (output.clamp(0, 1) * 255).byte().cpu().numpy().astype('uint8')
        for frame_idx in range(output_np.shape[0]):
            frame = output_np[frame_idx]
            frame_filename = os.path.join(output_path, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        log(f"[Single-GPU] ✓ Saved {output_np.shape[0]} frames to {output_path}", "finish")
    else:
        log(f"[Single-GPU] Saving video to {output_path}...", "info")
        from save_recovered_video import save_video
        save_video(output, output_path, input_fps)
        log(f"[Single-GPU] ✓ Video saved: {output_path}", "finish")
    
    log(f"[Single-GPU] ✓ All steps completed!", "finish")
    return output

# ==============================================================
#              分布式推理主函数
# ==============================================================

def process_tile_batch_distributed(pipe, frames, device, dtype, args, tile_batch: List[Tuple[int, int, int, int]], batch_idx: int, rank: int):
    """处理一批tiles（从原版复制并适配分布式）。"""
    N, H, W, C = frames.shape
    num_aligned_frames = largest_8n1_leq(N + 4) - 4

    results = []
    
    for tile_idx, (x1, y1, x2, y2) in enumerate(tile_batch):
        input_tile = frames[:, y1:y2, x1:x2, :]

        LQ_tile, th, tw, F = prepare_input_tensor(input_tile, device, scale=args.scale, dtype=dtype)
        if "long" not in args.mode:
            LQ_tile = LQ_tile.to(device)

        topk_ratio = args.sparse_ratio * 768 * 1280 / (th * tw)

        # 验证参数：确保 num_frames 足够大
        if F < 17:  # FlashVSR 需要至少 17 帧才能正常工作（这是处理后的帧数）
            log(f"[Rank {rank}] WARNING: Tile has only {F} frames, minimum is 17. Skipping this tile.", "warning", rank)
            # 返回一个占位结果（用最后一帧填充）
            placeholder = frames[-1:, y1:y2, x1:x2, :].repeat(F * args.scale, 1, 1, 1)
            return [{
                'coords': (x1, y1, x2, y2),
                'tile': placeholder,
                'mask': torch.ones(1, placeholder.shape[1], placeholder.shape[2], 1)
            }]

        with torch.no_grad():
            try:
                output_tile = pipe(
                    prompt="",
                    negative_prompt="",
                    cfg_scale=1.0,
                    num_inference_steps=1,
                    seed=args.seed,
                    tiled=args.tiled_vae,
                    LQ_video=LQ_tile,
                    num_frames=F,
                    height=th,
                    width=tw,
                    is_full_block=False,
                    if_buffer=True,
                    topk_ratio=topk_ratio,
                    kv_ratio=args.kv_ratio,
                    local_range=args.local_range,
                    color_fix=args.color_fix,
                    unload_dit=args.unload_dit,
                )
            except ValueError as e:
                if "expected a non-empty list" in str(e):
                    log(f"[Rank {rank}] ERROR: Pipeline returned empty latents. This may be due to insufficient frames (F={F}) or tile size issue.", "error", rank)
                    log(f"[Rank {rank}] Tile info: coords=({x1},{y1},{x2},{y2}), th={th}, tw={tw}, F={F}", "error", rank)
                    raise RuntimeError(f"Pipeline failed: empty latents. Tile may be too small or have insufficient frames (F={F}, min=17)") from e
                else:
                    raise

        processed_tile_cpu = tensor2video(output_tile).to("cpu")

        mask_nchw = create_feather_mask(
            (processed_tile_cpu.shape[1], processed_tile_cpu.shape[2]),
            args.tile_overlap * args.scale,
        ).to("cpu")
        mask_nhwc = mask_nchw.permute(0, 2, 3, 1)
        
        results.append({
            'coords': (x1, y1, x2, y2),
            'tile': processed_tile_cpu,
            'mask': mask_nhwc
        })
        
        del LQ_tile, output_tile, processed_tile_cpu, input_tile
        clean_vram()
    
    return results

def run_inference_distributed_segment(pipe, frames, device, dtype, args, rank: int, checkpoint_dir: str = None):
    """在单个segment上运行完整的推理流程（支持内存映射和断点续跑）。
    
    关键优化：
    1. 使用内存映射文件（mmap）存储 canvas，避免内存不足
    2. Tile 级流式输出：处理完一个 tile 就写入磁盘
    3. 支持断点续跑：可以从中断点恢复
    """
    N, H, W, C = frames.shape
    num_aligned_frames = largest_8n1_leq(N + 4) - 4
    out_H, out_W = H * args.scale, W * args.scale
    
    # 计算tile坐标
    tile_coords = calculate_tile_coords(H, W, args.tile_size, args.tile_overlap)
    log(f"[Rank {rank}] Input resolution: {H}x{W}, Tile size: {args.tile_size}, Overlap: {args.tile_overlap}", "info", rank)
    log(f"[Rank {rank}] Calculated {len(tile_coords)} tiles to process", "info", rank)
    
    # 检查断点续跑：读取已处理的 tile 列表
    processed_tiles_file = os.path.join(checkpoint_dir, f"rank_{rank}_processed_tiles.json") if checkpoint_dir else None
    processed_tiles = set()
    if processed_tiles_file and os.path.exists(processed_tiles_file):
        try:
            with open(processed_tiles_file, 'r') as f:
                processed_tiles = set(json.load(f))
            log(f"[Rank {rank}] Resuming from checkpoint: {len(processed_tiles)}/{len(tile_coords)} tiles already processed", "info", rank)
        except:
            pass
    
    # 使用内存映射文件存储 canvas（避免内存不足）
    use_mmap = checkpoint_dir is not None
    canvas = None
    weight_canvas = None
    canvas_mmap = None
    weight_mmap = None
    canvas_mmap_path = None
    weight_mmap_path = None
    
    if use_mmap:
        canvas_mmap_path = os.path.join(checkpoint_dir, f"rank_{rank}_canvas.npy")
        weight_mmap_path = os.path.join(checkpoint_dir, f"rank_{rank}_weight.npy")
        
        # 创建或加载内存映射文件
        if os.path.exists(canvas_mmap_path):
            log(f"[Rank {rank}] Loading existing memory-mapped canvas from {canvas_mmap_path}", "info", rank)
            canvas_mmap = np.memmap(canvas_mmap_path, dtype=np.float16, mode='r+', 
                                    shape=(num_aligned_frames, out_H, out_W, C))
            weight_mmap = np.memmap(weight_mmap_path, dtype=np.float16, mode='r+',
                                    shape=(num_aligned_frames, out_H, out_W, C))
        else:
            log(f"[Rank {rank}] Creating memory-mapped canvas: {canvas_mmap_path} (shape: {num_aligned_frames}x{out_H}x{out_W}x{C})", "info", rank)
            # 创建内存映射文件
            canvas_mmap = np.memmap(canvas_mmap_path, dtype=np.float16, mode='w+',
                                    shape=(num_aligned_frames, out_H, out_W, C))
            weight_mmap = np.memmap(weight_mmap_path, dtype=np.float16, mode='w+',
                                    shape=(num_aligned_frames, out_H, out_W, C))
            canvas_mmap[:] = 0
            weight_mmap[:] = 0
            # 同步到磁盘
            canvas_mmap.flush()
            weight_mmap.flush()
        
        # 转换为 torch tensor（共享内存）
        canvas = torch.from_numpy(canvas_mmap)
        weight_canvas = torch.from_numpy(weight_mmap)
    else:
        # 回退到内存模式（如果 checkpoint_dir 为 None）
        canvas = torch.zeros((num_aligned_frames, out_H, out_W, C), dtype=torch.float16, device="cpu")
        weight_canvas = torch.zeros_like(canvas)
    
    # 确定最优批量大小
    optimal_batch_size = determine_optimal_batch_size(device, tile_coords, frames, args, rank)
    log(f"[Rank {rank}] Processing {len(tile_coords)} tiles with batch_size={optimal_batch_size} (this may take a while)...", "info", rank)
    
    # 将 tiles 分成批次
    tile_batches = [tile_coords[i:i + optimal_batch_size] 
                    for i in range(0, len(tile_coords), optimal_batch_size)]
    
    total_processed = 0
    flush_counter = 0  # 用于控制内存映射 flush 频率
    
    for batch_idx, tile_batch in enumerate(tile_batches):
        # 检查批次中哪些 tiles 已处理（断点续跑）
        tiles_to_process = []
        tile_keys_in_batch = []
        
        for tile_coord in tile_batch:
            x1, y1, x2, y2 = tile_coord
            tile_key = f"{x1}_{y1}_{x2}_{y2}"
            tile_keys_in_batch.append(tile_key)
            
            if tile_key not in processed_tiles:
                tiles_to_process.append(tile_coord)
            else:
                log(f"[Rank {rank}] Tile {tile_key} already processed, skipping", "info", rank)
        
        # 如果批次中所有 tiles 都已处理，跳过
        if not tiles_to_process:
            # 这些 tiles 已经处理过，更新计数（但不重复添加到 processed_tiles）
            total_processed += len(tile_batch)
            continue
        
        # 批量处理 tiles
        try:
            results = process_tile_batch_distributed(pipe, frames, device, dtype, args, tiles_to_process, batch_idx, rank)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # 如果 OOM，回退到单个 tile 处理
                log(f"[Rank {rank}] OOM with batch_size={len(tiles_to_process)}, falling back to single tile processing", "warning", rank)
                for tile_coord in tiles_to_process:
                    x1, y1, x2, y2 = tile_coord
                    tile_key = f"{x1}_{y1}_{x2}_{y2}"
                    if tile_key in processed_tiles:
                        continue
                    
                    single_tile_batch = [tile_coord]
                    try:
                        results = process_tile_batch_distributed(pipe, frames, device, dtype, args, single_tile_batch, batch_idx, rank)
                        for result in results:
                            x1, y1, x2, y2 = result['coords']
                            processed_tile_cpu = result['tile']
                            mask_nhwc = result['mask']
                            
                            out_x1, out_y1 = x1 * args.scale, y1 * args.scale
                            tile_H_scaled = processed_tile_cpu.shape[1]
                            tile_W_scaled = processed_tile_cpu.shape[2]
                            out_x2, out_y2 = out_x1 + tile_W_scaled, out_y1 + tile_H_scaled
                            
                            canvas[:, out_y1:out_y2, out_x1:out_x2, :] += processed_tile_cpu * mask_nhwc
                            weight_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_nhwc
                            
                            processed_tiles.add(tile_key)
                    except RuntimeError as e2:
                        log(f"[Rank {rank}] ERROR: Failed to process tile {tile_key}: {e2}", "error", rank)
                        raise
            else:
                raise
        
        # 写入结果到 canvas
        for result in results:
            x1, y1, x2, y2 = result['coords']
            processed_tile_cpu = result['tile']
            mask_nhwc = result['mask']
            
            out_x1, out_y1 = x1 * args.scale, y1 * args.scale
            tile_H_scaled = processed_tile_cpu.shape[1]
            tile_W_scaled = processed_tile_cpu.shape[2]
            out_x2, out_y2 = out_x1 + tile_W_scaled, out_y1 + tile_H_scaled
            
            # 流式写入：立即写入 canvas（内存映射会自动同步到磁盘）
            canvas[:, out_y1:out_y2, out_x1:out_x2, :] += processed_tile_cpu * mask_nhwc
            weight_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_nhwc
        
        # 标记为已处理（断点续跑）
        # 从 results 中获取实际处理的 tiles
        for result in results:
            x1, y1, x2, y2 = result['coords']
            tile_key = f"{x1}_{y1}_{x2}_{y2}"
            if tile_key not in processed_tiles:
                processed_tiles.add(tile_key)
                total_processed += 1
        
        # 保存处理状态（批量保存，减少 I/O）
        if processed_tiles_file and (batch_idx % 5 == 0 or batch_idx == len(tile_batches) - 1):
            try:
                with open(processed_tiles_file, 'w') as f:
                    json.dump(list(processed_tiles), f)
            except:
                pass
        
        # 批量 flush 内存映射（每 5 个批次 flush 一次，减少 I/O 开销）
        flush_counter += 1
        if use_mmap and (flush_counter % 5 == 0 or batch_idx == len(tile_batches) - 1):
            canvas_mmap.flush()
            weight_mmap.flush()
        
        clean_vram()
        
        # 每10%或每10个批次打印一次进度，并写入进度文件
        progress_interval = max(1, min(len(tile_batches) // 10, 10))
        if (batch_idx + 1) % progress_interval == 0 or (batch_idx + 1) == len(tile_batches):
            percentage = 100.0 * total_processed / len(tile_coords)
            log(f"[Rank {rank}] Tile progress: {total_processed}/{len(tile_coords)} ({percentage:.1f}%) [batch {batch_idx + 1}/{len(tile_batches)}]", "info", rank)
            
            # 写入进度文件，供 rank 0 读取并显示总进度
            if checkpoint_dir:
                progress_file = os.path.join(checkpoint_dir, f"rank_{rank}_progress.txt")
                try:
                    with open(progress_file, 'w') as f:
                        f.write(f"{total_processed}\n{len(tile_coords)}\n{percentage:.1f}\n")
                except:
                    pass
    
    # 归一化canvas（需要先同步到磁盘，然后转换为普通 tensor 进行计算）
    if use_mmap:
        # 同步到磁盘
        canvas_mmap.flush()
        weight_mmap.flush()
        # 转换为普通 numpy array 进行计算（创建副本，避免修改原始 memmap）
        canvas_np = np.array(canvas_mmap, copy=True).astype(np.float32)
        weight_np = np.array(weight_mmap, copy=True).astype(np.float32)
        weight_np[weight_np == 0] = 1.0
        output_np = canvas_np / weight_np
        output = torch.from_numpy(output_np)  # 已经是 float32
    else:
        weight_canvas[weight_canvas == 0] = 1.0
        output = canvas / weight_canvas
    
    # 写入最终进度（100%）
    if checkpoint_dir:
        progress_file = os.path.join(checkpoint_dir, f"rank_{rank}_progress.txt")
        try:
            with open(progress_file, 'w') as f:
                f.write(f"{len(tile_coords)}\n{len(tile_coords)}\n100.0\n")
        except:
            pass
    
    # 裁剪到实际处理的帧数（考虑 prepare_input_tensor 可能减少的帧数）
    # 注意：prepare_input_tensor 中的 largest_8n1_leq 可能会减少帧数
    # 例如：22帧 -> 21帧，24帧 -> 21帧，23帧 -> 21帧
    # 但我们应该保持原始输入帧数 N，因为后续合并时会处理 overlap
    # 如果 output 的帧数多于 N，说明 prepare_input_tensor 增加了帧数（不应该发生）
    # 如果 output 的帧数少于 N，说明 prepare_input_tensor 减少了帧数，这是正常的
    # 我们保持 output 的原始帧数，不进行裁剪，让后续的 overlap 处理来正确合并
    # 但为了安全，如果 output 帧数明显多于 N，进行裁剪
    if output.shape[0] > N + 4:  # 允许一些padding，但如果太多则裁剪
        log(f"[Rank {rank}] WARNING: Output frames ({output.shape[0]}) > input frames ({N}) + padding, cropping to {N}", "warning", rank)
        output = output[:N]
    # 否则保持 output 的原始帧数，让后续的 overlap 处理来正确合并
    
    # 清理内存映射文件（可选：保留用于调试）
    # if use_mmap and canvas_mmap_path and os.path.exists(canvas_mmap_path):
    #     try:
    #         os.remove(canvas_mmap_path)
    #         os.remove(weight_mmap_path)
    #     except:
    #         pass
    
    return output

def run_with_device(rank: int, world_size: int, args, total_frames: int, input_fps: float, device_indices: List[int]):
    """包装函数，将设备索引传递给分布式推理函数（必须在模块顶层，可被 pickle）。"""
    device_id = device_indices[rank] if rank < len(device_indices) else rank
    return run_distributed_inference(rank, world_size, args, total_frames, input_fps, device_id=device_id)

def run_distributed_inference(rank: int, world_size: int, args, total_frames: int, input_fps: float, device_id: int = None):
    """在单个进程中运行分布式推理（流式处理，不一次性加载所有帧）。
    
    Args:
        rank: 进程的 rank（用于分布式通信）
        world_size: 总进程数
        args: 参数对象
        total_frames: 总帧数
        input_fps: 输入帧率
        device_id: 实际使用的 GPU 设备索引（如果为 None，则使用 rank）
    """
    # 确保 time 模块可用（避免 UnboundLocalError）
    import time as time_module
    # 初始化分布式环境
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    
    # 确定实际使用的设备索引
    if device_id is None:
        device_id = rank
    
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{args.master_addr}:{args.master_port}',
        world_size=world_size,
        rank=rank
    )
    
    try:
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device_id)
        
        # 处理 attention_mode
        if args.attention_mode == "sparse_sage_attention":
            wan_video_dit.USE_BLOCK_ATTN = False
        else:
            wan_video_dit.USE_BLOCK_ATTN = True
            if not wan_video_dit.BLOCK_ATTN_AVAILABLE:
                log(f"[Rank {rank}] Warning: block_sparse_attention not available. Auto-switching to sparse_sage_attention", "warning", rank)
                wan_video_dit.USE_BLOCK_ATTN = False
        
        # 初始化pipeline
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        dtype = dtype_map.get(args.precision, torch.bfloat16)
        
        model_dir = f"/app/models/v{args.model_ver}"
        pipe, device_obj = init_pipeline_distributed(rank, world_size, args.mode, dtype, model_dir, use_shared_memory=True, device_id=device_id)
        # 确保 device 是字符串格式（用于后续函数调用）
        device = str(device_obj) if isinstance(device_obj, torch.device) else device_obj
        
        # 分割视频帧（确保每个 segment 有足够帧数）
        segment_overlap = getattr(args, 'segment_overlap', 2)
        # 允许动态调整（如果视频太短）
        force_num_workers = False
        segments = split_video_by_frames(total_frames, world_size, overlap=segment_overlap, force_num_workers=force_num_workers)
        actual_num_segments = len(segments)
        
        log(f"[Rank {rank}] ========== Rank {rank} Initialization ==========", "info", rank)
        log(f"[Rank {rank}] Using GPU device: cuda:{device_id}", "info", rank)
        log(f"[Rank {rank}] World size: {world_size} GPUs", "info", rank)
        log(f"[Rank {rank}] Total frames: {total_frames}", "info", rank)
        log(f"[Rank {rank}] Number of segments: {actual_num_segments}", "info", rank)
        
        # 如果 segments 数量少于 world_size，说明视频太短，某些 rank 可能没有任务
        if rank >= actual_num_segments:
            log(f"[Rank {rank}] No segment assigned (video too short, only {actual_num_segments} segments for {world_size} ranks). Exiting gracefully.", "info", rank)
            # 创建一个空的结果文件，标记为完成
            video_dir_name = f"{hashlib.md5(args.input.encode()).hexdigest()}_{args.scale}x"
            temp_base = getattr(args, 'temp_base', '/app/output')
            checkpoint_dir = os.path.join(temp_base, 'flashvsr_distributed', video_dir_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            done_file = os.path.join(checkpoint_dir, f"rank_{rank}_done.flag")
            with open(done_file, 'w') as f:
                f.write(f"rank_{rank}_skipped\n")
            log(f"[Rank {rank}] ==============================================", "info", rank)
            return
        
        start_idx, end_idx = segments[rank]
        log(f"[Rank {rank}] Assigned segment: frames {start_idx}-{end_idx} (total: {end_idx - start_idx} frames)", "info", rank)
        log(f"[Rank {rank}] ==============================================", "info", rank)
        
        # 流式读取：只读取当前 rank 负责的帧段
        log(f"[Rank {rank}] [Step 1/4] Reading frames {start_idx}-{end_idx} from input...", "info", rank)
        input_fps_for_read = getattr(args, 'fps', 30.0)
        segment_frames = read_input_frames_range(args.input, start_idx, end_idx, fps=input_fps_for_read)[0]
        log(f"[Rank {rank}] [Step 1/4] ✓ Loaded {segment_frames.shape[0]} frames, shape: {segment_frames.shape}", "info", rank)
        
        # 验证 segment 帧数
        if segment_frames.shape[0] < 21:
            log(f"[Rank {rank}] ERROR: Segment has only {segment_frames.shape[0]} frames, minimum is 21. Cannot process.", "error", rank)
            log(f"[Rank {rank}] Note: prepare_input_tensor reduces frame count (e.g., 19→17, 21→25), so we need at least 21 input frames.", "error", rank)
            raise ValueError(f"Segment too short: {segment_frames.shape[0]} frames (minimum: 21)")
        
        log(f"[Rank {rank}] [Step 2/4] Segment info: frames {start_idx}-{end_idx} ({segment_frames.shape[0]} frames), resolution: {segment_frames.shape[1]}x{segment_frames.shape[2]}", "info", rank)
        
        # 保存结果到临时文件
        video_dir_name = f"{hashlib.md5(args.input.encode()).hexdigest()}_{args.scale}x"
        temp_base = getattr(args, 'temp_base', '/app/output')
        checkpoint_dir = os.path.join(temp_base, 'flashvsr_distributed', video_dir_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        output_file = os.path.join(checkpoint_dir, f"rank_{rank}_result.pt")
        log(f"[Rank {rank}] [Step 3/4] Output file: {output_file}", "info", rank)
        
        # 运行完整的推理流程（添加异常处理）
        log(f"[Rank {rank}] [Step 3/4] Starting inference on {segment_frames.shape[0]} frames...", "info", rank)
        try:
            segment_output = run_inference_distributed_segment(pipe, segment_frames, device, dtype, args, rank, checkpoint_dir)
            log(f"[Rank {rank}] [Step 3/4] ✓ Inference completed, output shape: {segment_output.shape}", "info", rank)
            
            # 处理overlap：裁剪重叠部分（除了第一个和最后一个segment）
            # 注意：由于 prepare_input_tensor 可能会减少帧数，我们需要根据实际输出帧数来处理overlap
            actual_num_segments = len(segments)
            original_output_frames = segment_output.shape[0]
            
            # 计算这个segment应该输出的帧数（去掉overlap后）
            # 根据原始分割信息计算：segment读取的帧数 - overlap（前面和后面）
            segment_start, segment_end = segments[rank]
            segment_read_frames = segment_end - segment_start
            
            # 计算应该输出的帧数（去掉overlap）
            expected_output_frames = segment_read_frames
            if rank > 0:
                expected_output_frames -= segment_overlap  # 去掉前面的overlap
            if rank < actual_num_segments - 1:
                expected_output_frames -= segment_overlap  # 去掉后面的overlap
            
            # 根据实际输出帧数和期望输出帧数来处理overlap
            if rank > 0:
                # 去掉前面的overlap帧
                if segment_output.shape[0] > segment_overlap:
                    segment_output = segment_output[segment_overlap:]
                else:
                    log(f"[Rank {rank}] WARNING: Output frames ({original_output_frames}) <= overlap ({segment_overlap}), cannot remove front overlap. Keeping all frames.", "warning", rank)
            
            if rank < actual_num_segments - 1:
                # 去掉后面的overlap帧
                if segment_output.shape[0] > segment_overlap:
                    segment_output = segment_output[:-segment_overlap]
                else:
                    log(f"[Rank {rank}] WARNING: Output frames ({segment_output.shape[0]}) <= overlap ({segment_overlap}), cannot remove back overlap. Keeping all frames.", "warning", rank)
            
            # 如果输出帧数与期望不一致，进行填充或裁剪
            actual_output_frames = segment_output.shape[0]
            if actual_output_frames < expected_output_frames:
                # 如果输出帧数少于期望，使用最后一帧填充
                missing = expected_output_frames - actual_output_frames
                log(f"[Rank {rank}] Output frames ({actual_output_frames}) < expected ({expected_output_frames}), padding {missing} frames with last frame", "info", rank)
                last_frame = segment_output[-1:, :, :, :]
                padding = last_frame.repeat(missing, 1, 1, 1)
                segment_output = torch.cat([segment_output, padding], dim=0)
            elif actual_output_frames > expected_output_frames:
                # 如果输出帧数多于期望，裁剪到期望帧数
                log(f"[Rank {rank}] Output frames ({actual_output_frames}) > expected ({expected_output_frames}), cropping to {expected_output_frames} frames", "info", rank)
                segment_output = segment_output[:expected_output_frames]
            
            if original_output_frames != segment_output.shape[0]:
                log(f"[Rank {rank}] Overlap processing: {original_output_frames} frames -> {segment_output.shape[0]} frames (expected: {expected_output_frames})", "info", rank)
            
            # 保存结果
            log(f"[Rank {rank}] [Step 4/4] Saving result to {output_file}...", "info", rank)
            torch.save(segment_output, output_file)
            log(f"[Rank {rank}] [Step 4/4] ✓ Result saved: {segment_output.shape[0]} frames, file size: {os.path.getsize(output_file) / (1024**2):.2f} MB", "finish", rank)
            
            # 清理内存映射文件（如果存在，已保存结果后可以删除以节省空间）
            canvas_mmap_path = os.path.join(checkpoint_dir, f"rank_{rank}_canvas.npy")
            weight_mmap_path = os.path.join(checkpoint_dir, f"rank_{rank}_weight.npy")
            processed_tiles_file = os.path.join(checkpoint_dir, f"rank_{rank}_processed_tiles.json")
            
            # 可选：删除内存映射文件以节省空间（默认保留用于调试和恢复）
            cleanup_mmap = getattr(args, 'cleanup_mmap', False)
            if cleanup_mmap:
                try:
                    if os.path.exists(canvas_mmap_path):
                        os.remove(canvas_mmap_path)
                        log(f"[Rank {rank}] Cleaned up canvas memory map file", "info", rank)
                    if os.path.exists(weight_mmap_path):
                        os.remove(weight_mmap_path)
                        log(f"[Rank {rank}] Cleaned up weight memory map file", "info", rank)
                    if os.path.exists(processed_tiles_file):
                        os.remove(processed_tiles_file)
                        log(f"[Rank {rank}] Cleaned up processed tiles file", "info", rank)
                except Exception as e:
                    log(f"[Rank {rank}] Warning: Failed to cleanup memory map files: {e}", "warning", rank)
            
            # 创建一个完成标志文件
            done_file = os.path.join(checkpoint_dir, f"rank_{rank}_done.flag")
            with open(done_file, 'w') as f:
                f.write(f"rank_{rank}_completed\n")
            log(f"[Rank {rank}] ✓ All steps completed for Rank {rank}!", "finish", rank)
        except Exception as e:
            # 记录错误，但继续运行（让其他 rank 能完成）
            error_file = os.path.join(checkpoint_dir, f"rank_{rank}_error.txt")
            with open(error_file, 'w') as f:
                import traceback
                f.write(f"Rank {rank} inference failed:\n{str(e)}\n\n{traceback.format_exc()}")
            log(f"[Rank {rank}] ERROR: Inference failed: {e}", "error", rank)
            log(f"[Rank {rank}] Error details saved to {error_file}", "warning", rank)
            # 创建一个失败标志文件
            done_file = os.path.join(checkpoint_dir, f"rank_{rank}_done.flag")
            with open(done_file, 'w') as f:
                f.write(f"rank_{rank}_failed\n")
            # 不重新抛出异常，让其他 rank 继续运行
        
        # Rank 0 负责合并结果（使用文件轮询，避免 barrier 死锁）
        if rank == 0:
            log(f"[Rank 0] ========== Rank 0: Merging Results ==========", "info", rank)
            log(f"[Rank 0] Waiting for all {world_size} ranks to complete...", "info", rank)
            max_wait_time = 3600  # 最多等待1小时
            wait_start = time_module.time()
            all_done = False
            
            while not all_done and (time_module.time() - wait_start) < max_wait_time:
                all_done = True
                completed_ranks = []
                pending_ranks = []
                skipped_ranks = []
                failed_ranks = []
                
                for r in range(world_size):
                    done_flag = os.path.join(checkpoint_dir, f"rank_{r}_done.flag")
                    if os.path.exists(done_flag):
                        with open(done_flag, 'r') as f:
                            status = f.read().strip()
                        if 'completed' in status:
                            completed_ranks.append(r)
                        elif 'skipped' in status:
                            skipped_ranks.append(r)
                        elif 'failed' in status:
                            failed_ranks.append(r)
                    else:
                        pending_ranks.append(r)
                        all_done = False
                
                if not all_done:
                    time_module.sleep(5)  # 每5秒检查一次
                    elapsed = time_module.time() - wait_start
                    if int(elapsed) % 30 == 0:  # 每30秒打印一次状态
                        log(f"[Rank 0] Status update ({int(elapsed)}s elapsed):", "info", rank)
                        log(f"[Rank 0]   Completed: {len(completed_ranks)} ranks {completed_ranks}", "info", rank)
                        log(f"[Rank 0]   Pending: {len(pending_ranks)} ranks {pending_ranks}", "info", rank)
                        if skipped_ranks:
                            log(f"[Rank 0]   Skipped: {len(skipped_ranks)} ranks {skipped_ranks}", "info", rank)
                        if failed_ranks:
                            log(f"[Rank 0]   Failed: {len(failed_ranks)} ranks {failed_ranks}", "warning", rank)
                        
                        # 读取并显示所有 rank 的 tile 处理进度
                        rank_progresses = []
                        total_tiles_processed = 0
                        total_tiles = 0
                        for r in range(world_size):
                            progress_file = os.path.join(checkpoint_dir, f"rank_{r}_progress.txt")
                            if os.path.exists(progress_file):
                                try:
                                    with open(progress_file, 'r') as f:
                                        lines = f.read().strip().split('\n')
                                        if len(lines) >= 3:
                                            tiles_processed = int(lines[0])
                                            tiles_total = int(lines[1])
                                            percentage = float(lines[2])
                                            rank_progresses.append((r, tiles_processed, tiles_total, percentage))
                                            total_tiles_processed += tiles_processed
                                            total_tiles += tiles_total
                                except:
                                    pass
                        
                        if rank_progresses:
                            log(f"[Rank 0] ========== Overall Tile Progress ==========", "info", rank)
                            for r, processed, total, pct in sorted(rank_progresses):
                                log(f"[Rank 0]   Rank {r}: {processed}/{total} tiles ({pct:.1f}%)", "info", rank)
                            if total_tiles > 0:
                                overall_pct = 100.0 * total_tiles_processed / total_tiles
                                log(f"[Rank 0]   Total: {total_tiles_processed}/{total_tiles} tiles ({overall_pct:.1f}%)", "info", rank)
                            log(f"[Rank 0] ============================================", "info", rank)
            
            if not all_done:
                log(f"[Rank 0] WARNING: Not all ranks completed within timeout. Proceeding with available results...", "warning", rank)
            
            log(f"[Rank 0] All ranks finished. Starting streaming merge...", "info", rank)
            
            # 收集所有有效的结果文件信息
            result_files = []
            for r in range(world_size):
                result_file = os.path.join(checkpoint_dir, f"rank_{r}_result.pt")
                done_file = os.path.join(checkpoint_dir, f"rank_{r}_done.flag")
                
                # 检查是否被跳过或失败
                if os.path.exists(done_file):
                    with open(done_file, 'r') as f:
                        status = f.read().strip()
                    if 'skipped' in status:
                        log(f"[Rank 0]   Rank {r}: SKIPPED (no segment assigned)", "info", rank)
                        continue
                    elif 'failed' in status:
                        log(f"[Rank 0]   Rank {r}: FAILED (check {checkpoint_dir}/rank_{r}_error.txt)", "warning", rank)
                        continue
                
                if os.path.exists(result_file):
                    file_size_mb = os.path.getsize(result_file) / (1024**2)
                    result_files.append((r, result_file, file_size_mb))
                    log(f"[Rank 0]   Rank {r}: ✓ Result file found, {file_size_mb:.2f} MB", "info", rank)
                else:
                    log(f"[Rank 0]   Rank {r}: ✗ Result file not found", "warning", rank)
            
            if not result_files:
                raise RuntimeError("[Rank 0] No valid results to merge!")
            
            # 按 rank 顺序排序
            result_files.sort(key=lambda x: x[0])
            rank_order = [r[0] for r in result_files]
            total_size_mb = sum(r[2] for r in result_files)
            log(f"[Rank 0] Merging {len(result_files)} rank results in order: {rank_order}", "info", rank)
            log(f"[Rank 0] Using streaming merge to avoid OOM (total result files: {total_size_mb:.2f} MB)", "info", rank)
            
            # 流式合并：逐个加载、处理、写入视频，避免一次性加载所有到内存
            output_mode = getattr(args, 'output_mode', 'video')  # 默认为 video
            output_path = args.output if args.output else get_default_output_path(args.input, args.scale)
            
            if output_mode == "pictures":
                log(f"[Rank 0] Saving final frames to {output_path} (image sequence mode)...", "info", rank)
            else:
                log(f"[Rank 0] Saving final video to {output_path} (streaming mode)...", "info", rank)
            
            # 使用流式保存方法（分批加载和写入）
            from save_recovered_video import save_video_streaming
            
            # 创建流式写入器
            tmp_yuv_path = None  # 在外层定义，确保在所有异常处理中可见
            try:
                import subprocess
                import tempfile
                import cv2
                
                F, H, W, C = None, None, None, None
                fps = input_fps
                
                # 先读取第一个文件获取尺寸信息
                first_rank, first_file, _ = result_files[0]
                first_result = torch.load(first_file, map_location='cpu')
                F, H, W, C = first_result.shape
                log(f"[Rank 0] Video dimensions: {F} frames (first rank), {H}x{W}x{C}, {fps} fps", "info", rank)
                del first_result
                gc.collect()
                
                # 根据输出模式选择不同的处理方式
                if output_mode == "pictures":
                    # 序列帧模式：直接输出为图像序列，无需临时文件
                    log(f"[Rank 0] Output mode: image sequence (pictures)", "info", rank)
                    log(f"[Rank 0] Creating output directory: {output_path}", "info", rank)
                    
                    # 确保输出目录存在
                    os.makedirs(output_path, exist_ok=True)
                    
                    total_frames_written = 0
                    last_frame = None
                    
                    # 流式处理每个 rank 的结果，直接保存为图像序列
                    log(f"[Rank 0] [Streaming] Processing {len(result_files)} rank results and saving as image sequence (CPU memory only)...", "info", rank)
                    for rank_idx, (r, result_file, file_size_mb) in enumerate(result_files):
                        log(f"[Rank 0] [Streaming] [{rank_idx + 1}/{len(result_files)}] Processing Rank {r}...", "info", rank)
                        log(f"[Rank 0] [Streaming]   - File: {result_file}", "info", rank)
                        log(f"[Rank 0] [Streaming]   - File size: {file_size_mb:.2f} MB ({file_size_mb/1024:.2f} GB)", "info", rank)
                        
                        # 加载当前 rank 的结果（使用 CPU 内存，不是显存）
                        log(f"[Rank 0] [Streaming]   - Loading to CPU memory...", "info", rank)
                        segment = torch.load(result_file, map_location='cpu')
                        segment_frames = segment.shape[0]
                        log(f"[Rank 0] [Streaming]   - ✓ Loaded {segment_frames} frames, shape: {segment.shape}", "success", rank)
                        
                        # 转换为 numpy 并逐帧保存
                        log(f"[Rank 0] [Streaming]   - Converting to numpy and saving as images...", "info", rank)
                        segment_np = (segment.clamp(0, 1) * 255).byte().cpu().numpy().astype('uint8')
                        
                        frames_in_rank = 0
                        for frame_idx in range(segment_np.shape[0]):
                            frame = segment_np[frame_idx]  # [H, W, C]
                            # 保存为 PNG 格式（支持无损压缩）
                            frame_filename = os.path.join(output_path, f"frame_{total_frames_written:06d}.png")
                            cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                            frames_in_rank += 1
                            total_frames_written += 1
                            
                            # 每处理50帧打印一次进度
                            if frames_in_rank % 50 == 0 or frames_in_rank == segment_np.shape[0]:
                                log(f"[Rank 0] [Streaming]   - Saved {frames_in_rank}/{segment_np.shape[0]} frames from Rank {r} (total: {total_frames_written} frames)", "info", rank)
                        
                        # 保存最后一帧（用于可能的填充）
                        last_frame = segment[-1:, :, :, :]
                        
                        # 释放内存
                        del segment, segment_np
                        gc.collect()
                        
                        log(f"[Rank 0] [Streaming]   - ✓ Rank {r} completed. Total frames saved: {total_frames_written}", "success", rank)
                    
                    # 检查是否需要填充
                    if total_frames_written < total_frames:
                        missing = total_frames - total_frames_written
                        log(f"[Rank 0] Padding {missing} frames using the last frame...", "info", rank)
                        if last_frame is not None:
                            padding_np = (last_frame.clamp(0, 1) * 255).byte().cpu().numpy().astype('uint8')[0]  # [H, W, C]
                            for i in range(missing):
                                frame_filename = os.path.join(output_path, f"frame_{total_frames_written:06d}.png")
                                cv2.imwrite(frame_filename, cv2.cvtColor(padding_np, cv2.COLOR_RGB2BGR))
                                total_frames_written += 1
                                if (i + 1) % 10 == 0:
                                    log(f"[Rank 0]   Padded {i + 1}/{missing} frames...", "info", rank)
                        log(f"[Rank 0] ✓ Padded to {total_frames_written} frames (target: {total_frames})", "info", rank)
                    elif total_frames_written > total_frames:
                        log(f"[Rank 0] WARNING: Saved {total_frames_written} frames > target {total_frames}. May have extra frames.", "warning", rank)
                    
                    log(f"[Rank 0] ✓ Image sequence saved successfully: {output_path}", "finish", rank)
                    log(f"[Rank 0]   - Total frames: {total_frames_written}", "finish", rank)
                    log(f"[Rank 0]   - Image dimensions: {H}x{W}x{C}", "finish", rank)
                    log(f"[Rank 0]   - Frame format: PNG (frame_XXXXXX.png)", "finish", rank)
                    
                else:
                    # 视频模式：使用临时文件方式（更稳定，避免 BrokenPipeError，特别适合16K视频）
                    log(f"[Rank 0] Output mode: video file", "info", rank)
                    log(f"[Rank 0] Using temp file method (more stable for 16K videos, avoids BrokenPipeError)", "info", rank)
                    try:
                        # 创建临时文件
                        tmp_yuv = tempfile.NamedTemporaryFile(suffix='.yuv', delete=False)
                        tmp_yuv_path = tmp_yuv.name
                        log(f"[Rank 0] Temporary file: {tmp_yuv_path}", "info", rank)
                        
                        total_frames_written = 0
                        last_frame = None
                        
                        # 流式处理每个 rank 的结果，写入临时文件
                        log(f"[Rank 0] [Streaming] Processing {len(result_files)} rank results and writing to temp file (CPU memory only)...", "info", rank)
                        for rank_idx, (r, result_file, file_size_mb) in enumerate(result_files):
                            log(f"[Rank 0] [Streaming] [{rank_idx + 1}/{len(result_files)}] Processing Rank {r}...", "info", rank)
                            log(f"[Rank 0] [Streaming]   - File: {result_file}", "info", rank)
                            log(f"[Rank 0] [Streaming]   - File size: {file_size_mb:.2f} MB ({file_size_mb/1024:.2f} GB)", "info", rank)
                            
                            # 加载当前 rank 的结果（使用 CPU 内存，不是显存）
                            log(f"[Rank 0] [Streaming]   - Loading to CPU memory...", "info", rank)
                            segment = torch.load(result_file, map_location='cpu')
                            segment_frames = segment.shape[0]
                            log(f"[Rank 0] [Streaming]   - ✓ Loaded {segment_frames} frames, shape: {segment.shape}", "success", rank)
                            
                            # 转换为 numpy 并分批写入临时文件
                            log(f"[Rank 0] [Streaming]   - Converting to numpy and writing to temp file...", "info", rank)
                            segment_np = (segment.clamp(0, 1) * 255).byte().cpu().numpy().astype('uint8')
                            batch_size = 50  # 临时文件可以写入更多帧
                            
                            frames_in_rank = 0
                            for i in range(0, segment_np.shape[0], batch_size):
                                end_idx = min(i + batch_size, segment_np.shape[0])
                                batch = segment_np[i:end_idx]
                                tmp_yuv.write(batch.tobytes())
                                frames_in_rank += batch.shape[0]
                                total_frames_written += batch.shape[0]
                                
                                # 每处理100帧打印一次进度
                                if frames_in_rank % 100 == 0 or frames_in_rank == segment_np.shape[0]:
                                    log(f"[Rank 0] [Streaming]   - Written {frames_in_rank}/{segment_np.shape[0]} frames from Rank {r} to temp file (total: {total_frames_written} frames)", "info", rank)
                            
                            # 保存最后一帧（用于可能的填充）
                            last_frame = segment[-1:, :, :, :]
                            
                            # 释放内存
                            del segment, segment_np
                            gc.collect()
                            
                            log(f"[Rank 0] [Streaming]   - ✓ Rank {r} completed. Total frames in temp file: {total_frames_written}", "success", rank)
                    
                        # 检查是否需要填充
                        if total_frames_written < total_frames:
                            missing = total_frames - total_frames_written
                            log(f"[Rank 0] Padding {missing} frames using the last frame...", "info", rank)
                            if last_frame is not None:
                                padding_np = (last_frame.clamp(0, 1) * 255).byte().cpu().numpy().astype('uint8')
                                for i in range(missing):
                                    tmp_yuv.write(padding_np[0].tobytes())
                                    total_frames_written += 1
                                    if (i + 1) % 10 == 0:
                                        log(f"[Rank 0]   Padded {i + 1}/{missing} frames...", "info", rank)
                            log(f"[Rank 0] ✓ Padded to {total_frames_written} frames (target: {total_frames})", "info", rank)
                        elif total_frames_written > total_frames:
                            log(f"[Rank 0] WARNING: Written {total_frames_written} frames > target {total_frames}. Video may have extra frames.", "warning", rank)
                        
                        # 关闭临时文件
                        tmp_yuv.close()
                        tmp_yuv_size_mb = os.path.getsize(tmp_yuv_path) / (1024**2)
                        log(f"[Rank 0] ✓ Temp file created: {tmp_yuv_size_mb:.2f} MB ({tmp_yuv_size_mb/1024:.2f} GB)", "success", rank)
                        
                        # 使用 FFmpeg 从临时文件编码
                        log(f"[Rank 0] Starting FFmpeg encoding from temp file...", "info", rank)
                        import time
                        encoding_start = time.time()
                        
                        cmd = [
                            'ffmpeg', '-y',
                            '-f', 'rawvideo',
                            '-vcodec', 'rawvideo',
                            '-s', f'{W}x{H}',
                            '-pix_fmt', 'rgb24',
                            '-r', str(fps),
                            '-i', tmp_yuv_path,
                            '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p',
                            '-movflags', '+faststart',
                            '-crf', '18',
                            output_path
                        ]
                        
                        result = subprocess.run(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            check=False
                        )
                        
                        encoding_time = time.time() - encoding_start
                        
                        encoding_success = False
                        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                            output_size_mb = os.path.getsize(output_path) / (1024**2)
                            log(f"[Rank 0] ✓ Video saved successfully: {output_path}", "finish", rank)
                            log(f"[Rank 0]   - Output size: {output_size_mb:.2f} MB ({output_size_mb/1024:.2f} GB)", "finish", rank)
                            log(f"[Rank 0]   - Total frames: {total_frames_written}", "finish", rank)
                            log(f"[Rank 0]   - Encoding time: {int(encoding_time)}s", "finish", rank)
                            encoding_success = True
                        else:
                            stderr = result.stderr.decode('utf-8', errors='ignore')
                            log(f"[Rank 0] ERROR: FFmpeg failed with return code {result.returncode}", "error", rank)
                            log(f"[Rank 0] FFmpeg stderr: {stderr[:1000]}", "error", rank)
                            log(f"[Rank 0] WARNING: Temporary file preserved for debugging/retry: {tmp_yuv_path}", "warning", rank)
                            log(f"[Rank 0] You can manually retry FFmpeg encoding with:", "warning", rank)
                            log(f"[Rank 0]   ffmpeg -y -f rawvideo -vcodec rawvideo -s {W}x{H} -pix_fmt rgb24 -r {fps} -i {tmp_yuv_path} -c:v libx264 -pix_fmt yuv420p -movflags +faststart -crf 18 {output_path}", "warning", rank)
                            raise RuntimeError(f"FFmpeg failed with return code {result.returncode}: {stderr[:500]}")
                        
                        # 只在成功时清理临时文件
                        if encoding_success and tmp_yuv_path and os.path.exists(tmp_yuv_path):
                            try:
                                os.unlink(tmp_yuv_path)
                                log(f"[Rank 0] Cleaned up temporary file", "info", rank)
                            except Exception as e:
                                log(f"[Rank 0] Warning: Failed to delete temp file {tmp_yuv_path}: {e}", "warning", rank)
                                
                    except Exception as e:
                        # 如果发生异常，保留临时文件以便调试
                        if tmp_yuv_path and os.path.exists(tmp_yuv_path):
                            log(f"[Rank 0] ERROR: Exception occurred, temporary file preserved: {tmp_yuv_path}", "error", rank)
                            log(f"[Rank 0] You can manually retry FFmpeg encoding with the preserved temp file", "error", rank)
                        raise
                        
            except Exception as e:
                log(f"[Rank 0] ERROR: Streaming save failed: {e}", "error", rank)
                # 如果外层异常，也保留临时文件
                if tmp_yuv_path and os.path.exists(tmp_yuv_path):
                    log(f"[Rank 0] WARNING: Temporary file preserved due to error: {tmp_yuv_path}", "warning", rank)
                log(f"[Rank 0] Falling back to standard save method (may use more memory)...", "warning", rank)
                
                # 回退到标准方法（如果流式失败）
                results = []
                for r, result_file, _ in result_files:
                    result = torch.load(result_file, map_location='cpu')
                    results.append((r, result))
                
                results.sort(key=lambda x: x[0])
                merged = torch.cat([r[1] for r in results], dim=0)
                
                if merged.shape[0] != total_frames:
                    if merged.shape[0] < total_frames:
                        missing = total_frames - merged.shape[0]
                        last_frame = merged[-1:, :, :, :]
                        padding = last_frame.repeat(missing, 1, 1, 1)
                        merged = torch.cat([merged, padding], dim=0)
                    else:
                        merged = merged[:total_frames]
                
                save_video(merged, output_path, input_fps)
            output_size_mb = os.path.getsize(output_path) / (1024**2) if os.path.exists(output_path) else 0
            log(f"[Rank 0] ✓ Final video saved: {output_path} ({output_size_mb:.2f} MB)", "finish", rank)
            log(f"[Rank 0] ==============================================", "finish", rank)
            
            # 清理临时文件
            try:
                shutil.rmtree(checkpoint_dir)
                log(f"[Rank 0] Cleaned up temporary files", "info", rank)
            except:
                pass
        
    except Exception as e:
        # 记录错误，但不阻止其他 rank 继续运行
        log(f"[Rank {rank}] FATAL ERROR: {e}", "error", rank)
        import traceback
        log(f"[Rank {rank}] Traceback: {traceback.format_exc()}", "error", rank)
        # 尝试保存错误信息
        try:
            video_dir_name = f"{hashlib.md5(args.input.encode()).hexdigest()}_{args.scale}x"
            temp_base = getattr(args, 'temp_base', '/app/output')
            checkpoint_dir = os.path.join(temp_base, 'flashvsr_distributed', video_dir_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            error_file = os.path.join(checkpoint_dir, f"rank_{rank}_fatal_error.txt")
            with open(error_file, 'w') as f:
                f.write(f"Rank {rank} fatal error:\n{str(e)}\n\n{traceback.format_exc()}")
        except:
            pass
    finally:
        # 确保清理分布式环境（即使出错也要清理）
        try:
            dist.destroy_process_group()
        except:
            pass

# ==============================================================
#              主入口函数
# ==============================================================

def parse_devices(devices_str: str, total_gpus: int) -> List[int]:
    """解析设备字符串，返回设备索引列表。
    
    Args:
        devices_str: 设备字符串，支持：
            - "all": 使用所有GPU
            - "0,1,2": 使用指定的GPU索引（逗号分隔）
            - "0-2": 使用范围（支持，但当前不实现）
        total_gpus: 系统中可用的GPU总数
    
    Returns:
        设备索引列表，例如 [0, 1, 2]
    
    Raises:
        ValueError: 如果设备字符串格式无效或索引超出范围
    """
    if devices_str is None or devices_str.strip().lower() == "all":
        return list(range(total_gpus))
    
    devices_str = devices_str.strip()
    device_indices = []
    
    # 解析逗号分隔的设备索引
    for part in devices_str.split(','):
        part = part.strip()
        if not part:
            continue
        
        # 检查是否是范围格式（如 "0-2"）
        if '-' in part:
            try:
                start, end = part.split('-', 1)
                start_idx = int(start.strip())
                end_idx = int(end.strip())
                if start_idx < 0 or end_idx >= total_gpus or start_idx > end_idx:
                    raise ValueError(f"Invalid device range: {part}")
                device_indices.extend(range(start_idx, end_idx + 1))
            except ValueError as e:
                raise ValueError(f"Invalid device range format: {part}. Error: {e}")
        else:
            # 单个设备索引
            try:
                idx = int(part)
                if idx < 0 or idx >= total_gpus:
                    raise ValueError(f"Device index {idx} is out of range (0-{total_gpus-1})")
                device_indices.append(idx)
            except ValueError as e:
                raise ValueError(f"Invalid device index: {part}. Error: {e}")
    
    # 去重并排序
    device_indices = sorted(list(set(device_indices)))
    
    if not device_indices:
        raise ValueError("No valid devices specified")
    
    return device_indices

def get_temp_dir(args) -> str:
    """获取临时目录路径（优先使用用户指定或输出目录，避免 /tmp 空间不足）。"""
    if args.temp_dir:
        temp_base = args.temp_dir
    elif args.output:
        # 使用输出目录的父目录
        temp_base = os.path.dirname(os.path.abspath(args.output))
        if not temp_base or temp_base == '/':
            temp_base = '/app/output'  # 回退到 /app/output
    else:
        # 默认使用 /app/output（通常有更多空间）
        temp_base = '/app/output'
    
    # 确保目录存在
    os.makedirs(temp_base, exist_ok=True)
    return temp_base

def main(args):
    """主函数：启动分布式推理。"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for distributed inference!")
    
    total_gpus = torch.cuda.device_count()
    if total_gpus < 1:
        raise RuntimeError(f"No CUDA devices found!")
    
    # 解析设备列表
    devices_str = getattr(args, 'devices', None)
    try:
        device_indices = parse_devices(devices_str, total_gpus)
    except ValueError as e:
        raise RuntimeError(f"Invalid --devices parameter: {e}")
    
    world_size = len(device_indices)
    if world_size < 1:
        raise RuntimeError(f"At least 1 GPU is required, but {world_size} devices specified")
    
    log(f"[Main] Starting distributed inference", "info")
    log(f"[Main] Total available GPUs: {total_gpus}", "info")
    log(f"[Main] Selected GPUs: {device_indices} (using {world_size} GPUs)", "info")
    
    # 设置临时目录（避免 /tmp 空间不足）
    args.temp_base = get_temp_dir(args)
    log(f"[Main] Using temporary directory: {args.temp_base}", "info")
    
    # 获取总帧数（不加载所有帧到内存，避免序列化问题）
    log(f"[Main] Getting frame count from input: {args.input}", "info")
    total_frames = get_total_frame_count(args.input)
    log(f"[Main] Total frames in input: {total_frames}", "info")
    
    # 应用 max_frames 限制（用于测试）
    max_frames = getattr(args, 'max_frames', None)
    if max_frames and max_frames > 0:
        if max_frames < total_frames:
            log(f"[Main] Limiting processing to first {max_frames} frames (for testing)", "info")
            total_frames = max_frames
        else:
            log(f"[Main] max_frames ({max_frames}) >= total frames ({total_frames}), processing all frames", "info")
    
    # 获取 FPS（如果是视频文件，需要读取一次；如果是图片序列，使用参数）
    input_fps = getattr(args, 'fps', 30.0)
    if os.path.isfile(args.input):
        # 视频文件：读取一次获取 FPS（只读取第一帧）
        cap = cv2.VideoCapture(args.input)
        if cap.isOpened():
            input_fps = cap.get(cv2.CAP_PROP_FPS)
            if input_fps <= 0 or input_fps > 1000:
                input_fps = 30.0
            cap.release()
    
    log(f"[Main] Input FPS: {input_fps}", "info")
    
    # 确定实际使用的进程数
    segment_overlap = getattr(args, 'segment_overlap', 2)
    
    # 使用所有选定的设备，允许动态调整（如果视频太短）
    force_num_workers = False
    log(f"[Main] Using all {world_size} selected GPUs (will auto-adjust if video is too short)", "info")
    
    # 使用确定的进程数和强制标志计算segments
    segments = split_video_by_frames(total_frames, world_size, overlap=segment_overlap, force_num_workers=force_num_workers)
    
    # 显示分配计划
    log(f"[Main] ========== Distributed Processing Plan ==========", "info")
    log(f"[Main] Total available GPUs: {total_gpus}", "info")
    log(f"[Main] Selected GPU indices: {device_indices}", "info")
    log(f"[Main] Processes to launch: {world_size}", "info")
    log(f"[Main] Total frames to process: {total_frames}", "info")
    log(f"[Main] Segment overlap: {segment_overlap} frames", "info")
    log(f"[Main] Number of segments: {len(segments)}", "info")
    log(f"[Main] Frame allocation per rank:", "info")
    total_expected_output = 0
    for i, (start, end) in enumerate(segments):
        frames_read = end - start
        expected_output = frames_read
        if i > 0:
            expected_output -= segment_overlap  # 去掉前面的overlap
        if i < len(segments) - 1:
            expected_output -= segment_overlap  # 去掉后面的overlap
        total_expected_output += max(0, expected_output)
        log(f"[Main]   Rank {i}: frames {start}-{end} (读取{frames_read}帧, 期望输出{max(0, expected_output)}帧)", "info")
    log(f"[Main] Expected total output after overlap removal: {total_expected_output} frames", "info")
    if total_expected_output != total_frames:
        log(f"[Main] WARNING: Expected output ({total_expected_output}) != input frames ({total_frames}). Merge logic will handle this.", "warning")
    if world_size < len(device_indices):
        log(f"[Main]   Note: Some selected GPUs may be skipped if video is too short", "info")
    log(f"[Main] =================================================", "info")
    
    # 设置分布式参数（自动检测可用端口）
    args.master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    base_port = int(os.environ.get('MASTER_PORT', 29500))
    
    # 尝试找到可用端口
    import socket
    port = base_port
    max_port_attempts = 10
    for attempt in range(max_port_attempts):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(1)
            sock.bind((args.master_addr, port))
            sock.close()
            # 端口可用
            break
        except OSError as e:
            if e.errno == 98 or 'Address already in use' in str(e):  # EADDRINUSE
                port = base_port + attempt + 1
                if attempt < max_port_attempts - 1:
                    log(f"[Main] Port {base_port + attempt} is in use, trying {port}...", "warning")
            else:
                port = base_port + attempt + 1
                if attempt < max_port_attempts - 1:
                    log(f"[Main] Port check failed: {e}, trying {port}...", "warning")
        except Exception as e:
            port = base_port + attempt + 1
            if attempt < max_port_attempts - 1:
                log(f"[Main] Port check failed: {e}, trying {port}...", "warning")
    else:
        # 如果所有端口都不可用，使用最后一个尝试的端口（可能会失败，但至少会给出明确的错误）
        log(f"[Main] WARNING: Could not find available port after {max_port_attempts} attempts, using {port}", "warning")
    
    if port != base_port:
        log(f"[Main] Using port {port} instead of {base_port}", "info")
    args.master_port = port
    
    # 单 GPU 模式：直接运行，不需要分布式初始化
    if world_size == 1:
        log(f"[Main] Single GPU mode detected, using simplified inference path", "info")
        device_id = device_indices[0]
        run_single_gpu_inference(args, total_frames, input_fps, device_id)
    else:
        # 多 GPU 分布式模式
        log(f"[Main] Launching {world_size} distributed processes on port {port}...", "info")
        log(f"[Main] Device mapping: Rank -> GPU", "info")
        for rank in range(world_size):
            log(f"[Main]   Rank {rank} -> GPU {device_indices[rank]}", "info")
        
        mp.spawn(
            run_with_device,
            args=(world_size, args, total_frames, input_fps, device_indices),
            nprocs=world_size,
            join=True
        )
    
    log(f"[Main] Distributed inference completed", "finish")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashVSR Distributed Inference - 真正的分布式/模型并行版本")
    parser.add_argument("--input", type=str, required=True, 
                       help="Input path: video file (e.g., video.mp4) or image sequence directory (e.g., /path/to/frames/)")
    parser.add_argument("--output", type=str, default=None, 
                       help="Output path: video file (if --output_mode=video) or directory (if --output_mode=pictures)")
    parser.add_argument("--output_mode", type=str, default="video", choices=["video", "pictures"],
                       help="Output mode: 'video' for video file (default), 'pictures' for image sequence")
    parser.add_argument("--fps", type=float, default=30.0,
                       help="Frames per second (used when input is image sequence, default: 30.0)")
    parser.add_argument("--model_ver", type=str, default="1.1", choices=["1.0", "1.1"], help="Model version")
    parser.add_argument("--mode", type=str, default="tiny", choices=["tiny", "full", "tiny-long"], help="Model mode")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4], help="Upscale factor")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Precision")
    parser.add_argument("--attention_mode", type=str, default="sparse_sage_attention", 
                       choices=["sparse_sage_attention", "block_sparse_attention"], help="Attention mode")
    parser.add_argument("--segment_overlap", type=int, default=2, help="Overlap frames between segments")
    
    # 推理参数（从原版复制）
    parser.add_argument("--color_fix", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True, help="Use color fix")
    parser.add_argument("--tiled_vae", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True, help="Use tiled VAE")
    parser.add_argument("--tiled_dit", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False, help="Use tiled DiT")
    parser.add_argument("--tile_size", type=int, default=256, help="Tile size")
    parser.add_argument("--tile_overlap", type=int, default=24, help="Tile overlap")
    parser.add_argument("--unload_dit", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False, help="Unload DiT before decoding")
    parser.add_argument("--sparse_ratio", type=float, default=2.0, help="Sparse ratio")
    parser.add_argument("--kv_ratio", type=float, default=3.0, help="KV ratio")
    parser.add_argument("--local_range", type=int, default=11, help="Local range")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    # 分布式参数
    parser.add_argument("--master_addr", type=str, default="localhost", help="Master address for distributed training")
    parser.add_argument("--master_port", type=int, default=29500, help="Master port for distributed training")
    parser.add_argument("--use_shared_memory", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True, 
                       help="Use shared memory (/dev/shm) for model loading (reduces memory usage)")
    parser.add_argument("--cleanup_mmap", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False,
                        help="Clean up memory-mapped canvas files after saving results (default: False, keep for recovery)")
    parser.add_argument("--tile_batch_size", type=int, default=0,
                        help="Number of tiles to process simultaneously (0 = auto-detect based on GPU memory, default: 0)")
    parser.add_argument("--adaptive_tile_batch", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True,
                        help="Enable adaptive tile batch size based on available GPU memory (default: True)")
    parser.add_argument("--temp_dir", type=str, default=None,
                       help="Temporary directory for intermediate files (default: output directory or /app/output)")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum number of frames to process (for testing, e.g., 10 or 20 frames)")
    parser.add_argument("--devices", type=str, default=None,
                       help="GPU devices to use. Options: 'all' (use all GPUs, default), or comma-separated indices like '0,1,2' or '0-2' (range). Examples: 'all', '0,1,2', '0-3', '0,2,4'")
    
    args = parser.parse_args()
    main(args)
