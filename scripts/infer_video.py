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
    python scripts/infer_video.py --input video.mp4 --output output.mp4
    
    # 图片序列：默认 PNG；10-bit DPX 用 --output_format dpx --output_bit_depth 10
    python scripts/infer_video.py --input video.mp4 --output_mode pictures --output /path/to/frames
    python scripts/infer_video.py --input video.mp4 --output_mode pictures --output /path/to/frames --output_format dpx --output_bit_depth 10
    
    # HDR 通道：--dynamic_range hdr；预处理可选 HLG（推荐）或 Tone Mapping
    python scripts/infer_video.py \
        --input hdr_dpx_sequence \
        --output_mode pictures \
        --output /path/to/output \
        --output_format dpx \
        --output_bit_depth 10 \
        --dynamic_range hdr \
        --hdr_preprocess hlg \
        --tone_mapping_method logarithmic \
        --tone_mapping_exposure 1.0
    
    # 指定使用的GPU（使用 --devices 参数）
    python scripts/infer_video.py --input video.mp4 --output output.mp4 --devices 0,1,2,3
    
    # 16K-32K 大分辨率视频（推荐使用更大的 tile_size）
    python scripts/infer_video.py \
        --input 16k_video.mp4 \
        --output 16k_2x.mp4 \
        --mode tiny \
        --scale 2 \
        --tile_size 512 \
        --tile_overlap 48 \
        --devices all
    
    # 完整参数示例（包含新功能）
    python scripts/infer_video.py \
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
    
    # 断点续跑：默认 --resume，同一命令再跑会自动从断点继续；加 --force 则覆盖原缓存从头跑
    # 已处理的 tiles 会自动跳过（--resume 时）
    
    # 检查状态和恢复（使用恢复工具，checkpoint 目录见启动日志 [Main] Checkpoint directory: ...）
    python tools/recover_distributed_inference.py \
        --checkpoint_dir /app/tmp/checkpoints/{输出名} \
        --status
    
    # 合并部分结果（即使有些 rank 失败）
    python tools/recover_distributed_inference.py \
        --checkpoint_dir /app/tmp/checkpoints/{输出名} \
        --merge_partial \
        --output output_partial.mp4 \
        --output_fps 30.0

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
- 权重/checkpoint 默认存放在 /app/tmp/checkpoints/{输出文件名或文件夹名}/（容器内，宿主机可挂载），命名随 --output 自动推断，无需额外参数；启动时在日志中输出
- 使用 --cleanup_mmap true 可以在保存结果后删除内存映射文件以节省空间

恢复工具：
- 使用 recover_distributed_inference.py 检查任务状态
- 可以合并部分结果（即使有些 rank 失败）
- 支持手动恢复失败的 rank
- checkpoint 目录示例：--output test16K-32K 且 output_mode=pictures 时，为 /app/tmp/checkpoints/test16K-32K/

HDR 支持（v3）：
- 使用 --dynamic_range hdr 或 --dynamic_range auto 启用 HDR 通道；默认 --dynamic_range sdr 为 SDR 通道
- --hdr_preprocess：HDR 预处理方式，hlg（推荐，PQ→HLG 工作流）或 tone_mapping（可逆 Tone Mapping）
- 输入支持：10-bit DPX 序列、HDR 视频（H.265/HEVC）；SDR 与 HDR 通道分离，互不粘连
- 输出：--output_format dpx --output_bit_depth 10 可输出 10-bit DPX；视频输出 FPS 由 --output_fps 指定（默认继承输入 FPS）
- Tone Mapping 工作流仅当 --hdr_preprocess tone_mapping 时生效；可选 --global_l_max 避免帧间闪烁
"""

# ==============================================================
#                         依赖导入
# ==============================================================

# -------------------- Python 标准库 --------------------
import os
import sys
import math
import argparse
import re
import time
import json
import glob
import gc
import shutil
import tempfile
import subprocess
import socket
import traceback
from typing import List, Tuple, Optional

# -------------------- 项目路径配置 --------------------
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# -------------------- 第三方库 --------------------
import cv2
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# -------------------- 环境配置 --------------------
_torch_lib_path = "/usr/local/lib/python3.10/dist-packages/torch/lib"
if os.path.exists(_torch_lib_path):
    _current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if _torch_lib_path not in _current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{_torch_lib_path}:{_current_ld_path}"

# -------------------- 推理 I/O（HDR/SDR、帧输入输出、合并保存） --------------------
from utils.io import inference_io


# ==============================================================
#                      基础工具函数
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

# 统一 inference_io 的日志输出
inference_io.log = log

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


def _spawn_target_run_with_device(
    rank: int,
    world_size: int,
    args,
    total_frames: int,
    input_fps: float,
    device_indices: List[int],
) -> None:
    """mp.spawn 的顶层目标函数（可被 pickle）。在子进程内加载 inference_runner 并调用 run_with_device。"""
    import importlib.util
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _runner_path = os.path.join(_script_dir, "inference_runner.py")
    _spec = importlib.util.spec_from_file_location("inference_runner", _runner_path)
    inference_runner = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(inference_runner)
    inference_runner.run_with_device(rank, world_size, args, total_frames, input_fps, device_indices)


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

def main(args):
    """主函数：启动分布式推理。
    
    处理流程：
        1. 检测 GPU 并解析设备参数
        2. 获取输入信息（帧数、FPS）
        3. 计算分布式任务分配
        4. 启动推理进程（单 GPU 或多 GPU 分布式）
    """
    # ==================== Step 1: 检测 GPU 并解析设备参数 ====================
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for distributed inference!")
    
    total_gpus = torch.cuda.device_count()
    if total_gpus < 1:
        raise RuntimeError(f"No CUDA devices found!")
    
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
    
    # 设置 checkpoint 目录
    args.checkpoint_dir = inference_io.get_checkpoint_dir(args)
    log(f"[Main] Checkpoint directory: {args.checkpoint_dir}", "info")
    
    # --force 或 --no-resume：清空 checkpoint，从头开始
    if getattr(args, 'force', False) or not getattr(args, 'resume', True):
        if os.path.isdir(args.checkpoint_dir):
            log(f"[Main] --force/--no-resume: clearing checkpoint directory for fresh run", "info")
            shutil.rmtree(args.checkpoint_dir, ignore_errors=True)
    else:
        log(f"[Main] --resume (default): will resume from checkpoint if present", "info")
    
    # ==================== Step 2: 获取输入信息 ====================
    log(f"[Main] Getting frame count from input: {args.input}", "info")
    dynamic_range = getattr(args, 'dynamic_range', 'sdr')
    try:
        from utils.io.hdr_io import detect_hdr_input
        enable_hdr = (dynamic_range == 'hdr') or (dynamic_range == 'auto' and detect_hdr_input(args.input, True))
    except ImportError:
        enable_hdr = (dynamic_range == 'hdr')
    args._enable_hdr = enable_hdr
    total_frames = inference_io.get_total_frame_count(args.input, enable_hdr=enable_hdr)
    log(f"[Main] Total frames in input: {total_frames}", "info")
    
    # 应用 max_frames 限制（用于测试）
    max_frames = getattr(args, 'max_frames', None)
    if max_frames and max_frames > 0:
        if max_frames < total_frames:
            log(f"[Main] Limiting processing to first {max_frames} frames (for testing)", "info")
            total_frames = max_frames
        else:
            log(f"[Main] max_frames ({max_frames}) >= total frames ({total_frames}), processing all frames", "info")
    
    # 获取输入 FPS（视频文件从文件探测，图像序列使用 --input_fps）
    input_fps = getattr(args, 'input_fps', 30.0)
    if os.path.isfile(args.input):
        cap = cv2.VideoCapture(args.input)
        if cap.isOpened():
            input_fps = cap.get(cv2.CAP_PROP_FPS)
            if input_fps <= 0 or input_fps > 1000:
                input_fps = 30.0
            cap.release()
    # 输出 FPS（仅 output_mode=video 时使用，未指定则继承输入 FPS）
    if getattr(args, 'output_mode', 'video') == 'video' and getattr(args, 'output_fps', None) is None:
        args.output_fps = input_fps
    
    log(f"[Main] Input FPS: {input_fps}", "info")
    
    # ==================== Step 3: HDR 工作流配置 ====================
    if enable_hdr:
        hdr_preprocess = getattr(args, 'hdr_preprocess', 'hlg')
        if hdr_preprocess == 'hlg':
            log(f"[Main] [HDR] ✓ 使用 HLG 工作流（推荐）", "info")
            log(f"[Main] [HDR]   - 输入转换: PQ (ST2084) → HLG (arib-std-b67)", "info")
            log(f"[Main] [HDR]   - 输出编码: HLG → PQ (通过 FFmpeg zscale)", "info")
        else:
            log(f"[Main] [HDR] 使用 Tone Mapping 工作流: {getattr(args, 'tone_mapping_method', 'logarithmic')}", "info")
    
    # HDR 预扫描（仅当 dynamic_range=hdr 且 hdr_preprocess=tone_mapping 时需要）
    if enable_hdr and getattr(args, 'hdr_preprocess', 'hlg') == 'tone_mapping' and inference_io.HDR_TONE_MAPPING_AVAILABLE:
        if getattr(args, 'global_l_max', None) is None:
            exposure = getattr(args, 'tone_mapping_exposure', 1.0)
            precomputed_l_max = inference_io.precompute_global_hdr_params(
                args.input,
                enable_hdr=True,
                exposure=exposure,
                sample_interval=10,
                max_sample_frames=500
            )
            if precomputed_l_max is not None:
                args.global_l_max = precomputed_l_max
                log(f"[Main] [HDR] 已设置全局 l_max: {args.global_l_max:.4f}（将用于所有 rank）", "info")
        else:
            log(f"[Main] [HDR] 使用用户指定的 global_l_max: {args.global_l_max}", "info")
    
    # ==================== Step 4: 计算分布式任务分配 ====================
    segment_overlap = getattr(args, 'segment_overlap', 2)
    force_num_workers = False
    log(f"[Main] Using all {world_size} selected GPUs (will auto-adjust if video is too short)", "info")
    
    segments = inference_io.split_video_by_frames(total_frames, world_size, overlap=segment_overlap, force_num_workers=force_num_workers)
    
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
    
    # ==================== Step 4: 启动推理进程 ====================
    # 设置分布式参数（自动检测可用端口）
    args.master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    base_port = int(os.environ.get('MASTER_PORT', 29500))
    
    # 尝试找到可用端口
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
    
    # 单 GPU 模式：由 inference_runner 提供 run_single_gpu_inference（late load 避免循环依赖）
    if world_size == 1:
        import importlib.util
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _runner_path = os.path.join(_script_dir, "inference_runner.py")
        _spec = importlib.util.spec_from_file_location("inference_runner", _runner_path)
        inference_runner = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(inference_runner)
        log(f"[Main] Single GPU mode detected, using simplified inference path", "info")
        device_id = device_indices[0]
        inference_runner.run_single_gpu_inference(args, total_frames, input_fps, device_id)
    else:
        # 多 GPU 分布式模式：使用顶层包装函数 _spawn_target_run_with_device（可被 pickle）
        log(f"[Main] Launching {world_size} distributed processes on port {port}...", "info")
        log(f"[Main] Device mapping: Rank -> GPU", "info")
        for rank in range(world_size):
            log(f"[Main]   Rank {rank} -> GPU {device_indices[rank]}", "info")
        
        mp.spawn(
            _spawn_target_run_with_device,
            args=(world_size, args, total_frames, input_fps, device_indices),
            nprocs=world_size,
            join=True
        )
    
    log(f"[Main] Distributed inference completed", "finish")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashVSR Distributed Inference - 真正的分布式/模型并行版本")
    
    # -------------------- 输入/输出参数 --------------------
    parser.add_argument("--input", type=str, required=True, 
                       help="Input path: video file (e.g., video.mp4) or image sequence directory (e.g., /path/to/frames/)")
    parser.add_argument("--output", type=str, default=None, 
                       help="Output path: video file (if --output_mode=video) or directory (if --output_mode=pictures)")
    parser.add_argument("--output_mode", type=str, default="video", choices=["video", "pictures"],
                       help="Output mode: 'video' for video file (default), 'pictures' for image sequence")
    parser.add_argument(
        "--output_format",
        type=str,
        default=None,
        help=(
            "Output container/format. "
            "When --output_mode=video: one of 'mp4', 'mov', 'mkv'. "
            "When --output_mode=pictures: one of 'png', 'dpx'. "
            "If not set, defaults to 'mp4' for video mode and 'png' for pictures mode."
        ),
    )
    parser.add_argument(
        "--output_bit_depth",
        type=int,
        default=8,
        choices=[8, 10],
        help="Output bit depth. 8 or 10. "
             "For video this maps to yuv420p (8-bit) or yuv420p10le (10-bit). "
             "For pictures this maps to 8-bit PNG or 10-bit DPX.",
    )
    parser.add_argument(
        "--input_fps",
        type=float,
        default=30.0,
        help="Frames per second for image sequence input. "
             "Ignored when input is a video file (FPS is probed from the file).",
    )
    parser.add_argument(
        "--output_fps",
        type=float,
        default=None,
        help="Output video FPS when --output_mode=video. "
             "If not set, inherits the input FPS.",
    )
    
    # -------------------- HDR / 动态范围参数 --------------------
    parser.add_argument(
        "--dynamic_range",
        type=str,
        default="sdr",
        choices=["sdr", "hdr", "auto"],
        help="Input dynamic range: 'sdr' (standard dynamic range), "
             "'hdr' (high dynamic range), or 'auto' (auto-detect).",
    )
    parser.add_argument(
        "--hdr_preprocess",
        type=str,
        default="hlg",
        choices=["hlg", "tone_mapping"],
        help="HDR preprocessing pipeline when --dynamic_range=hdr: "
             "'hlg' for PQ→HLG workflow (recommended), "
             "'tone_mapping' for reversible tone-mapping workflow.",
    )
    parser.add_argument(
        "--tone_mapping_method",
        type=str,
        default="logarithmic",
        choices=["reinhard", "logarithmic", "aces"],
        help="Tone mapping method when --hdr_preprocess=tone_mapping.",
    )
    parser.add_argument(
        "--tone_mapping_exposure",
        type=float,
        default=1.0,
        help="Tone mapping exposure adjustment (only used in tone-mapping workflow).",
    )
    parser.add_argument(
        "--global_l_max",
        type=float,
        default=None,
        help="Global l_max for tone mapping (optional). "
             "If provided, all frames/segments share this l_max to avoid flicker.",
    )
    parser.add_argument(
        "--hdr_transfer",
        type=str,
        default="hdr10",
        choices=["hdr10", "hlg"],
        help="HDR transfer when output_mode=video and dynamic_range=hdr: "
             "'hdr10' (PQ, 10-bit + metadata) or 'hlg' (HLG curve). Ignored for SDR or output_mode=pictures.",
    )
    
    # -------------------- 模型参数 --------------------
    parser.add_argument("--model_ver", type=str, default="1.1", choices=["1.0", "1.1"], help="Model version")
    parser.add_argument("--mode", type=str, default="tiny", choices=["tiny", "full", "tiny-long"], help="Model mode")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4], help="Upscale factor")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Precision")
    parser.add_argument("--attention_mode", type=str, default="sparse_sage_attention", 
                       choices=["sparse_sage_attention", "block_sparse_attention"], help="Attention mode")
    
    # -------------------- 推理参数 --------------------
    parser.add_argument("--color_fix", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True, help="Use color fix")
    parser.add_argument("--tiled_vae", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True, help="Use tiled VAE")
    parser.add_argument("--tiled_dit", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False, help="Use tiled DiT")
    parser.add_argument("--tile_size", type=int, default=256, help="Tile size")
    parser.add_argument("--tile_overlap", type=int, default=24, help="Tile overlap")
    parser.add_argument("--tile_shift", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True,
                        help="Use tile shift (two-pass) to reduce grid seams when tiled_dit is on")
    parser.add_argument("--unload_dit", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False, help="Unload DiT before decoding")
    parser.add_argument("--sparse_ratio", type=float, default=2.0, help="Sparse ratio")
    parser.add_argument("--kv_ratio", type=float, default=3.0, help="KV ratio")
    parser.add_argument("--local_range", type=int, default=11, help="Local range")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    # -------------------- 分布式参数 --------------------
    parser.add_argument("--devices", type=str, default=None,
                       help="GPU devices to use. Options: 'all' (default), or '0,1,2' or '0-2' (range)")
    parser.add_argument("--segment_overlap", type=int, default=2, help="Overlap frames between segments")
    parser.add_argument("--master_addr", type=str, default="localhost", help="Master address for distributed training")
    parser.add_argument("--master_port", type=int, default=29500, help="Master port for distributed training")
    parser.add_argument("--use_shared_memory", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True, 
                       help="Use shared memory (/dev/shm) for model loading")
    
    # -------------------- 断点续跑参数 --------------------
    parser.add_argument("--resume", dest="resume", action="store_true",
                        help="从断点续跑（默认）；存在 checkpoint 时自动接着上次进度")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="不从断点续跑，等价于覆盖缓存从头跑")
    parser.add_argument("--force", action="store_true",
                        help="覆盖原 checkpoint 缓存，从头开始（与 --no-resume 等效）")
    parser.set_defaults(resume=True)
    
    # -------------------- 内存优化参数 --------------------
    parser.add_argument("--cleanup_mmap", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False,
                        help="Clean up memory-mapped canvas files after saving results")
    parser.add_argument("--tile_batch_size", type=int, default=0,
                        help="Number of tiles to process simultaneously (0 = auto-detect)")
    parser.add_argument("--adaptive_tile_batch", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True,
                        help="Enable adaptive tile batch size based on available GPU memory")
    
    # -------------------- 调试参数 --------------------
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum number of frames to process (for testing)")
    
    args = parser.parse_args()
    
    # 解析后：根据 output_mode 和输入类型设置 output_format 默认值并校验
    # - 当 output_mode=video 且输入为单个视频文件时：
    #   默认继承输入容器格式（mp4/mov/mkv），否则回退为 mp4
    # - 当 output_mode=pictures 时：默认 png
    output_mode = getattr(args, 'output_mode', 'video')
    output_format = getattr(args, 'output_format', None)
    if output_format is None:
        if output_mode == 'video':
            inferred_fmt = None
            # 如果输入本身是视频文件，则尝试继承其扩展名
            if os.path.isfile(args.input):
                _, ext = os.path.splitext(args.input)
                ext = ext.lower().lstrip('.')
                if ext in ('mp4', 'mov', 'mkv'):
                    inferred_fmt = ext
            args.output_format = inferred_fmt or 'mp4'
        else:
            args.output_format = 'png'
    else:
        if output_mode == 'video' and args.output_format not in ('mp4', 'mov', 'mkv'):
            args.output_format = 'mp4'
        elif output_mode == 'pictures' and args.output_format not in ('png', 'dpx', 'dpx10'):
            args.output_format = 'png'
    
    main(args)
