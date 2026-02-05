#!/usr/bin/env python3
"""
FlashVSR 分布式推理恢复工具

功能：
1. 检查哪些 rank 失败或未完成
2. 手动重新运行失败的 rank
3. 合并部分结果
4. 从断点恢复任务

使用方法：
    # 检查状态
    python recover_distributed_inference.py --checkpoint_dir /app/output/flashvsr_distributed/{video_dir_name} --status
    
    # 恢复失败的 rank
    python recover_distributed_inference.py --checkpoint_dir /app/output/flashvsr_distributed/{video_dir_name} --recover_rank 0
    
    # 合并部分结果（即使有些 rank 失败）
    python recover_distributed_inference.py --checkpoint_dir /app/output/flashvsr_distributed/{video_dir_name} --merge_partial --output output_partial.mp4
"""

import os
import sys
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

# 添加项目根目录到 sys.path，确保可以导入 utils 模块
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import cv2
import numpy as np
from typing import List, Tuple, Optional
import hashlib
import tempfile

def log(message: str, message_type: str = 'normal'):
    """Colored logging for console output (with flush for real-time output)."""
    import sys
    colors = {
        'normal': '\033[0m',
        'info': '\033[94m',
        'success': '\033[92m',
        'warning': '\033[93m',
        'error': '\033[91m',
        'finish': '\033[92m',
    }
    color = colors.get(message_type, colors['normal'])
    reset = '\033[0m'
    print(f"{color}{message}{reset}", flush=True)  # 添加 flush=True 确保实时输出

def check_status(checkpoint_dir: str):
    """检查所有 rank 的状态。"""
    if not os.path.exists(checkpoint_dir):
        log(f"ERROR: Checkpoint directory does not exist: {checkpoint_dir}", "error")
        return
    
    log(f"Checking status in: {checkpoint_dir}", "info")
    
    # 查找所有 rank 的状态
    ranks_status = {}
    for rank in range(8):  # 假设最多 8 个 rank
        done_file = os.path.join(checkpoint_dir, f"rank_{rank}_done.flag")
        result_file = os.path.join(checkpoint_dir, f"rank_{rank}_result.pt")
        error_file = os.path.join(checkpoint_dir, f"rank_{rank}_error.txt")
        progress_file = os.path.join(checkpoint_dir, f"rank_{rank}_progress.txt")
        processed_tiles_file = os.path.join(checkpoint_dir, f"rank_{rank}_processed_tiles.json")
        canvas_file = os.path.join(checkpoint_dir, f"rank_{rank}_canvas.npy")
        weight_file = os.path.join(checkpoint_dir, f"rank_{rank}_weight.npy")
        
        status = {
            'done': os.path.exists(done_file),
            'result': os.path.exists(result_file),
            'error': os.path.exists(error_file),
            'progress': None,
            'processed_tiles': None,
            'canvas': os.path.exists(canvas_file),
            'weight': os.path.exists(weight_file),
        }
        
        # 读取 done 状态
        if status['done']:
            try:
                with open(done_file, 'r') as f:
                    content = f.read().strip()
                    if 'completed' in content:
                        status['done_status'] = 'completed'
                    elif 'failed' in content:
                        status['done_status'] = 'failed'
                    elif 'skipped' in content:
                        status['done_status'] = 'skipped'
            except:
                pass
        
        # 读取进度
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    lines = f.read().strip().split('\n')
                    if len(lines) >= 3:
                        status['progress'] = {
                            'processed': int(lines[0]),
                            'total': int(lines[1]),
                            'percentage': float(lines[2])
                        }
            except:
                pass
        
        # 读取已处理的 tiles
        if os.path.exists(processed_tiles_file):
            try:
                with open(processed_tiles_file, 'r') as f:
                    status['processed_tiles'] = len(json.load(f))
            except:
                pass
        
        # 读取错误信息
        if status['error']:
            try:
                with open(error_file, 'r') as f:
                    status['error_msg'] = f.read()[:500]  # 只显示前500字符
            except:
                pass
        
        # 读取结果文件大小
        if status['result']:
            try:
                status['result_size_mb'] = os.path.getsize(result_file) / (1024**2)
                result = torch.load(result_file, map_location='cpu')
                status['result_shape'] = list(result.shape)
            except Exception as e:
                status['result_error'] = str(e)
        
        if any([status['done'], status['result'], status['error'], status['canvas']]):
            ranks_status[rank] = status
    
    # 显示状态
    log(f"\n{'='*60}", "info")
    log(f"Status Summary:", "info")
    log(f"{'='*60}", "info")
    
    for rank, status in sorted(ranks_status.items()):
        log(f"\nRank {rank}:", "info")
        if status['done']:
            log(f"  Status: {status.get('done_status', 'unknown')}", "info")
        else:
            log(f"  Status: In progress or not started", "warning")
        
        if status['result']:
            log(f"  Result: ✓ {status.get('result_size_mb', 0):.2f} MB, shape: {status.get('result_shape', 'unknown')}", "success")
        else:
            log(f"  Result: ✗ Not found", "error")
        
        if status['progress']:
            p = status['progress']
            log(f"  Progress: {p['processed']}/{p['total']} tiles ({p['percentage']:.1f}%)", "info")
        
        if status['processed_tiles'] is not None:
            log(f"  Processed tiles: {status['processed_tiles']}", "info")
        
        if status['canvas']:
            canvas_size_mb = os.path.getsize(os.path.join(checkpoint_dir, f"rank_{rank}_canvas.npy")) / (1024**2)
            log(f"  Canvas (mmap): ✓ {canvas_size_mb:.2f} MB", "info")
        
        if status['error']:
            log(f"  Error: {status.get('error_msg', 'Unknown error')[:200]}...", "error")

def merge_partial_results(
    checkpoint_dir: str,
    output_path: str,
    input_fps: float = 30.0,
    world_size: Optional[int] = None,
    total_frames: Optional[int] = None,
    output_mode: str = "video",
    output_format: str = "png",
    output_frame_prefix: Optional[str] = None,
    output_frame_digits: int = 6,
    output_workers: int = 0,
):
    """流式合并部分结果（委托 utils.io.streaming_merge_output，与正常跑共用同一实现）。"""
    from utils.io import streaming_merge_output

    def _log_fn(msg: str, msg_type: str) -> None:
        log(msg, msg_type)

    streaming_merge_output.streaming_merge_from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        output_path=output_path,
        input_fps=input_fps,
        world_size=world_size,
        total_frames=total_frames,
        output_mode=output_mode,
        output_format=output_format,
        output_frame_prefix=output_frame_prefix,
        output_frame_digits=output_frame_digits,
        output_workers=output_workers,
        result_files=None,
        input_filenames=None,
        enable_hdr=False,
        global_l_max=None,
        log_fn=_log_fn,
    )

def recover_rank(checkpoint_dir: str, rank: int, args_dict: dict):
    """恢复单个 rank 的处理（需要手动提供参数）。"""
    log(f"Recovering Rank {rank}...", "info")
    log(f"NOTE: This requires re-running the inference for this rank.", "warning")
    log(f"Please use the original command with --resume_from_checkpoint {checkpoint_dir}", "info")
    log(f"Or manually re-run the inference for rank {rank} with the same parameters.", "info")

def main():
    parser = argparse.ArgumentParser(description="FlashVSR 分布式推理恢复工具")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Checkpoint directory (e.g., /app/output/flashvsr_distributed/{video_dir_name})")
    parser.add_argument("--status", action="store_true",
                        help="Check status of all ranks")
    parser.add_argument("--merge_partial", action="store_true",
                        help="Merge partial results (even if some ranks failed)")
    parser.add_argument("--output", type=str,
                        help="Output video path (required for --merge_partial)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="FPS for output video (default: 30.0)")
    parser.add_argument("--world_size", type=int, default=None,
                        help="Number of ranks (default: auto-detect from checkpoint dir)")
    parser.add_argument("--total_frames", type=int, default=None,
                        help="Expected total frames (for validation and padding/cropping)")
    parser.add_argument("--recover_rank", type=int,
                        help="Recover a specific rank (shows instructions)")
    parser.add_argument("--output_mode", type=str, default="video", choices=["video", "pictures"],
                        help="Output mode: 'video' for video file (default), 'pictures' for image sequence")
    parser.add_argument("--output_format", type=str, default="png", choices=["png", "dpx10"],
                        help="When output_mode=pictures: 'png' (default) or 'dpx10' (10-bit DPX). Ignored when output_mode=video.")
    parser.add_argument("--output_frame_prefix", type=str, default=None,
                        help="When output_mode=pictures: output frame name prefix, e.g. H001_11261139_C001 -> H001_11261139_C001.00000000.png. If not set, use frame_000000.png.")
    parser.add_argument("--output_frame_digits", type=int, default=6,
                        help="When output_mode=pictures: number of digits for frame index (default: 6). Use 8 for e.g. .02525084.png.")
    parser.add_argument("--workers", type=int, default=0,
                        help="When output_mode=pictures: number of threads for writing frames (default: 0 = auto = min(cpu_count, 32)). Use 1 for single-threaded.")
    
    args = parser.parse_args()
    
    if args.status:
        check_status(args.checkpoint_dir)
    elif args.merge_partial:
        if not args.output:
            log("ERROR: --output is required for --merge_partial", "error")
            return
        merge_partial_results(
            args.checkpoint_dir,
            args.output,
            args.fps,
            world_size=args.world_size,
            total_frames=args.total_frames,
            output_mode=args.output_mode,
            output_format=args.output_format,
            output_frame_prefix=args.output_frame_prefix,
            output_frame_digits=args.output_frame_digits,
            output_workers=args.workers,
        )
    elif args.recover_rank is not None:
        recover_rank(args.checkpoint_dir, args.recover_rank, vars(args))
    else:
        log("ERROR: Please specify --status, --merge_partial, or --recover_rank", "error")
        parser.print_help()

if __name__ == "__main__":
    main()
