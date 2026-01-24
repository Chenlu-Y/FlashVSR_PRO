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
import torch
import cv2
from typing import List, Tuple, Optional
import hashlib
import tempfile

# 将项目根目录添加到 sys.path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

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

def merge_partial_results(checkpoint_dir: str, output_path: str, input_fps: float = 30.0, world_size: int = 8, total_frames: int = None, output_mode: str = "video"):
    """流式合并部分结果（即使有些 rank 失败）。
    
    使用流式合并策略，逐个加载 rank 结果并直接写入视频，避免一次性加载所有到内存。
    这对于大分辨率视频（16K-32K）和2T内存限制非常重要。
    
    内存使用：
    - 只使用 CPU 内存（map_location='cpu'），不使用 GPU 显存
    - 每次只加载一个 rank 的结果（~60GB），处理完立即释放
    - 峰值内存：单个 rank 结果 + 转换后的 numpy 数组 ≈ 120GB
    """
    import subprocess
    import gc
    
    log(f"========== Starting Streaming Merge ==========", "info")
    log(f"Checkpoint directory: {checkpoint_dir}", "info")
    log(f"Output path: {output_path}", "info")
    log(f"Output mode: {output_mode} ({'video file' if output_mode == 'video' else 'image sequence'})", "info")
    log(f"Expected FPS: {input_fps}", "info")
    log(f"Expected total frames: {total_frames if total_frames else 'auto-detect'}", "info")
    log(f"World size: {world_size}", "info")
    log(f"Using STREAMING merge (memory-efficient, CPU only)", "info")
    log(f"=============================================", "info")
    
    # 收集所有有效的结果文件信息（不加载数据）
    result_files = []
    total_size_mb = 0
    for rank in range(world_size):
        result_file = os.path.join(checkpoint_dir, f"rank_{rank}_result.pt")
        if os.path.exists(result_file):
            try:
                file_size_mb = os.path.getsize(result_file) / (1024**2)
                total_size_mb += file_size_mb
                result_files.append((rank, result_file, file_size_mb))
                log(f"  Rank {rank}: ✓ Result file found, {file_size_mb:.2f} MB", "success")
            except Exception as e:
                log(f"  Rank {rank}: ✗ Error checking file: {e}", "error")
        else:
            log(f"  Rank {rank}: ✗ Result file not found", "warning")
    
    if not result_files:
        log("ERROR: No valid results to merge!", "error")
        return
    
    # 按 rank 顺序排序
    result_files.sort(key=lambda x: x[0])
    rank_order = [r[0] for r in result_files]
    log(f"\n[Merge] Found {len(result_files)} rank results in order: {rank_order}", "info")
    log(f"[Merge] Total result files size: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)", "info")
    log(f"[Merge] Using streaming merge to avoid OOM (peak memory: ~120GB per rank)", "info")
    
    # 流式合并：逐个加载、处理、写入视频
    log(f"\n[Merge] Starting streaming merge and video encoding...", "info")
    tmp_yuv_path = None  # 在外层定义，确保在所有异常处理中可见
    try:
        F, H, W, C = None, None, None, None
        
        # 先读取第一个文件获取尺寸信息（只加载一次，获取后立即释放）
        first_rank, first_file, first_size_mb = result_files[0]
        log(f"[Merge] [Step 1/3] Reading first rank ({first_rank}) to get video dimensions...", "info")
        first_result = torch.load(first_file, map_location='cpu')  # 使用 CPU 内存，不是显存
        F, H, W, C = first_result.shape
        log(f"[Merge] [Step 1/3] ✓ Video dimensions: {F} frames (first rank), {H}x{W}x{C}, {input_fps} fps", "success")
        log(f"[Merge] [Step 1/3] First rank file size: {first_size_mb:.2f} MB", "info")
        del first_result
        gc.collect()
        
        # 根据输出模式选择不同的处理方式
        if output_mode == "pictures":
            # 序列帧模式：直接输出为图像序列，无需临时文件
            log(f"[Merge] [Step 2/2] Output mode: image sequence (pictures)", "info")
            log(f"[Merge] Creating output directory: {output_path}", "info")
            
            # 确保输出目录存在
            os.makedirs(output_path, exist_ok=True)
            
            total_frames_written = 0
            last_frame = None
            
            # 流式处理每个 rank 的结果，直接保存为图像序列
            log(f"[Merge] [Step 2/2] Processing {len(result_files)} rank results and saving as image sequence...", "info")
            for rank_idx, (r, result_file, file_size_mb) in enumerate(result_files):
                log(f"[Merge]   [{rank_idx + 1}/{len(result_files)}] Processing Rank {r}...", "info")
                log(f"[Merge]     - Loading from: {result_file}", "info")
                log(f"[Merge]     - File size: {file_size_mb:.2f} MB ({file_size_mb/1024:.2f} GB)", "info")
                
                # 加载当前 rank 的结果（使用 CPU 内存）
                segment = torch.load(result_file, map_location='cpu')
                segment_frames = segment.shape[0]
                log(f"[Merge]     - ✓ Loaded {segment_frames} frames, shape: {segment.shape}", "success")
                
                # 转换为 numpy 并逐帧保存
                log(f"[Merge]     - Converting to numpy and saving as images...", "info")
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
                        log(f"[Merge]     - Saved {frames_in_rank}/{segment_np.shape[0]} frames from Rank {r} (total: {total_frames_written} frames)", "info")
                
                # 保存最后一帧（用于可能的填充）
                last_frame = segment[-1:, :, :, :]
                
                # 释放内存
                del segment, segment_np
                gc.collect()
                
                log(f"[Merge]     - ✓ Rank {r} completed. Total frames saved: {total_frames_written}", "success")
            
            # 检查是否需要填充
            if total_frames is not None and total_frames_written < total_frames:
                missing = total_frames - total_frames_written
                log(f"[Merge] Padding {missing} frames using the last frame...", "info")
                if last_frame is not None:
                    padding_np = (last_frame.clamp(0, 1) * 255).byte().cpu().numpy().astype('uint8')[0]  # [H, W, C]
                    for i in range(missing):
                        frame_filename = os.path.join(output_path, f"frame_{total_frames_written:06d}.png")
                        cv2.imwrite(frame_filename, cv2.cvtColor(padding_np, cv2.COLOR_RGB2BGR))
                        total_frames_written += 1
                        if (i + 1) % 10 == 0:
                            log(f"[Merge]   Padded {i + 1}/{missing} frames...", "info")
                log(f"[Merge] ✓ Padded to {total_frames_written} frames (target: {total_frames})", "success")
            elif total_frames is not None and total_frames_written > total_frames:
                log(f"[Merge] WARNING: Saved {total_frames_written} frames > target {total_frames}. May have extra frames.", "warning")
            
            log(f"\n========== Merge Completed Successfully ==========", "finish")
            log(f"Output directory: {output_path}", "finish")
            log(f"Total frames: {total_frames_written}", "finish")
            log(f"Image dimensions: {H}x{W}x{C}", "finish")
            log(f"Frame format: PNG (frame_XXXXXX.png)", "finish")
            log(f"=============================================", "finish")
            
        else:
            # 视频模式：使用临时文件方式（更稳定，避免 BrokenPipeError，特别适合16K视频）
            log(f"[Merge] [Step 2/3] Output mode: video file", "info")
            log(f"[Merge] Creating temporary raw video file...", "info")
            log(f"[Merge] Using temp file method (more stable for 16K videos, avoids BrokenPipeError)", "info")
            
            try:
                # 创建临时文件
                tmp_yuv = tempfile.NamedTemporaryFile(suffix='.yuv', delete=False)
                tmp_yuv_path = tmp_yuv.name
                log(f"[Merge] Temporary file: {tmp_yuv_path}", "info")
                
                total_frames_written = 0
                last_frame = None
                
                # 流式处理每个 rank 的结果，写入临时文件
                log(f"[Merge] [Step 3/3] Processing {len(result_files)} rank results and writing to temp file...", "info")
                for rank_idx, (r, result_file, file_size_mb) in enumerate(result_files):
                    log(f"[Merge]   [{rank_idx + 1}/{len(result_files)}] Processing Rank {r}...", "info")
                    log(f"[Merge]     - Loading from: {result_file}", "info")
                    log(f"[Merge]     - File size: {file_size_mb:.2f} MB ({file_size_mb/1024:.2f} GB)", "info")
                    
                    # 加载当前 rank 的结果（使用 CPU 内存）
                    segment = torch.load(result_file, map_location='cpu')
                    segment_frames = segment.shape[0]
                    log(f"[Merge]     - ✓ Loaded {segment_frames} frames, shape: {segment.shape}", "success")
                    
                    # 转换为 numpy 并分批写入临时文件
                    log(f"[Merge]     - Converting to numpy and writing to temp file...", "info")
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
                            log(f"[Merge]     - Written {frames_in_rank}/{segment_np.shape[0]} frames from Rank {r} to temp file (total: {total_frames_written} frames)", "info")
                    
                    # 保存最后一帧（用于可能的填充）
                    last_frame = segment[-1:, :, :, :]
                    
                    # 释放内存
                    del segment, segment_np
                    gc.collect()
                    
                    log(f"[Merge]     - ✓ Rank {r} completed. Total frames in temp file: {total_frames_written}", "success")
                
                # 检查是否需要填充
                if total_frames is not None and total_frames_written < total_frames:
                    missing = total_frames - total_frames_written
                    log(f"[Merge] Padding {missing} frames using the last frame...", "info")
                    if last_frame is not None:
                        padding_np = (last_frame.clamp(0, 1) * 255).byte().cpu().numpy().astype('uint8')
                        for i in range(missing):
                            tmp_yuv.write(padding_np[0].tobytes())
                            total_frames_written += 1
                            if (i + 1) % 10 == 0:
                                log(f"[Merge]   Padded {i + 1}/{missing} frames...", "info")
                    log(f"[Merge] ✓ Padded to {total_frames_written} frames (target: {total_frames})", "success")
                elif total_frames is not None and total_frames_written > total_frames:
                    log(f"[Merge] WARNING: Written {total_frames_written} frames > target {total_frames}. Video may have extra frames.", "warning")
                
                # 关闭临时文件
                tmp_yuv.close()
                tmp_yuv_size_mb = os.path.getsize(tmp_yuv_path) / (1024**2)
                log(f"[Merge] ✓ Temp file created: {tmp_yuv_size_mb:.2f} MB ({tmp_yuv_size_mb/1024:.2f} GB)", "success")
                
                # 使用 FFmpeg 从临时文件编码
                log(f"[Merge] Starting FFmpeg encoding from temp file...", "info")
                import time
                encoding_start = time.time()
                
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-s', f'{W}x{H}',
                    '-pix_fmt', 'rgb24',
                    '-r', str(input_fps),
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
                
                # 检查结果
                encoding_success = False
                if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    output_size_mb = os.path.getsize(output_path) / (1024**2)
                    log(f"\n========== Merge Completed Successfully ==========", "finish")
                    log(f"Output video: {output_path}", "finish")
                    log(f"Output size: {output_size_mb:.2f} MB ({output_size_mb/1024:.2f} GB)", "finish")
                    log(f"Total frames: {total_frames_written}", "finish")
                    log(f"Video dimensions: {H}x{W}x{C}", "finish")
                    log(f"FPS: {input_fps}", "finish")
                    log(f"Encoding time: {int(encoding_time)}s", "finish")
                    log(f"=============================================", "finish")
                    encoding_success = True
                else:
                    stderr = result.stderr.decode('utf-8', errors='ignore')
                    log(f"[Merge] ERROR: FFmpeg failed with return code {result.returncode}", "error")
                    log(f"[Merge] FFmpeg stderr: {stderr[:1000]}", "error")
                    log(f"[Merge] WARNING: Temporary file preserved for debugging/retry: {tmp_yuv_path}", "warning")
                    log(f"[Merge] You can manually retry FFmpeg encoding with:", "warning")
                    log(f"[Merge]   ffmpeg -y -f rawvideo -vcodec rawvideo -s {W}x{H} -pix_fmt rgb24 -r {input_fps} -i {tmp_yuv_path} -c:v libx264 -pix_fmt yuv420p -movflags +faststart -crf 18 {output_path}", "warning")
                    raise RuntimeError(f"FFmpeg failed with return code {result.returncode}: {stderr[:500]}")
                    
                # 只在成功时清理临时文件
                if encoding_success and tmp_yuv_path and os.path.exists(tmp_yuv_path):
                    try:
                        os.unlink(tmp_yuv_path)
                        log(f"[Merge] Cleaned up temporary file", "info")
                    except Exception as e:
                        log(f"[Merge] Warning: Failed to delete temp file {tmp_yuv_path}: {e}", "warning")
                        
            except Exception as e:
                # 如果发生异常，保留临时文件以便调试
                if tmp_yuv_path and os.path.exists(tmp_yuv_path):
                    log(f"[Merge] ERROR: Exception occurred, temporary file preserved: {tmp_yuv_path}", "error")
                    log(f"[Merge] You can manually retry FFmpeg encoding with the preserved temp file", "error")
                raise
                
    except Exception as e:
        log(f"[Merge] ERROR: Streaming merge failed: {e}", "error")
        import traceback
        log(f"[Merge] Traceback: {traceback.format_exc()}", "error")
        # 如果外层异常，也保留临时文件
        if tmp_yuv_path and os.path.exists(tmp_yuv_path):
            log(f"[Merge] WARNING: Temporary file preserved due to error: {tmp_yuv_path}", "warning")
        raise

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
    parser.add_argument("--world_size", type=int, default=8,
                        help="Number of ranks (default: 8)")
    parser.add_argument("--total_frames", type=int, default=None,
                        help="Expected total frames (for validation and padding/cropping)")
    parser.add_argument("--recover_rank", type=int,
                        help="Recover a specific rank (shows instructions)")
    parser.add_argument("--output_mode", type=str, default="video", choices=["video", "pictures"],
                        help="Output mode: 'video' for video file (default), 'pictures' for image sequence")
    
    args = parser.parse_args()
    
    if args.status:
        check_status(args.checkpoint_dir)
    elif args.merge_partial:
        if not args.output:
            log("ERROR: --output is required for --merge_partial", "error")
            return
        merge_partial_results(args.checkpoint_dir, args.output, args.fps, args.world_size, args.total_frames, args.output_mode)
    elif args.recover_rank is not None:
        recover_rank(args.checkpoint_dir, args.recover_rank, vars(args))
    else:
        log("ERROR: Please specify --status, --merge_partial, or --recover_rank", "error")
        parser.print_help()

if __name__ == "__main__":
    main()
