import os
import sys
import math
import argparse

# 设置 LD_LIBRARY_PATH 以支持 block_sparse_attention
# 确保在导入 torch 之前设置，以便后续导入的 C++ 扩展能找到 PyTorch 库
_torch_lib_path = "/usr/local/lib/python3.10/dist-packages/torch/lib"
if os.path.exists(_torch_lib_path):
    _current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if _torch_lib_path not in _current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{_torch_lib_path}:{_current_ld_path}"

import torch
import torch.nn.functional as F
import torchvision
import cv2
from tqdm import tqdm
from einops import rearrange
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
import uuid
import time
import hashlib
import json
import glob
import gc

# 将项目根目录添加到 sys.path，而不是 src 目录
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
#                      Utility Functions
# ==============================================================

def get_device_list():
    """Return list of available devices."""
    devs = ["auto"]
    try:
        if torch.cuda.is_available():
            devs += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    try:
        if hasattr(torch, "mps") and torch.mps.is_available():
            devs += [f"mps:{i}" for i in range(torch.mps.device_count())]
    except Exception:
        pass
    return devs

def get_gpu_memory_info(device: str) -> Tuple[float, float]:
    """Get GPU memory info (used, total) in GB."""
    if not device.startswith("cuda:"):
        return 0.0, 0.0
    try:
        idx = int(device.split(":")[1])
        torch.cuda.set_device(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(idx) / (1024**3)
        allocated = torch.cuda.memory_allocated(idx) / (1024**3)
        used = reserved  # 使用reserved memory作为使用量
        free = total - used
        return used, total
    except Exception as e:
        log(f"Error getting GPU memory info: {e}", "warning")
        return 0.0, 0.0

def get_available_memory_gb(device: str) -> float:
    """Get available GPU memory in GB."""
    used, total = get_gpu_memory_info(device)
    return total - used

def get_system_memory_info() -> Tuple[float, float]:
    """Get system RAM memory info (used, total) in GB.
    
    Returns:
        (used_gb, total_gb): Used and total system memory in GB
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        used_gb = mem.used / (1024**3)
        return used_gb, total_gb
    except ImportError:
        # 如果没有psutil，尝试从/proc/meminfo读取
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                for line in meminfo.split('\n'):
                    if line.startswith('MemTotal:'):
                        total_kb = int(line.split()[1])
                        total_gb = total_kb / (1024**2)
                    elif line.startswith('MemAvailable:'):
                        avail_kb = int(line.split()[1])
                        avail_gb = avail_kb / (1024**2)
                        used_gb = total_gb - avail_gb
                        return used_gb, total_gb
        except Exception:
            pass
        return 0.0, 0.0
    except Exception as e:
        log(f"Error getting system memory info: {e}", "warning")
        return 0.0, 0.0

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
    
    # 中间激活：由于使用tiled处理，实际激活显存较小，约5-8x输入（更激进的估算）
    # 同时处理多个tile时，某些激活可以共享
    intermediate_size = input_size * 6  # 从12倍降低到6倍，更符合实际
    
    # 添加一些额外开销（梯度缓冲区等，虽然inference不需要，但框架可能有保留）
    overhead = 0.5  # 0.5GB额外开销
    
    return input_size + intermediate_size + output_size + overhead

def log(message: str, message_type: str = 'normal'):
    """Colored logging for console output."""
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    elif message_type == 'info':
        message = '\033[1;33m' + message + '\033[m'
     # 确保换行并刷新输出，避免与 tqdm 进度条冲突
    print(message, flush=True)

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
    """Convert tensor (B,C,F,H,W) to normalized video tensor (F,H,W,C) - 与 nodes.py 一致"""
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
    for i in tqdm(range(F_), desc="Preparing frames", ncols=80):
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
    """Create blending mask for overlapping tiles with Gaussian blur for smoother blending.
    
    Args:
        size: (H, W) tuple of mask dimensions
        overlap: Overlap size in pixels (determined by user parameter)
    
    Returns:
        mask: (1, 1, H, W) tensor with smooth blending weights
    """
    import torch.nn.functional as F
    
    H, W = size
    mask = torch.ones(1, 1, H, W, dtype=torch.float32)
    
    # 创建线性ramp作为基础
    ramp = torch.linspace(0, 1, overlap, dtype=torch.float32)
    
    # 应用高斯模糊使过渡更平滑
    # 使用1D高斯核，sigma约为overlap的1/3，使过渡更自然
    sigma = max(1.0, overlap / 3.0)
    kernel_size = int(2 * sigma * 2) + 1  # 确保是奇数
    if kernel_size > 1 and kernel_size <= overlap:
        # 创建1D高斯核
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        
        # 对ramp应用高斯模糊
        ramp_padded = F.pad(ramp.unsqueeze(0).unsqueeze(0), (kernel_size//2, kernel_size//2), mode='reflect')
        ramp_blurred = F.conv1d(ramp_padded, gaussian_1d.unsqueeze(0).unsqueeze(0), padding=0)
        ramp = ramp_blurred.squeeze()
        # 重新归一化到[0, 1]
        if ramp.max() > ramp.min():
            ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min())
    
    # 应用ramp到四个边缘
    ramp_h = ramp.view(1, 1, -1, 1)  # 用于垂直边缘
    ramp_w = ramp.view(1, 1, 1, -1)  # 用于水平边缘
    
    # 左边缘
    mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp_w)
    # 右边缘
    mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp_w.flip(-1))
    # 上边缘
    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp_h)
    # 下边缘
    mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp_h.flip(-2))
    
    return mask

def pad_or_crop_video(frames):
    """Ensure temporal and spatial alignment (8n+1, 32x multiple)."""
    T, C, H, W = frames.shape
    aligned_F = largest_8n1_leq(T)
    if aligned_F < T:
        frames = frames[:aligned_F]
    elif aligned_F > T:
        pad = frames[-1:].repeat(aligned_F - T, 1, 1, 1)
        frames = torch.cat([frames, pad], dim=0)
    new_H = (H // 32) * 32
    new_W = (W // 32) * 32
    frames = frames[:, :, :new_H, :new_W]
    return frames

class StreamingVideoWriter:
    """流式视频写入器，支持追加写入帧，用于流式合成。
    
    使用FFmpeg管道实现真正的流式写入，避免将所有帧保存在内存中。
    """
    def __init__(self, path, fps=30, height=None, width=None, codec='libx264'):
        self.path = path
        self.fps = fps
        self.height = height
        self.width = width
        self.codec = codec
        self.process = None
        self.frame_count = 0
        self.initialized = False
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    def _initialize(self, height, width):
        """初始化FFmpeg进程"""
        if self.initialized:
            return
        
        self.height = height
        self.width = width
        
        import subprocess
        self.subprocess = subprocess
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgb24',
            '-r', str(self.fps),
            '-i', 'pipe:0',  # Read from stdin
            '-c:v', self.codec,
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-crf', '18',  # High quality
            self.path
        ]
        
        try:
            self.process = self.subprocess.Popen(
                cmd,
                stdin=self.subprocess.PIPE,
                stdout=self.subprocess.PIPE,
                stderr=self.subprocess.PIPE
            )
            self.initialized = True
            log(f"[StreamingVideoWriter] Initialized: {width}x{height} @ {self.fps}fps", "info")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Streaming video writer requires FFmpeg.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize streaming video writer: {e}")
    
    def write_frames(self, frames):
        """写入帧（可以是单个帧或帧序列）
        
        Args:
            frames: torch.Tensor, shape (F, H, W, C) or (H, W, C)
        """
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)  # (H, W, C) -> (1, H, W, C)
        
        # 转换为numpy
        frames_np = (frames.clamp(0, 1) * 255).byte().cpu().numpy().astype('uint8')
        
        # 初始化（使用第一帧的尺寸）
        if not self.initialized:
            F, H, W, C = frames_np.shape
            self._initialize(H, W)
        
        # 写入帧
        for frame in frames_np:
            try:
                self.process.stdin.write(frame.tobytes())
                self.frame_count += 1
            except BrokenPipeError:
                raise RuntimeError("FFmpeg process terminated unexpectedly")
            except Exception as e:
                raise RuntimeError(f"Failed to write frame: {e}")
    
    def close(self):
        """关闭写入器"""
        if self.process:
            # 根据帧数和分辨率动态计算超时时间
            # 对于大视频（8K、高帧数），编码可能需要更长时间
            # 估算：每帧约需要 0.1-0.5 秒编码时间（取决于分辨率和复杂度）
            # 使用更保守的估算：每帧 1 秒，最小 30 秒，最大 600 秒（10分钟）
            if self.frame_count > 0 and self.height and self.width:
                # 计算像素数（影响编码时间）
                pixels_per_frame = self.height * self.width
                # 对于高分辨率视频，需要更多时间
                # 8K (7680x4320) ≈ 33M pixels, 4K (3840x2160) ≈ 8M pixels
                # 估算：每百万像素每帧需要约 0.03 秒
                estimated_time_per_frame = max(0.1, (pixels_per_frame / 1e6) * 0.03)
                timeout = max(30, min(600, self.frame_count * estimated_time_per_frame * 2))  # 2倍安全系数
            else:
                timeout = 300  # 默认5分钟
            
            try:
                self.process.stdin.close()
                log(f"[StreamingVideoWriter] Waiting for FFmpeg to finish encoding ({self.frame_count} frames, timeout: {timeout:.1f}s)...", "info")
                self.process.wait(timeout=timeout)
                if self.process.returncode != 0:
                    stderr = self.process.stderr.read().decode('utf-8', errors='ignore')
                    log(f"[StreamingVideoWriter] FFmpeg warning: {stderr[:200]}", "warning")
                else:
                    log(f"[StreamingVideoWriter] FFmpeg encoding completed successfully", "info")
            except self.subprocess.TimeoutExpired:
                self.process.kill()
                log(f"[StreamingVideoWriter] FFmpeg process killed due to timeout ({timeout:.1f}s). Video may be incomplete!", "error")
                log(f"[StreamingVideoWriter] This usually happens with very large videos. Consider increasing timeout or processing in smaller segments.", "warning")
            except Exception as e:
                log(f"[StreamingVideoWriter] Error closing: {e}", "warning")
            finally:
                self.process = None
        
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            log(f"[StreamingVideoWriter] Saved {self.frame_count} frames to {self.path}", "info")
        else:
            raise RuntimeError(f"Failed to create output video: {self.path}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def save_video(frames, path, fps=30):
    """Save tensor video frames to MP4 with H.264 encoding for Windows compatibility.
    
    Priority: FFmpeg subprocess (H.264) > torchvision.io.write_video > OpenCV with H.264 > OpenCV with mp4v
    frames is (F, H, W, C) format.
    
    Returns:
        True if video was successfully saved and verified, raises RuntimeError otherwise
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    # Convert frames to numpy for all methods
    frames_np = (frames.clamp(0, 1) * 255).byte().cpu().numpy().astype('uint8')
    h, w = frames_np.shape[1:3]
    num_frames = frames_np.shape[0]
    
    # Method 1: Try FFmpeg subprocess (most reliable, guaranteed H.264)
    try:
        import subprocess
        import tempfile
        
        # Write frames to temporary raw video file
        with tempfile.NamedTemporaryFile(suffix='.yuv', delete=False) as tmp_yuv:
            tmp_yuv_path = tmp_yuv.name
            # Write raw RGB24 frames
            for f in frames_np:
                tmp_yuv.write(f.tobytes())
        
        # Use FFmpeg to encode with H.264
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', tmp_yuv_path,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-crf', '18',  # High quality
            path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        
        # Clean up temp file
        try:
            os.unlink(tmp_yuv_path)
        except:
            pass
        
        if result.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 0:
            log(f"[save_video] Successfully saved video using FFmpeg (H.264): {path}", "info")
            return
        else:
            log(f"[save_video] FFmpeg failed: {result.stderr.decode('utf-8', errors='ignore')[:200]}", "warning")
    except FileNotFoundError:
        log("[save_video] FFmpeg not found, trying torchvision...", "warning")
    except Exception as e:
        log(f"[save_video] FFmpeg subprocess failed: {e}, trying torchvision...", "warning")
    
    # Method 2: Try torchvision.io.write_video (uses H.264, best Windows compatibility)
    try:
        # frames is (F, H, W, C), convert to (F, C, H, W) for torchvision
        frames_torch = frames.permute(0, 3, 1, 2).clamp(0, 1)  # (F, C, H, W)
        frames_torch = (frames_torch * 255).byte().cpu()
        
        # Verify shape is correct: should be (F, C, H, W)
        if len(frames_torch.shape) != 4 or frames_torch.shape[1] != 3:
            raise ValueError(f"Invalid tensor shape for torchvision: {frames_torch.shape}, expected (F, 3, H, W)")
        
        # torchvision.io.write_video uses H.264 codec by default
        # Note: Some versions may not support video_codec parameter, so we try without it first
        try:
            torchvision.io.write_video(
                path,
                frames_torch,
                fps=fps,
                video_codec='h264',
                options={'pix_fmt': 'yuv420p', 'movflags': 'faststart'}
            )
        except (TypeError, RuntimeError) as e:
            # Fallback for versions that don't support video_codec parameter
            try:
                torchvision.io.write_video(path, frames_torch, fps=fps)
            except Exception as e2:
                raise e2 from e
        
        # Verify file was created
        if os.path.exists(path) and os.path.getsize(path) > 0:
            log(f"[save_video] Successfully saved video using torchvision (H.264): {path}", "info")
            return
        else:
            log(f"[save_video] torchvision created empty file, falling back", "warning")
    except Exception as e:
        log(f"[save_video] torchvision.write_video failed: {e}, falling back to OpenCV", "warning")
    
    # Method 3: Try OpenCV with H.264 codec (frames_np already converted above)
    
    # Try different H.264 fourcc codes for better compatibility
    h264_codecs = [
        ('avc1', 'H.264/AVC1 (best Windows compatibility)'),
        ('H264', 'H.264 (alternative)'),
        ('mp4v', 'MPEG-4 Part 2 (fallback, may not work on Windows)')
    ]
    
    for fourcc_str, description in h264_codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
            
            if not writer.isOpened():
                log(f"[save_video] OpenCV failed to open writer with {fourcc_str}", "warning")
                continue
            
            for f in frames_np:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            
            # Verify file was created and has content
            if os.path.exists(path) and os.path.getsize(path) > 0:
                log(f"[save_video] Successfully saved video using OpenCV ({description}): {path}", "info")
                return
            else:
                log(f"[save_video] File created but empty with {fourcc_str}, trying next codec", "warning")
        except Exception as e:
            log(f"[save_video] OpenCV with {fourcc_str} failed: {e}, trying next codec", "warning")
            if 'writer' in locals():
                writer.release()
    
    raise RuntimeError(f"[save_video] All methods failed to save video: {path}")

def read_video_to_tensor(video_path):
    """Read video and convert to (N, H, W, C) format for prepare_input_tensor.
    
    Returns: (frames_tensor, fps)
    - frames_tensor: (N, H, W, C) format
    - fps: float, frames per second

    Primary backend: torchvision.io.read_video (ffmpeg-based)
    Fallback 1: OpenCV VideoCapture with different backends
    Fallback 2: FFmpeg via subprocess (most compatible)
    """
    # 首先检查文件是否存在和可读
    if not os.path.exists(video_path):
        raise RuntimeError(f"[read_video] Video file does not exist: {video_path}")
    if not os.path.isfile(video_path):
        raise RuntimeError(f"[read_video] Path is not a file: {video_path}")
    if not os.access(video_path, os.R_OK):
        raise RuntimeError(f"[read_video] Video file is not readable: {video_path}")
    
    file_size = os.path.getsize(video_path)
    log(f"[read_video] Attempting to read video: {video_path} (size: {file_size / (1024*1024):.2f} MB)", "info")
    
    # Method 1: Try torchvision.io.read_video
    try:
        video_data = torchvision.io.read_video(video_path, pts_unit='sec')
        vr = video_data[0]
        # video_data[1] is audio, video_data[2] is info dict
        info = video_data[2] if len(video_data) > 2 else {}
        # Get fps from info dict or use default
        if isinstance(info, dict):
            fps = info.get('video_fps', 30.0)
        else:
            fps = 30.0  # Default if info is not a dict
        
        if vr.numel() > 0 and vr.shape[0] > 0:
            vr = vr.permute(0, 3, 1, 2).float() / 255.0  # (N, C, H, W)
            log(f"[read_video] Successfully read {vr.shape[0]} frames using torchvision (fps: {fps:.2f})", "info")
            return vr.permute(0, 2, 3, 1), fps  # (N, H, W, C), fps
        else:
            log(f"[read_video] torchvision returned empty tensor for: {video_path}", "warning")
    except Exception as e:
        log(f"[read_video] torchvision read_video failed: {e}", "warning")

    # Method 2: Try OpenCV VideoCapture with different backends
    log("[read_video] Falling back to OpenCV VideoCapture...", "warning")
    
    # Try different OpenCV backends for better compatibility
    backends_to_try = [
        cv2.CAP_FFMPEG,  # FFmpeg backend (most compatible)
        cv2.CAP_ANY,     # Auto-detect
    ]
    
    cap = None
    for backend in backends_to_try:
        try:
            cap = cv2.VideoCapture(video_path, backend)
            if cap.isOpened():
                # Verify we can actually read frames
                ret, _ = cap.read()
                if ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                    log(f"[read_video] OpenCV opened video with backend: {backend}", "info")
                    break
            if cap:
                cap.release()
                cap = None
        except Exception as e:
            log(f"[read_video] OpenCV backend {backend} failed: {e}", "warning")
            if cap:
                cap.release()
                cap = None
    
    if cap and cap.isOpened():
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 1000:  # Sanity check
            fps = 30.0
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Validate frame count (some codecs return invalid values)
        if frame_count <= 0 or frame_count > 1000000:
            log(f"[read_video] OpenCV reported invalid frame count ({frame_count}), will count by reading", "warning")
            frame_count = -1  # Will count by reading
        
        log(f"[read_video] OpenCV video info: {frame_count if frame_count > 0 else 'unknown'} frames, {width}x{height}, {fps:.2f} fps", "info")
        
        frames = []
        frame_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                if frame_idx == 0:
                    log(f"[read_video] OpenCV could not read any frames", "warning")
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame_rgb).float() / 255.0)
            frame_idx += 1
        
        cap.release()
        
        if len(frames) > 0:
            # Update fps if we have actual frame count
            if frame_count <= 0 and len(frames) > 0:
                # Try to estimate fps from duration if available
                pass  # Keep the fps from CAP_PROP_FPS
            log(f"[read_video] Successfully read {len(frames)} frames using OpenCV (fps: {fps:.2f})", "info")
            vr = torch.stack(frames, dim=0)  # (N,H,W,C)
            return vr, fps
    
    # Method 3: Try FFmpeg via subprocess (most reliable fallback)
    log("[read_video] Falling back to FFmpeg subprocess...", "warning")
    try:
        import subprocess
        import tempfile
        import numpy as np
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use FFmpeg to extract frames as raw RGB images
            # FFmpeg command: extract frames as raw RGB24 (packed)
            cmd = [
                'ffmpeg', '-i', video_path,
                '-f', 'rawvideo',
                '-pix_fmt', 'rgb24',
                '-'
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            
            if result.returncode == 0 and len(result.stdout) > 0:
                # Parse FFmpeg output to get video dimensions
                # Try to get dimensions from stderr or use ffprobe
                probe_cmd = [
                    'ffprobe', '-v', 'error',
                    '-select_streams', 'v:0',
                    '-show_entries', 'stream=width,height,r_frame_rate',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    video_path
                ]
                
                probe_result = subprocess.run(
                    probe_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                if probe_result.returncode == 0:
                    lines = probe_result.stdout.strip().split('\n')
                    width = int(lines[0]) if len(lines) > 0 and lines[0].isdigit() else 1920
                    height = int(lines[1]) if len(lines) > 1 and lines[1].isdigit() else 1080
                    # Parse fps from r_frame_rate (format: "30/1" or "29.97/1")
                    fps_str = lines[2] if len(lines) > 2 else "30/1"
                    try:
                        if '/' in fps_str:
                            num, den = map(float, fps_str.split('/'))
                            fps = num / den if den > 0 else 30.0
                        else:
                            fps = float(fps_str) if fps_str.replace('.', '').isdigit() else 30.0
                    except:
                        fps = 30.0
                    
                    # Parse raw video data
                    frame_size = width * height * 3
                    num_frames = len(result.stdout) // frame_size
                    
                    if num_frames > 0:
                        frames_data = np.frombuffer(result.stdout[:num_frames * frame_size], dtype=np.uint8)
                        frames_data = frames_data.reshape(num_frames, height, width, 3)
                        
                        frames = [torch.from_numpy(frame).float() / 255.0 for frame in frames_data]
                        log(f"[read_video] Successfully read {len(frames)} frames using FFmpeg subprocess (fps: {fps:.2f})", "info")
                        vr = torch.stack(frames, dim=0)  # (N,H,W,C)
                        return vr, fps
    except FileNotFoundError:
        log("[read_video] FFmpeg not found in PATH", "warning")
    except Exception as e:
        log(f"[read_video] FFmpeg subprocess failed: {e}", "warning")
    
    # All methods failed
    error_msg = f"[read_video] All methods failed to read video: {video_path}\n"
    error_msg += f"  - File exists: {os.path.exists(video_path)}\n"
    error_msg += f"  - File readable: {os.access(video_path, os.R_OK)}\n"
    error_msg += f"  - File size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)\n"
    error_msg += "\nPossible causes:\n"
    error_msg += "  1. Unsupported video codec/container format\n"
    error_msg += "  2. Corrupted video file\n"
    error_msg += "  3. Missing codec support (try installing: apt-get install -y ffmpeg libavcodec-dev)\n"
    error_msg += "  4. Very low frame rate or unusual encoding parameters\n"
    raise RuntimeError(error_msg)

def split_video_by_frames(frames: torch.Tensor, num_gpus: int, overlap: int = 10):
    N = frames.shape[0]
    segment_size = N // num_gpus
    segments = []
    for i in range(num_gpus):
        start_idx = max(0, i * segment_size - overlap if i > 0 else 0)
        end_idx = min(N, (i + 1) * segment_size + overlap if i < num_gpus - 1 else N)
        segments.append((start_idx, end_idx))  # 仅返回区间
    return segments

def merge_video_segments_streaming(segments_info: List[Tuple[int, int, str]], original_length: int) -> torch.Tensor:
    """流式合并segments，逐个加载和处理，避免OOM。
    
    Args:
        segments_info: List of (start_idx, end_idx, file_path) tuples，按start_idx排序
        original_length: Original number of frames
    
    Returns:
        Merged video tensor (F, H, W, C)
    """
    if not segments_info:
        raise ValueError("No segments to merge")
    
    segments_info = sorted(segments_info, key=lambda x: x[0])  # 按start_idx排序
    
    merged_parts = []
    last_expected_end_idx = 0  # 使用期望的end_idx来判断overlap和计算期望帧数
    import gc
    
    for i, (start_idx, end_idx, path) in enumerate(segments_info):
        log(f"[Multi-GPU] Loading segment {i+1}/{len(segments_info)} from {path}...", "info")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Segment file not found: {path}")
        
        # 检查文件大小
        file_size = os.path.getsize(path)
        file_size_gb = file_size / (1024**3)
        log(f"[Multi-GPU] Segment {i+1} file size: {file_size_gb:.2f} GB", "info")
        
        # 加载segment
        segment = torch.load(path, map_location='cpu')
        log(f"[Multi-GPU] Loaded segment {i+1}: {segment.shape}", "info")
        
        # 处理overlap：跳过与上一个segment重叠的部分（基于期望的end_idx）
        overlap_frames = 0
        if last_expected_end_idx > 0 and start_idx < last_expected_end_idx:
            overlap_frames = last_expected_end_idx - start_idx
            if overlap_frames < segment.shape[0]:
                segment = segment[overlap_frames:]
                log(f"[Multi-GPU] Skipped {overlap_frames} overlap frames (start_idx={start_idx}, last_expected_end_idx={last_expected_end_idx})", "info")
            elif overlap_frames >= segment.shape[0]:
                log(f"[Multi-GPU] WARNING: Segment {i+1} is completely overlapped, skipping", "warning")
                del segment
                gc.collect()
                last_expected_end_idx = end_idx  # 更新期望的结束位置
                continue
        
        # 计算期望的帧数（考虑overlap，使用绝对帧索引）
        if overlap_frames > 0:
            # 如果处理了overlap，期望的帧数应该是从last_expected_end_idx到end_idx
            expected_frames = end_idx - last_expected_end_idx
        else:
            # 如果没有overlap，期望的帧数就是end_idx - start_idx
            expected_frames = end_idx - start_idx
        
        # 如果实际帧数不足，填充到期望的帧数
        if segment.shape[0] < expected_frames:
            missing_frames = expected_frames - segment.shape[0]
            log(f"[Multi-GPU] Segment {i+1} frame count insufficient: actual {segment.shape[0]} frames, expected {expected_frames} frames", "warning")
            log(f"[Multi-GPU] Padding {missing_frames} frames (using last frame)...", "info")
            last_frame = segment[-1:, :, :, :]
            padding_frames = last_frame.repeat(missing_frames, 1, 1, 1)
            segment = torch.cat([segment, padding_frames], dim=0)
            log(f"[Multi-GPU] Padding complete: {segment.shape[0]} frames", "info")
        elif segment.shape[0] > expected_frames:
            # 如果实际帧数超过期望，裁剪到期望的帧数
            segment = segment[:expected_frames]
            log(f"[Multi-GPU] Cropped to expected frame count: {segment.shape[0]} frames", "info")
        
        # 添加到合并列表
        if segment.shape[0] > 0:
            merged_parts.append(segment)
            last_expected_end_idx = end_idx  # 更新期望的结束位置
            
            # 每处理2个segments就合并一次，释放内存（避免累积太多）
            if len(merged_parts) >= 2:
                log(f"[Multi-GPU] Merging {len(merged_parts)} parts to reduce memory...", "info")
                merged = torch.cat(merged_parts, dim=0)
                merged_parts = [merged]
                # 注意：merged_parts中的旧tensor引用会自动被Python的GC处理
                gc.collect()
            # segment已经被添加到merged_parts，不需要手动删除
        else:
            # segment为空，直接删除
            del segment
            gc.collect()
    
    # 最终合并
    if len(merged_parts) > 1:
        log(f"[Multi-GPU] Final merge of {len(merged_parts)} parts...", "info")
        final_output = torch.cat(merged_parts, dim=0)
    elif len(merged_parts) == 1:
        final_output = merged_parts[0]
    else:
        raise ValueError("No segments to merge")
    
    # 确保长度正确
    if final_output.shape[0] > original_length:
        final_output = final_output[:original_length]
        log(f"[Multi-GPU] Trimmed to original length: {original_length}", "info")
    elif final_output.shape[0] < original_length:
        # 如果长度不够，重复最后一帧
        log(f"[Multi-GPU] Padding to original length: {original_length} (current: {final_output.shape[0]})", "info")
        last_frame = final_output[-1:].repeat(original_length - final_output.shape[0], 1, 1, 1)
        final_output = torch.cat([final_output, last_frame], dim=0)
    
    return final_output

def merge_video_segments(segments: List[Tuple[int, int, torch.Tensor]], original_length: int) -> torch.Tensor:
    """Merge processed video segments back into a single video.
    
    Args:
        segments: List of (start_idx, end_idx, processed_segment) tuples
        original_length: Original number of frames
    
    Returns:
        Merged video tensor (F, H, W, C)
    """
    if not segments:
        raise ValueError("No segments to merge")
    
    segments = sorted(segments, key=lambda x: x[0])
    
    # 简单的合并策略：直接连接segments，处理overlap区域
    merged_parts = []
    
    for i, (start_idx, end_idx, segment) in enumerate(segments):
        segment_frames = segment.shape[0]
        
        if i == 0:
            # 第一个segment：保留全部，但需要根据原始长度调整
            # 计算应该保留多少帧
            if len(segments) == 1:
                # 只有一个segment，直接裁剪
                merged_parts.append(segment[:original_length])
            else:
                # 保留到下一个segment的start_idx（考虑overlap）
                next_start = segments[i+1][0] if i+1 < len(segments) else original_length
                keep_frames = min(segment_frames, next_start - start_idx)
                merged_parts.append(segment[:keep_frames])
        else:
            # 后续segments：跳过overlap部分
            prev_end = segments[i-1][1]
            overlap = max(0, start_idx - prev_end)
            
            # 计算当前segment应该从哪一帧开始
            segment_start_frame = min(overlap, segment_frames)
            
            # 计算应该保留到哪一帧
            if i == len(segments) - 1:
                # 最后一个segment：保留到original_length
                frames_needed = original_length - (sum(p.shape[0] for p in merged_parts) + segment_start_frame)
                keep_frames = min(segment_frames - segment_start_frame, frames_needed)
                if keep_frames > 0:
                    merged_parts.append(segment[segment_start_frame:segment_start_frame + keep_frames])
            else:
                # 中间segments：保留到下一个segment的start_idx
                next_start = segments[i+1][0]
                current_merged_length = sum(p.shape[0] for p in merged_parts)
                frames_needed = next_start - current_merged_length - segment_start_frame
                keep_frames = min(segment_frames - segment_start_frame, frames_needed)
                if keep_frames > 0:
                    merged_parts.append(segment[segment_start_frame:segment_start_frame + keep_frames])
    
    if not merged_parts:
        raise ValueError("Failed to merge segments")
    
    merged = torch.cat(merged_parts, dim=0)
    
    # 确保长度正确
    if merged.shape[0] > original_length:
        merged = merged[:original_length]
    elif merged.shape[0] < original_length:
        # 如果长度不够，重复最后一帧
        last_frame = merged[-1:].repeat(original_length - merged.shape[0], 1, 1, 1)
        merged = torch.cat([merged, last_frame], dim=0)
    
    return merged

# ==============================================================
#                     Checkpoint Functions
# ==============================================================

def get_checkpoint_id(input_path, args, segments, num_gpus):
    """生成检查点ID，基于输入视频路径和关键参数"""
    # 获取文件的修改时间和大小，确保文件变化时检查点失效
    try:
        stat = os.stat(input_path)
        file_info = (stat.st_mtime, stat.st_size)
    except:
        file_info = (0, 0)
    
    # 使用输入路径、参数、segments信息生成唯一ID
    key_data = {
        'input': os.path.abspath(input_path),
        'file_info': file_info,
        'scale': args.scale,
        'mode': args.mode,
        'tile_size': args.tile_size,
        'tile_overlap': args.tile_overlap,
        'tiled_dit': args.tiled_dit,
        'tiled_vae': args.tiled_vae,
        'segments': segments,
        'num_gpus': num_gpus,
        'original_frame_count': segments[-1][1] if segments else 0
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()

def load_checkpoint(checkpoint_dir):
    """加载检查点信息"""
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.json')
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            log(f"[Checkpoint] Failed to load checkpoint: {e}", "warning")
            return None
    return None

def save_checkpoint(checkpoint_dir, checkpoint_data):
    """保存检查点信息"""
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.json')
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    except Exception as e:
        log(f"[Checkpoint] Failed to save checkpoint: {e}", "warning")

def find_completed_segments(checkpoint_dir, segments):
    """查找已完成的segment"""
    completed = {}
    if not os.path.exists(checkpoint_dir):
        return completed
    
    checkpoint_data = load_checkpoint(checkpoint_dir)
    if not checkpoint_data:
        return completed
    
    # 检查每个segment的结果文件是否存在
    for i, (start_idx, end_idx) in enumerate(segments):
        segment_key = f"segment_{i}_{start_idx}_{end_idx}"
        if segment_key in checkpoint_data:
            result_path = checkpoint_data[segment_key]['path']
            if os.path.exists(result_path):
                # 验证文件是否完整（检查文件大小）
                try:
                    file_size = os.path.getsize(result_path)
                    if file_size > 0:  # 文件存在且非空
                        completed[i] = checkpoint_data[segment_key]
                        log(f"[Resume] Found completed segment {i} (frames {start_idx}-{end_idx}) at {result_path}", "info")
                except:
                    pass
    
    return completed

def clean_checkpoint(checkpoint_dir):
    """清理检查点目录"""
    try:
        if os.path.exists(checkpoint_dir):
            import shutil
            shutil.rmtree(checkpoint_dir)
            log(f"[Checkpoint] Cleaned checkpoint directory: {checkpoint_dir}", "info")
            return True
    except Exception as e:
        log(f"[Checkpoint] Failed to clean checkpoint: {e}", "warning")
        return False
    return False

# ==============================================================
#               Unified Checkpoint System (Frame-based)
# ==============================================================

def get_video_based_dir_name(input_path, scale):
    """生成基于视频名称和放大倍数的目录名
    
    Args:
        input_path: 输入视频路径
        scale: 放大倍数
    
    Returns:
        目录名，例如: "3D_cat_1080_4x"
    """
    # 获取视频文件名（不含扩展名）
    video_basename = os.path.basename(input_path)
    video_name = os.path.splitext(video_basename)[0]
    
    # 清理文件名中的特殊字符（避免目录名问题）
    import re
    video_name = re.sub(r'[^\w\-_\.]', '_', video_name)
    
    # 生成目录名
    dir_name = f"{video_name}_{scale}x"
    return dir_name

def get_video_based_dir_name(input_path, scale):
    """生成基于视频名称和放大倍数的目录名
    
    Args:
        input_path: 输入视频路径
        scale: 放大倍数
    
    Returns:
        目录名，例如: "3D_cat_1080_4x"
    """
    # 获取视频文件名（不含扩展名）
    video_basename = os.path.basename(input_path)
    video_name = os.path.splitext(video_basename)[0]
    
    # 清理文件名中的特殊字符（避免目录名问题）
    import re
    video_name = re.sub(r'[^\w\-_\.]', '_', video_name)
    
    # 生成目录名
    dir_name = f"{video_name}_{scale}x"
    return dir_name

def get_unified_checkpoint_id(input_path, args, total_frames):
    """生成统一的checkpoint ID，不依赖处理模式，只基于输入视频和参数"""
    try:
        stat = os.stat(input_path)
        file_info = (stat.st_mtime, stat.st_size)
    except:
        file_info = (0, 0)
    
    key_data = {
        'input': os.path.abspath(input_path),
        'file_info': file_info,
        'scale': args.scale,
        'mode': args.mode,
        'tile_size': args.tile_size,
        'tile_overlap': args.tile_overlap,
        'tiled_dit': args.tiled_dit,
        'tiled_vae': args.tiled_vae,
        'total_frames': total_frames
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()

def scan_all_completed_frames(input_path, args, total_frames):
    """扫描所有checkpoint位置（multi_gpu和segmented），返回已完成的帧范围列表
    
    Returns:
        List of tuples: [(start_frame, end_frame, file_path, mode), ...]
        mode: 'multi_gpu' or 'segmented'
    """
    checkpoint_id = get_unified_checkpoint_id(input_path, args, total_frames)
    completed_ranges = []  # [(start, end, file_path, mode), ...]
    
    # 1. 扫描 multi_gpu checkpoint
    multi_gpu_base_dir = os.path.join('/tmp', 'flashvsr_checkpoints')
    if os.path.exists(multi_gpu_base_dir):
        # 优先使用基于视频名称的目录
        video_dir_name = get_video_based_dir_name(input_path, args.scale)
        checkpoint_path = os.path.join(multi_gpu_base_dir, video_dir_name)
        
        if os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path):
            checkpoint_data = load_checkpoint(checkpoint_path)
            if checkpoint_data:
                for key, value in checkpoint_data.items():
                    if key.startswith('segment_') and isinstance(value, dict):
                        if 'start_idx' in value and 'end_idx' in value and 'path' in value:
                            start = value['start_idx']
                            end = value['end_idx']
                            path = value['path']
                            if os.path.exists(path):
                                try:
                                    file_size = os.path.getsize(path)
                                    if file_size > 0:
                                        completed_ranges.append((start, end, path, 'multi_gpu'))
                                except:
                                    pass
        
        # 向后兼容：遍历所有checkpoint目录，查找匹配的（旧格式）
        for checkpoint_dir in os.listdir(multi_gpu_base_dir):
            checkpoint_path = os.path.join(multi_gpu_base_dir, checkpoint_dir)
            if not os.path.isdir(checkpoint_path):
                continue
            # 跳过已经检查过的目录
            if checkpoint_dir == video_dir_name:
                continue
            
            checkpoint_data = load_checkpoint(checkpoint_path)
            if not checkpoint_data:
                continue
            
            # 检查是否匹配当前输入和参数
            # 通过检查checkpoint中的segment信息来判断
            for key, value in checkpoint_data.items():
                if key.startswith('segment_') and isinstance(value, dict):
                    if 'start_idx' in value and 'end_idx' in value and 'path' in value:
                        start = value['start_idx']
                        end = value['end_idx']
                        path = value['path']
                        if os.path.exists(path):
                            try:
                                file_size = os.path.getsize(path)
                                if file_size > 0:
                                    completed_ranges.append((start, end, path, 'multi_gpu'))
                            except:
                                pass
    
    # 2. 扫描 segmented checkpoint
    segmented_base_dir = os.path.join('/tmp', 'flashvsr_segments')
    if os.path.exists(segmented_base_dir):
        # 优先使用基于视频名称的目录
        video_dir_name = get_video_based_dir_name(input_path, args.scale)
        segment_dir = os.path.join(segmented_base_dir, video_dir_name)
        
        if os.path.exists(segment_dir) and os.path.isdir(segment_dir):
            segment_files = sorted(glob.glob(os.path.join(segment_dir, "segment_*.pt")))
            if segment_files:
                # 处理这个目录的segments
                for seg_file in segment_files:
                    if os.path.getsize(seg_file) == 0:
                        continue
                    
                    # 查找对应的metadata文件
                    basename = os.path.basename(seg_file)
                    metadata_file = os.path.join(segment_dir, basename.replace('.pt', '.json'))
                    
                    if os.path.exists(metadata_file):
                        # 从metadata文件读取帧范围
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            start_frame = metadata.get('start_frame', 0)
                            end_frame = metadata.get('end_frame', 0)
                            if start_frame < end_frame and end_frame <= total_frames:
                                completed_ranges.append((start_frame, end_frame, seg_file, 'segmented'))
                        except:
                            pass
        
        # 向后兼容：遍历所有segment目录（旧格式）
        for segment_dir_name in os.listdir(segmented_base_dir):
            segment_dir = os.path.join(segmented_base_dir, segment_dir_name)
            if not os.path.isdir(segment_dir):
                continue
            # 跳过已经检查过的目录
            if segment_dir_name == video_dir_name:
                continue
            
            segment_files = sorted(glob.glob(os.path.join(segment_dir, "segment_*.pt")))
            if not segment_files:
                continue
            
            # 从metadata文件读取帧范围信息（更准确）
            for seg_file in segment_files:
                if os.path.getsize(seg_file) == 0:
                    continue
                
                # 查找对应的metadata文件
                basename = os.path.basename(seg_file)
                metadata_file = os.path.join(segment_dir, basename.replace('.pt', '.json'))
                
                if os.path.exists(metadata_file):
                    # 从metadata文件读取帧范围
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        start_frame = metadata.get('start_frame', 0)
                        end_frame = metadata.get('end_frame', 0)
                        if start_frame < end_frame and end_frame <= total_frames:
                            completed_ranges.append((start_frame, end_frame, seg_file, 'segmented'))
                    except:
                        pass
                else:
                    # 如果没有metadata文件，尝试从文件名和文件内容推断（向后兼容）
                    try:
                        seg_idx = int(basename[8:12])  # segment_0000 -> 0000
                        seg = torch.load(seg_file, map_location='cpu')
                        actual_frames = seg.shape[0]
                        segment_overlap = getattr(args, 'segment_overlap', 2)
                        
                        # 粗略估算：假设每个segment大约100帧（实际值可能不同）
                        # 这是向后兼容的估算方法
                        estimated_start = seg_idx * 100
                        estimated_end = estimated_start + actual_frames
                        if estimated_end <= total_frames and estimated_start < estimated_end:
                            completed_ranges.append((estimated_start, estimated_end, seg_file, 'segmented'))
                        del seg
                    except:
                        pass
    
    # 3. 合并重叠的帧范围并去重
    if not completed_ranges:
        return []
    
    # 按start_frame排序
    completed_ranges.sort(key=lambda x: x[0])
    
    # 合并重叠范围
    merged_ranges = []
    for start, end, path, mode in completed_ranges:
        if merged_ranges and start <= merged_ranges[-1][1]:
            # 有重叠，合并（取更大的end）
            prev_start, prev_end, prev_path, prev_mode = merged_ranges[-1]
            merged_ranges[-1] = (prev_start, max(prev_end, end), prev_path, prev_mode)
        else:
            merged_ranges.append((start, end, path, mode))
    
    return merged_ranges

def find_missing_frame_ranges(total_frames, completed_ranges):
    """找出未完成的帧范围
    
    Args:
        total_frames: 总帧数
        completed_ranges: 已完成的帧范围列表 [(start, end, path, mode), ...]
    
    Returns:
        List of tuples: [(start_frame, end_frame), ...]
    """
    if not completed_ranges:
        return [(0, total_frames)]
    
    missing = []
    completed_ranges.sort(key=lambda x: x[0])
    
    # 检查开头
    if completed_ranges[0][0] > 0:
        missing.append((0, completed_ranges[0][0]))
    
    # 检查中间
    for i in range(len(completed_ranges) - 1):
        if completed_ranges[i][1] < completed_ranges[i+1][0]:
            missing.append((completed_ranges[i][1], completed_ranges[i+1][0]))
    
    # 检查结尾
    if completed_ranges[-1][1] < total_frames:
        missing.append((completed_ranges[-1][1], total_frames))
    
    return missing

def merge_all_completed_frames(completed_ranges, output_path, fps, total_frames):
    """合并所有已完成的帧范围到最终输出，正确处理overlap
    
    Args:
        completed_ranges: 已完成的帧范围列表 [(start, end, path, mode), ...]
        output_path: 输出视频路径
        fps: 视频帧率
        total_frames: 总帧数
    
    注意：
    - segmented模式的segments已经裁剪掉了overlap，可以直接拼接
    - multi_gpu模式的segments可能包含overlap，需要处理重叠部分
    """
    log(f"[Resume] Merging {len(completed_ranges)} completed frame ranges...", "info")
    
    # 按start_frame排序
    completed_ranges.sort(key=lambda x: x[0])
    
    merged_parts = []
    last_end_idx = 0
    
    for start, end, path, mode in completed_ranges:
        log(f"[Resume] Loading frames {start}-{end} from {path} ({mode})", "info")
        try:
            segment = torch.load(path, map_location='cpu')
            
            # 处理overlap：如果与上一个segment有重叠，跳过重叠部分
            if last_end_idx > 0 and start < last_end_idx:
                overlap_frames = last_end_idx - start
                if overlap_frames < segment.shape[0]:
                    segment = segment[overlap_frames:]
                    log(f"[Resume] Skipped {overlap_frames} overlap frames (start={start}, last_end={last_end_idx})", "info")
                elif overlap_frames >= segment.shape[0]:
                    log(f"[Resume] Warning: Segment is completely overlapped, skipping", "warning")
                    del segment
                    continue
            
            # 裁剪到实际需要的帧数
            actual_frames_needed = end - start
            if segment.shape[0] > actual_frames_needed:
                segment = segment[:actual_frames_needed]
            elif segment.shape[0] < actual_frames_needed:
                log(f"[Resume] Warning: Segment has {segment.shape[0]} frames, expected {actual_frames_needed}", "warning")
            
            merged_parts.append(segment)
            last_end_idx = end  # 更新最后结束位置
            del segment
        except Exception as e:
            log(f"[Resume] Error loading {path}: {e}", "error")
            raise
    
    # 合并所有部分
    log(f"[Resume] Concatenating {len(merged_parts)} parts...", "info")
    if not merged_parts:
        raise RuntimeError("[Resume] No valid segments to merge")
    
    final_output = torch.cat(merged_parts, dim=0)
    
    # 裁剪到原始帧数
    if final_output.shape[0] > total_frames:
        final_output = final_output[:total_frames]
    
    log(f"[Resume] Final output: {final_output.shape[0]} frames", "info")
    
    # 保存视频
    save_video(final_output, output_path, fps=fps)
    log(f"[Resume] Saved merged video to {output_path}", "finish")

def run_inference_multi_gpu(frames: torch.Tensor, devices: List[str], args, input_fps: float = 30.0):
    """Run inference using multiple GPUs in parallel.
    
    Args:
        frames: Full video frames tensor (N, H, W, C)
        devices: List of device strings
        args: Arguments namespace
        input_fps: Input video FPS (to avoid re-reading video in workers)
    """
    process_start = time.time()
    num_gpus = len(devices)
    if num_gpus == 0:
        raise ValueError("No GPUs specified for multi-GPU processing")
    
    log(f"[Multi-GPU] Processing video with {num_gpus} GPUs", "info")
    
    # 检查系统内存（RAM）
    sys_used, sys_total = get_system_memory_info()
    sys_free = sys_total - sys_used
    log(f"[Multi-GPU] System RAM: {sys_free:.2f} GB free / {sys_total:.2f} GB total (used: {sys_used:.2f} GB)", "info")
    if sys_free < 8.0:  # 至少需要8GB可用系统内存
        log(f"[Multi-GPU] Warning: Low system RAM ({sys_free:.2f} GB). OOM risk! Each worker needs ~4-8GB RAM.", "warning")
    
    # 检查每个GPU的可用内存
    log(f"[Multi-GPU] Checking GPU memory availability...", "info")
    for device in devices:
        used, total = get_gpu_memory_info(device)
        free = total - used
        log(f"[Multi-GPU] {device}: {free:.2f} GB free / {total:.2f} GB total (used: {used:.2f} GB)", "info")
        if free < 2.0:  # 至少需要2GB可用内存
            log(f"[Multi-GPU] Warning: {device} has low available memory ({free:.2f} GB). OOM risk!", "warning")
    
    # 保存原始帧数（在删除frames之前）
    original_frame_count = frames.shape[0]
    
    # 将视频分割成segments
    segment_overlap = getattr(args, 'segment_overlap', 2)
    segments = split_video_by_frames(frames, num_gpus, overlap=segment_overlap)
    log(f"[Multi-GPU] Split video into {len(segments)} segments (overlap: {segment_overlap} frames)", "info")
    for i, (start, end) in enumerate(segments):
        segment_frames = end - start
        # 估算segment内存占用（假设每帧是1920x1080x3的float32）
        estimated_memory_mb = segment_frames * 1920 * 1080 * 3 * 4 / (1024**2)  # 粗略估算
        log(f"[Multi-GPU] Segment {i}: frames {start}-{end} ({segment_frames} frames, ~{estimated_memory_mb:.1f} MB) -> GPU {devices[i % num_gpus]}", "info")
    
    # 检查点续传支持
    # 使用基于视频名称和放大倍数的目录名
    video_dir_name = get_video_based_dir_name(args.input, args.scale)
    checkpoint_dir = None
    completed_segments = {}
    
    # 只有在明确指定 --resume 时才进行断点续传
    if getattr(args, 'resume', False):
        if getattr(args, 'clean_checkpoint', False):
            # 清理检查点
            checkpoint_dir = os.path.join('/tmp', 'flashvsr_checkpoints', video_dir_name)
            clean_checkpoint(checkpoint_dir)
            log(f"[Multi-GPU] Checkpoint cleaned, starting fresh", "info")
        else:
            # 使用resume模式，检查已完成的segments
            checkpoint_dir = os.path.join('/tmp', 'flashvsr_checkpoints', video_dir_name)
            completed_segments = find_completed_segments(checkpoint_dir, segments)
            log(f"[Multi-GPU] Resume mode: Found {len(completed_segments)} completed segments out of {len(segments)}", "info")
            if len(completed_segments) == len(segments):
                log(f"[Multi-GPU] All segments already completed! Merging results...", "info")
                # 所有segment都已完成，直接合并结果
                results = completed_segments
                # 跳转到合并逻辑
                use_streaming = getattr(args, 'streaming', False)
                if use_streaming:
                    # 流式合并逻辑
                    output_path = args.output if args.output else args.input.replace('.mp4', '_4x.mp4')
                    # 需要先加载一个segment获取尺寸
                    first_segment_path = list(results.values())[0]['path']
                    first_segment = torch.load(first_segment_path, map_location='cpu')
                    F, H, W, C = first_segment.shape
                    streaming_writer = StreamingVideoWriter(output_path, input_fps, height=H, width=W)
                    del first_segment
                    sorted_results = sorted(results.items(), key=lambda x: x[1]['start_idx'])
                    last_end_idx = 0
                    for worker_id, result_data in sorted_results:
                        if os.path.exists(result_data['path']):
                            segment = torch.load(result_data['path'], map_location='cpu')
                            if last_end_idx > 0 and result_data['start_idx'] < last_end_idx:
                                overlap_frames = last_end_idx - result_data['start_idx']
                                if overlap_frames < segment.shape[0]:
                                    segment = segment[overlap_frames:]
                            streaming_writer.write_frames(segment)
                            last_end_idx = result_data['end_idx']
                            del segment
                            gc.collect()
                    streaming_writer.close()
                    log(f"[Multi-GPU] Streaming merge completed. Output saved to {output_path}", "finish")
                    return
    else:
        # 没有指定 --resume，默认覆盖模式：清理旧的checkpoint和临时文件
        checkpoint_dir = os.path.join('/tmp', 'flashvsr_checkpoints', video_dir_name)
        if os.path.exists(checkpoint_dir):
            clean_checkpoint(checkpoint_dir)
            log(f"[Multi-GPU] No --resume specified, cleaned old checkpoint. Starting fresh...", "info")
        
        # 清理旧的worker输出文件
        multi_gpu_tmp_dir = os.path.join('/tmp', 'flashvsr_multigpu', video_dir_name)
        if os.path.exists(multi_gpu_tmp_dir):
            import shutil
            try:
                shutil.rmtree(multi_gpu_tmp_dir)
                log(f"[Multi-GPU] Cleaned old worker output directory: {multi_gpu_tmp_dir}", "info")
            except Exception as e:
                log(f"[Multi-GPU] Warning: Failed to clean old worker output: {e}", "warning")
        
        # 创建新的checkpoint目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        completed_segments = {}  # 空列表，表示没有已完成的segments
    
    # 确定需要处理的segment
    segments_to_process = [i for i in range(len(segments)) if i not in completed_segments]
    
    # 优化：在主进程中分割frames并保存为临时文件，避免每个worker都读取完整视频
    # 进一步优化：保存后立即释放内存，避免同时保存多个segments
    import tempfile
    import numpy as np
    import gc
    temp_files = []
    try:
        if segments_to_process:
            log(f"[Multi-GPU] Saving {len(segments_to_process)} frame segments to temporary files (one at a time to save memory)...", "info")
            for i in segments_to_process:
                start_idx, end_idx = segments[i]
                log(f"[Multi-GPU] Processing segment {i}...", "info")
                frames_segment = frames[start_idx:end_idx].cpu().numpy()
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
                np.save(temp_file.name, frames_segment)
                temp_file.close()
                temp_files.append((i, temp_file.name))
                # 立即释放内存
                del frames_segment
                gc.collect()
                log(f"[Multi-GPU] Saved segment {i} ({end_idx-start_idx} frames) to {temp_file.name}", "info")
        else:
            log(f"[Multi-GPU] All segments already completed, skipping file saving", "info")
        
        # 释放frames tensor的内存（如果可能）
        # 注意：我们已经保存了 original_frame_count，所以可以安全删除 frames
        del frames
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        log(f"[Multi-GPU] Released main process memory", "info")
        
        # 使用多进程处理
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        processes = []
        
        # 准备参数字典
        args_dict = vars(args)
        
        # 启动worker进程（只处理未完成的segment）
        if segments_to_process:
            # 优化：错开启动时间，避免同时加载模型导致内存峰值过高
            # 等待模型加载完成后再启动下一个进程，避免OOM
            log(f"[Multi-GPU] Launching {len(segments_to_process)} worker processes (with model-loading-aware startup)...", "info")
            model_loaded_flags = {}  # 跟踪每个worker的模型加载状态
            
            # 创建segment索引到temp_file的映射
            segment_to_tempfile = {seg_idx: temp_file for seg_idx, temp_file in temp_files}
            
            for seg_idx in segments_to_process:
                start_idx, end_idx = segments[seg_idx]
                device = devices[seg_idx % num_gpus]
                temp_file = segment_to_tempfile[seg_idx]
                
                # 再次检查GPU内存
                used, total = get_gpu_memory_info(device)
                free = total - used
                log(f"[Multi-GPU] Starting worker {seg_idx} on {device} (available: {free:.2f} GB)", "info")
                
                # 检查系统内存
                sys_used, sys_total = get_system_memory_info()
                sys_free = sys_total - sys_used
                log(f"[Multi-GPU] System RAM: {sys_free:.2f} GB free before starting worker {seg_idx}", "info")
                if sys_free < 8.0:
                    log(f"[Multi-GPU] WARNING: Low system RAM ({sys_free:.2f} GB). Waiting longer before next worker...", "warning")
                
                p = ctx.Process(
                    target=_worker_process,
                    args=(seg_idx, device, temp_file, input_fps, args_dict, result_queue, start_idx, end_idx)
                )
                p.start()
                processes.append((seg_idx, p))
                model_loaded_flags[seg_idx] = False
                log(f"[Multi-GPU] Worker {seg_idx} process started (PID: {p.pid})", "info")
        else:
            log(f"[Multi-GPU] All segments already completed, skipping worker launch", "info")
            processes = []
        
        # 等待模型加载完成后再启动下一个worker（避免OOM）
        if segments_to_process and len(segments_to_process) > 1:
            for idx, (seg_idx, p) in enumerate(processes):
                if idx < len(processes) - 1:  # 最后一个进程不需要等待
                    log(f"[Multi-GPU] Waiting for worker {seg_idx} to finish loading model before starting next worker...", "info")
                    model_loaded = False
                    wait_start = time.time()
                    max_wait_time = 120  # 最多等待2分钟
                    
                    while not model_loaded and (time.time() - wait_start) < max_wait_time:
                        # 检查进程是否还活着
                        if not p.is_alive():
                            exit_code = p.exitcode
                            if exit_code == -9:
                                raise RuntimeError(f"Worker {seg_idx} was killed (SIGKILL) during model loading - likely OOM!")
                            elif exit_code is not None and exit_code != 0:
                                raise RuntimeError(f"Worker {seg_idx} exited unexpectedly during model loading (exit code: {exit_code})")
                        
                        # 检查是否有进度消息
                        try:
                            while not result_queue.empty():
                                result = result_queue.get(timeout=0.1)
                                if result.get('worker_id') == seg_idx:
                                    if result.get('type') == 'progress':
                                        stage = result.get('stage', '')
                                        message = result.get('message', '')
                                        # 检查是否模型加载完成（进入PROCESS阶段表示模型已加载）
                                        if stage == 'PROCESS' and 'Loading frame segment' in message:
                                            model_loaded = True
                                            model_loaded_flags[seg_idx] = True
                                            log(f"[Multi-GPU] Worker {seg_idx} model loaded successfully. Starting next worker...", "info")
                                            break
                                        elif stage == 'ERROR':
                                            raise RuntimeError(f"Worker {seg_idx} error during model loading: {message}")
                                    elif result.get('type') == 'heartbeat':
                                        # 心跳消息，继续等待
                                        pass
                        except:
                            pass
                        
                        # 如果还没加载完成，等待一小段时间
                        if not model_loaded:
                            time.sleep(1)
                    
                    if not model_loaded:
                        elapsed = time.time() - wait_start
                        log(f"[Multi-GPU] WARNING: Worker {seg_idx} model loading timeout after {elapsed:.1f}s. Proceeding anyway...", "warning")
                        # 继续启动下一个worker，但增加额外延迟
                        time.sleep(10)  # 额外等待10秒
                    else:
                        # 模型加载完成后，再等待几秒确保内存稳定
                        time.sleep(3)
        
        log(f"[Multi-GPU] All workers started. Waiting for results...", "info")
        log(f"[Multi-GPU] Monitoring progress (checking every 2 seconds)...", "info")
        
        # 收集结果（添加进度显示）
        # 流式合并：边处理边输出，减少内存占用
        use_streaming = getattr(args, 'streaming', False)  # 使用streaming参数
        results = completed_segments.copy()  # 从已完成的segment开始
        completed = len(completed_segments)
        total = len(segments)  # 总segment数
        last_progress_time = time.time()
        # 用于检测长期无进展且进程已退出的情况，避免死循环
        last_message_time = time.time()
        
        # 跟踪每个worker的进度（用于显示总体进度）
        worker_progress = {}  # {worker_id: {'tiles_processed': 0, 'total_tiles': 0, 'frames_processed': 0, 'total_frames': 0}}
        for i, (start, end) in enumerate(segments):
            worker_progress[i] = {
                'tiles_processed': 0,
                'total_tiles': 0,
                'frames_processed': 0,
                'total_frames': end - start,
                'progress_percent': 0.0
            }
        last_overall_progress_display = 0.0  # 上次显示的总体进度百分比
        overall_progress_display_interval = 5.0  # 每5%显示一次总体进度
        
        # 流式写入器（如果启用）
        streaming_writer = None
        if use_streaming:
            # 从第一个segment获取视频尺寸（需要等待第一个完成）
            log(f"[Multi-GPU] Streaming merge enabled: will write segments as they complete", "info")
        
        while completed < total:
            try:
                # 检查是否有进度消息
                if not result_queue.empty():
                    result = result_queue.get(timeout=0.1)
                    
                    if result.get('type') == 'progress':
                        # 显示进度消息
                        log(f"[Worker {result['worker_id']}@{result['device']}] {result['stage']}: {result['message']}", "info")
                        last_progress_time = time.time()
                        last_message_time = time.time()
                    elif result.get('type') == 'tile_progress':
                        # 处理tile进度消息，更新worker进度并显示总体进度
                        worker_id = result['worker_id']
                        if worker_id in worker_progress:
                            worker_progress[worker_id]['tiles_processed'] = result.get('tiles_processed', 0)
                            worker_progress[worker_id]['total_tiles'] = result.get('total_tiles', 0)
                            worker_progress[worker_id]['frames_processed'] = result.get('frames_processed', 0)
                            worker_progress[worker_id]['progress_percent'] = result.get('progress_percent', 0.0)
                            
                            # 计算总体进度
                            total_frames_processed = sum(wp['frames_processed'] for wp in worker_progress.values())
                            total_frames_all = sum(wp['total_frames'] for wp in worker_progress.values())
                            overall_percent = (total_frames_processed / total_frames_all) * 100 if total_frames_all > 0 else 0
                            
                            # 只在总体进度变化超过阈值时显示（避免日志过多）
                            if overall_percent - last_overall_progress_display >= overall_progress_display_interval or overall_percent >= 100.0:
                                log(f"[Multi-GPU] Overall progress: {total_frames_processed}/{total_frames_all} frames "
                                    f"({overall_percent:.1f}%) | Worker {worker_id}: {result.get('frames_processed', 0)}/{result.get('total_frames', 0)} frames "
                                    f"({result.get('progress_percent', 0.0):.1f}%)", "info")
                                last_overall_progress_display = overall_percent
                        
                        last_progress_time = time.time()
                        last_message_time = time.time()
                    elif result.get('type') == 'heartbeat':
                        # 心跳消息，不显示但更新时间戳
                        last_progress_time = time.time()
                        last_message_time = time.time()
                    elif 'success' in result:
                        # 这是最终结果
                        completed += 1
                        if result.get('success', False):
                            worker_id = result['worker_id']
                            results[worker_id] = {
                                'start_idx': result['start_idx'],
                                'end_idx': result['end_idx'],
                                'path': result['path']
                            }
                            
                            # 更新worker进度为100%
                            if worker_id in worker_progress:
                                worker_progress[worker_id]['frames_processed'] = worker_progress[worker_id]['total_frames']
                                worker_progress[worker_id]['progress_percent'] = 100.0
                            
                            # 计算并显示最终总体进度
                            total_frames_processed = sum(wp['frames_processed'] for wp in worker_progress.values())
                            total_frames_all = sum(wp['total_frames'] for wp in worker_progress.values())
                            overall_percent = (total_frames_processed / total_frames_all) * 100 if total_frames_all > 0 else 0
                            
                            log(f"[Multi-GPU] Worker {worker_id} completed ({completed}/{total}) | "
                                f"Overall progress: {total_frames_processed}/{total_frames_all} frames ({overall_percent:.1f}%)", "finish")
                            
                            # 更新检查点（每个segment完成时立即保存）
                            if checkpoint_dir:
                                checkpoint_data = load_checkpoint(checkpoint_dir) or {}
                                start_idx, end_idx = segments[worker_id]
                                segment_key = f"segment_{worker_id}_{start_idx}_{end_idx}"
                                checkpoint_data[segment_key] = {
                                    'start_idx': result['start_idx'],
                                    'end_idx': result['end_idx'],
                                    'path': result['path']
                                }
                                save_checkpoint(checkpoint_dir, checkpoint_data)
                            
                            # 流式合并：如果启用，按顺序写入segments
                            if use_streaming:
                                # 初始化写入器（使用第一个完成的segment获取视频尺寸）
                                # 但不立即写入，等待所有segments完成后再按start_idx顺序写入
                                if streaming_writer is None:
                                    # 加载segment获取尺寸（但不写入）
                                    segment = torch.load(result['path'], map_location='cpu')
                                    F, H, W, C = segment.shape
                                    # 获取输出路径
                                    output_path = getattr(args, 'output', None)
                                    if not output_path:
                                        output_path = os.path.join("/app/output", 
                                                                  os.path.basename(getattr(args, 'input', 'output.mp4')).replace(".mp4", "_out.mp4"))
                                    # 确保目录存在
                                    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
                                    
                                    streaming_writer = StreamingVideoWriter(
                                        output_path, fps=input_fps, height=H, width=W
                                    )
                                    log(f"[Multi-GPU] Streaming writer initialized: {W}x{H} @ {input_fps}fps", "info")
                                    log(f"[Multi-GPU] Will write segments in order by start_idx after all workers complete", "info")
                                    
                                    # 不立即写入，释放segment内存
                                    del segment
                                # 所有segments都先保存到results，最后按start_idx顺序写入
                        else:
                            error_msg = result.get('error', 'Unknown error')
                            log(f"[Multi-GPU] Worker {result['worker_id']} failed: {error_msg}", "error")
                            raise RuntimeError(f"Worker {result['worker_id']} failed: {error_msg}")
                    last_message_time = time.time()
                else:
                    # 如果没有消息，显示等待状态
                    elapsed = time.time() - last_progress_time
                    
                    # 检查进程状态（更频繁地检查，以便及时发现问题）
                    alive_count = sum(1 for _, p in processes if p.is_alive())
                    exit_codes = {seg_idx: p.exitcode for seg_idx, p in processes}
                    
                    # 检查是否有进程被杀死（-9 表示 SIGKILL，通常是 OOM）
                    killed_processes = [seg_idx for seg_idx, p in processes if p.exitcode == -9]
                    if killed_processes:
                        killed_devices = [devices[seg_idx % num_gpus] for seg_idx in killed_processes]
                        error_msg = f"Worker process(es) {killed_processes} were killed (SIGKILL, exit code -9). "
                        error_msg += f"This usually indicates Out-Of-Memory (OOM). Devices: {killed_devices}. "
                        error_msg += f"Exit codes: {exit_codes}\n"
                        
                        # 检查是GPU显存OOM还是系统内存OOM
                        gpu_memory_ok = True
                        for device in devices:
                            used, total = get_gpu_memory_info(device)
                            free = total - used
                            if free < 1.0:  # GPU显存几乎用尽
                                gpu_memory_ok = False
                        
                        # 检查系统内存
                        sys_used, sys_total = get_system_memory_info()
                        sys_free = sys_total - sys_used
                        
                        error_msg += "\n当前内存状态:\n"
                        error_msg += f"  系统RAM: {sys_used:.2f} GB / {sys_total:.2f} GB used ({sys_free:.2f} GB free)\n"
                        for device in devices:
                            used, total = get_gpu_memory_info(device)
                            free = total - used
                            error_msg += f"  {device}: {used:.2f} GB / {total:.2f} GB used ({free:.2f} GB free)\n"
                        
                        # 判断OOM类型
                        if not gpu_memory_ok:
                            error_msg += "\n诊断: GPU显存不足 (GPU OOM)\n"
                        elif sys_free < 2.0:
                            error_msg += "\n诊断: 系统内存不足 (RAM OOM) - 这是最可能的原因！\n"
                            error_msg += "  每个worker进程需要约4-8GB系统内存来加载模型和视频数据。\n"
                            error_msg += f"  当前只有 {sys_free:.2f} GB可用，不足以支持 {num_gpus} 个worker进程。\n"
                        else:
                            error_msg += "\n诊断: 可能是系统内存不足 (RAM OOM)，或模型加载时临时内存峰值过高\n"
                        
                        log(error_msg, "error")
                        log("OOM 解决方案建议:", "warning")
                        if sys_free < 8.0:
                            log("  [系统内存不足] 优先尝试以下方案:", "warning")
                            log("  1. 减少GPU数量（使用更少的GPU，减少并发worker进程）", "warning")
                            log("  2. 增加系统交换空间（swap space）", "warning")
                            log("  3. 关闭其他占用内存的程序", "warning")
                            log("  4. 使用更小的视频片段或降低视频分辨率", "warning")
                        log("  5. 降低tile_size参数（减少每次处理的瓦片大小）", "warning")
                        log("  6. 启用tiled_dit模式（如果可用）", "warning")
                        log("  7. 降低precision（使用bf16而不是fp32）", "warning")
                        raise RuntimeError(error_msg)
                    
                    # 如果进程数量减少，检查是否有异常退出的进程
                    # 注意：正常完成的进程（退出码0）不应该被报告为错误
                    # 退出码为0表示正常完成，消息可能在队列中等待处理，继续正常处理即可
                    if alive_count < len(processes) and completed < total:
                        # 只检查异常退出的进程（退出码非0且非None）
                        # 退出码为0表示正常完成，消息会在队列中被正常处理
                        dead_processes = [seg_idx for seg_idx, p in processes 
                                        if not p.is_alive() and p.exitcode is not None and p.exitcode != 0]
                        if dead_processes:
                            error_msg = f"Worker process(es) {dead_processes} exited unexpectedly. Exit codes: {exit_codes}"
                            log(error_msg, "error")
                            for seg_idx in dead_processes:
                                if exit_codes.get(seg_idx) == -9:
                                    log(f"Process {seg_idx} was killed (SIGKILL) - likely OOM", "error")
                                else:
                                    log(f"Process {seg_idx} exited with code {exit_codes.get(seg_idx)}", "error")
                            raise RuntimeError(error_msg)
                        
                        # 对于正常退出（退出码0）的进程，不报告错误
                        # 它们的成功消息会在队列中被正常处理，主循环会继续等待
                    
                    # 检查进程是否卡住：进程还在运行但没有消息超过60秒
                    if alive_count > 0 and (time.time() - last_message_time) > 60:
                        stuck_processes = [seg_idx for seg_idx, p in processes if p.is_alive()]
                        if stuck_processes:
                            error_msg = f"Worker process(es) {stuck_processes} appear to be stuck (no messages for 60s). "
                            error_msg += f"Processes are alive but not responding. Exit codes: {exit_codes}"
                            log(error_msg, "error")
                            log("建议：进程可能卡在模型加载或推理中，检查GPU显存和系统资源", "warning")
                            # 不立即抛出异常，再等待一段时间
                            if (time.time() - last_message_time) > 120:
                                raise RuntimeError(error_msg)
                    
                    if elapsed > 5:
                        log(f"[Multi-GPU] Still waiting... ({completed}/{total} completed, {elapsed:.1f}s since last update)", "info")
                        last_progress_time = time.time()
                        log(f"[Multi-GPU] {alive_count}/{len(processes)} processes are alive", "info")
                        
                        # 如果全部进程已经退出，但未完成，则中断并报告错误
                        if alive_count == 0 and completed < total:
                            error_msg = f"All worker processes exited prematurely. Exit codes: {exit_codes}"
                            log(error_msg, "error")
                            # 检查是否有 -9 退出码
                            if -9 in exit_codes:
                                log("检测到进程被 SIGKILL 杀死，可能是内存不足 (OOM)", "error")
                            # 尝试获取最后的错误信息
                            for seg_idx, p in processes:
                                if p.exitcode != 0:
                                    if p.exitcode == -9:
                                        log(f"Process {seg_idx} was killed (SIGKILL) - likely OOM", "error")
                                    else:
                                        log(f"Process {seg_idx} exited with code {p.exitcode}", "error")
                            raise RuntimeError(error_msg)
                        
                        # 如果长时间无任何消息且进程数量减少，也视为异常
                        if (time.time() - last_message_time) > 120 and completed < total:
                            error_msg = f"No progress messages for 120s; possible worker hang/crash. Exit codes: {exit_codes}"
                            log(error_msg, "error")
                            if -9 in exit_codes.values():
                                log("检测到进程被 SIGKILL 杀死，可能是内存不足 (OOM)", "error")
                            raise RuntimeError(error_msg)
                    time.sleep(0.5)
            except RuntimeError as e:
                # 重新抛出 RuntimeError，不要继续循环
                raise
            except Exception as e:
                # 其他异常才继续循环
                log(f"[Multi-GPU] Unexpected error in wait loop: {e}", "warning")
                time.sleep(0.5)
                continue
        
        # 等待所有进程完成
        log(f"[Multi-GPU] All workers finished. Waiting for processes to exit...", "info")
        for seg_idx, p in processes:
            p.join(timeout=30)
            if p.exitcode != 0:
                log(f"[Multi-GPU] Process {seg_idx} exited with code {p.exitcode}", "error")
            else:
                log(f"[Multi-GPU] Process {seg_idx} exited successfully", "info")
        
        # 更新检查点（保存所有完成的结果）
        if checkpoint_dir and results:
            checkpoint_data = load_checkpoint(checkpoint_dir) or {}
            for worker_id, result_data in results.items():
                start_idx, end_idx = segments[worker_id]
                segment_key = f"segment_{worker_id}_{start_idx}_{end_idx}"
                checkpoint_data[segment_key] = result_data
            save_checkpoint(checkpoint_dir, checkpoint_data)
            log(f"[Multi-GPU] Checkpoint updated: {len(results)} segments saved", "info")
        
        # 合并segments（流式或传统方式）
        if use_streaming and streaming_writer is not None:
            # 流式合并：按start_idx顺序写入所有segments（确保视频顺序正确）
            log(f"[Multi-GPU] Streaming merge: writing all segments in order by start_idx...", "info")
            
            # 按start_idx顺序排序所有segments
            sorted_results = sorted(results.items(), key=lambda x: x[1]['start_idx'])
            # 构建segments顺序信息（避免f-string中的反斜杠问题）
            segments_info = [f"worker_{wid}({r['start_idx']}-{r['end_idx']})" for wid, r in sorted_results]
            log(f"[Multi-GPU] Segments order: {segments_info}", "info")
            
            last_end_idx = 0
            
            for worker_id, result_data in sorted_results:
                # 检查文件是否存在
                if not os.path.exists(result_data['path']):
                    log(f"[Multi-GPU] WARNING: Segment file for worker {worker_id} not found: {result_data['path']}", "warning")
                    continue
                
                # 加载segment
                segment = torch.load(result_data['path'], map_location='cpu')
                
                # 处理overlap：跳过与上一个segment重叠的部分
                if last_end_idx > 0 and result_data['start_idx'] < last_end_idx:
                    overlap_frames = last_end_idx - result_data['start_idx']
                    if overlap_frames < segment.shape[0]:
                        segment = segment[overlap_frames:]
                        log(f"[Multi-GPU] Skipped {overlap_frames} overlap frames for segment {worker_id} (start_idx={result_data['start_idx']}, last_end_idx={last_end_idx})", "info")
                    elif overlap_frames >= segment.shape[0]:
                        log(f"[Multi-GPU] WARNING: Segment {worker_id} is completely overlapped, skipping", "warning")
                        del segment
                        try:
                            os.remove(result_data['path'])
                        except:
                            pass
                        continue
                
                # 写入segment
                if segment.shape[0] > 0:
                    streaming_writer.write_frames(segment)
                    log(f"[Multi-GPU] Written segment {worker_id} (frames {result_data['start_idx']}-{result_data['end_idx']}, {segment.shape[0]} frames written)", "info")
                    last_end_idx = result_data['end_idx']
                else:
                    log(f"[Multi-GPU] WARNING: Segment {worker_id} has 0 frames after overlap removal", "warning")
                
                # 删除临时文件
                try:
                    os.remove(result_data['path'])
                except:
                    pass
                del segment
            
            # 关闭写入器
            streaming_writer.close()
            process_time = time.time() - process_start
            log(f"[Multi-GPU] Streaming merge completed", "finish")
            log(f"[Multi-GPU] Processing time: {format_duration(process_time)}", "finish")
            
            # 流式模式下，返回None（视频已保存到文件）
            return None
        else:
            # 流式合并方式：逐个加载和处理segments，避免OOM
            log(f"[Multi-GPU] Streaming merge: merging {len(results)} segments (avoiding OOM)...", "info")
            loaded_files = []  # 跟踪已加载的文件，只有在合并成功后才删除
            
            try:
                # 构建segments信息列表（start_idx, end_idx, path）
                segments_info = []
                for i in sorted(results.keys()):
                    path = results[i]['path']
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"Segment file not found: {path}")
                    segments_info.append((results[i]['start_idx'], results[i]['end_idx'], path))
                    loaded_files.append(path)
                
                # 使用流式合并函数
                log(f"[Multi-GPU] Starting streaming merge of {len(segments_info)} segments...", "info")
                merged_output = merge_video_segments_streaming(segments_info, original_frame_count)
                log(f"[Multi-GPU] Merged output shape: {merged_output.shape}", "info")
                
                # 注意：不在这里删除worker文件，让main函数在保存视频成功后再删除
                # 这样可以确保在视频成功保存之前，临时文件不会被删除
                log(f"[Multi-GPU] Worker files preserved (will be cleaned after video is saved):", "info")
                for path in loaded_files:
                    if os.path.exists(path):
                        log(f"[Multi-GPU]   - {path}", "info")
                
                process_time = time.time() - process_start
                log(f"[Multi-GPU] Successfully processed and merged {num_gpus} segments", "finish")
                log(f"[Multi-GPU] Processing time: {format_duration(process_time)}", "finish")
                return merged_output
                
            except Exception as merge_error:
                log(f"[Multi-GPU] ERROR: Failed to merge segments: {merge_error}", "error")
                import traceback
                log(f"[Multi-GPU] Traceback: {traceback.format_exc()}", "error")
                # 保留文件以便手动恢复
                log(f"[Multi-GPU] Segment files preserved for manual recovery:", "warning")
                for path in loaded_files:
                    if os.path.exists(path):
                        log(f"[Multi-GPU]   - {path}", "warning")
                # 也保留未加载的文件
                for i in sorted(results.keys()):
                    path = results[i]['path']
                    if path not in loaded_files and os.path.exists(path):
                        log(f"[Multi-GPU]   - {path}", "warning")
                raise
    
    finally:
        # 确保所有进程都被正确终止和清理
        for seg_idx, p in processes:
            try:
                if p.is_alive():
                    log(f"[Multi-GPU] Terminating process {seg_idx} (PID: {p.pid})", "warning")
                    try:
                        p.terminate()
                        p.join(timeout=5)
                        if p.is_alive():
                            log(f"[Multi-GPU] Force killing process {seg_idx} (PID: {p.pid})", "warning")
                            p.kill()
                            p.join(timeout=2)
                            if p.is_alive():
                                log(f"[Multi-GPU] Process {seg_idx} still alive after kill, may need manual cleanup", "error")
                    except Exception as e:
                        log(f"[Multi-GPU] Error terminating process {seg_idx}: {e}", "warning")
                else:
                    # 即使进程已退出，也确保 join 完成
                    try:
                        p.join(timeout=1)
                    except:
                        pass
            except Exception as e:
                log(f"[Multi-GPU] Error cleaning up process {seg_idx}: {e}", "warning")
        
        # 清理临时文件
        for seg_idx, temp_file_path in temp_files:
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    log(f"[Multi-GPU] Cleaned up temporary file: {temp_file_path}", "info")
            except Exception as e:
                log(f"[Multi-GPU] Failed to remove temporary file {temp_file_path}: {e}", "warning")
        
        # 清理队列：清空队列并关闭
        try:
            # 先等待所有进程完全退出，确保没有进程在使用队列
            time.sleep(0.5)  # 短暂等待，确保所有进程都已完成
            
            # 清空队列中剩余的所有项目（使用非阻塞方式）
            items_cleared = 0
            max_clear_attempts = 100  # 限制清空尝试次数，避免无限循环
            clear_attempts = 0
            while clear_attempts < max_clear_attempts:
                try:
                    result_queue.get_nowait()
                    items_cleared += 1
                    clear_attempts += 1
                except:
                    break
            if items_cleared > 0:
                log(f"[Multi-GPU] Cleared {items_cleared} remaining items from queue", "info")
            
            # 关闭队列以释放资源
            result_queue.close()
            
            # 等待队列的后台线程完成
            # 使用超时机制避免无限等待
            import threading
            join_completed = threading.Event()
            
            def join_with_timeout():
                try:
                    result_queue.join_thread()
                    join_completed.set()
                except Exception as e:
                    log(f"[Multi-GPU] Queue join_thread error: {e}", "warning")
                    join_completed.set()
            
            join_thread = threading.Thread(target=join_with_timeout, daemon=True)
            join_thread.start()
            join_thread.join(timeout=5.0)  # 最多等待5秒
            
            if not join_completed.is_set():
                log(f"[Multi-GPU] Queue join_thread timeout, continuing cleanup", "warning")
        except Exception as e:
            log(f"[Multi-GPU] Error cleaning up queue: {e}", "warning")
        finally:
            # 确保队列对象被删除
            try:
                del result_queue
                # 强制垃圾回收，确保资源被释放
                import gc
                gc.collect()
            except:
                pass

def _worker_process(worker_id: int, device: str, segment_file: str,
                   input_fps: float, args_dict: dict, result_queue: mp.Queue,
                   start_idx: int = 0, end_idx: int = 0):
    """Worker process for multi-GPU processing (separate function to avoid import issues).
    
    Args:
        worker_id: Worker process ID
        device: CUDA device string (e.g., 'cuda:0')
        segment_file: Path to temporary file containing pre-split frame segment (numpy .npy file)
        input_fps: Input video FPS
        args_dict: Arguments dictionary
        result_queue: Queue for reporting progress and results
        start_idx: Start frame index of this segment
        end_idx: End frame index of this segment
    """
    heartbeat_timer = None
    try:
        import sys
        import threading
        sys.stdout.flush()  # 确保输出立即显示
        sys.stderr.flush()
        
        # 添加进度报告
        def report_progress(stage, message):
            """向主进程报告进度"""
            try:
                result_queue.put({
                    'worker_id': worker_id,
                    'type': 'progress',
                    'stage': stage,
                    'message': message,
                    'device': device
                }, block=False)
            except:
                pass
            print(f"[Worker {worker_id}@{device}] {stage}: {message}", flush=True)
            sys.stdout.flush()
        
        # 添加心跳机制：每30秒发送一次心跳消息
        def send_heartbeat():
            """定期发送心跳消息，表明进程仍在运行"""
            try:
                result_queue.put({
                    'worker_id': worker_id,
                    'type': 'heartbeat',
                    'stage': 'HEARTBEAT',
                    'message': 'Process is alive',
                    'device': device
                }, block=False)
            except:
                pass
        
        def start_heartbeat():
            """启动心跳定时器"""
            nonlocal heartbeat_timer
            if heartbeat_timer is not None:
                heartbeat_timer.cancel()
            heartbeat_timer = threading.Timer(30.0, heartbeat_loop)
            heartbeat_timer.daemon = True
            heartbeat_timer.start()
        
        def heartbeat_loop():
            """心跳循环"""
            send_heartbeat()
            start_heartbeat()  # 重新启动定时器
        
        report_progress("INIT", "Worker process started")
        start_heartbeat()  # 启动心跳
        
        # 重新导入必要的模块（在子进程中）
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 将项目根目录添加到 sys.path，而不是 src 目录（与主文件保持一致）
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        
        report_progress("INIT", "Importing modules...")
        
        # 设置当前进程使用的GPU
        if device.startswith("cuda:"):
            torch.cuda.set_device(int(device.split(":")[1]))
            report_progress("INIT", f"Set CUDA device to {device}")
            
            # 检查GPU内存
            try:
                used, total = get_gpu_memory_info(device)
                free = total - used
                report_progress("INIT", f"GPU memory: {free:.2f} GB free / {total:.2f} GB total (used: {used:.2f} GB)")
                if free < 2.0:
                    report_progress("WARNING", f"Low GPU memory ({free:.2f} GB). OOM risk!")
            except Exception as e:
                report_progress("WARNING", f"Could not check GPU memory: {e}")
        
        # 解析参数
        from argparse import Namespace
        args = Namespace(**args_dict)
        args.device = device
        
        # 在 worker 进程中打印参数（帮助调试）
        print(f"[Worker {worker_id}@{device}] [参数检查] 接收到的 tile 相关参数:", flush=True)
        print(f"[Worker {worker_id}@{device}]   tiled_dit = {args.tiled_dit} (type: {type(args.tiled_dit).__name__})", flush=True)
        print(f"[Worker {worker_id}@{device}]   tiled_vae = {args.tiled_vae} (type: {type(args.tiled_vae).__name__})", flush=True)
        print(f"[Worker {worker_id}@{device}]   tile_size = {args.tile_size}", flush=True)
        print(f"[Worker {worker_id}@{device}]   tile_overlap = {args.tile_overlap}", flush=True)
        sys.stdout.flush()
        
        # 导入必要的模块（使用 src. 前缀，与主文件保持一致）
        report_progress("LOAD", "Loading model modules...")
        # 在worker进程中，需要重新导入get_gpu_memory_info（因为spawn模式会重新导入模块）
        # 但由于函数定义在模块顶层，应该可以直接访问
        from src.models.utils import clean_vram
        from src.models import wan_video_dit
        from src.models.model_manager import ModelManager
        from src.models.TCDecoder import build_tcdecoder
        from src.models.utils import Buffer_LQ4x_Proj
        from src.pipelines.flashvsr_full import FlashVSRFullPipeline
        from src.pipelines.flashvsr_tiny import FlashVSRTinyPipeline
        from src.pipelines.flashvsr_tiny_long import FlashVSRTinyLongPipeline
        
        # 确保可以访问get_gpu_memory_info（在spawn模式下可能需要从模块获取）
        try:
            # 尝试直接使用（如果函数在模块作用域中）
            _ = get_gpu_memory_info
        except NameError:
            # 如果不可用，从当前模块获取
            import sys
            current_module = sys.modules.get('__main__') or sys.modules.get('infer_video')
            if current_module and hasattr(current_module, 'get_gpu_memory_info'):
                get_gpu_memory_info = current_module.get_gpu_memory_info
            else:
                # 如果还是找不到，定义一个简单的版本
                def get_gpu_memory_info(device: str):
                    if device.startswith("cuda:"):
                        try:
                            idx = int(device.split(":")[1])
                            torch.cuda.set_device(idx)
                            total = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
                            reserved = torch.cuda.memory_reserved(idx) / (1024**3)
                            return reserved, total
                        except:
                            return 0.0, 0.0
                    return 0.0, 0.0
        
        # 处理attention_mode
        # 注意：最新版本的block_sparse_attention（2025/12更新）已支持Blackwell (sm_120)
        if args.attention_mode == "sparse_sage_attention":
            wan_video_dit.USE_BLOCK_ATTN = False
        else:
            wan_video_dit.USE_BLOCK_ATTN = True
            # 如果BLOCK_ATTN_AVAILABLE为False，说明模块未正确安装或编译
            if not wan_video_dit.BLOCK_ATTN_AVAILABLE:
                log(f"[Worker {worker_id}] Warning: block_sparse_attention not available. Auto-switching to sparse_sage_attention", "warning")
                wan_video_dit.USE_BLOCK_ATTN = False
        
        # 初始化pipeline
        report_progress("LOAD", "Initializing pipeline...")
        # 根据 model_ver 参数构建模型路径
        model_path = f"/app/models/v{args.model_ver}"
        
        # 验证模型路径和文件
        if not os.path.exists(model_path):
            raise RuntimeError(f"[Worker {worker_id}] Model directory does not exist: {model_path}")
        if not os.path.isdir(model_path):
            raise RuntimeError(f"[Worker {worker_id}] Model path is not a directory: {model_path}")
        
        ckpt_path = os.path.join(model_path, "diffusion_pytorch_model_streaming_dmd.safetensors")
        vae_path = os.path.join(model_path, "Wan2.1_VAE.pth")
        lq_path = os.path.join(model_path, "LQ_proj_in.ckpt")
        tcd_path = os.path.join(model_path, "TCDecoder.ckpt")
        prompt_path = os.path.join(script_dir, "posi_prompt.pth")
        
        # 验证必需的文件是否存在
        required_files = [ckpt_path]
        if args.mode == "full":
            required_files.append(vae_path)
        else:
            required_files.extend([lq_path, tcd_path])
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            raise RuntimeError(f"[Worker {worker_id}] Missing required model files:\n  " + "\n  ".join(missing_files))
        
        # 验证文件大小和可读性
        for file_path in required_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    raise RuntimeError(f"[Worker {worker_id}] Model file is empty: {file_path}")
                # 检查文件是否可读
                if not os.access(file_path, os.R_OK):
                    raise RuntimeError(f"[Worker {worker_id}] Model file is not readable: {file_path}")
                report_progress("LOAD", f"  {os.path.basename(file_path)}: {file_size / (1024**2):.2f} MB")
        
        report_progress("LOAD", f"Model directory: {model_path} (verified)")
        
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        dtype = dtype_map.get(args.precision, torch.bfloat16)
        
        # 优化：添加小延迟，进一步错开模型加载时间
        import random
        delay = random.uniform(0, 1)  # 0-1秒随机延迟
        time.sleep(delay)
        
        report_progress("LOAD", "Loading model weights...")
        # 优化：直接加载到目标设备，避免先加载到 CPU 再转移（减少内存峰值）
        # 但 ModelManager 需要先加载到 CPU，所以我们在加载后立即清理 CPU 缓存
        # 在加载前先清理内存
        import gc
        gc.collect()
        
        mm = ModelManager(torch_dtype=dtype, device="cpu")
        if args.mode == "full":
            mm.load_models([ckpt_path, vae_path])
            pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
            pipe.vae.model.encoder = None
            pipe.vae.model.conv1 = None
        else:
            mm.load_models([ckpt_path])
            pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device) if args.mode == "tiny" else FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
            multi_scale_channels = [512, 256, 128, 128]
            pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, device=device, dtype=dtype, new_latent_channels=16+768)
            pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location=device), strict=False)
            pipe.TCDecoder.clean_mem()
        
        # 立即清理 CPU 内存，释放 ModelManager 占用的内存
        del mm
        gc.collect()
        # 强制Python垃圾回收，释放更多内存
        import sys
        if hasattr(sys, 'getsizeof'):
            # 触发更彻底的垃圾回收
            for _ in range(3):
                gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 报告模型加载完成，通知主进程可以启动下一个worker
        report_progress("LOAD", "Model weights loaded, cleaning up temporary memory...")
        
        report_progress("LOAD", "Loading additional components...")
        pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
        # 优化：直接加载到目标设备，避免中间占用 CPU 内存
        lq_state_dict = torch.load(lq_path, map_location=device)
        pipe.denoising_model().LQ_proj_in.load_state_dict(lq_state_dict, strict=True)
        del lq_state_dict
        gc.collect()
        
        pipe.to(device, dtype=dtype)
        pipe.enable_vram_management(num_persistent_param_in_dit=None)
        pipe.init_cross_kv(prompt_path=prompt_path)
        pipe.load_models_to_device(["dit", "vae"])
        
        # 再次清理内存
        gc.collect()
        for _ in range(2):
            gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 报告模型完全加载完成（这是关键消息，主进程会等待这个）
        report_progress("PROCESS", "Pipeline initialized. Loading frame segment...")

        # 从临时文件加载预分割的帧segment（避免读取完整视频）
        import numpy as np
        try:
            frames_segment_np = np.load(segment_file)
            frames_segment = torch.from_numpy(frames_segment_np).float()
            del frames_segment_np
            report_progress("PROCESS", f"Loaded {frames_segment.shape[0]} frames from segment file")
        except Exception as e:
            error_msg = f"[Worker {worker_id}] Failed to load segment from {segment_file}: {e}"
            report_progress("ERROR", error_msg)
            raise RuntimeError(error_msg) from e

        # 执行推理
        report_progress("PROCESS", f"Starting inference on {frames_segment.shape[0]} frames...")
        
        # 标记这是worker进程，用于进度报告
        args._is_worker = True
        args._worker_id = worker_id
        args._result_queue = result_queue
        args._total_frames = frames_segment.shape[0]
        args._start_idx = start_idx
        args._end_idx = end_idx
        
        # 确保run_inference函数可用（在spawn模式下可能需要重新导入）
        try:
            # 尝试直接使用
            _ = run_inference
        except NameError:
            # 如果不可用，从当前模块获取
            import sys
            current_module = sys.modules.get(__name__) or sys.modules.get('__main__')
            if current_module and hasattr(current_module, 'run_inference'):
                run_inference = current_module.run_inference
            else:
                # 如果还是找不到，尝试导入
                try:
                    from infer_video import run_inference
                except ImportError:
                    # 最后尝试：从文件路径导入
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("infer_video", "/app/FlashVSR_Ultra_Fast/infer_video.py")
                    if spec and spec.loader:
                        infer_video_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(infer_video_module)
                        run_inference = infer_video_module.run_inference
        
        try:
            output = run_inference(pipe, frames_segment, device, dtype, args)
        except Exception as inference_error:
            # 捕获推理过程中的错误
            import traceback
            error_msg = f"Inference failed: {str(inference_error)}\n{traceback.format_exc()}"
            report_progress("ERROR", error_msg)
            raise RuntimeError(error_msg) from inference_error

        # 保存到临时文件，返回路径
        # 使用基于视频名称和放大倍数的目录名
        # 注意：在worker进程中，input路径应该从args_dict传递过来
        input_path = getattr(args, 'input', None)
        if input_path and input_path != 'unknown' and os.path.exists(input_path):
            video_dir_name = get_video_based_dir_name(input_path, args.scale)
        else:
            # 如果无法获取input路径，使用worker信息作为fallback
            # 这种情况下，主进程会使用正确的目录名
            video_dir_name = f"worker_{start_idx}_{end_idx}_{args.scale}x"
        tmp_dir = os.path.join('/tmp', 'flashvsr_multigpu', video_dir_name)
        os.makedirs(tmp_dir, exist_ok=True)
        out_path = os.path.join(tmp_dir, f"worker_{worker_id}_{uuid.uuid4().hex}.pt")
        torch.save(output, out_path)

        result_queue.put({
            'worker_id': worker_id,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'path': out_path,
            'success': True
        })

        report_progress("DONE", f"Results saved to {out_path}")
        
        # 清理worker标记
        if hasattr(args, '_is_worker'):
            delattr(args, '_is_worker')
        if hasattr(args, '_worker_id'):
            delattr(args, '_worker_id')
        if hasattr(args, '_result_queue'):
            delattr(args, '_result_queue')
        if hasattr(args, '_total_frames'):
            delattr(args, '_total_frames')
        if hasattr(args, '_start_idx'):
            delattr(args, '_start_idx')
        if hasattr(args, '_end_idx'):
            delattr(args, '_end_idx')

        del pipe, output, frames_segment
        clean_vram()
        
        # 停止心跳
        if heartbeat_timer is not None:
            heartbeat_timer.cancel()
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[Worker {worker_id} ERROR] {error_msg}", flush=True)
        sys.stderr.write(f"[Worker {worker_id} ERROR] {error_msg}\n")
        sys.stderr.flush()
        try:
            result_queue.put({
                'worker_id': worker_id,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'error': error_msg,
                'success': False
            }, timeout=10)
        except Exception as queue_error:
            print(f"[Worker {worker_id} ERROR] Failed to put error in queue: {queue_error}", flush=True)
            sys.stderr.write(f"[Worker {worker_id} ERROR] Failed to put error in queue: {queue_error}\n")
            sys.stderr.flush()
    except BaseException as e:
        # 捕获所有异常，包括 SystemExit 和 KeyboardInterrupt
        import traceback
        error_msg = f"BaseException: {str(e)}\n{traceback.format_exc()}"
        print(f"[Worker {worker_id} FATAL] {error_msg}", flush=True)
        sys.stderr.write(f"[Worker {worker_id} FATAL] {error_msg}\n")
        sys.stderr.flush()
        # 停止心跳
        if heartbeat_timer is not None:
            heartbeat_timer.cancel()
        raise
    finally:
        # 确保心跳定时器被停止
        if heartbeat_timer is not None:
            heartbeat_timer.cancel()

# ==============================================================
#                     Padding for Model Input
# ==============================================================

def pad_to_window_multiple(frames: torch.Tensor, window=(2, 8, 8)):
    """Pad tensor so that F/H/W are multiples of given window size."""
    win_t, win_h, win_w = window
    shape = list(frames.shape)
    if len(shape) == 4:  # (F,C,H,W)
        f, c, h, w = shape
        prefix = ()
    elif len(shape) == 5:  # (B,C,F,H,W)
        prefix = (0, 1)
        f, c, h, w = shape[2:]
    else:
        raise ValueError(f"Unexpected input shape: {shape}")

    new_f = math.ceil(f / win_t) * win_t
    new_h = math.ceil(h / win_h) * win_h
    new_w = math.ceil(w / win_w) * win_w
    pad_f, pad_h, pad_w = new_f - f, new_h - h, new_w - w

    if pad_f == 0 and pad_h == 0 and pad_w == 0:
        print(f"[INFO] Already aligned to window multiples ({f},{h},{w})")
        return frames, (f, h, w)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    frames = F.pad(frames, (pad_left, pad_right, pad_top, pad_bottom))

    if pad_f > 0:
        if len(shape) == 4:
            last = frames[-1:].repeat(pad_f, 1, 1, 1)
            frames = torch.cat([frames, last], dim=0)
        elif len(shape) == 5:
            last = frames[:, :, -1:].repeat(1, 1, pad_f, 1, 1)
            frames = torch.cat([frames, last], dim=2)

    print(f"[INFO] Padded to ({new_f},{new_h},{new_w}) for window compatibility")
    return frames, (f, h, w)

def pad_frames_auto(input_data, window=(2, 8, 8)):
    """Detects input type (Tensor, dict, or list) and applies padding."""
    if isinstance(input_data, torch.Tensor):
        return pad_to_window_multiple(input_data, window)
    elif isinstance(input_data, dict):
        for k, v in input_data.items():
            if isinstance(v, torch.Tensor):
                padded, orig = pad_to_window_multiple(v, window)
                input_data[k] = padded
                return input_data, orig
    elif isinstance(input_data, (list, tuple)):
        for i in range(len(input_data)):
            if isinstance(input_data[i], torch.Tensor):
                padded, orig = pad_to_window_multiple(input_data[i], window)
                input_data[i] = padded
                return input_data, orig
    raise TypeError(f"[ERROR] Unsupported input type for padding: {type(input_data)}")

# ==============================================================
#                     FlashVSR Pipeline
# ==============================================================

def init_pipeline(mode, device, dtype, model_dir):
    """Initialize FlashVSR pipeline and load model weights."""
    model_path = model_dir
    ckpt_path = os.path.join(model_path, "diffusion_pytorch_model_streaming_dmd.safetensors")
    vae_path = os.path.join(model_path, "Wan2.1_VAE.pth")
    lq_path = os.path.join(model_path, "LQ_proj_in.ckpt")
    tcd_path = os.path.join(model_path, "TCDecoder.ckpt")
    prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "posi_prompt.pth")

    for p in [ckpt_path, vae_path, lq_path, tcd_path]:
        if not os.path.exists(p):
            raise RuntimeError(f"Missing model file: {p}")

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
        pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, device=device, dtype=dtype, new_latent_channels=16+768)
        pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location=device), strict=False)
        pipe.TCDecoder.clean_mem()

    pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to(device)
    pipe.to(device, dtype=dtype)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path=prompt_path)
    pipe.load_models_to_device(["dit", "vae"])

    log(f"Pipeline initialized successfully in mode: {mode}", "finish")
    return pipe

def process_tile_batch(pipe, frames, device, dtype, args, tile_batch: List[Tuple[int, int, int, int]], batch_idx: int):
    """Process a batch of tiles and return results."""
    N, H, W, C = frames.shape
    num_aligned_frames = largest_8n1_leq(N + 4) - 4

    results = []
    
    for tile_idx, (x1, y1, x2, y2) in enumerate(tile_batch):
        input_tile = frames[:, y1:y2, x1:x2, :]

        LQ_tile, th, tw, F = prepare_input_tensor(input_tile, device, scale=args.scale, dtype=dtype)
        if "long" not in args.mode:
            LQ_tile = LQ_tile.to(device)

        topk_ratio = args.sparse_ratio * 768 * 1280 / (th * tw)

        with torch.no_grad():
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

def run_inference_segmented(pipe, frames, device, dtype, args, max_segment_frames=None):
    """分段处理视频：将视频分成多个sub-segments，每个独立处理并保存，最后拼接。
    
    这是streaming模式的替代方案，更简单、更可靠。
    只在用户明确启用 --segmented 时使用。
    
    支持在 multi_gpu worker 模式下使用，此时会记录相对于原始视频的绝对帧范围。
    
    Args:
        max_segment_frames: 每个sub-segment的最大帧数（None表示自动计算）
    
    Returns:
        处理后的视频tensor (F, H, W, C)
    """
    N, H, W, C = frames.shape
    N_original = N
    
    # 检查是否在 multi_gpu worker 模式下（需要记录绝对帧范围）
    is_worker_mode = hasattr(args, '_is_worker') and args._is_worker
    worker_start_idx = getattr(args, '_start_idx', 0)  # worker 接收的 segment 在原始视频中的起始帧
    worker_end_idx = getattr(args, '_end_idx', N)  # worker 接收的 segment 在原始视频中的结束帧
    
    # 自动计算合适的segment大小（基于可用内存）
    # 优先使用用户指定的值
    if max_segment_frames is None:
        max_segment_frames = getattr(args, 'max_segment_frames', None)
    
    if max_segment_frames is None:
        try:
            sys_used, sys_total = get_system_memory_info()
            sys_free = sys_total - sys_used
            
            # 计算每个segment的canvas内存需求
            # 使用可用内存的40%作为每个segment的canvas（更激进，减少segment数量）
            segment_canvas_memory_gb = sys_free * 0.40
            segment_canvas_memory_gb = max(10.0, min(segment_canvas_memory_gb, 60.0))  # 10-60GB（提高上限）
            
            log(f"[Segmented] Available memory: {sys_free:.2f} GB, using {segment_canvas_memory_gb:.2f} GB per segment", "info")
            
            # 计算可以处理的帧数
            max_segment_frames = int((segment_canvas_memory_gb * (1024**3)) / 
                                    (H * args.scale * W * args.scale * C * 2 * 2))  # 2个canvas
            max_segment_frames = max(21, max_segment_frames)  # 至少21帧
            max_segment_frames = largest_8n1_leq(max_segment_frames)  # 对齐到8n+1
            max_segment_frames = min(max_segment_frames, 1000)  # 最多1000帧（提高上限）
            
            log(f"[Segmented] Calculated max_segment_frames: {max_segment_frames} frames", "info")
        except Exception as e:
            log(f"[Segmented] Failed to calculate segment size: {e}, using default", "warning")
            max_segment_frames = 200  # 提高默认值（从100到200）
    
    # 计算需要多少个sub-segments
    num_segments = (N + max_segment_frames - 1) // max_segment_frames
    segment_overlap = getattr(args, 'segment_overlap', 2)
    
    log(f"[Segmented] Processing video in {num_segments} sub-segments "
        f"(max {max_segment_frames} frames per segment, overlap: {segment_overlap} frames)", "info")
    
    # 创建基于视频名称和放大倍数的临时目录名，支持断点续跑
    # 如果在worker模式下，必须使用worker信息作为标识，避免不同worker使用同一个目录
    if is_worker_mode:
        # worker模式下，始终使用worker信息作为标识，确保每个worker有独立的目录
        video_dir_name = f"worker_{worker_start_idx}_{worker_end_idx}_{args.scale}x"
    else:
        # 非worker模式：使用基于视频名称的目录名
        input_path = getattr(args, 'input', None)
        if input_path and input_path != 'unknown' and os.path.exists(input_path):
            video_dir_name = get_video_based_dir_name(input_path, args.scale)
        else:
            # 如果无法获取有效的input路径，使用hash作为fallback
            import hashlib
            segment_key_data = {
                'input': os.path.abspath(getattr(args, 'input', 'unknown')),
                'scale': args.scale,
                'mode': args.mode,
                'tile_size': args.tile_size,
                'tile_overlap': args.tile_overlap,
                'tiled_dit': args.tiled_dit,
                'tiled_vae': args.tiled_vae,
                'max_segment_frames': max_segment_frames,
                'num_segments': num_segments,
                'N_original': N_original
            }
            segment_id = hashlib.md5(json.dumps(segment_key_data, sort_keys=True).encode()).hexdigest()
            video_dir_name = segment_id
    
    tmp_dir = os.path.join('/tmp', 'flashvsr_segments', video_dir_name)
    os.makedirs(tmp_dir, exist_ok=True)
    log(f"[Segmented] Using directory: {tmp_dir}", "info")
    
    # 检查是否已有已完成的segment文件（断点续跑）
    segment_files = []
    completed_segments = 0
    for seg_idx in range(num_segments):
        start_frame = seg_idx * max_segment_frames
        end_frame = min((seg_idx + 1) * max_segment_frames, N)
        segment_file = os.path.join(tmp_dir, f"segment_{seg_idx:04d}.pt")
        metadata_file = os.path.join(tmp_dir, f"segment_{seg_idx:04d}.json")
        
        if os.path.exists(segment_file):
            try:
                # 验证文件是否完整（检查文件大小）
                file_size = os.path.getsize(segment_file)
                if file_size > 0:
                    # 如果有metadata文件，从中读取准确的帧范围（优先使用绝对帧范围）
                    if os.path.exists(metadata_file):
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            # 优先使用绝对帧范围（如果存在）
                            start_frame = metadata.get('start_frame', metadata.get('relative_start_frame', start_frame))
                            end_frame = metadata.get('end_frame', metadata.get('relative_end_frame', end_frame))
                        except:
                            pass
                    
                    segment_files.append((seg_idx, segment_file, start_frame, end_frame))
                    completed_segments += 1
                    log(f"[Segmented] Found existing segment {seg_idx + 1}/{num_segments} (frames {start_frame}-{end_frame}) at {segment_file}", "info")
            except:
                pass
    
    # 检查是否应该进行断点续传（只有在明确指定--resume时才续传）
    should_resume = getattr(args, 'resume', False)
    
    if completed_segments == num_segments and should_resume:
        log(f"[Segmented] All {num_segments} segments already completed! Merging results...", "info")
        # 所有segment都已完成，直接合并结果
    elif completed_segments == num_segments and not should_resume:
        # 没有指定--resume，但找到了已完成的segments，清理并重新开始
        log(f"[Segmented] Found {num_segments} completed segments but --resume not specified. Cleaning and starting fresh...", "info")
        import shutil
        try:
            shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)
            log(f"[Segmented] Cleaned old segments directory: {tmp_dir}", "info")
        except Exception as e:
            log(f"[Segmented] Warning: Failed to clean old segments: {e}", "warning")
        completed_segments = 0
        segment_files = []
    elif completed_segments > 0 and should_resume:
        log(f"[Segmented] Resume mode: Found {completed_segments}/{num_segments} completed segments, processing remaining segments...", "info")
    elif completed_segments > 0 and not should_resume:
        # 没有指定--resume，但找到了部分已完成的segments，清理并重新开始
        log(f"[Segmented] Found {completed_segments}/{num_segments} completed segments but --resume not specified. Cleaning and starting fresh...", "info")
        import shutil
        try:
            shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)
            log(f"[Segmented] Cleaned old segments directory: {tmp_dir}", "info")
        except Exception as e:
            log(f"[Segmented] Warning: Failed to clean old segments: {e}", "warning")
        completed_segments = 0
        segment_files = []
    else:
        log(f"[Segmented] Starting fresh processing of all segments...", "info")
    
    try:
        # 处理每个未完成的sub-segment
        for seg_idx in range(num_segments):
            # 检查是否已存在
            start_frame = seg_idx * max_segment_frames
            end_frame = min((seg_idx + 1) * max_segment_frames, N)
            segment_file = os.path.join(tmp_dir, f"segment_{seg_idx:04d}.pt")
            
            # 如果segment已存在且完整，跳过处理
            metadata_file = os.path.join(tmp_dir, f"segment_{seg_idx:04d}.json")
            if os.path.exists(segment_file):
                try:
                    file_size = os.path.getsize(segment_file)
                    if file_size > 0:
                        # 如果有metadata文件，从中读取准确的帧范围（优先使用绝对帧范围）
                        if os.path.exists(metadata_file):
                            try:
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                # 优先使用绝对帧范围（如果存在）
                                start_frame = metadata.get('start_frame', metadata.get('relative_start_frame', start_frame))
                                end_frame = metadata.get('end_frame', metadata.get('relative_end_frame', end_frame))
                            except:
                                pass
                        
                        # 确保在segment_files中（可能之前已添加）
                        found = False
                        for existing in segment_files:
                            if existing[0] == seg_idx:
                                found = True
                                break
                        if not found:
                            segment_files.append((seg_idx, segment_file, start_frame, end_frame))
                        log(f"[Segmented] Skipping sub-segment {seg_idx + 1}/{num_segments} (frames {start_frame}-{end_frame}, already completed)", "info")
                        continue
                except:
                    pass
            
            # 提取segment（包含重叠）
            seg_start = max(0, start_frame - segment_overlap if seg_idx > 0 else 0)
            seg_end = min(N, end_frame + segment_overlap if seg_idx < num_segments - 1 else N)
            
            log(f"[Segmented] Processing sub-segment {seg_idx + 1}/{num_segments}: "
                f"frames {seg_start}-{seg_end} (output: {start_frame}-{end_frame})", "info")
            
            # 提取segment frames
            segment_frames = frames[seg_start:seg_end]
            
            # 确保至少有21帧
            if segment_frames.shape[0] < 21:
                add = 21 - segment_frames.shape[0]
                last_frame = segment_frames[-1:, :, :, :]
                padding_frames = last_frame.repeat(add, 1, 1, 1)
                segment_frames = torch.cat([segment_frames, padding_frames], dim=0)
            
            # 处理这个segment（递归调用run_inference，但禁用segmented避免无限递归）
            # 临时禁用segmented标志
            original_segmented = getattr(args, 'segmented', False)
            args.segmented = False
            try:
                segment_output = run_inference(pipe, segment_frames, device, dtype, args, streaming_writer=None)
            finally:
                args.segmented = original_segmented
            
            # 裁剪重叠部分
            if seg_idx > 0:
                segment_output = segment_output[segment_overlap:]
            if seg_idx < num_segments - 1:
                segment_output = segment_output[:-segment_overlap]
            
            # 裁剪或填充到实际需要的帧数
            actual_frames = end_frame - start_frame
            if segment_output.shape[0] > actual_frames:
                segment_output = segment_output[:actual_frames]
            elif segment_output.shape[0] < actual_frames:
                # 如果输出帧数不足，用最后一帧填充
                missing_frames = actual_frames - segment_output.shape[0]
                last_frame = segment_output[-1:, :, :, :]
                padding_frames = last_frame.repeat(missing_frames, 1, 1, 1)
                segment_output = torch.cat([segment_output, padding_frames], dim=0)
                log(f"[Segmented] Padded {missing_frames} frames to match expected {actual_frames} frames", "warning")
            
            # 保存到临时文件
            segment_file = os.path.join(tmp_dir, f"segment_{seg_idx:04d}.pt")
            torch.save(segment_output, segment_file)
            
            # 保存元数据文件，记录帧范围信息（用于统一checkpoint系统）
            metadata_file = os.path.join(tmp_dir, f"segment_{seg_idx:04d}.json")
            
            # 如果在 multi_gpu worker 模式下，需要转换为相对于原始视频的绝对帧范围
            if is_worker_mode:
                # start_frame 和 end_frame 是相对于 worker 接收的 frames 的
                # 需要转换为相对于原始视频的绝对帧范围
                absolute_start_frame = worker_start_idx + start_frame
                absolute_end_frame = worker_start_idx + end_frame
            else:
                absolute_start_frame = start_frame
                absolute_end_frame = end_frame
            
            metadata = {
                'seg_idx': seg_idx,
                'start_frame': absolute_start_frame,  # 相对于原始视频的绝对帧范围
                'end_frame': absolute_end_frame,
                'relative_start_frame': start_frame,  # 相对于当前 frames 的相对帧范围（向后兼容）
                'relative_end_frame': end_frame,
                'actual_frames': segment_output.shape[0],
                'segment_file': segment_file,
                'is_worker_mode': is_worker_mode,
                'worker_start_idx': worker_start_idx if is_worker_mode else None,
                'worker_end_idx': worker_end_idx if is_worker_mode else None
            }
            try:
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            except:
                pass
            
            # 使用绝对帧范围（用于统一checkpoint系统）
            segment_files.append((seg_idx, segment_file, absolute_start_frame, absolute_end_frame))
            
            log(f"[Segmented] Saved sub-segment {seg_idx + 1}/{num_segments} "
                f"({segment_output.shape[0]} frames, absolute frames {absolute_start_frame}-{absolute_end_frame}) to {segment_file}", "info")
            
            # 释放内存
            del segment_output, segment_frames
            clean_vram()
        
        # 按顺序加载并拼接所有segments（确保按seg_idx排序）
        segment_files.sort(key=lambda x: x[0])  # 按seg_idx排序
        log(f"[Segmented] Merging {len(segment_files)} sub-segments...", "info")
        
        try:
            # 流式合并：逐个加载和合并，避免同时保存多个大tensor在内存中
            final_output = None
            last_end_frame = None
            
            for seg_idx, segment_file, start_frame, end_frame in segment_files:
                if not os.path.exists(segment_file):
                    raise FileNotFoundError(f"Segment file not found: {segment_file}")
                
                log(f"[Segmented] Loading segment {seg_idx + 1}/{len(segment_files)}: {os.path.basename(segment_file)}", "info")
                segment = torch.load(segment_file, map_location='cpu')
                
                # 处理overlap（如果存在）
                if last_end_frame is not None and start_frame < last_end_frame:
                    overlap_frames = last_end_frame - start_frame
                    if overlap_frames < segment.shape[0]:
                        segment = segment[overlap_frames:]
                        log(f"[Segmented] Skipped {overlap_frames} overlap frames", "info")
                    elif overlap_frames >= segment.shape[0]:
                        log(f"[Segmented] Warning: Segment {seg_idx} is completely overlapped, skipping", "warning")
                        del segment
                        gc.collect()
                        continue
                
                # 合并到final_output（使用临时文件避免内存峰值）
                if final_output is None:
                    final_output = segment
                else:
                    # 对于大文件，使用临时文件来避免内存峰值
                    # 先检查内存使用情况
                    import psutil
                    process = psutil.Process()
                    mem_info = process.memory_info()
                    mem_gb = mem_info.rss / (1024**3)
                    
                    # 如果内存使用超过150GB，使用临时文件合并
                    if mem_gb > 150 or (final_output.numel() * final_output.element_size() / (1024**3)) > 50:
                        log(f"[Segmented] High memory usage ({mem_gb:.1f} GB), using temporary file for merging...", "info")
                        # 保存当前final_output到临时文件
                        temp_merged_file = os.path.join(tmp_dir, f"_temp_merged_{seg_idx}.pt")
                        torch.save(final_output, temp_merged_file)
                        del final_output
                        gc.collect()
                        
                        # 加载两个文件并合并
                        final_output = torch.load(temp_merged_file, map_location='cpu')
                        final_output = torch.cat([final_output, segment], dim=0)
                        del segment
                        gc.collect()
                        
                        # 保存合并结果
                        torch.save(final_output, temp_merged_file)
                        del final_output
                        gc.collect()
                        
                        # 重新加载
                        final_output = torch.load(temp_merged_file, map_location='cpu')
                        
                        # 删除旧的临时文件
                        if seg_idx > 0:
                            old_temp_file = os.path.join(tmp_dir, f"_temp_merged_{seg_idx-1}.pt")
                            if os.path.exists(old_temp_file):
                                try:
                                    os.remove(old_temp_file)
                                except:
                                    pass
                    else:
                        # 内存充足，直接合并
                        log(f"[Segmented] Merging segment {seg_idx + 1}...", "info")
                        final_output = torch.cat([final_output, segment], dim=0)
                        # 立即释放segment内存
                        del segment
                        gc.collect()
                
                last_end_frame = end_frame
            
            if final_output is None:
                raise RuntimeError("[Segmented] No valid segments to merge")
            
            # 裁剪到原始帧数
            if final_output.shape[0] > N_original:
                final_output = final_output[:N_original]
            
            log(f"[Segmented] Merged {len(segment_files)} sub-segments into final output "
                f"({final_output.shape[0]} frames)", "info")
            
            # 注意：不在这里删除临时文件，让main函数在保存视频成功后再删除
            # 这样可以确保在视频成功保存之前，临时文件不会被删除
            log(f"[Segmented] Temporary files preserved in: {tmp_dir} (will be cleaned after video is saved)", "info")
            
            return final_output
            
        except Exception as merge_error:
            # 合并失败，保留临时文件以便恢复
            log(f"[Segmented] ERROR: Failed to merge segments: {merge_error}", "error")
            log(f"[Segmented] Temporary files preserved in: {tmp_dir}", "warning")
            log(f"[Segmented] You can recover the result by manually merging files:", "warning")
            for seg_idx, segment_file, start_frame, end_frame in segment_files:
                if os.path.exists(segment_file):
                    log(f"[Segmented]   - {segment_file} (frames {start_frame}-{end_frame})", "warning")
            raise RuntimeError(f"Failed to merge segments: {merge_error}. Temporary files preserved in {tmp_dir}") from merge_error
        
    except Exception as outer_error:
        # 外层错误处理：如果处理过程中出现错误，保留已完成的segments
        log(f"[Segmented] ERROR: Processing failed: {outer_error}", "error")
        if segment_files:
            log(f"[Segmented] Some segments may be preserved. Check: {tmp_dir}", "warning")
        raise
    finally:
        # 注意：临时文件清理已在成功合并时完成
        # 如果合并失败，文件会被保留以便恢复
        pass

def run_inference_chunked(pipe, frames, device, dtype, args, streaming_writer, chunk_size_frames, N_original):
    """按chunk流式处理视频，避免创建整个canvas。
    
    将视频分成多个chunks，每个chunk独立处理并立即流式写入，然后释放内存。
    注意：这个函数直接处理chunk，不调用run_inference以避免递归。
    """
    N, H, W, C = frames.shape
    
    # 计算需要多少个chunks
    num_chunks = (N + chunk_size_frames - 1) // chunk_size_frames
    overlap_frames = getattr(args, 'segment_overlap', 2)
    log(f"[Streaming] Processing video in {num_chunks} chunks (chunk size: {chunk_size_frames} frames, overlap: {overlap_frames} frames)", "info")
    
    # 进度报告设置（仅在worker进程中）
    report_progress_to_queue = False
    if hasattr(args, '_is_worker') and args._is_worker and hasattr(args, '_result_queue'):
        report_progress_to_queue = True
    
    # 处理每个chunk
    for chunk_idx in range(num_chunks):
        start_frame = chunk_idx * chunk_size_frames
        end_frame = min((chunk_idx + 1) * chunk_size_frames, N)
        
        # 提取chunk（需要包含一些重叠帧以确保边界平滑）
        chunk_start = max(0, start_frame - overlap_frames)
        chunk_end = min(N, end_frame + overlap_frames)
        
        log(f"[Streaming] Processing chunk {chunk_idx + 1}/{num_chunks}: frames {chunk_start}-{chunk_end} "
            f"(output frames: {start_frame}-{end_frame})", "info")
        
        # 提取chunk
        chunk_frames = frames[chunk_start:chunk_end]
        chunk_N_original = chunk_frames.shape[0]
        
        # 确保chunk至少有21帧
        if chunk_frames.shape[0] < 21:
            add = 21 - chunk_frames.shape[0]
            last_frame = chunk_frames[-1:, :, :, :]
            padding_frames = last_frame.repeat(add, 1, 1, 1)
            chunk_frames = torch.cat([chunk_frames, padding_frames], dim=0)
        
        # 直接处理这个chunk（使用tiled_dit路径，但创建小的canvas）
        chunk_N, chunk_H, chunk_W, chunk_C = chunk_frames.shape
        chunk_num_aligned_frames = largest_8n1_leq(chunk_N + 4) - 4
        
        # 为这个chunk创建canvas（内存需求小得多）
        chunk_canvas = torch.zeros(
            (chunk_num_aligned_frames, chunk_H * args.scale, chunk_W * args.scale, chunk_C),
            dtype=torch.float16,
            device="cpu",
        )
        chunk_weight_canvas = torch.zeros_like(chunk_canvas)
        
        # 处理tiles
        tile_coords = calculate_tile_coords(chunk_H, chunk_W, args.tile_size, args.tile_overlap)
        batch_size = determine_optimal_batch_size(device, tile_coords, chunk_frames, args)
        tile_batches = [tile_coords[i:i + batch_size] 
                       for i in range(0, len(tile_coords), batch_size)]
        
        for batch_idx, tile_batch in enumerate(tile_batches):
            results = process_tile_batch(pipe, chunk_frames, device, dtype, args, tile_batch, batch_idx)
            
            for result in results:
                x1, y1, x2, y2 = result['coords']
                processed_tile_cpu = result['tile']
                mask_nhwc = result['mask']
                
                out_x1, out_y1 = x1 * args.scale, y1 * args.scale
                tile_H_scaled = processed_tile_cpu.shape[1]
                tile_W_scaled = processed_tile_cpu.shape[2]
                out_x2, out_y2 = out_x1 + tile_W_scaled, out_y1 + tile_H_scaled
                
                chunk_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += processed_tile_cpu * mask_nhwc
                chunk_weight_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_nhwc
            
            clean_vram()
        
        chunk_weight_canvas[chunk_weight_canvas == 0] = 1.0
        chunk_output = chunk_canvas / chunk_weight_canvas
        
        # 裁剪回原始帧数
        if chunk_output.shape[0] > chunk_N_original:
            chunk_output = chunk_output[:chunk_N_original]
        
        # 裁剪掉重叠部分（除了第一个和最后一个chunk）
        if chunk_idx > 0:
            chunk_output = chunk_output[overlap_frames:]
        if chunk_idx < num_chunks - 1:
            chunk_output = chunk_output[:-overlap_frames]
        
        # 裁剪到实际需要的帧数
        actual_frames_needed = end_frame - start_frame
        if chunk_output.shape[0] > actual_frames_needed:
            chunk_output = chunk_output[:actual_frames_needed]
        
        # 流式写入这个chunk（或累积到内存）
        streaming_writer.write_frames(chunk_output)
        log(f"[Streaming] Processed chunk {chunk_idx + 1}/{num_chunks} ({chunk_output.shape[0]} frames)", "info")
        
        # 报告进度到主进程（worker模式下，每完成一个chunk报告一次）
        if report_progress_to_queue:
            # 计算已处理的帧数（基于chunk进度）
            frames_processed = min((chunk_idx + 1) * chunk_size_frames, N_original)
            progress_percent = ((chunk_idx + 1) / num_chunks) * 100
            try:
                args._result_queue.put({
                    'worker_id': args._worker_id,
                    'type': 'tile_progress',
                    'tiles_processed': chunk_idx + 1,  # 使用chunk索引作为进度
                    'total_tiles': num_chunks,
                    'frames_processed': frames_processed,
                    'total_frames': args._total_frames,
                    'progress_percent': progress_percent,
                    'start_idx': args._start_idx,
                    'end_idx': args._end_idx
                }, block=False)
            except:
                pass
        
        # 释放内存（但保留chunk_output的引用，如果streaming_writer需要累积）
        del chunk_frames, chunk_canvas, chunk_weight_canvas
        clean_vram()
    
    # 检查 streaming_writer 是否有累积的结果（用于 worker 进程）
    if hasattr(streaming_writer, 'chunks') and streaming_writer.chunks:
        # 合并所有 chunk 的结果
        final_output = torch.cat(streaming_writer.chunks, dim=0)
        # 裁剪回原始帧数
        if final_output.shape[0] > N_original:
            final_output = final_output[:N_original]
        return final_output
    
    # 流式模式下返回None（视频已保存到文件）
    return None

def determine_optimal_batch_size(device: str, tile_coords: List[Tuple[int, int, int, int]], 
                                  frames: torch.Tensor, args) -> int:
    """Determine optimal batch size based on available GPU memory."""
    if not args.adaptive_batch_size or not device.startswith("cuda:"):
        return 1
    
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
        max_batch_limit = 16  # 32GB GPU可以支持更多并发
    else:
        safe_memory = max(2.0, available_gb - 2.0)
        max_batch_limit = 8
    
    # 计算可以同时处理的tile数量
    max_batch = max(1, int(safe_memory / tile_memory))
    optimal_batch = min(max_batch, max_batch_limit, len(tile_coords))
    
    if optimal_batch > 1:
        log(f"[Optimization] GPU: {device}, Total: {total_gb:.1f}GB, Used: {used_gb:.1f}GB, "
            f"Available: {available_gb:.2f}GB", "info")
        log(f"[Optimization] Estimated per-tile: {tile_memory:.2f}GB, "
            f"Safe memory: {safe_memory:.2f}GB, Using batch_size={optimal_batch}", "info")
    
    return optimal_batch

def run_inference(pipe, frames, device, dtype, args, streaming_writer=None):
    """Run inference; 支持整图与 DiT 瓦片两种路径（与 nodes.py 对齐）。
    
    新增功能：
    1. 动态batch_size：根据显存情况同时处理多个tile
    2. 流式处理：当内存需求过大时，将视频分成chunks，每个chunk独立处理并流式写入
    3. 分段处理：如果启用 --segmented，将视频分成多个sub-segments独立处理
    
    Args:
        streaming_writer: 可选的StreamingVideoWriter实例，用于流式写入
    """
    # 基本输入校验
    if frames is None or not hasattr(frames, 'shape') or frames.ndim != 4 or frames.shape[0] == 0:
        raise RuntimeError("[run_inference] Input frames is empty. Please check video decoding and path.")
    
    # 如果启用了分段处理模式，直接使用分段处理
    if getattr(args, 'segmented', False):
        log(f"[Segmented] Segmented processing mode enabled (--segmented)", "info")
        return run_inference_segmented(pipe, frames, device, dtype, args)
    
    # 保存原始帧数（用于最后裁剪）
    N_original = frames.shape[0]
    
    # 确保最少 21 帧（便于首尾填充）
    if frames.shape[0] < 21:
        add = 21 - frames.shape[0]
        last_frame = frames[-1:, :, :, :]
        padding_frames = last_frame.repeat(add, 1, 1, 1)
        frames = torch.cat([frames, padding_frames], dim=0)

    # 记录 tile 相关参数状态（帮助调试）
    log(f"[Tile Settings] tiled_dit={args.tiled_dit} (type: {type(args.tiled_dit).__name__}), tiled_vae={args.tiled_vae} (type: {type(args.tiled_vae).__name__}), tile_size={args.tile_size}, tile_overlap={args.tile_overlap}", "info")

    # 瓦片 DiT 路径：参考 nodes.py 的实现
    if args.tiled_dit:
        N, H, W, C = frames.shape
        num_aligned_frames = largest_8n1_leq(N + 4) - 4

        # 计算所需内存（GB）- 使用float16而不是float32
        required_memory_gb = (num_aligned_frames * H * args.scale * W * args.scale * C * 2) / (1024**3)  # float16 = 2 bytes
        # weight_sum_canvas也需要相同大小的内存
        total_required_gb = required_memory_gb * 2  # canvas + weight_sum_canvas
        
        log(f"[Memory Check] Canvas size: {num_aligned_frames} frames × {H*args.scale}×{W*args.scale}×{C}, "
            f"Required memory: {total_required_gb:.2f} GB (canvas: {required_memory_gb:.2f} GB + weight: {required_memory_gb:.2f} GB, using float16)", "info")
        
        # 检查系统可用内存
        try:
            sys_used, sys_total = get_system_memory_info()
            sys_free = sys_total - sys_used
            log(f"[Memory Check] System RAM: {sys_free:.2f} GB free / {sys_total:.2f} GB total", "info")
            
            # 检查是否有足够内存创建canvas
            # 需要保留一些安全边界（根据总内存动态调整）
            # 对于大内存系统（>200GB），保留10GB；对于小内存系统，保留20GB
            if sys_total > 200:
                safe_memory_gb = sys_free - 10.0
            else:
                safe_memory_gb = sys_free - 20.0
            
            # 如果需求超过可用内存
            if total_required_gb > sys_free:
                # 如果只是稍微超过（5%以内），允许尝试（因为有SWAP空间和重试机制）
                if total_required_gb <= sys_free * 1.05:
                    log(f"[WARNING] Canvas memory requirement ({total_required_gb:.2f} GB) slightly exceeds available RAM ({sys_free:.2f} GB). "
                        f"Will attempt allocation (SWAP space may be used, or will retry after other workers complete).", "warning")
                    # 不抛出错误，继续到重试机制
                else:
                    # 超过太多，给出警告但允许尝试（让重试机制处理）
                    log(f"[WARNING] Canvas memory requirement ({total_required_gb:.2f} GB) significantly exceeds available RAM ({sys_free:.2f} GB). "
                        f"Will attempt allocation with retry mechanism. If all retries fail, consider:", "warning")
                    log("  1. Use smaller scale (e.g., scale=2 instead of 4)", "warning")
                    log("  2. Process shorter video segments", "warning")
                    log("  3. Increase system SWAP space (recommended: 50-100GB)", "warning")
                    log("  4. Use single GPU mode instead of multi-GPU", "warning")
                    # 不抛出错误，继续到重试机制
            
            # 如果需求超过安全限制但仍在可用内存内，给出警告但允许继续
            if total_required_gb > safe_memory_gb:
                log(f"[WARNING] Canvas memory requirement ({total_required_gb:.2f} GB) exceeds safe limit ({safe_memory_gb:.2f} GB) "
                    f"but is within available RAM ({sys_free:.2f} GB). Proceeding with caution...", "warning")
                log("  Note: If OOM occurs, consider increasing SWAP space or using smaller scale.", "warning")
            
            if total_required_gb > sys_free * 0.8:  # 如果需求超过可用内存的80%
                log(f"[WARNING] Canvas memory requirement ({total_required_gb:.2f} GB) exceeds 80% of available RAM ({sys_free:.2f} GB). "
                    f"OOM risk! Consider:", "warning")
                log(f"  - Using smaller scale (current: {args.scale})", "warning")
                log(f"  - Processing shorter video segments", "warning")
                log(f"  - Increasing system SWAP space", "warning")
        except RuntimeError:
            raise  # 重新抛出内存不足错误
        except:
            pass
        
        if total_required_gb > 200:  # 如果超过200GB
            log(f"[WARNING] Canvas memory requirement ({total_required_gb:.2f} GB) is very large. "
                f"Consider using smaller scale or shorter video.", "warning")
        
        # 检查是否需要使用chunk流式处理
        # 如果用户启用了 --streaming，或者 streaming_writer 不为 None，则检查是否需要 chunk 流式处理
        use_chunk_streaming = False
        chunk_size_frames = None
        if args.streaming or streaming_writer is not None:
            try:
                sys_used, sys_total = get_system_memory_info()
                sys_free = sys_total - sys_used
            except:
                sys_total = 200  # 默认值
                sys_free = 100
            
            # 计算内存阈值：使用系统可用内存的30%作为阈值
            # 对于大内存系统（>200GB），使用30%；对于小内存系统，使用40%
            if sys_total > 200:
                memory_threshold_ratio = 0.30  # 大内存系统使用30%
            else:
                memory_threshold_ratio = 0.40  # 小内存系统使用40%
            
            memory_threshold_gb = sys_free * memory_threshold_ratio
            
            # 如果用户启用了 --streaming，或者 canvas 内存需求超过阈值，使用chunk流式处理
            if args.streaming or total_required_gb > memory_threshold_gb:
                use_chunk_streaming = True
                # 计算合适的chunk大小：每个chunk的canvas不超过可用内存的10%（更保守，考虑VAE解码的额外内存）
                # VAE解码需要额外的内存（latents、中间激活等），所以需要更小的chunk
                # 如果 tiled_vae=False，需要更小的chunk（因为VAE会解码整个chunk）
                if args.tiled_vae:
                    chunk_canvas_memory_ratio = 0.12  # VAE tile时可以用稍大的chunk
                else:
                    chunk_canvas_memory_ratio = 0.08  # VAE不tile时，需要更小的chunk（VAE解码整个chunk需要更多内存）
                
                chunk_canvas_memory_gb = sys_free * chunk_canvas_memory_ratio
                # 但至少保留1.5GB，最多使用15GB（更保守，避免VAE解码时OOM）
                chunk_canvas_memory_gb = max(1.5, min(chunk_canvas_memory_gb, 15.0))
                
                chunk_size_frames = int((chunk_canvas_memory_gb * (1024**3)) / (H * args.scale * W * args.scale * C * 2 * 2))  # 2个canvas
                # 确保chunk大小至少为21帧（模型要求），且是8的倍数（对齐要求）
                chunk_size_frames = max(21, chunk_size_frames)
                chunk_size_frames = largest_8n1_leq(chunk_size_frames)
                # 限制最大chunk大小为300帧（更保守，避免VAE解码时OOM）
                chunk_size_frames = min(chunk_size_frames, 300)
                
                reason = "user enabled --streaming" if args.streaming else f"canvas memory ({total_required_gb:.2f} GB) exceeds threshold ({memory_threshold_gb:.2f} GB)"
                vae_note = " (VAE not tiled, using smaller chunks)" if not args.tiled_vae else " (VAE tiled)"
                log(f"[Streaming] {reason}. Will use chunk streaming with chunk size: {chunk_size_frames} frames "
                    f"(chunk canvas: {chunk_canvas_memory_gb:.2f} GB, {chunk_canvas_memory_ratio*100:.0f}% of available RAM{vae_note})", "info")
        
        # 如果使用chunk流式处理
        if use_chunk_streaming:
            # 如果 streaming_writer 为 None（例如在 worker 进程中），创建一个临时的累积器
            if streaming_writer is None:
                # 在 worker 进程中，我们需要累积所有 chunk 的结果
                # 创建一个临时列表来存储 chunk 结果
                chunk_results = []
                
                # 创建一个临时的 StreamingVideoWriter，但重写 write_frames 来累积结果
                class ChunkAccumulator:
                    def __init__(self):
                        self.chunks = []
                    
                    def write_frames(self, frames):
                        self.chunks.append(frames.clone())
                
                accumulator = ChunkAccumulator()
                run_inference_chunked(pipe, frames, device, dtype, args, accumulator, chunk_size_frames, N_original)
                
                # 合并所有 chunk 的结果
                if accumulator.chunks:
                    final_output = torch.cat(accumulator.chunks, dim=0)
                    # 裁剪回原始帧数
                    if final_output.shape[0] > N_original:
                        final_output = final_output[:N_original]
                    return final_output
                else:
                    raise RuntimeError("No chunks were processed in streaming mode")
            else:
                # 如果有 streaming_writer，直接使用 chunk 流式处理
                return run_inference_chunked(pipe, frames, device, dtype, args, streaming_writer, chunk_size_frames, N_original)
        
        # 如果提供了streaming_writer但不需要chunk处理，仍然需要创建canvas（但会在最后流式写入）
        # 这种情况下，我们仍然创建canvas，但在最后通过streaming_writer写入
        
        # 使用float16而不是float32，减少50%内存占用
        # 如果内存不足，等待并重试（适用于多GPU场景，前一个worker可能正在使用内存）
        max_retries = 10
        retry_delay = 5  # 每次重试等待5秒
        canvas_allocated = False
        
        for retry in range(max_retries):
            try:
                # 在每次重试前检查内存
                if retry > 0:
                    try:
                        sys_used, sys_total = get_system_memory_info()
                        sys_free = sys_total - sys_used
                        log(f"[Memory Check] Retry {retry}/{max_retries}: Available memory: {sys_free:.2f} GB", "info")
                        if total_required_gb > sys_free:
                            log(f"[Memory Check] Still insufficient memory. Waiting {retry_delay}s before retry...", "warning")
                            time.sleep(retry_delay)
                            continue
                    except:
                        pass
                    time.sleep(retry_delay)
                
                final_output_canvas = torch.zeros(
                    (num_aligned_frames, H * args.scale, W * args.scale, C),
                    dtype=torch.float16,  # 改为float16以减少内存占用
                    device="cpu",
                )
                weight_sum_canvas = torch.zeros_like(final_output_canvas)
                log(f"[Memory Check] Successfully allocated canvas ({total_required_gb:.2f} GB)", "info")
                canvas_allocated = True
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cannot allocate" in str(e).lower():
                    if retry < max_retries - 1:
                        log(f"[Memory Check] Failed to allocate canvas (attempt {retry + 1}/{max_retries}): {e}. "
                            f"Waiting {retry_delay}s and retrying...", "warning")
                        time.sleep(retry_delay)
                        continue
                    else:
                        error_msg = (f"[ERROR] Failed to allocate canvas memory after {max_retries} attempts: {e}. "
                                    f"Required: {total_required_gb:.2f} GB. "
                                    f"Please use smaller scale or increase SWAP space.")
                        log(error_msg, "error")
                        raise RuntimeError(error_msg) from e
                raise
        
        if not canvas_allocated:
            error_msg = (f"[ERROR] Failed to allocate canvas after {max_retries} attempts. "
                        f"Required: {total_required_gb:.2f} GB. "
                        f"Please use smaller scale or increase SWAP space.")
            raise RuntimeError(error_msg)

        tile_coords = calculate_tile_coords(H, W, args.tile_size, args.tile_overlap)
        
        # 确定最优batch_size
        batch_size = determine_optimal_batch_size(device, tile_coords, frames, args)
        
        # 将tile_coords分成batch
        tile_batches = [tile_coords[i:i + batch_size] 
                       for i in range(0, len(tile_coords), batch_size)]
        
        total_tiles = len(tile_coords)
        processed = 0
        
        # 进度报告设置（仅在worker进程中）
        report_progress_to_queue = False
        last_reported_tiles = 0
        progress_report_interval = max(1, total_tiles // 10)  # 每10%报告一次
        
        if hasattr(args, '_is_worker') and args._is_worker and hasattr(args, '_result_queue'):
            report_progress_to_queue = True
        
        for batch_idx, tile_batch in enumerate(tqdm(tile_batches, desc="Processing Tile Batches")):
            # 处理当前batch的tiles
            results = process_tile_batch(pipe, frames, device, dtype, args, tile_batch, batch_idx)
            
            # 合并结果到canvas
            for result in results:
                x1, y1, x2, y2 = result['coords']
                processed_tile_cpu = result['tile']
                mask_nhwc = result['mask']

                out_x1, out_y1 = x1 * args.scale, y1 * args.scale
                tile_H_scaled = processed_tile_cpu.shape[1]
                tile_W_scaled = processed_tile_cpu.shape[2]
                out_x2, out_y2 = out_x1 + tile_W_scaled, out_y1 + tile_H_scaled

                final_output_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += processed_tile_cpu * mask_nhwc
                weight_sum_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_nhwc

                processed += 1
                log(f"[FlashVSR] Processed tile {processed}/{total_tiles}: ({x1},{y1})-({x2},{y2})", "info")
            
            # 报告进度到主进程（worker模式下，每10%报告一次）
            if report_progress_to_queue and (processed - last_reported_tiles >= progress_report_interval or processed == total_tiles):
                # 计算已处理的帧数（基于tile进度估算）
                frames_processed = int((processed / total_tiles) * args._total_frames)
                progress_percent = (processed / total_tiles) * 100
                try:
                    args._result_queue.put({
                        'worker_id': args._worker_id,
                        'type': 'tile_progress',
                        'tiles_processed': processed,
                        'total_tiles': total_tiles,
                        'frames_processed': frames_processed,
                        'total_frames': args._total_frames,
                        'progress_percent': progress_percent,
                        'start_idx': args._start_idx,
                        'end_idx': args._end_idx
                    }, block=False)
                except:
                    pass
                last_reported_tiles = processed
                
            # 每次batch后清理显存
            clean_vram()
            
            # 动态调整batch_size（如果启用）
            if args.adaptive_batch_size and batch_idx > 0 and batch_idx % 5 == 0:
                new_batch_size = determine_optimal_batch_size(device, tile_coords[processed:], frames, args)
                if new_batch_size != batch_size:
                    batch_size = new_batch_size
                    log(f"[Optimization] Adjusted batch_size to {batch_size}", "info")

        weight_sum_canvas[weight_sum_canvas == 0] = 1.0
        final_output = final_output_canvas / weight_sum_canvas
        
        # 裁剪回原始帧数（对齐可能增加了帧数）
        if final_output.shape[0] > N_original:
            final_output = final_output[:N_original]
        
        # 如果提供了streaming_writer，流式写入并返回None
        if streaming_writer is not None:
            streaming_writer.write_frames(final_output)
            del final_output, final_output_canvas, weight_sum_canvas
            clean_vram()
            return None
        
        return final_output

    # 整图路径（tiled_dit=False 时使用）
    log(f"[Tile Settings] ✓ 使用整图处理路径（tiled_dit=False，不会产生 DiT tile 边界）", "info")
    LQ, th, tw, F = prepare_input_tensor(frames, device, scale=args.scale, dtype=dtype)
    if "long" not in args.mode:
        LQ = LQ.to(device)

    topk_ratio = args.sparse_ratio * 768 * 1280 / (th * tw)

    with torch.no_grad():
        output = pipe(
            prompt="",
            negative_prompt="",
            cfg_scale=1.0,
            num_inference_steps=1,
            seed=args.seed,
            tiled=args.tiled_vae,
            progress_bar_cmd=tqdm,
            LQ_video=LQ,
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

    if isinstance(output, (tuple, list)):
        output = output[0]

    final_output = tensor2video(output).to("cpu")
    
    # 裁剪回原始帧数（对齐可能增加了帧数）
    if final_output.shape[0] > N_original:
        final_output = final_output[:N_original]
    
    del output, LQ
    clean_vram()
    return final_output

# ==============================================================
#                            Main
# ==============================================================

def main(args):
    total_start = time.time()
    
    # 打印所有 tile 相关参数（帮助调试）
    log("=" * 80, "info")
    log("[参数检查] 实际接收到的 tile 相关参数:", "info")
    log(f"  --tiled_dit = {args.tiled_dit} (type: {type(args.tiled_dit).__name__})", "info")
    log(f"  --tiled_vae = {args.tiled_vae} (type: {type(args.tiled_vae).__name__})", "info")
    log(f"  --tile_size = {args.tile_size}", "info")
    log(f"  --tile_overlap = {args.tile_overlap}", "info")
    log(f"  --unload_dit = {args.unload_dit}", "info")
    log(f"  --segment_overlap = {getattr(args, 'segment_overlap', 2)}", "info")
    log(f"  --segmented = {getattr(args, 'segmented', False)}", "info")
    log("=" * 80, "info")
    
    # 读取视频获取总帧数（用于统一checkpoint系统）
    frames, input_fps = read_video_to_tensor(args.input)
    total_frames = frames.shape[0]
    
    # 统一的断点续跑检查（如果启用resume）
    if getattr(args, 'resume', False) and not getattr(args, 'clean_checkpoint', False):
        log("[Resume] Scanning for completed frames...", "info")
        # 只有在明确指定 --resume 时才扫描已完成的帧
        if getattr(args, 'resume', False):
            completed_ranges = scan_all_completed_frames(args.input, args, total_frames)
        else:
            # 没有指定 --resume，默认覆盖模式：不扫描已完成的帧
            completed_ranges = []
        
        if completed_ranges:
            log(f"[Resume] Found {len(completed_ranges)} completed frame ranges:", "info")
            total_completed = 0
            for start, end, path, mode in completed_ranges:
                frames_count = end - start
                total_completed += frames_count
                log(f"  - Frames {start}-{end} ({frames_count} frames) from {mode}: {os.path.basename(path)}", "info")
            
            # 检查是否全部完成
            if total_completed >= total_frames:
                log("[Resume] All frames already processed! Merging results...", "info")
                output_path = args.output if args.output else os.path.join("/app/output", os.path.basename(args.input).replace(".mp4", "_out.mp4"))
                merge_all_completed_frames(completed_ranges, output_path, input_fps, total_frames)
                del frames
                clean_vram()
                total_time = time.time() - total_start
                log(f"Output saved to {output_path}", "finish")
                log(f"[Total] Total elapsed time: {format_duration(total_time)}", "finish")
                return
            
            # 找出未完成的帧范围
            missing_ranges = find_missing_frame_ranges(total_frames, completed_ranges)
            if missing_ranges:
                log(f"[Resume] Found {len(missing_ranges)} missing frame ranges:", "info")
                for start, end in missing_ranges:
                    log(f"  - Frames {start}-{end} ({end-start} frames)", "info")
                
                # 只处理缺失的帧范围
                # 注意：这里需要根据用户选择的模式（multi_gpu或segmented）来处理
                # 目前先提示用户，后续可以扩展为自动处理缺失范围
                log("[Resume] Processing missing frames...", "info")
                # TODO: 实现只处理缺失帧范围的逻辑
                # 目前继续正常流程，但会跳过已完成的segments
    
    # 处理多GPU模式
    if args.multi_gpu:
        # 获取所有可用的CUDA设备
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for multi-GPU processing!")
        
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            log(f"Warning: Only {num_gpus} GPU(s) available, falling back to single GPU mode", "warning")
            args.multi_gpu = False
        else:
            devices = [f"cuda:{i}" for i in range(num_gpus)]
            log(f"[Multi-GPU] Using {num_gpus} GPUs: {devices}", "info")
            
            # 处理 attention_mode
            # 注意：最新版本的block_sparse_attention（2025/12更新）已支持Blackwell (sm_120)
            if args.attention_mode == "sparse_sage_attention":
                wan_video_dit.USE_BLOCK_ATTN = False
            else:
                wan_video_dit.USE_BLOCK_ATTN = True
                # 如果BLOCK_ATTN_AVAILABLE为False，说明模块未正确安装或编译
                if not wan_video_dit.BLOCK_ATTN_AVAILABLE:
                    log(f"Warning: block_sparse_attention not available. Auto-switching to sparse_sage_attention", "warning")
                    wan_video_dit.USE_BLOCK_ATTN = False
            
            # 使用多GPU处理（frames已在上面读取）
            output = run_inference_multi_gpu(frames, devices, args, input_fps)
            
            # 保存视频（流式模式下output为None，视频已保存）
            if output is not None:
                output_dir = args.output if args.output else os.path.join("/app/output", os.path.basename(args.input).replace(".mp4", "_out.mp4"))
                save_video(output, output_dir, fps=input_fps)
                del output
            else:
                # 流式模式下，视频已保存到args.output
                output_dir = args.output if args.output else os.path.join("/app/output", os.path.basename(args.input).replace(".mp4", "_out.mp4"))
            
            # 验证视频文件是否成功保存
            if not os.path.exists(output_dir):
                raise RuntimeError(f"Video file was not created: {output_dir}")
            file_size = os.path.getsize(output_dir)
            if file_size == 0:
                raise RuntimeError(f"Video file is empty: {output_dir}")
            log(f"[Verify] Video file verified: {output_dir} ({file_size / (1024**2):.2f} MB)", "info")
            
            # 只有在视频成功保存并验证后才清理临时文件和checkpoint
            log("[Cleanup] Cleaning up temporary files and checkpoints...", "info")
            
            # 清理multi_gpu临时文件
            import glob
            video_dir_name = get_video_based_dir_name(args.input, args.scale)
            multi_gpu_tmp = os.path.join("/tmp", "flashvsr_multigpu", video_dir_name)
            if os.path.exists(multi_gpu_tmp):
                worker_files = glob.glob(os.path.join(multi_gpu_tmp, "worker_*.pt"))
                for worker_file in worker_files:
                    try:
                        os.remove(worker_file)
                        log(f"[Cleanup] Removed: {worker_file}", "info")
                    except:
                        pass
                # 如果目录为空，尝试删除目录
                try:
                    if not os.listdir(multi_gpu_tmp):
                        os.rmdir(multi_gpu_tmp)
                except:
                    pass
            
            # 清理segmented临时文件
            video_dir_name = get_video_based_dir_name(args.input, args.scale)
            segmented_base = "/tmp/flashvsr_segments"
            if os.path.exists(segmented_base):
                # 优先使用基于视频名称的目录
                segment_dir = os.path.join(segmented_base, video_dir_name)
                if os.path.exists(segment_dir) and os.path.isdir(segment_dir):
                    segment_files = glob.glob(os.path.join(segment_dir, "segment_*.pt"))
                    for seg_file in segment_files:
                        try:
                            os.remove(seg_file)
                            log(f"[Cleanup] Removed: {seg_file}", "info")
                        except:
                            pass
                    # 如果目录为空，尝试删除目录
                    try:
                        if not os.listdir(segment_dir):
                            os.rmdir(segment_dir)
                    except:
                        pass
                else:
                    # 向后兼容：查找旧的基于hash的目录
                    for segment_dir_name in os.listdir(segmented_base):
                        segment_dir = os.path.join(segmented_base, segment_dir_name)
                        if not os.path.isdir(segment_dir):
                            continue
                        # 检查是否有metadata文件指向当前输入
                        metadata_files = glob.glob(os.path.join(segment_dir, "segment_*.json"))
                        if metadata_files:
                            try:
                                with open(metadata_files[0], 'r') as f:
                                    metadata = json.load(f)
                                # 如果metadata中有worker信息，说明是multi_gpu+segmented模式，可以清理
                                if metadata.get('is_worker_mode', False):
                                    segment_files = glob.glob(os.path.join(segment_dir, "segment_*.pt"))
                                    for seg_file in segment_files:
                                        try:
                                            os.remove(seg_file)
                                            log(f"[Cleanup] Removed: {seg_file}", "info")
                                        except:
                                            pass
                            except:
                                pass
            
            # 清理checkpoint（可选，如果用户想要保留checkpoint可以跳过）
            if not getattr(args, 'keep_checkpoint', False):
                video_dir_name = get_video_based_dir_name(args.input, args.scale)
                multi_gpu_checkpoint_dir = os.path.join('/tmp', 'flashvsr_checkpoints')
                if os.path.exists(multi_gpu_checkpoint_dir):
                    # 优先清理基于视频名称的checkpoint目录
                    checkpoint_path = os.path.join(multi_gpu_checkpoint_dir, video_dir_name)
                    if os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path):
                        try:
                            import shutil
                            shutil.rmtree(checkpoint_path)
                            log(f"[Cleanup] Removed checkpoint: {checkpoint_path}", "info")
                        except:
                            pass
                    # 向后兼容：清理旧的基于hash的checkpoint目录
                    for checkpoint_dir_name in os.listdir(multi_gpu_checkpoint_dir):
                        checkpoint_path = os.path.join(multi_gpu_checkpoint_dir, checkpoint_dir_name)
                        if not os.path.isdir(checkpoint_path):
                            continue
                        # 跳过已经清理过的目录
                        if checkpoint_dir_name == video_dir_name:
                            continue
                        try:
                            checkpoint_data = load_checkpoint(checkpoint_path)
                            # 检查是否匹配当前输入
                            # 这里简化处理，如果checkpoint中有segment信息，就清理
                            if checkpoint_data and any(k.startswith('segment_') for k in checkpoint_data.keys()):
                                import shutil
                                shutil.rmtree(checkpoint_path)
                                log(f"[Cleanup] Removed checkpoint: {checkpoint_path}", "info")
                        except:
                            pass
            
            del frames
            clean_vram()
            
            total_time = time.time() - total_start
            log(f"Output saved to {output_dir}", "finish")
            log(f"[Total] Total elapsed time: {format_duration(total_time)}", "finish")
            return
    
    # 单GPU模式
    _device = args.device
    if _device == "auto":
        _device = "cuda:0" if torch.cuda.is_available() else "mps" if hasattr(torch, "mps") and torch.mps.is_available() else "cpu"
    
    if _device == "auto":
        raise RuntimeError("No devices found to run FlashVSR!")
    
    if _device.startswith("cuda"):
        torch.cuda.set_device(_device)
    
    if args.tiled_dit and (args.tile_overlap > args.tile_size / 2):
        raise ValueError('The "tile_overlap" must be less than half of "tile_size"!')
    
    # 处理 attention_mode
    # 注意：最新版本的block_sparse_attention（2025/12更新）已支持Blackwell (sm_120)
    if args.attention_mode == "sparse_sage_attention":
        wan_video_dit.USE_BLOCK_ATTN = False
    else:
        wan_video_dit.USE_BLOCK_ATTN = True
        # 如果BLOCK_ATTN_AVAILABLE为False，说明模块未正确安装或编译
        if not wan_video_dit.BLOCK_ATTN_AVAILABLE:
            log(f"Warning: block_sparse_attention not available. Auto-switching to sparse_sage_attention", "warning")
            wan_video_dit.USE_BLOCK_ATTN = False
    
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.precision, torch.bfloat16)
    
    # 根据 model_ver 参数构建模型路径
    model_dir = f"/app/models/v{args.model_ver}"
    pipe = init_pipeline(args.mode, _device, dtype, model_dir)
    frames, input_fps = read_video_to_tensor(args.input)
    
    # 检查是否需要流式处理（用户控制，类似 --multi_gpu）
    N, H, W, C = frames.shape
    # 估算canvas内存需求
    num_aligned_frames = largest_8n1_leq(N + 4) - 4
    estimated_canvas_memory_gb = (num_aligned_frames * H * args.scale * W * args.scale * C * 2 * 2) / (1024**3)  # 2个canvas，float16
    
    # 用户控制流式处理
    use_streaming = False
    if args.streaming:
        # 用户明确启用流式处理
        use_streaming = True
        log(f"[Streaming] Streaming mode enabled by user (--streaming)", "info")
    else:
        # 如果用户未启用，但内存需求过大，给出警告建议
        try:
            sys_used, sys_total = get_system_memory_info()
            sys_free = sys_total - sys_used
            
            # 计算内存阈值：使用系统可用内存的30%作为阈值
            # 对于大内存系统（>200GB），使用30%；对于小内存系统，使用40%
            if sys_total > 200:
                memory_threshold_ratio = 0.30  # 大内存系统使用30%
            else:
                memory_threshold_ratio = 0.40  # 小内存系统使用40%
            
            memory_threshold_gb = sys_free * memory_threshold_ratio
            
            if estimated_canvas_memory_gb > memory_threshold_gb:
                log(f"[WARNING] Estimated canvas memory ({estimated_canvas_memory_gb:.2f} GB) exceeds threshold "
                    f"({memory_threshold_gb:.2f} GB, {memory_threshold_ratio*100:.0f}% of available RAM {sys_free:.2f} GB). "
                    f"Consider using --streaming to avoid OOM.", "warning")
        except:
            # 如果无法获取系统内存信息，使用保守的默认值（30GB）
            if estimated_canvas_memory_gb > 30.0:
                log(f"[WARNING] Estimated canvas memory ({estimated_canvas_memory_gb:.2f} GB) is large. "
                    f"Consider using --streaming to avoid OOM.", "warning")
    
    streaming_writer = None
    
    if use_streaming:
        output_dir = args.output if args.output else os.path.join("/app/output", os.path.basename(args.input).replace(".mp4", "_out.mp4"))
        # 初始化streaming writer（需要先知道输出尺寸）
        # 由于我们不知道确切的输出尺寸，先处理第一帧来获取尺寸
        # 或者，我们可以从输入尺寸推断输出尺寸
        output_H = H * args.scale
        output_W = W * args.scale
        streaming_writer = StreamingVideoWriter(output_dir, fps=input_fps, height=output_H, width=output_W)
        log(f"[Streaming] Initialized streaming writer: {output_W}x{output_H} @ {input_fps}fps", "info")
    
    output = run_inference(pipe, frames, _device, dtype, args, streaming_writer=streaming_writer)
    
    # 如果使用流式处理，output为None，视频已保存
    if output is None:
        if streaming_writer is not None:
            streaming_writer.close()
        output_dir = args.output if args.output else os.path.join("/app/output", os.path.basename(args.input).replace(".mp4", "_out.mp4"))
    else:
        # 保存视频（使用原始FPS）
        output_dir = args.output if args.output else os.path.join("/app/output", os.path.basename(args.input).replace(".mp4", "_out.mp4"))
        save_video(output, output_dir, fps=input_fps)
        del output
    
    # 验证视频文件是否成功保存
    if not os.path.exists(output_dir):
        raise RuntimeError(f"Video file was not created: {output_dir}")
    file_size = os.path.getsize(output_dir)
    if file_size == 0:
        raise RuntimeError(f"Video file is empty: {output_dir}")
    log(f"[Verify] Video file verified: {output_dir} ({file_size / (1024**2):.2f} MB)", "info")
    
    # 只有在视频成功保存并验证后才清理临时文件
    log("[Cleanup] Cleaning up temporary files...", "info")
    
    # 清理segmented临时文件（如果使用了segmented模式）
    if getattr(args, 'segmented', False):
        video_dir_name = get_video_based_dir_name(args.input, args.scale)
        segmented_base = "/tmp/flashvsr_segments"
        if os.path.exists(segmented_base):
            import glob
            # 优先使用基于视频名称的目录
            segment_dir = os.path.join(segmented_base, video_dir_name)
            if os.path.exists(segment_dir) and os.path.isdir(segment_dir):
                segment_files = glob.glob(os.path.join(segment_dir, "segment_*.pt"))
                for seg_file in segment_files:
                    try:
                        os.remove(seg_file)
                        log(f"[Cleanup] Removed: {seg_file}", "info")
                    except:
                        pass
                # 如果目录为空，尝试删除目录
                try:
                    if not os.listdir(segment_dir):
                        os.rmdir(segment_dir)
                except:
                    pass
            else:
                # 向后兼容：查找旧的基于hash的目录
                for segment_dir_name in os.listdir(segmented_base):
                    segment_dir = os.path.join(segmented_base, segment_dir_name)
                    if not os.path.isdir(segment_dir):
                        continue
                    # 跳过已经检查过的目录
                    if segment_dir_name == video_dir_name:
                        continue
                    # 检查metadata文件是否匹配当前输入
                    metadata_files = glob.glob(os.path.join(segment_dir, "segment_*.json"))
                    if metadata_files:
                        try:
                            with open(metadata_files[0], 'r') as f:
                                metadata = json.load(f)
                            # 如果metadata中没有worker信息，说明是单GPU的segmented模式
                            if not metadata.get('is_worker_mode', False):
                                segment_files = glob.glob(os.path.join(segment_dir, "segment_*.pt"))
                                for seg_file in segment_files:
                                    try:
                                        os.remove(seg_file)
                                        log(f"[Cleanup] Removed: {seg_file}", "info")
                                    except:
                                        pass
                        except:
                            pass
    
    del pipe, frames
    clean_vram()
    
    total_time = time.time() - total_start
    log(f"Output saved to {output_dir}", "finish")
    log(f"[Total] Total elapsed time: {format_duration(total_time)}", "finish")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashVSR Standalone Inference with Optimizations")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--model_ver", type=str, default="1.1", choices=["1.0", "1.1"], help="Model version (1.0 or 1.1, default: 1.1)")
    parser.add_argument("--mode", type=str, default="tiny", choices=["tiny", "full", "tiny-long"], help="Model mode")
    parser.add_argument("--device", type=str, default="cuda:0", choices=get_device_list(), help="Device (ignored if --multi_gpu is used)")
    
    # 从 nodes.py 添加所有参数
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4], help="Upscale factor")
    # 布尔参数转换函数（正确处理 "True"/"False" 字符串）
    def str_to_bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')
    
    parser.add_argument("--color_fix", type=str_to_bool, default=True, help="Use color fix (True/False)")
    parser.add_argument("--tiled_vae", type=str_to_bool, default=True, help="Use tiled VAE (True/False)")
    parser.add_argument("--tiled_dit", type=str_to_bool, default=False, help="Use tiled DiT (True/False)")
    parser.add_argument("--tile_size", type=int, default=256, help="Tile size")
    parser.add_argument("--tile_overlap", type=int, default=24, help="Tile overlap")
    parser.add_argument("--unload_dit", type=str_to_bool, default=False, help="Unload DiT before decoding (True/False)")
    parser.add_argument("--sparse_ratio", type=float, default=2.0, help="Sparse ratio")
    parser.add_argument("--kv_ratio", type=float, default=3.0, help="KV ratio")
    parser.add_argument("--local_range", type=int, default=11, help="Local range")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Precision")
    parser.add_argument("--attention_mode", type=str, default="sparse_sage_attention", choices=["sparse_sage_attention", "block_sparse_attention"], help="Attention mode")
    
    # 新增优化参数
    parser.add_argument("--multi_gpu", action="store_true", help="Enable multi-GPU parallel processing (splits video by frames)")
    parser.add_argument("--adaptive_batch_size", action="store_true", help="Enable adaptive batch size for tiles (dynamically adjust based on GPU memory)")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode for long videos (processes video in chunks to reduce memory usage)")
    parser.add_argument("--segmented", action="store_true", help="Enable segmented processing mode (processes video in sub-segments to reduce memory usage, similar to --multi_gpu but for single worker)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint: automatically detects and merges completed frames from previous runs (works with --multi_gpu and --segmented)")
    parser.add_argument("--clean-checkpoint", action="store_true", help="Clean checkpoint directory before starting (disable resume)")
    parser.add_argument("--segment_overlap", type=int, default=2, help="Number of overlap frames between segments/chunks (default: 2, range: 1-10, recommended: 1-5)")
    parser.add_argument("--max-segment-frames", type=int, default=None, help="Maximum frames per segment in segmented mode (default: auto-calculate based on memory, higher value = fewer segments = faster but more memory)")
    
    args = parser.parse_args()
    
    main(args)
