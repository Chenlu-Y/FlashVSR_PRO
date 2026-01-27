#!/usr/bin/env python3
"""
SDR 视频编码工具

将 DPX 图像序列编码为 SDR 视频（H.264 with sRGB/bt709）

关键：DPX 文件是 sRGB 编码的，需要正确指定输入颜色空间
"""

import os
import subprocess
import glob
from typing import Optional


def encode_dpx_to_sdr_video(
    dpx_dir: str,
    output_path: str,
    fps: float = 30.0,
    crf: int = 18,
    preset: str = 'slow'
) -> bool:
    """将 DPX 序列编码为 SDR 视频（H.264）。
    
    Args:
        dpx_dir: DPX 文件目录（包含 frame_XXXXXX.dpx 文件）
        output_path: 输出视频路径（.mp4）
        fps: 帧率
        crf: 质量参数（18-28，越小质量越高）
        preset: 编码预设（ultrafast, fast, medium, slow, veryslow）
    
    Returns:
        bool: 是否成功
    """
    dpx_files = sorted(glob.glob(os.path.join(dpx_dir, "frame_*.dpx")))
    if not dpx_files:
        raise ValueError(f"未找到 DPX 文件在目录: {dpx_dir}")
    
    print(f"[SDR Encode] 找到 {len(dpx_files)} 个 DPX 文件")
    
    # 获取第一帧的尺寸
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "default=noprint_wrappers=1:nokey=1",
        dpx_files[0]
    ]
    
    try:
        probe_result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        lines = probe_result.stdout.decode().strip().split('\n')
        w, h = int(lines[0]), int(lines[1])
        print(f"[SDR Encode] 视频尺寸: {w}x{h}")
    except Exception as e:
        raise RuntimeError(f"无法获取 DPX 图像尺寸: {e}")
    
    # 关键：DPX 文件是 sRGB 编码的，需要显式指定输入颜色空间
    # 输出也应该是 sRGB/bt709，这样在标准显示器上才能正确显示
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", os.path.join(dpx_dir, "frame_*.dpx"),
        # 显式指定输入为 sRGB (bt709)，输出也为 sRGB (bt709)
        # 这样 FFmpeg 不会进行不必要的颜色空间转换
        "-vf", f"scale={w}:{h}:flags=lanczos",
        "-color_primaries", "bt709",      # sRGB 使用 bt709 原色
        "-color_trc", "bt709",            # sRGB 使用 bt709 传输特性（伽马 2.2）
        "-colorspace", "bt709",           # sRGB 使用 bt709 颜色空间
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",            # SDR 使用 8-bit YUV
        "-movflags", "+faststart",
        output_path
    ]
    
    print(f"[SDR Encode] 开始编码 SDR 视频...")
    print(f"[SDR Encode] 输出: {output_path}")
    print(f"[SDR Encode] 颜色空间: sRGB (bt709) -> sRGB (bt709)")
    print(f"[SDR Encode] 命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            file_size_mb = os.path.getsize(output_path) / (1024**2)
            print(f"[SDR Encode] ✓ 编码成功: {output_path} ({file_size_mb:.2f} MB)")
            return True
        else:
            stderr = result.stderr.decode('utf-8', errors='ignore')
            print(f"[SDR Encode] ✗ 编码失败 (返回码: {result.returncode})")
            print(f"[SDR Encode] 错误信息: {stderr[-1000:] if len(stderr) > 1000 else stderr}")
            return False
    except Exception as e:
        print(f"[SDR Encode] ✗ 编码异常: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将 DPX 序列编码为 SDR 视频")
    parser.add_argument("--input", type=str, required=True, help="DPX 文件目录")
    parser.add_argument("--output", type=str, required=True, help="输出视频路径")
    parser.add_argument("--fps", type=float, default=30.0, help="帧率")
    parser.add_argument("--crf", type=int, default=18, help="质量参数 (18-28)")
    parser.add_argument("--preset", type=str, default="slow", help="编码预设")
    
    args = parser.parse_args()
    
    success = encode_dpx_to_sdr_video(
        args.input, args.output, args.fps, args.crf, args.preset
    )
    
    exit(0 if success else 1)
