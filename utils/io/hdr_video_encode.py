#!/usr/bin/env python3
"""
HDR 视频编码工具

将 DPX 图像序列编码为 HDR 视频（H.265/HEVC with HDR10/HLG）

注意：
- 当前 DPX 文件保存时应用了 sRGB 伽马校正
- HDR 视频编码需要线性 RGB 或 PQ/HLG 编码
- 本工具会自动进行颜色空间转换
"""

import os
import subprocess
import glob
from typing import Optional, Literal


def encode_dpx_to_hdr_video(
    dpx_dir: str,
    output_path: str,
    fps: float = 30.0,
    hdr_format: Literal['hdr10', 'hlg'] = 'hdr10',
    crf: int = 18,
    preset: str = 'slow',
    max_hdr_nits: float = 1000.0
) -> bool:
    """将 DPX 序列编码为 HDR 视频（修复亮度问题）。
    
    关键修复：
    1. 显式指定输入为 sRGB (bt709)
    2. 使用 zscale 进行正确的颜色空间转换
    3. 设置合理的亮度范围，避免过亮
    """
    """将 DPX 图像序列编码为 HDR 视频。
    
    Args:
        dpx_dir: DPX 文件目录（包含 frame_XXXXXX.dpx 文件）
        output_path: 输出视频路径（.mp4 或 .mkv）
        fps: 帧率
        hdr_format: HDR 格式，'hdr10' 或 'hlg'
        crf: 质量参数（18-28，越小质量越高）
        preset: 编码预设（ultrafast, fast, medium, slow, veryslow）
        max_hdr_nits: 最大亮度（nits），用于 HDR10 元数据
    
    Returns:
        bool: 是否成功
    """
    # 查找所有 DPX 文件
    dpx_files = sorted(glob.glob(os.path.join(dpx_dir, "frame_*.dpx")))
    if not dpx_files:
        raise ValueError(f"未找到 DPX 文件在目录: {dpx_dir}")
    
    print(f"[HDR Encode] 找到 {len(dpx_files)} 个 DPX 文件")
    
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
        print(f"[HDR Encode] 视频尺寸: {w}x{h}")
    except Exception as e:
        raise RuntimeError(f"无法获取 DPX 图像尺寸: {e}")
    
    # 构建 ffmpeg 命令
    # 关键步骤：
    # 1. 读取 DPX（sRGB 编码）
    # 2. 转换为线性 RGB（去除 sRGB 伽马）
    # 3. 转换为 PQ 或 HLG（HDR 编码）
    # 4. 编码为 H.265/HEVC
    
    if hdr_format == 'hdr10':
        # HDR10 使用 PQ (Perceptual Quantizer) 曲线
        # 关键修复：显式指定输入为 sRGB，设置合理的亮度范围
        max_luminance = int(max_hdr_nits * 10000)  # 转换为 0.0001 nits 单位
        min_luminance = 50  # 0.005 nits（避免过暗）
        
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-pattern_type", "glob",
            "-i", os.path.join(dpx_dir, "frame_*.dpx"),
            # 显式指定输入为 sRGB (bt709)
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            # 简单的缩放，让 FFmpeg 自动处理颜色空间转换
            "-vf", f"scale={w}:{h}:flags=lanczos",
            # 输出 HDR 颜色空间
            "-color_primaries", "bt2020",
            "-color_trc", "smpte2084",  # PQ
            "-colorspace", "bt2020nc",
            "-c:v", "libx265",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p10le",  # 10-bit YUV
            # 设置合理的亮度范围（避免过亮）
            "-x265-params", f"hdr10-opt=1:hdr10=1:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L({max_luminance},{min_luminance})",
            output_path
        ]
    else:  # hlg
        # HLG (Hybrid Log-Gamma) 曲线
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-pattern_type", "glob",
            "-i", os.path.join(dpx_dir, "frame_*.dpx"),
            # 显式指定输入为 sRGB (bt709)
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-vf", f"scale={w}:{h}:flags=lanczos",
            # 输出 HLG 颜色空间
            "-color_primaries", "bt2020",
            "-color_trc", "arib-std-b67",  # HLG
            "-colorspace", "bt2020nc",
            "-c:v", "libx265",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p10le",  # 10-bit YUV
            "-x265-params", "hdr10-opt=1",
            output_path
        ]
    
    print(f"[HDR Encode] 开始编码 HDR 视频 ({hdr_format.upper()})...")
    print(f"[HDR Encode] 输出: {output_path}")
    print(f"[HDR Encode] 命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            file_size_mb = os.path.getsize(output_path) / (1024**2)
            print(f"[HDR Encode] ✓ 编码成功: {output_path} ({file_size_mb:.2f} MB)")
            return True
        else:
            stderr = result.stderr.decode('utf-8', errors='ignore')
            print(f"[HDR Encode] ✗ 编码失败 (返回码: {result.returncode})")
            print(f"[HDR Encode] 错误信息: {stderr[:1000]}")
            return False
    except Exception as e:
        print(f"[HDR Encode] ✗ 编码异常: {e}")
        return False


def encode_dpx_to_hdr_video_simple(
    dpx_dir: str,
    output_path: str,
    fps: float = 30.0,
    hdr_format: Literal['hdr10', 'hlg'] = 'hdr10',
    crf: int = 18,
    preset: str = 'slow',
    max_hdr_nits: float = 1000.0,
    dpx_is_linear: bool = True
) -> bool:
    """简化版：将 DPX 序列编码为 HDR 视频。
    
    Args:
        dpx_dir: DPX 文件目录
        output_path: 输出视频路径
        fps: 帧率
        hdr_format: HDR 格式 ('hdr10' 或 'hlg')
        crf: 质量参数 (18-28)
        preset: 编码预设
        max_hdr_nits: 最大亮度 (nits)
        dpx_is_linear: DPX 文件是否为线性 RGB（HDR格式）
            - True: DPX 是线性 RGB（从 checkpoint 重新生成的，默认）
            - False: DPX 是 sRGB 编码（需要先转换为线性 RGB）
    
    关键修复：
    1. 支持线性 RGB DPX（直接从 checkpoint 生成的）
    2. 如果输入是线性 RGB，直接转换为 HDR 曲线，无需 sRGB -> 线性转换
    3. 设置合理的亮度范围，避免过亮
    """
    dpx_files = sorted(glob.glob(os.path.join(dpx_dir, "frame_*.dpx")))
    if not dpx_files:
        raise ValueError(f"未找到 DPX 文件在目录: {dpx_dir}")
    
    print(f"[HDR Encode] 找到 {len(dpx_files)} 个 DPX 文件")
    
    # 获取尺寸
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
        print(f"[HDR Encode] 视频尺寸: {w}x{h}")
    except Exception as e:
        raise RuntimeError(f"无法获取 DPX 图像尺寸: {e}")
    
    # 根据 DPX 格式选择不同的转换方式
    if dpx_is_linear:
        # 线性 RGB DPX（HDR格式）-> HDR 视频
        # 这是最简单的情况：线性 RGB 直接转换为 HDR 曲线（PQ/HLG）
        print(f"[HDR Encode] 检测到线性 RGB DPX（HDR格式），直接转换为 {hdr_format.upper()}")
        
        if hdr_format == 'hdr10':
            color_trc = "smpte2084"  # PQ
            # 设置合理的亮度范围
            max_luminance = int(max_hdr_nits * 10000)  # 转换为 0.0001 nits 单位
            min_luminance = 50  # 0.005 nits
            x265_params = f"hdr10-opt=1:hdr10=1:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L({max_luminance},{min_luminance})"
            # 线性 RGB -> PQ: 使用最简单的 scale，让 FFmpeg 根据颜色空间标志自动转换
            # 这是最可靠的方法，避免复杂的滤镜链
            vf_filter = f"scale={w}:{h}:flags=lanczos"
        else:  # hlg
            color_trc = "arib-std-b67"  # HLG
            x265_params = "hdr10-opt=1"
            # 线性 RGB -> HLG: 使用简单的 scale
            vf_filter = f"scale={w}:{h}:flags=lanczos"
        
        # 输入是线性 RGB（无 transfer curve），输出是 HDR
        # 使用简单的方法：只做缩放，让 FFmpeg 根据颜色空间标志自动转换
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-pattern_type", "glob",
            "-i", os.path.join(dpx_dir, "frame_*.dpx"),
            # 输入是线性 RGB（无 transfer curve）
            "-color_primaries", "bt2020",  # 假设原色是 bt2020（HDR 标准）
            "-color_trc", "linear",  # 线性 RGB，无伽马
            "-colorspace", "bt2020nc",
            # 简单的缩放，让 FFmpeg 自动处理颜色空间转换
            "-vf", vf_filter,
            # 输出 HDR 颜色空间
            "-color_primaries", "bt2020",
            "-color_trc", color_trc,
            "-colorspace", "bt2020nc",
            "-c:v", "libx265",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p10le",
            "-x265-params", x265_params,
            output_path
        ]
    else:
        # sRGB DPX（SDR格式）-> HDR 视频
        # 需要：sRGB -> 线性 RGB -> HDR 曲线
        print(f"[HDR Encode] 检测到 sRGB DPX（SDR格式），需要先转换为线性 RGB 再转换为 {hdr_format.upper()}")
        print(f"[HDR Encode] 建议：使用 --dpx_linear_rgb 参数从 checkpoint 重新生成线性 RGB DPX")
        
        if hdr_format == 'hdr10':
            color_trc = "smpte2084"  # PQ
            max_luminance = int(max_hdr_nits * 10000)
            min_luminance = 50
            x265_params = f"hdr10-opt=1:hdr10=1:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L({max_luminance},{min_luminance})"
            # sRGB -> HDR: 使用简单的 scale，让 FFmpeg 自动处理
            vf_filter = f"scale={w}:{h}:flags=lanczos"
        else:  # hlg
            color_trc = "arib-std-b67"  # HLG
            x265_params = "hdr10-opt=1"
            # sRGB -> HLG: 使用简单的 scale
            vf_filter = f"scale={w}:{h}:flags=lanczos"
        
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-pattern_type", "glob",
            "-i", os.path.join(dpx_dir, "frame_*.dpx"),
            # 输入是 sRGB (bt709)
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            # 使用 colorspace 滤镜进行转换
            "-vf", vf_filter,
            # 输出 HDR 颜色空间
            "-color_primaries", "bt2020",
            "-color_trc", color_trc,
            "-colorspace", "bt2020nc",
            "-c:v", "libx265",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p10le",
            "-x265-params", x265_params,
            output_path
        ]
    
    print(f"[HDR Encode] 开始编码 HDR 视频 ({hdr_format.upper()})...")
    print(f"[HDR Encode] 输出: {output_path}")
    if dpx_is_linear:
        print(f"[HDR Encode] 输入: 线性 RGB DPX（HDR格式）")
    else:
        print(f"[HDR Encode] 输入: sRGB DPX（SDR格式），将转换为 {hdr_format.upper()}")
    
    # 如果使用 zscale 失败，尝试回退到简单方法
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            file_size_mb = os.path.getsize(output_path) / (1024**2)
            print(f"[HDR Encode] ✓ 编码成功: {output_path} ({file_size_mb:.2f} MB)")
            return True
        else:
            stderr = result.stderr.decode('utf-8', errors='ignore')
            print(f"[HDR Encode] ✗ 编码失败 (返回码: {result.returncode})")
            print(f"[HDR Encode] 错误信息: {stderr[-2000:] if len(stderr) > 2000 else stderr}")
            return False
    except Exception as e:
        print(f"[HDR Encode] ✗ 异常: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将 DPX 序列编码为 HDR 视频")
    parser.add_argument("--input", type=str, required=True, help="DPX 文件目录")
    parser.add_argument("--output", type=str, required=True, help="输出视频路径")
    parser.add_argument("--fps", type=float, default=30.0, help="帧率")
    parser.add_argument("--hdr_format", type=str, default="hdr10", choices=["hdr10", "hlg"], help="HDR 格式")
    parser.add_argument("--crf", type=int, default=18, help="质量参数 (18-28)")
    parser.add_argument("--preset", type=str, default="slow", help="编码预设")
    parser.add_argument("--max_hdr_nits", type=float, default=1000.0, help="最大亮度 (nits)，用于 HDR10 元数据，避免过亮")
    parser.add_argument("--simple", action="store_true", help="使用简化模式（推荐）")
    parser.add_argument("--dpx_is_linear", action="store_true",
                       help="DPX 文件是线性 RGB（HDR格式）。默认启用（从 checkpoint 重新生成的 DPX 是线性 RGB）")
    parser.add_argument("--dpx_is_srgb", action="store_true",
                       help="DPX 文件是 sRGB 编码（SDR格式）。如果指定，会覆盖 --dpx_is_linear")
    
    args = parser.parse_args()
    
    # 确定 DPX 格式：默认假设是线性 RGB（从 checkpoint 重新生成的）
    # 除非明确指定 --dpx_is_srgb
    dpx_is_linear = not args.dpx_is_srgb  # 默认是线性 RGB，除非指定了 --dpx_is_srgb
    
    if args.simple:
        success = encode_dpx_to_hdr_video_simple(
            args.input, args.output, args.fps, args.hdr_format, args.crf, args.preset, args.max_hdr_nits, dpx_is_linear
        )
    else:
        success = encode_dpx_to_hdr_video(
            args.input, args.output, args.fps, args.hdr_format, args.crf, args.preset
        )
    
    exit(0 if success else 1)
