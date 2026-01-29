#!/usr/bin/env python3
"""
HDR 工作流预览工具

预览 HDR 视频在 AI 超分前后的效果：
1. 超分前：HDR(PQ) → HLG 转换后的第一帧
2. 超分后：AI 处理后的第一帧（需要实际运行超分）

使用方法：
    # 预览超分前的第一帧（HLG 转换后）
    python tools/preview_hdr_workflow.py --input your_hdr_video.mov --output /tmp/preview_before.png
    
    # 预览超分前后的对比（需要先运行超分）
    python tools/preview_hdr_workflow.py --input your_hdr_video.mov --sr_output /path/to/sr_output --output /tmp/comparison.png
"""

import os
import sys
import argparse
import subprocess
import numpy as np
from PIL import Image

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_hdr_frame_as_hlg(video_path: str, frame_idx: int = 0) -> np.ndarray:
    """读取 HDR 视频的一帧，转换为 HLG 编码。
    
    这是 AI 超分前看到的数据。
    """
    # 获取视频信息
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "default=noprint_wrappers=1",
        video_path
    ]
    
    probe_result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = probe_result.stdout.decode().strip().split('\n')
    
    width = height = fps = None
    for line in lines:
        if line.startswith('width='):
            width = int(line.split('=')[1])
        elif line.startswith('height='):
            height = int(line.split('=')[1])
        elif line.startswith('r_frame_rate='):
            fps_str = line.split('=')[1]
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
    
    if not all([width, height, fps]):
        raise RuntimeError(f"无法获取视频信息: {video_path}")
    
    print(f"[Preview] 视频尺寸: {width}x{height}, FPS: {fps:.2f}")
    
    # 使用 zscale 做 PQ → HLG 转换
    start_time = frame_idx / fps
    
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-frames:v", "1",
        "-vf", "zscale=t=arib-std-b67:m=bt2020nc:r=full,format=rgb48le",
        "-f", "rawvideo",
        "-pix_fmt", "rgb48le",
        "-"
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        stderr = result.stderr.decode('utf-8', errors='ignore')
        raise RuntimeError(f"FFmpeg 失败: {stderr[-500:]}")
    
    # 解析原始数据
    frame_data = np.frombuffer(result.stdout, dtype=np.uint16)
    expected_size = width * height * 3
    
    if len(frame_data) != expected_size:
        raise RuntimeError(f"数据大小不匹配: 期望 {expected_size}, 实际 {len(frame_data)}")
    
    frame = frame_data.reshape((height, width, 3))
    
    # 归一化到 [0, 1]
    frame_float = frame.astype(np.float32) / 65535.0
    
    print(f"[Preview] HLG 帧范围: [{frame_float.min():.4f}, {frame_float.max():.4f}]")
    
    return frame_float, width, height


def save_frame_as_image(frame: np.ndarray, output_path: str, title: str = None):
    """保存帧为图片。
    
    HLG 数据已经是感知编码的，可以直接显示。
    """
    # HLG 数据在 [0, 1] 范围，直接转换为 8-bit
    frame_clipped = np.clip(frame, 0, 1)
    frame_uint8 = (frame_clipped * 255).astype(np.uint8)
    
    img = Image.fromarray(frame_uint8, mode='RGB')
    img.save(output_path)
    print(f"[Preview] 已保存: {output_path}")


def create_comparison_image(frame_before: np.ndarray, frame_after: np.ndarray, 
                           output_path: str, scale: int = 2):
    """创建超分前后对比图。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    h_before, w_before = frame_before.shape[:2]
    h_after, w_after = frame_after.shape[:2]
    
    # 缩放"前"图到相同尺寸以便对比
    from PIL import Image
    img_before = Image.fromarray((np.clip(frame_before, 0, 1) * 255).astype(np.uint8))
    img_before_resized = img_before.resize((w_after, h_after), Image.Resampling.LANCZOS)
    frame_before_resized = np.array(img_before_resized) / 255.0
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    axes[0].imshow(np.clip(frame_before_resized, 0, 1))
    axes[0].set_title(f'Before SR (upscaled from {w_before}x{h_before})\nRange: [{frame_before.min():.3f}, {frame_before.max():.3f}]')
    axes[0].axis('off')
    
    axes[1].imshow(np.clip(frame_after, 0, 1))
    axes[1].set_title(f'After SR ({w_after}x{h_after})\nRange: [{frame_after.min():.3f}, {frame_after.max():.3f}]')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Preview] 对比图已保存: {output_path}")


def read_sr_output_frame(sr_output_dir: str, frame_idx: int = 0) -> np.ndarray:
    """读取超分输出的帧。
    
    支持 DPX 和 PNG 格式。
    
    注意：HLG 工作流下，DPX 保存的是 HLG 编码的数据，已经是感知编码的。
    读取时不需要额外处理，直接归一化到 [0, 1] 即可。
    """
    import glob
    
    # 查找帧文件
    dpx_pattern = os.path.join(sr_output_dir, f"frame_{frame_idx:06d}.dpx")
    png_pattern = os.path.join(sr_output_dir, f"frame_{frame_idx:06d}.png")
    
    if os.path.exists(dpx_pattern):
        # 读取 DPX（10-bit，存储在 16-bit 容器中）
        # 注意：FFmpeg 读取 DPX 时默认输出 16-bit，高 10 位有效
        cmd = [
            "ffmpeg", "-y",
            "-i", dpx_pattern,
            "-f", "rawvideo",
            "-pix_fmt", "rgb48le",
            "-"
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            raise RuntimeError(f"无法读取 DPX: {dpx_pattern}")
        
        # 获取尺寸
        probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=width,height", "-of", "csv=p=0", dpx_pattern]
        probe_result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        w, h = map(int, probe_result.stdout.decode().strip().split(','))
        
        frame_data = np.frombuffer(result.stdout, dtype=np.uint16).reshape((h, w, 3))
        
        # 10-bit DPX 在 16-bit 容器中，值范围是 0-65535（FFmpeg 自动扩展）
        # 归一化到 [0, 1]
        frame_float = frame_data.astype(np.float32) / 65535.0
        
        print(f"[Preview] 读取 DPX: {dpx_pattern}")
        print(f"[Preview] 超分后帧范围: [{frame_float.min():.4f}, {frame_float.max():.4f}]")
        print(f"[Preview] 注意：HLG 工作流下，DPX 保存的是 HLG 编码数据（已经是感知编码）")
        
        return frame_float
    
    elif os.path.exists(png_pattern):
        # 读取 PNG
        img = Image.open(png_pattern)
        frame = np.array(img).astype(np.float32) / 255.0
        
        print(f"[Preview] 读取 PNG: {png_pattern}")
        print(f"[Preview] 超分后帧范围: [{frame.min():.4f}, {frame.max():.4f}]")
        
        return frame
    
    else:
        # 尝试找任何帧文件
        all_dpx = sorted(glob.glob(os.path.join(sr_output_dir, "frame_*.dpx")))
        all_png = sorted(glob.glob(os.path.join(sr_output_dir, "frame_*.png")))
        
        if all_dpx:
            print(f"[Preview] 找到 {len(all_dpx)} 个 DPX 文件，使用第一个")
            return read_sr_output_frame(sr_output_dir, 0)
        elif all_png:
            print(f"[Preview] 找到 {len(all_png)} 个 PNG 文件，使用第一个")
            return read_sr_output_frame(sr_output_dir, 0)
        else:
            raise FileNotFoundError(f"未找到超分输出文件: {sr_output_dir}")


def main():
    parser = argparse.ArgumentParser(description="HDR 工作流预览工具")
    parser.add_argument("--input", type=str, required=True, help="输入 HDR 视频路径")
    parser.add_argument("--sr_output", type=str, default=None, help="超分输出目录（可选，用于对比）")
    parser.add_argument("--output", type=str, default="/tmp/hdr_preview.png", help="输出图片路径")
    parser.add_argument("--frame", type=int, default=0, help="要预览的帧索引")
    parser.add_argument("--scale", type=int, default=2, help="超分倍数")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HDR 工作流预览")
    print("=" * 60)
    
    # 读取超分前的帧（HLG 转换后）
    print("\n[Step 1] 读取 HDR 视频并转换为 HLG...")
    frame_before, w, h = read_hdr_frame_as_hlg(args.input, args.frame)
    
    if args.sr_output and os.path.exists(args.sr_output):
        # 有超分输出，创建对比图
        print("\n[Step 2] 读取超分输出...")
        frame_after = read_sr_output_frame(args.sr_output, args.frame)
        
        print("\n[Step 3] 创建对比图...")
        create_comparison_image(frame_before, frame_after, args.output, args.scale)
    else:
        # 只保存超分前的帧
        print("\n[Step 2] 保存超分前的帧...")
        save_frame_as_image(frame_before, args.output, "Before SR (HLG)")
        
        # 同时保存一个单独的文件
        before_path = args.output.replace('.png', '_before_sr.png')
        save_frame_as_image(frame_before, before_path)
    
    print("\n" + "=" * 60)
    print("预览完成！")
    print(f"这是 AI 超分模型实际看到的输入图像。")
    print(f"如果这个图像看起来正常（色彩自然、对比度合适），")
    print(f"那么 AI 超分的输入就是正确的。")
    print("=" * 60)


if __name__ == "__main__":
    main()
