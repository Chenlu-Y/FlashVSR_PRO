#!/usr/bin/env python3
"""
预览 Tone Mapping 效果

功能：
1. 读取 HDR 视频/DPX 的指定帧
2. 应用当前代码的 Tone Mapping（logarithmic/reinhard）
3. 保存为 PNG 对比图（原始 HDR clip 到 0-1 vs Tone Mapped）

用法：
    python tools/preview_tone_mapping.py --input <hdr_video_or_dpx_dir> [options]
    
示例：
    # 预览视频的第 0 帧
    python tools/preview_tone_mapping.py --input /path/to/hdr_video.mp4
    
    # 预览 DPX 序列的第 100 帧
    python tools/preview_tone_mapping.py --input /path/to/dpx_folder --frame 100
    
    # 使用 Mu-law 对比（固定 μ=5000）
    python tools/preview_tone_mapping.py --input /path/to/hdr_video.mp4 --compare_mulaw
    
    # 指定曝光和输出路径
    python tools/preview_tone_mapping.py --input video.mp4 --exposure 1.0 --output preview.png
"""

import os
import sys
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import cv2


def read_hdr_frame(input_path: str, frame_idx: int = 0, use_hlg: bool = True) -> tuple:
    """读取 HDR 帧（视频或 DPX 序列）
    
    Args:
        input_path: 输入路径
        frame_idx: 帧索引
        use_hlg: 是否使用 HLG 转换（推荐）
    
    Returns:
        (frame, fps): frame 是 (H, W, 3) float32
    """
    from utils.io.hdr_io import read_dpx_frame, read_hdr_video_frame_range
    
    if os.path.isfile(input_path):
        # HDR 视频
        print(f"[INFO] Reading HDR video: {input_path}, frame {frame_idx}")
        print(f"[INFO] Using HLG conversion: {use_hlg}")
        frames, fps = read_hdr_video_frame_range(input_path, frame_idx, frame_idx + 1, convert_to_hlg=use_hlg)
        return frames[0], fps
    elif os.path.isdir(input_path):
        # DPX 序列
        files = os.listdir(input_path)
        dpx_files = sorted([f for f in files if f.lower().endswith('.dpx')])
        if not dpx_files:
            raise ValueError(f"No DPX files in directory: {input_path}")
        
        if frame_idx >= len(dpx_files):
            raise ValueError(f"Frame index {frame_idx} out of range, total {len(dpx_files)} frames")
        
        dpx_path = os.path.join(input_path, dpx_files[frame_idx])
        print(f"[INFO] Reading DPX file: {dpx_path}")
        frame = read_dpx_frame(dpx_path)
        return frame, 30.0  # DPX 默认 30fps
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def logarithmic_tone_map(hdr: np.ndarray, exposure: float = 1.0, l_max: float = None) -> np.ndarray:
    """当前代码使用的对数 Tone Mapping
    
    公式: sdr = log(1 + ldr) / log(1 + l_max)
    
    问题: l_max 是从数据动态计算的，当 l_max 很大时（如 55），
    压缩曲线变得很平缓，导致图像灰淡
    """
    ldr = hdr * exposure
    if l_max is None:
        l_max = ldr.max()
    
    if l_max <= 0:
        return np.zeros_like(hdr)
    
    sdr = np.log1p(ldr) / np.log1p(l_max)
    return np.clip(sdr, 0.0, 1.0)


def mulaw_compress(hdr: np.ndarray, mu: float = 5000.0) -> np.ndarray:
    """标准 Mu-law 压缩（μ=5000，文档推荐）
    
    公式: sdr = ln(1 + μ*x) / ln(1 + μ)
    
    关键: 先归一化到 [0, 1]，再应用固定的 μ=5000 压缩
    这样暗部细节会被大幅提升，直方图更接近 SDR
    """
    # 先归一化到 [0, 1]
    hdr_max = max(hdr.max(), 1.0)
    x = hdr / hdr_max
    x = np.clip(x, 0.0, 1.0)
    
    # 应用 Mu-law: ln(1 + μ*x) / ln(1 + μ)
    sdr = np.log1p(mu * x) / np.log1p(mu)
    return np.clip(sdr, 0.0, 1.0)


def reinhard_tone_map(hdr: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """Reinhard Tone Mapping
    
    公式: sdr = ldr / (1 + ldr)
    """
    ldr = hdr * exposure
    sdr = ldr / (1.0 + ldr)
    return np.clip(sdr, 0.0, 1.0)


def simple_clip(hdr: np.ndarray) -> np.ndarray:
    """简单裁剪到 [0, 1]（作为对比基准）"""
    return np.clip(hdr, 0.0, 1.0)


def normalize_for_display(hdr: np.ndarray) -> np.ndarray:
    """归一化 HDR 用于显示（保持相对亮度关系）"""
    hdr_max = max(hdr.max(), 1.0)
    return np.clip(hdr / hdr_max, 0.0, 1.0)


def apply_srgb_gamma(linear: np.ndarray) -> np.ndarray:
    """将线性 RGB 转换为 sRGB（用于显示）"""
    linear = np.clip(linear, 0.0, 1.0)
    # sRGB 伽马曲线
    srgb = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055
    )
    return np.clip(srgb, 0.0, 1.0)


def create_comparison_image(images: list, titles: list, output_path: str, apply_gamma: list = None):
    """创建对比图
    
    Args:
        images: 图像列表
        titles: 标题列表
        output_path: 输出路径
        apply_gamma: 是否对每个图像应用 sRGB 伽马（列表）
    """
    import matplotlib
    matplotlib.use('Agg')  # 无头模式
    import matplotlib.pyplot as plt
    
    n = len(images)
    if apply_gamma is None:
        apply_gamma = [True] * n  # 默认全部应用
    
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    
    for ax, img, title, gamma in zip(axes, images, titles, apply_gamma):
        if gamma:
            # 线性数据需要应用 sRGB 伽马
            img_display = apply_srgb_gamma(img)
        else:
            # 已经是感知编码的数据，直接显示
            img_display = np.clip(img, 0.0, 1.0)
        ax.imshow(img_display)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Comparison image saved: {output_path}")


def save_individual_images(images: list, names: list, output_dir: str, apply_gamma: list = None):
    """保存单独的图像文件"""
    os.makedirs(output_dir, exist_ok=True)
    if apply_gamma is None:
        apply_gamma = [True] * len(images)
    
    for img, name, gamma in zip(images, names, apply_gamma):
        if gamma:
            img_display = apply_srgb_gamma(img)
        else:
            img_display = np.clip(img, 0.0, 1.0)
        img_uint8 = (img_display * 255).astype(np.uint8)
        path = os.path.join(output_dir, f"{name}.png")
        cv2.imwrite(path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Saved: {path}")


def simple_normalize_gamma(hdr: np.ndarray) -> np.ndarray:
    """最简单的方法：线性归一化 + Gamma 2.2
    
    这可能是最接近"正常"SDR 效果的方法
    """
    # 线性归一化
    normalized = hdr / max(hdr.max(), 1.0)
    # 应用 Gamma 2.2（模拟 SDR 显示）
    gamma = np.power(np.clip(normalized, 0.0, 1.0), 1.0 / 2.2)
    return gamma


def main():
    parser = argparse.ArgumentParser(description="Preview Tone Mapping effect")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="HDR video file or DPX directory")
    parser.add_argument("--frame", "-f", type=int, default=0,
                       help="Frame index to preview (default: 0)")
    parser.add_argument("--exposure", "-e", type=float, default=1.0,
                       help="Exposure adjustment (default: 1.0)")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output image path (default: preview_tone_mapping.png)")
    parser.add_argument("--compare_mulaw", action="store_true", default=True,
                       help="Show Mu-law (mu=5000) comparison (default: True)")
    parser.add_argument("--no_mulaw", action="store_true",
                       help="Disable Mu-law comparison")
    parser.add_argument("--compare_reinhard", action="store_true",
                       help="Show Reinhard comparison")
    parser.add_argument("--method", type=str, default="logarithmic",
                       choices=["logarithmic", "reinhard", "mulaw"],
                       help="Main method to display (default: logarithmic)")
    parser.add_argument("--no_hlg", action="store_true",
                       help="Disable HLG conversion (use old buggy method for comparison)")
    
    args = parser.parse_args()
    
    # 处理 mulaw 开关
    if args.no_mulaw:
        args.compare_mulaw = False
    
    # 默认输出路径
    if args.output is None:
        args.output = os.path.join(_project_root, "preview_tone_mapping.png")
    
    # 读取 HDR 帧
    print("=" * 70)
    print("HDR Tone Mapping Preview Tool")
    print("=" * 70)
    
    use_hlg = not args.no_hlg
    
    try:
        hdr_frame, fps = read_hdr_frame(args.input, args.frame, use_hlg=use_hlg)
    except Exception as e:
        print(f"[ERROR] Read failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    hdr_min = hdr_frame.min()
    hdr_max = hdr_frame.max()
    
    print(f"[INFO] Frame size: {hdr_frame.shape[1]}x{hdr_frame.shape[0]}")
    print(f"[INFO] Value range: [{hdr_min:.4f}, {hdr_max:.4f}]")
    
    if use_hlg:
        print(f"[INFO] *** Using NEW HLG conversion (like your FFmpeg command) ***")
        print(f"[INFO] Values should be in [0, 1] range, looking like SDR")
        if hdr_max <= 1.05:
            print(f"[SUCCESS] Values correctly normalized to [0, 1]!")
        else:
            print(f"[WARN] Values > 1.0, HLG conversion may have issues")
    else:
        print(f"[WARN] Using OLD buggy method (for comparison)")
        if hdr_max > 1.5:
            print(f"[ERROR] Values way out of range ({hdr_max:.2f}), this is the bug!")
    
    # 计算 l_max（旧代码的做法，仅在 no_hlg 模式下有意义）
    l_max = hdr_max * args.exposure
    if not use_hlg:
        print(f"\n[INFO] Old buggy l_max: {l_max:.4f}")
        print(f"[INFO] This caused the washed-out images!")
    
    # 准备对比图
    images = []
    titles = []
    names = []
    apply_gamma_list = []  # 是否需要应用 sRGB 伽马
    
    if use_hlg:
        # 新方法：HLG 转换后，直接就是 SDR 兼容的，不需要额外 Tone Mapping
        images.append(hdr_frame)
        titles.append(f"NEW: HLG Conversion\nRange: [{hdr_min:.3f}, {hdr_max:.3f}]")
        names.append("01_hlg_converted")
        apply_gamma_list.append(False)  # HLG 已经是感知编码
        
        # 保存单独的图像（这是送入 AI 的正确格式）
        output_dir = os.path.dirname(args.output) or "."
        save_individual_images(images, names, output_dir, apply_gamma_list)
        
        # 生成预览图
        create_comparison_image(images, titles, args.output, apply_gamma_list)
        
        print("\n" + "=" * 70)
        print("SUCCESS! HLG Conversion Result:")
        print("=" * 70)
        print(f"  Value range: [{hdr_min:.4f}, {hdr_max:.4f}]")
        print(f"  This is what should be sent to AI model!")
        print(f"\n  The image should look similar to your FFmpeg output:")
        print(f"  ffmpeg -i input.mov -vf \"zscale=t=arib-std-b67:m=bt2020nc:r=limited\" ...")
        print("=" * 70)
        print(f"Preview saved to: {args.output}")
        print(f"AI input saved to: {output_dir}/01_hlg_converted.png")
        print("=" * 70)
        
    else:
        # 旧方法对比（展示问题）
        # 1. 简单方法：线性归一化 + Gamma（最接近正常 SDR）
        simple_tm = simple_normalize_gamma(hdr_frame)
        images.append(simple_tm)
        titles.append(f"Simple: Normalize+Gamma\n(Looks most natural)")
        names.append("01_simple_normalize_gamma")
        apply_gamma_list.append(False)
        
        # 2. 当前代码的 Logarithmic Tone Mapping
        log_tm = logarithmic_tone_map(hdr_frame, args.exposure, l_max)
        images.append(log_tm)
        titles.append(f"OLD BUG: Logarithmic\nl_max={l_max:.2f} (WASHED OUT!)")
        names.append("02_old_logarithmic_bug")
        apply_gamma_list.append(False)
        
        # 3. Mu-law 对比
        if args.compare_mulaw:
            mulaw_tm = mulaw_compress(hdr_frame, mu=5000.0)
            images.append(mulaw_tm)
            titles.append(f"Mu-law (mu=5000)\n(Also washed out)")
            names.append("03_mulaw_5000")
            apply_gamma_list.append(False)
        
        # 4. Reinhard 对比
        if args.compare_reinhard:
            reinhard_tm = reinhard_tone_map(hdr_frame, args.exposure)
            images.append(reinhard_tm)
            titles.append(f"Reinhard\nexp={args.exposure}")
            names.append("04_reinhard")
            apply_gamma_list.append(False)
        
        # 生成对比图
        print(f"\n[INFO] Generating comparison image...")
        create_comparison_image(images, titles, args.output, apply_gamma_list)
        
        # 保存单独的图像
        output_dir = os.path.dirname(args.output) or "."
        save_individual_images(images, names, output_dir, apply_gamma_list)
        
        # 打印统计信息
        print("\n" + "=" * 70)
        print("OLD BUGGY METHOD - Value Range Comparison:")
        print("=" * 70)
        print(f"  Simple (norm+gamma):   [{simple_tm.min():.4f}, {simple_tm.max():.4f}]")
        print(f"  Logarithmic (old bug): [{log_tm.min():.4f}, {log_tm.max():.4f}]")
        if args.compare_mulaw:
            print(f"  Mu-law (mu=5000):      [{mulaw_tm.min():.4f}, {mulaw_tm.max():.4f}]")
        
        print("\n" + "=" * 70)
        print("BUG EXPLANATION:")
        print("=" * 70)
        print(f"""
  The old code had TWO bugs:
  
  1. WRONG NORMALIZATION:
     - rgb48le is 16-bit (0-65535)
     - Old code divided by 1023 instead of 65535
     - Result: values up to {hdr_max:.2f} instead of [0, 1]
  
  2. NO PQ DECODING:
     - HDR video uses PQ (ST 2084) encoding
     - Old code treated PQ values as linear light
     - Result: completely wrong tone mapping
  
  FIX: Use HLG conversion (zscale=t=arib-std-b67)
  Run: python tools/preview_tone_mapping.py --input ... (without --no_hlg)
""")
        print("=" * 70)
        print(f"Preview saved to: {args.output}")
        print("=" * 70)


if __name__ == "__main__":
    main()
