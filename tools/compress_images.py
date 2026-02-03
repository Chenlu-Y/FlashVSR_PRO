#!/usr/bin/env python3
"""无损/高质量图片批量压缩工具

支持指定源文件夹、缩放比例，输出为无损 PNG 或高质量 JPEG，保留细节（Lanczos 重采样）。

使用方法：
    python compress_images.py --input_dir /path/to/images --output_dir /path/output --scale 0.5
    python compress_images.py -i ./photos -o ./compressed -s 0.5 --format png

示例：
    # 当前目录图片压缩到 50%，输出到 hq_compressed_all
    python compress_images.py -i . -o hq_compressed_all -s 0.5

    # 指定文件夹，压缩到 30% 尺寸，PNG 无损
    python compress_images.py -i /data/frames -o /data/frames_small -s 0.3 --format png

    # 输出为高质量 JPEG（quality=95，视觉上接近无损）
    python compress_images.py -i ./imgs -o ./out -s 0.5 --format jpg --quality 95
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

try:
    from PIL import Image
except ImportError:
    print("请先安装 Pillow: pip install Pillow")
    sys.exit(1)

# 支持的图片格式
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.tif'}


def find_image_files(input_dir: Path) -> List[Path]:
    """查找输入目录中的所有图片文件"""
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"不是目录: {input_dir}")

    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(input_dir.glob(f'*{ext}'))
        files.extend(input_dir.glob(f'*{ext.upper()}'))
    return sorted(set(files))


def compress_image(
    src: Path,
    dst: Path,
    scale: float,
    out_format: str,
    png_compression: int,
    jpg_quality: int,
) -> Tuple[bool, str]:
    """
    单张图片无损/高质量压缩：Lanczos 缩放 + 无损或高质量编码。
    返回 (成功, 错误信息)。
    """
    try:
        with Image.open(src) as im:
            # 保留 RGBA（若有 alpha），否则 RGB
            if im.mode in ('RGBA', 'LA', 'P'):
                im = im.convert('RGBA')
            elif im.mode != 'RGB':
                im = im.convert('RGB')

            w, h = im.size
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))

            # Lanczos 重采样，保留细节
            resized = im.resize((new_w, new_h), Image.Resampling.LANCZOS)

            dst.parent.mkdir(parents=True, exist_ok=True)

            if out_format == 'png':
                resized.save(dst, 'PNG', compress_level=png_compression)
            else:
                if resized.mode == 'RGBA':
                    resized = resized.convert('RGB')
                resized.save(dst, 'JPEG', quality=jpg_quality, subsampling=0)
        return True, ""
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="批量压缩图片：指定文件夹与缩放比例，无损/高质量输出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i", "--input_dir",
        required=True,
        type=Path,
        help="源图片所在文件夹路径",
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=Path,
        default=None,
        help="输出文件夹路径（默认：<input_dir>/hq_compressed）",
    )
    parser.add_argument(
        "-s", "--scale",
        type=float,
        default=0.5,
        help="缩放比例，例如 0.5 表示宽高各缩到 50%%（默认 0.5）",
    )
    parser.add_argument(
        "--format",
        choices=["png", "jpg", "jpeg"],
        default="png",
        help="输出格式：png 无损，jpg 为有损高质量（默认 png）",
    )
    parser.add_argument(
        "--png_compression",
        type=int,
        default=6,
        choices=range(10),
        metavar="0-9",
        help="PNG 压缩级别 0-9，0 最快/体积大，9 最慢/体积小，均为无损（默认 6）",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        choices=range(1, 101),
        metavar="1-100",
        help="输出为 JPEG 时的质量 1-100（默认 95，建议 90-98 保留细节）",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="若输出文件已存在则跳过",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = input_dir / "hq_compressed"
    output_dir = output_dir.resolve()

    if args.scale <= 0 or args.scale > 1:
        print("错误：--scale 必须在 (0, 1] 之间，例如 0.5 表示缩小到 50%")
        sys.exit(1)

    try:
        files = find_image_files(input_dir)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"错误：{e}")
        sys.exit(1)

    if not files:
        print("未在指定目录中找到图片文件（支持 png/jpg/jpeg/bmp/webp/tiff）")
        sys.exit(1)

    total = len(files)
    print(f"找到 {total} 张图片。源目录: {input_dir}")
    print(f"输出目录: {output_dir}，缩放比例: {args.scale}，格式: {args.format}")
    if args.format == "png":
        print(f"PNG 压缩级别: {args.png_compression}（无损）")
    else:
        print(f"JPEG 质量: {args.quality}")
    print("可随时按 Ctrl+C 停止。\n")

    ext = ".jpg" if args.format in ("jpg", "jpeg") else ".png"
    ok = 0
    for i, src in enumerate(files, 1):
        try:
            rel = src.relative_to(input_dir)
        except ValueError:
            rel = Path(src.name)
        dst = output_dir / rel.parent / (src.stem + ext)

        if args.skip_existing and dst.exists():
            print(f"[{i}/{total}] 跳过（已存在）: {src.name}")
            ok += 1
            continue

        print(f"[{i}/{total}] 处理: {src.name}")
        success, err = compress_image(
            src,
            dst,
            scale=args.scale,
            out_format=args.format,
            png_compression=args.png_compression,
            jpg_quality=args.quality,
        )
        if success:
            ok += 1
        else:
            print(f"  失败: {err}")

    print(f"\n完成：成功 {ok}/{total}，输出目录: {output_dir}")


if __name__ == "__main__":
    main()
