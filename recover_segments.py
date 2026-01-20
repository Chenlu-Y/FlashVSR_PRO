#!/usr/bin/env python3
"""
恢复并合并已保存的分段文件
用法: python recover_segments.py <segment_dir> <output_file>
"""
import os
import sys
import torch
import json
import gc
from pathlib import Path

def recover_segments(segment_dir, output_file, original_length=None):
    """
    恢复并合并分段文件
    
    Args:
        segment_dir: 分段文件目录（如 /tmp/flashvsr_segments/worker_0_308_4x）
        output_file: 输出文件路径
        original_length: 原始帧数（如果知道的话，用于裁剪）
    """
    segment_dir = Path(segment_dir)
    if not segment_dir.exists():
        raise FileNotFoundError(f"Segment directory not found: {segment_dir}")
    
    # 查找所有分段文件
    segment_files = []
    for seg_file in sorted(segment_dir.glob("segment_*.pt")):
        # 尝试读取对应的 JSON 文件获取帧范围信息
        json_file = seg_file.with_suffix('.json')
        start_frame = 0
        end_frame = None
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    info = json.load(f)
                    start_frame = info.get('start_frame', 0)
                    end_frame = info.get('end_frame', None)
            except:
                pass
        
        # 从文件名提取索引
        seg_idx = int(seg_file.stem.split('_')[-1])
        segment_files.append((seg_idx, str(seg_file), start_frame, end_frame))
    
    if not segment_files:
        raise ValueError(f"No segment files found in {segment_dir}")
    
    # 按索引排序
    segment_files.sort(key=lambda x: x[0])
    
    print(f"[Recover] Found {len(segment_files)} segments in {segment_dir}")
    for seg_idx, seg_file, start_frame, end_frame in segment_files:
        print(f"  - segment_{seg_idx:04d}.pt (frames {start_frame}-{end_frame if end_frame else '?'})")
    
    # 合并分段
    final_output = None
    last_end_frame = None
    
    for seg_idx, segment_file, start_frame, end_frame in segment_files:
        print(f"\n[Recover] Loading segment {seg_idx + 1}/{len(segment_files)}: {os.path.basename(segment_file)}")
        segment = torch.load(segment_file, map_location='cpu')
        print(f"  Shape: {segment.shape}, Size: {segment.numel() * segment.element_size() / (1024**3):.2f} GB")
        
        # 处理overlap（如果存在）
        if last_end_frame is not None and start_frame < last_end_frame:
            overlap_frames = last_end_frame - start_frame
            if overlap_frames < segment.shape[0]:
                print(f"  Skipping {overlap_frames} overlap frames")
                segment = segment[overlap_frames:]
            elif overlap_frames >= segment.shape[0]:
                print(f"  Warning: Segment {seg_idx} is completely overlapped, skipping")
                del segment
                gc.collect()
                continue
        
        # 合并
        if final_output is None:
            final_output = segment
        else:
            print(f"  Merging with previous output...")
            final_output = torch.cat([final_output, segment], dim=0)
            print(f"  Merged shape: {final_output.shape}")
        
        del segment
        gc.collect()
        
        if end_frame is not None:
            last_end_frame = end_frame
        else:
            # 估算结束帧
            last_end_frame = start_frame + final_output.shape[0] if final_output is not None else 0
    
    if final_output is None:
        raise RuntimeError("No valid segments to merge")
    
    # 裁剪到原始帧数（如果指定）
    if original_length is not None and final_output.shape[0] > original_length:
        print(f"\n[Recover] Trimming from {final_output.shape[0]} to {original_length} frames")
        final_output = final_output[:original_length]
    
    print(f"\n[Recover] Final output shape: {final_output.shape}")
    print(f"[Recover] Saving to {output_file}...")
    
    # 保存结果
    torch.save(final_output, output_file)
    print(f"[Recover] ✓ Saved successfully!")
    
    # 显示文件大小
    file_size = os.path.getsize(output_file) / (1024**3)
    print(f"[Recover] Output file size: {file_size:.2f} GB")
    
    return final_output

def main():
    if len(sys.argv) < 3:
        print("Usage: python recover_segments.py <segment_dir> <output_file> [original_length]")
        print("\nExample:")
        print("  python recover_segments.py /tmp/flashvsr_segments/worker_0_308_4x merged_0_308.pt 308")
        sys.exit(1)
    
    segment_dir = sys.argv[1]
    output_file = sys.argv[2]
    original_length = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    try:
        recover_segments(segment_dir, output_file, original_length)
    except Exception as e:
        print(f"\n[Recover] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
