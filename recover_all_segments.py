#!/usr/bin/env python3
"""
恢复并合并所有 worker 的分段文件
用法: python recover_all_segments.py <base_dir> <output_file> [total_frames]
"""
import os
import sys
import torch
import json
import gc
from pathlib import Path

def recover_worker_segments(segment_dir, worker_name):
    """恢复单个 worker 的所有分段"""
    segment_dir = Path(segment_dir)
    if not segment_dir.exists():
        print(f"[Recover] Warning: {segment_dir} not found, skipping")
        return None
    
    # 查找所有分段文件
    segment_files = []
    for seg_file in sorted(segment_dir.glob("segment_*.pt")):
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
        
        seg_idx = int(seg_file.stem.split('_')[-1])
        segment_files.append((seg_idx, str(seg_file), start_frame, end_frame))
    
    if not segment_files:
        print(f"[Recover] No segments found in {segment_dir}")
        return None
    
    segment_files.sort(key=lambda x: x[0])
    
    print(f"\n[Recover] {worker_name}: Found {len(segment_files)} segments")
    
    # 合并该 worker 的所有分段
    final_output = None
    last_end_frame = None
    
    for seg_idx, segment_file, start_frame, end_frame in segment_files:
        print(f"  Loading segment_{seg_idx:04d}.pt (frames {start_frame}-{end_frame if end_frame else '?'})")
        segment = torch.load(segment_file, map_location='cpu')
        
        # 处理overlap
        if last_end_frame is not None and start_frame < last_end_frame:
            overlap_frames = last_end_frame - start_frame
            if overlap_frames < segment.shape[0]:
                print(f"    Skipping {overlap_frames} overlap frames")
                segment = segment[overlap_frames:]
            elif overlap_frames >= segment.shape[0]:
                print(f"    Warning: Segment completely overlapped, skipping")
                del segment
                gc.collect()
                continue
        
        if final_output is None:
            final_output = segment
        else:
            final_output = torch.cat([final_output, segment], dim=0)
        
        del segment
        gc.collect()
        
        if end_frame is not None:
            last_end_frame = end_frame
    
    if final_output is not None:
        print(f"  {worker_name} merged: {final_output.shape}")
    
    return final_output

def recover_all_segments(base_dir, output_file, total_frames=None):
    """
    恢复并合并所有 worker 的分段
    
    Args:
        base_dir: 分段文件基础目录（如 /tmp/flashvsr_segments）
        output_file: 输出文件路径
        total_frames: 总帧数（用于验证）
    """
    base_dir = Path(base_dir)
    
    # 查找所有 worker 目录
    worker_dirs = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith('worker_'):
            worker_dirs.append(item)
    
    if not worker_dirs:
        raise ValueError(f"No worker directories found in {base_dir}")
    
    worker_dirs.sort(key=lambda x: x.name)
    print(f"[Recover] Found {len(worker_dirs)} worker directories:")
    for wd in worker_dirs:
        print(f"  - {wd.name}")
    
    # 恢复每个 worker 的分段
    worker_outputs = []
    for worker_dir in worker_dirs:
        worker_output = recover_worker_segments(worker_dir, worker_dir.name)
        if worker_output is not None:
            # 从目录名提取帧范围信息
            parts = worker_dir.name.split('_')
            if len(parts) >= 3:
                try:
                    worker_start = int(parts[1])
                    worker_end = int(parts[2].split('_')[0])
                    worker_outputs.append((worker_start, worker_output))
                except:
                    worker_outputs.append((0, worker_output))
            else:
                worker_outputs.append((0, worker_output))
    
    if not worker_outputs:
        raise RuntimeError("No valid segments found")
    
    # 按 worker_start 排序
    worker_outputs.sort(key=lambda x: x[0])
    
    print(f"\n[Recover] Merging {len(worker_outputs)} worker outputs...")
    
    # 合并所有 worker 的输出
    final_output = None
    for worker_start, worker_output in worker_outputs:
        if final_output is None:
            final_output = worker_output
            print(f"  Worker {worker_start}: {final_output.shape}")
        else:
            # 检查是否有重叠
            prev_frames = final_output.shape[0]
            print(f"  Worker {worker_start}: {worker_output.shape}, merging...")
            final_output = torch.cat([final_output, worker_output], dim=0)
            print(f"  Merged: {final_output.shape} (added {final_output.shape[0] - prev_frames} frames)")
        
        del worker_output
        gc.collect()
    
    if final_output is None:
        raise RuntimeError("Failed to merge worker outputs")
    
    # 裁剪到总帧数（如果指定）
    if total_frames is not None and final_output.shape[0] > total_frames:
        print(f"\n[Recover] Trimming from {final_output.shape[0]} to {total_frames} frames")
        final_output = final_output[:total_frames]
    
    print(f"\n[Recover] Final output shape: {final_output.shape}")
    print(f"[Recover] Saving to {output_file}...")
    
    # 保存结果
    torch.save(final_output, output_file)
    print(f"[Recover] ✓ Saved successfully!")
    
    file_size = os.path.getsize(output_file) / (1024**3)
    print(f"[Recover] Output file size: {file_size:.2f} GB")
    
    return final_output

def main():
    if len(sys.argv) < 3:
        print("Usage: python recover_all_segments.py <base_dir> <output_file> [total_frames]")
        print("\nExample:")
        print("  python recover_all_segments.py /tmp/flashvsr_segments merged_output.pt 612")
        print("\nOr recover single worker:")
        print("  python recover_segments.py /tmp/flashvsr_segments/worker_0_308_4x worker0.pt 308")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    output_file = sys.argv[2]
    total_frames = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    try:
        recover_all_segments(base_dir, output_file, total_frames)
    except Exception as e:
        print(f"\n[Recover] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
