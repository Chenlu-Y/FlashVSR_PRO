#!/usr/bin/env python3
"""查找未合并的视频文件

使用方法：
    python find_unmerged.py

这个脚本会扫描临时目录，找出所有未合并的checkpoint和worker文件。
"""

import os
import json
import glob
from pathlib import Path

def load_checkpoint(checkpoint_file):
    """加载checkpoint数据"""
    try:
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    except:
        return None

def scan_checkpoints():
    """扫描所有checkpoint目录"""
    checkpoints = []
    
    # 扫描 /tmp/flashvsr_checkpoints
    checkpoint_base = "/tmp/flashvsr_checkpoints"
    if os.path.exists(checkpoint_base):
        for checkpoint_dir in os.listdir(checkpoint_base):
            checkpoint_path = os.path.join(checkpoint_base, checkpoint_dir)
            if not os.path.isdir(checkpoint_path):
                continue
            
            checkpoint_file = os.path.join(checkpoint_path, "checkpoint.json")
            if os.path.exists(checkpoint_file):
                checkpoint_data = load_checkpoint(checkpoint_file)
                if checkpoint_data:
                    # 检查是否有worker文件
                    worker_files = []
                    for key, value in checkpoint_data.items():
                        if key.startswith('segment_') and isinstance(value, dict):
                            path = value.get('path', '')
                            if path and os.path.exists(path):
                                file_size = os.path.getsize(path) / (1024**3)
                                worker_files.append({
                                    'key': key,
                                    'path': path,
                                    'size_gb': file_size,
                                    'start_idx': value.get('start_idx', 0),
                                    'end_idx': value.get('end_idx', 0),
                                })
                    
                    if worker_files:
                        checkpoints.append({
                            'type': 'multi_gpu',
                            'dir': checkpoint_path,
                            'name': checkpoint_dir,
                            'worker_files': worker_files,
                            'num_workers': len(worker_files),
                        })
    
    # 扫描 /tmp/flashvsr_multigpu
    multigpu_base = "/tmp/flashvsr_multigpu"
    if os.path.exists(multigpu_base):
        for video_dir in os.listdir(multigpu_base):
            video_path = os.path.join(multigpu_base, video_dir)
            if not os.path.isdir(video_path):
                continue
            
            # 查找worker文件
            worker_files = []
            for worker_file in glob.glob(os.path.join(video_path, "worker_*.pt")):
                if os.path.exists(worker_file):
                    file_size = os.path.getsize(worker_file) / (1024**3)
                    worker_files.append({
                        'path': worker_file,
                        'size_gb': file_size,
                    })
            
            if worker_files:
                checkpoints.append({
                    'type': 'multi_gpu_worker',
                    'dir': video_path,
                    'name': video_dir,
                    'worker_files': worker_files,
                    'num_workers': len(worker_files),
                })
    
    return checkpoints

def scan_segments():
    """扫描所有segmented目录"""
    segments = []
    
    segment_base = "/tmp/flashvsr_segments"
    if os.path.exists(segment_base):
        for segment_dir in os.listdir(segment_base):
            segment_path = os.path.join(segment_base, segment_dir)
            if not os.path.isdir(segment_path):
                continue
            
            # 查找segment文件
            segment_files = sorted(glob.glob(os.path.join(segment_path, "segment_*.pt")))
            if segment_files:
                total_size = sum(os.path.getsize(f) for f in segment_files) / (1024**3)
                segments.append({
                    'type': 'segmented',
                    'dir': segment_path,
                    'name': segment_dir,
                    'segment_files': segment_files,
                    'num_segments': len(segment_files),
                    'total_size_gb': total_size,
                })
    
    return segments

def main():
    print("=" * 80)
    print("查找未合并的视频文件")
    print("=" * 80)
    print()
    
    # 扫描checkpoints
    print("扫描 checkpoint 目录...")
    checkpoints = scan_checkpoints()
    
    # 扫描segments
    print("扫描 segmented 目录...")
    segments = scan_segments()
    
    # 显示结果
    if not checkpoints and not segments:
        print("\n未找到任何未合并的文件。")
        return
    
    print(f"\n找到 {len(checkpoints)} 个 multi-GPU checkpoint，{len(segments)} 个 segmented 目录\n")
    
    # 显示multi-GPU checkpoints
    if checkpoints:
        print("=" * 80)
        print("Multi-GPU Checkpoints:")
        print("=" * 80)
        for i, cp in enumerate(checkpoints, 1):
            print(f"\n{i}. {cp['name']}")
            print(f"   类型: {cp['type']}")
            print(f"   目录: {cp['dir']}")
            print(f"   Worker数量: {cp['num_workers']}")
            
            total_size = sum(w['size_gb'] for w in cp['worker_files'])
            print(f"   总大小: {total_size:.2f} GB")
            
            print(f"   Worker文件:")
            for wf in cp['worker_files']:
                if 'start_idx' in wf:
                    print(f"     - {Path(wf['path']).name}: {wf['size_gb']:.2f} GB (frames {wf['start_idx']}-{wf['end_idx']})")
                else:
                    print(f"     - {Path(wf['path']).name}: {wf['size_gb']:.2f} GB")
            
            # 生成恢复命令
            if cp['type'] == 'multi_gpu':
                checkpoint_dir = cp['dir']
            else:
                # 对于multi_gpu_worker类型，需要找到对应的checkpoint
                checkpoint_dir = None
                checkpoint_file = os.path.join(cp['dir'], "..", "flashvsr_checkpoints", cp['name'], "checkpoint.json")
                if os.path.exists(checkpoint_file):
                    checkpoint_dir = os.path.dirname(checkpoint_file)
            
            if checkpoint_dir:
                print(f"\n   恢复命令:")
                print(f"   python recover_multigpu.py \"{checkpoint_dir}\" output.mp4 --fps 30")
                print(f"   # 或者只保存为.pt文件（避免OOM）:")
                print(f"   python recover_multigpu.py \"{checkpoint_dir}\" output.pt --save-pt-only")
    
    # 显示segmented目录
    if segments:
        print("\n" + "=" * 80)
        print("Segmented 目录:")
        print("=" * 80)
        for i, seg in enumerate(segments, 1):
            print(f"\n{i}. {seg['name']}")
            print(f"   目录: {seg['dir']}")
            print(f"   Segment数量: {seg['num_segments']}")
            print(f"   总大小: {seg['total_size_gb']:.2f} GB")
            
            print(f"\n   恢复命令:")
            print(f"   python recover_segments.py \"{seg['dir']}\" output.mp4 --fps 30")
            print(f"   # 或者只保存为.pt文件（避免OOM）:")
            print(f"   python recover_segments.py \"{seg['dir']}\" output.mp4 --save-pt-only")
    
    print("\n" + "=" * 80)
    print("提示:")
    print("  - 如果合并时出现OOM，使用 --save-pt-only 参数只保存为.pt文件")
    print("  - .pt文件可以稍后使用 recover_segments.py 或 recover_multigpu.py 转换为视频")
    print("=" * 80)

if __name__ == "__main__":
    main()
