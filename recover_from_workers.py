#!/usr/bin/env python3
"""直接从worker文件恢复并合并视频

使用方法：
    python recover_from_workers.py <worker_dir> <output_path> [--fps FPS] [--save-pt-only] [--segment-overlap N]

示例：
    python recover_from_workers.py /tmp/flashvsr_multigpu/3D_cat_1080_30fps_4x output.mp4 --fps 30
    python recover_from_workers.py /tmp/flashvsr_multigpu/3D_cat_1080_30fps_4x output.pt --save-pt-only
"""

import os
import sys
import argparse
import torch
import glob
import gc
import re
import subprocess

class StreamingVideoWriter:
    """流式视频写入器，支持追加写入帧，用于流式合成。"""
    def __init__(self, path, fps=30, height=None, width=None, codec='libx264'):
        self.path = path
        self.fps = fps
        self.height = height
        self.width = width
        self.codec = codec
        self.process = None
        self.frame_count = 0
        self.initialized = False
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    def _initialize(self, height, width):
        if self.initialized:
            return
        self.height = height
        self.width = width
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', str(self.fps),
            '-i', 'pipe:0', '-c:v', self.codec, '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', '-crf', '18', self.path
        ]
        try:
            self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.initialized = True
            print(f"[StreamingVideoWriter] Initialized: {width}x{height} @ {self.fps}fps")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize streaming video writer: {e}")
    
    def write_frames(self, frames):
        if frames is None or frames.numel() == 0:
            return
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
        if frames.is_cuda:
            frames = frames.cpu()
        if frames.dtype != torch.uint8:
            if frames.max() <= 1.0:
                frames = (frames * 255).clamp(0, 255).to(torch.uint8)
            else:
                frames = frames.clamp(0, 255).to(torch.uint8)
        N, H, W, C = frames.shape
        if not self.initialized:
            self._initialize(H, W)
        if H != self.height or W != self.width:
            raise ValueError(f"Frame size mismatch: expected {self.width}x{self.height}, got {W}x{H}")
        frames_np = frames.numpy()
        for i in range(N):
            frame_bytes = frames_np[i].tobytes()
            try:
                self.process.stdin.write(frame_bytes)
                self.frame_count += 1
            except BrokenPipeError:
                stderr = self.process.stderr.read().decode() if self.process.stderr else ""
                raise RuntimeError(f"FFmpeg process closed unexpectedly. stderr: {stderr}")
    
    def close(self):
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception as e:
                print(f"Warning: Error closing video writer: {e}")
            finally:
                self.process = None
                self.initialized = False
                print(f"[StreamingVideoWriter] Closed. Total frames written: {self.frame_count}")

def extract_frame_range_from_filename(filename):
    """从文件名中提取帧范围（如果可能）"""
    # 尝试从文件名中提取，例如 worker_0_0_308.pt
    match = re.search(r'(\d+)_(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def recover_from_worker_dir(worker_dir, output_path, fps=30, segment_overlap=2, save_pt_only=False):
    """从worker目录恢复并合并视频"""
    
    if not os.path.exists(worker_dir):
        print(f"错误：worker目录不存在: {worker_dir}")
        return False
    
    # 先尝试从checkpoint.json加载帧范围信息
    checkpoint_data = None
    checkpoint_file = None
    
    # 尝试找到对应的checkpoint文件
    # 从worker_dir推断checkpoint路径
    dir_name = os.path.basename(worker_dir.rstrip('/'))
    checkpoint_path = os.path.join('/tmp', 'flashvsr_checkpoints', dir_name, 'checkpoint.json')
    if os.path.exists(checkpoint_path):
        try:
            import json
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            print(f"找到checkpoint文件: {checkpoint_path}")
        except:
            pass
    
    # 查找所有worker文件
    worker_files = sorted(glob.glob(os.path.join(worker_dir, "worker_*.pt")))
    
    if not worker_files:
        print(f"错误：在 {worker_dir} 中未找到worker文件")
        return False
    
    print(f"找到 {len(worker_files)} 个worker文件:")
    for i, wf in enumerate(worker_files):
        file_size = os.path.getsize(wf) / (1024**3)
        print(f"  {i+1}. {os.path.basename(wf)}: {file_size:.2f} GB")
    
    # 从checkpoint或文件名推断帧范围
    worker_data = []
    
    # 首先，从checkpoint构建多种映射（路径、文件名、worker编号）
    path_to_info = {}
    basename_to_info = {}
    worker_num_to_info = {}
    
    if checkpoint_data:
        for key, value in checkpoint_data.items():
            if isinstance(value, dict) and 'path' in value:
                path = value['path']
                basename = os.path.basename(path)
                # 提取worker编号（例如 worker_0_xxx.pt -> 0）
                worker_num = None
                if basename.startswith('worker_'):
                    try:
                        worker_num = int(basename.split('_')[1])
                    except:
                        pass
                
                info = {
                    'start_idx': value.get('start_idx'),
                    'end_idx': value.get('end_idx'),
                    'key': key,
                    'path': path
                }
                
                path_to_info[path] = info
                basename_to_info[basename] = info
                if worker_num is not None:
                    worker_num_to_info[worker_num] = info
    
    for wf in worker_files:
        start_idx = None
        end_idx = None
        basename = os.path.basename(wf)
        
        # 方法1: 从checkpoint.json中查找（按优先级：路径 -> 文件名 -> worker编号）
        if wf in path_to_info:
            start_idx = path_to_info[wf]['start_idx']
            end_idx = path_to_info[wf]['end_idx']
            print(f"  从checkpoint路径匹配: {basename} -> frames {start_idx}-{end_idx}")
        elif basename in basename_to_info:
            start_idx = basename_to_info[basename]['start_idx']
            end_idx = basename_to_info[basename]['end_idx']
            print(f"  从checkpoint文件名匹配: {basename} -> frames {start_idx}-{end_idx}")
        elif basename.startswith('worker_'):
            try:
                worker_num = int(basename.split('_')[1])
                if worker_num in worker_num_to_info:
                    start_idx = worker_num_to_info[worker_num]['start_idx']
                    end_idx = worker_num_to_info[worker_num]['end_idx']
                    print(f"  从checkpoint worker编号匹配: {basename} (worker_{worker_num}) -> frames {start_idx}-{end_idx}")
            except:
                pass
        
        # 方法2: 从文件名提取
        if start_idx is None:
            start_idx, end_idx = extract_frame_range_from_filename(basename)
            if start_idx is not None:
                print(f"  从文件名提取: {basename} -> frames {start_idx}-{end_idx}")
        
        # 方法3: 如果还是无法获取，尝试加载文件获取形状
        if start_idx is None:
            try:
                temp_data = torch.load(wf, map_location='cpu')
                frames = temp_data.shape[0]
                # 使用相对位置作为start_idx（不准确，但可以用于排序）
                start_idx = len(worker_data) * 300
                end_idx = start_idx + frames
                del temp_data
                gc.collect()
                print(f"  从文件内容推断: {basename} -> frames {start_idx}-{end_idx} (估计值)")
            except:
                start_idx = len(worker_data) * 300
                end_idx = start_idx + 300
                print(f"  使用默认值: {basename} -> frames {start_idx}-{end_idx} (默认值)")
        
        worker_data.append({
            'path': wf,
            'start_idx': start_idx if start_idx is not None else 0,
            'end_idx': end_idx if end_idx is not None else 0,
        })
    
    # 按start_idx排序（确保顺序正确）
    worker_data.sort(key=lambda x: x['start_idx'])
    print(f"\n排序后的Worker顺序:")
    for i, wd in enumerate(worker_data):
        print(f"  {i+1}. {os.path.basename(wd['path'])}: frames {wd['start_idx']}-{wd['end_idx']}")
    
    print(f"\nWorker帧范围:")
    for wd in worker_data:
        print(f"  {os.path.basename(wd['path'])}: frames {wd['start_idx']}-{wd['end_idx']}")
    
    # 检查输出格式，决定是否使用流式写入
    use_streaming_video = not (save_pt_only or output_path.endswith('.pt'))
    
    if use_streaming_video:
        print(f"\n使用流式视频写入模式（避免OOM）...")
        try:
            # StreamingVideoWriter已经在文件顶部定义，直接使用
            
            # 先加载第一个worker获取尺寸
            first_worker_path = worker_data[0]['path']
            first_worker = torch.load(first_worker_path, map_location='cpu')
            H, W = first_worker.shape[1:3]
            del first_worker
            gc.collect()
            
            # 初始化流式写入器
            writer = StreamingVideoWriter(output_path, fps=fps, height=H, width=W)
            print(f"初始化流式写入器: {W}x{H} @ {fps}fps")
        except Exception as e:
            print(f"警告：无法初始化流式写入器: {e}，回退到普通合并模式")
            import traceback
            traceback.print_exc()
            use_streaming_video = False
            writer = None
    else:
        writer = None
        print(f"\n流式合并 {len(worker_data)} 个worker文件（避免OOM）...")
    
    # 流式合并
    merged_parts = []
    last_expected_end_idx = 0  # 使用期望的end_idx来判断overlap
    
    for i, wd in enumerate(worker_data):
        path = wd['path']
        start_idx = wd['start_idx']
        end_idx = wd['end_idx']  # 期望的结束帧索引
        
        print(f"\n处理 Worker {i+1}/{len(worker_data)}: {os.path.basename(path)}")
        file_size = os.path.getsize(path) / (1024**3)
        print(f"  文件大小: {file_size:.2f} GB")
        print(f"  期望帧范围: {start_idx}-{end_idx} (共 {end_idx - start_idx} 帧)")
        
        # 加载worker文件
        try:
            output = torch.load(path, map_location='cpu')
            print(f"  加载成功: {output.shape} (实际 {output.shape[0]} 帧)")
        except Exception as e:
            print(f"  错误：加载失败: {e}")
            return False
        
        # 处理overlap（基于期望的end_idx来判断，而不是实际输出帧数）
        overlap_frames = 0
        if last_expected_end_idx > 0 and start_idx < last_expected_end_idx:
            overlap_frames = last_expected_end_idx - start_idx
            print(f"    检测到overlap: last_expected_end_idx={last_expected_end_idx}, start_idx={start_idx}, overlap={overlap_frames}帧")
            if overlap_frames < output.shape[0]:
                output = output[overlap_frames:]
                print(f"    跳过 {overlap_frames} 帧overlap，剩余 {output.shape[0]} 帧")
            elif overlap_frames >= output.shape[0]:
                print(f"    警告: Worker {i+1} 完全重叠，跳过")
                del output
                gc.collect()
                continue
        elif last_expected_end_idx > 0 and start_idx >= last_expected_end_idx:
            # 没有overlap，正常情况
            print(f"    无overlap (last_expected_end_idx={last_expected_end_idx}, start_idx={start_idx})")
        
        # 计算期望的帧数（考虑overlap）
        # 期望的帧数 = end_idx - start_idx，但如果处理了overlap，实际需要的帧数 = end_idx - last_expected_end_idx
        if overlap_frames > 0:
            # 如果处理了overlap，期望的帧数应该是从last_expected_end_idx到end_idx
            expected_frames = end_idx - last_expected_end_idx
        else:
            # 如果没有overlap，期望的帧数就是end_idx - start_idx
            expected_frames = end_idx - start_idx
        
        # 如果输出帧数不足，填充到期望的帧数
        if output.shape[0] < expected_frames:
            missing_frames = expected_frames - output.shape[0]
            print(f"    输出帧数不足: 实际 {output.shape[0]} 帧, 期望 {expected_frames} 帧")
            print(f"    填充 {missing_frames} 帧（使用最后一帧）...")
            last_frame = output[-1:, :, :, :]
            padding_frames = last_frame.repeat(missing_frames, 1, 1, 1)
            output = torch.cat([output, padding_frames], dim=0)
            print(f"    填充完成: {output.shape[0]} 帧")
        elif output.shape[0] > expected_frames:
            # 如果输出帧数超过期望，裁剪到期望的帧数
            output = output[:expected_frames]
            print(f"    裁剪到期望帧数: {output.shape[0]} 帧")
        
        # 如果使用流式视频写入，直接写入，不保存到内存
        if use_streaming_video and writer is not None:
            if output.shape[0] > 0:
                print(f"    流式写入 {output.shape[0]} 帧到视频...")
                writer.write_frames(output)
                del output
                gc.collect()
                last_expected_end_idx = end_idx
        else:
            # 添加到合并列表（output已经处理过overlap和填充）
            if output.shape[0] > 0:
                merged_parts.append(output)
                # 更新last_expected_end_idx：使用期望的end_idx，而不是实际输出的帧数
                last_expected_end_idx = end_idx
    
    # 如果使用流式视频写入，直接关闭写入器
    if use_streaming_video and writer is not None:
        writer.close()
        file_size_gb = os.path.getsize(output_path) / (1024**3)
        print(f"\n成功！视频已保存到: {output_path}")
        print(f"文件大小: {file_size_gb:.2f} GB")
        return True
    
    # 最终合并（直接合并，因为只有2个部分）
    print(f"\n最终合并 {len(merged_parts)} 个部分...")
    if len(merged_parts) == 0:
        print("错误：没有可合并的输出")
        return False
    
    if len(merged_parts) == 1:
        final_output = merged_parts[0]
    else:
        # 对于2个部分，直接合并
        print(f"  合并 {len(merged_parts)} 个部分...")
        print(f"  部分1形状: {merged_parts[0].shape}")
        print(f"  部分2形状: {merged_parts[1].shape}")
        # 先强制垃圾回收
        gc.collect()
        final_output = torch.cat(merged_parts, dim=0)
        print(f"  合并成功: {final_output.shape}")
        # 立即释放原始parts
        del merged_parts
        gc.collect()
    
    # 检查是否需要填充到期望的总帧数
    if checkpoint_data:
        max_end_idx = max(v.get('end_idx', 0) for v in checkpoint_data.values() if isinstance(v, dict))
        expected_frames = max_end_idx
        if final_output.shape[0] < expected_frames:
            missing_frames = expected_frames - final_output.shape[0]
            print(f"\n检测到帧数不足: 实际 {final_output.shape[0]} 帧, 期望 {expected_frames} 帧")
            print(f"填充 {missing_frames} 帧（使用最后一帧）...")
            last_frame = final_output[-1:, :, :, :]
            padding_frames = last_frame.repeat(missing_frames, 1, 1, 1)
            final_output = torch.cat([final_output, padding_frames], dim=0)
            print(f"填充完成: {final_output.shape[0]} 帧")
    
    print(f"最终输出形状: {final_output.shape}, 总帧数: {final_output.shape[0]}")
    
    # 保存结果（使用流式保存，避免内存峰值）
    if save_pt_only or output_path.endswith('.pt'):
        print(f"\n保存为torch文件: {output_path}")
        try:
            # 使用流式保存：如果merged_parts有多个，逐个追加保存
            if len(merged_parts) > 1:
                print(f"使用流式保存（{len(merged_parts)}个部分）...")
                # 先保存第一部分
                torch.save(merged_parts[0], output_path)
                print(f"  已保存第1部分: {merged_parts[0].shape}")
                del merged_parts[0]
                gc.collect()
                
                # 然后逐个加载并追加（对于.pt文件，我们需要重新加载并合并）
                # 实际上，对于.pt文件，我们无法直接追加，所以还是需要合并
                # 但我们可以分批合并和保存
                print("警告：.pt文件不支持流式追加，需要完整合并后保存")
                print("如果内存不足，请考虑使用视频格式输出")
            
            # 最终保存
            torch.save(final_output, output_path)
            file_size_gb = os.path.getsize(output_path) / (1024**3)
            print(f"成功！已保存到: {output_path}")
            print(f"文件大小: {file_size_gb:.2f} GB")
            return True
        except Exception as e:
            print(f"错误：保存失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        # 保存为视频：使用流式写入
        print("\n使用流式写入模式保存视频（避免OOM）...")
        try:
            # StreamingVideoWriter已经在文件顶部定义，直接使用
            
            H, W = final_output.shape[1:3]
            writer = StreamingVideoWriter(output_path, fps=fps, height=H, width=W)
            print(f"初始化流式写入器: {W}x{H} @ {fps}fps")
            
            # 分批写入（避免一次性写入所有帧）
            batch_size = 50  # 每次写入50帧
            for i in range(0, final_output.shape[0], batch_size):
                batch = final_output[i:i+batch_size]
                writer.write_frames(batch)
                print(f"  已写入 {min(i+batch_size, final_output.shape[0])}/{final_output.shape[0]} 帧")
                del batch
                gc.collect()
            
            writer.close()
            file_size_gb = os.path.getsize(output_path) / (1024**3)
            print(f"成功！视频已保存到: {output_path}")
            print(f"文件大小: {file_size_gb:.2f} GB")
            return True
            
        except Exception as e:
            print(f"错误：保存视频失败: {e}")
            import traceback
            traceback.print_exc()
            # 尝试保存为.pt文件作为备份
            backup_path = output_path.replace('.mp4', '.pt').replace('.avi', '.pt')
            print(f"\n尝试保存为torch文件作为备份: {backup_path}")
            try:
                torch.save(final_output, backup_path)
                file_size_gb = os.path.getsize(backup_path) / (1024**3)
                print(f"已保存为torch文件: {backup_path}")
                print(f"文件大小: {file_size_gb:.2f} GB")
            except Exception as e2:
                print(f"保存torch文件也失败: {e2}")
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从worker文件恢复并合并视频")
    parser.add_argument("worker_dir", type=str, help="Worker文件目录路径")
    parser.add_argument("output_path", type=str, help="输出视频或.pt文件路径")
    parser.add_argument("--fps", type=float, default=30, help="视频帧率 (默认: 30)")
    parser.add_argument("--segment-overlap", type=int, default=2, help="Segment重叠帧数 (默认: 2)")
    parser.add_argument("--save-pt-only", action="store_true", help="只保存为.pt文件，不转换为视频（避免OOM）")
    
    args = parser.parse_args()
    
    success = recover_from_worker_dir(args.worker_dir, args.output_path, args.fps, args.segment_overlap, args.save_pt_only)
    sys.exit(0 if success else 1)
