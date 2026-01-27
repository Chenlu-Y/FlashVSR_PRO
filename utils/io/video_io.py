#!/usr/bin/env python3
"""
从恢复的推理结果保存为最终视频
用法: 
  python save_recovered_video.py <recovered_pt_file> [original_input_video] [fps]
  
如果不提供 original_input_video，会尝试从checkpoint查找，或使用默认名称
"""
import os
import sys
import subprocess
import tempfile
import numpy as np
import torch
import cv2
import torchvision

def log(msg, level="info"):
    """简单的日志函数"""
    prefix = {
        "info": "[INFO]",
        "warning": "[WARN]",
        "error": "[ERROR]",
        "finish": "[DONE]"
    }.get(level, "[INFO]")
    print(f"{prefix} {msg}")


def save_frame_as_dpx10(frame: np.ndarray, path: str, hdr_max: float = None, apply_srgb_gamma: bool = True) -> bool:
    """将单帧 float RGB (H,W,3) 保存为 10-bit DPX。
    
    支持 SDR [0,1] 和 HDR (>1) 输入。
    
    Args:
        frame: 输入帧 (H,W,3)，可能是 HDR（值 > 1），应该是线性RGB
        path: 输出路径
        hdr_max: HDR 全局最大值（用于归一化）。如果为 None，使用帧内最大值（每帧独立归一化）。
                如果提供，使用全局归一化（保留绝对亮度关系）。
        apply_srgb_gamma: 是否应用 sRGB 伽马校正（默认 True）
            - True: 保存为 sRGB 编码（SDR格式），在标准显示器上正确显示
            - False: 保存为线性 RGB（HDR格式），用于生成 HDR 视频
    
    依赖 FFmpeg，优先用 gbrp10le；若不支持则回退到 rgb48le（10bit 置高位的 16bit 容器）。
    
    注意：
    - 输入应该是线性RGB（经过逆色调映射后的HDR值）
    - 如果 apply_srgb_gamma=True：保存为 sRGB 编码（SDR），在标准显示器上正确显示
    - 如果 apply_srgb_gamma=False：保存为线性 RGB（HDR），用于生成 HDR 视频
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"save_frame_as_dpx10: need (H,W,3) RGB, got shape {frame.shape}")
    h, w = frame.shape[0], frame.shape[1]
    frame = np.ascontiguousarray(frame.astype(np.float32))
    
    # 处理 HDR：归一化到 [0, 1]
    frame_max = frame.max()
    if frame_max > 1.0:
        # HDR 输入：使用全局最大值或帧内最大值归一化
        if hdr_max is not None and hdr_max > 1.0:
            # 使用全局最大值（保留绝对亮度关系）
            frame = frame / hdr_max
        else:
            # 使用帧内最大值（每帧独立归一化，丢失绝对亮度）
            frame = frame / frame_max
    else:
        # SDR 输入：确保在 [0, 1]
        frame = np.clip(frame, 0.0, 1.0)
    
    # 关键修复：根据选项决定是否应用 sRGB 伽马校正
    if apply_srgb_gamma:
        # 应用 sRGB 伽马校正（线性RGB -> sRGB）
        # 这解决了"灰色视频"问题：DPX保存的是线性RGB，但播放器期望sRGB
        # sRGB伽马曲线：对于 x <= 0.0031308，使用线性；否则使用 1.055 * x^(1/2.4) - 0.055
        linear_threshold = 0.0031308
        srgb_gamma = 1.0 / 2.4
        
        # 应用sRGB伽马校正
        frame_encoded = np.where(
            frame <= linear_threshold,
            frame * 12.92,  # 线性部分
            1.055 * np.power(frame, srgb_gamma) - 0.055  # 非线性部分
        )
        frame_encoded = np.clip(frame_encoded, 0.0, 1.0)
    else:
        # 保持线性 RGB（HDR格式）
        # 不应用伽马校正，保持线性编码
        frame_encoded = np.clip(frame, 0.0, 1.0)
    
    # 10-bit: 0–1023（按照图片建议的方式）
    f10 = (frame_encoded * 1023.0).round().clip(0, 1023).astype(np.uint16)
    # gbrp10le: planar G,B,R，每通道 H*W 个 uint16 LE
    g = f10[:, :, 1].tobytes()
    b = f10[:, :, 2].tobytes()
    r = f10[:, :, 0].tobytes()
    raw = g + b + r
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "gbrp10le",
        "-s", f"{w}x{h}", "-r", "1",
        "-i", "pipe:0",
        "-frames:v", "1", "-c:v", "dpx",
        path,
    ]
    try:
        p = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        p.stdin.write(raw)
        p.stdin.close()
        p.wait()
        if p.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 0:
            return True
    except Exception:
        pass
    # 回退：rgb48le，10bit 置于高 10 位
    f16 = (f10.astype(np.uint32) << 6).astype(np.uint16)
    raw16 = np.ascontiguousarray(f16).tobytes()
    cmd2 = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb48le",
        "-s", f"{w}x{h}", "-r", "1",
        "-i", "pipe:0",
        "-frames:v", "1", "-c:v", "dpx",
        path,
    ]
    try:
        p2 = subprocess.Popen(
            cmd2, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        p2.stdin.write(raw16)
        p2.stdin.close()
        p2.wait()
        return p2.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 0
    except Exception:
        return False


def save_video_streaming(frames_tensor, path, fps=30, batch_size=10):
    """分批保存视频，避免一次性处理所有帧"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    num_frames, h, w, c = frames_tensor.shape
    log(f"准备分批保存视频: {num_frames} 帧, {w}x{h}, {fps} fps (每批 {batch_size} 帧)", "info")
    
    # 使用 FFmpeg 管道进行流式编码
    import subprocess
    
    # 启动 FFmpeg 进程
    cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', 'pipe:0',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-crf', '18',
        path
    ]
    
    try:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        
        frame_count = 0
        # 分批处理
        for batch_start in range(0, num_frames, batch_size):
            batch_end = min(batch_start + batch_size, num_frames)
            batch = frames_tensor[batch_start:batch_end]
            
            # 转换为numpy并写入
            batch_np = (batch.clamp(0, 1) * 255).byte().cpu().numpy().astype('uint8')
            for frame in batch_np:
                process.stdin.write(frame.tobytes())
                frame_count += 1
            
            if batch_end % 50 == 0 or batch_end == num_frames:
                log(f"已处理 {frame_count}/{num_frames} 帧 ({frame_count*100//num_frames}%)...", "info")
        
        process.stdin.close()
        process.wait()
        
        if process.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 0:
            log(f"成功使用 FFmpeg 流式保存视频: {path} ({frame_count} 帧)", "info")
            return True
        else:
            stderr = process.stderr.read().decode('utf-8', errors='ignore')
            raise RuntimeError(f"FFmpeg 失败: {stderr[:500]}")
    except Exception as e:
        log(f"流式保存失败: {e}，回退到普通方法", "warning")
        return False

def save_video(frames, path, fps=30):
    """Save tensor video frames to MP4 with H.264 encoding for Windows compatibility.
    
    Priority: FFmpeg subprocess (H.264) > torchvision.io.write_video > OpenCV with H.264 > OpenCV with mp4v
    frames is (F, H, W, C) format.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    # Convert frames to numpy for all methods
    frames_np = (frames.clamp(0, 1) * 255).byte().cpu().numpy().astype('uint8')
    h, w = frames_np.shape[1:3]
    num_frames = frames_np.shape[0]
    
    log(f"准备保存视频: {num_frames} 帧, {w}x{h}, {fps} fps", "info")
    
    # Method 1: Try FFmpeg subprocess (most reliable, guaranteed H.264)
    try:
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
            log(f"成功使用 FFmpeg (H.264) 保存视频: {path}", "info")
            return True
        else:
            log(f"FFmpeg 失败: {result.stderr.decode('utf-8', errors='ignore')[:200]}", "warning")
    except FileNotFoundError:
        log("FFmpeg 未找到，尝试 torchvision...", "warning")
    except Exception as e:
        log(f"FFmpeg 失败: {e}，尝试 torchvision...", "warning")
    
    # Method 2: Try torchvision.io.write_video (uses H.264, best Windows compatibility)
    try:
        # frames is (F, H, W, C), convert to (F, C, H, W) for torchvision
        frames_torch = frames.permute(0, 3, 1, 2).clamp(0, 1)  # (F, C, H, W)
        frames_torch = (frames_torch * 255).byte().cpu()
        
        # Verify shape is correct: should be (F, C, H, W)
        if len(frames_torch.shape) != 4 or frames_torch.shape[1] != 3:
            raise ValueError(f"Invalid tensor shape for torchvision: {frames_torch.shape}, expected (F, 3, H, W)")
        
        # torchvision.io.write_video uses H.264 codec by default
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
            log(f"成功使用 torchvision (H.264) 保存视频: {path}", "info")
            return True
        else:
            log("torchvision 创建了空文件，回退到 OpenCV", "warning")
    except Exception as e:
        log(f"torchvision.write_video 失败: {e}，回退到 OpenCV", "warning")
    
    # Method 3: OpenCV (fallback)
    fourcc_options = [
        ('H264', cv2.VideoWriter_fourcc(*'H264')),
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
    ]
    
    for fourcc_str, fourcc in fourcc_options:
        try:
            writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
            if not writer.isOpened():
                log(f"OpenCV 无法使用 {fourcc_str} 打开 writer", "warning")
                continue
            
            for frame in frames_np:
                # OpenCV uses BGR, so convert RGB to BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
            
            writer.release()
            
            if os.path.exists(path) and os.path.getsize(path) > 0:
                log(f"成功使用 OpenCV ({fourcc_str}) 保存视频: {path}", "info")
                return True
            else:
                log(f"使用 {fourcc_str} 创建了空文件，尝试下一个编码器", "warning")
        except Exception as e:
            log(f"OpenCV 使用 {fourcc_str} 失败: {e}，尝试下一个编码器", "warning")
    
    raise RuntimeError(f"所有方法都失败，无法保存视频: {path}")

def find_original_input(segment_dir):
    """尝试从checkpoint或segment metadata中找到原始输入视频路径"""
    import glob
    import json
    
    # 方法1: 查找checkpoint文件
    checkpoint_dirs = [
        '/tmp/flashvsr_checkpoints',
        '/tmp'
    ]
    
    for checkpoint_dir in checkpoint_dirs:
        if not os.path.exists(checkpoint_dir):
            continue
        for checkpoint_file in glob.glob(os.path.join(checkpoint_dir, '**', '*.json'), recursive=True):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        # 查找input字段
                        input_path = data.get('input') or data.get('input_path') or data.get('video_path')
                        if input_path and os.path.exists(input_path):
                            return input_path
            except:
                continue
    
    # 方法2: 从segment目录名推断（如果可能）
    # worker_0_308_4x 这种格式不包含文件名，无法推断
    
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python save_recovered_video.py <recovered_pt_file> [original_input_video] [fps]")
        print("\n如果不提供 original_input_video，会尝试自动查找或使用默认名称")
        print("\nExample:")
        print("  python save_recovered_video.py /tmp/recovered_output.pt")
        print("  python save_recovered_video.py /tmp/recovered_output.pt /path/to/original.mp4")
        print("  python save_recovered_video.py /tmp/recovered_output.pt /path/to/original.mp4 30")
        sys.exit(1)
    
    recovered_file = sys.argv[1]
    
    # 解析参数：第二个参数可能是原始输入路径或FPS
    original_input = None
    fps = 30.0
    
    if len(sys.argv) > 2:
        arg2 = sys.argv[2]
        if os.path.exists(arg2):
            original_input = arg2
            fps = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
        elif arg2.replace('.', '').isdigit():
            fps = float(arg2)
        else:
            original_input = arg2  # 即使不存在也尝试使用
    
    # 确定输出路径
    if original_input and os.path.exists(original_input):
        # 使用原始文件名，保存到 /app/output
        original_basename = os.path.basename(original_input)
        original_name = os.path.splitext(original_basename)[0]
        output_video = os.path.join("/app/output", f"{original_name}_out.mp4")
        log(f"使用原始输入名称: {output_video}", "info")
    else:
        # 尝试从checkpoint查找原始输入
        segment_base = os.path.dirname(os.path.dirname(recovered_file)) if '/flashvsr_segments' in recovered_file else None
        if segment_base:
            found_input = find_original_input(segment_base)
            if found_input:
                original_basename = os.path.basename(found_input)
                original_name = os.path.splitext(original_basename)[0]
                output_video = os.path.join("/app/output", f"{original_name}_out.mp4")
                log(f"从checkpoint找到原始输入，使用: {output_video}", "info")
            else:
                output_video = os.path.join("/app/output", "recovered_output.mp4")
                log(f"未找到原始输入，使用默认名称: {output_video}", "info")
        else:
            output_video = os.path.join("/app/output", "recovered_output.mp4")
            log(f"使用默认输出路径: {output_video}", "info")
    
    if not os.path.exists(recovered_file):
        log(f"错误: 文件不存在: {recovered_file}", "error")
        sys.exit(1)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_video) if os.path.dirname(output_video) else "/app/output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        log(f"加载恢复的推理结果: {recovered_file}", "info")
        
        # 使用流式加载和保存，避免一次性加载115GB到内存
        log("使用流式处理模式（避免内存溢出）...", "info")
        
        # 先加载数据（由于是.pt文件，需要完整加载）
        log("读取文件数据（这可能需要几分钟，请耐心等待）...", "info")
        checkpoint = torch.load(recovered_file, map_location='cpu')
        
        # 如果是字典，提取tensor
        if isinstance(checkpoint, dict):
            # 尝试找到tensor
            frames = None
            for key in ['frames', 'output', 'data', 'tensor']:
                if key in checkpoint and isinstance(checkpoint[key], torch.Tensor):
                    frames = checkpoint[key]
                    break
            if frames is None:
                # 取第一个tensor值
                for v in checkpoint.values():
                    if isinstance(v, torch.Tensor):
                        frames = v
                        break
            if frames is None:
                raise ValueError("无法从checkpoint中找到tensor数据")
        else:
            frames = checkpoint
        
        log(f"数据信息: shape={frames.shape}, dtype={frames.dtype}", "info")
        
        # 验证格式
        if len(frames.shape) != 4 or frames.shape[3] != 3:
            raise ValueError(f"无效的帧格式: {frames.shape}，期望 (F, H, W, 3)")
        
        num_frames, h, w, c = frames.shape
        log(f"总帧数: {num_frames}, 分辨率: {w}x{h}", "info")
        
        # 确保值在 [0, 1] 范围内
        if frames.dtype != torch.float32 and frames.dtype != torch.float16:
            frames = frames.float() / 255.0
        else:
            frames = frames.clamp(0, 1)
        
        log(f"保存视频到: {output_video} (FPS: {fps})", "info")
        log("开始编码（这可能需要较长时间，请耐心等待）...", "info")
        
        # 使用分批流式保存（避免一次性处理所有帧）
        if not save_video_streaming(frames, output_video, fps=fps, batch_size=10):
            # 如果流式保存失败，回退到普通方法（但会占用大量内存）
            log("流式保存失败，使用普通方法（需要大量内存）...", "warning")
            save_video(frames, output_video, fps=fps)
        
        # 验证文件
        if os.path.exists(output_video):
            file_size = os.path.getsize(output_video) / (1024**2)
            log(f"✓ 视频保存成功: {output_video} ({file_size:.2f} MB)", "finish")
        else:
            raise RuntimeError(f"视频文件未创建: {output_video}")
        
    except Exception as e:
        log(f"错误: {e}", "error")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
