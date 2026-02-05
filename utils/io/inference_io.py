#!/usr/bin/env python3
"""
推理 I/O 与 HDR/SDR 编排：帧输入、HDR 检测与 Tone Mapping、输出保存、分布式合并保存。
供 scripts/infer_video.py 与 scripts/inference_runner.py 复用。
调用方可设置 inference_io.log = custom_log 以统一日志输出。
"""
import os
import re
import json
import time
import gc
import shutil
import tempfile
import subprocess
from typing import List, Tuple, Optional, Sequence

import numpy as np
import cv2
import torch

# -------------------- 日志（可被调用方覆盖） --------------------
def _default_log(message: str, message_type: str = 'normal', rank: int = 0):
    print(message, flush=True)

log = _default_log


# -------------------- HDR Tone Mapping（可选） --------------------
try:
    from utils.hdr.tone_mapping import (
        detect_hdr_range,
        apply_tone_mapping_to_frames,
        apply_inverse_tone_mapping_to_frames,
    )
    HDR_TONE_MAPPING_AVAILABLE = True
except ImportError:
    HDR_TONE_MAPPING_AVAILABLE = False


# ==============================================================
#                    输入与分段
# ==============================================================

def natural_sort_key(filename: str):
    """自然排序键，正确处理文件名中的数字部分。"""
    parts = re.split(r'(\d+)', filename)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def get_total_frame_count(input_path: str, enable_hdr: bool = False) -> int:
    """获取输入的总帧数（不加载所有帧到内存）。"""
    if os.path.isfile(input_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count
    elif os.path.isdir(input_path):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        if enable_hdr:
            image_extensions.add('.dpx')
        image_files = []
        for filename in os.listdir(input_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(filename)
        image_files.sort(key=lambda x: natural_sort_key(x))
        return len(image_files)
    else:
        raise RuntimeError(f"Input path does not exist: {input_path}")


def get_input_image_sequence_filenames(input_path: str, enable_hdr: bool = False) -> Optional[List[str]]:
    """当输入为图片序列目录时，返回按自然序排序的文件名列表；否则返回 None。"""
    if not os.path.isdir(input_path):
        return None
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    if enable_hdr:
        image_extensions.add('.dpx')
    image_files = []
    for filename in os.listdir(input_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(filename)
    if not image_files:
        return None
    image_files.sort(key=lambda x: natural_sort_key(x))
    return image_files


# ---------- 流式输入（按 tile，不加载整段 (N,H,W,C)）----------

def get_segment_shape(
    input_path: str, start_idx: int, end_idx: int, enable_hdr: bool = False
) -> Tuple[int, int, int, int]:
    """获取 segment 的 (N, H, W, C) 而不加载全部帧。用于流式输入时计算 tile 坐标。"""
    N = end_idx - start_idx
    if N <= 0:
        raise RuntimeError(f"Invalid segment range: start_idx={start_idx}, end_idx={end_idx}")
    if os.path.isfile(input_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError(f"Could not read frame {start_idx} from {input_path}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[0], frame_rgb.shape[1]
        return N, h, w, 3
    elif os.path.isdir(input_path):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        if enable_hdr:
            image_extensions.add('.dpx')
        image_files = []
        for filename in os.listdir(input_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(filename)
        image_files.sort(key=lambda x: natural_sort_key(x))
        if start_idx >= len(image_files):
            raise RuntimeError(f"Start index {start_idx} exceeds total frames {len(image_files)}")
        N = min(N, len(image_files) - start_idx)
        first_path = os.path.join(input_path, image_files[start_idx])
        if first_path.lower().endswith('.dpx') and enable_hdr:
            try:
                from utils.io.hdr_io import read_dpx_frame
                frame_np = read_dpx_frame(first_path)
                h, w = frame_np.shape[0], frame_np.shape[1]
                return N, h, w, 3
            except Exception:
                pass
        img = cv2.imread(first_path)
        if img is None:
            raise RuntimeError(f"Could not read image {first_path}")
        h, w = img.shape[0], img.shape[1]
        return N, h, w, 3
    else:
        raise RuntimeError(f"Input path does not exist: {input_path}")


def _read_single_frame(
    input_path: str, frame_idx: int, enable_hdr: bool
) -> torch.Tensor:
    """读取单帧 (H,W,C) float。"""
    if os.path.isfile(input_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError(f"Could not read frame {frame_idx} from {input_path}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(frame_rgb).float() / 255.0
    elif os.path.isdir(input_path):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        if enable_hdr:
            image_extensions.add('.dpx')
        image_files = []
        for filename in os.listdir(input_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(filename)
        image_files.sort(key=lambda x: natural_sort_key(x))
        if frame_idx >= len(image_files):
            raise RuntimeError(f"Frame index {frame_idx} out of range")
        path = os.path.join(input_path, image_files[frame_idx])
        if path.lower().endswith('.dpx') and enable_hdr:
            try:
                from utils.io.hdr_io import read_dpx_frame
                frame_np = read_dpx_frame(path)
                return torch.from_numpy(frame_np).float()
            except Exception:
                pass
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Could not read {path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img_rgb).float() / 255.0
    else:
        raise RuntimeError(f"Input path does not exist: {input_path}")


def _apply_tone_mapping_single_frame(
    frame: torch.Tensor, args, global_l_max: float
) -> torch.Tensor:
    """对单帧应用 tone mapping（用于流式 HDR）。"""
    if not HDR_TONE_MAPPING_AVAILABLE:
        return frame
    from utils.hdr.tone_mapping import apply_tone_mapping_to_frames
    batch = frame.unsqueeze(0)
    out, _ = apply_tone_mapping_to_frames(
        batch,
        method=args.tone_mapping_method,
        exposure=args.tone_mapping_exposure,
        per_frame=False,
        global_l_max=global_l_max,
    )
    return out[0]


def get_hdr_tone_mapping_params_lightweight(
    input_path: str, start_idx: int, end_idx: int, args, rank: int = 0,
    sample_step: int = 10, checkpoint_dir: Optional[str] = None
) -> Optional[dict]:
    """流式 HDR：采样少量帧估计 global_l_max，避免整段加载。"""
    if not getattr(args, "_enable_hdr", False) or not HDR_TONE_MAPPING_AVAILABLE:
        return None
    if getattr(args, "global_l_max", None) is not None:
        g = args.global_l_max
        return {
            "global_l_max": g,
            "method": getattr(args, "tone_mapping_method", "reinhard"),
            "exposure": getattr(args, "tone_mapping_exposure", 1.0),
            "max_hdr": g,
            "white_point": g,
        }
    N = end_idx - start_idx
    n_sample = min(20, max(1, N // sample_step))
    step = max(1, N // n_sample) if n_sample else 1
    indices = [start_idx + i * step for i in range(n_sample) if start_idx + i * step < end_idx]
    if not indices:
        indices = [start_idx]
    frames_list = []
    for i in indices:
        frames_list.append(_read_single_frame(input_path, i, enable_hdr=True))
    stack = torch.stack(frames_list, dim=0)
    if not detect_hdr_range(stack):
        return None
    global_l_max = stack.max().item() * getattr(args, "tone_mapping_exposure", 1.0)
    log(f"[Rank {rank}] [HDR] 流式: 采样 {len(indices)} 帧估计 global_l_max={global_l_max:.4f}", "info", rank)
    method = getattr(args, "tone_mapping_method", "reinhard")
    exposure = getattr(args, "tone_mapping_exposure", 1.0)
    params = {
        "global_l_max": global_l_max,
        "method": method,
        "exposure": exposure,
        "max_hdr": global_l_max,
        "white_point": global_l_max,
    }
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        params_file = os.path.join(checkpoint_dir, f"rank_{rank}_streaming_hdr_params.json")
        with open(params_file, "w") as fp:
            json.dump(params, fp, indent=2)
    return params


def read_input_frames_range_tile_crops(
    input_path: str,
    start_idx: int,
    end_idx: int,
    tile_coords: List[Tuple[int, int, int, int]],
    fps: float = 30.0,
    enable_hdr: bool = False,
    streaming_hdr_params: Optional[dict] = None,
    args=None,
) -> Tuple[List[torch.Tensor], float]:
    """流式按 tile 读取：逐帧读入并裁成多个 tile 的 (N,tile_h,tile_w,C)，不保留整幅 (N,H,W,C)。"""
    N = end_idx - start_idx
    if N <= 0:
        raise RuntimeError(f"Invalid segment range: {start_idx}-{end_idx}")
    global_l_max = None
    if streaming_hdr_params and isinstance(streaming_hdr_params, dict):
        global_l_max = streaming_hdr_params.get("global_l_max")
    if args is None:
        class _DummyArgs:
            tone_mapping_method = "reinhard"
            tone_mapping_exposure = 1.0
        args = _DummyArgs()

    tile_tensors = []
    for (x1, y1, x2, y2) in tile_coords:
        th, tw = y2 - y1, x2 - x1
        tile_tensors.append(torch.empty((N, th, tw, 3), dtype=torch.float32))

    for local_i in range(N):
        frame_idx = start_idx + local_i
        frame = _read_single_frame(input_path, frame_idx, enable_hdr=enable_hdr)
        if enable_hdr and global_l_max is not None and frame.max().item() > 1.05:
            frame = _apply_tone_mapping_single_frame(frame, args, global_l_max)
        for t_idx, (x1, y1, x2, y2) in enumerate(tile_coords):
            tile_tensors[t_idx][local_i] = frame[y1:y2, x1:x2, :].clone()
        del frame
    return tile_tensors, fps


def read_input_frames_range(
    input_path: str, start_idx: int, end_idx: int, fps: float = 30.0, enable_hdr: bool = False
) -> Tuple[torch.Tensor, float]:
    """读取指定范围的帧。HDR 时走 hdr_io，否则走 OpenCV。"""
    if enable_hdr:
        try:
            from utils.io.hdr_io import read_dpx_frame, read_hdr_video_frame_range

            if os.path.isfile(input_path):
                log(f"[read_input_frames_range] [HDR] 检测到视频文件，使用 HLG 转换读取...", "info", 0)
                frames_np, fps = read_hdr_video_frame_range(input_path, start_idx, end_idx, convert_to_hlg=True)
                frames_tensor = torch.from_numpy(frames_np).float()
                log(f"[read_input_frames_range] [HDR] 读取完成，范围: [{frames_tensor.min():.4f}, {frames_tensor.max():.4f}]", "info", 0)
                return frames_tensor, fps

            elif os.path.isdir(input_path):
                files = os.listdir(input_path)
                dpx_files = [f for f in files if f.lower().endswith('.dpx')]
                if len(dpx_files) > 0:
                    log(f"[read_input_frames_range] [HDR] 检测到 DPX 图片序列，使用 HDR 读取方式...", "info", 0)
                    dpx_files.sort(key=lambda x: natural_sort_key(x))
                    if end_idx > len(dpx_files):
                        end_idx = len(dpx_files)
                    if start_idx >= len(dpx_files):
                        raise RuntimeError(f"Start index {start_idx} exceeds total frames {len(dpx_files)}")
                    frames = []
                    for idx in range(start_idx, end_idx):
                        dpx_path = os.path.join(input_path, dpx_files[idx])
                        try:
                            frame_np = read_dpx_frame(dpx_path)
                            frames.append(torch.from_numpy(frame_np).float())
                        except Exception as e:
                            log(f"[read_input_frames_range] [HDR] Warning: Error reading {dpx_path}: {e}, skipping", "warning", 0)
                    if not frames:
                        raise RuntimeError(f"No valid DPX frames read from range {start_idx}-{end_idx}")
                    video_tensor = torch.stack(frames, dim=0)
                    log(f"[read_input_frames_range] [HDR] 读取完成，范围: [{video_tensor.min():.4f}, {video_tensor.max():.4f}]", "info", 0)
                    return video_tensor, fps
                else:
                    log(f"[read_input_frames_range] [HDR] 未检测到 DPX 文件，使用普通读取方式", "warning", 0)
        except ImportError:
            log(f"[read_input_frames_range] [HDR] Warning: read_hdr_input 模块不可用，回退到普通读取方式", "warning", 0)
        except Exception as e:
            log(f"[read_input_frames_range] [HDR] Warning: HDR 读取失败: {e}，回退到普通读取方式", "warning", 0)

    # 非 HDR 或回退
    if os.path.isfile(input_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 1000:
            fps = 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        frames = []
        for i in range(end_idx - start_idx):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame_rgb).float() / 255.0)
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames read from video range {start_idx}-{end_idx}")
        return torch.stack(frames, dim=0), fps

    elif os.path.isdir(input_path):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_files = []
        for filename in os.listdir(input_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(filename)
        image_files.sort(key=lambda x: natural_sort_key(x))
        if end_idx > len(image_files):
            end_idx = len(image_files)
        if start_idx >= len(image_files):
            raise RuntimeError(f"Start index {start_idx} exceeds total frames {len(image_files)}")
        frames = []
        for idx in range(start_idx, end_idx):
            img_path = os.path.join(input_path, image_files[idx])
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(torch.from_numpy(img_rgb).float() / 255.0)
            except Exception as e:
                log(f"[read_input_frames_range] Warning: Error reading {img_path}: {e}, skipping", "warning", 0)
        if not frames:
            raise RuntimeError(f"No valid frames read from range {start_idx}-{end_idx}")
        return torch.stack(frames, dim=0), fps
    else:
        raise RuntimeError(f"Input path does not exist: {input_path}")


def split_video_by_frames(
    total_frames: int, num_workers: int, overlap: int = 2, force_num_workers: bool = False
) -> List[Tuple[int, int]]:
    """根据总帧数分割视频段，满足 FlashVSR 最小 21 帧/段。"""
    N = total_frames
    min_frames_per_segment = 21

    if not force_num_workers:
        if N < min_frames_per_segment * num_workers:
            new_num_workers = max(1, N // min_frames_per_segment)
            log(f"[Split] Video has only {N} frames, reducing workers from {num_workers} to {new_num_workers}", "warning", 0)
            num_workers = new_num_workers
    else:
        if N < min_frames_per_segment * num_workers:
            log(f"[Split] WARNING: Video has only {N} frames, but forcing {num_workers} workers.", "warning", 0)

    base_segment_size = N // num_workers if num_workers > 0 else N
    remainder = N % num_workers
    segments = []
    current_start = 0

    for i in range(num_workers):
        segment_base_size = base_segment_size + (1 if i < remainder else 0)
        if i == 0:
            start_idx = 0
            end_idx = min(N, segment_base_size + overlap)
        elif i == num_workers - 1:
            start_idx = max(0, current_start - overlap)
            end_idx = N
        else:
            start_idx = max(0, current_start - overlap)
            end_idx = min(N, current_start + segment_base_size + overlap)

        actual_frames = end_idx - start_idx
        if actual_frames < min_frames_per_segment:
            if i < num_workers - 1:
                end_idx = min(N, start_idx + min_frames_per_segment)
            else:
                start_idx = max(0, end_idx - min_frames_per_segment)

        if end_idx <= start_idx and N > 0:
            if i < N:
                start_idx = i
                end_idx = min(i + 1, N)
            else:
                start_idx = N
                end_idx = N

        segments.append((start_idx, end_idx))
        current_start = end_idx - overlap

    total_after_overlap = 0
    for i, (start, end) in enumerate(segments):
        frames_read = end - start
        if i > 0:
            frames_read -= overlap
        if i < len(segments) - 1:
            frames_read -= overlap
        total_after_overlap += max(0, frames_read)

    if total_after_overlap != N:
        log(f"[Split] WARNING: After removing overlaps, total frames ({total_after_overlap}) != input frames ({N}).", "warning", 0)

    return segments


def get_default_output_path(input_path: str, scale: int = 4) -> str:
    """根据输入路径生成默认输出路径。"""
    if os.path.isfile(input_path):
        base_name = os.path.splitext(input_path)[0]
        return f"{base_name}_{scale}x.mp4"
    elif os.path.isdir(input_path):
        dir_name = os.path.basename(input_path.rstrip('/'))
        parent_dir = os.path.dirname(input_path) if os.path.dirname(input_path) else "."
        return os.path.join(parent_dir, f"{dir_name}_{scale}x.mp4")
    else:
        return "output.mp4"


def get_checkpoint_dir(args) -> str:
    """返回本次推理的 checkpoint 目录。"""
    output_path = args.output if args.output else get_default_output_path(args.input, args.scale)
    output_path = output_path.rstrip(os.sep)
    output_mode = getattr(args, 'output_mode', 'video')
    if output_mode == 'pictures':
        name = os.path.basename(output_path) or "output"
    else:
        name = os.path.splitext(os.path.basename(output_path))[0] or "output"
    return os.path.join("/app/tmp/checkpoints", name)


# ==============================================================
#                HDR Tone Mapping 编排
# ==============================================================

def serialize_tone_mapping_params(params_list: list) -> list:
    """序列化 tone mapping 参数。"""
    serialized = []
    for params in params_list:
        serialized_params = {}
        for key, value in params.items():
            if isinstance(value, torch.Tensor):
                serialized_params[key] = value.item() if value.numel() == 1 else value.cpu().numpy().tolist()
            elif isinstance(value, np.ndarray):
                serialized_params[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serialized_params[key] = float(value) if isinstance(value, np.floating) else int(value)
            else:
                serialized_params[key] = value
        serialized.append(serialized_params)
    return serialized


def deserialize_tone_mapping_params(serialized_list: list) -> list:
    """反序列化 tone mapping 参数。"""
    return serialized_list


def apply_hdr_tone_mapping_if_needed(
    frames: torch.Tensor, args, rank: int = 0, checkpoint_dir: str = None
):
    """检测 HDR 并应用 Tone Mapping。"""
    if not getattr(args, '_enable_hdr', False) or not HDR_TONE_MAPPING_AVAILABLE:
        return frames, None

    use_hlg_workflow = getattr(args, 'hdr_preprocess', 'hlg') == 'hlg'

    if use_hlg_workflow:
        if frames.max() <= 1.05:
            log(f"[Rank {rank}] [HDR] ✓ HLG 工作流：数据已在 [0, 1] 范围，直接送入 AI 模型", "info", rank)
            hlg_params = {'workflow': 'hlg', 'input_range': [frames.min().item(), frames.max().item()]}
            return frames, hlg_params
        else:
            log(f"[Rank {rank}] [HDR] ⚠ HLG 工作流但数据超出 [0, 1]，回退到传统 Tone Mapping", "warning", rank)

    if not detect_hdr_range(frames):
        log(f"[Rank {rank}] [HDR] 未检测到 HDR 值，跳过 Tone Mapping", "info", rank)
        return frames, None

    log(f"[Rank {rank}] [HDR] 检测到 HDR 输入，应用 Tone Mapping ({args.tone_mapping_method})...", "info", rank)
    global_l_max = getattr(args, 'global_l_max', None)
    if global_l_max is not None:
        log(f"[Rank {rank}] [HDR] ✓ 使用全局 l_max: {global_l_max:.4f}", "info", rank)
    else:
        global_l_max = frames.max().item() * args.tone_mapping_exposure
        log(f"[Rank {rank}] [HDR] ⚠ 使用当前 segment 的 l_max: {global_l_max:.4f}", "warning", rank)

    processed_frames, tone_mapping_params = apply_tone_mapping_to_frames(
        frames,
        method=args.tone_mapping_method,
        exposure=args.tone_mapping_exposure,
        per_frame=False,
        global_l_max=global_l_max,
    )
    log(f"[Rank {rank}] [HDR] Tone Mapping 完成，SDR 范围: [{processed_frames.min():.4f}, {processed_frames.max():.4f}]", "info", rank)

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        params_file = os.path.join(checkpoint_dir, f"rank_{rank}_tone_mapping_params.json")
        with open(params_file, 'w') as f:
            json.dump(serialize_tone_mapping_params(tone_mapping_params), f, indent=2)
        log(f"[Rank {rank}] [HDR] Tone Mapping 参数已保存: {params_file}", "info", rank)

    return processed_frames, tone_mapping_params


def apply_inverse_hdr_if_needed(output: torch.Tensor, tone_mapping_params, args, rank: int = 0):
    """应用 Inverse Tone Mapping 还原 HDR。"""
    if not getattr(args, '_enable_hdr', False) or tone_mapping_params is None:
        return output

    if isinstance(tone_mapping_params, dict) and tone_mapping_params.get('workflow') == 'hlg':
        log(f"[Rank {rank}] [HDR] HLG 工作流：保持 HLG 编码，不做逆 Tone Mapping", "info", rank)
        output = torch.clamp(output, 0.0, 1.0)
        return output

    if not HDR_TONE_MAPPING_AVAILABLE:
        return output

    log(f"[Rank {rank}] [HDR] 应用 Inverse Tone Mapping 还原 HDR...", "info", rank)
    if output.min() < 0:
        output = (output + 1.0) / 2.0
    # 流式时 tone_mapping_params 为单段 dict，需展开为每帧一份
    if isinstance(tone_mapping_params, dict):
        params_list = [tone_mapping_params] * output.shape[0]
    else:
        params_list = tone_mapping_params
    output = apply_inverse_tone_mapping_to_frames(output, params_list)
    log(f"[Rank {rank}] [HDR] HDR 还原完成，范围: [{output.min():.4f}, {output.max():.4f}]", "info", rank)
    return output


def precompute_global_hdr_params(
    input_path: str, enable_hdr: bool, exposure: float = 1.0,
    sample_interval: int = 10, max_sample_frames: int = 500,
) -> Optional[float]:
    """预扫描视频计算全局 HDR l_max。"""
    if not enable_hdr:
        return None

    log(f"[Main] [HDR] 预扫描视频以计算全局 HDR 参数...", "info")

    try:
        total_frames = get_total_frame_count(input_path, enable_hdr=True)
        if total_frames <= max_sample_frames:
            sample_indices = list(range(total_frames))
        else:
            actual_interval = max(1, total_frames // max_sample_frames)
            sample_indices = list(range(0, total_frames, actual_interval))[:max_sample_frames]

        log(f"[Main] [HDR] 总帧数: {total_frames}, 采样帧数: {len(sample_indices)}", "info")
        global_max = 0.0
        hdr_detected = False
        batch_size = 50
        for batch_start in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[batch_start : batch_start + batch_size]
            for idx in batch_indices:
                try:
                    frames, _ = read_input_frames_range(input_path, idx, idx + 1, fps=30.0, enable_hdr=True)
                    frame_max = frames.max().item()
                    if frame_max > 1.0:
                        hdr_detected = True
                    if frame_max > global_max:
                        global_max = frame_max
                except Exception:
                    continue
            progress = min(100, (batch_start + len(batch_indices)) * 100 // len(sample_indices))
            if progress % 25 == 0 or batch_start + len(batch_indices) >= len(sample_indices):
                log(f"[Main] [HDR] 预扫描进度: {progress}%, 当前最大值: {global_max:.4f}", "info")

        if not hdr_detected:
            log(f"[Main] [HDR] 未检测到 HDR 值，将跳过 HDR 处理", "info")
            return None

        global_l_max = global_max * exposure
        log(f"[Main] [HDR] ✓ 预扫描完成！全局 l_max: {global_l_max:.4f}", "info")
        return global_l_max

    except Exception as e:
        log(f"[Main] [HDR] 预扫描失败: {e}，将使用每个 rank 独立计算的参数", "warning")
        return None


def get_global_hdr_max(
    output: torch.Tensor, args, checkpoint_dir: str = None, rank_list: List[int] = None
) -> Optional[float]:
    """获取全局 HDR 最大值（用于 DPX 输出）。"""
    is_hdr = getattr(args, '_enable_hdr', False) and (output > 1.0).any().item()
    if not is_hdr:
        return None

    global_hdr_max = output.max().item()
    if checkpoint_dir and rank_list:
        for r in rank_list:
            params_file = os.path.join(checkpoint_dir, f"rank_{r}_tone_mapping_params.json")
            if os.path.exists(params_file):
                try:
                    with open(params_file, 'r') as f:
                        params_list = json.load(f)
                    if params_list:
                        max_hdr_values = [p.get('max_hdr', 1.0) for p in params_list]
                        rank_max = max(max_hdr_values) if max_hdr_values else 1.0
                        global_hdr_max = max(global_hdr_max, rank_max)
                except Exception:
                    pass
    return global_hdr_max


# ==============================================================
#                    输出保存
# ==============================================================

def _output_filename_for_frame(
    logical_idx: int,
    output_path: str,
    output_format: str,
    input_filenames: Optional[Sequence[str]] = None,
) -> str:
    """根据逻辑帧索引生成输出文件名；若提供 input_filenames 则保持与输入名称一致（仅扩展名按输出格式）。"""
    if input_filenames is not None and 0 <= logical_idx < len(input_filenames):
        base, _ = os.path.splitext(input_filenames[logical_idx])
        ext = ".dpx" if output_format in ("dpx", "dpx10") else ".png"
        return os.path.join(output_path, base + ext)
    ext = ".dpx" if output_format in ("dpx", "dpx10") else ".png"
    return os.path.join(output_path, f"frame_{logical_idx:06d}" + ext)


def save_frames_as_sequence(
    output: torch.Tensor, output_path: str, args,
    rank: int = 0, start_frame_idx: int = 0,
    global_hdr_max: float = None,
    input_filenames: Optional[Sequence[str]] = None,
) -> int:
    """将帧保存为图片序列（PNG 或 DPX10）。若提供 input_filenames（图片序列输入），输出文件名与输入一致（仅扩展名按 output_format）。"""
    os.makedirs(output_path, exist_ok=True)
    output_format = getattr(args, 'output_format', 'png')
    is_hdr = getattr(args, '_enable_hdr', False) and (output > 1.0).any().item()
    use_hlg_workflow = getattr(args, '_enable_hdr', False) and getattr(args, 'hdr_preprocess', 'hlg') == 'hlg'
    frames_saved = 0

    if output_format in ("dpx", "dpx10"):
        from utils.io.video_io import save_frame_as_dpx10
        if is_hdr and global_hdr_max:
            log(f"[Rank {rank}] [HDR] 使用全局 HDR 最大值: {global_hdr_max:.4f}", "info", rank)
        apply_srgb_gamma = not use_hlg_workflow
        for frame_idx in range(output.shape[0]):
            frame = output[frame_idx].cpu().numpy()
            logical_idx = start_frame_idx + frame_idx
            frame_filename = _output_filename_for_frame(logical_idx, output_path, output_format, input_filenames)
            save_frame_as_dpx10(frame, frame_filename, hdr_max=global_hdr_max, apply_srgb_gamma=apply_srgb_gamma)
            frames_saved += 1
    else:
        if is_hdr:
            log(f"[Rank {rank}] [HDR] 输出包含 HDR 值，PNG 输出会 clip 到 [0,1]", "warning", rank)
        output_np = (output.clamp(0, 1) * 255).byte().cpu().numpy().astype('uint8')
        for frame_idx in range(output_np.shape[0]):
            logical_idx = start_frame_idx + frame_idx
            frame_filename = _output_filename_for_frame(logical_idx, output_path, output_format, input_filenames)
            cv2.imwrite(frame_filename, cv2.cvtColor(output_np[frame_idx], cv2.COLOR_RGB2BGR))
        frames_saved = output_np.shape[0]

    return frames_saved


def save_merged_as_hdr_video(merged: torch.Tensor, output_path: str, args, fps: float) -> None:
    """HDR + output_mode=video：先写临时 DPX，再编码为 HDR 视频。"""
    from utils.io.video_io import save_frame_as_dpx10

    temp_dir = tempfile.mkdtemp(prefix="flashvsr_hdr_")
    try:
        global_hdr_max = merged.max().item() if (merged > 1.0).any().item() else None
        use_hlg_workflow = getattr(args, "hdr_preprocess", "hlg") == "hlg"
        apply_srgb_gamma = not use_hlg_workflow
        hdr_transfer = getattr(args, "hdr_transfer", "hdr10")
        crf = 18
        preset = "slow"
        max_hdr_nits = 1000.0

        log(f"[HDR Video] 写入临时 DPX 到 {temp_dir}（HLG: {use_hlg_workflow}）", "info")
        for i in range(merged.shape[0]):
            frame = merged[i].cpu().numpy()
            path = os.path.join(temp_dir, f"frame_{i:06d}.dpx")
            save_frame_as_dpx10(frame, path, hdr_max=global_hdr_max, apply_srgb_gamma=apply_srgb_gamma)

        if use_hlg_workflow:
            if hdr_transfer == "hdr10":
                from utils.io.hdr_video_encode import encode_hlg_dpx_to_hdr_video
                ok = encode_hlg_dpx_to_hdr_video(temp_dir, output_path, fps=fps, crf=crf, preset=preset, max_hdr_nits=max_hdr_nits)
            else:
                from utils.io.hdr_video_encode import encode_hlg_dpx_to_hlg_video
                ok = encode_hlg_dpx_to_hlg_video(temp_dir, output_path, fps=fps, crf=crf, preset=preset)
        else:
            from utils.io.hdr_video_encode import encode_dpx_to_hdr_video_simple
            ok = encode_dpx_to_hdr_video_simple(
                temp_dir, output_path, fps=fps, hdr_format=hdr_transfer, crf=crf, preset=preset,
                max_hdr_nits=max_hdr_nits, dpx_is_linear=False,
            )
        if not ok:
            raise RuntimeError(f"HDR 视频编码失败: {output_path}")
        log(f"[HDR Video] ✓ 已编码 HDR 视频: {output_path} (hdr_transfer={hdr_transfer})", "info")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ==============================================================
#                分布式合并保存（委托流式合并模块）
# ==============================================================

def merge_and_save_distributed_results(checkpoint_dir, args, world_size, total_frames, input_fps):
    """Rank 0：等待所有 rank 完成，再调用流式合并并保存（与恢复工具共用同一实现）。"""
    from utils.io import streaming_merge_output

    rank = 0
    output_path = args.output if args.output else get_default_output_path(args.input, args.scale)
    log(f"[Rank 0] ========== Rank 0: Merging Results ==========", "info", rank)
    log(f"[Rank 0] Waiting for all {world_size} ranks to complete...", "info", rank)
    max_wait_time = 3600
    wait_start = time.time()
    all_done = False
    while not all_done and (time.time() - wait_start) < max_wait_time:
        all_done = True
        for r in range(world_size):
            done_flag = os.path.join(checkpoint_dir, f"rank_{r}_done.flag")
            if not os.path.exists(done_flag):
                all_done = False
                break
        if not all_done:
            time.sleep(5)

    log(f"[Rank 0] All ranks finished. Starting streaming merge...", "info", rank)
    result_files = []
    for r in range(world_size):
        result_file = os.path.join(checkpoint_dir, f"rank_{r}_result.pt")
        done_file = os.path.join(checkpoint_dir, f"rank_{r}_done.flag")
        if os.path.exists(done_file):
            with open(done_file, 'r') as f:
                status = f.read().strip()
            if 'skipped' in status or 'failed' in status:
                continue
        if os.path.exists(result_file):
            result_files.append((r, result_file, os.path.getsize(result_file) / (1024 ** 2)))

    if not result_files:
        raise RuntimeError("[Rank 0] No valid results to merge!")

    result_files.sort(key=lambda x: x[0])
    output_mode = getattr(args, 'output_mode', 'video')
    output_format = getattr(args, 'output_format', 'png')
    input_filenames = None
    if output_mode == "pictures":
        input_filenames = get_input_image_sequence_filenames(
            args.input, enable_hdr=getattr(args, '_enable_hdr', False)
        )

    def _log_fn(msg: str, msg_type: str) -> None:
        log(f"[Rank 0] {msg}", msg_type, rank)

    streaming_merge_output.streaming_merge_from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        output_path=output_path,
        input_fps=input_fps,
        world_size=world_size,
        total_frames=total_frames,
        output_mode=output_mode,
        output_format=output_format,
        output_frame_prefix=getattr(args, 'output_frame_prefix', None),
        output_frame_digits=getattr(args, 'output_frame_digits', 6),
        output_workers=getattr(args, 'output_workers', 0),
        result_files=result_files,
        input_filenames=input_filenames,
        enable_hdr=getattr(args, '_enable_hdr', False),
        global_l_max=getattr(args, 'global_l_max', None),
        log_fn=_log_fn,
    )
    log(f"[Rank 0] Checkpoint directory preserved: {checkpoint_dir}", "info", rank)
