#!/usr/bin/env python3
"""
流式合并 + 流式输出：从 checkpoint 目录按 rank 逐个加载结果并写入视频/序列帧。

供 inference_io.merge_and_save_distributed_results 与 tools.recover_distributed_inference 共用，
保证正常跑与从 checkpoint 恢复使用同一套合并逻辑，输出一致。

内存策略：每次仅加载一个 rank 的 .pt（map_location='cpu'），转换后立即写盘并释放，
峰值约单段 + 转换 buffer，适合大分辨率与有限内存。
"""

import os
import re
import gc
import json
import tempfile
import subprocess
from typing import List, Tuple, Optional, Sequence, Callable

import numpy as np
import cv2
import torch


# -------------------- 日志（可被调用方覆盖） --------------------
def _default_log(message: str, message_type: str = "normal", rank: int = 0):
    """默认日志：仅 rank 0 时打印；调用方可替换为 inference_io.log 等。"""
    if rank != 0:
        return
    colors = {
        "normal": "\033[0m",
        "info": "\033[94m",
        "success": "\033[92m",
        "warning": "\033[93m",
        "error": "\033[91m",
        "finish": "\033[92m",
    }
    c = colors.get(message_type, colors["normal"])
    print(f"{c}{message}\033[0m", flush=True)


# 可被 inference_io 等覆盖，签名为 (message, message_type, rank=0)
log = _default_log


def detect_result_files(
    checkpoint_dir: str,
    world_size: Optional[int] = None,
) -> List[Tuple[int, str, float]]:
    """从 checkpoint 目录收集 rank_*_result.pt。

    Returns:
        List of (rank, result_file_path, size_mb), 按 rank 排序。
    若指定 world_size 则只扫描 rank 0..world_size-1；否则扫描目录下所有 rank_*_result.pt。
    """
    result_files = []
    if world_size is not None:
        for rank in range(world_size):
            result_file = os.path.join(checkpoint_dir, f"rank_{rank}_result.pt")
            if os.path.exists(result_file):
                file_size_mb = os.path.getsize(result_file) / (1024 ** 2)
                result_files.append((rank, result_file, file_size_mb))
    else:
        pattern = re.compile(r"rank_(\d+)_result\.pt$")
        for name in os.listdir(checkpoint_dir):
            m = pattern.match(name)
            if m:
                rank = int(m.group(1))
                result_file = os.path.join(checkpoint_dir, name)
                try:
                    file_size_mb = os.path.getsize(result_file) / (1024 ** 2)
                    result_files.append((rank, result_file, file_size_mb))
                except Exception as e:
                    log(f"  Rank {rank}: ✗ Error: {e}", "error")
        result_files.sort(key=lambda x: x[0])
    return result_files


def _frame_filename(
    output_path: str,
    frame_idx: int,
    output_format: str,
    output_frame_prefix: Optional[str] = None,
    output_frame_digits: int = 6,
    input_filenames: Optional[Sequence[str]] = None,
) -> str:
    """生成单帧输出路径。若提供 input_filenames 且 frame_idx 有效则与输入名一致（仅扩展名按格式）。"""
    ext = ".dpx" if output_format in ("dpx", "dpx10") else ".png"
    if input_filenames is not None and 0 <= frame_idx < len(input_filenames):
        base, _ = os.path.splitext(input_filenames[frame_idx])
        return os.path.join(output_path, base + ext)
    if output_frame_prefix:
        return os.path.join(output_path, f"{output_frame_prefix}.{frame_idx:0{output_frame_digits}d}{ext}")
    return os.path.join(output_path, f"frame_{frame_idx:06d}{ext}")


def streaming_merge_from_checkpoint(
    checkpoint_dir: str,
    output_path: str,
    input_fps: float = 30.0,
    world_size: Optional[int] = None,
    total_frames: Optional[int] = None,
    output_mode: str = "video",
    output_format: str = "png",
    output_frame_prefix: Optional[str] = None,
    output_frame_digits: int = 6,
    output_workers: int = 0,
    result_files: Optional[List[Tuple[int, str, float]]] = None,
    input_filenames: Optional[Sequence[str]] = None,
    enable_hdr: bool = False,
    global_l_max: Optional[float] = None,
    log_fn: Optional[Callable[[str, str], None]] = None,
) -> None:
    """流式合并 checkpoint 中的 rank 结果并写入视频或序列帧。

    使用方式：
    - 正常跑：inference_io.merge_and_save_distributed_results 在等待所有 rank 后，
      构建 result_files（仅 completed 的 rank），再调用本函数并传入 result_files。
    - 恢复工具：直接调用本函数，result_files=None 时自动 detect_result_files。

    参数
    -----
    checkpoint_dir : str
        存放 rank_*_result.pt 与可选 rank_*_tone_mapping_params.json 的目录。
    output_path : str
        输出路径：视频时为 .mp4 等文件路径；序列帧时为目录路径。
    input_fps : float
        帧率，用于视频编码。
    world_size : int, optional
        rank 数量；result_files 为 None 时用于 detect_result_files。
    total_frames : int, optional
        期望总帧数，用于填充不足或校验超出。
    output_mode : str
        "video" 或 "pictures"。
    output_format : str
        序列帧时 "png" 或 "dpx10"；video 时忽略。
    output_frame_prefix : str, optional
        序列帧文件名前缀（如 H001_11261139_C001）。
    output_frame_digits : int
        序列帧索引位数。
    output_workers : int
        序列帧多线程写帧数，0 表示自动。
    result_files : list, optional
        [(rank, result_file_path, size_mb), ...]。若为 None 则调用 detect_result_files。
    input_filenames : sequence, optional
        输入序列帧名列表，用于保持输出文件名与输入一致（仅扩展名按 output_format）。
    enable_hdr : bool
        为 True 且 output_mode=video 时走 HDR 视频路径（临时 DPX + encode_hlg_dpx_to_hdr_video）。
    global_l_max : float, optional
        HDR/DPX 用全局亮度上限。
    log_fn : callable, optional
        (message, message_type) -> None，用于替代本模块 log；便于与 inference_io 统一前缀（如 [Rank 0]）。
    """
    _log = log_fn if log_fn is not None else lambda msg, mt: log(msg, mt, 0)

    if result_files is None:
        result_files = detect_result_files(checkpoint_dir, world_size)
    else:
        result_files = list(result_files)
    if not result_files:
        _log("No valid results to merge!", "error")
        raise RuntimeError("No valid results to merge!")

    result_files.sort(key=lambda x: x[0])
    effective_world_size = world_size if world_size is not None else (max(r[0] for r in result_files) + 1)
    num_workers = output_workers if output_workers > 0 else min(os.cpu_count() or 8, 32)

    _log("========== Starting Streaming Merge ==========", "info")
    _log(f"Checkpoint directory: {checkpoint_dir}", "info")
    _log(f"Output path: {output_path}", "info")
    _log(f"Output mode: {output_mode}", "info")
    _log(f"FPS: {input_fps}", "info")
    _log(f"Total frames (expected): {total_frames if total_frames is not None else 'auto'}", "info")
    _log(f"Streaming merge (memory-efficient, CPU only)", "info")
    total_size_mb = sum(mb for _, _, mb in result_files)
    _log(f"Found {len(result_files)} rank results, ~{total_size_mb:.0f} MB total", "info")
    _log("=============================================", "info")

    first_rank, first_file, first_size_mb = result_files[0]
    first_result = torch.load(first_file, map_location="cpu")
    F, H, W, C = first_result.shape
    del first_result
    gc.collect()

    tmp_yuv_path = None
    try:
        if output_mode == "pictures":
            from utils.io.video_io import save_frame_as_dpx10
            os.makedirs(output_path, exist_ok=True)
            total_frames_written = 0
            last_frame = None
            global_hdr_max_all = None

            for rank_idx, (r, result_file, file_size_mb) in enumerate(result_files):
                _log(f"[{rank_idx + 1}/{len(result_files)}] Loading rank {r} (~{file_size_mb:.0f} MB)...", "info")
                segment = torch.load(result_file, map_location="cpu")
                n_frames = segment.shape[0]
                last_frame = segment[-1:, :, :, :]

                if output_format in ("dpx", "dpx10"):
                    is_hdr = (segment > 1.0).any().item()
                    global_hdr_max = global_l_max
                    if is_hdr:
                        params_file = os.path.join(checkpoint_dir, f"rank_{r}_tone_mapping_params.json")
                        if os.path.exists(params_file):
                            try:
                                with open(params_file, "r") as f:
                                    params_list = json.load(f)
                                    vals = [p.get("max_hdr") for p in params_list if p.get("max_hdr") is not None]
                                    if vals:
                                        global_hdr_max = max(global_hdr_max or 0, max(vals))
                            except Exception:
                                pass
                        if global_hdr_max is None:
                            global_hdr_max = segment.max().item()
                            for other in range(effective_world_size):
                                if other == r:
                                    continue
                                op = os.path.join(checkpoint_dir, f"rank_{other}_tone_mapping_params.json")
                                if os.path.exists(op):
                                    try:
                                        with open(op, "r") as f:
                                            for p in json.load(f):
                                                if "max_hdr" in p:
                                                    global_hdr_max = max(global_hdr_max, p["max_hdr"])
                                    except Exception:
                                        pass
                        if global_hdr_max_all is None or (global_hdr_max and global_hdr_max > (global_hdr_max_all or 0)):
                            global_hdr_max_all = global_hdr_max
                    apply_srgb_gamma = False
                    segment_np = segment.cpu().numpy()
                    for i in range(n_frames):
                        frame = segment_np[i]
                        if not is_hdr:
                            frame = np.clip(frame, 0, 1)
                        path = _frame_filename(
                            output_path, total_frames_written + i, output_format,
                            output_frame_prefix, output_frame_digits, input_filenames,
                        )
                        save_frame_as_dpx10(frame, path, hdr_max=global_hdr_max, apply_srgb_gamma=apply_srgb_gamma)
                    total_frames_written += n_frames
                    del segment_np
                    del segment
                    gc.collect()
                else:
                    seg_np = (segment.clamp(0, 1) * 255).byte().cpu().numpy().astype("uint8")
                    for i in range(n_frames):
                        path = _frame_filename(
                            output_path, total_frames_written + i, output_format,
                            output_frame_prefix, output_frame_digits, input_filenames,
                        )
                        cv2.imwrite(path, cv2.cvtColor(seg_np[i], cv2.COLOR_RGB2BGR))
                    total_frames_written += n_frames
                    del seg_np
                del segment
                gc.collect()
                _log(f"  Rank {r}: {n_frames} frames written (total: {total_frames_written})", "success")

            if total_frames is not None and total_frames_written < total_frames:
                missing = total_frames - total_frames_written
                _log(f"Padding {missing} frames with last frame", "info")
                if last_frame is not None:
                    if output_format in ("dpx", "dpx10"):
                        pad_np = last_frame.cpu().numpy()[0]
                        if global_hdr_max_all is None:
                            pad_np = np.clip(pad_np, 0, 1)
                        for i in range(missing):
                            path = _frame_filename(
                                output_path, total_frames_written, output_format,
                                output_frame_prefix, output_frame_digits, input_filenames,
                            )
                            save_frame_as_dpx10(pad_np, path, hdr_max=global_hdr_max_all, apply_srgb_gamma=False)
                            total_frames_written += 1
                    else:
                        pad_np = (last_frame.clamp(0, 1) * 255).byte().cpu().numpy().astype("uint8")[0]
                        for i in range(missing):
                            path = _frame_filename(
                                output_path, total_frames_written, output_format,
                                output_frame_prefix, output_frame_digits, input_filenames,
                            )
                            cv2.imwrite(path, cv2.cvtColor(pad_np, cv2.COLOR_RGB2BGR))
                            total_frames_written += 1
            elif total_frames is not None and total_frames_written > total_frames:
                _log(f"WARNING: {total_frames_written} frames > expected {total_frames}", "warning")

            _log(f"========== Merge Completed ==========", "finish")
            _log(f"Output: {output_path}", "finish")
            _log(f"Frames: {total_frames_written}, size {H}x{W}x{C}", "finish")

        else:
            # video
            if enable_hdr:
                from utils.io.video_io import save_frame_as_dpx10
                from utils.io.hdr_video_encode import encode_hlg_dpx_to_hdr_video
                tmp_dpx_dir = tempfile.mkdtemp(prefix="flashvsr_hdr_")
                try:
                    total_frames_written = 0
                    for rank_idx, (r, result_file, file_size_mb) in enumerate(result_files):
                        _log(f"[{rank_idx + 1}/{len(result_files)}] Rank {r}...", "info")
                        segment = torch.load(result_file, map_location="cpu")
                        for i in range(segment.shape[0]):
                            frame = segment[i].cpu().numpy()
                            path = os.path.join(tmp_dpx_dir, f"frame_{total_frames_written:06d}.dpx")
                            save_frame_as_dpx10(frame, path, hdr_max=global_l_max, apply_srgb_gamma=False)
                            total_frames_written += 1
                        del segment
                        gc.collect()
                    if total_frames_written > 0:
                        ok = encode_hlg_dpx_to_hdr_video(
                            tmp_dpx_dir, output_path, fps=input_fps, crf=18, preset="slow", max_hdr_nits=1000.0
                        )
                        if not ok:
                            raise RuntimeError("HDR video encode failed")
                    _log(f"HDR video saved: {output_path}", "finish")
                finally:
                    import shutil
                    shutil.rmtree(tmp_dpx_dir, ignore_errors=True)
            else:
                tmp_yuv = tempfile.NamedTemporaryFile(suffix=".yuv", delete=False)
                tmp_yuv_path = tmp_yuv.name
                total_frames_written = 0
                last_frame = None
                for rank_idx, (r, result_file, file_size_mb) in enumerate(result_files):
                    _log(f"[{rank_idx + 1}/{len(result_files)}] Rank {r}...", "info")
                    segment = torch.load(result_file, map_location="cpu")
                    seg_np = (segment.clamp(0, 1) * 255).byte().cpu().numpy().astype("uint8")
                    batch_size = 50
                    for i in range(0, seg_np.shape[0], batch_size):
                        end_i = min(i + batch_size, seg_np.shape[0])
                        tmp_yuv.write(seg_np[i:end_i].tobytes())
                        total_frames_written += end_i - i
                    last_frame = segment[-1:, :, :, :]
                    del segment, seg_np
                    gc.collect()

                if total_frames is not None and total_frames_written < total_frames and last_frame is not None:
                    pad_np = (last_frame.clamp(0, 1) * 255).byte().cpu().numpy()
                    for _ in range(total_frames - total_frames_written):
                        tmp_yuv.write(pad_np[0].tobytes())
                        total_frames_written += 1
                tmp_yuv.close()

                cmd = [
                    "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                    "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(input_fps),
                    "-i", tmp_yuv_path,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "18",
                    output_path,
                ]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                if result.returncode != 0 or not os.path.exists(output_path):
                    stderr = result.stderr.decode("utf-8", errors="ignore")[:500]
                    raise RuntimeError(f"FFmpeg failed: {stderr}")
                try:
                    os.unlink(tmp_yuv_path)
                except Exception:
                    pass
                tmp_yuv_path = None
                _log(f"Video saved: {output_path}", "finish")
    finally:
        if tmp_yuv_path and os.path.exists(tmp_yuv_path):
            try:
                os.unlink(tmp_yuv_path)
            except Exception:
                pass
