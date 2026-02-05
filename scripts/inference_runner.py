# inference_runner: 推理执行核心（单卡/多卡、整图/tile、pipeline 加载与运行）。
# 由 infer_video.py 加载并调用；I/O、HDR、合并与保存由 utils.io.inference_io 提供。

from __future__ import annotations

import os
import sys
import math
import time
import json
import gc
import traceback
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist

# 项目根目录
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.io import inference_io

from utils.inference_support import (
    tensor2video,
    prepare_input_tensor,
    calculate_tile_coords,
    create_feather_mask,
    estimate_tile_memory,
    get_gpu_memory_info,
    get_available_memory_gb,
)
from src.models.model_manager import ModelManager
from src.models.TCDecoder import build_tcdecoder
from src.models.utils import Buffer_LQ4x_Proj, clean_vram
from src.models import wan_video_dit
from src.pipelines.flashvsr_full import FlashVSRFullPipeline
from src.pipelines.flashvsr_tiny import FlashVSRTinyPipeline
from src.pipelines.flashvsr_tiny_long import FlashVSRTinyLongPipeline


# ---------- 小工具 ----------
def log(message: str, message_type: str = "normal", rank: int = 0):
    """Colored logging for console output (with flush for real-time output)."""
    if dist.is_initialized() and rank != 0:
        return
    if message_type == "error":
        message = "\033[1;41m" + message + "\033[m"
    elif message_type == "warning":
        message = "\033[1;31m" + message + "\033[m"
    elif message_type == "finish":
        message = "\033[1;32m" + message + "\033[m"
    elif message_type == "info":
        message = "\033[1;33m" + message + "\033[m"
    print(message, flush=True)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def ensure_device_str(device) -> str:
    """Ensure device is string format."""
    if isinstance(device, torch.device):
        return str(device)
    return device


def determine_optimal_batch_size(
    device: str,
    tile_coords: List[Tuple[int, int, int, int]],
    frames: torch.Tensor,
    args,
    rank: int = 0,
    segment_n_frames: int = None,
) -> int:
    """Determine optimal batch size based on available GPU memory.
    多 GPU 时实际推理为逐 tile(B=1)，tile_batch_size 仅作进度分组；单 GPU 时仍可 batched 加速。
    When frames is None (streaming), pass segment_n_frames.
    """
    if hasattr(args, "tile_batch_size") and args.tile_batch_size > 0:
        batch_size = args.tile_batch_size
        log(
            f"[Rank {rank}] Using user-specified tile_batch_size: {batch_size}",
            "info",
            rank,
        )
        return min(batch_size, len(tile_coords))

    if hasattr(args, "adaptive_tile_batch") and not args.adaptive_tile_batch:
        return 1

    if isinstance(device, torch.device):
        device = str(device)
    if not device.startswith("cuda:"):
        return 1

    try:
        available_gb = get_available_memory_gb(device)
        used_gb, total_gb = get_gpu_memory_info(device)
        N = segment_n_frames if segment_n_frames is not None else (frames.shape[0] if frames is not None else 1)

        tile_size = args.tile_size
        dtype_size = 2 if args.precision in ["fp16", "bf16"] else 4
        tile_memory = estimate_tile_memory(tile_size, N, args.scale, dtype_size)
        # batched 前向有额外激活/attention 峰值，按 1.5x 保守估算每 tile
        tile_memory_batched = tile_memory * 1.5

        if total_gb >= 24:
            # 大显存卡预留 4–6GB 给前向峰值，避免 OOM
            safe_memory = max(2.0, available_gb - 6.0)
            max_batch_limit = 8
        else:
            safe_memory = max(2.0, available_gb - 3.0)
            max_batch_limit = 6

        max_batch = max(1, int(safe_memory / tile_memory_batched))
        optimal_batch = min(max_batch, max_batch_limit, len(tile_coords))

        if optimal_batch > 1:
            log(
                f"[Rank {rank}] Adaptive tile_batch_size: {optimal_batch} (GPU avail: {available_gb:.1f}GB, ~{tile_memory:.2f}GB/tile)",
                "info",
                rank,
            )

        return optimal_batch
    except Exception as e:
        log(
            f"[Rank {rank}] Error determining optimal batch size: {e}, falling back to batch_size=1",
            "warning",
            rank,
        )
        return 1


# ---------- Pipeline 初始化 ----------
def init_pipeline_distributed(
    rank: int,
    world_size: int,
    mode: str,
    dtype: torch.dtype,
    model_dir: str,
    use_shared_memory: bool = True,
    device_id: int = None,
):
    """在分布式环境中初始化 pipeline（共享内存 / 错开加载）。"""
    if device_id is None:
        device_id = rank
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)

    model_path = model_dir
    ckpt_path = os.path.join(
        model_path, "diffusion_pytorch_model_streaming_dmd.safetensors"
    )
    vae_path = os.path.join(model_path, "Wan2.1_VAE.pth")
    lq_path = os.path.join(model_path, "LQ_proj_in.ckpt")
    tcd_path = os.path.join(model_path, "TCDecoder.ckpt")
    prompt_path = os.path.join(_project_root, "data", "posi_prompt.pth")

    if not os.path.exists(prompt_path):
        raise RuntimeError(
            f"[Rank {rank}] Missing prompt file: {prompt_path}\n"
            f"  Project root: {_project_root}\n"
            f"  Please ensure data/posi_prompt.pth exists in the project root."
        )

    required_files = [ckpt_path]
    if mode == "full":
        required_files.append(vae_path)
    else:
        required_files.extend([lq_path, tcd_path])

    for p in required_files:
        if not os.path.exists(p):
            raise RuntimeError(f"[Rank {rank}] Missing model file: {p}")

    log(f"[Rank {rank}] Loading model weights...", "info", rank)

    shm_base = "/dev/shm/flashvsr_models"
    use_shm = use_shared_memory and os.path.exists("/dev/shm")

    if use_shm and rank == 0:
        os.makedirs(shm_base, exist_ok=True)
        log(f"[Rank 0] Using shared memory for model loading: {shm_base}", "info", rank)

        shm_ckpt = os.path.join(shm_base, "model_ckpt.pt")
        shm_vae = os.path.join(shm_base, "model_vae.pt") if mode == "full" else None
        shm_tcd = os.path.join(shm_base, "model_tcd.pt") if mode != "full" else None
        shm_lq = os.path.join(shm_base, "model_lq.pt")

        if not os.path.exists(shm_ckpt):
            log(
                f"[Rank 0] Loading model to shared memory (one-time operation)...",
                "info",
                rank,
            )
            mm = ModelManager(torch_dtype=dtype, device="cpu")
            if mode == "full":
                mm.load_models([ckpt_path, vae_path])
                pipe = FlashVSRFullPipeline.from_model_manager(mm, device="cpu")
                pipe.vae.model.encoder = None
                pipe.vae.model.conv1 = None
            else:
                mm.load_models([ckpt_path])
                pipe = (
                    FlashVSRTinyPipeline.from_model_manager(mm, device="cpu")
                    if mode == "tiny"
                    else FlashVSRTinyLongPipeline.from_model_manager(mm, device="cpu")
                )
                multi_scale_channels = [512, 256, 128, 128]
                pipe.TCDecoder = build_tcdecoder(
                    new_channels=multi_scale_channels,
                    device="cpu",
                    dtype=dtype,
                    new_latent_channels=16 + 768,
                )
                pipe.TCDecoder.load_state_dict(
                    torch.load(tcd_path, map_location="cpu"), strict=False
                )
                pipe.TCDecoder.clean_mem()

            pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(
                in_dim=3, out_dim=1536, layer_num=1
            ).to("cpu", dtype=dtype)
            pipe.denoising_model().LQ_proj_in.load_state_dict(
                torch.load(lq_path, map_location="cpu"), strict=True
            )

            torch.save(
                pipe.dit.state_dict() if hasattr(pipe, "dit") else None, shm_ckpt
            )
            if mode == "full" and hasattr(pipe, "vae"):
                torch.save(pipe.vae.state_dict(), shm_vae)
            if mode != "full" and hasattr(pipe, "TCDecoder"):
                torch.save(pipe.TCDecoder.state_dict(), shm_tcd)
            torch.save(
                (
                    pipe.denoising_model().LQ_proj_in.state_dict()
                    if hasattr(pipe, "denoising_model")
                    else None
                ),
                shm_lq,
            )

            del mm, pipe
            gc.collect()
            log(f"[Rank 0] Model saved to shared memory", "info", rank)

        mm = ModelManager(torch_dtype=dtype, device="cpu")
        if mode == "full":
            mm.load_models([ckpt_path, vae_path])
            pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
            pipe.vae.model.encoder = None
            pipe.vae.model.conv1 = None
            if os.path.exists(shm_vae):
                pipe.vae.load_state_dict(torch.load(shm_vae, map_location="cpu"))
        else:
            mm.load_models([ckpt_path])
            pipe = (
                FlashVSRTinyPipeline.from_model_manager(mm, device=device)
                if mode == "tiny"
                else FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
            )
            multi_scale_channels = [512, 256, 128, 128]
            pipe.TCDecoder = build_tcdecoder(
                new_channels=multi_scale_channels,
                device="cpu",
                dtype=dtype,
                new_latent_channels=16 + 768,
            )
            if os.path.exists(shm_tcd):
                pipe.TCDecoder.load_state_dict(
                    torch.load(shm_tcd, map_location="cpu"), strict=False
                )
            pipe.TCDecoder.clean_mem()

        pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(
            in_dim=3, out_dim=1536, layer_num=1
        ).to("cpu", dtype=dtype)
        if os.path.exists(shm_lq):
            pipe.denoising_model().LQ_proj_in.load_state_dict(
                torch.load(shm_lq, map_location="cpu"), strict=True
            )

        del mm
        gc.collect()
    else:
        if rank != 0 and use_shm:
            shm_ckpt = os.path.join(shm_base, "model_ckpt.pt")
            max_wait = 300
            wait_start = time.time()
            while (
                not os.path.exists(shm_ckpt) and (time.time() - wait_start) < max_wait
            ):
                time.sleep(1)
            if not os.path.exists(shm_ckpt):
                log(
                    f"[Rank {rank}] WARNING: Shared memory model file not found after waiting",
                    "warning",
                    rank,
                )

            delay = rank * 2.0
            log(
                f"[Rank {rank}] Waiting {delay:.1f}s before loading model (staggered loading)...",
                "info",
                rank,
            )
            time.sleep(delay)

        mm = ModelManager(torch_dtype=dtype, device="cpu")
        if mode == "full":
            mm.load_models([ckpt_path, vae_path])
            pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
            pipe.vae.model.encoder = None
            pipe.vae.model.conv1 = None
        else:
            mm.load_models([ckpt_path])
            pipe = (
                FlashVSRTinyPipeline.from_model_manager(mm, device=device)
                if mode == "tiny"
                else FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
            )
            multi_scale_channels = [512, 256, 128, 128]
            pipe.TCDecoder = build_tcdecoder(
                new_channels=multi_scale_channels,
                device="cpu",
                dtype=dtype,
                new_latent_channels=16 + 768,
            )
            pipe.TCDecoder.load_state_dict(
                torch.load(tcd_path, map_location="cpu"), strict=False
            )
            pipe.TCDecoder.clean_mem()

        pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(
            in_dim=3, out_dim=1536, layer_num=1
        ).to("cpu", dtype=dtype)
        pipe.denoising_model().LQ_proj_in.load_state_dict(
            torch.load(lq_path, map_location="cpu"), strict=True
        )

        del mm
        gc.collect()

    log(f"[Rank {rank}] Moving model to GPU {rank}...", "info", rank)
    pipe.to(device, dtype=dtype)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path=prompt_path)
    pipe.load_models_to_device(["dit", "vae"])

    gc.collect()
    torch.cuda.empty_cache()

    log(f"[Rank {rank}] Pipeline initialized successfully", "finish", rank)
    return pipe, device


# ---------- Tile 批处理与 Segment 推理 ----------
def _process_tile_batch_distributed_single(
    pipe, frames, device, dtype, args, tile_batch, batch_idx, rank,
):
    """逐 tile 调用 pipe（兼容 F<17 或 batched 失败时的回退）。"""
    N, H, W, C = frames.shape
    results = []
    for tile_idx, (x1, y1, x2, y2) in enumerate(tile_batch):
        input_tile = frames[:, y1:y2, x1:x2, :]
        N_tile = input_tile.shape[0]
        LQ_tile, th, tw, F, N0_tile = prepare_input_tensor(
            input_tile, device, scale=args.scale, dtype=dtype
        )
        if "long" not in args.mode:
            LQ_tile = LQ_tile.to(device)
        topk_ratio = args.sparse_ratio * 768 * 1280 / (th * tw)
        if F < 17:
            log(
                f"[Rank {rank}] WARNING: Tile has only {F} frames, minimum is 17. Skipping this tile.",
                "warning",
                rank,
            )
            placeholder = frames[-1:, y1:y2, x1:x2, :].repeat(N0_tile, 1, 1, 1)
            results.append({
                "coords": (x1, y1, x2, y2),
                "tile": placeholder,
                "mask": torch.ones(1, placeholder.shape[1], placeholder.shape[2], 1),
            })
            continue
        with torch.no_grad():
            try:
                # ===== 验证：pipe 前清 cache + 同步 =====
                pipe.denoising_model().LQ_proj_in.clear_cache()
                torch.cuda.synchronize()

                output_tile = pipe(
                    prompt="",
                    negative_prompt="",
                    cfg_scale=1.0,
                    num_inference_steps=1,
                    seed=args.seed,
                    tiled=args.tiled_vae,
                    LQ_video=LQ_tile,
                    num_frames=F,
                    height=th,
                    width=tw,
                    is_full_block=False,
                    if_buffer=True,
                    topk_ratio=topk_ratio,
                    kv_ratio=args.kv_ratio,
                    local_range=args.local_range,
                    color_fix=args.color_fix,
                    unload_dit=args.unload_dit,
                )

                # ===== 验证：pipe 后清 cache + 同步 =====
                pipe.denoising_model().LQ_proj_in.clear_cache()
                torch.cuda.synchronize()
                
            except ValueError as e:
                if "expected a non-empty list" in str(e):
                    log(
                        f"[Rank {rank}] ERROR: Pipeline returned empty latents. Tile coords=({x1},{y1})-({x2},{y2}), th={th}, tw={tw}, F={F}",
                        "error",
                        rank,
                    )
                    raise RuntimeError(
                        f"Pipeline failed: empty latents. Tile may be too small or have insufficient frames (F={F}, min=17)"
                    ) from e
                raise
        if isinstance(output_tile, (tuple, list)):
            output_tile = output_tile[0]
        processed_tile_cpu = tensor2video(output_tile).to("cpu")
        if processed_tile_cpu.shape[0] < N0_tile:
            raise RuntimeError(
                f"[Rank {rank}] ERROR: Pipeline returned only {processed_tile_cpu.shape[0]} frames, need {N0_tile}."
            )
        if processed_tile_cpu.shape[0] > N0_tile:
            processed_tile_cpu = processed_tile_cpu[:N0_tile]
        sH, sW = (y2 - y1) * args.scale, (x2 - x1) * args.scale
        th_out, tw_out = processed_tile_cpu.shape[1], processed_tile_cpu.shape[2]
        if th_out > sH or tw_out > sW:
            processed_tile_cpu = processed_tile_cpu[:, :sH, :sW, :]
        mask_nchw = create_feather_mask((th_out, tw_out), args.tile_overlap).to("cpu")
        mask_nchw = mask_nchw[:, :, :sH, :sW]
        mask_nhwc = mask_nchw.permute(0, 2, 3, 1).expand(
            processed_tile_cpu.shape[0], -1, -1, -1
        )
        results.append(
            {"coords": (x1, y1, x2, y2), "tile": processed_tile_cpu, "mask": mask_nhwc}
        )
        del LQ_tile, output_tile, processed_tile_cpu, input_tile
        clean_vram()
    return results


def process_tile_batch_distributed(
    pipe,
    frames,
    device,
    dtype,
    args,
    tile_batch: List[Tuple[int, int, int, int]],
    batch_idx: int,
    rank: int,
):
    """处理一批 tiles，返回 coords/tile/mask 列表。B>1 时一次 pipe 调用，否则或失败时逐 tile 调用。
    多 GPU + 多 tile 时默认强制逐 tile(B=1) 避免乱码；传 --allow_tile_batch_multigpu 时则允许多 tile 批处理。
    """
    N, H, W, C = frames.shape
    world_size = getattr(args, "world_size", 1)
    allow_multi = getattr(args, "allow_tile_batch_multigpu", False)
    if world_size > 1 and len(tile_batch) > 1 and not allow_multi:
        if batch_idx == 0:
            log(
                f"[Rank {rank}] Multi-GPU + tile_batch>1 易乱码，本 rank 改为逐 tile 调 pipe(B=1)（与多 GPU 单 tile 同路径）",
                "info",
                rank,
            )
        out = []
        for one_coord in tile_batch:
            part = _process_tile_batch_distributed_single(
                pipe, frames, device, dtype, args, [one_coord], batch_idx, rank
            )
            out.extend(part)
        return out

    if len(tile_batch) <= 1:
        return _process_tile_batch_distributed_single(
            pipe, frames, device, dtype, args, tile_batch, batch_idx, rank
        )

    # 准备所有 tile 的输入
    tile_data = []
    for (x1, y1, x2, y2) in tile_batch:
        input_tile = frames[:, y1:y2, x1:x2, :]
        LQ_tile, th, tw, F, N0_tile = prepare_input_tensor(
            input_tile, device, scale=args.scale, dtype=dtype
        )
        if F < 17:
            log(
                f"[Rank {rank}] WARNING: Batched tile has F={F}<17, falling back to single-tile processing.",
                "warning",
                rank,
            )
            return _process_tile_batch_distributed_single(
                pipe, frames, device, dtype, args, tile_batch, batch_idx, rank
            )
        tile_data.append({
            "LQ_tile": LQ_tile,
            "th": th,
            "tw": tw,
            "F": F,
            "N0_tile": N0_tile,
            "coords": (x1, y1, x2, y2),
            "sH": (y2 - y1) * args.scale,
            "sW": (x2 - x1) * args.scale,
        })

    max_F = max(d["F"] for d in tile_data)
    max_th = max(d["th"] for d in tile_data)
    max_tw = max(d["tw"] for d in tile_data)

    # Pad 到同一 (max_F, max_th, max_tw) 并 stack
    batched_list = []
    for d in tile_data:
        LQ = d["LQ_tile"]
        F, th, tw = d["F"], d["th"], d["tw"]
        pad_F = max_F - F
        pad_th = max_th - th
        pad_tw = max_tw - tw
        if pad_F > 0 or pad_th > 0 or pad_tw > 0:
            LQ = torch.nn.functional.pad(
                LQ,
                (0, pad_tw, 0, pad_th, 0, pad_F),
                mode="replicate",
            )
        batched_list.append(LQ)
    batched_LQ = torch.cat(batched_list, dim=0)
    if "long" not in args.mode:
        batched_LQ = batched_LQ.to(device)

    topk_ratio = args.sparse_ratio * 768 * 1280 / (max_th * max_tw)

    B = len(tile_batch)
    log(
        f"[Rank {rank}] Batched inference: {B} tiles in one pipe call (batch {batch_idx + 1})",
        "info",
        rank,
    )
    if batch_idx == 0:
        log(
            f"[Rank {rank}] pipe() input: batch_size={B}, LQ_video.shape={tuple(batched_LQ.shape)}",
            "info",
            rank,
        )
    # 每个 (rank, batch_idx) 使用不同 seed，避免多 batch 重复同一噪声导致条纹/周期伪影
    base_seed = getattr(args, "seed", None) or 0
    batch_seed = base_seed + rank * 10000 + batch_idx * 100
    with torch.no_grad():
        try:
            # ===== 验证：pipe 前清 cache + 同步 =====
            pipe.denoising_model().LQ_proj_in.clear_cache()
            torch.cuda.synchronize()

            output_batched = pipe(
                prompt="",
                negative_prompt="",
                cfg_scale=1.0,
                num_inference_steps=1,
                seed=batch_seed,
                tiled=args.tiled_vae,
                LQ_video=batched_LQ,
                num_frames=max_F,
                height=max_th,
                width=max_tw,
                is_full_block=False,
                if_buffer=True,
                topk_ratio=topk_ratio,
                kv_ratio=args.kv_ratio,
                local_range=args.local_range,
                color_fix=args.color_fix,
                unload_dit=args.unload_dit,
            )
            # ===== 验证：pipe 后清 cache + 同步 =====
            pipe.denoising_model().LQ_proj_in.clear_cache()
            torch.cuda.synchronize()

        except (RuntimeError, ValueError) as e:
            log(
                f"[Rank {rank}] Batched pipe failed ({e}), falling back to single-tile processing.",
                "warning",
                rank,
            )
            del batched_LQ, batched_list
            clean_vram()
            return _process_tile_batch_distributed_single(
                pipe, frames, device, dtype, args, tile_batch, batch_idx, rank
            )

    if isinstance(output_batched, (tuple, list)):
        output_batched = output_batched[0]
    # output_batched: (B, C, T, H, W)
    B = output_batched.shape[0]
    results = []
    for b in range(B):
        out_b = output_batched[b : b + 1]
        processed_b = tensor2video(out_b).to("cpu")
        d = tile_data[b]
        N0_tile, sH, sW = d["N0_tile"], d["sH"], d["sW"]
        if processed_b.shape[0] < N0_tile:
            raise RuntimeError(
                f"[Rank {rank}] ERROR: Batched output tile {b} has {processed_b.shape[0]} frames, need {N0_tile}."
            )
        if processed_b.shape[0] > N0_tile:
            processed_b = processed_b[:N0_tile]
        th_out, tw_out = processed_b.shape[1], processed_b.shape[2]
        if th_out > sH or tw_out > sW:
            processed_b = processed_b[:, :sH, :sW, :]
        mask_nchw = create_feather_mask((th_out, tw_out), args.tile_overlap).to("cpu")
        mask_nchw = mask_nchw[:, :, :sH, :sW]
        mask_nhwc = mask_nchw.permute(0, 2, 3, 1).expand(
            processed_b.shape[0], -1, -1, -1
        )
        results.append({
            "coords": d["coords"],
            "tile": processed_b,
            "mask": mask_nhwc,
        })

    if batch_idx == 0 or (batch_idx + 1) % 100 == 0:
        log(
            f"[Rank {rank}] Batched completed: batch {batch_idx + 1} ({B} tiles)",
            "info",
            rank,
        )
    del batched_LQ, output_batched, batched_list
    clean_vram()
    return results


def _process_tile_batch_distributed_single_with_crops(
    pipe,
    tile_tensors: List[torch.Tensor],
    tile_batch: List[Tuple[int, int, int, int]],
    device,
    dtype,
    args,
    batch_idx: int,
    rank: int,
):
    """逐 tile 调用 pipe，输入为已裁好的 tile 张量列表（流式用）。tile_tensors[i] 对应 tile_batch[i]，形状 (N, tile_h, tile_w, C)。"""
    results = []
    for tile_idx, (x1, y1, x2, y2) in enumerate(tile_batch):
        input_tile = tile_tensors[tile_idx]
        N_tile = input_tile.shape[0]
        LQ_tile, th, tw, F, N0_tile = prepare_input_tensor(
            input_tile, device, scale=args.scale, dtype=dtype
        )
        if "long" not in args.mode:
            LQ_tile = LQ_tile.to(device)
        topk_ratio = args.sparse_ratio * 768 * 1280 / (th * tw)
        if F < 17:
            log(
                f"[Rank {rank}] WARNING: Tile has only {F} frames, minimum is 17. Skipping this tile.",
                "warning",
                rank,
            )
            placeholder = input_tile[-1:].repeat(N0_tile, 1, 1, 1)
            results.append({
                "coords": (x1, y1, x2, y2),
                "tile": placeholder,
                "mask": torch.ones(1, placeholder.shape[1], placeholder.shape[2], 1),
            })
            continue
        with torch.no_grad():
            try:
                output_tile = pipe(
                    prompt="",
                    negative_prompt="",
                    cfg_scale=1.0,
                    num_inference_steps=1,
                    seed=args.seed,
                    tiled=args.tiled_vae,
                    LQ_video=LQ_tile,
                    num_frames=F,
                    height=th,
                    width=tw,
                    is_full_block=False,
                    if_buffer=True,
                    topk_ratio=topk_ratio,
                    kv_ratio=args.kv_ratio,
                    local_range=args.local_range,
                    color_fix=args.color_fix,
                    unload_dit=args.unload_dit,
                )
            except ValueError as e:
                if "expected a non-empty list" in str(e):
                    log(
                        f"[Rank {rank}] ERROR: Pipeline returned empty latents. Tile coords=({x1},{y1})-({x2},{y2}), th={th}, tw={tw}, F={F}",
                        "error",
                        rank,
                    )
                    raise RuntimeError(
                        f"Pipeline failed: empty latents. Tile may be too small or have insufficient frames (F={F}, min=17)"
                    ) from e
                raise
        if isinstance(output_tile, (tuple, list)):
            output_tile = output_tile[0]
        processed_tile_cpu = tensor2video(output_tile).to("cpu")
        if processed_tile_cpu.shape[0] < N0_tile:
            raise RuntimeError(
                f"[Rank {rank}] ERROR: Pipeline returned only {processed_tile_cpu.shape[0]} frames, need {N0_tile}."
            )
        if processed_tile_cpu.shape[0] > N0_tile:
            processed_tile_cpu = processed_tile_cpu[:N0_tile]
        sH, sW = (y2 - y1) * args.scale, (x2 - x1) * args.scale
        th_out, tw_out = processed_tile_cpu.shape[1], processed_tile_cpu.shape[2]
        if th_out > sH or tw_out > sW:
            processed_tile_cpu = processed_tile_cpu[:, :sH, :sW, :]
        mask_nchw = create_feather_mask((th_out, tw_out), args.tile_overlap).to("cpu")
        mask_nchw = mask_nchw[:, :, :sH, :sW]
        mask_nhwc = mask_nchw.permute(0, 2, 3, 1).expand(
            processed_tile_cpu.shape[0], -1, -1, -1
        )
        results.append(
            {"coords": (x1, y1, x2, y2), "tile": processed_tile_cpu, "mask": mask_nhwc}
        )
        del LQ_tile, output_tile, processed_tile_cpu, input_tile
        clean_vram()
    return results


def process_tile_batch_distributed_with_crops(
    pipe,
    tile_tensors: List[torch.Tensor],
    device,
    dtype,
    args,
    tile_batch: List[Tuple[int, int, int, int]],
    batch_idx: int,
    rank: int,
):
    """处理一批 tiles，输入为已裁好的 tile 张量列表（流式用）。tile_tensors[i] 对应 tile_batch[i]。"""
    if len(tile_batch) <= 1:
        return _process_tile_batch_distributed_single_with_crops(
            pipe, tile_tensors, tile_batch, device, dtype, args, batch_idx, rank
        )

    tile_data = []
    for t_idx, (x1, y1, x2, y2) in enumerate(tile_batch):
        input_tile = tile_tensors[t_idx]
        LQ_tile, th, tw, F, N0_tile = prepare_input_tensor(
            input_tile, device, scale=args.scale, dtype=dtype
        )
        if F < 17:
            log(
                f"[Rank {rank}] WARNING: Batched tile has F={F}<17, falling back to single-tile processing.",
                "warning",
                rank,
            )
            return _process_tile_batch_distributed_single_with_crops(
                pipe, tile_tensors, tile_batch, device, dtype, args, batch_idx, rank
            )
        tile_data.append({
            "LQ_tile": LQ_tile,
            "th": th,
            "tw": tw,
            "F": F,
            "N0_tile": N0_tile,
            "coords": (x1, y1, x2, y2),
            "sH": (y2 - y1) * args.scale,
            "sW": (x2 - x1) * args.scale,
        })

    max_F = max(d["F"] for d in tile_data)
    max_th = max(d["th"] for d in tile_data)
    max_tw = max(d["tw"] for d in tile_data)

    batched_list = []
    for d in tile_data:
        LQ = d["LQ_tile"]
        F, th, tw = d["F"], d["th"], d["tw"]
        pad_F = max_F - F
        pad_th = max_th - th
        pad_tw = max_tw - tw
        if pad_F > 0 or pad_th > 0 or pad_tw > 0:
            LQ = torch.nn.functional.pad(
                LQ,
                (0, pad_tw, 0, pad_th, 0, pad_F),
                mode="replicate",
            )
        batched_list.append(LQ)
    batched_LQ = torch.cat(batched_list, dim=0)
    if "long" not in args.mode:
        batched_LQ = batched_LQ.to(device)

    topk_ratio = args.sparse_ratio * 768 * 1280 / (max_th * max_tw)
    B = len(tile_batch)
    log(
        f"[Rank {rank}] Batched inference (streaming): {B} tiles in one pipe call (batch {batch_idx + 1})",
        "info",
        rank,
    )
    with torch.no_grad():
        try:
            output_batched = pipe(
                prompt="",
                negative_prompt="",
                cfg_scale=1.0,
                num_inference_steps=1,
                seed=args.seed,
                tiled=args.tiled_vae,
                LQ_video=batched_LQ,
                num_frames=max_F,
                height=max_th,
                width=max_tw,
                is_full_block=False,
                if_buffer=True,
                topk_ratio=topk_ratio,
                kv_ratio=args.kv_ratio,
                local_range=args.local_range,
                color_fix=args.color_fix,
                unload_dit=args.unload_dit,
            )
        except (RuntimeError, ValueError) as e:
            log(
                f"[Rank {rank}] Batched pipe failed ({e}), falling back to single-tile processing.",
                "warning",
                rank,
            )
            del batched_LQ, batched_list
            clean_vram()
            return _process_tile_batch_distributed_single_with_crops(
                pipe, tile_tensors, tile_batch, device, dtype, args, batch_idx, rank
            )

    if isinstance(output_batched, (tuple, list)):
        output_batched = output_batched[0]
    B = output_batched.shape[0]
    results = []
    for b in range(B):
        out_b = output_batched[b : b + 1]
        processed_b = tensor2video(out_b).to("cpu")
        d = tile_data[b]
        N0_tile, sH, sW = d["N0_tile"], d["sH"], d["sW"]
        if processed_b.shape[0] < N0_tile:
            raise RuntimeError(
                f"[Rank {rank}] ERROR: Batched output tile {b} has {processed_b.shape[0]} frames, need {N0_tile}."
            )
        if processed_b.shape[0] > N0_tile:
            processed_b = processed_b[:N0_tile]
        th_out, tw_out = processed_b.shape[1], processed_b.shape[2]
        if th_out > sH or tw_out > sW:
            processed_b = processed_b[:, :sH, :sW, :]
        mask_nchw = create_feather_mask((th_out, tw_out), args.tile_overlap).to("cpu")
        mask_nchw = mask_nchw[:, :, :sH, :sW]
        mask_nhwc = mask_nchw.permute(0, 2, 3, 1).expand(
            processed_b.shape[0], -1, -1, -1
        )
        results.append({
            "coords": d["coords"],
            "tile": processed_b,
            "mask": mask_nhwc,
        })
    del batched_LQ, output_batched, batched_list
    clean_vram()
    return results


def _run_tiled_inference_one_pass(
    pipe,
    frames,
    device,
    dtype,
    args,
    rank: int,
    tile_coords: List[Tuple[int, int, int, int]],
    N: int,
    out_H: int,
    out_W: int,
    C: int,
    checkpoint_dir: str | None,
    *,
    input_path: str = None,
    start_idx: int = None,
    end_idx: int = None,
    input_fps: float = 30.0,
    enable_hdr: bool = False,
    streaming_hdr_params: dict = None,
):
    """跑一遍 tiled 推理并累积到 canvas/weight_canvas，返回 (canvas, weight_canvas)。
    checkpoint_dir 为 None 时不使用 mmap 与断点续跑（用于 tile_shift 双遍时的每一遍）。
    多 GPU 时强制用内存 canvas：与单 GPU 一致（单 GPU 传 checkpoint_dir=None 故用内存），
    避免 memmap + batched 累加时的视图/写入顺序导致乱码。
    流式输入：当 frames 为 None 时，必须传 input_path, start_idx, end_idx；按 tile 逐批从磁盘读取。
    """
    use_streaming = frames is None and input_path is not None and start_idx is not None and end_idx is not None
    processed_tiles_file = (
        os.path.join(checkpoint_dir, f"rank_{rank}_processed_tiles.json")
        if checkpoint_dir
        else None
    )
    processed_tiles = set()
    if processed_tiles_file and os.path.exists(processed_tiles_file):
        try:
            with open(processed_tiles_file, "r") as f:
                processed_tiles = set(json.load(f))
            log(
                f"[Rank {rank}] Resuming from checkpoint: {len(processed_tiles)}/{len(tile_coords)} tiles already processed",
                "info",
                rank,
            )
        except Exception:
            pass

    # 单 GPU 时 run_inference_distributed_segment 传 checkpoint_dir=None → 用内存；多 GPU 时传了 checkpoint_dir 会走 mmap。
    # 多 GPU + batched tile 时 memmap 上的 in-place 累加易出乱码，故多 GPU 也改用内存 canvas，与单 GPU 行为一致。
    world_size = getattr(args, "world_size", 1)
    use_mmap = checkpoint_dir is not None and world_size <= 1
    if checkpoint_dir and world_size > 1:
        log(
            f"[Rank {rank}] Multi-GPU: using in-memory canvas (same as single-GPU) to avoid mmap batched-write artifacts",
            "info",
            rank,
        )
    canvas = None
    weight_canvas = None
    canvas_mmap = None
    weight_mmap = None
    canvas_mmap_path = None
    weight_mmap_path = None

    if use_mmap:
        canvas_mmap_path = os.path.join(checkpoint_dir, f"rank_{rank}_canvas.npy")
        weight_mmap_path = os.path.join(checkpoint_dir, f"rank_{rank}_weight.npy")

        if os.path.exists(canvas_mmap_path):
            log(
                f"[Rank {rank}] Loading existing memory-mapped canvas from {canvas_mmap_path}",
                "info",
                rank,
            )
            existing_mmap = np.memmap(canvas_mmap_path, dtype=np.float16, mode="r")
            if existing_mmap.shape[0] != N:
                log(
                    f"[Rank {rank}] WARNING: Existing canvas has {existing_mmap.shape[0]} frames, expected {N}. Recreating...",
                    "warning",
                    rank,
                )
                os.remove(canvas_mmap_path)
                os.remove(weight_mmap_path)
                canvas_mmap = np.memmap(
                    canvas_mmap_path,
                    dtype=np.float16,
                    mode="w+",
                    shape=(N, out_H, out_W, C),
                )
                weight_mmap = np.memmap(
                    weight_mmap_path,
                    dtype=np.float16,
                    mode="w+",
                    shape=(N, out_H, out_W, C),
                )
                canvas_mmap[:] = 0
                weight_mmap[:] = 0
            else:
                canvas_mmap = np.memmap(
                    canvas_mmap_path,
                    dtype=np.float16,
                    mode="r+",
                    shape=(N, out_H, out_W, C),
                )
                weight_mmap = np.memmap(
                    weight_mmap_path,
                    dtype=np.float16,
                    mode="r+",
                    shape=(N, out_H, out_W, C),
                )
        else:
            log(
                f"[Rank {rank}] Creating memory-mapped canvas: {canvas_mmap_path} (shape: {N}x{out_H}x{out_W}x{C})",
                "info",
                rank,
            )
            canvas_mmap = np.memmap(
                canvas_mmap_path,
                dtype=np.float16,
                mode="w+",
                shape=(N, out_H, out_W, C),
            )
            weight_mmap = np.memmap(
                weight_mmap_path,
                dtype=np.float16,
                mode="w+",
                shape=(N, out_H, out_W, C),
            )
            canvas_mmap[:] = 0
            weight_mmap[:] = 0
            canvas_mmap.flush()
            weight_mmap.flush()

        canvas = torch.from_numpy(canvas_mmap)
        weight_canvas = torch.from_numpy(weight_mmap)
    else:
        canvas = torch.zeros((N, out_H, out_W, C), dtype=torch.float16, device="cpu")
        weight_canvas = torch.zeros_like(canvas)

    optimal_batch_size = determine_optimal_batch_size(
        device, tile_coords,
        frames if not use_streaming else None,
        args, rank,
        segment_n_frames=N if use_streaming else None,
    )
    log(
        f"[Rank {rank}] Processing {len(tile_coords)} tiles with batch_size={optimal_batch_size} "
        f"(batched tile inference when batch_size>1; this may take a while)...",
        "info",
        rank,
    )

    tile_batches = [
        tile_coords[i : i + optimal_batch_size]
        for i in range(0, len(tile_coords), optimal_batch_size)
    ]

    total_processed = 0
    flush_counter = 0

    for batch_idx, tile_batch in enumerate(tile_batches):
        tiles_to_process = []
        tile_keys_in_batch = []

        for tile_coord in tile_batch:
            x1, y1, x2, y2 = tile_coord
            tile_key = f"{x1}_{y1}_{x2}_{y2}"
            tile_keys_in_batch.append(tile_key)
            if tile_key not in processed_tiles:
                tiles_to_process.append(tile_coord)
            else:
                log(
                    f"[Rank {rank}] Tile {tile_key} already processed, skipping",
                    "info",
                    rank,
                )

        if not tiles_to_process:
            total_processed += len(tile_batch)
            continue

        try:
            if use_streaming:
                tile_tensors, _ = inference_io.read_input_frames_range_tile_crops(
                    input_path, start_idx, end_idx, tiles_to_process,
                    fps=input_fps, enable_hdr=enable_hdr,
                    streaming_hdr_params=streaming_hdr_params, args=args,
                )
                results = process_tile_batch_distributed_with_crops(
                    pipe, tile_tensors, device, dtype, args, tiles_to_process, batch_idx, rank
                )
                del tile_tensors
            else:
                results = process_tile_batch_distributed(
                    pipe, frames, device, dtype, args, tiles_to_process, batch_idx, rank
                )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log(
                    f"[Rank {rank}] OOM with batch_size={len(tiles_to_process)}, falling back to single tile processing",
                    "warning",
                    rank,
                )
                for tile_coord in tiles_to_process:
                    x1, y1, x2, y2 = tile_coord
                    tile_key = f"{x1}_{y1}_{x2}_{y2}"
                    if tile_key in processed_tiles:
                        continue
                    single_tile_batch = [tile_coord]
                    try:
                        if use_streaming:
                            tile_tensors, _ = inference_io.read_input_frames_range_tile_crops(
                                input_path, start_idx, end_idx, single_tile_batch,
                                fps=input_fps, enable_hdr=enable_hdr,
                                streaming_hdr_params=streaming_hdr_params, args=args,
                            )
                            results = process_tile_batch_distributed_with_crops(
                                pipe, tile_tensors, device, dtype, args, single_tile_batch, batch_idx, rank
                            )
                            del tile_tensors
                        else:
                            results = process_tile_batch_distributed(
                                pipe,
                                frames,
                                device,
                                dtype,
                                args,
                                single_tile_batch,
                                batch_idx,
                                rank,
                            )
                        for result in results:
                            x1, y1, x2, y2 = result["coords"]
                            processed_tile_cpu = result["tile"]
                            mask_nhwc = result["mask"]
                            if processed_tile_cpu.shape[0] != N:
                                raise RuntimeError(
                                    f"[Rank {rank}] ERROR: Tile output frames ({processed_tile_cpu.shape[0]}) != canvas frames ({N}). "
                                    f"Tile coords: ({x1},{y1})-({x2},{y2})"
                                )
                            out_x1, out_y1 = x1 * args.scale, y1 * args.scale
                            tile_H_scaled = processed_tile_cpu.shape[1]
                            tile_W_scaled = processed_tile_cpu.shape[2]
                            out_x2, out_y2 = (
                                out_x1 + tile_W_scaled,
                                out_y1 + tile_H_scaled,
                            )
                            canvas[:, out_y1:out_y2, out_x1:out_x2, :] += (
                                processed_tile_cpu * mask_nhwc
                            )
                            weight_canvas[
                                :, out_y1:out_y2, out_x1:out_x2, :
                            ] += mask_nhwc
                            processed_tiles.add(tile_key)
                    except RuntimeError as e2:
                        log(
                            f"[Rank {rank}] ERROR: Failed to process tile {tile_key}: {e2}",
                            "error",
                            rank,
                        )
                        raise
            else:
                raise

        for result in results:
            x1, y1, x2, y2 = result["coords"]
            processed_tile_cpu = result["tile"]
            mask_nhwc = result["mask"]
            if processed_tile_cpu.shape[0] != N:
                raise RuntimeError(
                    f"[Rank {rank}] ERROR: Tile output frames ({processed_tile_cpu.shape[0]}) != canvas frames ({N}). "
                    f"Tile coords: ({x1},{y1})-({x2},{y2})"
                )
            out_x1, out_y1 = x1 * args.scale, y1 * args.scale
            tile_H_scaled = processed_tile_cpu.shape[1]
            tile_W_scaled = processed_tile_cpu.shape[2]
            out_x2, out_y2 = out_x1 + tile_W_scaled, out_y1 + tile_H_scaled
            canvas[:, out_y1:out_y2, out_x1:out_x2, :] += processed_tile_cpu * mask_nhwc
            weight_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_nhwc

        for result in results:
            x1, y1, x2, y2 = result["coords"]
            tile_key = f"{x1}_{y1}_{x2}_{y2}"
            if tile_key not in processed_tiles:
                processed_tiles.add(tile_key)
                total_processed += 1

        if processed_tiles_file and (
            batch_idx % 5 == 0 or batch_idx == len(tile_batches) - 1
        ):
            try:
                with open(processed_tiles_file, "w") as f:
                    json.dump(list(processed_tiles), f)
            except Exception:
                pass

        flush_counter += 1
        if use_mmap and (flush_counter % 5 == 0 or batch_idx == len(tile_batches) - 1):
            canvas_mmap.flush()
            weight_mmap.flush()

        clean_vram()

        progress_interval = max(1, min(len(tile_batches) // 10, 10))
        if (batch_idx + 1) % progress_interval == 0 or (batch_idx + 1) == len(
            tile_batches
        ):
            percentage = 100.0 * total_processed / len(tile_coords)
            log(
                f"[Rank {rank}] Tile progress: {total_processed}/{len(tile_coords)} ({percentage:.1f}%) [batch {batch_idx + 1}/{len(tile_batches)}]",
                "info",
                rank,
            )
            if checkpoint_dir:
                progress_file = os.path.join(
                    checkpoint_dir, f"rank_{rank}_progress.txt"
                )
                try:
                    with open(progress_file, "w") as f:
                        f.write(
                            f"{total_processed}\n{len(tile_coords)}\n{percentage:.1f}\n"
                        )
                except Exception:
                    pass

    if use_mmap:
        canvas_mmap.flush()
        weight_mmap.flush()

    return canvas, weight_canvas


def run_inference_distributed_segment(
    pipe, frames, device, dtype, args, rank: int, checkpoint_dir: str = None,
    *,
    input_path: str = None,
    start_idx: int = None,
    end_idx: int = None,
    input_fps: float = 30.0,
    streaming_input: bool = False,
    streaming_hdr_params: dict = None,
):
    """在单个 segment 上运行推理。streaming_input=True 时 frames 可为 None，按 tile 从 input_path 流式读取。"""
    enable_hdr = getattr(args, "_enable_hdr", False)
    if streaming_input and input_path is not None and start_idx is not None and end_idx is not None:
        N, H, W, C = inference_io.get_segment_shape(input_path, start_idx, end_idx, enable_hdr=enable_hdr)
        if not args.tiled_dit:
            log(
                f"[Rank {rank}] Streaming input only supports tiled_dit=True; whole-frame path requires loading full segment.",
                "warning",
                rank,
            )
            streaming_input = False
            frames, _ = inference_io.read_input_frames_range(
                input_path, start_idx, end_idx, fps=input_fps, enable_hdr=enable_hdr
            )
            frames, _ = inference_io.apply_hdr_tone_mapping_if_needed(
                frames, args, rank=rank, checkpoint_dir=checkpoint_dir
            )
            N, H, W, C = frames.shape
    else:
        N, H, W, C = frames.shape
    out_H, out_W = H * args.scale, W * args.scale

    if not args.tiled_dit:
        log(
            f"[Rank {rank}] Whole-frame path (tiled_dit=False, tiled_vae={args.tiled_vae})",
            "info",
            rank,
        )
        LQ, th, tw, F, N0 = prepare_input_tensor(
            frames, device, scale=args.scale, dtype=dtype
        )
        if "long" not in args.mode:
            LQ = LQ.to(device)
        topk_ratio = args.sparse_ratio * 768 * 1280 / (th * tw)
        with torch.no_grad():
            output = pipe(
                prompt="",
                negative_prompt="",
                cfg_scale=1.0,
                num_inference_steps=1,
                seed=args.seed,
                tiled=args.tiled_vae,
                LQ_video=LQ,
                num_frames=F,
                height=th,
                width=tw,
                is_full_block=False,
                if_buffer=True,
                topk_ratio=topk_ratio,
                kv_ratio=args.kv_ratio,
                local_range=args.local_range,
                color_fix=args.color_fix,
                unload_dit=args.unload_dit,
            )
        if isinstance(output, (tuple, list)):
            output = output[0]
        final_output = tensor2video(output).to("cpu")
        if final_output.shape[0] > N:
            final_output = final_output[:N]
        # 按真实输出尺寸裁剪，避免 128 对齐多出的右/下边出现黑边
        if final_output.shape[1] > out_H or final_output.shape[2] > out_W:
            final_output = final_output[:, :out_H, :out_W, :]
        del output, LQ
        clean_vram()
        log(
            f"[Rank {rank}] Whole-frame done: {final_output.shape[0]} frames (1-to-1 with input)",
            "info",
            rank,
        )
        return final_output

    log(
        f"[Rank {rank}] Tile path (tiled_dit=True, tiled_vae={args.tiled_vae})",
        "info",
        rank,
    )
    tile_shift = getattr(args, "tile_shift", True)
    if streaming_input and tile_shift:
        log(
            f"[Rank {rank}] Streaming input: tile_shift disabled (single-pass tiled only).",
            "info",
            rank,
        )
        tile_shift = False

    if tile_shift:
        # Tile Shift 双遍推理：正常网格 + 错位半 stride，两遍结果平均以打散固定网格纹
        stride = args.tile_size - args.tile_overlap
        shift_x, shift_y = stride // 2, stride // 2
        coords1 = calculate_tile_coords(H, W, args.tile_size, args.tile_overlap)
        coords2 = calculate_tile_coords(
            H, W, args.tile_size, args.tile_overlap, shift_x, shift_y
        )
        log(
            f"[Rank {rank}] Tile shift enabled: Pass1 {len(coords1)} tiles, Pass2 {len(coords2)} tiles (shift={shift_x},{shift_y})",
            "info",
            rank,
        )
        log(
            f"[Rank {rank}] Input resolution: {H}x{W}, Tile size: {args.tile_size}, Overlap: {args.tile_overlap}",
            "info",
            rank,
        )
        log(f"[Rank {rank}] Pass 1: normal grid", "info", rank)
        canvas1, weight1 = _run_tiled_inference_one_pass(
            pipe, frames, device, dtype, args, rank,
            coords1, N, out_H, out_W, C, checkpoint_dir=None,
        )
        log(f"[Rank {rank}] Pass 2: shifted grid (half stride)", "info", rank)
        canvas2, weight2 = _run_tiled_inference_one_pass(
            pipe, frames, device, dtype, args, rank,
            coords2, N, out_H, out_W, C, checkpoint_dir=None,
        )
        # 按权重合并两遍：仅第一遍覆盖的区域（上/左边界）保持正常亮度，重叠区按权重混合
        total_canvas = canvas1 + canvas2
        total_weight = weight1 + weight2
        output = (total_canvas / (total_weight + 1e-6)).float()
    else:
        tile_coords = calculate_tile_coords(H, W, args.tile_size, args.tile_overlap)
        log(
            f"[Rank {rank}] Input resolution: {H}x{W}, Tile size: {args.tile_size}, Overlap: {args.tile_overlap}",
            "info",
            rank,
        )
        log(f"[Rank {rank}] Calculated {len(tile_coords)} tiles to process", "info", rank)
        if streaming_input:
            log(f"[Rank {rank}] Using streaming input (per-tile load from disk).", "info", rank)
        canvas, weight_canvas = _run_tiled_inference_one_pass(
            pipe, frames, device, dtype, args, rank,
            tile_coords, N, out_H, out_W, C, checkpoint_dir,
            input_path=input_path if streaming_input else None,
            start_idx=start_idx if streaming_input else None,
            end_idx=end_idx if streaming_input else None,
            input_fps=input_fps if streaming_input else 30.0,
            enable_hdr=enable_hdr if streaming_input else False,
            streaming_hdr_params=streaming_hdr_params if streaming_input else None,
        )
        if checkpoint_dir:
            progress_file = os.path.join(checkpoint_dir, f"rank_{rank}_progress.txt")
            try:
                with open(progress_file, "w") as f:
                    f.write(f"{len(tile_coords)}\n{len(tile_coords)}\n100.0\n")
            except Exception:
                pass
        weight_canvas = weight_canvas.clone()
        weight_canvas[weight_canvas == 0] = 1.0
        output = (canvas / weight_canvas).float()

    if output.shape[0] != N:
        log(
            f"[Rank {rank}] WARNING: Output frames ({output.shape[0]}) != input frames ({N}), adjusting to match",
            "warning",
            rank,
        )
        if output.shape[0] > N:
            output = output[:N]
        else:
            raise RuntimeError(
                f"[Rank {rank}] ERROR: Output frames ({output.shape[0]}) < input frames ({N})."
            )

    log(
        f"[Rank {rank}] Final output: {output.shape[0]} frames (1-to-1 with input)",
        "info",
        rank,
    )
    return output


# ---------- I/O 由 utils.io.inference_io 提供 ----------
inference_io.log = log
inference_io.log = log


def run_single_gpu_inference(args, total_frames: int, input_fps: float, device_id: int):
    """单 GPU 推理入口。I/O、HDR、保存由 utils.io.inference_io 提供。"""
    log(f"[Single-GPU] Starting single GPU inference on GPU {device_id}", "info")
    torch.cuda.set_device(device_id)
    if args.attention_mode == "sparse_sage_attention":
        wan_video_dit.USE_BLOCK_ATTN = False
    else:
        wan_video_dit.USE_BLOCK_ATTN = True
        if not wan_video_dit.BLOCK_ATTN_AVAILABLE:
            log(
                f"[Single-GPU] Warning: block_sparse_attention not available. Auto-switching to sparse_sage_attention",
                "warning",
            )
            wan_video_dit.USE_BLOCK_ATTN = False
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map.get(args.precision, torch.bfloat16)
    model_dir = f"/app/models/v{args.model_ver}"
    pipe, device_obj = init_pipeline_distributed(
        0, 1, args.mode, dtype, model_dir, use_shared_memory=False, device_id=device_id
    )
    device_str = ensure_device_str(device_obj)
    enable_hdr = getattr(args, "_enable_hdr", False)
    checkpoint_dir = args.checkpoint_dir
    streaming_input = getattr(args, "streaming_input", False) and args.tiled_dit
    tone_mapping_params = None
    segment_frames = None
    if streaming_input:
        if total_frames < 21:
            log(
                f"[Single-GPU] ERROR: Video has only {total_frames} frames, minimum is 21. Cannot process.",
                "error",
            )
            raise ValueError(f"Video too short: {total_frames} frames (minimum: 21)")
        tone_mapping_params = inference_io.get_hdr_tone_mapping_params_lightweight(
            args.input, 0, total_frames, args, rank=0, checkpoint_dir=checkpoint_dir
        )
        if enable_hdr and tone_mapping_params is None:
            tone_mapping_params = {}
        log(f"[Single-GPU] Streaming input: per-tile read from disk ({total_frames} frames).", "info")
    else:
        log(f"[Single-GPU] Reading all {total_frames} frames...", "info")
        segment_frames = inference_io.read_input_frames_range(
            args.input, 0, total_frames, fps=input_fps, enable_hdr=enable_hdr
        )[0]
        log(
            f"[Single-GPU] Loaded {segment_frames.shape[0]} frames, shape: {segment_frames.shape}",
            "info",
        )
        segment_frames, tone_mapping_params = inference_io.apply_hdr_tone_mapping_if_needed(
            segment_frames, args, rank=0, checkpoint_dir=checkpoint_dir
        )
        if segment_frames.shape[0] < 21:
            log(
                f"[Single-GPU] ERROR: Video has only {segment_frames.shape[0]} frames, minimum is 21. Cannot process.",
                "error",
            )
            raise ValueError(
                f"Video too short: {segment_frames.shape[0]} frames (minimum: 21)"
            )
    log(
        f"[Single-GPU] Starting inference on {total_frames} frames...",
        "info",
    )
    t0 = time.perf_counter()
    if streaming_input:
        output = run_inference_distributed_segment(
            pipe, None, device_str, dtype, args, rank=0, checkpoint_dir=None,
            input_path=args.input, start_idx=0, end_idx=total_frames,
            input_fps=input_fps, streaming_input=True,
            streaming_hdr_params=tone_mapping_params,
        )
    else:
        output = run_inference_distributed_segment(
            pipe, segment_frames, device_str, dtype, args, rank=0, checkpoint_dir=None
        )
    elapsed_min = (time.perf_counter() - t0) / 60.0
    log(f"[Single-GPU] ✓ Inference done: {output.shape[0]} frames, total {elapsed_min:.1f} min", "info")
    output = inference_io.apply_inverse_hdr_if_needed(
        output, tone_mapping_params, args, rank=0
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    output_file = os.path.join(checkpoint_dir, "rank_0_result.pt")
    log(f"[Single-GPU] Saving result to {output_file}...", "info")
    torch.save(output, output_file)
    log(f"[Single-GPU] ✓ Result saved: {output.shape[0]} frames", "finish")
    done_file = os.path.join(checkpoint_dir, "rank_0_done.flag")
    with open(done_file, "w") as f:
        f.write("rank_0_completed\n")
    log(f"[Single-GPU] Saving final output...", "info")
    output_mode = getattr(args, "output_mode", "video")
    output_path = (
        args.output
        if args.output
        else inference_io.get_default_output_path(args.input, args.scale)
    )
    if output_mode == "pictures":
        output_format = getattr(args, "output_format", "png")
        global_hdr_max = inference_io.get_global_hdr_max(
            output, args, checkpoint_dir, rank_list=[0]
        )
        input_filenames = inference_io.get_input_image_sequence_filenames(
            args.input, enable_hdr=enable_hdr
        )
        frames_saved = inference_io.save_frames_as_sequence(
            output,
            output_path,
            args,
            rank=0,
            start_frame_idx=0,
            global_hdr_max=global_hdr_max,
            input_filenames=input_filenames,
        )
        log(f"[Single-GPU] ✓ Saved {frames_saved} frames to {output_path}", "finish")
    else:
        fps_out = getattr(args, "output_fps", input_fps)
        if getattr(args, "_enable_hdr", False):
            inference_io.save_merged_as_hdr_video(output, output_path, args, fps_out)
        else:
            from utils.io.video_io import save_video

            save_video(output, output_path, fps_out, hdr_mode=False)
        log(f"[Single-GPU] ✓ Video saved: {output_path}", "finish")
    log(f"[Single-GPU] ✓ All steps completed!", "finish")
    return output


def run_distributed_inference(
    rank: int,
    world_size: int,
    args,
    total_frames: int,
    input_fps: float,
    device_id: int = None,
):
    """多 GPU 分布式推理入口。I/O、HDR、合并保存由 utils.io.inference_io 提供。

    流程简述（devices all 时）：
    1) 按时间切段：视频帧按 segment 分给各 GPU（rank 0 拿 [0, N0]，rank 1 拿 [N0-overlap, N1]，…）
    2) 每 GPU 只处理自己的 segment_frames，在空间上再按 tile 切块
    3) tile_batch_size：每个 GPU 上每次可同时推理的 tile 个数（batch 维拼在一起一次 pipe 调用）
    4) 多 GPU + tile_batch_size>1 已通过 DiT 内 B>1 时 KV cache 用重复块填充修复，可正常使用
    """
    if device_id is None:
        device_id = rank
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        world_size=world_size,
        rank=rank,
    )
    try:
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device_id)
        if args.attention_mode == "sparse_sage_attention":
            wan_video_dit.USE_BLOCK_ATTN = False
        else:
            wan_video_dit.USE_BLOCK_ATTN = True
            if not wan_video_dit.BLOCK_ATTN_AVAILABLE:
                log(
                    f"[Rank {rank}] Warning: block_sparse_attention not available. Auto-switching to sparse_sage_attention",
                    "warning",
                    rank,
                )
                wan_video_dit.USE_BLOCK_ATTN = False
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        dtype = dtype_map.get(args.precision, torch.bfloat16)
        model_dir = f"/app/models/v{args.model_ver}"
        pipe, device_obj = init_pipeline_distributed(
            rank,
            world_size,
            args.mode,
            dtype,
            model_dir,
            use_shared_memory=True,
            device_id=device_id,
        )
        device = ensure_device_str(device_obj)
        args.world_size = world_size  # 供 determine_optimal_batch_size 判断多 GPU 时是否限制 tile_batch
        segment_overlap = getattr(args, "segment_overlap", 2)
        segments = inference_io.split_video_by_frames(
            total_frames, world_size, overlap=segment_overlap, force_num_workers=False
        )
        actual_num_segments = len(segments)
        log(
            f"[Rank {rank}] Using GPU cuda:{device_id}, segment: frames {segments[rank][0]}-{segments[rank][1]}",
            "info",
            rank,
        )
        if rank >= actual_num_segments:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            with open(
                os.path.join(args.checkpoint_dir, f"rank_{rank}_done.flag"), "w"
            ) as f:
                f.write(f"rank_{rank}_skipped\n")
            return
        t0 = time.perf_counter()
        start_idx, end_idx = segments[rank]
        enable_hdr = getattr(args, "_enable_hdr", False)
        checkpoint_dir = args.checkpoint_dir
        streaming_input = getattr(args, "streaming_input", False) and args.tiled_dit
        tone_mapping_params = None
        segment_frames = None
        if streaming_input:
            N_seg = end_idx - start_idx
            if N_seg < 21:
                raise ValueError(
                    f"Segment too short: {N_seg} frames (minimum: 21)"
                )
            tone_mapping_params = inference_io.get_hdr_tone_mapping_params_lightweight(
                args.input, start_idx, end_idx, args, rank=rank,
                checkpoint_dir=checkpoint_dir,
            )
            if enable_hdr and tone_mapping_params is None:
                tone_mapping_params = {}
            log(
                f"[Rank {rank}] Streaming input: no full segment load, per-tile read from disk.",
                "info",
                rank,
            )
        else:
            segment_frames = inference_io.read_input_frames_range(
                args.input, start_idx, end_idx, fps=input_fps, enable_hdr=enable_hdr
            )[0]
            segment_frames, tone_mapping_params = (
                inference_io.apply_hdr_tone_mapping_if_needed(
                    segment_frames, args, rank=rank, checkpoint_dir=checkpoint_dir
                )
            )
            if segment_frames.shape[0] < 21:
                raise ValueError(
                    f"Segment too short: {segment_frames.shape[0]} frames (minimum: 21)"
                )
        os.makedirs(checkpoint_dir, exist_ok=True)
        output_file = os.path.join(checkpoint_dir, f"rank_{rank}_result.pt")
        try:
            if streaming_input:
                segment_output = run_inference_distributed_segment(
                    pipe, None, device, dtype, args, rank, checkpoint_dir,
                    input_path=args.input, start_idx=start_idx, end_idx=end_idx,
                    input_fps=input_fps, streaming_input=True,
                    streaming_hdr_params=tone_mapping_params,
                )
            else:
                segment_output = run_inference_distributed_segment(
                    pipe, segment_frames, device, dtype, args, rank, checkpoint_dir
                )
            segment_output = inference_io.apply_inverse_hdr_if_needed(
                segment_output, tone_mapping_params, args, rank=rank
            )
            segment_overlap_val = getattr(args, "segment_overlap", 2)
            if rank > 0 and segment_output.shape[0] > segment_overlap_val:
                segment_output = segment_output[segment_overlap_val:]
            if (
                rank < actual_num_segments - 1
                and segment_output.shape[0] > segment_overlap_val
            ):
                segment_output = segment_output[:-segment_overlap_val]
            torch.save(segment_output, output_file)
            with open(os.path.join(checkpoint_dir, f"rank_{rank}_done.flag"), "w") as f:
                f.write(f"rank_{rank}_completed\n")
        except Exception as e:
            with open(os.path.join(checkpoint_dir, f"rank_{rank}_error.txt"), "w") as f:
                f.write(f"{e}\n{traceback.format_exc()}")
            with open(os.path.join(checkpoint_dir, f"rank_{rank}_done.flag"), "w") as f:
                f.write(f"rank_{rank}_failed\n")
        if rank == 0:
            inference_io.merge_and_save_distributed_results(
                checkpoint_dir, args, world_size, total_frames, input_fps
            )
            elapsed_min = (time.perf_counter() - t0) / 60.0
            log(
                f"[Distributed] ✓ Inference done: {total_frames} frames, total {elapsed_min:.1f} min",
                "info",
                0,
            )
    except Exception as e:
        log(f"[Rank {rank}] FATAL: {e}", "error", rank)
        try:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            with open(
                os.path.join(args.checkpoint_dir, f"rank_{rank}_fatal_error.txt"), "w"
            ) as f:
                f.write(f"{e}\n{traceback.format_exc()}")
        except Exception:
            pass
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def run_with_device(
    rank: int,
    world_size: int,
    args,
    total_frames: int,
    input_fps: float,
    device_indices: List[int],
):
    """包装函数，将设备索引传递给分布式推理（可被 mp.spawn pickle）。"""
    device_id = device_indices[rank] if rank < len(device_indices) else rank
    return run_distributed_inference(
        rank, world_size, args, total_frames, input_fps, device_id=device_id
    )
