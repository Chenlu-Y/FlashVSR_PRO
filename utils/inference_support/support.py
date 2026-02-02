"""
推理公共支持：张量准备、tile 几何、显存估算。
与是否分布式无关，供 infer_video 与 inference_runner 等复用。
"""
import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

# 可选：若在项目内运行，可导入 clean_vram
try:
    from src.models.utils import clean_vram
except ImportError:
    def clean_vram():
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def tensor2video(frames: torch.Tensor):
    """Convert tensor (B,C,F,H,W) to normalized video tensor (F,H,W,C)"""
    video_squeezed = frames.squeeze(0)
    video_permuted = rearrange(video_squeezed, "C F H W -> F H W C")
    video_final = (video_permuted.float() + 1.0) / 2.0
    return video_final


def largest_8n1_leq(n):
    """Return largest (8n+1) less than or equal to n."""
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def compute_scaled_and_target_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128):
    """Compute scaled and target dimensions aligned to multiple (round up to avoid boundary gaps)."""
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid input size")
    sW, sH = w0 * scale, h0 * scale
    tW = max(multiple, int(math.ceil(sW / multiple)) * multiple)
    tH = max(multiple, int(math.ceil(sH / multiple)) * multiple)
    return sW, sH, tW, tH


def tensor_upscale_then_pad(frame_tensor: torch.Tensor, scale: int, tW: int, tH: int):
    """Upscale to (sW, sH) then pad on right/bottom to (tW, tH). Avoids boundary gaps when tW/tH are rounded up."""
    h0, w0, c = frame_tensor.shape
    tensor_bchw = frame_tensor.permute(2, 0, 1).unsqueeze(0)
    sW, sH = w0 * scale, h0 * scale
    upscaled = F.interpolate(tensor_bchw, size=(sH, sW), mode='bicubic', align_corners=False)
    if sW < tW or sH < tH:
        pad_right = max(0, tW - sW)
        pad_bottom = max(0, tH - sH)
        upscaled = F.pad(upscaled, (0, pad_right, 0, pad_bottom), mode='replicate')
    return upscaled.squeeze(0)


def prepare_input_tensor(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16):
    """Prepare video tensor by upscaling and padding.

    返回: (vid_final, tH, tW, F_, N0)
    - F_: 算法处理后的帧数（可能包含补帧）
    - N0: 原始输入帧数（用于后续裁剪，确保输出帧数 = 输入帧数）
    """
    N0, h0, w0, _ = image_tensor.shape
    multiple = 128
    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=multiple)

    num_frames_with_padding = N0 + 4
    F_min = largest_8n1_leq(num_frames_with_padding)
    if F_min == 0:
        raise RuntimeError(f"Not enough frames after padding: {num_frames_with_padding}")

    safety_margin = 8
    if F_min < N0 + safety_margin:
        target_frames = N0 + safety_margin + 4
        F_ = largest_8n1_leq(target_frames)
        if F_ <= F_min:
            F_ = largest_8n1_leq(target_frames + 8)
    else:
        F_ = F_min

    frames = []
    for i in range(F_):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_chw = tensor_upscale_then_pad(frame_slice, scale, tW, tH)
        tensor_out = tensor_chw * 2.0 - 1.0
        tensor_out = tensor_out.to('cpu').to(dtype)
        frames.append(tensor_out)

    vid_stacked = torch.stack(frames, 0)
    vid_final = vid_stacked.permute(1, 0, 2, 3).unsqueeze(0)

    del vid_stacked
    clean_vram()

    return vid_final, tH, tW, F_, N0


def get_gpu_memory_info(device) -> Tuple[float, float]:
    """Get GPU memory info (used, total) in GB. On error returns (0.0, 0.0)."""
    if isinstance(device, torch.device):
        device = str(device)
    if not (isinstance(device, str) and device.startswith("cuda:")):
        return 0.0, 0.0
    try:
        idx = int(device.split(":")[1])
        torch.cuda.set_device(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(idx) / (1024 ** 3)
        return float(reserved), float(total)
    except Exception:
        return 0.0, 0.0


def get_available_memory_gb(device) -> float:
    """Get available GPU memory in GB."""
    used, total = get_gpu_memory_info(device)
    return total - used


def estimate_tile_memory(tile_size: int, num_frames: int, scale: int, dtype_size: int = 2) -> float:
    """Estimate memory needed for processing one tile in GB."""
    input_size = tile_size * tile_size * num_frames * 3 * dtype_size / (1024 ** 3)
    output_size = (tile_size * scale) * (tile_size * scale) * num_frames * 3 * dtype_size / (1024 ** 3)
    intermediate_size = input_size * 6
    overhead = 0.5
    return input_size + intermediate_size + output_size + overhead


def calculate_tile_coords(height, width, tile_size, overlap, shift_x=0, shift_y=0):
    """Calculate tile coordinates for patch-based inference.

    When shift_x/shift_y are set (e.g. half stride for tile shift), the grid is
    offset so the second pass covers the same area with a shifted pattern.
    """
    coords = []
    stride = tile_size - overlap
    num_rows = math.ceil((height - overlap) / stride)
    num_cols = math.ceil((width - overlap) / stride)
    # With shift, we may need one extra row/col to cover boundaries
    if shift_y:
        num_rows += 1
    if shift_x:
        num_cols += 1
    for r in range(num_rows):
        for c in range(num_cols):
            y1 = shift_y + r * stride
            x1 = shift_x + c * stride
            y2 = min(y1 + tile_size, height)
            x2 = min(x1 + tile_size, width)
            if y1 >= height or x1 >= width:
                continue
            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)
            if y1 < 0 or x1 < 0:
                continue
            coords.append((x1, y1, x2, y2))
    return coords


def create_feather_mask(size, overlap):
    """Create blending mask for overlapping tiles with Gaussian blur."""
    H, W = size
    mask = torch.ones(1, 1, H, W, dtype=torch.float32)

    ramp = torch.linspace(0, 1, overlap, dtype=torch.float32)
    sigma = max(1.0, overlap / 3.0)
    kernel_size = int(2 * sigma * 2) + 1
    if kernel_size > 1 and kernel_size <= overlap:
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        ramp_padded = F.pad(ramp.unsqueeze(0).unsqueeze(0), (kernel_size // 2, kernel_size // 2), mode='reflect')
        ramp_blurred = F.conv1d(ramp_padded, gaussian_1d.unsqueeze(0).unsqueeze(0), padding=0)
        ramp = ramp_blurred.squeeze()
        if ramp.max() > ramp.min():
            ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min())

    ramp_h = ramp.view(1, 1, -1, 1)
    ramp_w = ramp.view(1, 1, 1, -1)

    mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp_w)
    mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp_w.flip(-1))
    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp_h)
    mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp_h.flip(-2))

    return mask
