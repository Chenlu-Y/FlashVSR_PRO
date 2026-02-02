"""推理公共支持：张量准备、tile 几何、显存估算。"""
from .support import (
    tensor2video,
    largest_8n1_leq,
    compute_scaled_and_target_dims,
    tensor_upscale_then_pad,
    prepare_input_tensor,
    get_gpu_memory_info,
    get_available_memory_gb,
    estimate_tile_memory,
    calculate_tile_coords,
    create_feather_mask,
)

__all__ = [
    "tensor2video",
    "largest_8n1_leq",
    "compute_scaled_and_target_dims",
    "tensor_upscale_then_pad",
    "prepare_input_tensor",
    "get_gpu_memory_info",
    "get_available_memory_gb",
    "estimate_tile_memory",
    "calculate_tile_coords",
    "create_feather_mask",
]
