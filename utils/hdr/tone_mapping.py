#!/usr/bin/env python3
"""
HDR Tone Mapping 工具模块

实现：
1. Tone Mapping (HDR → SDR): 将 HDR 线性值压缩到 [0, 1]
2. Inverse Tone Mapping (SDR → HDR): 将 SDR 还原回 HDR

支持的算法：
- Reinhard (简单可逆)
- Logarithmic (对数映射，可逆)
- ACES (行业标准，近似可逆)
"""

import torch
import numpy as np
from typing import Tuple, Optional, Literal


def reinhard_tone_map(hdr: torch.Tensor, exposure: float = 1.0, white_point: Optional[float] = None) -> Tuple[torch.Tensor, dict]:
    """Reinhard Tone Mapping: HDR → SDR
    
    Args:
        hdr: HDR 图像 tensor，值可能 > 1
        exposure: 曝光调整（默认 1.0）
        white_point: 白点值（None 时自动计算）
    
    Returns:
        (sdr, params): SDR 图像 [0, 1] 和 tone mapping 参数（用于逆映射）
    """
    # 应用曝光
    ldr = hdr * exposure
    
    # 计算白点（如果未指定）
    if white_point is None:
        # 使用 99.9% 分位数作为白点（避免极端值影响）
        flat = ldr.flatten()
        white_point = torch.quantile(flat, 0.999).item()
        if white_point < 1.0:
            white_point = ldr.max().item()
    
    # Reinhard 公式: L_out = L_in / (1 + L_in / L_white)
    # 为了可逆性，使用: L_out = L_in / (1 + L_in)
    # 然后归一化到 [0, 1]
    ldr_normalized = ldr / (1.0 + ldr)
    
    # 归一化到 [0, 1]（基于白点）
    if white_point > 0:
        ldr_normalized = ldr_normalized / (white_point / (1.0 + white_point))
    
    sdr = torch.clamp(ldr_normalized, 0.0, 1.0)
    
    # 保存参数用于逆映射
    params = {
        'method': 'reinhard',
        'exposure': exposure,
        'white_point': white_point,
        'max_hdr': hdr.max().item(),
    }
    
    return sdr, params


def reinhard_inverse_tone_map(sdr: torch.Tensor, params: dict) -> torch.Tensor:
    """Reinhard Inverse Tone Mapping: SDR → HDR
    
    Args:
        sdr: SDR 图像 tensor [0, 1]
        params: tone mapping 时保存的参数
    
    Returns:
        hdr: 还原的 HDR 图像
    """
    method = params.get('method', 'reinhard')
    if method != 'reinhard':
        raise ValueError(f"参数方法 {method} 与 reinhard_inverse_tone_map 不匹配")
    
    exposure = params.get('exposure', 1.0)
    white_point = params.get('white_point', 1.0)
    max_hdr = params.get('max_hdr', 1.0)
    
    # 反归一化
    if white_point > 0:
        ldr_normalized = sdr * (white_point / (1.0 + white_point))
    else:
        ldr_normalized = sdr
    
    # 逆 Reinhard: L_in = L_out / (1 - L_out)
    # 注意：需要避免除零
    ldr = ldr_normalized / torch.clamp(1.0 - ldr_normalized, min=1e-6)
    
    # 反曝光
    hdr = ldr / exposure
    
    # 限制到原始最大值（避免过度还原）
    hdr = torch.clamp(hdr, 0.0, max_hdr * 1.1)  # 允许 10% 的余量
    
    return hdr


def logarithmic_tone_map(hdr: torch.Tensor, exposure: float = 1.0) -> Tuple[torch.Tensor, dict]:
    """对数 Tone Mapping: HDR → SDR
    
    Args:
        hdr: HDR 图像 tensor
        exposure: 曝光调整
    
    Returns:
        (sdr, params): SDR 图像 [0, 1] 和参数
    """
    ldr = hdr * exposure
    l_max = ldr.max().item()
    
    if l_max <= 0:
        return torch.zeros_like(hdr), {'method': 'logarithmic', 'exposure': exposure, 'l_max': 0.0, 'max_hdr': hdr.max().item()}
    
    # 对数映射: L_out = log(1 + L_in) / log(1 + L_max)
    sdr = torch.log1p(ldr) / np.log1p(l_max)
    sdr = torch.clamp(sdr, 0.0, 1.0)
    
    params = {
        'method': 'logarithmic',
        'exposure': exposure,
        'l_max': l_max,
        'max_hdr': hdr.max().item(),
    }
    
    return sdr, params


def logarithmic_inverse_tone_map(sdr: torch.Tensor, params: dict) -> torch.Tensor:
    """对数 Inverse Tone Mapping: SDR → HDR
    
    改进：
    1. 使用 global_l_max 或 l_max 进行逆映射（优先使用 global_l_max）
    2. 移除过度的 clamp 限制，允许超分模型可能产生的更高值
    3. 使用 global_max_hdr 作为参考上限（而非硬性限制）
    """
    method = params.get('method', 'logarithmic')
    if method != 'logarithmic':
        raise ValueError(f"参数方法 {method} 与 logarithmic_inverse_tone_map 不匹配")
    
    exposure = params.get('exposure', 1.0)
    # 优先使用 global_l_max（全局参数），确保帧间一致性
    l_max = params.get('global_l_max', params.get('l_max', 1.0))
    max_hdr = params.get('global_max_hdr', params.get('max_hdr', 1.0))
    
    if l_max <= 0:
        return torch.zeros_like(sdr)
    
    # 逆对数映射: L_in = exp(L_out * log(1 + L_max)) - 1
    ldr = torch.expm1(sdr * np.log1p(l_max))
    
    # 反曝光
    hdr = ldr / exposure
    
    # 改进：使用更宽松的上限（允许超分模型可能产生的更高值）
    # 不再硬性 clamp 到 max_hdr * 1.1，而是使用更大的余量
    # 只限制极端值（避免 inf/nan）
    hdr = torch.clamp(hdr, 0.0, max_hdr * 2.0)  # 允许 100% 的余量
    
    return hdr


def aces_tone_map(hdr: torch.Tensor, exposure: float = 1.0) -> Tuple[torch.Tensor, dict]:
    """ACES Filmic Tone Mapping (近似): HDR → SDR
    
    注意：ACES 不完全可逆，这里使用近似逆映射
    """
    ldr = hdr * exposure
    
    # ACES Filmic (简化版)
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    
    # ACES: (x * (a * x + b)) / (x * (c * x + d) + e)
    x = ldr
    sdr = (x * (a * x + b)) / (x * (c * x + d) + e)
    sdr = torch.clamp(sdr, 0.0, 1.0)
    
    params = {
        'method': 'aces',
        'exposure': exposure,
        'max_hdr': hdr.max().item(),
    }
    
    return sdr, params


def aces_inverse_tone_map(sdr: torch.Tensor, params: dict) -> torch.Tensor:
    """ACES Inverse Tone Mapping (近似)"""
    # ACES 的逆映射比较复杂，这里使用数值方法近似
    # 或者使用查找表
    # 简化：使用对数映射作为近似
    exposure = params.get('exposure', 1.0)
    max_hdr = params.get('max_hdr', 1.0)
    
    # 使用对数逆映射作为近似（ACES 不完全可逆）
    l_max = max_hdr * exposure
    if l_max <= 0:
        return torch.zeros_like(sdr)
    
    ldr = torch.expm1(sdr * np.log1p(l_max))
    hdr = ldr / exposure
    hdr = torch.clamp(hdr, 0.0, max_hdr * 1.1)
    
    return hdr


def tone_map_hdr_to_sdr(
    hdr: torch.Tensor,
    method: Literal['reinhard', 'logarithmic', 'aces'] = 'reinhard',
    exposure: float = 1.0,
    white_point: Optional[float] = None
) -> Tuple[torch.Tensor, dict]:
    """统一的 Tone Mapping 接口: HDR → SDR
    
    Args:
        hdr: HDR 图像 tensor (N, H, W, C) 或 (H, W, C)，值可能 > 1
        method: tone mapping 方法
        exposure: 曝光调整
        white_point: 白点值（仅用于 reinhard）
    
    Returns:
        (sdr, params): SDR 图像 [0, 1] 和 tone mapping 参数
    """
    if method == 'reinhard':
        return reinhard_tone_map(hdr, exposure, white_point)
    elif method == 'logarithmic':
        return logarithmic_tone_map(hdr, exposure)
    elif method == 'aces':
        return aces_tone_map(hdr, exposure)
    else:
        raise ValueError(f"未知的 tone mapping 方法: {method}")


def inverse_tone_map_sdr_to_hdr(sdr: torch.Tensor, params: dict) -> torch.Tensor:
    """统一的 Inverse Tone Mapping 接口: SDR → HDR
    
    Args:
        sdr: SDR 图像 tensor [0, 1]
        params: tone mapping 时保存的参数
    
    Returns:
        hdr: 还原的 HDR 图像
    """
    method = params.get('method', 'reinhard')
    
    if method == 'reinhard':
        return reinhard_inverse_tone_map(sdr, params)
    elif method == 'logarithmic':
        return logarithmic_inverse_tone_map(sdr, params)
    elif method == 'aces':
        return aces_inverse_tone_map(sdr, params)
    else:
        raise ValueError(f"未知的逆映射方法: {method}")


def detect_hdr_range(tensor: torch.Tensor, threshold: float = 1.01) -> bool:
    """检测 tensor 是否包含 HDR 值（> threshold）
    
    Args:
        tensor: 图像 tensor
        threshold: HDR 阈值（默认 1.01，略大于 1.0 以避免浮点误差）
    
    Returns:
        bool: 如果包含 > threshold 的值，返回 True
    """
    return (tensor > threshold).any().item()


def apply_tone_mapping_to_frames(
    frames: torch.Tensor,
    method: Literal['reinhard', 'logarithmic', 'aces'] = 'reinhard',
    exposure: float = 1.0,
    per_frame: bool = True,
    global_l_max: Optional[float] = None
) -> Tuple[torch.Tensor, list]:
    """对帧序列应用 Tone Mapping
    
    Args:
        frames: 帧 tensor (N, H, W, C)
        method: tone mapping 方法
        exposure: 曝光调整
        per_frame: 如果 True，每帧独立计算参数；如果 False，使用全局参数
                   注意：per_frame=True 可能导致帧间亮度不一致（频闪），建议设为 False
        global_l_max: 全局最大值（如果提供，将覆盖自动计算的值）
                      强烈建议提供此参数以确保帧间一致性
    
    Returns:
        (sdr_frames, params_list): SDR 帧和参数列表
    """
    N = frames.shape[0]
    sdr_frames = []
    params_list = []
    
    # 计算全局最大值（用于全局模式或作为参考）
    if global_l_max is None:
        global_max = frames.max().item()
    else:
        global_max = global_l_max
    
    if per_frame and global_l_max is None:
        # 每帧独立处理（不推荐，可能导致频闪）
        # 注意：这个警告只在用户明确指定 --per_frame_tone_mapping 时才会触发
        for i in range(N):
            frame = frames[i]  # (H, W, C)
            sdr_frame, params = tone_map_hdr_to_sdr(frame, method, exposure)
            sdr_frames.append(sdr_frame)
            params_list.append(params)
    else:
        # 全局处理（使用所有帧的最大值，避免频闪）
        # 对于 logarithmic 方法，需要使用全局 l_max
        white_point = global_max * exposure if method == 'reinhard' else None
        
        for i in range(N):
            frame = frames[i]
            if method == 'logarithmic':
                # 使用全局 l_max 的对数映射
                sdr_frame, params = _logarithmic_tone_map_with_global_lmax(
                    frame, exposure, global_max * exposure
                )
            elif method == 'reinhard':
                sdr_frame, params = tone_map_hdr_to_sdr(frame, method, exposure, white_point)
            else:
                sdr_frame, params = tone_map_hdr_to_sdr(frame, method, exposure)
            
            # 确保记录全局最大值
            params['global_l_max'] = global_max * exposure
            params['global_max_hdr'] = global_max
            sdr_frames.append(sdr_frame)
            params_list.append(params)
    
    return torch.stack(sdr_frames, dim=0), params_list


def _logarithmic_tone_map_with_global_lmax(
    hdr: torch.Tensor, 
    exposure: float, 
    global_l_max: float
) -> Tuple[torch.Tensor, dict]:
    """使用全局 l_max 的对数 Tone Mapping（避免帧间不一致）"""
    ldr = hdr * exposure
    
    if global_l_max <= 0:
        return torch.zeros_like(hdr), {
            'method': 'logarithmic', 
            'exposure': exposure, 
            'l_max': 0.0, 
            'max_hdr': hdr.max().item(),
            'frame_max': hdr.max().item()
        }
    
    # 使用全局 l_max 进行对数映射（关键：避免每帧参数不同导致频闪）
    sdr = torch.log1p(ldr) / np.log1p(global_l_max)
    sdr = torch.clamp(sdr, 0.0, 1.0)
    
    params = {
        'method': 'logarithmic',
        'exposure': exposure,
        'l_max': global_l_max,  # 使用全局 l_max
        'max_hdr': hdr.max().item(),  # 保留原始帧的最大值
        'frame_max': hdr.max().item(),  # 帧级最大值（用于调试）
    }
    
    return sdr, params


def apply_inverse_tone_mapping_to_frames(
    sdr_frames: torch.Tensor,
    params_list: list
) -> torch.Tensor:
    """对帧序列应用 Inverse Tone Mapping
    
    Args:
        sdr_frames: SDR 帧 tensor (N, H, W, C)
        params_list: tone mapping 参数列表（每帧一个）
    
    Returns:
        hdr_frames: 还原的 HDR 帧
    """
    N = sdr_frames.shape[0]
    hdr_frames = []
    
    for i in range(N):
        frame = sdr_frames[i]
        params = params_list[i]
        hdr_frame = inverse_tone_map_sdr_to_hdr(frame, params)
        hdr_frames.append(hdr_frame)
    
    return torch.stack(hdr_frames, dim=0)


def serialize_tone_mapping_params(params_list: list) -> list:
    """序列化 tone mapping 参数（将 torch/numpy 类型转换为 Python 原生类型）
    
    Args:
        params_list: tone mapping 参数列表
    
    Returns:
        serialized: 可 JSON 序列化的参数列表
    """
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
    """反序列化 tone mapping 参数
    
    Args:
        serialized_list: JSON 序列化后的参数列表
    
    Returns:
        params_list: 参数列表（dict 格式，可直接用于 inverse_tone_map_sdr_to_hdr）
    """
    # 参数已经是 Python 原生类型，直接返回即可
    return serialized_list
