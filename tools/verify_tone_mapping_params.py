#!/usr/bin/env python3
"""
验证 Tone Mapping 参数保存和还原的正确性

检查：
1. 参数文件是否存在
2. 参数内容是否完整
3. 参数是否足以进行逆映射
"""

import os
import sys
import json
import argparse

# 添加项目根目录到 sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def verify_params_file(params_file: str) -> bool:
    """验证参数文件的内容是否完整"""
    if not os.path.exists(params_file):
        print(f"❌ 参数文件不存在: {params_file}")
        return False
    
    print(f"✓ 参数文件存在: {params_file}")
    
    try:
        with open(params_file, 'r') as f:
            params_list = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取参数文件: {e}")
        return False
    
    if not isinstance(params_list, list):
        print(f"❌ 参数文件格式错误: 应该是列表，实际是 {type(params_list)}")
        return False
    
    if len(params_list) == 0:
        print(f"❌ 参数列表为空")
        return False
    
    print(f"✓ 参数列表包含 {len(params_list)} 帧的参数")
    
    # 检查每帧的参数
    required_keys = {
        'logarithmic': ['method', 'exposure', 'l_max', 'max_hdr'],
        'reinhard': ['method', 'exposure', 'white_point', 'max_hdr'],
        'aces': ['method', 'exposure', 'max_hdr']
    }
    
    all_valid = True
    method = None
    
    for i, params in enumerate(params_list):
        if not isinstance(params, dict):
            print(f"❌ 第 {i} 帧参数不是字典格式")
            all_valid = False
            continue
        
        # 检查方法
        if 'method' not in params:
            print(f"❌ 第 {i} 帧缺少 'method' 字段")
            all_valid = False
            continue
        
        method = params['method']
        if method not in required_keys:
            print(f"❌ 第 {i} 帧使用了未知的方法: {method}")
            all_valid = False
            continue
        
        # 检查必需字段
        required = required_keys[method]
        missing = [key for key in required if key not in params]
        if missing:
            print(f"❌ 第 {i} 帧缺少必需字段: {missing}")
            all_valid = False
            continue
        
        # 检查值是否合理
        if 'exposure' in params:
            if params['exposure'] <= 0:
                print(f"⚠️  第 {i} 帧的 exposure 值异常: {params['exposure']}")
        
        if 'l_max' in params:
            if params['l_max'] <= 0:
                print(f"⚠️  第 {i} 帧的 l_max 值异常: {params['l_max']}")
        
        if 'max_hdr' in params:
            if params['max_hdr'] <= 0:
                print(f"⚠️  第 {i} 帧的 max_hdr 值异常: {params['max_hdr']}")
            elif params['max_hdr'] < 1.0:
                print(f"⚠️  第 {i} 帧的 max_hdr < 1.0，可能不是 HDR: {params['max_hdr']}")
    
    if all_valid:
        print(f"✓ 所有参数格式正确")
        print(f"✓ 使用方法: {method}")
        
        # 显示统计信息
        if method == 'logarithmic':
            exposures = [p.get('exposure', 1.0) for p in params_list]
            l_maxs = [p.get('l_max', 0.0) for p in params_list]
            max_hdrs = [p.get('max_hdr', 0.0) for p in params_list]
            
            print(f"\n统计信息:")
            print(f"  - Exposure: 范围 [{min(exposures):.4f}, {max(exposures):.4f}]")
            print(f"  - l_max: 范围 [{min(l_maxs):.4f}, {max(l_maxs):.4f}]")
            print(f"  - max_hdr: 范围 [{min(max_hdrs):.4f}, {max(max_hdrs):.4f}]")
            print(f"  - 全局最大 HDR 值: {max(max_hdrs):.4f}")
    
    return all_valid


def main():
    parser = argparse.ArgumentParser(description="验证 Tone Mapping 参数文件")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Checkpoint 目录（包含 rank_*_tone_mapping_params.json 文件）")
    parser.add_argument("--rank", type=int, default=None,
                        help="检查特定 rank 的参数（如果不指定，检查所有 rank）")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_dir):
        print(f"❌ Checkpoint 目录不存在: {args.checkpoint_dir}")
        sys.exit(1)
    
    print(f"检查目录: {args.checkpoint_dir}\n")
    
    if args.rank is not None:
        # 检查特定 rank
        params_file = os.path.join(args.checkpoint_dir, f"rank_{args.rank}_tone_mapping_params.json")
        print(f"检查 Rank {args.rank} 的参数...")
        success = verify_params_file(params_file)
        sys.exit(0 if success else 1)
    else:
        # 检查所有 rank
        all_success = True
        found_any = False
        
        for rank in range(8):  # 假设最多 8 个 rank
            params_file = os.path.join(args.checkpoint_dir, f"rank_{rank}_tone_mapping_params.json")
            if os.path.exists(params_file):
                found_any = True
                print(f"\n{'='*60}")
                print(f"检查 Rank {rank} 的参数...")
                print(f"{'='*60}")
                success = verify_params_file(params_file)
                if not success:
                    all_success = False
                print()
        
        if not found_any:
            print(f"❌ 未找到任何参数文件在目录: {args.checkpoint_dir}")
            print(f"   期望的文件名格式: rank_*_tone_mapping_params.json")
            sys.exit(1)
        
        if all_success:
            print(f"\n{'='*60}")
            print(f"✓ 所有参数文件验证通过！")
            print(f"{'='*60}")
            sys.exit(0)
        else:
            print(f"\n{'='*60}")
            print(f"❌ 部分参数文件验证失败")
            print(f"{'='*60}")
            sys.exit(1)


if __name__ == "__main__":
    main()
