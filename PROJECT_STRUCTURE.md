# 项目结构重组说明

## 新的目录结构

```
FlashVSR_Ultra_Fast/
├── README.md, README_zh.md, LICENSE, requirements.txt  # 根目录文档
│
├── scripts/                    # 主要推理脚本（入口点）
│   ├── infer_video.py
│   └── infer_video_distributed.py
│
├── tools/                      # 工具脚本
│   ├── batch_process.py
│   ├── find_unmerged.py
│   ├── recover_distributed_inference.py
│   └── recover_all_segments.py
│
├── utils/                      # 工具函数模块
│   ├── __init__.py
│   ├── io/                     # 输入输出相关
│   │   ├── __init__.py
│   │   ├── video_io.py         # 原 save_recovered_video.py
│   │   └── hdr_io.py           # 原 read_hdr_input.py
│   └── hdr/                    # HDR 相关工具
│       ├── __init__.py
│       └── tone_mapping.py     # 原 hdr_tone_mapping.py
│
├── tests/                      # 测试文件
│   ├── test_hdr_integration.py
│   ├── test_hdr_support.py
│   └── test_tone_mapping.py
│
├── docs/                       # 文档
│   ├── HDR_USAGE.md
│   ├── HDR_NORMALIZATION_EXPLANATION.md
│   └── hdr_integration_plan.md
│
├── data/                       # 数据文件
│   └── posi_prompt.pth
│
└── src/                        # 核心代码（保持不变）
    ├── models/
    ├── pipelines/
    ├── schedulers/
    └── ...
```

## 文件移动清单

### 已移动的文件

1. **主要推理脚本** → `scripts/`
   - `infer_video.py` → `scripts/infer_video.py`
   - `infer_video_distributed.py` → `scripts/infer_video_distributed.py`

2. **工具脚本** → `tools/`
   - `batch_process.py` → `tools/batch_process.py`
   - `find_unmerged.py` → `tools/find_unmerged.py`
   - `recover_distributed_inference.py` → `tools/recover_distributed_inference.py`
   - `recover_all_segments.py` → `tools/recover_all_segments.py`

3. **HDR 相关** → `utils/hdr/` 和 `utils/io/`
   - `hdr_tone_mapping.py` → `utils/hdr/tone_mapping.py`
   - `read_hdr_input.py` → `utils/io/hdr_io.py`
   - `save_recovered_video.py` → `utils/io/video_io.py`

4. **测试文件** → `tests/`
   - `test_hdr_integration.py` → `tests/test_hdr_integration.py`
   - `test_hdr_support.py` → `tests/test_hdr_support.py`
   - `test_tone_mapping.py` → `tests/test_tone_mapping.py`

5. **文档** → `docs/`
   - `HDR_USAGE.md` → `docs/HDR_USAGE.md`
   - `HDR_NORMALIZATION_EXPLANATION.md` → `docs/HDR_NORMALIZATION_EXPLANATION.md`
   - `hdr_integration_plan.md` → `docs/hdr_integration_plan.md`

6. **数据文件** → `data/`
   - `posi_prompt.pth` → `data/posi_prompt.pth`

## 更新的导入路径

所有文件中的导入路径已更新：

### 旧导入 → 新导入

```python
# HDR Tone Mapping
from hdr_tone_mapping import ... 
→ from utils.hdr.tone_mapping import ...

# HDR 输入读取
from read_hdr_input import ...
→ from utils.io.hdr_io import ...

# 视频保存
from save_recovered_video import ...
→ from utils.io.video_io import ...
```

## 使用方法更新

### 旧方式
```bash
python infer_video_distributed.py --input video.mp4 --output output.mp4
python recover_distributed_inference.py --checkpoint_dir ...
```

### 新方式
```bash
python scripts/infer_video_distributed.py --input video.mp4 --output output.mp4
python tools/recover_distributed_inference.py --checkpoint_dir ...
```

## 优势

1. **清晰的模块划分**：scripts/、tools/、utils/、tests/、docs/ 各司其职
2. **易于维护**：相关文件集中管理
3. **符合 Python 项目规范**：使用 utils/ 模块化工具函数
4. **便于扩展**：新功能可以轻松添加到对应目录
5. **保持兼容性**：核心代码（src/）保持不变

## 注意事项

- 所有脚本已自动添加项目根目录到 `sys.path`，确保可以正确导入 `utils` 模块
- 测试文件已更新 `sys.path` 设置，指向项目根目录
- Shell 脚本（.sh）保留在根目录，便于直接执行
