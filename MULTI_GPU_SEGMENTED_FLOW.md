# multi_gpu + segmented 模式下的视频分割和存储流程

## 概述
当同时使用 `--multi_gpu` 和 `--segmented` 时，视频会经过两层分割：
1. **第一层（multi_gpu）**：按GPU数量分割成多个worker segments
2. **第二层（segmented）**：每个worker内部再分割成多个sub-segments

## 详细流程

### 1. 第一层分割：multi_gpu模式

#### 分割逻辑
- 函数：`split_video_by_frames(frames, num_gpus, overlap=segment_overlap)`
- 计算方式：
  ```python
  segment_size = N // num_gpus  # N是总帧数
  for i in range(num_gpus):
      start_idx = max(0, i * segment_size - overlap if i > 0 else 0)
      end_idx = min(N, (i + 1) * segment_size + overlap if i < num_gpus - 1 else N)
  ```

#### 示例（612帧，2个GPU，overlap=2）
- **Segment 0 (Worker 0)**: frames 0-308 (共308帧)
- **Segment 1 (Worker 1)**: frames 304-612 (共308帧)
  - 注意：有4帧overlap (308-304=4)

### 2. 目录结构和文件命名

#### 2.1 主目录名
- 函数：`get_video_based_dir_name(input_path, scale)`
- 格式：`{视频名}_{scale}x`
- 示例：`3D_cat_1080_30fps_4x`

#### 2.2 multi_gpu checkpoint
- **路径**：`/tmp/flashvsr_checkpoints/{video_dir_name}/`
- **文件**：`checkpoint.json`
- **内容**：
  ```json
  {
    "segment_0_0_308": {
      "start_idx": 0,
      "end_idx": 308,
      "path": "/tmp/flashvsr_multigpu/3D_cat_1080_30fps_4x/worker_0_{uuid}.pt"
    },
    "segment_1_304_612": {
      "start_idx": 304,
      "end_idx": 612,
      "path": "/tmp/flashvsr_multigpu/3D_cat_1080_30fps_4x/worker_1_{uuid}.pt"
    }
  }
  ```

#### 2.3 multi_gpu worker输出
- **路径**：`/tmp/flashvsr_multigpu/{video_dir_name}/`
- **文件命名**：`worker_{worker_id}_{uuid}.pt`
  - `worker_id`: 0, 1, 2, ... (对应segment索引)
  - `uuid`: 随机UUID，避免文件名冲突
- **示例**：
  - `/tmp/flashvsr_multigpu/3D_cat_1080_30fps_4x/worker_0_6d001b10027a4a5d94292b1f1ff42311.pt`
  - `/tmp/flashvsr_multigpu/3D_cat_1080_30fps_4x/worker_1_75c1f91edc7e4a2b9d6c0582c0ab22c0.pt`

### 3. 第二层分割：segmented模式（在worker内部）

#### 3.1 如果worker启用了segmented模式
每个worker进程会：
1. 接收分配给它的frames（例如Worker 0接收frames 0-308）
2. 在worker内部，再次分割成多个sub-segments
3. 每个sub-segment独立处理并保存

#### 3.2 segmented目录结构
- **路径**：`/tmp/flashvsr_segments/{video_dir_name}/`
- **video_dir_name的确定**：
  - 如果在worker模式下：`worker_{worker_start_idx}_{worker_end_idx}_{scale}x`
    - 示例：`worker_0_308_4x` (Worker 0处理frames 0-308)
  - 如果不在worker模式：使用`get_video_based_dir_name(input_path, scale)`

#### 3.3 segmented文件命名
- **.pt文件**：`segment_{seg_idx:04d}.pt`
  - `seg_idx`: 0, 1, 2, ... (sub-segment索引，从0开始)
  - 示例：`segment_0000.pt`, `segment_0001.pt`, ...
- **.json元数据文件**：`segment_{seg_idx:04d}.json`
  - 记录绝对帧范围（相对于原始视频）
  - 内容示例：
    ```json
    {
      "seg_idx": 0,
      "start_frame": 0,        // 绝对帧范围（相对于原始视频）
      "end_frame": 100,
      "relative_start_frame": 0,  // 相对帧范围（相对于worker接收的frames）
      "relative_end_frame": 100,
      "actual_frames": 98,
      "segment_file": "/tmp/flashvsr_segments/worker_0_308_4x/segment_0000.pt",
      "is_worker_mode": true,
      "worker_start_idx": 0,
      "worker_end_idx": 308
    }
    ```

### 4. 完整示例流程

假设：视频612帧，2个GPU，启用segmented，每个sub-segment最大100帧

#### 步骤1：multi_gpu分割
```
原始视频: 612帧
├── Worker 0: frames 0-308 (308帧)
└── Worker 1: frames 304-612 (308帧)
```

#### 步骤2：Worker 0内部segmented分割
```
Worker 0接收: 308帧
├── Sub-segment 0: frames 0-100 (相对于worker: 0-100, 绝对: 0-100)
├── Sub-segment 1: frames 98-200 (相对于worker: 98-200, 绝对: 98-200)
├── Sub-segment 2: frames 198-300 (相对于worker: 198-300, 绝对: 198-300)
└── Sub-segment 3: frames 298-308 (相对于worker: 298-308, 绝对: 298-308)

保存位置: /tmp/flashvsr_segments/worker_0_308_4x/
├── segment_0000.pt + segment_0000.json
├── segment_0001.pt + segment_0001.json
├── segment_0002.pt + segment_0002.json
└── segment_0003.pt + segment_0003.json
```

#### 步骤3：Worker 0合并sub-segments
```
Worker 0处理完所有sub-segments后：
1. 按seg_idx顺序加载所有sub-segments
2. 处理overlap（跳过重复帧）
3. 合并成最终输出
4. 保存到: /tmp/flashvsr_multigpu/3D_cat_1080_30fps_4x/worker_0_{uuid}.pt
```

#### 步骤4：Worker 1内部segmented分割
```
Worker 1接收: 308帧 (frames 304-612)
├── Sub-segment 0: frames 0-100 (相对于worker: 0-100, 绝对: 304-404)
├── Sub-segment 1: frames 98-200 (相对于worker: 98-200, 绝对: 402-504)
├── Sub-segment 2: frames 198-300 (相对于worker: 198-300, 绝对: 502-604)
└── Sub-segment 3: frames 298-308 (相对于worker: 298-308, 绝对: 602-612)

保存位置: /tmp/flashvsr_segments/worker_304_612_4x/
├── segment_0000.pt + segment_0000.json
├── segment_0001.pt + segment_0001.json
├── segment_0002.pt + segment_0002.json
└── segment_0003.pt + segment_0003.json
```

#### 步骤5：Worker 1合并sub-segments
```
Worker 1处理完所有sub-segments后：
1. 按seg_idx顺序加载所有sub-segments
2. 处理overlap（跳过重复帧）
3. 合并成最终输出
4. 保存到: /tmp/flashvsr_multigpu/3D_cat_1080_30fps_4x/worker_1_{uuid}.pt
```

#### 步骤6：主进程合并所有workers
```
主进程：
1. 从checkpoint.json读取所有worker信息
2. 按start_idx排序
3. 加载每个worker的输出文件
4. 处理overlap（Worker 1跳过前4帧）
5. 合并成最终视频
```

### 5. 关键点总结

1. **目录命名规则**：
   - multi_gpu: `/tmp/flashvsr_multigpu/{video_dir_name}/`
   - segmented (worker模式): `/tmp/flashvsr_segments/worker_{start}_{end}_{scale}x/`
   - segmented (非worker模式): `/tmp/flashvsr_segments/{video_dir_name}/`
   - checkpoint: `/tmp/flashvsr_checkpoints/{video_dir_name}/`

2. **文件命名规则**：
   - worker输出: `worker_{worker_id}_{uuid}.pt`
   - sub-segment: `segment_{seg_idx:04d}.pt` + `segment_{seg_idx:04d}.json`

3. **帧范围记录**：
   - checkpoint.json: 记录worker的绝对帧范围
   - segment_*.json: 记录sub-segment的绝对帧范围（相对于原始视频）

4. **Overlap处理**：
   - multi_gpu层：worker之间有overlap（例如4帧）
   - segmented层：sub-segment之间有overlap（例如2帧）
   - 合并时都会跳过overlap部分

5. **断点续传**：
   - multi_gpu: 检查`/tmp/flashvsr_checkpoints/{video_dir_name}/checkpoint.json`
   - segmented: 检查`/tmp/flashvsr_segments/{video_dir_name}/segment_*.pt`文件
