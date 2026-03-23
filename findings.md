# Findings & Decisions

## Requirements
<!-- Captured from user request -->
- 用简洁的语言和代码解释什么是 group gemm
- 基于 hpc-ops 代码库和 "CUDA Programming 10.2.md" 文档
- 包括概念、数学表述、代码示例、实际应用场景

## Research Findings
<!-- Key discoveries during exploration -->

### Group GEMM 基本概念
- **普通 GEMM**: Y = X * W^T，其中 X [M, K], W [N, K], Y [M, N]
- **Group GEMM**: 一次操作中执行多个独立的矩阵乘法
  - 有 `num_group` 个独立的权重矩阵 W_0, W_1, ..., W_{num_group-1}
  - 输入 X 被分割成 `num_group` 个连续的组
  - 每个 X_i 与对应的 W_i 相乘
  - 所有结果拼接起来得到最终输出 Y

### 数据结构发现
- **x**: [total_seq, hidden_size] (fp8) - 输入激活
- **weight**: [num_group, output_dim, hidden_size] (fp8) - 权重
- **seqlens**: [num_group] (int32) - 每组的 token 数量
- **cu_seqlens**: [num_group + 1] (int32) - 累积序列长度，用于快速定位各组位置
- **y_scale**: [num_group] (float32) - FP8 量化缩放因子
- **output**: [total_seq, output_dim] (bfloat16) - 输出

### cu_seqlens 的计算
```python
cu_seqlens = torch.cumsum(
    torch.cat([torch.tensor([0]), seqlens]),
    dim=0
)
```
示例：seqlens = [4, 6, 3] → cu_seqlens = [0, 4, 10, 13]

### Group GEMM 的真正优势
- **一次 kernel launch**：减少 launch 开销
- **统一的内存访问调度**：在一个 kernel 内协调多个组的内存访问
- **灵活的工作调度**：不同 thread block/warp 处理不同的组，整体填满硬件
- **共享初始化开销**：TMA 设置、barrier 初始化等工作可以共享

### 注意事项
- Group GEMM 并不改变"每个组仍是小矩阵"这个事实
- 之前关于"小矩阵乘法硬件利用率低"的说法不准确，已修正

## Technical Decisions
<!-- Decisions made with rationale -->
| Decision | Rationale |
|----------|-----------|
| 从测试中的 naive 实现开始讲解 | 代码清晰，易于理解基本逻辑 |
| 使用 ASCII 图展示张量形状 | 可视化帮助理解数据布局 |
| 先讲概念再讲代码 | 建立直觉后再看细节 |
| 用中文编写 | 与现有文档风格一致 |

## Issues Encountered
<!-- Errors and how they were resolved -->
| Issue | Resolution |
|-------|------------|
| 用户指出"小矩阵利用率低"描述不准确 | 重新分析 Group GEMM 的真正优势并更新文档 |
| 文件被用户修改后编辑失败 | 重新读取文件再进行编辑 |

## Resources
<!-- URLs, file paths, API references -->
- `/cyq/Projects/hpc-ops/CUDA Programming 10.2.md` - 主文档
- `/cyq/Projects/hpc-ops/tests/test_group_gemm_pertensor.py` - 包含 naive 实现
- `/cyq/Projects/hpc-ops/hpc/group_gemm.py` - Python API
- `/cyq/Projects/hpc-ops/src/group_gemm/` - CUDA 实现

## Visual/Browser Findings
<!-- CRITICAL: Update after every 2 view/browser operations -->
- 从代码中理解了 Group GEMM 的数据流向
- 从 naive 实现中清晰看到了分组计算的逻辑
- 用户的反馈帮助修正了对 Group GEMM 优势的理解

---

## group_gemm_pertensor_fp8_kernel 算法概述

### 整体架构
这是一个针对 NVIDIA SM90 (H20) 架构优化的 FP8 Group GEMM kernel。

### 核心设计特点

**1. 双流处理器划分 (Dual Warpgroup Partition)**
- **数学 Warpgroup** (前 256 线程): 执行 GMMA (Grouped Matrix Multiply-Accumulate) 计算
  - 使用 `warpgroup_reg_alloc<168>()` 分配更多寄存器
  - 128 线程组成一个 warpgroup 执行 MMA
  - 共 2 个 warpgroup (kWarpgroupM=2)
- **加载 Warpgroup** (后 128 线程): 负责通过 TMA 加载数据
  - 使用 `warpgroup_reg_dealloc<24>()` 减少寄存器占用
  - 只由 leader 线程 (idx=0 在 load warp 中) 发起 TMA 传输

**2. 分阶段流水线 (kStage=8)**
- 使用 8 个阶段的双缓冲机制
- `readable[]` 和 `writable[]` 两套 barrier 数组
- 允许计算和内存传输重叠进行

**3. TMA (Tensor Memory Accelerator) 优化**
- 每个 group 有独立的 TMA descriptor (td_xy[igroup*2 + 0] 用于 X, td_xy[igroup*2 + 1] 用于 Y)
- `update_grouped_tma` kernel 预先为每个 group 设置好 TMA descriptor
- 使用 barrier 同步 TMA 传输完成

**4. 动态任务调度**
两种 tile 遍历模式：
- **水平模式 (IsLoopH=true)**: 用于小矩阵 (k<=1024 || n<=1024)
  - `get_next_tile_horizon`: 线性扫描查找下一个 tile
  - 使用 `flat_divider` 将 iblock 分解为 (itile_m_total, itile_n)
- **垂直模式 (IsLoopH=false)**: 用于大矩阵
  - `get_next_tile_vert`: 二分查找确定 igroup
  - 使用 cu_tiles_ptr 累积索引快速定位

**5. Tile 大小自适应**
根据 `num_seq_per_group_avg` 选择不同的 kTileM：
- <=16: kTileM=16
- <=32: kTileM=32
- <=48: kTileM=48
- else: kTileM=64
- kTileN=128, kTileK=128 (固定)

### 数据流向

```
Global Memory ──TMA──> Shared Memory ──GMMA──> Registers ──Epilogue──> Shared Memory ──TMA──> Global Memory
     (X, W)               (sA, sB)          (tAr, tBr, tCr)             (sCT)               (Y)
```

### 关键代码路径

**Kernel 入口**: `group_gemm_pertensor_fp8_kernel` (kernels.cuh:144-424)

**数学 Warpgroup 流程**:
1. `get_next_tile_vert/horizon`: 获取当前要处理的 tile
2. `wait_barrier(readable[ismem_read], phase)`: 等待数据加载完成
3. `cute::gemm(tiled_mma, ...)`: 执行 GMMA 矩阵乘累加
4. `tDr(i) = tCr(i) * scale + tDr(i)`: 应用量化缩放并累加
5. Epilogue: 寄存器 → shared memory → TMA store

**加载 Warpgroup 流程**:
1. `wait_barrier(writable[ismem_write], phase)`: 等待 shared memory 可用
2. `cute::copy(tma_a.with(...), ...)`: TMA load X tile
3. `cute::copy(tma_b.with(...), ...)`: TMA load W tile
4. `set_barrier_transaction_bytes(readable[ismem_write], ...)`: 设置 barrier 事务字节数

### 共享内存布局
```
shm_data[]:
├─ shm_a (SLayoutX): [kTileM, kTileK, kStage] × sizeof(Tin)
├─ shm_b (SLayoutW): [kTileN, kTileK, kStage] × sizeof(Tin)
├─ shm_c (SLayoutY): [kTileN, kTileM] × sizeof(Tout)
└─ shm_tiles: [num_group+1] × sizeof(int)
```

### GMMA 配置
使用 SM90 架构的 FP8 GMMA 指令：
- 例如 `SM90_64x16x32_F32E4M3E4M3_SS_TN` (kTileM=16)
- 输出累积在 fp32 寄存器中保证精度
- 最后转换回 bf16

### 更新记录
- 2026-03-23: 初始分析完成

---
*Update this file after every 2 view/browser/search operations*
*This prevents visual information from being lost*
