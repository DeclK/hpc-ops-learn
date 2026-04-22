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

## update_grouped_tma Kernel 详解

### 核心作用
`update_grouped_tma` kernel 有两个主要功能：

1. **为每个 group 计算 tile 统计信息**（最后一个 block 负责）
   - 计算每个 group 需要多少个 tile M：`tiles[i] = (seqlens_ptr[igroup] + kTileM - 1) / kTileM`
   - 使用 cub::BlockScan 做 exclusive sum scan 得到累积 tile 索引
   - 输出：
     - `tiles_ptr[igroup]`: 第 igroup 个 group 的 tile 数量
     - `cu_tiles_ptr[igroup]`: 累积 tile 数量（类似 cu_seqlens）

2. **为每个 group 配置独立的 TMA descriptor**（前 num_group 个 block 负责）
   - 每个 block 处理一个 group
   - 为该 group 的 X 和 Y 张量分别设置 TMA descriptor
   - 每个 group 有 2 个 TMA descriptor：`td_xy[igroup*2 + 0]` (X), `td_xy[igroup*2 + 1]` (Y)

### 详细代码分析

**第一部分：Tile 统计计算（blockIdx.x == num_group）**
```cpp
if (igroup == num_group) {
  // 每个线程处理 kGroupPerThread=8 个 group
  int tiles[kGroupPerThread];
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = idx * kGroupPerThread + i;
    if (igroup < num_group) {
      // 计算该 group 需要多少个 tile M
      tiles[i] = (seqlens_ptr[igroup] + kTileM - 1) / kTileM;
      tiles_ptr[igroup] = tiles[i];
    }
  }

  // 使用 BlockScan 做 exclusive sum
  using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage;
  int block_aggregate;
  BlockScan(temp_storage).ExclusiveSum(tiles, tiles, block_aggregate);

  // 写入 cu_tiles_ptr
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = idx * kGroupPerThread + i;
    if (igroup < num_group) {
      cu_tiles_ptr[igroup] = tiles[i];
    }
  }
  if (idx == 0) {
    cu_tiles_ptr[num_group] = block_aggregate;
  }
}
```

**第二部分：TMA Descriptor 更新（blockIdx.x < num_group）**
```cpp
else {
  __shared__ cute::TmaDescriptor smem_tma_desc[2];

  // 获取该 group 的信息
  int num_seq = seqlens_ptr[igroup];
  int cu_seqlen = cu_seqlens_ptr[igroup];
  auto *x_ibatch_ptr = x_ptr + cu_seqlen * k;  // X 在该 group 的起始地址
  auto *y_ibatch_ptr = y_ptr + cu_seqlen * n;  // Y 在该 group 的起始地址

  // 从全局内存拷贝模板 TMA descriptor 到 shared memory
  if (idx < 2) {
    smem_tma_desc[idx] = td_xy[idx];
  }
  __syncwarp();

  // 更新 X 的 TMA descriptor
  if (idx == 0) {
    auto gX = make_tensor(make_gmem_ptr(x_ibatch_ptr),
                          make_shape(num_seq, k),
                          make_stride(k, Int<1>{}));
    update_tma_gtensor<TmaX>(smem_tma_desc[idx], gX);
  }

  // 更新 Y 的 TMA descriptor
  if (idx == 1) {
    auto gY = make_tensor(make_gmem_ptr(y_ibatch_ptr),
                          make_shape(n, num_seq),
                          make_stride(Int<1>{}, n));
    update_tma_gtensor<TmaY>(smem_tma_desc[idx], gY);
  }

  // 提交并写回全局内存
  for (int i = 0; i < 2; i++) {
    __syncwarp();
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    tma_descriptor_cp_fence_release(tma_xy + igroup * 2 + i, smem_tma_desc[i]);
  }
}
```

### 为什么需要 update_grouped_tma？

1. **每个 group 的数据位置不同**
   - X 张量在全局内存中是连续存储的：`X[total_seq, k]`
   - 第 igroup 个 group 的 X 起始位置：`x_ptr + cu_seqlens[igroup] * k`
   - TMA descriptor 需要知道确切的全局内存地址

2. **每个 group 的形状不同**
   - 不同 group 可能有不同的 seqlen（seqlens_ptr[igroup]）
   - TMA descriptor 需要配置正确的 tensor shape

3. **动态调度的需求**
   - `tiles_ptr` 和 `cu_tiles_ptr` 用于后续 kernel 的动态任务调度
   - `get_next_tile_horizon` 和 `get_next_tile_vert` 需要这些信息

### TMA Descriptor 更新的关键步骤

在 `update_tma_gtensor` 中（tma.cuh:37-57）：
1. 提取 tensor 的 shape 和 stride
2. 更新 TMA descriptor 的全局内存地址
3. 更新 TMA descriptor 的 shape 维度

### 对后续 Group GEMM 的帮助

1. **简化主 kernel 逻辑**：主 kernel 不需要为每个 group 重新计算 TMA descriptor
2. **减少重复工作**：TMA descriptor 配置一次，多次使用
3. **支持动态调度**：tiles_ptr 和 cu_tiles_ptr 让主 kernel 可以高效地查找下一个要处理的 tile
4. **内存访问优化**：每个 group 有独立的 TMA descriptor，可以更高效地利用 TMA 引擎

### 更新记录
- 2026-03-23: 初始分析完成
- 2026-03-24: 添加 update_grouped_tma kernel 详解

---

## Transposed MMA Tiler 详解

### 问题背景

在普通的 GEMM kernel 中，通常固定 kTileM=128（M 维度的 tile 大小），这是因为 Tensor Core 的 MMA 指令通常在 M 维度有较大的粒度。但在 Group GEMM 场景中，每个 group 的 seqlen 可能很小（小 M 场景），如果 kTileM 太大，会导致：
- 硬件利用率低（小矩阵无法填满 Tensor Core）
- 需要大量 padding，浪费计算和内存

---

### 核心概念：为什么转置是必要的？

#### 标准 MMA Atom 的约定

首先理解 CuTe/CUTLASS 中 MMA atom 的标准约定：

```
标准 MMA: C = A @ B
  - A.shape = (M, K)   ← 通常是 Input X
  - B.shape = (N, K)   ← 通常是 Weight W
  - C.shape = (M, N)   ← 输出 Y
```

但 SM90 架构的 MMA 指令有一个特点：**M 维度的粒度通常较大（64/128），而 N 维度的粒度可以更小（16/32）**。

看 `config.h` 中的指令选择：
```cpp
SM90_64x16x32_F32E4M3E4M3_SS_TN  // M=64, N=16
SM90_64x32x32_F32E4M3E4M3_SS_TN  // M=64, N=32
SM90_64x64x32_F32E4M3E4M3_SS_TN  // M=64, N=64
```

注意：**M 固定是 64，而 N 可以是 16/32/64！**

#### Group GEMM 的痛点

在 Group GEMM 中：
- **M 维度** = seqlen（每个 group 的 token 数），可能很小（如 4, 8, 16）
- **N 维度** = output_dim（输出维度），通常很大（如 7168, 14336）

如果用标准 MMA：
- kTileM 最小是 64，对于 seqlen=16 来说太大了
- 大量 padding 浪费计算和内存

**解决方案：把问题转置过来！**

---

### hpc-ops 的转置设计详解

#### 1. 整体思路

```
原始问题: Y[M, N] = X[M, K] @ W^T[K, N]

转置后:   Y^T[N, M] = W[N, K] @ X^T[K, M]
           ↑              ↑           ↑
         输出          Weight      Input
         (现在 N 维度大了！)
```

通过转置，原来的小 M 变成了小 N，而原来的大 N 变成了大 M！

#### 2. 关键代码点：A 和 B 的互换

在 `kernels.cuh` 第 313-317 行：

```cpp
// sA 是 X [M, K], sB 是 W [N, K]

auto tBs4r = thr_mma.partition_A(sB);  // ← sB (W) 作为 MMA 的 A
auto tAs4r = thr_mma.partition_B(sA);  // ← sA (X) 作为 MMA 的 B

auto tBr = thr_mma.make_fragment_A(tBs4r);  // fragment A ← W
auto tAr = thr_mma.make_fragment_B(tAs4r);  // fragment B ← X
```

**这是最关键的一步！** 角色完全互换了：

| 角色 | 标准 GEMM | hpc-ops Group GEMM |
|------|-----------|-------------------|
| MMA A 矩阵 | X [M, K] | **W [N, K]** ← 互换 |
| MMA B 矩阵 | W [N, K] | **X [M, K]** ← 互换 |
| 结果 C | Y [M, N] | **Y^T [N, M]** ← 转置 |

#### 3. MMA 指令的 _TN 后缀

看 `config.h` 第 42-61 行的指令命名：

```cpp
SM90_64x16x32_F32E4M3E4M3_SS_TN
                           ↑↑
                           TN
```

`_TN` 后缀的含义：
- **T** = Transposed → A 矩阵在 MMA 内部是转置的
- **N** = Normal → B 矩阵在 MMA 内部是正常的

结合我们的使用方式：
- A = W [N, K]，经过 T 转置后，MMA 看到的是 [K, N]
- B = X [M, K]，经过 N 正常，MMA 看到的是 [M, K]
- **等等，这里需要更仔细的理解...**

实际上，更准确的理解是：**我们选择 _TN 指令是为了配合数据在 shared memory 中的布局。**

#### 4. 共享内存布局也转置了

看 `config.h` 第 81-86 行：

```cpp
using SLayoutX = decltype(tile_to_shape(SLayoutXAtom{},
                                        make_shape(Int<kTileM>{}, Int<kTileK>{})));
using SLayoutW = decltype(tile_to_shape(SLayoutWAtom{},
                                        make_shape(Int<kTileN>{}, Int<kTileK>{})));
using SLayoutY =
    decltype(tile_to_shape(SLayoutYAtom{}, make_shape(Int<kTileN>{}, Int<kTileM>{})));
```

注意：
- SLayoutX (X矩阵): `[kTileM, kTileK]` ✓ 正常
- SLayoutW (W矩阵): `[kTileN, kTileK]` ✓ 正常
- **SLayoutY (输出): `[kTileN, kTileM]` ← 转置了！**

输出 shared memory 是 `[N, M]` 而不是 `[M, N]`！

#### 5. GMMA 调用

在 `kernels.cuh` 第 360 行：

```cpp
cute::gemm(tiled_mma, tBr(_, _, ik, ismem_read), tAr(_, _, ik, ismem_read), tCr(_, _, _));
//                    ↑                          ↑                          ↑
//                  fragment A               fragment B               fragment C
//                  (W数据)                   (X数据)                   (Y^T)
```

#### 6. Epilogue: 存储时也要考虑转置

在 `kernels.cuh` 第 411-419 行：

```cpp
// gD 的形状是 [n, m]，不是 [m, n]！
auto gD = tma_d.get_tma_tensor(make_shape(n, m));

// ...

// 注意索引顺序: itile_n 在前，itile_m 在后
cute::copy(tma_d.with(td_y), tDs(_, iwarpgroup, Int<0>{}),
           tDg(_, itile_n * 2 + iwarpgroup, itile_m));
```

---

### 完整数据流总结

让我们用一个具体例子来说明：

假设：
- seqlen (M) = 16
- output_dim (N) = 128
- hidden_size (K) = 128

**标准 MMA 的问题：**
```
kTileM = 64 (最小)，但我们只有 M=16
→ 需要 padding 到 64，浪费 75% 的计算
```

**hpc-ops 转置方案：**

| 阶段 | 操作 | 形状 | 说明 |
|------|------|------|------|
| 输入 | X | [16, 128] | seqlen=16 |
| 输入 | W | [128, 128] | output_dim=128 |
| TMA 加载 | sA = X_tile | [16, 128] | kTileM=16 ✓ |
| TMA 加载 | sB = W_tile | [128, 128] | kTileN=128 |
| **关键** | MMA A = sB | [128, 128] | W 作为 A |
| **关键** | MMA B = sA | [16, 128] | X 作为 B |
| GMMA | C = A @ B | [128, 16] | Y^T！ |
| Epilogue | 存储 Y^T | [128, 16] | 全局内存 |
| 最终结果 | Y = (Y^T)^T | [16, 128] | 逻辑上正确 |

---

### 为什么固定 kTileN=128 而不是 kTileM？

这是一个非常好的问题。答案在于：**N 维度通常很大，而 M 维度可能很小。**

看 `group_gemm_pertensor_fp8.cu` 中的配置：

```cpp
// kTileN 固定是 128！
constexpr static int kTileN = 128;
constexpr static int kTileK = 128;

// kTileM 根据平均 seqlen 自适应选择
if (num_seq_per_group_avg <= 16) {
  run_with_config<16, 128, 128, 8>(...);
} else if (num_seq_per_group_avg <= 32) {
  run_with_config<32, 128, 128, 8>(...);
} else if (num_seq_per_group_avg <= 48) {
  run_with_config<48, 128, 128, 8>(...);
} else {
  run_with_config<64, 128, 128, 8>(...);
}
```

**策略的合理性：**

| 维度 | 特点 | tile 大小策略 |
|------|------|-------------|
| **M** (seqlen) | 可能很小 (4, 8, 16, ...) | **自适应** (16/32/48/64) |
| **N** (output_dim) | 通常很大 (7168, 14336) | **固定 128**，充分利用硬件 |
| **K** (hidden_size) | 通常是 128 的倍数 | **固定 128** |

---

### 数学正确性验证

让我们严谨地证明转置的正确性：

```
原始问题:
  Y[M, N] = X[M, K] * W^T[K, N]

转置等式两边:
  Y^T[N, M] = (X * W^T)^T
             = (W^T)^T * X^T           (矩阵转置性质: (AB)^T = B^T A^T)
             = W[N, K] * X^T[K, M]     ✓

hpc-ops 实际计算的就是:
  C[N, M] = A[N, K] * B[M, K]^T       (A=W, B=X)
           = W[N, K] * X^T[K, M]
           = Y^T[N, M]

最后在逻辑上, 用户拿到的 Y 就是正确的结果!
```

---

### 需要注意的代码点总结

当阅读 hpc-ops 代码时，如果看到 "奇怪" 的索引顺序，不要困惑，这都是转置设计的一部分：

| 文件 | 行号 | 代码 | 注意事项 |
|------|------|------|---------|
| `config.h` | 42-61 | `mma_selector()` | 都是 `_TN` 指令 |
| `config.h` | 85-86 | `SLayoutY` | 形状是 `[kTileN, kTileM]` |
| `kernels.cuh` | 313-314 | `partition_A(sB)` / `partition_B(sA)` | **A 和 B 互换** |
| `kernels.cuh` | 360 | `gemm(tiled_mma, tBr, tAr, tCr)` | **tBr 在前，tAr 在后** |
| `kernels.cuh` | 411 | `make_shape(n, m)` | **n 在前，m 在后** |
| `kernels.cuh` | 418-419 | `tDg(_, itile_n * 2 + iwarpgroup, itile_m)` | **itile_n 在前** |

---

### 总结

Transposed MMA Tiler 的核心思想可以用一句话概括：

> **既然 Tensor Core 在 N 维度粒度较小，那我们就通过转置，把小的 seqlen (M) 放到 N 维度，把大的 output_dim (N) 放到 M 维度！**

这个设计展示了如何通过**改变问题的表述方式**（而不是改变硬件）来获得更好的性能。这是一个非常巧妙的工程权衡！

---

## Scheduler for Group Gemm 详解

### 两种调度模式

hpc-ops 提供了两种 tile 调度模式，根据矩阵大小自动选择：

| 模式 | 适用场景 | 查找方式 |
|------|---------|---------|
| Horizontal (IsLoopH=true) | 小矩阵 (k<=1024 \|\| n<=1024) | 线性扫描 |
| Vertical (IsLoopH=false) | 大矩阵 | 二分查找 |

### 1. Horizontal 模式（线性扫描）

**核心思想**：将 block index 扁平化为 (itile_m_total, itile_n)，然后线性扫描找到对应的 group。

**关键代码** (`kernels.cuh` 第 22-40 行)：

```cpp
__device__ __forceinline__ void get_next_tile_horizon(
    const int *tiles_ptr, int iblock, int num_group,
    int &igroup, int &itile_m, int &itile_n, int &sum_tile_m,
    cutlass::FastDivmod flat_divider) {
  int num_tile_m, itile_m_total;

  // 步骤 1: 将 iblock 分解为 (itile_m_total, itile_n)
  flat_divider(itile_m_total, itile_n, iblock);

  // 步骤 2: 线性扫描找到 igroup
  for (int i = igroup; i < num_group; i++) {
    num_tile_m = tiles_ptr[i];
    sum_tile_m += num_tile_m;
    if (itile_m_total < sum_tile_m) {
      igroup = i;
      sum_tile_m = sum_tile_m - num_tile_m;
      itile_m = itile_m_total - sum_tile_m;
      return;
    }
  }
  igroup = -1;  // 结束
}
```

**关键点解析**：
- `flat_divider(itile_m_total, itile_n, iblock)`: 使用 fast divmod 将 `iblock` 分解为 `iblock = itile_m_total * num_tile_n + itile_n`
- `igroup` 作为 in/out 参数，从上次的位置继续扫描，避免重复检查
- `sum_tile_m` 累积 tile 数量，用于判断 `itile_m_total` 落在哪个 group
- 返回时 `igroup` 是找到的 group，`itile_m` 是在该 group 内的 tile 索引

**为什么小矩阵用线性扫描？**
- 小矩阵意味着 num_group 不大
- 线性扫描实现简单，指令数少
- 缓存友好，因为 tiles_ptr 是连续访问的
- 从上次的 `igroup` 位置继续搜索，实际复杂度远低于 O(n)

### 2. Vertical 模式（二分查找）

**核心思想**：利用 cu_tiles_ptr 的累积索引结构，通过二分查找快速定位 igroup。

**关键代码** (`kernels.cuh` 第 42-61 行)：

```cpp
__device__ __forceinline__ void get_next_tile_vert(
    const int *cu_tiles_ptr, int iblock, int num_group,
    int &igroup, int &itile_m, int &itile_n, int total_m) {
  // 步骤 1: 分解 iblock
  int itile_m_total = iblock % total_m;
  itile_n = iblock / total_m;

  // 步骤 2: 二分查找 igroup
  int left = 0;
  int right = num_group;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (cu_tiles_ptr[mid] > itile_m_total) {
      right = mid - 1;
    } else {
      left = mid + 1;
    }
  }

  // 步骤 3: 计算 itile_m
  itile_m = itile_m_total - cu_tiles_ptr[right];
  igroup = right;
}
```

**关键点解析**：
- `iblock` 分解方式不同：`itile_m_total = iblock % total_m`，`itile_n = iblock / total_m`
- 二分查找在 `cu_tiles_ptr` 数组中找最大的 `right` 满足 `cu_tiles_ptr[right] <= itile_m_total`
- 因为 `cu_tiles_ptr` 是 exclusive sum，所以 `right` 就是对应的 `igroup`
- `itile_m = itile_m_total - cu_tiles_ptr[right]` 得到在 group 内的 tile 索引

**为什么大矩阵用二分查找？**
- 大矩阵意味着 num_group 可能很大
- 二分查找时间复杂度 O(log n)，远优于线性扫描 O(n)
- cu_tiles_ptr 是有序的（累积和），天然适合二分查找
- 不需要从上次位置继续，每次都是独立的 O(log n) 查找

### cu_tiles_ptr 的作用

在 `update_grouped_tma` kernel 中计算（`kernels.cuh` 第 88-91 行）：

```cpp
using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
__shared__ typename BlockScan::TempStorage temp_storage;
int block_aggregate;
BlockScan(temp_storage).ExclusiveSum(tiles, tiles, block_aggregate);
```

**Exclusive Sum 示例**：
```
输入 tiles:    [4, 6, 3]  (每个 group 的 tile 数)
输出 cu_tiles: [0, 4, 10, 13]  (累积索引)
```

这样，`cu_tiles_ptr[igroup]` 表示第 igroup 个 group 之前的总 tile 数。

### 调度模式选择

在主 kernel 调用处（`group_gemm_pertensor_fp8.cu` 第 69-89 行），根据问题规模选择模式：

```cpp
if (k <= 1024 || n <= 1024) {
  // Horizontal 模式：小矩阵，线性扫描
  constexpr bool IsLoopH = true;
  auto kernel =
      kernels::group_gemm_pertensor_fp8_kernel<decltype(config), decltype(tma_x),
                                               decltype(tma_w), decltype(tma_y), IsLoopH>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  kernel<<<grid, block, shm_size, stream>>>(tma_w, tma_xy, (int *)seqlens_ptr, (float *)y_scale,
                                            (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m,
                                            n, k, flat_divider);
} else {
  // Vertical 模式：大矩阵，二分查找
  constexpr bool IsLoopH = false;
  auto kernel =
      kernels::group_gemm_pertensor_fp8_kernel<decltype(config), decltype(tma_x),
                                               decltype(tma_w), decltype(tma_y), IsLoopH>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  kernel<<<grid, block, shm_size, stream>>>(tma_w, tma_xy, (int *)seqlens_ptr, (float *)y_scale,
                                            (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m,
                                            n, k, flat_divider);
}
```

**选择阈值**：`k <= 1024 || n <= 1024`
- k = hidden_size，n = output_dim
- 当这两个维度较小时，意味着问题规模小，group 数量可能也少
- 当维度大时，问题规模大，group 数量可能很多

### shared memory 中的 tiles 缓存

在 kernel 开始时（`kernels.cuh` 第 228-236 行），将 tiles 或 cu_tiles 缓存到 shared memory：

```cpp
if constexpr (IsLoopH) {
  for (int i = idx; i < num_group; i += blockDim.x) {
    shm_tiles[i] = tiles_ptr[i];
  }
} else {
  for (int i = idx; i < (num_group + 1); i += blockDim.x) {
    shm_tiles[i] = cu_tiles_ptr[i];
  }
}
```

这样后续的 scheduler 调用可以访问 shared memory，减少 global memory 访问延迟。

### 两种模式在 kernel 中的使用

在 load warpgroup 和 math warpgroup 中都需要调用 scheduler：

**Load warpgroup** (`kernels.cuh` 第 262-273 行)：
```cpp
if constexpr (IsLoopH) {
  get_next_tile_horizon(shm_tiles, iblock, num_group, igroup, itile_m, itile_n, sum_tile_m,
                        flat_divider);
  if (igroup < 0) {
    break;
  }
} else {
  get_next_tile_vert(shm_tiles, iblock, num_group, igroup, itile_m, itile_n, total_m);
  if (itile_n >= num_tile_n) {
    break;
  }
}
```

**Math warpgroup** (`kernels.cuh` 第 329-340 行) 有完全相同的代码。

注意：两个 warpgroup 独立地调度任务，通过 `iblock += gridDim.x` 来分配不同的 tile 给不同的 block。

### 总结

Scheduler 的设计体现了**实用主义**：
- **小矩阵 (k<=1024 || n<=1024)**：Horizontal 模式，简单直接的线性扫描，利用缓存和增量搜索，实际开销很小
- **大矩阵**：Vertical 模式，高效的二分查找，O(log n) 时间复杂度，扩展性好
- 通过编译期条件（`if constexpr`）选择实现，无运行时开销
- 两种模式都将 tiles/cu_tiles 缓存到 shared memory 优化访问

---

## 关于两个 warpgroup 做 TMA store 的澄清

### 之前的误解
我之前认为使用两个 warpgroup 做 TMA store 是因为 **TMA copy box 硬件限制**。

### 用户的澄清（准确答案）

真正的原因是**减少同步开销**！

#### 分析

**方案对比：**

| 方案 | 描述 | 缺点 |
|------|------|------|
| 单 thread 发起 | 两个 warpgroup 同步，等 rmem→smem 完成后，单个 thread 发起 TMA store | 同步开销大 |
| **两个 warpgroup 各自发起**（hpc-ops 方案）| 每个 warpgroup 完成 rmem→smem 读取后，直接发起 store | 同步开销小 |

#### 代码证据

```cpp
// kernels.cuh 第 403-404, 408, 418-419 行
syncwarpgroup(iwarpgroup);              // ← 只需要 warpgroup 级别同步
cute::tma_store_fence();
// ...
cute::copy(tma_d.with(td_y), tDs(_, iwarpgroup, Int<0>{}),
           tDg(_, itile_n * 2 + iwarpgroup, itile_m));
```

#### TMA copy box 的实际限制（参考知乎文章）

根据用户提供的参考资料：
- **单个维度的元素数量最大为 256**（对于 kTileN=128 来说完全够用）
- **最小的 copy 单元为 16 bytes**

这些限制都不是 hpc-ops 使用两个 warpgroup 的原因。

#### 用户的参考链接
1. [写给大家看的 CuTe 教程：TMA Copy](https://zhuanlan.zhihu.com/p/2003198909405763007)
2. [cute 之 Hopper TMA](https://zhuanlan.zhihu.com/p/1985678344352731952)

---

## Vertical Scheduler 的数据局部性分析

### 问题背景
用户提出了一个非常好的问题：在 Vertical 模式中，我们对 iblock 进行 divide 的时候沿着 m 轴进行划分，这样很有可能两个不同的 iblock 划分到了不同的 igroup 当中。这会不会对数据的局部性产生较大的影响？

### 两种模式的 iblock 分解方式对比

让我们先回顾两种模式的 iblock 分解方式：

**Horizontal 模式（先 N 后 M）：**
```cpp
// iblock = itile_m_total * num_tile_n + itile_n
flat_divider(itile_m_total, itile_n, iblock);
```

**Vertical 模式（先 M 后 N）：**
```cpp
// iblock = itile_n * total_m + itile_m_total
int itile_m_total = iblock % total_m;
itile_n = iblock / total_m;
```

### 具体例子可视化

假设：
- `total_m = 10`（总 tile M 数）
- `num_tile_n = 5`（tile N 总数）
- `tiles_ptr = [4, 3, 3]`（3 个 group，每个 group 的 tile M 数）
- `cu_tiles_ptr = [0, 4, 7, 10]`

**Horizontal 模式**（iblock 增长方向：先遍历完 N，再 M+1）：
```
iblock:    0  1  2  3  4 |  5  6  7  8  9 | 10 11 12 13 14 | ...
itile_m:   0  0  0  0  0 |  1  1  1  1  1 |  2  2  2  2  2 | ...
itile_n:   0  1  2  3  4 |  0  1  2  3  4 |  0  1  2  3  4 | ...
igroup:    0  0  0  0  0 |  0  0  0  0  0 |  0  0  0  0  0 | ... (连续 5 个 block 都在 igroup 0)
```

**Vertical 模式**（iblock 增长方向：先遍历完 M，再 N+1）：
```
iblock:    0  1  2  3 |  4  5  6 |  7  8  9 | 10 11 12 13 | 14 15 16 | 17 18 19 | ...
itile_m:   0  1  2  3 |  4  5  6 |  7  8  9 |  0  1  2  3 |  4  5  6 |  7  8  9 | ...
itile_n:   0  0  0  0 |  0  0  0 |  0  0  0 |  1  1  1  1 |  1  1  1 |  1  1  1 | ...
igroup:    0  0  0  0 |  1  1  1 |  2  2  2 |  0  0  0  0 |  1  1  1 |  2  2  2 | ...
           ↑              ↑        ↑        ↑              ↑        ↑
         igroup 0      igroup 1  igroup 2  igroup 0      igroup 1  igroup 2
         (连续 4 个)   (3 个)   (3 个)   (连续 4 个)   (3 个)   (3 个)
```

### 数据局部性影响分析

**问题确认：是的，Vertical 模式确实存在频繁的 igroup 跳变！**

在 Vertical 模式下，连续的 thread block 会：
1. 先处理 igroup 0 的 4 个 tile M（iblock 0-3）
2. 然后跳到 igroup 1 的 3 个 tile M（iblock 4-6）
3. 然后跳到 igroup 2 的 3 个 tile M（iblock 7-9）
4. 然后回到 igroup 0 的下一个 tile N（iblock 10-13）
5. 如此反复...

**这确实会影响数据局部性！** 具体影响：

1. **TMA descriptor 切换**：每个 group 有独立的 TMA descriptor，频繁切换 igroup 意味着：
   - 需要加载不同的 TMA descriptor
   - TMA 引擎可能需要重新初始化
   - 缓存的 descriptor 可能失效

2. **权重 W 的访问模式**：
   - 每个 group 有独立的 W[igroup, :, :]
   - 频繁切换 igroup 会导致权重的访问不连续
   - L2 缓存命中率可能下降

3. **输入 X 的访问模式**：
   - 每个 group 的 X 也是不连续的（中间隔着其他 group 的数据）
   - 同样影响缓存命中率

### 为什么 hpc-ops 还要这样设计？

这是一个很好的工程权衡问题。让我们分析可能的原因：

1. **Horizontal 模式在大矩阵下的问题**：
   - Horizontal 模式在 `num_group` 很大时，线性扫描的开销可能很大
   - 虽然有增量搜索优化，但最坏情况仍然是 O(n)
   - 大矩阵通常意味着 `num_group` 也很大

2. **二分查找的 O(log n) 优势**：
   - Vertical 模式使用二分查找，稳定的 O(log n) 时间复杂度
   - 在 `num_group` 很大时，这比线性扫描快得多

3. **数据局部性问题的缓解**：
   - **L2 缓存是共享的**：同一个 SM 内的多个 warp 可以共享 L2 缓存
   - **TMA 的预取能力**：TMA 引擎可以自动预取数据，减轻缓存不命中的影响
   - **工作集大小**：如果每个 group 的数据不大，可能仍然能缓存在 L2 中
   - **每个 block 处理多个 tile**：在主循环中，`iblock += gridDim.x`，同一个 block 会处理：
     - Vertical 模式下：同一个 block 会处理 `(itile_m_total, itile_n)`, `(itile_m_total, itile_n + k)`, `(itile_m_total, itile_n + 2k)`...
     - 注意！同一个 block 处理的是**相同的 itile_m_total**，也就是**相同的 igroup**！

### 关键点：同一个 Block 的访问模式

让我们重新审视主 kernel 中的循环：

```cpp
for (int iblock = blockIdx.x; ; iblock += gridDim.x) {
  // 获取 tile
  // ...
  
  // 处理这个 tile
  // ...
}
```

**Vertical 模式下，同一个 block 处理的 tile 序列：**
```
iblock = blockIdx.x, blockIdx.x + gridDim.x, blockIdx.x + 2*gridDim.x, ...

假设 gridDim.x = 128（128 个 block）
blockIdx.x = 0:
  itile_m_total = 0 % 10 = 0, itile_n = 0 / 10 = 0   ← igroup 0
  itile_m_total = 128 % 10 = 8, itile_n = 128 / 10 = 12 ← igroup 2
  itile_m_total = 256 % 10 = 6, itile_n = 256 / 10 = 25 ← igroup 1
  ...

等等，这看起来也不理想... 但等一下，gridDim.x 通常很大！
```

实际上，更重要的是：**每个 block 内部，一旦找到 igroup，就会一直处理这个 igroup 的 tile 吗？** 不，不是的，每次都是独立的。

### 重新思考：也许问题没有那么严重

让我们从另一个角度看：

1. **每个 group 的数据是独立的**：Group GEMM 的本质就是处理多个独立的矩阵乘法
2. **不同 group 的数据之间没有依赖**：所以访问顺序不影响正确性
3. **缓存是按地址工作的**：即使跳变，如果数据总大小不超过缓存容量，仍然可以命中
4. **现代 GPU 的 L2 缓存很大**：H20 有 50MB+ 的 L2 缓存，可以容纳很多 group 的数据

### 总结

| 问题 | 答案 |
|------|------|
| Vertical 模式是否会频繁跳变 igroup？ | **是的**，连续的 iblock 确实会在不同 igroup 之间跳变 |
| 是否会影响数据局部性？ | **理论上是的**，但实际影响可能没有想象中大 |
| 为什么还要这样设计？ | **工程权衡**：二分查找的 O(log n) 扩展性在大矩阵下更重要 |
| 有什么缓解措施？ | L2 缓存、TMA 预取、同一个 block 处理的模式 |

这是一个典型的**算法复杂度 vs 数据局部性**的权衡。在小矩阵（num_group 小）时，Horizontal 模式的线性扫描 + 好的数据局部性更优；在大矩阵（num_group 大）时，Vertical 模式的 O(log n) 二分查找更重要，即使牺牲一些数据局部性。

---

## TMA for W 的三维 Tensor + 二维 Copy Box 设计详解

### 关键发现：W 的实际内存布局

在 `group_gemm_pertensor_fp8.cu` 第 28-32 行，我们找到了 W 的完整定义：

```cpp
auto W = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_ptr)),
                     make_shape(n, k, num_group), 
                     make_stride(k, Int<1>{}, n * k));
//                     ↑        ↑          ↑
//                   stride0  stride1    stride2
```

**这是理解整个设计的关键！**

---

### 第一层：W 的 Shape 和 Stride 详解

让我们详细分析 W 的 shape 和 stride：

| 维度 | Shape | Stride | 说明 |
|------|-------|--------|------|
| **维度 0** | `n` (output_dim) | `k` | 每一行 n 间隔 k 个元素 |
| **维度 1** | `k` (hidden_size) | `1` | k 维度是最内层，连续存储 |
| **维度 2** | `num_group` | `n * k` | 每个 group 占用 n*k 个元素 |

**内存布局图示：**

```
全局内存中的 W:
┌─────────────────────────────────────────────────────────────┐
│ group 0:                                                     │
│   W[0, 0, 0], W[0, 1, 0], ..., W[0, k-1, 0]  ← n=0      │
│   W[1, 0, 0], W[1, 1, 0], ..., W[1, k-1, 0]  ← n=1      │
│   ...                                                       │
│   W[n-1, 0, 0], W[n-1, 1, 0], ..., W[n-1, k-1, 0]       │
├─────────────────────────────────────────────────────────────┤
│ group 1:                                                     │
│   W[0, 0, 1], W[0, 1, 1], ..., W[0, k-1, 1]  ← n=0      │
│   W[1, 0, 1], W[1, 1, 1], ..., W[1, k-1, 1]  ← n=1      │
│   ...                                                       │
│   W[n-1, 0, 1], W[n-1, 1, 1], ..., W[n-1, k-1, 1]       │
├─────────────────────────────────────────────────────────────┤
│ group 2:                                                     │
│   ...                                                       │
└─────────────────────────────────────────────────────────────┘

注意：
- 同一个 group 内：形状是 (n, k)，k 连续
- 不同 group 之间：间隔 n*k 个元素
- 整个 W 的大小：n * k * num_group
```

**内存地址计算公式：**

```
W(in, ik, igroup) 的地址 = 
  w_ptr + in * k          ← n 维度
         + ik * 1          ← k 维度（连续）
         + igroup * (n*k)  ← group 维度
```

---

### 第二层：为什么全局 Tensor 是三维的？

现在我们可以回答第一个问题：**为什么 W 的全局 tensor 是三维的 `(n, k, num_group)`？**

**答案：因为所有 group 的权重在全局内存中是连续存储的！**

这种设计有几个重要优势：

1. **单个 TMA descriptor 可以描述所有 group**：不需要为每个 group 创建独立的 TMA descriptor
2. **通过索引选择 group**：可以通过 `igroup` 索引直接访问不同 group 的数据
3. **内存连续性**：同一个 group 的数据在内存中是连续的，便于 TMA 传输

对比 X（激活）的设计：
- **X**：每个 group 的数据在全局内存中不连续 → 需要为每个 group 独立的 TMA descriptor
- **W**：所有 group 的数据在全局内存中连续 → 可以用一个三维 tensor + 索引来访问

---

### 第三层：为什么 Copy Box 是二维的？

现在回答第二个问题：**为什么 TMA copy box 是二维的 `take<0, 2>(SLayoutW{})`？**

让我们再看 config.h 中的代码：

```cpp
// SLayoutW 的形状是三维的：(kTileN, kTileK, kStage)
using SLayoutW = decltype(tile_to_shape(SLayoutWAtom{}, 
                          make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

// 但 TMA copy box 只取前两维！
auto tma_w = make_tma_copy(SM90_TMA_LOAD{}, w, take<0, 2>(SLayoutW{}));
//                                                        ↑
//                                                     只取前两维！
```

**关键答案：TMA copy box 描述的是「每次拷贝什么形状」，而不是「全局 tensor 是什么形状」。**

对于权重 W，每次 TMA 拷贝只需要：
- **一个 tile** 的数据
- 属于 **一个特定的 group**

也就是说，每次 TMA 拷贝的是 `[kTileN, kTileK]` 的二维数据！

```
take<0, 2>(SLayoutW{})
= 取 SLayoutW 的前两维
= (kTileN, kTileK)
= 正好是一个 tile 的形状！
```

**为什么不需要包含 kStage 维度？**
- kStage 是 shared memory 中用于双缓冲的阶段数
- TMA 每次只拷贝一个 stage 的数据到 shared memory
- kStage 维度是 shared memory layout 的一部分，不是 TMA copy box 的一部分

**为什么不需要包含 num_group 维度？**
- num_group 是全局 tensor 的维度
- 每次 TMA 拷贝只拷贝一个 group 的数据
- 通过索引 `igroup` 来选择具体的 group，而不是通过 copy box

---

### 第四层：实际使用时的索引方式

现在看 kernels.cuh 中的实际使用方式（第 191, 202, 287-288 行）：

```cpp
// 步骤 1: 创建全局的三维 TMA tensor
auto gB = tma_b.get_tma_tensor(make_shape(n, k, num_group));
//              ↑ shape: (n, k, num_group)

// 步骤 2: 分区得到四维的 tile tensor
auto tBg = btma_b.partition_S(gB);  
//              ↑ shape: (TMA, TMA_N, TMA_K, num_group)

// 步骤 3: 使用时通过索引选择具体的 tile 和 group
cute::copy(tma_b.with(readable[ismem_write]), 
           tBg(_, itile_n, itile_k, igroup),  // ← 四维索引！
           tBs(_, 0, 0, ismem_write));
//               ↑        ↑        ↑
//            itile_n  itile_k  igroup
```

**关键点：`tBg(_, itile_n, itile_k, igroup)`**

这里的索引是四维的：
- `_` : TMA 维度（保持不变）
- `itile_n` : 第几个 tile N
- `itile_k` : 第几个 tile K
- `igroup` : **第几个 group** ← 这就是我们选择 group 的方式！

**不是通过 copy box 来选择 group，而是通过索引来选择 group！**

---

### 第五层：完整的数据流总结

让我们用一个具体例子来说明整个流程：

假设：
- `n = 128` (output_dim)
- `k = 128` (hidden_size)
- `num_group = 4`
- `kTileN = 128`, `kTileK = 128`

**步骤 1: Host 端创建 W tensor**
```cpp
// group_gemm_pertensor_fp8.cu 第 28-32 行
auto W = make_tensor(make_gmem_ptr(w_ptr),
                     make_shape(128, 128, 4), 
                     make_stride(128, Int<1>{}, 128*128));
```

**步骤 2: 创建 TMA copy（config.h 第 93 行）**
```cpp
auto tma_w = make_tma_copy(SM90_TMA_LOAD{}, w, take<0, 2>(SLayoutW{}));
// copy box shape: (128, 128) ← 二维！
```

**步骤 3: Device 端使用（kernels.cuh）**
```cpp
// 创建全局三维 tensor
auto gB = tma_b.get_tma_tensor(make_shape(128, 128, 4));

// 分区得到四维 tile tensor
auto tBg = btma_b.partition_S(gB);  // (TMA, TMA_N, TMA_K, 4)

// 选择 group 0 的 tile (0, 0)
tBg(_, 0, 0, 0)  ← 对应 W[0:128, 0:128, 0]

// 选择 group 1 的 tile (0, 0)  
tBg(_, 0, 0, 1)  ← 对应 W[0:128, 0:128, 1]
```

---

### 第六层：TMA Descriptor 的样子（根据用户笔记推断）

根据用户分享的 TMA 理解笔记，我们可以推断：

**TMA descriptor 包含：**
1. **全局内存地址**：指向 W 的起始位置 `w_ptr`
2. **全局 tensor shape**：`(n, k, num_group)` ← 三维！
3. **全局 tensor stride**：`(k, 1, n*k)` ← 三维！
4. **Copy box shape**：`(kTileN, kTileK)` ← 二维！

**关键点：**
- TMA descriptor 可以描述**任意维度**的全局 tensor（最多 5 维，从 tma.cuh 第 11 行可以看出）
- 但 **copy box 只需要描述每次拷贝的子区域形状**
- 通过坐标系统来选择具体的子区域

**三维坐标系统的工作原理：**

根据用户笔记：
```
普通 tensor: layout function 输出标量（内存偏移）
  Tensor(1, 2) = 1 * stride0 + 2 * stride1

TMA tensor: layout function 输出向量（坐标）
  Tensor(1, 2, 3) = (1, 2, 3) ← 三维坐标！
```

TMA 硬件根据：
1. **起始坐标**：`(itile_n * kTileN, itile_k * kTileK, igroup)`
2. **Box 维度**：`(kTileN, kTileK)` ← 注意！这里只需要两维！
3. **全局 tensor 的 stride**：`(k, 1, n*k)`

来确定要拷贝的内存区域。

**为什么 box 维度只需要两维？**
因为我们选择的是：
- 第 `igroup` 个 group（固定一个 group）
- 在该 group 内，拷贝 `(kTileN, kTileK)` 的二维 tile

也就是说，**第三维（num_group）被固定为某个具体的 igroup，所以 copy box 只需要描述前两维！**

---

### 总结

| 问题 | 答案 |
|------|------|
| W 的全局 tensor 为什么是三维的？ | 因为所有 group 的权重连续存储，形状为 `(n, k, num_group)`，stride 为 `(k, 1, n*k)` |
| copy box 为什么是二维的？ | 因为每次 TMA 只拷贝一个 tile 的数据 `(kTileN, kTileK)`，igroup 通过索引选择，不是通过 copy box |
| 如何选择不同的 group？ | 通过四维索引 `tBg(_, itile_n, itile_k, igroup)` 中的 `igroup` 来选择 |
| TMA descriptor 是什么样子？ | 包含三维全局 tensor 的 shape/stride，以及二维 copy box 的 shape |
| 为什么 W 可以共享 TMA descriptor？ | 因为所有 group 的权重在同一个连续内存区域中，可以通过三维坐标系统访问 |
| 为什么 X 不行？ | 因为每个 group 的 X 数据在全局内存中不连续，需要独立的 TMA descriptor |

**核心思想：**
- **全局 tensor 形状** = 完整的数据布局（可以是多维）
- **Copy box 形状** = 每次搬运的子区域形状（通常是二维）
- **索引系统** = 选择具体子区域的坐标

这个设计展示了 CuTe 和 TMA 的强大之处：**通过灵活的多维坐标系统，可以用一个 TMA descriptor 高效地访问三维 tensor 中的任意二维切片！**

---

## TMA Copy Box 维度选择机制详解

### 用户的深刻问题

用户提出了一个非常深刻的问题：

> 我们拷贝的形状是二维，而坐标是三维的。现在我们要 copy 一个二维的形状，那么我们至少需要在三维之中选择两维来进行 copy 计算吧。那么为什么我们是选择了前两个维度，而不是一维度和三维度进行 copy 呢？这是由什么决定的呢？
>
> 一个例子:如果我们在构建 tma tiled copy 所使用的 gmem tensor 形状是 (num_Group, n, k) 而不是 (n, k, num_group)，我们在进行 copy(tma) 的时候，是不是这个 copy box 就会沿着 num_group & n 维度进行 copy 了？

---

### 第一层：答案不是"选择"，而是"映射"

**关键发现：不是我们主动"选择"前两维，而是 slayout（shared memory layout）的形状决定了哪些 gmem 维度会被映射到 TMA copy box！**

让我通过分析 `construct_tma_gbasis` 函数（copy_traits_sm90_tma.hpp 第 690 行）来解释这个机制。

---

### 第二层：construct_tma_gbasis 的核心流程

整个机制的核心在 `construct_tma_gbasis` 函数中：

```cpp
// 第 721 行：获取 smem layout 的逆布局
// smem idx -> smem coord
auto inv_smem_layout = right_inverse(get_nonswizzle_portion(slayout));

// 第 725 行：组合 cta_v_map 和 inv_smem_layout
// 关键！这一步决定了哪些 gmem 维度被映射到 smem
auto sidx2gmode_full = coalesce(composition(cta_v_map, inv_smem_layout));
//                      ↑               ↑
//              smem coord ->     smem idx ->
//              gmem mode          smem coord
//
//              结果：smem idx -> gmem mode
```

**这里的关键是 `cta_v_map`！**

`cta_v_map` 是从 `cta_tiler`（我们传给 `make_tma_copy` 的第三个参数）来的：

```cpp
// make_tma_copy 调用中
auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler);
//                                                     ↑
//                                              这就是我们传入的 slayout！
```

**也就是说：我们传入的 `slayout`（shared memory layout）的形状决定了哪些 gmem 维度会被映射到 TMA copy box！**

---

### 第三层：具体例子分析

让我们用 hpc-ops 中的实际例子来说明：

**情况 1：hpc-ops 的实际设计**

```cpp
// 1. Global memory tensor W
shape: (n, k, num_group)  ← 三维
stride: (k, 1, n*k)

// 2. Shared memory layout SLayoutW (我们传入 make_tma_copy 的)
shape: (kTileN, kTileK, kStage)  ← 注意！前两维是 (kTileN, kTileK)
//                                   第三维 kStage 是双缓冲用的

// 3. 因为 SLayoutW 的前两维是 (kTileN, kTileK)
//    所以映射到 gmem 的前两维 (n, k)
//    这就是为什么 copy box 沿着 n 和 k 维度！
```

**情况 2：如果 gmem tensor 形状是 (num_group, n, k)**

```cpp
// 假设我们这样定义：
shape: (num_group, n, k)  ← 注意顺序变了！
stride: (n*k, k, 1)

// 如果 SLayoutW 仍然是 (kTileN, kTileK, kStage)
// 那么映射关系就会变成：
//   slayout 第 0 维 (kTileN) → gmem 第 0 维 (num_group)
//   slayout 第 1 维 (kTileK) → gmem 第 1 维 (n)
//
// 这样 copy box 就会沿着 num_group 和 n 维度！
// 这正是用户猜测的！
```

---

### 第四层：关键代码细节

让我们看 `construct_tma_gbasis` 中的关键步骤：

**步骤 1：构建 smem idx -> gmem mode 的映射**

```cpp
// 第 721 行：smem idx -> smem coord
auto inv_smem_layout = right_inverse(get_nonswizzle_portion(slayout));

// 第 725 行：smem coord -> gmem mode
auto sidx2gmode_full = coalesce(composition(cta_v_map, inv_smem_layout));
```

**步骤 2：提取有效维度**

```cpp
// 第 737-744 行：找出哪些 gmem 维度被映射到了
auto smem_rank = find_if(stride(sidx2gmode_full), [](auto e) {
  [[maybe_unused]] auto v = basis_value(e);
  return not is_constant<1,decltype(v)>{};
});

auto sidx2gmode = take<0,smem_rank>(sidx2gmode_full);
//              ↑
//         只取被映射到的 gmem 维度！
```

**步骤 3：构建 TMA basis**

```cpp
// 第 760 行：coalesce_256 合并维度（最大 256）
auto tma_gstride = coalesce_256(tile_gstride);
```

---

### 第五层：coalesce_256 的作用

`coalesce_256` 函数（第 681 行）很有意思，它会尝试合并维度，但单个维度最大不超过 256（TMA 硬件限制）：

```cpp
// 如果可以合并且不超过 256，就合并
if (s0 * s1 <= 256) {
  merge them;
} else {
  keep as separate dimensions;
}
```

**但这不是维度选择的关键，关键还是 slayout 的形状！**

---

### 第六层：fill_tma_gmem_shape_stride 的作用

最后看 `fill_tma_gmem_shape_stride` 函数（第 819 行）：

```cpp
// 第 834-861 行：遍历每个 TMA 模式
for_each(make_seq<tma_rank>{}, [&](auto i) {
  constexpr int tma_i_rank = decltype(rank<i>(tma_gbasis_stride))::value;
  if constexpr (tma_i_rank == 1) {
    // 简单情况：这个 TMA 模式对应单个 gmem 模式
    auto ej = unwrap(get<i>(tma_gbasis_stride));
    gmem_prob_shape[i]  = basis_get(ej, gmem_shape);
    gmem_prob_stride[i] = basis_get(ej, gmem_stride);
  } else {
    // 复杂情况：这个 TMA 模式对应多个 gmem 模式
    // 需要用 GCD 来计算 shape 和 stride
    ...
  }
});
```

这里的 `tma_gbasis_stride` 就是之前 `construct_tma_gbasis` 的结果，它决定了 TMA 模式如何映射到 gmem 模式。

---

### 第七层：最终答案总结

| 问题 | 答案 |
|------|------|
| 为什么选择前两维？ | **不是我们选择的，而是 slayout 的形状决定的！** slayout 的前两维 (kTileN, kTileK) 映射到 gmem 的前两维 (n, k) |
| 由什么决定？ | 由传给 `make_tma_copy` 的 `slayout` 参数决定！slayout 的形状决定了哪些 gmem 维度被映射 |
| 如果 gmem 形状是 (num_group, n, k) 会怎样？ | **是的！** 如果 gmem 维度顺序变了，但 slayout 还是 (kTileN, kTileK)，那么 copy box 就会沿着 num_group 和 n 维度 |
| 如何验证？ | 改变 gmem tensor 的维度顺序，或者改变 slayout 的维度顺序，看看 TMA descriptor 如何变化 |

---

### 第八层：核心思想

**整个机制的核心思想：**

1. **Shared memory layout 是"主"**：slayout 的形状决定了我们要如何访问数据
2. **Global memory tensor 是"从"**：gtensor 提供数据，但访问模式由 slayout 决定
3. **Basis 映射是桥梁**：`construct_tma_gbasis` 构建了 slayout 维度到 gtensor 维度的映射
4. **不是"选择"维度，而是"映射"维度**：slayout 的第 i 维映射到 gtensor 的哪个维度，取决于 basis 映射

**换句话说：**
- 不是我们说"我要 copy 前两维"
- 而是我们说"我要 copy slayout 形状的数据"
- CuTe 自动找出 slayout 维度和 gtensor 维度之间的映射关系

---

### 第九层：实际验证建议

如果想验证这个理解，可以做以下实验：

**实验 1：改变 gmem 维度顺序**
```cpp
// 原来的
auto W = make_tensor(w_ptr, make_shape(n, k, num_group), make_stride(k, 1, n*k));

// 改成
auto W = make_tensor(w_ptr, make_shape(num_group, n, k), make_stride(n*k, k, 1));
//                                                 ↑
//                                          维度顺序变了！
```
看看 TMA copy 是否会沿着 num_group 和 n 维度。

**实验 2：改变 slayout 维度顺序**
```cpp
// 原来的
using SLayoutWw = decltype(tile_to_shape(SLayoutWAtom{}, 
                          make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

// 改成
using SLayoutWw = decltype(tile_to_shape(SLayoutWAtom{}, 
                          make_shape(Int<kStage>{}, Int<kTileN>{}, Int<kTileK>{})));
//                                                 ↑
//                                          维度顺序变了！
```
看看 TMA copy 是否会沿着不同的 gmem 维度。

---

## Scale for DeQuantization 详解

### Pertensor 模式

**数据格式**：
```cpp
float *y_scale_ptr  // 形状: [num_group]
// y_scale_ptr[igroup] 是第 igroup 个 group 的 scale 值
```

**实现非常简洁**（kernels.cuh 第 347, 374 行）：
```cpp
// 步骤 1: 获取当前 group 的 scale
float scale = yscale_ptr[igroup];

// 步骤 2: 创建累积寄存器
auto tDr = make_tensor_like(tCr);
clear(tDr);

// 步骤 3: K 维度累加循环
for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
  // ... GMMA 计算 ...

  // 步骤 4: 反量化并累加
#pragma unroll
  for (int i = 0; i < size(tCr); ++i) {
    tDr(i) = tCr(i) * scale + tDr(i);
  }
}

// 步骤 5: 转换为 bfloat16
auto tCrh = make_tensor_like<cute::bfloat16_t>(tCr);
#pragma unroll
  for (int i = 0; i < size(tCr); ++i) {
    tCrh(i) = (Tout)(tDr(i));
}
```

**为什么 Pertensor 这么简单？**

因为每个 group 只有一个 scale 值，所有线程在处理同一个 group 时使用相同的 scale。不需要复杂的 layout algebra 来查找每个元素的对应 scale。

---

### Blockwise 模式

**数据格式**：

```cpp
// xscale: 输入激活的 scale
// 形状: [num_block_k, m_pad]
// 其中 m_pad 是 padding 后的维度（通常对齐到 tileM）
// 内存布局: (num_block_k, m_pad) with stride (m_pad, 1)
float *xscale_ptr

// wscale: 权重的 scale
// 形状: [num_block_n, num_block_k_pad4, num_group]
// 注意: num_block_k_pad4 是 padding 到 4 的倍数（TMA 硬件要求）
// 内存布局: (num_block_n, num_block_k_pad4, num_group) with stride (num_block_k_pad4, 1, num_block_n * num_block_k_pad4)
float *wscale_ptr
```

**为什么需要 reformat_x_scale？**

原始的 xscale 布局是 `[m, n]`（与激活数据相同），但 blockwise kernel 需要的格式是 `[num_block_k, m_pad]`。这两个布局之间存在转置和对齐关系，所以需要 `reformat_x_scale_kernel` 进行转换。

**reformat_x_scale_kernel 的工作原理**（`group_gemm_blockwise_fp8.cu` 第 19-104 行）：

```cpp
// 输入: xscale_ptr [m, n]
// 输出: output_ptr [num_block_k, m_pad]

// 关键步骤：转置 + 对齐到 tileM
// 1. 计算每个 group 的起始位置
int src_global_row = cu_seqlens_ptr[iblock];  // 原 m 维度的起始位置
int dst_global_col = 计算转置后的列位置;       // 新的列位置

// 2. 转置拷贝
for (int i = 0; i < valid_seq_num; i += TM) {
  // 读取: xscale_ptr[src_row * n + src_col]
  auto r = load<float, kElementsPerThread>(xscale_ptr_ptr + src_row * n + src_col);

  // 写入: output_ptr[dst_row * m_pad + dst_col]
  store<float, kElementsPerThread>(output_ptr + dst_row * m_pad + dst_col, r);
}
```

**转置的原因**：为了优化访问模式。重新格式化后的 xscale 在 kernel 中可以更高效地通过 TMA 加载。

---

#### Blockwise Kernel 中的 Scale 使用

在 `kernels.cuh` 第 670-700 行：

```cpp
// Shared memory 中的 scale 张量布局
// SLayoutXS: (kStage, kTileS) with stride (kTileS, 1)
// SLayoutWS: (kStage, kTileS) with stride (kTileS, 1)
auto sAS = make_tensor(make_smem_ptr(shm_as), SLayoutAS{});
auto sBS = make_tensor(make_smem_ptr(shm_bs), SLayoutBS{});

// K 维度累加循环
for (int itile_k = 0; itile_k < ntiletile_k; ++itile_k) {
  // 等待数据加载完成
  wait_barrier(readable[ismem_read], phase);

  // 步骤 1: 从 shared memory 读取 wscale
  // 注意: itile_k % 4 是因为 wscale 的 K 维度 padding 到 4 的倍数
  float wscale = sBS(ismem_read, itile_k % 4);

  // 步骤 2: 计算 xscale * wscale 得到中间 scale tCS
  // 这是一个逐元素的乘法，为每个 N 维度准备独立的 scale
  float tCS[kN];
#pragma unroll
  for (int in = 0; in < kN; in++) {
    tCS[in] = sAS(ismem_read, get<1>(tI_mn(0, in))) * wscale;
  }

  // 步骤 3: GMMA 计算（不包含 scale）
  tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
  for (int ik = 0; ik < size<2>(tAr); ++ik) {
    cute::gemm(tiled_mma, tBr, tAr, tCr);
  }

  // 步骤 4: 使用 tCS 进行反量化
  auto tDr_mn = retile_fragment(tDr);
#pragma unroll
  for (int in = 0; in < kN; in++) {
    float yscale = tCS[in];  // 每个 N 维度有独立的 scale
#pragma unroll
    for (int im = 0; im < kM; im++) {
      tDr_mn(im, in) = tCr_mn(im, in) * yscale + tDr_mn(im, in);
    }
  }
}
```

**关键点分析：**

1. **wscale 的选择**：`sBS(ismem_read, itile_k % 4)`
   - 每个 stage 都有对应的 wscale 数据（通过 TMA 预加载）
   - `itile_k % 4` 是因为 wscale 的 K 维度 padding 到 4 的倍数

2. **tCS 的计算**：`tCS[in] = sAS(...) * wscale`
   - `sAS` 是 xscale，从 shared memory 读取
   - `sAS(ismem_read, get<1>(tI_mn(0, in)))` 中的 `get<1>(tI_mn(0, in))` 是通过 layout algebra 找到对应 N 维度的 xscale 值
   - 结果 `tCS` 是每个 N 维度的独立 scale（因为 xscale 和 wscale 都是 blockwise 的）

3. **反量化的应用**：`tCr_mn(im, in) * yscale + tDr_mn(im, in)`
   - `tCr_mn` 是当前 K 维度的 GMMA 累积结果
   - `yscale = tCS[in]` 是对应 N 维度的 scale
   - 累加到 `tDr_mn` 中

---

### 两种模式的对比

| 特性 | Pertensor | Blockwise |
|------|-----------|-----------|
| **Scale 数量** | num_group | num_block_n * num_block_k * num_group |
| **Scale 张量** | y_scale [num_group] | xscale [num_block_k, m_pad], wscale [num_block_n, num_block_k_pad4, num_group] |
| **查找复杂度** | O(1) - 直接索引 | 需要 layout algebra 和 shared memory 访问 |
| **内存开销** | 小 | 较大（需要额外的 scale 张量） |
| **精度** | 每个 group 一个 scale（可能不够精确） | 每个 tile 独立 scale（更精确） |
| **实现复杂度** | 简单 | 复杂（需要 reformat_x_scale + layout algebra） |

---

### 数学公式对比

```
Pertensor:
  Y = (X @ W^T) * scale
  其中 scale 对整个 group 是常数

Blockwise:
  Y[i,j] = Σ_k (X[i,k] / xscale[k]) * (W[j,k] / wscale[j,k])
          = Σ_k X[i,k] * W[j,k] / (xscale[k] * wscale[j,k])
  其中 xscale[k] 是每个 K 维度的 scale，wscale[j,k] 是每个 (N,K) 块的 scale
  最终的 scale 是 xscale * wscale 的乘积
```

两种模式的选择取决于具体的量化策略和精度需求。Pertensor 适合简单的应用场景，Blockwise 适合需要更高精度的场景。

---
*Update this file after every 2 view/browser/search operations*
*This prevents visual information from being lost*
