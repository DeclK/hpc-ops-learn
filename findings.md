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

**为什么小矩阵用线性扫描？**
- 小矩阵意味着 num_group 不大
- 线性扫描实现简单，指令数少
- 缓存友好，因为 tiles_ptr 是连续访问的

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

**为什么大矩阵用二分查找？**
- 大矩阵意味着 num_group 可能很大
- 二分查找时间复杂度 O(log n)，远优于线性扫描 O(n)
- cu_tiles_ptr 是有序的（累积和），天然适合二分查找

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

在主 kernel 调用处（`group_gemm_pertensor_fp8.cu`），根据问题规模选择模式：

```cpp
if (k <= 1024 || n <= 1024) {
  // Horizontal 模式：小矩阵，线性扫描
  group_gemm_pertensor_fp8_kernel<Config, TmaX, TmaW, TmaY, true>
      <<<num_block, kThreadPerBlock, shm_size, stream>>>(...);
} else {
  // Vertical 模式：大矩阵，二分查找
  group_gemm_pertensor_fp8_kernel<Config, TmaX, TmaW, TmaY, false>
      <<<num_block, kThreadPerBlock, shm_size, stream>>>(...);
}
```

### 总结

Scheduler 的设计体现了**实用主义**：
- 小矩阵：简单直接的线性扫描，开销小
- 大矩阵：高效的二分查找，扩展性好
- 通过编译期条件（`if constexpr`）选择实现，无运行时开销

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
*Update this file after every 2 view/browser/search operations*
*This prevents visual information from being lost*
