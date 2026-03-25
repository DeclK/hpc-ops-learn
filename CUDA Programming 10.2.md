# CUDA Programming 10.2

这是在 Blackwell 和 Hopper 之间的一个笔记。其目的在于学习 [Tencent/hpc-ops: High Performance LLM Inference Operator Library](https://github.com/Tencent/hpc-ops) 对于 fp8 kernel 的高效实现

在之前我已经对其中的内容有了大致的了解。不过由于一些事情搁置了，现在又有时间进行完整的整理。本次整理我希望借助 AI agent like claude code 的力量，完成更清晰的理解

先拟出一个大纲，列出自己想要学习的内容：

1. fp8 gemm vs fp16 gemm，他们之间除了精度外，还是否有其他的重要区别
2. group gemm 中对 tma 的应用技巧
3. 如何理解在 gemm 当中把 AB 矩阵反过来的操作
4. 如何设计简单的 scheduler 来统一应对 group gemm

从 hpc-ops 的代码量上来看，kernel 的核心代码也非常精简，~300 lines of code。不过如果我们直接暴力从头去解读其中的代码，或许并不是一个好的事情。我们还是从 top to bottom 的角度，从大的算法图景开始，理解这个 group gemm algorithm 到底做了什么事情，然后根据这个 overall algorithm 进行展开，再去看代码中的细节实现，这样能够理解更轻松

## Group GEMM 详解

### What's A Group Gemm?

要理解 Group GEMM，我们先从普通的 GEMM（通用矩阵乘法）开始。

**普通 GEMM** 计算的是：
```
Y = X * W^T
```
其中：
- X 形状为 [M, K]（输入激活）
- W 形状为 [N, K]（权重）
- Y 形状为 [M, N]（输出）

**Group GEMM** 是 GEMM 的扩展，它在一次操作中执行多个独立的矩阵乘法。可以把它想象成"批量处理"不同组的矩阵乘法。

在 Group GEMM 中：
- 我们有 `num_group` 个独立的权重矩阵 `W_0, W_1, ..., W_{num_group-1}`
- 输入 X 被分割成 `num_group` 个连续的组 `X_0, X_1, ..., X_{num_group-1}`
- 每个 `X_i` 与对应的 `W_i` 相乘
- 所有结果拼接起来得到最终输出 `Y`

使用简介的代码语言来表示

```python
# X (M, K)
# W (num_group, N, K)
# Y (M, N)
# M is the length of all tokens from different groups
Y[start_i : end_i, :] = X[start_i : end_i, :] * W[i, :, :]^T

# A torch version
def naive_group_gemm_pertensor_fp8(x, w, seqlens, cu_seqlens, scale):
    # 步骤 1: 获取张量形状
    m, k = x.shape           # m = total_seq, k = hidden_size
    num_group, n, _ = w.shape  # n = output_dim

    # 步骤 2: 初始化输出张量
    y = torch.zeros((m, n), dtype=torch.bfloat16, device=x.device)

    # 步骤 3: 遍历每个组，执行独立的矩阵乘法
    start_idx = 0
    for i in range(num_group):
        # 获取当前组的起始和结束位置
        start_idx = int(cu_seqlens[i].item())
        end_idx = int(start_idx + seqlens[i].item())

        # 如果该组没有数据，跳过
        if seqlens[i].item() == 0:
            continue

        # 提取当前组的输入和权重
        x_group = x[start_idx:end_idx]  # 形状: [seqlens[i], k]
        w_group = w[i]                   # 形状: [n, k]

        # 执行矩阵乘法（使用缩放的 FP8 运算）
        y_group = torch._scaled_mm(
            x_group, w_group.t(),        # w_group.t() 形状: [k, n]
            scale_a=scale, scale_b=scale,
            bias=None, out_dtype=torch.bfloat16
        )

        # 将结果写入输出的对应位置
        y[start_idx:end_idx] = y_group

    return y
```

### Why Group Gemm?

Group GEMM 的主要应用场景是**混合专家模型（Mixture of Experts, MoE）**的推理。如果不使用 Group GEMM，我们需要把输入 X 按照专家分组拆开，对每个专家分别调用一次 GEMM，最后再把结果拼回去

这样做的问题：
- **多次 kernel launch 开销**：每个专家都需要一次独立的 kernel launch，launch 本身有固定开销
- **分散的内存访问**：每个小 kernel 独立访问内存，难以形成高效的流水线访问模式。例如：前一个 group 的最后一部分数据在进行计算时，就可以开始预取下一个 group 的数据了，但独立的 kernel launch 无法做到这一点。另外，GroupGemm 最后也不需要再对各个 group 的计算结果进行拼接，减少数据读写

这么看来 GroupGemm 的优势就明显了：一次 kernel launch + 高效的内存流水线访问模式

## GroupGemm in hpc-ops Overview

本次学习的 kernel 代码在 `src/group_gemm/kernels.cuh`，这部分代码包含了对 group gemm pertensor & blockwise 进行了优化实现。我们先从简单的 pertensor group gemm 开始，当我们吃透了这部分代码过后，再来看下 blockwise 的实现有什么细节上的更改

接下来是针对 pertensor group gemm 算法的精髓总结：

1. **Warp Specialization**: 384 线程 = 256 线程 (数学计算 warpgroup) + 128 线程 (数据加载 warpgroup)
2. **Producer-Consumer Model**: 使用 mbarrier 在阶段之间进行同步和流水线控制
3. **TMA for Groups**: 为每个 group 预配置独立的 TMA descriptor，实现高效数据搬运（而不是所有的 group 使用一个 tma descriptor）
4. **Adaptive Tile Sizing (自适应 Tile 大小)**: 根据每组的平均序列长度选择 kTileM (16/32/48/64)
5. **Dual Scheduling Modes (双调度模式)**: Horizontal 模式（小矩阵用线性扫描）vs Vertical 模式（大矩阵用二分查找）

前两个算法算是老生常谈了，是 GEMM 算法中的基本。后面三个优化就是 hpc-ops 中的核心。其中我想提前提下第 4 点，对于 Tile Size Configuration，kernel 会根据 `num_seq_per_group_avg`（每组平均序列长度）选择不同的 tile 大小：

| avg_seqlen | kTileM | kTileN | kTileK | kStage |
|------------|--------|--------|--------|--------|
| ≤16        | 16     | 128    | 128    | 8      |
| ≤32        | 32     | 128    | 128    | 8      |
| ≤48        | 48     | 128    | 128    | 8      |
| >48        | 64     | 128    | 128    | 8      |

这样可以确保在不同的序列长度分布下都有良好的硬件利用率。这里的 Tile 看上去很奇怪，一般来说 Tensor Core 都是固定矩阵乘当中的 M 维度，i.e. `kTileM = 128`，而在这里确是固定了 `kTileN = 128`，这是因为 hpc-ops 对于 mma 进行了转置处理，这是一个非常巧妙的用法。这样的转置一下子就让 M 维度的粒度变得非常细，对于小 M 的场景非常友好

### Pseudocode with Producer-Consumer Structure

下面是一个简洁的伪代码，帮助我们抓住整体的算法流程。其中隐藏了对 schduler & tma & mbarrier & mma 等模块的大量细节，但不妨碍我们理解这个 producer-consumer gemm 的核心思想

```cpp
__global__ void group_gemm_pertensor_fp8_kernel(...) {

  // ===== PRELOGUE =====
  int idx = threadIdx.x;
  bool is_producer = (idx >= 256);  // 128 threads for load
  bool is_consumer = (idx < 256);    // 256 threads for math

  extern __shared__ uint8_t shm_data[];
  auto* shm_a = (Tin*)shm_data;          // [kTileM, kTileK, kStage]
  auto* shm_b = shm_a + ...;              // [kTileN, kTileK, kStage]
  int* shm_tiles = (int*)(shm_data + ...); // For scheduler

  // Initialize mbarriers (producer-consumer sync)
  if (is_leader) {
    for (int s = 0; s < kStage; s++) {
      initialize_barrier(readable[s], 1);  // Consumer waits on this
      initialize_barrier(writable[s], 1);  // Producer waits on this
    }
  }
  // Load scheduler metadata to shared memory
  for (int i = idx; i < num_group; i += 384)
    shm_tiles[i] = tiles_ptr[i];
  __syncthreads();

  // ===== MAINLOOP: Producer-Consumer Pipeline =====
  int phase = 0;

  if (is_producer && is_leader_in_load) {
    // PRODUCER: Load data via TMA
    int s_write = 0;  // Current stage to write
    while (true) {
      // SCHEDULER: Get next tile (igroup, itile_m, itile_n)
      if (!get_next_tile(shm_tiles, iblock, ...)) break;

      for (int k = 0; k < ntile_k; k++) {
        // MBARRIER: Wait for consumer to release stage
        wait_barrier(writable[s_write], phase);

        // TMA load X and W tiles to shared memory
        tma_copy(shm_a[_, _, s_write], global_X[igroup, itile_m, k]);
        tma_copy(shm_b[_, _, s_write], global_W[igroup, itile_n, k]);

        // MBARRIER: Signal consumer data is ready
        set_barrier_transaction_bytes(readable[s_write], ...);

        // Circular stage buffer
        s_write = (s_write + 1) % kStage;
        if (s_write == 0) phase ^= 1;
      }
    }
  }

  if (is_consumer) {
    // CONSUMER: Compute via GMMA
    int s_read = 0;  // Current stage to read
    while (true) {
      // SCHEDULER: Same as producer, get next tile
      if (!get_next_tile(shm_tiles, iblock, ...)) break;

      for (int k = 0; k < ntile_k; k++) {
        // MBARRIER: Wait for producer to fill stage
        wait_barrier(readable[s_read], phase);

        // GMMA: Compute on shared memory data
        gemm(shm_a[_, _, s_read], shm_b[_, _, s_read], accum);

        // MBARRIER: Signal producer stage is consumed
        if (is_leader_in_warpgroup)
          arrive_barrier(writable[s_read]);

        // Circular stage buffer
        s_read = (s_read + 1) % kStage;
        if (s_read == 0) phase ^= 1;
      }

      // ===== EPILOGUE =====
      cast_to_bf16(accum, output, pertensor_scale);
      tma_store(output, global_Y[igroup, itile_m, itile_n]);
    }
  }
}
```

下面我们将对这些核心优化进行逐个分析，把他们的原理和实现解释清楚。

## TMA for Group Gemm

在 hpc-ops 的 Group GEMM 实现中，TMA（Tensor Memory Accelerator）的使用非常巧妙。不同于普通 GEMM 中使用单一 TMA descriptor，这里为每个 group 预配置了独立的 TMA descriptor。这一节我们来详细分析这个设计。

### 为什么需要为每个 group 配置独立的 TMA descriptor？

核心原因就是**每个 group 的数据位置不同**：X 张量在全局内存中是连续存储的 `[total_seq, k]`，第 `igroup` 个 group 的起始位置是 `x_ptr + cu_seqlens[igroup] * k`

如果我们只有一个 tma descriptor，则只能按照这个 tma 的 gmem ptr + copy box offset 的方式进行 copy。对于 group gemm 来说，每个 group 的数据起始位置不可能都正好在 copy box offset 中。因此我们有两个选项：1. 把原始数据 Padding 为 copy box aligned 结构，这样每一个 group 都能和 copy box offset 对齐；2. 给每一个 group 都配置一个独立的 tma descriptor，这样每个 group 的数据都能按照自己的起始位置进行 copy

### Kernel Launch 配置

```cpp
constexpr int kGroupPerThread = 8;
constexpr int kThreadPerBlock = 32;
kernels::update_grouped_tma<...>
    <<<num_group + 1, kThreadPerBlock, 0, stream>>>(...);
```

- **Grid/Block 配置**：`num_group + 1` 个 block，每个 block 32 个线程
- **Block 分工**：
  - Block `0 ~ num_group-1`：每个 block 处理一个 group，更新该 group 的 X 和 Y 的 TMA descriptor
  - Block `num_group`：计算所有 group 的 tile 统计信息

### Kernel 参数详解

| 参数 | 类型 | 说明 |
|------|------|------|
| `td_xy` | `vec_t<TmaDescriptor, 2>` | **模板 TMA descriptor**，在 Host 端预配置好，包含正确的 stride 等信息 |
| `tma_xy` | `TmaDescriptor*` | **输出数组**，大小 `num_group * 2`，`tma_xy[igroup*2+0]` 是 X 的 desc，`tma_xy[igroup*2+1]` 是 Y 的 desc |
| `x_ptr` / `y_ptr` | `const Tin*` / `const Tout*` | X 和 Y 张量的全局指针 |
| `seqlens_ptr` | `const int*` | 每个 group 的 seqlen，形状 `[num_group]` |
| `cu_seqlens_ptr` | `const int*` | 累积 seqlen，形状 `[num_group + 1]` |
| `tiles_ptr` | `int*` | **输出**：每个 group 的 tile 数，形状 `[num_group]` |
| `cu_tiles_ptr` | `int*` | **输出**：累积 tile 数，形状 `[num_group + 1]` |
| `num_group` / `m` / `n` / `k` | `int` | 问题维度 |

(补充) BlockScan 的使用

`cub::BlockScan` 是一个并行前缀和计算原语。这里使用的是 **Exclusive Sum Scan**：

Exclusive Scan
是一种并行计算原语，对数组进行前缀和计算，但每个位置的结果是该位置之前所有元素的和。

```txt
示例：
输入:  [a, b, c, d]
输出:  [0, a, a+b, a+b+c]  ← exclusive sum
总和:  a+b+c+d               ← block_aggregate

对比 Inclusive Scan：
输入:  [a, b, c, d]
输出:  [a, a+b, a+b+c, a+b+c+d]  ← inclusive sum
```

可以从 hpc-ops 中的代码代表了 block scan 的一般用法

```cpp
// 第 88 行：定义 BlockScan 类型
using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
// - 模板参数 1: int - 扫描的数据类型
// - 模板参数 2: kThreadPerBlock = 32 - block 中的线程数

// 第 89 行：分配共享内存
__shared__ typename BlockScan::TempStorage temp_storage;
// - TempStorage 是 cub 内部定义的结构体
// - 需要共享内存来协调线程间的通信
// - 大小由 cub 自动计算

// 第 90 行：用于返回总和
int block_aggregate;

// 第 91 行：执行 Exclusive Sum Scan
BlockScan(temp_storage).ExclusiveSum(tiles, tiles, block_aggregate);
// 参数说明：
// - tiles (输入): 每个线程贡献的数据数组
// - tiles (输出): 扫描后的结果（原地修改）
// - block_aggregate: 返回整个 block 的总和
```

### TMA Descriptor 更新

当 `blockIdx.x < num_group` 时，为该 group 更新 TMA descriptor。注意，我们**不是在 device 端从头创建 TMA descriptor**，而是：
1. **Host 端创建模板**：`td_xy` 包含了正确的 stride、tile size 等配置
2. **Device 端只更新必要字段**：
   - **全局内存地址**：指向该 group 数据的起始位置
   - **Shape**：根据该 group 的 seqlen 设置

**为什么 stride 不需要更新？**

| 张量 | Shape | Stride | 是否变化 |
|------|-------|--------|----------|
| X | `[num_seq, k]` | `(k, 1)` | **k 固定**，所有 group 一样 |
| Y | `[n, num_seq]` | `(1, n)` | **n 固定**，所有 group 一样 |

- `k` 是隐藏层维度（hidden_size），对所有 group 相同
- `n` 是输出维度（output_dim），对所有 group 相同
- 只有 `num_seq` 变化（`seqlens_ptr[igroup]`）

#### `tma_desc_commit_group` 的作用

```cpp
if (cute::elect_one_sync()) {
  cute::tma_desc_commit_group();
  cute::tma_desc_wait_group();
}
```

在我们的代码中，**同一个 warp 中的不同线程在修改不同的 TMA descriptor**：
- 线程 0 更新 `smem_tma_desc[0]` (X)
- 线程 1 更新 `smem_tma_desc[1]` (Y)

这时候需要用 `tma_desc_commit_group` 来确保 warp 中所有线程对 TMA descriptor 的修改都完成并且可见。这里的 PTX 和 `tma_store_fence` 是一样的，我们之前使用 `tma_store_fence` 是为了确保 tma store 操作必须要在 smem 写入完成之后。在这里起到同样的作用，因为我们之后要把修改好的 smem 内容写回到 gmem 中存储的 cuTensorMap 当中，必须要保证所有的 smem 写入完成才发起该操作。

#### `tma_descriptor_cp_fence_release` 的作用

这个函数做两件事（fused copy + fence）：
1. **Copy**：把 128 字节的 TMA descriptor 从 shared memory 拷贝到 global memory
2. **Fence**：带 `release` 语义的内存屏障。此屏障的作用是：确保之后使用 tma 的操作，都必须在该写入操作完成之后执行。可以想象为，这个 release fence 把之前的所有写代码都拦住了，编译器不可能把他们重排到这个 fence 之后。还有另一种带 `acquire` 语义的内存屏障，它会保证之后的所有读操作都必须在该读操作完成之后执行。这也是为什么 `acquire & release` 通常成对出现，我查阅了下 `tma_store_fence` 它到底属于 acquire 还是 release 呢？我认为答案应该是 both！我们既不能让 smem 写操作跨越该 fence，也不让 tma store 操作跨越该 fence

**与主 kernel 配对使用**：

Producer（update_grouped_tma）:
```cpp
tma_descriptor_cp_fence_release(tma_xy + i, smem_tma_desc[i]);
// "Release": 保证之前的所有写操作都可见
```

Consumer（主 kernel）:
```cpp
tma_descriptor_fence_acquire(td_xy + i);
// "Acquire": 保证之后的读操作能看到完整的 descriptor
```

#### `update_tma_gtensor` 的作用 

该 device function 是更新 tma descriptor 的核心。会从 gmem tensor 中提取 shape & stride & gmem ptr，然后把这些信息更新到 TMA descriptor 中。我一开始还有疑问：为什么一定要用 shared memory 创建 cuTensorMap？虽然我之前了解到 tma 存储的信息都是放在 smem 当中的，但是我们仍然可以把这些信息放到寄存器当中，然后修改，最后再存回 gmem 当中呀。后来 agent 了解到 `tma_descriptor_replace_shapes_in_shared_mem` 该 PTX 要求操作源必须在 shared memory 当中，所以必须使用 smem


### 总结

| 问题 | 答案 |
|------|------|
| 为什么需要 update_grouped_tma？ | 为每个 group 预配置 TMA descriptor，简化主 kernel 逻辑 |
| Launch 多少个 block？ | `num_group + 1` 个 |
| 每个 block 做什么？ | 前 num_group 个更新 TMA descriptor，最后一个计算 tile 统计 |
| 为什么用 BlockScan？ | 高效计算 Exclusive Sum，得到累积 tile 索引 |
| 为什么只更新 shape/addr？ | stride 对所有 group 都一样，不需要更新 |
| tma_desc_commit_group 作用？ | 同一个 warp 中多个线程修改 TMA descriptor 时确保一致性 |
| cp_fence_release 作用？ | Fused copy + release fence，与主 kernel 的 acquire 配对 |

## Transposed MMA Tiler

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

## Scheduler for Group Gemm

TODO

## Questions

1. hpc-ops 并没有使用 multicast 功能。由于该原因，hpc-ops 在 H100 上的性能就不如 DeepGemm。这在 [issue](https://github.com/Tencent/hpc-ops/issues/28) 当中有提到：For the H100, which has higher compute throughput but lower memory bandwidth, the pipeline places more emphasis on memory access patterns. 这说明在 roofline model 当中，H100 需要更大的 GEMM 计算来达到 compute bound，否则很容易就变得 memory bound。在 Thor 上更是如此，估计其 fp16 算力为 250TFLOPS，而其带宽只有 275GB/s，此使需要计算强度超过 930+ Flops/Byte 才能达到 compute bound。对于端侧来说，几乎大部分的 gemm 都达不到这个计算强度，i.e. 都是 memory bound kernel。而对于 H20 来说，其算力低，带宽大，计算强度拐点 37 Flops/Byte = (148 TFlops / 4000 GB/s) 几乎所有的算子都是 compute bound，所以打不打开 multicast 对性能没那么大影响
  