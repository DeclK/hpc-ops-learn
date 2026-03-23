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

TODO

## Transposed MMA Tiler

TODO

## Scheduler for Group Gemm

TODO

