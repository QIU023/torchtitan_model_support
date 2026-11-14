# 读懂 trace：练习册（2026-11-10）

**"读懂"的标准：** 任意一个 rank 上随便框 1 毫秒，你能说出四件事：
- 它属于哪个 stage、哪个 micro-batch、前向还是反向；
- 是什么 kernel；
- 它受什么限制：计算、访存、启动开销，还是通信；
- 它在等谁。

整步还要能列出一本账：计算、各维度的暴露通信、空闲、气泡，加起来等于步时间。再预测改一个配置后这本账怎么变，用真机验证（`PRACTICE_PLAN.md`）。

下面的练习都用 11-08 的数据（`../results/20261108_h100_together/`，`raw/` 只在本机），先 `export PYTHONUTF8=1`。

## 0. trace 里有什么、没有什么

PyTorch profiler（Kineto）的 trace 到 kernel 级别，但看不到 kernel 内部：

| 有 | 没有（用什么补） |
|---|---|
| 每个 GPU kernel 的名字、开始时间、时长、stream | kernel 内部的硬件指标：实际带宽、Tensor Core 利用率、stall 原因（Nsight Compute） |
| 启动配置：grid、block、每线程寄存器、shared memory、估算 occupancy | Python 调用栈（这次没开 `with_stack`） |
| NCCL kernel 的 collective 名、元素个数、dtype、组大小、进程组哈希 | NCCL 选的通道、算法和协议（NCCL 的 INFO 日志，`TUNING` 子系统） |
| CPU 侧每个 aten 算子及其输入形状，以及它到所启动 kernel 的连线 | 机器层面的 CPU、OS、PCIe 活动（Nsight Systems） |
| GPU 标注：PP 调度的每个动作、KDA 各段、collective 名 | 整卡的频率、温度、功耗（DCGM 和 `monitor.py`，`gpu_by_cell.tsv`） |

三种文件：
- **合成 trace** `k3_*_measure_trace.json.gz`：16 个 rank，只有 GPU 侧和 `ProfilerStep`，看全局用；
- **单 rank 原始 trace** `raw/h100/manual/measure/<格>/tt/profiling/traces/iteration_5/rank<r>_trace.json.gz`：GPU 侧加 CPU 侧，看一张卡的细节用；
- **报告 TSV** `measure_trace_report*.tsv`：数出来的账，第 7 节解释各列。

## 1. 第一层：时间线结构

先认清这几样：
- **stream：** `stream 7` 是计算流，其他 stream 是 NCCL（每个进程组一条或几条）和拷贝。
- **哪一步：** 只录了第 5 步，trace 里叫 `ProfilerStep#4`。
- **PP 调度的 GPU 标注：**
  - `PP:<虚拟 stage><F|B><micro-batch>` 是一次前向或反向，例如 `PP:1B14` 是虚拟 stage 1 的第 14 个 micro-batch 的反向；
  - 另有 `SEND_F`、`RECV_B`、`WAIT_SEND_*`、`UNSHARD`、`RESHARD`、`REDUCE_GRAD`。
- **虚拟 stage 在哪：** PP rank p 持有虚拟 stage p、p+pp、p+2·pp……。pp 2、VPP 2 时，node 0 是 0 和 2，node 1 是 1 和 3。

练习：
1. 在 k3_A 的合成 trace 里搜 `PP:0F0`，再搜 `PP:1F0`。两者的时间差就是 stage 0 算完、发送、stage 1 收到并开始算的时间。
2. 从 stage 0 第一个前向开始，数到第一个反向之前有几个前向，和 Interleaved 1F1B 的预热阶段对照。
3. 在 stage 1 的 `stream 7` 上找没有 `PP:` 标注覆盖的空档，这就是气泡和等待。把它们的总长和 (pp−1)/(v·m) 的理论值比。

能回答就算过：k3_A 一步里，stage 0 和 stage 1 各在做什么、各等了多久。

## 2. 第二层：通信

点一个 NCCL kernel，读它的参数：
- **`Collective name`：** `all_to_all`、`allGatherSingleImpl`、`reduceScatterSingleImpl`、`all_reduce`、`send`、`recv`。all-to-all 和 PP 的 send / recv 都用 `ncclDevKernel_SendRecv` 这个 kernel，只能靠这个字段分开。
- **`Process Group Name`：** 到 `measure_trace_report_groups.tsv` 查是哪一维、哪些 rank。
- **字节和带宽：** 字节数 = `In msg nelems` × dtype 的字节数，带宽 = 字节数 ÷ 时长。对照 11-08 的通信基线：
  - 节点内 8 卡 alltoall 约 330 GB/s；
  - 16 卡 all_reduce 439 GB/s；
  - 跨节点 sendrecv 43 GB/s。

练习：
1. **all-to-all 的带宽和等待：** 选一个 EP 的 `all_to_all`，算 GB/s。再看同一次 collective 在 8 个 rank 上的开始时间差：先到的 rank 的 kernel 更长，多出来的就是在等。
2. **混合精度：** 分别点一个 `ReduceScatter_bf16_RING` 和一个 `ReduceScatter_f32_RING`，查它们属于哪个组。核对 bf16 的是不是 TP 的激活，f32 的是不是 FSDP 的梯度，并说明为什么梯度用 fp32 归约。
3. **NCCL 的算法：** 11-08 训练里所有 NCCL kernel 都是 RING。到 `raw/h100/comm/comm_node0.log` 里找两样：
   - NCCL 打印的各算法、各协议的延迟 / 带宽表（`AllGather |`、`ReduceScatter |` 开头那几行）；
   - `NVLS multicast support is available` 这一行：NVLS 是可用的。

   再解释训练里为什么还是选了 RING。
4. **被盖住的比例：** 一个 collective 跑的时候，`stream 7` 上有没有计算 kernel 同时在跑？对照 `measure_trace_report.tsv` 的 `overlap_pct`。

能回答就算过：k3_A 的 PP 暴露 25% 里，传输占多少、等待占多少；EP 的 all-to-all 为什么只有 48 GB/s。

## 3. 第三层：计算 kernel

`measure_trace_report_kernels.tsv` 把每个 rank 的 kernel 时间按类分好了（第 7 节）。k3_A 的 rank 12（一张正常的卡）第 5 步：

| 类 | kernel 数 | 时间 |
|---|---|---|
| NCCL（含被盖住的和等待的时间） | 3,808 | 9.11 s |
| aten 逐元素、归约、索引 | 91,677 | 5.02 s |
| GEMM | 19,168 | 1.66 s |
| KDA（`kda/` 标注里的 kernel） | 2,160 | 0.35 s |
| flex attention | 240 | 0.25 s |

计算流上 12.6 万个 kernel，49% 短于 10 微秒，中位数 11 微秒。PP 前向动作的 GPU 时间合计 6.8 秒，反向 10.2 秒。

练习：
1. **eager 的 MFU 为什么低：** 用上表解释 k3_A 的 MFU 只有 5.9%。逐元素 kernel 的时间是 GEMM 的 3 倍，compile 的融合就是针对它；暴露通信 compile 管不到。
2. **GEMM 跑到峰值的多少**（用 11-08 的 trace；之后的 run 默认不记形状，要做这题就在那一行加 `MN_PROFILE_SHAPES=1`）：在单 rank 原始 trace 里点一个 GEMM，顺着连线找到启动它的 `aten::mm` 或 `aten::addmm`，读 `Input Dims` 得到 M、N、K。用 2·M·N·K ÷ 时长算出 TFLOPS，和 989 比。
3. **降频的代价：** 在 rank 12 和 rank 13 上各找同一个 GEMM，比时长。再和 `gpu_by_cell.tsv` 里两张卡第 5 步的 SM 频率之比（1980 对约 800 MHz）对照，看两个比例是否接近；不接近的话想想为什么（访存型 kernel 受频率影响小）。
4. **前向和反向：** 用 `pp_f_ms` 和 `pp_b_ms` 算每个 micro-batch 前向、反向各多久，算出反向 / 前向的比例。它也是模拟文档要校准的量。

能回答就算过：k3_A 一步 17.5 秒里，有多少是真正有用的 FLOPs，其余花在哪。

## 4. 第四层：CPU 和 GPU 的衔接（单 rank 原始 trace）

- 选中一个 kernel，Perfetto 会画出从 CPU 侧 `cudaLaunchKernel` 到它的箭头（flow）。
- GPU 空着、CPU 在忙着发 kernel：CPU 是瓶颈。eager 下一步要发约 12.5 万个 kernel，就是这种。
- GPU 空着、CPU 也在等（`cudaStreamSynchronize`、等 NCCL、等数据）：依赖造成的等待。

练习：
1. **CPU 下发 kernel 的开销：** 数一步里 `cudaLaunchKernel` 的次数和平均耗时，估算 CPU 光下发 kernel 要多久，和步时间比。这就是 compile 和 CUDA graph 要省的那部分。
2. **没查过的空闲：** `k3_D` 的 rank 12 空闲 14.1 秒，`k3_B` 的 rank 2 空闲 22 秒，原因都还没查。打开它们的原始 trace，在计算流上找长空档，看同一时刻 CPU 线程在做什么；CPU 那边也没有事件的话，就是 trace 漏了。
   - 这两个合成 trace 有 300 多 MB，可以只合这一个 rank：`python kit/merge_traces.py <格> --ranks 12 --cpu --out <文件>`。

## 5. 第五层：显存

- 文件：`raw/h100/manual/probe/<格>/tt/profiling/memory_snapshot/step_000000000002/` 下的 `000000_step_2.pickle`（rank 0）和 `000008_step_2.pickle`（rank 8），拖进 https://pytorch.org/memory_viz。
- "Active Memory Timeline" 里分三层看：
  - 底下平的一层：参数、梯度、优化器状态；
  - 前向时一级级往上叠、反向时往下落的：在途 micro-batch 的激活；
  - 尖刺：临时张量。

练习：
1. 打开 k3_A 的 probe 快照，比 rank 0（stage 0）和 rank 8（stage 1）：数台阶，就是在途 micro-batch 的个数，再解释 stage 0 为什么高（11-08 实测 62 GiB 对 52 GiB）。
2. 点最高的那块分配，看调用栈：是哪一层、哪个算子的激活。

## 6. 第六层：kernel 内部（本地 GPU 就能练）

- **Nsight Compute：** 用 debug 模型在本地单卡上跑几步，只抓一个 kernel，例如 `ncu --set full -k regex:chunk_kda -c 1 python ...`。读 roofline、SM 吞吐、DRAM 吞吐、实际 occupancy 和理论 occupancy。
- **Nsight Systems：** `nsys profile -t cuda,nvtx,osrt` 看 CPU 线程、kernel 和 NCCL 调用在同一条时间线上。

练习：
1. 各抓一个 KDA kernel 和一个 aten 逐元素 kernel，判断谁受计算限制、谁受访存限制。
2. 把 Nsight Compute 里的寄存器和 shared memory 数，和 Kineto trace 里同一个 kernel 的参数对上。

## 7. 报告各列

- `measure_trace_report_kernels.tsv`：每格、每个 rank 一行，都是录 trace 的那一步。
  - `gemm_ms`、`attention_ms`、`kda_ms`、`triton_ms`、`aten_ms`、`other_ms`：各类计算 kernel 的时间；
  - `copy_ms`：memcpy 和 memset；
  - `nccl_ms`：NCCL kernel 的时间，含被盖住的和等待的；
  - `stream_kernels`、`small_pct`、`median_us`：计算流上的 kernel 数、短于 10 微秒的比例、时长中位数；
  - `pp_f_ms`、`pp_b_ms`、`pp_actions`：PP 前向、反向动作在 GPU 上的总时长和动作个数。动作里面的通信（EP、TP）也算在内，动作之间的等待不算。
- `measure_trace_report_topk.tsv`：每格耗时最多的 20 个 kernel，`per_rank` 是每个 rank 每步调用几次，`ms_per_rank` 是每个 rank 每步花多少毫秒。
- 其余几个表（计算、暴露、空闲、按进程组、按 rank、进程组哈希）见 `../results/20261108_h100_together/README.md` 第 6 节。

重新生成：`python kit/trace_report.py <stage 目录> --jobs 3`；一格 16 个 rank 在本机约 10 分钟。

## 8. 写成案例

每个案例一页，按"现象 → 假设 → 测量 → 改动 → 验证 → 结论"写。手头已有三个：

| 案例 | 现象 | 测量 | 改动和验证 | 对应的岗位要求 |
|---|---|---|---|---|
| 慢卡 | rank 13 的计算时间比其他 rank 多 2 到 8 秒 | trace 的 `compute_ms`、DCGM 的温度和降频原因、第 5 步的 SM 频率 | kit 开跑前先查降频卡；下一轮换节点后复测 | 独立排查模糊的性能问题；GPU 架构；大规模实验的监控 |
| FSDP 跨节点 | k3_B 最慢，FSDP 加专家 FSDP 的暴露占 34% | 每步 1024 次参数 all-gather，等于 16 个 micro-batch × 64 次 | `k3_B_keep` 行（`MN_FSDP_KEEP=1`）对照 | 分布式并行方式；可扩展性优化 |
| eager 的开销 | MFU 5.9% | 逐元素 kernel 5.0 秒，GEMM 1.7 秒；12.5 万个 kernel，一半短于 10 微秒 | 挑一两行开 compile 对照，量编译时间和编译后的 MFU | 计算效率优化；PyTorch 和 kernel 层面的理解 |
