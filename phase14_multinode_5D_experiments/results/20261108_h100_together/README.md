# 2026-11-08 Together 2 × 8 H100：K3 缩小版四种布局的 measure

**结论：**
- 四种布局在 16K、eager、VPP 2 下都跑通。
  - 最快的是 `k3_A`（PP 跨节点，TP 4），每步 17.0 秒，MFU 5.9%；
  - 最慢的是 `k3_B`（没有 PP，FSDP 跨节点），每步 30.9 秒，MFU 3.3%。
- 四格里大多数 rank 每步的计算时间都是 7 到 9 秒；步时间的差别几乎全在没被盖住的通信和空闲上。
- node 1 的 GPU 5（rank 13）在四格里都过热降频。录 trace 的那一步它降频得最厉害，所以 trace 的拆分里有它的影响（第 3.5 节）。

## 1. 环境

| 项 | 值 |
|---|---|
| 集群 | Together 抢占式，2 个节点 × 8 张 H100 80GB HBM3，Kubernetes，每个节点一个 pod |
| 网络 | 每个节点 8 个 400G IB 口（`mlx5_0`-`3`、`mlx5_6`-`9`）。NCCL 走 IB，有 GPUDirect RDMA |
| 网络实测 | 每个口 RDMA 原始带宽 364 到 368 Gb/s（`comm/rdma_node1.txt`）；1 GiB 消息的 busbw：16 卡 all_reduce 439 GB/s，16 卡 alltoall 91 GB/s（节点内 8 卡 330 GB/s），跨节点 sendrecv 43 GB/s（`comm/verdict_node0.txt`） |
| 软件 | torch `2.16.0.dev20261024+cu130`，NCCL 2.32.3，titan 分支 `phase14_multinode_5d` 的 `ef6f2acb1` |
| 模型 | `k3_16x80g`：K3 缩小版，29 层，D 2304，128 个专家，总参数 14.47B，激活 3.68B；随机初始化，AdamW，fp32 主参数加 bf16 计算 |
| 数据 | c4，tokenizer 用 Qwen3-8B 的（K3 的转换失败时的退路，`MN_SPECIAL=qwen`） |
| 每步 | 524,288 token，序列 16K；`k3_A`、`k3_B`、`k3_C` 每个 dp rank 16 个 micro-batch，`k3_D`（dp 1）32 个 |
| 设置 | eager（`MN_COMPILE=0`），PP 的格子用 `Interleaved1F1B`，每个 rank 2 个虚拟 stage，每个 8 层；5 步，第 5 步录 trace |

这四格是用 `k8s.sh cell measure <行>` 手动逐格跑的，设置和现在清单里的快速路径相同（`../../kit/PLAYBOOK.md` 第 0 节）。

## 2. 每格的数

步时间和吞吐取第 2 到 4 步的中位数，这三步不录 trace。第 5 步在录 trace，单独列出。tps 是每张卡每秒的 token 数。

| 行 | 度数 pp·dp·cp·tp·ep | 跨节点的维度 | 步时间 | 第 5 步 | 每卡 tps | MFU | 峰值显存（rank） |
|---|---|---|---|---|---|---|---|
| `k3_A` | 2·2·1·4·8 | PP | 17.0 s | 18.1 s | 1929 | 5.9% | 61.9 GiB（0） |
| `k3_B` | 1·2·4·2·8 | FSDP（8 路），专家 FSDP（2 路） | 30.9 s | 36.5 s | 1060 | 3.3% | 31.0 GiB（3） |
| `k3_C` | 2·2·4·1·8 | PP | 19.0 s | 19.9 s | 1777 | 5.5% | 63.4 GiB（3） |
| `k3_D` | 2·1·4·2·8 | PP | 24.7 s | 32.2 s | 1328 | 4.1% | 43.2 GiB（7） |

第 1 步含启动和 flex attention 的编译：`k3_A` 49 秒，`k3_B` 88 秒，`k3_C` 239 秒，`k3_D` 73 秒。

trace 里第 5 步 GPU 时间的拆分（16 个 rank 的平均，`measure_trace_report.tsv`）：

| 行 | 计算 | 暴露通信 | 空闲 | 通信被计算盖住的比例 | 暴露最多的三个组（占步时间） |
|---|---|---|---|---|---|
| `k3_A` | 44.3% | 47.7% | 7.5% | 22.0% | PP 25.3%，EP 13.3%，TP 8.1% |
| `k3_B` | 24.6% | 59.0% | 16.1% | 19.4% | FSDP/LOSS 20.4%，EFSDP 13.7%，CPxTP/EP 13.3% |
| `k3_C` | 41.1% | 51.3% | 7.1% | 19.4% | PP 25.7%，EP/FSDP/LOSS 20.5%，CP 5.0% |
| `k3_D` | 26.7% | 51.7% | 21.2% | 14.4% | PP 23.9%，CPxTP/EP 11.9%，CP/FSDP/LOSS 8.1% |

## 3. 每格在说什么

### 3.1 `k3_A`：PP 跨节点，TP 4

- **PP：** 25.3%。
  - 主要是 `recv`：每个 rank 每步 96 次，暴露 3.3 秒；
  - 它的 `eff_busbw` 只有 0.38 GB/s，说明这段时间几乎都是在等对面的 stage 把激活发过来（含流水线气泡），不是带宽不够。
- **EP：** 13.3%。
  - 每步 1120 次 all-to-all，组在节点内（NVLink）；
  - `eff_busbw` 48 GB/s，远低于节点内 alltoall 实测的 330 GB/s，说明 kernel 时间里大部分是等最慢的那个 rank，或者路由不均。
- **TP：** 8.1%，是序列并行的 all-gather 和 reduce-scatter，节点内 `eff_busbw` 70 到 82 GB/s。

### 3.2 `k3_B`：没有 PP，FSDP 跨节点

- **FSDP：** 20.4%。
  - dp 2 在最外层，所以 dp × cp = 8 个 rank 的 FSDP 组跨两个节点；
  - 每步 1024 次参数 all-gather，等于 16 个 micro-batch 各 64 次：默认每层 forward 后 reshard，前向、反向都要重新 gather。
- **EFSDP：** 13.7%。专家参数的 FSDP 是 2 路、跨节点，reduce-scatter 的 `eff_busbw` 只有 6 GB/s。
- **EP：** 13.3%，组在节点内，但 `eff_busbw` 只有 21 GB/s。
- 这是四格里最慢的。矩阵里 `k3_B_keep`（`MN_FSDP_KEEP=1`，整步不 reshard）就是用来验证"每个 micro-batch 重新跨节点 gather"是不是主因，还没跑。

### 3.3 `k3_C`：PP 跨节点，不用 TP，CP 4

- **PP：** 25.7%，和 `k3_A` 一样主要是 `recv` 在等。
- **EP/FSDP/LOSS：** 20.5%。
  - 这个格子里 dp × cp = 8 正好就是 EP 的 8 个 rank，FSDP、EP、loss 归约用同一组卡，所以合成一个标签；
  - 其中 18.0% 是 all-to-all（EP），1.8% 是 reduce-scatter，0.7% 是 all-gather（FSDP）。
- **CP：** 5.0%，K/V 的 all-gather 和梯度的 reduce-scatter。

### 3.4 `k3_D`：PP 跨节点，没有 FSDP，每个 stage 8 卡

- dp 1，每步 32 个 micro-batch，PP `recv` 每步 192 次，PP 占 23.9%。
- 空闲 21.2%，四格里最高。
  - rank 12 空闲 14.1 秒，其他 rank 4 到 8 秒；`k3_B` 的 rank 2 也有类似的情况（空闲 22 秒）；
  - 原因还没查。GPU 上什么都没跑，可能是 CPU 那边没及时下发 kernel，也可能是 trace 本身漏了事件，要按第 5.4 节看 CPU 侧才能分辨。
- **CP/FSDP/LOSS：** 8.1%。
  - dp 1 时 FSDP 组就是 CP 组；
  - 这里的 all-gather 既有 CP 的 K/V，也有 FSDP 的参数，从 trace 里分不开。

### 3.5 慢卡：node 1 的 GPU 5（rank 13）

| 行 | rank 13 的计算时间 | 其他 rank 的中位数 | rank 13 第 5 步的平均 SM 频率 | 旁边 GPU 4 |
|---|---|---|---|---|
| `k3_A` | 13.1 s | 7.4 s | 802 MHz | 1980 MHz |
| `k3_B` | 12.3 s | 8.5 s | 1216 MHz | 1980 MHz |
| `k3_C` | 15.4 s | 7.5 s | 917 MHz | 1980 MHz |
| `k3_D` | 10.1 s | 7.8 s | 1529 MHz | 1980 MHz |

- 这张卡最高 88 到 90°C，旁边的 GPU 4 不超过 47°C。第 5 步里它 83% 到 100% 的采样处于软件热降频（原因码 `0x20`），每格的数在 `gpu_by_cell.tsv`。
- 频率高的两格（`k3_B`、`k3_D`）它多算 2 到 4 秒，频率低的两格（`k3_A`、`k3_C`）多算 6 到 8 秒。
- 每个 collective 都要等最慢的 rank，所以：
  - rank 13 自己的暴露通信很少（`k3_A` 2.6 秒，其他 rank 7 到 10 秒）；
  - 其他 rank 的暴露通信里有一部分是在等它。
- 这四格的绝对值都因为它偏慢；四格都受它影响，但影响的大小不同（上表）。格子之间差得多的地方（`k3_B` 比 `k3_A` 慢 80%）结论可靠，差得少的地方（`k3_A` 和 `k3_C`）要在没有慢卡的节点上重测。

## 4. 文件

提交进 git 的（小文件）：

| 文件 | 内容 |
|---|---|
| `measure_summary.tsv` | `summarize.py`：每格一行，退出码、步数、loss、峰值显存、tps / MFU / 步时间（第 3 步以后的中位数，含第 5 步）、第 1 步时间 |
| `measure_trace_report.tsv` | `trace_report.py`：每格一行，第 2 节第二张表的来源 |
| `measure_trace_report_pg.tsv` | 每格、每个进程组、每种 collective 一行：次数、kernel 时间、暴露时间、字节数、`eff_busbw` |
| `measure_trace_report_rank.tsv` | 每格、每个 rank 一行：计算、暴露通信、拷贝、空闲、通信总时间 |
| `measure_trace_report_groups.tsv` | 每格每个进程组的哈希名、标签和成员 rank：Perfetto 里点 NCCL kernel 看到的 `Process Group Name` 在这里查 |
| `gpu_by_cell.tsv` | 每格、每张卡：平均利用率、功耗、最高温度、SM 频率、降频采样占比（从 `monitor.py` 的 2 秒采样按每格的训练时间段切出来） |
| `comm/` | 通信检查的结论和每个 IB 口的 RDMA 带宽 |
| `cells/<格>/` | 每格两个节点的 `env_node*.txt`（全部 `MN_*`、`NCCL_*` 变量）、`tree.txt`（titan 提交号）、`rc_node*` |

只在本机的（`../.gitignore` 排除，不进 git）：

| 文件 | 内容 | 用什么打开 |
|---|---|---|
| `k3_<行>_<时间>_measure_trace.json.gz` | 一格 16 个 rank 合成一份的 GPU trace：A 142 MB，B 333 MB，C 143 MB，D 301 MB | https://ui.perfetto.dev |
| `raw/h100/manual/measure/<格>/tt/profiling/traces/iteration_5/rank<r>_trace.json.gz` | 每个 rank 自己的完整 trace，含 CPU 侧（40 到 110 MB） | Perfetto |
| `raw/h100/manual/measure/<格>/node0.log`、`node1.log` | 两个节点 16 个 rank 的完整训练日志 | 文本编辑器 |
| `raw/h100/manual/measure/<格>/tt/tb/` | TensorBoard（最后一个 PP stage 的 loss 等） | `tensorboard --logdir` |
| `raw/h100/manual/probe/<格>/tt/profiling/memory_snapshot/step_000000000002/*.pickle` | probe 第 2 步 rank 0 和 rank 8 的显存快照（160 到 185 MB） | https://pytorch.org/memory_viz |
| `raw/h100/monitor/gpu_node*.csv` | 两个节点每 2 秒一行的 GPU 采样（利用率、显存、功耗、SM 频率、温度、降频原因） | 表格软件 |
| `raw/h100/comm/` | 通信检查的完整日志（NCCL INFO、`ib_write_bw` 输出、`comm_bench.json`） | 文本编辑器 |
| `raw/top/` | setup、流水线、采样器的日志，`scope.env` | 文本编辑器 |

`raw/` 一共 7.4 GB，509 个文件，是集群删掉前从共享卷整份拉回来的。另外几次失败或中断的运行也在 `raw/h100/manual/` 里：
- `k3_5d` 在 32K OOM；
- `k3_5d` 在 4096 跑了 25 步，但写 trace 时被杀，16 个 trace 全部截断，读不了；
- 几次 smoke、probe、warm。

## 5. 怎么看 trace

### 5.1 打开

1. 浏览器打开 https://ui.perfetto.dev，左边点 "Open trace file"，选 `k3_A_1108_170956_measure_trace.json.gz`。不用解压，文件在浏览器本地解析，不会上传。一两分钟才能加载完。
2. 四格的合成文件和进程组哈希表都已生成。要从 `raw/` 重新生成，在 Git Bash 里跑下面的命令，合成每格 6 到 14 分钟，哈希表约 10 分钟：

```sh
cd phase14_multinode_5D_experiments
export PYTHONUTF8=1
R=results/20261108_h100_together; M=$R/raw/h100/manual/measure
for c in k3_B_1108_171837 k3_C_1108_172528 k3_D_1108_173242; do
  python kit/merge_traces.py $M/$c --out $R/${c}_measure_trace.json.gz
done
python kit/trace_report.py $M --jobs 3 && cp $M/trace_report_groups.tsv $R/measure_trace_report_groups.tsv
```

3. 标签页内存不够、加载失败时，只合几个 rank，文件会小很多：

```sh
python kit/merge_traces.py $M/k3_A_1108_170956 --ranks 5,12,13 --out $R/k3_A_r5_12_13.json.gz
```

### 5.2 界面

- 左边每个进程是一个 rank，名字是 `rank 13, node 1`；展开后每一行是这张卡上的一个 CUDA stream：
  - `stream 7` 是计算流，几乎所有计算 kernel 都在这条上；
  - 其他 stream 是 NCCL 的通信流（每个进程组一条或几条）和拷贝。
- 时间轴：`W` / `S` 放大缩小，`A` / `D` 左右移，或者按住 Ctrl 滚轮。
- 只录了第 5 步。trace 里这一步叫 `ProfilerStep#4`（profiler 从 0 开始数），顶上搜索框输入 `ProfilerStep` 可以跳过去。
- 两个节点的时钟差：合成时打印每个节点第一个 `ProfilerStep` 的时间差，`k3_A` 只差 3.8 ms，所以按绝对时间合成就能对齐。差得多时加 `--align-step`。

### 5.3 点一个 kernel 看什么

下方 "Current Selection" 面板显示这个 kernel 的参数：
- NCCL kernel（名字以 `ncclDevKernel_` 开头）：
  - `Collective name`：`all_to_all`、`allGatherSingleImpl`、`reduceScatterSingleImpl`、`all_reduce`、`send`、`recv`；
  - `Group size`、`In msg nelems`（元素个数）、`dtype`；
  - `Process Group Name`：一个哈希，到 `measure_trace_report_groups.tsv` 里查它是 TP、EP、PP 还是哪一组，以及有哪些 rank。
- all-to-all 和 PP 的 send / recv 都用 `ncclDevKernel_SendRecv` 这个 kernel，要看 `Collective name` 才能分开。
- 计算流上还有 GPU 标注：`kda/...` 是 KDA 的各段 kernel（前向 `chunk_kda_fwd_intra`、反向 `backward_*` 等）。通信流上的标注是 collective 的名字。
- 在时间轴上拖出一段，下方会按名字汇总这段里每种 slice 的次数和总时长，可以估算一段里计算和通信各占多少。

### 5.4 该看的几处

- **PP 的等待**（`k3_A`、`k3_C`、`k3_D`）：
  - node 0 的 rank 0 到 7 是第一个 stage，node 1 的 rank 8 到 15 是第二个 stage；
  - 一个 stage 的计算流出现空档时，看它的 PP 流上是不是有一段很长的 `recv`：那就是在等另一个 stage；
  - VPP 2 下每个 rank 有两个虚拟 stage，前向和反向交替出现。
- **EP 的 all-to-all：** MoE 层前后各一次。16 个 rank 同时出现时，长短不齐，说明有的 rank 先到、在等后到的。
- **慢卡：**
  - 把 rank 12 和 rank 13 并排看（同一个节点、同一个 stage），rank 13 计算流上同一个 kernel 明显更长；
  - 再看其他 rank 的通信流：同一个 collective 在它们那里开始得早、结束得和 rank 13 一样晚，多出来的就是在等它。
- **FSDP**（`k3_B`）：每个 micro-batch 前有一串 `allGatherSingleImpl`，`Group size` 8 的组跨节点，比节点内的组慢。
- **空闲**（`k3_D` 的 rank 12，`k3_B` 的 rank 2）：
  - 先在合成 trace 里找到它们计算流上的长空档；
  - 再用 `--ranks 12 --cpu` 只合这一个 rank、带上 CPU 侧，看空档期间 CPU 线程在做什么（Python 调用、`cudaLaunchKernel`、同步等待）；
  - 如果 CPU 那边也没有事件，就是 trace 漏了。

### 5.5 显存快照和 TensorBoard

- 显存快照：打开 https://pytorch.org/memory_viz，把 `.pickle` 拖进去。"Active Memory Timeline" 显示第 2 步里显存随时间的变化，点一块能看到分配它的调用栈。
- TensorBoard：`pip install tensorboard`，然后 `tensorboard --logdir raw/h100/manual/measure`，浏览器开 http://localhost:6006。

## 6. 怎么读 TSV

- `measure_trace_report.tsv`：GPU 时间在每个时刻只算进四块中的一块：
  - 计算：有计算 kernel 在跑，同时在跑的通信算被盖住；
  - 暴露通信：只有 NCCL kernel 在跑；
  - 拷贝：只有拷贝在跑；
  - 空闲：什么都没跑。

  `overlap_pct` 是 NCCL kernel 时间里和计算重叠的比例；`exposed_max_pct`、`idle_max_pct` 是最差那个 rank 的值。各列的定义在 `../../kit/K8S_RUN.md` 第 4 节。
- `measure_trace_report_pg.tsv`：
  - `kernels`、`kernel_ms`、`exposed_ms`、`mbytes` 都是每个 rank 每步的平均；
  - `kernel_ms` 含被盖住的部分，`exposed_ms` 只算没被盖住的；
  - `eff_busbw` 是字节数除以 kernel 时间，按 nccl-tests 的系数折成 busbw。kernel 时间里含等对方的时间，所以它是下限。PP `recv` 的 0.38 GB/s 是在等，不是链路只有这么快。
- **合成的标签**（如 `EP/FSDP/LOSS`）：几个维度用的是同一组卡。按 collective 分开：
  - all-to-all 是 EP；
  - 参数的 all-gather 和梯度的 reduce-scatter 是 FSDP；
  - loss 的 all-reduce 是 LOSS。
- `measure_trace_report_rank.tsv`：找慢 rank 先看 `compute_ms` 特别大的。被等的 rank 自己的 `exposed_ms` 反而小，等它的 rank 的 `exposed_ms` 大。
- `ib_tx_GBps`、`ib_rx_GBps` 是空的：Together 的 pod 读不到 IB 口的字节计数器。

## 7. 局限

- 每格只有 5 步、一次运行、一步 trace，没有同一格重复两次的噪声底线，所以差几个百分点的格子不能排先后。
- eager：kernel 没调优，MFU 比编译后低，四格同口径。
- 录 trace 的第 5 步比前几步慢 5% 到 30%（profiler 的开销，加上 rank 13 降频更厉害），trace 拆分是这一步的。
- 一张降频的卡影响了全部四格（第 3.5 节）。
- MFU 是 titan 的口径：(6 × 激活参数 + 注意力算子 FLOPs) × tps ÷ 989 TFLOPS（H100 bf16 稠密峰值）。

## 8. 下一轮

- 按 `../../kit/PLAYBOOK.md` 第 0 节的快速路径，一次跑完矩阵的 9 行。
- 跑之前先看 dashboard 的 DCGM 和 `gpu_node*.csv`，有过热降频的卡就换节点。
- `k3_B` 和 `k3_B_keep` 对照，看 FSDP 跨节点时每个 micro-batch 重新 gather 的代价。
- `k3_D` rank 12 和 `k3_B` rank 2 的空闲，按第 5.4 节看 CPU 侧。
