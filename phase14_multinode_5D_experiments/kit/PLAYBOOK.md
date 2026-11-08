# Phase 14 完整流程：Together 上 2 × 8 卡的 5D 实验（2026-11-08）

从开集群到出结果的全部步骤、每一种并行配置怎么跑、对应的 profiler，以及 NCCL / RDMA 的验证和调优，都在这一份里。**这套流程只在本机模拟过（两节点协同、合成 trace、假 kubectl），还没在真集群上跑过。**

其他文件只放细节：
- `together/README.md`：建集群、kubeconfig、IB 排查；
- `K8S_RUN.md`：运维细节、状态文件、report 各列的定义；
- `RUNBOOK.md`：不用 k8s、在裸机或 VM 上手动跑的版本；
- `README.md`：titan 分支、两个缩小版模型、矩阵的来由；
- `../multinode.md`：纸面规划和每格的预期。

下文的命令都在本机 logbook 根目录下执行，先设：

```sh
K8S=phase14_multinode_5D_experiments/kit/k8s.sh
PK=/mnt/shared/p14/logbook/phase14_multinode_5D_experiments/kit    # pod 里 kit 的位置
```

## 0. 总览

| 步骤 | 在哪 | 做什么 | 纸面时间 |
|---|---|---|---|
| 1 | 本机 | 充值、装工具、改清单 | 10 分钟 |
| 2 | Together 控制台 | 建集群（2 个节点、Kubernetes、共享卷） | 等资源 |
| 3 | 本机 | 拿 kubeconfig、填卷名、`kubectl apply` | 5 分钟 |
| 4 | pod 里自动 | setup → 通信检查（RDMA + NCCL）→ tokenizer → 每行 smoke / probe / warm / measure → NCCL 调优扫描 → 报告 | 第一轮约 2 到 3 小时 |
| 5 | 本机 | 看进度、看 report / trace / 通信结论 | 随时 |
| 6 | 本机 + 控制台 | collect、删集群、处理共享卷 | 10 分钟 |

费用：16 张抢占式按 $1.995 算，每小时 $31.92，从节点上线算到删集群，setup 和空转也算。第一轮只跑 `k3_5d` 一行加调优扫描，约 $65 到 $100；两个模型全矩阵约 8 到 10 小时，$260 到 $320。都是纸面估计。

## 1. 开始前（本机）

1. **余额：** 创建时余额要够"抢占式目标的 4 小时 + 按需部分的最短时长 + 同期存储"，16 卡约 $128 起。第一轮建议充到 $200 左右，跑全量准备 $400 左右。余额用完，集群立刻暂停，宽限期后删除，而且扣完之前没有提醒。
2. **工具：** `kubectl`，以及 Together CLI：`uv tool install "together[cli]" --upgrade`。
3. **改清单** `phase14_multinode_5D_experiments/kit/together/p14_pods.yaml`：

```yaml
  volumeName: <共享卷对应的 PV 名，第 3 步查>
...
        - name: HW
          value: h200        # 控制台标价 $1.995 和文档里 H100 抢占式的价一致；是 H100 就写 h100
        - name: MN_MODELS
          value: "k3"        # 第一轮只跑 K3
        - name: MN_ROWS
          value: "k3_5d"     # 第一轮只跑 5D 全开这一行；留空就是整个矩阵
        - name: MN_COMM_SWEEP
          value: "1"         # 矩阵跑完后做 NCCL 调优扫描，约 20 分钟；"0" 不做
```

kit 只支持 2 个节点，不要建 32 卡。

## 2. 建集群（Together 控制台）

- 抢占式 16 卡，也就是 2 个节点；编排选 Kubernetes。驱动用默认的。
- 如果控制台要求至少一个按需（standard）节点，就建 1 个按需节点加 1 个抢占式节点，两个 pod 一边一个，清单里的 `nodeSelector` 保持注释。
- 共享卷 500 GiB 左右，跟集群一起建，记下卷名。卷能扩不能缩。

细节和依据见 `together/README.md` 第 0、1 节。

## 3. 拉起两个 pod

```sh
tg clusters list
tg clusters get-credentials <CLUSTER_ID> --set-default-context --no-json
kubectl get nodes -L together.ai/compute-class
kubectl get pv                                          # 共享卷同名的那个 PV，填进清单的 volumeName
kubectl create secret generic hf-token --from-literal=token=<hf_token>   # 可选
kubectl apply -f phase14_multinode_5D_experiments/kit/together/p14_pods.yaml
bash $K8S status
```

## 4. pod 起来以后自动做的事

不用手动操作，按顺序：

1. **容器包**（每次启动，1 到 2 分钟）：git、编译器、rdma-core、libibverbs、perftest。
2. **共享 setup**（第一次，10 到 15 分钟；一个 pod 装，另一个等）：venv、torch nightly、titan 分支 `phase14_multinode_5d` 固定在 `env.sh` 的 `BRANCH_SHA`。
3. **采样器** `monitor.py`：每个节点每 2 秒记 GPU 和每个 IB 口的速率，之后一直开着。
4. **通信检查 `comm`**（约 5 分钟），见第 7.2 节：
   1. 每个 IB 口的 RDMA 原始带宽；
   2. NCCL 走没走 IB、有没有 GPUDirect RDMA；
   3. 256 B 到 1 GiB 的 busbw 和 α–β 拟合。
5. **tokenizer**（约 2 分钟）。K3 的转换失败时，自动退到 Qwen3 的 tokenizer。
6. **矩阵的每一行**依次跑四个阶段：

| 阶段 | 模型 | 数据 | 步数 | 顺带记录 |
|---|---|---|---|---|
| smoke | debug 模型，K3 4096 / M3 2048 token | c4_test | 5 | 每种布局能起来、能走步 |
| probe | 缩小版，矩阵里的长度 | c4_test | 3 | 每个 rank 的峰值显存；OOM 时长度减半重试，最低 4096；第 2 步存显存快照 |
| warm | 缩小版，probe 定下的长度 | c4 | 10 | 填满这个模型所有行共用的编译 cache |
| measure | 同上 | c4 | 30 | 第 25 步所有 rank 的 profiler trace，TensorBoard |

   每步约 512K token，随机初始化，AdamW，fp32 主参数加 bf16 计算。这是吞吐和通信的测量，不是真训练。
7. **NCCL 调优扫描**（`MN_COMM_SWEEP=1`，约 20 分钟），见第 7.3 节。
8. **收尾：** 生成每个阶段的 `summary.tsv`、`monitor_summary.tsv`、`comm_compare.tsv`、measure 的 `trace_report*.tsv`、`REPORT.txt` 和 `ALL_DONE`。之后 pod 空转，但照样计费。

被抢占时，换上来的 pod 会从第一个没完成的任务接着跑，被打断的那次不算失败（`K8S_RUN.md` 第 5 节）。

## 5. 每一种并行配置

### 5.1 矩阵

度数按 `pp · dp · cp · tp · ep` 写，`dp` 是 FSDP 的分片度，M3 另有 `dpr`（HSDP 的副本数）。titan 的 mesh 从外到内是 `pp, dpr, dp, cp, tp`，每个节点 8 卡，所以最外层大于 1 的轴落在节点之间。长度一列依次是 H100 / H200（B200 同 H200）/ A100。

K3（`matrix_k3.tsv`）：

| 行 | 度数 | 长度 | 跨节点的是 | 用来回答 | report 里看 |
|---|---|---|---|---|---|
| `k3_5d` | 2·2·2·2·8 | 32K / 16K / 24K | PP；FSDP、CP、TP、EP 都在节点内 | 5D 全开的基线 | `mfu`、`idle_pct`（气泡）、`top_exposed` |
| `k3_A` | 2·2·1·4·8 | 32K / 16K / 24K | PP | 不用 CP，TP 加到 4 | TP 的暴露 |
| `k3_B` | 1·2·4·2·8 | 64K / 32K / 64K | FSDP（专家 FSDP 2 路跨节点）；H200 上整步不 reshard | 没有 PP，FSDP 跨节点 | FSDP、EFSDP 的暴露，`ib_tx_GBps` |
| `k3_C` | 2·2·4·1·8 | 32K / 16K / 24K | PP | 不用 TP，CP 4 | CP 的暴露 |
| `k3_D` | 2·1·4·2·8 | 64K / 64K / 48K | PP | 没有 FSDP，每个 stage 8 卡，最长序列 | CP 的暴露，`peak_mem_gib` |
| `k3_D_ulysses` | 同 `k3_D`，MLA 用 Ulysses | 同上 | PP | 和 `k3_D` 比 CP 的两种实现 | CP 的暴露 |
| `k3_B_ep16` | 1·2·4·2·16 | 64K / 32K / 64K | EP（坏放置） | 和 `k3_B` 比 EP 跨节点的代价 | EP 的暴露、`eff_busbw`，`ib_tx_GBps` |
| `k3_B_keep` | 1·2·4·2·8，整步不 reshard | 32K 全部 | FSDP | 和下一行一起，在同一长度、同一 FSDP 模式下比 EP 放置 | 同上 |
| `k3_B_ep16_keep` | 1·2·4·2·16，整步不 reshard | 32K 全部 | EP | 同上 | 同上 |

M3（`matrix_m3.tsv`，没有 CP，TP 最多 4）：

| 行 | 度数（dpr·pp·dp·tp·ep） | 长度 | 跨节点的是 | 用来回答 |
|---|---|---|---|---|
| `m3_A` | 1·2·2·4·8 | 16K | PP | 和 K3 的 (A) 同度数 |
| `m3_B` | 1·1·4·4·8 | 16K | FSDP（4 路里 2 路跨节点，专家 FSDP 2 路跨节点） | 没有 PP |
| `m3_C` | 1·2·8·1·8 | 8K | PP | 不用 TP，每卡整条序列 |
| `m3_D` | 1·4·1·4·4 | 16K | PP 的三跳里一跳 | 没有 FSDP，pp 4 |
| `m3_E` | 2·1·8·1·8 | 8K | 只有 HSDP 副本之间的梯度 all-reduce | HSDP 跨节点 |
| `m3_B_ep16` | 1·1·4·4·16 | 16K | EP（坏放置） | 和 `m3_B` 比 |

可以直接比的对照组：`k3_D` 和 `k3_D_ulysses`；`k3_B` 和 `k3_B_ep16`；`k3_B_keep` 和 `k3_B_ep16_keep`；`m3_B` 和 `m3_B_ep16`。它们的长度相同，FSDP 模式也相同。H100 上的 `k3_B`（64K，每层 reshard）和 `k3_B_keep`（32K，整步不 reshard）长度不同，不能直接比。

长度都是纸面估计，probe 会自动缩短放不下的行。

### 5.2 命令

**只跑某几行或某几个阶段**（不用重建 pod）：

```sh
bash $K8S scope MN_MODELS=k3 MN_ROWS='k3_B k3_B_ep16'    # 也可以加 MN_STAGES='probe warm measure'、MN_ORDER=stage、MN_COMM_SWEEP=0
bash $K8S retask                                          # 已完成的任务不重跑
bash $K8S scope                                           # 不带参数：撤销，回到清单里的值
```

**改长度、加一行、改一行的设置：** 在本机改 `kit/matrix_k3.tsv` 或 `matrix_m3.tsv`，然后 `bash $K8S sync`，再 `bash $K8S retask`。每行 6 列，用 tab 分隔：行名、度数、H100 / H200 / A100 的长度、额外变量。额外变量可以是：
- `MN_SCHEDULE=Interleaved1F1B`：PP 用 VPP，每个 rank 2 个虚拟 stage；
- `MN_CP=ulysses`：MLA 的 CP 用 Ulysses；
- `MN_FSDP_KEEP=1`：FSDP 整步不 reshard；
- `MN_MICROBATCHES=<n>`：每个 dp rank 的 micro-batch 数，默认取最接近每步 512K token 的值；
- `MN_COMPILE=0`：K3 的局部编译区改成 eager，flex attention 不做 autotune（flex 不能完全不编译，长序列 eager 会 OOM）；省编译时间，但 kernel 没调优，MFU 偏低；
- 任何 `NCCL_*` 变量（第 7.4 节）；
- `h200:VAR=值`：只在这种卡上生效；
- `nosmoke`：跳过 smoke。

例：

```
k3_5d_vpp	pp=2,dp=2,cp=2,tp=2,ep=8	32768	16384	24576	MN_SCHEDULE=Interleaved1F1B
```

要重跑一个已完成的任务，先删掉共享卷上 `runs/<hw>/pipeline/done/<任务名>`，再 `retask`。

**按行手动逐格跑**（流水线先 `stop` 或 `kill`）：`bash $K8S cell <阶段> <行名> [VAR=值 ...]` 用流水线对这一行这一阶段的同一套设置（tokenizer、长度、NCCL 网卡、显存快照、trace），在两个 pod 上同时跑一格，跑完打印退出码和这一格的 `summary`，期间每 20 秒打印最新一步。结果写在 `runs/<hw>/manual/<阶段>/<行名>_<时间>/`，measure 的拆分用 `bash $K8S report manual/measure`。手动跑时 probe 不会自动减半：OOM 了就加 `MN_SEQ=<一半>` 重跑，warm 和 measure 也带上同样的 `MN_SEQ`。

```sh
bash $K8S cell smoke k3_5d
bash $K8S cell probe k3_5d
bash $K8S cell warm k3_5d
bash $K8S cell measure k3_5d
bash $K8S cell probe k3_B MN_SEQ=32768     # 例：长度减半重跑
```

**任意度数手动跑一格**（不进矩阵）：先 `bash $K8S stop`，等当前这次尝试结束，再让两个 pod 同时执行：

```sh
bash $K8S both "MN_MODEL=k3 MN_FLAVOR=k3_16x141g MN_ASSETS=/mnt/shared/p14/assets/kimi_k3_tokenizer \
  MN_CELL=pp=2,dp=2,cp=2,tp=2,ep=8 MN_SEQ=16384 MN_STEPS=30 MN_PROFILE=25 MN_PORT=29700 \
  bash $PK/launch.sh /mnt/shared/p14/runs/manual/try1"
bash $K8S start        # 手动跑完以后，让流水线接着跑
```

tokenizer 的实际路径和 special token 的设置记在 `runs/<hw>/pipeline/assets/k3.json`（K3 退到 Qwen3 时要加 `MN_SPECIAL=qwen`）。`MN_*` 变量的完整列表在 titan 分支 `torchtitan_recipes/multinode_5d.py` 的模块 docstring 里。度数的乘积必须等于 16；EP 不单独占卡，它要整除 `dp·cp·tp`，不小于 `tp`，还要整除专家数 128。

## 6. Profiler

### 6.1 自动记录的

| 什么 | 什么时候 | 在哪（共享卷 `/mnt/shared/p14/runs/<hw>/`） |
|---|---|---|
| 每个 rank 每步的 loss、tps、TFLOPS、MFU、峰值显存 | 所有阶段 | `<stage>/<行>/node*.log`，汇总在 `<stage>/summary.tsv` |
| 显存快照（每个节点的 local rank 0） | probe 第 2 步，以及 OOM 时 | `probe/<行>/tt/profiling/memory_snapshot/` |
| 所有 rank 的 CPU 和 CUDA trace | measure 第 25 步 | `measure/<行>/tt/profiling/traces/iteration_25/` |
| TensorBoard（最后一个 PP stage 的 rank） | measure | `measure/<行>/tt/tb/` |
| 每个节点的 GPU 和 IB / RoCE 速率，标着任务 | 一直 | `monitor/gpu_node*.csv`、`ib_node*.csv` |
| NCCL flight recorder（卡死时 dump） | 所有运行 | `<stage>/<行>/tt/comm_traces/` |

实时的卡状态来自 Together 在每个节点上装的 DCGM exporter（`gpu-operator` 命名空间）。Together 的 pod 里读不到 IB 口的字节计数器（sysfs 没有 `counters`，`perfquery` 也超时），所以跨节点流量看 PCIe 速率：开了 GPUDirect RDMA，跨节点的数据从 GPU 经 PCIe 直接到网卡。

MFU 是 titan 的口径：(6 × 激活参数 + MLA 和 KDA 的注意力算子 FLOPs) × tps ÷ bf16 稠密峰值（H100 / H200 989T，B200 2.25P），和 multinode.md 的纸面口径一致。

### 6.2 命令

```sh
bash $K8S top                     # 16 张卡的利用率、SM / Tensor Core 活跃度、显存、功耗、NVLink 和 PCIe 速率，以及正在跑的那一格最新一步（tps、MFU）
bash $K8S watch 5                 # 每 5 秒刷新一次上面这一屏，ctrl-c 退出
bash $K8S report                  # measure 每格一行：步时间、tps、MFU、峰值显存、计算 / 暴露通信 / 拷贝 / 空闲、暴露最多的进程组、IB 收发
bash $K8S trace measure k3_5d     # 一格所有 rank 合成一份 trace，拉回本机，拖进 https://ui.perfetto.dev
bash $K8S tb                      # TensorBoard，浏览器开 http://localhost:6006
bash $K8S fetch h200/probe/k3_5d/tt/profiling/memory_snapshot   # 显存快照，拖进 https://pytorch.org/memory_viz
bash $K8S collect                 # 结果拉回本机（不含 trace 和 cache），每个阶段写 summary.tsv
```

### 6.3 怎么读

- `trace_report.tsv` 每格一行：每个 rank 的 GPU 时间分成四块：
  - 计算：有计算 kernel 在跑，同时在跑的通信算被盖住；
  - 暴露通信：只有 NCCL kernel 在跑；
  - 拷贝：只有拷贝在跑；
  - 空闲：什么都没跑。

  各列的定义见 `K8S_RUN.md` 第 4 节。
- `trace_report_pg.tsv`：暴露通信按进程组拆，标签有 TP、CP、EP、FSDP、EFSDP（专家 FSDP）、DPR（HSDP 副本）、PP、LOSS，另外给出每组的 `eff_busbw`。
- `trace_report_rank.tsv`：每个 rank 一行，用来找慢 rank。
- 两点注意：
  - NCCL kernel 的时间包含等对方的时间，所以 PP 那一行含气泡等待，EP 那一行含路由不均，`eff_busbw` 是下限；
  - 第一次拿到真 trace，先看 `unattributed_pct`。进程组是靠 `record_param_comms` 对上的，只在合成 trace 上测过，不是 0 就要先修 `trace_report.py`。
- 第 5.1 节的"report 里看"一列，就是每一行该看的列；对照组看两行同一列的差值。

## 7. NCCL / RDMA

### 7.1 已经设好的

- **pod：** privileged（能看到 IB 设备）、`IPC_LOCK`、`/dev/shm` 放在内存里、`ulimit -l unlimited`；容器里装了 rdma-core、libibverbs 和 perftest。
- **网卡和 HCA**（`env.sh`，按清单的 `MN_NET`）：
  - Together：bootstrap 和 Gloo 走 `eth0`，NCCL 自己找 HCA；
  - Nebius：另加 `NCCL_IB_HCA=mlx5`、`UCX_NET_DEVICES=eth0`、`SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1`、`NCCL_COLLNET_ENABLE=0`；
  - A100：`NCCL_NVLS_ENABLE=0`。
- **日志级别：** 训练时 `NCCL_DEBUG=WARN`；`comm` 任务用 `INFO`，打印 `INIT`、`NET`、`ENV`、`TUNING` 几个子系统，`TUNING` 会把 NCCL 自己的算法和协议延迟、带宽模型打出来，可以和 α–β 拟合对照。

### 7.2 验证（`comm` 任务自动做）

`bash $K8S comm` 打印每个节点的结论和调优对比表。按这个顺序看：

1. **RDMA 原始带宽：** 每个 IB 口单独用 `ib_write_bw` 测（主机内存，node 1 写到 node 0，不经过 NCCL）。每个口应该接近链路速率；Together 健康检查的阈值是每个设备 320 Gb/s。
2. **`InfiniBand: yes`：** NCCL 用了 IB。如果是 NO，按 `together/README.md` 第 4 节排查。
3. **`GPUDirect RDMA: yes`：** 跨节点的数据直接在 GPU 和网卡之间走，不经过主机内存。如果没看到，查 `setup_node_*.log` 里的 `nvidia_peermem` 那一行。
4. **busbw（256 MiB 以上）：** 16 卡 all_reduce 的参考值：Together 健康检查的多节点阈值 330 GB/s，Nebius 文档里 nccl-tests 的例子平均 317 GB/s。
5. **α–β 拟合：** 每个 op、每个组一行，`alpha us` 是延迟，`busbw` 是大消息的极限。这些是经过 torch.distributed 测的，α 含每次调用的开销，比 nccl-tests 测到的大。

### 7.3 调优扫描（`comm_sweep.tsv`，`MN_COMM_SWEEP=1` 时自动做）

每个变体只改 NCCL 的一个设置，然后完整重测一次 `comm_bench`，结果写在 `runs/<hw>/comm_<tag>/`；`comm_compare.tsv` 把每个变体和 NCCL 默认选择放在一起比。

| tag | 设置 | 看什么 |
|---|---|---|
| `ring` | `NCCL_ALGO=Ring` | NCCL 默认选的算法（大消息 all_reduce 常选 NVLS 或 Tree）是否比 Ring 快 |
| `nvls_off` | `NCCL_NVLS_ENABLE=0` | NVLink SHARP 对节点内 all_reduce / all_gather 的贡献 |
| `simple`、`ll`、`no_ll128` | `NCCL_PROTO=Simple`、`LL`、`^LL128` | 小消息看 α，大消息看 busbw，各自哪个协议最好 |
| `gdr_off` | `NCCL_NET_GDR_LEVEL=LOC` | 关掉 GPUDirect RDMA 以后跨节点 busbw 掉多少，这就是 GDR 的定量收益 |
| `qps4` | `NCCL_IB_QPS_PER_CONNECTION=4` | 每个连接多开 QP，对跨节点带宽和 all-to-all 的影响 |
| `chunk512k` | `NCCL_P2P_NET_CHUNKSIZE=524288` | 跨节点 send / recv 和 all-to-all（EP、PP）的分块大小 |
| `channels32` | `NCCL_MIN_NCHANNELS=32` | 通道数对大消息带宽的影响 |

读 `comm_compare.tsv`：
- `max_vs_default` 大于 1：这个设置让该 op、该组的大消息更快；
- `alpha_us` 更小：小消息更快；
- 一个设置在某个 op 上更快，不代表整步更快，要按第 7.4 节放进训练里验证。

变体的 NCCL 设置如果不被支持（比如某个协议在这个 op 上不可用），这次运行会失败两次，然后被放弃，流水线继续往下跑，不影响其他任务。

### 7.4 放进训练里验证

在矩阵里加一行：度数和长度跟对照行一样，额外变量里加上 NCCL 设置。比如测跨节点 all-to-all 的分块：

```
k3_B_ep16_chunk	dp=2,cp=4,tp=2,ep=16	65536	32768	65536	h200:MN_FSDP_KEEP=1 nosmoke NCCL_P2P_NET_CHUNKSIZE=524288
```

然后：

```sh
bash $K8S sync
bash $K8S scope MN_MODELS=k3 MN_ROWS='k3_B_ep16 k3_B_ep16_chunk' MN_COMM_SWEEP=0
bash $K8S retask
bash $K8S report        # 两行比 step_ms、tps、EP 的 exposed 和 eff_busbw
```

两行共用同一个模型的编译 cache（warm 阶段已经填满），所以它们的差别来自 NCCL 设置本身。`env_node*.txt` 会记下每格实际生效的 `NCCL_*` 变量。

### 7.5 手动单独测一个 NCCL 设置

先 `bash $K8S stop`，然后：

```sh
bash $K8S both "MN_COMM_TAG=qps8 NCCL_IB_QPS_PER_CONNECTION=8 MN_PORT=29950 bash $PK/run_comm.sh"
bash $K8S comm          # 新的 comm_qps8 会出现在对比表里
bash $K8S start
```

## 8. 收尾

1. `bash $K8S status` 显示 `ALL_DONE`（或者你决定停在这里），然后 `bash $K8S collect`，把结果拉回本机。
2. 要带回 logbook 的，放在 `phase14_multinode_5D_experiments/results/<日期>_<hw>/` 下：
   - 每个阶段的 `summary.tsv`；
   - measure 的 `trace_report*.tsv`；
   - `monitor/monitor_summary.tsv`；
   - `comm*/verdict_node*.txt`、`comm*/comm_bench.json` 和 `comm_compare.tsv`；
   - `pipeline/REPORT.txt`；
   - 每格的 `env_node*.txt` 和 `tree.txt`。

   再写一份说明：卡型、网络、titan 提交号、`comm` 的结论。trace 和显存快照太大，需要的格子用 `fetch` 单独拉。
3. 在控制台删集群。集群不能暂停，只能删除；删集群时不勾 "Delete attached shared volume"，共享卷就会保留。
4. 处理共享卷：几天内还要跑，就留着，venv 和编译 cache 下次能直接用；不跑了就删，它一直按月计费。

### 8.1 中途停下，过几天接着跑

Together 的集群不能暂停，停止计费只能删集群。共享卷默认保留，可以挂到同一区域的新集群上，但同一时间只能挂一个集群。kit 的全部状态都在共享卷上（venv、titan 树、编译 cache、结果、流水线进度），容器里只有每次启动都会重装的系统包，所以：

1. （可选）`bash $K8S collect`；
2. 在控制台删集群，不勾 "Delete attached shared volume"；
3. 要接着跑时，在同一个区域建新集群，共享卷选这个已有的卷，卡型和驱动跟原来一样；
4. `tg clusters get-credentials ...`，`kubectl get pv` 确认 PV 名还是卷名，再 `kubectl apply` 同一份清单；
5. pod 起来后跳过 setup，流水线从第一个没完成的任务接着跑；删集群时正在跑的那次尝试算中断，会重跑。

另外几点：
- **只缩抢占式节点：** 不删集群，只把抢占式缩到 0（`tg clusters update <ID> --num-preemptible-gpus 0`，或控制台的 Scale），抢占式那部分会停止计费，但按需节点照样计费。kit 要两个节点，只剩一个也跑不了，所以一般不如删集群。
- **卷没有快照和备份：** 删卷数据就没了；没有集群用它的时候也照样按月计费。重要结果定期 `collect` 回本机。
- **换了卡型：** 在 H100 和 H200 之间换，清单里的 `HW` 要跟着改，结果会写在另一个 `runs/<hw>/` 下。
- **驱动变旧：** 新集群驱动的 CUDA 版本比原来低时，删掉卷上的 `SETUP_OK`，bootstrap 会按新驱动重装 torch。

## 9. 出问题时

| 现象 | 先看 | 怎么办 |
|---|---|---|
| `setup: running` 很久 | `bash $K8S status` 显示的 setup 日志尾部 | 网络慢就等；失败会每 5 分钟自动重试 |
| `InfiniBand: NO` 或 RDMA 原始带宽 failed | `bash $K8S comm`、`setup_node_*.log` 里的 `ib devices` | `together/README.md` 第 4 节 |
| 某格 OOM | `REPORT.txt`、probe 的 `summary.tsv` | probe 会自动减半；被放弃的行在矩阵里缩短后再 `retask` |
| 某格卡住或超时 | `bash $K8S logs`，以及那格的 `node*.log` 和 `tt/comm_traces/` | 看门狗会杀掉对方已经离开的运行；flight recorder 的 dump 里有卡在哪个 collective |
| 节点被抢占 | `bash $K8S status` 里两个节点的心跳 | 自动续跑；节点一直不回来就在控制台看容量 |
| report 的 `unattributed_pct` 不是 0 | `trace_report_pg.tsv` 里 `?` 那一行 | 用 `trace` 拉一格看 NCCL kernel 的参数，修 `trace_report.py` |
| 余额快用完 | Together 控制台 | 先 `collect`；余额用完时集群会被立刻暂停 |
