# k8s 上每节点一个 pod 的无人值守流程（2、4 或 8 个节点；Nebius、Together 共用，2026-11-07）

完整流程从 `PLAYBOOK.md` 开始，这里是运维细节。集群和 pod 怎么准备，见 `nebius/README.md` 或 `together/README.md`。pod 起来以后的事，两家一样，都在这里。**2026-11-08 在 Together 的 2 × 8 H100 上跑过：setup、tokenizer、通信检查由流水线完成，四行 measure 是用 `k8s.sh cell` 手动逐格跑的（设置和清单现在的默认值相同）。清单默认的快速路径（`MN_STAGES=measure`、`MN_COMM=0`、`MN_SEQ_FORCE`）由 `pipeline.py` 自己连续跑完，这一点只在本机模拟过。**

## 1. pod 起来后自动做什么

- **bootstrap.sh**（容器入口）：
  1. 装容器的系统包（每次启动都装，约 1 到 2 分钟）；
  2. 共享存储上的 setup：venv、torch nightly、titan 分支 `phase14_multinode_5d`。只让一个 pod 做，另一个等；
  3. 启动 GPU 和 IB 采样器 `monitor.py`，再启动 `pipeline.py`；流水线崩了就自动重启，直到全部完成或者有 `STOP` 文件。
- **pipeline.py** 按固定顺序跑：
  - `preflight`（默认开，`MN_PREFLIGHT=0` 跳过）：每张卡 30 秒 GEMM 加大消息通信和 RDMA，不过关就停在矩阵之前（`PLAYBOOK.md` 第 4.1 节）；
  - `comm`（`MN_COMM=1` 时）：两节点通信检查，判断 IB 和 GPUDirect RDMA，测 busbw；快速路径不跑，每一格自己的 NCCL INFO 日志（`<格>/nccl/`）给出同样的 IB / GDR 结论；
  - `assets.k3`、`assets.m3`：准备 tokenizer；K3 的转换失败时，自动退到 Qwen3 的 tokenizer；
  - 矩阵的每一行依次跑 `MN_STAGES` 里的阶段（`MN_ORDER=row`，默认）。清单默认只有 measure；完整研究是 smoke、probe、warm、measure。设 `MN_ROWS=k3_5d`，就只跑 5D 全开这一行。
- **断点续跑：** 状态放在共享存储的 `<BASE>/runs/<hw>/pipeline/` 下。pod 被抢占、重建以后，从第一个没完成的任务接着跑；被打断的那次尝试不计入失败。
- **自动调长度：** probe 遇到 OOM 时把长度减半重试（最低 4096），warm 和 measure 用 probe 最后定下的长度。这一行没有 probe 任务时（快速路径），warm 和 measure 自己在 OOM 后减半重试。
- **失败处理：** 一格连续真失败 2 次（`MN_MAX_FAILS`），就放弃这一格，同时跳过它后面的阶段，继续下一格。
- **收尾：** 写出 `REPORT.txt`、每个阶段的 `summary.tsv`、采样汇总、measure 的 trace 报告（第 4 节）、成对对照 `pairs_report_<model>.tsv` 和 `ALL_DONE`。pod 不退出，方便收集结果。

## 2. 手动操作（本机，在 logbook 根目录下）

```sh
K8S=phase14_multinode_5D_experiments/kit/k8s.sh
bash $K8S status       # pod、setup、每个节点的心跳、最近的决策；跑完后显示 REPORT.txt
bash $K8S logs         # 每个节点的流水线日志
bash $K8S collect      # 结果拉回本机（不含 trace 和 cache），每个阶段写 summary.tsv；隔几个小时收一次
```

- **改矩阵**（长度、加行、额外变量）：在本地改 `kit/matrix_*.tsv`，然后 `bash $K8S sync`，再 `bash $K8S retask`。已经完成的任务不会重跑；要重跑某一格，先删掉 `pipeline/done/<任务名>`（放弃过的删 `giveup/` 和 `fails/` 里的同名文件），再 `retask`。
- **改范围**（`MN_ROWS`、`MN_MODELS`、`MN_STAGES`、`MN_ORDER`、`MN_COMM_SWEEP`）：`bash $K8S scope MN_ROWS='k3_5d k3_B' MN_MODELS=k3`，再 `bash $K8S retask`；`bash $K8S scope` 不带参数是撤销。要改 `HW` 这类清单里的值，就改完清单后 `kubectl delete job p14`，再 `kubectl apply -f <清单>`，最后 `retask`。状态都在共享存储上，删 Job 不会丢。
- **改脚本**（`launch.sh`、`run_comm.sh` 等）：`bash $K8S sync` 即可，从下一次尝试开始生效。titan 分支有新提交时（`env.sh` 的 `BRANCH_SHA` 变了），`sync` 之后再 `retask`：bootstrap 发现树不在这个提交上，会重跑 setup，把树切过去。
- **停和启：**
  - `bash $K8S stop`：当前这次尝试跑完再停（所有节点都退出，任务的进度保留）；
  - `bash $K8S kill`：立刻停，当前这次尝试作废，以后重跑；
  - `bash $K8S start`：重新开始。
- **手动跑一格：** 先 `stop` 或 `kill`，再用 `bash $K8S both '<命令>'` 让所有 pod 同时执行，例如：

```sh
bash $K8S both "MN_MODEL=k3 MN_FLAVOR=k3_16x141g MN_ASSETS=/mnt/shared/p14/assets/kimi_k3_tokenizer \
  MN_CELL=pp=2,dp=2,cp=2,tp=2,ep=8 MN_SEQ=16384 MN_STEPS=10 MN_PORT=29700 \
  bash /mnt/shared/p14/logbook/phase14_multinode_5D_experiments/kit/launch.sh /mnt/shared/p14/runs/manual/try1"
```

## 3. 文件在哪（都在共享存储，`BASE=/mnt/shared/p14`）

- `runs/<hw>/<stage>/<行名>/`：每格的结果。里面有 `node0.log`、`node1.log`、`env_node*.txt`、`tree.txt`；trace 在 `tt/profiling/traces/`，显存快照在 `tt/profiling/memory_snapshot/`，TensorBoard 在 `tt/tb/`。被打断或失败的尝试移到 `<stage>/_attempts/`。
- `runs/<hw>/measure/trace_report*.tsv`：trace 报告（第 4 节）。
- `runs/<hw>/monitor/`：每个节点的采样 `gpu_node*.csv`、`ib_node*.csv`，以及 `monitor_summary.tsv`（每个节点、每个任务一行）。
- `runs/<hw>/comm/`：`verdict_node0.txt`（IB、GDR、busbw）、`comm_bench.json`、NCCL 的 INFO 日志。
- `runs/<hw>/pipeline/`：流水线的状态：
  - `tasks.json`：冻结的任务列表；
  - `done/`、`giveup/`：每个任务的结局；
  - `go/`、`rc/`：每次尝试的通知和每个节点各自的退出码；
  - `hb/`：心跳；
  - `len/`、`try/`：probe 定下的长度和正在试的长度；
  - `log_node*.txt`：流水线日志；`run_node*.log`：每次尝试的启动输出；
  - `REPORT.txt`、`ALL_DONE`。
- `setup_*.log`、`setup_node_*.log`、`pipeline_<hostname>.log`：setup 和 bootstrap 的日志。

## 4. 看性能：MFU、显存、通信

运行中自动记录的：
- **每步指标：** 每个 rank 每步的 loss、tps、TFLOPS、MFU、峰值显存，都在日志里。MFU 是 titan 的口径：(6 × 激活参数 + MLA 和 KDA 的注意力算子 FLOPs) × tps ÷ bf16 稠密峰值，和 multinode.md 的纸面口径一致。
- **显存快照：** probe 的第 2 步，每个节点的 local rank 0 各存一份；OOM 时也存。
- **trace 和 TensorBoard：** measure 的第 `PROFILE_FREQ` 步（快速路径第 5 步，默认第 25 步），所有 rank 都录 CPU 和 CUDA 事件；measure 同时写 TensorBoard。
- **节点采样：** 每个节点每 2 秒记一次 GPU（利用率、显存、功耗、SM 频率、降频原因）和每个 IB / RoCE 口的收发速率，每行标着当时在跑的任务。
- **通信基线：** `comm` 从 256 B 扫到 1 GiB，并给每个 op、每个组拟合 α–β。

在本机看（logbook 根目录下）：

```sh
bash $K8S top                     # 每个节点最新的 GPU 和 IB
bash $K8S report                  # measure 每格一行，各列见下
bash $K8S trace measure k3_5d     # 一格所有 rank 合成一份 trace 拉回本机，拖进 https://ui.perfetto.dev
bash $K8S tb                      # TensorBoard，浏览器开 http://localhost:6006
bash $K8S fetch h200/probe/k3_5d/tt/profiling/memory_snapshot   # 显存快照，拖进 https://pytorch.org/memory_viz
```

`report` 写在 `runs/<hw>/measure/` 下，读法：
- `trace_report.tsv`，每格一行：
  - `step_ms`、`tps`、`mfu`、`peak_mem_gib`（`peak_rank` 是哪个 rank）；
  - 每个 rank 的 GPU 时间分四块：有计算 kernel 在跑算计算（同时在跑的通信算被盖住），只有 NCCL kernel 在跑算暴露通信，只有拷贝在跑算拷贝，什么都没跑算空闲。`compute_pct`、`exposed_pct`、`copy_pct`、`idle_pct` 是它们占步时间的比例（rank 平均，`_max` 是最差的 rank），`overlap_pct` 是通信被盖住的比例；
  - `top_exposed`：暴露通信最多的三个进程组，各占步时间多少；
  - `ib_tx_GBps`、`ib_rx_GBps`：这一格运行期间，每个节点 IB 的平均收发速率。
- `trace_report_pg.tsv`：按进程组和 collective 拆。标签有 TP、CP、EP、FSDP、EFSDP（专家 FSDP）、DPR（HSDP 的副本）、PP、LOSS；`eff_busbw` 是消息字节数除以 kernel 时间，再按 nccl-tests 的系数折成 busbw，可以和 `comm` 的结果比。
- `trace_report_rank.tsv`：每个 rank 一行，用来找慢 rank。
- `trace_report_groups.tsv`：每个进程组的哈希名、标签和成员 rank。Perfetto 里点一个 NCCL kernel，参数里的 `Process Group Name` 就在这里查。
- `trace_report_kernels.tsv`：每个 rank 的 kernel 时间按类分（GEMM、注意力、KDA、aten 逐元素、NCCL 等），计算流上 kernel 的个数和大小，PP 前向、反向动作的 GPU 时间；`trace_report_topk.tsv`：每格耗时最多的 20 个 kernel。读法和练习见 `TRACE_READING.md`。
- 两点注意：
  - NCCL kernel 的时间包含等对方的时间，所以 PP 那一行含气泡等待，EP 那一行含路由不均，`eff_busbw` 是下限；
  - trace 里的进程组只有哈希名、组大小和组内 rank，`trace_report.py` 按格子的 mesh 推出每个组的成员，再标成 TP、CP、FSDP、EP 这些名字；几个维度成员相同时合成一个标签（如 `FSDP/LOSS`）。`unattributed_pct` 不是 0，说明有 NCCL kernel 没对上；11-08 的四格都是 0。

## 5. 被抢占时会发生什么

- 一个节点没了，另一个节点上的运行在 2 到 3 分钟内被看门狗杀掉，这次尝试记为"中断"。
- Job 会补 pod，但新 pod 要等有节点才能调度：Together 在有容量时会自动补节点；Nebius 的抢占式 VM 只会停机，节点能不能回来，看 `nebius/README.md` 第 5 节。
- 新 pod 先装容器包，共享 setup 已经完成就直接跳过，然后接着跑流水线。
- 损失只是被打断的那一格，几分钟到几十分钟。

## 6. 第一次先看什么

1. `bash $K8S status` 显示 `setup: done`。
2. 网络：快速路径看第一格 `summary.tsv` 的 `net` 列，要是 `IB+GDR`；跑了 `comm` 的话看它的结论（`status` 也会显示），要有 `InfiniBand: yes` 和 `GPUDirect RDMA: yes`。16 卡 all_reduce 的大消息 busbw 参考：Nebius 文档里 nccl-tests 的例子平均 317 GB/s（文档说高于 300 正常），Together 健康检查的多节点阈值是 330 GB/s。
3. 第一格跑完（快速路径是 `k3.measure.k3_5d: done`），`bash $K8S dash` 里没有过热降频的卡（`PLAYBOOK.md` 第 4.3 节）。
4. 之后定期 `collect`。

`InfiniBand: NO` 时，按各自 README 里的排查步骤处理。
