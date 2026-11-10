# Phase 14 完整流程：Together 上 2 × 8 卡的 5D 实验（2026-11-08 实跑后定稿）

2026-11-08 在 Together 的 2 × 8 H100 上实跑过一次，结果和 trace 的读法在 `../results/20261108_h100_together/README.md`。**以后租多节点，默认走第 0 节的快速路径：setup 完成后直接 measure 矩阵的每一行，不做单独的环境测试。** 四个阶段的完整研究（smoke / probe / warm / measure、通信基线、NCCL 调优扫描）在第 4.2 节，需要时再打开。

其他文件只放细节：
- `together/README.md`：建集群、kubeconfig、IB 排查；
- `K8S_RUN.md`：运维细节、状态文件、report 各列的定义；
- `RUNBOOK.md`：不用 k8s、在裸机或 VM 上手动跑的版本；
- `README.md`：titan 分支、两个缩小版模型、矩阵的来由；
- `../multinode.md`：纸面规划和每格的预期。

下文的命令都在本机 logbook 根目录下执行（Windows 上用 Git Bash），先设：

```sh
K8S=phase14_multinode_5D_experiments/kit/k8s.sh
PK=/mnt/shared/p14/logbook/phase14_multinode_5D_experiments/kit    # pod 里 kit 的位置
```

## 0. 快速路径（默认）

| 步骤 | 在哪 | 做什么 | 11-08 实际用时 |
|---|---|---|---|
| 1 | 本机 | 充值，装 `tg` 和 `kubectl`，清单里填 `volumeName` 和 `HW` | 10 分钟 |
| 2 | Together 控制台 | 建集群：2 个节点（32 / 64 卡见第 0.1 节）、Kubernetes、共享卷 | 等资源 |
| 3 | 本机 | `get-credentials`，`kubectl apply` | 5 分钟 |
| 4 | pod 里自动 | 容器包、共享 setup、tokenizer | 约 15 分钟 |
| 5 | pod 里自动 | K3 矩阵 9 行，每行 measure 一次：5 步，第 5 步出所有 rank 的 trace；OOM 的行长度减半重跑 | 每行 3 到 7 分钟，9 行约 50 分钟 |
| 6 | 本机 | 跑的时候看 dashboard；`ALL_DONE` 后 `report`、整份拉回本机、删集群 | 15 分钟 |

从 `apply` 到 `ALL_DONE` 约 1 小时 15 分钟。16 张 H100 抢占式每小时约 $32，一轮约 $50，另加等资源和收尾的时间。

清单 `together/p14_pods.yaml` 已经写好的设置，也就是 11-08 跑通的配置：

| 变量 | 值 | 作用 |
|---|---|---|
| `MN_STAGES` | `measure` | 只跑 measure。smoke 只证明布局能起来，probe 只量显存，warm 只填编译 cache；不编译、长度固定时这三步都用不上 |
| `MN_COMM` | `0` | 不跑单独的通信检查。每次运行都把 NCCL 的 INFO 日志写进 `<格>/nccl/`，`summary.tsv` 的 `net` 列写着这一格走的是 `IB+GDR`、`IB` 还是 `Socket`；`0` 同时关掉 NCCL 调优扫描 |
| `MN_SEQ_FORCE` | `16384` | 所有行用 16K；OOM 的行自动减半，最低 4096 |
| `MN_COMPILE` | `0` | eager：K3 的局部编译区不编译，flex attention 不 autotune。每种布局省掉编译时间；MFU 偏低，但各行同口径，可以互相比 |
| `MN_VPP` | `2` | PP 的行用 `Interleaved1F1B`，每个 rank 2 个虚拟 stage（K3 80G 缩小版每个虚拟 stage 8 层） |
| `MEAS_STEPS`、`PROFILE_FREQ` | `5`、`5` | 跑 5 步，第 5 步出 trace（profiler 的 warmup 3 步加 active 1 步正好放得下） |
| `MN_ROWS` | 空 | 矩阵的每一行；只跑几行就写行名，用空格隔开 |

NCCL 超时清单不设，用 titan 的默认：初始化和第 1 步 300 秒，之后 100 秒。11-08 的四格设过 `MN_NCCL_TIMEOUT=15`，但 measure 要写 trace，recipe 会把它抬到至少 120 秒，所以实际是 120 秒（`node0.log` 的 `[multinode_5d]` 行打印 `nccl_timeout=300/120s`）。

全部命令：

```sh
# 本机：清单里填 volumeName（kubectl get pv 里和共享卷同名的那个）和 HW（h100 / h200 / b200）
tg clusters list
tg clusters get-credentials <CLUSTER_ID> --set-default-context --no-json
kubectl get pv
bash $K8S apply 2         # 节点数：2、4 或 8

# 跑的时候
bash $K8S status          # setup、每个节点的心跳、流水线最近的决定；跑完后打印 REPORT.txt
bash $K8S dash            # 浏览器开 http://localhost:8765（第 4.3 节）
bash $K8S watch 5         # 终端版：16 张卡的 DCGM 和正在跑的那一格的最新一步

# ALL_DONE 之后
R=phase14_multinode_5D_experiments/results/<日期>_h100_together
bash $K8S report                    # measure 每格一行的时间拆分（trace_report*.tsv）
bash $K8S fetch h100 $R/raw         # runs/h100 整份拉回本机，含 trace（11-08 是 7.4 GB），raw/ 不进 git
```

拉回本机以后的分析（第 6.2 节）不需要集群，确认 `raw/` 完整以后就可以删集群。

### 0.1 32 卡和 64 卡（4 个或 8 个节点）

- **拉起：** 建集群时选 4 或 8 个节点，用 `bash $K8S apply 4` 或 `apply 8` 拉起。它把清单里的 `completions`、`parallelism` 和 `NNODES` 一起改掉。其余步骤和清单的快速路径设置都不变。
- **矩阵和输出：** 4、8 节点各有自己的一份：
  - 矩阵是 `matrix_k3_n4.tsv`、`matrix_k3_n8.tsv`。只有 K3，M3 只在 2 节点上跑；
  - 输出在共享卷的 `runs_n4/`、`runs_n8/`，和 2 节点的 `runs/` 分开，同一个卷上可以先后跑不同的节点数；
  - `k8s.sh` 按当前 Job 的节点数自动找对目录，`fetch`、`report`、`trace` 的用法和 2 节点一样。
- **前提不变：** 每步 524,288 token，dp 2（或 1）。多出来的卡给跨节点的 PP 和更大的模型，TP、CP、EP 默认留在节点内（第 5.1 节）。
- **模型：** 下表是 titan 分支 `c8c229b59` 加的 flavor，宽度和 16 卡的相同，每卡的参数量大致不变。H200 / B200 用对应的 `141g` 版本。

| flavor | 层数 | 专家 | AttnRes block | 总参数（估） | 用在 |
|---|---|---|---|---|---|
| `k3_16x80g` | 29 | 128 | 4 | 14.5B | 2 节点；32、64 卡的 `_ref` 行（同一个模型，更多卡） |
| `k3_32x80g` | 61 | 128 | 8 | 约 30B | 4 节点的默认 |
| `k3_32x80g_e256` | 29 | 256 | 4 | 约 26B | 4 节点的 EP 对照 |
| `k3_64x80g` | 93 | 128 | 12 | 约 46B | 8 节点的默认：K3 的真实层数和注意力排布（每 4 层一个 MLA，加最后一层） |
| `k3_64x80g_e512` | 29 | 512 | 4 | 约 50B | 8 节点的 EP 对照 |

总参数是按 16 卡版本的参数分布推算的，没在机器上数过。

- **时间和费用**（纸面估计）：
  - 每行 5 到 10 分钟；16K 放不下的行会自动减半重跑，每次多几分钟；
  - 4 节点 10 行、8 节点 9 行，各约 1 到 1.5 小时；
  - 32 卡每小时约 $64，64 卡约 $128；
  - 余额门槛按卡数涨：抢占式目标 4 小时，32 卡约 $256，64 卡约 $511。
- **还没在真集群上跑过：** N 个节点的协同只在本机模拟过。8 节点下，正常完成、OOM 减半、杀 follower、杀 leader、STOP 都通过了。

## 1. 开始前（本机）

1. **余额：** 创建时余额要够"抢占式目标的 4 小时 + 按需部分的最短时长 + 同期存储"，16 卡约 $128 起，快速路径一轮充 $200 够用。余额用完，集群立刻暂停，宽限期后删除，而且扣完之前没有提醒。
2. **工具：**
   - Together CLI：`uv tool install "together[cli]" --upgrade`；
   - `kubectl`：Windows 上 `winget install Kubernetes.kubectl`。`k8s.sh` 会自己找 winget 装的位置，所以新开的终端里没有 `kubectl` 也能用；
   - Windows 上所有 `bash $K8S ...` 都在 Git Bash 里跑，`k8s.sh` 已经关掉了 Git Bash 对 `/mnt/...` 参数的路径改写。
3. **改清单** `phase14_multinode_5D_experiments/kit/together/p14_pods.yaml`，只有两处：
   - `volumeName`：共享卷对应的 PV 名，第 3 步查；
   - `HW`：`h100`、`h200` 或 `b200`。控制台标价 $1.995/GPU·时的是 H100 抢占式。写错卡型会按另一种卡的缩小版和长度跑。

kit 支持 2、4、8 个节点（16、32、64 卡），4 和 8 节点见第 0.1 节。

## 2. 建集群（Together 控制台）

- 抢占式 16、32 或 64 卡，也就是 2、4 或 8 个节点；编排选 Kubernetes。驱动用默认的（11-08 是 610 / CUDA 13.3）。
- 如果控制台要求至少一个按需（standard）节点，就建 1 个按需节点加 1 个抢占式节点，两个 pod 一边一个，清单里的 `nodeSelector` 保持注释。
- 共享卷 500 GiB 左右，跟集群一起建，记下卷名。卷能扩不能缩。

细节和依据见 `together/README.md` 第 0、1 节。

## 3. 拉起 pod（每个节点一个）

```sh
tg clusters list
tg clusters get-credentials <CLUSTER_ID> --set-default-context --no-json
kubectl get nodes -L together.ai/compute-class
kubectl get pv                                          # 共享卷同名的那个 PV，填进清单的 volumeName
kubectl create secret generic hf-token --from-literal=token=<hf_token>   # 可选
bash $K8S apply 2                                       # 节点数：2、4 或 8
bash $K8S status
```

## 4. pod 起来以后自动做的事

### 4.1 快速路径（清单的默认值）

不用手动操作，按顺序：

1. **容器包**（每次启动，1 到 2 分钟）：git、编译器、rdma-core、libibverbs、perftest。
2. **共享 setup**（第一次约 10 到 15 分钟；一个 pod 装，另一个等）：venv、torch nightly、titan 分支 `phase14_multinode_5d` 固定在 `env.sh` 的 `BRANCH_SHA`。共享卷上已有 `SETUP_OK` 并且提交号一致时跳过。
3. **采样器** `monitor.py`：每个节点每 2 秒记一次每张卡的利用率、显存、功耗、SM 频率、温度和降频原因，之后一直开着。
4. **tokenizer**（约 2 分钟）。K3 的转换失败时，自动退到 Qwen3 的 tokenizer。
5. **矩阵的每一行跑一次 measure**：缩小版模型，c4，16K，5 步，第 5 步所有 rank 写 trace，TensorBoard。每步约 512K token，随机初始化，AdamW，fp32 主参数加 bf16 计算。这是吞吐和通信的测量，不是真训练。OOM 的行减半重跑，失败两次的行放弃，流水线接着跑下一行。
6. **收尾：** 写 `measure/summary.tsv`、`monitor_summary.tsv`、measure 的 `trace_report*.tsv`、`REPORT.txt` 和 `ALL_DONE`。之后 pod 空转，但照样计费。

被抢占时，换上来的 pod 会从第一个没完成的任务接着跑，被打断的那次不算失败（`K8S_RUN.md` 第 5 节）。

### 4.2 完整研究（按需打开）

清单里改这几项（或者 pod 起来以后用 `bash $K8S scope ...` 再 `bash $K8S retask`）：

| 变量 | 改成 | 多出来的 |
|---|---|---|
| `MN_STAGES` | `smoke probe warm measure` | 每行四个阶段（下表） |
| `MN_COMM` | `1` | 第一个任务是通信检查（第 7.2 节），矩阵跑完后做 NCCL 调优扫描（第 7.3 节，约 20 分钟；`MN_COMM_SWEEP=0` 单独关掉扫描） |
| `MN_SEQ_FORCE` | 删掉 | 用矩阵里每种卡的长度，probe 定下每行能放下的长度 |
| `MN_COMPILE` | 删掉 | 编译局部区域，flex attention autotune；MFU 是调过的口径，每种布局多十几分钟编译 |
| `MEAS_STEPS`、`PROFILE_FREQ` | 删掉 | 30 步，第 25 步出 trace |

| 阶段 | 模型 | 数据 | 步数 | 顺带记录 |
|---|---|---|---|---|
| smoke | debug 模型，K3 4096 / M3 2048 token | c4_test | 5 | 每种布局能起来、能走步 |
| probe | 缩小版，矩阵里的长度 | c4_test | 3 | 每个 rank 的峰值显存；OOM 时长度减半重试，最低 4096；第 2 步存显存快照 |
| warm | 缩小版，probe 定下的长度 | c4 | 10 | 填满这个模型所有行共用的编译 cache |
| measure | 同上 | c4 | 30 | 第 25 步所有 rank 的 profiler trace，TensorBoard |

两个模型、全矩阵、四个阶段约 8 到 10 小时（纸面估计）。

### 4.3 跑的时候看什么

- **`bash $K8S dash`**（浏览器开 `http://localhost:8765`）：
  - 正在跑的那一格每步的 MFU、tps 和步时间；
  - 16 个 rank 各自的最新一步；非最后一个 PP stage 的 rank 没有 loss，显示 "no loss stage"；
  - 每张卡的 DCGM：SM 活跃度、Tensor Core 活跃度、显存、功耗、NVLink 和 PCIe 速率；
  - 每个节点的训练日志（可以只看步数行或报错），通信结论，这一格的 profiler 文件。
- **`bash $K8S tail 0|1 [steps]`**：在写日志的那个 pod 里跟踪 node 0 或 node 1 的日志。不要从另一个 pod 读共享卷上的日志，NFS 的缓存会让它停在旧的长度。
- **显存不均匀是正常的**：1F1B 下第一个 PP stage 存的激活最多，11-08 的 `k3_A` 是 node 0 62 GiB，node 1 52 GiB。
- **先看有没有过热降频的卡：**
  - 11-08 node 1 的 GPU 5（rank 13）在四格 measure 里都到 88 到 90°C，其他 15 张卡不超过 58°C；
  - 它有 18% 到 40% 的采样处于软件热降频（原因码 `0x20`），SM 频率最低掉到 345 MHz；
  - 每个 collective 都要等最慢的 rank，所以一张卡降频，整格都跟着慢；
  - 能看出这种卡的地方：`monitor_summary.tsv` 的 `sm_mhz_min` 和 `throttle_pct`，`gpu_node*.csv` 的 `temp_c` 和 `reasons`，`trace_report_rank.tsv` 里 `compute_ms` 特别大的 rank；
  - 看到了就在结果里注明，严重的话找 Together 换节点。

### 4.4 跑完以后（`ALL_DONE`）

1. `bash $K8S report`：measure 每格一行的时间拆分。
2. `bash $K8S fetch h100 $R/raw`：整份拉回本机（日志、trace、TensorBoard、NCCL 日志、监控）。再把共享卷顶层的 setup、流水线、采样器日志和 `scope.env` 拉进 `$R/raw/top/`：

   ```sh
   mkdir -p $R/raw/top
   P=$(kubectl get pods -l job-name=p14 -o name | head -n 1)
   kubectl exec $P -- bash -c 'cd /mnt/shared/p14 && tar -cf - *.log scope.env 2>/dev/null' | tar -C $R/raw/top -xf -
   ```

3. 本机检查每个 trace 能不能读、出报告、合成 trace（第 6.2 节），写 `$R/README.md`。
4. 确认 `raw/` 完整（文件数和大小与共享卷一致）后，在控制台删集群（第 8 节）。

**不要在 rank 写 trace 时杀进程。** 最后一步之后每个 rank 要花 1 分钟以上写 trace，这时杀掉会留下截断、读不了的文件。11-08 的 4096 那次 `k3_5d` measure 就是这样丢了全部 16 个 trace。`bash $K8S kill` 现在会在还有 rank 在写 trace 时拒绝执行（`kill force` 强制）。

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

32 卡（`matrix_k3_n4.tsv`，默认 `k3_32x80g`，61 层）：

| 行 | 度数 pp·dp·cp·tp·ep | 模型 | 跨节点的是 | 用来回答 |
|---|---|---|---|---|
| `k3_A4_ref`、`k3_A4_ref_v4` | 4·2·1·4·8 | `k3_16x80g`，VPP 2 / 4 | PP | 同一个模型从 pp 2 到 pp 4 的代价，和 11-08 的 `k3_A` 比 |
| `k3_A4`、`k3_A4_v4` | 4·2·1·4·8 | 61 层，VPP 2 / 4 | PP | 更深的模型，PP 跨 4 个节点 |
| `k3_5d4` | 4·2·2·2·8 | 61 层 | PP | 5D 全开 |
| `k3_C4` | 4·2·4·1·8 | 61 层 | PP | 不用 TP，CP 4 |
| `k3_D4` | 4·1·4·2·8 | 61 层 | PP | 没有 FSDP，32 个 micro-batch |
| `k3_E4_ep8`、`k3_E4_ep16` | 2·2·2·4·8 / 16 | 256 个专家 | PP；`_ep16` 的 EP 跨两个节点 | EP 留在节点内，对 EP 跨节点 |
| `k3_L4` | 1·2·4·4·8，64K | 61 层 | CP（CP 4 × TP 4 跨两个节点）、FSDP | 长上下文下 CP 跨节点 |

64 卡（`matrix_k3_n8.tsv`，默认 `k3_64x80g`，93 层）：

| 行 | 度数 pp·dp·cp·tp·ep | 模型 | 跨节点的是 | 用来回答 |
|---|---|---|---|---|
| `k3_A8_ref` | 8·2·1·4·8 | `k3_16x80g`，VPP 2 | PP | 同一个模型到 pp 8（29 层最多 VPP 2） |
| `k3_A8`、`k3_A8_v2` | 8·2·1·4·8 | 93 层，VPP 4 / 2 | PP | K3 的 32 个虚拟 stage（每个 3 层）；VPP 4 对 VPP 2 |
| `k3_5d8` | 8·2·2·2·8 | 93 层，VPP 4 | PP | 5D 全开 |
| `k3_C8` | 8·2·4·1·8 | 93 层，VPP 4 | PP | 不用 TP，CP 4 |
| `k3_D8` | 8·1·4·2·8 | 93 层，VPP 4 | PP | 没有 FSDP |
| `k3_E8_ep8`、`k3_E8_ep16` | 4·2·2·4·8 / 16 | 512 个专家 | PP；`_ep16` 的 EP 跨两个节点 | EP 留在节点内，对 EP 跨节点 |
| `k3_L8` | 1·2·8·4·8，128K | 93 层 | CP（CP 8 × TP 4 跨四个节点）、FSDP | 长上下文下 CP 跨节点 |

32 / 64 卡的两点提醒：
- **气泡：** 16K 时每个 dp rank 有 16 个 micro-batch，Interleaved 1F1B 的气泡约为 (pp−1)/(v·m)：pp4·v2 9%，pp4·v4 5%，pp8·v2 22%，pp8·v4 11%。长度减半时 micro-batch 翻倍，气泡减半。
- **显存：** 1F1B 下第一个 stage 存的激活和总层数成正比，与 pp 无关。61 层、93 层在 16K 大概率放不下，会自动减半重跑。

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

长度都是纸面估计，probe 会自动缩短放不下的行。快速路径不用这几列，所有行都跑 `MN_SEQ_FORCE`（16K）。11-08 在 H100、eager、VPP 2 下，`k3_A`、`k3_B`、`k3_C`、`k3_D` 的 16K 都放得下（峰值显存最多 62 GiB）。

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
- `MN_VPP=<v>`：PP 的格子用 `Interleaved1F1B`，每个 rank v 个虚拟 stage；层数切不出这么多时降到能整除的最大值（K3 缩小版 29 层在 pp 2 下能到 4，debug 小模型只能到 2），实际值打印在 `node0.log` 的 `[multinode_5d]` 那一行；
- `MN_NCCL_TIMEOUT=<秒>`：第 1 步以后的 NCCL 超时；有 rank 在存显存快照或 trace 的阶段（probe、measure）不低于 120 秒，否则会把正常的运行判成超时；`MN_NCCL_INIT_TIMEOUT` 管初始化和第 1 步（默认 300）；
- `MN_SEQ_FORCE=<n>`（清单或 `scope`）：所有行的 probe、warm、measure 都用这个长度，代替矩阵和 probe 定下的长度；没有 probe 的时候，warm 和 measure 自己在 OOM 后减半重试；
- `MN_COMPILE=0`：K3 的局部编译区改成 eager，flex attention 不做 autotune（flex 不能完全不编译，长序列 eager 会 OOM）；省编译时间，但 kernel 没调优，MFU 偏低；
- 任何 `NCCL_*` 变量（第 7.4 节）；
- `MN_FLAVOR=k3_<卡数>x{mem}[_e<专家数>]`：这一行换一个 flavor，`{mem}` 按卡型换成 `80g` 或 `141g`；
- `MN_SEQ=<n>`：固定这一行的长度，`MN_SEQ_FORCE` 不覆盖它（长上下文的行用）；
- `h200:VAR=值`：只在这种卡上生效；
- `nosmoke`：跳过 smoke。

例：

```
k3_5d_vpp	pp=2,dp=2,cp=2,tp=2,ep=8	32768	16384	24576	MN_SCHEDULE=Interleaved1F1B
```

要重跑一个已完成的任务，先删掉共享卷上 `runs/<hw>/pipeline/done/<任务名>`，再 `retask`。

**按行手动逐格跑**（流水线先 `stop` 或 `kill`）：`bash $K8S cell <阶段> <行名> [VAR=值 ...]` 用流水线对这一行这一阶段的同一套设置（tokenizer、长度、NCCL 网卡、显存快照、trace），在所有 pod 上同时跑一格，跑完打印退出码和这一格的 `summary`，期间每 20 秒打印最新一步。结果写在 `runs/<hw>/manual/<阶段>/<行名>_<时间>/`，measure 的拆分用 `bash $K8S report manual/measure`。手动跑时 probe 不会自动减半：OOM 了就加 `MN_SEQ=<一半>` 重跑，warm 和 measure 也带上同样的 `MN_SEQ`。

```sh
bash $K8S cell smoke k3_5d
bash $K8S cell probe k3_5d
bash $K8S cell warm k3_5d
bash $K8S cell measure k3_5d
bash $K8S cell probe k3_B MN_SEQ=32768     # 例：长度减半重跑
```

**任意度数手动跑一格**（不进矩阵）：先 `bash $K8S stop`，等当前这次尝试结束，再让所有 pod 同时执行：

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
| 所有 rank 的 CPU 和 CUDA trace | measure 第 `PROFILE_FREQ` 步（快速路径第 5 步，完整研究第 25 步） | `measure/<行>/tt/profiling/traces/iteration_<步>/` |
| TensorBoard（最后一个 PP stage 的 rank） | measure | `measure/<行>/tt/tb/` |
| 每个节点的 GPU 和 IB / RoCE 速率，标着任务 | 一直 | `monitor/gpu_node*.csv`、`ib_node*.csv` |
| NCCL flight recorder（卡死时 dump） | 所有运行 | `<stage>/<行>/tt/comm_traces/` |
| NCCL 的 INFO 日志（`INIT`、`NET`：用了哪些 HCA、IB 还是 socket、有没有 GDR） | 所有运行 | `<stage>/<行>/nccl/<host>.<pid>.log`，结论在 `summary.tsv` 的 `net` 列 |

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
bash $K8S fetch h100 $R/raw       # 整份拉回本机（含 trace），之后的分析都在本机做
```

本机分析（集群删掉以后也能做；`M=$R/raw/h100/measure`，手动跑的格子在 `$R/raw/h100/manual/measure`）：

```sh
for f in $M/*/tt/profiling/traces/*/rank*_trace.json.gz; do gzip -t $f || echo "truncated: $f"; done
python phase14_multinode_5D_experiments/kit/trace_report.py $M --jobs 4     # 写 $M/trace_report*.tsv
python phase14_multinode_5D_experiments/kit/summarize.py $M > $R/measure_summary.tsv
python phase14_multinode_5D_experiments/kit/merge_traces.py $M/<格> --out $R/<格>_measure_trace.json.gz   # 拖进 ui.perfetto.dev
tensorboard --logdir $M/<格>/tt/tb
```

截断的 trace 读不了，`trace_report.py` 和 `merge_traces.py` 会跳过它们并说明是哪些 rank。Windows 上先 `export PYTHONUTF8=1`。

### 6.3 怎么读

- `trace_report.tsv` 每格一行：每个 rank 的 GPU 时间分成四块：
  - 计算：有计算 kernel 在跑，同时在跑的通信算被盖住；
  - 暴露通信：只有 NCCL kernel 在跑；
  - 拷贝：只有拷贝在跑；
  - 空闲：什么都没跑。

  各列的定义见 `K8S_RUN.md` 第 4 节。
- `trace_report_pg.tsv`：暴露通信按进程组拆，标签有 TP、CP、EP、FSDP、EFSDP（专家 FSDP）、DPR（HSDP 副本）、PP、LOSS，另外给出每组的 `eff_busbw`。
- `trace_report_rank.tsv`：每个 rank 一行，用来找慢 rank。
- `trace_report_groups.tsv`：每个进程组的哈希名、标签和成员 rank。Perfetto 里点一个 NCCL kernel，参数里的 `Process Group Name` 就在这里查。
- 两点注意：
  - NCCL kernel 的时间包含等对方的时间，所以 PP 那一行含气泡等待，EP 那一行含路由不均，`eff_busbw` 是下限；
  - trace 里的进程组只有哈希名、组大小和组内 rank，`trace_report.py` 按格子的 mesh 推出每个组的成员，标成 TP、CP、FSDP、EP 这些名字；几个维度成员相同时合成一个标签（如 `FSDP/LOSS`、`CPxTP/EP`）。11-08 的四格 `unattributed_pct` 都是 0。
- 第 5.1 节的"report 里看"一列，就是每一行该看的列；对照组看两行同一列的差值。
- 一步一步怎么在 Perfetto 里看 trace、11-08 每格的数和结论：`../results/20261108_h100_together/README.md`。

## 7. NCCL / RDMA

### 7.1 已经设好的

- **pod：** privileged（能看到 IB 设备）、`IPC_LOCK`、`/dev/shm` 放在内存里、`ulimit -l unlimited`；容器里装了 rdma-core、libibverbs 和 perftest。
- **网卡和 HCA**（`env.sh`，按清单的 `MN_NET`）：
  - Together：bootstrap 和 Gloo 走 `eth0`，NCCL 自己找 HCA；
  - Nebius：另加 `NCCL_IB_HCA=mlx5`、`UCX_NET_DEVICES=eth0`、`SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1`、`NCCL_COLLNET_ENABLE=0`；
  - A100：`NCCL_NVLS_ENABLE=0`。
- **Together 的 HCA：** 11-08 的 H100 节点除了 8 个 IB 口，还有两个 200G RoCE 前端口（`mlx5_4`、`mlx5_5`）。`MN_NET=together` 时 `env.sh` 读 `/sys/class/infiniband/*/ports/1/link_layer`，只把 InfiniBand 的口交给 NCCL（`NCCL_IB_HCA==mlx5_0,...`，`=` 开头是精确匹配）。
- **日志级别：** 训练时 NCCL 用 `INFO`，只开 `INIT`、`NET` 两个子系统，每个进程写进那一格的 `nccl/<host>.<pid>.log`，不进 `node*.log`（WARN 也在这些文件里）；`MN_NCCL_INFO=0` 时回到 `NCCL_DEBUG=WARN`，打在 `node*.log` 里。`comm` 任务用 `INFO`，打印 `INIT`、`NET`、`ENV`、`TUNING` 几个子系统，`TUNING` 会把 NCCL 自己的算法和协议延迟、带宽模型打出来，可以和 α–β 拟合对照。

### 7.2 验证（`comm` 任务，`MN_COMM=1` 时自动做）

快速路径不跑这一步：每一格的 `net` 列已经说明 NCCL 走的是 IB 还是 socket、有没有 GDR。11-08 跑过一次的结论：8 个 IB 口全部用上，有 GPUDirect RDMA；每个口 RDMA 原始带宽约 365 Gb/s；16 卡 1 GiB all_reduce busbw 439 GB/s；跨节点 alltoall 只有 91 GB/s，节点内 330 GB/s；跨节点 sendrecv 43 GB/s。原始结果在 `../results/20261108_h100_together/raw/h100/comm/`（不进 git）。

要重跑时 `bash $K8S comm` 打印每个节点的结论和调优对比表。按这个顺序看：

1. **RDMA 原始带宽：** 每个 IB 口单独用 `ib_write_bw` 测（主机内存，node 1 写到 node 0，不经过 NCCL）。每个口应该接近链路速率；Together 健康检查的阈值是每个设备 320 Gb/s。
2. **`InfiniBand: yes`：** NCCL 用了 IB。如果是 NO，按 `together/README.md` 第 4 节排查。
3. **`GPUDirect RDMA: yes`：** 跨节点的数据直接在 GPU 和网卡之间走，不经过主机内存。如果没看到，查 `setup_node_*.log` 里的 `nvidia_peermem` 那一行。
4. **busbw（256 MiB 以上）：** 16 卡 all_reduce 的参考值：Together 健康检查的多节点阈值 330 GB/s，Nebius 文档里 nccl-tests 的例子平均 317 GB/s。
5. **α–β 拟合：** 每个 op、每个组一行，`alpha us` 是延迟，`busbw` 是大消息的极限。这些是经过 torch.distributed 测的，α 含每次调用的开销，比 nccl-tests 测到的大。

### 7.3 调优扫描（`comm_sweep.tsv`，`MN_COMM=1` 时自动做，`MN_COMM_SWEEP=0` 单独关掉）

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

1. `bash $K8S status` 显示 `ALL_DONE`（或者你决定停在这里），然后 `bash $K8S fetch h100 $R/raw`，把整份运行文件拉回本机（`raw/` 被 `results/.gitignore` 排除，不进 git）。只要结果、不要 trace 时用 `bash $K8S collect $R`。
2. 要带回 logbook 的，放在 `phase14_multinode_5D_experiments/results/<日期>_<hw>/` 下：
   - 每个阶段的 `summary.tsv`；
   - measure 的 `trace_report*.tsv`；
   - `monitor/monitor_summary.tsv`；
   - `comm*/verdict_node*.txt`、`comm*/comm_bench.json` 和 `comm_compare.tsv`；
   - `pipeline/REPORT.txt`；
   - 每格的 `env_node*.txt` 和 `tree.txt`。

   再写一份说明 `README.md`：卡型、网络、titan 提交号、每格的数、慢 rank 和降频的卡。trace、显存快照和完整日志只留在 `raw/`；合成后的 trace（`*.json.gz`）也不进 git。
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
| 节点被抢占 | `bash $K8S status` 里每个节点的心跳 | 自动续跑；节点一直不回来就在控制台看容量 |
| report 的 `unattributed_pct` 不是 0 | `trace_report_pg.tsv` 里 `?` 那一行 | 用 `trace` 拉一格看 NCCL kernel 的参数，修 `trace_report.py` |
| 余额快用完 | Together 控制台 | 先 `collect`；余额用完时集群会被立刻暂停 |
| setup 一开始就退出，日志最后是 `CUDA_DRV=` | `setup_*.log` | 610 驱动的 `nvidia-smi` 写的是 `CUDA UMD Version`，`setup.sh` 已经两种都认；换了驱动又出现时看 `nvidia-smi` 的表头 |
| 一个节点 OOM 退出，其他节点不动 | 每个节点的 `node*.log` | 其他节点卡在 NCCL 里，要等初始化超时（300 秒）；流水线的看门狗和 `k8s.sh cell` 会在 60 秒后杀掉它 |
| `tail` 看不到新日志 | 日志在哪个 pod 写 | 用 `bash $K8S tail 0` / `tail 1` 或 dashboard，它们在写日志的那个 pod 里读 |
| 某格比同类慢很多，或者 rank 之间 `compute_ms` 差很多 | `trace_report_rank.tsv`、`gpu_node*.csv` 的 `temp_c` 和 `reasons` | 先排除降频的卡（第 4.3 节），再看并行布局 |
| trace 读不了（`gzip -t` 报错） | 写 trace 时有没有被杀 | 没法恢复，只能重跑这一格；以后等 `Finished dumping profiler traces` 出现在所有 rank 再停 |
| 非最后一个 PP stage 的 rank 的 loss 是负数（如 -2、-4） | — | 正常：那些 rank 不算 loss，titan 打印的是占位值 |
| Windows 新终端里 `kubectl: command not found` | — | 直接用 `bash $K8S ...`，它会找到 winget 装的 `kubectl`；要直接用 `kubectl` 就把它的目录加进 `PATH` |
