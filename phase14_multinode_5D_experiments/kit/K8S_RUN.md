# k8s 上两个 pod 的无人值守流程（Nebius、Together 共用，2026-11-07）

集群和 pod 怎么准备，见 `nebius/README.md` 或 `together/README.md`。pod 起来以后的事，两家一样，都在这里。**这套流程只在本机模拟过两个节点的协同（正常完成、probe OOM 减半、任意一个节点中途被杀再重启），还没在真集群上跑过。**

## 1. pod 起来后自动做什么

- **bootstrap.sh**（容器入口）：
  1. 装容器的系统包（每次启动都装，约 1 到 2 分钟）；
  2. 共享存储上的 setup：venv、torch nightly、titan 分支 `phase14_multinode_5d`。只让一个 pod 做，另一个等；
  3. 启动 `pipeline.py`，崩了就自动重启，直到全部完成或者有 `STOP` 文件。
- **pipeline.py** 按固定顺序跑：
  - `comm`：两节点通信检查，判断 IB 和 GPUDirect RDMA，测 busbw；
  - `assets.k3`、`assets.m3`：准备 tokenizer；K3 的转换失败时，自动退到 Qwen3 的 tokenizer；
  - 矩阵的每一行依次跑 smoke、probe、warm、measure（`MN_ORDER=row`，默认）。设 `MN_ROWS=k3_5d`，就只跑 5D 全开这一行。
- **断点续跑：** 状态放在共享存储的 `<BASE>/runs/<hw>/pipeline/` 下。pod 被抢占、重建以后，从第一个没完成的任务接着跑；被打断的那次尝试不计入失败。
- **自动调长度：** probe 遇到 OOM 时把长度减半重试（最低 4096），warm 和 measure 用 probe 最后定下的长度。
- **失败处理：** 一格连续真失败 2 次（`MN_MAX_FAILS`），就放弃这一格，同时跳过它后面的阶段，继续下一格。
- **收尾：** 写出 `REPORT.txt`、每个阶段的 `summary.tsv` 和 `ALL_DONE`。pod 不退出，方便收集结果。

## 2. 手动操作（本机，在 logbook 根目录下）

```sh
K8S=phase14_multinode_5D_experiments/kit/k8s.sh
bash $K8S status       # pod、setup、两个节点的心跳、最近的决策；跑完后显示 REPORT.txt
bash $K8S logs         # 两个节点的流水线日志
bash $K8S collect      # 结果拉回本机（不含 trace 和 cache），每个阶段写 summary.tsv；隔几个小时收一次
```

- **改矩阵**（长度、加行、额外变量）：在本地改 `kit/matrix_*.tsv`，然后 `bash $K8S sync`，再 `bash $K8S retask`。已经完成的任务不会重跑；要重跑某一格，先删掉 `pipeline/done/<任务名>`（放弃过的删 `giveup/` 和 `fails/` 里的同名文件），再 `retask`。
- **改范围**（清单里的 `MN_ROWS`、`MN_MODELS`、`MN_ORDER`、`HW`）：改完清单后 `kubectl delete job p14`，再 `kubectl apply -f <清单>`，最后 `bash $K8S retask`。状态都在共享存储上，删 Job 不会丢。
- **改脚本**（`launch.sh`、`run_comm.sh` 等）：`bash $K8S sync` 即可，从下一次尝试开始生效。
- **停和启：**
  - `bash $K8S stop`：当前这次尝试跑完再停（两个节点都退出，任务的进度保留）；
  - `bash $K8S kill`：立刻停，当前这次尝试作废，以后重跑；
  - `bash $K8S start`：重新开始。
- **手动跑一格：** 先 `stop` 或 `kill`，再用 `bash $K8S both '<命令>'` 让两个 pod 同时执行，例如：

```sh
bash $K8S both "MN_MODEL=k3 MN_FLAVOR=k3_16x141g MN_ASSETS=/mnt/shared/p14/assets/kimi_k3_tokenizer \
  MN_CELL=pp=2,dp=2,cp=2,tp=2,ep=8 MN_SEQ=16384 MN_STEPS=10 MN_PORT=29700 \
  bash /mnt/shared/p14/logbook/phase14_multinode_5D_experiments/kit/launch.sh /mnt/shared/p14/runs/manual/try1"
```

## 3. 文件在哪（都在共享存储，`BASE=/mnt/shared/p14`）

- `runs/<hw>/<stage>/<行名>/`：每格的结果。里面有 `node0.log`、`node1.log`、`env_node*.txt`、`tree.txt`，trace 在 `tt/profiling/traces/` 下。被打断或失败的尝试移到 `<stage>/_attempts/`。
- `runs/<hw>/comm/`：`verdict_node0.txt`（IB、GDR、busbw）、`comm_bench.json`、NCCL 的 INFO 日志。
- `runs/<hw>/pipeline/`：流水线的状态：
  - `tasks.json`：冻结的任务列表；
  - `done/`、`giveup/`：每个任务的结局；
  - `go/`、`rc/`：每次尝试的通知和两个节点各自的退出码；
  - `hb/`：心跳；
  - `len/`、`try/`：probe 定下的长度和正在试的长度；
  - `log_node*.txt`：流水线日志；`run_node*.log`：每次尝试的启动输出；
  - `REPORT.txt`、`ALL_DONE`。
- `setup_*.log`、`setup_node_*.log`、`pipeline_<hostname>.log`：setup 和 bootstrap 的日志。

## 4. 被抢占时会发生什么

- 一个节点没了，另一个节点上的运行在 2 到 3 分钟内被看门狗杀掉，这次尝试记为"中断"。
- Job 会补 pod，但新 pod 要等有节点才能调度：Together 在有容量时会自动补节点；Nebius 的抢占式 VM 只会停机，节点能不能回来，看第 `nebius/README.md` 第 5 节。
- 新 pod 先装容器包，共享 setup 已经完成就直接跳过，然后接着跑流水线。
- 损失只是被打断的那一格，几分钟到几十分钟。

## 5. 第一次先看什么

1. `bash $K8S status` 显示 `setup: done`。
2. comm 的结论（`status` 也会显示）：要有 `InfiniBand: yes` 和 `GPUDirect RDMA: yes`。16 卡 all_reduce 的大消息 busbw 参考：Nebius 文档里 nccl-tests 的例子平均 317 GB/s（文档说高于 300 正常），Together 健康检查的多节点阈值是 330 GB/s。
3. 第一格的 smoke 跑完（`k3.smoke.k3_5d: done`）。
4. 之后定期 `collect`。

`InfiniBand: NO` 时，按各自 README 里的排查步骤处理。
