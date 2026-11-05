# Nebius：同一个 GPU cluster 里的两个 pod 跑 phase 14 kit（2026-11-05）

依据：Nebius 文档 [Interconnecting GPUs in Managed Kubernetes clusters using InfiniBand](https://docs.nebius.com/kubernetes/gpu/clusters) 和 [NCCL test](https://docs.nebius.com/kubernetes/gpu/nccl-test)（官方用 MPIJob 跑 nccl-tests），以及 SkyPilot 的 [Nebius InfiniBand 示例](https://docs.skypilot.co/en/latest/examples/performance/nebius_infiniband.html)。**这套 pod 流程还没在 Nebius 上跑过**，第一次按第 3 步的检查确认。

## 0. 组网是怎么通的

- **同一张 IB 网：** Nebius 的 GPU cluster 绑定一个 InfiniBand fabric。mk8s 的 node group 创建时用 `--template-gpu-cluster-id` 挂到这个 GPU cluster，组里的节点就在同一张 IB 网上。
- **pod 之间直接互通：** 两个 pod 走 k8s 的 pod 网络（eth0），本来就能互相连接。不需要 vast 那种 overlay，也不需要 SSH，进 pod 用 `kubectl exec`。
  - torchrun 的 rendezvous 和 NCCL 的 bootstrap 走 eth0。`MASTER_ADDR` 用 headless Service 给出的 `p14-0.p14`。
  - NCCL 的数据走 IB（`NCCL_IB_HCA=mlx5`）：容器是 privileged，能看到宿主机的 IB 设备；容器里要有 libibverbs 和 mlx5 provider，由 `setup.sh` 安装。缺了它们，NCCL 会悄悄退回 TCP，第 3 步的检查会报出来。
- **设置照 Nebius 官方 NCCL 测试：**
  - privileged，`/dev/shm` 挂内存 emptyDir；
  - `NCCL_SOCKET_IFNAME=eth0`、`NCCL_IB_HCA=mlx5`、`UCX_NET_DEVICES=eth0`、`SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1`、`NCCL_COLLNET_ENABLE=0`；
  - pod 里设了 `MN_NET=nebius`，`kit/env.sh` 会据此把这些变量设好；
  - memlock 由 `launch.sh` 和 `run_comm.sh` 在启动前 `ulimit -l unlimited`。

## 1. 建集群

- **支持 IB 的平台**（Nebius 文档的表）：
  - gpu-h100-sxm：eu-north1，preset 8gpu-128vcpu-1600gb；
  - gpu-h200-sxm：eu-north1、eu-north2、eu-west1、us-central1，同一 preset；
  - gpu-b200-sxm：us-central1、me-west1，preset 8gpu-160vcpu-1792gb；
  - gpu-b300-sxm；
  - 没有 A100。kit 的 `HW` 对应：H100 用 `h100`，H200 用 `h200`，B200 用 `b200`（和 h200 同一组模型和长度）。B300 没有对应的档。
- **命令**（来自 Nebius 文档，fabric 按区域和卡型选，文档 H100 的例子是 `fabric-3`）：

```sh
export INFINIBAND_FABRIC=<fabric>
export GPU_CLUSTER_ID=$(nebius compute gpu-cluster create --name p14-gpus \
  --infiniband-fabric $INFINIBAND_FABRIC --format jsonpath='{.metadata.id}')
nebius mk8s node-group create --parent-id <mk8s 集群 id> --name p14-ib --fixed-node-count 2 \
  --template-resources-platform gpu-h100-sxm --template-resources-preset 8gpu-128vcpu-1600gb \
  --template-gpu-cluster-id $GPU_CLUSTER_ID --template-gpu-settings-drivers-preset cuda13.0
```

- **其他要求：**
  - mk8s 集群本身按 Nebius 文档建；GPU cluster、node group 和 mk8s 集群必须在同一个 project；
  - 驱动由 node 镜像带（控制台里保持 "Install NVIDIA GPU drivers and other components" 勾选）；
  - node group 的启动盘留至少 300 GiB：每个 pod 的 venv、编译 cache、日志和 trace 会占几十到上百 GB。参数名以 `nebius mk8s node-group create --help` 为准；
  - pod 要能访问外网：拉镜像、git clone、pip、HF 都要用。
- 本机拿到 kubeconfig 后，`kubectl get nodes` 应该能看到两个 GPU 节点。

## 2. 拉起两个 pod（本机，在 logbook 根目录下）

```sh
# H200 / B200：先把 p14_pods.yaml 里 HW 的值改成 h200 / b200
kubectl apply -f phase14_multinode_5D_experiments/kit/nebius/p14_pods.yaml
# 可选，tokenizer 下载要 HF token 时：
kubectl create secret generic hf-token --from-literal=token=<hf_token>
K8S=phase14_multinode_5D_experiments/kit/nebius/k8s.sh
bash $K8S status          # 两个 pod Running；setup 装完会显示 /root/p14/SETUP_OK
```

- `p14_pods.yaml` 是一个 Indexed Job（2 个 pod，每个要 8 卡，所以会各占一个节点）加一个 headless Service。
- 每个 pod 启动后会 clone logbook 的 phase14 目录、跑 `kit/setup.sh`（apt、venv、torch nightly、titan 分支 `phase14_multinode_5d`），然后空闲等命令。
- setup 的日志在 pod 里的 `/root/p14/setup_<hostname>.log`。

## 3. 组网检查

```sh
bash $K8S shell 0                # 进 pod 0 看一眼，再 exit
  ibv_devinfo -l                 # 应列出 mlx5_x（SkyPilot 的 Nebius 示例里是 mlx5_0 到 mlx5_7）
  getent hosts p14-0.p14 p14-1.p14
bash $K8S comm                   # 两个 pod 同时跑 kit/run_comm.sh
bash $K8S logs                   # 看输出；跑完后 status 里会显示结论
```

通过的标准：
- `InfiniBand: yes`：NCCL 选了 IB；
- `GPUDirect RDMA: yes`：跨节点通道是 GDRDMA；
- 16 卡 all_reduce 的大消息 busbw：Nebius 文档里 nccl-tests 的例子平均是 317 GB/s，文档的原话是高于 300 GB/s 就说明连接正常。comm_bench 用的是 torch.distributed，不是 nccl-tests，量级应该相同。

`InfiniBand: NO` 时先查四件事：pod 是不是 privileged、`ibv_devinfo -l` 有没有设备、`NCCL_IB_HCA` 是不是 `mlx5`、setup 日志里 rdma-core 有没有装上。

## 4. 跑 5D 全开

`run` 会立刻返回，命令在两个 pod 里后台跑。上一个阶段跑完（用 `status` 或 `logs` 确认）再开下一个，否则两次运行会抢同一批 GPU。

```sh
bash $K8S run k3 smoke k3_5d     # debug 模型 5 步：2 x 8 的 pp2 · dp2 · cp2 · tp2 · ep8 能起来
bash $K8S logs
# probe 之前两个 pod 都要准备 tokenizer（Kimi 的转换失败时见 kit/README.md 的 Qwen3 退路）
bash $K8S both "cd /root/p14/wt_p14 && . /root/p14/venv/bin/activate && \
  python /root/logbook/phase14_multinode_5D_experiments/kit/make_kimi_tokenizer.py /root/p14/assets/kimi_k3_tokenizer"
bash $K8S run k3 probe k3_5d     # 真模型（H100 是 k3_16x80g）3 步，看各 rank 的峰值显存
bash $K8S run k3 warm k3_5d      # 10 步，填满编译 cache
bash $K8S run k3 measure k3_5d   # 30 步，第 25 步出 profiler trace
bash $K8S collect                # 结果拉回本机（不含 trace 和 cache），每个阶段写 summary.tsv
```

去掉行名就是跑整张矩阵（`kit/matrix_k3.tsv`、`kit/matrix_m3.tsv`）。

## 5. 手动改 5D setup

- **改矩阵（推荐）：** 在 `kit/matrix_k3.tsv` 里改一行或复制一行改名，然后 `bash $K8S sync` 把 kit 同步进两个 pod，再 `run` 那一行。

```text
# 列：行名  度数  H100 长度  H200/B200 长度  A100 长度  额外变量
k3_5d	pp=2,dp=2,cp=2,tp=2,ep=8	32768	16384	24576
k3_5d_i1f1b	pp=2,dp=2,cp=2,tp=2,ep=8	16384	16384	16384	MN_SCHEDULE=Interleaved1F1B
```

  - 度数的键：`dpr`（HSDP replicate）、`dp`（dp_shard）、`cp`、`tp`、`ep`、`pp`，没写的就是 1。
  - 额外变量（写进最后一列，空格分隔，`h200:KEY=VAL` 只在那种卡上生效）：
    - `MN_SCHEDULE=Interleaved1F1B`：调度换成 Interleaved1F1B；
    - `MN_CP=ulysses`：MLA 改用 Ulysses CP；
    - `MN_FSDP_KEEP=1`：FSDP 整步不 reshard，并推迟梯度规约；
    - `MN_MICROBATCHES=<n>`：每个 DP rank 每步的序列数，默认凑到约 512K token；
    - 任何 `NCCL_*` 变量，比如 `NCCL_ALGO=Ring`。
  - 全部 `MN_*` 变量的说明在 titan 分支 `torchtitan_recipes/multinode_5d.py` 的开头。
- **临时试一个格子：** 两个 pod 必须用同一条命令，端口要选没人用的。

```sh
bash $K8S both "MN_MODEL=k3 MN_FLAVOR=k3_16x80g MN_ASSETS=/root/p14/assets/kimi_k3_tokenizer \
  MN_CELL=pp=2,dp=2,cp=2,tp=2,ep=8 MN_SEQ=16384 MN_STEPS=10 MN_PORT=29700 \
  bash /root/logbook/phase14_multinode_5D_experiments/kit/launch.sh /root/p14/runs/manual/try1"
```

- **约束**（titan 会在启动时检查）：
  - `dpr · dp · cp · tp · pp = 16`；
  - `ep` 要整除 `dp · cp · tp`，并且 `ep ≥ tp`，128 个专家能被 `ep` 整除；
  - Ulysses 要求头数能被 `tp · cp` 整除（k3_16x80g 是 32 头，k3_16x141g 是 48 头）；
  - K3 的 PP 只支持 1F1B 和 Interleaved1F1B；micro-batch 数要 ≥ pp，并且是 pp 的倍数。

## 6. profiling：现在有的和还没有的

- **现在有：**
  - 每个 rank 每一步的 loss、grad norm、峰值显存（max reserved）、tps、tflops、mfu。`collect` 会用 `kit/summarize.py` 汇总成每格一行，峰值显存会标出是哪个 rank。
  - measure 阶段的 torch profiler trace：pod 里的 `/root/p14/runs/<hw>/measure/<行名>/tt/profiling/traces/`。`collect` 不带 trace，要的话用 `kubectl cp` 单独拷。
  - 通信：`run_comm.sh` 给出 16 卡（跨节点）和单节点 8 卡的 all_reduce、all_gather、reduce_scatter、alltoall 的 busbw，跨节点和同节点的 sendrecv，以及 IB 和 GDR 的结论。
- **还没有（下一步补）：**
  - α–β 拟合（要从小消息扫起，再做拟合）；
  - 从 trace 拆出暴露通信占一步的比例（logbook 里 `Raising_PRs/PR_K3_PARALLELISM/kit_h100_*` 下有以前的 `analyze_trace.py` 可以改）；
  - NCCL 算法和协议的扫描；
  - DeepEP 和 NCCL all-to-all 的对比；
  - 后训练的几种 setup：multinode.md 只规划了预训练（每步 512K token、全参 AdamW）。现在可以先用 `MN_SEQ` 和 `MN_MICROBATCHES` 改全局 batch 和长度来近似。

## 7. 收尾

先 `collect`，再 `kubectl delete -f phase14_multinode_5D_experiments/kit/nebius/p14_pods.yaml`。pod 删掉以后，里面的文件全部丢失。
