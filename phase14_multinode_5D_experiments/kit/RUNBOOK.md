# 租卡之后怎么跑这个 kit（2026-10-29）

照着从上往下复制粘贴。**node 0**、**node 1** 标的是在哪台机器上执行；"两台"表示两台都要执行，而且同一步要在一两分钟内一起开始（torchrun 会等另一台来 rendezvous）。

## 0. 先确定租什么

| 租法 | 能做什么 | `HW` |
|---|---|---|
| 两台 8 卡 H100 / H200 / A100 80GB，节点之间能直连（最好带 InfiniBand） | 全部：通信、smoke、probe、warm、measure | `h100` / `h200` / `a100` |
| vast cluster 80 的两台 8×RTX 5090，用 overlay 组网（`VAST_A100_SEARCH_2026-10-28.md`） | 只有通信和 smoke；缩小版要 80 GB，5090 放不下 | `generic` |

两台的卡型、驱动、镜像要一样。磁盘每台至少留 200 GB（venv、编译 cache、profiler trace）。

## 1. 开机（只有 vast 需要；在你自己的电脑上执行）

```sh
vastai search offers 'cluster_id=80 num_gpus=8' --raw | grep -E '"id"|machine_id|dph_total'
vastai create overlay 80 p14net
vastai create instance <offer_A> --image pytorch/pytorch:latest --disk 200 --ssh --env "-n p14net"
vastai create instance <offer_B> --image pytorch/pytorch:latest --disk 200 --ssh --env "-n p14net"
vastai show instances        # 记下两台的 SSH 地址和端口
```

## 2. 装环境（两台）

```sh
apt-get update && apt-get install -y git rsync tmux iproute2 iputils-ping   # 镜像里缺什么补什么
git clone --depth 1 --filter=blob:none --sparse https://github.com/QIU023/torchtitan_model_support.git ~/logbook
cd ~/logbook && git sparse-checkout set phase14_multinode_5D_experiments
export KIT=~/logbook/phase14_multinode_5D_experiments/kit
HW=a100 bash $KIT/setup.sh          # HW 按第 0 步；最后一行是 SETUP_OK 才算成功
```

`setup.sh` 会把 titan 实验分支 `phase14_multinode_5d` 克隆到 `~/p14/wt_p14`，并核对提交号。

## 3. 网络（两台）

```sh
ip -4 addr                  # 找两台互相能通的那块网卡和地址；vast overlay 上是 eth0
ibstat 2>/dev/null | grep -E "CA '|State|Rate"   # 有 IB 就记下 mlx5_x 的名字
```

后面每个终端都先设这几个变量（两台一样，只有 `NODE_RANK` 不同）：

```sh
export KIT=~/logbook/phase14_multinode_5D_experiments/kit
export HW=a100                       # 按第 0 步
export MN_IFNAME=eth0                # 上一步找到的网卡
export MN_IB_HCA=                    # 有 IB 就填，例如 mlx5_0,mlx5_1；没有就留空
export MASTER_ADDR=<node 0 在这块网卡上的地址>
export NODE_RANK=0                   # node 1 上写 1
```

连通性：node 1 上执行 `ping -c 3 $MASTER_ADDR`。

## 4. 通信基线 E2（两台一起）

```sh
. ~/p14/venv/bin/activate
torchrun --nnodes 2 --nproc_per_node 8 --node_rank $NODE_RANK --rdzv_backend c10d \
  --rdzv_endpoint $MASTER_ADDR:29501 --rdzv_id comm $KIT/comm_bench.py --out ~/p14/runs/comm_16.json
```

node 0 会打印每种集合通信的 busbw。跨节点的数（`world` 和 `cross_node` 两组）只要明显低于节点内，第 5 步以后的多节点运行就能通。有 MPI 和节点间 ssh 的话，再在 node 0 上跑 `NODE0=... NODE1=... bash $KIT/nccl_tests.sh`，表里的 E2 用 nccl-tests 的数。

## 5. smoke（两台一起，每个模型一次）

建议在 tmux 里跑，断线也不会停。

```sh
bash $KIT/run_matrix.sh k3 smoke
bash $KIT/run_matrix.sh m3 smoke
```

- 每行用 debug 模型跑 5 步。debug 模型做不了的行（ep 16、Ulysses）会自动跳过。
- 进度看 `~/p14/runs/$HW/smoke/progress_node0.txt`，每行有 `rc`。
- 失败的行，看 `~/p14/runs/$HW/smoke/<行名>/node0.log` 和 `node1.log` 里最早出现的那个 Traceback。

## 6. 汇总（node 0）

没有共享文件系统时，先让 node 0 能 ssh 到 node 1（把 node 0 的 `~/.ssh/id_ed25519.pub` 加进 node 1 的 `~/.ssh/authorized_keys`），然后：

```sh
NODE1=<node 1 的地址> bash $KIT/collect.sh
```

它会把 node 1 的日志拉过来，并给每个阶段写一份 `summary.tsv`：每行的退出码、是否 OOM、loss、峰值显存（以及在哪个 rank）、tps / tflops / mfu。

## 7. 正式运行（只在 80 GB 以上的卡上）

先准备 tokenizer（node 0 下载后复制到 node 1，或者两台各下一次）：

```sh
cd ~/p14/wt_p14 && . ~/p14/venv/bin/activate
python scripts/download_hf_assets.py --repo_id MiniMaxAI/MiniMax-M3 --assets tokenizer --local_dir ~/p14/assets
python $KIT/make_kimi_tokenizer.py ~/p14/assets/kimi_k3_tokenizer
# 转换失败时：下载 Qwen/Qwen3-8B 的 tokenizer，export ASSETS_K3=~/p14/assets/Qwen3-8B MN_SPECIAL=qwen
```

然后两台一起，每个模型依次跑三个阶段：

```sh
bash $KIT/run_matrix.sh k3 probe     # 3 步，看显存
# 用 collect.sh 看 summary.tsv：OOM 或峰值超过 90% 的行，改 matrix_k3.tsv 里这种卡那一列的长度，再 probe 那一行
bash $KIT/run_matrix.sh k3 warm      # 每行 10 步，填满编译 cache
bash $KIT/run_matrix.sh k3 measure   # 每行 30 步，第 25 步出 profiler trace
```

M3 同样，把 `k3` 换成 `m3`。只重跑某几行，就把行名加在后面，例如 `bash $KIT/run_matrix.sh k3 probe k3_5d k3_A`。改了矩阵要在两台上同步。

## 8. 把结果带回 logbook

- 要提交的：每个阶段的 `summary.tsv`、`progress_node*.txt`，以及每行的 `env_node*.txt`、`rc_node*`、`tree.txt`、两台的 `setup_*.log` 和 `venv_freeze_*.txt`、`comm_16.json`、nccl-tests 的输出。
- 不提交的：profiler trace 和完整日志（太大）。trace 留在机器上，或者另外打包。
- 放在 `phase14_multinode_5D_experiments/results/<日期>_<hw>/` 下，再写一份说明，写明卡型、网络（IB 还是以太网）、titan 提交号。

## 大概要多久

纸面估计，以实际为准：通信基线约 10 分钟；每个模型的 smoke 约 1 小时（大部分时间在编译）；probe 约 1 小时；warm 加 measure 约 2 到 3 小时。两台 8 卡一起跑，全流程大约 8 到 10 小时。
