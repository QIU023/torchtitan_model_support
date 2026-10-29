# Phase 14 kit：2 节点 × 8 卡的 5D 实验（Kimi K3 / MiniMax M3 缩小版）

2026-10-28，CPU 会话按 `../multinode.md` 准备（用户："把里面设置的对应titan脚本和新开的titan实验分支全部准备好，里面还缺一个配置，A100 80G；准备两个模型Kimi K3 downscale和Minimax M3 downscale"）。

**本 kit 一步都还没跑过。** 代码只在 Windows 上做了 `py_compile` 和 black；titan 的模块在本机 import 不了（`models/common` 要 CuTeDSL），第一次上机先跑 smoke。

## 1. titan 分支

`phase14_multinode_5d` = `ac36e017b`（fork QIU023/torchtitan），实验分支，不开 PR。
- base：`minimax_m3_5d_review1` = `70b6efa63`。它是 upstream main `e111f10d4` 加上 M3 的 FSDP2 基线和 TP/EP/PP（#5635 + #5619 待同步的 head），所以 K3 用 main 上的代码，M3 用 5D 分支的代码。
- 四个提交：
  - `19a94ae72` kda：`KDAKernel.Config` 加 `gate_impl`（`"fused"` 默认，`"reference"` 给 A100）。对应 multinode.md §9.1 的本地改动，CP 的 KDA 也走同一个 `prepare_inputs`。
  - `4f7595d8f` 两个模型的缩小版 flavor（见第 2 节）。
  - `6f7ca8643` `torchtitan_recipes/multinode_5d.py`：一个 `cell()`，全部设置读 `MN_*` 环境变量（变量表在模块 docstring 里）；`seed()` 写单机 seed checkpoint（性能实验用不到，数值对照才要）。
  - `ac36e017b` 修默认 micro-batch 数：原来先取整、再取成 pp 的倍数，24K 在 dp 2 · pp 2 时会得到 12 个（589,824 token），现在一次取整得到 10 个（491,520 token）。
- 和 multinode.md 依据的 `432b95dcf` 不同的一处：当前 base 的 K3 CP 按 `MLAAttention` 映射到 `AllGatherKVMLAFlexAttention` / `UlyssesMLAAttention`，`InnerKDA` 映射到 `ContextParallelInnerKDA`（写法照 `tests/integration_tests/h100.py`），不是文档里的 `KVAllGatherCPFlexInnerAttention`。all-gather KV 默认配 `HeadTailCPLoadBalancer`，Ulysses 不配。
- 统一设置：AdamW、fp32 主参数（`training.dtype=float32`，和文档 16 B/参的口径一致；K3 的测试 recipe 原本是 bf16）、默认 SelectiveAC、CUDA graph 关、一条序列一个 micro-batch、约 512K token / step（micro-batch 数取最接近的值并取成 pp 的倍数，比如 A100 的 24K 是 10 个 micro-batch、491,520 token）、K3 注意力用 flex。

## 2. 两个缩小版

| flavor | 用在 | 结构 | 总参数 / 激活 | AdamW 状态 / 卡 |
|---|---|---|---|---|
| `k3_16x80g` | H100、A100 | D 2304，32 头，MLA 秩 512/192，专家 960，latent 1152，dense 10880 | 14.47B / 3.68B | 14.5 GB |
| `k3_16x80g_small` | 80 GB 保守档 | D 2048，24 头，448/128，896，1024，9600 | 11.84B / 2.87B | 11.8 GB |
| `k3_16x141g` | H200 | D 3584，48 头，768/256，1536，1792，16896 | 35.06B / 8.58B | 35.1 GB |
| `k3_16x141g_small` | 141 GB 保守档 | D 3328，48 头，704/256，1408，1664，15744 | 30.16B / 7.57B | 30.2 GB |
| `m3_16x80g` | H100、A100 | D 2304，16 层，24 头 | 14.67B / 1.83B | 14.7 GB |
| `m3_16x80g_small` | 80 GB 保守档 | D 1920，16 层，20 头 | 10.32B / 1.41B | 10.3 GB |
| `m3_16x141g` | H200 | D 3072，20 层，32 头 | 33.06B / 3.22B | 33.1 GB |
| `m3_16x141g_small` | 141 GB 保守档 | D 2688，20 层，28 头 | 25.46B / 2.61B | 25.5 GB |

- K3 的四档照 multinode.md §2.3（参数量是那里 refconfig 的数）：29 层（MLA 在 3、7、…、27 和 28），AttnRes 块 4，128 专家选 16 加 2 个共享，vocab 163840，视觉塔用 debugmodel 的 2 层 4 头。
- M3 的四档是本 kit 新定的，按 M3 的比例只改宽度和层数：
  - 头数 D/96、4 个 KV 组、头维 128、partial RoPE 64；
  - dense FFN 2D、专家隐藏维 D/2、128 专家选 4 加 1 个共享；
  - 前 3 层是 dense FFN + 全注意力，其余是 MSA + MoE；MSA 块 128、选 16 块；vocab 200064；
  - 视觉塔用 M3 debugmodel 的（2 层、8 头，TP 4 时每卡 2 头）。
  - 宽度取成和 K3 同档的总参数相当（14.7B 对 14.5B，33.1B 对 35.1B），方便两个模型在同一档对比。参数量是公式估算，同一公式在全尺寸 M3 上给 426.2B 总参数、23.5B 激活（不含 embedding 和视觉塔），和发布的 428B / 23B 对得上；上机后以 titan 日志打印的参数量为准。
  - M3 的激活比只有 3.1%（4/128），K3 是 12.5%（16/128），所以同样的总参数下 M3 每卡算量小得多，MFU 不能直接和 K3 比。

## 3. 组合矩阵

`matrix_k3.tsv`、`matrix_m3.tsv`，每行：名字、度数、H100 / H200 / A100 的序列长度、额外变量（`hw:` 前缀只在那种卡上生效，`nosmoke` 表示 debug 模型做不了这个布局）。

K3（multinode.md §3.6 推荐配置，加 §5.2 的放置对照）：

| 行 | pp · dp · cp · tp · ep | H100 | H200 | A100 | 说明 |
|---|---|---|---|---|---|
| `k3_5d` | 2 · 2 · 2 · 2 · 8 | 32K | 16K | 24K | |
| `k3_A` | 2 · 2 · 1 · 4 · 8 | 32K | 16K | 24K | |
| `k3_B` | 1 · 2 · 4 · 2 · 8 | 64K | 32K | 64K | H200 整步不 reshard |
| `k3_C` | 2 · 2 · 4 · 1 · 8 | 32K | 16K | 24K | |
| `k3_D` | 2 · 1 · 4 · 2 · 8 | 64K | 64K | 48K | |
| `k3_D_ulysses` | 同上 | 64K | 64K | 48K | MLA 用 Ulysses |
| `k3_B_ep16` | 1 · 2 · 4 · 2 · 16 | 64K | 32K | 64K | EP 跨节点（坏放置） |
| `k3_B_keep` / `k3_B_ep16_keep` | ep 8 / 16 | 32K | 32K | 32K | §5.2 的「整步不 reshard」对照 |

- A100 的长度按 multinode.md §9.3 降：reference 门控在 5D、(A)、(C) 的 32K 放不下，粗估上限约 31K，取 24K；(D) 在 64K 也多出约 7.7 GB，取 48K；(B) 没有 PP、只有 1 个在途 micro-batch，64K 还放得下。都是纸面估计，probe 后改矩阵。另一个办法是 `MN_FLAVOR_OVERRIDE=k3_16x80g_small` 保持 H100 的长度。
- §5.2 里「TP 跨节点」那组 what-if 要改 rank 到节点的映射，titan 默认 tp 在最内层，本 kit 没做。§5.3 的 DeepEP 对 NCCL（E4）要单独的 DeepEP micro-benchmark，A100 做不了（§9.5），也不在本 kit 里。

M3（没有 CP，所以 K3 里 cp > 1 的行没有对应；长度是 probe 的起点）：

| 行 | dpr · pp · dp · tp · ep | 长度 | 说明 |
|---|---|---|---|
| `m3_A` | 1 · 2 · 2 · 4 · 8 | 16K | 和 K3 的 (A) 同度数；PP 跨节点，其余节点内 |
| `m3_B` | 1 · 1 · 4 · 4 · 8 | 16K | 没有 PP，dp_shard 跨节点，专家 FSDP 2 路跨节点 |
| `m3_B_ep16` | 1 · 1 · 4 · 4 · 16 | 16K | EP 跨节点（坏放置） |
| `m3_C` | 1 · 2 · 8 · 1 · 8 | 8K | 没有 TP，每卡整条序列 |
| `m3_D` | 1 · 4 · 1 · 4 · 4 | 16K | 没有 FSDP；stage 4 卡，PP 两跳在节点内、一跳跨节点 |
| `m3_E` | 2 · 1 · 8 · 1 · 8 | 8K | HSDP：只有梯度 all-reduce 跨节点 |

M3 的 TP 最多 4（index 头跟 KV 组切，KV 组是 4），所以 K3 (D) 那种 dp 1、stage 8 卡的布局在 M3 上要 tp 8，做不了，换成 pp 4。

## 4. 步骤

逐条可复制的命令在 `RUNBOOK.md`（2026-10-29），这里是概要。两台机器都要执行同一条命令（`NODE_RANK` 不同），或者用 `srun` 每节点一个任务。下面的 `kit` 指本目录。`HW` 除了 `h100` / `h200` / `a100`，还有 `generic`（比如 vast 上的 8×5090），只做通信和 smoke。

1. 每个节点：`HW=h100 bash kit/setup.sh`（A100 用 `HW=a100`）。看到 `SETUP_OK`，并核对 `nvidia-smi` 的卡型和 compute capability（A100 应是 `A100-SXM4-80GB`、`8.0`，见 multinode.md §9.6）。
2. 在 `kit/env.sh` 里填 `NCCL_IB_HCA`、`NCCL_SOCKET_IFNAME`。
3. tokenizer：
   - M3：`python scripts/download_hf_assets.py --repo_id MiniMaxAI/MiniMax-M3 --assets tokenizer --local_dir $B/assets`。
   - K3：`python kit/make_kimi_tokenizer.py $B/assets/kimi_k3_tokenizer`。失败的话下载 `Qwen/Qwen3-8B` 的 tokenizer，`ASSETS_K3` 指过去，并加 `MN_SPECIAL=qwen`（词表 151669 小于 163840，自带 `<|image_pad|>` 那一套特殊 token）。
4. 通信基线（E2）：在 node 0 上运行 `NODE0=... NODE1=... bash kit/nccl_tests.sh`。没有 MPI 的话，两台机器都用 torchrun 跑 `kit/comm_bench.py`（16 卡一次，再 `--nnodes 1` 单机 8 卡一次）。
5. smoke：`export HW=h100 MASTER_ADDR=<node0> NODE_RANK=<0|1>`，然后 `bash kit/run_matrix.sh k3 smoke` 和 `bash kit/run_matrix.sh m3 smoke`。每行 debug 模型跑 5 步，确认每种布局都能起来。
6. probe：`run_matrix.sh <k3|m3> probe` 用真模型在矩阵长度上跑 3 步，看 `summary_node0.tsv` 的 `peak_mem_gib` 和 `oom`。超了就在矩阵里缩短那一行，按 multinode.md §7.1 先拿 8K / 16K 校准。
7. warm 再 measure：`run_matrix.sh <k3|m3> warm`，然后 `run_matrix.sh <k3|m3> measure`。measure 和 warm 共用一份编译 cache（每个节点各一份，在 `$OUT/cache/$HW/<model>/node<k>`）；measure 跑 30 步，第 25 步出 profiler trace，在 `<cell>/tt/` 下。
8. `python kit/summarize.py $OUT/$HW/measure`：每格的峰值显存（哪个 rank）、第 3 步之后的 tps / tflops / mfu 中位数。结果写进本 phase 的 `results/<日期>_<hw>.md`。

输出目录是 `$OUT/<hw>/<stage>/<行名>/`，里面有每个节点的 `node<k>.log`、`rc_node<k>`、`env_node<k>.txt`、`tree.txt`。所有 rank 都打印步日志（`LOG_RANK` 设成了全部 rank），这样 PP 每个 stage 的显存都看得到。

## 5. 还没验证、上机先看的

- **K3 的 tokenizer：** titan 只认 `tokenizer.json` 或 vocab + merges，Kimi 的仓库很可能只有 tiktoken。`make_kimi_tokenizer.py` 是按 transformers 的 `TikTokenConverter` 写的，没跑过。
- **K3 的纯文本数据：** recipe 用的是 M3 kit 跑过的 packed c4（`ConcatThenSplitPackingConfig` + `c4_document_capacity`）。K3 的测试一直用多模态 loader，纯文本 batch 走 dummy 视觉前向这条路在 K3 上没在 kit 里跑过。
- **A100 的 reference 门控：** multinode.md §9.1 在 4070 Ti 上用本地改动跑过 debugmodel，本分支的 `gate_impl` 开关还没跑过。
- **K3 的 CP + PP、CP + EP 8：** CI 里 K3 的 CP 只有 dp 1 · cp 2 的 debug 格子，PP 只有 b200 的 tp 2 · ep 2 · pp 2 · vpp 4 格子。5D、(C)、(D) 这种 CP + PP 的组合在当前 base 上第一次跑，所以先 smoke。
- **M3：** HSDP + EP（`m3_E`）和 16 层的 pp 4 切分（`m3_D`）都没跑过。M3 的 4D 矩阵在 8 x 5060 上跑的是 debugmodel。
- **序列长度：** 全是纸面估计（K3 来自 multinode.md 的页面模型和 §9 的比例粗估，M3 是起点猜测）。probe 之前不要拿它们排表。
- **数值对照：** 这套 kit 测的是性能。要做逐位或第 1 步的数值对照，按 CLAUDE.md 的数值规则：同一条数据流配对，用 `seed()` 写的 checkpoint（`MN_SEED_CKPT`）加 `MN_DETERMINISTIC=1`，共用一份预热过的 cache。

## 6. 文件

| 文件 | 内容 |
|---|---|
| `env.sh` | 路径、分支提交号、torch nightly、每种卡的 flavor 和门控、NCCL 变量 |
| `setup.sh` | 克隆分支、建 venv、装依赖、检查卡和 import |
| `launch.sh` | 一个格子：torchrun 2 × 8，`--module torchtitan_recipes.multinode_5d --config cell` |
| `run_matrix.sh` | 一个模型、一个阶段，跑矩阵的每一行 |
| `matrix_k3.tsv`、`matrix_m3.tsv` | 两个模型的组合和每种卡的长度 |
| `summarize.py` | 解析日志，出每格一行的 TSV |
| `collect.sh` | 没有共享文件系统时，在 node 0 上拉回 node 1 的日志并汇总 |
| `RUNBOOK.md` | 租卡之后从开机到结果回 logbook 的逐步命令 |
| `nccl_tests.sh`、`comm_bench.py` | busbw：nccl-tests（主）和 torch.distributed（备） |
| `make_kimi_tokenizer.py` | 把 Kimi K3 的 tiktoken 转成 `tokenizer.json` |
