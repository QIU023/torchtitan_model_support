# K3-16G：2 节点 × 8 卡（H100 / H200，InfiniBand）训练缩小版 Kimi K3 的纸面规划

> **全文是纸面估算，以实测为准。** 数字只来自两处：「5D 并行通信时间线」页面模型（`<SP>/5d/model3.js`，经 `lib.js` 原样载入）或统一参考配置 `<SP>/kb/refconfig.py` 算出来的；以及下文 §6 明确写出的假设。§0.3 的四个实验目标（E1 模型规模、E2 节点内与跨节点 busbw、E3 放置前后的暴露通信、E4 DeepEP 对 NCCL all-to-all 的加速比）最后都以实测为准，本文只给规划用的预期。
>
> 换成 A100 80GB 的差异（本地改动、显存、通信瓶颈、做不了的实验）见 §9，那一节是按硬件比例的粗估。
>
> `<SP>` = `/tmp/claude-0/-home-claude/8c27c119-789c-5d1c-aee7-eb93201861e6/scratchpad`。源码依据的写法：**没写仓库名的路径都是 torchtitan@432b95dcf**（`<SP>/repos/torchtitan_upstream`，含 `torchtitan_recipes/`）；其他仓库写全：PyTorch@d31a3f14、DeepEP@v1.2.1（9af0e0d）、DeepEP main@b5a33ee（只用 `git show` 读）、attn_gym 0.0.16、nccl-tests@afd59ab；页面模型行号指 `<SP>/5d/model3.js`。

---

## 0. 结论速览

### 0.1 缩小版 K3（结构族：29 层 = 21 KDA + 8 MLA，AttnRes 4 层一块共 8 块，128 专家选 16 + 2 共享，vocab 163840；其余维度按 K3 比例随宽度 D 缩放）

| | D | 头数 | MLA q / kv 秩 | 专家隐藏维 / latent | 总参数 | 激活参数 | 相对 2.78T / 104.2B |
|---|---|---|---|---|---|---|---|
| **H100 推荐** | 2304 | 32 | 512 / 192 | 960 / 1152 | **14.5B** | **3.68B** | **192× / 28×** |
| **H200 推荐** | 3584（K3 的一半） | 48 | 768 / 256 | 1536 / 1792 | **35.1B** | **8.58B** | **79× / 12×** |
| H100 保守 | 2048 | 24 | 448 / 128 | 896 / 1024 | 11.8B | 2.87B | 235× / 36× |
| H200 保守 | 3328 | 48 | 704 / 256 | 1408 / 1664 | 30.2B | 7.57B | 92× / 14× |

推荐档 = 同族里「5D 全开与 (A)–(D) 在 32K 都能放下（torchtitan 口径、碎片 20% + 6 GB）」的最宽者，最紧的配置在 32K 只剩约 4 GB；保守档宽度降一级，多留约 16 GB。torchtitan@432b95dcf 里 debugmodel（26.9M）和 Kimi-K3（2.78T）之间**没有**现成的中间规格（`torchtitan/models/kimi_k3/flavors.py:594-597` 只注册这两个），要自己用 `_kimi_k3_config`（`flavors.py:397-509`）加一个 flavor。

### 0.2 四种组合（EP 常开，各组合用同一个推荐模型；MFU = 页面口径，6 × 激活 + 注意力核，相对 BF16 稠密 989 TFLOPS）

**H100（D=2304，14.5B / 3.68B）**

| 组合 | pp · dp_shard · cp · tp · ep | 放置 | 上下文区间（下限 → 上限，碎片 20% / 10%） | 推荐 | MFU @推荐 | MFU @8K | MFU @32K |
|---|---|---|---|---|---|---|---|
| 5D 全开 | 2 · 2 · 2 · 2 · 8，1F1B，CP = all-gather KV | PP 跨节点；TP、CP、FSDP、EP 全在节点内 | 8K → 37K / 43K | 32K | 14.6% | 13.7% | 14.6% |
| (A) EP+FSDP+TP+PP | 2 · 2 · 1 · 4 · 8，1F1B | PP 跨节点；其余节点内 | 8K → 34K / 39K | 32K | 14.4% | 13.6% | 14.4% |
| (B) EP+FSDP+TP+CP | 1 · 2 · 4 · 2 · 8，FSDP 每层 reshard（默认） | dp_shard 跨节点（FSDP 8 路、专家 FSDP 2 路跨节点）；TP、CP、EP 节点内 | 64K（硬下限 16K）→ 92K / 103K | 64K | 17.8% | 7.9% | 14.2% |
| (C) EP+FSDP+PP+CP | 2 · 2 · 4 · 1 · 8，1F1B | PP 跨节点；其余节点内 | 8K → 38K / 44K | 32K | 14.5% | 13.5% | 14.5% |
| (D) EP+TP+PP+CP | 2 · 1 · 4 · 2 · 8，1F1B | PP 跨节点；其余节点内 | 32K（硬下限 16K）→ 73K / 84K | 64K | 16.0% | 13.1% | 15.0% |

**H200（D=3584，35.1B / 8.58B）**

| 组合 | pp · dp_shard · cp · tp · ep | 放置 | 上下文区间（下限 → 上限，碎片 20% / 10%） | 推荐 | MFU @推荐 | MFU @8K | MFU @32K |
|---|---|---|---|---|---|---|---|
| 5D 全开 | 2 · 2 · 2 · 2 · 8，1F1B，CP = all-gather KV | PP 跨节点；其余节点内 | 8K → 36K / 43K | 16K | 21.3% | 21.1% | 21.0% |
| (A) | 2 · 2 · 1 · 4 · 8，1F1B | 同上 | 8K → 33K / 39K | 16K | 21.1% | 20.9% | 20.7% |
| (B) | 1 · 2 · 4 · 2 · 8，FSDP 整步不 reshard | dp_shard 跨节点；TP、CP、EP 节点内 | 16K → 48K / 62K | 32K | 23.7% | 20.5% | 23.7% |
| (C) | 2 · 2 · 4 · 1 · 8，1F1B | PP 跨节点；其余节点内 | 8K → 35K / 43K | 16K | 21.0% | 20.6% | 20.8% |
| (D) | 2 · 1 · 4 · 2 · 8，1F1B，CP = Ulysses | PP 跨节点；其余节点内 | 16K → 72K / 86K | 64K | 21.9% | 20.3% | 21.9% |

下限 = 硬判据（每卡每 micro-batch ≥ 2048 token、CP 暴露 ≤ 5%、PP 的 M ≥ P）与「MFU ≥ 本配置最高值 90%」两者取大，括号里是只看硬判据的值；上限按 torchtitan 口径显存，页面口径的上限见 §3。

### 0.3 实验目标与纸面预期（以实测为准）

| 目标 | 纸面预期 | 依据 |
|---|---|---|
| E1 模型规模 | H100 方案 14.5B（激活 3.7B）；H200 方案 35.1B（激活 8.6B） | 本文 §2；实测以 torchtitan 日志打印的参数量为准 |
| E2 节点内 busbw | ≈ 350 GB/s（8 卡 all_reduce / all_gather；经验值，页面公式给 358–359） | NVLink 450 GB/s 单向（规格，近似值）× 0.80 |
| E2 跨节点 busbw | ≈ 42 GB/s 每卡（跨节点 sendrecv，单网卡）；16 卡 alltoall_perf 的 busbw ≈ 75 GB/s；16 卡 all_reduce_perf 的 busbw 页面模型给 ≈ 338 GB/s（8 张网卡并行的分层假设，拿不准） | 每卡一张 400 Gb/s 网卡 ≈ 50 GB/s（规格，近似值）× 0.85 / 0.80 |
| E3 暴露通信：坏放置 → 推荐放置 | EP 跨节点（EP = 16，NCCL all-to-all）≈ 22–28% → EP 收回节点内 ≈ 7–8%；TP 跨节点 ≈ 12–18% → ≈ 5–7% | §5.2 |
| E4 DeepEP 对 NCCL all-to-all | ≈ 4–9×（页面公式 4.2×；NVLink 段按目标 GPU 去重 6.6×；DeepEP 文档 43 GB/s 且网卡受限 8.6×）；落到整步只有 ≈ 1.2× | §5.3 |

### 0.4 最卡的约束

- **卡住缩小幅度的是「5D 全开在 32K」时 PP stage 0 的显存。** H100 推荐档在 32K：激活 30.4 GB（1F1B 下 stage 0 有 2 个在途 micro-batch，含 torchtitan 多存的 all-gather 完整输入 4.5 GB）+ AdamW fp32 分片状态 14.8 GB + PP 下常驻的 6P 8.8 GB（含 edp = 1 专家按 torchtitan 口径补的 4.4 GB）+ buffer 1.8 GB + 6 GB = 61.8 GB，乘碎片后 73.0 GB / 80 GB。宽一档（D = 2560）只能到 30K。H200 推荐档对应 111.6 GB → 132.7 GB / 141 GB，宽一档（D = 3840）只能到 29K。
- 同等级别的第二个约束是 (A)（cp = 1）：32K 只有 tp = 4 的选项放得下；dp_shard 4 · tp 2 在 H100 上限 19K、H200 上限 18K。
- 结构上的硬约束：要用 Ulysses 时头数得被 tp × cp 整除，本文的组合里最大是 8（`torchtitan/config/validation.py:202-212`）；E 要能被 16 整除（EP = 16 的 DeepEP 实验）；层数取 4n + 1 才能复刻 K3 的「每 4 层 1 个 MLA + 最后一层 MLA」；视觉塔头数要能被 tp 整除（`torchtitan/models/kimi_k3/model.py:451-455`），所以 tp ≤ 4。
- 优化器只能用 AdamW（16 B/参）：DistMuon 还不支持 TP（`torchtitan_recipes/tests/suites/b200.py:110-113`）。

---

## 1. 依据与口径

### 1.1 页面模型的一致性

| 对象 | 核对结果 |
|---|---|
| 本地 `<SP>/5d/5d-timeline.html` | 按 `build3.js` 的逻辑（`page3.html` + 去掉导出尾巴的 `model3.js` + `search3.js` + `u1/u2/u3.js`）重建，与文件逐字节相同 |
| 已发布页面（Artifact read，版本 1791740432-1f90） | = 平台的文档骨架 + 本地 `5d-timeline.html` 原文；`model3.js`（sha256 `2f220a4f…d284ba`）原文完整在其中 |
| 用法 | `lib.js` 用 Node `vm` 原样载入 `model3.js`，只往 `MODELS` 加缩小版条目；计时、bubble、MFU 全部走页面 `simulate()`。页面没有的项目只在 `formulas.md` §10 列出的几处补算 |

公式与行号见 `<SP>/k3_16g/formulas.md`。

### 1.2 torchtitan 的并行约束（都已在 432b95dcf 上核对）

| 约束 | 内容 | 依据 |
|---|---|---|
| 卡数 | `dp_replicate × dp_shard × cp × tp × pp = world_size` | `torchtitan/distributed/parallelism_context.py:219-223` |
| EP 借卡 | EP 从 `dp_shard × cp × tp` 里借：`ep` 必须整除它，`edp_shard = dp_shard·cp·tp / ep`；稀疏 mesh 是 `(pp, dp_replicate, edp_shard, ep)`，ep 在最内层，所以先借 TP、再 CP、再 dp_shard | `parallelism_context.py:225-230, 307-336`；`torchtitan/config/parallelism.py:174-184` |
| EP 下限 | MoE 要求 `ep ≥ tp`，`num_experts % ep == 0` | `torchtitan/config/validation.py:87-99` |
| TP | `tp` 整除 `n_heads` | `validation.py:69-78` |
| Ulysses | `n_heads`、`n_kv_heads` 都要被 `tp × cp` 整除，且不能配 load balancer | `validation.py:202-221` |
| CP 后端 | softmax 层只有 all-gather KV（`KVAllGatherCPFlexInnerAttention`，可配 HeadTail 负载均衡）与 Ulysses；KDA 层走 Attention Gym 原生 CP（`ContextParallelInnerKDA` → `context_parallel_kda`：每段的仿射摘要 `[HV, V+K, K]` fp32 all-gather 后前缀合成初始状态，卷积历史另做一次 all-gather） | `torchtitan/models/common/attention/cp_attention.py:79-313`；`cp_kda.py:56-219`；`kimi_k3/README.md:48`；attn_gym 0.0.16 `linear/context_parallel.py:98-102, 639-660`（版本钉在 `.ci/docker/requirements.txt:11`） |
| FSDP 轴 | FSDP 在 `dp_shard × cp` 上切参数 | `torchtitan/distributed/fsdp.py:101-106` |
| reshard 默认 | 有 PP 时不 reshard，无 PP 时每层 reshard | `fsdp.py:176-179` |
| PP 放置 | K3 的 AttnRes cache 只接受 loop 式放置（stage s 在 rank s mod p）：1F1B、Interleaved1F1B 可以，ZBV / DualPipeV 被拒 | `torchtitan/models/kimi_k3/pipeline_parallel/__init__.py:66-79, 105-108` |
| micro-batch | `num_pp_microbatches` 默认 1；`Schedule1F1B` 要求 ≥ stage 数 | `torchtitan/config/parallelism.py:130`；PyTorch@d31a3f14 `torch/distributed/pipelining/schedules.py:1468-1472` |
| CUDA graph | 默认开（`disable_cuda_graphs=False`）；`ep > 1` 时只放行 HybridEP non-blocking，其余 dispatcher 要 `training.disable_cuda_graphs=True` | `torchtitan/config/configs.py:73-84`；`validation.py:129-155` |
| 默认 dispatcher | `AllToAllTokenDispatcher`：每个 (token, expert) 各一行，按专家排序后 all-to-all，不去重，有一次主机同步取 split | `torchtitan/models/common/config_utils.py:400-403`；`token_dispatcher.py:76-90, 455-470` |
| DeepEP dispatcher | 要 DeepEP v2（≥ 2.0.0）的 `ElasticBuffer` | `torchtitan/distributed/deepep/deepep.py:8-57` |
| 默认 AC | `SelectiveAC`：保存所有模型声明的区域，只重算 routed experts 的 w13 | `torchtitan/training_engine.py:117-119`；`torchtitan/distributed/activation_checkpoint.py:210-246` |
| 精度 | 默认 `training.dtype = float32`、参数 bf16 混合精度、规约 fp32 | `configs.py:100, 107, 115` |
| 优化器 | DistMuon 还不支持 TP，带 TP 的格子用 AdamW | `torchtitan_recipes/tests/suites/b200.py:110-113` |

### 1.3 页面开关怎么对应 torchtitan

| 页面字段 | 取值 | 理由 |
|---|---|---|
| `prec` | `bf16` | torchtitan 的 K3 训练不走 FP8 |
| `opt` | `adam`（16 B/参分片 + 6 B/参未切分） | DistMuon 不支持 TP；torchtitan 默认 fp32 主参数 |
| `rc` | `sac` | 页面把它标为「torchtitan op 级 SAC」；torchtitan 现在的 `SelectiveAC` 注释写明它是为贴近旧的 op 级 SelectiveAC 默认值而定的（`activation_checkpoint.py:215-217`）。另按 §1.4 补 all-gather 完整输入 |
| `ep` | `deepep` 路径（节点内时与 NCCL A2A 同一公式 + notify + 主机同步）；跨节点时另算 NCCL 不去重（`formulas.md` §10） | torchtitan 默认 NCCL A2A |
| `ov` | `off` | torchtitan 主 trainer 没有 Megatron 的 EP 通信重叠（只有实验性 graph_trainer） |
| `cp` | `allgather` / `ulysses` | torchtitan 没有 Ring |
| `pp` | `i1f1b`，V = 1（1F1B）或 V = 2（Interleaved1F1B） | 都是 loop 式 |
| `fs` | 有 PP：keep；无 PP：reshard（默认）或 keep（`fsdp_reshard_after_forward="never"` + `fsdp_defer_gradient_reduction=True`） | `fsdp.py:176-179`；`parallelism.py:46, 61` |
| `s`、`mbs` | `s` = `training.num_tokens_per_microbatch_per_dp_rank` = `max_context_length`，`mbs = 1` | K3 recipe 就是一条序列一个 micro-batch（`torchtitan_recipes/tests/models/kimi_k3.py:119`）；注意力按整条序列算（不按文档打包） |
| `gbs` | 每 step 512K token（§6 假设）÷ s；PP 时补到 M ≥ P 且 M % P == 0 | — |

### 1.4 显存口径（演练章确认过的两条）

| 口径 | 页面 | torchtitan（结论用这个） |
|---|---|---|
| 同步 TP + 默认 SelectiveAC 的列切输入 | 只存 SP 分片（1/t） | wqkv、w13 各存一份 all-gather 后的完整输入：每层每个在途 micro-batch 多 `2 × tokTP × D × 2 B`（`torchtitan/models/common/linear.py:233-254` 的 `tp_gather` 区域被 SelectiveAC 保存） |
| PP + FSDP | dense 与 `edp > 1` 的专家按 6 B/参常驻（与「6P」一致） | 同上，另外 `edp = 1` 的专家也补 6P：EP > 1 时专家的 FSDP 轴大小为 1 也保留（`parallelism_context.py:238-241`） |
| 放得下的判据 | `total ≤ 0.95 × HBM` | `(状态 + 未切分 + 激活 + buffer) × (1 + f) + 6 GB ≤ HBM`，f = 10% / 20%；6 GB 是页面 `overheadGB`，当作 CUDA context、NCCL 缓冲与 workspace |

两种口径都算了，上限差别见 §3 表里的「页面口径上限」一列。

### 1.5 实际用到的硬件数（页面 HW 预设优先）

| 项 | H100 | H200 | 标注 |
|---|---|---|---|
| HBM | 80 GB | 141 GB | 近似值 |
| BF16 稠密峰值 | 989 TFLOPS | 989 TFLOPS | 近似值 |
| HBM 带宽 | 3.35 TB/s | 4.8 TB/s | 近似值（页面预设） |
| NVLink | 450 GB/s 单向 × 0.80 = 360 GB/s 有效（集合）、× 0.75 = 337.5 GB/s（A2A） | 同左 | 规格近似值 × 页面效率；题设的 350 GB/s（经验值）与之差 3% |
| 网卡 | 每卡一张 400 Gb/s ≈ 50 GB/s × 0.85 = 42.5 GB/s（集合 / P2P）、× 0.80 = 40 GB/s（A2A） | 同左 | 规格（近似值）× 页面效率 |
| GEMM 效率 | 0.55 × m/(m+256) | 同左 | 页面按 NeMo 标定 |
| 访存型算子系数 | `em = 4.0` | 同左 | 页面按 NeMo 标定；小模型里这一项很重（§7） |
| DeepEP 文档带宽 | internode EP16 43 GB/s（H800 + CX7 400 Gb/s，约 50 GB/s 上限）；intranode EP8 153 / 158 GB/s（H800 NVLink 约 160 GB/s） | — | DeepEP@v1.2.1 `README.md:15-22` |

---

## 2. 缩小版 K3

### 2.1 保留的结构和比例

| 项 | Kimi-K3 | 缩小版 | 说明 |
|---|---|---|---|
| KDA : MLA | 93 层：MLA 在 `i % 4 == 3` 再加最后一层（23 + 1 = 24），其余 69 层 KDA（`flavors.py:564`） | 29 层：MLA 在 3、7、…、27 再加 28，共 8 层；KDA 21 层 | 层数取 4n + 1 才能同时保留「每 4 层 1 个」和「最后一层 MLA」；debugmodel 也是这个模式（17 层 = 12 + 5，`flavors.py:522-523`） |
| 第 0 层 | dense FFN（33792），其余 LatentMoE | 同（dense FFN = 4.71 D） | `flavors.py:459-475` |
| AttnRes | 块大小 12，共 8 块（最后一块 9 层） | 块大小 4（= 1 个 KDA:MLA 周期），共 8 块（最后一块 1 层，同 debugmodel） | 块数决定每层深度注意力的源数和 PP 传输的块栈（`model.py:262-263, 310-316`） |
| 注意力 | 96 头，头维 128；MLA q/kv 秩 1536 / 512，nope 128 / rope 64 / v 128；KDA 头数 = MLA 头数，头维 128，低秩门 128 | 头数 ≈ 96·D/7168 取 8 的倍数；q/kv 秩按 D 缩放取 64 的倍数；头维、门秩不变 | 头数是 8 的倍数，tp × cp ≤ 8 的 Ulysses 都能用 |
| MoE | latent 3584 = D/2；专家隐藏维 3072 = 0.857 latent；896 选 16；2 个共享专家（隐藏维 2 × 3072） | latent = D/2；专家隐藏维 = 0.857 latent；**128 选 16**；共享专家 2 × 专家隐藏维 | **top-k = 16 不变**：每个 token 发往 16 个专家，dispatch 量（16 × latent）和按节点去重的效果都和 K3 一样，这是 DeepEP 对比实验的前提。E 从 896 降到 128 后路由专家激活比从 1.8% 变成 12.5%，这是为放进 16 卡付出的代价 |
| vocab | 163840 | 163840 | 用真实 tokenizer 训练时必须一致 |

### 2.2 怎么定的宽度

1. 先做三个方向的粗搜（`size_search.js`，层数 21–61、D 1536–4096、E ∈ {64, 128, 256}，每个组合只试 1F1B、all-gather KV、默认 FSDP）：5 种配置都能在 32K 放下的，H100 上最大激活 4.75B（D 3072 · L 21 · E 64）、最大总参数 18.7B（D 2048 · L 25 · E 256）；H200 上最大激活 11.08B（D 4096 · L 29 · E 64）、最大总参数 46.6B（D 3584 · L 21 · E 256）。`size_search2.js` 把四种组合的要求放宽到 8K 再搜一遍，H100 的上界几乎不变——说明卡住的是「5D 全开在 32K」。
2. 固定结构族（L = 29、E = 128、k = 16、块 4）只改宽度，看 5D 全开（ep 8、1F1B）的上限（`probe5.js`）：

| D | 头数 | 总 / 激活 | H100 torchtitan 口径 | H100 页面口径 | H200 torchtitan 口径 | H200 页面口径 |
|---|---|---|---|---|---|---|
| 1792 | 24 | 9.1B / 2.30B | 60K | 74K | 126K | 151K |
| 2048 | 24 | 11.8B / 2.87B | 50K | 64K | 111K | 136K |
| **2304** | 32 | 14.5B / 3.68B | **37K** | 49K | 88K | 108K |
| 2560 | 32 | 17.9B / 4.39B | 30K | 41K | 77K | 97K |
| 2816 | 40 | 22.1B / 5.52B | 20K | 30K | 61K | 78K |
| 3072 | 40 | 26.3B / 6.39B | 14K | 24K | 53K | 70K |
| 3328 | 48 | 30.2B / 7.57B | 8K | 17K | 43K | 57K |
| **3584** | 48 | 35.1B / 8.57B | 3K | 12K | **36K** | 50K |
| 3840 | 48 | 40.3B / 9.64B | 放不下 | 7K | 29K | 44K |
| 4096 | 56 | 45.0B / 11.08B | 放不下 | 2K | 22K | 35K |

3. 再查 (A)–(D) 在 32K 是否也各有一个选项放得下，并看最紧的那个还剩多少（`probe7.js`，碎片 20% + 6 GB 之后）：

| 宽度 | 5D 全开 | (A) | (B) | (C) | (D) | 最小余量 |
|---|---|---|---|---|---|---|
| H100 · D=2048 | 19 GB | 16 GB | 44 GB | 20 GB | 35 GB | 16 GB |
| **H100 · D=2304** | 7 GB | 4 GB | 37 GB | 8 GB | 26 GB | **4 GB** |
| H200 · D=3328 | 20 GB | 16 GB | 70 GB | 19 GB | 48 GB | 16 GB |
| **H200 · D=3584** | 9 GB | 4 GB | 63 GB | 8 GB | 38 GB | **4 GB** |

（每格取该组合里最省显存的选项：5D、(A)、(C)、(D) 为 ep 8 + 1F1B，(B) 为 ep 16 + 每层 reshard。）

结论按 torchtitan 口径：H100 推荐 D = 2304，H200 推荐 D = 3584；按页面口径，5D 全开的界限会放宽到 D = 2560（H100）/ D = 4096（H200）。同族之外也有在粗搜判据下放得下的变体：H100 的 D 2560 · L 25 · E 128（15.5B / 3.84B）、D 2048 · L 37 · E 128（15.0B / 3.57B）、D 2048 · L 25 · E 256（18.7B / 2.53B）；H200 的 D 3584 · L 21 · E 256（46.6B / 6.37B）。D 4096 · L 25 · E 128（38.9B / 9.64B）在 H200 上 5D 全开 32K 刚好放下（碎片 20% 后 140 GB），但 (A)、(C) 在 32K 放不下。它们要么让 AttnRes 只剩 7 块（L 25 / L 21），要么块大小不再是 KDA:MLA 周期的整数倍（L 37 的块大小 5），要么更稀疏（E 256），没选。

### 2.3 推荐配置全表（参数量用 refconfig 的 K3 计数复核，含 norm、AttnRes proj、2 层视觉塔）

| 规格 | 层数（KDA+MLA） | D | 头数（MLA = KDA） | MLA q / kv 秩 | E / k | 专家隐藏维 | latent | 共享专家 | dense FFN（第 0 层） | vocab | AttnRes 块 | 总参数 | 激活参数 | 相对 2.78T / 104.2B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **H100 推荐** | 29（21+8） | 2304 | 32 | 512 / 192 | 128 / 16 | 960 | 1152 | 2 × 960 | 10880 | 163840 | 4 层 × 8 块 | 14.47B | 3.68B | 192× / 28.3× |
| **H200 推荐** | 29（21+8） | 3584 | 48 | 768 / 256 | 128 / 16 | 1536 | 1792 | 2 × 1536 | 16896 | 163840 | 4 层 × 8 块 | 35.06B | 8.58B | 79× / 12.2× |
| H100 保守 | 29（21+8） | 2048 | 24 | 448 / 128 | 128 / 16 | 896 | 1024 | 2 × 896 | 9600 | 163840 | 4 层 × 8 块 | 11.84B | 2.87B | 235× / 36.3× |
| H200 保守 | 29（21+8） | 3328 | 48 | 704 / 256 | 128 / 16 | 1408 | 1664 | 2 × 1408 | 15744 | 163840 | 4 层 × 8 块 | 30.16B | 7.57B | 92× / 13.8× |
| debugmodel（现成） | 17（12+5） | 256 | 4 | 128 / 64 | 8 / 2 | 128 | 128 | 2 × 128 | 512 | 2048 | 4 层 × 5 块 | 26.9M | 18.3M | 103412× / 5681× |
| Kimi-K3（现成，配置 (c)） | 93（69+24） | 7168 | 96 | 1536 / 512 | 896 / 16 | 3072 | 3584 | 2 × 3072 | 33792 | 163840 | 12 层 × 8 块 | 2.780T | 104.2B | 1× |

其余字段与 K3 相同：qk_nope 128、qk_rope 64、v_head 128、kda_head_dim 128、conv 4。视觉塔建议用 debugmodel 的尺寸（2 层、4 头），头数要能被 tp 整除。页面 `modelParams` 与 refconfig 的计数只差 norm 与视觉塔（5–7M）。路由专家占总参数 82%（H100：11.89B）/ 84%（H200：29.60B）。debugmodel 每卡算量太小，MFU 没有意义，只适合跑通流程。

### 2.4 5D 全开的 ep 选项（pp = tp = cp = dp_shard = 2，ep ∈ {2, 4, 8}：ep 要整除 dp_shard·cp·tp = 8、≥ tp、整除 E）

| 卡 | ep | 调度 | 上限 20% / 10% | 页面口径上限 20% | 推荐长度 | MFU @推荐 | MFU @8K | MFU @32K |
|---|---|---|---|---|---|---|---|---|
| H100 | 2 | 1F1B | 24K / 30K | 28K | 16K | 14.1% | 13.3% | — |
| H100 | 2 | Interleaved1F1B | 18K / 22K | 21K | 16K | 14.5% | 13.6% | — |
| H100 | 4 | 1F1B | 33K / 38K | 38K | 32K | 14.6% | 13.6% | 14.6% |
| H100 | 4 | Interleaved1F1B | 24K / 28K | 28K | 16K | 14.6% | 13.8% | — |
| H100 | **8** | **1F1B** | **37K / 43K** | 49K | 32K | 14.6% | 13.7% | 14.6% |
| H100 | 8 | Interleaved1F1B | 28K / 32K | 36K | 16K | 14.6% | 13.9% | 15.4%（超余量） |
| H200 | 2 | 1F1B | 14K / 21K | 17K | 8K | 19.8% | 19.8% | — |
| H200 | 2 | Interleaved1F1B | 11K / 15K | 12K | 8K | 20.3% | 20.3% | — |
| H200 | 4 | 1F1B | 29K / 35K | 34K | 16K | 21.1% | 20.7% | 20.9%（超余量） |
| H200 | 4 | Interleaved1F1B | 21K / 26K | 25K | 16K | 21.8% | 21.1% | — |
| H200 | **8** | **1F1B** | **36K / 43K** | 50K | 16K | 21.3% | 21.1% | 21.0% |
| H200 | 8 | Interleaved1F1B | 27K / 32K | 37K | 16K | 22.0% | 21.4% | 22.3%（超余量） |

推荐 ep = 8：EP 借满 stage 内的 dp_shard × cp × tp，专家不再做 FSDP，显存最省，MFU 与 ep 4 持平。Interleaved1F1B 的 bubble 小、MFU 高 0.5–1 个点，但在途 micro-batch 多，32K 放不下。CP 的两种算法在 cp = 2、节点内时几乎没有差别（MFU 差 < 0.1 个点）。

---

## 3. 四种组合的明细

### 3.1 共同前提

- 2 节点 × 8 卡，rank r 在节点 ⌊r/8⌋；torchtitan 的 rank 顺序 pp → dp_replicate → dp_shard → cp → tp（tp 最内），页面 `rankOf()` 同序（`model3.js:190`）。所以 **pp = 2 时 PP 占掉跨节点那一维，stage 内的 8 卡就是一台机器，TP、CP、FSDP、EP 都在 NVLink 内；pp = 1 时最外层的 dp_shard 跨节点。**
- 每 step 512K token；M = 512K / (s · dp_shard)。
- 「上限」= torchtitan 口径显存在碎片 20% / 10% 下放得下的最长长度（步长 1K）；「页面口径」一列是不加 §1.4 两条修正的上限。
- 「下限」：硬判据（每卡每 micro-batch ≥ 2048 token，即 `s/(cp·tp) ≥ 2048`；CP 暴露 ≤ 5%；M ≥ P）与「MFU ≥ 最高值的 90%」取大。CP 都在节点内，暴露只占 0.7–3.1%，**CP 的「藏得住」这一条在下列配置里从不卡下限**；M ≥ P 在 512K token / step 下也只在长端起作用。
- 「推荐」= 区间内 MFU 最高的 2 的幂长度。有 PP 的组合 MFU 曲线在 8K–32K 很平（差 1 个点以内），实际可以按显存余量往短处挪。

### 3.2 选度数的过程（每个组合三种三维分解 × 能取的 ep，完整表见附录 A）

- **(A) EP+FSDP+TP+PP（cp = 1）**：(pp, dp_shard, tp) ∈ {(2, 4, 2), (2, 2, 4), (4, 2, 2)}。没有 CP，整条序列的激活压在一个 TP 组上：dp_shard 4 · tp 2 在 H100 只到 19K（ep 8）、H200 18K；pp 4 的 stage 只有 4 卡，ep ≤ 4，上限 17–22K。只有 **2 · 2 · 1 · 4 · 8**（tp = 4 把激活再切一半）在 32K 放得下。
- **(B) EP+FSDP+TP+CP（pp = 1）**：(dp_shard, cp, tp) ∈ {(4, 2, 2), (2, 4, 2), (2, 2, 4)}，ep ∈ {2, 4, 8, 16}。没有 PP，dp_shard 必然跨节点：FSDP 组（dp_shard × cp）每节点 4–8 个、跨 2 节点；ep ≤ 8 时专家还要做跨节点的 2 路 FSDP。torchtitan 默认每层 reshard，每个 micro-batch 每层都要跨节点 all-gather 参数、reduce-scatter 梯度，短序列时这部分完全藏不住（H100 8K：FSDP 暴露 43%，MFU 7.9%）。选 **1 · 2 · 4 · 2 · 8**（cp 4 把 token 切得最细）：H100 用默认 reshard、推荐 64K；H200 显存够，改成整步不 reshard（`fsdp_reshard_after_forward="never"` + `fsdp_defer_gradient_reduction=True`），32K 时暴露 7.6%。H100 若也改成整步不 reshard：上限 59K，32K MFU 16.8%（暴露 8.0%），比默认在 32K 高 2.6 个点，但到不了 64K。
- **(C) EP+FSDP+PP+CP（tp = 1）**：(pp, dp_shard, cp) ∈ {(2, 4, 2), (2, 2, 4), (4, 2, 2)}。没有 TP，stage 内 dense 参数不切，6P 常驻按整份算（H100 32K：未切分 12.7 GB）。cp 4 把激活切到 1/4 才在 32K 放得下：选 **2 · 2 · 4 · 1 · 8**。
- **(D) EP+TP+PP+CP（dp_shard = 1）**：(pp, cp, tp) ∈ {(2, 4, 2), (2, 2, 4), (4, 2, 2)}。数据并行度 = 1，全局 batch 全靠 PP 的 micro-batch（M = 512K / s），参数仍在 cp 轴上被 FSDP 切（FSDP 轴 = dp_shard × cp）。选 **2 · 1 · 4 · 2 · 8**；H200 上 Ulysses 比 all-gather KV 高 0.3 个点（21.9% vs 21.6%，64K），两者都可以。

### 3.3 推荐长度与参考长度下的 step 分解（各 stage 平均；「暴露通信合计」= TP + CP + EP + 主机同步 + FSDP + 全局 all-reduce）

**H100（D=2304）**

| 组合 | 长度 | step | M | 计算 | TP | CP | EP（含主机同步） | FSDP | PP bubble | 暴露通信合计 | 显存 torchtitan / 页面口径（最重 stage） | 碎片 20% 需求 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5D 全开 | 32K（推荐） | 6.93 s | 8 | 79.4% | 1.2% | 0.7% | 3.1% | 0.1% | 15.5% | 5.1% | 61.8 / 52.8 GB | 73.0 GB |
| 5D 全开 | 8K | 5.92 s | 32 | 86.0% | 1.9% | 1.1% | 3.9% | 0.1% | 7.0% | 7.1% | 37.7 / 32.1 GB | 44.0 GB |
| (A) | 32K（推荐） | 7.00 s | 8 | 78.6% | 3.4% | 0.0% | 2.7% | 0.0% | 15.3% | 6.1% | 64.5 / 51.0 GB | 76.2 GB |
| (A) | 8K | 5.95 s | 32 | 85.2% | 4.5% | 0.0% | 3.5% | 0.0% | 6.7% | 8.1% | 37.2 / 30.4 GB | 43.4 GB |
| (B) | 64K（推荐） | 7.19 s | 4 | 86.2% | 1.2% | 2.0% | 3.0% | 7.7% | 0.0% | 13.8% | 53.4 / 49.0 GB | 62.9 GB |
| (B) | 32K | 7.11 s | 8 | 78.0% | 1.3% | 2.1% | 3.1% | 15.5% | 0.0% | 22.0% | 37.4 / 35.2 GB | 43.6 GB |
| (B) | 8K | 10.27 s | 32 | 51.0% | 1.5% | 1.9% | 2.5% | 43.1% | 0.0% | 49.0% | 25.4 / 24.8 GB | 29.2 GB |
| (C) | 32K（推荐） | 6.99 s | 8 | 78.8% | 0.0% | 2.1% | 3.2% | 0.2% | 15.7% | 5.5% | 61.8 / 57.4 GB | 73.0 GB |
| (C) | 8K | 6.02 s | 32 | 85.4% | 0.0% | 2.9% | 4.1% | 0.3% | 7.3% | 7.3% | 40.4 / 36.0 GB | 47.3 GB |
| (D) | 64K（推荐） | 7.98 s | 8 | 77.6% | 1.1% | 1.8% | 2.7% | 0.1% | 16.8% | 5.6% | 62.8 / 53.8 GB | 74.2 GB |
| (D) | 32K | 6.76 s | 16 | 81.9% | 1.4% | 2.2% | 3.2% | 0.1% | 11.1% | 6.9% | 46.2 / 39.5 GB | 54.3 GB |
| (D) | 8K | 6.20 s | 64 | 84.4% | 2.5% | 3.1% | 4.2% | 0.1% | 5.7% | 9.9% | 33.8 / 28.7 GB | 39.3 GB |

**H200（D=3584）**

| 组合 | 长度 | step | M | 计算 | TP | CP | EP（含主机同步） | FSDP | PP bubble | 暴露通信合计 | 显存 torchtitan / 页面口径（最重 stage） | 碎片 20% 需求 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5D 全开 | 16K（推荐） | 9.01 s | 16 | 85.9% | 1.5% | 0.9% | 3.2% | 0.2% | 8.3% | 5.8% | 87.0 / 72.3 GB | 103.2 GB |
| 5D 全开 | 8K | 8.64 s | 32 | 88.5% | 1.8% | 1.0% | 3.5% | 0.2% | 5.0% | 6.5% | 74.7 / 61.8 GB | 88.4 GB |
| 5D 全开 | 32K | 10.11 s | 8 | 80.9% | 1.2% | 0.7% | 2.8% | 0.2% | 14.1% | 4.9% | 111.6 / 93.4 GB | 132.7 GB |
| (A) | 16K（推荐） | 9.13 s | 16 | 84.7% | 4.1% | 0.0% | 3.0% | 0.1% | 8.1% | 7.2% | 87.0 / 68.9 GB | 103.2 GB |
| (A) | 8K | 8.71 s | 32 | 87.4% | 4.5% | 0.0% | 3.3% | 0.1% | 4.8% | 7.8% | 73.0 / 58.4 GB | 86.4 GB |
| (A) | 32K | 10.25 s | 8 | 79.8% | 3.5% | 0.0% | 2.7% | 0.1% | 13.9% | 6.3% | 115.0 / 89.8 GB | 136.8 GB |
| (B) | 32K（推荐） | 8.95 s | 8 | 92.4% | 1.5% | 2.4% | 3.2% | 0.4% | 0.0% | 7.6% | 105.8 / 102.3 GB | 125.7 GB |
| (B) | 8K | 8.89 s | 32 | 89.8% | 2.2% | 3.0% | 3.6% | 1.3% | 0.0% | 10.2% | 87.4 / 86.5 GB | 103.7 GB |
| (C) | 16K（推荐） | 9.15 s | 16 | 85.1% | 0.0% | 2.5% | 3.4% | 0.4% | 8.7% | 6.2% | 91.9 / 80.8 GB | 109.1 GB |
| (C) | 8K | 8.83 s | 32 | 87.8% | 0.0% | 2.9% | 3.6% | 0.4% | 5.3% | 6.9% | 81.0 / 69.9 GB | 96.0 GB |
| (C) | 32K | 10.21 s | 8 | 80.2% | 0.0% | 2.1% | 3.0% | 0.3% | 14.4% | 5.4% | 113.6 / 102.5 GB | 135.2 GB |
| (D) | 64K（推荐） | 11.51 s | 8 | 80.1% | 1.1% | 0.7% | 2.5% | 0.1% | 15.5% | 4.4% | 111.6 / 93.4 GB | 132.7 GB |
| (D) | 32K | 9.70 s | 16 | 85.2% | 1.4% | 0.9% | 3.0% | 0.2% | 9.3% | 5.5% | 87.0 / 72.3 GB | 103.2 GB |
| (D) | 8K | 8.97 s | 64 | 89.0% | 2.2% | 1.6% | 3.6% | 0.2% | 3.4% | 7.5% | 68.5 / 56.5 GB | 81.0 GB |

### 3.4 上下文区间汇总（含页面口径）

| 卡 | 组合 | 下限（硬下限 / MFU 平台） | 上限 torchtitan 口径 20% / 10% | 上限 页面口径 20% / 10% | 推荐 |
|---|---|---|---|---|---|
| H100 | 5D 全开 | 8K（8K / 8K） | 37K / 43K | 49K / 55K | 32K |
| H100 | (A) | 8K（8K / 4K） | 34K / 39K | 51K / 57K | 32K |
| H100 | (B) | 64K（16K / 64K） | 92K / 103K | 107K / 120K | 64K |
| H100 | (C) | 8K（8K / 8K） | 38K / 44K | 43K / 49K | 32K |
| H100 | (D) | 32K（16K / 32K） | 73K / 84K | 94K / 107K | 64K |
| H200 | 5D 全开 | 8K（8K / 4K） | 36K / 43K | 50K / 58K | 16K |
| H200 | (A) | 8K（8K / 4K） | 33K / 39K | 53K / 61K | 16K |
| H200 | (B) | 16K（16K / 16K） | 48K / 62K | 56K / 72K | 32K |
| H200 | (C) | 8K（8K / 4K） | 35K / 43K | 43K / 51K | 16K |
| H200 | (D) | 16K（16K / 8K） | 72K / 86K | 101K / 117K | 64K |

### 3.5 横向读法

- 有 PP 的三种（5D、A、C）和 (D) 在 8K–32K 的 MFU 只差 1 个点左右：PP 把跨节点的那一维吃掉以后，剩下的通信全在 NVLink 里，暴露通信只有 5–8%；拖 MFU 的是 PP bubble（每 step 512K token 时，32K 下 M = 8，bubble 约 15%）和页面里很重的访存型算子（§7）。
- (B) 没有 bubble，但 dp_shard 只能跨节点：默认每层 reshard 时短序列很差（8K 暴露 49%），要到 64K 才把 FSDP 摊薄；FSDP 改成整步不 reshard 后 8K 就有 13.8%（H100）/ 20.5%（H200），代价是常驻未切分参数（H200 32K 时 39.8 GB）。
- (D) 没有数据并行，同样的 token 全变成 micro-batch，M 大一倍，bubble 小，能跑更长（cp 4 · tp 2 把每卡 token 切到 1/8）。
- 上下文长度在这里主要受显存和 bubble 两头夹：有 PP 的 5D、(A)、(C) 上限 33–38K；(D) 把每卡 token 切到 1/8，上限 72–73K；(B) 用默认每层 reshard 时到 92K（H100），整步不 reshard 时 48K（H200，常驻未切分参数占掉了空间）。

### 3.6 torchtitan 侧要改的配置（推荐配置；s 为上下文长度）

| 项 | 5D 全开 | (A) | (B) | (C) | (D) |
|---|---|---|---|---|---|
| `pipeline_parallel_degree` | 2 | 2 | 1 | 2 | 2 |
| `data_parallel_shard_degree` | 2 | 2 | 2 | 2 | 1 |
| `context_parallel_degree` | 2 | 1 | 4 | 4 | 4 |
| `tensor_parallel_degree` | 2 | 4 | 2 | 1 | 2 |
| `expert_parallel_degree` | 8 | 8 | 8 | 8 | 8 |
| `pipeline_parallel_schedule` / `num_pp_microbatches` | 1F1B / 512K ÷ (2s) | 1F1B / 512K ÷ (2s) | — | 1F1B / 512K ÷ (2s) | 1F1B / 512K ÷ s |
| CP | `ContextParallelTransform`：`FlexInnerAttention → KVAllGatherCPFlexInnerAttention`（可配 `HeadTailCPLoadBalancer`），`InnerKDA → ContextParallelInnerKDA` | — | 同 5D | 同 5D | 同 5D；H200 可换 `UlyssesCPFlexInnerAttention`（不配 load balancer） |
| FSDP | 默认 | 默认 | H100 默认；H200 `fsdp_reshard_after_forward="never"`、`fsdp_defer_gradient_reduction=True` | 默认 | 默认 |

所有组合：`training.num_tokens_per_microbatch_per_dp_rank = training.max_context_length = s`（且等于模型的 `max_context_length`，`validation.py:51-61`）、`training.num_tokens_per_train_step = 524288`、`training.disable_cuda_graphs = True`、优化器 AdamW、AC 用默认 `SelectiveAC`。CP 的写法参照 `torchtitan_recipes/tests/suites/h100.py:54-110`，PP + TP + EP 的写法参照 `b200.py:74-114`。

---

## 4. 页面口径与 torchtitan 口径的差别

| 项 | H100 5D @32K | H200 5D @32K |
|---|---|---|
| 激活（页面 → torchtitan） | 25.9 → 30.4 GB | 39.5 → 46.5 GB |
| 未切分参数 + 梯度（页面 → torchtitan） | 4.4 → 8.8 GB | 9.4 → 20.5 GB |
| 合计 | 52.8 → 61.8 GB | 93.4 → 111.6 GB |
| 碎片 20% 需求 | 62.2 → 73.0 GB | 110.9 → 132.7 GB |

按页面口径，5D 全开的上限是 49K（H100）/ 50K（H200），按 torchtitan 口径是 37K / 36K。tp 越大，「all-gather 完整输入」这一条差得越多（(A) 的 tp = 4：上限 51K → 34K）；tp = 1 的 (C) 只差专家 6P 一项（43K → 38K）。

---

## 5. 实验目标的纸面预期（全部以实测为准）

### 5.1 nccl-tests busbw（页面 `coll()` / `a2a()`，每 rank 1 GiB；nccl-tests 的 busbw 折算：all_reduce 2(n−1)/n，all_gather / reduce_scatter / alltoall (n−1)/n，sendrecv 1，见 nccl-tests@afd59ab `doc/PERFORMANCE.md:136-145`）

| 测试 | rank | 位置 | algbw | busbw |
|---|---|---|---|---|
| all_reduce_perf | 8 | 节点内 NVLink | 205 GB/s | 359 GB/s |
| all_gather_perf / reduce_scatter_perf | 8 | 节点内 NVLink | 409 GB/s | 358 GB/s |
| alltoall_perf | 8 | 节点内 NVLink | 384 GB/s | 336 GB/s |
| sendrecv_perf（跨节点的一对 rank） | 2 | 跨节点 IB（单网卡） | 42 GB/s | 42 GB/s |
| alltoall_perf | 16 | 跨 2 节点 | 80 GB/s | 75 GB/s |
| all_reduce_perf | 16 | 跨 2 节点 | 180 GB/s | 338 GB/s |
| all_gather_perf | 16 | 跨 2 节点 | 358 GB/s | 335 GB/s |

H100 与 H200 的 NVLink、网卡相同，预期一样。E2 的节点内带宽取「节点内 8 卡 all_reduce」（≈ 350 GB/s，经验值），跨节点带宽取「跨节点单网卡 sendrecv」（≈ 42 GB/s，规格 50 GB/s），两者差约 8 倍，这正是 TP / EP 要留在节点内的原因。16 卡 all_reduce 的 busbw 页面给 338 GB/s，是因为它假设 8 张网卡并行、分层收发，不代表单卡跨节点带宽，报这个数时要说明是哪种测试。

### 5.2 暴露通信：坏放置 → 推荐放置（页面 `simulate()`，各 stage 平均）

| 卡 | 场景 | 长度 | 坏：暴露（主要项） | 好：暴露（主要项） | MFU 坏 → 好 |
|---|---|---|---|---|---|
| H100 | 5D 全开：TP 跨节点（what-if） | 32K | 11.9%（TP 8.8%） | 5.1%（TP 1.2%） | 13.4% → 14.6% |
| H100 | (A)：tp = 4 跨节点（what-if） | 32K | 14.8%（TP 12.5%） | 6.1%（TP 3.4%） | 13.0% → 14.4% |
| H100 | (B) 整步不 reshard：EP = 16 跨节点（NCCL）vs EP = 8 节点内 | 32K | 24.8%（EP 21.3%） | 8.0%（EP 3.6%） | 13.8% → 16.8% |
| H100 | (B) 每层 reshard：EP = 16 跨节点（NCCL）vs EP = 8 节点内 | 32K | 27.2%（EP 20.6%，FSDP 3.4%） | 22.0%（FSDP 15.5%） | 13.3% → 14.2% |
| H100 | (B) 每层 reshard，同上 | 64K | 23.5%（EP 19.1%） | 13.8%（FSDP 7.7%） | 15.8% → 17.8% |
| H200 | 5D 全开：TP 跨节点（what-if） | 16K | 14.2%（TP 10.4%） | 5.8%（TP 1.5%） | 19.4% → 21.3% |
| H200 | 5D 全开：TP 跨节点（what-if） | 32K | 12.4%（TP 9.1%） | 4.9%（TP 1.2%） | 19.2% → 21.0% |
| H200 | (A)：tp = 4 跨节点（what-if） | 16K | 17.6%（TP 14.8%） | 7.2%（TP 4.1%） | 18.6% → 21.1% |
| H200 | (B) 整步不 reshard：EP = 16 跨节点（NCCL）vs EP = 8 节点内 | 32K | 24.4%（EP 21.0%） | 7.6%（EP 3.2%） | 19.5% → 23.7% |
| H200 | (B) 每层 reshard：EP = 16 跨节点（NCCL）vs EP = 8 节点内 | 32K | 28.2%（EP 20.0%，FSDP 5.1%） | 28.6%（FSDP 23.0%） | 18.6% → 18.3% |

读法：

- **E3 的量级：EP 跨节点约 22–28% → 约 7–8%，TP 跨节点约 12–18% → 约 5–7%。** EP 跨节点那组要配「FSDP 整步不 reshard」（或像 5D 那样让 PP 占跨节点维），否则 EP 收回节点内以后，专家 FSDP 又跨节点，暴露从 EP 挪到 FSDP（上表 (B) 每层 reshard 两行）。
- TP 跨节点在 torchtitan 的默认 rank 顺序下不会出现（tp 永远在最内层），要靠打乱 rank→节点的映射来做这组坏对照；这里只把 TP 的 AG / RS 换成跨节点带宽（`formulas.md` §10），其他维的放置没跟着变。
- 页面 DeepEP 公式下，EP = 16 跨节点只比节点内多 0.5–0.7 个点的暴露（H100 32K：8.7% vs 8.0%；H200 32K：8.3% vs 7.6%）；坏放置要用 NCCL all-to-all 才显得坏。

### 5.3 E4：EP = 16 跨 2 节点，DeepEP internode 对 NCCL all-to-all

每个 token 过网卡的行数（均匀路由，k = 16）：

- NCCL（torchtitan `AllToAllTokenDispatcher`，不去重）：16 份里落在另一节点的 8 份各发一行 → **8 行**；
- DeepEP（按节点去重）：只要另一节点上至少有 1 个选中专家就发 1 行 → `1 − 0.5^16` ≈ **1 行**（页面 `model3.js:330-331`）；行数比 8×。

时间比还要看瓶颈在哪一段：

| 估法 | DeepEP 时间（每 token，单位 = 一行字节） | NCCL 时间 | 比 |
|---|---|---|---|
| 页面公式（DeepEP 的 NVLink 段按 16 份全部转发，337.5 GB/s；网卡 40 GB/s） | max(1/40, 16/337.5) → NVLink 受限 | max(8/40, 7/337.5) | **4.2×** |
| NVLink 段按「每个目标 GPU 一份」去重（补算：均匀路由下平均 10.3 个目标 GPU） | max(1/40, 10.3/337.5) | 同上 | **6.6×** |
| DeepEP 文档 43 GB/s，按网卡上实际的 1 行计（网卡受限） | 1/43 | 同上 | **8.6×** |
| DeepEP 文档 43 GB/s，按 DeepEP 测试脚本的计数（含本节点，约 2 行；deepep@v1.2.1 `tests/test_internode.py:49-52, 179`） | 2/43 | 同上 | 4.3× |

页面整层：(B) 的网格改成 EP = 16，每层 dispatch + combine（前向 + 反向）H100 32K 时 NCCL 7.67 ms vs DeepEP 1.91 ms（4.0×），64K 时 15.22 vs 3.70 ms（4.1×）；H200 32K 时 11.86 vs 2.90 ms（4.1×）。每卡每次 dispatch 151 MB（H100 32K）/ 235 MB（H200 32K），都在带宽区。整步只快 1.19–1.21×（EP 暴露从约 21% 降到约 4%）。

**纸面预期：E4 ≈ 4–9×；纸面判断最可能落在 4–7×**，因为 k = 16 时 DeepEP 自己的 NVLink 转发段很重。对 DeepEP 文档数字的解读（补算，按 DeepEP 测试脚本的默认设置 256 专家选 8、`num_topk_groups = min(节点数, 4)`、均匀路由折算，`tests/test_internode.py:301-318`）：EP16 时 NVLink 侧的转发量约是 RDMA 计数的 3.2 倍，43 GB/s 对应约 140 GB/s 的 NVLink 流量，接近 H800 约 160 GB/s 的上限，所以 H800 上 EP16 其实是 NVLink 受限；EP32 / EP64 的 58 / 51 GB/s 折成网卡上实际的行数约 43–45 GB/s，「网卡受限 43 GB/s」可以当 H100 / H200 上限的参照。若 NCCL 跨节点 all-to-all 实际到不了每卡 40 GB/s，E4 按比例变大。

### 5.4 实测时怎么对上这些预期

- E2：同一台机器上跑 `all_reduce_perf -b 8 -e 8G -f 2 -g 1`（8 进程）与跨节点 `sendrecv_perf`、`alltoall_perf`（16 进程），记大消息区的 busbw。
- E3：用 torchtitan profiler trace 按发起方拆暴露通信（之前的 `trace_breakdown.py` 思路），坏放置跑 (B) 的 EP = 16（NCCL）和打乱 rank 映射的 TP；好放置跑推荐配置。
- E4：torchtitan@432b95dcf 的 DeepEP dispatcher 要 v2 的 `ElasticBuffer`；DeepEP main@b5a33ee 已把它拆成 `EPBuffer` 等（README「V2.5 release」），v1.2.1 是旧的 `Buffer` 接口，两者都接不上这一版 torchtitan。比较稳的是单独做 micro-benchmark：DeepEP v1.2.1 的 `tests/test_internode.py`（改成 hidden = latent 1152 / 1792、topk 16、128 专家、每卡 4096–8192 token；normal kernel 支持 `num_topk ≤ 32`，`csrc/kernels/internode.cu:408`）对照 NCCL all-to-all（同样的 token × 16 份），比 dispatch + combine 的时间；要在 torchtitan 里比，就得钉一个带 `ElasticBuffer` 的 DeepEP 2.x。

---

## 6. 假设清单

1. **硬件**：H100 SXM 80 GB、H200 SXM 141 GB（近似值）；BF16 稠密 989 TFLOPS（近似值）；HGX 8 卡 NVLink4 + NVSwitch，450 GB/s 单向（规格近似值），有效 360 GB/s（页面效率 0.80；经验值 350 GB/s）；每卡一张 400 Gb/s 网卡 ≈ 50 GB/s（规格，近似值）；2 节点、rail 对齐、无 SHARP。
2. **余量**：`(状态 + 未切分 + 激活 + buffer) × (1 + f) + 6 GB ≤ HBM`，f = 10% / 20%，6 GB = 页面 `overheadGB`（CUDA context、NCCL 缓冲、workspace）。
3. **训练设置**：BF16 GEMM；AdamW + fp32 主参数（16 B/参分片）；默认 SelectiveAC（页面 `sac` + all-gather 完整输入）；不开 EP 通信重叠；一条序列一个 micro-batch，注意力按整条序列（不按文档打包）；每 step 512K token（页面「预训练」场景 32M token / 1024 卡 = 每卡 32K token，按 16 卡缩放）。
4. **EP**：节点内 EP 用页面的 a2a 公式（含 notify 0.012 ms + 主机同步 0.03 ms）代表 NCCL all-to-all；跨节点 NCCL 不去重（补算）；路由均匀，页面按平均 token × 1.2 留负载不均余量（显存）。
5. **下限阈值**：每卡每 micro-batch ≥ 2048 token；CP 暴露 ≤ 5%；M ≥ P；MFU ≥ 最高值的 90%。
6. **结构**：层数 4n + 1、AttnRes 块数 ≈ 8、k = 16、E 是 16 的倍数、头数是 8 的倍数、vocab 163840、视觉塔取 debugmodel 尺寸。
7. **MFU 口径**：页面 `modelFlopsPerToken` = 3 ×（2 × 激活参数 + 注意力核，MLA 按 causal 折半），相对 989 TFLOPS。

## 7. 拿不准的地方

1. **torchtitan 的激活显存**：现在的 `SelectiveAC` 是「保存所有声明的区域、只重算 routed w13」（`activation_checkpoint.py:210-246`）。本文用页面 `sac` + all-gather 完整输入近似；若实际保存得更多（例如各投影输出都留着），上限会更低。建议先在 8K / 16K 跑一个 step 记录 `max_memory_allocated`，按本文表里同长度的值校准，再往 32K 推。
2. **edp = 1 时专家的 6P**：按「PP + FSDP 时 bf16 参数 + 未分片 fp32 梯度常驻」对大小为 1 的专家 FSDP 也算上（H100 4.4 GB、H200 11.1 GB）。若 FSDP2 在这种情况下复用了缓冲，余量会多出这些。
3. **MFU 的绝对值**：页面 GEMM 效率只随 token 维变化，不管 D 只有 2304 / 3584 时 N、K 维偏小；访存型算子乘 `em = 4.0`（NeMo 在大模型上标定），在 k = 16 的细粒度 MoE 里 permute / 合并很重：5D 全开 32K 时一层 MoE 前向里，H200 的「升回 + permute / 合并 / 残差」4.7 ms、专家 GEMM 4.1 ms；H100 是 4.2 ms 对 1.6 ms。H100 与 H200 峰值相同、MFU 却差 6–7 个点，主要就是 HBM 带宽（3.35 vs 4.8 TB/s）乘上这个系数。MFU 的相对比较比绝对值可靠。
4. **KDA 在 H100 / H200 上走 Attention Gym 的 Triton 核**（`kimi_k3/README.md:14-16`），页面 `linF 0.30 / linB 0.25` 没在 SM90 上标定过。
5. **AttnRes 的 PP 传输**：页面按 `1 + L/(块大小·V)` = 8.25 份算 P2P；pp = 2 的 1F1B 实际一跳只带 stage 0 提交的 4 块加 hidden（约 5 份）。P2P 是异步的，对 step 影响很小。
6. **全局 batch**：bubble 和 M 都由「每 step 512K token」决定。把全局 batch 加倍，32K 下 M 从 8 变 16，PP 组合的 bubble 约减半，推荐长度可能往长处移。
7. **跨节点集合通信**：页面把 16 卡 AG / RS / AR 当作每节点 8 张网卡并行（338 GB/s busbw），NCCL 的实际算法和 rail 拓扑可能给出明显不同的值；跨节点 all-to-all 每卡 40 GB/s 也是页面效率假设。
8. **E4 的区间**取决于 DeepEP 在 H100 / H200 上是 NVLink 转发受限还是网卡受限，以及 latent 只有 1152 / 1792 维（每行 2.3–3.6 KB，比 DeepEP 测试的 7168 维小）时的效率；DeepEP 文档的数字来自 H800。
9. **torchtitan 日志里的 mfu** 不按 causal 折半 MLA（`torchtitan/models/utils.py:409-440`），KDA 按 `6·3·H·dk·dv`（:443-470），和本文口径不同，对照时要换算。
10. **torchtitan 的 DeepEP 路径与 DeepEP 版本**不匹配（§5.4），E4 的实验大概率要在 torchtitan 外面做。
11. 层数 29 时 (L + 2) = 31 不能被 2 整除，stage 0 比 stage 1 多一层（页面 `assignLayers`，torchtitan 的 first / last stage 各少一层的切法相同），stage 0 是最重的 stage。
12. 本文不涉及 2.78T 全尺寸的任何实测；全尺寸只作为缩小倍数的分母（refconfig 配置 (c)：2.7799T / 104.19B，K3 报告 2.78T / 104.2B）。

---

## 8. 文件

| 文件 | 内容 |
|---|---|
| `<SP>/k3_16g/plan.md` | 本文 |
| `<SP>/k3_16g/formulas.md` | 页面模型公式与行号、补算项 |
| `<SP>/k3_16g/lib.js` | vm 载入 `model3.js`、what-if 钩子、torchtitan 显存修正、余量判据 |
| `<SP>/k3_16g/configs.js` | 尺寸规则 `dimsFor()`、页面开关映射 `baseCfg()`、组合与 ep 候选 |
| `<SP>/k3_16g/maxctx.js` | 二分最长上下文 |
| `<SP>/k3_16g/plan_calc.js` → `results.json` | 全部组合 × 选项的区间、MFU 曲线、推荐与明细 |
| `<SP>/k3_16g/bullet_calc.js` → `bullet.json` | busbw、暴露通信对照、DeepEP / NCCL、DeepEP 文档带宽的折算 |
| `<SP>/k3_16g/size_search.js`、`size_search2.js` → `size_search_32K.json`、`size_search2_32K_8K.json` | 粗搜 |
| `<SP>/k3_16g/probe5.js` → `width_scan.json`；`probe7.js` → `slack_32K.json` | 同族宽度扫描、32K 余量 |
| `<SP>/k3_16g/refcheck.py` → `refcheck.json` | refconfig 复核参数量与缩小倍数 |
| `<SP>/k3_16g/make_tables.js` → `tables.md` | 本文表格的生成源（含附录 A 的完整候选表） |

重算：`cd <SP>/k3_16g && node plan_calc.js && node bullet_calc.js && node probe5.js && node probe7.js && python3 -I refcheck.py && node make_tables.js`（`plan_calc.js` 约 5 分钟）。

---

## 9. 换成 A100 80GB（2 节点 × 8 卡）

> 2026-10-28 补充。本节没有用页面模型重算，数字是按硬件比例的粗估。正式使用前，在 `<SP>` 所在的会话里把 HW 预设换成 A100，重跑 `plan_calc.js` 和 `bullet_calc.js`。

### 9.1 跑 K3 要的两处本地改动

- KDA 主体（`chunk_kda`）是 Attention Gym 的 Triton 核，sm_80 能跑。挡路的是门控变换：torchtitan 写死了 `bound_gate(..., impl="fused")`（`torchtitan/models/common/attention/kda.py:170`），而 fused 的 bounded 门控只有 CuTeDSL 核，要求 CC ≥ 9.0（attn_gym `linear/_delta_rule/gate.py` 的 NOTE「Softplus Backend Selection」）。A100 上把它改成 `impl="reference"`。
- 注意力：432b95dcf 的 K3 recipe 默认 `attn_backend="flex"`（`kimi_k3/flavors.py:602`），不用改。新 main 从 #5307 起默认 varlen，而 MLA 的 q/k 头维 192 ≠ v 头维 128，sm_80 / sm_89 上 FA varlen 会报错，所以要显式传 `attn_backend="flex"`。
- 已验证：在 4070 Ti（sm_89）上，K3 debugmodel 单卡跑了 10 步，loss 8.10 → 6.52。A100 还没验证；开机后先跑 debugmodel 单卡 10 步和 KDA 单测（`impl="reference"`）。

### 9.2 什么能照搬

| 项 | 能否照搬 | 说明 |
|---|---|---|
| 宽度 D、参数量 14.5B / 3.68B | 能 | 只取决于 HBM，A100 SXM 也是 80 GB |
| 五种组合的度数、放置、§3.6 的配置 | 能 | 并行约束和硬件无关；另加 9.1 的两处改动 |
| 上下文上限（显存口径） | 基本能，但要扣掉 reference 门控 | 见 9.3 |
| 上下文下限、推荐长度 | 要重算 | 下限里的「CP 暴露 ≤ 5%」「MFU ≥ 最高值 90%」都按时间算 |
| step 时间、MFU、暴露通信 %、E2–E4 | 都要重算 | 见 9.4、9.5 |

### 9.3 reference 门控多出的显存

reference 是普通 PyTorch：`lower_bound * sigmoid(exp(A_log) * (raw_gate.float() + dt_bias))`。autograd 会存两份 fp32 中间量：加完 dt_bias 的输入，和 sigmoid 的输出。fused 只存输入，反向时重算。

按全部保存算上界：每个 KDA 层、每个在途 micro-batch 多出 `8 B × 每 rank token 数 × 每 rank 头数 × 128`。

- **5D 全开 @32K 的 stage 0：**
    - 每 rank 16K token（cp 2）× 16 头（tp 2），每层 268 MB。
    - stage 0 的 15 层里有 12 层是 KDA，1F1B 下有 2 个在途 micro-batch，合计约 6.4 GB，乘碎片 20% 后约 7.7 GB。
    - §0.4 这一档在 32K 只剩约 7 GB 余量。**所以 A100 上 5D 全开放不下 32K**，粗估上限从 37K 降到约 31K。
- **(A)、(C)：** 每层同样多 268 MB，32K 的余量也只有几 GB，同样要降长度。
- **(B)、(D)：** cp 4 · tp 2，每层多 134 MB，影响小一些。
- **对策（三选一）：**
    - 推荐长度降到 24K–28K；
    - 改用保守档 D = 2048，多留约 16 GB；
    - 写 Triton 版的 bounded 门控核（见 9.7）。
- 不论选哪种，都先在 8K / 16K 跑一个 step，记下 `max_memory_allocated`，再往上推（同 §7.1）。

### 9.4 硬件差异和对通信瓶颈的影响

| 项 | H100 | A100 SXM4 80GB | H100 / A100 |
|---|---|---|---|
| BF16 稠密峰值 | 989 TFLOPS | 312 TFLOPS | 3.2× |
| HBM 带宽 | 3.35 TB/s | 2.04 TB/s | 1.6× |
| NVLink（单向 → × 0.80 有效） | 450 → 360 GB/s | 300 → 240 GB/s | 1.5× |
| 网卡（每卡一张 → × 0.85 有效） | NDR 400G：50 → 42.5 GB/s | 常见 HDR 200G：25 → 21 GB/s | 2× |
| 节点内 / 跨节点带宽比 | 8.5× | 11.3× | — |
| NVLS（NVSwitch 上做规约） | 有 | 无 | — |
| FP8 | 有 | 无 | 本文本来就用 BF16，不受影响 |

以上都是规格近似值。

读法：

- 从 H100 换到 A100，各项慢的倍数不同：
    - 计算慢 1.6–3.2 倍（GEMM 慢 3.2 倍，访存型算子慢 1.6 倍）；本模型的访存型算子很重（§7.3），整步更接近低的一端；
    - NVLink 只慢 1.5 倍；
    - 网卡慢 2 倍。
    - 所以暴露通信的占比整体下降，节点内通信降得更多。
- 粗估（暴露占比 = E·r_comm / (C·r_comp + E·r_comm)，r_comp 取 1.6–3.2）：
    - 推荐放置（通信全在 NVLink）：5–8% → 约 2–7%；
    - EP 跨节点（NCCL）：22–28% → 约 15–30%。
- 结果是两种观察场景不同：
    - 在 A100 上，推荐放置下通信不是瓶颈，这一点比在 H100 上更明显；瓶颈集中在 PP bubble 和访存型算子。
    - 坏放置的代价（E3 前后的差）还在。跨节点和节点内的带宽差距更大（11× 对 8.5×），但其中一部分被更慢的计算盖住了。
    - 要在 A100 上看到通信瓶颈，用坏放置或者跨节点的那一维做对照。
- MFU：torchtitan 的峰值表里有 A100（312 TFLOPS），日志里的 mfu 可以直接用。相对 312 的 MFU 会比 §0.2 里 H100 的那几列高：分母小了 3.2 倍，step 只慢 1.6–3.2 倍。

### 9.5 E2–E4 在 A100 上

- E2 节点内 8 卡 all_reduce 的 busbw ≈ 230–240 GB/s（300 × 0.80）。没有 NVLS，只有 Ring / Tree 两种算法。
- E2 跨节点单网卡 sendrecv ≈ 21 GB/s（HDR 200G）。16 卡 alltoall_perf ≈ 37 GB/s，按 §5.1 的 75 GB/s 把网卡带宽减半粗估。
- E4 在 A100 上做不了：DeepEP v1.2.1 的 internode 内核要 Hopper，README 的 Roadmap 里 A100 只列了「intranode only」，而且还是计划项。A100 上的 EP 只能拿 NCCL all-to-all 在不同放置下对照，E4 留给 H100。
- MoonEP 也做不了：它要 NVSwitch multicast，A100 的 NVSwitch 没有这个功能。

### 9.6 租前检查（在 tasks/03 档 2 的清单上再加几条）

- 必须是 A100 SXM4 80GB（HGX，带 NVSwitch）：
    - 40GB 版放不下本文任何一档；
    - PCIe 版没有 NVSwitch，TP / EP 放在节点内的前提就不成立。
- 问清网卡配置。常见的是每卡一张 HDR 200G；如果每个节点只有 1–2 张，或者只有以太网，跨节点的预期还要按比例往下降。
- 开机后跑 `nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv`，应该看到 `A100-SXM4-80GB` 和 `8.0`。

### 9.7 可选：Triton 版的 bounded 门控核

- 现状：attn_gym 的 softplus 门控已经有可移植的 Triton 核（`linear/_delta_rule/triton/softplus_gate.py`），bounded 门控却只有 CuTeDSL 核。
- 做法：照 softplus 的写法写一个 bounded 版，自定义 autograd，只存 bf16 输入，反向时重算。
- 收益：A100 / 4070 Ti 上不用再改成 reference，9.3 多出的显存也省回来。
- 本机就能写和测：在 4070 Ti 上对照 `impl="reference"` 的前反向数值。
- 写好后可以给 attn_gym 提一个 PR。

---

## 附录 A：每个组合的全部候选度数（NCCL 后端；每行按主推荐的同一规则挑 CP 算法 / 调度 / FSDP 模式）

**H100（D=2304）**

| 组合 | pp·dp_shard·cp·tp·ep | 变体 | 上限 20% | 推荐 | MFU @推荐 | MFU @8K | MFU @32K |
|---|---|---|---|---|---|---|---|
| full5d | 2·2·2·2·2 | AG-KV / I1F1B / keep | 18K | 16K | 14.5% | 13.6% | — |
| full5d | 2·2·2·2·4 | AG-KV / 1F1B / keep | 33K | 32K | 14.6% | 13.6% | 14.6% |
| full5d（选中） | 2·2·2·2·8 | AG-KV / 1F1B / keep | 37K | 32K | 14.6% | 13.7% | 14.6% |
| A | 2·4·1·2·2 | AG-KV / I1F1B / keep | 9K | 8K | 14.1% | 14.1% | — |
| A | 2·4·1·2·4 | AG-KV / 1F1B / keep | 16K | 16K | 13.8% | 13.7% | — |
| A | 2·4·1·2·8 | AG-KV / 1F1B / keep | 19K | 8K | 13.8% | 13.8% | — |
| A | 2·2·1·4·4 | AG-KV / 1F1B / keep | 30K | 16K | 14.0% | 13.5% | 14.5%（超余量） |
| A（选中） | 2·2·1·4·8 | AG-KV / 1F1B / keep | 34K | 32K | 14.4% | 13.6% | 14.4% |
| A | 4·2·1·2·2 | AG-KV / I1F1B / keep | 12K | 8K | 13.9% | 13.9% | — |
| A | 4·2·1·2·4 | AG-KV / I1F1B / keep | 13K | 8K | 13.9% | 13.9% | — |
| B | 1·4·2·2·2 | AG-KV / — / reshard | 46K | 32K | 15.2% | 8.9% | 15.2% |
| B | 1·4·2·2·4 | AG-KV / — / reshard | 47K | 32K | 15.3% | 9.5% | 15.3% |
| B | 1·4·2·2·8 | AG-KV / — / reshard | 47K | 32K | 15.8% | 10.5% | 15.8% |
| B | 1·4·2·2·16 | AG-KV / — / reshard | 47K | 32K | 13.8% | 11.1% | 13.8% |
| B | 1·2·4·2·2 | AG-KV / — / reshard | 89K | 64K | 17.2% | 5.9% | 13.1% |
| B | 1·2·4·2·4 | AG-KV / — / reshard | 91K | 64K | 17.3% | 6.5% | 13.4% |
| B（选中） | 1·2·4·2·8 | AG-KV / — / reshard | 92K | 64K | 17.8% | 7.9% | 14.2% |
| B | 1·2·4·2·16 | AG-KV / — / reshard | 92K | 64K | 15.8% | 9.9% | 13.3% |
| B | 1·2·2·4·4 | AG-KV / — / reshard | 81K | 64K | 17.2% | 6.5% | 13.4% |
| B | 1·2·2·4·8 | AG-KV / — / reshard | 82K | 64K | 17.7% | 7.9% | 14.2% |
| B | 1·2·2·4·16 | AG-KV / — / reshard | 82K | 64K | 15.8% | 9.9% | 13.3% |
| C | 2·4·2·1·2 | AG-KV / I1F1B / keep | 9K | 8K | 14.1% | 14.1% | — |
| C | 2·4·2·1·4 | AG-KV / 1F1B / keep | 17K | 16K | 13.7% | 13.7% | — |
| C | 2·4·2·1·8 | Ulysses / 1F1B / keep | 20K | 16K | 13.7% | 13.7% | — |
| C | 2·2·4·1·2 | AG-KV / I1F1B / keep | 17K | 16K | 14.4% | 13.4% | — |
| C | 2·2·4·1·4 | AG-KV / 1F1B / keep | 33K | 32K | 14.5% | 13.4% | 14.5% |
| C（选中） | 2·2·4·1·8 | AG-KV / 1F1B / keep | 38K | 32K | 14.5% | 13.5% | 14.5% |
| C | 4·2·2·1·2 | AG-KV / I1F1B / keep | 13K | 8K | 13.8% | 13.8% | — |
| C | 4·2·2·1·4 | AG-KV / I1F1B / keep | 14K | 8K | 13.8% | 13.8% | — |
| D | 2·1·4·2·2 | AG-KV / I1F1B / keep | 35K | 32K | 15.3% | 12.4% | 15.3% |
| D | 2·1·4·2·4 | AG-KV / 1F1B / keep | 64K | 64K | 16.0% | 12.8% | 14.9% |
| D（选中） | 2·1·4·2·8 | AG-KV / 1F1B / keep | 73K | 64K | 16.0% | 13.1% | 15.0% |
| D | 2·1·2·4·4 | Ulysses / 1F1B / keep | 61K | 32K | 14.9% | 12.9% | 14.9% |
| D | 2·1·2·4·8 | AG-KV / 1F1B / keep | 68K | 64K | 16.0% | 13.2% | 14.9% |
| D | 4·1·2·2·2 | AG-KV / 1F1B / keep | 39K | 32K | 13.6% | 13.2% | 13.6% |
| D | 4·1·2·2·4 | AG-KV / 1F1B / keep | 43K | 16K | 13.6% | 13.4% | 13.6% |

**H200（D=3584）**

| 组合 | pp·dp_shard·cp·tp·ep | 变体 | 上限 20% | 推荐 | MFU @推荐 | MFU @8K | MFU @32K |
|---|---|---|---|---|---|---|---|
| full5d | 2·2·2·2·2 | AG-KV / 1F1B / keep | 14K | 8K | 19.8% | 19.8% | — |
| full5d | 2·2·2·2·4 | AG-KV / I1F1B / keep | 21K | 16K | 21.8% | 21.1% | — |
| full5d（选中） | 2·2·2·2·8 | AG-KV / 1F1B / keep | 36K | 16K | 21.3% | 21.1% | 21.0% |
| A | 2·4·1·2·2 | AG-KV / 1F1B / keep | 7K | 4K | 19.9% | 20.6%（超余量） | — |
| A | 2·4·1·2·4 | AG-KV / I1F1B / keep | 11K | 8K | 21.8% | 21.8% | — |
| A | 2·4·1·2·8 | AG-KV / I1F1B / keep | 13K | 8K | 22.0% | 22.0% | — |
| A | 2·2·1·4·4 | AG-KV / I1F1B / keep | 20K | 16K | 21.5% | 20.9% | — |
| A（选中） | 2·2·1·4·8 | AG-KV / 1F1B / keep | 33K | 16K | 21.1% | 20.9% | 20.7% |
| A | 4·2·1·2·2 | AG-KV / I1F1B / keep | 11K | 8K | 20.6% | 20.6% | — |
| A | 4·2·1·2·4 | AG-KV / I1F1B / keep | 14K | 8K | 20.9% | 20.9% | — |
| B | 1·4·2·2·2 | AG-KV / — / reshard | 48K | 32K | 19.7% | 10.3% | 19.7% |
| B | 1·4·2·2·4 | AG-KV / — / reshard | 50K | 32K | 20.2% | 11.4% | 20.2% |
| B | 1·4·2·2·8 | AG-KV / — / reshard | 50K | 32K | 21.2% | 13.5% | 21.2% |
| B | 1·4·2·2·16 | AG-KV / — / reshard | 51K | 32K | 19.4% | 16.2% | 19.4% |
| B | 1·2·4·2·2 | AG-KV / — / reshard | 94K | 64K | 21.1% | 6.1% | 16.0% |
| B | 1·2·4·2·4 | AG-KV / — / reshard | 96K | 64K | 21.6% | 6.9% | 16.8% |
| B（选中） | 1·2·4·2·8 | AG-KV / — / keep | 48K | 32K | 23.7% | 20.5% | 23.7% |
| B | 1·2·4·2·16 | Ulysses / — / reshard | 102K | 64K | 21.1% | 14.2% | 18.8% |
| B | 1·2·2·4·4 | AG-KV / — / reshard | 86K | 64K | 21.4% | 6.9% | 16.7% |
| B | 1·2·2·4·8 | AG-KV / — / keep | 51K | 32K | 23.5% | 20.6% | 23.5% |
| B | 1·2·2·4·16 | AG-KV / — / reshard | 87K | 64K | 20.8% | 14.2% | 18.5% |
| C | 2·4·2·1·2 | AG-KV / 1F1B / keep | 5K | 4K | 19.7% | 20.5%（超余量） | — |
| C | 2·4·2·1·4 | AG-KV / I1F1B / keep | 10K | 8K | 21.7% | 21.7% | — |
| C | 2·4·2·1·8 | AG-KV / I1F1B / keep | 13K | 8K | 21.8% | 21.8% | — |
| C | 2·2·4·1·2 | Ulysses / 1F1B / keep | 11K | 8K | 19.8% | 19.8% | — |
| C | 2·2·4·1·4 | AG-KV / I1F1B / keep | 20K | 16K | 21.6% | 20.6% | — |
| C（选中） | 2·2·4·1·8 | AG-KV / 1F1B / keep | 35K | 16K | 21.0% | 20.6% | 20.8% |
| C | 4·2·2·1·2 | AG-KV / I1F1B / keep | 12K | 8K | 20.6% | 20.6% | — |
| C | 4·2·2·1·4 | AG-KV / I1F1B / keep | 15K | 8K | 20.9% | 20.9% | — |
| D | 2·1·4·2·2 | Ulysses / 1F1B / keep | 29K | 16K | 20.1% | 18.0% | 21.3%（超余量） |
| D | 2·1·4·2·4 | AG-KV / I1F1B / keep | 42K | 32K | 22.0% | 19.4% | 22.0% |
| D（选中） | 2·1·4·2·8 | Ulysses / 1F1B / keep | 72K | 64K | 21.9% | 20.3% | 21.9% |
| D | 2·1·2·4·4 | AG-KV / I1F1B / keep | 40K | 32K | 21.9% | 19.5% | 21.9% |
| D | 2·1·2·4·8 | AG-KV / 1F1B / keep | 67K | 64K | 21.5% | 20.2% | 21.4% |
| D | 4·1·2·2·2 | AG-KV / 1F1B / keep | 34K | 16K | 19.7% | 18.9% | 19.5% |
| D | 4·1·2·2·4 | AG-KV / 1F1B / keep | 44K | 16K | 20.1% | 19.7% | 19.6% |

（「keep」在有 PP 的行里是页面的固定行为：有 PP 时 FSDP 不 reshard；「超余量」= 该长度下页面能算但超过碎片 20% 的判据。）