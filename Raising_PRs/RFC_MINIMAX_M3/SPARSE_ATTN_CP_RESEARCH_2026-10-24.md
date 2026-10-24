# 稀疏 attention 的 CP：调研（2026-10-24）

起因：M3 的 5D 分支按规则跳过了 CP；titan main 上的 DeepSeek V4 也拒绝 CP。问题是这算不算 titan 的核心缺失。

基线：titan main `3ed5aa97`（10-24），attn_gym 0.0.17（PyPI 最新，10-13）。

## 结论

算核心缺失。缺的是"带 indexer 的稀疏 attention"这一族的 CP 后端；titan 的 CP 框架本身不缺抽象，PyTorch 也不缺东西。

- titan 的 CP 框架（`CPInnerAttention` + `prepare_cp_metadata` + `ContextParallelTransform`）装得下这一族。线性 attention 的 CP 就是这样补上的：attn_gym 先做 kernel 和路由（#472、#473、#576），titan 再在 common 层加 `cp_kda.py`、`cp_gdn.py`（#4639、#5184）。
- 稀疏这一族现在三层都是空的：kernel 接口、titan common 层的通信原语、模型层的接缝。
- 受影响的模型：DSV4（main，已合入）、M3（本地分支）、GLM-5（#5578，开放中）。

## titan main 现状

- DSV4：`torchtitan/models/deepseek_v4/model.py:258-261` 在 cp>1 时抛 `NotImplementedError`，从 #3634（09-03 合入）起一直这样。按标题和正文搜 issue/PR，没有人在做 DSV4 的 CP。
- M3：`wt_m3_5d/torchtitan/models/minimax_m3/model.py:181-183` 同样拒绝。
- GLM-5 #5578（外部贡献者，10-21 开，还没有 review）声称 FSDP2+TP2+CP2+EP2 跑通。做法：
  - indexer 冻结，没有 KL loss。
  - `preprocess_inputs` 先在每个 rank 上建全局 `[1,T,T]` 稠密 additive mask，再切 Q 维。
  - indexer 用 fp32 算稠密 `[N,Q,K]` 分数。
  - 从 top-k 建 token 级 BlockMask，走 `AllGatherKVFlexInnerAttention`，gather 的是展开后的 K/V。
  - 结论：debug 长度能跑，长上下文用不了。T=128K 时光这个 mask 就至少 32 GiB。
- CP 框架的形状：
  - 接缝在 InnerAttention。`ContextParallelTransform` 只替换 decoder 层里的 `InnerAttention.Config`（`config/transform/context_parallel.py:93-105`），MLA 已经要特判 `wkv_b`（:69-87）。
  - metadata 每个 batch 在 forward 之前按 positions 建一次；`Decoder` 生成一个 permutation，再对每种 CP inner attention 调 `prepare_cp_metadata`（`models/common/decoder.py:362-381`）。稀疏 attention 的选择是每层在 forward 里由 indexer 从激活算出来的，只能在 forward 里处理。
  - 已有后端：
    - AllGatherKV flex（含 SWA，以及 MLA 压缩 latent）。
    - Ulysses（flex、SWA、varlen、MLA）。
    - 线性 attention 状态路由（KDA、GDN），conv history 的 halo 在 attn_gym 里。
  - 进行中：varlen all-gather KV（#5071，fegin），KDA/GDN 的 Ulysses（#5590，Jessica Zhong）。
  - 现有后端都只吃 batch 级 metadata，接不了"每层、每个 query 的全局选择索引"。

## 缺口

### 1. kernel（attn_gym，DSV4 用的 kernel）

- `lightning_indexer`：硬阻塞。
  - docstring 原话："This API processes whole documents, not query chunks with cached history."
  - Q 和候选共用一套 `cu_seqlens`。因果上界 `(t+1)//ratio` 用的是行号，见 CuTe 的 `visible_candidates`（`indexer/impl/cute_kernel_base.py:27-31`）。
  - CP 下，本地 query 在文档里的位置不等于行号，需要独立的 Q/K offsets 加 per-segment 的 q 因果偏移。绕开的办法只有把文档前缀整段补成假 query，那等于没切。
  - 内部其实已经有 per-query `candidate_bounds[T,2]`（`indexer/impl/cute.py:224-227`），只是约定写死成"完整文档"，所以接口改动面不大。但 CuTe 和 Triton 两个后端都得改。
  - Megatron 的同类 kernel 已经是这个形状：`indexer_topk(..., cu_seqlens_q, cu_seqlens_kv, q_causal_offsets)`，见 `csa_utils/cp_utils.py` 的 `build_cp_indexer_layout`。
- `gather_attn`：软阻塞。
  - 稀疏分支本来就是"索引进任意 pool"，直接可用于 all-gather 后的全局压缩池。
  - SWA 分支按位置取窗口，`local_kv` 长度必须等于 query 长度（`gather_attn/api.py` 的 `_validate_inputs`），所以放不进左侧 halo。
  - 绕法：在前面补 window-1 行假 query（开销约 W/T_local），或者把窗口改写成显式索引。正式方案是加一个 history 前缀参数。
- attn_gym 仓库里的 CP 工作全是线性 attention（drisspg、acisseJZhong）。`attn_gym/sparse` 下没有任何 context_parallel。

### 2. titan common 层

这些应放在 `torchtitan/distributed/context_parallel.py` 或 `models/common/attention/`。

- 左边界 halo 交换：
  - 反向时把梯度送回原 rank。
  - 按 fragment 规划，兼容 head-tail。
  - 现在只有 attn_gym 线性 attention 路由里的 conv history 能做这件事。
- 变长压缩池的 all-gather 加 select-and-pad：
  - 压缩按文档 floor，所以每个 rank 的条数不同，要 pad 到固定容量。
  - 记录 seq-major 到 rank-major 的行映射。
  - 反向做 reduce-scatter。
  - 现有 `cp_all_gather_tensors` 只做等长、按 rank 顺序的 gather。
- 稀疏族的 CP metadata：本地 Q 段、全局文档表、q 因果偏移、halo 计划。形态上对应线性 attention 的 `ContextParallelLinearAttentionMetadata.cp_routing`。

### 3. 模型层

- DSV4：
  - compressor 和 indexer 在 `DeepSeekV4Attention.forward` 里、InnerAttention 之前就算完了（`attention.py:388-397`），而 CP 的接缝在 InnerAttention。两种改法：把压缩和打分挪进 inner attention（线性 attention 就是把 conv 放进 InnerKDA）；或者让外层模块自己感知 CP（GLM-5 的做法）。
  - 压缩组可能跨 rank，要定归属。CSA 带 overlap，需要 8 行 halo；HCA 需要 128 行。
  - HCA 的可见规则要用全局的文档内位置。
  - indexer KL loss 的分母要在 CP 上求全局。
- M3：
  - indexer 是纯 torch（`select_msa_blocks`），attention 走 flex，所以不需要改 kernel。
  - 要做的事：all-gather index key（单头）和 K/V；在 forward 里建"本地 Q × 全局 KV"的 BlockMask，因为选择是每层算的，不能走 `prepare_cp_metadata`；处理 permutation。
  - 没有压缩也没有 SWA，不需要 halo。M3 是这一族里最容易做的。

## 别人的做法

各家的做法一致：Q 留在本地，all-gather 压缩 KV 和 indexer K，左边界做 halo，然后恢复全局顺序。

- DeepSeek V4 报告 §3.4.3，"two-stage contextual parallelism"：
  - 按连续块切分（从 rank i 往 i+1 的边界交换可以看出）。
  - 第一阶段：rank i 把最后 m 个未压缩 KV 发给 rank i+1，后者连同自己的 s 个一起压缩，得到固定 s/m+1 条（含 padding）。
  - 第二阶段：all-gather 各 rank 压缩后的 KV，再用 fused select-and-pad 拼成全长 cp_size×s/m，padding 放在尾部。
  - HCA 和 CSA indexer 的可见范围按规则预先算好；CSA 的稀疏 attention 用 top-k 给出的显式索引。
- Megatron-LM #7469（10-09 合入 main，17 个文件，+3460/-230）：
  - DSv4 packed THD 的 CP，`attention_cp_layout="contiguous"`，只支持 TP=1。
  - 覆盖 SWA、CSA（4）、HCA（128）和 MTP。
  - 交换的是边界 hidden，`d_window = max(window, 8 或 128)`；还做了压缩组归属、异步 gather 压缩 KV 和 indexer K、梯度 reduce-scatter。
  - 后续：#7921 做确定性；#8018（开放中）扩到 V4.1 的 CSA2。
- Megatron Lite M3（#7621，dev 分支，开放中）：
  - flex 后端在 zigzag 切分上做 all-gather CP。
  - MagiAttention 后端由 Magi 负责 CP 分发和 index-K 通信。
  - indexer 冻结。
- Megatron #7614（开放中）：CuTe 简化稀疏 attention 的 THD+CP。Q 留本地，main K/V 和 indexer K 做 all-gather，zigzag 后恢复全局顺序。
- SGLang prefill CP：
  - DSA 用 interleave（round-robin）切分，zigzag 不可用。
  - 本地 query 跑 indexer，gather indexer key 和 MLA KV。

## 通信量估算

参数取 DSV4-Flash：64 头 × 512，window 128，CSA 4 / HCA 128，indexer 64 头 × 128，topk 512。

- all-gather 方案：
  - CSA 层每 token 要 gather (512+128)/4 = 160 个值，HCA 层 4 个。
  - halo 每个边界 128 行 hidden（4096 维）。
  - 对照：DSV3 的 MLA all-gather 是每 token 576 个值。
- Ulysses 方案：
  - q 和 o 都要搬，每 token 2×64×512 ≈ 6.5 万个值。
  - indexer 的分数要对 64 个 index 头求和，如果按头切，就得 all-reduce `[T, T/4]` 的分数。
  - 所以现有 Ulysses 后端对这一族没用。

## 如果要做

1. 先私信维护者对齐：
   - Driss（drisspg）：管 attn_gym，也是 titan 里把 DSV4 接到 gather_attn 的人（#4855、#4858、#4943）。
   - Jessica Zhong（acisseJZhong）：titan CP 的主要作者，load balancer、metadata 归属、KDA/GDN CP 都是她做的。
   - fegin（Chien-Chin Huang）：Ulysses 和 varlen all-gather（#5071）。
2. attn_gym：`lightning_indexer` 加 query 分段模式（独立的 Q/K offsets 加 q 因果偏移）；`gather_attn` 加 SWA 的 history 前缀。
3. titan common：halo 交换、压缩池 gather/select-and-pad、稀疏族的 CP metadata 和后端。
4. 模型：先做 M3（不依赖 kernel 改动），再做 DSV4。

## 来源

- DeepSeek-V4 报告：https://arxiv.org/html/2606.19348 （§3.4.3）
- Megatron-LM #7469：https://github.com/NVIDIA/Megatron-LM/pull/7469 ；`cp_utils.py`：https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/experimental_attention_variant/csa_utils/cp_utils.py
- Megatron-LM #7621（M3）：https://github.com/NVIDIA/Megatron-LM/pull/7621 ；#7614：https://github.com/NVIDIA/Megatron-LM/pull/7614
- SGLang prefill CP：https://lmsysorg.mintlify.app/docs/advanced_features/prefill_cp.md
- titan #5578（GLM-5）：https://github.com/pytorch/torchtitan/pull/5578 ；#5071：https://github.com/pytorch/torchtitan/pull/5071 ；#5590：https://github.com/pytorch/torchtitan/pull/5590
