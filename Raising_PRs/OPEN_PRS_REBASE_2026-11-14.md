# 全部 open PR rebase 到 main `631d544b1`，以及 Summary (human-written) 要点（2026-11-14，CPU 会话）

用户："拉取 #4272，里面我有大量的 tracking PRs open，可能都缺失了 Summary (Human Written)，帮我把我所有的 open PR 都 rebase 到最新的 main，并且整理每一个的 summary 你的 draft/suggestion，如果需要 GPU verify，则 rebase 完成后在 logbook 记录，我后面再去验证"。随后："不需要评估栈分支，那个不是 PR，只看 open PR"。

## 背景

- 上游 10-29 合入 #5694：PR 模板顶部加 `## Summary (human-written)`，CONTRIBUTING 新增 "AI-assisted contributions" 一节。summary 必须本人写（2-5 句：问题、为什么这样做、自己查过什么），空的或 AI 写的 PR 可能被关掉或退回 draft。
- 模板之后的各节是 `## Why`、`## What changed`、`## Numerics / performance`、`## Testing`，这些可以由 AI 起草，但每一句都由作者负责。
- 上游 `.claude/skills/pr-description/SKILL.md` 明确写了："Never write the Summary (human-written) section ... Don't draft text for them to paste."
- 所以本文件给每个 PR 列的是**中文要点**（问题、做法和理由、你已经查过的、rebase 后还算不算数），不是可以直接粘贴的英文。英文的 2-5 句请自己写。
- 11 个 open PR（GitHub search，作者 QIU023，11-14）的现有 body 都只有旧格式的 `## Summary`，没有 `Summary (human-written)`。旧 `## Summary` 建议改名为 `## What changed`，免得同一个 body 里出现两个 summary。
- 2026-10-14 的规则是不写 `Relation to ...` 一节，但 #4381、#4656、#4963 的 body 里还有，同步时顺手删掉。

## 做法

- 目标 base：upstream main `631d544b1`（11-14 fetch）。
- 试 rebase 时 11 个 PR 全部在第一个提交就冲突。main 在这期间合入了几次重构：
  - #6084：多模态模型拆成 text `Decoder` 和 `MultimodalModel` 两个类；
  - #5873：`VisionProcessor`；
  - #5802 到 #5807：spmd 的集合通信显式化，去掉隐式的 `ShardingConfig` 重分布；
  - #4582：loss 重设计；
  - #5776：`PrecisionConfig`；
  - #5939：attention residual 改成 mul+sum。
- 按依赖分 7 组并行处理。每组一个 detached worktree（`C:/Users/78532/AppData/Local/Temp/claude/rb*`），不建分支。提交的 committer 是 QIU023，不带 trailer。
- 每组都在本机做了：`git diff --check`、冲突标记检查、`py_compile`、`black --check`（22.12）、`range-diff` 和原补丁逐项对照、新增注释行检查。
- 本机 import 不了 `models/common`（CuTeDSL），所以**单测和数值都没在本机跑**，都列进下面的 GPU 待验证。
- 改写前用的源：
  - #4656 用 review 分支 `attnres_review1` 的 `507a2f7d5`。线上 PR 分支 `k3_ac_reuse_attention` 的 `7dea4cbaf` 是 #5033 之前的版本。
  - #4381 用 `dep_review1` 的 `caa8726fa`，比 PR 头多 `grad_dtype` 修复。
  - #5619 用 `minimax_m3_5d_review1` 的 `70b6efa63`，5D 叠在 amend 后的 M3 上。
  - 其余用 PR 分支本身。

## 总览

| PR | 新 head | 提交 | main 已覆盖、因此丢掉的 | 会改行为的适配 |
| --- | --- | --- | --- | --- |
| #4380 dynamic CP | `40a38f456` | 11/11 | 无 | 开 dynamic CP 时跳过 `_prepare_multimodal_embeds` 里的 CP I→R（#5803），切图路径加 TP gather |
| #4381 DEP | `56e5e196c` | 12/12 | 无 | `vision_embeds` 接口放在 TP/CP I→R 之前，特征梯度改在 stage 0 的反向里归约 |
| #4656 列表载体 | `809fffc42` | 1/1 | `sharding.py` 那部分（#5806） | 无；但编译区的 lowering 变了，B200 golden 可能要刷新 |
| #4963 block 只存一份 | `383b58d9f` | 2/2 | 无 | 无 |
| #4765 offload | `c9581d655` | 1（用 review 分支内容） | 重算提交（已过时） | 无 |
| #4764 balance | `a336356dd` | 2（用 review 分支内容） | 同上 | 无 |
| #4751 MoonEP | `945f114e2` | 10/10 | 无 | `gate_up` 融合 activation（#5086），多一次 `[R, 2F]` 拷贝；调用方要传 `num_max_tokens_per_rank` |
| #4576 LoRA | `575530d83` | 4/5 | 适配器放置（#5884） | 无 |
| #5469 视频 | `06836222a` | 5/5 | 无（main 只给了接口） | K3 flavors 加 `video_resize` |
| #5635 M3 | `b809f1eee` | 5/5 | 无 | 按 #6084 拆类；FLOPs 开始算 vision tower |
| #5619 M3 5D | `796550353` | 2/3 | post-scatter 配置那个提交（#5806） | 集合通信显式化 |

### 要你定的（细节在各 PR 一节）

1. 推送（下一节的命令），以及 #4576、#5469 要不要新开 review 分支。
2. #5469：K3 flavors 加 `video_resize` 这件事。
3. #4380：接线挂 `_parallelize` 还是 override `parallelize`；第 11 个提交的 message 要不要补一句。
4. #4381：适配写进改过 message 的提交还是单开；测试目录要不要挪。
5. #4751：Kimi K3 上 `shared_experts_stream` 其实不起作用（rebase 之前就是这样）；`torch.cuda` 和 `torch.accelerator`；core 里的私有 helper。
6. #4576：公开的 `trainable_state_dict` 和 RL 的私有 `_trainable_model_state_dict` 重叠。
7. #4656 和 #4963：`_wait_async_collective_inputs` 不遍历容器（潜在风险，现在不触发）。#4765：要不要改到 main 的 `saved_tensors_hooks` 接口上。
8. #5635：FLOPs 算上 vision tower；新的 kwargs 常量和 pragma；新加的 integration 格子。

### GPU 机上的顺序

- 环境先按 `631d544b1` 的 `pyproject.toml` 装固定依赖：spmd_types、torch_remat、attn-gym。本机的 spmd_types 0.2.5 没有 `rules`，已经跟不上 main。
- 每个 head 先跑它那一节列的单测，过了再做数值。
- 数值按 CLAUDE.md 的规则做：
  - 每个对比只用一份 inductor / triton 缓存，每种配置先跑约 10 步预热；
  - 第 1 步逐位相同是正确性标准；
  - 每张表带一行噪声底；
  - 只报 debug 集被记住之前的步数（第 1 / 10 / 20 步）。
- 有格子对不上时先定位，结果出来之前不写进任何 body。
- 建议的顺序：
  1. 两个会改行为的视觉 PR：#4380 的 2 卡梯度测试，#4381 的 DEP 开关两格；
  2. #4656：B200 golden；
  3. #4751：跨步 `_grouped_mm` 第一次实跑；
  4. 其余。

## 上游已合入或 open 的改动，哪些让我们的 PR 过期（11-14 查）

方法：

- 在 main 的 `d6810e4d6..631d544b1` 里按关键词 grep 提交；
- GitHub search 查 open PR，关键词 minimax、lora、video、vision encoder、ViT、offload、attention residual、expert load balancing、dist_moe、frozen、kimi_k3；
- 重叠的那些，读 body 和改动的文件。

| 我们的 PR | 上游 | 状态 | 结论 |
| --- | --- | --- | --- |
| #4765 PP offload | #6020 anijain2305 "Activation offloading on torch_remat saved-tensor hooks"（+1671，ready）；#6022 同一作者 "Schedule-driven activation offload for pipeline stages"（draft） | open | **基本被取代。** 两者都按 block 装 `torch_remat.saved_tensors_hooks`，remat 开关都能用，D2H 走 side stream；#6022 让 PP 调度决定何时 offload 和 reload。通用、不限模型，#4765 只是 K3 stage 范围的同一件事。main 的 #5090 早就有按 block 的 saved-tensor hook 接口 |
| #4764 PP balance | 同上 | open | **底层被取代。** 剩下独有的只有：按 profile 规划 offload，以及经 Mooncake 停到较轻的 PP rank 上。Mooncake 不是 PyTorch 库，违反上游 `.claude/CLAUDE.md` 原则 3（核心训练基础设施不依赖非 PyTorch 库），原样很难进 |
| #4576 LoRA | 已合入：#5884（适配器放置）、#6057（有冻结参数时 HF 导出直接 raise，带 PEFT 导出的 TODO）、#5844（RL 只同步 adapter，私有方法）、#6003、#5893。open：#6058 fegin "Leave frozen parameters out of training checkpoints"（draft）、#5738 HJSang shared-outer 的 expert adapter 布局 | 部分合入 | **5 个提交里 1 个被取代**；"main 上建不出 K3" 这个前提大概率不成立了。导出这块：维护者选了先 raise、以后做 PEFT 格式导出，#6058 让 checkpoint 只存可训练的部分；PR 的 merged 导出要和 fegin 的方向对齐。#5738 合入后 merge 要支持 shared-outer 布局。**冻结 PP rank 照样 step 的那部分，没有人做，是最站得住的** |
| #4751 MoonEP | open：shuhuayu 的 Dist-MoE 跑 K3 系列（#5684 SiTU-GLU、#6049 PP stage、#6050 planner overflow，都是 draft）；#5517 felipemello1 推理时 shared experts 走 side stream（draft） | open | **功能上没被取代**（Dist-MoE 的 README 写明 "does not truncate or rebalance routing"），**方向上被挤**：维护者正把 Dist-MoE 做成 K3 的后端，而它和 `TokenDispatcherTransform` 互斥。`shared_experts_stream` 一项和 #5517 部分重叠，而且在 K3 上本来就不起作用 |
| #4963 | #6049（shuhuayu draft）给 K3 的 PP stage 加静态 metadata，按 `(hidden, delta_TND [T, n_i, D])` 的载体写 | open | 没被取代，**直接冲突**：#4963 把载体改成每个 block 一个张量。#6049 大概率先合，之后 #4963 的静态 metadata 要改成按 block 列 |
| #4381 DEP | #6049 改同一个 pipeline builder 和 stage | open | 没被取代。DEP 多出来的 `vision_embeds` 输入要进 #6049 的静态 metadata |
| #4656 | 已合入：#5033（编译区）、#5939（mul+sum）。open：#5187 elfiegg 降低反向显存（draft）、#5156 和 #5913 felipemello1（draft） | 有合入，有 open | 没被取代（main 仍然每打开一个 block 就 cat 一次 stack），但这个区域维护者正在持续优化，冲突多，收益要对 mul+sum 版的 main 重测 |
| #4380 dynamic CP | 已合入 #5266（vision encoder 在 CP 上不切，只在 dp_shard 上切）；open #5993、#5992 pianpwk（按 `vision_encoder_cost` 把重的 micro-batch 排在前面，跨 DP 平衡，draft） | 有合入，有 open | 没被取代。#5266 是维护者选的另一条路（不切），#5992 和 #5993 是在 DP 层面做平衡，和切图是两件事 |
| #5469 视频 | #5873 只提供接口 | 已合入 | 没被取代，也没有别的 K3 视频 PR |
| #5635、#5619 M3 | 没有 | — | 没被取代。search "minimax" 只搜到这两个 |

## 推送（没推成，命令留给用户）

- 11-14 读过远端，7 个 review 分支和 11 个 PR 分支的 tip 都和上面的源相同。
- 推 backup 那一步被 Claude Code 的权限分类器拦下（Destructive），和 10-12 一样，没有绕开。**现在什么都没推。** 新提交只在本机的 worktree 里（`C:/Users/78532/AppData/Local/Temp/claude/rb*`，detached HEAD）。推之前别删这些 worktree：`rb4576` 和 `rb5469` 的提交没有别的引用。
- PR 分支一个都不动。按规则，你在 GPU 机上验过、点头之后，再同步 PR 分支。

### 一览

| PR | PR 分支（不动） | review 分支：旧 head 到新 head | 旧 head 由谁保住 |
| --- | --- | --- | --- |
| #4380 | `k3_cp_mm` `a2a97b88a` | `cpmm_review1`：`a2a97b88a` 到 `40a38f456` | PR 分支本身 |
| #4381 | `k3_pp_mm` `7529daeea` | `dep_review1`：`caa8726fa` 到 `56e5e196c` | 新建 `backup/dep_review1_pre_20261114` |
| #4656 | `k3_ac_reuse_attention` `7dea4cbaf` | `attnres_review1`：`507a2f7d5` 到 `809fffc42` | `pp_review_optimize`（是它的祖先） |
| #4963 | `pp_review_optimize` `62b32e5af` | 没有 review 分支；新 head `383b58d9f` | 不用（PR 分支不动） |
| #4765 | `k3_pp_offload` `2140058c5` | `pp_offload_review1`：`fb5d35b56` 到 `c9581d655` | `backup/pp_balance_review1_pre_20261114`（是它的祖先） |
| #4764 | `k3_pp_balance` `224bdbaf4` | `pp_balance_review1`：`908db8609` 到 `a336356dd` | 新建 `backup/pp_balance_review1_pre_20261114` |
| #4751 | `k3_moonep_seam` `16ff9da7f` | `moonep_review1`：`16ff9da7f` 到 `945f114e2` | PR 分支本身 |
| #4576 | `lora_review2` `676e443e0` | 没有 review 分支；新 head `575530d83` | 不用 |
| #5469 | `k3_video` `e117882e6` | 没有 review 分支；新 head `06836222a` | 不用 |
| #5635 | `minimax_m3` `8a2651476` | 没有 review 分支；新 head `b809f1eee` | 不用 |
| #5619 | `minimax_m3_5d` `43ccfadb5` | `minimax_m3_5d_review1`：`70b6efa63` 到 `796550353` | 新建 `backup/minimax_m3_5d_review1_pre_20261114` |

- 只要新建 3 个 backup 分支，别的旧 head 都有现成的分支保着。
- #4963 和 #5635 的新 head 分别是 `a336356dd` 和 `796550353` 的祖先，推了 review 分支就会跟着上远端，GPU 机可以按 sha checkout。
- #4576 和 #5469 没有 review 分支。CLAUDE.md 规定新开 review 分支要先问你，所以没开。

### 命令（在 `torchtitan/` 下跑）

```bash
# 1. 三个 backup
git push origin \
  caa8726fa2d526c69d325a33be9bf18192dcaaf3:refs/heads/backup/dep_review1_pre_20261114 \
  908db8609d7bbac1875f542ddeebc57271eb4e2c:refs/heads/backup/pp_balance_review1_pre_20261114 \
  70b6efa631257ad50dfa3470dee054a6d324041c:refs/heads/backup/minimax_m3_5d_review1_pre_20261114

# 2. 七个 review 分支，按读到的旧 head 做 lease
git push origin \
  --force-with-lease=cpmm_review1:a2a97b88a623aa43f2734c75587375a0a4412cb0 \
  --force-with-lease=dep_review1:caa8726fa2d526c69d325a33be9bf18192dcaaf3 \
  --force-with-lease=attnres_review1:507a2f7d559756dcb80547d0f5f8bcebb929599a \
  --force-with-lease=pp_offload_review1:fb5d35b5670df89767d2ad85e83de4576a156c24 \
  --force-with-lease=pp_balance_review1:908db8609d7bbac1875f542ddeebc57271eb4e2c \
  --force-with-lease=moonep_review1:16ff9da7f4a55915a590799414889d8201726bb7 \
  --force-with-lease=minimax_m3_5d_review1:70b6efa631257ad50dfa3470dee054a6d324041c \
  40a38f4568b748b77a7a15fc73bdb455549f4c30:refs/heads/cpmm_review1 \
  56e5e196cc9aa54824099a82792fac667ac001d5:refs/heads/dep_review1 \
  809fffc4200aeac20e306cc404852221f2878d87:refs/heads/attnres_review1 \
  c9581d6555bb90c7f27874b8a2cf7d4d62bc896a:refs/heads/pp_offload_review1 \
  a336356dde7e253eec71bc2610ef357c54969d2f:refs/heads/pp_balance_review1 \
  945f114e20efcd931caf88b6f7987338282b4dd2:refs/heads/moonep_review1 \
  796550353d4afdc51407b8caa6379cc8e169d488:refs/heads/minimax_m3_5d_review1

# 3. #4576、#5469 要先定放在哪里（新开 review 分支，比如 lora_review3、video_review1，要你点头）
#    #4576 的新 head: 575530d83dfdf9910c262b2ab7f90278fb763277
#    #5469 的新 head: 06836222aa1672b13647430cd3135d5159feecf5
```

### 验过之后同步 PR 分支（每个都要你点头）

- 同步的方式是 `git push --force-with-lease=<PR 分支>:<表里的旧 head> origin <新 head>:refs/heads/<PR 分支>`。
- 同步前，GitHub 上的 body 先补上 `## Summary (human-written)`：GitHub 合并时用的是 PR 描述，不是本地 commit message。这几个 PR 都不是 ghstack 或 stack-pr，推送不会覆盖 body。
- 叠着的 PR，同步顺序：#4656、#4963、#4765、#4764；#5635、#5619。#4963、#4764、#4765 的 body 写的是 "stack on #4656"，GitHub 上的 diff 会带着下层的提交。

## 逐个 PR

### #4576 `[Kimi K3][LoRA]` LoRA 导出、冻结 PP rank

- 源 `lora_review2` = `676e443e0`（PR 分支本身），rebase 后为 `575530d83`（worktree `rb4576`）。
- 提交对应：
  - `8393252ad` 对应 `93401a47b`；
  - `91212bd8b`（适配器跟随 base 的放置）**整条丢掉**，被 main 的 #5884（`6eb5cd783`，"Derive adapter sharding from the base weight"）完全覆盖，连同 4 个放置测试；
  - `b61c686dc` 对应 `b144648fb`；
  - `fb4079831` 对应 `d998692ac`，补丁相同；
  - `676e443e0` 对应 `575530d83`，补丁相同。
- 9 个文件，+450/-6（原来 +538/-19）。
- 适配，都是机械的：
  - `lora.py` 的 import 两边都留；
  - CPU `test_lora.py` 用 main 的文件，再接上 PR 的 6 个测试；
  - GPU 测试的 `ParallelismConfig` 改从 `torchtitan.config.parallelism` 导入（#5744）；
  - `lr_scheduler.py` 的空 optimizer 断言跟着 #5775 挪进新的 `__init__`，删掉；
  - `test_lr_scheduler.py` 改用 `LRSchedulersContainer.Config().build(...)`。
- **body 要改：**
  - 第一句 "LoRA on main fails to build the Kimi K3 debug model" 的原因就是那个放置断言，main 的 #5884 之后大概率已经不成立（要在 GPU 机上确认，见下）；
  - 标题里的 "place adapters for every base sharding" 和 Design 的放置几条要删；
  - Test plan 的通过数要改。
- **要你定的：**
  - main 的 #5844 加了私有的 RL 方法 `Trainer._trainable_model_state_dict`（adapter-only 同步，含 buffer），和 PR 公开的 `trainable_state_dict` 重叠。agent 保留了 PR 的版本。要留就在 body 里一句话说明为什么不同。
  - `_strip_wrappers` 和 `components/checkpointer/utils.py` 的 `canonical_fqn(name, prefixes)` 重叠。这是 base 上就有的，不是 main 带来的，但按复用规则算一条发现。
  - main 的 #6003 之后，`trainable_state_dict` 也会返回 `trainable_modules` 的参数，docstring 里 "under LoRA, the adapters" 不完整了。
- **GPU 待验证**（全部在 `575530d83` 上，同一份热缓存）：
  - `pytest tests/unit_tests/cpu/models/common/test_lora.py`：新增 6 个，原来 10 个；
  - `pytest tests/unit_tests/cpu/components/optim/test_optimizer_param_groups.py`：新增 2 个；
  - `pytest tests/unit_tests/cpu/components/optim/test_lr_scheduler.py`：新增 1 个；
  - `pytest -m multi_gpu tests/unit_tests/gpu/models/common/test_lora.py`：4 卡；
  - `pytest -m multi_gpu tests/unit_tests/gpu/distributed/test_clip_grad_norm.py`：2 卡；
  - 在**纯 main** `631d544b1` 上跑 kit 的 `lora_build_all.py`（meta 上用全部默认 target 建 LoRA），确认 K3 和其他模型在 main 上已经能建，决定 body 第一句怎么改；
  - 在纯 main 上跑 clip 测试，确认 main 仍然报 `No backend type associated with device type cpu`；
  - Results 表全部是 `b61c686dc`（base `b663c1673`）上测的，包括第 1 步 "identical / +0.00x%" 和 merged-vs-expected 为 0，都要在新 head 上重跑；
  - `pyrefly check` 对照 main 重跑。
- **Summary 要点（你自己写成 2-5 句英文）：**
  - 问题：main 上没有办法导出训好的 LoRA。HF 导出在有冻结参数时直接 raise（#6057 加的，带一个 PEFT 导出的 TODO）；原生 state dict 里适配器是分开的。一个 stage 里没有可训练参数的 PP rank 会连着在三处失败：`matched no parameters`，然后 `empty parameter list`，然后 CPU 上的 `tensor(0.)` grad norm 被 NCCL all-reduce 拒绝。main 的注释说支持这种情况。
  - 为什么这样做：
    - merge 在本地分片上用 fp32 做，保留 base 的 placement。用 DTensor 算子做时，FSDP × TP 下堆叠的 `w13` 报 `Cannot unflatten ... _StridedShard`。
    - 原地换参数不行，因为 FSDP2 的 state_dict pre-hook 会换回自己的分片参数。
    - main 的 #5893 里 `_merge_expert_lora` 是 Dist-MoE 编译前向里的 merge，不是导出：复用它会把 delta 截到 bf16，还占它 8 个位置的重编译缓存。
  - 你查过的：
    - 18 格矩阵，merged-vs-expected 最坏差 0；
    - 冻结 stage 的 pp8 和 pp4 × vpp2 两格都是 8.10062 / 0.1309（`PR_K3_PARALLELISM/STACK_FULL_AND_LORA_MM18_2026-10-12.md` 的 "bug 2"、"bug 3"、"18 格矩阵"）。
    - 这些都是旧 base 上的结果，rebase 后重跑之前只能说明意图。

### #5469 `[Kimi K3]` 视频训练

- 源 `k3_video` = `e117882e6`（PR 分支本身），rebase 后为 `06836222a`（worktree `rb5469`）。
- 5 个提交全部保留，没有丢也没有压：
  - `3cb2ea626` 对应 `b125307aa`：message 加一条 `video_resize`；
  - `75b934a8f` 对应 `493b77209`：message 最后一句改写，"names its pixel bounds" 那处改动已经不存在；
  - `659c72b74` 对应 `4f401fff3`；
  - `7cadb6fcc` 对应 `c7a2fc27f`，补丁相同；
  - `e117882e6` 对应 `06836222a`：message 去掉 "load_video shares their decode loop"。
- 8 个文件，+444/-33（原来 7 个文件，+516/-101）。
- main 的 #5873（`27ee255fc`，`VisionProcessor`）只提供接口：`video_resize`、`resize_and_normalize_video`、`patchify_all`。抽帧、时间戳、分 chunk、chunk header、`max_num_frames` 拒绝都没有，PR 的 5 项行为 main 一项都没覆盖。
- 适配：
  - `mm_datasets.py`：`_process_chunked_video_text_sample` 改收 `vision_processor`，替掉原来的 11 个 image 参数；resize 用 `resize_and_normalize_video`，每个 chunk 的 token 数用 `num_placeholder_tokens`；照 main 的 `_process_video_text_sample` 加了 `vision_processor is None` 的 raise。chunk 形状、prompt、各种拒绝都不变。
  - `utils/video.py`：#5873 把整个文件连同 `load_video` 删了。按 `utils/audio.py` 的样子重建，只留数据集一侧的 `format_video_timestamp`、`sample_frame_indices`、`load_video_frames`。PR 原来的 `load_video` 重构和它的测试 `test_load_video_keeps_its_sampling` 不要了。`_decode_sampled_frames` 只剩一个调用者，内联进 `load_video_frames`。
  - `numerical_tests_kimi_k3.py`：改用 `resize_and_normalize_video` 和 `patchify_all`，预算从 K3 的 kwargs 读，去掉三个像素常量。
  - `model.py` 的拒绝跟着 #6084 进了 `KimiK3MultimodalModel`，自动合并。`vision_encoder.py` 把 main 的 `build_flops_estimator` 和 PR 的 override 都留下。
- **会改行为的设计选择：** main 上 `video_resize=None` 表示模型不收视频，所以在 `KIMI_K3_VISION_PROCESSOR_KWARGS` 里加了 `"video_resize": NaViTPatchGridResize.Config(max_patches=16_384, max_patches_per_side=512)`。
  - 数值是 HF 的 `in_patch_limit_each_frame` 和 `patch_limit_on_one_side`，PR 的对拍脚本原来就用这两个。
  - Qwen3.5 和 Muse Spark 也是在各自 flavors 的 kwargs 里设 `video_resize`，K3 照同样的模式。
- **要你定的：**
  - 上面这个 `video_resize` 留不留；
  - 内联 `_decode_sampled_frames` 行不行；
  - flavors 里 kwargs 上方那条注释的括号，要不要补上 `in_patch_limit_each_frame`（按无新注释规则没动）；
  - main 删 `load_video` 是因为树里没人调用。PR 的 `load_video_frames` 在树里同样没有调用者（没有加 recipe），review 时可能被问到；
  - `KIMI_K2_5_VISION_PROCESSOR_KWARGS` 仍然不收视频；整段视频的总预算 `in_patch_limit_video` 仍然没建模，和 rebase 前一样。
- 本机：
  - 加了 torchvision、renderers、CuTeDSL 的桩，用 `C:/Users/78532/AppData/Local/Temp/claude/e487daee1c710758/a4c67aa2-abec-4ce6-b41f-2c320fd84b25/scratchpad/chk5469.py` 复现了 PR 期望的时间戳（`00:00:09.599`、`00:00:02.999`）、帧号、chunk 形状、prompt、四种拒绝，以及 5 帧时的 grid `[[4,24,24],[1,24,24]]`。
  - resize 和 PyAV 两条路径没跑到。
- **GPU 待验证**（`06836222a`）：
  - `pytest tests/unit_tests/cpu/components/data/test_mm_dataset_preprocessing.py`：应为 36 项，新增 9 个（body 写的是 10 个，少的是 `load_video` 那个）。`TestLoadVideo` 要装 PyAV。
  - `pytest tests/unit_tests/gpu/models/kimi_k3/test_kimi_k3.py`：应为 8 项，新增 2 个（body 写的是 7 项）。
  - `python -m scripts.checkpoint_conversion.numerical_tests_kimi_k3`，再加 `--num_frames 8`、`--num_frames 5` 各跑一次。
  - kit 的 `k3v_cells.py` 和 `k25check/compare_k25.py` 用了已删的 API（11 个参数、`utils.image`、`utils.video.load_npy_video_frames`），要先移植。合成帧最大 42 px，所以 K3 的视频预算和原来 debug 的图片预算给出同样的几何。
  - body 的 22 行 Results 表是 `8a3f5bcaa` 上测的。之后 main 改了 vision 的 TP/CP 集合通信（#6043、#5803、#5806、#5807），拆了模型（#6084），重做了 loss（#4582），所以要在新 head 上重跑。
  - K2.5 的对照（2888 个用例、5 个视频、像素差不超过 2/255）用的是旧 `process_video`，也要重跑。
- **body 要改：**
  - `VisionProcessor.Config(sample_processor=...)` 现在是 `VisionSampleProcessor.Config(sample_processor=...)`，再加 `Trainer.Config.vision_processor`；
  - 去掉 "load_video shares their decode loop"；
  - 加一条 `kimi_k3/flavors.py` 的 `video_resize`；
  - 测试数照上面改。
- **Summary 要点：**
  - 问题：main 上 K3 的 processor 是 `video_resize=None`，模型拒收 `pixel_values_videos`，所以 K3 不能用视频训练。
  - 为什么把 chunk 放进 `pixel_values`：K3 的图片和视频共用一个 `<|media_pad|>`，另开一条流就没法和占位符对上。`kimi_k2_7/model.py` 的 TODO 指的也是一条按文档顺序的流。
  - 为什么每个 chunk 最多 4 帧：encoder 在时间上对每个 item 做 pooling，`max_num_frames=4` 是它能收的最长 item。
  - 你查过的：用 K3 的配置，对照官方发布的 Kimi K2.5 processor，比了帧号和 prompt（2888 个用例）、5 个编码后的视频、像素差（`PR_K3_VIDEO/PR_BODY_K3_VIDEO_2026-10-15.md`、`kit_2026-10-15/k25check/`）。全部在 rebase 之前，要重跑。
  - 没确认的：K2.5 的 processor 对长视频是否用 `in_patch_limit_video`。

### #5635 和 #5619 `[MiniMax M3]`

- worktree `rbM3`，#5619 叠在 #5635 上：
  - #5635：源 `minimax_m3` = `8a2651476`（PR 分支本身），rebase 后为 `b809f1eee`，34 个文件，+3388/-94。提交对应：`0e146e96a` 对 `f7ca92206`，`1b733eba3` 对 `31dba19e4`，`a1a1ef432` 对 `e163929b2`，`848102e45` 对 `5a694f6fc`，`8a2651476` 对 `b809f1eee`。
  - #5619：源是 review 分支 `minimax_m3_5d_review1` = `70b6efa63`，rebase 后为 `796550353`，39 个文件，+3504/-114。`b1730e085` **丢掉**；`c70a694d7` 对 `6cd975b19`；`70b6efa63` 对 `796550353`。
- 丢 `b1730e085` 的原因：#5806 删掉了 `after_vision_scatter`，Kimi K3 也不再声明 block 边界，要挪的东西已经没有了。
- 改过 message 的提交：`5a694f6fc`（Qwen3.5 的 10 个 ignore 已被 #6084 去掉，只剩 Qwen3.8 的 1 个）、`6cd975b19`、`796550353`。
- 准备提交里的适配，都是机械的：
  - `qwen3_5/flavors.py` 和 `model.py` 的 import 冲突，来自 #6084 新加的 `Qwen35MultimodalModel`；
  - "Partial RoPE" 的 docstring 改动跟着挪到 `Qwen35Model`；
  - `deepseek_v3/state_dict_adapter.py` 保留 #5582 去掉的 save-map 过滤，再加上 `_build_from_hf_map` 接口；
  - `decoder.py` 保留 main 的 `VocabParallelEmbedding` 和有类型的 `layers`（#5802、#5917），只放宽 norm 的类型。
- M3 本体，全部是 main 的 API 变化逼出来的：
  - 照 #6084 和 main 上其他 VLM 拆类：`MiniMaxM3Model(Decoder)` 是纯文本类，`MiniMaxM3MultimodalModel(MiniMaxM3Model, MultimodalModel)` 必须带 vision encoder；去掉了按配置判断 `vision_encoder is None` 的分支；adapter 按配置类型选 vision 的映射；
  - #5807：`in_src/in_dst/out_src` 改成 `input_shardings/output_shardings`；
  - #5802：`Embedding` 改成 `VocabParallelEmbedding`，scatter 之前显式 `maybe_gather_tp_input`；`pre_lm_head_norm_sharding_config` 改成 `norm_sharding_config`；
  - #6043 和 #5803：vision tower 的 `InvariantRowParallelLinear` 改成 `@disable_dense_sp()` 下的 `RowParallelLinear`，`fc1`、`linear_1`、`merge_linear_1` 改成 `ColumnParallelLinear`；vision RoPE 的缓存在 TP 上复制，那条注释跟着改了；
  - #5779：`preprocess_inputs` 去掉 `max_context_length`；
  - `local_compile_regions` 改成 `local_compile: LocalCompileConfig`；
  - #4819：`get_nparams_and_flops` 去掉，MSA 的 FLOPs 计算挪进 `MiniMaxSparseAttention.Config.flops_per_token`；
  - #5873：recipe 和对拍脚本走 `VisionProcessor`，flavors 里新加 `MINIMAX_M3_VISION_PROCESSOR_KWARGS`（值来自原 recipe）；
  - `disable_cuda_graphs` 改名为 `disable_accelerator_graphs`；
  - #5917：重新声明 `layers` 和 block 的 `attention`，带 `# pyrefly: ignore [bad-override]`。
- 5D 的适配：
  - MSA 边界原来隐式的 gather，改成 `MiniMaxSparseAttention.forward` 里的 `maybe_gather_tp_input(self, x_TD)`；去掉 `msa_attention_sharding_config`，拷过来的 GQA `gqa_attention_sharding_config` 已经声明了同样的边界；
  - 第 0 层 scatter 之后的配置照 #5806 改成 `replicate_to_residual`；
  - projector 的 I→R 照 #5803 改成 `invariant_to_replicate(..., TP)`；
  - `6cd975b19`：main 上这两个配置仍然是一模一样的两份，所以去重保留，docstring 用 main 的 "Projector output: TP- and CP-invariant."。
- **要你定的：**
  - 纯文本类 `MiniMaxM3Model` 没有文本 flavor，也没有 docstring，原来的 docstring 放在多模态类上；
  - **FLOPs 现在算上了 vision tower：** encoder 配置上新加约 40 行的 `build_flops_estimator`，main 的 `MultimodalModel` 要求有它。报出来的 MFU 会变；
  - 原 PR 没有、这次新增的：公开的 `MINIMAX_M3_VISION_PROCESSOR_KWARGS`，两行 pyrefly pragma；
  - 集合通信拆在两个 PR 里：gather 在 #5635，restore 和 replicate 转换在 #5619；
  - 原 PR 新加的那个 integration 格子（`minimax_m3_mm_fsdp`）留着。新模型带自己的格子是不是在 10-02 规则的例外里，你定。
- 本机：AST 检查没有多余或缺失的 import；pyrefly 0.45.1（装在 scratchpad 里）对照 main 没有新增错误，只有缺 `transformers` 的那些；pytest 没跑。
- **GPU 待验证：**
  - 在两个 head 和 `631d544b1` 上各跑一遍 `pytest tests/unit_tests/cpu`，只报分支独有的失败；
  - `pytest tests/unit_tests/gpu/models/minimax_m3 tests/unit_tests/gpu/models/common/test_offset_rmsnorm.py -q`；
  - #5635：`python -m tests.integration_tests.run_tests <out> --test_suite models --test_name minimax_m3_mm_fsdp --ngpu 4`；#5619：`--test_name "minimax_m3_mm_fsdp+tp+sp+ep"`；
  - `python -m scripts.checkpoint_conversion.numerical_tests_minimax_m3` 跑三遍：不加参数；加 `--force-hf-blocks --force-hf-routing`；加 `--dtype float32 --attn_backend flex`；
  - Qwen3.5 的 RoPE 提交对照 `631d544b1`：dense 1 卡，MoE tp2 × pp2 × ep2；
  - 对改过的文件跑 pre-commit（只 stage 本 PR 的文件），再跑 HSDP 的 kit 和 4D 矩阵；
  - 两个 body 和两个 README 里的数全部要重跑：对拍的 KL 和匹配率、HSDP 和 fsdp4、所有 4D 表、CPU 测试数。main 的 #5543（router 分数在 W2 之前乘）改了 MoE 的数值。
- **Summary 要点：**
  - 问题：main 上没有 M3，也没有 MSA；`OffsetRMSNorm` 在 `qwen3_5` 里；partial RoPE 只是 Qwen 的 helper；DSV3 的 adapter 写死了 MLA 的 key 映射和 `ComplexRoPE`。
  - 做法：MSA 是 `FlexInnerAttention` 和 `GQAttention` 的子类，再加一个 `AuxLoss` 的 KL。M3 通过新开的接口复用 DSV3 的逐 expert 转换。
  - 排除的方案：DSV4 的 `SparseIndexerLoss` 目标不同（对 head 求和），所以没复用。
  - 你查过的：`PARITY_HF_2026-10-24.md`、`CHECKS_2026-10-25.md`、`M3_4D_TABLES_EN_2026-10-25.md`、`kit_m3_5060_2026-10-25/`。都在 rebase 之前，rebase 后一项都没有重验。

### #4380 `[Kimi K3]` MoonViT Dynamic CP

- 源 `k3_cp_mm` = `cpmm_review1` = `a2a97b88a`，rebase 后为 `40a38f456`（worktree `rb4380`）。
- 11 个提交全部保留，原 message 都没改。954+/13-（原来 957+/12-）。新增的注释和 docstring 和原 PR 是同一组 60 行。
- 适配，每一处都放回引入那段代码的提交里：
  - #6084：`KimiK3Model` 现在是纯文本，没有 `parallelize`。
    - 接线挪到 `KimiK3MultimodalModel._parallelize`：先调 super，再调 `install_vision_cp`。位置不变，仍在 `activate_spmd` 里，模块树 parallelize 之后、remat 和 FSDP 之前。
    - `_encoder_is_cp_invariant` 挪到 `KimiK3MultimodalModel`。`vision_encoder` 现在是必填字段，所以去掉了 `vision is None` 的判断。
  - **#5803，model 文件多了一处改动：**
    - main 把 encoder 输出的 CP I→R 挪进了 K3 的 `_prepare_multimodal_embeds`。开 dynamic CP 时跳过这一步（`if self._encoder_is_cp_invariant(...)`）。
    - 这样保住了 `a2a97b88a` 的行为：不跳过，梯度会求和两次，就是之前的 2 倍 bug。
    - 原 PR 在 model 文件里没有这处改动。
    - 另外，`VisionCPAttention` 的切图路径在 `_qkv` 之前加了 `maybe_gather_tp_input(self, x)`，和 main 的 `VisionAttention.forward` 一样。不加的话，TP 反向会缺一次 all-reduce。
  - #6043：`MoonViTEncoder.forward` 现在带 `@disable_dense_sp()`，重写它的 `MoonViTCPEncoder.forward` 也加上同样的装饰器。不加的话，切图路径的 patch_embed 和 projector 会在 SP 布局下跑。
  - #5807：
    - `_sharding_replicated_over_cp` 改写 `state_shardings`、`input_shardings`、`output_shardings`；
    - 原来隐式的 `pixel_values` I→R，在 `MoonViTCPEncoder.forward` 的 dynamic CP 分支开头改成显式的 `invariant_to_replicate(pixel_values, CP)`；
    - 去掉了 `multimodal_input_sharding` 的导入。
  - #5889：`self.flex_attention` 改名为 `self.inner_attention`。
  - #6035：`"debugmodel"` 不再带 vision encoder，GPU 测试改用 `"debugmodel_mm"`。
  - #5744：`ParallelismConfig` 改从 `torchtitan.config.parallelism` 导入。
  - #5776 和 #5463：测试的 `model.parallelize(...)` 加上 `precision=PrecisionConfig(param_dtype="float32", reduce_dtype="float32")` 和 `local_compile=LocalCompileConfig()`。
- **要你定的：**
  - 接线挂在 `_parallelize` 上，没有照 K2.5 的样子 override `parallelize`。这样顺序和原来完全一样，省掉约 20 行签名样板，graph_trainer 直接调 `_parallelize` 时也能走到。另一种写法是 override `parallelize`，先调 `super().parallelize()` 再调 `install_vision_cp`。
  - 第 11 个提交的 message 没提 `_prepare_multimodal_embeds` 里新加的跳过，值得补一句。
- **和 #4381 的交叉：** 两个 PR 都改 `_prepare_multimodal_embeds` 里同一段 I→R 转换。#4381 把 `vision_embeds` 接口放在转换之前，#4380 在开 dynamic CP 时跳过 CP 那一步。后合入的那个 PR 要解这处冲突，还要负责 DEP 加 dynamic CP 的组合（副本用 sub-CP groups，在 dp × cp 上归约）。
- **GPU 待验证**（`40a38f456`）：
  - `python -m pytest tests/unit_tests/cpu/test_kimi_k3_vision_cp_plan.py -q`：应为 7 passed；
  - `python -m pytest tests/unit_tests/gpu/test_kimi_k3_vision_cp.py -q`：应为 3 passed，要 4 卡。其中 2 卡的梯度测试就是检验新加的跳过对不对：对时相对误差小于 1e-5，错了约为 1.0；
  - `python -m tests.integration_tests.run_tests <out> --test_suite h100 --test_name kimi_k3_mm_allgather_kv_cp --ngpu 2 --no-parallel`，`kimi_k3_mm_ulysses_cp` 同样跑一遍，覆盖显式 `pixel_values` I→R 和 tower 输出的类型检查；
  - Results 里所有的数都要在 `631d544b1` 上重跑，同一份热缓存，加一行噪声底：
    - debug flavor 变了（`kv_lora_rank` 从 64 改成 512，MLA 的 `wkv_b` 挪进 `inner_attention`），recipe 现在走 VisionProcessor；
    - 可能不再成立的说法："dynamic CP 关时 100 步和 main 相同"；第 1 步梯度逐位相同的个数（465/465）；clipping 的解释（4.5625 对 4.5938）；"阈值 256 在第 5 步第一次切图"；
    - tower 显存和时间表，至少 main 那一列要重测；
  - 注意：#4819 的 FLOPs 估计在每个 CP rank 上都按整个 tower 算，开 dynamic CP 时 MFU 会多算。
- **body 要改：**
  - `KimiK3Model.parallelize` 改成 `KimiK3MultimodalModel._parallelize`；
  - "four places" 改成五处，多出 CP 转换的跳过；
  - "its output boundary all-reduces" 改成 `_prepare_multimodal_embeds` 里显式的 `invariant_to_replicate`。
- **Summary 要点：**
  - 问题：main 上每个 CP rank 都编码完整的视觉输入（见 `multimodal_input_sharding` 的 docstring），tower 的显存和时间不随 CP 减少。一张 4032 px 的图，selective remat 下 55.2 GiB、10.9 s，不开 remat 直接 OOM（body，`e61df55b6` 上测的）。
  - 做法：K3 报告 5.2.3 节。大图按 merge-block 的行带切到 sub-CP group 上，K/V 在组内 gather。每个 micro-batch 只调一次 tower，DP rank 之间的 FSDP 集合通信才能对上。
  - 为什么用部分梯度加 FSDP 在 dp_shard × cp 上切（方案 1），而不是在 #5266 的方案里补偿（方案 2）：方案 2 要只对切片做反向、乘 1/C、再挂一个梯度 hook（`DYNCP_VS_5266_2026-10-15.md`）。
  - 你查过的：
    - 2 倍分叉的探针，以及 fp32 harness 上 1.2e-6 的差（`DYNCP_5266_FIX_2026-10-16.md`）；
    - bf16 下 bank-swap 和整图路径逐位相同；
    - 都是在 #5266 时期的树上测的，机制仍然成立，数要重跑；
    - 给 Jessica 的说明（`NOTE_JESSICA_DYNCP_5266_2026-10-16.md`）发没发出去，没有确认。

### #4381 `[Kimi K3]` MoonViT PP DEP

- 源是 review 分支 `dep_review1` = `caa8726fa`：PR 头 `7529daeea` 加上 10-20 的 `grad_dtype` 修复（`caa8726fa` 还不在 PR 分支上）。rebase 后为 `56e5e196c`（worktree `rb4381`）。
- 12 个提交全部保留。7 个补丁相同，两个只是上下文不同（`4e3e3eee1`、`d663ff690`）。有实质改动的三个：
  - `ebbd1d832` 对应 `1abbb7c81`；
  - `7529daeea` 对应 `d3bc60aff`，message 加长了；
  - `caa8726fa` 对应 `56e5e196c`。
- `plan.py`、`runtime.py`、`stage.py`、`schedule.py`、规划测试、GPU 测试，和 `caa8726fa` 逐字节相同。
- 适配：
  - `model.py`，因为 #6084：`KimiK3Model` 现在是纯文本的 `Decoder`。`vision_dep` 配置手工挪到 `KimiK3MultimodalModel.Config`；`vision_embeds` 接口在 `KimiK3MultimodalModel._prepare_multimodal_embeds` 和 `forward` 上。
  - **会改行为的一处**，同一段，因为 #5803：main 把 encoder 输出的 TP、CP I→R 转换从 tower 的 ShardingConfig 挪进了 `_prepare_multimodal_embeds`。
    - DEP 的特征放在这两步转换**之前**，所以和本地 encoder 的输出走同样的转换。
    - 特征处收到的梯度，在 stage 0 上已经做过 TP 加 CP 的归约。tower 没有了那条边界，副本反向时在输出处不再归约。
    - 求和的量相同，只是这次集合通信从副本的反向挪到了 stage 0 的反向。放在转换之后会留下部分梯度。
  - `pipeline_parallel/__init__.py`：按配置类型取 `dep`，是 `KimiK3MultimodalModel.Config` 才读，否则用默认（关）的 `KimiK3VisionDepConfig()`，和 #6084 "按配置类型选"一致。纯文本 K3 的 PP 路径不变。
  - `d3bc60aff`，因为 #5776：
    - `training.mixed_precision_param` 改成 `precision.param_dtype`；
    - `training.enable_cpu_offload` 改成 `parallelism.fsdp_enable_cpu_offloading`；
    - `build_vision_replica` 改收 `precision=` 和 `parallelism=`；
    - 跟着 #6084 去掉副本构建里的 `annotate_replicated_parameters`，和 main 的 `MultimodalModel.parallelize` 一致，不改行为。
    - 这个提交的标题和正文写明了以上几点。
  - `56e5e196c`：`grad_dtype` 改读 `precision.reduce_dtype`。
  - `tests/integration_tests/b200.py` 没有冲突，仍然只在已有格子里开 `vision_dep`，没有加格子。
- **要你定的：**
  - 适配写进改了 message 的 `d3bc60aff`，还是单独开一个适配提交；
  - PR 的测试放在 `tests/unit_tests/{cpu,gpu}/` 根目录，main 已经有 `.../models/kimi_k3/`。旧 base 上就是这样，没有挪；挪的话 GPU 测试导入 `_VisionDepChecks` 也要跟着改。
- **和 #4380 的交叉：**
  - 两个 PR 都改 `KimiK3MultimodalModel`；
  - DEP 的 `_reduce_gradients` 只在 `dp` 上归约，而 #4380 开了 dynamic CP 后要在 dp × cp 上归约；
  - 这一处组合由后合入的那个 PR 负责。栈上有过这段组合代码（`STACK_REASSESS_2026-10-19.md` 第 2 节），这次没看栈。
- **GPU 待验证**（`56e5e196c`）：
  - `pytest tests/unit_tests/cpu/test_kimi_k3_vision_dep_plan.py tests/unit_tests/cpu/test_kimi_k3_vision_dep.py -q`：23 个（body 写的 22 个是 `caa8726fa` 之前的数）；
  - `pytest tests/unit_tests/gpu/test_kimi_k3_vision_dep.py -q`：2 个，4 卡；
  - `pytest tests/unit_tests/cpu/models/kimi_k3 tests/unit_tests/cpu/models/test_build_model_config.py tests/unit_tests/cpu/test_integration_test_definitions.py tests/unit_tests/gpu/models/kimi_k3 -q`；
  - `python -m tests.integration_tests.run_tests <out> --test_suite b200 --test_name kimi_k3_fsdp2_tp2_ep2_pp2_vpp4 --ngpu 8`；
  - DEP 关和 DEP 开两格，seed 固定，同一份热缓存，对照 `631d544b1`（kit 的 `dep_cells.py` 和 `run_seeded.sh`）。#5803 改了特征梯度在哪里归约，所以 10-20 在 `efdde5eda` 上"第 1、2 步逐位相同、第 3 步起分开"要重测。
- **body 里不再成立的：**
  - "identical to DEP off on all 100 steps"，自 #5250 起就不成立；
  - Results 表和 69% 到 71% 的覆盖率，是 10-04 的旧 main 上测的；
  - H100 上覆盖率降到 48% 到 52%（多出约 38 s 空闲，profiler 下 5160 次 C++ 异常），仍然没有定位；
  - body 写的是 `torchtitan_recipes/tests/suites/b200.py`，现在是 `tests/integration_tests/b200.py`；
  - `vision_embeds` 现在在 `KimiK3MultimodalModel.forward` 上；
  - "Relation to earlier revisions" 一节要删。
- **Summary 要点：**
  - 问题：main 的 `KimiK3MultimodalModel.pipeline_first_stage_module_fqns = ("vision_encoder",)` 把整个 ViT 的前向和反向都钉在 stage 0 上，其他 PP rank 在调度的 bubble 里空等。
  - 做法：K2.5 报告的 DEP，每个 PP rank 持有一份 ViT 的 TP 副本，每步从 stage 0 刷新；再加上 K3 报告 5.2.3 节把 ViT 的计算排进 bubble。
  - 为什么这样做：参数仍然只在 stage 0 上，optimizer、checkpoint、FSDP 都不用动。早先的版本给 ViT 单开一个 pipeline stage，这一版取代了它。
  - 你查过的：
    - `REBASE_PLAN_4380_4381_2026-10-18.md`：CPU 90 passed、GPU 2 passed，8 × 5060 上 seed 固定的 B200 形状格子；
    - `H100_RERUN_4380_4381_2026-10-19.md`：DEP 开关之间的差定位到 #5250 的 `grad_dtype`，由 `caa8726fa` 修掉，剩下的只是 fp32 加法顺序。
    - 都在旧 base 上，要在 `631d544b1` 上重跑。

### #4751 `[Kimi K3]` MoonEP

- 源 `k3_moonep_seam` = `moonep_review1` = `16ff9da7f`，rebase 后为 `945f114e2`（worktree `rb4751`）。
- 10 个提交全部保留，message 逐字节不变。13 个文件，+1187/-9（原来 12 个文件，+1150/-7）。多出来的文件是 `test_moe.py` 的桩。
- 适配：
  - #5277 和 #5345 挪了目录：
    - 代码现在在 `models/common/moe/{moe,token_dispatcher}.py`；
    - 测试挪到 `cpu/config/test_transforms.py`、`cpu/distributed/test_moonep_ops.py`、`gpu/distributed/test_moonep.py`（和 DeepEP 的测试放在一起）、`gpu/models/common/test_moe_shared_experts_stream.py`。
  - `pyproject.toml`：用 main 的那一行（#5530 去掉了 helion，加了 mlflow），再加上 `moonep.*`。
  - **#5280，调用方的用法变了：** main 的 `TokenDispatcherTransform` 改成显式接收 `num_max_tokens_per_rank`，也没有 context 了，PR 原来从训练形状推导容量的那段就没了。
    - `routed_experts` 字段、转换循环、PR 的那句 docstring 都叠在 main 的版本上；
    - 测试里 `_MOONEP` 传 `num_max_tokens_per_rank=512`；
    - recipe 用 `parallelism=ParallelismConfig(expert_parallel_degree=4)` 构建。
  - #5323：`self.ep_mesh.get_group()` 改成 `_ep_process_group()`。
  - #5516：`dispatch` 和 `MoonEPRoutedExperts.forward` 改收 `routing_map_TE`。dispatcher 内部用 `routing_map_TE.sum(dim=0)`，和 `MoE.forward` 原来的计算相同，计数一样。
  - #5046：`forward` 收 `padding_mask_T` 后直接删掉，和原生的 `RoutedExperts` 一样。
  - **#5086，会改行为：** activation 现在收融合的 `gate_up [R, 2, F]`。
    - 第 1 个提交的 `_compute` 先把 gate 和 up stack 起来再做 activation；
    - 从第 3 个提交起，expert op 把两个 pool GEMM 的输出 stack 起来，存一份 `gate_up` 给反向（字节数不变），op 返回 `(out, gate_up)`；
    - 反向对 `gate_up` leaf 求导再 unbind；`_grouped_mm` 改在跨步的 view 上跑（leading dim 是 2F，要求 F % 8 == 0）；
    - 代价：每次 expert 前向多一次 `[R, 2F]` 拷贝，另有一份同样大小的临时张量。
  - remat 配置（#4893、#5101、#5102），只在 GPU 测试里：
    - `SelectiveAC` 改成 `DefaultActivationRematConfig`；
    - `FullAC` 改成 `FullActivationRematConfig`；
    - `RegionAC(save_regions=[])` 改成 `ActivationRematConfig(save_regions=[])`；
    - 后两个现在是同一个策略。
  - #5353：`disable_cuda_graphs` 改名为 `disable_accelerator_graphs`。
  - #5280 和 #5345：`make_*_config` 加 `enable_ep`、`enable_sp`，`w1/w2w3_param_init` 改成 `w13/w2_param_init`。
  - side stream 落在 #5829 的 `_add_shared_experts_and_reduce_across_tp` 上：
    - 启动仍在 `MoE.forward` 开头，结果通过新加的可选参数 `shared` 传进去；
    - PR 的 join 抽成一个新的私有方法 `_shared_experts_output`，main 的两条路径都调它，否则 batch-invariant 模式下会把 shared experts 跑两遍；
    - 去掉了 PR 的 `recompute_needs_tensor(shared_TD)` 和裸加法，main 现在在保留的 `shared_add` 区域里做加法；
    - 不在 batch-invariant 模式时，行为不变。
- **要你定的：**
  - 新的私有 helper `_shared_experts_output` 在 core 的 `moe.py` 里。agent 说 main 没有等价的东西，按复用规则在 diff 审查里记一笔。
  - side stream 用的是 `torch.cuda.Stream` 和 `torch.cuda.current_stream()`，main 的 #5450 已经改成 `torch.accelerator`，review 可能会指出来。
  - **rebase 之前就有的缺口：** Kimi K3 上 `shared_experts_stream` 不起作用，因为 `KimiLatentMoE.forward` 从来不启动 side stream，原 head 上就是这样。要么在 `KimiLatentMoE.forward` 里接上启动，要么从 body 里删掉 "as the K3 report does"。
  - dist_moe：`DistMoeRoutedExperts` 替换的是 routed experts，不是 token dispatcher，由 `DistMoeTransform` 选中。`relations.py` 已经声明它和 `TokenDispatcherTransform` 冲突（base `3f087cf15` 上就这样），它也不是默认（默认仍是 `AllToAllTokenDispatcher`）。MoonEP 和它互斥，代码上没有碰撞。review 可能会问：为什么 MoonEP 给 `TokenDispatcherTransform` 加 `routed_experts=`，而不是像 Dist-MoE 那样单开一个 transform。
  - 可选：
    - `test_selective_ac_*` 改名为 `test_default_*`；
    - `275f29f20` 和 `ed1142c75` 的 message 里还写着 `SelectiveAC`、`FullAC`、`RegionAC`；
    - main 的 MoE README 后端表里没有 MoonEP 一行，没加。
- 本机：`test_moonep_ops.py` 1 passed；`test_transforms.py` 收集不了。
- **GPU 待验证**（`945f114e2`）：
  - CPU：`pytest tests/unit_tests/cpu/config/test_transforms.py tests/unit_tests/cpu/distributed/test_moonep_ops.py tests/unit_tests/cpu/models/common/test_moe.py -q`，原来的 39 个不再适用；
  - `pytest tests/unit_tests/gpu/models/common/test_moe_shared_experts_stream.py -q`：应为 2 passed。再做一次变异检查：去掉 `x_TD.register_hook`，第二个测试应该失败；
  - 4 张 H100，MoonEP `33327eb`：`pytest tests/unit_tests/gpu/distributed/test_moonep.py tests/unit_tests/gpu/models/common/test_moe_shared_experts_stream.py -q`，应为 11 passed。这是跨步 `_grouped_mm` 和 `gate_up` 移植第一次实跑；
  - 端到端 6 格，同一份热缓存，MoonEP 对标准 EP，都在 `631d544b1` 上。kit 的 transform 现在要传 `num_max_tokens_per_rank`。标准是第 1 步逐位相同；
  - 要重跑的说法：整张 loss / grad norm 表（标准 EP 的格子也要重跑）；"标准 EP 跑两遍相同；FullAC 和 selective 相同"；微基准的 ms 和 GiB（多了 stack 拷贝）；
  - routed rows 表照原标注保留（dispatch 没变），1.85 GB 的 pool 大小不变。
- **body 要改：**
  - 路径改成 `models/common/moe/token_dispatcher.py`；
  - 选择 MoonEP 的代码片段要加 `num_max_tokens_per_rank`；
  - 删掉 "which the transform derives from the training shape"；
  - remat 的说法：main 只有 `ActivationRematConfig` 的预设，MoonEP 的 op 在任何策略下都是保留区域（`recompute=False`）；
  - Costs 一段加上 stack 拷贝；
  - Test plan 的路径。
- **Summary 要点：**
  - 问题：main 的 EP 后端只搬 token，一个 rank 的 routed rows 跟着路由的偏斜走。最热的 rank 在自然路由下是均值的 1.54 倍，偏置路由下 3.19 倍（K3 debug，128 个 expert，top-8，FSDP4 × EP4，较早的版本；`MOONEP_H100_2026-09-30.md` 第 4 节）。
  - 做法：MoonEP 预取热门 expert 的副本，每次 dispatch 都正好 S × K 行（两种路由下都是 320/320）。
  - 为什么放在这里：放在 dispatcher 接口上，和 DeepEP、HybridEP 并列，取代了早先模型本地的 experts 类和 `KimiLatentMoE.parallelize` override（`DIFF_AUDIT_MOONEP_2026-09-19/28`）。需要自己的 `RoutedExperts` 子类，因为 GEMM 要覆盖 slot 的行，slot 的梯度要归约回原 rank。
  - 你查过的，在 `16ff9da7f` / base `3f087cf15` 上（`MOONEP_DIFF_REVIEW_2026-10-06.md`）：
    - 3 种布局下第 1 步和标准 EP 逐位相同；
    - 20 步内最大相对差 2.03e-3，而标准 EP 换一种布局本身就到 1.1e-2；
    - 11 个 GPU 测试通过；
    - hook 变异检查 3/3 失败。
  - 这些对新 head 都还不成立。

### AttnRes 栈：#4656、#4963、#4765、#4764

- worktree `rbA`，一条线性的链，每个 head 都是下面那个的后代：

| PR | 源 | rebase 后 head | 文件 / 增删（对 `631d544b1`，含下层） |
| --- | --- | --- | --- |
| #4656 | `attnres_review1` `507a2f7d5` | `809fffc42` | 4 个文件，+187/-152 |
| #4963 | `pp_review_optimize` 的 `57790e1c6`、`62b32e5af` | `383b58d9f`（`c7cc2af00`、`383b58d9f`） | 10 个文件，+384/-296 |
| #4765 | review 分支 `pp_offload_review1` 的 `fb5d35b56` | `c9581d655` | 15 个文件，+1302/-309 |
| #4764 | review 分支 `pp_balance_review1` 的 `50ef4dd1d`、`908db8609` | `a336356dd`（`76a49b021`、`a336356dd`） | 17 个文件，+2332/-309 |

- #4765 和 #4764 用的是 review 分支的内容。它比 PR 分支新，叠在 #4963 的 `312bc8144` 上。
- 丢掉了重算提交 `e66a9442b` / `4ae9422db`，以及 #4963 提交的旧拷贝 `5a8163a58`。没有压提交。丢重算提交的理由：
  - `fb5d35b56` 不用它的任何东西，不带它也能干净地应用；
  - 它包的是 eager 的 `_apply_attention_residual`，main 已经换成编译的 `AttentionResidual` 区域；
  - 它按 `SelectiveAC/FullAC/RegionAC` 分支，这几个类已经不存在了。现在 `ActivationRematConfig` 给每个 block 包一层 `torch_remat` checkpoint，而这层不允许嵌套；
  - 10-12 时 #4963 已经把它去掉了。
- 适配：
  - #4656 的 `sharding.py`：main 的 #5806 已经删了 `multimodal_first_layer_sharding_config` 和 `_block_residual_placement`，直接用 main 的文件，#4656 不再改 `sharding.py`。message 删了一句 "The multimodal decoder boundary converts no stack any more..."。message 里还写着 "RegionAC"，main 已经不用这个名字了，没改。
  - #4656 和 #4963 的 `model.py`：#6084 把层循环挪进了 `KimiK3Model.forward_embeddings`，列表的改动跟着挪过去，都是机械的。#4963 把 `forward`、`forward_embeddings` 和多模态 `forward` 三处的 `block_residual_TND` 改成 `blocks_TD: list | None`。
  - #4765 的 `stage.py`：保留 main 的 `_forward_chunk_states` 和 FSDP `set_manual_backward_finalization` 的记账，`saves` 上下文包在 forward 外面。`pp_memory` 字段接在 main 的 `Config` 后面。
  - #4765 和 #4764 的新测试按 #5277 的目录挪到 `tests/unit_tests/cpu/distributed/` 和 `tests/unit_tests/cpu/models/kimi_k3/`。
- **main 已经覆盖的：** #5939（mul+sum）和 10-12 没放进 #4656 的归约提交 `41d67c159` 是同一张图。`weighted_sum_reduction.patch` 不用再管了，10-12 记的 bmm 批量不变的问题也随之消失。
- **要你定的：**
  - #5886 的潜在风险，这次没改代码：`_wait_async_collective_inputs` 只等顶层参数。本机 torch 2.13、aot_eager 下，列表里的 AsyncCollectiveTensor 会丢 `requires_grad`。`631d544b1` 上没有异步张量进列表（`async_op=True` 只在 `cp_mla` 里用，#5807 去掉了模块边界的异步重分布）。要么不动，要么让包装函数遍历容器。
  - #4765 和 main 的 #5090 重叠：main 现在有按 block 的 `ActivationRematConfig.apply(saved_tensors_hooks=...)` 接口，`REMAT.md` 里有 CPU offload 的例子。`activation_storage` 在整个 stage 上装自己的 `remat.saved_tensors_hooks`，remat 关着也能用。要不要改到 main 的接口上，是设计问题。它还在用 `torch.cuda` 的 stream，main 已经改成 `torch.accelerator`（#5450）。
- 本机：
  - `test_activation_storage.py` 加 `test_activation_storage_pool.py`：#4764 head 上 14 passed，#4765 head 上 7 passed；
  - 探针：在 fullgraph、动态、编译过的列表区域外面套 `remat.saved_tensors_hooks`（aot_eager），能打包区域存的张量，梯度一致；
  - K3 的测试在本机 import 不了。
- **GPU 待验证：**
  - #4656（`809fffc42`）：
    - `pytest tests/unit_tests/cpu/models/kimi_k3 -q`；
    - `pytest tests/unit_tests/gpu/models/kimi_k3/test_kimi_k3_attention_residual.py tests/unit_tests/gpu/recipes/test_kimi_attention_residual_override.py -q`；
    - B200 格子 `kimi_k3_mm_muon` 会和 `tests/assets/losses/real_pg/kimi_k3_b200.txt` 做相等断言。每张图的源数变成常量，lowering 会变，golden 很可能要刷新，到时 body 要说明原因；
    - 再跑 `kimi_k3_fsdp2_tp2_ep2_pp2_vpp4`。
  - #4963（`383b58d9f`）：`pytest tests/unit_tests/cpu/models/kimi_k3/test_kimi_k3_pp_block_grads.py tests/unit_tests/cpu/models/kimi_k3/test_kimi_k3_pp_stage.py tests/unit_tests/cpu/models/kimi_k3/test_kimi_k3_pp_layout.py tests/unit_tests/cpu/distributed/test_pipeline_parallel.py -q`。body 里的 `test_kimi_k3_attention_residual_recompute.py` 已经不存在。
  - #4765（`c9581d655`）：`pytest tests/unit_tests/cpu/distributed/test_activation_storage.py tests/unit_tests/cpu/models/kimi_k3/test_kimi_k3_pp_memory.py tests/unit_tests/cpu/models/kimi_k3/test_kimi_k3_pp_stage.py -q`，再确认 offload 在 inductor 编译的区域上能用。
  - #4764（`a336356dd`）：在 #4765 的基础上加 `tests/unit_tests/cpu/distributed/test_activation_storage_pool.py tests/unit_tests/cpu/models/kimi_k3/test_kimi_k3_pp_memory_plan.py`。
  - 数值：
    - #4656 的所有表都要以 `631d544b1` 为参照重测。main 的区域现在是 mul+sum、源数是符号维度，所以"和 main 相同"、第 1 步梯度都要重新测，噪声底用 main 自己的编译对 eager；
    - #4963 的所有表是 `b663c1673` 上测的，也要重测；
    - #4765 和 #4764 没有实测的表。
- **body 要改：**
  - #4765 和 #4764 的 body 描述的还是旧设计（`RankStore(offload=...)`、`kimi_k3/pipeline_stage.py`、`kimi_k3/pp_balance.py` 的 `PPBalanceEngine`），和分支上的代码（`distributed/activation_storage.py`、`pipeline_parallel/activations.py`、`pp_memory.cpu_offload`）对不上，要整篇重写。它们是 "DO NOT review" 的 draft，不急。
  - #4656 和 #4963 的 `Relation to ...` 一节要删。
  - #4656 Summary 第二条 "`kimi_k3/sharding.py` ... converts no stack" 要删。
- **Summary 要点：**
  - #4656：
    - 问题：main 上每打开一个 block 就 `torch.cat` 出一个更长的 `[T, N, D]` stack，每个 block 的 remat checkpoint 都留着它收到的那份 stack。debug flavor 上多出 10 份 `[T, D]`，一直活到反向。
    - 做法：用 `[T, D]` 的列表按引用持有各个 block，所有 checkpoint 共用同一份。
    - 排除的方案：在调用处先 stack 再传，只有一张图，但编译区每次调用都多存一份拷贝（10-12 文档）。
    - 代价：每个源数一张图（debug 5 张、K3 8 张），所以 `recompile_limit=32`。main 自己的其他区域也用 16 或 32。
    - 你查过的：5060 上对 bmm 版的 main，第 1 步 loss 和 grad norm 逐位相同，第 1 步梯度 465 个里 462 个相同（最大相对差 2.2e-8），还有显存和 PP 格子的数（`ATTNRES_4656_4963_ON_5033_2026-10-12.md`）。都在 #5939 之前，对新 base 不成立。
  - #4963：
    - 问题：每个 stage 把收到的 block stack 成一个新的 leaf，另有一份 payload 的 stack，cache 的 view 又钉住 buffer，所以一个 rank 上每个 block 存了好几份。
    - 做法：每个 block 一个张量；前向的 send 在本 stage 对同一个 micro-batch 做反向时才 wait。
    - 你查过的：body 里的 H100 表，以及 10-12 文档里 B200 格子 160/160 逐位相同，都在旧 base 上。
  - #4765：`pp_memory.cpu_offload="all"` 把一个 pipeline stage 存下的张量放到 pinned host 内存。
  - #4764：先跑一步 profile，再据此规划 offload；可以经 Mooncake 的 `RemoteBackend` 把存下的张量停到较轻的 PP rank 上。
