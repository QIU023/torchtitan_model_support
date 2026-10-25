# MiniMax M3 两个 PR 的审核：对照 #4025 的合入标准（2026-10-25，CPU 会话）

用户："审，我准备开两个PR了 ... 我们已有的验收标准达到了4025 JZ当时的那种和它合并的最终diff标准吗？（maintainers的要求）"

对象：
- FSDP2 基线：`minimax_m3` = `de7806dae`（含 prep 提交 P1 `9172a00f1`、P2 `64be06102`、P3 `69dda3713`、P1b `a48507a84`），正文 `PR_BODIES_M3_FSDP_2026-10-24.md`；
- 5D draft：`minimax_m3_5d` = `798f8817f`，正文 `PR_BODY_M3_5D_2026-10-24.md`。

对照物：
- #4025 合入时的正文（GitHub API）；
- 它的合入提交 `5fecad929`（20 个文件，+2752/-34）；
- `LESSONS_4025_MAINTAINER_REVIEW.md` 里维护者的 72 条评论。

两个分支和 upstream main `89e4339c9` 都能干净合并（`git merge-tree` 没有冲突）。

## 结论

- **代码：** #4025 上维护者提过的要求，M3 都做到了。测试和对拍比 #4025 合入时多。
- **正文的证据：** 少了 #4025 合入时有的一项，就是并行验证。而且 HSDP 现在是"写了支持、没跑过"。这一条补上之前，基线不该开成 ready。
- 另有 5 处开 PR 前要处理（第 3 节）。

## 1. 逐项对照

| 项 | #4025 合入时 | M3 基线现在 | 判断 |
|---|---|---|---|
| 复用 titan 组件 | 正文列出复用的类；Kimi 的子类只改不同的地方 | MSA 是 `GQAttention` / `FlexInnerAttention` 的子类；router、`RoutedExperts`、`ClampedSwiGLU`、`VisionTransformerBlock`、`AuxLoss`、DSV3 adapter 的 seam、common 的 dummy 视觉输入、`_register_optimizer_hooks`（`BaseModel` 的 seam）都是复用 | 达到 |
| 和 HF 对拍 | 固定 revision 的 config 缩到 debugmodel；fp32 一行加多种 dtype 的表；路由一致率、末 token KL、top-1 / top-5 | 固定 revision `f0e1c1e04d40`，默认 bf16（shuhuayu 08-23 的口径）加 fp32；选块和路由一致率、末 token KL、全位置 KL；强制 HF 的选择后 4.1e-5；fp32 的残差定位到 titan 的 bf16 grouped GEMM | 超过 |
| 并行验证 | 单卡、FSDP、HSDP 各 200 步的 loss 对比（逐步差的均值 -0.0004，std 0.0027） | 基线正文里没有；只有 4 卡 FSDP 集成格子跑 10 步；HSDP 在 logbook 里一次都没跑过 | **缺** |
| 单测 | 3 个：flex mask、KDA kernel 对递归参考、adapter 往返 | CPU 16 个，adapter 2 个，GPU 2 个（稀疏 flex 对 masked SDPA；KL 梯度只进 indexer） | 超过 |
| GPU 测试进 CI | 改 workflow 才能跑到那个 GPU 测试 | main 现在的 CI 单卡线会跑 `tests/unit_tests/gpu`；M3 的两个是单卡 `unittest` | 达到，不用改 workflow |
| 集成格子 | 加了 `kimi_k3_mm_fsdp`（2 卡） | 加了 `minimax_m3_mm_fsdp`（4 卡；models 套件里本来就有 4 卡和 8 卡的格子） | 达到，和 #4025 的先例一样 |
| README | 架构、发布配置、并行方式、对拍 | 同一模板；问题见第 3 节 | 基本达到 |
| 范围说明 | 正文有 "Current scope" 一节（不支持的并行、视频、compile、packing、generation） | Design 里只有 Parallelism 一条；没写 compile 和性能 | 建议补 |
| 外部改动的说明 | 有 "Changes outside `models/kimi_k3/`" 一节 | 靠 "Stacked on #<P1> #<P2> #<P3>" | 取决于怎么开（第 3 节第 1 条） |
| 注释和 docstring 密度 | 合入的 diff 6.2%（含 license 头） | 7.1%（含 license 头和 prep 提交） | 相当 |

维护者在 #4025 上提过的其他要求，逐条看过，都满足：
- 层号从 0 起；
- debug recipe 用默认并行（只开 FSDP）；
- 注意力后端支持 flex；
- 支持样本打包；
- indexer 在第一个 PR 里就有 KL 梯度，不会出现 DCP 加载时缺 optimizer state 的问题；
- 纯文本 batch 也跑 dummy 视觉前向，FSDP 的 collective 次数一致；
- README 写明对拍用的是缩小后的 HF config；
- 有形状后缀的说明；
- MoE 走 `RoutedExperts`，量化 override 能生效。

## 2. 必须补（基线开成 ready 之前）

1. **并行验证和 HSDP：**
   - 正文 Design 写了 "FSDP2 and HSDP"，README 写了 "FSDP2 / HSDP Supported"，5D 正文也写了 "composes FSDP2/HSDP"，但没有任何一次 HSDP 运行。
   - #4025 合入时给了单卡、FSDP、HSDP 三条 loss。
   - 按 logbook 的数值表规则，dp1 和 dp2 是不同的数据流，不能配对。建议在同一条数据流上配：
     - fsdp4 对 HSDP 2 x 2（都是 dp4）；
     - 再加一行 noise floor；
     - 报第 1 / 3 / 10 步的 loss 和 grad norm。
     - 第 1 步 loss 应逐位相同；梯度的差应在规约顺序的量级。
   - 跑不了的话，就把三处的 HSDP 去掉。
2. **CPU 全量：**
   - 正文的 CPU 命令只跑了指定的几个文件，CI 会跑整个 CPU 套件。
   - 开 PR 前在两个分支上各跑一遍全量，和 main 比，只在分支上失败的应该是 0（10-13 栈验证的做法）。
   - P1、P2 动了 Qwen3.5 / 3.6 / 3.8，P4、P5 动了 Qwen3.5、Kimi K2.7、Kimi K3 的 sharding，这些模型自己的测试都要在全量里。

## 3. 开 PR 前要处理

1. **只开两个 PR 的话，正文要改：**
   - 现在的写法是先开三个 prep PR（P1 + P1b、P2、P3），M3 的 Summary 写 "Stacked on #<P1> ... #<P2> ... #<P3>"；5D 正文第一行写 "The first five commits are #<FSDP2> and its three prep PRs"。
   - 如果 prep 提交留在基线 PR 里，就要照 #4025 加一节 "Changes outside `models/minimax_m3/`"，四个提交各一条（P2 改了 Qwen3.5 的 attention，要写上前后 loss 每步相同的那组检查）。
   - 5D 那句也要改。
   - P4、P5 同理：要么单独开，要么在 5D 正文里说明。
2. **flex 的 inductor bug：**
   - 正文 Design 和代码注释都写了 "flex backward autotuning binds two or more captured tensors in a different order than its kernel reads them"。这是对 PyTorch 的一个断言，可是 issue 和修复 PR 都还没发。
   - drisspg 是 flex 的作者，也在 titan 评审。
   - 建议先用 `flex_bwd_capture_order/repro_flex_bwd_capture_order.py` 发 PyTorch issue，正文那一句链接过去。不发的话就把这句改成只讲约束，比如 "the mask captures one tensor"。
   - 发 issue 是对外动作，由用户定。
3. **"only the index projections"：**
   - README 和 `MiniMaxSparseAttention` 的 docstring 都这么写，但 KL 也训两个 index norm（正文 Design 已经写成 "`wq`, `wk` and their norms"）。
   - README 是同一个提交，所以改 docstring 时一起改。
   - 两个分支都没有 PR，amend `de7806da` 再把 5D 的三个提交 rebase 上去即可（force-push 前先推备份）。
4. **范围说明：**
   - 建议照 #4025 在正文加一节 "Current scope"：
     - 支持：eager、FSDP2；
     - 不支持：TP / CP / EP / PP；
     - 视频支持到哪；
     - compile 的情况；
     - 选块和 KL 是 PyTorch 按 128 个 query 一块循环算的（`select_msa_blocks`、`MSAIndexerLoss.forward`），有 host 同步（每个 batch 建 metadata 时一次 `.tolist()`，KL 每层一次 `nonzero().tolist()`，每个 tile 一次 `torch.tensor(...)`），融合 kernel 是后续工作。
   - 维护者在 #4025 上要过 "training ready version using kernels"，先写明比被问到好。
5. **README 的两处（可选）：**
   - 有一节 "Indexer Warmup" 讲 `build_model_config(..., sparse_attention=False)` 怎么用，还有一句 M3.1-Flash 变体。shuhuayu 在 #4025 说过 README 不列变体、不写怎么运行。
   - main 的 K3 README 也有一节 Prerequisites，这条不算硬。

## 4. 5D draft

- 正文按规则写：数据流标明了，有 noise floor，报第 1 / 3 / 10 步，PP / EP / TP 的差都定位了，CP 不支持的原因也写了。作为 `[DO NOT MERGE]` 的占位可以开。
- 要跟着基线改的：
  - HSDP（第 2 节第 1 条）；
  - 第一行的 `#<FSDP2>` 和 prep PR 的说法（第 3 节第 1 条）。
- PP 进不进 CI，仍然要用户定。

## 5. 没做的

- 本机（Windows）跑不了 titan 的 CPU 测试（main 的 `models/common` 要 CuTeDSL），所以没有重跑任何测试，上面的测试数都来自 GPU 会话的记录。
- 代码逐行看了 `minimax_m3/attention.py` 全文、`model.py` 和 `vision_encoder.py` 的相关部分。
  - 用脚本统计了注释和 docstring 的密度，列了私有 `def`，查了 logbook 路径、hook、全局状态、换 `__class__`，都没有问题。
  - 私有 `def` 里，`_apply_vision_rope` 是手写的部分旋转。同一个分支的 P2 已经让 `RoPE.forward` 支持部分旋转，评审可能会问为什么视觉这里不用；视觉塔走的是 common block 的 `rope_apply` 回调，P2 只改了模块的 `forward`，没改静态的 `apply_rotary_emb`。
