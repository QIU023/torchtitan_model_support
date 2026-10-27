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

## 第二次审核（2026-10-26，CPU 会话）

用户："重新拉取，审核Minimax M3 FSDP2基线body和diff，以及4D body和diff"。

对象：
- 基线：`minimax_m3` = `9a5ab9ebe`；正文是 `PR_BODIES_M3_FSDP_2026-10-24.md` 的 M3 粘贴区，数字来自 4 x H100 NVL。
- 4D：`minimax_m3_5d` = `89fc094d1`，已开成 #5619（draft，线上 body 是 10-25 的短版）。文件里的粘贴区已经扩成 4 张表（8 x RTX 5060 Ti），还没贴上去。

### 开基线 PR 之前要做

1. **和 main 冲突：**
   - main 10-27 合了 #5583（Tied RoPE cache），在 `models/common/decoder.py` 加了一行 `RoPE` 的 import，和 P1b 加 `OffsetRMSNorm` import 的位置冲突。两个分支都这样。
   - 冲突本身只要两行都留。#5583 会让参数相同的 RoPE 共用一份 cache；M3 的 indexer 用的就是注意力自己的 RoPE，所以会被共享，按 #5583 的说法数值不变。
   - rebase 后要重跑：CPU 全量（main 新加了 `test_rope_cache_tying.py`）、M3 的 GPU 单测、集成格子；再用一次运行确认对拍表和 HSDP 表的第 1 步没变。
   - 推送规则：`minimax_m3_5d` 现在是 #5619 的 PR 分支，按 CLAUDE.md 要先推 review 分支、用户点头后再同步，而新开 review 分支要先问用户。`minimax_m3` 还没有 PR，备份后直接 force-push 即可。
2. **4D 正文（文件版）第一段过时了：**
   - 现在写的是 "which open as their own PRs once the full parallelism matrix has run"，但矩阵已经跑完；
   - 基线也已改成一个 PR，"PRs" 要改成单数；
   - 基线开了以后，换成它的编号。
3. **4D 正文 PP 那条只写了一半：**
   - 现在只写了动态形状这一个原因；
   - 10-24 的定位里，两边都用静态形状之后，还剩 46 个 dense 投影权重的梯度不逐位，原因是 PP 用另一种顺序累加各个 micro-batch 的 bf16 梯度，未改动的 DeepSeek V3 也一样（`M3_5D_2026-10-24.md`）；
   - 按数值验收规则，差距要整个定位，所以补一条。
4. **4D 正文 TP 那句因果写反了：**
   - 原句 "TP layouts differ from step 1 through the MSA block selections and expert choices"；
   - 最早的差异其实来自 row-parallel 规约的顺序，它改变了舍入，进而翻转了一部分选块和路由。下面那条 bullet 也是这么说的：强制选择后还剩 5.4e-3，那就是规约顺序。
   - 建议改成先说规约顺序、再说翻转。

### 建议改

5. **基线 Results 那句 "HSDP matches FSDP bitwise through step 3"：**
   - 下面紧接着说第 1 步梯度有 33 个参数不同；
   - 建议写成 "HSDP's loss and grad norm match FSDP's bitwise through step 3"，免得读成矛盾。
6. **4D 表后面几步漂移很大：**
   - 第 50 步 grad norm 最多差 22%（ep2 x fsdp2），但每张表的 noise floor 只是同一个格子跑两次（全同），没有"同一份数据、换一种规约顺序"的那种底。
   - #4312 的教训是评审会把后面几步的差读成 bug。
   - 可以加一句，指向基线 PR 的 HSDP 那行作为规约顺序的量级（第 50 步 grad norm -3.43%）；或者把 4D 表收成只报第 1、10 步。
   - draft 不急，转正式前要定。

### 代码层面评审可能会问（不挡开 PR）

- P2 按宽度隐式触发部分旋转（`query.shape[-1] > self.config.dim`）：原来宽度不对会直接报错，现在会静默地只转前 `dim` 个通道。评审可能更想要显式的配置或断言。
- `MiniMaxM3StateDictAdapter` 继承 DeepSeek V3 的 adapter，`super().__init__` 上要加 `pyrefly: ignore[bad-argument-type]`。先例是 DeepSeek V4 和 Kimi K2.7 也继承它，所以说得过去。
- P4 移进 common 的 `transformer_block_sharding_config`，和 Muse Spark 自己的一个同名函数内容不同。
- `_apply_vision_rope` 手写了部分旋转，而同一个 PR 的 P2 让 `RoPE.forward` 支持部分旋转（上一轮已记）。
- 对拍脚本 602 行，GLM-5 的是 276 行。`--force-hf-*` 那些诊断功能可以删。

### 核对无误的

- 代码：
  - 4 个 prep 提交逐行看了（P2 的 `_partial_forward` 就是 Qwen3.5 原来的 `_partial_rope` 挪进 `RoPE`，编译区名字不变；P3 只加了 `_build_from_hf_map` 和 `hf_rope_config_cls` 两个 seam）；
  - `model.py`、`attention.py`、`vision_encoder.py`、`state_dict_adapter.py` 全文看了：都用 common 的多模态工具和 dummy 视觉前向，没有 hook，没有全局状态，也没有换 `__class__`。
- 测试：CPU 测试覆盖选块和 KL 对参考实现、dense 模式、flavor、发布 checkpoint 的参数量；GPU 测试覆盖稀疏 flex 对 SDPA、KL 梯度只进 indexer。对拍脚本固定 revision `f0e1c1e`，默认 bf16。
- 正文：
  - 两份都没有 we/our/us，没有破折号，全部是 ASCII；基线正文 1004 词，4D 597 词（都不算表格）；
  - 4D 表里一共 29 种布局，和正文说的一致；
  - kit 链接指向的 logbook 仓库是公开的，链接能打开。

## 第三次审核（2026-10-27，CPU 会话）

用户："拉取 基线body删除了不用的版本，再次审核"。

对象：
- 基线 `minimax_m3` = `a7db544ff`，base `e111f10d4`。upstream main 现在是 `13806cbae`（只多了 #5634，改 `qwen3/flavors.py`），合并没有冲突。
- 正文：`PR_BODIES_M3_FSDP_2026-10-24.md`，标题第 118 行，粘贴区第 120 到 203 行。

### 开 PR 前要改

1. **README 的对拍数字和正文不是同一次运行：**
   - README（在 M3 提交里）写的是 10-24 A4000 那次的数：末 token KL 8.43e-6，选块 94.4%，路由 98.3%，强制 HF 选择后全位置 KL 4.10e-5（`PARITY_HF_2026-10-24.md` 表的前两行）。
   - 正文 Results 第一张表是 10-26 H100 的数：1.50e-5、93.8%、97.9%、4.07e-5。
   - 评审对照 diff 和正文，会看到同一个测试有两组数。README 应该引用正文那一次，四个数换成 H100 的。像素 4.8e-7 和视觉特征 6.1e-5 两次相同，不用改。
   - 做法：amend M3 提交，先把 `a7db544ff` 推成 backup，再 force-with-lease `minimax_m3`（还没有 PR）。只改 README，不用重跑测试。
   - 5D 分支的 README 也是旧数（3.40e-5 / 92.2% / 95.5% / 4.86e-5，视觉特征 1.2e-4，和 #5619 旧正文是同一次），但文件里的新 5D 正文写的是 5060 的 2.21e-5 / 4.74e-5。5D 是 draft，可以等下次同步 #5619 时一起改（走 review 分支）。

### 建议改（可选）

2. **fp32 两行的末 token KL 是负数**（-9.8e-7、-1.9e-7）：
   - KL 本身不会是负的。这是 fp32 下两个分布几乎相同时的舍入，评审可能会问；
   - 可以在 fp32 那条 bullet 里加半句，说明这个 KL 在舍入的量级，所以读出来略小于 0；或者表里改写成 "< 1e-6"。
3. **"Pixels match to 4.8e-7." 没说比的是什么：**
   - 脚本比的是 titan 和 HF 处理器各自预处理出来的 pixel values；
   - 可以写成 "The image preprocessing matches HF's processor to 4.8e-7"，再把视觉特征（bf16 6.1e-5，fp32 1.2e-8）并进同一条。
4. **Results 的两段设置说明各有两三句：**
   - 按 10-07 的规则（"any other body or reply section that makes more than one point"），可以拆成 bullet；
   - 但 #4312 那种 "一段说明加一张表" 的格式用户一直接受，这条由用户定。
5. **README 的 "Indexer Warmup" 一节和 M3.1-Flash 那段还在**（第一次审核第 3 节第 5 条）：shuhuayu 在 #4025 说过 README 不列变体、不写怎么运行。可选。

### 核对无误的

- **正文：**
  - 对拍表 4 行，和 H100 原始输出（`kit_m3_5060_2026-10-25/results_h100/runs/parity/*.txt`）逐个对上；
  - "Changes outside" 里 P2 的检查：H100 上 dense 用 1 卡、MoE 用 tp2 x pp2 x ep2 4 卡，各自先预热，loss_compare 全等（`results_h100/runs/p2`）；
  - "eleven `bad-argument-type` ignores"：P1b 正好删了 11 行；
  - Test plan：16 + 2 个测试和 compare 文件对上；集成测试命令的参数名和 `run_tests.py` 一致；
  - pre-commit 那条是 rebase 前跑的。rebase 只改了一行 import，而且 `import av` 上的 ignore 在 main `13806cbae` 上还在，所以 "as on main" 仍然成立。
  - 数值表规则：参照 fsdp4 的 loss 在第 20 步左右进入 2.6 到 3.0 的平台，没有记住数据集的迹象，报第 50、100 步不违反规则；
  - 没有 we/our/us，没有破折号，全部是 ASCII。
- **代码：**
  - `git range-diff` 看，只有 P1b 的 import 上下文不同；
  - 第一次审核提的两处已经改了：docstring 和 README 都写了 "and their norms"；flex inductor 那句已经删掉。
  - 新加的 10 处 pyrefly ignore 在 main 上都有同样的先例：`re.search(...).group(0)` 上的 `missing-attribute`（DSV3、flux 的 adapter），`forward` 的 `bad-override`，NamedTuple 字段的 `noqa: N815`（common 的 MoE）。
  - diff 里没有 logbook 路径、硬件名或实验记录；纯注释行占新增行的 3.6%；唯一的 TODO 和 Qwen3.5 在 main 上的那句相同。
  - 第二次审核"代码层面"那几条（P2 按宽度隐式触发、adapter 的 ignore、602 行的对拍脚本）没有变化，仍然不挡开 PR。
