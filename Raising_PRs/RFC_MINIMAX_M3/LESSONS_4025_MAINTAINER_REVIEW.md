# #4025 维护者评审给 M3 eager PR 的清单（2026-10-24 整理）

只收维护者（tianyu-l、shuhuayu、drisspg）在 pytorch/torchtitan#4025（"[kimi k3] add eager reference model with FSDP2"，JavaZeroo，07-30 开、08-24 合入）上的 72 条评论；我们自己和 claude[bot] 的评论不算。抓取：GitHub API（review comments、issue comments、review summaries），全文在 scratchpad `m3/pr4025/maintainer_comments.txt`。每条后面是 `[日期 评审人 文件:行]`。

## 0. 定性

- tianyu-l 第一次评审直接 CHANGES_REQUESTED："seems an AI-slopped implementation" [08-09 tianyu-l]。
- 作者的解释是："many modules were directly based on modeling_kimi_k3.py instead of reusing some titian fused modules"；tianyu 的回复是这次 M3 最要记住的一句："My worry is that if we use slow HF implementation without torchtitan components during numerical verification; we'd have to do it again when swapping to titan modules." [08-09 tianyu-l]
- 所以：**HF 实现只当数值参考，不当代码模板**；数值验证必须在用 titan 组件写好的实现上做，不能先移植 HF、对完数再换。

## 1. 复用 titan 模块（不要照搬 HF）

- "should use our fused feed forward" [08-09 tianyu-l model.py:116]
- "Is this from DeepSeek V3 MLA? If not, we have an existing impl in torchtitan." [08-09 tianyu-l model.py:138]
- "should use torchtitan grouped mm" [08-09 tianyu-l model.py:398]；"should switch to `self._grouped_mm()` so overrides can take effect when quantization is applied." [08-14 shuhuayu model.py:393]；算子的选择在 `common/moe.py`，drisspg 补充那里应该用公开的 `F.grouped_mm` [08-14 至 08-16 drisspg]
- "Can you reuse more common modules in torchtitan today, like `MoonViT` in `common/vision_encoder`, `common/multi_modal`, and reuse MoE router." [07-31 shuhuayu]
- 视觉塔："can we reuse the same methods in kimi 2.7?"（同一文件 5 处）[08-22 至 08-23 shuhuayu vision_encoder.py:39/55/102/129/161]；"can reuse `VisionTransformerBlock` in the `common` folder with additional configs." [08-23 shuhuayu vision_encoder.py:206]
- 训练用的 kernel："we should target for a training ready version using kernels for kda ... put the pytorch native reference implementation into the numerical tests" [07-31 shuhuayu]

M3 对应：注意力用 common 的 GQA 注意力；FFN 用 common 的 fused feed forward；MoE 用 common 的 router 和 GroupedExperts（经 `_grouped_mm`，量化 override 才能生效）；clamped SwiGLU 用 `common/activation.py` 已有的；视觉塔用 `common` 的 `VisionTransformerBlock` 加配置项，参考同类模型而不是 HF；MSA 的稀疏注意力用已有的 kernel 接口（attn-gym `gather_attn` 或 Flex），纯 PyTorch 的 MSA 参考实现只放在数值测试里。

## 2. 数值对齐测试

- "this pr should include a reproducible numerical parity check test, likely using some reduced hf reference, since the original hf ckpt is too big." [07-31 shuhuayu]
- 做法：用 `snapshot_download` 下载官方 config，`AutoConfig.from_pretrained(...)`，再把 config 缩到和本地 debug model 一致，然后建 HF 模型对比 [08-13 shuhuayu]；"we should not rely on local hf assets, to make it reproducible, use `snapshot_download()` with the pinned revision." [08-14 shuhuayu numerical_tests_kimi_k3.py:168]
- 形式参考已有脚本："is this trying to achieve the similar to `scripts/checkpoint_conversion/numerical_tests_kimi.py` or not?" [08-09 tianyu-l]
- 精度："titan's gemm only supports bf16, so we can simply use bf16, no fp32." [08-23 shuhuayu]；默认 bf16 [08-21 shuhuayu]；小心把 FP32 的 RoPE buffer 转成 BF16 的 bug [08-24 shuhuayu]
- README 写清楚是缩小后的 HF config："people may think this is the actual k3 config." [08-23 shuhuayu]

M3 对应：transformers 已有原生 `minimax_m3_vl`，不需要 trust_remote_code；脚本放 `scripts/checkpoint_conversion/numerical_tests_minimax_m3.py`，`snapshot_download` 固定 revision 拿 config，缩到 debug 尺寸；序列长度要超过 `topk_blocks * block_size`，才能覆盖到稀疏选块。

## 3. 测试

- 用缩小的 config 测三件事：state dict adapter、前向和反向、FSDP 和非分布式的一致性；kernel 单独用张量测，不需要模型 config [08-14 shuhuayu test_kimi_k3.py:1]
- "can we test a real backward gradient parity?" [08-14 shuhuayu]
- 用模型自己的 config 构造函数造小 config [08-14 shuhuayu]
- "why we need specific tests for Kimi K3 + FSDP?" [08-09 tianyu-l]；测试名要反映测的是什么 [08-09 tianyu-l]
- 不跑的测试要删："this would be not run as a cpu unit test ... removing it and keep this test simple" [08-23 shuhuayu]；"this test is not specific to kimi k3, so i would suggest removing it." [08-23 shuhuayu]
- GPU 测试默认会被 CI 忽略，要在 workflow 里接上 [08-23 shuhuayu]

## 4. 范围和配置

- "does this change belong to this PR?" [08-09 tianyu-l test_multimodal.py]；"I don't think we need to do anything to support this, do we?" [08-09 tianyu-l parallelize.py:42]
- debug recipe 不要写 `dp_shard=1`，"remove the parallelism setup and use all default values, which is fsdp only." [08-13 shuhuayu]
- 不必沿用 HF 文件的约定，层号用 0 起 [08-14 shuhuayu]
- 注意力后端不能只有 eager，"we need to support flex as the attention backend"，mask 和 attention metadata 参考 #3984 [08-14 shuhuayu]
- 打包样本（sample packing）不支持就加 TODO，并用现在的检查方式（`MMSamplePackingConfig` 时报错）[08-21 shuhuayu]
- 一直拿不到梯度的参数会让 DCP 加载 optimizer state 出问题（K3 是第 0 层的两个参数）[08-20 shuhuayu]
- 集成测试的格子要等基础设施 PR 落地 [08-09 tianyu-l]

M3 对应：**MSA 的 indexer 参数只从 KL loss 拿梯度**，第一个 PR 不带 KL loss 的话它们永远没有梯度，正好踩第 4 节这个坑。所以 KL loss 应该进第一个 PR（或者给出明确的冻结方案，并说明 DCP 怎么处理）。

## 5. 多模态和 FSDP

- 单独包成 FSDP 单元的视觉塔在不同 DP rank 上不能各自跳过：一个 rank 有图、另一个只有文本时，两边执行的 FSDP collective 不一样，前向反向会卡死（shuhuayu 两卡复现过）[08-14 shuhuayu parallelize.py:76]；tianyu 希望和其他多模态模型一起解决 [08-09 tianyu-l fsdp.py:107]

M3 对应：照 main 上现在多模态模型（K3、Qwen3.5、Muse）处理视觉塔条件执行的方式来，不要自己写。

## 6. 文档和代码风格

- README："too much slop in this README" [08-09 tianyu-l]；只写架构上的主要创新、完整模型的主要配置、支持的并行方式、数值对齐的结果和参考脚本；不列变体，不写怎么运行 [08-14 shuhuayu]
- 多处 "nit: remove" [08-13 至 08-21 shuhuayu]；"Is this necessary or just slop?" [08-09 tianyu-l model.py:44]
- 形状后缀要准确："`probs_T1N` is more accurate." [08-23 shuhuayu]
- 配置字段名和发布的 config 一致时可以保留，但要注释说明含义 [08-20 shuhuayu]
