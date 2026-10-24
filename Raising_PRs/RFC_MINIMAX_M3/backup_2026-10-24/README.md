# A4000 实例上 M3 工作的备份（2026-10-24）

实例的 `/workspace` 不是 volume，销毁前把只在本机的运行输出和 scratch 脚本备份到这里。解包：`tar -xJf <文件>`。

- `m3_runs_text.tar.xz`（319 个文件）：原 `/workspace/m3_runs/` 里的日志、progress、`.res` 和汇总，解开是 `m3_runs/...`。`M3_5D_2026-10-24.md` 和 `PR_BODY_M3_5D_2026-10-24.md` 引用的 `/workspace/m3_runs/text18_1024_1121` 就是其中的 `m3_runs/text18_1024_1121/`。
- `scratch_runs_text.tar.xz`（116 个文件）：会话 scratchpad 里的运行输出。
  - `p2_check/`：P2 前后 Qwen3.5 debug recipe 的 loss 对比，P2 正文 Test plan 里 "the loss is identical on every step" 那条的依据。
  - `m3/flex_dyn/`：2 卡冷 cache 下 flex 反向 capture 顺序问题的复现和 memcheck，以及动态重编译前后的梯度对比。
  - `m3/flex_upstream/`：打上 `patch_bwd_autotune_inputs.py` 之后的复跑。
  - `m3/tests_1024/`：两个分支复核时的测试输出和 bf16 对拍。
  - `m3/vlm_before.txt`、`m3/vlm_after.txt`：P4/P5 前后 Qwen3.5、Kimi K2.7、Kimi K3 debug config 的哈希（`scratch_scripts/m3/dump_vlm_configs.py`），前后相同。
- `scratch_scripts/`：scratchpad 里还没进 logbook 的 27 个脚本，路径照原样。内容相同、已经在 `kit_m3_5d_2026-10-24/`、`hf_parity_probes/`、`flex_bwd_capture_order/` 里的脚本没有重复放。
- 没有备份：
  - inductor/triton cache、compile trace、seed checkpoint 和 `.pt` 梯度 dump（`m3_runs` 里约 5 GB，scratchpad 里约 8 GB）。用 `kit_m3_5d_2026-10-24/` 的脚本，在同一个 nightly（`PR_K3_PARALLELISM/kit_a4000_2026-10-24/env/`）上可以重跑出来。
  - 第三方的下载和代码副本：HF 的 config、README、tokenizer 和权重 index，MSA 论文全文，PR-4025 的 API 导出，Megatron-Bridge、SGLang 和 GLM-5 的代码。都可以从原处重新获取。
