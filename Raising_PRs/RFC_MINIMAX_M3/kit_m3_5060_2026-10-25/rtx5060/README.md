# 8 x RTX 5060 Ti kit（2026-10-26，本机）

用户 10-26 定："4D全部数值只用5060"。4D 正文里的每个数字都来自这一轮，结果在 `../results_5060/runs/`。

- 环境：venv `/workspace/venv_1025`（torch 2.16.0.dev20261024+cu130，和 H100、H200 用的是同一个 nightly）；5D 树 `/workspace/wt_m3_5d` = `89fc094d1`，没有未提交的改动。
- 脚本和 H100 的 kit（`../h100/`）逐字节相同，只有 `env_h100.sh` 换成了本机路径（文件名沿用 H100 那套，脚本都按这个名字 source 它）。
- 运行顺序（`chain_4d_5060.sh`）：
  1. `run_m3_4d_h100.sh`：只用 FSDP，29 个布局，`c4_test` 跑 100 步。一个单卡 seed，一份 10 步预热的 cache；正式运行用 cache 的副本，四个参照（dp1、fsdp2、fsdp4、fsdp8）各再跑一次。
  2. `tables_4312.py`（4312 格式的表）、`summarize_m4d.py`（第 1 步的 loss 和梯度对参照）、`ep_param_diff.py`（EP 改动了哪些梯度）。
  3. `run_tp_force_s100.sh`：在同一份 cache 上记录单卡的选块和路由，再强制 tp2 x ep2 和 tp4 x ep4 用这份记录。
  4. 集成格子 `minimax_m3_mm_fsdp+tp+sp+ep`（4 卡，`--no-parallel`），M3 的 GPU 单测，以及 5D 树上的 HF 对拍（bf16，不强制和强制 HF 的选择各一次）。
- `shim/sitecustomize.py` 用来关掉 FA 选择（`M3_FA_OFF=1`，5060 上需要）。
