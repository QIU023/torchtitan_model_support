# H100 kit（2026-10-26，`root@87.116.91.146 -p 11468`）

4 x H100 NVL 95 GB（vast.ai），driver 595.91（CUDA 13.2），venv 是 `/root/m3/venv`（torch 2.16.0.dev20261024+cu130，和 5060、H200 用的是同一个 nightly）。titan 的两个树：`wt_m3` = `9a5ab9ebe`（FSDP2 基线），`wt_m3_5d` = `89fc094d1`（5D/4D 草稿）；main 用的是 `tt_main` = upstream `3ed5aa976`。结果在 `../results_h100/runs/`。

用户 10-26 的要求：基线正文只用这台机器的数；4D 正文只用 5060 的数，这台另外把 4D 里 4 卡能跑的组合跑一遍。

按顺序跑过的脚本：

1. `setup_h100.sh`：clone 仓库，建 worktree 和 venv。
2. `chain_fsdp2_h100.sh`，基线正文引用的全部内容：
   - `run_cpu_h100.sh`：CPU 全量，main 和基线同时跑。基线有 8 个 `test_parallelism_context` 的测试失败，原因是两份抢同一个写死的端口 12356（EADDRINUSE）；
   - `run_hsdp_h100.sh` + `tables_4312.py`：dp4 表，`c4_test` 跑 100 步，数据和 H200 逐位相同；
   - `run_hsdp_rebuild_h100.sh` + `reduce_emulate.py`：按规约顺序逐位重建；
   - `run_parity_h100.sh`：HF 对拍，4 种组合；
   - `run_tests_fsdp2_h100.sh`：GPU 单测，加上集成格子 `minimax_m3_mm_fsdp`；
   - `run_p2_h100.sh` + `p2cfg/`：P2 的 Qwen3.5 检查，dense 用 1 卡，MoE 用 4 卡（tp 2 x pp 2 x ep 2）；
   - `run_precommit_h100.sh`：pre-commit。
3. `run_cpu_baseline_alone_h100.sh`：基线 CPU 全量单独重跑，结果 1931 passed，只在分支上失败的是 0。
4. `run_pyrefly_probe_h100.sh`：查清 Pyrefly 改写的是哪一行。
5. `chain_4d_h100_diag_first.sh`：
   - `run_diag_h100.sh`：10-24 的 fp32 逐模块探针（`hf_parity_probes/`）；
   - 然后是 `chain_4d_h100.sh`：`run_m3_4d_h100.sh`（只用 FSDP，4 卡以内的格子，100 步）、`ep_param_diff.py`，以及 `run_tp_force_s100.sh`。

`chain_h100.sh`、`chain_s100.sh` 是 H200 时期留在 kit 里的，这台机器上没有跑过。`shim/sitecustomize.py` 用来关掉 FA 选择（`M3_FA_OFF=1`）。

## 4D 的 4 卡组合（10-26 结果）

- 15 个格子，加上 3 个参照再跑一次，共 18 次 100 步的正式运行。全部 rank 的 54 个 loss 文件和 H200 那次（`../results_h200/runs/m4d_s100/`）逐字节相同；`table_4312.md` 和 `summary.md`（包括第 1 步的梯度对比）也完全相同。需要 8 卡的 14 个格子跳过了。
- EP（`ep_param_diff.txt`）：ep2 x fsdp2 和 ep4 x fsdp4 都只有 14/159 个梯度和参照不同，全部是 routed experts 的 w13 和 w2。
- TP 强制实验（`tpforce/`）：带 hook 的单卡记录和正式运行的 dp1 梯度 159/159 逐位相同。强制用单卡的选块和路由后，最差一组的相对 L2 差变化如下：
  - tp2 x ep2：从 2.70e-2 降到 4.50e-3；
  - tp4 x ep4：从 4.36e-2 降到 4.97e-3。
- 探针不打印"已强制"的提示，但强制前后的梯度不同（同一份 seed、同样的数据、同一份 cache），说明替换确实生效了。
- 用户定的是 4D 正文只用 5060 的数，所以这批结果目前不进任何正文。
