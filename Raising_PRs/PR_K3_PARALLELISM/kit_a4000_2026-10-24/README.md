# A4000 环境与 K3 CI 格子（2026-10-24）

在 8 × RTX A4000 16 GB（sm86，PCIe，driver 580 / CUDA 13.0）上搭环境，用 main 的 `run_tests` 跑 Kimi K3 的五个 CI 格子，看这台机器能不能用。MiniMax M3 的工作（`Raising_PRs/RFC_MINIMAX_M3/`）也在这个环境里做。实例的 `/workspace` 不是 volume，销毁前把结果备份到这里。

- 树：main `3ed5aa97`；torch `2.16.0.dev20261023+cu130`（`/workspace/venv_1024`）。
- 脚本：
  - `setup.sh`：建 venv_1024。
  - `run_k3_ci_cells.sh`：五个格子，冒烟，默认 cache。
  - `shims/sitecustomize.py`：sm86 的两个 shim，只在设了 `RANK` 的训练进程里生效。KDA gate 走 attn-gym 的 reference 实现；FA2 前把 MLA 的 V 补零到 Q/K 的头宽，输出再切回。
  - `check_pad_v.py`：V 补零后的 varlen 对照 SDPA，比输出和 q/k/v 梯度。运行：`PYTHONPATH=shims RANK=0 python check_pad_v.py`。
  - `nccl_smoke.py`：NCCL all_reduce 冒烟，加 128 MiB 的带宽。运行：`torchrun --nproc_per_node 8 nccl_smoke.py`。
- `env/`：venv_1024 的 `pip freeze`，和 `setup.sh` 写的 torch 版本锁。
- `results/k3_ci_a4000_text.tar.xz`：三轮的日志、progress 和训练指标 jsonl，不含 checkpoint。解开是 `k3_ci_1024/`、`k3_ci_1024_r2/`、`k3_ci_1024_r3_pp/`。

## 结果

- 第一轮，不带 shim：`kimi_k3_mm_muon`、`kimi_k3_mm`、`kimi_k3_mm_allgather_kv_cp`、`kimi_k3_mm_ulysses_cp` 都挂在 `gate_transform requires CUDA capability >= 9.0`。PP 格子的日志停在启动阶段，没有结束行。
- 第二轮，带 KDA gate 的 shim：
  - `kimi_k3_mm`、`kimi_k3_mm_allgather_kv_cp`、`kimi_k3_mm_ulysses_cp` 跑完 10 步，rc=0。
  - `kimi_k3_mm_muon` 跑完 10 步，但第 1 步的 loss 和 B200 golden 不相等（baseline 7.982885，imported 7.983041），`--assert-equal` 失败。
  - `kimi_k3_fsdp2_tp2_ep2_pp2_vpp4` 挂在 `v must have shape (total_k, num_heads_k, head_size_og)`。sm86 上 varlen 走 FA2，FA2 的 Q/K/V 只能是同一个头宽。
- 第三轮，再加 V 补零，只跑 PP 格子：rc=0，最后一个 stage 第 10 步 loss 3.62684，grad norm 2.9844。V 补零是 07:09 加进 `sitecustomize.py` 的，第三轮 07:10 开跑。
