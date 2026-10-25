# PyTorch issue 草稿：flex_attention backward autotune 的捕获张量顺序（2026-10-25）

## 状态（不粘贴）

- 对应 goal 第 4 条：发 issue 是对外动作，等用户点头。没发之前，M3 的代码注释和基线正文都只讲约束（"mask_mod 只捕获一个张量"），不提 PyTorch 的 bug。
- 10-25 在本机（8 x RTX 5060 Ti，sm_120，venv_1025：torch 2.16.0.dev20261024+cu130，triton 3.9.0）重新核对：
  - 装的 nightly 里 forward lowering 已经用 `list(choices[0].input_nodes)` 做 autotune 输入，backward lowering 还是 `bwd_input_nodes + score_mod_other_buffers + mask_mod_other_buffers + mutated_grads`（`torch/_inductor/kernel/flex/flex_attention.py`，`autotune_select_algorithm("flex_attention_backward", ...)` 那一处）。
  - 冷 cache 上跑 `spy.py repro_flex_bwd_capture_order.py`：两个 backward choice 的 benchmark 都是 `in_ptr args: [(2048,), (2048, 4)]`，kernel 签名是 `[(2048, 4), (2048,)]`，错位。
  - 加上 `patch_bwd_autotune_inputs.py`（backward 也改用 `list(choices[0].input_nodes)`）再跑：两边都是 `[(2048, 4), (2048,)]`。
  - 草稿最后一句（block mask 张量在生成的输入里位置不变，`input_gen_fns` 不用动）在复现上核过：backward 两份输入列表前 16 个是同一批 IR 节点（16 / 16），`input_gen_fns` 的键是 8 到 15；两份列表都是 18 个。
  - A4000 上（2.16.0.dev20261023）的原始现象和 memcheck 记录在 `../PARITY_HF_2026-10-24.md`。
- 查重（GitHub 搜索，10-25）：
  - 没有 backward 的 issue 或 PR。
  - forward 和 decoding 的同一个问题是 drisspg 的 #194555（"[inductor] Fix FlexAttention autotuning with inlined captured buffers"，08-24 合入），它只改了 forward 和 decoding。草稿里引用它。
- 草稿没有 @ 任何人；flex 的 issue 会按 module label 自动抄送。要不要 cc drisspg（他也在 titan 评审），由用户定。
- 发出之后：
  - M3 的注释可以改成带链接的一行，基线正文的 Design 也可以加回原因和链接，都是新提交，不 amend；
  - 只讲约束的版本也站得住，不改也行。

标题：`[inductor] flex_attention backward autotuning passes captured buffers in a different order than the generated kernel takes them`

--- PASTE BEGIN ---
### Describe the bug

With max-autotune, the flex_attention backward lowering benchmarks its choices on a hand-built input list, `bwd_input_nodes + score_mod_other_buffers + mask_mod_other_buffers + mutated_grads`, which lists the captured buffers in the order the HOP passes them. The generated backward kernel takes them in the order its code first loads them. When a `mask_mod` captures two or more tensors and the two orders differ, every benchmark call reads one captured tensor through another's pointer.

- The out-of-bounds read is silent in most runs. In a 2-GPU FSDP run of MiniMax Sparse Attention in torchtitan it surfaced as `CUDA error: an illegal memory access` during backward autotuning (not under `CUDA_LAUNCH_BLOCKING=1`), and `compute-sanitizer --tool memcheck` with `PYTORCH_NO_CUDA_MEMORY_CACHING=1` reported the read in `triton_flex_attention_backward` just past the end of the smaller captured tensor.
- The kernel launch outside autotuning binds the buffers correctly; the gradients match SDPA.
- #194555 fixed the same mismatch in the forward and decoding lowerings by autotuning on `list(choices[0].input_nodes)`; the backward lowering still builds its list by hand.

Repro (cold inductor cache):

```python
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

T, H, D, B = 2048, 4, 64, 128
dev = "cuda"
block_of_key = torch.arange(T, device=dev) // B                      # [T], read at kv
picked = (torch.arange(T, device=dev) // B)[:, None].repeat(1, 4)    # [T, 4], read at q


def mask_mod(b, h, q_idx, kv_idx):
    key_block = block_of_key[kv_idx]          # traced (lifted) first
    hit = picked[q_idx, 0] == key_block       # but the kernel loads `picked` first
    for s in range(1, 4):
        hit = hit | (picked[q_idx, s] == key_block)
    return hit & (kv_idx <= q_idx)


block_mask = create_block_mask(mask_mod, 1, 1, T, T, device=dev)
flex = torch.compile(flex_attention, options={"max_autotune": True})
q, k, v = (torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
flex(q, k, v, block_mask=block_mask).sum().backward()
torch.cuda.synchronize()
```

A wrapper around `TritonTemplateCaller.benchmark` that prints the shapes of the benchmark arguments after the first 16 (`in_ptr args`) next to the shapes in the choice's `bmreq.input_tensor_meta` (`kernel argdefs`):

```
BENCH triton_flex_attention_backward_7 in_ptr args: [(2048,), (2048, 4)] kernel argdefs: [(2048, 4), (2048,)]
BENCH triton_flex_attention_backward_8 in_ptr args: [(2048,), (2048, 4)] kernel argdefs: [(2048, 4), (2048,)]
```

With the backward lowering autotuning on `list(choices[0].input_nodes)`, as #194555 does for the forward, both read `[(2048, 4), (2048,)]`, memcheck reports no error on the torchtitan case, and the 2-GPU FSDP run completes. The block-mask tensors keep their positions at the front of the generated inputs, so `input_gen_fns` stays as it is.

### Versions

- torch 2.16.0.dev20261024+cu130, triton 3.9.0, RTX 5060 Ti (sm_120)
- also seen on torch 2.16.0.dev20261023, RTX A4000 (sm_86)
--- PASTE END ---
