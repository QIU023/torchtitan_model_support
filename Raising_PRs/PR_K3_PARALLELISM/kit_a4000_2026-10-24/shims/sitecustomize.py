"""A4000 kit shims, logbook kit only, for the sm86 box. Training ranks only (RANK set), so the run_tests and torchrun parents
import nothing.

(1) attn-gym's fused bounded gate is a CuTeDSL kernel for CUDA capability 9.0+, so the KDA gate runs attn-gym's own reference
implementation (impl="reference"); chunk_kda, l2norm and every other kernel stay as main calls them.
(2) Below sm90 varlen_attn runs FA2, which takes one head size for Q, K and V, while MLA's V head is narrower than its Q/K
head (FA3/FA4 take both). V is zero-padded to the Q/K head size and the output sliced back: softmax(QK^T) [V, 0] =
[softmax(QK^T) V, 0], the padded columns get zero gradient, and the scale comes from Q, so the attention is unchanged.
Installed before torchtitan imports varlen_attn, so its module-level alias picks it up."""

import os
import sys

if "RANK" in os.environ:
    import attn_gym.linear.kda.gate as _kda_gate

    _gate_transform = _kda_gate.gate_transform

    def _reference_gate_transform(*args, impl=None, **kwargs):
        return _gate_transform(*args, impl="reference", **kwargs)

    _kda_gate.gate_transform = _reference_gate_transform
    sys.stderr.write("a4000 shim: the KDA bounded gate runs impl='reference' (the fused gate needs sm90)\n")

    import torch.nn.attention.varlen as _varlen
    import torch.nn.functional as _F

    _varlen_attn = _varlen.varlen_attn

    def _varlen_attn_padded_v(query, key, value, *args, **kwargs):
        qk_head_dim, v_head_dim = query.shape[-1], value.shape[-1]
        if v_head_dim >= qk_head_dim:
            return _varlen_attn(query, key, value, *args, **kwargs)
        result = _varlen_attn(query, key, _F.pad(value, (0, qk_head_dim - v_head_dim)), *args, **kwargs)
        if isinstance(result, tuple):
            out, lse = result
            return out[..., :v_head_dim].contiguous(), lse
        return result[..., :v_head_dim].contiguous()

    _varlen.varlen_attn = _varlen_attn_padded_v
    sys.stderr.write("a4000 shim: varlen_attn pads V to the Q/K head size for FA2\n")
