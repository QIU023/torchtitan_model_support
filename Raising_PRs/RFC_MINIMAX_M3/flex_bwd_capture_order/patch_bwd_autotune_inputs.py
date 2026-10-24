"""Apply the proposed inductor fix: autotune flex backward on the choice's own input nodes."""
import torch._inductor.kernel.flex.flex_attention as fa

_orig = fa.autotune_select_algorithm


def _autotune_select_algorithm(name, choices, input_nodes, layout, **kwargs):
    if name == "flex_attention_backward" and choices:
        input_nodes = list(choices[0].input_nodes)
    return _orig(name, choices, input_nodes, layout, **kwargs)


fa.autotune_select_algorithm = _autotune_select_algorithm
