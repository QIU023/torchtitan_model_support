import torch
from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLForCausalLM

torch.manual_seed(0)
cfg = MiniMaxM3VLTextConfig(
    vocab_size=1024, hidden_size=256, intermediate_size=64, num_hidden_layers=4,
    num_attention_heads=8, num_key_value_heads=2, head_dim=32, rotary_dim=16,
    num_local_experts=8, num_experts_per_tok=2, dense_intermediate_size=512, shared_intermediate_size=64,
    index_n_heads=2, index_head_dim=32, index_block_size=16, index_topk_blocks=4, index_local_blocks=1,
    layer_types=["full_attention"] + ["minimax_m3_sparse"] * 3, mlp_layer_types=["dense"] + ["sparse"] * 3,
    max_position_embeddings=4096,
)
for impl in ("eager", "sdpa"):
    cfg._attn_implementation = impl
    torch.manual_seed(0)
    model = MiniMaxM3VLForCausalLM(cfg).float()
    ids = torch.randint(0, 1024, (1, 256))
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    named = dict(model.named_parameters())
    idx = {n: p for n, p in named.items() if "index" in n}
    print(impl, "loss", round(out.loss.item(), 6), "| params", sum(p.numel() for p in named.values()))
    print("  indexer params:", sorted({n.split("layers.1.")[-1] for n in idx if "layers.1." in n}))
    print("  indexer grads non-zero:", any(p.grad is not None and p.grad.abs().sum() > 0 for p in idx.values()))
    print("  layer-1 q_proj grad norm", named["model.layers.1.self_attn.q_proj.weight"].grad.norm().item())
