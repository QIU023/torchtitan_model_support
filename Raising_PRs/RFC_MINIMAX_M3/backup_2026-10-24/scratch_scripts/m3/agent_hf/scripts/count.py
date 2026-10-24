# Parameter count derived from HF code + M3 config, compared to index total_size.
H = 6144; V = 200064; NH = 64; NKV = 4; HD = 128
L = 60; dense_layers = 3; moe_layers = 57; sparse_layers = 57
E = 128; I_moe = 3072; I_shared = 3072; I_dense = 12288
IH = 4; ID = 128
VH = 1280; VI = 5120; VL = 32; P = 14; TP = 2; C = 3
PROJ = 6144; MERGE_IN = H * 4

shapes = {}
shapes["embed"] = V * H
shapes["lm_head"] = V * H
shapes["final_norm"] = H
attn = H * NH * HD + 2 * H * NKV * HD + NH * HD * H + 2 * HD
shapes["attn_all"] = L * attn
shapes["layer_norms"] = L * 2 * H
indexer = H * IH * ID + H * ID + 2 * ID
shapes["indexer"] = sparse_layers * indexer
shapes["dense_mlp"] = dense_layers * 3 * H * I_dense
moe_bf16 = E * 3 * H * I_moe + E * H + 3 * H * I_shared
shapes["moe_bf16_wo_bias"] = moe_layers * moe_bf16
bias_elems = moe_layers * E
v_layer = 4 * (VH * VH + VH) + (VH * VI + VI) + (VI * VH + VH) + 2 * 2 * VH
shapes["vision"] = VL * v_layer + VH * C * TP * P * P + 2 * VH
shapes["projector"] = (VH * PROJ + PROJ) + (PROJ * H + H) + (MERGE_IN * PROJ + PROJ) + (PROJ * H + H)

total_bf16 = sum(shapes.values())
for k, v in shapes.items():
    print(f"{k:20s} {v:,}")
print("bf16 elems (excl. e_score bias):", f"{total_bf16:,}")
print("bias elems:", bias_elems)
total_size = 869157697024
for bias_bytes in (2, 4):
    est = total_bf16 * 2 + bias_elems * bias_bytes
    print(f"bias as {bias_bytes}B -> bytes {est:,} diff vs index {total_size - est:,}")
print("total params (B):", (total_bf16 + bias_elems) / 1e9)
act = (dense_layers * 3 * H * I_dense + moe_layers * (4 * 3 * H * I_moe + E * H + 3 * H * I_shared)
       + L * attn + sparse_layers * indexer + 2 * V * H)
print("activated text params approx (B):", act / 1e9)
