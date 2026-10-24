import hashlib, importlib, re
from torchtitan.config.parallelism import ParallelismConfig
for name in ("qwen3_5", "kimi_k2_7", "kimi_k3"):
    mod = importlib.import_module(f"torchtitan.models.{name}")
    for flavor in [f for f in mod.MODEL_FLAVORS if "debug" in f][:2]:
        for sp in (True, False):
            c = mod.build_model_config(flavor, parallelism=ParallelismConfig(tensor_parallel_degree=2, expert_parallel_degree=2, enable_sequence_parallel=sp))
            r = re.sub(r"0x[0-9a-f]+", "0x", repr(c))
            print(name, flavor, sp, hashlib.sha256(r.encode()).hexdigest()[:16])
