import hashlib, sys
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_5 import build_model_config
for sp in (True, False):
    c = build_model_config("debugmodel", parallelism=ParallelismConfig(tensor_parallel_degree=2, enable_sequence_parallel=sp))
    print(sp, hashlib.sha256(__import__("re").sub(r"0x[0-9a-f]+", "0x", repr(c)).encode()).hexdigest()[:16], len(repr(c)))
