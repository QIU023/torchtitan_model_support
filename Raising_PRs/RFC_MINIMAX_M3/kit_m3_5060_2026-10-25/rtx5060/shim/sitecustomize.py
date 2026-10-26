"""5060 kit only, never committed: on sm120 main picks FA4, which needs the flash_attn package this box lacks; return None
from get_cuda_flash_attention_impl, the path the sm86 A4000 runs took."""
import os
import sys

if "RANK" in os.environ or os.environ.get("M3_SEED_OUT") or os.environ.get("M3_FA_OFF") == "1":
    import torchtitan.tools.utils as _tu

    _tu.get_cuda_flash_attention_impl = lambda: None
    sys.stderr.write(f"m3_5060_shim: rank {os.environ.get('RANK', '-')} flash attention selection off\n")
