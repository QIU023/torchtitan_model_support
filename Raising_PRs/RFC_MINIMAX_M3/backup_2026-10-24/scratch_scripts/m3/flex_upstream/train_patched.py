import os, runpy, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import patch_bwd_autotune_inputs  # noqa: F401
sys.argv = ["torchtitan.train"] + sys.argv[1:]
runpy.run_module("torchtitan.train", run_name="__main__")
