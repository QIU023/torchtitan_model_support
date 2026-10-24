import runpy, sys
sys.path.insert(0, sys.argv[1]); import patch_bwd_autotune_inputs  # noqa: F401
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
