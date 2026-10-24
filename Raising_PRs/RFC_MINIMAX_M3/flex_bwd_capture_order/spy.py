"""Print the argument shapes each flex backward autotune benchmark call receives, then run a script."""
import runpy, sys
import torch._inductor.select_algorithm as sa

seen = set()
orig = sa.TritonTemplateCaller.benchmark
def benchmark(self, *args, out, **kw):
    name = getattr(self, "name", "")
    if "backward" in name and name not in seen and len(seen) < 2:
        seen.add(name)
        print("BENCH", name, "in_ptr args:", [tuple(a.shape) for a in args][16:], "kernel argdefs:",
              getattr(getattr(self, "bmreq", None), "input_tensor_meta", None) and [tuple(m.sizes) for m in self.bmreq.input_tensor_meta][16:], flush=True)
    return orig(self, *args, out=out, **kw)
sa.TritonTemplateCaller.benchmark = benchmark
script = sys.argv[1]; sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
