"""List the parameters and elements where reduce_emulate's rebuilt trees miss the run's gradients.
usage: reduce_mismatch.py <probe dir>"""
import runpy
import sys

import torch

ns = runpy.run_path(sys.argv[0].replace("reduce_mismatch.py", "reduce_emulate.py"), run_name="emulate")
names, f_in, h_in, f_out, h_out = ns["names"], ns["f_in"], ns["h_in"], ns["f_out"], ns["h_out"]
for label, rebuilt, produced in (
    ("HSDP", {n: ns["hsdp_tree"](h_in, n) for n in names}, h_out),
    ("FSDP ring 0123", {n: ns["fsdp_tree"](f_in, n, (0, 1, 2, 3)) for n in names}, f_out),
):
    for n in names:
        bad = rebuilt[n] != produced[n]
        if bad.any():
            idx = bad.nonzero()
            rel = ((rebuilt[n] - produced[n]).abs()[bad] / produced[n].abs()[bad].clamp_min(1e-30)).max()
            dt = sorted({str(f_in[r][n][m].dtype) for r in range(4) for m in range(2)})
            print(f"{label}: {n} shape {tuple(produced[n].shape)} input {dt} mismatched {int(bad.sum())} max rel {rel:.2e} first {idx[:3].tolist()}")
