"""Compare the comm runs of one hardware directory: python comm_compare.py <OUT>/<hw>.

comm/ holds the run with NCCL's own choices, comm_<tag>/ the comm_sweep.tsv variants (or a manual MN_COMM_TAG run).
For every op and group the table lists each run's alpha (us) and asymptotic busbw (GB/s) from comm_bench.py's fit,
its measured busbw at 1 MiB and at the largest size, and that largest-size busbw relative to the default run. Writes
comm_compare.tsv into the hardware directory and prints it.
"""

import json
import sys
from pathlib import Path

COLUMNS = ["op", "group", "ranks", "run", "alpha_us", "fit_busbw", "busbw_1MiB", "busbw_max", "max_vs_default"]


def load(path: Path):
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data, []
    return data.get("results", []), data.get("fits", [])


def fmt(v) -> str:
    if v is None:
        return "-"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def main() -> int:
    hw_dir = Path(sys.argv[1])
    runs = {}
    for d in sorted(hw_dir.glob("comm*")):
        f = d / "comm_bench.json"
        if d.is_dir() and f.exists():
            runs["default" if d.name == "comm" else d.name[len("comm_"):]] = load(f)
    if not runs:
        print(f"no comm*/comm_bench.json under {hw_dir}")
        return 1
    order = sorted(runs, key=lambda name: (name != "default", name))
    keys = sorted({(r["op"], r["group"], r["ranks"]) for res, _ in runs.values() for r in res})
    rows = []
    for key in keys:
        group_rows, base = [], None
        for name in order:
            res, fits = runs[name]
            sel = [r for r in res if (r["op"], r["group"], r["ranks"]) == key]
            if not sel:
                continue
            big = max(sel, key=lambda r: r["bytes"])
            mib = next((r for r in sel if r["bytes"] == 1 << 20), None)
            fit = next((f for f in fits if (f["op"], f["group"], f["ranks"]) == key), None)
            group_rows.append(dict(
                op=key[0], group=key[1], ranks=key[2], run=name,
                alpha_us=fit["alpha_us"] if fit else None, fit_busbw=fit["busbw"] if fit else None,
                busbw_1MiB=mib["busbw"] if mib else None, busbw_max=big["busbw"],
            ))
            if name == "default":
                base = big["busbw"]
        for row in group_rows:
            row["max_vs_default"] = row["busbw_max"] / base if base else None
        rows += group_rows
    lines = ["\t".join(COLUMNS)] + ["\t".join(fmt(r[c]) for c in COLUMNS) for r in rows]
    (hw_dir / "comm_compare.tsv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
