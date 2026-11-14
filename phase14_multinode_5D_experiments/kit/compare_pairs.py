"""Paired cells of one stage: python compare_pairs.py <stage_dir> <pairs.tsv> [--out F].

pairs.tsv holds one pair per line, a<TAB>b<TAB>question; a question that starts with "noise" or "噪声" marks the same
cell run twice, whose change is the noise floor. For every pair this prints the clean step time and MFU of both cells
(summarize.py step_s_clean and mfu_clean), the change from a to b in percent, whether it exceeds the noise floor, and,
from the stage's trace_report.tsv, the three groups with the most exposed communication and the straggler gap. The
conclusion column is left for a sentence by hand. A cell is the directory named after its row, or the newest manual
run of it (<row>_<MMDD_HHMMSS>). Two cells run at different lengths (MN_SEQ, e.g. one halved after running out of
memory) get no change, and the beyond_noise column says so.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize import parse_cell  # noqa: E402

COLUMNS = [
    "a", "b", "question", "a_step_s", "b_step_s", "step_change_pct", "a_mfu", "b_mfu", "noise_pct", "beyond_noise",
    "a_top_exposed", "b_top_exposed", "a_straggler_ms", "b_straggler_ms", "conclusion",
]


def find_cell(stage: Path, row: str) -> Path | None:
    exact = stage / row
    if (exact / "node0.log").exists():
        return exact
    runs = sorted(p for p in stage.iterdir() if p.is_dir() and re.fullmatch(re.escape(row) + r"_\d{4}_\d{6}", p.name))
    return runs[-1] if runs else None


def seq_of(cell: Path) -> str | None:
    env = cell / "env_node0.txt"
    m = re.search(r"^MN_SEQ=(\d+)$", env.read_text(errors="replace"), re.M) if env.exists() else None
    return m.group(1) if m else None


def trace_rows(stage: Path) -> dict[str, dict]:
    path = stage / "trace_report.tsv"
    if not path.exists():
        return {}
    lines = path.read_text().splitlines()
    head = lines[0].split("\t")
    return {r["cell"]: r for r in (dict(zip(head, line.split("\t"))) for line in lines[1:])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage_dir", type=Path)
    parser.add_argument("pairs", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    traces = trace_rows(args.stage_dir)
    pairs = [
        line.split("\t")[:3] for line in args.pairs.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    summaries: dict[str, dict | None] = {}
    seqs: dict[str, str | None] = {}
    for a, b, _ in pairs:
        for row in (a, b):
            if row not in summaries:
                cell = find_cell(args.stage_dir, row)
                summaries[row] = parse_cell(cell, 8, 3) if cell else None
                seqs[row] = seq_of(cell) if cell else None

    def change(a: str, b: str) -> float | None:
        sa, sb = summaries[a], summaries[b]
        if not sa or not sb or not sa.get("step_s_clean") or not sb.get("step_s_clean") or seqs[a] != seqs[b]:
            return None
        if sa["step_s_clean"] != sa["step_s_clean"] or sb["step_s_clean"] != sb["step_s_clean"]:
            return None
        return 100 * (sb["step_s_clean"] - sa["step_s_clean"]) / sa["step_s_clean"]

    noise = [abs(c) for a, b, q in pairs if q.startswith(("noise", "噪声")) for c in [change(a, b)] if c is not None]
    floor = max(noise) if noise else None
    out = []
    for a, b, question in pairs:
        sa, sb, c = summaries[a], summaries[b], change(a, b)
        ta = traces.get(sa["cell"]) if sa else None
        tb = traces.get(sb["cell"]) if sb else None
        out.append(dict(
            a=a, b=b, question=question,
            a_step_s=sa.get("step_s_clean") if sa else None, b_step_s=sb.get("step_s_clean") if sb else None,
            step_change_pct=c,
            a_mfu=sa.get("mfu_clean") if sa else None, b_mfu=sb.get("mfu_clean") if sb else None,
            noise_pct=floor,
            beyond_noise=(f"seq {seqs[a]} vs {seqs[b]}" if sa and sb and seqs[a] != seqs[b]
                          else None if c is None or floor is None or question.startswith(("noise", "噪声"))
                          else ("yes" if abs(c) > floor else "no")),
            a_top_exposed=ta.get("top_exposed") if ta else None, b_top_exposed=tb.get("top_exposed") if tb else None,
            a_straggler_ms=ta.get("straggler_ms") if ta else None, b_straggler_ms=tb.get("straggler_ms") if tb else None,
            conclusion="",
        ))

    def fmt(v) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            return "-" if v != v else f"{v:.2f}"
        return str(v)

    text = "\n".join(["\t".join(COLUMNS)] + ["\t".join(fmt(r[c]) for c in COLUMNS) for r in out]) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
