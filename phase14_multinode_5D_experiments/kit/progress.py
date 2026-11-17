"""Readable progress of the pipeline, for k8s.sh status (piped into a pod's python, no sync needed).

python progress.py <runs>/<hw>/pipeline ...   one line per task: done / RUN / waiting / GIVE UP, with start, end and
    duration from the attempt and done marks; then each node's heartbeat, and for the running cell the newest
    training step of node 0's log (or its last line while it initializes or compiles).
"""

import json
import re
import sys
import time
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"step:\s*\d+.*")


def hm(t: float) -> str:
    return time.strftime("%H:%M", time.localtime(t))


def mins(s: float) -> str:
    return f"{s / 60:.0f} min" if s < 5400 else f"{s / 3600:.1f} h"


def attempts(s: Path, tid: str) -> list[tuple[int, float]]:
    out = []
    for p in (s / "go").glob(f"{tid}.a*"):
        m = re.fullmatch(re.escape(tid) + r"\.a(\d+)", p.name)
        if m:
            out.append((int(m.group(1)), p.stat().st_mtime))
    return sorted(out)


def last_lines(path: Path, n: int = 400) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 200_000))
            text = f.read().decode(errors="replace")
    except OSError:
        return []
    return [ANSI.sub("", x).rstrip() for x in text.splitlines()[-n:] if x.strip()]


def cell_line(run_dir: Path) -> str:
    lines = last_lines(run_dir / "node0.log")
    steps = [STEP.search(x).group(0) for x in lines if STEP.search(x)]
    if steps:
        return re.sub(r"\s+", " ", steps[-1])[:150]
    if not lines:
        return "no log yet"
    told = [x for x in lines if re.search(r"\[titan\]|AUTOTUNE|multinode_5d|Error|Traceback", x)] or lines
    errors = sum("Traceback" in x for x in lines)
    since = time.strftime("%H:%M", time.localtime((run_dir / "node0.log").stat().st_mtime))
    return (f"no step yet (initializing or compiling the first step; log written {since}"
            + (f", {errors} tracebacks" if errors else "") + "): " + told[-1][-110:])


def show(s: Path) -> None:
    now = time.time()
    try:
        tasks = json.loads((s / "tasks.json").read_text())
    except (OSError, ValueError):
        print(f"== {s}: no tasks yet (setup or the first start still running)")
        return
    done = sum((s / "done" / t["id"]).exists() for t in tasks)
    given = sum((s / "giveup" / t["id"]).exists() for t in tasks)
    first = (s / "tasks.json").stat().st_mtime
    print(f"== {s.parent}: {done}/{len(tasks)} done, {given} given up, started {hm(first)} ({mins(now - first)} ago)")
    running = None
    for t in tasks:
        tid = t["id"]
        tries = attempts(s, tid)
        start = tries[0][1] if tries else None
        what = f"{t['name']} [{t['cell']}]" if t["kind"] == "cell" else tid
        if (s / "done" / tid).exists():
            end = (s / "done" / tid).stat().st_mtime
            took = f"{hm(start)}-{hm(end)} {mins(end - start)}" if start else hm(end)
            more = f", {len(tries)} attempts" if len(tries) > 1 else ""
            print(f"  done     {what:<44} {took}{more}")
        elif (s / "giveup" / tid).exists():
            print(f"  GIVE UP  {what:<44} {(s / 'giveup' / tid).read_text().strip()[:60]}")
        elif tries:
            n, t0 = tries[-1]
            print(f"  RUN      {what:<44} since {hm(t0)} ({mins(now - t0)}), attempt {n}")
            running = running or t
        else:
            print(f"  waiting  {what}")
    for h in sorted((s / "hb").glob("node*")):
        try:
            d = json.loads(h.read_text())
            age = now - d["t"]
            print(f"  {h.name}: {d['state']} ({age:.0f} s ago{', STALE' if age > 120 else ''})")
        except (OSError, ValueError, KeyError):
            print(f"  {h.name}: unreadable heartbeat")
    if running and running["kind"] == "cell":
        print(f"  {running['name']} node 0: {cell_line(s.parent / running['stage'] / running['name'])}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if Path(arg).is_dir():
            show(Path(arg))
