"""Unattended phase 14 pipeline for two nodes on a shared filesystem; resumes after preemption or restarts.

bootstrap.sh starts it on both nodes. Tasks run in one order, frozen in $OUT/$HW/pipeline/tasks.json the first
time node 0 starts: the communication check, the tokenizers, then the matrix cells (row by row: smoke, probe, warm,
measure; MN_ORDER=stage runs every row's smoke first, and so on). Node 0 decides attempt numbers, retries, probe
lengths and skips, and records each decision as a file under $OUT/$HW/pipeline/; node 1 follows those files. A
finished task is never rerun, so a restarted pod resumes at the first unfinished one. Each node keeps a heartbeat
with its state; a run whose peer died or moved on is killed and retried without counting as a failure. A probe that
runs out of memory retries at half the length; warm and measure use the length the probe settled on.

Environment (bootstrap.sh exports env.sh's variables): MN_MODELS ("k3 m3"), MN_STAGES ("smoke probe warm measure"),
MN_ROWS (only these matrix rows), MN_ORDER (row | stage), MN_MAX_FAILS (2), MN_MIN_SEQ (4096), MEAS_STEPS,
PROFILE_FREQ, MN_FLAVOR_OVERRIDE. $B/STOP stops both nodes once the current attempt is over.
"""

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ENV = os.environ
K = Path(__file__).resolve().parent
B = Path(ENV["B"])
OUT = Path(ENV["OUT"])
HW = ENV["HW"]
RANK = int(ENV.get("NODE_RANK", "0"))
PEER = 1 - RANK
S = OUT / HW / "pipeline"
MASTER_PORT = int(ENV.get("MASTER_PORT", "29500"))
MAX_FAILS = int(ENV.get("MN_MAX_FAILS", "2"))
MIN_SEQ = int(ENV.get("MN_MIN_SEQ", "4096"))
HB_STALE = 120  # seconds without a heartbeat before the peer counts as gone
GRACE_START = 240  # seconds the peer may take to pick up a new attempt
GRACE_PEER = 150  # seconds a run continues after its peer left the attempt
GRACE_FAILED = 300  # seconds a run continues after its peer's run failed
ABORTED = (97, 98)  # 97: killed because the peer left; 98: attempt skipped (the peer had already finished it)
PEER_FAILED = 96  # killed after the peer's run failed
OOM = re.compile(r"OutOfMemoryError|CUDA out of memory")
STATE = "boot"


def log(msg: str) -> None:
    line = f"{time.strftime('%m-%d %H:%M:%S')} node{RANK} {msg}"
    print(line, flush=True)
    try:
        with open(S / f"log_node{RANK}.txt", "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp{RANK}"
    tmp.write_text(text)
    for _ in range(50):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.1)
    os.replace(tmp, path)


def read(path: Path):
    try:
        return path.read_text().strip()
    except OSError:
        return None


def mark(kind: str, name: str) -> Path:
    return S / kind / name


def finished(tid: str) -> bool:
    return mark("done", tid).exists() or mark("giveup", tid).exists()


def set_state(state: str) -> None:
    global STATE
    STATE = state


def heartbeat() -> None:
    while True:
        try:
            write(S / "hb" / f"node{RANK}", json.dumps({"t": time.time(), "state": STATE}))
        except OSError:
            pass
        time.sleep(10)


def peer() -> tuple[str, float]:
    try:
        d = json.loads((S / "hb" / f"node{PEER}").read_text())
        return d["state"], time.time() - d["t"]
    except (OSError, ValueError, KeyError):
        return "missing", 1e9


def attempts(tid: str) -> list[int]:
    found = []
    for p in (S / "go").glob(f"{tid}.a*"):
        m = re.fullmatch(re.escape(tid) + r"\.a(\d+)", p.name)
        if m:
            found.append(int(m.group(1)))
    return sorted(found)


def rc_path(tid: str, a: int, node: int) -> Path:
    return mark("rc", f"{tid}.a{a}.node{node}")


def get_rc(tid: str, a: int, node: int):
    v = read(rc_path(tid, a, node))
    return None if v is None else int(v)


# ---------------------------------------------------------------- tasks


def matrix_rows(model: str) -> list[dict]:
    rows = []
    for line in (K / f"matrix_{model}.tsv").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, cell, s_h100, s_h200, s_a100, extra = (line.split("\t") + [""] * 6)[:6]
        seq = {"h100": s_h100, "h200": s_h200, "b200": s_h200}.get(HW, s_a100)
        envs, nosmoke = {}, False
        for item in extra.split():
            key = item.split("=", 1)[0]
            if item == "nosmoke":
                nosmoke = True
            elif ":" in key:
                tag, kv = item.split(":", 1)
                if tag == HW or (HW == "b200" and tag == "h200"):
                    k, v = kv.split("=", 1)
                    envs[k] = v
            elif "=" in item:
                k, v = item.split("=", 1)
                envs[k] = v
        rows.append(dict(name=name, cell=cell, seq=int(seq), envs=envs, nosmoke=nosmoke))
    return rows


def build_tasks() -> list[dict]:
    models = ENV.get("MN_MODELS", "k3 m3").split()
    stages = ENV.get("MN_STAGES", "smoke probe warm measure").split()
    only = set(ENV.get("MN_ROWS", "").split())
    tasks = [dict(id="comm", kind="comm")]
    if set(stages) & {"probe", "warm", "measure"}:
        tasks += [dict(id=f"assets.{m}", kind="assets", model=m) for m in models]
    for m in models:
        rows = [r for r in matrix_rows(m) if not only or r["name"] in only]
        cells = [(st, r) for st in stages for r in rows]
        if ENV.get("MN_ORDER", "row") == "row":
            cells.sort(key=lambda c: [r["name"] for r in rows].index(c[1]["name"]))
        for st, r in cells:
            if st == "smoke" and r["nosmoke"]:
                continue
            tasks.append(dict(id=f"{m}.{st}.{r['name']}", kind="cell", model=m, stage=st, **r))
    return tasks


def load_tasks() -> list[dict]:
    path = S / "tasks.json"
    if RANK == 0 and not path.exists():
        write(path, json.dumps(build_tasks(), indent=1))
        log(f"tasks frozen in {path}")
    while not path.exists():
        time.sleep(10)
    return json.loads(path.read_text())


def run_dir(t: dict) -> Path:
    return OUT / HW / t["stage"] / t["name"]


def assets_env(model: str) -> dict:
    v = read(mark("assets", f"{model}.json"))
    return json.loads(v) if v else {"MN_ASSETS": ENV[f"ASSETS_{model}"], "MN_SPECIAL": "model"}


def cell_env(t: dict, seq: int) -> dict:
    m, st = t["model"], t["stage"]
    pp = int((re.search(r"pp=(\d+)", t["cell"]) or [0, "1"])[1])
    e = dict(MN_MODEL=m, MN_CELL=t["cell"], MN_GATE=ENV["GATE"])
    e.update(t["envs"])
    if st == "smoke":
        e.update(
            MN_FLAVOR="debugmodel",
            MN_SEQ=str(4096 if m == "k3" else 2048),
            MN_MICROBATCHES=str(2 * pp if pp > 1 else 1),
            MN_STEPS="5",
            MN_DATASET="c4_test",
            MN_CACHE=f"{OUT}/cache/{HW}/smoke_{m}",
        )
        return e
    e.update(assets_env(m))
    e.update(MN_FLAVOR=ENV.get("MN_FLAVOR_OVERRIDE") or ENV[f"FLAVOR_{m}"], MN_SEQ=str(seq))
    if st == "probe":
        e.update(MN_STEPS="3", MN_DATASET="c4_test", MN_CACHE=f"{OUT}/cache/{HW}/probe_{m}")
    elif st == "warm":
        e.update(MN_STEPS="10", MN_DATASET="c4", MN_CACHE=f"{OUT}/cache/{HW}/{m}")
    else:
        e.update(
            MN_STEPS=ENV.get("MEAS_STEPS", "30"),
            MN_PROFILE=ENV.get("PROFILE_FREQ", "25"),
            MN_DATASET="c4",
            MN_CACHE=f"{OUT}/cache/{HW}/{m}",
        )
    return e


# ---------------------------------------------------------------- running


def session_pids(sid: int) -> list[int]:
    pids = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
            if int(fields[3]) == sid:
                pids.append(int(stat.parent.name))
        except (OSError, IndexError, ValueError):
            pass
    return pids


def kill_run(proc: subprocess.Popen) -> None:
    for sig, wait in ((signal.SIGTERM, 60), (signal.SIGKILL, 10)):
        for pid in session_pids(proc.pid):
            try:
                os.kill(pid, sig)
            except OSError:
                pass
        try:
            proc.wait(timeout=wait)
            break
        except subprocess.TimeoutExpired:
            pass
    # torchrun's workers run in sessions of their own; on this node only this pipeline starts them.
    subprocess.call(["pkill", "-KILL", "-f", "torchtitan.train|comm_bench.py"])


def run_attempt(t: dict, idx: int, a: int, seq: int) -> int:
    tid = t["id"]
    env = dict(ENV)
    env["MN_PORT"] = str(MASTER_PORT + 1 + (idx * 16 + a) % 3000)
    if t["kind"] == "comm":
        cmd = ["bash", str(K / "run_comm.sh")]
    else:
        env.update(cell_env(t, seq))
        cmd = ["bash", str(K / "launch.sh"), str(run_dir(t))]
    with open(S / f"run_node{RANK}.log", "a") as out:
        out.write(f"\n==== {time.strftime('%m-%d %H:%M:%S')} {tid} attempt {a} seq {seq}\n")
        out.flush()
        proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        return watch(proc, tid, a)


def watch(proc: subprocess.Popen, tid: str, a: int) -> int:
    """Wait for the run; kill it when the peer has left this attempt or its run failed."""
    start, bad_since = time.time(), None
    while proc.poll() is None:
        time.sleep(10)
        state, age = peer()
        here = state in (f"run {tid} {a}", f"wait {tid} {a}")
        starting = state == f"ready {tid}" and time.time() - start < GRACE_START
        peer_rc = get_rc(tid, a, PEER)
        reason, grace = None, GRACE_PEER
        if age >= HB_STALE or not (here or starting):
            reason = f"peer gone or moved on ({state}, heartbeat {age:.0f}s old)"
        elif peer_rc in ABORTED:
            reason = "peer aborted this attempt"
        elif peer_rc not in (None, 0):
            reason, grace = "peer's run failed", GRACE_FAILED
        if reason is None:
            bad_since = None
        elif bad_since is None:
            bad_since = time.time()
        elif time.time() - bad_since > grace:
            log(f"{tid} attempt {a}: killing the run, {reason}")
            kill_run(proc)
            return PEER_FAILED if reason == "peer's run failed" else 97
    return proc.returncode


def oom_in(path: Path) -> bool:
    for log_file in path.glob("node*.log"):
        try:
            if OOM.search(log_file.read_text(errors="replace")):
                return True
        except OSError:
            pass
    return False


def archive(t: dict, a: int, tag: str) -> None:
    src = run_dir(t)
    if src.exists():
        dst = OUT / HW / t["stage"] / "_attempts" / f"{t['name']}.a{a}.{tag}"
        if dst.exists():
            dst = dst.with_name(f"{dst.name}.{int(time.time())}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)


def wait_peer_rc(tid: str, a: int):
    bad_since = None
    start = time.time()
    while True:
        rc = get_rc(tid, a, PEER)
        if rc is not None:
            return rc
        state, age = peer()
        ok = age < HB_STALE and (
            state in (f"run {tid} {a}", f"wait {tid} {a}")
            or (state == f"ready {tid}" and time.time() - start < GRACE_START)
        )
        if ok:
            bad_since = None
        elif bad_since is None:
            bad_since = time.time()
        elif time.time() - bad_since > GRACE_PEER:
            return None
        time.sleep(5)


def wait_peer_free(tid: str, last: int) -> bool:
    while True:
        state, age = peer()
        if age < HB_STALE and (state == f"ready {tid}" or state == f"wait {tid} {last}"):
            return True
        if stopped():
            return False
        time.sleep(5)


def stopped() -> bool:
    return (B / "STOP").exists()


def current_seq(t: dict) -> int:
    if t["kind"] != "cell" or t["stage"] == "smoke":
        return 0
    key = f"{t['model']}.{t['name']}"
    if t["stage"] == "probe":
        return int(read(mark("try", t["id"])) or t["seq"])
    return int(read(mark("len", key)))


def blocked(t: dict, ids: set) -> str | None:
    if t["kind"] != "cell":
        return None
    m, st, name = t["model"], t["stage"], t["name"]
    need = [f"assets.{m}"] if st != "smoke" else []
    prev = {"probe": "smoke", "warm": "probe", "measure": "warm"}.get(st)
    if prev and f"{m}.{prev}.{name}" in ids:
        need.append(f"{m}.{prev}.{name}")
    for n in need:
        if n in ids and not mark("done", n).exists():
            return f"skipped: {n} did not finish"
    if st in ("warm", "measure") and not mark("len", f"{m}.{name}").exists():
        return "skipped: no probe length"
    return None


def lead(t: dict, idx: int, ids: set) -> None:
    tid = t["id"]
    # An attempt this node announced but never finished was cut off by a restart: mark it aborted for node 1.
    announced = attempts(tid)
    if announced and t["kind"] != "assets" and get_rc(tid, announced[-1], 0) is None:
        write(rc_path(tid, announced[-1], 0), "97")
    while not finished(tid):
        if stopped():
            return
        why = blocked(t, ids)
        if why:
            write(mark("giveup", tid), why)
            log(f"{tid}: {why}")
            return
        fails = int(read(mark("fails", tid)) or 0)
        if fails >= MAX_FAILS:
            write(mark("giveup", tid), f"failed {fails} times")
            log(f"{tid}: gave up after {fails} failures")
            return
        done_attempts = attempts(tid)
        a = (done_attempts[-1] if done_attempts else 0) + 1
        if t["kind"] == "assets":
            set_state(f"run {tid} {a}")
            write(mark("go", f"{tid}.a{a}"), "{}")
            with open(S / f"run_node{RANK}.log", "a") as out:
                rc = subprocess.call(
                    ["bash", str(K / "assets.sh"), t["model"], str(mark("assets", f"{t['model']}.json"))],
                    stdout=out,
                    stderr=subprocess.STDOUT,
                )
            if rc == 0:
                write(mark("done", tid), "ok")
                log(f"{tid}: done ({read(mark('assets', t['model'] + '.json'))})")
            else:
                write(mark("fails", tid), str(fails + 1))
                log(f"{tid}: failed (rc {rc})")
            continue
        seq = current_seq(t)
        set_state(f"ready {tid}")
        if not wait_peer_free(tid, a - 1):
            return
        # A run directory still here belongs to an attempt cut off by a restart (finished attempts are archived).
        if t["kind"] == "cell":
            archive(t, a - 1, "cut")
        write(mark("go", f"{tid}.a{a}"), json.dumps({"seq": seq}))
        set_state(f"run {tid} {a}")
        log(f"{tid}: attempt {a} starts" + (f" at seq {seq}" if seq else ""))
        rc0 = run_attempt(t, idx, a, seq)
        write(rc_path(tid, a, 0), str(rc0))
        set_state(f"wait {tid} {a}")
        rc1 = wait_peer_rc(tid, a)
        if rc1 is None or rc0 in ABORTED or rc1 in ABORTED:
            log(f"{tid}: attempt {a} aborted (rc {rc0} / {rc1}), not counted")
            if t["kind"] == "cell":
                archive(t, a, "aborted")
            continue
        if rc0 == 0 and rc1 == 0:
            if t["kind"] == "cell" and t["stage"] == "probe":
                write(mark("len", f"{t['model']}.{t['name']}"), str(seq))
            write(mark("done", tid), "ok")
            log(f"{tid}: done")
            return
        if t["kind"] == "cell" and t["stage"] == "probe" and oom_in(run_dir(t)) and seq // 2 >= MIN_SEQ:
            write(mark("try", tid), str(seq // 2))
            archive(t, a, f"oom{seq}")
            log(f"{tid}: out of memory at seq {seq}, retrying at {seq // 2} (not counted)")
            continue
        write(mark("fails", tid), str(fails + 1))
        if t["kind"] == "cell":
            archive(t, a, "failed")
        log(f"{tid}: attempt {a} failed (rc {rc0} / {rc1}), {fails + 1} of {MAX_FAILS}")


def follow(t: dict, idx: int) -> None:
    tid = t["id"]
    set_state(f"ready {tid}")
    # Node 0 announces an attempt only after this node said "ready", so an attempt that is already announced when this
    # node arrives at the task was cut off by this node's own restart: it cannot rejoin a run already in progress.
    if t["kind"] != "assets":
        for a in attempts(tid):
            if get_rc(tid, a, 1) is None:
                write(rc_path(tid, a, 1), "97" if get_rc(tid, a, 0) is None else "98")
    while not finished(tid):
        if t["kind"] != "assets":
            pending = [a for a in attempts(tid) if get_rc(tid, a, 1) is None]
            if pending:
                a = pending[-1]
                for old in pending[:-1]:
                    write(rc_path(tid, old, 1), "98")
                if get_rc(tid, a, 0) is not None:
                    write(rc_path(tid, a, 1), "98")
                    set_state(f"wait {tid} {a}")
                    continue
                seq = int(json.loads(read(mark("go", f"{tid}.a{a}")) or "{}").get("seq", 0))
                set_state(f"run {tid} {a}")
                log(f"{tid}: attempt {a} starts")
                rc = run_attempt(t, idx, a, seq)
                write(rc_path(tid, a, 1), str(rc))
                set_state(f"wait {tid} {a}")
                log(f"{tid}: attempt {a} rc {rc}")
                continue
        if stopped():
            return
        time.sleep(5)


def finish(tasks: list[dict]) -> None:
    lines = []
    for t in tasks:
        tid = t["id"]
        if mark("done", tid).exists():
            status = "done"
            if t["kind"] == "cell" and t["stage"] != "smoke":
                status += f" seq {read(mark('len', t['model'] + '.' + t['name'])) or current_seq(t)}"
        elif mark("giveup", tid).exists():
            status = "GIVE UP: " + (read(mark("giveup", tid)) or "")
        else:
            status = "not run"
        lines.append(f"{tid}\t{status}")
    write(S / "REPORT.txt", "\n".join(lines) + "\n")
    for stage_dir in sorted((OUT / HW).iterdir()):
        if stage_dir.is_dir() and any(stage_dir.glob("*/node0.log")):
            with open(stage_dir / "summary.tsv", "w") as f:
                subprocess.call([sys.executable, str(K / "summarize.py"), str(stage_dir)], stdout=f)
    write(S / "ALL_DONE", time.strftime("%m-%d %H:%M:%S"))
    log("pipeline finished; REPORT.txt and the summaries are written")


def main() -> int:
    if int(ENV.get("NNODES", "2")) != 2:
        print("pipeline.py drives exactly two nodes", file=sys.stderr)
        return 2
    S.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    tasks = load_tasks()
    ids = {t["id"] for t in tasks}
    log(f"start: {len(tasks)} tasks, {sum(finished(t['id']) for t in tasks)} already finished")
    for idx, t in enumerate(tasks):
        if finished(t["id"]):
            continue
        if stopped():
            break
        if RANK == 0:
            lead(t, idx, ids)
        else:
            follow(t, idx)
    if not all(finished(t["id"]) for t in tasks):
        log("STOP found, exiting")
        return 0
    if RANK == 0:
        finish(tasks)
    else:
        while not (S / "ALL_DONE").exists():
            time.sleep(10)
    set_state("done-all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
