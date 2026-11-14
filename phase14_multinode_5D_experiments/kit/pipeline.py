"""Unattended phase 14 pipeline for 2, 4 or 8 nodes on a shared filesystem; resumes after preemption or restarts.

bootstrap.sh starts it on every node. Tasks run in one order, frozen in $OUT/$HW/pipeline/tasks.json the first
time node 0 starts: the preflight (every GPU's GEMM rate and clocks, the quick network check; a failure stops the
pipeline before the matrix, MN_PREFLIGHT=warn only reports, 0 skips it), the communication check, the tokenizers, the matrix cells (row by row: smoke, probe, warm,
measure; MN_ORDER=stage runs every row's smoke first, and so on), then one communication run per NCCL variant of
comm_sweep.tsv (MN_COMM_SWEEP=0 leaves them out). Node 0 decides attempt numbers, retries, probe
lengths and skips, and records each decision as a file under $OUT/$HW/pipeline/; the other nodes follow those files.
A finished task is never rerun, so a restarted pod resumes at the first unfinished one. Each node keeps a heartbeat
with its state; a run whose peer (any other node) died or moved on is killed and retried without counting as a
failure. The matrix is matrix_<model>.tsv on 2 nodes and matrix_<model>_n<NNODES>.tsv on 4 or 8. A probe that
runs out of memory retries at half the length; warm and measure use the length the probe settled on. Probe stores a
memory snapshot at step 2 (MN_MEMSNAP_PROBE, empty for none); measure writes TensorBoard metrics and a profiler trace;
at the end node 0 writes the stage summaries, the monitor summary, the NCCL variant comparison, the measure trace
report and the paired comparison of pairs_<model>[_n<NNODES>].tsv (compare_pairs.py).

Environment (bootstrap.sh exports env.sh's variables): MN_MODELS ("k3 m3"), MN_STAGES ("smoke probe warm measure"),
MN_ROWS (only these matrix rows; +<tag> selects the rows a matrix tags so; empty runs every row but the
+optional ones), MN_ORDER (row | stage), MN_MAX_FAILS (2), MN_MIN_SEQ (4096), MEAS_STEPS,
PROFILE_FREQ, MN_FLAVOR_OVERRIDE. $B/STOP stops every node once the current attempt is over.
"""

import json
import os
import re
import shlex
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
NNODES = int(ENV.get("NNODES", "2"))
OTHERS = [n for n in range(NNODES) if n != RANK]
FOLLOWERS = list(range(1, NNODES))
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


def peer(node: int) -> tuple[str, float]:
    try:
        d = json.loads((S / "hb" / f"node{node}").read_text())
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


def matrix_path(model: str) -> Path:
    return K / (f"matrix_{model}.tsv" if NNODES == 2 else f"matrix_{model}_n{NNODES}.tsv")


def matrix_rows(model: str) -> list[dict]:
    path = matrix_path(model)
    if not path.exists():
        raise SystemExit(f"{path.name} does not exist: no {model} matrix for {NNODES} nodes (MN_MODELS)")
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, cell, s_h100, s_h200, s_a100, extra = (line.split("\t") + [""] * 6)[:6]
        seq = {"h100": s_h100, "h200": s_h200, "b200": s_h200}.get(HW, s_a100)
        envs, nosmoke, tags = {}, False, []
        for item in extra.split():
            key = item.split("=", 1)[0]
            if item == "nosmoke":
                nosmoke = True
            elif item.startswith("+"):
                tags.append(item[1:])
            elif ":" in key:
                tag, kv = item.split(":", 1)
                if tag == HW or (HW == "b200" and tag == "h200"):
                    k, v = kv.split("=", 1)
                    envs[k] = v
            elif "=" in item:
                k, v = item.split("=", 1)
                envs[k] = v
        rows.append(dict(name=name, cell=cell, seq=int(seq), envs=envs, nosmoke=nosmoke, tags=tags))
    return rows


def build_tasks() -> list[dict]:
    models = ENV.get("MN_MODELS", "k3 m3").split()
    stages = ENV.get("MN_STAGES", "smoke probe warm measure").split()
    only = set(ENV.get("MN_ROWS", "").split())

    def selected(r: dict) -> bool:
        if not only:
            return "optional" not in r["tags"]
        return r["name"] in only or any("+" + tag in only for tag in r["tags"])

    tasks = [dict(id="preflight", kind="preflight")] if ENV.get("MN_PREFLIGHT", "1") != "0" else []
    tasks += [dict(id="comm", kind="comm")] if ENV.get("MN_COMM", "1") == "1" else []
    if set(stages) & {"probe", "warm", "measure"}:
        tasks += [dict(id=f"assets.{m}", kind="assets", model=m) for m in models]
    for m in models:
        rows = [r for r in matrix_rows(m) if selected(r)]
        cells = [(st, r) for st in stages for r in rows]
        if ENV.get("MN_ORDER", "row") == "row":
            cells.sort(key=lambda c: [r["name"] for r in rows].index(c[1]["name"]))
        for st, r in cells:
            if st == "smoke" and r["nosmoke"]:
                continue
            tasks.append(dict(id=f"{m}.{st}.{r['name']}", kind="cell", model=m, stage=st, **r))
    return tasks + comm_variants()


def comm_variants() -> list[dict]:
    """The NCCL tuning variants of comm_sweep.tsv as comm tasks after the matrix; MN_COMM_SWEEP=0 leaves them out."""
    if ENV.get("MN_COMM_SWEEP", ENV.get("MN_COMM", "1")) != "1":
        return []
    out = []
    for line in (K / "comm_sweep.tsv").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tag, _, rest = line.partition("\t")
        envs = dict(item.split("=", 1) for item in rest.split())
        out.append(dict(id=f"comm.{tag}", kind="comm", tag=tag, envs=envs))
    return out


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
    # A row may pick its own flavor; {mem} stands for the hardware's size class (env.sh MEM: 80g or 141g).
    row_flavor = t["envs"].get("MN_FLAVOR", "").replace("{mem}", ENV.get("MEM", "80g"))
    e.update(MN_FLAVOR=ENV.get("MN_FLAVOR_OVERRIDE") or row_flavor or ENV[f"FLAVOR_{m}"], MN_SEQ=str(seq))
    if st == "probe":
        e.update(MN_STEPS="3", MN_DATASET="c4_test", MN_CACHE=f"{OUT}/cache/{HW}/probe_{m}")
        e.setdefault("MN_MEMSNAP", ENV.get("MN_MEMSNAP_PROBE", "2"))
    elif st == "warm":
        e.update(MN_STEPS="10", MN_DATASET="c4", MN_CACHE=f"{OUT}/cache/{HW}/{m}")
    else:
        e.update(
            MN_STEPS=ENV.get("MEAS_STEPS", "30"),
            MN_PROFILE=ENV.get("PROFILE_FREQ", "25"),
            MN_DATASET="c4",
            MN_CACHE=f"{OUT}/cache/{HW}/{m}",
        )
        e.setdefault("MN_TB", "1")
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
    if t["kind"] == "preflight":
        cmd = ["bash", str(K / "preflight.sh")]
    elif t["kind"] == "comm":
        env.update(t.get("envs", {}))
        env["MN_COMM_TAG"] = t.get("tag", "")
        cmd = ["bash", str(K / "run_comm.sh")]
    else:
        env.update(cell_env(t, seq))
        cmd = ["bash", str(K / "launch.sh"), str(run_dir(t))]
    with open(S / f"run_node{RANK}.log", "a") as out:
        out.write(f"\n==== {time.strftime('%m-%d %H:%M:%S')} {tid} attempt {a} seq {seq}\n")
        out.flush()
        proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        return watch(proc, tid, a)


def peer_problem(tid: str, a: int, start: float) -> tuple[str | None, float]:
    """Why this node's run cannot finish (another node left the attempt, aborted it or failed), with its grace."""
    for n in OTHERS:
        state, age = peer(n)
        here = state in (f"run {tid} {a}", f"wait {tid} {a}")
        starting = state == f"ready {tid}" and time.time() - start < GRACE_START
        rc = get_rc(tid, a, n)
        if age >= HB_STALE or not (here or starting):
            return f"node {n} gone or moved on ({state}, heartbeat {age:.0f}s old)", GRACE_PEER
        if rc in ABORTED:
            return f"node {n} aborted this attempt", GRACE_PEER
        if rc not in (None, 0):
            return f"node {n}'s run failed", GRACE_FAILED
    return None, GRACE_PEER


def watch(proc: subprocess.Popen, tid: str, a: int) -> int:
    """Wait for the run; kill it when another node has left this attempt or its run failed."""
    start, bad_since = time.time(), None
    while proc.poll() is None:
        time.sleep(10)
        reason, grace = peer_problem(tid, a, start)
        if reason is None:
            bad_since = None
        elif bad_since is None:
            bad_since = time.time()
        elif time.time() - bad_since > grace:
            log(f"{tid} attempt {a}: killing the run, {reason}")
            kill_run(proc)
            return PEER_FAILED if reason.endswith("run failed") else 97
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


def wait_followers_rc(tid: str, a: int) -> list[int] | None:
    """Every follower's exit code of this attempt, or None when one left it without writing one."""
    bad_since = None
    start = time.time()
    while True:
        rcs = [get_rc(tid, a, n) for n in FOLLOWERS]
        if all(rc is not None for rc in rcs):
            return rcs
        ok = True
        for n, rc in zip(FOLLOWERS, rcs):
            if rc is not None:
                continue
            state, age = peer(n)
            ok = ok and age < HB_STALE and (
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


def wait_followers_free(tid: str, last: int) -> bool:
    while True:
        states = [peer(n) for n in FOLLOWERS]
        if all(age < HB_STALE and state in (f"ready {tid}", f"wait {tid} {last}") for state, age in states):
            return True
        if stopped():
            return False
        time.sleep(5)


def stopped() -> bool:
    return (B / "STOP").exists()


def current_seq(t: dict) -> int:
    """The length a cell runs at: a halved retry, else the probe's, else the row's MN_SEQ, else MN_SEQ_FORCE, else the
    matrix column's."""
    if t["kind"] != "cell" or t["stage"] == "smoke":
        return 0
    settled = read(mark("len", f"{t['model']}.{t['name']}")) if t["stage"] != "probe" else None
    pinned = t["envs"].get("MN_SEQ")
    return int(read(mark("try", t["id"])) or settled or pinned or ENV.get("MN_SEQ_FORCE") or t["seq"])


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
    if st in ("warm", "measure") and f"{m}.probe.{name}" in ids and not mark("len", f"{m}.{name}").exists():
        return "skipped: no probe length"
    return None


def preflight_passed() -> bool:
    """Node 0's verdict on the preflight; a failure stops every node (STOP) unless MN_PREFLIGHT=warn."""
    with open(S / "preflight_verdict.log", "w") as out:
        rc = subprocess.call(
            [sys.executable, str(K / "preflight.py"), "verdict", str(OUT / HW / "preflight"),
             "--comm", str(OUT / HW / "comm_preflight"), "--nodes", str(NNODES)],
            stdout=out, stderr=subprocess.STDOUT,
        )
    if rc == 0:
        log("preflight passed")
        return True
    if ENV.get("MN_PREFLIGHT", "1") == "warn":
        log(f"preflight failed (MN_PREFLIGHT=warn, going on): {OUT / HW / 'preflight' / 'VERDICT.txt'}")
        return True
    log(f"preflight FAILED, stopping before the matrix: {OUT / HW / 'preflight' / 'VERDICT.txt'}; "
        "replace the node, then k8s.sh start (the preflight runs again)")
    write(B / "STOP", "preflight failed")
    return False


def lead(t: dict, idx: int, ids: set) -> None:
    tid = t["id"]
    # An attempt this node announced but never finished was cut off by a restart: mark it aborted for the followers.
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
        if not wait_followers_free(tid, a - 1):
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
        rcs = wait_followers_rc(tid, a)
        rc_text = " / ".join(map(str, [rc0] + (rcs or ["gone"])))
        if rcs is None or rc0 in ABORTED or any(rc in ABORTED for rc in rcs):
            log(f"{tid}: attempt {a} aborted (rc {rc_text}), not counted")
            if t["kind"] == "cell":
                archive(t, a, "aborted")
            continue
        if rc0 == 0 and all(rc == 0 for rc in rcs):
            if t["kind"] == "preflight" and not preflight_passed():
                return
            if t["kind"] == "cell" and t["stage"] == "probe":
                write(mark("len", f"{t['model']}.{t['name']}"), str(seq))
            write(mark("done", tid), "ok")
            log(f"{tid}: done")
            return
        # Without a probe in the plan, the first stage of the row that runs the real model halves on OOM instead.
        halves = t["kind"] == "cell" and (
            t["stage"] == "probe" or (t["stage"] != "smoke" and f"{t['model']}.probe.{t['name']}" not in ids)
        )
        if halves and oom_in(run_dir(t)) and seq // 2 >= MIN_SEQ:
            write(mark("try", tid), str(seq // 2))
            archive(t, a, f"oom{seq}")
            log(f"{tid}: out of memory at seq {seq}, retrying at {seq // 2} (not counted)")
            continue
        write(mark("fails", tid), str(fails + 1))
        if t["kind"] == "cell":
            archive(t, a, "failed")
        log(f"{tid}: attempt {a} failed (rc {rc_text}), {fails + 1} of {MAX_FAILS}")


def follow(t: dict, idx: int) -> None:
    tid = t["id"]
    set_state(f"ready {tid}")
    # Node 0 announces an attempt only after this node said "ready", so an attempt that is already announced when this
    # node arrives at the task was cut off by this node's own restart: it cannot rejoin a run already in progress.
    if t["kind"] != "assets":
        for a in attempts(tid):
            if get_rc(tid, a, RANK) is None:
                write(rc_path(tid, a, RANK), "97" if get_rc(tid, a, 0) is None else "98")
    while not finished(tid):
        if t["kind"] != "assets":
            pending = [a for a in attempts(tid) if get_rc(tid, a, RANK) is None]
            if pending:
                a = pending[-1]
                for old in pending[:-1]:
                    write(rc_path(tid, old, RANK), "98")
                if get_rc(tid, a, 0) is not None:
                    write(rc_path(tid, a, RANK), "98")
                    set_state(f"wait {tid} {a}")
                    continue
                seq = int(json.loads(read(mark("go", f"{tid}.a{a}")) or "{}").get("seq", 0))
                set_state(f"run {tid} {a}")
                log(f"{tid}: attempt {a} starts")
                rc = run_attempt(t, idx, a, seq)
                write(rc_path(tid, a, RANK), str(rc))
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
    with open(S / "report.log", "a") as f:
        subprocess.call([sys.executable, str(K / "monitor.py"), "--summary"], stdout=f, stderr=subprocess.STDOUT)
        subprocess.call([sys.executable, str(K / "comm_compare.py"), str(OUT / HW)], stdout=f, stderr=subprocess.STDOUT)
        if (OUT / HW / "measure").is_dir():
            subprocess.call(
                [sys.executable, str(K / "trace_report.py"), str(OUT / HW / "measure")], stdout=f, stderr=subprocess.STDOUT
            )
            for m in {t["model"] for t in tasks if t["kind"] == "cell"}:
                pairs = K / matrix_path(m).name.replace("matrix_", "pairs_")
                if pairs.exists():
                    subprocess.call(
                        [sys.executable, str(K / "compare_pairs.py"), str(OUT / HW / "measure"), str(pairs),
                         "--out", str(OUT / HW / "measure" / f"pairs_report_{m}.tsv")],
                        stdout=f, stderr=subprocess.STDOUT,
                    )
    write(S / "ALL_DONE", time.strftime("%m-%d %H:%M:%S"))
    log("pipeline finished; REPORT.txt, the summaries and the measure trace report are written")


def main() -> int:
    if NNODES not in (2, 4, 8):
        print(f"pipeline.py drives 2, 4 or 8 nodes, got NNODES={NNODES}", file=sys.stderr)
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


def print_cell_env(model: str, stage: str, row: str) -> int:
    """Shell exports of one cell's settings, as the pipeline would run it (k8s.sh cell runs a cell by hand)."""
    rows = [r for r in matrix_rows(model) if r["name"] == row]
    if not rows or stage not in ("smoke", "probe", "warm", "measure"):
        print(f"echo 'no row {row} in {matrix_path(model).name}, or stage {stage} unknown' >&2; false")
        return 1
    t = dict(id=f"{model}.{stage}.{row}", kind="cell", model=model, stage=stage, **rows[0])
    seq = current_seq(t)
    for k, v in cell_env(t, seq).items():
        print(f"export {k}={shlex.quote(v)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--cell-env":
        sys.exit(print_cell_env(*sys.argv[2:]))
    sys.exit(main())
