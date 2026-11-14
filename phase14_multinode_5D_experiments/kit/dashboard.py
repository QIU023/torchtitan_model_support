"""Live dashboard of the phase 14 runs: python dashboard.py [--port 8765] [--dcgm 0=ip,1=ip] [--out DIR].

k8s.sh dash starts it in pod 0 and forwards the port; the page refreshes every 3 seconds. It shows the newest cell
(pipeline or manual): the pipeline state of each node, mean / min / max MFU and tps over the ranks per step, the
latest step line of every rank, every GPU from the nodes' DCGM exporters (utilization, SM and tensor activity, DRAM
activity, memory, power, NVLink and PCIe rates), the communication verdict, and the profiler traces and memory
snapshots the cell holds. Files are reopened on every poll, so another pod's writes show up promptly.
"""

import argparse
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from monitor import dcgm, default_out  # noqa: E402
from summarize import ANSI, cell_rows  # noqa: E402

DCGM_FIELDS = [
    ("util", "DCGM_FI_DEV_GPU_UTIL", 1.0), ("sm", "DCGM_FI_PROF_SM_ACTIVE", 1.0),
    ("tensor", "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", 1.0), ("dram", "DCGM_FI_PROF_DRAM_ACTIVE", 1.0),
    ("mem_gib", "DCGM_FI_DEV_FB_USED", 1 / 1024), ("power_w", "DCGM_FI_DEV_POWER_USAGE", 1.0),
    ("nvl_tx", "DCGM_FI_PROF_NVLINK_TX_BYTES", 1e-9), ("nvl_rx", "DCGM_FI_PROF_NVLINK_RX_BYTES", 1e-9),
    ("pcie_tx", "DCGM_FI_PROF_PCIE_TX_BYTES", 1e-9), ("pcie_rx", "DCGM_FI_PROF_PCIE_RX_BYTES", 1e-9),
]


def newest_cell(out: Path):
    logs = list(out.glob("*/*/*/node*.log")) + list(out.glob("*/manual/*/*/node*.log"))
    return max(logs, key=lambda p: p.stat().st_mtime).parent if logs else None


def steps(cell: Path) -> dict[int, list[dict]]:
    ranks: dict[int, list[dict]] = {}
    for row in cell_rows(cell)[0]:
        if row["rank"] >= 0:
            ranks.setdefault(row["rank"], []).append(row)
    return ranks


def log_tail(log: Path, size: int = 1 << 17) -> list[str]:
    with open(log, "rb") as f:
        f.seek(0, 2)
        end = f.tell()
        f.seek(max(0, end - size))
        lines = f.read().decode(errors="replace").splitlines()
    return [ANSI.sub("", line) for line in (lines[1:] if end > size else lines)][-400:]


def data(out: Path, dcgm_ips: dict[str, str]) -> dict:
    cell = newest_cell(out)
    result: dict = dict(time=time.time(), cell=str(cell) if cell else None, states={}, ranks={}, per_step=[], gpus={})
    for hb in sorted(out.glob("*/pipeline/hb/node*")):
        try:
            beat = json.loads(hb.read_text())
            result["states"][f"pipeline {hb.name}"] = f"{beat.get('state')} ({time.time() - beat.get('t', 0):.0f} s ago)"
        except (OSError, ValueError):
            pass
    if cell:
        ranks = steps(cell)
        result["ranks"] = {r: v[-1] for r, v in sorted(ranks.items())}
        by_step: dict[int, list[dict]] = {}
        for rows in ranks.values():
            for row in rows:
                by_step.setdefault(row["step"], []).append(row)
        for step in sorted(by_step):
            mfus = [r["mfu"] for r in by_step[step] if r["mfu"] is not None]
            tps = [r["tps"] for r in by_step[step]]
            secs = [r["step_s"] for r in by_step[step] if r["step_s"] is not None]
            result["per_step"].append(dict(
                step=step, ranks=len(by_step[step]),
                mfu=sum(mfus) / len(mfus) if mfus else None, mfu_min=min(mfus, default=None),
                mfu_max=max(mfus, default=None), tps=sum(tps) / len(tps) if tps else None,
                mem_max=max(r["mem"] for r in by_step[step]),
                step_s=sum(secs) / len(secs) if secs else None, step_s_max=max(secs, default=None),
            ))
        result["logs"] = {log.stem: log_tail(log) for log in sorted(cell.glob("node*.log"))}
        env = cell / "env_node0.txt"
        result["env"] = [l for l in env.read_text().splitlines() if l.startswith(("MN_CELL", "MN_FLAVOR", "MN_SEQ",
                         "MN_STEPS", "MN_COMPILE", "MN_MICROBATCHES", "NCCL_IB_HCA"))] if env.exists() else []
        result["rc"] = {p.name: p.read_text().strip() for p in sorted(cell.glob("rc_node*"))}
        result["traces"] = sorted(str(p.relative_to(cell)) for p in cell.glob("tt/profiling/traces/*/*"))[:32]
        result["snapshots"] = sorted(str(p.relative_to(cell)) for p in cell.glob("tt/profiling/memory_snapshot/*/*"))
        hw = cell.relative_to(out).parts[0]
        verdict = out / hw / "comm" / "verdict_node0.txt"
        result["comm"] = verdict.read_text()[:4000] if verdict.exists() else ""
        preflight = out / hw / "preflight" / "VERDICT.txt"
        result["preflight"] = preflight.read_text()[:4000] if preflight.exists() else ""
    for node, ip in sorted(dcgm_ips.items()):
        gpus = dcgm(ip)
        result["gpus"][node] = {
            g: {k: gpus[g].get(name, float("nan")) * scale for k, name, scale in DCGM_FIELDS} for g in sorted(gpus)
        }
    return result


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Phase 14 live</title>
<style>
body{font:13px system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}
h2{font-size:15px;margin:18px 0 6px}table{border-collapse:collapse;background:#fff}
td,th{border:1px solid #ddd;padding:2px 6px;text-align:right;font-variant-numeric:tabular-nums}
th{background:#f0f0f0}.bar{display:inline-block;height:8px;background:#4a7bd0;vertical-align:middle}
pre{background:#fff;border:1px solid #ddd;padding:6px;max-height:220px;overflow:auto}
canvas{background:#fff;border:1px solid #ddd}.muted{color:#777}
.logs{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:8px}.logs>div{min-width:0}
.logs pre{height:420px;max-height:none;font:11px ui-monospace,monospace;white-space:pre-wrap;word-break:break-all;margin:0}
.err{color:#c00;font-weight:600}.step{color:#06c}
</style></head><body>
<div id="head"></div>
<h2>Training logs of the newest cell, every node (newest at the bottom)
 <select id="lf"><option value="nowarn">hide warnings</option><option value="all">all lines</option><option value="steps">steps and errors</option></select></h2>
<div class="logs" id="logs"></div>
<h2>MFU and tps per step (mean over ranks; band = min..max MFU)</h2>
<canvas id="mfu" width="900" height="220"></canvas><br><canvas id="tps" width="900" height="160"></canvas>
<br><canvas id="secs" width="900" height="160"></canvas>
<h2>Latest step of every rank</h2><div id="ranks"></div>
<h2>GPUs (DCGM, live)</h2><div id="gpus"></div>
<h2>Profiler output of this cell</h2><div id="prof"></div>
<h2>Preflight (preflight/VERDICT.txt)</h2><pre id="preflight"></pre>
<h2>Communication check (comm/verdict_node0.txt)</h2><pre id="comm"></pre>
<script>
function f(v,d){return v==null||isNaN(v)?"-":Number(v).toFixed(d)}
function plot(id,xs,series,label){
  const c=document.getElementById(id),g=c.getContext("2d"),W=c.width,H=c.height,L=50,B=22;
  g.clearRect(0,0,W,H);const all=series.flatMap(s=>s.ys).filter(v=>v!=null);
  if(!xs.length||!all.length){g.fillText("no step lines yet",L,H/2);return}
  const x0=Math.min(...xs),x1=Math.max(...xs,x0+1),y1=Math.max(...all)*1.1||1;
  const X=x=>L+(x-x0)/(x1-x0)*(W-L-10),Y=y=>H-B-y/y1*(H-B-10);
  g.strokeStyle="#ccc";g.fillStyle="#555";
  for(let i=0;i<=4;i++){const y=y1*i/4;g.beginPath();g.moveTo(L,Y(y));g.lineTo(W-10,Y(y));g.stroke();g.fillText(f(y,y1<10?2:0),4,Y(y)+4)}
  xs.forEach(x=>g.fillText(x,X(x)-3,H-6));g.fillText(label,L+4,12);
  series.forEach(s=>{g.strokeStyle=s.color;g.lineWidth=s.width||2;g.beginPath();let on=false;
    xs.forEach((x,i)=>{const y=s.ys[i];if(y==null){on=false;return}on?g.lineTo(X(x),Y(y)):g.moveTo(X(x),Y(y));on=true});g.stroke()});
}
function bar(v,max){return '<span class="bar" style="width:'+Math.max(0,Math.min(60,60*v/max))+'px"></span> '}
async function tick(){
  let d;try{d=await (await fetch("data")).json()}catch(e){document.getElementById("head").textContent="no data: "+e;return}
  const st=Object.entries(d.states).map(([k,v])=>k+": "+v).join(" | ");
  document.getElementById("head").innerHTML="<b>"+(d.cell||"no cell yet")+"</b><br>"+st+
    "<br><span class=muted>"+(d.env||[]).join("  ")+"  rc: "+JSON.stringify(d.rc||{})+"  updated "+new Date(d.time*1000).toLocaleTimeString()+"</span>";
  const ps=d.per_step||[],xs=ps.map(p=>p.step);
  plot("mfu",xs,[{ys:ps.map(p=>p.mfu_min),color:"#bbb",width:1},{ys:ps.map(p=>p.mfu_max),color:"#bbb",width:1},{ys:ps.map(p=>p.mfu),color:"#d04a4a"}],"MFU %");
  plot("tps",xs,[{ys:ps.map(p=>p.tps),color:"#4a7bd0"}],"tps per rank (mean)");
  plot("secs",xs,[{ys:ps.map(p=>p.step_s_max),color:"#bbb",width:1},{ys:ps.map(p=>p.step_s),color:"#2a9d55"}],"step time s (mean over ranks; grey = slowest rank; step 1 includes startup and compilation)");
  let h="<table><tr><th>rank</th><th>step</th><th>step s</th><th>loss</th><th>mem GiB</th><th>tps</th><th>TFLOPS</th><th>MFU %</th></tr>";
  for(const [r,v] of Object.entries(d.ranks||{}))h+="<tr><td>"+r+"</td><td>"+v.step+"</td><td>"+f(v.step_s,2)+"</td><td>"+(v.loss<0?'<span class=muted>no loss stage</span>':f(v.loss,4))+"</td><td>"+f(v.mem,1)+"</td><td>"+f(v.tps,0)+"</td><td>"+f(v.tflops,1)+"</td><td>"+f(v.mfu,2)+"</td></tr>";
  document.getElementById("ranks").innerHTML=h+"</table>";
  h="<table><tr><th>node</th><th>gpu</th><th>util %</th><th>SM active</th><th>tensor active</th><th>DRAM active</th><th>mem GiB</th><th>power W</th><th>NVLink tx/rx GB/s</th><th>PCIe tx/rx GB/s</th></tr>";
  for(const [n,gs] of Object.entries(d.gpus||{}))for(const [g,m] of Object.entries(gs))
    h+="<tr><td>"+n+"</td><td>"+g+"</td><td>"+bar(m.util,100)+f(m.util,0)+"</td><td>"+bar(m.sm,1)+f(m.sm,2)+"</td><td>"+bar(m.tensor,1)+f(m.tensor,2)+
       "</td><td>"+f(m.dram,2)+"</td><td>"+f(m.mem_gib,1)+"</td><td>"+f(m.power_w,0)+"</td><td>"+f(m.nvl_tx,1)+" / "+f(m.nvl_rx,1)+"</td><td>"+f(m.pcie_tx,1)+" / "+f(m.pcie_rx,1)+"</td></tr>";
  document.getElementById("gpus").innerHTML=Object.keys(d.gpus||{}).length?h+"</table><span class=muted>PCIe carries the inter-node traffic under GPUDirect RDMA.</span>":"no DCGM data (start with --dcgm)";
  const t=(d.traces||[]).length,s=(d.snapshots||[]).length;
  document.getElementById("prof").innerHTML=(t?t+" trace files (measure's profiled step): k8s.sh trace &lt;stage&gt; &lt;row&gt; merges them for ui.perfetto.dev<br>":"no profiler trace in this cell (measure writes one at its profiled step)<br>")+
    (s?s+" memory snapshots: k8s.sh fetch the cell's tt/profiling/memory_snapshot for pytorch.org/memory_viz":"no memory snapshot (probe writes one at step 2)");
  document.getElementById("comm").textContent=d.comm||"";
  document.getElementById("preflight").textContent=d.preflight||"no preflight verdict yet";
  showLogs(d.logs||{});
}
const ERR=/Traceback|Error|error:|OutOfMemory|out of memory|Timeout|timed out|Watchdog|NCCL WARN|exitcode|SIGTERM|Killed/;
function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;")}
function showLogs(logs){
  const mode=document.getElementById("lf").value;
  const box=document.getElementById("logs");
  const nodes=Object.keys(logs).map(k=>+k.slice(4)).sort((a,b)=>a-b);
  for(const n of nodes.length?nodes:[0]){
    if(!document.getElementById("log"+n)){const div=document.createElement("div");
      div.innerHTML="<b>node "+n+" (ranks "+8*n+"-"+(8*n+7)+")</b><pre id=log"+n+"></pre>";box.appendChild(div)}
    let lines=logs["node"+n]||[];
    if(mode==="nowarn"){const keep=[];let skipNext=false;
      for(const l of lines){if(skipNext){skipNext=false;continue}
        if(/Warning:/.test(l)&&!ERR.test(l)){skipNext=true;continue}keep.push(l)}lines=keep}
    if(mode==="steps")lines=lines.filter(l=>/step:/.test(l)||ERR.test(l)||/\[multinode_5d\]/.test(l));
    const el=document.getElementById("log"+n),atEnd=el.scrollTop+el.clientHeight>=el.scrollHeight-20;
    el.innerHTML=lines.slice(-250).map(l=>ERR.test(l)?'<span class="err">'+esc(l)+'</span>':/step:/.test(l)?'<span class="step">'+esc(l)+'</span>':esc(l)).join("\n")||"(no log yet)";
    if(atEnd)el.scrollTop=el.scrollHeight;
  }
}
document.getElementById("lf").onchange=()=>tick();
tick();setInterval(tick,3000);
</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dcgm", default="")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or default_out()
    ips = dict(item.split("=", 1) for item in args.dcgm.split(",") if "=" in item)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/data"):
                body, kind = json.dumps(data(out, ips)).encode(), "application/json"
            else:
                body, kind = PAGE.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
