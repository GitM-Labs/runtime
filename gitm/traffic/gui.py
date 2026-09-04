"""A localhost viewer for the traffic library. stdlib only, one page, read-only.

``python -m gitm.traffic --gui`` and open the printed URL. Describe a trace,
replay it and see the validation table with the arrival profiles drawn properly,
or sweep the parameterized grid.

**It is a viewer, not a control panel.** Everything here is CPU-only. It does not
fire traffic — that is :mod:`gitm.traffic.runner` (``--fire`` on the CLI), which
needs vLLM and a live server. The viewer deliberately does not reach for it:
firing from a browser page means a long-running subprocess behind a synchronous
handler and a result nobody is waiting for. Saying so up front is the point — a
page with a "replay" button that only writes a file is a page someone will assume
hit a server.

No Flask, no React, no build step: the library's functions already return
pydantic models that serialize straight to JSON, so the server is a thin shell
around ``model_dump`` and the page is one string.

Three things this gets right because it is browser-reachable, even on loopback:

* **Binds 127.0.0.1 only.** Never ``0.0.0.0`` — that would expose a filesystem
  reader to the network the moment someone runs it on a shared box.
* **Never accepts a path from the form.** The client sends a *name*, chosen from
  a list the server produced; the server joins it under one configured root and
  re-checks containment after resolving. Path traversal is a real vector here,
  and "it's only localhost" has never been a defence.
* **Checks the Host header.** A page on any other origin can still POST to
  ``127.0.0.1`` via DNS rebinding. Requests whose Host is not loopback are
  refused.
"""

from __future__ import annotations

import json
import os
import tempfile
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from gitm.traffic.adapters import ADAPTERS
from gitm.traffic.parameterize import fit, grid
from gitm.traffic.regime import DEFAULT_BIN_S, Regime, SourceKind
from gitm.traffic.replay import read_timed_trace, write_timed_trace
from gitm.traffic.validate import REPLAY_THRESHOLDS, compare

#: Loopback only. Not a default to be overridden — a constant, so binding
#: anywhere else is an edit somebody has to justify in a diff.
HOST = "127.0.0.1"

#: Upper bound on rows read per request. The page is a viewer; a 1.4 M-row file
#: behind a synchronous handler is a hung browser tab, not a feature.
MAX_ROWS_CAP = 20_000

#: Where traces may be read from. One directory, resolved once at startup.
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "benchmarks" / "traffic_replay" / "fixtures"

_SUFFIX_ADAPTER = {".csv": "burstgpt", ".jsonl": "mooncake"}


class _Rejected(Exception):
    """A request that will not be served, with the reason shown to the user."""


def _resolve(root: Path, name: str) -> Path:
    """Resolve a client-supplied trace *name* under ``root``, or refuse.

    Two independent guards, because either alone has a bypass: rejecting
    separators stops the obvious ``../..``, and the containment check after
    ``resolve()`` catches what symlinks and Windows path quirks let through.
    """
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise _Rejected(f"not a trace name: {name!r}")
    path = (root / name).resolve()
    if path.parent != root or not path.is_file():
        raise _Rejected(f"{name!r} is not a file under the configured trace root")
    return path


def _list_traces(root: Path) -> list[dict]:
    return sorted(
        (
            {
                "name": p.name,
                "bytes": p.stat().st_size,
                "adapter": _SUFFIX_ADAPTER.get(p.suffix, ""),
            }
            for p in root.iterdir()
            if p.is_file() and p.suffix in _SUFFIX_ADAPTER
        ),
        key=lambda d: d["name"],
    )


def _load(root: Path, body: dict):
    """Adapter + trace, from a validated request body."""
    adapter = body.get("adapter", "")
    if adapter not in ADAPTERS:
        raise _Rejected(f"unknown adapter {adapter!r}")
    path = _resolve(root, body.get("trace", ""))
    max_rows = body.get("max_rows") or None
    if max_rows is not None:
        max_rows = max(1, min(int(max_rows), MAX_ROWS_CAP))
    return ADAPTERS[adapter](path, max_rows=max_rows)


def _regime_of(trace, body: dict) -> Regime:
    concurrency = body.get("concurrency") or None
    return Regime.from_trace(
        trace,
        source_kind=SourceKind(body.get("source_kind", "production")),
        bin_s=float(body.get("bin_s") or DEFAULT_BIN_S),
        concurrency=int(concurrency) if concurrency else None,
    )


def _describe(root: Path, body: dict) -> dict:
    trace = _load(root, body)
    regime = _regime_of(trace, body)
    return {
        "meta": trace.meta.model_dump(mode="json"),
        "summary": trace.meta.summary(),
        "regime": regime.model_dump(mode="json"),
        "label": regime.label(),
        "regime_summary": regime.summary(),
    }


def _replay(root: Path, body: dict) -> dict:
    trace = _load(root, body)
    # The output path is ours, never the client's: writing where a form says to
    # is the same vulnerability as reading where it says to.
    with tempfile.TemporaryDirectory(prefix="gitm-gui-") as tmp:
        out = Path(tmp) / "replay.jsonl"
        plan = write_timed_trace(trace, out)
        report = compare(trace, read_timed_trace(out), thresholds=REPLAY_THRESHOLDS)
    return {
        "plan": plan.model_dump(mode="json"),
        "argv": plan.bench_serve_argv(model=body.get("model") or "MODEL"),
        "report": report.model_dump(mode="json"),
        "passed": report.passed,
        "explain": report.explain(),
    }


def _sweep(root: Path, body: dict) -> dict:
    trace = _load(root, body)
    fitted = fit(trace, bin_s=float(body.get("bin_s") or DEFAULT_BIN_S))
    rows = [
        {
            "label": reg.label(),
            "requests": len(sampled),
            "rate_rps": reg.rate_rps,
            "burstiness": reg.burstiness,
            "input_p50": reg.input_p50,
            "output_p50": reg.output_p50,
            "in_envelope": reg.in_envelope,
        }
        for sampled, reg in grid(fitted)
    ]
    return {"fit": fitted.model_dump(mode="json"), "rows": rows}


ROUTES = {"/api/describe": _describe, "/api/replay": _replay, "/api/sweep": _sweep}


class _Handler(BaseHTTPRequestHandler):
    server_version = "gitm-traffic-viewer"
    root: Path = DEFAULT_ROOT

    def log_message(self, fmt, *args):  # noqa: A003 - stdlib hook name
        pass  # the terminal belongs to whatever else is running

    def _host_is_loopback(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        return host in {"127.0.0.1", "localhost", "::1"}

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # The page is self-contained; nothing should be able to load anything.
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
        if not self._host_is_loopback():
            return self._json(403, {"error": "non-loopback Host header refused"})
        if self.path == "/":
            return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if self.path == "/api/traces":
            return self._json(200, {"root": str(self.root), "traces": _list_traces(self.root)})
        self._json(404, {"error": "no such path"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib hook name
        if not self._host_is_loopback():
            return self._json(403, {"error": "non-loopback Host header refused"})
        handler = ROUTES.get(self.path)
        if handler is None:
            return self._json(404, {"error": "no such path"})
        try:
            n = min(int(self.headers.get("Content-Length") or 0), 64 * 1024)
            body = json.loads(self.rfile.read(n) or b"{}")
            return self._json(200, handler(self.root, body))
        except _Rejected as exc:
            return self._json(400, {"error": str(exc)})
        except Exception as exc:  # a bad adapter/threshold error belongs on the page
            return self._json(400, {"error": f"{type(exc).__name__}: {exc}"})


def serve(port: int = 8765, root: Path | None = None, *, open_browser: bool = True) -> int:
    """Run the viewer until interrupted."""
    _Handler.root = (root or Path(os.environ.get("GITM_TRAFFIC_FIXTURES", DEFAULT_ROOT))).resolve()
    if not _Handler.root.is_dir():
        raise SystemExit(f"trace root does not exist: {_Handler.root}")
    httpd = ThreadingHTTPServer((HOST, port), _Handler)
    url = f"http://{HOST}:{httpd.server_address[1]}/"
    print(f"traffic viewer on {url}\n  trace root: {_Handler.root}\n"
          f"  read-only: it does not fire traffic (that is --fire, which needs vLLM)\n"
          f"  ctrl-c to stop")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


PAGE = r"""<!doctype html>
<meta charset="utf-8"><title>gitm traffic viewer</title>
<style>
:root{--bg:#fbfbfa;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e0dedb;--card:#fff;--ok:#1f7a4d;--bad:#b3261e;--accent:#2b5fd9}
@media(prefers-color-scheme:dark){:root{--bg:#16181a;--fg:#e8e6e3;--mut:#9a9a9a;--line:#2e3235;--card:#1d2023;--ok:#4ec98a;--bad:#f2836b;--accent:#7aa2f7}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{padding:14px 20px;border-bottom:1px solid var(--line)}
h1{margin:0;font-size:15px;letter-spacing:.02em}
.sub{color:var(--mut);font-size:12px;margin-top:3px}
main{padding:16px 20px;max-width:1100px}
form{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end;margin-bottom:14px}
label{display:flex;flex-direction:column;gap:3px;font-size:11px;color:var(--mut)}
select,input{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:5px 7px;font:inherit;font-size:12px}
button{background:var(--accent);color:#fff;border:0;border-radius:4px;padding:6px 12px;font:inherit;font-size:12px;cursor:pointer}
button.alt{background:transparent;color:var(--fg);border:1px solid var(--line)}
button:disabled{opacity:.45;cursor:default}
section{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:12px 14px;margin-bottom:12px;overflow-x:auto}
h2{margin:0 0 8px;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut)}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{text-align:left;padding:3px 10px 3px 0;white-space:nowrap}
th{color:var(--mut);font-weight:500;cursor:pointer;user-select:none}
tbody tr:hover{background:color-mix(in srgb,var(--accent) 8%,transparent)}
.num{text-align:right;font-variant-numeric:tabular-nums}
.ok{color:var(--ok)}.bad{color:var(--bad)}
.tag{display:inline-block;border:1px solid var(--line);border-radius:3px;padding:0 5px;font-size:11px;color:var(--mut)}
.err{color:var(--bad);white-space:pre-wrap}
pre{margin:0;white-space:pre-wrap;word-break:break-all;font-size:11.5px;color:var(--mut)}
.k{color:var(--mut)}
svg{display:block;width:100%;height:110px}
.note{color:var(--mut);font-size:11.5px;margin-top:6px}
</style>
<header>
  <h1>gitm — traffic viewer</h1>
  <div class="sub">read-only. describe, replay-and-validate, sweep. it does not fire traffic — that is <code>--fire</code> on the CLI, which needs vLLM and a live server.</div>
</header>
<main>
<form id="f">
  <label>trace<select id="trace"></select></label>
  <label>adapter<select id="adapter"><option>burstgpt</option><option>mooncake</option></select></label>
  <label>max rows<input id="max_rows" type="number" min="1" max="20000" placeholder="all" style="width:80px"></label>
  <label>source kind<select id="source_kind"><option>production</option><option>synthetic</option><option>scoreboard</option></select></label>
  <label>bin s<input id="bin_s" type="number" step="0.1" min="0.1" value="1" style="width:70px"></label>
  <label>concurrency<input id="concurrency" type="number" min="1" placeholder="open" style="width:80px"></label>
  <button id="b1">describe</button>
  <button id="b2" class="alt">replay + validate</button>
  <button id="b3" class="alt">sweep</button>
</form>
<div id="out"></div>
</main>
<script>
const $=i=>document.getElementById(i), out=$("out");
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const num=(v,d=3)=>typeof v==="number"?(Number.isInteger(v)?v.toLocaleString():v.toFixed(d)):esc(v);

async function api(path,body){
  const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const j=await r.json(); if(!r.ok||j.error) throw new Error(j.error||("HTTP "+r.status)); return j;
}
function body(){return {trace:$("trace").value,adapter:$("adapter").value,
  max_rows:$("max_rows").value?+$("max_rows").value:null,source_kind:$("source_kind").value,
  bin_s:+$("bin_s").value||1,concurrency:$("concurrency").value?+$("concurrency").value:null};}

function chart(a,b,binS){
  const n=Math.max(a.length,b.length); if(!n) return "";
  const max=Math.max(1,...a,...b), W=1000,H=100,dx=W/Math.max(1,n-1);
  const line=xs=>xs.map((v,i)=>`${(i*dx).toFixed(1)},${(H-v/max*H).toFixed(1)}`).join(" ");
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    <polyline fill="none" stroke="var(--accent)" stroke-width="2" points="${line(a)}"/>
    <polyline fill="none" stroke="var(--ok)" stroke-width="2" stroke-dasharray="5 4" points="${line(b)}"/></svg>
    <div class="note">arrivals per ${binS}s bin — <span style="color:var(--accent)">source</span>,
    <span style="color:var(--ok)">replayed (dashed)</span>. peak ${max}.</div>`;
}
function kv(o,keys){return `<table><tbody>${keys.map(k=>
  `<tr><td class="k">${esc(k)}</td><td class="num">${num(o[k])}</td></tr>`).join("")}</tbody></table>`;}

function sortable(el){
  el.querySelectorAll("th").forEach((th,i)=>th.onclick=()=>{
    const tb=el.querySelector("tbody"), rows=[...tb.rows];
    const dir=th.dataset.d==="1"?-1:1; el.querySelectorAll("th").forEach(x=>x.dataset.d="");
    th.dataset.d=dir===1?"1":"";
    rows.sort((x,y)=>{const a=x.cells[i].dataset.v??x.cells[i].textContent,
      b=y.cells[i].dataset.v??y.cells[i].textContent;
      const fa=parseFloat(a),fb=parseFloat(b);
      return (isNaN(fa)||isNaN(fb)?String(a).localeCompare(String(b)):fa-fb)*dir;});
    rows.forEach(r=>tb.appendChild(r));
  });
}

async function run(fn){ out.innerHTML='<section>working…</section>';
  try{ await fn(); }catch(e){ out.innerHTML=`<section><h2>error</h2><div class="err">${esc(e.message)}</div></section>`; } }

$("b1").onclick=e=>{e.preventDefault();run(async()=>{
  const d=await api("/api/describe",body()), m=d.meta, drops=Object.entries(m.drops||{});
  out.innerHTML=`
  <section><h2>trace</h2><pre>${esc(d.summary)}</pre>
    ${kv(m,["rows_read","rows_emitted","span_s","raw_time_unit","prefix_block_tokens","session_rows","sessions"])}
    <div class="note">sha256 ${esc(m.sha256)}</div>
    ${(m.notes||[]).map(n=>`<div class="note">note: ${esc(n)}</div>`).join("")}</section>
  <section><h2>drops</h2>${drops.length?`<table><tbody>${drops.map(([k,v])=>
      `<tr><td class="k">${esc(k)}</td><td class="num">${v}</td></tr>`).join("")}</tbody></table>`
      :`<div class="note">none — every row emitted</div>`}</section>
  <section><h2>regime</h2><pre>${esc(d.regime_summary)}</pre>
    <div class="tag">${esc(d.label)}</div>
    ${kv(d.regime,["requests","rate_rps","io_ratio","input_p50","input_p95","output_p50","output_p95","burstiness","bin_s"])}</section>`;
});};

$("b2").onclick=e=>{e.preventDefault();run(async()=>{
  const d=await api("/api/replay",body()), r=d.report;
  out.innerHTML=`
  <section><h2>validation — ${esc(r.standard)} standard
      <span class="${d.passed?"ok":"bad"}">${d.passed?"PASS":"FAIL"}</span></h2>
    <table><thead><tr><th>check</th><th>statistic</th><th>threshold</th><th>result</th><th>detail</th></tr></thead><tbody>
    ${r.checks.map(c=>`<tr><td>${esc(c.name)}</td><td class="num">${num(c.statistic,6)}</td>
      <td class="num">${num(c.threshold,6)}</td>
      <td class="${c.passed?"ok":"bad"}">${c.passed?"ok":"FAIL"}</td>
      <td class="k">${esc(c.detail||"")}</td></tr>`).join("")}
    </tbody></table></section>
  <section><h2>arrival profile</h2>${chart(r.source_hist,r.replayed_hist,r.hist_bin_s)}</section>
  <section><h2>explanation</h2><pre>${esc(d.explain)}</pre></section>
  <section><h2>plan</h2>
    ${kv(d.plan,["requests","span_s","chunk_hash_size","self_timed","prefix_synthesized"])}
    ${(d.plan.notes||[]).map(n=>`<div class="note">note: ${esc(n)}</div>`).join("")}
    <div class="note" style="margin-top:8px">this command is <b>not</b> run — the viewer has no runner:</div>
    <pre>${esc(d.argv.join(" "))}</pre></section>`;
});};

$("b3").onclick=e=>{e.preventDefault();run(async()=>{
  const d=await api("/api/sweep",body());
  out.innerHTML=`<section><h2>parameterized grid — ${d.rows.length} points
    <span class="tag">${d.rows.filter(r=>!r.in_envelope).length} beyond the envelope</span></h2>
    <table id="g"><thead><tr><th>regime label</th><th>req</th><th>rps</th><th>D</th>
      <th>in p50</th><th>out p50</th><th>envelope</th></tr></thead><tbody>
    ${d.rows.map(r=>`<tr>
      <td data-v="${esc(r.label)}">${esc(r.label)}</td>
      <td class="num" data-v="${r.requests}">${r.requests.toLocaleString()}</td>
      <td class="num" data-v="${r.rate_rps}">${r.rate_rps.toFixed(3)}</td>
      <td class="num" data-v="${r.burstiness}">${r.burstiness.toFixed(2)}</td>
      <td class="num" data-v="${r.input_p50}">${r.input_p50.toLocaleString()}</td>
      <td class="num" data-v="${r.output_p50}">${r.output_p50.toLocaleString()}</td>
      <td data-v="${r.in_envelope?1:0}">${r.in_envelope?"in":'<span class="bad">/xenv</span>'}</td>
    </tr>`).join("")}</tbody></table>
    <div class="note">click a header to sort. <b>/xenv</b> = sampled beyond any observed trace.</div></section>`;
  sortable($("g"));
});};

fetch("/api/traces").then(r=>r.json()).then(d=>{
  $("trace").innerHTML=d.traces.map(t=>
    `<option value="${esc(t.name)}" data-a="${esc(t.adapter)}">${esc(t.name)} (${t.bytes.toLocaleString()} B)</option>`).join("");
  const sync=()=>{const o=$("trace").selectedOptions[0]; if(o&&o.dataset.a) $("adapter").value=o.dataset.a;};
  $("trace").onchange=sync; sync();
  document.querySelector(".sub").textContent+=`  •  root: ${d.root}`;
});
</script>
"""
