"""Round-2 proposers, first 10 forward passes side by side (one HTML chart with hover and a table view).  Owner: Claude.

    python3 -I scripts/weight_edit/plot_first10.py [--out outputs/weight_edit/claude/plots/first10_passes.html]

Reads outputs/weight_edit/proposers/{opus_r2,fable_r2,astra_r2}/requests/fwd_<k>.done.json (k = 1..10). Each generation
pass compares the proposer's edit with the base model on the same calls (128 calls x 2 draws, from the search
repositories only; no gate data), so every point is edit minus base on that pass's own call set. Pass roles (tuning
calls, held-out calls, matched random control, read) come from each proposer's notebook / final answer and are listed
in ROLES below. The compile metric is the server's frozen-new_string one, which is inflated for edits that fix
old_string (doc Thread 2026-11-05 03:1x).
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "outputs/weight_edit/proposers"
AGENTS = [("opus_r2", "Opus 5.5"), ("fable_r2", "Fable 5.1"), ("astra_r2", "Astra (gpt-6-astra)")]
# role: T tuning calls, H held-out calls (not used to fit the edit), C matched random control, R read, B base-only gen
ROLES = {
    "opus_r2": {1: ("R", "read, base"), 2: ("R", "read, base + attention L35/L41"), 3: ("R", "read, base"),
                4: ("R", "read, base"), 5: ("T", "v0, no gate"), 6: ("T", "v1, graded flag"), 7: ("R", "read, base"),
                8: ("R", "read (returned one position)"), 9: ("T", "v2, sharp gate (in-sample)"),
                10: ("H", "v3f, fit on fastapi, tested on rich")},
    "fable_r2": {1: ("R", "read, base"), 2: ("R", "read, base"), 3: ("T", "v1 rank-1"), 4: ("T", "v2 rank-3"),
                 5: ("R", "read"), 6: ("R", "read"), 7: ("T", "v3 rank-3"), 8: ("H", "v3 on natural set B"),
                 9: ("T", "v4 rank-3, boost 0.5"), 10: ("H", "v4 on natural set B")},
    "astra_r2": {1: ("B", "base generation, no edit"), 2: ("R", "read, base"), 3: ("R", "read, base"),
                 4: ("T", "broad rank-28"), 5: ("T", "compile-conscious rank-13"), 6: ("T", "rank-13 at half strength"),
                 7: ("H", "frozen candidate on validation tasks"), 8: ("C", "random control, seed 104729 (validation)"),
                 9: ("C", "random control, seed 130363 (validation)"), 10: ("R", "read with the candidate")},
}
ROLE_NAME = {"T": "tuning calls", "H": "held-out calls", "C": "matched random control", "R": "read pass",
             "B": "base generation only"}
METRICS = [("escape_artefact", "Escape artefacts", "lower is better", None),
           ("accepted", "Accepted edits", "higher is better", (-1.0, "gate guard −1 pt")),
           ("compile_fail_among_accepted", "Compile failures among accepted edits",
            "lower is better · frozen new_string, inflated for edits that fix old_string", (2.0, "gate guard +2 pt"))]


def load() -> list[dict]:
    rows = []
    for run, name in AGENTS:
        for k in range(1, 11):
            d = json.loads((P / run / "requests" / f"fwd_{k}.done.json").read_text())
            role, desc = ROLES[run][k]
            r = {"run": run, "agent": name, "pass": k, "role": role, "desc": desc, "secs": round(d["secs"])}
            arms = (d.get("result") or {}).get("arms") or {}
            if "base" in arms and "edit" in arms:
                r["n_calls"] = d["result"]["n_calls"]
                for m, *_ in METRICS:
                    b, e = arms["base"][m], arms["edit"][m]
                    r[m] = {"base": 100 * b, "edit": 100 * e, "delta": 100 * (e - b)}
            rows.append(r)
    return rows


def build(rows: list[dict]) -> str:
    W, L, R = 980, 190, 40
    xs = {k: L + (k - 0.5) * (W - L - R) / 10 for k in range(1, 11)}
    off = {"opus_r2": -13, "fable_r2": 0, "astra_r2": 13}
    svg = []
    tip = lambda r: html.escape(json.dumps(r), quote=True)

    def marker(x, y, run, role, r, hit=True):
        c = f"var(--{run})"
        if role == "T":
            m = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{c}" stroke="var(--surface)" stroke-width="2"/>'
        elif role == "H":
            m = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="var(--surface)" stroke="{c}" stroke-width="2"/>'
        elif role == "C":
            m = (f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" transform="rotate(45 {x:.1f} {y:.1f})" '
                 f'fill="var(--surface)" stroke="{c}" stroke-width="2"/>')
        elif role == "B":
            m = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="var(--muted)"/>'
        else:
            m = f'<line x1="{x - 5:.1f}" x2="{x + 5:.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--muted)" stroke-width="2" stroke-linecap="round"/>'
        if hit:
            m += f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="11" fill="transparent" data-tip="{tip(r)}"/>'
        return m

    # strip: what each pass did
    y0 = 70
    svg.append(f'<text x="{L - 180}" y="{y0 - 22}" class="ptitle">What each pass did</text>')
    for i, (run, name) in enumerate(AGENTS):
        y = y0 + i * 24
        svg.append(f'<rect x="{L - 180}" y="{y - 5}" width="10" height="10" rx="2" fill="var(--{run})"/>')
        svg.append(f'<text x="{L - 164}" y="{y + 4}" class="lab">{html.escape(name)}</text>')
        for r in rows:
            if r["run"] == run:
                svg.append(marker(xs[r["pass"]], y, run, r["role"], r))
    for k in range(1, 11):
        svg.append(f'<text x="{xs[k]:.1f}" y="{y0 - 22}" class="tick" text-anchor="middle">{k}</text>')
    top = y0 + 3 * 24 + 64
    PH, GAP = 190, 86
    for j, (m, title, sub, guard) in enumerate(METRICS):
        t = top + j * (PH + GAP)
        vals = [r[m]["delta"] for r in rows if m in r] + [0, guard[0] if guard else 0]
        lo, hi = min(vals), max(vals)
        step = 5 if hi - lo <= 30 else 10
        lo, hi = step * ((lo - 1) // step), step * (-(-(hi + 1) // step))  # one-point pad so marks at 0 clear the frame
        ymap = lambda v: t + PH - (v - lo) / (hi - lo) * PH
        svg.append(f'<text x="{L - 180}" y="{t - 26}" class="ptitle">{html.escape(title)}, edit − base (points)</text>')
        svg.append(f'<text x="{L - 180}" y="{t - 10}" class="sub">{html.escape(sub)}</text>')
        v = lo
        while v <= hi + 1e-9:
            yy = ymap(v)
            svg.append(f'<line x1="{L}" x2="{W - R}" y1="{yy:.1f}" y2="{yy:.1f}" class="{"zero" if v == 0 else "grid"}"/>')
            svg.append(f'<text x="{L - 8}" y="{yy + 4:.1f}" class="tick" text-anchor="end">{v:+.0f}</text>'.replace("+0<", "0<"))
            v += step
        if guard:
            gy = ymap(guard[0])
            svg.append(f'<line x1="{L}" x2="{W - R}" y1="{gy:.1f}" y2="{gy:.1f}" class="guard"/>')
            svg.append(f'<text x="{L + 6}" y="{gy + (15 if guard[0] < 0 else -6):.1f}" class="sub">{guard[1]}</text>')
        for k in range(1, 11):
            svg.append(f'<text x="{xs[k]:.1f}" y="{t + PH + 16}" class="tick" text-anchor="middle">{k}</text>')
        for r in rows:
            if m in r:
                svg.append(marker(xs[r["pass"]] + off[r["run"]], ymap(r[m]["delta"]), r["run"], r["role"], r))
    svg.append(f'<text x="{(L + W - R) / 2:.1f}" y="{top + 3 * (PH + GAP) - 22}" class="tick" text-anchor="middle">forward pass</text>')
    H = top + 3 * (PH + GAP) - 8
    legend = "".join(
        f'<span class="li"><svg width="16" height="16" viewBox="-8 -8 16 16">{marker(0, 0, "opus_r2" if ro != "B" else "opus_r2", ro, {}, hit=False).replace("var(--opus_r2)", "var(--ink2)")}</svg>{ROLE_NAME[ro]}</span>'
        for ro in ("T", "H", "C", "R", "B"))
    agents = "".join(f'<span class="li"><span class="sw" style="background:var(--{run})"></span>{html.escape(n)}</span>' for run, n in AGENTS)
    head = ["agent", "pass", "role", "what", "artefacts base → edit", "accepted base → edit", "compile fail base → edit", "server s"]
    trs = []
    for r in rows:
        f = lambda m: f'{r[m]["base"]:.1f} → {r[m]["edit"]:.1f} ({r[m]["delta"]:+.1f})' if m in r else "–"
        trs.append("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in
                   (r["agent"], r["pass"], ROLE_NAME[r["role"]], r["desc"], f("escape_artefact"), f("accepted"),
                    f("compile_fail_among_accepted"), r["secs"])) + "</tr>")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Round 2: first 10 passes of each proposer</title>
<style>
.viz-root{{color-scheme:light;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#9a9993;--grid:#e6e5e1;
 --opus_r2:#2a78d6;--fable_r2:#eb6834;--astra_r2:#1baf7a;background:var(--surface);color:var(--ink);
 font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:20px 24px;max-width:1000px}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;--surface:#1a1a19;
 --ink:#f0efec;--ink2:#c3c2b7;--muted:#7d7c76;--grid:#2e2d2a;--opus_r2:#3987e5;--fable_r2:#d95926;--astra_r2:#199e70}}}}
:root[data-theme="dark"] .viz-root{{color-scheme:dark;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#7d7c76;
 --grid:#2e2d2a;--opus_r2:#3987e5;--fable_r2:#d95926;--astra_r2:#199e70}}
body{{margin:0;background:#fcfcfb}} @media (prefers-color-scheme:dark){{body{{background:#1a1a19}}}}
h1{{font-size:18px;margin:0 0 4px}} p{{margin:2px 0;color:var(--ink2)}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 18px;margin:10px 0 0;color:var(--ink2)}} .li{{display:inline-flex;align-items:center;gap:6px}}
.sw{{width:12px;height:12px;border-radius:3px;display:inline-block}}
svg text{{fill:var(--ink2)}} .ptitle{{fill:var(--ink);font-weight:600;font-size:14px}} .sub{{font-size:12px}} .tick{{font-size:12px}}
.lab{{fill:var(--ink);font-size:13px}} .grid{{stroke:var(--grid);stroke-width:1}} .zero{{stroke:var(--muted);stroke-width:1.5}}
.guard{{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:5 4}}
#tip{{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--grid);border-radius:6px;
 padding:8px 10px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.15);display:none;max-width:320px}}
table{{border-collapse:collapse;font-size:12px;margin-top:8px}} td,th{{border-bottom:1px solid var(--grid);padding:3px 8px;text-align:left}}
details{{margin-top:12px;color:var(--ink2)}}
</style></head><body><div class="viz-root">
<h1>Round 2: the first 10 forward passes of each proposer</h1>
<p>Each point compares the proposer's edit with the base model on that pass's own 128 calls × 2 draws, all from the search repositories (no gate data).</p>
<p>Call sets differ between passes and proposers, so compare signs and sizes, not exact levels. Passes without a point are reads (activations or log-probs, no generation).</p>
<div class="legend">{agents}</div><div class="legend">{legend}</div>
<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Edit minus base for escape artefacts, acceptance and compile failures, passes 1 to 10, Opus, Fable and Astra">
{''.join(svg)}</svg>
<details><summary>Table view</summary><table><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr>{''.join(trs)}</table></details>
<div id="tip"></div></div>
<script>
const tip=document.getElementById('tip');const f=(o)=>o?`${{o.base.toFixed(1)}} → ${{o.edit.toFixed(1)}} (${{o.delta>=0?'+':''}}${{o.delta.toFixed(1)}})`:'–';
const RN={json.dumps(ROLE_NAME)};
document.querySelectorAll('.hit').forEach(h=>{{h.addEventListener('mousemove',e=>{{const r=JSON.parse(h.dataset.tip);
tip.innerHTML=`<b>${{r.agent}}, pass ${{r.pass}}</b><br>${{RN[r.role]}}: ${{r.desc}}<br>`+(r.accepted?`artefacts % ${{f(r.escape_artefact)}}<br>accepted % ${{f(r.accepted)}}<br>compile fail % ${{f(r.compile_fail_among_accepted)}}<br>`:'')+`server time ${{r.secs}} s`;
tip.style.display='block';tip.style.left=Math.min(e.clientX+14,innerWidth-340)+'px';tip.style.top=(e.clientY+14)+'px';}});
h.addEventListener('mouseleave',()=>tip.style.display='none');}});
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/weight_edit/claude/plots/first10_passes.html"))
    a = ap.parse_args()
    rows = load()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(rows))
    (out.with_suffix(".json")).write_text(json.dumps(rows, indent=1))
    print(f"wrote {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
