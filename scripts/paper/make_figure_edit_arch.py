"""Architecture view of the edits: where each weight edit sits in Gemma 4 E4B.

    uv run --offline --no-project --with matplotlib python -I scripts/paper/make_figure_edit_arch.py [--paper1]

(a) the 42-layer stack as seven groups of five sliding-window layers and one global layer (the framing of
Grootendorst's "A Visual Guide to Gemma 4"), with per-layer embeddings, KV sharing and the edits stacked on their
layers; (b) one decoder layer in the order of the transformers gemma4 forward pass, with the edited matrices and the
edits on each. Data: outputs/paper/figures/edit_locations/edit_locations.json (frozen edit-file headers, gate reports;
architecture from the checkpoint config and safetensors header). Our own drawing; nothing is copied from the guide.
Writes outputs/paper/figures/edit_map/edit_arch.{pdf,png}. Not placed in either writeup yet (user, 2026-11-11).
--paper1 leaves out the three Paper 2 usability-test edits (lt_*) everywhere, with every count recomputed, and writes
edit_arch_p1.{pdf,png} instead.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs/paper/figures/edit_locations/edit_locations.json"
OUT = ROOT / "outputs/paper/figures/edit_map"

INK, INK2, MUTED, GRID, AXIS, PANEL = "#0f172a", "#475569", "#8492a6", "#e2e8f0", "#c3ccd8", "#eef2f6"
EDITED_BG = "#e1eafb"
PLE_C = "#d9d2f3"
OPUS, FABLE, GRAD, ASTRA, RECIPE = "#2a78d6", "#eb6834", "#0f9d8a", "#b5179e", "#5b6b7c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.0, "pdf.fonttype": 42,
                     "figure.facecolor": "white", "axes.facecolor": "white"})

N = 42
MOD_RE = re.compile(r"layers\.(\d+)\.(self_attn|mlp)\.(\w+_proj)")
# id: (colour, hollow, live); order = stacking order in panel (a)
STYLE = {
    "r1_opus": (OPUS, False, False), "r1_fable": (FABLE, False, False), "r1_recipe": (RECIPE, False, False),
    "r2_opus": (OPUS, False, False), "r2_fable": (FABLE, False, True), "r2_astra": (ASTRA, True, False),
    "r2_opusinf": (OPUS, False, False), "r2_fable52inf_v6": (FABLE, True, False), "grad_yer39": (GRAD, False, False),
    "lt_lib": (OPUS, False, False), "lt_nolib": (OPUS, False, False), "lt_libfable": (FABLE, False, False),
}
ALL_LAYER = ("grad_tam39", "grad_tam411")
PAPER2_ONLY = ("lt_lib", "lt_nolib", "lt_libfable")  # the companion Paper 2 usability test; left out by --paper1


def and_list(items):
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def load(style):
    d = json.loads(SRC.read_text())
    arch = d["architecture"]
    assert arch["num_decoder_layers"] == N and arch["full_attention_layers"] == [5, 11, 17, 23, 29, 35, 41]
    assert arch["kv_sharing"]["kv_source"] == {"sliding shared layers 24-28, 30-34, 36-40": 22,
                                                "full shared layers 29, 35, 41": 23}
    E = {e["id"]: e for e in d["edits"]}
    fp = {}
    for k in style:
        s = set()
        for m in E[k]["modules"]:
            g = MOD_RE.search(m["name"])
            assert g, (k, m["name"])
            s.add((int(g.group(1)), g.group(3)))
        assert {l for l, _ in s} == set(E[k]["layers"]), k
        fp[k] = s
    for k in ALL_LAYER:
        assert E[k]["layers"] == list(range(N)), k
    units = {k: len(E[k]["units_layer41"]) for k in style if "units_layer41" in E[k]}  # FFN units rewired in layer 41
    return arch, fp, units


def marker(ax, x, y, attn, col, hollow, live, ms=4.6):
    ax.plot([x], [y], marker="s" if attn else "o", ms=ms if attn else ms + 0.4, mfc="white" if hollow else col,
            mec=col, mew=1.1 if hollow else 0, linestyle="none", zorder=5)
    if live:
        ax.plot([x], [y], marker="o", ms=ms + 4.2, mfc="none", mec=col, mew=0.8, linestyle="none", zorder=5)


def box(ax, x, y, w, h, text, fc, ec="none", fs=6.6, color=INK, bold=False, r=0.04):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=0.8,
                                zorder=2))
    ax.text(x + w / 2, y + h / 2, text, fontsize=fs, color=color, ha="center", va="center", zorder=3,
            fontweight="bold" if bold else "normal", family="DejaVu Sans Mono" if text.endswith("_proj") else None)


def vline(ax, x, y0, y1):
    ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7,
                                                               mutation_scale=6, shrinkA=0, shrinkB=0), zorder=1)


def figure(arch, fp, units_n, style):
    W, H = 6.3, 7.75
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    full = set(arch["full_attention_layers"])

    # ---------------- (a) the stack ----------------------------------------------------------------------------
    ax.text(0.08, H - 0.2, "(a) Gemma 4 E4B: 42 decoder layers, seven groups of five sliding-window layers + one global",
            fontsize=7.6, fontweight="bold", color=INK, va="center")
    x0, x1 = 0.92, 5.42
    unit = (x1 - x0) / (N + 6 * 0.6)
    X = [x0 + unit * (i + 0.6 * (i // 6)) + unit / 2 for i in range(N)]
    bw = unit * 0.74
    yb = H - 1.98  # bottom of layer blocks
    hl, hg = 0.30, 0.42
    for i in range(N):
        h = hg if i in full else hl
        ax.add_patch(FancyBboxPatch((X[i] - bw / 2, yb), bw, h, boxstyle="round,pad=0,rounding_size=0.015",
                                    fc=AXIS if i in full else PANEL, ec="none", zorder=2))
    for g in range(7):
        a, b = X[6 * g] - bw / 2, X[6 * g + 5] + bw / 2
        ax.plot([a, b], [yb - 0.05, yb - 0.05], color=MUTED, lw=0.6)
    for i in sorted(full | {0}):
        ax.text(X[i], yb - 0.15, str(i), fontsize=6.2, ha="center", va="center", color=INK if i in full else INK2,
                fontweight="bold" if i in full else "normal")
    # embedding in, head out
    box(ax, 0.08, yb + 0.02, 0.66, 0.38, "token\nembedding", PANEL, fs=6.2)
    ax.annotate("", xy=(X[0] - bw / 2 - 0.02, yb + 0.21), xytext=(0.75, yb + 0.21),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7, mutation_scale=6))
    box(ax, 5.56, yb + 0.02, 0.68, 0.38, "norm, head\n(tied)", PANEL, fs=6.2)
    ax.annotate("", xy=(5.555, yb + 0.21), xytext=(X[-1] + bw / 2 + 0.01, yb + 0.21),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7, mutation_scale=6))
    # gradient all-layer arms
    yg = yb - 0.36
    ax.add_patch(FancyBboxPatch((X[0] - bw / 2, yg - 0.035), X[-1] - X[0] + bw, 0.07,
                                boxstyle="round,pad=0,rounding_size=0.035", fc=GRAD, ec="none", zorder=3))
    ax.text(X[0] - bw / 2 - 0.06, yg, "gradient LoRA,\nall linear (2 arms)", fontsize=5.9, color=INK2, ha="right",
            va="center", linespacing=1.15)
    # per-layer embeddings
    yp = yb - 0.62
    for i in range(N):
        ax.plot([X[i], X[i]], [yp + 0.05, yb], color=PLE_C, lw=0.9, zorder=1)
    ax.add_patch(FancyBboxPatch((X[0] - bw / 2, yp - 0.07), X[-1] - X[0] + bw, 0.14,
                                boxstyle="round,pad=0,rounding_size=0.05", fc=PLE_C, ec="none", zorder=2))
    ax.text((X[0] + X[-1]) / 2, yp, "a 256-d slice per layer and token, added after the FFN  ·  not edited",
            fontsize=5.9, color=INK, ha="center", va="center", zorder=3)
    ax.text(X[0] - bw / 2 - 0.06, yp, "per-layer\nembeddings", fontsize=5.9, color=INK2, ha="right", va="center",
            linespacing=1.15)
    # edits stacked on their layers
    stack = {}
    for k, (col, hol, live) in style.items():
        for l in sorted({l for l, _ in fp[k]}):
            projs = {p for ll, p in fp[k] if ll == l}
            for attn in sorted({p in ("q_proj", "k_proj", "v_proj", "o_proj") for p in projs}, reverse=True):
                n = stack.get(l, 0)
                stack[l] = n + 1
                marker(ax, X[l], yb + (hg if l in full else hl) + 0.09 + n * 0.105, attn, col, hol, live)
    top41 = yb + hg + 0.09 + (stack[41] - 1) * 0.105
    ax.text(X[41] - 0.1, top41, f"layer 41:\n{stack[41]} edits", fontsize=6.0, color=INK, ha="right", va="center",
            linespacing=1.15)
    ax.text(X[23], yb + hg + 0.09 + stack[23] * 0.105 + 0.03, "round 1", fontsize=5.9, color=INK2, ha="center",
            va="bottom")
    # KV sharing
    yk = yb + 1.38
    xa, xb = X[24] - bw / 2, X[41] + bw / 2
    ax.plot([xa, xb], [yk, yk], color=MUTED, lw=0.7)
    for xx in (xa, xb):
        ax.plot([xx, xx], [yk, yk - 0.06], color=MUTED, lw=0.7)
    ax.text(xb, yk + 0.12, "layers 24–41 have no K/V weights: sliding layers reuse layer 22's keys and values, "
            "global layers layer 23's", fontsize=5.9, color=INK2, ha="right", va="center")
    # legend for (a)
    yl = yp - 0.32
    items = [(OPUS, "Opus 5.5"), (FABLE, "Fable 5.1 / 5.2"), (ASTRA, "GPT-6 Astra"), (GRAD, "gradient LoRA"),
             (RECIPE, "scripted recipe")]
    x = 0.12
    for col, lab in items:
        ax.add_patch(Rectangle((x, yl - 0.035), 0.12, 0.07, fc=col, ec="none"))
        ax.text(x + 0.16, yl, lab, fontsize=6.2, color=INK2, va="center")
        x += 0.2 + 0.052 * len(lab)
    yl2 = yl - 0.2
    x = 0.12
    for kind, lab in (("s", "attention matrix"), ("o", "FFN matrix"), ("hollow", "withdrawn or not submitted"),
                      ("live", "served in live agent runs")):
        if kind == "hollow":
            marker(ax, x + 0.05, yl2, False, INK2, True, False)
        elif kind == "live":
            marker(ax, x + 0.05, yl2, False, INK2, False, True)
        else:
            marker(ax, x + 0.05, yl2, kind == "s", INK2, False, False)
        ax.text(x + 0.16, yl2, lab, fontsize=6.2, color=INK2, va="center")
        x += 0.32 + 0.052 * len(lab)
    ax.plot([0.08, W - 0.08], [yl2 - 0.2, yl2 - 0.2], color=GRID, lw=0.8)

    # ---------------- (b) one decoder layer --------------------------------------------------------------------
    yt = yl2 - 0.38
    ax.text(0.08, yt, "(b) Inside one decoder layer, in the order of the forward pass, with the edits on each matrix",
            fontsize=7.6, fontweight="bold", color=INK, va="center")
    cx, cw = 0.25, 2.05  # flow column
    mid = cx + cw / 2
    rows = {}
    y = yt - 0.32
    ax.text(mid, y, "h from layer i − 1", fontsize=6.2, color=INK2, ha="center", va="center")
    steps = [
        ("norm", "RMSNorm"),
        ("qkv", None),
        ("attn", "attention: sliding (window 512) or global"),
        ("o", None),
        ("norm", "RMSNorm, then add to h"),
        ("norm", "RMSNorm"),
        ("gu", None),
        ("act", "GELU(gate) × up"),
        ("down", None),
        ("norm", "RMSNorm, then add to h"),
        ("ple", None),
        ("norm", "RMSNorm, add to h, × layer scalar"),
    ]
    prev = y - 0.07
    for kind, label in steps:
        h = 0.2 if kind in ("qkv", "o", "gu", "down", "ple") else 0.15
        y -= h + 0.1
        vline(ax, mid, prev, y + h)
        if kind in ("norm", "attn", "act"):
            box(ax, cx + 0.12, y, cw - 0.24, h, label, "white", ec=AXIS, fs=5.9, color=INK2, r=0.07)
        elif kind == "qkv":
            ww = (cw - 0.1) / 3
            for j, p in enumerate(("q_proj", "k_proj", "v_proj")):
                box(ax, cx + j * (ww + 0.05), y, ww, h, p, EDITED_BG, fs=6.2)
            rows["qkv"] = y + h / 2
        elif kind == "gu":
            ww = (cw - 0.05) / 2
            for j, p in enumerate(("gate_proj", "up_proj")):
                box(ax, cx + j * (ww + 0.05), y, ww, h, p, EDITED_BG, fs=6.2)
            rows["gu"] = y + h / 2
        elif kind in ("o", "down"):
            p = "o_proj" if kind == "o" else "down_proj"
            box(ax, cx + 0.35, y, cw - 0.7, h, p, EDITED_BG, fs=6.2)
            rows[kind] = y + h / 2
        elif kind == "ple":
            ww = (cw - 0.05) / 2
            box(ax, cx, y, ww, h, "PLE gate 2560→256", PLE_C, fs=5.8)
            box(ax, cx + ww + 0.05, y, ww, h, "PLE proj 256→2560", PLE_C, fs=5.8)
            ax.text(cx + cw + 0.04, y + h / 2, "× per-layer embedding;\nno edits on this path", fontsize=5.8,
                    color=INK2, va="center", linespacing=1.15)
            rows["ple"] = y + h / 2
        prev = y
    vline(ax, mid, prev, prev - 0.12)
    ax.text(mid, prev - 0.2, "h to layer i + 1", fontsize=6.2, color=INK2, ha="center", va="center")

    # annotations: which edits sit on which matrix
    ax_x = 3.32

    def note(yc, lines, attn=False):
        """lines: list of (edit ids or None, text), from the row's centre downward; dots first, then text."""
        for j, (ids, text) in enumerate(lines):
            yy = yc - j * 0.155
            x = ax_x
            for k in ids or []:
                col, hol, live = style[k]
                marker(ax, x, yy, attn, col, hol, live, ms=4.0)
                x += 0.12 if not live else 0.15
            ax.text(x + (0.04 if ids else 0), yy, text, fontsize=6.1, color=INK, va="center")
        ax.plot([cx + cw + 0.05, ax_x - 0.1], [yc, yc], color=GRID, lw=0.7, zorder=0)

    def where(proj):
        return {k for k in style if any(p == proj for _, p in fp[k])}

    assert where("q_proj") == where("k_proj") == {"r1_fable"} and where("v_proj") == {"r1_opus"}
    assert where("o_proj") == {"r1_recipe"}
    note(rows["qkv"], [(["r1_fable"], "q, k: Fable 5.1, round 1, layers 11, 17, 23"),
                       (["r1_opus"], "v: Opus 5.5, round 1, layer 23, one KV head"),
                       (None, "k and v exist only in layers 0–23")], attn=True)
    note(rows["o"], [(["r1_recipe"], "scripted recipe, round 1, layer 24 (with its down_proj)")], attn=True)
    units = [k for k in ["r2_opus", "r2_opusinf", "r2_fable52inf_v6", "lt_lib", "lt_libfable"] if k in style]
    assert where("gate_proj") == where("up_proj") == set(units) == set(units_n)
    note(rows["gu"], [(units, f"layer 41: whole FFN units ({and_list(units_n[k] for k in units)})"),
                      (None, "one unit = a row of gate and up and a column of down")])
    d41 = [k for k in style if (41, "down_proj") in fp[k]]
    assert set(d41) == set(units) | {"r2_fable", "r2_astra", "grad_yer39"}
    order41 = [k for k in ["r2_opus", "r2_fable", "r2_astra", "r2_opusinf", "r2_fable52inf_v6", "grad_yer39", "lt_lib",
                           "lt_libfable"] if k in style]
    assert set(order41) == set(d41) and len(order41) == stack[41]
    down = [(order41, f"layer 41: {len(order41)} edits"),
            (None, f"the {len(units)} unit edits; Fable round 2, rank 2, served live;"),
            (None, "Astra, rank 13, not submitted; gradient matched, rank 2"),
            (["r2_opus"], "layer 40: Opus round 2, a rank-1 flag"),
            (["lt_nolib"], "layers 30, 31: Opus without library, rank 1"),
            (["r1_recipe"], "layer 24: scripted recipe, rank 1")]
    note(rows["down"], [(ids, text) for ids, text in down if not ids or all(k in style for k in ids)])
    yf = 0.22
    ax.text(0.08, yf, "Gradient LoRA (all linear, 2 arms): rank 8 on every q, o, gate, up and down in all 42 layers "
            "and on k, v in 0–23.", fontsize=5.9, color=INK2, va="center")
    ax.text(0.08, yf - 0.15, "Order and shapes: transformers gemma4 forward pass and the E4B checkpoint (hidden 2560, "
            "FFN 10240). Edits: the frozen edit files.", fontsize=5.9, color=INK2, va="center")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper1", action="store_true", help="leave out the Paper 2 usability-test edits (lt_*)")
    a = ap.parse_args()
    style = {k: v for k, v in STYLE.items() if not (a.paper1 and k in PAPER2_ONLY)}
    arch, fp, units_n = load(style)
    OUT.mkdir(parents=True, exist_ok=True)
    fig = figure(arch, fp, units_n, style)
    name = "edit_arch_p1" if a.paper1 else "edit_arch"
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    print(OUT / f"{name}.png")


if __name__ == "__main__":
    main()
