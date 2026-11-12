"""Edit map: which editor changed which layers and blocks of Gemma 4 E4B, plus two Kaggle Writeup cards derived from it.

    uv run --offline --no-project --with matplotlib python -I scripts/paper/make_figure_edit_map.py

Reads outputs/paper/figures/edit_locations/edit_locations.json (compiled 2026-11-11 from the frozen edit files'
safetensors/npz headers, FROZEN.txt hashes and gate reports; see edit_locations.md next to it) and writes
outputs/paper/figures/edit_map/edit_map.{pdf,png} and outputs/paper/figures/edit_map/card_paper{1,2}.png (1120 x 560).
Not yet placed in either writeup (user, 2026-11-11).
"""

from __future__ import annotations

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
HILITE = "#e1eafb"
# editor identity (validated with the dataviz palette checker, light mode, all pairs: CVD and normal-vision pass)
OPUS, FABLE, GRAD, ASTRA, RECIPE = "#2a78d6", "#eb6834", "#0f9d8a", "#b5179e", "#5b6b7c"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42,
                     "figure.facecolor": "white", "axes.facecolor": "white"})

N = 42
LAYER_RE = re.compile(r"layers\.(\d+)\.(self_attn|mlp)\.")

# lanes: (edit ids, label, detail, colour, hollow, live)
GROUPS = [
    ("Round 1  ·  target: old text found", [
        (["r1_opus"], "Opus 5.5", "v_proj, rank 64, one KV head", OPUS, False, False),
        (["r1_fable"], "Fable 5.1", "q and k, six rank-1 terms", FABLE, False, False),
        (["r1_recipe"], "Scripted recipe", "o_proj and down_proj, rank 1", RECIPE, False, False),
    ]),
    ("Round 2  ·  target: escape artefacts", [
        (["r2_opus"], "Opus 5.5", "40 FFN units; rank 1 at layer 40", OPUS, False, False),
        (["r2_fable"], "Fable 5.1", "down_proj, rank 2; deployed live", FABLE, False, True),
        (["r2_astra"], "GPT-6 Astra", "down_proj, rank 13; not submitted", ASTRA, True, False),
        (["r2_opusinf"], "Opus 5.5, informed", "80 FFN units", OPUS, False, False),
        (["r2_fable52inf_v6"], "Fable 5.2, informed", "86 FFN units; withdrawn", FABLE, True, False),
    ]),
    ("Gradient training on the same calls", [
        (["grad_yer39"], "Matched to Fable's edit", "down_proj, rank 2", GRAD, False, False),
        (["grad_tam39", "grad_tam411"], "All linear layers (2 arms)", "rank 8 in 258 matrices", GRAD, False, False),
    ]),
    ("Paper 2 usability test  ·  new target", [
        (["lt_lib"], "Opus 5.5, with library", "2 FFN units", OPUS, False, False),
        (["lt_nolib"], "Opus 5.5, without library", "down_proj, rank 1, two layers", OPUS, False, False),
        (["lt_libfable"], "Fable 5.2, with library", "1 FFN unit", FABLE, False, False),
    ]),
]


def footprint(edit):
    """Set of (layer, 'attn'|'ffn') touched, read from the stored module names."""
    names = [m["name"] for m in edit["modules"]]
    if any("{0..41}" in n or "same 258 modules" in n for n in names):  # all-linear gradient arms
        assert edit["layers"] == list(range(N)), edit["id"]
        return {(i, f) for i in range(N) for f in ("attn", "ffn")}
    out = set()
    for n in names:
        m = LAYER_RE.search(n)
        assert m, n
        out.add((int(m.group(1)), "attn" if m.group(2) == "self_attn" else "ffn"))
    assert {l for l, _ in out} == set(edit["layers"]), (edit["id"], out, edit["layers"])
    return out


def load():
    d = json.loads(SRC.read_text())
    arch = d["architecture"]
    assert arch["num_decoder_layers"] == N
    edits = {e["id"]: e for e in d["edits"]}
    lanes = []
    for title, rows in GROUPS:
        lanes.append(("group", title))
        for ids, label, detail, col, hollow, live in rows:
            fps = [footprint(edits[i]) for i in ids]
            assert all(fp == fps[0] for fp in fps), ids  # merged lanes must share one footprint
            lanes.append(("lane", (label, detail, col, hollow, live, fps[0])))
    return arch, lanes


def schematic(ax, y, full, shared, x0=0.0, w=0.78, h=0.36, gap=0.06, dark=False):
    """42 layer columns: attention cell over feed-forward cell. Returns (attn_y, ffn_y) centres."""
    a_full, a_slide, ffn = ("#3b4a5a", "#1f2b3a", "#1f2b3a") if dark else (AXIS, PANEL, PANEL)
    ay, fy = y + gap / 2 + h / 2, y - gap / 2 - h / 2
    for i in range(N):
        x = x0 + i - w / 2
        ax.add_patch(Rectangle((x, ay - h / 2), w, h, fc=a_full if i in full else a_slide, ec="none", zorder=1))
        ax.add_patch(Rectangle((x, fy - h / 2), w, h, fc=ffn, ec="none", zorder=1))
    return ay, fy


def edit_map(arch, lanes):
    full = set(arch["full_attention_layers"])
    shared = set(range(24, N))
    PITCH = 0.245  # inches per lane
    XL, XR = -24.0, 42.8
    W = 6.3
    n_lane = sum(k == "lane" for k, _ in lanes)
    n_group = sum(k == "group" for k, _ in lanes)
    top_units = 3.25  # schematic, labels and bracket
    units = top_units + n_lane + 0.95 * n_group + 1.6
    H = units * PITCH
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(XL, XR)
    ax.set_ylim(-units, 0)
    ax.axis("off")

    # --- schematic ---------------------------------------------------------------------------------------------
    y = -1.75
    ax.text(XL + 0.6, -0.55, "Gemma 4 E4B, 42 decoder layers", fontsize=8.5, fontweight="bold", color=INK, va="center")
    ax.annotate("", xy=(23.7, -0.85), xytext=(41.3, -0.85),
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=0, shrinkB=0))
    for xx in (23.7, 41.3):
        ax.plot([xx, xx], [-0.85, -1.0], color=MUTED, lw=0.8)
    ax.text(41.3, -0.6, "layers 24–41 reuse the keys and values of layer 22 or 23",
            fontsize=6.3, color=INK2, ha="right", va="center")
    ay, fy = schematic(ax, y, full, shared)
    ax.text(-1.0, ay, "Attention", fontsize=7, color=INK2, ha="right", va="center")
    ax.text(-1.0, fy, "Feed-forward (FFN)", fontsize=7, color=INK2, ha="right", va="center")
    for i in range(N):
        if i in full or i == 0:
            ax.text(i, y - 0.72, str(i), fontsize=6.3, color=INK if i in full else INK2, ha="center", va="center",
                    fontweight="bold" if i in full else "normal")
    ax.add_patch(Rectangle((XL + 0.6, y - 1.38), 0.8, 0.32, fc=AXIS, ec="none"))
    ax.text(XL + 1.8, y - 1.22, "full attention (bold numbers)", fontsize=6.3, color=INK2, va="center")
    ax.add_patch(Rectangle((XL + 18.6, y - 1.38), 0.8, 0.32, fc=PANEL, ec="none"))
    ax.text(XL + 19.8, y - 1.22, "sliding window", fontsize=6.3, color=INK2, va="center")

    # --- lanes -------------------------------------------------------------------------------------------------
    cur = -top_units
    lane_top = cur
    n41 = n_edits = 0
    ys = []
    for kind, item in lanes:
        if kind == "group":
            cur -= 0.95
            ax.text(XL + 0.6, cur + 0.28, item, fontsize=7, color=INK, fontweight="bold", va="center")
            ax.plot([XL + 0.6, XR - 0.3], [cur + 0.02, cur + 0.02], color=GRID, lw=0.7)
            continue
        label, detail, col, hollow, live, fp = item
        cur -= 1
        yc = cur + 0.5
        ys.append(yc)
        n_edits += 1
        n41 += (41, "ffn") in fp
        ax.text(-1.0, yc + 0.17, label, fontsize=7.2, color=INK, ha="right", va="center")
        ax.text(-1.0, yc - 0.2, detail, fontsize=6.2, color=INK2, ha="right", va="center")
        ax.plot([-0.5, N - 0.5], [yc, yc], color=GRID, lw=0.5, zorder=0)
        layers = {l for l, _ in fp}
        if len(layers) == N:  # all-linear arms: two bands
            for off in (0.16, -0.16):
                ax.add_patch(FancyBboxPatch((-0.35, yc + off - 0.07), N - 0.3, 0.14,
                                            boxstyle="round,pad=0,rounding_size=0.07", fc=col, ec="none", zorder=3))
            continue
        for (l, f) in sorted(fp):
            off, mk, ms = (0.16, "s", 5.0) if f == "attn" else (-0.16, "o", 5.4)
            ax.plot([l], [yc + off], marker=mk, ms=ms, mfc="white" if hollow else col, mec=col,
                    mew=1.3 if hollow else 0.0, zorder=4, linestyle="none")
            if live:
                ax.plot([l], [yc + off], marker="o", ms=9.5, mfc="none", mec=col, mew=0.9, zorder=4)
    lane_bot = cur
    # highlight layer 41 through the lanes and the schematic
    ax.add_patch(Rectangle((40.55, lane_bot - 0.1), 0.9, y + 0.5 - lane_bot + 0.1, fc=HILITE, ec="none", zorder=0))
    ax.text(41.45, lane_bot - 0.45, f"{n41} of {n_edits} edits change layer 41's FFN", fontsize=6.5, color=INK,
            ha="right", va="center")

    # --- legend ------------------------------------------------------------------------------------------------
    ly = lane_bot - 1.15
    x = XL + 0.6
    for mk, ms, txt in (("s", 5.0, "attention matrix"), ("o", 5.4, "FFN matrix")):
        ax.plot([x], [ly], marker=mk, ms=ms, color=INK2, linestyle="none")
        ax.text(x + 0.8, ly, txt, fontsize=6.3, color=INK2, va="center")
        x += 9.5
    ax.plot([x], [ly], marker="o", ms=5.4, mfc="white", mec=INK2, mew=1.2, linestyle="none")
    ax.text(x + 0.8, ly, "withdrawn or not submitted", fontsize=6.3, color=INK2, va="center")
    x += 17.5
    ax.plot([x], [ly], marker="o", ms=5.4, color=INK2, linestyle="none")
    ax.plot([x], [ly], marker="o", ms=9.5, mfc="none", mec=INK2, mew=0.9, linestyle="none")
    ax.text(x + 1.3, ly, "served in live agent runs", fontsize=6.3, color=INK2, va="center")
    # editor colours need no legend: every lane names its editor
    return fig, n41, n_edits


def card(arch, lanes, which):
    """1120 x 560 Kaggle card: the 42-layer strip with the edits stacked on it."""
    full = set(arch["full_attention_layers"])
    BG, FG, FG2 = "#0f172a", "#f8fafc", "#94a3b8"
    fig = plt.figure(figsize=(5.6, 2.8), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-3, 45)
    ax.set_ylim(-12.5, 11.5)
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.add_patch(Rectangle((-3, -12.5), 48, 24, fc=BG, ec="none", zorder=-5))
    edits = [item for k, item in lanes if k == "lane"]
    if which == 1:
        sy = -7.2  # strip near the bottom
        ay, fy = schematic(ax, sy, full, None, w=0.74, h=1.25, gap=0.22, dark=True)
        ax.text(-1.2, 8.6, "Frontier models", fontsize=17, fontweight="bold", color=FG, va="center")
        ax.text(-1.2, 5.5, "as weight editors", fontsize=17, fontweight="bold", color=FG, va="center")
        ax.text(-1.2, 2.6, "Gemma 4 E4B  ·  where each edit was written", fontsize=7.6, color=FG2, va="center")
        # stack one dot per edit above its FFN/attention layer (skip the all-layer arms), growing upward
        stack = {}
        for label, detail, col, hollow, live, fp in edits:
            if len({l for l, _ in fp}) == N:
                continue
            for (l, f) in sorted(fp):
                k = stack.get(l, 0)
                stack[l] = k + 1
                yy = ay + 1.45 + k * 1.15
                ax.plot([l], [yy], marker="s" if f == "attn" else "o", ms=6.2, mfc=BG if hollow else col, mec=col,
                        mew=1.4 if hollow else 0, linestyle="none", zorder=4)
        ax.text(41, ay + 1.45 + stack[41] * 1.15 + 0.5, "layer 41", fontsize=6.5, color=FG2, ha="center", va="bottom")
        # gradient all-layer arm as a teal underline
        ax.add_patch(FancyBboxPatch((-0.37, sy - 2.25), N - 0.26, 0.42, boxstyle="round,pad=0,rounding_size=0.2",
                                    fc=GRAD, ec="none"))
        ax.text(-0.4, sy - 3.35, "gradient training: all 42 layers", fontsize=6.3, color=FG2, va="center")
        ax.text(41.4, sy - 3.35, "0 → 41", fontsize=6.3, color=FG2, ha="right", va="center")
    else:
        ax.text(-1.2, 8.6, "Testing agent-written", fontsize=17, fontweight="bold", color=FG, va="center")
        ax.text(-1.2, 5.5, "weight edits", fontsize=17, fontweight="bold", color=FG, va="center")
        ax.text(-1.2, 2.6, "weightprog  ·  OOD-C  ·  320-call replay set", fontsize=7.6, color=FG2, va="center")
        # the three registered checks, with a mark so state is not carried by colour alone
        for i, (t, c, mk) in enumerate((("target", "#34d399", "✓"), ("side effects", "#34d399", "✓"),
                                        ("keep", "#f87171", "✗"))):
            xx = 21.4 + i * 7.3
            ax.add_patch(FancyBboxPatch((xx, 4.6), 6.8, 2.0, boxstyle="round,pad=0,rounding_size=1.0", fc="#1e293b",
                                        ec=c, lw=1.0))
            ax.text(xx + 3.4, 5.6, f"{mk} {t}", fontsize=6.6, color=FG, ha="center", va="center")
        # one edit and two size-matched random controls: same module (layer 41 FFN), random direction
        rows = (("edit", FABLE, False), ("random control", "#94a3b8", True), ("random control", "#94a3b8", True))
        x0, dx, w, h = 6.8, 0.83, 0.62, 1.15
        for r, (lab, col, ctrl) in enumerate(rows):
            yy = -2.6 - r * 2.25
            ax.text(x0 - 1.0, yy, lab, fontsize=6.6, color=FG if not ctrl else FG2, ha="right", va="center")
            for i in range(N):
                lit = i == 41
                fc = (col if not ctrl else "#475569") if lit else ("#1f2b3a" if i not in full else "#2a3747")
                ax.add_patch(Rectangle((x0 + i * dx - w / 2, yy - h / 2), w, h, fc=fc, ec="none"))
            if ctrl:
                ax.plot([x0 + 41 * dx], [yy], marker="o", ms=3.2, mfc="none", mec=col, mew=1.0, linestyle="none")
        ax.text(x0 - w / 2, -9.75, "layer 0", fontsize=6.0, color=FG2, ha="left", va="center")
        ax.text(x0 + 41 * dx + w / 2, -9.75, "controls: same layer, module, rank and size  ·  41", fontsize=6.0,
                color=FG2, ha="right", va="center")
    return fig


def main():
    arch, lanes = load()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, n41, n = edit_map(arch, lanes)
    fig.savefig(OUT / "edit_map.pdf")
    fig.savefig(OUT / "edit_map.png", dpi=200)
    plt.close(fig)
    for w in (1, 2):
        f = card(arch, lanes, w)
        f.savefig(OUT / f"card_paper{w}.png", dpi=200, facecolor=f.get_facecolor())
        plt.close(f)
    print(f"{OUT}: edit_map ({n41} of {n} edits touch layer 41 FFN), card_paper1.png, card_paper2.png")


if __name__ == "__main__":
    main()
