"""One-panel version of the activation figure for the Paper 1 writeup.  Owner: Claude (paper session).

    uv run --offline --no-project --with matplotlib python -I scripts/paper/make_figure_activations_simple.py

Reads only outputs/paper/figures/activations/fig_activations_numbers.json (written by make_figure_activations.py from
the capture of scripts/paper/activation_capture.py) and draws, for four changes of the last layer's feed-forward output
matrix with the same rank and size, how often each changes the most likely next token: at the positions where an
escape code was written, and at ordinary positions. Descriptive, post hoc, one capture. The four-panel figure stays in
the full paper. Writes outputs/paper/figures/activations/fig_activations_simple.{pdf,png}.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs/paper/figures/activations/fig_activations_numbers.json"
OUT = ROOT / "outputs/paper/figures/activations"

INK, INK2, MUTED, GRID, AXIS = "#0f172a", "#475569", "#8492a6", "#e2e8f0", "#c3ccd8"
EDIT, TRAIN, RAND = "#2563c9", "#0f9d8a", "#b4bfcc"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
})

ROWS = [("fable", "Fable's edit", EDIT), ("trained", "gradient-trained", TRAIN), ("rand1", "random 1", RAND),
        ("rand2", "random 2", RAND)]


def main() -> None:
    d = json.loads(SRC.read_text())["changes"]
    W, H = 6.3, 2.25
    fig = plt.figure(figsize=(W, H))
    panels = (("escape", ["where an escape code", "was written"], 100, [0, 50, 100]),
              ("other", ["at ordinary", "positions"], 0.15, [0, 0.05, 0.1, 0.15]))
    x0s, aw = (1.12, 4.05), 2.0
    for (key, title, xmax, ticks), x0 in zip(panels, x0s):
        n = d["fable"][key]["n"]
        ax = fig.add_axes([x0 / W, 0.62 / H, aw / W, 1.08 / H])
        ys = list(range(len(ROWS) - 1, -1, -1))
        cnt = [d[k][key]["first_token_changes"] for k, _, _ in ROWS]
        pct = [100 * c / n for c in cnt]
        ax.barh(ys, pct, height=0.62, color=[c for _, _, c in ROWS], zorder=2)
        for y, c, p in zip(ys, cnt, pct):
            ax.annotate(f"{c:,}", (p, y), (3, 0), textcoords="offset points", size=7.5, color=INK, va="center",
                        weight="bold")
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, len(ROWS) - 0.4)
        ax.set_xticks(ticks, [f"{t:g} %" for t in ticks], size=7)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        if x0 == x0s[0]:
            ax.set_yticks(ys, [lab for _, lab, _ in ROWS], size=7.5, color=INK)
        else:
            ax.set_yticks([])
        fig.text(x0 / W, 1.76 / H, f"{title[0]} {title[1]}: {n:,}", size=7.5, weight="bold", color=INK, va="bottom")
    fig.text(0.05 / W, 2.02 / H, "Share of positions where the change alters the most likely next token (counts on the bars)",
             size=8.2, weight="bold", color=INK, va="bottom")
    for j, t in enumerate(("Four changes of the last layer's feed-forward output matrix with the same rank and size; 120 test calls,",
                           "teacher-forced along the unedited model's own old text. Note the different scales of the two panels.")):
        fig.text(0.05 / W, (0.2 - 0.15 * j) / H, t, size=7, color=INK2, va="bottom")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_activations_simple.{ext}", dpi=200)
    print("wrote", OUT / "fig_activations_simple.png")


if __name__ == "__main__":
    main()
