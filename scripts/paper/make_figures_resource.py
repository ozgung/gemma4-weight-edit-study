"""Figures F1-F5 of the resource Writeup (docs/paper2/outline-joint.md), English.  Owner: Claude.

    uv run --offline --no-project --with matplotlib python -I scripts/paper/make_figures_resource.py
    (--out DIR, default outputs/paper/figures/resource/en/)

Writes f1-f5 as .pdf and .png (dpi 200), in the visual language of make_figures_v4.py (palette B; the helpers
are copied because -I keeps the script directory off sys.path).

Sources:
  F1  docs/paper2/notes-fork.md (library facts), weightprog/README.md, weightprog_study/README.md, scripts/oodc/DATASET_README.md
  F2  weightprog/README.md (pipeline); weightprog/examples/spelling/cases.jsonl (the case rows us-barn, uk-barn, other-0,
      copied by hand: the file is in another repository; no result of that example is shown, it has not run on E4B)
  F3  docs/plans/methods-table-20261110.md (support, editor use), docs/paper2/notes-fork.md must-fix 5 and
      weightprog/docs/real-model-validation.md (validation level)
  F5  outputs/weight_edit/claude/wp7r/tasks_oodc_final.jsonl (186 rows: source, license, gradable, final_gold_cp313/312);
      the three earlier funnel counts are from docs/experiments/oodc-build-20261107.md (258 selected, 198 built, 189 kept)
  F4  outputs/weight_edit/libtest/judge/judge_summary.json and report_{lib,nolib}.json (usability test, judged 23:35 UTC
      11-10 on the 54 holdout cases; plan docs/plans/libtest-pydantic-20261110.md)
F1-F3 and F5 show structure and verified counts only; F4 is the one measured result.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OODC = ROOT / "outputs/weight_edit/claude/wp7r/tasks_oodc_final.jsonl"

INK, INK2, MUTED, GRID, AXIS, PANEL = "#0f172a", "#475569", "#8492a6", "#e2e8f0", "#c3ccd8", "#eef2f6"
EDIT, TRAIN, BROKEN = "#2563c9", "#0f9d8a", "#c2185b"
GOOD, BAD, ESC, LIGHT = "#0ca30c", "#d03b3b", "#ffe9a8", "#e1eafb"
GOOD_BG, BAD_BG, PART = "#dff3df", "#f9e1e1", "#8a5a00"
MONO = "DejaVu Sans Mono"
CW = 1233 / 2048 / 72  # DejaVu Sans Mono advance width, inches per point of font size
W = 6.3

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "legend.fontsize": 7,
})


def canvas(w: float, h: float):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, fc=PANEL, ec="none", lw=0.8, r=0.05):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1, mutation_scale=9,
                                                         shrinkA=0, shrinkB=0))


def elbow(ax, pts):
    """A polyline with an arrow head on its last segment."""
    xs, ys = zip(*pts[:-1])
    ax.plot(xs, ys, color=MUTED, lw=1, solid_capstyle="round", solid_joinstyle="round")
    arrow(ax, *pts[-2], *pts[-1])


def chip(ax, x, y, s, fc="white", ec=EDIT, mono=False, size=7, color=INK):
    """An outlined label whose lower-left corner is (x, y); returns its width."""
    w = 0.16 + (CW * size * len(s) if mono else 0.0535 * size / 7 * len(s))
    box(ax, x, y, w, 0.2, fc=fc, ec=ec, lw=0.8, r=0.08)
    ax.text(x + w / 2, y + 0.095, s, size=size, color=color, va="center", ha="center", family=MONO if mono else None)
    return w


def badge(ax, x, y, n, r=0.085):
    ax.add_patch(Circle((x, y), r, fc=EDIT, ec="none"))
    ax.text(x, y - 0.004, str(n), size=7, weight="bold", color="white", va="center", ha="center")


def check_inside(fig, name):
    """Print every text that leaves the figure (a cheap overflow check; collisions are checked by eye)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    w, h = fig.get_size_inches() * fig.dpi
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_text().strip() or not t.get_visible():
            continue
        b = t.get_window_extent(r)
        if b.x0 < 0 or b.y0 < 0 or b.x1 > w or b.y1 > h:
            print(f"{name}: text leaves the figure: {t.get_text()!r}")
        if t.get_fontsize() < 7:
            print(f"{name}: text under 7 pt: {t.get_text()!r}")


# ---------------------------------------------------------------- F1 overview


def card(ax, x, top, w, h, title, tag, sections, mono_title=None):
    box(ax, x, top - h, w, h)
    xi = x + 0.1
    ax.text(xi, top - 0.1, title, weight="bold", size=8.5, va="top", color=INK)
    if mono_title:
        ax.text(xi + 0.0655 * len(title) + 0.1, top - 0.115, mono_title, family=MONO, size=7.5, va="top", color=INK2)
    chip(ax, xi, top - 0.56, tag)
    y = top - 0.7
    for label, rows in sections:
        ax.text(xi, y, label.upper(), size=7, color=MUTED, va="top")
        for j, r in enumerate(rows):
            ax.text(xi, y - 0.15 * (j + 1), r, size=7, color=INK2, va="top")
        y -= 0.15 * (len(rows) + 1) + 0.09


def f1():
    """Overview: two tools (left) and two evaluation resources (right). Arrows mean "uses" or "feeds"."""
    H = 5.0
    fig, ax = canvas(W, H)
    cw, gap = 2.78, 0.64
    xl, xr = 0.05, 0.05 + cw + gap
    h1, h2, rg = 2.0, 1.72, 0.4
    top1 = H - 0.05 - 0.2 - 0.26
    top2 = top1 - h1 - rg
    # entry points
    for x, s in ((xl, "your coding agent, with a behaviour to change"), (xr, "a whole agent to test")):
        wc = chip(ax, x + 0.1, H - 0.25, s, fc=LIGHT)
        arrow(ax, x + 0.1 + wc / 2, H - 0.26, x + 0.1 + wc / 2, top1 + 0.01)
        ax.text(x + 0.1 + wc / 2 + 0.07, H - 0.37, "uses" if x == xl else "runs on", size=7, color=MUTED, va="center")
    card(ax, xl, top1, cw, h1, "weightprog", "library  ·  Apache-2.0  ·  0.2.1", [
        ("takes in", ["target and keep cases:", "prompts with good / bad rules"]),
        ("gives back", ["a closed-form edit  W′ = W + BA", "a comparison with the base model and matched",
                        "random edits:  PASS / FAIL / INCOMPLETE", "a LoRA adapter"])])
    card(ax, xr, top1, cw, h1, "OOD-C", "186 agent tasks  ·  148 repositories", [
        ("takes in", ["a whole agent, run offline in the", "competition harness"]),
        ("gives back", ["traces and patches: what the agent wrote"])])
    card(ax, xl, top2, cw, h2, "Study kit", "in the study repository", [
        ("contains", ["replay, the registered gate, the editors'", "forward-pass sandbox"]),
        ("gives back", ["the companion study's gate, reproduced"])], mono_title="weightprog_study")
    card(ax, xr, top2, cw, h2, "Replay test set", "320 recorded edit calls", [
        ("holds", ["101 tasks, 69 held-out repositories; per call:", "prompt, prefix, outcome and file state"]),
        ("gives back", ["a test of any edit to E4B on the same calls"])])
    # arrows: uses / feeds
    xm = xl + cw + gap / 2
    xv = xl + cw / 2
    arrow(ax, xv, top2 + 0.01, xv, top1 - h1 - 0.01)
    ax.text(xv + 0.08, top2 + rg / 2, "built on", size=7, color=MUTED, va="center")
    ya = top1 - 0.46
    arrow(ax, xl + cw + 0.01, ya, xr - 0.01, ya)
    ax.text(xm, ya + 0.05, "adapter", size=7, color=MUTED, va="bottom", ha="center")
    yf = top2 - h2 + 0.45
    arrow(ax, xr - 0.01, yf, xl + cw + 0.01, yf)
    ax.text(xm, yf + 0.05, "feeds", size=7, color=MUTED, va="bottom", ha="center")
    y0, y1 = top2 - 0.46, top1 - h1 + 0.3
    elbow(ax, [(xr - 0.01, y0), (xm, y0), (xm, y1), (xl + cw + 0.01, y1)])
    ax.text(xm + 0.06, (y0 + y1) / 2, "regression fixture", size=7, color=MUTED, va="center", ha="left", rotation=90)
    ax.text(xl + 0.02, 0.12, "Left: tools.  Right: evaluation data.  Arrows: uses or feeds.", size=7, color=INK2,
            va="center", style="italic")
    return fig


# ---------------------------------------------------------------- F2 workflow

STEPS = [
    ("Cases", ["JSONL: target and keep", "cases, each with a rule;", "tune and holdout splits"]),
    ("Census", ["how the base model", "behaves on the cases"]),
    ("Locate", ["find the decision token,", "read activations there"]),
    ("Construct", ["closed-form W′ = W + BA:", "•  gated edit: a detector", "    adds a logit direction",
                   "•  unit patch", "•  direction ablation", "•  key → value update", "(restrictions: Section 2)"]),
    ("Predict", ["the edit's effect: exact", "for the last layer's", "down_proj, measured", "elsewhere"]),
    ("Evaluate", ["base vs edit vs matched", "random edits, in paired", "draws: target, keep and", "side-effect checks",
                  "(defaults below)"]),
    ("Verdict", []),
    ("Export", ["as a LoRA adapter"]),
]
# weightprog Criteria defaults (src/weightprog/evaluate.py, controls.py), as stated in Section 2 of the writeup
CHECKS = [
    ("Random edits", "two by default; each changes the same modules with the same rank and the same Frobenius norm of BA "
                     "per module, in a Gaussian random direction"),
    ("Target", "the bad rate (generated targets) or the log-probability margin (contrast targets) improves against the "
               "base and against every random edit in one-sided sign tests over cases, p < 0.05, and a 95 % bootstrap "
               "interval over groups for the gain against the base lies above zero; for generated targets, good outputs "
               "also rise by at least half the fall in bad ones"),
    ("Keep", "at most 2 keep cases with a rule turn from good to not good per draw, and at most 10 % of the rule-less "
             "keep cases change their greedy output; the random edits' own losses are shown beside, not used"),
    ("Side effects", "on reference texts kept apart from fitting: mean KL(base ‖ edit) at most 0.05 and mean increase "
                     "in negative log-likelihood (NLL) at most 0.02, in nats per token"),
    ("Verdict", "PASS when every check was measured and passed; FAIL when a check failed; INCOMPLETE when one is missing"),
]
# examples/spelling/cases.jsonl in the weightprog repository: rows us-barn, uk-barn, other-0 (prompt, good, bad)
EXAMPLE = [
    ("target", '"In US English, write one sentence describing the shade of a barn."', r"\bcolor", r"\bcolour"),
    ("keep", '"In British English, write one sentence describing the shade of a barn."', r"\bcolour", r"\bcolor\b"),
    ("keep", '"List three prime numbers."', None, None),
]


def f2():
    """weightprog workflow (README pipeline) with the spelling example's case format; no numbers, no verdict."""
    bw, gap, bh, rg = 1.47, 0.11, 1.5, 0.34
    eh = 1.97
    checks = [(lab, textwrap.wrap(t, 92)) for lab, t in CHECKS]
    ch = 0.36 + sum(0.15 * len(t) + 0.07 for _, t in checks)
    eb = 0.05 + ch + 0.12
    H = eb + 2 * bh + rg + 0.2 + eh + 0.05
    fig, ax = canvas(W, H)
    x0 = (W - 4 * bw - 3 * gap) / 2
    for i, (title, rows) in enumerate(STEPS):
        r, c = divmod(i, 4)
        x, top = x0 + c * (bw + gap), H - 0.05 - r * (bh + rg)
        box(ax, x, top - bh, bw, bh)
        badge(ax, x + 0.18, top - 0.2, i + 1)
        ax.text(x + 0.34, top - 0.2, title, weight="bold", size=8.5, color=INK, va="center")
        for j, s in enumerate(rows):
            ax.text(x + 0.1, top - 0.42 - 0.15 * j, s, size=7, color=INK2, va="top")
        if title == "Verdict":
            for j, s in enumerate(("PASS", "FAIL", "INCOMPLETE")):
                box(ax, x + 0.1, top - 0.64 - 0.27 * j, bw - 0.2, 0.2, fc="white", ec=AXIS, r=0.04)
                ax.text(x + bw / 2, top - 0.545 - 0.27 * j, s, family=MONO, size=7, color=INK, va="center", ha="center")
            ax.text(x + 0.1, top - 1.27, "one of three", size=7, color=INK2, va="top")
        if c:
            arrow(ax, x - gap + 0.01, top - bh / 2, x - 0.01, top - bh / 2)
    # from step 4 down and back to step 5
    ym = H - 0.05 - bh - rg / 2
    elbow(ax, [(x0 + 3 * (bw + gap) + bw / 2, H - 0.05 - bh - 0.01), (x0 + 3 * (bw + gap) + bw / 2, ym), (x0 + bw / 2, ym),
               (x0 + bw / 2, H - 0.05 - bh - rg + 0.01)])
    # the example
    top = eb + eh
    box(ax, x0, eb, W - 2 * x0, eh)
    ax.text(x0 + 0.1, top - 0.1, "Illustrative case format", weight="bold", size=8.5, color=INK, va="top")
    ax.text(x0 + 1.72, top - 0.115, "the library's spelling example (its run on E4B: Section 4)", size=7, color=INK2,
            va="top")
    xb, xt, xc, dy = x0 + 0.2, x0 + 0.42, x0 + 1.02, 0.17
    y = top - 0.5
    badge(ax, xb, y, 1)
    for role, prompt, good, bad in EXAMPLE:
        ax.text(xt, y, role, family=MONO, size=7, weight="bold", color=EDIT if role == "target" else TRAIN, va="center")
        ax.text(xc, y, prompt, family=MONO, size=7, color=INK, va="center")
        y -= dy
        if good:
            ax.text(xc, y, "good", size=7, color=GOOD, weight="bold", va="center")
            ax.text(xc + 0.33, y, good, family=MONO, size=7, color=INK, va="center")
            ax.text(xc + 1.25, y, "bad", size=7, color=BAD, weight="bold", va="center")
            ax.text(xc + 1.5, y, bad, family=MONO, size=7, color=INK, va="center")
        else:
            ax.text(xc, y, "no rule: the output should not change", size=7, color=INK2, va="center")
        y -= dy + 0.06
    y -= 0.03
    ax.plot([x0 + 0.1, W - x0 - 0.1], [y + 0.14, y + 0.14], color=GRID, lw=0.8)
    for n, s in ((3, "the decision token: the first token at which “ color” and “ colour” differ"),):
        badge(ax, xb, y - 0.03, n)
        ax.text(xt, y - 0.03, s, size=7, color=INK2, va="center")
        y -= 0.25
    # default checks (weightprog Criteria defaults; evaluate.py)
    box(ax, x0, 0.05, W - 2 * x0, ch)
    yy = 0.05 + ch - 0.1
    ax.text(x0 + 0.1, yy, "Default checks", weight="bold", size=8.5, color=INK, va="top")
    ax.text(x0 + 1.25, yy - 0.015, "step 6; users can change them, and should fix them before the holdout run", size=7,
            color=INK2, va="top")
    yy -= 0.3
    for lab, lines in checks:
        ax.text(x0 + 0.1, yy, lab, size=7, weight="bold", color=INK, va="top")
        for j, line in enumerate(lines):
            ax.text(x0 + 1.05, yy - 0.15 * j, line, size=7, color=INK2, va="top")
        yy -= 0.15 * len(lines) + 0.07
    return fig


# ---------------------------------------------------------------- F3 method coverage

Y, P, N = "yes", "part", "no"
REAL, TINY, NA = "real E4B", "tiny models only", "—"
# (method family, weightprog support, validation level, editor arms of the earlier study that used it)
MATRIX = [
    ("Find the behaviour", [
        ("Linear probes, read-outs [14]", Y, TINY, "all 9 runs"),
        ("Logit lens, direct logit attribution [15, 16]", Y, TINY, "Fable r2, Astra r2, Opus inf"),
        ("Activation capture at decision tokens [10–12]", Y, REAL, "all 9 runs"),
        ("Difference of means [17]", Y, TINY, "recipe r1, Opus r1"),
        ("Exact prediction of an edit's effect (§2)", Y, REAL, "Fable r2, Opus inf,\na Fable 5.2 run"),
        ("Activation patching [3, 18]", Y, TINY, ""),
        ("Attention and head analysis [19]", P, TINY, "Fable r1"),
        ("Sparse autoencoders [20]", N, NA, ""),
    ]),
    ("Change it", [
        ("Gated logit-direction edit (§2; cf. [21])", Y, TINY, "Fable r2, Astra r2,\nFable 5.2 rep"),
        ("Unit (neuron) patching [21, 22]", Y, REAL, "Opus r2, Opus inf,\nFable 5.2 inf"),
        ("Key → value update with covariance [3, 4]", Y, TINY, ""),
        ("Null-space restriction [5]", Y, TINY, ""),
        ("Directional ablation [23]", Y, TINY, "recipe r1, Fable 5.2 rep"),
        ("Steering written into weights [24]", Y, TINY, "Opus inf"),
        ("Attention-side edits (§3)", P, TINY, "Fable r1, Opus r1"),
        ("Scaling and merging edits [25]", Y, TINY, "Fable 5.2 rep"),
    ]),
    ("Ship it", [
        ("Export as a LoRA adapter [6]", Y, REAL, None),
    ]),
]
SUPPORT = {Y: ("✓", GOOD, GOOD_BG), P: ("~", PART, ESC), N: ("✗", BAD, BAD_BG)}
# docs/plans/methods-table-20261110.md (arms); docs/paper/full/paper-full.md Table 1 and section 3.5
RUN_KEY = [
    [("r1", "round 1: is the old text found at all?"), ("r2", "round 2: escape artefacts"),
     ("Opus", "Claude Opus 5.5"), ("Fable", "Claude Fable 5.1; 5.2 where marked"), ("Astra", "GPT-6 Astra"),
     ("no edit", "Astra r2; Fable 5.2 inf and rep withdrew")],
    [("recipe", "our scripted difference-of-means ablation"), ("inf", "informed: read the earlier runs' records"),
     ("rep", "repair: began from a gradient-trained adapter"), ("round 1", "Opus, Fable, recipe"),
     ("round 2", "Opus, Fable, Astra, Opus inf,"), ("", "Fable 5.2 inf, Fable 5.2 rep")],
]


def f3():
    """Method family x library support x validation level, with the earlier study's editor use as an annotation."""
    rh, rh2, gh, hh = 0.25, 0.39, 0.3, 0.42
    n1 = sum(1 for _, rows in MATRIX for r in rows if "\n" not in (r[3] or ""))
    n2 = sum(1 for _, rows in MATRIX for r in rows if "\n" in (r[3] or ""))
    H = 0.05 + hh + len(MATRIX) * (gh + 0.06) + n1 * rh + n2 * rh2 + 0.62 + 0.3 + 0.24 + 6 * 0.16
    fig, ax = canvas(W, H)
    xm, xs, xv, xe, x1 = 0.15, 2.72, 3.52, 4.72, W - 0.05
    y = H - 0.05
    for x, s in ((xm, "Method family"), (xs, "weightprog\nsupport"), (xv, "Validated on"),
                 (xe, "Used in the companion\nstudy's runs (key below)")):
        ax.text(x, y - hh + 0.1, s, size=7, color=MUTED if x == xe else INK2, va="bottom", linespacing=1.25,
                weight="normal" if x == xe else "bold")
    y -= hh
    for group, rows in MATRIX:
        y -= 0.06
        box(ax, 0.05, y - gh, x1 - 0.05, gh, r=0.04)
        ax.text(xm, y - gh / 2 - 0.005, group, size=8, weight="bold", color=INK, va="center")
        y -= gh
        for k, (name, sup, val, used) in enumerate(rows):
            h = rh2 if "\n" in (used or "") else rh
            yc = y - h / 2
            if k:
                ax.plot([0.05, x1], [y, y], color=GRID, lw=0.6)
            ax.text(xm, yc, name, size=7.5, color=INK, va="center")
            sym, col, bg = SUPPORT[sup]
            box(ax, xs, yc - 0.085, 0.58, 0.17, fc=bg, r=0.04)
            ax.text(xs + 0.12, yc - 0.003, sym, size=8, weight="bold", color=col, va="center", ha="center")
            ax.text(xs + 0.24, yc - 0.003, sup, size=7, color=INK, va="center")
            if val == NA:
                ax.text(xv + 0.04, yc, NA, size=7, color=MUTED, va="center", ha="center")
            else:
                real = val == REAL
                ax.add_patch(Circle((xv + 0.04, yc), 0.04, fc=EDIT if real else "white", ec=EDIT if real else MUTED, lw=1))
                ax.text(xv + 0.15, yc, val, size=7, color=INK if real else INK2, va="center",
                        weight="bold" if real else "normal")
            if used is None:
                ax.text(xe + 0.03, yc, NA, size=7, color=MUTED, va="center", ha="center")
            elif used:
                ax.add_patch(Circle((xe + 0.03, yc + (0.065 if "\n" in used else 0)), 0.025, fc=MUTED, ec="none"))
                ax.text(xe + 0.12, yc, used, size=7, color=MUTED, va="center", linespacing=1.25)
            else:
                ax.add_patch(Circle((xe + 0.03, yc), 0.025, fc="white", ec=MUTED, lw=0.8))
                ax.text(xe + 0.12, yc, "none", size=7, color=MUTED, va="center")
            y -= h
    ax.plot([0.05, x1], [y, y], color=GRID, lw=0.6)
    yl = y - 0.2
    x = xm
    for sup, lab in ((Y, "a function and docs"), (P, "building blocks only"), (N, "not in the library")):
        sym, col, bg = SUPPORT[sup]
        box(ax, x, yl - 0.085, 0.2, 0.17, fc=bg, r=0.04)
        ax.text(x + 0.1, yl - 0.003, sym, size=8, weight="bold", color=col, va="center", ha="center")
        ax.text(x + 0.27, yl, lab, size=7, color=INK2, va="center")
        x += 0.42 + 0.0535 * len(lab)
    yl -= 0.24
    ax.add_patch(Circle((xm + 0.1, yl), 0.04, fc=EDIT, ec=EDIT, lw=1))
    ax.text(xm + 0.27, yl, "checked on the real E4B model", size=7, color=INK2, va="center")
    ax.add_patch(Circle((xm + 2.25, yl), 0.04, fc="white", ec=MUTED, lw=1))
    ax.text(xm + 2.42, yl, "tested only on tiny random Gemma 4 models (Section 4)", size=7, color=INK2, va="center")
    # run key: the companion study's nine editing runs
    yk = yl - 0.3
    ax.plot([0.05, x1], [yk + 0.12, yk + 0.12], color=GRID, lw=0.6)
    ax.text(xm, yk, "Runs of the companion study", size=7.5, weight="bold", color=INK, va="top")
    yk -= 0.24
    for col, entries in enumerate(RUN_KEY):
        xk = xm + col * 3.05
        for j, (k, v) in enumerate(entries):
            ax.text(xk, yk - 0.16 * j, k, size=7, weight="bold", color=INK2, va="top")
            ax.text(xk + 0.62, yk - 0.16 * j, v, size=7, color=INK2, va="top")
    return fig


# ---------------------------------------------------------------- F5 OOD-C composition

FUNNEL = [("candidates selected", 258), ("built", 198), ("kept: the agent can run", 189)]  # oodc-build-20261107.md
EXPECT_LICENCES = {"MIT": 78, "Apache-2.0": 50, "BSD-3-Clause": 39, "AGPL-3.0": 6, "BSD-2-Clause": 5, "Unlicense": 3,
                   "MIT-0": 2, "MPL-2.0": 1, "UPL-1.0": 1, "GPL-3.0": 1}
SOURCE_NAMES = {"nebius/SWE-rebench-leaderboard": "SWE-rebench leaderboard", "nebius/SWE-rebench-V2": "SWE-rebench V2",
                "nebius/SWE-rebench": "SWE-rebench V1"}


def oodc_counts():
    rows = [json.loads(line) for line in OODC.read_text().splitlines() if line.strip()]
    grad = [r for r in rows if r["gradable"]]
    g313 = [r for r in grad if r["final_gold_cp313"]]
    g312 = [r for r in g313 if r["final_gold_cp312"]]
    assert len(g312) == sum(1 for r in grad if r["final_gold_cp312"]), "the Python 3.12 subset is not nested in 3.13"
    lic = Counter(r["license"] for r in rows)
    src = Counter(r["source"] for r in rows)
    out = {"n": len(rows), "repos": len({r["repo"] for r in rows}), "flags": (len(grad), len(g313), len(g312)),
           "licences": lic.most_common(), "sources": src.most_common()}
    print("f5:", out)
    assert out["n"] == 186 and out["repos"] == 148 and out["flags"] == (60, 58, 53), out
    assert dict(lic) == EXPECT_LICENCES, lic
    assert sorted(src.values()) == [5, 72, 109] and set(src) == set(SOURCE_NAMES), src
    return out


def bars(fig, rect, items, xmax, colors=None, label_w=None):
    """Horizontal bars with direct labels; items = [(label, count)], top first. rect is in inches."""
    fw, fh = fig.get_size_inches()
    ax = fig.add_axes([rect[0] / fw, rect[1] / fh, rect[2] / fw, rect[3] / fh])
    n = len(items)
    ys = list(range(n - 1, -1, -1))
    vals = [v for _, v in items]
    ax.barh(ys, vals, height=0.62, color=colors or EDIT, zorder=2)
    for yv, (lab, v) in zip(ys, items):
        ax.annotate(str(v), (v, yv), (3, 0), textcoords="offset points", size=7.5, color=INK, va="center", weight="bold")
    ax.set_yticks(ys, [lab for lab, _ in items], size=7, color=INK)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", pad=4)
    return ax


def f5():
    """OOD-C composition in one row: (a) build funnel and nested grading flags, (b) sources, (c) repository licences."""
    c = oodc_counts()
    bar_h = 0.165
    nl = len(c["licences"])
    H = 0.3 + bar_h * nl + 0.12
    fig, bg = canvas(W, H)
    top = H - 0.07
    bg.text(0.05, top, "(a) Build and grading flags", weight="bold", size=8, color=INK, va="top")
    bg.text(2.62, top, "(b) Sources", weight="bold", size=8, color=INK, va="top")
    bg.text(4.12, top, "(c) Repository licences", weight="bold", size=8, color=INK, va="top")
    y0 = top - 0.26
    funnel = [("candidates", FUNNEL[0][1]), ("built", FUNNEL[1][1]), ("agent can run", FUNNEL[2][1]),
              ("Python 3.12 check", c["n"])]
    flags = [("build rule", c["flags"][0]), ("+ gold, Python 3.13", c["flags"][1]), ("+ gold, Python 3.12", c["flags"][2])]
    ya = y0 - bar_h * 4
    bars(fig, (1.12, ya, 1.25, bar_h * 4), funnel, 300)
    bg.text(0.05, ya - 0.13, "nested grading flags:", size=7, color=INK2, va="center", style="italic")
    yb = ya - 0.24 - bar_h * 3
    bars(fig, (1.12, yb, 1.25, bar_h * 3), flags, 300, colors="#8fb0ea")
    ns = len(c["sources"])
    names = {"nebius/SWE-rebench-leaderboard": "leaderboard", "nebius/SWE-rebench-V2": "V2", "nebius/SWE-rebench": "V1"}
    bars(fig, (3.32, y0 - bar_h * ns, 0.62, bar_h * ns), [(names[k], v) for k, v in c["sources"]], 140)
    for j, t in enumerate(("all from SWE-rebench:", f"{c['n']} tasks,", f"{c['repos']} repositories;", "metadata CC-BY-4.0")):
        bg.text(2.62, y0 - bar_h * ns - 0.14 - 0.14 * j, t, size=7, color=INK2, va="top")
    bars(fig, (5.0, y0 - bar_h * nl, 1.0, bar_h * nl), c["licences"], 100)
    return fig


# ---------------------------------------------------------------- F4 usability test

LIBTEST = ROOT / "outputs/weight_edit/libtest/judge"
NOISE = ROOT / "outputs/weight_edit/libtest/noise_floor/summary.json"  # post hoc, registered 13:14 UTC 11-11
OTHER = "#c9d1dc"


def libtest():
    summ = json.loads((LIBTEST / "judge_summary.json").read_text())
    out = {}
    for arm, path in (("lib", LIBTEST / "report_lib.json"), ("nolib", LIBTEST / "report_nolib.json"),
                      ("fable", LIBTEST.parent / "judge_fable" / "report_libfable.json")):
        rep = json.loads(path.read_text())
        assert rep["verdict"] == "FAIL", arm
        if arm in summ["arms"]:
            a = summ["arms"][arm]
            assert a["verdict"] == "FAIL" and a["target_pass"] and not a["keep_pass"], arm
            assert a["side_effects_pass"] and not a["grad_training_hits"], arm
        k, se, t = rep["keep"], rep["side_effects"]["by_arm"], rep["target"]["by_mode"]["judged"]["tests"]
        out[arm] = dict(
            rates={c: rep["rates"][c]["target"] for c in ("base", "edit", "rand1", "rand2")},
            broken=max(len(v) for v in k["broken_per_draw"].values()), changed=len(k["changed"]),
            n_unchanged=k["unchanged_cases"], ctrl_changed=[k["controls"][c]["changed"] for c in ("rand1", "rand2")],
            kl=se["edit"]["kl_mean"], ctrl_kl=[se[c]["kl_mean"] for c in ("rand1", "rand2")],
            p=max(t[c]["p_one_sided"] for c in ("base", "rand1", "rand2")), n_target=t["base"]["cases"])
    print("f4:", out)
    assert out["lib"]["changed"] == 1 and out["nolib"]["changed"] == 2 and out["fable"]["changed"] == 1
    assert all(out[a]["n_unchanged"] == 4 for a in out)
    return out


def f4():
    """Usability test: (a) target outputs on the holdout for base, edit and the two matched random edits, for the two
    registered arms and the follow-up Fable arm; (b) the three registered checks for all three."""
    d = libtest()
    arms = ("lib", "nolib", "fable")
    H = 6.35
    fig, bg = canvas(W, H)
    rows = [("base model", "base"), ("the arm's edit", "edit"), ("random edit 1", "rand1"), ("random edit 2", "rand2")]
    segs = [("bad", BAD, "white", "bad: v2 name in a pydantic 1 project"), ("other", OTHER, INK, "other: neither name"),
            ("good", GOOD, "white", "good: the v1 API")]
    bg.text(0.05, H - 0.07, "(a) Outputs on the 24 held-out target cases, two draws each", weight="bold", size=8,
            color=INK, va="top")
    lx = 0.05
    for _, col, _, lab in segs:
        box(bg, lx, H - 0.36, 0.13, 0.1, fc=col, r=0.015)
        bg.text(lx + 0.18, H - 0.31, lab, size=7, color=INK2, va="center")
        lx += 0.18 + len(lab) * 0.05 + 0.25
    titles = {"lib": ("Opus 5.5 with weightprog", ""), "nolib": ("Opus 5.5 without weightprog", ""),
              "fable": ("Fable 5.2 with weightprog", "")}
    lw, gapc = 0.92, 0.1
    colw = (W - 0.05 - lw - 2 * gapc) / 3
    top = H - 0.52
    for i, arm in enumerate(arms):
        v = d[arm]
        bx = 0.05 + lw + i * (colw + gapc)
        t, sub = titles[arm]
        bg.text(bx, top, t, weight="bold", size=7.5, color=INK, va="top")
        if sub:
            bg.text(bx + colw, top, sub, size=7, color=MUTED, va="top", ha="right", style="italic")
        bh, gap = 0.2, 0.08
        y = top - 0.4
        for lab, key in rows:
            r = v["rates"][key]
            if i == 0:
                bg.text(0.05, y + bh / 2, lab, size=7, color=INK if key == "edit" else INK2, va="center",
                        weight="bold" if key == "edit" else "normal")
            xx = bx
            for name, col, tc, _ in segs:
                frac = r.get(name, 0.0)
                if frac <= 0:
                    continue
                wseg = colw * frac
                box(bg, xx, y, max(wseg - 0.02, 0.005), bh, fc=col, r=0.02)
                if wseg > 0.28:
                    bg.text(xx + (wseg - 0.02) / 2, y + bh / 2, f"{frac:.2f}", size=7, color=tc, va="center", ha="center")
                xx += wseg
            y -= bh + gap
    # (b) registered checks, one row each, three arms
    yt = H - 0.52 - 0.4 - 3 * 0.28 - 0.2
    bg.text(0.05, yt, "(b) The three registered checks on the holdout; an edit passes only if all three pass", weight="bold",
            size=8, color=INK, va="top")
    cx = [1.12, 1.12 + (W - 1.12) / 3, 1.12 + 2 * (W - 1.12) / 3]
    y = yt - 0.27
    for x0, t in zip(cx, ("Opus, with weightprog", "Opus, without", "Fable 5.2, with (follow-up)")):
        bg.text(x0, y, t, size=7, color=INK, va="top", style="italic")
    y -= 0.22

    def keep(a):
        return [f"with a rule, broken: {a['broken']}", f"unrelated, changed: {a['changed']} of {a['n_unchanged']}",
                f"(random edits: {a['ctrl_changed'][0]} and {a['ctrl_changed'][1]})"]

    def side(a):
        return [f"KL {a['kl']:.5f} nats/token", f"(random {a['ctrl_kl'][0]:.5f}, {a['ctrl_kl'][1]:.5f})"]

    cells = [
        ("Target", "pass", [[f"p ≤ {d[a]['p']:.3f} against the", "base and both random edits"] for a in arms]),
        ("Keep", "fail", [keep(d[a]) for a in arms]),
        ("Side effects", "pass", [side(d[a]) for a in arms]),
    ]
    for name, res, cols in cells:
        col = GOOD if res == "pass" else BROKEN
        mark = "✓" if res == "pass" else "✗"
        bg.plot([0.05, W - 0.05], [y + 0.07, y + 0.07], color=GRID, lw=0.6)
        bg.text(0.05, y, name, size=7, color=INK, va="top", weight="bold")
        bg.text(0.05, y - 0.15, f"{mark} {res}, all three", size=7, color=col, va="top", weight="bold")
        for x0, lines in zip(cx, cols):
            for j, line in enumerate(lines):
                bg.text(x0, y - 0.15 * j, line, size=7, color=INK2, va="top")
        y -= 0.08 + 0.15 * max(len(c) for c in cols)
    foot = ("Holdout: 54 cases in domains not used for tuning: 24 target cases, 24 pydantic 2 keep cases and 6 unrelated",
            "keep cases, 4 of them without a rule. Keep passes if at most 2 keep cases with a rule break per draw and at most",
            "10 % of the 4 unrelated prompts without a rule change their greedy output; with 4 prompts, none may change.",
            "Side effects: mean KL at most 0.05 and mean NLL increase at most 0.02 nats per token, on 16 texts neither arm saw.")
    for j, t in enumerate(foot):
        bg.text(0.05, y + 0.02 - 0.15 * j, t, size=7, color=INK2, va="top", style="italic")
    # (c) post hoc noise floor: 20 new random edits per arm on 48 plain-code prompts (greedy)
    nf = json.loads(NOISE.read_text())
    yc = y + 0.02 - 0.15 * len(foot) - 0.12
    bg.text(0.05, yc, "(c) After judging, post hoc: unrelated outputs that change on 48 plain-code prompts", weight="bold",
            size=8, color=INK, va="top")
    for j, t in enumerate(("each arm's edit against 20 new random edits of its size; the 48 are the 4 holdout, 4 tune and 40 new prompts;",
                           "changed: the greedy output differs from the base model's; fail keep: changes any of the 4 holdout prompts")):
        bg.text(0.05, yc - 0.21 - 0.15 * j, t, size=7, color=INK2, va="top")
    ah = 0.78
    ay = yc - 0.57 - ah
    ax = fig.add_axes([1.12 / W, ay / H, 3.35 / W, ah / H])
    names = {"lib": "Opus, with", "nolib": "Opus, without", "libfable": "Fable 5.2, with"}
    order = ("lib", "nolib", "libfable")
    for i, arm in enumerate(order):
        a = nf["arms"][arm]
        row = len(order) - 1 - i
        seen = Counter()
        for v in sorted(a["controls_changed_all"]):
            k = seen[v]
            seen[v] += 1
            ax.plot(v, row + (k - 0) * 0.14 - 0.14, "s", ms=3.2, color=OTHER, mec=MUTED, mew=0.4, zorder=2)
        ax.plot(a["edit_changed_all"], row, "D", ms=5.5, color=EDIT, zorder=3)
        fail = a["controls_fail_holdout4"]
        bg.text(4.62, ay + (row + 0.5) / len(order) * ah, f"{fail} of 20 random edits fail keep", size=7, color=INK2,
                va="center")
    ax.set_yticks(range(len(order)), [names[a] for a in reversed(order)], size=7, color=INK)
    ax.set_ylim(-0.5, len(order) - 0.5)
    ax.set_xlim(-0.5, 21)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("outputs changed, of 48 (greedy)", size=7, color=INK2, labelpad=2)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    tot = sum(nf["arms"][a]["controls_fail_holdout4"] for a in order)
    bg.text(4.62, ay - 0.05, f"all three: {tot} of 60 fail keep", size=7, color=INK, va="top", weight="bold")
    bg.plot([4.45, 4.55], [ay + ah + 0.02, ay + ah + 0.02], color="white")
    lx = 0.05
    for mk, col, lab in (("D", EDIT, "the arm's edit"), ("s", OTHER, "random edit")):
        bg.plot([lx + 0.05], [ay - 0.42], mk, ms=5 if mk == "D" else 3.4, color=col, mec=MUTED if mk == "s" else col,
                mew=0.4)
        bg.text(lx + 0.15, ay - 0.42, lab, size=7, color=INK2, va="center")
        lx += 1.1
    return fig

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/paper/figures/resource/en")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name, f in (("f1", f1), ("f2", f2), ("f3", f3), ("f4", f4), ("f5", f5)):
        fig = f()
        check_inside(fig, name)
        fig.savefig(a.out / f"{name}.pdf")
        fig.savefig(a.out / f"{name}.png", dpi=200)
        plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
