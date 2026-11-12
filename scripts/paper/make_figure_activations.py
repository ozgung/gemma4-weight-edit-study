"""Activation view of one edit: where it acts, what it reads and what it writes.  Owner: Claude (review session).

    uv run --offline --no-project --with matplotlib --with numpy python -I scripts/paper/make_figure_activations.py [--key CALL_KEY]

Reads outputs/paper/figures/activations/{positions.npz,tokens.jsonl,summary.json} written by
scripts/paper/activation_capture.py (descriptive, post hoc; local E4B, bf16; 120 round-2 gate calls with the base
model's own greedy old text). Writes fig_activations.{pdf,png} and fig_activations_numbers.json (every number a
caption may quote) into the same folder.

(a) one recorded call's old text, token by token, shaded by how much the frozen Fable round-2 edit changes the
    model's choice of that token (KL divergence of the next-token distribution, base against edited);
(b) the edit's two detector values (the rank components A h of the activations entering layer 41's down_proj) at
    escape positions and at the other old-text positions;
(c) the same KL for three changes of the same matrix, rank and size: the authored edit, the gradient-trained adapter
    (G-yer-39) and the two random controls, with the number of positions where the first token changes;
(d) the tokens whose probability the edit moves most at escape positions.

Position classes (old text only). `escape`: the written token holds the backslash of a literal \\n or \\" that the file
does not contain anywhere (the gate's artefact rule, per token). `other`: every other old-text position whose written
token holds no backslash. The few remaining positions (a backslash token the rule does not count, because the file
contains that escape code somewhere) are reported in the numbers file and left out of the panels.
The example call in (a) is picked by a fixed rule (most escape positions among old texts of at most 70 tokens) unless
--key names one. Style follows make_figures_v4.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs/paper/figures/activations"
INK, INK2, MUTED, GRID, AXIS = "#0f172a", "#475569", "#8492a6", "#e2e8f0", "#c3ccd8"
EDIT, TRAIN, RAND, BAD = "#2563c9", "#0f9d8a", "#b4bfcc", "#c2185b"
MONO = "DejaVu Sans Mono"
CW = 1233 / 2048 / 72  # DejaVu Sans Mono advance width, inches per point of font size
HEAT = LinearSegmentedColormap.from_list("heat", ["#ffffff", "#ffe9a8", "#f59e0b", "#c2185b"])
KL_LO, KL_HI = 1e-2, 5.0

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "legend.fontsize": 7,
})

NAMES = {"fable": "Authored edit (Fable, round 2)", "trained": "Gradient-trained, same size", "rand": "Random, same size (2 controls)"}
COL = {"fable": EDIT, "trained": TRAIN, "rand": RAND}


def show(tok: str) -> str:
    """A token as printed in the figure: Gemma's space marker and real line breaks made visible."""
    return tok.replace("▁", " ").replace("\n", "↵").replace("\t", "→")


def layout_tokens(row: dict, width_in: float, fs: float) -> tuple[list, float]:
    """Monospace layout in inches: [(x, line, width, text, kl, is_escape)], line height."""
    cw, lh = fs * CW, fs * 1.9 / 72
    x, line, out = 0.0, 0, []
    for t, lab, kl in zip(row["tokens"], row["label"], row["kl_fable"]):
        s = show(t)
        w = len(s) * cw
        if x + w > width_in:
            x, line = 0.0, line + 1
        out.append((x, line, w, s, kl, lab == "escape"))
        x += w
        if "\n" in t:
            x, line = 0.0, line + 1
    return out, lh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key")
    a = ap.parse_args()
    z = np.load(D / "positions.npz", allow_pickle=False)
    rows = [json.loads(l) for l in (D / "tokens.jsonl").open()]
    summ = json.loads((D / "summary.json").read_text())
    label, call, kind, tix = z["label"], z["call"], z["pos_kind"], z["tok_index"]

    # per old-text position: does the written token hold a backslash? (tokens.jsonl keeps the token texts in order)
    bs = np.zeros(len(label), dtype=bool)
    for r in rows:
        m = np.flatnonzero((call == r["call"]) & (kind == "old"))
        assert len(m) == len(r["tokens"]) and (tix[m] == np.arange(len(m))).all()
        bs[m] = ["\\" in t for t in r["tokens"]]
    esc = label == "escape"
    other = (label == "other") & ~bs
    skipped = (label == "other") & bs
    first = np.array([m[np.argmin(tix[m])] for c in np.unique(call[esc]) for m in [np.flatnonzero((call == c) & esc)]])
    in_art = np.isin(call, np.unique(call[esc]))

    def both(key: str) -> np.ndarray:
        return np.concatenate([z[f"rand1_{key}"], z[f"rand2_{key}"]])

    kl = {"fable": z["fable_kl"], "trained": z["trained_kl"], "rand": both("kl")}
    moved = {"fable": z["fable_top1_changed"], "trained": z["trained_top1_changed"], "rand": both("top1_changed")}
    dup = lambda m, n: np.concatenate([m, m]) if n == "rand" else m  # noqa: E731

    numbers = {"source": "outputs/paper/figures/activations/{positions.npz,tokens.jsonl,summary.json}",
               "calls": int(summ["calls"]), "calls_with_escape_positions": int(len(first)),
               "positions": {"escape": int(esc.sum()), "other": int(other.sum()), "backslash_token_not_counted": int(skipped.sum()),
                             "other_in_texts_without_artefact": int((other & ~in_art).sum())},
               "base_top1_equals_written_token": round(float(summ["base_top1_equals_written_token_old_text"]), 4),
               "detector_fable": {c: {f"value{j + 1}_{q}": round(float(np.quantile(z["fable_gate"][m][:, j], p)), 3)
                                      for j in (0, 1) for q, p in (("p10", 0.1), ("median", 0.5), ("p90", 0.9))}
                                  for c, m in (("escape", esc), ("other", other))},
               "changes": {}}
    for n in ("fable", "trained", "rand1", "rand2"):
        k_, mv = z[f"{n}_kl"], z[f"{n}_top1_changed"]
        numbers["changes"][n] = {
            c: {"n": int(m.sum()), "first_token_changes": int(mv[m].sum()), "kl_median": float(f"{np.median(k_[m]):.3g}"),
                "kl_p99": float(f"{np.quantile(k_[m], 0.99):.3g}"), "share_kl_over_0.1": round(float(np.mean(k_[m] > 0.1)), 4)}
            for c, m in (("escape", esc), ("first_escape_of_each_text", np.isin(np.arange(len(esc)), first)), ("other", other),
                         ("other_in_texts_without_artefact", other & ~in_art), ("backslash_token_not_counted", skipped))}
        numbers["changes"][n]["write_size_relative_to_mlp_output_median"] = {
            "escape": round(float(np.median(z[f"{n}_rel"][esc])), 3), "other": round(float(np.median(z[f"{n}_rel"][other])), 3)}
    by_code = {"\\n": {"n": 0, "v1": [], "v2": [], "moved": 0}, '\\"': {"n": 0, "v1": [], "v2": [], "moved": 0}}
    for r in rows:
        text, off = "".join(r["tokens"]), 0
        for t, lab, g_, t1 in zip(r["tokens"], r["label"], r["gate_fable"], r["top1_fable"]):
            if lab == "escape":
                i = text.find("\\", off)
                c = by_code.get(text[i:i + 2])
                if c is not None:
                    c["n"] += 1
                    c["v1"].append(g_[0])
                    c["v2"].append(g_[1])
                    c["moved"] += int(t1.replace("▁", " ") != t)
            off += len(t)
    numbers["escape_positions_by_code_fable"] = {
        k: {"n": c["n"], "first_token_changes": c["moved"], "detector_value1_median": round(float(np.median(c["v1"])), 3),
            "detector_value2_median": round(float(np.median(c["v2"])), 3)} for k, c in by_code.items()}
    numbers["fable_probability_moved_at_escape_positions"] = {"up": summ["changes"]["fable"]["prob_up"][:6],
                                                              "down": summ["changes"]["fable"]["prob_down"][:6]}

    ex = next(r for r in rows if r["key"] == a.key) if a.key else \
        max((r for r in rows if len(r["tokens"]) <= 70), key=lambda r: (r["n_escape"], -len(r["tokens"])))
    numbers["example_call"] = {"key": ex["key"], "tokens": len(ex["tokens"]), "escape_positions": ex["n_escape"]}

    W, H = 7.4, 6.2
    fig = plt.figure(figsize=(W, H))
    left, right = 0.05, 0.985
    wa = (right - left) * W
    fs = 6.8
    lay, lh = layout_tokens(ex, wa, fs)
    n_lines = max(t[1] for t in lay) + 1
    ha = n_lines * lh + 0.03
    top = 0.925
    axa = fig.add_axes([left, top - ha / H, wa / W, ha / H])
    axa.set_axis_off()
    axa.set_xlim(0, wa)
    axa.set_ylim(-ha, 0)
    norm = LogNorm(KL_LO, KL_HI, clip=True)
    for x, line, w, s, k_, e in lay:
        y0 = -(line + 1) * lh
        axa.add_patch(Rectangle((x, y0 + lh * 0.08), w, lh * 0.86, facecolor=HEAT(norm(max(k_, KL_LO))),
                                edgecolor=BAD if e else "none", linewidth=0.8 if e else 0))
        axa.text(x, y0 + lh * 0.30, s, family=MONO, fontsize=fs, color=INK, va="baseline", ha="left")
    fig.text(left, 0.968, "(a) One old text the unedited model wrote: how much the edit changes the choice of each token",
             fontsize=8.2, color=INK, weight="bold", va="center")
    ykey = top - ha / H - 0.032
    cax = fig.add_axes([left, ykey - 0.006, 0.20, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(KL_LO, KL_HI), cmap=HEAT), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([1e-2, 1e-1, 1, 5])
    cb.set_ticklabels(["≤0.01", "0.1", "1", "≥5"])
    cb.ax.tick_params(labelsize=6.3, length=0, pad=1.5)
    cb.ax.minorticks_off()
    fig.text(left + 0.21, ykey, "change of the next-token distribution (KL, nats)", fontsize=6.8, color=INK2, va="center")
    fig.add_artist(Rectangle((0.60, ykey - 0.009), 0.02, 0.018, transform=fig.transFigure, facecolor="white", edgecolor=BAD, linewidth=0.8))
    fig.text(0.626, ykey, "escape code the file does not contain", fontsize=6.8, color=INK2, va="center")

    ytitle = ykey - 0.075
    bottom, hp = 0.085, ytitle - 0.085 - 0.075

    # (b) detector values
    axb = fig.add_axes([left + 0.035, bottom, 0.25, hp])
    g = z["fable_gate"]
    idx = np.random.default_rng(0).choice(np.flatnonzero(other), size=min(5000, int(other.sum())), replace=False)
    axb.scatter(g[idx, 0], g[idx, 1], s=2.0, color=RAND, alpha=0.5, linewidths=0, rasterized=True)
    axb.scatter(g[esc, 0], g[esc, 1], s=4.5, color=BAD, alpha=0.75, linewidths=0, rasterized=True)
    axb.axhline(0, color=GRID, lw=0.7, zorder=0)
    axb.axvline(0, color=GRID, lw=0.7, zorder=0)
    axb.set_xlabel("detector value 1 (high before \\n)")
    axb.set_ylabel("detector value 2 (high before \\\")")
    fig.text(left, ytitle, "(b) What the edit reads", fontsize=8.2, color=INK, weight="bold", va="center")
    for i, (c, t) in enumerate(((BAD, f"escape positions ({esc.sum()})"), (RAND, f"other positions (sample of {other.sum():,})"))):
        fig.text(left + 0.012, ytitle - 0.030 - i * 0.022, "●", color=c, fontsize=6.5, va="center")
        fig.text(left + 0.028, ytitle - 0.030 - i * 0.022, t, color=INK2, fontsize=6.9, va="center")

    # (c) KL by change and class, with the count of first-token changes
    xc = 0.445
    axc = fig.add_axes([xc, bottom, 0.235, hp])
    y, yt, yl = 0.0, [], []
    for cname, m in (("escape\npositions", esc), ("other\npositions", other)):
        for n in ("fable", "trained", "rand"):
            mm = dup(m, n)
            v = np.maximum(kl[n][mm], 1e-7)
            q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
            axc.plot([q[0], q[4]], [y, y], color=COL[n], lw=1.0, solid_capstyle="butt")
            axc.plot([q[1], q[3]], [y, y], color=COL[n], lw=4.2, solid_capstyle="butt")
            axc.plot([q[2]], [y], marker="o", ms=3.6, color="white", mec=COL[n], mew=1.1, zorder=3)
            axc.text(55, y, f"{int(moved[n][mm].sum()):,} of {int(mm.sum()):,}", fontsize=6.6, color=INK2, va="center", ha="left")
            y += 1
        yt.append(y - 2)
        yl.append(cname)
        y += 0.9
    axc.set_xscale("log")
    axc.set_xlim(1e-7, 30)
    axc.set_ylim(y - 0.5, -0.9)
    axc.set_yticks(yt)
    axc.set_yticklabels(yl, fontsize=7)
    axc.set_xticks([1e-6, 1e-4, 1e-2, 1])
    axc.minorticks_off()
    axc.grid(axis="x", color=GRID, lw=0.6)
    axc.set_axisbelow(True)
    axc.set_xlabel("change of the next-token distribution (KL, nats)")
    axc.text(55, -0.95, "first token\nchanges", fontsize=6.4, color=MUTED, va="bottom", ha="left", linespacing=0.95)
    fig.text(xc - 0.055, ytitle, "(c) Three same-size changes of one matrix", fontsize=8.2, color=INK, weight="bold", va="center")
    for i, n in enumerate(("fable", "trained", "rand")):
        fig.text(xc - 0.043, ytitle - 0.030 - i * 0.022, "■", color=COL[n], fontsize=7.2, va="center")
        fig.text(xc - 0.027, ytitle - 0.030 - i * 0.022, NAMES[n], color=INK2, fontsize=6.9, va="center")

    # (d) probability moved at escape positions
    xd = 0.87
    axd = fig.add_axes([xd, bottom, 0.11, hp])
    ch = summ["changes"]["fable"]
    items = ch["prob_up"][:4] + ch["prob_down"][:3][::-1]
    ys = np.arange(len(items))[::-1]
    for yy, (t, d, _) in zip(ys, items):
        axd.barh(yy, d, color=EDIT if d > 0 else BAD, height=0.6)
        axd.text(max(d, 0) + 0.012, yy, f"{d:+.2f}", fontsize=6.5, color=INK2, va="center", ha="left")
    axd.set_yticks(ys)
    axd.set_yticklabels([show(t) for t, _, _ in items], family=MONO, fontsize=7)
    axd.axvline(0, color=AXIS, lw=0.8)
    lim = max(abs(d) for _, d, _ in items) * 1.5
    axd.set_xlim(-lim, lim)
    axd.set_xticks([-0.2, 0.2])
    axd.set_xlabel("mean change\nof probability")
    axd.spines["left"].set_visible(False)
    fig.text(xd - 0.06, ytitle, "(d) What it writes", fontsize=8.2, color=INK, weight="bold", va="center")

    fig.savefig(D / "fig_activations.pdf")
    fig.savefig(D / "fig_activations.png", dpi=220)
    (D / "fig_activations_numbers.json").write_text(json.dumps(numbers, indent=1, ensure_ascii=False))
    print(json.dumps(numbers, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
