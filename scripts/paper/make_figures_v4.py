"""Figures 1-9 of the simple Paper 1 draft (docs/paper/draft-v4-simple.md), in English or Turkish, and the Writeup's
new Figures 2 and 5 (w2: evidence map, w5: behaviour change and preservation).  Owner: Claude.

    uv run --offline --no-project --with matplotlib python -I scripts/paper/make_figures_v4.py --lang en
    (--lang tr for the Turkish PDF; --out DIR, default outputs/paper/figures/v4/<lang>/)

Writes fig1.pdf ... fig9.pdf, w2.pdf, w5.pdf and a .png of each. Chart numbers are read from the frozen reports:
  gate_r2_report.json, gate_r2_opusinf_report.json (Figure 4), results/wp7a/wp7_w4a16/stage1_report.json (Figure 6),
  outputs/weight_edit/claude/wp7/stage2_report.json, wp7r/{report,descriptive}.json and wp7t/{report,descriptive}.json (Figure 7),
  retry_gonogo/gonogo_report.json and retry_gonogo/mask/report.json (Figure 8), edit_vs_train/gate_g_report.json (Figure 9). Figure 5's transitions come from the weight-edit log,
  2026-11-07 01:00 UTC table (base applied_unescaped 84 -> exact / other accepted / still unescaped / not found).
Figures 1, 3 and 5b show synthetic examples written for the paper, not task data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "outputs/weight_edit/claude/gate_r2"
STAGE1 = ROOT / "results/wp7a/wp7_w4a16/stage1_report.json"
STAGE2 = ROOT / "outputs/weight_edit/claude/wp7/stage2_report.json"
WP7R = ROOT / "outputs/weight_edit/claude/wp7r/report.json"
WP7R_DESC = ROOT / "outputs/weight_edit/claude/wp7r/descriptive.json"
WP7T = ROOT / "outputs/weight_edit/claude/wp7t/report.json"
WP7T_DESC = ROOT / "outputs/weight_edit/claude/wp7t/descriptive.json"
GATE_G = ROOT / "outputs/weight_edit/claude/edit_vs_train/gate_g_report.json"
GONOGO = ROOT / "outputs/weight_edit/claude/retry_gonogo/gonogo_report.json"
MASK = ROOT / "outputs/weight_edit/claude/retry_gonogo/mask/report.json"
# weight-edit log 2026-11-07 23:00 UTC: of 71 held-out retry calls, draws (of 8) byte-identical to a failed old text
REPEATS = {0: 32, 5: 2, 7: 2, 8: 35}
# log 2026-11-07 01:00 UTC: of the base model's 84 copies accepted only by the unescape fallback, per edit:
# written raw and accepted (exact + other accepted), written raw and not found, still escaped
TRANSITIONS = {"opus": (18 + 1, 5, 60), "fable": (31 + 2, 12, 39), "opusinf": (39 + 4, 24, 17)}

INK, INK2, MUTED, GRID, AXIS, PANEL = "#0f172a", "#475569", "#8492a6", "#e2e8f0", "#c3ccd8", "#eef2f6"
BASE, EDIT, RAND, BROKEN = "#3b4a5a", "#2563c9", "#b4bfcc", "#c2185b"
TRAIN = "#0f9d8a"
GOOD, BAD, ESC = "#0ca30c", "#d03b3b", "#ffe9a8"
MONO = "DejaVu Sans Mono"
CW = 1233 / 2048 / 72  # DejaVu Sans Mono advance width, inches per point of font size

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "legend.fontsize": 7,
})

T = {
    "en": {
        "f1a": "(a) The file on disk",
        "f1b": "(b) What the model reads: the same text inside the tool result, written with escape codes",
        "f1c": "(c) The old text the model writes when it edits this file",
        "unedited": "Unedited model", "edited": "Edited model",
        "f1bad": "not in the file; only a fallback that removes escapes rescues it",
        "f1good": "found in the file as written",
        "esc": "escape code:  \\n stands for a line break,  \\\" for a quote",
        "b1": ["1  Editor", "Opus 5.5, Fable 5.1 or", "GPT-6 Astra, alone in", "a sandbox", "",
               "✓ read any weight", "✓ 25 forward-pass requests", "   (Astra 10) on calls from", "   4 training repositories",
               "✓ closed-form fits", "   (least squares)", "✗ gradients, backprop", "   or optimisers",
               "✗ the test repositories"],
        "b2": ["2  One edit", "W + B·A: a small", "change to chosen", "weight matrices", "(LoRA form), frozen", "before the test"],
        "b3": ["3  Test fixed in advance", "320 recorded edit calls", "from 101 tasks in 69", "repositories no editor saw,",
               "each generated twice", "", "Compared: base model,", "the edit, and 2 random", "controls of the same size"],
        "b4": ["4  Verdict", "Pass only if both:", "• fewer escape", "  artefacts per task", "• all three side-", "  effect guards hold"],
        "f3h1": "Recorded agent run (kept fixed)", "f3h2": "Regenerate only\nthe old text",
        "f3h3": "Score with the\nharness's matcher",
        "turns": ["Task: issue text and instructions", "read_file  →  file content, escaped", "… earlier turns …"],
        "arms": ["base model", "the edit", "random control"], "same": "same prompt, token for token",
        "outs": ["Was the old text", "accepted or", "not found?", "", "Escape artefact", "in the old text?"],
        "fid": "Check against the served model, on calls from unseen repositories:\nthe replay reproduced "
               "95 of 103 failures and 213 of 213 successes.",
        "f4rows": ["Base model", "Opus 5.5 edit", "Fable 5.1 edit", "Informed Opus 5.5 edit", "6 random controls"],
        "f4x": "Escape artefacts in 640 replayed copies (fewer is better)",
        "f5a": "(a) What each edit did to the 84 escaped copies that the fallback had rescued for the base model",
        "f5rows": ["Opus 5.5", "Fable 5.1", "Informed Opus 5.5"],
        "f5seg": ["off the fallback, accepted", "off the fallback, not found", "unchanged: still escaped, still rescued"],
        "f5b": "(b) How a raw copy breaks (synthetic example)",
        "file": "File", "f5ok": "escaped, rescued by the fallback, applied",
        "f5bad": "raw, but the closing brace is missing: not found",
        "f6a": "(a) Escape artefacts in 640 replayed copies",
        "f6g": ["16-bit weights\n(our test)", "4-bit model in vLLM\n(as served)"],
        "f6arms": ["Base model", "Fable's edit", "Random control"],
        "f6b": "(b) Working calls lost", "guard": "by the edit, per draw (greedy, T 0.6);\nred line: limit (2)",
        "f6bg": ["16-bit", "4-bit vLLM"],
        "f7a": "(a) Edit calls with an escape artefact", "f7b": "(b) Solved tasks",
        "f7arms": ["Base model", "Fable's edit", "Random adapter"], "of": "{a} of {b}", "tie": "ties at these sizes",
        "f7g": ["First run, 115 tasks\n(exploratory)", "OOD-C, 186 tasks\nthinking off\n(planned in advance)",
                "OOD-C, 186 tasks\nthinking on\n(planned in advance)"],
        "f7gb": ["of 115", "of 53\ngradable,\nthinking off", "of 53\ngradable,\nthinking on"],
        "f9a": "(a) Escape artefacts in 640 replayed copies (fewer is better)",
        "f9rows": ["Base model", "Fable 5.1 edit", "Trained: Fable's spot and size, 39 calls", "Trained: all layers, 39 calls",
                   "Trained: all layers, 411 calls"],
        "f9b": "(b) Working calls lost", "f9guard": "of 213 per draw (greedy, T 0.6);\nred line: limit (2)",
        "f9pass": "other guards", "f9yes": "pass", "f9no": "fail",
        "f8a": "(a) How often each retry repeats a failed old text",
        "f8x": "of 8 draws, how many repeat a failed old text", "f8y": "retry calls",
        "f8never": "never repeats:\n{n} calls", "f8stuck": "stuck: {n} calls\n(4 or more of 8)",
        "f8b": "(b) Retries accepted", "f8g": ["stuck calls", "other calls"],
        "f8arms": ["as recorded", "failed text deleted\nfrom earlier turns"], "f8bar": "bar for a third round",
        "w2t": ["(a) Held-out replay", "(b) Served as in\nthe competition", "(c) Live agent"],
        "w2chip": ["3 editors' edits", "Fable's edit only", "Fable's edit only"],
        "w2a": ["320 recorded edit calls,", "101 tasks, 69 held-out", "repositories; against size-", "matched random controls"],
        "w2b": ["LoRA on the 4-bit model", "in vLLM, the format the", "competition serves"],
        "w2metric": "Metric",
        "w2ma": ["escape artefacts in the", "regenerated old text"], "w2mb": ["same replay metric"],
        "w2mc": ["share of edit calls with an", "artefact in old or new text"],
        "w2ha": ["fewer artefacts;", "controls: no comparable cut"], "w2hb": "{r} fewer",
        "w2hb2": ["effect kept", "({r} fewer in 16-bit)"],
        "w2rows": [("First run", ["exploratory, 115 tasks"]),
                   ("OOD-C replication", ["planned in advance, 186 tasks,", "148 new repositories"]),
                   ("OOD-C, thinking on", [])],
        "w2be": "base → edit", "w2rand": "random adapter ≈ base in all three",
        "w2note": "→  tested next.  Each group has its own metric; the numbers are not on a common scale.",
        "w2out": [("Change", "yes, in all three groups"),
                  ("Preservation", "no tested change met the rule\n(3 edits, 3 gradient-trained adapters)"),
                  ("Task outcomes", "no detectable difference")],
        "w2badge": ["Runs in the competition scorer", "(deployment check only)"],
        "w5a": "(a) Behaviour change and preservation, six candidates",
        "w5x": "Escape artefacts remaining, of 640 replay generations (fewer = more change)",
        "w5y": "Working calls lost\n(worse draw, of {n})",
        "w5names": ["Opus 5.5", "Fable 5.1", "Informed Opus 5.5", "Fable's location and size, 39 calls",
                    "all layers, 39 calls", "all layers, 411 calls"],
        "w5leg": ["edit written by an editor", "trained, size-matched to Fable's edit",
                  "trained, all layers (not size-matched)"],
        "w5acc": "acceptance", "w5comp": "compile", "w5ok": "other guards", "w5size": "size",
        "w5base": "base 100", "w5limit": "preservation limit (≤ 2 per draw)",
        "w5note": "✓ ✗: the other guards, acceptance and compile (see the full paper).  Size: Frobenius norm of the weight change.",
    },
    "tr": {
        "f1a": "(a) Diskteki dosya",
        "f1b": "(b) Modelin okuduğu: aynı metin, araç sonucunun içinde kaçış kodlarıyla yazılmış",
        "f1c": "(c) Model bu dosyayı düzenlerken yazdığı eski metin",
        "unedited": "Düzenlenmemiş model", "edited": "Düzenlenmiş model",
        "f1bad": "dosyada yok; onu yalnızca kaçışları kaldıran bir yedek yöntem kurtarıyor",
        "f1good": "dosyada yazıldığı gibi bulunuyor",
        "esc": "kaçış kodu:  \\n satır sonu,  \\\" tırnak işareti yerine geçer",
        "b1": ["1  Editör", "Opus 5.5, Fable 5.1 ya da", "GPT-6 Astra, kum havuzunda", "tek başına", "",
               "✓ her ağırlığı okuyabilir", "✓ 25 ileri geçiş isteği", "   (Astra 10), 4 eğitim", "   deposundaki çağrılarda",
               "✓ kapalı biçimli fit", "   (en küçük kareler)", "✗ gradyan, geri yayılım", "   ya da optimizer",
               "✗ test depoları"],
        "b2": ["2  Tek düzenleme", "W + B·A: seçilen", "ağırlık matrislerine", "küçük bir değişiklik", "(LoRA biçimi), testten", "önce donduruldu"],
        "b3": ["3  Önceden planlı test", "kayıtlı 320 düzenleme", "çağrısı; hiçbir editörün", "görmediği 69 depodaki",
               "101 görevden, her biri iki kez", "", "Karşılaştırılan: temel model,", "düzenleme ve aynı boyutta",
               "2 rastgele kontrol"],
        "b4": ["4  Karar", "Geçmek için ikisi de:", "• görev başına daha", "  az kaçış kalıntısı", "• üç yan etki", "  korumasının üçü"],
        "f3h1": "Kayıtlı ajan çalışması (sabit tutulur)", "f3h2": "Yalnızca eski metni\nyeniden üret",
        "f3h3": "Harness'ın kendi\neşleyicisiyle puanla",
        "turns": ["Görev: issue metni ve talimatlar", "read_file  →  dosya içeriği, kaçışlı", "… önceki adımlar …"],
        "arms": ["temel model", "düzenleme", "rastgele kontrol"], "same": "aynı istem, token token",
        "outs": ["Eski metin kabul", "edildi mi, yoksa", "bulunamadı mı?", "", "Eski metinde kaçış", "kalıntısı var mı?"],
        "fid": "Sunulan modelle kontrol, görülmemiş depolardaki çağrılarda:\nyeniden oynatma 103 başarısızlığın "
               "95'ini ve 213 başarının 213'ünü yeniden üretti.",
        "f4rows": ["Temel model", "Opus 5.5 düzenlemesi", "Fable 5.1 düzenlemesi", "Bilgilendirilmiş Opus 5.5", "6 rastgele kontrol"],
        "f4x": "Yeniden oynatılan 640 kopyadaki kaçış kalıntısı (az olan iyi)",
        "f5a": "(a) Her düzenlemenin, yedek yöntemin temel model için kurtardığı 84 kaçışlı kopyaya etkisi",
        "f5rows": ["Opus 5.5", "Fable 5.1", "Bilgilendirilmiş Opus 5.5"],
        "f5seg": ["fallback'ten çıktı, kabul ediliyor", "fallback'ten çıktı, bulunamadı", "değişmedi: hâlâ kaçışlı, kurtarılıyor"],
        "f5b": "(b) Ham bir kopya nasıl bozulur (sentetik örnek)",
        "file": "Dosya", "f5ok": "kaçışlı; yedek yöntem kurtarıyor, uygulanıyor",
        "f5bad": "ham, ama kapanış parantezi eksik: bulunamadı",
        "f6a": "(a) Yeniden oynatılan 640 kopyadaki kaçış kalıntısı",
        "f6g": ["16 bit ağırlıklar\n(bizim testimiz)", "vLLM'de 4 bit model\n(sunulduğu gibi)"],
        "f6arms": ["Temel model", "Fable'ın düzenlemesi", "Rastgele kontrol"],
        "f6b": "(b) Kaybedilen çalışan çağrılar", "guard": "düzenleme yüzünden, çekiliş başına (greedy, T 0,6);\nkırmızı çizgi: sınır (2)",
        "f6bg": ["16 bit", "4 bit vLLM"],
        "f7a": "(a) Kaçış kalıntısı taşıyan düzenleme çağrıları", "f7b": "(b) Çözülen görevler",
        "f7arms": ["Temel model", "Fable'ın düzenlemesi", "Rastgele adaptör"], "of": "{a}/{b}",
        "tie": "bu boyutlarda beraberlik",
        "f7g": ["İlk çalışma, 115 görev\n(keşif amaçlı)", "OOD-C, 186 görev\ndüşünme kapalı\n(önceden planlanmış)",
                "OOD-C, 186 görev\ndüşünme açık\n(önceden planlanmış)"],
        "f7gb": ["115 görevden", "notlanabilen\n53, düşünme\nkapalı", "notlanabilen\n53, düşünme\naçık"],
        "f9a": "(a) 640 yeniden üretilen kopyada kaçış kalıntısı (az olan iyi)",
        "f9rows": ["Temel model", "Fable 5.1 düzenlemesi", "Eğitilmiş: Fable'ın yerinde, 39 çağrı",
                   "Eğitilmiş: tüm katmanlar, 39 çağrı", "Eğitilmiş: tüm katmanlar, 411 çağrı"],
        "f9b": "(b) Kaybedilen çalışan çağrılar", "f9guard": "çekiliş başına 213 çağrıdan (greedy, T 0,6);\nkırmızı çizgi: sınır (2)",
        "f9pass": "diğer korumalar", "f9yes": "geçti", "f9no": "kaldı",
        "f8a": "(a) Her deneme başarısız metni ne sıklıkla tekrarlıyor",
        "f8x": "8 çekilişten kaçı başarısız metni tekrarlıyor", "f8y": "çağrı",
        "f8never": "hiç tekrarlamıyor:\n{n} çağrı", "f8stuck": "takılı: {n} çağrı\n(8'de 4 ya da fazlası)",
        "f8b": "(b) Kabul edilen denemeler", "f8g": ["takılı çağrılar", "diğer çağrılar"],
        "f8arms": ["kaydedildiği gibi", "başarısız metin önceki\nadımlardan silinince"], "f8bar": "üçüncü tur eşiği",
        "w2t": ["(a) Ayrılmış depolarda\nyeniden oynatma", "(b) Yarışmadaki gibi\nsunum", "(c) Canlı ajan"],
        "w2chip": ["3 editörün düzenlemesi", "yalnızca Fable'ın düzenlemesi", "yalnızca Fable'ın düzenlemesi"],
        "w2a": ["kayıtlı 320 düzenleme çağrısı,", "101 görev, ayrılmış 69 depo;", "aynı boyutta rastgele", "kontrollere karşı"],
        "w2b": ["LoRA, vLLM'deki 4 bit", "model üzerinde; yarışmanın", "sunduğu biçim"],
        "w2metric": "Ölçüt",
        "w2ma": ["yeniden üretilen eski", "metindeki kaçış kalıntısı"], "w2mb": ["aynı yeniden oynatma ölçütü"],
        "w2mc": ["eski ya da yeni metninde kalıntı", "olan düzenleme çağrılarının payı"],
        "w2ha": ["daha az kalıntı;", "kontrollerde benzer düşüş yok"], "w2hb": "{r} daha az",
        "w2hb2": ["etki korundu", "(16 bitte {r} daha az)"],
        "w2rows": [("İlk çalışma", ["keşif amaçlı, 115 görev"]),
                   ("OOD-C tekrarı", ["önceden planlanmış, 186 görev,", "148 yeni depo"]),
                   ("OOD-C, düşünme açık", [])],
        "w2be": "temel → düzenleme", "w2rand": "rastgele adaptör ≈ temel (üçünde de)",
        "w2note": "→  sıradaki test.  Her grubun kendi ölçütü var; sayılar ortak bir ölçekte değil.",
        "w2out": [("Değişim", "evet, üç grupta da"),
                  ("Koruma", "test edilen hiçbir değişiklik kuralı karşılamadı\n(3 düzenleme, 3 gradyanla eğitilmiş adaptör)"),
                  ("Görev sonuçları", "saptanabilir bir fark yok")],
        "w2badge": ["Yarışma puanlayıcısında çalışıyor", "(yalnızca dağıtım kontrolü)"],
        "w5a": "(a) Davranış değişimi ve koruma, altı aday",
        "w5x": "640 yeniden oynatma üretiminde kalan kaçış kalıntısı (az olan = daha çok değişim)",
        "w5y": "Kaybedilen çalışan çağrı\n(kötü çekiliş, {n}'te)",
        "w5names": ["Opus 5.5", "Fable 5.1", "Bilgilendirilmiş Opus 5.5", "Fable'ın yeri ve boyutu, 39 çağrı",
                    "tüm katmanlar, 39 çağrı", "tüm katmanlar, 411 çağrı"],
        "w5leg": ["editörün yazdığı düzenleme", "eğitilmiş, Fable'ın düzenlemesiyle aynı boyutta",
                  "eğitilmiş, tüm katmanlar (boyut eşleşmiyor)"],
        "w5acc": "kabul", "w5comp": "derleme", "w5ok": "diğer korumalar", "w5size": "boyut",
        "w5base": "temel 100", "w5limit": "koruma sınırı (çekiliş başına ≤ 2)",
        "w5note": "✓ ✗: diğer korumalar, kabul ve derleme (ayrıntı tam makalede).  Boyut: ağırlık değişikliğinin Frobenius normu.",
    },
}


def pct(v: float, lang: str) -> str:
    return f"%{v:.0f}" if lang == "tr" else f"{v:.0f} %"


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


def code(ax, x, y, s, size=7.5, esc=True, color=INK):
    """One line of code at baseline y; escape codes get a wash behind them."""
    cw = CW * size
    if esc:
        for i in range(len(s) - 1):
            if s[i] == "\\" and s[i + 1] in 'n"':
                ax.add_patch(Rectangle((x + i * cw, y - 0.26 * size / 72), 2 * cw, 1.16 * size / 72, fc=ESC, ec="none"))
    ax.text(x, y, s, family=MONO, size=size, color=color, va="baseline", ha="left")
    return x + len(s) * cw


def status(ax, x, y, ok: bool, text: str, size=7):
    ax.text(x, y, "✓" if ok else "✗", color=GOOD if ok else BAD, size=size + 1, weight="bold", va="baseline")
    ax.text(x + 0.14, y, text, color=INK2, size=size, va="baseline")


def lines(ax, x, y, rows, dy=0.14, size=6.9, title=True):
    for i, r in enumerate(rows):
        head = title and i == 0
        if r.startswith(("✓", "✗")):
            ax.text(x, y - i * dy, r[0], color=GOOD if r[0] == "✓" else BAD, size=size + 0.5, weight="bold", va="top")
            ax.text(x + 0.13, y - i * dy, r[1:].strip(), color=INK2, size=size, va="top")
        else:
            ax.text(x, y - i * dy, r, color=INK if head else INK2, size=size + (0.8 if head else 0),
                    weight="bold" if head else "normal", va="top")


def fig1(t, lang):
    fig, ax = canvas(6.3, 3.3)
    dy = 0.168
    ax.text(0.05, 3.2, t["f1a"], weight="bold", size=8, va="top", color=INK)
    box(ax, 0.05, 2.58, 6.2, 0.4)
    code(ax, 0.18, 2.84, "def total(items):", esc=False)
    code(ax, 0.18, 2.84 - dy, '    return sum(item["price"] for item in items)', esc=False)
    ax.text(0.05, 2.43, t["f1b"], weight="bold", size=8, va="top", color=INK)
    box(ax, 0.05, 2.0, 6.2, 0.26)
    code(ax, 0.18, 2.09, '{"content": "def total(items):\\n    return sum(item[\\"price\\"] for item in items)\\n"}')
    ax.text(0.05, 1.83, t["f1c"], weight="bold", size=8, va="top", color=INK)
    x0 = 1.55
    box(ax, 0.05, 1.22, 6.2, 0.45)
    ax.text(0.15, 1.53, t["unedited"], size=7.5, va="baseline", color=INK)
    code(ax, x0, 1.53, 'def total(items):\\n    return sum(item[\\"price\\"] for item in items)')
    status(ax, x0, 1.33, False, t["f1bad"])
    box(ax, 0.05, 0.42, 6.2, 0.68)
    ax.text(0.15, 0.96, t["edited"], size=7.5, va="baseline", color=INK)
    code(ax, x0, 0.96, "def total(items):", esc=False)
    code(ax, x0, 0.96 - dy, '    return sum(item["price"] for item in items)', esc=False)
    status(ax, x0, 0.54, True, t["f1good"])
    ax.add_patch(Rectangle((0.08, 0.16), 0.18, 0.11, fc=ESC, ec="none"))
    ax.text(0.33, 0.2, t["esc"], size=7, color=INK2, va="baseline")
    return fig


def fig2(t, lang):
    h = 0.22 + 0.14 * max(len(t[k]) for k in ("b1", "b2", "b3", "b4"))
    fig, ax = canvas(6.3, h + 0.06)
    xs, ws, gap = [0.03], [1.66, 1.3, 1.68, 1.34], 0.09
    for w in ws[:-1]:
        xs.append(xs[-1] + w + gap)
    y0 = 0.03
    for i, (x, w, key) in enumerate(zip(xs, ws, ["b1", "b2", "b3", "b4"])):
        box(ax, x, y0, w, h, fc=PANEL if i != 1 else "#e1eafb")
        lines(ax, x + 0.09, y0 + h - 0.1, t[key])
        if i:
            arrow(ax, x - gap + 0.01, y0 + h / 2, x - 0.01, y0 + h / 2)
    return fig


def fig3(t, lang):
    fig, ax = canvas(6.3, 2.3)
    top, row = 2.25, 0.34
    ax.text(0.05, top, t["f3h1"], weight="bold", size=8, va="top", color=INK)
    for i, s in enumerate(t["turns"]):
        y = top - 0.4 - i * row
        box(ax, 0.05, y - 0.13, 2.85, 0.26)
        ax.text(0.15, y, s, size=7.2, color=INK2, va="center")
    y = top - 0.4 - 3 * row
    box(ax, 0.05, y - 0.15, 2.85, 0.3, fc=PANEL, ec=EDIT, lw=1)
    xe = code(ax, 0.15, y - 0.04, "edit_file(path, old_text=", size=7, esc=False)
    ax.add_patch(Rectangle((xe + 0.02, y - 0.08), 0.42, 0.15, fc=EDIT, ec="none"))
    ax.text(xe + 0.23, y - 0.005, "?", color="white", size=7, weight="bold", ha="center", va="center")
    code(ax, xe + 0.46, y - 0.04, ", new_text)", size=7, esc=False)
    ax.text(3.2, top, t["f3h2"], weight="bold", size=8, va="top", color=INK, linespacing=1.3)
    for i, (s, c) in enumerate(zip(t["arms"], [BASE, EDIT, RAND])):
        yy = top - 0.74 - i * row
        box(ax, 3.2, yy - 0.13, 1.4, 0.26, fc="white", ec=AXIS)
        ax.add_patch(Rectangle((3.28, yy - 0.05), 0.1, 0.1, fc=c, ec="none"))
        ax.text(3.46, yy, s, size=7.2, color=INK, va="center")
    ax.text(3.2, top - 0.74 - 2 * row - 0.24, t["same"], size=6.8, color=MUTED, va="top", style="italic")
    arrow(ax, 2.93, y, 3.17, top - 0.74 - row)
    ax.text(4.86, top, t["f3h3"], weight="bold", size=8, va="top", color=INK, linespacing=1.3)
    box(ax, 4.86, top - 1.66, 1.39, 1.1)
    lines(ax, 4.95, top - 0.66, t["outs"], dy=0.16, size=7.2, title=False)
    arrow(ax, 4.63, top - 0.74 - row, 4.83, top - 0.74 - row)
    ax.text(0.05, 0.06, t["fid"], size=6.8, color=INK2, va="baseline", linespacing=1.35)
    return fig


def gate_numbers():
    g = json.loads((GATE / "gate_r2_report.json").read_text())
    gi = json.loads((GATE / "gate_r2_opusinf_report.json").read_text())
    cands = {"opus": g["candidates"]["opus_r2"], "fable": g["candidates"]["fable_r2"],
             "opusinf": gi["candidates"]["opusinf_r2"]}
    edit = {k: c["primary_escape_artefact"]["vs_base"]["call_micro"][0] for k, c in cands.items()}
    rand = [c["primary_escape_artefact"][f"vs_rand{j}"]["call_micro"][1] for c in cands.values() for j in (1, 2)]
    assert g["base_escape_artefacts"] == gi["base_escape_artefacts"]
    return g["base_escape_artefacts"], edit, rand


def fig4(t, lang):
    base, edit, rand = gate_numbers()
    print("fig4: base", base, "edits", edit, "random", sorted(rand))
    vals = [base, edit["opus"], edit["fable"], edit["opusinf"]]
    fig, ax = plt.subplots(figsize=(6.3, 2.0))
    fig.subplots_adjust(left=0.25, right=0.97, top=0.97, bottom=0.2)
    ys = list(range(len(vals)))[::-1]
    ys = [y + 1 for y in ys]
    ax.barh(ys, vals, height=0.52, color=[BASE] + [EDIT] * 3, zorder=2)
    for y, v in zip(ys, vals):
        lab = f"{v}" if v == base else f"{v}   (−{pct(100 * (base - v) / base, lang)})"
        ax.text(v + 1.5, y, lab, va="center", size=7.5, color=INK)
    jitter = [-0.16, 0.0, 0.16, -0.08, 0.08, 0.0]
    for v, j in zip(sorted(rand), jitter):
        ax.plot(v, 0 + j, "o", ms=4.5, color=RAND, mec="white", mew=0.8, zorder=3)
    ax.text(max(rand) + 2.5, 0, f"{min(rand)}–{max(rand)}", va="center", size=7.5, color=INK)
    ax.set_yticks([0, *ys])
    ax.set_yticklabels([t["f4rows"][4], *t["f4rows"][:4]][::1], size=7.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(t["f4x"], size=7.5)
    return fig


def fig5(t, lang):
    fig = plt.figure(figsize=(6.3, 3.25))
    ax = fig.add_axes([0.235, 0.6, 0.735, 0.27])
    fig.text(0.008, 0.975, t["f5a"], weight="bold", size=8, va="top", color=INK)
    keys = ["opus", "fable", "opusinf"]
    for i, k in enumerate(keys):
        y = 2 - i
        left = 0
        for v, c in zip(TRANSITIONS[k], [EDIT, BROKEN, RAND]):
            ax.barh(y, v - 0.35, left=left, height=0.56, color=c, zorder=2)
            if v >= 4:
                ax.text(left + (v - 0.35) / 2, y, str(v), ha="center", va="center", size=7,
                        color="white" if c != RAND else INK)
            left += v
        assert left == 84, (k, left)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(t["f5rows"], size=7.5)
    ax.set_xlim(0, 84)
    ax.set_xticks([0, 21, 42, 63, 84])
    ax.tick_params(axis="x", labelsize=7)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    handles = [Rectangle((0, 0), 1, 1, fc=c) for c in (EDIT, BROKEN, RAND)]
    fig.legend(handles, t["f5seg"], loc="upper left", bbox_to_anchor=(0.008, 0.935), ncol=3, fontsize=6.8,
               handlelength=1, handleheight=0.9, columnspacing=1.2, borderaxespad=0)
    raw_copy_panel(fig.add_axes([0, 0, 1, 0.5]), t)
    return fig


def raw_copy_panel(ov, t):
    """Figure 5 panel (b): the synthetic raw-copy example, drawn on an axes 6.3 in wide and 1.625 in high."""
    ov.set_xlim(0, 6.3)
    ov.set_ylim(0, 1.625)
    ov.axis("off")
    ov.text(0.05, 1.56, t["f5b"], weight="bold", size=8, va="top", color=INK)
    x0 = 1.55
    rows = [(t["file"], 'return {"total": total(items), "count": len(items)}', None, False, 1.24),
            (t["unedited"], 'return {\\"total\\": total(items), \\"count\\": len(items)}', (True, t["f5ok"]), True, 0.86),
            (t["edited"], 'return {"total": total(items), "count": len(items)', (False, t["f5bad"]), False, 0.34)]
    for i, (lab, s, st, esc, y) in enumerate(rows):
        box(ov, 0.05, y - (0.3 if st else 0.1), 6.2, 0.42 if st else 0.22)
        ov.text(0.15, y - 0.03, lab, size=7.5, color=INK, va="baseline")
        xe = code(ov, x0, y - 0.03, s, esc=esc)
        if i == 2:
            ov.add_patch(Rectangle((xe + 0.005, y - 0.055), CW * 7.5, 0.13, fc="none", ec=BAD, lw=1))
        if st:
            status(ov, x0, y - 0.22, *st)


def fig6(t, lang):
    s = json.loads(STAGE1.read_text())
    arms = ["base", "edit", "rand"]
    art = {p: [s[p][a]["artefacts"] for a in arms] for p in ("bf16", "w4a16")}
    lost = {p: s[p]["edit"]["served_ok_lost_per_draw"] for p in ("bf16", "w4a16")}
    print("fig6: artefacts", art, "broken per draw", lost)
    fig = plt.figure(figsize=(6.3, 2.3))
    ax = fig.add_axes([0.07, 0.24, 0.52, 0.6])
    fig.text(0.008, 0.97, t["f6a"], weight="bold", size=8, va="top", color=INK)
    w = 0.24
    for gi, p in enumerate(("bf16", "w4a16")):
        for ai, (v, c) in enumerate(zip(art[p], [BASE, EDIT, RAND])):
            x = gi + (ai - 1) * (w + 0.03)
            ax.bar(x, v, width=w, color=c, zorder=2, label=t["f6arms"][ai] if gi == 0 else None)
            ax.text(x, v + 2, str(v), ha="center", va="bottom", size=7, color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(t["f6g"], size=7)
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 50, 100])
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.spines["bottom"].set_color(AXIS)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.2), ncol=3, fontsize=6.8, handlelength=1, columnspacing=1)
    bx = fig.add_axes([0.7, 0.24, 0.28, 0.52])
    fig.text(0.665, 0.97, t["f6b"], weight="bold", size=8, va="top", color=INK)
    xs, labels = [], []
    for gi, p in enumerate(("bf16", "w4a16")):
        for d, v in enumerate(lost[p]):
            x = gi * 1.0 + (d - 0.5) * 0.36
            bx.bar(x, v, width=0.3, color=EDIT, zorder=2)
            bx.text(x, v + 0.3, str(v), ha="center", va="bottom", size=7, color=INK)
            xs.append(x)
            labels.append(str(d + 1))
    bx.axhline(2, color=BAD, lw=0.9, zorder=3)
    fig.text(0.665, 0.89, t["guard"], va="top", size=6.3, color=INK2, linespacing=1.3)
    bx.set_xticks(xs)
    bx.set_xticklabels(labels, size=6.5)
    for gi, g in enumerate(t["f6bg"]):
        bx.text(gi, -3.4, g, ha="center", va="top", size=7, color=INK2)
    bx.set_ylim(0, 13)
    bx.set_yticks([0, 5, 10])
    bx.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    bx.spines["bottom"].set_color(AXIS)
    return fig


def live_numbers():
    """Per run (first, OOD-C, OOD-C thinking on) and arm: artefact calls, counted calls, solved, tasks."""
    r = json.loads(STAGE2.read_text())["descriptive"]
    q = json.loads(WP7R.read_text())["descriptive"]
    g = json.loads(WP7R_DESC.read_text())
    arms = ["base", "edit", "rand"]
    first = {a: {k: sum(r[f"{s}/{a}"][k] for s in ("ood", "oodb")) for k in ("artefact_calls", "counted_calls", "resolved", "tasks")}
             for a in arms}
    rep = {a: {k: q[f"oodc/{a}"][k] for k in ("artefact_calls", "counted_calls", "tasks")} for a in arms}
    rep["edit"]["resolved"], rep["base"]["resolved"] = g["gradable_resolved_edit_vs_base"]["resolved"]
    rep["rand"]["resolved"] = g["gradable_resolved_edit_vs_rand"]["resolved"][1]
    qt = json.loads(WP7T.read_text())["descriptive"]
    gt = json.loads(WP7T_DESC.read_text())
    think = {a: {k: qt[f"oodct/{a}"][k] for k in ("artefact_calls", "counted_calls", "tasks")} for a in arms}
    think["edit"]["resolved"], think["base"]["resolved"] = gt["gradable_resolved_edit_vs_base"]["resolved"]
    think["rand"]["resolved"] = gt["gradable_resolved_edit_vs_rand"]["resolved"][1]
    assert all(first[a]["tasks"] == 115 and rep[a]["tasks"] == 186 and think[a]["tasks"] == 186 for a in arms)
    return first, rep, think


def fig7(t, lang):
    first, rep, think = live_numbers()
    arms = ["base", "edit", "rand"]
    print("fig7:", first, rep, think)
    groups = [first, rep, think]
    cols = [BASE, EDIT, RAND]
    fig = plt.figure(figsize=(6.3, 2.7))
    w = 0.24
    ax = fig.add_axes([0.07, 0.25, 0.53, 0.53])
    fig.text(0.008, 0.97, t["f7a"], weight="bold", size=8, va="top", color=INK)
    for gi, tot in enumerate(groups):
        for ai, a in enumerate(arms):
            x = gi + (ai - 1) * (w + 0.03)
            v = 100 * tot[a]["artefact_calls"] / tot[a]["counted_calls"]
            ax.bar(x, v, width=w, color=cols[ai], zorder=2, label=t["f7arms"][ai] if gi == 0 else None)
            ax.text(x, v + 0.6, pct(v, lang), ha="center", va="bottom", size=6.8, color=INK, weight="bold")
            ax.text(x, v + 3.0, t["of"].format(a=tot[a]["artefact_calls"], b=tot[a]["counted_calls"]), ha="center",
                    va="bottom", size=4.6, color=MUTED)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(t["f7g"], size=6.0)
    ax.set_ylim(0, 30)
    ax.set_yticks([0, 10, 20])
    ax.set_yticklabels([pct(v, lang) for v in (0, 10, 20)])
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.spines["bottom"].set_color(AXIS)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.22), ncol=3, fontsize=6.4 if lang == "tr" else 6.8,
              handlelength=1, columnspacing=0.7 if lang == "tr" else 1)
    bx = fig.add_axes([0.68, 0.25, 0.31, 0.53])
    fig.text(0.64, 0.97, t["f7b"], weight="bold", size=8, va="top", color=INK)
    for gi, tot in enumerate(groups):
        for ai, a in enumerate(arms):
            x = gi + (ai - 1) * (w + 0.03)
            v = tot[a]["resolved"]
            bx.bar(x, v, width=w, color=cols[ai], zorder=2)
            bx.text(x, v + 0.5, str(v), ha="center", va="bottom", size=6.8, color=INK, weight="bold")
    bx.text(1.0, 28, t["tie"], ha="center", va="bottom", size=6.8, color=INK2, style="italic")
    bx.set_xticks([0, 1, 2])
    bx.set_xticklabels(t["f7gb"], size=5.6)
    bx.set_ylim(0, 32)
    bx.set_yticks([0, 10, 20, 30])
    bx.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    bx.spines["bottom"].set_color(AXIS)
    return fig


def fig8(t, lang):
    g = json.loads(GONOGO.read_text())["g1_heldout"]
    m = json.loads(MASK.read_text())
    assert sum(REPEATS.values()) == g["calls"] and sum(v for k, v in REPEATS.items() if k >= 4) == g["stuck_calls"]
    acc = [[m["stuck_base"]["acceptance"], m["stuck_masked"]["acceptance"]],
           [m["not_stuck_base"]["acceptance"], m["not_stuck_masked"]["acceptance"]]]
    print("fig8:", REPEATS, acc)
    fig = plt.figure(figsize=(6.3, 2.4))
    ax = fig.add_axes([0.07, 0.22, 0.46, 0.6])
    fig.text(0.008, 0.97, t["f8a"], weight="bold", size=8, va="top", color=INK)
    for k in range(9):
        v = REPEATS.get(k, 0)
        ax.bar(k, v, width=0.7, color=BAD if k >= 4 else BASE, zorder=2)
        if v:
            ax.text(k, v + 0.8, str(v), ha="center", va="bottom", size=7, color=INK, weight="bold")
    ax.text(0.6, 25, t["f8never"].format(n=REPEATS[0]), ha="left", va="center", size=6.5, color=INK2)
    ax.text(7.4, 25, t["f8stuck"].format(n=g["stuck_calls"]), ha="right", va="center", size=6.5, color=BAD)
    ax.set_xticks(range(9))
    ax.set_xlabel(t["f8x"], size=7)
    ax.set_ylabel(t["f8y"], size=7)
    ax.set_ylim(0, 42)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.spines["bottom"].set_color(AXIS)
    bx = fig.add_axes([0.64, 0.22, 0.34, 0.52])
    fig.text(0.6, 0.97, t["f8b"], weight="bold", size=8, va="top", color=INK)
    w = 0.32
    for gi in range(2):
        for ai, c in enumerate([BASE, EDIT]):
            x = gi + (ai - 0.5) * (w + 0.04)
            v = 100 * acc[gi][ai]
            bx.bar(x, v, width=w, color=c, zorder=2, label=t["f8arms"][ai] if gi == 0 else None)
            bx.text(x, v + 1.5, pct(v, lang), ha="center", va="bottom", size=6.8, color=INK, weight="bold")
    bx.plot([-0.4, 0.4], [40, 40], color=BAD, lw=0.9, zorder=3)
    bx.text(0, 42, t["f8bar"], ha="center", va="bottom", size=6, color=BAD)
    bx.set_xticks([0, 1])
    bx.set_xticklabels(t["f8g"], size=7)
    bx.set_ylim(0, 75)
    bx.set_yticks([0, 20, 40, 60])
    bx.set_yticklabels([pct(v, lang) for v in (0, 20, 40, 60)])
    bx.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    bx.spines["bottom"].set_color(AXIS)
    bx.legend(loc="upper left", bbox_to_anchor=(-0.05, 1.32), ncol=2, fontsize=6.2, handlelength=1, columnspacing=0.8)
    return fig


def fig9(t, lang):
    g = json.loads(GATE_G.read_text())
    names = ["G-yer-39", "G-tam-39", "G-tam-411"]
    art = [g["base"]["artefacts"], g["fable_r2"]["artefacts"], *[g["arms"][n]["artefacts"] for n in names]]
    lost = [[g["fable_r2"]["served_ok"][d]["n_lost"] for d in ("draw0", "draw1")],
            *[[g["arms"][n]["served_ok"][d]["n_lost"] for d in ("draw0", "draw1")] for n in names]]
    fab_noninf = g["fable_r2"]["vs_base_accepted"]["task_macro_mean_diff"] >= -0.01
    fab_comp = g["fable_r2"]["fullcall_compile_fail"] <= g["base"]["fullcall_compile_fail"] + 0.02
    other = [fab_noninf and fab_comp, *[g["arms"][n]["acceptance_noninferior"] and g["arms"][n]["fullcall_compile_guard_pass"] for n in names]]
    print("fig9: artefacts", art, "broken per draw", lost, "other guards", other)
    fig = plt.figure(figsize=(6.3, 2.3))
    ax = fig.add_axes([0.33, 0.13, 0.3, 0.64])
    fig.text(0.008, 0.97, t["f9a"], weight="bold", size=8, va="top", color=INK)
    ys = [5, 4, 3, 2, 1]
    ax.barh(ys, art, height=0.56, color=[BASE, EDIT, TRAIN, TRAIN, TRAIN], zorder=2)
    for y, v in zip(ys, art):
        ax.text(v + 2, y, str(v), va="center", size=7.5, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(t["f9rows"], size=7)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 50, 100])
    ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.spines["left"].set_visible(False)
    bx = fig.add_axes([0.7, 0.13, 0.19, 0.64], sharey=ax)
    fig.text(0.68, 0.97, t["f9b"], weight="bold", size=8, va="top", color=INK)
    fig.text(0.68, 0.89, t["f9guard"], va="top", size=6.3, color=INK2, linespacing=1.3)
    for y, pair, c in zip(ys[1:], lost, [EDIT, TRAIN, TRAIN, TRAIN]):
        for d, v in enumerate(pair):
            yy = y + (0.14 if d == 0 else -0.14)
            bx.barh(yy, v, height=0.24, color=c, zorder=2)
            bx.text(v + 0.3, yy, str(v), va="center", size=6.5, color=INK)
    bx.axvline(2, color=BAD, lw=0.9, zorder=3)
    bx.set_xlim(0, 15)
    bx.set_xticks([0, 5, 10])
    bx.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    bx.tick_params(axis="y", labelleft=False)
    bx.spines["left"].set_visible(False)
    fig.text(0.95, 0.78, t["f9pass"].replace(" ", "\n"), ha="center", va="bottom", size=6.3, color=INK2, linespacing=1.1)
    for y, ok in zip(ys[1:], other):
        yf = 0.13 + 0.64 * (y - 0.5) / 5.0
        fig.text(0.95, yf, t["f9yes"] if ok else t["f9no"], ha="center", va="center", size=6.8,
                 color=GOOD if ok else BAD, weight="bold")
    bx.set_ylim(0.5, 5.5)
    return fig


def num(v: float, lang: str, fmt: str = ".2f") -> str:
    s = format(v, fmt)
    return s.replace(".", ",") if lang == "tr" else s


def w2(t, lang):
    """Writeup Figure 2: evidence map (schematic). Arrows mean "tested next"; the groups do not share a scale."""
    base, edit, rand = gate_numbers()
    cuts = sorted(100 * (base - v) / base for v in edit.values())
    s = json.loads(STAGE1.read_text())
    fewer = {p: 100 * (1 - s[p]["edit"]["artefacts"] / s[p]["base"]["artefacts"]) for p in ("bf16", "w4a16")}
    live = [{a: 100 * g[a]["artefact_calls"] / g[a]["counted_calls"] for a in ("base", "edit", "rand")} for g in live_numbers()]
    print("w2: cuts", cuts, "random", sorted(rand), "fewer", fewer, "live", live)
    W, H, yb, gap = 6.3, 4.66, 1.5, 0.13
    fig, ax = canvas(W, H)
    ws = [1.92, 1.82, 2.24]
    xs = [0.03, 0.03 + ws[0] + gap, 0.03 + ws[0] + ws[1] + 2 * gap]
    top = H - 0.03
    for i, (x, w) in enumerate(zip(xs, ws)):
        box(ax, x, yb, w, top - yb)
        if i:
            arrow(ax, x - gap + 0.01, (top + yb) / 2, x - 0.01, (top + yb) / 2)
        xi = x + 0.1
        ax.text(xi, top - 0.1, t["w2t"][i], weight="bold", size=8.5, va="top", color=INK, linespacing=1.25)
        cw = 0.2 + 0.049 * len(t["w2chip"][i])
        box(ax, xi, top - 0.68, cw, 0.2, fc="#e1eafb" if i else "white", ec=EDIT, lw=0.8, r=0.08)
        ax.text(xi + cw / 2, top - 0.585, t["w2chip"][i], size=7, color=INK, va="center", ha="center")
    dy = 0.15

    def para(x, y, rows, color=INK2):
        for j, r in enumerate(rows):
            ax.text(x, y - j * dy, r, size=7, color=color, va="top")
        return y - len(rows) * dy

    def metric(x, y, rows):
        ax.text(x, y, t["w2metric"].upper(), size=7, color=MUTED, va="top")
        return para(x, y - dy, rows)

    def headline(x, big, rows):
        ax.text(x, yb + 0.44, big, size=13, weight="bold", color=INK, va="baseline")
        para(x, yb + 0.36, rows)

    x = xs[0] + 0.1
    y = para(x, top - 0.84, t["w2a"])
    metric(x, y - 0.1, t["w2ma"])
    lo, hi = f"{cuts[0]:.0f}", f"{cuts[-1]:.0f}"
    headline(x, f"%{lo}–{hi}" if lang == "tr" else f"{lo}–{hi} %", t["w2ha"])
    x = xs[1] + 0.1
    y = para(x, top - 0.84, t["w2b"])
    metric(x, y - 0.1, t["w2mb"])
    headline(x, t["w2hb"].format(r=pct(fewer["w4a16"], lang)), [r.format(r=pct(fewer["bf16"], lang)) for r in t["w2hb2"]])
    x, xr = xs[2] + 0.1, xs[2] + ws[2] - 0.1
    y = metric(x, top - 0.84, t["w2mc"]) - 0.12
    ax.text(xr, y, t["w2be"], size=7, color=MUTED, va="top", ha="right")
    y -= dy + 0.05
    for (name, sub), g in zip(t["w2rows"], live):
        ax.plot([x, xr], [y + 0.035, y + 0.035], color=GRID, lw=0.8)
        ax.text(x, y - 0.03, name, size=7.5, color=INK, va="top")
        ax.text(xr, y - 0.015, f"{pct(g['base'], lang)} → {pct(g['edit'], lang)}", size=9, weight="bold", color=INK,
                va="top", ha="right")
        y = para(x, y - 0.03 - dy, sub) - 0.1
    ax.plot([x, xr], [y + 0.035, y + 0.035], color=GRID, lw=0.8)
    ax.text(x, y - 0.03, t["w2rand"], size=7, color=INK2, va="top")
    ax.text(0.05, yb - 0.09, t["w2note"], size=7, color=INK2, va="top", style="italic")
    # outcome strips and the deployment badge
    y, sw = yb - 0.36, 3.95
    for (lead, text), ok in zip(t["w2out"], [True, False, None]):
        n = text.count("\n") + 1
        h = 0.27 + 0.14 * (n - 1)
        box(ax, 0.03, y - h, sw, h)
        yt = y - 0.175
        if ok is None:
            ax.text(0.11, yt, "–", color=MUTED, size=8, weight="bold", va="baseline")
        else:
            ax.text(0.11, yt, "✓" if ok else "✗", color=GOOD if ok else BAD, size=8, weight="bold", va="baseline")
        ax.text(0.28, yt, lead, size=7.5, weight="bold", color=INK, va="baseline")
        for j, row in enumerate(text.split("\n")):
            ax.text(1.32, yt - 0.14 * j, row, size=7, color=INK2, va="baseline")
        y -= h + 0.06
    bx, bw, bh = 0.03 + sw + 0.15, W - 0.06 - sw - 0.15, 0.52
    by = (yb - 0.36 + y + 0.06) / 2 - bh / 2
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=0.05", fc="white", ec=MUTED, lw=0.8,
                                ls=(0, (3, 2))))
    ax.text(bx + bw / 2, by + bh - 0.2, t["w2badge"][0], size=7, weight="bold", color=INK, va="baseline", ha="center")
    ax.text(bx + bw / 2, by + 0.12, t["w2badge"][1], size=7, color=INK2, va="baseline", ha="center")
    return fig


def w5_points():
    """Six candidates: artefacts of 640, served-ok calls lost per draw, acceptance and full-call compile guards, size."""
    g = json.loads((GATE / "gate_r2_report.json").read_text())
    gi = json.loads((GATE / "gate_r2_opusinf_report.json").read_text())
    ga = json.loads((GATE / "gate_r2_report_amended.json").read_text())
    gia = json.loads((GATE / "gate_r2_opusinf_report_amended.json").read_text())
    tr = json.loads(GATE_G.read_text())
    pts = []
    for name, rpt, amd in (("opus_r2", g, ga), ("fable_r2", g, ga), ("opusinf_r2", gi, gia)):
        c = rpt["candidates"][name]
        fro = sum(m["target_fro"] ** 2 for m in json.loads((GATE / f"{name}_rand1/control.json").read_text())) ** 0.5
        pts.append(dict(kind=0, x=c["primary_escape_artefact"]["vs_base"]["call_micro"][0],
                        lost=[c["served_ok_loss"][d]["n_lost"] for d in ("draw0", "draw1")], acc=c["acceptance_noninferior"],
                        comp=amd["candidates"][name]["fullcall_compile_guard_pass"], size=fro))
    for kind, name in ((1, "G-yer-39"), (2, "G-tam-39"), (2, "G-tam-411")):
        c = tr["arms"][name]
        fro = json.loads((GATE_G.parent / f"arms/{name}/edit_norms.json").read_text())["total_fro"]
        pts.append(dict(kind=kind, x=c["artefacts"], lost=[c["served_ok"][d]["n_lost"] for d in ("draw0", "draw1")],
                        acc=c["acceptance_noninferior"], comp=c["fullcall_compile_guard_pass"], size=fro))
    assert g["base_escape_artefacts"] == tr["base"]["artefacts"] == 100
    assert g["candidates"]["fable_r2"]["served_ok_loss"]["draw0"]["limit"] == 2
    # denominator of the y axis: recorded-success calls the base model also gets accepted, per draw
    n_ok = {c["served_ok_loss"][d]["base_accepted_served_ok"] for rpt in (g, gi) for c in rpt["candidates"].values()
            for d in ("draw0", "draw1")}
    assert len(n_ok) == 1
    return g["base_escape_artefacts"], pts, n_ok.pop()


def w5(t, lang):
    """Writeup Figure 5: behaviour change against preservation (no fitted line), with the raw-copy example below."""
    from matplotlib.lines import Line2D

    base, pts, n_ok = w5_points()
    print("w5:", [(p["x"], p["lost"], p["acc"], p["comp"], round(p["size"], 2)) for p in pts])
    H = 5.0
    fig = plt.figure(figsize=(6.3, H))
    fig.text(0.008, 0.99, t["w5a"], weight="bold", size=8, va="top", color=INK)
    ax = fig.add_axes([0.105, 0.475, 0.875, 0.46])
    ax.set_xlim(-3, 113)
    ax.set_ylim(0, 15)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 5, 10, 15])
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.axvline(base, color=BASE, lw=0.9, zorder=1)
    ax.text(base + 1, 14.8, t["w5base"], size=7, color=INK2, va="top")
    ax.axhline(2, color=BAD, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(base - 1.5, 1.7, t["w5limit"], size=7, color=INK2, va="top", ha="right")
    style = [("o", EDIT, 7.5), ("s", TRAIN, 7), ("^", TRAIN, 8)]
    for p, name in zip(pts, t["w5names"]):
        m, c, ms = style[p["kind"]]
        y = max(p["lost"])
        ax.plot(p["x"], y, m, ms=ms, color=c, mec="white", mew=1, zorder=3, clip_on=False)
        bad = [t[k] for k, ok in (("w5acc", p["acc"]), ("w5comp", p["comp"])) if not ok]
        guards, size = ", ".join(bad) if bad else t["w5ok"], f"{t['w5size']} {num(p['size'], lang, '.3g')}"
        # near the base line there is no room for a long second line: stack guards and size in three short lines
        rows = [(14, None), (4, guards), (-6, size)] if p["x"] > 70 else [(2, None), (-8, f"{guards}  ·  {size}")]
        ax.annotate(name, (p["x"], y), (8, rows[0][0]), textcoords="offset points", size=7.5, color=INK,
                    va="bottom" if len(rows) == 2 else "baseline")
        ax.annotate("✗" if bad else "✓", (p["x"], y), (8, rows[1][0]), textcoords="offset points", size=8, weight="bold",
                    color=BAD if bad else GOOD, va="baseline")
        for dy, text in rows[1:]:
            ax.annotate(text, (p["x"], y), (17, dy), textcoords="offset points", size=7, color=INK2, va="baseline")
    ax.set_xlabel(t["w5x"], size=7.5)
    ax.set_ylabel(t["w5y"].format(n=n_ok), size=7.5, linespacing=1.3)
    ax.spines["left"].set_color(AXIS)
    handles = [Line2D([], [], marker=m, ls="none", ms=ms, color=c, mec="white", mew=1) for m, c, ms in style]
    ax.legend(handles, t["w5leg"], loc="upper right", bbox_to_anchor=(0.875, 0.645), fontsize=7, handletextpad=0.3,
              labelspacing=0.6, borderaxespad=0, labelcolor=INK2)
    fig.text(0.008, 0.365, t["w5note"], size=7, color=INK2, va="top")
    raw_copy_panel(fig.add_axes([0, 0, 1, 1.625 / H]), t)
    return fig


def w3(t, lang):
    """Writeup Figure 3: fig2 (editor sandbox flow) above fig3 (fixed-prefix replay), drawn by the same functions."""
    global canvas
    W, tag, gap, real = 6.3, 0.2, 0.16, canvas
    hs = []
    for f in (fig2, fig3):  # heights of the two panels as the stand-alone figures draw them
        probe = f(t, lang)
        hs.append(probe.get_size_inches()[1])
        plt.close(probe)
    H = 2 * tag + sum(hs) + gap
    fig = plt.figure(figsize=(W, H))
    y = H
    try:
        for f, h, label in zip((fig2, fig3), hs, ("(a)", "(b)")):
            fig.text(0.03 / W, (y - 0.04) / H, label, weight="bold", size=8.5, va="top", color=INK)
            y -= tag + h
            ax = fig.add_axes([0, y / H, 1, h / H])  # one data unit = one inch, as in canvas()
            ax.set_xlim(0, W)
            ax.set_ylim(0, h)
            ax.axis("off")
            canvas = lambda w_, h_, ax=ax: (fig, ax)  # noqa: E731  the panel functions draw into this axes
            f(t, lang)
            y -= gap
    finally:
        canvas = real
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=list(T), default="en")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = a.out or ROOT / f"outputs/paper/figures/v4/{a.lang}"
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate([fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9], 1):
        fig = f(T[a.lang], a.lang)
        fig.savefig(out / f"fig{i}.pdf")
        fig.savefig(out / f"fig{i}.png", dpi=200)
        plt.close(fig)
    for name, f in (("w2", w2), ("w3", w3), ("w5", w5)):  # figures of the 3,000-word Writeup that are not in the simple draft
        fig = f(T[a.lang], a.lang)
        fig.savefig(out / f"{name}.pdf")
        fig.savefig(out / f"{name}.png", dpi=200)
        plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
