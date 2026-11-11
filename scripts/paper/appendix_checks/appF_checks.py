"""Checks for Appendix F (read-only; frozen gate_r2 outputs). Run: python3 -I appF_checks.py <repo root>.

1. Outcome transitions of the base model's fallback-rescued generations (applied_unescaped) under each arm.
2. Escape artefacts split by whether the recorded new_string is escaped (the 2026-11-06 23:2x split), all arms.
Artefact definition copied from scripts/weight_edit/gate_r2_analyze.py: artefact().
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
G = ROOT / "outputs/weight_edit/claude/gate_r2"
R = ROOT / "outputs/weight_edit/claude/replay_ood"
CALLS = {r["key"]: r for r in map(json.loads, open(R / "calls.jsonl"))}
KEYS = [k.strip() for k in open(R / "usable_keys.txt") if k.strip()]
FILES = {}


def ftext(key):
    sha = CALLS[key]["file_sha"]
    if sha not in FILES:
        FILES[sha] = (R / "files" / f"{sha}.txt").read_text()
    return FILES[sha]


def esc(key, s):
    return int(any(e in s and e not in ftext(key) for e in ("\\n", '\\"')))


def load(arm):
    g = {(r["key"], int(r["draw"])): r for r in map(json.loads, open(G / f"gen_{arm}.jsonl"))}
    s = {(r["key"], int(r["draw"])): r for r in map(json.loads, open(G / f"gen_{arm}.scored.jsonl"))}
    return g, s


ARMS = ["base", "opus_r2", "fable_r2", "opusinf_r2", "fable52inf_v6"]
data = {a: load(a) for a in ARMS}
universe = [(k, d) for k in KEYS for d in (0, 1)]
print("universe", len(universe))
bg, bs = data["base"]

print("\n1. transitions of base applied_unescaped generations")
for a in ARMS[1:]:
    eg, es = data[a]
    c = collections.Counter()
    nf = 0
    for u in universe:
        b, e = bs[u], es[u]
        if b["outcome"] == "applied_unescaped":
            o = e["outcome"]
            c["exact" if o == "applied_exact" else "unesc" if o == "applied_unescaped"
              else "not_found" if o == "not_found" else "other_acc" if e["accepted"] else "other_rej"] += 1
        if b["outcome"] == "not_found" and e["accepted"]:
            nf += 1
    print(a, dict(c), "base not_found -> accepted:", nf)

print("\n2. artefacts by recorded new_string escaped / clean")
grp = {u: ("escaped" if esc(u[0], CALLS[u[0]]["new_string"]) else "clean") for u in universe}
print("generations per group", collections.Counter(grp.values()))
for a in ARMS:
    eg, es = data[a]
    art = collections.Counter()
    acc = collections.Counter()
    for u in universe:
        art[grp[u]] += esc(u[0], eg[u]["gen"])
        acc[grp[u]] += int(es[u]["accepted"])
    print(a, "artefacts", dict(art), "accepted", dict(acc))
print("\n3. working calls lost per draw (recorded ok and base-accepted, edit rejects) and accepted, controls included")
for c in ARMS[1:]:
    for s in ("", "_rand1", "_rand2"):
        es = load(c + s)[1]
        lost = [sum(1 for u in universe if u[1] == d and bs[u]["recorded_kind"] == "ok" and bs[u]["accepted"]
                    and not es[u]["accepted"]) for d in (0, 1)]
        print(c + s, "lost", lost, "accepted", sum(int(es[u]["accepted"]) for u in universe))

for a in ARMS[1:]:
    eg, es = data[a]
    gains = collections.Counter()
    losses = collections.Counter()
    for u in universe:
        db = int(bs[u]["accepted"])
        de = int(es[u]["accepted"])
        if de > db:
            gains[grp[u]] += 1
        if de < db:
            losses[grp[u]] += 1
    print(a, "acceptance gains", dict(gains), "losses", dict(losses))
