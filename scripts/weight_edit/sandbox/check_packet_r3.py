"""Check a round-3 proposer packet before launch.  Owner: Claude.

    python3 -I scripts/weight_edit/sandbox/check_packet_r3.py PACKET_DIR [--heldout TASKS.jsonl ...]

Fails (exit 1, one line per finding) unless:
- the packet holds exactly brief.md, dossier.md, server_spec.md, wsp_kernel.py, search/, prior/, scratch/;
- search/ is byte-identical to shared/search_r3, and its calls are exactly search_r2 (train) + the used gate set's usable calls (dev);
- prior/ matches round3/prior_r3/MANIFEST.sha256, with no extra file;
- no file has the name of a gate output (generations, scored rows, reports, key lists);
- no text file names a held-out set or a gate output: OOD-C in any spelling anywhere; outside search/ also the gate and replay paths;
- with --heldout (a tasks.jsonl of a held-out set, once it exists): no repository or task of that set is in the packet.
Reads only; the scratch/ folder must be empty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path("/Users/ozgun/Dev/gemma-4-developer-agent-competition")
SHARED = REPO / "outputs/weight_edit/shared"
C = REPO / "outputs/weight_edit/claude"
TOP = {"brief.md", "dossier.md", "server_spec.md", "wsp_kernel.py", "search", "prior", "scratch"}
HELDOUT_NAME = re.compile(r"(?<![a-z0-9])(oodc|ood-c|ood_c|ood c)(?![a-z0-9])", re.I)
PATHS = re.compile(r"gate_r[0-9]|replay_ood|retry_gonogo|retry_census|wp7r|weight_edit/claude|data/ext|docs/experiments|STATUS\.md", re.I)
GATE_FILE = re.compile(r"^(gen_.*\.jsonl|.*\.scored\.jsonl|usable_keys\.txt|keys_.*\.txt|failed_sets\.json|.*report.*\.json|calls_raw\.jsonl)$")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("packet", type=Path)
    ap.add_argument("--heldout", type=Path, action="append", default=[])
    a = ap.parse_args()
    P = a.packet
    bad: list[str] = []
    got = {p.name for p in P.iterdir()}
    if got != TOP:
        bad.append(f"top level differs: extra {sorted(got - TOP)}, missing {sorted(TOP - got)}")
    if any((P / "scratch").iterdir()):
        bad.append("scratch/ is not empty")
    S, R = P / "search", SHARED / "search_r3"
    if {p.name for p in S.iterdir()} != {"calls.jsonl", "files", "MANIFEST.txt"}:
        bad.append("search/ has other entries than calls.jsonl, files, MANIFEST.txt")
    for f in ("calls.jsonl", "MANIFEST.txt"):
        if sha(S / f) != sha(R / f):
            bad.append(f"search/{f} differs from shared/search_r3")
    fp, fr = {p.name: p for p in (S / "files").iterdir()}, {p.name: p for p in (R / "files").iterdir()}
    if set(fp) != set(fr) or any(sha(fp[n]) != sha(fr[n]) for n in fr):
        bad.append("search/files differs from shared/search_r3/files")
    rows = [json.loads(l) for l in (S / "calls.jsonl").open()]
    train = {json.loads(l)["key"] for l in (SHARED / "search_r2/calls.jsonl").open()}
    dev = {l.strip() for l in (C / "replay_ood/usable_keys.txt").open() if l.strip()}
    if {r["key"] for r in rows if r["set"] == "train"} != train or {r["key"] for r in rows if r["set"] == "dev"} != dev or len(rows) != len(train) + len(dev):
        bad.append("calls are not exactly search_r2 + the used gate set's usable calls")
    prior = C / "round3/prior_r3"
    man = dict(reversed(l.split(None, 1)) for l in (prior / "MANIFEST.sha256").read_text().splitlines())
    man = {k.strip().removeprefix("./"): v for k, v in man.items()}
    files = {str(p.relative_to(P / "prior")): p for p in (P / "prior").rglob("*") if p.is_file()}
    extra = set(files) - set(man) - {"MANIFEST.sha256"}
    if extra or set(man) - set(files) or any(sha(files[k]) != v for k, v in man.items() if k in files):
        bad.append(f"prior/ differs from its manifest (extra {sorted(extra)[:3]})")
    for p in sorted(P.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(P))
        if GATE_FILE.match(p.name):
            bad.append(f"gate-output file name: {rel}")
        if p.suffix == ".npz":
            continue
        text = p.read_text(errors="replace")
        if m := HELDOUT_NAME.search(text):
            bad.append(f"held-out set named in {rel}: {text[max(0, m.start() - 40):m.end() + 40]!r}")
        if not rel.startswith("search/"):
            if m := PATHS.search(text):
                bad.append(f"gate or log path in {rel}: {text[max(0, m.start() - 40):m.end() + 40]!r}")
    repos, tasks = {r["repo"] for r in rows}, {r["instance_id"] for r in rows}
    for f in a.heldout:
        t = [json.loads(l) for l in f.open() if l.strip()]
        hit = repos & {x["repo"] for x in t} or tasks & {x["instance_id"] for x in t}
        print(f"{f}: {len(t)} held-out tasks, overlap {sorted(hit)[:5]}")
        if hit:
            bad.append(f"held-out overlap with the packet: {sorted(hit)[:5]}")
    for b in bad:
        print("BAD", b)
    print(f"{P.name}: {'FAILED' if bad else 'clean'}: {len(rows)} calls, {len(fp)} file states, {len(files)} prior files")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
