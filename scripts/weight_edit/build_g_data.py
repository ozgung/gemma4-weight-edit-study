"""Training data for the edit-vs-training arms (docs/plans/edit-vs-train-20261108.md).  Owner: Claude.

    outputs/small_model/mlx_venv/bin/python -I scripts/weight_edit/build_g_data.py

Calls: the 730 round-2 search calls (outputs/weight_edit/sandbox/fable_r2/search/calls.jsonl). Label: the first of
(recorded old_string, its JSON-unescaped form) that occurs exactly once in the true file state; calls without one are
dropped. Input: tokens of prompt + prefix (the gate's input, ending at old_string:<|"|>), then the label tokens and the
closing <|"|> token; loss_mask 1 on the label tokens and the closing token.
Writes data/edit_vs_train/all411/train.jsonl (no val) and data/edit_vs_train/fable39/{train,val}.jsonl, where fable39
is the labelled calls among the unique keys of fable_r2's read passes 1, 2 and 5, and val is 32 labelled calls drawn
(seed 20261108) from tasks outside fable39's tasks. Also MANIFEST.json with counts and keys.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "outputs/weight_edit/sandbox/fable_r2/search"
P = ROOT / "outputs/weight_edit/proposers/fable_r2/requests"
MODEL = ROOT / "data/models/gemma-4-e4b-it-qat-q4_0-unquantized"
OUT = ROOT / "data/edit_vs_train"
Q = '<|"|>'


def unescape(s: str) -> str:
    try:
        return json.loads('"' + s + '"')
    except Exception:  # noqa: BLE001
        return s.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\\\", "\\")


def label(r: dict) -> tuple[str | None, str]:
    content = (S / "files" / f"{r['file_sha']}.txt").read_text()
    for cand, tag in ((r["old_string"], "recorded"), (unescape(r["old_string"]), "unescaped")):
        if cand and content.count(cand) == 1:
            return cand, tag
    return None, "none"


def main() -> None:
    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    qid = tok.token_to_id(Q)
    assert qid is not None
    rows = {}
    for line in (S / "calls.jsonl").open():
        r = json.loads(line)
        rows[r["key"]] = r
    lab = {}
    tags = {}
    for k, r in rows.items():
        t, tag = label(r)
        tags[tag] = tags.get(tag, 0) + 1
        if t is not None:
            lab[k] = t

    def example(k: str) -> dict:
        r = rows[k]
        ids = tok.encode(r["prompt"] + r["prefix"], add_special_tokens=False).ids
        tgt = tok.encode(lab[k], add_special_tokens=False).ids + [qid]
        return {"id": k, "instance_id": r["instance_id"], "repo": r["repo"], "input_ids": ids + tgt,
                "loss_mask": [0] * len(ids) + [1] * len(tgt), "label_kind": "recorded" if lab[k] == r["old_string"] else "unescaped"}

    keys = []
    for n in (1, 2, 5):
        keys += [t["key"] for t in json.load(open(P / f"fwd_{n}.json.claimed"))["texts"]]
    fable = [k for k in dict.fromkeys(keys) if k in lab]
    ftasks = {rows[k]["instance_id"] for k in fable}
    pool = sorted(k for k in lab if rows[k]["instance_id"] not in ftasks)
    val = sorted(random.Random(20261108).sample(pool, 32))
    allk = sorted(lab)

    def write(path: Path, ks: list[str]) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with path.open("w") as f:
            for k in ks:
                e = example(k)
                n += len(e["input_ids"])
                f.write(json.dumps(e) + "\n")
        return n

    man = {"label_tags_730": tags, "all411": {"calls": len(allk), "tokens": write(OUT / "all411/train.jsonl", allk)},
           "fable39": {"calls": len(fable), "tasks": len(ftasks), "unique_read_keys": len(dict.fromkeys(keys)),
                       "tokens": write(OUT / "fable39/train.jsonl", fable), "keys": fable},
           "fable39_val": {"calls": len(val), "tokens": write(OUT / "fable39/val.jsonl", val), "keys": val}}
    (OUT / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a != "keys"}) for k, v in man.items()}))


if __name__ == "__main__":
    main()
