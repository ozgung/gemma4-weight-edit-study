"""WP10 copy pilot: can base E4B (bf16, local MLX) reproduce a source span as edit_file's old_string?  Owner: Claude.

    outputs/small_model/mlx_venv/bin/python -I scripts/weight_edit/copy_pilot.py build     # snippets + prompts
    outputs/small_model/mlx_venv/bin/python -I scripts/weight_edit/copy_pilot.py run [--limit N] [--samples 3]
    outputs/small_model/mlx_venv/bin/python -I scripts/weight_edit/copy_pilot.py report

Design (Astra's WP10 spec, 2026-11-02 20:20, plus two factors from the census):
- Source: real code the agent read in discovery runs (training repos only), one read_file view per file,
  at most 150 lines / 10,000 characters as the harness allows. Per file, one short (3-5 lines) and one long
  (10-14 lines) contiguous span, unique in the view. The same snippets are used in every condition.
- distance: near = filler then snippet (snippet at the end of the view), far = snippet then the same filler.
  The view's content, total length and every other token are identical between near and far.
- order: new_first = prefill `new_string:<|"|>NEW<|"|>,old_string:<|"|>` (E4B's own argument order in
  785/1,146 recorded edit calls) vs old_first = prefill `old_string:<|"|>`. NEW is the snippet with one
  inserted line and one changed line, so contamination from new_string is measurable.
- render: json = the tool response exactly as served (json.dumps of the read_file dict, inside the
  template's string value, so newlines appear as \\n and quotes as \\"), raw = the same dict text with
  the content unescaped (an off-distribution control that removes the unescaping step).
- Prompt: the real e001nt system turn and tool declarations (decoded from data/sft/rft_e4b_v1, which matches
  the scorer's vLLM rendering token for token), a user turn in the real format with a synthetic problem
  statement naming only the file and function, then a model turn: read_file call, its response, a short
  thought, and the edit_file prefill. Generation stops at the closing <|"|>.
- Decoding: greedy plus --samples draws at the served sampling (T 0.6, top_p 0.95, top_k 64).
- Scores: exact copy, occurs in the view, unique in the view, char similarity (difflib ratio), normalised
  Levenshtein per character, lines borrowed from NEW, literal \\n / \\" artifacts, first divergence.
Weights: data/models/gemma-4-e4b-it-qat-q4_0-unquantized (bf16), not the served W4A16 model.
"""

from __future__ import annotations

import argparse
import difflib
import glob
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "data/models/gemma-4-e4b-it-qat-q4_0-unquantized"
SFT = ROOT / "data/sft/rft_e4b_v1/train.jsonl"
OUT = ROOT / "outputs/weight_edit/claude/copy_pilot"
DISCOVERY_RUNS = [
    "results/e4r/results/rft1_e001nt_baseline_nothink_t06_e4b_cap12",
    "results/e4r/results/rft3_e001nt_baseline_nothink_t06_e4b_cap12",
    "results/e4r/results/rft5_e001nt_baseline_nothink_t06_e4b_cap12",
]
Q = '<|"|>'
FILES_WANTED = {"fastapi": 2, "rich": 1, "requests": 1}


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(MODEL / "tokenizer.json"))


def real_header_and_user(tok) -> dict:
    """System turn + tool declarations, and one real user turn per repo (decoded from the SFT render)."""
    out = {}
    for line in SFT.open():
        s = json.loads(line)
        t = tok.decode(s["input_ids"], skip_special_tokens=False)
        u, m = t.find("<|turn>user"), t.find("<|turn>model")
        repo = re.search(r"repository (\S+?)\.\n", t[u:m]).group(1).split("/")[0]
        out.setdefault("header", t[:u])
        out.setdefault(repo, t[u:m])
    return out


def candidate_views() -> list[dict]:
    """read_file views of source files (not tests) with 100-150 lines, from discovery traces."""
    views = {}
    for run in DISCOVERY_RUNS:
        for p in sorted(glob.glob(str(ROOT / run / "traces/*.json"))):
            for s in json.load(open(p))["steps"]:
                for tc in s.get("tool_calls") or []:
                    if tc.get("function_name") != "read_file":
                        continue
                    try:
                        o = json.loads(s["observation"]["content"])
                    except (KeyError, ValueError, TypeError):
                        continue
                    fp = o.get("filepath", "")
                    c = o.get("content", "")
                    if o.get("status") != "ok" or "test" in fp or not fp.endswith(".py"):
                        continue
                    n = c.count("\n") + 1
                    if 100 <= n <= 150 and len(c) <= 9000 and c.isascii():
                        views.setdefault(fp, o)
    return list(views.values())


START_RE = re.compile(r"^\s*(def |async def |if |elif |else:|for |while |return |with |try:|raise |[A-Za-z_][\w.]*(\[[^\]]*\])? *[+\-*/|]?= )")


def balanced(text: str) -> bool:
    return all(text.count(a) == text.count(b) for a, b in ("()", "[]", "{}"))


def blocks(lines: list[str], lo: int, hi: int) -> list[tuple[int, int]]:
    """Whole-statement spans of lo..hi lines: start at a statement line, end where every bracket is closed,
    no blank lines, comments-only lines or triple-quoted strings inside."""
    res = []
    for i in range(len(lines)):
        if not START_RE.match(lines[i]):
            continue
        for n in range(lo, hi + 1):
            span = lines[i:i + n]
            text = "\n".join(span)
            if (len(span) < n or any(not l.strip() for l in span) or '"""' in text or "'''" in text
                    or span[-1].rstrip().endswith(("(", "[", "{", ",", "\\", ":")) or not balanced(text)):
                continue
            res.append((i, n))
    return res


def make_new(snippet: str) -> str:
    """NEW = snippet with one inserted line after the first line and the last line marked."""
    ls = snippet.split("\n")
    nxt = ls[1] if len(ls) > 1 else ls[0]
    indent = nxt[: len(nxt) - len(nxt.lstrip())]
    ins = indent + "_edit_marker = True"
    ls = [ls[0], ins] + ls[1:]
    ls[-1] = ls[-1] + "  # updated"
    return "\n".join(ls)


def build(args) -> None:
    tok = tokenizer()
    real = real_header_and_user(tok)
    rng = random.Random(20261102)
    by_repo: dict[str, list] = {}
    for v in candidate_views():
        repo = v["filepath"].split("/")[0].replace("src", "requests") if v["filepath"].startswith("src/") else v["filepath"].split("/")[0]
        by_repo.setdefault(repo, []).append(v)
    snippets = []
    for repo, k in FILES_WANTED.items():
        views = sorted(by_repo.get(repo, []), key=lambda v: v["filepath"])
        rng.shuffle(views)
        took = 0
        for v in views:
            if took >= k:
                break
            lines = v["content"].split("\n")
            picked = {}
            for kind, lo, hi in (("short", 3, 5), ("long", 10, 14)):
                cands = [(i, n) for i, n in blocks(lines, lo, hi)
                         if v["content"].count("\n".join(lines[i:i + n])) == 1]
                cands = [c for c in cands if not any(abs(c[0] - p[0]) < 16 for p in picked.values())]
                quoted = [c for c in cands if any(q in "\n".join(lines[c[0]:c[0] + c[1]]) for q in "\"'")]
                pool = quoted or cands
                if not pool:
                    break
                picked[kind] = rng.choice(pool)
            if len(picked) < 2:
                continue
            took += 1
            for kind, (i, n) in picked.items():
                snip = "\n".join(lines[i:i + n])
                filler = "\n".join(lines[:i] + lines[i + n:]).strip("\n")
                snippets.append({
                    "id": f"{repo}:{v['filepath']}:{kind}", "repo": repo, "filepath": v["filepath"], "kind": kind,
                    "snippet": snip, "new": make_new(snip), "filler": filler,
                    "snippet_lines": n, "snippet_tokens": len(tok.encode(snip, add_special_tokens=False).ids),
                    "snippet_quotes": snip.count('"') + snip.count("'"),
                })
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = []
    for sn in snippets:
        user = real.get(sn["repo"]) or real["fastapi"]
        first, last = sn["snippet"].split("\n")[0].strip(), sn["snippet"].split("\n")[-1].strip()
        stmt = (f"In `{sn['filepath']}`, one block of code must be changed: insert the line "
                f"`_edit_marker = True` after its first line and append `  # updated` to its last line. "
                f"The block is {sn['snippet_lines']} lines long; its first line starts with "
                f"`{first[:50]}` and its last line starts with `{last[:50]}`.")
        user = re.sub(r"(Problem Statement:\n).*?(\n\n## )", lambda m: m.group(1) + stmt + m.group(2), user, count=1, flags=re.S)
        for distance in ("near", "far"):
            content = (sn["filler"] + "\n\n" + sn["snippet"]) if distance == "near" else (sn["snippet"] + "\n\n" + sn["filler"])
            n_lines = content.count("\n") + 1
            resp = {"status": "ok", "filepath": sn["filepath"], "content": content, "start_line": 1,
                    "end_line": n_lines, "total_lines": n_lines, "is_truncated": False}
            for render in ("json", "raw"):
                body = json.dumps(resp)
                if render == "raw":
                    body = body.replace(json.dumps(content), '"' + content + '"')
                for order in ("new_first", "old_first"):
                    pre = (f"new_string:{Q}{sn['new']}{Q},old_string:{Q}" if order == "new_first" else f"old_string:{Q}")
                    text = (real["header"] + user + "<|turn>model\n"
                            + f"<|channel>thought\nI will read `{sn['filepath']}` to find the block.\n<channel|>"
                            + f"<|tool_call>call:read_file{{filepath:{Q}{sn['filepath']}{Q}}}<tool_call|>"
                            + f"<|tool_response>response:read_file{{value:{Q}{body}{Q}}}<tool_response|>"
                            + "<|channel>thought\nI found the block. I will apply the edit now.\n<channel|>"
                            + f"<|tool_call>call:edit_file{{filepath:{Q}{sn['filepath']}{Q},{pre}")
                    # distance in tokens from the end of the snippet (as rendered) to the generation point
                    rendered_snip = json.dumps(sn["snippet"])[1:-1] if render == "json" else sn["snippet"]
                    cut = text.rfind(rendered_snip, 0, text.rfind("<tool_response|>")) + len(rendered_snip)
                    dist = len(tok.encode(text[cut:], add_special_tokens=False).ids)
                    prompts.append({"cid": f"{sn['id']}|{distance}|{render}|{order}", "snippet_id": sn["id"],
                                    "distance": distance, "render": render, "order": order,
                                    "dist_tokens": dist, "n_tokens": len(tok.encode(text, add_special_tokens=False).ids),
                                    "text": text})
    (OUT / "snippets.json").write_text(json.dumps(snippets, indent=1))
    with (OUT / "prompts.jsonl").open("w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    print(f"{len(snippets)} snippets, {len(prompts)} prompts -> {OUT.relative_to(ROOT)}")
    for sn in snippets:
        print(f"  {sn['id']:60} lines {sn['snippet_lines']:2} tokens {sn['snippet_tokens']:3} quotes {sn['snippet_quotes']}")
    for p in prompts[:8]:
        print(f"  {p['cid'][-40:]:40} n_tokens {p['n_tokens']} dist {p['dist_tokens']}")


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(gen: str, sn: dict, content: str) -> dict:
    target = sn["snippet"]
    new_only = set(l.strip() for l in sn["new"].split("\n")) - set(l.strip() for l in target.split("\n"))
    first_div = next((i for i, (a, b) in enumerate(zip(gen, target)) if a != b), min(len(gen), len(target)))
    return {
        "exact": int(gen == target), "in_view": int(bool(gen) and gen in content),
        "unique_in_view": int(bool(gen) and content.count(gen) == 1),
        "sim": round(difflib.SequenceMatcher(None, gen, target).ratio(), 4),
        "lev_per_char": round(lev(gen, target) / max(1, len(target)), 4),
        "borrowed_new_lines": sum(1 for l in gen.split("\n") if l.strip() in new_only),
        "lit_bs_n": int("\\n" in gen and "\\n" not in target), "lit_bs_quote": int('\\"' in gen and '\\"' not in target),
        "first_div_char": first_div, "gen_chars": len(gen), "target_chars": len(target),
    }


def run(args) -> None:
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load_model

    tok = tokenizer()
    q_id = tok.token_to_id(Q)
    snippets = {s["id"]: s for s in json.loads((OUT / "snippets.json").read_text())}
    prompts = [json.loads(l) for l in (OUT / "prompts.jsonl").open()]
    if args.limit:
        prompts = prompts[: args.limit]
    res_path = OUT / "results.jsonl"
    done = set()
    if res_path.exists():
        done = {(r["cid"], r["draw"]) for r in map(json.loads, res_path.open())}
    t0 = time.time()
    model, _ = load_model(MODEL, lazy=False, strict=True)
    print(f"model loaded in {time.time() - t0:.0f}s; q token id {q_id}", flush=True)
    draws = ["greedy"] + [f"s{k}" for k in range(args.samples)]
    with res_path.open("a") as out:
        for pi, p in enumerate(prompts):
            sn = snippets[p["snippet_id"]]
            ids = tok.encode(p["text"], add_special_tokens=False).ids
            view = (sn["filler"] + "\n\n" + sn["snippet"]) if p["distance"] == "near" else (sn["snippet"] + "\n\n" + sn["filler"])
            max_new = 2 * sn["snippet_tokens"] + 48
            for d in draws:
                if (p["cid"], d) in done:
                    continue
                sampler = make_sampler(temp=0.0) if d == "greedy" else make_sampler(temp=0.6, top_p=0.95, top_k=64)
                if d != "greedy":
                    mx.random.seed(1000 * pi + int(d[1:]))
                t1 = time.time()
                gen_ids = []
                for tok_id, _ in generate_step(mx.array(ids), model, max_tokens=max_new, sampler=sampler):
                    tok_id = int(tok_id)
                    if tok_id == q_id:
                        break
                    gen_ids.append(tok_id)
                gen = tok.decode(gen_ids, skip_special_tokens=False)
                row = {"cid": p["cid"], "draw": d, "snippet_id": p["snippet_id"], "kind": sn["kind"],
                       "distance": p["distance"], "render": p["render"], "order": p["order"],
                       "dist_tokens": p["dist_tokens"], "n_tokens": p["n_tokens"], "closed": int(len(gen_ids) < max_new),
                       "secs": round(time.time() - t1, 1), "gen": gen}
                row.update(score(gen, sn, view))
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(f"[{pi + 1}/{len(prompts)} {d}] {p['cid'][-48:]} exact={row['exact']} sim={row['sim']} "
                      f"borrowed={row['borrowed_new_lines']} {row['secs']}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


def report(args) -> None:
    import collections
    rows = [json.loads(l) for l in (OUT / "results.jsonl").open()]
    print(f"{len(rows)} generations")
    for fac in ("distance", "order", "render", "kind"):
        g = collections.defaultdict(list)
        for r in rows:
            g[r[fac]].append(r)
        print(fac + ":")
        for k, v in sorted(g.items()):
            n = len(v)
            print(f"  {k:10} n={n:3} exact {sum(r['exact'] for r in v)/n:.2f}  in_view {sum(r['in_view'] for r in v)/n:.2f}"
                  f"  lev/char {sum(r['lev_per_char'] for r in v)/n:.3f}  borrowed-lines/gen {sum(r['borrowed_new_lines'] for r in v)/n:.2f}"
                  f"  lit\\n {sum(r['lit_bs_n'] for r in v)}  lit\\\" {sum(r['lit_bs_quote'] for r in v)}")
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["kind"], r["distance"], r["render"], r["order"])].append(r["exact"])
    print("cells (kind, distance, render, order): exact rate")
    for k, v in sorted(g.items()):
        print(f"  {k}: {sum(v)}/{len(v)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--samples", type=int, default=3)
    sub.add_parser("report")
    a = ap.parse_args()
    {"build": build, "run": run, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
