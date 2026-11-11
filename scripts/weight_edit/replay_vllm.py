"""Regenerate old_string from replayed prefixes on the served W4A16 model in vLLM, with optional LoRA arms.  Owner: Claude.

WP7 stage 1 (docs/plans/wp7-agent-test-20261107.md): the frozen-prefix replay of replay_gen.py, moved from local bf16
MLX to the competition's serving path, so an exported edit is measured as it would be served.

    python -I scripts/weight_edit/replay_vllm.py --model-dir /content/models/gemma-4-e4b-it-qat-w4a16-ct \\
        --calls outputs/weight_edit/claude/replay_ood/calls.jsonl --keys outputs/weight_edit/claude/replay_ood/usable_keys.txt \\
        --arm base --arm edit=outputs/weight_edit/claude/wp7/adapter_fable_r2 \\
        --arm rand=outputs/weight_edit/claude/wp7/adapter_fable_r2_rand1 --draws 2 --out-dir outputs/weight_edit/claude/wp7/w4a16
    (then score each gen_<arm>.jsonl on the Mac with scripts/weight_edit/replay_score.py)

Same inputs and rules as replay_gen.py: usable calls only; the prompt is `prompt + prefix` tokenized without special
tokens added; generation stops at the closing <|"|> token; max_new = min(cap, 2 * len(old_string tokens) + 64);
draw 0 greedy, draws 1.. at T 0.6, top_p 0.95, top_k 64 with seed 7919 * call_index + draw. Sampled draws are seeded but
are not the same random streams as MLX's, so only greedy draws are comparable across backends; arms within this script
share seeds. One vLLM engine serves every arm (LoRA requests per arm); output gen_<arm>.jsonl in replay_gen's format,
resumable. --dry tokenizes and prints the request plan without vLLM (runs on the Mac).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_gen import Q, usable  # noqa: E402  (stdlib-only at import time)


def requests(rows: list[dict], tok, draws: int, cap: int) -> list[dict]:
    q_id = tok.token_to_id(Q)
    out = []
    for ci, r in enumerate(rows):
        ids = tok.encode(r["prompt"] + r["prefix"], add_special_tokens=False).ids
        ref = len(tok.encode(r["old_string"], add_special_tokens=False).ids)
        for d in range(draws):
            out.append({"key": r["key"], "draw": d, "ids": ids, "max_new": min(cap, 2 * ref + 64),
                        "seed": 7919 * ci + d, "greedy": d == 0, "q_id": q_id})
    return out


def main() -> None:
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True, help="W4A16 checkpoint (compressed-tensors) or any vLLM model")
    ap.add_argument("--calls", type=Path, required=True)
    ap.add_argument("--keys", type=Path, required=True, help="one replay key per line")
    ap.add_argument("--arm", action="append", required=True, help="'base' or NAME=ADAPTER_DIR; repeatable")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--max-new-cap", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-lora-rank", type=int, default=8)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    # vLLM's default 256 OOMed on an L4 warming up the sampler over the 262k vocabulary (2026-11-07); the KV cache holds
    # about 9 of the longest requests at once anyway
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    want = [l.strip() for l in a.keys.read_text().splitlines() if l.strip()]
    idx = {r["key"]: r for r in usable(a.calls)}
    rows = [idx[k] for k in want if k in idx]
    tok = Tokenizer.from_file(str(a.model_dir / "tokenizer.json"))
    reqs = requests(rows, tok, a.draws, a.max_new_cap)
    arms = [(s, None) if s == "base" else tuple(s.split("=", 1)) for s in a.arm]
    longest = max(len(q["ids"]) + q["max_new"] for q in reqs)
    print(f"calls {len(rows)} of {len(want)} keys; requests per arm {len(reqs)}; arms {[n for n, _ in arms]}; "
          f"longest prompt+new {longest} tokens; <|\"|> id {reqs[0]['q_id']}", flush=True)
    if longest > a.max_model_len:
        raise SystemExit(f"a request needs {longest} tokens > --max-model-len {a.max_model_len}")
    if a.dry:
        return

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    t0 = time.time()
    llm = LLM(model=str(a.model_dir), enable_lora=any(p for _, p in arms), max_lora_rank=a.max_lora_rank,
              max_loras=max(1, sum(1 for _, p in arms if p)), max_model_len=a.max_model_len,
              max_num_seqs=a.max_num_seqs, enable_prefix_caching=True, gpu_memory_utilization=a.gpu_mem, seed=0)
    print(f"engine up {time.time() - t0:.0f}s", flush=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for li, (name, path) in enumerate(arms):
        out_path = a.out_dir / f"gen_{name}.jsonl"
        done = set()
        if out_path.exists():
            done = {(j["key"], j["draw"]) for j in map(json.loads, out_path.open())}
        todo = [q for q in reqs if (q["key"], q["draw"]) not in done]
        if not todo:
            continue
        lora = LoRARequest(name, li + 1, str(path)) if path else None
        params = [SamplingParams(temperature=0.0, max_tokens=q["max_new"], stop_token_ids=[q["q_id"]],
                                 skip_special_tokens=False) if q["greedy"] else
                  SamplingParams(temperature=0.6, top_p=0.95, top_k=64, seed=q["seed"], max_tokens=q["max_new"],
                                 stop_token_ids=[q["q_id"]], skip_special_tokens=False) for q in todo]
        t1 = time.time()
        outs = llm.generate([TokensPrompt(prompt_token_ids=q["ids"]) for q in todo], params, lora_request=lora)
        with out_path.open("a") as f:
            for q, o in zip(todo, outs):
                gen = list(o.outputs[0].token_ids)
                closed = int(o.outputs[0].stop_reason == q["q_id"] or (bool(gen) and gen[-1] == q["q_id"]))
                if gen and gen[-1] == q["q_id"]:
                    gen = gen[:-1]
                f.write(json.dumps({"key": q["key"], "draw": q["draw"], "gen": tok.decode(gen, skip_special_tokens=False),
                                    "closed": closed, "n_gen": len(gen), "prompt_tokens": len(q["ids"]),
                                    "prefill_s": None, "gen_s": None, "edit": str(path) if path else None,
                                    "backend": "vllm", "arm": name}) + "\n")
        print(f"arm {name}: {len(todo)} requests in {time.time() - t1:.0f}s", flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
