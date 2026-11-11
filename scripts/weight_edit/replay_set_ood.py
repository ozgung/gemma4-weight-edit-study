"""Round-2 fresh test set: replay_set.py applied to E4B e001nt (no-think, T 0.6) runs on held-out repositories
(OOD-43 x4 runs, OOD-B-73 x1). Owner: Claude (fork of the weight-edit session), 2026-11-03.

    .venv/bin/python -I scripts/weight_edit/replay_set_ood.py [--runs o43a,o43b,o43t,o43v,oodb] [--limit-traces N]

Differences from replay_set.py: tasks come from the OOD task files (OOD-43: data/ext/ood45/tasks.jsonl, statement
'g1' = `problem_statement`, as run_ood45.py uses; OOD-B: tasks.jsonl of the Kaggle dataset ozgung/g4-oodb-pilot,
fetched to data/ext/ood_files/oodfiles/oodb/), and base file contents come from the task snapshots
(data/ext/ood_files/oodfiles/<suite>/<iid>/<path>, extracted on a Colab CPU VM from the Kaggle snapshot tgz files,
only for files the traces read/edit/write). Everything else (prompt rebuild and token verification, prefix, true
file state by re-applying successful edits, state_uncertain) is unchanged. For each edit_file call:
- prompt: rebuilt with the sft_export machinery (ADK flow preprocessing, LiteLLM patch, vLLM conversation
  parsing, the served chat template), checked against the trace's prompt_tokens (`verified`).
- prefix: the model's generated text for that call up to the start of the old_string value, in the recorded
  (generation) argument order: thought channel, visible text, earlier parallel tool calls, then
  `<|tool_call>call:edit_file{` and the arguments recorded before old_string.
- file state: the snapshot file (data/snapshots/<iid>.tgz) with every earlier successful edit_file / write_file
  of the same trace re-applied by swegemma.edit.apply_replacement (the harness's own function). A file touched
  by a shell command that may modify it (sed -i, redirection, git checkout/apply, patch, mv/cp/rm) is marked
  `state_uncertain` from then on.
Output: outputs/weight_edit/claude/replay/calls.jsonl and files/<sha256>.txt (file states, deduplicated).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/weight_edit"))
import sft_export as X  # noqa: E402

RUNS = {  # label -> (run dir, suite)
    "o43a": ("results/e4a/results/ood_e001nt_baseline_nothink_t06_e4b_cap12", "ood45"),
    "o43b": ("results/e4a/results/ood_e001nt_baseline_nothink_t06_e4b_cap12_r2", "ood45"),
    "o43t": ("results/e4t/results/ood_e001nt_baseline_nothink_t06_e4b_cap12", "ood45"),
    "o43v": ("results/e4v/results/ood_e001nt_baseline_nothink_t06_e4b_cap12", "ood45"),
    "oodb": ("outputs/oodb_e001nt/results", "oodb"),
}
FILES = ROOT / "data/ext/ood_files/oodfiles"
TASK_FILES = {"ood45": ROOT / "data/ext/ood45/tasks.jsonl", "oodb": FILES / "oodb/tasks.jsonl"}

SUB = ROOT / "submissions/e001nt_baseline_nothink_t06_e4b"
OUT = ROOT / "outputs/weight_edit/claude/replay_ood"
MODIFY_RE = re.compile(r"sed\s+-i|>\s*\S+\.py|\btee\b|git\s+(checkout|apply|restore|stash|reset)|\bpatch\b|\bmv\b|\bcp\b|\brm\b")


def snapshot_text(suite: str, iid: str, rel: str) -> str | None:
    f = FILES / suite / iid / rel.removeprefix("./")
    if not f.is_file():
        return None
    return f.read_text(encoding="utf-8", errors="replace")


def load_tasks() -> dict[str, dict]:
    out = {}
    for suite, p in TASK_FILES.items():
        for line in p.read_text().splitlines():
            if line.strip():
                t = json.loads(line)
                out[t["instance_id"]] = t
    return out


def fmt_args(args: dict) -> str:
    return ",".join(f"{k}:{X._fmt_arg(v)}" for k, v in args.items())


def main() -> None:
    from swegemma.edit import apply_replacement

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("--limit-traces", type=int, default=0)
    a = ap.parse_args()
    tasks = load_tasks()
    rnd = X.Renderer(X.DEFAULT_TOKENIZER, X.DEFAULT_TEMPLATE)
    env = X.AgentEnv(SUB)
    agent_name = env.root.name
    enable_thinking = env.enable_thinking(agent_name)
    (OUT / "files").mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    out = (OUT / "calls.jsonl").open("w")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for label in a.runs.split(","):
            rdir, suite = ROOT / RUNS[label][0], RUNS[label][1]
            traces = sorted((rdir / "traces").glob("trace_*.json"))
            if a.limit_traces:
                traces = traces[: a.limit_traces]
            for trace in traces:
                iid = trace.stem.removeprefix("trace_")
                try:
                    system, items = X.parse_trace(trace)
                except Exception as e:  # noqa: BLE001
                    stats["parse_error"] += 1
                    print(f"{label} {iid}: parse error {e}", file=sys.stderr)
                    continue
                X.recover_thought_whitespace(items, rnd, stats)
                state = X.base_state(tasks[iid])
                events, index = X.calls_to_events(items, agent_name, with_thoughts=True)
                files: dict[str, str | None] = {}
                uncertain: set[str] = set()
                all_uncertain = False
                for i, it in enumerate(items):
                    if not isinstance(it, X.Call):
                        continue
                    prompt = None
                    for j, (name, args) in enumerate(it.tool_calls):
                        resp = it.responses[j][1] if j < len(it.responses) else None
                        try:
                            o = json.loads(resp) if isinstance(resp, str) else (resp or {})
                        except ValueError:
                            o = {}
                        fp = args.get("filepath")
                        if name == "edit_file" and fp:
                            if fp not in files:
                                files[fp] = snapshot_text(suite, iid, fp)
                            if prompt is None:
                                msgs, tools = env.request(agent_name, events[: index[i]], state)
                                conv = X.vllm_conversation(msgs)
                                prompt = rnd.render(conv, X.vllm_tools(tools), add_generation_prompt=True,
                                                    enable_thinking=enable_thinking)
                                n_ids = len(rnd.encode(prompt))
                            pre = ""
                            if it.thought is not None:
                                pre += "<|channel>thought\n" + it.thought + "<channel|>"
                            pre += it.text
                            for name2, args2 in it.tool_calls[:j]:
                                pre += "<|tool_call>call:" + name2 + "{" + fmt_args(args2) + "}<tool_call|>"
                            keys = list(args)
                            before = {k: args[k] for k in keys[: keys.index("old_string")]} if "old_string" in keys else dict(args)
                            pre += "<|tool_call>call:edit_file{" + fmt_args(before) + ("," if before else "") + 'old_string:<|"|>'
                            content = files.get(fp)
                            sha = None
                            if content is not None:
                                sha = hashlib.sha256(content.encode()).hexdigest()[:16]
                                fpath = OUT / "files" / f"{sha}.txt"
                                if not fpath.exists():
                                    fpath.write_text(content)
                            status = o.get("status") or ("error" if "error" in o else None)
                            msg = str(o.get("error_message") or o.get("error") or "")
                            kind = ("ok" if status == "ok" else "not_found" if "not found" in msg else
                                    "multiple" if "occurs" in msg else "missing_param" if "mandatory" in msg else "other")
                            row = {
                                "key": f"{label}|{iid}|{i}|{j}", "run": label, "suite": suite, "instance_id": iid,
                                "repo": tasks[iid]["repo"], "item": i, "tc": j, "filepath": fp,
                                "prompt_tokens_trace": it.prompt_tokens, "prompt_tokens_ours": n_ids,
                                "verified": int(it.prompt_tokens is not None and n_ids == it.prompt_tokens),
                                "prompt": prompt, "prefix": pre,
                                "old_string": args.get("old_string"), "new_string": args.get("new_string"),
                                "allow_multiple": bool(args.get("allow_multiple", False)),
                                "order": ",".join(keys), "error_kind": kind, "strategy": o.get("strategy"),
                                "file_sha": sha, "file_missing": int(content is None),
                                "state_uncertain": int(all_uncertain or fp in uncertain),
                            }
                            out.write(json.dumps(row) + "\n")
                            stats["calls"] += 1
                            stats["verified"] += row["verified"]
                            stats[f"kind_{kind}"] += 1
                            # re-apply successful edits to the true state
                            if status == "ok" and content is not None and args.get("old_string"):
                                try:
                                    res = apply_replacement(content, args["old_string"], args.get("new_string", ""),
                                                            filepath=fp, allow_multiple=bool(args.get("allow_multiple", False)))
                                    files[fp] = res.new_content if hasattr(res, "new_content") else res[0]
                                except Exception:  # noqa: BLE001
                                    uncertain.add(fp)
                                    stats["reapply_failed"] += 1
                        elif name == "write_file" and fp and status_ok(o):
                            files[fp] = args.get("content", "")
                        elif name == "run_command" and MODIFY_RE.search(str(args.get("command", ""))):
                            hit = [f for f in files if f in str(args.get("command", ""))]
                            if hit:
                                uncertain.update(hit)
                            else:
                                all_uncertain = True
                            stats["modifying_commands"] += 1
    out.close()
    print(json.dumps(stats, indent=1))


def status_ok(o: dict) -> bool:
    return o.get("status") == "ok"


if __name__ == "__main__":
    main()
