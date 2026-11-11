"""Run a submission on OOD-45 (rodrigospalma/ood45-bugfix-split: 45 post-release tasks, 17 repos the
agent has never seen), using the dataset's own tools/ood45_harness.py for per-task environments.

    nohup /content/venv/bin/python /content/g4/scripts/colab/run_ood45.py --ood /content/ood45 \
        --submission-dir /content/g4/submissions/v002_guards --api-base http://127.0.0.1:8000/v1 \
        --concurrency 4 --results-dir /content/results/<run> > /content/results/<run>.log 2>&1 &

--gold / --baseline grade the reference patch / an empty patch (no model). The rewritten statement
('g1') is used unless --statement g2. urllib3_5279 is skipped on Python 3.12, as the dataset says.
One EvalConfig per repo, because each repo has its own wheel directory. Results are written in the
same layout as run_eval.py (task_results.jsonl, traces/, patches/), so scripts/analyze_run.py works.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

for k, v in {"LITELLM_LOCAL_MODEL_COST_MAP": "True", "LITELLM_TELEMETRY": "False", "LITELLM_LOG": "WARNING",
             "OPENAI_API_KEY": "EMPTY", "OTEL_SDK_DISABLED": "true"}.items():
    os.environ.setdefault(k, v)

import yaml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood", type=Path, required=True)
    ap.add_argument("--submission-dir", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--api-base")
    ap.add_argument("--models-root", type=Path, default=Path("/content/models"))
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--statement", choices=["g1", "g2"], default="g1")
    ap.add_argument("--gold", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--tasks", help="comma-separated subset")
    ap.add_argument("--wheels-tmp", default=None,
                    help="per-run wheel symlink dir; concurrent runs must not share one (default /tmp/ood_wheels_<results dir name>)")
    a = ap.parse_args()

    sys.path.insert(0, str(a.ood / "tools"))
    os.environ.setdefault("OOD45_ENV_LOG", str(a.results_dir / "ood_env_install.jsonl"))
    import ood45_harness as ood
    import litellm
    import types
    import venv as _venv
    import swegemma.sandbox.subprocess as sp_mod
    from adk_eval_core.tracing import SessionTrace
    from adk_submission import discover_adapters
    from adk_submission.discovery import discover_declared_models
    from adk_submission.registry import ModelRegistry
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator

    litellm.drop_params = True

    # Ubuntu's Python lacks ensurepip; the venv still sees the host's pip through system site-packages.
    def _create_no_pip(env_dir, *args, **kwargs):
        kwargs["with_pip"] = False
        return _venv.create(env_dir, *args, **kwargs)
    sp_mod.venv = types.SimpleNamespace(create=_create_no_pip)

    run = ood.setup(a.ood, wheels_tmp=a.wheels_tmp or f"/tmp/ood_wheels_{a.results_dir.name}")
    sub = a.submission_dir.resolve()
    ev = (yaml.safe_load((sub / "eval_config.yaml").read_text()) or {}).get("evaluation", {})
    (model,) = discover_declared_models(sub)
    adapters = discover_adapters(str(sub), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    models = ModelRegistry()
    if not (a.gold or a.baseline):
        from google.adk.models.lite_llm import LiteLlm
        from adk_submission.resolvers.generation import install_litellm_reasoning_patch
        install_litellm_reasoning_patch()
        models.register(model, LiteLlm(model=f"openai/{a.models_root / model}", api_base=a.api_base,
                                       api_key="EMPTY"))
        for name in getattr(adapters, "adapters", None) or {}:  # the server must have these adapters mounted
            for alias in (name, f"adapter:{name}"):
                models.register(alias, LiteLlm(model=f"openai/{name}", api_base=a.api_base, api_key="EMPTY"))

    ids = sorted(run.tasks)
    if a.tasks:
        ids = [i for i in ids if i in set(a.tasks.split(","))]
    if sys.version_info[:2] == (3, 12):
        ids = [i for i in ids if i != "urllib3_5279"]
    a.results_dir.mkdir(parents=True, exist_ok=True)
    limits, gen = build_submission_limits()

    class GoldEvaluator(Evaluator):
        async def _run_agent_sandbox(self, task, snapshot_path, task_index, total_tasks, **kw):
            return task.patch, None, SessionTrace()

    evaluators = {}

    def evaluator_for(repo: str):
        if repo not in evaluators:
            cfg = EvalConfig(
                tasks_path=a.ood / "tasks.jsonl", snapshots_dir=run.snapshots, results_dir=a.results_dir,
                submission_dir=sub, models=models, sandbox="subprocess",
                timeout_seconds=int(ev.get("timeout_seconds", 300)), verification_timeout_seconds=300,
                max_time_minutes=float(ev.get("max_time_minutes", 60.0)),
                max_tool_calls=int(ev.get("max_tool_calls", 100)),
                max_turns=int(ev["max_turns"]) if ev.get("max_turns") is not None else None,
                limits=limits, generation_constraints=gen, adapter_manifest=adapters,
                context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
                events_compaction_config=EventsCompactionConfig(
                    compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
                wheels_dir=run.wheels_for(repo), display_mode="quiet", skip_agent_patch=a.baseline)
            evaluators[repo] = (GoldEvaluator if a.gold else Evaluator)(cfg)
        return evaluators[repo]

    out = a.results_dir / "task_results.jsonl"
    sem = asyncio.Semaphore(a.concurrency)

    async def one(idx: int, tid: str):
        task = run.tasks[tid][a.statement]
        async with sem:
            t0 = time.time()
            try:
                r = await evaluator_for(task.repo).evaluate_task(task=task, task_index=idx, total_tasks=len(ids))
                (a.results_dir / "patches").mkdir(exist_ok=True)
                (a.results_dir / "patches" / f"{tid}.patch").write_text(r.agent_patch or "")
                (a.results_dir / "test_outputs").mkdir(exist_ok=True)
                (a.results_dir / "test_outputs" / f"{tid}.log").write_text(r.test_output or "")
                row = {"instance_id": tid, "repo": task.repo, "resolved": bool(r.resolved),
                       "agent_patch_size": len(r.agent_patch or ""), "test_exit_code": r.test_exit_code,
                       "duration_seconds": round(r.duration_seconds or (time.time() - t0), 2),
                       "error": r.error, "tool_calls": r.tool_calls,
                       "total_llm_calls": getattr(r, "total_llm_calls", None), "task_index": idx}
            except Exception as e:  # keep going; one broken env must not stop the run
                row = {"instance_id": tid, "repo": task.repo, "resolved": False, "agent_patch_size": 0,
                       "test_exit_code": -1, "duration_seconds": round(time.time() - t0, 2),
                       "error": f"runner: {e!r}"[:300], "tool_calls": 0, "total_llm_calls": 0, "task_index": idx}
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{idx}/{len(ids)}] {tid} resolved={row['resolved']} {row['duration_seconds']}s "
                  f"{(row['error'] or '')[:80]}", flush=True)

    async def all_tasks():
        await asyncio.gather(*(one(i, t) for i, t in enumerate(ids, 1)))

    t0 = time.time()
    asyncio.run(all_tasks())
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    meta = {"submission_dir": str(sub), "model": model, "eval_config": ev, "tasks": ids, "ood45": True,
            "statement": a.statement, "gold": a.gold, "baseline": a.baseline, "concurrency": a.concurrency,
            "wall_seconds": round(time.time() - t0, 1), "resolved": sum(r["resolved"] for r in rows),
            "total": len(rows)}
    (a.results_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print("RUN DONE", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
