"""Tests of harness v3 without a model: generation is stubbed with the stored round-2 gate generations.  Owner: Claude.

    python3 -I scripts/weight_edit/test_wsp_server_r3.py

Everything else is the real code path: request files, handle(), budget, expect_edit, v2 do_gen and do_fullcall, the
scorers run as subprocesses against search_r3, the by_set figures. Expected values come from the retry census
(outputs/weight_edit/claude/retry_census/census_20261107.txt) and the round-2 reports. No GPU, no writes outside a temp dir.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/weight_edit"))
import wsp_server as V2  # noqa: E402
import wsp_server_r3 as V3  # noqa: E402

G = ROOT / "outputs/weight_edit/claude/gate_r2"


def rows(p: Path) -> dict:
    return {(g["key"], g["draw"]): g for g in map(json.loads, p.open())}


class Stub(V3.ServerR3):
    def __init__(self, cache: Path):
        V2.CACHE = cache
        self.load_tables()
        self.arm = "base"
        self.store = {"base": rows(G / "gen_base.jsonl"), "edit": rows(G / "gen_fable_r2.jsonl")}
        self.store_fc = {"base": rows(G / "fullcall/gen_base.jsonl"), "edit": rows(G / "fullcall/gen_fable_r2.jsonl")}
        for g in map(json.loads, (ROOT / "outputs/weight_edit/wsp_cache/base_gen.jsonl").open()):  # train calls
            self.store["base"].setdefault((g["key"], g["draw"]), g)
            self.store["edit"].setdefault((g["key"], g["draw"]), dict(g, gen=g["gen"] + " "))
        self.n_generate = {"base": 0, "edit": 0}

    def apply(self, run_dir):
        self.arm = "edit"
        return {"x": 1}

    def restore(self, originals):
        self.arm = "base"

    def generate(self, r, draws):
        self.n_generate[self.arm] += 1
        return [{k: self.store[self.arm][(r["key"], d)][k] for k in ("key", "draw", "gen", "closed", "n_gen")} for d in range(draws)]

    def generate_fullcall(self, r, draws):
        keep = ("key", "draw", "gen_new", "gen", "closed_new", "closed", "n_gen_new", "n_gen")
        return [{k: self.store_fc[self.arm][(r["key"], d)][k] for k in keep} for d in range(draws)]


def main() -> None:
    V3.check_v2_source()
    V2.LIMITS["gen_calls"] = 400  # the whole dev set in one request, for the comparison with the census
    dev = [l.strip() for l in (ROOT / "outputs/weight_edit/claude/replay_ood/usable_keys.txt").read_text().splitlines() if l.strip()]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        S = Stub(td / "cache")
        assert len(S.calls) == 1050 and sum(c["set"] == "dev" for c in S.calls.values()) == 320
        assert sum(c["is_retry"] for c in S.calls.values() if c["set"] == "dev") == 71

        class A:
            run_id, proposer, model, effort, budget, fullcall, proposers = "t3", "test", "none", "max", 8, True, td / "props"
        V3.init_run(A)
        run = A.proposers / "t3"
        state = json.loads((run / "state.json").read_text())
        assert state["harness_version"] == 3 and state["fullcall"] is True
        (run / "edit" / "a.npz").write_bytes(b"not a real edit; apply() is stubbed")
        snap = {"a.npz": hashlib.sha256(b"not a real edit; apply() is stubbed").hexdigest()}

        def req(k, body):
            p = run / "requests" / f"fwd_{k}.json"
            p.write_text(json.dumps(body))
            S.handle(p)
            return json.loads((run / "requests" / f"fwd_{k}.done.json").read_text())

        # 1. whole dev set, base + edit, plain and full call, against the census and the round-2 reports
        d = req(1, {"type": "gen", "calls": dev, "draws": 2, "expect_edit": snap})
        assert d["status"] == "ok", d
        assert d["edit_snapshot"] == snap and d["passes_used"] == 1
        r = d["result"]
        b, e = r["arms"]["base"]["by_set"]["dev"], r["arms"]["edit"]["by_set"]["dev"]
        assert "train" not in r["arms"]["base"]["by_set"]
        assert (b["retry"]["n_calls"], b["retry"]["n_gens"], b["retry"]["repeats"], b["retry"]["accepted"]) == (71, 142, 78, 38), b
        assert b["retry"]["repeats_accepted"] == 0
        assert (b["first"]["n_gens"], b["first"]["accepted"], b["first"]["identical_to_base"]) == (498, 405, 498), b
        assert (e["retry"]["repeats"], e["retry"]["accepted"], e["first"]["accepted"], e["first"]["identical_to_base"]) == (64, 45, 400, 418), e
        assert e["served_ok_lost_per_draw"]["0"]["lost"] == 5 and e["served_ok_lost_per_draw"]["1"]["lost"] == 5, e
        rep = json.loads((G / "gate_r2_report.json").read_text())["candidates"]["fable_r2"]
        assert e["served_ok_lost_per_draw"]["0"]["base_accepted_served_ok"] == rep["served_ok_loss"]["draw0"]["base_accepted_served_ok"]
        assert abs(e["compile_fail_among_accepted"] - rep["compile"]["edit"]["compile_fail_rate_among_accepted"]) < 1e-12
        assert abs(b["compile_fail_among_accepted"] - rep["compile"]["base"]["compile_fail_rate_among_accepted"]) < 1e-12
        assert abs(e["overall_acceptance"] - 445 / 640) < 1e-12 and abs(b["overall_acceptance"] - 443 / 640) < 1e-12
        # scorer against search_r3 agrees with the gate's scorer, record by record
        for arm, f in (("base", "gen_base"), ("edit", "gen_fable_r2")):
            s = {k: x for k, x in rows(run / "work/fwd_1" / f"gen_{arm}.scored.jsonl").items()}
            g = rows(G / f"{f}.scored.jsonl")
            assert set(s) == set(g)
            assert all(s[k] == g[k] for k in g), arm
        fc = r["fullcall"]
        assert fc["skipped"] == ["oodb|haystack_10547|3|0"], fc["skipped"]
        fb, fe = fc["arms"]["base"]["by_set"]["dev"], fc["arms"]["edit"]["by_set"]["dev"]
        assert fb["n_calls"] == 319 and fb["first"]["identical_to_base"] == fb["first"]["n_gens"], fb
        amended = json.loads((G / "gate_r2_report_amended.json").read_text())
        # full-call figures miss one call here (v2's stricter cut), so they are near the amended report, not equal to it
        assert abs(fb["compile_fail_among_accepted"] - amended["base"]["compile_fail_rate_among_accepted"]) < 0.01, fb
        assert abs(fe["compile_fail_among_accepted"] - amended["candidates"]["fable_r2"]["fullcall"]["compile_fail_rate_among_accepted"]) < 0.01, fe

        # 2. base is cached: a second pass generates nothing for base; "edit" alone still gets base figures
        before = dict(S.n_generate)
        d = req(2, {"type": "gen", "arms": ["edit"], "calls": dev[:8], "draws": 2, "fullcall": False})
        assert d["status"] == "ok", d
        assert S.n_generate["base"] == before["base"] and S.n_generate["edit"] == before["edit"] + 8
        assert set(d["result"]["arms"]) == {"base", "edit"} and "fullcall" not in d["result"]
        assert "identical_to_base" in d["result"]["arms"]["edit"]["by_set"]["dev"]["first"]

        # 3. a request mixing train and dev is split by set
        train = sorted(k for k, c in S.calls.items() if c["set"] == "train" and (k, 1) in S.store["base"])[:12]
        d = req(3, {"type": "gen", "calls": train + dev[:4], "draws": 2, "fullcall": False})
        bs = d["result"]["arms"]["edit"]["by_set"]
        assert d["status"] == "ok" and bs["train"]["n_calls"] == 12 and bs["dev"]["n_calls"] == 4, bs
        assert bs["train"]["first"]["identical_to_base"] == 0 and bs["train"]["retry"]["repeats"] <= bs["train"]["retry"]["n_gens"]

        # 4. refusals do not count
        used = json.loads((run / "state.json").read_text())["used"]
        assert used == 3
        d = req(4, {"type": "gen", "calls": dev[:2], "rendering": "raw"})
        assert d["status"].startswith("failed") and "raw" in d["error"], d
        d = req(5, {"type": "gen", "calls": dev[:2], "expect_edit": {"a.npz": "0"}})
        assert d["status"].startswith("failed") and "expect_edit" in d["error"], d
        d = req(6, {"type": "gen", "calls": ["no|such|key|0"]})
        assert d["status"].startswith("failed"), d
        assert json.loads((run / "state.json").read_text())["used"] == 3

        # 5. budget
        st = json.loads((run / "state.json").read_text())
        st["used"] = st["budget"]
        (run / "state.json").write_text(json.dumps(st))
        d = req(7, {"type": "gen", "calls": dev[:2]})
        assert d["status"] == "refused", d

        # 6. the v3 loop leaves v2 runs alone
        old = A.proposers / "old_v2"
        (old / "requests").mkdir(parents=True)
        (old / "state.json").write_text(json.dumps({"harness_version": 2, "used": 0, "budget": 1}))
        (old / "requests/fwd_1.json").write_text("{}")
        S.loop(once=True, proposers=A.proposers)
        assert (old / "requests/fwd_1.json").exists()

    # the real spec extraction works on this file
    import ast
    t = ast.parse((ROOT / "scripts/weight_edit/wsp_server_r3.py").read_text())
    assert any(isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "LIMITS" for n in t.body)
    print("all harness v3 tests passed")


if __name__ == "__main__":
    main()
