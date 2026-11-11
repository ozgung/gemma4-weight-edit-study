"""Tests of gate_r3_analyze.py on synthetic arms built from the used round-2 gate set.  Owner: Claude.

    python3 -I scripts/weight_edit/test_gate_r3.py

Each arm is the base arm with a known change, so every verdict below is known in advance. No GPU, writes only to a temp dir.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/weight_edit"))
import gate_r3_analyze as GA  # noqa: E402

C = ROOT / "outputs/weight_edit/claude"
SCRIPT = ROOT / "scripts/weight_edit/gate_r3_analyze.py"


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open()]


class World:
    def __init__(self, d: Path):
        self.d = d
        (d / "fullcall").mkdir(parents=True)
        self.calls = {c["key"]: c for c in jl(C / "replay_ood/calls.jsonl")}
        self.failed = GA.failed_sets(list(self.calls.values()))
        self.base_g = jl(C / "gate_r2/gen_base.jsonl")
        self.base_s = jl(C / "gate_r2/gen_base.scored.jsonl")
        self.fc_g = jl(C / "gate_r2/fullcall/gen_base.jsonl")
        self.fc_s = jl(C / "gate_r2/fullcall/gen_base.scored.jsonl")
        self.write("base", self.base_g, self.base_s, self.fc_g, self.fc_s, edit=None)

    def write(self, arm, g, s, fc_g=None, fc_s=None, edit="x"):
        e = None if edit is None else f"gate/{arm}/edit"
        with (self.d / f"gen_{arm}.jsonl").open("w") as f:
            for r in g:
                f.write(json.dumps({**r, "edit": e}) + "\n")
        (self.d / f"gen_{arm}.scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in s))
        if fc_g is not None:
            with (self.d / f"fullcall/gen_{arm}.jsonl").open("w") as f:
                for r in fc_g:
                    f.write(json.dumps({**r, "edit": e}) + "\n")
            (self.d / f"fullcall/gen_{arm}.scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in fc_s))

    def arm(self, name, change=None, fc_change=None):
        """change(g, s) edits one base record in place."""
        g = [dict(r) for r in self.base_g]
        s = [dict(r) for r in self.base_s]
        if change:
            for a, b in zip(g, s):
                change(a, b)
        fg = [dict(r) for r in self.fc_g]
        fs = [dict(r) for r in self.fc_s]
        if fc_change:
            for a, b in zip(fg, fs):
                fc_change(a, b)
        self.write(name, g, s, fg, fs)
        for x in ("rand1", "rand2"):
            self.write(f"{name}_{x}", [dict(r) for r in self.base_g], [dict(r) for r in self.base_s], edit="x")

    def run(self, *cands, m=2, expect_block=False, replay=None):
        out = self.d / "report.json"
        p = subprocess.run([sys.executable, "-I", str(SCRIPT), "--replay", str(replay or C / "replay_ood"), "--gen-dir", str(self.d), "--m", str(m),
                            "--out", str(out), *cands], capture_output=True, text=True)
        if expect_block:
            assert p.returncode != 0 and "BLOCKED" in p.stderr, (p.returncode, p.stderr[-300:])
            return None
        assert p.returncode == 0, p.stderr[-800:]
        return json.loads(out.read_text())


def main() -> None:
    assert [GA.served_ok_limit(n) for n in (159, 160, 319, 320, 480, 640, 1000)] == [0, 1, 1, 2, 3, 4, 6]

    # failed sets: unusable calls count, order is (item, tc), runs do not mix, tasks do not mix
    mk = lambda key, run, iid, item, tc, kind, old: {"key": key, "run": run, "instance_id": iid, "item": item, "tc": tc, "error_kind": kind, "old_string": old}  # noqa: E731
    fs = GA.failed_sets([mk("a3", "r", "t", 3, 0, "ok", "z"), mk("a1", "r", "t", 1, 0, "not_found", "x"), mk("a2", "r", "t", 2, 1, "not_found", "y"),
                         mk("a0", "r", "t", 2, 0, "multiple", "w"), mk("b", "s", "t", 9, 0, "ok", "x"), mk("c", "r", "u", 9, 0, "ok", "x")])
    assert fs == {"a1": [], "a0": ["x"], "a2": ["x"], "a3": ["x", "y"], "b": [], "c": []}, fs

    with tempfile.TemporaryDirectory() as td:
        w = World(Path(td) / "gate")
        ood_failed = w.failed
        retry_keys = {k for k, v in ood_failed.items() if v}

        def fix_repeats(g, s, accept=True):
            if g["gen"] in ood_failed[g["key"]]:
                g["gen"] = g["gen"] + " changed"
                if accept:
                    s["accepted"] = 1

        # a clean winner: every repeat replaced and accepted; nothing else touched
        w.arm("good", fix_repeats)
        r = w.run("good", m=2)
        c = r["candidates"]["good"]
        assert r["base"]["repeats"] == 78 and r["base"]["retry_accepted"] == 38 and r["retry_calls"] == 71 and r["tasks_with_retry"] == 33
        assert r["reportable"] is False
        assert c["primary_repeat_rate"]["negative_means"] and c["primary_repeat_rate"]["p"] < 1e-3, c["primary_repeat_rate"]["p"]
        assert c["reported"]["counts"]["candidate"]["repeats"] == 0 and c["reported"]["counts"]["candidate"]["first_identical_to_base_share"] == 1.0
        assert c["guards_pass"] and c["passes"] and c["secondary_passes"], c

        # primary without the secondary: repeats gone, nothing accepted more
        w.arm("noacc", lambda g, s: fix_repeats(g, s, accept=False))
        c = w.run("noacc")["candidates"]["noacc"]
        assert c["passes"] and not c["secondary_passes"], c["secondary_retry_acceptance"]["p"]
        # a repeat changed by a few bytes passes the primary: the rule allows it, near_repeats shows it (long strings only reach 0.95)
        nc = c["reported"]["counts"]["candidate"]
        assert nc["repeats"] == 0 and nc["near_repeats"] >= 60, nc

        # an arm equal to its controls has no primary effect
        w.arm("same")
        c = w.run("same")["candidates"]["same"]
        assert c["primary_repeat_rate"]["p"] == 1.0 and not c["passes"]
        assert c["secondary_retry_acceptance"]["p"] == 1.0  # not tested

        # guard (b): exactly the limit passes, one more fails; first-attempt or retry, any recorded-ok call counts
        ok0 = [r for r in w.base_s if r["recorded_kind"] == "ok" and r["accepted"] and r["draw"] == 0]

        def lose(n):
            keys = {(r["key"], r["draw"]) for r in ok0[:n]}

            def ch(g, s):
                fix_repeats(g, s)
                if (s["key"], s["draw"]) in keys:
                    s["accepted"] = 0
            return ch
        w.arm("lose2", lose(2))
        assert w.run("lose2")["candidates"]["lose2"]["guard_b_served_ok"]["pass"]
        w.arm("lose3", lose(3))
        c = w.run("lose3")["candidates"]["lose3"]
        assert not c["guard_b_served_ok"]["pass"] and not c["passes"] and c["secondary_retry_acceptance"]["p"] == 1.0

        # a served not_found that base accepted and the candidate loses is not a served-ok loss
        nf0 = [r for r in w.base_s if r["recorded_kind"] == "not_found" and r["accepted"]][:3]
        ks = {(r["key"], r["draw"]) for r in nf0}

        def ch(g, s):
            fix_repeats(g, s)
            if (s["key"], s["draw"]) in ks:
                s["accepted"] = 0
        w.arm("nfloss", ch)
        assert w.run("nfloss")["candidates"]["nfloss"]["guard_b_served_ok"]["pass"]

        # guard (a): one small task loses everything; task-macro falls below base - 0.01
        by_task = {}
        for r in w.base_s:
            by_task.setdefault(r["key"].split("|")[1], []).append(r)
        small = next(v for t, v in sorted(by_task.items()) if len(v) == 2 and all(x["accepted"] for x in v) and not any(ood_failed[x["key"]] for x in v))
        ks2 = {(x["key"], x["draw"]) for x in small}

        def drop(g, s):
            if (s["key"], s["draw"]) in ks2:
                s["accepted"] = 0
        def drop_fix(g, s):
            fix_repeats(g, s)
            drop(g, s)
        w.arm("acc", lambda g, s: drop(g, s))  # 1 / 101 = -0.0099: passes guard (a) alone
        c = w.run("acc")["candidates"]["acc"]
        assert c["guard_a_acceptance"]["pass"], c["guard_a_acceptance"]
        second = next(v for t, v in sorted(by_task.items()) if len(v) == 2 and v != small and all(x["accepted"] for x in v) and not any(ood_failed[x["key"]] for x in v))
        ks3 = {(x["key"], x["draw"]) for x in second}

        def drop2(g, s):
            if (s["key"], s["draw"]) in ks2 | ks3:
                s["accepted"] = 0
        w.arm("acc2", drop2)  # 2 / 101
        c = w.run("acc2")["candidates"]["acc2"]
        assert not c["guard_a_acceptance"]["pass"], c["guard_a_acceptance"]

        # guard (c): the plain replay is not used for it
        rows = [r for r in w.fc_s if r["accepted"] and r["compiles"] == 1]

        def compile_up(n):
            ks = {(r["key"], r["draw"]) for r in rows[:n]}

            def ch(g, s):
                if (s["key"], s["draw"]) in ks:
                    s["compiles"] = 0
            return ch
        w.arm("comp8", fix_repeats, compile_up(8))  # 8 / 444 = 1.8 points
        assert w.run("comp8")["candidates"]["comp8"]["guard_c_fullcall_compile"]["pass"]
        w.arm("comp9", fix_repeats, compile_up(9))  # 2.03 points
        assert not w.run("comp9")["candidates"]["comp9"]["guard_c_fullcall_compile"]["pass"]

        # Holm: two candidates; a large m lifts the threshold above their p
        w.arm("weak", lambda g, s: fix_repeats(g, s) if s["key"] in sorted(retry_keys)[:30] else None)
        r = w.run("good", "weak", m=2)
        assert r["holm_m"] == 2 and r["candidates"]["good"]["passes"]
        tiny = max(r["candidates"][c]["primary_repeat_rate"]["p"] for c in ("good", "weak"))
        r2 = w.run("good", "weak", m=int(0.05 / tiny) + 2)
        assert not r2["candidates"]["good"]["passes"] or not r2["candidates"]["weak"]["passes"]

        # completeness blocks
        g = [dict(r) for r in w.base_g]
        s = [dict(r) for r in w.base_s]
        w.write("short", g[:-1], s[:-1], w.fc_g, w.fc_s)
        for x in ("rand1", "rand2"):
            w.write(f"short_{x}", g, s)
        w.run("short", expect_block=True)
        w.write("dup", g + g[-1:], s, w.fc_g, w.fc_s)
        for x in ("rand1", "rand2"):
            w.write(f"dup_{x}", g, s)
        w.run("dup", expect_block=True)
        w.write("noedit", g, s, w.fc_g, w.fc_s, edit=None)
        for x in ("rand1", "rand2"):
            w.write(f"noedit_{x}", g, s)
        w.run("noedit", expect_block=True)
        w.write("nofc", g, s)
        for x in ("rand1", "rand2"):
            w.write(f"nofc_{x}", g, s)
        p = subprocess.run([sys.executable, "-I", str(SCRIPT), "--replay", str(C / "replay_ood"), "--gen-dir", str(w.d), "nofc"], capture_output=True, text=True)
        assert p.returncode != 0

        # a usable_keys.txt that differs from the rule blocks; no usable_keys.txt falls back to the rule
        rp = Path(td) / "replay"
        rp.mkdir()
        shutil.copy(C / "replay_ood/calls.jsonl", rp / "calls.jsonl")
        w.arm("good")
        (rp / "usable_keys.txt").write_text("\n".join(sorted(k for k in w.calls)[:10]) + "\n")
        w.run("good", expect_block=True, replay=rp)
        (rp / "usable_keys.txt").unlink()
        p = subprocess.run([sys.executable, "-I", str(SCRIPT), "--replay", str(rp), "--gen-dir", str(w.d), "--prepare"], capture_output=True, text=True)
        assert p.returncode == 0 and "usable 320, full-call 320" in p.stdout, p.stdout + p.stderr
        assert set((w.d / "keys_usable.txt").read_text().split("\n")) - {""} == {l.strip() for l in (C / "replay_ood/usable_keys.txt").open()}
    print("all gate tests passed")


if __name__ == "__main__":
    main()
