"""Select OOD-C candidate tasks from the SWE-rebench family (nebius, CC-BY-4.0). Pure data processing: no repo code runs.

    python3 -I scripts/oodc/select_candidates.py --source data/ext/oodb/source --out data/ext/oodc/candidates.jsonl \
        --repo-meta data/ext/oodc/repo_meta.jsonl --per-repo 2 --n 400 \
        --old outputs/oodb/build2-s0 outputs/oodb/build2-s1 \
        --extra data/ext/oodc/source_filter/v2_python_2025.jsonl.gz data/ext/oodc/source_filter/v1_test_2025.jsonl.gz

Sources, in order of preference when a repository has rows in several: SWE-rebench-leaderboard (as OOD-B), then V2 Python
rows and V1 test-split rows from 2025 on (`--extra`, filtered on a Kaggle kernel; V2 rows must carry the dataset's clean
code `A` and a `*_bug` PR category). Rows from before 2025-01 are not used.

OOD-C is a held-out set for the weight-edit gate and a live-agent replication: every repository must be new to every
agent of ours. Same static rules as scripts/oodb/select_candidates.py, except:
  - repositories are also excluded when an agent of ours has run on them (OOD-B's kept tasks, the replay set of
    `outputs/weight_edit/claude/replay_ood/calls.jsonl`), compared on the GitHub canonical name, the bare repository
    name and the fork parent (`--repo-meta`, from the GitHub API; a repository that is a fork of an excluded one goes too);
  - tasks and repositories the OOD-B builder already showed cannot run are not rebuilt (`--old`);
  - the test-shape filter (it exists only for grading) is a preference: `tests_strict` is recorded per task, and relaxed
    tasks are used only with --relax-tests;
  - leaderboard rows with an `interface` field add a new API rather than fix a bug: --no-interface drops them.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_REPO_NAMES = {"fastapi", "rich", "requests", "httpx"}
OOD45_REPOS = {
    "apache/iceberg-python", "caronc/apprise", "feast-dev/feast", "fsspec/filesystem_spec", "has2k1/plotnine",
    "hgrecco/pint", "kedro-org/kedro", "kedro-org/kedro-viz", "pdm-project/pdm", "pola-rs/polars",
    "posit-dev/great-tables", "posit-dev/py-shiny", "pyjanitor-devs/pyjanitor", "tkem/cachetools", "tox-dev/tox",
    "urllib3/urllib3", "wntrblm/nox",
}
OK_PYTHON = {"3.10", "3.11", "3.12", "3.13"}
COMPILED = re.compile(r"\b(meson|cython|maturin|cmake|cargo|rustup|npm|yarn|pnpm|conda|mamba|make|build_ext|"
                      r"--no-build-isolation|setup\.py\s+build|f2py|bazel)\b", re.I)
APT_OK = {"gcc", "g++", "build-essential", "make", "git", "curl", "wget", "pkg-config", "libffi-dev", "libssl-dev",
          "python3-dev", "autoconf", "automake", "libtool", "ca-certificates"}
DOC_EXT = (".md", ".rst", ".txt", ".toml", ".cfg", ".ini", ".yml", ".yaml", ".json")
OSI = re.compile(r"^(MIT|BSD-[234]-Clause.*|Apache-2\.0|ISC|MPL-2\.0|PSF-2\.0|Python-2\.0|LGPL.*|GPL.*|AGPL.*|"
                 r"Unlicense|0BSD|Zlib|EPL-2\.0|BSL-1\.0|CC0-1\.0|MIT-0|UPL-1\.0)$")
# OOD-B build outcomes that say the repository itself cannot run in this harness (other tasks of it are skipped too)
REPO_DEAD = ("heavy dependencies", "no Python project at the repository root", "no importable package",
             "wheel set")
# Need torch or JAX by design (the builder would install them only to reject them, at about 10 minutes each)
HEAVY_REPOS = {"jax-ml/jax", "patrick-kidger/equinox", "microsoft/torchgeo", "huggingface/smolagents",
               "Lightning-AI/litdata", "autrainer/autrainer"}


def diff_files(patch: str) -> list[str]:
    return re.findall(r"^diff --git a/(\S+) b/\S+", patch, re.M)


def changed_lines_per_file(patch: str) -> dict[str, int]:
    out, cur = collections.Counter(), None
    for line in patch.splitlines():
        m = re.match(r"^diff --git a/(\S+) b/(\S+)", line)
        if m:
            cur = m.group(2)
            continue
        if line.startswith(("+++", "---")):
            continue
        if cur and line[:1] in "+-":
            out[cur] += 1
    return dict(out)


def is_test_path(p: str) -> bool:
    parts = p.lower().split("/")
    name = parts[-1]
    return (any(x in ("tests", "test", "testing") for x in parts[:-1]) or name.startswith("test_")
            or name.endswith("_test.py") or name == "conftest.py")


def apt_ok(pre: list[str]) -> bool:
    for cmd in pre or []:
        c = cmd.strip()
        if re.match(r"^(sudo\s+)?apt(-get)?\s+update", c):
            continue
        m = re.match(r"^(sudo\s+)?apt(-get)?\s+install\s+(.*)$", c)
        if not m:
            return False
        pkgs = [p for p in m.group(3).split() if not p.startswith("-")]
        if any(p not in APT_OK for p in pkgs):
            return False
    return True


def short_id(repo: str, iid: str) -> str:
    num = iid.rsplit("-", 1)[-1]  # V1 and V2 ids look like owner__repo-123
    return f"{repo.split('/', 1)[1].lower().replace('-', '_').replace('.', '_')}_{num}"


def load_extra(paths: list[Path]) -> list[dict]:
    """Rows of the V1 test split and of V2, in the leaderboard's row format."""
    out = []
    for p in paths:
        for line in gzip.open(p, "rt"):
            r = json.loads(line)
            if str(r["created_at"]) < "2025-01":
                continue
            r["_src"] = r.get("_source") or "nebius/SWE-rebench"
            ic = r.get("install_config") or {}
            if r["_src"].endswith("V2"):
                llm = (r.get("meta") or {}).get("llm_metadata") or {}
                r["_v2_ok"] = llm.get("code") == "A" and any(c.endswith("_bug") for c in llm.get("pr_categories") or [])
                r["_bug_label"] = ",".join(llm.get("pr_categories") or [])
                m = re.search(r"(\d)(\d+)$", str(ic.get("base_image_name")))
                r["install_config"] = {"python": f"{m.group(1)}.{m.group(2)}" if m else None,
                                       "install": " && ".join(ic.get("install") or []), "pip_packages": [],
                                       "reqs_path": [], "pre_install": [], "env_vars": None, "packages": "",
                                       "test_cmd": ic.get("test_cmd"), "env_yml_path": []}
                r["hints_text"] = ""
                r["interface"] = None
            out.append(r)
    return out


def excluded_repos() -> dict[str, str]:
    """repo -> why, for every repository an agent of ours has seen."""
    why = {}
    for r in OOD45_REPOS:
        why[r] = "OOD-45"
    for f, tag in ((ROOT / "data/ext/oodb/tasks.jsonl", "OOD-B kept"),
                   (ROOT / "research/raw4/oodb/tasks.jsonl", "OOD-B kept"),
                   (ROOT / "outputs/weight_edit/claude/replay_ood/calls.jsonl", "replay_ood")):
        for line in f.read_text().splitlines():
            if line.strip():
                why.setdefault(json.loads(line)["repo"], tag)
    return why


def key_names(repo: str, meta: dict) -> set[str]:
    """Every lower-case form under which a repository can be matched: asked, canonical, fork parent, source, and the
    bare repository names of those."""
    m = meta.get(repo) or {}
    full = {repo.lower()} | {str(m[k]).lower() for k in ("full_name", "parent", "source") if m.get(k)}
    return full | {x.split("/", 1)[1] for x in full}


def old_outcomes(dirs: list[Path]) -> tuple[dict, dict, set]:
    recs, gates = {}, {}
    for d in dirs:
        for line in (d / "build_report.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                recs[r["instance_id"]] = r
        g = d / "gate" / "gate.json"
        if g.exists():
            gates.update(json.loads(g.read_text()))
    dead = {r["repo"] for r in recs.values() if r.get("status") != "ok"
            and any(x in (r.get("error") or "") for x in REPO_DEAD)}
    return recs, gates, dead


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo-meta", type=Path, required=True, help="JSON lines from the GitHub API")
    ap.add_argument("--old", type=Path, nargs="*", default=[], help="OOD-B shard output dirs (build report, gate)")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--per-repo", type=int, default=2)
    ap.add_argument("--relax-tests", action="store_true", help="also use tasks that fail the grading-only test filter")
    ap.add_argument("--no-interface", action="store_true", help="drop rows with an `interface` field (new-API tasks)")
    ap.add_argument("--extra", type=Path, nargs="*", default=[], help="gz JSON lines of V2 / V1 rows")
    ap.add_argument("--dump-repos", type=Path, help="write repos lacking GitHub metadata at the licence step here")
    a = ap.parse_args()

    import pyarrow.parquet as pq

    meta = {}
    for line in a.repo_meta.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            meta[d["asked"]] = d
    rows = []
    for f in sorted(glob.glob(str(a.source / "*.parquet"))):
        rows += pq.read_table(f).to_pylist()
    for r in rows:
        r["_src"] = "nebius/SWE-rebench-leaderboard"
    rows += load_extra(a.extra)
    seen, uniq = set(), []
    for r in rows:
        key = short_id(r["repo"], r["instance_id"])  # one PR can sit in two sources, with different id case
        if key not in seen:  # leaderboard rows come first
            seen.add(key)
            uniq.append(r)
    funnel = []

    def step(name, keep):
        nonlocal uniq
        uniq = [r for r in uniq if keep(r)]
        funnel.append((len(uniq), len({r["repo"] for r in uniq}), name))

    funnel.append((len(uniq), len({r["repo"] for r in uniq}), "source rows (distinct instance ids, 2025-01 on)"))
    step("V2 rows: clean code A and a *_bug PR category", lambda r: r.get("_v2_ok", True))
    excl = excluded_repos()
    excl_keys = set(TRAIN_REPO_NAMES)
    for r in excl:
        excl_keys |= key_names(r, meta)
    for r in ("fastapi/fastapi", "Textualize/rich", "psf/requests", "encode/httpx"):
        excl_keys |= key_names(r, meta)
    step("repo not training, OOD-45, OOD-B kept or replay set (canonical name, bare name, fork parent)",
         lambda r: not (key_names(r["repo"], meta) & excl_keys))
    step("python 3.10-3.13", lambda r: str((r["install_config"] or {}).get("python")) in OK_PYTHON)
    step("no conda env, no compiled build",
         lambda r: not (r["install_config"] or {}).get("env_yml_path")
         and not COMPILED.search((r["install_config"] or {}).get("install") or ""))
    step("apt pre-install only compilers/-dev", lambda r: apt_ok((r["install_config"] or {}).get("pre_install")))

    def gold_ok(r):
        files = diff_files(r["patch"])
        if not files:
            return False
        src = [f for f in files if f.endswith(".py") and not is_test_path(f)]
        other = [f for f in files if not f.endswith(".py") and not f.lower().endswith(DOC_EXT)]
        if other or not 1 <= len(src) <= 3:
            return False
        cl = changed_lines_per_file(r["patch"])
        n = sum(cl.get(f, 0) for f in src)
        r["_src_files"], r["_src_lines"] = len(src), n
        return 2 <= n <= 120
    step("gold: 1-3 .py files, 2-120 lines, no non-py code", gold_ok)
    step("issue text >= 250 chars", lambda r: len((r["problem_statement"] or "").strip()) >= 250)
    print("repos without GitHub metadata at the licence step:", len({r["repo"] for r in uniq if r["repo"] not in meta}))
    if a.dump_repos:
        a.dump_repos.write_text("\n".join(sorted({r["repo"] for r in uniq if r["repo"] not in meta})) + "\n")
    step("repo licence OSI-approved (GitHub API, 2026-11-07)",
         lambda r: bool(OSI.match(str((meta.get(r["repo"]) or {}).get("spdx") or ""))))
    if a.no_interface:
        step("no `interface` field (rows that add a new API)", lambda r: not r.get("interface"))

    recs, gates, dead = old_outcomes(a.old)

    def tests_ok(r):
        tfiles = [f for f in diff_files(r["test_patch"]) if f.endswith(".py")]
        f2p = list(r["FAIL_TO_PASS"] or [])
        inside = [t for t in list(r["PASS_TO_PASS"] or []) + f2p if t.split("::", 1)[0] in tfiles]
        r["_tests_in_files"] = len(inside)
        return (1 <= len(tfiles) <= 3 and 1 <= len(f2p) <= 60 and len(inside) <= 400
                and not any("::" not in t or t.split("::", 1)[0] not in tfiles for t in f2p))
    for r in uniq:
        r["_tests_strict"] = tests_ok(r)
        r["_old"] = recs.get(short_id(r["repo"], r["instance_id"]))
        g = gates.get(short_id(r["repo"], r["instance_id"]))
        r["_old_gate"] = g
    if not a.relax_tests:
        step("tests: 1-3 files, F2P in them, <=400 tests (grading only)", lambda r: r["_tests_strict"])

    def not_dead(r):
        old = r["_old"]
        if old is not None and old.get("status") != "ok":
            return False  # failed to build in OOD-B
        err = str((r["_old_gate"] or {}).get("gold_error") or "")
        if "not importable" in err or "SandboxEnvError" in err:
            return False  # environment broken in the sandbox: the agent cannot run either
        return r["repo"] not in dead and r["repo"] not in HEAVY_REPOS
    step("not a known build or sandbox failure of the OOD-B builder (task or repo)", not_dead)

    # prefer tasks that may grade: no known gold failure first, strict test shape first, leaderboard first, then newest
    def rank(r):
        known_fail = r["_old_gate"] is not None
        return (known_fail, not r["_tests_strict"], r["_src"] != "nebius/SWE-rebench-leaderboard",
                [-ord(c) for c in str(r["created_at"])])
    uniq.sort(key=rank)
    per, out, names = collections.Counter(), [], {}
    for r in uniq:
        name = r["repo"].split("/", 1)[1].lower()
        if names.setdefault(name, r["repo"]) != r["repo"]:
            continue  # wheel dirs are keyed by repo name; skip a second owner of the same name
        if per[r["repo"]] >= a.per_repo:
            continue
        per[r["repo"]] += 1
        out.append(r)
        if len(out) >= a.n:
            break
    funnel.append((len(out), len(per), f"selected (<= {a.per_repo} per repo; prefer no known gold failure, newest)"))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as fh:
        for r in out:
            ic = r["install_config"] or {}
            row = {
                "instance_id": short_id(r["repo"], r["instance_id"]), "source_instance_id": r["instance_id"],
                "source": r["_src"], "repo": r["repo"], "base_commit": r["base_commit"],
                "patch": r["patch"], "test_patch": r["test_patch"], "problem_statement": r["problem_statement"],
                "hints_text_original": r.get("hints_text") or "", "created_at": str(r["created_at"]),
                "FAIL_TO_PASS": list(r["FAIL_TO_PASS"] or []),
                "PASS_TO_PASS_source_count": len(r["PASS_TO_PASS"] or []),
                "install_config": {k: ic.get(k) for k in ("python", "install", "pip_packages", "pre_install",
                                                          "reqs_path", "packages", "env_vars", "test_cmd")},
                "license": (meta.get(r["repo"]) or {}).get("spdx", ""), "src_files": r["_src_files"],
                "src_lines": r["_src_lines"], "tests_in_files": r["_tests_in_files"],
                "tests_strict": r["_tests_strict"], "has_interface": bool(r.get("interface")),
                "oodb_candidate": r["_old"] is not None, "oodb_gold_failed": r["_old_gate"] is not None,
                "bug_label": r.get("_bug_label", ""),
            }
            fh.write(json.dumps(row) + "\n")
    lines = [f"{n:6d} {m:5d}  {s}" for n, m, s in funnel]
    lines.append("months: " + str(dict(sorted(collections.Counter(str(r["created_at"])[:7] for r in out).items()))))
    lines.append("sources: " + str(collections.Counter(r["_src"] for r in out).most_common()))
    lines.append("interface rows: " + str(sum(bool(r.get("interface")) for r in out)))
    lines.append("known gold failures in OOD-B: " + str(sum(r["_old_gate"] is not None for r in out)))
    lines.append("licences: " + str(collections.Counter((meta.get(r["repo"]) or {}).get("spdx") for r in out).most_common()))
    print("\n".join(lines))
    a.out.with_suffix(".funnel.txt").write_text("\n".join(lines) + "\n")
    dup = [k for k, v in collections.Counter(short_id(r["repo"], r["instance_id"]) for r in out).items() if v > 1]
    if dup:
        sys.exit(f"duplicate short ids: {dup}")


if __name__ == "__main__":
    main()
