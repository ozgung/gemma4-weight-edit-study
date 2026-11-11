"""OOD-C builder, run inside a Kaggle CPU kernel with internet (never on the Mac: it executes repository code).
Copy of scripts/oodb/build_env.py; the differences are the gate (a real agent-path pass with a stub model) and the merge
rule (keep what the agent can run; record gradability per task instead of requiring it), plus SQUASH_MB and a build deadline.

    python build_env.py build --candidates candidates.jsonl --out /kaggle/working/oodc --work /tmp/oodc --workers 3
    python build_env.py gate --ood /kaggle/working/oodc --submission-dir sub --agent-submission e4b \
        --runner run_ood45.py --stub stub_llm.py --results /kaggle/working/gate

`build`, per candidate task:
  1. clone the repository (bare, one clone per repo) and make the snapshot the way the harness README describes:
     `git fast-export` of the history up to base_commit into a fresh repository on branch `main`, checked out,
     tarred as snapshots/<instance_id>.tgz (history squashed to one commit when .git exceeds SQUASH_MB);
  2. build the task environment in a clean venv (host Python, uv): SWE-rebench install_config requirements,
     pip_packages and install command, with pytest pinned to the host's 8.4.2;
  3. turn the frozen environment into wheels (pip wheel --no-deps for every pin, plus the PEP 517 build backend
     so the sandbox's `pip install --no-build-isolation -e /workspace` works), skipping what the harness supplies
     (pytest, pluggy, pip, setuptools, wheel, pytest-timeout) and disruptive plugins; pytest-asyncio and
     pytest-httpbin are renamed `oodb_*` (the harness skips their real names, as OOD-45 does);
  4. best effort: counterparts of the binary wheels for the other interpreter (the host is Kaggle's Python 3.13,
     like the scorer's Docker image; the alternative set cp312 matches the Colab runtime).
  Writes the OOD-45 layout (tasks.jsonl, snapshots/, wheels/house, wheels/x86_64-cp3XY/manifest.json,
  tools/oodb_meta.json) and build_report.jsonl with stage, timings, sizes and the error tail of each task.

`gate` runs run_ood45.py on the built set and writes gate.json: gold on every task; a second gold and an empty patch
on the tasks gold1 resolved; a real agent pass on every task against stub_llm.py.
  usable   = the agent pass got through sandbox setup and the tool loop with no harness error (the agent can run);
  gradable = gold resolved twice and the empty patch did not (solved counts can be reported).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

SKIP = {"pytest", "pluggy", "pip", "setuptools", "wheel", "pytest_timeout", "distribute",
        # plugins that change collection, ordering or output when merely installed
        "pytest_randomly", "pytest_xdist", "pytest_sugar", "pytest_pretty", "pytest_testmon", "pytest_picked",
        "pytest_random_order", "pytest_rerunfailures"}
RENAME = {"pytest_asyncio", "pytest_httpbin"}  # in the harness skip list; shipped as oodb_<name> (OOD-45 trick)
HEAVY = re.compile(r"^(torch|torchvision|torchaudio|tensorflow|tensorflow_cpu|tensorflow_intel|jax|jaxlib|triton|"
                   r"nvidia_.*|tf_keras|paddlepaddle|mxnet|onnxruntime_gpu|cupy.*)$")
SQUASH_MB = 150  # OOD-B: 300. A kernel output holds about 20 GB; snapshots were over half of OOD-B's 9.2 GB
MAX_WHEELS_MB = 400
HOST_PYTEST = "8.4.2"
HOST_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"   # cp313 on Kaggle (2026-10) and in the scorer image
ALT_TAG, ALT_VER = ("cp312", "3.12") if HOST_TAG == "cp313" else ("cp313", "3.13")  # cp312: Colab runtime
LOCKS: dict[str, threading.Lock] = {}
LOCKS_GUARD = threading.Lock()
# Errors that mean the task cannot run at all; a grading error ("Required test node did not pass", ...) is not one.
HARNESS_ERR = re.compile(r"^runner:|Evaluation error|not importable|ContainerSetupError|Agent timed out during setup", re.I)


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def sh(cmd, cwd=None, env=None, timeout=1800, check=True) -> subprocess.CompletedProcess:
    """Run in its own process group and kill the whole group on timeout (a plain subprocess.run timeout kills only
    the shell, and a pipeline's children keep the pipes open, so the call never returns)."""
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, shell=isinstance(cmd, str), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True, errors="replace",
                            executable="/bin/bash" if isinstance(cmd, str) else None, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        raise
    p = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    if check and p.returncode != 0:
        raise StageError(f"exit {p.returncode}: {cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))}\n"
                         f"{(p.stdout or '')[-1500:]}\n{(p.stderr or '')[-2500:]}")
    return p


class StageError(RuntimeError):
    pass


def du_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


def repo_lock(repo: str) -> threading.Lock:
    with LOCKS_GUARD:
        return LOCKS.setdefault(repo, threading.Lock())


# ---------------------------------------------------------------- snapshot
def clone(repo: str, base: str, clones: Path) -> Path:
    src = clones / (repo.replace("/", "__") + ".git")
    with repo_lock(repo):
        if not src.exists():
            sh(["git", "clone", "--bare", "--quiet", f"https://github.com/{repo}.git", str(src)], timeout=1800,
               env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
        if sh(["git", "--git-dir", str(src), "cat-file", "-e", f"{base}^{{commit}}"], check=False).returncode:
            sh(["git", "--git-dir", str(src), "fetch", "--quiet", "origin", base], timeout=900)
    return src


def tree_listing(git_args: list[str], tree: str) -> list[str]:
    """Recursive listing of a tree without gitlinks: submodules are not part of a snapshot (fast-export drops
    them, and the harness snapshots are built the same way)."""
    out = sh(["git", *git_args, "ls-tree", "-r", "--full-tree", tree], timeout=900).stdout.splitlines()
    return [ln for ln in out if not ln.startswith("160000 ")]


def make_snapshot(src: Path, base: str, iid: str, snap: Path) -> str:
    ref = f"refs/heads/oodb_{iid}"
    sh(["git", "--git-dir", str(src), "update-ref", ref, base])
    snap.mkdir(parents=True)
    sh(["git", "init", "-q", "-b", "main", str(snap)])
    base_tree = sh(["git", "--git-dir", str(src), "rev-parse", f"{base}^{{tree}}"]).stdout.strip()
    want = tree_listing(["--git-dir", str(src)], base_tree)

    def same_tree() -> bool:
        return tree_listing(["-C", str(snap)], "HEAD") == want
    history, why = None, []
    for extra in ("", "--full-tree "):
        try:
            shutil.rmtree(snap, ignore_errors=True)
            snap.mkdir(parents=True)
            sh(["git", "init", "-q", "-b", "main", str(snap)])
            sh(["git", "-C", str(snap), "config", "gc.auto", "0"])
            sh(f"git --git-dir {src} fast-export {extra}--signed-tags=strip --tag-of-filtered-object=drop "
               f"--reencode=yes {ref} | git -C {snap} fast-import --quiet", timeout=2400)
            sh(["git", "-C", str(snap), "branch", "-m", f"oodb_{iid}", "main"])
            if not same_tree():
                raise StageError(f"fast-export {extra or 'plain'}: tree differs from base_commit")
            sh(["git", "-C", str(snap), "checkout", "-q", "-f", "main"], timeout=900)
            sh(["git", "-C", str(snap), "repack", "-adq"], timeout=1800)
            if du_bytes(snap / ".git") > SQUASH_MB * 2**20:
                raise StageError(f"history over {SQUASH_MB} MB")
            history = "full" if not extra else "full (--full-tree)"
            break
        except (StageError, subprocess.TimeoutExpired) as e:
            why.append(str(e).splitlines()[0][:100])
            if "MB" in why[-1]:
                break
    if history is None:  # one commit holding exactly the base tree (no .gitattributes export filtering)
        history = f"squashed ({'; '.join(why)})"
        shutil.rmtree(snap, ignore_errors=True)
        snap.mkdir(parents=True)
        sh(["git", "init", "-q", "-b", "main", str(snap)])
        sh(f"git --git-dir {src} archive --format=tar {base_tree} | tar -x -C {snap}", timeout=1800)
        idx = snap.parent / "base.index"
        env = {**os.environ, "GIT_INDEX_FILE": str(idx)}
        sh(["git", "--git-dir", str(src), "--work-tree", str(snap), "read-tree", base_tree], env=env)
        sh(["git", "--git-dir", str(src), "--work-tree", str(snap), "checkout-index", "-a", "-f"], env=env,
           timeout=1800)
        idx.unlink(missing_ok=True)
        sh(["git", "-C", str(snap), "add", "-A", "-f"], timeout=900)
        sh(["git", "-C", str(snap), "-c", "user.name=oodb", "-c", "user.email=oodb@local", "commit", "-q", "-m",
            f"snapshot of {base}"], timeout=900)
        if not same_tree():
            raise StageError("squashed snapshot tree differs from base_commit tree")
    head = sh(["git", "-C", str(snap), "rev-parse", "HEAD"]).stdout.strip()
    links = sh(["git", "--git-dir", str(src), "ls-tree", "-r", base_tree], timeout=900).stdout.count("160000 commit")
    return f"{history}; head {head[:12]}" + (f"; {links} submodule(s) dropped" if links else "")


# ---------------------------------------------------------------- environment
def scm_version(src: Path, base: str) -> str:
    p = sh(["git", "--git-dir", str(src), "describe", "--tags", "--abbrev=0", base], check=False, timeout=300)
    m = re.search(r"(\d+(?:\.\d+)+)", p.stdout.strip()) if p.returncode == 0 else None
    return m.group(1) if m else ""

PIP_RE = re.compile(r"(?<![\w/.-])(?:python3?\s+-m\s+)?pip3?\s+(install|uninstall)\b")


def uvify(cmd: str) -> str:
    cmd = PIP_RE.sub(lambda m: f"uv pip {m.group(1)}", cmd)
    return re.sub(r"\s--user\b", "", cmd)


def install_cmds(ic: dict, use_uv: bool) -> list[str]:
    cmds = []
    for r in ic.get("reqs_path") or []:
        cmds.append(f"pip install -r {r}")
    for p in ic.get("pip_packages") or []:
        cmds.append(f"pip install {p}")
    if ic.get("install"):
        cmds.append(ic["install"])
    return [uvify(c) if use_uv else c for c in cmds]


def as_of(created_at: str) -> str:
    """uv --exclude-newer cutoff: the day after the task was created, so dependencies resolve as they did then."""
    import datetime as dt
    d = dt.datetime.fromisoformat(created_at[:19].replace(" ", "T")) + dt.timedelta(days=1)
    return d.strftime("%Y-%m-%dT00:00:00Z")


def build_env(task: dict, srcdir: Path, venv: Path, work: Path, use_uv: bool, pin_pytest: bool,
              exclude_newer: bool = True, scm_version: str = "") -> dict:
    ic = task["install_config"]
    shutil.rmtree(venv, ignore_errors=True)
    sh(["uv", "venv", "--seed", "-q", "-p", sys.executable, str(venv)], timeout=600)
    env = {**os.environ, "VIRTUAL_ENV": str(venv), "PATH": f"{venv}/bin:{os.environ['PATH']}", "PIP_NO_INPUT": "1",
           "UV_HTTP_TIMEOUT": "120", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "UV_PYTHON": str(venv / "bin/python")}
    if exclude_newer and use_uv:
        env["UV_EXCLUDE_NEWER"] = as_of(task["created_at"])
    if scm_version:  # the snapshot has no tags; setuptools-scm / hatch-vcs would report 0.1.dev and break pins
        env["SETUPTOOLS_SCM_PRETEND_VERSION"] = scm_version
    ev = ic.get("env_vars")
    for k, v in (ev.items() if isinstance(ev, dict) else []):
        env[str(k)] = str(v)
    if pin_pytest:
        cons = work / "constraints.txt"
        cons.write_text(f"pytest=={HOST_PYTEST}\n")
        env["UV_CONSTRAINT"] = env["PIP_CONSTRAINT"] = str(cons)
    log = []
    for c in install_cmds(ic, use_uv):
        p = sh(c, cwd=srcdir, env=env, timeout=1800, check=False)
        log.append({"cmd": c, "exit": p.returncode, "tail": ((p.stdout or "") + (p.stderr or ""))[-1200:]})
        if p.returncode:
            raise StageError(f"install failed: {c}\n{log[-1]['tail']}")
    return {"env": env, "log": log}


def frozen(venv: Path, env: dict) -> tuple[list[str], list[str]]:
    py = str(venv / "bin/python")
    lines = sh(["uv", "pip", "freeze", "--python", py], env=env).stdout.splitlines()
    edit = json.loads(sh(["uv", "pip", "list", "--python", py, "--editable", "--format", "json"], env=env).stdout
                      or "[]")
    editable = [norm(e["name"]) for e in edit]
    pins = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith(("#", "-e")):
            continue
        name = norm(re.split(r"[=<>@ \[;]", ln, 1)[0])
        if name in editable:
            continue
        pins.append(ln)
    return pins, editable


def build_requires(srcdir: Path) -> list[str]:
    pp = srcdir / "pyproject.toml"
    if not pp.exists():
        return []
    try:
        req = tomllib.loads(pp.read_text()).get("build-system", {}).get("requires", [])
    except Exception:
        return []
    return [r for r in req if norm(re.split(r"[=<>!~ \[;]", r.strip(), 1)[0]) not in {"setuptools", "wheel"}]


EDITABLE_TOP = r'''
import importlib.metadata as md, json
out = []
for d in md.distributions():
    durl = (d.read_text("direct_url.json") or "").replace(" ", "")
    if '"editable":true' not in durl:
        continue
    top = (d.read_text("top_level.txt") or "").split()
    for f in d.files or []:
        s = str(f)
        if s.endswith(".pth") or "__editable__" in s or s.endswith(".dist-info/RECORD"):
            continue
        part = s.split("/")[0]
        if part and not part.endswith((".dist-info", ".data")) and part != "..":
            top.append(part[:-3] if part.endswith(".py") else part)
    out.append({"name": d.metadata["Name"], "top": sorted(set(t for t in top if not t.startswith("_")))})
print(json.dumps(out))
'''


def importable(venv: Path, name: str, srcdir: Path, env: dict) -> bool:
    """Importable from /tmp (installed) or from the repository root (how the sandbox probe and pytest see it)."""
    for cwd in ("/tmp", str(srcdir)):
        if sh([str(venv / "bin/python"), "-c", f"import {name}"], cwd=cwd, env=env, check=False,
              timeout=300).returncode == 0:
            return True
    return False


def import_name(srcdir: Path, venv: Path, editable: list[str], env: dict) -> str:
    bad = {"tests", "test", "docs", "doc", "examples", "example", "benchmarks", "scripts", "tools", "dist",
           "ci", "conftest", "setup", "noxfile", "tasks", "fabfile", "locustfile", "site", "bin"}
    first = []
    p = sh([str(venv / "bin/python"), "-c", EDITABLE_TOP], cwd="/tmp", env=env, check=False, timeout=300)
    try:
        for d in json.loads(p.stdout.strip().splitlines()[-1]):
            first += [t for t in d["top"] if t.lower() not in bad]
    except (ValueError, IndexError):
        pass
    cands = []
    for root in (srcdir / "src", srcdir):
        if not root.is_dir():
            continue
        for p in sorted(root.iterdir()):
            if p.name.startswith((".", "_")) or (p.name.lower() in bad and norm(p.name) not in editable):
                continue
            if p.is_dir() and ((p / "__init__.py").exists() or norm(p.name) in editable):
                cands.append(p.name)
            elif p.suffix == ".py" and p.stem.lower() not in bad:
                cands.append(p.stem)
    want = list(editable)
    order = [c for c in cands if norm(c) in want]
    order += [c for c in cands if any(w.startswith(norm(c)) or norm(c).startswith(w) for w in want)]
    order += first
    order += [c for c in cands if (srcdir / c).is_dir() or (srcdir / "src" / c).is_dir()]
    seen = []
    for c in order:
        if c in seen or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", c):
            continue
        seen.append(c)
        if importable(venv, c, srcdir, env):
            return c
    raise StageError(f"no importable package (tried {seen[:8]}; editable {editable})")


def collect_wheels(pins: list[str], breq: list[str], wdir: Path, env: dict) -> dict:
    wdir.mkdir(parents=True, exist_ok=True)
    keep = [p for p in pins if norm(re.split(r"[=<>@ \[;]", p, 1)[0]) not in SKIP]
    heavy = [p for p in keep if HEAVY.match(norm(re.split(r"[=<>@ \[;]", p, 1)[0]))]
    if heavy:
        raise StageError(f"heavy dependencies: {heavy[:6]}")
    reqf = wdir.parent / "pins.txt"
    reqf.write_text("\n".join(keep) + "\n")
    hostenv = {k: v for k, v in env.items() if k not in ("VIRTUAL_ENV", "UV_CONSTRAINT", "PIP_CONSTRAINT")}
    hostenv["PATH"] = os.environ["PATH"]
    missing = []
    p = sh([sys.executable, "-m", "pip", "wheel", "-q", "--no-deps", "--prefer-binary", "-w", str(wdir), "-r",
            str(reqf)], env=hostenv, timeout=3600, check=False)
    if p.returncode:
        for pin in keep:
            q = sh([sys.executable, "-m", "pip", "wheel", "-q", "--no-deps", "--prefer-binary", "-w", str(wdir), pin],
                   env=hostenv, timeout=1200, check=False)
            if q.returncode:
                missing.append(pin)
    if breq:
        cons = wdir.parent / "pins_constraints.txt"
        cons.write_text("\n".join(x for x in keep if " @ " not in x) + "\n")
        q = sh([sys.executable, "-m", "pip", "wheel", "-q", "--prefer-binary", "-w", str(wdir), "-c", str(cons),
                *breq], env=hostenv, timeout=1800, check=False)
        if q.returncode:
            missing += [f"build-backend: {b}" for b in breq]
    for w in list(wdir.glob("*.whl")):
        n = norm(w.name.split("-", 1)[0])
        if n in SKIP:
            w.unlink()
        elif n in RENAME:
            w.rename(w.with_name("oodb_" + w.name))
    return {"missing": missing, "pins": len(keep)}


ABI_RE = re.compile(r"-(cp3\d+)-(cp3\d+|abi3|none)-")


def alt_wheels(wdir: Path, w313: Path, env: dict) -> list[str]:
    """ALT_TAG builds of the version-specific wheels; pure and abi3 wheels serve both interpreters."""
    w313.mkdir(parents=True, exist_ok=True)
    plats = ["manylinux2014_x86_64", "manylinux_2_17_x86_64", "manylinux1_x86_64", "manylinux2010_x86_64"] + \
        [f"manylinux_2_{i}_x86_64" for i in range(5, 40)] + ["linux_x86_64"]
    missing = []
    hostenv = {k: v for k, v in env.items() if k not in ("VIRTUAL_ENV", "UV_CONSTRAINT", "PIP_CONSTRAINT")}
    hostenv["PATH"] = os.environ["PATH"]
    for w in sorted(wdir.glob("*.whl")):
        m = ABI_RE.search(w.name)
        if not m or m.group(2) in ("abi3", "none"):
            continue
        name, ver = w.name.split("-")[:2]
        name = name.removeprefix("oodb_")
        args = [sys.executable, "-m", "pip", "download", "-q", "--no-deps", "--only-binary=:all:", "--python-version",
                ALT_VER, "--implementation", "cp", "--abi", ALT_TAG, "-d", str(w313)]
        for pl in plats:
            args += ["--platform", pl]
        if sh(args + [f"{name}=={ver}"], env=hostenv, timeout=600, check=False).returncode:
            missing.append(f"{name}=={ver}")
    return missing


# ---------------------------------------------------------------- per task
def build_one(task: dict, work_root: Path, clones: Path, out: Path, deadline: float) -> dict:
    iid = task["instance_id"]
    rec = {"instance_id": iid, "repo": task["repo"], "status": "fail", "stage": "clone", "t": {}}
    if time.time() > deadline:
        rec.update(stage="deadline", error="StageError: build deadline passed before this task started")
        rec["t"]["total_s"] = 0.0
        return rec
    w = work_root / iid
    shutil.rmtree(w, ignore_errors=True)
    w.mkdir(parents=True)
    t = time.time()
    try:
        src = clone(task["repo"], task["base_commit"], clones)
        rec["t"]["clone_s"] = round(time.time() - t, 1)
        rec["stage"], t = "snapshot", time.time()
        snap = w / "snap"
        rec["history"] = make_snapshot(src, task["base_commit"], iid, snap)
        if not any((snap / f).exists() for f in ("pyproject.toml", "setup.py", "setup.cfg")):
            raise StageError("no Python project at the repository root")
        tgz = w / f"{iid}.tgz"
        sh(f"tar -czf {tgz} -C {snap} .", timeout=1800)
        rec["snapshot_bytes"] = tgz.stat().st_size
        rec["t"]["snapshot_s"] = round(time.time() - t, 1)
        rec["stage"], t = "env", time.time()
        srcdir = w / "src"
        shutil.copytree(snap, srcdir, symlinks=True)
        venv = w / "venv"
        attempts = []
        res = None
        scm = scm_version(src, task["base_commit"])
        rec["scm_version"] = scm
        for use_uv, pin, newer in ((True, True, True), (True, False, True), (True, True, False), (False, False, False)):
            try:
                res = build_env(task, srcdir, venv, w, use_uv, pin, newer, scm)
                rec["env_mode"] = (f"{'uv' if use_uv else 'pip'}{'+pytest' + HOST_PYTEST if pin else ''}"
                                   f"{'+as-of ' + as_of(task['created_at'])[:10] if newer and use_uv else ''}")
                break
            except (StageError, subprocess.TimeoutExpired) as e:
                attempts.append(str(e)[-800:])
                shutil.rmtree(srcdir, ignore_errors=True)
                shutil.copytree(snap, srcdir, symlinks=True)
        if res is None:
            raise StageError("all install attempts failed:\n" + "\n---\n".join(attempts))
        rec["t"]["env_s"] = round(time.time() - t, 1)
        rec["stage"], t = "wheels", time.time()
        pins, editable = frozen(venv, res["env"])
        rec["editable"] = editable
        rec["pytest_pin"] = next((p for p in pins if norm(p.split("=")[0]) == "pytest"), "")
        rec["import_name"] = import_name(srcdir, venv, editable, res["env"])
        breq = build_requires(srcdir)
        rec["build_requires"] = breq
        wdir = w / "wheels"
        rec.update(collect_wheels(pins, breq, wdir, res["env"]))
        rec["wheels_n"] = len(list(wdir.glob("*.whl")))
        rec["wheels_bytes"] = du_bytes(wdir)
        if rec["wheels_bytes"] > MAX_WHEELS_MB * 2**20:
            raise StageError(f"wheel set {rec['wheels_bytes'] / 2**20:.0f} MB > {MAX_WHEELS_MB} MB")
        hard_missing = [m for m in rec["missing"] if not m.startswith("build-backend")]
        if hard_missing:
            raise StageError(f"no wheel for {hard_missing[:8]}")
        rec["t"]["wheels_s"] = round(time.time() - t, 1)
        t = time.time()
        rec["alt_tag"] = ALT_TAG
        rec["alt_missing"] = alt_wheels(wdir, w / "wheels_alt", res["env"])
        rec["t"]["alt_s"] = round(time.time() - t, 1)
        rec["env_overlay"] = sorted(f"plugin:{w_.name.split('-')[0].removeprefix('oodb_').replace('_', '-')}"
                                    for w_ in wdir.glob("oodb_*.whl"))
        # publish into the shared layout
        rec["stage"] = "publish"
        (out / "snapshots").mkdir(parents=True, exist_ok=True)
        shutil.move(str(tgz), out / "snapshots" / f"{iid}.tgz")
        rec["status"], rec["stage"] = "ok", "done"
    except Exception as e:  # keep going; record why
        rec["error"] = f"{type(e).__name__}: {e}"[-3000:]
    finally:
        rec["t"]["total_s"] = round(sum(v for k, v in rec["t"].items() if k != "total_s"), 1)
    return rec


def assemble(tasks: list[dict], recs: list[dict], work_root: Path, out: Path) -> None:
    house = out / "wheels" / "house"
    house.mkdir(parents=True, exist_ok=True)
    ok = {r["instance_id"]: r for r in recs if r["status"] == "ok"}
    manifests = {f"x86_64-{HOST_TAG}": {}, f"x86_64-{ALT_TAG}": {}}
    meta = {"import_names": {}, "project_subdirs": {}}
    rows = []
    for t in tasks:
        r = ok.get(t["instance_id"])
        if not r:
            continue
        repo_dir = t["repo"].split("/", 1)[1]
        w = work_root / t["instance_id"]
        names_host = []
        for whl in sorted((w / "wheels").glob("*.whl")):
            if not (house / whl.name).exists():
                shutil.copy2(whl, house / whl.name)
            names_host.append(whl.name)
        names_alt = [n for n in names_host
                     if not (ABI_RE.search(n) and ABI_RE.search(n).group(2) not in ("abi3", "none"))]
        for whl in sorted((w / "wheels_alt").glob("*.whl")):
            if norm(whl.name.split("-", 1)[0]) in RENAME:
                dst = "oodb_" + whl.name
            else:
                dst = whl.name
            if not (house / dst).exists():
                shutil.copy2(whl, house / dst)
            names_alt.append(dst)
        for key, names in ((f"x86_64-{HOST_TAG}", names_host), (f"x86_64-{ALT_TAG}", names_alt)):
            cur = manifests[key].setdefault(repo_dir, [])
            cur.extend(n for n in names if n not in cur)
        meta["import_names"][t["repo"]] = r["import_name"]
        rows.append({
            "instance_id": t["instance_id"], "repo": t["repo"], "base_commit": t["base_commit"], "patch": t["patch"],
            "test_patch": t["test_patch"], "hints_text": "", "created_at": t["created_at"],
            "FAIL_TO_PASS": t["FAIL_TO_PASS"], "PASS_TO_PASS": [], "license": t.get("license", ""),
            "source": t["source"], "source_instance_id": t["source_instance_id"],
            "problem_statement_source": f"issue ({t['source']} problem_statement)",
            "env_overlay": r.get("env_overlay", []), "excluded": "", "validated_in": "",
            "problem_statement": t["problem_statement"], "problem_statement_original": t["problem_statement"],
            "problem_statement_rewritten": False, "hints_text_original": t.get("hints_text_original", ""),
            "src_files": t.get("src_files"), "src_lines": t.get("src_lines"),
            "install_python": t["install_config"].get("python"), "tests_strict": t.get("tests_strict"),
            "has_interface": t.get("has_interface"), "oodb_candidate": t.get("oodb_candidate"),
            "bug_label": t.get("bug_label", ""),
        })
    for key, man in manifests.items():
        (out / "wheels" / key).mkdir(parents=True, exist_ok=True)
        (out / "wheels" / key / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    (out / "tools").mkdir(parents=True, exist_ok=True)
    (out / "tools" / "oodb_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    with (out / "tasks.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def docker_sandbox_extras(house: Path) -> None:
    """The exact wheels ood45_harness.DOCKER_SANDBOX_WHEELS expects in wheels/house."""
    for spec in ("pytest-timeout==2.1.0", "editables==0.5", "pdm-backend==2.5.0", "hatchling==1.32.4",
                 "pathspec==1.1.1", "trove-classifiers==2026.9.21.13"):
        sh([sys.executable, "-m", "pip", "download", "-q", "--no-deps", "-d", str(house), spec], check=False,
           timeout=600)


def cmd_build(a) -> None:
    tasks = [json.loads(l) for l in a.candidates.read_text().splitlines() if l.strip()]
    if a.limit:
        tasks = tasks[: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    clones = a.work / "_clones"
    clones.mkdir(parents=True, exist_ok=True)
    sh([sys.executable, "-m", "pip", "install", "-q", "uv"], timeout=900)
    recs = []
    rep = a.out.parent / "build_report.jsonl"
    with cf.ThreadPoolExecutor(a.workers) as ex:
        deadline = time.time() + a.deadline_hours * 3600
        futs = {ex.submit(build_one, t, a.work, clones, a.out, deadline): t for t in tasks}
        for f in cf.as_completed(futs):
            r = f.result()
            recs.append(r)
            with rep.open("a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(f"[build {len(recs)}/{len(tasks)}] {r['instance_id']} {r['status']} stage={r['stage']} "
                  f"t={r['t'].get('total_s')}s {(r.get('error') or '')[:160]!r}", flush=True)
            # free per-task scratch except wheels (assemble needs them)
            for sub in ("venv", "src", "snap"):
                shutil.rmtree(a.work / r["instance_id"] / sub, ignore_errors=True)
    assemble(tasks, recs, a.work, a.out)
    docker_sandbox_extras(a.out / "wheels" / "house")
    print("BUILD DONE", sum(r["status"] == "ok" for r in recs), "of", len(recs), flush=True)


def run_pass(a, name: str, flags: list[str], tasks: list[str] | None, submission: Path) -> dict:
    rd = a.results / name
    t = time.time()
    cmd = [a.python, str(a.runner), "--ood", str(a.ood), "--submission-dir", str(submission), "--results-dir", str(rd),
           "--concurrency", str(a.concurrency), "--wheels-tmp", f"/tmp/oodc_wheels_{a.results.name}_{name}", *flags]
    if tasks is not None:
        cmd += ["--tasks", ",".join(tasks)]
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    (a.results / f"{name}.log").write_text((p.stdout or "") + "\n" + (p.stderr or ""))
    rows = [json.loads(l) for l in (rd / "task_results.jsonl").read_text().splitlines()] \
        if (rd / "task_results.jsonl").exists() else []
    print(f"[gate] {name}: exit {p.returncode}, resolved {sum(r['resolved'] for r in rows)}/{len(rows)}, "
          f"{time.time() - t:.0f}s; stderr tail {(p.stderr or '')[-300:]!r}", flush=True)
    return {r["instance_id"]: r for r in rows}


def agent_clean(row: dict) -> bool:
    """The real agent path ran: setup passed, the model loop ran, tools executed, no harness error."""
    return bool(row) and (row.get("tool_calls") or 0) >= 2 and not HARNESS_ERR.search(str(row.get("error") or ""))


def cmd_gate(a) -> None:
    passes = a.passes.split(",")
    res: dict[str, dict] = {}
    a.results.mkdir(parents=True, exist_ok=True)
    if "gold1" in passes:
        res["gold1"] = run_pass(a, "gold1", ["--gold"], None, a.submission_dir)
    good = sorted(i for i, r in res.get("gold1", {}).items() if r["resolved"])
    if "gold2" in passes and good:
        res["gold2"] = run_pass(a, "gold2", ["--gold"], good, a.submission_dir)
    if "empty" in passes and good:
        res["empty"] = run_pass(a, "empty", ["--baseline"], good, a.submission_dir)
    if "agent" in passes:
        stub = subprocess.Popen([sys.executable, str(a.stub), "--port", "8123"])
        time.sleep(2)
        try:
            res["agent"] = run_pass(a, "agent", ["--api-base", "http://127.0.0.1:8123/v1"], None, a.agent_submission)
        finally:
            stub.terminate()
    ids = sorted(set().union(*[set(v) for v in res.values()])) if res else []
    gate = {}
    for i in ids:
        g1, g2, e, ag = (res.get(n, {}).get(i, {}) for n in ("gold1", "gold2", "empty", "agent"))
        gradable = bool(g1.get("resolved")) and ("gold2" not in passes or bool(g2.get("resolved"))) \
            and ("empty" not in passes or (e != {} and not e.get("resolved")))
        gate[i] = {"usable": agent_clean(ag) if "agent" in passes else None, "gradable": gradable,
                   "gold1": g1.get("resolved"), "gold2": g2.get("resolved"), "empty": e.get("resolved"),
                   "gold1_s": g1.get("duration_seconds"), "agent_s": ag.get("duration_seconds"),
                   "agent_tool_calls": ag.get("tool_calls"), "agent_llm_calls": ag.get("total_llm_calls"),
                   "agent_resolved": ag.get("resolved"),
                   "gold_error": (g1.get("error") or g2.get("error") or "")[:400],
                   "empty_error": (e.get("error") or "")[:200], "agent_error": (ag.get("error") or "")[:400]}
    (a.results / "gate.json").write_text(json.dumps(gate, indent=1))
    print("GATE DONE usable", sum(bool(v["usable"]) for v in gate.values()), "gradable",
          sum(v["gradable"] for v in gate.values()), "of", len(gate), flush=True)


def cmd_merge(a) -> None:
    """Merge shard outputs, keeping the tasks the agent can run (gate `usable`) and recording `gradable` per task.
    Wheels are copied only if a kept repo's manifest names them."""
    out = a.out
    shutil.rmtree(out, ignore_errors=True)
    (out / "snapshots").mkdir(parents=True)
    (out / "wheels" / "house").mkdir(parents=True)
    (out / "tools").mkdir(parents=True)
    drop = set()
    if a.drop_gate and a.drop_gate.exists():
        drop = {i for i, v in json.loads(a.drop_gate.read_text()).items() if not v["usable"]}
    rows, meta, mans, reports, gates = [], {"import_names": {}, "project_subdirs": {}}, {}, [], {}
    # first pass: what could be kept, and what each task and repository costs
    items, shard_mans, snapshot_size, house = [], {}, {}, {}
    for shard in a.shards:
        ood = shard / "oodc"
        gate = json.loads((shard / "gate" / "gate.json").read_text()) if (shard / "gate" / "gate.json").exists() else {}
        gates.update(gate)
        if (shard / "build_report.jsonl").exists():
            reports += (shard / "build_report.jsonl").read_text().splitlines()
        if not (ood / "tasks.jsonl").exists():
            continue
        shard_mans[ood] = {man.parent.name: json.loads(man.read_text()) for man in (ood / "wheels").glob("*/manifest.json")}
        for line in (ood / "tasks.jsonl").read_text().splitlines():
            r = json.loads(line)
            g = gate.get(r["instance_id"], {})
            if not g.get("usable") or r["instance_id"] in drop:
                continue
            r["gradable"] = bool(g.get("gradable"))
            r["gold_gate"] = "pass" if g.get("gradable") else "fail" if g.get("gold1") is not None else "unknown"
            r["validated_in"] = ("kaggle-cpu subprocess sandbox: agent path with a stub model ran; "
                                 f"gold {int(bool(g.get('gold1'))) + int(bool(g.get('gold2')))}/2, "
                                 f"empty {'resolved' if g.get('empty') else 'not resolved' if g.get('empty') is False else 'not run'}")
            snapshot_size[r["instance_id"]] = (ood / "snapshots" / f"{r['instance_id']}.tgz").stat().st_size
            items.append((ood, r))

    def repo_wheels(ood: Path, repo_dir: str) -> set[str]:
        return {n for man in shard_mans[ood].values() for n in man.get(repo_dir, [])}

    def wheel_bytes(ood: Path, names: set[str]) -> int:
        return sum((ood / "wheels" / "house" / n).stat().st_size for n in names)
    # gradable tasks first, then small ones, until the size cap (a kernel output holds about 20 GB)
    items.sort(key=lambda x: (not x[1]["gradable"], snapshot_size[x[1]["instance_id"]]
                              + wheel_bytes(x[0], repo_wheels(x[0], x[1]["repo"].split("/", 1)[1]))))
    used, chosen, seen_wheels, over = 0, [], set(), []
    for ood, r in items:
        names = repo_wheels(ood, r["repo"].split("/", 1)[1]) - seen_wheels
        cost = snapshot_size[r["instance_id"]] + wheel_bytes(ood, names)
        if used + cost > a.max_gb * 2**30:
            over.append(r["instance_id"])
            continue
        used += cost
        seen_wheels |= names
        chosen.append((ood, r))
    print(f"MERGE size: {used / 2**30:.1f} GB for {len(chosen)} tasks; over the {a.max_gb} GB cap: {len(over)} {over}",
          flush=True)
    # second pass: copy
    for ood in shard_mans:
        here = [r for o, r in chosen if o == ood]
        if not here:
            continue
        m = json.loads((ood / "tools" / "oodb_meta.json").read_text())
        kept_repos = {r["repo"] for r in here}
        for r in here:
            rows.append(r)
            shutil.copy2(ood / "snapshots" / f"{r['instance_id']}.tgz", out / "snapshots")
        for repo in kept_repos:
            meta["import_names"][repo] = m["import_names"][repo]
        for key, man in shard_mans[ood].items():
            dst = mans.setdefault(key, {})
            for repo_dir, names in man.items():
                if not any(r.split("/", 1)[1] == repo_dir for r in kept_repos):
                    continue
                dst[repo_dir] = names
                for n in names:
                    if not (out / "wheels" / "house" / n).exists():
                        shutil.copy2(ood / "wheels" / "house" / n, out / "wheels" / "house" / n)
        for extra in ("pytest_timeout-2.1.0-py3-none-any.whl", "editables-0.5-py3-none-any.whl",
                      "pdm_backend-2.5.0-py3-none-any.whl", "hatchling-1.32.4-py3-none-any.whl",
                      "pathspec-1.1.1-py3-none-any.whl", "trove_classifiers-2026.9.21.13-py3-none-any.whl"):
            if (ood / "wheels" / "house" / extra).exists() and not (out / "wheels" / "house" / extra).exists():
                shutil.copy2(ood / "wheels" / "house" / extra, out / "wheels" / "house" / extra)
        for f in ("ood45_harness.py", "run_ood45.py"):
            if (ood / "tools" / f).exists():
                shutil.copy2(ood / "tools" / f, out / "tools" / f)
    rows.sort(key=lambda r: r["instance_id"])
    with (out / "tasks.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    for key, man in mans.items():
        (out / "wheels" / key).mkdir(parents=True, exist_ok=True)
        (out / "wheels" / key / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    (out / "tools" / "oodb_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    (out / "validation").mkdir()
    (out / "validation" / "over_size_cap.json").write_text(json.dumps(over, indent=0))
    (out / "validation" / "build_report.jsonl").write_text("\n".join(reports) + "\n")
    (out / "validation" / "shard_gate.json").write_text(json.dumps(gates, indent=1))
    print("MERGE DONE", len(rows), "tasks,", len({r["repo"] for r in rows}), "repos", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--candidates", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--work", type=Path, default=Path("/tmp/oodc"))
    b.add_argument("--workers", type=int, default=3)
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--deadline-hours", type=float, default=6.0, help="no new task starts after this")
    g = sub.add_parser("gate")
    g.add_argument("--ood", type=Path, required=True)
    g.add_argument("--submission-dir", type=Path, required=True)
    g.add_argument("--runner", type=Path, required=True)
    g.add_argument("--results", type=Path, required=True)
    g.add_argument("--concurrency", type=int, default=3)
    g.add_argument("--agent-submission", type=Path, required=True)
    g.add_argument("--stub", type=Path, required=True)
    g.add_argument("--python", default=sys.executable, help="interpreter of the run (the wheel set follows from it)")
    g.add_argument("--passes", default="gold1,gold2,empty,agent")
    m = sub.add_parser("merge")
    m.add_argument("--shards", type=Path, nargs="+", required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--drop-gate", type=Path)
    m.add_argument("--max-gb", type=float, default=17.0, help="size cap of the merged layout")
    a = ap.parse_args()
    {"build": cmd_build, "gate": cmd_gate, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
