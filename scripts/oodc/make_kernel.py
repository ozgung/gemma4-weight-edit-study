"""Make the Kaggle CPU kernels (internet on, no GPU, no TPU) that build and check OOD-C.

    python3 -I scripts/oodc/make_kernel.py --candidates data/ext/oodc/candidates.jsonl --shard 0 --nshards 4 --name build-s0
    python3 -I scripts/oodc/make_kernel.py --candidates data/ext/oodc/candidates.jsonl --name assemble \
        --assemble g4-oodc-build-s0,g4-oodc-build-s1,...       # merge the shards, check the merged layout on Python 3.13
    python3 -I scripts/oodc/make_kernel.py --candidates data/ext/oodc/candidates.jsonl --name check312 \
        --check312 g4-oodc-assemble                            # the same check on Python 3.12, Colab's interpreter
    kaggle kernels push -p kernels/oodc_build-s0               # never pass --timeout

The kernel is `ozgung/g4-oodc-<name>`. Shards are by repository (all tasks of a repo in one shard), so each shard's wheel
manifest is complete. Payload (embedded, base64 tar.gz): scripts/oodc/build_env.py, scripts/oodc/stub_llm.py,
scripts/oodb/ood45_harness.py (unchanged), scripts/colab/run_ood45.py (unchanged), submissions/v001_gemma_sampling (only its
model name and budgets are read in --gold/--baseline mode), submissions/e001tt_think_t06_e4b (the agent pass, against the stub
model), the shard's candidates. Output in /kaggle/working: oodc/ (OOD-45 layout), build_report.jsonl, gate/ (gate.json and one
directory per pass), kernel.log. Repository code runs only inside these kernels.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import tarfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
USER = "ozgung"
AGENT_SUB = "submissions/e001tt_think_t06_e4b"
HOST_PINS = ["pytest==8.4.2", "litellm==1.101.6", "mcp>=1.24,<2", "starlette>=0.49.1,<1", "docker", "pyyaml", "packaging"]

COMMON = r'''
import base64, glob, io, json, os, platform, shutil, subprocess, sys, tarfile, time
from pathlib import Path
T0 = time.time()
WORK = Path("/kaggle/working"); WORK.mkdir(parents=True, exist_ok=True)
LOG = open(WORK / "kernel.log", "a")
def log(*a):
    msg = " ".join(str(x) for x in a); print(msg, flush=True); LOG.write(msg + "\n"); LOG.flush()
def run(cmd, **kw):
    log("$", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)))
    p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, errors="replace", **kw)
    log((p.stdout or "")[-3000:], (p.stderr or "")[-3000:], "exit", p.returncode)
    return p
def harness_wheels():
    whs = []
    for pat in ("swegemma-*.whl", "adk_eval_core-*.whl", "adk_submission-*.whl", "google_adk-*.whl", "google_genai-*.whl"):
        whs += sorted(glob.glob(f"/kaggle/input/**/gemma-4-developer-agent-wheelhouse/{pat}", recursive=True))[:1]
    log("harness wheels", whs)
    return whs
data = base64.b64decode(PAYLOAD_B64)
assert __import__("hashlib").sha256(data).hexdigest() == SHA256
PAY = Path("/tmp/oodc_payload"); shutil.rmtree(PAY, ignore_errors=True); PAY.mkdir(parents=True)
tarfile.open(fileobj=io.BytesIO(data)).extractall(PAY)
log("python", sys.version, sys.executable, platform.machine(), "cpus", os.cpu_count())
run("df -h /kaggle/working /tmp | tail -2; free -g | head -2; git --version")
run("git config --global user.email oodb@local && git config --global user.name oodb && git config --global init.defaultBranch main && git config --global gc.auto 0 && git config --global gc.autoDetach false")
GATE = [sys.executable, str(PAY / "build_env.py"), "gate", "--submission-dir", str(PAY / "submission"),
        "--agent-submission", str(PAY / "e4b"), "--runner", str(PAY / "run_ood45.py"), "--stub", str(PAY / "stub_llm.py"),
        "--concurrency", str(GATE_CONC)]
'''

BODY = COMMON + r'''
# 1. build
t = time.time()
p = subprocess.run([sys.executable, str(PAY / "build_env.py"), "build", "--candidates", str(PAY / "candidates.jsonl"),
                    "--out", str(WORK / "oodc"), "--work", "/tmp/oodc", "--workers", str(WORKERS),
                    "--deadline-hours", str(DEADLINE_H)] + (["--limit", str(LIMIT)] if LIMIT else []), text=True)
log("build exit", p.returncode, "in", round(time.time() - t), "s")
for f in ("ood45_harness.py", "run_ood45.py"):
    shutil.copy2(PAY / f, WORK / "oodc" / "tools" / f)
run(f"du -sh {WORK}/oodc {WORK}/oodc/snapshots {WORK}/oodc/wheels/house; ls {WORK}/oodc/snapshots | wc -l")

# 2. harness for the gate: the competition wheels plus what run_ood45.py needs on the host (pins as in setup_vm.py)
run([sys.executable, "-m", "pip", "install", "-q", *harness_wheels(), *HOST_PINS])
run([sys.executable, "-c", "import swegemma, google.adk, litellm, pytest; from importlib.metadata import version; "
     "print(version('swegemma'), version('google-adk'), version('litellm'), pytest.__version__)"])

# 3. gate: gold on all, gold again and empty on gold passers, the real agent path against the stub model on all
if (WORK / "oodc" / "tasks.jsonl").exists() and (WORK / "oodc" / "tasks.jsonl").read_text().strip():
    t = time.time()
    p = subprocess.run(GATE + ["--ood", str(WORK / "oodc"), "--results", str(WORK / "gate")], text=True)
    log("gate exit", p.returncode, "in", round(time.time() - t), "s")
log("KERNEL DONE in", round(time.time() - T0), "s")
'''

ASSEMBLE_BODY = COMMON + r'''
def disk(tag):
    run("df -h /kaggle/working /tmp | tail -2; du -sh /kaggle/working")
shards = sorted({Path(p).parent.parent for p in glob.glob("/kaggle/input/**/oodc/tasks.jsonl", recursive=True)})
log("shards", shards)
B = [sys.executable, str(PAY / "build_env.py")]
OUT = WORK / "oodc_set"
# 1. the final layout, written once: tasks the agent can run on their shard
subprocess.run(B + ["merge", "--shards", *map(str, shards), "--out", str(OUT)], check=True)
for f in ("ood45_harness.py", "run_ood45.py", "build_env.py", "stub_llm.py"):
    shutil.copy2(PAY / f, OUT / "tools" / f)
shutil.copy2(PAY / "README.md", OUT / "README.md")
disk("after merge")
# 2. check the final layout itself: one gold pass and one agent pass
run([sys.executable, "-m", "pip", "install", "-q", *harness_wheels(), *HOST_PINS])
subprocess.run(GATE + ["--ood", str(OUT), "--results", "/tmp/final_gate", "--passes", "gold1,agent"], text=True)
fg = json.loads(Path("/tmp/final_gate/gate.json").read_text())
shutil.copy2("/tmp/final_gate/gate.json", OUT / "validation" / "final_gate_cp313.json")
for f in ("gold1.log", "agent.log"):
    if Path("/tmp/final_gate", f).exists():
        shutil.copy2(Path("/tmp/final_gate", f), WORK / f"final_gate_{f}")
disk("after gate")
# 3. record, and drop any task the agent path cannot run on the merged layout (snapshot too)
rows = [json.loads(l) for l in OUT.joinpath("tasks.jsonl").read_text().splitlines() if l]
keep = [r for r in rows if fg.get(r["instance_id"], {}).get("usable")]
for r in rows:
    if r not in keep:
        (OUT / "snapshots" / f"{r['instance_id']}.tgz").unlink(missing_ok=True)
for r in keep:
    r["final_gold_cp313"] = bool(fg[r["instance_id"]].get("gold1"))
OUT.joinpath("tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
log("dropped by the final check:", sorted(r["instance_id"] for r in rows if r not in keep))
shutil.rmtree(OUT / "tools" / "__pycache__", ignore_errors=True)
disk("final")
log("ASSEMBLE DONE", len(keep), "tasks in", len({r["repo"] for r in keep}), "repos;", round(time.time() - T0), "s")
'''

CHECK312_BODY = COMMON + r'''
OOD = Path(glob.glob("/kaggle/input/**/oodc_set/tasks.jsonl", recursive=True)[0]).parent
log("layout", OOD)
run([sys.executable, "-m", "pip", "install", "-q", "uv"])
run("uv venv -p 3.12 --seed /tmp/venv312")
PY = "/tmp/venv312/bin/python"
run(["uv", "pip", "install", "--python", PY, *harness_wheels(), *HOST_PINS])
run([PY, "-c", "import sys, swegemma, google.adk, litellm, pytest; print(sys.version, pytest.__version__)"])
p = subprocess.run(GATE + ["--ood", str(OOD), "--results", str(WORK / "gate312"), "--python", PY,
                           "--passes", "gold1,agent"], text=True)
log("gate exit", p.returncode)
log("KERNEL DONE in", round(time.time() - T0), "s")
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--gate-concurrency", type=int, default=3)
    ap.add_argument("--deadline-hours", type=float, default=6.0)
    ap.add_argument("--assemble", help="comma-separated shard kernel slugs: make the merge-and-check kernel")
    ap.add_argument("--check312", help="assemble kernel slug: make the Python 3.12 check kernel")
    a = ap.parse_args()

    rows = [json.loads(l) for l in a.candidates.read_text().splitlines() if l.strip()]
    repos = sorted({r["repo"] for r in rows})
    mine = {r for r in repos if zlib.crc32(r.encode()) % a.nshards == a.shard}
    shard = [r for r in rows if r["repo"] in mine]
    if a.assemble or a.check312:
        shard = []
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for src, arc in ((ROOT / "scripts/oodc/build_env.py", "build_env.py"),
                         (ROOT / "scripts/oodc/stub_llm.py", "stub_llm.py"),
                         (ROOT / "scripts/oodb/ood45_harness.py", "ood45_harness.py"),
                         (ROOT / "scripts/colab/run_ood45.py", "run_ood45.py"),
                         (ROOT / "scripts/oodc/DATASET_README.md", "README.md")):
            tf.add(src, arcname=arc)
        for sub_dir, arc in ((ROOT / "submissions/v001_gemma_sampling", "submission"), (ROOT / AGENT_SUB, "e4b")):
            for f in sorted(sub_dir.rglob("*")):
                if f.is_file():
                    tf.add(f, arcname=f"{arc}/{f.relative_to(sub_dir)}")
        data = "".join(json.dumps(r) + "\n" for r in shard).encode()
        ti = tarfile.TarInfo("candidates.jsonl")
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))
    payload = buf.getvalue()
    kdir = ROOT / "kernels" / f"oodc_{a.name}"
    kdir.mkdir(parents=True, exist_ok=True)
    head = (f"PAYLOAD_B64 = {base64.b64encode(payload).decode()!r}\nSHA256 = {hashlib.sha256(payload).hexdigest()!r}\n"
            f"WORKERS = {a.workers}\nLIMIT = {a.limit}\nGATE_CONC = {a.gate_concurrency}\n"
            f"DEADLINE_H = {a.deadline_hours}\nHOST_PINS = {HOST_PINS!r}\n")
    body = ASSEMBLE_BODY if a.assemble else CHECK312_BODY if a.check312 else BODY
    (kdir / "oodc_kernel.py").write_text(head + body)
    sources = a.assemble.split(",") if a.assemble else [a.check312] if a.check312 else []
    meta = {"id": f"{USER}/g4-oodc-{a.name}", "title": f"g4-oodc-{a.name}", "code_file": "oodc_kernel.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False,
            "enable_tpu": False, "enable_internet": True,
            "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse"], "competition_sources": [],
            "kernel_sources": [f"{USER}/{k}" for k in sources], "model_sources": []}
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    n = min(len(shard), a.limit) if a.limit else len(shard)
    print(f"{kdir}: shard {a.shard}/{a.nshards}: {len(shard)} tasks in {len(mine)} repos (running {n}); "
          f"payload {len(payload) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
