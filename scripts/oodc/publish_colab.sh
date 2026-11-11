#!/bin/bash
# Publish OOD-C as the private dataset ozgung/g4-oodc from a Colab CPU VM, so the 15+ GB never pass through the Mac and the
# Kaggle key stays on the user's own VM. Runs no repository code. Upload first: /root/.kaggle/kaggle.json,
# /content/finalize_tasks.py, /content/README_dataset.md.
#   (setsid nohup bash /content/publish_colab.sh > /content/publish.log 2>&1 &) > /dev/null 2>&1
set -uo pipefail
pip install -q -U kaggle
cd /content
retry() { for i in 1 2 3 4 5 6; do "$@" && return 0; echo "retry $i: $*"; sleep 60; done; return 1; }
mkdir -p oodc_out oodc_check
retry kaggle kernels output ozgung/g4-oodc-assemble -p /content/oodc_out --file-pattern '^oodc_set/' --page-size 200 -q
retry kaggle kernels output ozgung/g4-oodc-check312 -p /content/oodc_check --file-pattern 'gate312/gate\.json$' --page-size 200 -q
O=/content/oodc_out/oodc_set
python3 -I /content/finalize_tasks.py --ood $O --gate312 /content/oodc_check/gate312/gate.json || exit 1
cp /content/README_dataset.md $O/README.md
python3 -I - "$O" <<'PY' || exit 1
import gzip, json, sys
from pathlib import Path
o = Path(sys.argv[1])
rows = [json.loads(l) for l in (o / "tasks.jsonl").read_text().splitlines() if l.strip()]
snaps = {p.stem for p in (o / "snapshots").glob("*.tgz")}
ids = {r["instance_id"] for r in rows}
print("tasks", len(rows), "repos", len({r["repo"] for r in rows}), "snapshots", len(snaps))
assert ids <= snaps, sorted(ids - snaps)[:5]
for extra in snaps - ids:
    (o / "snapshots" / f"{extra}.tgz").unlink()
for i in sorted(ids):
    with gzip.open(o / "snapshots" / f"{i}.tgz", "rb") as fh:
        while fh.read(1 << 20):
            pass
print("all snapshots gunzip clean")
house = {p.name for p in (o / "wheels" / "house").iterdir()}
for key in ("x86_64-cp312", "x86_64-cp313"):
    man = json.loads((o / "wheels" / key / "manifest.json").read_text())
    repos = {r["repo"].split("/", 1)[1] for r in rows}
    assert repos <= set(man), sorted(repos - set(man))[:5]
    names = {n for k in repos for n in man[k]}
    print(key, "wheels referenced", len(names), "missing", len(names - house))
    assert names <= house
PY
du -sh $O
cat > $O/dataset-metadata.json <<'JSON'
{"title": "g4-oodc", "id": "ozgung/g4-oodc", "subtitle": "", "description": "OOD-C held-out Python bug-fix tasks", "licenses": [{"name": "other"}]}
JSON
# the kaggle CLI exits 0 on a refused create (2026-11-08: private quota), so check its output
kaggle datasets create -p $O --dir-mode zip -t ${PUBLIC:+-u} 2>&1 | tee /content/create.log  # PUBLIC=1: public (11-12 release)
if grep -qi "error" /content/create.log; then echo "PUBLISH FAILED"; exit 1; fi
echo PUBLISH DONE
