#!/bin/bash
# Build the arms' packets (ARMS, default "lib nolib"; an added arm named lib* gets the lib packet), run folders and the library arm's environment for the weightprog usability test.
# Owner: Claude (fork, libtest). Run once, after the plan is frozen and before any arm starts:
#   bash scripts/weight_edit/libtest/build_packets.sh HOURS
# Inputs: outputs/weight_edit/libtest/sets/cases_tune.jsonl (from make_pydantic_cases.py full), the weightprog working
# tree at /Users/ozgun/Dev/weightprog (snapshot copied, excluding .git, .venv and caches), the brief template.
set -euo pipefail
ROOT=/Users/ozgun/Dev/gemma-4-developer-agent-competition; cd "$ROOT"
D=outputs/weight_edit/libtest; HOURS=$1
WP=/Users/ozgun/Dev/weightprog
MD=$ROOT/data/models/gemma-4-e4b-it-qat-q4_0-unquantized
N_TUNE=$(grep -c . $D/sets/cases_tune.jsonl)
KEEP_MAX=$(python3 -I -c "import json;print(json.load(open('$D/criteria.json'))['keep_max_broken'])")
for ARM in ${ARMS:-lib nolib}; do
  PK=$D/packets/$ARM; RUN=$ROOT/$D/runs/$ARM
  [ -e "$PK" ] && { echo "$PK exists: refusing to overwrite" >&2; exit 2; }
  mkdir -p "$PK/scratch" "$RUN/edit"
  cp $D/sets/cases_tune.jsonl "$PK/cases_tune.jsonl"
  if [ $ARM != nolib ]; then
    # an added lib* arm gets the registered lib packet's snapshot, not the current working tree
    if [ $ARM != lib ] && [ -d $D/packets/lib/weightprog ]; then SRC=$D/packets/lib/weightprog; else SRC=$WP; fi
    rsync -a --exclude .git --exclude .venv --exclude __pycache__ --exclude .pytest_cache --exclude '*.egg-info' \
      --exclude build --exclude dist "$SRC/" "$PK/weightprog/"
    LIBTXT="
- **The \`weightprog\` library** (version 0.2.0) is installed in this environment, and its repository is in
  \`./weightprog/\` in this packet. It was written for this kind of task: tools to load the model, read activations,
  build closed-form edits and evaluate them. Start with \`weightprog/AGENTS.md\`."
  else
    LIBTXT=""
  fi
  python3 -I - "$PK/brief.md" "$RUN" "$MD" "$N_TUNE" "$HOURS" "$KEEP_MAX" "$LIBTXT" <<'EOF'
import sys
out, run, md, n, hours, keep, lib = sys.argv[1:8]
t = open("scripts/weight_edit/libtest/brief_template.md").read()
t = t.format(RUN=run, MODEL_DIR=md, N_TUNE=n, HOURS=hours, KEEP_MAX=keep, LIBRARY=lib, MLX="0.32.4", MLX_LM="0.32.2")
open(out, "w").write(t)
EOF
  (cd "$PK" && find . -type f -not -path './scratch/*' | sort | xargs shasum -a 256) > "$D/packets/MANIFEST_$ARM.sha256"
  chmod -R a-w "$PK"; chmod u+w "$PK" "$PK/scratch"
done
case " ${ARMS:-lib nolib} " in *" lib "*) ;; *) shasum -a 256 $D/packets/MANIFEST_*.sha256; exit 0 ;; esac
# the library arm's environment: same packages as nolib plus weightprog installed from the packet's snapshot
# (setuptools writes egg-info into the source tree, so install from a writable copy of the read-only snapshot)
BTMP=$(mktemp -d); cp -R "$D/packets/lib/weightprog" "$BTMP/weightprog"; chmod -R u+w "$BTMP/weightprog"
uv pip install -q --offline --no-deps --python $D/envs/lib/bin/python "$BTMP/weightprog"; rm -rf "$BTMP"
$D/envs/lib/bin/python -c "import weightprog; print('weightprog', weightprog.__version__, weightprog.__file__)"
$D/envs/nolib/bin/python -c "import importlib.util as u; assert u.find_spec('weightprog') is None; print('nolib: no weightprog')"
diff <(sed -n '1,200p' $D/packets/lib/brief.md) <(sed -n '1,200p' $D/packets/nolib/brief.md) || true
shasum -a 256 $D/packets/MANIFEST_*.sha256
