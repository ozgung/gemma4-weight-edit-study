#!/bin/bash
# Test of run_gate_r3.sh without a model: the generators are stubbed with the stored round-2 generations; freezing, controls, scorers and analysis are real.
#   scripts/weight_edit/test_gate_runner_r3.sh
# Owner: Claude. Writes only to a temp dir. The expected figures are the dry-run ones (base repeats 78 of 142, retry acceptance 38 of 142).
set -u
REPO=/Users/ozgun/Dev/gemma-4-developer-agent-competition; cd "$REPO" || exit 1
T=$(mktemp -d "${TMPDIR:-/tmp}/r3gate.XXXXXX")
C=outputs/weight_edit/claude
fail() { echo "FAIL: $*"; tail -20 "$T/g/gate_r3.log"; exit 1; }
mkdir -p "$T/runs/fable_r2" "$T/runs/empty_r3/edit"
cp -R $C/gate_r2/fable_r2/edit "$T/runs/fable_r2/edit"; chmod -R u+w "$T/runs/fable_r2/edit"
cat > "$T/py" <<'SH'
#!/bin/bash
# stub for the MLX python: real numpy for gate_controls.py, stored generations for the two generators
case "$2" in
  */gate_controls.py) exec outputs/small_model/mlx_venv/bin/python "$@";;
  */replay_gen.py|*/replay_fullcall.py)
    shift; exec python3 -I -c '
import json, sys
from pathlib import Path
args = sys.argv[2:]
out = Path(args[args.index("--out") + 1])
edit = args[args.index("--edit") + 1] if "--edit" in args else None
keys = {l.strip() for l in open(args[args.index("--keys") + 1]) if l.strip()}
draws = int(args[args.index("--draws") + 1])
src = Path("outputs/weight_edit/claude/gate_r2") / out.relative_to(out.parent.parent if out.parent.name == "fullcall" else out.parent)
with out.open("w") as f:
    for r in map(json.loads, src.open()):
        assert r["key"] in keys and r["draw"] < draws
        r["edit"] = edit
        f.write(json.dumps(r) + "\n")
' "$@";;
esac
exit 9
SH
chmod +x "$T/py"
export WSP_PY=$T/py GATE_DIR=$T/g GATE_RUNS_DIR=$T/runs
# a used set is refused unless it is declared as one
$C/gate_r3/run_gate_r3.sh $C/replay_ood fable_r2 >/dev/null 2>&1
[ $? -eq 3 ] && grep -q "open to the editors" "$T/g/gate_r3.log" || fail "the used set was not refused"
rm -f "$T/g/gate_r3.log"
GATE_USED_SET=1 GATE_M=2 $C/gate_r3/run_gate_r3.sh $C/replay_ood fable_r2 empty_r3
[ $? -eq 0 ] || fail "runner failed"
[ -f "$T/g/gate_r3.done" ] || fail "no done marker"
grep -q "empty_r3: empty edit, not a candidate" "$T/g/gate_r3.log" || fail "empty edit not logged"
[ -f "$T/g/fable_r2/FROZEN.txt" ] || fail "no freeze list"
[ ! -w "$T/g/fable_r2/edit" ] || fail "frozen edit is writable"
python3 -I - "$T/g" <<'PY' || fail "report"
import json, sys
from pathlib import Path
g = Path(sys.argv[1])
r = json.loads((g / "gate_r3_report.json").read_text())
c = r["candidates"]["fable_r2"]
assert r["holm_m"] == 2 and r["base"]["repeats"] == 78 and r["base"]["retry_accepted"] == 38 and r["reportable"] is False
assert c["reported"]["counts"]["candidate"]["repeats"] == 64 and c["reported"]["counts"]["candidate"]["retry_accepted"] == 45
assert not c["guard_b_served_ok"]["pass"] and c["guard_c_fullcall_compile"]["pass"] and not c["passes"]
ctrl = json.loads((g / "fable_r2_rand1/control.json").read_text())
assert ctrl[0]["seed"] == 20261500 + 10 * 5 + 1 and abs(ctrl[0]["control_fro"] - ctrl[0]["target_fro"]) < 1e-3, ctrl[0]
PY
echo "gate runner test passed"
