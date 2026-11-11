#!/bin/bash
# Tests of run_round3.sh with stubs: no model, no Claude, no real run folders.  Owner: Claude.
#   scripts/weight_edit/test_round3_runner.sh
# Stub server: a sleeping process named wsp_server_r3.py. Stub launcher: writes a transcript and, on a chosen conversation, a final answer.
set -u
REPO=/Users/ozgun/Dev/gemma-4-developer-agent-competition
RUNNER=$REPO/outputs/weight_edit/claude/round3/run_round3.sh
T=$(mktemp -d "${TMPDIR:-/tmp}/r3runner.XXXXXX")
fail() { echo "FAIL: $*"; tail -20 "$T/d/round3.log" 2>/dev/null; pkill -f "$T/wsp_server_r3.py" ; exit 1; }
cat > "$T/wsp_server_r3.py" <<'PY'
import json, sys, time
from pathlib import Path
if sys.argv[1] == "serve":
    time.sleep(600)
a = sys.argv
run = Path(sys.argv[3] if False else __import__("os").environ["R3_P"]) / a[2]
(run / "requests").mkdir(parents=True); (run / "edit").mkdir()
run.joinpath("state.json").write_text(json.dumps({"used": 0, "budget": 25, "model": a[a.index("--model") + 1], "fullcall": "--fullcall" in a}))
PY
cat > "$T/launch.sh" <<'SH'
#!/bin/bash
# RUN MODEL EFFORT BRIEF [PF]; behaviour by run id
[ -n "${WSP_DRY:-}" ] && exit 0
run=$1; R=$R3_P/$run; conv=${WSP_CONV:-1}
echo "$run conv=$conv model=$2 effort=$3 prior=$WSP_PRIOR pf=${5:-}" >> "$R3_D/launches.txt"
T=$R/transcript_$conv.jsonl
echo '{}' > "$T"
sleep "${STUB_SLEEP:-1}"
case "$run" in
  opus_r3a) [ "$conv" -ge 2 ] && echo done > "$R/final_answer.md";;
  opus_r3b) if [ ! -f "$R/limited" ]; then touch "$R/limited"; echo 'usage limit reached' > "$T"; else echo done > "$R/final_answer.md"; fi;;
  fable55_r3) echo done > "$R/final_answer.md";;
esac
exit 0
SH
chmod +x "$T/launch.sh"
printf 'import sys\nsys.exit(0)\n' > "$T/ok.py"
setup() { rm -rf "$T/d" "$T/p" "$T/g"; mkdir -p "$T/d" "$T/p" "$T/g"; echo x > "$T/d/x.txt"; (cd "$T/d" && shasum -a 256 x.txt > FROZEN.sha256); touch "$T/g/gonogo.done"; echo '{"GO": true}' > "$T/g/gonogo_report.json"; }
export R3_D=$T/d R3_P=$T/p R3_PY=python3 R3_SRV=$T/wsp_server_r3.py R3_LAUNCH=$T/launch.sh R3_CHECK=$T/ok.py R3_AUDIT=$T/ok.py R3_GONOGO=$T/g
export R3_LIMIT_WAIT=1 R3_TICK=1 R3_SRV_WAIT=1 R3_GAP=1
# 1. refusals
setup; rm "$T/g/gonogo.done"
"$RUNNER" 2>/dev/null; [ $? -eq 3 ] || fail "ran without gonogo.done"
setup; echo '{"GO": false}' > "$T/g/gonogo_report.json"
"$RUNNER" 2>/dev/null; [ $? -eq 3 ] || fail "ran on a no-go"
setup; echo y > "$T/d/x.txt"
"$RUNNER" 2>/dev/null; [ $? -eq 3 ] || fail "ran with a changed frozen file"
[ ! -e "$T/d/launches.txt" ] || fail "launched after a refusal"
"$RUNNER" --bogus >/dev/null 2>&1; [ $? -eq 2 ] || fail "bad argument accepted"
echo "ok: refusals"
# 2. two proposers: relaunch on an early end, a usage limit does not count, audit, server stopped, done marker
setup
"$RUNNER"
[ -f "$T/d/round3.done" ] || fail "no done marker"
[ "$(grep -c 'opus_r3a' "$T/d/launches.txt")" -eq 2 ] || fail "opus_r3a should run 2 conversations"
grep -q 'opus_r3a conv=2' "$T/d/launches.txt" && grep -q "pf=.*prompt_opus_r3a_2.txt" "$T/d/launches.txt" || fail "no continuation prompt"
[ "$(grep -c 'opus_r3b conv=1' "$T/d/launches.txt")" -eq 2 ] || fail "the limited conversation must be repeated as conversation 1"
grep -q 'limit: waiting' "$T/d/round3.log" || fail "no limit wait"
ls "$T/p/opus_r3b/" | grep -q 'transcript_1_limit' || fail "limited transcript not kept"
grep -q 'server stopped' "$T/d/round3.log" || fail "server not stopped"
grep -q 'claude-opus-5-5' "$T/p/opus_r3a/state.json" && grep -q '"fullcall": true' "$T/p/opus_r3a/state.json" || fail "init arguments"
grep -q 'effort=max' "$T/d/launches.txt" || fail "effort"
grep -q 'budget 25\|25 passes' "$T/d/brief_opus_r3a.md" || fail "brief budget"
[ -z "$(pgrep -f "$T/wsp_server_r3.py")" ] || fail "server still running"
echo "ok: two proposers"
# 3. all three in one go
setup
"$RUNNER" --fable55 claude-fable-5-5
grep -q 'fable55_r3 conv=1 model=claude-fable-5-5' "$T/d/launches.txt" || fail "fable run not started"
grep -q 'claude-fable-5-5' "$T/p/fable55_r3/state.json" || fail "fable init"
echo "ok: three proposers"
# 4. the third run joins later on the same server; the last invocation stops it; finished runs are skipped
setup
STUB_SLEEP=6 "$RUNNER" &
A=$!
sleep 3
"$RUNNER" --fable55 claude-fable-5-5
[ "$(grep -c 'server started' "$T/d/round3.log")" -eq 1 ] || fail "second server started"
grep -q 'opus_r3a already running' "$T/d/round3.log" || fail "opus runs not skipped"
[ ! -f "$T/d/round3.done" ] || fail "done while the opus runs were active"
[ -n "$(pgrep -f "$T/wsp_server_r3.py")" ] || fail "fable invocation stopped the server under the opus runs"
wait $A
[ -f "$T/d/round3.done" ] || fail "no done marker after the last invocation"
[ -z "$(pgrep -f "$T/wsp_server_r3.py")" ] || fail "server left running"
"$RUNNER" --fable55 claude-fable-5-5
grep -q 'nothing to start' "$T/d/round3.log" || fail "finished runs not skipped"
echo "ok: late third run"
# 5. the server dies mid-pass: the claimed request is re-queued and a new server starts
setup
STUB_SLEEP=7 "$RUNNER" &
A=$!
sleep 3
mkdir -p "$T/p/opus_r3a/work/fwd_3"; echo '{}' > "$T/p/opus_r3a/requests/fwd_3.json.claimed"
kill "$(cat "$T/d/server.pid")"
sleep 4
[ -f "$T/p/opus_r3a/requests/fwd_3.json" ] && [ ! -d "$T/p/opus_r3a/work/fwd_3" ] || fail "not re-queued"
[ "$(grep -c 'server started' "$T/d/round3.log")" -eq 2 ] || fail "no restart"
echo '{}' > "$T/p/opus_r3a/requests/fwd_3.done.json"
wait $A
[ -f "$T/d/round3.done" ] || fail "no done marker after a restart"
echo "ok: server restart"
# 6. markers of a dead invocation do not block the end
setup
mkdir -p "$T/d/active"; echo 999999 > "$T/d/active/fable55_r3"
"$RUNNER"
[ -f "$T/d/round3.done" ] && grep -q 'dropped stale marker fable55_r3' "$T/d/round3.log" || fail "stale marker blocked the end"
echo "ok: stale marker"
pkill -f "$T/wsp_server_r3.py"
echo "all runner tests passed"
