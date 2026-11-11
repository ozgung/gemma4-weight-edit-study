#!/bin/bash
# Added arm of the weightprog usability test, chained after the registered lib/nolib test. Owner: Claude (fork,
# libtest). Rule registered in docs/experiments/weight-edit-collab-20261102.md (11-10), before any judge
# output existed (user, about 21:30 UTC: "Evet, Opus sonucuna göre Fable otomatik devam et."):
#   lib PASS and nolib PASS            -> no added arm
#   any other pair of verdicts         -> Fable 5.2 in the lib packet (packets/libfable), everything else unchanged
#   judge summary missing or unreadable -> no added arm (logged; the session decides)
# Start window: the arm starts only between 17:00 and 04:45 UTC (the Mac is moved, and sleeps, after about 09:00 UTC);
# outside it the script waits for 17:00 UTC.
# Launch:  (nohup bash scripts/weight_edit/libtest/auto_added_arm.sh RUNNER_PID > outputs/weight_edit/libtest/auto_added_arm.out 2>&1 &)
# Stop:    pkill -f auto_added_arm.sh   (before the launch; after it, stop run_libtest.sh as its header says)
set -uo pipefail
ROOT=/Users/ozgun/Dev/gemma-4-developer-agent-competition; cd "$ROOT" || exit 1
D=${AA_D:-outputs/weight_edit/libtest}; LOG=$D/libtest.log   # AA_D, AA_HM, AA_DRY: stub tests only
RUNNER=${1:?pid of the running run_libtest.sh}
log() { echo "[$(date -u +%FT%TZ)] added arm: $*" >> $LOG; }
gpu_busy() { pgrep -f "wsp_serve[r].py|replay_ge[n].py|replay_fullcal[l].py|libtest/censu[s].py|libtest/judg[e].py" >/dev/null; }

log "armed (pid $$), waiting for libtest.done and runner $RUNNER"
until [ -f $D/libtest.done ]; do
  if ! kill -0 "$RUNNER" 2>/dev/null; then
    sleep 5; [ -f $D/libtest.done ] && break
    log "runner $RUNNER ended without libtest.done: no added arm"; exit 1
  fi
  sleep 60
done
while kill -0 "$RUNNER" 2>/dev/null; do sleep 5; done

read -r V_LIB V_NOLIB < <($D/envs/lib/bin/python -I -c '
import json, sys
a = json.load(open(sys.argv[1]))["arms"]
print(*(a.get(k, {}).get("verdict", "missing").replace(" ", "_") for k in ("lib", "nolib")))' \
  $D/judge/judge_summary.json 2>/dev/null) || true
V_LIB=${V_LIB:-missing}; V_NOLIB=${V_NOLIB:-missing}
log "reading: lib $V_LIB, nolib $V_NOLIB"
if [ "$V_LIB" = missing ] || [ "$V_NOLIB" = missing ]; then log "judge summary incomplete: no added arm"; exit 1; fi
if [ "$V_LIB" = PASS ] && [ "$V_NOLIB" = PASS ]; then log "PASS / PASS: no added arm (registered rule)"; exit 0; fi

said=0
while :; do
  hm=${AA_HM:-$(date -u +%H%M)}
  if (( 10#$hm >= 445 && 10#$hm < 1700 )); then
    [ $said -eq 0 ] && log "outside the start window: waiting for 17:00 UTC"; said=1; sleep 60; continue
  fi
  gpu_busy && { sleep 60; continue; }
  break
done
[ -n "${AA_DRY:-}" ] && { log "dry run: would launch libfable"; exit 0; }
log "launching libfable: claude-fable-5-2 effort max, 4 h, $(~/.local/bin/claude --version 2>/dev/null)"
exec env ARMS=libfable MODEL=claude-fable-5-2 JUDGE_OUT=judge_fable FIRST=3 DONE=libtest_fable.done \
  bash scripts/weight_edit/libtest/run_libtest.sh > $D/runner_fable.out 2>&1
