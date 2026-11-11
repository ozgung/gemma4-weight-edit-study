#!/bin/bash
# Run the weightprog usability test.  Owner: Claude (fork, libtest).
# Launch only on the user's go (each arm spends model tokens):
#   (nohup bash scripts/weight_edit/libtest/run_libtest.sh > outputs/weight_edit/libtest/runner.out 2>&1 &)
# Default (the registered test): ARMS "lib nolib", Claude Opus 5.5 effort max through Claude Code, one after the
#   other, then the judge on both -> judge/, marker libtest.done.
# An added arm, decided after that result and registered separately before its launch, reuses this script with env
#   overrides; arms named lib* use the lib environment, others the nolib one; packets/<arm> must exist
#   (ARMS=<arm> bash build_packets.sh HOURS). Examples:
#   ARMS=libfable MODEL=claude-fable-5-2 JUDGE_OUT=judge_fable FIRST=3 DONE=libtest_fable.done
#   ARMS=libcarbon MODEL=<id from agy models> LAUNCH=launch_arm_agy.sh JUDGE_OUT=judge_carbon FIRST=3 DONE=libtest_carbon.done
#   (the Carbon arm runs Gemini 4 Carbon through the Antigravity CLI in proposer_agy.sb, signed in to
#   outputs/weight_edit/libtest/agyhome; on hold since 11-10 16:40 UTC, user).
# Budget: HOURS (default 4) of wall clock per arm, not counting waits after a usage or quota limit (the conversation is
# resumed from the notebook after 30 min, at most 6 times). At the limit the arm's processes are stopped and its
# edit/ is frozen. Log: outputs/weight_edit/libtest/libtest.log.
# Stop: pkill -f run_libtest.sh; pkill -f "libtest/packets"; pkill -f "libtest/envs/(lib|nolib)/bin/python".
set -uo pipefail
ROOT=/Users/ozgun/Dev/gemma-4-developer-agent-competition; cd "$ROOT" || exit 1
D=outputs/weight_edit/libtest; LOG=$D/libtest.log
HOURS=${HOURS:-4}; ARMS=${ARMS:-"lib nolib"}; MODEL=${MODEL:-claude-opus-5-5}; EFFORT=${EFFORT:-max}
LAUNCH=${LAUNCH:-launch_arm.sh}; JUDGE_OUT=${JUDGE_OUT:-judge}; FIRST=${FIRST:-1}; DONE=${DONE:-libtest.done}
MD=$ROOT/data/models/gemma-4-e4b-it-qat-q4_0-unquantized
log() { echo "[$(date -u +%FT%TZ)] $*" >> $LOG; }
gpu_busy() { pgrep -f "wsp_serve[r].py|replay_ge[n].py|replay_fullcal[l].py|libtest/censu[s].py|libtest/judg[e].py" >/dev/null; }
iso() { date -u -r "$1" +%FT%TZ; }
killtree() { local c; for c in $(pgrep -P "$1"); do killtree "$c"; done; kill -TERM "$1" 2>/dev/null; }
sweep() {  # orphaned arm processes re-parented to launchd: python started bare (the sandbox PATH) or from the arm's
           # env, and agy processes whose environment carries the arm's TMPDIR
  ps -axo pid=,ppid=,command= | awk -v v="$1" '$2 == 1 && ($3 ~ /^python[0-9.]*$/ || index($3, v) == 1) {print $1}' |
    while read -r p; do kill -TERM "$p" 2>/dev/null; done
  ps -axE -o pid=,ppid=,command= 2>/dev/null | awk -v t="TMPDIR=$2/" '$2 == 1 && index($0, t) {print $1}' |
    while read -r p; do kill -TERM "$p" 2>/dev/null; done
}
LIMIT_RE='"status":"rejected"|usage limit|rate limit reached|hit your limit|RESOURCE_EXHAUSTED|[Qq]uota exceeded|exhausted your|out of quota'

run_arm() {  # ARM LAUNCHER MODEL VENV_NAME PACKET_NAME
  local ARM=$1 LAUNCH=$2 M=$3 RUN=$ROOT/$D/runs/$1 VENV=$ROOT/$D/envs/$4 PK=$ROOT/$D/packets/$5
  local budget=$((HOURS * 3600)) used=0 conv=1 tries=0 timed_out=0 start deadline P pid rc T t_start lim
  until ! gpu_busy; do sleep 60; done
  t_start=$(date +%s)
  log "$ARM start: $M effort $EFFORT via $(basename "$LAUNCH"), packet $(basename "$PK"), env $4"
  while :; do
    start=$(date +%s); deadline=$((start + budget - used))
    if [ $conv -eq 1 ]; then
      P="Read brief.md in the current directory and follow it exactly. Your run folder is $RUN. The run started at $(iso $start); your time limit is $(iso $deadline) (UTC; check with \`date -u\`)."
    else
      P="Read brief.md in the current directory and follow it exactly. Your run folder is $RUN. This is conversation $conv of the run: the previous one ended early. Read the run folder's notebook.md first and continue from where it stopped. Your time limit is now $(iso $deadline) (UTC; check with \`date -u\`)."
    fi
    log "$ARM conversation $conv start, deadline $(iso $deadline)"
    WP_MODEL_DIR=$MD WP_VENV=$VENV WP_CONV=$conv WP_PROMPT="$P" bash "$LAUNCH" "$RUN" "$PK" "$M" $EFFORT &
    pid=$!
    while kill -0 $pid 2>/dev/null; do
      if [ "$(date +%s)" -ge $deadline ]; then
        log "$ARM time limit reached: stopping"
        killtree $pid; sleep 5; pkill -KILL -P $pid 2>/dev/null
        timed_out=1; break
      fi
      sleep 20
    done
    wait $pid 2>/dev/null; rc=$?
    used=$((used + $(date +%s) - start))
    sweep "$VENV" "/private/tmp/wp_sbx_$ARM"; sleep 3
    T=$RUN/transcript_$conv.jsonl
    log "$ARM conversation $conv exit rc=$rc after $(( ($(date +%s) - start) / 60 )) min (used $((used / 60)) of $((budget / 60)) min)"
    [ $timed_out -eq 1 ] && break
    # grep -c reads all its input (no SIGPIPE) and `|| true` keeps pipefail from hiding a match
    lim=$( { tail -n 30 "$T"; tail -n 30 "$RUN/launcher.log"; } 2>/dev/null | grep -Ec "$LIMIT_RE" || true)
    if [ $used -lt $budget ] && [ "${lim:-0}" -gt 0 ] && [ $tries -lt 6 ]; then
      tries=$((tries + 1)); log "$ARM usage limit: waiting 30 min (not counted)"; sleep 1800; conv=$((conv + 1)); continue
    fi
    break
  done
  mkdir -p "$RUN/frozen"; chmod -R u+w "$RUN/frozen/edit" 2>/dev/null; rm -rf "$RUN/frozen/edit"; cp -R "$RUN/edit" "$RUN/frozen/edit"; chmod -R a-w "$RUN/frozen/edit"
  (cd "$RUN/frozen/edit" && shasum -a 256 *.npz 2>/dev/null) > "$RUN/frozen/FROZEN.txt"
  log "$ARM frozen: $(grep -c . "$RUN/frozen/FROZEN.txt") files, wall $(( ($(date +%s) - t_start) / 60 )) min"
}

judge() {  # OUT FIRST_INDEX ARM...
  local out=$1 first=$2; shift 2
  local specs=(); for a in "$@"; do specs+=("$a=$D/runs/$a/frozen/edit"); done
  until ! gpu_busy; do sleep 60; done
  log "judge start: $* -> $out"
  $D/envs/lib/bin/python -I scripts/weight_edit/libtest/judge.py --cases $D/sets/cases_holdout.jsonl \
    --texts $D/judge_texts.json --criteria $D/criteria.json --seed-base 20261110 --first-index "$first" --out $D/$out \
    "${specs[@]}" > $D/$out.out 2> $D/$out.err
  log "judge rc=$? ($out)"
}

log "runner armed (pid $$): arms [$ARMS], $HOURS h each, $MODEL effort $EFFORT via $LAUNCH -> $JUDGE_OUT"
for ARM in $ARMS; do
  case $ARM in lib*) ENV=lib ;; *) ENV=nolib ;; esac
  run_arm "$ARM" "$ROOT/scripts/weight_edit/libtest/$LAUNCH" "$MODEL" "$ENV" "$ARM"
done
judge "$JUDGE_OUT" "$FIRST" $ARMS
touch "$D/$DONE"; log "all done ($DONE)"
