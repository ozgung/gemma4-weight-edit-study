#!/bin/bash
# Launch one weight-edit proposer as a headless Codex CLI process inside the macOS sandbox.  Owner: Claude.
#   scripts/weight_edit/sandbox/launch_codex.sh RUN_ID MODEL EFFORT BRIEF [PROMPT_FILE]
# Same packet as launch_proposer.sh (built by it with WSP_DRY=1). Codex runs with its own sandbox and approvals off,
# because macOS does not allow nesting a second Seatbelt profile; the OS sandbox (proposer_codex.sb) is the boundary,
# as it is for the Claude proposers (which run with bypassPermissions). CODEX_HOME is an isolated dir in the run's temp
# dir holding a minimal config.toml (model, effort; no plugins, MCP servers, notify hooks or user instructions) and a
# symlink to the user's auth.json.
# Output: RUN folder/transcript_<WSP_CONV>.jsonl (codex exec --json events), RUN folder/launcher.log
set -euo pipefail
REPO=/Users/ozgun/Dev/gemma-4-developer-agent-competition
RUN_ID=$1; MODEL=$2; EFFORT=$3; BRIEF=$4; PROMPT_FILE=${5:-}
PACKET=$REPO/outputs/weight_edit/sandbox/$RUN_ID
RUN=$REPO/outputs/weight_edit/proposers/$RUN_ID
MODELDIR=$REPO/data/models/gemma-4-e4b-it-qat-q4_0-unquantized
VENV=$REPO/outputs/small_model/mlx_venv
TMPD=/private/tmp/wsp_sbx_$RUN_ID
SB=$REPO/scripts/weight_edit/sandbox/proposer_codex.sb
WSP_DRY=1 "$REPO/scripts/weight_edit/sandbox/launch_proposer.sh" "$RUN_ID" "$MODEL" "$EFFORT" "$BRIEF" > /dev/null
CH=$TMPD/codex_home
mkdir -p "$CH"
printf 'model = "%s"\nmodel_reasoning_effort = "%s"\n' "$MODEL" "$EFFORT" > "$CH/config.toml"
ln -sfn "$HOME/.codex/auth.json" "$CH/auth.json"
PROMPT="Read brief.md in the current directory and follow it exactly. Your run folder is $RUN."
[ -n "$PROMPT_FILE" ] && PROMPT=$(cat "$PROMPT_FILE")
echo "[$(date -u +%FT%TZ)] launch $RUN_ID codex model=$MODEL effort=$EFFORT packet=$PACKET" >> "$RUN/launcher.log"
# Usage-limit handling (added 2026-11-04 21:4x UTC, user: "can be out of usage before that. in that case cont. when
# available"): a conversation that ends on a usage/rate-limit error (an `error` or `turn.failed` event) waits until
# WSP_RESET_UTC (file round2/astra_reset_utc.txt, else +30 min) and relaunches with a continuation prompt, without using
# up one of the runner's conversations. At most 12 such retries per conversation.
RESET_FILE=$REPO/outputs/weight_edit/claude/round2/astra_reset_utc.txt
usage_hit() {  # transcript -> 0 if it ends on a usage/rate-limit failure
  [ -f "$1" ] && /usr/bin/python3 -I - "$1" <<'PY'
import json, re, sys
bad = re.compile(r"usage limit|rate.?limit|quota|429|too many requests|insufficient|credits", re.I)
hit = False
for line in open(sys.argv[1]):
    try:
        e = json.loads(line)
    except ValueError:
        continue
    if e.get("type") in ("error", "turn.failed") and bad.search(json.dumps(e)):
        hit = True
    elif e.get("type") == "turn.completed":
        hit = False
sys.exit(0 if hit else 1)
PY
}
transient_hit() {  # transcript -> 0 if it ends on a provider-side capacity/server failure (added 2026-11-05 13:5x UTC,
  # after conversation 4 ended on "Selected model is at capacity"; user approved a 5th conversation)
  [ -f "$1" ] && /usr/bin/python3 -I - "$1" <<'PY'
import json, re, sys
bad = re.compile(r"at capacity|overloaded|server error|internal error|50[234]|stream disconnected|try a different model", re.I)
hit = False
for line in open(sys.argv[1]):
    try:
        e = json.loads(line)
    except ValueError:
        continue
    if e.get("type") in ("error", "turn.failed") and bad.search(json.dumps(e)):
        hit = True
    elif e.get("type") == "turn.completed":
        hit = False
sys.exit(0 if hit else 1)
PY
}
log_hit() {  # stderr of the latest launch in launcher.log mentions a usage/rate limit
  awk '/ launch | relaunch /{buf=""} {buf=buf"\n"$0} END{print buf}' "$RUN/launcher.log" 2>/dev/null \
    | grep -v -E "^\[[0-9T:-]+Z\] (launch|relaunch|usage limit)" | grep -q -i -E "usage limit|rate.?limit|quota|429|too many requests"
}
wait_reset() {
  local now t
  now=$(date -u +%s)
  t=$( [ -f "$RESET_FILE" ] && date -j -u -f "%Y-%m-%dT%H:%M:%SZ" "$(cat "$RESET_FILE")" +%s 2>/dev/null || echo 0)
  if [ "$t" -gt "$now" ]; then
    echo "[$(date -u +%FT%TZ)] usage limit: waiting until $(cat "$RESET_FILE")" >> "$RUN/launcher.log"; sleep $((t - now + 180))
  else
    echo "[$(date -u +%FT%TZ)] usage limit: reset time passed or unknown; waiting 30 min" >> "$RUN/launcher.log"; sleep 1800
  fi
}
CONTINUE="Read brief.md in the current directory and follow it exactly. Your run folder is $RUN. Your previous conversation was cut off by a usage limit before the budget was used. Read the run folder's notebook.md, state.json and requests/*.done.json first, then continue from where it stopped."
CONTINUE_T="Read brief.md in the current directory and follow it exactly. Your run folder is $RUN. Your previous conversation was cut off by a provider error before it finished. Read the run folder's notebook.md, state.json and requests/*.done.json first, then continue from where it stopped."
PREV=$(ls -t "$RUN"/transcript_$(( ${WSP_CONV:-1} - 1 )).jsonl "$RUN"/transcript_$(( ${WSP_CONV:-1} - 1 ))_retry*.jsonl 2>/dev/null | head -1 || true)  # newest file of the previous conversation
if [ "${WSP_CONV:-1}" -gt 1 ] && { usage_hit "$PREV" || log_hit; } && [ ! -f "$RUN/final_answer.md" ]; then wait_reset; PROMPT=$CONTINUE; fi
OUT=$RUN/transcript_${WSP_CONV:-1}.jsonl
cd "$PACKET"
env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" PATH="$VENV/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin" TERM=dumb \
    LANG=en_US.UTF-8 TMPDIR="$TMPD/" CODEX_HOME="$CH" SHELL=/bin/bash \
  /usr/bin/sandbox-exec -f "$SB" -D HOME="$HOME" -D REPO="$REPO" -D PACKET="$PACKET" -D RUN="$RUN" -D MODEL="$MODELDIR" \
    -D VENV="$VENV" -D TMPD="$TMPD" \
  /opt/homebrew/bin/codex exec --skip-git-repo-check -s danger-full-access -c approval_policy='"never"' \
    -C "$PACKET" --add-dir "$RUN" --json "$PROMPT" \
  < /dev/null > "$OUT" 2>> "$RUN/launcher.log" && rc=0 || rc=$?
echo "[$(date -u +%FT%TZ)] exit rc=$rc" >> "$RUN/launcher.log"
for retry in $(seq 1 12); do
  [ ! -f "$RUN/final_answer.md" ] || break
  if usage_hit "$OUT" || log_hit; then
    wait_reset; CP=$CONTINUE; why="usage limit"
  elif transient_hit "$OUT"; then
    echo "[$(date -u +%FT%TZ)] provider error: waiting 10 min" >> "$RUN/launcher.log"; sleep 600; CP=$CONTINUE_T; why="provider error"
  else
    break
  fi
  OUT=$RUN/transcript_${WSP_CONV:-1}_retry$retry.jsonl
  echo "[$(date -u +%FT%TZ)] relaunch after $why (retry $retry)" >> "$RUN/launcher.log"
  env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" PATH="$VENV/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin" TERM=dumb \
      LANG=en_US.UTF-8 TMPDIR="$TMPD/" CODEX_HOME="$CH" SHELL=/bin/bash \
    /usr/bin/sandbox-exec -f "$SB" -D HOME="$HOME" -D REPO="$REPO" -D PACKET="$PACKET" -D RUN="$RUN" -D MODEL="$MODELDIR" \
      -D VENV="$VENV" -D TMPD="$TMPD" \
    /opt/homebrew/bin/codex exec --skip-git-repo-check -s danger-full-access -c approval_policy='"never"' \
      -C "$PACKET" --add-dir "$RUN" --json "$CP" \
    < /dev/null > "$OUT" 2>> "$RUN/launcher.log" && rc=0 || rc=$?
  echo "[$(date -u +%FT%TZ)] exit rc=$rc" >> "$RUN/launcher.log"
done
exit $rc
