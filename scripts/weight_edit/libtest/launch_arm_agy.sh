#!/bin/bash
# Launch one weightprog usability-test arm as headless Antigravity CLI (agy, e.g. Gemini 4 Carbon) inside a macOS
# Seatbelt sandbox. Owner: Claude (fork, libtest). Same interface as launch_arm.sh:
#
#   launch_arm_agy.sh RUN_DIR PACKET_DIR MODEL EFFORT
#     RUN_DIR     read/write: edit/, notebook.md, final_answer.md, transcripts
#     PACKET_DIR  read-only except scratch/; the working directory; holds brief.md
# Env (required): WP_MODEL_DIR (model directory, read-only), WP_VENV (the arm's Python environment)
# Env (optional): WP_CONV (conversation number, default 1), WP_PROMPT (override the first prompt), WP_DRY=1,
#   WP_AGYHOME (isolated agy home, default outputs/weight_edit/libtest/agyhome; sign in once there, outside the
#   sandbox: HOME=<that dir> ~/.gemini/bin/agy).
set -euo pipefail
RUN=$(cd "$1" && pwd); PACKET=$(cd "$2" && pwd); MODEL=$3; EFFORT=$4
: "${WP_MODEL_DIR:?set WP_MODEL_DIR}"; : "${WP_VENV:?set WP_VENV}"
MODELDIR=$(cd "$WP_MODEL_DIR" && pwd); VENV=$(cd "$WP_VENV" && pwd)
ROOT=/Users/ozgun/Dev/gemma-4-developer-agent-competition
AGYHOME=$(cd "${WP_AGYHOME:-$ROOT/outputs/weight_edit/libtest/agyhome}" && pwd)
AGYBIN=$HOME/.gemini/bin
AGY=$AGYBIN/agy
PYHOME=$HOME/.local/share/uv/python
RUN_ID=$(basename "$RUN")
TMPD=/private/tmp/wp_sbx_$RUN_ID
SB=$ROOT/scripts/weight_edit/libtest/proposer_agy.sb
[ -f "$PACKET/brief.md" ] || { echo "packet $PACKET has no brief.md" >&2; exit 2; }
mkdir -p "$TMPD" "$RUN/edit"
PROMPT=${WP_PROMPT:-"Read brief.md in the current directory and follow it exactly. Your run folder is $RUN."}
CMD=(env -i HOME="$AGYHOME" USER="$USER" LOGNAME="$USER" PATH="$VENV/bin:/usr/bin:/bin:/usr/sbin:/sbin" TERM=dumb
     LANG=en_US.UTF-8 TMPDIR="$TMPD/" SHELL=/bin/bash HF_HUB_OFFLINE=1
     /usr/bin/sandbox-exec -f "$SB" -D HOME="$HOME" -D AGYHOME="$AGYHOME" -D AGYBIN="$AGYBIN" -D PACKET="$PACKET"
     -D RUN="$RUN" -D MODEL="$MODELDIR" -D VENV="$VENV" -D PYHOME="$PYHOME" -D TMPD="$TMPD"
     "$AGY" -p "$PROMPT" --model "$MODEL" --effort "$EFFORT" --dangerously-skip-permissions --disable-slash-commands
     --add-dir "$RUN" --output-format stream-json)
if [ -n "${WP_DRY:-}" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
echo "[$(date -u +%FT%TZ)] launch $RUN_ID agy $(python3 -I -c 'import json,sys;print(json.load(open(sys.argv[1]))["version"])' "$AGYBIN/agy.version.json" 2>/dev/null) model=$MODEL effort=$EFFORT packet=$PACKET venv=$VENV" >> "$RUN/launcher.log"
cd "$PACKET"
"${CMD[@]}" < /dev/null > "$RUN/transcript_${WP_CONV:-1}.jsonl" 2>> "$RUN/launcher.log" && rc=0 || rc=$?
echo "[$(date -u +%FT%TZ)] exit rc=$rc" >> "$RUN/launcher.log"
exit $rc
