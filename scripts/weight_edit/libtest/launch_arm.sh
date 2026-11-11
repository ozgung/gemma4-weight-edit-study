#!/bin/bash
# Launch one weightprog usability-test arm as headless Claude Code inside the study's macOS Seatbelt sandbox.
# Owner: Claude (fork, libtest). Derived from weightprog_study/src/weightprog_study/proposer/sandbox/launch_claude.sh
# without the forward-pass server: the arm loads the model itself in its own Python environment.
#
#   launch_arm.sh RUN_DIR PACKET_DIR MODEL EFFORT
#     RUN_DIR     read/write: edit/, notebook.md, final_answer.md, transcripts
#     PACKET_DIR  read-only except scratch/; the working directory; holds brief.md
# Env (required): WP_MODEL_DIR (model directory, read-only), WP_VENV (the arm's Python environment)
# Env (optional): WP_CONV (conversation number, default 1), WP_PROMPT (override the first prompt), WP_DRY=1
set -euo pipefail
RUN=$(cd "$1" && pwd); PACKET=$(cd "$2" && pwd); MODEL=$3; EFFORT=$4
: "${WP_MODEL_DIR:?set WP_MODEL_DIR}"; : "${WP_VENV:?set WP_VENV}"
MODELDIR=$(cd "$WP_MODEL_DIR" && pwd); VENV=$(cd "$WP_VENV" && pwd)
ROOT=/Users/ozgun/Dev/gemma-4-developer-agent-competition
PYHOME=$HOME/.local/share/uv/python
CLAUDE=$HOME/.local/bin/claude
RUN_ID=$(basename "$RUN")
TMPD=/private/tmp/wp_sbx_$RUN_ID
SB=$ROOT/weightprog_study/src/weightprog_study/proposer/sandbox/proposer_claude.sb
[ -f "$PACKET/brief.md" ] || { echo "packet $PACKET has no brief.md" >&2; exit 2; }
mkdir -p "$TMPD" "$RUN/edit"
SLUG=$(printf '%s' "$PACKET" | sed 's/[^A-Za-z0-9]/-/g')
PROMPT=${WP_PROMPT:-"Read brief.md in the current directory and follow it exactly. Your run folder is $RUN."}
CMD=(env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" PATH="$VENV/bin:/usr/bin:/bin:/usr/sbin:/sbin" TERM=dumb
     LANG=en_US.UTF-8 TMPDIR="$TMPD/" CLAUDE_CODE_TMPDIR="$TMPD" SHELL=/bin/bash HF_HUB_OFFLINE=1
     /usr/bin/sandbox-exec -f "$SB" -D HOME="$HOME" -D PACKET="$PACKET" -D RUN="$RUN" -D MODEL="$MODELDIR"
     -D VENV="$VENV" -D PYHOME="$PYHOME" -D TMPD="$TMPD" -D SLUG="$SLUG"
     "$CLAUDE" -p "$PROMPT" --model "$MODEL" --effort "$EFFORT" --permission-mode bypassPermissions
     --setting-sources "" --strict-mcp-config --mcp-config '{"mcpServers":{}}' --tools Bash Read Write Edit Glob Grep
     --add-dir "$RUN" --output-format stream-json --verbose)
if [ -n "${WP_DRY:-}" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
echo "[$(date -u +%FT%TZ)] launch $RUN_ID model=$MODEL effort=$EFFORT packet=$PACKET venv=$VENV" >> "$RUN/launcher.log"
cd "$PACKET"
"${CMD[@]}" < /dev/null > "$RUN/transcript_${WP_CONV:-1}.jsonl" 2>> "$RUN/launcher.log" && rc=0 || rc=$?
echo "[$(date -u +%FT%TZ)] exit rc=$rc" >> "$RUN/launcher.log"
exit $rc
