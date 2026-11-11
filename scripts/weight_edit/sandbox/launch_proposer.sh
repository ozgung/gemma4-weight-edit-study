#!/bin/bash
# Launch one weight-edit proposer as a headless Claude Code process inside a macOS sandbox.  Owner: Claude.
#
#   scripts/weight_edit/sandbox/launch_proposer.sh RUN_ID MODEL EFFORT BRIEF [PROMPT_FILE]
#     RUN_ID   run id (the server's run folder outputs/weight_edit/proposers/RUN_ID must already exist: wsp_server init)
#     MODEL    claude model alias or id (opus, fable, sonnet, haiku, claude-opus-5-5 ...)
#     EFFORT   low | medium | high | xhigh | max
#     BRIEF    brief file to copy into the packet as brief.md
#     PROMPT   optional prompt file (default: a one-line instruction to read brief.md and follow it)
# Env: WSP_TOOLS (default "Bash Read Write Edit Glob Grep"), WSP_MAX_USD (optional --max-budget-usd),
#      WSP_DRY=1 (build the packet only, do not start Claude Code), WSP_SEARCH (default search_r2: outputs/weight_edit/shared/<it>), WSP_DOSSIER (default dossier_r3.md in shared/),
#      WSP_PRIOR (optional dir copied read-only into the packet as prior/; informed arm, doc Thread 2026-11-05 16:10).
#
# Packet (read-only for the proposer): outputs/weight_edit/sandbox/RUN_ID/
#   brief.md, dossier.md, server_spec.md (wsp_server docstring + LIMITS), search/ (calls.jsonl, files/, MANIFEST.txt),
#   wsp_kernel.py (absolute model path), scratch/ (rw)
# Run folder (rw): outputs/weight_edit/proposers/RUN_ID/  (requests/, work/, edit/, notebook.md, state.json)
# Output: RUN folder/transcript_<WSP_CONV>.jsonl (stream-json; WSP_CONV default 1), RUN folder/launcher.log
set -euo pipefail
REPO=/Users/ozgun/Dev/gemma-4-developer-agent-competition
RUN_ID=$1; MODEL=$2; EFFORT=$3; BRIEF=$4; PROMPT_FILE=${5:-}
PACKET=$REPO/outputs/weight_edit/sandbox/$RUN_ID
RUN=$REPO/outputs/weight_edit/proposers/$RUN_ID
MODELDIR=$REPO/data/models/gemma-4-e4b-it-qat-q4_0-unquantized
VENV=$REPO/outputs/small_model/mlx_venv
TMPD=/private/tmp/wsp_sbx_$RUN_ID
SB=$REPO/scripts/weight_edit/sandbox/proposer.sb
[ -d "$RUN/requests" ] || { echo "run folder $RUN missing: run wsp_server.py init first" >&2; exit 2; }
# packet
[ -d "$PACKET" ] && chmod -R u+w "$PACKET"
mkdir -p "$PACKET/search" "$PACKET/scratch" "$TMPD"
cp "$BRIEF" "$PACKET/brief.md"
SEARCH=$REPO/outputs/weight_edit/shared/${WSP_SEARCH:-search_r2}
cp "$REPO/outputs/weight_edit/shared/${WSP_DOSSIER:-dossier_r3.md}" "$PACKET/dossier.md"
cp "$SEARCH/calls.jsonl" "$PACKET/search/calls.jsonl"
[ -f "$SEARCH/calls_raw.jsonl" ] && cp "$SEARCH/calls_raw.jsonl" "$PACKET/search/calls_raw.jsonl"
[ -f "$SEARCH/MANIFEST.txt" ] && cp "$SEARCH/MANIFEST.txt" "$PACKET/search/"
rsync -a --delete "$SEARCH/files/" "$PACKET/search/files/"
if [ -n "${WSP_PRIOR:-}" ]; then
  [ -d "$PACKET/prior" ] && chmod -R u+w "$PACKET/prior"
  rsync -a --delete "$WSP_PRIOR/" "$PACKET/prior/"
  chmod -R a-w "$PACKET/prior"
fi
/usr/bin/python3 -I "$REPO/scripts/weight_edit/sandbox/make_spec.py" "$REPO/scripts/weight_edit/wsp_server.py" > "$PACKET/server_spec.md"
sed "s|^ROOT = Path(__file__).resolve().parents\[2\]|ROOT = Path(\"$REPO\")|" "$REPO/scripts/weight_edit/wsp_kernel.py" > "$PACKET/wsp_kernel.py"
grep -q "^ROOT = Path(\"$REPO\")" "$PACKET/wsp_kernel.py" || { echo "kernel patch failed" >&2; exit 3; }
chmod -R a-w "$PACKET/search" "$PACKET/brief.md" "$PACKET/dossier.md" "$PACKET/server_spec.md" "$PACKET/wsp_kernel.py"
[ -n "${WSP_DRY:-}" ] && { echo "dry run: packet built at $PACKET"; exit 0; }
# Claude Code's per-cwd state dir name (non-alphanumerics -> '-')
SLUG=$(printf '%s' "$PACKET" | sed 's/[^A-Za-z0-9]/-/g')
PROMPT="Read brief.md in the current directory and follow it exactly. Your run folder is $RUN."
[ -n "$PROMPT_FILE" ] && PROMPT=$(cat "$PROMPT_FILE")
TOOLS=${WSP_TOOLS:-"Bash Read Write Edit Glob Grep"}
EXTRA=()
[ -n "${WSP_MAX_USD:-}" ] && EXTRA+=(--max-budget-usd "$WSP_MAX_USD")
echo "[$(date -u +%FT%TZ)] launch $RUN_ID model=$MODEL effort=$EFFORT tools=[$TOOLS] packet=$PACKET" >> "$RUN/launcher.log"
cd "$PACKET"
# shellcheck disable=SC2086
env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" PATH="$VENV/bin:/usr/bin:/bin:/usr/sbin:/sbin" TERM=dumb LANG=en_US.UTF-8 \
    TMPDIR="$TMPD/" CLAUDE_CODE_TMPDIR="$TMPD" SHELL=/bin/bash \
  /usr/bin/sandbox-exec -f "$SB" -D HOME="$HOME" -D REPO="$REPO" -D PACKET="$PACKET" -D RUN="$RUN" -D MODEL="$MODELDIR" \
    -D VENV="$VENV" -D TMPD="$TMPD" -D SLUG="$SLUG" \
  "$HOME/.local/bin/claude" -p "$PROMPT" --model "$MODEL" --effort "$EFFORT" \
    --permission-mode bypassPermissions --setting-sources "" --strict-mcp-config --mcp-config '{"mcpServers":{}}' \
    --tools $TOOLS --add-dir "$RUN" --output-format stream-json --verbose ${EXTRA[@]+"${EXTRA[@]}"} \
  < /dev/null > "$RUN/transcript_${WSP_CONV:-1}.jsonl" 2>> "$RUN/launcher.log" && rc=0 || rc=$?
echo "[$(date -u +%FT%TZ)] exit rc=$rc" >> "$RUN/launcher.log"
exit $rc
