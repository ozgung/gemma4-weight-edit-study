#!/bin/bash
# Fill the round-3 brief template for one run.  Usage: make_brief_r3.sh RUN_ID PROPOSER MODEL EFFORT BUDGET OUT
set -euo pipefail
REPO=/Users/ozgun/Dev/gemma-4-developer-agent-competition
sed -e "s|{RUN_DIR}|$REPO/outputs/weight_edit/proposers/$1|g" -e "s|{PROPOSER}|$2|g" -e "s|{MODEL}|$3|g" \
    -e "s|{EFFORT}|$4|g" -e "s|{BUDGET}|$5|g" -e "s|{PACKET}|$REPO/outputs/weight_edit/sandbox/$1|g" \
    -e "s|{VENV}|$REPO/outputs/small_model/mlx_venv|g" -e "s|{MODEL_DIR}|$REPO/data/models/gemma-4-e4b-it-qat-q4_0-unquantized|g" \
    "$REPO/scripts/weight_edit/wsp_brief_r3.md" > "$6"
! grep -n "{[A-Z_]*}" "$6"
