#!/usr/bin/env bash
# One-click: batch MATLAB + MATPOWER PF, no desktop window.
#
# Optional: case name passed to runpf/rundcpf (MATPOWER resolves it on path).
#   - Local: case14.m, case33bw.m in this folder
#   - Built-in: e.g. case118 (MATPOWER data/case118.m) — no copy needed here
#   CASE=case118 ./start_sim.sh
#   CASE=case33bw ./start_sim.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CASE="${CASE:-case14}"

# Default: detect common macOS install (override with env MATLAB_PATH)
if [[ -z "${MATLAB_PATH:-}" ]]; then
  if [[ -x "/Applications/MATLAB_R2025b.app/bin/matlab" ]]; then
    MATLAB_PATH="/Applications/MATLAB_R2025b.app/bin/matlab"
  elif command -v matlab >/dev/null 2>&1; then
    MATLAB_PATH="$(command -v matlab)"
  else
    echo "ERROR: Set MATLAB_PATH to your matlab binary, e.g." >&2
    echo "  export MATLAB_PATH=/Applications/MATLAB_R2025b.app/bin/matlab" >&2
    exit 1
  fi
fi

OUT_PF="$SCRIPT_DIR/output_pf_${CASE}.txt"
OUT_DCPF="$SCRIPT_DIR/output_dcpf_${CASE}.txt"

echo "Using MATLAB: $MATLAB_PATH"
echo "Case:         $CASE"
echo "Working dir:  $SCRIPT_DIR"
echo "Outputs:"
echo "  $OUT_PF"
echo "  $OUT_DCPF"

# MATLAB starts in $HOME; cd so *.m resolve here. Pass CASE into run_my_pf.
"$MATLAB_PATH" -nodesktop -nosplash -batch "cd('${SCRIPT_DIR//\'/\'\'}'); run_my_pf('${CASE}');"

echo "Done."
