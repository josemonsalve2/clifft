#!/bin/bash
# probe_all.sh — measure COMPILED peak_rank for every live fixture.
# One circuit per process (rank_probe buffers stdout when redirected, and a
# per-circuit timeout keeps one pathological compile from stalling the sweep).
set -u
cd "$CLIFFT"
OUT="$1"
: > "$OUT"
find tests/fixtures tools/bench/fixtures docs/guide/circuits -name "*.stim" 2>/dev/null | sort \
  | xargs -P "$(nproc)" -I{} bash -c '
      r=$(timeout 120 ./V2_performance/scratch/rank_probe "{}" 2>/dev/null \
          | grep -oE "peak_rank=[0-9]+" | grep -oE "[0-9]+$")
      printf "%s %s\n" "${r:-TIMEOUT}" "{}"
    ' >> "$OUT"
echo "PROBE_DONE $(wc -l < "$OUT") circuits"
