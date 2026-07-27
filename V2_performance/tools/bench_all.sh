#!/bin/bash
#SBATCH --job-name=v2benchall
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:55:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/V2_performance/runs/benchall_%j.log
#
# /benchmark-all: the full-corpus V2-vs-SVM characterization behind report §15.
#
# This script produced 20260726T182433Z_report-final-postdust and every run
# before it, but was never committed -- it lived in /tmp and was lost. That is
# exactly the failure mode VERIFIED_FACTS warns about, so it is now in-tree.
#
# Usage:
#   sbatch V2_performance/tools/bench_all.sh <label>
#
# Writes V2_performance/runs/<UTC>Z_<label>/ containing:
#   node.json      - which node, which arch. mi350x-es is HETEROGENEOUS; ratios
#                    from different nodes are NOT comparable (feedback_numa_bind).
#   manifest.json  - commit, branch, dirty flag, circuit count
#   raw/<circ>/<engine>/{kt,hsa,pmcA,pmcB,pmcC}/  - rocprofv3 CSVs, unaggregated
#   gpu/<circ>.json, summary.json, summary.md     - written by summarize_bench.py
#
# V2_SPECIALIZE=1 is MANDATORY and set here. Unset, run_v2 measures its own
# bytecode interpreter and reports a fake 2-3x regression (project_v2_specialize_env).
set -u

LABEL="${1:-unlabeled}"
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
cd "$BASE" || exit 1

for ROCM in /opt/rocm-7.2.3 /opt/rocm; do
  [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"
                          export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
export LLVM_PREFIX=/shared/jmonsalv/software/modules/llvm/upstream_05082025

# V2_NO_CPU=1 skips the f64 CPU reference: it is a correctness check, not part
# of the measurement, and it dominates profiler wall time at high rank.
export V2_NO_CPU=1
export V2_SPECIALIZE=1

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_${LABEL}"
RUN_DIR="$BASE/V2_performance/runs/$RUN_ID"
RAW="$RUN_DIR/raw"
mkdir -p "$RAW"
echo "RUN_DIR=$RUN_DIR"

cmake --build build-v2-nohip -j"$(nproc)" --target run_v2 2>&1 \
  | grep -iE "error:|Built target run_v2" | head

V2=build-v2-nohip/run_v2
SVM=build-gpu-mlir-mi355x/run_gpu

cat > "$RUN_DIR/node.json" <<EOF
{
  "slurm_job_id": ${SLURM_JOB_ID:-0},
  "node": "$(hostname -s)",
  "partition": "${SLURM_JOB_PARTITION:-unknown}",
  "arch": "$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+' || echo unknown)",
  "note": "mi350x-es is heterogeneous. Do NOT compare ratios across nodes."
}
EOF

cat > "$RUN_DIR/manifest.json" <<EOF
{
  "run_id": "$RUN_ID",
  "label": "$LABEL",
  "commit": "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)",
  "branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)",
  "dirty": $(git diff --quiet 2>/dev/null && echo false || echo true),
  "v2_specialize": true,
  "n_circuits": 26
}
EOF

# FETCH_SIZE/WRITE_SIZE are NOT collectable on this gfx950 config (rocprofv3
# aborts "Request exceeds the capabilities of the hardware"). TCC L2 traffic +
# SQ busy/wait cycles substitute. All three sets verified single-pass.
PMC_A="SQ_WAVES,SQ_INSTS_VALU,SQ_INSTS_MFMA,SQ_INSTS_SALU,SQ_INSTS_LDS"
PMC_B="TCC_HIT_sum,TCC_MISS_sum"
PMC_C="SQ_BUSY_CYCLES,GRBM_GUI_ACTIVE,SQ_WAIT_INST_LDS,SQ_WAVE_CYCLES"

# Shot counts fall with rank so each circuit lands in a comparable wall-time
# band; ratios are per-circuit so the counts need not match across rows.
CIRCS=(
"tests/fixtures/incremental/01_frame_only/frame_h.stim 20000"
"tests/fixtures/incremental/02_single_expand/four_t.stim 20000"
"tests/fixtures/large/circuit_d3_p0.001.stim 20000"
"tests/fixtures/large/surface_d7_t5.stim 20000"
"tests/fixtures/qv10.stim 20000"
"tests/fixtures/cultivation_d5.stim 20000"
"tests/fixtures/large/circuit_d5_p0.0005.stim 10000"
"tests/fixtures/large/circuit_d5_p0.001.stim 10000"
"tests/fixtures/large/circuit_d5_p0.002.stim 10000"
"tests/fixtures/large/circuit_d5_p0.003.stim 10000"
"tests/fixtures/large/circuit_d5_p0.005.stim 10000"
"tests/fixtures/large/surface_d7_t10.stim 10000"
"tests/fixtures/large/surface_d11_t10.stim 10000"
"tests/fixtures/large/surface_d7_t15.stim 10000"
"tests/fixtures/large/surface_d9_t10.stim 10000"
"tests/fixtures/large/surface_d7_t19.stim 5000"
"tests/fixtures/large/surface_d9_t15.stim 5000"
"tests/fixtures/large/surface_d9_t19.stim 5000"
"tests/fixtures/large/surface_d11_t15.stim 5000"
"tests/fixtures/large/surface_d11_t19.stim 5000"
"tools/bench/fixtures/qv20_seed42.stim 2000"
"tests/fixtures/large/qv20_L8_seed42.stim 2000"
"tests/fixtures/large/qv21_L8_seed42.stim 2000"
"tests/fixtures/large/qv22_L6_seed42.stim 1000"
"tests/fixtures/large/qv23_L5_seed42.stim 1000"
"tests/fixtures/large/qv24_L4_seed42.stim 500"
)

prof() {  # $1=outdir  rest=cmd
  local outdir="$1"; shift; local cmd="$*"
  mkdir -p "$outdir"
  eval "$cmd" >/dev/null 2>&1   # warm: compile+gate the .hsaco OFF the traced path
  timeout 300 rocprofv3 --kernel-trace   --stats --output-format csv -d "$outdir/kt"   -- $cmd >/dev/null 2>&1
  timeout 300 rocprofv3 --hsa-core-trace --stats --output-format csv -d "$outdir/hsa"  -- $cmd >/dev/null 2>&1
  timeout 300 rocprofv3 --pmc $PMC_A --output-format csv -d "$outdir/pmcA" -- $cmd >/dev/null 2>&1
  timeout 300 rocprofv3 --pmc $PMC_B --output-format csv -d "$outdir/pmcB" -- $cmd >/dev/null 2>&1
  timeout 300 rocprofv3 --pmc $PMC_C --output-format csv -d "$outdir/pmcC" -- $cmd >/dev/null 2>&1
}

for entry in "${CIRCS[@]}"; do
  c=$(echo "$entry" | awk '{print $1}'); s=$(echo "$entry" | awk '{print $2}')
  name=$(basename "$c" .stim)
  echo "=== $name (shots=$s) ==="
  # Fail loudly on a stale fixture path. Job 50785 lost 8 of 26 circuits to
  # renamed fixtures: rocprofv3 aborted on every pass, stderr went to
  # /dev/null, and the summary reported "wins 18/18" over a silently
  # truncated corpus. A missing input must not look like a clean result.
  if [ ! -f "$c" ]; then
    echo "  *** FIXTURE MISSING: $c -- aborting run ***" >&2
    exit 1
  fi
  prof "$RAW/$name/v2"  "$V2  --circuit $c --shots $s --seed 1"
  echo "  v2 done"
  prof "$RAW/$name/svm" "$SVM --circuit $c --shots $s --seed 1 --no-postselection"
  echo "  svm done"
done

python3 "$BASE/V2_performance/tools/summarize_bench.py" "$RUN_DIR"
echo "DONE $RUN_DIR"
