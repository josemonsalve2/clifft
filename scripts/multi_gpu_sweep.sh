#!/bin/bash
# Multi-GPU benchmark sweep: vary ONE parameter at a time for comparison plots.
#
# Usage:
#   ./scripts/multi_gpu_sweep.sh <partition> <approach> [shots]
#
# Partition: mi300x, mi300x-es, mi325x, mi350x-es
# Approach:  svm, compiled, per-op, graph, split, persistent, svm-opt
#
# Produces three CSV files in results/multi_gpu/:
#   <partition>_<approach>_tgate_sweep.csv   (fixed q=17, d=3, vary t)
#   <partition>_<approach>_qubit_sweep.csv   (fixed t=2, d=3, vary q)
#   <partition>_<approach>_depth_sweep.csv   (fixed q=17, t=2, vary d)
#
# Columns: gpu,approach,variable,value,peak_rank,shots_per_sec

set -euo pipefail

PARTITION="${1:?Usage: $0 <partition> <approach> [shots]}"
APPROACH="${2:?Usage: $0 <partition> <approach> [shots]}"
SHOTS="${3:-1000000}"

PROJ_ROOT="/shared/jmonsalv/quantum/clifft_rl/clifft"
CIRCUIT_DIR="${PROJ_ROOT}/tests/fixtures/sweep"
BINARY="${PROJ_ROOT}/build-gpu/run_gpu"
OUTDIR="${PROJ_ROOT}/results/multi_gpu"

mkdir -p "$OUTDIR"

# ---------- approach -> CLI flag ----------
case "$APPROACH" in
    svm)        FLAG="--interpreted" ;;
    compiled)   FLAG="--compiled" ;;
    per-op)     FLAG="--per-op" ;;
    graph)      FLAG="--graph" ;;
    split)      FLAG="--split" ;;
    persistent) FLAG="--persistent" ;;
    svm-opt)    FLAG="--svm-opt" ;;
    *)          echo "Unknown approach: $APPROACH"; exit 1 ;;
esac

# ---------- partition -> GPU arch ----------
case "$PARTITION" in
    mi300x|mi300x-es) ARCH="gfx942" ;;
    mi325x)           ARCH="gfx942" ;;  # same ISA as MI300X
    mi350x-es)        ARCH="gfx950" ;;
    *)                echo "Unknown partition: $PARTITION"; exit 1 ;;
esac

# ---------- helper: run one circuit, emit CSV row ----------
run_one() {
    local GPU="$1" APPR="$2" VAR="$3" VAL="$4" CIRCUIT="$5" OUTFILE="$6"

    if [ ! -f "$CIRCUIT" ]; then
        echo "  SKIP (not found): $CIRCUIT"
        echo "${GPU},${APPR},${VAR},${VAL},-1,0" >> "$OUTFILE"
        return
    fi

    echo -n "  ${VAR}=${VAL} ... "

    OUTPUT=$("$BINARY" --circuit "$CIRCUIT" --shots "$SHOTS" --arch "$ARCH" $FLAG 2>&1) || {
        echo "FAILED"
        echo "${GPU},${APPR},${VAR},${VAL},-1,0" >> "$OUTFILE"
        return
    }

    PEAK_RANK=$(echo "$OUTPUT" | grep '"peak_rank"' | head -1 | grep -o '[0-9]*' || echo "-1")
    SPS=$(echo "$OUTPUT" | grep '"shots_per_second_sampling_only"' | head -1 | grep -oE '[0-9]+\.?[0-9eE+\-]*' || echo "0")

    echo "${SPS} shots/s (rank=${PEAK_RANK})"
    echo "${GPU},${APPR},${VAR},${VAL},${PEAK_RANK},${SPS}" >> "$OUTFILE"
}

# ============================================================
# Sweep 1: T-gate sweep (fixed q=17, d=3, vary t)
# ============================================================
T_GATES=(0 1 2 3 4 5 8 10)
OUTFILE_T="${OUTDIR}/${PARTITION}_${APPROACH}_tgate_sweep.csv"
echo "gpu,approach,variable,value,peak_rank,shots_per_sec" > "$OUTFILE_T"

echo "=== T-gate sweep: partition=${PARTITION} approach=${APPROACH} (q=17, d=3) ==="
for T in "${T_GATES[@]}"; do
    CIRCUIT="${CIRCUIT_DIR}/sweep_q17_t${T}_d3.stim"
    run_one "$PARTITION" "$APPROACH" "t_gates" "$T" "$CIRCUIT" "$OUTFILE_T"
done
echo ""

# ============================================================
# Sweep 2: Qubit sweep (fixed t=2, d=3, vary q)
# ============================================================
QUBITS=(5 9 17 33 65)
OUTFILE_Q="${OUTDIR}/${PARTITION}_${APPROACH}_qubit_sweep.csv"
echo "gpu,approach,variable,value,peak_rank,shots_per_sec" > "$OUTFILE_Q"

echo "=== Qubit sweep: partition=${PARTITION} approach=${APPROACH} (t=2, d=3) ==="
for Q in "${QUBITS[@]}"; do
    CIRCUIT="${CIRCUIT_DIR}/sweep_q${Q}_t2_d3.stim"
    run_one "$PARTITION" "$APPROACH" "qubits" "$Q" "$CIRCUIT" "$OUTFILE_Q"
done
echo ""

# ============================================================
# Sweep 3: Depth sweep (fixed q=17, t=2, vary d)
# ============================================================
DEPTHS=(1 3 5 10)
OUTFILE_D="${OUTDIR}/${PARTITION}_${APPROACH}_depth_sweep.csv"
echo "gpu,approach,variable,value,peak_rank,shots_per_sec" > "$OUTFILE_D"

echo "=== Depth sweep: partition=${PARTITION} approach=${APPROACH} (q=17, t=2) ==="
for D in "${DEPTHS[@]}"; do
    CIRCUIT="${CIRCUIT_DIR}/sweep_q17_t2_d${D}.stim"
    run_one "$PARTITION" "$APPROACH" "depth" "$D" "$CIRCUIT" "$OUTFILE_D"
done
echo ""

echo "=== Done: ${PARTITION} / ${APPROACH} ==="
echo "Results:"
echo "  $OUTFILE_T"
echo "  $OUTFILE_Q"
echo "  $OUTFILE_D"
