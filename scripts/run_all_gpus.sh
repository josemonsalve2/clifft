#!/bin/bash
# Launch multi_gpu_sweep.sh on every available GPU partition in parallel via srun,
# then merge results and generate comparison plots.
#
# Usage:
#   ./scripts/run_all_gpus.sh [approach1 approach2 ...]
#
# If no approaches given, defaults to: svm compiled
# Available approaches: svm, compiled, per-op, graph, split, persistent, svm-opt
#
# Results land in results/multi_gpu/ with plots in results/multi_gpu/plots/

set -euo pipefail

PROJ_ROOT="/shared/jmonsalv/quantum/clifft_rl/clifft"
SCRIPT="${PROJ_ROOT}/scripts/multi_gpu_sweep.sh"
OUTDIR="${PROJ_ROOT}/results/multi_gpu"
PLOTDIR="${OUTDIR}/plots"
SHOTS=1000000

# GPU partitions to sweep
PARTITIONS=(mi300x mi300x mi325x mi350x)

# Approaches (from args, or default to svm + compiled)
if [ $# -gt 0 ]; then
    APPROACHES=("$@")
else
    APPROACHES=(svm compiled)
fi

mkdir -p "$OUTDIR" "$PLOTDIR"

echo "============================================================"
echo "Multi-GPU benchmark sweep"
echo "  Partitions: ${PARTITIONS[*]}"
echo "  Approaches: ${APPROACHES[*]}"
echo "  Shots:      ${SHOTS}"
echo "  Output:     ${OUTDIR}"
echo "============================================================"
echo ""

# ---------- launch all jobs in parallel ----------
PIDS=()
JOBS=()

for PART in "${PARTITIONS[@]}"; do
    for APPR in "${APPROACHES[@]}"; do
        LOGFILE="${OUTDIR}/${PART}_${APPR}.log"
        echo "Launching: srun -p ${PART} --gres=gpu:1 -n1 -- ${SCRIPT} ${PART} ${APPR} ${SHOTS}"

        srun -p "$PART" --gres=gpu:1 -n1 --job-name="sweep_${PART}_${APPR}" \
             -- bash "$SCRIPT" "$PART" "$APPR" "$SHOTS" \
             > "$LOGFILE" 2>&1 &

        PIDS+=($!)
        JOBS+=("${PART}/${APPR}")
    done
done

echo ""
echo "Waiting for ${#PIDS[@]} jobs..."
echo ""

# ---------- wait and report ----------
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  DONE: ${JOBS[$i]}"
    else
        echo "  FAIL: ${JOBS[$i]} (see ${OUTDIR}/${JOBS[$i]//\//_}.log)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ $FAILED -gt 0 ]; then
    echo "WARNING: ${FAILED} job(s) failed. Check logs in ${OUTDIR}/"
fi

# ---------- merge per-sweep CSVs into combined files ----------
echo ""
echo "Merging CSV results..."

for SWEEP in tgate_sweep qubit_sweep depth_sweep; do
    MERGED="${OUTDIR}/all_${SWEEP}.csv"
    echo "gpu,approach,variable,value,peak_rank,shots_per_sec" > "$MERGED"
    for F in "${OUTDIR}"/*_"${SWEEP}".csv; do
        [ -f "$F" ] || continue
        # Skip header line from each file
        tail -n +2 "$F" >> "$MERGED"
    done
    ROWS=$(tail -n +2 "$MERGED" | wc -l)
    echo "  ${MERGED} (${ROWS} rows)"
done

# ---------- generate plots ----------
echo ""
echo "Generating plots..."
python3 "${PROJ_ROOT}/scripts/plot_results.py" --datadir "$OUTDIR" --plotdir "$PLOTDIR"

echo ""
echo "============================================================"
echo "All done. Results in:"
echo "  CSV:   ${OUTDIR}/"
echo "  Plots: ${PLOTDIR}/"
echo "============================================================"
