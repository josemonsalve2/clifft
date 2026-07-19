#!/bin/bash
# Profile hardware counters for a subset of sweep circuits.
# Uses rocprof --stats to capture VGPRs, SGPRs, scratch, occupancy.
#
# Usage:
#   ./scripts/profile_sweep.sh <approach> [gpu_partition]
#
# Output: results/profile_<approach>_<gpu>_<timestamp>.csv

set -euo pipefail

APPROACH="${1:-svm}"
PARTITION="${2:-mi300x}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="results"
OUTFILE="${OUTDIR}/profile_${APPROACH}_${PARTITION}_${TIMESTAMP}.csv"
BINARY="build-gpu/run_gpu"
SHOTS=500000

mkdir -p "$OUTDIR"

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

case "$PARTITION" in
    mi300x|mi300x) ARCH="gfx942" ;;
    mi325x)           ARCH="gfx942" ;;
    mi350x-es)        ARCH="gfx950" ;;
    *)                ARCH="gfx942" ;;
esac

echo "approach,partition,circuit,qubits,t_gates,peak_rank,kernel_name,arch_vgpr,accum_vgpr,sgpr,scratch,lds,duration_ns" > "$OUTFILE"

# Profile a representative subset: one circuit per T-gate count at q=17
CIRCUITS=(
    "sweep_q17_t0_d3.stim 17 0"
    "sweep_q17_t1_d3.stim 17 1"
    "sweep_q17_t2_d3.stim 17 2"
    "sweep_q17_t3_d3.stim 17 3"
    "sweep_q17_t4_d3.stim 17 4"
    "sweep_q17_t5_d3.stim 17 5"
    "sweep_q17_t8_d3.stim 17 8"
    "sweep_q17_t10_d3.stim 17 10"
    "sweep_q17_t15_d3.stim 17 15"
)

echo "=== Profiling: approach=$APPROACH partition=$PARTITION ==="

for ENTRY in "${CIRCUITS[@]}"; do
    read -r FNAME NQ NT <<< "$ENTRY"
    CIRCUIT="tests/fixtures/sweep/$FNAME"
    [ ! -f "$CIRCUIT" ] && continue

    echo -n "Profiling $FNAME ... "

    TMP_CSV="/tmp/rocprof_${APPROACH}_${FNAME%.stim}.csv"

    cd build-gpu
    rocprof --stats -o "$TMP_CSV" \
        ./run_gpu --circuit "../$CIRCUIT" --shots $SHOTS --arch "$ARCH" $FLAG \
        2>/dev/null || { echo "FAILED"; cd ..; continue; }
    cd ..

    # Parse rocprof CSV — find the main kernel (not __amd_rocclr_copyBuffer)
    if [ -f "$TMP_CSV" ]; then
        while IFS=',' read -r IDX KNAME GPU QID QIDX PID TID GRD WGR LDS SCR VGPR AVPR SGPR WAVE SIG OBJ DISP BEGIN END COMP DUR; do
            [[ "$KNAME" == *"__amd_rocclr"* ]] && continue
            [[ "$KNAME" == *"KernelName"* ]] && continue
            [[ -z "$KNAME" ]] && continue
            # Extract just the function name
            SHORT_NAME=$(echo "$KNAME" | sed 's/.*::\(.*\)(.*/\1/' | tr -d '"')
            PEAK=$(echo "$FNAME" | grep -oP 't\K[0-9]+')
            echo "$APPROACH,$PARTITION,$FNAME,$NQ,$NT,$PEAK,$SHORT_NAME,$VGPR,$AVPR,$SGPR,$SCR,$LDS,$DUR" >> "$OUTFILE"
            echo "vgpr=$VGPR sgpr=$SGPR scratch=$SCR dur=${DUR}ns"
            break
        done < "$TMP_CSV"
    else
        echo "no rocprof output"
    fi
done

echo ""
echo "=== Profile results: $OUTFILE ==="
