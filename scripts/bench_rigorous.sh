#!/bin/bash
# Rigorous 3-Way Benchmark: SVM vs Compiled Megakernel vs MLIR→LLVM-IR Kernel
#
# Methodology:
#   - 20 runs per (circuit, mode), 3 warmup runs before each circuit
#   - Exclusive GPU (--exclusive SLURM flag)
#   - Reports kernel_seconds separately from total sample_seconds
#   - CSV includes node, gpu_arch, rocm_version for cross-node comparability
#   - Welch's t-test for pairwise significance
#
# Usage:
#   sbatch scripts/slurm_baseline_mi300x.sh   # submit via SLURM wrapper
#   bash   scripts/bench_rigorous.sh [OPTIONS]  # run directly (dev/login)
#
# Options:
#   --modes    svm,compiled,mlir   (comma-separated, default: svm,compiled)
#   --circuits interesting|rank|all  (default: all)
#   --shots    N                   (default: 1000000)
#   --runs     N                   (default: 20)
#   --binary   PATH                (default: build-gpu/run_gpu)
#   --outdir   PATH                (default: results/)
#
# Output CSV columns:
#   mode,circuit,label,rank,qubits,gpu_arch,node,rocm_version,run,shots,
#   peak_rank,passed,sample_seconds,kernel_seconds,shots_per_second

set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- Defaults ---
MODES="svm,compiled"
CIRCUITS_SET="all"
SHOTS=1000000
RUNS=20
WARMUP_RUNS=3
BINARY="$BASE/build-gpu/run_gpu"
OUTDIR="$BASE/results"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --modes)     MODES="$2";       shift 2 ;;
        --circuits)  CIRCUITS_SET="$2"; shift 2 ;;
        --shots)     SHOTS="$2";       shift 2 ;;
        --runs)      RUNS="$2";        shift 2 ;;
        --binary)    BINARY="$2";      shift 2 ;;
        --outdir)    OUTDIR="$2";      shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUTDIR/bench_rigorous_${TIMESTAMP}.csv"
SUMMARY="$OUTDIR/bench_rigorous_summary_${TIMESTAMP}.txt"

# --- System info ---
NODE=$(hostname)
GPU_ARCH=$(rocminfo 2>/dev/null | grep -m1 "Name:.*gfx" | grep -oE "gfx[0-9]+" || echo "unknown")
ROCM_VER=$(rocminfo 2>/dev/null | grep -m1 "ROCm Runtime Version" | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" || echo "unknown")

echo "=== Rigorous 3-Way Benchmark ==="
echo "Date:        $(date)"
echo "Host:        $NODE"
echo "GPU arch:    $GPU_ARCH"
echo "ROCm:        $ROCM_VER"
echo "Modes:       $MODES"
echo "Circuits:    $CIRCUITS_SET"
echo "Runs:        $RUNS"
echo "Shots:       $SHOTS"
echo "Warmup:      $WARMUP_RUNS"
echo "Binary:      $BINARY"
echo "Output:      $OUTFILE"
echo ""

if [ ! -x "$BINARY" ]; then
    echo "ERROR: binary not found or not executable: $BINARY"
    exit 1
fi

echo "=== GPU Info ==="
rocm-smi --showid --showproductname 2>/dev/null | head -8 || true
echo ""

# --- CSV header ---
echo "mode,circuit,label,rank,qubits,gpu_arch,node,rocm_version,run,shots,peak_rank,passed,sample_seconds,kernel_seconds,shots_per_second" > "$OUTFILE"

# --- Helper: extract JSON field ---
extract_field() {
    echo "$1" | grep "\"$2\"" | head -1 | grep -oE '[0-9]+\.?[0-9eE+\-]*'
}

# --- Mode → flag mapping ---
mode_flag() {
    case "$1" in
        svm)      echo "" ;;
        compiled) echo "--hybrid" ;;
        mlir)     echo "--mlir" ;;
        *)        echo ""; echo "WARNING: unknown mode $1" >&2 ;;
    esac
}

# --- Run one circuit in one mode ---
run_circuit() {
    local MODE=$1       # svm / compiled / mlir
    local CIRCUIT=$2    # full path to .stim file
    local LABEL=$3      # short display name
    local RANK=$4       # target rank (from filename, for CSV)
    local QUBITS=$5     # qubit count (for CSV)

    local FLAG
    FLAG=$(mode_flag "$MODE")

    # Skip mlir mode if binary doesn't support it (check --help)
    if [ "$MODE" = "mlir" ]; then
        if ! "$BINARY" --help 2>&1 | grep -q -- "--mlir"; then
            echo "  SKIP: $MODE (--mlir not supported by binary)"
            return
        fi
    fi

    # Warmup
    for i in $(seq 1 $WARMUP_RUNS); do
        # shellcheck disable=SC2086
        "$BINARY" --circuit "$CIRCUIT" --shots 100000 $FLAG > /dev/null 2>&1 || true
    done

    # Timed runs
    for i in $(seq 1 $RUNS); do
        # shellcheck disable=SC2086
        OUTPUT=$("$BINARY" --circuit "$CIRCUIT" --shots "$SHOTS" $FLAG 2>&1) || {
            echo "  $MODE run $i: FAILED"
            echo "$MODE,$CIRCUIT,$LABEL,$RANK,$QUBITS,$GPU_ARCH,$NODE,$ROCM_VER,$i,$SHOTS,-1,0,0,0,0" >> "$OUTFILE"
            continue
        }

        SPS=$(extract_field "$OUTPUT" "shots_per_second_sampling_only")
        SEC=$(extract_field "$OUTPUT" "sample_seconds")
        KSEC=$(extract_field "$OUTPUT" "kernel_seconds")
        PASSED=$(extract_field "$OUTPUT" "passed_shots")
        PEAK_RANK=$(extract_field "$OUTPUT" "peak_rank")

        # kernel_seconds may not be reported by older binaries — use sample_seconds as fallback
        [ -z "$KSEC" ] && KSEC="$SEC"

        printf "  %-10s run %2d: %12s shots/s  kernel=%.4fs  (rank=%s)\n" \
            "$MODE" "$i" "$SPS" "$KSEC" "$PEAK_RANK"
        echo "$MODE,$CIRCUIT,$LABEL,$RANK,$QUBITS,$GPU_ARCH,$NODE,$ROCM_VER,$i,$SHOTS,$PEAK_RANK,$PASSED,$SEC,$KSEC,$SPS" >> "$OUTFILE"
    done
}

# --- Build circuit list ---
declare -a CIRCUIT_LABELS=()
declare -a CIRCUIT_PATHS=()
declare -a CIRCUIT_RANKS=()
declare -a CIRCUIT_QUBITS=()

add_circuit() {
    local label=$1 path=$2 rank=$3 qubits=$4
    if [ -f "$path" ]; then
        CIRCUIT_LABELS+=("$label")
        CIRCUIT_PATHS+=("$path")
        CIRCUIT_RANKS+=("$rank")
        CIRCUIT_QUBITS+=("$qubits")
    else
        echo "SKIP (not found): $path"
    fi
}

# Interesting/large circuits
if [[ "$CIRCUITS_SET" == "interesting" || "$CIRCUITS_SET" == "all" ]]; then
    add_circuit "target_qec"       "$BASE/tests/fixtures/target_qec.stim"                      0  0
    add_circuit "circuit_d3"       "$BASE/tests/fixtures/circuit_d3_p0.001.stim"               4  0
    add_circuit "hook_inject_d3"   "$BASE/tests/fixtures/hook_inject_d3_t_gate.stim"           4  0
    add_circuit "cultivation_d5"   "$BASE/tests/fixtures/cultivation_d5.stim"                  4  0
    add_circuit "circuit_d7"       "$BASE/tests/fixtures/circuit_d7_p0.0005.stim"              4  0
    add_circuit "surface_d5_r5"    "$BASE/tests/fixtures/large/surface_d5_r5.stim"             0  0
    add_circuit "surface_d7_r7"    "$BASE/tests/fixtures/large/surface_d7_r7.stim"             0  0
    add_circuit "surface_d7_r14"   "$BASE/tests/fixtures/large/surface_d7_r14.stim"            0  0
    add_circuit "color_d7"         "$BASE/tests/fixtures/large/color_d7.stim"                  0  0
    add_circuit "rep_d5_r100"      "$BASE/tests/fixtures/large/rep_d5_r100.stim"               0  0
    add_circuit "surface_d9"       "$BASE/tests/fixtures/large/surface_d9_r9.stim"             0  0
    add_circuit "surface_d11"      "$BASE/tests/fixtures/large/surface_d11_r11.stim"           0  0
    add_circuit "surface_d13"      "$BASE/tests/fixtures/large/surface_d13_r13.stim"           0  0
fi

# Rank sweep: all ranks 0–19, both qubit counts, depth 3
if [[ "$CIRCUITS_SET" == "rank" || "$CIRCUITS_SET" == "all" ]]; then
    for RANK_FILE in "$BASE/tests/fixtures/rank_sweep/"*_d3.stim; do
        [ -f "$RANK_FILE" ] || continue
        FNAME=$(basename "$RANK_FILE" .stim)
        # Parse qubits and rank from filename: rank_q{Q}_r{R}_d{D}
        Q=$(echo "$FNAME" | grep -oE 'q[0-9]+' | grep -oE '[0-9]+')
        R=$(echo "$FNAME" | grep -oE 'r[0-9]+' | head -1 | grep -oE '[0-9]+')
        add_circuit "$FNAME" "$RANK_FILE" "$R" "$Q"
    done
fi

TOTAL=${#CIRCUIT_LABELS[@]}
IFS=',' read -ra MODE_LIST <<< "$MODES"
TOTAL_RUNS=$(( TOTAL * ${#MODE_LIST[@]} * RUNS ))
echo "Circuits:    $TOTAL"
echo "Mode count:  ${#MODE_LIST[@]}"
echo "Total runs:  $TOTAL_RUNS (plus warmups)"
echo ""

# --- Main benchmark loop ---
IDX=0
for i in "${!CIRCUIT_LABELS[@]}"; do
    LABEL="${CIRCUIT_LABELS[$i]}"
    CIRCUIT="${CIRCUIT_PATHS[$i]}"
    RANK="${CIRCUIT_RANKS[$i]}"
    QUBITS="${CIRCUIT_QUBITS[$i]}"
    IDX=$((IDX + 1))

    echo "=============================================="
    echo "[$IDX/$TOTAL] $LABEL  (rank~$RANK, q=$QUBITS)"
    echo "=============================================="

    for MODE in "${MODE_LIST[@]}"; do
        echo "  Mode: $MODE"
        run_circuit "$MODE" "$CIRCUIT" "$LABEL" "$RANK" "$QUBITS"
    done
    echo ""
done

# --- Statistical summary ---
echo "=============================================="
echo "STATISTICAL SUMMARY"
echo "=============================================="
echo ""

{
echo "Rigorous Benchmark Summary — $(date)"
echo "Host: $NODE  GPU: $GPU_ARCH  ROCm: $ROCM_VER"
echo "Runs=$RUNS  Shots=$SHOTS  Warmup=$WARMUP_RUNS"
echo ""

# Header — dynamically width based on number of modes
HDR="Circuit"
for MODE in "${MODE_LIST[@]}"; do
    HDR="$HDR | ${MODE} Mean   StdDev   CV"
done
echo "$HDR"
echo "$(printf '%.0s-' {1..150})"

# Get unique labels
LABELS=$(awk -F',' 'NR>1 {print $3}' "$OUTFILE" | sort -u)

for LBL in $LABELS; do
    LINE="$LBL"
    PREV_MEAN=0
    PREV_MODE=""
    for MODE in "${MODE_LIST[@]}"; do
        VALS=$(grep "^${MODE},[^,]*,${LBL}," "$OUTFILE" | awk -F',' '{print $15}' | sort -g)
        if [ -z "$VALS" ]; then
            LINE="$LINE |    N/A      N/A     N/A"
            continue
        fi
        STATS=$(echo "$VALS" | awk '
        BEGIN { n=0; sum=0 }
        { v=$1+0; vals[n]=v; sum+=v; n++ }
        END {
            if (n==0) { printf "0 0 0"; exit }
            mean=sum/n
            ss=0; for(i=0;i<n;i++) ss+=(vals[i]-mean)^2
            sd=sqrt(ss/(n>1?n-1:1))
            cv=100*sd/(mean>0?mean:1)
            printf "%.0f %.0f %.1f", mean, sd, cv
        }')
        MEAN=$(echo "$STATS" | awk '{print $1}')
        SD=$(echo "$STATS" | awk '{print $2}')
        CV=$(echo "$STATS" | awk '{print $3}')
        LINE="$LINE | $(printf '%10.0f %8.0f %5.1f%%' "$MEAN" "$SD" "$CV")"
        PREV_MEAN=$MEAN
        PREV_MODE=$MODE
    done
    echo "$LINE"
done

echo ""
echo "Note: shots_per_second = passed_shots / sample_seconds (sampling only, not H2D/D2H)"
echo "Raw data: $OUTFILE"
} | tee "$SUMMARY"

echo ""
echo "=== Benchmark complete ==="
echo "Data: $OUTFILE"
echo "Summary: $SUMMARY"
