#!/bin/bash
#SBATCH --job-name=rocprof-mlir
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/rocprof_mlir_%j.log

# ============================================================================
# rocprofv3 kernel-trace comparison: SVM vs MLIR
# Measures pure GPU kernel execution time (no host overhead) for both paths.
# ============================================================================

set +e   # Don't abort on first failure; we want both runs even if one fails.

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BINARY="$BASE/build-gpu-mlir-mi355x/run_gpu"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="$BASE/results/rocprof_mlir_${TIMESTAMP}"
mkdir -p "$RESULTS"

SHOTS=100000
SEED=42

# --- ROCm environment ---
for ROCM_CANDIDATE in /opt/rocm-7.2.3 /opt/rocm-7.0.0 /opt/rocm /opt/rocm-*; do
    if [ -d "$ROCM_CANDIDATE/lib" ]; then
        export ROCM_PATH="$ROCM_CANDIDATE"
        export PATH="$ROCM_CANDIDATE/bin:$PATH"
        export LD_LIBRARY_PATH="$ROCM_CANDIDATE/lib:${LD_LIBRARY_PATH:-}"
        echo "Using ROCm: $ROCM_CANDIDATE"
        break
    fi
done

# --- LLVM environment (needed for MLIR codegen) ---
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    if [ -d "$LLVM/bin" ]; then
        export LLVM_PREFIX="$LLVM"
        export PATH="$LLVM/bin:$PATH"
        echo "Using LLVM: $LLVM"
        break
    fi
done

# --- Locate rocprofv3 or rocprof ---
ROCPROF=""
ROCPROF_FLAGS=""
for candidate in /opt/rocm*/bin/rocprofv3; do
    if [ -x "$candidate" ]; then
        ROCPROF="$candidate"
        ROCPROF_FLAGS="--kernel-trace"
        echo "Found rocprofv3: $ROCPROF"
        break
    fi
done
if [ -z "$ROCPROF" ]; then
    for candidate in /opt/rocm*/bin/rocprof; do
        if [ -x "$candidate" ]; then
            ROCPROF="$candidate"
            ROCPROF_FLAGS="--hip-trace --hsa-trace"
            echo "Found rocprof (v2): $ROCPROF"
            break
        fi
    done
fi
if [ -z "$ROCPROF" ]; then
    echo "FATAL: Neither rocprofv3 nor rocprof found in /opt/rocm*/bin/"
    exit 1
fi

echo "=================================================================="
echo "rocprof kernel-trace: SVM vs MLIR"
echo "Node: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Date: $(date)"
echo "Binary: $BINARY"
echo "Shots: $SHOTS  Seed: $SEED"
echo "=================================================================="

# Check binary
if [ ! -x "$BINARY" ]; then
    echo "FATAL: Binary not found: $BINARY"
    exit 1
fi

# GPU info
rocm-smi --showid --showproductname 2>/dev/null | head -8 || true
echo ""

# ============================================================================
# parse_kernel_trace DIR LABEL
#   Finds the kernel-trace CSV in DIR, prints a formatted table, and echoes
#   the total kernel duration in nanoseconds to stdout (last line).
# ============================================================================
parse_kernel_trace() {
    local DIR="$1"
    local LABEL="$2"

    # rocprofv3 writes results into a subdirectory; find the CSV
    local CSV=""
    CSV=$(find "$DIR" -name "*kernel_trace*" -o -name "*kernel*stats*" 2>/dev/null | grep -i "\.csv" | head -1)
    if [ -z "$CSV" ]; then
        # Fallback: any CSV with "trace" in the name
        CSV=$(find "$DIR" -name "*.csv" 2>/dev/null | head -1)
    fi
    if [ -z "$CSV" ]; then
        echo "  WARNING: No kernel-trace CSV found in $DIR"
        echo "  Contents:" && ls -R "$DIR" 2>/dev/null | head -20
        echo "TOTAL_NS=0"
        return
    fi

    echo "  Trace CSV: $CSV"
    echo ""

    # Detect column layout.  rocprofv3 kernel-trace CSV has columns like:
    #   "Kind","Agent_Id","Queue_Id","Kernel_Id","Kernel_Name","Correlation_Id",
    #   "Start_Timestamp","End_Timestamp","Private_Segment_Size","Group_Segment_Size",
    #   "Workgroup_Size_X","Workgroup_Size_Y","Workgroup_Size_Z",
    #   "Grid_Size_X","Grid_Size_Y","Grid_Size_Z"
    # Duration = End_Timestamp - Start_Timestamp (nanoseconds).
    #
    # We need to handle both rocprofv3 and rocprof v2 CSV formats.
    local HEADER
    HEADER=$(head -1 "$CSV" | tr -d '"')

    # Find column indices (0-based)
    local col_name col_start col_end col_grid_x col_grid_y col_grid_z col_wg_x col_wg_y col_wg_z
    col_name=$(echo "$HEADER" | tr ',' '\n' | grep -n -i "kernel_name\|KernelName\|Name" | head -1 | cut -d: -f1)
    col_start=$(echo "$HEADER" | tr ',' '\n' | grep -n -i "start_timestamp\|BeginNs\|Start" | head -1 | cut -d: -f1)
    col_end=$(echo "$HEADER" | tr ',' '\n' | grep -n -i "end_timestamp\|EndNs\|End" | head -1 | cut -d: -f1)
    col_grid_x=$(echo "$HEADER" | tr ',' '\n' | grep -n -i "grid_size_x\|grd" | head -1 | cut -d: -f1)
    col_grid_y=$(echo "$HEADER" | tr ',' '\n' | grep -n -i "grid_size_y" | head -1 | cut -d: -f1)
    col_grid_z=$(echo "$HEADER" | tr ',' '\n' | grep -n -i "grid_size_z" | head -1 | cut -d: -f1)
    col_wg_x=$(echo "$HEADER" | tr ',' '\n' | grep -n -i "workgroup_size_x\|wgr" | head -1 | cut -d: -f1)
    col_wg_y=$(echo "$HEADER" | tr ',' '\n' | grep -n -i "workgroup_size_y" | head -1 | cut -d: -f1)
    col_wg_z=$(echo "$HEADER" | tr ',' '\n' | grep -n -i "workgroup_size_z" | head -1 | cut -d: -f1)

    if [ -z "$col_name" ] || [ -z "$col_start" ] || [ -z "$col_end" ]; then
        echo "  WARNING: Could not identify columns in CSV header:"
        echo "  $HEADER"
        echo "  Dumping raw CSV (first 30 lines):"
        head -30 "$CSV"
        echo "TOTAL_NS=0"
        return
    fi

    echo "  --- $LABEL Kernel Launches ---"
    printf "  %-60s %14s %14s  %s\n" "Kernel" "Duration(ns)" "Duration(us)" "Grid / Block"
    printf "  %-60s %14s %14s  %s\n" \
        "------------------------------------------------------------" \
        "--------------" "--------------" "--------------------"

    local TOTAL_NS=0
    local COUNT=0

    # Process data rows (skip header)
    while IFS=',' read -r line; do
        # Skip header and empty lines
        [ -z "$line" ] && continue

        # Split into fields, stripping quotes
        local fields
        IFS=',' read -ra fields <<< "$(echo "$line" | tr -d '"')"

        local kname="${fields[$((col_name-1))]}"
        local ts_start="${fields[$((col_start-1))]}"
        local ts_end="${fields[$((col_end-1))]}"

        # Skip non-numeric timestamps
        [[ "$ts_start" =~ ^[0-9]+$ ]] || continue
        [[ "$ts_end" =~ ^[0-9]+$ ]] || continue

        local dur_ns=$((ts_end - ts_start))
        local dur_us
        dur_us=$(awk "BEGIN{printf \"%.2f\", $dur_ns/1000.0}")

        # Grid/block info
        local grid_str="n/a"
        if [ -n "$col_grid_x" ] && [ -n "$col_wg_x" ]; then
            local gx="${fields[$((col_grid_x-1))]:-1}"
            local gy="${fields[$((col_grid_y-1))]:-1}"
            local gz="${fields[$((col_grid_z-1))]:-1}"
            local wx="${fields[$((col_wg_x-1))]:-1}"
            local wy="${fields[$((col_wg_y-1))]:-1}"
            local wz="${fields[$((col_wg_z-1))]:-1}"
            grid_str="grid(${gx}x${gy}x${gz}) block(${wx}x${wy}x${wz})"
        fi

        # Truncate long kernel names for display
        local kshort="$kname"
        if [ ${#kshort} -gt 58 ]; then
            kshort="${kshort:0:55}..."
        fi

        printf "  %-60s %14d %14s  %s\n" "$kshort" "$dur_ns" "$dur_us" "$grid_str"
        TOTAL_NS=$((TOTAL_NS + dur_ns))
        COUNT=$((COUNT + 1))
    done < <(tail -n +2 "$CSV")

    echo ""
    printf "  %-60s %14d %14s  (%d launches)\n" \
        "TOTAL" "$TOTAL_NS" "$(awk "BEGIN{printf \"%.2f\", $TOTAL_NS/1000.0}")" "$COUNT"
    echo ""
    echo "TOTAL_NS=$TOTAL_NS"
}

# ============================================================================
# profile_circuit CIRCUIT_PATH CIRCUIT_LABEL
#   Runs rocprofv3 for both SVM and MLIR, then compares.
# ============================================================================
profile_circuit() {
    local CIRCUIT="$1"
    local LABEL="$2"

    if [ ! -f "$CIRCUIT" ]; then
        echo "SKIP: Circuit not found: $CIRCUIT"
        return
    fi

    echo ""
    echo "=================================================================="
    echo "Circuit: $LABEL"
    echo "Path:    $CIRCUIT"
    echo "=================================================================="

    # ---- SVM run ----
    local SVM_DIR="$RESULTS/${LABEL}_svm"
    mkdir -p "$SVM_DIR"
    echo ""
    echo ">>> SVM path (no --mlir)"
    echo "  Command: $ROCPROF $ROCPROF_FLAGS -o $SVM_DIR -- $BINARY --circuit $CIRCUIT --shots $SHOTS --seed $SEED"
    echo ""

    $ROCPROF $ROCPROF_FLAGS -o "$SVM_DIR" -- \
        "$BINARY" --circuit "$CIRCUIT" --shots "$SHOTS" --seed "$SEED" \
        2>&1 | tee "$SVM_DIR/stdout.log"
    local SVM_RC=${PIPESTATUS[0]}
    echo "  Exit code: $SVM_RC"

    # ---- MLIR run ----
    local MLIR_DIR="$RESULTS/${LABEL}_mlir"
    mkdir -p "$MLIR_DIR"
    echo ""
    echo ">>> MLIR path (--mlir)"
    echo "  Command: $ROCPROF $ROCPROF_FLAGS -o $MLIR_DIR -- $BINARY --circuit $CIRCUIT --shots $SHOTS --seed $SEED --mlir"
    echo ""

    $ROCPROF $ROCPROF_FLAGS -o "$MLIR_DIR" -- \
        "$BINARY" --circuit "$CIRCUIT" --shots "$SHOTS" --seed "$SEED" --mlir \
        2>&1 | tee "$MLIR_DIR/stdout.log"
    local MLIR_RC=${PIPESTATUS[0]}
    echo "  Exit code: $MLIR_RC"

    # ---- Parse and compare ----
    echo ""
    echo "--- Parsing SVM kernel trace ---"
    local SVM_RESULT
    SVM_RESULT=$(parse_kernel_trace "$SVM_DIR" "SVM")
    echo "$SVM_RESULT" | grep -v "^TOTAL_NS="
    local SVM_NS
    SVM_NS=$(echo "$SVM_RESULT" | grep "^TOTAL_NS=" | tail -1 | cut -d= -f2)
    SVM_NS=${SVM_NS:-0}

    echo ""
    echo "--- Parsing MLIR kernel trace ---"
    local MLIR_RESULT
    MLIR_RESULT=$(parse_kernel_trace "$MLIR_DIR" "MLIR")
    echo "$MLIR_RESULT" | grep -v "^TOTAL_NS="
    local MLIR_NS
    MLIR_NS=$(echo "$MLIR_RESULT" | grep "^TOTAL_NS=" | tail -1 | cut -d= -f2)
    MLIR_NS=${MLIR_NS:-0}

    # ---- Comparison ----
    echo "=================================================================="
    echo "  COMPARISON: $LABEL"
    echo "=================================================================="
    printf "  %-20s %14s %14s\n" "Path" "Total GPU (ns)" "Total GPU (us)"
    printf "  %-20s %14s %14s\n" "--------------------" "--------------" "--------------"
    printf "  %-20s %14d %14s\n" "SVM" "$SVM_NS" "$(awk "BEGIN{printf \"%.2f\", $SVM_NS/1000.0}")"
    printf "  %-20s %14d %14s\n" "MLIR" "$MLIR_NS" "$(awk "BEGIN{printf \"%.2f\", $MLIR_NS/1000.0}")"

    if [ "$SVM_NS" -gt 0 ] && [ "$MLIR_NS" -gt 0 ]; then
        local SPEEDUP
        SPEEDUP=$(awk "BEGIN{printf \"%.2fx\", $SVM_NS * 1.0 / $MLIR_NS}")
        local DELTA_PCT
        DELTA_PCT=$(awk "BEGIN{printf \"%+.1f%%\", 100.0*($SVM_NS - $MLIR_NS)/$SVM_NS}")
        echo ""
        echo "  MLIR speedup vs SVM: $SPEEDUP  ($DELTA_PCT GPU kernel time reduction)"
    elif [ "$SVM_NS" -eq 0 ]; then
        echo ""
        echo "  WARNING: SVM trace had 0 kernel time. Check $SVM_DIR for raw output."
    elif [ "$MLIR_NS" -eq 0 ]; then
        echo ""
        echo "  WARNING: MLIR trace had 0 kernel time. Check $MLIR_DIR for raw output."
    fi
    echo ""
}

# ============================================================================
# Run profiling on each circuit
# ============================================================================

# Circuit 1: Frame-only (Tier 0 / SVM baseline)
profile_circuit \
    "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" \
    "frame_h"

# Circuit 2: Cooperative tier (rank=5, needs coop kernel)
profile_circuit \
    "$BASE/tests/fixtures/rank_sweep/rank_q17_r5_d1.stim" \
    "rank_q17_r5_d1"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=================================================================="
echo "All profiling complete.  $(date)"
echo "Results directory: $RESULTS"
echo "=================================================================="
echo ""
echo "Raw trace data per run:"
ls -la "$RESULTS"/*/  2>/dev/null
