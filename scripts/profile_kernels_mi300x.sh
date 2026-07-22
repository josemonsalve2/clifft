#!/bin/bash
#SBATCH --job-name=profile-kernels
#SBATCH --partition=mi300x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/profile_kernels_%j.log
set +e

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
RESULTS="$BASE/results/profile_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS"
export PATH="/shared/jmonsalv/software/cmake-3.31.7-linux-x86_64/bin:$PATH"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

GPU_ARCH=$(rocminfo 2>/dev/null | grep -m1 'Name:.*gfx' | grep -oE 'gfx[0-9]+')
echo "=== GEAK-Style Hardware Counter Profiling ==="
echo "Node: $(hostname)  GPU: $GPU_ARCH  ROCm: $ROCM_PATH"
echo "Date: $(date)"
echo "Output: $RESULTS"
echo ""

# ====================================================================
# Detect profiling tools (GEAK priority order)
# ====================================================================
PROFILER=""
RPC_BIN=""
for cmd in rocprof-compute omniperf; do
    if command -v $cmd &>/dev/null; then
        PROFILER="rocprof-compute"
        RPC_BIN=$cmd
        break
    fi
done
if [ -z "$PROFILER" ] && [ -f "$ROCM_PATH/libexec/rocprofiler-compute/rocprof-compute" ]; then
    PROFILER="rocprof-compute"
    RPC_BIN="python3 $ROCM_PATH/libexec/rocprofiler-compute/rocprof-compute"
fi
if [ -z "$PROFILER" ] && command -v rocprofv3 &>/dev/null; then
    PROFILER="rocprofv3"
fi
if [ -z "$PROFILER" ] && command -v rocprof &>/dev/null; then
    PROFILER="rocprof"
fi
echo "Profiler: $PROFILER ($RPC_BIN)"

# ====================================================================
# Build
# ====================================================================
BUILD="$BASE/build-gpu-perf"
[ -x "$BUILD/run_gpu" ] || {
    rm -rf "$BUILD" && mkdir -p "$BUILD" && cd "$BUILD"
    cmake "$BASE" -DCLIFFT_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=$GPU_ARCH -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
    make -j$(nproc) run_gpu 2>&1 | tail -5
    cd "$BASE"
}
BINARY="$BUILD/run_gpu"
[ -x "$BINARY" ] || { echo "BUILD FAILED"; exit 1; }

SHOTS=1000000

# ====================================================================
# Select representative circuits across all tiers
# ====================================================================
# Tier 1: Register (rank 1-4) — small, compute-light
# Tier 2: LDS/Coop (rank 5-10) — shared memory, barriers
# Tier 3: Global (rank 11-19) — HBM, large state vectors
# Plus key QEC circuits

declare -a CIRCUITS
declare -a LABELS

add_circuit() {
    local path="$1" label="$2"
    if [ -f "$path" ]; then
        CIRCUITS+=("$path")
        LABELS+=("$label")
    else
        echo "SKIP: $path (not found)"
    fi
}

# --- Tier 1: Register (rank 1-4) ---
add_circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" "tier1_frame_h"
add_circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_cnot.stim" "tier1_frame_cnot"
add_circuit "$BASE/tests/fixtures/incremental/02_single_expand/one_t.stim" "tier1_one_t"
add_circuit "$BASE/tests/fixtures/incremental/05_combinations/cnot_h_r2.stim" "tier1_cnot_h_r2"
add_circuit "$BASE/tests/fixtures/incremental/06_small_circuits/rank4_mixed.stim" "tier1_rank4_mixed"

# --- Tier 2: LDS/Coop (rank 5-10) ---
add_circuit "$BASE/tests/fixtures/rank_sweep/rank_q17_r5_d1.stim" "tier2_r5_d1"
add_circuit "$BASE/tests/fixtures/rank_sweep/rank_q17_r6_d1.stim" "tier2_r6_d1"
add_circuit "$BASE/tests/fixtures/rank_sweep/rank_q17_r8_d1.stim" "tier2_r8_d1"
add_circuit "$BASE/tests/fixtures/rank_sweep/rank_q17_r10_d1.stim" "tier2_r10_d1"
add_circuit "$BASE/tests/fixtures/rank_sweep/rank_q17_r5_d3.stim" "tier2_r5_d3"
add_circuit "$BASE/tests/fixtures/rank_sweep/rank_q17_r8_d3.stim" "tier2_r8_d3_deep"

# --- Tier 3: Global/HBM (rank 11-19) ---
add_circuit "$BASE/tests/fixtures/rank_sweep/rank_q33_r12_d1.stim" "tier3_r12_d1"
add_circuit "$BASE/tests/fixtures/rank_sweep/rank_q33_r15_d1.stim" "tier3_r15_d1"
add_circuit "$BASE/tests/fixtures/rank_sweep/rank_q33_r18_d1.stim" "tier3_r18_d1"
add_circuit "$BASE/tests/fixtures/rank_sweep/rank_q33_r12_d3.stim" "tier3_r12_d3_deep"
add_circuit "$BASE/tests/fixtures/rank_sweep/rank_q33_r15_d3.stim" "tier3_r15_d3_deep"

# --- Key QEC circuits ---
add_circuit "$BASE/tests/fixtures/target_qec.stim" "qec_target"
add_circuit "$BASE/tests/fixtures/cultivation_d5.stim" "qec_cultivation_d5"
add_circuit "$BASE/tests/fixtures/large/surface_d7_t5.stim" "qec_surface_d7_t5"
add_circuit "$BASE/tests/fixtures/large/surface_d7_t10.stim" "qec_surface_d7_t10"
add_circuit "$BASE/tests/fixtures/large/color_d7_r7.stim" "qec_color_d7_r7"
add_circuit "$BASE/tests/fixtures/large/rep_d5_r100.stim" "qec_rep_d5_r100"
add_circuit "$BASE/tests/fixtures/large/surface_d11_t5.stim" "qec_surface_d11_t5"
add_circuit "$BASE/tests/fixtures/large/color_d9_r9.stim" "qec_color_d9_r9"

echo ""
echo "Profiling ${#CIRCUITS[@]} circuits"
echo ""

# ====================================================================
# Phase 1: Warm all kernel caches
# ====================================================================
echo "=== Phase 1: Warming kernel cache ==="
for i in "${!CIRCUITS[@]}"; do
    CIRC="${CIRCUITS[$i]}"
    LABEL="${LABELS[$i]}"
    printf "  Warming %-30s " "$LABEL"
    $BINARY --circuit "$CIRC" --shots 100 --hybrid 2>/dev/null
    echo "done"
done
echo "Cached: $(ls ~/.clifft/kernel_cache/*.hsaco 2>/dev/null | wc -l) .hsaco files"
echo ""

# ====================================================================
# Phase 2: Execution timing (SVM vs Compiled, 5 runs each)
# ====================================================================
echo "=== Phase 2: Execution Timing (5 runs x ${SHOTS} shots, warm cache) ==="
echo ""

RUNS=5
TIMING_CSV="$RESULTS/timing.csv"
echo "label,circuit,tier,peak_rank,svm_avg_sps,compiled_avg_sps,delta_pct,svm_kernel_sec,compiled_kernel_sec" > "$TIMING_CSV"

extract_field() { echo "$1" | grep "\"$2\"" | head -1 | grep -oE '[0-9]+\.?[0-9eE+\-]*' | head -1 || true; }

printf "%-30s %6s %12s %12s %8s\n" "Circuit" "Rank" "SVM(M sps)" "Compiled(M)" "Delta"
echo "--------------------------------------------------------------------------"

for i in "${!CIRCUITS[@]}"; do
    CIRC="${CIRCUITS[$i]}"
    LABEL="${LABELS[$i]}"

    # Get rank
    CHECK=$($BINARY --circuit "$CIRC" --shots 100 2>&1)
    RANK=$(extract_field "$CHECK" "peak_rank")
    RANK="${RANK:-?}"

    # Determine tier
    TIER="unknown"
    if [[ "$LABEL" == tier1_* ]]; then TIER="register"
    elif [[ "$LABEL" == tier2_* ]]; then TIER="lds_coop"
    elif [[ "$LABEL" == tier3_* ]]; then TIER="global"
    elif [[ "$LABEL" == qec_* ]]; then TIER="qec"
    fi

    SVM_SUM=0; COMP_SUM=0
    SVM_KTIME_SUM=0; COMP_KTIME_SUM=0

    for r in $(seq 1 $RUNS); do
        # SVM
        OUT=$($BINARY --circuit "$CIRC" --shots $SHOTS 2>&1)
        SPS=$(extract_field "$OUT" "shots_per_second_sampling_only"); SPS="${SPS:-0}"
        KSEC=$(extract_field "$OUT" "sample_seconds"); KSEC="${KSEC:-0}"
        SVM_SUM=$(python3 -c "print(float('$SVM_SUM')+float('$SPS'))")
        SVM_KTIME_SUM=$(python3 -c "print(float('$SVM_KTIME_SUM')+float('$KSEC'))")

        # Compiled
        OUT=$($BINARY --circuit "$CIRC" --shots $SHOTS --hybrid 2>&1)
        SPS=$(extract_field "$OUT" "shots_per_second_sampling_only"); SPS="${SPS:-0}"
        KSEC=$(extract_field "$OUT" "sample_seconds"); KSEC="${KSEC:-0}"
        COMP_SUM=$(python3 -c "print(float('$COMP_SUM')+float('$SPS'))")
        COMP_KTIME_SUM=$(python3 -c "print(float('$COMP_KTIME_SUM')+float('$KSEC'))")
    done

    SVM_AVG=$(python3 -c "print(f'{float(\"$SVM_SUM\")/$RUNS:.0f}')")
    COMP_AVG=$(python3 -c "print(f'{float(\"$COMP_SUM\")/$RUNS:.0f}')")
    SVM_KTIME=$(python3 -c "print(f'{float(\"$SVM_KTIME_SUM\")/$RUNS:.6f}')")
    COMP_KTIME=$(python3 -c "print(f'{float(\"$COMP_KTIME_SUM\")/$RUNS:.6f}')")
    DELTA=$(python3 -c "
s=float('$SVM_SUM'); c=float('$COMP_SUM')
print(f'{(c-s)/s*100:+.1f}' if s > 0 and c > 0 else 'N/A')
")
    SVM_M=$(python3 -c "print(f'{float(\"$SVM_AVG\")/1e6:.2f}')")
    COMP_M=$(python3 -c "print(f'{float(\"$COMP_AVG\")/1e6:.2f}')")

    printf "%-30s %6s %10sM %10sM %7s%%\n" "$LABEL" "$RANK" "$SVM_M" "$COMP_M" "$DELTA"
    echo "$LABEL,$CIRC,$TIER,$RANK,$SVM_AVG,$COMP_AVG,$DELTA,$SVM_KTIME,$COMP_KTIME" >> "$TIMING_CSV"
done

echo ""

# ====================================================================
# Phase 3: Hardware counter profiling
# ====================================================================
echo "=== Phase 3: Hardware Counter Profiling ==="
echo ""

COUNTER_CSV="$RESULTS/counters.csv"

profile_with_rocprofv3() {
    local circ="$1" label="$2" mode="$3"  # mode: svm or compiled
    local outdir="$RESULTS/rocprofv3/${label}_${mode}"
    mkdir -p "$outdir"

    local cmd="$BINARY --circuit $circ --shots $SHOTS"
    [ "$mode" = "compiled" ] && cmd="$cmd --hybrid"

    # Collect kernel trace + stats
    rocprofv3 --kernel-trace --stats -d "$outdir" -- $cmd 2>"$outdir/stderr.log"

    # Collect hardware counters in groups (gfx942 allows ~4 per pass)
    # Group 1: Instruction mix
    cat > "$outdir/counters_insts.txt" <<'COUNTERS'
pmc: SQ_INSTS_VALU SQ_INSTS_VMEM SQ_INSTS_SALU SQ_INSTS_LDS
COUNTERS
    rocprofv3 -i "$outdir/counters_insts.txt" -d "$outdir/insts" -- $cmd 2>>"$outdir/stderr.log"

    # Group 2: Wait/stall counters
    cat > "$outdir/counters_wait.txt" <<'COUNTERS'
pmc: SQ_WAIT_INST_LDS SQ_WAIT_INST_VMEM SQ_WAIT_INST_ANY SQ_BUSY_CU_CYCLES
COUNTERS
    rocprofv3 -i "$outdir/counters_wait.txt" -d "$outdir/wait" -- $cmd 2>>"$outdir/stderr.log"

    # Group 3: Cache hierarchy
    cat > "$outdir/counters_cache.txt" <<'COUNTERS'
pmc: TCP_TOTAL_READ TCP_TOTAL_WRITE TCC_HIT TCC_MISS
COUNTERS
    rocprofv3 -i "$outdir/counters_cache.txt" -d "$outdir/cache" -- $cmd 2>>"$outdir/stderr.log"

    echo "  rocprofv3 data in: $outdir"
}

profile_with_rocprof_compute() {
    local circ="$1" label="$2" mode="$3"
    local outdir="$RESULTS/rpc/${label}_${mode}"
    mkdir -p "$outdir"

    local cmd="$BINARY --circuit $circ --shots $SHOTS"
    [ "$mode" = "compiled" ] && cmd="$cmd --hybrid"

    # Full SoL analysis (no roofline to save time)
    $RPC_BIN profile --no-roof -n "${label}_${mode}" -- $cmd 2>"$outdir/profile_stderr.log"

    # Find the workload directory
    local wl_dir=$(find workloads -maxdepth 3 -name "MI300*" -type d 2>/dev/null | grep "${label}_${mode}" | head -1)
    if [ -n "$wl_dir" ]; then
        # Analyze: SoL blocks 0 (speed of light) and 2 (system)
        $RPC_BIN analyze -p "$wl_dir" -b 0 2 4 --output-format csv -o "$outdir" 2>"$outdir/analyze_stderr.log"
        echo "  rocprof-compute SoL in: $outdir"
    else
        echo "  WARNING: no workload directory found for ${label}_${mode}"
    fi
}

profile_with_rocprof_legacy() {
    local circ="$1" label="$2" mode="$3"
    local outdir="$RESULTS/rocprof/${label}_${mode}"
    mkdir -p "$outdir"

    local cmd="$BINARY --circuit $circ --shots $SHOTS"
    [ "$mode" = "compiled" ] && cmd="$cmd --hybrid"

    rocprof --stats -o "$outdir/stats.csv" $cmd 2>"$outdir/stderr.log"
    echo "  rocprof stats in: $outdir"
}

# Use the best available profiler
profile_circuit() {
    local circ="$1" label="$2" mode="$3"
    case "$PROFILER" in
        rocprof-compute)
            profile_with_rocprof_compute "$circ" "$label" "$mode"
            ;;
        rocprofv3)
            profile_with_rocprofv3 "$circ" "$label" "$mode"
            ;;
        rocprof)
            profile_with_rocprof_legacy "$circ" "$label" "$mode"
            ;;
        *)
            echo "  No profiler available — skipping $label $mode"
            ;;
    esac
}

# Profile a representative subset (full set takes too long with profiling)
# Select ~8 circuits spanning tiers + key QEC
PROFILE_INDICES=(0 4 5 8 10 13 15 16 17 19)

for idx in "${PROFILE_INDICES[@]}"; do
    [ $idx -ge ${#CIRCUITS[@]} ] && continue
    CIRC="${CIRCUITS[$idx]}"
    LABEL="${LABELS[$idx]}"
    echo ""
    echo "--- Profiling: $LABEL ---"

    # Profile SVM
    printf "  SVM:      "
    profile_circuit "$CIRC" "$LABEL" "svm"

    # Profile Compiled
    printf "  Compiled: "
    profile_circuit "$CIRC" "$LABEL" "compiled"
done

# ====================================================================
# Phase 4: Register pressure analysis (VGPR/SGPR from .hsaco)
# ====================================================================
echo ""
echo "=== Phase 4: Register Pressure Analysis ==="
echo ""

REG_CSV="$RESULTS/registers.csv"
echo "label,vgpr_count,sgpr_count,lds_bytes,spill_sgpr,spill_vgpr,max_occupancy_waves" > "$REG_CSV"

OBJDUMP=$(which llvm-objdump 2>/dev/null || echo "$ROCM_PATH/lib/llvm/bin/llvm-objdump")

if [ -x "$OBJDUMP" ]; then
    for hsaco in ~/.clifft/kernel_cache/*.hsaco; do
        [ -f "$hsaco" ] || continue
        HASH=$(basename "$hsaco" | sed 's/_gfx[0-9]*.hsaco//')

        # Extract register info
        INFO=$($OBJDUMP --disassemble-symbols=compiled_sample_kernel "$hsaco" 2>/dev/null | head -30)
        if [ -z "$INFO" ]; then
            # Try other kernel names
            INFO=$($OBJDUMP --disassemble-symbols=compiled_sample_kernel_coop "$hsaco" 2>/dev/null | head -30)
        fi
        if [ -z "$INFO" ]; then
            INFO=$($OBJDUMP --disassemble-symbols=compiled_sample_kernel_global "$hsaco" 2>/dev/null | head -30)
        fi

        VGPR=$(echo "$INFO" | grep -oP '\.vgpr_count:\s*\K[0-9]+' | head -1)
        SGPR=$(echo "$INFO" | grep -oP '\.sgpr_count:\s*\K[0-9]+' | head -1)
        LDS=$(echo "$INFO" | grep -oP '\.lds_size:\s*\K[0-9]+' | head -1)
        SPILL_S=$(echo "$INFO" | grep -oP '\.sgpr_spill_count:\s*\K[0-9]+' | head -1)
        SPILL_V=$(echo "$INFO" | grep -oP '\.vgpr_spill_count:\s*\K[0-9]+' | head -1)

        # Try readelf for more details if objdump failed
        if [ -z "$VGPR" ]; then
            INFO2=$($OBJDUMP -s "$hsaco" 2>/dev/null | head -50)
            VGPR=$(echo "$INFO2" | grep -oP 'NumVgprs:\s*\K[0-9]+' | head -1)
            SGPR=$(echo "$INFO2" | grep -oP 'NumSgprs:\s*\K[0-9]+' | head -1)
        fi

        VGPR="${VGPR:-?}"
        SGPR="${SGPR:-?}"
        LDS="${LDS:-0}"
        SPILL_S="${SPILL_S:-0}"
        SPILL_V="${SPILL_V:-0}"

        # Occupancy: gfx942 has 512 VGPRs per SIMD, wavefront=64
        # Max waves per SIMD = floor(512 / ceil(VGPR/4)*4)  (rounded up to 4)
        if [ "$VGPR" != "?" ] && [ "$VGPR" -gt 0 ] 2>/dev/null; then
            MAX_WAVES=$(python3 -c "
import math
vgpr = int('$VGPR')
# gfx942: 512 VGPRs per SIMD, each SIMD handles 1 wavefront at a time
# Max waves = floor(512 / (ceil(vgpr/8)*8))  # granularity of 8
vgpr_alloc = math.ceil(vgpr / 8) * 8
max_w = 512 // vgpr_alloc if vgpr_alloc > 0 else 0
# LDS limit: 64KB per workgroup on gfx942
print(min(max_w, 10))  # cap at 10 for display
")
        else
            MAX_WAVES="?"
        fi

        printf "  %-20s VGPR=%-4s SGPR=%-4s LDS=%-6s Spill(S/V)=%s/%s MaxWaves=%s\n" \
            "$HASH" "$VGPR" "$SGPR" "$LDS" "$SPILL_S" "$SPILL_V" "$MAX_WAVES"
        echo "$HASH,$VGPR,$SGPR,$LDS,$SPILL_S,$SPILL_V,$MAX_WAVES" >> "$REG_CSV"
    done
else
    echo "  llvm-objdump not found — skipping register analysis"
    echo "  Tried: $OBJDUMP"
fi

# ====================================================================
# Phase 5: GEAK Bottleneck Classification Summary
# ====================================================================
echo ""
echo "=== Phase 5: Bottleneck Classification Summary ==="
echo ""

python3 - "$RESULTS" <<'PYEOF'
import sys, os, csv, glob, json

results_dir = sys.argv[1]

# Parse timing data
timing_file = os.path.join(results_dir, "timing.csv")
if os.path.exists(timing_file):
    print("--- Timing Summary ---")
    print(f"{'Label':<30} {'Rank':>5} {'SVM(M sps)':>12} {'Compiled(M)':>12} {'Delta':>8} {'SVM(ms)':>10} {'Comp(ms)':>10}")
    print("-" * 95)
    with open(timing_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            svm_sps = float(row.get('svm_avg_sps', 0) or 0)
            comp_sps = float(row.get('compiled_avg_sps', 0) or 0)
            svm_ms = float(row.get('svm_kernel_sec', 0) or 0) * 1000
            comp_ms = float(row.get('compiled_kernel_sec', 0) or 0) * 1000
            delta = row.get('delta_pct', 'N/A')
            print(f"{row['label']:<30} {row.get('peak_rank','?'):>5} "
                  f"{svm_sps/1e6:>10.2f}M {comp_sps/1e6:>10.2f}M {delta:>7}% "
                  f"{svm_ms:>9.2f} {comp_ms:>9.2f}")
    print()

# Parse rocprofv3 kernel stats if available
stats_dirs = glob.glob(os.path.join(results_dir, "rocprofv3/*/"))
if stats_dirs:
    print("--- rocprofv3 Kernel Stats ---")
    for sdir in sorted(stats_dirs):
        label = os.path.basename(sdir.rstrip('/'))
        # Look for kernel_stats.csv
        for f in glob.glob(os.path.join(sdir, "**/kernel_stats.csv"), recursive=True):
            print(f"\n  {label}:")
            try:
                with open(f) as csvf:
                    reader = csv.DictReader(csvf)
                    for row in reader:
                        name = row.get('Name', row.get('KernelName', '?'))
                        if 'compiled' in name.lower() or 'sample_kernel' in name.lower():
                            dur = row.get('DurationNs', row.get('TotalDurationNs', '?'))
                            calls = row.get('Calls', '?')
                            print(f"    Kernel: {name[:60]}")
                            print(f"    Duration: {dur}ns  Calls: {calls}")
            except Exception as e:
                print(f"    Error reading {f}: {e}")

# Parse counter data if available
inst_dirs = glob.glob(os.path.join(results_dir, "rocprofv3/*/insts/"))
if inst_dirs:
    print("\n--- Instruction Mix (GEAK Classification) ---")
    for idir in sorted(inst_dirs):
        label = os.path.basename(os.path.dirname(idir))
        for f in glob.glob(os.path.join(idir, "**/*.csv"), recursive=True):
            try:
                with open(f) as csvf:
                    reader = csv.DictReader(csvf)
                    for row in reader:
                        valu = int(row.get('SQ_INSTS_VALU', 0) or 0)
                        vmem = int(row.get('SQ_INSTS_VMEM', 0) or 0)
                        salu = int(row.get('SQ_INSTS_SALU', 0) or 0)
                        lds = int(row.get('SQ_INSTS_LDS', 0) or 0)
                        total = valu + vmem + salu + lds
                        if total == 0:
                            continue
                        valu_pct = valu * 100 / total
                        vmem_pct = vmem * 100 / total
                        lds_pct = lds * 100 / total

                        # GEAK bottleneck classification
                        if valu_pct > 60 and vmem_pct < 40:
                            bottleneck = "COMPUTE-BOUND"
                        elif vmem_pct > 60 and valu_pct < 40:
                            bottleneck = "MEMORY-BOUND"
                        elif valu_pct < 40 and vmem_pct < 40:
                            bottleneck = "LATENCY-BOUND"
                        elif lds_pct > 50:
                            bottleneck = "LDS-BOUND"
                        else:
                            bottleneck = "BALANCED"

                        print(f"  {label:<40} VALU={valu_pct:5.1f}% VMEM={vmem_pct:5.1f}% "
                              f"SALU={salu*100/total:5.1f}% LDS={lds_pct:5.1f}%  => {bottleneck}")
            except Exception as e:
                pass

# Parse register data
reg_file = os.path.join(results_dir, "registers.csv")
if os.path.exists(reg_file):
    print("\n--- Register Pressure ---")
    print(f"{'Hash':<20} {'VGPR':>5} {'SGPR':>5} {'LDS':>8} {'Spill(S)':>8} {'Spill(V)':>8} {'MaxWaves':>9}")
    print("-" * 65)
    with open(reg_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(f"{row['label']:<20} {row.get('vgpr_count','?'):>5} {row.get('sgpr_count','?'):>5} "
                  f"{row.get('lds_bytes','0'):>8} {row.get('spill_sgpr','0'):>8} {row.get('spill_vgpr','0'):>8} "
                  f"{row.get('max_occupancy_waves','?'):>9}")

print("\n\nDone: profiling complete")
PYEOF

echo ""
echo "=== All Results ==="
echo "Timing CSV: $RESULTS/timing.csv"
echo "Counter data: $RESULTS/rocprofv3/ or $RESULTS/rpc/"
echo "Register CSV: $RESULTS/registers.csv"
echo "Full log: check SLURM output"
echo ""
echo "Done: $(date)"
