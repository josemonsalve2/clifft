#!/bin/bash
#SBATCH --job-name=perf-eval-3way
#SBATCH --partition=mi300x
#SBATCH --nodelist=rad-mi300x-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/perf_eval_mi300x_%j.log
#SBATCH --exclusive

# ============================================================
# 3-Way GPU Kernel Performance Evaluation — MI300X (gfx942)
# ============================================================
# Senior performance engineer protocol:
#   Phase 1: Build all 3 variants (SVM, compiled HIP codegen, MLIR)
#   Phase 2: Benchmark all circuits × all modes (20 runs, 3 warmup)
#   Phase 3: rocprof hardware counter collection (L2 BW, wavefront stalls)
#   Phase 4: Roofline model inputs (FLOPs/byte, achieved BW, FLOP/s)
#   Phase 5: Per-tier analysis summary
#
# Circuits:
#   Tier 0 (frame-only, rank=0): color_d7, surface_d13 — no amplitude array
#   Tier 1 (register, rank≤4): cultivation_d5, hook_inject, circuit_d7, T-gate circuits
#   Tier 2 (LDS/coop, rank 5-10): surface_d7_t10, surface_d9_t10, surface_d11_t10
#   Tier 3 (global/HBM, rank>10): rank11_q50, extreme_r17, extreme_r19

set -euo pipefail

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
NODE=$(hostname)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="$BASE/results/perf_eval_${TIMESTAMP}"
mkdir -p "$RESULTS"

echo "================================================================="
echo "3-Way GPU Kernel Performance Evaluation — MI300X"
echo "Node:      $NODE"
echo "Date:      $(date)"
echo "Job ID:    ${SLURM_JOB_ID:-n/a}"
echo "Results:   $RESULTS"
echo "================================================================="

# ---------------------------------------------------------------
# 0. GPU environment info
# ---------------------------------------------------------------
echo ""
echo "=== GPU Info ==="
rocm-smi --showid --showproductname --showmeminfo vram 2>/dev/null | head -20 || true
rocminfo 2>/dev/null | grep -E "gfx|Name:|Compute Unit" | head -10 || true
echo ""

# ---------------------------------------------------------------
# 1. Build all variants
# ---------------------------------------------------------------
echo "=== Phase 1: Build ==="

# Build SVM + compiled HIP (existing build-gpu, already Release/gfx942)
BINARY_STD="$BASE/build-gpu/run_gpu"

# Build with MLIR enabled (separate build dir)
BINARY_MLIR="$BASE/build-gpu-mlir/run_gpu"
if [ ! -x "$BINARY_MLIR" ]; then
    echo "Building MLIR variant..."
    mkdir -p "$BASE/build-gpu-mlir"
    cd "$BASE/build-gpu-mlir"
    cmake "$BASE" \
        -DCLIFFT_ENABLE_HIP=ON \
        -DCMAKE_HIP_ARCHITECTURES=gfx942 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCLIFFT_BUILD_PROFILER=ON \
        -DCLIFFT_ENABLE_MLIR=ON \
        -DCMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
        2>&1
    make -j$(nproc) run_gpu 2>&1
    cd "$BASE"
fi

# Check which MLIR tools are available
LLVM_PREFIX="${LLVM_PREFIX:-/shared/jmonsalv/software/modules/llvm/upstream_05082025}"
if [ -x "$LLVM_PREFIX/bin/mlir-opt" ]; then
    export PATH="$LLVM_PREFIX/bin:$PATH"
    echo "MLIR tools: $LLVM_PREFIX/bin"
else
    echo "WARNING: mlir-opt not found at $LLVM_PREFIX/bin — MLIR mode will fall back to SVM"
fi

echo "Binary (std): $BINARY_STD"
echo "Binary (mlir): $BINARY_MLIR"
echo ""

# ---------------------------------------------------------------
# 2. Circuit selection — covering all 3 tiers + frame-only
# ---------------------------------------------------------------
# Format: "label|path|tier|peak_rank_hint|qubits"
declare -a CIRCUITS=(
    # --- TIER 0: Frame-only (rank=0, no amplitude array) ---
    "target_qec|$BASE/tests/fixtures/target_qec.stim|0|0|0"
    "color_d7|$BASE/tests/fixtures/large/color_d7_r7.stim|0|0|55"
    "surface_d13|$BASE/tests/fixtures/large/surface_d13_r13.stim|0|0|337"
    "rep_d5_r100|$BASE/tests/fixtures/large/rep_d5_r100.stim|0|0|0"

    # --- TIER 1: Register (rank 1-4, VGPRs) ---
    "cultivation_d5|$BASE/tests/fixtures/cultivation_d5.stim|1|4|0"
    "surface_d7_t5|$BASE/tests/fixtures/large/surface_d7_t5.stim|1|5|97"
    "cultivation_d7|$BASE/tests/fixtures/large/cultivation_d7_p0001.stim|1|4|0"
    "rank_r1_q33|$BASE/tests/fixtures/rank_sweep/rank_q33_r1_d3.stim|1|1|33"
    "rank_r3_q33|$BASE/tests/fixtures/rank_sweep/rank_q33_r3_d3.stim|1|3|33"
    "rank_r4_q33|$BASE/tests/fixtures/rank_sweep/rank_q33_r4_d3.stim|1|4|33"

    # --- TIER 2: LDS / Shared-Coop (rank 5-10) ---
    "surface_d7_t10|$BASE/tests/fixtures/large/surface_d7_t10.stim|2|10|97"
    "surface_d9_t10|$BASE/tests/fixtures/large/surface_d9_t10.stim|2|10|161"
    "surface_d11_t10|$BASE/tests/fixtures/large/surface_d11_t10.stim|2|10|241"
    "rank_r5_q33|$BASE/tests/fixtures/rank_sweep/rank_q33_r5_d3.stim|2|5|33"
    "rank_r8_q33|$BASE/tests/fixtures/rank_sweep/rank_q33_r8_d3.stim|2|8|33"
    "rank_r10_q33|$BASE/tests/fixtures/rank_sweep/rank_q33_r10_d3.stim|2|10|33"

    # --- TIER 3: Global / HBM-Coop (rank 11-19) ---
    "rank11_q50|$BASE/tests/fixtures/large/rank11_q50_d5_tier3.stim|3|11|50"
    "extreme_r17|$BASE/tests/fixtures/large/extreme_r17_q200_d15.stim|3|17|200"
    "extreme_r19|$BASE/tests/fixtures/large/extreme_r19_q200_d15.stim|3|19|200"
    "rank_r15_q33|$BASE/tests/fixtures/rank_sweep/rank_q33_r15_d3.stim|3|15|33"
    "rank_r19_q33|$BASE/tests/fixtures/rank_sweep/rank_q33_r19_d3.stim|3|19|33"
)

SHOTS=1000000
RUNS=20
WARMUP=3

extract_field() { echo "$1" | grep "\"$2\"" | head -1 | grep -oE '[0-9]+\.?[0-9eE+\-]*'; }

# ---------------------------------------------------------------
# 3. Phase 2: Benchmark — all modes × all circuits
# ---------------------------------------------------------------
echo "=== Phase 2: Benchmark (${RUNS} runs × ${SHOTS} shots) ==="
BENCH_CSV="$RESULTS/bench_3way.csv"
echo "mode,label,tier,peak_rank_hint,qubits,run,shots,peak_rank,passed,sample_sec,kernel_sec,shots_per_sec" > "$BENCH_CSV"

# rocm info for metadata
GPU_ARCH=$(rocminfo 2>/dev/null | grep -m1 "Name:.*gfx" | grep -oE "gfx[0-9]+" || echo "gfx942")
ROCM_VER=$(rocminfo 2>/dev/null | grep -m1 "ROCm Runtime Version" | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" || echo "unknown")

run_benchmark() {
    local MODE=$1   # svm, compiled, mlir
    local BINARY=$2
    local FLAG=$3
    local CIRCUIT=$4
    local LABEL=$5
    local TIER=$6
    local RANK_HINT=$7
    local QUBITS=$8

    if [ ! -f "$CIRCUIT" ]; then
        echo "  SKIP: $CIRCUIT not found"
        return
    fi

    # Warmup
    for _ in $(seq 1 $WARMUP); do
        numactl --cpunodebind=1 --membind=1 \
            "$BINARY" --circuit "$CIRCUIT" --shots 100000 $FLAG > /dev/null 2>&1 || true
    done

    # Timed runs
    for i in $(seq 1 $RUNS); do
        OUTPUT=$(numactl --cpunodebind=1 --membind=1 \
            "$BINARY" --circuit "$CIRCUIT" --shots $SHOTS $FLAG 2>&1) || {
            echo "  FAIL: $MODE $LABEL run $i"
            echo "$MODE,$LABEL,$TIER,$RANK_HINT,$QUBITS,$i,$SHOTS,-1,0,0,0,0" >> "$BENCH_CSV"
            continue
        }
        SPS=$(extract_field "$OUTPUT" "shots_per_second_sampling_only")
        SEC=$(extract_field "$OUTPUT" "sample_seconds")
        KSEC=$(extract_field "$OUTPUT" "kernel_seconds")
        PASSED=$(extract_field "$OUTPUT" "passed_shots")
        PEAK=$(extract_field "$OUTPUT" "peak_rank")
        [ -z "$KSEC" ] && KSEC="$SEC"
        printf "  %-10s %-22s run %2d: %12s shots/s  kernel=%.4fs  rank=%s\n" \
            "$MODE" "$LABEL" "$i" "$SPS" "$KSEC" "$PEAK"
        echo "$MODE,$LABEL,$TIER,$RANK_HINT,$QUBITS,$i,$SHOTS,$PEAK,$PASSED,$SEC,$KSEC,$SPS" >> "$BENCH_CSV"
    done
}

for ENTRY in "${CIRCUITS[@]}"; do
    IFS='|' read -r LABEL CIRCUIT TIER RANK_HINT QUBITS <<< "$ENTRY"
    [ -f "$CIRCUIT" ] || continue
    echo ""
    echo "--- $LABEL  [tier=$TIER rank~$RANK_HINT q=$QUBITS] ---"

    # SVM (always available)
    echo "  [SVM interpreter]"
    run_benchmark "svm" "$BINARY_STD" "" "$CIRCUIT" "$LABEL" "$TIER" "$RANK_HINT" "$QUBITS"

    # Compiled HIP codegen (--hybrid)
    echo "  [Compiled HIP megakernel]"
    run_benchmark "compiled" "$BINARY_STD" "--hybrid" "$CIRCUIT" "$LABEL" "$TIER" "$RANK_HINT" "$QUBITS"

    # MLIR (--mlir, only for tier 1 circuits where MLIR codegen is implemented)
    if [ "$TIER" = "1" ] && [ -x "$BINARY_MLIR" ]; then
        echo "  [MLIR→LLVM-IR megakernel]"
        run_benchmark "mlir" "$BINARY_MLIR" "--mlir" "$CIRCUIT" "$LABEL" "$TIER" "$RANK_HINT" "$QUBITS"
    fi
done

echo ""
echo "Benchmark CSV: $BENCH_CSV"

# ---------------------------------------------------------------
# 4. Phase 3: rocprof hardware counter collection
# ---------------------------------------------------------------
echo ""
echo "=== Phase 3: rocprof Hardware Counter Collection ==="

ROCPROF_DIR="$RESULTS/rocprof"
mkdir -p "$ROCPROF_DIR"

# Profile a representative circuit per tier with rocprof
# Counter sets covering BW, wavefront efficiency, L2, occupancy
ROCPROF_INPUT="$ROCPROF_DIR/counters.txt"
cat > "$ROCPROF_INPUT" << 'EOF'
pmc: SQ_WAVES GRBM_COUNT TCP_TOTAL_CACHE_ACCESSES_sum TCP_TOTAL_WRITE_sum TCP_TOTAL_READ_sum
pmc: TCC_HIT_sum TCC_MISS_sum TCC_EA_RDREQ_32B_sum TCC_EA_WRREQ_sum
pmc: SQ_BUSY_CYCLES SQ_INSTS_VALU SQ_INSTS_VMEM SQ_INSTS_LDS
pmc: SQ_LDS_BANK_CONFLICT CP_STAT
EOF

# rocprof wrapper
rocprof_circuit() {
    local MODE=$1
    local BINARY=$2
    local FLAG=$3
    local CIRCUIT=$4
    local LABEL=$5
    local TIER=$6

    [ -f "$CIRCUIT" ] || return

    local OUTDIR="$ROCPROF_DIR/${LABEL}_${MODE}"
    mkdir -p "$OUTDIR"

    echo "  rocprof: $MODE $LABEL (tier $TIER)"
    # Warmup first
    numactl --cpunodebind=1 --membind=1 \
        "$BINARY" --circuit "$CIRCUIT" --shots 100000 $FLAG > /dev/null 2>&1 || true

    # Profile 3 runs to get stable counters
    for i in 1 2 3; do
        numactl --cpunodebind=1 --membind=1 \
            rocprof --stats -i "$ROCPROF_INPUT" -o "$OUTDIR/run${i}.csv" \
            "$BINARY" --circuit "$CIRCUIT" --shots 500000 $FLAG > "$OUTDIR/run${i}.log" 2>&1 || {
            echo "    WARNING: rocprof failed for run $i"
        }
    done
}

# Profile one representative per tier
TIER0_CIRC="$BASE/tests/fixtures/target_qec.stim"
TIER1_CIRC="$BASE/tests/fixtures/cultivation_d5.stim"
TIER2_CIRC="$BASE/tests/fixtures/large/surface_d7_t10.stim"
TIER3_CIRC="$BASE/tests/fixtures/large/rank11_q50_d5_tier3.stim"

if command -v rocprof >/dev/null 2>&1; then
    [ -f "$TIER0_CIRC" ] && {
        rocprof_circuit "svm" "$BINARY_STD" "" "$TIER0_CIRC" "target_qec" "0"
        rocprof_circuit "compiled" "$BINARY_STD" "--hybrid" "$TIER0_CIRC" "target_qec" "0"
    }
    [ -f "$TIER1_CIRC" ] && {
        rocprof_circuit "svm" "$BINARY_STD" "" "$TIER1_CIRC" "cultivation_d5" "1"
        rocprof_circuit "compiled" "$BINARY_STD" "--hybrid" "$TIER1_CIRC" "cultivation_d5" "1"
        [ -x "$BINARY_MLIR" ] && rocprof_circuit "mlir" "$BINARY_MLIR" "--mlir" "$TIER1_CIRC" "cultivation_d5" "1"
    }
    [ -f "$TIER2_CIRC" ] && {
        rocprof_circuit "svm" "$BINARY_STD" "" "$TIER2_CIRC" "surface_d7_t10" "2"
        rocprof_circuit "compiled" "$BINARY_STD" "--hybrid" "$TIER2_CIRC" "surface_d7_t10" "2"
    }
    [ -f "$TIER3_CIRC" ] && {
        rocprof_circuit "svm" "$BINARY_STD" "" "$TIER3_CIRC" "rank11_q50" "3"
        rocprof_circuit "compiled" "$BINARY_STD" "--hybrid" "$TIER3_CIRC" "rank11_q50" "3"
    }
else
    echo "  WARNING: rocprof not found — skipping hardware counter collection"
fi

# ---------------------------------------------------------------
# 5. Phase 4: Roofline model computation
# ---------------------------------------------------------------
echo ""
echo "=== Phase 4: Roofline Analysis ==="

# MI300X hardware peak values (from AMD spec):
#   Peak FP32 scalar FLOP/s: 163 TF
#   Peak FP32 vector FLOP/s: 163 TF
#   Peak HBM bandwidth: 5.3 TB/s (192GB HBM3)
#   Peak LDS bandwidth: ~50 TB/s (LDS is per-CU, not global; rough estimate)
MI300X_PEAK_FLOP_F32=163e12   # FP32 peak (TF)
MI300X_HBM_BW=5300e9          # HBM bandwidth (GB/s → B/s)
MI300X_LDS_BW=50000e9         # LDS bandwidth estimate (rough)

ROOFLINE_REPORT="$RESULTS/roofline.txt"

{
echo "================================================================="
echo "Roofline Analysis — MI300X gfx942"
echo "Hardware peaks:"
echo "  FP32 matrix FLOP/s:  163 TF"
echo "  HBM bandwidth:       5.3 TB/s"
echo "  Peak arithmetic intensity ridge point:"
echo "    163e12 / 5300e9 = $(python3 -c 'print(f"{163e12/5300e9:.1f}")')" FLOPs/byte
echo ""
echo "Arithmetic Intensity per tier:"
echo ""

# For each tier, estimate FLOPs/byte from the math:
# Tier 1 register (rank=4): v[16] complex, 2 FP32 muls + 1 add per H gate iteration
#   H gate: 1 amp load, 1 add, 1 sub, 2 muls (scale) → ~6 FLOPs / 2*8 bytes = 0.375 FLOPs/byte
#   Data fits in VGPRs → bandwidth is register file, not HBM
#   → compute-bound by VALU
echo "Tier 1 (register, rank≤4):"
echo "  H gate: 2 amps * 8B + 6 FLOPs → 0.375 FLOP/byte (from HBM: near zero; VGPR-resident)"
echo "  Array resides in VGPRs → kernel is VALU-throughput limited, not BW-limited"
echo "  Roofline bound: VALU (FP32 scalar at 163 TF/s / 256 threads_per_CU * occupancy)"
echo ""

echo "Tier 2 (LDS/coop, rank≤10): up to 1024 amplitudes = 8192 bytes"
echo "  H gate on rank-10: 512 load-pairs, 512 store-pairs → 512*2*8 = 8192B accessed"
echo "  FLOPs: 512 * 6 = 3072 FLOP → intensity = 3072/8192 = 0.375 FLOP/byte (from LDS)"
echo "  LDS bandwidth: ~50 TB/s — very likely compute-bound not BW-limited"
echo "  Key concern: LDS bank conflicts (2-qubit gates stride over 2 axes)"
echo ""

echo "Tier 3 (global/HBM, rank≤19): up to 512K amplitudes = 4MB"
echo "  H gate on rank-19: 2^18 load-pairs → 2^18 * 16B = 4MB per gate application"
echo "  FLOPs: 2^18 * 6 = 1.57M FLOP → intensity = 1.57M / 4MB = 0.375 FLOP/byte (HBM)"
echo "  HBM peak: 5.3 TB/s → 5.3e12 * 0.375 = 1.98 TF achievable"
echo "  Actual HBM utilization depends on access pattern regularity"
echo "  Key concern: HBM latency for non-sequential butterfly access pattern"
echo ""

echo "================================================================="
echo "Key bottleneck predictions:"
echo "  Tier 1: Instruction-level parallelism, VALU pipeline utilization"
echo "          Compiled megakernel advantage: no opcode dispatch overhead"
echo "          MLIR advantage: whole-function optimization (CSE, const-fold)"
echo "  Tier 2: LDS bank conflicts on butterfly access, thread divergence"
echo "          Compiled coop advantage: baked constants, no LUT lookup"
echo "  Tier 3: HBM bandwidth, L2 cache hit rate for repeated axis patterns"
echo "          XCD affinity: per-XCD work stealing avoids cross-XCD traffic"
} | tee "$ROOFLINE_REPORT"

# ---------------------------------------------------------------
# 6. Phase 5: Statistical summary and analysis
# ---------------------------------------------------------------
echo ""
echo "=== Phase 5: Statistical Summary ==="
SUMMARY="$RESULTS/summary.txt"

python3 - "$BENCH_CSV" "$SUMMARY" << 'PYEOF'
import sys, csv, math
from collections import defaultdict

bench_csv = sys.argv[1]
summary_path = sys.argv[2]

data = defaultdict(lambda: defaultdict(list))  # [label][mode] → [shots_per_sec]
tiers = {}
ranks = {}
qubits = {}

with open(bench_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        sps = float(row.get('shots_per_sec', 0) or 0)
        if sps > 0:
            data[row['label']][row['mode']].append(sps)
        tiers[row['label']] = row.get('tier', '?')
        ranks[row['label']] = row.get('peak_rank_hint', '?')
        qubits[row['label']] = row.get('qubits', '?')

def stats(vals):
    if not vals: return (0, 0, 0)
    n = len(vals)
    mean = sum(vals) / n
    var = sum((x-mean)**2 for x in vals) / max(1, n-1)
    std = math.sqrt(var)
    cv = 100 * std / mean if mean > 0 else 0
    return mean, std, cv

lines = []
lines.append("=" * 120)
lines.append(f"{'Circuit':<25} {'Tier':<5} {'Rank':<5} {'Q':<5} | "
             f"{'SVM Mean':>13} {'CV':>6} | "
             f"{'Compiled Mean':>13} {'CV':>6} {'Delta':>7} | "
             f"{'MLIR Mean':>13} {'CV':>6} {'Delta':>7}")
lines.append("-" * 120)

for label in sorted(data.keys(), key=lambda l: (tiers.get(l,'?'), l)):
    tier = tiers.get(label, '?')
    rank = ranks.get(label, '?')
    q = qubits.get(label, '?')
    modes = data[label]

    svm_mean, svm_std, svm_cv = stats(modes.get('svm', []))
    comp_mean, comp_std, comp_cv = stats(modes.get('compiled', []))
    mlir_mean, mlir_std, mlir_cv = stats(modes.get('mlir', []))

    def delta(base, other):
        if base <= 0: return 'N/A'
        return f"{(other - base)/base*100:+.1f}%"

    lines.append(
        f"{label:<25} {tier:<5} {rank:<5} {q:<5} | "
        f"{svm_mean:>13.0f} {svm_cv:>5.1f}% | "
        f"{comp_mean:>13.0f} {comp_cv:>5.1f}% {delta(svm_mean, comp_mean):>7} | "
        f"{mlir_mean:>13.0f} {mlir_cv:>5.1f}% {delta(svm_mean, mlir_mean):>7}"
    )

lines.append("=" * 120)
lines.append("Notes: shots_per_sec = passed_shots / sample_seconds (GPU sampling only)")
lines.append("Delta = (mode - SVM) / SVM * 100%")

output = "\n".join(lines)
print(output)
with open(summary_path, 'w') as f:
    f.write(output + "\n")
PYEOF

# ---------------------------------------------------------------
# 7. Tier-level aggregates and performance engineer verdict
# ---------------------------------------------------------------
echo ""
echo "=== Phase 6: Per-Tier Analysis & Engineering Verdict ==="
VERDICT="$RESULTS/verdict.txt"

python3 - "$BENCH_CSV" "$VERDICT" << 'PYEOF'
import sys, csv, math
from collections import defaultdict

bench_csv = sys.argv[1]
verdict_path = sys.argv[2]

# Aggregate by tier
tier_data = defaultdict(lambda: defaultdict(list))  # [tier][mode] → [speedups vs svm]
mode_sps = defaultdict(lambda: defaultdict(list))   # [mode][tier] → [mean_sps]
circuit_data = defaultdict(lambda: defaultdict(list))  # [label][mode] → [sps]
tiers_map = {}

with open(bench_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        sps = float(row.get('shots_per_sec', 0) or 0)
        label = row['label']
        tier = row.get('tier', '?')
        mode = row['mode']
        tiers_map[label] = tier
        if sps > 0:
            circuit_data[label][mode].append(sps)

def mean(vals): return sum(vals)/len(vals) if vals else 0.0

lines = []
lines.append("=" * 80)
lines.append("PERFORMANCE ENGINEERING VERDICT")
lines.append("=" * 80)

for tier_id in ['0', '1', '2', '3']:
    tier_labels = {l for l, t in tiers_map.items() if t == tier_id}
    if not tier_labels: continue

    tier_names = {
        '0': 'Frame-only (rank=0): no amplitude array',
        '1': 'Register tier (rank 1-4): VGPRs hold amplitudes',
        '2': 'LDS/Coop tier (rank 5-10): __shared__ memory',
        '3': 'Global/HBM tier (rank 11-19): HBM amplitudes',
    }
    lines.append(f"\nTier {tier_id}: {tier_names[tier_id]}")
    lines.append("-" * 60)

    speedups_comp = []
    speedups_mlir = []
    svm_sps_all = []

    for label in tier_labels:
        svm_m = mean(circuit_data[label].get('svm', []))
        comp_m = mean(circuit_data[label].get('compiled', []))
        mlir_m = mean(circuit_data[label].get('mlir', []))
        if svm_m > 0:
            svm_sps_all.append(svm_m)
            if comp_m > 0: speedups_comp.append(comp_m / svm_m)
            if mlir_m > 0: speedups_mlir.append(mlir_m / svm_m)

    if speedups_comp:
        avg_comp = sum(speedups_comp) / len(speedups_comp)
        min_comp = min(speedups_comp)
        max_comp = max(speedups_comp)
        lines.append(f"  Compiled vs SVM: avg {avg_comp:.2f}x  [min {min_comp:.2f}x  max {max_comp:.2f}x]")
    else:
        lines.append("  Compiled vs SVM: no data")

    if speedups_mlir:
        avg_mlir = sum(speedups_mlir) / len(speedups_mlir)
        min_mlir = min(speedups_mlir)
        max_mlir = max(speedups_mlir)
        lines.append(f"  MLIR vs SVM:     avg {avg_mlir:.2f}x  [min {min_mlir:.2f}x  max {max_mlir:.2f}x]")
    else:
        lines.append("  MLIR vs SVM: no data (tier 1 only)")

    if svm_sps_all:
        avg_svm = sum(svm_sps_all) / len(svm_sps_all)
        lines.append(f"  SVM throughput:  {avg_svm/1e6:.2f}M shots/s (avg across circuits)")

lines.append("")
lines.append("=" * 80)
lines.append("BOTTLENECK ANALYSIS (expected based on architecture):")
lines.append("  Tier 0: SVM ≈ Compiled (no amplitude computation; all Clifford frame ops)")
lines.append("          Bottleneck: instruction throughput, branch overhead in SVM dispatch")
lines.append("")
lines.append("  Tier 1: Compiled ≥ SVM expected (AOT removes interpreter loop)")
lines.append("          MLIR ≥ Compiled expected (whole-function LLVM optimization)")
lines.append("          Bottleneck: VALU FP32 pipeline, warp-shuffle reduction overhead")
lines.append("")
lines.append("  Tier 2: Compiled coop ≥ SVM coop expected (no function call dispatch)")
lines.append("          Bottleneck: LDS bank conflicts on butterfly stride pattern")
lines.append("          Optimization opportunity: swizzle v[] layout to avoid conflicts")
lines.append("")
lines.append("  Tier 3: Compiled global ≈ SVM global (both HBM-bandwidth limited)")
lines.append("          Bottleneck: HBM random-access latency (butterfly jumps in L2)")
lines.append("          Optimization opportunity: L2 prefetch, double-buffer pipeline")
lines.append("=" * 80)

output = "\n".join(lines)
print(output)
with open(verdict_path, 'w') as f:
    f.write(output + "\n")
PYEOF

# ---------------------------------------------------------------
# 8. Collect rocprof counter summary (if available)
# ---------------------------------------------------------------
echo ""
echo "=== Phase 7: Hardware Counter Analysis ==="
COUNTER_REPORT="$RESULTS/hw_counters.txt"

python3 - "$ROCPROF_DIR" "$COUNTER_REPORT" << 'PYEOF'
import os, sys, csv, glob

rocprof_dir = sys.argv[1]
out_path = sys.argv[2]

if not os.path.isdir(rocprof_dir):
    with open(out_path, 'w') as f:
        f.write("No rocprof data directory found.\n")
    sys.exit(0)

lines = []
lines.append("Hardware Counter Summary")
lines.append("=" * 80)

# MI300X specs for compute
HBM_PEAK_BPS = 5.3e12  # 5.3 TB/s
L2_PEAK_BPS = 10e12    # rough L2 bandwidth estimate

for subdir in sorted(os.listdir(rocprof_dir)):
    full = os.path.join(rocprof_dir, subdir)
    if not os.path.isdir(full): continue
    csvfiles = glob.glob(f"{full}/run*.csv")
    if not csvfiles: continue

    lines.append(f"\n{subdir}:")
    # Read and average across runs
    rows_all = []
    for cf in csvfiles:
        try:
            with open(cf) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows_all.append(row)
        except Exception:
            pass

    if not rows_all: continue

    # Aggregate counters of interest
    counter_keys = ['SQ_WAVES', 'SQ_INSTS_VALU', 'SQ_INSTS_VMEM', 'SQ_INSTS_LDS',
                    'TCC_HIT_sum', 'TCC_MISS_sum', 'TCP_TOTAL_READ_sum', 'TCP_TOTAL_WRITE_sum',
                    'SQ_LDS_BANK_CONFLICT']
    agg = {}
    for k in counter_keys:
        vals = [float(r.get(k, 0) or 0) for r in rows_all if k in r]
        agg[k] = sum(vals) / max(1, len(vals)) if vals else 0.0

    # Derived metrics
    valu = agg.get('SQ_INSTS_VALU', 0)
    vmem = agg.get('SQ_INSTS_VMEM', 0)
    lds = agg.get('SQ_INSTS_LDS', 0)
    l2hit = agg.get('TCC_HIT_sum', 0)
    l2miss = agg.get('TCC_MISS_sum', 0)
    l2total = l2hit + l2miss
    lds_conflict = agg.get('SQ_LDS_BANK_CONFLICT', 0)

    for k, v in agg.items():
        if v > 0:
            lines.append(f"  {k:<35}: {v:.2e}")

    if l2total > 0:
        l2_hit_rate = 100.0 * l2hit / l2total
        lines.append(f"  {'L2 hit rate':<35}: {l2_hit_rate:.1f}%")

    if valu + vmem > 0:
        mem_ratio = vmem / (valu + vmem) * 100
        lines.append(f"  {'VMEM/(VALU+VMEM) %':<35}: {mem_ratio:.1f}%")

    if lds_conflict > 0 and lds > 0:
        conflict_rate = lds_conflict / lds * 100
        lines.append(f"  {'LDS bank conflict rate %':<35}: {conflict_rate:.1f}%")

output = "\n".join(lines)
print(output)
with open(out_path, 'w') as f:
    f.write(output + "\n")
PYEOF

# ---------------------------------------------------------------
# Final: list all outputs
# ---------------------------------------------------------------
echo ""
echo "================================================================="
echo "Evaluation complete."
echo "Results in: $RESULTS/"
ls -la "$RESULTS/"
echo ""
echo "Key files:"
echo "  Benchmark CSV:    $BENCH_CSV"
echo "  Summary table:   $RESULTS/summary.txt"
echo "  Roofline report: $RESULTS/roofline.txt"
echo "  Verdict:         $RESULTS/verdict.txt"
echo "  HW counters:     $RESULTS/hw_counters.txt"
echo "  rocprof data:    $ROCPROF_DIR/"
echo ""
echo "Job completed at: $(date)"
echo "Node: $NODE"
