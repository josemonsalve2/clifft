#!/bin/bash
#SBATCH --job-name=perf-bench-3way
#SBATCH --partition=mi300x
#SBATCH --nodelist=rad-mi300x-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:50:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/perf_bench_mi300x_%j.log
#SBATCH --exclusive

# 3-Way GPU Kernel Benchmark — MI300X gfx942
# 20 runs per circuit per mode, 3 warmup, all tiers

set +e
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="$BASE/results/bench_${TIMESTAMP}_mi300x"
mkdir -p "$RESULTS"

# Set up ROCm environment — find ROCm install on compute node
for ROCM_CANDIDATE in /opt/rocm-7.2.3 /opt/rocm-7.0.0 /opt/rocm-6.3.0 /opt/rocm /opt/rocm-*; do
    if [ -d "$ROCM_CANDIDATE/lib" ]; then
        export ROCM_PATH="$ROCM_CANDIDATE"
        export PATH="$ROCM_CANDIDATE/bin:$PATH"
        export LD_LIBRARY_PATH="$ROCM_CANDIDATE/lib:${LD_LIBRARY_PATH:-}"
        echo "Using ROCm: $ROCM_CANDIDATE"
        break
    fi
done

echo "Node: $(hostname)  Job: ${SLURM_JOB_ID:-n/a}  Date: $(date)"
rocm-smi --showid --showproductname 2>/dev/null | head -8 || true
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-unset}"
echo ""

# Rebuild on this compute node to ensure correct toolchain/libraries
BUILD_DIR="$BASE/build-gpu-node"
if [ ! -f "$BUILD_DIR/run_gpu" ]; then
    echo "Building run_gpu on $(hostname)..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake "$BASE" \
        -DCLIFFT_ENABLE_HIP=ON \
        -DCMAKE_HIP_ARCHITECTURES=gfx942 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCLIFFT_BUILD_PROFILER=ON \
        -DCMAKE_BUILD_PARALLEL_LEVEL=$(nproc) 2>&1
    make -j$(nproc) run_gpu 2>&1
    cd "$BASE"
fi

BINARY="$BASE/build-gpu-node/run_gpu"
if [ ! -x "$BINARY" ]; then
    echo "ERROR: build failed, falling back to pre-built binary"
    BINARY="$BASE/build-gpu/run_gpu"
fi
echo "Binary: $BINARY"

GPU_ARCH=$(rocminfo 2>/dev/null | grep -m1 "Name:.*gfx" | grep -oE "gfx[0-9]+" || echo "gfx942")
ROCM_VER=$(cat "$ROCM_PATH/.info/version" 2>/dev/null || rocminfo 2>/dev/null | grep "ROCm Runtime" | head -1 || echo "unknown")

SHOTS=1000000
RUNS=20
WARMUP=3

BENCH_CSV="$RESULTS/bench.csv"
echo "mode,label,tier,peak_rank,qubits,run,shots,actual_rank,passed,sample_sec,kernel_sec,shots_per_sec,node,gpu_arch,rocm_version" > "$BENCH_CSV"

extract_field() { echo "$1" | grep "\"$2\"" | head -1 | grep -oE '[0-9]+\.?[0-9eE+\-]*' || true; }

run_one() {
    local MODE=$1 FLAG=$2 CIRCUIT=$3 LABEL=$4 TIER=$5 RANK=$6 QUBITS=$7
    [ -f "$CIRCUIT" ] || { echo "  SKIP $CIRCUIT"; return; }
    # Test numactl node binding validity (node 1 may not exist)
    NUMA_CMD=""
    if numactl --cpunodebind=1 --membind=1 true 2>/dev/null; then
        NUMA_CMD="numactl --cpunodebind=1 --membind=1"
    fi
    for _wi in $(seq 1 $WARMUP); do
        $NUMA_CMD "$BINARY" --circuit "$CIRCUIT" --shots 100000 $FLAG >/dev/null 2>&1 || true
    done
    for i in $(seq 1 $RUNS); do
        set +e
        OUT=$($NUMA_CMD "$BINARY" --circuit "$CIRCUIT" --shots $SHOTS $FLAG 2>&1)
        RC=$?
        set -e
        if [ $RC -ne 0 ]; then
            echo "    FAIL (rc=$RC): $MODE $LABEL run $i"
            echo "$MODE,$LABEL,$TIER,$RANK,$QUBITS,$i,$SHOTS,-1,0,0,0,0,$(hostname),$GPU_ARCH,$ROCM_VER" >> "$BENCH_CSV"
            continue
        fi
        SPS=$(extract_field "$OUT" "shots_per_second_sampling_only"); SPS="${SPS:-0}"
        SEC=$(extract_field "$OUT" "sample_seconds"); SEC="${SEC:-0}"
        KSEC=$(extract_field "$OUT" "kernel_seconds"); KSEC="${KSEC:-$SEC}"
        PASSED=$(extract_field "$OUT" "passed_shots"); PASSED="${PASSED:-0}"
        ARANK=$(extract_field "$OUT" "peak_rank"); ARANK="${ARANK:-0}"
        printf "  %-10s %-24s run %2d: %12s shots/s  kern=%ss\n" "$MODE" "$LABEL" "$i" "$SPS" "$KSEC"
        echo "$MODE,$LABEL,$TIER,$RANK,$QUBITS,$i,$SHOTS,$ARANK,$PASSED,$SEC,$KSEC,$SPS,$(hostname),$GPU_ARCH,$ROCM_VER" >> "$BENCH_CSV"
    done
}

bench_circuit() {
    local LABEL=$1 CIRCUIT=$2 TIER=$3 RANK=$4 QUBITS=$5
    [ -f "$CIRCUIT" ] || return
    echo ""
    echo "=== $LABEL  tier=$TIER rank~$RANK q=$QUBITS ==="
    echo "  [svm]"
    run_one "svm" "" "$CIRCUIT" "$LABEL" "$TIER" "$RANK" "$QUBITS"
    echo "  [compiled]"
    run_one "compiled" "--hybrid" "$CIRCUIT" "$LABEL" "$TIER" "$RANK" "$QUBITS"
}

# Quick diagnostic before starting
echo "=== Binary diagnostic ==="
set +e
"$BINARY" --diagnose 2>&1 || echo "diagnose exit: $?"
echo "=== Quick 1-shot test ==="
"$BINARY" --circuit "$BASE/tests/fixtures/target_qec.stim" --shots 100 2>&1
TEST_RC=$?
echo "Quick test exit: $TEST_RC"
set -e
echo "=== Starting benchmarks ==="

# TIER 0: Frame-only (rank=0) — no amplitude array
bench_circuit "target_qec"       "$BASE/tests/fixtures/target_qec.stim"                         "0" "0" "0"
bench_circuit "color_d7_r7"      "$BASE/tests/fixtures/large/color_d7_r7.stim"                  "0" "0" "55"
bench_circuit "surface_d13_r13"  "$BASE/tests/fixtures/large/surface_d13_r13.stim"              "0" "0" "337"
bench_circuit "rep_d5_r100"      "$BASE/tests/fixtures/large/rep_d5_r100.stim"                  "0" "0" "0"

# TIER 1: Register (rank 1-4, VGPRs)
bench_circuit "cultivation_d5"   "$BASE/tests/fixtures/cultivation_d5.stim"                     "1" "4" "0"
bench_circuit "cultivation_d7"   "$BASE/tests/fixtures/large/cultivation_d7_p0001.stim"         "1" "4" "0"
bench_circuit "surface_d7_t5"    "$BASE/tests/fixtures/large/surface_d7_t5.stim"                "1" "5" "97"
bench_circuit "rank_r1_q33_d3"   "$BASE/tests/fixtures/rank_sweep/rank_q33_r1_d3.stim"          "1" "1" "33"
bench_circuit "rank_r2_q33_d3"   "$BASE/tests/fixtures/rank_sweep/rank_q33_r2_d3.stim"          "1" "2" "33"
bench_circuit "rank_r3_q33_d3"   "$BASE/tests/fixtures/rank_sweep/rank_q33_r3_d3.stim"          "1" "3" "33"
bench_circuit "rank_r4_q33_d3"   "$BASE/tests/fixtures/rank_sweep/rank_q33_r4_d3.stim"          "1" "4" "33"

# TIER 2: LDS/Coop (rank 5-10)
bench_circuit "surface_d7_t10"   "$BASE/tests/fixtures/large/surface_d7_t10.stim"               "2" "10" "97"
bench_circuit "surface_d9_t10"   "$BASE/tests/fixtures/large/surface_d9_t10.stim"               "2" "10" "161"
bench_circuit "surface_d11_t10"  "$BASE/tests/fixtures/large/surface_d11_t10.stim"              "2" "10" "241"
bench_circuit "rank_r5_q33_d3"   "$BASE/tests/fixtures/rank_sweep/rank_q33_r5_d3.stim"          "2" "5" "33"
bench_circuit "rank_r6_q33_d3"   "$BASE/tests/fixtures/rank_sweep/rank_q33_r6_d3.stim"          "2" "6" "33"
bench_circuit "rank_r8_q33_d3"   "$BASE/tests/fixtures/rank_sweep/rank_q33_r8_d3.stim"          "2" "8" "33"
bench_circuit "rank_r10_q33_d3"  "$BASE/tests/fixtures/rank_sweep/rank_q33_r10_d3.stim"         "2" "10" "33"

# TIER 3: Global/HBM (rank 11-19)
bench_circuit "rank11_q50"        "$BASE/tests/fixtures/large/rank11_q50_d5_tier3.stim"          "3" "11" "50"
bench_circuit "extreme_r17"       "$BASE/tests/fixtures/large/extreme_r17_q200_d15.stim"         "3" "17" "200"
bench_circuit "extreme_r19"       "$BASE/tests/fixtures/large/extreme_r19_q200_d15.stim"         "3" "19" "200"
bench_circuit "rank_r12_q33_d3"   "$BASE/tests/fixtures/rank_sweep/rank_q33_r12_d3.stim"         "3" "12" "33"
bench_circuit "rank_r15_q33_d3"   "$BASE/tests/fixtures/rank_sweep/rank_q33_r15_d3.stim"         "3" "15" "33"
bench_circuit "rank_r19_q33_d3"   "$BASE/tests/fixtures/rank_sweep/rank_q33_r19_d3.stim"         "3" "19" "33"

echo ""
echo "=== Statistical Summary ==="
python3 - "$BENCH_CSV" << 'EOF'
import csv, math, sys
from collections import defaultdict

data = defaultdict(lambda: defaultdict(list))
meta = {}

with open(sys.argv[1]) as f:
    for row in csv.DictReader(f):
        sps = float(row.get('shots_per_sec', 0) or 0)
        if sps > 0:
            data[row['label']][row['mode']].append(sps)
        meta[row['label']] = (row.get('tier','?'), row.get('peak_rank','?'), row.get('qubits','?'))

def stats(vals):
    if not vals: return 0,0,0
    n, m = len(vals), sum(vals)/len(vals)
    sd = math.sqrt(sum((x-m)**2 for x in vals)/(max(1,n-1)))
    return m, sd, 100*sd/m if m > 0 else 0

def fmtd(base, other):
    if base <= 0: return 'N/A'
    return f'{(other-base)/base*100:+.1f}%'

print(f"\n{'Circuit':<24} T Rk Q   {'SVM (M/s)':>10} CV   {'Compiled':>10} CV  Delta")
print("-"*90)

for lbl in sorted(data, key=lambda l: (meta.get(l,('?','?','?'))[0], l)):
    tier, rank, q = meta.get(lbl, ('?','?','?'))
    sm, ss, scv = stats(data[lbl].get('svm',[]))
    cm, cs, ccv = stats(data[lbl].get('compiled',[]))
    print(f"{lbl:<24} {tier} {rank:>2} {q:>3}  "
          f"{sm/1e6:>9.2f}M {scv:>4.1f}%  "
          f"{cm/1e6:>9.2f}M {ccv:>4.1f}% {fmtd(sm,cm):>7}")

# Tier aggregate
print("\n=== Per-Tier Aggregate ===")
tier_svm = defaultdict(list)
tier_comp = defaultdict(list)
for lbl, modes in data.items():
    tier = meta.get(lbl,('?',))[0]
    sm = sum(modes.get('svm',[]))/max(1,len(modes.get('svm',[]))) if modes.get('svm') else 0
    cm = sum(modes.get('compiled',[]))/max(1,len(modes.get('compiled',[]))) if modes.get('compiled') else 0
    if sm > 0: tier_svm[tier].append(sm)
    if cm > 0 and sm > 0: tier_comp[tier].append(cm/sm)

for t in sorted(tier_svm):
    avgsvm = sum(tier_svm[t])/len(tier_svm[t])
    if tier_comp[t]:
        avgcomp = sum(tier_comp[t])/len(tier_comp[t])
        print(f"  Tier {t}: SVM avg {avgsvm/1e6:.2f}M shots/s  Compiled {avgcomp:.2f}x speedup")
    else:
        print(f"  Tier {t}: SVM avg {avgsvm/1e6:.2f}M shots/s  (no compiled data)")
EOF

echo ""
echo "Benchmark CSV: $BENCH_CSV"
echo "Done: $(date)"
