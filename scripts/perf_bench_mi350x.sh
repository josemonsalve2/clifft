#!/bin/bash
#SBATCH --job-name=perf-bench-mi350x
#SBATCH --partition=mi350x
#SBATCH --nodelist=smci350-rck-g03-d13-21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:50:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/perf_bench_mi350x_%j.log
#SBATCH --exclusive

# 3-Way GPU Kernel Benchmark — MI350X gfx950
# Same circuit set as MI300X for direct hardware comparison.

set +e
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="$BASE/results/bench_${TIMESTAMP}_mi350x"
mkdir -p "$RESULTS"

# Set up ROCm environment on this compute node
for ROCM_CANDIDATE in /opt/rocm-7.2.3 /opt/rocm-7.0.0 /opt/rocm /opt/rocm-*; do
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

# Build for gfx950 on this node
BINARY="$BASE/build-gpu-mi350x-node/run_gpu"
if [ ! -x "$BINARY" ]; then
    echo "Building for gfx950 on $(hostname)..."
    mkdir -p "$BASE/build-gpu-mi350x-node"
    cd "$BASE/build-gpu-mi350x-node"
    cmake "$BASE" \
        -DCLIFFT_ENABLE_HIP=ON \
        -DCMAKE_HIP_ARCHITECTURES=gfx950 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCLIFFT_BUILD_PROFILER=ON \
        -DCMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
        2>&1
    make -j$(nproc) run_gpu 2>&1
    cd "$BASE"
fi

GPU_ARCH=$(rocminfo 2>/dev/null | grep -m1 "Name:.*gfx" | grep -oE "gfx[0-9]+" || echo "gfx950")
ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null || echo "unknown")
SHOTS=1000000
RUNS=20
WARMUP=3

BENCH_CSV="$RESULTS/bench.csv"
echo "mode,label,tier,peak_rank,qubits,run,shots,actual_rank,passed,sample_sec,kernel_sec,shots_per_sec,node,gpu_arch,rocm_version" > "$BENCH_CSV"

extract_field() { echo "$1" | grep "\"$2\"" | head -1 | grep -oE '[0-9]+\.?[0-9eE+\-]*' || true; }

run_one() {
    local MODE=$1 FLAG=$2 CIRCUIT=$3 LABEL=$4 TIER=$5 RANK=$6 QUBITS=$7
    [ -f "$CIRCUIT" ] || return
    for _wi in $(seq 1 $WARMUP); do
        "$BINARY" --circuit "$CIRCUIT" --shots 100000 $FLAG >/dev/null 2>&1 || true
    done
    for i in $(seq 1 $RUNS); do
        set +e
        OUT=$("$BINARY" --circuit "$CIRCUIT" --shots $SHOTS $FLAG 2>&1)
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
        printf "  %-10s %-24s run %2d: %12s shots/s\n" "$MODE" "$LABEL" "$i" "$SPS"
        echo "$MODE,$LABEL,$TIER,$RANK,$QUBITS,$i,$SHOTS,$ARANK,$PASSED,$SEC,$KSEC,$SPS,$(hostname),$GPU_ARCH,$ROCM_VER" >> "$BENCH_CSV"
    done
}

bench_circuit() {
    local LABEL=$1 CIRCUIT=$2 TIER=$3 RANK=$4 QUBITS=$5
    [ -f "$CIRCUIT" ] || return
    echo "=== $LABEL  tier=$TIER rank~$RANK ==="
    run_one "svm" "" "$CIRCUIT" "$LABEL" "$TIER" "$RANK" "$QUBITS"
    run_one "compiled" "--hybrid" "$CIRCUIT" "$LABEL" "$TIER" "$RANK" "$QUBITS"
}

bench_circuit "target_qec"       "$BASE/tests/fixtures/target_qec.stim"                   "0" "0" "0"
bench_circuit "color_d7_r7"      "$BASE/tests/fixtures/large/color_d7_r7.stim"            "0" "0" "55"
bench_circuit "surface_d13_r13"  "$BASE/tests/fixtures/large/surface_d13_r13.stim"        "0" "0" "337"
bench_circuit "cultivation_d5"   "$BASE/tests/fixtures/cultivation_d5.stim"               "1" "4" "0"
bench_circuit "cultivation_d7"   "$BASE/tests/fixtures/large/cultivation_d7_p0001.stim"   "1" "4" "0"
bench_circuit "surface_d7_t5"    "$BASE/tests/fixtures/large/surface_d7_t5.stim"          "1" "5" "97"
bench_circuit "rank_r1_q33_d3"   "$BASE/tests/fixtures/rank_sweep/rank_q33_r1_d3.stim"    "1" "1" "33"
bench_circuit "rank_r4_q33_d3"   "$BASE/tests/fixtures/rank_sweep/rank_q33_r4_d3.stim"    "1" "4" "33"
bench_circuit "surface_d7_t10"   "$BASE/tests/fixtures/large/surface_d7_t10.stim"         "2" "10" "97"
bench_circuit "surface_d9_t10"   "$BASE/tests/fixtures/large/surface_d9_t10.stim"         "2" "10" "161"
bench_circuit "rank_r5_q33_d3"   "$BASE/tests/fixtures/rank_sweep/rank_q33_r5_d3.stim"    "2" "5" "33"
bench_circuit "rank_r10_q33_d3"  "$BASE/tests/fixtures/rank_sweep/rank_q33_r10_d3.stim"   "2" "10" "33"
bench_circuit "rank11_q50"       "$BASE/tests/fixtures/large/rank11_q50_d5_tier3.stim"    "3" "11" "50"
bench_circuit "extreme_r17"      "$BASE/tests/fixtures/large/extreme_r17_q200_d15.stim"   "3" "17" "200"
bench_circuit "rank_r15_q33_d3"  "$BASE/tests/fixtures/rank_sweep/rank_q33_r15_d3.stim"   "3" "15" "33"
bench_circuit "rank_r19_q33_d3"  "$BASE/tests/fixtures/rank_sweep/rank_q33_r19_d3.stim"   "3" "19" "33"

echo "Benchmark CSV: $BENCH_CSV  Done: $(date)"
