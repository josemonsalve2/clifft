#!/bin/bash
#SBATCH --job-name=eval-3way
#SBATCH --partition=mi300x
#SBATCH --nodelist=rad-mi300x-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:50:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/eval_3way_mi300x_%j.log


# Full 3-Way Performance Evaluation: SVM vs Compiled(HSA) vs MLIR
# All circuit tiers, 20 runs each, MI300X gfx942

set +e
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="$BASE/results/eval_3way_${TIMESTAMP}_mi300x"
mkdir -p "$RESULTS"

# ROCm setup
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

# MLIR tools
LLVM_PREFIX="/shared/jmonsalv/software/modules/llvm/upstream_05082025"
[ -x "$LLVM_PREFIX/bin/mlir-opt" ] && export PATH="$LLVM_PREFIX/bin:$PATH"

echo "=== 3-Way Performance Evaluation ==="
echo "Node: $(hostname)  Job: ${SLURM_JOB_ID:-n/a}  Date: $(date)"
rocm-smi --showid --showproductname 2>/dev/null | head -5 || true
GPU_ARCH=$(rocminfo 2>/dev/null | grep -m1 "Name:.*gfx" | grep -oE "gfx[0-9]+" || echo "gfx942")
ROCM_VER=$(cat "$ROCM_PATH/.info/version" 2>/dev/null || echo "unknown")
echo "GPU: $GPU_ARCH  ROCm: $ROCM_VER"

# Build — try with MLIR, fall back to without
BUILD="$BASE/build-gpu-eval"
BINARY="$BUILD/run_gpu"
HAS_MLIR=false

if [ ! -f "$BINARY" ]; then
    echo "=== Building (MLIR=ON attempt) ==="
    mkdir -p "$BUILD" && cd "$BUILD"
    cmake "$BASE" -DCLIFFT_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
        -DCMAKE_BUILD_TYPE=Release -DCLIFFT_BUILD_PROFILER=ON \
        -DCLIFFT_ENABLE_MLIR=ON 2>&1 | tail -10
    make -j$(nproc) run_gpu 2>&1
    if [ $? -eq 0 ]; then
        HAS_MLIR=true
        echo "MLIR build succeeded"
    else
        echo "MLIR build failed, rebuilding without MLIR"
        rm -rf "$BUILD"
        mkdir -p "$BUILD" && cd "$BUILD"
        cmake "$BASE" -DCLIFFT_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
            -DCMAKE_BUILD_TYPE=Release -DCLIFFT_BUILD_PROFILER=ON 2>&1 | tail -5
        make -j$(nproc) run_gpu 2>&1 | tail -10
    fi
    cd "$BASE"
fi

[ -x "$BINARY" ] || { echo "Build failed completely"; exit 1; }
echo "Binary: $BINARY  MLIR=$HAS_MLIR"

SHOTS=1000000
RUNS=20
WARMUP=3
CSV="$RESULTS/bench.csv"
echo "mode,label,tier,peak_rank,qubits,run,shots,actual_rank,passed,sample_sec,kernel_sec,shots_per_sec,node,gpu_arch,rocm_version" > "$CSV"

extract_field() { echo "$1" | grep "\"$2\"" | head -1 | grep -oE '[0-9]+\.?[0-9eE+\-]*' || true; }

run_mode() {
    local MODE=$1 FLAG=$2 CIRCUIT=$3 LABEL=$4 TIER=$5 RANK=$6 QUBITS=$7
    [ -f "$CIRCUIT" ] || return
    for _w in $(seq 1 $WARMUP); do
        "$BINARY" --circuit "$CIRCUIT" --shots 100000 $FLAG >/dev/null 2>&1 || true
    done
    for i in $(seq 1 $RUNS); do
        OUT=$("$BINARY" --circuit "$CIRCUIT" --shots $SHOTS $FLAG 2>&1)
        RC=$?
        if [ $RC -ne 0 ]; then
            echo "$MODE,$LABEL,$TIER,$RANK,$QUBITS,$i,$SHOTS,-1,0,0,0,0,$(hostname),$GPU_ARCH,$ROCM_VER" >> "$CSV"
            continue
        fi
        SPS=$(extract_field "$OUT" "shots_per_second_sampling_only")
        SEC=$(extract_field "$OUT" "sample_seconds")
        KSEC=$(extract_field "$OUT" "kernel_seconds")
        PASSED=$(extract_field "$OUT" "passed_shots")
        ARANK=$(extract_field "$OUT" "peak_rank")
        SPS="${SPS:-0}"; SEC="${SEC:-0}"; KSEC="${KSEC:-$SEC}"; PASSED="${PASSED:-0}"; ARANK="${ARANK:-0}"
        printf "  %-8s %-22s run %2d: %12s shots/s\n" "$MODE" "$LABEL" "$i" "$SPS"
        echo "$MODE,$LABEL,$TIER,$RANK,$QUBITS,$i,$SHOTS,$ARANK,$PASSED,$SEC,$KSEC,$SPS,$(hostname),$GPU_ARCH,$ROCM_VER" >> "$CSV"
    done
}

bench() {
    local LABEL=$1 CIRCUIT=$2 TIER=$3 RANK=$4 QUBITS=$5
    [ -f "$CIRCUIT" ] || { echo "SKIP: $CIRCUIT"; return; }
    echo ""
    echo "=== $LABEL  tier=$TIER rank~$RANK q=$QUBITS ==="
    run_mode "svm"      ""         "$CIRCUIT" "$LABEL" "$TIER" "$RANK" "$QUBITS"
    run_mode "compiled"  "--hybrid"  "$CIRCUIT" "$LABEL" "$TIER" "$RANK" "$QUBITS"
    # MLIR: only for register tier (rank ≤ 4) where codegen is implemented
    if [ "$HAS_MLIR" = true ] && { [ "$TIER" = "1" ] || [ "$TIER" = "0" ]; }; then
        run_mode "mlir" "--mlir"   "$CIRCUIT" "$LABEL" "$TIER" "$RANK" "$QUBITS"
    fi
}

# TIER 0: Frame-only
bench "target_qec"       "$BASE/tests/fixtures/target_qec.stim"                    "0" "0" "0"
bench "color_d7_r7"      "$BASE/tests/fixtures/large/color_d7_r7.stim"             "0" "0" "55"
bench "surface_d13_r13"  "$BASE/tests/fixtures/large/surface_d13_r13.stim"         "0" "0" "337"
bench "rep_d5_r100"      "$BASE/tests/fixtures/large/rep_d5_r100.stim"             "0" "0" "0"

# TIER 1: Register (rank 1-4)
bench "cultivation_d5"   "$BASE/tests/fixtures/cultivation_d5.stim"                "1" "4" "0"
bench "cultivation_d7"   "$BASE/tests/fixtures/large/cultivation_d7_p0001.stim"    "1" "4" "0"
bench "surface_d7_t5"    "$BASE/tests/fixtures/large/surface_d7_t5.stim"           "1" "5" "97"
bench "rank_r1_q33_d3"   "$BASE/tests/fixtures/rank_sweep/rank_q33_r1_d3.stim"     "1" "1" "33"
bench "rank_r2_q33_d3"   "$BASE/tests/fixtures/rank_sweep/rank_q33_r2_d3.stim"     "1" "2" "33"
bench "rank_r3_q33_d3"   "$BASE/tests/fixtures/rank_sweep/rank_q33_r3_d3.stim"     "1" "3" "33"
bench "rank_r4_q33_d3"   "$BASE/tests/fixtures/rank_sweep/rank_q33_r4_d3.stim"     "1" "4" "33"

# TIER 2: LDS/Coop (rank 5-10)
bench "surface_d7_t10"   "$BASE/tests/fixtures/large/surface_d7_t10.stim"          "2" "10" "97"
bench "surface_d9_t10"   "$BASE/tests/fixtures/large/surface_d9_t10.stim"          "2" "10" "161"
bench "surface_d11_t10"  "$BASE/tests/fixtures/large/surface_d11_t10.stim"         "2" "10" "241"
bench "rank_r5_q33_d3"   "$BASE/tests/fixtures/rank_sweep/rank_q33_r5_d3.stim"     "2" "5" "33"
bench "rank_r6_q33_d3"   "$BASE/tests/fixtures/rank_sweep/rank_q33_r6_d3.stim"     "2" "6" "33"
bench "rank_r8_q33_d3"   "$BASE/tests/fixtures/rank_sweep/rank_q33_r8_d3.stim"     "2" "8" "33"
bench "rank_r10_q33_d3"  "$BASE/tests/fixtures/rank_sweep/rank_q33_r10_d3.stim"    "2" "10" "33"

# TIER 3: Global/HBM (rank 11-19)
bench "rank11_q50"       "$BASE/tests/fixtures/large/rank11_q50_d5_tier3.stim"     "3" "11" "50"
bench "extreme_r17"      "$BASE/tests/fixtures/large/extreme_r17_q200_d15.stim"    "3" "17" "200"
bench "extreme_r19"      "$BASE/tests/fixtures/large/extreme_r19_q200_d15.stim"    "3" "19" "200"
bench "rank_r12_q33_d3"  "$BASE/tests/fixtures/rank_sweep/rank_q33_r12_d3.stim"    "3" "12" "33"
bench "rank_r15_q33_d3"  "$BASE/tests/fixtures/rank_sweep/rank_q33_r15_d3.stim"    "3" "15" "33"
bench "rank_r19_q33_d3"  "$BASE/tests/fixtures/rank_sweep/rank_q33_r19_d3.stim"    "3" "19" "33"

echo ""
echo "=== Analysis ==="
python3 "$BASE/scripts/analyze_perf.py" "$CSV" -o "$RESULTS/report.txt" 2>&1 | head -80
echo ""
echo "CSV: $CSV"
echo "Report: $RESULTS/report.txt"
echo "Done: $(date)"
