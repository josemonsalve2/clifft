#!/bin/bash
#SBATCH --job-name=rpc-ab
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/rpc_ab_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
STIM="$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== rocprof-compute / omniperf A/B comparison ==="
echo "Node: $(hostname)"

# Check tool availability
RPC=""
for tool in rocprof-compute omniperf; do
    if command -v $tool &>/dev/null; then RPC=$tool; break; fi
done
echo "Tool: ${RPC:-NOT FOUND}"

if [ -z "$RPC" ]; then
    echo "Falling back to rocprofv3 with hardware counters"
    
    # Warm up
    $BIN --circuit "$STIM" --shots 100 --seed 42 >/dev/null 2>&1
    timeout 120 $BIN --circuit "$STIM" --shots 100 --seed 42 --mlir >/dev/null 2>&1
    
    OUTDIR="$BASE/results/rocprof_counters_$$"
    mkdir -p "$OUTDIR"
    
    # Create counter input file
    cat > "$OUTDIR/counters.txt" << 'COUNTERS'
pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS SQ_INSTS_FLAT_LDS_ONLY
pmc: TCC_HIT_sum TCC_MISS_sum FETCH_SIZE WRITE_SIZE
pmc: SQ_WAIT_INST_ANY SQ_ACTIVE_INST_ANY
COUNTERS
    
    echo ""
    echo "=== rocprofv3 SVM with counters ==="
    rocprofv3 -i "$OUTDIR/counters.txt" --kernel-trace --stats \
        -d "$OUTDIR/svm" \
        -- $BIN --circuit "$STIM" --shots 100000 --seed 42 2>&1 | tail -5
    echo ""
    echo "--- SVM counter results ---"
    for f in "$OUTDIR/svm"/*/*.csv "$OUTDIR/svm"/*.csv; do
        [ -f "$f" ] || continue
        echo "File: $(basename $f)"
        cat "$f"
        echo ""
    done
    
    echo ""
    echo "=== rocprofv3 MLIR with counters ==="
    rocprofv3 -i "$OUTDIR/counters.txt" --kernel-trace --stats \
        -d "$OUTDIR/mlir" \
        -- $BIN --circuit "$STIM" --shots 100000 --seed 42 --mlir 2>&1 | tail -5
    echo ""
    echo "--- MLIR counter results ---"
    for f in "$OUTDIR/mlir"/*/*.csv "$OUTDIR/mlir"/*.csv; do
        [ -f "$f" ] || continue
        echo "File: $(basename $f)"
        cat "$f"
        echo ""
    done
    
    echo ""
    echo "=== rocprofv3 Hybrid with counters ==="
    rocprofv3 -i "$OUTDIR/counters.txt" --kernel-trace --stats \
        -d "$OUTDIR/hybrid" \
        -- $BIN --circuit "$STIM" --shots 100000 --seed 42 --hybrid 2>&1 | tail -5
    echo ""
    echo "--- Hybrid counter results ---"
    for f in "$OUTDIR/hybrid"/*/*.csv "$OUTDIR/hybrid"/*.csv; do
        [ -f "$f" ] || continue
        echo "File: $(basename $f)"
        cat "$f"
        echo ""
    done
else
    echo "Using $RPC for profiling"
    
    # Warm up
    $BIN --circuit "$STIM" --shots 100 --seed 42 >/dev/null 2>&1
    timeout 120 $BIN --circuit "$STIM" --shots 100 --seed 42 --mlir >/dev/null 2>&1
    
    OUTDIR="$BASE/results/rpc_$$"
    
    echo ""
    echo "=== $RPC profile: SVM ==="
    $RPC profile -n svm_run --path "$OUTDIR" -- \
        $BIN --circuit "$STIM" --shots 100000 --seed 42 2>&1 | tail -10
    
    echo ""
    echo "=== $RPC profile: MLIR ==="
    $RPC profile -n mlir_run --path "$OUTDIR" -- \
        $BIN --circuit "$STIM" --shots 100000 --seed 42 --mlir 2>&1 | tail -10
    
    echo ""
    echo "=== $RPC analyze: SVM ==="
    $RPC analyze --path "$OUTDIR/svm_run" 2>&1 | head -80
    
    echo ""
    echo "=== $RPC analyze: MLIR ==="
    $RPC analyze --path "$OUTDIR/mlir_run" 2>&1 | head -80
fi

# Also do LLVM-IR / ISA comparison inline
echo ""
echo "=== LLVM-IR / ISA Comparison ==="

echo "--- MLIR kernel ---"
MLIR_HSACO=$(ls -t ~/.clifft/kernel_cache/mlir/*_gfx950.hsaco 2>/dev/null | head -1)
if [ -f "$MLIR_HSACO" ]; then
    echo "File: $MLIR_HSACO ($(stat -c%s "$MLIR_HSACO") bytes)"
    $LLVM_PREFIX/bin/llvm-readelf -n "$MLIR_HSACO" 2>/dev/null | grep -E "\.name|vgpr|sgpr|private|group|flat_work|dynamic|kernarg|wavefront"
    echo "ISA lines: $($LLVM_PREFIX/bin/llvm-objdump -d "$MLIR_HSACO" 2>/dev/null | wc -l)"
    echo "Instruction mix (top 20):"
    $LLVM_PREFIX/bin/llvm-objdump -d "$MLIR_HSACO" 2>/dev/null | \
        grep -oE '\b[sv]_[a-z_]+\b' | sort | uniq -c | sort -rn | head -20
    echo ""
    echo "FP compute ops:"
    $LLVM_PREFIX/bin/llvm-objdump -d "$MLIR_HSACO" 2>/dev/null | \
        grep -cE 'v_fma|v_mul_f|v_add_f|v_sub_f|v_mac_f|v_mad_f|v_fmac'
fi

echo ""
echo "--- Hybrid kernel ---"
HYB_HSACO=$(ls -t ~/.clifft/kernel_cache/*_gfx950.hsaco 2>/dev/null | grep -v mlir | head -1)
if [ -f "$HYB_HSACO" ]; then
    echo "File: $HYB_HSACO ($(stat -c%s "$HYB_HSACO") bytes)"
    $LLVM_PREFIX/bin/llvm-readelf -n "$HYB_HSACO" 2>/dev/null | grep -E "\.name|vgpr|sgpr|private|group|flat_work|dynamic|kernarg|wavefront"
    echo "ISA lines: $($LLVM_PREFIX/bin/llvm-objdump -d "$HYB_HSACO" 2>/dev/null | wc -l)"
    echo "Instruction mix (top 20):"
    $LLVM_PREFIX/bin/llvm-objdump -d "$HYB_HSACO" 2>/dev/null | \
        grep -oE '\b[sv]_[a-z_]+\b' | sort | uniq -c | sort -rn | head -20
    echo ""
    echo "FP compute ops:"
    $LLVM_PREFIX/bin/llvm-objdump -d "$HYB_HSACO" 2>/dev/null | \
        grep -cE 'v_fma|v_mul_f|v_add_f|v_sub_f|v_mac_f|v_mad_f|v_fmac'
fi

echo ""
echo "Done: $(date)"
