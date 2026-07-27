#!/bin/bash
#SBATCH --job-name=prof-300x
#SBATCH --partition=mi300x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/prof_mi300x_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-hsa"

for ROCM in /opt/rocm-6.4.0 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "Node: $(hostname)"
echo "ROCm: $ROCM_PATH"
echo "LLVM: $LLVM_PREFIX"
echo ""

# Step 1: Build for MI300X (gfx942)
echo "=== Step 1: Build for MI300X ==="
mkdir -p "$BUILD"
cd "$BUILD"
if [ ! -f CMakeCache.txt ]; then
    cmake "$BASE" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCLIFFT_ENABLE_HIP=ON \
        -DCLIFFT_ENABLE_MLIR=ON \
        -DCMAKE_HIP_ARCHITECTURES=gfx942 \
        2>&1 | tail -5
fi
make -j$(nproc) 2>&1 | tail -5
RC=$?
if [ $RC -ne 0 ]; then
    echo "BUILD FAILED"
    make -j1 2>&1 | grep "error:" | head -10
    exit 1
fi
echo "BUILD OK"
BIN="$BUILD/run_gpu"
echo ""

# Step 2: Warm up — compile MLIR kernels
echo "=== Step 2: Kernel compilation ==="
STIM_REG="$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim"
STIM_REP="$BASE/tests/fixtures/incremental/06_small_circuits/rep_d3.stim"
rm -rf "$HOME/.clifft/kernel_cache/mlir" 2>/dev/null

echo "--- MLIR register warmup ---"
timeout 120 $BIN --circuit "$STIM_REG" --shots 100 --seed 42 --mlir 2>&1 | grep -E "compiled|loaded|bound"
echo "--- SVM warmup ---"
timeout 30 $BIN --circuit "$STIM_REG" --shots 100 --seed 42 2>/dev/null
echo "--- Hybrid warmup ---"
timeout 120 $BIN --circuit "$STIM_REG" --shots 100 --seed 42 --hybrid 2>&1 | grep -E "compiled|loaded|bound"
echo ""

# Step 3: Performance comparison — SVM vs MLIR vs Hybrid
echo "=== Step 3: Performance (register tier, frame_h) ==="
for shots in 10000 100000 1000000; do
    echo "--- $shots shots ---"
    svm_t=$(timeout 30 $BIN --circuit "$STIM_REG" --shots $shots --seed 42 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
    mlir_t=$(timeout 30 $BIN --circuit "$STIM_REG" --shots $shots --seed 42 --mlir 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
    hyb_t=$(timeout 30 $BIN --circuit "$STIM_REG" --shots $shots --seed 42 --hybrid 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
    echo "  SVM:    ${svm_t}s"
    echo "  MLIR:   ${mlir_t}s"
    echo "  Hybrid: ${hyb_t}s"
    if [ -n "$svm_t" ] && [ -n "$mlir_t" ]; then
        echo "  MLIR/SVM: $(python3 -c "print(f'{float(\"$mlir_t\")/float(\"$svm_t\"):.2f}x')" 2>/dev/null)"
    fi
done

echo ""
echo "=== Step 3b: Performance (rep_d3, rank 4) ==="
timeout 120 $BIN --circuit "$STIM_REP" --shots 100 --seed 42 --mlir 2>/dev/null
for shots in 10000 100000; do
    echo "--- $shots shots ---"
    svm_t=$(timeout 30 $BIN --circuit "$STIM_REP" --shots $shots --seed 42 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
    mlir_t=$(timeout 30 $BIN --circuit "$STIM_REP" --shots $shots --seed 42 --mlir 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
    echo "  SVM:  ${svm_t}s"
    echo "  MLIR: ${mlir_t}s"
    if [ -n "$svm_t" ] && [ -n "$mlir_t" ]; then
        echo "  MLIR/SVM: $(python3 -c "print(f'{float(\"$mlir_t\")/float(\"$svm_t\"):.2f}x')" 2>/dev/null)"
    fi
done

# Step 4: Correctness check
echo ""
echo "=== Step 4: Correctness vs SVM ==="
PASS=0; FAIL=0; TOTAL=0
for stim in "$BASE"/tests/fixtures/incremental/01_frame_only/frame_h.stim \
            "$BASE"/tests/fixtures/incremental/06_small_circuits/rep_d3.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r0_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r2_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r3_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r4_d1.stim; do
    [ -f "$stim" ] || continue
    TOTAL=$((TOTAL + 1))
    name=$(basename "$(dirname "$stim")")/$(basename "$stim")
    svm_p=$(timeout 30 $BIN --circuit "$stim" --shots 1000 --seed 42 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    mlir_p=$(timeout 120 $BIN --circuit "$stim" --shots 1000 --seed 42 --mlir 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    if [ "$svm_p" = "$mlir_p" ]; then
        echo "  PASS: $name (passed=$svm_p)"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $name (SVM=$svm_p MLIR=$mlir_p)"
        FAIL=$((FAIL + 1))
    fi
done
echo "Results: $PASS/$TOTAL passed, $FAIL failed"

# Step 5: LLVM-IR analysis
echo ""
echo "=== Step 5: LLVM-IR analysis ==="
# Get the simplest MLIR .hsaco for analysis
HSACO_MLIR=$(ls -t $HOME/.clifft/kernel_cache/mlir/*_gfx942.hsaco 2>/dev/null | head -1)
HSACO_HYB=$(ls -t $HOME/.clifft/kernel_cache/*_gfx942.hsaco 2>/dev/null | grep -v mlir | head -1)

if [ -f "$HSACO_MLIR" ]; then
    echo "--- MLIR kernel metadata ---"
    $LLVM_PREFIX/bin/llvm-readelf -n "$HSACO_MLIR" 2>/dev/null | grep -E "\.name|vgpr_count|sgpr_count|private_segment|group_segment|kernarg_segment|flat_work_group|wavefront_size|dynamic_stack|max_flat_work"
    echo ""
    echo "--- MLIR ISA instruction count ---"
    $LLVM_PREFIX/bin/llvm-objdump -d "$HSACO_MLIR" 2>/dev/null | wc -l
    echo ""
    echo "--- MLIR ISA first 30 instructions ---"
    $LLVM_PREFIX/bin/llvm-objdump -d "$HSACO_MLIR" 2>/dev/null | head -40
fi

if [ -f "$HSACO_HYB" ]; then
    echo ""
    echo "--- Hybrid (SVM codegen) kernel metadata ---"
    $LLVM_PREFIX/bin/llvm-readelf -n "$HSACO_HYB" 2>/dev/null | grep -E "\.name|vgpr_count|sgpr_count|private_segment|group_segment|kernarg_segment|flat_work_group|wavefront_size|dynamic_stack|max_flat_work"
    echo ""
    echo "--- Hybrid ISA instruction count ---"
    $LLVM_PREFIX/bin/llvm-objdump -d "$HSACO_HYB" 2>/dev/null | wc -l
fi

# Step 6: rocprofv3 kernel trace
echo ""
echo "=== Step 6: rocprofv3 kernel trace ==="
ROCPROF=""
for rp in rocprofv3 rocprof; do
    if command -v $rp &>/dev/null; then ROCPROF=$rp; break; fi
done
echo "rocprof: ${ROCPROF:-NOT FOUND}"

if [ -n "$ROCPROF" ]; then
    OUTDIR="$BASE/results/rocprof_mi300x_$$"
    mkdir -p "$OUTDIR"

    echo "--- SVM kernel trace (100k shots) ---"
    $ROCPROF --kernel-trace -o "$OUTDIR/svm" -- $BIN --circuit "$STIM_REG" --shots 100000 --seed 42 2>&1 | tail -3

    echo "--- MLIR kernel trace (100k shots) ---"
    $ROCPROF --kernel-trace -o "$OUTDIR/mlir" -- $BIN --circuit "$STIM_REG" --shots 100000 --seed 42 --mlir 2>&1 | tail -3

    echo "--- Hybrid kernel trace (100k shots) ---"
    $ROCPROF --kernel-trace -o "$OUTDIR/hybrid" -- $BIN --circuit "$STIM_REG" --shots 100000 --seed 42 --hybrid 2>&1 | tail -3

    echo ""
    echo "--- Extracting kernel durations ---"
    for csv in "$OUTDIR"/*/*.csv "$OUTDIR"/*.csv; do
        [ -f "$csv" ] || continue
        echo "File: $csv"
        head -1 "$csv"
        awk -F',' 'NR>1 {print $0}' "$csv" | head -5
        echo ""
    done

    # Also try sys-trace for HSA-level data
    echo "--- HSA trace (SVM 1000 shots) ---"
    $ROCPROF --sys-trace -o "$OUTDIR/svm_sys" -- $BIN --circuit "$STIM_REG" --shots 1000 --seed 42 2>&1 | tail -3
    echo "--- HSA trace (MLIR 1000 shots) ---"
    $ROCPROF --sys-trace -o "$OUTDIR/mlir_sys" -- $BIN --circuit "$STIM_REG" --shots 1000 --seed 42 --mlir 2>&1 | tail -3

    # Parse sys-trace databases
    echo ""
    echo "--- Sys-trace kernel dispatches ---"
    for db in "$OUTDIR"/*_sys*.db "$OUTDIR"/*_sys*/*.db; do
        [ -f "$db" ] || continue
        echo "DB: $db"
        python3 -c "
import sqlite3
db = sqlite3.connect('$db')
c = db.cursor()
c.execute(\"SELECT name FROM sqlite_master WHERE type='table'\")
tables = c.fetchall()
for (tbl,) in tables:
    if 'kernel' in tbl.lower() or 'dispatch' in tbl.lower():
        c.execute(f'SELECT COUNT(*) FROM {tbl}')
        cnt = c.fetchone()[0]
        print(f'  {tbl}: {cnt} rows')
        if cnt > 0:
            c.execute(f'SELECT * FROM {tbl} LIMIT 3')
            cols = [d[0] for d in c.description]
            print(f'  Columns: {cols}')
            for row in c.fetchall():
                print(f'    {row}')
db.close()
" 2>&1
    done
fi

echo ""
echo "Done: $(date)"
