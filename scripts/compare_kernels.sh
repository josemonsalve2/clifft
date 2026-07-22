#!/bin/bash
#SBATCH --job-name=cmp-kernels
#SBATCH --partition=mi300x
#SBATCH --nodelist=rad-mi300x-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/compare_kernels_%j.log

# Compare ISA between AOT sample_kernel and compiled_sample_kernel
set +e
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

OBJDUMP="$ROCM_PATH/lib/llvm/bin/llvm-objdump"
BINARY="$BASE/build-gpu-hsa/run_gpu"
[ -x "$BINARY" ] || BINARY="$BASE/build-gpu-node/run_gpu"

echo "=== Kernel Comparison ==="
echo "Node: $(hostname)  Date: $(date)"

# 1. Extract AOT kernel info from the binary
echo ""
echo "=== AOT KERNEL (sample_kernel, embedded in run_gpu) ==="
echo "--- Symbol table ---"
$OBJDUMP --mcpu=gfx942 -t "$BINARY" 2>/dev/null | grep -i "sample_kernel\|\.kd" | head -10
echo ""
echo "--- Code object bundles ---"
$OBJDUMP --mcpu=gfx942 -d "$BINARY" 2>/dev/null | grep -c "v_" | head -1
echo "(total v_ instructions in binary)"

# 2. Generate a compiled kernel, examine its .hsaco
echo ""
echo "=== COMPILED KERNEL (.hsaco via --offload-device-only) ==="
# Trigger kernel compilation for target_qec
"$BINARY" --circuit "$BASE/tests/fixtures/target_qec.stim" --shots 100 --hybrid 2>&1 | grep -i "compiled\|hash\|cache"

# Find the .hsaco
HSACO=$(ls -t ~/.clifft/kernel_cache/*.hsaco 2>/dev/null | head -1)
if [ -n "$HSACO" ]; then
    echo ""
    echo "HSACO: $HSACO"
    echo "Size: $(wc -c < "$HSACO") bytes"
    echo ""

    echo "--- Symbol table ---"
    $OBJDUMP --mcpu=gfx942 -t "$HSACO" 2>/dev/null | head -20

    echo ""
    echo "--- Kernel descriptor (.kd) ---"
    $OBJDUMP --mcpu=gfx942 -d --disassemble-symbols=compiled_sample_kernel.kd "$HSACO" 2>/dev/null | head -30

    echo ""
    echo "--- Kernel resource usage ---"
    $OBJDUMP --mcpu=gfx942 --disassemble "$HSACO" 2>/dev/null | grep -E "\.vgpr_count|\.sgpr_count|\.private_segment|\.group_segment|\.wavefront_size|\.kernarg_size" | head -10

    echo ""
    echo "--- ISA instruction count ---"
    TOTAL_ISA=$($OBJDUMP --mcpu=gfx942 -d "$HSACO" 2>/dev/null | grep -c "v_\|s_\|ds_\|flat_\|global_" || echo "?")
    echo "Total device instructions: $TOTAL_ISA"

    echo ""
    echo "--- First 50 ISA instructions ---"
    $OBJDUMP --mcpu=gfx942 -d "$HSACO" 2>/dev/null | grep -E "^\s+[0-9a-f]+:" | head -50
else
    echo "No .hsaco found in cache"
fi

# 3. NOW compile the same source WITH -ffast-math and compare
echo ""
echo "=== COMPILED KERNEL WITH -ffast-math ==="
# Find the cached source
SRC=$(ls -t /tmp/clifft_kernel_*.hip 2>/dev/null | head -1)
if [ -z "$SRC" ]; then
    # Trigger a compilation that leaves the source file
    echo "Generating kernel source..."
    "$BINARY" --circuit "$BASE/tests/fixtures/target_qec.stim" --shots 10 --hybrid 2>&1 >/dev/null
    SRC=$(ls -t /tmp/clifft_kernel_*.hip 2>/dev/null | head -1)
fi

if [ -n "$SRC" ]; then
    echo "Source: $SRC"
    HSACO_FAST="/tmp/clifft_kernel_fastmath.hsaco"

    echo "Compiling WITH -ffast-math..."
    $ROCM_PATH/lib/llvm/bin/clang++ \
        -x hip --offload-arch=gfx942 -O3 \
        -ffast-math \
        --offload-device-only \
        -o "$HSACO_FAST" "$SRC" 2>&1 | tail -5

    if [ -f "$HSACO_FAST" ]; then
        echo ""
        echo "--- Fastmath kernel resource usage ---"
        $OBJDUMP --mcpu=gfx942 --disassemble "$HSACO_FAST" 2>/dev/null | grep -E "\.vgpr_count|\.sgpr_count|\.private_segment|\.group_segment" | head -10

        echo ""
        echo "--- ISA instruction count (fastmath) ---"
        FAST_ISA=$($OBJDUMP --mcpu=gfx942 -d "$HSACO_FAST" 2>/dev/null | grep -c "v_\|s_\|ds_\|flat_\|global_" || echo "?")
        echo "Total device instructions (fastmath): $FAST_ISA"
        echo "Total device instructions (original): $TOTAL_ISA"

        echo ""
        echo "--- Quick benchmark: original vs fastmath ---"
        # Can't easily swap the .hsaco at runtime, but we can compare sizes
        echo "Original .hsaco: $(wc -c < "$HSACO") bytes"
        echo "Fastmath .hsaco: $(wc -c < "$HSACO_FAST") bytes"
    else
        echo "Fastmath compilation failed"
    fi
else
    echo "No source file available (kernel_cache.cc deletes it after compilation)"
    echo ""
    echo "FIXING: need to keep the temp source file for comparison"
fi

# 4. Compare the generated HIP source vs the AOT HIP source
echo ""
echo "=== KEY QUESTION: What's different about the generated kernel code? ==="
echo "The compiled kernel generates its own RNG, ShotState, etc. as raw C strings."
echo "The SVM kernel uses the AOT-compiled versions from hip_sampler.hip."
echo "The code SHOULD be identical — but the compilation flags differ:"
echo "  AOT:     -ffast-math -ffp-contract=off (from cmake Release mode)"
echo "  Dynamic: -O3 only (no -ffast-math, no other project flags)"

echo ""
echo "Done: $(date)"
