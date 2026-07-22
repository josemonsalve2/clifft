#!/bin/bash
# run_mfma_bench.sh -- Build and run the MFMA sweep microbenchmarks on MI300X
#
# Usage:
#   ./run_mfma_bench.sh              # build + run
#   ./run_mfma_bench.sh --run-only   # skip build, just run
#   ./run_mfma_bench.sh --profile    # run with rocprof

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build-mfma-bench"
BENCH_EXE="${BUILD_DIR}/mfma_sweep_bench"

RUN_ONLY=0
PROFILE=0
for arg in "$@"; do
    case "$arg" in
        --run-only) RUN_ONLY=1 ;;
        --profile)  PROFILE=1 ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# Try to find hipcc / ROCm
if [ -z "${ROCM_PATH:-}" ]; then
    for candidate in /opt/rocm /opt/rocm-7.2.3 /opt/rocm-6.3.0; do
        if [ -d "$candidate" ]; then
            export ROCM_PATH="$candidate"
            break
        fi
    done
fi

if [ -z "${ROCM_PATH:-}" ]; then
    echo "ERROR: ROCM_PATH not set and /opt/rocm not found."
    echo "Set ROCM_PATH to your ROCm installation directory."
    exit 1
fi

export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/lib/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"

echo "Using ROCm at: ${ROCM_PATH}"

# Build
if [ "$RUN_ONLY" -eq 0 ]; then
    echo "Building mfma_sweep_bench..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Try cmake first, fall back to direct hipcc
    if command -v cmake &>/dev/null; then
        cmake "$SCRIPT_DIR" \
            -DCMAKE_HIP_ARCHITECTURES=gfx942 \
            -DCMAKE_BUILD_TYPE=Release \
            2>&1 | tail -5
        make -j"$(nproc)" 2>&1 | tail -5
    elif command -v hipcc &>/dev/null; then
        echo "cmake not found, using hipcc directly..."
        hipcc -O3 --offload-arch=gfx942 -ffp-contract=off \
              -o mfma_sweep_bench "${SCRIPT_DIR}/mfma_sweep_bench.hip"
    else
        echo "ERROR: Neither cmake nor hipcc found in PATH."
        exit 1
    fi

    cd "$SCRIPT_DIR"
    echo "Build complete: ${BENCH_EXE}"
fi

# Check executable exists
if [ ! -f "$BENCH_EXE" ]; then
    echo "ERROR: ${BENCH_EXE} not found. Run without --run-only to build first."
    exit 1
fi

# Select GPU (use first HIP-visible device)
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

echo ""
echo "=== MFMA Sweep Microbenchmarks ==="
echo "GPU device: HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
echo ""

if [ "$PROFILE" -eq 1 ] && command -v rocprof &>/dev/null; then
    echo "Running with rocprof..."
    rocprof --stats --hip-trace --hsa-trace "$BENCH_EXE" 2>&1
    echo ""
    echo "Profiling results in: results.csv, results.stats.csv"
else
    "$BENCH_EXE"
fi
