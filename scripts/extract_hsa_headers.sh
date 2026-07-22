#!/bin/bash
#SBATCH --job-name=hsa-headers
#SBATCH --partition=mi300x
#SBATCH --nodelist=rad-mi300x-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/hsa_headers_%j.log

# Extract HSA header files from compute node for offline analysis
set +e
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
OUT="$BASE/results/hsa_headers"
mkdir -p "$OUT"

echo "=== HSA Headers from $(hostname) ==="
echo "ROCm path:"
ls -d /opt/rocm* 2>/dev/null

# Find HSA headers
HSA_INC=$(find /opt/rocm* -name "hsa.h" -path "*/include/hsa/*" 2>/dev/null | head -1)
HSA_DIR=$(dirname "$HSA_INC" 2>/dev/null)
echo "HSA include dir: $HSA_DIR"

if [ -d "$HSA_DIR" ]; then
    # Copy all HSA headers
    cp "$HSA_DIR"/hsa.h "$OUT/" 2>/dev/null
    cp "$HSA_DIR"/hsa_ext_amd.h "$OUT/" 2>/dev/null
    cp "$HSA_DIR"/hsa_ext_finalize.h "$OUT/" 2>/dev/null
    cp "$HSA_DIR"/hsa_ext_image.h "$OUT/" 2>/dev/null
    cp "$HSA_DIR"/hsa_api_trace.h "$OUT/" 2>/dev/null
    cp "$HSA_DIR"/amd_hsa_*.h "$OUT/" 2>/dev/null
    ls -la "$OUT/"
    echo ""
    echo "=== Key API signatures ==="
    # Extract the most important function declarations
    grep -E "^hsa_status_t.*hsa_(init|shut_down|iterate_agents|agent_get_info|queue_create|signal_create|signal_wait|executable_create|executable_load|executable_get_symbol|executable_symbol_get_info|code_object_reader|memory)" "$HSA_INC" 2>/dev/null | head -40
    echo ""
    grep -E "^hsa_status_t.*hsa_amd_(memory_pool|memory_async_copy|memory_fill|agents_allow_access)" "$HSA_DIR/hsa_ext_amd.h" 2>/dev/null | head -30
    echo ""
    echo "=== AQL Dispatch Packet ==="
    grep -A20 "typedef struct.*hsa_kernel_dispatch_packet" "$HSA_INC" 2>/dev/null | head -25
    echo ""
    echo "=== Memory Pool Info ==="
    grep -E "HSA_AMD_MEMORY_POOL|hsa_amd_memory_pool_" "$HSA_DIR/hsa_ext_amd.h" 2>/dev/null | head -30
else
    echo "HSA headers not found!"
    find /opt/rocm* -name "*.h" -path "*hsa*" 2>/dev/null | head -10
fi

echo ""
echo "=== CMake HSA config ==="
cat /opt/rocm*/lib/cmake/hsa-runtime64/hsa-runtime64-config.cmake 2>/dev/null | head -30

echo "Done: $(date)"
