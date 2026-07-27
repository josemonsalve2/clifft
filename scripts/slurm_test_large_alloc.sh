#!/bin/bash
#SBATCH --job-name=lg-alloc
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/lg_alloc_%j.log

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

echo "=== Large GPU Memory Allocation Test ==="
echo "Node: $(hostname)"

# Test hipMalloc with large sizes
python3 << 'PYEOF'
import ctypes, os

hip = ctypes.CDLL(os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "lib", "libamdhip64.so"))

# hipMalloc
hipMalloc = hip.hipMalloc
hipMalloc.restype = ctypes.c_int
hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]

hipFree = hip.hipFree
hipFree.restype = ctypes.c_int
hipFree.argtypes = [ctypes.c_void_p]

hipGetDeviceProperties = hip.hipGetDeviceProperties
# Skip device props for now

sizes_gb = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256]
for gb in sizes_gb:
    sz = gb * 1024 * 1024 * 1024
    ptr = ctypes.c_void_p()
    rc = hipMalloc(ctypes.byref(ptr), sz)
    if rc == 0:
        print(f"  {gb:4d} GB: OK (ptr={hex(ptr.value)})")
        hipFree(ptr)
    else:
        print(f"  {gb:4d} GB: FAILED (rc={rc})")
        break
PYEOF

echo ""
echo "Done: $(date)"
