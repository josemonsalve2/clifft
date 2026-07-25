#!/bin/bash
set -u
cd "$CLIFFT"
for ROCM in /opt/rocm-7.2.3 /opt/rocm; do
  [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; \
     export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
export LLVM_PREFIX=${LLVM_PREFIX:-/shared/jmonsalv/software/modules/llvm/upstream_05082025}
export V2_NO_CPU=1
python3 V2_performance/scratch/verify_corr.py "$@"
