#!/bin/bash
# Diagnose why clang++ picks /usr/include/hip over /opt/rocm-7.2.3/include/hip.
set -u
echo "node=$(hostname)"
ls -d /opt/rocm* /usr/include/hip 2>&1
echo "--- version files ---"
cat /opt/rocm-7.2.3/.info/version 2>/dev/null
cat /usr/include/hip/hip_version.h 2>/dev/null | grep -E "HIP_VERSION_(MAJOR|MINOR|PATCH)" | head -3
echo "--- clang hip detection (default) ---"
/opt/rocm-7.2.3/lib/llvm/bin/clang++ -x hip -c /dev/null -o /dev/null --offload-arch=gfx950 -v 2>&1 | grep -iE "rocm|hip.?path|Found HIP" | head -10
echo "--- with HIP_PATH ---"
HIP_PATH=/opt/rocm-7.2.3 /opt/rocm-7.2.3/lib/llvm/bin/clang++ -x hip -c /dev/null -o /dev/null --offload-arch=gfx950 -v 2>&1 | grep -iE "rocm|hip.?path|Found HIP" | head -10
