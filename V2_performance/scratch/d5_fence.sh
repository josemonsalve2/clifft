#!/bin/bash
# Verify the barrier memory-fence fix for the coop_r10_n1720 gate failure.
#
# Root cause: v2_barrier() was a bare __builtin_amdgcn_s_barrier(). LLVM models
# that intrinsic as IntrNoMem, so it is an EXECUTION barrier only -- no
# `s_waitcnt lgkmcnt(0)` is emitted before it and the scheduler may move LDS
# accesses across it. Measured on the pre-fix ISA: 1439/1509 barriers in the
# specialized kernel (95.4%) had a ds_* op in flight with no preceding lgkm
# wait, vs 52/81 (64.2%) in the interpreter. Both are wrong; the specializer is
# 27x more exposed, which is why only it failed the gate.
#
# The fix wraps s_barrier in a release/acquire workgroup fence pair -- exactly
# what HIP's __syncthreads() expands to, which is why the SVM backend was immune.
#
# Arm 1 (selftest) is the decisive one: spec-vs-spec must now be deterministic
# on 6/6 seeds. Arm 2 runs the real gate. Arm 3 re-checks the ISA.
set -x
# build_v2.sh runs under `set -u` and reads $CLIFFT, so it must be exported.
export CLIFFT=/shared/jmonsalv/quantum/clifft_rl/clifft
cd "$CLIFFT"
# The .hsaco custom command in cmake/ClifftAmdgcn.cmake lists DEPENDS "${_src}"
# only -- the C file, NOT the headers it includes. A change to v2_ops.h (which
# is where the barrier lives) therefore does NOT invalidate the interpreter
# .hsaco, and an incremental build would silently keep serving the PRE-FIX
# kernel. Delete them so they are forced to regenerate.
rm -f build-v2-nohip/*.hsaco
bash V2_performance/scratch/build_v2.sh 2>&1 | tail -8
ls -la --time-style=+%m-%d_%H:%M build-v2-nohip/*.hsaco

CIRC=tests/fixtures/large/circuit_d5_p0.001.stim
[ -f "$CIRC" ] || { echo "MISSING CIRCUIT $CIRC"; exit 1; }
[ -x ./build-v2-nohip/run_v2 ] || { echo "BUILD FAILED - no run_v2"; exit 1; }
echo "CIRCUIT=$CIRC"

export V2_SPECIALIZE=1
BASE=/shared/jmonsalv/quantum/clifft_rl/clifft/V2_performance/scratch/fence_cache
rm -rf "$BASE"; mkdir -p "$BASE/self" "$BASE/gate"

echo "########## ARM 1: selftest (expect spec-vs-spec deterministic 6/6) ##########"
V2_SPEC_CACHE_DIR="$BASE/self" V2_GATE_SELFTEST=1 V2_GATE_SHOTS=5000 \
  ./build-v2-nohip/run_v2 --circuit "$CIRC" --shots 500 --seed 42 2>&1 | grep -E 'selftest|gate|spec' | head -40

echo "########## ARM 2: real gate ##########"
V2_SPEC_CACHE_DIR="$BASE/gate" V2_GATE_VERBOSE=1 \
  ./build-v2-nohip/run_v2 --circuit "$CIRC" --shots 500 --seed 42 2>&1 | tail -40

echo "########## ARM 3: ISA barrier audit ##########"
LLVM=/shared/jmonsalv/software/modules/llvm/upstream_05082025
for f in "$BASE"/gate/*.hsaco; do
  echo "--- $f"; $LLVM/bin/llvm-objdump -d --mcpu=gfx950 "$f" > "$f.s" 2>/dev/null
done
python3 - "$BASE" <<'PY'
import sys,glob
for f in sorted(glob.glob(sys.argv[1]+"/gate/*.hsaco.s")):
    full=[l.split('//')[0].strip() for l in open(f) if l.split('//')[0].strip()]
    nb=pre=0
    for n,l in enumerate(full):
        if not l.startswith('s_barrier'): continue
        nb+=1
        if n>0 and 'lgkmcnt(0)' in full[n-1]: pre+=1
    print(f"{f.split('/')[-1]:50s} barriers={nb:6d} with lgkm-wait immediately before={pre:6d} ({100*pre/max(nb,1):.1f}%)")
PY

echo "########## ARM 4: every coop fixture, gate verdicts ##########"
# The .gate verdict is keyed on the generated-C hash, which does NOT change when
# v2_ops.h changes -- a stale "0" would hide the fix. Each arm uses a fresh
# V2_SPEC_CACHE_DIR so every verdict below is recomputed from scratch.
mkdir -p "$BASE/all"
for C in tests/fixtures/large/*.stim; do
  [ -f "$C" ] || continue
  R=$(V2_SPEC_CACHE_DIR="$BASE/all" V2_SPECIALIZE_VERBOSE=1 \
      ./build-v2-nohip/run_v2 --circuit "$C" --shots 500 --seed 42 2>&1 \
      | grep -E '"match"|FAILED correctness' | tr '\n' ' ')
  echo "$(basename "$C"): $R"
done
echo "--- gate verdicts (1=specialized used, 0=fell back to interpreter) ---"
for g in "$BASE"/all/*.gate; do [ -f "$g" ] && echo "  $(basename "$g" .hsaco.gate): $(cat "$g")"; done
