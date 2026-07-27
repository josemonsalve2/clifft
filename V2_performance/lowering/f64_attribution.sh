#!/usr/bin/env bash
# f64_attribution.sh — settle WHERE V1's f64 instruction volume comes from.
#
# The raw ISA census says V1 issues 4,347 v_*_f64 for circuit_d3 against V2's
# 187. Read naively that says "V1 did its amplitude arithmetic in f64 and V2
# did it in f32", which would be a precision difference, not a codegen one.
# That reading is wrong, and this script is the experiment that shows it.
#
# Method: take V1's own stage-3 IR (post-mlir-translate, pre-opt) and compile
# it twice through the exact pipeline V1 uses (opt -O2 | llc -O2):
#
#   A  as-is                     -> reproduces the shipped artifact byte-for-byte
#   B  clifft_log + clifft_draw_next_noise forced `noinline`
#
# Nothing else differs. B's kernel therefore contains exactly V1's amplitude
# and PRNG-conversion f64 and none of the transcendental expansion. The delta
# between A and B is the inlined log() polynomial, and nothing else.
#
# Emits a CSV on stdout. Runs on the login node -- no GPU needed, this is a
# compile-only experiment.
set -u
LP=${LLVM_PREFIX:-/shared/jmonsalv/software/modules/llvm/upstream_05082025}
cd "$(dirname "$0")"
IN=v1/circuit_d3.3_translate.ll
W=$(mktemp -d); trap 'rm -rf $W' EXIT

cp "$IN" "$W/A.ll"

# B: mark the two hand-written f64 helpers noinline. An attribute group is
# appended and referenced from both defines; the IR is otherwise untouched.
python3 - "$IN" "$W/B.ll" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
ids = [int(m) for m in re.findall(r"^attributes #(\d+) =", src, re.M)]
new = max(ids) + 1 if ids else 0
src += f"\nattributes #{new} = {{ noinline }}\n"
for fn in ("clifft_log", "clifft_draw_next_noise"):
    src = re.sub(rf"^(define [^@]*@{fn}\([^)]*\)) \{{", rf"\1 #{new} {{", src, flags=re.M)
open(sys.argv[2], "w").write(src)
PY

count() { grep -coE "$2" "$1"; }

echo "variant,isa_lines,isa_total_instrs,v_f64,v_f32,v_pk_f32,scratch_ops,scratch_bytes,log_expansions"
for v in A B; do
    "$LP/bin/opt" -O2 -S -o "$W/$v.opt.ll" "$W/$v.ll" || continue
    "$LP/bin/llc" -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 \
        -mattr=+wavefrontsize64 -O2 -o "$W/$v.s" "$W/$v.opt.ll" || continue
    # 0x3FD5555555555555 is a coefficient unique to the log polynomial, so it
    # counts inlined expansions exactly.
    echo "$v,$(wc -l <"$W/$v.s"),$(count "$W/$v.s" '^\s+[a-z]'),\
$(count "$W/$v.s" '^\s+v_[a-z0-9_]*_f64'),$(count "$W/$v.s" '^\s+v_[a-z0-9_]*_f32'),\
$(count "$W/$v.s" '^\s+v_pk_[a-z0-9_]*_f32'),$(count "$W/$v.s" 'scratch_'),\
$(grep -E '^; ScratchSize:' "$W/$v.s" | tail -1 | sed 's/.*: *//'),\
$(grep -c '0x3FD5555555555555' "$W/$v.opt.ll")" | tr -d ' '
done

# Sanity: A must reproduce the committed artifact, or the experiment is void.
if ! diff -q "$W/A.s" v1/circuit_d3.5_isa.s >/dev/null; then
    echo "WARNING: variant A did not reproduce v1/circuit_d3.5_isa.s" >&2
fi
