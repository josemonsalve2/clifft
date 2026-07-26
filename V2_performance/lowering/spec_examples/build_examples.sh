#!/bin/bash
# build_examples.sh — per-specialization before/after evidence.
#
# For each specialization class the report claims, build TWO amdgcn objects
# from the SAME v2_op_* body:
#   interp/<name>.s   the operand reached through a runtime CV2Instr load
#                     (opcode + fields read from memory), i.e. what the
#                     for(pc)switch interpreter in coop_interpreter.c does
#   spec/<name>.s     the operand called with compile-time constants, i.e.
#                     what v2_specializer.cc emits
#
# Diffing the two .s files shows exactly what the specializer bought, per class,
# with no appeal to authority. Both are compiled with the production flags from
# v2_compile_cache.cc so the comparison is apples-to-apples.
set -u
cd "$(dirname "$0")"
LLVM=${LLVM_PREFIX:-/shared/jmonsalv/software/modules/llvm/upstream_05082025}
ROOT=$(cd ../../.. && pwd)
ARCH=${ARCH:-gfx950}
mkdir -p interp spec

CFLAGS=(--target=amdgcn-amd-amdhsa -mcpu="$ARCH" -ffreestanding -nostdlib -nogpulib
        -std=c23 -O2 -ffp-contract=off -I"$ROOT/src")

echo "case,form,isa_lines,vgpr,sgpr,scratch,s_load,ds_op,v_alu,branch"

stat_of() {  # $1 = .s file
    local f=$1
    printf "%s,%s,%s,%s,%s,%s,%s,%s" \
      "$(wc -l < "$f")" \
      "$(grep -m1 '; NumVgprs:' "$f" | grep -oE '[0-9]+$' || echo 0)" \
      "$(grep -m1 '; NumSgprs:' "$f" | grep -oE '[0-9]+$' || echo 0)" \
      "$(grep -m1 '; ScratchSize:' "$f" | grep -oE '[0-9]+$' || echo 0)" \
      "$(grep -cE '^\s+s_load' "$f")" \
      "$(grep -cE '^\s+ds_(read|write)' "$f")" \
      "$(grep -cE '^\s+v_' "$f")" \
      "$(grep -cE '^\s+s_(cbranch|branch)' "$f")"
}

for c in cases/*.c; do
    name=$(basename "$c" .c)
    for form in interp spec; do
        def=$([ "$form" = spec ] && echo -DSPEC_FORM=1 || echo -DSPEC_FORM=0)
        tier=$(grep -m1 '^// TIER:' "$c" | sed 's|// TIER: *||')
        tflag=$([ "$tier" = register ] && echo -DV2_REGISTER=1 || echo)
        "$LLVM/bin/clang" "${CFLAGS[@]}" $def $tflag -S -o "$form/$name.s" "$c" 2>"$form/$name.err" || {
            echo "$name,$form,FAILED,,,,,,,"; continue; }
        echo "$name,$form,$(stat_of "$form/$name.s")"
    done
done
