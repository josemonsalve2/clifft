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

echo "case,form,isa_lines,instrs,vgpr,sgpr,scratch,s_load,ds_op,v_alu,branch"

# Resource counts come from the .set directives, NOT the "; NumVgprs:" comments.
# The comments print a symbolic expression -- max(56, amdgpu.max_num_vgpr) -- for
# any kernel that calls an external function, because the true count is not known
# until link time. S7 calls __ocml_log_f64, so its comment is unparseable as a
# number; scraping it with a trailing-digit regex silently yielded 0 and made the
# noise case look like it used no vector registers at all. The .set lines carry
# the same max(...) form but the leading literal IS the kernel's own count, and
# `; NumSgprs:` does not exist in this LLVM's output at all -- that column read 0
# for all 16 files. Both are parsed out of the first numeric literal here.
setval() {  # $1 = .s file, $2 = .set suffix
    grep -m1 -oE "\.set case_kernel\.$2, .*" "$1" | grep -oE '[0-9]+' | head -1
}

# A literal tab, not the escape: /usr/bin/grep here is ugrep, which does not
# expand \t inside an ERE when the script runs non-interactively (it does when
# typed at a prompt -- so this silently counted 0 in batch and the right number
# by hand). Build the character once and interpolate it.
TAB=$(printf '\t')

stat_of() {  # $1 = .s file
    local f=$1
    # isa_lines is every line in the file -- directives, comments, labels and all
    # (~205-316 of them per file, i.e. most of a small kernel). instrs counts only
    # real instructions: tab-indented, not a `.directive` and not a `; comment`.
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s" \
      "$(wc -l < "$f")" \
      "$(grep -cE "^${TAB}[a-z]" "$f")" \
      "$(setval "$f" num_vgpr)" \
      "$(setval "$f" numbered_sgpr)" \
      "$(setval "$f" private_seg_size)" \
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
