#!/bin/bash
# ir_density.sh -- source-IR lines emitted per bytecode instruction, for all
# three per-circuit code generators. This is the single number that separates
# V2's architecture from V1's: V1 and Hybrid emit a per-instruction BLOCK, V2
# emits a per-instruction CALL.
#
# v1 MLIR  : docs/v2/ir_reference/<circuit>.mlir          (hand-emitted MLIR text)
# Hybrid   : docs/v2/ir_reference/<circuit>.hybrid.hip    (HIP C++ string templates)
# v2       : build-v2-nohip/v2_spec_cache/<key>.c         (specializer output)
#
# The instruction count comes from the v2 spec-cache key (`_n<NNNN>_`), which the
# specializer writes from flat.instrs.size() -- the same bytecode all three
# backends consume. Run from the clifft repo root.
set -u
CLIFFT="${CLIFFT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$CLIFFT"
IR=docs/v2/ir_reference
SPEC=build-v2-nohip/v2_spec_cache

# circuit -> spec-cache basename. Paired by hand because the ir_reference corpus
# predates the spec cache and uses different names for the same fixtures.
pairs="
reg_frame_h           reg_r0_n4_5692d2913637cdb5
reg_four_t            reg_r0_n11_31516aa85504261c
reg_circuit_d3        reg_r4_n344_43971bba89d44888
coop_qv10             coop_r10_n140_4dfa2381ce3e045a
coop_circuit_d5       coop_r10_n1720_977e1e830813621d
coop_surface_d7_t10   coop_r7_n4134_112c068fdca1d2d9
glob_surface_d7_t19   global_r12_n4296_44f8c3889ac430aa
"

echo "circuit,instrs,v1_mlir_lines,v1_per_instr,hybrid_hip_lines,hybrid_per_instr,v2_c_lines,v2_per_instr"
echo "$pairs" | while read -r c key; do
    [ -n "$c" ] || continue
    n=$(echo "$key" | sed 's/.*_n\([0-9]*\)_.*/\1/')
    wcl() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo ""; }
    m=$(wcl "$IR/$c.mlir"); h=$(wcl "$IR/$c.hybrid.hip"); v=$(wcl "$SPEC/$key.c")
    per() { [ -n "$1" ] && echo "scale=2; $1/$n" | bc || echo ""; }
    echo "$c,$n,$m,$(per "$m"),$h,$(per "$h"),$v,$(per "$v")"
done
