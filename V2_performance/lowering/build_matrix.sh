#!/bin/bash
# build_matrix.sh — lower every ir_reference circuit through BOTH pipelines and
# record IR size / ISA size / ScratchSize / VGPR / LDS / occupancy per stage.
#
# v1 chain: emitted .mlir -> mlir-opt(canonicalize,cse,convert-func-to-llvm)
#           -> mlir-translate -> opt -O2 -> llc  (mlir_codegen.cc:63-101)
# v2 chain: emitted .c -> clang -O2 -> llc       (v2_compile_cache.cc:122-139)
# Both end at gfx950 amdgcn ISA, so the ISA columns are directly comparable.
set -u
LP=${LLVM_PREFIX:-/shared/jmonsalv/software/modules/llvm/upstream_05082025}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
IRREF=$ROOT/docs/v2/ir_reference
OUT=$ROOT/V2_performance/lowering
W=$(mktemp -d); trap 'rm -rf $W' EXIT

# Report the final (kernel) ScratchSize, not the leaf-function ones.
kstat() {  # kstat <isa.s> <field>
  grep -E "^; $2:" "$1" | tail -1 | sed "s/^; $2: *//"
}
secs() { /usr/bin/time -f "%e" "$@" >/dev/null 2>"$W/tm"; tail -1 "$W/tm"; }

echo "pipeline,circuit,ir_lines,isa_lines,scratch,vgpr,lds,occupancy,compile_s"
for m in "$IRREF"/*.mlir; do
  [ -e "$m" ] || continue
  case "$m" in *.hybrid.*) continue;; esac
  n=$(basename "$m" .mlir)
  t0=$(secs "$LP/bin/mlir-opt" --canonicalize --cse --convert-func-to-llvm "$m" -o "$W/a.mlir") || continue
  t1=$(secs "$LP/bin/mlir-translate" --mlir-to-llvmir "$W/a.mlir" -o "$W/a.ll") || continue
  t2=$(secs timeout 1800 "$LP/bin/opt" -O2 -S -o "$W/a2.ll" "$W/a.ll") || continue
  t3=$(secs timeout 3600 "$LP/bin/llc" -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 \
         -mattr=+wavefrontsize64 -O2 -o "$W/a.s" "$W/a2.ll") || continue
  [ -s "$W/a.s" ] || continue
  echo "v1,$n,$(wc -l <"$m"),$(wc -l <"$W/a.s"),$(kstat "$W/a.s" ScratchSize),$(kstat "$W/a.s" NumVgprs),$(kstat "$W/a.s" LDSByteSize),$(kstat "$W/a.s" Occupancy),$(echo "$t0+$t1+$t2+$t3"|bc)"
done

for c in "$ROOT"/V2_performance/lowering/v2_src/*.c; do
  [ -e "$c" ] || continue
  n=$(basename "$c" .c)
  t0=$(secs "$LP/bin/clang" --target=amdgcn-amd-amdhsa -mcpu=gfx950 -ffreestanding \
        -nostdlib -nogpulib -std=c23 -O2 -ffp-contract=off -I "$ROOT/src" \
        -S -emit-llvm -o "$W/b.ll" "$c") || continue
  t1=$(secs timeout 3600 "$LP/bin/llc" -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 \
        -mattr=+wavefrontsize64 -O2 -o "$W/b.s" "$W/b.ll") || continue
  [ -s "$W/b.s" ] || continue
  echo "v2,$n,$(wc -l <"$c"),$(wc -l <"$W/b.s"),$(kstat "$W/b.s" ScratchSize),$(kstat "$W/b.s" NumVgprs),$(kstat "$W/b.s" LDSByteSize),$(kstat "$W/b.s" Occupancy),$(echo "$t0+$t1"|bc)"
done
