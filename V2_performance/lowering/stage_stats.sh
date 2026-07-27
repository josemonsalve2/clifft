#!/usr/bin/env bash
# Per-stage op-histogram + size stats for the progressive-lowering corpus.
# Emits CSV: pipeline,circuit,stage,file_lines,op_kind,count
# The report uses this to show *what each stage actually changed* rather than
# asserting it. Ops are counted by their IR mnemonic; for ISA we count the
# instruction mnemonic in column 1.
set -u
cd "$(dirname "$0")"

# SSA-name character class. clang emits names with a leading '.' and an embedded
# '-' (%.atomictmp, %atomic-temp), so both must be in the class -- an earlier
# version omitted them and undercounted alloca by 3 at -O0.
NAME='[0-9a-zA-Z_.-]+'

hist() {  # $1 = file, $2 = kind (mlir|ll|isa)
    case "$2" in
        mlir) grep -oE '\bllvm\.[a-z_.]+' "$1" | sort | uniq -c | sort -rn ;;
        ll)   grep -oE "^\s+(%$NAME = )?[a-z][a-z0-9_.]*" "$1" \
                  | sed -E "s/^\s+//; s/^%$NAME = //" | sort | uniq -c | sort -rn ;;
        isa)  grep -oE '^\s+[a-z][a-z0-9_]*' "$1" | sed -E 's/^\s+//' \
                  | sort | uniq -c | sort -rn ;;
    esac
}

# Ops that must ALWAYS be emitted, even when they fall outside the top-N
# histogram. A stage that eliminates an op drives it down the ranking, so
# truncating the histogram makes "optimized away" indistinguishable from "still
# there but rare" -- and a missing CSV row reads as zero. alloca at -O2 ranks
# 36th with 3 occurrences and was misreported as 0 for exactly this reason.
ALWAYS='alloca addrspacecast call shufflevector'
TOPN=30

emit() {  # pipeline circuit stage file kind
    local p=$1 c=$2 s=$3 f=$4 k=$5
    [ -f "$f" ] || return 0
    local n; n=$(wc -l < "$f")
    local h; h=$(hist "$f" "$k")
    { echo "$h" | head -$TOPN
      # Append any ALWAYS op not already in the top N (count 0 if truly absent,
      # which is now an assertion rather than an artifact of truncation).
      for op in $ALWAYS; do
          echo "$h" | head -$TOPN | awk -v o="$op" '$2==o {f=1} END{exit !f}' && continue
          local cnt; cnt=$(echo "$h" | awk -v o="$op" '$2==o {print $1; exit}')
          echo "  ${cnt:-0} $op"
      done
    } | while read -r cnt op; do
        [ -n "$op" ] && echo "$p,$c,$s,$n,$op,$cnt"
    done
}

echo "pipeline,circuit,stage,file_lines,op_kind,count"
for f in v1/*.1_emitted.mlir; do
    c=$(basename "$f" .1_emitted.mlir)
    emit v1 "$c" 1_emitted   "v1/$c.1_emitted.mlir"   mlir
    emit v1 "$c" 2_opt       "v1/$c.2_opt.mlir"       mlir
    emit v1 "$c" 3_translate "v1/$c.3_translate.ll"   ll
    emit v1 "$c" 4_optO2     "v1/$c.4_optO2.ll"       ll
    emit v1 "$c" 5_isa       "v1/$c.5_isa.s"          isa
done
for f in v2/*.2_clangO0.ll; do
    c=$(basename "$f" .2_clangO0.ll)
    emit v2 "$c" 2_clangO0 "v2/$c.2_clangO0.ll" ll
    emit v2 "$c" 3_clangO2 "v2/$c.3_clangO2.ll" ll
    emit v2 "$c" 4_isa     "v2/$c.4_isa.s"      isa
done
