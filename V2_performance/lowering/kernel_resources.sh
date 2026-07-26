#!/usr/bin/env bash
# Dump per-kernel resource footprints from every .hsaco we have on disk.
# CSV: source,file,kernel,vgpr,agpr,sgpr,vgpr_spill,sgpr_spill,scratch_bytes,lds_bytes
# Fed from the AMDHSA metadata note, i.e. what the finalizer actually committed --
# not from ISA comments, which are symbolic for extern-linked kernels.
set -u
LLVM=${LLVM:-/shared/jmonsalv/software/modules/llvm/upstream_05082025/bin}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

echo "source,file,kernel,vgpr,agpr,sgpr,vgpr_spill,sgpr_spill,scratch_bytes,lds_bytes"
# Dedupe by basename: the spec cache is content-hashed, so the same file shows up
# under build-v2-nohip/ and every per-run scratch dir. One row per distinct kernel.
find "$ROOT/build-v2-nohip" "$ROOT/V2_performance/scratch" -name '*.hsaco' 2>/dev/null \
  | awk -F/ '!seen[$NF]++' | sort -t/ -k99 | while read -r h; do
    case "$(basename "$h")" in
        clifft_v2_*.hsaco) src=interpreter ;;
        *) src=specialized ;;
    esac
    "$LLVM/llvm-readelf" --notes "$h" 2>/dev/null | awk -v src="$src" -v f="$(basename "$h")" '
        # $NF, not $2: list entries render as "- .key:  value" so the field
        # index shifts depending on whether the key opens a YAML list item.
        /\.name: *clifft/ { k=$NF }
        /\.vgpr_count:/        { vgpr=$NF }
        /\.agpr_count:/        { agpr=$NF }
        /\.sgpr_count:/        { sgpr=$NF }
        /\.vgpr_spill_count:/  { vs=$NF }
        /\.sgpr_spill_count:/  { ss=$NF }
        /\.private_segment_fixed_size:/ { scr=$NF }
        /\.group_segment_fixed_size:/   { lds=$NF }
        END { printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", src, f, k, vgpr, agpr, sgpr, vs, ss, scr, lds }'
done
