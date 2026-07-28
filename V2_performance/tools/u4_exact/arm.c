// One arm of the U4 prefetch byte-exactness test (see run.sh). Compiled twice,
// once per V2_U4_PREFETCH value, with WRAPNAME distinguishing the symbols.
//
// V2_REGISTER suppresses the amdgcn intrinsics and LDS externs so the header
// compiles for the host; V2_STRIDE and v2_tid() are then redefined to the
// coop/global values, because stride 1 cannot exercise the cross-thread half
// of the disjointness claim. The U4 sweep has no barrier inside its loop, so
// nothing else needs emulating.
#define V2_REGISTER 1
// v2_bid() is the one intrinsic V2_REGISTER does not macro away. It is unused
// on the U4 path; stub it so the build is warning-clean and a real diagnostic
// is not lost in the noise.
#define __builtin_amdgcn_workgroup_id_x() 0u
#include "clifft/gpu/mlir/v2/v2_ops.h"
#undef V2_STRIDE
#define V2_STRIDE 256u
static u32 g_tid = 0;
#undef v2_tid
#define v2_tid() (g_tid)
#include "u4_loop_only.h"
void WRAPNAME(void* st, CV2Complex* v, u32 k, u32 lo, u32 hi,
              const CV2FusedU4Entry* f, u32 cp, u32 tid) {
    g_tid = tid;
    u4_strided((V2State*)st, v, k, lo, hi, f, cp);
}
