// Cross-thread byte-exactness: 256 threads, stride 256, prefetch vs baseline.
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "clifft/gpu/mlir/v2/device_abi.h"
typedef unsigned int u32; typedef unsigned long u64;
struct V2State;
void pf_run  (void* st, CV2Complex* v, u32 k, u32 lo, u32 hi, const CV2FusedU4Entry* f, u32 cp, u32 tid);
void base_run(void* st, CV2Complex* v, u32 k, u32 lo, u32 hi, const CV2FusedU4Entry* f, u32 cp, u32 tid);

static void fill(CV2Complex* v, u64 n, unsigned seed) {
    u64 s = 0x9e3779b97f4a7c15ULL ^ seed;
    for (u64 i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        v[i].re = (float)((double)(int64_t)(s >> 11) / 9.007199254740992e15);
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        v[i].im = (float)((double)(int64_t)(s >> 11) / 9.007199254740992e15);
    }
}
int main(int argc, char** argv) {
    unsigned K = argc > 1 ? (unsigned)atoi(argv[1]) : 14;
    u64 n = 1ull << K;
    CV2Complex *a = malloc(n * sizeof *a), *b = malloc(n * sizeof *b);
    CV2FusedU4Entry* fu = calloc(1, sizeof *fu);
    unsigned char sa[4096], sb[4096];
    int bad = 0, cases = 0;
    for (unsigned lo = 0; lo < K; lo++)
    for (unsigned hi = lo + 1; hi < K; hi++) {
        for (int st = 0; st < 16; st++) {
            for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) {
                fu->entries[st].matrix[r][c].re = (float)(0.37*(r+1) - 0.11*(c+1) + 0.013*st);
                fu->entries[st].matrix[r][c].im = (float)(-0.21*(r+1) + 0.29*(c+1) - 0.007*st);
            }
            fu->entries[st].out_state = (unsigned char)st;
        }
        fill(a, n, lo*131 + hi); memcpy(b, a, n * sizeof *a);
        memset(sa, 0, sizeof sa); memset(sb, 0, sizeof sb);
        // All 256 threads sweep the same array, interleaved by stride.
        for (u32 t = 0; t < 256; t++) pf_run  (sa, a, K, lo, hi, fu, 0, t);
        for (u32 t = 0; t < 256; t++) base_run(sb, b, K, lo, hi, fu, 0, t);
        cases++;
        if (memcmp(a, b, n * sizeof *a)) {
            if (bad < 3) printf("  MISMATCH lo=%u hi=%u\n", lo, hi);
            bad++;
        }
    }
    printf("K=%u stride=256 threads=256  cases=%d  mismatched=%d  -> %s\n",
           K, cases, bad, bad ? "*** NOT BYTE-EXACT ***" : "BYTE-EXACT");
    return bad != 0;
}
