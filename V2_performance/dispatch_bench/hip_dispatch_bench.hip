// hip_dispatch_bench.hip — measure HIP launch-to-completion latency.
//
// Same empty kernel, same 5 args, same grid (64 threads), same
// launch-to-completion definition as hsa_dispatch_bench.cc, so the two
// numbers subtract cleanly into "HIP runtime overhead over raw HSA".
//
// Modes:
//   sync         <<<>>> + hipDeviceSynchronize per launch  (naive user code)
//   stream_sync  <<<>>> + hipStreamSynchronize per launch  (typical)
//   launch_only  <<<>>> x N then ONE hipStreamSynchronize  (host-side
//                per-launch submission cost with the round trip amortized;
//                the analogue of the HSA batched mode)
//
// Built with hipcc from ROCm 7.2.3 on the compute node. This file is NOT part
// of the clifft build -- it exists purely to quantify the runtime we removed.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

__global__ void bench_empty(unsigned char* out, unsigned long a, unsigned long b,
                            unsigned long c, unsigned long d) {
    if (threadIdx.x == 0) out[0] = (unsigned char)(a ^ b ^ c ^ d);
}

#define HC(e)                                                                 \
    do {                                                                      \
        hipError_t e_ = (e);                                                  \
        if (e_ != hipSuccess) {                                               \
            std::fprintf(stderr, "FATAL %s: %s\n", #e, hipGetErrorString(e_)); \
            return 1;                                                         \
        }                                                                     \
    } while (0)

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 2000;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 5;

    hipDeviceProp_t prop{};
    HC(hipGetDeviceProperties(&prop, 0));

    unsigned char* out = nullptr;
    HC(hipMalloc(&out, 4096));

    hipStream_t stream;
    HC(hipStreamCreate(&stream));

    // warm up: first launch pays module load + code object finalize
    hipLaunchKernelGGL(bench_empty, dim3(1), dim3(64), 0, stream, out, 1ul, 2ul, 3ul, 4ul);
    HC(hipStreamSynchronize(stream));

    std::printf("# arch=%s hip_runtime_version=%d iters=%d reps=%d\n",
                prop.gcnArchName, HIP_VERSION, iters, reps);
    std::printf("mode,rep,ns_per_dispatch\n");

    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            hipLaunchKernelGGL(bench_empty, dim3(1), dim3(64), 0, 0, out, 1ul, 2ul, 3ul, 4ul);
            hipDeviceSynchronize();
        }
        auto t1 = std::chrono::steady_clock::now();
        std::printf("sync,%d,%.1f\n", r,
                    std::chrono::duration<double, std::nano>(t1 - t0).count() / iters);
    }

    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            hipLaunchKernelGGL(bench_empty, dim3(1), dim3(64), 0, stream, out, 1ul, 2ul, 3ul, 4ul);
            hipStreamSynchronize(stream);
        }
        auto t1 = std::chrono::steady_clock::now();
        std::printf("stream_sync,%d,%.1f\n", r,
                    std::chrono::duration<double, std::nano>(t1 - t0).count() / iters);
    }

    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            hipLaunchKernelGGL(bench_empty, dim3(1), dim3(64), 0, stream, out, 1ul, 2ul, 3ul, 4ul);
        }
        hipStreamSynchronize(stream);
        auto t1 = std::chrono::steady_clock::now();
        std::printf("launch_only,%d,%.1f\n", r,
                    std::chrono::duration<double, std::nano>(t1 - t0).count() / iters);
    }

    HC(hipStreamDestroy(stream));
    HC(hipFree(out));
    return 0;
}
