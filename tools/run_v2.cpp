// run_v2.cpp — CLI for the MLIR-V2 (HIP-free, HSA-only) GPU backend.
// P0(a): a --probe mode that proves the no-HIP build/load/dispatch path.
#include "clifft/gpu/mlir/v2/v2_kernel.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    bool probe = false;
    uint32_t n = 4096;
    std::string hsaco;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--probe") probe = true;
        else if (a == "--n" && i + 1 < argc) n = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (a == "--hsaco" && i + 1 < argc) hsaco = argv[++i];
        else if (a == "--help" || a == "-h") {
            std::cout << "usage: run_v2 --probe [--n N] [--hsaco PATH]\n"
                         "  --probe   dispatch the no-HIP probe kernel and verify out[i]==i*2\n";
            return 0;
        }
    }

    if (!probe) {
        std::cerr << "run_v2: no mode selected. Try --probe (see --help).\n";
        return 2;
    }

    auto r = clifft::gpu::v2::run_probe_detailed(n, hsaco);
    std::cout << "{\n"
              << "  \"mode\": \"probe\",\n"
              << "  \"n\": " << n << ",\n"
              << "  \"ok\": " << (r.ok ? "true" : "false") << ",\n"
              << "  \"kernel_seconds\": " << r.kernel_seconds << ",\n"
              << "  \"error\": \"" << r.error << "\"\n"
              << "}\n";
    return r.ok ? 0 : 1;
}
