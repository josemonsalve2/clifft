// run_v2.cpp — CLI for the MLIR-V2 (HIP-free, HSA-only) GPU backend.
//   --probe            : prove the no-HIP build/load/dispatch path.
//   --circuit F.stim   : run the V2 coop interpreter + CPU reference, diff them.
#include "clifft/gpu/mlir/v2/v2_kernel.h"

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/backend/backend.h"
#include "clifft/api/reference_syndrome.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/svm/svm.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

static clifft::CompiledModule compile_program(const std::string& path) {
    auto circuit = clifft::parse_file(path);
    auto hir = clifft::trace(circuit);
    auto hpm = clifft::default_hir_pass_manager();
    hpm.run(hir);
    auto ref = clifft::compute_reference_syndrome(hir);
    std::vector<uint8_t> mask;  // no postselection by default
    auto program = clifft::lower(hir, mask, ref.detectors, ref.observables);
    auto bpm = clifft::default_bytecode_pass_manager();
    bpm.run(program);
    return program;
}

static void print_vec(const char* name, const std::vector<uint64_t>& v) {
    std::cout << "  \"" << name << "\": [";
    for (size_t i = 0; i < v.size(); ++i) { if (i) std::cout << ", "; std::cout << v[i]; }
    std::cout << "]";
}

int main(int argc, char** argv) {
    bool probe = false;
    std::string circuit;
    uint32_t n = 4096, shots = 10000;
    uint64_t seed = 42;
    std::string hsaco;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--probe") probe = true;
        else if (a == "--circuit" && i + 1 < argc) circuit = argv[++i];
        else if (a == "--n" && i + 1 < argc) n = std::stoul(argv[++i]);
        else if (a == "--shots" && i + 1 < argc) shots = std::stoul(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
        else if (a == "--hsaco" && i + 1 < argc) hsaco = argv[++i];
        else if (a == "--help" || a == "-h") {
            std::cout << "usage: run_v2 --probe [--n N]\n"
                         "       run_v2 --circuit F.stim [--shots S] [--seed X]\n";
            return 0;
        }
    }

    if (probe) {
        auto r = clifft::gpu::v2::run_probe_detailed(n, hsaco);
        std::cout << "{ \"mode\": \"probe\", \"n\": " << n << ", \"ok\": "
                  << (r.ok ? "true" : "false") << ", \"kernel_seconds\": "
                  << r.kernel_seconds << ", \"error\": \"" << r.error << "\" }\n";
        return r.ok ? 0 : 1;
    }

    if (!circuit.empty()) {
        auto program = compile_program(circuit);
        // CPU reference (f64 gold).
        auto cpu = clifft::sample_survivors(program, shots, seed, false);
        // V2 GPU (HIP-free HSA).
        double ks = 0.0;
        auto v2 = clifft::gpu::v2::v2_sample(program, shots, seed, &ks, hsaco);

        bool match = (cpu.passed_shots == v2.passed_shots) &&
                     (cpu.observable_ones == v2.observable_ones);
        std::cout << "{\n"
                  << "  \"circuit\": \"" << circuit << "\",\n"
                  << "  \"shots\": " << shots << ", \"seed\": " << seed << ",\n"
                  << "  \"peak_rank\": " << program.peak_rank << ",\n"
                  << "  \"cpu_passed\": " << cpu.passed_shots << ",\n"
                  << "  \"v2_passed\": " << v2.passed_shots << ",\n";
        print_vec("cpu_observable_ones", cpu.observable_ones); std::cout << ",\n";
        print_vec("v2_observable_ones", v2.observable_ones); std::cout << ",\n";
        std::cout << "  \"v2_kernel_seconds\": " << ks << ",\n"
                  << "  \"match\": " << (match ? "true" : "false") << "\n}\n";
        return match ? 0 : 1;
    }

    std::cerr << "run_v2: no mode selected (see --help)\n";
    return 2;
}
