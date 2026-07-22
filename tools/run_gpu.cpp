#include "clifft/gpu/gpu_sampler.h"

#include "clifft/api/reference_syndrome.h"
#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/svm/svm.h"

#include <chrono>
#include <cstdint>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

double rate(uint64_t n, double d) {
    return d == 0.0 ? 0.0 : static_cast<double>(n) / d;
}

clifft::CompiledModule compile_program(const std::string& circuit_path, bool postselect_all_detectors,
                                       double& probe_seconds, double& compile_seconds) {
    auto circuit = clifft::parse_file(circuit_path);
    auto hir = clifft::trace(circuit);
    auto hpm = clifft::default_hir_pass_manager();
    hpm.run(hir);

    auto probe_start = std::chrono::steady_clock::now();
    auto ref = clifft::compute_reference_syndrome(hir);
    probe_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - probe_start).count();

    std::vector<uint8_t> postselection_mask;
    if (postselect_all_detectors) {
        postselection_mask.assign(ref.detectors.size(), 1);
    }
    auto compile_start = std::chrono::steady_clock::now();
    auto program = clifft::lower(hir, postselection_mask, ref.detectors, ref.observables);
    auto bpm = clifft::default_bytecode_pass_manager();
    bpm.run(program);
    compile_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - compile_start).count();
    return program;
}

}  // namespace

int main(int argc, char** argv) {
    std::string circuit_path = "./circuit_d5_p=0.001.stim";
    uint64_t shots = 10000000ULL;
    uint64_t seed = 42;
    uint32_t block_size = 256;
    bool cpu_reference = false;
    bool hybrid = false;
    bool use_mlir = false;
    std::string postselection_mode = "all";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + arg);
            }
            return argv[++i];
        };
        if (arg == "--circuit") {
            circuit_path = next();
        } else if (arg == "--shots") {
            shots = std::stoull(next());
        } else if (arg == "--seed") {
            seed = std::stoull(next());
        } else if (arg == "--block-size") {
            block_size = static_cast<uint32_t>(std::stoul(next()));
        } else if (arg == "--postselection") {
            postselection_mode = next();
            if (postselection_mode != "all" && postselection_mode != "none") {
                throw std::runtime_error("--postselection must be 'all' or 'none'");
            }
        } else if (arg == "--no-postselection") {
            postselection_mode = "none";
        } else if (arg == "--cpu-reference") {
            cpu_reference = true;
        } else if (arg == "--hybrid") {
            hybrid = true;
        } else if (arg == "--mlir") {
            use_mlir = true;
        } else if (arg == "--diagnose") {
            std::cout << clifft::gpu::gpu_backend_info() << "\n";
            return 0;
        }
    }

    auto total_start = std::chrono::steady_clock::now();
    double probe_seconds = 0.0;
    double compile_seconds = 0.0;
    auto program = compile_program(circuit_path, postselection_mode == "all", probe_seconds,
                                   compile_seconds);

    auto sample_start = std::chrono::steady_clock::now();
    uint64_t passed = 0;
    uint64_t logical = 0;
    std::vector<uint64_t> obs;
    std::string backend;

    if (cpu_reference) {
        if (shots > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("CPU reference path only supports up to uint32_t shots");
        }
        auto result =
            clifft::sample_survivors(program, static_cast<uint32_t>(shots), seed, false);
        passed = result.passed_shots;
        logical = result.logical_errors;
        obs = result.observable_ones;
        backend = "cpu-reference-clifft-core";
    } else {
        clifft::gpu::GpuSamplerOptions options;
        options.seed = seed;
        options.block_size = block_size;
        options.hybrid = hybrid;
        options.mlir = use_mlir;
        auto result = clifft::gpu::gpu_sample_survivors(program, shots, options);
        passed = result.passed_shots;
        logical = result.logical_errors;
        obs = result.observable_ones;
        if (use_mlir) backend = "hip-gpu-mlir";
        else if (hybrid) backend = "hip-gpu-hybrid";
        else backend = "hip-gpu";
    }

    double sample_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - sample_start).count();
    double total_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
    uint64_t discards = shots - passed;

    std::cout << "{\n";
    std::cout << "  \"backend\": \"" << backend << "\",\n";
    std::cout << "  \"circuit_path\": \"" << circuit_path << "\",\n";
    std::cout << "  \"shots\": " << shots << ",\n";
    std::cout << "  \"seed\": " << seed << ",\n";
    std::cout << "  \"postselection\": \"" << postselection_mode << "\",\n";
    std::cout << "  \"has_postselection\": " << (program.has_postselection ? "true" : "false")
              << ",\n";
    std::cout << "  \"peak_rank\": " << program.peak_rank << ",\n";
    std::cout << "  \"detectors\": " << program.num_detectors << ",\n";
    std::cout << "  \"observables\": " << program.num_observables << ",\n";
    std::cout << "  \"measurements\": " << program.num_measurements << ",\n";
    std::cout << "  \"total_meas_slots\": " << program.total_meas_slots << ",\n";
    std::cout << "  \"noise_sites\": " << program.constant_pool.noise_sites.size() << ",\n";
    std::cout << "  \"num_instructions\": " << program.bytecode.size() << ",\n";
    std::cout << "  \"passed_shots\": " << passed << ",\n";
    std::cout << "  \"discarded_shots\": " << discards << ",\n";
    std::cout << "  \"logical_errors\": " << logical << ",\n";
    std::cout << "  \"observable_ones\": [";
    for (size_t i = 0; i < obs.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << obs[i];
    }
    std::cout << "],\n";
    std::cout << "  \"discard_rate\": " << (static_cast<double>(discards) / shots) << ",\n";
    std::cout << "  \"survival_rate\": " << (static_cast<double>(passed) / shots) << ",\n";
    std::cout << "  \"error_rate_per_survivor\": "
              << (passed == 0 ? 0.0 : static_cast<double>(logical) / passed) << ",\n";
    std::cout << "  \"sample_seconds\": " << sample_seconds << ",\n";
    std::cout << "  \"total_seconds\": " << total_seconds << ",\n";
    std::cout << "  \"shots_per_second_sampling_only\": " << rate(shots, sample_seconds) << ",\n";
    std::cout << "  \"shots_per_second_total\": " << rate(shots, total_seconds) << "\n";
    std::cout << "}\n";
    return 0;
}
