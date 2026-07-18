#ifdef CLIFFT_ENABLE_GPU

#include "clifft/gpu/gpu_sampler.h"
#include "clifft/gpu/device_program.h"

#include "clifft/api/reference_syndrome.h"
#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/svm/svm.h"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <string>

namespace {

clifft::CompiledModule compile_stim(const std::string& stim_text, bool postselect = true) {
    auto circuit = clifft::parse(stim_text);
    auto hir = clifft::trace(circuit);
    auto hpm = clifft::default_hir_pass_manager();
    hpm.run(hir);
    auto ref = clifft::compute_reference_syndrome(hir);
    std::vector<uint8_t> ps_mask;
    if (postselect) ps_mask.assign(ref.detectors.size(), 1);
    auto program = clifft::lower(hir, ps_mask, ref.detectors, ref.observables);
    auto bpm = clifft::default_bytecode_pass_manager();
    bpm.run(program);
    return program;
}

clifft::CompiledModule compile_file(const std::string& path, bool postselect = true) {
    auto circuit = clifft::parse_file(path);
    auto hir = clifft::trace(circuit);
    auto hpm = clifft::default_hir_pass_manager();
    hpm.run(hir);
    auto ref = clifft::compute_reference_syndrome(hir);
    std::vector<uint8_t> ps_mask;
    if (postselect) ps_mask.assign(ref.detectors.size(), 1);
    auto program = clifft::lower(hir, ps_mask, ref.detectors, ref.observables);
    auto bpm = clifft::default_bytecode_pass_manager();
    bpm.run(program);
    return program;
}

}  // namespace

TEST_CASE("GPU: backend info returns device string", "[gpu]") {
    auto info = clifft::gpu::gpu_backend_info();
    REQUIRE(info.find("hip devices:") != std::string::npos);
}

TEST_CASE("GPU: pure Clifford circuit (frame ops only)", "[gpu]") {
    auto program = compile_stim(R"(
        H 0
        CX 0 1
        M 0 1
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    )", false);

    auto gpu = clifft::gpu::gpu_sample_survivors(program, 10000, {.seed = 42});
    auto cpu = clifft::sample_survivors(program, 10000, 42, false);

    CHECK(gpu.passed_shots == cpu.passed_shots);
    CHECK(gpu.logical_errors == cpu.logical_errors);
}

TEST_CASE("GPU: T gate circuit (array ops + expand)", "[gpu]") {
    auto program = compile_stim(R"(
        H 0
        T 0
        H 0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
    )", false);

    REQUIRE(program.peak_rank > 0);
    auto gpu = clifft::gpu::gpu_sample_survivors(program, 50000, {.seed = 42});
    auto cpu = clifft::sample_survivors(program, 50000, 42, false);

    double gpu_rate = static_cast<double>(gpu.logical_errors) / gpu.passed_shots;
    double cpu_rate = static_cast<double>(cpu.logical_errors) / cpu.passed_shots;
    double sigma = std::sqrt(cpu_rate * (1 - cpu_rate) / cpu.passed_shots);
    CHECK(std::abs(gpu_rate - cpu_rate) < 5.0 * std::max(sigma, 1e-6));
}

TEST_CASE("GPU: noisy circuit with postselection", "[gpu]") {
    auto program = compile_stim(R"(
        H 0
        DEPOLARIZE1(0.01) 0
        CX 0 1
        DEPOLARIZE2(0.01) 0 1
        M 0 1
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    )");

    auto gpu = clifft::gpu::gpu_sample_survivors(program, 100000, {.seed = 42});
    auto cpu = clifft::sample_survivors(program, 100000, 42, false);

    double gpu_surv = static_cast<double>(gpu.passed_shots) / gpu.total_shots;
    double cpu_surv = static_cast<double>(cpu.passed_shots) / cpu.total_shots;
    double sigma = std::sqrt(cpu_surv * (1 - cpu_surv) / cpu.total_shots);
    CHECK(std::abs(gpu_surv - cpu_surv) < 5.0 * sigma);
}

TEST_CASE("GPU: multi-qubit T circuit (multiple expands)", "[gpu]") {
    auto program = compile_stim(R"(
        H 0
        T 0
        H 1
        T 1
        CX 0 1
        H 0
        H 1
        M 0 1
        OBSERVABLE_INCLUDE(0) rec[-1]
    )", false);

    REQUIRE(program.peak_rank >= 2);
    auto gpu = clifft::gpu::gpu_sample_survivors(program, 50000, {.seed = 42});
    CHECK(gpu.passed_shots == gpu.total_shots);
}

TEST_CASE("GPU: cultivation d5 fixture circuit", "[gpu][e2e]") {
    auto program = compile_file("tests/fixtures/cultivation_d5.stim");

    auto gpu = clifft::gpu::gpu_sample_survivors(program, 100000, {.seed = 42});
    auto cpu = clifft::sample_survivors(program, 100000, 42, false);

    double gpu_surv = static_cast<double>(gpu.passed_shots) / gpu.total_shots;
    double cpu_surv = static_cast<double>(cpu.passed_shots) / cpu.total_shots;
    double sigma = std::sqrt(cpu_surv * (1 - cpu_surv) / cpu.total_shots);
    CHECK(std::abs(gpu_surv - cpu_surv) < 5.0 * sigma);
}

TEST_CASE("GPU: validate_program rejects forced opcodes", "[gpu]") {
    clifft::CompiledModule prog;
    prog.peak_rank = 2;
    prog.total_meas_slots = 10;
    prog.num_observables = 1;
    prog.num_qubits = 2;
    prog.bytecode.push_back(clifft::make_meas(
        clifft::Opcode::OP_MEAS_DORMANT_STATIC_FORCED, 0, 0, false));

    REQUIRE_THROWS_AS(clifft::gpu::validate_program(prog), std::runtime_error);
}

TEST_CASE("GPU: validate_program rejects peak_rank > 19", "[gpu]") {
    clifft::CompiledModule prog;
    prog.peak_rank = 20;
    prog.total_meas_slots = 10;
    prog.num_observables = 1;
    prog.num_qubits = 2;

    REQUIRE_THROWS_AS(clifft::gpu::validate_program(prog), std::runtime_error);
}

TEST_CASE("GPU: determinism - same seed same results", "[gpu]") {
    auto program = compile_stim(R"(
        H 0
        DEPOLARIZE1(0.01) 0
        CX 0 1
        DEPOLARIZE2(0.01) 0 1
        M 0 1
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    )");

    auto r1 = clifft::gpu::gpu_sample_survivors(program, 100000, {.seed = 123});
    auto r2 = clifft::gpu::gpu_sample_survivors(program, 100000, {.seed = 123});

    CHECK(r1.passed_shots == r2.passed_shots);
    CHECK(r1.logical_errors == r2.logical_errors);
    for (size_t i = 0; i < r1.observable_ones.size(); ++i) {
        CHECK(r1.observable_ones[i] == r2.observable_ones[i]);
    }
}

TEST_CASE("GPU: target_qec fixture circuit", "[gpu][e2e]") {
    auto program = compile_file("tests/fixtures/target_qec.stim");

    auto gpu = clifft::gpu::gpu_sample_survivors(program, 100000, {.seed = 42});
    auto cpu = clifft::sample_survivors(program, 100000, 42, false);

    double gpu_surv = static_cast<double>(gpu.passed_shots) / gpu.total_shots;
    double cpu_surv = static_cast<double>(cpu.passed_shots) / cpu.total_shots;
    double sigma = std::sqrt(cpu_surv * (1 - cpu_surv) / cpu.total_shots);
    CHECK(std::abs(gpu_surv - cpu_surv) < 5.0 * sigma);
}

#endif  // CLIFFT_ENABLE_GPU
