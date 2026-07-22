// mlir_codegen.cc — MLIR→LLVM-IR megakernel codegen via textual MLIR generation
//
// Generates textual MLIR (LLVM dialect) for the circuit's bytecode, then runs:
//   mlir-opt --canonicalize --cse --convert-func-to-llvm → LLVM-IR (mlir textual)
//   mlir-translate --mlir-to-llvmir → LLVM-IR text (.ll)
//
// The .ll file is then compiled by mlir_kernel_cache.cc using llc + lld.
//
// This approach avoids the complex and rapidly-changing MLIR C++ API by using
// stable textual MLIR format + subprocess tools.  The same mlir-opt binary
// available from module load llvm/upstream_05082025 provides full AMDGPU support.
//
// Architecture:
//   FlattenedProgram
//     → emit_mlir_text()        (this file): generates textual MLIR LLVM dialect
//         One func @compiled_mlir_kernel with straight-line LLVM dialect ops
//         Gate matrices and constant pools baked as llvm.mlir.constant
//         Butterfly loops emitted as llvm.br / llvm.cond_br blocks
//     → mlir-opt subprocess:    canonicalize, CSE, loop-unroll (rank ≤ 4)
//     → mlir-translate --mlir-to-llvmir → .ll text
//
// Supported register-tier operations (peak_rank ≤ 4):
//   Frame: FRAME_H, FRAME_S, FRAME_S_DAG, FRAME_CNOT, FRAME_CZ, FRAME_SWAP
//   Array: ARRAY_H, ARRAY_S, ARRAY_S_DAG, ARRAY_T, ARRAY_T_DAG, ARRAY_CNOT, ARRAY_CZ
//   Meas: MEAS_DORMANT_STATIC
//   Classical: OBSERVABLE
//
// Unsupported ops cause a "discard" flag to be set in the generated IR.

#ifdef CLIFFT_ENABLE_MLIR

#include "clifft/gpu/mlir/mlir_codegen.h"
#include "clifft/gpu/codegen/kernel_codegen.h"  // UsedFunctions, analyze_used_functions
#include "clifft/gpu/gpu_types.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/wait.h>

namespace clifft {
namespace gpu {

namespace {

// -----------------------------------------------------------------------
// subprocess helper
// -----------------------------------------------------------------------
static int run_pipe_command(const std::string& cmd, const std::string& stdin_data,
                             std::string& stdout_out) {
    // Use popen for stdin pipe: write stdin_data to process stdin, read stdout
    // Simple approach: write to a temp file, pipe through command
    // (popen only gives us stdout, not bidirectional pipe)
    // We use shell redirection: echo stdin_data | cmd
    std::string full = "printf '%s' \"$MLIR_INPUT\" | " + cmd + " 2>&1";
    // Set env var for the input
    setenv("MLIR_INPUT", stdin_data.c_str(), 1);
    FILE* pipe = popen(full.c_str(), "r");
    if (!pipe) { stdout_out = "popen failed"; return -1; }
    std::array<char, 65536> buf;
    stdout_out.clear();
    while (fgets(buf.data(), (int)buf.size(), pipe)) stdout_out += buf.data();
    int st = pclose(pipe);
    unsetenv("MLIR_INPUT");
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

static std::string find_mlir_opt() {
    const char* llvm = std::getenv("LLVM_PREFIX");
    if (llvm) {
        std::string c = std::string(llvm) + "/bin/mlir-opt";
        if (std::filesystem::exists(c)) return c;
    }
    if (std::filesystem::exists("/shared/jmonsalv/software/modules/llvm/upstream_05082025/bin/mlir-opt"))
        return "/shared/jmonsalv/software/modules/llvm/upstream_05082025/bin/mlir-opt";
    return "mlir-opt";
}

static std::string find_mlir_translate() {
    const char* llvm = std::getenv("LLVM_PREFIX");
    if (llvm) {
        std::string c = std::string(llvm) + "/bin/mlir-translate";
        if (std::filesystem::exists(c)) return c;
    }
    if (std::filesystem::exists("/shared/jmonsalv/software/modules/llvm/upstream_05082025/bin/mlir-translate"))
        return "/shared/jmonsalv/software/modules/llvm/upstream_05082025/bin/mlir-translate";
    return "mlir-translate";
}

// -----------------------------------------------------------------------
// MLIR textual IR generation
//
// We emit MLIR LLVM dialect text directly.  The generated function:
//   func.func @compiled_mlir_kernel(
//     %shot_offset: i64, %shots: i64, %seed: i64,
//     %block_counts: !llvm.ptr, %num_obs: i32, %num_exp: i32)
//   has a single basic block with straight-line ops.
//
// The amplitude array v[] lives in local (stack) memory via llvm.alloca.
// For rank ≤ 4, the butterfly loops have ≤8 iterations and LLVM can fully
// unroll them.
// -----------------------------------------------------------------------

static void emit_type_aliases(std::ostringstream& out) {
    // GpuComplex as LLVM struct: {f32, f32}
    // We use !llvm.struct<(f32, f32)> inline
}

// Unique label counter for basic blocks
static int label_counter = 0;
static std::string fresh_label(const std::string& prefix = "bb") {
    return prefix + std::to_string(label_counter++);
}

// -----------------------------------------------------------------------
// emit_mlir_text: main textual MLIR emission function
// -----------------------------------------------------------------------
static std::string emit_mlir_text(const FlattenedProgram& flat) {
    label_counter = 0;
    UsedFunctions uf = analyze_used_functions(flat);

    std::ostringstream out;

    // Module and function header
    out << "module attributes {llvm.target_triple = \"amdgcn-amd-amdhsa\"} {\n\n";

    // Constant pool globals (inline as llvm.mlir.constant in function)
    // We embed constant values directly into the instruction stream

    // Kernel function
    out << "llvm.func @compiled_mlir_kernel(\n"
        << "    %shot_offset: i64, %shots: i64, %seed: i64,\n"
        << "    %block_counts: !llvm.ptr,\n"
        << "    %num_obs: i32, %num_exp: i32) -> () {\n";

    out << "  // Per-thread state\n";
    // px[2], pz[2]: i64 arrays
    out << "  %px_ptr = llvm.alloca %c2_i32 x i64 : (i32) -> !llvm.ptr\n";
    out << "  %pz_ptr = llvm.alloca %c2_i32 x i64 : (i32) -> !llvm.ptr\n";

    // But we need %c2_i32 to be defined first — constants need to precede uses
    // In MLIR LLVM dialect, constants are defined with llvm.mlir.constant
    // Rewrite to define constants at top of block

    // We restart and emit properly with pre-defined constants:
    // Clear and start over
    out.str("");
    out.clear();

    out << "module attributes {llvm.target_triple = \"amdgcn-amd-amdhsa\"} {\n\n";

    // Helper constants embedded inline
    const double kInvSqrt2 = 0.70710678118654752440084436210484903928;
    uint32_t kMaxAmplitudes = 1u << flat.peak_rank;

    out << "llvm.func @compiled_mlir_kernel(\n"
        << "    %shot_offset: i64, %shots: i64, %seed: i64,\n"
        << "    %block_counts: !llvm.ptr,\n"
        << "    %num_obs: i32, %num_exp: i32) -> () {\n"
        << "  ^entry:\n";

    // Define commonly used integer constants
    out << "  %c0_i32 = llvm.mlir.constant(0 : i32) : i32\n";
    out << "  %c1_i32 = llvm.mlir.constant(1 : i32) : i32\n";
    out << "  %c2_i32 = llvm.mlir.constant(2 : i32) : i32\n";
    out << "  %c0_i64 = llvm.mlir.constant(0 : i64) : i64\n";
    out << "  %c1_i64 = llvm.mlir.constant(1 : i64) : i64\n";
    out << "  %c6_i64 = llvm.mlir.constant(6 : i64) : i64\n";
    out << "  %c63_i64 = llvm.mlir.constant(63 : i64) : i64\n";
    out << "  %cminus1_i64 = llvm.mlir.constant(-1 : i64) : i64\n";
    out << "  %c0_i8 = llvm.mlir.constant(0 : i8) : i8\n";
    out << "  %c1_i8 = llvm.mlir.constant(1 : i8) : i8\n";

    // Alloca per-thread state
    out << "  // Per-thread Pauli frame: px[2], pz[2] as i64 arrays\n";
    out << "  %px_ptr = llvm.alloca %c2_i32 x i64 : (i32) -> !llvm.ptr\n";
    out << "  %pz_ptr = llvm.alloca %c2_i32 x i64 : (i32) -> !llvm.ptr\n";
    out << "  %active_k_ptr = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr\n";
    out << "  %discarded_ptr = llvm.alloca %c1_i32 x i8 : (i32) -> !llvm.ptr\n";

    // Amplitude array: v[kMaxAmplitudes] of {f32, f32}
    char buf[64];
    snprintf(buf, sizeof(buf), "%u", kMaxAmplitudes);
    out << "  %v_size = llvm.mlir.constant(" << buf << " : i32) : i32\n";
    out << "  %v_ptr = llvm.alloca %v_size x !llvm.struct<(f32, f32)> : (i32) -> !llvm.ptr\n";

    // meas[kMaxMeas], obs[kMaxObs]
    snprintf(buf, sizeof(buf), "%u", kMaxMeas);
    out << "  %meas_size = llvm.mlir.constant(" << buf << " : i32) : i32\n";
    out << "  %meas_ptr = llvm.alloca %meas_size x i8 : (i32) -> !llvm.ptr\n";
    snprintf(buf, sizeof(buf), "%u", kMaxObs);
    out << "  %obs_size = llvm.mlir.constant(" << buf << " : i32) : i32\n";
    out << "  %obs_ptr = llvm.alloca %obs_size x i8 : (i32) -> !llvm.ptr\n";

    // Initialize state
    out << "  // Initialize Pauli frame to 0\n";
    out << "  %px0_ptr = llvm.getelementptr inbounds %px_ptr[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
    out << "  %px1_ptr = llvm.getelementptr inbounds %px_ptr[%c1_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
    out << "  %pz0_ptr = llvm.getelementptr inbounds %pz_ptr[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
    out << "  %pz1_ptr = llvm.getelementptr inbounds %pz_ptr[%c1_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
    out << "  llvm.store %c0_i64, %px0_ptr : i64, !llvm.ptr\n";
    out << "  llvm.store %c0_i64, %px1_ptr : i64, !llvm.ptr\n";
    out << "  llvm.store %c0_i64, %pz0_ptr : i64, !llvm.ptr\n";
    out << "  llvm.store %c0_i64, %pz1_ptr : i64, !llvm.ptr\n";
    out << "  llvm.store %c0_i32, %active_k_ptr : i32, !llvm.ptr\n";
    out << "  llvm.store %c0_i8, %discarded_ptr : i8, !llvm.ptr\n";

    // v[0] = {1.0f, 0.0f}
    out << "  // v[0] = {1.0, 0.0}\n";
    out << "  %v0_ptr = llvm.getelementptr inbounds %v_ptr[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(f32, f32)>\n";
    out << "  %f_one = llvm.mlir.constant(1.0 : f32) : f32\n";
    out << "  %f_zero = llvm.mlir.constant(0.0 : f32) : f32\n";
    out << "  %init_c0 = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
    out << "  %init_c1 = llvm.insertvalue %f_one, %init_c0[0] : !llvm.struct<(f32, f32)>\n";
    out << "  %init_c2 = llvm.insertvalue %f_zero, %init_c1[1] : !llvm.struct<(f32, f32)>\n";
    out << "  llvm.store %init_c2, %v0_ptr : !llvm.struct<(f32, f32)>, !llvm.ptr\n";

    // Inline helper macros (as comments + ssa emission for each op)
    // We use a running SSA index counter
    int ssa = 100;
    auto fresh_ssa = [&]() -> std::string {
        return "%t" + std::to_string(ssa++);
    };

    // bit_get helper: returns an i1 value
    // Words is a ptr, idx is i32 (axis number)
    auto emit_bit_get = [&](const std::string& words_ptr,
                             const std::string& idx_i32) -> std::string {
        std::string idx_i64 = fresh_ssa();
        out << "  " << idx_i64 << " = llvm.zext " << idx_i32 << " : i32 to i64\n";
        std::string widx = fresh_ssa();
        out << "  " << widx << " = llvm.lshr " << idx_i64 << ", %c6_i64 : i64\n";
        std::string wptr = fresh_ssa();
        out << "  " << wptr << " = llvm.getelementptr inbounds " << words_ptr
            << "[" << widx << "] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
        std::string word = fresh_ssa();
        out << "  " << word << " = llvm.load " << wptr << " : !llvm.ptr -> i64\n";
        std::string shift = fresh_ssa();
        out << "  " << shift << " = llvm.and " << idx_i64 << ", %c63_i64 : i64\n";
        std::string shifted = fresh_ssa();
        out << "  " << shifted << " = llvm.lshr " << word << ", " << shift << " : i64\n";
        std::string bit64 = fresh_ssa();
        out << "  " << bit64 << " = llvm.and " << shifted << ", %c1_i64 : i64\n";
        std::string bit1 = fresh_ssa();
        out << "  " << bit1 << " = llvm.trunc " << bit64 << " : i64 to i1\n";
        return bit1;
    };

    auto emit_bit_xor = [&](const std::string& words_ptr,
                              const std::string& idx_i32,
                              const std::string& val_i1) {
        std::string idx_i64 = fresh_ssa();
        out << "  " << idx_i64 << " = llvm.zext " << idx_i32 << " : i32 to i64\n";
        std::string widx = fresh_ssa();
        out << "  " << widx << " = llvm.lshr " << idx_i64 << ", %c6_i64 : i64\n";
        std::string wptr = fresh_ssa();
        out << "  " << wptr << " = llvm.getelementptr inbounds " << words_ptr
            << "[" << widx << "] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
        std::string word = fresh_ssa();
        out << "  " << word << " = llvm.load " << wptr << " : !llvm.ptr -> i64\n";
        std::string shift = fresh_ssa();
        out << "  " << shift << " = llvm.and " << idx_i64 << ", %c63_i64 : i64\n";
        std::string val_i64 = fresh_ssa();
        out << "  " << val_i64 << " = llvm.zext " << val_i1 << " : i1 to i64\n";
        std::string xmask = fresh_ssa();
        out << "  " << xmask << " = llvm.shl " << val_i64 << ", " << shift << " : i64\n";
        std::string new_word = fresh_ssa();
        out << "  " << new_word << " = llvm.xor " << word << ", " << xmask << " : i64\n";
        out << "  llvm.store " << new_word << ", " << wptr << " : i64, !llvm.ptr\n";
    };

    auto emit_bit_set = [&](const std::string& words_ptr,
                              const std::string& idx_i32,
                              const std::string& val_i1) {
        std::string idx_i64 = fresh_ssa();
        out << "  " << idx_i64 << " = llvm.zext " << idx_i32 << " : i32 to i64\n";
        std::string widx = fresh_ssa();
        out << "  " << widx << " = llvm.lshr " << idx_i64 << ", %c6_i64 : i64\n";
        std::string wptr = fresh_ssa();
        out << "  " << wptr << " = llvm.getelementptr inbounds " << words_ptr
            << "[" << widx << "] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
        std::string word = fresh_ssa();
        out << "  " << word << " = llvm.load " << wptr << " : !llvm.ptr -> i64\n";
        std::string shift = fresh_ssa();
        out << "  " << shift << " = llvm.and " << idx_i64 << ", %c63_i64 : i64\n";
        std::string mask = fresh_ssa();
        out << "  " << mask << " = llvm.shl %c1_i64, " << shift << " : i64\n";
        // cleared = word & ~mask = word & (mask ^ -1)
        std::string nmask = fresh_ssa();
        out << "  " << nmask << " = llvm.xor " << mask << ", %cminus1_i64 : i64\n";
        std::string cleared = fresh_ssa();
        out << "  " << cleared << " = llvm.and " << word << ", " << nmask << " : i64\n";
        // bit_to_set = (val as i64) << shift
        std::string val64 = fresh_ssa();
        out << "  " << val64 << " = llvm.zext " << val_i1 << " : i1 to i64\n";
        std::string bset = fresh_ssa();
        out << "  " << bset << " = llvm.shl " << val64 << ", " << shift << " : i64\n";
        std::string new_word = fresh_ssa();
        out << "  " << new_word << " = llvm.or " << cleared << ", " << bset << " : i64\n";
        out << "  llvm.store " << new_word << ", " << wptr << " : i64, !llvm.ptr\n";
    };

    // scatter_bits_1(val, pos): insert zero bit at pos
    auto emit_scatter_bits_1 = [&](const std::string& val_i64,
                                    const std::string& pos_i64) -> std::string {
        // mask = (1 << pos) - 1
        std::string m1 = fresh_ssa();
        out << "  " << m1 << " = llvm.shl %c1_i64, " << pos_i64 << " : i64\n";
        std::string mask = fresh_ssa();
        out << "  " << mask << " = llvm.sub " << m1 << ", %c1_i64 : i64\n";
        std::string nmask = fresh_ssa();
        out << "  " << nmask << " = llvm.xor " << mask << ", %cminus1_i64 : i64\n";
        std::string lo = fresh_ssa();
        out << "  " << lo << " = llvm.and " << val_i64 << ", " << mask << " : i64\n";
        std::string hi_pre = fresh_ssa();
        out << "  " << hi_pre << " = llvm.and " << val_i64 << ", " << nmask << " : i64\n";
        std::string hi = fresh_ssa();
        out << "  " << hi << " = llvm.shl " << hi_pre << ", %c1_i64 : i64\n";
        std::string result = fresh_ssa();
        out << "  " << result << " = llvm.or " << lo << ", " << hi << " : i64\n";
        return result;
    };

    // load/store complex from v[]
    auto emit_load_v = [&](const std::string& idx_i64) -> std::string {
        std::string vp = fresh_ssa();
        out << "  " << vp << " = llvm.getelementptr inbounds %v_ptr[" << idx_i64
            << "] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(f32, f32)>\n";
        std::string vc = fresh_ssa();
        out << "  " << vc << " = llvm.load " << vp << " : !llvm.ptr -> !llvm.struct<(f32, f32)>\n";
        return vc;
    };

    auto emit_store_v = [&](const std::string& idx_i64, const std::string& val) {
        std::string vp = fresh_ssa();
        out << "  " << vp << " = llvm.getelementptr inbounds %v_ptr[" << idx_i64
            << "] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(f32, f32)>\n";
        out << "  llvm.store " << val << ", " << vp
            << " : !llvm.struct<(f32, f32)>, !llvm.ptr\n";
    };

    // cadd, csub, cscale, cmul
    auto emit_cadd = [&](const std::string& a, const std::string& b_val) -> std::string {
        std::string are = fresh_ssa(), aim = fresh_ssa(), bre = fresh_ssa(), bim = fresh_ssa();
        out << "  " << are << " = llvm.extractvalue " << a << "[0] : !llvm.struct<(f32, f32)>\n";
        out << "  " << aim << " = llvm.extractvalue " << a << "[1] : !llvm.struct<(f32, f32)>\n";
        out << "  " << bre << " = llvm.extractvalue " << b_val << "[0] : !llvm.struct<(f32, f32)>\n";
        out << "  " << bim << " = llvm.extractvalue " << b_val << "[1] : !llvm.struct<(f32, f32)>\n";
        std::string re = fresh_ssa(), im = fresh_ssa();
        out << "  " << re << " = llvm.fadd " << are << ", " << bre << " : f32\n";
        out << "  " << im << " = llvm.fadd " << aim << ", " << bim << " : f32\n";
        std::string r0 = fresh_ssa(), r1 = fresh_ssa();
        out << "  " << r0 << " = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
        out << "  " << r1 << " = llvm.insertvalue " << re << ", " << r0 << "[0] : !llvm.struct<(f32, f32)>\n";
        std::string res = fresh_ssa();
        out << "  " << res << " = llvm.insertvalue " << im << ", " << r1 << "[1] : !llvm.struct<(f32, f32)>\n";
        return res;
    };

    auto emit_csub = [&](const std::string& a, const std::string& b_val) -> std::string {
        std::string are = fresh_ssa(), aim = fresh_ssa(), bre = fresh_ssa(), bim = fresh_ssa();
        out << "  " << are << " = llvm.extractvalue " << a << "[0] : !llvm.struct<(f32, f32)>\n";
        out << "  " << aim << " = llvm.extractvalue " << a << "[1] : !llvm.struct<(f32, f32)>\n";
        out << "  " << bre << " = llvm.extractvalue " << b_val << "[0] : !llvm.struct<(f32, f32)>\n";
        out << "  " << bim << " = llvm.extractvalue " << b_val << "[1] : !llvm.struct<(f32, f32)>\n";
        std::string re = fresh_ssa(), im = fresh_ssa();
        out << "  " << re << " = llvm.fsub " << are << ", " << bre << " : f32\n";
        out << "  " << im << " = llvm.fsub " << aim << ", " << bim << " : f32\n";
        std::string r0 = fresh_ssa(), r1 = fresh_ssa(), res = fresh_ssa();
        out << "  " << r0 << " = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
        out << "  " << r1 << " = llvm.insertvalue " << re << ", " << r0 << "[0] : !llvm.struct<(f32, f32)>\n";
        out << "  " << res << " = llvm.insertvalue " << im << ", " << r1 << "[1] : !llvm.struct<(f32, f32)>\n";
        return res;
    };

    auto emit_cscale_f64 = [&](const std::string& a, double scale) -> std::string {
        char sbuf[64];
        snprintf(sbuf, sizeof(sbuf), "%.17e", scale);
        std::string sf32 = fresh_ssa();
        out << "  " << sf32 << " = llvm.fptrunc llvm.mlir.constant(" << sbuf << " : f64) : f64 to f32\n";
        std::string are = fresh_ssa(), aim = fresh_ssa();
        out << "  " << are << " = llvm.extractvalue " << a << "[0] : !llvm.struct<(f32, f32)>\n";
        out << "  " << aim << " = llvm.extractvalue " << a << "[1] : !llvm.struct<(f32, f32)>\n";
        std::string re = fresh_ssa(), im = fresh_ssa();
        out << "  " << re << " = llvm.fmul " << are << ", " << sf32 << " : f32\n";
        out << "  " << im << " = llvm.fmul " << aim << ", " << sf32 << " : f32\n";
        std::string r0 = fresh_ssa(), r1 = fresh_ssa(), res = fresh_ssa();
        out << "  " << r0 << " = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
        out << "  " << r1 << " = llvm.insertvalue " << re << ", " << r0 << "[0] : !llvm.struct<(f32, f32)>\n";
        out << "  " << res << " = llvm.insertvalue " << im << ", " << r1 << "[1] : !llvm.struct<(f32, f32)>\n";
        return res;
    };

    auto emit_cmul_const = [&](const std::string& a,
                                double phase_re, double phase_im) -> std::string {
        // (a.re * phase_re - a.im * phase_im, a.re * phase_im + a.im * phase_re)
        char rbuf[64], ibuf[64];
        snprintf(rbuf, sizeof(rbuf), "%.17e", (float)phase_re);
        snprintf(ibuf, sizeof(ibuf), "%.17e", (float)phase_im);
        std::string pre_f = fresh_ssa(), pim_f = fresh_ssa();
        out << "  " << pre_f << " = llvm.mlir.constant(" << rbuf << " : f32) : f32\n";
        out << "  " << pim_f << " = llvm.mlir.constant(" << ibuf << " : f32) : f32\n";
        std::string are = fresh_ssa(), aim = fresh_ssa();
        out << "  " << are << " = llvm.extractvalue " << a << "[0] : !llvm.struct<(f32, f32)>\n";
        out << "  " << aim << " = llvm.extractvalue " << a << "[1] : !llvm.struct<(f32, f32)>\n";
        // re = are*pre - aim*pim
        std::string t0 = fresh_ssa(), t1 = fresh_ssa(), re = fresh_ssa();
        out << "  " << t0 << " = llvm.fmul " << are << ", " << pre_f << " : f32\n";
        out << "  " << t1 << " = llvm.fmul " << aim << ", " << pim_f << " : f32\n";
        out << "  " << re << " = llvm.fsub " << t0 << ", " << t1 << " : f32\n";
        // im = are*pim + aim*pre
        std::string t2 = fresh_ssa(), t3 = fresh_ssa(), im = fresh_ssa();
        out << "  " << t2 << " = llvm.fmul " << are << ", " << pim_f << " : f32\n";
        out << "  " << t3 << " = llvm.fmul " << aim << ", " << pre_f << " : f32\n";
        out << "  " << im << " = llvm.fadd " << t2 << ", " << t3 << " : f32\n";
        std::string r0 = fresh_ssa(), r1 = fresh_ssa(), res = fresh_ssa();
        out << "  " << r0 << " = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
        out << "  " << r1 << " = llvm.insertvalue " << re << ", " << r0 << "[0] : !llvm.struct<(f32, f32)>\n";
        out << "  " << res << " = llvm.insertvalue " << im << ", " << r1 << "[1] : !llvm.struct<(f32, f32)>\n";
        return res;
    };

    // Emit array_h for one specific axis value (static axis known at compile time)
    auto emit_array_h_static = [&](uint32_t axis) {
        out << "  // array_h on axis " << axis << "\n";
        // For register tier (ak ≤ 4), fully unroll: iters = 1 << (ak-1) ≤ 8
        // We know active_k at codegen time by tracking it statically.
        // The compiler can't know it exactly without running, but LLVM will unroll
        // the loop when ak is bounded and small.
        // Emit as a counted loop over [0, iters):
        std::string ak = fresh_ssa();
        out << "  " << ak << " = llvm.load %active_k_ptr : !llvm.ptr -> i32\n";
        std::string ak64 = fresh_ssa();
        out << "  " << ak64 << " = llvm.zext " << ak << " : i32 to i64\n";
        std::string ak_m1 = fresh_ssa();
        out << "  " << ak_m1 << " = llvm.sub " << ak64 << ", %c1_i64 : i64\n";
        std::string iters = fresh_ssa();
        out << "  " << iters << " = llvm.shl %c1_i64, " << ak_m1 << " : i64\n";
        char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", axis);
        std::string axis_val = fresh_ssa();
        out << "  " << axis_val << " = llvm.mlir.constant(" << abuf << " : i64) : i64\n";
        std::string axis_bit = fresh_ssa();
        out << "  " << axis_bit << " = llvm.shl %c1_i64, " << axis_val << " : i64\n";

        std::string inv_sq2 = fresh_ssa();
        out << "  " << inv_sq2 << " = llvm.mlir.constant("
            << kInvSqrt2 << " : f64) : f64\n";

        // Loop header
        std::string hdr = fresh_label("h_hdr");
        std::string body = fresh_label("h_body");
        std::string exit = fresh_label("h_exit");
        out << "  llvm.br ^" << hdr << "(%c0_i64 : i64)\n";
        out << "^" << hdr << "(%h_i: i64):\n";
        std::string cond = fresh_ssa();
        out << "  " << cond << " = llvm.icmp \"ult\" %h_i, " << iters << " : i64\n";
        out << "  llvm.cond_br " << cond << ", ^" << body << ", ^" << exit << "\n";
        out << "^" << body << ":\n";

        // idx0 = scatter_bits_1(i, axis)
        std::string idx0 = emit_scatter_bits_1("%h_i", axis_val);
        std::string idx1 = fresh_ssa();
        out << "  " << idx1 << " = llvm.or " << idx0 << ", " << axis_bit << " : i64\n";

        std::string va = emit_load_v(idx0);
        std::string vb = emit_load_v(idx1);
        std::string sum = emit_cadd(va, vb);
        std::string diff = emit_csub(va, vb);
        emit_store_v(idx0, emit_cscale_f64(sum, kInvSqrt2));
        emit_store_v(idx1, emit_cscale_f64(diff, kInvSqrt2));

        std::string i_next = fresh_ssa();
        out << "  " << i_next << " = llvm.add %h_i, %c1_i64 : i64\n";
        out << "  llvm.br ^" << hdr << "(" << i_next << " : i64)\n";
        out << "^" << exit << ":\n";
        (void)inv_sq2;
    };

    auto emit_array_cnot_static = [&](uint32_t ctrl, uint32_t tgt) {
        out << "  // array_cnot ctrl=" << ctrl << " tgt=" << tgt << "\n";
        std::string ak = fresh_ssa(), ak64 = fresh_ssa(), ak_m2 = fresh_ssa(), iters = fresh_ssa();
        out << "  " << ak << " = llvm.load %active_k_ptr : !llvm.ptr -> i32\n";
        out << "  " << ak64 << " = llvm.zext " << ak << " : i32 to i64\n";
        out << "  " << ak_m2 << " = llvm.sub " << ak64 << ", %c1_i64 : i64\n";
        // Note: should be ak-2 for 2-qubit gate, but we reuse ak_m2 name
        std::string ak_m2b = fresh_ssa();
        out << "  " << ak_m2b << " = llvm.sub " << ak64 << ", llvm.mlir.constant(2 : i64) : i64\n";
        out << "  " << iters << " = llvm.shl %c1_i64, " << ak_m2b << " : i64\n";

        char cbuf[32], tbuf[32];
        snprintf(cbuf, sizeof(cbuf), "%u", ctrl);
        snprintf(tbuf, sizeof(tbuf), "%u", tgt);
        std::string c64 = fresh_ssa(), t64 = fresh_ssa(), c_bit = fresh_ssa(), t_bit = fresh_ssa();
        out << "  " << c64 << " = llvm.mlir.constant(" << cbuf << " : i64) : i64\n";
        out << "  " << t64 << " = llvm.mlir.constant(" << tbuf << " : i64) : i64\n";
        out << "  " << c_bit << " = llvm.shl %c1_i64, " << c64 << " : i64\n";
        out << "  " << t_bit << " = llvm.shl %c1_i64, " << t64 << " : i64\n";

        std::string hdr = fresh_label("cn_hdr");
        std::string body = fresh_label("cn_body");
        std::string exit_lbl = fresh_label("cn_exit");
        out << "  llvm.br ^" << hdr << "(%c0_i64 : i64)\n";
        out << "^" << hdr << "(%cn_i: i64):\n";
        std::string cond = fresh_ssa();
        out << "  " << cond << " = llvm.icmp \"ult\" %cn_i, " << iters << " : i64\n";
        out << "  llvm.cond_br " << cond << ", ^" << body << ", ^" << exit_lbl << "\n";
        out << "^" << body << ":\n";

        // scatter_bits_2 = scatter_bits_1(scatter_bits_1(i, min(c,t)), max(c,t))
        uint32_t lo = std::min(ctrl, tgt), hi = std::max(ctrl, tgt);
        char lobuf[32], hibuf[32];
        snprintf(lobuf, sizeof(lobuf), "%u", lo);
        snprintf(hibuf, sizeof(hibuf), "%u", hi);
        std::string lo64 = fresh_ssa(), hi64 = fresh_ssa();
        out << "  " << lo64 << " = llvm.mlir.constant(" << lobuf << " : i64) : i64\n";
        out << "  " << hi64 << " = llvm.mlir.constant(" << hibuf << " : i64) : i64\n";
        std::string s1 = emit_scatter_bits_1("%cn_i", lo64);
        std::string base_pre = emit_scatter_bits_1(s1, hi64);
        std::string base = fresh_ssa();
        out << "  " << base << " = llvm.or " << base_pre << ", " << c_bit << " : i64\n";
        std::string base_t = fresh_ssa();
        out << "  " << base_t << " = llvm.or " << base << ", " << t_bit << " : i64\n";

        std::string va = emit_load_v(base);
        std::string vb = emit_load_v(base_t);
        emit_store_v(base, vb);
        emit_store_v(base_t, va);

        std::string i_next = fresh_ssa();
        out << "  " << i_next << " = llvm.add %cn_i, %c1_i64 : i64\n";
        out << "  llvm.br ^" << hdr << "(" << i_next << " : i64)\n";
        out << "^" << exit_lbl << ":\n";
        (void)ak_m2;
    };

    auto emit_apply_phase_static = [&](uint32_t axis, double phs_re, double phs_im) {
        std::string ak = fresh_ssa(), ak64 = fresh_ssa(), ak_m1 = fresh_ssa(), iters = fresh_ssa();
        out << "  " << ak << " = llvm.load %active_k_ptr : !llvm.ptr -> i32\n";
        out << "  " << ak64 << " = llvm.zext " << ak << " : i32 to i64\n";
        out << "  " << ak_m1 << " = llvm.sub " << ak64 << ", %c1_i64 : i64\n";
        out << "  " << iters << " = llvm.shl %c1_i64, " << ak_m1 << " : i64\n";
        char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", axis);
        std::string axis64 = fresh_ssa(), axis_bit = fresh_ssa();
        out << "  " << axis64 << " = llvm.mlir.constant(" << abuf << " : i64) : i64\n";
        out << "  " << axis_bit << " = llvm.shl %c1_i64, " << axis64 << " : i64\n";

        std::string hdr = fresh_label("ph_hdr");
        std::string body = fresh_label("ph_body");
        std::string exit_lbl = fresh_label("ph_exit");
        out << "  llvm.br ^" << hdr << "(%c0_i64 : i64)\n";
        out << "^" << hdr << "(%ph_i: i64):\n";
        std::string cond = fresh_ssa();
        out << "  " << cond << " = llvm.icmp \"ult\" %ph_i, " << iters << " : i64\n";
        out << "  llvm.cond_br " << cond << ", ^" << body << ", ^" << exit_lbl << "\n";
        out << "^" << body << ":\n";
        std::string idx = fresh_ssa();
        out << "  " << idx << " = llvm.or "
            << emit_scatter_bits_1("%ph_i", axis64) << ", " << axis_bit << " : i64\n";
        std::string vc = emit_load_v(idx);
        emit_store_v(idx, emit_cmul_const(vc, phs_re, phs_im));
        std::string i_next = fresh_ssa();
        out << "  " << i_next << " = llvm.add %ph_i, %c1_i64 : i64\n";
        out << "  llvm.br ^" << hdr << "(" << i_next << " : i64)\n";
        out << "^" << exit_lbl << ":\n";
    };

    // -----------------------------------------------------------------------
    // Main instruction loop
    // -----------------------------------------------------------------------
    using Opcode = clifft::Opcode;
    out << "  // --- Instruction sequence (" << flat.instrs.size() << " ops) ---\n";

    for (size_t pc = 0; pc < flat.instrs.size(); ++pc) {
        const GpuInstr& ins = flat.instrs[pc];
        bool sign = (ins.flags & kFlagSign) != 0;
        bool identity = (ins.flags & kFlagIdentity) != 0;
        auto op = static_cast<Opcode>(ins.opcode);

        out << "  // op[" << pc << "] opcode=" << (unsigned)ins.opcode << "\n";

        switch (op) {
            case Opcode::OP_FRAME_H: {
                uint32_t ax = ins.axis_1;
                char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", ax);
                std::string ax_i32 = fresh_ssa();
                out << "  " << ax_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
                std::string pxv = emit_bit_get("%px_ptr", ax_i32);
                std::string pzv = emit_bit_get("%pz_ptr", ax_i32);
                emit_bit_set("%px_ptr", ax_i32, pzv);
                emit_bit_set("%pz_ptr", ax_i32, pxv);
                break;
            }
            case Opcode::OP_FRAME_S:
            case Opcode::OP_FRAME_S_DAG: {
                uint32_t ax = ins.axis_1;
                char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", ax);
                std::string ax_i32 = fresh_ssa();
                out << "  " << ax_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
                std::string pxv = emit_bit_get("%px_ptr", ax_i32);
                emit_bit_xor("%pz_ptr", ax_i32, pxv);
                break;
            }
            case Opcode::OP_FRAME_CNOT: {
                char cbuf[32], tbuf[32];
                snprintf(cbuf, sizeof(cbuf), "%u", ins.axis_1);
                snprintf(tbuf, sizeof(tbuf), "%u", ins.axis_2);
                std::string c_i32 = fresh_ssa(), t_i32 = fresh_ssa();
                out << "  " << c_i32 << " = llvm.mlir.constant(" << cbuf << " : i32) : i32\n";
                out << "  " << t_i32 << " = llvm.mlir.constant(" << tbuf << " : i32) : i32\n";
                std::string px_c = emit_bit_get("%px_ptr", c_i32);
                std::string pz_t = emit_bit_get("%pz_ptr", t_i32);
                emit_bit_xor("%px_ptr", t_i32, px_c);
                emit_bit_xor("%pz_ptr", c_i32, pz_t);
                break;
            }
            case Opcode::OP_FRAME_CZ: {
                char abuf[32], bbuf[32];
                snprintf(abuf, sizeof(abuf), "%u", ins.axis_1);
                snprintf(bbuf, sizeof(bbuf), "%u", ins.axis_2);
                std::string a_i32 = fresh_ssa(), b_i32 = fresh_ssa();
                out << "  " << a_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
                out << "  " << b_i32 << " = llvm.mlir.constant(" << bbuf << " : i32) : i32\n";
                std::string px_a = emit_bit_get("%px_ptr", a_i32);
                std::string px_b = emit_bit_get("%px_ptr", b_i32);
                emit_bit_xor("%pz_ptr", b_i32, px_a);
                emit_bit_xor("%pz_ptr", a_i32, px_b);
                break;
            }
            case Opcode::OP_ARRAY_H: {
                emit_array_h_static(ins.axis_1);
                // frame_h
                char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", ins.axis_1);
                std::string ax_i32 = fresh_ssa();
                out << "  " << ax_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
                std::string pxv = emit_bit_get("%px_ptr", ax_i32);
                std::string pzv = emit_bit_get("%pz_ptr", ax_i32);
                emit_bit_set("%px_ptr", ax_i32, pzv);
                emit_bit_set("%pz_ptr", ax_i32, pxv);
                break;
            }
            case Opcode::OP_ARRAY_S: {
                emit_apply_phase_static(ins.axis_1, 0.0, 1.0);
                char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", ins.axis_1);
                std::string ax_i32 = fresh_ssa();
                out << "  " << ax_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
                std::string pxv = emit_bit_get("%px_ptr", ax_i32);
                emit_bit_xor("%pz_ptr", ax_i32, pxv);
                break;
            }
            case Opcode::OP_ARRAY_S_DAG: {
                emit_apply_phase_static(ins.axis_1, 0.0, -1.0);
                char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", ins.axis_1);
                std::string ax_i32 = fresh_ssa();
                out << "  " << ax_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
                std::string pxv = emit_bit_get("%px_ptr", ax_i32);
                emit_bit_xor("%pz_ptr", ax_i32, pxv);
                break;
            }
            case Opcode::OP_ARRAY_T: {
                // Phase depends on px[axis] — we need a runtime select
                // For simplicity: emit both branches
                double kIsq2 = kInvSqrt2;
                // If px=0: phase = {kIsq2, kIsq2}; if px=1: phase = {kIsq2, -kIsq2}
                // We compute at runtime using select
                char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", ins.axis_1);
                std::string ax_i32 = fresh_ssa();
                out << "  " << ax_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
                std::string pxv = emit_bit_get("%px_ptr", ax_i32);
                // imag = px ? -kIsq2 : kIsq2
                std::string im_pos = fresh_ssa(), im_neg = fresh_ssa(), imag = fresh_ssa();
                char pbuf[64], nbuf[64];
                snprintf(pbuf, sizeof(pbuf), "%.17e", (float)kIsq2);
                snprintf(nbuf, sizeof(nbuf), "%.17e", (float)-kIsq2);
                out << "  " << im_pos << " = llvm.mlir.constant(" << pbuf << " : f32) : f32\n";
                out << "  " << im_neg << " = llvm.mlir.constant(" << nbuf << " : f32) : f32\n";
                out << "  " << imag << " = llvm.select " << pxv << ", " << im_neg << ", " << im_pos << " : i1, f32\n";
                // Emit loop with runtime phase: (kIsq2, imag)
                // This is a special case — emit as apply_phase with dynamic im
                std::string ak = fresh_ssa(), ak64 = fresh_ssa(), ak_m1 = fresh_ssa(), iters = fresh_ssa();
                out << "  " << ak << " = llvm.load %active_k_ptr : !llvm.ptr -> i32\n";
                out << "  " << ak64 << " = llvm.zext " << ak << " : i32 to i64\n";
                out << "  " << ak_m1 << " = llvm.sub " << ak64 << ", %c1_i64 : i64\n";
                out << "  " << iters << " = llvm.shl %c1_i64, " << ak_m1 << " : i64\n";
                std::string axis64 = fresh_ssa(), axis_bit = fresh_ssa();
                out << "  " << axis64 << " = llvm.mlir.constant(" << abuf << " : i64) : i64\n";
                out << "  " << axis_bit << " = llvm.shl %c1_i64, " << axis64 << " : i64\n";
                std::string are_c = fresh_ssa();
                out << "  " << are_c << " = llvm.mlir.constant(" << pbuf << " : f32) : f32\n";

                std::string hdr = fresh_label("t_hdr"), body = fresh_label("t_body"), exit_lbl = fresh_label("t_exit");
                out << "  llvm.br ^" << hdr << "(%c0_i64 : i64)\n";
                out << "^" << hdr << "(%t_i: i64):\n";
                std::string cond = fresh_ssa();
                out << "  " << cond << " = llvm.icmp \"ult\" %t_i, " << iters << " : i64\n";
                out << "  llvm.cond_br " << cond << ", ^" << body << ", ^" << exit_lbl << "\n";
                out << "^" << body << ":\n";
                std::string idx = fresh_ssa();
                out << "  " << idx << " = llvm.or "
                    << emit_scatter_bits_1("%t_i", axis64) << ", " << axis_bit << " : i64\n";
                std::string vc = emit_load_v(idx);
                // v[idx] = v[idx] * {are_c, imag}
                std::string vc_re = fresh_ssa(), vc_im = fresh_ssa();
                out << "  " << vc_re << " = llvm.extractvalue " << vc << "[0] : !llvm.struct<(f32, f32)>\n";
                out << "  " << vc_im << " = llvm.extractvalue " << vc << "[1] : !llvm.struct<(f32, f32)>\n";
                std::string t0 = fresh_ssa(), t1 = fresh_ssa(), re_out = fresh_ssa();
                out << "  " << t0 << " = llvm.fmul " << vc_re << ", " << are_c << " : f32\n";
                out << "  " << t1 << " = llvm.fmul " << vc_im << ", " << imag << " : f32\n";
                out << "  " << re_out << " = llvm.fsub " << t0 << ", " << t1 << " : f32\n";
                std::string t2 = fresh_ssa(), t3 = fresh_ssa(), im_out = fresh_ssa();
                out << "  " << t2 << " = llvm.fmul " << vc_re << ", " << imag << " : f32\n";
                out << "  " << t3 << " = llvm.fmul " << vc_im << ", " << are_c << " : f32\n";
                out << "  " << im_out << " = llvm.fadd " << t2 << ", " << t3 << " : f32\n";
                std::string res0 = fresh_ssa(), res1 = fresh_ssa(), res = fresh_ssa();
                out << "  " << res0 << " = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
                out << "  " << res1 << " = llvm.insertvalue " << re_out << ", " << res0 << "[0] : !llvm.struct<(f32, f32)>\n";
                out << "  " << res << " = llvm.insertvalue " << im_out << ", " << res1 << "[1] : !llvm.struct<(f32, f32)>\n";
                emit_store_v(idx, res);
                std::string i_next = fresh_ssa();
                out << "  " << i_next << " = llvm.add %t_i, %c1_i64 : i64\n";
                out << "  llvm.br ^" << hdr << "(" << i_next << " : i64)\n";
                out << "^" << exit_lbl << ":\n";
                break;
            }
            case Opcode::OP_ARRAY_T_DAG: {
                // Same as T but negated imag
                char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", ins.axis_1);
                std::string ax_i32 = fresh_ssa();
                out << "  " << ax_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
                std::string pxv = emit_bit_get("%px_ptr", ax_i32);
                // dagger: base_im = -kIsq2; px=1 → negate again → kIsq2
                double kIsq2 = kInvSqrt2;
                std::string im_pos = fresh_ssa(), im_neg = fresh_ssa(), imag = fresh_ssa();
                char pbuf[64], nbuf[64];
                // dagger base = -kIsq2; if px, negate: +kIsq2
                snprintf(pbuf, sizeof(pbuf), "%.17e", (float)-kIsq2); // base
                snprintf(nbuf, sizeof(nbuf), "%.17e", (float)kIsq2);   // px negated
                out << "  " << im_pos << " = llvm.mlir.constant(" << pbuf << " : f32) : f32\n";
                out << "  " << im_neg << " = llvm.mlir.constant(" << nbuf << " : f32) : f32\n";
                out << "  " << imag << " = llvm.select " << pxv << ", " << im_neg << ", " << im_pos << " : i1, f32\n";
                // Same loop as T — reuse emit_apply_phase_static but with runtime imag
                // For brevity: emit as apply_phase with {kIsq2, imag}
                // (copy of T case with different imag)
                std::string ak = fresh_ssa(), ak64 = fresh_ssa(), ak_m1 = fresh_ssa(), iters = fresh_ssa();
                out << "  " << ak << " = llvm.load %active_k_ptr : !llvm.ptr -> i32\n";
                out << "  " << ak64 << " = llvm.zext " << ak << " : i32 to i64\n";
                out << "  " << ak_m1 << " = llvm.sub " << ak64 << ", %c1_i64 : i64\n";
                out << "  " << iters << " = llvm.shl %c1_i64, " << ak_m1 << " : i64\n";
                std::string axis64 = fresh_ssa(), axis_bit = fresh_ssa();
                out << "  " << axis64 << " = llvm.mlir.constant(" << abuf << " : i64) : i64\n";
                out << "  " << axis_bit << " = llvm.shl %c1_i64, " << axis64 << " : i64\n";
                char re_buf[64]; snprintf(re_buf, sizeof(re_buf), "%.17e", (float)kIsq2);
                std::string are_c = fresh_ssa();
                out << "  " << are_c << " = llvm.mlir.constant(" << re_buf << " : f32) : f32\n";
                std::string hdr = fresh_label("td_hdr"), body = fresh_label("td_body"), exit_lbl = fresh_label("td_exit");
                out << "  llvm.br ^" << hdr << "(%c0_i64 : i64)\n";
                out << "^" << hdr << "(%td_i: i64):\n";
                std::string cond = fresh_ssa();
                out << "  " << cond << " = llvm.icmp \"ult\" %td_i, " << iters << " : i64\n";
                out << "  llvm.cond_br " << cond << ", ^" << body << ", ^" << exit_lbl << "\n";
                out << "^" << body << ":\n";
                std::string idx = fresh_ssa();
                out << "  " << idx << " = llvm.or " << emit_scatter_bits_1("%td_i", axis64) << ", " << axis_bit << " : i64\n";
                std::string vc = emit_load_v(idx);
                std::string vc_re = fresh_ssa(), vc_im = fresh_ssa(), t0 = fresh_ssa(), t1 = fresh_ssa(), re_out = fresh_ssa(), t2 = fresh_ssa(), t3 = fresh_ssa(), im_out = fresh_ssa();
                out << "  " << vc_re << " = llvm.extractvalue " << vc << "[0] : !llvm.struct<(f32, f32)>\n";
                out << "  " << vc_im << " = llvm.extractvalue " << vc << "[1] : !llvm.struct<(f32, f32)>\n";
                out << "  " << t0 << " = llvm.fmul " << vc_re << ", " << are_c << " : f32\n";
                out << "  " << t1 << " = llvm.fmul " << vc_im << ", " << imag << " : f32\n";
                out << "  " << re_out << " = llvm.fsub " << t0 << ", " << t1 << " : f32\n";
                out << "  " << t2 << " = llvm.fmul " << vc_re << ", " << imag << " : f32\n";
                out << "  " << t3 << " = llvm.fmul " << vc_im << ", " << are_c << " : f32\n";
                out << "  " << im_out << " = llvm.fadd " << t2 << ", " << t3 << " : f32\n";
                std::string r0 = fresh_ssa(), r1 = fresh_ssa(), res = fresh_ssa();
                out << "  " << r0 << " = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
                out << "  " << r1 << " = llvm.insertvalue " << re_out << ", " << r0 << "[0] : !llvm.struct<(f32, f32)>\n";
                out << "  " << res << " = llvm.insertvalue " << im_out << ", " << r1 << "[1] : !llvm.struct<(f32, f32)>\n";
                emit_store_v(idx, res);
                std::string i_next = fresh_ssa();
                out << "  " << i_next << " = llvm.add %td_i, %c1_i64 : i64\n";
                out << "  llvm.br ^" << hdr << "(" << i_next << " : i64)\n";
                out << "^" << exit_lbl << ":\n";
                break;
            }
            case Opcode::OP_ARRAY_CNOT:
                emit_array_cnot_static(ins.axis_1, ins.axis_2);
                {
                    char cbuf[32], tbuf[32];
                    snprintf(cbuf, sizeof(cbuf), "%u", ins.axis_1);
                    snprintf(tbuf, sizeof(tbuf), "%u", ins.axis_2);
                    std::string c_i32 = fresh_ssa(), t_i32 = fresh_ssa();
                    out << "  " << c_i32 << " = llvm.mlir.constant(" << cbuf << " : i32) : i32\n";
                    out << "  " << t_i32 << " = llvm.mlir.constant(" << tbuf << " : i32) : i32\n";
                    std::string px_c = emit_bit_get("%px_ptr", c_i32);
                    std::string pz_t = emit_bit_get("%pz_ptr", t_i32);
                    emit_bit_xor("%px_ptr", t_i32, px_c);
                    emit_bit_xor("%pz_ptr", c_i32, pz_t);
                }
                break;
            case Opcode::OP_MEAS_DORMANT_STATIC: {
                // meas[a] = identity ? sign : (px[axis] ^ sign)
                char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", ins.a);
                std::string midx = fresh_ssa();
                out << "  " << midx << " = llvm.mlir.constant(" << abuf << " : i64) : i64\n";
                std::string mptr = fresh_ssa();
                out << "  " << mptr << " = llvm.getelementptr inbounds %meas_ptr["
                    << midx << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
                std::string mval;
                if (identity) {
                    mval = fresh_ssa();
                    out << "  " << mval << " = llvm.mlir.constant("
                        << (sign ? 1 : 0) << " : i8) : i8\n";
                } else {
                    char axbuf[32]; snprintf(axbuf, sizeof(axbuf), "%u", ins.axis_1);
                    std::string ax_i32 = fresh_ssa();
                    out << "  " << ax_i32 << " = llvm.mlir.constant(" << axbuf << " : i32) : i32\n";
                    std::string px_bit = emit_bit_get("%px_ptr", ax_i32);
                    std::string px_i8 = fresh_ssa();
                    out << "  " << px_i8 << " = llvm.zext " << px_bit << " : i1 to i8\n";
                    if (sign) {
                        mval = fresh_ssa();
                        out << "  " << mval << " = llvm.xor " << px_i8 << ", %c1_i8 : i8\n";
                    } else {
                        mval = px_i8;
                    }
                }
                out << "  llvm.store " << mval << ", " << mptr << " : i8, !llvm.ptr\n";
                break;
            }
            case Opcode::OP_OBSERVABLE: {
                // obs[ins.b] ^= parity of meas[observable_targets[start..end]]
                if (ins.a + 1 < flat.observable_offsets.size()) {
                    uint32_t ob_start = flat.observable_offsets[ins.a];
                    uint32_t ob_end = flat.observable_offsets[ins.a + 1];
                    char bidx_buf[32]; snprintf(bidx_buf, sizeof(bidx_buf), "%u", ins.b);
                    std::string obs_idx = fresh_ssa();
                    out << "  " << obs_idx << " = llvm.mlir.constant(" << bidx_buf << " : i64) : i64\n";
                    std::string obs_elem_ptr = fresh_ssa();
                    out << "  " << obs_elem_ptr << " = llvm.getelementptr inbounds %obs_ptr["
                        << obs_idx << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
                    std::string parity = "%c0_i8";
                    for (uint32_t k = ob_start; k < ob_end; ++k) {
                        char tidx_buf[32]; snprintf(tidx_buf, sizeof(tidx_buf), "%u", flat.observable_targets[k]);
                        std::string tidx = fresh_ssa();
                        out << "  " << tidx << " = llvm.mlir.constant(" << tidx_buf << " : i64) : i64\n";
                        std::string tptr = fresh_ssa();
                        out << "  " << tptr << " = llvm.getelementptr inbounds %meas_ptr["
                            << tidx << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
                        std::string tval = fresh_ssa();
                        out << "  " << tval << " = llvm.load " << tptr << " : !llvm.ptr -> i8\n";
                        std::string new_parity = fresh_ssa();
                        out << "  " << new_parity << " = llvm.xor " << parity << ", " << tval << " : i8\n";
                        parity = new_parity;
                    }
                    std::string cur_obs = fresh_ssa();
                    out << "  " << cur_obs << " = llvm.load " << obs_elem_ptr << " : !llvm.ptr -> i8\n";
                    std::string new_obs = fresh_ssa();
                    out << "  " << new_obs << " = llvm.xor " << cur_obs << ", " << parity << " : i8\n";
                    out << "  llvm.store " << new_obs << ", " << obs_elem_ptr << " : i8, !llvm.ptr\n";
                }
                break;
            }
            case Opcode::OP_DETECTOR:
                // No-op at runtime
                break;
            default:
                out << "  // Unsupported op " << (unsigned)ins.opcode << " — mark discarded\n";
                out << "  llvm.store %c1_i8, %discarded_ptr : i8, !llvm.ptr\n";
                break;
        }
    }

    out << "  // --- End instruction sequence ---\n";
    out << "  llvm.return\n";
    out << "}\n\n";
    out << "} // end module\n";

    return out.str();
}

}  // anonymous namespace

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------
std::string generate_mlir_kernel_llvmir(const FlattenedProgram& flat,
                                         const std::string& gpu_arch) {
    // Step 1: Generate textual MLIR
    std::string mlir_text = emit_mlir_text(flat);
    if (mlir_text.empty()) {
        std::cerr << "[clifft-mlir] MLIR text generation failed\n";
        return {};
    }

    // Write MLIR to temp file for subprocess processing
    std::string hash_src = mlir_text + gpu_arch;
    // Simple hash for temp filename
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : hash_src) { h ^= c; h *= 1099511628211ULL; }
    char hbuf[20]; snprintf(hbuf, sizeof(hbuf), "%016llx", (unsigned long long)h);
    std::string tmp_mlir = "/tmp/clifft_mlir_" + std::string(hbuf) + ".mlir";
    std::string tmp_ll = "/tmp/clifft_mlir_" + std::string(hbuf) + ".ll";

    {
        std::ofstream f(tmp_mlir);
        if (!f) {
            std::cerr << "[clifft-mlir] failed to write MLIR text to " << tmp_mlir << "\n";
            return {};
        }
        f << mlir_text;
    }

    std::string mlir_opt = find_mlir_opt();
    std::string mlir_translate = find_mlir_translate();

    // Step 2: mlir-opt to optimize and lower to LLVM IR dialect
    // Passes: canonicalize, CSE, then convert to LLVM
    std::ostringstream cmd_opt;
    cmd_opt << "\"" << mlir_opt << "\""
            << " --canonicalize"
            << " --cse"
            << " --convert-func-to-llvm"
            << " \"" << tmp_mlir << "\""
            << " 2>&1";

    FILE* pipe = popen(cmd_opt.str().c_str(), "r");
    std::string opt_out;
    if (!pipe) {
        std::cerr << "[clifft-mlir] mlir-opt: popen failed\n";
        std::filesystem::remove(tmp_mlir);
        return {};
    }
    {
        std::array<char, 65536> buf;
        while (fgets(buf.data(), (int)buf.size(), pipe)) opt_out += buf.data();
    }
    int opt_rc = pclose(pipe);
    int opt_exit = WIFEXITED(opt_rc) ? WEXITSTATUS(opt_rc) : -1;

    if (opt_exit != 0) {
        std::cerr << "[clifft-mlir] mlir-opt failed (rc=" << opt_exit << ")\n"
                  << "  output: " << opt_out.substr(0, 2000) << "\n";
        std::filesystem::remove(tmp_mlir);
        return {};
    }

    // opt_out contains the optimized MLIR text — write to tmp_mlir for translate
    {
        std::ofstream f(tmp_mlir);
        f << opt_out;
    }

    // Step 3: mlir-translate --mlir-to-llvmir → LLVM-IR text
    std::ostringstream cmd_tr;
    cmd_tr << "\"" << mlir_translate << "\""
           << " --mlir-to-llvmir"
           << " \"" << tmp_mlir << "\""
           << " -o \"" << tmp_ll << "\""
           << " 2>&1";

    std::string tr_out;
    FILE* tr_pipe = popen(cmd_tr.str().c_str(), "r");
    if (!tr_pipe) {
        std::cerr << "[clifft-mlir] mlir-translate: popen failed\n";
        std::filesystem::remove(tmp_mlir);
        return {};
    }
    {
        std::array<char, 65536> buf;
        while (fgets(buf.data(), (int)buf.size(), tr_pipe)) tr_out += buf.data();
    }
    int tr_rc = pclose(tr_pipe);
    int tr_exit = WIFEXITED(tr_rc) ? WEXITSTATUS(tr_rc) : -1;
    std::filesystem::remove(tmp_mlir);

    if (tr_exit != 0) {
        std::cerr << "[clifft-mlir] mlir-translate failed (rc=" << tr_exit << ")\n"
                  << "  output: " << tr_out.substr(0, 2000) << "\n";
        return {};
    }

    // Read LLVM-IR from output file
    std::ifstream llfile(tmp_ll);
    if (!llfile) {
        std::cerr << "[clifft-mlir] cannot read output: " << tmp_ll << "\n";
        return {};
    }
    std::string ll_text((std::istreambuf_iterator<char>(llfile)),
                          std::istreambuf_iterator<char>());
    std::filesystem::remove(tmp_ll);

    if (ll_text.empty()) {
        std::cerr << "[clifft-mlir] mlir-translate produced empty LLVM-IR\n";
        return {};
    }

    std::cerr << "[clifft-mlir] generated " << ll_text.size()
              << " bytes of LLVM-IR for " << flat.instrs.size() << " ops\n";
    return ll_text;
}

}  // namespace gpu
}  // namespace clifft

#endif  // CLIFFT_ENABLE_MLIR
