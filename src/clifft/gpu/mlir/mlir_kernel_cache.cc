// mlir_kernel_cache.cc — compile and cache MLIR→LLVM-IR megakernels
//
// Takes LLVM-IR text produced by generate_mlir_kernel_llvmir() and compiles it
// to a .hsaco using llc + lld (or clang++ fallback).
// Loads via HSA (hsa_load_kernel). Caches on disk.

#ifdef CLIFFT_ENABLE_MLIR

#include "clifft/gpu/mlir/mlir_kernel_cache.h"
#include "clifft/gpu/mlir/mlir_codegen.h"
#include "clifft/gpu/runtime/hsa_runtime.h"
#include "clifft/gpu/runtime/hsa_kernel_dispatch.h"

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

std::string fnv1a_hex_mlir(const std::string& data) {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : data) { hash ^= c; hash *= 1099511628211ULL; }
    char buf[20];
    snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(hash));
    return buf;
}

std::string mlir_cache_dir() {
    const char* home = std::getenv("HOME");
    if (!home) home = "/tmp";
    std::string dir = std::string(home) + "/.clifft/kernel_cache/mlir";
    std::filesystem::create_directories(dir);
    return dir;
}

std::string find_llc() {
    const char* llvm = std::getenv("LLVM_PREFIX");
    if (llvm) {
        std::string c = std::string(llvm) + "/bin/llc";
        if (std::filesystem::exists(c)) return c;
    }
    if (std::filesystem::exists("/shared/jmonsalv/software/modules/llvm/upstream_05082025/bin/llc"))
        return "/shared/jmonsalv/software/modules/llvm/upstream_05082025/bin/llc";
    return "llc";
}

std::string find_lld() {
    const char* llvm = std::getenv("LLVM_PREFIX");
    if (llvm) {
        std::string c = std::string(llvm) + "/bin/ld.lld";
        if (std::filesystem::exists(c)) return c;
    }
    if (std::filesystem::exists("/shared/jmonsalv/software/modules/llvm/upstream_05082025/bin/ld.lld"))
        return "/shared/jmonsalv/software/modules/llvm/upstream_05082025/bin/ld.lld";
    return "ld.lld";
}

std::string find_clangpp_mlir() {
    const char* rocm = std::getenv("ROCM_PATH");
    if (rocm) {
        std::string c = std::string(rocm) + "/lib/llvm/bin/clang++";
        if (std::filesystem::exists(c)) return c;
    }
    return "clang++";
}

int run_cmd(const std::string& cmd, std::string& out) {
    std::string full = cmd + " 2>&1";
    FILE* pipe = popen(full.c_str(), "r");
    if (!pipe) { out = "popen failed"; return -1; }
    std::array<char, 4096> buf;
    out.clear();
    while (fgets(buf.data(), (int)buf.size(), pipe)) out += buf.data();
    int st = pclose(pipe);
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

}  // namespace

HsaLoadedKernel compile_or_load_mlir_kernel(const FlattenedProgram& flat) {
    HsaLoadedKernel lk;

    // Get GPU arch from HSA runtime
    auto& rt = hsa_runtime();
    std::string gpu_arch = rt.device(0).arch_name;

    // Generate LLVM-IR from MLIR
    std::string llvmir = generate_mlir_kernel_llvmir(flat, gpu_arch);
    if (llvmir.empty()) {
        std::cerr << "[clifft-mlir] LLVM-IR generation failed\n";
        return lk;
    }

    std::string hash = fnv1a_hex_mlir(llvmir + gpu_arch);
    std::string dir = mlir_cache_dir();
    std::string hsaco_path = dir + "/mlir_" + hash + "_" + gpu_arch + ".hsaco";

    if (!std::filesystem::exists(hsaco_path)) {
        std::string tmp_ll = "/tmp/clifft_mlir_" + hash + ".ll";
        std::string tmp_obj = "/tmp/clifft_mlir_" + hash + ".o";

        {
            std::ofstream f(tmp_ll);
            if (!f) {
                std::cerr << "[clifft-mlir] failed to write LLVM-IR to " << tmp_ll << "\n";
                return lk;
            }
            f << llvmir;
        }

        auto t0 = std::chrono::steady_clock::now();

        // Try llc → lld pipeline first
        std::string llc = find_llc();
        std::ostringstream cmd_llc;
        cmd_llc << "\"" << llc << "\""
                << " --march=amdgcn --mcpu=" << gpu_arch
                << " -mattr=+wavefrontsize64 -filetype=obj"
                << " -o \"" << tmp_obj << "\" \"" << tmp_ll << "\"";

        std::string out_llc;
        int rc_llc = run_cmd(cmd_llc.str(), out_llc);

        if (rc_llc == 0) {
            // Link to .hsaco
            std::string lld = find_lld();
            std::ostringstream cmd_lld;
            cmd_lld << "\"" << lld << "\" -shared -o \"" << hsaco_path << "\" \"" << tmp_obj << "\"";
            std::string out_lld;
            int rc_lld = run_cmd(cmd_lld.str(), out_lld);
            std::filesystem::remove(tmp_obj);
            if (rc_lld != 0) {
                std::cerr << "[clifft-mlir] lld failed: " << out_lld << "\n";
                // Fall through to clang++ fallback below
                rc_llc = -1;
            }
        }

        if (rc_llc != 0) {
            // Fallback: clang++ -x ir
            std::cerr << "[clifft-mlir] llc/lld failed, trying clang++ fallback\n";
            std::string clangpp = find_clangpp_mlir();
            std::ostringstream cmd_cl;
            cmd_cl << "\"" << clangpp << "\" -x ir --offload-arch=" << gpu_arch
                   << " -O3 --offload-device-only"
                   << " -o \"" << hsaco_path << "\" \"" << tmp_ll << "\"";
            std::string out_cl;
            int rc_cl = run_cmd(cmd_cl.str(), out_cl);
            if (rc_cl != 0) {
                std::cerr << "[clifft-mlir] COMPILATION FAILED (all methods):\n"
                          << "  llc output: " << out_llc << "\n"
                          << "  clang++ output: " << out_cl << "\n";
                std::filesystem::remove(tmp_ll);
                return lk;
            }
        }

        std::filesystem::remove(tmp_ll);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cerr << "[clifft-mlir] compiled in " << ms << " ms, hash=" << hash << "\n";
    } else {
        std::cerr << "[clifft-mlir] cache hit, hash=" << hash << "\n";
    }

    // Load via HSA
    return hsa_load_kernel(hsaco_path, "compiled_mlir_kernel", 0);
}

}  // namespace gpu
}  // namespace clifft

#endif  // CLIFFT_ENABLE_MLIR
