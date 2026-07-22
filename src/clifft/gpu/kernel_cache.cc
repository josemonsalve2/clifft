// kernel_cache.cc — compile HIP kernel source and load via HSA
//
// Generates HIP source → compiles via clang++ → produces .hsaco →
// loads via HSA (hsa_executable + hsa_code_object_reader) → returns
// HsaLoadedKernel ready for AQL dispatch.
//
// Replaces the previous hipModuleLoad/hipModuleGetFunction path.
// The .hsaco caching on disk is preserved (same hash-based scheme).

#include "clifft/gpu/kernel_cache.h"
#include "clifft/gpu/kernel_codegen.h"
#include "clifft/gpu/hsa_runtime.h"
#include "clifft/gpu/hsa_kernel_dispatch.h"

#include <sys/wait.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace clifft {
namespace gpu {

namespace {

std::string fnv1a_hex(const std::string& data) {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : data) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    char buf[20];
    snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(hash));
    return buf;
}

std::string cache_dir() {
    const char* home = std::getenv("HOME");
    if (!home) home = "/tmp";
    std::string dir = std::string(home) + "/.clifft/kernel_cache";
    std::filesystem::create_directories(dir);
    return dir;
}

std::string find_clangpp() {
    const char* rocm_path = std::getenv("ROCM_PATH");
    if (rocm_path) {
        std::string candidate = std::string(rocm_path) + "/lib/llvm/bin/clang++";
        if (std::filesystem::exists(candidate)) return candidate;
    }
    if (std::filesystem::exists("/opt/rocm/lib/llvm/bin/clang++"))
        return "/opt/rocm/lib/llvm/bin/clang++";
    return "clang++";
}

int run_command(const std::string& cmd, std::string& output) {
    std::string full_cmd = cmd + " 2>&1";
    FILE* pipe = popen(full_cmd.c_str(), "r");
    if (!pipe) { output = "popen failed"; return -1; }
    std::array<char, 4096> buf;
    output.clear();
    while (fgets(buf.data(), static_cast<int>(buf.size()), pipe))
        output += buf.data();
    int status = pclose(pipe);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

/// Compile source to .hsaco (if not cached), then load via HSA.
HsaLoadedKernel compile_and_load(const std::string& source,
                                  const std::string& kernel_func_name,
                                  const std::string& tier_suffix) {
    HsaLoadedKernel lk;
    if (source.empty()) {
        std::cerr << "[clifft-gpu] compiled kernel (" << kernel_func_name
                  << "): codegen produced empty source\n";
        return lk;
    }

    // Get GPU arch from HSA runtime
    auto& rt = hsa_runtime();
    std::string gpu_arch = rt.device(0).arch_name;

    std::string hash = fnv1a_hex(source + gpu_arch + tier_suffix);
    std::string dir = cache_dir();
    std::string hsaco_path = dir + "/" + hash + "_" + gpu_arch + ".hsaco";

    // Compile if not cached
    if (!std::filesystem::exists(hsaco_path)) {
        std::string tmp_src = "/tmp/clifft_kernel_" + hash + ".hip";
        {
            std::ofstream out(tmp_src);
            if (!out) {
                std::cerr << "[clifft-gpu] failed to write temp source " << tmp_src << "\n";
                return lk;
            }
            out << source;
        }

        std::string clangpp = find_clangpp();

        // Step 1: Compile HIP source → offload bundle (.bundle)
        // --offload-device-only produces a Clang Offload Bundle, not a raw ELF.
        // HSA needs a raw AMDGCN ELF, so we unbundle in step 2.
        std::string bundle_path = hsaco_path + ".bundle";
        std::ostringstream cmd;
        cmd << "\"" << clangpp << "\""
            << " -x hip"
            << " --offload-arch=" << gpu_arch
            << " -O2 -ffast-math"
            << " --offload-device-only"
            << " -o \"" << bundle_path << "\""
            << " \"" << tmp_src << "\"";

        auto t0 = std::chrono::steady_clock::now();
        std::string compile_output;
        int rc = run_command(cmd.str(), compile_output);

        if (rc == 0) {
            // Step 2: Unbundle to extract raw AMDGCN ELF for HSA.
            // --offload-device-only produces a Clang Offload Bundle wrapper.
            // HSA hsa_executable_load_agent_code_object needs a raw ELF.
            std::string bundler = clangpp.substr(0, clangpp.rfind('/') + 1)
                                  + "clang-offload-bundler";

            // First, list what's in the bundle to find the correct target ID
            std::ostringstream list_cmd;
            list_cmd << "\"" << bundler << "\" --list --type=o"
                     << " --input=\"" << bundle_path << "\"";
            std::string list_out;
            run_command(list_cmd.str(), list_out);
            std::cerr << "[clifft-gpu] bundle entries: " << list_out;

            // Try unbundling with each known target ID format
            const char* target_formats[] = {
                "hipv4-amdgcn-amd-amdhsa--",
                "hip-amdgcn-amd-amdhsa--",
                "host-x86_64-unknown-linux-gnu-",  // skip host
                nullptr
            };
            bool unbundled = false;
            for (int i = 0; target_formats[i] && !unbundled; ++i) {
                if (std::string(target_formats[i]).find("host") != std::string::npos)
                    continue;
                std::ostringstream ucmd;
                ucmd << "\"" << bundler << "\" --unbundle --type=o"
                     << " --targets=" << target_formats[i] << gpu_arch
                     << " --input=\"" << bundle_path << "\""
                     << " --output=\"" << hsaco_path << "\"";
                std::string uout;
                int urc = run_command(ucmd.str(), uout);
                if (urc == 0 && std::filesystem::exists(hsaco_path)) {
                    unbundled = true;
                    std::cerr << "[clifft-gpu] unbundled with target "
                              << target_formats[i] << gpu_arch << "\n";
                }
            }
            if (!unbundled) {
                std::cerr << "[clifft-gpu] unbundle failed, using bundle as-is\n";
                std::filesystem::rename(bundle_path, hsaco_path);
            } else {
                std::filesystem::remove(bundle_path);
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        double compile_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rc != 0) {
            std::cerr << "[clifft-gpu] clang++ failed for " << kernel_func_name
                      << " (rc=" << rc << ")\n"
                      << "  cmd: " << cmd.str() << "\n"
                      << "  out: " << compile_output << "\n";
            return lk;
        }

        std::cerr << "[clifft-gpu] compiled " << kernel_func_name << " in "
                  << compile_ms << " ms, hash=" << hash << "\n";
        // Keep source for debugging: std::filesystem::remove(tmp_src);
        std::cerr << "[clifft-gpu] generated source kept at: " << tmp_src << "\n";
    } else {
        std::cerr << "[clifft-gpu] cache hit for " << kernel_func_name
                  << ", hash=" << hash << "\n";
    }

    // Load via HSA (replaces hipModuleLoad + hipModuleGetFunction)
    return hsa_load_kernel(hsaco_path, kernel_func_name, 0);
}

}  // namespace

HsaLoadedKernel compile_or_load_kernel(const FlattenedProgram& flat) {
    return compile_and_load(generate_compiled_kernel(flat),
                             "compiled_sample_kernel", "thread");
}

HsaLoadedKernel compile_or_load_kernel_coop(const FlattenedProgram& flat) {
    return compile_and_load(generate_compiled_kernel_coop(flat),
                             "compiled_sample_kernel_coop", "coop");
}

HsaLoadedKernel compile_or_load_kernel_global(const FlattenedProgram& flat) {
    return compile_and_load(generate_compiled_kernel_global(flat),
                             "compiled_sample_kernel_global", "global");
}

void free_compiled_kernel(HsaLoadedKernel& lk) {
    hsa_free_kernel(lk);
}

}  // namespace gpu
}  // namespace clifft
