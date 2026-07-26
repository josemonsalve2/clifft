// v2_compile_cache.cc — see header. Mirrors cmake/ClifftAmdgcn.cmake OCML path.
#include "clifft/gpu/mlir/v2/v2_compile_cache.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>

// Toolchain/config passed from cmake (same values ClifftAmdgcn.cmake uses).
#ifndef CLIFFT_V2_LLVM_PREFIX
#define CLIFFT_V2_LLVM_PREFIX ""
#endif
#ifndef CLIFFT_V2_AMDGPU_ARCH
#define CLIFFT_V2_AMDGPU_ARCH "gfx950"
#endif
#ifndef CLIFFT_V2_ROCM_BITCODE_DIR
#define CLIFFT_V2_ROCM_BITCODE_DIR ""
#endif
#ifndef CLIFFT_V2_SRC_INCLUDE
#define CLIFFT_V2_SRC_INCLUDE ""   // -I dir so v2_ops.h resolves
#endif
#ifndef CLIFFT_V2_CACHE_DIR
#define CLIFFT_V2_CACHE_DIR ""     // where to write generated .c/.hsaco
#endif

namespace clifft::gpu::v2 {

namespace fs = std::filesystem;

namespace {

std::string llvm_bin(const char* tool) {
    std::string prefix = CLIFFT_V2_LLVM_PREFIX;
    if (const char* e = getenv("LLVM_PREFIX")) prefix = e;
    if (!prefix.empty()) {
        std::string c = prefix + "/bin/" + tool;
        if (fs::exists(c)) return c;
    }
    return tool;  // rely on PATH
}

std::string bitcode_dir() {
    if (const char* e = getenv("CLIFFT_V2_ROCM_BITCODE_DIR")) return e;
    return CLIFFT_V2_ROCM_BITCODE_DIR;
}

std::string cache_dir() {
    if (const char* e = getenv("V2_SPEC_CACHE_DIR")) return e;
    std::string d = CLIFFT_V2_CACHE_DIR;
    if (!d.empty()) return d;
    return (fs::temp_directory_path() / "clifft_v2_spec").string();
}

std::string arch() {
    if (const char* e = getenv("CLIFFT_V2_AMDGPU_ARCH")) return e;
    return CLIFFT_V2_AMDGPU_ARCH;
}

// Run a command; throw with captured stderr on nonzero exit.
void run(const std::string& cmd) {
    std::string full = cmd + " 2>&1";
    FILE* p = popen(full.c_str(), "r");
    if (!p) throw std::runtime_error("v2 compile: popen failed: " + cmd);
    std::string out;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), p)) > 0) out.append(buf, n);
    int rc = pclose(p);
    if (rc != 0)
        throw std::runtime_error("v2 compile step failed (rc=" + std::to_string(rc) + "):\n" +
                                 cmd + "\n--- output ---\n" + out);
}

}  // namespace

bool specializer_toolchain_available() {
    std::string bc = bitcode_dir();
    return fs::exists(llvm_bin("clang")) && fs::exists(llvm_bin("llc")) &&
           fs::exists(llvm_bin("ld.lld")) && fs::exists(llvm_bin("opt")) &&
           fs::exists(llvm_bin("llvm-link")) &&
           !bc.empty() && fs::exists(bc + "/ocml.bc");
}

std::string compile_specialized(const std::string& csrc, const std::string& key) {
    const std::string dir = cache_dir();
    fs::create_directories(dir);

    // Content hash of source + toolchain identity so cache invalidates on change.
    std::string ident = csrc + "|" + llvm_bin("clang") + "|" + arch() + "|" + bitcode_dir();
    size_t h = std::hash<std::string>{}(ident);
    std::ostringstream tag;
    tag << key << "_" << std::hex << h;
    const std::string base = (fs::path(dir) / tag.str()).string();
    const std::string hsaco = base + ".hsaco";
    if (fs::exists(hsaco)) return hsaco;   // cache hit — no recompile

    const std::string src = base + ".c";
    const std::string bc = base + ".bc";
    const std::string linked = base + ".linked.bc";
    const std::string obj = base + ".o";
    { std::ofstream f(src); f << csrc; }

    const std::string clang = llvm_bin("clang");
    const std::string llvmlink = llvm_bin("llvm-link");
    const std::string opt = llvm_bin("opt");
    const std::string llc = llvm_bin("llc");
    const std::string lld = llvm_bin("ld.lld");
    const std::string cpu = arch();
    const std::string ctl = bitcode_dir();
    std::string isa = cpu;
    if (isa.rfind("gfx", 0) == 0) isa = isa.substr(3);
    const std::string inc = CLIFFT_V2_SRC_INCLUDE;

    // 1) C -> amdgcn bitcode. Flags are IDENTICAL to the build-time interpreter
    // pipeline (ClifftAmdgcn.cmake) so the shared v2_op_*() bodies compile to the
    // same code -> byte-exact. -ffp-contract=off with no fast-math is what makes
    // that hold regardless of inlining decisions: without reassociation or
    // contraction the optimizer cannot change an FP result. Do NOT add
    // vectorize/unroll/fast-math flags that would diverge from the interpreter.
    run(clang + " --target=amdgcn-amd-amdhsa -mcpu=" + cpu +
        " -ffreestanding -nostdlib -nogpulib -std=c23 -O2 -ffp-contract=off"
        " -I" + inc + " -emit-llvm -c -o " + bc + " " + src);
    // 2) link ocml + oclc controls (identical set to ClifftAmdgcn.cmake)
    run(llvmlink + " -o " + linked + " " + bc + " " +
        ctl + "/ocml.bc " + ctl + "/ockl.bc " +
        ctl + "/oclc_wavefrontsize64_on.bc " +
        ctl + "/oclc_daz_opt_off.bc " +
        ctl + "/oclc_finite_only_off.bc " +
        ctl + "/oclc_unsafe_math_off.bc " +
        ctl + "/oclc_correctly_rounded_sqrt_on.bc " +
        ctl + "/oclc_abi_version_500.bc " +
        ctl + "/oclc_isa_version_" + isa + ".bc");
    // 3) internalize + optimize (identical to ClifftAmdgcn.cmake)
    run(opt + " -O2 -o " + linked + " " + linked);
    // 4) bitcode -> object (identical to ClifftAmdgcn.cmake)
    run(llc + " -mtriple=amdgcn-amd-amdhsa -mcpu=" + cpu +
        " -mattr=+wavefrontsize64 -filetype=obj -O2 -o " + obj + " " + linked);
    // 5) object -> .hsaco
    run(lld + " -shared -o " + hsaco + " " + obj);

    if (!fs::exists(hsaco))
        throw std::runtime_error("v2 compile: .hsaco not produced for " + key);
    return hsaco;
}

}  // namespace clifft::gpu::v2
