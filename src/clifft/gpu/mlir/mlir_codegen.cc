// mlir_codegen.cc — MLIR→LLVM-IR pipeline orchestration
//
// Pipeline:
//   1. emit_mlir_text()     (mlir_emit.cc) → textual MLIR (LLVM dialect)
//   2. mlir-opt subprocess  → canonicalize, CSE, convert-func-to-llvm
//   3. mlir-translate       → LLVM-IR text (.ll)
//
// The .ll file is then compiled by mlir_kernel_cache.cc using llc + lld.

#ifdef CLIFFT_ENABLE_MLIR

#include "clifft/gpu/mlir/mlir_codegen.h"
#include "clifft/gpu/mlir/mlir_emit.h"
#include "clifft/gpu/gpu_types.h"

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/wait.h>

namespace clifft {
namespace gpu {

std::string generate_mlir_kernel_llvmir(const FlattenedProgram& flat,
                                         const std::string& gpu_arch) {
    std::string mlir_text = mlir_emit::emit_mlir_text(flat);
    if (mlir_text.empty()) {
        std::cerr << "[clifft-mlir] MLIR text generation failed\n";
        return {};
    }

    std::string hash_src = mlir_text + gpu_arch;
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

    std::string mlir_opt = mlir_emit::find_mlir_opt();
    std::string mlir_translate = mlir_emit::find_mlir_translate();

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

    {
        std::ofstream f(tmp_mlir);
        f << opt_out;
    }

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

std::string generate_mlir_kernel_llvmir_coop(const FlattenedProgram& flat,
                                              const std::string& gpu_arch) {
    std::string mlir_text = mlir_emit::emit_mlir_text_coop(flat);
    if (mlir_text.empty()) {
        std::cerr << "[clifft-mlir-coop] MLIR text generation failed\n";
        return {};
    }

    std::string hash_src = mlir_text + gpu_arch;
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : hash_src) { h ^= c; h *= 1099511628211ULL; }
    char hbuf[20]; snprintf(hbuf, sizeof(hbuf), "%016llx", (unsigned long long)h);
    std::string tmp_mlir = "/tmp/clifft_mlir_coop_" + std::string(hbuf) + ".mlir";
    std::string tmp_ll = "/tmp/clifft_mlir_coop_" + std::string(hbuf) + ".ll";

    { std::ofstream f(tmp_mlir); if (!f) return {}; f << mlir_text; }

    std::string mlir_opt = mlir_emit::find_mlir_opt();
    std::string mlir_translate = mlir_emit::find_mlir_translate();

    std::ostringstream cmd_opt;
    cmd_opt << "\"" << mlir_opt << "\" --canonicalize --cse --convert-func-to-llvm"
            << " \"" << tmp_mlir << "\" 2>&1";
    FILE* pipe = popen(cmd_opt.str().c_str(), "r");
    std::string opt_out;
    if (!pipe) { std::filesystem::remove(tmp_mlir); return {}; }
    { std::array<char, 65536> buf; while (fgets(buf.data(), (int)buf.size(), pipe)) opt_out += buf.data(); }
    int opt_rc = pclose(pipe);
    int opt_exit = WIFEXITED(opt_rc) ? WEXITSTATUS(opt_rc) : -1;
    if (opt_exit != 0) {
        std::cerr << "[clifft-mlir-coop] mlir-opt failed (rc=" << opt_exit << ")\n"
                  << "  output: " << opt_out.substr(0, 2000) << "\n";
        std::filesystem::remove(tmp_mlir);
        return {};
    }
    { std::ofstream f(tmp_mlir); f << opt_out; }

    std::ostringstream cmd_tr;
    cmd_tr << "\"" << mlir_translate << "\" --mlir-to-llvmir \"" << tmp_mlir
           << "\" -o \"" << tmp_ll << "\" 2>&1";
    std::string tr_out;
    FILE* tr_pipe = popen(cmd_tr.str().c_str(), "r");
    if (!tr_pipe) { std::filesystem::remove(tmp_mlir); return {}; }
    { std::array<char, 65536> buf; while (fgets(buf.data(), (int)buf.size(), tr_pipe)) tr_out += buf.data(); }
    int tr_rc = pclose(tr_pipe);
    int tr_exit = WIFEXITED(tr_rc) ? WEXITSTATUS(tr_rc) : -1;
    std::filesystem::remove(tmp_mlir);
    if (tr_exit != 0) {
        std::cerr << "[clifft-mlir-coop] mlir-translate failed\n";
        return {};
    }

    std::ifstream llfile(tmp_ll);
    if (!llfile) return {};
    std::string ll_text((std::istreambuf_iterator<char>(llfile)), std::istreambuf_iterator<char>());
    std::filesystem::remove(tmp_ll);
    std::cerr << "[clifft-mlir-coop] generated " << ll_text.size() << " bytes of LLVM-IR\n";
    return ll_text;
}

std::string generate_mlir_kernel_llvmir_global(const FlattenedProgram& flat,
                                                const std::string& gpu_arch) {
    std::string mlir_text = mlir_emit::emit_mlir_text_global(flat);
    if (mlir_text.empty()) {
        std::cerr << "[clifft-mlir-global] MLIR text generation failed\n";
        return {};
    }

    std::string hash_src = mlir_text + gpu_arch;
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : hash_src) { h ^= c; h *= 1099511628211ULL; }
    char hbuf[20]; snprintf(hbuf, sizeof(hbuf), "%016llx", (unsigned long long)h);
    std::string tmp_mlir = "/tmp/clifft_mlir_global_" + std::string(hbuf) + ".mlir";
    std::string tmp_ll = "/tmp/clifft_mlir_global_" + std::string(hbuf) + ".ll";

    { std::ofstream f(tmp_mlir); if (!f) return {}; f << mlir_text; }

    std::string mlir_opt = mlir_emit::find_mlir_opt();
    std::string mlir_translate = mlir_emit::find_mlir_translate();

    std::ostringstream cmd_opt;
    cmd_opt << "\"" << mlir_opt << "\" --canonicalize --cse --convert-func-to-llvm"
            << " \"" << tmp_mlir << "\" 2>&1";
    FILE* pipe = popen(cmd_opt.str().c_str(), "r");
    std::string opt_out;
    if (!pipe) { std::filesystem::remove(tmp_mlir); return {}; }
    { std::array<char, 65536> buf; while (fgets(buf.data(), (int)buf.size(), pipe)) opt_out += buf.data(); }
    int opt_rc = pclose(pipe);
    int opt_exit = WIFEXITED(opt_rc) ? WEXITSTATUS(opt_rc) : -1;
    if (opt_exit != 0) {
        std::cerr << "[clifft-mlir-global] mlir-opt failed (rc=" << opt_exit << ")\n"
                  << "  output: " << opt_out.substr(0, 2000) << "\n";
        std::filesystem::remove(tmp_mlir);
        return {};
    }
    { std::ofstream f(tmp_mlir); f << opt_out; }

    std::ostringstream cmd_tr;
    cmd_tr << "\"" << mlir_translate << "\" --mlir-to-llvmir \"" << tmp_mlir
           << "\" -o \"" << tmp_ll << "\" 2>&1";
    std::string tr_out;
    FILE* tr_pipe = popen(cmd_tr.str().c_str(), "r");
    if (!tr_pipe) { std::filesystem::remove(tmp_mlir); return {}; }
    { std::array<char, 65536> buf; while (fgets(buf.data(), (int)buf.size(), tr_pipe)) tr_out += buf.data(); }
    int tr_rc = pclose(tr_pipe);
    int tr_exit = WIFEXITED(tr_rc) ? WEXITSTATUS(tr_rc) : -1;
    std::filesystem::remove(tmp_mlir);
    if (tr_exit != 0) {
        std::cerr << "[clifft-mlir-global] mlir-translate failed\n";
        return {};
    }

    std::ifstream llfile(tmp_ll);
    if (!llfile) return {};
    std::string ll_text((std::istreambuf_iterator<char>(llfile)), std::istreambuf_iterator<char>());
    std::filesystem::remove(tmp_ll);
    std::cerr << "[clifft-mlir-global] generated " << ll_text.size() << " bytes of LLVM-IR\n";
    return ll_text;
}

}  // namespace gpu
}  // namespace clifft

#endif  // CLIFFT_ENABLE_MLIR
