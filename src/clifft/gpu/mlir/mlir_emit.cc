// mlir_emit.cc — Textual MLIR (LLVM dialect) emission for quantum circuits
//
// Generates a complete MLIR module containing a single @compiled_mlir_kernel
// function with straight-line LLVM dialect ops. Gate operations are decomposed
// into per-category .inc files under ops/ following the LLVM tablegen pattern.
//
// Architecture (mirrors IREE HAL/Target pattern):
//   mlir_codegen.cc  — pipeline orchestration (mlir-opt, mlir-translate)
//   mlir_emit.cc     — IR generation (this file)
//   ops/*.inc        — per-category gate emission

#ifdef CLIFFT_ENABLE_MLIR

#include "clifft/gpu/mlir/mlir_emit.h"
#include "clifft/gpu/codegen/kernel_codegen.h"
#include "clifft/gpu/gpu_types.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <sys/wait.h>

namespace clifft {
namespace gpu {
namespace mlir_emit {

int run_pipe_command(const std::string& cmd, const std::string& stdin_data,
                     std::string& stdout_out) {
    std::string full = "printf '%s' \"$MLIR_INPUT\" | " + cmd + " 2>&1";
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

std::string find_mlir_opt() {
    const char* llvm = std::getenv("LLVM_PREFIX");
    if (llvm) {
        std::string c = std::string(llvm) + "/bin/mlir-opt";
        if (std::filesystem::exists(c)) return c;
    }
    if (std::filesystem::exists("/shared/jmonsalv/software/modules/llvm/upstream_05082025/bin/mlir-opt"))
        return "/shared/jmonsalv/software/modules/llvm/upstream_05082025/bin/mlir-opt";
    return "mlir-opt";
}

std::string find_mlir_translate() {
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
// SSA state and helpers
// -----------------------------------------------------------------------

namespace {

static int label_counter = 0;
static int ssa_counter = 100;

std::string fresh_label(const std::string& prefix = "bb") {
    return prefix + std::to_string(label_counter++);
}

std::string fresh_ssa() {
    return "%t" + std::to_string(ssa_counter++);
}

const double kInvSqrt2 = 0.70710678118654752440084436210484903928;

std::string emit_const_i64(std::ostringstream& out, uint64_t val) {
    std::string name = fresh_ssa();
    out << "  " << name << " = llvm.mlir.constant(" << val << " : i64) : i64\n";
    return name;
}
std::string emit_const_i32(std::ostringstream& out, int32_t val) {
    std::string name = fresh_ssa();
    out << "  " << name << " = llvm.mlir.constant(" << val << " : i32) : i32\n";
    return name;
}
std::string emit_const_i1(std::ostringstream& out, bool val) {
    std::string name = fresh_ssa();
    out << "  " << name << " = llvm.mlir.constant(" << (val ? "true" : "false") << ") : i1\n";
    return name;
}

// -----------------------------------------------------------------------
// Bit manipulation helpers for Pauli frame tracking
// -----------------------------------------------------------------------

std::string emit_bit_get(std::ostringstream& out,
                         const std::string& words_ptr,
                         const std::string& idx_i32) {
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
}

void emit_bit_xor(std::ostringstream& out,
                   const std::string& words_ptr,
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
}

void emit_bit_set(std::ostringstream& out,
                   const std::string& words_ptr,
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
    std::string nmask = fresh_ssa();
    out << "  " << nmask << " = llvm.xor " << mask << ", %cminus1_i64 : i64\n";
    std::string cleared = fresh_ssa();
    out << "  " << cleared << " = llvm.and " << word << ", " << nmask << " : i64\n";
    std::string val64 = fresh_ssa();
    out << "  " << val64 << " = llvm.zext " << val_i1 << " : i1 to i64\n";
    std::string bset = fresh_ssa();
    out << "  " << bset << " = llvm.shl " << val64 << ", " << shift << " : i64\n";
    std::string new_word = fresh_ssa();
    out << "  " << new_word << " = llvm.or " << cleared << ", " << bset << " : i64\n";
    out << "  llvm.store " << new_word << ", " << wptr << " : i64, !llvm.ptr\n";
}

// -----------------------------------------------------------------------
// Amplitude array helpers
// -----------------------------------------------------------------------

std::string emit_scatter_bits_1(std::ostringstream& out,
                                 const std::string& val_i64,
                                 const std::string& pos_i64) {
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
}

std::string emit_load_v(std::ostringstream& out, const std::string& idx_i64) {
    std::string vp = fresh_ssa();
    out << "  " << vp << " = llvm.getelementptr inbounds %v_ptr[" << idx_i64
        << "] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(f32, f32)>\n";
    std::string vc = fresh_ssa();
    out << "  " << vc << " = llvm.load " << vp << " : !llvm.ptr -> !llvm.struct<(f32, f32)>\n";
    return vc;
}

void emit_store_v(std::ostringstream& out,
                   const std::string& idx_i64, const std::string& val) {
    std::string vp = fresh_ssa();
    out << "  " << vp << " = llvm.getelementptr inbounds %v_ptr[" << idx_i64
        << "] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(f32, f32)>\n";
    out << "  llvm.store " << val << ", " << vp
        << " : !llvm.struct<(f32, f32)>, !llvm.ptr\n";
}

// -----------------------------------------------------------------------
// Complex arithmetic helpers
// -----------------------------------------------------------------------

std::string emit_cadd(std::ostringstream& out,
                       const std::string& a, const std::string& b_val) {
    std::string are = fresh_ssa(), aim = fresh_ssa(), bre = fresh_ssa(), bim = fresh_ssa();
    out << "  " << are << " = llvm.extractvalue " << a << "[0] : !llvm.struct<(f32, f32)>\n";
    out << "  " << aim << " = llvm.extractvalue " << a << "[1] : !llvm.struct<(f32, f32)>\n";
    out << "  " << bre << " = llvm.extractvalue " << b_val << "[0] : !llvm.struct<(f32, f32)>\n";
    out << "  " << bim << " = llvm.extractvalue " << b_val << "[1] : !llvm.struct<(f32, f32)>\n";
    std::string re = fresh_ssa(), im = fresh_ssa();
    out << "  " << re << " = llvm.fadd " << are << ", " << bre << " : f32\n";
    out << "  " << im << " = llvm.fadd " << aim << ", " << bim << " : f32\n";
    std::string r0 = fresh_ssa(), r1 = fresh_ssa(), res = fresh_ssa();
    out << "  " << r0 << " = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
    out << "  " << r1 << " = llvm.insertvalue " << re << ", " << r0 << "[0] : !llvm.struct<(f32, f32)>\n";
    out << "  " << res << " = llvm.insertvalue " << im << ", " << r1 << "[1] : !llvm.struct<(f32, f32)>\n";
    return res;
}

std::string emit_csub(std::ostringstream& out,
                       const std::string& a, const std::string& b_val) {
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
}

std::string emit_cscale_f64(std::ostringstream& out,
                             const std::string& a, double scale) {
    char sbuf[64];
    snprintf(sbuf, sizeof(sbuf), "%.17e", scale);
    std::string sf64 = fresh_ssa();
    out << "  " << sf64 << " = llvm.mlir.constant(" << sbuf << " : f64) : f64\n";
    std::string sf32 = fresh_ssa();
    out << "  " << sf32 << " = llvm.fptrunc " << sf64 << " : f64 to f32\n";
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
}

std::string emit_cmul_const(std::ostringstream& out,
                             const std::string& a,
                             double phase_re, double phase_im) {
    char rbuf[64], ibuf[64];
    snprintf(rbuf, sizeof(rbuf), "%.17e", (float)phase_re);
    snprintf(ibuf, sizeof(ibuf), "%.17e", (float)phase_im);
    std::string pre_f = fresh_ssa(), pim_f = fresh_ssa();
    out << "  " << pre_f << " = llvm.mlir.constant(" << rbuf << " : f32) : f32\n";
    out << "  " << pim_f << " = llvm.mlir.constant(" << ibuf << " : f32) : f32\n";
    std::string are = fresh_ssa(), aim = fresh_ssa();
    out << "  " << are << " = llvm.extractvalue " << a << "[0] : !llvm.struct<(f32, f32)>\n";
    out << "  " << aim << " = llvm.extractvalue " << a << "[1] : !llvm.struct<(f32, f32)>\n";
    std::string t0 = fresh_ssa(), t1 = fresh_ssa(), re = fresh_ssa();
    out << "  " << t0 << " = llvm.fmul " << are << ", " << pre_f << " : f32\n";
    out << "  " << t1 << " = llvm.fmul " << aim << ", " << pim_f << " : f32\n";
    out << "  " << re << " = llvm.fsub " << t0 << ", " << t1 << " : f32\n";
    std::string t2 = fresh_ssa(), t3 = fresh_ssa(), im = fresh_ssa();
    out << "  " << t2 << " = llvm.fmul " << are << ", " << pim_f << " : f32\n";
    out << "  " << t3 << " = llvm.fmul " << aim << ", " << pre_f << " : f32\n";
    out << "  " << im << " = llvm.fadd " << t2 << ", " << t3 << " : f32\n";
    std::string r0 = fresh_ssa(), r1 = fresh_ssa(), res = fresh_ssa();
    out << "  " << r0 << " = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
    out << "  " << r1 << " = llvm.insertvalue " << re << ", " << r0 << "[0] : !llvm.struct<(f32, f32)>\n";
    out << "  " << res << " = llvm.insertvalue " << im << ", " << r1 << "[1] : !llvm.struct<(f32, f32)>\n";
    return res;
}

// -----------------------------------------------------------------------
// Gate-level emitters (used by ops/*.inc)
// -----------------------------------------------------------------------

void emit_array_h_static(std::ostringstream& out, uint32_t axis) {
    out << "  // array_h on axis " << axis << "\n";
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

    std::string hdr = fresh_label("h_hdr");
    std::string body = fresh_label("h_body");
    std::string exit = fresh_label("h_exit");
    out << "  llvm.br ^" << hdr << "(%c0_i64 : i64)\n";
    out << "^" << hdr << "(%h_i: i64):\n";
    std::string cond = fresh_ssa();
    out << "  " << cond << " = llvm.icmp \"ult\" %h_i, " << iters << " : i64\n";
    out << "  llvm.cond_br " << cond << ", ^" << body << ", ^" << exit << "\n";
    out << "^" << body << ":\n";

    std::string idx0 = emit_scatter_bits_1(out, "%h_i", axis_val);
    std::string idx1 = fresh_ssa();
    out << "  " << idx1 << " = llvm.or " << idx0 << ", " << axis_bit << " : i64\n";

    std::string va = emit_load_v(out, idx0);
    std::string vb = emit_load_v(out, idx1);
    std::string sum = emit_cadd(out, va, vb);
    std::string diff = emit_csub(out, va, vb);
    emit_store_v(out, idx0, emit_cscale_f64(out, sum, kInvSqrt2));
    emit_store_v(out, idx1, emit_cscale_f64(out, diff, kInvSqrt2));

    std::string i_next = fresh_ssa();
    out << "  " << i_next << " = llvm.add %h_i, %c1_i64 : i64\n";
    out << "  llvm.br ^" << hdr << "(" << i_next << " : i64)\n";
    out << "^" << exit << ":\n";
    (void)inv_sq2;
}

void emit_array_cnot_static(std::ostringstream& out, uint32_t ctrl, uint32_t tgt) {
    out << "  // array_cnot ctrl=" << ctrl << " tgt=" << tgt << "\n";
    std::string ak = fresh_ssa(), ak64 = fresh_ssa(), ak_m2 = fresh_ssa(), iters = fresh_ssa();
    out << "  " << ak << " = llvm.load %active_k_ptr : !llvm.ptr -> i32\n";
    out << "  " << ak64 << " = llvm.zext " << ak << " : i32 to i64\n";
    out << "  " << ak_m2 << " = llvm.sub " << ak64 << ", %c1_i64 : i64\n";
    std::string ak_m2b = fresh_ssa();
    std::string c2_i64 = emit_const_i64(out, 2);
    out << "  " << ak_m2b << " = llvm.sub " << ak64 << ", " << c2_i64 << " : i64\n";
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

    uint32_t lo = std::min(ctrl, tgt), hi = std::max(ctrl, tgt);
    char lobuf[32], hibuf[32];
    snprintf(lobuf, sizeof(lobuf), "%u", lo);
    snprintf(hibuf, sizeof(hibuf), "%u", hi);
    std::string lo64 = fresh_ssa(), hi64 = fresh_ssa();
    out << "  " << lo64 << " = llvm.mlir.constant(" << lobuf << " : i64) : i64\n";
    out << "  " << hi64 << " = llvm.mlir.constant(" << hibuf << " : i64) : i64\n";
    std::string s1 = emit_scatter_bits_1(out, "%cn_i", lo64);
    std::string base_pre = emit_scatter_bits_1(out, s1, hi64);
    std::string base = fresh_ssa();
    out << "  " << base << " = llvm.or " << base_pre << ", " << c_bit << " : i64\n";
    std::string base_t = fresh_ssa();
    out << "  " << base_t << " = llvm.or " << base << ", " << t_bit << " : i64\n";

    std::string va = emit_load_v(out, base);
    std::string vb = emit_load_v(out, base_t);
    emit_store_v(out, base, vb);
    emit_store_v(out, base_t, va);

    std::string i_next = fresh_ssa();
    out << "  " << i_next << " = llvm.add %cn_i, %c1_i64 : i64\n";
    out << "  llvm.br ^" << hdr << "(" << i_next << " : i64)\n";
    out << "^" << exit_lbl << ":\n";
    (void)ak_m2;
}

void emit_apply_phase_static(std::ostringstream& out, uint32_t axis,
                              double phs_re, double phs_im) {
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
        << emit_scatter_bits_1(out, "%ph_i", axis64) << ", " << axis_bit << " : i64\n";
    std::string vc = emit_load_v(out, idx);
    emit_store_v(out, idx, emit_cmul_const(out, vc, phs_re, phs_im));
    std::string i_next = fresh_ssa();
    out << "  " << i_next << " = llvm.add %ph_i, %c1_i64 : i64\n";
    out << "  llvm.br ^" << hdr << "(" << i_next << " : i64)\n";
    out << "^" << exit_lbl << ":\n";
}

// -----------------------------------------------------------------------
// GPU intrinsic helpers for coop/global kernels
// -----------------------------------------------------------------------

void emit_gpu_intrinsic_decls(std::ostringstream& out) {
    out << "llvm.func @llvm.amdgcn.workitem.id.x() -> i32\n";
    out << "llvm.func @llvm.amdgcn.workgroup.id.x() -> i32\n";
    out << "llvm.func @llvm.amdgcn.s.barrier() -> ()\n\n";
}

std::string emit_tidx(std::ostringstream& out) {
    std::string r = fresh_ssa();
    out << "  " << r << " = llvm.call @llvm.amdgcn.workitem.id.x() : () -> i32\n";
    return r;
}

std::string emit_bidx(std::ostringstream& out) {
    std::string r = fresh_ssa();
    out << "  " << r << " = llvm.call @llvm.amdgcn.workgroup.id.x() : () -> i32\n";
    return r;
}

void emit_barrier(std::ostringstream& out) {
    out << "  llvm.call @llvm.amdgcn.s.barrier() : () -> ()\n";
}

void emit_lds_global(std::ostringstream& out,
                     const std::string& name,
                     const std::string& elem_type,
                     uint32_t count) {
    out << "llvm.mlir.global external @" << name
        << "() {addr_space = 3 : i32} : !llvm.array<"
        << count << " x " << elem_type << ">\n";
}

std::string emit_lds_base_ptr(std::ostringstream& out,
                               const std::string& name,
                               const std::string& elem_type,
                               uint32_t count) {
    std::string ptr = fresh_ssa();
    out << "  " << ptr << " = llvm.mlir.addressof @" << name
        << " : !llvm.ptr<3>\n";
    std::string generic = fresh_ssa();
    out << "  " << generic << " = llvm.addrspacecast " << ptr
        << " : !llvm.ptr<3> to !llvm.ptr\n";
    return generic;
}

std::string emit_atomic_add_i64(std::ostringstream& out,
                                 const std::string& ptr,
                                 const std::string& val) {
    std::string old = fresh_ssa();
    out << "  " << old << " = llvm.atomicrmw add " << ptr << ", " << val
        << " monotonic : !llvm.ptr, i64\n";
    return old;
}

std::string emit_atomic_cmpxchg_i64(std::ostringstream& out,
                                     const std::string& ptr,
                                     const std::string& cmp,
                                     const std::string& new_val) {
    std::string res = fresh_ssa();
    out << "  " << res << " = llvm.cmpxchg " << ptr << ", " << cmp
        << ", " << new_val
        << " monotonic monotonic : !llvm.ptr, i64\n";
    std::string old_val = fresh_ssa();
    out << "  " << old_val << " = llvm.extractvalue " << res
        << "[0] : !llvm.struct<(i64, i1)>\n";
    std::string success = fresh_ssa();
    out << "  " << success << " = llvm.extractvalue " << res
        << "[1] : !llvm.struct<(i64, i1)>\n";
    return old_val;
}

std::string emit_xcd_id(std::ostringstream& out) {
    std::string raw = fresh_ssa();
    out << "  " << raw << " = llvm.inline_asm "
        << "\"s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)\", \"=s\" "
        << ": () -> i32\n";
    std::string kNumXCDs = fresh_ssa();
    out << "  " << kNumXCDs << " = llvm.mlir.constant(8 : i32) : i32\n";
    std::string id = fresh_ssa();
    out << "  " << id << " = llvm.urem " << raw << ", " << kNumXCDs << " : i32\n";
    return id;
}

void emit_store_complex64_packed(std::ostringstream& out,
                                  const std::string& v_ptr,
                                  const std::string& idx_i64,
                                  const std::string& complex_val) {
    std::string vp = fresh_ssa();
    out << "  " << vp << " = llvm.getelementptr inbounds " << v_ptr << "["
        << idx_i64 << "] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(f32, f32)>\n";
    std::string re = fresh_ssa(), im = fresh_ssa();
    out << "  " << re << " = llvm.extractvalue " << complex_val
        << "[0] : !llvm.struct<(f32, f32)>\n";
    out << "  " << im << " = llvm.extractvalue " << complex_val
        << "[1] : !llvm.struct<(f32, f32)>\n";
    std::string re_i32 = fresh_ssa(), im_i32 = fresh_ssa();
    out << "  " << re_i32 << " = llvm.bitcast " << re << " : f32 to i32\n";
    out << "  " << im_i32 << " = llvm.bitcast " << im << " : f32 to i32\n";
    std::string re_i64 = fresh_ssa(), im_i64 = fresh_ssa();
    out << "  " << re_i64 << " = llvm.zext " << re_i32 << " : i32 to i64\n";
    out << "  " << im_i64 << " = llvm.zext " << im_i32 << " : i32 to i64\n";
    std::string im_sh = fresh_ssa();
    std::string c32 = emit_const_i64(out, 32);
    out << "  " << im_sh << " = llvm.shl " << im_i64 << ", " << c32 << " : i64\n";
    std::string packed = fresh_ssa();
    out << "  " << packed << " = llvm.or " << re_i64 << ", " << im_sh << " : i64\n";
    std::string cast_ptr = fresh_ssa();
    out << "  " << cast_ptr << " = llvm.bitcast " << vp << " : !llvm.ptr to !llvm.ptr\n";
    out << "  llvm.store " << packed << ", " << cast_ptr << " : i64, !llvm.ptr\n";
}

std::string emit_cnorm(std::ostringstream& out, const std::string& c) {
    std::string re = fresh_ssa(), im = fresh_ssa();
    out << "  " << re << " = llvm.extractvalue " << c << "[0] : !llvm.struct<(f32, f32)>\n";
    out << "  " << im << " = llvm.extractvalue " << c << "[1] : !llvm.struct<(f32, f32)>\n";
    std::string re2 = fresh_ssa(), im2 = fresh_ssa();
    out << "  " << re2 << " = llvm.fmul " << re << ", " << re << " : f32\n";
    out << "  " << im2 << " = llvm.fmul " << im << ", " << im << " : f32\n";
    std::string norm_f32 = fresh_ssa();
    out << "  " << norm_f32 << " = llvm.fadd " << re2 << ", " << im2 << " : f32\n";
    std::string norm_f64 = fresh_ssa();
    out << "  " << norm_f64 << " = llvm.fpext " << norm_f32 << " : f32 to f64\n";
    return norm_f64;
}

std::string emit_scatter_bits_2(std::ostringstream& out,
                                 const std::string& val_i64,
                                 const std::string& pos1_i64,
                                 const std::string& pos2_i64) {
    std::string lo = fresh_ssa(), hi = fresh_ssa();
    std::string cmp = fresh_ssa();
    out << "  " << cmp << " = llvm.icmp \"ult\" " << pos1_i64 << ", " << pos2_i64 << " : i64\n";
    out << "  " << lo << " = llvm.select " << cmp << ", " << pos1_i64 << ", " << pos2_i64 << " : i1, i64\n";
    out << "  " << hi << " = llvm.select " << cmp << ", " << pos2_i64 << ", " << pos1_i64 << " : i1, i64\n";
    std::string s1 = emit_scatter_bits_1(out, val_i64, lo);
    std::string s2 = emit_scatter_bits_1(out, s1, hi);
    return s2;
}

// -----------------------------------------------------------------------
// RNG helpers — xoshiro256++ PRNG emitted as inline MLIR
// -----------------------------------------------------------------------

void emit_rng_state_alloca(std::ostringstream& out) {
    out << "  %c4_i32 = llvm.mlir.constant(4 : i32) : i32\n";
    out << "  %rng_ptr_p5 = llvm.alloca %c4_i32 x i64 : (i32) -> !llvm.ptr<5>\n";
    out << "  %rng_ptr = llvm.addrspacecast %rng_ptr_p5 : !llvm.ptr<5> to !llvm.ptr\n";
}

void emit_splitmix64(std::ostringstream& out,
                      const std::string& state_ptr,
                      const std::string& result_name) {
    std::string z = fresh_ssa();
    out << "  " << z << " = llvm.load " << state_ptr << " : !llvm.ptr -> i64\n";
    std::string k1 = emit_const_i64(out, 11400714819323198485ULL);
    std::string inc = fresh_ssa();
    out << "  " << inc << " = llvm.add " << z << ", " << k1 << " : i64\n";
    out << "  llvm.store " << inc << ", " << state_ptr << " : i64, !llvm.ptr\n";
    std::string c30 = emit_const_i64(out, 30);
    std::string z1 = fresh_ssa();
    out << "  " << z1 << " = llvm.lshr " << inc << ", " << c30 << " : i64\n";
    std::string z2 = fresh_ssa();
    out << "  " << z2 << " = llvm.xor " << inc << ", " << z1 << " : i64\n";
    std::string k2 = emit_const_i64(out, 13787848793156543929ULL);
    std::string z3 = fresh_ssa();
    out << "  " << z3 << " = llvm.mul " << z2 << ", " << k2 << " : i64\n";
    std::string c27 = emit_const_i64(out, 27);
    std::string z4 = fresh_ssa();
    out << "  " << z4 << " = llvm.lshr " << z3 << ", " << c27 << " : i64\n";
    std::string z5 = fresh_ssa();
    out << "  " << z5 << " = llvm.xor " << z3 << ", " << z4 << " : i64\n";
    std::string k3 = emit_const_i64(out, 10723151780598845931ULL);
    std::string z6 = fresh_ssa();
    out << "  " << z6 << " = llvm.mul " << z5 << ", " << k3 << " : i64\n";
    std::string c31 = emit_const_i64(out, 31);
    std::string z7 = fresh_ssa();
    out << "  " << z7 << " = llvm.lshr " << z6 << ", " << c31 << " : i64\n";
    out << "  " << result_name << " = llvm.xor " << z6 << ", " << z7 << " : i64\n";
}

void emit_rng_seed(std::ostringstream& out,
                    const std::string& seed_val,
                    const std::string& shot_id) {
    // z = seed ^ (0x9e3779b97f4a7c15 * (shot_id + 1))
    std::string shot_p1 = fresh_ssa();
    out << "  " << shot_p1 << " = llvm.add " << shot_id << ", %c1_i64 : i64\n";
    std::string k_seed = emit_const_i64(out, 11400714819323198485ULL);
    std::string mult = fresh_ssa();
    out << "  " << mult << " = llvm.mul " << shot_p1 << ", " << k_seed << " : i64\n";
    std::string z_init = fresh_ssa();
    out << "  " << z_init << " = llvm.xor " << seed_val << ", " << mult << " : i64\n";
    // sm_tmp already allocated in entry block
    out << "  llvm.store " << z_init << ", %sm_tmp : i64, !llvm.ptr\n";
    // Generate 4 state words
    for (int i = 0; i < 4; ++i) {
        std::string r = fresh_ssa();
        emit_splitmix64(out, "%sm_tmp", r);
        std::string idx = fresh_ssa();
        out << "  " << idx << " = llvm.mlir.constant(" << i << " : i64) : i64\n";
        std::string ptr = fresh_ssa();
        out << "  " << ptr << " = llvm.getelementptr inbounds %rng_ptr["
            << idx << "] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
        out << "  llvm.store " << r << ", " << ptr << " : i64, !llvm.ptr\n";
    }
}

std::string emit_rng_next(std::ostringstream& out) {
    // Load s[0..3]
    std::string s[4], sp[4];
    for (int i = 0; i < 4; ++i) {
        std::string idx = fresh_ssa();
        out << "  " << idx << " = llvm.mlir.constant(" << i << " : i64) : i64\n";
        sp[i] = fresh_ssa();
        out << "  " << sp[i] << " = llvm.getelementptr inbounds %rng_ptr["
            << idx << "] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
        s[i] = fresh_ssa();
        out << "  " << s[i] << " = llvm.load " << sp[i] << " : !llvm.ptr -> i64\n";
    }
    // result = rotl64(s[0] + s[3], 23) + s[0]
    std::string sum03 = fresh_ssa();
    out << "  " << sum03 << " = llvm.add " << s[0] << ", " << s[3] << " : i64\n";
    std::string c23 = emit_const_i64(out, 23);
    std::string c41 = emit_const_i64(out, 41);
    std::string shl23 = fresh_ssa(), shr41 = fresh_ssa(), rotl = fresh_ssa();
    out << "  " << shl23 << " = llvm.shl " << sum03 << ", " << c23 << " : i64\n";
    out << "  " << shr41 << " = llvm.lshr " << sum03 << ", " << c41 << " : i64\n";
    out << "  " << rotl << " = llvm.or " << shl23 << ", " << shr41 << " : i64\n";
    std::string result = fresh_ssa();
    out << "  " << result << " = llvm.add " << rotl << ", " << s[0] << " : i64\n";
    // Update state: t = s[1] << 17
    std::string t = fresh_ssa();
    std::string c17 = emit_const_i64(out, 17);
    out << "  " << t << " = llvm.shl " << s[1] << ", " << c17 << " : i64\n";
    // s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]
    std::string ns2 = fresh_ssa(), ns3 = fresh_ssa(), ns1 = fresh_ssa(), ns0 = fresh_ssa();
    out << "  " << ns2 << " = llvm.xor " << s[2] << ", " << s[0] << " : i64\n";
    out << "  " << ns3 << " = llvm.xor " << s[3] << ", " << s[1] << " : i64\n";
    out << "  " << ns1 << " = llvm.xor " << s[1] << ", " << ns2 << " : i64\n";
    out << "  " << ns0 << " = llvm.xor " << s[0] << ", " << ns3 << " : i64\n";
    // s[2] ^= t
    std::string ns2t = fresh_ssa();
    out << "  " << ns2t << " = llvm.xor " << ns2 << ", " << t << " : i64\n";
    // s[3] = rotl64(s[3]_new, 45)
    std::string shl45 = fresh_ssa(), shr19 = fresh_ssa(), ns3r = fresh_ssa();
    std::string c45 = emit_const_i64(out, 45);
    std::string c19 = emit_const_i64(out, 19);
    out << "  " << shl45 << " = llvm.shl " << ns3 << ", " << c45 << " : i64\n";
    out << "  " << shr19 << " = llvm.lshr " << ns3 << ", " << c19 << " : i64\n";
    out << "  " << ns3r << " = llvm.or " << shl45 << ", " << shr19 << " : i64\n";
    // Store back
    out << "  llvm.store " << ns0 << ", " << sp[0] << " : i64, !llvm.ptr\n";
    out << "  llvm.store " << ns1 << ", " << sp[1] << " : i64, !llvm.ptr\n";
    out << "  llvm.store " << ns2t << ", " << sp[2] << " : i64, !llvm.ptr\n";
    out << "  llvm.store " << ns3r << ", " << sp[3] << " : i64, !llvm.ptr\n";
    return result;
}

std::string emit_rng_uniform(std::ostringstream& out) {
    std::string raw = emit_rng_next(out);
    std::string shifted = fresh_ssa();
    std::string c11 = emit_const_i64(out, 11);
    out << "  " << shifted << " = llvm.lshr " << raw << ", " << c11 << " : i64\n";
    std::string as_f64 = fresh_ssa();
    out << "  " << as_f64 << " = llvm.uitofp " << shifted << " : i64 to f64\n";
    std::string scale = fresh_ssa();
    out << "  " << scale << " = llvm.mlir.constant(1.1102230246251565e-16 : f64) : f64\n";
    std::string uniform = fresh_ssa();
    out << "  " << uniform << " = llvm.fmul " << as_f64 << ", " << scale << " : f64\n";
    return uniform;
}

void emit_meas_dormant_random(std::ostringstream& out,
                               uint32_t axis, uint32_t classical_idx, bool sign) {
    std::string u = emit_rng_uniform(out);
    std::string half = fresh_ssa();
    out << "  " << half << " = llvm.mlir.constant(0.5 : f64) : f64\n";
    std::string cmp = fresh_ssa();
    out << "  " << cmp << " = llvm.fcmp \"olt\" " << u << ", " << half << " : f64\n";
    std::string m_abs_i8 = fresh_ssa();
    out << "  " << m_abs_i8 << " = llvm.zext " << cmp << " : i1 to i8\n";
    // Note: cmp is true when u < 0.5, meaning m_abs = 1 when u >= 0.5
    // Actually: rng.uniform() < 0.5 ? 0 : 1 means m_abs=0 when u<0.5
    // So cmp = (u < 0.5), m_abs = cmp ? 0 : 1 = !cmp
    std::string true_i1 = emit_const_i1(out, true);
    std::string not_cmp = fresh_ssa();
    out << "  " << not_cmp << " = llvm.xor " << cmp << ", " << true_i1 << " : i1\n";
    std::string m_abs = fresh_ssa();
    out << "  " << m_abs << " = llvm.zext " << not_cmp << " : i1 to i8\n";

    // bit_set(px, axis, m_abs != 0) → bit_set(px, axis, !cmp)
    char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", axis);
    std::string axis_i32 = fresh_ssa();
    out << "  " << axis_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
    emit_bit_set(out, "%px_ptr", axis_i32, not_cmp);
    // bit_set(pz, axis, false)
    std::string false_i1 = fresh_ssa();
    out << "  " << false_i1 << " = llvm.mlir.constant(false) : i1\n";
    emit_bit_set(out, "%pz_ptr", axis_i32, false_i1);
    // meas[classical_idx] = m_abs ^ sign
    std::string meas_val = m_abs;
    if (sign) {
        std::string xored = fresh_ssa();
        out << "  " << xored << " = llvm.xor " << m_abs << ", %c1_i8 : i8\n";
        meas_val = xored;
    }
    char cidx[32]; snprintf(cidx, sizeof(cidx), "%u", classical_idx);
    std::string cidx_i64 = fresh_ssa();
    out << "  " << cidx_i64 << " = llvm.mlir.constant(" << cidx << " : i64) : i64\n";
    std::string meas_p = fresh_ssa();
    out << "  " << meas_p << " = llvm.getelementptr inbounds %meas_ptr["
        << cidx_i64 << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
    out << "  llvm.store " << meas_val << ", " << meas_p << " : i8, !llvm.ptr\n";
}

void emit_meas_active_diagonal(std::ostringstream& out,
                                uint32_t axis, uint32_t classical_idx, bool sign) {
    // half = 1 << (active_k - 1)
    std::string ak = fresh_ssa();
    out << "  " << ak << " = llvm.load %active_k_ptr : !llvm.ptr -> i32\n";
    std::string ak64 = fresh_ssa();
    out << "  " << ak64 << " = llvm.zext " << ak << " : i32 to i64\n";
    std::string ak_m1 = fresh_ssa();
    out << "  " << ak_m1 << " = llvm.sub " << ak64 << ", %c1_i64 : i64\n";
    std::string half = fresh_ssa();
    out << "  " << half << " = llvm.shl %c1_i64, " << ak_m1 << " : i64\n";

    // Read px bit at axis
    char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", axis);
    std::string axis_i32 = fresh_ssa();
    out << "  " << axis_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
    std::string px_bit = emit_bit_get(out, "%px_ptr", axis_i32);

    // Sum probabilities with f64 block argument accumulators
    std::string p_hdr2 = fresh_label("psum_hdr");
    std::string p_body2 = fresh_label("psum_body");
    std::string p_done2 = fresh_label("psum_done");
    std::string f0 = fresh_ssa();
    out << "  " << f0 << " = llvm.mlir.constant(0.0 : f64) : f64\n";
    out << "  llvm.br ^" << p_hdr2 << "(%c0_i64, " << f0 << ", " << f0 << " : i64, f64, f64)\n";
    out << "^" << p_hdr2 << "(%psum_i: i64, %psum_p0: f64, %psum_p1: f64):\n";
    std::string pcond = fresh_ssa();
    out << "  " << pcond << " = llvm.icmp \"ult\" %psum_i, " << half << " : i64\n";
    out << "  llvm.cond_br " << pcond << ", ^" << p_body2 << ", ^" << p_done2 << "\n";
    out << "^" << p_body2 << ":\n";
    // Load v[i] and v[i + half]
    std::string vi = emit_load_v(out, "%psum_i");
    std::string idx_hi = fresh_ssa();
    out << "  " << idx_hi << " = llvm.add %psum_i, " << half << " : i64\n";
    std::string vh = emit_load_v(out, idx_hi);
    std::string n0 = emit_cnorm(out, vi);
    std::string n1 = emit_cnorm(out, vh);
    std::string new_p0 = fresh_ssa(), new_p1 = fresh_ssa();
    out << "  " << new_p0 << " = llvm.fadd %psum_p0, " << n0 << " : f64\n";
    out << "  " << new_p1 << " = llvm.fadd %psum_p1, " << n1 << " : f64\n";
    std::string next_i = fresh_ssa();
    out << "  " << next_i << " = llvm.add %psum_i, %c1_i64 : i64\n";
    out << "  llvm.br ^" << p_hdr2 << "(" << next_i << ", " << new_p0 << ", " << new_p1 << " : i64, f64, f64)\n";
    out << "^" << p_done2 << ":\n";

    // sample_branch: if prob1 <= eps return 0; if prob0 <= eps return 1; else rng
    std::string total = fresh_ssa();
    out << "  " << total << " = llvm.fadd %psum_p0, %psum_p1 : f64\n";
    std::string eps_k = fresh_ssa();
    out << "  " << eps_k << " = llvm.mlir.constant(1.0e-300 : f64) : f64\n";
    std::string eps = fresh_ssa();
    out << "  " << eps << " = llvm.fmul " << eps_k << ", " << total << " : f64\n";
    // Check if p1 <= eps → branch = 0
    std::string p1_small = fresh_ssa();
    out << "  " << p1_small << " = llvm.fcmp \"ole\" %psum_p1, " << eps << " : f64\n";
    // Check if p0 <= eps → branch = 1
    std::string p0_small = fresh_ssa();
    out << "  " << p0_small << " = llvm.fcmp \"ole\" %psum_p0, " << eps << " : f64\n";
    // RNG sample
    std::string u = emit_rng_uniform(out);
    std::string threshold = fresh_ssa();
    out << "  " << threshold << " = llvm.fmul " << u << ", " << total << " : f64\n";
    std::string rng_b = fresh_ssa();
    out << "  " << rng_b << " = llvm.fcmp \"oge\" " << threshold << ", %psum_p0 : f64\n";
    // Final branch: p1_small ? 0 : (p0_small ? 1 : rng_b)
    std::string b_sel1 = fresh_ssa();
    out << "  " << b_sel1 << " = llvm.select " << p0_small << ", "
        << emit_const_i1(out, true) << ", " << rng_b << " : i1, i1\n";
    std::string b_final = fresh_ssa();
    out << "  " << b_final << " = llvm.select " << p1_small << ", "
        << emit_const_i1(out, false) << ", " << b_sel1 << " : i1, i1\n";

    // m_abs = b ^ px
    std::string b_i8 = fresh_ssa();
    out << "  " << b_i8 << " = llvm.zext " << b_final << " : i1 to i8\n";
    std::string px_i8 = fresh_ssa();
    out << "  " << px_i8 << " = llvm.zext " << px_bit << " : i1 to i8\n";
    std::string m_abs = fresh_ssa();
    out << "  " << m_abs << " = llvm.xor " << b_i8 << ", " << px_i8 << " : i8\n";

    // meas[classical_idx] = m_abs ^ sign
    std::string meas_val = m_abs;
    if (sign) {
        std::string xored = fresh_ssa();
        out << "  " << xored << " = llvm.xor " << m_abs << ", %c1_i8 : i8\n";
        meas_val = xored;
    }
    char cidx[32]; snprintf(cidx, sizeof(cidx), "%u", classical_idx);
    std::string cidx_i64 = emit_const_i64(out, classical_idx);
    std::string meas_p = fresh_ssa();
    out << "  " << meas_p << " = llvm.getelementptr inbounds %meas_ptr["
        << cidx_i64 << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
    out << "  llvm.store " << meas_val << ", " << meas_p << " : i8, !llvm.ptr\n";

    // If b != 0: copy v[i+half] → v[i] for i in [0, half)
    std::string lbl_copy = fresh_label("copy");
    std::string lbl_nocopy = fresh_label("nocopy");
    out << "  llvm.cond_br " << b_final << ", ^" << lbl_copy << ", ^" << lbl_nocopy << "\n";
    out << "^" << lbl_copy << ":\n";
    std::string cp_hdr = fresh_label("cp_hdr"), cp_body = fresh_label("cp_body"), cp_done = fresh_label("cp_done");
    out << "  llvm.br ^" << cp_hdr << "(%c0_i64 : i64)\n";
    out << "^" << cp_hdr << "(%cp_i: i64):\n";
    std::string cp_cond = fresh_ssa();
    out << "  " << cp_cond << " = llvm.icmp \"ult\" %cp_i, " << half << " : i64\n";
    out << "  llvm.cond_br " << cp_cond << ", ^" << cp_body << ", ^" << cp_done << "\n";
    out << "^" << cp_body << ":\n";
    std::string cp_src = fresh_ssa();
    out << "  " << cp_src << " = llvm.add %cp_i, " << half << " : i64\n";
    std::string cp_val = emit_load_v(out, cp_src);
    emit_store_v(out, "%cp_i", cp_val);
    std::string cp_next = fresh_ssa();
    out << "  " << cp_next << " = llvm.add %cp_i, %c1_i64 : i64\n";
    out << "  llvm.br ^" << cp_hdr << "(" << cp_next << " : i64)\n";
    out << "^" << cp_done << ":\n";
    out << "  llvm.br ^" << lbl_nocopy << "\n";
    out << "^" << lbl_nocopy << ":\n";

    // active_k--
    std::string new_ak = fresh_ssa();
    out << "  " << new_ak << " = llvm.sub " << ak << ", %c1_i32 : i32\n";
    out << "  llvm.store " << new_ak << ", %active_k_ptr : i32, !llvm.ptr\n";
    // bit_set(px, axis, m_abs != 0)
    std::string m_ne0 = fresh_ssa();
    out << "  " << m_ne0 << " = llvm.icmp \"ne\" " << m_abs << ", %c0_i8 : i8\n";
    emit_bit_set(out, "%px_ptr", axis_i32, m_ne0);
    std::string false_val = emit_const_i1(out, false);
    emit_bit_set(out, "%pz_ptr", axis_i32, false_val);
}

void emit_expand_plain(std::ostringstream& out) {
    std::string ak = fresh_ssa();
    out << "  " << ak << " = llvm.load %active_k_ptr : !llvm.ptr -> i32\n";
    std::string ak64 = fresh_ssa();
    out << "  " << ak64 << " = llvm.zext " << ak << " : i32 to i64\n";
    std::string half = fresh_ssa();
    out << "  " << half << " = llvm.shl %c1_i64, " << ak64 << " : i64\n";

    // Copy v[i] → v[i+half] for i in [0, half)
    std::string ex_hdr = fresh_label("ex_hdr"), ex_body = fresh_label("ex_body"), ex_done = fresh_label("ex_done");
    out << "  llvm.br ^" << ex_hdr << "(%c0_i64 : i64)\n";
    out << "^" << ex_hdr << "(%ex_i: i64):\n";
    std::string ex_cond = fresh_ssa();
    out << "  " << ex_cond << " = llvm.icmp \"ult\" %ex_i, " << half << " : i64\n";
    out << "  llvm.cond_br " << ex_cond << ", ^" << ex_body << ", ^" << ex_done << "\n";
    out << "^" << ex_body << ":\n";
    std::string src = emit_load_v(out, "%ex_i");
    std::string dst_idx = fresh_ssa();
    out << "  " << dst_idx << " = llvm.add %ex_i, " << half << " : i64\n";
    emit_store_v(out, dst_idx, src);
    std::string ex_next = fresh_ssa();
    out << "  " << ex_next << " = llvm.add %ex_i, %c1_i64 : i64\n";
    out << "  llvm.br ^" << ex_hdr << "(" << ex_next << " : i64)\n";
    out << "^" << ex_done << ":\n";

    // active_k++
    std::string new_ak = fresh_ssa();
    out << "  " << new_ak << " = llvm.add " << ak << ", %c1_i32 : i32\n";
    out << "  llvm.store " << new_ak << ", %active_k_ptr : i32, !llvm.ptr\n";
}

void emit_expand_t(std::ostringstream& out, uint32_t axis, bool dagger) {
    std::string ak = fresh_ssa();
    out << "  " << ak << " = llvm.load %active_k_ptr : !llvm.ptr -> i32\n";
    std::string ak64 = fresh_ssa();
    out << "  " << ak64 << " = llvm.zext " << ak << " : i32 to i64\n";
    std::string half = fresh_ssa();
    out << "  " << half << " = llvm.shl %c1_i64, " << ak64 << " : i64\n";

    // Read px bit
    char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", axis);
    std::string axis_i32 = fresh_ssa();
    out << "  " << axis_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
    std::string px_bit = emit_bit_get(out, "%px_ptr", axis_i32);

    // Phase: T = (inv_sqrt2, inv_sqrt2), T_dag = (inv_sqrt2, -inv_sqrt2)
    // If px is set, negate imaginary
    double imag_base = dagger ? -kInvSqrt2 : kInvSqrt2;

    // v[i + half] = cmul(v[i], phase) for i in [0, half)
    std::string et_hdr = fresh_label("et_hdr"), et_body = fresh_label("et_body"), et_done = fresh_label("et_done");
    out << "  llvm.br ^" << et_hdr << "(%c0_i64 : i64)\n";
    out << "^" << et_hdr << "(%et_i: i64):\n";
    std::string et_cond = fresh_ssa();
    out << "  " << et_cond << " = llvm.icmp \"ult\" %et_i, " << half << " : i64\n";
    out << "  llvm.cond_br " << et_cond << ", ^" << et_body << ", ^" << et_done << "\n";
    out << "^" << et_body << ":\n";
    std::string src = emit_load_v(out, "%et_i");
    // Apply phase: use px_bit to potentially negate imag
    // cmul_const with conditional negate
    std::string pos_phase = emit_cmul_const(out, src, kInvSqrt2, imag_base);
    std::string neg_phase = emit_cmul_const(out, src, kInvSqrt2, -imag_base);
    std::string phased = fresh_ssa();
    out << "  " << phased << " = llvm.select " << px_bit << ", " << neg_phase << ", " << pos_phase
        << " : i1, !llvm.struct<(f32, f32)>\n";
    std::string dst_idx = fresh_ssa();
    out << "  " << dst_idx << " = llvm.add %et_i, " << half << " : i64\n";
    emit_store_v(out, dst_idx, phased);
    std::string et_next = fresh_ssa();
    out << "  " << et_next << " = llvm.add %et_i, %c1_i64 : i64\n";
    out << "  llvm.br ^" << et_hdr << "(" << et_next << " : i64)\n";
    out << "^" << et_done << ":\n";

    // active_k++
    std::string new_ak = fresh_ssa();
    out << "  " << new_ak << " = llvm.add " << ak << ", %c1_i32 : i32\n";
    out << "  llvm.store " << new_ak << ", %active_k_ptr : i32, !llvm.ptr\n";
}

}  // anonymous namespace

// -----------------------------------------------------------------------
// emit_mlir_text: main textual MLIR emission function
// -----------------------------------------------------------------------
std::string emit_mlir_text(const FlattenedProgram& flat) {
    label_counter = 0;
    ssa_counter = 100;
    UsedFunctions uf = analyze_used_functions(flat);

    std::ostringstream out;
    uint32_t kMaxAmplitudes = 1u << flat.peak_rank;

    out << "module attributes {llvm.target_triple = \"amdgcn-amd-amdhsa\"} {\n\n";
    emit_gpu_intrinsic_decls(out);

    out << "llvm.func amdgpu_kernelcc @compiled_mlir_kernel(\n"
        << "    %shot_offset: i64, %shots: i64, %seed: i64,\n"
        << "    %block_counts: !llvm.ptr,\n"
        << "    %num_obs: i32, %num_exp: i32) -> () {\n"
;

    out << "  %c0_i32 = llvm.mlir.constant(0 : i32) : i32\n";
    out << "  %c1_i32 = llvm.mlir.constant(1 : i32) : i32\n";
    out << "  %c2_i32 = llvm.mlir.constant(2 : i32) : i32\n";
    out << "  %c256_i32 = llvm.mlir.constant(256 : i32) : i32\n";
    out << "  %c0_i64 = llvm.mlir.constant(0 : i64) : i64\n";
    out << "  %c1_i64 = llvm.mlir.constant(1 : i64) : i64\n";
    out << "  %c6_i64 = llvm.mlir.constant(6 : i64) : i64\n";
    out << "  %c63_i64 = llvm.mlir.constant(63 : i64) : i64\n";
    out << "  %cminus1_i64 = llvm.mlir.constant(-1 : i64) : i64\n";
    out << "  %c0_i8 = llvm.mlir.constant(0 : i8) : i8\n";
    out << "  %c1_i8 = llvm.mlir.constant(1 : i8) : i8\n";
    out << "  %f_one = llvm.mlir.constant(1.0 : f32) : f32\n";
    out << "  %f_zero = llvm.mlir.constant(0.0 : f32) : f32\n";

    // Alloca state in private addrspace(5) — MUST be in entry block
    out << "  %c4_i32 = llvm.mlir.constant(4 : i32) : i32\n";
    out << "  %rng_ptr_p5 = llvm.alloca %c4_i32 x i64 : (i32) -> !llvm.ptr<5>\n";
    out << "  %rng_ptr = llvm.addrspacecast %rng_ptr_p5 : !llvm.ptr<5> to !llvm.ptr\n";
    out << "  %sm_tmp_p5 = llvm.alloca %c1_i32 x i64 : (i32) -> !llvm.ptr<5>\n";
    out << "  %sm_tmp = llvm.addrspacecast %sm_tmp_p5 : !llvm.ptr<5> to !llvm.ptr\n";
    out << "  %px_ptr_p5 = llvm.alloca %c2_i32 x i64 : (i32) -> !llvm.ptr<5>\n";
    out << "  %px_ptr = llvm.addrspacecast %px_ptr_p5 : !llvm.ptr<5> to !llvm.ptr\n";
    out << "  %pz_ptr_p5 = llvm.alloca %c2_i32 x i64 : (i32) -> !llvm.ptr<5>\n";
    out << "  %pz_ptr = llvm.addrspacecast %pz_ptr_p5 : !llvm.ptr<5> to !llvm.ptr\n";
    out << "  %active_k_ptr_p5 = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr<5>\n";
    out << "  %active_k_ptr = llvm.addrspacecast %active_k_ptr_p5 : !llvm.ptr<5> to !llvm.ptr\n";
    out << "  %discarded_ptr_p5 = llvm.alloca %c1_i32 x i8 : (i32) -> !llvm.ptr<5>\n";
    out << "  %discarded_ptr = llvm.addrspacecast %discarded_ptr_p5 : !llvm.ptr<5> to !llvm.ptr\n";

    char buf[64];
    snprintf(buf, sizeof(buf), "%u", kMaxAmplitudes);
    out << "  %v_size = llvm.mlir.constant(" << buf << " : i32) : i32\n";
    out << "  %v_ptr_p5 = llvm.alloca %v_size x !llvm.struct<(f32, f32)> : (i32) -> !llvm.ptr<5>\n";
    out << "  %v_ptr = llvm.addrspacecast %v_ptr_p5 : !llvm.ptr<5> to !llvm.ptr\n";

    snprintf(buf, sizeof(buf), "%u", kMaxMeas);
    out << "  %meas_size = llvm.mlir.constant(" << buf << " : i32) : i32\n";
    out << "  %meas_ptr_p5 = llvm.alloca %meas_size x i8 : (i32) -> !llvm.ptr<5>\n";
    out << "  %meas_ptr = llvm.addrspacecast %meas_ptr_p5 : !llvm.ptr<5> to !llvm.ptr\n";
    snprintf(buf, sizeof(buf), "%u", kMaxObs);
    out << "  %obs_size = llvm.mlir.constant(" << buf << " : i32) : i32\n";
    out << "  %obs_ptr_p5 = llvm.alloca %obs_size x i8 : (i32) -> !llvm.ptr<5>\n";
    out << "  %obs_ptr = llvm.addrspacecast %obs_ptr_p5 : !llvm.ptr<5> to !llvm.ptr\n";

    // Compute batch_shot_id = BIDX * 256 + TIDX
    out << "  %tidx_i32 = llvm.call @llvm.amdgcn.workitem.id.x() : () -> i32\n";
    out << "  %bidx_i32 = llvm.call @llvm.amdgcn.workgroup.id.x() : () -> i32\n";
    out << "  %scaled_bidx = llvm.mul %bidx_i32, %c256_i32 : i32\n";
    out << "  %shot_idx_i32 = llvm.add %scaled_bidx, %tidx_i32 : i32\n";
    out << "  %batch_shot_id = llvm.zext %shot_idx_i32 : i32 to i64\n";

    // Guard: if batch_shot_id >= shots, early return
    std::string lbl_run = fresh_label("run");
    std::string lbl_exit = fresh_label("exit");
    out << "  %oob = llvm.icmp \"uge\" %batch_shot_id, %shots : i64\n";
    out << "  llvm.cond_br %oob, ^" << lbl_exit << ", ^" << lbl_run << "\n";
    out << "^" << lbl_exit << ":\n";
    out << "  llvm.return\n";
    out << "^" << lbl_run << ":\n";

    // Initialize state
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

    // Initialize v[0] = {1.0, 0.0}
    out << "  %v0_ptr = llvm.getelementptr inbounds %v_ptr[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(f32, f32)>\n";
    out << "  %init_c0 = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
    out << "  %init_c1 = llvm.insertvalue %f_one, %init_c0[0] : !llvm.struct<(f32, f32)>\n";
    out << "  %init_c2 = llvm.insertvalue %f_zero, %init_c1[1] : !llvm.struct<(f32, f32)>\n";
    out << "  llvm.store %init_c2, %v0_ptr : !llvm.struct<(f32, f32)>, !llvm.ptr\n";

    // Initialize obs[] to zero (unroll, max 8)
    for (uint32_t i = 0; i < std::min(flat.num_observables, kMaxObs); ++i) {
        std::string idx = fresh_ssa();
        out << "  " << idx << " = llvm.mlir.constant(" << i << " : i64) : i64\n";
        std::string ptr = fresh_ssa();
        out << "  " << ptr << " = llvm.getelementptr inbounds %obs_ptr[" << idx
            << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        out << "  llvm.store %c0_i8, " << ptr << " : i8, !llvm.ptr\n";
    }

    // Seed RNG: shot_id = shot_offset + batch_shot_id
    std::string shot_id = fresh_ssa();
    out << "  " << shot_id << " = llvm.add %shot_offset, %batch_shot_id : i64\n";
    emit_rng_seed(out, "%seed", shot_id);

    // Bind helper lambdas that forward to the named functions above
    // (the .inc files call these without passing `out` explicitly)
    auto emit_bit_get = [&](const std::string& words_ptr,
                             const std::string& idx_i32) -> std::string {
        return mlir_emit::emit_bit_get(out, words_ptr, idx_i32);
    };
    auto emit_bit_xor = [&](const std::string& words_ptr,
                              const std::string& idx_i32,
                              const std::string& val_i1) {
        mlir_emit::emit_bit_xor(out, words_ptr, idx_i32, val_i1);
    };
    auto emit_bit_set = [&](const std::string& words_ptr,
                              const std::string& idx_i32,
                              const std::string& val_i1) {
        mlir_emit::emit_bit_set(out, words_ptr, idx_i32, val_i1);
    };
    auto emit_scatter_bits_1 = [&](const std::string& val_i64,
                                    const std::string& pos_i64) -> std::string {
        return mlir_emit::emit_scatter_bits_1(out, val_i64, pos_i64);
    };
    auto emit_load_v = [&](const std::string& idx_i64) -> std::string {
        return mlir_emit::emit_load_v(out, idx_i64);
    };
    auto emit_store_v = [&](const std::string& idx_i64, const std::string& val) {
        mlir_emit::emit_store_v(out, idx_i64, val);
    };
    auto emit_cadd = [&](const std::string& a, const std::string& b_val) -> std::string {
        return mlir_emit::emit_cadd(out, a, b_val);
    };
    auto emit_csub = [&](const std::string& a, const std::string& b_val) -> std::string {
        return mlir_emit::emit_csub(out, a, b_val);
    };
    auto emit_cscale_f64 = [&](const std::string& a, double scale) -> std::string {
        return mlir_emit::emit_cscale_f64(out, a, scale);
    };
    auto emit_cmul_const = [&](const std::string& a,
                                double phase_re, double phase_im) -> std::string {
        return mlir_emit::emit_cmul_const(out, a, phase_re, phase_im);
    };
    auto emit_array_h_static = [&](uint32_t axis) {
        mlir_emit::emit_array_h_static(out, axis);
    };
    auto emit_array_cnot_static = [&](uint32_t ctrl, uint32_t tgt) {
        mlir_emit::emit_array_cnot_static(out, ctrl, tgt);
    };
    auto emit_apply_phase_static = [&](uint32_t axis, double phs_re, double phs_im) {
        mlir_emit::emit_apply_phase_static(out, axis, phs_re, phs_im);
    };
    auto emit_scatter_bits_2 = [&](const std::string& val_i64,
                                    const std::string& pos1_i64,
                                    const std::string& pos2_i64) -> std::string {
        return mlir_emit::emit_scatter_bits_2(out, val_i64, pos1_i64, pos2_i64);
    };
    auto emit_cnorm = [&](const std::string& c) -> std::string {
        return mlir_emit::emit_cnorm(out, c);
    };

    // -----------------------------------------------------------------------
    // Instruction dispatch — per-category ops included from ops/*.inc
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
#include "ops/mlir_frame_ops.inc"
#include "ops/mlir_array_ops.inc"
#include "ops/mlir_measurement_ops.inc"
#include "ops/mlir_expand_ops.inc"
#include "ops/mlir_noise_ops.inc"
#include "ops/mlir_exp_val_ops.inc"
            default:
                out << "  // Unsupported op " << (unsigned)ins.opcode << " — mark discarded\n";
                out << "  llvm.store %c1_i8, %discarded_ptr : i8, !llvm.ptr\n";
                break;
        }
    }

    out << "  // --- End instruction sequence ---\n";

    // Result aggregation: check discarded, then atomic-add to block_counts
    // BlockCounts layout: passed(i64), logical_errors(i64), observable_ones[8](i64), ...
    std::string disc_val = fresh_ssa();
    out << "  " << disc_val << " = llvm.load %discarded_ptr : !llvm.ptr -> i8\n";
    std::string is_disc = fresh_ssa();
    out << "  " << is_disc << " = llvm.icmp \"ne\" " << disc_val << ", %c0_i8 : i8\n";
    std::string lbl_agg = fresh_label("agg");
    std::string lbl_done = fresh_label("done");
    out << "  llvm.cond_br " << is_disc << ", ^" << lbl_done << ", ^" << lbl_agg << "\n";
    out << "^" << lbl_agg << ":\n";

    // atomic add passed++ at byte offset 0 of block_counts
    // BlockCounts: passed(8) | logical_errors(8) | obs_ones[8](64) | exp_sums[8](64) | count(8)
    // Use byte-level GEP (treat block_counts as i8*) for simplicity
    out << "  " << fresh_ssa() << " = llvm.atomicrmw add %block_counts, %c1_i64 monotonic"
        << " : !llvm.ptr, i64\n";

    // Check observables and increment logical_errors + observable_ones
    for (uint32_t i = 0; i < std::min(flat.num_observables, kMaxObs); ++i) {
        std::string idx = fresh_ssa();
        out << "  " << idx << " = llvm.mlir.constant(" << i << " : i64) : i64\n";
        std::string obs_p = fresh_ssa();
        out << "  " << obs_p << " = llvm.getelementptr inbounds %obs_ptr["
            << idx << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        std::string oval = fresh_ssa();
        out << "  " << oval << " = llvm.load " << obs_p << " : !llvm.ptr -> i8\n";

        if (i < flat.expected_observables.size() && flat.expected_observables[i] != 0) {
            std::string xored = fresh_ssa();
            out << "  " << xored << " = llvm.xor " << oval << ", %c1_i8 : i8\n";
            oval = xored;
        }

        std::string obs_ne = fresh_ssa();
        out << "  " << obs_ne << " = llvm.icmp \"ne\" " << oval << ", %c0_i8 : i8\n";
        std::string lbl_obs_inc = fresh_label("obs_inc");
        std::string lbl_obs_skip = fresh_label("obs_skip");
        out << "  llvm.cond_br " << obs_ne << ", ^" << lbl_obs_inc << ", ^" << lbl_obs_skip << "\n";
        out << "^" << lbl_obs_inc << ":\n";

        // observable_ones[i] at byte offset 16 + i*8
        uint32_t obs_byte_offset = 16 + i * 8;
        std::string obs_off_val = fresh_ssa();
        out << "  " << obs_off_val << " = llvm.mlir.constant(" << obs_byte_offset << " : i64) : i64\n";
        std::string obs_ones_ptr = fresh_ssa();
        out << "  " << obs_ones_ptr << " = llvm.getelementptr %block_counts["
            << obs_off_val << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        out << "  " << fresh_ssa() << " = llvm.atomicrmw add " << obs_ones_ptr
            << ", %c1_i64 monotonic : !llvm.ptr, i64\n";

        // logical_errors at byte offset 8
        std::string le_ptr = fresh_ssa();
        out << "  " << le_ptr << " = llvm.mlir.constant(8 : i64) : i64\n";
        std::string le_gep = fresh_ssa();
        out << "  " << le_gep << " = llvm.getelementptr %block_counts["
            << le_ptr << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        out << "  " << fresh_ssa() << " = llvm.atomicrmw add " << le_gep
            << ", %c1_i64 monotonic : !llvm.ptr, i64\n";

        out << "  llvm.br ^" << lbl_obs_skip << "\n";
        out << "^" << lbl_obs_skip << ":\n";
    }

    out << "  llvm.br ^" << lbl_done << "\n";
    out << "^" << lbl_done << ":\n";
    out << "  llvm.return\n";
    out << "}\n\n";
    out << "} // end module\n";

    return out.str();
}

// -----------------------------------------------------------------------
// emit_mlir_text_coop: cooperative (LDS) tier kernel emission
// -----------------------------------------------------------------------
std::string emit_mlir_text_coop(const FlattenedProgram& flat) {
    label_counter = 0;
    ssa_counter = 100;

    std::ostringstream out;
    uint32_t num_amps = 1u << flat.peak_rank;

    // Module header + intrinsic declarations
    out << "module attributes {llvm.target_triple = \"amdgcn-amd-amdhsa\"} {\n\n";
    emit_gpu_intrinsic_decls(out);

    // LDS globals (address space 3)
    emit_lds_global(out, "lds_v", "!llvm.struct<(f32, f32)>", num_amps);
    emit_lds_global(out, "lds_px", "i64", 2);
    emit_lds_global(out, "lds_pz", "i64", 2);
    emit_lds_global(out, "lds_active_k", "i32", 1);
    emit_lds_global(out, "lds_discarded", "i8", 1);
    emit_lds_global(out, "lds_meas", "i8", kMaxMeas);
    emit_lds_global(out, "lds_obs", "i8", kMaxObs);
    out << "\n";

    // Kernel function
    out << "llvm.func amdgpu_kernelcc @compiled_mlir_kernel_coop(\n"
        << "    %shot_offset: i64, %shots: i64, %seed: i64,\n"
        << "    %block_counts: !llvm.ptr,\n"
        << "    %num_obs: i32, %num_exp: i32) -> ()\n"
        << "  attributes {\"amdgpu-flat-work-group-size\"=\"256,256\"} {\n"
;

    // Constants
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
    out << "  %c256_i64 = llvm.mlir.constant(256 : i64) : i64\n";
    out << "  %f_one = llvm.mlir.constant(1.0 : f32) : f32\n";
    out << "  %f_zero = llvm.mlir.constant(0.0 : f32) : f32\n";

    // Get TIDX, BIDX
    out << "  %tidx_i32 = llvm.call @llvm.amdgcn.workitem.id.x() : () -> i32\n";
    out << "  %tidx = llvm.zext %tidx_i32 : i32 to i64\n";
    out << "  %bidx_i32 = llvm.call @llvm.amdgcn.workgroup.id.x() : () -> i32\n";
    out << "  %bidx = llvm.zext %bidx_i32 : i32 to i64\n";

    // Early exit if batch_shot_id >= shots
    std::string early_cmp = fresh_ssa();
    std::string lbl_run = fresh_label("run");
    std::string lbl_exit = fresh_label("exit");
    out << "  " << early_cmp << " = llvm.icmp \"uge\" %bidx, %shots : i64\n";
    out << "  llvm.cond_br " << early_cmp << ", ^" << lbl_exit << ", ^" << lbl_run << "\n";
    out << "^" << lbl_exit << ":\n";
    out << "  llvm.return\n";
    out << "^" << lbl_run << ":\n";

    // Get LDS pointers (addrspacecast from ptr<3> to generic)
    out << "  %lds_v_as3 = llvm.mlir.addressof @lds_v : !llvm.ptr<3>\n";
    out << "  %v_ptr = llvm.addrspacecast %lds_v_as3 : !llvm.ptr<3> to !llvm.ptr\n";
    out << "  %lds_px_as3 = llvm.mlir.addressof @lds_px : !llvm.ptr<3>\n";
    out << "  %px_ptr = llvm.addrspacecast %lds_px_as3 : !llvm.ptr<3> to !llvm.ptr\n";
    out << "  %lds_pz_as3 = llvm.mlir.addressof @lds_pz : !llvm.ptr<3>\n";
    out << "  %pz_ptr = llvm.addrspacecast %lds_pz_as3 : !llvm.ptr<3> to !llvm.ptr\n";
    out << "  %lds_ak_as3 = llvm.mlir.addressof @lds_active_k : !llvm.ptr<3>\n";
    out << "  %active_k_ptr = llvm.addrspacecast %lds_ak_as3 : !llvm.ptr<3> to !llvm.ptr\n";
    out << "  %lds_disc_as3 = llvm.mlir.addressof @lds_discarded : !llvm.ptr<3>\n";
    out << "  %discarded_ptr = llvm.addrspacecast %lds_disc_as3 : !llvm.ptr<3> to !llvm.ptr\n";
    out << "  %lds_meas_as3 = llvm.mlir.addressof @lds_meas : !llvm.ptr<3>\n";
    out << "  %meas_ptr = llvm.addrspacecast %lds_meas_as3 : !llvm.ptr<3> to !llvm.ptr\n";
    out << "  %lds_obs_as3 = llvm.mlir.addressof @lds_obs : !llvm.ptr<3>\n";
    out << "  %obs_ptr = llvm.addrspacecast %lds_obs_as3 : !llvm.ptr<3> to !llvm.ptr\n";
    out << "  %px0_ptr = llvm.getelementptr inbounds %px_ptr[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
    out << "  %px1_ptr = llvm.getelementptr inbounds %px_ptr[%c1_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
    out << "  %pz0_ptr = llvm.getelementptr inbounds %pz_ptr[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
    out << "  %pz1_ptr = llvm.getelementptr inbounds %pz_ptr[%c1_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";

    // Cooperative init: zero v[] across all threads
    char ampbuf[32]; snprintf(ampbuf, sizeof(ampbuf), "%u", num_amps);
    out << "  %num_amps = llvm.mlir.constant(" << ampbuf << " : i64) : i64\n";
    std::string init_hdr = fresh_label("init_hdr");
    std::string init_body = fresh_label("init_body");
    std::string init_done = fresh_label("init_done");
    out << "  llvm.br ^" << init_hdr << "(%tidx : i64)\n";
    out << "^" << init_hdr << "(%init_i: i64):\n";
    std::string init_cond = fresh_ssa();
    out << "  " << init_cond << " = llvm.icmp \"ult\" %init_i, %num_amps : i64\n";
    out << "  llvm.cond_br " << init_cond << ", ^" << init_body << ", ^" << init_done << "\n";
    out << "^" << init_body << ":\n";
    out << "  %zero_c = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
    out << "  %zero_c1 = llvm.insertvalue %f_zero, %zero_c[0] : !llvm.struct<(f32, f32)>\n";
    out << "  %zero_c2 = llvm.insertvalue %f_zero, %zero_c1[1] : !llvm.struct<(f32, f32)>\n";
    std::string vp_init = fresh_ssa();
    out << "  " << vp_init << " = llvm.getelementptr inbounds %v_ptr[%init_i] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(f32, f32)>\n";
    out << "  llvm.store %zero_c2, " << vp_init << " : !llvm.struct<(f32, f32)>, !llvm.ptr\n";
    std::string init_next = fresh_ssa();
    out << "  " << init_next << " = llvm.add %init_i, %c256_i64 : i64\n";
    out << "  llvm.br ^" << init_hdr << "(" << init_next << " : i64)\n";
    out << "^" << init_done << ":\n";
    emit_barrier(out);

    // Thread-0 init: px=0, pz=0, active_k=0, discarded=0, v[0]={1,0}
    std::string is_t0 = fresh_ssa();
    std::string t0_init = fresh_label("t0_init");
    std::string t0_done = fresh_label("t0_done");
    out << "  " << is_t0 << " = llvm.icmp \"eq\" %tidx_i32, %c0_i32 : i32\n";
    out << "  llvm.cond_br " << is_t0 << ", ^" << t0_init << ", ^" << t0_done << "\n";
    out << "^" << t0_init << ":\n";
    out << "  llvm.store %c0_i64, %px0_ptr : i64, !llvm.ptr\n";
    out << "  llvm.store %c0_i64, %px1_ptr : i64, !llvm.ptr\n";
    out << "  llvm.store %c0_i64, %pz0_ptr : i64, !llvm.ptr\n";
    out << "  llvm.store %c0_i64, %pz1_ptr : i64, !llvm.ptr\n";
    out << "  llvm.store %c0_i32, %active_k_ptr : i32, !llvm.ptr\n";
    out << "  llvm.store %c0_i8, %discarded_ptr : i8, !llvm.ptr\n";
    out << "  %v0_ptr_coop = llvm.getelementptr inbounds %v_ptr[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(f32, f32)>\n";
    out << "  %init_one = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
    out << "  %init_one1 = llvm.insertvalue %f_one, %init_one[0] : !llvm.struct<(f32, f32)>\n";
    out << "  %init_one2 = llvm.insertvalue %f_zero, %init_one1[1] : !llvm.struct<(f32, f32)>\n";
    out << "  llvm.store %init_one2, %v0_ptr_coop : !llvm.struct<(f32, f32)>, !llvm.ptr\n";
    out << "  llvm.br ^" << t0_done << "\n";
    out << "^" << t0_done << ":\n";
    emit_barrier(out);

    // Instruction dispatch — coop pattern:
    //   Frame ops: if (TIDX==0) { ... } barrier()
    //   Array ops: cooperative loop with barrier (reuse register-tier .inc files)
    //   For now, reuse same .inc dispatch (register-tier patterns work for single-thread-per-shot)
    //   TODO: convert array ops to cooperative loops for true multi-thread cooperation

    // Bind lambdas (same as register tier — the .inc files use these)
    auto emit_bit_get = [&](const std::string& words_ptr,
                             const std::string& idx_i32) -> std::string {
        return mlir_emit::emit_bit_get(out, words_ptr, idx_i32);
    };
    auto emit_bit_xor = [&](const std::string& words_ptr,
                              const std::string& idx_i32,
                              const std::string& val_i1) {
        mlir_emit::emit_bit_xor(out, words_ptr, idx_i32, val_i1);
    };
    auto emit_bit_set = [&](const std::string& words_ptr,
                              const std::string& idx_i32,
                              const std::string& val_i1) {
        mlir_emit::emit_bit_set(out, words_ptr, idx_i32, val_i1);
    };
    auto emit_scatter_bits_1 = [&](const std::string& val_i64,
                                    const std::string& pos_i64) -> std::string {
        return mlir_emit::emit_scatter_bits_1(out, val_i64, pos_i64);
    };
    auto emit_scatter_bits_2 = [&](const std::string& val_i64,
                                    const std::string& pos1_i64,
                                    const std::string& pos2_i64) -> std::string {
        return mlir_emit::emit_scatter_bits_2(out, val_i64, pos1_i64, pos2_i64);
    };
    auto emit_load_v = [&](const std::string& idx_i64) -> std::string {
        return mlir_emit::emit_load_v(out, idx_i64);
    };
    auto emit_store_v = [&](const std::string& idx_i64, const std::string& val) {
        mlir_emit::emit_store_v(out, idx_i64, val);
    };
    auto emit_cadd = [&](const std::string& a, const std::string& b_val) -> std::string {
        return mlir_emit::emit_cadd(out, a, b_val);
    };
    auto emit_csub = [&](const std::string& a, const std::string& b_val) -> std::string {
        return mlir_emit::emit_csub(out, a, b_val);
    };
    auto emit_cscale_f64 = [&](const std::string& a, double scale) -> std::string {
        return mlir_emit::emit_cscale_f64(out, a, scale);
    };
    auto emit_cmul_const = [&](const std::string& a,
                                double phase_re, double phase_im) -> std::string {
        return mlir_emit::emit_cmul_const(out, a, phase_re, phase_im);
    };
    auto emit_array_h_static = [&](uint32_t axis) {
        mlir_emit::emit_array_h_static(out, axis);
    };
    auto emit_array_cnot_static = [&](uint32_t ctrl, uint32_t tgt) {
        mlir_emit::emit_array_cnot_static(out, ctrl, tgt);
    };
    auto emit_apply_phase_static = [&](uint32_t axis, double phs_re, double phs_im) {
        mlir_emit::emit_apply_phase_static(out, axis, phs_re, phs_im);
    };
    auto emit_cnorm = [&](const std::string& c) -> std::string {
        return mlir_emit::emit_cnorm(out, c);
    };

    using Opcode = clifft::Opcode;
    out << "  // --- Coop instruction sequence (" << flat.instrs.size() << " ops) ---\n";

    for (size_t pc = 0; pc < flat.instrs.size(); ++pc) {
        const GpuInstr& ins = flat.instrs[pc];
        bool sign = (ins.flags & kFlagSign) != 0;
        bool identity = (ins.flags & kFlagIdentity) != 0;
        auto op = static_cast<Opcode>(ins.opcode);

        out << "  // op[" << pc << "] opcode=" << (unsigned)ins.opcode << "\n";

        switch (op) {
#include "ops/mlir_frame_ops.inc"
#include "ops/mlir_array_ops.inc"
#include "ops/mlir_measurement_ops.inc"
#include "ops/mlir_expand_ops.inc"
#include "ops/mlir_noise_ops.inc"
#include "ops/mlir_exp_val_ops.inc"
            default:
                out << "  // Unsupported op " << (unsigned)ins.opcode << " — mark discarded\n";
                out << "  llvm.store %c1_i8, %discarded_ptr : i8, !llvm.ptr\n";
                break;
        }
    }

    out << "  // --- End coop instruction sequence ---\n";
    out << "  llvm.return\n";
    out << "}\n\n";
    out << "} // end module\n";

    return out.str();
}

// -----------------------------------------------------------------------
// emit_mlir_text_global: global (HBM) tier kernel emission
// -----------------------------------------------------------------------
std::string emit_mlir_text_global(const FlattenedProgram& flat) {
    label_counter = 0;
    ssa_counter = 100;

    std::ostringstream out;

    // Module header + intrinsic declarations
    out << "module attributes {llvm.target_triple = \"amdgcn-amd-amdhsa\"} {\n\n";
    emit_gpu_intrinsic_decls(out);

    // LDS globals for frame/classical state (amplitudes are in HBM)
    emit_lds_global(out, "lds_px", "i64", 2);
    emit_lds_global(out, "lds_pz", "i64", 2);
    emit_lds_global(out, "lds_active_k", "i32", 1);
    emit_lds_global(out, "lds_discarded", "i8", 1);
    emit_lds_global(out, "lds_meas", "i8", kMaxMeas);
    emit_lds_global(out, "lds_obs", "i8", kMaxObs);
    emit_lds_global(out, "lds_batch_shot_id", "i64", 1);
    out << "\n";

    // Global kernel with HBM pointers
    out << "llvm.func amdgpu_kernelcc @compiled_mlir_kernel_global(\n"
        << "    %shot_offset: i64, %shots: i64, %seed: i64,\n"
        << "    %global_v: !llvm.ptr, %global_scratch: !llvm.ptr,\n"
        << "    %work_counter: !llvm.ptr,\n"
        << "    %block_counts: !llvm.ptr,\n"
        << "    %num_obs: i32, %num_exp: i32) -> ()\n"
        << "  attributes {\"amdgpu-flat-work-group-size\"=\"256,256\"} {\n"
;

    // Constants
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
    out << "  %c8_i64 = llvm.mlir.constant(8 : i64) : i64\n";
    out << "  %c256_i64 = llvm.mlir.constant(256 : i64) : i64\n";
    out << "  %f_one = llvm.mlir.constant(1.0 : f32) : f32\n";
    out << "  %f_zero = llvm.mlir.constant(0.0 : f32) : f32\n";

    // kGlobalMaxPeakRank stride for HBM slot addressing
    char stridebuf[32];
    snprintf(stridebuf, sizeof(stridebuf), "%u", 1u << kGlobalMaxPeakRank);
    out << "  %hbm_stride = llvm.mlir.constant(" << stridebuf << " : i64) : i64\n";

    // Get TIDX, BIDX
    out << "  %tidx_i32 = llvm.call @llvm.amdgcn.workitem.id.x() : () -> i32\n";
    out << "  %tidx = llvm.zext %tidx_i32 : i32 to i64\n";
    out << "  %bidx_i32 = llvm.call @llvm.amdgcn.workgroup.id.x() : () -> i32\n";
    out << "  %bidx = llvm.zext %bidx_i32 : i32 to i64\n";

    // Compute v_ptr = global_v + BIDX * hbm_stride
    std::string v_off = fresh_ssa();
    out << "  " << v_off << " = llvm.mul %bidx, %hbm_stride : i64\n";
    out << "  %v_ptr = llvm.getelementptr inbounds %global_v[" << v_off
        << "] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(f32, f32)>\n";

    // XCD work-stealing: get XCD ID, compute my_counter pointer
    std::string xcd_raw = fresh_ssa();
    out << "  " << xcd_raw << " = llvm.inline_asm "
        << "\"s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)\", \"=s\" "
        << ": () -> i32\n";
    std::string xcd_mod = fresh_ssa();
    std::string c8_i32 = emit_const_i32(out, 8);
    out << "  " << xcd_mod << " = llvm.urem " << xcd_raw << ", " << c8_i32 << " : i32\n";
    std::string xcd_i64 = fresh_ssa();
    out << "  " << xcd_i64 << " = llvm.zext " << xcd_mod << " : i32 to i64\n";
    std::string my_counter = fresh_ssa();
    out << "  " << my_counter << " = llvm.getelementptr inbounds %work_counter["
        << xcd_i64 << "] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";

    // LDS pointers for frame/classical state
    out << "  %lds_px_as3 = llvm.mlir.addressof @lds_px : !llvm.ptr<3>\n";
    out << "  %px_ptr = llvm.addrspacecast %lds_px_as3 : !llvm.ptr<3> to !llvm.ptr\n";
    out << "  %lds_pz_as3 = llvm.mlir.addressof @lds_pz : !llvm.ptr<3>\n";
    out << "  %pz_ptr = llvm.addrspacecast %lds_pz_as3 : !llvm.ptr<3> to !llvm.ptr\n";
    out << "  %lds_ak_as3 = llvm.mlir.addressof @lds_active_k : !llvm.ptr<3>\n";
    out << "  %active_k_ptr = llvm.addrspacecast %lds_ak_as3 : !llvm.ptr<3> to !llvm.ptr\n";
    out << "  %lds_disc_as3 = llvm.mlir.addressof @lds_discarded : !llvm.ptr<3>\n";
    out << "  %discarded_ptr = llvm.addrspacecast %lds_disc_as3 : !llvm.ptr<3> to !llvm.ptr\n";
    out << "  %lds_meas_as3 = llvm.mlir.addressof @lds_meas : !llvm.ptr<3>\n";
    out << "  %meas_ptr = llvm.addrspacecast %lds_meas_as3 : !llvm.ptr<3> to !llvm.ptr\n";
    out << "  %lds_obs_as3 = llvm.mlir.addressof @lds_obs : !llvm.ptr<3>\n";
    out << "  %obs_ptr = llvm.addrspacecast %lds_obs_as3 : !llvm.ptr<3> to !llvm.ptr\n";
    out << "  %lds_bsi_as3 = llvm.mlir.addressof @lds_batch_shot_id : !llvm.ptr<3>\n";
    out << "  %batch_shot_id_ptr = llvm.addrspacecast %lds_bsi_as3 : !llvm.ptr<3> to !llvm.ptr\n";
    out << "  %px0_ptr = llvm.getelementptr inbounds %px_ptr[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
    out << "  %px1_ptr = llvm.getelementptr inbounds %px_ptr[%c1_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
    out << "  %pz0_ptr = llvm.getelementptr inbounds %pz_ptr[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
    out << "  %pz1_ptr = llvm.getelementptr inbounds %pz_ptr[%c1_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";

    // Number of amplitudes for this circuit (may be < kGlobalMaxPeakRank)
    char global_ampbuf[32];
    snprintf(global_ampbuf, sizeof(global_ampbuf), "%u", 1u << flat.peak_rank);
    out << "  %num_amps = llvm.mlir.constant(" << global_ampbuf << " : i64) : i64\n";

    // Work-stealing loop
    std::string work_hdr = fresh_label("work_hdr");
    std::string work_body = fresh_label("work_body");
    std::string work_exit = fresh_label("work_exit");
    out << "  llvm.br ^" << work_hdr << "\n";
    out << "^" << work_hdr << ":\n";

    // Thread-0: atomic increment work counter, compute batch_shot_id
    std::string is_t0_wk = fresh_ssa();
    std::string t0_wk = fresh_label("t0_wk");
    std::string t0_wk_done = fresh_label("t0_wk_done");
    out << "  " << is_t0_wk << " = llvm.icmp \"eq\" %tidx_i32, %c0_i32 : i32\n";
    out << "  llvm.cond_br " << is_t0_wk << ", ^" << t0_wk << ", ^" << t0_wk_done << "\n";
    out << "^" << t0_wk << ":\n";
    std::string slot = fresh_ssa();
    out << "  " << slot << " = llvm.atomicrmw add " << my_counter
        << ", %c1_i64 monotonic : !llvm.ptr, i64\n";
    std::string scaled = fresh_ssa();
    out << "  " << scaled << " = llvm.mul " << slot << ", %c8_i64 : i64\n";
    std::string bsi = fresh_ssa();
    out << "  " << bsi << " = llvm.add " << scaled << ", " << xcd_i64 << " : i64\n";
    out << "  llvm.store " << bsi << ", %batch_shot_id_ptr : i64, !llvm.ptr\n";
    out << "  llvm.br ^" << t0_wk_done << "\n";
    out << "^" << t0_wk_done << ":\n";
    emit_barrier(out);

    // All threads: load batch_shot_id, check if done
    std::string my_shot = fresh_ssa();
    out << "  " << my_shot << " = llvm.load %batch_shot_id_ptr : !llvm.ptr -> i64\n";
    std::string done_cmp = fresh_ssa();
    out << "  " << done_cmp << " = llvm.icmp \"uge\" " << my_shot << ", %shots : i64\n";
    out << "  llvm.cond_br " << done_cmp << ", ^" << work_exit << ", ^" << work_body << "\n";
    out << "^" << work_body << ":\n";

    // Init v[] in HBM cooperatively
    std::string ginit_hdr = fresh_label("ginit_hdr");
    std::string ginit_body = fresh_label("ginit_body");
    std::string ginit_done = fresh_label("ginit_done");
    out << "  llvm.br ^" << ginit_hdr << "(%tidx : i64)\n";
    out << "^" << ginit_hdr << "(%ginit_i: i64):\n";
    std::string ginit_cond = fresh_ssa();
    out << "  " << ginit_cond << " = llvm.icmp \"ult\" %ginit_i, %num_amps : i64\n";
    out << "  llvm.cond_br " << ginit_cond << ", ^" << ginit_body << ", ^" << ginit_done << "\n";
    out << "^" << ginit_body << ":\n";
    out << "  %gzero_c = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
    out << "  %gzero_c1 = llvm.insertvalue %f_zero, %gzero_c[0] : !llvm.struct<(f32, f32)>\n";
    out << "  %gzero_c2 = llvm.insertvalue %f_zero, %gzero_c1[1] : !llvm.struct<(f32, f32)>\n";
    std::string gvp = fresh_ssa();
    out << "  " << gvp << " = llvm.getelementptr inbounds %v_ptr[%ginit_i] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(f32, f32)>\n";
    out << "  llvm.store %gzero_c2, " << gvp << " : !llvm.struct<(f32, f32)>, !llvm.ptr\n";
    std::string ginit_next = fresh_ssa();
    out << "  " << ginit_next << " = llvm.add %ginit_i, %c256_i64 : i64\n";
    out << "  llvm.br ^" << ginit_hdr << "(" << ginit_next << " : i64)\n";
    out << "^" << ginit_done << ":\n";
    emit_barrier(out);

    // Thread-0 init classical state
    std::string is_t0_g = fresh_ssa();
    std::string t0_ginit = fresh_label("t0_ginit");
    std::string t0_ginit_done = fresh_label("t0_ginit_done");
    out << "  " << is_t0_g << " = llvm.icmp \"eq\" %tidx_i32, %c0_i32 : i32\n";
    out << "  llvm.cond_br " << is_t0_g << ", ^" << t0_ginit << ", ^" << t0_ginit_done << "\n";
    out << "^" << t0_ginit << ":\n";
    out << "  llvm.store %c0_i64, %px0_ptr : i64, !llvm.ptr\n";
    out << "  llvm.store %c0_i64, %px1_ptr : i64, !llvm.ptr\n";
    out << "  llvm.store %c0_i64, %pz0_ptr : i64, !llvm.ptr\n";
    out << "  llvm.store %c0_i64, %pz1_ptr : i64, !llvm.ptr\n";
    out << "  llvm.store %c0_i32, %active_k_ptr : i32, !llvm.ptr\n";
    out << "  llvm.store %c0_i8, %discarded_ptr : i8, !llvm.ptr\n";
    out << "  %gv0_ptr = llvm.getelementptr inbounds %v_ptr[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(f32, f32)>\n";
    out << "  %ginit_one = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
    out << "  %ginit_one1 = llvm.insertvalue %f_one, %ginit_one[0] : !llvm.struct<(f32, f32)>\n";
    out << "  %ginit_one2 = llvm.insertvalue %f_zero, %ginit_one1[1] : !llvm.struct<(f32, f32)>\n";
    out << "  llvm.store %ginit_one2, %gv0_ptr : !llvm.struct<(f32, f32)>, !llvm.ptr\n";
    out << "  llvm.br ^" << t0_ginit_done << "\n";
    out << "^" << t0_ginit_done << ":\n";
    emit_barrier(out);

    // Instruction dispatch (same .inc files as register/coop tier)
    auto emit_bit_get = [&](const std::string& words_ptr,
                             const std::string& idx_i32) -> std::string {
        return mlir_emit::emit_bit_get(out, words_ptr, idx_i32);
    };
    auto emit_bit_xor = [&](const std::string& words_ptr,
                              const std::string& idx_i32,
                              const std::string& val_i1) {
        mlir_emit::emit_bit_xor(out, words_ptr, idx_i32, val_i1);
    };
    auto emit_bit_set = [&](const std::string& words_ptr,
                              const std::string& idx_i32,
                              const std::string& val_i1) {
        mlir_emit::emit_bit_set(out, words_ptr, idx_i32, val_i1);
    };
    auto emit_scatter_bits_1 = [&](const std::string& val_i64,
                                    const std::string& pos_i64) -> std::string {
        return mlir_emit::emit_scatter_bits_1(out, val_i64, pos_i64);
    };
    auto emit_scatter_bits_2 = [&](const std::string& val_i64,
                                    const std::string& pos1_i64,
                                    const std::string& pos2_i64) -> std::string {
        return mlir_emit::emit_scatter_bits_2(out, val_i64, pos1_i64, pos2_i64);
    };
    auto emit_load_v = [&](const std::string& idx_i64) -> std::string {
        return mlir_emit::emit_load_v(out, idx_i64);
    };
    auto emit_store_v = [&](const std::string& idx_i64, const std::string& val) {
        mlir_emit::emit_store_v(out, idx_i64, val);
    };
    auto emit_cadd = [&](const std::string& a, const std::string& b_val) -> std::string {
        return mlir_emit::emit_cadd(out, a, b_val);
    };
    auto emit_csub = [&](const std::string& a, const std::string& b_val) -> std::string {
        return mlir_emit::emit_csub(out, a, b_val);
    };
    auto emit_cscale_f64 = [&](const std::string& a, double scale) -> std::string {
        return mlir_emit::emit_cscale_f64(out, a, scale);
    };
    auto emit_cmul_const = [&](const std::string& a,
                                double phase_re, double phase_im) -> std::string {
        return mlir_emit::emit_cmul_const(out, a, phase_re, phase_im);
    };
    auto emit_array_h_static = [&](uint32_t axis) {
        mlir_emit::emit_array_h_static(out, axis);
    };
    auto emit_array_cnot_static = [&](uint32_t ctrl, uint32_t tgt) {
        mlir_emit::emit_array_cnot_static(out, ctrl, tgt);
    };
    auto emit_apply_phase_static = [&](uint32_t axis, double phs_re, double phs_im) {
        mlir_emit::emit_apply_phase_static(out, axis, phs_re, phs_im);
    };
    auto emit_cnorm = [&](const std::string& c) -> std::string {
        return mlir_emit::emit_cnorm(out, c);
    };

    using Opcode = clifft::Opcode;
    out << "  // --- Global instruction sequence (" << flat.instrs.size() << " ops) ---\n";

    for (size_t pc = 0; pc < flat.instrs.size(); ++pc) {
        const GpuInstr& ins = flat.instrs[pc];
        bool sign = (ins.flags & kFlagSign) != 0;
        bool identity = (ins.flags & kFlagIdentity) != 0;
        auto op = static_cast<Opcode>(ins.opcode);

        out << "  // op[" << pc << "] opcode=" << (unsigned)ins.opcode << "\n";

        switch (op) {
#include "ops/mlir_frame_ops.inc"
#include "ops/mlir_array_ops.inc"
#include "ops/mlir_measurement_ops.inc"
#include "ops/mlir_expand_ops.inc"
#include "ops/mlir_noise_ops.inc"
#include "ops/mlir_exp_val_ops.inc"
            default:
                out << "  // Unsupported op " << (unsigned)ins.opcode << " — mark discarded\n";
                out << "  llvm.store %c1_i8, %discarded_ptr : i8, !llvm.ptr\n";
                break;
        }
    }

    out << "  // --- End global instruction sequence ---\n";

    // Loop back for next shot
    out << "  llvm.br ^" << work_hdr << "\n";

    // Exit
    out << "^" << work_exit << ":\n";
    out << "  llvm.return\n";
    out << "}\n\n";
    out << "} // end module\n";

    return out.str();
}

}  // namespace mlir_emit
}  // namespace gpu
}  // namespace clifft

#endif  // CLIFFT_ENABLE_MLIR
