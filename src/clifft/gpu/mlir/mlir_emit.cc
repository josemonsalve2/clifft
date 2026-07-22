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
    out << "  " << im_sh << " = llvm.shl " << im_i64
        << ", llvm.mlir.constant(32 : i64) : i64\n";
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

    out << "llvm.func @compiled_mlir_kernel(\n"
        << "    %shot_offset: i64, %shots: i64, %seed: i64,\n"
        << "    %block_counts: !llvm.ptr,\n"
        << "    %num_obs: i32, %num_exp: i32) -> () {\n"
        << "  ^entry:\n";

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

    out << "  %px_ptr = llvm.alloca %c2_i32 x i64 : (i32) -> !llvm.ptr\n";
    out << "  %pz_ptr = llvm.alloca %c2_i32 x i64 : (i32) -> !llvm.ptr\n";
    out << "  %active_k_ptr = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr\n";
    out << "  %discarded_ptr = llvm.alloca %c1_i32 x i8 : (i32) -> !llvm.ptr\n";

    char buf[64];
    snprintf(buf, sizeof(buf), "%u", kMaxAmplitudes);
    out << "  %v_size = llvm.mlir.constant(" << buf << " : i32) : i32\n";
    out << "  %v_ptr = llvm.alloca %v_size x !llvm.struct<(f32, f32)> : (i32) -> !llvm.ptr\n";

    snprintf(buf, sizeof(buf), "%u", kMaxMeas);
    out << "  %meas_size = llvm.mlir.constant(" << buf << " : i32) : i32\n";
    out << "  %meas_ptr = llvm.alloca %meas_size x i8 : (i32) -> !llvm.ptr\n";
    snprintf(buf, sizeof(buf), "%u", kMaxObs);
    out << "  %obs_size = llvm.mlir.constant(" << buf << " : i32) : i32\n";
    out << "  %obs_ptr = llvm.alloca %obs_size x i8 : (i32) -> !llvm.ptr\n";

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

    out << "  %v0_ptr = llvm.getelementptr inbounds %v_ptr[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(f32, f32)>\n";
    out << "  %f_one = llvm.mlir.constant(1.0 : f32) : f32\n";
    out << "  %f_zero = llvm.mlir.constant(0.0 : f32) : f32\n";
    out << "  %init_c0 = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
    out << "  %init_c1 = llvm.insertvalue %f_one, %init_c0[0] : !llvm.struct<(f32, f32)>\n";
    out << "  %init_c2 = llvm.insertvalue %f_zero, %init_c1[1] : !llvm.struct<(f32, f32)>\n";
    out << "  llvm.store %init_c2, %v0_ptr : !llvm.struct<(f32, f32)>, !llvm.ptr\n";

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

}  // namespace mlir_emit
}  // namespace gpu
}  // namespace clifft

#endif  // CLIFFT_ENABLE_MLIR
