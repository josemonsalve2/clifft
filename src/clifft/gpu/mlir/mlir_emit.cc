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
#include <utility>

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
static bool cooperative_mode = false;
static bool lds_amplitudes = false;

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
    const char* pt = lds_amplitudes ? "!llvm.ptr<3>" : "!llvm.ptr";
    out << "  " << vp << " = llvm.getelementptr inbounds %v_ptr[" << idx_i64
        << "] : (" << pt << ", i64) -> " << pt << ", !llvm.struct<(f32, f32)>\n";
    std::string vc = fresh_ssa();
    // alignment=8 for coop/global tiers enables ds_read_b64 / global_load_b64
    // EXCLUDED from register tier to avoid VGPR spill catastrophe (P1.9)
    const char* align = (lds_amplitudes || cooperative_mode) ? " {alignment = 8 : i64}" : "";
    out << "  " << vc << " = llvm.load " << vp << align << " : " << pt << " -> !llvm.struct<(f32, f32)>\n";
    return vc;
}

void emit_store_v(std::ostringstream& out,
                   const std::string& idx_i64, const std::string& val) {
    std::string vp = fresh_ssa();
    const char* pt = lds_amplitudes ? "!llvm.ptr<3>" : "!llvm.ptr";
    out << "  " << vp << " = llvm.getelementptr inbounds %v_ptr[" << idx_i64
        << "] : (" << pt << ", i64) -> " << pt << ", !llvm.struct<(f32, f32)>\n";
    const char* align = (lds_amplitudes || cooperative_mode) ? " {alignment = 8 : i64}" : "";
    out << "  llvm.store " << val << ", " << vp << align
        << " : !llvm.struct<(f32, f32)>, " << pt << "\n";
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
    // Match gold cscale (hip_sampler.hip): extend f32 amplitude to f64,
    // multiply by the f64 scale, then truncate back to f32. Truncating the
    // scale to f32 and multiplying in f32 loses precision on the fold path.
    char sbuf[64];
    snprintf(sbuf, sizeof(sbuf), "%.17e", scale);
    std::string sf64 = fresh_ssa();
    out << "  " << sf64 << " = llvm.mlir.constant(" << sbuf << " : f64) : f64\n";
    std::string are = fresh_ssa(), aim = fresh_ssa();
    out << "  " << are << " = llvm.extractvalue " << a << "[0] : !llvm.struct<(f32, f32)>\n";
    out << "  " << aim << " = llvm.extractvalue " << a << "[1] : !llvm.struct<(f32, f32)>\n";
    std::string are64 = fresh_ssa(), aim64 = fresh_ssa();
    out << "  " << are64 << " = llvm.fpext " << are << " : f32 to f64\n";
    out << "  " << aim64 << " = llvm.fpext " << aim << " : f32 to f64\n";
    std::string re64 = fresh_ssa(), im64 = fresh_ssa();
    out << "  " << re64 << " = llvm.fmul " << are64 << ", " << sf64 << " : f64\n";
    out << "  " << im64 << " = llvm.fmul " << aim64 << ", " << sf64 << " : f64\n";
    std::string re = fresh_ssa(), im = fresh_ssa();
    out << "  " << re << " = llvm.fptrunc " << re64 << " : f64 to f32\n";
    out << "  " << im << " = llvm.fptrunc " << im64 << " : f64 to f32\n";
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

// Forward declarations for functions defined later but called here
void emit_barrier(std::ostringstream& out);
std::string emit_tid0_guard_begin(std::ostringstream& out);
void emit_tid0_guard_end(std::ostringstream& out, const std::string& lbl_done);
void emit_tid0_guard_end_nobarrier(std::ostringstream& out, const std::string& lbl_done);
std::string emit_lds_base_ptr(std::ostringstream& out,
                              const std::string& name,
                              const std::string& elem_type,
                              uint32_t count);

// Compile-time active_k tracking. Updated by the instruction loop in
// emit_mlir_text() as it walks expand/meas ops. Each gate emitter
// receives this value so it can emit loop trip counts as constants.
static uint32_t g_emit_ak = 0;

// -----------------------------------------------------------------------
// Gate-level emitters (used by ops/*.inc)
// -----------------------------------------------------------------------

void emit_array_h_static(std::ostringstream& out, uint32_t axis) {
    out << "  // array_h on axis " << axis << " (active_k=" << g_emit_ak << ")\n";
    uint64_t iters_val = (g_emit_ak > 0) ? (1ULL << (g_emit_ak - 1)) : 0;
    std::string iters = emit_const_i64(out, iters_val);
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
    std::string lv = fresh_ssa();
    std::string loop_init = cooperative_mode ? "%tidx" : "%c0_i64";
    std::string loop_step = cooperative_mode ? "%c256_i64" : "%c1_i64";
    out << "  llvm.br ^" << hdr << "(" << loop_init << " : i64)\n";
    out << "^" << hdr << "(" << lv << ": i64):\n";
    std::string cond = fresh_ssa();
    out << "  " << cond << " = llvm.icmp \"ult\" " << lv << ", " << iters << " : i64\n";
    out << "  llvm.cond_br " << cond << ", ^" << body << ", ^" << exit << "\n";
    out << "^" << body << ":\n";

    std::string idx0 = emit_scatter_bits_1(out, lv, axis_val);
    std::string idx1 = fresh_ssa();
    out << "  " << idx1 << " = llvm.or " << idx0 << ", " << axis_bit << " : i64\n";

    std::string va = emit_load_v(out, idx0);
    std::string vb = emit_load_v(out, idx1);
    std::string sum = emit_cadd(out, va, vb);
    std::string diff = emit_csub(out, va, vb);
    emit_store_v(out, idx0, emit_cscale_f64(out, sum, kInvSqrt2));
    emit_store_v(out, idx1, emit_cscale_f64(out, diff, kInvSqrt2));

    std::string i_next = fresh_ssa();
    out << "  " << i_next << " = llvm.add " << lv << ", " << loop_step << " : i64\n";
    out << "  llvm.br ^" << hdr << "(" << i_next << " : i64)\n";
    out << "^" << exit << ":\n";
    if (cooperative_mode) emit_barrier(out);
    (void)inv_sq2;
}

void emit_array_cnot_static(std::ostringstream& out, uint32_t ctrl, uint32_t tgt) {
    out << "  // array_cnot ctrl=" << ctrl << " tgt=" << tgt << " (active_k=" << g_emit_ak << ")\n";
    uint64_t iters_val = (g_emit_ak >= 2) ? (1ULL << (g_emit_ak - 2)) : 0;
    std::string iters = emit_const_i64(out, iters_val);

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
    std::string lv = fresh_ssa();
    std::string cn_init = cooperative_mode ? "%tidx" : "%c0_i64";
    std::string cn_step = cooperative_mode ? "%c256_i64" : "%c1_i64";
    out << "  llvm.br ^" << hdr << "(" << cn_init << " : i64)\n";
    out << "^" << hdr << "(" << lv << ": i64):\n";
    std::string cond = fresh_ssa();
    out << "  " << cond << " = llvm.icmp \"ult\" " << lv << ", " << iters << " : i64\n";
    out << "  llvm.cond_br " << cond << ", ^" << body << ", ^" << exit_lbl << "\n";
    out << "^" << body << ":\n";

    uint32_t lo = std::min(ctrl, tgt), hi = std::max(ctrl, tgt);
    char lobuf[32], hibuf[32];
    snprintf(lobuf, sizeof(lobuf), "%u", lo);
    snprintf(hibuf, sizeof(hibuf), "%u", hi);
    std::string lo64 = fresh_ssa(), hi64 = fresh_ssa();
    out << "  " << lo64 << " = llvm.mlir.constant(" << lobuf << " : i64) : i64\n";
    out << "  " << hi64 << " = llvm.mlir.constant(" << hibuf << " : i64) : i64\n";
    std::string s1 = emit_scatter_bits_1(out, lv, lo64);
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
    out << "  " << i_next << " = llvm.add " << lv << ", " << cn_step << " : i64\n";
    out << "  llvm.br ^" << hdr << "(" << i_next << " : i64)\n";
    out << "^" << exit_lbl << ":\n";
    if (cooperative_mode) emit_barrier(out);
}

void emit_apply_phase_static(std::ostringstream& out, uint32_t axis,
                              double phs_re, double phs_im) {
    uint64_t iters_val = (g_emit_ak > 0) ? (1ULL << (g_emit_ak - 1)) : 0;
    std::string iters = emit_const_i64(out, iters_val);
    char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", axis);
    std::string axis64 = fresh_ssa(), axis_bit = fresh_ssa();
    out << "  " << axis64 << " = llvm.mlir.constant(" << abuf << " : i64) : i64\n";
    out << "  " << axis_bit << " = llvm.shl %c1_i64, " << axis64 << " : i64\n";

    std::string hdr = fresh_label("ph_hdr");
    std::string body = fresh_label("ph_body");
    std::string exit_lbl = fresh_label("ph_exit");
    std::string lv = fresh_ssa();
    std::string ph_init = cooperative_mode ? "%tidx" : "%c0_i64";
    std::string ph_step = cooperative_mode ? "%c256_i64" : "%c1_i64";
    out << "  llvm.br ^" << hdr << "(" << ph_init << " : i64)\n";
    out << "^" << hdr << "(" << lv << ": i64):\n";
    std::string cond = fresh_ssa();
    out << "  " << cond << " = llvm.icmp \"ult\" " << lv << ", " << iters << " : i64\n";
    out << "  llvm.cond_br " << cond << ", ^" << body << ", ^" << exit_lbl << "\n";
    out << "^" << body << ":\n";
    std::string scattered = emit_scatter_bits_1(out, lv, axis64);
    std::string idx = fresh_ssa();
    out << "  " << idx << " = llvm.or " << scattered << ", " << axis_bit << " : i64\n";
    std::string vc = emit_load_v(out, idx);
    std::string phased = emit_cmul_const(out, vc, phs_re, phs_im);
    emit_store_v(out, idx, phased);
    std::string i_next = fresh_ssa();
    out << "  " << i_next << " = llvm.add " << lv << ", " << ph_step << " : i64\n";
    out << "  llvm.br ^" << hdr << "(" << i_next << " : i64)\n";
    out << "^" << exit_lbl << ":\n";
    if (cooperative_mode) emit_barrier(out);
}

// -----------------------------------------------------------------------
// GPU intrinsic helpers for coop/global kernels
// -----------------------------------------------------------------------

void emit_gpu_intrinsic_decls(std::ostringstream& out) {
    out << "llvm.func @llvm.amdgcn.workitem.id.x() -> i32\n";
    out << "llvm.func @llvm.amdgcn.workgroup.id.x() -> i32\n";
    // No llvm.log.f64 — AMDGCN needs device math library. Use inline approx instead.
    out << "llvm.func @llvm.amdgcn.s.barrier() -> ()\n";
    out << "llvm.func @llvm.amdgcn.ds.bpermute(i32, i32) -> i32\n\n";
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

// Warp shuffle: ds_bpermute XOR for i32
std::string emit_shfl_xor_i32(std::ostringstream& out,
                               const std::string& val_i32,
                               const std::string& lane_mask_i32) {
    std::string c63 = emit_const_i32(out, 63);
    std::string c2_shfl = emit_const_i32(out, 2);
    std::string self_lane = fresh_ssa();
    out << "  " << self_lane << " = llvm.and %tidx_i32, " << c63 << " : i32\n";
    std::string target = fresh_ssa();
    out << "  " << target << " = llvm.xor " << self_lane << ", " << lane_mask_i32 << " : i32\n";
    std::string byte_off = fresh_ssa();
    out << "  " << byte_off << " = llvm.shl " << target << ", " << c2_shfl << " : i32\n";
    std::string result = fresh_ssa();
    out << "  " << result << " = llvm.call @llvm.amdgcn.ds.bpermute("
        << byte_off << ", " << val_i32 << ") : (i32, i32) -> i32\n";
    return result;
}

// Warp shuffle XOR for i64 (split into two i32 halves)
std::string emit_shfl_xor_i64(std::ostringstream& out,
                               const std::string& val_i64,
                               const std::string& lane_mask_i32) {
    std::string c32 = emit_const_i64(out, 32);
    std::string lo = fresh_ssa();
    out << "  " << lo << " = llvm.trunc " << val_i64 << " : i64 to i32\n";
    std::string hi_shift = fresh_ssa();
    out << "  " << hi_shift << " = llvm.lshr " << val_i64 << ", " << c32 << " : i64\n";
    std::string hi = fresh_ssa();
    out << "  " << hi << " = llvm.trunc " << hi_shift << " : i64 to i32\n";
    std::string lo_shfl = emit_shfl_xor_i32(out, lo, lane_mask_i32);
    std::string hi_shfl = emit_shfl_xor_i32(out, hi, lane_mask_i32);
    std::string lo64 = fresh_ssa();
    out << "  " << lo64 << " = llvm.zext " << lo_shfl << " : i32 to i64\n";
    std::string hi64 = fresh_ssa();
    out << "  " << hi64 << " = llvm.zext " << hi_shfl << " : i32 to i64\n";
    std::string hi_placed = fresh_ssa();
    out << "  " << hi_placed << " = llvm.shl " << hi64 << ", " << c32 << " : i64\n";
    std::string combined = fresh_ssa();
    out << "  " << combined << " = llvm.or " << lo64 << ", " << hi_placed << " : i64\n";
    return combined;
}

std::string emit_shfl_xor_f64(std::ostringstream& out,
                               const std::string& val_f64,
                               const std::string& lane_mask_i32) {
    std::string bits = fresh_ssa();
    out << "  " << bits << " = llvm.bitcast " << val_f64 << " : f64 to i64\n";
    std::string shuffled_bits = emit_shfl_xor_i64(out, bits, lane_mask_i32);
    std::string shuffled = fresh_ssa();
    out << "  " << shuffled << " = llvm.bitcast " << shuffled_bits << " : i64 to f64\n";
    return shuffled;
}

std::pair<std::string, std::string> emit_coop_reduce2(
        std::ostringstream& out,
        const std::string& local0,
        const std::string& local1) {
    std::string acc0 = local0;
    std::string acc1 = local1;
    for (int offset : {32, 16, 8, 4, 2, 1}) {
        std::string mask = emit_const_i32(out, offset);
        std::string peer0 = emit_shfl_xor_f64(out, acc0, mask);
        std::string peer1 = emit_shfl_xor_f64(out, acc1, mask);
        std::string sum0 = fresh_ssa();
        std::string sum1 = fresh_ssa();
        out << "  " << sum0 << " = llvm.fadd " << acc0 << ", " << peer0 << " : f64\n";
        out << "  " << sum1 << " = llvm.fadd " << acc1 << ", " << peer1 << " : f64\n";
        acc0 = sum0;
        acc1 = sum1;
    }

    std::string red0 = emit_lds_base_ptr(out, "lds_mred0", "f64", 256);
    std::string red1 = emit_lds_base_ptr(out, "lds_mred1", "f64", 256);
    std::string lane = fresh_ssa();
    std::string c63 = emit_const_i32(out, 63);
    out << "  " << lane << " = llvm.and %tidx_i32, " << c63 << " : i32\n";
    std::string lane0 = fresh_ssa();
    out << "  " << lane0 << " = llvm.icmp \"eq\" " << lane << ", %c0_i32 : i32\n";
    std::string store_wave = fresh_label("mred_store_wave");
    std::string stored_wave = fresh_label("mred_stored_wave");
    out << "  llvm.cond_br " << lane0 << ", ^" << store_wave << ", ^" << stored_wave << "\n";
    out << "^" << store_wave << ":\n";
    std::string warp = fresh_ssa();
    std::string c6 = emit_const_i32(out, 6);
    out << "  " << warp << " = llvm.lshr %tidx_i32, " << c6 << " : i32\n";
    std::string warp64 = fresh_ssa();
    out << "  " << warp64 << " = llvm.zext " << warp << " : i32 to i64\n";
    std::string wave_p0 = fresh_ssa();
    std::string wave_p1 = fresh_ssa();
    out << "  " << wave_p0 << " = llvm.getelementptr inbounds " << red0 << "["
        << warp64 << "] : (!llvm.ptr, i64) -> !llvm.ptr, f64\n";
    out << "  " << wave_p1 << " = llvm.getelementptr inbounds " << red1 << "["
        << warp64 << "] : (!llvm.ptr, i64) -> !llvm.ptr, f64\n";
    out << "  llvm.store " << acc0 << ", " << wave_p0 << " : f64, !llvm.ptr\n";
    out << "  llvm.store " << acc1 << ", " << wave_p1 << " : f64, !llvm.ptr\n";
    out << "  llvm.br ^" << stored_wave << "\n";
    out << "^" << stored_wave << ":\n";
    emit_barrier(out);

    std::string tid_lt4 = fresh_ssa();
    std::string c4 = emit_const_i32(out, 4);
    out << "  " << tid_lt4 << " = llvm.icmp \"ult\" %tidx_i32, " << c4 << " : i32\n";
    std::string reduce_waves = fresh_label("mred_reduce_waves");
    std::string waves_reduced = fresh_label("mred_waves_reduced");
    out << "  llvm.cond_br " << tid_lt4 << ", ^" << reduce_waves << ", ^"
        << waves_reduced << "\n";
    out << "^" << reduce_waves << ":\n";
    std::string tid64 = fresh_ssa();
    out << "  " << tid64 << " = llvm.zext %tidx_i32 : i32 to i64\n";
    std::string partial_p0 = fresh_ssa();
    std::string partial_p1 = fresh_ssa();
    out << "  " << partial_p0 << " = llvm.getelementptr inbounds " << red0 << "["
        << tid64 << "] : (!llvm.ptr, i64) -> !llvm.ptr, f64\n";
    out << "  " << partial_p1 << " = llvm.getelementptr inbounds " << red1 << "["
        << tid64 << "] : (!llvm.ptr, i64) -> !llvm.ptr, f64\n";
    std::string wave0 = fresh_ssa();
    std::string wave1 = fresh_ssa();
    out << "  " << wave0 << " = llvm.load " << partial_p0 << " : !llvm.ptr -> f64\n";
    out << "  " << wave1 << " = llvm.load " << partial_p1 << " : !llvm.ptr -> f64\n";
    for (int offset : {2, 1}) {
        std::string mask = emit_const_i32(out, offset);
        std::string peer0 = emit_shfl_xor_f64(out, wave0, mask);
        std::string peer1 = emit_shfl_xor_f64(out, wave1, mask);
        std::string sum0 = fresh_ssa();
        std::string sum1 = fresh_ssa();
        out << "  " << sum0 << " = llvm.fadd " << wave0 << ", " << peer0 << " : f64\n";
        out << "  " << sum1 << " = llvm.fadd " << wave1 << ", " << peer1 << " : f64\n";
        wave0 = sum0;
        wave1 = sum1;
    }
    std::string is_tid0 = fresh_ssa();
    out << "  " << is_tid0 << " = llvm.icmp \"eq\" %tidx_i32, %c0_i32 : i32\n";
    std::string store_total = fresh_label("mred_store_total");
    std::string total_stored = fresh_label("mred_total_stored");
    out << "  llvm.cond_br " << is_tid0 << ", ^" << store_total << ", ^"
        << total_stored << "\n";
    out << "^" << store_total << ":\n";
    std::string total_p0 = fresh_ssa();
    std::string total_p1 = fresh_ssa();
    out << "  " << total_p0 << " = llvm.getelementptr inbounds " << red0
        << "[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, f64\n";
    out << "  " << total_p1 << " = llvm.getelementptr inbounds " << red1
        << "[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, f64\n";
    out << "  llvm.store " << wave0 << ", " << total_p0 << " : f64, !llvm.ptr\n";
    out << "  llvm.store " << wave1 << ", " << total_p1 << " : f64, !llvm.ptr\n";
    out << "  llvm.br ^" << total_stored << "\n";
    out << "^" << total_stored << ":\n";
    out << "  llvm.br ^" << waves_reduced << "\n";
    out << "^" << waves_reduced << ":\n";
    emit_barrier(out);

    std::string final_p0_ptr = fresh_ssa();
    std::string final_p1_ptr = fresh_ssa();
    out << "  " << final_p0_ptr << " = llvm.getelementptr inbounds " << red0
        << "[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, f64\n";
    out << "  " << final_p1_ptr << " = llvm.getelementptr inbounds " << red1
        << "[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, f64\n";
    std::string out0 = fresh_ssa();
    std::string out1 = fresh_ssa();
    out << "  " << out0 << " = llvm.load " << final_p0_ptr << " : !llvm.ptr -> f64\n";
    out << "  " << out1 << " = llvm.load " << final_p1_ptr << " : !llvm.ptr -> f64\n";
    return {out0, out1};
}

// Intra-wavefront reduction: reduce 64 threads to 1 value via ds_bpermute
// Offsets: 32, 16, 8, 4, 2, 1 (6 rounds for 64-wide wavefront)
std::string emit_wavefront_reduce_i64(std::ostringstream& out,
                                       const std::string& val_i64) {
    std::string acc = val_i64;
    for (int offset : {32, 16, 8, 4, 2, 1}) {
        std::string mask = emit_const_i32(out, offset);
        std::string peer = emit_shfl_xor_i64(out, acc, mask);
        std::string sum = fresh_ssa();
        out << "  " << sum << " = llvm.add " << acc << ", " << peer << " : i64\n";
        acc = sum;
    }
    return acc;
}

// Emit begin/end for tid-0 guard block in cooperative mode
std::string emit_tid0_guard_begin(std::ostringstream& out) {
    if (!cooperative_mode) return "";
    std::string lbl_t0 = fresh_label("t0_guard");
    std::string lbl_done = fresh_label("t0_done");
    std::string is_t0 = fresh_ssa();
    out << "  " << is_t0 << " = llvm.icmp \"eq\" %tidx_i32, %c0_i32 : i32\n";
    out << "  llvm.cond_br " << is_t0 << ", ^" << lbl_t0 << ", ^" << lbl_done << "\n";
    out << "^" << lbl_t0 << ":\n";
    return lbl_done;
}

void emit_tid0_guard_end(std::ostringstream& out, const std::string& lbl_done) {
    if (!cooperative_mode || lbl_done.empty()) return;
    out << "  llvm.br ^" << lbl_done << "\n";
    out << "^" << lbl_done << ":\n";
    emit_barrier(out);
}

// Close a tid0 guard WITHOUT a trailing barrier. Use when the guard is nested
// inside divergent control flow (e.g. a conditional/loop body) where a barrier
// would be illegal on AMDGPU (not all threads reach it). The caller is
// responsible for emitting a barrier at a later uniform point.
void emit_tid0_guard_end_nobarrier(std::ostringstream& out, const std::string& lbl_done) {
    if (!cooperative_mode || lbl_done.empty()) return;
    out << "  llvm.br ^" << lbl_done << "\n";
    out << "^" << lbl_done << ":\n";
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
        << " syncscope(\"agent\") monotonic : !llvm.ptr, i64\n";
    return old;
}

std::string emit_atomic_cmpxchg_i64(std::ostringstream& out,
                                     const std::string& ptr,
                                     const std::string& cmp,
                                     const std::string& new_val) {
    std::string res = fresh_ssa();
    out << "  " << res << " = llvm.cmpxchg " << ptr << ", " << cmp
        << ", " << new_val
        << " syncscope(\"agent\") monotonic monotonic : !llvm.ptr, i64\n";
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
    // Match gold cnorm (hip_sampler.hip): extend each f32 component to f64,
    // then square and sum in f64. Squaring in f32 first (then extending) loses
    // precision and flips borderline stabilizer measurement branches on
    // RNG-path-dependent shots.
    std::string re = fresh_ssa(), im = fresh_ssa();
    out << "  " << re << " = llvm.extractvalue " << c << "[0] : !llvm.struct<(f32, f32)>\n";
    out << "  " << im << " = llvm.extractvalue " << c << "[1] : !llvm.struct<(f32, f32)>\n";
    std::string re64 = fresh_ssa(), im64 = fresh_ssa();
    out << "  " << re64 << " = llvm.fpext " << re << " : f32 to f64\n";
    out << "  " << im64 << " = llvm.fpext " << im << " : f32 to f64\n";
    std::string re2 = fresh_ssa(), im2 = fresh_ssa();
    out << "  " << re2 << " = llvm.fmul " << re64 << ", " << re64 << " : f64\n";
    out << "  " << im2 << " = llvm.fmul " << im64 << ", " << im64 << " : f64\n";
    std::string norm_f64 = fresh_ssa();
    out << "  " << norm_f64 << " = llvm.fadd " << re2 << ", " << im2 << " : f64\n";
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

// Emit SVM-faithful sample_branch RNG behaviour and return the i1 "branch" bit
// (true = branch 1). SVM draws rng.uniform() ONLY when BOTH branches are
// non-dust; when a branch is dust it early-returns without drawing. To keep the
// RNG stream in lockstep with SVM we draw behind a branch on (both non-dust).
// The returned bit is only meaningful when both branches are non-dust; the
// caller's outer selects (on p0_small/p1_small) override it in the dust cases.
std::string emit_sample_rng_branch(std::ostringstream& out,
                                   const std::string& p0_small,
                                   const std::string& p1_small,
                                   const std::string& total,
                                   const std::string& final_p0) {
    // both_nondust = !p0_small & !p1_small
    std::string t_i1 = emit_const_i1(out, true);
    std::string n0 = fresh_ssa(), n1 = fresh_ssa(), both = fresh_ssa();
    out << "  " << n0 << " = llvm.xor " << p0_small << ", " << t_i1 << " : i1\n";
    out << "  " << n1 << " = llvm.xor " << p1_small << ", " << t_i1 << " : i1\n";
    out << "  " << both << " = llvm.and " << n0 << ", " << n1 << " : i1\n";
    std::string lbl_draw = fresh_label("rng_draw");
    std::string lbl_merge = fresh_label("rng_merge");
    std::string false_i1 = emit_const_i1(out, false);
    out << "  llvm.cond_br " << both << ", ^" << lbl_draw
        << ", ^" << lbl_merge << "(" << false_i1 << " : i1)\n";
    out << "^" << lbl_draw << ":\n";
    std::string u = emit_rng_uniform(out);
    std::string threshold = fresh_ssa();
    out << "  " << threshold << " = llvm.fmul " << u << ", " << total << " : f64\n";
    std::string rb = fresh_ssa();
    out << "  " << rb << " = llvm.fcmp \"oge\" " << threshold << ", " << final_p0 << " : f64\n";
    out << "  llvm.br ^" << lbl_merge << "(" << rb << " : i1)\n";
    std::string result = fresh_ssa();
    out << "^" << lbl_merge << "(" << result << ": i1):\n";
    return result;
}

std::string emit_sample_branch(std::ostringstream& out,
                               const std::string& prob0,
                               const std::string& prob1) {
    auto emit_local_sample = [&]() -> std::string {
        std::string total = fresh_ssa();
        out << "  " << total << " = llvm.fadd " << prob0 << ", " << prob1 << " : f64\n";
        std::string eps_k = fresh_ssa();
        out << "  " << eps_k << " = llvm.mlir.constant(1.0e-18 : f64) : f64\n";
        std::string eps = fresh_ssa();
        out << "  " << eps << " = llvm.fmul " << eps_k << ", " << total << " : f64\n";
        std::string p1_small = fresh_ssa();
        std::string p0_small = fresh_ssa();
        out << "  " << p1_small << " = llvm.fcmp \"ole\" " << prob1 << ", " << eps
            << " : f64\n";
        out << "  " << p0_small << " = llvm.fcmp \"ole\" " << prob0 << ", " << eps
            << " : f64\n";
        std::string rng_branch =
            emit_sample_rng_branch(out, p0_small, p1_small, total, prob0);
        std::string true_val = emit_const_i1(out, true);
        std::string false_val = emit_const_i1(out, false);
        std::string select_p0 = fresh_ssa();
        out << "  " << select_p0 << " = llvm.select " << p0_small << ", "
            << true_val << ", " << rng_branch << " : i1, i1\n";
        std::string branch = fresh_ssa();
        out << "  " << branch << " = llvm.select " << p1_small << ", "
            << false_val << ", " << select_p0 << " : i1, i1\n";
        return branch;
    };

    if (!cooperative_mode) {
        return emit_local_sample();
    }

    std::string branch_ptr = emit_lds_base_ptr(out, "lds_mbranch", "i8", 1);
    std::string is_tid0 = fresh_ssa();
    out << "  " << is_tid0 << " = llvm.icmp \"eq\" %tidx_i32, %c0_i32 : i32\n";
    std::string sample = fresh_label("meas_sample");
    std::string sampled = fresh_label("meas_sampled");
    out << "  llvm.cond_br " << is_tid0 << ", ^" << sample << ", ^" << sampled << "\n";
    out << "^" << sample << ":\n";
    std::string branch = emit_local_sample();
    std::string branch_i8 = fresh_ssa();
    out << "  " << branch_i8 << " = llvm.zext " << branch << " : i1 to i8\n";
    out << "  llvm.store " << branch_i8 << ", " << branch_ptr << " : i8, !llvm.ptr\n";
    out << "  llvm.br ^" << sampled << "\n";
    out << "^" << sampled << ":\n";
    emit_barrier(out);
    std::string broadcast = fresh_ssa();
    out << "  " << broadcast << " = llvm.load " << branch_ptr << " : !llvm.ptr -> i8\n";
    std::string result = fresh_ssa();
    out << "  " << result << " = llvm.trunc " << broadcast << " : i8 to i1\n";
    return result;
}

void emit_meas_dormant_random(std::ostringstream& out,
                               uint32_t axis, uint32_t classical_idx, bool sign) {
    // COOP: the scalar RNG draw MUST be tid0-only (matches gold
    // hip_sampler.hip:1646). Every coop scalar RNG draw is tid0-only so tid0's
    // stream stays aligned with the SVM single-thread stream. A redundant-full
    // draw here would desync tid0 (already ahead from tid0-only sample_branch
    // draws) vs the other 255 threads and corrupt shots >=1. The entire draw +
    // frame/meas mutation runs inside one tid0 guard, exactly like gold.
    char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", axis);
    std::string mg = cooperative_mode ? emit_tid0_guard_begin(out) : "";
    std::string u = emit_rng_uniform(out);
    std::string half = fresh_ssa();
    out << "  " << half << " = llvm.mlir.constant(0.5 : f64) : f64\n";
    std::string cmp = fresh_ssa();
    out << "  " << cmp << " = llvm.fcmp \"olt\" " << u << ", " << half << " : f64\n";
    // cmp = (u < 0.5); gold: m_abs = (u < 0.5) ? 0 : 1 = !cmp
    std::string true_i1 = emit_const_i1(out, true);
    std::string not_cmp = fresh_ssa();
    out << "  " << not_cmp << " = llvm.xor " << cmp << ", " << true_i1 << " : i1\n";
    std::string m_abs = fresh_ssa();
    out << "  " << m_abs << " = llvm.zext " << not_cmp << " : i1 to i8\n";
    std::string meas_val = m_abs;
    if (sign) {
        std::string xored = fresh_ssa();
        out << "  " << xored << " = llvm.xor " << m_abs << ", %c1_i8 : i8\n";
        meas_val = xored;
    }
    std::string axis_i32 = fresh_ssa();
    out << "  " << axis_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
    emit_bit_set(out, "%px_ptr", axis_i32, not_cmp);
    std::string false_i1 = fresh_ssa();
    out << "  " << false_i1 << " = llvm.mlir.constant(false) : i1\n";
    emit_bit_set(out, "%pz_ptr", axis_i32, false_i1);
    char cidx[32]; snprintf(cidx, sizeof(cidx), "%u", classical_idx);
    std::string cidx_i64 = fresh_ssa();
    out << "  " << cidx_i64 << " = llvm.mlir.constant(" << cidx << " : i64) : i64\n";
    std::string meas_p = fresh_ssa();
    out << "  " << meas_p << " = llvm.getelementptr inbounds %meas_ptr["
        << cidx_i64 << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
    out << "  llvm.store " << meas_val << ", " << meas_p << " : i8, !llvm.ptr\n";
    if (cooperative_mode) emit_tid0_guard_end(out, mg);
}

void emit_meas_active_diagonal(std::ostringstream& out,
                                uint32_t axis, uint32_t classical_idx, bool sign) {
    // half = 1 << (active_k - 1), active_k known at compile time
    uint64_t half_val = (g_emit_ak > 0) ? (1ULL << (g_emit_ak - 1)) : 0;
    std::string half = emit_const_i64(out, half_val);

    // Read px bit at axis
    char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", axis);
    std::string axis_i32 = fresh_ssa();
    out << "  " << axis_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
    std::string px_bit = emit_bit_get(out, "%px_ptr", axis_i32);

    // Sum probabilities with f64 block argument accumulators
    std::string p_hdr2 = fresh_label("psum_hdr");
    std::string p_body2 = fresh_label("psum_body");
    std::string p_done2 = fresh_label("psum_done");
    std::string psum_i_var = fresh_ssa(), psum_p0_var = fresh_ssa(), psum_p1_var = fresh_ssa();
    std::string final_p0_var = fresh_ssa(), final_p1_var = fresh_ssa();
    std::string f0 = fresh_ssa();
    out << "  " << f0 << " = llvm.mlir.constant(0.0 : f64) : f64\n";
    std::string psum_init = cooperative_mode ? "%tidx" : "%c0_i64";
    std::string psum_step = cooperative_mode ? "%c256_i64" : "%c1_i64";
    out << "  llvm.br ^" << p_hdr2 << "(" << psum_init << ", " << f0 << ", " << f0
        << " : i64, f64, f64)\n";
    out << "^" << p_hdr2 << "(" << psum_i_var << ": i64, " << psum_p0_var << ": f64, " << psum_p1_var << ": f64):\n";
    std::string pcond = fresh_ssa();
    out << "  " << pcond << " = llvm.icmp \"ult\" " << psum_i_var << ", " << half << " : i64\n";
    out << "  llvm.cond_br " << pcond << ", ^" << p_body2
        << ", ^" << p_done2 << "(" << psum_p0_var << ", " << psum_p1_var << " : f64, f64)\n";
    out << "^" << p_body2 << ":\n";
    std::string vi = emit_load_v(out, psum_i_var);
    std::string idx_hi = fresh_ssa();
    out << "  " << idx_hi << " = llvm.add " << psum_i_var << ", " << half << " : i64\n";
    std::string vh = emit_load_v(out, idx_hi);
    std::string n0 = emit_cnorm(out, vi);
    std::string n1 = emit_cnorm(out, vh);
    std::string new_p0 = fresh_ssa(), new_p1 = fresh_ssa();
    out << "  " << new_p0 << " = llvm.fadd " << psum_p0_var << ", " << n0 << " : f64\n";
    out << "  " << new_p1 << " = llvm.fadd " << psum_p1_var << ", " << n1 << " : f64\n";
    std::string next_i = fresh_ssa();
    out << "  " << next_i << " = llvm.add " << psum_i_var << ", " << psum_step
        << " : i64\n";
    out << "  llvm.br ^" << p_hdr2 << "(" << next_i << ", " << new_p0 << ", " << new_p1 << " : i64, f64, f64)\n";
    out << "^" << p_done2 << "(" << final_p0_var << ": f64, " << final_p1_var << ": f64):\n";

    if (cooperative_mode) {
        auto totals = emit_coop_reduce2(out, final_p0_var, final_p1_var);
        final_p0_var = totals.first;
        final_p1_var = totals.second;
    }
    std::string b_final = emit_sample_branch(out, final_p0_var, final_p1_var);

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
    // If b != 0: copy v[i+half] → v[i] for i in [0, half)
    std::string lbl_copy = fresh_label("copy");
    std::string lbl_nocopy = fresh_label("nocopy");
    out << "  llvm.cond_br " << b_final << ", ^" << lbl_copy << ", ^" << lbl_nocopy << "\n";
    out << "^" << lbl_copy << ":\n";
    std::string cp_var = fresh_ssa();
    std::string cp_hdr = fresh_label("cp_hdr"), cp_body = fresh_label("cp_body"), cp_done = fresh_label("cp_done");
    std::string copy_init = cooperative_mode ? "%tidx" : "%c0_i64";
    std::string copy_step = cooperative_mode ? "%c256_i64" : "%c1_i64";
    out << "  llvm.br ^" << cp_hdr << "(" << copy_init << " : i64)\n";
    out << "^" << cp_hdr << "(" << cp_var << ": i64):\n";
    std::string cp_cond = fresh_ssa();
    out << "  " << cp_cond << " = llvm.icmp \"ult\" " << cp_var << ", " << half << " : i64\n";
    out << "  llvm.cond_br " << cp_cond << ", ^" << cp_body << ", ^" << cp_done << "\n";
    out << "^" << cp_body << ":\n";
    std::string cp_src = fresh_ssa();
    out << "  " << cp_src << " = llvm.add " << cp_var << ", " << half << " : i64\n";
    std::string cp_val = emit_load_v(out, cp_src);
    emit_store_v(out, cp_var, cp_val);
    std::string cp_next = fresh_ssa();
    out << "  " << cp_next << " = llvm.add " << cp_var << ", " << copy_step << " : i64\n";
    out << "  llvm.br ^" << cp_hdr << "(" << cp_next << " : i64)\n";
    out << "^" << cp_done << ":\n";
    out << "  llvm.br ^" << lbl_nocopy << "\n";
    out << "^" << lbl_nocopy << ":\n";

    // Barrier: all threads finish the redundant compaction before next op reads.
    if (cooperative_mode) emit_barrier(out);

    // active_k-- and frame bit updates are single-writer shared state. In coop
    // they MUST be tid0-only: emit_bit_set does a read-modify-write on a shared
    // LDS word holding MANY qubits' bits; 256 concurrent RMWs interleave and
    // clobber unrelated bits -> corrupted Pauli frame -> wrong measurements
    // (a genuine, seed-dependent race). GPU-SVM does these on thread 0.
    {
        std::string mg = cooperative_mode ? emit_tid0_guard_begin(out) : "";
        std::string new_ak_diag = emit_const_i32(out, g_emit_ak - 1);
        out << "  llvm.store " << new_ak_diag << ", %active_k_ptr : i32, !llvm.ptr\n";
        std::string cidx_i64 = emit_const_i64(out, classical_idx);
        std::string meas_p = fresh_ssa();
        out << "  " << meas_p << " = llvm.getelementptr inbounds %meas_ptr["
            << cidx_i64 << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        out << "  llvm.store " << meas_val << ", " << meas_p << " : i8, !llvm.ptr\n";
        std::string m_ne0 = fresh_ssa();
        out << "  " << m_ne0 << " = llvm.icmp \"ne\" " << m_abs << ", %c0_i8 : i8\n";
        emit_bit_set(out, "%px_ptr", axis_i32, m_ne0);
        std::string false_val = emit_const_i1(out, false);
        emit_bit_set(out, "%pz_ptr", axis_i32, false_val);
        if (cooperative_mode) emit_tid0_guard_end(out, mg);
    }
}

void emit_meas_active_interfere(std::ostringstream& out,
                                 uint32_t axis, uint32_t classical_idx, bool sign) {
    uint64_t half_val = (g_emit_ak > 0) ? (1ULL << (g_emit_ak - 1)) : 0;
    std::string half = emit_const_i64(out, half_val);

    char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", axis);
    std::string axis_i32 = fresh_ssa();
    out << "  " << axis_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
    std::string pz_bit = emit_bit_get(out, "%pz_ptr", axis_i32);

    // Sum p_plus = sum(cnorm(v[i]+v[i+half])), p_minus = sum(cnorm(v[i]-v[i+half]))
    std::string pi_hdr = fresh_label("pi_hdr"), pi_body = fresh_label("pi_body"), pi_done = fresh_label("pi_done");
    std::string pi_i_var = fresh_ssa(), pi_pp_var = fresh_ssa(), pi_pm_var = fresh_ssa();
    std::string fp_plus_var = fresh_ssa(), fp_minus_var = fresh_ssa();
    std::string fz = fresh_ssa();
    out << "  " << fz << " = llvm.mlir.constant(0.0 : f64) : f64\n";
    std::string psum_init = cooperative_mode ? "%tidx" : "%c0_i64";
    std::string psum_step = cooperative_mode ? "%c256_i64" : "%c1_i64";
    out << "  llvm.br ^" << pi_hdr << "(" << psum_init << ", " << fz << ", " << fz
        << " : i64, f64, f64)\n";
    out << "^" << pi_hdr << "(" << pi_i_var << ": i64, " << pi_pp_var << ": f64, " << pi_pm_var << ": f64):\n";
    std::string pi_cond = fresh_ssa();
    out << "  " << pi_cond << " = llvm.icmp \"ult\" " << pi_i_var << ", " << half << " : i64\n";
    out << "  llvm.cond_br " << pi_cond << ", ^" << pi_body
        << ", ^" << pi_done << "(" << pi_pp_var << ", " << pi_pm_var << " : f64, f64)\n";
    out << "^" << pi_body << ":\n";
    std::string vi = emit_load_v(out, pi_i_var);
    std::string hi_idx = fresh_ssa();
    out << "  " << hi_idx << " = llvm.add " << pi_i_var << ", " << half << " : i64\n";
    std::string vh = emit_load_v(out, hi_idx);
    std::string sum_c = emit_cadd(out, vi, vh);
    std::string diff_c = emit_csub(out, vi, vh);
    std::string np = emit_cnorm(out, sum_c);
    std::string nm = emit_cnorm(out, diff_c);
    std::string new_pp = fresh_ssa(), new_pm = fresh_ssa();
    out << "  " << new_pp << " = llvm.fadd " << pi_pp_var << ", " << np << " : f64\n";
    out << "  " << new_pm << " = llvm.fadd " << pi_pm_var << ", " << nm << " : f64\n";
    std::string pi_next = fresh_ssa();
    out << "  " << pi_next << " = llvm.add " << pi_i_var << ", " << psum_step
        << " : i64\n";
    out << "  llvm.br ^" << pi_hdr << "(" << pi_next << ", " << new_pp << ", " << new_pm << " : i64, f64, f64)\n";
    out << "^" << pi_done << "(" << fp_plus_var << ": f64, " << fp_minus_var << ": f64):\n";

    if (cooperative_mode) {
        auto totals = emit_coop_reduce2(out, fp_plus_var, fp_minus_var);
        fp_plus_var = totals.first;
        fp_minus_var = totals.second;
    }
    std::string bx_final = emit_sample_branch(out, fp_plus_var, fp_minus_var);

    // m_abs = b_x ^ pz
    std::string bx_i8 = fresh_ssa();
    out << "  " << bx_i8 << " = llvm.zext " << bx_final << " : i1 to i8\n";
    std::string pz_i8 = fresh_ssa();
    out << "  " << pz_i8 << " = llvm.zext " << pz_bit << " : i1 to i8\n";
    std::string m_abs = fresh_ssa();
    out << "  " << m_abs << " = llvm.xor " << bx_i8 << ", " << pz_i8 << " : i8\n";
    std::string meas_val = m_abs;
    if (sign) {
        std::string xored = fresh_ssa();
        out << "  " << xored << " = llvm.xor " << m_abs << ", %c1_i8 : i8\n";
        meas_val = xored;
    }
    // Fold amplitudes: v[i] = cscale(b_x ? csub(v[i],v[i+half]) : cadd(v[i],v[i+half]), inv_sqrt2)
    std::string fold_var = fresh_ssa();
    std::string fold_hdr = fresh_label("fold_hdr"), fold_body = fresh_label("fold_body"), fold_done = fresh_label("fold_done");
    std::string fold_init = cooperative_mode ? "%tidx" : "%c0_i64";
    std::string fold_step = cooperative_mode ? "%c256_i64" : "%c1_i64";
    out << "  llvm.br ^" << fold_hdr << "(" << fold_init << " : i64)\n";
    out << "^" << fold_hdr << "(" << fold_var << ": i64):\n";
    std::string fold_cond = fresh_ssa();
    out << "  " << fold_cond << " = llvm.icmp \"ult\" " << fold_var << ", " << half << " : i64\n";
    out << "  llvm.cond_br " << fold_cond << ", ^" << fold_body << ", ^" << fold_done << "\n";
    out << "^" << fold_body << ":\n";
    std::string fvi = emit_load_v(out, fold_var);
    std::string fhi = fresh_ssa();
    out << "  " << fhi << " = llvm.add " << fold_var << ", " << half << " : i64\n";
    std::string fvh = emit_load_v(out, fhi);
    std::string fsum = emit_cadd(out, fvi, fvh);
    std::string fdiff = emit_csub(out, fvi, fvh);
    std::string folded = fresh_ssa();
    out << "  " << folded << " = llvm.select " << bx_final << ", " << fdiff << ", " << fsum
        << " : i1, !llvm.struct<(f32, f32)>\n";
    std::string scaled = emit_cscale_f64(out, folded, kInvSqrt2);
    emit_store_v(out, fold_var, scaled);
    std::string fold_next = fresh_ssa();
    out << "  " << fold_next << " = llvm.add " << fold_var << ", " << fold_step
        << " : i64\n";
    out << "  llvm.br ^" << fold_hdr << "(" << fold_next << " : i64)\n";
    out << "^" << fold_done << ":\n";
    if (cooperative_mode) emit_barrier(out);

    // active_k-- and frame bit updates: tid0-only in coop (shared-word RMW race).
    {
        std::string mg = cooperative_mode ? emit_tid0_guard_begin(out) : "";
        std::string new_ak_ifc = emit_const_i32(out, g_emit_ak - 1);
        out << "  llvm.store " << new_ak_ifc << ", %active_k_ptr : i32, !llvm.ptr\n";
        std::string cidx_i64 = emit_const_i64(out, classical_idx);
        std::string meas_p = fresh_ssa();
        out << "  " << meas_p << " = llvm.getelementptr inbounds %meas_ptr["
            << cidx_i64 << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        out << "  llvm.store " << meas_val << ", " << meas_p << " : i8, !llvm.ptr\n";
        std::string m_ne0 = fresh_ssa();
        out << "  " << m_ne0 << " = llvm.icmp \"ne\" " << m_abs << ", %c0_i8 : i8\n";
        emit_bit_set(out, "%px_ptr", axis_i32, m_ne0);
        std::string false_val2 = emit_const_i1(out, false);
        emit_bit_set(out, "%pz_ptr", axis_i32, false_val2);
        if (cooperative_mode) emit_tid0_guard_end(out, mg);
    }
}

void emit_inline_log(std::ostringstream& out, const std::string& x, std::string& result) {
    // log(x) via IEEE 754 decomposition and an atanh series through 1/29.
    // The previous truncation at 1/13 had errors near 1e-8, large enough to
    // move gap-sampling targets across cumulative-noise-hazard boundaries.
    std::string x_bits = fresh_ssa();
    out << "  " << x_bits << " = llvm.bitcast " << x << " : f64 to i64\n";
    std::string c52 = emit_const_i64(out, 52);
    std::string exp_raw = fresh_ssa();
    out << "  " << exp_raw << " = llvm.lshr " << x_bits << ", " << c52 << " : i64\n";
    std::string c1023 = emit_const_i64(out, 1023);
    std::string exp_val = fresh_ssa();
    out << "  " << exp_val << " = llvm.sub " << exp_raw << ", " << c1023 << " : i64\n";
    std::string exp_f64 = fresh_ssa();
    out << "  " << exp_f64 << " = llvm.sitofp " << exp_val << " : i64 to f64\n";
    std::string ln2 = fresh_ssa();
    out << "  " << ln2 << " = llvm.mlir.constant(0.6931471805599453 : f64) : f64\n";
    std::string exp_part = fresh_ssa();
    out << "  " << exp_part << " = llvm.fmul " << exp_f64 << ", " << ln2 << " : f64\n";
    std::string exp_1023_bits = emit_const_i64(out, 0x3FF0000000000000ULL);
    std::string not_exp_mask = emit_const_i64(out, ~0x7FF0000000000000ULL);
    std::string bits_no_exp = fresh_ssa();
    out << "  " << bits_no_exp << " = llvm.and " << x_bits << ", " << not_exp_mask << " : i64\n";
    std::string bits_m = fresh_ssa();
    out << "  " << bits_m << " = llvm.or " << bits_no_exp << ", " << exp_1023_bits << " : i64\n";
    std::string m_val = fresh_ssa();
    out << "  " << m_val << " = llvm.bitcast " << bits_m << " : i64 to f64\n";
    std::string one = fresh_ssa();
    out << "  " << one << " = llvm.mlir.constant(1.0 : f64) : f64\n";
    std::string m_m1 = fresh_ssa();
    out << "  " << m_m1 << " = llvm.fsub " << m_val << ", " << one << " : f64\n";
    std::string m_p1 = fresh_ssa();
    out << "  " << m_p1 << " = llvm.fadd " << m_val << ", " << one << " : f64\n";
    std::string g = fresh_ssa();
    out << "  " << g << " = llvm.fdiv " << m_m1 << ", " << m_p1 << " : f64\n";
    std::string g2 = fresh_ssa();
    out << "  " << g2 << " = llvm.fmul " << g << ", " << g << " : f64\n";

    std::string series = fresh_ssa();
    out << "  " << series
        << " = llvm.mlir.constant(3.44827586206896547e-02 : f64) : f64\n";
    for (int denominator = 27; denominator >= 3; denominator -= 2) {
        std::string product = fresh_ssa();
        out << "  " << product << " = llvm.fmul " << g2 << ", " << series << " : f64\n";
        std::string coefficient = fresh_ssa();
        char buf[64];
        snprintf(buf, sizeof(buf), "%.17e", 1.0 / denominator);
        out << "  " << coefficient << " = llvm.mlir.constant(" << buf << " : f64) : f64\n";
        std::string next = fresh_ssa();
        out << "  " << next << " = llvm.fadd " << product << ", " << coefficient << " : f64\n";
        series = next;
    }
    std::string series_tail = fresh_ssa();
    out << "  " << series_tail << " = llvm.fmul " << g2 << ", " << series << " : f64\n";
    std::string atanh_factor = fresh_ssa();
    out << "  " << atanh_factor << " = llvm.fadd " << one << ", " << series_tail << " : f64\n";
    std::string two = fresh_ssa();
    out << "  " << two << " = llvm.mlir.constant(2.0 : f64) : f64\n";
    std::string two_g = fresh_ssa();
    out << "  " << two_g << " = llvm.fmul " << two << ", " << g << " : f64\n";
    std::string mantissa_part = fresh_ssa();
    out << "  " << mantissa_part << " = llvm.fmul " << two_g << ", " << atanh_factor << " : f64\n";
    result = fresh_ssa();
    out << "  " << result << " = llvm.fadd " << exp_part << ", " << mantissa_part << " : f64\n";
}

void emit_draw_next_noise(std::ostringstream& out,
                           const FlattenedProgram& flat) {
    uint32_t n = flat.noise_sites.size();
    if (n == 0) {
        std::string sentinel = emit_const_i32(out, 0xFFFFFFFFu);
        out << "  llvm.store " << sentinel << ", %nni_ptr : i32, !llvm.ptr\n";
        return;
    }

    // current_hazard = (nni == 0) ? 0.0 : hazards[nni - 1]
    std::string nni = fresh_ssa();
    out << "  " << nni << " = llvm.load %nni_ptr : !llvm.ptr -> i32\n";
    std::string nni64 = fresh_ssa();
    out << "  " << nni64 << " = llvm.zext " << nni << " : i32 to i64\n";

    // CRITICAL: GPU-SVM draw_next_noise (hip_sampler.hip:253) EARLY-RETURNS
    // without drawing rng.uniform() when next_noise_idx >= num_noise_sites.
    // MLIR must match this exactly or it draws one extra RNG value once noise
    // is exhausted, desyncing every downstream random measurement/observable.
    // Guard: if (nni >= num_noise_sites) { store sentinel; skip draw. }
    std::string dn_exhausted = fresh_ssa();
    out << "  " << dn_exhausted << " = llvm.icmp \"uge\" " << nni64
        << ", %num_noise_sites : i64\n";
    std::string dn_skip = fresh_label("dn_skip");
    std::string dn_do = fresh_label("dn_do");
    std::string dn_end = fresh_label("dn_end");
    out << "  llvm.cond_br " << dn_exhausted << ", ^" << dn_skip << ", ^" << dn_do << "\n";
    out << "^" << dn_skip << ":\n";
    {
        std::string sent = emit_const_i32(out, 0xFFFFFFFFu);
        out << "  llvm.store " << sent << ", %nni_ptr : i32, !llvm.ptr\n";
    }
    out << "  llvm.br ^" << dn_end << "\n";
    out << "^" << dn_do << ":\n";

    std::string nni_is_0 = fresh_ssa();
    out << "  " << nni_is_0 << " = llvm.icmp \"eq\" " << nni << ", %c0_i32 : i32\n";
    std::string ch_zero = fresh_ssa();
    out << "  " << ch_zero << " = llvm.mlir.constant(0.0 : f64) : f64\n";
    std::string ch_nonzero = fresh_label("haz_nonzero");
    std::string ch_done = fresh_label("haz_done");
    std::string current_hazard = fresh_ssa();
    out << "  llvm.cond_br " << nni_is_0 << ", ^" << ch_done << "(" << ch_zero
        << " : f64), ^" << ch_nonzero << "\n";
    out << "^" << ch_nonzero << ":\n";
    // Load hazards[nni-1] only when the index is valid.
    std::string nni_m1 = fresh_ssa();
    out << "  " << nni_m1 << " = llvm.sub " << nni64 << ", %c1_i64 : i64\n";
    std::string haz_ptr = fresh_ssa();
    out << "  " << haz_ptr << " = llvm.getelementptr inbounds %noise_hazards_ptr["
        << nni_m1 << "] : (!llvm.ptr, i64) -> !llvm.ptr, f64\n";
    std::string haz_val = fresh_ssa();
    out << "  " << haz_val << " = llvm.load " << haz_ptr << " : !llvm.ptr -> f64\n";
    out << "  llvm.br ^" << ch_done << "(" << haz_val << " : f64)\n";
    out << "^" << ch_done << "(" << current_hazard << ": f64):\n";

    // gap = -log(1-u)
    std::string u = emit_rng_uniform(out);
    std::string f64_one = fresh_ssa();
    out << "  " << f64_one << " = llvm.mlir.constant(1.0 : f64) : f64\n";
    std::string one_minus_u = fresh_ssa();
    out << "  " << one_minus_u << " = llvm.fsub " << f64_one << ", " << u << " : f64\n";
    std::string log_val;
    emit_inline_log(out, one_minus_u, log_val);
    std::string gap = fresh_ssa();
    out << "  " << gap << " = llvm.fneg " << log_val << " : f64\n";
    std::string target = fresh_ssa();
    out << "  " << target << " = llvm.fadd " << current_hazard << ", " << gap << " : f64\n";

    // Binary search over noise_hazards_ptr[0..num_noise_sites)
    // upper_bound: find first i where hazards[i] > target
    std::string ns_i32 = fresh_ssa();
    out << "  " << ns_i32 << " = llvm.add %num_noise_sites, %c0_i64 : i64\n";  // copy to fresh SSA
    std::string lo_var = fresh_ssa(), hi_var = fresh_ssa();
    std::string bs_hdr = fresh_label("bs_hdr"), bs_body = fresh_label("bs_body"), bs_done = fresh_label("bs_done");
    out << "  llvm.br ^" << bs_hdr << "(%c0_i64, " << ns_i32 << " : i64, i64)\n";
    out << "^" << bs_hdr << "(" << lo_var << ": i64, " << hi_var << ": i64):\n";
    std::string bs_cond = fresh_ssa();
    out << "  " << bs_cond << " = llvm.icmp \"ult\" " << lo_var << ", " << hi_var << " : i64\n";
    out << "  llvm.cond_br " << bs_cond << ", ^" << bs_body << ", ^" << bs_done << "(" << lo_var << " : i64)\n";
    out << "^" << bs_body << ":\n";
    std::string diff = fresh_ssa();
    out << "  " << diff << " = llvm.sub " << hi_var << ", " << lo_var << " : i64\n";
    std::string half = fresh_ssa();
    out << "  " << half << " = llvm.lshr " << diff << ", %c1_i64 : i64\n";
    std::string mid = fresh_ssa();
    out << "  " << mid << " = llvm.add " << lo_var << ", " << half << " : i64\n";
    std::string mid_ptr = fresh_ssa();
    out << "  " << mid_ptr << " = llvm.getelementptr inbounds %noise_hazards_ptr["
        << mid << "] : (!llvm.ptr, i64) -> !llvm.ptr, f64\n";
    std::string mid_val = fresh_ssa();
    out << "  " << mid_val << " = llvm.load " << mid_ptr << " : !llvm.ptr -> f64\n";
    std::string go_right = fresh_ssa();
    out << "  " << go_right << " = llvm.fcmp \"ole\" " << mid_val << ", " << target << " : f64\n";
    // if hazards[mid] <= target: lo = mid + 1, else: hi = mid
    std::string mid_p1 = fresh_ssa();
    out << "  " << mid_p1 << " = llvm.add " << mid << ", %c1_i64 : i64\n";
    std::string new_lo = fresh_ssa();
    out << "  " << new_lo << " = llvm.select " << go_right << ", " << mid_p1 << ", " << lo_var << " : i1, i64\n";
    std::string new_hi = fresh_ssa();
    out << "  " << new_hi << " = llvm.select " << go_right << ", " << hi_var << ", " << mid << " : i1, i64\n";
    out << "  llvm.br ^" << bs_hdr << "(" << new_lo << ", " << new_hi << " : i64, i64)\n";
    std::string bs_result_var = fresh_ssa();
    out << "^" << bs_done << "(" << bs_result_var << ": i64):\n";
    // If result >= num_noise_sites, set to sentinel 0xFFFFFFFF
    std::string bs_ge = fresh_ssa();
    out << "  " << bs_ge << " = llvm.icmp \"uge\" " << bs_result_var << ", " << ns_i32 << " : i64\n";
    std::string sentinel = emit_const_i32(out, 0xFFFFFFFFu);
    std::string result_i32 = fresh_ssa();
    out << "  " << result_i32 << " = llvm.trunc " << bs_result_var << " : i64 to i32\n";
    std::string final_nni = fresh_ssa();
    out << "  " << final_nni << " = llvm.select " << bs_ge << ", " << sentinel << ", " << result_i32 << " : i1, i32\n";
    out << "  llvm.store " << final_nni << ", %nni_ptr : i32, !llvm.ptr\n";
    out << "  llvm.br ^" << dn_end << "\n";
    out << "^" << dn_end << ":\n";
}

// -----------------------------------------------------------------------
// Fused SWAP + MEAS_ACTIVE_INTERFERE
// -----------------------------------------------------------------------

void emit_swap_meas_interfere(std::ostringstream& out,
                               uint32_t swap_from, uint32_t swap_to,
                               uint32_t classical_idx, bool sign) {
    out << "  // swap_meas_interfere: from=" << swap_from << " to=" << swap_to
        << " meas=" << classical_idx << " (active_k=" << g_emit_ak << ")\n";

    if (swap_from == swap_to) {
        emit_meas_active_interfere(out, swap_to, classical_idx, sign);
        return;
    }

    // Frame swap (thread-0 only in coop mode)
    char from_buf[32], to_buf[32];
    snprintf(from_buf, sizeof(from_buf), "%u", swap_from);
    snprintf(to_buf, sizeof(to_buf), "%u", swap_to);
    {
        std::string g = cooperative_mode ? emit_tid0_guard_begin(out) : "";
        std::string fi32 = fresh_ssa(), ti32 = fresh_ssa();
        out << "  " << fi32 << " = llvm.mlir.constant(" << from_buf << " : i32) : i32\n";
        out << "  " << ti32 << " = llvm.mlir.constant(" << to_buf << " : i32) : i32\n";
        std::string px_f = emit_bit_get(out, "%px_ptr", fi32);
        std::string px_t = emit_bit_get(out, "%px_ptr", ti32);
        emit_bit_set(out, "%px_ptr", fi32, px_t);
        emit_bit_set(out, "%px_ptr", ti32, px_f);
        std::string pz_f = emit_bit_get(out, "%pz_ptr", fi32);
        std::string pz_t = emit_bit_get(out, "%pz_ptr", ti32);
        emit_bit_set(out, "%pz_ptr", fi32, pz_t);
        emit_bit_set(out, "%pz_ptr", ti32, pz_f);
        if (cooperative_mode) { emit_tid0_guard_end(out, g); emit_barrier(out); }
    }

    // Fused swap+interfere measurement on axis 'to' (post-swap):
    // half = 1 << to, f_bit = 1 << from
    // Prob sum: for idx in [0, half): base = (idx & ~f_bit) | ((idx>>from)&1)<<to
    //   sum = v[base] + v[base|f_bit], diff = v[base] - v[base|f_bit]
    //   p_plus += |sum|^2, p_minus += |diff|^2
    // Then fold: v[idx] = inv_sqrt2 * (b==0 ? sum : diff)
    // Then active_k--, frame update on 'to' axis

    uint64_t half_val = 1ULL << swap_to;
    std::string half = emit_const_i64(out, half_val);
    std::string f64 = fresh_ssa(), f_bit = fresh_ssa();
    out << "  " << f64 << " = llvm.mlir.constant(" << from_buf << " : i64) : i64\n";
    out << "  " << f_bit << " = llvm.shl %c1_i64, " << f64 << " : i64\n";
    std::string t64 = fresh_ssa();
    out << "  " << t64 << " = llvm.mlir.constant(" << to_buf << " : i64) : i64\n";

    // Read pz bit at 'to' axis
    std::string to_i32 = fresh_ssa();
    out << "  " << to_i32 << " = llvm.mlir.constant(" << to_buf << " : i32) : i32\n";
    std::string pz_bit = emit_bit_get(out, "%pz_ptr", to_i32);

    // Precompute loop-invariant values before any loop
    std::string not_fbit = fresh_ssa();
    out << "  " << not_fbit << " = llvm.xor " << f_bit << ", %cminus1_i64 : i64\n";

    // Probability summation
    std::string p_hdr = fresh_label("smip_hdr"), p_body = fresh_label("smip_body"), p_done = fresh_label("smip_done");
    std::string p_i = fresh_ssa(), p_plus_var = fresh_ssa(), p_minus_var = fresh_ssa();
    std::string fp_plus = fresh_ssa(), fp_minus = fresh_ssa();
    std::string fz = fresh_ssa();
    out << "  " << fz << " = llvm.mlir.constant(0.0 : f64) : f64\n";
    std::string loop_init = cooperative_mode ? "%tidx" : "%c0_i64";
    std::string loop_step = cooperative_mode ? "%c256_i64" : "%c1_i64";
    out << "  llvm.br ^" << p_hdr << "(" << loop_init << ", " << fz << ", " << fz << " : i64, f64, f64)\n";
    out << "^" << p_hdr << "(" << p_i << ": i64, " << p_plus_var << ": f64, " << p_minus_var << ": f64):\n";
    std::string pcond = fresh_ssa();
    out << "  " << pcond << " = llvm.icmp \"ult\" " << p_i << ", " << half << " : i64\n";
    out << "  llvm.cond_br " << pcond << ", ^" << p_body
        << ", ^" << p_done << "(" << p_plus_var << ", " << p_minus_var << " : f64, f64)\n";
    out << "^" << p_body << ":\n";

    // base = (idx & ~f_bit) | (((idx >> from) & 1) << to)
    std::string idx_masked = fresh_ssa();
    out << "  " << idx_masked << " = llvm.and " << p_i << ", " << not_fbit << " : i64\n";
    std::string shifted = fresh_ssa();
    out << "  " << shifted << " = llvm.lshr " << p_i << ", " << f64 << " : i64\n";
    std::string bit_f = fresh_ssa();
    out << "  " << bit_f << " = llvm.and " << shifted << ", %c1_i64 : i64\n";
    std::string bit_at_to = fresh_ssa();
    out << "  " << bit_at_to << " = llvm.shl " << bit_f << ", " << t64 << " : i64\n";
    std::string base = fresh_ssa();
    out << "  " << base << " = llvm.or " << idx_masked << ", " << bit_at_to << " : i64\n";

    std::string base_hi = fresh_ssa();
    out << "  " << base_hi << " = llvm.or " << base << ", " << f_bit << " : i64\n";
    std::string v_lo = emit_load_v(out, base);
    std::string v_hi = emit_load_v(out, base_hi);
    std::string sum_val = emit_cadd(out, v_lo, v_hi);
    std::string diff_val = emit_csub(out, v_lo, v_hi);
    std::string n_sum = emit_cnorm(out, sum_val);
    std::string n_diff = emit_cnorm(out, diff_val);
    std::string new_pp = fresh_ssa(), new_pm = fresh_ssa();
    out << "  " << new_pp << " = llvm.fadd " << p_plus_var << ", " << n_sum << " : f64\n";
    out << "  " << new_pm << " = llvm.fadd " << p_minus_var << ", " << n_diff << " : f64\n";
    std::string next_i = fresh_ssa();
    out << "  " << next_i << " = llvm.add " << p_i << ", " << loop_step << " : i64\n";
    out << "  llvm.br ^" << p_hdr << "(" << next_i << ", " << new_pp << ", " << new_pm << " : i64, f64, f64)\n";
    out << "^" << p_done << "(" << fp_plus << ": f64, " << fp_minus << ": f64):\n";

    if (cooperative_mode) {
        auto totals = emit_coop_reduce2(out, fp_plus, fp_minus);
        fp_plus = totals.first;
        fp_minus = totals.second;
    }
    std::string b_final = emit_sample_branch(out, fp_plus, fp_minus);

    // m_abs = b_final ^ pz
    std::string b_i8 = fresh_ssa();
    out << "  " << b_i8 << " = llvm.zext " << b_final << " : i1 to i8\n";
    std::string pz_i8 = fresh_ssa();
    out << "  " << pz_i8 << " = llvm.zext " << pz_bit << " : i1 to i8\n";
    std::string m_abs = fresh_ssa();
    out << "  " << m_abs << " = llvm.xor " << b_i8 << ", " << pz_i8 << " : i8\n";

    // Store measurement result
    std::string meas_val = m_abs;
    if (sign) {
        std::string xored = fresh_ssa();
        out << "  " << xored << " = llvm.xor " << m_abs << ", %c1_i8 : i8\n";
        meas_val = xored;
    }
    // Fold amplitudes: v[idx] = inv_sqrt2 * (b==0 ? v[base]+v[base|f] : v[base]-v[base|f])
    std::string f_hdr = fresh_label("smif_hdr"), f_body = fresh_label("smif_body"), f_done = fresh_label("smif_done");
    std::string f_i = fresh_ssa();
    std::string fold_init = cooperative_mode ? "%tidx" : "%c0_i64";
    std::string fold_step = cooperative_mode ? "%c256_i64" : "%c1_i64";
    out << "  llvm.br ^" << f_hdr << "(" << fold_init << " : i64)\n";
    out << "^" << f_hdr << "(" << f_i << ": i64):\n";
    std::string fcond = fresh_ssa();
    out << "  " << fcond << " = llvm.icmp \"ult\" " << f_i << ", " << half << " : i64\n";
    out << "  llvm.cond_br " << fcond << ", ^" << f_body << ", ^" << f_done << "\n";
    out << "^" << f_body << ":\n";

    // Recompute base for fold loop
    std::string f_idx_masked = fresh_ssa();
    out << "  " << f_idx_masked << " = llvm.and " << f_i << ", " << not_fbit << " : i64\n";
    std::string f_shifted = fresh_ssa();
    out << "  " << f_shifted << " = llvm.lshr " << f_i << ", " << f64 << " : i64\n";
    std::string f_bit_f = fresh_ssa();
    out << "  " << f_bit_f << " = llvm.and " << f_shifted << ", %c1_i64 : i64\n";
    std::string f_bit_at_to = fresh_ssa();
    out << "  " << f_bit_at_to << " = llvm.shl " << f_bit_f << ", " << t64 << " : i64\n";
    std::string f_base = fresh_ssa();
    out << "  " << f_base << " = llvm.or " << f_idx_masked << ", " << f_bit_at_to << " : i64\n";
    std::string f_base_hi = fresh_ssa();
    out << "  " << f_base_hi << " = llvm.or " << f_base << ", " << f_bit << " : i64\n";
    std::string fv_lo = emit_load_v(out, f_base);
    std::string fv_hi = emit_load_v(out, f_base_hi);
    std::string fold_sum = emit_cadd(out, fv_lo, fv_hi);
    std::string fold_diff = emit_csub(out, fv_lo, fv_hi);
    std::string fold_val = fresh_ssa();
    out << "  " << fold_val << " = llvm.select " << b_final << ", " << fold_diff << ", " << fold_sum
        << " : i1, !llvm.struct<(f32, f32)>\n";
    std::string scaled = emit_cscale_f64(out, fold_val, kInvSqrt2);
    if (cooperative_mode) {
        std::string scratch_p = fresh_ssa();
        out << "  " << scratch_p << " = llvm.getelementptr inbounds %scratch_ptr[" << f_i
            << "] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(f32, f32)>\n";
        out << "  llvm.store " << scaled << ", " << scratch_p
            << " {alignment = 8 : i64} : !llvm.struct<(f32, f32)>, !llvm.ptr\n";
    } else {
        emit_store_v(out, f_i, scaled);
    }
    std::string f_next = fresh_ssa();
    out << "  " << f_next << " = llvm.add " << f_i << ", " << fold_step << " : i64\n";
    out << "  llvm.br ^" << f_hdr << "(" << f_next << " : i64)\n";
    out << "^" << f_done << ":\n";
    if (cooperative_mode) {
        emit_barrier(out);
        std::string cp_hdr = fresh_label("smif_cp_hdr");
        std::string cp_body = fresh_label("smif_cp_body");
        std::string cp_done = fresh_label("smif_cp_done");
        std::string cp_i = fresh_ssa();
        out << "  llvm.br ^" << cp_hdr << "(%tidx : i64)\n";
        out << "^" << cp_hdr << "(" << cp_i << ": i64):\n";
        std::string cp_cond = fresh_ssa();
        out << "  " << cp_cond << " = llvm.icmp \"ult\" " << cp_i << ", " << half
            << " : i64\n";
        out << "  llvm.cond_br " << cp_cond << ", ^" << cp_body << ", ^" << cp_done << "\n";
        out << "^" << cp_body << ":\n";
        std::string scratch_p = fresh_ssa();
        out << "  " << scratch_p << " = llvm.getelementptr inbounds %scratch_ptr[" << cp_i
            << "] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(f32, f32)>\n";
        std::string scratch_v = fresh_ssa();
        out << "  " << scratch_v << " = llvm.load " << scratch_p
            << " {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(f32, f32)>\n";
        emit_store_v(out, cp_i, scratch_v);
        std::string cp_next = fresh_ssa();
        out << "  " << cp_next << " = llvm.add " << cp_i << ", %c256_i64 : i64\n";
        out << "  llvm.br ^" << cp_hdr << "(" << cp_next << " : i64)\n";
        out << "^" << cp_done << ":\n";
        emit_barrier(out);
    }

    // active_k-- and frame update on 'to' axis
    {
        std::string g = cooperative_mode ? emit_tid0_guard_begin(out) : "";
        std::string new_ak = emit_const_i32(out, g_emit_ak - 1);
        out << "  llvm.store " << new_ak << ", %active_k_ptr : i32, !llvm.ptr\n";
        std::string cidx = emit_const_i64(out, classical_idx);
        std::string meas_p = fresh_ssa();
        out << "  " << meas_p << " = llvm.getelementptr inbounds %meas_ptr["
            << cidx << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        out << "  llvm.store " << meas_val << ", " << meas_p << " : i8, !llvm.ptr\n";
        std::string m_ne0 = fresh_ssa();
        out << "  " << m_ne0 << " = llvm.icmp \"ne\" " << m_abs << ", %c0_i8 : i8\n";
        emit_bit_set(out, "%px_ptr", to_i32, m_ne0);
        std::string false_val = emit_const_i1(out, false);
        emit_bit_set(out, "%pz_ptr", to_i32, false_val);
        if (cooperative_mode) emit_tid0_guard_end(out, g);
    }
}

// -----------------------------------------------------------------------
// U2/U4 fused unitary emission — inlines constant pool at compile time
// -----------------------------------------------------------------------

void emit_array_u2(std::ostringstream& out, const FlattenedProgram& flat,
                   uint32_t axis, uint32_t cp_idx) {
    if (cp_idx >= flat.fused_u2.size()) {
        out << "  // U2 cp_idx out of range — discard\n";
        out << "  llvm.store %c1_i8, %discarded_ptr : i8, !llvm.ptr\n";
        return;
    }
    const auto& u2 = flat.fused_u2[cp_idx];
    out << "  // array_u2 axis=" << axis << " cp_idx=" << cp_idx
        << " (active_k=" << g_emit_ak << ")\n";

    if (g_emit_ak == 0) return;

    // The Pauli frame selects the constant-pool matrix branch. Synchronize
    // before reading it so every workitem dispatches on the same shared state.
    if (cooperative_mode) emit_barrier(out);

    char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", axis);
    std::string axis_i32 = fresh_ssa();
    out << "  " << axis_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
    std::string px_bit = emit_bit_get(out, "%px_ptr", axis_i32);
    std::string pz_bit = emit_bit_get(out, "%pz_ptr", axis_i32);

    std::string px_i32 = fresh_ssa();
    out << "  " << px_i32 << " = llvm.zext " << px_bit << " : i1 to i32\n";
    std::string pz_i32 = fresh_ssa();
    out << "  " << pz_i32 << " = llvm.zext " << pz_bit << " : i1 to i32\n";
    std::string pz_sh = fresh_ssa();
    out << "  " << pz_sh << " = llvm.shl " << pz_i32 << ", %c1_i32 : i32\n";
    std::string in_state = fresh_ssa();
    out << "  " << in_state << " = llvm.or " << px_i32 << ", " << pz_sh << " : i32\n";

    uint64_t axis_val64 = axis;
    std::string axis_pos = emit_const_i64(out, axis_val64);
    std::string axis_bit = fresh_ssa();
    out << "  " << axis_bit << " = llvm.shl %c1_i64, " << axis_pos << " : i64\n";
    uint64_t iters_val = 1ULL << (g_emit_ak - 1);
    std::string iters = emit_const_i64(out, iters_val);

    std::string lbl_done = fresh_label("u2_done");

    for (int s = 0; s < 4; ++s) {
        // Separator-free s-embedding can collide (see u4 note); counter is unique.
        std::string lbl_apply = fresh_label("u2_apply_");
        std::string lbl_next = fresh_label("u2_nx_");

        std::string sval = emit_const_i32(out, s);
        std::string cmp = fresh_ssa();
        out << "  " << cmp << " = llvm.icmp \"eq\" " << in_state << ", " << sval << " : i32\n";
        out << "  llvm.cond_br " << cmp << ", ^" << lbl_apply << ", ^" << lbl_next << "\n";
        out << "^" << lbl_apply << ":\n";

        const GpuComplex* mat = u2.matrices[s];
        uint8_t out_st = u2.out_states[s];

        std::string u2_loop_init = cooperative_mode ? "%tidx" : "%c0_i64";
        std::string u2_loop_step = cooperative_mode ? "%c256_i64" : "%c1_i64";
        std::string loop_var = fresh_ssa();
        std::string lp_hdr = fresh_label("u2lp_hdr"), lp_body = fresh_label("u2lp_body"), lp_end = fresh_label("u2lp_end");
        out << "  llvm.br ^" << lp_hdr << "(" << u2_loop_init << " : i64)\n";
        out << "^" << lp_hdr << "(" << loop_var << ": i64):\n";
        std::string lcond = fresh_ssa();
        out << "  " << lcond << " = llvm.icmp \"ult\" " << loop_var << ", " << iters << " : i64\n";
        out << "  llvm.cond_br " << lcond << ", ^" << lp_body << ", ^" << lp_end << "\n";
        out << "^" << lp_body << ":\n";

        std::string scat = emit_scatter_bits_1(out, loop_var, axis_pos);
        std::string idx0 = scat;
        std::string idx1 = fresh_ssa();
        out << "  " << idx1 << " = llvm.or " << idx0 << ", " << axis_bit << " : i64\n";

        std::string v0 = emit_load_v(out, idx0);
        std::string v1 = emit_load_v(out, idx1);

        std::string t00 = emit_cmul_const(out, v0, mat[0].re, mat[0].im);
        std::string t01 = emit_cmul_const(out, v1, mat[1].re, mat[1].im);
        std::string new_v0 = emit_cadd(out, t00, t01);

        std::string t10 = emit_cmul_const(out, v0, mat[2].re, mat[2].im);
        std::string t11 = emit_cmul_const(out, v1, mat[3].re, mat[3].im);
        std::string new_v1 = emit_cadd(out, t10, t11);

        emit_store_v(out, idx0, new_v0);
        emit_store_v(out, idx1, new_v1);

        std::string lp_next = fresh_ssa();
        out << "  " << lp_next << " = llvm.add " << loop_var << ", " << u2_loop_step << " : i64\n";
        out << "  llvm.br ^" << lp_hdr << "(" << lp_next << " : i64)\n";
        out << "^" << lp_end << ":\n";

        bool new_px = (out_st & 1) != 0;
        bool new_pz = (out_st & 2) != 0;
        std::string npx = emit_const_i1(out, new_px);
        std::string npz = emit_const_i1(out, new_pz);
        { std::string g = cooperative_mode ? emit_tid0_guard_begin(out) : "";
        emit_bit_set(out, "%px_ptr", axis_i32, npx);
        emit_bit_set(out, "%pz_ptr", axis_i32, npz);
        if (cooperative_mode) emit_tid0_guard_end_nobarrier(out, g); }

        out << "  llvm.br ^" << lbl_done << "\n";
        out << "^" << lbl_next << ":\n";
    }
    out << "  llvm.br ^" << lbl_done << "\n";
    out << "^" << lbl_done << ":\n";
    if (cooperative_mode) emit_barrier(out);
}

void emit_array_u4(std::ostringstream& out, const FlattenedProgram& flat,
                   uint32_t axis_lo, uint32_t axis_hi, uint32_t cp_idx) {
    if (cp_idx >= flat.fused_u4.size()) {
        out << "  // U4 cp_idx out of range — discard\n";
        out << "  llvm.store %c1_i8, %discarded_ptr : i8, !llvm.ptr\n";
        return;
    }
    const auto& u4 = flat.fused_u4[cp_idx];
    out << "  // array_u4 axis_lo=" << axis_lo << " axis_hi=" << axis_hi
        << " cp_idx=" << cp_idx << " (active_k=" << g_emit_ak << ")\n";

    if (g_emit_ak < 2) return;

    // The Pauli frame selects the constant-pool matrix branch. Synchronize
    // before reading it so every workitem dispatches on the same shared state.
    if (cooperative_mode) emit_barrier(out);

    char alo_buf[32], ahi_buf[32];
    snprintf(alo_buf, sizeof(alo_buf), "%u", axis_lo);
    snprintf(ahi_buf, sizeof(ahi_buf), "%u", axis_hi);
    std::string alo_i32 = fresh_ssa(), ahi_i32 = fresh_ssa();
    out << "  " << alo_i32 << " = llvm.mlir.constant(" << alo_buf << " : i32) : i32\n";
    out << "  " << ahi_i32 << " = llvm.mlir.constant(" << ahi_buf << " : i32) : i32\n";

    std::string px_lo = emit_bit_get(out, "%px_ptr", alo_i32);
    std::string pz_lo = emit_bit_get(out, "%pz_ptr", alo_i32);
    std::string px_hi = emit_bit_get(out, "%px_ptr", ahi_i32);
    std::string pz_hi = emit_bit_get(out, "%pz_ptr", ahi_i32);

    std::string px_lo_i32 = fresh_ssa(), pz_lo_i32 = fresh_ssa();
    std::string px_hi_i32 = fresh_ssa(), pz_hi_i32 = fresh_ssa();
    out << "  " << px_lo_i32 << " = llvm.zext " << px_lo << " : i1 to i32\n";
    out << "  " << pz_lo_i32 << " = llvm.zext " << pz_lo << " : i1 to i32\n";
    out << "  " << px_hi_i32 << " = llvm.zext " << px_hi << " : i1 to i32\n";
    out << "  " << pz_hi_i32 << " = llvm.zext " << pz_hi << " : i1 to i32\n";

    std::string c2 = emit_const_i32(out, 2);
    std::string c3 = emit_const_i32(out, 3);
    std::string pz_lo_sh = fresh_ssa(), px_hi_sh = fresh_ssa(), pz_hi_sh = fresh_ssa();
    out << "  " << pz_lo_sh << " = llvm.shl " << pz_lo_i32 << ", %c1_i32 : i32\n";
    out << "  " << px_hi_sh << " = llvm.shl " << px_hi_i32 << ", " << c2 << " : i32\n";
    out << "  " << pz_hi_sh << " = llvm.shl " << pz_hi_i32 << ", " << c3 << " : i32\n";
    std::string s01 = fresh_ssa(), s012 = fresh_ssa(), in_state = fresh_ssa();
    out << "  " << s01 << " = llvm.or " << px_lo_i32 << ", " << pz_lo_sh << " : i32\n";
    out << "  " << s012 << " = llvm.or " << s01 << ", " << px_hi_sh << " : i32\n";
    out << "  " << in_state << " = llvm.or " << s012 << ", " << pz_hi_sh << " : i32\n";

    std::string alo_pos = emit_const_i64(out, (uint64_t)axis_lo);
    std::string ahi_pos = emit_const_i64(out, (uint64_t)axis_hi);
    std::string lo_bit = fresh_ssa(), hi_bit = fresh_ssa();
    out << "  " << lo_bit << " = llvm.shl %c1_i64, " << alo_pos << " : i64\n";
    out << "  " << hi_bit << " = llvm.shl %c1_i64, " << ahi_pos << " : i64\n";
    uint64_t iters_val = 1ULL << (g_emit_ak - 2);
    std::string iters = emit_const_i64(out, iters_val);

    std::string lbl_done = fresh_label("u4_done");

    for (int s = 0; s < 16; ++s) {
        const auto& entry = u4.entries[s];
        bool all_zero = true;
        for (int r = 0; r < 4 && all_zero; ++r)
            for (int c2_ = 0; c2_ < 4 && all_zero; ++c2_)
                if (entry.matrix[r][c2_].re != 0.0f || entry.matrix[r][c2_].im != 0.0f)
                    all_zero = false;
        if (all_zero) continue;

        // Note: do NOT embed `s` in the prefix without a separator — fresh_label
        // appends a counter, so "u4_s"+s+counter can collide (s=15,c=862 vs
        // s=1,c=5862 both give u4_s15862). The counter alone is unique.
        std::string lbl_apply = fresh_label("u4_apply_");
        std::string lbl_next = fresh_label("u4_nx_");

        std::string sval = emit_const_i32(out, s);
        std::string cmp = fresh_ssa();
        out << "  " << cmp << " = llvm.icmp \"eq\" " << in_state << ", " << sval << " : i32\n";
        out << "  llvm.cond_br " << cmp << ", ^" << lbl_apply << ", ^" << lbl_next << "\n";
        out << "^" << lbl_apply << ":\n";

        std::string loop_var = fresh_ssa();
        std::string u4_loop_init = cooperative_mode ? "%tidx" : "%c0_i64";
        std::string u4_loop_step = cooperative_mode ? "%c256_i64" : "%c1_i64";
        std::string lp_hdr = fresh_label("u4lp_hdr"), lp_body = fresh_label("u4lp_body"), lp_end = fresh_label("u4lp_end");
        out << "  llvm.br ^" << lp_hdr << "(" << u4_loop_init << " : i64)\n";
        out << "^" << lp_hdr << "(" << loop_var << ": i64):\n";
        std::string lcond = fresh_ssa();
        out << "  " << lcond << " = llvm.icmp \"ult\" " << loop_var << ", " << iters << " : i64\n";
        out << "  llvm.cond_br " << lcond << ", ^" << lp_body << ", ^" << lp_end << "\n";
        out << "^" << lp_body << ":\n";

        std::string base = emit_scatter_bits_2(out, loop_var, alo_pos, ahi_pos);
        std::string idx1 = fresh_ssa(), idx2 = fresh_ssa(), idx3 = fresh_ssa();
        out << "  " << idx1 << " = llvm.or " << base << ", " << lo_bit << " : i64\n";
        out << "  " << idx2 << " = llvm.or " << base << ", " << hi_bit << " : i64\n";
        std::string both_bits = fresh_ssa();
        out << "  " << both_bits << " = llvm.or " << lo_bit << ", " << hi_bit << " : i64\n";
        out << "  " << idx3 << " = llvm.or " << base << ", " << both_bits << " : i64\n";

        std::string v[4];
        v[0] = emit_load_v(out, base);
        v[1] = emit_load_v(out, idx1);
        v[2] = emit_load_v(out, idx2);
        v[3] = emit_load_v(out, idx3);

        std::string new_v[4];
        for (int r = 0; r < 4; ++r) {
            std::string acc;
            for (int k = 0; k < 4; ++k) {
                float mre = entry.matrix[r][k].re;
                float mim = entry.matrix[r][k].im;
                if (mre == 0.0f && mim == 0.0f) continue;
                std::string term = emit_cmul_const(out, v[k], mre, mim);
                if (acc.empty()) {
                    acc = term;
                } else {
                    acc = emit_cadd(out, acc, term);
                }
            }
            if (acc.empty()) {
                std::string z0 = fresh_ssa(), z1 = fresh_ssa(), z2 = fresh_ssa();
                out << "  " << z0 << " = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
                out << "  " << z1 << " = llvm.insertvalue %f_zero, " << z0 << "[0] : !llvm.struct<(f32, f32)>\n";
                out << "  " << z2 << " = llvm.insertvalue %f_zero, " << z1 << "[1] : !llvm.struct<(f32, f32)>\n";
                acc = z2;
            }
            new_v[r] = acc;
        }

        emit_store_v(out, base, new_v[0]);
        emit_store_v(out, idx1, new_v[1]);
        emit_store_v(out, idx2, new_v[2]);
        emit_store_v(out, idx3, new_v[3]);

        std::string lp_next = fresh_ssa();
        out << "  " << lp_next << " = llvm.add " << loop_var << ", " << u4_loop_step << " : i64\n";
        out << "  llvm.br ^" << lp_hdr << "(" << lp_next << " : i64)\n";
        out << "^" << lp_end << ":\n";

        uint8_t os = entry.out_state;
        { std::string g = cooperative_mode ? emit_tid0_guard_begin(out) : "";
        emit_bit_set(out, "%px_ptr", alo_i32, emit_const_i1(out, (os & 1) != 0));
        emit_bit_set(out, "%pz_ptr", alo_i32, emit_const_i1(out, (os & 2) != 0));
        emit_bit_set(out, "%px_ptr", ahi_i32, emit_const_i1(out, (os & 4) != 0));
        emit_bit_set(out, "%pz_ptr", ahi_i32, emit_const_i1(out, (os & 8) != 0));
        if (cooperative_mode) emit_tid0_guard_end_nobarrier(out, g); }

        out << "  llvm.br ^" << lbl_done << "\n";
        out << "^" << lbl_next << ":\n";
    }
    out << "  llvm.br ^" << lbl_done << "\n";
    out << "^" << lbl_done << ":\n";
    if (cooperative_mode) emit_barrier(out);
}

void emit_expand_plain(std::ostringstream& out) {
    uint64_t half_val = 1ULL << g_emit_ak;
    std::string half = emit_const_i64(out, half_val);

    // Copy v[i] → v[i+half] for i in [0, half)
    std::string loop_var = fresh_ssa();
    std::string ex_hdr = fresh_label("ex_hdr"), ex_body = fresh_label("ex_body"), ex_done = fresh_label("ex_done");
    out << "  llvm.br ^" << ex_hdr << "(%c0_i64 : i64)\n";
    out << "^" << ex_hdr << "(" << loop_var << ": i64):\n";
    std::string ex_cond = fresh_ssa();
    out << "  " << ex_cond << " = llvm.icmp \"ult\" " << loop_var << ", " << half << " : i64\n";
    out << "  llvm.cond_br " << ex_cond << ", ^" << ex_body << ", ^" << ex_done << "\n";
    out << "^" << ex_body << ":\n";
    std::string src = emit_load_v(out, loop_var);
    std::string dst_idx = fresh_ssa();
    out << "  " << dst_idx << " = llvm.add " << loop_var << ", " << half << " : i64\n";
    emit_store_v(out, dst_idx, src);
    std::string ex_next = fresh_ssa();
    out << "  " << ex_next << " = llvm.add " << loop_var << ", %c1_i64 : i64\n";
    out << "  llvm.br ^" << ex_hdr << "(" << ex_next << " : i64)\n";
    out << "^" << ex_done << ":\n";

    // Barrier: all threads must finish the redundant copy before the next op
    // reads the expanded array (coop/global shared LDS/HBM).
    if (cooperative_mode) emit_barrier(out);

    // active_k++ (compile-time tracked via g_emit_ak; store runtime copy for coop/global tiers)
    std::string new_ak_val = emit_const_i32(out, g_emit_ak + 1);
    out << "  llvm.store " << new_ak_val << ", %active_k_ptr : i32, !llvm.ptr\n";
}

void emit_expand_t(std::ostringstream& out, uint32_t axis, bool dagger) {
    uint64_t half_val = 1ULL << g_emit_ak;
    std::string half = emit_const_i64(out, half_val);

    // Read px bit
    char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", axis);
    std::string axis_i32 = fresh_ssa();
    out << "  " << axis_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
    std::string px_bit = emit_bit_get(out, "%px_ptr", axis_i32);

    // Phase: T = (inv_sqrt2, inv_sqrt2), T_dag = (inv_sqrt2, -inv_sqrt2)
    // If px is set, negate imaginary
    double imag_base = dagger ? -kInvSqrt2 : kInvSqrt2;

    // v[i + half] = cmul(v[i], phase) for i in [0, half)
    std::string et_loop_var = fresh_ssa();
    std::string et_hdr = fresh_label("et_hdr"), et_body = fresh_label("et_body"), et_done = fresh_label("et_done");
    out << "  llvm.br ^" << et_hdr << "(%c0_i64 : i64)\n";
    out << "^" << et_hdr << "(" << et_loop_var << ": i64):\n";
    std::string et_cond = fresh_ssa();
    out << "  " << et_cond << " = llvm.icmp \"ult\" " << et_loop_var << ", " << half << " : i64\n";
    out << "  llvm.cond_br " << et_cond << ", ^" << et_body << ", ^" << et_done << "\n";
    out << "^" << et_body << ":\n";
    std::string src = emit_load_v(out, et_loop_var);
    std::string pos_phase = emit_cmul_const(out, src, kInvSqrt2, imag_base);
    std::string neg_phase = emit_cmul_const(out, src, kInvSqrt2, -imag_base);
    std::string phased = fresh_ssa();
    out << "  " << phased << " = llvm.select " << px_bit << ", " << neg_phase << ", " << pos_phase
        << " : i1, !llvm.struct<(f32, f32)>\n";
    std::string dst_idx = fresh_ssa();
    out << "  " << dst_idx << " = llvm.add " << et_loop_var << ", " << half << " : i64\n";
    emit_store_v(out, dst_idx, phased);
    std::string et_next = fresh_ssa();
    out << "  " << et_next << " = llvm.add " << et_loop_var << ", %c1_i64 : i64\n";
    out << "  llvm.br ^" << et_hdr << "(" << et_next << " : i64)\n";
    out << "^" << et_done << ":\n";

    // Barrier: all threads finish the redundant phased-copy before next op reads.
    if (cooperative_mode) emit_barrier(out);

    // active_k++ (compile-time tracked via g_emit_ak)
    std::string new_ak_val2 = emit_const_i32(out, g_emit_ak + 1);
    out << "  llvm.store " << new_ak_val2 << ", %active_k_ptr : i32, !llvm.ptr\n";
}

}  // anonymous namespace

// -----------------------------------------------------------------------
// emit_mlir_text: main textual MLIR emission function
// -----------------------------------------------------------------------
std::string emit_mlir_text(const FlattenedProgram& flat) {
    label_counter = 0;
    ssa_counter = 100;
    cooperative_mode = false;
    lds_amplitudes = false;
    UsedFunctions uf = analyze_used_functions(flat);

    std::ostringstream out;
    uint32_t kMaxAmplitudes = 1u << flat.peak_rank;

    out << "module attributes {llvm.target_triple = \"amdgcn-amd-amdhsa\"} {\n\n";
    emit_gpu_intrinsic_decls(out);

    // LDS for warp shuffle inter-wavefront reduction (4 wavefronts × values)
    emit_lds_global(out, "lds_red_passed", "i64", 4);
    emit_lds_global(out, "lds_red_logical", "i64", 4);
    emit_lds_global(out, "lds_red_obs", "i64", 4 * kMaxObs);
    out << "\n";

    out << "llvm.func amdgpu_kernelcc @compiled_mlir_kernel(\n"
        << "    %shot_offset: i64 {llvm.inreg}, %shots: i64 {llvm.inreg}, %seed: i64 {llvm.inreg},\n"
        << "    %block_counts: !llvm.ptr {llvm.noalias},\n"
        << "    %noise_hazards_ptr: !llvm.ptr {llvm.noalias}, %noise_sites_ptr: !llvm.ptr {llvm.noalias},\n"
        << "    %noise_channels_ptr: !llvm.ptr {llvm.noalias}, %num_noise_sites: i64 {llvm.inreg},\n"
        << "    %num_obs: i32 {llvm.inreg}, %num_exp: i32 {llvm.inreg}) -> ()\n"
        << "  attributes {\"amdgpu-flat-work-group-size\"=\"256,256\", \"denormal-fp-math-f32\"=\"preserve-sign\", \"uniform-work-group-size\"=\"true\", \"amdgpu-no-implicitarg-ptr\"} {\n"
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
    out << "  %nni_ptr_p5 = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr<5>\n";
    out << "  %nni_ptr = llvm.addrspacecast %nni_ptr_p5 : !llvm.ptr<5> to !llvm.ptr\n";
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

    // Guard: if batch_shot_id >= shots, skip to reduction with zero values
    // OOB threads must participate in warp reduction (ds_bpermute reads all lanes)
    std::string lbl_run = fresh_label("run");
    std::string lbl_oob = fresh_label("oob");
    out << "  %oob = llvm.icmp \"uge\" %batch_shot_id, %shots : i64\n";
    out << "  llvm.cond_br %oob, ^" << lbl_oob << ", ^" << lbl_run << "\n";
    out << "^" << lbl_oob << ":\n";
    // OOB: mark as discarded so reduction treats as passed=0
    out << "  llvm.store %c1_i8, %discarded_ptr : i8, !llvm.ptr\n";
    // Zero obs array for OOB threads
    for (uint32_t i = 0; i < std::min(flat.num_observables, kMaxObs); ++i) {
        std::string idx = emit_const_i64(out, i);
        std::string ptr = fresh_ssa();
        out << "  " << ptr << " = llvm.getelementptr inbounds %obs_ptr[" << idx
            << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        out << "  llvm.store %c0_i8, " << ptr << " : i8, !llvm.ptr\n";
    }
    out << "  llvm.br ^reduce_start\n";
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

    // Initialize meas[] to zero (loop for all meas slots)
    if (flat.total_meas_slots > 0) {
        std::string meas_count = emit_const_i64(out, flat.total_meas_slots);
        std::string mz_hdr = fresh_label("mz_hdr"), mz_body = fresh_label("mz_body"), mz_done = fresh_label("mz_done");
        out << "  llvm.br ^" << mz_hdr << "(%c0_i64 : i64)\n";
        out << "^" << mz_hdr << "(%mz_i: i64):\n";
        std::string mz_cond = fresh_ssa();
        out << "  " << mz_cond << " = llvm.icmp \"ult\" %mz_i, " << meas_count << " : i64\n";
        out << "  llvm.cond_br " << mz_cond << ", ^" << mz_body << ", ^" << mz_done << "\n";
        out << "^" << mz_body << ":\n";
        std::string mz_ptr = fresh_ssa();
        out << "  " << mz_ptr << " = llvm.getelementptr inbounds %meas_ptr[%mz_i] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        out << "  llvm.store %c0_i8, " << mz_ptr << " : i8, !llvm.ptr\n";
        std::string mz_next = fresh_ssa();
        out << "  " << mz_next << " = llvm.add %mz_i, %c1_i64 : i64\n";
        out << "  llvm.br ^" << mz_hdr << "(" << mz_next << " : i64)\n";
        out << "^" << mz_done << ":\n";
    }

    // Initialize next_noise_idx = 0
    out << "  llvm.store %c0_i32, %nni_ptr : i32, !llvm.ptr\n";

    // Seed RNG: shot_id = shot_offset + batch_shot_id
    std::string shot_id = fresh_ssa();
    out << "  " << shot_id << " = llvm.add %shot_offset, %batch_shot_id : i64\n";
    emit_rng_seed(out, "%seed", shot_id);

    // Initial noise draw (if circuit has noise)
    if (!flat.noise_sites.empty()) {
        emit_draw_next_noise(out, flat);
    }

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
    std::string coop_init = cooperative_mode ? "%tidx" : "%c0_i64";
    std::string coop_step = cooperative_mode ? "%c256_i64" : "%c1_i64";
    out << "  // --- Instruction sequence (" << flat.instrs.size() << " ops) ---\n";
    g_emit_ak = 0;

    for (size_t pc = 0; pc < flat.instrs.size(); ++pc) {
        const GpuInstr& ins = flat.instrs[pc];
        bool sign = (ins.flags & kFlagSign) != 0;
        bool identity = (ins.flags & kFlagIdentity) != 0;
        auto op = static_cast<Opcode>(ins.opcode);

        out << "  // op[" << pc << "] opcode=" << (unsigned)ins.opcode
            << " active_k=" << g_emit_ak << "\n";

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
        // Track active_k at compile time
        switch (op) {
            case Opcode::OP_EXPAND:
            case Opcode::OP_EXPAND_T:
            case Opcode::OP_EXPAND_T_DAG:
            case Opcode::OP_EXPAND_ROT:
                g_emit_ak++;
                break;
            case Opcode::OP_MEAS_ACTIVE_DIAGONAL:
            case Opcode::OP_MEAS_ACTIVE_INTERFERE:
            case Opcode::OP_SWAP_MEAS_INTERFERE:
                if (g_emit_ak > 0) g_emit_ak--;
                break;
            default:
                break;
        }
    }

    out << "  // --- End instruction sequence ---\n";
    out << "  llvm.br ^reduce_start\n";
    out << "^reduce_start:\n";

    // Warp-shuffle result aggregation (256→1 via ds_bpermute + LDS)
    // Phase 0: Compute per-thread values
    // Both valid and OOB threads reach here. OOB threads have discarded=1, obs=0.
    std::string disc_val = fresh_ssa();
    out << "  " << disc_val << " = llvm.load %discarded_ptr : !llvm.ptr -> i8\n";
    std::string is_valid = fresh_ssa();
    out << "  " << is_valid << " = llvm.icmp \"eq\" " << disc_val << ", %c0_i8 : i8\n";
    std::string local_passed = fresh_ssa();
    out << "  " << local_passed << " = llvm.zext " << is_valid << " : i1 to i64\n";

    // Compute local_logical (1 if any observable mismatch)
    std::string any_obs_fail = fresh_ssa();
    out << "  " << any_obs_fail << " = llvm.mlir.constant(0 : i64) : i64\n";
    std::string local_obs_arr[kMaxObs];
    for (uint32_t i = 0; i < std::min(flat.num_observables, kMaxObs); ++i) {
        std::string idx = emit_const_i64(out, i);
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
        std::string obs_val = fresh_ssa();
        out << "  " << obs_val << " = llvm.zext " << obs_ne << " : i1 to i64\n";
        // Mask with is_valid (only count if not discarded)
        std::string masked = fresh_ssa();
        out << "  " << masked << " = llvm.and " << obs_val << ", " << local_passed << " : i64\n";
        local_obs_arr[i] = masked;
        std::string new_fail = fresh_ssa();
        out << "  " << new_fail << " = llvm.or " << any_obs_fail << ", " << masked << " : i64\n";
        any_obs_fail = new_fail;
    }
    std::string local_logical = fresh_ssa();
    out << "  " << local_logical << " = llvm.and " << any_obs_fail << ", " << local_passed << " : i64\n";

    // Phase 1: Intra-wavefront reduction (5 rounds with ds_bpermute)
    std::string red_passed = emit_wavefront_reduce_i64(out, local_passed);
    std::string red_logical = emit_wavefront_reduce_i64(out, local_logical);
    std::string red_obs[kMaxObs];
    for (uint32_t i = 0; i < std::min(flat.num_observables, kMaxObs); ++i) {
        red_obs[i] = emit_wavefront_reduce_i64(out, local_obs_arr[i]);
    }

    // Phase 2: Inter-wavefront via LDS (4 wavefronts)
    // Get LDS pointers for reduction scratch
    out << "  %lds_red_p_as3 = llvm.mlir.addressof @lds_red_passed : !llvm.ptr<3>\n";
    out << "  %lds_red_p = llvm.addrspacecast %lds_red_p_as3 : !llvm.ptr<3> to !llvm.ptr\n";
    out << "  %lds_red_l_as3 = llvm.mlir.addressof @lds_red_logical : !llvm.ptr<3>\n";
    out << "  %lds_red_l = llvm.addrspacecast %lds_red_l_as3 : !llvm.ptr<3> to !llvm.ptr\n";
    out << "  %lds_red_o_as3 = llvm.mlir.addressof @lds_red_obs : !llvm.ptr<3>\n";
    out << "  %lds_red_o = llvm.addrspacecast %lds_red_o_as3 : !llvm.ptr<3> to !llvm.ptr\n";

    // Lane 0 of each wavefront writes to LDS[warp_id]
    std::string c63_i32 = emit_const_i32(out, 63);
    std::string c6_i32 = emit_const_i32(out, 6);
    std::string lane_id = fresh_ssa();
    out << "  " << lane_id << " = llvm.and %tidx_i32, " << c63_i32 << " : i32\n";
    std::string is_lane0 = fresh_ssa();
    out << "  " << is_lane0 << " = llvm.icmp \"eq\" " << lane_id << ", %c0_i32 : i32\n";
    std::string warp_id = fresh_ssa();
    out << "  " << warp_id << " = llvm.lshr %tidx_i32, " << c6_i32 << " : i32\n";
    std::string warp_i64 = fresh_ssa();
    out << "  " << warp_i64 << " = llvm.zext " << warp_id << " : i32 to i64\n";

    std::string lbl_wr_lds = fresh_label("wr_lds");
    std::string lbl_wr_done = fresh_label("wr_done");
    out << "  llvm.cond_br " << is_lane0 << ", ^" << lbl_wr_lds << ", ^" << lbl_wr_done << "\n";
    out << "^" << lbl_wr_lds << ":\n";
    // Store to LDS
    std::string lds_p_ptr = fresh_ssa();
    out << "  " << lds_p_ptr << " = llvm.getelementptr inbounds %lds_red_p["
        << warp_i64 << "] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
    out << "  llvm.store " << red_passed << ", " << lds_p_ptr << " : i64, !llvm.ptr\n";
    std::string lds_l_ptr = fresh_ssa();
    out << "  " << lds_l_ptr << " = llvm.getelementptr inbounds %lds_red_l["
        << warp_i64 << "] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
    out << "  llvm.store " << red_logical << ", " << lds_l_ptr << " : i64, !llvm.ptr\n";
    for (uint32_t i = 0; i < std::min(flat.num_observables, kMaxObs); ++i) {
        std::string obs_lds_idx = emit_const_i64(out, i * 4);
        std::string obs_lds_off = fresh_ssa();
        out << "  " << obs_lds_off << " = llvm.add " << obs_lds_idx << ", " << warp_i64 << " : i64\n";
        std::string obs_lds_p = fresh_ssa();
        out << "  " << obs_lds_p << " = llvm.getelementptr inbounds %lds_red_o["
            << obs_lds_off << "] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
        out << "  llvm.store " << red_obs[i] << ", " << obs_lds_p << " : i64, !llvm.ptr\n";
    }
    out << "  llvm.br ^" << lbl_wr_done << "\n";
    out << "^" << lbl_wr_done << ":\n";
    emit_barrier(out);

    // Phase 3: tid < 4 reads LDS and reduces with ds_bpermute
    std::string c4_red = emit_const_i32(out, 4);
    std::string is_first4 = fresh_ssa();
    out << "  " << is_first4 << " = llvm.icmp \"ult\" %tidx_i32, " << c4_red << " : i32\n";
    std::string lbl_final = fresh_label("final_red");
    std::string lbl_done = fresh_label("done");
    out << "  llvm.cond_br " << is_first4 << ", ^" << lbl_final << ", ^" << lbl_done << "\n";
    out << "^" << lbl_final << ":\n";

    // Load from LDS
    std::string tid_i64 = fresh_ssa();
    out << "  " << tid_i64 << " = llvm.zext %tidx_i32 : i32 to i64\n";
    std::string fp_ptr = fresh_ssa();
    out << "  " << fp_ptr << " = llvm.getelementptr inbounds %lds_red_p["
        << tid_i64 << "] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
    std::string fp_val = fresh_ssa();
    out << "  " << fp_val << " = llvm.load " << fp_ptr << " : !llvm.ptr -> i64\n";
    std::string fl_ptr = fresh_ssa();
    out << "  " << fl_ptr << " = llvm.getelementptr inbounds %lds_red_l["
        << tid_i64 << "] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
    std::string fl_val = fresh_ssa();
    out << "  " << fl_val << " = llvm.load " << fl_ptr << " : !llvm.ptr -> i64\n";

    // Reduce 4→1 with ds_bpermute (offsets 2, 1)
    std::string fp2 = fp_val, fl2 = fl_val;
    for (int off : {2, 1}) {
        std::string mask = emit_const_i32(out, off);
        std::string pp = emit_shfl_xor_i64(out, fp2, mask);
        std::string fp_sum = fresh_ssa();
        out << "  " << fp_sum << " = llvm.add " << fp2 << ", " << pp << " : i64\n";
        fp2 = fp_sum;
        std::string lp = emit_shfl_xor_i64(out, fl2, mask);
        std::string fl_sum = fresh_ssa();
        out << "  " << fl_sum << " = llvm.add " << fl2 << ", " << lp << " : i64\n";
        fl2 = fl_sum;
    }

    // tid==0: single atomic add to block_counts
    std::string is_tid0 = fresh_ssa();
    out << "  " << is_tid0 << " = llvm.icmp \"eq\" %tidx_i32, %c0_i32 : i32\n";
    std::string lbl_t0_wr = fresh_label("t0_wr");
    std::string lbl_t0_done = fresh_label("t0_done");
    out << "  llvm.cond_br " << is_tid0 << ", ^" << lbl_t0_wr << ", ^" << lbl_t0_done << "\n";
    out << "^" << lbl_t0_wr << ":\n";
    out << "  " << fresh_ssa() << " = llvm.atomicrmw add %block_counts, " << fp2
        << " syncscope(\"agent\") monotonic : !llvm.ptr, i64\n";
    // logical_errors at byte offset 8
    std::string le_off = emit_const_i64(out, 8);
    std::string le_p = fresh_ssa();
    out << "  " << le_p << " = llvm.getelementptr %block_counts["
        << le_off << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
    out << "  " << fresh_ssa() << " = llvm.atomicrmw add " << le_p << ", " << fl2
        << " syncscope(\"agent\") monotonic : !llvm.ptr, i64\n";
    // observable_ones
    for (uint32_t i = 0; i < std::min(flat.num_observables, kMaxObs); ++i) {
        // Load & reduce obs[i] from LDS
        std::string oi_base = emit_const_i64(out, i * 4);
        std::string oi_ptr = fresh_ssa();
        out << "  " << oi_ptr << " = llvm.getelementptr inbounds %lds_red_o["
            << oi_base << "] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
        std::string oi_v0 = fresh_ssa();
        out << "  " << oi_v0 << " = llvm.load " << oi_ptr << " : !llvm.ptr -> i64\n";
        std::string oi1_ptr = fresh_ssa();
        std::string oi1_off = emit_const_i64(out, i * 4 + 1);
        out << "  " << oi1_ptr << " = llvm.getelementptr inbounds %lds_red_o["
            << oi1_off << "] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
        std::string oi_v1 = fresh_ssa();
        out << "  " << oi_v1 << " = llvm.load " << oi1_ptr << " : !llvm.ptr -> i64\n";
        std::string oi2_ptr = fresh_ssa();
        std::string oi2_off = emit_const_i64(out, i * 4 + 2);
        out << "  " << oi2_ptr << " = llvm.getelementptr inbounds %lds_red_o["
            << oi2_off << "] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
        std::string oi_v2 = fresh_ssa();
        out << "  " << oi_v2 << " = llvm.load " << oi2_ptr << " : !llvm.ptr -> i64\n";
        std::string oi3_ptr = fresh_ssa();
        std::string oi3_off = emit_const_i64(out, i * 4 + 3);
        out << "  " << oi3_ptr << " = llvm.getelementptr inbounds %lds_red_o["
            << oi3_off << "] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
        std::string oi_v3 = fresh_ssa();
        out << "  " << oi_v3 << " = llvm.load " << oi3_ptr << " : !llvm.ptr -> i64\n";
        std::string oi_s01 = fresh_ssa();
        out << "  " << oi_s01 << " = llvm.add " << oi_v0 << ", " << oi_v1 << " : i64\n";
        std::string oi_s23 = fresh_ssa();
        out << "  " << oi_s23 << " = llvm.add " << oi_v2 << ", " << oi_v3 << " : i64\n";
        std::string oi_total = fresh_ssa();
        out << "  " << oi_total << " = llvm.add " << oi_s01 << ", " << oi_s23 << " : i64\n";
        uint32_t obs_byte_offset = 16 + i * 8;
        std::string obs_off = emit_const_i64(out, obs_byte_offset);
        std::string obs_gep = fresh_ssa();
        out << "  " << obs_gep << " = llvm.getelementptr %block_counts["
            << obs_off << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        out << "  " << fresh_ssa() << " = llvm.atomicrmw add " << obs_gep << ", " << oi_total
            << " syncscope(\"agent\") monotonic : !llvm.ptr, i64\n";
    }
    out << "  llvm.br ^" << lbl_t0_done << "\n";
    out << "^" << lbl_t0_done << ":\n";
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
    cooperative_mode = true;
    lds_amplitudes = true;

    std::ostringstream out;
    uint32_t num_amps = 1u << flat.peak_rank;

    // Module header + intrinsic declarations
    out << "module attributes {llvm.target_triple = \"amdgcn-amd-amdhsa\"} {\n\n";
    emit_gpu_intrinsic_decls(out);

    // LDS globals (address space 3)
    emit_lds_global(out, "lds_v", "!llvm.struct<(f32, f32)>", num_amps);
    emit_lds_global(out, "lds_scratch", "!llvm.struct<(f32, f32)>", num_amps);
    emit_lds_global(out, "lds_px", "i64", 2);
    emit_lds_global(out, "lds_pz", "i64", 2);
    emit_lds_global(out, "lds_active_k", "i32", 1);
    emit_lds_global(out, "lds_discarded", "i8", 1);
    emit_lds_global(out, "lds_meas", "i8", kMaxMeas);
    emit_lds_global(out, "lds_obs", "i8", kMaxObs);
    emit_lds_global(out, "lds_mred0", "f64", 256);
    emit_lds_global(out, "lds_mred1", "f64", 256);
    emit_lds_global(out, "lds_mbranch", "i8", 1);
    out << "\n";

    // Kernel function
    out << "llvm.func amdgpu_kernelcc @compiled_mlir_kernel_coop(\n"
        << "    %shot_offset: i64 {llvm.inreg}, %shots: i64 {llvm.inreg}, %seed: i64 {llvm.inreg},\n"
        << "    %block_counts: !llvm.ptr {llvm.noalias},\n"
        << "    %noise_hazards_ptr: !llvm.ptr {llvm.noalias}, %noise_sites_ptr: !llvm.ptr {llvm.noalias},\n"
        << "    %noise_channels_ptr: !llvm.ptr {llvm.noalias}, %num_noise_sites: i64 {llvm.inreg},\n"
        << "    %num_obs: i32 {llvm.inreg}, %num_exp: i32 {llvm.inreg}) -> ()\n"
        << "  attributes {\"amdgpu-flat-work-group-size\"=\"256,256\", \"denormal-fp-math-f32\"=\"preserve-sign\", \"uniform-work-group-size\"=\"true\", \"amdgpu-no-implicitarg-ptr\"} {\n"
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

    // Private allocations in entry block (MUST be before any branch)
    out << "  %c4_i32 = llvm.mlir.constant(4 : i32) : i32\n";
    out << "  %c256_i32 = llvm.mlir.constant(256 : i32) : i32\n";
    out << "  %nni_ptr_p5 = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr<5>\n";
    out << "  %nni_ptr = llvm.addrspacecast %nni_ptr_p5 : !llvm.ptr<5> to !llvm.ptr\n";
    out << "  %rng_ptr_p5 = llvm.alloca %c4_i32 x i64 : (i32) -> !llvm.ptr<5>\n";
    out << "  %rng_ptr = llvm.addrspacecast %rng_ptr_p5 : !llvm.ptr<5> to !llvm.ptr\n";
    out << "  %sm_tmp_p5 = llvm.alloca %c1_i32 x i64 : (i32) -> !llvm.ptr<5>\n";
    out << "  %sm_tmp = llvm.addrspacecast %sm_tmp_p5 : !llvm.ptr<5> to !llvm.ptr\n";

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
    out << "  %v_ptr = llvm.mlir.addressof @lds_v : !llvm.ptr<3>\n";
    out << "  %lds_scratch_as3 = llvm.mlir.addressof @lds_scratch : !llvm.ptr<3>\n";
    out << "  %scratch_ptr = llvm.addrspacecast %lds_scratch_as3 : !llvm.ptr<3> to !llvm.ptr\n";
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
    out << "  " << vp_init << " = llvm.getelementptr inbounds %v_ptr[%init_i] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.struct<(f32, f32)>\n";
    out << "  llvm.store %zero_c2, " << vp_init << " : !llvm.struct<(f32, f32)>, !llvm.ptr<3>\n";
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
    out << "  %v0_ptr_coop = llvm.getelementptr inbounds %v_ptr[%c0_i64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.struct<(f32, f32)>\n";
    out << "  %init_one = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
    out << "  %init_one1 = llvm.insertvalue %f_one, %init_one[0] : !llvm.struct<(f32, f32)>\n";
    out << "  %init_one2 = llvm.insertvalue %f_zero, %init_one1[1] : !llvm.struct<(f32, f32)>\n";
    out << "  llvm.store %init_one2, %v0_ptr_coop : !llvm.struct<(f32, f32)>, !llvm.ptr<3>\n";
    out << "  llvm.br ^" << t0_done << "\n";
    out << "^" << t0_done << ":\n";
    emit_barrier(out);

    // Initialize meas[] to zero cooperatively
    if (flat.total_meas_slots > 0) {
        std::string meas_count = emit_const_i64(out, flat.total_meas_slots);
        std::string mz_hdr = fresh_label("cmz_hdr"), mz_body = fresh_label("cmz_body"), mz_done = fresh_label("cmz_done");
        out << "  llvm.br ^" << mz_hdr << "(%tidx : i64)\n";
        out << "^" << mz_hdr << "(%cmz_i: i64):\n";
        std::string mz_cond = fresh_ssa();
        out << "  " << mz_cond << " = llvm.icmp \"ult\" %cmz_i, " << meas_count << " : i64\n";
        out << "  llvm.cond_br " << mz_cond << ", ^" << mz_body << ", ^" << mz_done << "\n";
        out << "^" << mz_body << ":\n";
        std::string mz_ptr = fresh_ssa();
        out << "  " << mz_ptr << " = llvm.getelementptr inbounds %meas_ptr[%cmz_i] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        out << "  llvm.store %c0_i8, " << mz_ptr << " : i8, !llvm.ptr\n";
        std::string mz_next = fresh_ssa();
        out << "  " << mz_next << " = llvm.add %cmz_i, %c256_i64 : i64\n";
        out << "  llvm.br ^" << mz_hdr << "(" << mz_next << " : i64)\n";
        out << "^" << mz_done << ":\n";
    }

    // Initialize obs[] to zero (thread-0 only, small array)
    for (uint32_t i = 0; i < std::min(flat.num_observables, kMaxObs); ++i) {
        std::string idx = fresh_ssa();
        out << "  " << idx << " = llvm.mlir.constant(" << i << " : i64) : i64\n";
        std::string ptr = fresh_ssa();
        out << "  " << ptr << " = llvm.getelementptr inbounds %obs_ptr[" << idx
            << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        out << "  llvm.store %c0_i8, " << ptr << " : i8, !llvm.ptr\n";
    }

    // Initialize next_noise_idx = 0
    out << "  llvm.store %c0_i32, %nni_ptr : i32, !llvm.ptr\n";

    // Seed RNG: shot_id = shot_offset + bidx (one shot per workgroup in coop)
    std::string shot_id_coop = fresh_ssa();
    out << "  " << shot_id_coop << " = llvm.add %shot_offset, %bidx : i64\n";
    emit_rng_seed(out, "%seed", shot_id_coop);

    // Initial noise draw (if circuit has noise)
    if (!flat.noise_sites.empty()) {
        emit_draw_next_noise(out, flat);
    }
    emit_barrier(out);

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
    std::string coop_init = "%tidx";
    std::string coop_step = "%c256_i64";
    out << "  // --- Coop instruction sequence (" << flat.instrs.size() << " ops) ---\n";
    g_emit_ak = 0;

    for (size_t pc = 0; pc < flat.instrs.size(); ++pc) {
        const GpuInstr& ins = flat.instrs[pc];
        bool sign = (ins.flags & kFlagSign) != 0;
        bool identity = (ins.flags & kFlagIdentity) != 0;
        auto op = static_cast<Opcode>(ins.opcode);

        out << "  // op[" << pc << "] opcode=" << (unsigned)ins.opcode
            << " active_k=" << g_emit_ak << "\n";

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
        switch (op) {
            case Opcode::OP_EXPAND:
            case Opcode::OP_EXPAND_T:
            case Opcode::OP_EXPAND_T_DAG:
            case Opcode::OP_EXPAND_ROT:
                g_emit_ak++;
                break;
            case Opcode::OP_MEAS_ACTIVE_DIAGONAL:
            case Opcode::OP_MEAS_ACTIVE_INTERFERE:
            case Opcode::OP_SWAP_MEAS_INTERFERE:
                if (g_emit_ak > 0) g_emit_ak--;
                break;
            default:
                break;
        }
    }

    out << "  // --- End coop instruction sequence ---\n";

    // Result aggregation (thread-0 only: check discarded, atomic-add to block_counts)
    emit_barrier(out);
    std::string coop_is_t0_agg = fresh_ssa();
    std::string coop_t0_agg = fresh_label("coop_t0_agg");
    std::string coop_agg_done = fresh_label("coop_agg_done");
    out << "  " << coop_is_t0_agg << " = llvm.icmp \"eq\" %tidx_i32, %c0_i32 : i32\n";
    out << "  llvm.cond_br " << coop_is_t0_agg << ", ^" << coop_t0_agg << ", ^" << coop_agg_done << "\n";
    out << "^" << coop_t0_agg << ":\n";

    std::string coop_disc = fresh_ssa();
    out << "  " << coop_disc << " = llvm.load %discarded_ptr : !llvm.ptr -> i8\n";
    std::string coop_is_disc = fresh_ssa();
    out << "  " << coop_is_disc << " = llvm.icmp \"ne\" " << coop_disc << ", %c0_i8 : i8\n";
    std::string coop_lbl_agg = fresh_label("coop_agg");
    std::string coop_lbl_skip = fresh_label("coop_agg_skip");
    out << "  llvm.cond_br " << coop_is_disc << ", ^" << coop_lbl_skip << ", ^" << coop_lbl_agg << "\n";
    out << "^" << coop_lbl_agg << ":\n";

    // atomic add passed++
    out << "  " << fresh_ssa() << " = llvm.atomicrmw add %block_counts, %c1_i64"
        << " syncscope(\"agent\") monotonic : !llvm.ptr, i64\n";

    // Check observables
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
        std::string lbl_obs_inc = fresh_label("cobs_inc");
        std::string lbl_obs_done = fresh_label("cobs_done");
        out << "  llvm.cond_br " << obs_ne << ", ^" << lbl_obs_inc << ", ^" << lbl_obs_done << "\n";
        out << "^" << lbl_obs_inc << ":\n";

        uint32_t obs_byte_offset = 16 + i * 8;
        std::string obs_off_val = fresh_ssa();
        out << "  " << obs_off_val << " = llvm.mlir.constant(" << obs_byte_offset << " : i64) : i64\n";
        std::string obs_ones_ptr = fresh_ssa();
        out << "  " << obs_ones_ptr << " = llvm.getelementptr %block_counts["
            << obs_off_val << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        out << "  " << fresh_ssa() << " = llvm.atomicrmw add " << obs_ones_ptr
            << ", %c1_i64 syncscope(\"agent\") monotonic : !llvm.ptr, i64\n";

        std::string le_ptr = fresh_ssa();
        out << "  " << le_ptr << " = llvm.mlir.constant(8 : i64) : i64\n";
        std::string le_gep = fresh_ssa();
        out << "  " << le_gep << " = llvm.getelementptr %block_counts["
            << le_ptr << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        out << "  " << fresh_ssa() << " = llvm.atomicrmw add " << le_gep
            << ", %c1_i64 syncscope(\"agent\") monotonic : !llvm.ptr, i64\n";

        out << "  llvm.br ^" << lbl_obs_done << "\n";
        out << "^" << lbl_obs_done << ":\n";
    }

    out << "  llvm.br ^" << coop_lbl_skip << "\n";
    out << "^" << coop_lbl_skip << ":\n";
    out << "  llvm.br ^" << coop_agg_done << "\n";
    out << "^" << coop_agg_done << ":\n";

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
    cooperative_mode = true;
    lds_amplitudes = false;

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
    emit_lds_global(out, "lds_mred0", "f64", 256);
    emit_lds_global(out, "lds_mred1", "f64", 256);
    emit_lds_global(out, "lds_mbranch", "i8", 1);
    out << "\n";

    // Global kernel with HBM pointers
    out << "llvm.func amdgpu_kernelcc @compiled_mlir_kernel_global(\n"
        << "    %shot_offset: i64 {llvm.inreg}, %shots: i64 {llvm.inreg}, %seed: i64 {llvm.inreg},\n"
        << "    %global_v: !llvm.ptr {llvm.noalias}, %global_scratch: !llvm.ptr {llvm.noalias},\n"
        << "    %work_counter: !llvm.ptr {llvm.noalias},\n"
        << "    %block_counts: !llvm.ptr {llvm.noalias},\n"
        << "    %noise_hazards_ptr: !llvm.ptr {llvm.noalias}, %noise_sites_ptr: !llvm.ptr {llvm.noalias},\n"
        << "    %noise_channels_ptr: !llvm.ptr {llvm.noalias}, %num_noise_sites: i64 {llvm.inreg},\n"
        << "    %num_obs: i32 {llvm.inreg}, %num_exp: i32 {llvm.inreg}) -> ()\n"
        << "  attributes {\"amdgpu-flat-work-group-size\"=\"256,256\", \"denormal-fp-math-f32\"=\"preserve-sign\", \"uniform-work-group-size\"=\"true\", \"amdgpu-no-implicitarg-ptr\"} {\n"
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
    out << "  %scratch_ptr = llvm.getelementptr inbounds %global_scratch[" << v_off
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

    // Private allocations for per-thread noise/RNG state (addrspace 5) — in entry block
    out << "  %c4_i32 = llvm.mlir.constant(4 : i32) : i32\n";
    out << "  %c256_i32 = llvm.mlir.constant(256 : i32) : i32\n";
    out << "  %nni_ptr_p5 = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr<5>\n";
    out << "  %nni_ptr = llvm.addrspacecast %nni_ptr_p5 : !llvm.ptr<5> to !llvm.ptr\n";
    out << "  %rng_ptr_p5 = llvm.alloca %c4_i32 x i64 : (i32) -> !llvm.ptr<5>\n";
    out << "  %rng_ptr = llvm.addrspacecast %rng_ptr_p5 : !llvm.ptr<5> to !llvm.ptr\n";
    out << "  %sm_tmp_p5 = llvm.alloca %c1_i32 x i64 : (i32) -> !llvm.ptr<5>\n";
    out << "  %sm_tmp = llvm.addrspacecast %sm_tmp_p5 : !llvm.ptr<5> to !llvm.ptr\n";

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
        << ", %c1_i64 syncscope(\"agent\") monotonic : !llvm.ptr, i64\n";
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

    // Initialize meas[] to zero cooperatively
    if (flat.total_meas_slots > 0) {
        std::string meas_count = emit_const_i64(out, flat.total_meas_slots);
        std::string gmz_hdr = fresh_label("gmz_hdr"), gmz_body = fresh_label("gmz_body"), gmz_done = fresh_label("gmz_done");
        out << "  llvm.br ^" << gmz_hdr << "(%tidx : i64)\n";
        out << "^" << gmz_hdr << "(%gmz_i: i64):\n";
        std::string gmz_cond = fresh_ssa();
        out << "  " << gmz_cond << " = llvm.icmp \"ult\" %gmz_i, " << meas_count << " : i64\n";
        out << "  llvm.cond_br " << gmz_cond << ", ^" << gmz_body << ", ^" << gmz_done << "\n";
        out << "^" << gmz_body << ":\n";
        std::string gmz_ptr = fresh_ssa();
        out << "  " << gmz_ptr << " = llvm.getelementptr inbounds %meas_ptr[%gmz_i] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        out << "  llvm.store %c0_i8, " << gmz_ptr << " : i8, !llvm.ptr\n";
        std::string gmz_next = fresh_ssa();
        out << "  " << gmz_next << " = llvm.add %gmz_i, %c256_i64 : i64\n";
        out << "  llvm.br ^" << gmz_hdr << "(" << gmz_next << " : i64)\n";
        out << "^" << gmz_done << ":\n";
    }

    // Initialize obs[] to zero
    for (uint32_t i = 0; i < std::min(flat.num_observables, kMaxObs); ++i) {
        std::string idx = fresh_ssa();
        out << "  " << idx << " = llvm.mlir.constant(" << i << " : i64) : i64\n";
        std::string ptr = fresh_ssa();
        out << "  " << ptr << " = llvm.getelementptr inbounds %obs_ptr[" << idx
            << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        out << "  llvm.store %c0_i8, " << ptr << " : i8, !llvm.ptr\n";
    }

    // Initialize next_noise_idx = 0
    out << "  llvm.store %c0_i32, %nni_ptr : i32, !llvm.ptr\n";

    // Seed RNG: shot_id = shot_offset + batch_shot_id
    std::string g_my_shot_seed = fresh_ssa();
    out << "  " << g_my_shot_seed << " = llvm.load %batch_shot_id_ptr : !llvm.ptr -> i64\n";
    std::string g_shot_id = fresh_ssa();
    out << "  " << g_shot_id << " = llvm.add %shot_offset, " << g_my_shot_seed << " : i64\n";
    emit_rng_seed(out, "%seed", g_shot_id);

    // Initial noise draw (if circuit has noise)
    if (!flat.noise_sites.empty()) {
        emit_draw_next_noise(out, flat);
    }
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
    std::string coop_init = "%tidx";
    std::string coop_step = "%c256_i64";
    out << "  // --- Global instruction sequence (" << flat.instrs.size() << " ops) ---\n";
    g_emit_ak = 0;

    for (size_t pc = 0; pc < flat.instrs.size(); ++pc) {
        const GpuInstr& ins = flat.instrs[pc];
        bool sign = (ins.flags & kFlagSign) != 0;
        bool identity = (ins.flags & kFlagIdentity) != 0;
        auto op = static_cast<Opcode>(ins.opcode);

        out << "  // op[" << pc << "] opcode=" << (unsigned)ins.opcode
            << " active_k=" << g_emit_ak << "\n";

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
        switch (op) {
            case Opcode::OP_EXPAND:
            case Opcode::OP_EXPAND_T:
            case Opcode::OP_EXPAND_T_DAG:
            case Opcode::OP_EXPAND_ROT:
                g_emit_ak++;
                break;
            case Opcode::OP_MEAS_ACTIVE_DIAGONAL:
            case Opcode::OP_MEAS_ACTIVE_INTERFERE:
            case Opcode::OP_SWAP_MEAS_INTERFERE:
                if (g_emit_ak > 0) g_emit_ak--;
                break;
            default:
                break;
        }
    }

    out << "  // --- End global instruction sequence ---\n";

    // Result aggregation (thread-0 only)
    emit_barrier(out);
    std::string g_is_t0_agg = fresh_ssa();
    std::string g_t0_agg = fresh_label("g_t0_agg");
    std::string g_agg_done = fresh_label("g_agg_done");
    out << "  " << g_is_t0_agg << " = llvm.icmp \"eq\" %tidx_i32, %c0_i32 : i32\n";
    out << "  llvm.cond_br " << g_is_t0_agg << ", ^" << g_t0_agg << ", ^" << g_agg_done << "\n";
    out << "^" << g_t0_agg << ":\n";

    std::string g_disc = fresh_ssa();
    out << "  " << g_disc << " = llvm.load %discarded_ptr : !llvm.ptr -> i8\n";
    std::string g_is_disc = fresh_ssa();
    out << "  " << g_is_disc << " = llvm.icmp \"ne\" " << g_disc << ", %c0_i8 : i8\n";
    std::string g_lbl_agg = fresh_label("g_agg");
    std::string g_lbl_skip = fresh_label("g_agg_skip");
    out << "  llvm.cond_br " << g_is_disc << ", ^" << g_lbl_skip << ", ^" << g_lbl_agg << "\n";
    out << "^" << g_lbl_agg << ":\n";

    // atomic add passed++
    out << "  " << fresh_ssa() << " = llvm.atomicrmw add %block_counts, %c1_i64"
        << " syncscope(\"agent\") monotonic : !llvm.ptr, i64\n";

    // Check observables
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
        std::string lbl_obs_inc = fresh_label("gobs_inc");
        std::string lbl_obs_done = fresh_label("gobs_done");
        out << "  llvm.cond_br " << obs_ne << ", ^" << lbl_obs_inc << ", ^" << lbl_obs_done << "\n";
        out << "^" << lbl_obs_inc << ":\n";

        uint32_t obs_byte_offset = 16 + i * 8;
        std::string obs_off_val = fresh_ssa();
        out << "  " << obs_off_val << " = llvm.mlir.constant(" << obs_byte_offset << " : i64) : i64\n";
        std::string obs_ones_ptr = fresh_ssa();
        out << "  " << obs_ones_ptr << " = llvm.getelementptr %block_counts["
            << obs_off_val << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        out << "  " << fresh_ssa() << " = llvm.atomicrmw add " << obs_ones_ptr
            << ", %c1_i64 syncscope(\"agent\") monotonic : !llvm.ptr, i64\n";

        std::string le_ptr = fresh_ssa();
        out << "  " << le_ptr << " = llvm.mlir.constant(8 : i64) : i64\n";
        std::string le_gep = fresh_ssa();
        out << "  " << le_gep << " = llvm.getelementptr %block_counts["
            << le_ptr << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        out << "  " << fresh_ssa() << " = llvm.atomicrmw add " << le_gep
            << ", %c1_i64 syncscope(\"agent\") monotonic : !llvm.ptr, i64\n";

        out << "  llvm.br ^" << lbl_obs_done << "\n";
        out << "^" << lbl_obs_done << ":\n";
    }

    out << "  llvm.br ^" << g_lbl_skip << "\n";
    out << "^" << g_lbl_skip << ":\n";
    out << "  llvm.br ^" << g_agg_done << "\n";
    out << "^" << g_agg_done << ":\n";

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
