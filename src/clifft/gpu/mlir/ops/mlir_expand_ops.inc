// mlir_expand_ops.inc — MLIR emission for subspace expansion operations
//
// Included inside the instruction switch in mlir_emit.cc.
// Available in scope: out, fresh_ssa, fresh_label, kInvSqrt2,
//                     emit_bit_get, emit_scatter_bits_1,
//                     emit_load_v, emit_store_v, emit_cscale_f64,
//                     emit_cmul_const, ins (current GpuInstr)

case Opcode::OP_EXPAND: {
    out << "  // expand: k -> k+1\n";
    std::string ak = fresh_ssa(), ak64 = fresh_ssa(), half = fresh_ssa();
    out << "  " << ak << " = llvm.load %active_k_ptr : !llvm.ptr -> i32\n";
    out << "  " << ak64 << " = llvm.zext " << ak << " : i32 to i64\n";
    out << "  " << half << " = llvm.shl %c1_i64, " << ak64 << " : i64\n";

    std::string hdr = fresh_label("exp_hdr"), body = fresh_label("exp_body"), exit_lbl = fresh_label("exp_exit");
    out << "  llvm.br ^" << hdr << "(%c0_i64 : i64)\n";
    out << "^" << hdr << "(%exp_i: i64):\n";
    std::string cond = fresh_ssa();
    out << "  " << cond << " = llvm.icmp \"ult\" %exp_i, " << half << " : i64\n";
    out << "  llvm.cond_br " << cond << ", ^" << body << ", ^" << exit_lbl << "\n";
    out << "^" << body << ":\n";
    // v[i+half] = cscale(v[i], inv_sqrt2)
    std::string vc = emit_load_v("%exp_i");
    std::string hi_idx = fresh_ssa();
    out << "  " << hi_idx << " = llvm.add %exp_i, " << half << " : i64\n";
    std::string scaled = emit_cscale_f64(vc, kInvSqrt2);
    emit_store_v(hi_idx, scaled);
    // v[i] = cscale(v[i], inv_sqrt2) — reuse already-scaled value
    emit_store_v("%exp_i", scaled);
    std::string i_next = fresh_ssa();
    out << "  " << i_next << " = llvm.add %exp_i, %c1_i64 : i64\n";
    out << "  llvm.br ^" << hdr << "(" << i_next << " : i64)\n";
    out << "^" << exit_lbl << ":\n";
    // active_k++
    std::string ak_new = fresh_ssa();
    out << "  " << ak_new << " = llvm.add " << ak << ", %c1_i32 : i32\n";
    out << "  llvm.store " << ak_new << ", %active_k_ptr : i32, !llvm.ptr\n";
    break;
}
case Opcode::OP_EXPAND_T:
case Opcode::OP_EXPAND_T_DAG: {
    bool is_dag = (op == Opcode::OP_EXPAND_T_DAG);
    out << "  // expand_t" << (is_dag ? "_dag" : "") << " axis=" << ins.axis_1 << "\n";
    uint32_t axis = ins.axis_1;
    char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", axis);
    std::string ax_i32 = fresh_ssa();
    out << "  " << ax_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
    std::string pxv = emit_bit_get("%px_ptr", ax_i32);

    // T = e^(i*pi/4) = (cos(pi/4), sin(pi/4)) = (inv_sqrt2, inv_sqrt2)
    // T_dag = e^(-i*pi/4) = (inv_sqrt2, -inv_sqrt2)
    // If px bit set, use conjugate
    double base_im = is_dag ? -kInvSqrt2 : kInvSqrt2;
    char im_pos_buf[64], im_neg_buf[64];
    snprintf(im_pos_buf, sizeof(im_pos_buf), "%.17e", (float)base_im);
    snprintf(im_neg_buf, sizeof(im_neg_buf), "%.17e", (float)-base_im);
    std::string im_pos = fresh_ssa(), im_neg = fresh_ssa(), imag = fresh_ssa();
    out << "  " << im_pos << " = llvm.mlir.constant(" << im_pos_buf << " : f32) : f32\n";
    out << "  " << im_neg << " = llvm.mlir.constant(" << im_neg_buf << " : f32) : f32\n";
    out << "  " << imag << " = llvm.select " << pxv << ", " << im_neg << ", " << im_pos << " : i1, f32\n";
    char re_buf[64]; snprintf(re_buf, sizeof(re_buf), "%.17e", (float)kInvSqrt2);
    std::string re_c = fresh_ssa();
    out << "  " << re_c << " = llvm.mlir.constant(" << re_buf << " : f32) : f32\n";

    std::string ak = fresh_ssa(), ak64 = fresh_ssa(), half = fresh_ssa();
    out << "  " << ak << " = llvm.load %active_k_ptr : !llvm.ptr -> i32\n";
    out << "  " << ak64 << " = llvm.zext " << ak << " : i32 to i64\n";
    out << "  " << half << " = llvm.shl %c1_i64, " << ak64 << " : i64\n";
    std::string axis64 = fresh_ssa(), axis_bit = fresh_ssa();
    out << "  " << axis64 << " = llvm.mlir.constant(" << abuf << " : i64) : i64\n";
    out << "  " << axis_bit << " = llvm.shl %c1_i64, " << axis64 << " : i64\n";

    std::string hdr = fresh_label("expt_hdr"), body_lbl = fresh_label("expt_body"), exit_lbl = fresh_label("expt_exit");
    out << "  llvm.br ^" << hdr << "(%c0_i64 : i64)\n";
    out << "^" << hdr << "(%expt_i: i64):\n";
    std::string cond = fresh_ssa();
    out << "  " << cond << " = llvm.icmp \"ult\" %expt_i, " << half << " : i64\n";
    out << "  llvm.cond_br " << cond << ", ^" << body_lbl << ", ^" << exit_lbl << "\n";
    out << "^" << body_lbl << ":\n";
    std::string vc = emit_load_v("%expt_i");
    std::string hi_idx = fresh_ssa();
    out << "  " << hi_idx << " = llvm.add %expt_i, " << half << " : i64\n";
    // Check if this amplitude sits on the axis bit
    std::string has_axis = fresh_ssa();
    out << "  " << has_axis << " = llvm.and %expt_i, " << axis_bit << " : i64\n";
    std::string is_upper = fresh_ssa();
    out << "  " << is_upper << " = llvm.icmp \"ne\" " << has_axis << ", %c0_i64 : i64\n";
    // Lower half: v[i] = cscale(v[i], inv_sqrt2), v[i+half] = cscale(v[i], inv_sqrt2)
    // Upper half (axis bit set): apply T phase to v[i+half]
    // Both get scaled by inv_sqrt2
    std::string scaled = emit_cscale_f64(vc, kInvSqrt2);
    emit_store_v("%expt_i", scaled);
    // For the upper half copy: if axis bit is set, also apply T phase
    std::string then_lbl = fresh_label("expt_phase"), else_lbl = fresh_label("expt_nophase"), merge_lbl = fresh_label("expt_merge");
    out << "  llvm.cond_br " << is_upper << ", ^" << then_lbl << ", ^" << else_lbl << "\n";
    out << "^" << then_lbl << ":\n";
    // Apply T phase to scaled value: cmul(scaled, (re_c, imag))
    {
        std::string s_re = fresh_ssa(), s_im = fresh_ssa();
        out << "  " << s_re << " = llvm.extractvalue " << scaled << "[0] : !llvm.struct<(f32, f32)>\n";
        out << "  " << s_im << " = llvm.extractvalue " << scaled << "[1] : !llvm.struct<(f32, f32)>\n";
        std::string t0 = fresh_ssa(), t1 = fresh_ssa(), ro = fresh_ssa();
        out << "  " << t0 << " = llvm.fmul " << s_re << ", " << re_c << " : f32\n";
        out << "  " << t1 << " = llvm.fmul " << s_im << ", " << imag << " : f32\n";
        out << "  " << ro << " = llvm.fsub " << t0 << ", " << t1 << " : f32\n";
        std::string t2 = fresh_ssa(), t3 = fresh_ssa(), io = fresh_ssa();
        out << "  " << t2 << " = llvm.fmul " << s_re << ", " << imag << " : f32\n";
        out << "  " << t3 << " = llvm.fmul " << s_im << ", " << re_c << " : f32\n";
        out << "  " << io << " = llvm.fadd " << t2 << ", " << t3 << " : f32\n";
        std::string r0 = fresh_ssa(), r1 = fresh_ssa(), phased = fresh_ssa();
        out << "  " << r0 << " = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
        out << "  " << r1 << " = llvm.insertvalue " << ro << ", " << r0 << "[0] : !llvm.struct<(f32, f32)>\n";
        out << "  " << phased << " = llvm.insertvalue " << io << ", " << r1 << "[1] : !llvm.struct<(f32, f32)>\n";
        emit_store_v(hi_idx, phased);
    }
    out << "  llvm.br ^" << merge_lbl << "\n";
    out << "^" << else_lbl << ":\n";
    emit_store_v(hi_idx, scaled);
    out << "  llvm.br ^" << merge_lbl << "\n";
    out << "^" << merge_lbl << ":\n";
    std::string i_next = fresh_ssa();
    out << "  " << i_next << " = llvm.add %expt_i, %c1_i64 : i64\n";
    out << "  llvm.br ^" << hdr << "(" << i_next << " : i64)\n";
    out << "^" << exit_lbl << ":\n";
    // active_k++
    std::string ak_new = fresh_ssa();
    out << "  " << ak_new << " = llvm.add " << ak << ", %c1_i32 : i32\n";
    out << "  llvm.store " << ak_new << ", %active_k_ptr : i32, !llvm.ptr\n";
    break;
}
case Opcode::OP_EXPAND_ROT: {
    out << "  // expand_rot axis=" << ins.axis_1 << "\n";
    uint32_t axis = ins.axis_1;
    char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", axis);
    std::string ax_i32 = fresh_ssa();
    out << "  " << ax_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
    std::string pxv = emit_bit_get("%px_ptr", ax_i32);

    // Prepare phase: if px set, use conjugate
    char wre_buf[64], wim_buf[64], wim_neg_buf[64];
    snprintf(wre_buf, sizeof(wre_buf), "%.17e", (float)ins.weight_re);
    snprintf(wim_buf, sizeof(wim_buf), "%.17e", (float)ins.weight_im);
    snprintf(wim_neg_buf, sizeof(wim_neg_buf), "%.17e", (float)-ins.weight_im);
    std::string phs_re = fresh_ssa(), phs_im_pos = fresh_ssa(), phs_im_neg = fresh_ssa(), phs_im = fresh_ssa();
    out << "  " << phs_re << " = llvm.mlir.constant(" << wre_buf << " : f32) : f32\n";
    out << "  " << phs_im_pos << " = llvm.mlir.constant(" << wim_buf << " : f32) : f32\n";
    out << "  " << phs_im_neg << " = llvm.mlir.constant(" << wim_neg_buf << " : f32) : f32\n";
    out << "  " << phs_im << " = llvm.select " << pxv << ", " << phs_im_neg << ", " << phs_im_pos << " : i1, f32\n";

    std::string ak = fresh_ssa(), ak64 = fresh_ssa(), half = fresh_ssa();
    out << "  " << ak << " = llvm.load %active_k_ptr : !llvm.ptr -> i32\n";
    out << "  " << ak64 << " = llvm.zext " << ak << " : i32 to i64\n";
    out << "  " << half << " = llvm.shl %c1_i64, " << ak64 << " : i64\n";
    std::string axis64 = fresh_ssa(), axis_bit = fresh_ssa();
    out << "  " << axis64 << " = llvm.mlir.constant(" << abuf << " : i64) : i64\n";
    out << "  " << axis_bit << " = llvm.shl %c1_i64, " << axis64 << " : i64\n";

    std::string hdr = fresh_label("expr_hdr"), body_lbl = fresh_label("expr_body"), exit_lbl = fresh_label("expr_exit");
    out << "  llvm.br ^" << hdr << "(%c0_i64 : i64)\n";
    out << "^" << hdr << "(%expr_i: i64):\n";
    std::string cond = fresh_ssa();
    out << "  " << cond << " = llvm.icmp \"ult\" %expr_i, " << half << " : i64\n";
    out << "  llvm.cond_br " << cond << ", ^" << body_lbl << ", ^" << exit_lbl << "\n";
    out << "^" << body_lbl << ":\n";
    std::string vc = emit_load_v("%expr_i");
    std::string hi_idx = fresh_ssa();
    out << "  " << hi_idx << " = llvm.add %expr_i, " << half << " : i64\n";
    std::string has_axis = fresh_ssa();
    out << "  " << has_axis << " = llvm.and %expr_i, " << axis_bit << " : i64\n";
    std::string is_upper = fresh_ssa();
    out << "  " << is_upper << " = llvm.icmp \"ne\" " << has_axis << ", %c0_i64 : i64\n";
    std::string scaled = emit_cscale_f64(vc, kInvSqrt2);
    emit_store_v("%expr_i", scaled);
    std::string then_lbl = fresh_label("expr_phase"), else_lbl = fresh_label("expr_nophase"), merge_lbl = fresh_label("expr_merge");
    out << "  llvm.cond_br " << is_upper << ", ^" << then_lbl << ", ^" << else_lbl << "\n";
    out << "^" << then_lbl << ":\n";
    // Apply rotation phase to scaled value
    {
        std::string s_re = fresh_ssa(), s_im = fresh_ssa();
        out << "  " << s_re << " = llvm.extractvalue " << scaled << "[0] : !llvm.struct<(f32, f32)>\n";
        out << "  " << s_im << " = llvm.extractvalue " << scaled << "[1] : !llvm.struct<(f32, f32)>\n";
        std::string t0 = fresh_ssa(), t1 = fresh_ssa(), ro = fresh_ssa();
        out << "  " << t0 << " = llvm.fmul " << s_re << ", " << phs_re << " : f32\n";
        out << "  " << t1 << " = llvm.fmul " << s_im << ", " << phs_im << " : f32\n";
        out << "  " << ro << " = llvm.fsub " << t0 << ", " << t1 << " : f32\n";
        std::string t2 = fresh_ssa(), t3 = fresh_ssa(), io = fresh_ssa();
        out << "  " << t2 << " = llvm.fmul " << s_re << ", " << phs_im << " : f32\n";
        out << "  " << t3 << " = llvm.fmul " << s_im << ", " << phs_re << " : f32\n";
        out << "  " << io << " = llvm.fadd " << t2 << ", " << t3 << " : f32\n";
        std::string r0 = fresh_ssa(), r1 = fresh_ssa(), phased = fresh_ssa();
        out << "  " << r0 << " = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
        out << "  " << r1 << " = llvm.insertvalue " << ro << ", " << r0 << "[0] : !llvm.struct<(f32, f32)>\n";
        out << "  " << phased << " = llvm.insertvalue " << io << ", " << r1 << "[1] : !llvm.struct<(f32, f32)>\n";
        emit_store_v(hi_idx, phased);
    }
    out << "  llvm.br ^" << merge_lbl << "\n";
    out << "^" << else_lbl << ":\n";
    emit_store_v(hi_idx, scaled);
    out << "  llvm.br ^" << merge_lbl << "\n";
    out << "^" << merge_lbl << ":\n";
    std::string i_next = fresh_ssa();
    out << "  " << i_next << " = llvm.add %expr_i, %c1_i64 : i64\n";
    out << "  llvm.br ^" << hdr << "(" << i_next << " : i64)\n";
    out << "^" << exit_lbl << ":\n";
    std::string ak_new = fresh_ssa();
    out << "  " << ak_new << " = llvm.add " << ak << ", %c1_i32 : i32\n";
    out << "  llvm.store " << ak_new << ", %active_k_ptr : i32, !llvm.ptr\n";
    break;
}
