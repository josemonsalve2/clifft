// mlir_array_ops.inc — MLIR emission for array (amplitude sweep) operations
//
// Included inside the instruction switch in mlir_emit.cc.
// Available in scope: out, fresh_ssa, fresh_label, kInvSqrt2,
//                     emit_bit_get, emit_bit_xor, emit_bit_set,
//                     emit_scatter_bits_1, emit_load_v, emit_store_v,
//                     emit_cadd, emit_csub, emit_cscale_f64, emit_cmul_const,
//                     emit_array_h_static, emit_array_cnot_static,
//                     emit_apply_phase_static, ins (current GpuInstr)

case Opcode::OP_ARRAY_H: {
    emit_array_h_static(ins.axis_1);
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
    double kIsq2 = kInvSqrt2;
    char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", ins.axis_1);
    std::string ax_i32 = fresh_ssa();
    out << "  " << ax_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
    std::string pxv = emit_bit_get("%px_ptr", ax_i32);
    std::string im_pos = fresh_ssa(), im_neg = fresh_ssa(), imag = fresh_ssa();
    char pbuf[64], nbuf[64];
    snprintf(pbuf, sizeof(pbuf), "%.17e", (float)kIsq2);
    snprintf(nbuf, sizeof(nbuf), "%.17e", (float)-kIsq2);
    out << "  " << im_pos << " = llvm.mlir.constant(" << pbuf << " : f32) : f32\n";
    out << "  " << im_neg << " = llvm.mlir.constant(" << nbuf << " : f32) : f32\n";
    out << "  " << imag << " = llvm.select " << pxv << ", " << im_neg << ", " << im_pos << " : i1, f32\n";
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
    char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", ins.axis_1);
    std::string ax_i32 = fresh_ssa();
    out << "  " << ax_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
    std::string pxv = emit_bit_get("%px_ptr", ax_i32);
    double kIsq2 = kInvSqrt2;
    std::string im_pos = fresh_ssa(), im_neg = fresh_ssa(), imag = fresh_ssa();
    char pbuf[64], nbuf[64];
    snprintf(pbuf, sizeof(pbuf), "%.17e", (float)-kIsq2);
    snprintf(nbuf, sizeof(nbuf), "%.17e", (float)kIsq2);
    out << "  " << im_pos << " = llvm.mlir.constant(" << pbuf << " : f32) : f32\n";
    out << "  " << im_neg << " = llvm.mlir.constant(" << nbuf << " : f32) : f32\n";
    out << "  " << imag << " = llvm.select " << pxv << ", " << im_neg << ", " << im_pos << " : i1, f32\n";
    std::string ak = fresh_ssa(), ak64 = fresh_ssa(), ak_m1 = fresh_ssa(), iters = fresh_ssa();
    out << "  " << ak << " = llvm.load %active_k_ptr : !llvm.ptr -> i32\n";
    out << "  " << ak64 << " = llvm.zext " << ak << " : i32 to i64\n";
    out << "  " << ak_m1 << " = llvm.sub " << ak64 << ", %c1_i64 : i64\n";
    out << "  " << iters << " = llvm.shl %c1_i64, " << ak_m1 << " : i64\n";
    std::string axis64 = fresh_ssa(), axis_bit = fresh_ssa();
    out << "  " << axis64 << " = llvm.mlir.constant(" << abuf << " : i64) : i64\n";
    out << "  " << axis_bit << " = llvm.shl %c1_i64, " << axis64 << " : i64\n";
    char re_buf[64]; snprintf(re_buf, sizeof(re_buf), "%.17e", (float)kInvSqrt2);
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
case Opcode::OP_ARRAY_CZ: {
    out << "  // array_cz ctrl=" << ins.axis_1 << " tgt=" << ins.axis_2 << "\n";
    uint32_t ctrl = ins.axis_1, tgt = ins.axis_2;
    std::string ak = fresh_ssa(), ak64 = fresh_ssa(), ak_m2 = fresh_ssa(), iters = fresh_ssa();
    out << "  " << ak << " = llvm.load %active_k_ptr : !llvm.ptr -> i32\n";
    out << "  " << ak64 << " = llvm.zext " << ak << " : i32 to i64\n";
    out << "  " << ak_m2 << " = llvm.sub " << ak64 << ", llvm.mlir.constant(2 : i64) : i64\n";
    out << "  " << iters << " = llvm.shl %c1_i64, " << ak_m2 << " : i64\n";
    char cbuf[32], tbuf[32];
    snprintf(cbuf, sizeof(cbuf), "%u", ctrl);
    snprintf(tbuf, sizeof(tbuf), "%u", tgt);
    std::string c64 = fresh_ssa(), t64 = fresh_ssa(), c_bit = fresh_ssa(), t_bit = fresh_ssa();
    out << "  " << c64 << " = llvm.mlir.constant(" << cbuf << " : i64) : i64\n";
    out << "  " << t64 << " = llvm.mlir.constant(" << tbuf << " : i64) : i64\n";
    out << "  " << c_bit << " = llvm.shl %c1_i64, " << c64 << " : i64\n";
    out << "  " << t_bit << " = llvm.shl %c1_i64, " << t64 << " : i64\n";
    std::string ct_bit = fresh_ssa();
    out << "  " << ct_bit << " = llvm.or " << c_bit << ", " << t_bit << " : i64\n";

    std::string hdr = fresh_label("cz_hdr"), body = fresh_label("cz_body"), exit_lbl = fresh_label("cz_exit");
    out << "  llvm.br ^" << hdr << "(%c0_i64 : i64)\n";
    out << "^" << hdr << "(%cz_i: i64):\n";
    std::string cond = fresh_ssa();
    out << "  " << cond << " = llvm.icmp \"ult\" %cz_i, " << iters << " : i64\n";
    out << "  llvm.cond_br " << cond << ", ^" << body << ", ^" << exit_lbl << "\n";
    out << "^" << body << ":\n";
    // scatter_bits_2 to skip both control and target positions
    std::string base = emit_scatter_bits_2("%cz_i", c64, t64);
    std::string idx = fresh_ssa();
    out << "  " << idx << " = llvm.or " << base << ", " << ct_bit << " : i64\n";
    // Negate v[idx]: multiply by -1
    std::string vc = emit_load_v(idx);
    std::string vc_re = fresh_ssa(), vc_im = fresh_ssa();
    out << "  " << vc_re << " = llvm.extractvalue " << vc << "[0] : !llvm.struct<(f32, f32)>\n";
    out << "  " << vc_im << " = llvm.extractvalue " << vc << "[1] : !llvm.struct<(f32, f32)>\n";
    std::string neg_re = fresh_ssa(), neg_im = fresh_ssa();
    out << "  " << neg_re << " = llvm.fneg " << vc_re << " : f32\n";
    out << "  " << neg_im << " = llvm.fneg " << vc_im << " : f32\n";
    std::string r0 = fresh_ssa(), r1 = fresh_ssa(), res = fresh_ssa();
    out << "  " << r0 << " = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
    out << "  " << r1 << " = llvm.insertvalue " << neg_re << ", " << r0 << "[0] : !llvm.struct<(f32, f32)>\n";
    out << "  " << res << " = llvm.insertvalue " << neg_im << ", " << r1 << "[1] : !llvm.struct<(f32, f32)>\n";
    emit_store_v(idx, res);
    std::string i_next = fresh_ssa();
    out << "  " << i_next << " = llvm.add %cz_i, %c1_i64 : i64\n";
    out << "  llvm.br ^" << hdr << "(" << i_next << " : i64)\n";
    out << "^" << exit_lbl << ":\n";
    // Frame update: pz[axis_1] ^= px[axis_2]; pz[axis_2] ^= px[axis_1]
    {
        std::string a_i32 = fresh_ssa(), b_i32 = fresh_ssa();
        out << "  " << a_i32 << " = llvm.mlir.constant(" << cbuf << " : i32) : i32\n";
        out << "  " << b_i32 << " = llvm.mlir.constant(" << tbuf << " : i32) : i32\n";
        std::string px_a = emit_bit_get("%px_ptr", b_i32);
        std::string px_b = emit_bit_get("%px_ptr", a_i32);
        emit_bit_xor("%pz_ptr", a_i32, px_a);
        emit_bit_xor("%pz_ptr", b_i32, px_b);
    }
    break;
}
case Opcode::OP_ARRAY_SWAP: {
    out << "  // array_swap axis_1=" << ins.axis_1 << " axis_2=" << ins.axis_2 << "\n";
    uint32_t a1 = ins.axis_1, a2 = ins.axis_2;
    std::string ak = fresh_ssa(), ak64 = fresh_ssa(), ak_m2 = fresh_ssa(), iters = fresh_ssa();
    out << "  " << ak << " = llvm.load %active_k_ptr : !llvm.ptr -> i32\n";
    out << "  " << ak64 << " = llvm.zext " << ak << " : i32 to i64\n";
    out << "  " << ak_m2 << " = llvm.sub " << ak64 << ", llvm.mlir.constant(2 : i64) : i64\n";
    out << "  " << iters << " = llvm.shl %c1_i64, " << ak_m2 << " : i64\n";
    char abuf[32], bbuf[32];
    snprintf(abuf, sizeof(abuf), "%u", a1);
    snprintf(bbuf, sizeof(bbuf), "%u", a2);
    std::string a64 = fresh_ssa(), b64 = fresh_ssa(), a_bit = fresh_ssa(), b_bit = fresh_ssa();
    out << "  " << a64 << " = llvm.mlir.constant(" << abuf << " : i64) : i64\n";
    out << "  " << b64 << " = llvm.mlir.constant(" << bbuf << " : i64) : i64\n";
    out << "  " << a_bit << " = llvm.shl %c1_i64, " << a64 << " : i64\n";
    out << "  " << b_bit << " = llvm.shl %c1_i64, " << b64 << " : i64\n";

    std::string hdr = fresh_label("sw_hdr"), body = fresh_label("sw_body"), exit_lbl = fresh_label("sw_exit");
    out << "  llvm.br ^" << hdr << "(%c0_i64 : i64)\n";
    out << "^" << hdr << "(%sw_i: i64):\n";
    std::string cond = fresh_ssa();
    out << "  " << cond << " = llvm.icmp \"ult\" %sw_i, " << iters << " : i64\n";
    out << "  llvm.cond_br " << cond << ", ^" << body << ", ^" << exit_lbl << "\n";
    out << "^" << body << ":\n";
    std::string base = emit_scatter_bits_2("%sw_i", a64, b64);
    std::string idx_a = fresh_ssa(), idx_b = fresh_ssa();
    out << "  " << idx_a << " = llvm.or " << base << ", " << a_bit << " : i64\n";
    out << "  " << idx_b << " = llvm.or " << base << ", " << b_bit << " : i64\n";
    std::string va = emit_load_v(idx_a);
    std::string vb = emit_load_v(idx_b);
    emit_store_v(idx_a, vb);
    emit_store_v(idx_b, va);
    std::string i_next = fresh_ssa();
    out << "  " << i_next << " = llvm.add %sw_i, %c1_i64 : i64\n";
    out << "  llvm.br ^" << hdr << "(" << i_next << " : i64)\n";
    out << "^" << exit_lbl << ":\n";
    // Frame: swap px bits, swap pz bits
    {
        std::string ai32 = fresh_ssa(), bi32 = fresh_ssa();
        out << "  " << ai32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
        out << "  " << bi32 << " = llvm.mlir.constant(" << bbuf << " : i32) : i32\n";
        std::string px_a = emit_bit_get("%px_ptr", ai32);
        std::string px_b = emit_bit_get("%px_ptr", bi32);
        emit_bit_set("%px_ptr", ai32, px_b);
        emit_bit_set("%px_ptr", bi32, px_a);
        std::string pz_a = emit_bit_get("%pz_ptr", ai32);
        std::string pz_b = emit_bit_get("%pz_ptr", bi32);
        emit_bit_set("%pz_ptr", ai32, pz_b);
        emit_bit_set("%pz_ptr", bi32, pz_a);
    }
    break;
}
case Opcode::OP_ARRAY_ROT: {
    out << "  // array_rot axis=" << ins.axis_1 << "\n";
    uint32_t axis = ins.axis_1;
    char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", axis);
    std::string ax_i32 = fresh_ssa();
    out << "  " << ax_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
    // Check px bit: if set, use conjugate phase
    std::string pxv = emit_bit_get("%px_ptr", ax_i32);
    // Prepare phase constants
    char wre_buf[64], wim_buf[64], wim_neg_buf[64];
    snprintf(wre_buf, sizeof(wre_buf), "%.17e", (float)ins.weight_re);
    snprintf(wim_buf, sizeof(wim_buf), "%.17e", (float)ins.weight_im);
    snprintf(wim_neg_buf, sizeof(wim_neg_buf), "%.17e", (float)-ins.weight_im);
    std::string phs_re = fresh_ssa(), phs_im_pos = fresh_ssa(), phs_im_neg = fresh_ssa(), phs_im = fresh_ssa();
    out << "  " << phs_re << " = llvm.mlir.constant(" << wre_buf << " : f32) : f32\n";
    out << "  " << phs_im_pos << " = llvm.mlir.constant(" << wim_buf << " : f32) : f32\n";
    out << "  " << phs_im_neg << " = llvm.mlir.constant(" << wim_neg_buf << " : f32) : f32\n";
    out << "  " << phs_im << " = llvm.select " << pxv << ", " << phs_im_neg << ", " << phs_im_pos << " : i1, f32\n";

    std::string ak = fresh_ssa(), ak64 = fresh_ssa(), ak_m1 = fresh_ssa(), iters = fresh_ssa();
    out << "  " << ak << " = llvm.load %active_k_ptr : !llvm.ptr -> i32\n";
    out << "  " << ak64 << " = llvm.zext " << ak << " : i32 to i64\n";
    out << "  " << ak_m1 << " = llvm.sub " << ak64 << ", %c1_i64 : i64\n";
    out << "  " << iters << " = llvm.shl %c1_i64, " << ak_m1 << " : i64\n";
    std::string axis64 = fresh_ssa(), axis_bit = fresh_ssa();
    out << "  " << axis64 << " = llvm.mlir.constant(" << abuf << " : i64) : i64\n";
    out << "  " << axis_bit << " = llvm.shl %c1_i64, " << axis64 << " : i64\n";

    std::string hdr = fresh_label("rot_hdr"), body_lbl = fresh_label("rot_body"), exit_lbl = fresh_label("rot_exit");
    out << "  llvm.br ^" << hdr << "(%c0_i64 : i64)\n";
    out << "^" << hdr << "(%rot_i: i64):\n";
    std::string cond = fresh_ssa();
    out << "  " << cond << " = llvm.icmp \"ult\" %rot_i, " << iters << " : i64\n";
    out << "  llvm.cond_br " << cond << ", ^" << body_lbl << ", ^" << exit_lbl << "\n";
    out << "^" << body_lbl << ":\n";
    std::string idx = fresh_ssa();
    out << "  " << idx << " = llvm.or " << emit_scatter_bits_1("%rot_i", axis64) << ", " << axis_bit << " : i64\n";
    std::string vc = emit_load_v(idx);
    // cmul: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    std::string vc_re = fresh_ssa(), vc_im = fresh_ssa();
    out << "  " << vc_re << " = llvm.extractvalue " << vc << "[0] : !llvm.struct<(f32, f32)>\n";
    out << "  " << vc_im << " = llvm.extractvalue " << vc << "[1] : !llvm.struct<(f32, f32)>\n";
    std::string t0 = fresh_ssa(), t1 = fresh_ssa(), re_out = fresh_ssa();
    out << "  " << t0 << " = llvm.fmul " << vc_re << ", " << phs_re << " : f32\n";
    out << "  " << t1 << " = llvm.fmul " << vc_im << ", " << phs_im << " : f32\n";
    out << "  " << re_out << " = llvm.fsub " << t0 << ", " << t1 << " : f32\n";
    std::string t2 = fresh_ssa(), t3 = fresh_ssa(), im_out = fresh_ssa();
    out << "  " << t2 << " = llvm.fmul " << vc_re << ", " << phs_im << " : f32\n";
    out << "  " << t3 << " = llvm.fmul " << vc_im << ", " << phs_re << " : f32\n";
    out << "  " << im_out << " = llvm.fadd " << t2 << ", " << t3 << " : f32\n";
    std::string r0 = fresh_ssa(), r1 = fresh_ssa(), res = fresh_ssa();
    out << "  " << r0 << " = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
    out << "  " << r1 << " = llvm.insertvalue " << re_out << ", " << r0 << "[0] : !llvm.struct<(f32, f32)>\n";
    out << "  " << res << " = llvm.insertvalue " << im_out << ", " << r1 << "[1] : !llvm.struct<(f32, f32)>\n";
    emit_store_v(idx, res);
    std::string i_next = fresh_ssa();
    out << "  " << i_next << " = llvm.add %rot_i, %c1_i64 : i64\n";
    out << "  llvm.br ^" << hdr << "(" << i_next << " : i64)\n";
    out << "^" << exit_lbl << ":\n";
    break;
}
case Opcode::OP_ARRAY_MULTI_CNOT: {
    out << "  // array_multi_cnot axis=" << ins.axis_1 << " mask=0x"
        << std::hex << ins.mask << std::dec << "\n";
    uint32_t axis = ins.axis_1;
    uint64_t mask_val = ins.mask;
    char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", axis);
    std::string ak = fresh_ssa(), ak64 = fresh_ssa(), ak_m1 = fresh_ssa(), iters = fresh_ssa();
    out << "  " << ak << " = llvm.load %active_k_ptr : !llvm.ptr -> i32\n";
    out << "  " << ak64 << " = llvm.zext " << ak << " : i32 to i64\n";
    out << "  " << ak_m1 << " = llvm.sub " << ak64 << ", %c1_i64 : i64\n";
    out << "  " << iters << " = llvm.shl %c1_i64, " << ak_m1 << " : i64\n";
    std::string axis64 = fresh_ssa(), axis_bit = fresh_ssa();
    out << "  " << axis64 << " = llvm.mlir.constant(" << abuf << " : i64) : i64\n";
    out << "  " << axis_bit << " = llvm.shl %c1_i64, " << axis64 << " : i64\n";
    char mbuf[32]; snprintf(mbuf, sizeof(mbuf), "%llu", (unsigned long long)mask_val);
    std::string mask_c = fresh_ssa();
    out << "  " << mask_c << " = llvm.mlir.constant(" << mbuf << " : i64) : i64\n";

    std::string hdr = fresh_label("mcn_hdr"), body_lbl = fresh_label("mcn_body"), exit_lbl = fresh_label("mcn_exit");
    out << "  llvm.br ^" << hdr << "(%c0_i64 : i64)\n";
    out << "^" << hdr << "(%mcn_i: i64):\n";
    std::string cond = fresh_ssa();
    out << "  " << cond << " = llvm.icmp \"ult\" %mcn_i, " << iters << " : i64\n";
    out << "  llvm.cond_br " << cond << ", ^" << body_lbl << ", ^" << exit_lbl << "\n";
    out << "^" << body_lbl << ":\n";
    std::string scat = emit_scatter_bits_1("%mcn_i", axis64);
    std::string base = fresh_ssa();
    out << "  " << base << " = llvm.or " << scat << ", " << axis_bit << " : i64\n";
    // popcount(base & mask)
    std::string masked = fresh_ssa();
    out << "  " << masked << " = llvm.and " << base << ", " << mask_c << " : i64\n";
    std::string pc = fresh_ssa();
    out << "  " << pc << " = llvm.intr.ctpop(" << masked << ") : (i64) -> i64\n";
    std::string pc_odd = fresh_ssa();
    out << "  " << pc_odd << " = llvm.and " << pc << ", %c1_i64 : i64\n";
    std::string is_odd = fresh_ssa();
    out << "  " << is_odd << " = llvm.icmp \"ne\" " << pc_odd << ", %c0_i64 : i64\n";
    // If odd popcount, negate v[base]
    std::string then_lbl = fresh_label("mcn_then"), merge_lbl = fresh_label("mcn_merge");
    out << "  llvm.cond_br " << is_odd << ", ^" << then_lbl << ", ^" << merge_lbl << "\n";
    out << "^" << then_lbl << ":\n";
    std::string vc = emit_load_v(base);
    std::string vc_re = fresh_ssa(), vc_im = fresh_ssa();
    out << "  " << vc_re << " = llvm.extractvalue " << vc << "[0] : !llvm.struct<(f32, f32)>\n";
    out << "  " << vc_im << " = llvm.extractvalue " << vc << "[1] : !llvm.struct<(f32, f32)>\n";
    std::string neg_re = fresh_ssa(), neg_im = fresh_ssa();
    out << "  " << neg_re << " = llvm.fneg " << vc_re << " : f32\n";
    out << "  " << neg_im << " = llvm.fneg " << vc_im << " : f32\n";
    std::string r0 = fresh_ssa(), r1 = fresh_ssa(), res = fresh_ssa();
    out << "  " << r0 << " = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
    out << "  " << r1 << " = llvm.insertvalue " << neg_re << ", " << r0 << "[0] : !llvm.struct<(f32, f32)>\n";
    out << "  " << res << " = llvm.insertvalue " << neg_im << ", " << r1 << "[1] : !llvm.struct<(f32, f32)>\n";
    emit_store_v(base, res);
    out << "  llvm.br ^" << merge_lbl << "\n";
    out << "^" << merge_lbl << ":\n";
    std::string i_next = fresh_ssa();
    out << "  " << i_next << " = llvm.add %mcn_i, %c1_i64 : i64\n";
    out << "  llvm.br ^" << hdr << "(" << i_next << " : i64)\n";
    out << "^" << exit_lbl << ":\n";
    // Frame update: for each bit j in mask, pz[j] ^= px[axis_1]
    // Then px[axis_1] ^= XOR of pz bits at mask positions
    {
        std::string ax_i32 = fresh_ssa();
        out << "  " << ax_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
        std::string px_ax = emit_bit_get("%px_ptr", ax_i32);
        for (uint32_t j = 0; j < 64; ++j) {
            if (mask_val & (1ULL << j)) {
                char jbuf[32]; snprintf(jbuf, sizeof(jbuf), "%u", j);
                std::string j_i32 = fresh_ssa();
                out << "  " << j_i32 << " = llvm.mlir.constant(" << jbuf << " : i32) : i32\n";
                emit_bit_xor("%pz_ptr", j_i32, px_ax);
            }
        }
        // px[axis_1] ^= XOR of pz[j] for each j in mask
        // Accumulate XOR
        std::string accum = "%c0_i1_mcn";
        // Create i1 zero constant
        std::string zero_i1 = fresh_ssa();
        out << "  " << zero_i1 << " = llvm.mlir.constant(false) : i1\n";
        accum = zero_i1;
        for (uint32_t j = 0; j < 64; ++j) {
            if (mask_val & (1ULL << j)) {
                char jbuf[32]; snprintf(jbuf, sizeof(jbuf), "%u", j);
                std::string j_i32 = fresh_ssa();
                out << "  " << j_i32 << " = llvm.mlir.constant(" << jbuf << " : i32) : i32\n";
                std::string pz_j = emit_bit_get("%pz_ptr", j_i32);
                std::string new_accum = fresh_ssa();
                out << "  " << new_accum << " = llvm.xor " << accum << ", " << pz_j << " : i1\n";
                accum = new_accum;
            }
        }
        emit_bit_xor("%px_ptr", ax_i32, accum);
    }
    break;
}
case Opcode::OP_ARRAY_MULTI_CZ: {
    out << "  // array_multi_cz axis=" << ins.axis_1 << " mask=0x"
        << std::hex << ins.mask << std::dec << "\n";
    uint32_t axis = ins.axis_1;
    uint64_t mask_val = ins.mask;
    char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", axis);
    std::string ak = fresh_ssa(), ak64 = fresh_ssa(), ak_m1 = fresh_ssa(), iters = fresh_ssa();
    out << "  " << ak << " = llvm.load %active_k_ptr : !llvm.ptr -> i32\n";
    out << "  " << ak64 << " = llvm.zext " << ak << " : i32 to i64\n";
    out << "  " << ak_m1 << " = llvm.sub " << ak64 << ", %c1_i64 : i64\n";
    out << "  " << iters << " = llvm.shl %c1_i64, " << ak_m1 << " : i64\n";
    std::string axis64 = fresh_ssa(), axis_bit = fresh_ssa();
    out << "  " << axis64 << " = llvm.mlir.constant(" << abuf << " : i64) : i64\n";
    out << "  " << axis_bit << " = llvm.shl %c1_i64, " << axis64 << " : i64\n";
    char mbuf[32]; snprintf(mbuf, sizeof(mbuf), "%llu", (unsigned long long)mask_val);
    std::string mask_c = fresh_ssa();
    out << "  " << mask_c << " = llvm.mlir.constant(" << mbuf << " : i64) : i64\n";

    std::string hdr = fresh_label("mcz_hdr"), body_lbl = fresh_label("mcz_body"), exit_lbl = fresh_label("mcz_exit");
    out << "  llvm.br ^" << hdr << "(%c0_i64 : i64)\n";
    out << "^" << hdr << "(%mcz_i: i64):\n";
    std::string cond = fresh_ssa();
    out << "  " << cond << " = llvm.icmp \"ult\" %mcz_i, " << iters << " : i64\n";
    out << "  llvm.cond_br " << cond << ", ^" << body_lbl << ", ^" << exit_lbl << "\n";
    out << "^" << body_lbl << ":\n";
    std::string scat = emit_scatter_bits_1("%mcz_i", axis64);
    std::string base = fresh_ssa();
    out << "  " << base << " = llvm.or " << scat << ", " << axis_bit << " : i64\n";
    // popcount(base & mask)
    std::string masked = fresh_ssa();
    out << "  " << masked << " = llvm.and " << base << ", " << mask_c << " : i64\n";
    std::string pc = fresh_ssa();
    out << "  " << pc << " = llvm.intr.ctpop(" << masked << ") : (i64) -> i64\n";
    std::string pc_odd = fresh_ssa();
    out << "  " << pc_odd << " = llvm.and " << pc << ", %c1_i64 : i64\n";
    std::string is_odd = fresh_ssa();
    out << "  " << is_odd << " = llvm.icmp \"ne\" " << pc_odd << ", %c0_i64 : i64\n";
    std::string then_lbl = fresh_label("mcz_then"), merge_lbl = fresh_label("mcz_merge");
    out << "  llvm.cond_br " << is_odd << ", ^" << then_lbl << ", ^" << merge_lbl << "\n";
    out << "^" << then_lbl << ":\n";
    std::string vc = emit_load_v(base);
    std::string vc_re = fresh_ssa(), vc_im = fresh_ssa();
    out << "  " << vc_re << " = llvm.extractvalue " << vc << "[0] : !llvm.struct<(f32, f32)>\n";
    out << "  " << vc_im << " = llvm.extractvalue " << vc << "[1] : !llvm.struct<(f32, f32)>\n";
    std::string neg_re = fresh_ssa(), neg_im = fresh_ssa();
    out << "  " << neg_re << " = llvm.fneg " << vc_re << " : f32\n";
    out << "  " << neg_im << " = llvm.fneg " << vc_im << " : f32\n";
    std::string r0 = fresh_ssa(), r1 = fresh_ssa(), res = fresh_ssa();
    out << "  " << r0 << " = llvm.mlir.undef : !llvm.struct<(f32, f32)>\n";
    out << "  " << r1 << " = llvm.insertvalue " << neg_re << ", " << r0 << "[0] : !llvm.struct<(f32, f32)>\n";
    out << "  " << res << " = llvm.insertvalue " << neg_im << ", " << r1 << "[1] : !llvm.struct<(f32, f32)>\n";
    emit_store_v(base, res);
    out << "  llvm.br ^" << merge_lbl << "\n";
    out << "^" << merge_lbl << ":\n";
    std::string i_next = fresh_ssa();
    out << "  " << i_next << " = llvm.add %mcz_i, %c1_i64 : i64\n";
    out << "  llvm.br ^" << hdr << "(" << i_next << " : i64)\n";
    out << "^" << exit_lbl << ":\n";
    // Frame update: for each bit j in mask, pz[j] ^= px[axis_1]; pz[axis_1] ^= px[j]
    {
        std::string ax_i32 = fresh_ssa();
        out << "  " << ax_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
        std::string px_ax = emit_bit_get("%px_ptr", ax_i32);
        for (uint32_t j = 0; j < 64; ++j) {
            if (mask_val & (1ULL << j)) {
                char jbuf[32]; snprintf(jbuf, sizeof(jbuf), "%u", j);
                std::string j_i32 = fresh_ssa();
                out << "  " << j_i32 << " = llvm.mlir.constant(" << jbuf << " : i32) : i32\n";
                std::string px_j = emit_bit_get("%px_ptr", j_i32);
                emit_bit_xor("%pz_ptr", j_i32, px_ax);
                emit_bit_xor("%pz_ptr", ax_i32, px_j);
            }
        }
    }
    break;
}
case Opcode::OP_ARRAY_U2:
case Opcode::OP_ARRAY_U4: {
    out << "  // U2/U4 requires constant pool — not available in MLIR path, discard\n";
    out << "  llvm.store %c1_i8, %discarded_ptr : i8, !llvm.ptr\n";
    break;
}
