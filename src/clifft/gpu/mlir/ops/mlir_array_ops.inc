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
