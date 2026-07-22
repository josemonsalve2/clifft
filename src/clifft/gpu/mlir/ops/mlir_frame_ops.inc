// mlir_frame_ops.inc — MLIR emission for Pauli frame operations
//
// Included inside the instruction switch in mlir_emit.cc.
// In cooperative mode, frame ops run on thread-0 only with barrier after.

case Opcode::OP_FRAME_H: {
    auto g = emit_tid0_guard_begin(out);
    uint32_t ax = ins.axis_1;
    char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", ax);
    std::string ax_i32 = fresh_ssa();
    out << "  " << ax_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
    std::string pxv = emit_bit_get("%px_ptr", ax_i32);
    std::string pzv = emit_bit_get("%pz_ptr", ax_i32);
    emit_bit_set("%px_ptr", ax_i32, pzv);
    emit_bit_set("%pz_ptr", ax_i32, pxv);
    emit_tid0_guard_end(out, g);
    break;
}
case Opcode::OP_FRAME_S:
case Opcode::OP_FRAME_S_DAG: {
    auto g = emit_tid0_guard_begin(out);
    uint32_t ax = ins.axis_1;
    char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", ax);
    std::string ax_i32 = fresh_ssa();
    out << "  " << ax_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
    std::string pxv = emit_bit_get("%px_ptr", ax_i32);
    emit_bit_xor("%pz_ptr", ax_i32, pxv);
    emit_tid0_guard_end(out, g);
    break;
}
case Opcode::OP_FRAME_CNOT: {
    auto g = emit_tid0_guard_begin(out);
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
    emit_tid0_guard_end(out, g);
    break;
}
case Opcode::OP_FRAME_CZ: {
    auto g = emit_tid0_guard_begin(out);
    char abuf[32], bbuf[32];
    snprintf(abuf, sizeof(abuf), "%u", ins.axis_1);
    snprintf(bbuf, sizeof(bbuf), "%u", ins.axis_2);
    std::string a_i32 = fresh_ssa(), b_i32 = fresh_ssa();
    out << "  " << a_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
    out << "  " << b_i32 << " = llvm.mlir.constant(" << bbuf << " : i32) : i32\n";
    std::string px_a = emit_bit_get("%px_ptr", a_i32);
    std::string px_b = emit_bit_get("%px_ptr", b_i32);
    emit_bit_xor("%pz_ptr", b_i32, px_a);
    emit_bit_xor("%pz_ptr", a_i32, px_b);
    emit_tid0_guard_end(out, g);
    break;
}
case Opcode::OP_FRAME_SWAP: {
    auto g = emit_tid0_guard_begin(out);
    char abuf[32], bbuf[32];
    snprintf(abuf, sizeof(abuf), "%u", ins.axis_1);
    snprintf(bbuf, sizeof(bbuf), "%u", ins.axis_2);
    std::string a_i32 = fresh_ssa(), b_i32 = fresh_ssa();
    out << "  " << a_i32 << " = llvm.mlir.constant(" << abuf << " : i32) : i32\n";
    out << "  " << b_i32 << " = llvm.mlir.constant(" << bbuf << " : i32) : i32\n";
    std::string px_a = emit_bit_get("%px_ptr", a_i32);
    std::string px_b = emit_bit_get("%px_ptr", b_i32);
    emit_bit_set("%px_ptr", a_i32, px_b);
    emit_bit_set("%px_ptr", b_i32, px_a);
    std::string pz_a = emit_bit_get("%pz_ptr", a_i32);
    std::string pz_b = emit_bit_get("%pz_ptr", b_i32);
    emit_bit_set("%pz_ptr", a_i32, pz_b);
    emit_bit_set("%pz_ptr", b_i32, pz_a);
    emit_tid0_guard_end(out, g);
    break;
}
