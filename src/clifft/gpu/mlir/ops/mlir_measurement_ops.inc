// mlir_measurement_ops.inc — MLIR emission for measurement and classical ops
//
// Included inside the instruction switch in mlir_emit.cc.
// Available in scope: out, fresh_ssa, emit_bit_get, ins, sign, identity, flat

case Opcode::OP_MEAS_DORMANT_STATIC: {
    char abuf[32]; snprintf(abuf, sizeof(abuf), "%u", ins.a);
    std::string midx = fresh_ssa();
    out << "  " << midx << " = llvm.mlir.constant(" << abuf << " : i64) : i64\n";
    std::string mptr = fresh_ssa();
    out << "  " << mptr << " = llvm.getelementptr inbounds %meas_ptr["
        << midx << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
    std::string mval;
    if (identity) {
        mval = fresh_ssa();
        out << "  " << mval << " = llvm.mlir.constant("
            << (sign ? 1 : 0) << " : i8) : i8\n";
    } else {
        char axbuf[32]; snprintf(axbuf, sizeof(axbuf), "%u", ins.axis_1);
        std::string ax_i32 = fresh_ssa();
        out << "  " << ax_i32 << " = llvm.mlir.constant(" << axbuf << " : i32) : i32\n";
        std::string px_bit = emit_bit_get("%px_ptr", ax_i32);
        std::string px_i8 = fresh_ssa();
        out << "  " << px_i8 << " = llvm.zext " << px_bit << " : i1 to i8\n";
        if (sign) {
            mval = fresh_ssa();
            out << "  " << mval << " = llvm.xor " << px_i8 << ", %c1_i8 : i8\n";
        } else {
            mval = px_i8;
        }
    }
    out << "  llvm.store " << mval << ", " << mptr << " : i8, !llvm.ptr\n";
    break;
}
case Opcode::OP_OBSERVABLE: {
    if (ins.a + 1 < flat.observable_offsets.size()) {
        uint32_t ob_start = flat.observable_offsets[ins.a];
        uint32_t ob_end = flat.observable_offsets[ins.a + 1];
        char bidx_buf[32]; snprintf(bidx_buf, sizeof(bidx_buf), "%u", ins.b);
        std::string obs_idx = fresh_ssa();
        out << "  " << obs_idx << " = llvm.mlir.constant(" << bidx_buf << " : i64) : i64\n";
        std::string obs_elem_ptr = fresh_ssa();
        out << "  " << obs_elem_ptr << " = llvm.getelementptr inbounds %obs_ptr["
            << obs_idx << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        std::string parity = "%c0_i8";
        for (uint32_t k = ob_start; k < ob_end; ++k) {
            char tidx_buf[32]; snprintf(tidx_buf, sizeof(tidx_buf), "%u", flat.observable_targets[k]);
            std::string tidx = fresh_ssa();
            out << "  " << tidx << " = llvm.mlir.constant(" << tidx_buf << " : i64) : i64\n";
            std::string tptr = fresh_ssa();
            out << "  " << tptr << " = llvm.getelementptr inbounds %meas_ptr["
                << tidx << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
            std::string tval = fresh_ssa();
            out << "  " << tval << " = llvm.load " << tptr << " : !llvm.ptr -> i8\n";
            std::string new_parity = fresh_ssa();
            out << "  " << new_parity << " = llvm.xor " << parity << ", " << tval << " : i8\n";
            parity = new_parity;
        }
        std::string cur_obs = fresh_ssa();
        out << "  " << cur_obs << " = llvm.load " << obs_elem_ptr << " : !llvm.ptr -> i8\n";
        std::string new_obs = fresh_ssa();
        out << "  " << new_obs << " = llvm.xor " << cur_obs << ", " << parity << " : i8\n";
        out << "  llvm.store " << new_obs << ", " << obs_elem_ptr << " : i8, !llvm.ptr\n";
    }
    break;
}
case Opcode::OP_DETECTOR:
    break;
case Opcode::OP_MEAS_DORMANT_RANDOM: {
    emit_meas_dormant_random(out, ins.axis_1, ins.a, sign);
    break;
}
case Opcode::OP_MEAS_ACTIVE_DIAGONAL: {
    emit_meas_active_diagonal(out, ins.axis_1, ins.a, sign);
    break;
}
case Opcode::OP_MEAS_ACTIVE_INTERFERE: {
    emit_meas_active_interfere(out, ins.axis_1, ins.a, sign);
    break;
}
case Opcode::OP_SWAP_MEAS_INTERFERE: {
    out << "  // swap_meas_interfere — discard (needs frame_swap + interfere)\n";
    out << "  llvm.store %c1_i8, %discarded_ptr : i8, !llvm.ptr\n";
    break;
}
case Opcode::OP_MEAS_DORMANT_STATIC_FORCED:
case Opcode::OP_MEAS_DORMANT_RANDOM_FORCED:
case Opcode::OP_MEAS_ACTIVE_DIAGONAL_FORCED:
case Opcode::OP_MEAS_ACTIVE_INTERFERE_FORCED:
case Opcode::OP_SWAP_MEAS_INTERFERE_FORCED: {
    out << "  // forced measurement op " << (unsigned)ins.opcode
        << " — discard\n";
    out << "  llvm.store %c1_i8, %discarded_ptr : i8, !llvm.ptr\n";
    break;
}
