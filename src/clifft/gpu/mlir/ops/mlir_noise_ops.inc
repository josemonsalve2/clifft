// mlir_noise_ops.inc — MLIR emission for noise/error operations
//
// Included inside the instruction switch in mlir_emit.cc.
// All noise ops require RNG state and constant pool data which are
// NOT available in the MLIR path yet. Emit discard markers for now.
// Circuits without noise will work; noisy circuits are correctly rejected.

case Opcode::OP_NOISE:
case Opcode::OP_NOISE_BLOCK:
case Opcode::OP_READOUT_NOISE: {
    out << "  // noise op " << (unsigned)ins.opcode
        << " requires RNG/constant pool — discard\n";
    out << "  llvm.store %c1_i8, %discarded_ptr : i8, !llvm.ptr\n";
    break;
}
case Opcode::OP_POSTSELECT: {
    bool expected_one = (ins.flags & kFlagExpectedOne) != 0;
    if (ins.a + 1 < flat.detector_offsets.size()) {
        uint32_t ps_start = flat.detector_offsets[ins.a];
        uint32_t ps_end = flat.detector_offsets[ins.a + 1];
        std::string parity = fresh_ssa();
        out << "  " << parity << " = llvm.mlir.constant("
            << (expected_one ? 1 : 0) << " : i8) : i8\n";
        for (uint32_t k = ps_start; k < ps_end; ++k) {
            char tidx_buf[32]; snprintf(tidx_buf, sizeof(tidx_buf), "%u", flat.detector_targets[k]);
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
        std::string ps_fail = fresh_ssa();
        out << "  " << ps_fail << " = llvm.icmp \"ne\" " << parity << ", %c0_i8 : i8\n";
        std::string lbl_ps_disc = fresh_label("ps_disc");
        std::string lbl_ps_ok = fresh_label("ps_ok");
        out << "  llvm.cond_br " << ps_fail << ", ^" << lbl_ps_disc << ", ^" << lbl_ps_ok << "\n";
        out << "^" << lbl_ps_disc << ":\n";
        out << "  llvm.store %c1_i8, %discarded_ptr : i8, !llvm.ptr\n";
        out << "  llvm.br ^" << lbl_ps_ok << "\n";
        out << "^" << lbl_ps_ok << ":\n";
    }
    break;
}
case Opcode::OP_APPLY_PAULI: {
    out << "  // apply_pauli requires constant pool — discard\n";
    out << "  llvm.store %c1_i8, %discarded_ptr : i8, !llvm.ptr\n";
    break;
}
