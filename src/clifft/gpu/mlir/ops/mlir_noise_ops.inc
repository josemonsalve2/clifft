// mlir_noise_ops.inc — MLIR emission for noise/error operations
//
// Included inside the instruction switch in mlir_emit.cc.
// All noise ops require RNG state and constant pool data which are
// NOT available in the MLIR path yet. Emit discard markers for now.
// Circuits without noise will work; noisy circuits are correctly rejected.

case Opcode::OP_NOISE:
case Opcode::OP_NOISE_BLOCK:
case Opcode::OP_READOUT_NOISE:
case Opcode::OP_POSTSELECT: {
    out << "  // noise/postselect op " << (unsigned)ins.opcode
        << " requires RNG/constant pool — discard\n";
    out << "  llvm.store %c1_i8, %discarded_ptr : i8, !llvm.ptr\n";
    break;
}
case Opcode::OP_APPLY_PAULI: {
    out << "  // apply_pauli requires constant pool — discard\n";
    out << "  llvm.store %c1_i8, %discarded_ptr : i8, !llvm.ptr\n";
    break;
}
