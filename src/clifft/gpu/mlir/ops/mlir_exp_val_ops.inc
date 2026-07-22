// mlir_exp_val_ops.inc — MLIR emission for expectation value operations
//
// Included inside the instruction switch in mlir_emit.cc.
// OP_EXP_VAL requires constant pool (exp_val_masks) which is NOT
// available in the MLIR path yet. Emit discard marker for now.

case Opcode::OP_EXP_VAL: {
    out << "  // exp_val requires constant pool — discard\n";
    out << "  llvm.store %c1_i8, %discarded_ptr : i8, !llvm.ptr\n";
    break;
}
