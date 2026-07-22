// mlir_expand_ops.inc — MLIR emission for subspace expansion operations
//
// Included inside the instruction switch in mlir_emit.cc.

case Opcode::OP_EXPAND: {
    emit_expand_plain(out);
    break;
}
case Opcode::OP_EXPAND_T: {
    emit_expand_t(out, ins.axis_1, false);
    break;
}
case Opcode::OP_EXPAND_T_DAG: {
    emit_expand_t(out, ins.axis_1, true);
    break;
}
case Opcode::OP_EXPAND_ROT: {
    // expand_rot: expand + apply rotation phase on upper half
    // For now use expand_plain (rotation applied separately)
    emit_expand_plain(out);
    break;
}
