#pragma once

#ifdef CLIFFT_ENABLE_MLIR

#include "clifft/gpu/device_program.h"

#include <sstream>
#include <string>

namespace clifft {
namespace gpu {
namespace mlir_emit {

std::string emit_mlir_text(const FlattenedProgram& flat);

std::string find_mlir_opt();
std::string find_mlir_translate();

int run_pipe_command(const std::string& cmd, const std::string& stdin_data,
                     std::string& stdout_out);

}  // namespace mlir_emit
}  // namespace gpu
}  // namespace clifft

#endif  // CLIFFT_ENABLE_MLIR
