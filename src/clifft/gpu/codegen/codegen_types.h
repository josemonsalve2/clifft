#pragma once

#include "clifft/gpu/codegen/kernel_codegen.h"
#include "clifft/gpu/gpu_types.h"

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace clifft {
namespace gpu {
namespace codegen {

std::string hex_double(double v);
std::string hex_float_pair(float re, float im);

void emit_preamble(std::ostringstream& out, const UsedFunctions& uf);
void emit_needed_functions(std::ostringstream& out, const UsedFunctions& uf);
void emit_constant_pool(std::ostringstream& out, const FlattenedProgram& flat,
                        const UsedFunctions& uf);
void emit_instructions(std::ostringstream& out, const FlattenedProgram& flat);
void emit_coop_instructions(std::ostringstream& out, const FlattenedProgram& flat,
                            const std::vector<PipelineOp>& pipe_ops);

}  // namespace codegen
}  // namespace gpu
}  // namespace clifft
