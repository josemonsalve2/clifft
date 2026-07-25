// rank_probe.cpp — compile .stim files through the full clifft frontend+backend
// pass pipeline (identical to run_v2's compile_program) and print the resulting
// peak_rank. Host-only, so it runs on the login node where there is no GPU.
// Used to verify what rank a generated fixture ACTUALLY compiles to after the
// StatevectorSqueezePass, which routinely squeezes nominally-high-rank circuits.
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/backend/backend.h"
#include "clifft/api/reference_syndrome.h"
#include "clifft/optimizer/pass_factory.h"

#include <iostream>

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        try {
            auto circuit = clifft::parse_file(argv[i]);
            auto hir = clifft::trace(circuit);
            auto hpm = clifft::default_hir_pass_manager();
            hpm.run(hir);
            auto ref = clifft::compute_reference_syndrome(hir);
            auto program = clifft::lower(hir, {}, ref.detectors, ref.observables);
            auto bpm = clifft::default_bytecode_pass_manager();
            bpm.run(program);
            std::cout << argv[i]
                      << "  peak_rank=" << program.peak_rank
                      << "  qubits=" << program.num_qubits
                      << "  instrs=" << program.bytecode.size()
                      << "  meas=" << program.total_meas_slots
                      << "  obs=" << program.num_observables << "\n";
        } catch (const std::exception& e) {
            std::cout << argv[i] << "  ERROR: " << e.what() << "\n";
        }
    }
    return 0;
}
