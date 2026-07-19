#include "clifft/gpu/device_program.h"

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>

namespace clifft {
namespace gpu {

namespace {

bool is_supported_opcode(Opcode op) {
    switch (op) {
        case Opcode::OP_FRAME_CNOT:
        case Opcode::OP_FRAME_CZ:
        case Opcode::OP_FRAME_H:
        case Opcode::OP_FRAME_S:
        case Opcode::OP_FRAME_S_DAG:
        case Opcode::OP_FRAME_SWAP:
        case Opcode::OP_ARRAY_CNOT:
        case Opcode::OP_ARRAY_CZ:
        case Opcode::OP_ARRAY_SWAP:
        case Opcode::OP_ARRAY_MULTI_CNOT:
        case Opcode::OP_ARRAY_MULTI_CZ:
        case Opcode::OP_ARRAY_H:
        case Opcode::OP_ARRAY_S:
        case Opcode::OP_ARRAY_S_DAG:
        case Opcode::OP_ARRAY_T:
        case Opcode::OP_ARRAY_T_DAG:
        case Opcode::OP_ARRAY_ROT:
        case Opcode::OP_ARRAY_U2:
        case Opcode::OP_ARRAY_U4:
        case Opcode::OP_EXPAND:
        case Opcode::OP_EXPAND_T:
        case Opcode::OP_EXPAND_T_DAG:
        case Opcode::OP_EXPAND_ROT:
        case Opcode::OP_MEAS_DORMANT_STATIC:
        case Opcode::OP_MEAS_DORMANT_RANDOM:
        case Opcode::OP_MEAS_ACTIVE_DIAGONAL:
        case Opcode::OP_MEAS_ACTIVE_INTERFERE:
        case Opcode::OP_SWAP_MEAS_INTERFERE:
        case Opcode::OP_APPLY_PAULI:
        case Opcode::OP_NOISE:
        case Opcode::OP_NOISE_BLOCK:
        case Opcode::OP_READOUT_NOISE:
        case Opcode::OP_DETECTOR:
        case Opcode::OP_POSTSELECT:
        case Opcode::OP_OBSERVABLE:
        case Opcode::OP_EXP_VAL:
            return true;
        default:
            return false;
    }
}

GpuInstr flatten_instr(const Instruction& src) {
    GpuInstr dst{};
    dst.opcode = static_cast<uint8_t>(src.opcode);
    dst.flags = src.flags;
    dst.axis_1 = src.axis_1;
    dst.axis_2 = src.axis_2;
    dst.weight_re = src.math.weight_re;
    dst.weight_im = src.math.weight_im;
    dst.mask = src.multi_gate.mask;

    switch (src.opcode) {
        case Opcode::OP_MEAS_DORMANT_STATIC:
        case Opcode::OP_MEAS_DORMANT_RANDOM:
        case Opcode::OP_MEAS_ACTIVE_DIAGONAL:
        case Opcode::OP_MEAS_ACTIVE_INTERFERE:
        case Opcode::OP_SWAP_MEAS_INTERFERE:
            dst.a = src.classical.classical_idx;
            dst.b = src.classical.expected_val;
            break;
        case Opcode::OP_APPLY_PAULI:
            dst.a = src.pauli.cp_mask_idx;
            dst.b = src.pauli.condition_idx;
            break;
        case Opcode::OP_NOISE:
        case Opcode::OP_READOUT_NOISE:
            dst.a = src.pauli.cp_mask_idx;
            dst.b = src.pauli.condition_idx;
            break;
        case Opcode::OP_NOISE_BLOCK:
            dst.a = src.pauli.cp_mask_idx;
            dst.b = src.pauli.condition_idx;
            break;
        case Opcode::OP_DETECTOR:
        case Opcode::OP_POSTSELECT:
            dst.a = src.pauli.cp_mask_idx;
            dst.b = src.pauli.condition_idx;
            break;
        case Opcode::OP_OBSERVABLE:
            dst.a = src.pauli.cp_mask_idx;
            dst.b = src.pauli.condition_idx;
            break;
        case Opcode::OP_ARRAY_U2:
            dst.a = src.u2.cp_idx;
            break;
        case Opcode::OP_ARRAY_U4:
            dst.a = src.u4.cp_idx;
            break;
        case Opcode::OP_EXP_VAL:
            dst.a = src.exp_val.cp_exp_val_idx;
            dst.b = src.exp_val.exp_val_idx;
            break;
        default:
            dst.a = src.pauli.cp_mask_idx;
            dst.b = src.pauli.condition_idx;
            break;
    }
    return dst;
}

GpuMask flatten_arena_mask(const PauliMaskArena& arena, PauliMaskHandle handle) {
    GpuMask dst{};
    auto view = arena.at(handle);
    auto x_view = view.x();
    auto z_view = view.z();
    for (uint32_t w = 0; w < kPauliWords && w < x_view.num_words(); ++w)
        dst.x[w] = x_view.words[w];
    for (uint32_t w = 0; w < kPauliWords && w < z_view.num_words(); ++w)
        dst.z[w] = z_view.words[w];
    dst.sign = view.sign() ? 1 : 0;
    return dst;
}

GpuComplex to_gpu_complex(std::complex<double> c) {
    return {static_cast<float>(c.real()), static_cast<float>(c.imag())};
}

void append_target_lists(const std::vector<std::vector<uint32_t>>& lists,
                         std::vector<uint32_t>& offsets, std::vector<uint32_t>& targets) {
    offsets.clear();
    targets.clear();
    offsets.reserve(lists.size() + 1);
    offsets.push_back(0);
    for (const auto& list : lists) {
        targets.insert(targets.end(), list.begin(), list.end());
        offsets.push_back(static_cast<uint32_t>(targets.size()));
    }
}

// ---------------------------------------------------------------------------
// K-profile segment analysis for the hybrid kernel.
// ---------------------------------------------------------------------------
//
// Walks the compile-time active_k history and groups bytecode instructions
// into segments at k-transition boundaries. Each segment gets a tier
// assignment (per-thread / shared-coop / global-coop) based on its local
// peak k value.

/// Extract raw segments from the active_k history.
/// Splits occur at transitions where k drops to 0 from k > 0.
std::vector<GpuSegment> extract_k_segments(
    const std::vector<uint32_t>& k_hist)
{
    std::vector<GpuSegment> segments;
    if (k_hist.empty()) return segments;

    uint32_t seg_start = 0;
    uint32_t local_peak = k_hist[0];

    for (size_t i = 1; i < k_hist.size(); ++i) {
        uint32_t k_prev = k_hist[i - 1];
        uint32_t k_cur = k_hist[i];

        local_peak = std::max(local_peak, k_cur);

        // Split at transitions from k > 0 to k = 0
        if (k_prev > 0 && k_cur == 0) {
            segments.push_back({seg_start, static_cast<uint32_t>(i),
                                local_peak, tier_for_peak_k(local_peak)});
            seg_start = static_cast<uint32_t>(i);
            local_peak = 0;
        }
    }
    // Close final segment
    segments.push_back({seg_start, static_cast<uint32_t>(k_hist.size()),
                        local_peak, tier_for_peak_k(local_peak)});
    return segments;
}

/// Merge consecutive segments that are too small to justify a split boundary.
/// Adjacent segments are merged if either is shorter than min_instructions,
/// as long as they have the same tier. If tiers differ, only merge the
/// smaller segment into its neighbor (adopting the higher tier).
std::vector<GpuSegment> merge_small_segments(
    const std::vector<GpuSegment>& segments, uint32_t min_instructions = 64)
{
    std::vector<GpuSegment> merged;
    if (segments.empty()) return merged;

    merged.push_back(segments[0]);
    for (size_t i = 1; i < segments.size(); ++i) {
        auto& prev = merged.back();
        const auto& cur = segments[i];
        uint32_t prev_len = prev.bc_end - prev.bc_start;
        uint32_t cur_len = cur.bc_end - cur.bc_start;

        // Merge if either is too small
        if (prev_len < min_instructions || cur_len < min_instructions) {
            prev.bc_end = cur.bc_end;
            prev.local_peak_k = std::max(prev.local_peak_k, cur.local_peak_k);
            prev.tier = tier_for_peak_k(prev.local_peak_k);
        } else {
            merged.push_back(cur);
        }
    }
    return merged;
}

/// Determine if the hybrid kernel is profitable for the given segments.
/// Returns true when segments span multiple tiers and enough instructions
/// benefit from tier downgrading to overcome dispatch overhead.
bool should_use_hybrid(const std::vector<GpuSegment>& segments) {
    if (segments.size() <= 1) return false;

    // Check that segments span multiple tiers
    std::set<uint8_t> tiers;
    for (const auto& s : segments) tiers.insert(s.tier);
    if (tiers.size() <= 1) return false;

    // Compute total instructions
    uint32_t total_instrs = 0;
    for (const auto& s : segments) total_instrs += s.bc_end - s.bc_start;
    if (total_instrs < 256) return false;

    // Compute the global tier (what the whole-program tier would be)
    uint32_t global_peak = 0;
    for (const auto& s : segments) global_peak = std::max(global_peak, s.local_peak_k);
    uint8_t global_tier = tier_for_peak_k(global_peak);

    // Count instructions that would be downgraded from global_tier
    uint32_t downgraded_instrs = 0;
    for (const auto& s : segments) {
        if (s.tier < global_tier) {
            downgraded_instrs += s.bc_end - s.bc_start;
        }
    }

    // Profitability: at least 20% of instructions benefit (conservative).
    // The persistent kernel has zero launch overhead between segments, so
    // even modest downgrade fractions are beneficial.
    return downgraded_instrs > total_instrs / 5;
}

/// Build the segment table for the hybrid kernel from a CompiledModule.
/// Populates the segments vector and use_hybrid_kernel flag in the
/// FlattenedProgram.
void build_segments(FlattenedProgram& flat, const CompiledModule& program) {
    const auto& k_hist = program.source_map.active_k_history();
    if (k_hist.empty() || k_hist.size() != program.bytecode.size()) {
        // No active_k history or size mismatch -- skip segment analysis.
        flat.use_hybrid_kernel = false;
        return;
    }

    auto raw_segments = extract_k_segments(k_hist);
    auto segments = merge_small_segments(raw_segments, 64);
    flat.use_hybrid_kernel = should_use_hybrid(segments);
    flat.segments = std::move(segments);
}

}  // namespace

void validate_program(const CompiledModule& program) {
    if (program.peak_rank > kMaxPeakRank) {
        std::ostringstream ss;
        ss << "GPU sampler supports peak_rank <= " << kMaxPeakRank
           << "; program peak_rank is " << program.peak_rank;
        throw std::runtime_error(ss.str());
    }
    if (program.total_meas_slots > kMaxMeas) {
        throw std::runtime_error("GPU sampler measurement record limit exceeded");
    }
    if (program.num_observables > kMaxObs) {
        throw std::runtime_error("GPU sampler observable limit exceeded");
    }
    if (program.num_exp_vals > kMaxExpVals) {
        throw std::runtime_error("GPU sampler expectation value limit exceeded");
    }
    if (program.num_qubits > kMaxQubits) {
        std::ostringstream ss;
        ss << "GPU sampler supports at most " << kMaxQubits << " qubits"
           << " (" << kPauliWords << " Pauli words); circuit has "
           << program.num_qubits;
        throw std::runtime_error(ss.str());
    }
    for (const auto& instr : program.bytecode) {
        if (!is_supported_opcode(instr.opcode)) {
            std::ostringstream ss;
            ss << "GPU sampler encountered unsupported opcode "
               << static_cast<int>(instr.opcode);
            throw std::runtime_error(ss.str());
        }
    }
}

FlattenedProgram flatten_program(const CompiledModule& program) {
    validate_program(program);

    FlattenedProgram flat;
    flat.peak_rank = program.peak_rank;
    flat.total_meas_slots = program.total_meas_slots;
    flat.num_observables = program.num_observables;
    flat.num_exp_vals = program.num_exp_vals;

    flat.instrs.reserve(program.bytecode.size());
    for (const auto& instr : program.bytecode) {
        flat.instrs.push_back(flatten_instr(instr));
        switch (instr.opcode) {
            case Opcode::OP_ARRAY_ROT:
            case Opcode::OP_ARRAY_U2:
            case Opcode::OP_ARRAY_U4:
            case Opcode::OP_EXPAND_ROT:
            case Opcode::OP_EXP_VAL:
                flat.has_extended_opcodes = true;
                break;
            default:
                break;
        }
    }

    const auto& pool = program.constant_pool;

    flat.pauli_masks.reserve(pool.pauli_masks.size());
    for (size_t i = 0; i < pool.pauli_masks.size(); ++i) {
        flat.pauli_masks.push_back(
            flatten_arena_mask(pool.pauli_masks, static_cast<PauliMaskHandle>(i)));
    }

    flat.noise_sites.reserve(pool.noise_sites.size());
    for (const auto& site : pool.noise_sites) {
        GpuNoiseSite flat_site{};
        flat_site.offset = static_cast<uint32_t>(flat.noise_channels.size());
        flat_site.count = static_cast<uint32_t>(site.channels.size());
        flat_site.prob_sum = 0.0;
        for (const auto& ch : site.channels) {
            GpuChannel flat_ch{};
            auto mask_view = pool.noise_channel_masks.at(ch.mask);
            auto x_view = mask_view.x();
            auto z_view = mask_view.z();
            for (uint32_t w = 0; w < kPauliWords && w < x_view.num_words(); ++w)
                flat_ch.x[w] = x_view.words[w];
            for (uint32_t w = 0; w < kPauliWords && w < z_view.num_words(); ++w)
                flat_ch.z[w] = z_view.words[w];
            flat_ch.prob = ch.prob;
            flat_site.prob_sum += ch.prob;
            flat.noise_channels.push_back(flat_ch);
        }
        flat.noise_sites.push_back(flat_site);
    }

    flat.noise_hazards = pool.noise_hazards;

    flat.readout_noise.reserve(pool.readout_noise.size());
    for (const auto& entry : pool.readout_noise) {
        flat.readout_noise.push_back({entry.meas_idx, entry.prob});
    }

    append_target_lists(pool.detector_targets, flat.detector_offsets, flat.detector_targets);
    append_target_lists(pool.observable_targets, flat.observable_offsets, flat.observable_targets);

    flat.expected_observables.resize(program.num_observables, 0);
    for (size_t i = 0; i < flat.expected_observables.size() &&
                        i < program.expected_observables.size();
         ++i) {
        flat.expected_observables[i] = program.expected_observables[i];
    }

    flat.fused_u2.reserve(pool.fused_u2_nodes.size());
    for (const auto& node : pool.fused_u2_nodes) {
        GpuFusedU2Entry entry{};
        for (int s = 0; s < 4; ++s) {
            for (int e = 0; e < 4; ++e) {
                entry.matrices[s][e] = to_gpu_complex(node.matrices[s][e]);
            }
            entry.gamma_multipliers[s] = to_gpu_complex(node.gamma_multipliers[s]);
            entry.out_states[s] = node.out_states[s];
        }
        flat.fused_u2.push_back(entry);
    }

    flat.fused_u4.reserve(pool.fused_u4_nodes.size());
    for (const auto& node : pool.fused_u4_nodes) {
        GpuFusedU4Entry gpu_node{};
        for (int s = 0; s < 16; ++s) {
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    gpu_node.entries[s].matrix[r][c] =
                        to_gpu_complex(node.entries[s].matrix[r][c]);
                }
            }
            gpu_node.entries[s].gamma_multiplier =
                to_gpu_complex(node.entries[s].gamma_multiplier);
            gpu_node.entries[s].out_state = node.entries[s].out_state;
        }
        flat.fused_u4.push_back(gpu_node);
    }

    flat.exp_val_masks.reserve(pool.exp_val_masks.size());
    for (size_t i = 0; i < pool.exp_val_masks.size(); ++i) {
        GpuExpValMask evm{};
        auto view = pool.exp_val_masks.at(static_cast<PauliMaskHandle>(i));
        auto x_view = view.x();
        auto z_view = view.z();
        for (uint32_t w = 0; w < kPauliWords && w < x_view.num_words(); ++w)
            evm.x[w] = x_view.words[w];
        for (uint32_t w = 0; w < kPauliWords && w < z_view.num_words(); ++w)
            evm.z[w] = z_view.words[w];
        evm.sign = view.sign() ? 1 : 0;
        flat.exp_val_masks.push_back(evm);
    }

    // Build segment table from compile-time active_k profile.
    build_segments(flat, program);

    return flat;
}

}  // namespace gpu
}  // namespace clifft
