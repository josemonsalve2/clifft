#pragma once

#include "clifft/sampling/hip/device_program.h"
#include "clifft/sampling/plan.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace clifft::sampling::hip {

// Execution tiers, selected by peak active width.
//
// The first tier assigns a complete shot to one thread, so the whole 2^k coefficient
// state is private and k is bounded by what one thread can afford. The cooperative
// tiers give a shot to a whole workgroup instead, which removes that bound: the state
// lives either in on-chip shared memory (fast, capacity-limited) or in a global slab
// (slower, effectively unbounded).
//
// Per-shot coefficient storage is coefficient_elements_per_shot(k) = 3 * 2^k elements,
// i.e. 24 * 2^k bytes in FP64 and 12 * 2^k bytes in FP32.
enum class ExecutionTier : uint8_t {
    ThreadPerShot,      // one shot per thread, state private to the thread
    CooperativeLds,     // one shot per workgroup, state in LDS
    CooperativeGlobal,  // one shot per workgroup, state in a global slab
};

[[nodiscard]] constexpr const char* tier_name(ExecutionTier tier) {
    switch (tier) {
        case ExecutionTier::ThreadPerShot:
            return "thread-per-shot";
        case ExecutionTier::CooperativeLds:
            return "cooperative-lds";
        case ExecutionTier::CooperativeGlobal:
            return "cooperative-global";
    }
    return "unknown";
}

inline constexpr uint32_t kThreadPerShotMaxActiveWidth = 4;

// Widest plan any tier can hold. 2^30 coefficients already exceeds device memory at
// every precision; the real limit is checked against the device at Sampler construction.
inline constexpr uint32_t kMaxSupportedActiveWidth = 30;

// Bytes of coefficient storage one shot needs at a given width and element size.
[[nodiscard]] constexpr uint64_t coefficient_bytes_per_shot(uint32_t peak_active_width,
                                                            uint64_t element_bytes) {
    return detail::coefficient_elements_per_shot(peak_active_width) * element_bytes;
}

// Tier for one shot, given the element size and the device's per-workgroup LDS budget.
// Precision is a property of the Sampler, not of the plan, so this is deliberately not
// decided during lowering.
[[nodiscard]] constexpr ExecutionTier select_execution_tier(uint32_t peak_active_width,
                                                            uint64_t element_bytes,
                                                            uint64_t lds_bytes_per_workgroup) {
    if (peak_active_width <= kThreadPerShotMaxActiveWidth) {
        return ExecutionTier::ThreadPerShot;
    }
    // The reduction scratch is static LDS the cooperative kernel always occupies, so the
    // coefficient state only gets what is left of the workgroup budget.
    const uint64_t usable = lds_bytes_per_workgroup > detail::kCooperativeReductionBytes
                                ? lds_bytes_per_workgroup - detail::kCooperativeReductionBytes
                                : 0;
    if (coefficient_bytes_per_shot(peak_active_width, element_bytes) <= usable) {
        return ExecutionTier::CooperativeLds;
    }
    return ExecutionTier::CooperativeGlobal;
}

class ExecutablePlan {
  public:
    explicit ExecutablePlan(const SamplingPlan& plan);

    [[nodiscard]] uint32_t initial_active_width() const { return initial_active_width_; }
    [[nodiscard]] uint32_t peak_active_width() const { return peak_active_width_; }
    [[nodiscard]] uint32_t num_symbols() const { return num_symbols_; }
    [[nodiscard]] uint32_t num_records() const {
        return num_visible_records_ + num_hidden_records_;
    }
    [[nodiscard]] uint32_t num_visible_records() const { return num_visible_records_; }
    [[nodiscard]] uint32_t num_detectors() const { return num_detectors_; }
    [[nodiscard]] uint32_t num_observables() const { return num_observables_; }
    [[nodiscard]] uint32_t num_exp_vals() const { return num_exp_vals_; }
    [[nodiscard]] bool has_postselection() const { return has_postselection_; }
    [[nodiscard]] uint32_t num_actions() const { return static_cast<uint32_t>(actions_.size()); }
    [[nodiscard]] size_t packed_bytes() const;
    [[nodiscard]] std::string inspect() const;

    [[nodiscard]] std::span<const detail::Action> actions() const { return actions_; }
    [[nodiscard]] std::span<const detail::Expression> expressions() const { return expressions_; }
    [[nodiscard]] std::span<const uint32_t> expression_terms() const { return expression_terms_; }
    [[nodiscard]] std::span<const detail::NoiseSite> noise_sites() const { return noise_sites_; }
    [[nodiscard]] std::span<const detail::NoiseOutcome> noise_outcomes() const {
        return noise_outcomes_;
    }

  private:
    uint32_t append_expression(const AffineBool& expression);
    uint32_t append_record_parity(const RecordParity& parity);
    void lower_observable_value(detail::Action& action, const ObservableValue& value);
    detail::Action lower_action(const PlannedAction& planned);

    uint32_t initial_active_width_ = 0;
    uint32_t peak_active_width_ = 0;
    uint32_t num_symbols_ = 0;
    uint32_t num_visible_records_ = 0;
    uint32_t num_hidden_records_ = 0;
    uint32_t num_detectors_ = 0;
    uint32_t num_observables_ = 0;
    uint32_t num_exp_vals_ = 0;
    bool has_postselection_ = false;
    std::vector<detail::Action> actions_;
    std::vector<detail::Expression> expressions_;
    std::vector<uint32_t> expression_terms_;
    std::vector<detail::NoiseSite> noise_sites_;
    std::vector<detail::NoiseOutcome> noise_outcomes_;
};

}  // namespace clifft::sampling::hip
