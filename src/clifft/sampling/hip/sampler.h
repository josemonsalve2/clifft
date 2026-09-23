#pragma once

#include "clifft/sampling/hip/executable_plan.h"
#include "clifft/sampling/results.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>

namespace clifft::sampling::hip {

enum class CoefficientPrecision : uint8_t {
    FP64,
    FP32,
};

// Asks the backend to size the launch itself. Thread-per-shot packs independent shots
// into a block and uses 256; the block tiers derive a size from the peak active width,
// because a block wider than the state has pairs adds idle lanes and extra reduction
// levels rather than parallelism -- measured 2.2x slower at k = 5..8 on gfx950 when a
// flat 256 was used instead. Any other value is validated and used exactly as given.
inline constexpr uint32_t kAutoBlockSize = 0;
inline constexpr uint32_t kThreadPerShotBlockSize = 256;
inline constexpr uint32_t kDefaultBlockSize = kAutoBlockSize;
inline constexpr uint32_t kDefaultMaxBatchShots = 65536;

struct SamplingOptions {
    std::optional<uint64_t> seed = std::nullopt;
    CoefficientPrecision coefficient_precision = CoefficientPrecision::FP64;
    // Auto resolves from the plan and device; an explicit tier is rejected when the plan
    // does not fit it.
    ExecutionTier tier = ExecutionTier::Auto;
    uint32_t block_size = kDefaultBlockSize;
    uint32_t max_batch_shots = kDefaultMaxBatchShots;
};

struct ReplayResult {
    bool reachable = true;
    bool survived = true;
    double log_probability = 0.0;
    SamplingResult outputs;
};

[[nodiscard]] bool is_available() noexcept;
[[nodiscard]] std::string backend_info();

// Reports the tier this executable would run on for the current device and precision,
// without uploading anything. Tier choice depends on the device's shared-memory budget,
// so a test that means to exercise a particular kernel has to ask rather than assume.
[[nodiscard]] ExecutionTier selected_tier(
    const ExecutablePlan& executable,
    CoefficientPrecision coefficient_precision = CoefficientPrecision::FP64);

// Owns one uploaded executable and a precision-specific reusable workspace.
// The object is synchronous and bound to the device current at construction.
// Overlapping calls are rejected; use a separate Sampler per caller.
class Sampler {
  public:
    explicit Sampler(const ExecutablePlan& executable,
                     CoefficientPrecision coefficient_precision = CoefficientPrecision::FP64,
                     uint32_t max_batch_shots = kDefaultMaxBatchShots,
                     ExecutionTier tier = ExecutionTier::Auto);
    ~Sampler();

    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;
    Sampler(Sampler&&) noexcept;
    Sampler& operator=(Sampler&&) noexcept;

    [[nodiscard]] SamplingResult sample(uint32_t shots, std::optional<uint64_t> seed = std::nullopt,
                                        uint32_t block_size = kDefaultBlockSize);
    [[nodiscard]] SamplingSurvivorResult sample_survivors(
        uint32_t shots, bool keep_records = false, std::optional<uint64_t> seed = std::nullopt,
        uint32_t block_size = kDefaultBlockSize);
    [[nodiscard]] ReplayResult replay_shot(std::span<const uint8_t> forced_records);

    [[nodiscard]] CoefficientPrecision coefficient_precision() const;
    [[nodiscard]] ExecutionTier execution_tier() const;
    [[nodiscard]] uint32_t max_batch_shots() const;
    [[nodiscard]] size_t allocated_device_bytes() const;
    [[nodiscard]] uint32_t num_visible_records() const;
    [[nodiscard]] uint32_t num_records() const;
    [[nodiscard]] uint32_t num_detectors() const;
    [[nodiscard]] uint32_t num_observables() const;
    [[nodiscard]] uint32_t num_exp_vals() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] SamplingResult sample(const ExecutablePlan& executable, uint32_t shots,
                                    const SamplingOptions& options = {});
[[nodiscard]] SamplingSurvivorResult sample_survivors(const ExecutablePlan& executable,
                                                      uint32_t shots, bool keep_records = false,
                                                      const SamplingOptions& options = {});
[[nodiscard]] ReplayResult replay_shot(
    const ExecutablePlan& executable, std::span<const uint8_t> forced_records,
    CoefficientPrecision coefficient_precision = CoefficientPrecision::FP64,
    ExecutionTier tier = ExecutionTier::Auto);

}  // namespace clifft::sampling::hip
