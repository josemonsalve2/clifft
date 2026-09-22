<!--pytest-codeblocks:skipfile-->

# HIP Backend

!!! warning "Experimental and source-build only"
    The AMD HIP backend is not part of Clifft's published wheels or stable API.
    It supports a limited hardware and workload tier, requires an explicit
    source build, and may change without compatibility guarantees. It is never
    selected automatically.

Clifft's CPU implementation is the stable reference. The experimental HIP
backend shares circuit parsing, compilation, and symbolic planning with it,
then lowers the prepared plan into a private GPU executable. It uses separate
`Program` and `Sampler` types so backend-specific precision, workspace, and
launch controls stay outside the stable API.

## Current capabilities

| Workflow or feature | HIP support |
|---|---|
| Ordinary fixed-row sampling | Supported for eligible programs |
| Post-selected survivor sampling | Supported for eligible programs |
| Measurements, detectors, observables, and `EXP_VAL` | Supported |
| Pauli and readout noise | Supported |
| Peak active width | `k <= 30`, subject to available device memory (see execution tiers below) |
| Coefficient precision | FP64 default; FP32 experimental |
| Fixed-fault importance sampling | Not supported |
| Leakage, loss, and transition instruments | Not supported |
| Exact-probability and state-vector queries | Not supported |
| Asynchronous or multi-GPU execution | Not supported |

Unsupported programs are rejected during lowering; there is no automatic CPU fallback.

### Execution tiers

The backend runs one of three tiers. These are the same categories the CUDA backend uses,
under the same names; the capacity thresholds and lane counts below are AMD-specific, the
categories are not. Per-shot coefficient storage is `3 * 2^k` elements, so `24 * 2^k` bytes
in FP64 and `12 * 2^k` in FP32.

| Tier | Threads per shot | Coefficient storage | Auto selection |
|---|---|---|---|
| `thread_per_shot` | 1 | global slab owned by the shot | `k <= 4` |
| `block_shared` | the launch block | shared memory (LDS) | state fits the block's LDS budget: `k <= 11` on a 64 KB device, `k <= 12` on 160 KB, one width more in FP32 |
| `block_global` | the launch block | global slab owned by the shot | everything wider |

Selection is automatic from the peak active width, the coefficient precision and the
device's per-block LDS budget. `selected_tier(executable, precision)` reports the choice
without uploading anything, and `Sampler::execution_tier()` reports what a constructed
sampler resolved. Forcing a tier is not supported yet; unlike CUDA there is no
`ExecutionTier::Auto` enumerator, because there is nothing yet to distinguish it from.

Only the coefficient storage differs between the two block tiers. `thread_per_shot` also
keeps its coefficients in a global slab; what makes it distinct is that one thread owns
the whole shot, so the width is bounded by what a single thread can stream.

#### Block size

`block_size` is the launch block in every tier, but it means different things:

- `thread_per_shot` packs independent shots into the block, so any value in `1..1024` works.
- The block tiers split **one** shot's coefficients across the block, which also makes the
  block the width of the measurement reduction. That reduction is a tree that halves the
  active lane count, so the value must be a **power of two** between the 64-lane wavefront
  and the static reduction capacity (256). Values outside that range are rejected rather
  than silently replaced, so a caller tuning launch geometry is never given a size it did
  not ask for.

The default is `kAutoBlockSize` (0), which asks the backend to size the launch: 256 for
`thread_per_shot`, and for the block tiers a size derived from the peak active width. A
block wider than the state has pairs adds idle lanes and extra reduction levels rather
than parallelism -- forcing a flat 256 everywhere measured 2.2x slower across `k = 5..8`
on gfx950, with no change at `k >= 9`. Pass an explicit power of two to override.

Raising the reduction capacity above 256 is a separate performance question and is not
part of this behaviour.

#### Batching and determinism

On the block tiers a batch larger than the retained workspace is split into several
synchronous launches, as on `thread_per_shot`. Because the RNG keys off the global shot
index, splitting never changes seeded rows. On `block_global` the workspace is also
clamped to what the device can hold, so a wide plan runs as more batches rather than
failing to allocate.

Rows are not promised to match bit-for-bit **between** tiers: a block sums measurement
probabilities in a different order than a single thread, so a near-tie can resolve the
other way. Within one tier a fixed seed reproduces exactly. This matches the existing
contract between the CPU's scalar and packed modes.

Each shot on a block tier is owned by a whole block, so per-shot symbols, records and
outputs are shared state. Every write to them has a single writer, and any action that
reads a location it also writes synchronises before publishing.

## Hardware and source build

The documented target is Linux `x86_64` with ROCm and an MI300X-class `gfx942`
device. Hardware validation is still manual. Other AMD architectures can be
development targets but do not have the same conformance coverage.

HIP builds require CMake 3.21 or newer and a ROCm toolchain. The commands below
assume the HIP compiler and ROCm root are discoverable automatically unless
their locations are supplied explicitly.

Build an editable installation from a checkout:

```bash
git clone https://github.com/unitaryfoundation/clifft.git
cd clifft

uv venv
CMAKE_ARGS="-DCLIFFT_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942" \
    uv pip install -e .
```

If the HIP compiler and ROCm root are installed under `/usr`, provide them
explicitly:

```bash
CMAKE_ARGS="-DCLIFFT_ENABLE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx942 \
    -DCMAKE_HIP_COMPILER=/usr/bin/clang++-17 \
    -DCMAKE_HIP_COMPILER_ROCM_ROOT=/usr" \
    uv pip install -e .
```

For standalone C++ development, configure and build the HIP target directly:

```bash
cmake -S . -B build-hip -G Ninja \
    -DCLIFFT_ENABLE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx942
cmake --build build-hip -j
```

The build can compile device code without a visible GPU. Sampling requires a
compatible device at runtime:

```python
from clifft.experimental import hip

print(hip.is_built())
print(hip.is_available())
print(hip.backend_info())
```

Clifft does not yet publish a supported ROCm and driver matrix. Treat local
conformance testing as a requirement for experimental use.

## Compile and reuse a sampler

```python
from clifft.experimental import hip

program = hip.compile("""
    H 0
    T 0
    H 0
    M 0
    OBSERVABLE_INCLUDE(0) rec[-1]
""")

sampler = hip.Sampler(program)
result = sampler.sample(100_000, seed=1234)
print(result.measurements.shape)
print(result.observables.shape)
```

`hip.Program` and `clifft.Program` are not interchangeable. `hip.compile()`
currently accepts Stim circuit text and does not expose the CPU
`input_format` switch.

Construct one sampler per concurrently active caller and reuse it. Construction
uploads the program and allocates a bounded workspace on the device that is
current at that time. Calls on one sampler are synchronous and must not
overlap.

For post-selection, compile the detector mask into the HIP program and call
`sampler.sample_survivors()`. Fixed-row `sample()` rejects a post-selected
program. Survivor sampling always returns aggregate counts; set
`keep_records=True` to retain survivor rows.

### Precision and launch controls

FP64 coefficient evolution is the default. FP32 reduces coefficient storage
and is a separate experimental numerical mode:

```python
sampler = hip.Sampler(
    program,
    precision="fp32",
    max_batch_shots=16_384,
)
result = sampler.sample(100_000, seed=42, block_size=256)
print(sampler.allocated_device_bytes)
```

Probability reductions, normalization factors, aggregate statistics, replay
log-probabilities, and `EXP_VAL` outputs remain FP64 in both modes.

- `max_batch_shots` bounds retained device workspace. Larger requests are
  split into synchronous launches that reuse it.
- `block_size` controls launch geometry and must be between 1 and 1024.
- `allocated_device_bytes` exposes retained workspace size for experiments.

These controls are not equivalents of CPU `batch_size`, `threads`, or
`thread_layout`. A fixed seed repeats within the same HIP precision and
configuration, including across workspace batch sizes. CPU and HIP use
separate random-stream domains, so compare deterministic branches directly and
stochastic results statistically.

## Architecture

The compiler/runtime boundary is `sampling::SamplingPlan`:

```text
HIR -> SamplingPlan -> CPU ExecutablePlan -> trusted CPU sampling oracle
                    -> private HIP executable -> device interpreter
```

The HIP executable is a backend-specific packing of prepared `SamplingAction`
alternatives. It stores host-computed Pauli phases, pairings, active-width
transitions, expressions, and noise distributions. The device executes the
plan without topology planning or allocation in its dispatch loop.

CPU and HIP lowering share execution-ready Pauli preparation and result
containers. Their executable layouts, mutable state, dispatch order, and
workspace ownership remain backend-specific.

## Testing and contribution boundary

Ordinary CPU builds compile host-side HIP lowering tests. They check packed
actions, expressions, noise tables, prepared Pauli data, supported-width
validation, and rejection of unsupported plans. Adding a `SamplingAction`
without HIP lowering support fails during this build.

HIP-enabled CI additionally compiles `gfx942` device code and runs GPU-free
conformance cases. Kernel-launch tests are skipped without a visible AMD GPU;
therefore this coverage does not establish runtime correctness on hardware.

Manual MI300X tests exercise FP64 and FP32 repeatability, forced branches and
expectation values against the CPU executor, noisy distributions,
post-selection, and retained output rows. A supported backend will require
regular hardware testing and a declared ROCm/driver matrix.

For the experimental Python workflow, source map, kernel invariants, and
extension checklist, continue to
[HIP Kernel Development](hip-kernel-development.md).
