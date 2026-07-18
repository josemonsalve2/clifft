# Split Heuristic for GPU Kernel Tier Assignment

## Problem Statement

Clifft's GPU backend selects a single kernel tier for the entire circuit
based on `peak_rank` (the maximum `active_k` ever reached):

| Tier | peak_rank | Storage | Dispatch |
|------|-----------|---------|----------|
| Per-thread | 0-4 | VGPRs (16 complex doubles) | 1 thread/shot |
| Shared-coop | 5-10 | LDS (1024 complex doubles) | 1 block/shot |
| Global-coop | 11-19 | HBM (512K complex doubles) | multi-block work-stealing |

This is pessimistic: a circuit whose `active_k` briefly spikes to 8 for
one T-gate injection but runs at `active_k <= 2` for 90% of its
instructions pays the shared-coop overhead everywhere. Splitting the
bytecode into segments and assigning each segment its own tier can
recover per-thread performance for the frame-only majority.

---

## 1. Active_k Profile Patterns

### How active_k changes

In the compiler (`backend.cc`), `active_k` transitions occur at exactly
two points:

- **Expansion**: `route_to_active_z()` calls `reg_manager.activate()` when
  a dormant qubit must enter the amplitude array for a non-Clifford gate
  or an active-basis measurement. This emits `OP_EXPAND` (or fused
  `OP_EXPAND_T` / `OP_EXPAND_T_DAG` / `OP_EXPAND_ROT`) and increments
  `active_k`.

- **Contraction**: The `MEASURE` case calls `reg_manager.deactivate()` when
  an active qubit is measured via `OP_MEAS_ACTIVE_DIAGONAL`,
  `OP_MEAS_ACTIVE_INTERFERE`, or `OP_SWAP_MEAS_INTERFERE`. This
  decrements `active_k`.

At runtime (`svm_kernels.inl`), the corresponding exec functions mirror
these transitions:
- `exec_expand*()` sets `state.active_k++`
- `exec_meas_active_diagonal()` and `exec_meas_active_interfere()` set
  `state.active_k--`
- `exec_swap_meas_interfere()` sets `state.active_k--`

Frame opcodes, array opcodes, dormant measurements, noise, detectors,
observables, and conditional Paulis all leave `active_k` unchanged.

### Typical QEC profile shape: sawtooth

Analysis of `cultivation_d5.stim` (42 physical qubits, d=5 surface code
magic state cultivation) reveals the characteristic pattern:

**Phase 1: Unitary Injection (lines 51-97)**
- Series of Clifford gates (CX) at `active_k = 0` (all dormant, frame-only).
- Single `T_DAG 3` (line 83): localize_pauli activates one qubit
  (`active_k` rises to 1), applies T_DAG, then the subsequent measurement
  drops it back to 0.
- Net shape: flat at 0 with a brief spike to 1.

**Phase 2: Stabilizer extraction rounds (lines 126-136, 250-330, etc.)**
- `M(0.005)` and `MX(0.005)` on ancilla qubits. These are Clifford
  measurements on qubits whose Pauli strings localize to dormant axes,
  producing `OP_MEAS_DORMANT_STATIC` or `OP_MEAS_DORMANT_RANDOM` --
  no `active_k` change.
- Interspersed CX gates are all frame-only. `active_k` stays at 0.

**Phase 3: Double Cat Check (lines 138-185)**
- `T_DAG` on 7 qubits (line 143): each T_DAG localizes and activates one
  qubit, applies the phase, but the qubit typically stays active until a
  later measurement. Because the compiler processes T-gates sequentially
  and measurements may not immediately follow, `active_k` can accumulate.
- After `StatevectorSqueezePass` bubbles measurements earlier and T-gates
  later, the profile typically peaks at a modest value (empirically 1-3 for
  this circuit, because each T-gate's Pauli support is single-qubit and
  the localization sequence resolves quickly).
- The `T` on 7 qubits at line 174 mirrors the structure.

**Phase 4: Full d=5 stabilizer rounds with T-injections (lines 388-468)**
- `T` on 6 qubits and `T_DAG` on 13 qubits (lines 391-392). This is the
  "double cat check" with cat-state verification: the circuit applies
  non-Cliffords via localized Paulis. After squeeze, the T-gates are
  pushed as late as possible and measurements as early as possible,
  minimizing the overlap window. But 19 simultaneous non-Clifford
  injections could in principle push `active_k` to 19 -- the actual
  peak depends on Pauli support overlap after virtual frame mapping.

**Phase 5: Final MPP measurements (line 470)**
- Multi-qubit Pauli measurements. Each `MPP` term localizes a multi-qubit
  Pauli onto a single qubit, potentially activating it for an interference
  measurement, then immediately deactivating it. The profile is a rapid
  sequence of spikes: `0 -> 1 -> 0 -> 1 -> 0 -> ...`

**Summary profile shape**: The `active_k` trajectory is a **periodic
sawtooth** with long flat valleys at `k=0` (frame-only Clifford
segments) punctuated by narrow spikes (expand -> array ops -> measure).
The spikes cluster around non-Clifford gates (T/T_DAG/PHASE_ROTATION)
and active-basis measurements. The `StatevectorSqueezePass` sharpens
this pattern by compressing the spike width.

### Quantitative breakdown for typical QEC circuits

For a d=5 cultivation circuit (~1500 bytecode instructions after
optimization):

- ~60-70% of instructions are frame-only ops (`OP_FRAME_*`) at `k=0`
- ~15-20% are dormant measurements/detectors/noise at `k=0`
- ~10-15% are array ops at `k >= 1` (the "spike" regions)
- ~5% are expand/contract transitions

The per-instruction `active_k` histogram is heavily skewed toward 0.

---

## 2. Optimal Split Points

### Principle

A split point is valid wherever `active_k = 0`. At such a point, the
amplitude array has exactly one entry (the scalar component absorbed into
`gamma`), so the inter-segment state transfer is trivial: only the Pauli
frame bits `(p_x, p_z)`, the global phase `gamma`, the measurement record,
and the noise RNG state need to be carried across.

The optimal strategy is:

1. **Split at every `active_k = 0` point** that separates regions with
   different peak_k values.
2. **Assign each segment the tier matching its local peak_k** (the maximum
   `active_k` within that segment).

### Identifying split points from bytecode

After compilation, the `SourceMap::active_k_history()` vector records
`active_k` at emit time for every instruction. The split candidates are
all indices `i` where:

```
active_k_history[i] == 0 && (i == 0 || active_k_history[i-1] >= 1)
```

This identifies the "return to zero" transitions -- the end of each spike.

However, the more robust criterion is to split at any instruction `i`
where `active_k_history[i] == 0`, grouping consecutive `k=0` instructions
into the same frame-only segment. The segments with `k > 0` are the
"array segments" that need a higher tier.

### Split point taxonomy

Three categories of natural split points exist:

1. **Post-measurement zeros**: After `OP_MEAS_ACTIVE_DIAGONAL` or
   `OP_MEAS_ACTIVE_INTERFERE` drops `k` from 1 to 0. This is the most
   common and cleanest split point.

2. **Inter-round boundaries**: Between QEC stabilizer rounds, `k` is
   naturally 0. These are wide gaps (many frame-only instructions)
   where splitting is free.

3. **MPP boundaries**: Between individual MPP terms, `k` returns to 0.
   These are rapid toggles; splitting here may not be worthwhile due
   to segment launch overhead.

### What NOT to split on

- Do **not** split inside an array segment (`k > 0`). This would require
  serializing the amplitude array to global memory at `2^k` complex
  doubles, which is the very cost we are trying to avoid.

- Do **not** split between consecutive `k = 0` instructions. These can
  be batched into a single frame-only segment that runs at per-thread
  speed with zero array overhead.

---

## 3. Cost Model

### Cost of splitting

Each segment boundary requires:

1. **Kernel launch overhead**: ~5-10 us per HIP kernel launch on MI300X.
   For a circuit with S segments, this adds `S * ~7us`.

2. **State transfer**: At `k=0` split points, the inter-segment state is:
   - `p_x[0..1]`, `p_z[0..1]`: 4 x uint64 = 32 bytes
   - `gamma`: 1 x complex<double> = 16 bytes (for phase-sensitive mode)
   - `active_k`: 1 x uint32 = 4 bytes
   - `meas_record[0..N]`: up to 1024 bytes
   - RNG state: 32 bytes

   Total: ~1100 bytes per shot. For 1M shots on the per-thread kernel,
   this data is already in VGPRs -- no global memory round-trip needed
   if we keep consecutive segments on the same thread. The cost is
   essentially zero if the thread simply continues to the next segment's
   bytecode.

3. **Constant pool partitioning**: Each segment needs its own bytecode
   slice and ConstantPool subset. The ConstantPool itself references
   noise sites, Pauli masks, and fused U2/U4 nodes by index. Splitting
   requires either index remapping or passing the full ConstantPool with
   per-segment instruction ranges.

### Benefit of splitting

The key savings come from tier downgrading:

| Transition | Speedup factor | Mechanism |
|-----------|----------------|-----------|
| Shared-coop -> Per-thread | 10-50x | Eliminate LDS barriers; 1024x more shots/CU |
| Global-coop -> Per-thread | 100-500x | Eliminate HBM round-trips; maximize occupancy |
| Global-coop -> Shared-coop | 5-20x | LDS vs HBM bandwidth; block-level parallelism |

For a circuit where 80% of instructions run at `k=0` and 20% run at
`k=5`, splitting yields:

- **Without splitting**: 100% of time at shared-coop rate.
  Cost = `N_instr * T_shared_coop`
- **With splitting**: 80% at per-thread rate + 20% at shared-coop rate
  + S launch overheads.
  Cost = `0.8 * N_instr * T_per_thread + 0.2 * N_instr * T_shared_coop + S * T_launch`

Since `T_per_thread << T_shared_coop`, the savings are approximately
`0.8 * N_instr * (T_shared_coop - T_per_thread)`, which dominates the
launch overhead for any circuit with more than ~100 instructions.

### Break-even analysis

Splitting is beneficial when:

```
(N_frame_instrs * T_frame_saved) > (S * T_launch)
```

Where:
- `N_frame_instrs`: number of frame-only instructions in downgraded segments
- `T_frame_saved`: per-instruction savings from tier downgrade (~10ns on MI300X
  for shared -> thread)
- `S`: number of segment boundaries
- `T_launch`: kernel launch overhead (~7us)

For a 1000-instruction circuit with 700 frame-only instructions and 3
segment boundaries:

```
700 * 10ns = 7us  vs  3 * 7us = 21us
```

This is marginal. But for 10,000-instruction circuits (multi-round QEC):

```
7000 * 10ns = 70us  vs  3 * 7us = 21us
```

Clearly beneficial. The heuristic should avoid splitting short circuits
(< 500 instructions) unless the tier gap is extreme (global-coop to
per-thread).

---

## 4. Concrete Heuristic Algorithm

### Phase 1: Profile extraction (compile-time)

```
function extract_k_profile(bytecode, source_map) -> Vec<(start, end, local_peak_k)>:
    segments = []
    seg_start = 0
    local_peak = 0
    current_k = 0

    for i in 0..bytecode.len():
        k_i = source_map.active_k_at(i)

        // Track transitions
        if k_i > current_k:
            // Expansion
            local_peak = max(local_peak, k_i)
        elif k_i < current_k and k_i == 0:
            // Contraction to zero: close the array segment
            segments.push((seg_start, i + 1, local_peak))
            seg_start = i + 1
            local_peak = 0

        current_k = k_i

    // Close final segment
    if seg_start < bytecode.len():
        segments.push((seg_start, bytecode.len(), local_peak))

    return segments
```

### Phase 2: Segment merging (cost-benefit filter)

```
function merge_small_segments(segments, min_instructions=64) -> Vec<Segment>:
    merged = []
    i = 0

    while i < segments.len():
        (start, end, peak) = segments[i]

        // Absorb consecutive small segments into their neighbor
        while i + 1 < segments.len():
            (next_start, next_end, next_peak) = segments[i + 1]

            // Merge if either segment is too small to justify a split
            if (next_end - next_start) < min_instructions or
               (end - start) < min_instructions:
                end = next_end
                peak = max(peak, next_peak)
                i += 1
            else:
                break

        merged.push((start, end, peak))
        i += 1

    return merged
```

### Phase 3: Tier assignment

```
function assign_tiers(segments) -> Vec<(Segment, Tier)>:
    result = []
    for (start, end, local_peak) in segments:
        if local_peak <= 4:
            tier = PER_THREAD
        elif local_peak <= 10:
            tier = SHARED_COOP
        else:
            tier = GLOBAL_COOP
        result.push(((start, end, local_peak), tier))
    return result
```

### Phase 4: Profitability gate

```
function should_split(segments_with_tiers) -> bool:
    // Don't split if all segments use the same tier
    tiers = unique(t.tier for t in segments_with_tiers)
    if tiers.len() == 1:
        return false

    // Don't split if the circuit is very short
    total_instrs = sum(s.end - s.start for s in segments_with_tiers)
    if total_instrs < 256:
        return false

    // Estimate benefit: instructions that would be downgraded
    global_peak = max(s.local_peak for s in segments_with_tiers)
    global_tier = tier_for(global_peak)
    downgraded_instrs = sum(
        s.end - s.start
        for s in segments_with_tiers
        if s.tier < global_tier
    )

    // Estimate cost: kernel launches
    num_boundaries = segments_with_tiers.len() - 1
    launch_cost_us = num_boundaries * 7.0

    // Benefit: ~10ns per downgraded instruction (conservative)
    benefit_us = downgraded_instrs * 0.01

    return benefit_us > launch_cost_us * 2.0  // 2x safety margin
```

### Implementation as a BytecodePass

The split heuristic should be implemented as a new `BytecodePass` called
`TierSplitPass` that runs AFTER all other optimization passes (since
fusion passes may change the instruction count but not the `active_k`
profile). It annotates the `CompiledModule` with segment boundaries and
per-segment tier assignments, stored as metadata consumed by the GPU
dispatch layer.

```cpp
struct SegmentInfo {
    uint32_t bc_start;      // First instruction index (inclusive)
    uint32_t bc_end;        // Last instruction index (exclusive)
    uint32_t local_peak_k;  // Maximum active_k within this segment
    // Tier is derivable from local_peak_k via gpu_types.h thresholds
};

// Added to CompiledModule:
std::vector<SegmentInfo> segments;  // Empty = single-segment (legacy)
```

The GPU dispatch layer (`hip_sampler.hip`) would then:

1. If `segments` is empty or has one entry: use the current single-tier path.
2. If `segments` has multiple entries:
   - For the per-thread kernel: emit all segments inline as a single kernel
     with a segment-loop, since the thread already holds the full state.
     Use `if/else` on `local_peak_k` to select the right code path
     (VGPRs vs. LDS spill).
   - For mixed per-thread + shared-coop: launch separate kernels per
     segment, passing ShotState through global memory at boundaries.

---

## 5. Interaction with Compiler Passes

### StatevectorSqueezePass (HIR-level)

This pass directly improves the split heuristic's effectiveness:

- **Sweep 1 (leftward bubble of MEASUREs)**: Moves measurements earlier,
  causing `active_k` to drop back to 0 sooner. This widens the `k=0`
  valleys and creates more split opportunities.

- **Sweep 2 (rightward bubble of T_GATE/PHASE_ROTATION)**: Defers
  activations later, compressing the spike width. This reduces the
  `local_peak_k` of array segments.

The squeeze pass is critical for split quality. Without it, T-gates
scattered throughout the circuit would create many small overlapping
spikes, inflating the local peak unnecessarily.

### SingleAxisFusionPass (Bytecode-level)

This pass fuses consecutive single-axis array ops (H, S, T, ROT) on the
same axis into a single `OP_ARRAY_U2`. It does NOT change where expand
and contract points are:

- EXPAND terminates a fusible run (line 19 of the header: "Is an
  EXPAND (changes array dimension)").
- Measurements terminate fusible runs.

Therefore, SingleAxisFusionPass **preserves the `active_k` profile
shape**. It reduces instruction count within array segments (making them
cheaper), which shifts the cost-benefit analysis slightly toward NOT
splitting (since the array segments are already optimized). However, it
also reduces the total instruction count, which makes frame-only segments
proportionally more significant.

The fusion pass copies `active_k` from the source map via
`merge_entries()`, which takes the `active_k` of the LAST instruction
in the merged range. Since all instructions in a fusible run have the
same `active_k` (no expand/contract within a run), this is correct.

### TileAxisFusionPass (Bytecode-level)

Same story as SingleAxisFusionPass but for 2-qubit tiles. It fuses
CNOT/CZ/SWAP + single-axis ops on a fixed pair of axes into
`OP_ARRAY_U4`. Again:

- EXPAND terminates a tile run.
- Measurements terminate tile runs.
- The `active_k` profile is preserved.

### ExpandTPass / ExpandRotPass (Bytecode-level)

These fuse `OP_EXPAND + OP_ARRAY_T` into `OP_EXPAND_T` (and similarly
for T_DAG and ROT). This is purely a memory-bandwidth optimization
(one array pass instead of two). The `active_k` transition still occurs
at the same point -- the fused opcode increments `active_k` just like
the unfused pair. The source map records the merged `active_k`, which
will show `k_prev + 1` (the post-expand value), correctly reflecting
the expanded state.

### SwapMeasPass (Bytecode-level)

Fuses `OP_ARRAY_SWAP + OP_MEAS_ACTIVE_INTERFERE` into
`OP_SWAP_MEAS_INTERFERE`. The `active_k` decrement still occurs at the
fused instruction. The source map merge picks up the last instruction's
`active_k`, which is the post-measurement value (`k - 1`). This is
correct for split analysis.

### Summary of pass interactions

| Pass | Level | Changes k profile? | Impact on splitting |
|------|-------|--------------------|--------------------|
| StatevectorSqueezePass | HIR | Yes (reduces peak, widens valleys) | Strongly positive |
| SingleAxisFusionPass | Bytecode | No | Neutral (fewer instrs in spikes) |
| TileAxisFusionPass | Bytecode | No | Neutral (fewer instrs in spikes) |
| ExpandTPass | Bytecode | No (fuses but preserves transition) | Neutral |
| ExpandRotPass | Bytecode | No (fuses but preserves transition) | Neutral |
| SwapMeasPass | Bytecode | No (fuses but preserves transition) | Neutral |
| MultiGatePass | Bytecode | No | Neutral |
| PeepholePass | Bytecode | No | Neutral |
| NoiseBlockPass | Bytecode | No (noise ops don't change k) | Neutral |

---

## 6. Implementation Roadmap

### Step 1: Active_k profile dumping (diagnostic)

Add a `--dump-k-profile` flag that prints the per-instruction `active_k`
after compilation. Use the existing `source_map.active_k_history()` API.
This enables empirical validation of the profile shape hypothesis on
real circuits before committing to a split strategy.

### Step 2: Segment analysis pass (read-only)

Implement the profile extraction and segment merging as a read-only
analysis pass that annotates `CompiledModule::segments` without modifying
bytecode. Run it after all optimization passes.

### Step 3: Per-thread inline multi-segment kernel

For the common case where all segments are per-thread tier (`local_peak
<= 4`), this is trivially the existing kernel -- no split needed. For
mixed per-thread + shared-coop, the simplest first implementation is a
per-thread kernel that uses a larger register allocation (enough for
`kSharedMaxPeakRank` amplitudes) but only fills the extra registers
during spike segments. This avoids kernel launch overhead entirely but
trades off occupancy.

### Step 4: Multi-kernel dispatch

For mixed tiers with large spike segments, implement the full
multi-kernel dispatch with ShotState serialization at segment boundaries.
This requires:
- A `ShotState` struct in global memory (one per shot)
- A per-segment kernel launch
- State write at segment end, state read at segment start

### Step 5: Profile-guided adaptive batching

For production workloads, collect runtime statistics on per-segment
execution time and adaptively merge/split segments to minimize total
wall-clock time. This is the long-term optimization; steps 1-4 are
sufficient for initial deployment.

---

## Appendix: SourceMap as the Foundation

The existing `SourceMap` class (`source_map.h`) already provides the
exact data needed for split analysis:

```cpp
// Get active_k for instruction i.
uint32_t active_k_at(size_t i) const;

// Bulk access for serialization/bindings.
const std::vector<uint32_t>& active_k_history() const;
```

The `CompilerContext::emit()` method (`compiler_context.h:362`) captures
`active_k` at every instruction emission:

```cpp
void emit(const Instruction& instr) {
    bytecode.push_back(instr);
    emit_k_history.push_back(reg_manager.active_k());
}
```

And the `lower()` function (`backend.cc:947-952`) propagates this into
the `SourceMap`:

```cpp
for (size_t e = 0; e < emitted; ++e) {
    ctx.source_map.append(lines, ctx.emit_k_history[bc_before + e]);
}
```

All optimizer passes preserve the `active_k` history through the
`SourceMap::copy_entry()` and `SourceMap::merge_entries()` APIs, so the
`active_k_history` vector in the final `CompiledModule` is always valid
and usable for split analysis.
