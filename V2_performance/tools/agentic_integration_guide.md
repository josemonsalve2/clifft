# Agentic GPU-Kernel-Optimization Tools — Integration Guide for the V2 HSA Kernel

**Date:** 2026-07-25
**Scope:** RESEARCH ONLY. This documents the *minimal viable path* to point one of
AMD's agentic optimization loops (KernelForge, Apex, Hyperloom) at our HIP-free,
HSA-dispatched quantum-circuit-simulator kernel. No tool was run and nothing was
modified while writing this.

---

## 0. Our kernel vs. what these tools assume

| Aspect | These tools assume (LLM kernels) | OUR kernel (V2) |
|---|---|---|
| Kernel language | Triton `.py` / HIP `.hip` / CK C++ | plain C → amdgcn → `.hsaco`, loaded via raw HSA |
| Launch | PyTorch calls the kernel | C++ CLI `build-v2-nohip/run_v2` |
| Build | `ninja` / triton JIT / `pip install -e` | `cmake` + amdgcn compile (LLVM upstream) |
| Correctness | `torch.allclose` / SNR(dB) on tensors | EXACT integer match (`passed_shots` + `observable_ones`) vs GPU-SVM reference (`run_gpu --no-postselection`) |
| Bench | Magpie compare / wall-clock driver | `rocprofv3 --kernel-trace` kernel time, or host wall time |

The correctness contract is the hard part: every tool was written around
`torch.allclose` / SNR. The build and bench contracts are actually flexible.

Confirmed on disk:
- `build-v2-nohip/run_v2` (14.7 MB, exec) — the V2 driver
- `build-gpu-mlir-mi355x/run_gpu` (2.2 MB, exec) — the SVM reference
- `/shared/jmonsalv/quantum/clifft_rl/clifft` **is a git repo** (required by the
  KernelForge loop, which commits every iteration).

---

## 1. KernelForge / Hyperloom-Forge forge-loop — DETAILED

Path: `/shared/jmonsalv/quantum/clifft_rl/KernelForge/`

### 1.1 How the loop actually works (from source)

`src/kernel_agents/loop/runner.py` — `IterationLoop.run()`:
1. `git checkout -b kernel-agent-optimize`
2. Measure baseline (optional build + one bench).
3. Each iteration: agent edits source → **git commit** → 5-stage validation →
   bench → keep (commit stays) or revert (`git revert` / `git checkout -- .`).
4. Profiling runs only on baseline + after a KEEP; feeds a bottleneck hint back.
5. Runs until `target_wall_ms` gate met or time/iteration budget exhausted;
   never self-stops on a plateau (a supervisor injects fresh directions).

`IterationConfig` (runner.py lines 50-132) — the driver contract:
- `kernel_file: str` — the file the agent edits (anchor).
- `driver_script: str` — the test/bench driver (**invoked as
  `sys.executable driver_script ...`, i.e. `python <driver>`**).
- `build_command: list[str] | None`, `build_dir: str | None` — arbitrary build
  command (e.g. our cmake/make). **NOTE:** the CLI entry points below do NOT set
  `build_command` — build is expected to happen inside the driver or the agent's
  in-session gate. To use `build_command`, drive `IterationLoop` directly (§1.5).
- `shapes: dict` with keys `minimal` / `primary` / `validation`.
- `snr_threshold: float = 30.0` — gate for the SNR path (we bypass via allclose).
- Budget: `max_iterations`, `max_time_hours`, per-step timeouts.

### 1.2 The three pluggable "MCP tools" — exact contracts

These are the load-bearing files. All three are plain subprocess wrappers that
**parse stdout with regex** — completely backend-agnostic.

**`mcp_server/tools/test.py` → `test_correctness()`** (correctness):
- Runs `python <driver> <args>` as a subprocess.
- `returncode != 0` → FAIL ("DRIVER CRASHED").
- Otherwise greps stdout+stderr for, in priority order:
  - `SNR:\s*([-\d.]+)\s*dB`  → pass if `snr_db >= threshold`
  - `allclose:\s*(True|False)` → **pass if True**
  - `max_diff:\s*(...)`
  - none found → FAIL ("NO CORRECTNESS METRIC FOUND").
- **KEY: printing `allclose: True` on exact match is a valid pass.** No SNR needed.

**`mcp_server/tools/bench.py` → `bench_wallclock()`** (performance):
- Runs `python <driver> <args> --warmup N --iters M --bench-mode`.
- Parses either many `wall_ms:\s*([\d.]+)` lines (reports their median) **or** a
  single pre-aggregated `median_ms:` / `mean_ms:` line (passed through verbatim).
- **KEY: our driver just needs to print `median_ms: <kernel_ms>`** (we can source
  that number from `rocprofv3 --kernel-trace` inside the driver).

**`mcp_server/tools/registers.py` → `check_registers()`** (optional):
- `llvm-objdump -d --mcpu=gfx950 <binary>` → VGPR/AGPR/SGPR/LDS/spill via
  `parsers/compiler_output.py:parse_register_info`.
- Defaults to searching `build_dir` for `*.so`. **Our artifact is `.hsaco`, not
  `.so`** — so pass `binary_path=<...>.hsaco` explicitly, or it finds nothing
  (non-fatal; VGPR just shows as None).

### 1.3 The 5-stage validation pipeline

`loop/validation.py` → `run_validation_pipeline()` calls `test_correctness` five
times with different `--shape`/`--mode` args:
1. smoke (`--mode smoke`, relaxed threshold), 2. shape sweep (over
`shapes["validation"]`), 3. stability (`--mode stability`), 4. determinism (runs
twice, `--mode determinism`), 5. full correctness (`--shape primary`).

Our driver can treat all `--mode` values identically (just run the exact-match
check) and ignore `--shape` if we hardcode the circuit — or map `--shape
circuit=<path>,shots=<n>` to `run_v2` args. Since we print `allclose: True`, every
stage passes on an exact match. Determinism (stage 4) is naturally satisfied:
same seed → same integer result.

### 1.4 CLI entry points

`src/kernel_agents/cli.py`:
- **`kernel-agents loop <task.yaml> --kernel <f> --driver <d>`** (line 388):
  loads shapes from YAML, builds `IterationConfig`, runs. **Does not set
  `build_command`.** Simplest for a first smoke test with `--no-agent`.
- **`kernel-agents forge-loop --kernel --driver --workspace --shapes-json
  --experiments-dir ...`** (line 496): the Hyperloom subprocess entry. Takes flags
  directly (no YAML). Adds supervisor, in-session gate, KB warm-start. Also does
  **not** set `build_command` (build is the agent's job via its Bash gate).

Neither CLI wires `build_command`, so with the stock CLI the **build must live
inside the driver** (driver rebuilds before measuring) or be done by the agent.

### 1.5 CAN we define our task? YES.

Two options:

**Option A (recommended, stock CLI):** put the build inside the driver.
Write one Python driver `V2_performance/tools/forge_driver.py` that:
- accepts `--shape`, `--mode`, `--warmup`, `--iters`, `--bench-mode` (all optional);
- in correctness mode: runs `run_v2` and `run_gpu`, compares the integer tallies,
  prints `allclose: True` / `allclose: False`;
- in `--bench-mode`: runs `rocprofv3 --kernel-trace` on `run_v2`, extracts the
  kernel time, prints `median_ms: <ms>`;
- (optional) rebuilds via `cmake --build build-v2-nohip` at the top so the agent's
  edits to the amdgcn source take effect.

Then:
```
kernel-agents forge-loop \
  --kernel  <the C source file the agent should edit> \
  --driver  V2_performance/tools/forge_driver.py \
  --workspace /shared/jmonsalv/quantum/clifft_rl/clifft \
  --shapes-json '{"primary":{"circuit":"tests/fixtures/qv10.stim","shots":50000},
                  "validation":[{"circuit":"tests/fixtures/large/circuit_d3_p0.001.stim","shots":200000}]}' \
  --snr-threshold 0 \
  --experiments-dir V2_performance/tools/forge_experiments \
  --max-iters 8 --max-hours 2 --gpu-target gfx950
```

**Option B (direct, uses `build_command`):** ~40-line Python script that
constructs `IterationConfig(build_command=["cmake","--build","build-v2-nohip"],
build_dir=..., driver_script=..., ...)`, an `ExperimentTracker`, an
`IterationLoop`, and `asyncio.run(loop.run(agent_fn=...))`. This is the only way
to get the loop itself to rebuild (cleaner separation: driver only measures).

### 1.6 Verdict on KernelForge

**Best fit.** The correctness gate accepts `allclose: True`, the bench gate
accepts `median_ms:`, the build is arbitrary. The only friction:
- build must be routed through the driver or a direct-`IterationConfig` script;
- `check_registers` expects `.so`, not `.hsaco` (minor, optional);
- the whole loop needs a **GPU node** (mi350x-es), so it runs under SLURM;
- `--kernel` must be the actual C source the agent edits, and the amdgcn compile
  must be reproducible on the compute node (LLVM upstream path + ROCm).

---

## 2. Apex grade-kernel / correctness modes — DETAILED

Path: `/shared/jmonsalv/quantum/clifft_rl/Apex/`, entry `workload_optimizer.py`,
grader `graders/kernel_grader.py`.

Apex exposes standalone `optimize-kernel` and `grade-kernel` subcommands with
`--correctness-mode {pytorch, library_test, accordo}`.

### 2.1 The three modes (from `graders/kernel_grader.py`)

**pytorch** (default): `run_magpie_compare` + `torch.allclose`. Needs a PyTorch
reference and `get_test_inputs()`. **Does not fit us** (no tensors, no torch ref).

**library_test** (`_run_library_test`, line 499): runs `unit_test_command` as a
subprocess; **exit code 0 = correct.** THIS is the permissive path we want —
*except* it gates the command string:
```python
_TRUSTED_TEST_CMD_PREFIXES = ("python -m pytest", "pytest",
                              "python -m unittest", "python -c")
```
A raw shell script (`./check.sh`) is **rejected**. But `python -c "<code>"` is
allowed, and that Python can `subprocess.run(["run_v2", ...])`, compare, and
`sys.exit(0/1)`. So our exact-match check is expressible as a one-liner
`--test-cmd 'python -c "import subprocess,sys; ...; sys.exit(0 if match else 1)"'`.

**accordo** (`_run_accordo_check`, line 596): shells out to the `accordo validate`
CLI to compare two GPU **binaries'** buffer outputs at the HSA level:
```yaml
accordo:
  kernel_name: <name>
  reference_binary: <path>     # e.g. run_gpu (SVM)
  optimized_binary: <path>     # e.g. run_v2
  tolerance: 0.001
```
Conceptually the closest to our setup (compare two HSA binaries). **BUT** it
requires the `accordo` CLI on PATH (`pip install intellikit[accordo]`), which is
**not installed** here, and Accordo compares *floating-point GPU buffers by
tolerance* — our correctness is *exact integer tallies from stdout*, not a GPU
buffer diff. Accordo would need our binaries to expose a comparable buffer and a
matching invocation contract. High friction, uncertain semantics.

### 2.2 The speedup problem in Apex

For **both** library_test and accordo, correctness is domain-specific but the
**speedup** number comes from `_try_magpie_perf_measurement` → `run_magpie_compare`
(Triton/HIP-oriented). Magpie does not know how to time our `.hsaco`/CLI, so it
will likely report no timing and the score's speedup term degrades. There is a
`_run_benchmark_script` fallback, but it's only wired into the **pytorch** path
(imports the solution `.py` and times it), not library_test/accordo. So on Apex,
we'd get a correct/incorrect verdict but **no reliable speedup**, which is the
whole point of a perf loop.

### 2.3 Verdict on Apex

**Correctness fits (via `python -c` library_test), speedup does not.** Apex's
value is its polished CLI + reflector + knowledge base, but its perf measurement
is bolted to Magpie and won't time our HSA binary. Usable for a
correctness-gated grade, poor as a perf optimization loop for this kernel.

---

## 3. Reusable components (lift WITHOUT the full framework)

If we don't adopt a full loop, these standalone pieces are directly callable:

| File | What it does | How to use standalone |
|---|---|---|
| `KernelForge/src/kernel_agents/mcp_server/tools/registers.py` `check_registers()` | `llvm-objdump -d --mcpu=gfx950` → VGPR/AGPR/SGPR/LDS/spill/occupancy | `await check_registers(binary_path="<...>.hsaco")`. Pass the `.hsaco` explicitly (its `build_dir` glob only finds `.so`). |
| `KernelForge/.../parsers/compiler_output.py` `parse_register_info` | Regex parse of objdump/readelf kernel metadata into a struct | Call directly on any objdump text. |
| `KernelForge/src/kernel_agents/loop/rocpc.py` | `rocprofiler-compute` wrapper → per-kernel System Speed-of-Light + bottleneck classifier (COMPUTE/BANDWIDTH/OCCUPANCY/LATENCY-BOUND). `resolve_rocpc()` auto-detects a usable interpreter, returns None if deps missing. | Import and call for our circuits. **Caveat:** our `TOOLING_STATUS.md` already found rocprof-compute unusable here (no pandas in any venv on the compute node). Its threshold constants (`_COMPUTE_HI=60`, `_BW_HI=50`, `_OCC_LOW=40`) are a good template for a manual classifier on our rocprofv3 counters. |
| `KernelForge/src/kernel_agents/mcp_server/tools/pmc.py` | Lighter rocprofv3 PMC path + `derive_kernel_names()` (find our dispatch name), `is_target_kernel()` (filter our kernel from reference/runtime dispatches) | Reuse `derive_kernel_names`/`is_target_kernel` to isolate `run_v2`'s kernel in a multi-dispatch rocprofv3 CSV. |
| `KernelForge/src/kernel_agents/loop/experience.py` `ExperienceLedger` + `loop/archive.py` `CandidateArchive` | The "keep a git-committed, diff+measurement+lesson ledger per iteration" pattern | Reusable pattern even for a manual optimization campaign; small, self-contained. |
| `Apex/pipeline/kernel_bottleneck.py` + `Apex/graders/score.py` | Bottleneck classification from profiler output; scoring formula (`compiled×20 + correct×100 + speedup×100`) | Score/rank our own manual attempts. |
| `Apex/tools/skills/*` (SKILL.md) | CDNA3/CDNA4 arch + rocprof-compute knowledge docs | Read-only reference for hand optimization (e.g. `mi300-cdna3-architecture`, `rocprof-compute`). |

The single highest-value lift is **`registers.py` + `parse_register_info`** on our
`.hsaco` (directly complements the VGPR=64/LDS=24.5KB findings already in
`TOOLING_STATUS.md`), plus **`pmc.py:is_target_kernel`** to isolate our dispatch.

---

## 4. The correctness-adapter problem — the smallest shim

Our check: `run_v2 --circuit C --shots S --seed 1` must produce the same
`passed_shots` + `observable_ones` as `run_gpu --circuit C --shots S --seed 1
--no-postselection`.

**Permissiveness ranking of the correctness contracts (most → least permissive):**

1. **KernelForge `test_correctness`** — print `allclose: True` on stdout. Any
   language, any comparison. Exit 0 required. *(Most permissive.)*
2. **Apex `library_test`** — subprocess exit 0 = pass, BUT command must start with
   `pytest`/`python -m pytest`/`python -m unittest`/`python -c`.
3. **Apex `accordo`** — needs `accordo` CLI + two binaries + a float-tolerance GPU
   buffer diff (semantic mismatch with our exact-int check).
4. **Apex `pytorch` / KernelForge SNR path** — needs a torch reference + tensor
   SNR/allclose. *(Most rigid; does not fit us at all.)*

**The shim (works for BOTH #1 and #2).** A single Python file that runs both
binaries, compares the integer tallies, and speaks both dialects:

```python
# V2_performance/tools/forge_driver.py  (sketch — NOT created)
import argparse, re, subprocess, sys, os
BASE = "/shared/jmonsalv/quantum/clifft_rl/clifft"
V2   = f"{BASE}/build-v2-nohip/run_v2"
SVM  = f"{BASE}/build-gpu-mlir-mi355x/run_gpu"

def tally(out):                       # extract passed_shots + observable_ones
    p = re.search(r"passed_shots[:=]\s*(\d+)", out)
    o = re.search(r"observable_ones[:=]\s*(\d+)", out)
    return (p.group(1) if p else None, o.group(1) if o else None)

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=BASE).stdout

ap = argparse.ArgumentParser()
ap.add_argument("--shape", default="")            # "circuit=...,shots=..."
ap.add_argument("--mode", default="")             # smoke/stability/determinism — ignored
ap.add_argument("--bench-mode", action="store_true")
ap.add_argument("--warmup", type=int, default=0); ap.add_argument("--iters", type=int, default=1)
a = ap.parse_args()
kv = dict(p.split("=") for p in a.shape.split(",") if "=" in p)
circ = kv.get("circuit", "tests/fixtures/qv10.stim"); shots = kv.get("shots", "50000")

if a.bench_mode:
    # rocprofv3 kernel-trace on run_v2 → print median_ms
    #   (parse Avg/Median duration from kt CSV; print "median_ms: <ms>")
    ...
    print(f"median_ms: {kernel_ms:.4f}"); sys.exit(0)

v2  = tally(run([V2,  "--circuit", circ, "--shots", shots, "--seed", "1"]))
svm = tally(run([SVM, "--circuit", circ, "--shots", shots, "--seed", "1", "--no-postselection"]))
print(f"allclose: {v2 == svm and None not in v2}")
sys.exit(0)                            # exit 0; the loop reads allclose from stdout
```

For Apex library_test, the same logic collapses into one `--test-cmd`:
`python -c "import subprocess,re,sys; ...; sys.exit(0 if v2==svm else 1)"`.

(Regex field names above are placeholders — confirm the exact stdout labels
`run_v2`/`run_gpu` print and adjust.)

---

## 5. Recommendation

### Ranking by effort-to-integrate (lowest first)

1. **KernelForge forge-loop** — LOW/MEDIUM. Correctness (`allclose:`) and bench
   (`median_ms:`) contracts already fit; one driver file + a SLURM wrapper.
   Build routed through the driver (Option A) or a ~40-line direct-`IterationLoop`
   script (Option B). It is a real measurement-driven keep/revert perf loop.
2. **Reusable components only** — LOW. Lift `registers.py`/`pmc.py`/`rocpc`
   classifier into our existing `profile_sweep.sh` workflow; no agentic loop.
   Best if we want profiling insight without an LLM in the loop yet.
3. **Apex optimize-kernel (library_test via `python -c`)** — MEDIUM. Correctness
   fits, but **speedup measurement is bound to Magpie and won't time our HSA
   binary**, gutting the perf signal. Good for grading, weak as an optimizer.
4. **Apex accordo** — HIGH. `accordo` CLI not installed; float-tolerance buffer
   diff mismatches our exact-integer check.

### Wire up FIRST: **KernelForge forge-loop (Option A driver)**

Concrete next steps:
1. **Create `V2_performance/tools/forge_driver.py`** (the §4 shim) with:
   - correctness mode → `allclose: True/False` from the integer-tally comparison;
   - `--bench-mode` → `rocprofv3 --kernel-trace` on `run_v2`, print `median_ms:`;
   - a top-of-script `cmake --build build-v2-nohip` so agent edits take effect
     (or use Option B and pass `build_command`).
2. **Confirm the exact stdout labels** `run_v2` and `run_gpu` emit for
   `passed_shots`/`observable_ones`; adjust the regex.
3. **Pick the `--kernel`** file (the C amdgcn source the agent may edit) and
   verify the amdgcn compile is reproducible on `mi350x-es` (LLVM upstream +
   ROCm 7.2.3 paths from `TOOLING_STATUS.md`).
4. **Smoke-test with no agent** first:
   `kernel-agents loop <task.yaml> --kernel <f> --driver forge_driver.py --no-agent`
   — verifies the 5-stage pipeline + bench parse against our binaries before any
   LLM cost.
5. **Run under SLURM** on `mi350x-es` (never `radha`/login — no GPU/ROCm there):
   the `forge-loop` command inside an `sbatch` wrapper mirroring `profile_sweep.sh`
   (ROCm 7.2.3 + LLVM upstream env), `--gpu-target gfx950`, small `--max-iters`
   first.
6. Optionally point `check_registers(binary_path=<.hsaco>)` at our artifact so
   VGPR/LDS show up per iteration.

### Honest caveat

If the bottleneck we care about is the **compile-time StatevectorSqueezePass /
tier classification / MLIR emission** (host-side codegen decisions) rather than
the runtime amdgcn kernel body, none of these loops help — they optimize a fixed
kernel body against a fixed driver. They fit best if the target is the *runtime
interpreter kernel* (VGPR/LDS/occupancy, the AGPR-traffic flag in
`TOOLING_STATUS.md`). Given LDS (24.5 KB) is already flagged as the occupancy
limiter and VGPR is low (64/256), there IS a runtime-kernel optimization surface
worth pointing a loop at — so KernelForge is a reasonable investment. Start with
the no-agent smoke test (step 4): if the driver + bench parse cleanly, the
LLM-in-the-loop cost is the only remaining question.
