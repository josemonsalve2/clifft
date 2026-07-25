# Profiler tooling status on mi350x-es (gfx950) — 2026-07-25

## Available and WORKING through HSA dispatch (our kernels are NOT HIP-launched)

- **rocprofv3** (`/opt/rocm-7.2.3/bin/rocprofv3`) — PRIMARY. Confirmed working:
  - `--kernel-trace --stats --output-format csv` → per-dispatch CSV with
    Start/End timestamps, **LDS_Block_Size, Scratch_Size, VGPR_Count,
    Accum_VGPR_Count, SGPR_Count, Grid/Workgroup sizes** + domain_stats
    (Calls/TotalDurationNs/Avg/Min/Max/StdDev). This is the clean kernel time.
  - `--pmc <counters>` or `-i pmc.txt` → counter_collection.csv. Confirmed
    SQ_WAVES, SQ_INSTS_VALU, SQ_INSTS_MFMA collect fine through HSA. NOTE:
    "job fails if entire counter set can't be collected in a single pass" —
    split large sets across multiple runs.
  - `--hsa-trace` / `--hsa-core-trace` → HSA API timeline = dispatch overhead.
  - `--sys-trace`, `--runtime-trace` also available.
- **llvm-objdump**: `/shared/jmonsalv/software/modules/llvm/upstream_05082025/bin/llvm-objdump`
  (NOT in ROCm tree). Use for full disasm / register analysis.
- **readelf**: /usr/bin/readelf (kernel metadata notes).
- **rocprofv2, rocprof**: present (fallback).

## NOT usable

- **rocprof-compute**: binary present but its python env lacks pandas/astunparse/
  colorlover; no discoverable venv (Apex/Hyperloom/KernelForge) has pandas on the
  compute node. → SoL/roofline panels unavailable. WORKAROUND: compute roofline
  manually from rocprofv3 counters (VALU insts, FETCH/WRITE bytes, wave time) +
  known MI355X specs (HBM 8 TB/s, achievable bf16 1686 TFLOP/s from Hyperloom
  HW_SPECS). f32 FLOPs from SQ_INSTS_VALU * lanes.
- **rocprofiler-compute / omniperf / roofline**: MISSING (renamed to
  rocprof-compute, which is unusable per above).

## First-look numbers (qv10, 20000 shots, coop kernel)

- VGPR=64 (LOW — ceiling is 256; NOT register-bound), SGPR=112,
  LDS_Block_Size=25088 B (~24.5 KB → THE occupancy limiter on CDNA4;
  <=80KB allows dual-occupancy but 24.5KB*N wgs competes), Scratch=416 B.
- SQ_INSTS_MFMA = 0  ✓ (no accidental GEMM lowering — butterfly compute as designed)
- SQ_INSTS_VALU = 9.64e8, SQ_WAVES = 80000, kernel = 5.68 ms.
- FLAG: disasm shows v_accvgpr_write/read (AGPR traffic) despite MFMA=0 — the
  backend is parking values in AGPRs; likely wasteful, investigate.

## Counter set to collect (split across passes if needed)

Pass A (occupancy/compute): SQ_WAVES SQ_INSTS_VALU SQ_INSTS_MFMA SQ_INSTS_SALU SQ_INSTS_LDS
Pass B (memory):            TCC_HIT_sum TCC_MISS_sum TCC_EA_RDREQ_sum TCC_EA_WRREQ_sum FETCH_SIZE WRITE_SIZE
Pass C (stalls/occupancy):  SQ_BUSY_CYCLES GRBM_GUI_ACTIVE SQ_WAVE_CYCLES SQ_WAIT_INST_LDS

## Standard invocation (per circuit, warm binary)

rocprofv3 --kernel-trace --stats --output-format csv -d <out> -- run_v2 --circuit C --shots S --seed 1
rocprofv3 --pmc <setA> --output-format csv -d <out> -- run_v2 --circuit C --shots S --seed 1
