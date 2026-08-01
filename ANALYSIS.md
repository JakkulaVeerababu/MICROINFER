# MicroInfer Performance Analysis

## Phase 3 vs Phase 5 Throughput Drift (Resolved)

**The Issue:**
Phase 3 (`benchmark_scheduler.py`) and Phase 5 (`baseline_vllm.py` Fallback) were presented as conceptually identical tests of the MicroInfer ContinuousBatchScheduler under a 16-request concurrent load. However, the README numbers drifted massively over time, reaching a point where Phase 3 reported `55.57 tok/s` while Phase 5 reported `128.52 tok/s`.

**The Diagnosis:**
1. **Code Duplication Drift:** The Phase 5 script originally implemented its own bespoke `drain_wave()` loop that mimicked Phase 3's `_run_wave()` but did not actually share code.
2. **Measurement Artifacts:** Phase 3 includes a background thread (`GpuUtilSampler`) that spins up `nvidia-smi` subprocesses every 1 second. When run locally, this dropping of the Python GIL overhead caused Phase 3 to measure `~30.98 tok/s`, while the identical workload in Phase 5's loop (without the sampler) measured `~35.88 tok/s`.
3. **Hallucinated / Stale Data:** The `128.52 tok/s` figure previously recorded in the README was a massive outlier (2-3x the actual capacity of the RTX 4050 GPU for this workload) and was likely hallucinated in a previous documentation pass or captured on a completely different machine.

**The Solution:**
To structurally guarantee that Phase 5 fallback numbers never diverge from Phase 3, we refactored `baseline_vllm.py`. When vLLM is unavailable, Phase 5 no longer implements its own bespoke loop. Instead, it literally imports and executes `run_scheduler_benchmark` directly from `benchmark_scheduler.py`.

As a result:
- Both Phase 3 and Phase 5 now execute the exact same Python bytecode.
- Both now report `30.98 ± 5.43 tok/s` under identical 16-request concurrency on the local hardware.
- The `README.md` has been updated to reflect the true, measured performance without duplicate drift.
