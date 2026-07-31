"""
MicroInfer - Master Benchmark Runner
Executes all six phase harnesses under canonical matched conditions and
prints a consolidated results table.

Usage:
    python benchmarks/run_all_benchmarks.py
    python benchmarks/run_all_benchmarks.py --phases 0 1 2   # subset
    python benchmarks/run_all_benchmarks.py --skip 4         # skip quant

Results are also saved to benchmarks/results/master_results.json.

Canonical conditions enforced across all phases:
  - Same 3 prompts
  - Same max_new_tokens = 64
  - 3 discarded warm-up runs/waves before timing
  - 10 timed runs (sequential phases) OR 3 waves x 16 concurrent requests (batch phases)
  - mean / p50 / p99 reported for every latency metric
"""

import sys
import json
import argparse
import statistics
import traceback
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def _mean_across_prompts(phase_data: dict, metric: str, stat: str) -> float:
    """Return mean of a stat across all prompt results for sequential phases."""
    vals = []
    for r in phase_data.get("results", []):
        m = r.get(metric, {})
        if isinstance(m, dict):
            vals.append(m.get(stat, 0.0))
        else:
            vals.append(float(m))
    return statistics.mean(vals) if vals else 0.0


def _get_batch_stat(phase_data: dict, metric: str, stat: str) -> float:
    """Return aggregate stat from batch phases (Phase 3 / Phase 5)."""
    agg = phase_data.get("aggregate", {})
    m = agg.get(metric, {})
    if isinstance(m, dict):
        return m.get(stat, 0.0)
    return float(m) if m else 0.0


def run_phase(name: str, fn, *args, **kwargs):
    print(f"\n{'=' * 60}")
    print(f"  RUNNING {name}")
    print(f"{'=' * 60}")
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"\n[ERROR] {name} failed: {e}")
        traceback.print_exc()
        return None


def print_master_table(results: dict):
    gpu_name = "CPU"
    for v in results.values():
        if v and v.get("gpu_name"):
            gpu_name = v["gpu_name"]
            break

    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print("\n\n" + "=" * 90)
    print(f"  MICROINFER MASTER BENCHMARK RESULTS")
    print(f"  GPU     : {gpu_name}")
    print(f"  Tokens  : 64 max_new_tokens (all phases)")
    print(f"  Warm-up : 3 discarded runs/waves (all phases)")
    print(f"  Timed   : 10 runs (sequential) | 3 waves x 16 requests (batch phases)")
    print(f"  Run at  : {ts}")
    print("=" * 90)

    # Header
    col = 28
    print(f"\n{'Phase':<{col}} {'TTFT mean':>12} {'TPOT mean':>12} {'Throughput mean':>17} {'VRAM':>8}")
    print(f"{'':-<{col}} {'(ms)':>12} {'(ms/tok)':>12} {'(tok/sec)':>17} {'(GB)':>8}")

    phases_order = ["phase0", "phase1", "phase2", "phase3", "phase4", "phase5"]
    labels = {
        "phase0": "Phase 0 - HF Baseline",
        "phase1": "Phase 1 - Naive (uncached)",
        "phase2": "Phase 2 - KV-Cache",
        "phase3": "Phase 3 - Sched (16-concurrent)",
        "phase4": "Phase 4 - INT8 Quant",
        "phase5": "Phase 5 - Fallback Scheduler (16-concurrent)",
    }
    batch_phases = {"phase3", "phase5"}

    for key in phases_order:
        d = results.get(key)
        if d is None:
            print(f"  {labels[key]:<{col-2}} {'-- SKIPPED / FAILED --':>53}")
            continue

        vram = d.get("peak_vram_gb", 0.0)

        if key in batch_phases:
            ttft = _get_batch_stat(d, "ttft_ms", "mean")
            tpot = 0.0   # batch phases don't compute per-token decode latency
            tp   = _get_batch_stat(d, "throughput_tok_per_sec", "mean")
            tpot_str = "  --"
            ttft_str = f"{ttft:>12.1f}" if ttft else "  --"
        else:
            ttft = _mean_across_prompts(d, "ttft_ms", "mean")
            tpot = _mean_across_prompts(d, "tpot_ms", "mean")
            tp   = _mean_across_prompts(d, "throughput_tok_per_sec", "mean")
            ttft_str = f"{ttft:>12.1f}"
            tpot_str = f"{tpot:>12.2f}"

        tp_str = f"{tp:>17.2f}"
        vram_str = f"{vram:>8.2f}"
        print(f"  {labels[key]:<{col-2}} {ttft_str} {tpot_str} {tp_str} {vram_str}")

    print("\n")
    print("Notes:")
    print("  Phase 3/5: TTFT is mean per-request TTFT across all requests in all waves.")
    print("             TPOT (--) is not tracked at wave level; see per-phase JSON.")
    print("             Throughput = total tokens / wall time across all concurrent requests.")
    print("  Phase 3  : Scheduler drains requests sequentially inside step(). If Phase 2")
    print("             throughput exceeds Phase 3, that is correct -- the scheduler adds")
    print("             queue-management overhead without batched-tensor GPU execution.")
    print("  Phase 5  : If vLLM was not installed, fallback = MicroInfer scheduler")
    print("             under same concurrency as Phase 3. No synthetic multipliers.")
    print("  All raw per-run numbers are in benchmarks/results/*_raw.json.")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run all MicroInfer benchmark phases.")
    parser.add_argument("--phases", nargs="*", type=int,
                        help="Phases to run (0-5). Default: all.")
    parser.add_argument("--skip", nargs="*", type=int, default=[],
                        help="Phases to skip.")
    args = parser.parse_args()

    all_phases = list(range(6))
    if args.phases is not None:
        phases_to_run = [p for p in args.phases if p in all_phases]
    else:
        phases_to_run = all_phases
    phases_to_run = [p for p in phases_to_run if p not in (args.skip or [])]

    print(f"[Runner] Running phases: {phases_to_run}")
    print(f"[Runner] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Runner] GPU: {torch.cuda.get_device_name(0)}")

    results = {}

    if 0 in phases_to_run:
        from benchmarks.baseline_hf import benchmark_hf_generate
        results["phase0"] = run_phase("PHASE 0: HuggingFace Baseline", benchmark_hf_generate)

    if 1 in phases_to_run:
        from benchmarks.benchmark_naive import run_naive_benchmark
        results["phase1"] = run_phase("PHASE 1: Naive (Uncached)", run_naive_benchmark)

    if 2 in phases_to_run:
        from benchmarks.benchmark_cached import run_cached_benchmark
        results["phase2"] = run_phase("PHASE 2: KV-Cache Generator", run_cached_benchmark)

    if 3 in phases_to_run:
        from benchmarks.benchmark_scheduler import run_scheduler_benchmark
        results["phase3"] = run_phase("PHASE 3: Continuous Batching Scheduler", run_scheduler_benchmark)

    if 4 in phases_to_run:
        from benchmarks.benchmark_quant import run_quant_benchmark
        results["phase4"] = run_phase("PHASE 4: INT8 Quantized Engine", run_quant_benchmark)

    if 5 in phases_to_run:
        from benchmarks.baseline_vllm import run_vllm_benchmark
        results["phase5"] = run_phase("PHASE 5: Fallback Scheduler Under Concurrent Load", run_vllm_benchmark)

    print_master_table(results)

    master_out = {
        "run_timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "phases_run": phases_to_run,
        "results": {k: v for k, v in results.items() if v is not None},
    }
    master_file = RESULTS_DIR / "master_results.json"
    with open(master_file, "w") as f:
        json.dump(master_out, f, indent=2)
    print(f"[Runner] Full master results -> '{master_file}'")


if __name__ == "__main__":
    main()
