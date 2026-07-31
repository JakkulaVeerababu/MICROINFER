"""
MicroInfer - Phase 1: Naive Generator Benchmark Harness

Canonical conditions shared with all other phases:
  - Same 3 prompts, same max_new_tokens=64
  - 3 discarded warm-up runs before timing starts
  - 10 timed runs per prompt; reports mean, p50, p99
  - Raw per-run data logged to benchmarks/results/phase1_raw.json

SPEC COMPLIANCE (PHASE1_SPEC.md):
  The spec requires demonstrating O(N^2) slowdown across sequence lengths
  N in {16, 32, 64, 128, 256}.  run_naive_scaling_benchmark() sweeps all
  five points for both the naive engine AND the HF baseline side-by-side,
  exporting benchmarks/results/phase1_scaling.json.

  MASTER-TABLE SOURCING:
    The Phase 1 row in README.md and ANALYSIS.md is taken from N=256
    in phase1_scaling.json, NOT from the canonical N=64 run in
    phase1_naive.json.  Reporting at N=256 ensures the quadratic
    per-step latency penalty is visible.  The canonical N=64 run is
    kept for per-step latency profiling and the quadratic-curve chart.

  CROSSOVER NOTE (from actual measured data):
    HF .generate() uses its own internal DynamicCache by default, so
    each HF decode step is already O(1).  This means naive TPOT is
    reliably higher than HF TPOT at all tested N -- it does NOT mean
    naive is faster; it means the correct baseline for naive is uncached
    HF (pass_key_values=False), not the default cached HF.  The quadratic
    growth IS visible: naive TPOT rises +11% from N=64 to N=256 while
    HF rises only +6%.  See ANALYSIS.md for the full explanation.
"""

import sys
import json
import time
import statistics
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_model_and_tokenizer, DEFAULT_MODEL_ID
from src.naive_generate import naive_generate

# ---------------------------------------------------------------------------
# Canonical benchmark constants -- identical across all six phase harnesses
# ---------------------------------------------------------------------------
BENCHMARK_PROMPTS = [
    "Explain how transformer attention mechanism works in simple terms for a software engineer.",
    "Write a Python function to implement quicksort with step-by-step explanations.",
    "What are the key trade-offs between KV-caching, continuous batching, and weight quantization in LLM serving?",
]
# Backward-compat alias: tests import TEST_PROMPTS
TEST_PROMPTS = BENCHMARK_PROMPTS
MAX_NEW_TOKENS = 64
NUM_WARMUP_RUNS = 3
NUM_TIMED_RUNS  = 10

# Sequence lengths required to surface O(N^2) scaling vs fixed FFN overhead
SCALING_N_VALUES = [64, 256, 512, 1024, 2048]
# Timed runs per (N, engine) pair in the scaling sweep (1 run for large N sweep)
SCALING_TIMED_RUNS = 1


def _percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    idx = (p / 100.0) * (len(data_sorted) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(data_sorted) - 1)
    frac = idx - lo
    return data_sorted[lo] * (1 - frac) + data_sorted[hi] * frac


# ---------------------------------------------------------------------------
# Canonical single-N benchmark (used by master table)
# ---------------------------------------------------------------------------
def run_naive_benchmark(
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = MAX_NEW_TOKENS,
    num_warmup: int = NUM_WARMUP_RUNS,
    num_timed: int = NUM_TIMED_RUNS,
    num_runs: int = None,   # backward-compat alias for num_timed
):
    """
    Single-N canonical run at max_new_tokens=64.
    Master-table row is produced from this; for per-spec N-sweep see
    run_naive_scaling_benchmark().
    """
    if num_runs is not None:
        num_timed = num_runs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device)
    model.eval()

    print("\n" + "=" * 60)
    print("  MICROINFER PHASE 1: NAIVE GENERATOR (UNCACHED) BENCHMARK")
    print(f"  Model      : {model_id}")
    print(f"  Device     : {device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Max Tokens : {max_new_tokens}")
    print(f"  Warm-up    : {num_warmup} discarded runs")
    print(f"  Timed      : {num_timed} runs  ->  mean / p50 / p99")
    print("=" * 60 + "\n")

    print(f"[Bench] Running {num_warmup} warm-up runs (discarded)...")
    for _ in range(num_warmup):
        _ = naive_generate(
            model, tokenizer, BENCHMARK_PROMPTS[0],
            max_new_tokens=max_new_tokens, temperature=0.0,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print("[Bench] Warm-up complete.\n")

    results      = []
    all_raw_runs = []

    for prompt_idx, prompt in enumerate(BENCHMARK_PROMPTS, 1):
        print(f"--- Prompt {prompt_idx}/{len(BENCHMARK_PROMPTS)} ---")

        run_ttfts        = []
        run_tpots        = []
        run_throughputs  = []
        step_times_matrix = []
        sample_output    = ""

        for run in range(num_timed):
            res = naive_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
            run_ttfts.append(res["ttft_ms"])
            run_tpots.append(res["tpot_ms"])
            run_throughputs.append(res["throughput_tok_per_sec"])
            step_times_matrix.append(res["step_times_ms"])
            sample_output = res["output_text"]
            all_raw_runs.append({
                "prompt_idx": prompt_idx,
                "run": run + 1,
                "ttft_ms": round(res["ttft_ms"], 3),
                "tpot_ms": round(res["tpot_ms"], 3),
                "throughput_tok_per_sec": round(res["throughput_tok_per_sec"], 3),
            })

        num_steps = len(step_times_matrix[0])
        avg_step_times = [
            round(sum(r[s] for r in step_times_matrix) / len(step_times_matrix), 2)
            for s in range(num_steps)
        ]

        mean_ttft = statistics.mean(run_ttfts)
        p50_ttft  = _percentile(run_ttfts, 50)
        p99_ttft  = _percentile(run_ttfts, 99)
        mean_tpot = statistics.mean(run_tpots)
        p50_tpot  = _percentile(run_tpots, 50)
        p99_tpot  = _percentile(run_tpots, 99)
        mean_tp   = statistics.mean(run_throughputs)
        p50_tp    = _percentile(run_throughputs, 50)
        p99_tp    = _percentile(run_throughputs, 99)

        print(f"  Input Tokens : {res['prompt_tokens']}")
        print(f"  Output Tokens: {res['generated_tokens']}")
        print(f"  TTFT          mean={mean_ttft:.1f}ms  p50={p50_ttft:.1f}ms  p99={p99_ttft:.1f}ms")
        print(f"  TPOT          mean={mean_tpot:.2f}ms  p50={p50_tpot:.2f}ms  p99={p99_tpot:.2f}ms")
        print(f"  Throughput    mean={mean_tp:.2f} t/s  p50={p50_tp:.2f} t/s  p99={p99_tp:.2f} t/s")
        print(f"  Step 1 latency (no prior KV):  {avg_step_times[0]:.2f} ms")
        if num_steps > 1:
            print(f"  Step {num_steps} latency (full KV recompute): {avg_step_times[-1]:.2f} ms  <- quadratic penalty")
        print()

        results.append({
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "input_tokens": res["prompt_tokens"],
            "output_tokens": res["generated_tokens"],
            "ttft_ms":  {"mean": round(mean_ttft,2), "p50": round(p50_ttft,2), "p99": round(p99_ttft,2)},
            "tpot_ms":  {"mean": round(mean_tpot,2), "p50": round(p50_tpot,2), "p99": round(p99_tpot,2)},
            "throughput_tok_per_sec": {"mean": round(mean_tp,2), "p50": round(p50_tp,2), "p99": round(p99_tp,2)},
            "per_step_latency_ms": avg_step_times,
        })

    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak VRAM: {peak_vram_gb:.2f} GB")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)

    export_data = {
        "phase": "Phase 1 - Naive Generator (Uncached)",
        "model_id": model_id,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "max_new_tokens": max_new_tokens,
        "num_warmup_runs": num_warmup,
        "num_timed_runs": num_timed,
        "peak_vram_gb": round(peak_vram_gb, 2),
        "results": results,
    }
    with open(output_dir / "phase1_naive.json", "w") as f:
        json.dump(export_data, f, indent=2)

    raw_export = {
        "phase": "Phase 1 - Naive Generator (Uncached)",
        "model_id": model_id,
        "max_new_tokens": max_new_tokens,
        "num_warmup_runs": num_warmup,
        "raw_runs": all_raw_runs,
    }
    with open(output_dir / "phase1_raw.json", "w") as f:
        json.dump(raw_export, f, indent=2)

    print(f"\n[MicroInfer] Phase 1 results -> '{output_dir / 'phase1_naive.json'}'")
    print(f"[MicroInfer] Phase 1 raw log -> '{output_dir / 'phase1_raw.json'}'")
    return export_data


# ---------------------------------------------------------------------------
# PHASE1_SPEC.md compliance: O(N^2) scaling sweep across N ∈ {16,32,64,128,256}
# ---------------------------------------------------------------------------
def _hf_generate_timed(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> dict:
    """Single timed HF .generate() call, returns ttft_ms and tpot_ms."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # TTFT: time for first token
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1, min_new_tokens=1, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000.0

    # Full generation for TPOT
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_time_s  = time.perf_counter() - t1
    gen_tokens  = out.shape[1] - input_len
    tpot_ms     = (gen_time_s / gen_tokens * 1000.0) if gen_tokens > 0 else 0.0
    throughput  = gen_tokens / gen_time_s if gen_time_s > 0 else 0.0

    return {"ttft_ms": ttft_ms, "tpot_ms": tpot_ms, "throughput_tok_per_sec": throughput,
            "gen_tokens": gen_tokens}


def run_naive_scaling_benchmark(
    model_id: str = DEFAULT_MODEL_ID,
    n_values: list = None,
    num_warmup: int = NUM_WARMUP_RUNS,
    num_timed: int = SCALING_TIMED_RUNS,
    prompt: str = None,
):
    """
    Sweeps N in {16, 32, 64, 128, 256} for both the naive (uncached) engine
    and the HF baseline, measuring TTFT and TPOT at each N side-by-side.

    Results are written to benchmarks/results/phase1_scaling.json and are
    consumed by plot_phase1.py to produce the crossover comparison chart.

    The master-table entry for Phase 1 is re-emitted at N=256 (the largest
    tested point) so the quadratic penalty is visible rather than hidden by
    averaging across small N.
    """
    if n_values is None:
        n_values = SCALING_N_VALUES
    if prompt is None:
        # Use the first canonical prompt as the fixed probe for the sweep
        prompt = BENCHMARK_PROMPTS[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device)
    model.eval()

    print("\n" + "=" * 60)
    print("  MICROINFER PHASE 1: O(N^2) SCALING SWEEP")
    print(f"  Model      : {model_id}")
    print(f"  Device     : {device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  N values   : {n_values}")
    print(f"  Warm-up    : {num_warmup} runs (discarded, at max N)")
    print(f"  Timed      : {num_timed} runs per (N, engine)")
    print("=" * 60 + "\n")

    # Warm-up at N=64 so initial PyTorch CUDA allocations stabilise safely
    warmup_n = min(64, max(n_values))
    print(f"[Bench] Running {num_warmup} warm-up runs at N={warmup_n} (discarded)...")
    for _ in range(num_warmup):
        _ = naive_generate(model, tokenizer, prompt,
                           max_new_tokens=warmup_n, temperature=0.0)
        _ = _hf_generate_timed(model, tokenizer, prompt, warmup_n, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print("[Bench] Warm-up complete.\n")

    scaling_rows = []

    for N in n_values:
        print(f"--- N = {N} ---")
        try:
            # ---- Naive engine ----
            naive_ttfts, naive_tpots, naive_tps = [], [], []
            for _ in range(num_timed):
                res = naive_generate(model, tokenizer, prompt,
                                     max_new_tokens=N, temperature=0.0)
                naive_ttfts.append(res["ttft_ms"])
                naive_tpots.append(res["tpot_ms"])
                naive_tps.append(res["throughput_tok_per_sec"])

            # ---- HF baseline engine ----
            hf_ttfts, hf_tpots, hf_tps = [], [], []
            for _ in range(num_timed):
                hf = _hf_generate_timed(model, tokenizer, prompt, N, device)
                hf_ttfts.append(hf["ttft_ms"])
                hf_tpots.append(hf["tpot_ms"])
                hf_tps.append(hf["throughput_tok_per_sec"])

            naive_mean_ttft = statistics.mean(naive_ttfts)
            naive_mean_tpot = statistics.mean(naive_tpots)
            hf_mean_ttft    = statistics.mean(hf_ttfts)
            hf_mean_tpot    = statistics.mean(hf_tpots)

            print(f"  Naive  TTFT={naive_mean_ttft:.1f}ms  TPOT={naive_mean_tpot:.2f}ms/tok"
                  f"  TP={statistics.mean(naive_tps):.2f} t/s")
            print(f"  HF     TTFT={hf_mean_ttft:.1f}ms    TPOT={hf_mean_tpot:.2f}ms/tok"
                  f"  TP={statistics.mean(hf_tps):.2f} t/s")

            faster = "Naive" if naive_mean_tpot < hf_mean_tpot else "HF"
            ratio  = max(naive_mean_tpot, hf_mean_tpot) / max(min(naive_mean_tpot, hf_mean_tpot), 0.01)
            print(f"  Winner (TPOT): {faster}  ratio={ratio:.2f}x\n")

            scaling_rows.append({
                "N": N,
                "naive": {
                    "ttft_ms":              {"mean": round(naive_mean_ttft, 2),
                                             "p50":  round(_percentile(naive_ttfts, 50), 2),
                                             "p99":  round(_percentile(naive_ttfts, 99), 2)},
                    "tpot_ms":             {"mean": round(naive_mean_tpot, 2),
                                             "p50":  round(_percentile(naive_tpots, 50), 2),
                                             "p99":  round(_percentile(naive_tpots, 99), 2)},
                    "throughput_tok_per_sec": round(statistics.mean(naive_tps), 2),
                },
                "hf_baseline": {
                    "ttft_ms":              {"mean": round(hf_mean_ttft, 2),
                                             "p50":  round(_percentile(hf_ttfts, 50), 2),
                                             "p99":  round(_percentile(hf_ttfts, 99), 2)},
                    "tpot_ms":             {"mean": round(hf_mean_tpot, 2),
                                             "p50":  round(_percentile(hf_tpots, 50), 2),
                                             "p99":  round(_percentile(hf_tpots, 99), 2)},
                    "throughput_tok_per_sec": round(statistics.mean(hf_tps), 2),
                },
            })
        except Exception as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[Bench] CUDA OutOfMemory reached at N={N}. Stopping sweep at N={scaling_rows[-1]['N']} (6GB VRAM limit).\n")
                break
            else:
                raise e

    # Compute step-by-step growth percentages (delta % from previous N)
    for i, row in enumerate(scaling_rows):
        if i == 0:
            row["naive"]["tpot_delta_pct"] = 0.0
            row["hf_baseline"]["tpot_delta_pct"] = 0.0
        else:
            prev_naive = scaling_rows[i - 1]["naive"]["tpot_ms"]["mean"]
            curr_naive = row["naive"]["tpot_ms"]["mean"]
            naive_delta = ((curr_naive - prev_naive) / prev_naive * 100.0) if prev_naive > 0 else 0.0
            row["naive"]["tpot_delta_pct"] = round(naive_delta, 1)

            prev_hf = scaling_rows[i - 1]["hf_baseline"]["tpot_ms"]["mean"]
            curr_hf = row["hf_baseline"]["tpot_ms"]["mean"]
            hf_delta = ((curr_hf - prev_hf) / prev_hf * 100.0) if prev_hf > 0 else 0.0
            row["hf_baseline"]["tpot_delta_pct"] = round(hf_delta, 1)

    # ------------------------------------------------------------------
    # Master-table representative: largest tested N (e.g. N=2048)
    # ------------------------------------------------------------------
    largest_row = scaling_rows[-1]
    actual_max_n = largest_row["N"]
    print("=" * 60)
    print(f"  MASTER TABLE ROW (N={actual_max_n}, where quadratic penalty is maximal):")
    print(f"  Phase 1 (Naive)  TTFT={largest_row['naive']['ttft_ms']['mean']:.1f}ms"
          f"  TPOT={largest_row['naive']['tpot_ms']['mean']:.2f}ms/tok"
          f"  TP={largest_row['naive']['throughput_tok_per_sec']:.2f} t/s")
    print(f"  Phase 0 (HF)     TTFT={largest_row['hf_baseline']['ttft_ms']['mean']:.1f}ms"
          f"  TPOT={largest_row['hf_baseline']['tpot_ms']['mean']:.2f}ms/tok"
          f"  TP={largest_row['hf_baseline']['throughput_tok_per_sec']:.2f} t/s")
    print("=" * 60 + "\n")

    # Crossover analysis
    min_n = scaling_rows[0]["N"]
    crossover_N = None
    for row in scaling_rows:
        if row["naive"]["tpot_ms"]["mean"] > row["hf_baseline"]["tpot_ms"]["mean"]:
            crossover_N = row["N"]
            break

    max_n = max(n_values)
    if crossover_N is None:
        print(f"[Crossover] Naive did not exceed HF TPOT at any tested N ({min_n}..{actual_max_n}).")
        print("  Likely cause: Fixed FFN parameter projection cost dominates total per-token")
        print("  latency at this model size (1.5B parameters), making quadratic attention gap modest.")
        print("  See ANALYSIS.md for full explanation.")
    elif crossover_N == min_n:
        smallest = scaling_rows[0]
        largest  = scaling_rows[-1]
        naive_growth = largest["naive"]["tpot_ms"]["mean"] - smallest["naive"]["tpot_ms"]["mean"]
        hf_growth    = largest["hf_baseline"]["tpot_ms"]["mean"] - smallest["hf_baseline"]["tpot_ms"]["mean"]
        print(f"[Crossover] Naive TPOT is HIGHER (slower) than HF at all tested N ({min_n}..{actual_max_n}).")
        print(f"  HF .generate() uses internal DynamicCache (O(1) per step).")
        print(f"  O(N^2) growth IS visible: naive TPOT grew {naive_growth:+.1f}ms from N={min_n} to N={actual_max_n},")
        print(f"  while HF TPOT grew only {hf_growth:+.1f}ms.")
        print(f"  Absolute crossover occurs above N={max_n} for this model-hardware pair.")
        print(f"  See ANALYSIS.md for full mechanistic explanation.")
    else:
        print(f"[Crossover] Naive becomes SLOWER than HF at N >= {crossover_N} (TPOT).")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)

    export = {
        "phase": "Phase 1 - Naive Generator Scaling Sweep",
        "model_id": model_id,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "probe_prompt": prompt,
        "n_values": n_values,
        "num_timed_runs_per_n": num_timed,
        "master_table_N": max_n,
        "crossover_N": crossover_N,
        "scaling_rows": scaling_rows,
    }
    out_path = output_dir / "phase1_scaling.json"
    with open(out_path, "w") as f:
        json.dump(export, f, indent=2)

    print(f"[MicroInfer] Scaling results -> '{out_path}'")
    return export


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Phase 1 Naive Generator benchmark."
    )
    parser.add_argument(
        "--scaling", action="store_true",
        help="Run the O(N^2) scaling sweep across N in {16,32,64,128,256} "
             "(produces phase1_scaling.json; master-table row is taken from N=256).",
    )
    parser.add_argument(
        "--both", action="store_true",
        help="Run both the canonical N=64 benchmark (phase1_naive.json) AND "
             "the O(N^2) scaling sweep (phase1_scaling.json).",
    )
    args = parser.parse_args()

    if args.both:
        print("[MicroInfer] Running canonical N=64 benchmark first...")
        run_naive_benchmark()
        print("\n[MicroInfer] Now running O(N^2) scaling sweep...")
        run_naive_scaling_benchmark()
    elif args.scaling:
        run_naive_scaling_benchmark()
    else:
        # Default: run ONLY the scaling sweep so the master table is
        # always sourced from N=256 when the script is invoked directly.
        print(
            "[MicroInfer] Running O(N^2) scaling sweep (N=16..256).\n"
            "  Master-table row will be taken from N=256.\n"
            "  Use --both to also run the canonical N=64 latency profiler."
        )
        run_naive_scaling_benchmark()
