"""
MicroInfer - Phase 2: KV-Cache Generator Benchmark Harness

Canonical conditions shared with all other phases:
  - Same 3 prompts, same max_new_tokens=64
  - 3 discarded warm-up runs before timing starts
  - 10 timed runs per prompt; reports mean, p50, p99
  - Raw per-run data logged to benchmarks/results/phase2_raw.json

What MicroInfer contributes (src/cached_generate.py):
  - Explicit two-phase Prefill/Decode separation with independent
    CUDA-synchronised TTFT and TPOT timers.
  - Manual DynamicCache lifecycle management (instantiate once per sequence,
    pass through each forward call) — equivalent to how a production serving
    engine manages its own cache object per sequence slot.
  - Structured per-step latency collection for downstream plotting.

What HuggingFace provides:
  - DynamicCache: the actual K/V tensor storage (a Python list of per-layer
    CUDA tensor pairs that grows one step at a time).
  - The attention kernel that reads those tensors during each forward call.

Each decoding step reuses cached K/V tensors from previous steps, so attention
cost is O(1) per new token rather than O(N^2). Per-step latency stays flat --
contrasting with the quadratic growth visible in Phase 1.
"""


import sys
import json
import statistics
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_model_and_tokenizer, DEFAULT_MODEL_ID
from src.cached_generate import cached_generate

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


def _percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    idx = (p / 100.0) * (len(data_sorted) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(data_sorted) - 1)
    frac = idx - lo
    return data_sorted[lo] * (1 - frac) + data_sorted[hi] * frac


def run_cached_benchmark(
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = MAX_NEW_TOKENS,
    num_warmup: int = NUM_WARMUP_RUNS,
    num_timed: int = NUM_TIMED_RUNS,
    num_runs: int = None,   # backward-compat alias for num_timed
):
    if num_runs is not None:
        num_timed = num_runs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device)
    model.eval()

    print("\n" + "=" * 60)
    print("  MICROINFER PHASE 2: KV-CACHE GENERATOR BENCHMARK")
    print(f"  Model      : {model_id}")
    print(f"  Device     : {device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Max Tokens : {max_new_tokens}")
    print(f"  Warm-up    : {num_warmup} discarded runs")
    print(f"  Timed      : {num_timed} runs  ->  mean / p50 / p99")
    print("=" * 60 + "\n")

    print(f"[Bench] Running {num_warmup} warm-up runs (discarded)...")
    for _ in range(num_warmup):
        _ = cached_generate(
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
            res = cached_generate(
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
        print(f"  TTFT (Prefill) mean={mean_ttft:.1f}ms  p50={p50_ttft:.1f}ms  p99={p99_ttft:.1f}ms")
        print(f"  TPOT (Decode)  mean={mean_tpot:.2f}ms  p50={p50_tpot:.2f}ms  p99={p99_tpot:.2f}ms")
        print(f"  Throughput     mean={mean_tp:.2f} t/s  p50={p50_tp:.2f} t/s  p99={p99_tp:.2f} t/s")
        print(f"  Step 1 (prefill):  {avg_step_times[0]:.2f} ms")
        if num_steps > 1:
            print(f"  Step {num_steps} (decode):   {avg_step_times[-1]:.2f} ms  <- should be flat vs Phase 1")
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
        "phase": "Phase 2 - KV-Cache Generator",
        "model_id": model_id,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "max_new_tokens": max_new_tokens,
        "num_warmup_runs": num_warmup,
        "num_timed_runs": num_timed,
        "peak_vram_gb": round(peak_vram_gb, 2),
        "results": results,
    }
    with open(output_dir / "phase2_cached.json", "w") as f:
        json.dump(export_data, f, indent=2)

    raw_export = {
        "phase": "Phase 2 - KV-Cache Generator",
        "model_id": model_id,
        "max_new_tokens": max_new_tokens,
        "num_warmup_runs": num_warmup,
        "raw_runs": all_raw_runs,
    }
    with open(output_dir / "phase2_raw.json", "w") as f:
        json.dump(raw_export, f, indent=2)

    print(f"\n[MicroInfer] Phase 2 results -> '{output_dir / 'phase2_cached.json'}'")
    print(f"[MicroInfer] Phase 2 raw log -> '{output_dir / 'phase2_raw.json'}'")
    return export_data


if __name__ == "__main__":
    run_cached_benchmark()
