"""
MicroInfer - Phase 0: HuggingFace Baseline Benchmark Harness

Canonical conditions shared with all other phases:
  - Same 3 prompts, same max_new_tokens=64
  - 3 discarded warm-up runs before timing starts
  - 10 timed runs per prompt; reports mean, p50, p99
  - Raw per-run data logged to benchmarks/results/phase0_raw.json
"""

import os
import sys
import json
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_model_and_tokenizer, DEFAULT_MODEL_ID
from benchmarks.bench_stats import compute_stats, flag_outliers

# ---------------------------------------------------------------------------
# Shared benchmark constants — identical across all six phase harnesses
# ---------------------------------------------------------------------------
BENCHMARK_PROMPTS = [
    "Explain how transformer attention mechanism works in simple terms for a software engineer.",
    "Write a Python function to implement quicksort with step-by-step explanations.",
    "What are the key trade-offs between KV-caching, continuous batching, and weight quantization in LLM serving?",
]
# Backward-compat alias: tests import TEST_PROMPTS
TEST_PROMPTS = BENCHMARK_PROMPTS
MAX_NEW_TOKENS = 64      # identical across all phases
NUM_WARMUP_RUNS = 3      # discarded before timing
NUM_TIMED_RUNS  = 10     # runs actually measured


def benchmark_hf_generate(
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = MAX_NEW_TOKENS,
    num_warmup: int = NUM_WARMUP_RUNS,
    num_timed: int = NUM_TIMED_RUNS,
    num_runs: int = None,   # backward-compat alias for num_timed
):
    # Resolve backward-compat alias
    if num_runs is not None:
        num_timed = num_runs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device)
    model.eval()

    print("\n" + "=" * 60)
    print("  MICROINFER PHASE 0: HUGGINGFACE BASELINE BENCHMARK")
    print(f"  Model      : {model_id}")
    print(f"  Device     : {device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Max Tokens : {max_new_tokens}")
    print(f"  Warm-up    : {num_warmup} discarded runs")
    print(f"  Timed      : {num_timed} runs  ->  mean / p50 / p99")
    print("=" * 60 + "\n")

    # -----------------------------------------------------------------------
    # Warm-up: run with a real prompt so CUDA graph capture, cuBLAS
    # workspace sizing, and memory allocation all stabilise.
    # -----------------------------------------------------------------------
    print(f"[Bench] Running {num_warmup} warm-up runs (discarded)...")
    wu_input = tokenizer(BENCHMARK_PROMPTS[0], return_tensors="pt").to(device)
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model.generate(**wu_input, max_new_tokens=max_new_tokens, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print("[Bench] Warm-up complete.\n")

    results = []
    all_raw_runs = []     # for raw per-run log

    for prompt_idx, prompt in enumerate(BENCHMARK_PROMPTS, 1):
        print(f"--- Prompt {prompt_idx}/{len(BENCHMARK_PROMPTS)} ---")
        prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_length  = prompt_inputs.input_ids.shape[1]

        run_ttfts       = []
        run_tpots       = []
        run_throughputs = []

        for run in range(num_timed):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # --- TTFT: time to first token ---
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model.generate(**prompt_inputs, max_new_tokens=1, min_new_tokens=1, do_sample=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ttft_ms = (time.perf_counter() - t0) * 1000.0

            # --- Full generation (TPOT + throughput) ---
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            with torch.no_grad():
                full_output = model.generate(
                    **prompt_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gen_time_s     = time.perf_counter() - t1
            gen_tokens     = full_output.shape[1] - input_length
            tpot_ms        = (gen_time_s / gen_tokens * 1000.0) if gen_tokens > 0 else 0.0
            throughput_tps = gen_tokens / gen_time_s if gen_time_s > 0 else 0.0

            run_ttfts.append(ttft_ms)
            run_tpots.append(tpot_ms)
            run_throughputs.append(throughput_tps)
            all_raw_runs.append({
                "prompt_idx": prompt_idx,
                "run": run + 1,
                "ttft_ms": round(ttft_ms, 3),
                "tpot_ms": round(tpot_ms, 3),
                "throughput_tok_per_sec": round(throughput_tps, 3),
            })

        ttft_stats = compute_stats(run_ttfts)
        tpot_stats = compute_stats(run_tpots)
        tp_stats   = compute_stats(run_throughputs)

        flag_outliers(run_ttfts, "TTFT (ms)")
        flag_outliers(run_tpots, "TPOT (ms)")
        flag_outliers(run_throughputs, "Throughput (t/s)")

        print(f"  Input Tokens : {input_length}")
        print(f"  Output Tokens: {max_new_tokens}")
        print(f"  TTFT          mean={ttft_stats['mean']:.1f}ms ± {ttft_stats['std']:.1f}ms  p50={ttft_stats['p50']:.1f}ms  p99={ttft_stats['p99']:.1f}ms")
        print(f"  TPOT          mean={tpot_stats['mean']:.2f}ms ± {tpot_stats['std']:.2f}ms  p50={tpot_stats['p50']:.2f}ms  p99={tpot_stats['p99']:.2f}ms")
        print(f"  Throughput    mean={tp_stats['mean']:.2f} ± {tp_stats['std']:.2f} t/s  p50={tp_stats['p50']:.2f} t/s  p99={tp_stats['p99']:.2f} t/s\n")

        results.append({
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "input_tokens": input_length,
            "output_tokens": max_new_tokens,
            "ttft_ms":  ttft_stats,
            "tpot_ms":  tpot_stats,
            "throughput_tok_per_sec": tp_stats,
        })

    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak VRAM: {peak_vram_gb:.2f} GB")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)

    export_data = {
        "phase": "Phase 0 - HuggingFace Baseline",
        "model_id": model_id,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "max_new_tokens": max_new_tokens,
        "num_warmup_runs": num_warmup,
        "num_timed_runs": num_timed,
        "peak_vram_gb": round(peak_vram_gb, 2),
        "results": results,
    }
    with open(output_dir / "phase0_baseline_hf.json", "w") as f:
        json.dump(export_data, f, indent=2)

    raw_export = {
        "phase": "Phase 0 - HuggingFace Baseline",
        "model_id": model_id,
        "max_new_tokens": max_new_tokens,
        "num_warmup_runs": num_warmup,
        "raw_runs": all_raw_runs,
    }
    with open(output_dir / "phase0_raw.json", "w") as f:
        json.dump(raw_export, f, indent=2)

    print(f"\n[MicroInfer] Phase 0 results -> '{output_dir / 'phase0_baseline_hf.json'}'")
    print(f"[MicroInfer] Phase 0 raw log -> '{output_dir / 'phase0_raw.json'}'")
    return export_data


if __name__ == "__main__":
    benchmark_hf_generate()
