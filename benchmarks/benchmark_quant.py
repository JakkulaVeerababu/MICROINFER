"""
MicroInfer - Phase 4: INT8 Quantized Model Benchmark Harness

Canonical conditions shared with all other phases:
  - Same 3 prompts, same max_new_tokens=64
  - 3 discarded warm-up runs before timing starts
  - 10 timed runs per prompt; reports mean, p50, p99
  - Raw per-run data logged to benchmarks/results/phase4_raw.json

Phase 4 uses bitsandbytes INT8 weight quantization loaded via
src/quant_loader.py. The expected result is LOWER VRAM (roughly 50%
of FP16 weight footprint) at the cost of slightly higher TPOT due to
de-quantization overhead per matmul. Throughput may be similar to or
slightly below Phase 2 because quantized GEMM on consumer GPUs is not
always faster than cuBLAS FP16 on small batch sizes.
"""

import sys
import json
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import DEFAULT_MODEL_ID
from src.quant_loader import load_quantized_model_and_tokenizer
from src.quant_generate import quant_generate
from benchmarks.quant_accuracy import run_accuracy_eval
from benchmarks.bench_stats import compute_stats, flag_outliers

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



def run_quant_benchmark(
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = MAX_NEW_TOKENS,
    num_warmup: int = NUM_WARMUP_RUNS,
    num_timed: int = NUM_TIMED_RUNS,
    num_runs: int = None,   # backward-compat alias for num_timed
):
    if num_runs is not None:
        num_timed = num_runs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_quantized_model_and_tokenizer(model_id=model_id, device=device)
    model.eval()

    print("\n" + "=" * 60)
    print("  MICROINFER PHASE 4: INT8 QUANTIZED ENGINE BENCHMARK")
    print(f"  Model      : {model_id}")
    print(f"  Device     : {device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Max Tokens : {max_new_tokens}")
    print(f"  Warm-up    : {num_warmup} discarded runs")
    print(f"  Timed      : {num_timed} runs  ->  mean / p50 / p99")
    print("=" * 60 + "\n")

    print(f"[Bench] Running {num_warmup} warm-up runs (discarded)...")
    for _ in range(num_warmup):
        _ = quant_generate(
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

        run_ttfts       = []
        run_tpots       = []
        run_throughputs = []
        step_times_matrix = []
        sample_output   = ""

        for run in range(num_timed):
            res = quant_generate(
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

        ttft_stats = compute_stats(run_ttfts)
        tpot_stats = compute_stats(run_tpots)
        tp_stats   = compute_stats(run_throughputs)

        flag_outliers(run_ttfts, "TTFT (ms)")
        flag_outliers(run_tpots, "TPOT (ms)")
        flag_outliers(run_throughputs, "Throughput (t/s)")

        print(f"  Input Tokens : {res['prompt_tokens']}")
        print(f"  Output Tokens: {res['generated_tokens']}")
        print(f"  TTFT (Prefill) mean={ttft_stats['mean']:.1f}ms ± {ttft_stats['std']:.1f}ms  p50={ttft_stats['p50']:.1f}ms  p99={ttft_stats['p99']:.1f}ms")
        print(f"  TPOT (Decode)  mean={tpot_stats['mean']:.2f}ms ± {tpot_stats['std']:.2f}ms  p50={tpot_stats['p50']:.2f}ms  p99={tpot_stats['p99']:.2f}ms")
        print(f"  Throughput     mean={tp_stats['mean']:.2f} ± {tp_stats['std']:.2f} t/s  p50={tp_stats['p50']:.2f} t/s  p99={tp_stats['p99']:.2f} t/s\n")

        results.append({
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "input_tokens": res["prompt_tokens"],
            "output_tokens": res["generated_tokens"],
            "ttft_ms":  ttft_stats,
            "tpot_ms":  tpot_stats,
            "throughput_tok_per_sec": tp_stats,
            "per_step_latency_ms": avg_step_times,
        })

    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak VRAM (INT8 weights): {peak_vram_gb:.2f} GB")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)

    export_data = {
        "phase": "Phase 4 - INT8 Quantized Engine",
        "model_id": model_id,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "max_new_tokens": max_new_tokens,
        "num_warmup_runs": num_warmup,
        "num_timed_runs": num_timed,
        "peak_vram_gb": round(peak_vram_gb, 2),
        "results": results,
    }
    with open(output_dir / "phase4_quant.json", "w") as f:
        json.dump(export_data, f, indent=2)

    raw_export = {
        "phase": "Phase 4 - INT8 Quantized Engine",
        "model_id": model_id,
        "max_new_tokens": max_new_tokens,
        "num_warmup_runs": num_warmup,
        "raw_runs": all_raw_runs,
    }
    with open(output_dir / "phase4_raw.json", "w") as f:
        json.dump(raw_export, f, indent=2)

    print(f"\n[MicroInfer] Phase 4 results -> '{output_dir / 'phase4_quant.json'}'")
    print(f"[MicroInfer] Phase 4 raw log -> '{output_dir / 'phase4_raw.json'}'")

    # --- Accuracy evaluation (perplexity) ---
    print("\n" + "-" * 60)
    print("[MicroInfer] Running Phase 4 accuracy evaluation (perplexity)...")
    print("-" * 60)
    accuracy_result = run_accuracy_eval(model_id=model_id)
    export_data["accuracy"] = accuracy_result
    # Update phase4_quant.json with accuracy data
    with open(output_dir / "phase4_quant.json", "w") as f:
        json.dump(export_data, f, indent=2)
    print("[MicroInfer] Phase 4 accuracy merged into phase4_quant.json")

    return export_data


if __name__ == "__main__":
    run_quant_benchmark()
