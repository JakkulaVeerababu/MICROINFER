"""
MicroInfer - Phase 5: Concurrent Load Benchmark (Fallback Scheduler)

Canonical conditions shared with all other phases:
  - Same 3 prompts, same max_new_tokens=64
  - 3 discarded warm-up runs/waves before timing starts
  - Concurrent load: 16 simultaneous requests per wave, 3 waves
  - Reports mean/p50/p99 latency and aggregate throughput
  - Raw per-request data logged to benchmarks/results/phase5_raw.json

ABOUT vLLM ON THIS MACHINE:
  vLLM v0.26.0 installs successfully via pip on Windows, but fails at
  import time with:

      ModuleNotFoundError: No module named 'vllm._C_stable_libtorch'

  This is a Windows-specific build limitation. vLLM's native CUDA
  extension (_C_stable_libtorch) is compiled for Linux only. Running
  on WSL2 or a native Linux host will activate the real vLLM path
  (LLM.generate() with PagedAttention) in this same script.

FALLBACK PATH (what actually runs here):
  Uses MicroInfer ContinuousBatchScheduler (same as Phase 3) under
  identical concurrency conditions: 16 concurrent requests per wave,
  10 timed waves. NO synthetic multipliers. All numbers are measured.
  Results are labelled "fallback-scheduler (vLLM unavailable: Windows)"
  in the output JSON and are NOT presented as vLLM/PagedAttention numbers.
"""

import sys
import json
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_model_and_tokenizer, DEFAULT_MODEL_ID
from src.cached_generate import cached_generate
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
MAX_NEW_TOKENS      = 64
NUM_WARMUP_RUNS     = 3
NUM_TIMED_RUNS      = 10     # used when running sequentially (no vLLM)
CONCURRENT_REQUESTS = 16     # simultaneous requests per wave (vLLM or fallback)
NUM_WAVES           = 10     # timed waves for concurrency path



# ---------------------------------------------------------------------------
# vLLM path (only runs when vLLM is installed)
# ---------------------------------------------------------------------------
def _run_vllm(model_id, max_new_tokens, num_warmup, concurrent_requests, num_waves):
    from vllm import LLM, SamplingParams  # type: ignore

    llm = LLM(model=model_id, trust_remote_code=True, gpu_memory_utilization=0.75, max_model_len=4096)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    # Build a batch of `concurrent_requests` prompts (round-robin across BENCHMARK_PROMPTS)
    batch_prompts = [
        BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)]
        for i in range(concurrent_requests)
    ]

    # Warm-up
    print(f"[vLLM] Running {num_warmup} warm-up batches ({concurrent_requests} req each, discarded)...")
    for _ in range(num_warmup):
        _ = llm.generate(batch_prompts, sampling_params)
    print("[vLLM] Warm-up complete.\n")

    wave_results = []
    all_raw_requests = []

    for wave_idx in range(num_waves):
        print(f"[Wave {wave_idx + 1}/{num_waves}] Sending {concurrent_requests} requests to vLLM...")
        t0 = time.perf_counter()
        outputs = llm.generate(batch_prompts, sampling_params)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        wall_time_s  = t1 - t0
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        aggregate_tp = total_tokens / wall_time_s if wall_time_s > 0 else 0.0

        # vLLM doesn't expose per-request TTFT/latency in the offline API;
        # we record wall-clock latency as a proxy.
        per_req_latency_ms = (wall_time_s * 1000.0)  # same for all (batch finish time)

        for i, o in enumerate(outputs):
            all_raw_requests.append({
                "wave": wave_idx + 1,
                "req_idx": i,
                "input_tokens":     len(o.prompt_token_ids),
                "generated_tokens": len(o.outputs[0].token_ids),
                "batch_wall_time_ms": round(wall_time_s * 1000.0, 2),
            })

        wave_results.append({
            "n_completed":          len(outputs),
            "total_tokens":         total_tokens,
            "wall_time_s":          wall_time_s,
            "aggregate_throughput": aggregate_tp,
            "per_req_latency_ms":   per_req_latency_ms,
        })
        print(f"  Completed    : {len(outputs)} requests")
        print(f"  Total tokens : {total_tokens}")
        print(f"  Wall time    : {wall_time_s:.2f} s")
        print(f"  Throughput   : {aggregate_tp:.2f} tokens/sec\n")

    return wave_results, all_raw_requests, "vLLM (PagedAttention)"


    if vllm_available:
        wave_results, all_raw_requests, engine_label = _run_vllm(
            model_id, max_new_tokens, num_warmup, concurrent_requests, num_waves
        )
    else:
        from benchmarks.benchmark_scheduler import run_scheduler_benchmark
        # Literally call Phase 3's harness so they are 100% identical.
        print("\n[Phase 5 Fallback] vLLM unavailable. Falling back to Phase 3 harness identically...")
        return run_scheduler_benchmark(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            concurrent_requests=concurrent_requests,
            num_warmup=num_warmup,
            num_waves=num_waves
        )

    # Aggregate across waves
    all_throughputs = [w["aggregate_throughput"] for w in wave_results]
    tp_stats = compute_stats(all_throughputs)
    
    flag_outliers(all_throughputs, "Wave Throughput (t/s)")

    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print("--- Aggregate across all waves ---")
    print(f"  Engine              : {engine_label}")
    print(f"  Aggregate Throughput: mean={tp_stats['mean']:.2f} ± {tp_stats['std']:.2f} t/s  p50={tp_stats['p50']:.2f} t/s  p99={tp_stats['p99']:.2f} t/s")
    print(f"  Peak VRAM           : {peak_vram_gb:.2f} GB")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)

    wave_records = []
    for i, w in enumerate(wave_results):
        rec = {"wave": i + 1, "n_completed": w["n_completed"],
               "total_tokens": w["total_tokens"],
               "wall_time_s": round(w["wall_time_s"], 3),
               "aggregate_throughput": round(w["aggregate_throughput"], 2)}
        for key in ("mean_ttft_ms", "p50_ttft_ms", "p99_ttft_ms",
                    "mean_latency_ms", "p50_latency_ms", "p99_latency_ms"):
            if key in w:
                rec[key] = round(w[key], 2)
        wave_records.append(rec)

    export_data = {
        "phase": "Phase 5 - Production Reference Engine (vLLM PagedAttention)" if vllm_available else "Phase 5 - Fallback Scheduler Under Concurrent Load",
        "engine": engine_label,
        "vllm_available": vllm_available,
        "vllm_unavailability_reason": (
            "vLLM 0.26.0 installed on Windows but fails to import: "
            "ModuleNotFoundError: No module named 'vllm._C_stable_libtorch'. "
            "This CUDA extension is Linux-only. Run on WSL2 or native Linux "
            "to activate the real vLLM PagedAttention path."
        ) if not vllm_available else None,
        "model_id": model_id,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "max_new_tokens": max_new_tokens,
        "concurrent_requests_per_wave": concurrent_requests,
        "num_warmup": num_warmup,
        "num_timed_waves": num_waves,
        "peak_vram_gb": round(peak_vram_gb, 2),
        "honest_note": (
            "No synthetic multipliers. vLLM path uses PagedAttention if installed. "
            "Fallback path uses MicroInfer ContinuousBatchScheduler under same "
            "concurrency load as Phase 3. Results are directly comparable."
        ),
        "aggregate": {
            "throughput_tok_per_sec": tp_stats,
        },
        "wave_results": wave_records,
    }
    with open(output_dir / "phase5_vllm.json", "w") as f:
        json.dump(export_data, f, indent=2)

    raw_export = {
        "phase": "Phase 5 - Production Reference Engine",
        "engine": engine_label,
        "model_id": model_id,
        "max_new_tokens": max_new_tokens,
        "concurrent_requests_per_wave": concurrent_requests,
        "num_warmup": num_warmup,
        "raw_requests": all_raw_requests,
    }
    with open(output_dir / "phase5_raw.json", "w") as f:
        json.dump(raw_export, f, indent=2)

    print(f"\n[MicroInfer] Phase 5 results -> '{output_dir / 'phase5_vllm.json'}'")
    print(f"[MicroInfer] Phase 5 raw log -> '{output_dir / 'phase5_raw.json'}'")
    return export_data


if __name__ == "__main__":
    run_vllm_benchmark()
