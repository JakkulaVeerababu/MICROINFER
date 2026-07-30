"""
MicroInfer - Phase 3: Continuous Batching Scheduler Benchmark Harness

Canonical conditions shared with all other phases:
  - Same 3 prompt templates, same max_new_tokens=64
  - 3 discarded warm-up passes before timing starts
  - Concurrency load: 16 simultaneous requests fired per wave, 3 waves
    (48 total requests) so the scheduler must manage a real queue
  - Reports aggregate throughput (tokens/sec across all concurrent requests),
    mean/p50/p99 per-request latency, and mean TTFT
  - Raw per-request data logged to benchmarks/results/phase3_raw.json

WHY CONCURRENCY MATTERS HERE:
  The scheduler's value is invisible when requests are serialised.  A batch
  size of 1 is identical to Phase 2.  This harness fires CONCURRENT_REQUESTS
  requests into the waiting queue simultaneously; the scheduler then drains
  them with whatever degree of in-flight parallelism its step() loop allows.
  If Phase 2 still beats Phase 3 after this fix, that result is reported
  honestly -- it means our scheduler's sequential step() loop does not yet
  achieve true batched-tensor execution.
"""

import sys
import json
import time
import statistics
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_model_and_tokenizer, DEFAULT_MODEL_ID
from src.scheduler import ContinuousBatchScheduler

# ---------------------------------------------------------------------------
# Canonical benchmark constants -- identical across all six phase harnesses
# ---------------------------------------------------------------------------
BENCHMARK_PROMPTS = [
    "Explain how transformer attention mechanism works in simple terms for a software engineer.",
    "Write a Python function to implement quicksort with step-by-step explanations.",
    "What are the key trade-offs between KV-caching, continuous batching, and weight quantization in LLM serving?",
]
MAX_NEW_TOKENS     = 64
NUM_WARMUP_RUNS    = 3      # scheduler drain cycles to discard before timing
CONCURRENT_REQUESTS = 16   # simultaneous requests per wave
NUM_WAVES          = 10     # independent load waves to average over


def _percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    idx = (p / 100.0) * (len(data_sorted) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(data_sorted) - 1)
    frac = idx - lo
    return data_sorted[lo] * (1 - frac) + data_sorted[hi] * frac


def _run_wave(model, tokenizer, device: str, n_requests: int, max_new_tokens: int) -> dict:
    """
    Fire n_requests simultaneously into a fresh scheduler, drain them to
    completion, and return aggregate statistics for that wave.
    """
    scheduler = ContinuousBatchScheduler(max_batch_size=n_requests, device=device)

    # Distribute requests across the 3 canonical prompts in round-robin
    for i in range(n_requests):
        prompt = BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)]
        scheduler.add_request(prompt, tokenizer, max_new_tokens=max_new_tokens, temperature=0.0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_wave_start = time.perf_counter()
    step_count = 0
    while scheduler.has_pending_work():
        scheduler.step(model, tokenizer)
        step_count += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_wave_end = time.perf_counter()

    completed = scheduler.finished_sequences
    total_tokens = sum(len(seq.generated_tokens) for seq in completed)
    wall_time_s  = t_wave_end - t_wave_start
    aggregate_tp = total_tokens / wall_time_s if wall_time_s > 0 else 0.0

    per_req_latencies = [seq.total_time_ms for seq in completed]
    ttfts             = [seq.ttft_ms for seq in completed]

    return {
        "n_completed":          len(completed),
        "total_tokens":         total_tokens,
        "wall_time_s":          wall_time_s,
        "aggregate_throughput": aggregate_tp,
        "step_count":           step_count,
        "mean_ttft_ms":         statistics.mean(ttfts) if ttfts else 0.0,
        "p50_ttft_ms":          _percentile(ttfts, 50),
        "p99_ttft_ms":          _percentile(ttfts, 99),
        "mean_latency_ms":      statistics.mean(per_req_latencies) if per_req_latencies else 0.0,
        "p50_latency_ms":       _percentile(per_req_latencies, 50),
        "p99_latency_ms":       _percentile(per_req_latencies, 99),
        "requests": [
            {
                "seq_id":           seq.seq_id,
                "prompt":           seq.prompt,
                "input_tokens":     len(seq.prompt_tokens),
                "generated_tokens": len(seq.generated_tokens),
                "ttft_ms":          round(seq.ttft_ms, 2),
                "total_latency_ms": round(seq.total_time_ms, 2),
            }
            for seq in completed
        ],
    }


def run_scheduler_benchmark(
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = MAX_NEW_TOKENS,
    concurrent_requests: int = CONCURRENT_REQUESTS,
    num_warmup: int = NUM_WARMUP_RUNS,
    num_waves: int = NUM_WAVES,
    max_batch_size: int = None,  # backward-compat alias for concurrent_requests
):
    # Resolve backward-compat alias
    if max_batch_size is not None:
        concurrent_requests = max_batch_size
        num_warmup = 1   # keep tests fast
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device)
    model.eval()

    print("\n" + "=" * 60)
    print("  MICROINFER PHASE 3: CONTINUOUS BATCHING SCHEDULER BENCHMARK")
    print(f"  Model               : {model_id}")
    print(f"  Device              : {device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Max Tokens          : {max_new_tokens}")
    print(f"  Concurrent Requests : {concurrent_requests} per wave")
    print(f"  Warm-up waves       : {num_warmup} discarded")
    print(f"  Timed waves         : {num_waves}")
    print("=" * 60 + "\n")
    print("NOTE: The scheduler drains requests sequentially inside step().")
    print("      Aggregate throughput here measures scheduling efficiency")
    print("      (queue management + lifecycle), NOT batched-tensor execution.")
    print("      If Phase 2 single-request throughput exceeds this, that is")
    print("      expected and honest -- it is reported as-is.\n")

    # -----------------------------------------------------------------------
    # Warm-up waves: drain the scheduler N times to stabilise CUDA memory
    # -----------------------------------------------------------------------
    print(f"[Bench] Running {num_warmup} warm-up waves (discarded)...")
    for _ in range(num_warmup):
        _run_wave(model, tokenizer, device, n_requests=min(4, concurrent_requests),
                  max_new_tokens=max_new_tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print("[Bench] Warm-up complete.\n")

    wave_results = []
    all_raw_requests = []

    for wave_idx in range(num_waves):
        print(f"[Wave {wave_idx + 1}/{num_waves}] Firing {concurrent_requests} simultaneous requests...")
        wave = _run_wave(model, tokenizer, device,
                         n_requests=concurrent_requests,
                         max_new_tokens=max_new_tokens)
        wave_results.append(wave)
        for req in wave["requests"]:
            req["wave"] = wave_idx + 1
            all_raw_requests.append(req)

        print(f"  Completed     : {wave['n_completed']} requests")
        print(f"  Total tokens  : {wave['total_tokens']}")
        print(f"  Wall time     : {wave['wall_time_s']:.2f} s")
        print(f"  Aggregate tp  : {wave['aggregate_throughput']:.2f} tokens/sec")
        print(f"  TTFT          : mean={wave['mean_ttft_ms']:.1f}ms  "
              f"p50={wave['p50_ttft_ms']:.1f}ms  p99={wave['p99_ttft_ms']:.1f}ms")
        print(f"  Req latency   : mean={wave['mean_latency_ms']:.1f}ms  "
              f"p50={wave['p50_latency_ms']:.1f}ms  p99={wave['p99_latency_ms']:.1f}ms\n")

    # -----------------------------------------------------------------------
    # Aggregate across all timed waves
    # -----------------------------------------------------------------------
    all_throughputs = [w["aggregate_throughput"] for w in wave_results]
    all_ttfts       = [w["mean_ttft_ms"]         for w in wave_results]
    all_latencies   = [w["mean_latency_ms"]       for w in wave_results]

    mean_tp  = statistics.mean(all_throughputs)
    p50_tp   = _percentile(all_throughputs, 50)
    p99_tp   = _percentile(all_throughputs, 99)
    mean_ttft = statistics.mean(all_ttfts)
    mean_lat  = statistics.mean(all_latencies)
    p50_lat   = _percentile(all_latencies, 50)
    p99_lat   = _percentile(all_latencies, 99)

    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print("--- Aggregate across all waves ---")
    print(f"  Aggregate Throughput  mean={mean_tp:.2f} t/s  p50={p50_tp:.2f} t/s  p99={p99_tp:.2f} t/s")
    print(f"  Mean TTFT             {mean_ttft:.1f} ms")
    print(f"  Request Latency       mean={mean_lat:.1f}ms  p50={p50_lat:.1f}ms  p99={p99_lat:.1f}ms")
    print(f"  Peak VRAM             {peak_vram_gb:.2f} GB")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)

    export_data = {
        "phase": "Phase 3 - Dynamic Request Scheduler with Lifecycle Management",
        "model_id": model_id,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "max_new_tokens": max_new_tokens,
        "concurrent_requests_per_wave": concurrent_requests,
        "num_warmup_waves": num_warmup,
        "num_timed_waves": num_waves,
        "peak_vram_gb": round(peak_vram_gb, 2),
        "honest_note": (
            "Scheduler manages request queue lifecycles (WAITING -> RUNNING -> FINISHED). "
            "Active sequences are stepped in an iteration loop. "
            "True tensor-level batching across concurrent sequences requires "
            "restructuring the forward pass to stack (B, T) inputs — this is a documented next step."
        ),
        "aggregate": {
            "throughput_tok_per_sec": {"mean": round(mean_tp,2), "p50": round(p50_tp,2), "p99": round(p99_tp,2)},
            "ttft_ms":                {"mean": round(mean_ttft,2)},
            "request_latency_ms":     {"mean": round(mean_lat,2), "p50": round(p50_lat,2), "p99": round(p99_lat,2)},
        },
        "wave_results": [
            {
                "wave": i + 1,
                "n_completed":          w["n_completed"],
                "total_tokens":         w["total_tokens"],
                "wall_time_s":          round(w["wall_time_s"], 3),
                "aggregate_throughput": round(w["aggregate_throughput"], 2),
                "mean_ttft_ms":         round(w["mean_ttft_ms"], 2),
                "p50_ttft_ms":          round(w["p50_ttft_ms"], 2),
                "p99_ttft_ms":          round(w["p99_ttft_ms"], 2),
                "mean_latency_ms":      round(w["mean_latency_ms"], 2),
                "p50_latency_ms":       round(w["p50_latency_ms"], 2),
                "p99_latency_ms":       round(w["p99_latency_ms"], 2),
            }
            for i, w in enumerate(wave_results)
        ],
    }
    with open(output_dir / "phase3_scheduler.json", "w") as f:
        json.dump(export_data, f, indent=2)

    raw_export = {
        "phase": "Phase 3 - Dynamic Request Scheduler with Lifecycle Management",
        "model_id": model_id,
        "max_new_tokens": max_new_tokens,
        "concurrent_requests_per_wave": concurrent_requests,
        "num_warmup_waves": num_warmup,
        "raw_requests": all_raw_requests,
    }
    with open(output_dir / "phase3_raw.json", "w") as f:
        json.dump(raw_export, f, indent=2)

    print(f"\n[MicroInfer] Phase 3 results -> '{output_dir / 'phase3_scheduler.json'}'")
    print(f"[MicroInfer] Phase 3 raw log -> '{output_dir / 'phase3_raw.json'}'")
    return export_data


if __name__ == "__main__":
    run_scheduler_benchmark()
