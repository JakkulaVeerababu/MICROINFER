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
import argparse
import subprocess
import threading
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_model_and_tokenizer, DEFAULT_MODEL_ID
from src.scheduler import ContinuousBatchScheduler
from benchmarks.bench_stats import compute_stats, flag_outliers

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

    ttft_stats = compute_stats(ttfts)
    lat_stats = compute_stats(per_req_latencies)

    return {
        "n_completed":          len(completed),
        "total_tokens":         total_tokens,
        "wall_time_s":          wall_time_s,
        "aggregate_throughput": aggregate_tp,
        "step_count":           step_count,
        "mean_ttft_ms":         ttft_stats["mean"],
        "p50_ttft_ms":          ttft_stats["p50"],
        "p99_ttft_ms":          ttft_stats["p99"],
        "mean_latency_ms":      lat_stats["mean"],
        "p50_latency_ms":       lat_stats["p50"],
        "p99_latency_ms":       lat_stats["p99"],
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

    tp_stats = compute_stats(all_throughputs)
    ttft_stats = compute_stats(all_ttfts)
    lat_stats = compute_stats(all_latencies)

    flag_outliers(all_throughputs, "Wave Throughput (t/s)")
    flag_outliers(all_ttfts, "Wave Mean TTFT (ms)")
    flag_outliers(all_latencies, "Wave Mean Latency (ms)")

    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print("--- Aggregate across all waves ---")
    print(f"  Aggregate Throughput  mean={tp_stats['mean']:.2f} ± {tp_stats['std']:.2f} t/s  p50={tp_stats['p50']:.2f} t/s  p99={tp_stats['p99']:.2f} t/s")
    print(f"  Mean TTFT             mean={ttft_stats['mean']:.1f}ms ± {ttft_stats['std']:.1f}ms")
    print(f"  Request Latency       mean={lat_stats['mean']:.1f}ms ± {lat_stats['std']:.1f}ms  p50={lat_stats['p50']:.1f}ms  p99={lat_stats['p99']:.1f}ms")
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
            "throughput_tok_per_sec": tp_stats,
            "ttft_ms":                ttft_stats,
            "request_latency_ms":     lat_stats,
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


def sample_gpu_util(duration_sec: float, interval_sec: float = 1.0) -> list:
    samples = []
    t_end = time.perf_counter() + duration_sec
    while time.perf_counter() < t_end:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                val = result.stdout.strip()
                if val:
                    samples.append(float(val))
        except Exception:
            pass
        time.sleep(interval_sec)
    return samples

def run_staggered_arrival_benchmark(
    model_id: str = DEFAULT_MODEL_ID,
    max_batch_size: int = 16,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device)
    model.eval()

    print("\n" + "=" * 60)
    print("  PHASE 3: STAGGERED ARRIVAL BENCHMARK")
    print("=" * 60 + "\n")

    schedule = [
        (0.0, "What is the capital of France?", 16),
        (0.5, "Write a quicksort in python.", 64),
        (1.2, "Explain transformer attention.", 128),
        (1.5, "Translate hello to Spanish.", 8),
        (2.0, "Why is the sky blue? Explain in extreme detail.", 256),
        (2.5, "Tell me a short joke.", 32),
        (3.0, "Summarize the history of Rome.", 200),
        (3.5, "What is 2+2?", 4),
    ]

    scheduler = ContinuousBatchScheduler(max_batch_size=max_batch_size, device=device)
    
    t_start = time.perf_counter()
    next_req_idx = 0
    total_tokens = 0
    
    samples = []
    def sampler_thread():
        nonlocal samples
        samples = sample_gpu_util(duration_sec=20, interval_sec=0.5)
    
    t_bg = threading.Thread(target=sampler_thread)
    t_bg.start()

    while next_req_idx < len(schedule) or scheduler.has_pending_work():
        current_time = time.perf_counter() - t_start
        
        while next_req_idx < len(schedule) and current_time >= schedule[next_req_idx][0]:
            _, prompt, new_tokens = schedule[next_req_idx]
            scheduler.add_request(prompt, tokenizer, max_new_tokens=new_tokens, temperature=0.0)
            next_req_idx += 1
            
        if scheduler.has_pending_work():
            scheduler.step(model, tokenizer)
        else:
            time.sleep(0.01)

    wall_time = time.perf_counter() - t_start
    t_bg.join(timeout=1.0)
    
    completed = scheduler.finished_sequences
    total_tokens = sum(len(seq.generated_tokens) for seq in completed)
    throughput = total_tokens / wall_time if wall_time > 0 else 0
    avg_gpu = sum(samples) / len(samples) if samples else 0.0

    print(f"  Total Requests   : {len(completed)}")
    print(f"  Wall Time        : {wall_time:.2f} s")
    print(f"  Total Tokens     : {total_tokens}")
    print(f"  Throughput       : {throughput:.2f} tok/s")
    print(f"  Avg GPU Util     : {avg_gpu:.1f}%")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / "phase3_staggered.json", "w") as f:
        json.dump({
            "phase": "Phase 3 - Staggered Arrival",
            "throughput_tok_per_sec": round(throughput, 2),
            "avg_gpu_util_pct": round(avg_gpu, 2),
        }, f, indent=2)

def run_capacity_ceiling_benchmark(
    model_id: str = DEFAULT_MODEL_ID,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device)
    model.eval()

    print("\n" + "=" * 60)
    print("  PHASE 3: CAPACITY CEILING OOM BENCHMARK")
    print("=" * 60 + "\n")

    batch_size = 1
    prompt = BENCHMARK_PROMPTS[0]
    results = []
    
    while True:
        print(f"Testing Batch Size = {batch_size}...", flush=True)
        scheduler = ContinuousBatchScheduler(max_batch_size=batch_size, device=device)
        for _ in range(batch_size):
            scheduler.add_request(prompt, tokenizer, max_new_tokens=1, temperature=0.0)
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        try:
            # We only need 1 step to force admission and cache allocation
            if scheduler.has_pending_work():
                scheduler.step(model, tokenizer)
                
            peak_vram = 0
            if torch.cuda.is_available():
                peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            
            print(f"  -> Success! Peak VRAM: {peak_vram:.2f} GB", flush=True)
            results.append({"batch_size": batch_size, "peak_vram_gb": peak_vram})
            batch_size *= 2
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                print(f"  -> OOM Triggered at Batch Size {batch_size}!", flush=True)
                break
            else:
                raise
                
    if len(results) >= 2:
        last = results[-1]
        prev = results[-2]
        vram_diff = last["peak_vram_gb"] - prev["peak_vram_gb"]
        seq_diff = last["batch_size"] - prev["batch_size"]
        per_seq_mb = (vram_diff * 1024) / seq_diff if seq_diff > 0 else 0
        print(f"\nCapacity Ceiling: {last['batch_size']} concurrent sequences")
        print(f"Est. VRAM Cost per Sequence: {per_seq_mb:.2f} MB")
        
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True, parents=True)
        with open(output_dir / "phase3_capacity.json", "w") as f:
            json.dump({
                "phase": "Phase 3 - Capacity Ceiling",
                "max_safe_batch_size": last['batch_size'],
                "oom_batch_size": batch_size,
                "vram_per_sequence_mb": round(per_seq_mb, 2),
                "runs": results
            }, f, indent=2)
    else:
        print("Not enough successful runs to calculate delta.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--staggered", action="store_true", help="Run the staggered arrival benchmark")
    parser.add_argument("--capacity", action="store_true", help="Run the capacity ceiling benchmark")
    args = parser.parse_args()

    if not args.staggered and not args.capacity:
        run_scheduler_benchmark()
    else:
        if args.staggered:
            run_staggered_arrival_benchmark()
        if args.capacity:
            run_capacity_ceiling_benchmark()
