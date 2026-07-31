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

    # Start GPU utilization sampler — samples nvidia-smi once per second in background
    gpu_sampler = GpuUtilSampler(interval_sec=1.0)

    with gpu_sampler:
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

    gpu_util_result = {
        "mean_pct":  gpu_sampler.mean,
        "std_pct":   gpu_sampler.std,
        "peak_pct":  gpu_sampler.peak,
        "n_samples": gpu_sampler.n_samples,
    }

    print("--- Aggregate across all waves ---")
    print(f"  Aggregate Throughput  mean={tp_stats['mean']:.2f} ± {tp_stats['std']:.2f} t/s  p50={tp_stats['p50']:.2f} t/s  p99={tp_stats['p99']:.2f} t/s")
    print(f"  Mean TTFT             mean={ttft_stats['mean']:.1f}ms ± {ttft_stats['std']:.1f}ms")
    print(f"  Request Latency       mean={lat_stats['mean']:.1f}ms ± {lat_stats['std']:.1f}ms  p50={lat_stats['p50']:.1f}ms  p99={lat_stats['p99']:.1f}ms")
    print(f"  Peak VRAM             {peak_vram_gb:.2f} GB")
    print(f"  GPU Utilization       mean={gpu_sampler.mean:.1f}% ± {gpu_sampler.std:.1f}%  peak={gpu_sampler.peak:.0f}%  (n={gpu_sampler.n_samples} samples)")

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
        "gpu_utilization_pct": gpu_util_result,
        "honest_note": (
            "Scheduler manages request queue lifecycles (WAITING -> RUNNING -> FINISHED). "
            "Decode pass is genuinely tensor-batched: all B active sequences are stacked into "
            "a single (B, 1) input tensor per step. Prefill remains sequential. "
            "GPU utilization sampled via nvidia-smi once per second during the timed benchmark."
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


class GpuUtilSampler:
    """
    Context manager that samples nvidia-smi GPU utilization % once per second
    in a background thread during the benchmark run.

    Usage:
        with GpuUtilSampler() as sampler:
            run_benchmark_waves()
        print(sampler.mean, sampler.std, sampler.peak)
    """

    def __init__(self, interval_sec: float = 1.0):
        self.interval_sec = interval_sec
        self.samples: list = []
        self._stop_event = threading.Event()
        self._thread = None

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3,
                )
                if result.returncode == 0:
                    val = result.stdout.strip()
                    if val:
                        self.samples.append(float(val))
            except Exception:
                pass
            self._stop_event.wait(timeout=self.interval_sec)

    def __enter__(self):
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def mean(self) -> float:
        return round(sum(self.samples) / len(self.samples), 1) if self.samples else 0.0

    @property
    def std(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        m = self.mean
        return round((sum((x - m) ** 2 for x in self.samples) / (len(self.samples) - 1)) ** 0.5, 1)

    @property
    def peak(self) -> float:
        return max(self.samples) if self.samples else 0.0

    @property
    def n_samples(self) -> int:
        return len(self.samples)

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


# ===========================================================================
# Static Batch Baseline & Head-to-Head Comparison
# ===========================================================================

class StaticBatchBaseline:
    """
    Naive static batcher: collects all requests upfront, processes each to
    completion sequentially (no interleaving), simulating a system that
    waits for the longest request before admitting new ones (head-of-line blocking).

    This is the baseline that continuous batching improves upon.
    """

    def __init__(self, batch_size: int = 8, device: str = "cuda"):
        self.batch_size = batch_size
        self.device = device
        self.pending: list = []
        self.finished: list = []

    def add_request(self, prompt: str, max_new_tokens: int):
        """Queue a request (prompt, max_new_tokens) for processing."""
        self.pending.append({"prompt": prompt, "max_new_tokens": max_new_tokens})

    def run(self, model, tokenizer, progress: bool = True) -> float:
        """
        Process all pending requests sequentially (one at a time per batch slot).
        Returns total wall time in seconds.
        """
        from src.cached_generate import cached_generate

        t_total_start = time.perf_counter()

        batches = [
            self.pending[i : i + self.batch_size]
            for i in range(0, len(self.pending), self.batch_size)
        ]

        for b_idx, batch in enumerate(batches):
            if progress:
                print(f"  [StaticBatch {b_idx+1}/{len(batches)}] Processing {len(batch)} requests sequentially...")
            for req in batch:
                t0 = time.perf_counter()
                result = cached_generate(
                    model, tokenizer,
                    req["prompt"],
                    max_new_tokens=req["max_new_tokens"],
                    temperature=0.0,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                self.finished.append({
                    "prompt":           req["prompt"][:40],
                    "generated_tokens": result["generated_tokens"],
                    "latency_ms":       round(latency_ms, 2),
                    "ttft_ms":          result["ttft_ms"],
                })

        return time.perf_counter() - t_total_start

    def total_tokens(self) -> int:
        return sum(r["generated_tokens"] for r in self.finished)

    def mean_latency_ms(self) -> float:
        if not self.finished:
            return 0.0
        return round(sum(r["latency_ms"] for r in self.finished) / len(self.finished), 2)


def run_static_vs_continuous_benchmark(
    model_id: str = DEFAULT_MODEL_ID,
    max_batch_size: int = 8,
):
    """
    Runs both StaticBatchBaseline and ContinuousBatchScheduler against the
    identical 8-request staggered arrival workload and saves a side-by-side
    comparison to benchmarks/results/phase3_static_vs_continuous.json.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device)
    model.eval()

    print("\n" + "=" * 60)
    print("  PHASE 3: STATIC vs CONTINUOUS BATCHING — HEAD TO HEAD")
    print("=" * 60 + "\n")

    # Identical workload used for both systems
    WORKLOAD = [
        (0.0,  "What is the capital of France?",                  16),
        (0.5,  "Write a quicksort in python.",                    64),
        (1.2,  "Explain transformer attention.",                  128),
        (1.5,  "Translate hello to Spanish.",                       8),
        (2.0,  "Why is the sky blue? Explain in extreme detail.", 256),
        (2.5,  "Tell me a short joke.",                            32),
        (3.0,  "Summarize the history of Rome.",                  200),
        (3.5,  "What is 2+2?",                                      4),
    ]

    # -------------------------------------------------------------------
    # System A: Static Batching
    # Ignores arrival times — receives all at t=0, processes sequentially
    # -------------------------------------------------------------------
    print("[A] Running STATIC BATCH Baseline (sequential, no interleaving)...")
    static = StaticBatchBaseline(batch_size=max_batch_size, device=device)
    for _, prompt, max_new_tokens in WORKLOAD:
        static.add_request(prompt, max_new_tokens)

    static_wall_time = static.run(model, tokenizer, progress=True)
    static_total_tokens = static.total_tokens()
    static_throughput = static_total_tokens / static_wall_time if static_wall_time > 0 else 0
    static_mean_lat = static.mean_latency_ms()
    static_active_ms = sum(r["latency_ms"] for r in static.finished)
    static_idle_ms = max(0.0, static_wall_time * 1000 - static_active_ms)

    print(f"  Wall Time    : {static_wall_time:.2f} s")
    print(f"  Total Tokens : {static_total_tokens}")
    print(f"  Throughput   : {static_throughput:.2f} tok/s")
    print(f"  Mean Latency : {static_mean_lat:.1f} ms")
    print(f"  GPU Idle Est.: {static_idle_ms:.0f} ms\n")

    # -------------------------------------------------------------------
    # System B: Continuous Batching Scheduler (staggered arrival)
    # -------------------------------------------------------------------
    print("[B] Running CONTINUOUS BATCH Scheduler (staggered arrival)...")
    scheduler = ContinuousBatchScheduler(max_batch_size=max_batch_size, device=device)

    t_cb_start = time.perf_counter()
    next_req_idx = 0
    while next_req_idx < len(WORKLOAD) or scheduler.has_pending_work():
        current_time = time.perf_counter() - t_cb_start
        while next_req_idx < len(WORKLOAD) and current_time >= WORKLOAD[next_req_idx][0]:
            _, prompt, max_new_tokens = WORKLOAD[next_req_idx]
            scheduler.add_request(prompt, tokenizer,
                                   max_new_tokens=max_new_tokens,
                                   temperature=0.0,
                                   model=model)
            next_req_idx += 1
        if scheduler.has_pending_work():
            scheduler.step(model, tokenizer)
        else:
            time.sleep(0.005)

    cb_wall_time = time.perf_counter() - t_cb_start
    cb_seqs = scheduler.finished_sequences
    cb_total_tokens = sum(len(s.generated_tokens) for s in cb_seqs)
    cb_throughput = cb_total_tokens / cb_wall_time if cb_wall_time > 0 else 0
    cb_mean_lat = round(
        sum(s.total_time_ms for s in cb_seqs) / len(cb_seqs) if cb_seqs else 0, 2
    )
    cb_active_ms = sum(s.total_time_ms for s in cb_seqs)
    cb_idle_ms = max(0.0, cb_wall_time * 1000 - cb_active_ms)

    print(f"  Wall Time    : {cb_wall_time:.2f} s")
    print(f"  Total Tokens : {cb_total_tokens}")
    print(f"  Throughput   : {cb_throughput:.2f} tok/s")
    print(f"  Mean Latency : {cb_mean_lat:.1f} ms")
    print(f"  GPU Idle Est.: {cb_idle_ms:.0f} ms\n")

    # -------------------------------------------------------------------
    # Comparison summary
    # -------------------------------------------------------------------
    throughput_gain_pct = (
        (cb_throughput - static_throughput) / static_throughput * 100
        if static_throughput > 0 else 0.0
    )
    latency_reduction_pct = (
        (static_mean_lat - cb_mean_lat) / static_mean_lat * 100
        if static_mean_lat > 0 else 0.0
    )
    wall_time_reduction_pct = (
        (static_wall_time - cb_wall_time) / static_wall_time * 100
        if static_wall_time > 0 else 0.0
    )

    print("--- Head-to-Head Summary ---")
    print(f"  Throughput gain   (CB vs Static) : {throughput_gain_pct:+.1f}%")
    print(f"  Latency reduction (CB vs Static) : {latency_reduction_pct:+.1f}%")
    print(f"  Wall time savings (CB vs Static) : {wall_time_reduction_pct:+.1f}%")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    result = {
        "phase": "Phase 3 - Static vs Continuous Batching Comparison",
        "model_id": model_id,
        "workload": [
            {"arrival_t": t, "prompt": p[:40], "max_new_tokens": n}
            for t, p, n in WORKLOAD
        ],
        "static_batch": {
            "wall_time_s":            round(static_wall_time, 3),
            "total_tokens":           static_total_tokens,
            "throughput_tok_per_sec": round(static_throughput, 2),
            "mean_latency_ms":        static_mean_lat,
            "gpu_idle_estimate_ms":   round(static_idle_ms, 0),
        },
        "continuous_batch": {
            "wall_time_s":            round(cb_wall_time, 3),
            "total_tokens":           cb_total_tokens,
            "throughput_tok_per_sec": round(cb_throughput, 2),
            "mean_latency_ms":        cb_mean_lat,
            "gpu_idle_estimate_ms":   round(cb_idle_ms, 0),
        },
        "comparison": {
            "throughput_gain_pct":     round(throughput_gain_pct, 1),
            "latency_reduction_pct":   round(latency_reduction_pct, 1),
            "wall_time_reduction_pct": round(wall_time_reduction_pct, 1),
        },
    }
    with open(output_dir / "phase3_static_vs_continuous.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[MicroInfer] Comparison -> '{output_dir / 'phase3_static_vs_continuous.json'}'")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--staggered", action="store_true", help="Run the staggered arrival benchmark")
    parser.add_argument("--capacity",  action="store_true", help="Run the capacity ceiling benchmark")
    parser.add_argument("--compare",   action="store_true", help="Run static vs continuous batching comparison")
    args = parser.parse_args()

    if not args.staggered and not args.capacity and not args.compare:
        run_scheduler_benchmark()
    else:
        if args.staggered:
            run_staggered_arrival_benchmark()
        if args.capacity:
            run_capacity_ceiling_benchmark()
        if args.compare:
            run_static_vs_continuous_benchmark()

