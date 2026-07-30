"""
MicroInfer - Phase 3: Continuous Batching Scheduler Benchmarking Harness
Evaluates system throughput, concurrent request latency, and GPU memory under mixed workload.
"""

import sys
import json
import time
import torch
from pathlib import Path

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_model_and_tokenizer, DEFAULT_MODEL_ID
from src.scheduler import ContinuousBatchScheduler


MIXED_WORKLOAD_PROMPTS = [
    ("Explain the transformer attention mechanism in detail.", 32),
    ("Write a Python function for binary search.", 20),
    ("What are the advantages of continuous batching?", 25),
    ("Describe the GPU memory hierarchy.", 30),
    ("How does post-training quantization reduce model size?", 24),
]


def run_scheduler_benchmark(
    model_id: str = DEFAULT_MODEL_ID,
    max_batch_size: int = 4,
):
    """
    Executes benchmark for Phase 3 Continuous Batching Scheduler under mixed workload.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device)
    model.eval()

    print("\n" + "=" * 60)
    print(f"  MICROINFER PHASE 3: CONTINUOUS BATCHING SCHEDULER BENCHMARK")
    print(f"  Model: {model_id}")
    print(f"  Device: {device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Max Concurrent Batch Size: {max_batch_size}")
    print("=" * 60 + "\n")

    # Initialize scheduler
    scheduler = ContinuousBatchScheduler(max_batch_size=max_batch_size, device=device)

    # Queue all workload prompts
    for prompt, max_tokens in MIXED_WORKLOAD_PROMPTS:
        scheduler.add_request(prompt, tokenizer, max_new_tokens=max_tokens, temperature=0.0)

    print(f"[Bench] Queued {len(MIXED_WORKLOAD_PROMPTS)} mixed workload requests into scheduler.")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()

    # Process all requests continuously until queue & batch are empty
    step_count = 0
    while scheduler.has_pending_work():
        scheduler.step(model, tokenizer)
        step_count += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_total = time.perf_counter() - t_start

    # Compute aggregate throughput across all completed requests
    completed_requests = scheduler.finished_sequences
    total_tokens_generated = sum(len(seq.generated_tokens) for seq in completed_requests)
    aggregate_throughput = total_tokens_generated / t_total if t_total > 0 else 0.0

    avg_ttft = sum(seq.ttft_ms for seq in completed_requests) / len(completed_requests)
    avg_total_latency_ms = (sum(seq.total_time_ms for seq in completed_requests) / len(completed_requests))

    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print(f"\n--- Continuous Batching Benchmark Results ---")
    print(f"  Completed Requests:    {len(completed_requests)}")
    print(f"  Total Step Iterations: {step_count}")
    print(f"  Total Tokens Gen:      {total_tokens_generated}")
    print(f"  Execution Time:        {t_total:.2f} s")
    print(f"  Average TTFT:          {avg_ttft:.2f} ms")
    print(f"  Average Req Latency:   {avg_total_latency_ms:.2f} ms")
    print(f"  Aggregate Throughput:  {aggregate_throughput:.2f} tokens/sec")
    print(f"  Peak VRAM Allocation:  {peak_vram_gb:.2f} GB\n")

    request_details = []
    for seq in completed_requests:
        request_details.append({
            "seq_id": seq.seq_id,
            "prompt": seq.prompt,
            "input_tokens": len(seq.prompt_tokens),
            "generated_tokens": len(seq.generated_tokens),
            "ttft_ms": round(seq.ttft_ms, 2),
            "total_latency_ms": round(seq.total_time_ms, 2),
            "output_sample": tokenizer.decode(seq.generated_tokens, skip_special_tokens=True)[:80] + "...",
        })

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    out_file = output_dir / "phase3_scheduler.json"

    export_data = {
        "phase": "Phase 3 - Continuous Batching Scheduler",
        "model_id": model_id,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "max_batch_size": max_batch_size,
        "total_requests": len(completed_requests),
        "total_tokens_generated": total_tokens_generated,
        "execution_time_sec": round(t_total, 3),
        "aggregate_throughput_tok_per_sec": round(aggregate_throughput, 2),
        "average_ttft_ms": round(avg_ttft, 2),
        "average_request_latency_ms": round(avg_total_latency_ms, 2),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "requests": request_details,
    }

    with open(out_file, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"[MicroInfer] Phase 3 Scheduler metrics saved to '{out_file}'.")
    return export_data


if __name__ == "__main__":
    run_scheduler_benchmark()
