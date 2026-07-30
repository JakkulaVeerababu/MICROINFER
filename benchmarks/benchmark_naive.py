"""
MicroInfer - Phase 1: Naive Generator Benchmarking Harness
Measures step-by-step latency growth and overall throughput for uncached generation.
"""

import sys
import json
import time
import torch
from pathlib import Path

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_model_and_tokenizer, DEFAULT_MODEL_ID
from src.naive_generate import naive_generate


TEST_PROMPTS = [
    "Explain how transformer attention mechanism works in simple terms for a software engineer.",
    "Write a Python function to implement quicksort with step-by-step explanations.",
    "What are the key trade-offs between KV-caching, continuous batching, and weight quantization in LLM serving?",
]


def run_naive_benchmark(
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 64,
    num_runs: int = 2,
):
    """
    Executes benchmark for Phase 1 Naive Generator (uncached).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device)
    model.eval()

    print("\n" + "=" * 60)
    print(f"  MICROINFER PHASE 1: NAIVE GENERATOR (UNCACHED) BENCHMARK")
    print(f"  Model: {model_id}")
    print(f"  Device: {device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Target Max New Tokens: {max_new_tokens}")
    print("=" * 60 + "\n")

    # 1. Warm-up run
    print("[Bench] Running warm-up run (discarded)...")
    _ = naive_generate(model, tokenizer, "Warm up prompt", max_new_tokens=5, temperature=0.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    print("[Bench] Warm-up complete.\n")

    results = []

    for prompt_idx, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"--- Scenario {prompt_idx}: Prompt Length {len(prompt.split())} words ---")
        
        prompt_ttfts = []
        prompt_tpots = []
        prompt_throughputs = []
        step_times_matrix = []
        sample_output = ""

        for run in range(num_runs):
            res = naive_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
            
            prompt_ttfts.append(res["ttft_ms"])
            prompt_tpots.append(res["tpot_ms"])
            prompt_throughputs.append(res["throughput_tok_per_sec"])
            step_times_matrix.append(res["step_times_ms"])
            sample_output = res["output_text"]

        avg_ttft = sum(prompt_ttfts) / len(prompt_ttfts)
        avg_tpot = sum(prompt_tpots) / len(prompt_tpots)
        avg_throughput = sum(prompt_throughputs) / len(prompt_throughputs)
        
        # Calculate average per-step latency across runs to highlight quadratic curve
        num_steps = len(step_times_matrix[0])
        avg_step_times = [
            round(sum(run_steps[s] for run_steps in step_times_matrix) / len(step_times_matrix), 2)
            for s in range(num_steps)
        ]

        print(f"  Input Tokens:  {res['prompt_tokens']}")
        print(f"  Output Tokens: {res['generated_tokens']}")
        print(f"  TTFT:          {avg_ttft:.2f} ms")
        print(f"  TPOT:          {avg_tpot:.2f} ms/token")
        print(f"  Throughput:    {avg_throughput:.2f} tokens/sec")
        print(f"  Step 1 Latency (N=1):   {avg_step_times[0]} ms")
        if num_steps > 1:
            print(f"  Step {num_steps} Latency (N={num_steps}): {avg_step_times[-1]} ms  <-- Quadratic Slowdown Penalty!")
        print(f"  Sample Text:   \"{sample_output[:100]}...\"\n")

        results.append({
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "input_tokens": res["prompt_tokens"],
            "output_tokens": res["generated_tokens"],
            "ttft_ms": round(avg_ttft, 2),
            "tpot_ms": round(avg_tpot, 2),
            "throughput_tok_per_sec": round(avg_throughput, 2),
            "per_step_latency_ms": avg_step_times,
        })

    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak VRAM Usage across all runs: {peak_vram_gb:.2f} GB")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    out_file = output_dir / "phase1_naive.json"

    export_data = {
        "phase": "Phase 1 - Naive Generator (Uncached)",
        "model_id": model_id,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "peak_vram_gb": round(peak_vram_gb, 2),
        "results": results,
    }

    with open(out_file, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"\n[MicroInfer] Phase 1 Naive Generator metrics saved to '{out_file}'.")
    return export_data


if __name__ == "__main__":
    run_naive_benchmark()
