"""
MicroInfer - Phase 0: HuggingFace Baseline Benchmarking Script
Measures generation throughput, TTFT, TPOT, and peak VRAM using standard HuggingFace .generate().
"""

import os
import sys
import json
import time
import torch
from pathlib import Path

# Ensure src directory is in Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import load_model_and_tokenizer, DEFAULT_MODEL_ID


TEST_PROMPTS = [
    "Explain how transformer attention mechanism works in simple terms for a software engineer.",
    "Write a Python function to implement quicksort with step-by-step explanations.",
    "What are the key trade-offs between KV-caching, continuous batching, and weight quantization in LLM serving?",
]


def benchmark_hf_generate(
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 128,
    num_runs: int = 3,
):
    """
    Runs baseline benchmarking using HuggingFace's built-in .generate() method.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device)
    model.eval()

    print("\n" + "=" * 60)
    print(f"  MICROINFER PHASE 0: HUGGINGFACE BASELINE BENCHMARK")
    print(f"  Model: {model_id}")
    print(f"  Device: {device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Target Max New Tokens: {max_new_tokens}")
    print("=" * 60 + "\n")

    # 1. Warm-up run (cuda kernel compilation / allocation)
    print("[Bench] Running warm-up run (discarded)...")
    warmup_input = tokenizer("Hello, world!", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**warmup_input, max_new_tokens=10)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    print("[Bench] Warm-up complete.\n")

    results = []

    for prompt_idx, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"--- Scenario {prompt_idx}: Prompt Length {len(prompt.split())} words ---")
        prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_length = prompt_inputs.input_ids.shape[1]

        prompt_ttfts = []
        prompt_tpots = []
        prompt_throughputs = []
        prompt_generated_texts = []

        for run in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            t_start = time.perf_counter()

            # First token (TTFT estimation)
            with torch.no_grad():
                first_output = model.generate(**prompt_inputs, max_new_tokens=1, min_new_tokens=1)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_first = time.perf_counter()
            ttft_ms = (t_first - t_start) * 1000.0

            # Rest of tokens
            t_gen_start = time.perf_counter()
            with torch.no_grad():
                full_output = model.generate(
                    **prompt_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy decoding for consistent benchmarks
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_gen_end = time.perf_counter()

            total_gen_time = t_gen_end - t_gen_start
            generated_tokens = full_output.shape[1] - input_length
            
            tpot_ms = (total_gen_time / generated_tokens) * 1000.0 if generated_tokens > 0 else 0
            tokens_per_sec = generated_tokens / total_gen_time if total_gen_time > 0 else 0

            prompt_ttfts.append(ttft_ms)
            prompt_tpots.append(tpot_ms)
            prompt_throughputs.append(tokens_per_sec)

            if run == num_runs - 1:
                decoded = tokenizer.decode(full_output[0][input_length:], skip_special_tokens=True)
                prompt_generated_texts.append(decoded[:100] + "...")

        avg_ttft = sum(prompt_ttfts) / len(prompt_ttfts)
        avg_tpot = sum(prompt_tpots) / len(prompt_tpots)
        avg_throughput = sum(prompt_throughputs) / len(prompt_throughputs)

        print(f"  Input Tokens:  {input_length}")
        print(f"  Output Tokens: {max_new_tokens}")
        print(f"  TTFT:          {avg_ttft:.2f} ms")
        print(f"  TPOT:          {avg_tpot:.2f} ms/token")
        print(f"  Throughput:    {avg_throughput:.2f} tokens/sec")
        print(f"  Sample Text:   \"{prompt_generated_texts[-1]}\"\n")

        results.append({
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "input_tokens": input_length,
            "output_tokens": max_new_tokens,
            "ttft_ms": round(avg_ttft, 2),
            "tpot_ms": round(avg_tpot, 2),
            "throughput_tok_per_sec": round(avg_throughput, 2),
        })

    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak VRAM Usage across all runs: {peak_vram_gb:.2f} GB")

    # Export results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    out_file = output_dir / "phase0_baseline_hf.json"

    export_data = {
        "phase": "Phase 0 - HuggingFace Baseline",
        "model_id": model_id,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "peak_vram_gb": round(peak_vram_gb, 2),
        "results": results,
    }

    with open(out_file, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"\n[MicroInfer] Phase 0 Baseline metrics saved to '{out_file}'.")
    return export_data


if __name__ == "__main__":
    benchmark_hf_generate()
