"""
MicroInfer - PyTorch CUDA Profiler Harness
Profiles Naive Generator (Phase 1) vs KV-Cached Generator (Phase 2) using torch.profiler.
Exports detailed kernel timing tables, self CUDA execution times, and trace files to analysis/profiler_results/.
"""

import sys
import os
import json
import torch
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.model_loader import load_model_and_tokenizer
from src.naive_generate import naive_generate
from src.cached_generate import cached_generate


def prepare_prompt_with_token_count(tokenizer, target_tokens: int = 256) -> str:
    base_text = "The field of artificial intelligence and high-performance GPU serving algorithms has grown rapidly in recent years. "
    prompt = base_text
    tokens = tokenizer.encode(prompt)
    while len(tokens) < target_tokens:
        prompt += base_text
        tokens = tokenizer.encode(prompt)
    
    tokens = tokens[:target_tokens]
    return tokenizer.decode(tokens)


def run_profiler():
    print("==========================================================", flush=True)
    print(" MicroInfer - PyTorch CUDA Engine Profiler Harness", flush=True)
    print("==========================================================", flush=True)
    
    if not torch.cuda.is_available():
        print("[ERROR] CUDA GPU is required to run PyTorch CUDA profiler.", flush=True)
        sys.exit(1)
        
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Device: {gpu_name} (CUDA {torch.version.cuda})", flush=True)

    output_dir = ROOT_DIR / "analysis" / "profiler_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading model and tokenizer...", flush=True)
    model, tokenizer = load_model_and_tokenizer(device=device)

    target_n = 256
    gen_tokens = 32
    prompt = prepare_prompt_with_token_count(tokenizer, target_tokens=target_n)
    actual_tokens = len(tokenizer.encode(prompt))
    print(f"Prepared prompt with length N = {actual_tokens} tokens. Target new tokens = {gen_tokens}.", flush=True)

    # --- Warm-up ---
    print("\n[2/4] Executing warm-up runs...", flush=True)
    for _ in range(2):
        _ = naive_generate(model, tokenizer, prompt, max_new_tokens=2)
        _ = cached_generate(model, tokenizer, prompt, max_new_tokens=2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # --- Profiling Naive Generation ---
    print("\n[3/4] Profiling Naive Generation (Uncached O(N^2) re-computation)...", flush=True)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof_naive:
        _ = naive_generate(model, tokenizer, prompt, max_new_tokens=gen_tokens)
        torch.cuda.synchronize()

    print("Extracting Naive profile stats...", flush=True)
    naive_cuda_str = str(prof_naive.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    naive_self_cuda_str = str(prof_naive.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
    
    trace_naive_path = output_dir / "trace_naive.json"
    try:
        prof_naive.export_chrome_trace(str(trace_naive_path))
        print(f" -> Exported naive trace: {trace_naive_path}", flush=True)
    except Exception as e:
        print(f" -> Trace export warning: {e}", flush=True)

    torch.cuda.empty_cache()

    # --- Profiling KV-Cached Generation ---
    print("\n[4/4] Profiling KV-Cached Generation (Phase 2 O(1) step decode)...", flush=True)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof_cached:
        _ = cached_generate(model, tokenizer, prompt, max_new_tokens=gen_tokens)
        torch.cuda.synchronize()

    print("Extracting Cached profile stats...", flush=True)
    cached_cuda_str = str(prof_cached.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    cached_self_cuda_str = str(prof_cached.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

    trace_cached_path = output_dir / "trace_cached.json"
    try:
        prof_cached.export_chrome_trace(str(trace_cached_path))
        print(f" -> Exported cached trace: {trace_cached_path}", flush=True)
    except Exception as e:
        print(f" -> Trace export warning: {e}", flush=True)

    # Save profiler text report
    report_path = output_dir / "profiler_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"MicroInfer PyTorch CUDA Profiler Empirical Analysis\n")
        f.write(f"GPU: {gpu_name}\n")
        f.write(f"Prompt Context Length N: {actual_tokens}\n")
        f.write(f"Generated Tokens: {gen_tokens}\n\n")
        
        f.write("=" * 85 + "\n")
        f.write("1. NAIVE GENERATION - TOP CUDA TIME KERNELS\n")
        f.write("=" * 85 + "\n")
        f.write(naive_cuda_str + "\n\n")

        f.write("=" * 85 + "\n")
        f.write("2. NAIVE GENERATION - TOP SELF CUDA KERNELS (CORE FLOP / BOTTLENECK)\n")
        f.write("=" * 85 + "\n")
        f.write(naive_self_cuda_str + "\n\n")

        f.write("=" * 85 + "\n")
        f.write("3. KV-CACHED GENERATION - TOP CUDA TIME KERNELS\n")
        f.write("=" * 85 + "\n")
        f.write(cached_cuda_str + "\n\n")

        f.write("=" * 85 + "\n")
        f.write("4. KV-CACHED GENERATION - TOP SELF CUDA KERNELS (CORE FLOP / BOTTLENECK)\n")
        f.write("=" * 85 + "\n")
        f.write(cached_self_cuda_str + "\n\n")

    print(f"\n[SUCCESS] Profiler report saved to {report_path}", flush=True)
    print("\n--- Naive Top Self CUDA Kernels ---", flush=True)
    print(naive_self_cuda_str, flush=True)
    print("\n--- KV-Cached Top Self CUDA Kernels ---", flush=True)
    print(cached_self_cuda_str, flush=True)


if __name__ == "__main__":
    run_profiler()
