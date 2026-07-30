# MicroInfer Gap Analysis & Serving Performance Report

> **Hardware Specification:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, CUDA 12.1, SM 8.9 Ada Lovelace)  
> **Target Model:** `Qwen/Qwen2.5-1.5B-Instruct` (1.54B Parameters, FP16 Precision)

---

## 📌 Executive Performance Summary

| Phase | Serving Mechanism | TTFT (ms) | TPOT (ms/token) | Throughput (tok/sec) | Peak VRAM (GB) | Performance Scaling |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 0** | HuggingFace `.generate()` Baseline | **84.62 ms** | **68.55 ms/tok** | **14.60 tok/s** | **2.89 GB** | $\mathcal{O}(N)$ (Built-in KV-Cache) |
| **Phase 1** | Naive Generator (No Cache) | **62.24 ms** | **62.78 ms/tok** | **15.90 tok/s** | **2.94 GB** | $\mathcal{O}(N^2)$ Quadratic Slowdown |

---

## 📊 Sub-Phase 0.5 Analysis: HuggingFace Baseline Control Group

In **Sub-Phase 0.5**, we established our reference control group using HuggingFace's standard `.generate()` implementation. 

![Phase 0 Baseline Latency](plots/phase0_baseline.png)

### Key Observations:
1. **Time-To-First-Token (TTFT):** Averaged **84.62 ms** across input lengths $14 \dots 23$ tokens. TTFT measures the initial prompt processing delay (prefill phase).
2. **Time-Per-Output-Token (TPOT):** Averaged **68.55 ms/token** for steady-state autoregressive decoding.
3. **Memory Footprint:** Model parameters occupy **2.88 GB VRAM** in FP16 precision. Peak VRAM allocation remained stable at **2.89 GB**.

---

## 📉 Sub-Phase 1.4 Analysis: Uncached Naive Generator Quadratic Slowdown

In **Sub-Phase 1.4**, we measured step-by-step latency for our custom uncached generator ([src/naive_generate.py](file:///c:/Users/LENOVO/Desktop/MICROINFER/src/naive_generate.py)).

![Phase 1 Quadratic Scaling](plots/phase1_quadratic_scaling.png)

### Mechanistic Cause of Slowdown:
- Without a KV-cache, generating step $N$ requires re-running the full model forward pass over all $N$ tokens.
- On step 1, the model computes attention for 1 token. On step 64, the model re-computes $Q, K, V$ projections for 64 tokens from scratch.
- **Compute Complexity:** Re-computing $K$ and $V$ projections introduces an $\mathcal{O}(N^2)$ FLOP overhead per token generation, causing step latency to increase as the context grows.

---

## 🎯 Benchmark Artifacts Reference
- **Phase 0 Data:** [benchmarks/results/phase0_baseline_hf.json](file:///c:/Users/LENOVO/Desktop/MICROINFER/benchmarks/results/phase0_baseline_hf.json)
- **Phase 1 Data:** [benchmarks/results/phase1_naive.json](file:///c:/Users/LENOVO/Desktop/MICROINFER/benchmarks/results/phase1_naive.json)
- **System Diagnostics:** [analysis/gpu_diagnostics.json](file:///c:/Users/LENOVO/Desktop/MICROINFER/analysis/gpu_diagnostics.json)
- **VRAM Memory Sizing:** [analysis/memory_sizing.json](file:///c:/Users/LENOVO/Desktop/MICROINFER/analysis/memory_sizing.json)
