# MicroInfer Gap Analysis & MLSys Technical Whitepaper

> **Hardware Specification:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, CUDA 12.1, SM 8.9 Ada Lovelace)  
> **Target Model:** `Qwen/Qwen2.5-1.5B-Instruct` (1.54B Parameters)  
> **Author:** MicroInfer LLM Serving Architecture Engine

---

## 📌 Master Executive Performance Matrix

| Phase | Serving Mechanism | TTFT (ms) | TPOT (ms/token) | Throughput (tok/sec) | Peak VRAM (GB) | Performance Scaling Model |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 0** | HuggingFace `.generate()` Baseline | **84.62 ms** | **68.55 ms/tok** | **14.60 tok/s** | **2.89 GB** | $\mathcal{O}(N)$ (Built-in KV-Cache) |
| **Phase 1** | Naive Generator (No Cache) | **62.24 ms** | **62.78 ms/tok** | **15.90 tok/s** | **2.94 GB** | $\mathcal{O}(N^2)$ Quadratic Slowdown |
| **Phase 2** | KV-Cache Generator | **63.73 ms** | **51.89 ms/tok** | **19.16 tok/s** | **2.89 GB** | $\mathcal{O}(N)$ Linear ($O(1)$ Decode Step) |
| **Phase 3** | Continuous Batching Scheduler | **117.19 ms** | **N/A (Concurrent)** | **18.13 tok/s** | **2.89 GB** | In-Flight Queue Scheduling |
| **Phase 4** | INT8 Quantized Engine | **551.70 ms** | **446.90 ms/tok** | **2.31 tok/s** | **1.68 GB** | 8-Bit Weight Quantization (-41.9% VRAM) |
| **Phase 5** | Production vLLM Reference Engine | **45.65 ms** | **53.01 ms/tok** | **18.75 tok/s** | **2.89 GB** | PagedAttention + Fused CUDA Kernels |

---

## 📈 Master Visualization Charts

![Master Throughput Comparison](plots/master_throughput_comparison.png)

![Master VRAM Memory Footprint](plots/master_vram_footprint.png)

---

## 🔬 Mechanistic Deep-Dive & Architectural Analysis

### 1. The Bottleneck of Uncached Generation ($\mathcal{O}(N^2)$ FLOP Overhead)
Without key-value caching, generating token $N$ requires calculating attention projections over all previous $1 \dots N-1$ tokens from scratch. The computational FLOP complexity grows quadratically:
$$\text{FLOPs}_{\text{Naive}} = \sum_{i=1}^{N} \mathcal{O}(i) = \mathcal{O}(N^2)$$
This introduces severe memory bandwidth saturation during long context generation.

### 2. KV-Caching Mechanics & Linear Scaling ($\mathcal{O}(N)$ Total / $\mathcal{O}(1)$ Per-Step)
Pre-allocating key and value tensors converts token generation into a 2-phase process:
- **Prefill Step:** Process all input prompt tokens in parallel, populating the cache.
- **Decode Steps:** Process single input tokens $(1 \times 1)$, updating the cache incrementally in constant time $\mathcal{O}(1)$ per step.
- **Speedup:** Achieved **+20.5% throughput boost** over uncached generation on RTX 4050.

### 3. Continuous Batching vs Static Padding Waste
Static batching waits for the slowest request in a batch to finish, idling GPU Tensor Cores. Continuous batching operates at iteration granularity, admitting new requests as soon as cache slots open up, maximizing aggregate hardware throughput under mixed workloads.

### 4. INT8 Quantization Trade-offs
Weight quantization reduces memory allocation from **2.89 GB down to 1.68 GB** (**-41.9% VRAM savings**). While 8-bit dequantization adds arithmetic overhead during matrix multiplications on laptop GPUs, the saved memory enables serving double the batch size or context window.

---

## 💡 Tier-1 MLSys Interview Questions & System Design Answers

### Q1: *"Why is LLM decoding memory-bandwidth bound rather than compute bound?"*
> **Answer:** During decoding, we execute a single-token forward pass $(B=1, T=1)$. The model must fetch all 1.5B parameters (~3GB in FP16) from VRAM to GPU SRAM to perform arithmetic over a single token. The arithmetic intensity (FLOPs / byte) is approximately 1.0, far below NVIDIA GPU saturation thresholds (~100-300 FLOPs/byte). Thus, decoding speed is strictly limited by GPU memory bandwidth (GB/s).

### Q2: *"How does PagedAttention in vLLM address memory fragmentation?"*
> **Answer:** Traditional KV-caches allocate contiguous VRAM arrays for maximum sequence length ($L_{\text{max}}=2048$), causing severe internal and external memory fragmentation (up to 60-80% wasted VRAM). PagedAttention borrows virtual memory paging principles from OS design, partitioning the KV-cache into fixed-size physical blocks (e.g., 16 tokens). Blocks are allocated dynamically on-demand, virtually eliminating fragmentation and enabling up to 2-4x higher concurrent batch sizes.

---

## 🎯 Benchmark Artifacts Reference
- **Phase 0 Data:** [benchmarks/results/phase0_baseline_hf.json](file:///c:/Users/LENOVO/Desktop/MICROINFER/benchmarks/results/phase0_baseline_hf.json)
- **Phase 1 Data:** [benchmarks/results/phase1_naive.json](file:///c:/Users/LENOVO/Desktop/MICROINFER/benchmarks/results/phase1_naive.json)
- **Phase 2 Data:** [benchmarks/results/phase2_cached.json](file:///c:/Users/LENOVO/Desktop/MICROINFER/benchmarks/results/phase2_cached.json)
- **Phase 3 Data:** [benchmarks/results/phase3_scheduler.json](file:///c:/Users/LENOVO/Desktop/MICROINFER/benchmarks/results/phase3_scheduler.json)
- **Phase 4 Data:** [benchmarks/results/phase4_quant.json](file:///c:/Users/LENOVO/Desktop/MICROINFER/benchmarks/results/phase4_quant.json)
- **Phase 5 Data:** [benchmarks/results/phase5_vllm.json](file:///c:/Users/LENOVO/Desktop/MICROINFER/benchmarks/results/phase5_vllm.json)
