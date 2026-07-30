# MicroInfer 🚀

> **A From-Scratch LLM Inference Engine & Production vLLM Benchmarking Suite**  
> *Built with PyTorch, CUDA 12.1, and Transformers on NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)*

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![PyTest 33 Passed](https://img.shields.io/badge/PyTest-33%2F33%20Passed-2ecc71?style=flat-square&logo=pytest&logoColor=white)](https://pytest.org)
[![License MIT](https://img.shields.io/badge/License-MIT-blue.style=flat-square)](LICENSE)

**MicroInfer** is a high-performance transformer serving engine engineered from first principles to implement, profile, and benchmark core LLM serving algorithms: **Pre-allocated Key-Value (KV) Caching**, **In-Flight Continuous Batching Scheduling**, and **8-Bit INT8 Weight Quantization**.

---

## 📌 Master Performance Matrix (RTX 4050 6GB)

| Phase | Serving System & Architecture | TTFT (1st Token) | TPOT (Decode Speed) | Aggregate Throughput | Peak VRAM | Complexity Scaling |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 0** | HuggingFace `.generate()` Baseline | **84.62 ms** | **68.55 ms/tok** | **14.60 tok/s** | **2.89 GB** | $\mathcal{O}(N)$ (HF KV-Cache) |
| **Phase 1** | Naive Generator (No Cache) | **62.24 ms** | **62.78 ms/tok** | **15.90 tok/s** | **2.94 GB** | $\mathcal{O}(N^2)$ Quadratic Slowdown |
| **Phase 2** | KV-Cache Generator Engine | **63.73 ms** | **51.89 ms/tok** | **19.16 tok/s** | **2.89 GB** | $\mathcal{O}(N)$ Linear ($\mathcal{O}(1)$ Decode Step) |
| **Phase 3** | Continuous Batching Scheduler | **117.19 ms** | **N/A (Concurrent)** | **18.13 tok/s** | **2.89 GB** | Dynamic Request Scheduling |
| **Phase 4** | INT8 Quantized Model Engine | **551.70 ms** | **446.90 ms/tok** | **2.31 tok/s** | **1.68 GB** | 8-Bit Weights (-41.9% VRAM) |
| **Phase 5** | Production vLLM Reference Engine | **45.65 ms** | **53.01 ms/tok** | **18.75 tok/s** | **2.89 GB** | PagedAttention + Fused Kernels |

> 🚀 **Key Performance Milestones:**
> - **KV-Caching Speedup:** Achieved **+20.5% generation throughput boost** (19.16 tok/s vs 15.90 tok/s) and reduced per-step decoding latency to a flat constant **~50.8 ms/token**.
> - **INT8 VRAM Savings:** Reduced GPU memory allocation from **2.89 GB down to 1.68 GB** (**-41.9% GPU memory savings**).

---

## 📊 Visual Benchmark Portfolio

| Master Comparative Throughput | Master VRAM Memory Footprint |
| :---: | :---: |
| ![Master Throughput](analysis/plots/master_throughput_comparison.png) | ![Master VRAM](analysis/plots/master_vram_footprint.png) |

| Phase 1 Quadratic Scaling | Phase 2 Flat Step Latency |
| :---: | :---: |
| ![Phase 1 Scaling](analysis/plots/phase1_quadratic_scaling.png) | ![Phase 2 Latency](analysis/plots/phase2_flat_step_latency.png) |

| Phase 3 Continuous Batching | Phase 4 VRAM Reduction |
| :---: | :---: |
| ![Phase 3 Batching](analysis/plots/phase3_scheduler_performance.png) | ![Phase 4 VRAM](analysis/plots/phase4_vram_reduction.png) |

---

## 🏗️ Repository Architecture & File Inventory

```
MICROINFER/
├── README.md                           # Master project guide & metrics table
├── requirements.txt                    # Project dependencies
├── PHASE0_SPEC.md                      # Phase 0 specification & deliverable matrix
├── PHASE1_SPEC.md                      # Phase 1 specification & deliverable matrix
├── PHASE2_SPEC.md                      # Phase 2 specification & deliverable matrix
├── PHASE3_SPEC.md                      # Phase 3 specification & deliverable matrix
├── PHASE4_SPEC.md                      # Phase 4 specification & deliverable matrix
├── PHASE5_SPEC.md                      # Phase 5 specification & deliverable matrix
│
├── src/                                # Core Engine Source Modules
│   ├── diagnostics.py                  # GPU hardware capability & CUDA diagnostic profiler
│   ├── memory_sizing.py                # Mathematical VRAM memory profiler & KV-cache sizing
│   ├── model_loader.py                 # HuggingFace FP16 model & tokenizer loader
│   ├── naive_generate.py               # Phase 1 Uncached Naive Generator (O(N^2) complexity)
│   ├── kv_cache.py                     # Phase 2 Pre-allocated KV-Cache Tensor Store
│   ├── cached_generate.py              # Phase 2 2-Phase Incremental Generator (O(1) step)
│   ├── scheduler.py                    # Phase 3 Continuous Batching Scheduler & Queue Engine
│   ├── quant_loader.py                 # Phase 4 8-Bit Quantized Weight Model Loader
│   └── quant_generate.py               # Phase 4 INT8 Quantized Generator Engine
│
├── benchmarks/                         # Benchmarking & Profiling Harnesses
│   ├── baseline_hf.py                  # Phase 0 HuggingFace .generate() control baseline
│   ├── benchmark_naive.py              # Phase 1 Naive Generator latency & scaling profiler
│   ├── benchmark_cached.py             # Phase 2 KV-Cache speedup & flat step latency profiler
│   ├── benchmark_scheduler.py          # Phase 3 Continuous batching mixed workload profiler
│   ├── benchmark_quant.py              # Phase 4 INT8 quantization memory & latency profiler
│   ├── baseline_vllm.py                # Phase 5 Production vLLM reference engine harness
│   └── results/                        # Raw JSON Benchmark Results Export
│
├── analysis/                           # Analysis Scripts, Visualizations & Technical Reports
│   ├── ANALYSIS.md                     # MLSys technical whitepaper & system design report
│   ├── plot_master.py                  # Plotter for master comparative throughput & VRAM charts
│   └── plots/                          # Rendered PNG Benchmark Charts
│
└── tests/                              # Automated PyTest Test Suites (33 Tests)
    ├── test_master_suite.py            # Master test suite verifying all 6 phases
    └── ...                             # 20 additional test modules (33/33 tests passing)
```

---

## 💻 Quick Start & Setup

### 1. Clone & Activate Environment
```bash
git clone https://github.com/JakkulaVeerababu/MICROINFER.git
cd MICROINFER
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install bitsandbytes>=0.50.0
```

### 3. Run Test Suite (33 Automated Tests)
```bash
python -m pytest tests/ -v
```

### 4. Run Benchmarks
```bash
# Run Phase 0 HuggingFace Baseline
python benchmarks/baseline_hf.py

# Run Phase 2 KV-Cache Benchmark
python benchmarks/benchmark_cached.py

# Run Phase 3 Continuous Batching Benchmark
python benchmarks/benchmark_scheduler.py

# Run Phase 4 INT8 Quantized Benchmark
python benchmarks/benchmark_quant.py

# Run Phase 5 vLLM Reference Benchmark
python benchmarks/baseline_vllm.py
```

---

## 📜 Published Reports & Specifications

- 📙 **MLSys Technical Whitepaper:** [analysis/ANALYSIS.md](file:///c:/Users/LENOVO/Desktop/MICROINFER/analysis/ANALYSIS.md)  
- 📋 **Phase 0 Spec Matrix:** [PHASE0_SPEC.md](file:///c:/Users/LENOVO/Desktop/MICROINFER/PHASE0_SPEC.md) (100% Complete)  
- 📋 **Phase 1 Spec Matrix:** [PHASE1_SPEC.md](file:///c:/Users/LENOVO/Desktop/MICROINFER/PHASE1_SPEC.md) (100% Complete)  
- 📋 **Phase 2 Spec Matrix:** [PHASE2_SPEC.md](file:///c:/Users/LENOVO/Desktop/MICROINFER/PHASE2_SPEC.md) (100% Complete)  
- 📋 **Phase 3 Spec Matrix:** [PHASE3_SPEC.md](file:///c:/Users/LENOVO/Desktop/MICROINFER/PHASE3_SPEC.md) (100% Complete)  
- 📋 **Phase 4 Spec Matrix:** [PHASE4_SPEC.md](file:///c:/Users/LENOVO/Desktop/MICROINFER/PHASE4_SPEC.md) (100% Complete)  
- 📋 **Phase 5 Spec Matrix:** [PHASE5_SPEC.md](file:///c:/Users/LENOVO/Desktop/MICROINFER/PHASE5_SPEC.md) (100% Complete)  

---

## 🌐 Live GitHub Repository
👉 **[https://github.com/JakkulaVeerababu/MICROINFER](https://github.com/JakkulaVeerababu/MICROINFER)**
