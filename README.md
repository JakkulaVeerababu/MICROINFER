# MicroInfer 🚀

> **A From-Scratch LLM Inference Engine & vLLM Benchmarking Suite**  
> *Built with PyTorch on NVIDIA GeForce RTX 4050 GPU (6GB VRAM)*

MicroInfer is a custom transformer serving engine built from first principles to understand, implement, and benchmark the core algorithmic techniques that make production LLM serving fast: **KV-Caching**, **Continuous Batching**, and **Weight-Only Quantization**.

---

## 🎯 Key Features & Technical Highlights

- **KV-Cache Store (`src/kv_cache.py`):** Pre-allocated key-value projection store converting $O(n^2)$ attention re-computation into $O(n)$ linear generation.
- **Continuous Batching Scheduler (`src/scheduler.py`):** Dynamic in-flight request joining/leaving mechanism to maximize GPU memory bandwidth utilization.
- **Weight-Only INT8 Quantization (`src/quantization.py`):** Symmetric per-channel weight quantization shrinking VRAM usage with minimal accuracy degradation.
- **Empirical Profiling & vLLM Comparison (`benchmarks/`):** Full benchmark suite evaluating TTFT (Time-To-First-Token), TPOT (Time-Per-Output-Token), tokens/sec, and peak VRAM.

---

## 🏗️ Repository Layout

```
microinfer/
├── README.md                  # Project overview & benchmark summary
├── requirements.txt           # Environment dependencies
├── microinfer_build_guide.pdf # Original architecture & build specification
├── src/
│   ├── model_loader.py        # Model loading harness (FP16/BF16)
│   ├── naive_generate.py      # Phase 1: Uncached quadratic generator
│   ├── kv_cache.py            # Phase 2: KV-Cache tensor manager
│   ├── cached_generate.py     # Phase 2: Cached linear generator
│   ├── scheduler.py           # Phase 3: Continuous batching scheduler
│   ├── quantization.py       # Phase 4: INT8 Weight-only quantization
│   └── engine.py              # Main unified MicroInfer engine API
├── benchmarks/
│   ├── baseline_hf.py         # Phase 0: HuggingFace baseline runner
│   ├── run_benchmark.py       # Comprehensive 5-scenario benchmark harness
│   └── results/               # Committed raw benchmark results (JSON/CSV)
├── analysis/
│   ├── ANALYSIS.md            # Profiler-backed gap analysis vs vLLM
│   └── plots/                 # Benchmark charts & scaling curves
└── tests/
    └── test_correctness.py    # Logit verification against HuggingFace baseline
```

---

## 📊 Benchmark Results Summary (RTX 4050 6GB)

> Detailed profiler analysis and visual latency charts are documented in **[analysis/ANALYSIS.md](file:///c:/Users/LENOVO/Desktop/MICROINFER/analysis/ANALYSIS.md)**.

| Phase | Serving Mechanism | TTFT (ms) | TPOT (ms/token) | Throughput (tok/s) | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Phase 0** | HuggingFace `.generate()` Baseline | **84.62 ms** | **68.55 ms/tok** | **14.60 tok/s** | **2.89 GB** |
| **Phase 1** | Naive Generator (No Cache) | **62.24 ms** | **62.78 ms/tok** | **15.90 tok/s** | **2.94 GB** |
| **Phase 2** | KV-Cache Generator | *TBD* | *TBD* | *TBD* | *TBD* |
| **Phase 3** | Continuous Batching Scheduler | *TBD* | *TBD* | *TBD* | *TBD* |
| **Phase 4** | INT8 Quantized Engine | *TBD* | *TBD* | *TBD* | *TBD* |
| **Ref** | Production vLLM | *TBD* | *TBD* | *TBD* | *TBD* |

### 📈 Latency & Scaling Charts
- **Phase 0 Baseline Chart:** `analysis/plots/phase0_baseline.png`
- **Phase 1 Quadratic Scaling Chart:** `analysis/plots/phase1_quadratic_scaling.png`

---

## 💻 Quick Start & Environment Setup

```bash
# Clone the repository
git clone https://github.com/JakkulaVeerababu/MICROINFER.git
cd microinfer

# Install dependencies
pip install -r requirements.txt

# Run Phase 0 HuggingFace Baseline Benchmark
python benchmarks/baseline_hf.py
```

---

## 🛠️ Hardware Specification

All benchmarks recorded on:
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6141 MiB VRAM, Ada Lovelace)
- **CUDA Version:** 12.1 / Driver 560+
- **PyTorch:** 2.5.1+cu121
- **Host OS:** Windows 11 / Python 3.11

---

## 📄 License & Attribution

Designed and built for educational MLSys research & inference engineering development.
