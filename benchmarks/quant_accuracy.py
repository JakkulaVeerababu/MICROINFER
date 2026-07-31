"""
MicroInfer - Phase 4 Accuracy Evaluation: INT8 vs FP16 Perplexity
==================================================================
Computes token-level cross-entropy perplexity for both the FP16 baseline
and the INT8 bitsandbytes-quantized model on an identical held-out corpus.

Eval corpus: Wikipedia passage on General Relativity (CC BY-SA 3.0).
Method: Sliding-window perplexity (stride=256, window=512).
Output: benchmarks/results/phase4_accuracy.json
Run:    python benchmarks/quant_accuracy.py
"""

import gc
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import DEFAULT_MODEL_ID, load_model_and_tokenizer
from src.quant_loader import load_quantized_model_and_tokenizer

EVAL_CORPUS = (
    "General relativity is a theory of gravitation developed by Albert Einstein "
    "between 1907 and 1915. The theory was published in 1915 and is based on the "
    "equivalence principle, which states that the local effects of gravity and "
    "acceleration are indistinguishable. According to general relativity, the "
    "observed gravitational attraction between masses results from their warping "
    "of space and time. "
    "Before general relativity, Newton law of universal gravitation had been "
    "accepted for over two centuries as a valid description of gravitational force. "
    "Einstein proposed that gravity is a consequence of how mass and energy curve "
    "spacetime, the four-dimensional fabric formed by combining three spatial "
    "dimensions and one time dimension. Objects in free fall follow geodesics, the "
    "curved paths equivalent to straight lines in curved spacetime. "
    "One of the most remarkable predictions of general relativity was the bending "
    "of light around massive objects, confirmed during the solar eclipse of 1919. "
    "General relativity predicted the existence of black holes, regions of spacetime "
    "where gravity is so strong that nothing, not even light, can escape. "
    "Gravitational waves, ripples in spacetime caused by accelerating massive objects, "
    "were first directly detected by the LIGO and Virgo collaborations in 2016. "
    "The Einstein field equations relate the geometry of spacetime described by the "
    "metric tensor to the distribution of matter and energy within it. These ten "
    "coupled, nonlinear partial differential equations are notoriously difficult to "
    "solve in full generality, but exact solutions exist for simple configurations "
    "such as a spherically symmetric mass distribution described by the Schwarzschild metric."
)

STRIDE = 256
WINDOW = 512
RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_JSON = RESULTS_DIR / "phase4_accuracy.json"


def compute_perplexity(model, tokenizer, text: str, stride: int, window: int, device: str) -> float:
    """Sliding-window token-level cross-entropy perplexity."""
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    seq_len = input_ids.size(1)
    nlls, n_tokens, prev_end_loc = [], 0, 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + window, seq_len)
        target_len = end_loc - prev_end_loc
        chunk = input_ids[:, begin_loc:end_loc]
        target_ids = chunk.clone()
        target_ids[:, :-target_len] = -100
        with torch.no_grad():
            nll = model(chunk, labels=target_ids).loss
        nlls.append(nll.item() * target_len)
        n_tokens += target_len
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    return math.exp(sum(nlls) / n_tokens)


def run_accuracy_eval(
    model_id: str = DEFAULT_MODEL_ID,
    corpus: str = EVAL_CORPUS,
    stride: int = STRIDE,
    window: int = WINDOW,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    print("=" * 62)
    print("  MICROINFER PHASE 4: INT8 ACCURACY EVALUATION (PERPLEXITY)")
    print(f"  Model   : {model_id}")
    print(f"  Device  : {device.upper()}")
    print(f"  Method  : Sliding-window PPL  stride={stride}  window={window}")
    print(f"  Corpus  : General Relativity Wikipedia excerpt")
    print("=" * 62)

    # FP16 baseline
    print("\n[1/2] Loading FP16 (baseline) model...")
    fp16_model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device)
    fp16_model.eval()
    corpus_tokens = tokenizer(corpus, return_tensors="pt").input_ids.size(1)
    print(f"      Corpus tokens : {corpus_tokens}")
    print("      Computing FP16 perplexity...")
    fp16_ppl = compute_perplexity(fp16_model, tokenizer, corpus, stride, window, device)
    print(f"  FP16 Perplexity : {fp16_ppl:.4f}\n")
    del fp16_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # INT8 quantized
    print("[2/2] Loading INT8 (bitsandbytes) quantized model...")
    int8_model, _ = load_quantized_model_and_tokenizer(model_id=model_id, device=device)
    int8_model.eval()
    print("      Computing INT8 perplexity...")
    int8_ppl = compute_perplexity(int8_model, tokenizer, corpus, stride, window, device)
    print(f"  INT8 Perplexity : {int8_ppl:.4f}\n")
    del int8_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    delta_abs = int8_ppl - fp16_ppl
    delta_pct = (delta_abs / fp16_ppl) * 100.0
    if abs(delta_abs) < 2.0:
        verdict = "negligible"
    elif abs(delta_abs) < 10.0:
        verdict = "moderate"
    else:
        verdict = "significant"

    print("=" * 62)
    print("  PERPLEXITY COMPARISON RESULT")
    print("=" * 62)
    print(f"  FP16 Perplexity  : {fp16_ppl:.4f}")
    print(f"  INT8 Perplexity  : {int8_ppl:.4f}")
    print(f"  Delta (abs)      : {delta_abs:+.4f}")
    print(f"  Delta (pct)      : {delta_pct:+.2f}%")
    print(f"  Verdict          : {verdict.upper()}")
    print("=" * 62)
    if verdict == "negligible":
        print("\n  -> INT8 causes NEGLIGIBLE accuracy degradation.")
        print("     The VRAM savings are a genuine win with no meaningful quality cost.")
    elif verdict == "moderate":
        print("\n  -> INT8 causes MODERATE accuracy degradation.")
        print("     Trade-off: -42% VRAM at cost of a measurable perplexity increase.")
    else:
        print("\n  -> INT8 causes SIGNIFICANT accuracy degradation.")
        print("     The VRAM savings may not justify quality loss for this model/hardware.")
    print()

    result = {
        "phase": "Phase 4 - INT8 Accuracy Evaluation",
        "model_id": model_id,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "eval_method": "sliding_window_perplexity",
        "corpus_description": "Wikipedia: General Relativity excerpt (CC BY-SA 3.0)",
        "corpus_tokens": corpus_tokens,
        "stride": stride,
        "window": window,
        "fp16_perplexity": round(fp16_ppl, 4),
        "int8_perplexity": round(int8_ppl, 4),
        "perplexity_delta_abs": round(delta_abs, 4),
        "perplexity_delta_pct": round(delta_pct, 2),
        "verdict": verdict,
        "verdict_thresholds": {
            "negligible": "delta_abs < 2.0",
            "moderate": "2.0 <= delta_abs < 10.0",
            "significant": "delta_abs >= 10.0",
        },
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[MicroInfer] Accuracy results -> {OUTPUT_JSON}")
    return result


if __name__ == "__main__":
    run_accuracy_eval()
