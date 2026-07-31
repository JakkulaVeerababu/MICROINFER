"""
MicroInfer - Unified Engine Entry Point
=======================================

Provides a single MicroInferEngine class that ties all four inference phases
together into one callable interface. Accepts a config dict at init specifying
mode, model_id, and hardware parameters; exposes a single generate() method
that routes to the correct backend.

Usage:
    from src.engine import MicroInferEngine

    engine = MicroInferEngine({"mode": "scheduled", "model_id": "Qwen/Qwen2.5-1.5B-Instruct"})
    outputs = engine.generate(["What is a transformer?"], max_new_tokens=64)
    print(outputs[0])

Supported modes:
    "naive"      - Phase 1: Uncached O(N^2) generator (src/naive_generate.py)
    "cached"     - Phase 2: Pre-allocated KV-Cache generator (src/cached_generate.py)
    "scheduled"  - Phase 3: Continuous batch scheduler (src/scheduler.py)
    "quantized"  - Phase 4: INT8 quantized generator (src/quant_generate.py)
"""

import time
import torch
from typing import Any, Dict, List, Optional

from src.model_loader import load_model_and_tokenizer, DEFAULT_MODEL_ID
from src.naive_generate import naive_generate
from src.cached_generate import cached_generate
from src.quant_loader import load_quantized_model_and_tokenizer
from src.quant_generate import quant_generate
from src.scheduler import ContinuousBatchScheduler


class MicroInferEngine:
    """
    Unified inference engine routing prompts to the correct MicroInfer backend.

    Args:
        config (dict): Engine configuration with keys:
            mode         (str):  "naive" | "cached" | "scheduled" | "quantized"
            model_id     (str):  HuggingFace model repository ID
            max_batch_size (int): Maximum concurrent sequences for scheduled mode (default 16)
            max_seq_len  (int):  Maximum sequence length for KV-cache pre-allocation (default 2048)
            device       (str):  "cuda" or "cpu" (default: auto-detect)
    """

    VALID_MODES = {"naive", "cached", "scheduled", "quantized"}

    def __init__(self, config: Dict[str, Any]):
        self.mode = config.get("mode", "cached")
        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be one of: {self.VALID_MODES}"
            )

        self.model_id = config.get("model_id", DEFAULT_MODEL_ID)
        self.max_batch_size = config.get("max_batch_size", 16)
        self.max_seq_len = config.get("max_seq_len", 2048)
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"[MicroInferEngine] Initializing in '{self.mode}' mode on {self.device.upper()}...")

        if self.mode == "quantized":
            self.model, self.tokenizer = load_quantized_model_and_tokenizer(
                model_id=self.model_id,
                device=self.device,
            )
        else:
            self.model, self.tokenizer = load_model_and_tokenizer(
                model_id=self.model_id,
                device=self.device,
            )

        self.model.eval()
        print(f"[MicroInferEngine] Ready. Mode='{self.mode}' | Model='{self.model_id}'")

    def generate(self, prompts: List[str], max_new_tokens: int = 128) -> List[str]:
        """
        Generates text for a list of prompts.

        Args:
            prompts       (list[str]): Input prompt strings.
            max_new_tokens (int):      Maximum tokens to generate per prompt.

        Returns:
            list[str]: Generated output strings in the same order as input prompts.
        """
        if not prompts:
            return []

        if self.mode == "naive":
            return self._generate_naive(prompts, max_new_tokens)
        elif self.mode == "cached":
            return self._generate_cached(prompts, max_new_tokens)
        elif self.mode == "scheduled":
            return self._generate_scheduled(prompts, max_new_tokens)
        elif self.mode == "quantized":
            return self._generate_quantized(prompts, max_new_tokens)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    # ------------------------------------------------------------------
    # Private backend implementations
    # ------------------------------------------------------------------

    def _generate_naive(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        """Phase 1: Uncached O(N^2) sequential generation."""
        results = []
        for prompt in prompts:
            result = naive_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
            results.append(result["output_text"])
        return results

    def _generate_cached(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        """Phase 2: KV-Cache two-phase (Prefill + Decode) sequential generation."""
        results = []
        for prompt in prompts:
            result = cached_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
            results.append(result["output_text"])
        return results

    def _generate_scheduled(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        """
        Phase 3: Continuous batch scheduler with lifecycle management.

        Submits all prompts to the scheduler simultaneously and drains to
        completion. Results are re-ordered to match the input prompt order
        using seq_id tracking.
        """
        scheduler = ContinuousBatchScheduler(
            max_batch_size=self.max_batch_size,
            device=self.device,
        )

        # Track seq_id -> prompt index mapping for ordering
        seq_id_to_idx: Dict[int, int] = {}
        for idx, prompt in enumerate(prompts):
            seq = scheduler.add_request(
                prompt,
                self.tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
            seq_id_to_idx[seq.seq_id] = idx

        # Drain the scheduler
        while scheduler.has_pending_work():
            scheduler.step(self.model, self.tokenizer)

        # Reconstruct ordered outputs
        results = [""] * len(prompts)
        for seq in scheduler.finished_sequences:
            idx = seq_id_to_idx.get(seq.seq_id, -1)
            if idx >= 0:
                results[idx] = self.tokenizer.decode(
                    seq.generated_tokens, skip_special_tokens=True
                )
        return results

    def _generate_quantized(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        """Phase 4: INT8 quantized sequential generation."""
        results = []
        for prompt in prompts:
            result = quant_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
            results.append(result["output_text"])
        return results


# ---------------------------------------------------------------------------
# Smoke-test __main__ block: runs all 4 modes on a single prompt
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_PROMPT = "What is a transformer in machine learning?"
    MAX_NEW_TOKENS = 32

    print("\n" + "=" * 70)
    print("  MicroInferEngine Smoke Test — All 4 Modes")
    print("=" * 70)

    for mode in ["naive", "cached", "scheduled", "quantized"]:
        print(f"\n[{mode.upper()}] Initializing engine...")
        t0 = time.perf_counter()
        engine = MicroInferEngine({"mode": mode, "model_id": DEFAULT_MODEL_ID})
        outputs = engine.generate([TEST_PROMPT], max_new_tokens=MAX_NEW_TOKENS)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[{mode.upper()}] Output  : '{outputs[0][:80]}...'")
        print(f"[{mode.upper()}] Latency : {elapsed_ms:.0f} ms (load + generate)")
        del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("  Smoke test complete — all 4 modes operational.")
    print("=" * 70)
