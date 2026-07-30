"""
MicroInfer - Phase 4: INT8 Quantized Model Loader Module
Loads target LLM with 8-bit quantized weights using bitsandbytes / PyTorch INT8.
"""

import time
import torch
from pathlib import Path
from typing import Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.model_loader import DEFAULT_MODEL_ID


def load_quantized_model_and_tokenizer(
    model_id: str = DEFAULT_MODEL_ID,
    load_in_8bit: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[Any, Any]:
    """
    Loads target LLM with INT8 quantized weight precision.
    """
    print(f"\n[MicroInfer] Loading tokenizer for '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print(f"[MicroInfer] Loading 8-bit quantized model '{model_id}' on {device}...")
    t_start = time.time()

    if load_in_8bit and torch.cuda.is_available():
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[Warning] bitsandbytes 8-bit config failed ({e}). Falling back to FP16 half precision.")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device != "cuda" else "auto",
            trust_remote_code=True,
        )

    t_elapsed = time.time() - t_start
    print(f"[MicroInfer] Quantized model loaded successfully in {t_elapsed:.2f}s.")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[MicroInfer] VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB\n")

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_quantized_model_and_tokenizer()
