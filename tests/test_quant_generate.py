"""
MicroInfer - Unit Tests for Sub-Phase 4.2 / 4.3 INT8 Quantized Generator Engine
"""

import sys
import pytest
import torch
from pathlib import Path

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quant_loader import load_quantized_model_and_tokenizer
from src.quant_generate import quant_generate


@pytest.fixture(scope="module")
def loaded_quantized_model_and_tokenizer():
    model, tokenizer = load_quantized_model_and_tokenizer()
    model.eval()
    return model, tokenizer


@pytest.mark.parametrize("prompt", [
    "Artificial Intelligence is transforming software engineering by",
    "Explain how transformer attention mechanism works in simple terms for",
    "What are the key trade-offs between KV-caching and continuous batching in",
])
def test_quant_generate_outputs_valid_text(loaded_quantized_model_and_tokenizer, prompt):
    """
    Verifies that quant_generate produces valid text outputs and positive latency metrics under INT8 precision.
    """
    model, tokenizer = loaded_quantized_model_and_tokenizer
    res = quant_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=15,
        temperature=0.0,
    )

    assert "output_text" in res and len(res["output_text"].strip()) > 0
    assert res["generated_tokens"] == 15
    assert res["ttft_ms"] > 0.0
    assert res["tpot_ms"] > 0.0
    assert res["throughput_tok_per_sec"] > 0.0


def test_quant_generate_temperature_sampling(loaded_quantized_model_and_tokenizer):
    """
    Verifies 8-bit quantized generation under temperature sampling (temperature=0.7).
    """
    model, tokenizer = loaded_quantized_model_and_tokenizer
    res = quant_generate(
        model=model,
        tokenizer=tokenizer,
        prompt="The future of 8-bit neural network quantization",
        max_new_tokens=10,
        temperature=0.7,
    )

    assert "output_text" in res and len(res["output_text"].strip()) > 0
    assert res["generated_tokens"] == 10
