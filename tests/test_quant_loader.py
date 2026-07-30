"""
MicroInfer - Unit Tests for Sub-Phase 4.1 INT8 Quantized Model Loader Module
"""

import sys
import pytest
import torch
from pathlib import Path

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quant_loader import load_quantized_model_and_tokenizer


def test_quant_loader_execution():
    """
    Verifies that load_quantized_model_and_tokenizer successfully loads the model and tokenizer.
    """
    assert torch.cuda.is_available(), "CUDA required for Sub-Phase 4.1 test."

    model, tokenizer = load_quantized_model_and_tokenizer()

    assert model is not None
    assert tokenizer is not None
    assert hasattr(model, "generate")

    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    assert allocated_gb > 0.5, f"Expected non-trivial VRAM allocation, got {allocated_gb} GB"
