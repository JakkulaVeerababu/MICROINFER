import sys
from pathlib import Path

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from src.model_loader import load_model_and_tokenizer, DEFAULT_MODEL_ID


@pytest.fixture(scope="module")
def loaded_model_and_tokenizer():
    """
    Module-scoped fixture to load model and tokenizer once for the test suite.
    """
    model, tokenizer = load_model_and_tokenizer(model_id=DEFAULT_MODEL_ID)
    return model, tokenizer


def test_model_device_and_dtype(loaded_model_and_tokenizer):
    """
    Verifies model parameters are placed on CUDA and in FP16 precision.
    """
    model, _ = loaded_model_and_tokenizer
    first_param = next(model.parameters())

    assert torch.cuda.is_available(), "CUDA must be available for Sub-Phase 0.3 unit test."
    assert first_param.is_cuda, f"Expected model parameters on CUDA, but got {first_param.device}"
    assert first_param.dtype == torch.float16, f"Expected FP16 parameters, but got {first_param.dtype}"
    assert not model.training, "Model must be in evaluation mode (model.eval())."


def test_tokenizer_encoding_decoding(loaded_model_and_tokenizer):
    """
    Verifies tokenizer encodes prompt and decodes back accurately.
    """
    _, tokenizer = loaded_model_and_tokenizer
    prompt = "MicroInfer Transformer Engine"
    
    tokens = tokenizer(prompt, return_tensors="pt")
    assert tokens.input_ids.shape[1] > 0, "Tokenized output must have > 0 tokens."
    
    decoded = tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)
    assert prompt in decoded, f"Expected '{prompt}' in decoded output, got '{decoded}'"


def test_model_forward_pass_shape(loaded_model_and_tokenizer):
    """
    Verifies that running a single forward pass produces logits with expected tensor shape.
    """
    model, tokenizer = loaded_model_and_tokenizer
    device = model.device
    
    prompt = "Testing forward pass shape"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model(**inputs)

    assert hasattr(outputs, "logits"), "Model output must contain logits."
    logits = outputs.logits
    
    # Expected logits shape: (batch_size=1, seq_len, vocab_size)
    assert logits.dim() == 3, f"Expected 3D logits tensor, got {logits.dim()}D"
    assert logits.shape[0] == 1, f"Expected batch size 1, got {logits.shape[0]}"
    assert logits.shape[1] == seq_len, f"Expected sequence length {seq_len}, got {logits.shape[1]}"
    assert logits.shape[2] == model.config.vocab_size, f"Expected vocab size {model.config.vocab_size}, got {logits.shape[2]}"
