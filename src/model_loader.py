"""
MicroInfer - Model Loader Module
Provides unified loading functionality for open-weight Transformer models.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def load_model_and_tokenizer(
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype: torch.dtype = torch.float16,
):
    """
    Loads a pretrained model and tokenizer onto the specified device.
    
    Args:
        model_id (str): HuggingFace model repository ID.
        device (str): Destination device ('cuda' or 'cpu').
        torch_dtype (torch.dtype): Precision for model parameters (default float16).
        
    Returns:
        tuple: (model, tokenizer)
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[MicroInfer] Loading model '{model_id}' on {device} ({torch_dtype})...")
    start_time = time.perf_counter()
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
            local_files_only=True,
        )
    model.eval()
    
    load_time = time.perf_counter() - start_time
    print(f"[MicroInfer] Model loaded successfully in {load_time:.2f}s.")
    
    if torch.cuda.is_available() and device.startswith("cuda"):
        vram_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[MicroInfer] VRAM Allocated: {vram_allocated:.2f} GB | Reserved: {vram_reserved:.2f} GB")

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    print("[MicroInfer] Model Architecture:", type(model).__name__)
