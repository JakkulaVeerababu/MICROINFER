"""
MicroInfer - Unit Tests for Sub-Phase 2.1 KV-Cache Tensor Store Module
"""

import sys
import pytest
import torch
from pathlib import Path

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kv_cache import KVCache


def test_kv_cache_initialization():
    """
    Verifies KVCache initialization, tensor shapes, and memory footprint math.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = KVCache(
        num_layers=28,
        num_kv_heads=2,
        head_dim=128,
        max_seq_len=2048,
        batch_size=1,
        device=device,
        dtype=torch.float16,
    )

    assert cache.current_len == 0
    assert cache.k_cache.shape == (28, 1, 2, 2048, 128)
    assert cache.v_cache.shape == (28, 1, 2, 2048, 128)
    
    # 2 * 28 * 1 * 2 * 2048 * 128 * 2 bytes = 58,720,256 bytes = 56.0 MB
    assert cache.get_memory_footprint_mb() == 56.0


def test_kv_cache_update_and_get():
    """
    Verifies update, get, advance, and reset operations.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = KVCache(
        num_layers=4,
        num_kv_heads=2,
        head_dim=64,
        max_seq_len=128,
        batch_size=1,
        device=device,
        dtype=torch.float16,
    )

    # 1. Prefill update (3 tokens)
    new_k = torch.ones((1, 2, 3, 64), device=device, dtype=torch.float16) * 1.5
    new_v = torch.ones((1, 2, 3, 64), device=device, dtype=torch.float16) * 2.5
    
    cache.update(layer_idx=0, new_k=new_k, new_v=new_v)
    cache.advance(num_tokens=3)
    
    assert cache.current_len == 3
    k_slice, v_slice = cache.get(layer_idx=0)
    assert k_slice.shape == (1, 2, 3, 64)
    assert v_slice.shape == (1, 2, 3, 64)
    assert torch.allclose(k_slice, new_k)
    assert torch.allclose(v_slice, new_v)

    # 2. Decode update (1 token)
    step_k = torch.ones((1, 2, 1, 64), device=device, dtype=torch.float16) * 3.5
    step_v = torch.ones((1, 2, 1, 64), device=device, dtype=torch.float16) * 4.5

    cache.update(layer_idx=0, new_k=step_k, new_v=step_v)
    cache.advance(num_tokens=1)

    assert cache.current_len == 4
    k_slice_2, v_slice_2 = cache.get(layer_idx=0)
    assert k_slice_2.shape == (1, 2, 4, 64)
    assert v_slice_2.shape == (1, 2, 4, 64)

    # 3. Reset
    cache.reset()
    assert cache.current_len == 0
    k_empty, v_empty = cache.get(layer_idx=0)
    assert k_empty.shape == (1, 2, 0, 64)


def test_kv_cache_overflow():
    """
    Verifies that attempting to insert past max_seq_len raises ValueError.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = KVCache(
        num_layers=2,
        num_kv_heads=2,
        head_dim=32,
        max_seq_len=5,
        batch_size=1,
        device=device,
        dtype=torch.float16,
    )

    new_k = torch.zeros((1, 2, 6, 32), device=device, dtype=torch.float16)
    new_v = torch.zeros((1, 2, 6, 32), device=device, dtype=torch.float16)

    with pytest.raises(ValueError):
        cache.update(layer_idx=0, new_k=new_k, new_v=new_v)
