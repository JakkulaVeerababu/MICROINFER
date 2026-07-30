"""
MicroInfer - Phase 2: Key-Value (KV) Cache Store Module
Provides pre-allocated CUDA tensor store for managing Key and Value attention projections across layers.
"""

import torch
from typing import Tuple


class KVCache:
    """
    Pre-allocated Key-Value cache tensor store for Transformer attention layers.
    Shape per cache: (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim)
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        batch_size: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.current_len = 0

        # Pre-allocate Key and Value cache tensors
        cache_shape = (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim)
        self.k_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(cache_shape, device=device, dtype=dtype)

    def update(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        """
        Inserts new Key and Value projections for a specific layer.
        
        Args:
            layer_idx (int): Index of the attention layer (0 ... num_layers - 1).
            new_k (torch.Tensor): Key tensor of shape (batch_size, num_kv_heads, num_new_tokens, head_dim).
            new_v (torch.Tensor): Value tensor of shape (batch_size, num_kv_heads, num_new_tokens, head_dim).
        """
        num_new_tokens = new_k.shape[2]
        end_pos = self.current_len + num_new_tokens

        if end_pos > self.max_seq_len:
            raise ValueError(
                f"Exceeded max sequence length capacity! "
                f"Current len ({self.current_len}) + new tokens ({num_new_tokens}) > max_seq_len ({self.max_seq_len})"
            )

        # Slice and assign into pre-allocated memory
        self.k_cache[layer_idx, :, :, self.current_len : end_pos, :] = new_k
        self.v_cache[layer_idx, :, :, self.current_len : end_pos, :] = new_v

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves historical Key and Value tensors for a specific layer up to current_len.
        
        Returns:
            tuple: (k_slice, v_slice) each of shape (batch_size, num_kv_heads, current_len, head_dim)
        """
        k_slice = self.k_cache[layer_idx, :, :, : self.current_len, :]
        v_slice = self.v_cache[layer_idx, :, :, : self.current_len, :]
        return k_slice, v_slice

    def advance(self, num_tokens: int = 1) -> None:
        """
        Advances the sequence length pointer after completing layer updates for a step.
        """
        self.current_len += num_tokens

    def reset(self) -> None:
        """
        Resets cache pointer and clears stored tensors.
        """
        self.current_len = 0
        self.k_cache.zero_()
        self.v_cache.zero_()

    def get_memory_footprint_mb(self) -> float:
        """
        Calculates total memory in Megabytes occupied by the pre-allocated cache tensors.
        """
        bytes_k = self.k_cache.element_size() * self.k_cache.nelement()
        bytes_v = self.v_cache.element_size() * self.v_cache.nelement()
        return round((bytes_k + bytes_v) / (1024 ** 2), 2)


if __name__ == "__main__":
    cache = KVCache(num_layers=28, num_kv_heads=2, head_dim=128, max_seq_len=2048)
    print(f"[MicroInfer] KVCache initialized successfully.")
    print(f"[MicroInfer] Pre-allocated VRAM Footprint: {cache.get_memory_footprint_mb()} MB")
