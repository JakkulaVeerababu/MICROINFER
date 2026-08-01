"""
MicroInfer - Phase 2: Key-Value (KV) Cache Store Module
Provides pre-allocated CUDA tensor store for managing Key and Value attention projections across layers.
Subclasses HuggingFace's Cache interface so it can be passed directly to model(..., past_key_values=...)
while pre-allocating contiguous 5D CUDA tensors up front.
"""

import torch
from torch.profiler import record_function
from typing import Tuple, Optional, Any
try:
    from transformers.cache_utils import Cache, CacheLayerMixin
except ImportError:
    from transformers.cache_utils import Cache
    class CacheLayerMixin:
        """Fallback mixin for transformers versions without CacheLayerMixin."""
        pass


class KVCacheLayer(CacheLayerMixin):
    """
    Individual attention layer cache operating on a pre-allocated CUDA tensor slice.
    """

    is_sliding = False

    def __init__(self, k_slice: torch.Tensor, v_slice: torch.Tensor):
        super().__init__()
        self.k_slice = k_slice  # (batch_size, num_kv_heads, max_seq_len, head_dim)
        self.v_slice = v_slice  # (batch_size, num_kv_heads, max_seq_len, head_dim)
        self._seq_len = 0
        self.is_initialized = True

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        pass

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with record_function("KVCacheLayer::update"):
            num_new_tokens = key_states.shape[2]
            start_pos = self._seq_len
            end_pos = start_pos + num_new_tokens

            max_len = self.k_slice.shape[2]
            if end_pos > max_len:
                raise ValueError(
                    f"Exceeded max sequence length capacity! "
                    f"Current len ({self._seq_len}) + new tokens ({num_new_tokens}) > max_seq_len ({max_len})"
                )

            self.k_slice[:, :, start_pos:end_pos, :] = key_states
            self.v_slice[:, :, start_pos:end_pos, :] = value_states
            self._seq_len = end_pos
            self._last_updated_len = end_pos

            return self.k_slice[:, :, :end_pos, :], self.v_slice[:, :, :end_pos, :]

    def get_mask_sizes(self, query_length: int) -> Tuple[int, int]:
        return self._seq_len + query_length, 0

    def get_seq_length(self) -> int:
        return self._seq_len

    def get_max_cache_shape(self) -> int:
        return self.k_slice.shape[2]

    def reset(self) -> None:
        self._seq_len = 0
        self._last_updated_len = None
        self.k_slice.zero_()
        self.v_slice.zero_()


class KVCache(Cache):
    """
    Pre-allocated Key-Value cache tensor store for Transformer attention layers.
    Shape per cache: (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim)
    Implements HuggingFace's Cache interface while pre-allocating memory upfront.
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

        # Pre-allocate Key and Value 5D cache tensors
        cache_shape = (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim)
        self.k_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(cache_shape, device=device, dtype=dtype)

        layers = [
            KVCacheLayer(self.k_cache[i], self.v_cache[i])
            for i in range(num_layers)
        ]
        try:
            super().__init__(layers=layers)
        except TypeError:
            super().__init__()
        self.layers = layers

    @property
    def current_len(self) -> int:
        if self.layers:
            return self.layers[0].get_seq_length()
        return 0

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.current_len

    def get_max_length(self) -> Optional[int]:
        return self.max_seq_len

    @current_len.setter
    def current_len(self, val: int) -> None:
        for layer in self.layers:
            layer._seq_len = val
            layer._last_updated_len = None

    def update(
        self,
        key_states: Any = None,
        value_states: Any = None,
        layer_idx: Any = None,
        cache_kwargs: Any = None,
        **kwargs,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Supports two call patterns:

        1. HuggingFace Cache protocol (primary, called 28× per decode step by the model):
               update(key_states, value_states, layer_idx, ...)
           where key_states and value_states are CUDA tensors and layer_idx is an int.

        2. Legacy MicroInfer direct-call signature (used in tests and diagnostics):
               update(layer_idx=0, new_k=k, new_v=v)
           Both keyword arguments are forwarded through **kwargs and handled below.
        """
        with record_function("KVCache::update"):
            # ---- Fast path: HF positional protocol (tensor, tensor, int) ----
            if torch.is_tensor(key_states) and torch.is_tensor(value_states):
                idx = layer_idx if isinstance(layer_idx, int) else 0
                return self.layers[idx].update(key_states, value_states)

            # ---- Legacy keyword path: update(layer_idx=N, new_k=k, new_v=v) ----
            new_k = kwargs.get("new_k", None)
            new_v = kwargs.get("new_v", None)
            kw_key = kwargs.get("key_states", None)
            kw_val = kwargs.get("value_states", None)

            k_tensor = kw_key if kw_key is not None else new_k
            v_tensor = kw_val if kw_val is not None else new_v

            # Resolve layer index: first check positional layer_idx, then kwargs
            if isinstance(layer_idx, int):
                idx = layer_idx
            elif isinstance(key_states, int):
                # Positional legacy: update(layer_idx, new_k, new_v) with all positional
                idx = key_states
                k_tensor = value_states
                v_tensor = layer_idx if isinstance(layer_idx, torch.Tensor) else k_tensor
            else:
                idx = kwargs.get("layer_idx", 0)

            return self.layers[idx].update(k_tensor, v_tensor)

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves historical Key and Value tensors for a specific layer up to current_len.
        """
        seq_len = self.layers[layer_idx].get_seq_length()
        k_slice = self.k_cache[layer_idx, :, :, :seq_len, :]
        v_slice = self.v_cache[layer_idx, :, :, :seq_len, :]
        return k_slice, v_slice

    def advance(self, num_tokens: int = 1) -> None:
        """
        Advances sequence length pointer across layers if not already advanced during update.
        """
        for layer in self.layers:
            if getattr(layer, "_last_updated_len", None) == layer._seq_len:
                pass
            else:
                layer._seq_len += num_tokens
            layer._last_updated_len = None

    def reset(self) -> None:
        """
        Resets cache pointers and clears stored tensors.
        """
        for layer in self.layers:
            layer.reset()
        self.k_cache.zero_()
        self.v_cache.zero_()

    def get_memory_footprint_mb(self) -> float:
        """
        Calculates total memory in Megabytes occupied by the pre-allocated cache tensors.
        """
        bytes_k = self.k_cache.element_size() * self.k_cache.nelement()
        bytes_v = self.v_cache.element_size() * self.v_cache.nelement()
        return round((bytes_k + bytes_v) / (1024 ** 2), 2)

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        max_seq_len: int = 2048,
        batch_size: int = 1,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KVCache":
        """
        Constructs a pre-allocated KVCache instance matching a HuggingFace model config.
        """
        config = model.config
        num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 28))
        num_kv_heads = getattr(config, "num_key_value_heads", getattr(config, "n_head", 2))
        hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", 1536))
        num_attn_heads = getattr(config, "num_attention_heads", getattr(config, "n_head", 12))
        head_dim = hidden_size // num_attn_heads

        param = next(model.parameters(), None)
        target_device = device if device is not None else (str(param.device) if param is not None else "cuda")
        target_dtype = dtype if dtype is not None else (param.dtype if param is not None else torch.float16)

        return cls(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            device=target_device,
            dtype=target_dtype,
        )


if __name__ == "__main__":
    cache = KVCache(num_layers=28, num_kv_heads=2, head_dim=128, max_seq_len=2048)
    print(f"[MicroInfer] KVCache initialized successfully.")
    print(f"[MicroInfer] Pre-allocated VRAM Footprint: {cache.get_memory_footprint_mb()} MB")
