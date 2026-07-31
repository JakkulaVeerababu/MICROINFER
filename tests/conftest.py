import gc
import pytest
import torch

@pytest.fixture(autouse=True)
def cleanup_vram():
    """Autouse fixture to release VRAM and clear CUDA cache between tests."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
