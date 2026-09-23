from pathlib import Path

import numpy as np


def load(path: str | Path) -> dict[str, np.ndarray]:
    """Parse a .safetensors file by hand and return every tensor as a float32 numpy array (upcast from BF16/F16)."""
    raise NotImplementedError("Phase 1.1: safetensors loader")
