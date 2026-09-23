import numpy as np


def sample(logits: np.ndarray, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
           rng: np.random.Generator | None = None) -> int:
    """logits: (vocab,). temperature 0 means greedy. top_k 0 and top_p 1.0 disable those filters."""
    raise NotImplementedError("Phase 1.8: sampling")
