import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        raw = json.loads(Path(path).read_text())
        return cls(**{name: raw[name] for name in cls.__dataclass_fields__})


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """x: (seq, hidden) -> (seq, hidden)."""
    raise NotImplementedError("Phase 1.3: RMSNorm")


def rope_tables(head_dim: int, max_positions: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (cos, sin), each (max_positions, head_dim)."""
    raise NotImplementedError("Phase 1.4: RoPE frequency tables")


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """x: (n_heads, seq, head_dim); cos/sin: (seq, head_dim). Qwen uses the rotate-half layout, not interleaved pairs."""
    raise NotImplementedError("Phase 1.4: apply RoPE")


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Causal grouped-query attention. q: (n_heads, seq, d); k, v: (n_kv_heads, seq, d) -> (n_heads, seq, d)."""
    raise NotImplementedError("Phase 1.5: causal GQA")


def swiglu_mlp(x: np.ndarray, w_gate: np.ndarray, w_up: np.ndarray, w_down: np.ndarray) -> np.ndarray:
    """x: (seq, hidden). Weights use the PyTorch Linear layout (out_features, in_features)."""
    raise NotImplementedError("Phase 1.6: SwiGLU MLP")


class Qwen2:
    def __init__(self, config: Config, weights: dict[str, np.ndarray]):
        self.config = config
        self.w = weights

    def forward(self, token_ids: list[int]) -> tuple[np.ndarray, list[np.ndarray]]:
        """Return (logits (seq, vocab), hidden states).

        Hidden states follow the Hugging Face convention so tests can pinpoint the first wrong layer:
        [embeddings, output of layer 0, ..., output of layer L-2, final_norm(output of layer L-1)].
        """
        raise NotImplementedError("Phase 1.7: full forward pass")
