from pathlib import Path


class Tokenizer:
    """Byte-level BPE tokenizer driven by a Hugging Face tokenizer.json (vocab, merges, pre-tokenizer regex)."""

    @classmethod
    def from_file(cls, path: str | Path) -> "Tokenizer":
        raise NotImplementedError("Phase 1.2: parse tokenizer.json")

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError("Phase 1.2: BPE encode")

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError("Phase 1.2: BPE decode")
