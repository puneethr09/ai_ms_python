"""Run the Hugging Face model in fp32 and save ground truth for the Phase 1 tests."""
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "qwen2.5-0.5b"
OUT = ROOT / "fixtures" / "reference.npz"

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n",
]

TOKENIZER_CASES = [
    "Hello, world!",
    "  leading spaces and\ttabs\n\nnewlines",
    "Numbers: 1234567 3.14159 -42",
    "Unicode: naïve café 日本語 🚀",
    "don't won't I'm they're",
    "x = [i**2 for i in range(10)]  # comment",
]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32, attn_implementation="eager")
    model.eval()

    arrays: dict[str, np.ndarray] = {}
    for i, case in enumerate(TOKENIZER_CASES):
        arrays[f"tok_text_{i}"] = np.array(case)
        arrays[f"tok_ids_{i}"] = np.array(tok.encode(case), dtype=np.int64)

    for i, prompt in enumerate(PROMPTS):
        ids = tok.encode(prompt)
        with torch.no_grad():
            out = model(torch.tensor([ids]), output_hidden_states=True)
        arrays[f"prompt_text_{i}"] = np.array(prompt)
        arrays[f"prompt_ids_{i}"] = np.array(ids, dtype=np.int64)
        arrays[f"logits_{i}"] = out.logits[0].numpy()
        # hidden_states[0] = embeddings, [k] = output of layer k-1, last entry has the final RMSNorm applied.
        arrays[f"hidden_{i}"] = torch.stack(out.hidden_states)[:, 0].numpy()

    OUT.parent.mkdir(exist_ok=True)
    np.savez(OUT, **arrays)
    print(f"saved {len(arrays)} arrays -> {OUT}")


if __name__ == "__main__":
    main()
