"""python -m nanoinfer.generate "The capital of France is" --max-tokens 20"""
import argparse
import time
from pathlib import Path

import numpy as np

from nanoinfer import safetensors
from nanoinfer.model import Config, Qwen2
from nanoinfer.sampling import sample
from nanoinfer.tokenizer import Tokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen2.5-0.5b"
EOS_TOKEN_ID = 151643


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tok = Tokenizer.from_file(MODEL_DIR / "tokenizer.json")
    model = Qwen2(Config.from_json(MODEL_DIR / "config.json"), safetensors.load(MODEL_DIR / "model.safetensors"))
    rng = np.random.default_rng(args.seed)

    ids = tok.encode(args.prompt)
    print(args.prompt, end="", flush=True)
    n_prompt = len(ids)
    t0 = time.perf_counter()
    for _ in range(args.max_tokens):
        # No KV cache yet: the whole sequence is recomputed every step. Phase 2 fixes this.
        logits, _ = model.forward(ids)
        next_id = sample(logits[-1], args.temperature, args.top_k, args.top_p, rng)
        if next_id == EOS_TOKEN_ID:
            break
        ids.append(next_id)
        print(tok.decode([next_id]), end="", flush=True)
    dt = time.perf_counter() - t0
    n_new = len(ids) - n_prompt
    print(f"\n\n[{n_new} tokens in {dt:.2f}s = {n_new / dt:.2f} tok/s]")


if __name__ == "__main__":
    main()
