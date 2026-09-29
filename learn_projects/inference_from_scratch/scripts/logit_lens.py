"""Logit lens: after each layer, ask "which word would the model say if it stopped here?"

Reproduces the table in knowledge/inference_textbook/06_layers_and_the_forward_pass.md.

    python scripts/logit_lens.py
    python scripts/logit_lens.py "The largest planet is" " Jupiter"
"""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen2.5-0.5b"


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
    target = sys.argv[2] if len(sys.argv) > 2 else " Paris"

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()

    ids = tok(prompt, return_tensors="pt").input_ids
    target_id = tok.encode(target)[0]
    with torch.no_grad():
        hidden = model(ids, output_hidden_states=True).hidden_states

    print(f"prompt {prompt!r}, tracking {tok.decode([target_id])!r}\n")
    print(f"{'layer':>5}  {'rank':>7}  top 3 guesses")
    for layer, h in enumerate(hidden):
        row = h[0, -1]
        # HF applies the final norm to the last entry already; earlier ones need it for the lens.
        if layer < len(hidden) - 1:
            row = model.model.norm(row)
        logits = model.lm_head(row)
        rank = int((logits > logits[target_id]).sum()) + 1
        top = [repr(tok.decode([t])) for t in logits.topk(3).indices]
        print(f"{layer:>5}  {rank:>7,}  {'  '.join(top)}")


if __name__ == "__main__":
    main()
