from pathlib import Path
import json
import struct

HERE = Path(__file__).resolve().parent  # = .../inference_from_scratch/scripts
MODEL = HERE.parent / "models" / "qwen2.5-0.5b" / "model.safetensors"

with open(MODEL, "rb") as f:
    b = f.read(8)
    n = int.from_bytes(b, "little")
    print("N =", n)
    assert n == 32280

    header_bytes = f.read(n)
    header = json.loads(header_bytes)
    print("entries:", len(header))
    print(header["model.embed_tokens.weight"])

    for name in header:
        if not name.startswith("model"):
            print(name, header[name])

    w = f.read(2)
    print(w.hex(" "))

    w = bytes(2) + w  # zeros first: in little-endian the first bytes are the least significant (bottom half)
    value = struct.unpack("<f", w)[0]
    print(value)



