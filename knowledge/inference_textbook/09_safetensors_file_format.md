# Chapter 9: The Model File (safetensors), Lesson 1.1

> [← Chapter 8](08_bandwidth_and_speed.md) · [Contents](README.md) · [Self-test →](self_test.md)

**Reading time:** about 20 minutes. **Status:** 🟡 in progress. Exercise A is done; the tools class, Exercise B and Exercise C (your code) are next.

## Where we are on the map

```
Phase 0 ✅ measured the pipe: 92 GB/s
Phase 1: build the transformer in numpy
  ▶ 1.1 read the weights file   ← here
    1.2 tokenizer (text → IDs)
    1.3 rms_norm, MLP
    1.4 RoPE (positions)
    1.5 attention
    1.6 full forward pass
    1.7 sampling (scores → word)
```

Everything in chapters 2–7 (Wq, Wk, Wv, Wo, gate, up, down, the dictionary) is just numbers in a file on disk. Before any math can run, they must be in memory. Step 1.1 is the doorway: **turn a file into numpy arrays.**

## In one minute

- `model.safetensors` = **8 bytes** (header length N, little-endian) + **N bytes of JSON** (table of contents) + **raw numbers**.
- For your file: N = **32,280**, so the data starts at byte **32,288**. 290 tensors, all BF16.
- **Little-endian** means reverse the **bytes**, not the hex digits.
- **BF16 is the top 2 bytes of an FP32.** To convert, put two zero bytes underneath.

---

## 0. The model folder: which file is which

The download gives you a folder, not one file. The files are listed here in the order a word passes through them:

```
models/qwen2.5-0.5b/
├── config.json              681 B   the model's shape: the "blueprint"
├── generation_config.json   138 B   default settings for the generation loop
├── tokenizer.json           7.0 MB  text ↔ token IDs: the full rulebook
├── vocab.json               2.8 MB  ┐ the same tokenizer, split into two
├── merges.txt               1.7 MB  ┘ older-style files (redundant)
├── tokenizer_config.json    7.2 KB  special tokens + the chat template
├── model.safetensors        988 MB  the 494M weights: the only big file
└── .cache/                          download bookkeeping (ignore)
```

| File | What it is | Used in step |
| :--- | :--- | :--- |
| `config.json` | The shape numbers: `hidden_size: 896`, `num_hidden_layers: 24`, `num_attention_heads: 14`, `num_key_value_heads: 2`, `intermediate_size: 4864`, `rope_theta: 1000000`, `rms_norm_eps: 1e-06`, `tie_word_embeddings: true`. It's plain text and contains **no weights**. | 1.6 |
| `model.safetensors` | The weights: 8-byte length, JSON header, raw BF16 numbers (this chapter) | **1.1** |
| `tokenizer.json` | 151,643 regular tokens, 151,387 merge rules, 22 special tokens, and the regex that splits text ([Chapter 2](02_tokens_and_embeddings.md)) | 1.2 |
| `vocab.json` + `merges.txt` | The same tokenizer in the older GPT-2 format. You only need one of them. | — |
| `tokenizer_config.json` | Which IDs are special (end-of-text = 151643), and the **chat template** (`<\|im_start\|>user…`) | Later (chat, serving) |
| `generation_config.json` | Loop defaults: `do_sample: false` (always pick the top word), stop at ID 151643, at most 2048 new tokens | 1.7 |

**How to tell them apart at a glance:**
- **Size.** The weights are always the huge one.
- **Extension.** `.json` and `.txt` are text: open them in the editor. `.safetensors` is binary, so you need code or `xxd` to read it (`head -c 8 model.safetensors | xxd`).

**Spare dictionary rows:** `config.json` says `vocab_size: 151936`, but the tokenizer has only 151,643 + 22 = 151,665 tokens. That leaves 271 rows that no text can ever produce. 151,936 = 1,187 × 128: the row count is a multiple of 128, which suits GPU matrix shapes, and it leaves room to add special tokens later without changing the tensor shapes.

> **Lesson from Exercise B:** the first attempt opened `config.json` instead of `model.safetensors`. Python didn't complain: it turned the text `{\n  "arc` into the integer 7,165,896,756,295,633,531. Code that reads bytes never errors on the wrong file. **Always check against a number you already know** (`assert n == 32280`).

## 1. The file's layout

```
┌──────────┬────────────────────┬──────────────────┐
│ 8 bytes  │ N bytes of JSON    │ raw numbers,     │
│ = N      │ = table of contents│ back to back     │
└──────────┴────────────────────┴──────────────────┘
```

A table-of-contents entry:

```json
"model.layers.0.self_attn.k_proj.weight": {
  "dtype": "BF16",
  "shape": [128, 896],
  "data_offsets": [start, end]
}
```

That's Wk for layer 0: 896 → 128, as in [Chapter 4](04_attention.md). Offsets are counted **from the start of the data section** (byte 8 + N), not from the start of the file. There's also a `__metadata__` key (`{"format": "pt"}`) that is not a tensor.

Facts about your real file:

| Fact | Value |
| :--- | ---: |
| First 8 bytes | `18 7e 00 00 00 00 00 00` |
| N (header length) | 32,280 |
| Data starts at | 32,288 |
| Tensors | 290 (24 layers × 12 + embedding + final norm) |
| dtype | all BF16 |
| Data bytes | 988,065,536 |
| File size = 8 + N + data | 988,097,824 ✅ |
| `lm_head.weight` present? | No: tied to the embedding ([Chapter 2](02_tokens_and_embeddings.md)) |

## 2. Little-endian: the lesson from Exercise A

Your first answer was N = `e781` = 59,265. The first 8 bytes are `18 7e 00 00 00 00 00 00`. You reversed the **hex digits** (`187e` → `e781`). Little-endian means reversing the **bytes**, and each byte's two hex digits stay together:

```
bytes in file:  [18] [7e] [00] [00] [00] [00] [00] [00]
reversed:       [00] [00] [00] [00] [00] [00] [7e] [18]
value:          0x7e18 = 7×4096 + 14×256 + 1×16 + 8
              = 32,280
```

**Why it matters:** the smallest addressable unit of memory is a byte. Little-endian puts the least significant **byte** at the lowest address. Get it wrong in a loader and every offset shifts: you read garbage that still looks like valid floats. That's the worst kind of bug, because nothing crashes.

Practice (you answered this correctly): `01 02 00 00` → reversed `00 00 02 01` → 0x0201 = 2×256 + 1 = **513**.

## 3. A toy file, byte by byte

One tensor, `x = [1.0, 2.0]`, in BF16:

```
bytes 0–7:   37 00 00 00 00 00 00 00
             → N = 0x37 = 55
bytes 8–62:  {"x":{"dtype":"BF16","shape":[2],
              "data_offsets":[0,4]}}   (55 bytes)
bytes 63–66: 80 3F 00 40
             → 0x3F80 = 1.0, 0x4000 = 2.0
```

Reading any safetensors file:
1. Read 8 bytes → decode as a little-endian unsigned 64-bit integer → N.
2. Read N bytes → parse as JSON → the table of contents.
3. For each tensor: take bytes `[start, end)` counted from byte 8 + N, turn them into numbers, give them the right shape.

## 4. BF16: the top half of an FP32

An FP32 is 4 bytes: 1 sign bit, 8 exponent bits, 23 fraction bits. **BF16 keeps the first 16 bits** (sign, all 8 exponent bits, 7 fraction bits) and drops the rest. Same range as FP32, less precision.

```
1.0 as FP32 = 0x3F80_0000
1.0 as BF16 = 0x3F80        (top half)
2.0 as FP32 = 0x4000_0000 → BF16 0x4000
```

To go back: **glue two zero bytes underneath** (shift left by 16 bits). numpy has no bfloat16 type, so you'll do this with integer arrays and a `.view`. That's the tools class.

### 4a. Reading a BF16 by hand: the embedding's first weight

**Step 1: the bytes on disk.** At byte 32,288 (the start of the embedding) the file holds `25 bc 2c 3d 22 3c …`. Each weight is 2 bytes, so the first one is `25 bc`.

**Step 2: undo little-endian.** The first byte is the least significant, so the 16-bit number is `0xBC25`.

**Step 3: write out the 16 bits and cut them into three fields.**

```
0xBC25 =  1    0111 1000    010 0101
          │    └ 8 bits ┘   └ 7 bits ┘
        sign    exponent     fraction
         =1     =120         =37
```

**Step 4: apply the rule.**

```
value = (−1)^sign × (1 + fraction/128) × 2^(exponent − 127)
      = −1        × (1 + 37/128)       × 2^(120 − 127)
      = −1        × 1.2890625          × 1/128
      = −0.01007080078125
```

- **sign:** 0 means positive, 1 means negative.
- **exponent:** stored with a **bias of 127**, so that negative powers (small numbers) need no sign bit of their own. A stored 120 means 2^−7. A stored 127 means 2^0 = 1.
- **fraction:** the digits after an **implicit leading 1**. 7 bits give 128 steps between one power of 2 and the next: 1, 1+1/128, 1+2/128, …

It's scientific notation in base 2: `−1.2890625 × 2⁻⁷`, just like `−1.007 × 10⁻²`.

The next two weights, the same way:

| Bytes | 16-bit | sign | exp | frac | Value |
| :--- | :--- | :-: | :-: | :-: | ---: |
| `25 bc` | `0xBC25` | 1 | 120 | 37 | −(1+37/128) × 2⁻⁷ = **−0.010071** |
| `2c 3d` | `0x3D2C` | 0 | 122 | 44 | +(1+44/128) × 2⁻⁵ = **+0.041992** |
| `22 3c` | `0x3C22` | 0 | 120 | 34 | +(1+34/128) × 2⁻⁷ = **+0.009888** |

These match the reference loader's `[-0.0101, 0.0420, 0.0099]`.

### 4b. Converting to FP32: why "add zeros underneath" is exact

FP32 uses **the same rule** with wider fields: 1 sign bit, the **same 8 exponent bits with the same bias 127**, and 23 fraction bits (so it divides by 2²³ instead of 2⁷).

```
BF16:  1 01111000 0100101
FP32:  1 01111000 0100101 0000000000000000
       └─ the same 16 bits ─┘└─ 16 new zeros ─┘
```

Field by field:
- **sign:** the same bit, 1.
- **exponent:** the same 8 bits, 120. That's why BF16 and FP32 have the same range.
- **fraction:** `0100101` followed by 16 zeros. As a 23-bit fraction that's 37 × 2¹⁶ / 2²³ = 37/128. The zeros add nothing, just as 0.37 = 0.3700000.

So the value is identical, and the conversion is exact: no rounding, nothing gained, nothing lost. As an operation it's `0xBC25 << 16 = 0xBC250000`.

**On disk (little-endian), the FP32 bytes are `00 00 25 bc`:** the new zero bytes are the least significant, so they come **first**. That's why the byte-level version is `bytes(2) + w`, not `w + bytes(2)`.

The reverse (FP32 → BF16) *does* lose information, because it throws away the bottom 16 fraction bits. Real converters round rather than chop. You only need the lossless direction in Phase 1.

### 4c. Why floats are built this way (and not "just store the number")

16 bits give only 65,536 patterns. The question is which numbers they stand for.

- **Plain integers** (0 to 65,535): no fractions at all, so −0.0101 is impossible.
- **Fixed point** (say, always `integer / 10,000`): evenly spaced steps of 0.0001. A weight of 0.00003 **vanishes** to 0, 0.000071 is off by 40%, and an activation of 12.5 **doesn't fit** (the maximum is about ±3.2).
- **Floating point** = scientific notation in base 2: the **exponent says how big**, the **fraction says which digits**. The spacing grows with the number's size, so the **relative** error is about the same (~0.4% for BF16) everywhere from 10⁻³⁸ to 10³⁸. Neural nets have tiny weights and large activations together, so this is what they need.

Why each piece of the rule looks the way it does:
- **`fraction / 128`**: the digits after the binary point. 7 bits → 2⁷ = 128, just like 3 decimal digits `289` mean 289/1000. FP32 has 23 bits, so it divides by 2²³; that's the whole precision difference.
- **`1 +`**: in binary scientific notation the leading digit is always 1, so it isn't stored (a free extra bit).
- **`exponent − 127`** (the **bias**): the 8-bit field stores 0–255, but negative powers are needed too. Shifting by 127 gives −126 to +127 without a separate sign bit, and larger floats get larger bit patterns, so hardware can compare them almost like integers.

| Format | Exponent bits | Fraction bits | Max value | Digits |
| :--- | :-: | :-: | :--- | :-: |
| FP16 | 5 | 10 | 65,504 | ~3.3 |
| BF16 | 8 | 7 | ~3.4 × 10³⁸ (same as FP32) | ~2.4 |

BF16 gave up digits so that it never overflows: for neural nets, **range matters more than digits**.

**Looking ahead (Phase 3):** INT8/INT4 quantization goes back to fixed point, which is small and fast but has the "vanishes / doesn't fit" problem. It fixes that with one **scale per block** of weights: a shared exponent.

**Special cases** (for recognising them, not needed in Phase 1): exponent all zeros with fraction 0 is ±0.0 (smaller values with exponent 0 are "subnormals", which have no implicit 1). Exponent all ones (255) means ±infinity or NaN.

## 5. Why this format, and why it matters for inference

- **Safety:** the older `.bin`/`.pt` format is a Python **pickle**, and loading one can run arbitrary code: a downloaded model could run code on your machine. safetensors is pure data. That's why it became the standard.
- **Zero-copy:** because the numbers are raw and packed, a real engine can `mmap` the file and use the tensors in place, without copying (chapters 02 and 05 of the systems book). llama.cpp and vLLM do this; llama.cpp loads a 4.68 GB model in 624 ms this way.
- **Two different 988 MB costs:**
  - **Loading:** once, from SSD at ~3 GB/s ≈ 0.3 s.
  - **Decoding:** 988 MB from RAM **for every word** (10.7 ms each). The loader is paid once; the forward pass is paid on every token.
- **FP32 doubles memory:** after your conversion the model takes ~2 GB in RAM and ~2 GB of traffic per word, halving the ceiling. Fine for Phase 1 (correctness first); a problem for Phase 2 (speed), where you'll keep BF16 or quantize.

## 6. The tools (for Exercise B)

Try each in a Python REPL before combining them.

**Tool 0: find the file, and close it automatically**

A relative path like `"learn_projects/..."` is resolved from the terminal's **current folder**, not from the script's folder, so it breaks when you run the script from somewhere else. Build the path from the script's own location instead:

```python
from pathlib import Path

HERE = Path(__file__).resolve().parent     # the folder this script is in
MODEL = HERE.parent / "models" / "qwen2.5-0.5b" / "model.safetensors"

with open(MODEL, "rb") as f:               # "rb" = read raw bytes
    first = f.read(8)
# the file is closed here, even if something crashed inside
```

**Tool 1: read raw bytes**
```python
f = open("models/qwen2.5-0.5b/model.safetensors", "rb")
first = f.read(8)      # bytes object
first.hex(" ")         # '18 7e 00 00 00 00 00 00'
f.tell()               # 8: the next read continues here
```

**Tool 2: bytes → numbers with `struct`**

| Char | Meaning |
| :--- | :--- |
| `<` / `>` | little-endian / big-endian |
| `H` / `I` / `Q` | unsigned 16 / 32 / 64-bit integer |
| `f` | 32-bit float |

```python
import struct
struct.unpack("<I", bytes([1, 2, 0, 0]))  # (513,)
struct.unpack(">I", bytes([1, 2, 0, 0]))  # (16908288,)
```

**Tool 3: JSON from bytes**
```python
import json
h = json.loads(b'{"a": {"dtype": "BF16", "shape": [2, 3]}}')
h["a"]["shape"]              # [2, 3]
```

**Tool 4: summarise**
```python
import math
math.prod([896, 151936])     # 136134656
{v["dtype"] for v in h.values()}   # unique dtypes
```

Still to teach before Exercise C: `np.frombuffer`, integer dtypes (`uint16`, `uint32`), shifting, and `.view` to reinterpret bits.

## 7. Exercises

**Exercise A ✅ done:** decode the first 8 bytes by hand (`xxd -l 64 models/qwen2.5-0.5b/model.safetensors`).

**Exercise B (next):** in a scratch file `explore_header.py`, step by step:
1. Read 8 bytes, unpack N. Expect **32280**.
2. `f.read(N)`, `json.loads`, print `len(header)`. (Watch out for `__metadata__`.)
3. Print dtype, shape and offsets of layer 0's `q_proj.weight` and `k_proj.weight`. Why is k smaller? (Answer: GQA, [Chapter 4](04_attention.md).)
4. Is `"lm_head.weight"` there? Check `tie_word_embeddings` in `config.json`.
5. Sum `end - begin` over all tensors; compare `8 + N + total` with `os.path.getsize(path)`.

**Exercise C:** implement `load()` in [`nanoinfer/safetensors.py`](../../learn_projects/inference_from_scratch/nanoinfer/safetensors.py) (BF16 → FP32) until this passes:
```bash
pytest tests/test_1_1_safetensors.py
```

---

## Check yourself

1. The first 8 bytes are `10 00 00 00 00 00 00 00`. How long is the JSON header?
2. In that file, a tensor has `data_offsets: [0, 8]`. At which byte of the file does its data start, and how many BF16 numbers does it hold?
3. BF16 `0x4000` is 2.0. What is it as an FP32, in hex?
4. Why do we convert to FP32 in Phase 1, and what does it cost in decode speed?

<details>
<summary>Answers</summary>

1. 0x10 = 16 bytes.
2. Byte 8 + 16 + 0 = 24. 8 bytes ÷ 2 bytes each = 4 numbers.
3. 0x40000000.
4. numpy has no bfloat16, and FP32 matches the reference tests. It doubles the bytes per word (~2 GB), halving the bandwidth ceiling (~46 words/sec instead of ~93).

</details>

---

> [← Chapter 8](08_bandwidth_and_speed.md) · [Contents](README.md) · [Self-test →](self_test.md)
