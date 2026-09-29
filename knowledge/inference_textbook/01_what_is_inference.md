# Chapter 1: What Is Inference?

> [Contents](README.md) · Next: [Chapter 2, Tokens and Embeddings →](02_tokens_and_embeddings.md)

**Reading time:** about 10 minutes. **Status:** ✅ covered.

## In one minute

- A trained language model is a **frozen function**: text goes in, and out comes **a score for every possible next word**.
- Generating text is a **loop**: predict one word, append it, predict again.
- **Inference** means running that loop. **Inference engineering** means running it **fast, cheaply and for many users at once**.
- Every word generated requires reading **every weight** in the model once. So memory speed, not compute, is the wall.

---

## 1. Training vs inference

A language model has two phases in its life.

| | Training | Inference |
| :--- | :--- | :--- |
| What happens | The model learns from trillions of words | The model is *used*: you give it text and it answers |
| Who does it | Big labs, once, for months, on thousands of GPUs | Everyone, billions of times a day |
| The numbers inside | **Change** (they are being learned) | **Frozen**, only read |

After training, the model is a frozen function:

```
text in  ──►  [ model ]  ──►  one score for each of
                              151,936 possible next tokens
```

That's all a chatbot is at its core. It predicts one next word, adds it to the text, and predicts again.

Nobody gets paid in inference engineering to make the model **smarter**. You get paid to make it **faster per dollar**: more words per second, more users per GPU, lower latency.

---

## 2. One full trip through the machine

Here is what happens inside **Qwen2.5-0.5B** (the model we build by hand), with its real numbers. The IDs are real output from its tokenizer.

```
"The capital of France is"
        │
 ① TOKENIZER: split text into pieces → ID numbers
        │
 [785, 6722, 315, 9625, 374]
  The  capital of France  is        ← 5 tokens
        │
 ② EMBEDDING: look up each ID in a table
        │
 5 rows × 896 numbers               ← one "meaning row" per word
        │
 ③ 24 LAYERS, each one:
     • ATTENTION: rows read earlier rows
     • MLP: each row looks up knowledge
        │
 ④ OUTPUT MATCH: last row vs every
    dictionary entry
        │
 151,936 scores (logits)
        │
 ⑤ SAMPLING: pick one → " Paris"
        │
 append " Paris" and go back to ③
```

Every chatbot runs this same loop: ChatGPT, Claude, Llama, Qwen. Models differ in size and in details, not in shape.

Each box has its own chapter:

| Box | Chapter |
| :--- | :--- |
| ① Tokenizer, ② Embedding, ④ Output match | [Chapter 2](02_tokens_and_embeddings.md) |
| The weights all of this uses | [Chapter 3](03_weights_and_matrix_multiply.md) |
| ③ Attention | [Chapter 4](04_attention.md) |
| ③ MLP | [Chapter 5](05_mlp.md) |
| ③ How the 24 layers fit together | [Chapter 6](06_layers_and_the_forward_pass.md) |
| Not redoing old work | [Chapter 7](07_kv_cache.md) |
| Why it's slow | [Chapter 8](08_bandwidth_and_speed.md) |
| Where the weights come from on disk | [Chapter 9](09_safetensors_file_format.md) |

---

## 3. The loop: one word per trip

```
Trip 1:  [The capital of France is]           → 24 layers → " Paris"
Trip 2:  [The capital of France is Paris]     → 24 layers → "."
Trip 3:  [The capital of France is Paris .]   → 24 layers → next
...until an end token or a length limit
```

- **One new word = one complete trip through all 24 layers.**
- The previous words must be included, because attention is how "is" finds out about "France". Feed the model only "is" and it has no idea what you're asking.
- The trip always runs **all 24 layers**, even when the answer is obvious early on. The model can't know in advance when it's done.
- It stops when it produces a special **end token**. For this base model that's `<|endoftext|>` (ID 151643); chat models use `<|im_end|>`.

### The two modes: prefill and decode

| | Prefill | Decode |
| :--- | :--- | :--- |
| What | Process the whole prompt | Generate one new word |
| Rows per trip | All prompt words **at once** | **1** new row |
| Speed feel | Fast: reads your question almost instantly | Slow: types the answer word by word |
| Limited by | Compute | Memory bandwidth |
| Produces | The **first** new word | Each following word |

The **first** answer word comes from the output match at the end of prefill. After that, each word is one decode trip. So a 10-word answer takes **10 trips**: 1 prefill + 9 decode.

Why prefill is fast and decode is slow is the central question of the whole field. [Chapter 8](08_bandwidth_and_speed.md) answers it fully.

---

## 4. The most important fact in inference engineering

Qwen2.5-0.5B has **494,032,768** learned numbers (weights). Each is stored in 2 bytes (the BF16 format), so the file is **988 MB**.

> To generate **one** word, the machine must read **every weight once**.
> One word = 988 MB read from memory.

We measured this Mac's memory speed in Phase 0: **92 GB/s**.

```
92 GB/s ÷ 0.988 GB per word  ≈  93 words/sec  (the ceiling)
```

No code, however clever, can beat that for one user on this machine. Every optimization in this course fights that wall.

---

## 5. Why build it yourself?

llama.cpp, vLLM and TensorRT-LLM are all **the loop in section 2, heavily optimized**. Each optimization attacks one box. You can't understand an optimization if you've never seen the box it optimizes.

| Optimization | What it attacks | Roadmap phase |
| :--- | :--- | :--- |
| Writing the loop at all | Everything | **1 (now)** |
| KV cache: don't recompute old words | ③ attention | 2 |
| C with threads and SIMD | ③ the matrix math | 2 |
| Quantization: 4 bits instead of 16 | The 988 MB read | 3 |
| GPU kernels (Metal, CUDA) | ③ on faster hardware | 4, 5 |
| Batching: many users per weight read | The loop at scale | 6 |
| Multiple GPUs | Models too big for one machine | 7 |

In Phase 1, you build every box in plain numpy. It will be **slow** (about 1 word/sec) and that's fine: its job is to be correct and to teach you what each box does. At the end, this prints " Paris" using only code you wrote:

```bash
python -m nanoinfer.generate "The capital of France is"
```

---

## Check yourself

1. In your own words, what is the difference between training and inference?
2. A model produces a 10-word answer. How many full trips through the 24 layers happen, and which one is prefill?
3. Why does the machine need to read all 988 MB for every generated word?

<details>
<summary>Answers</summary>

1. Training changes the weights, over months, to make predictions better. Inference uses the frozen weights to predict, billions of times.
2. 10 trips. The first is prefill (the whole prompt at once, which produces word 1); the other 9 are decode trips.
3. The new word's row has to go through every layer's multiplications, and each multiplication needs that layer's weights. Every word passes through every weight. (Chapters 3 and 8 explain this.)

</details>

---

> [Contents](README.md) · Next: [Chapter 2, Tokens and Embeddings →](02_tokens_and_embeddings.md)
