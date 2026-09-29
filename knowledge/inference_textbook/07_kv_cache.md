# Chapter 7: The KV Cache (the Filing Cabinet)

> [← Chapter 6](06_layers_and_the_forward_pass.md) · [Contents](README.md) · [Chapter 8 →](08_bandwidth_and_speed.md)

**Reading time:** about 12 minutes. **Status:** ✅ concept covered (you build it in Phase 2).

## In one minute

- Earlier words' **k and v never change** once computed, because words only look backward. So we **save** them instead of recomputing them. That saved store is the **KV cache**.
- It is **per layer**: each layer has its own k and v for every word (one drawer per period).
- Qwen2.5-0.5B: **12 KB per token**. That's about 400 MB for one user at the full 32K context, and it can exceed the size of the model itself.
- In decode, the cache is **read through the memory pipe on every step**, just like the weights. That's why GQA shrinks it and why PagedAttention manages it.

---

## 1. Why caching is allowed

Without a cache, every trip redoes the whole prompt:

```
Trip 1: [The capital of France is]        → all 5 rows
Trip 2: [The capital of France is Paris]  → all 6 rows
                                             ↑ 5 already done!
```

When " Paris" arrives, do "capital", "France" or "is" change? **No.** They only look backward, so a new word *after* them can never affect them. Their k and v in every layer are exactly the same as on the previous trip.

> **Save every old word's k and v, in every layer.** A new word computes only its own q, k, v and reads the saved ones.

The backward-only rule (causal mask) is exactly what makes this possible. Your Phase 1 code has no cache and redoes everything each trip. That's one reason it runs at about 1 word/sec.

## 2. What goes in the cabinet, and what doesn't

| Item | Where it comes from | In the KV cache? |
| :--- | :--- | :--- |
| Wq, Wk, Wv, Wo, MLP | The 988 MB file (the basement) | ❌ They're **weights** |
| Earlier words' k (labels) | Computed on earlier trips | ✅ |
| Earlier words' v (handouts) | Computed on earlier trips | ✅ |
| New word's q | Computed fresh now | ❌ Only used once, by itself |
| New word's k, v | Computed fresh now | ✅ Added after use, for later words |
| Rows (activations) | Recomputed | ❌ |

Classroom version: Paris, in period 7, doing Step A.

| Item | Answer |
| :--- | :--- |
| Paris's own question cards | **Fresh** (made with period 7's Wq) |
| France's label cards | **Filing cabinet**, drawer 7 |
| Paris's own handouts | **Fresh**, then filed in drawer 7 |
| The translator (Wo) for period 7 | **Neither**: it's a weight from the basement, carried upstairs through the pipe |

**Is the cache layer-specific?** Yes. France's label at layer 7 = France's row **at layer 7** × **layer 7's** Wk. The row and the Wk both differ per layer, so each layer needs its own drawer. That's why "24 layers ×" appears in the size formula.

## 3. A decode step with the cache

When " Paris" goes through one layer:

1. Make its own q, k, v (needs this layer's Wq, Wk, Wv, which is why **decode still needs all the weights**).
2. Compare its q against **all cached k's** (The, capital, of, France, is) plus its own k.
3. Blend **all cached v's** plus its own v by those percentages.
4. × Wo, add to its row, then MLP, add.
5. **Append its k and v to this layer's drawer** for the next word.

The work per step no longer grows with a full recompute of the prompt, but the cache that must be **read** does grow by one entry per word.

## 4. How big is it?

For Qwen2.5-0.5B, per token:

```
24 layers × 2 (k and v) × 2 KV heads × 64 numbers × 2 bytes
= 12,288 bytes ≈ 12 KB per token
```

| Situation | KV cache |
| :--- | ---: |
| 1 user, 4,000 tokens | ~49 MB |
| 1 user, full 32,768 context | ~403 MB (41% of the model) |
| 10 users, full context | ~4.0 GB (≈ 4× the model!) |

**Can it exceed the model size?** Yes, and on real servers it usually does. On busy servers the KV cache is typically the biggest thing in memory, bigger than the weights. Managing that memory without waste is what **PagedAttention** (vLLM) solves, by borrowing OS memory **pages** (chapter 02 of the systems book). That's Phase 6.

## 5. Why GQA exists: the cache is read every step

In decode, the cache for **every earlier token** travels through the memory pipe **on every step**, like the weights. So its size directly costs speed:

```
                     per token   at 4,000 tokens
14 KV heads (full)    84 KB      ~344 MB read per word
 2 KV heads (GQA)     12 KB       ~49 MB read per word
```

Without GQA, at 4,000 tokens, each word would read a third of the model's size again just for the cache. Two KV heads shared by 14 query heads make it **7× smaller**. **A model design decision made for inference bandwidth.**

---

## Check yourself

1. Why can't a new word change the k and v of earlier words?
2. When " Paris" goes through layer 1 during decode, list what it computes fresh and what it takes from the cache.
3. What's in the KV cache: Wk or k?
4. Your model at 8,000 tokens of chat: roughly how big is the KV cache for one user?
5. In decode, which of these travel through the memory pipe on every step: (a) the new row, (b) the weights, (c) the KV cache?

<details>
<summary>Answers</summary>

1. Earlier words only look backward (causal mask), so a word appearing after them isn't part of their computation.
2. Fresh: its q, k, v, the scores, the mix, Wo's output, the MLP. From the cache: every earlier word's k and v for layer 1. (Wq, Wk, Wv, Wo and the MLP weights come from memory, not the cache.)
3. k (the computed label cards). Wk is a weight in the file.
4. 8,000 × 12,288 bytes ≈ 98 MB.
5. (b) and (c). The row is only 2–4 KB and stays on the chip.

</details>

---

> [← Chapter 6](06_layers_and_the_forward_pass.md) · [Contents](README.md) · [Chapter 8 →](08_bandwidth_and_speed.md)
