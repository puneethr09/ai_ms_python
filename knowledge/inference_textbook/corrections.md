# Corrections Log: What I Believed vs What's True

> [Contents](README.md) · [Self-test](self_test.md) · [Glossary](glossary.md)

Every misconception that came up while learning, with the fix. **Read this before interviews.** These are the exact places intuition goes wrong, and most people get them wrong too.

The pattern to watch: most of these are **a correct idea stretched too far**.

---

## Model structure

| # | I believed | What's true | Chapter |
| :--- | :--- | :--- | :--- |
| 1 | The prompt's words are fed into the model one by one. | In prefill, all prompt rows go in **together** as one grid and are processed in parallel. It only *behaves* sequentially because of the backward-only rule. | [1](01_what_is_inference.md), [4](04_attention.md) |
| 2 | Embedding means multiplying the ID by 896 numbers. | It's a **lookup**: fetch row #ID from a 151,936 × 896 table. | [2](02_tokens_and_embeddings.md) |
| 3 | The five word embeddings are combined (e.g. averaged), and the answer is the closest word. | An average would land somewhere bland. The **24 layers transform** the rows first; only then is the last row matched. | [2](02_tokens_and_embeddings.md) |
| 4 | Wq, Wk, Wv are "just for reference". | They're **real tensors in the file** (`self_attn.q_proj.weight`, etc.), one set per layer, used on every word. | [4](04_attention.md) |
| 5 | Wq lives in some node of a layer. | **Weights are wires, not nodes.** Wq is the full set of 802,816 wires between the row and the q card. Nodes are the temporary row numbers. | [3](03_weights_and_matrix_multiply.md) |
| 6 | Wires are assigned to particular words in parallel. | **Every word uses all the same wires** (the cookie cutter). Parallelism comes from rows being independent in the multiply. | [3](03_weights_and_matrix_multiply.md) |
| 7 | Someone designed Wq/Wk/Wv, and the "knowledge weights" were made differently. | **Humans choose only shapes.** All 494M numbers start random and are set by the **same** training. Their roles come from their position in the circuit. | [3](03_weights_and_matrix_multiply.md) |
| 8 | MLP patterns are predefined rules. | They're **learned weights**: rows of gate/up are patterns, columns of down are notes. | [5](05_mlp.md) |
| 9 | Facts are universal, so one shared fact book (or one only at the last layer) should be enough. | The book is organized as "if the row looks like X, add Y", and the row looks different at every stage. Multi-step facts need several lookups in sequence. | [5](05_mlp.md) |
| 10 | Wqkv is about 30% of the model / should be 30 MB. | 30 MB is **one whole layer**. Attention is only **9%** (88 MB); the MLP is 64%; the dictionary 28%. | [3](03_weights_and_matrix_multiply.md), [5](05_mlp.md) |
| 11 | The model needs 22 layers. | It always runs **all 24**. 22 was just where " Paris" first became readable for that sentence. | [6](06_layers_and_the_forward_pass.md) |
| 12 | Layers form a cone: the network narrows toward the answer. | A transformer is a **tube**: 896 wide throughout, equal work per layer, a bulge in each MLP, and it gets *wider* at the output (151,936). Only the content converges. | [6](06_layers_and_the_forward_pass.md) |

## Attention

| # | I believed | What's true | Chapter |
| :--- | :--- | :--- | :--- |
| 13 | The query finds the right key, and the value is the answer. | The query scores **all** earlier keys and takes a **blend**. The value is that word's **notes**, added to the row, not the answer. | [4](04_attention.md) |
| 14 | The query is the main thing; k and v are secondary. | All three are computed for every word, in every layer. **Every word plays both roles**: searcher (q) and searched (k, v). | [4](04_attention.md) |
| 15 | A word's query is only about itself. | True in layer 1. From layer 2 on, the row already contains other words' notes, so q is **indirectly influenced** by earlier words. | [4](04_attention.md) |
| 16 | Wo is for readability: it only matters for the final output word. | Wo runs in **every layer, for every word**. It translates the 14 heads' mixes into row language so they can be added. The step that exists only for output is the **output match**. | [4](04_attention.md) |
| 17 | The glued 896 of the heads is the layer's output, which goes on to the next layer. | The glued 896 is 14 private languages side by side. Wo translates it, it's **added** to the row, and then the **MLP** still runs. | [4](04_attention.md), [6](06_layers_and_the_forward_pass.md) |

## Speed and hardware

| # | I believed | What's true | Chapter |
| :--- | :--- | :--- | :--- |
| 18 | I'm limited only by the GPU's compute units. | Two limits: compute and **memory bandwidth**. Decode is bandwidth-bound; prefill is compute-bound. | [8](08_bandwidth_and_speed.md) |
| 19 | Once the model is loaded in the GPU, there's no bandwidth issue. | Weights sit in memory chips (the warehouse). The cores (the desk) hold a few MB, so all 988 MB stream through the pipe **for every word**. | [8](08_bandwidth_and_speed.md) |
| 20 | Dictionary matching happens on the CPU. | It happens wherever the model runs (the GPU): it's just another matrix multiply. | [8](08_bandwidth_and_speed.md) |
| 21 | Dictionary matching only happens in decode. | It happens **once per generated word**, including at the end of prefill (which produces word 1). | [1](01_what_is_inference.md) |
| 22 | In decode, the row travels through the pipe along with the weights. | The row (~2–4 KB) stays on the chip. The **weights and the KV cache** travel. | [7](07_kv_cache.md), [8](08_bandwidth_and_speed.md) |
| 23 | A 10-word answer = 11 passes (prefill + 10 decode). | **10 passes**: prefill's pass produces word 1. | [8](08_bandwidth_and_speed.md) |
| 24 | The KV cache can never exceed the model size. | It often does: 12 KB/token, ~4 GB for 10 users at full context (≈ 4× this model). | [7](07_kv_cache.md) |

## Files and bytes

| # | I believed | What's true | Chapter |
| :--- | :--- | :--- | :--- |
| 25 | Little-endian: reverse the hex digits (`187e` → `e781`). | Reverse the **bytes** (`18 7e` → `7e 18` = 32,280). A byte's two hex digits stay together. | [9](09_safetensors_file_format.md) |
| 26 | `config.json` is where the model lives (it's the first file listed). | `config.json` is only the **shape** (896, 24 layers, …). The weights are in `model.safetensors`, the 988 MB file. Reading 8 bytes of the wrong file silently gives a nonsense number. | [9](09_safetensors_file_format.md) |

---

## Precision habit

Two answers were right in my head but wrong on paper: "v gets stored" (meant k) and "Wo produces the final output of the layer" (meant: added to the row, then the MLP). In code, writing v where you meant k doesn't crash; it gives a silently wrong answer. **Say exactly what you mean: which tensor, which axis, which count.**
