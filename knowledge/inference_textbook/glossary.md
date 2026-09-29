# Glossary

> [Contents](README.md) · [Self-test](self_test.md) · [Corrections](corrections.md)

Short definitions, with the classroom analogy where one exists. Numbers are for Qwen2.5-0.5B.

| Term | Meaning | Classroom | Chapter |
| :--- | :--- | :--- | :--- |
| **Activation** | The temporary numbers flowing through the model for your sentence (the rows). | Notebook page contents | [3](03_weights_and_matrix_multiply.md) |
| **Arithmetic intensity** | Operations done per byte fetched from memory. Decode ≈ 1; the M3 GPU can do ≈ 40. | | [8](08_bandwidth_and_speed.md) |
| **Attention** | The step where each row reads earlier rows and adds a blend of their handouts. The only place words exchange information. | Step A: look left and copy | [4](04_attention.md) |
| **Bandwidth** | How fast memory can feed the cores (bytes/sec). Measured: 92 GB/s CPU, 87 GB/s GPU. | The basement stairs | [8](08_bandwidth_and_speed.md) |
| **Base model** | A model trained only to continue internet text, not to act as an assistant (hence the "______" guesses). | | [2](02_tokens_and_embeddings.md) |
| **Batching** | Running many users' rows through one read of the weights. | Carrying one toolkit up for many classes | [8](08_bandwidth_and_speed.md) |
| **BF16** | 16-bit float: the top half of an FP32 (1 sign, 8 exponent, 7 fraction bits). | | [9](09_safetensors_file_format.md) |
| **Bias** | A learned number added after a multiplication. Qwen's q, k, v projections have biases. | | [4](04_attention.md) |
| **Causal mask** | The rule that a word sees only itself and earlier words. | Only look left | [4](04_attention.md) |
| **Compute-bound** | Speed limited by how fast the cores multiply (prefill). | | [8](08_bandwidth_and_speed.md) |
| **Decode** | Generating one new token per trip; bandwidth-bound. | The newcomer in seat 6 | [1](01_what_is_inference.md) |
| **Dot product** | Multiply matching positions, add them up. A similarity score. | Comparing a question card to a label | [4](04_attention.md) |
| **Embedding** | A token's 896-number meaning row, looked up from a 151,936 × 896 table. | Dictionary definition copied onto the page | [2](02_tokens_and_embeddings.md) |
| **FP32** | 32-bit float (4 bytes). | | [9](09_safetensors_file_format.md) |
| **Forward pass** | One full trip: embedding → 24 layers → output match. | One school day | [6](06_layers_and_the_forward_pass.md) |
| **GQA** | Grouped-Query Attention: 14 query heads share 2 key/value heads, making the KV cache 7× smaller. | 14 question cards, 2 label cards | [4](04_attention.md) |
| **Gradient descent** | Training: nudge every weight slightly in the direction that reduces the error. | | [3](03_weights_and_matrix_multiply.md) |
| **Head** | One complete attention search, 64 numbers wide. 14 per layer. | One question card and its search | [4](04_attention.md) |
| **Hidden size** | Row width: 896. | Boxes on a page | [2](02_tokens_and_embeddings.md) |
| **Inference** | Using a trained, frozen model to produce outputs. | | [1](01_what_is_inference.md) |
| **Key (k)** | A word's searchable label: row × Wk. Cached. | Label card | [4](04_attention.md) |
| **KV cache** | Saved k and v of every earlier token, per layer. 12 KB/token. | Filing cabinet, one drawer per period | [7](07_kv_cache.md) |
| **Layer** | One round of attention + MLP, each adding to the row. 24 of them, in order. | Period | [6](06_layers_and_the_forward_pass.md) |
| **Little-endian** | Least significant **byte** first in memory. | | [9](09_safetensors_file_format.md) |
| **Logit** | A raw score for one vocabulary entry. 151,936 per trip. | | [2](02_tokens_and_embeddings.md) |
| **Logit lens** | Running the output match on an intermediate layer's row, to see what the model "thinks" so far. | Peeking at a page mid-day | [6](06_layers_and_the_forward_pass.md) |
| **MLP** | Per-row detectors (gate, up, down) that add knowledge. 4,864 per layer, 64% of weights. | Step B: reference book | [5](05_mlp.md) |
| **Mix** | The softmax-weighted blend of earlier words' values. | Blended handouts | [4](04_attention.md) |
| **mmap** | Mapping a file into memory so it can be used in place, without copying. | | [9](09_safetensors_file_format.md) |
| **Output match (lm_head)** | Last row × embedding table → 151,936 logits. Tied to the embedding. | Checking the page against the dictionary | [2](02_tokens_and_embeddings.md) |
| **PagedAttention** | Managing the KV cache in fixed-size pages, like OS memory pages (vLLM). | | [7](07_kv_cache.md) |
| **Parameter** | One learned number. 494,032,768 in this model. | | [3](03_weights_and_matrix_multiply.md) |
| **Prefill** | Processing the whole prompt in parallel; compute-bound; produces word 1. | All 5 students at once | [1](01_what_is_inference.md) |
| **Quantization** | Storing weights in fewer bits (8, 4) to move fewer bytes. | | [8](08_bandwidth_and_speed.md) |
| **Query (q)** | What a word is searching for: row × Wq. Not cached. | Question card | [4](04_attention.md) |
| **Residual (add)** | Adding a step's output onto the row instead of replacing it. | Writing in the margin, never erasing | [4](04_attention.md) |
| **RMSNorm** | Rescales a row so numbers stay in range; 896 learned scales. Built in step 1.3. | | [6](06_layers_and_the_forward_pass.md) |
| **RoPE** | Rotary position encoding: stamps each word's position into q and k. Built in step 1.4. *Not covered yet.* | | [6](06_layers_and_the_forward_pass.md) |
| **safetensors** | Weight file format: 8-byte length + JSON header + raw data. | | [9](09_safetensors_file_format.md) |
| **Sampling** | Picking the next token from the logits (greedy, temperature, top-k, top-p). Step 1.7. *Not covered yet.* | | [2](02_tokens_and_embeddings.md) |
| **SiLU** | Smooth on/off switch: x × sigmoid(x). Used in the MLP's gate. | | [5](05_mlp.md) |
| **Softmax** | Turns scores into percentages: e^score ÷ sum. Exaggerates the winner. | | [4](04_attention.md) |
| **Speculative decoding** | A small model guesses several tokens; the big model verifies them in one pass. | | [8](08_bandwidth_and_speed.md) |
| **SwiGLU** | Qwen's MLP form: silu(gate) × up, then down. | | [5](05_mlp.md) |
| **Token** | A piece of text the tokenizer knows, with an ID. 151,936 in the vocabulary. | Student | [2](02_tokens_and_embeddings.md) |
| **Value (v)** | What a word hands over if picked: row × Wv. Cached. | Handout | [4](04_attention.md) |
| **Weight** | A learned, fixed number; a wire. | Teacher's toolkit | [3](03_weights_and_matrix_multiply.md) |
| **Weight tying** | Reusing the embedding table as the output matrix. | Same dictionary at both ends | [2](02_tokens_and_embeddings.md) |
| **Wo** | Output projection of attention: combines and translates the heads' mixes. | Translator | [4](04_attention.md) |
| **Wq, Wk, Wv** | Weights that make q, k, v from a row. One set per layer. | Question, label, handout forms | [4](04_attention.md) |

## Qwen2.5-0.5B at a glance

| Setting | Value |
| :--- | ---: |
| Parameters | 494,032,768 |
| File size (BF16) | 988 MB |
| Layers | 24 |
| Hidden size (row width) | 896 |
| Query heads / KV heads | 14 / 2 |
| Head dim | 64 |
| MLP detectors (intermediate size) | 4,864 |
| Vocabulary | 151,936 |
| Max context | 32,768 |
| Tied embeddings | yes |
| RoPE theta | 1,000,000 |
| RMSNorm epsilon | 1e-6 |
| KV cache per token | 12 KB |
