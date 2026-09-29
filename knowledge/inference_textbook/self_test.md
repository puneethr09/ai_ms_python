# Self-Test: Flashcards for Travel

> [Contents](README.md) · [Corrections](corrections.md) · [Glossary](glossary.md)

Answer each one out loud or on paper, **then** tap to reveal. If you get one wrong, follow the chapter link. Aim to answer every card in one sentence.

---

## Round 1: The big picture ([Ch 1](01_what_is_inference.md), [Ch 2](02_tokens_and_embeddings.md))

**1.** What does a trained model compute, in one line?
<details><summary>Answer</summary>Text in → one score (logit) for every token in the vocabulary, for the next position.</details>

**2.** What's the difference between prefill and decode?
<details><summary>Answer</summary>Prefill processes the whole prompt in parallel and produces word 1. Decode processes one new row per trip and produces each later word.</details>

**3.** What is an embedding, and is it a multiplication?
<details><summary>Answer</summary>A row of 896 learned numbers describing a token's meaning. It's a lookup (fetch row #ID), not a multiplication.</details>

**4.** How many logits come out per trip for Qwen2.5-0.5B, and why that number?
<details><summary>Answer</summary>151,936: one per vocabulary entry. They're scores; softmax turns them into probabilities.</details>

**5.** Why is there no `lm_head.weight` in the file?
<details><summary>Answer</summary>Weight tying: the output match reuses the embedding table.</details>

**6.** Why do we read the prediction from the last row?
<details><summary>Answer</summary>Rows only look backward, so only the last row has seen the whole sentence, and each row predicts the word after its own position.</details>

## Round 2: Weights ([Ch 3](03_weights_and_matrix_multiply.md))

**7.** Wires or nodes: which are the weights, and which change per sentence?
<details><summary>Answer</summary>Wires are the weights (fixed). Nodes are the row's numbers (activations), which change for every sentence and every layer.</details>

**8.** Compute `[1, 0, 2] × W` where W's columns are (1,0,2), (0,3,1), (2,1,0).
<details><summary>Answer</summary>[5, 2, 2].</details>

**9.** Who chose Wq's values, and who chose its shape?
<details><summary>Answer</summary>Training chose the values (random start, nudged by gradient descent). Humans chose the shape.</details>

**10.** What does "0.5B parameters" mean here, exactly?
<details><summary>Answer</summary>494,032,768 learned numbers; at 2 bytes each (BF16) that's 988 MB.</details>

**11.** How big is one layer, and what share of the model is attention, MLP and embedding?
<details><summary>Answer</summary>~14.9M numbers ≈ 30 MB per layer. Attention 9% (88 MB), MLP 64% (628 MB), embedding 28% (272 MB).</details>

## Round 3: Attention ([Ch 4](04_attention.md))

**12.** Name the three cards and what each one means.
<details><summary>Answer</summary>q = what I'm looking for; k = my label, for others to find me; v = my handout, what I give if picked.</details>

**13.** Why can't "is" just compare its own row with France's row?
<details><summary>Answer</summary>Its row says what it *is* (a verb), not what it's *looking for*. Wq rewrites the row into a search query.</details>

**14.** Softmax of scores [0, 4, −2]?
<details><summary>Answer</summary>≈ 1.8%, 98.0%, 0.2%.</details>

**15.** If Wq did nothing (q_is = [1, −1]), what would "is" score against capital [2,0], France [0,2], itself [1,−1]?
<details><summary>Answer</summary>2, −2, 2 → it reads capital and itself (≈49.5% each), France ≈ 0.9%. Wq is what points the search at the right thing.</details>

**16.** What is the mix?
<details><summary>Answer</summary>The blend of earlier words' handouts (v), weighted by the softmax percentages: the earlier words' influence on this word.</details>

**17.** What is a head, and why 64 numbers?
<details><summary>Answer</summary>One complete search (q, scores, %, mix). 14 heads × 64 = 896: the row's budget split 14 ways.</details>

**18.** What does Wo do, and when does it run?
<details><summary>Answer</summary>Combines the 14 heads' mixes and translates them into row language, so they can be added. It runs in every layer, for every word.</details>

**19.** Why 14 query heads but only 2 KV heads?
<details><summary>Answer</summary>GQA: k and v are cached and read every decode step, so sharing 2 sets among 14 queries makes the cache 7× smaller. Query head i uses KV head i // 7.</details>

**20.** Wk vs k?
<details><summary>Answer</summary>Wk is a weight in the file (the label-card machine). k = row × Wk is one word's label card, stored in the KV cache.</details>

## Round 4: MLP ([Ch 5](05_mlp.md))

**21.** Why does the MLP become the fact store, not attention?
<details><summary>Answer</summary>Attention can only move information already in the sentence. The MLP is the only place outside information (its weights) enters a row.</details>

**22.** What are gate, up and down, in one line each?
<details><summary>Answer</summary>Gate: does each detector fire (on/off switch)? Up: how strongly? Down: what does each detector write back (896 numbers)?</details>

**23.** Why is 4,864 there, if every read/write is 896 long?
<details><summary>Answer</summary>4,864 is the number of detectors. The row widens to 4,864 strengths inside the MLP, then shrinks back to 896.</details>

**24.** Why does each layer have its own MLP?
<details><summary>Answer</summary>Rules fire on what the row looks like, and the row looks different at each stage (spelling → meaning → answer). Chains of facts also need several lookups in sequence.</details>

**25.** ⏸ *(Pending, before step 1.3.)* Toy MLP (gate D1 [1,1,0], D2 [0,0,1]; up D1 [0,1,0], D2 [1,0,0]; notes D1 [0,0,0.5], D2 [1,0,0]), row [0,0,5]. What gets added?
<details><summary>Answer</summary>D1 gate 0 → off. D2 gate 5 → on, but up = 0, so strength 0. Nothing is added. Gate and up must both agree.</details>

## Round 5: Layers ([Ch 6](06_layers_and_the_forward_pass.md))

**26.** Write the two ADD lines of a layer.
<details><summary>Answer</summary>row = row + mix × Wo (other words enter); row = row + MLP(row) (knowledge enters).</details>

**27.** Where's the only place information moves between words?
<details><summary>Answer</summary>The attention ADD: row = row + mix × Wo.</details>

**28.** Does layer 20 cost less than layer 2?
<details><summary>Answer</summary>No. Same shapes, same compute, same ~30 MB of weights. A tube, not a cone.</details>

**29.** Why not 2 layers? Why not 100?
<details><summary>Answer</summary>2: multi-step answers need sequential rounds. 100: ~3.25 GB per word, ~28 words/sec, and harder to train. The shape is an educated trial-and-error choice.</details>

**30.** At which layer did " Paris" become the top guess, and what was the second guess there?
<details><summary>Answer</summary>Layer 22; second was 巴黎, "Paris" in Chinese: the concept before the spelling.</details>

## Round 6: KV cache ([Ch 7](07_kv_cache.md))

**31.** Why can we cache k and v?
<details><summary>Answer</summary>Words only look backward, so a later word can't change earlier words' k and v.</details>

**32.** Is the KV cache per layer?
<details><summary>Answer</summary>Yes. A word's k at layer 7 = its row at layer 7 × layer 7's Wk. One drawer per layer.</details>

**33.** KV cache size per token for Qwen2.5-0.5B?
<details><summary>Answer</summary>24 × 2 × 2 × 64 × 2 B = 12 KB (84 KB without GQA).</details>

**34.** Paris in period 7: fresh or cabinet? Its question cards, France's labels, its handouts, Wo.
<details><summary>Answer</summary>Fresh; cabinet; fresh (then filed); neither: Wo is a weight from memory.</details>

## Round 7: Speed ([Ch 8](08_bandwidth_and_speed.md))

**35.** The ceiling formula, and its value for this model on this Mac?
<details><summary>Answer</summary>words/sec ≤ bandwidth ÷ bytes per word = 92 GB/s ÷ 0.988 GB ≈ 93.</details>

**36.** In decode, what travels through the pipe each step?
<details><summary>Answer</summary>All the weights and the KV cache. Not the row.</details>

**37.** 10-word answer: passes, GB, time?
<details><summary>Answer</summary>10 passes, ~9.9 GB, ~107 ms.</details>

**38.** Why is decode bandwidth-bound?
<details><summary>Answer</summary>Each weight fetched is used for ~1 operation (one row); the cores could do ~40 per byte fetched, so they wait on memory.</details>

**39.** Three ways to use decode's idle compute?
<details><summary>Answer</summary>Batching (one weight read, N users), quantization (fewer bytes), speculative decoding (verify several guessed words per weight read).</details>

## Round 8: The file ([Ch 9](09_safetensors_file_format.md))

**40.** `18 7e 00 00 00 00 00 00` as little-endian u64?
<details><summary>Answer</summary>0x7e18 = 32,280.</details>

**41.** Where does the tensor data start in your file?
<details><summary>Answer</summary>Byte 8 + 32,280 = 32,288. Offsets in the header count from there.</details>

**42.** How do you turn BF16 into FP32?
<details><summary>Answer</summary>Put the 16 bits in the top half of a 32-bit value (shift left 16; the bottom 16 bits are zero).</details>

**43.** Why safetensors instead of pickle?
<details><summary>Answer</summary>Loading a pickle can run arbitrary code; safetensors is pure data, and its raw layout allows zero-copy mmap.</details>
