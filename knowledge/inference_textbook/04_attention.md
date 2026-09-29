# Chapter 4: Attention, From Zero

> [← Chapter 3](03_weights_and_matrix_multiply.md) · [Contents](README.md) · [Chapter 5 →](05_mlp.md)

**Reading time:** about 25 minutes. Read it with pen and paper. **Status:** ✅ covered.

## In one minute

- Attention is **the only place in the whole model where words exchange information**.
- Each word makes three cards from its row: a **question** (q = row × Wq), a **label** (k = row × Wk) and a **handout** (v = row × Wv).
- Each word scores its question against the labels of itself and every **earlier** word, turns the scores into percentages (**softmax**), and blends the handouts by those percentages. That blend is the **mix**.
- The mix goes through **Wo** and is **added** to the word's row.
- The real model runs **14 of these searches (heads)** in parallel, each 64 numbers wide, sharing **2** label/handout sets (**GQA**).

---

## Step 0: The problem attention solves

After the embedding lookup, the "is" row holds **only the dictionary meaning of "is"**. It has no idea France is in the sentence. But the "is" row is the one that must predict the next word, so it needs information from the other rows.

Two questions:
1. **Which** earlier rows should "is" take information from?
2. **What** information does it take?

Attention answers both. To stay small: **2 numbers per row** instead of 896, and 3 words.

```
capital row = [2,  0]
France  row = [0,  2]
is      row = [1, -1]
```

---

## Step 1: The only tool, a similarity score (dot product)

Multiply matching positions, then add:

```
[2, 0]  · [0, 2] = 2×0 + 0×2    =  0   unrelated
[0, 2]  · [0, 2] = 0×0 + 2×2    =  4   strongly similar
[1, -1] · [0, 2] = 1×0 + (-1)×2 = -2   opposite
```

**A big number means "these match".**

---

## Step 2: The naive attempt fails, which is why q and k exist

The obvious idea: compare the "is" row directly with each other row.

```
is · capital = 1×2 + (-1)×0 =  2
is · France  = 1×0 + (-1)×2 = -2   ← France scores LOWEST
```

That's a failure. The "is" row describes **what "is" is** (a verb). France's row describes **what France is** (a country). They don't look alike, so they score low. But "is" **needs** France.

What "is" should compare is **what it's looking for** against **what the others are**. Those are different things:
- **q (query)** = "what I'm looking for", made from my row.
- **k (key)** = "what I am", a label others can search, made from each row.

That's why Wq and Wk exist: they **rewrite a row into a search query, or into a searchable label**.

**Library analogy:** you type a **search query** (q), it's matched against the **title on each book's index card** (k), and you take home the **book's contents** (v). The names query, key and value come from database lookups.

---

## Step 3: Make the query, q = row × Wq

In the toy, Wq is 2 × 2:

```
         → q[0]  → q[1]
row[0] →   0       1
row[1] →   0      -1
```

```
q[0] = row[0]×0 + row[1]×0    = 1×0 + (-1)×0    = 0
q[1] = row[0]×1 + row[1]×(-1) = 1×1 + (-1)×(-1) = 2

q_is = [0, 2]
```

The same wires, applied to the "is" row, produced a **new vector that points toward country-like things**. Training found wires that do this.

---

## Step 4: Make the keys, k = row × Wk

Every word makes its own key with the **same** Wk. To keep the arithmetic short, in this toy Wk leaves rows unchanged (k = row). A real Wk rewrites the row, just as Wq did.

```
k_capital = [2,  0]
k_France  = [0,  2]
k_is      = [1, -1]   ("is" can also look at itself)
```

---

## Step 5: Score my query against every key

```
q_is · k_capital = 0×2 + 2×0    =  0
q_is · k_France  = 0×0 + 2×2    =  4   ← France now WINS
q_is · k_is      = 0×1 + 2×(-1) = -2
```

Compare with Step 2. The rows are identical; the only change is that "is" now asks **through Wq**. France went from the lowest score to the highest.

---

## Step 6: Scores become percentages (softmax)

Raise e (2.718) to each score, then divide by the total:

```
e^0 = 1.00    e^4 = 54.60    e^-2 = 0.14
total = 55.73

capital = 1.00  / 55.73 =  1.8%
France  = 54.60 / 55.73 = 98.0%
is      = 0.14  / 55.73 =  0.2%
```

Why e^?
- every value becomes positive;
- the percentages add up to 100%;
- it **exaggerates the winner**: a score gap of 4 becomes 98%.

**No human sets these percentages.** They come entirely from how well each label matches the question, recalculated for every sentence.

> **Real-model detail for step 1.5:** before softmax, the real model divides the scores by √64 = 8 (√head_dim) so they don't get too extreme. The toy skips it.

---

## Step 7: What to hand over, v = row × Wv

Scores decided **who** to read from. Now: **what** does each word hand over? That's the value, made with its own wires. Toy Wv = `[[0.5, 0], [0, 1.5]]`:

```
v_capital = [1,    0  ]
v_France  = [0,    3  ]
v_is      = [0.5, -1.5]
```

**Why not just hand over k?** A book's title (k) is good for **finding** it, but you want its **contents** (v). "The label you're searched by" and "the information you share" are different jobs, so they get different wires.

---

## Step 8: The mix, a blend of handouts

```
  0.018 × [1,    0  ] = [0.018,  0    ]
+ 0.980 × [0,    3  ] = [0,      2.939]
+ 0.002 × [0.5, -1.5] = [0.001, -0.004]
  ─────────────────────────────────────
  mix                ≈ [0.019,  2.935]
```

**The mix is the influence of the earlier words on the current word**, blended by relevance. Here it's almost pure "France information".

---

## Step 9: Wo, then ADD onto the row

The mix is written in "handout language" (2 numbers in the toy, 64 per head in the real model), not in row language (896). **Wo translates it into row format** so it can be added. In the toy, imagine Wo changes nothing:

```
is row (before) = [1,     -1   ]
+ mix × Wo      = [0.019,  2.935]
is row (after)  = [1.019,  1.935]
```

**Why ADD and not replace?** Replacing would make "is" forget it's the word "is". Adding keeps the old content and writes new notes on top, like writing in the margin without erasing. (This add is called the **residual connection**.)

**The "is" row now carries France's information. That's attention.**

---

## The whole thing on one screen

```
1. q = my row × Wq        "what am I looking for?"
2. k = each row × Wk      "what is each word, as a label?"
3. score = q · k          "how well does each match?"
4. softmax → percentages
5. v = each row × Wv      "what does each word hand over?"
6. mix = Σ percentage × v
7. my row = my row + mix × Wo
```

- **Wq, Wk, Wv, Wo** are fixed wires: in the 988 MB file, learned in training, one set **per layer**.
- **q, k, v** are temporary numbers, computed fresh for each word, in each layer, for this sentence.

> **Wk vs k:** Wk is the machine that makes label cards (a weight, in the file). k is one label card (an activation, computed, and later stored in the KV cache).

---

## Every word plays both roles

All three cards are computed **for every word, in every layer**:

```
each row ─× Wq─→ q   used by ME to search
         ─× Wk─→ k   used by LATER words to find me
         ─× Wv─→ v   handed to later words that pick me
```

Wk and Wv are not "just for reference". France's k and v are exactly what let "is" find it.

Two refinements to "the query finds the key, and voilà, the value":
- The query doesn't find **one** key. It scores **all** earlier keys and takes a **blend** (98% France, 2% capital).
- The value **is not the answer**. It's France's **notes**, added to the "is" row. " Paris" only appears after 22 layers of this plus the MLP.

### Is q only about its own word?

- **Layer 1:** q comes from the word's own row, which is still pure dictionary meaning, so it's **unaffected** by other words.
- **Layer 2 onwards:** the row already has other words' notes added into it, so q is built from a mixed row and is **indirectly influenced** by earlier words.

---

## The backward-only rule (causal mask)

A word may look at itself and **earlier** words, never later ones. When generating, the future doesn't exist yet.

```
            looks at →  capital  France  is
capital                   ✓        ✗     ✗
France                    ✓        ✓     ✗
is                        ✓        ✓     ✓
```

The whole score table is computed at once; the ✗ cells are **masked out** (set to −∞ before softmax, so they become 0%).

This rule is what makes prefill feel sequential even though it's parallel: row 3 can only see rows 1–3, so each row behaves as if words arrived one at a time. It's also exactly what makes the KV cache possible ([Chapter 7](07_kv_cache.md)).

---

## Heads: 14 searches at once

A **head** is one complete search: one question card, one set of scores, one set of percentages, one mix.

The real model runs **14 heads per word, per layer**, because a word usually needs several different things from the words before it:

```
"is" head 1: "which is the country?"      → 98% France  → mix 1
"is" head 2: "what is being asked about?" → 90% capital → mix 2
"is" head 3: "singular or plural?"        → ...         → mix 3
... 14 of these
```

(The job descriptions are illustrative; training decides what each head looks for.) With one search, "is" would have to pick one blend. With 14, it can look at France for one reason and at capital for another.

### Why 64 numbers per head?

896 ÷ 14 heads = **64**. The row's budget is split 14 ways.

- It's a designer's choice: long enough to describe a search, short enough to have many heads.
- It's GPU-friendly (most models use 64 or 128).
- Why not give every head 896? That would cost 14× more attention weights and a 14× bigger KV cache for little gain.

### Wo glues and translates

```
mix₁(64) | mix₂(64) | ... | mix₁₄(64)
   → glued side by side = 896 numbers
   → × Wo (896 × 896)   = 896 numbers in row format
   → ADD to the row
```

The glued 896 is **not** a finished result. It's 14 private languages sitting side by side. Head 1's 64 numbers dropped straight into boxes 0–63 would mean something else in the row. Wo **combines** the 14 findings and **translates** them into the row's language.

**Wo runs in every layer, for every word, every time.** It's internal, not for readability. The only step that exists purely to produce the output word is the **output match** at the very end ([Chapter 2](02_tokens_and_embeddings.md)).

---

## GQA: 14 questions, only 2 label/handout sets

Real shapes from your file (every layer):

```
q_proj.weight  [896, 896]  → q: 14 heads × 64 = 896
k_proj.weight  [128, 896]  → k:  2 heads × 64 = 128
v_proj.weight  [128, 896]  → v:  2 heads × 64 = 128
o_proj.weight  [896, 896]
(q, k and v have biases; o does not)
```

Each word writes **14 question cards** but only **2 label cards and 2 handouts**. The questions share them:

```
query heads 0–6   → search label/handout set A
query heads 7–13  → search label/handout set B
(query head i uses set i // 7)
```

Like 14 people searching a library with 14 different questions, but only 2 catalogs.

**Why:** k and v are what gets saved in the **KV cache** for every token, and read back through the memory pipe on every decode step. Two sets instead of 14 makes the cache **7× smaller** and 7× cheaper to read, with a small quality loss. This is **Grouped-Query Attention (GQA)**: a model design decision made for bandwidth reasons ([Chapter 7](07_kv_cache.md)).

---

## Proof: your real model does this

"When the 'is' row reads the sentence, where does it look?" The head that looks at France most, in several layers:

```
L0  h6: The  0% capital 11% of 1% France 88% is 0%
L5  h5: The 13% capital 23% of 6% France 54% is 3%
L15 h2: The 21% capital  0% of 3% France 68% is 8%
L21 h6: The 13% capital  0% of 5% France 76% is 6%
```

Layer 0, head 6 puts **88% on France**, almost exactly the toy's 98%. (Honest note: this picks the France-focused head in each layer; the other 13 heads look for other things.)

---

## Check yourself

1. **Wq does nothing** (q = row, so q_is = `[1, -1]`). Compute the three scores against k_capital `[2,0]`, k_France `[0,2]`, k_is `[1,-1]`. Who does "is" read from? What does that tell you Wq is for?
2. The "France" row has query `[2, 1]`. It can only see capital `[2,0]` and itself `[0,2]`. What are its scores, and who does it attend to more?
3. What is the difference between Wk and k? Which is in the file, and which is in the KV cache?
4. Why does Wo exist? Does it run only for the final word?
5. Why does Wq output 896 numbers but Wk only 128?
6. In "I sat on the river bank", which word would "bank"'s question card probably match, and why does that change bank's row?

<details>
<summary>Answers</summary>

1. Scores: capital 2, France −2, is 2. Percentages ≈ 49.5% capital, 0.9% France, 49.5% itself. "is" ignores France. Wq exists to turn "what I am" into "what I'm looking for"; without it, words only find words that look like themselves.
2. capital: 2×2 + 1×0 = 4. France: 2×0 + 1×2 = 2. Softmax gives capital ≈ 88%, France ≈ 12%. It attends to capital more.
3. Wk is the weight (the machine that makes label cards), stored in the 988 MB file. k = row × Wk is one word's label card, computed fresh and stored in the KV cache.
4. The 14 heads' mixes are each in their own 64-number language. Wo combines them and translates them into the 896-number row language so they can be added. It runs in every layer, for every word; it's not about readability.
5. 14 query heads × 64 = 896, but only 2 key/value heads × 64 = 128 (GQA), to make the KV cache 7× smaller.
6. "river". Its handout gets added to the bank row, so the row now means "riverbank", not "financial bank".

</details>

---

> [← Chapter 3](03_weights_and_matrix_multiply.md) · [Contents](README.md) · [Chapter 5 →](05_mlp.md)
