# Chapter 2: Tokens, Embeddings and the Output Match

> [← Chapter 1](01_what_is_inference.md) · [Contents](README.md) · [Chapter 3 →](03_weights_and_matrix_multiply.md)

**Reading time:** about 15 minutes. **Status:** ✅ covered (the tokenizer algorithm itself comes in step 1.2).

## In one minute

- The **tokenizer** turns text into ID numbers: `"The capital of France is"` → `[785, 6722, 315, 9625, 374]`.
- An ID is just a name tag. The **embedding table** gives each ID a row of **896 numbers** that describes its meaning. It's a **lookup**, not a multiplication.
- At the end, the last row is compared against **every row of that same table**, which gives **151,936 scores (logits)**, one per token. The highest score is the next word.
- Using the same table at both ends is called **weight tying**, and it's why the file has no `lm_head.weight`.

---

## 1. Tokens: text becomes numbers

Computers do math on numbers, not letters. The tokenizer splits text into pieces it knows and replaces each piece with its ID:

```
"The capital of France is"
   ↓
'The'  ' capital'  ' of'  ' France'  ' is'
  785     6722      315     9625      374
```

Notes:
- Pieces usually **include the leading space**: `' Paris'` (ID 12095) is a different token from `'Paris'`.
- The model knows **151,936** pieces (its **vocabulary**): whole words, word parts like `"ing"`, punctuation, Chinese characters, code symbols.
- Qwen uses **byte-level BPE**. You'll build it in step 1.2.

---

## 2. Embeddings: an ID becomes a meaning row

### The problem

ID 12095 is only a name tag, like a roll number in class. Roll number 12095 is not "close" to 12096 in any meaningful way, and you can't do math on name tags.

### The fix: coordinates of meaning

Think of a map. Every city has 2 coordinates (latitude, longitude), and cities that are close on the map have close numbers. An embedding is the same idea with **896 coordinates instead of 2**. Every token gets a position in an 896-dimensional "meaning space", where similar meanings sit near each other.

Real numbers from your model (cosine similarity: 1.0 means pointing the same way, 0 means unrelated):

```
' Paris'  vs ' London'   0.39   ← both capital cities
' Paris'  vs ' Tokyo'    0.35
' banana' vs ' apple'    0.34   ← both fruits
' Paris'  vs ' banana'  -0.03   ← unrelated
```

Nobody typed these numbers in. **Training discovered them.**

### It's a lookup, not a multiplication

The embedding is a table:

```
         896 columns (the coordinates) →
row 0    [ ... ]
row 1    [ ... ]
...
row 12095 [0.024, -0.0036, -0.002, ...]  ← ' Paris'
...
row 151935
```

"Embedding a token" means: take its ID and fetch that row. That's all.

- **Size:** 151,936 × 896 = **136,134,656 numbers**, 272 MB, about **28% of the whole model**.
- **Why 896?** The Qwen team chose it (`hidden_size` in `config.json`). It sets how wide the model's "thinking space" is. Wider means more room for nuance, but more weights to read per word. Llama 8B uses 4,096.

### The prompt becomes a grid

Look up each word's row and stack them:

```
                  896 numbers ──────►
row 0  "The"      [ 0.01, -0.03, ... ]
row 1  "capital"  [-0.02,  0.04, ... ]
row 2  "of"       [ 0.03,  0.01, ... ]
row 3  "France"   [ 0.00, -0.02, ... ]
row 4  "is"       [ 0.02,  0.00, ... ]

= one grid of 5 × 896
```

**Everything the model does from here on is transforming this grid.** The grid keeps the shape 5 × 896 through all 24 layers. The numbers change; the shape doesn't.

---

## 3. Dictionary meaning vs meaning in context

Take the word **bank**:
- "I sat on the river **bank**"
- "I got a loan from the **bank**"

The embedding lookup gives "bank" **the same row in both sentences**, because the lookup doesn't know the sentence. That's the **dictionary meaning**.

Attention ([Chapter 4](04_attention.md)) is how "bank" reads the words around it. It looks back, finds "river" or "loan", and pulls their information into its row. After 24 layers, the "bank" row holds its **meaning in this sentence**.

> **Embedding** = dictionary meaning.
> **Attention** = reading the sentence.
> **Final row** = meaning in context.

---

## 4. The output match: 151,936 scores

After 24 layers, **take only the last row** (the "is" row). Only it has seen every word, because rows only look backward ([Chapter 4](04_attention.md)).

Compare it with **every row of the embedding table**. Each comparison is a dot product (multiply matching positions, then add) and gives one score. These raw scores are called **logits**.

Real scores after "The capital of France is":

```
rank  token      logit   probability
  1   ' Paris'   17.84   31.6%   ✅
  2   ' ______'  16.83   11.5%   ← ?!
  3   ' ____'    16.24    6.4%
  4   ' __'      16.10    5.5%
  5   ':\n'      16.03    5.1%
  6   ' located' 15.67    3.6%
```

**Why do the blanks score so high?** This is a **base model**. It learned from raw internet text, which is full of quizzes like "The capital of France is ______." It isn't answering you; it's predicting what text usually looks like. Chat models get extra training on top so they behave like assistants.

**Logits vs probabilities:** logits are raw scores. **Softmax** turns them into percentages that add up to 100% ([Chapter 4, step 6](04_attention.md#step-6-scores-become-percentages-softmax)). **Sampling** then picks one: always the top (greedy), or sometimes a lower one for variety (step 1.7).

### Weight tying: why there's no `lm_head.weight`

The output match needs a 151,936 × 896 table of word vectors, exactly the shape of the embedding table. Qwen2.5-0.5B **reuses the embedding table**: the same table turns words into rows at the start and rows back into words at the end. `config.json` says `"tie_word_embeddings": true`.

So the 272 MB dictionary is used **twice** per trip:
1. at the start: look up **one row** per new word (cheap, about 1.8 KB);
2. at the end: compare against **all 151,936 rows** (reads the whole 272 MB).

### Every row predicts the word after itself

The output match can be run on any row, not just the last. Real result from your model, at the end of the 24 layers:

```
row:        The        capital   of    France   is
predicts:   following  of        Ber   is       Paris
```

"capital" predicts "of", "France" predicts "is", "is" predicts "Paris": each row predicts **the word after its own position**. For generation we only need the last row, so real engines run the output match **only on the last row** and save 151,936 × 896 multiply-adds for every other prompt row. (During training, every row's prediction is used; that's how one sentence teaches the model many next words at once.)

---

## 5. Two different "matchings": don't mix them up

Both use the same math (a dot product gives a similarity score), but they are separate jobs:

| | Attention matching | Output matching |
| :--- | :--- | :--- |
| Where | Inside **every** layer | **Once**, at the very end |
| Compares | A word vs the **other words in your sentence** | The final row vs **all 151,936 vocabulary words** |
| Purpose | **Gather context** | **Pick the next word** |
| Uses | Wq, Wk, Wv | The embedding table |

---

## 6. A thought on prompts (your idea, refined)

Your idea: *"we should always end our question meaningfully, for better and non-hallucinated answers."*

- **What's right:** the prediction is read from the **last row**. A prompt that ends where the next word is obvious ("The capital of France is") narrows the answer a lot. That's why "Answer in one word:" works.
- **What's missing:** the last row gathered from **every** earlier word through attention, so the whole prompt matters, not only its ending.
- **What it can't fix:** hallucination. The output match **always** picks the highest-scoring token; there's no built-in "I don't know" box. If a fact isn't in the weights, the model still outputs whatever looks most likely. A clear prompt can't create a fact that isn't there.

---

## Check yourself

1. Why can't the model use the ID 12095 directly? What does the embedding give it?
2. A model has a vocabulary of 50,000 tokens. How many logits come out of each trip?
3. Why is there no `lm_head.weight` in the Qwen2.5-0.5B file?
4. Why do we use the **last** row's prediction and not the "France" row's?

<details>
<summary>Answers</summary>

1. An ID is a name tag with no meaning; nearby IDs aren't related. The embedding gives each token 896 coordinates in which similar meanings are close, so the model can do math on meaning.
2. 50,000: one score per vocabulary entry.
3. Weight tying: the output match reuses the embedding table (`tie_word_embeddings: true`).
4. Rows only look backward. The "France" row has never seen "is"; only the last row has seen the whole sentence. Also, each row predicts the word after its own position, and we want the word after the whole sentence.

</details>

---

> [← Chapter 1](01_what_is_inference.md) · [Contents](README.md) · [Chapter 3 →](03_weights_and_matrix_multiply.md)
