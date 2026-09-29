# Chapter 6: Layers and the Forward Pass (the Classroom)

> [← Chapter 5](05_mlp.md) · [Contents](README.md) · [Chapter 7 →](07_kv_cache.md)

**Reading time:** about 25 minutes. **Status:** ✅ covered.

## In one minute

- A layer = **attention** (words read each other) + **MLP** (each word looks up knowledge). Both **add** onto the row.
- There are **24 layers**, run **in order**. Each has its own weights, but identical shapes and identical cost (about 30 MB each).
- The row stays **896 wide** the whole way: the model is a **tube, not a cone**. What "converges" is the content, not the size.
- Why 24? Each layer is one round of "look back, then think", and answers need several rounds in sequence. The exact number is an educated trial-and-error choice.
- You watched " Paris" go from rank 33,039 at layer 8 to rank 1 at layer 22.

---

## 1. The classroom (the analogy used throughout)

### The setup

**5 students** sit in a row, one per word:

```
seat:      1      2       3     4       5
student:  The  capital   of  France    is
```

**Rule of the room:** a student may only look at students to their **left** (earlier words) and at themselves. Never to the right.

**Each student has one notebook page with 896 boxes** (their row). At the start of the day, each copies **their own word's definition** from the class dictionary onto their page (the embedding). At this point, "is" knows only "I am the word *is*".

**The school day has 24 periods** (the layers). **Each period has its own teacher**, who brings a **toolkit** (that layer's weights):

| In the toolkit | Model name |
| :--- | :--- |
| A form for writing a **question card** | Wq |
| A form for writing a **label card** | Wk |
| A form for writing a **handout** | Wv |
| A **translator** | Wo |
| A **reference book** of 4,864 "if you see X, write Y" rules | MLP |

### One period

**Step A: look left and copy (attention)**

- **A1.** Every student fills out three cards from their own page, using the teacher's forms: a **question card** (q, "what I'm looking for"), a **label card** (k, "what I'm about", left on the desk for others), a **handout** (v, "what I'll give to anyone who picks me", also left on the desk).
- **A2.** Each student compares their question card with the label cards on their own desk and every desk to the left. Each comparison is a **score**.
  → "is" compares with 5 label cards; France's scores highest.
- **A3.** The scores become **percentages** adding to 100%.
  → "is": France 98%, capital 1.8%, itself 0.2%.
- **A4.** The student takes that percentage of each handout and blends them. **The blend is the mix.**
- **A5.** The **translator** turns the mix into notebook format, and the student **adds** it to their page. Nothing is erased.
  → The "is" page now carries France-ness.

All 5 students do Step A **at the same time**, each from their own seat. (In the real model, each student actually writes 14 question cards per period, one per head, and 2 label cards and 2 handouts: [Chapter 4](04_attention.md).)

**Step B: think alone (MLP)**

Each student opens **this period's reference book** and checks its 4,864 rules against **their own page**. They write the results of the rules that fire.
→ "is" now shows capital + France, so "capital AND France → write Paris-ness" fires.

No looking at anyone else during Step B.

**The bell rings.** The next period starts with a **new teacher, new forms, a new translator and a new reference book.** The pages carry over, now richer.

### End of the day (after period 24)

**The last student ("is") compares their page with every entry in the class dictionary.** The closest entry wins: " Paris". The other students' pages also point at a next word, but we ignore them.

### The next day: a newcomer (decode)

"Paris" sits in **seat 6**, copies the Paris definition onto a fresh page, and goes through **all 24 periods alone**.

In each period's Step A, Paris needs the label cards and handouts of students 1–5 **for that period**. Have they changed? **No.** Those students only look left, so a newcomer on their right can't change anything they wrote. The school kept their cards in a **filing cabinet, one drawer per period**. That's the **KV cache** ([Chapter 7](07_kv_cache.md)).

### Where speed comes from

**The teachers' toolkits are stored in the basement** (memory chips). Each period, the toolkit is carried upstairs to the classroom (the memory pipe, 92 GB/s).
- **Prefill:** one trip upstairs serves **5 students**.
- **Decode:** one trip upstairs serves **1 student**. The carrying is the bottleneck ([Chapter 8](08_bandwidth_and_speed.md)).

### Classroom ↔ model cheat sheet

| Classroom | Model |
| :--- | :--- |
| Student | Word (token) |
| Notebook page (896 boxes) | Row (activations) |
| Only look left | Causal mask |
| Period | Layer |
| Teacher's toolkit | That layer's weights |
| Question / label / handout form | Wq / Wk / Wv |
| Translator | Wo |
| Reference book | MLP (gate, up, down) |
| Class dictionary | Embedding table |
| Filing cabinet, one drawer per period | KV cache, per layer |
| Basement → classroom stairs | Memory → compute cores (bandwidth) |

---

## 2. One layer, precisely

Every layer has this exact structure:

```
row (896)
  │
  ├─ RMSNorm                       (keeps numbers in range)
  ├─ attention: Wq Wk Wv → scores → softmax → mix → Wo
  ├─ ADD to row                    ← other words' influence
  │
  ├─ RMSNorm
  ├─ MLP: gate, up → switch → down
  ├─ ADD to row                    ← knowledge
  ▼
row (896) → next layer (its own weights)
```

(RMSNorm rescales the row so its numbers don't grow or shrink out of control over 24 layers. You'll build it in step 1.3; each norm has 896 learned scales. After layer 24 there's one final norm before the output match.)

### The life of the "is" row in one layer

```
"is" row
  ├─ × Wq → q_is ─┐
  ├─ × Wk → k_is  │ (saved: later words search it)
  ├─ × Wv → v_is  │ (saved: later words copy it)
  │               ▼
  │   q_is · [k_The … k_France, k_is] → scores → %
  │   mix = % × [v_The … v_France, v_is]
  │
  ├─ row = row + mix × Wo   ← PREVIOUS WORDS enter
  ├─ row = row + MLP(row)   ← KNOWLEDGE enters
  ▼
next layer
```

### The two ADD lines, with numbers

A 3-number row with human labels (real rows don't have labels):

```
position: [ capital-ness, France-ness, Paris-ness ]
```

The "is" row at the start of the layer: `[1, -1, 0]`.

**Line 1: `row = row + mix × Wo`.** Attention found France at 98%; the mix is `[0.019, 2.935]`. The toy Wo copies it into positions 0 and 1:

```
[1, -1, 0] + [0.019, 2.935, 0] = [1.019, 1.935, 0]
                                  ↑ France-ness arrived
```

**This is the only line in the whole model where information moves from one word to another.**

**Line 2: `row = row + MLP(row)`.** Detector "capital AND France" fires with strength 0.954 and writes 0.954 × `[0, 0, 3]`:

```
[1.019, 1.935, 0] + [0, 0, 2.862] = [1.019, 1.935, 2.862]
                                                  ↑ Paris-ness
```

That's one layer. The model repeats it 24 times: compute, then add, **48 times** in total.

---

## 3. Watching " Paris" appear (logit lens)

**Experiment:** after each layer, take the "is" row and run the output match on it early: "if the model stopped here, which word would it say?" Real output from your model:

```
layer  top guess        " Paris" rank
  0    ' is'                 4,609
  1    ';"\n (junk)          3,915
  8    'апр' (junk)         33,039
 12    ';"\n (junk)         37,809
 16    ' ____'               6,412
 18    ' ____'               3,552
 20    ' __'                   488
 21    ' __'                    82
 22    ' Paris'  (+ 巴黎)       1 ✅
 23    ' Paris'                 1
 24    ' Paris'  (final)        1   31.6%
```

What this shows:
1. **At the start, the "is" row just means "is"** (the plain embedding).
2. **Layers 1–15 look like gibberish.** The row is in the model's own working shorthand, not English, like a student's rough work: messy halfway, only the final answer is readable.
3. **Around layer 16, the model realizes "an answer slot goes here"** (the blanks), before it knows *what* the answer is.
4. **At layer 22, " Paris" appears, together with 巴黎 ("Paris" in Chinese).** The model is holding the *concept* of Paris, not the English spelling.

Every student's page, after selected periods (what each would predict):

```
period  The        capital  of          France  is
  0     The        capital  of          France  is
  1     **         ization  lin         밌      ';"
 10     ascus      家都知道  家都知道      iện     ';"
 20     ascus      ization  whiteColor  is      __
 22     🏽         ization  :<?         is      Paris
 24     following  of       Ber         is      Paris
```

Every student predicts the word **after their own seat**: "France" predicts "is", "capital" predicts "of".

> Rerun it yourself: [`scripts/logit_lens.py`](../../learn_projects/inference_from_scratch/scripts/logit_lens.py).

**Why "22" is not special:** the model always runs **all 24** layers for every word. 22 is just where " Paris" first became readable **for this sentence**. A harder sentence might only settle at layer 24; the model can't know in advance.

---

## 4. A tube, not a cone

Diagrams often draw networks narrowing to the right, like a funnel. That's true for **older** networks, such as a handwritten-digit classifier:

```
784 → 128 → 64 → 10
█████████
 ███████
   ████
    ██          a real funnel
```

**A transformer is a straight tube**, with a bulge inside every layer and a flare at the very end:

```
5 words → [896] ─ L1 ─ [896] ─ ... ─ L24 ─ [896]
                                     → 151,936 scores

inside each layer's MLP: 896 → 4,864 → 896

      ▄█▄  ▄█▄  ▄█▄        ▄█▄      ██████
  ════███══███══███═ ... ══███══════██████
      ▀█▀  ▀█▀  ▀█▀        ▀█▀      ██████
       L1   L2   L3        L24    output match
```

1. **Constant width:** every layer takes 896 in and gives 896 out, for every word, so every layer costs the same (about 14.9M multiply-adds per row, about 30 MB of weights).
2. **A bulge per layer:** the MLP expands to 4,864 detectors, then compresses back.
3. **The end gets wider, not narrower:** 896 → 151,936 scores, then 1 chosen token.

**"Converging" describes the content, not the shape.** The page never shrinks; the writing on it becomes more decided (rank 33,039 → 1).

---

## 5. Why 24 layers? Why not 2, or 100?

### Why not 1 or 2?

Each layer allows **one round** of "look back, then think". Many answers need several rounds **in order**, because round 2 needs round 1's result already written on the row:

```
"The capital of the country where the Eiffel Tower is ___"

round 1: gather "Eiffel Tower" → recall "France" (write it)
round 2: gather "capital" + that France → "Paris"
```

Researchers have shown that even the simplest skill, "copy a pattern that appeared earlier in the text" (an *induction head*), needs **at least 2 attention layers in sequence**. Real language needs many more rounds.

### Why not 100?

Every layer costs time on every word, and layers run **one after another** (layer 5 needs layer 4's output):

```
24 layers:  988 MB per word  → ~93 words/sec on this M3
100 layers: ~3.25 GB per word → ~28 words/sec
```

Very deep models are also harder to train.

### So is it trial and error?

**Mostly, yes: educated trial and error.** Designers have a budget (say, 0.5B numbers) and choose how to spend it:
- **Deep and narrow:** many rounds, but a small row (little room for ideas).
- **Shallow and wide:** a big row, but few rounds of reasoning.

The **scaling laws** research found quality is surprisingly similar across a wide range of shapes. Teams train small test models, pick a shape that works, and prefer GPU-friendly sizes (multiples of 64 or 128). Bigger models get both wider and deeper:

| Qwen2.5 | Layers | Row width |
| :--- | ---: | ---: |
| 0.5B | 24 | 896 |
| 7B | 28 | 3,584 |
| 72B | 80 | 8,192 |

### Layers are sequential: an inference consequence

You can't compute layer 5 before layer 4 finishes, for the same word. This sets a floor on latency per word, and later it's why **pipeline parallelism** (placing different layers on different GPUs) exists.

---

## 6. The whole forward pass, one screen

```
text → tokenizer → IDs
IDs  → embedding lookup → grid (n × 896)
for each of 24 layers (in order):
    grid += attention(norm(grid))  (rows read earlier rows)
    grid += MLP(norm(grid))        (each row alone)
last row → final norm → × embedding table
         → 151,936 logits → sample → next token
```

Not covered yet: **positions**. Attention as described can't tell "dog bites man" from "man bites dog". Qwen stamps each word's position into q and k with **RoPE** (step 1.4).

---

## Check yourself

1. Your summary: in the classroom, which step (A or B) needs to see other students' pages?
2. Does layer 20 do less work than layer 2 because it's "closer to the answer"?
3. The model found " Paris" at layer 22. Could we stop there and save 2 layers?
4. Where does information from earlier words enter the "is" row? Where does knowledge enter?
5. Explain the whole forward pass to a friend in 5 sentences, without looking.

<details>
<summary>Answers</summary>

1. Step A (attention). Step B (MLP) is done alone.
2. No. Every layer has identical shapes, so identical compute and identical weight traffic (~30 MB). Only the content of the row changes.
3. Not in general. The model can't know in advance which layer will be enough; a harder sentence might need all 24. (Research on "early exit" tries this, but standard models always run every layer.)
4. Earlier words: `row = row + mix × Wo` (attention). Knowledge: `row = row + MLP(row)`.
5. One version: all prompt words enter at once as a grid, one row per word. Each of 24 layers updates every row using that layer's fixed weights: attention lets each row add in information from earlier rows, then the MLP adds knowledge to each row alone. Rows only look backward, so the last row is the only one that has seen the whole sentence. After the last layer, that row is compared against every entry of the embedding table, giving 151,936 scores. The highest-scoring token is the next word, which is appended and the loop runs again.

</details>

---

> [← Chapter 5](05_mlp.md) · [Contents](README.md) · [Chapter 7 →](07_kv_cache.md)
