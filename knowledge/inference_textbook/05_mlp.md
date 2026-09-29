# Chapter 5: The MLP, the Model's Reference Books

> [← Chapter 4](04_attention.md) · [Contents](README.md) · [Chapter 6 →](06_layers_and_the_forward_pass.md)

**Reading time:** about 20 minutes. **Status:** 🟡 the role of the MLP is covered; **the gate/up/down arithmetic (section 3) is not yet covered.** You skipped it on purpose; it comes back before step 1.3 (`swiglu_mlp`).

## In one minute

- The MLP is a room of **4,864 detectors per layer**. Each detector checks the row for one learned **pattern**; if it fires, it **writes its note** onto the row.
- The MLP sees **only its own row** (blinkers on). Attention looks around; the MLP looks inside.
- Attention can only **move** information already in the sentence. The MLP is the **only** place information from **outside** the sentence can enter, so training turns it into the **fact store**.
- It is **64% of the model** (628 MB).
- Every layer has **its own** MLP, because the row looks different at each stage.

---

## 1. Detectors: the idea, with a 3-number row

Give the row's positions human labels so you can follow along (real rows don't have labels; meaning is spread over all 896 numbers):

```
position:  [ capital-ness, France-ness, Paris-ness ]
```

After attention ([Chapter 4](04_attention.md)), the "is" row is `[1.019, 1.935, 0]`: it carries capital and France information.

A toy MLP with 2 detectors:

```
D1: pattern "capital AND France" = [1, 1, 0]
    threshold 2, note to write   = [0, 0, 3]
D2: pattern "capital, NOT France" = [1, -1, 0]
    threshold 0, note            = (something else)
```

**Step 1. Every detector checks the row.** Score = row · pattern − threshold. Anything below zero becomes 0 (the detector stays off).

```
D1: 1.019×1 + 1.935×1 + 0×0 − 2 =  0.954 → ON
D2: 1.019×1 + 1.935×(−1) + 0 − 0 = −0.916 → OFF (0)
```

**Step 2. Every active detector writes its note, scaled by its strength.**

```
MLP(row) = 0.954 × [0, 0, 3] + 0 × (D2's note)
         = [0, 0, 2.862]
```

**Step 3. ADD it to the row.**

```
row = [1.019, 1.935, 0] + [0, 0, 2.862]
    = [1.019, 1.935, 2.862]
                     ↑ Paris-ness appeared
```

At the end, the output match: the " Paris" embedding (in this toy `[0, 0, 1]`) · row = 2.862, the highest score, so " Paris" wins.

### Why attention must come first

D1 needs **both** capital **and** France. The raw "is" row `[1, -1, 0]` scores 1 − 1 − 2 = −2, so D1 stays off. **Attention had to bring France's notes in first.** Attention gathers the ingredients; the MLP cooks with them. That's why every layer runs attention, then MLP.

---

## 2. The patterns are weights, not predefined rules

Nobody wrote "capital AND France". The patterns and notes are **learned weights**:

```
gate_proj, up_proj  [4864, 896]
  → each of the 4,864 rows is one detector's PATTERN
down_proj           [896, 4864]
  → each of the 4,864 columns is one detector's NOTE
```

Training found that a detector with roughly that pattern and a Paris-like note lowered the error, so it nudged some detector toward it. Many real detectors respond to things no human would name.

**Sizes:** 3 × 4,864 × 896 = **13,074,432 numbers per layer** (26 MB). × 24 layers = **628 MB, 64% of the model**, about 117,000 detectors in total.

---

## 3. Gate, up, down: how Qwen's detectors really work

> ⏸ **Not yet covered.** You chose to skip this and learn it hands-on. It must be answered before step 1.3. Read it when you get there.

The toy above used one pattern and a threshold. Qwen's MLP (**SwiGLU**) reads the row **twice** per detector:

- **gate:** does this detector fire? (a smooth on/off switch)
- **up:** how strongly, and with which sign?
- **down:** what gets written?

Everything that touches the row is 896 long, because it reads from or writes to the 896-box row. **4,864 is simply how many detectors there are.** The row briefly widens to 4,864 and then shrinks back to 896: a bulge inside every layer.

```
row (896) ──× gate──► 4,864 scores ─► SiLU (on/off)
         ──× up────► 4,864 amounts        │
                               multiply ◄─┘
                     4,864 strengths
                          ──× down──► 896 ─► ADD to row
```

**The on/off switch, SiLU:** `silu(x) = x × sigmoid(x)`. A large positive x passes through almost unchanged; zero or negative becomes about 0.

```
silu(3) = 2.858    silu(5) = 4.967
silu(0) = 0        silu(−1) = −0.269  (small)
```

### Worked toy: 3-box row, 2 detectors

The row is `[1, 2, 0]`.

```
gate patterns: D1 [1,1,0]   D2 [0,0,1]
up patterns:   D1 [0,1,0]   D2 [1,0,0]
notes (down):  D1 [0,0,0.5] D2 [1,0,0]
```

**Step 1, gate.** Does each detector fire?
```
D1: [1,2,0]·[1,1,0] = 3 → silu(3) ≈ 2.86 (on)
D2: [1,2,0]·[0,0,1] = 0 → silu(0) = 0     (off)
```

**Step 2, up.** How strongly?
```
D1: [1,2,0]·[0,1,0] = 2
D2: [1,2,0]·[1,0,0] = 1
```

**Strength = gate switch × up:**
```
D1 = 2.86 × 2 = 5.72
D2 = 0    × 1 = 0
```

**Step 3, down.** Each detector writes strength × its note:
```
5.72 × [0, 0, 0.5] = [0, 0, 2.86]
0    × [1, 0, 0]   = [0, 0, 0]
MLP(row) = [0, 0, 2.86]
```

**Add to the row:** `[1, 2, 0] + [0, 0, 2.86] = [1, 2, 2.86]`.

In one line: **gate decides whether a detector fires, up decides how strongly, down decides what gets written.** Turning this into numpy, for a whole grid of rows at once, is the `swiglu_mlp` you'll write in step 1.3. (Remember the file stores weights as `(out, in)`, see [Chapter 3](03_weights_and_matrix_multiply.md).)

---

## 4. Why the MLP becomes the fact store

Compare what each part is **able** to do:

| | Can see | Can do |
| :--- | :--- | :--- |
| Attention | Other words' rows | **Move** information already somewhere in the sentence |
| MLP | Only its own row | **Add** information from its weights |

**Where can " Paris" come from?** The word Paris isn't in "The capital of France is". Attention can only copy what other words already hold, so it can't produce Paris. **The only source outside the sentence is the MLP weights.**

So during training, whenever the model needed a fact that wasn't in the text, the only way to lower the error was to nudge MLP weights into holding it. The MLP becomes the fact store because **it's the only place that can bring in information from outside the sentence.** (Research on MLPs as "key-value memories" backs this up. Real models are messier: facts are spread across several middle layers.)

**Your framing, refined:** attention "looks everywhere to the left"; the MLP "wears blinkers" and sees only its own row, contrary to what the names suggest. Attention doesn't carry a *question* into the row; it carries *information* ("capital + France"). The MLP recognizes the combination and writes "Paris-ness".

---

## 5. Why does every layer have its own reference book?

**"Facts are facts, they're universal. Why not one god-level fact book that every layer uses? Or a book only in the last layer?"**

The key idea: **the book isn't organized by facts. It's organized as "if the row looks like X, add Y".** The "if" part is a detector pattern, and it only fires on a row that looks a certain way. **The row looks very different at different layers:**

| Layers | What the row mostly holds | What that layer's book does (roughly, found by researchers probing) |
| :--- | :--- | :--- |
| Early (1–6) | Word pieces, spelling, grammar | Join word pieces ("Par"+"is", "New"+"York" is a place), word types |
| Middle (7–18) | Meaning, who relates to whom | **Facts:** capital + France → Paris-ness |
| Late (19–24) | "The answer is Paris" | Formatting: produce the exact token `' Paris'` (leading space, capital P), push down rivals |

You saw this in the logit lens ([Chapter 6](06_layers_and_the_forward_pass.md)): Paris-ness was built in the middle layers, but only became the top **readable** word at layer 22, after the late layers formatted it.

**Why not one shared book?** It has been tried (ALBERT). It works worse for the same compute, because one set of patterns can't match rows at 24 different stages: a single key for 24 different locks.

**Why not a book only at the last layer?** Chains need more than one lookup:

```
"The capital of the country where the Eiffel Tower is ___"

round 1: gather "Eiffel Tower" → book: "that's France"
         (write France onto the row)
round 2: gather "capital" + the France just written
         → book: "Paris"
```

With one book at the end, you get one lookup and no chains. Separate books also give **24× more room** for rules.

**Nobody wrote these books.** Training found that dividing the work this way lowers the error. The table above is what researchers later discovered by probing, and the boundaries are fuzzy.

---

## Check yourself

1. Where can information that isn't in the sentence enter the model: attention, MLP, or both? Why?
2. In the section 1 toy, what would happen at the MLP if attention had been skipped (row still `[1, -1, 0]`)?
3. Why can't one shared fact book do the job of 24 separate ones?
4. ⏸ *(Pending, answer before step 1.3.)* Using the section 3 toy (gate D1 `[1,1,0]`, D2 `[0,0,1]`; up D1 `[0,1,0]`, D2 `[1,0,0]`; notes D1 `[0,0,0.5]`, D2 `[1,0,0]`), the row is `[0, 0, 5]`. Which detector's gate fires, and what gets added to the row? Take silu(5) ≈ 5.

<details>
<summary>Answers (try question 4 on paper first)</summary>

1. Only the MLP. Attention can only move information already in other rows of the sentence; the MLP adds information from its weights.
2. D1 scores 1 − 1 − 2 = −2, so it stays off: nothing about Paris is added. The MLP needs attention's ingredients.
3. A detector's pattern only fires on a row that looks a certain way, and the row looks different at each stage (spelling → meaning → answer formatting). Separate books let each layer specialise, and allow multi-step chains.
4. Gate: D1 = 0 → off; D2 = 5 → silu ≈ 5, on. Up: D2 = `[0,0,5]·[1,0,0]` = 0. Strength = 5 × 0 = 0. **Nothing is added** (`[0,0,0]`). The gate says "fire", but up says "with strength 0": both readings must agree. That's the point of having two.

</details>

---

> [← Chapter 4](04_attention.md) · [Contents](README.md) · [Chapter 6 →](06_layers_and_the_forward_pass.md)
