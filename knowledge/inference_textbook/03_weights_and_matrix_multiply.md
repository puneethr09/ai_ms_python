# Chapter 3: Weights, Wires and Matrix Multiplication

> [← Chapter 2](02_tokens_and_embeddings.md) · [Contents](README.md) · [Chapter 4 →](04_attention.md)

**Reading time:** about 15 minutes. **Status:** ✅ covered.

## In one minute

- A **weight matrix** is a grid of learned numbers. "Row × W" gives a new row; that's the only operation the model really does.
- **Weights are the wires, not the nodes.** The nodes are the row's numbers (temporary, per sentence). The wires are the weights (fixed, in the file).
- **Every word uses all the same wires**, like one cookie cutter pressed into many lumps of dough. That's why rows can be processed in parallel.
- **Nobody designs the numbers.** Humans choose the shapes; training sets the values.
- "0.5B parameters" means about 494 million learned numbers: 988 MB at 2 bytes each.

---

## 1. Row × W, by hand

Take one row, "France" = `[1, 0, 2]`, and a weight matrix W learned in training:

```
        col0 col1 col2
W  =  [  1    0    2  ]
      [  0    3    1  ]
      [  2    1    0  ]
```

Each output number = multiply the row with one **column** of W, then add up:

```
France = [1, 0, 2]

out[0] = 1×1 + 0×0 + 2×2 = 5   (column 0: 1,0,2)
out[1] = 1×0 + 0×3 + 2×1 = 2   (column 1: 0,3,1)
out[2] = 1×2 + 0×1 + 2×0 = 2   (column 2: 2,1,0)

new France row = [5, 2, 2]
```

**How is that "knowledge"?** Think of each column as a question the model learned to ask: "how much do you match *this* pattern?" Column 0 is the pattern `(1, 0, 2)`; France matches it strongly, so it scores 5. Training chose these patterns. A real matrix has hundreds or thousands of columns, so a row gets asked that many learned questions at once, and the answers become its new row.

> **File layout note for step 1.3:** in the safetensors file, weights are stored as **(out_features, in_features)**. For example, `k_proj.weight` is `[128, 896]`: 128 outputs, 896 inputs. So in numpy you compute `x @ W.T`.

---

## 2. Weights are wires, not nodes

Neural network diagrams show circles joined by lines. People often read them the wrong way round:

```
 input row (896)            output q (896)
     ○ ──────────────────────── ○
     ○ ─────────╲──────╱─────── ○
     ○ ──────────╲────╱──────── ○
     ...  every ○ connects to every ○
     ○ ──────────────────────── ○

each line = ONE number in Wq
```

- **The nodes (circles)** are the numbers of a row. They're temporary and different for every word. These are the **activations**.
- **The wires (lines)** are the weights. Wq is 896 × 896, so it has **802,816 wires**, and each wire is one number in the file.

"Row × Wq" means: each output node = the sum of (every input node × the wire connecting them).

So Wq isn't *in* a node. **Wq is the complete set of wires between the row and the q card.**

### Weights vs activations

| | Weights | Activations (rows) |
| :--- | :--- | :--- |
| What | The 494M learned numbers in the file | The 5 × 896 grid for *your* sentence |
| Changes? | **Never** during inference | **Every layer** |
| Size | 988 MB | Tiny: 5 × 896 = 4,480 numbers |
| Analogy | A chef's recipe book | The ingredients being cooked |

For every word, the machine reads the whole 988 MB recipe book to cook a tiny grid. That imbalance is why memory speed is the wall ([Chapter 8](08_bandwidth_and_speed.md)).

---

## 3. Every word uses the same wires (the cookie cutter)

There's no "wire for France" and "wire for is". **Every word goes through exactly the same wires.**

Picture one cookie cutter and five lumps of dough. You press the same cutter into each lump and get five different cookies, because each lump was different.

- the cutter = the wires (e.g. Wq)
- the lumps = the word rows
- the cookies = each word's result

With numbers, using `Wq = [[0, 1], [0, -1]]`:

```
          row        same wires            q
capital  [2,  0]  → [2×0+0×0,  2×1+0×(-1)]  = [0,  2]
France   [0,  2]  → [0×0+2×0,  0×1+2×(-1)]  = [0, -2]
is       [1, -1]  → [1×0-1×0,  1×1+(-1)(-1)] = [0,  2]
```

### Why rows can run in parallel

Computing France's new row never needs capital's numbers. Each row is its own calculation, and all share the same W. So stack the rows into a grid and do them all in one operation:

```
[ capital ]           [ new capital ]
[ France  ]  ×  W  =  [ new France  ]
[ is      ]           [ new is      ]
```

**Weights never mix rows.** They transform each row on its own. (The only place rows mix is attention, [Chapter 4](04_attention.md).)

**The payoff:** W is read from memory **once** and used for all rows.
- Prefill: 988 MB read, **5 rows** computed. Efficient.
- Decode: 988 MB read, **1 row** computed. Wasteful.

Same weights, very different efficiency. [Chapter 8](08_bandwidth_and_speed.md) builds on this.

---

## 4. Where the numbers come from: training

**Humans choose only the shapes:** 24 layers, 896 wide, 14 heads, 4,864 MLP detectors. **Every number starts random** and is set by training.

### Training with one weight

Say the whole "model" is `prediction = w × input`, the input is 2, and the correct answer is 6:

```
w = 1  → prediction 2, should be 6 → too low → nudge w up
w = 2  → prediction 4, still low   → nudge up
w = 3  → prediction 6              → correct ✅
```

Real training does exactly this, at scale:
1. Show the model text: "The capital of France is".
2. It predicts the next word. At first, random garbage.
3. Compare with the real next word (" Paris") and measure how wrong it was.
4. For **each** of the 494M numbers, calculus answers: "if I nudge this number slightly, does the error go down?" Nudge **all** of them a tiny bit in the helpful direction. (This is **gradient descent**.)
5. Repeat for trillions of words. Qwen2.5 was trained on about 18 trillion tokens.

### Attention and MLP are trained the same way

It's the same training, the same error signal, and all weights are nudged at once. They end up with different jobs only because of **where they sit** in the circuit:
- Wq and Wk can only affect **who looks at whom**, so they become a search system.
- Wv and Wo can only affect **what gets copied**.
- The MLP can only transform **one row by itself**, so it becomes the fact store ([Chapter 5](05_mlp.md)).

Same school, same exams, but students become good at whatever their seat lets them do.

**Training is done before you download the model. Inference only uses the frozen numbers.**

---

## 5. What "0.5B parameters" means

A **parameter** is one learned number in the file. Counted from your real file:

| Per layer | Numbers |
| :--- | ---: |
| Wq + Wk + Wv (+ their biases) | 1,033,344 |
| Wo | 802,816 |
| MLP: gate + up + down (3 × 4,864 × 896) | 13,074,432 |
| 2 norm scales | 1,792 |
| **One layer** | **14,912,384** (≈ 30 MB) |

| Whole model | Numbers | Bytes (BF16) | Share |
| :--- | ---: | ---: | ---: |
| 24 layers | 357,897,216 | 716 MB | 72% |
| Embedding table | 136,134,656 | 272 MB | 28% |
| Final norm | 896 | 2 KB | ~0% |
| **Total** | **494,032,768** | **988 MB** | |

494 million ≈ "0.5B", which is where the name comes from. A "7B" model has 7 billion numbers, 14 GB in BF16. **More numbers = more bytes to move per word = slower.** That's why model size predicts speed so directly.

---

## Check yourself

1. Wq has 802,816 numbers. Are they nodes or wires? Which of the two changes for every sentence?
2. With 100 words in the prompt, how many copies of Wq are used?
3. Who decided the values inside Wq? Who decided its shape?
4. Why is prefill more efficient than decode, in terms of weight reads?

<details>
<summary>Answers</summary>

1. Wires (weights). The nodes (the row's numbers, the activations) change for every sentence; the wires never change during inference.
2. One. All 100 rows go through the same Wq (the cookie cutter).
3. Training decided the values (random start, then nudged). Humans decided the shape (896 × 896).
4. Each weight read from memory is used for all prompt rows in prefill, but for only one row in decode.

</details>

---

> [← Chapter 2](02_tokens_and_embeddings.md) · [Contents](README.md) · [Chapter 4 →](04_attention.md)
