# Learning Journal: Where I Stand

> [Contents](README.md)

A living record of progress, honest assessments and open items. Update it at the end of every lesson.

---

## Coaching agreement

- **I write the learning code**; the coach teaches, gives hints and reviews. The tests decide what's correct.
- **Every lesson starts with an intro** and a "where we are on the map" recap. Tools are taught on toy data before any exercise.
- **A topic is "covered" only after I've answered its check question** or confirmed I understand it.
- **LLM autosuggest:** on for tests, scripts, boilerplate and forgotten numpy syntax; off (or single-line) inside `nanoinfer/` function bodies on the first attempt. Every code review includes "why this line?" questions; a line I can't explain gets rewritten.
- Finished work is committed with meaningful conventional-commit messages.

---

## Assessment (Phase 1, before the first line of code)

**Latest check-in score:** 7 out of 7 attempted, with 1 skipped (MLP internals).

| Area | Level | Evidence |
| :--- | :--- | :--- |
| Big picture: forward pass, prefill vs decode | **Solid** | Explains it back correctly; reasoned out batching unprompted. |
| Bandwidth arithmetic | **Solid** | Rebuilt 988 MB per pass from its parts (24 × 30 MB + 272 MB) without looking. |
| Attention mechanics | **Good** | Correct after clarification (k vs v wording, where Wo sits). |
| MLP internals (gate/up/down) | **Not yet** | Skipped on purpose; revisit before step 1.3. |
| Precision | **Needs work** | Right in my head, imprecise on paper ("v" for k). In code that's a silent bug. |
| Writing code | **Unknown** | Zero lines written so far. The biggest gap. |

**Strengths:** questions that refuse hand-waving ("why 64?", "why a separate book per layer?", "who designs Wq?"). Reached the core insight (decode leaves the GPU mostly idle) independently.

**Pattern to fix:** taking a correct idea and over-applying it (see [Corrections](corrections.md)). Code and tests cure this.

**Position:** about 15% into Phase 1, about 5% of the whole roadmap. Concept work is front-loaded, so it feels like more.

**Most useful change:** stop the Q&A loop and start writing code. Misconceptions get fixed faster by a failing test than by another analogy.

---

## Roadmap and time estimate

At about 8–10 focused hours a week (scale to your real hours):

| Stretch | Time | What it gets you |
| :--- | :--- | :--- |
| Phase 1: numpy transformer | 3–5 weeks | "I built one." Most people who talk about LLMs can't say that. |
| Phases 2–3: C engine, KV cache, quantization | 2–3 months | A CPU engine at a real % of bandwidth. Junior inference-engineer level. |
| Phases 4–5: Metal, CUDA kernels | 2–3 months | GPU kernels measured against cuBLAS. Where many people stall. |
| Phases 6–7: serving, multi-GPU | 2–3 months | Batching, paged attention. Hireable as an inference engineer. |
| Phase 8: upstream PRs | Ongoing | Merged work in vLLM or llama.cpp. Visible credibility. |

- **Solid, hireable level:** about 8–12 months of steady work.
- **"Among the best":** years. It comes from shipped work, profiling real systems and upstream contributions.

---

## Topic status

| Topic | Status |
| :--- | :--- |
| Training vs inference, the generation loop | ✅ |
| Tokens, embeddings, logits, weight tying | ✅ |
| Weights as wires, matrix multiply, parallel rows | ✅ |
| Attention (q, k, v, softmax, mix, Wo, heads, GQA) | ✅ |
| KV cache (concept, per layer, sizes) | ✅ |
| Layers, the classroom, why 24, tube not cone | ✅ |
| Bandwidth ceiling, compute vs bandwidth-bound, batching | ✅ |
| MLP role (fact store, per-layer books) | ✅ |
| MLP arithmetic (gate, up, down, SiLU) | ⏸ skipped, before step 1.3 |
| safetensors layout, little-endian | 🟡 Exercise A done |
| Tokenizer (BPE) | ⬜ step 1.2 |
| RMSNorm | ⬜ step 1.3 |
| RoPE (positions) | ⬜ step 1.4 |
| Sampling (temperature, top-k, top-p) | ⬜ step 1.7 |

---

## Open items

- [ ] Answer the tools-class check questions in [Chapter 9](09_safetensors_file_format.md#check-yourself).
- [ ] Lesson 1.1: finish the tools class (numpy `frombuffer`, `.view`, shifts), Exercise B, Exercise C, then `pytest tests/test_1_1_safetensors.py` green.
- [ ] Before step 1.3: answer the gate/up/down question ([Chapter 5](05_mlp.md#check-yourself), question 4).
- [ ] After Phase 1: write the reflection, "where does the missing 99% go?" ([Chapter 8](08_bandwidth_and_speed.md#6-where-your-phase-1-engine-will-land)).

---

## Log

| Date | What happened |
| :--- | :--- |
| 2026-09 | Phase 0 done: measured 92 GB/s CPU / 87 GB/s GPU. Phase 1 scaffold and 9-phase roadmap committed. |
| 2026-09 | Class 0 (how a transformer produces a word) covered through many rounds of Q&A. Exercise A done (N = 32,280, after the little-endian correction). |
| 2026-09-29 | This textbook written from the full session. First honest assessment. |
