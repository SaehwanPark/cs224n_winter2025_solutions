# Stanford CS224n — Solutions & Projects (Winter 2025)

This repository contains my **personal solutions and course projects** from  
**Stanford CS224n: Natural Language Processing with Deep Learning (Winter 2025)**,  
offered between **January 7 and March 17, 2025**.

The materials here are published **after course completion** and are intended for:
- reinforcing my own understanding,
- showcasing implementation and reasoning skills,
- serving as a reference for practitioners already familiar with NLP concepts.

> ⚠️ **Academic Integrity Note**  
> This repository is **not recommended/intended for current or future students** to use for completing coursework.  

Course homepage: [https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1254/](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1254/)

---

## Quick Links

### Assignment Solutions

|         Item | Topic                                   | Materials                                                                                                                   |
| -----------: | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Assignment 1 | Word Vectors & Distributional Semantics | [Solution Notebook](solutions/assignment1-solution.ipynb)                                                                   |
| Assignment 2 | Neural Dependency Parsing               | [Problem PDF](https://web.stanford.edu/class/cs224n/assignments_w25/a2.pdf) · [Solution](solutions/assignment2-solution.md) |
| Assignment 3 | Neural Machine Translation              | [Problem PDF](https://web.stanford.edu/class/cs224n/assignments_w25/a3.pdf) · [Solution](solutions/assignment3-solution.md) |
| Assignment 4 | Attention & Transformers                | [Problem PDF](https://web.stanford.edu/class/cs224n/assignments_w25/a4.pdf) · [Solution](solutions/assignment4-solution.md) |


### Course+Extra Projects (Independent Implementations)

| Project                   | Model / Focus                 | Framework  | Repository                                                                                                           |
| ------------------------- | ----------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------- |
| Neural Dependency Parser  | Transition-based parsing      | JAX / Flax | [Github Repo](https://github.com/SaehwanPark/neural-dep-parser)                 |
| Neural Machine Translator | Seq2Seq + Attention           | JAX / Flax | [Github Repo](https://github.com/SaehwanPark/neural-machine-translator) |
| Minimal GPT-2             | Transformer LM (decoder-only) | JAX / Flax | [Github Repo](https://github.com/SaehwanPark/fp-gpt2)   |
| Minimal BERT (Extra)      | Transformer LM (encoder-only) | PyTorch | [Github Repo](https://github.com/SaehwanPark/minimal-bert) |
| Modern BERT (Extra)      | Transformer LM (encoder-only) | PyTorch | [Github Repo](https://github.com/SaehwanPark/modern-bert)   |


---

## Assignment Descriptions

### Assignment 1 — Distributional Semantics & Word Vectors

Focus areas:
- Word2Vec (skip-gram, naive softmax, negative sampling)
- Gradient derivations and geometric interpretations
- Practical embedding behavior and normalization effects

**My solution Provides:**
- Step-by-step mathematical derivations with clear intuition
- Explicit links between gradients and vector-space dynamics
- Commentary on when theoretical assumptions break in downstream tasks

---

### Assignment 2 — Neural Dependency Parsing

Focus areas:

- Transition-based dependency parsing
- Feature engineering for syntactic structure
- Neural classifier design and training dynamics

**My solution Provides:**

- Emphasis on *why* transition systems work, not just how
- Clear separation between algorithmic logic and model capacity
- Bridges classical parsing intuition with neural representations

---

### Assignment 3 — Neural Machine Translation (NMT)

Focus areas:

- Encoder–decoder architectures with attention
- Error analysis on real translation outputs
- BLEU score interpretation and limitations

**My solution Provides:**

- Linguistically grounded error analysis (plurality, idioms, repetition)
- Concrete model-level explanations for observed failures
- Practical improvement suggestions beyond the baseline architecture

---

### Assignment 4 — Attention & Transformers

Focus areas:
- Dot-product attention mechanics
- Single-head vs multi-head attention
- Variance, stability, and representation collapse

**My solution Provides:**
- Precise reasoning about attention distributions
- Probabilistic interpretation of key/query interactions
- Clear motivation for architectural design choices in Transformers

---

## Projects

All the course projects (Projects 1-3) were implemented **in JAX/Flax**, intentionally deviating from the
course’s default **PyTorch** requirement to explore:
- functional programming style,
- explicit state handling,
- JAX transformations (`jit`, `vmap`, `lax`).

Later extra projects (Projects 4-5) were implemented **in PyTorch** to compare development experience by myself.

### 1. Neural Transition-based Dependency Parser

Highlights:

- Immutable parser state as a JAX Pytree
- Fully vectorized inference
- Clean separation between parsing logic and neural scoring

---

### 2. Neural Machine Translator (NMT)

Highlights:

- Language-agnostic pipeline with SentencePiece
- BiLSTM encoder + attention-based decoder
- Beam search decoding and diagnostic tooling

---

### 3. Minimal GPT-2 (From Scratch)

Highlights:

- Faithful GPT-2 architecture in Flax
- Hugging Face → Flax parameter mapping
- Explicit, readable implementation over framework magic

---

### 4. minimalBERT (From Scratch): Extra / Implemented in PyTorch

Highlights:

- PyTorch encoder-only architecture from first principles
- Modular components with Hugging Face weight compatibility
- Dual pre-training objectives (MLM + NSP) with comprehensive tests

---

### 5. ModernBERT (From Scratch): Extra / Implemented in PyTorch

- Modern variant (released in 2024) of BERT with many advanced techniques
- Codebase size reduced by approx. 60% 
- Modular components with Hugging Face weight compatibility
- Fully implemented advanced features: alternating attention, dual RoPE scheduling, GeGLU activation, skip-first pre-norm, base-free LayerNorm, etc.

---

## Design Philosophy

Across assignments and projects, I optimized for:

- **Clarity over cleverness** — readable math, explicit assumptions
- **Interpretability** — understanding model behavior, not just scores
- **Framework literacy** — showing control over training, inference, and state
- **Faithful re-implementation** — no shortcuts hidden behind libraries

---

## Disclaimer

These materials reflect **my personal solutions and interpretations**.
They are **not official**, **not endorsed by Stanford**, and **not guaranteed optimal**.

If you are currently enrolled in CS224n, please refer only to:
👉 https://web.stanford.edu/class/cs224n/

---

## License

This repository is released under the **MIT License**, unless otherwise specified in individual project repositories.
