# interaction-clustering — Generator-aware intent pattern discovery (pilot)

Publication-ready micro-study (2–3 weeks) testing whether **short, single-turn user messages** can be grouped into **interaction-level patterns** using **one embedding model + unsupervised clustering**.

**Key contribution:** *generator-aware evaluation*  
We explicitly test whether clustering quality and structure remain stable under **unseen template families**, probing **template leakage** in synthetic datasets.

---

## Research question

Can meaningful and measurable interaction-level patterns be discovered from short user messages using text embeddings and unsupervised clustering, under realistic ambiguity and synthetic data constraints?

---

## Dataset design (synthetic, controlled)

- Language: **English only**
- Size: **2,400** short messages
- Intents (8):
  1) information_seeking  
  2) how_to  
  3) troubleshooting  
  4) summarization  
  5) recommendation  
  6) planning  
  7) creative  
  8) math
- Generator families (5) per intent:
  - `direct`, `polite`, `contextual`, `constraint_heavy`, `noisy`
- Messages per (intent, generator): **60**
- Perturbations:
  - Apply **exactly 3** per message (category-agnostic) to simulate real-world noise.

### Generator-aware split
- `train_gen`: `direct`, `polite`, `contextual`
- `test_gen`: `constraint_heavy`, `noisy`

This enables evaluating clustering robustness to **unseen prompt/template families**.

---

## Repository structure

