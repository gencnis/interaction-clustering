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
  - `direct`
  - `polite`
  - `contextual`
  - `constraint_heavy`
  - `noisy`

- Messages per (intent, generator): **60**
- Perturbations:
  - Apply **exactly 3** category-agnostic perturbations per message to simulate realistic user noise.

### Generator-aware split

- `train_gen`: `direct`, `polite`, `contextual`
- `test_gen`: `constraint_heavy`, `noisy`

This split enables evaluating whether clustering structure and quality persist under **unseen template / prompt families**.

---

## Repository structure

```text
interaction-clustering/
├── data/
│   ├── raw/                 # generated messages.csv
│   └── processed/           # reserved for downstream artifacts
├── generation/
│   ├── templates.py
│   ├── slot_pools.py
│   ├── perturbations.py
│   ├── check_slots.py
│   ├── generate_dataset.py
│   └── sanity_report.py
├── experiments/
│   └── clustering.ipynb     # embeddings + KMeans/HDBSCAN + evaluation
└── results/
    └── figures_tables/      # plots and tables (soon)
```

## Setup

Python **3.9+** recommended.

### Minimal dependencies

```bash
pip install -U pandas numpy scikit-learn
```

### Optional (for Parquet support)
```pip install -U pyarrow
# or
pip install -U fastparquet
```

## Dataset generation (reproducible)

### 1) Optional: slot sanity check
Verify that all template slots are defined:
```python3 generation/check_slots.py```

### 2) Generate the dataset
```python3 generation/generate_dataset.py```

### Outputs
- `data/raw/messages.csv`
- `data/raw/messages.parquet` (only if a Parquet engine is available)