# interaction-clustering: Generator-aware intent pattern discovery (pilot)

Micro-study testing whether **short, single-turn user messages** can be grouped into **interaction-level patterns** using **one embedding model + unsupervised clustering**.

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
```bash
pip install -U pyarrow
# or
pip install -U fastparquet
```

## Dataset generation (reproducible)

### 1) Optional: slot sanity check
Verify that all template slots are defined:
```bash
python3 generation/check_slots.py
```

### 2) Generate the dataset
```bash
python3 generation/generate_dataset.py
```

### Outputs
- `data/raw/messages.csv`
- `data/raw/messages.parquet` _(only if a Parquet engine is available)_

## Dataset columns

- `message_id` — unique identifier
- `text`— user message text
- `lang`— fixed to en
- `source` — synthetic
- `intent_gold`— gold intent label (for evaluation only)
- `generator_id`— generator family identifier
- `template_id` — stable template hash
- `seed`— per-row deterministic seed
- `split`— train_gen or test_gen
- `length_chars` — character length of the message
- `has_question_mark`— boolean
- `applied_perturbations`— exactly 3, semicolon-separated

## Sanity checks (required before clustering)
```bash
python3 -u generation/sanity_report.py
```
This prints:
- dataset balance checks
- split distributions
- message length statistics
- question mark rate
- perturbation frequency distribution
- sample messages per (intent, generator)
- quick keyword-overuse alarms (early leakage signal)

## Planned experiments (next stage)

### Embedding
- Single sentence embedding model
- No model benchmarking

### Clustering
- KMeans (fixed, justified k-selection)
- HDBSCAN (density-based)

### Evaluation
- **Intrinsic:** Silhouette, Davies–Bouldin
- **Extrinsic (using gold labels):** Purity, NMI (and/or ARI)

### Generator-aware reporting
- all data
- train_gen only
- test_gen only

### Qualitative analysis
- representative samples per cluster
- error cases under unseen generators

## Scope note

This is a pilot study intended as a reusable building block for my future Master’s thesis.
The contribution is methodological clarity and robustness analysis, not state-of-the-art performance claims.
Please reach out if any questions or concerns.