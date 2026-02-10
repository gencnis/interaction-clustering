"""
generate_dataset.py

Generate the synthetic dataset for:
"User Interaction Pattern Discovery Using Embedding-Based Clustering"

Dataset specification (locked for this micro-study):
- Language: English only
- Size: 2,400 messages
- Intents: 8
- Generator families per intent: 5
- Messages per generator family: 60

Key contribution support:
- Generator-aware evaluation:
  We assign a split label based on generator family:
    train_gen = {direct, polite, contextual}
    test_gen  = {constraint_heavy, noisy}
  This enables robustness testing against template leakage.

Outputs:
- data/raw/messages.csv
- data/raw/messages.parquet  (if pyarrow is available; otherwise parquet save will be skipped)

Columns:
- message_id (str)
- text (str)
- lang (str)                -> "en"
- source (str)              -> "synthetic"
- intent_gold (str)
- generator_id (str)
- template_id (str)
- seed (int)                -> per-row deterministic seed
- split (str)               -> "train_gen" or "test_gen"
- length_chars (int)
- has_question_mark (bool)
- applied_perturbations (str) -> semicolon-separated names (exactly 3)
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from perturbations import apply_perturbations
from slot_pools import SLOTS
from templates import GENERATORS, INTENTS, TEMPLATES, stable_template_id

# ---------------- Config (locked) ----------------

LANG = "en"
SOURCE = "synthetic"

N_PER_GENERATOR = 60  # 8 * 5 * 60 = 2400
N_PERTURB = 3

TRAIN_GENS = {"direct", "polite", "contextual"}
TEST_GENS = {"constraint_heavy", "noisy"}

SLOT_PATTERN = re.compile(r"\{([a-zA-Z0-9_]+)\}")


# ---------------- Helpers ----------------

def ensure_output_dirs() -> Dict[str, Path]:
    """
    Ensure output directories exist.

    Returns:
        dict with keys 'raw' and 'processed' pointing to paths.
    """
    root = Path(__file__).resolve().parents[1]  # project root
    raw_dir = root / "data" / "raw"
    proc_dir = root / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    return {"raw": raw_dir, "processed": proc_dir}


def extract_slots(template: str) -> List[str]:
    """Extract slot names from a template string."""
    return SLOT_PATTERN.findall(template)


def fill_template(template: str, rng: random.Random) -> str:
    """
    Fill a template by sampling values for the slots it contains.

    Args:
        template: template string containing {slots}
        rng: random.Random instance

    Returns:
        Slot-filled text.

    Raises:
        KeyError: if a required slot is missing from slot pools.
    """
    slots = extract_slots(template)
    values: Dict[str, str] = {}

    for s in slots:
        if s not in SLOTS:
            raise KeyError(f"Slot '{s}' not found in SLOTS. Template: {template}")
        values[s] = rng.choice(SLOTS[s])

    return template.format(**values)


def generator_split(generator_id: str) -> str:
    """Map generator_id to split label."""
    if generator_id in TRAIN_GENS:
        return "train_gen"
    if generator_id in TEST_GENS:
        return "test_gen"
    raise ValueError(f"Unknown generator_id for split: {generator_id}")


def make_message_id(intent: str, generator_id: str, idx: int) -> str:
    """
    Create a stable-ish message id.
    Determinism is ensured by idx ordering and fixed loops.
    """
    return f"msg_{intent}_{generator_id}_{idx:04d}"


# ---------------- Main generation ----------------

def generate(seed: int = 1337) -> pd.DataFrame:
    """
    Generate the full synthetic dataset.

    Args:
        seed: base seed for reproducibility.

    Returns:
        pandas DataFrame with 2,400 rows.
    """
    rows = []
    global_rng = random.Random(seed)

    for intent in INTENTS:
        if intent not in TEMPLATES:
            raise KeyError(f"Missing intent in TEMPLATES: {intent}")

        for gen in GENERATORS:
            if gen not in TEMPLATES[intent]:
                raise KeyError(f"Missing generator '{gen}' under intent '{intent}'")

            templates = TEMPLATES[intent][gen]
            if not templates:
                raise ValueError(f"No templates for intent='{intent}', gen='{gen}'")

            # For now, we assume 1 template per family as designed.
            # If you add more templates later, sample among them here.
            template = templates[0]
            template_id = stable_template_id(intent, gen, 0, template)

            for j in range(N_PER_GENERATOR):
                # Per-row deterministic seed derived from base seed + counters
                row_seed = global_rng.randint(0, 2**31 - 1)
                rng = random.Random(row_seed)

                filled = fill_template(template, rng)
                perturbed, applied = apply_perturbations(filled, rng, n=N_PERTURB)

                msg_id = make_message_id(intent, gen, j)

                rows.append({
                    "message_id": msg_id,
                    "text": perturbed,
                    "lang": LANG,
                    "source": SOURCE,
                    "intent_gold": intent,
                    "generator_id": gen,
                    "template_id": template_id,
                    "seed": row_seed,
                    "split": generator_split(gen),
                    "length_chars": len(perturbed),
                    "has_question_mark": "?" in perturbed,
                    "applied_perturbations": ";".join(applied),
                })

    df = pd.DataFrame(rows)

    # Fail-fast sanity checks
    expected = len(INTENTS) * len(GENERATORS) * N_PER_GENERATOR
    if len(df) != expected:
        raise AssertionError(f"Row count mismatch: got {len(df)} expected {expected}")

    # Check balance per (intent, generator)
    grp = df.groupby(["intent_gold", "generator_id"]).size()
    if (grp != N_PER_GENERATOR).any():
        bad = grp[grp != N_PER_GENERATOR]
        raise AssertionError(f"Unbalanced groups:\n{bad}")

    # Check perturbations count is exactly N_PERTURB per row
    counts = df["applied_perturbations"].apply(lambda s: 0 if not s else len(s.split(";")))
    if not (counts == N_PERTURB).all():
        bad_rows = df.loc[counts != N_PERTURB, ["message_id", "applied_perturbations"]].head(10)
        raise AssertionError(f"Perturbation count mismatch in some rows. Examples:\n{bad_rows}")

    return df


def save(df: pd.DataFrame) -> None:
    """
    Save dataset to CSV and Parquet (if pyarrow available).
    """
    dirs = ensure_output_dirs()
    raw_dir = dirs["raw"]

    csv_path = raw_dir / "messages.csv"
    df.to_csv(csv_path, index=False)

    parquet_path = raw_dir / "messages.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as e:
        # Parquet may fail if pyarrow/fastparquet isn't installed.
        # CSV is still sufficient for the micro-study.
        print(f"[WARN] Could not write parquet: {e}")

    print(f"[OK] Wrote: {csv_path}")
    print(f"[OK] Wrote: {parquet_path} (if supported)")


if __name__ == "__main__":
    df = generate(seed=1337)
    save(df)
    print(df.head(5).to_string(index=False))
    print("\n[OK] Dataset generation complete.")
