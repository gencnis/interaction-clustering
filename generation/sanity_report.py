"""
sanity_report.py

Sanity report for the synthetic dataset.

Checks:
- Row/column sanity
- Balance per (intent_gold, generator_id)
- Split distribution
- Length statistics
- Question mark rate
- Perturbation frequency counts
- A few sample messages per (intent, generator)
- Quick keyword overuse alarms (rough leakage signal)

Run:
  python3 -u generation/sanity_report.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_csv() -> pd.DataFrame:
    csv_path = project_root() / "data" / "raw" / "messages.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset: {csv_path}")
    return pd.read_csv(csv_path)


def main() -> None:
    df = load_csv()

    print("\n=== BASIC SHAPE ===")
    print("Rows:", len(df))
    print("Cols:", list(df.columns))

    print("\n=== BALANCE CHECKS ===")
    g = df.groupby(["intent_gold", "generator_id"]).size().reset_index(name="n")
    print("Min group size:", int(g["n"].min()), "Max group size:", int(g["n"].max()))
    bad = g[g["n"] != g["n"].iloc[0]]
    if len(bad) > 0:
        print("\n[WARN] Unbalanced groups:\n", bad.to_string(index=False))
    else:
        print("[OK] Groups look balanced.")

    print("\n=== SPLIT DISTRIBUTION ===")
    print(df["split"].value_counts().to_string())

    print("\n=== INTENT DISTRIBUTION ===")
    print(df["intent_gold"].value_counts().to_string())

    print("\n=== GENERATOR DISTRIBUTION ===")
    print(df["generator_id"].value_counts().to_string())

    print("\n=== LENGTH STATS (chars) ===")
    print(df["length_chars"].describe().to_string())

    print("\n=== QUESTION MARK RATE ===")
    print("has_question_mark=True:", float(df["has_question_mark"].mean()))

    print("\n=== PERTURBATION FREQUENCIES ===")
    all_p = []
    for s in df["applied_perturbations"].fillna(""):
        s = str(s).strip()
        if s:
            all_p.extend(s.split(";"))
    c = Counter(all_p)

    print("Total perturbation tokens logged:", sum(c.values()))
    print("Unique perturbations:", len(c))

    print("\nTop 10:")
    for k, v in c.most_common(10):
        print(f"  {k:24s}  {v}")

    print("\nBottom 10:")
    for k, v in sorted(c.items(), key=lambda x: x[1])[:10]:
        print(f"  {k:24s}  {v}")

    print("\n=== SAMPLE MESSAGES PER (intent, generator) ===")
    for (intent, gen), sub in df.groupby(["intent_gold", "generator_id"], sort=True):
        print(f"\n--- {intent} / {gen} ---")
        for t in sub["text"].head(2).tolist():
            print(" •", t)

    print("\n=== QUICK LEAKAGE ALARMS (keyword overuse) ===")
    alarms = {
        "summarization": ["summarize", "translate", "rewrite", "condense"],
        "how_to": ["how do i", "how to", "show me how"],
        "troubleshooting": ["error", "fails", "failing", "debug"],
        "math": ["compute", "calculate", "sqrt", "cosine similarity", "log2"],
        "planning": ["plan", "schedule", "checklist", "milestones"],
        "recommendation": ["recommend", "which is better", "pick for me", "should i use"],
    }

    def cue_rates(intent: str, cues: list[str]) -> None:
        sub = df[df["intent_gold"] == intent]
        txt = sub["text"].astype(str).str.lower()
        print(f"\n[{intent}] cue hit rates:")
        for cue in cues:
            r = float(txt.str.contains(cue, regex=False).mean())
            print(f"  {cue:16s}: {r:.3f}")

    for intent, cues in alarms.items():
        cue_rates(intent, cues)

    print("\n[OK] Sanity report complete.")


if __name__ == "__main__":
    main()
