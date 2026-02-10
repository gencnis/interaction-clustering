"""
perturbations.py

Controlled perturbations for synthetic short-message generation.

Perturbations simulate realistic user noise and variability (typos, punctuation, constraints,
politeness markers, minor hedging, short context prefixes). They are applied AFTER slot filling.

Design goals:
1) Realism: short user messages are noisy and stylistically diverse in real logs.
2) Controlled randomness: perturbations are sampled with a fixed seed so experiments are reproducible.
3) Category-agnostic: perturbations MUST NOT be tied to a specific intent, otherwise we introduce
   template leakage (the model clusters perturbation artifacts instead of semantic intent).
4) Reportability: we can log which perturbations were applied per message.

Usage:
- Call `apply_perturbations(text, rng, n=3)` to apply exactly n perturbations.
- Store returned `applied` list for analysis.

Important:
- Keep this module stable once experiments start; changing perturbations changes the dataset distribution.
"""

from __future__ import annotations

import random
import re
from typing import Callable, List, Tuple

# ---------- Small utilities ----------

_WORD_BOUNDARY = re.compile(r"\b")


def _lowercase_first_char(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:]


def _add_extra_question_mark(text: str) -> str:
    # If already ends with punctuation, add an extra '?' occasionally.
    if text.endswith("?"):
        return text + "?"
    if text.endswith(".") or text.endswith("!"):
        return text[:-1] + "?"
    return text + "?"


def _add_ellipsis(text: str) -> str:
    if text.endswith(("...", "…")):
        return text
    return text + "..."


def _strip_end_punct(text: str) -> str:
    return text.rstrip("?.!…")


def _inject_politeness(text: str, rng: random.Random) -> str:
    # Add a politeness marker either at the start or end.
    markers_start = ["Please", "Could you", "Can you", "Hey,"]
    markers_end = ["thanks", "thank you", "pls"]
    if rng.random() < 0.6:
        m = rng.choice(markers_start)
        # Avoid double "Please Please"
        if text.lower().startswith(m.lower()):
            return text
        return f"{m} {text}"
    else:
        m = rng.choice(markers_end)
        if text.lower().endswith(m):
            return text
        return f"{text} {m}"


def _inject_context_prefix(text: str, rng: random.Random) -> str:
    prefixes = ["For a class,", "At work,", "In my project,", "Quick question,", "Context:"]
    p = rng.choice(prefixes)
    # Avoid duplicating prefix if already present.
    if text.startswith(tuple(prefixes)):
        return text
    return f"{p} {text}"


def _inject_hedge(text: str, rng: random.Random) -> str:
    hedges = ["I think", "maybe", "I'm not sure but", "not sure if this is right, but"]
    h = rng.choice(hedges)
    # Insert hedge at beginning unless the text already begins with something similar.
    if text.lower().startswith(("i think", "maybe", "im not sure", "not sure")):
        return text
    return f"{h} {text}"


def _synonym_swap(text: str, rng: random.Random) -> str:
    """
    Very small, safe synonym swaps.
    We avoid large paraphrases to keep control.
    """
    swaps = [
        (r"\bexplain\b", "describe"),
        (r"\bfix\b", "resolve"),
        (r"\bplan\b", "schedule"),
        (r"\bcompute\b", "calculate"),
        (r"\bbriefly\b", "quickly"),
    ]
    pat, repl = rng.choice(swaps)
    return re.sub(pat, repl, text, flags=re.IGNORECASE)


def _minor_typo(text: str, rng: random.Random) -> str:
    """
    Introduce minor, common chat typos.
    Keep it mild: do NOT destroy readability.
    """
    replacements = [
        (r"\bwhat's\b", "whats"),
        (r"\bplease\b", "pls"),
        (r"\bcan't\b", "cant"),
        (r"\bthanks\b", "thx"),
        (r"\bI don't know\b", "idk"),
    ]
    pat, repl = rng.choice(replacements)
    return re.sub(pat, repl, text, flags=re.IGNORECASE)


def _inject_constraint_phrase(text: str, rng: random.Random) -> str:
    """
    Add a small constraint phrase that is broadly applicable.
    Must remain intent-agnostic.
    """
    phrases = [
        "briefly",
        "step by step",
        "no code",
        "with an example",
        "in 3 bullet points",
        "final answer only",
        "keep it short",
    ]
    ph = rng.choice(phrases)

    # Prefer appending as a trailing clause unless the sentence already has "Keep it ..."
    lower = text.lower()
    if "keep it" in lower or "final answer" in lower:
        return text

    # If ends with '?', insert before it.
    if text.endswith("?"):
        return text[:-1] + f", {ph}?"
    # Otherwise append.
    return text + f" ({ph})"


# ---------- Registry ----------

PerturbFn = Callable[[str], str]
PerturbFnRNG = Callable[[str, random.Random], str]

# Each entry: (name, function, uses_rng, weight)
# Weights define sampling probability.
PERTURBATIONS: List[Tuple[str, object, bool, float]] = [
    ("lowercase_first_char", _lowercase_first_char, False, 1.0),
    ("extra_question_mark", _add_extra_question_mark, False, 0.9),
    ("ellipsis", _add_ellipsis, False, 0.6),
    ("strip_end_punct", _strip_end_punct, False, 0.4),
    ("inject_politeness", _inject_politeness, True, 0.9),
    ("inject_context_prefix", _inject_context_prefix, True, 0.8),
    ("inject_hedge", _inject_hedge, True, 0.7),
    ("synonym_swap", _synonym_swap, True, 0.9),
    ("minor_typo", _minor_typo, True, 0.7),
    ("inject_constraint_phrase", _inject_constraint_phrase, True, 0.9),
]


def sample_perturbations(rng: random.Random, n: int = 3) -> List[Tuple[str, object, bool]]:
    """
    Sample exactly n unique perturbations according to weights.

    Args:
        rng: random.Random instance for reproducibility.
        n: number of perturbations to sample (default: 3).

    Returns:
        List of (name, fn, uses_rng) tuples.
    """
    if n <= 0:
        return []

    names = [p[0] for p in PERTURBATIONS]
    weights = [p[3] for p in PERTURBATIONS]

    chosen = []
    chosen_names = set()

    # Weighted sampling without replacement.
    # Simple loop is fine because the list is small.
    while len(chosen) < n:
        pick = rng.choices(PERTURBATIONS, weights=weights, k=1)[0]
        if pick[0] in chosen_names:
            continue
        chosen.append((pick[0], pick[1], pick[2]))
        chosen_names.add(pick[0])

        # Safety: avoid infinite loops if n > available
        if len(chosen_names) == len(names):
            break

    return chosen


def apply_perturbations(text: str, rng: random.Random, n: int = 3) -> Tuple[str, List[str]]:
    """
    Apply exactly n sampled perturbations to text.

    Args:
        text: input string (already slot-filled).
        rng: random.Random instance.
        n: number of perturbations to apply (default: 3).

    Returns:
        (new_text, applied_names)
    """
    applied = []
    out = text

    for name, fn, uses_rng in sample_perturbations(rng, n=n):
        before = out
        if uses_rng:
            out = fn(out, rng)  # type: ignore[misc]
        else:
            out = fn(out)       # type: ignore[misc]

        # Always log sampled perturbations to enforce "exactly n" reporting.
        applied.append(name)

    return out, applied
