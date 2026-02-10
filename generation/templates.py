"""
templates.py

Template families for synthetic short-message generation.

This module defines a controlled set of text templates organized by:
- intent (interaction-level user goal)
- generator family (stylistic template family)

Key design goals:
1) Generator-aware evaluation support:
   Each intent has multiple generator families (direct/polite/contextual/constraint-heavy/noisy)
   so we can hold out entire families and test robustness against template leakage.

2) Avoid trivial separability:
   Templates are written so that intent is not recoverable from a single keyword alone
   (e.g., summarization does not always contain "summarize").

3) Slot-based variability:
   Templates include slots (e.g., {concept}, {task}) filled from shared slot pools
   to create realistic ambiguity across intents.

Expected downstream usage:
- Select an intent
- Select a generator family
- Sample a template
- Fill slots from slot pools
- Apply perturbations (typos, punctuation noise, constraint injection, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Dict, List, Tuple

# Locked taxonomy (do not rename once experiments start; affects reproducibility).
INTENTS: Tuple[str, ...] = (
    "information_seeking",
    "how_to",
    "troubleshooting",
    "summarization",
    "recommendation",
    "planning",
    "creative",
    "math",
)

GENERATORS: Tuple[str, ...] = (
    "direct",
    "polite",
    "contextual",
    "constraint_heavy",
    "noisy",
)


@dataclass(frozen=True)
class TemplateRef:
    """
    A lightweight reference to a specific template string.

    Attributes:
        intent: One of INTENTS.
        generator: One of GENERATORS.
        index: Index in the templates list for (intent, generator).
        text: The template string itself.
        template_id: Stable ID derived from (intent, generator, index, text).
    """
    intent: str
    generator: str
    index: int
    text: str
    template_id: str


def stable_template_id(intent: str, generator: str, index: int, text: str) -> str:
    """
    Create a stable, human-safe template ID.

    We hash the defining fields to ensure IDs are stable across runs
    as long as template text and ordering do not change.

    Args:
        intent: intent label.
        generator: generator family label.
        index: template index within that family.
        text: template string.

    Returns:
        A short stable ID string, e.g., "tpl_3f8a1c2d".
    """
    payload = f"{intent}||{generator}||{index}||{text}".encode("utf-8")
    return "tpl_" + sha1(payload).hexdigest()[:8]


# 8 intents × 5 generator families × (1 template each by default)
# You may add more than 1 template per family later, but do not remove or reorder
# once you start reporting results.
TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "information_seeking": {
        "direct": [
            "What is {concept}?",
        ],
        "polite": [
            "Could you explain {concept} in simple terms?",
        ],
        "contextual": [
            "In machine learning, why does {phenomenon} happen?",
        ],
        "constraint_heavy": [
            "Explain {concept} briefly, with {constraint}.",
        ],
        "noisy": [
            "whats {concept} and why it matters??",
        ],
    },

    "how_to": {
        "direct": [
            "How do I {task}?",
        ],
        "polite": [
            "Please show me how to {task}.",
        ],
        "contextual": [
            "I'm new to {tool}; how can I {task}?",
        ],
        "constraint_heavy": [
            "How do I {task}? Keep it {constraint}.",
        ],
        "noisy": [
            "how to {task} on {tool}??",
        ],
    },

    "troubleshooting": {
        "direct": [
            "Why am I getting {error}?",
        ],
        "polite": [
            "Can you help me fix this error: {error}",
        ],
        "contextual": [
            "When I {action}, I get {error}. What should I check?",
        ],
        "constraint_heavy": [
            "Debug this: {error}. Assume {constraint}.",
        ],
        "noisy": [
            "it keeps failing: {error} idk why 😭",
        ],
    },

    "summarization": {
        "direct": [
            "Summarize this: {text_stub}",
        ],
        "polite": [
            "Please rewrite this to sound {tone}: {text_stub}",
        ],
        "contextual": [
            "Translate this to {lang}: {text_stub}",
        ],
        "constraint_heavy": [
            "Condense this to {constraint}: {text_stub}",
        ],
        "noisy": [
            "make this nicer/shorter pls: {text_stub}",
        ],
    },

    "recommendation": {
        "direct": [
            "Which is better: {a} or {b}?",
        ],
        "polite": [
            "What would you recommend for {goal}?",
        ],
        "contextual": [
            "Given {constraint}, should I use {option}?",
        ],
        "constraint_heavy": [
            "Recommend {k} options for {goal}, {constraint}.",
        ],
        "noisy": [
            "pick for me: {a} vs {b}",
        ],
    },

    "planning": {
        "direct": [
            "Make a plan for {goal}.",
        ],
        "polite": [
            "Can you schedule {goal} over {time_horizon}?",
        ],
        "contextual": [
            "I have {time_budget} per day. Plan {goal}.",
        ],
        "constraint_heavy": [
            "Plan {goal} with {constraint}.",
        ],
        "noisy": [
            "need a quick plan for {goal} by {time_horizon}!!",
        ],
    },

    "creative": {
        "direct": [
            "Write a {artifact} about {topic}.",
        ],
        "polite": [
            "Could you generate {k} {artifact_plural} for {topic}?",
        ],
        "contextual": [
            "Create a {style} {artifact} for {topic}.",
        ],
        "constraint_heavy": [
            "Generate {artifact} with {constraint} about {topic}.",
        ],
        "noisy": [
            "gimme a {artifact} thats {style}",
        ],
    },

    "math": {
        "direct": [
            "Compute {expr}.",
        ],
        "polite": [
            "Can you calculate {quantity} if {given}?",
        ],
        "contextual": [
            "Estimate {runtime_cost} for {setup}.",
        ],
        "constraint_heavy": [
            "Calculate {quantity}. Show {constraint}.",
        ],
        "noisy": [
            "quick math: {expr}??",
        ],
    },
}


def iter_templates() -> List[TemplateRef]:
    """
    Flatten the TEMPLATES structure into a list of TemplateRef objects.

    Returns:
        A list of TemplateRef, each containing intent, generator, index, template text, and template_id.
    """
    out: List[TemplateRef] = []
    for intent in INTENTS:
        if intent not in TEMPLATES:
            raise KeyError(f"Missing intent in TEMPLATES: {intent}")

        for gen in GENERATORS:
            if gen not in TEMPLATES[intent]:
                raise KeyError(f"Missing generator '{gen}' for intent '{intent}'")

            lst = TEMPLATES[intent][gen]
            if not lst:
                raise ValueError(f"Empty template list for intent='{intent}', generator='{gen}'")

            for i, text in enumerate(lst):
                tid = stable_template_id(intent=intent, generator=gen, index=i, text=text)
                out.append(TemplateRef(intent=intent, generator=gen, index=i, text=text, template_id=tid))

    return out
   