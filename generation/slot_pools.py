"""
slot_pools.py

Shared slot-value pools for synthetic short-message generation.

Slot pools define the controlled vocabulary used to fill templates.
Design goals:
1) Shared vocab across intents to enforce realistic ambiguity.
2) Controlled size to prevent accidental leakage and to keep generation reproducible.
3) Slots align with placeholders used in templates.py (e.g., {concept}, {task}, {tool}, ...).

Usage:
- Sample values for each slot referenced by a chosen template.
- Some slots are optional depending on the template; generator code should only
  sample the slots actually present in the template string.

Notes:
- Keep pools small and stable once experiments start; changing pools affects results.
- Avoid adding intent-unique keywords too aggressively (template leakage risk).
"""

from __future__ import annotations

from typing import Dict, List

SLOTS: Dict[str, List[str]] = {
    # Concepts/topics appear across multiple intents to intentionally create overlap.
    "concept": [
        "cosine similarity",
        "sentence embeddings",
        "overfitting",
        "cross-validation",
        "PCA",
        "transformers",
        "clustering",
        "gradient descent",
        "tokenization",
        "regularization",
    ],
    "phenomenon": [
        "overfitting",
        "vanishing gradients",
        "mode collapse",
        "data leakage",
        "covariate shift",
    ],

    # Tools/environments: shared across how-to, troubleshooting, planning, etc.
    "tool": [
        "Python",
        "PyTorch",
        "scikit-learn",
        "Docker",
        "Git",
        "Linux",
        "VS Code",
        "Jupyter",
    ],

    # Procedural tasks: used in how-to and sometimes planning.
    "task": [
        "create a virtual environment",
        "compute sentence embeddings",
        "save a dataframe to parquet",
        "run KMeans on embeddings",
        "normalize vectors",
        "load a CSV file",
        "tokenize a text dataset",
        "reduce dimensionality with PCA",
    ],

    # Troubleshooting-specific but still generic enough.
    "error": [
        "CUDA out of memory",
        "ModuleNotFoundError",
        "Permission denied",
        "container exits immediately",
        "segmentation fault",
        "invalid device ordinal",
        "connection refused",
        "SSL certificate verify failed",
    ],
    "action": [
        "train a model",
        "build a Docker image",
        "run pip install",
        "start a notebook",
        "load my dataset",
        "run my script",
        "connect to a server",
    ],

    # Summarization / rewrite / translation.
    "tone": [
        "formal",
        "friendly",
        "academic",
        "concise",
    ],
    "lang": [
        "English",
        "Spanish",
        "German",
        "French",
    ],

    # Recommendation choices.
    "a": [
        "KMeans",
        "HDBSCAN",
        "Agglomerative clustering",
        "UMAP",
        "PCA",
    ],
    "b": [
        "HDBSCAN",
        "KMeans",
        "Agglomerative clustering",
        "PCA",
        "UMAP",
    ],
    "option": [
        "KMeans",
        "HDBSCAN",
        "PCA",
        "UMAP",
    ],
    "goal": [
        "clustering short text embeddings",
        "reducing dimensionality before clustering",
        "finding interaction patterns in user queries",
        "grouping similar user requests",
    ],

    # Planning.
    "time_horizon": [
        "today",
        "this week",
        "next 2 weeks",
        "by Friday",
    ],
    "time_budget": [
        "30 minutes",
        "1 hour",
        "2 hours",
    ],

    # Creative generation.
    "artifact": [
        "tagline",
        "short poem",
        "micro-story",
        "product name",
    ],
    "artifact_plural": [
        "taglines",
        "poems",
        "short stories",
        "product names",
    ],
    "style": [
        "funny",
        "serious",
        "minimalist",
        "dramatic",
    ],
    "topic": [
        "an AI study assistant",
        "clustering user messages",
        "loneliness in winter",
        "a productivity app",
        "learning faster",
        "debugging late at night",
    ],

    # Math / estimation.
    "expr": [
        "17 * 23",
        "1024 / 8",
        "sqrt(144)",
        "cosine similarity between (1,2) and (2,1)",
        "log2(1024)",
    ],
    "quantity": [
        "the number of errors",
        "the cosine similarity",
        "the mean and standard deviation",
        "the estimated runtime",
    ],
    "given": [
        "accuracy is 0.82 on 500 samples",
        "vectors are (1,2) and (2,1)",
        "I have 50k texts and 768-d embeddings",
        "k is 20 clusters",
    ],
    "runtime_cost": [
        "runtime",
        "memory usage",
        "compute cost",
    ],
    "setup": [
        "embedding 50k texts into 768-d vectors",
        "running KMeans with k=20 on 50k vectors",
        "computing pairwise distances for 10k texts",
    ],

    # Constraints appear across intents to increase overlap and realism.
    "constraint": [
        "one example",
        "3 bullet points",
        "no equations",
        "step by step",
        "no code",
        "with code",
        "final answer only",
        "keep it short",
    ],

    # Small integers for templates that need {k}.
    "k": [
        "2",
        "3",
        "5",
    ],

        "text_stub": [
        "I need to send an update to my team about the project status.",
        "The experiment results look inconsistent across different random seeds.",
        "We collected 2,400 short user messages and want to cluster them using embeddings.",
        "My laptop fan gets loud when I run Docker containers for too long.",
        "Please review the following paragraph for clarity and grammar.",
        "I tried to install the package but the build step failed unexpectedly.",
        "The meeting agenda includes milestones, risks, and next steps for the sprint.",
        "I am comparing KMeans and HDBSCAN for clustering sentence embeddings.",
        "This report needs to be shorter, more direct, and easier to scan quickly.",
        "I want to translate a short note to German for a colleague.",
        "The code runs locally but fails in the CI pipeline with a timeout.",
        "The user asked for a simple explanation without any equations.",
        "We need a checklist for running experiments reproducibly in two weeks.",
        "The model accuracy improved, but the validation loss is still unstable.",
        "I wrote a short message, but it sounds too informal for an email.",
        "The dataset contains short queries, commands, and questions from users.",
        "I want a brief summary of the key findings and the main limitation.",
        "The results section should include both quantitative metrics and examples.",
        "I am not sure whether to include more context in the user messages.",
        "The instructions say to keep it concise and avoid unnecessary details.",
    ],

}
