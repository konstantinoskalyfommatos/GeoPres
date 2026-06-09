"""
Multilingual Retention Analysis for Dimensionality-Reduced Embedding Models.

Computes retention ratios (reduced score / backbone score) per language across
all reduced models and reports the retention gap (non-English minus English)
to assess whether English-only training biases the reduction.

Descriptive analysis only --- no statistical tests.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.config import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(PROJECT_ROOT) / "storage" / "evaluation_results"
TASK = "AmazonReviewsClassification.json"

# ---------------------------------------------------------------------------
# Language mapping (hf_subset -> readable name)
# ---------------------------------------------------------------------------
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
    "zh": "Chinese",
}

# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def _find_json(parent: Path, filename: str) -> Path | None:
    """Recursively find *filename* under *parent* and return the first match."""
    matches = list(parent.rglob(filename))
    if not matches:
        return None
    return matches[0]


def load_scores(path: Path) -> dict[str, float]:
    """Load an AmazonReviewsClassification JSON and return {hf_subset: main_score}."""
    with open(path) as f:
        data: dict[str, Any] = json.load(f)
    scores: dict[str, float] = {}
    for entry in data["scores"]["test"]:
        subset = entry["hf_subset"]
        scores[subset] = entry["main_score"]
    return scores


# ---------------------------------------------------------------------------
# Backbone definitions
# ---------------------------------------------------------------------------

BACKBONES = {
    "Alibaba-NLP__gte-multilingual-base": {
        "backbone_dir": BASE
        / "backbone"
        / "Alibaba-NLP__gte-multilingual-base"
        / "results"
        / "Alibaba-NLP__gte-multilingual-base"
        / "9bbca17d9273fd0d03d5725c7a4b0f6b45142062",
        "reduced_base": BASE
        / "trained_models"
        / "Alibaba-NLP__gte-multilingual-base"
        / "results",
        "dims": [2, 32, 64, 128, 256],
    },
    "Qwen__Qwen3-Embedding-0.6B": {
        "backbone_dir": BASE
        / "backbone"
        / "Qwen__Qwen3-Embedding-0.6B"
        / "results"
        / "Qwen__Qwen3-Embedding-0.6B"
        / "c54f2e6e80b2d7b7de06f51cec4959f6b3e03418",
        "reduced_base": BASE
        / "trained_models"
        / "Qwen__Qwen3-Embedding-0.6B"
        / "results",
        "dims": [2, 32, 64, 128, 256, 512],
    },
}


def _model_dir_name(backbone_key: str, dim: int) -> str:
    """Return the reduced model directory name for a given backbone and dim."""
    return f"{backbone_key}_reduced_{dim}_batch_20000_poslossfactor_1"


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_data() -> pd.DataFrame:
    """
    Build a DataFrame with one row per (backbone, dim, language).

    Columns:
        backbone, dim, language, backbone_score, reduced_score, retention_ratio
    """
    rows: list[dict[str, Any]] = []

    for backbone_key, info in BACKBONES.items():
        # Load backbone scores
        backbone_json = info["backbone_dir"] / TASK
        if not backbone_json.exists():
            print(f"[WARN] Backbone JSON not found: {backbone_json}", file=sys.stderr)
            continue
        backbone_scores = load_scores(backbone_json)

        for dim in info["dims"]:
            model_dir = info["reduced_base"] / _model_dir_name(backbone_key, dim)
            reduced_json = _find_json(model_dir, TASK)
            if reduced_json is None:
                print(f"[WARN] Reduced JSON not found for {backbone_key} dim={dim}", file=sys.stderr)
                continue

            reduced_scores = load_scores(reduced_json)

            for lang in backbone_scores:
                if lang not in reduced_scores:
                    continue
                backbone_score = backbone_scores[lang]
                reduced_score = reduced_scores[lang]
                if backbone_score == 0:
                    continue
                retention = reduced_score / backbone_score
                rows.append(
                    {
                        "backbone": backbone_key,
                        "dim": dim,
                        "language": lang,
                        "backbone_score": backbone_score,
                        "reduced_score": reduced_score,
                        "retention_ratio": retention,
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_retention(df: pd.DataFrame) -> None:
    """Print retention ratio and gap tables (descriptive, no statistical tests)."""

    # --- 1. Retention ratio table per backbone / dim / language ---------------
    print("=" * 80)
    print("RETENTION RATIOS (reduced_score / backbone_score)")
    print("=" * 80)

    for backbone_key in sorted(df["backbone"].unique()):
        print(f"\n--- {backbone_key} ---")
        sub = df[df["backbone"] == backbone_key]
        pivot = sub.pivot_table(
            index="dim", columns="language", values="retention_ratio", aggfunc="mean"
        )
        # Reorder columns: en first, then alphabetical
        cols = ["en"] + sorted(c for c in pivot.columns if c != "en")
        pivot = pivot[cols]
        pivot = pivot.sort_index()
        print(pivot.round(4).to_string())

    # --- 2. English vs. non-English gap ---------------------------------------
    print("\n" + "=" * 80)
    print("RETENTION GAP (non-English minus English)")
    print("(positive => non-English retains better than English)")
    print("=" * 80)

    gap_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        if row["language"] == "en":
            continue
        en_row = df[
            (df["backbone"] == row["backbone"])
            & (df["dim"] == row["dim"])
            & (df["language"] == "en")
        ]
        if en_row.empty:
            continue
        gap = row["retention_ratio"] - en_row.iloc[0]["retention_ratio"]
        gap_rows.append(
            {
                "backbone": row["backbone"],
                "dim": row["dim"],
                "language": row["language"],
                "gap": gap,
            }
        )
    gap_df = pd.DataFrame(gap_rows)

    for backbone_key in sorted(gap_df["backbone"].unique()):
        print(f"\n--- {backbone_key} ---")
        bsub = gap_df[gap_df["backbone"] == backbone_key]
        pivot = bsub.pivot_table(
            index="dim", columns="language", values="gap", aggfunc="mean"
        )
        pivot = pivot.sort_index()
        print(pivot.round(4).to_string())

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = collect_data()
    print(f"Collected {len(df)} data points across {df['backbone'].nunique()} backbones, "
          f"{df['dim'].nunique()} dimensionalities, {df['language'].nunique()} languages.\n")

    analyse_retention(df)

    output_dir = Path(PROJECT_ROOT) / "storage" / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw data for downstream use
    csv_path = output_dir / "retention_ratios.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[data] retention_ratios.csv  →  {csv_path}")


if __name__ == "__main__":
    main()
