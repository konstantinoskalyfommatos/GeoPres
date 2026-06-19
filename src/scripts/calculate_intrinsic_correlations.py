import pandas as pd
from scipy.stats import spearmanr
import os
import json

from config import EVALUATION_RESULTS_PATH


def _get_extrinsic_model_names() -> set:
    """Return the set of model names that appear in the extrinsic tables script."""
    BACKBONES = [
        "Alibaba-NLP__gte-multilingual-base",
        "jinaai__jina-embeddings-v2-small-en",
        "Qwen__Qwen3-Embedding-0.6B",
        "sentence-transformers__all-mpnet-base-v2",
    ]
    DIM_GROUPS = [256, 128, 64, 32, 2]
    METHOD_SUFFIXES = [
        "_pca",
        "_random_projection",
        "_random_selection",
        "_truncation",
        "_autoencoder",
        "_batch_20000_poslossfactor_1",
    ]

    names = set()
    for backbone in BACKBONES:
        for dim in DIM_GROUPS:
            for suffix in METHOD_SUFFIXES:
                names.add(f"{backbone}_reduced_{dim}{suffix}")
    return names


def calculate_intrinsic_correlations(df: pd.DataFrame) -> dict:
    extrinsic_names = _get_extrinsic_model_names()
    df = df[df["Model"].isin(extrinsic_names)]

    # Keep only the mteb and intrinsic metric columns, in order to calculate the correlations
    df_mteb_angular_loss = df[["**AVG_MTEB**", "angular_loss"]].dropna()

    df_mteb_positional_loss = df[["**AVG_MTEB**", "positional_loss"]].dropna()

    df_mteb_spearman_loss = df[["**AVG_MTEB**", "spearman_loss"]].dropna()

    # Calculate Spearman correlation for angular_loss
    spearman_angular_score = spearmanr(
        df_mteb_angular_loss["**AVG_MTEB**"].values, 
        df_mteb_angular_loss["angular_loss"].values
    )

    # Calculate Spearman correlation for positional_loss
    spearman_positional_score = spearmanr(
        df_mteb_positional_loss["**AVG_MTEB**"].values, 
        df_mteb_positional_loss["positional_loss"].values
    )

    # Calculate Spearman correlation for spearman_loss
    spearman_spearman_score = spearmanr(
        df_mteb_spearman_loss["**AVG_MTEB**"].values, 
        df_mteb_spearman_loss["spearman_loss"].values
    )

    return {
        "angular_loss": {
            "spearman": spearman_angular_score.statistic,
            "pvalue": spearman_angular_score.pvalue,
        },
        "positional_loss": {
            "spearman": spearman_positional_score.statistic,
            "pvalue": spearman_positional_score.pvalue,
        },
        "spearman_loss": {
            "spearman": spearman_spearman_score.statistic,
            "pvalue": spearman_spearman_score.pvalue,
        },
    }


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(EVALUATION_RESULTS_PATH, "comparison_results.csv"))
    correlations = calculate_intrinsic_correlations(df)


    OUTPUT_PATH = os.path.join(EVALUATION_RESULTS_PATH, "intrinsic_correlations.json")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(correlations, f, indent=4)

    print(f"Intrinsic extrinsic correlations saved at: {OUTPUT_PATH}")
        