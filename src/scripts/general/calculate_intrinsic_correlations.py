import pandas as pd
from scipy.stats import spearmanr
import os
import json

from utils.config import EVALUATION_RESULTS_PATH


def calculate_intrinsic_correlations(df: pd.DataFrame) -> dict:
    df = df[~df["Model"].isin(
    [
        "Alibaba-NLP__gte-multilingual-base", 
        "jinaai__jina-embeddings-v2-small-en", 
        "sentence-transformers__all-mpnet-base-v2",
        "Qwen__Qwen3-Embedding-0.6B"
        
        ]
    )]

    # Keep only the mteb and intrinsic metric columns, in order to calculate the correlations
    df_mteb_angular_loss = df[["**AVG_MTEB**", "angular_loss"]].dropna()
    df_mteb_angular_loss_weighted = df[["**AVG_MTEB**", "angular_loss_weighted"]].dropna()

    df_mteb_positional_loss = df[["**AVG_MTEB**", "positional_loss"]].dropna()
    df_mteb_positional_loss_weighted = df[["**AVG_MTEB**", "positional_loss_weighted"]].dropna()

    df_mteb_spearman_loss = df[["**AVG_MTEB**", "spearman_loss"]].dropna()
    df_mteb_spearman_loss_weighted = df[["**AVG_MTEB**", "spearman_loss_weighted"]].dropna()

    # Calculate Spearman correlation for angular_loss
    spearman_angular_score = spearmanr(
        df_mteb_angular_loss["**AVG_MTEB**"].values, 
        df_mteb_angular_loss["angular_loss"].values
    )

    # Calculate Spearman correlation for angular_loss_weighted
    spearman_angular_score_weighted = spearmanr(
        df_mteb_angular_loss_weighted["**AVG_MTEB**"].values, 
        df_mteb_angular_loss_weighted["angular_loss_weighted"].values
    )

    # Calculate Spearman correlation for positional_loss
    spearman_positional_score = spearmanr(
        df_mteb_positional_loss["**AVG_MTEB**"].values, 
        df_mteb_positional_loss["positional_loss"].values
    )

    # Calculate Spearman correlation for positional_loss_weighted
    spearman_positional_score_weighted = spearmanr(
        df_mteb_positional_loss_weighted["**AVG_MTEB**"].values, 
        df_mteb_positional_loss_weighted["positional_loss_weighted"].values
    )

    # Calculate Spearman correlation for spearman_loss
    spearman_spearman_score = spearmanr(
        df_mteb_spearman_loss["**AVG_MTEB**"].values, 
        df_mteb_spearman_loss["spearman_loss"].values
    )

    # Calculate Spearman correlation for spearman_loss_weighted
    spearman_spearman_score_weighted = spearmanr(
        df_mteb_spearman_loss_weighted["**AVG_MTEB**"].values, 
        df_mteb_spearman_loss_weighted["spearman_loss_weighted"].values
    )

    return {
        "angular_loss": {
            "spearman": spearman_angular_score.statistic,
            "pvalue": spearman_angular_score.pvalue,
        },
        "angular_loss_weighted": {
            "spearman": spearman_angular_score_weighted.statistic,
            "pvalue": spearman_angular_score_weighted.pvalue,
        },
        "positional_loss": {
            "spearman": spearman_positional_score.statistic,
            "pvalue": spearman_positional_score.pvalue,
        },
        "positional_loss_weighted": {
            "spearman": spearman_positional_score_weighted.statistic,
            "pvalue": spearman_positional_score_weighted.pvalue,
        },
        "spearman_loss": {
            "spearman": spearman_spearman_score.statistic,
            "pvalue": spearman_spearman_score.pvalue,
        },
        "spearman_loss_weighted": {
            "spearman": spearman_spearman_score_weighted.statistic,
            "pvalue": spearman_spearman_score_weighted.pvalue,
        }
    }


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(EVALUATION_RESULTS_PATH, "comparison_results.csv"))
    correlations = calculate_intrinsic_correlations(df)


    OUTPUT_PATH = os.path.join(EVALUATION_RESULTS_PATH, "intrinsic_correlations.json")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(correlations, f, indent=4)

    print(f"Intrinsic extrinsic correlations saved at: {OUTPUT_PATH}")
        