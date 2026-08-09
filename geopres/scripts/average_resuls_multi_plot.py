import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os
from config import EVALUATION_RESULTS_PATH, STORAGE_PATH


def extract_embedding_dim(model_name):
    """Extract embedding dimension from model name."""
    if '_reduced_' in model_name:
        parts = model_name.split('_reduced_')
        dim_part = parts[1].split('_')[0]
        return int(dim_part)
    elif 'gte-multilingual-base' in model_name or "all-mpnet-base-v2" in model_name:
        return 768
    elif 'jina-embeddings-v2-small-en' in model_name:
        return 512
    elif 'Qwen3-Embedding-0.6B' in model_name:
        return 1024
    raise ValueError(f"Unknown model name format: {model_name}")


def extract_method(model_name):
    """Extract dimensionality reduction method from model name."""
    if '_reduced_' not in model_name:
        return 'base'
    if 'batch_20000_poslossfactor_1' in model_name and "weighted" not in model_name:
        return 'GeoPres'
    elif '_pca' in model_name:
        return 'pca'
    elif 'random_projection' in model_name:
        return 'random_projection'
    elif 'random_selection' in model_name:
        return 'random_selection'
    elif 'truncation' in model_name:
        return 'truncation'
    elif 'autoencoder' in model_name:
        return 'autoencoder'
    return None


def create_multi_plot(df, output_dir, models_non_mrl, models_mrl):
    """
    Create a 2×4 multi-figure with grouped bar charts per task.

    Rows: non-MRL (top), MRL (bottom)
    Columns: STS, Retrieval, Classification, Clustering
    """
    colors = {
        'GeoPres': '#3b82f6',
        'pca': '#d1d5db',
        'random_projection': '#9ca3af',
        'random_selection': '#6b7280',
        'truncation': '#4b5563',
        'autoencoder': '#e5e7eb',
    }

    method_labels = {
        'GeoPres': 'GeoPres (Ours)',
        'random_selection': 'Random Selection',
        'random_projection': 'Random Projection',
        'autoencoder': 'Autoencoder',
        'pca': 'PCA',
        'truncation': 'Truncation',
    }

    method_order = list(method_labels.keys())
    hatches = {m: '' for m in method_order}
    hatches['GeoPres'] = '///'

    tasks = [
        ('STS', '**AVG_STS**'),
        ('Retrieval', '**AVG_RETRIEVAL**'),
        ('Classification', '**AVG_CLASSIFICATION**'),
        ('Clustering', '**AVG_CLUSTERING**'),
    ]

    rows = [
        ('Non-MRL', models_non_mrl),
        ('MRL', models_mrl),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(28, 14), sharey=True)

    legend_handles = None

    for row_idx, (row_label, model_list) in enumerate(rows):
        for col_idx, (task_label, column_name) in enumerate(tasks):
            ax = axes[row_idx, col_idx]

            # ── Collect data per backbone ──────────────────────────────────────
            all_ratios = []
            all_scores = {m: [] for m in method_order}

            for model_base_name in model_list:
                model_key = model_base_name.replace('/', '__')
                model_data = df[df['Model'].str.contains(model_key)].copy()
                model_data['dimension'] = model_data['Model'].apply(extract_embedding_dim)
                model_data['method'] = model_data['Model'].apply(extract_method)
                model_data = model_data[model_data['dimension'] != 2]
                model_data = model_data[model_data['method'].isin(method_order)]

                base_data = model_data[model_data['method'] == 'base']
                if base_data.empty:
                    # Try original df for base
                    base_data = df[
                        df['Model'].str.contains(model_key) &
                        (df['Model'].apply(extract_method) == 'base')
                    ]
                if base_data.empty:
                    continue

                base_score = base_data[column_name].values[0]
                original_dim = extract_embedding_dim(
                    base_data['Model'].values[0]
                )

                for method in method_order:
                    method_data = model_data[model_data['method'] == method]
                    for _, row in method_data.iterrows():
                        ratio = row['dimension'] / original_dim
                        norm_score = row[column_name] / base_score * 100
                        all_ratios.append(ratio)
                        all_scores[method].append(norm_score)

            # Unique sorted ratios → x-axis positions
            unique_ratios = sorted(set(r for r in all_ratios))
            if not unique_ratios:
                ax.text(0.5, 0.5, 'No data',
                        transform=ax.transAxes, ha='center', va='center', fontsize=20)
                continue

            x_positions = np.arange(len(unique_ratios))
            bar_width = 0.12

            # Build per-method values per ratio bucket
            ratio_to_idx = {r: i for i, r in enumerate(unique_ratios)}
            method_values = {m: [0.0] * len(unique_ratios) for m in method_order}

            # Re-collect with proper bucketing
            for model_base_name in model_list:
                model_key = model_base_name.replace('/', '__')
                model_data = df[df['Model'].str.contains(model_key)].copy()
                model_data['dimension'] = model_data['Model'].apply(extract_embedding_dim)
                model_data['method'] = model_data['Model'].apply(extract_method)
                model_data = model_data[model_data['dimension'] != 2]
                model_data = model_data[model_data['method'].isin(method_order)]

                base_data = df[
                    df['Model'].str.contains(model_key) &
                    (df['Model'].apply(extract_method) == 'base')
                ]
                if base_data.empty:
                    continue
                base_score = base_data[column_name].values[0]
                original_dim = extract_embedding_dim(base_data['Model'].values[0])

                for method in method_order:
                    method_data = model_data[model_data['method'] == method]
                    for _, row in method_data.iterrows():
                        ratio = row['dimension'] / original_dim
                        norm_score = row[column_name] / base_score * 100
                        idx = None
                        for i, r in enumerate(unique_ratios):
                            if abs(ratio - r) < 0.001:
                                idx = i
                                break
                        if idx is not None:
                            method_values[method][idx] = norm_score

            # Find winners per ratio bucket
            winners = {}
            for i in range(len(unique_ratios)):
                dim_values = {m: method_values[m][i] for m in method_order}
                winners[i] = max(dim_values, key=dim_values.get)

            # ── Title ──────────────────────────────────────────────────────────
            if row_idx == 0:
                ax.set_title(task_label, fontsize=28, pad=12)

            # ── Draw bars ──────────────────────────────────────────────────────
            for i, method in enumerate(method_order):
                values = method_values[method]
                offset = (i - len(method_order) / 2 + 0.5) * bar_width

                edgecolor = 'white' if method == 'GeoPres' else None
                linewidth = 1 if method == 'GeoPres' else 0

                bars = ax.bar(
                    [x + offset for x in x_positions], values, bar_width,
                    label=method_labels[method],
                    facecolor=colors[method],
                    edgecolor=edgecolor, linewidth=linewidth,
                    hatch=hatches[method],
                    alpha=0.9 if method == 'GeoPres' else 0.8,
                )

                for dim_idx, bar in enumerate(bars):
                    if winners[dim_idx] == method:
                        bar.set_edgecolor('black')
                        bar.set_linewidth(2)

            # Baseline
            ax.axhline(y=100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

            # ── Axes & styling ────────────────────────────────────────────────
            ax.set_xlim(-0.4, len(unique_ratios) - 0.6)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                [f'{r * 100:.0f}' for r in unique_ratios],
                fontsize=18, rotation=0,
            )
            ax.set_ylim(0, 105)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.set_yticklabels(['0', '20', '40', '60', '80', '100'], fontsize=18)
            ax.grid(False)

            # Row label on leftmost column
            if col_idx == 0:
                pass


            # Collect legend from first valid subplot
            if legend_handles is None:
                legend_handles = [
                    Patch(
                        facecolor=colors[m],
                        label=method_labels[m],
                        hatch=hatches[m],
                        edgecolor='white' if hatches[m] else None,
                        linewidth=1 if hatches[m] else 0,
                    )
                    for m in method_order
                ]

    # ── Shared axis labels ────────────────────────────────────────────────────
    fig.text(0.5, 0.10, 'Dimensions Retained (%)', fontsize=28, ha='center', va='top')
    fig.text(0.045, 0.54, 'Score Retained (%)', fontsize=28, ha='center', va='center', rotation='vertical')
    fig.text(0.045, 0.77, 'Non-MRL', fontsize=22, ha='center', va='center', rotation='vertical')
    fig.text(0.045, 0.31, 'MRL', fontsize=22, ha='center', va='center', rotation='vertical')

    # ── Single shared legend ──────────────────────────────────────────────────
    if legend_handles is not None:
        fig.legend(
            handles=legend_handles,
            fontsize=24,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.06),
            ncol=6,
            framealpha=0.9,
        )

    plt.tight_layout(rect=[0.06, 0.11, 1, 1], w_pad=3.0)
    output_path = os.path.join(output_dir, 'average_results_multi_plot.pdf')
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("Creating multi-plot")

    RESULTS_PATH = os.path.join(EVALUATION_RESULTS_PATH, "comparison_results.csv")
    PLOTS_PATH = os.path.join(STORAGE_PATH, "/plots/")

    os.makedirs(PLOTS_PATH, exist_ok=True)

    df = pd.read_csv(RESULTS_PATH)

    models_non_mrl = [
        "jinaai/jina-embeddings-v2-small-en",
        "sentence-transformers/all-mpnet-base-v2",
    ]
    models_mrl = [
        "Alibaba-NLP/gte-multilingual-base",
        "Qwen/Qwen3-Embedding-0.6B",
    ]

    create_multi_plot(df, PLOTS_PATH, models_non_mrl, models_mrl)
    print("Done")
