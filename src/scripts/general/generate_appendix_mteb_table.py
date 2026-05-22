import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import glob
from utils.config import EVALUATION_RESULTS_PATH, PROJECT_ROOT


def extract_embedding_dim(model_name):
    """Extract embedding dimension from model name."""
    if '_distilled_' in model_name:
        parts = model_name.split('_distilled_')
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
    if '_distilled_' not in model_name:
        return 'base'
    if 'batch_20000_poslossfactor_1_linear' in model_name and "weighted" not in model_name:
        return 'custom'
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


def create_scatter_plot(df, output_dir, models):
    """
    Backbone-aware scatter plot for AVG MTEB.

    - X-axis: % of original dimensions reduced
    - Y-axis: normalized AVG MTEB score (M(f)/M(base))
    - Point shape: backbone
    - Point/line color: reduction method
    - Line + shaded band: mean ± std across backbones per method per x-bin
    """
    method_colors = {
        'custom':           '#e63946',
        'pca':              '#1d3557',
        'random_projection':'#2a9d8f',
        'random_selection': '#e9c46a',
        'truncation':       '#f4a261',
        'autoencoder':      '#90a0b0',
    }
    method_labels = {
        'custom':           'GeoPres (Ours)',
        'pca':              'PCA',
        'random_projection':'Random Projection',
        'random_selection': 'Random Selection',
        'truncation':       'Truncation',
        'autoencoder':      'Autoencoder',
    }
    # Per-backbone marker shapes and short display names
    backbone_markers = {
        'gte-multilingual-base':      ('o', 'GTE'),
        'jina-embeddings-v2-small-en':('s', 'Jina'),
        'Qwen3-Embedding-0.6B':       ('^', 'Qwen'),
        'all-mpnet-base-v2':          ('D', 'MPNet'),
    }

    valid_methods = ['custom', 'pca', 'random_projection', 'random_selection', 'truncation', 'autoencoder']
    column_name = '**AVG_MTEB**'

    # Collect all individual (backbone, method, pct_reduced, norm_score) tuples
    records = []

    for model_base_name in models.keys():
        model_key = model_base_name.replace('/', '__')
        # Short backbone key for marker lookup
        backbone_key = next((k for k in backbone_markers if k in model_base_name), None)
        if backbone_key is None:
            continue

        base_data = df[
            df['Model'].str.contains(model_key) &
            (df['Model'].apply(extract_method) == 'base')
        ]
        if len(base_data) == 0:
            continue

        base_score = base_data[column_name].values[0]
        original_dim = base_data['Model'].apply(extract_embedding_dim).values[0]

        model_data = df[df['Model'].str.contains(model_key)].copy()
        model_data['dimension'] = model_data['Model'].apply(extract_embedding_dim)
        model_data['method'] = model_data['Model'].apply(extract_method)
        model_data = model_data.dropna(subset=['dimension'])
        model_data = model_data[model_data['dimension'] != 2]
        model_data = model_data[model_data['method'].isin(valid_methods)]

        for _, row in model_data.iterrows():
            pct_reduced = (1 - row['dimension'] / original_dim) * 100
            norm_score = row[column_name] / base_score
            records.append({
                'backbone': backbone_key,
                'method': row['method'],
                'pct_reduced': pct_reduced,
                'norm_score': norm_score,
            })

    all_df = pd.DataFrame(records)

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 7.5))

    # Mean ± std band per method (aggregated across backbones)
    for method in valid_methods:
        mdf = all_df[all_df['method'] == method]
        if mdf.empty:
            continue

        agg = (mdf.groupby('pct_reduced')['norm_score']
                  .agg(['mean', 'std'])
                  .reset_index()
                  .sort_values('pct_reduced'))
        agg['std'] = agg['std'].fillna(0)

        color = method_colors[method]
        ax.fill_between(agg['pct_reduced'],
                        agg['mean'] - agg['std'],
                        agg['mean'] + agg['std'],
                        color=color, alpha=0.12, zorder=2)
        ax.plot(agg['pct_reduced'], agg['mean'],
                color=color, linewidth=2.2, zorder=3)

    # Individual backbone points (on top)
    for method in valid_methods:
        mdf = all_df[all_df['method'] == method]
        if mdf.empty:
            continue
        color = method_colors[method]

        for backbone_key, (marker, _) in backbone_markers.items():
            bdf = mdf[mdf['backbone'] == backbone_key]
            if bdf.empty:
                continue
            ax.scatter(bdf['pct_reduced'], bdf['norm_score'],
                       color=color, marker=marker, s=90,
                       edgecolors='white', linewidths=0.7,
                       zorder=5, alpha=0.92)

    # Reference line
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.2,
               zorder=1, alpha=0.55, label='_nolegend_')
    ax.text(0.5, 1.005, 'Baseline (full dims)', transform=ax.get_yaxis_transform(),
            ha='left', va='bottom', fontsize=9, color='gray', style='italic')

    # ── Axes ─────────────────────────────────────────────────────────────────
    x_min = max(0, all_df['pct_reduced'].min() - 5)
    x_max = min(100, all_df['pct_reduced'].max() + 3)
    ax.set_xlim(x_min, x_max)

    y_min = max(0.55, all_df['norm_score'].min() - 0.03)
    ax.set_ylim(y_min, 1.05)

    ax.set_xlabel('Dimensions Reduced (%)', fontsize=14)
    ax.set_ylabel('Normalized AVG MTEB Score', fontsize=14)
    ax.set_title(
        'AVG MTEB Performance vs. Dimensionality Reduction\n'
        'Lines = mean across backbones · Bands = ±1 std · Shapes = backbone',
        fontsize=15, fontweight='bold', pad=14
    )
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.25, axis='y', zorder=0)
    ax.grid(True, alpha=0.12, axis='x', zorder=0)

    # ── Legends ──────────────────────────────────────────────────────────────
    # Method legend (color patches)
    method_handles = [
        mpatches.Patch(color=method_colors[m], label=method_labels[m])
        for m in valid_methods
    ]
    legend1 = ax.legend(
        handles=method_handles,
        title='Method', title_fontsize=11,
        fontsize=10.5, loc='lower left',
        framealpha=0.95, edgecolor='#cccccc',
        handletextpad=0.6, labelspacing=0.45
    )
    ax.add_artist(legend1)

    # Backbone legend (marker shapes, black)
    backbone_handles = [
        plt.scatter([], [], marker=marker, color='#444444', s=80,
                    edgecolors='white', linewidths=0.7, label=label)
        for _, (marker, label) in backbone_markers.items()
    ]
    ax.legend(
        handles=backbone_handles,
        title='Backbone', title_fontsize=11,
        fontsize=10.5, loc='lower center',
        framealpha=0.95, edgecolor='#cccccc',
        handletextpad=0.6, labelspacing=0.45,
        bbox_to_anchor=(0.5, 0.01)
    )

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'scatter_avg_mteb.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("Creating scatter plot")

    RESULTS_PATH = os.path.join(EVALUATION_RESULTS_PATH, "comparison_results.csv")
    PLOTS_PATH = os.path.join(PROJECT_ROOT, "storage/plots/scatter")

    os.makedirs(PLOTS_PATH, exist_ok=True)
    for old_file in glob.glob(os.path.join(PLOTS_PATH, '*.png')):
        os.remove(old_file)

    df = pd.read_csv(RESULTS_PATH)

    models_to_mlr = {
        "Alibaba-NLP/gte-multilingual-base":       True,
        "jinaai/jina-embeddings-v2-small-en":      False,
        "Qwen/Qwen3-Embedding-0.6B":               True,
        "sentence-transformers/all-mpnet-base-v2": False,
    }

    create_scatter_plot(df, PLOTS_PATH, models_to_mlr)
    print("Done")