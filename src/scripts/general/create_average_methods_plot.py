import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from utils.config import EVALUATION_RESULTS_PATH, PROJECT_ROOT


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


def create_plot(df, output_dir, models):
    """
    Backbone-independent plot for AVG MTEB.

    For each reduction method, scores are normalized per backbone then averaged.
    X-axis : ratio of original dimensions retained (backbone-agnostic)
    Y-axis : mean normalized AVG MTEB score across all backbones
    Each method → one clean line + subtle ±1 std band.
    """
    # ── Colour palette per method (same as scatter plot) ─────────────────────
    method_colors = {
        'GeoPres':            '#377eb8',   # blue
        'pca':               '#e41a1c',   # red
        'random_projection': '#4daf4a',   # green
        'random_selection':  '#ff7f00',   # orange
        'truncation':        '#984ea3',   # purple
        'autoencoder':       '#17becf',   # teal
    }
    method_labels = {
        'GeoPres':            'GeoPres (Ours)',
        'pca':               'PCA',
        'random_projection': 'Random Projection',
        'random_selection':  'Random Selection',
        'truncation':        'Truncation',
        'autoencoder':       'Autoencoder',
    }
    method_markers = {
        'GeoPres':            'X',   # filled x
        'pca':               'v',   # triangle down
        'random_projection': 'P',   # filled plus
        'random_selection':  '^',   # triangle up
        'truncation':        'p',   # pentagon
        'autoencoder':       'h',   # hexagon 2
    }
    draw_order = ['GeoPres', 'random_selection', 'random_projection',
                  'autoencoder', 'pca', 'truncation']

    valid_methods = list(method_colors.keys())
    column_name = '**AVG_MTEB**'

    # ── Collect (backbone, method, ratio_retained, norm_score) per backbone ───
    records = []
    for model_base_name in models.keys():
        model_key = model_base_name.replace('/', '__')

        base_data = df[
            df['Model'].str.contains(model_key) &
            (df['Model'].apply(extract_method) == 'base')
        ]
        if base_data.empty:
            continue
        base_score   = base_data[column_name].values[0]
        original_dim = base_data['Model'].apply(extract_embedding_dim).values[0]

        model_data = df[df['Model'].str.contains(model_key)].copy()
        model_data['dimension'] = model_data['Model'].apply(extract_embedding_dim)
        model_data['method']    = model_data['Model'].apply(extract_method)
        model_data = model_data[model_data['dimension'] != 2]
        model_data = model_data[model_data['method'].isin(valid_methods)]

        for _, row in model_data.iterrows():
            ratio_retained = round(row['dimension'] / original_dim, 4)
            norm_score  = row[column_name] / base_score * 100
            records.append({'backbone': model_base_name,
                            'method': row['method'],
                            'ratio_retained': ratio_retained,
                            'norm_score':  norm_score})

    all_df = pd.DataFrame(records)

    # Round x to 2 decimals to align points across backbones cleanly
    all_df['ratio_bin'] = all_df['ratio_retained'].round(2)

    # Count distinct backbones per ratio bin for x-axis labels
    backbone_counts = all_df.groupby('ratio_bin')['backbone'].nunique()

    # ── Aggregate per (method, ratio_bin): mean ───────────────────────────────
    agg_mean = (all_df.groupby(['ratio_bin', 'method'])['norm_score']
                   .mean()
                   .unstack(fill_value=0)
                   .reindex(columns=draw_order, fill_value=0)
                   .sort_index())

    ratio_bins = agg_mean.index.tolist()
    x_positions = np.arange(len(ratio_bins))

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 13))

    for method in draw_order:
        if method not in agg_mean.columns:
            continue
        mean_vals = agg_mean[method].values
        color = method_colors[method]

        ax.plot(x_positions, mean_vals,
                 marker=method_markers[method],
                 markersize=16,          # Increased marker size
                 linewidth=4.0,          # Thicker lines like reference
                 color=color, 
                 label=method_labels[method],
                 markeredgecolor='black',# Black edge for definition
                 markeredgewidth=0.5)    # Thin edge

    # Baseline reference
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1.5, zorder=0)

    # ─ Axes & styling ───────────────────────────────────────────────────────
    ax.set_xlabel('Dimensions Retained (%)', fontsize=28)
    ax.set_ylabel('MTEB Score Retained (%)', fontsize=28)

    ax.set_xlim(-0.45, len(ratio_bins) - 0.55)
    ax.set_ylim(65, 102)
    ax.set_yticks(np.arange(70, 105, 5))
    ax.set_yticklabels([f'{int(t)}' for t in np.arange(70, 105, 5)], fontsize=24)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'(n={backbone_counts[p]}) {p * 100:.0f}' for p in ratio_bins],
                        fontsize=24, rotation=45)
    ax.tick_params(axis='y', labelsize=24)
    
    # Enhance grid to match reference style
    ax.grid(True, axis='both', alpha=0.3, linestyle='-', linewidth=0.8)
    
    # Make spines (box) more visible like reference
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    # Update legend handle sizes to match the large markers
    ax.legend(fontsize=25, 
              loc='lower right', 
              framealpha=0.9,
              handlelength=2.0,
              markerscale=1.0)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'average_methods_plot.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("Creating plot")

    RESULTS_PATH = os.path.join(EVALUATION_RESULTS_PATH, "comparison_results.csv")
    PLOTS_PATH = os.path.join(PROJECT_ROOT, "storage/plots/")

    os.makedirs(PLOTS_PATH, exist_ok=True)

    df = pd.read_csv(RESULTS_PATH)

    models_to_mlr = {
        "Alibaba-NLP/gte-multilingual-base":       True,
        "jinaai/jina-embeddings-v2-small-en":      False,
        "Qwen/Qwen3-Embedding-0.6B":               True,
        "sentence-transformers/all-mpnet-base-v2": False,
    }

    create_plot(df, PLOTS_PATH, models_to_mlr)
    print("Done")