import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.patches import Patch
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
    if 'batch_20000_poslossfactor_1' in model_name and "weighted" not in model_name:
        return 'gpl'
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
    Scatter plot of normalized AVG MTEB vs. ratio of dimensions retained.

    X-axis : ratio of original dimensions retained (backbone-agnostic)
    Y-axis : normalized AVG MTEB score (per-backbone normalization)
    Shape   : backbone identity (4 distinct markers)
    Colour  : dimensionality reduction method
    """
    column_name = '**AVG_MTEB**'

    # ── Colour palette per method ────────────────────────────────────────────
    method_colors = {
        'gpl':            '#377eb8',   # blue
        'pca':               '#e41a1c',   # red
        'random_projection': '#4daf4a',   # green
        'random_selection':  '#ff7f00',   # orange
        'truncation':        '#984ea3',   # purple
        'autoencoder':       '#17becf',   # teal
    }
    method_labels = {
        'gpl':            'GPL (Ours)',
        'pca':               'PCA',
        'random_projection': 'Random Projection',
        'random_selection':  'Random Selection',
        'truncation':        'Truncation',
        'autoencoder':       'Autoencoder',
    }
    valid_methods = list(method_colors.keys())

    # ── Shape & label per backbone ───────────────────────────────────────────
    backbone_shapes = {
        "Alibaba-NLP/gte-multilingual-base":       'D',   # diamond
        "jinaai/jina-embeddings-v2-small-en":      's',   # square
        "Qwen/Qwen3-Embedding-0.6B":               '*',   # star
        "sentence-transformers/all-mpnet-base-v2": 'o',   # circle
    }
    backbone_labels = {
        "Alibaba-NLP/gte-multilingual-base":       'gte-multilingual-base',
        "jinaai/jina-embeddings-v2-small-en":      'jina-embeddings-v2-small-en',
        "Qwen/Qwen3-Embedding-0.6B":               'Qwen3-Embedding-0.6B',
        "sentence-transformers/all-mpnet-base-v2": 'all-mpnet-base-v2',
    }

    # ── Collect (backbone, method, ratio_retained, norm_score) ───────────────
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
            norm_score  = row[column_name] / base_score
            records.append({
                'backbone':    model_base_name,
                'method':      row['method'],
                'ratio_retained': ratio_retained,
                'norm_score':  norm_score,
            })

    all_df = pd.DataFrame(records)

    # ── Map each unique ratio retained to an equidistant index ───────────────
    sorted_ratios = sorted(all_df['ratio_retained'].unique())
    ratio_to_idx = {ratio: idx for idx, ratio in enumerate(sorted_ratios)}
    all_df['x_pos'] = all_df['ratio_retained'].map(ratio_to_idx)

    # Count distinct backbones per ratio retained for x-axis labels
    backbone_counts = all_df.groupby('ratio_retained')['backbone'].nunique()

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(30, 20))

    draw_order = ['gpl', 'random_selection', 'random_projection', 'autoencoder', 'pca', 'truncation']

    for backbone in models.keys():
        shape = backbone_shapes.get(backbone, 'o')
        bb_data = all_df[all_df['backbone'] == backbone]
        for method in draw_order:
            m_data = bb_data[bb_data['method'] == method]
            if m_data.empty:
                continue
            color = method_colors[method]
            ax.scatter(
                m_data['x_pos'], m_data['norm_score'],
                marker=shape, s=500, color=color,
                edgecolors='black', linewidth=0.8,
                zorder=3, alpha=0.9,
            )

    # Baseline reference
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, zorder=1)

    # Vertical dotted lines at each equidistant index position
    for idx in range(len(sorted_ratios)):
        ax.axvline(x=idx, color='gray', linestyle=':', linewidth=0.8, alpha=0.5, zorder=0)

    # ── Axes & styling ───────────────────────────────────────────────────────
    ax.set_xlabel('Dimensions Retained Ratio', fontsize=22)
    ax.set_ylabel('MTEB Score Retained Ratio\n(per-backbone normalization)', fontsize=22)

    ax.set_xlim(-0.5, len(sorted_ratios) - 0.5)
    ax.set_ylim(0.65, 1.0)
    ax.set_yticks([0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    ax.set_xticks(range(len(sorted_ratios)))
    ax.set_xticklabels([f'{ratio:.2f} (n={backbone_counts[ratio]})' for ratio in sorted_ratios], fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Legend: colour = method, shape = backbone ────────────────────────────
    from matplotlib.lines import Line2D

    # Method legend (colours)
    method_handles = []
    for m in draw_order:
        method_handles.append(
            Patch(facecolor=method_colors[m], label=method_labels[m])
        )

    # Backbone legend (shapes) — Line2D so markers render correctly in legend
    backbone_handles = []
    for backbone in models.keys():
        shape = backbone_shapes.get(backbone, 'o')
        backbone_handles.append(
            Line2D([0], [0], marker=shape, color='black',
                   label=backbone_labels[backbone],
                   markeredgewidth=1.5, markersize=12,
                   linestyle='None')
        )

    method_legend = ax.legend(
        handles=method_handles,
        title='Reduction Method',
        fontsize=28, title_fontsize=28,
        loc='upper left', framealpha=0.9,
    )
    ax.add_artist(method_legend)

    backbone_legend = ax.legend(
        handles=backbone_handles,
        title='Backbone',
        fontsize=28, title_fontsize=28,
        loc='lower right', framealpha=0.9,
    )
    ax.add_artist(backbone_legend)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'scatter_plot.png')
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("Creating scatter plot")

    RESULTS_PATH = os.path.join(EVALUATION_RESULTS_PATH, "comparison_results.csv")
    PLOTS_PATH = os.path.join(PROJECT_ROOT, "storage/plots/")

    os.makedirs(PLOTS_PATH, exist_ok=True)
    for old_file in glob.glob(os.path.join(PLOTS_PATH, 'scatter_plot*')):
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
