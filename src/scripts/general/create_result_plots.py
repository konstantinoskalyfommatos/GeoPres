import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
from utils.config import EVALUATION_RESULTS_PATH, PROJECT_ROOT


def get_desired_dimensions(model_base_name):
    """Return the list of desired dimensions for grouping."""
    if 'gte-multilingual' in model_base_name or "all-mpnet-base-v2" in model_base_name:
        return [32, 64, 128, 256, 768]
    elif 'jina-embeddings-v2-small-en' in model_base_name:
        return [32, 64, 128, 256, 512]
    raise ValueError(f"Unknown model base name: {model_base_name}")


def extract_embedding_dim(model_name):
    """Extract embedding dimension from model name."""
    if '_distilled_' in model_name:
        parts = model_name.split('_distilled_')
        dim_part = parts[1].split('_')[0]
        return int(dim_part)

    elif 'gte-multilingual-base' in model_name  or "all-mpnet-base-v2" in model_name:
        return 768
    elif 'jina-embeddings-v2-small-en' in model_name:
        return 512
    raise ValueError(f"Unknown model name format: {model_name}")


def get_grouped_dimension(dim, desired_dims):
    """Map actual dimension to the nearest desired dimension for grouping."""
    for d in desired_dims:
        if dim <= d:
            return d
    return desired_dims[-1]


def extract_method(model_name):
    """Extract dimensionality reduction method from model name."""
    
    if '_distilled_' not in model_name:
        return 'base'
    
    if 'batch_20000_poslossfactor_1' in model_name and "weighted" not in model_name:
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


def create_grouped_bar_plots(
    df,
    output_dir,
    task_columns,
    models
):
    """Create grouped bar charts for each task, grouped by dimensions."""
    colors = {
        'custom': '#1f77b4',      # Blue (Tableau)
        'pca': '#d62728',         # Red (Tableau)
        'random_projection': '#2ca02c',  # Green (Tableau)
        'random_selection': '#ff7f0e',   # Orange (Tableau)
        'truncation': '#9467bd',   # Purple (Tableau)
        'base': '#17becf',        # Cyan (Tableau)
        'autoencoder': '#bcbd22'   # Olive (Tableau)
    }

    for model_base_name, include_in_mlr in models.items():
        truncation_label = 'Truncation (Matryoshka)' if include_in_mlr else 'Truncation'
        model_dim_reduction_methods = {
            'custom': 'ALDRL (Ours)',
            'pca': 'PCA',
            'random_projection': 'Random Projection',
            'random_selection': 'Random Selection',
            'truncation': truncation_label,
            'autoencoder': 'Autoencoder'
        }
        desired_dims = get_desired_dimensions(model_base_name)
        model_data = df[df['Model'].str.contains(model_base_name.replace('/', '__'))]
        model_data = model_data.copy()
        model_data['dimension'] = model_data['Model'].apply(extract_embedding_dim)
        model_data['method'] = model_data['Model'].apply(extract_method)
        model_data['grouped_dim'] = model_data['dimension'].apply(
            lambda x: get_grouped_dimension(x, desired_dims)
        )
        model_data = model_data.dropna(subset=['dimension'])
        model_data = model_data[model_data['dimension'] != 2]
        valid_methods = ['base', 'custom', 'pca', 'random_projection', 'random_selection', 'truncation', 'autoencoder']
        model_data = model_data[model_data['method'].isin(valid_methods)]

        if len(model_data) == 0:
            print(f"No data found for model: {model_base_name}")
            continue

        for task_name, column_name in task_columns.items():
            fig, ax = plt.subplots(figsize=(14, 8))
            fig.suptitle(f'{task_name} by Dimension: {model_base_name}', 
                         fontsize=16, fontweight='bold')

            bar_width = 0.12
            x_positions = range(len(desired_dims))

            all_values = {}
            for method_key, method_name in model_dim_reduction_methods.items():
                method_data = model_data[model_data['method'] == method_key]
                values = []
                for dim in desired_dims:
                    dim_data = method_data[method_data['grouped_dim'] == dim]
                    if len(dim_data) > 0:
                        values.append(dim_data[column_name].values[0])
                    else:
                        values.append(0)
                all_values[method_key] = values

            base_data = model_data[model_data['method'] == 'base']
            base_score = base_data[column_name].values[0] if len(base_data) > 0 else 0

            winners = {}
            for dim_idx, dim in enumerate(desired_dims):
                dim_values = {k: v[dim_idx] for k, v in all_values.items()}
                if dim_idx == len(desired_dims) - 1:
                    dim_values['base'] = base_score
                winner = max(dim_values, key=dim_values.get)
                winners[dim_idx] = winner

            for i, (method_key, method_name) in enumerate(model_dim_reduction_methods.items()):
                values = all_values[method_key]
                offset = (i - len(dim_reduction_methods) / 2 + 0.5) * bar_width
                bars = ax.bar([x + offset for x in x_positions], values, bar_width,
                              label=method_name, color=colors[method_key], alpha=0.8)

                for dim_idx, bar in enumerate(bars):
                    if winners[dim_idx] == method_key:
                        bar.set_edgecolor('black')
                        bar.set_linewidth(2)

            last_pos = x_positions[-1]
            backbone_bar = ax.bar(last_pos, base_score, bar_width,
                                  label='Backbone (original)', color='gray', alpha=0.8,
                                  edgecolor='black', linewidth=2)

            ax.axhline(y=base_score, xmin=0, xmax=(last_pos + 0.5) / len(desired_dims),
                       color='gray', linestyle='--', linewidth=1.5)

            ax.set_xlabel('Embedding Dimension', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(d) for d in desired_dims])
            
            legend_handles = [Patch(color=colors[k], label=v) for k, v in model_dim_reduction_methods.items()]
            legend_handles.append(Patch(color='gray', label='Backbone (original)'))
            ax.legend(handles=legend_handles, fontsize=10, loc='best')
            
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1)

            plt.tight_layout()

            safe_model_name = model_base_name.replace('/', '_')
            output_path = os.path.join(output_dir, 
                f'{safe_model_name}_{task_name.lower()}_grouped_bar.jpg')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved grouped bar plot: {output_path}")
            plt.close()


if __name__ == "__main__":
    print("Creating performance plots")

    RESULTS_PATH = os.path.join(EVALUATION_RESULTS_PATH, "comparison_results.csv")
    PLOTS_PATH = os.path.join(PROJECT_ROOT, "storage/plots")

    os.makedirs(PLOTS_PATH, exist_ok=True)

    df = pd.read_csv(RESULTS_PATH)

    models_to_mlr = {
        "Alibaba-NLP/gte-multilingual-base": True,
        "jinaai/jina-embeddings-v2-small-en": False,
        "sentence-transformers/all-mpnet-base-v2": False
    }

    task_columns = {
        'AVG MTEB': '**AVG_MTEB**',
        'STS': '**AVG_STS**',
        'Retrieval': '**AVG_RETRIEVAL**',
        'Classification': '**AVG_CLASSIFICATION**',
        'Clustering': '**AVG_CLUSTERING**'
    }

    dim_reduction_methods = {
        'custom': 'ALDRL (Ours)',
        'pca': 'PCA',
        'random_projection': 'Random Projection',
        'random_selection': 'Random Selection',
        'truncation': 'Truncation (Matryoshka)',
        'autoencoder': 'Autoencoder'
    }
    
    # Create grouped bar chart plots
    create_grouped_bar_plots(df, PLOTS_PATH, task_columns, models_to_mlr)
    
    print("All plots created successfully")
