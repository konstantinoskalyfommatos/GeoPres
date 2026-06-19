import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
from config import EVALUATION_RESULTS_PATH, PROJECT_ROOT


def get_desired_dimensions(model_name=None):
    base_dims = [32, 64, 128, 256]
    if model_name and 'Qwen' in model_name:
        return base_dims + [512]
    return base_dims


def extract_embedding_dim(model_name):
    """Extract embedding dimension from model name."""
    if '_reduced_' in model_name:
        parts = model_name.split('_reduced_')
        dim_part = parts[1].split('_')[0]
        return int(dim_part)

    elif 'gte-multilingual-base' in model_name  or "all-mpnet-base-v2" in model_name:
        return 768
    elif 'jina-embeddings-v2-small-en' in model_name:
        return 512
    elif 'Qwen3-Embedding-0.6B' in model_name:
        return 1024
    raise ValueError(f"Unknown model name format: {model_name}")


def get_grouped_dimension(dim, desired_dims):
    """Map actual dimension to the nearest desired dimension for grouping."""
    for d in desired_dims:
        if dim <= d:
            return d
    return desired_dims[-1]


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


def create_grouped_bar_plots(
    df,
    output_dir,
    task_columns,
    models
):
    """Create grouped bar charts for each task, grouped by dimensions."""
    colors = {
        'GeoPres': '#417aa2',            # Muted blue (GeoPres)
        'random_selection': '#d27f36',   # Muted orange (Random Selection)
        'random_projection': '#499549',  # Muted green (Random Projection)
        'autoencoder': '#3ca8b3',        # Muted cyan (Autoencoder)
        'pca': '#b84646',                # Muted red (PCA)
        'truncation': '#8d70a7',         # Muted purple (Truncation)
        'base': '#1f2937',               # Almost Black
    }

    for model_base_name, include_in_mlr in models.items():
        truncation_label = 'Truncation (Matryoshka)' if include_in_mlr else 'Truncation'
        model_dim_reduction_methods = {
            'GeoPres': 'GeoPres (Ours)',
            'random_selection': 'Random Selection',
            'random_projection': 'Random Projection',
            'autoencoder': 'Autoencoder',
            'pca': 'PCA',
            'truncation': truncation_label
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
        valid_methods = ['base', 'GeoPres', 'pca', 'random_projection', 'random_selection', 'truncation', 'autoencoder']
        model_data = model_data[model_data['method'].isin(valid_methods)]

        if len(model_data) == 0:
            print(f"No data found for model: {model_base_name}")
            continue

        for task_name, column_name in task_columns.items():
            fig, ax = plt.subplots(figsize=(20, 14))
            
            # Get original dimension from base model data
            base_data = model_data[model_data['method'] == 'base']
            original_dim = base_data['dimension'].values[0] if len(base_data) > 0 else 'N/A'

            bar_width = 0.12
            x_positions = range(len(desired_dims))

            base_data = model_data[model_data['method'] == 'base']
            base_score = base_data[column_name].values[0] if len(base_data) > 0 else 1

            all_values = {}
            for method_key, method_name in model_dim_reduction_methods.items():
                method_data = model_data[model_data['method'] == method_key]
                values = []
                for dim in desired_dims:
                    dim_data = method_data[method_data['grouped_dim'] == dim]
                    if len(dim_data) > 0:
                        values.append(dim_data[column_name].values[0] / base_score * 100)
                    else:
                        values.append(0)
                all_values[method_key] = values

            winners = {}
            for dim_idx, dim in enumerate(desired_dims):
                dim_values = {k: v[dim_idx] for k, v in all_values.items()}
                winner = max(dim_values, key=dim_values.get)
                winners[dim_idx] = winner

            last_pos = x_positions[-1]

            hatches = {
                'GeoPres': '///',
                'pca': '',
                'random_projection': '',
                'random_selection': '',
                'truncation': '',
                'base': '',
                'autoencoder': ''
            }

            for i, (method_key, method_name) in enumerate(model_dim_reduction_methods.items()):
                values = all_values[method_key]
                offset = (i - len(dim_reduction_methods) / 2 + 0.5) * bar_width
                
                # Apply hatch and edgecolor white for GeoPres method
                edgecolor = 'white' if method_key == 'GeoPres' else None
                linewidth = 1 if method_key == 'GeoPres' else 0

                bars = ax.bar([x + offset for x in x_positions], values, bar_width,
                              label=method_name, facecolor=colors[method_key], 
                              edgecolor=edgecolor, linewidth=linewidth,
                              hatch=hatches.get(method_key, ''), alpha=0.9 if method_key == 'GeoPres' else 0.8)

                for dim_idx, bar in enumerate(bars):
                    is_winner = winners[dim_idx] == method_key
                    if is_winner:
                        bar.set_edgecolor('black')
                        bar.set_linewidth(2)
                    
                    height = bar.get_height()
                    if height > 0:
                        label = f'{height:.0f}'
                        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                 label, ha='center', va='bottom', fontsize=22,
                                 fontweight='bold')

            ax.axhline(y=100, xmin=0, xmax=1.0,
                        color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

            ax.set_xlabel('Embedding Dimension', fontsize=32)
            ax.set_ylabel('Score Retained (%)', fontsize=32)
            ax.set_xticks(x_positions)
            if original_dim != 'N/A' and original_dim > 0:
                x_labels = [f'{d} ({int(d/original_dim*100)}%)' for d in desired_dims]
            else:
                x_labels = [str(d) for d in desired_dims]
            ax.set_xticklabels(x_labels, fontsize=23)
            
            legend_handles = []
            for k, v in model_dim_reduction_methods.items():
                legend_handles.append(Patch(facecolor=colors[k], label=v, hatch=hatches.get(k, ''), 
                                            edgecolor='white' if hatches.get(k, '') else None, linewidth=1 if hatches.get(k, '') else 0))
            ax.legend(handles=legend_handles, fontsize=22, loc='lower left')
            
            ax.grid(False)
            ax.set_ylim(0, 105)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.set_yticklabels(['0', '20', '40', '60', '80', '100'], fontsize=24)

            plt.tight_layout()

            safe_model_name = model_base_name.replace('/', '_')
            output_path = os.path.join(output_dir, 
                f'{safe_model_name}_{task_name.lower()}_grouped_bar.pdf')
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Saved grouped bar plot: {output_path}")
            plt.close()


if __name__ == "__main__":
    print("Creating performance plots")

    RESULTS_PATH = os.path.join(EVALUATION_RESULTS_PATH, "comparison_results.csv")
    PLOTS_PATH = os.path.join(PROJECT_ROOT, "storage/plots/main_method")

    os.makedirs(PLOTS_PATH, exist_ok=True)

    df = pd.read_csv(RESULTS_PATH)

    models_to_mlr = {
        "Alibaba-NLP/gte-multilingual-base": True,
        "jinaai/jina-embeddings-v2-small-en": False,
        "Qwen/Qwen3-Embedding-0.6B": True,
        "sentence-transformers/all-mpnet-base-v2": False,
    }

    task_columns = {
        'AVG MTEB': '**AVG_MTEB**',
        'STS': '**AVG_STS**',
        'Retrieval': '**AVG_RETRIEVAL**',
        'Classification': '**AVG_CLASSIFICATION**',
        'Clustering': '**AVG_CLUSTERING**'
    }

    dim_reduction_methods = {
        'GeoPres': 'GeoPres (Ours)',
        'autoencoder': 'Autoencoder',
        'random_selection': 'Random Selection',
        'random_projection': 'Random Projection',
        'pca': 'PCA',
        'truncation': 'Truncation (Matryoshka)'
    }
    
    # Create grouped bar chart plots
    create_grouped_bar_plots(df, PLOTS_PATH, task_columns, models_to_mlr)
    
    print("All plots created successfully")
