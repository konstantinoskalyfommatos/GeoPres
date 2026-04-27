import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
from utils.config import EVALUATION_RESULTS_PATH, PROJECT_ROOT


def get_desired_dimensions():
    return [32, 64, 128, 256]


def extract_embedding_dim(model_name):
    """Extract embedding dimension from model name."""
    if '_distilled_' in model_name:
        parts = model_name.split('_distilled_')
        dim_part = parts[1].split('_')[0]
        return int(dim_part)

    elif 'gte-multilingual-base' in model_name:
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
    """Extract loss function variant from model name."""
    
    if '_distilled_' not in model_name:
        return 'base'
    
    if 'poslossfactor_1' in model_name and 'weighted' not in model_name:
        return 'aldrl'
    elif 'angular_loss' in model_name:
        return 'aldrl_angular'
    elif 'poslossfactor_0' in model_name and 'weighted' not in model_name:
        return 'aldrl_angular'
    elif 'poslossfactor_0_weighted' in model_name:
        return 'aldrl_angular_weighted'
    elif 'spearman' in model_name and 'weighted' not in model_name:
        return 'aldrl_spearman'
    elif 'spearman_weighted' in model_name:
        return 'aldrl_spearman_weighted'
    return None


def create_grouped_bar_plots(
    df,
    output_dir,
    task_columns,
    models
):
    """Create grouped bar charts for each task, grouped by dimensions."""
    colors = {
        'aldrl': '#3b82f6',                    # Blue
        'aldrl_angular': '#d1d5db',            # Light Gray
        'aldrl_angular_weighted': '#9ca3af',   # Gray
        'aldrl_spearman': '#6b7280',           # Dark Gray
        'aldrl_spearman_weighted': '#4b5563',  # Darker Gray
    }

    for model_base_name, include_in_mlr in models.items():
        model_dim_reduction_methods = {
            'aldrl': 'ALDRL (Ours)',
            'aldrl_angular': 'ALDRL Angular',
            'aldrl_angular_weighted': 'ALDRL Angular Weighted',
            'aldrl_spearman': 'ALDRL Spearman',
            'aldrl_spearman_weighted': 'ALDRL Spearman Weighted',
        }
        desired_dims = get_desired_dimensions()
        model_data = df[df['Model'].str.contains(model_base_name.replace('/', '__'))]
        model_data = model_data.copy()
        model_data['dimension'] = model_data['Model'].apply(extract_embedding_dim)
        model_data['method'] = model_data['Model'].apply(extract_method)
        model_data['grouped_dim'] = model_data['dimension'].apply(
            lambda x: get_grouped_dimension(x, desired_dims)
        )
        model_data = model_data.dropna(subset=['dimension'])
        model_data = model_data[model_data['dimension'] != 2]
        valid_methods = ['base', 'aldrl', 'aldrl_angular', 
                        'aldrl_angular_weighted', 'aldrl_spearman', 
                        'aldrl_spearman_weighted']
        model_data = model_data[model_data['method'].isin(valid_methods)]

        if len(model_data) == 0:
            print(f"No data found for model: {model_base_name}")
            continue

        base_only_data = model_data[model_data['method'] == 'base']
        original_dim = base_only_data['dimension'].values[0] if len(base_only_data) > 0 else 'N/A'

        for task_name, column_name in task_columns.items():
            fig, ax = plt.subplots(figsize=(20, 14))
            
            fig.suptitle(f'{task_name} by Dimension: {model_base_name} (Original: {original_dim}D)', 
                          fontsize=26, fontweight='bold')

            bar_width = 0.10
            x_positions = range(len(desired_dims))

            base_score = base_only_data[column_name].values[0] if len(base_only_data) > 0 else 1

            all_values = {}
            for method_key, method_name in model_dim_reduction_methods.items():
                method_data = model_data[model_data['method'] == method_key]
                values = []
                for dim in desired_dims:
                    dim_data = method_data[method_data['grouped_dim'] == dim]
                    if len(dim_data) > 0:
                        values.append(dim_data[column_name].values[0] / base_score)
                    else:
                        values.append(0)
                all_values[method_key] = values

            winners = {}
            for dim_idx, dim in enumerate(desired_dims):
                dim_values = {k: v[dim_idx] for k, v in all_values.items()}
                winner = max(dim_values, key=dim_values.get)
                winners[dim_idx] = winner

            dim_reduction_methods = model_dim_reduction_methods
            last_pos = x_positions[-1]

            hatches = {
                'aldrl': '///',
                'aldrl_angular': '',
                'aldrl_angular_weighted': '',
                'aldrl_spearman': '',
                'aldrl_spearman_weighted': ''
            }

            for i, (method_key, method_name) in enumerate(model_dim_reduction_methods.items()):
                values = all_values[method_key]
                offset = (i - len(dim_reduction_methods) / 2 + 0.5) * bar_width
                
                # Apply hatch and edgecolor white for aldrl
                edgecolor = 'white' if method_key == 'aldrl' else None
                linewidth = 1 if method_key == 'aldrl' else 0

                bars = ax.bar([x + offset for x in x_positions], values, bar_width,
                              label=method_name, facecolor=colors[method_key], 
                              edgecolor=edgecolor, linewidth=linewidth,
                              hatch=hatches.get(method_key, ''), alpha=0.9 if method_key == 'aldrl' else 0.8)

                for dim_idx, bar in enumerate(bars):
                    is_winner = winners[dim_idx] == method_key
                    if is_winner:
                        bar.set_edgecolor('black')
                        bar.set_linewidth(2)
                    
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                f'{height:.2f}', ha='center', va='bottom', fontsize=12,
                                fontweight='bold' if is_winner else 'normal')

            ax.axhline(y=1.0, xmin=0, xmax=1.0,
                       color='gray', linestyle='--', linewidth=1.5)

            ax.set_xlabel('Embedding Dimension', fontsize=22)
            ax.set_ylabel('Normalized Score (relative to original)', fontsize=22)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(d) for d in desired_dims], fontsize=18)
            
            legend_handles = []
            for k, v in dim_reduction_methods.items():
                legend_handles.append(Patch(facecolor=colors[k], label=v, hatch=hatches.get(k, ''), 
                                            edgecolor='white' if hatches.get(k, '') else None, linewidth=1 if hatches.get(k, '') else 0))
            ax.legend(handles=legend_handles, fontsize=16, loc='best')
            
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1.1)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

            plt.tight_layout()

            safe_model_name = model_base_name.replace('/', '_')
            output_path = os.path.join(output_dir, 
                f'{safe_model_name}_{task_name.lower()}_alternative_losses_grouped_bar.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved grouped bar plot: {output_path}")
            plt.close()


if __name__ == "__main__":
    print("Creating alternative loss functions comparison plots")

    RESULTS_PATH = os.path.join(EVALUATION_RESULTS_PATH, "comparison_results.csv")
    PLOTS_PATH = os.path.join(PROJECT_ROOT, "storage/plots/alternative_loss_functions")

    os.makedirs(PLOTS_PATH, exist_ok=True)

    df = pd.read_csv(RESULTS_PATH)

    models_to_mlr = {
        "Alibaba-NLP/gte-multilingual-base": True,
        "jinaai/jina-embeddings-v2-small-en": False,
    }

    task_columns = {
        'AVG MTEB': '**AVG_MTEB**',
        'STS': '**AVG_STS**',
        'Retrieval': '**AVG_RETRIEVAL**',
        'Classification': '**AVG_CLASSIFICATION**',
        'Clustering': '**AVG_CLUSTERING**'
    }

    dim_reduction_methods = {
        'aldrl': 'ALDRL (Ours)',
        'aldrl_angular': 'ALDRL Angular',
        'aldrl_angular_weighted': 'ALDRL Angular Weighted',
        'aldrl_spearman': 'ALDRL Spearman',
        'aldrl_spearman_weighted': 'ALDRL Spearman Weighted',
    }
    
    create_grouped_bar_plots(df, PLOTS_PATH, task_columns, models_to_mlr)
    
    print("All plots created successfully")
