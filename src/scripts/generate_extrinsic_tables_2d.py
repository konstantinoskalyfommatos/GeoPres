import pandas as pd
import os

from config import STORAGE_PATH


df = pd.read_csv(
    os.path.join(
        STORAGE_PATH,
        'evaluation_results/comparison_results.csv'
    )
)
df['Model'] = df['Model'].str.replace('/', '__', regex=False)

extrinsic_cols = ['**AVG_MTEB**', '**AVG_STS**', '**AVG_RETRIEVAL**',
                  '**AVG_CLASSIFICATION**', '**AVG_CLUSTERING**']

backbones = [
    'Alibaba-NLP__gte-multilingual-base',
    'jinaai__jina-embeddings-v2-small-en',
    'Qwen__Qwen3-Embedding-0.6B',
    'sentence-transformers__all-mpnet-base-v2'
]

backbone_display = {
    'Alibaba-NLP__gte-multilingual-base': 'GTE-Multilingual-Base',
    'jinaai__jina-embeddings-v2-small-en': 'Jina-Embeddings-V2-Small-En',
    'Qwen__Qwen3-Embedding-0.6B': 'Qwen3-Embedding-0.6B',
    'sentence-transformers__all-mpnet-base-v2': 'All-MPNet-Base-V2'
}

dimensions = [2]

baselines = {
    'PCA': 'pca',
    'Random Projection': 'random_projection',
    'Random Selection': 'random_selection',
    'Autoencoder': 'autoencoder',
    'Truncation': 'truncation',
    'GeoPres (Ours)': 'batch_20000_poslossfactor_1'
}

def format_val(val):
    if pd.isna(val):
        return '-'
    return f'{val:.4f}'

print("\n" + "="*80)
print("EXTRINSIC TABLES: Main Method 2D (Normalized by Backbone Score)")
print("="*80 + "\n")

for backbone in backbones:
    # Get base (full-dimensional) model scores for normalization
    base_row = df[df['Model'] == backbone]
    if len(base_row) == 0:
        print(f"WARNING: Base model not found: {backbone}")
        continue
    base_data = base_row.iloc[0]

    # Compute normalization factors
    norm_factors = {}
    for col in extrinsic_cols:
        base_score = base_data[col]
        if pd.isna(base_score) or base_score == 0:
            norm_factors[col] = None
        else:
            norm_factors[col] = base_score

    print(f"\\subsubsection{{{backbone_display[backbone]}}}")
    print("\\begin{landscape}")
    print("\\begin{longtable}{lccccc}")
    print("\\toprule")

    col_labels = ['Method', 'AVG MTEB', 'AVG STS', 'AVG Retrieval', 'AVG Classification', 'AVG Clustering']
    print(' & '.join(col_labels) + ' \\\\')
    print('\\midrule')
    print('\\midrule')

    for dim in dimensions:
        first_in_dim = True

        for method_name, method_suffix in baselines.items():
            model_name = f"{backbone}_reduced_{dim}_{method_suffix}"
            row = df[df['Model'] == model_name]

            if len(row) == 0:
                print(f"WARNING: Model not found: {model_name}")
                continue

            row_data = row.iloc[0]
            vals = []
            for col in extrinsic_cols:
                val = row_data[col]
                base_score = norm_factors[col]
                if pd.isna(val) or base_score is None or base_score == 0:
                    vals.append('-')
                else:
                    normalized = val / base_score
                    vals.append(f'{normalized:.4f}')

            if first_in_dim:
                print(f"\\multicolumn{{6}}{{l}}{{\\textbf{{Dim = {dim}}}}} \\\\")
                print("\\midrule")
                first_in_dim = False

            print(f"{method_name} & " + ' & '.join(vals) + ' \\\\')

    print("\\bottomrule")
    print("\\end{longtable}")
    print("\\end{landscape}")
    print("\n")
