import pandas as pd

df = pd.read_csv('storage/evaluation_results/comparison_results.csv')
df['Model'] = df['Model'].str.replace('/', '__', regex=False)

intrinsic_cols = ['spearman_loss', 'angular_loss', 'positional_loss']
metric_names = ['Spearman', 'Angular', 'Positional']

backbones = [
    'Alibaba-NLP__gte-multilingual-base',
    'jinaai__jina-embeddings-v2-small-en',
    'Qwen__Qwen3-Embedding-0.6B',
    'sentence-transformers__all-mpnet-base-v2'
]

backbone_caption = {
    'Alibaba-NLP__gte-multilingual-base': 'GTE-Multilingual-Base',
    'jinaai__jina-embeddings-v2-small-en': 'Jina-Embeddings-V2-Small-En',
    'Qwen__Qwen3-Embedding-0.6B': 'Qwen3-Embedding-0.6B',
    'sentence-transformers__all-mpnet-base-v2': 'All-MPNet-Base-V2'
}

backbone_label = {
    'Alibaba-NLP__gte-multilingual-base': 'gte',
    'jinaai__jina-embeddings-v2-small-en': 'jina',
    'Qwen__Qwen3-Embedding-0.6B': 'qwen',
    'sentence-transformers__all-mpnet-base-v2': 'mpnet'
}

backbone_dimensions = {
    'Alibaba-NLP__gte-multilingual-base': [32, 64, 128, 256],
    'jinaai__jina-embeddings-v2-small-en': [32, 64, 128, 256],
    'Qwen__Qwen3-Embedding-0.6B': [32, 64, 128, 256, 512],
    'sentence-transformers__all-mpnet-base-v2': [32, 64, 128, 256]
}

baselines = [
    ('PCA', 'pca'),
    ('Random Projection', 'random_projection'),
    ('Random Selection', 'random_selection'),
    ('Truncation', 'truncation'),
    ('Autoencoder', 'autoencoder'),
    (r'\method (Ours)', 'batch_20000_poslossfactor_1_linear'),
]

num_metrics = len(intrinsic_cols)
num_cols = num_metrics + 1

for backbone in backbones:
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\scriptsize")
    col_spec = 'l' + 'c' * num_metrics
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")
    header = "\\textbf{Method} & " + " & ".join(f"\\textbf{{{m}}}" for m in metric_names) + " \\\\"
    print(header)
    print("\\midrule")

    for dim_idx, dim in enumerate(backbone_dimensions[backbone]):
        print(f"\\multicolumn{{{num_cols}}}{{l}}{{\\textbf{{Dim = {dim}}}}} \\\\")
        print("\\midrule")

        for method_name, method_suffix in baselines:
            model_name = f"{backbone}_reduced_{dim}_{method_suffix}"
            row = df[df['Model'] == model_name]

            if len(row) == 0:
                print(f"WARNING: Model not found: {model_name}")
                continue

            row_data = row.iloc[0]
            vals = []
            for col in intrinsic_cols:
                val = row_data[col]
                if pd.isna(val):
                    vals.append('-')
                else:
                    vals.append(f'{val:.4f}')

            print(f"{method_name} & " + " & ".join(vals) + " \\\\")

        print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print(f"\\caption{{Intrinsic Evaluation Results for \\texttt{{{backbone_caption[backbone]}}}. Lower values indicate better preservation of geometric structure.}}")
    print(f"\\label{{tab:intrinsic_{backbone_label[backbone]}}}")
    print("\\end{table}")
    print()
