import pandas as pd

df = pd.read_csv('storage/evaluation_results/comparison_results.csv')
df['Model'] = df['Model'].str.replace('/', '__', regex=False)

extrinsic_cols = ['**AVG_MTEB**', '**AVG_STS**', '**AVG_RETRIEVAL**',
                  '**AVG_CLASSIFICATION**', '**AVG_CLUSTERING**']

backbones = [
    'Alibaba-NLP__gte-multilingual-base',
    'jinaai__jina-embeddings-v2-small-en',
]

backbone_display = {
    'Alibaba-NLP__gte-multilingual-base': 'GTE-Multilingual-Base',
    'jinaai__jina-embeddings-v2-small-en': 'Jina-Embeddings-V2-Small-En',
}

dimensions = [32, 64, 128, 256]

# Alternative loss function methods: (display_name, method_suffix)
baselines = {
    'Angular': 'batch_20000_poslossfactor_0',
    'Angular (W)': 'batch_20000_poslossfactor_0_weighted',
    'Spearman': 'batch_20000_spearman',
    'Spearman (W)': 'batch_20000_spearman_weighted',
    'Positional (Ours)': 'batch_20000_poslossfactor_1_linear',
}

def format_val(val):
    if pd.isna(val):
        return '-'
    return f'{val:.4f}'

print("\n" + "="*80)
print("EXTRINSIC TABLES: Alternative Loss Functions")
print("="*80 + "\n")

for backbone in backbones:
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
            model_name = f"{backbone}_distilled_{dim}_{method_suffix}"
            row = df[df['Model'] == model_name]

            if len(row) == 0:
                print(f"WARNING: Model not found: {model_name}")
                continue

            row_data = row.iloc[0]
            vals = []
            for col in extrinsic_cols:
                val = row_data[col]
                vals.append(format_val(val))

            if first_in_dim:
                print(f"\\multicolumn{{6}}{{l}}{{\\textbf{{Dim = {dim}}}}} \\\\")
                print("\\midrule")
                first_in_dim = False

            print(f"{method_name} & " + ' & '.join(vals) + ' \\\\')

        if dim != dimensions[-1]:
            print("\\midrule")

    print("\\bottomrule")
    print("\\end{longtable}")
    print("\\end{landscape}")
    print("\n")
