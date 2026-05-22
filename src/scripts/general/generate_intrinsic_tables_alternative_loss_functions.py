import pandas as pd

df = pd.read_csv('storage/evaluation_results/comparison_results.csv')
df['Model'] = df['Model'].str.replace('/', '__', regex=False)

intrinsic_cols = ['spearman_loss', 'spearman_loss_weighted', 'angular_loss',
                  'angular_loss_weighted', 'positional_loss', 'positional_loss_weighted']

backbones = [
    'Alibaba-NLP__gte-multilingual-base',
    'jinaai__jina-embeddings-v2-small-en'
]

backbone_display = {
    'Alibaba-NLP__gte-multilingual-base': 'GTE-Multilingual-Base',
    'jinaai__jina-embeddings-v2-small-en': 'Jina-Embeddings-V2-Small-En'
}

dimensions = [32, 64, 128, 256]

# Alternative loss function methods: (display_name, method_suffix)
# Note: poslossfactor_0 = angular loss, poslossfactor_1 = positional (main) loss
alternative_methods = {
    'GeoPres (Angular)': 'batch_20000_poslossfactor_0',
    'GeoPres (Angular W)': 'batch_20000_poslossfactor_0_weighted',
    'GeoPres (Spearman)': 'batch_20000_spearman',
    'GeoPres (Spearman W)': 'batch_20000_spearman_weighted',
    'GeoPres (Ours)': 'batch_20000_poslossfactor_1_linear'
}

for backbone in backbones:
    print(f"\n\\subsubsection{{{backbone_display[backbone]}}}")
    print("\\begin{landscape}")
    print("\\begin{longtable}{lcccccc}")
    print("\\toprule")

    col_labels = ['Method', 'Spearman', 'Spearman (W)', 'Angular', 'Angular (W)', 'Positional', 'Positional (W)']
    print(' & '.join(col_labels) + ' \\\\')
    print('\\midrule')
    print('\\midrule')

    for dim in dimensions:
        first_in_dim = True

        for method_name, method_suffix in alternative_methods.items():
            model_name = f"{backbone}_distilled_{dim}_{method_suffix}"
            row = df[df['Model'] == model_name]

            if len(row) == 0:
                continue

            row_data = row.iloc[0]
            vals = []
            for col in intrinsic_cols:
                val = row_data[col]
                if pd.isna(val):
                    vals.append('-')
                else:
                    vals.append(f'{val:.4f}')

            if first_in_dim:
                print(f"\\multicolumn{{7}}{{l}}{{\\textbf{{Dim = {dim}}}}} \\\\")
                print("\\midrule")
                first_in_dim = False

            print(f"{method_name} & " + ' & '.join(vals) + ' \\\\')

        if dim != dimensions[-1]:
            print("\\midrule")

    print("\\bottomrule")
    print("\\end{longtable}")
    print("\\end{landscape}")
