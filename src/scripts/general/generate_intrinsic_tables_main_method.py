import pandas as pd

df = pd.read_csv('storage/evaluation_results/comparison_results.csv')
df['Model'] = df['Model'].str.replace('/', '__', regex=False)

intrinsic_cols = ['spearman_loss', 'spearman_loss_weighted', 'angular_loss', 
                  'angular_loss_weighted', 'positional_loss', 'positional_loss_weighted']

backbones = [
    'Alibaba-NLP__gte-multilingual-base',
    'Qwen__Qwen3-Embedding-0.6B',
    'jinaai__jina-embeddings-v2-small-en',
    'sentence-transformers__all-mpnet-base-v2'
]

backbone_display = {
    'Alibaba-NLP__gte-multilingual-base': 'GTE-Multilingual-Base',
    'Qwen__Qwen3-Embedding-0.6B': 'Qwen3-Embedding-0.6B',
    'jinaai__jina-embeddings-v2-small-en': 'Jina-Embeddings-V2-Small-En',
    'sentence-transformers__all-mpnet-base-v2': 'All-MPNet-Base-V2'
}

dimensions = [32, 64, 128, 256]

baselines = {
    'PCA': 'pca',
    'Random Projection': 'random_projection',
    'Random Selection': 'random_selection',
    'Truncation': 'truncation',
    'Autoencoder': 'autoencoder',
    'ALDRL (Ours)': 'batch_20000_poslossfactor_1'
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
        
        for method_name, method_suffix in baselines.items():
            model_name = f"{backbone}_distilled_{dim}_{method_suffix}"
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
