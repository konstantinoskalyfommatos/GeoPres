import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests

# 1. Load Data and Define Configurations
df = pd.read_csv('storage/evaluation_results/comparison_results_filtered.csv')

base_backbones = [
    "Alibaba-NLP__gte-multilingual-base",
    "Qwen__Qwen3-Embedding-0.6B",
    "jinaai__jina-embeddings-v2-small-en",
    "sentence-transformers__all-mpnet-base-v2"
]

dimensions = [32, 64, 128, 256, 512]
baselines = ["autoencoder", "pca", "random_projection", "random_selection", "truncation"]
geopres_method = "batch_20000_geopres"

task_groups = {
    'STS': ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICK-R'],
    'Retrieval': ['QuoraRetrieval', 'HotpotQA', 'DBPedia', 'NQ', 'MSMARCO', 'ArguAna'],
    'Classification': ['AmazonCounterfactualClassification', 'AmazonPolarityClassification', 
                       'AmazonReviewsClassification', 'ImdbClassification', 'ToxicConversationsClassification'],
    'Clustering': ['ArxivClusteringS2S', 'RedditClustering', 'StackExchangeClustering']
}

# 2. Bootstrap Function (Returns P-value)
def paired_bootstrap_pvalue(scores_A, scores_B, n_bootstraps=10000):
    n = len(scores_A)
    # Vectorized bootstrap for speed
    indices = np.random.randint(0, n, size=(n_bootstraps, n))
    boot_A = scores_A[indices].mean(axis=1)
    boot_B = scores_B[indices].mean(axis=1)
    diffs = boot_A - boot_B
    
    # Two-tailed p-value: proportion of bootstrap samples crossing 0
    p_val = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return max(p_val, 1/n_bootstraps) # Avoid exact 0

# 3. Run Experiments
results = []
print("Running Bootstrap Tests... (This may take a minute)")

for backbone in base_backbones:
    for dim in dimensions:
        # Skip 512D for non-Qwen models
        if dim == 512 and "Qwen" not in backbone:
            continue
            
        geopres_model = f"{backbone}_reduced_{dim}_{geopres_method}"
        
        for task, datasets in task_groups.items():
            for baseline in baselines:
                baseline_model = f"{backbone}_reduced_{dim}_{baseline}"
                
                if geopres_model not in df['Model'].values or baseline_model not in df['Model'].values:
                    continue
                
                row_A = df[df['Model'] == geopres_model].iloc[0]
                row_B = df[df['Model'] == baseline_model].iloc[0]
                
                # Filter to datasets where BOTH methods have valid (non-NaN) scores
                valid_datasets = [d for d in datasets if not pd.isna(row_A[d]) and not pd.isna(row_B[d])]
                
                if len(valid_datasets) < 2:
                    continue
                    
                scores_A = row_A[valid_datasets].values.astype(float)
                scores_B = row_B[valid_datasets].values.astype(float)
                
                mean_diff = np.mean(scores_A) - np.mean(scores_B)
                p_val = paired_bootstrap_pvalue(scores_A, scores_B)
                
                results.append({
                    'Backbone': backbone.split('__')[-1], # Shorten name for readability
                    'Dim': dim,
                    'Task': task,
                    'Baseline': baseline,
                    'Mean_Diff_GeoPres_Win': mean_diff,
                    'P_Value': p_val
                })

res_df = pd.DataFrame(results)

# 4. Multiple Comparisons Correction (Benjamini-Hochberg FDR)
print("Applying Benjamini-Hochberg FDR Correction...")
reject, pvals_corrected, _, _ = multipletests(res_df['P_Value'], alpha=0.05, method='fdr_bh')
res_df['P_Value_FDR'] = pvals_corrected
res_df['Significant_FDR'] = reject

# Determine direction of significance
res_df['Significance_Direction'] = 'Non-Significant'
res_df.loc[(res_df['Significant_FDR']) & (res_df['Mean_Diff_GeoPres_Win'] > 0), 'Significance_Direction'] = 'GeoPres Wins'
res_df.loc[(res_df['Significant_FDR']) & (res_df['Mean_Diff_GeoPres_Win'] < 0), 'Significance_Direction'] = 'Baseline Wins'

# 5. Save and Print Summary
res_df.to_csv('storage/evaluation_results/significance_results.csv', index=False)
print("\n--- SUMMARY OF STRICTLY SIGNIFICANT RESULTS (FDR < 0.05) ---")
summary = res_df[res_df['Significant_FDR']].groupby(['Baseline', 'Significance_Direction']).size().unstack(fill_value=0)
print(summary)

print("\n--- DETAILED BREAKDOWN BY BASELINE ---")
for baseline in baselines:
    subset = res_df[res_df['Baseline'] == baseline]
    wins = subset[subset['Significance_Direction'] == 'GeoPres Wins'].shape[0]
    losses = subset[subset['Significance_Direction'] == 'Baseline Wins'].shape[0]
    total = subset.shape[0]
    print(f"GeoPres vs {baseline:20} | GeoPres Wins: {wins:3} | Baseline Wins: {losses:3} | Total Tests: {total}")