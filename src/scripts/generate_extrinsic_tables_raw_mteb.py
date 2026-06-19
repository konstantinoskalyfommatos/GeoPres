"""
Generate LaTeX tables with raw task scores for each backbone model.
Produces output identical to tables.tex.

Usage:
    python src/scripts/generate_extrinsic_tables_raw_mteb.py

Output:
    Prints LaTeX table code to stdout.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

RESULTS_PATH = 'storage/evaluation_results/comparison_results.csv'

# Task columns in order as they appear in the LaTeX table
TASK_COLUMNS = [
    'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    'STSBenchmark', 'SICK-R',
    'QuoraRetrieval', 'HotpotQA', 'DBPedia', 'NQ', 'MSMARCO', 'ArguAna',
    'AmazonCounterfactualClassification', 'AmazonPolarityClassification',
    'AmazonReviewsClassification', 'ImdbClassification',
    'ToxicConversationsClassification',
    'ArxivClusteringS2S', 'RedditClustering', 'StackExchangeClustering'
]

# Backbones and their original dimensions
# Each entry: (model_name, original_dim, display_name, label, note, dim_groups)
BACKBONES = [
    ('Alibaba-NLP__gte-multilingual-base', 768,
     r'\texttt{Alibaba-NLP/gte-multilingual-base}',
     'tab:raw_mteb_scores_1',
     'The autoencoder failed to converge at a target dimension of 2.',
     [256, 128, 64, 32, 2]),
    ('Qwen__Qwen3-Embedding-0.6B', 1024,
     r'\texttt{Qwen/Qwen3-Embedding-0.6B}',
     'tab:raw_mteb_scores_2',
     '',
     [512, 256, 128, 64, 32, 2]),
    ('jinaai__jina-embeddings-v2-small-en', 512,
     r'\texttt{jinaai/jina-embeddings-v2-small-en}',
     'tab:raw_mteb_scores_3',
     'The autoencoder failed to converge at a target dimension of 2.',
     [256, 128, 64, 32, 2]),
    ('sentence-transformers__all-mpnet-base-v2', 768,
     r'\texttt{sentence-transformers/all-mpnet-base-v2}',
     'tab:raw_mteb_scores_4',
     '',
     [256, 128, 64, 32, 2]),
]

# Method suffixes in the model name
METHODS = [
    ('PCA', '_pca'),
    ('RP', '_random_projection'),
    ('RS', '_random_selection'),
    ('Trunc', '_truncation'),
    ('AE', '_autoencoder'),
    (r'\method', '_batch_20000_poslossfactor_1'),
]

# Headers matching the LaTeX table
HEADERS = [
    'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    'STSBenchmark', 'SICK-R', 'QuoraRetrieval', 'HotpotQA', 'DBPedia',
    'NQ', 'MSMARCO', 'ArguAna',
    'AmazonCounterfactualClassification', 'AmazonPolarityClassification',
    'AmazonReviewsClassification', 'ImdbClassification',
    'ToxicConversationsClassification',
    'ArxivClusteringS2S', 'RedditClustering', 'StackExchangeClustering'
]


def format_val(val, decimals=4):
    """Format a value for LaTeX output."""
    if pd.isna(val) or val == '' or val is None:
        return '--'
    try:
        v = float(val)
        if np.isnan(v):
            return '--'
        return f'{v:.{decimals}f}'
    except (ValueError, TypeError):
        return '--'


def all_sts_nan(row_data):
    """Check if all STS columns (first 7 task columns) are NaN."""
    sts_cols = TASK_COLUMNS[:7]
    for col in sts_cols:
        val = row_data.get(col, None)
        if pd.isna(val) or val == '' or val is None:
            continue
        try:
            v = float(val)
            if not np.isnan(v):
                return False
        except (ValueError, TypeError):
            continue
    return True


def generate_table(df, backbone_name, original_dim, model_display, label, note, dim_groups):
    """Generate a single LaTeX table for a backbone model, matching tables.tex."""
    # Filter base model
    base_row = df[df['Model'] == backbone_name]

    if len(base_row) == 0:
        print(f"WARNING: Base model not found: {backbone_name}", file=sys.stderr)
        return

    base_data = base_row.iloc[0]

    lines = []

    # Table header
    lines.append('        \\begin{table}[H]')
    lines.append('        \\centering')
    lines.append('        \\tiny')
    lines.append('        \\setlength{\\tabcolsep}{2.0pt}')
    lines.append('        \\renewcommand{\\arraystretch}{0.9}')
    lines.append('        \\resizebox{\\textwidth}{!}{%')
    lines.append('        \\begin{tabular}{@{}c >{\\fontsize{5}{5}\\selectfont}l *{21}{>{\\centering\\arraybackslash}p{3em}}@{}}')

    # Column header row (single row, matching thesis.tex output)
    all_headers = ' & '.join(f'\\diaghead{{{h}}}' for h in HEADERS)
    lines.append(f'        \\textbf{{Dim}}\\rule{{0pt}}{{4.0em}} & \\textbf{{Method}} & {all_headers} \\\\')

    lines.append('        \\hline\\hline')
    lines.append('        \\noalign{\\vskip 0.4ex}')

    # Base model row (original dimension)
    base_vals = [format_val(base_data[col], 3) for col in TASK_COLUMNS]
    base_row_str = '        \\textbf{' + str(original_dim) + '} & \\methodpad{--} & ' + ' & '.join(base_vals) + ' \\\\'
    lines.append(base_row_str)
    lines.append('        \\hline')
    lines.append('        \\noalign{\\vskip 0.4ex}')

    # For each dimension group
    for dim_idx, dim in enumerate(dim_groups):
        # First pass: collect all method data for this dimension
        method_data = []  # list of (method_label, raw_values_list)
        for method_label, method_suffix in METHODS:
            model_name = f'{backbone_name}_reduced_{dim}{method_suffix}'
            row = df[df['Model'] == model_name]

            if len(row) == 0:
                # Some methods may not exist at certain dims (e.g., AE at dim 2)
                continue

            row_data = row.iloc[0]

            if all_sts_nan(row_data):
                # All NaN row – treat as None for every column
                raw_vals = [None] * len(TASK_COLUMNS)
            else:
                raw_vals = [row_data[col] for col in TASK_COLUMNS]

            method_data.append((method_label, raw_vals))

        if not method_data:
            continue

        # First, convert all values to their 3-decimal representation for fair comparison.
        # We store the rounded float (to 3 decimals) alongside each raw value.
        num_cols = len(TASK_COLUMNS)
        # rounded_vals[method_idx][col_idx] = (rounded_float or None, display_string)
        rounded_data = []
        for midx, (method_label, raw_vals) in enumerate(method_data):
            row_rounded = []
            for col_idx, v in enumerate(raw_vals):
                if v is None or pd.isna(v):
                    row_rounded.append((None, '--'))
                else:
                    try:
                        fv = float(v)
                        if np.isnan(fv):
                            row_rounded.append((None, '--'))
                        else:
                            rv = round(fv, 3)
                            s = f'{rv:.3f}'
                            row_rounded.append((rv, s))
                    except (ValueError, TypeError):
                        row_rounded.append((None, '--'))
            rounded_data.append((method_label, row_rounded))

        # Determine the winner(s) per column based on the rounded (3-decimal) value.
        # All methods sharing the maximum rounded value are bolded (ties).
        winners = {}  # col_idx -> set of method indices
        for col_idx in range(num_cols):
            best_val = -float('inf')
            best_midxs = []
            for midx, (_, row_rounded) in enumerate(rounded_data):
                rv, _ = row_rounded[col_idx]
                if rv is not None and rv > best_val:
                    best_val = rv
                    best_midxs = [midx]
                elif rv is not None and rv == best_val:
                    best_midxs.append(midx)
            if best_midxs:
                winners[col_idx] = set(best_midxs)

        # Second pass: render with winners bolded
        for midx, (method_label, row_rounded) in enumerate(rounded_data):
            formatted = []
            for col_idx, (rv, s) in enumerate(row_rounded):
                if s == '--':
                    formatted.append(s)
                else:
                    if col_idx in winners and midx in winners[col_idx]:
                        formatted.append(f'\\textbf{{{s}}}')
                    else:
                        formatted.append(s)

            if midx == 0:
                lines.append(f'        \\dimblock{{{dim}}} & \\methodpad{{{method_label}}}     & ' + ' & '.join(formatted) + ' \\\\')
            else:
                lines.append(f'         & \\methodpad{{{method_label}}}      & ' + ' & '.join(formatted) + ' \\\\')

        if dim_idx < len(dim_groups) - 1:
            lines.append('        \\hline')
            lines.append('        \\noalign{\\vskip 0.4ex}')

    lines.append('        \\end{tabular}%')
    lines.append('        }')

    # Caption
    if note:
        lines.append(f'        \\caption{{Raw task scores for {model_display}.')
        lines.append(f'          {note}}}')
    else:
        lines.append(f'        \\caption{{Raw task scores for {model_display}.}}')
    lines.append(f'        \\label{{{label}}}')
    lines.append('        \\end{table}')
    lines.append('        \\vspace*{\\fill}')

    return '\n'.join(lines)


def main():
    df = pd.read_csv(RESULTS_PATH)

    tables = []
    for backbone_name, original_dim, model_display, label, note, dim_groups in BACKBONES:
        table = generate_table(df, backbone_name, original_dim, model_display, label, note, dim_groups)
        tables.append(table)

    print('\n\n'.join(tables))


if __name__ == '__main__':
    main()
