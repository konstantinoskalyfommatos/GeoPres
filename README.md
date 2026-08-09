# GeoPres
Official implementation of GeoPres.

# Formal Research Problem
Given a function $g: X \to Y \subseteq \mathbb{R}^n$, where $X$ is an arbitrary set and $n$ is large, and given $k < n$, we are interested in a function $f^*: Y \to \mathbb{R}^k$, with the property that downstream task performance achieved by $f^* \circ g$ closely matches that of $g$. With such a function, for any $x \in X$, the embedding $f^*(g(x)) \in \mathbb{R}^k$ serves as a low-dimensional representative of $g(x) \in \mathbb{R}^n$, significantly reducing computational cost without substantial loss in task performance.

**Note:** In the present context, $g$ is an embedding model, $X$ is a set of sentences, and $g$ maps these sentences into a vector space.

# Idea
Our approach is straightforward: we generate a large collection of high-dimensional vectors using $g$, which serve as training data to approximate $f^*$ with a linear model $f(y) = W y$, optimized via backpropagation using a distance-preserving loss function. This approach rests on two assumptions: first, that a linear function with the desired properties exists; and second, that preserving intrinsic properties for a large set of vectors in the image of $g$ is predictive of downstream task performance.

# Loss Function
We optimize $f$ by minimizing a pairwise distance preservation loss. Formally, let $\mathcal{B} = \{y_1, y_2, \dots, y_m\} \subseteq \mathbb{R}^n$ be a training batch. For $y_i, y_j \in \mathcal{B}$, let $d_{i,j}$ denote the Euclidean distance between $y_i$ and $y_j$, and let $d_{i,j}^{f}$ denote the Euclidean distance between $f(y_i)$ and $f(y_j)$. The loss is defined as

$$\mathcal{L} = \frac{1}{\binom{m}{2}} \sum_{i < j}\left( d_{i,j} - d_{i,j}^{f} \right)^2.$$

# Implementation
## Training Setup
- **Training data**: 10 million English-only text passages sampled from the Colossal Clean Crawled Corpus (C4).
- **Optimizer**: AdamW with learning rate $10^{-2}$ and weight decay $0.1$.
- **Batch size**: 20,000 points.
- **Scheduler**: Linear learning rate scheduler with warmup ratio of $0.1$.
- **Epochs**: 10, with early stopping (patience of 3 evaluation steps).
- **Validation**: 10,000 paraphrase pairs (20,000 data points) from `agentlans/sentence-paraphrases`.

# Prerequisites
```bash
uv lock
uv sync
```

Then, create a `.env` file in the project root directory according to the `.env.example` file.

## Environment Configuration

To ensure the project modules are accessible, set the `PYTHONPATH` to include the `src` directory:

```bash
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
```

For a persistent setup, add this to your shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`):

```bash
echo 'export PYTHONPATH="$(pwd)/src:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```

# Project Structure

```
src/
├── config.py                          # .env loader + path/dtype helpers (PROJECT_ROOT, STORAGE_PATH, …)
├── precalculate_embeddings.py         # one-time per backbone: encode C4 / sentence-paraphrases, dump .pt
├── train_model.py                     # main entry: trains the GeoPres linear projection (HF Trainer)
├── eval_backbone.py                   # MTEB eval of a raw backbone (no projection), for reference
├── eval_model.py                      # MTEB eval of a trained projection model from a checkpoint
├── eval_utils.py                      # shared MTEB runner + intrinsic metrics (angular/positional/Spearman)
├── geopres_trainer.py                 # HF Trainer subclass implementing the pairwise distance loss
├── reduced_sentence_transformer.py    # SentenceTransformer wrapper that applies the learned linear W
├── baselines/                         # alternative compression methods, all read the same precalculated .pt
│   ├── fit_pca.py                     # fit PCA on precalculated embeddings → projection matrix
│   ├── eval_pca.py                    # MTEB eval of PCA-projected embeddings
│   ├── eval_random_projection.py      # random linear projection baseline
│   ├── eval_random_selection.py       # random coordinate-selection baseline
│   ├── eval_truncation.py             # coordinate truncation baseline
│   ├── train_autoencoder.py           # train a nonlinear autoencoder baseline
│   └── eval_autoencoder.py            # MTEB eval of the trained autoencoder
└── scripts/                           # post-hoc analysis & reporting
    └── compare_evaluation_results.py  # the ONLY essential script here; merges MTEB caches into
                                       # $EVALUATION_RESULTS_PATH/comparison_results.csv
```

The pipeline is **precalculate → train → evaluate**:

1. `precalculate_embeddings.py` encodes the C4 training corpus and the
   `agentlans/sentence-paraphrases` validation/test splits with the chosen backbone
   and dumps tensors to `$STORAGE_PATH/precalculated_embeddings/`.
2. `train_model.py` (and `baselines/train_autoencoder.py` for the nonlinear baseline)
   train a projection on those cached embeddings. `baselines/eval_*.py` /
   `baselines/fit_pca.py` produce PCA / random-projection / random-selection /
   truncation results the same way.
3. `eval_model.py` and `eval_backbone.py` run each trained model / raw backbone on
   MTEB through `eval_utils.evaluate_mteb`, which caches results under
   `$EVALUATION_RESULTS_PATH/`.
4. `scripts/compare_evaluation_results.py` is the only required post-processing
   step — it walks those MTEB caches and writes a single
   `comparison_results.csv` (one row per backbone × method × task). All other
   files under `src/scripts/` are exploratory LaTeX-table / plotting generators
   and are not part of the core pipeline.

The end-to-end CLI recipes live in `AGENTS.md` under "Workflow: precalculate → train → evaluate".

# Common Issues

## torchsort fails to install or build

The default `uv lock && uv sync` installs `torchsort` from PyPI, which requires the CUDA toolchain to build from source. If the build fails (e.g., no CUDA runtime, unsupported platform), you can install a pre-built wheel manually from the [torchsort GitHub releases](https://github.com/teddykoker/torchsort/releases):

```bash
# torchsort version, supports >= 0.1.10
export TORCHSORT=0.1.10
# PyTorch version, supports pt26, pt25, pt24, pt21, pt20, and pt113 for versions
# 2.6, 2.5, 2.4, 2.1, 2.0, and 1.13 respectively
export TORCH=pt26
# CUDA version, supports cpu, cu113, cu117, cu118, cu121, cu124, and cu126 for
# CPU-only, CUDA 11.3, CUDA 11.7, CUDA 11.8, CUDA 12.1, CUDA 12.4, and CUDA 12.6
# respectively
export CUDA=cu126
# Python version, supports cp310, cp311, and cp312 for versions 3.10, 3.11, and
# 3.12 respectively
export PYTHON=cp312

uv pip install https://github.com/teddykoker/torchsort/releases/download/v${TORCHSORT}/torchsort-${TORCHSORT}+${TORCH}${CUDA}-${PYTHON}-${PYTHON}-linux_x86_64.whl
```

Adjust the variables above to match your PyTorch version, CUDA version, and Python version. Pre-built wheels are currently available on Linux only.

## Alibaba-NLP/gte-multilingual-base model breaks with newer Transformers versions

This project pins `transformers==4.57.6` for a reason. Upgrading to a more recent version causes issues with the mGTE model. If you override this pin, expect runtime errors when encoding with mGTE.
