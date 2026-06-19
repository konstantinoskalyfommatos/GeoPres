# GeoPres
Official implementation of GeoPres.

# Formal Research Problem
Given a function $g: X \rightarrow Y \subseteq \mathbb{R}^n$, where $X$ is an arbitrary set\footnote{In the present context, $X$ is a set of sentences, and $g$ maps these sentences into a vector space.} and $n$ is large, and given $k < n$, we are interested in a function $f^*: Y \rightarrow \mathbb{R}^k$, with the property that downstream task performance achieved by $f^* \circ g$ closely matches that of $g$. With such a function, for any $x \in X$, the embedding $f^*(g(x)) \in \mathbb{R}^k$ serves as a low-dimensional representative of $g(x) \in\mathbb{R}^n$, significantly reducing computational cost without substantial loss in task performance.

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
