# GPL (Geometric Preservation Learning)

This project aims to create a framework for dimensionality reduction of foundation embedding models.

# Formal Research Problem

Given a function $g: X \rightarrow Y \subseteq \mathbb{R}^n$, where $X$ is an arbitrary set and $n$ is large, we are interested in a function $f^*: Y \rightarrow \mathbb{R}^k$, $k < n$, with the property that downstream task performance achieved by $f^* \circ g$ closely matches that of $g$. With such a function, for any $x \in X$, the embedding $f^*(g(x)) \in \mathbb{R}^k$ serves as a low-dimensional representative of $g(x) \in\mathbb{R}^n$, significantly reducing computational cost without substantial loss in task performance.

# Idea

The idea is simple: we generate a large collection of high-dimensional vectors using $g$, which serve as training data to approximate $f^*$ with a neural network $f \approx f^*$, optimized via backpropagation. This approach rests on two assumptions: the existence of $f^*$, and that intrinsic property preservation of any large set of vectors in the image of $g$ under $f^*$ implies preserved downstream task performance.

# Model

Let $n$ and $k$ denote the source (high) and target (low) dimensionalities, 
respectively. We model $f$ as a single-layer network with a ReLU activation. Despite its simplicity and the use of ReLU as the output activation, this architecture proves sufficient for our purposes.

# Loss Function

We optimize $f$ by minimizing a pairwise distance preservation loss. Formally, let $Β = \{y_1, y_2, ... y_m\} \subseteq \mathbb{R}^n$ be a training batch. Then, given $y_i, y_j \in B$, we denote by $d_{i,j}^{\text\small{{h}}}$ and $d_{i,j}^{\text\small{{l}}}$ the Euclidean distances between $y_i, y_j$, and $f(y_i), f(y_j)$ respectively (h stands for high, l for low). The loss is defined as:

$$\mathcal{L} = \frac{1}{\binom{m}{2}} \sum_{i < j} \left( d_{i,j}^{\text\small{{h}}} - d_{i,j}^{\text\small{{l}}} \right)^2.$$

# Implementation

## Pipeline

1. Precompute a large number of embeddings (output vectors in $\mathbb{R}^n$) using the backbone embedding model $g$.
2. Define a single-layer neural network with ReLU activation that maps these vectors into a low-dimensional space $\mathbb{R}^k$.
3. Train the network to preserve pairwise Euclidean distances using the loss function above.

## Training Setup

- **Training data**: 10 million English-only text passages sampled from the Colossal Clean Crawled Corpus (C4).
- **Optimizer**: AdamW with learning rate $10^{-2}$ and weight decay $0.1$.
- **Batch size**: 20,000 points.
- **Scheduler**: Linear learning rate scheduler with warmup ratio of $0.1$.
- **Epochs**: 10, with early stopping (patience of 3 evaluation steps).
- **Validation**: 10,000 paraphrase pairs (20,000 data points) from `agentlans/sentence-paraphrases`.

# Prerequisites

To set up the project, create a uv environment at the project root level using the following commands:
```bash
uv venv .venv --python 3.12
source .venv/bin/activate
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
