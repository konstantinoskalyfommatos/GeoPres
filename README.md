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
