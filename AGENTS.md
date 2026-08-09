# AGENTS.md — GeoPres

GeoPres trains a linear projection that compresses high-dimensional sentence embeddings while preserving pairwise Euclidean and cosine distances, then evaluates the projection on the MTEB benchmark. Single-author thesis research code — no test suite, no CI.

## Quick start

```bash
uv sync
cp .env.example .env   # edit paths to absolute values; PROJECT_ROOT is required
export PYTHONPATH="$(pwd)/geopres:$PYTHONPATH"
```

All scripts live in `geopres/` (not `src/`). They do bare imports (`from config import …`), so `PYTHONPATH` must include `geopres/`. There is no `__init__.py` — `geopres/` is a directory on `sys.path`, not a package. `pyproject.toml` sets `[tool.uv] package = false`.

## Workflow: precalculate → train → evaluate

Training reads pre-computed embeddings from disk; it will not encode text on its own.

1. **Precalculate embeddings** (one-time per backbone; slow, GPU-heavy):
   ```bash
   python geopres/precalculate_embeddings.py --model_name <hf-id> --train --val_test \
       --batch_size 8192 --max_train_examples 10000000
   ```
   Writes to `$STORAGE_PATH/precalculated_embeddings/{c4|sentence-paraphrases}/<backbone>/{train,validation,test}_embeddings.pt`. Streams C4, chunks every `--encode_every` (default 100k), saves intermediates, then concatenates and deletes them.

2. **Train the projection**:
   ```bash
   python geopres/train_model.py \
       --backbone_model jinaai/jina-embeddings-v2-small-en \
       --source_dim 512 --target_dim 32 \
       --train_batch_size 20000 --val_batch_size 20000 \
       --positional_loss_factor 1 --weighted_loss
   ```
   Defaults: AdamW lr=1e-2, weight_decay=0.1, 10 epochs, early stopping patience 3, linear schedule with 0.1 warmup. `--spearman` swaps positional+angular loss for differentiable Spearman loss. `--positional_loss_factor < 1` blends angular loss in. `--resume_from_checkpoint` resumes from the highest numbered `checkpoint-N` in the output dir.

3. **Baselines** (autoencoder, PCA, random projection, random selection, truncation) use the same precalculated embeddings:
   ```bash
   python geopres/baselines/train_autoencoder.py --backbone_model … --source_dim … --target_dim …
   python geopres/baselines/fit_pca.py …
   python geopres/baselines/eval_*.py …
   ```

4. **Collect results** into one CSV:
   ```bash
   python geopres/scripts/compare_evaluation_results.py
   ```
   The only essential script under `geopres/scripts/`. Others are exploratory LaTeX-table / plotting generators.

## Pinned / fragile dependencies

- `transformers==4.57.6` — upgrading breaks `Alibaba-NLP/gte-multilingual-base` (mGTE) at encode time. Do not bump without retesting.
- `torchsort` — builds from source, needs CUDA toolchain. If `uv sync` fails, install a pre-built wheel from [teddykoker/torchsort releases](https://github.com/teddykoker/torchsort/releases) (see `README.md` → "Common Issues").
- `sentence-transformers==5.3.0`, `mteb`, `safetensors`, `flash-attn>=2.5.0` (optional) are other notable pins.

## Environment (`.env`)

`geopres/config.py` calls `load_dotenv(override=True)` and raises immediately if `PROJECT_ROOT` is missing. `HF_TOKEN` is used by `eval_utils.evaluate_mteb` for gated datasets. Other vars (`STORAGE_PATH`, `EVALUATION_RESULTS_PATH`, `TRAINED_MODELS_PATH`, `TRAINED_AUTOENCODERS_PATH`) default under `$PROJECT_ROOT/storage/` but are usually overridden in `.env`.

## Storage layout

All artefacts under `$STORAGE_PATH` (default `$PROJECT_ROOT/storage`):

- `precalculated_embeddings/{c4,sentence-paraphrases}/<backbone>/{train,validation,test}_embeddings.pt`
- `trained_models/<backbone>/<model_name>/checkpoint-N/`
- `trained_autoencoders/<backbone>/<model_name>/checkpoint-N/`
- `evaluation_results/{trained_models,autoencoders}/<backbone>/<model_name>/results/<task>/…`

Model names embed backbone, target dim, batch size, and method suffix — e.g. `jinaai__jina-embeddings-v2-small-en_reduced_32_batch_20000_poslossfactor_1`. Adding a new suffix requires updating `_get_extrinsic_model_names` in `geopres/scripts/calculate_intrinsic_correlations.py` and likely `generate_extrinsic_tables_raw_mteb.py`.

## CUDA / hardware assumptions

`train_model.py`, `train_autoencoder.py`, `precalculate_embeddings.py`, `eval_backbone.py` all hardcode `device="cuda"`. No CPU fallback. Default training batch is 20 000 embeddings per step. MTEB retrieval uses `retrieval_batch_size=6` (higher values OOM).

## MTEB evaluation

`evaluate_mteb` in `geopres/eval_utils.py` runs four groups (STS, classification, clustering, retrieval) with `cache=ResultCache`. By default skips cached tasks (`overwrite_strategy="only-missing"`); pass `--overwrite_cache` to force reruns. Intrinsic metrics (angular / positional / Spearman loss) are written alongside MTEB results as `intrinsic.json`.

## Conventions

- `torch.manual_seed(42)` at module top of `eval_utils.py` and `geopres_trainer.py`.
- `TOKENIZERS_PARALLELISM=false` set in `train_model.py` / `train_autoencoder.py`.
- `HF_HUB_DOWNLOAD_TIMEOUT=300` set in `precalculate_embeddings.py`.
- Long jobs run detached (`nohup.out` and `trainer_output/` are gitignored).

## What not to do

- Don't bump `transformers` or `sentence-transformers` without checking mGTE compatibility.
- Don't add a `tests/` directory — there is no CI and no test suite.
- Don't "fix" stale `pyproject.toml` entrypoints without confirming; scripts are run directly with `python`.
- Don't `pip install` globally — everything goes through `uv`.
- Don't assume CPU works — gate work on `torch.cuda.is_available()` or warn before launching training.
