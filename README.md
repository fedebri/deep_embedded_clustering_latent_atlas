# Deep Embedded Clustering and Latent Space Diagnostics

This project is a notebook-based implementation and explanation of **Deep Embedded Clustering (DEC)**, based on Xie et al. (2016), with a second notebook focused on interpreting the learned latent space.

The notebooks are written as a step-by-step learning artifact: the markdown intentionally explains the mechanics, shapes, diagnostics, and common failure modes in detail.

## Contents

- `01_dec_algorithm_walkthrough.ipynb`  
  Walks through DEC from autoencoder pretraining to cluster initialization, soft assignments, target distribution, KL-divergence training, and a classic digits dataset example.

- `02_dec_latent_space_diagnostics.ipynb`  
  Focuses on the learned latent space: latent feature tables, cluster-separation scores, centroid profiles, and ablation tests using selected latent dimensions.

- `dec_model.py`  
  Reusable PyTorch implementation of the DEC components used by both notebooks.

- `dec_visualization.py`
  Visualization helpers for t-SNE diagnostics, kept separate from the modeling code.

- `latent_space_tools.py`  
  Helper methods for the latent-space notebook: digits preprocessing, clustering evaluation, latent-dimension scoring, cluster profiles, and ablation tests.

- `tests/`
  Minimal tests for deterministic helper logic and DEC input validation.

- `environment.yml`  
  Conda environment definition for reproducing the notebooks.

## Setup

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate dec-latent-space
python -m ipykernel install --user --name dec-latent-space --display-name "DEC latent space"
```

Then open the notebooks in JupyterLab:

```bash
jupyter lab
```

## Notebook Order

1. Run `01_dec_algorithm_walkthrough.ipynb` first if you want the full DEC algorithm walkthrough.
2. Run `02_dec_latent_space_diagnostics.ipynb` for the portfolio-ready latent-space interpretation workflow.

The latent-space notebook is self-contained and can be run independently.

## Tests

Run the lightweight test suite with:

```bash
pytest
```

## Privacy and Repository Hygiene

The `.gitignore` is intentionally conservative. It excludes local model checkpoints, notebook checkpoints, raw/preprocessed paper extraction artifacts, PDFs, caches, local environment files, and credentials.

Before pushing to GitHub, check what will be committed:

```bash
git status --short
git add --dry-run .
```

If you want notebooks to render with outputs on GitHub, review the outputs first to make sure they do not contain private paths, data, or large embedded images.

## Notes

- The digits example uses `sklearn.datasets.load_digits`, so it runs offline.
- UMAP is optional. The notebooks use PCA by default for stable visualization because UMAP can crash some notebook kernels through native OpenMP/numba interactions.
- The paper PDF and extracted paper assets are treated as local research materials and are ignored by default.

## License

This project is released under the MIT License. See `LICENSE`.
