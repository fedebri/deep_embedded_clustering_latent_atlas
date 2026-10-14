# Deep Embedded Clustering and Latent Space Diagnostics

An implementation and explanation of **Deep Embedded Clustering (DEC)** in
PyTorch, based on [Xie et al. (2016)](https://proceedings.mlr.press/v48/xieb16.html),
with supplementary diagnostics for the learned latent space.

The primary objective is structural replication of the paper's model and learning
procedure. Experimental accuracy is secondary. The notebooks explain the
mechanics, shapes, diagnostics, and common failure modes step by step.

The implementation uses the paper's architecture, momentum SGD and
iteration-based scheduling. The first two notebooks use short training budgets
for local demonstrations. The optional MNIST notebook uses the full pretraining
budget and includes a saved execution with **87.23% matched clustering accuracy**.
That is a single transductive run, not a held-out classification result or an
exact replay of the authors' Caffe implementation.
See [paper correspondence and remaining differences](docs/dec_paper_alignment.md).

## Contents

- [DEC algorithm walkthrough](notebooks/01_dec_algorithm_walkthrough.ipynb)
  Walks through DEC from autoencoder pretraining to cluster initialization, soft assignments, target distribution, KL-divergence training, and a classic digits dataset example.

- [Latent-space diagnostics](notebooks/02_dec_latent_space_diagnostics.ipynb)
  Focuses on the learned latent space: latent feature tables, cluster-separation scores, centroid profiles, and ablation tests using selected latent dimensions.

- [MNIST experiment](notebooks/03_dec_mnist_paper_experiment.ipynb)
  Full-budget pretraining, DEC refinement and evaluation on 70,000 MNIST images.

- `src/dec_model.py`
  Encoder models, centroid initialization, Student-t assignments, target distribution and KL loss.

- `src/dec_pretraining.py` and `src/dec_training.py`
  Shared denoising pretraining and joint encoder/centroid optimization. Experimental choices remain visible in the notebooks.

- `src/dec_visualization.py` and `src/latent_space_tools.py`
  Projection plots, digits preprocessing, clustering evaluation, latent-dimension scoring, cluster profiles and ablation tests.

- `tests/`
  Equation and gradient checks, synthetic training checks, deterministic helper tests and input validation.

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

1. Run `notebooks/01_dec_algorithm_walkthrough.ipynb` for the DEC algorithm walkthrough.
2. Run `notebooks/02_dec_latent_space_diagnostics.ipynb` for supplementary latent-space diagnostics.
3. Optionally run `notebooks/03_dec_mnist_paper_experiment.ipynb` for the MNIST experiment. It downloads data on execution and uses a much larger training budget.

The latent-space notebook is self-contained and can be run independently.
The two tutorial notebooks have cleared outputs: their earlier saved results
predated the reconstruction-loss correction. The MNIST notebook retains its
completed execution with the corrected loss.

## Tests

Run the lightweight test suite with:

```bash
python -m pytest -q
```

For equation checks without any dataset loading or training:

```bash
python -m pytest -q tests/test_dec_paper_equations.py
```

## Notes

- The digits example uses `sklearn.datasets.load_digits`, so it runs offline.
- UMAP is optional. The notebooks use PCA by default for stable visualization because UMAP can crash some notebook kernels through native OpenMP/numba interactions.
- Downloaded data, generated weights and runtime artifacts are ignored. The walkthrough saves its encoder under `outputs/`.
- Earlier development files are preserved locally in the ignored `archive/` folder. They are excluded from the published repository and test discovery.

## License

This project is released under the MIT License. See `LICENSE`.
