from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from dec_model import BATCH_SIZE, DEVICE, encode_dataset


def plot_tsne(
    Z: np.ndarray,
    labels: np.ndarray | None = None,
    title: str = "t-SNE of Latent Space",
) -> None:
    """Plot a 2D t-SNE view of latent representations."""
    print("\nRunning t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, init="random", random_state=42)
    Z_2d = tsne.fit_transform(Z)

    print("t-SNE completed. Plotting...")

    plt.figure(figsize=(8, 6))
    if labels is not None:
        plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c=labels, cmap="tab10", s=5, alpha=0.7)
        plt.colorbar()
    else:
        plt.scatter(Z_2d[:, 0], Z_2d[:, 1], s=5, alpha=0.7)

    plt.title(title)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_final_tsne(
    encoder: nn.Module,
    X: np.ndarray,
    title: str = "Final DEC Embeddings (t-SNE)",
    batch_size: int = BATCH_SIZE,
    device: torch.device | str = DEVICE,
) -> np.ndarray:
    """
    Encode X with the fine-tuned encoder, plot t-SNE, and return Z.

    Returning Z keeps the notebook diagnostic-friendly: the same final latent
    matrix can be reused for follow-up checks without recomputing it.
    """
    Z = encode_dataset(encoder, X, batch_size=batch_size, device=device)

    print("Running t-SNE on final embeddings...")
    Z_tsne = TSNE(n_components=2, perplexity=30, init="random", random_state=42).fit_transform(Z)

    plt.figure(figsize=(8, 6))
    plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=4, alpha=0.7)
    plt.title(title)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True)
    plt.show()

    return Z


__all__ = [
    "plot_final_tsne",
    "plot_tsne",
]
