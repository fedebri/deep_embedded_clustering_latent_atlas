"""Shared encoder models, initialization and DEC probability equations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterable

# Keep native numerical libraries conservative by default. This matters in
# notebooks where PyTorch, scikit-learn, numba, and UMAP can otherwise load
# competing OpenMP runtimes and hard-crash the kernel.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256


def _as_2d_finite_array(name: str, value: np.ndarray) -> np.ndarray:
    """Validate an array that represents samples by features."""
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; received shape {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(
            f"{name} must be non-empty in both dimensions; received shape {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numeric values.")
    return array


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; received {value!r}.")


def _validate_positive_float(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite value; received {value!r}.")


class Encoder(nn.Module):
    """
    Deep encoder used by DEC.

    Important detail: the activation functions live in this forward method.
    Returning a plain nn.Sequential of Linear layers would silently skip the
    nonlinear transformations learned during pretraining.
    """

    def __init__(
        self,
        input_dim: int,
        layer_dims: Iterable[int],
        activate_final: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.layer_dims = [int(dim) for dim in layer_dims]
        self.activate_final = activate_final
        self.dims = [self.input_dim, *self.layer_dims]

        if len(self.layer_dims) == 0:
            raise ValueError("layer_dims must contain at least one hidden/latent dimension.")

        self.layers = nn.ModuleList(
            nn.Linear(self.dims[i], self.dims[i + 1]) for i in range(len(self.dims) - 1)
        )

    @classmethod
    def from_linear_layers(
        cls,
        input_dim: int,
        layers: Iterable[nn.Linear],
        activate_final: bool = False,
    ) -> "Encoder":
        layers = list(layers)
        model = cls(input_dim, [layer.out_features for layer in layers], activate_final=activate_final)
        for target_layer, source_layer in zip(model.layers, layers):
            target_layer.load_state_dict(source_layer.state_dict())
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_index, layer in enumerate(self.layers):
            x = layer(x)

            is_final_layer = layer_index == len(self.layers) - 1
            if self.activate_final or not is_final_layer:
                x = F.relu(x)

        return x


class FullAutoencoder(nn.Module):
    """
    Full autoencoder used only for reconstruction fine-tuning.

    DEC keeps the encoder after this step and discards the decoder before the KL
    clustering phase.
    """

    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = encoder

        # Build decoder by reversing encoder layer sizes:
        # 784 -> 500 -> 500 -> 2000 -> 10 becomes 10 -> 2000 -> 500 -> 500 -> 784.
        dims = encoder.dims
        self.decoder_layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i - 1]) for i in range(len(dims) - 1, 0, -1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        for layer_index, layer in enumerate(self.decoder_layers):
            out = layer(out)

            is_final_reconstruction_layer = layer_index == len(self.decoder_layers) - 1
            if not is_final_reconstruction_layer:
                out = F.relu(out)

        return out


@dataclass
class DECTrainingResult:
    """Final DEC state and the outcome of the last target-refresh check."""

    encoder: nn.Module
    centroids: np.ndarray
    labels: np.ndarray
    loss_history: list[float]
    converged: bool
    label_change: float | None


def encode_dataset(
    encoder: nn.Module,
    X: np.ndarray,
    batch_size: int = BATCH_SIZE,
    device: torch.device | str = DEVICE,
) -> np.ndarray:
    """Transforms raw data X into latent features Z in batches."""
    X = _as_2d_finite_array("X", X)
    _validate_positive_int("batch_size", batch_size)

    device = torch.device(device)
    encoder.eval()

    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    dataset = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
    encoded_batches = []

    with torch.no_grad():
        for (batch,) in dataset:
            encoded_batches.append(encoder(batch.to(device)).detach().cpu())

    return torch.cat(encoded_batches, dim=0).numpy()


def initialize_clusters(
    encoder: nn.Module,
    X: np.ndarray,
    k: int,
    batch_size: int = BATCH_SIZE,
    device: torch.device | str = DEVICE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms input data into latent space, then applies KMeans on Z to
    initialize cluster centroids for DEC.

    Returns:
        Z: embedded feature representations, shape (n_samples, latent_dim)
        mu: cluster centroids in latent space, shape (k, latent_dim)
        cluster_labels: hard KMeans labels, shape (n_samples,)
    """
    _as_2d_finite_array("X", X)
    _validate_positive_int("k", k)
    _validate_positive_int("batch_size", batch_size)
    if k > len(X):
        raise ValueError(f"k must be <= number of samples; received k={k}, n_samples={len(X)}.")

    print("\nRunning KMeans clustering on latent features...")
    Z = encode_dataset(encoder, X, batch_size=batch_size, device=device)

    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    cluster_labels = kmeans.fit_predict(Z)
    mu = kmeans.cluster_centers_

    print(f"Latent shape Z: {Z.shape}")
    print(f"Initial centroids mu: {mu.shape}")
    print(f"Cluster size distribution: {np.bincount(cluster_labels, minlength=k)}")

    return Z, mu, cluster_labels


def compute_soft_assignments(Z: np.ndarray, mu: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Computes the soft assignment matrix Q using Student's t-distribution kernel.

    Q[i, j] is interpreted as the probability of assigning sample i to cluster j.
    Each row is normalized so that the probabilities across clusters sum to 1.
    """
    Z = _as_2d_finite_array("Z", Z)
    mu = _as_2d_finite_array("mu", mu)
    _validate_positive_float("alpha", alpha)
    if Z.shape[1] != mu.shape[1]:
        raise ValueError(
            "Z and mu must have the same latent dimension; "
            f"received Z.shape={Z.shape}, mu.shape={mu.shape}."
        )

    print("\nComputing soft assignments Q using Student's t-distribution...")

    squared_dist = np.sum((Z[:, np.newaxis, :] - mu[np.newaxis, :, :]) ** 2, axis=2)
    unnormalized_affinity = (1.0 + squared_dist / alpha) ** (-(alpha + 1.0) / 2.0)
    Q = unnormalized_affinity / np.sum(unnormalized_affinity, axis=1, keepdims=True)

    print(f"Q shape: {Q.shape} (Rows = samples, Cols = clusters)")
    print(f"Q min/max: {Q.min():.6f} / {Q.max():.6f}")
    print(
        "Q row sums (should all be near 1): "
        f"mean={np.mean(np.sum(Q, axis=1)):.4f}, std={np.std(np.sum(Q, axis=1)):.6f}"
    )
    print(f"Example soft assignments (first 3 rows):\n{Q[:3]}")

    return Q


def compute_target_distribution(Q: np.ndarray) -> np.ndarray:
    """
    Computes the auxiliary target distribution P from soft assignments Q.

    DEC sharpens confident assignments by squaring Q, then compensates for
    cluster frequency so very large clusters do not dominate the loss.
    """
    Q = _as_2d_finite_array("Q", Q)
    if np.any(Q < 0):
        raise ValueError("Q must contain non-negative soft assignment values.")
    if np.any(np.sum(Q, axis=1) <= 0):
        raise ValueError("Each row of Q must have positive probability mass.")

    print("\nComputing target distribution P...")

    eps = 1e-8
    f = np.sum(Q, axis=0)
    P = (Q**2) / np.maximum(f[np.newaxis, :], eps)
    P = P / np.maximum(np.sum(P, axis=1, keepdims=True), eps)

    print(f"P shape: {P.shape} (same as Q)")
    print(f"P min/max: {P.min():.6f} / {P.max():.6f}")
    print(
        "P row sums (should all be near 1): "
        f"mean={np.mean(np.sum(P, axis=1)):.4f}, std={np.std(np.sum(P, axis=1)):.6f}"
    )
    print(f"Example target assignments (first 3 rows):\n{P[:3]}")

    return P


def kl_loss(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Computes KL(P || Q), averaged over samples.

    P is the target distribution and Q is the current soft assignment. The small
    clamp avoids log(0), which would turn the loss into inf or nan.
    """
    if P.shape != Q.shape:
        raise ValueError(f"P and Q must have the same shape; received {P.shape} and {Q.shape}.")
    if P.ndim != 2:
        raise ValueError(f"P and Q must be 2D tensors; received shape {P.shape}.")
    if not torch.isfinite(P).all() or not torch.isfinite(Q).all():
        raise ValueError("P and Q must contain only finite values.")
    if torch.any(P < 0) or torch.any(Q < 0):
        raise ValueError("P and Q must contain non-negative probability values.")

    P = P.clamp(min=1e-8)
    Q = Q.clamp(min=1e-8)
    loss = torch.sum(P * torch.log(P / Q), dim=1)
    return loss.mean()


def _soft_assignments_torch(
    Z: torch.Tensor,
    mu: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    dist_sq = torch.sum((Z.unsqueeze(1) - mu.unsqueeze(0)) ** 2, dim=2)
    numerator = (1.0 + dist_sq / alpha) ** (-(alpha + 1.0) / 2.0)
    return numerator / torch.sum(numerator, dim=1, keepdim=True).clamp_min(1e-8)


def _target_distribution_torch(Q: torch.Tensor) -> torch.Tensor:
    f = torch.sum(Q, dim=0).clamp_min(1e-8)
    P = (Q**2) / f
    return P / torch.sum(P, dim=1, keepdim=True).clamp_min(1e-8)


def _encode_tensor_batches(
    encoder: nn.Module,
    X_tensor: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    dataset = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
    encoded_batches = []

    for (batch,) in dataset:
        encoded_batches.append(encoder(batch.to(device)))

    return torch.cat(encoded_batches, dim=0)


__all__ = [
    "BATCH_SIZE",
    "DECTrainingResult",
    "DEVICE",
    "Encoder",
    "FullAutoencoder",
    "compute_soft_assignments",
    "compute_target_distribution",
    "encode_dataset",
    "initialize_clusters",
    "kl_loss",
]
