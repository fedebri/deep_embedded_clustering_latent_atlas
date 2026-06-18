from __future__ import annotations

import os
from dataclasses import dataclass
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
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset


# ---- Config -----------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
PRETRAIN_EPOCHS = 10
FINETUNE_EPOCHS = 20
LEARNING_RATE = 1e-3


# ---- 1. Basic Denoising Autoencoder ----------------------------------------


class DenoisingAutoencoder(nn.Module):
    """
    One shallow denoising autoencoder used during greedy layer-wise pretraining.

    The small model learns to reconstruct a clean input from a noisy version of
    that input. After it has learned this reconstruction task, its encoder layer
    is reused as one layer in the final deep encoder.
    """

    def __init__(self, input_dim: int, hidden_dim: int, activate_hidden: bool = True):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)  # Encoder: reduces dimension.
        self.decoder = nn.Linear(hidden_dim, input_dim)  # Decoder: reconstructs the original input dimension.
        self.activate_hidden = activate_hidden

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        if self.activate_hidden:
            # ReLU keeps positive evidence and zeroes out negative evidence.
            # For DEC, the final embedding layer is often left linear so that
            # the bottleneck can use the whole real-valued feature space.
            h = F.relu(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        out = self.decoder(h)
        return out


# ---- 2. Stack of Encoders ---------------------------------------------------


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


class StackedEncoder(Encoder):
    """
    Backward-compatible name for the notebook's original StackedEncoder idea.

    Here layers_dims includes the input dimension as the first item, for example
    [784, 500, 500, 2000, 10].
    """

    def __init__(self, layers_dims: Iterable[int], activate_final: bool = False):
        dims = [int(dim) for dim in layers_dims]
        if len(dims) < 2:
            raise ValueError("layers_dims must include input_dim and at least one output dimension.")
        super().__init__(dims[0], dims[1:], activate_final=activate_final)


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
    """Returned by train_DEC so the notebook can keep diagnostics explicit."""

    encoder: nn.Module
    centroids: np.ndarray
    labels: np.ndarray
    loss_history: list[float]


# ---- 3. Greedy Layer-wise Pretraining --------------------------------------


def pretrain_autoencoder(
    X: np.ndarray,
    layer_dims: Iterable[int] = (500, 500, 2000, 10),
    batch_size: int = BATCH_SIZE,
    pretrain_epochs: int = PRETRAIN_EPOCHS,
    finetune_epochs: int = FINETUNE_EPOCHS,
    lr: float = LEARNING_RATE,
    noise_std: float = 0.2,
    device: torch.device | str = DEVICE,
    activate_final_encoder: bool = False,
) -> Encoder:
    """
    Pretrains a deep autoencoder using denoising + greedy layer-wise training.

    Input:
        X: np.ndarray of shape (n_samples, input_dim)
        layer_dims: hidden dimensions, excluding the input dimension
        batch_size: how many samples are processed per optimizer step
        pretrain_epochs: greedy layer-wise epochs per layer
        finetune_epochs: epochs for full autoencoder reconstruction fine-tuning
        lr: Adam learning rate
        noise_std: Gaussian noise scale added during denoising pretraining
        device: CPU/GPU device
        activate_final_encoder: whether to apply ReLU to the final latent layer

    Output:
        encoder: Encoder that maps X to Z (latent space)
    """
    device = torch.device(device)
    layer_dims = [int(dim) for dim in layer_dims]

    print("\nStarting Autoencoder Pretraining")
    print(f"Initial input shape: {X.shape}")

    # Prepare data for PyTorch.
    X_tensor = torch.as_tensor(X, dtype=torch.float32)

    current_X = X_tensor  # Input for the first layer.
    input_dim = X.shape[1]  # Starting input dimension.
    pretrained_layers: list[nn.Linear] = []  # Store pretrained encoder layers.

    # ----- GREEDY LAYER-WISE PRETRAINING -----
    # Each hidden layer is pretrained as a shallow denoising autoencoder (DAE)
    # using MSE loss. Gaussian noise is added to encourage robust feature
    # learning. After each DAE is trained, its encoder transforms the whole
    # dataset, and that transformed dataset becomes the input for the next layer.
    for i, dim in enumerate(layer_dims):
        print(f"\nPretraining Layer {i + 1}: {input_dim} -> {dim}")

        is_final_layer = i == len(layer_dims) - 1
        activate_hidden = activate_final_encoder or not is_final_layer

        dae = DenoisingAutoencoder(input_dim, dim, activate_hidden=activate_hidden).to(device)
        optimizer = optim.Adam(dae.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        dataset = DataLoader(TensorDataset(current_X), batch_size=batch_size, shuffle=True)

        for epoch in range(1, pretrain_epochs + 1):
            total_loss = 0.0

            for (batch,) in dataset:
                batch = batch.to(device)
                noisy = batch + noise_std * torch.randn_like(batch)
                output = dae(noisy)
                loss = loss_fn(output, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.size(0)

            avg_loss = total_loss / len(dataset.dataset)
            if epoch == 1 or epoch == pretrain_epochs:
                print(f"   Epoch {epoch:2d} | Reconstruction Loss: {avg_loss:.6f}")

        # After training, use this encoder's output as next layer's input.
        dae.eval()
        with torch.no_grad():
            current_X = dae.encode(current_X.to(device)).detach().cpu()

        layer_copy = nn.Linear(input_dim, dim)
        layer_copy.load_state_dict(dae.encoder.cpu().state_dict())
        pretrained_layers.append(layer_copy)

        print(f"Output shape after Layer {i + 1}: {current_X.shape}")
        input_dim = dim

    # ----- ASSEMBLE ENCODER AS A MODEL THAT INCLUDES ACTIVATIONS -----
    encoder = Encoder.from_linear_layers(
        X.shape[1],
        pretrained_layers,
        activate_final=activate_final_encoder,
    ).to(device)

    # ----- END-TO-END FINETUNING OF FULL AUTOENCODER -----
    model = FullAutoencoder(encoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print("\nFine-tuning full autoencoder")
    for epoch in range(1, finetune_epochs + 1):
        total_loss = 0.0
        dataset = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=True)

        for (batch,) in dataset:
            batch = batch.to(device)
            output = model(batch)
            loss = loss_fn(output, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(X_tensor)
        if epoch == 1 or epoch == finetune_epochs or epoch % 5 == 0:
            print(f"   Epoch {epoch:2d} | Fine-tune Loss: {avg_loss:.6f}")

    encoder.eval()

    # Diagnostic preview of latent space.
    with torch.no_grad():
        sample_encoded = encoder(torch.as_tensor(X[:10], dtype=torch.float32, device=device)).cpu().numpy()
        print("\nFinal encoder output sample:")
        print("   Shape:", sample_encoded.shape)
        print("   Mean per latent dim:", np.mean(sample_encoded, axis=0))
        print("   Std per latent dim: ", np.std(sample_encoded, axis=0))

    print("\nPretraining complete. Encoder ready for clustering.\n")
    return encoder


def encode_dataset(
    encoder: nn.Module,
    X: np.ndarray,
    batch_size: int = BATCH_SIZE,
    device: torch.device | str = DEVICE,
) -> np.ndarray:
    """Transforms raw data X into latent features Z in batches."""
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


def train_DEC(
    encoder: nn.Module,
    X: np.ndarray,
    mu: np.ndarray,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = 50,
    update_interval: int = 5,
    lr: float = 1e-3,
    alpha: float = 1.0,
    tol: float | None = 0.001,
    device: torch.device | str = DEVICE,
) -> DECTrainingResult:
    """
    Fine-tunes the encoder and cluster centers using KL divergence.

    This is the DEC phase after autoencoder pretraining and KMeans
    initialization. The target distribution P is recomputed every
    update_interval epochs, while Q is recomputed for each mini-batch so the
    gradient can update both the encoder parameters and the centroids.
    """
    device = torch.device(device)
    encoder.to(device)
    encoder.train()

    center_param = nn.Parameter(torch.as_tensor(mu, dtype=torch.float32, device=device).clone())
    optimizer = torch.optim.Adam([*encoder.parameters(), center_param], lr=lr)

    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    indices = torch.arange(len(X_tensor), dtype=torch.long)
    dataset = TensorDataset(X_tensor, indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    target_distribution = None
    previous_labels = None
    final_labels = None
    loss_history: list[float] = []

    for epoch in range(max_epochs):
        if epoch % update_interval == 0 or target_distribution is None:
            encoder.eval()
            with torch.no_grad():
                full_Z = _encode_tensor_batches(encoder, X_tensor, batch_size, device)
                full_Q = _soft_assignments_torch(full_Z, center_param, alpha=alpha)
                target_distribution = _target_distribution_torch(full_Q).detach()
                current_labels = torch.argmax(full_Q, dim=1).cpu().numpy()

            if previous_labels is not None and tol is not None:
                label_delta = np.mean(current_labels != previous_labels)
                print(f"   Target refresh | changed labels: {label_delta:.6f}")
                if label_delta < tol:
                    final_labels = current_labels
                    print(f"Stopping early because label change {label_delta:.6f} < tol {tol}.")
                    break

            previous_labels = current_labels
            final_labels = current_labels
            encoder.train()

        epoch_loss = 0.0

        for x_batch, idx_batch in dataloader:
            x_batch = x_batch.to(device)
            idx_batch = idx_batch.to(device)

            optimizer.zero_grad()

            z_batch = encoder(x_batch)
            q_batch = _soft_assignments_torch(z_batch, center_param, alpha=alpha)

            # P was computed over the full dataset; idx_batch selects matching rows.
            p_batch = target_distribution[idx_batch]
            loss = kl_loss(p_batch, q_batch)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(x_batch)

        mean_epoch_loss = epoch_loss / len(X)
        loss_history.append(mean_epoch_loss)
        print(f"Epoch {epoch + 1:2d} | KL Loss: {mean_epoch_loss:.6f}")

    encoder.eval()
    with torch.no_grad():
        full_Z = _encode_tensor_batches(encoder, X_tensor, batch_size, device)
        full_Q = _soft_assignments_torch(full_Z, center_param, alpha=alpha)
        final_labels = torch.argmax(full_Q, dim=1).cpu().numpy()

    print("DEC training complete.")

    return DECTrainingResult(
        encoder=encoder,
        centroids=center_param.detach().cpu().numpy(),
        labels=final_labels,
        loss_history=loss_history,
    )


def plot_tsne(
    Z: np.ndarray,
    labels: np.ndarray | None = None,
    title: str = "t-SNE of Latent Space",
) -> None:
    """Plots a 2D t-SNE visualization of latent representations Z."""
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
    """Encodes X with the fine-tuned encoder, plots t-SNE, and returns Z."""
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
    "BATCH_SIZE",
    "DECTrainingResult",
    "DEVICE",
    "DenoisingAutoencoder",
    "Encoder",
    "FINETUNE_EPOCHS",
    "FullAutoencoder",
    "LEARNING_RATE",
    "PRETRAIN_EPOCHS",
    "StackedEncoder",
    "compute_soft_assignments",
    "compute_target_distribution",
    "encode_dataset",
    "initialize_clusters",
    "kl_loss",
    "plot_final_tsne",
    "plot_tsne",
    "pretrain_autoencoder",
    "train_DEC",
]
