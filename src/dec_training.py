"""Iteration-based DEC optimization shared by all three notebooks.

Jointly train the encoder and centroids using momentum SGD, Student-t
affinities, full-sample targets and the KL objective.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dec_model import (
    DECTrainingResult,
    _as_2d_finite_array,
    _encode_tensor_batches,
    _soft_assignments_torch,
    _target_distribution_torch,
    _validate_positive_int,
    kl_loss,
)


def train_dec(
    encoder: nn.Module,
    X: np.ndarray,
    mu: np.ndarray,
    *,
    batch_size: int,
    max_steps: int,
    update_interval: int,
    lr: float,
    momentum: float,
    alpha: float,
    tol: float,
    device: torch.device | str,
) -> DECTrainingResult:
    """Jointly optimize the encoder and centroids with momentum SGD.

    Refresh a detached target over the entire clustering sample every
    ``update_interval`` mini-batch updates. Labels never enter training.
    Stop when the fraction of changed hard assignments between refreshes is
    below ``tol``, or when ``max_steps`` updates have completed. ``loss_history``
    contains the mean KL per sample for each training batch. A budget stop is
    not reported as convergence; ``label_change`` is the last refresh comparison.
    """
    X = _as_2d_finite_array("X", X)
    mu = _as_2d_finite_array("mu", mu)
    _validate_positive_int("max_steps", max_steps)
    _validate_positive_int("update_interval", update_interval)
    device = torch.device(device)
    encoder.to(device)
    centers = nn.Parameter(
        torch.as_tensor(mu, dtype=torch.float32, device=device).clone()
    )
    optimizer = torch.optim.SGD(
        [*encoder.parameters(), centers],
        lr=lr,
        momentum=momentum,
        weight_decay=0.0,
    )
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(X_tensor, torch.arange(len(X_tensor))),
        batch_size=batch_size,
        shuffle=True,
    )
    batches = iter(loader)
    previous_labels = None
    label_change = None
    converged = False
    loss_history: list[float] = []

    for step in range(max_steps):
        if step % update_interval == 0:
            encoder.eval()
            with torch.no_grad():
                Z = _encode_tensor_batches(encoder, X_tensor, batch_size, device)
                Q = _soft_assignments_torch(Z, centers, alpha=alpha)
                P = _target_distribution_torch(Q).detach()
                labels = Q.argmax(dim=1).cpu().numpy()
            if previous_labels is not None:
                label_change = float(np.mean(labels != previous_labels))
                print(f"Step {step} | Changed assignments: {label_change:.6f}")
                if label_change < tol:
                    converged = True
                    break
            previous_labels = labels
            encoder.train()

        try:
            batch, indices = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch, indices = next(batches)
        optimizer.zero_grad()
        Q_batch = _soft_assignments_torch(
            encoder(batch.to(device)), centers, alpha=alpha
        )
        # The shuffled batch must use its original rows of the full-sample target.
        loss = kl_loss(P[indices.to(device)], Q_batch)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if step == 0 or (step + 1) % update_interval == 0:
            print(f"Step {step + 1} | Batch mean KL: {loss.item():.6f}")

    encoder.eval()
    with torch.no_grad():
        Z = _encode_tensor_batches(encoder, X_tensor, batch_size, device)
        labels = _soft_assignments_torch(Z, centers, alpha=alpha).argmax(dim=1)
    reason = "assignment-change tolerance reached" if converged else "budget exhausted"
    print(f"DEC stopped after {len(loss_history)} updates: {reason}.")
    return DECTrainingResult(
        encoder=encoder,
        centroids=centers.detach().cpu().numpy(),
        labels=labels.cpu().numpy(),
        loss_history=loss_history,
        converged=converged,
        label_change=label_change,
    )
