"""Small synthetic training checks for the DEC loop; no external datasets."""

import numpy as np
import torch

import dec_training


def test_refresh_counts_updates_and_uses_all_samples(monkeypatch):
    """Five updates with interval two require targets at steps 0, 2 and 4."""
    torch.manual_seed(42)
    encoder = torch.nn.Linear(2, 2)
    X = np.array([[0, 0], [1, 0], [0, 1], [2, 1], [1, 3]], dtype=np.float32)
    centers = np.array([[0, 0], [2, 2]], dtype=np.float32)
    weights_before = encoder.weight.detach().clone()
    target = dec_training._target_distribution_torch
    refreshes = []

    def record_target(Q):
        assert not Q.requires_grad
        refreshes.append(tuple(Q.shape))
        return target(Q)

    monkeypatch.setattr(
        dec_training, "_target_distribution_torch", record_target
    )
    result = dec_training.train_dec(
        encoder, X, centers, batch_size=2, max_steps=5, update_interval=2,
        lr=0.01, momentum=0.9, alpha=1.0, tol=0.0, device="cpu",
    )
    assert refreshes == [(5, 2)] * 3
    assert len(result.loss_history) == 5
    assert not result.converged
    assert not torch.equal(weights_before, encoder.weight)
    assert not np.array_equal(centers, result.centroids)
    with torch.no_grad():
        Q = dec_training._soft_assignments_torch(
            encoder(torch.from_numpy(X)), torch.from_numpy(result.centroids)
        )
    np.testing.assert_array_equal(result.labels, Q.argmax(dim=1).numpy())


def test_stable_assignments_stop_at_refresh():
    """Zero learning rate isolates the stopping rule from optimizer behavior."""
    encoder = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        encoder.weight.fill_(1)
    result = dec_training.train_dec(
        encoder, np.array([[0.0], [2.0]], dtype=np.float32),
        np.array([[0.0], [2.0]], dtype=np.float32),
        batch_size=1, max_steps=5, update_interval=2,
        lr=0.0, momentum=0.9, alpha=1.0, tol=0.001, device="cpu",
    )
    assert result.converged
    assert result.label_change == 0.0
    assert len(result.loss_history) == 2
