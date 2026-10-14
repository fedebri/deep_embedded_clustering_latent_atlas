"""Deterministic checks derived from Xie et al. (2016), not published benchmarks.

These tests use fixed tiny tensors: no datasets, model fitting or optimization.
"""

import numpy as np
import torch

from dec_model import (
    _soft_assignments_torch,
    _target_distribution_torch,
    compute_soft_assignments,
    compute_target_distribution,
    kl_loss,
)


def test_assignments_and_targets_match_hand_calculated_example():
    """Equations 1 and 3 give these exact fractions for two centers."""
    Z = np.array([[0.0], [1.0], [3.0]])
    mu = np.array([[0.0], [2.0]])
    expected_Q = np.array([[5 / 6, 1 / 6], [1 / 2, 1 / 2], [1 / 6, 5 / 6]])
    expected_P = np.array([[25 / 26, 1 / 26], [1 / 2, 1 / 2], [1 / 26, 25 / 26]])
    Q = compute_soft_assignments(Z, mu, alpha=1.0)
    np.testing.assert_allclose(Q, expected_Q)
    np.testing.assert_allclose(compute_target_distribution(Q), expected_P)
    Q_torch = _soft_assignments_torch(torch.from_numpy(Z), torch.from_numpy(mu))
    torch.testing.assert_close(Q_torch, torch.from_numpy(expected_Q))
    torch.testing.assert_close(
        _target_distribution_torch(Q_torch), torch.from_numpy(expected_P)
    )


def test_embedding_and_center_gradients_match_paper_equations_4_and_5():
    """Account for this implementation's sample-mean rather than summed KL."""
    Z = torch.tensor([[0.0, 0.5], [1.0, -0.5], [3.0, 1.0]],
                     dtype=torch.float64, requires_grad=True)
    mu = torch.tensor([[0.0, 0.0], [2.0, 1.0]],
                      dtype=torch.float64, requires_grad=True)
    Q = _soft_assignments_torch(Z, mu, alpha=1.0)
    P = _target_distribution_torch(Q).detach()
    loss = kl_loss(P, Q)
    actual_Z, actual_mu = torch.autograd.grad(loss, (Z, mu))

    with torch.no_grad():
        difference = Z[:, None, :] - mu[None, :, :]
        affinity = 1 / (1 + difference.square().sum(dim=2))
        contribution = (2 * affinity * (P - Q))[:, :, None] * difference
        expected_Z = contribution.sum(dim=1) / len(Z)
        expected_mu = -contribution.sum(dim=0) / len(Z)
    torch.testing.assert_close(actual_Z, expected_Z)
    torch.testing.assert_close(actual_mu, expected_mu)
    torch.testing.assert_close(actual_Z.sum(dim=0) + actual_mu.sum(dim=0),
                               torch.zeros(2, dtype=torch.float64))


def test_target_uses_full_sample_cluster_frequencies():
    """Catch replacement of global soft frequencies with batch frequencies."""
    Q = torch.tensor([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]], dtype=torch.float64)
    # Frequencies are 1.5 and 1.5 over the whole sample, so the first row is 16:1.
    P = _target_distribution_torch(Q)
    torch.testing.assert_close(P[0], torch.tensor([16 / 17, 1 / 17],
                                                 dtype=torch.float64))
    assert not torch.allclose(P[:1], _target_distribution_torch(Q[:1]))
