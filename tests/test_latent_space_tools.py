import numpy as np
import pytest

from dec_model import compute_soft_assignments, compute_target_distribution
from latent_space_tools import (
    ClusteringStageDetails,
    build_cluster_profile,
    build_latent_feature_table,
    clustering_accuracy_from_labels,
    compare_clustering_stages,
    evaluate_latent_subsets,
    score_latent_dimensions,
)


def test_clustering_accuracy_matches_permuted_labels() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])

    accuracy, cluster_to_label, aligned_pred, confusion = clustering_accuracy_from_labels(
        y_true,
        y_pred,
    )

    assert accuracy == 1.0
    assert cluster_to_label == {1: 0, 0: 1}
    assert np.array_equal(aligned_pred, y_true)
    assert confusion.shape == (2, 2)


def test_compare_clustering_stages_returns_structured_details() -> None:
    y_true = np.array([0, 0, 1, 1])
    labels = np.array([1, 1, 0, 0])

    metrics_df, details = compare_clustering_stages(y_true, {"permuted": labels})

    assert list(metrics_df["stage"]) == ["permuted"]
    assert isinstance(details["permuted"], ClusteringStageDetails)
    assert details["permuted"].matched_accuracy == 1.0


def test_latent_feature_scoring_and_profile_shapes() -> None:
    Z = np.array(
        [
            [0.0, 0.1],
            [0.1, 0.2],
            [4.0, 0.3],
            [4.2, 0.4],
        ],
        dtype=float,
    )
    centroids = np.array([[0.05, 0.15], [4.1, 0.35]], dtype=float)
    cluster_labels = np.array([0, 0, 1, 1])

    latent_df, latent_dim_names, soft_assignments, confidence, soft_cluster = (
        build_latent_feature_table(Z, centroids, cluster_labels)
    )
    feature_importance, cluster_means = score_latent_dimensions(latent_df, latent_dim_names)
    top_dims = feature_importance.head(1).index.tolist()
    cluster_profile = build_cluster_profile(latent_df, cluster_means, top_dims)

    assert latent_dim_names == ["z_0", "z_1"]
    assert latent_df.shape == (4, 5)
    assert soft_assignments.shape == (4, 2)
    assert confidence.shape == (4,)
    assert soft_cluster.shape == (4,)
    assert feature_importance.index[0] == "z_0"
    assert cluster_profile.shape == (2, 2)


def test_evaluate_latent_subsets_returns_expected_rows() -> None:
    rng = np.random.default_rng(42)
    cluster_a = rng.normal(loc=0.0, scale=0.05, size=(8, 4))
    cluster_b = rng.normal(loc=2.0, scale=0.05, size=(8, 4))
    Z = np.vstack([cluster_a, cluster_b])
    y_true = np.array([0] * 8 + [1] * 8)
    latent_dim_names = [f"z_{index}" for index in range(Z.shape[1])]

    subset_results = evaluate_latent_subsets(
        Z,
        latent_dim_names,
        latent_dim_names,
        true_labels=y_true,
        n_clusters=2,
    )

    assert list(subset_results["subset"]) == [
        "all latent dimensions",
        "top 2 dimensions",
        "top 3 dimensions",
        "top 5 dimensions",
        "weakest 3 dimensions",
    ]
    assert subset_results["matched_accuracy"].between(0, 1).all()


def test_dec_probability_helpers_validate_shapes() -> None:
    Z = np.array([[0.0, 0.0], [1.0, 1.0]])
    centroids = np.array([[0.0, 0.0], [1.0, 1.0]])

    soft_assignments = compute_soft_assignments(Z, centroids)
    target_distribution = compute_target_distribution(soft_assignments)

    assert soft_assignments.shape == (2, 2)
    assert target_distribution.shape == (2, 2)
    assert np.allclose(soft_assignments.sum(axis=1), 1.0)
    assert np.allclose(target_distribution.sum(axis=1), 1.0)

    with pytest.raises(ValueError, match="same latent dimension"):
        compute_soft_assignments(Z, np.array([[0.0, 0.0, 0.0]]))

    with pytest.raises(ValueError, match="non-negative"):
        compute_target_distribution(np.array([[0.5, -0.5]]))
