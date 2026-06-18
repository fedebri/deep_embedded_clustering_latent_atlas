from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from dec_model import compute_soft_assignments


def load_preprocessed_digits() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pipeline]:
    """
    Load and preprocess the classic sklearn handwritten-digits dataset.

    Returns:
        X_raw: original 64-pixel vectors
        X_preprocessed: scaled vectors after removing constant pixels
        y: true digit labels, used only for evaluation
        images: 8x8 images for display
        preprocess: fitted sklearn preprocessing pipeline
    """
    digits = load_digits()

    # Raw data is useful for explaining the dataset; the model uses the
    # preprocessed version below.
    X_raw = digits.data.astype(np.float32)
    y = digits.target.astype(int)
    images = digits.images

    # Two deliberately simple preprocessing steps:
    # 1. Make pixel scales comparable.
    # 2. Drop pixels that never change, because they cannot help clustering.
    preprocess = Pipeline(
        steps=[
            ("scale_to_unit_interval", MinMaxScaler()),
            ("drop_constant_pixels", VarianceThreshold(threshold=0.0)),
        ]
    )
    X_preprocessed = preprocess.fit_transform(X_raw).astype(np.float32)

    return X_raw, X_preprocessed, y, images, preprocess


def clustering_accuracy_from_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, dict[int, int], np.ndarray, np.ndarray]:
    """
    Compute clustering accuracy after finding the best cluster-to-label mapping.

    Clustering IDs are arbitrary: cluster 0 does not mean label 0. The Hungarian
    matching step finds the best one-to-one assignment for evaluation only.
    """
    cm = confusion_matrix(y_true, y_pred)

    # linear_sum_assignment minimizes cost. Subtracting from cm.max() turns
    # "large overlap is good" into "small cost is good".
    row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
    accuracy = cm[row_ind, col_ind].sum() / cm.sum()

    # Keys are cluster IDs and values are the matched true labels.
    cluster_to_label = {cluster: label for label, cluster in zip(row_ind, col_ind)}
    aligned_pred = np.array([cluster_to_label.get(cluster, -1) for cluster in y_pred])

    return accuracy, cluster_to_label, aligned_pred, cm


def compare_clustering_stages(
    y_true: np.ndarray,
    stage_to_labels: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """
    Build a compact metrics table for multiple clustering stages.

    The notebook uses this to compare KMeans initialization with DEC refinement.
    The function also returns per-stage details for later confusion-matrix plots.
    """
    rows = []
    details = {}

    for stage_name, labels in stage_to_labels.items():
        matched_accuracy, cluster_to_label, aligned_labels, cm = clustering_accuracy_from_labels(
            y_true,
            labels,
        )

        rows.append(
            {
                "stage": stage_name,
                "ARI": adjusted_rand_score(y_true, labels),
                "NMI": normalized_mutual_info_score(y_true, labels),
                "matched_accuracy": matched_accuracy,
            }
        )

        details[stage_name] = {
            "matched_accuracy": matched_accuracy,
            "cluster_to_label": cluster_to_label,
            "aligned_labels": aligned_labels,
            "confusion_matrix": cm,
        }

    return pd.DataFrame(rows), details


def build_latent_feature_table(
    Z: np.ndarray,
    centroids: np.ndarray,
    cluster_labels: np.ndarray,
    true_labels: np.ndarray | None = None,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Turn a DEC latent matrix into a readable feature table.

    Each z_* column is one learned latent coordinate. Confidence is the largest
    soft assignment for a row, so it gives a simple "how sure is DEC?" signal.
    """
    latent_dim_names = [f"z_{i}" for i in range(Z.shape[1])]
    soft_assignments = compute_soft_assignments(Z, centroids)
    confidence = soft_assignments.max(axis=1)
    soft_cluster = soft_assignments.argmax(axis=1)

    latent_df = pd.DataFrame(Z, columns=latent_dim_names)
    latent_df["cluster"] = cluster_labels
    latent_df["soft_cluster"] = soft_cluster
    latent_df["confidence"] = confidence

    if true_labels is not None:
        latent_df["true_digit"] = true_labels

    return latent_df, latent_dim_names, soft_assignments, confidence, soft_cluster


def score_latent_dimensions(
    latent_df: pd.DataFrame,
    latent_dim_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score latent dimensions by how well they separate clusters.

    A high separation score means a dimension varies more between clusters than
    inside clusters. This is a simple, beginner-friendly way to rank latent axes.
    """
    cluster_means = latent_df.groupby("cluster")[latent_dim_names].mean()
    overall_variance = latent_df[latent_dim_names].var(axis=0)
    between_cluster_variance = cluster_means.var(axis=0)
    within_cluster_variance = (
        latent_df.groupby("cluster")[latent_dim_names].var().mean(axis=0).fillna(0)
    )
    confidence_correlation = (
        latent_df[latent_dim_names].corrwith(latent_df["confidence"]).abs().fillna(0)
    )

    feature_importance = pd.DataFrame(
        {
            "overall_variance": overall_variance,
            "between_cluster_variance": between_cluster_variance,
            "within_cluster_variance": within_cluster_variance,
            "separation_score": between_cluster_variance / (within_cluster_variance + 1e-8),
            "abs_corr_with_confidence": confidence_correlation,
        }
    ).sort_values("separation_score", ascending=False)

    return feature_importance, cluster_means


def build_cluster_profile(
    latent_df: pd.DataFrame,
    cluster_means: pd.DataFrame,
    top_latent_dims: list[str],
) -> pd.DataFrame:
    """
    Build a cluster profile table using the most important latent dimensions.

    Rows are clusters. Columns are the latent dimensions that best separate the
    clusters. The cluster_size column helps spot tiny or dominant clusters.
    """
    cluster_profile = cluster_means[top_latent_dims].copy()
    cluster_profile["cluster_size"] = latent_df["cluster"].value_counts().sort_index()
    return cluster_profile.sort_values("cluster_size", ascending=False)


def evaluate_latent_subset(
    Z_subset: np.ndarray,
    subset_name: str,
    true_labels: np.ndarray | None = None,
    n_clusters: int = 10,
) -> dict[str, float | int | str]:
    """
    Cluster a selected slice of latent space and score the result.

    This is the ablation unit: if a few dimensions are really important, KMeans
    should still find useful structure when it sees only those dimensions.
    """
    subset_kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    subset_labels = subset_kmeans.fit_predict(Z_subset)

    row: dict[str, float | int | str] = {
        "subset": subset_name,
        "n_dimensions": Z_subset.shape[1],
        "silhouette": (
            silhouette_score(Z_subset, subset_labels)
            if len(np.unique(subset_labels)) > 1
            else np.nan
        ),
    }

    if true_labels is not None:
        matched_acc, _, _, _ = clustering_accuracy_from_labels(true_labels, subset_labels)
        row["ARI"] = adjusted_rand_score(true_labels, subset_labels)
        row["NMI"] = normalized_mutual_info_score(true_labels, subset_labels)
        row["matched_accuracy"] = matched_acc

    return row


def evaluate_latent_subsets(
    Z: np.ndarray,
    latent_dim_names: list[str],
    ordered_latent_dims: list[str],
    true_labels: np.ndarray | None = None,
    n_clusters: int = 10,
) -> pd.DataFrame:
    """
    Compare clustering quality across useful and weak latent-dimension subsets.

    The weakest subset is included on purpose as a negative comparison.
    """
    subset_specs = [
        ("all latent dimensions", ordered_latent_dims),
        ("top 2 dimensions", ordered_latent_dims[: min(2, len(ordered_latent_dims))]),
        ("top 3 dimensions", ordered_latent_dims[: min(3, len(ordered_latent_dims))]),
        ("top 5 dimensions", ordered_latent_dims[: min(5, len(ordered_latent_dims))]),
        ("weakest 3 dimensions", ordered_latent_dims[-min(3, len(ordered_latent_dims)) :]),
    ]

    subset_results = []
    for subset_name, subset_dims in subset_specs:
        # Convert names like "z_3" back into integer column positions.
        subset_indices = [latent_dim_names.index(dim) for dim in subset_dims]
        subset_results.append(
            evaluate_latent_subset(
                Z[:, subset_indices],
                subset_name,
                true_labels=true_labels,
                n_clusters=n_clusters,
            )
        )

    return pd.DataFrame(subset_results)


__all__ = [
    "build_cluster_profile",
    "build_latent_feature_table",
    "clustering_accuracy_from_labels",
    "compare_clustering_stages",
    "evaluate_latent_subset",
    "evaluate_latent_subsets",
    "load_preprocessed_digits",
    "score_latent_dimensions",
]
