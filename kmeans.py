"""
Summary
-------
This script loads `data/crops.csv`, keeps only the selected crop-soil
features `N`, `P`, `K`, and `ph`, standardizes them, and then explores KMeans
clustering with nine plot outputs:

1. a cumulative PCA variance plot,
2. an elbow-method plot based on WCSS for k = 2 to 14,
3. a silhouette-score plot for k = 2 to 14,
4. a final standardized-space KMeans plot for k = 2,
5. a final standardized-space KMeans plot for k = 4,
6. a side-by-side plot of standardized-space k = 4 versus PCA-space k = 4,
7. a side-by-side plot of standardized-space k = 4 versus standardized-space k = 2,
8. a crop-count plot by cluster for standardized-space k = 2,
9. a crop-count plot by cluster for standardized-space k = 4.

After computing those diagnostics, it chooses a final cluster count, fits
KMeans on the standardized numeric features, also fits a comparison KMeans
directly in the projected 2D PCA space, saves the cluster-labelled dataset plus
supporting summary files in `outputs/kmeans`, and displays the plots by default
when you run `py kmeans.py`.

How cumulative PCA variance affects KMeans interpretation
---------------------------------------------------------
- If cumulative variance rises quickly, then a small number of principal
  components preserves most of the dataset's variation. In that case, patterns
  seen in a low-dimensional PCA summary are more likely to reflect the same
  broad geometry that drives KMeans in the full standardized feature space.
- If cumulative variance rises slowly, the variation is spread across many
  dimensions. Then a 2D or 3D PCA view can hide meaningful structure, so
  clusters that look overlapped in a reduced view may still be separable enough
  for KMeans in the original feature space.
- Because of that, low cumulative variance in the first few components does not
  automatically mean KMeans is invalid. It means PCA visualizations should be
  treated as incomplete summaries rather than definitive evidence about cluster
  quality.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


DATA_PATH = Path("data/crops.csv")
OUTPUT_DIR = Path("outputs/kmeans")
FEATURE_COLUMNS = ["N", "P", "K", "ph"]
LABEL_COLUMN = "label"
DEFAULT_MIN_K = 2
DEFAULT_MAX_K = 14
DEFAULT_RANDOM_STATE = 2026
COMPARISON_K_VALUES = [2, 4]
RIGHTMOST_CLUSTER_COLOR = "#1f77b4"
COMPARISON_CLUSTER_COLORS = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
PAPER_COLORS = {
    0: "#440154",  # purple
    1: "#31688e",  # blue
    2: "#35b779",  # green
    3: "#fde725",  # yellow
}


def progress_bar(iterable, desc: str, total: int | None = None, leave: bool = True):
    """Wrap an iterable with `tqdm` when progress reporting is available."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=leave)


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load the crop dataset and return the validated clustering feature table."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    missing_columns = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Required feature columns are missing from {data_path}: {', '.join(missing_columns)}")

    numeric_columns = FEATURE_COLUMNS.copy()
    non_numeric_columns = [column for column in numeric_columns if not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric_columns:
        raise ValueError(
            f"These feature columns must be numeric in {data_path}: {', '.join(non_numeric_columns)}"
        )

    numeric_df = df[numeric_columns].copy()
    if numeric_df.isna().any().any():
        raise ValueError(f"Missing numeric values were found in {data_path}; clean the data before clustering.")

    return df, numeric_df, numeric_columns


def standardize_features(numeric_df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Standardize the selected numeric features and return the fitted scaler."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(numeric_df)
    return features_scaled, scaler


def validate_standardized_features(features_scaled: np.ndarray) -> None:
    """Check that scaling produced near-zero means and unit standard deviations."""
    column_means = features_scaled.mean(axis=0)
    column_standard_deviations = features_scaled.std(axis=0, ddof=0)

    if not np.allclose(column_means, 0.0, atol=1e-8):
        raise ValueError("Feature standardization failed: scaled feature means are not approximately zero.")
    if not np.allclose(column_standard_deviations, 1.0, atol=1e-8):
        raise ValueError("Feature standardization failed: scaled feature standard deviations are not approximately one.")


def make_kmeans(n_clusters: int, random_state: int) -> KMeans:
    """Create a `KMeans` estimator with the project's default settings."""
    return KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)


def pca_cumulative_variance(features_scaled: np.ndarray) -> np.ndarray:
    """Compute cumulative explained variance across all PCA components."""
    pca = PCA()
    pca.fit(features_scaled)
    return np.cumsum(pca.explained_variance_ratio_)


def resolve_k_values(k_values: list[int], n_samples: int) -> list[int]:
    """Validate, de-duplicate, and sort the explored cluster counts."""
    resolved_k_values = sorted(set(k_values))

    if not resolved_k_values:
        raise ValueError("At least one k value must be provided.")
    if resolved_k_values[0] < 2:
        raise ValueError("All explored k values must be at least 2.")
    if resolved_k_values[-1] >= n_samples:
        raise ValueError("All explored k values must be smaller than the number of rows in the dataset.")

    return resolved_k_values


def detect_elbow(k_values: list[int], wcss_values: list[float], minimum_allowed_k: int) -> int:
    """Estimate the elbow point using maximum distance from the endpoint line."""
    points = np.column_stack([k_values, wcss_values])
    first_point = points[0]
    last_point = points[-1]
    line_vector = last_point - first_point
    line_length = np.linalg.norm(line_vector)

    if line_length == 0:
        return minimum_allowed_k

    line_unit = line_vector / line_length
    vectors_from_first = points - first_point
    projection_lengths = vectors_from_first @ line_unit
    projected_points = first_point + np.outer(projection_lengths, line_unit)
    distances = np.linalg.norm(points - projected_points, axis=1)

    elbow_index = int(np.argmax(distances))
    return max(minimum_allowed_k, int(k_values[elbow_index]))


def compute_wcss(features_scaled: np.ndarray, k_values: list[int], random_state: int) -> list[float]:
    """Compute within-cluster sum of squares for each candidate `k`."""
    wcss_values: list[float] = []
    for k in progress_bar(k_values, desc="Computing WCSS", total=len(k_values)):
        model = make_kmeans(n_clusters=k, random_state=random_state)
        model.fit(features_scaled)
        wcss_values.append(float(model.inertia_))
    return wcss_values


def compute_silhouette_scores(
    features_scaled: np.ndarray, k_values: list[int], random_state: int
) -> tuple[list[float], int]:
    """Compute silhouette scores for each candidate `k` and return the best one."""
    scores: list[float] = []
    for k in progress_bar(k_values, desc="Computing silhouette scores", total=len(k_values)):
        model = make_kmeans(n_clusters=k, random_state=random_state)
        labels = model.fit_predict(features_scaled)
        scores.append(float(silhouette_score(features_scaled, labels)))

    best_index = int(np.argmax(scores))
    return scores, int(k_values[best_index])


def choose_final_k(elbow_k: int, silhouette_k: int) -> tuple[int, str]:
    """Select the final cluster count from the elbow and silhouette recommendations."""
    if elbow_k == silhouette_k:
        return elbow_k, "elbow and silhouette recommendations agreed"
    return silhouette_k, "elbow and silhouette disagreed, so the silhouette peak was used"


def components_to_reach(cumulative_variance: np.ndarray, threshold: float) -> int:
    """Return how many principal components are needed to reach a variance target."""
    return int(np.searchsorted(cumulative_variance, threshold, side="left") + 1)


def plot_pca_cumulative_variance(cumulative_variance: np.ndarray, output_path: Path) -> None:
    """Plot and save cumulative explained variance across PCA components."""
    components = np.arange(1, len(cumulative_variance) + 1)
    components_80 = components_to_reach(cumulative_variance, 0.80)
    components_90 = components_to_reach(cumulative_variance, 0.90)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(components, cumulative_variance, marker="o", linewidth=2, color="#1f77b4")
    ax.axhline(0.80, color="#2ca02c", linestyle="--", linewidth=1, label="80% variance")
    ax.axhline(0.90, color="#d62728", linestyle="--", linewidth=1, label="90% variance")
    ax.axvline(components_80, color="#2ca02c", linestyle=":", linewidth=1)
    ax.axvline(components_90, color="#d62728", linestyle=":", linewidth=1)
    ax.set_title("Cumulative Variance Explained by Principal Components")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance ratio")
    ax.set_xticks(components)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def plot_elbow_method(k_values: list[int], wcss_values: list[float], elbow_k: int, output_path: Path) -> None:
    """Plot and save the elbow diagnostic for the explored `k` values."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, wcss_values, marker="o", linewidth=2, color="#ff7f0e")
    ax.axvline(elbow_k, color="#d62728", linestyle="--", linewidth=1.5, label=f"Recommended k = {elbow_k}")
    ax.set_title("Elbow Method for KMeans (WCSS)")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("WCSS")
    ax.set_xticks(k_values)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def plot_silhouette_method(k_values: list[int], scores: list[float], best_k: int, output_path: Path) -> None:
    """Plot and save silhouette scores across the explored cluster counts."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, scores, marker="o", linewidth=2, color="#2ca02c")
    ax.axvline(best_k, color="#d62728", linestyle="--", linewidth=1.5, label=f"Recommended k = {best_k}")
    ax.set_title("Silhouette Method for KMeans")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Average silhouette score")
    ax.set_xticks(k_values)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def assign_comparison_cluster_colors(projected_centers: np.ndarray) -> dict[int, str]:
    """Assign display colors to clusters based on centroid positions in PCA space."""
    rightmost_cluster_id = int(np.argmax(projected_centers[:, 0]))
    ordered_cluster_ids = [int(cluster_id) for cluster_id in np.argsort(projected_centers[:, 0])]

    color_map: dict[int, str] = {rightmost_cluster_id: RIGHTMOST_CLUSTER_COLOR}
    remaining_cluster_ids = [cluster_id for cluster_id in ordered_cluster_ids if cluster_id != rightmost_cluster_id]

    for cluster_id, color in zip(remaining_cluster_ids, COMPARISON_CLUSTER_COLORS):
        color_map[cluster_id] = color

    return color_map


def fit_standardized_kmeans(features_scaled: np.ndarray, k: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit KMeans in standardized feature space and return labels plus centroids."""
    model = make_kmeans(n_clusters=k, random_state=random_state)
    labels = model.fit_predict(features_scaled)
    return labels, model.cluster_centers_


def fit_pca_space_kmeans(projected_points: np.ndarray, k: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit KMeans directly in PCA space for comparison plots."""
    model = make_kmeans(n_clusters=k, random_state=random_state)
    labels = model.fit_predict(projected_points)
    return labels, model.cluster_centers_


def build_display_cluster_map(projected_centers: np.ndarray) -> dict[int, int]:
    """Map original cluster ids to a stable display order for plotting."""
    rightmost_cluster_id = int(np.argmax(projected_centers[:, 0]))
    remaining_cluster_ids = [cluster_id for cluster_id in range(len(projected_centers)) if cluster_id != rightmost_cluster_id]
    remaining_cluster_ids.sort(key=lambda cluster_id: projected_centers[cluster_id, 1])

    display_map = {
        remaining_cluster_ids[0]: 0,
        remaining_cluster_ids[1]: 1,
        rightmost_cluster_id: 2,
        remaining_cluster_ids[2]: 3,
    }
    return display_map


def remap_cluster_outputs(
    labels: np.ndarray, cluster_centers: np.ndarray, projected_centers: np.ndarray, display_map: dict[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reorder labels and centroids to match the chosen display mapping."""
    remapped_labels = np.array([display_map[label] for label in labels], dtype=int)

    remapped_cluster_centers = np.zeros_like(cluster_centers)
    remapped_projected_centers = np.zeros_like(projected_centers)

    for original_cluster_id, display_cluster_id in display_map.items():
        remapped_cluster_centers[display_cluster_id] = cluster_centers[original_cluster_id]
        remapped_projected_centers[display_cluster_id] = projected_centers[original_cluster_id]

    return remapped_labels, remapped_cluster_centers, remapped_projected_centers


def project_features_to_2d(features_scaled: np.ndarray) -> tuple[np.ndarray, PCA]:
    """Project standardized features into two PCA dimensions."""
    pca = PCA(n_components=2)
    projected_points = pca.fit_transform(features_scaled)
    return projected_points, pca


def plot_cluster_panel(
    ax,
    projected_points: np.ndarray,
    labels: np.ndarray,
    projected_centers: np.ndarray,
    explained_variance: np.ndarray,
    title: str,
) -> None:
    """Draw one PCA scatter panel with cluster assignments and centroids."""
    color_map = assign_comparison_cluster_colors(projected_centers)
    ordered_cluster_ids = [int(cluster_id) for cluster_id in np.argsort(projected_centers[:, 0])]
    rightmost_cluster_id = int(np.argmax(projected_centers[:, 0]))

    for cluster_id in ordered_cluster_ids:
        cluster_mask = labels == cluster_id
        ax.scatter(
            projected_points[cluster_mask, 0],
            projected_points[cluster_mask, 1],
            s=28,
            alpha=0.75,
            color=color_map[cluster_id],
            label=(
                f"Cluster {cluster_id} (rightmost)"
                if cluster_id == rightmost_cluster_id
                else f"Cluster {cluster_id}"
            ),
        )

    ax.scatter(
        projected_centers[:, 0],
        projected_centers[:, 1],
        marker="X",
        s=220,
        c="black",
        linewidths=1.2,
        label="Centroids",
    )

    for cluster_id, center in enumerate(projected_centers):
        ax.text(center[0], center[1], f" C{cluster_id}", fontsize=10, fontweight="bold", color="black")

    ax.set_title(title)
    ax.set_xlabel(f"Principal component 1 ({explained_variance[0]:.1%} variance)")
    ax.set_ylabel(f"Principal component 2 ({explained_variance[1]:.1%} variance)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)


def plot_research_style_cluster_panel(
    ax,
    projected_points: np.ndarray,
    labels: np.ndarray,
    explained_variance: np.ndarray,
    title: str,
    show_ylabel: bool = True,
) -> None:
    """Draw one research-style PCA scatter panel using the paper colour palette."""
    for cluster_id in range(len(np.unique(labels))):
        cluster_mask = labels == cluster_id
        ax.scatter(
            projected_points[cluster_mask, 0],
            projected_points[cluster_mask, 1],
            s=28,
            alpha=0.8,
            color=PAPER_COLORS[cluster_id],
            edgecolors="none",
            label=f"Cluster {cluster_id}",
        )

    ax.set_title(title)
    ax.set_xlabel(f"Principal component 1 ({explained_variance[0]:.1%} variance)")
    if show_ylabel:
        ax.set_ylabel(f"Principal component 2 ({explained_variance[1]:.1%} variance)")
    else:
        ax.set_ylabel("")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)


def plot_single_cluster_result(
    projected_points: np.ndarray,
    labels: np.ndarray,
    projected_centers: np.ndarray,
    explained_variance: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Save a single PCA-space clustering result figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_cluster_panel(ax, projected_points, labels, projected_centers, explained_variance, title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_research_style_cluster_result(
    projected_points: np.ndarray,
    labels: np.ndarray,
    explained_variance: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Save a single research-style clustering result figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_research_style_cluster_panel(ax, projected_points, labels, explained_variance, title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_cluster_result_pair(
    left_points: np.ndarray,
    left_labels: np.ndarray,
    left_centers: np.ndarray,
    left_title: str,
    right_points: np.ndarray,
    right_labels: np.ndarray,
    right_centers: np.ndarray,
    right_title: str,
    explained_variance: np.ndarray,
    figure_title: str,
    output_path: Path,
) -> None:
    """Save a side-by-side comparison of two clustering results."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_cluster_panel(axes[0], left_points, left_labels, left_centers, explained_variance, left_title)
    plot_cluster_panel(axes[1], right_points, right_labels, right_centers, explained_variance, right_title)
    fig.suptitle(figure_title, fontsize=16, y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_research_style_cluster_result_pair(
    projected_points: np.ndarray,
    left_labels: np.ndarray,
    left_title: str,
    right_labels: np.ndarray,
    right_title: str,
    explained_variance: np.ndarray,
    figure_title: str,
    output_path: Path,
) -> None:
    """Save a research-style side-by-side comparison of two label assignments."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_research_style_cluster_panel(
        axes[0], projected_points, left_labels, explained_variance, left_title, show_ylabel=True
    )
    plot_research_style_cluster_panel(
        axes[1], projected_points, right_labels, explained_variance, right_title, show_ylabel=False
    )
    fig.suptitle(figure_title, fontsize=16, y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def build_cluster_label_counts_long_df(original_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Build a long-form table of crop-label counts for each cluster."""
    return (
        original_df.assign(kmeans_cluster=labels)
        .groupby(["kmeans_cluster", LABEL_COLUMN])
        .size()
        .rename("crop_count")
        .reset_index()
    )


def build_cluster_label_counts_summary_df(original_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Build a wide summary table of crop-label counts by cluster."""
    counts_long_df = build_cluster_label_counts_long_df(original_df, labels)
    counts_summary_df = (
        counts_long_df.pivot(index=LABEL_COLUMN, columns="kmeans_cluster", values="crop_count")
        .fillna(0)
        .astype(int)
    )
    counts_summary_df = counts_summary_df.reindex(sorted(counts_summary_df.columns), axis=1)
    counts_summary_df.columns = [f"cluster_{int(cluster_id)}_count" for cluster_id in counts_summary_df.columns]
    counts_summary_df["total_count"] = counts_summary_df.sum(axis=1)
    counts_summary_df = counts_summary_df.reset_index().rename(columns={LABEL_COLUMN: "crop_label"})
    counts_summary_df = counts_summary_df.sort_values(["total_count", "crop_label"], ascending=[False, True])
    return counts_summary_df


def plot_cluster_label_counts(
    original_df: pd.DataFrame,
    labels: np.ndarray,
    cluster_colors: dict[int, str],
    title: str,
    output_path: Path,
) -> None:
    """Plot crop-label counts for each cluster as horizontal bar charts."""
    counts_df = build_cluster_label_counts_long_df(original_df, labels)
    cluster_ids = sorted(int(cluster_id) for cluster_id in np.unique(labels))
    n_clusters = len(cluster_ids)
    ncols = 2 if n_clusters > 1 else 1
    nrows = int(np.ceil(n_clusters / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.8 * nrows), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    for axis, cluster_id in zip(axes, cluster_ids):
        cluster_counts = counts_df[counts_df["kmeans_cluster"] == cluster_id].copy()
        cluster_counts = cluster_counts.sort_values(["crop_count", LABEL_COLUMN], ascending=[False, True])

        axis.barh(
            cluster_counts[LABEL_COLUMN],
            cluster_counts["crop_count"],
            color=cluster_colors.get(cluster_id, "#1f77b4"),
            alpha=0.85,
        )
        axis.invert_yaxis()
        axis.set_title(f"Cluster {cluster_id}")
        axis.set_xlabel("Crop count")
        axis.grid(axis="x", alpha=0.3)

    for axis in axes[n_clusters:]:
        axis.axis("off")

    fig.suptitle(title, fontsize=16, y=0.99)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def save_pca_space_cluster_centers_csv(cluster_centers: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    """Save PCA-space centroids and their cluster sizes to CSV."""
    cluster_centers_df = pd.DataFrame(cluster_centers, columns=["pca_component_1", "pca_component_2"])
    cluster_centers_df.insert(0, "kmeans_cluster", np.arange(len(cluster_centers_df)))
    cluster_centers_df["cluster_size"] = np.bincount(labels, minlength=len(cluster_centers_df))
    cluster_centers_df.to_csv(output_path, index=False)


def save_diagnostic_tables(
    output_dir: Path,
    elbow_k_values: list[int],
    wcss_values: list[float],
    evaluation_k_values: list[int],
    silhouette_scores: list[float],
    elbow_k: int,
    silhouette_k: int,
    final_k: int,
    final_k_reason: str,
) -> None:
    """Save diagnostic metrics and the cluster-selection summary tables."""
    diagnostic_k_values = sorted(set(elbow_k_values) | set(evaluation_k_values))
    diagnostics_df = pd.DataFrame({"k": diagnostic_k_values})
    wcss_df = pd.DataFrame({"k": elbow_k_values, "wcss": wcss_values})
    silhouette_df = pd.DataFrame({"k": evaluation_k_values, "silhouette_score": silhouette_scores})

    diagnostics_df = diagnostics_df.merge(wcss_df, on="k", how="left")
    diagnostics_df = diagnostics_df.merge(silhouette_df, on="k", how="left")
    diagnostics_df.to_csv(output_dir / "cluster_diagnostics.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {"method": "elbow", "recommended_k": elbow_k, "selection_reason": "maximum curvature heuristic"},
            {"method": "silhouette", "recommended_k": silhouette_k, "selection_reason": "highest silhouette score"},
            {"method": "final", "recommended_k": final_k, "selection_reason": final_k_reason},
        ]
    )
    summary_df.to_csv(output_dir / "cluster_selection_summary.csv", index=False)


def save_cluster_centers_csv(
    cluster_centers_scaled: np.ndarray,
    labels: np.ndarray,
    scaler: StandardScaler,
    numeric_columns: list[str],
    output_path: Path,
) -> None:
    """Save inverse-transformed cluster centers in the original feature units."""
    cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
    cluster_centers_df = pd.DataFrame(cluster_centers, columns=numeric_columns)
    cluster_centers_df.insert(0, "kmeans_cluster", np.arange(len(cluster_centers_df)))
    cluster_centers_df["cluster_size"] = np.bincount(labels, minlength=len(cluster_centers_df))
    cluster_centers_df.to_csv(output_path, index=False)


def save_explored_cluster_centers(
    features_scaled: np.ndarray,
    scaler: StandardScaler,
    numeric_columns: list[str],
    k_values: list[int],
    random_state: int,
    output_dir: Path,
) -> None:
    """Save cluster centers for the explicitly compared `k` values."""
    for k in k_values:
        model = make_kmeans(n_clusters=k, random_state=random_state)
        labels = model.fit_predict(features_scaled)
        save_cluster_centers_csv(
            model.cluster_centers_,
            labels,
            scaler,
            numeric_columns,
            output_dir / f"cluster_centers_k{k}.csv",
        )


def fit_final_model(
    original_df: pd.DataFrame,
    numeric_columns: list[str],
    features_scaled: np.ndarray,
    scaler: StandardScaler,
    final_k: int,
    random_state: int,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the final KMeans model and save the labelled dataset plus summaries."""
    model = make_kmeans(n_clusters=final_k, random_state=random_state)
    labels = model.fit_predict(features_scaled)

    clustered_df = original_df.copy()
    clustered_df["kmeans_cluster"] = labels
    clustered_df.to_csv(output_dir / "crops_kmeans_clusters.csv", index=False)

    save_cluster_centers_csv(model.cluster_centers_, labels, scaler, numeric_columns, output_dir / "cluster_centers.csv")

    cluster_size_df = pd.DataFrame(
        {"kmeans_cluster": np.arange(final_k), "cluster_size": np.bincount(labels, minlength=final_k)}
    )
    cluster_size_df.to_csv(output_dir / "cluster_sizes.csv", index=False)

    return labels, model.cluster_centers_


def main() -> None:
    """Run the full KMeans analysis workflow and save all outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    obsolete_files = [
        OUTPUT_DIR / "kmeans_comparison_k2_k4.png",
        OUTPUT_DIR / "kmeans_cluster_plot.png",
    ]
    for obsolete_file in obsolete_files:
        if obsolete_file.exists():
            obsolete_file.unlink()

    original_df, numeric_df, numeric_columns = load_dataset(DATA_PATH)
    features_scaled, scaler = standardize_features(numeric_df)
    validate_standardized_features(features_scaled)
    projected_points, pca_model = project_features_to_2d(features_scaled)
    explained_variance = pca_model.explained_variance_ratio_

    cumulative_variance = pca_cumulative_variance(features_scaled)
    explored_k_values = resolve_k_values(list(range(DEFAULT_MIN_K, DEFAULT_MAX_K + 1)), len(original_df))
    comparison_k_values = resolve_k_values(COMPARISON_K_VALUES, len(original_df))

    wcss_values = compute_wcss(features_scaled, explored_k_values, DEFAULT_RANDOM_STATE)
    elbow_k = detect_elbow(explored_k_values, wcss_values, DEFAULT_MIN_K)

    silhouette_scores, silhouette_k = compute_silhouette_scores(features_scaled, explored_k_values, DEFAULT_RANDOM_STATE)

    final_k, final_k_reason = choose_final_k(elbow_k=elbow_k, silhouette_k=silhouette_k)

    plot_pca_cumulative_variance(cumulative_variance, OUTPUT_DIR / "pca_cumulative_variance.png")
    plot_elbow_method(explored_k_values, wcss_values, elbow_k, OUTPUT_DIR / "elbow_method.png")
    plot_silhouette_method(explored_k_values, silhouette_scores, silhouette_k, OUTPUT_DIR / "silhouette_method.png")

    standardized_labels_k2, standardized_centers_k2 = fit_standardized_kmeans(
        features_scaled, comparison_k_values[0], DEFAULT_RANDOM_STATE
    )
    standardized_labels_k4, standardized_centers_k4 = fit_standardized_kmeans(
        features_scaled, comparison_k_values[1], DEFAULT_RANDOM_STATE
    )
    projected_centers_k2 = pca_model.transform(standardized_centers_k2)
    projected_centers_k4 = pca_model.transform(standardized_centers_k4)
    standardized_k4_display_map = build_display_cluster_map(projected_centers_k4)
    standardized_labels_k4, standardized_centers_k4, projected_centers_k4 = remap_cluster_outputs(
        standardized_labels_k4,
        standardized_centers_k4,
        projected_centers_k4,
        standardized_k4_display_map,
    )

    pca_space_labels_k4, pca_space_centers_k4 = fit_pca_space_kmeans(
        projected_points, comparison_k_values[1], DEFAULT_RANDOM_STATE
    )
    pca_space_display_map = build_display_cluster_map(pca_space_centers_k4)
    pca_space_labels_k4, pca_space_centers_k4, _ = remap_cluster_outputs(
        pca_space_labels_k4,
        pca_space_centers_k4,
        pca_space_centers_k4,
        pca_space_display_map,
    )

    plot_single_cluster_result(
        projected_points,
        standardized_labels_k2,
        projected_centers_k2,
        explained_variance,
        "KMeans in Standardized 4D Space (k = 2)",
        OUTPUT_DIR / "kmeans_standardized_k2.png",
    )
    plot_research_style_cluster_result(
        projected_points,
        standardized_labels_k4,
        explained_variance,
        "KMeans in Standardized 4D Space (k = 4)",
        OUTPUT_DIR / "kmeans_standardized_k4.png",
    )
    plot_research_style_cluster_result_pair(
        projected_points,
        standardized_labels_k4,
        "Standardized 4D Space (k = 4)",
        pca_space_labels_k4,
        "Projected PCA Space (k = 4)",
        explained_variance,
        "KMeans Comparison: Standardized 4D Space vs PCA Space",
        OUTPUT_DIR / "kmeans_standardized_k4_vs_pca_k4.png",
    )
    plot_cluster_result_pair(
        projected_points,
        standardized_labels_k4,
        projected_centers_k4,
        "Standardized 4D Space (k = 4)",
        projected_points,
        standardized_labels_k2,
        projected_centers_k2,
        "Standardized 4D Space (k = 2)",
        explained_variance,
        "KMeans Comparison: Standardized 4D Space k = 4 vs k = 2",
        OUTPUT_DIR / "kmeans_standardized_k4_vs_standardized_k2.png",
    )
    if LABEL_COLUMN in original_df.columns:
        k2_cluster_colors = assign_comparison_cluster_colors(projected_centers_k2)
        k4_cluster_colors = {cluster_id: PAPER_COLORS[cluster_id] for cluster_id in range(len(projected_centers_k4))}
        k2_label_counts_df = build_cluster_label_counts_summary_df(original_df, standardized_labels_k2)

        plot_cluster_label_counts(
            original_df,
            standardized_labels_k2,
            k2_cluster_colors,
            "Crop Counts by Cluster for KMeans in Standardized 4D Space (k = 2)",
            OUTPUT_DIR / "cluster_label_counts_k2.png",
        )
        plot_cluster_label_counts(
            original_df,
            standardized_labels_k4,
            k4_cluster_colors,
            "Crop Counts by Cluster for KMeans in Standardized 4D Space (k = 4)",
            OUTPUT_DIR / "cluster_label_counts_k4.png",
        )

    save_diagnostic_tables(
        output_dir=OUTPUT_DIR,
        elbow_k_values=explored_k_values,
        wcss_values=wcss_values,
        evaluation_k_values=explored_k_values,
        silhouette_scores=silhouette_scores,
        elbow_k=elbow_k,
        silhouette_k=silhouette_k,
        final_k=final_k,
        final_k_reason=final_k_reason,
    )
    save_explored_cluster_centers(
        features_scaled=features_scaled,
        scaler=scaler,
        numeric_columns=numeric_columns,
        k_values=comparison_k_values,
        random_state=DEFAULT_RANDOM_STATE,
        output_dir=OUTPUT_DIR,
    )
    save_pca_space_cluster_centers_csv(
        pca_space_centers_k4,
        pca_space_labels_k4,
        OUTPUT_DIR / "cluster_centers_pca_space_k4.csv",
    )
    labels, cluster_centers = fit_final_model(
        original_df=original_df,
        numeric_columns=numeric_columns,
        features_scaled=features_scaled,
        scaler=scaler,
        final_k=final_k,
        random_state=DEFAULT_RANDOM_STATE,
        output_dir=OUTPUT_DIR,
    )

    pca_80 = components_to_reach(cumulative_variance, 0.80)
    pca_90 = components_to_reach(cumulative_variance, 0.90)

    print("\nKMeans analysis completed ✅.")
    print("-" * 40)
    print(f"Input dataset: {DATA_PATH}")
    print(f"Rows analysed: {len(original_df)}")
    print(f"Numeric features used: {', '.join(numeric_columns)}")
    print("Feature preprocessing for KMeans: standardized with StandardScaler")
    print(f"Explored k values for diagnostics: {', '.join(str(k) for k in explored_k_values)}")
    print(f"Explicit k values used for final standardized-space plots: {', '.join(str(k) for k in comparison_k_values)}")
    print("Explicit PCA-space clustering implemented for the k = 4 comparison plot")
    print(f"Principal components needed for 80% variance: {pca_80}")
    print(f"Principal components needed for 90% variance: {pca_90}")
    print(f"Recommended k by elbow method: {elbow_k}")
    print(f"Recommended k by silhouette method: {silhouette_k}")
    print(f"Final k used for KMeans: {final_k} ({final_k_reason})")
    if LABEL_COLUMN in original_df.columns:
        print(f"Crop-count plots were generated from the '{LABEL_COLUMN}' column for k = 2 and k = 4.")
        print("\nCrop counts by label for standardized-space KMeans (k = 2):")
        print(k2_label_counts_df.to_string(index=False))
    else:
        print(f"Crop-count plots were skipped because the '{LABEL_COLUMN}' column is not present.")
    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")
    print("Generated files:")
    print("  - pca_cumulative_variance.png")
    print("  - elbow_method.png")
    print("  - silhouette_method.png")
    print("  - kmeans_standardized_k2.png")
    print("  - kmeans_standardized_k4.png")
    print("  - kmeans_standardized_k4_vs_pca_k4.png")
    print("  - kmeans_standardized_k4_vs_standardized_k2.png")
    if LABEL_COLUMN in original_df.columns:
        print("  - cluster_label_counts_k2.png")
        print("  - cluster_label_counts_k4.png")
    print("  - cluster_diagnostics.csv")
    print("  - cluster_selection_summary.csv")
    print("  - crops_kmeans_clusters.csv")
    print("  - cluster_centers.csv")
    print("  - cluster_centers_k2.csv")
    print("  - cluster_centers_k4.csv")
    print("  - cluster_centers_pca_space_k4.csv")
    print("  - cluster_sizes.csv\n")

if __name__ == "__main__":
    main()
