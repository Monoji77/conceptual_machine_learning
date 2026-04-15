"""
Summary
-------
This script loads `data/crops.csv`, selects the four crop-soil features `N`,
`P`, `K`, and `ph`, standardizes them, and performs principal component
analysis (PCA).

It generates two plots in sequential order:

1. a static scatter plot of PC2 against PC1
2. an interactive 3D scatter plot of PC1, PC2, and PC3

The plots are saved to `outputs/pca_analysis` and then shown with blocking
`plt.show()` calls so the next plot does not appear until the current plot is
closed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/crops.csv")
OUTPUT_DIR = Path("outputs/pca_analysis")
LABEL_COLUMN = "label"
FEATURE_COLUMNS = ["N", "P", "K", "ph"]


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load the dataset and validate the four PCA feature columns."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    missing_columns = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Required PCA feature columns are missing from {data_path}: {', '.join(missing_columns)}")

    non_numeric_columns = [column for column in FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric_columns:
        raise ValueError(f"These PCA feature columns must be numeric in {data_path}: {', '.join(non_numeric_columns)}")

    numeric_columns = FEATURE_COLUMNS.copy()
    numeric_df = df[numeric_columns].copy()
    if numeric_df.isna().any().any():
        raise ValueError(f"Missing numeric values were found in {data_path}; clean the data before PCA.")

    return df, numeric_df, numeric_columns


def standardize_features(numeric_df: pd.DataFrame) -> np.ndarray:
    """Standardize the selected PCA features with `StandardScaler`."""
    scaler = StandardScaler()
    return scaler.fit_transform(numeric_df)


def compute_pca(features_scaled: np.ndarray, n_components: int) -> tuple[np.ndarray, PCA]:
    """Fit PCA and return the transformed scores plus the fitted PCA model."""
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(features_scaled)
    return scores, pca


def build_pca_scores_df(df: pd.DataFrame, scores: np.ndarray, label_column: str) -> pd.DataFrame:
    """Build a PCA score table and include labels when that column exists."""
    pca_scores_df = pd.DataFrame(
        scores[:, :3],
        columns=["PC1", "PC2", "PC3"],
    )
    if label_column in df.columns:
        pca_scores_df.insert(0, label_column, df[label_column])
    return pca_scores_df


def build_explained_variance_df(pca: PCA) -> pd.DataFrame:
    """Build a per-component explained-variance summary table."""
    components = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    return pd.DataFrame(
        {
            "principal_component": components,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": cumulative_variance,
        }
    )


def plot_pc2_against_pc1(scores: np.ndarray, explained_variance: np.ndarray, output_path: Path) -> None:
    """Plot and save the two-dimensional PCA scatter plot."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        scores[:, 0],
        scores[:, 1],
        s=28,
        alpha=0.75,
        color="#1f77b4",
        edgecolors="none",
    )
    ax.set_title("PCA Scatter Plot: PC2 against PC1")
    ax.set_xlabel(f"PC1 ({explained_variance[0]:.1%} variance explained)")
    ax.set_ylabel(f"PC2 ({explained_variance[1]:.1%} variance explained)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_pc1_pc2_pc3_interactive(scores: np.ndarray, explained_variance: np.ndarray, output_path: Path) -> None:
    """Plot and save the three-dimensional PCA scatter plot."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        scores[:, 0],
        scores[:, 1],
        scores[:, 2],
        c=scores[:, 0],
        cmap="viridis",
        s=26,
        alpha=0.75,
        depthshade=True,
    )
    ax.set_title("Interactive 3D PCA Plot: PC1, PC2, PC3")
    ax.set_xlabel(f"PC1 ({explained_variance[0]:.1%})")
    ax.set_ylabel(f"PC2 ({explained_variance[1]:.1%})")
    ax.set_zlabel(f"PC3 ({explained_variance[2]:.1%})")
    fig.colorbar(scatter, ax=ax, pad=0.1, label="PC1 score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def main() -> None:
    """Run the PCA workflow and save its plots plus summary tables."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original_df, numeric_df, numeric_columns = load_dataset(DATA_PATH)
    features_scaled = standardize_features(numeric_df)
    scores, pca = compute_pca(features_scaled, n_components=len(numeric_columns))

    pca_scores_df = build_pca_scores_df(original_df, scores, LABEL_COLUMN)
    explained_variance_df = build_explained_variance_df(pca)

    pca_scores_df.to_csv(OUTPUT_DIR / "pca_scores.csv", index=False)
    explained_variance_df.to_csv(OUTPUT_DIR / "pca_explained_variance.csv", index=False)

    plot_pc2_against_pc1(
        scores,
        pca.explained_variance_ratio_,
        OUTPUT_DIR / "pca_pc2_vs_pc1.png",
    )
    plot_pc1_pc2_pc3_interactive(
        scores,
        pca.explained_variance_ratio_,
        OUTPUT_DIR / "pca_pc1_pc2_pc3_3d.png",
    )

    print("\nPCA analysis completed.")
    print("-" * 40)
    print(f"Input dataset: {DATA_PATH}")
    print(f"Rows analysed: {len(original_df)}")
    print(f"Numeric features used: {', '.join(numeric_columns)}")
    print("Feature preprocessing for PCA: standardized with StandardScaler")
    print(f"Explained variance by PC1: {pca.explained_variance_ratio_[0]:.4f}")
    print(f"Explained variance by PC2: {pca.explained_variance_ratio_[1]:.4f}")
    print(f"Explained variance by PC3: {pca.explained_variance_ratio_[2]:.4f}")
    print(
        "Cumulative variance by first three PCs: "
        f"{np.cumsum(pca.explained_variance_ratio_[:3])[-1]:.4f}"
    )
    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")
    print("SUCCESSFULLY SAVED FILES ✅:")
    print("  - pca_pc2_vs_pc1.png")
    print("  - pca_pc1_pc2_pc3_3d.png")
    print("  - pca_scores.csv")
    print("  - pca_explained_variance.csv\n")


if __name__ == "__main__":
    main()
