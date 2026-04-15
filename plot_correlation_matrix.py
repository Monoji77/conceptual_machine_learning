"""
Summary
-------
This script loads `data/crops.csv`, keeps the numeric columns, computes
their correlation matrix, and saves two outputs in `outputs/correlation_matrix`:

1. an annotated correlation-matrix heatmap,
2. a CSV copy of the correlation matrix plus a pairwise ranking table.

To make weak correlations easier to see, the heatmap colours use a signed
power transform that exaggerates small magnitudes. The numbers written in each
cell and the exported CSV files always contain the true correlation
coefficients.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path("data/crops.csv")
OUTPUT_DIR = Path("outputs/correlation_matrix")
DEFAULT_METHOD = "pearson"
COLOR_SCALE_EXPONENT = 0.25
COLORBAR_TICKS = np.array([-1.0, -0.75, -0.5, -0.25, -0.10, 0.0, 0.10, 0.25, 0.5, 0.75, 1.0])


def load_numeric_dataset(data_path: Path) -> pd.DataFrame:
    """Load the dataset and return only validated numeric columns."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    numeric_df = df.select_dtypes(include="number").copy()

    if numeric_df.shape[1] < 2:
        raise ValueError(f"At least two numeric columns are required in {data_path} to compute correlations.")
    if numeric_df.isna().any().any():
        raise ValueError(f"Missing numeric values were found in {data_path}; clean the data before plotting.")

    return numeric_df


def compute_correlation_matrix(numeric_df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Compute the correlation matrix for the requested method."""
    correlation_df = numeric_df.corr(method=method)
    if correlation_df.isna().any().any():
        raise ValueError("The correlation matrix contains missing values; check the input columns.")
    return correlation_df


def signed_power_scale(values: np.ndarray, exponent: float) -> np.ndarray:
    """Apply a signed power transform that boosts small magnitudes for plotting."""
    return np.sign(values) * np.abs(values) ** exponent


def save_pairwise_correlations(correlation_df: pd.DataFrame, output_path: Path) -> None:
    """Save a ranked table of pairwise correlations to CSV."""
    columns = correlation_df.columns.tolist()
    rows: list[dict[str, float | str]] = []

    for left_index, left_name in enumerate(columns):
        for right_index in range(left_index + 1, len(columns)):
            right_name = columns[right_index]
            correlation = float(correlation_df.iloc[left_index, right_index])
            rows.append(
                {
                    "feature_1": left_name,
                    "feature_2": right_name,
                    "correlation": correlation,
                    "absolute_correlation": abs(correlation),
                }
            )

    pairs_df = pd.DataFrame(rows).sort_values(
        by=["absolute_correlation", "feature_1", "feature_2"], ascending=[True, True, True]
    )
    pairs_df.to_csv(output_path, index=False)


def plot_correlation_matrix(correlation_df: pd.DataFrame, method: str, output_path: Path) -> None:
    """Plot and save the annotated correlation heatmap."""
    labels = correlation_df.columns.tolist()
    correlation_values = correlation_df.to_numpy()
    color_values = signed_power_scale(correlation_values, COLOR_SCALE_EXPONENT)

    figure_size = max(8.0, len(labels) * 1.35)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    image = ax.imshow(color_values, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_title(
        f"{method.title()} Correlation Matrix\nColour scale strongly expands weak correlations for readability",
        pad=16,
    )
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_index in range(len(labels)):
        for column_index in range(len(labels)):
            display_value = correlation_values[row_index, column_index]
            scaled_value = color_values[row_index, column_index]
            text_color = "white" if abs(scaled_value) >= 0.7 else "black"
            ax.text(
                column_index,
                row_index,
                f"{display_value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold" if row_index == column_index else "normal",
            )

    colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar_positions = signed_power_scale(COLORBAR_TICKS, COLOR_SCALE_EXPONENT)
    colorbar.set_ticks(colorbar_positions)
    colorbar.set_ticklabels([f"{tick:.2f}" for tick in COLORBAR_TICKS])
    colorbar.set_label(f"{method.title()} correlation coefficient")

    plt.figtext(
        0.5,
        0.015,
        "Cell labels show the true coefficients; colours are strongly rescaled so small magnitudes stand out.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def main() -> None:
    """Run the correlation analysis and save the plot plus CSV outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    numeric_df = load_numeric_dataset(DATA_PATH)
    correlation_df = compute_correlation_matrix(numeric_df, DEFAULT_METHOD)

    correlation_matrix_path = OUTPUT_DIR / "correlation_matrix.csv"
    correlation_pairs_path = OUTPUT_DIR / "correlation_pairs_sorted_by_absolute_value.csv"
    correlation_plot_path = OUTPUT_DIR / "correlation_matrix.png"

    correlation_df.to_csv(correlation_matrix_path)
    save_pairwise_correlations(correlation_df, correlation_pairs_path)
    plot_correlation_matrix(correlation_df, DEFAULT_METHOD, correlation_plot_path)

    print("\nCorrelation analysis completed✅.")
    print("-" * 40)
    print(f"Input dataset: {DATA_PATH}")
    print(f"Rows analysed: {len(numeric_df)}")
    print(f"Numeric features used: {', '.join(numeric_df.columns)}")
    print(f"Correlation method: {DEFAULT_METHOD}")
    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")
    print("Generated files:")
    print("  - correlation_matrix.png")
    print("  - correlation_matrix.csv")
    print("  - correlation_pairs_sorted_by_absolute_value.csv\n")


if __name__ == "__main__":
    main()
