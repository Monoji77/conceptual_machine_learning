"""
Summary
-------
This script loads `data/crops.csv`, removes rows with missing values, filters
numeric outliers using a z-score threshold, and saves both the cleaned dataset
and a cleaning summary to the `data` directory.
"""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/crops.csv")
OUTPUT_DIR = Path("data")
CLEANED_PATH = OUTPUT_DIR / "crops_cleaned.csv"
SUMMARY_PATH = OUTPUT_DIR / "crops_cleaning_summary.csv"
Z_THRESHOLD = 3.0


def main() -> None:
    """Clean the crop dataset and save both the filtered data and summary table."""
    df = pd.read_csv(DATA_PATH)
    numeric_columns = df.select_dtypes(include="number").columns

    if numeric_columns.empty:
        raise ValueError("No numeric columns were found in data/crops.csv.")

    rows_before_cleaning = len(df)

    df_no_missing = df.dropna().copy()
    missing_removed = rows_before_cleaning - len(df_no_missing)

    if df_no_missing.empty:
        raise ValueError("All rows were removed after dropping missing values.")

    z_scores = (df_no_missing[numeric_columns] - df_no_missing[numeric_columns].mean()) / df_no_missing[
        numeric_columns
    ].std(ddof=1)


    keep_mask = z_scores.abs().le(Z_THRESHOLD).all(axis=1)
    cleaned_df = df_no_missing.loc[keep_mask].copy()
    outlier_removed = len(df_no_missing) - len(cleaned_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_csv(CLEANED_PATH, index=False)

    summary_df = pd.DataFrame(
        [
            {"metric": "original_rows", "value": rows_before_cleaning},
            {"metric": "rows_removed_missing_values", "value": missing_removed},
            {"metric": "rows_after_missing_value_deletion", "value": len(df_no_missing)},
            {"metric": "z_score_threshold", "value": Z_THRESHOLD},
            {"metric": "rows_removed_as_outliers", "value": outlier_removed},
            {"metric": "final_rows", "value": len(cleaned_df)},
        ]
    )
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("\nCrop data cleaning completed.")
    print("-" * 40)
    print(f"Original rows: {rows_before_cleaning}")
    print(f"Rows removed for missing values: {missing_removed}")
    print(f"Rows remaining after missing-value deletion: {len(df_no_missing)}")
    print(f"Rows removed as outliers (|z| > {Z_THRESHOLD:.2f}): {outlier_removed}")
    print(f"Final cleaned rows: {len(cleaned_df)}")
    print(f"\nSaved cleaned dataset to {CLEANED_PATH}")
    print(f"Saved cleaning summary to {SUMMARY_PATH}\n")


if __name__ == "__main__":
    main()
