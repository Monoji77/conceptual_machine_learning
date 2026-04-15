"""
Summary
-------
This script downloads the Kaggle crop recommendation dataset and saves the
source CSV as `data/crops.csv` for the rest of the project scripts.
"""

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

def save_csv():
    """Download the Kaggle crop dataset and save it to `data/crops.csv`."""
    # Load a DataFrame with a specific version of a CSV
    df = kagglehub.dataset_load(
        adapter=KaggleDatasetAdapter.PANDAS,
        handle="atharvaingle/crop-recommendation-dataset",
        path="Crop_recommendation.csv",
    )

    # save the csv file 
    df.to_csv("data/crops.csv", index=False)

    print("\nSUCCESSFUL ✅\nDataset downloaded and saved to data/crops.csv\n")


save_csv()
