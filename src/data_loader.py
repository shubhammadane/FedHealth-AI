from pathlib import Path
import shutil

import pandas as pd
from ucimlrepo import fetch_ucirepo

from .config import RAW_DATA_PATH
from .utils import LOGGER


def load_uci_dataset(cache: bool = True):
    """Fetch UCI dataset 296 and cache the feature/target data locally."""
    try:
        LOGGER.info("Fetching UCI Diabetes 130-US Hospitals dataset (ID 296)...")
        dataset = fetch_ucirepo(id=296)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()

        if "readmitted" not in y.columns:
            raise ValueError("UCI target column 'readmitted' was not returned.")

        combined = X.copy()
        combined["readmitted"] = y["readmitted"].values

        if cache:
            RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(RAW_DATA_PATH, index=False)

        LOGGER.info("Dataset loaded successfully")
        LOGGER.info("Features shape: %s", X.shape)
        LOGGER.info("Target shape: %s", y.shape)
        print("Dataset Metadata")
        print(getattr(dataset, "metadata", {}))
        print("\nVariable Information")
        print(getattr(dataset, "variables", pd.DataFrame()).to_string(index=False))

        return X, y
    except Exception as exc:
        raise RuntimeError(
            "Unable to fetch UCI dataset ID 296. Check internet access and the "
            "ucimlrepo installation. If needed, obtain the UCI dataset manually "
            "and place a compatible diabetic_data.csv in data/raw/."
        ) from exc


def load_cached_raw() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_DATA_PATH} does not exist. Run `python -m src.data_loader` first."
        )
    return pd.read_csv(RAW_DATA_PATH)


def main():
    X, y = load_uci_dataset(cache=True)
    print("\nDataset loaded successfully")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print("\nTarget distribution:")
    print(y["readmitted"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
