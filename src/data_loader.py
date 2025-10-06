from pathlib import Path
import pandas as pd
from pandas import DataFrame


def load_boston_housing_data(csv_path: str = 'data/HousingData.csv') -> DataFrame:
    """
    Load Boston Housing dataset from local CSV.
    Args:
        csv_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded dataset with columns renamed (MEDV -> PRICE).
    Raises:
        FileNotFoundError: If the CSV file does not exist at the given path.
    """

    print("Loading Boston Housing dataset...")

    path = Path(csv_path)
    if path.exists():
        # Cargar CSV de Kaggle
        df = pd.read_csv(path)

        # Renombrar MEDV a PRICE para consistencia
        if 'MEDV' in df.columns:
            df = df.rename(columns={'MEDV': 'PRICE'})

        # Revisar valores faltantes
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            print(
                f"Warning: There are {missing_values} missing values in the dataset.")
        else:
            print("No missing values found.")

        print(f"Dataset loaded: {len(df)} samples")
        print(f"Features: {list(df.columns[:-1])}")
        return df
    else:
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}")


if __name__ == "__main__":
    df = load_boston_housing_data()
    print("\nFirst 5 rows:")
    print(df.head())
