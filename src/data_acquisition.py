import pandas as pd
from ucimlrepo import fetch_ucirepo
from constants import RAW_DIR, PROCESSED_DIR, TARGET_COLUMN


def fetch_heart_disease():
    """Fetch the UCI Heart Disease dataset using the assignment-specified method."""
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features.copy()
    y = heart_disease.data.targets.copy()
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    df = X.copy()
    df[TARGET_COLUMN] = y
    return df, heart_disease.metadata, heart_disease.variables


def clean_raw_dataframe(df):
    cleaned = df.copy().replace("?", pd.NA)
    for col in cleaned.columns:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned[TARGET_COLUMN] = (cleaned[TARGET_COLUMN] > 0).astype(int)
    return cleaned


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df, metadata, variables = fetch_heart_disease()
    df.to_csv(RAW_DIR / "heart_disease_raw.csv", index=False)
    clean_raw_dataframe(df).to_csv(PROCESSED_DIR / "heart_disease_clean.csv", index=False)
    print("Dataset downloaded and cleaned.")
    print(metadata)
    print(variables)


if __name__ == "__main__":
    main()
