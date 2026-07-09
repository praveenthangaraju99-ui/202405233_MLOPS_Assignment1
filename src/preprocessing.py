from typing import List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from constants import TARGET_COLUMN

CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col != TARGET_COLUMN]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def split_features_target(df: pd.DataFrame):
    X = df[get_feature_columns(df)]
    y = df[TARGET_COLUMN].astype(int)
    return X, y
