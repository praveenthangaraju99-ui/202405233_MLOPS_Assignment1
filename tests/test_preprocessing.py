import pandas as pd
from preprocessing import build_preprocessor, split_features_target


def test_split_features_target():
    df = pd.DataFrame({
        "age": [50], "sex": [1], "cp": [2], "trestbps": [120], "chol": [200],
        "fbs": [0], "restecg": [1], "thalach": [150], "exang": [0], "oldpeak": [1.0],
        "slope": [1], "ca": [0], "thal": [2], "target": [1]
    })
    X, y = split_features_target(df)
    assert "target" not in X.columns
    assert int(y.iloc[0]) == 1


def test_preprocessor_handles_missing_values():
    df = pd.DataFrame({
        "age": [50, 60], "sex": [1, 0], "cp": [2, 3], "trestbps": [120, None], "chol": [200, 220],
        "fbs": [0, 1], "restecg": [1, 0], "thalach": [150, 140], "exang": [0, 1], "oldpeak": [1.0, 2.0],
        "slope": [1, 2], "ca": [0, 1], "thal": [2, 3]
    })
    transformed = build_preprocessor().fit_transform(df)
    assert transformed.shape[0] == 2
