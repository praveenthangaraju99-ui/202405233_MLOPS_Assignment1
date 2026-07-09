import json
import joblib
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from constants import PROCESSED_DIR, MODELS_DIR, RANDOM_STATE
from preprocessing import build_preprocessor, split_features_target


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def plot_artifacts(model, X_test, y_test, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    roc_path = out_dir / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=160)
    plt.close()

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Confusion Matrix")
    cm_path = out_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=160)
    plt.close()
    return roc_path, cm_path


def train_models(df):
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        "logistic_regression": (
            LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE), {
                "model__C": [
                    0.1, 1.0, 10.0], "model__class_weight": [
                        None, "balanced"]}, ), "random_forest": (
                            RandomForestClassifier(
                                random_state=RANDOM_STATE), {
                                    "model__n_estimators": [
                                        100, 300], "model__max_depth": [
                                            None, 4, 8], "model__min_samples_split": [
                                                2, 5], }, ), }

    # SQLite is used because it is stable on Windows and works with current
    # MLflow versions.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("heart-disease-classification")

    results = []
    best = None
    for name, (estimator, param_grid) in candidates.items():
        pipe = Pipeline(
            [("preprocessor", build_preprocessor()), ("model", estimator)])
        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True)

        with mlflow.start_run(run_name=name):
            print(f"Training {name}...")
            grid.fit(X_train, y_train)
            metrics = evaluate(grid.best_estimator_, X_test, y_test)
            artifact_dir = MODELS_DIR / "artifacts" / name
            roc_path, cm_path = plot_artifacts(
                grid.best_estimator_, X_test, y_test, artifact_dir)

            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("cv_best_roc_auc", float(grid.best_score_))
            for metric_name, metric_value in metrics.items():
                if metric_name != "confusion_matrix":
                    mlflow.log_metric(metric_name, float(metric_value))

            mlflow.log_artifact(str(roc_path))
            mlflow.log_artifact(str(cm_path))

            model_path = MODELS_DIR / f"{name}_model.joblib"
            joblib.dump(grid.best_estimator_, model_path)
            mlflow.log_artifact(str(model_path))

            result = {
                "model_name": name,
                "best_params": grid.best_params_,
                "cv_roc_auc": float(
                    grid.best_score_),
                **metrics}
            results.append(result)
            if best is None or result["roc_auc"] > best["roc_auc"]:
                best = {**result, "estimator": grid.best_estimator_}

    return best, results


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_DIR / "heart_disease_clean.csv")
    best, results = train_models(df)
    joblib.dump(best["estimator"], MODELS_DIR / "final_model.joblib")
    serializable_best = {k: v for k, v in best.items() if k != "estimator"}
    output = {"best_model": serializable_best, "all_results": results}
    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2)
    print("\nTraining completed successfully.\n")
    print(json.dumps(serializable_best, indent=2))


if __name__ == "__main__":
    main()
