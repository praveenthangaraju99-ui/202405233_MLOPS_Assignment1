# Heart Disease Risk Prediction - MLOps Assignment Report

**Student:** PRAVEEN T.  
**Assignment:** End-to-End ML Model Development, CI/CD, Deployment, and Monitoring

## 1. Project Overview

I developed an end-to-end MLOps workflow for binary heart disease risk prediction. The workflow includes dataset acquisition, EDA, feature preprocessing, model training, experiment tracking, model packaging, API serving, containerization, CI/CD, Kubernetes deployment manifests, and monitoring.

## 2. Dataset Acquisition

The dataset is the UCI Heart Disease dataset. I used the `ucimlrepo` package with dataset id 45, as required in the assignment. The original diagnosis target is converted to a binary target: 0 for no disease and 1 for heart disease present.

## 3. EDA

The EDA script generates a class balance plot, feature histograms, a correlation heatmap, and missing value analysis. I will include screenshots from my own run in the final submission.

## 4. Preprocessing

The preprocessing pipeline uses median imputation and scaling for numeric features, and most-frequent imputation with one-hot encoding for categorical features. This pipeline is saved together with the trained model, so inference uses the same transformations as training.

## 5. Model Development

I trained Logistic Regression and Random Forest models. Logistic Regression is used as a simple baseline, while Random Forest is included to capture non-linear relationships. I used GridSearchCV with stratified 5-fold cross-validation and ROC-AUC scoring.

## 6. Experiment Tracking

MLflow tracks the parameters, metrics, ROC curves, confusion matrices, and joblib model artifacts. I used SQLite as the MLflow backend because it works reliably on Windows and avoids file-store compatibility issues.

## 7. Model Packaging

The best final pipeline is saved as `models/final_model.joblib`. This includes preprocessing and the final classifier, making the model reusable during API inference.

## 8. API Serving

The FastAPI application exposes `/health`, `/predict`, and `/metrics`. The `/predict` endpoint accepts JSON input and returns the predicted class with confidence.

## 9. CI/CD

The GitHub Actions workflow installs dependencies, runs linting, runs unit tests, executes data acquisition, generates EDA plots, trains the model, and uploads artifacts.

## 10. Containerization and Deployment

The Dockerfile creates a production-like container for the API. Kubernetes deployment and service files are provided for local deployment using Minikube or Docker Desktop Kubernetes.

## 11. Monitoring

The API logs prediction requests and exposes Prometheus metrics. Prometheus and Grafana starter configurations are included in the `monitoring/` folder.

## 12. Commands Used

```powershell
python -m venv mlops
.\mlops\Scripts\Activate.ps1
pip install --prefer-binary -r requirements.txt
python src\data_acquisition.py
python src\eda.py
python src\train.py
pytest -q
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 13. Final Items to Add

- GitHub repository link: TODO
- Deployment/local API URL: TODO
- Demo video link: TODO
- Screenshots: TODO
