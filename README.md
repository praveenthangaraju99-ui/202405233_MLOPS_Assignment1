# Heart Disease Risk Prediction - MLOps Assignment 01

Author: PRAVEEN T.2024ad05233

This is my MLOps assignment project for predicting heart disease risk from the UCI Heart Disease dataset. I kept the implementation practical and explainable: a reusable sklearn pipeline, two models, MLflow tracking, FastAPI serving, Docker, CI/CD, Kubernetes manifests, and simple monitoring.

## Why I designed it this way

- I used `ucimlrepo` because the assignment gave that dataset import method.
- I used Logistic Regression as an interpretable baseline.
- I used Random Forest to capture non-linear patterns.
- I used a sklearn `ColumnTransformer` so preprocessing is consistent during training and inference.
- I used MLflow with SQLite because it works reliably on Windows with Python 3.13.
- I used FastAPI because Swagger UI makes the API easy to test and screenshot.

## Setup on Windows with Python 3.13

```powershell
python -m venv mlops
.\mlops\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install --prefer-binary -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Run the pipeline

```powershell
python src\data_acquisition.py
python src\eda.py
python src\ preprocessing.py
python src\train.py
```

## MLflow

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open: http://127.0.0.1:5000

## Tests

```powershell
pytest -q
```

## FastAPI

```powershell
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open: http://localhost:8000/docs

PowerShell API test:

```powershell
curl.exe -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d "@sample_input.json"
```

Swagger UI: http://localhost:8000/docs 
Health Endpoint: http://localhost:8000/health 
Metrics Endpoint: http://localhost:8000/metrics


## Docker

```powershell
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```

Open: http://localhost:8000/metrics

## Kubernetes

```powershell
kubectl get nodes
Build Docker Image: docker build -t heart-disease-api:latest .
docker images
kubectl apply -f k8s\deployment.yaml
kubectl apply -f k8s\service.yaml
kubectl get pods
kubectl get svc
kubectl port-forward service/heart-disease-service 8000:80
```

## Monitoring

With the API running, open:

```text
docker run -d   --name prometheus   -p 9090:9090   prom/prometheus
docker run -d --name prometheus -p 9090:9090 prom/Prometheus
docker run -d   --name grafana   -p 3000:3000   grafana/Grafana
```
Open: http://localhost:8000/metrics

Prometheus UI: http://localhost:9090
http://localhost:9090/targets

Graphana UI: http://localhost:3000
http://host.docker.internal:9090