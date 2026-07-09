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

## Docker

```powershell
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```

## Kubernetes

```powershell
minikube start
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods
kubectl get svc
minikube service heart-disease-service --url
```

## Monitoring

With the API running, open:

```text
http://localhost:8000/metrics
```
## commands

python -m venv mlops 
.\mlops\Scripts\Activate.ps1 
python -m pip install --upgrade pip setuptools wheel 
pip install --prefer-binary -r requirements.txt 

Main pipeline :
cd C:\Praveen\AI\Bits\sem3\api\MLOPS\heart-disease-mlops-assignment 
.\mlops\Scripts\Activate.ps1 
python src\data_acquisition.py 
python src\eda.py 
python src\train.py 
pytest -q 

MLflow :
mlflow ui --backend-store-uri sqlite:///mlflow.db 

FastAPI :
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 

Prediction test :
curl.exe -X POST http://localhost:8000/predict ` -H "Content-Type: application/json" ` -d "@sample_input.json" 

Docker and Kubernetes :
docker build -t heart-disease-api . 
docker run -p 8000:8000 heart-disease-api 

Using minikube
minikube start 
kubectl apply -f k8s/deployment.yaml 
kubectl apply -f k8s/service.yaml 
kubectl get pods 
kubectl get svc

Using docker kubernetes
Verify Kubernetes: kubectl get nodes
Build Docker Image: docker build -t heart-disease-api:latest .
Verify image:docker images
Apply deployment: kubectl apply -f k8s\deployment.yaml
Apply service: kubectl apply -f k8s\service.yaml
Check pods: kubectl get pods
Check services: kubectl get svc
Access API: kubectl port-forward service/heart-disease-service 8000:80
Open: http://localhost:8000/docs


prometheus: docker run -d --name prometheus -p 9090:9090 prom/prometheus
http://localhost:9090


mONITORING
http://localhost:8000/metrics
Create Prometheus container: docker run -d   --name prometheus   -p 9090:9090   prom/prometheus
docker run -d --name prometheus -p 9090:9090 prom/prometheus
http://localhost:9090
Configure Prometheus: monitoring/prometheus.yml
Verify Metrics: http://localhost:9090/targets

Graphana:
Run Grafana:docker run -d   --name grafana   -p 3000:3000   grafana/grafana
http://localhost:3000
http://host.docker.internal:9090
monitoring/grafana/dashboard.json
Dashboards → Import

