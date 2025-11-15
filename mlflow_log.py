import os
import json
import mlflow
import mlflow.sklearn
from google.cloud import storage
import joblib


with open("params.json", "r") as f:
    params = json.load(f)

with open("metrics.txt", "r") as f:
    metrics_content = f.read().strip()
accuracy = float(metrics_content.split(":")[-1])

model_path = "model/model.joblib"
model = joblib.load(model_path)

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris_logreg_experiment")

mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment(mlflow_experiment_name)

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, artifact_path="model")

print("Model, metrics, and params logged to MLflow successfully!")
