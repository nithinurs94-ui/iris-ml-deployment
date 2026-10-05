import sys

import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

version = sys.argv[1]  # the version number you type after the script name
client.set_registered_model_alias("iris-classifier", "production", version)
print(f"'production' now points to version {version}")