import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

MODEL_NAME = "iris-classifier"

# 1. Find the best finished run
runs = mlflow.search_runs(
    experiment_names=["iris-classifier-models"],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.accuracy DESC", "metrics.f1_macro DESC"],
    max_results=1,
)
best = runs.iloc[0]
run_id = best["run_id"]
print("Best run:", best["tags.mlflow.runName"], "| accuracy:", best["metrics.accuracy"])

# 2. Add that run's model to the registry under a name
result = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
print("Registered as version", result.version)

# 3. Point the 'production' alias at this version
client.set_registered_model_alias(MODEL_NAME, "production", result.version)
print(f"Alias 'production' now points to version {result.version}")