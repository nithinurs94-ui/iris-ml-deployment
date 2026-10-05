import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 1. Ask MLflow for the run with the best accuracy (ties broken by f1)
runs = mlflow.search_runs(
    experiment_names=["iris-classifier-models"],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.accuracy DESC", "metrics.f1_macro DESC"],
    max_results=1,
)
best = runs.iloc[0]
run_id = best["run_id"]
print("Best run:", best["tags.mlflow.runName"])
print("Accuracy:", best["metrics.accuracy"])

# 2. Load the model that was saved inside that run
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# 3. Use it
iris = load_iris()
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)
print("Prediction:", iris.target_names[prediction[0]])