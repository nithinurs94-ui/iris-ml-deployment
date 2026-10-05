import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.datasets import load_iris

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Which version does 'production' point to right now?
mv = client.get_model_version_by_alias("iris-classifier", "production")
print("Production is version", mv.version, "| from run", mv.run_id)

# Load by name and alias (no run id needed)
model = mlflow.sklearn.load_model("models:/iris-classifier@production")
print("Model type:", type(model).__name__)

iris = load_iris()
prediction = model.predict([[6.3, 3.3, 6.0, 2.5]])
print("Prediction:", iris.target_names[prediction[0]])