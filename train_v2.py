import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-classifier-models")

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)

with mlflow.start_run(run_name="gradient-boosting-v2") as run:
    mlflow.log_param("model_type", "GradientBoosting")
    mlflow.log_param("n_estimators", 100)

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_macro", f1)

    mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=X_train[:2],
        serialization_format="pickle",
    )

    result = mlflow.register_model(f"runs:/{run.info.run_id}/model", "iris-classifier")
    print(f"accuracy={accuracy:.3f}, f1={f1:.3f}")
    print("Registered as version", result.version)