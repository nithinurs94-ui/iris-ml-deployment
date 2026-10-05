import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-classifier-models")

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)

n_estimators_options = [10, 50, 100]
max_depth_options = [1, 3, None]

for n_estimators in n_estimators_options:
    for max_depth in max_depth_options:
        run_name = f"rf-{n_estimators}-trees-depth-{max_depth}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average="macro")

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_macro", f1)

            # NEW: save the trained model as part of this run
            mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=X_train[:2],
                serialization_format="pickle",
            )

            print(f"{run_name}: accuracy={accuracy:.3f}, f1={f1:.3f}")