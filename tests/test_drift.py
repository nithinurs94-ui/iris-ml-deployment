import pickle
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from monitoring import check_drift, dataset_drifted, load_data, shift_petals


def test_no_drift_on_normal_data():
    reference, current = load_data()
    columns = check_drift(current, reference)
    assert len(columns) == 4, "Expected a drift result for each of the 4 columns"
    assert not dataset_drifted(columns)


def test_drift_is_detected_on_shifted_data():
    reference, current = load_data()
    columns = check_drift(shift_petals(current), reference)
    assert columns["petal_length"]["drifted"]
    assert columns["petal_width"]["drifted"]
    assert not columns["sepal_length"]["drifted"]
    assert not columns["sepal_width"]["drifted"]


def test_production_model_accuracy_floor():
    model_path = Path(__file__).resolve().parent.parent / "model" / "iris_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)["model"]

    iris = load_iris()
    _, X_test, _, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )
    accuracy = accuracy_score(y_test, model.predict(X_test))
    assert accuracy >= 0.9, f"Model accuracy fell to {accuracy:.3f}"