import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from evidently import Report
from evidently.presets import DataDriftPreset

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def load_data():
    """Return (reference, current): the training data and the held-out data."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=FEATURES)
    X_train, X_test = train_test_split(
        X, test_size=0.2, random_state=42, stratify=iris.target
    )
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True)


def shift_petals(df, factor=1.5):
    """Simulate a changed world: petals become 50% bigger."""
    shifted = df.copy()
    shifted["petal_length"] = shifted["petal_length"] * factor
    shifted["petal_width"] = shifted["petal_width"] * factor
    return shifted


def check_drift(current, reference):
    """Run Evidently and return {column: {value, threshold, method, drifted}}."""
    report = Report([DataDriftPreset()])
    result = report.run(current, reference)
    raw = result.dict()

    columns = {}
    for metric in raw["metrics"]:
        config = metric["config"]
        if not config["type"].endswith("ValueDrift"):
            continue  # skip summary metrics, keep per-column results

        value = float(metric["value"])
        threshold = float(config["threshold"])
        method = config["method"]

        # p-value tests flag drift when the value is BELOW the threshold;
        # distance tests flag drift when it is AT or ABOVE it
        drifted = value < threshold if "p_value" in method else value >= threshold

        columns[config["column"]] = {
            "value": value,
            "threshold": threshold,
            "method": method,
            "drifted": drifted,
        }
    return columns


def dataset_drifted(columns, share_threshold=0.5):
    """Our own rule: the dataset has drifted if at least half the columns have."""
    if not columns:
        raise ValueError("No per-column drift results were found")
    drifted = sum(1 for info in columns.values() if info["drifted"])
    return drifted / len(columns) >= share_threshold