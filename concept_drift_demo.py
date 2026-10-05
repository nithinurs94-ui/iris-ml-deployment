import pickle

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from evidently import DataDefinition, Dataset, MulticlassClassification, Report
from evidently.presets import ClassificationPreset, DataDriftPreset

# 1. Load the model your API uses
with open("model/iris_model.pkl", "rb") as f:
    artifact = pickle.load(f)
model = artifact["model"]
names = artifact["target_names"]

# 2. Load the data, with species as text labels
iris = load_iris()
columns = [c.replace(" (cm)", "").replace(" ", "_") for c in iris.feature_names]
X = pd.DataFrame(iris.data, columns=columns)
y = pd.Series([names[i] for i in iris.target])

# Same split as train.py, so X_test is data the model never saw
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=iris.target
)


def make_frame(features, true_labels):
    """Features + the true answer + what the model predicts."""
    df = features.reset_index(drop=True).copy()
    df["target"] = list(true_labels)
    df["prediction"] = [names[p] for p in model.predict(features.values)]
    return df


# 3. OLD WORLD: the true labels follow the original definition
reference = make_frame(X_test, y_test)

# 4. NEW WORLD: same flowers, but versicolor with long petals is now "virginica"
new_labels = y_test.copy()
changed = (X_test["petal_length"] > 4.5) & (y_test == "versicolor")
new_labels[changed] = "virginica"
print(f"Flowers whose correct label changed: {int(changed.sum())} of {len(X_test)}")

current = make_frame(X_test, new_labels)

# 5. Plain accuracy, so the numbers are easy to read
print("Accuracy, old world:", round(accuracy_score(reference["target"], reference["prediction"]), 3))
print("Accuracy, new world:", round(accuracy_score(current["target"], current["prediction"]), 3))

# 6. Report 1: did the INPUTS drift? (features only, no labels)
inputs_report = Report([DataDriftPreset()])
inputs_result = inputs_report.run(current[columns], reference[columns])
inputs_result.save_html("drift_report_concept_inputs.html")

# 7. Report 2: how is the model PERFORMING? (needs the true labels)
definition = DataDefinition(
    numerical_columns=columns,
    categorical_columns=["target", "prediction"],
    classification=[
        MulticlassClassification(target="target", prediction_labels="prediction")
    ],
)
reference_ds = Dataset.from_pandas(reference, data_definition=definition)
current_ds = Dataset.from_pandas(current, data_definition=definition)

performance_report = Report([ClassificationPreset()])
performance_result = performance_report.run(current_ds, reference_ds)
performance_result.save_html("drift_report_concept_performance.html")

print("Saved both reports.")