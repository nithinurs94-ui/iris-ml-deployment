import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from evidently import Report
from evidently.presets import DataDriftPreset

# 1. Build a table of the four measurements, with tidy column names
iris = load_iris()
columns = [c.replace(" (cm)", "").replace(" ", "_") for c in iris.feature_names]
X = pd.DataFrame(iris.data, columns=columns)

# 2. Same split as train.py: 80% was used for training, 20% held back
X_train, X_test = train_test_split(
    X, test_size=0.2, random_state=42, stratify=iris.target
)

# 3. The reference is what the model was trained on
reference = X_train.reset_index(drop=True)

# 4. Two "production" datasets: one normal, one with bigger petals
current_normal = X_test.reset_index(drop=True)

current_shifted = current_normal.copy()
current_shifted["petal_length"] = current_shifted["petal_length"] * 1.5
current_shifted["petal_width"] = current_shifted["petal_width"] * 1.5


def make_report(current, reference, filename):
    report = Report([DataDriftPreset()])
    result = report.run(current, reference)  # current first, reference second
    result.save_html(filename)
    print("Saved", filename)


make_report(current_normal, reference, "drift_report_normal.html")
make_report(current_shifted, reference, "drift_report_shifted.html")