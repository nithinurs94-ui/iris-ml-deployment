import pickle
from collections import Counter

from sklearn.datasets import load_iris

# Load the model your API uses
with open("model/iris_model.pkl", "rb") as f:
    artifact = pickle.load(f)
model = artifact["model"]
names = artifact["target_names"]

iris = load_iris()

# Reference = the normal world. Current = a world where petals are 50% bigger.
reference = iris.data.copy()
current = iris.data.copy()
current[:, 2] *= 1.5  # petal length
current[:, 3] *= 1.5  # petal width

print("Average of each feature:")
for i, feature in enumerate(iris.feature_names):
    ref_avg = reference[:, i].mean()
    cur_avg = current[:, i].mean()
    print(f"  {feature:20s} reference={ref_avg:.2f}   current={cur_avg:.2f}")


def prediction_counts(X):
    predictions = model.predict(X)
    counts = Counter(names[p] for p in predictions)
    return {name: counts.get(name, 0) for name in names}


print("\nModel's answers on reference data:", prediction_counts(reference))
print("Model's answers on current data:  ", prediction_counts(current))