import pickle
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. Load the data
iris = load_iris()
X = iris.data      # the 4 measurements for each flower
y = iris.target    # the species (0, 1 or 2)

# 2. Split: 80% to train on, 20% kept aside to test on
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Check how good it is on flowers it has never seen
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Test accuracy: {accuracy:.2f}")

# 5. Save the model (and the species names) to a pickle file
Path("model").mkdir(exist_ok=True)
with open("model/iris_model.pkl", "wb") as f:
    pickle.dump({"model": model, "target_names": iris.target_names.tolist()}, f)

print("Saved model to model/iris_model.pkl")