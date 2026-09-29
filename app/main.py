import pickle
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---------- 1. Load the saved model ----------
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "iris_model.pkl"

with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
target_names = artifact["target_names"]

# ---------- 2. Create the app ----------
app = FastAPI(title="Iris Classifier API", version="1.0.0")


# ---------- 3. Describe what input we expect ----------
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=15, examples=[5.1])
    sepal_width: float = Field(..., gt=0, lt=15, examples=[3.5])
    petal_length: float = Field(..., gt=0, lt=15, examples=[1.4])
    petal_width: float = Field(..., gt=0, lt=15, examples=[0.2])


# ---------- 4. Health check endpoint ----------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- 5. Prediction endpoint ----------
@app.post("/predict")
def predict(features: IrisFeatures):
    row = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]]
    class_index = int(model.predict(row)[0])
    confidence = float(model.predict_proba(row)[0][class_index])
    return {"species": target_names[class_index], "confidence": round(confidence, 3)}