from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_setosa():
    response = client.post("/predict", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    })
    assert response.status_code == 200
    assert response.json()["species"] == "setosa"


def test_bad_input_is_rejected():
    response = client.post("/predict", json={"sepal_length": -1})
    assert response.status_code == 422

def test_root():
    response = client.get("/")
    assert response.status_code == 200