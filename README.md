# Iris Classifier API

A machine learning model that predicts the species of an iris flower from four measurements, served as a web API with FastAPI, packaged with Docker, and deployed on Render.

**Live API:** https://iris-ml-deployment-jav0.onrender.com
**Try it in the browser:** https://iris-ml-deployment-jav0.onrender.com/docs

> The app runs on a free plan, so it sleeps after about 15 minutes of inactivity. The first request after a quiet period can take up to a minute.

## How to use the API

Send a `POST` request to `/predict` with four measurements (in cm):

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Response:

```json
{
  "species": "setosa",
  "confidence": 1.0
}
```

### Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/` | Welcome message |
| GET | `/health` | Health check (is the app alive?) |
| POST | `/predict` | Predict the species from four measurements |
| GET | `/docs` | Interactive documentation page |

## Run it locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python train.py
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/docs

## Run the tests

```powershell
pytest
```

## Run with Docker

```powershell
docker build -t iris-api .
docker run -p 8000:8000 iris-api
```

## Project structure

```
app/            The API code (FastAPI)
model/          The saved model (iris_model.pkl)
tests/          Automated tests
train.py        Trains the model and saves it
Dockerfile      Recipe for building the container
.github/        GitHub Actions workflow (runs tests on every push)
```

## How it is deployed

Push to GitHub, GitHub Actions runs the tests, and if they pass Render rebuilds the Docker image and deploys it automatically.

## Tech stack

Python, scikit-learn, FastAPI, Uvicorn, Docker, GitHub Actions, Render