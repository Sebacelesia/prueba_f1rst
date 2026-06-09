import json
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import EmployeeInput, PredictionOutput
from src.features.engineering import add_features

MODEL_URI = 'models:/attrition_model@champion'
INCOME_PATH = Path('artifacts/income_by_level.json')
TRACKING_URI = 'sqlite:///mlflow.db'

model = None
income_by_level = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, income_by_level

    if not INCOME_PATH.exists():
        raise RuntimeError("Artifacts no encontrados. Correr primero: python main.py")

    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.sklearn.load_model(MODEL_URI)

    with open(INCOME_PATH) as f:
        raw = json.load(f)
    income_by_level = {int(k): v for k, v in raw.items()}

    print(f"Modelo cargado desde registry: {MODEL_URI}")
    yield


app = FastAPI(title="Attrition Prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(employee: EmployeeInput):
    try:
        df = pd.DataFrame([employee.model_dump()])
        df = add_features(df, income_by_level=income_by_level)

        proba = float(model.predict_proba(df)[:, 1][0])
        pred = int(model.predict(df)[0])

        return PredictionOutput(
            prediction=pred,
            prediction_label="Yes" if pred == 1 else "No",
            probability=round(proba, 4),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
