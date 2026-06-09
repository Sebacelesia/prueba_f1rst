import pandas as pd
import mlflow.sklearn


def load_model(run_id: str):
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


def predict(model, data) -> dict:
    if isinstance(data, dict):
        data = pd.DataFrame([data])

    proba = model.predict_proba(data)[:, 1]
    pred = model.predict(data)

    return {
        'prediction': int(pred[0]),
        'probability': round(float(proba[0]), 4),
    }
