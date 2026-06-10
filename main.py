import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data.loader import load_data
from src.features.engineering import add_features, drop_useless
from src.models.train import (
    cross_validate_model,
    evaluate_model,
    train_model,
    tune_xgboost,
)
from src.preprocessing.pipeline import build_pipeline, build_preprocessor

DATA_PATH = 'data/attrition_dataset.pkl'
EXPERIMENT_NAME = 'attrition_prediction'
MODEL_NAME = 'attrition_model'
ARTIFACTS_DIR = Path(os.getenv('ARTIFACTS_DIR', 'artifacts'))
TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db')


def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 1. Carga y preparación del dataset
    df = load_data(DATA_PATH)
    df = drop_useless(df)

    income_by_level = df.groupby('JobLevel')['MonthlyIncome'].mean().to_dict()
    with open(ARTIFACTS_DIR / 'income_by_level.json', 'w') as f:
        json.dump(income_by_level, f)

    df = add_features(df, income_by_level=income_by_level)

    X = df.drop(columns='Attrition')
    y = (df['Attrition'] == 'Yes').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_features = X.select_dtypes(include='number').columns.tolist()

    # 2. Tuning de XGBoost con Optuna
    print("Tuning XGBoost con Optuna (100 trials)...")
    best_params = tune_xgboost(X_train, y_train, num_features=num_features, n_trials=50)

    # 3. Definición de modelos a entrenar
    models = {
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'xgb_base':            XGBClassifier(random_state=42, eval_metric='logloss'),
        'xgb_tuned':           XGBClassifier(**best_params, random_state=42, eval_metric='logloss'),
    }

    best_run_id = None
    best_auc = 0.0
    best_model = None

    # 4. Loop genérico: entrena, evalúa y loguea cada modelo en MLflow
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Entrenando: {name}")

        with mlflow.start_run(run_name=name) as run:
            preprocessor = build_preprocessor(num_features)
            pipeline = build_pipeline(model, preprocessor)
            pipeline = train_model(pipeline, X_train, y_train)

            metrics = evaluate_model(pipeline, X_test, y_test, X_train, y_train)
            cv_metrics = cross_validate_model(pipeline, X, y)

            mlflow.log_params({k: str(v) for k, v in model.get_params().items()})
            loggable = {k: v for k, v in {**metrics, **cv_metrics}.items() if k != 'report'}
            mlflow.log_metrics(loggable)
            mlflow.sklearn.log_model(pipeline, 'model', registered_model_name=MODEL_NAME)

            print(f"  Accuracy:  {metrics['accuracy']} | Train Accuracy: {metrics.get('train_accuracy', 'N/A')}")
            print(f"  AUC-ROC:   {metrics['auc_roc']}")
            print(f"  CV AUC:    {cv_metrics['cv_auc_mean']} ± {cv_metrics['cv_auc_std']}")
            print(f"\n{metrics['report']}")

            if metrics['auc_roc'] > best_auc:
                best_auc = metrics['auc_roc']
                best_run_id = run.info.run_id
                best_model = model

    # 5. Reentrenar el mejor modelo con todos los datos (X completo)
    print(f"\n{'='*50}")
    print("Reentrenando modelo final con todos los datos...")
    with mlflow.start_run(run_name='final_model_full_data') as run:
        preprocessor_final = build_preprocessor(num_features)
        pipeline_final = build_pipeline(best_model, preprocessor_final)
        pipeline_final = train_model(pipeline_final, X, y)
        mlflow.log_params({k: str(v) for k, v in best_model.get_params().items()})
        mlflow.log_param('trained_on', 'full_dataset')
        mlflow.sklearn.log_model(pipeline_final, 'model', registered_model_name=MODEL_NAME)
        best_run_id = run.info.run_id

    # 6. Promover el modelo final como "champion" en el registry
    client = MlflowClient(tracking_uri=TRACKING_URI)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    best_version = next(v for v in versions if v.run_id == best_run_id)

    client.set_registered_model_alias(MODEL_NAME, "champion", best_version.version)

    print(f"\n{'='*50}")
    print(f"Mejor modelo -> run_id: {best_run_id}  (AUC-ROC: {best_auc})")
    print(f"Registrado en MLflow como '{MODEL_NAME}' version {best_version.version} @champion")
    return best_run_id


if __name__ == '__main__':
    main()
