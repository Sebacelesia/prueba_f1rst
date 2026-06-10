import optuna
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline, X_test, y_test, X_train=None, y_train=None) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'auc_roc':   round(roc_auc_score(y_test, y_prob), 4),
        'report':    classification_report(y_test, y_pred, target_names=['No', 'Yes']),
    }

    if X_train is not None and y_train is not None:
        metrics['train_accuracy'] = round(accuracy_score(y_train, pipeline.predict(X_train)), 4)

    return metrics


def cross_validate_model(pipeline, X, y, n_splits: int = 5) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = cross_validate(pipeline, X, y, cv=cv,
                             scoring=['accuracy', 'roc_auc'],
                             return_train_score=True)
    return {
        'cv_accuracy_mean':      round(results['test_accuracy'].mean(), 4),
        'cv_accuracy_std':       round(results['test_accuracy'].std(), 4),
        'cv_auc_mean':           round(results['test_roc_auc'].mean(), 4),
        'cv_auc_std':            round(results['test_roc_auc'].std(), 4),
        'cv_train_accuracy_mean': round(results['train_accuracy'].mean(), 4),
        'cv_train_auc_mean':     round(results['train_roc_auc'].mean(), 4),
    }


def tune_xgboost(X_train, y_train, num_features: list, n_trials: int = 100) -> dict:
    from src.preprocessing.pipeline import build_pipeline, build_preprocessor

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 600),
            'max_depth':        trial.suggest_int('max_depth', 2, 6),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 2.0),
            'reg_lambda':       trial.suggest_float('reg_lambda', 0.5, 3.0),
        }
        model = XGBClassifier(**params, random_state=42, eval_metric='logloss')
        pipeline = build_pipeline(model, build_preprocessor(num_features))
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=1)
        auc = scores.mean()
        print(f'  Trial {trial.number + 1}/{n_trials} - AUC: {auc:.4f}', flush=True)
        return auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f'Best AUC-ROC (CV): {study.best_value:.4f}')
    return study.best_params
