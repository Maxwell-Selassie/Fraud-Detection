import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from category_encoders import HashingEncoder
from xgboost import XGBClassifier
import joblib, logging, numpy as np, mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from pathlib import Path
import matplotlib.pyplot as plt

# -----------------------
# PATHS AND LOGGING SETUP
# -----------------------

base_dir = Path(__file__).resolve().parents[1]
logs_dir = base_dir / 'logs'
logs_dir.mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)
Path('artifacts').mkdir(exist_ok=True)
Path('plots').mkdir(exist_ok=True)
logs_path = logs_dir / 'train_anomaly.log'

# --------------------
# MLFLOW CONFIGURATION
# --------------------
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('FraudDetection_AnomalyModels')

log = logging.getLogger('ModelTraining')
logging.basicConfig(
    filename=logs_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s : %(message)s',
    datefmt='%H:%M:%S'
)

# -----------------
# UTILITY FUNCTIONS
# -----------------
def load_dataset(filename: str = 'data/processed_bank_transaction_data.csv'):
    try:
        df = pd.read_csv(filename)
        log.info(f'Dataset loaded from {filename}')
        return df
    except FileNotFoundError:
        log.exception('Dataset not found!')
        raise

def load_preprocessor(filename: str = 'artifacts/preprocessor.joblib'):
    try:
        preprocessor = joblib.load(filename)
        log.info(f'Preprocessor loaded from {filename}')
        return preprocessor
    except FileNotFoundError:
        log.exception('Preprocessor not found!')
        raise

def get_hash_encoder(df: pd.DataFrame):
    hash_features = ['AccountID','DeviceID','IP Address']
    return HashingEncoder(cols=hash_features, n_components=16)

# ----------------------------
# GENERIC MLFLOW HELPER LOGIC
# ----------------------------
def register_and_promote_model(run, model_uri, model_name):
    """
    Safely register and promote an MLflow model.
    - If model exists, register new version
    - Promote latest version to PRODUCTION
    """
    client = MlflowClient()

    # Try registering new version of model
    try:
        result = mlflow.register_model(model_uri=model_uri, name=model_name)
        latest_version = result.version
        print(f"Registered new version ({latest_version}) of {model_name}")
    except mlflow.exceptions.RestException as e:
        print(f"Model {model_name} may already exist. Checking registry...")
        registered_models = [m.name for m in client.search_registered_models()]
        if model_name not in registered_models:
            result = mlflow.register_model(model_uri=model_uri, name=model_name)
            latest_version = result.version
            print(f"Registered new model: {model_name} (version {latest_version})")
        else:
            # Get the last registered version
            versions = client.search_model_versions(f"name='{model_name}'")
            latest_version = max(int(v.version) for v in versions)
            print(f"Model {model_name} already registered (latest version: {latest_version})")

    # Promote to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage='Production',
        archive_existing_versions=True
    )
    print(f"✅ Model {model_name} (v{latest_version}) promoted to PRODUCTION!")
    log.info(f"{model_name} v{latest_version} promoted to Production.")

# ==================================================
# UNSUPERVISED ANOMALY DETECTION (ISOLATION FOREST)
# ==================================================
def isolation_forest(df: pd.DataFrame, preprocessor, hash_encode):
    with mlflow.start_run(run_name='IsolationForest_Baseline') as run:
        iso_forest = IsolationForest(
            n_estimators=200,
            contamination=0.03,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        x_hashed = hash_encode.fit_transform(df)
        x = preprocessor.fit_transform(x_hashed)
        iso_forest.fit(x)

        preds = iso_forest.predict(x)
        scores = iso_forest.decision_function(x)
        df['AnomalyFlag'] = np.where(preds == -1, 1, 0)
        df['AnomalyScore'] = -scores

        anomaly_count = df['AnomalyFlag'].sum()
        mlflow.log_metric('Anomalies Detected', anomaly_count)
        mlflow.log_param('contamination', 0.03)

        model_name = 'IsolationForestModel'
        mlflow.sklearn.log_model(iso_forest, artifact_path=model_name)
        model_uri = f"runs:/{run.info.run_id}/{model_name}"

        register_and_promote_model(run, model_uri, model_name)

    return df

# ====================================
# SUPERVISED FRAUD DETECTION TRAINING
# ====================================
def train_test_split_(df: pd.DataFrame):
    y = df['AnomalyFlag']
    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2, stratify=y)
    x_train.to_csv('data/x_train.csv', index=False)
    x_test.to_csv('data/x_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    return x_train, x_test, y_train, y_test

def rf_xgb_training(df: pd.DataFrame, hash_encode, preprocessor):
    x_train, x_test, y_train, y_test = train_test_split_(df)
    x_hashed = hash_encode.fit_transform(x_train)
    preprocessor.fit(x_hashed)
    x_train = preprocessor.transform(x_hashed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=30)

    model_results = {}

    # ---------------- RANDOM FOREST ----------------
    with mlflow.start_run(run_name='RandomForest_Model') as run:
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
        }
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        rf_grid = GridSearchCV(rf, rf_params, cv=cv, n_jobs=-1, scoring='f1', verbose=1)
        log.info('Training RandomForest...')
        rf_grid.fit(x_train, y_train)

        rf_best_ = rf_grid.best_estimator_
        mlflow.log_params(rf_grid.best_params_)

        x_test_hashed = hash_encode.transform(x_test)
        x_test = preprocessor.transform(x_test_hashed)
        y_probs = rf_best_.predict_proba(x_test)[:, 1]
        threshold = 0.2
        y_pred = (y_probs >= threshold).astype(int)

        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1_Score': f1_score(y_test, y_pred),
            'ROC_AUC': roc_auc_score(y_test, y_probs)
        }
        mlflow.log_metrics(metrics)

        model_name = 'RandomForestModel'
        mlflow.sklearn.log_model(rf_best_, artifact_path=model_name)
        model_uri = f"runs:/{run.info.run_id}/{model_name}"
        register_and_promote_model(run, model_uri, model_name)

        model_results[model_name] = metrics['F1_Score']
        log.info(f"RandomForest metrics: {metrics}")

    # ---------------- XGBOOST ----------------
    with mlflow.start_run(run_name='XGBoost_Model') as run:
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        xgb = XGBClassifier(objective='binary:logistic', random_state=30, eval_metric='logloss')
        xgb_grid = GridSearchCV(xgb, xgb_params, scoring='f1', cv=cv, verbose=1, n_jobs=-1)
        log.info('Training XGBoost...')
        xgb_grid.fit(x_train, y_train)

        xgb_best_ = xgb_grid.best_estimator_
        mlflow.log_params(xgb_grid.best_params_)

        y_probs = xgb_best_.predict_proba(x_test)[:, 1]
        threshold = 0.2
        y_pred = (y_probs >= threshold).astype(int)

        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1_Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC_AUC': roc_auc_score(y_test, y_probs)
        }
        mlflow.log_metrics(metrics)

        model_name = 'XGBoostModel'
        mlflow.xgboost.log_model(xgb_best_, artifact_path=model_name)
        model_uri = f"runs:/{run.info.run_id}/{model_name}"
        register_and_promote_model(run, model_uri, model_name)

        model_results[model_name] = metrics['F1_Score']
        log.info(f"XGBoost metrics: {metrics}")

    # Compare and print the best model
    best_model_name = max(model_results, key=model_results.get)
    print(f"✅ Best Model: {best_model_name} (F1: {model_results[best_model_name]:.4f}) promoted to PRODUCTION!")

# -------------------------
# MAIN ENTRY POINT
# -------------------------
if __name__ == "__main__":
    df = load_dataset()
    preprocessor = load_preprocessor()
    hash_encode = get_hash_encoder(df)

    df = isolation_forest(df, preprocessor, hash_encode)
    rf_xgb_training(df, hash_encode, preprocessor)
