# src/batch_inference.py

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime
from feature_engineering import feature_engineer
from train_anomaly import get_hash_encoder
from eda import run_eda
import mlflow
from mlflow.tracking import MlflowClient

# -----------------------
# paths and logging setup
# -----------------------

base_dir = Path(__file__).resolve().parents[1]
logs_dir = base_dir / 'logs'

Path('logs').mkdir(exist_ok=True)
Path('data').mkdir(exist_ok=True)

log_path = logs_dir / 'batch_inference.log'

# setup logging
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger('Batch_Inference')

# --------------------
# MLflow configuration
# --------------------
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('FraudDetection_AnomalyModels')

# model regristry names 
iso_forest_name = 'IsolationForestModel'
rf_model_name = 'RandomForestModel'
xgb_model_name = 'XGBoostModel'


# ===============================
# load model from mlflow registry
# ===============================

def load_production_model(model_name: str):
    '''Load the Production version of a model registered in the mlflow model registry'''
    try:
        model_uri = f'models:/{model_name}/Production'
        log.info(f'Loading Production Model From Registry: {model_uri}')

        model = mlflow.sklearn.load_model(model_uri)
        log.info(f'Successfully loaded Model: {model_name}')
        return model
    except Exception as e:
        log.error(f'Failed to load model {model_name} : {e}')
        raise

# ========================
# batch inference function
# ========================

def run_batch_inference(filepath: str, threshold: float = 0.2):
    """Performs full batch inference using MLflow Model Registry.
    Combines unsupervised(isolationforest) and supervised learning"""
    log.info(f"Starting batch inference for file: {filepath}")

    # Load data 
    df_original = run_eda(filepath)
    df_orig = df_original['data']
    log.info(f"Loaded batch data with shape: {df_orig.shape}")



    # Feature engineering
    df = feature_engineer(df_orig.copy())

    # Load preprocessor
    preprocessor = joblib.load('artifacts/preprocessor.joblib')
    hash_encode = get_hash_encoder(df)

    x_hashed = hash_encode.fit_transform(df)
    X = preprocessor.fit_transform(x_hashed)
    log.info("Feature transformation complete")

    # ======================
    # Load production models
    #=======================

    iso_model = load_production_model(iso_forest_name)
    log.info('IsolationForest Model successfully loaded')

    try:
        rf_model = load_production_model(rf_model_name)
        model_type = 'RandomForest'
    except Exception:
        log.warning('Random Forest Model not found in production! Using XGBoost Model instead')
        rf_model = load_production_model(xgb_model_name)
        model_type = 'XGBoost'
    log.info(f'Using {model_type} model for fraud probability predictions')

    # =========================================
    # Unsupervised inference (Isolation Forest)
    # =========================================

    df['AnomalyScore'] = iso_model.decision_function(X)
    df['is_anomaly'] = iso_model.predict(X)
    df['is_anomaly'] = df['is_anomaly'].apply(lambda x: 1 if x == -1 else 0)
    log.info(f"IsolationForest anomalies detected: {df['is_anomaly'].sum()}")

    # =========================================
    # Semi-supervised (RandomForest or xgboost)
    # =========================================

    df['fraud_probability'] = rf_model.predict_proba(X)[:, 1]
    df['fraud_prediction'] = (df['fraud_probability'] >= 0.2).astype(int)

#     # Hybrid Decision Logic
    df['final_predictions'] = np.where(
        (df['is_anomaly'] == 1) & (df['fraud_prediction'] == 1), 1, 0
    )

    anomaly_count = df['final_predictions'].sum()
    log.info(f"Final fraud detected: {anomaly_count}/{len(df)}")

    # Save predictions
    output_file = f"data/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_file, index=False)
    log.info(f"Predictions saved to {output_file}")

    return df[['AccountID', 'fraud_probability', 'final_predictions']]

if __name__ == '__main__':
    sample_path = 'data/bank_transactions_data_2.csv'
    run_batch_inference(sample_path)
