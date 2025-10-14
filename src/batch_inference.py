# scripts/batch_inference.py

import pandas as pd
import numpy as np
import joblib
import logging
import os
from datetime import datetime
from feature_engineering import feature_engineer
from train_anomaly import transform_with_preprocessor
from eda import run_eda

# Ensure paths exist
os.makedirs('data', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# setup logging
logging.basicConfig(
    filename='logs/inference.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger('Batch_Inference')

iso_forest_path = 'artifacts/isolation_forest_model.joblib'
rf_model_path = 'artifacts/RandomForest_fraud_detector.pkl'
preprocessor_path = 'artifacts/preprocessor.joblib'


def load_model(filename: str):
    """Generic loader with logging"""
    try:
        model = joblib.load(filename)
        log.info(f"Loaded model: {filename}")
        return model
    except FileNotFoundError:
        log.error(f"Model not found: {filename}")
        raise


def run_batch_inference(filepath: str):
    """Performs full batch inference and saves predictions."""
    log.info(f"Starting batch inference for file: {filepath}")

    # Load data
    df_orig = run_eda(filepath)
    log.info(f"Loaded batch data with shape: {df_orig.shape}")



    # Feature engineering
    df = feature_engineer(df_orig.copy())

    # Load preprocessor
    preprocessor = load_model(preprocessor_path)
    hash_encode = transform_with_preprocessor(df)

    x_hashed = hash_encode.transform(df)
    X = preprocessor.transform(df)
    log.info("Feature transformation complete")

    # Unsupervised (Isolation Forest)
    iso_model = load_model(iso_forest_path)
    df['AnomalyScore'] = iso_model.decision_function(X)
    df['is_anomaly'] = iso_model.predict(X)
    df['is_anomaly'] = df['is_anomaly'].apply(lambda x: 1 if x == -1 else 0)
    log.info(f"IsolationForest anomalies detected: {df['is_anomaly'].sum()}")

    # Semi-supervised (RandomForest)
    hashed_df = preprocessor.transform(x_hashed)
    rf_model = load_model(rf_model_path)
    df['fraud_probability'] = rf_model.predict_proba(hashed_df)[:, 1]
    df['fraud_prediction'] = (df['fraud_probability'] >= 0.2).astype(int)

    # Combine
    df['final_predictions'] = np.where(
        (df['is_anomaly'] == 1) & (df['fraud_prediction'] == 1), 1, 0
    )

    anomaly_count = df['final_predictions'].sum()
    log.info(f"Final fraud detected: {anomaly_count}/{len(df)}")

    # Save
    output_file = f"data/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_file, index=False)
    log.info(f"Predictions saved to {output_file}")

    return df[['AccountID', 'fraud_probability', 'final_predictions']]
