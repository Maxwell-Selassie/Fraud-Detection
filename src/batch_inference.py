import pandas as pd
import numpy as np
import joblib
import logging
import os
from datetime import datetime
from feature_engineering import feature_engineer, feature_encoding
from train_anomaly import transform_with_preprocessor

os.makedirs('data',exist_ok=True)
os.makedirs('log',exist_ok=True)



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
preprocessor = 'artifacts/preprocessor.joblib'

# load iso_forest model
def load_iso_forest(filename: str = iso_forest_path):
    try:
        iso_forest_model = joblib.load(filename)
        log.info(f'Isolation Forest Model successfully loaded!')
        return iso_forest_model
    except FileNotFoundError:
        log.exception(f'Model not found! Check file path and try again!')
        raise

# load rf_model
def load_rf_model(filename: str = rf_model_path):
    try:
        rf_model = joblib.load(filename)
        log.info(f'rfoost Model successfully loaded!')
        return rf_model
    except FileNotFoundError:
        log.exception(f'Model not found! Check file path and try again!')
        raise

# load fitted preprocessor
def load_preprocessor(filename : str = preprocessor):
    try:
        preprocessor = joblib.load(filename)
        log.info(f'Preprocessor successfully loaded!')
        return preprocessor
    except FileNotFoundError:
        log.exception(f'Model not found! Check file path and try again!')
        raise

# load batch file
def load_new_batch(filename : str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filename)
        log.info(f'New batch loaded with shape: {df.shape}')
        return df
    except Exception as e:
        log.error(f'Failed to load batch file : {e}')
        raise e
    
# batch inference
def run_batch_inference(filepath: str):
    log.info(f'Starting batch inference process: ')

    # load dataset
    df = load_new_batch(filepath)

    # feature engineering and encoding
    df = feature_engineer(df)
    log.info(f'Feature Engineering completed')
    preprocessor = load_preprocessor()

    x = preprocessor.transform(df)
    log.info(f'Feature Encoding completed')

    # unsupervised predictions (isolation forest)
    iso_forest_model = load_iso_forest()
    df['AnomalyScore'] = iso_forest_model.decision_function(x)
    df['is_anomaly'] = iso_forest_model.predict(x)
    df['is_anomaly'] = df['is_anomaly'].apply(lambda x: 1 if x == -1 else 0)
    anomaly_sum = df['is_anomaly'].sum()
    log.info(f'IsolationForest Anomalies detected: {anomaly_sum} out of {len(df)}')

    # semi supervised predictions(rf_model)
    hash_encode = transform_with_preprocessor(df)
    df_hashed = hash_encode.fit_transform(df)
    df = preprocessor.transform(df_hashed)

    rf_model = load_rf_model()
    df['fraud_probability'] = rf_model.predict_proba(df)[:,1]
    threshold = 0.2
    df['fraud_prediction'] = (df['fraud_probability'] >= threshold).astype(int)

    # combine predictions
    df['final_predictions'] = np.where((df['is_anomaly'] == 1) & df(['fraud_prediction'] == 1), 1, 0)

    anomaly_count = df['final_predictions'].sum()
    log.info(f'Final fraud detected : {anomaly_count} out of {len(df)}')


    os.makedirs('data',exist_ok=True)
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = (f'data/predictions_{date}.csv')
    df.to_csv(output_file,index=False)
    log.info(f'Predictions saved to {output_file}')

    return df

# run script
if __name__ == '__main__':
    new_batch_path = (f'data/bank_transactions_data_2.csv')
    predictions = run_batch_inference(new_batch_path)