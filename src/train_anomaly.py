import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import logging
import numpy as np
from category_encoders import HashingEncoder
import matplotlib.pyplot as plt
import seaborn as sns

log = logging.getLogger('ModelTraining')
logging.basicConfig(level=logging.INFO,  format='%(asctime)s - %(levelname)s : %(message)s', datefmt='%H:%M:%S')

# load dataset
def load_dataset(filename: str = 'data/processed_bank_transaction_data.csv'):
    '''Loads the already processed dataset
    Input : filename
    Ouput : A dataframe'''
    try:
        df = pd.read_csv(filename)
        log.info(f'Dataset successfully loaded from {filename}')
        return df
    except FileNotFoundError:
        log.exception('Dataset not found! Check file path and try again')
        raise

# load preprocessor
def load_preprocessor(filename: str = 'artifacts/preprocessor.joblib'):
    '''Loads the preprocessor'''
    try:
        preprocessor = joblib.load(filename)
        log.info(f'Preprocessor successfully loaded from {filename}')
        return preprocessor
    except FileNotFoundError:
        log.exception('Preprocessor not found! Check file path and try again')
        raise
# ============================================
# UNSUPERVISED ANOMALY DETECTION (ISOLATION FOREST)
# ============================================

def isolation_forest(df: pd.DataFrame, preprocessor):
    '''Train an isolation forest algorithm as basline'''
    X = df.copy()

    # Define features
    hash_features = ['AccountID','DeviceID','IP Address']
    hash_encode = HashingEncoder(cols=hash_features, n_components=16)
    x_hashed = hash_encode.fit_transform(X)

    x = preprocessor.fit_transform(x_hashed)
    x

    # -------------------------
    # 2. Initialize Isolation Forest
    # -------------------------
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.03,        # expected fraction of anomalies (tune this)
        max_samples='auto',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # -------------------------
    # 3. Train the model
    # -------------------------
    iso_forest.fit(x)

    # -------------------------
    # 4. Get predictions and anomaly scores
    # -------------------------
    # predictions: -1 = anomaly, 1 = normal
    preds = iso_forest.predict(x)
    scores = iso_forest.decision_function(x)  # higher score = more normal

    # Convert to readable form
    df['AnomalyFlag'] = np.where(preds == -1, 1, 0)  # 1 = anomaly/fraud
    df['AnomalyScore'] = -scores  # invert so higher means more anomalous

    # -------------------------
    # 5. Analyze results
    # -------------------------
    print("Anomalies detected:", df['AnomalyFlag'].sum(), "out of", len(df))

    # Distribution of scores
    plt.figure(figsize=(8,5))
    sns.histplot(df['AnomalyScore'], bins=50, kde=True, color='orange')
    plt.title("Distribution of Anomaly Scores")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.show()

    # Quick overview of anomaly transactions
    log.info(df[df['AnomalyFlag'] == 1].head(10))

    # -------------------------
    # 6. Save model artifact
    # -------------------------
    joblib.dump(iso_forest, "artifacts/isolation_forest_model.joblib")
    print("Isolation Forest model saved successfully.")


