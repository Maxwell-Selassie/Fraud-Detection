import pandas as pd
import os
from sklearn.ensemble import IsolationForest
import joblib
import logging
import numpy as np
from category_encoders import HashingEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
)
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parents[1]
logs_dir = base_dir / 'logs'
logs_dir.mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)
Path('artifacts').mkdir(exist_ok=True)
Path('plots').mkdir(exist_ok=True)

logs_path = logs_dir / 'train_anomaly.log'

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('FraudDetection_AnomalyModels')

log = logging.getLogger('ModelTraining')
logging.basicConfig(filename=logs_path,
                    level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s : %(message)s', 
                    datefmt='%H:%M:%S')

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

# transform df into a numpy array by fitting the preprocessor pipeline
def get_hash_encoder(df: pd.DataFrame):
    df = df.copy()

    # Define features
    hash_features = ['AccountID','DeviceID','IP Address']
    hash_encode = HashingEncoder(cols=hash_features, n_components=16)

    return  hash_encode


# ============================================
# UNSUPERVISED ANOMALY DETECTION (ISOLATION FOREST)
# ============================================

def isolation_forest(df: pd.DataFrame, preprocessor):
    '''Train an isolation forest algorithm as basline'''

    # -------------------------
    # 2. Initialize Isolation Forest
    # -------------------------
    with mlflow.start_run(run_name='IsolationForest_Baseline'):
        iso_forest = IsolationForest(
            n_estimators=200,
            contamination=0.03,        # expected fraction of anomalies (tune this)
            max_samples='auto',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        x = preprocessor.fit_transform(df)
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
        anomaly_count = df['AnomalyFlag'].sum()
        mlflow.log_metric('Anomalies Detected',anomaly_count)
        mlflow.log_param('contamination',0.03)
        mlflow.sklearn.log_model(iso_forest,'IsolationForestModel')
        log.info(f"Anomalies detected: {anomaly_count}, out of {len(df)}")

        # Quick overview of anomaly transactions
        top_10 = df[df['AnomalyFlag'] == 1].head(10)
        log.info(f'\n{top_10}')

        # -------------------------
        # 6. Save model artifact
        # -------------------------
        joblib.dump(iso_forest, "artifacts/isolation_forest_model.joblib")
        log.info(f"Isolation Forest model saved successfully.")
    return df

def train_test_split_(df: pd.DataFrame):
    y = df['AnomalyFlag']
    x_train,x_test,y_train,y_test = train_test_split(
        df, y, test_size= 0.2, stratify=y
    )
    x_train.to_csv('data/x_train.csv',index=False)
    x_test.to_csv('data/x_test.csv',index=False)
    y_train.to_csv('data/y_train.csv',index=False)
    y_test.to_csv('data/y_test.csv',index=False)
    return x_train,x_test,y_train,y_test

def rf_xgb_training(df: pd.DataFrame, hash_encode):

    # load the train_test splits
    x_train,x_test,y_train,y_test = train_test_split_(df)

    x_hashed = hash_encode.fit_transform(x_train)

    preprocessor = load_preprocessor()

    preprocessor.fit(x_hashed)
    x_train = preprocessor.transform(x_hashed)

    # cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=30)

    # train random forest model
    with mlflow.start_run(run_name='RandomForest_Model'):
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
        }
        rf = RandomForestClassifier(random_state=42,class_weight='balanced')
        rf_grid = GridSearchCV(estimator=rf, param_grid=rf_params, cv=cv, n_jobs=-1, scoring='f1', verbose=1)
        log.info('Training RandomForest...(This May Take A While)')
        rf_grid.fit(x_train,y_train)

        # best model for random forest
        rf_best_ = rf_grid.best_estimator_
        mlflow.log_params(rf_grid.best_params_)
        print('-' * 70)


        x_test_hashed = hash_encode.transform(x_test)
        x_test = preprocessor.transform(x_test_hashed)


        y_probs = rf_best_.predict_proba(x_test)[:,1]
        threshold = 0.2
        y_pred = (y_probs >= threshold).astype(int)

        metrics = {
            'Accuracy' : accuracy_score(y_test,y_pred),
            'Precision' : precision_score(y_test,y_pred),
            'Recall' : recall_score(y_test,y_pred),
            'f1_score' : f1_score(y_test,y_pred),
            'roc_auc_score' : roc_auc_score(y_test, y_probs)
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(rf_best_,'RandomForestModel')

        timestamp = datetime.now().strftime('%Y:%m:%d-%H:%M:%S')
        joblib.dump(rf_best_, f"artifacts/RandomForest_fraud_detector_{timestamp}.pkl")
        log.info("Random Forest model logged to MLflow and saved.")

        feat_importance = pd.Series(rf_best_.feature_importances_)
        feat_importance.nlargest(20).plot(kind='barh',title='Top Features')
        plt.tight_layout()
        plt.savefig('plots/rf_feature_importance.png')
    # train xgboost model

    with mlflow.start_run(run_name='XGBoost_Model'):
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        xgb = XGBClassifier(objective='binary:logistic',random_state=30,eval_metric='logloss')
        xgb_grid = GridSearchCV(estimator=xgb,param_grid=xgb_params, scoring='f1',verbose=1,n_jobs=-1)
        log.info("Training XGBoost...(This May Take A While)")
        xgb_grid.fit(x_train,y_train)

        xgb_best_ = xgb_grid.best_estimator_
        mlflow.log_params(xgb_grid.best_params_)


        y_probs = xgb_best_.predict_proba(x_test)[:,1]
        threshold = 0.2
        y_pred = (y_probs >= threshold).astype(int)
        metrics = {
            'Accuracy' : accuracy_score(y_test,y_pred),
            'Precision' : precision_score(y_test,y_pred,zero_division=0),
            'Recall' : recall_score(y_test,y_pred,zero_division=0),
            'f1_score' : f1_score(y_test,y_pred,zero_division=0),
            'roc_auc_score' : roc_auc_score(y_test, y_probs)
        }
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(xgb_best_,'XGBoostModel')
        print('-'*70)

        joblib.dump(xgb_best_, f"artifacts/XGBoost_fraud_detector_{timestamp}.pkl")
        log.info("XGBoost model logged to MLflow and saved.")

        feat_importance = pd.Series(rf_best_.feature_importances_)
        feat_importance.nlargest(20).plot(kind='barh',title='Top Features')
        plt.tight_layout()
        plt.savefig('plots/rf_feature_importance.png')

print('Both models tracked and stored in MLflow!')
