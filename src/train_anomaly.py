import pandas as pd
import os
from sklearn.ensemble import IsolationForest
import joblib
import logging
import numpy as np
from category_encoders import HashingEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report
)

os.makedirs('artifacts',exist_ok=True)
os.makedirs('models',exist_ok=True)

log = logging.getLogger('ModelTraining')
logging.basicConfig(filename='logs/inference.log',
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
def transform_with_preprocessor(df: pd.DataFrame,preprocessor):
    df = df.copy()

    # Define features
    hash_features = ['AccountID','DeviceID','IP Address']
    hash_encode = HashingEncoder(cols=hash_features, n_components=16)
    x_hashed = hash_encode.fit_transform(df)

    x = preprocessor.fit_transform(x_hashed)
    return x

# target output - y
def target_feature(df):
    y = df['AnomalyFlag']
    return y

# ============================================
# UNSUPERVISED ANOMALY DETECTION (ISOLATION FOREST)
# ============================================

def isolation_forest(df: pd.DataFrame, preprocessor,x: np.array):
    '''Train an isolation forest algorithm as basline'''


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
    anomaly_flag = df['AnomalyFlag'].sum()
    log.info(f"Anomalies detected: {anomaly_flag}, out of {len(df)}")

    # Quick overview of anomaly transactions
    top_10 = df[df['AnomalyFlag'] == 1].head(10)
    log.info(f'\n{top_10}')

    # -------------------------
    # 6. Save model artifact
    # -------------------------
    joblib.dump(iso_forest, "artifacts/isolation_forest_model.joblib")
    log.info(f"Isolation Forest model saved successfully.")
    return df


def rf_xgb_training(df: pd.DataFrame, x: np.array, y: pd.Series):

    y = df['AnomalyFlag']

    x_train,x_test,y_train,y_test = train_test_split(
        x,y, test_size=0.2, stratify=y
    )
    # cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=30)

    # train random forest model
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
    print('-' * 70)
    # train xgboost model
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

    # evaluate both model and pick the best one
    models = {'RandomForest' : rf_best_, 'Xgboost' : xgb_best_}
    results = []

    for name, model in models.items():
        y_probs = model.predict_proba(x_test)[:,1]
        threshold = 0.2
        y_pred = (y_probs >= threshold).astype(int)
        metrics = {
            'Model' : name,
            'Accuracy' : accuracy_score(y_test,y_pred),
            'Precision' : precision_score(y_test,y_pred),
            'Recall' : recall_score(y_test,y_pred),
            'f1_score' : f1_score(y_test,y_pred),
            'roc_auc_score' : roc_auc_score(y_test, y_pred)
        }
        results.append(metrics)
        print('-'*70)
        print(f'{name} Classification Report \n',classification_report(y_test,y_pred))
    print('-'*70)
    results_df = pd.DataFrame(results)
    print('Model Comparison\n',results_df)

    # save the best model
    best_model_name = results_df.sort_values(by='f1_score',ascending=False).iloc[0]['Model']
    best_model = models[best_model_name]
    joblib.dump(best_model,f'models/{best_model_name}_fraud_detector.pkl')
    print('-'*70)
    log.info(f"Best model '{best_model_name}' saved as '{best_model_name}_fraud_detector.pkl")

def model_training():
    dataframe = load_dataset()
    preprocessor = load_preprocessor()
    x = transform_with_preprocessor(dataframe,preprocessor)
    df = isolation_forest(dataframe, preprocessor,x)
    y = target_feature(df)
    rf_xgb = rf_xgb_training(df, x, y)
if __name__ == '__main__':
    model_training()