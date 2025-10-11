import pandas as pd
import numpy as np
# import warnings
# warnings.filterwarnings('ignore')
import logging
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import CountEncoder, HashingEncoder

log = logging.getLogger('Exploratory_Data_Analysis')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s : %(message)s', datefmt='%H:%M:%S')

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Starting feature engineering...")

    # Ratios & Aggregates
    df['amount_to_balance_ratio'] = np.where(df['AccountBalance'] > 0,
                    df['TransactionAmount'] / df['AccountBalance'], 0)
    df['avg_txn_amount_account'] = df.groupby('AccountID')['TransactionAmount'].transform('mean')
    df['txn_amount_account'] = df.groupby('AccountID')['TransactionAmount'].transform('count')

    # Z-score
    user_stats = df.groupby('AccountID')['TransactionAmount'].agg(['mean','std']).reset_index()
    df = df.merge(user_stats, on='AccountID', how='left', suffixes=('','_user'))
    df['amount_zscore'] = (df['TransactionAmount'] - df['mean']) / df['std']
    df['amount_zscore'] = df['amount_zscore'].fillna(0)

    # Time-based
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'])
    df['diff_btwn_txn_times'] = abs(df['TransactionDate'] - df['PreviousTransactionDate']).dt.days
    df['avg_time_btwn_txns'] = df.groupby('AccountID')['diff_btwn_txn_times'].transform('mean')

    # Behavioral ratios
    df['amount_to_user_avg'] = df['TransactionAmount'] / df['avg_txn_amount_account']
    df['unique_merchant_user'] = df.groupby('AccountID')['MerchantID'].transform('nunique')

    # Temporal features
    df['TransactionHour'] = df['TransactionDate'].dt.hour
    df['TransactionDayOfTheWeek'] = df['TransactionDate'].dt.dayofweek
    df['is_night'] = df['TransactionHour'].apply(lambda h: 1 if h < 6 or h > 22 else 0)

    # Device/IP relationships
    df['unique_devices_per_account'] = df.groupby('AccountID')['DeviceID'].transform('nunique')
    df['unique_ips_per_account'] = df.groupby('AccountID')['IP Address'].transform('nunique')
    df['unique_accounts_per_devices'] = df.groupby('DeviceID')['AccountID'].transform('nunique')

    # Merchant behavior
    df['merchant_avg_amount'] = df.groupby('MerchantID')['TransactionAmount'].transform('mean')
    df['merchant_amount_deviation'] = df['TransactionAmount'] - df['merchant_avg_amount']

    # Login attempts
    df['avg_loginAttempts'] = df.groupby('AccountID')['LoginAttempts'].transform('mean')
    df['loginAttemps_excess'] = df['LoginAttempts'] - df['avg_loginAttempts']

    # Log transform
    df['TransactionAmount_log'] = np.log1p(df['TransactionAmount'])

    log.info(f"Feature engineering completed: {df.shape[1]} columns total.")
    df.to_csv('data/processed_bank_transaction_data.csv',index=True)
    return df

def feature_encoding(df: pd.DataFrame):
    # define numeric features
    log.info('Starting feature scaling and encoding...')
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    one_hot_features = ['Channel','CustomerOccupation','TransactionType']
    freq_features = ['Location','MerchantID']
    hash_features = ['AccountID','DeviceID','IP Address']

    df.drop(columns=['TransactionID','TransactionDate','PreviousTransactionDate'])
    # define transformers 
    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    robust_features = ['AccountBalance', 'TransactionAmount_log']
    robust_transformer = Pipeline(steps=[
        ('robust', RobustScaler())
    ])

    onehot_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    hash_transformer = Pipeline(steps=[
        ('encoder',HashingEncoder(n_components=16))
    ])

    freq_transformer = Pipeline(steps=[
        ('encoder',CountEncoder())
    ])

    # combine into a column transformer
    preprocessor = ColumnTransformer(transformers=[
        ('num',num_transformer, numeric_features),
        ('robust',robust_transformer,robust_features),
        ('onehot',onehot_transformer,one_hot_features),
        ('freq',freq_transformer, freq_features),
        ('hash',hash_transformer,hash_features)
    ],remainder='drop',verbose_feature_names_out=False)
    joblib.dump(preprocessor, 'artifacts/preprocessor.joblib')
    logging.info('Feature scaling and encoding completed')
    return preprocessor
