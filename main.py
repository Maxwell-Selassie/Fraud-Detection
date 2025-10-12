from src.eda import run_eda
from src.eda import advanced_visuals
from src.feature_engineering import feature_engineer
from src.feature_engineering import feature_encoding
import logging
import numpy as np
import pandas as pd

# setup logging
log = logging.getLogger('FraudDetection')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s : %(message)s',datefmt='%H:%M:%S')

def main():
    # load df from the eda.py file
    try:
        filename = 'data/bank_transactions_data_2.csv'
        raw_df = run_eda(filename)
        log.info(f'DataFrame has been loaded successfully!')
    except ValueError:
        log.exception('Error: Could not load dataframe')
        raise

    # visualization
    num_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = ['TransactionType','Channel','CustomerOccupation']
    visuals = advanced_visuals(raw_df, num_cols, cat_cols)

    # engineered features
    featured_df = feature_engineer(raw_df)

    # encoded features
    df = feature_encoding(featured_df)

if __name__ == '__main__':
    main()