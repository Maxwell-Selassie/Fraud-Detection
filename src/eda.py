# Bank Transactions 
# This dataset simulates transactional activities from a digital banking platform. It includes detailed information for 
# each transaction such as amount, location, customerAge, Login Attempts, etc.

# File : bank_transactions_data_2.csv

# Goal of this project is to build a machine learning model to detect fraudulent transactions by analyzing several 
# underlying factors like transactions, demographics and user-behaviour.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import logging
import os

os.makedirs('logs',exist_ok=True)

log = logging.getLogger('Exploratory_Data_Analysis')
logging.basicConfig(filename='logs/inference.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s : %(message)s', 
                    datefmt='%H:%M:%S')


def load_data(filename: str = 'data/bank_transactions_data_2.csv'):
    try:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            log.info('Data has successfully been loaded')
        else:
            log.error('File Not Found')
        return df
    except FileNotFoundError as e:
        log.exception('File Not Found: ',e)
        return None
    

# ---Decriptive Summary of the dataset----
def descriptive_overview(df: pd.DataFrame):
    if df is not None:
        log.info(f'Number of observations {df.shape[0]}')
        log.info(f'Number of features : {df.shape[1]}\n')
        describe = df.describe(include='all').T
        log.info(f'\n{describe}')
    else:
        log.warning('DataFrame is empty!')


# ---Analysis of the numerical columns---
def numeric_cols_summary(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    log.info(f'| Number of numeric columns : {len(numeric_cols)} | Examples : {numeric_cols[:3]}\n')
    for i,col in enumerate(numeric_cols,1):
        log.info(f'{i} {col:<24} | Min : {df[col].min():<15} | Max : {df[col].max():<10}\n')
    return numeric_cols


# ----Analysis of the categorical columns-----
def category_cols_summary(df: pd.DataFrame):
    category_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    log.info(f'| Number of categorical columns : {len(category_cols)} | Examples : {category_cols[:3]}\n')
    for i,col in enumerate(category_cols,1):
        uniques = df[col].unique()
        log.info(f'{i:<2}. {col:<25} |Unique : {df[col].nunique():<7} | Examples : {uniques[:3]}\n')



# --------check for duplicates------
def duplicate_data(df: pd.DataFrame):
    duplicates = df[df.duplicated()]
    log.info(f'Number of duplicates : {len(duplicates)}\n')
    if len(duplicates) == 0:
        log.info(f'No duplicates found in the data\n')
    else:
        log.info(duplicates)


# -----check for missing values-----
def missing_data(df: pd.DataFrame):
    missing_d = df.isnull().sum()
    missing_d = missing_d[missing_d > 0].sort_values(ascending=False)
    missing_pct = (missing_d / len(df)) * 100
    missing_data_df = pd.DataFrame({
        'missing_data' : missing_d,
        'missing_pct' : missing_pct.round(2)
    })
    log.info(missing_data_df)


# ---check for outliers----
def check_outliers(df: pd.DataFrame, col: str):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return outliers, lower_bound, upper_bound
def outlier_summary(df: pd.DataFrame, numeric_col: list[str]):
    for i, col in enumerate(numeric_col):
        outlier, lower, upper = check_outliers(df, col)
        log.info(f'{i}. {col:<20} | Number of outliers : {len(outlier):<4} | Range : ({lower} - {upper})')

def run_eda(filename: str = '../data/bank_transactions_data_2.csv'):
    df = load_data()
    descriptive_overview(df)
    num_cols = numeric_cols_summary(df)
    category_cols_summary(df)
    duplicate_data(df)
    missing_data(df)
    outlier_summary(df, num_cols)
    return df


# ---univariate analysis------
def advanced_visuals(df: pd.DataFrame, numeric_cols: list[str], category_cols: list[str]):
    plt.figure(figsize=(20,20))
    # distribution plots for numeric cols
    for i, col in enumerate(numeric_cols,1):
        plt.subplot(4, 4, i)
        sns.histplot(data=df,x=col,kde=True,color='indigo',alpha=0.6)
        plt.title(f'Distribution of {col.title()}',fontsize=12, fontweight='bold')
        plt.ylabel('Frequency',fontsize=13)
        plt.grid(True, alpha=0.4)

    # boxplots
    for i, col in enumerate(numeric_cols,6):
        plt.subplot(4, 4, i)
        sns.boxplot(data=df, y=col, color='red')
        plt.title(f'Boxplot - {col}',fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.4)

    plt.subplot(4, 4, 11)
    corr = df.corr(numeric_only=True)
    sns.heatmap(data=corr, annot=True, fmt='.2f')
    plt.title('Correlation Heatmap',fontsize=12,fontweight='bold')

    for i,col in enumerate(category_cols,12):
        plt.subplot(4, 4, i)
        ax = sns.countplot(data=df, x=col, gap=0.5, width=0.4, color='green')
        for container in ax.containers:
            ax.bar_label(container,label_type='edge')
        plt.title(f'Countplot - {col}', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency',fontsize=12)
    plt.savefig('plots/advanced_visuals',bbox_inches='tight',dpi=300)
    plt.tight_layout()
    plt.show()

