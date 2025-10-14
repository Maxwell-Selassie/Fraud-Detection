from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.train_anomaly import train_models

default_args = {
    'owner': 'maxwell',
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

with DAG(
    'fraud_detection_training',
    default_args=default_args,
    description='Retrain fraud models and log to MLflow',
    schedule_interval='@weekly',
    start_date=datetime(2025, 10, 11),
    catchup=False,
) as dag:
    
    retrain = PythonOperator(
        task_id='retrain_fraud_models',
        python_callable=train_models,
    )

    retrain
