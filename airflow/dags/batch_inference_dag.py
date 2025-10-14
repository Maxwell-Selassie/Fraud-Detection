from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
from src.batch_inference import run_batch_inference

log = logging.getLogger(__name__)

def batch_inference_task():
    filepath = 'data/bank_transactions_data_2.csv'
    run_batch_inference(filepath)
    log.info("Batch inference completed successfully.")

default_args = {
    'owner': 'maxwell',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['mlops-monitor@ltministry.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'fraud_detection_batch_inference',
    default_args=default_args,
    description='Automated batch inference for fraud detection',
    schedule_interval='@daily',  # daily run
    start_date=datetime(2025, 10, 11),
    catchup=False,
) as dag:
    
    run_batch = PythonOperator(
        task_id='run_batch_inference',
        python_callable=batch_inference_task,
    )

    run_batch
