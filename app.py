# api/main.py

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import pandas as pd
import logging
import os

from src.batch_inference import run_batch_inference

# Ensure logs directory
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    filename='logs/api.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger('Fraud_API')

app = FastAPI(
    title="Fraud Detection API",
    description="Batch + Real-time fraud detection API using IsolationForest + Semi-supervised model",
    version="1.0.0"
)


class PredictionResponse(BaseModel):
    AccountID: str
    fraud_probability: float
    final_predictions: int


@app.get("/health")
def health_check():
    """Check API health status."""
    return {"status": "online", "message": "Fraud detection API is running."}


@app.post("/predict_batch", response_model=List[PredictionResponse])
async def predict_batch(file: UploadFile = File(...)):
    """
    Accepts a CSV file of transactions, runs batch inference, and returns fraud predictions.
    """
    log.info(f"Received batch file: {file.filename}")

    # Save uploaded file temporarily
    temp_path = f"data/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Run inference pipeline
    df_preds = run_batch_inference(temp_path)
    results = df_preds.to_dict(orient="records")

    log.info(f"Batch inference complete: {len(results)} records processed.")
    return results
