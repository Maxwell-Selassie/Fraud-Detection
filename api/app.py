# api/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import logging
import os
import traceback

from src.batch_inference import run_batch_inference

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Setup logging
logging.basicConfig(
    filename='logs/api.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger('Fraud_API')

# Initialize FastAPI
app = FastAPI(
    title="Fraud Detection API",
    description="Batch + Real-time fraud detection API using IsolationForest + Semi-supervised model",
    version="1.0.0"
)


# Response schema
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

    try:
        # Save uploaded file temporarily
        temp_path = os.path.join("data", file.filename)
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Run batch inference
        df_preds = run_batch_inference(temp_path)

        # Ensure expected columns exist
        required_cols = {"AccountID", "fraud_probability", "final_predictions"}
        missing_cols = required_cols - set(df_preds.columns)
        if missing_cols:
            log.error(f"Missing columns in prediction output: {missing_cols}")
            raise HTTPException(status_code=500, detail=f"Missing columns: {missing_cols}")

        # Convert to response format
        results = df_preds[list(required_cols)].to_dict(orient="records")

        log.info(f"Batch inference complete: {len(results)} records processed.")
        return results

    except Exception as e:
        log.error(f"Error during batch prediction: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Batch prediction failed.")
