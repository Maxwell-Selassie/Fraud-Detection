import pandas as pd
import numpy as np
import joblib
import logging
import os
from datetime import datetime
from feature_engineering import feature_engineer, feature_encoding

# setup logging
logging.basicConfig(
    filename='logs/inference.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger('Batch_Inference')

