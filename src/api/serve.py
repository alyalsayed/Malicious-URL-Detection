"""
FastAPI service for Malicious URL Detection.

Defines:
- Startup loading of inference dependencies
- Prediction endpoint using pydantic models
"""

import logging
from fastapi import FastAPI
import uvicorn

from src.api.schemas import URLRequest, URLPrediction
from src.inference.predict import predict_url
from src.config import API_HOST, API_PORT, API_RELOAD, LOG_LEVEL


# --------------------------------------------------------
# Logger
# --------------------------------------------------------
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("API")


# --------------------------------------------------------
# FastAPI App
# --------------------------------------------------------
app = FastAPI(
    title="🚨 Malicious URL Detection API",
    description="Detects malicious URLs using machine learning.",
    version="1.0.0",
)


# --------------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------------
@app.post("/predict", response_model=URLPrediction)
def predict_endpoint(request: URLRequest):
    """
    API endpoint that receives a URL and returns model prediction.
    """
    logger.info(f"Received URL for prediction: {request.url}")
    result = predict_url(request.url)
    return URLPrediction(**result)


# --------------------------------------------------------
# Run API server
# --------------------------------------------------------
if __name__ == "__main__":
    logger.info(f"Starting API server on http://{API_HOST}:{API_PORT}")
    uvicorn.run(
        "src.api.serve:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD
    )
