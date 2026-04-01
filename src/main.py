import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.model_utils import ModelService, log_prediction
from src.schemas import PredictRequest, PredictResponse

MODEL_PATH = os.getenv("MODEL_PATH", "models/model_latest.joblib")
PREDICTIONS_LOG_PATH = os.getenv("PREDICTIONS_LOG_PATH", "logs/predictions.jsonl")

app = FastAPI(title="Spam Classification API", version="1.0.0")
model_service = ModelService(model_path=MODEL_PATH)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health() -> dict:
    return {
        "message": "ML API running",
        "model_loaded": model_service.is_ready(),
        "model_path": MODEL_PATH,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if not model_service.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    result = model_service.predict(request.text)
    log_prediction(request.text, result, PREDICTIONS_LOG_PATH)
    return PredictResponse(**result)
