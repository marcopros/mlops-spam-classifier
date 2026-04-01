from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text message to classify")


class PredictResponse(BaseModel):
    prediction: str
    spam_probability: float
