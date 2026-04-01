import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


class ModelService:
    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self.model = None
        self.load_model_if_available()

    def load_model_if_available(self) -> None:
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)

    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, text: str) -> dict[str, Any]:
        prediction_num = int(self.model.predict([text])[0])
        proba = float(self.model.predict_proba([text])[0][1])
        prediction_label = "spam" if prediction_num == 1 else "ham"
        return {"prediction": prediction_label, "spam_probability": proba}

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        predictions = self.model.predict(texts)
        probas = self.model.predict_proba(texts)[:, 1]
        results = []
        for pred, prob in zip(predictions, probas, strict=True):
            label = "spam" if int(pred) == 1 else "ham"
            results.append({"prediction": label, "spam_probability": float(prob)})
        return results


def log_prediction(text: str, result: dict[str, Any], logs_path: str) -> None:
    log_file = Path(logs_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "prediction": result["prediction"],
        "spam_probability": result["spam_probability"],
    }

    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
